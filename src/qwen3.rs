//! Qwen3-архитектура на nn-слоях (Ф1.7, веха M1).
//!
//! Сборка из [`WeightsMap`] (GGUF-тензоры в f32 через dequant-референс),
//! forward на `&dyn Backend`: prefill (батч) и decode (по токену, KV-кэш).
//!
//! RoPE: Qwen3 использует СТАНДАРТНЫЙ rotate-half (HF). `nn::Attention`
//! внутри применяет interleaved-RoPE — поэтому q/k предварительно
//! поворачиваются стандартным RoPE (CPU-хелпер, корректность-first;
//! GPU-кернел — в перф-фазе), а в Attention передаются identity-таблицы
//! (cos=1, sin=0 — no-op).
//!
//! Имена тензоров — конвенция llama.cpp для Qwen3:
//! token_embd.weight, blk.N.attn_norm/attn_q/attn_k/attn_v/attn_output/
//! attn_q_norm/attn_k_norm/ffn_norm/ffn_gate/ffn_up/ffn_down.weight,
//! output_norm.weight, output.weight (tied с token_embd — дедупликация
//! WeightsMap).

use crate::backend::{Backend, Tensor};
use crate::error::BackendError;
use crate::gguf::{GgufFile, GgufValue};
use crate::nn::{Attention, KvCache, Linear, RmsNorm, SwiGLU};
use crate::weights_map::WeightsMap;

/// Конфигурация Qwen3 из GGUF-метадаты (`qwen3.*`).
#[derive(Debug, Clone)]
pub struct Qwen3Config {
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
    pub rope_max_seq: usize,
    pub norm_eps: f32,
}

impl Qwen3Config {
    pub fn from_gguf(file: &GgufFile<'_>, vocab_size: usize) -> Result<Self, BackendError> {
        let u = |k: &str| -> Result<usize, BackendError> {
            file.get(k)
                .and_then(GgufValue::as_u32)
                .map(|v| v as usize)
                .ok_or_else(|| BackendError::Shape(format!("missing {k}")))
        };
        let dim = u("qwen3.embedding_length")?;
        let n_heads = u("qwen3.attention.head_count")?;
        let n_kv_heads = u("qwen3.attention.head_count_kv")?;
        let ffn_dim = u("qwen3.feed_forward_length")?;
        let n_layers = u("qwen3.block_count")?;
        // head_dim: llama.cpp хранит его отдельно (qwen3.attention.key_length);
        // fallback — dim/n_heads (classic).
        let head_dim = u("qwen3.attention.key_length")
            .unwrap_or(dim / n_heads);
        let theta = file
            .get("qwen3.rope.freq_base")
            .or_else(|| file.get("qwen3.attention.rope_theta"))
            .and_then(|v| match v {
                GgufValue::F32(f) => Some(*f as f64),
                GgufValue::F64(f) => Some(*f),
                _ => None,
            })
            .unwrap_or(1_000_000.0) as f32;
        let eps = file
            .get("qwen3.attention.layer_norm_rms_epsilon")
            .and_then(|v| match v {
                GgufValue::F32(f) => Some(*f),
                GgufValue::F64(f) => Some(*f as f32),
                _ => None,
            })
            .unwrap_or(1e-6);
        Ok(Self {
            dim,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            ffn_dim,
            vocab_size,
            rope_theta: theta,
            rope_max_seq: 8192,
            norm_eps: eps,
        })
    }
}

struct Qwen3Layer {
    attn_norm: RmsNorm,
    attn: Attention,
    ffn_norm: RmsNorm,
    ffn: SwiGLU,
}

/// Qwen3-модель (decode-состояние: KV-кэши в слоях).
pub struct Qwen3Model {
    pub cfg: Qwen3Config,
    embed: crate::nn::Embedding,
    layers: Vec<Qwen3Layer>,
    out_norm: RmsNorm,
    /// Выходная проекция [V, D]; при tied-весах — тот же стораж, что у embed.
    output: Tensor,
    caches: Vec<KvCache>,
}

impl Qwen3Model {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        backend: &dyn Backend,
        w: &WeightsMap<'_>,
        cfg: Qwen3Config,
    ) -> Result<Self, BackendError> {
        let t = |name: &str| -> Result<Vec<f32>, BackendError> {
            let tensor = w
                .get(name)
                .ok_or_else(|| BackendError::Shape(format!("missing tensor {name}")))?;
            Ok(crate::weights_map::dequant_f32(tensor))
        };
        let embed_w = t("token_embd.weight")?;
        let embed = crate::nn::Embedding::from_weights(
            backend,
            embed_w.clone(),
            vec![cfg.vocab_size, cfg.dim],
        )?;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        let mut caches = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = |s: &str| format!("blk.{i}.{s}");
            // fuse q/k/v → [3D, D]
            let q = t(&p("attn_q.weight"))?;
            let k = t(&p("attn_k.weight"))?;
            let v = t(&p("attn_v.weight"))?;
            let mut wqkv = q;
            wqkv.extend_from_slice(&k);
            wqkv.extend_from_slice(&v);
            let wo = t(&p("attn_output.weight"))?;
            let qn = t(&p("attn_q_norm.weight"))?;
            let kn = t(&p("attn_k_norm.weight"))?;
            let attn = Attention {
                wqkv: upload_w(
                    backend,
                    &wqkv,
                    vec![
                        cfg.n_heads * cfg.head_dim + 2 * cfg.n_kv_heads * cfg.head_dim,
                        cfg.dim,
                    ],
                )?,
                wo: upload_w(
                    backend,
                    &wo,
                    vec![cfg.dim, cfg.n_heads * cfg.head_dim],
                )?,
                q_norm: Some(backend.from_cpu_f32(&qn, vec![cfg.head_dim])?),
                k_norm: Some(backend.from_cpu_f32(&kn, vec![cfg.head_dim])?),
                n_heads: cfg.n_heads,
                n_kv_heads: cfg.n_kv_heads,
                head_dim: cfg.head_dim,
            };
            let attn_norm =
                RmsNorm::from_weights(backend, t(&p("attn_norm.weight"))?, cfg.norm_eps)?;
            let ffn_norm = RmsNorm::from_weights(backend, t(&p("ffn_norm.weight"))?, cfg.norm_eps)?;
            // ffn: gate/up [ffn, D], down [D, ffn] — GEMM-веса в f16 (Metal)
            let gate = t(&p("ffn_gate.weight"))?;
            let up = t(&p("ffn_up.weight"))?;
            let down = t(&p("ffn_down.weight"))?;
            let ffn = SwiGLU {
                w1: upload_w(backend, &gate, vec![cfg.ffn_dim, cfg.dim])?,
                w3: upload_w(backend, &up, vec![cfg.ffn_dim, cfg.dim])?,
                w2: upload_w(backend, &down, vec![cfg.dim, cfg.ffn_dim])?,
            };
            layers.push(Qwen3Layer {
                attn_norm,
                attn,
                ffn_norm,
                ffn,
            });
            caches.push(KvCache::new(
                backend,
                cfg.n_kv_heads,
                cfg.head_dim,
                crate::dtype::KvFormat::F32,
                crate::dtype::KvFormat::F32,
            )?);
        }

        let out_norm = RmsNorm::from_weights(backend, t("output_norm.weight")?, cfg.norm_eps)?;
        // tied: output.weight может делить стораж с token_embd
        let output = match w.get("output.weight") {
            Some(_) if !w.tied_aliases("output.weight").is_empty() => {
                upload_w(backend, &embed_w, vec![cfg.vocab_size, cfg.dim])?
            }
            Some(_) => upload_w(backend, &t("output.weight")?, vec![cfg.vocab_size, cfg.dim])?,
            None => upload_w(backend, &embed_w, vec![cfg.vocab_size, cfg.dim])?,
        };

        Ok(Self {
            cfg,
            embed,
            layers,
            out_norm,
            output,
            caches,
        })
    }

    /// Prefill: ids [T] → logits [1, T, V]; KV-кэши пополняются.
    pub fn prefill(&mut self, backend: &dyn Backend, ids: &[u32]) -> Result<Tensor, BackendError> {
        let t = ids.len();
        let mut x = self.embed.forward(backend, ids, t)?;
        let pos = self.caches[0].pos;
        for (i, layer) in self.layers.iter().enumerate() {
            let h = layer.attn_norm.forward(backend, &x)?;
            let h = Self::rope_and_attn(
                &self.cfg,
                backend,
                &h,
                &layer.attn,
                &mut self.caches[i],
                pos,
                t,
            )?;
            x = backend_add(backend, &x, &h)?;
            let h = layer.ffn_norm.forward(backend, &x)?;
            let f = ffn_forward(backend, &h, &layer.ffn)?;
            x = backend_add(backend, &x, &f)?;
        }
        let h = self.out_norm.forward(backend, &x)?;
        let (b, tq, d) = (h.shape()[0], h.shape()[1], h.shape()[2]);
        let h2 = h.view(vec![b * tq, d]);
        let logits = backend.zeros(vec![b * tq, self.cfg.vocab_size])?;
        backend.gemm_nt(&h2, &self.output, &logits)?;
        Ok(logits.view(vec![b, tq, self.cfg.vocab_size]))
    }

    /// Decode: один токен → logits [1, 1, V]; KV-кэш растёт.
    pub fn decode(&mut self, backend: &dyn Backend, id: u32) -> Result<Tensor, BackendError> {
        self.prefill(backend, &[id])
    }

    /// Сброс KV-кэшей (новый контекст).
    pub fn reset_cache(&mut self, backend: &dyn Backend) -> Result<(), BackendError> {
        for c in &mut self.caches {
            c.reset();
            c.shrink_to_init(backend)?;
        }
        Ok(())
    }

    /// Стандартный rotate-half RoPE (HF) на CPU-хелпере + Attention с
    /// identity-таблицами.
    fn rope_and_attn(
        cfg: &Qwen3Config,
        backend: &dyn Backend,
        h: &Tensor,
        attn: &Attention,
        cache: &mut KvCache,
        pos: usize,
        t: usize,
    ) -> Result<Tensor, BackendError> {
        // q/k из h через Attention — но Attention применяет interleaved-RoPE
        // сам. Делаем pre-rotation стандартным RoPE и передаём identity-таблицы:
        // для этого раскладываем h на q/k/v, поворачиваем q/k, собираем обратно
        // и отдаём Attention'у с no-op rope.
        let (b, tq, _d) = (h.shape()[0], h.shape()[1], h.shape()[2]);
        let q_size = attn.n_heads * attn.head_dim;
        let kv_size = attn.n_kv_heads * attn.head_dim;
        let qkv = Linear {
            w: attn.wqkv.clone(),
            b: None,
        }
        .forward(backend, h)?;
        let q = crate::nn::slice_last(backend, &qkv, q_size, 0)?;
        let k = crate::nn::slice_last(backend, &qkv, kv_size, q_size)?;
        let v = crate::nn::slice_last(backend, &qkv, kv_size, q_size + kv_size)?;
        let (q, k) = if attn.q_norm.is_some() && attn.k_norm.is_some() {
            (
                qk_norm(backend, &q, attn.q_norm.as_ref().unwrap(), attn.n_heads)?,
                qk_norm(backend, &k, attn.k_norm.as_ref().unwrap(), attn.n_kv_heads)?,
            )
        } else {
            (q, k)
        };
        let q = rope_standard(backend, &q, pos, t, attn.head_dim, cfg.rope_theta)?;
        let k = rope_standard(backend, &k, pos, t, attn.head_dim, cfg.rope_theta)?;
        // attention напрямую через attn_ar (q/k уже повёрнуты стандартным RoPE)
        // q/k/v 3D [1, T, D]; attn_ar сам разложит по головам
        cache.update(backend, &k, &v)?;
        let mask = if t > 1 {
            Some(crate::nn::build_causal_mask(backend, t, cache.pos, pos)?)
        } else {
            None
        };
        let ck = cache.view();
        let cv = cache.view_v();
        let attn_out = backend.zeros(vec![b, tq, q_size])?;
        backend.attn_ar(
            &q,
            &ck,
            &cv,
            mask.as_ref(),
            &attn_out,
            attn.n_heads,
            attn.n_kv_heads,
            attn.head_dim,
        )?;
        let out = Linear {
            w: attn.wo.clone(),
            b: None,
        }
        .forward(backend, &attn_out)?;
        Ok(out)
    }
}

/// QK-norm: per-head RMSNorm (w [dim], применяется по hd каждой головы).
fn qk_norm(
    backend: &dyn Backend,
    x: &Tensor,
    w: &Tensor,
    heads: usize,
) -> Result<Tensor, BackendError> {
    let y = backend.zeros(x.shape().to_vec())?;
    backend.rmsnorm_head(x, w, &y, heads, 1e-6)?;
    Ok(y)
}

/// Загрузка GEMM-весов: f16 на бэкендах с поддержкой (Metal — вдвое меньше
/// VRAM), f32-фоллбэк (CPU-референс).
fn upload_w(
    backend: &dyn Backend,
    data: &[f32],
    shape: Vec<usize>,
) -> Result<Tensor, BackendError> {
    if backend.supports_f16_weights() {
        let n: usize = shape.iter().product();
        if n != data.len() {
            eprintln!("DBG upload_w MISMATCH shape={shape:?} data={}", data.len());
        }
        let f16: Vec<half::f16> = data.iter().map(|&x| half::f16::from_f32(x)).collect();
        backend.from_cpu_f16(&f16, shape)
    } else {
        backend.from_cpu_f32(data, shape)
    }
}

/// Стандартный rotate-half RoPE (HF/Qwen3) через backend-оп
/// (Metal — кернел rope_std_f32; CPU — референс-вычисление).
fn rope_standard(
    backend: &dyn Backend,
    x: &Tensor,
    pos: usize,
    t: usize,
    hd: usize,
    theta: f32,
) -> Result<Tensor, BackendError> {
    let _ = t;
    backend.rope_standard(x, pos, hd, theta)
}

fn ffn_forward(backend: &dyn Backend, x: &Tensor, ffn: &SwiGLU) -> Result<Tensor, BackendError> {
    // Qwen3 FFN: ffn_dim ≠ dim — считаем вручную через поля SwiGLU.
    let (b, t, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    let x2 = x.view(vec![b * t, d]);
    let ffn_dim = ffn.w1.shape()[0];
    let a = backend.zeros(vec![b * t, ffn_dim])?;
    let g = backend.zeros(vec![b * t, ffn_dim])?;
    let gm = backend.zeros(vec![b * t, ffn_dim])?;
    backend.gemm_nt(&x2, &ffn.w1, &a)?;
    backend.gemm_nt(&x2, &ffn.w3, &g)?;
    backend.silu(&a, &a)?;
    backend.mul(&a, &g, &gm)?;
    let y = backend.zeros(vec![b * t, d])?;
    backend.gemm_nt(&gm, &ffn.w2, &y)?;
    Ok(y.view(vec![b, t, d]))
}

fn backend_add(backend: &dyn Backend, a: &Tensor, b: &Tensor) -> Result<Tensor, BackendError> {
    let y = backend.zeros(a.shape().to_vec())?;
    backend.add(a, b, &y)?;
    Ok(y)
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use crate::metal::MetalBackend;
    use crate::weights_map::{WeightTensor, WeightsMap};

    fn rng_u64(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        }
    }

    fn rand_vec(n: usize, rng: &mut impl FnMut() -> u64) -> Vec<f32> {
        (0..n)
            .map(|_| ((rng() >> 32) as f32 / (1u64 << 32) as f32) * 2.0 - 1.0)
            .collect()
    }

    fn cfg() -> Qwen3Config {
        Qwen3Config {
            dim: 16,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 32,
            vocab_size: 8,
            rope_theta: 10_000.0,
            rope_max_seq: 128,
            norm_eps: 1e-6,
        }
    }

    /// Мини-Qwen3: случайные веса; WeightsMap строится поверх owned-буферов.
    fn mini_model() -> (MetalBackend, Qwen3Model) {
        let backend = MetalBackend::new().expect("metal backend");
        let c = cfg();
        let mut rng = rng_u64(0xC0FFEE);
        let mut storage: Vec<(String, Vec<f32>, Vec<usize>)> = Vec::new();
        storage.push((
            "token_embd.weight".into(),
            rand_vec(c.vocab_size * c.dim, &mut rng),
            vec![c.vocab_size, c.dim],
        ));
        for i in 0..c.n_layers {
            let p = |s: &str| format!("blk.{i}.{s}");
            for (n, shape) in [
                ("attn_norm.weight", vec![c.dim]),
                ("attn_q.weight", vec![c.dim, c.dim]),
                // GQA: k/v высотой n_kv_heads*head_dim
                ("attn_k.weight", vec![c.n_kv_heads * c.head_dim, c.dim]),
                ("attn_v.weight", vec![c.n_kv_heads * c.head_dim, c.dim]),
                ("attn_output.weight", vec![c.dim, c.dim]),
                ("attn_q_norm.weight", vec![c.head_dim]),
                ("attn_k_norm.weight", vec![c.head_dim]),
                ("ffn_norm.weight", vec![c.dim]),
                ("ffn_gate.weight", vec![c.ffn_dim, c.dim]),
                ("ffn_up.weight", vec![c.ffn_dim, c.dim]),
                ("ffn_down.weight", vec![c.dim, c.ffn_dim]),
            ] {
                let numel = shape.iter().product::<usize>();
                storage.push((p(n), rand_vec(numel, &mut rng), shape));
            }
        }
        storage.push((
            "output_norm.weight".into(),
            rand_vec(c.dim, &mut rng),
            vec![c.dim],
        ));

        // f32-байты живут в storage до конца теста; WeightsMap ссылается на них
        let raw: Vec<(String, Vec<u8>, Vec<u64>)> = storage
            .iter()
            .map(|(n, data, shape)| {
                let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
                (n.clone(), bytes, shape.iter().map(|&d| d as u64).collect())
            })
            .collect();
        let tensors: Vec<WeightTensor> = raw
            .iter()
            .map(|(name, bytes, shape)| WeightTensor {
                name: name.clone(),
                shape: shape.clone(),
                format: crate::dtype::WeightFormat::Plain(crate::dtype::Dtype::F32),
                offset: 0,
                data: bytes,
            })
            .collect();
        let wm = WeightsMap {
            arch: "qwen3".into(),
            tensors,
            storage: vec![],
        };
        let model = Qwen3Model::load(&backend, &wm, c).expect("load mini qwen3");
        (backend, model)
    }

    /// Ключевой инвариант decode: лог-проблема декод-шага на позиции k
    /// совпадает с prefill-логитами на позиции k (при том же состоянии KV).
    #[test]
    fn decode_matches_prefill_last_logits() {
        let (backend, mut model) = mini_model();
        let ids = [1u32, 2, 3, 4];
        let pre = model.prefill(&backend, &ids).unwrap();
        let pre_v: Vec<f32> = backend.to_cpu_f32(&pre).unwrap(); // [1,4,8]
        model.reset_cache(&backend).unwrap();
        for (k, &id) in ids.iter().enumerate() {
            let logits = model.decode(&backend, id).unwrap();
            let v: Vec<f32> = backend.to_cpu_f32(&logits).unwrap(); // [1,1,8]
            let want = &pre_v[k * 8..k * 8 + 8];
            for (got, want) in v.iter().zip(want) {
                assert!((got - want).abs() < 1e-3, "pos {k}: got {got} want {want}");
            }
        }
    }

    #[test]
    fn prefill_positions_differ_and_kv_grows() {
        let (backend, mut model) = mini_model();
        let ids = [1u32, 2, 3];
        let pre = model.prefill(&backend, &ids).unwrap();
        let v: Vec<f32> = backend.to_cpu_f32(&pre).unwrap();
        // разные позиции дают разные логиты
        assert!(
            v[0..8]
                .iter()
                .zip(&v[8..16])
                .any(|(a, b)| (a - b).abs() > 1e-4)
        );
        // KV-кэш продвинулся
        assert_eq!(model.caches[0].pos, 3);
        model.reset_cache(&backend).unwrap();
        assert_eq!(model.caches[0].pos, 0);
    }

    #[test]
    fn generate_greedy_is_deterministic() {
        let (backend, mut model) = mini_model();
        let mut seq = vec![1u32, 2];
        for _ in 0..3 {
            let logits = model.decode(&backend, *seq.last().unwrap()).unwrap();
            let v: Vec<f32> = backend.to_cpu_f32(&logits).unwrap();
            let next = v
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap();
            seq.push(next);
        }
        assert_eq!(seq.len(), 5);
        // детерминизм: повторный прогон даёт ту же последовательность
        let (backend2, mut model2) = mini_model();
        let _ = backend2;
        let mut seq2 = vec![1u32, 2];
        for _ in 0..3 {
            let logits = model2.decode(&backend, *seq2.last().unwrap()).unwrap();
            let v: Vec<f32> = backend.to_cpu_f32(&logits).unwrap();
            let next = v
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap();
            seq2.push(next);
        }
        assert_eq!(seq, seq2);
    }
}
