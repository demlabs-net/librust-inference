//! DFlash-драфтер (Ф2.3/Ф2.4/Ф2.5) — по спеке llama.cpp PR #27342
//! (см. БЗ SLC: dflash-speka-llama-pr-27342).
//!
//! Структура: dense-слои (Qwen3-архитектура, эмбеддинги ОБЩИЕ с таргетом!)
//! + two-tap dynamic conv (attn/ffn_conv_base + conv_proj) + селектор (v2:
//! predecessor/successor codebooks, top-k лэттис). Два режима:
//! - инжект (embd): фичи таргета → K/V-проекции каждого слоя → кэш;
//! - диффузия (token): [committed, MASK...] → dense+conv → логиты/скрытые.

use crate::backend::{Backend, Tensor};
use crate::error::BackendError;
use crate::nn::{Attention, KvCache, RmsNorm, SwiGLU};
use crate::qwen3::{Qwen3Config, Qwen3Layer};
use crate::dtype::WeightFormat;
use crate::weights_map::{dequant_f32, WeightsMap};

/// Conv-модуль DFlash-слоя (two-tap dynamic, attn или ffn).
#[derive(Debug, Clone)]
pub struct DFlashConv {
    /// [n_embd, kernel, 2] — статическая база (side: 0 = вход, 1 = выход).
    pub base: Tensor,
    /// [n_groups·kernel·2, n_embd] — проекция шума в динамические коэффы.
    pub proj: Option<Tensor>,
}

/// Один слой драфтера: dense (Qwen3) + два conv-модуля.
#[derive(Debug, Clone)]
pub struct DFlashLayer {
    pub dense: Qwen3Layer,
    pub attn_conv: DFlashConv,
    pub ffn_conv: DFlashConv,
}

/// Селектор DFlash2: codebook-эмбеддинги предшественника/преемника.
#[derive(Debug, Clone)]
pub struct Selector {
    pub predecessor: Tensor, // [n_embd, rank·top_k] (codebook)
    pub successor: Tensor,   // [n_embd, rank·top_k]
    pub rank: usize,
    pub top_k: usize,
}

/// DFlash-драфтер.
pub struct DFlashModel {
    pub cfg: Qwen3Config,
    /// Эмбеддинги — ОБЩИЕ с таргетом (драфтер их не имеет — берутся у таргета).
    pub target_embed: Option<Tensor>,
    pub layers: Vec<DFlashLayer>,
    /// Финальная нормализация драфтера (result_norm; своя, не таргета).
    pub out_norm: RmsNorm,
    pub selector: Option<Selector>,
    pub caches: Vec<KvCache>,
    pub block_size: usize,
    pub conv_kernel: usize,
    pub conv_group: usize,
    pub extract_layers: Vec<usize>,
}

impl DFlashModel {
    /// Загрузка из GGUF-драфтера (conv/selector-тензоры + dense-слои).
    pub fn load(
        backend: &dyn Backend,
        w: &WeightsMap<'_>,
        cfg: Qwen3Config,
        block_size: usize,
        conv_kernel: usize,
        conv_group: usize,
        extract_layers: Vec<usize>,
    ) -> Result<Self, BackendError> {
        let up_q = |name: &str, shape: Vec<usize>| -> Result<Tensor, BackendError> {
            let tensor = w
                .get(name)
                .ok_or_else(|| BackendError::Shape(format!("missing tensor {name}")))?;
            match tensor.format {
                WeightFormat::Q4_0
                | WeightFormat::Q8_0
                | WeightFormat::Q4_K
                | WeightFormat::Q6_K => backend.from_quantized(tensor.raw(), shape, tensor.format),
                _ => {
                    let f = dequant_f32(tensor);
                    upload_f16(backend, &f, shape)
                }
            }
        };
        let up = |name: &str, shape: Vec<usize>| -> Result<Tensor, BackendError> {
            let f = dequant_f32(w.get(name).ok_or_else(|| {
                BackendError::Shape(format!("missing tensor {name}"))
            })?);
            upload_f16(backend, &f, shape)
        };
        let n_groups = cfg.dim / conv_group;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        let mut caches = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = |s: &str| format!("blk.{i}.{s}");
            let dense = load_dense_layer(backend, w, &p, &cfg)?;
            let attn_conv = DFlashConv {
                base: up_q(&p("attn_conv_base"), vec![cfg.dim, conv_kernel, 2])?,
                proj: if w.get(&p("attn_conv_proj")).is_some() {
                    Some(up_q(
                        &p("attn_conv_proj"),
                        vec![n_groups * conv_kernel * 2, cfg.dim],
                    )?)
                } else {
                    None
                },
            };
            let ffn_conv = DFlashConv {
                base: up_q(&p("ffn_conv_base"), vec![cfg.dim, conv_kernel, 2])?,
                proj: if w.get(&p("ffn_conv_proj")).is_some() {
                    Some(up_q(
                        &p("ffn_conv_proj"),
                        vec![n_groups * conv_kernel * 2, cfg.dim],
                    )?)
                } else {
                    None
                },
            };
            layers.push(DFlashLayer {
                dense,
                attn_conv,
                ffn_conv,
            });
            caches.push(KvCache::new(
                backend,
                cfg.n_kv_heads,
                cfg.head_dim,
                crate::dtype::KvFormat::F32,
                crate::dtype::KvFormat::F32,
            )?);
        }
        let selector = if w.get("selector_predecessor").is_some() {
            // rank/top_k — из метаданных драфтера; задел: параметры
            let (rank, top_k) = (4usize, 10usize);
            Some(Selector {
                predecessor: up(
                    "selector_predecessor",
                    vec![cfg.dim, rank * top_k],
                )?,
                successor: up("selector_successor", vec![cfg.dim, rank * top_k])?,
                rank,
                top_k,
            })
        } else {
            None
        };
        let _ = up;
        let out_norm = RmsNorm::from_weights(
            backend,
            dequant_f32(
                w.get("output_norm.weight")
                    .ok_or_else(|| BackendError::Shape("missing tensor output_norm.weight".into()))?,
            ),
            cfg.norm_eps,
        )?;
        Ok(Self {
            cfg,
            target_embed: None,
            layers,
            out_norm,
            selector,
            caches,
            block_size,
            conv_kernel,
            conv_group,
            extract_layers,
        })
    }

    /// Привязать общие эмбеддинги таргета (драфтер их не хранит).
    pub fn set_target_embed(&mut self, t: Tensor) {
        self.target_embed = Some(t);
    }

    /// Инжект фич таргета: для каждого слоя K/V = wv/wk(features) → кэш.
    /// features: [n_tokens, n_embd] (уже interleaved-конкатенация extract-слоёв,
    /// спроецированная в n_embd драфтера).
    pub fn inject_features(
        &mut self,
        backend: &dyn Backend,
        features: &Tensor,
        pos: usize,
    ) -> Result<(), BackendError> {
        backend.begin_batch()?;
        let r = (|| {
            let t = features.shape()[0];
            for (i, layer) in self.layers.iter().enumerate() {
                let kv_size = self.cfg.n_kv_heads * self.cfg.head_dim;
                let q_size = self.cfg.n_heads * self.cfg.head_dim;
                let qkv = crate::nn::Linear {
                    w: layer.dense.attn.wqkv.clone(),
                    b: None,
                }
                .forward(backend, features)?;
                let k = crate::nn::slice_last(backend, &qkv, kv_size, q_size)?;
                let v = crate::nn::slice_last(backend, &qkv, kv_size, q_size + kv_size)?;
                let k = crate::qwen3::rope_standard(backend, &k, pos, t, self.cfg.head_dim, self.cfg.rope_theta)?;
                self.caches[i].update(backend, &k, &v)?;
            }
            Ok(())
        })();
        backend.end_batch()?;
        r
    }

    /// Диффузия token-режима (dflash.cpp build_graph): mask-патч `ids`
    /// ([committed..., MASK...]) → dense+conv слои. Порядок слоя:
    /// attn_norm → conv_in(noise) → qkv/rope/attn (non-causal, по кэшу) →
    /// conv_out → +inpL → ffn_norm → conv_in → swiglu → conv_out → +ffn_inp.
    /// Логиты — через lm_head ТАРГЕТА (общий; спека: "lm_head from the target
    /// model"), скрытые — свой out_norm (t_embd, для селектора v2).
    /// ВНИМАНИЕ: мутирует кэши (пишет k/v draft-токенов) — после шага
    /// вызывающий должен откатить кэши до инжектной длины.
    pub fn diffuse(
        &mut self,
        backend: &dyn Backend,
        ids: &[u32],
        lm_head: &Tensor,
    ) -> Result<DiffusionResult, BackendError> {
        backend.begin_batch()?;
        let r = (|| {
            let t = ids.len();
            let embed = self
                .target_embed
                .as_ref()
                .ok_or_else(|| {
                    BackendError::Shape("diffuse: не установлен target_embed — драфтер без общих эмбеддингов".into())
                })?
                .clone();
            let mut x = backend.zeros(vec![1, t, self.cfg.dim])?;
            backend.embedding(ids, &embed, &mut x)?;
            for (i, layer) in self.layers.iter().enumerate() {
                // ── attn-часть ──
                let mut h = layer.dense.attn_norm.forward(backend, &x)?;
                let attn_dyn = match &layer.attn_conv.proj {
                    Some(p) => Some(conv_dynamic(backend, p, &h)?),
                    None => None,
                };
                if let Some(d) = &attn_dyn {
                    h = conv_side(backend, &h, &layer.attn_conv.base, d, 0, self.block_size)?;
                }
                let pos = self.caches[i].pos;
                let mut cur = crate::qwen3::Qwen3Model::rope_and_attn(
                    &self.cfg,
                    backend,
                    &h,
                    &layer.dense.attn,
                    &mut self.caches[i],
                    pos,
                    t,
                    None,
                    false,
                )?;
                if let Some(d) = &attn_dyn {
                    cur = conv_side(backend, &cur, &layer.attn_conv.base, d, 1, self.block_size)?;
                }
                let ffn_inp = add(backend, &cur, &x)?;
                // ── ffn-часть ──
                let mut h = layer.dense.ffn_norm.forward(backend, &ffn_inp)?;
                let ffn_dyn = match &layer.ffn_conv.proj {
                    Some(p) => Some(conv_dynamic(backend, p, &h)?),
                    None => None,
                };
                if let Some(d) = &ffn_dyn {
                    h = conv_side(backend, &h, &layer.ffn_conv.base, d, 0, self.block_size)?;
                }
                let mut f = crate::qwen3::ffn_forward(backend, &h, &layer.dense.ffn)?;
                if let Some(d) = &ffn_dyn {
                    f = conv_side(backend, &f, &layer.ffn_conv.base, d, 1, self.block_size)?;
                }
                x = add(backend, &f, &ffn_inp)?;
            }
            let hidden = self.out_norm.forward(backend, &x)?;
            let logits = crate::nn::Linear {
                w: lm_head.clone(),
                b: None,
            }
            .forward(backend, &hidden)?;
            Ok(DiffusionResult { logits, hidden })
        })();
        backend.end_batch()?;
        r
    }

    /// Сброс кэшей до инжектной длины (после диффузии; spec-decode
    /// сохраняет только принятые токены).
    pub fn truncate_caches(&mut self, inject_len: usize) {
        for c in &mut self.caches {
            c.pos = inject_len;
        }
    }
}

/// Результат диффузии: логиты (lm_head таргета) + скрытые (свой out_norm).
pub struct DiffusionResult {
    pub logits: Tensor, // [1, T, V]
    pub hidden: Tensor, // [1, T, D]
}

/// conv-проекция шума: dynamic = proj(x) — [n_groups·k·2, T].
fn conv_dynamic(
    backend: &dyn Backend,
    proj: &Tensor,
    x: &Tensor,
) -> Result<Tensor, BackendError> {
    let (b, t, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    let x2 = x.view(vec![b * t, d]);
    let y = backend.zeros(vec![proj.shape()[0], b * t])?;
    backend.gemm_nt(&x2, proj, &y)?;
    Ok(y)
}

/// Один tap-конв (side: 0 = вход, 1 = выход) по формуле спеки.
fn conv_side(
    backend: &dyn Backend,
    x: &Tensor,
    base: &Tensor,
    dynamic: &Tensor,
    side: usize,
    block_size: usize,
) -> Result<Tensor, BackendError> {
    let (b, t, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    let kernel = base.shape()[1];
    let n_groups = dynamic.shape()[0] / (kernel * 2);
    let group = d / n_groups;
    let y = backend.zeros(x.shape().to_vec())?;
    backend.dflash2_conv(
        &x.view(vec![b * t, d]),
        base,
        dynamic,
        &y.view(vec![b * t, d]),
        kernel,
        group,
        block_size,
        side,
    )?;
    Ok(y)
}

fn add(backend: &dyn Backend, a: &Tensor, b: &Tensor) -> Result<Tensor, BackendError> {
    let y = backend.zeros(a.shape().to_vec())?;
    backend.add(a, b, &y)?;
    Ok(y)
}

fn load_dense_layer(
    backend: &dyn Backend,
    w: &WeightsMap<'_>,
    p: &dyn Fn(&str) -> String,
    cfg: &Qwen3Config,
) -> Result<Qwen3Layer, BackendError> {
    let up = |name: &str, shape: Vec<usize>| -> Result<Tensor, BackendError> {
        let tensor = w
            .get(name)
            .ok_or_else(|| BackendError::Shape(format!("missing tensor {name}")))?;
        match tensor.format {
            WeightFormat::Q4_0
            | WeightFormat::Q8_0
            | WeightFormat::Q4_K
            | WeightFormat::Q6_K => backend.from_quantized(tensor.raw(), shape, tensor.format),
            _ => {
                let f = dequant_f32(tensor);
                upload_f16(backend, &f, shape)
            }
        }
    };
    // q/k/v fused (как Qwen3Model::load): wqkv [q_size + 2·kv_size, dim]
    let mut wqkv = dequant_f32(w.get(&p("attn_q.weight")).ok_or_else(|| {
        BackendError::Shape("attn_q.weight missing".into())
    })?);
    wqkv.extend(dequant_f32(w.get(&p("attn_k.weight")).ok_or_else(|| {
        BackendError::Shape("attn_k.weight missing".into())
    })?));
    wqkv.extend(dequant_f32(w.get(&p("attn_v.weight")).ok_or_else(|| {
        BackendError::Shape("attn_v.weight missing".into())
    })?));
    let attn = Attention {
        wqkv: upload_f16(
            backend,
            &wqkv,
            vec![cfg.n_heads * cfg.head_dim + 2 * cfg.n_kv_heads * cfg.head_dim, cfg.dim],
        )?,
        wo: up(&p("attn_output.weight"), vec![cfg.dim, cfg.dim])?,
        qkv_bias: None,
        q_norm: Some(backend.from_cpu_f32(
            &dequant_f32(w.get(&p("attn_q_norm.weight")).ok_or_else(|| {
                BackendError::Shape("attn_q_norm.weight missing".into())
            })?),
            vec![cfg.head_dim],
        )?),
        k_norm: Some(backend.from_cpu_f32(
            &dequant_f32(w.get(&p("attn_k_norm.weight")).ok_or_else(|| {
                BackendError::Shape("attn_k_norm.weight missing".into())
            })?),
            vec![cfg.head_dim],
        )?),
        n_heads: cfg.n_heads,
        n_kv_heads: cfg.n_kv_heads,
        head_dim: cfg.head_dim,
    };
    // gate/up fused (как Qwen3Model::load): w13 [2·ffn, dim]
    let mut w13 = dequant_f32(w.get(&p("ffn_gate.weight")).ok_or_else(|| {
        BackendError::Shape("ffn_gate.weight missing".into())
    })?);
    w13.extend(dequant_f32(w.get(&p("ffn_up.weight")).ok_or_else(|| {
        BackendError::Shape("ffn_up.weight missing".into())
    })?));
    let ffn = SwiGLU {
        w13: upload_f16(backend, &w13, vec![2 * cfg.ffn_dim, cfg.dim])?,
        w2: up(&p("ffn_down.weight"), vec![cfg.dim, cfg.ffn_dim])?,
    };
    Ok(Qwen3Layer {
        attn_norm: RmsNorm::from_weights(
            backend,
            dequant_f32(w.get(&p("attn_norm.weight")).ok_or_else(|| {
                BackendError::Shape("attn_norm.weight missing".into())
            })?),
            cfg.norm_eps,
        )?,
        attn,
        ffn_norm: RmsNorm::from_weights(
            backend,
            dequant_f32(w.get(&p("ffn_norm.weight")).ok_or_else(|| {
                BackendError::Shape("ffn_norm.weight missing".into())
            })?),
            cfg.norm_eps,
        )?,
        ffn,
    })
}

/// f16-загрузка весов (как upload_w в qwen3.rs).
fn upload_f16(backend: &dyn Backend, data: &[f32], shape: Vec<usize>) -> Result<Tensor, BackendError> {
    if backend.supports_f16_weights() {
        let f16: Vec<half::f16> = data.iter().map(|&x| half::f16::from_f32(x)).collect();
        backend.from_cpu_f16(&f16, shape)
    } else {
        backend.from_cpu_f32(data, shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::MetalBackend;
    use crate::qwen3::Qwen3Config;

    fn cfg() -> Qwen3Config {
        Qwen3Config {
            dim: 16,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 8,
            ffn_dim: 32,
            vocab_size: 8,
            rope_theta: 10000.0,
            norm_eps: 1e-6,
            rope_max_seq: 128,
            swa_window: None,
            mrope_sections: None,
            n_deepstack: None,
        }
    }

    fn storage(c: &Qwen3Config) -> Vec<(String, Vec<f32>, Vec<usize>)> {
        use crate::harness::Lcg;
        let mut rng = Lcg::new(42);
        let mut st = Vec::new();
        st.push((
            "token_embd.weight".into(),
            {
                let mut v = vec![0f32; c.vocab_size * c.dim];
                rng.fill(&mut v);
                v
            },
            vec![c.vocab_size, c.dim],
        ));
        for i in 0..c.n_layers {
            let p = |s: &str| format!("blk.{i}.{s}");
            for (n, shape) in [
                ("attn_norm.weight", vec![c.dim]),
                ("ffn_norm.weight", vec![c.dim]),
                ("attn_q_norm.weight", vec![c.head_dim]),
                ("attn_k_norm.weight", vec![c.head_dim]),
                ("attn_q.weight", vec![c.dim, c.dim]),
                ("attn_k.weight", vec![c.n_kv_heads * c.head_dim, c.dim]),
                ("attn_v.weight", vec![c.n_kv_heads * c.head_dim, c.dim]),
                ("attn_output.weight", vec![c.dim, c.dim]),
                ("ffn_gate.weight", vec![c.ffn_dim, c.dim]),
                ("ffn_up.weight", vec![c.ffn_dim, c.dim]),
                ("ffn_down.weight", vec![c.dim, c.ffn_dim]),
                ("attn_conv_base", vec![c.dim, 3, 2]),
                ("attn_conv_proj", vec![(c.dim / 4) * 3 * 2, c.dim]),
                ("ffn_conv_base", vec![c.dim, 3, 2]),
                ("ffn_conv_proj", vec![(c.dim / 4) * 3 * 2, c.dim]),
            ] {
                let mut v = vec![0f32; shape.iter().product()];
                rng.fill(&mut v);
                st.push((p(n), v, shape));
            }
        }
        st.push((
            "selector_predecessor".into(),
            {
                let mut v = vec![0f32; c.dim * 40];
                rng.fill(&mut v);
                v
            },
            vec![c.dim, 40],
        ));
        st.push((
            "selector_successor".into(),
            {
                let mut v = vec![0f32; c.dim * 40];
                rng.fill(&mut v);
                v
            },
            vec![c.dim, 40],
        ));
        st.push((
            "output_norm.weight".into(),
            {
                let mut v = vec![0f32; c.dim];
                rng.fill(&mut v);
                v
            },
            vec![c.dim],
        ));
        st.push((
            "output.weight".into(),
            {
                let mut v = vec![0f32; c.vocab_size * c.dim];
                rng.fill(&mut v);
                v
            },
            vec![c.vocab_size, c.dim],
        ));
        st
    }

    #[test]
    fn dflash_load_shapes() {
        let backend = MetalBackend::new().unwrap();
        let c = cfg();
        // собрать WeightsMap вручную (как load_thinker в omni.rs): два прохода
        // (borrow: bytes mut в 1-м, ссылки во 2-м)
        let st = storage(&c);
        let mut bytes: Vec<u8> = Vec::new();
        let mut layout: Vec<(String, Vec<u64>, u64, usize)> = Vec::new();
        for (name, data, shape) in &st {
            let off = bytes.len();
            for v in data {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            layout.push((
                name.clone(),
                shape.iter().map(|&d| d as u64).collect(),
                off as u64,
                data.len() * 4,
            ));
        }
        let tensors = layout
            .into_iter()
            .map(|(name, shape, off, len)| crate::weights_map::WeightTensor {
                name,
                shape,
                format: crate::dtype::WeightFormat::Plain(crate::dtype::Dtype::F32),
                offset: off,
                data: &bytes[off as usize..off as usize + len],
            })
            .collect();
        let wm = crate::weights_map::WeightsMap {
            arch: "dflash".into(),
            tensors,
            storage: vec![crate::weights_map::Storage {
                offset: 0,
                data: &bytes,
            }],
        };
        let m = DFlashModel::load(&backend, &wm, c.clone(), 16, 3, 4, vec![0]).unwrap();
        assert_eq!(m.layers.len(), 2);
        assert!(m.selector.is_some());
        assert_eq!(m.cfg.dim, 16);
        assert_eq!(m.conv_kernel, 3);
        // conv-тензоры загружены (f16 — проверяем shape)
        assert_eq!(m.layers[0].attn_conv.base.shape(), &[16, 3, 2]);
        assert_eq!(m.layers[0].attn_conv.proj.as_ref().unwrap().shape(), &[24, 16]);
        let _ = m;
    }

    #[test]
    fn dflash2_conv_metal_matches_cpu() {
        // Спека PR #27342: y[t,i] = Σ_tap (coeff + base)·x_block[t−tap,i],
        // сдвиг ВНУТРИ блока (паддинг нулями на границах), coeff по группам.
        use crate::harness::Lcg;
        let backend = MetalBackend::new().unwrap();
        let cpu = crate::cpu::CpuBackend::default();
        let (tokens, dim, kernel, group, block) = (9usize, 16usize, 3usize, 4usize, 4usize);
        let n_groups = dim / group; // 4
        let mut rng = Lcg::new(7);
        let mut xd = vec![0f32; tokens * dim];
        let mut bd = vec![0f32; dim * kernel * 2];
        let mut dd = vec![0f32; n_groups * kernel * 2 * tokens];
        rng.fill(&mut xd);
        rng.fill(&mut bd);
        rng.fill(&mut dd);
        let xt = backend.from_cpu_f32(&xd, vec![tokens, dim]).unwrap();
        let bt = backend.from_cpu_f32(&bd, vec![dim, kernel, 2]).unwrap();
        let dt = backend
            .from_cpu_f32(&dd, vec![n_groups * kernel * 2, tokens])
            .unwrap();
        let (xc, bc, dc) = (
            cpu.from_cpu_f32(&xd, vec![tokens, dim]).unwrap(),
            cpu.from_cpu_f32(&bd, vec![dim, kernel, 2]).unwrap(),
            cpu.from_cpu_f32(&dd, vec![n_groups * kernel * 2, tokens]).unwrap(),
        );
        for side in [0usize, 1] {
            let ym = backend
                .from_cpu_f32(&vec![0f32; tokens * dim], vec![tokens, dim])
                .unwrap();
            let yc = cpu
                .from_cpu_f32(&vec![0f32; tokens * dim], vec![tokens, dim])
                .unwrap();
            backend
                .dflash2_conv(&xt, &bt, &dt, &ym, kernel, group, block, side)
                .unwrap();
            cpu.dflash2_conv(&xc, &bc, &dc, &yc, kernel, group, block, side)
                .unwrap();
            let md = backend.to_cpu_f32(&ym).unwrap();
            let cd = cpu.to_cpu_f32(&yc).unwrap();
            crate::harness::assert_close(
                &md,
                &cd,
                crate::harness::standard_tolerance().0,
                crate::harness::standard_tolerance().1,
            )
            .unwrap_or_else(|e| panic!("side={side}: {e}"));
        }
        // Краевой случай: вход 0 → выход 0 при любых весах (паддинг нулями
        // на границах блоков не вносит мусора).
        let xz = vec![0f32; tokens * dim];
        let xzt = backend.from_cpu_f32(&xz, vec![tokens, dim]).unwrap();
        let yz = backend
            .from_cpu_f32(&vec![0f32; tokens * dim], vec![tokens, dim])
            .unwrap();
        backend
            .dflash2_conv(&xzt, &bt, &dt, &yz, kernel, group, block, 0)
            .unwrap();
        let zd_out = backend.to_cpu_f32(&yz).unwrap();
        assert!(zd_out.iter().all(|&v| v == 0.0));
    }

    /// Диффузия = чистый qwen3-forward, когда conv-модули выключены
    /// (proj=None в storage): те же эмбеддинги, dense-слои, out_norm,
    /// lm_head таргета. При t=1 non-causal attn тождествен causal.
    #[test]
    fn diffuse_matches_qwen3_forward_without_conv() {
        use crate::nn::{Embedding, Linear};
        let backend = MetalBackend::new().unwrap();
        let c = cfg();
        let st = storage(&c);
        let mut bytes: Vec<u8> = Vec::new();
        let mut layout: Vec<(String, Vec<u64>, u64, usize)> = Vec::new();
        for (name, data, shape) in &st {
            if name.ends_with("_conv_proj") {
                continue; // conv-модули выключены
            }
            let off = bytes.len();
            for v in data {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            layout.push((
                name.clone(),
                shape.iter().map(|&d| d as u64).collect(),
                off as u64,
                data.len() * 4,
            ));
        }
        let tensors = layout
            .into_iter()
            .map(|(name, shape, off, len)| crate::weights_map::WeightTensor {
                name,
                shape,
                format: crate::dtype::WeightFormat::Plain(crate::dtype::Dtype::F32),
                offset: off,
                data: &bytes[off as usize..off as usize + len],
            })
            .collect();
        let wm = crate::weights_map::WeightsMap {
            arch: "dflash".into(),
            tensors,
            storage: vec![crate::weights_map::Storage {
                offset: 0,
                data: &bytes,
            }],
        };
        let mut qwen = crate::qwen3::Qwen3Model::load(&backend, &wm, c.clone()).unwrap();
        let mut dflash = DFlashModel::load(&backend, &wm, c.clone(), 16, 3, 4, vec![0]).unwrap();
        assert!(dflash.layers[0].attn_conv.proj.is_none());
        // общие эмбеддинги (те же данные, что у qwen)
        let ew = dequant_f32(wm.get("token_embd.weight").unwrap());
        let embed = Embedding::from_weights(&backend, ew, vec![c.vocab_size, c.dim]).unwrap();
        dflash.set_target_embed(embed.table.clone());
        // lm_head таргета (f16, как Qwen3Model::load)
        let ow = dequant_f32(wm.get("output.weight").unwrap());
        let f16: Vec<half::f16> = ow.iter().map(|&x| half::f16::from_f32(x)).collect();
        let lm_head = backend.from_cpu_f16(&f16, vec![c.vocab_size, c.dim]).unwrap();

        // t=1: каузальная маска не строится ни в одном пути — пути тождественны
        let ids = [3u32];
        let h = qwen.forward_hidden(&backend, &ids).unwrap();
        let d = dflash.diffuse(&backend, &ids, &lm_head).unwrap();
        crate::harness::assert_close(
            &backend.to_cpu_f32(&d.hidden).unwrap(),
            &backend.to_cpu_f32(&h).unwrap(),
            1e-3,
            1e-4,
        )
        .unwrap();
        let lq = Linear {
            w: lm_head.clone(),
            b: None,
        }
        .forward(&backend, &h)
        .unwrap();
        crate::harness::assert_close(
            &backend.to_cpu_f32(&d.logits).unwrap(),
            &backend.to_cpu_f32(&lq).unwrap(),
            2e-3,
            1e-4,
        )
        .unwrap();

        // t=2: non-causal диффузия (взаимный attn) — только проверка конечности
        let ids2 = [3u32, 5];
        let d2 = dflash.diffuse(&backend, &ids2, &lm_head).unwrap();
        let lg = backend.to_cpu_f32(&d2.logits).unwrap();
        assert!(lg.iter().all(|v| v.is_finite()));
        dflash.truncate_caches(0);
    }
}
