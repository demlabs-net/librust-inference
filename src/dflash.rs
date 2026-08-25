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
        Ok(Self {
            cfg,
            target_embed: None,
            layers,
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
}
