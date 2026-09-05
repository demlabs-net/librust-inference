//! DFlash-драфтер (Ф2.3/Ф2.4/Ф2.5) — по спеке llama.cpp PR #27342
//! (см. БЗ SLC: dflash-speka-llama-pr-27342).
//!
//! Структура: dense-слои (Qwen3-архитектура, эмбеддинги ОБЩИЕ с таргетом!)
//! + two-tap dynamic conv (attn/ffn_conv_base + conv_proj) + селектор (v2:
//! predecessor/successor codebooks, top-k лэттис). Два режима:
//! - инжект (embd): фичи таргета → K/V-проекции каждого слоя → кэш;
//! - диффузия (token): [committed, MASK...] → dense+conv → логиты/скрытые.

use crate::backend::{Backend, Tensor};
use crate::dtype::WeightFormat;
use crate::error::BackendError;
use crate::nn::{Attention, KvCache, RmsNorm, SwiGLU};
use crate::qwen3::{Qwen3Config, Qwen3Layer};
use crate::weights_map::{WeightsMap, dequant_f32};

/// Conv-модуль DFlash-слоя (two-tap dynamic, attn или ffn). Отсутствует у
/// DFlash v1 (только v2 имеет conv/selector).
#[derive(Debug, Clone)]
pub struct DFlashConv {
    /// [2, kernel, n_embd] — статическая база (ggml-порядок: side·k·D + tap·D + i).
    pub base: Option<Tensor>,
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

/// Селектор DFlash2 (спека PR #27342): codebook-эмбеддинги
/// предшественника/преемника + проекция скрытых в ранг.
#[derive(Debug, Clone)]
pub struct Selector {
    /// [rank, n_vocab] — кодбук предшественника (по колонкам токенов).
    pub predecessor: Tensor,
    /// [rank, n_vocab] — кодбук преемника.
    pub successor: Tensor,
    /// [n_embd, rank] — проекция t_embd в ранг (лэттис-условие).
    pub hidden: Tensor,
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
    /// Encoder-модуль инжекта (v1): фичи таргета [T, n_target_layers·D] →
    /// fc [D, ·] → enc_norm → 2560-фичи для wkv-проекций.
    pub fc: Option<Tensor>,
    pub enc_norm: Option<RmsNorm>,
    pub selector: Option<Selector>,
    pub caches: Vec<KvCache>,
    pub block_size: usize,
    pub conv_kernel: usize,
    pub conv_group: usize,
    pub extract_layers: Vec<usize>,
    /// MASK-токен диффузии (из GGUF KV tokenizer.ggml.mask_token_id).
    pub mask_token: u32,
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
        mask_token: u32,
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
            let f = dequant_f32(
                w.get(name)
                    .ok_or_else(|| BackendError::Shape(format!("missing tensor {name}")))?,
            );
            upload_f16(backend, &f, shape)
        };
        // N (iter-9): load_with_selector is pub — guard here too (not just
        // in load_gguf): conv_group == 0 is a div-by-zero below, and a
        // kernel of 0 with conv tensors present makes degenerate shapes
        // that panic in conv_side.
        if conv_group == 0 {
            return Err(BackendError::Shape(
                "dflash: conv_group_size must be >= 1".into(),
            ));
        }
        let n_groups = cfg.dim / conv_group;
        let conv_tensors_present = ["attn_conv_base", "ffn_conv_base"]
            .iter()
            .any(|n| w.get(&format!("blk.0.{n}")).is_some());
        if conv_tensors_present && conv_kernel == 0 {
            return Err(BackendError::Shape(
                "dflash: conv_kernel_size must be >= 1 when conv tensors are present".into(),
            ));
        }
        let mut layers = Vec::with_capacity(cfg.n_layers);
        let mut caches = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = |s: &str| format!("blk.{i}.{s}");
            let dense = load_dense_layer(backend, w, &p, &cfg)?;
            let attn_conv = DFlashConv {
                // ggml-порядок base: [2, kernel, n_embd] (side·k·D + tap·D + i);
                // v1-драфтеры не имеют conv — base/proj опциональны.
                base: if w.get(&p("attn_conv_base")).is_some() {
                    Some(up_q(&p("attn_conv_base"), vec![2, conv_kernel, cfg.dim])?)
                } else {
                    None
                },
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
                base: if w.get(&p("ffn_conv_base")).is_some() {
                    Some(up_q(&p("ffn_conv_base"), vec![2, conv_kernel, cfg.dim])?)
                } else {
                    None
                },
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
                crate::dtype::KvFormat::F16,
                crate::dtype::KvFormat::F16,
                cfg.rope_theta,
                None,
            )?);
        }
        // N (iter-10): bound the KV to the context window (see qwen3).
        for c in &mut caches {
            c.set_max_cap(Some(cfg.rope_max_seq));
        }
        let selector = if w.get("selector_predecessor").is_some() {
            // rank/top_k — из метаданных драфтера (GGUF KV
            // dflash.selector_rank / dflash.selector_top_k); задел: параметры
            let (rank, top_k) = (4usize, 10usize);
            // Селектор — маленький и читается в select() на CPU (жадный путь):
            // держим в F32, чтобы to_cpu_f32 не падал на f16-тензорах.
            // ggml-порядок (row-major реверс): prev/next [n_vocab, rank]
            // (файл {rank, n_vocab}), hidden [rank, n_embd] (файл {n_embd, rank}).
            let up_f32 = |name: &str, shape: Vec<usize>| -> Result<Tensor, BackendError> {
                let f = dequant_f32(
                    w.get(name)
                        .ok_or_else(|| BackendError::Shape(format!("missing tensor {name}")))?,
                );
                backend.from_cpu_f32(&f, shape)
            };
            Some(Selector {
                predecessor: up_f32("selector_predecessor", vec![cfg.vocab_size, rank])?,
                successor: up_f32("selector_successor", vec![cfg.vocab_size, rank])?,
                hidden: up_f32("selector_hidden", vec![rank, cfg.dim])?,
                rank,
                top_k,
            })
        } else {
            None
        };
        let _ = up;
        let out_norm =
            RmsNorm::from_weights(
                backend,
                dequant_f32(w.get("output_norm.weight").ok_or_else(|| {
                    BackendError::Shape("missing tensor output_norm.weight".into())
                })?),
                cfg.norm_eps,
            )?;
        // encoder-модуль инжекта (v1; у v2 отсутствует): fc [D, n_target·D],
        // enc.output_norm.weight — своя нормализация инжекта.
        let fc = match w.get("fc.weight") {
            Some(t) => Some(up_q(
                "fc.weight",
                vec![
                    cfg.dim,
                    t.shape.iter().map(|&d| d as usize).product::<usize>() / cfg.dim,
                ],
            )?),
            None => None,
        };
        let enc_norm = match w.get("enc.output_norm.weight") {
            Some(t) => Some(RmsNorm::from_weights(
                backend,
                dequant_f32(t),
                cfg.norm_eps,
            )?),
            None => None,
        };
        Ok(Self {
            cfg,
            target_embed: None,
            layers,
            out_norm,
            fc,
            enc_norm,
            selector,
            caches,
            block_size,
            conv_kernel,
            conv_group,
            extract_layers,
            mask_token,
        })
    }

    /// Привязать общие эмбеддинги таргета (драфтер их не хранит).
    pub fn set_target_embed(&mut self, t: Tensor) {
        self.target_embed = Some(t);
    }

    /// Загрузка реального GGUF-драфтера z-lab (v1/v2): конфиг из метаданных
    /// {arch}.* (embedding_length/attention.*/block_count/rope…), гиперы
    /// диффузии ({arch}.block_size/conv_kernel_size/conv_group_size/
    /// selector_rank/selector_top_k/target_layers), MASK из вокабуляра.
    pub fn load_gguf(
        backend: &dyn Backend,
        gf: &crate::gguf::GgufFile<'_>,
        wm: &WeightsMap<'_>,
        vocab_size: usize,
    ) -> Result<Self, BackendError> {
        use crate::gguf::GgufValue;
        let arch = gf.architecture().unwrap_or("dflash").to_string();
        let p = |s: &str| format!("{arch}.{s}");
        let u = |k: &str| -> Result<usize, BackendError> {
            gf.get(k)
                .and_then(GgufValue::as_u32)
                .map(|v| v as usize)
                .ok_or_else(|| BackendError::Shape(format!("missing {k}")))
        };
        let f = |k: &str, d: f32| -> f32 {
            gf.get(k)
                .and_then(|v| match v {
                    GgufValue::F32(x) => Some(*x),
                    GgufValue::F64(x) => Some(*x as f32),
                    _ => None,
                })
                .unwrap_or(d)
        };
        let dim = u(&p("embedding_length"))?;
        let n_layers = u(&p("block_count"))?;
        let n_heads = u(&p("attention.head_count"))?;
        if n_heads == 0 {
            return Err(BackendError::Shape(format!(
                "{arch}.attention.head_count must be >= 1, got 0"
            )));
        }
        let cfg = Qwen3Config {
            dim,
            n_layers,
            n_heads,
            n_kv_heads: u(&p("attention.head_count_kv"))?,
            head_dim: u(&p("attention.key_length")).unwrap_or(dim / n_heads),
            ffn_dim: u(&p("feed_forward_length"))?,
            vocab_size,
            rope_theta: f(&p("rope.freq_base"), 1e6),
            rope_max_seq: u(&p("context_length")).unwrap_or(8192),
            norm_eps: f(&p("attention.layer_norm_rms_epsilon"), 1e-6),
            swa_window: gf
                .get(&p("attention.sliding_window"))
                .and_then(GgufValue::as_u32)
                .map(|v| v as usize),
            mrope_sections: None,
            n_deepstack: None,
        };
        let block_size = u(&p("block_size"))?;
        // N (iter-9): GGUF metadata is external input — 0/absent values
        // panicked later (Vec::with_capacity(t - 1) underflow at decode,
        // div-by-zero in dim / head_count).
        if block_size == 0 {
            return Err(BackendError::Shape(format!(
                "{arch}.block_size must be >= 1, got 0"
            )));
        }
        let conv_kernel = u(&p("conv_kernel_size")).unwrap_or(0);
        let conv_group = u(&p("conv_group_size")).unwrap_or(1);
        if conv_group == 0 {
            return Err(BackendError::Shape(format!(
                "{arch}.conv_group_size must be >= 1, got 0"
            )));
        }
        let extract = gf
            .get(&p("target_layers"))
            .and_then(GgufValue::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(GgufValue::as_u32)
                    .map(|v| v as usize)
                    .collect()
            })
            .unwrap_or_default();
        let mask_token = gf
            .get("tokenizer.ggml.mask_token_id")
            .and_then(GgufValue::as_u32)
            .unwrap_or(151850);
        let sel_rank = u(&p("selector_rank")).ok();
        let sel_top_k = u(&p("selector_top_k")).ok();
        Self::load_with_selector(
            backend,
            wm,
            cfg,
            block_size,
            conv_kernel,
            conv_group,
            extract,
            mask_token,
            sel_rank,
            sel_top_k,
        )
    }

    /// Загрузка драфтера из safetensors z-lab (+ config.json рядом):
    /// BF16-веса (HF-порядок [out, in] — как наш row-major), имена
    /// layers.K.* → blk.K.*, hidden_norm = инжект-норм (enc.output_norm).
    /// Параметры диффузии — из config.json (dflash_config: mask_token_id,
    /// target_layer_ids; block_size; rope_theta; rms_norm_eps).
    pub fn load_safetensors(
        backend: &dyn Backend,
        model_path: &std::path::Path,
        vocab_size: usize,
    ) -> Result<Self, BackendError> {
        let dir = model_path.parent().unwrap_or(std::path::Path::new("."));
        let cfg_path = dir.join("config.json");
        let cfgj: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&cfg_path)
                .map_err(|e| BackendError::Shape(format!("config.json: {e}")))?,
        )
        .map_err(|e| BackendError::Shape(format!("config.json: {e}")))?;
        let g = |k: &str| -> Result<usize, BackendError> {
            cfgj.get(k)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .ok_or_else(|| BackendError::Shape(format!("config: missing {k}")))
        };
        let f = |k: &str, d: f32| -> f32 {
            cfgj.get(k)
                .and_then(|v| v.as_f64())
                .map(|v| v as f32)
                .unwrap_or(d)
        };
        let dflash = cfgj
            .get("dflash_config")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let mask_token = dflash
            .get("mask_token_id")
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(151669);
        let target_layers = dflash
            .get("target_layer_ids")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64())
                    .map(|v| v as usize)
                    .collect()
            })
            .unwrap_or_default();
        let dim = g("hidden_size")?;
        let n_layers = g("num_hidden_layers")?;
        // N (iter-10): the safetensors config reader had NONE of the
        // load_gguf guards — head_count = 0 panicked in the div below and
        // block_size = 0 underflowed Vec::with_capacity(t - 1) at decode.
        let n_heads = g("num_attention_heads")?;
        if n_heads == 0 {
            return Err(BackendError::Shape(
                "dflash safetensors: num_attention_heads must be >= 1, got 0".into(),
            ));
        }
        let head_dim = g("head_dim").unwrap_or(dim / n_heads);
        let cfg = Qwen3Config {
            dim,
            n_layers,
            n_heads,
            n_kv_heads: g("num_key_value_heads")?,
            head_dim,
            ffn_dim: g("intermediate_size")?,
            vocab_size,
            rope_theta: f("rope_theta", 1e6),
            rope_max_seq: g("max_position_embeddings").unwrap_or(8192),
            norm_eps: f("rms_norm_eps", 1e-6),
            swa_window: None,
            mrope_sections: None,
            n_deepstack: None,
        };
        let block_size = g("block_size")?;
        if block_size == 0 {
            return Err(BackendError::Shape(
                "dflash safetensors: block_size must be >= 1, got 0".into(),
            ));
        }
        // safetensors → WeightsMap (f32, имена blk.*)
        let st = crate::weights::read_safetensors_f32(model_path)
            .map_err(|e| BackendError::Shape(format!("safetensors: {e}")))?;
        let mut blobs: Vec<Vec<u8>> = Vec::new();
        let mut meta: Vec<(String, Vec<u64>, usize)> = Vec::new();
        let map = |name: &str| -> Option<String> {
            if name == "fc.weight" {
                return Some("fc.weight".into());
            }
            if name == "norm.weight" {
                return Some("output_norm.weight".into());
            }
            if name == "hidden_norm.weight" {
                return Some("enc.output_norm.weight".into());
            }
            let parts: Vec<&str> = name.split('.').collect();
            if parts.len() >= 4 && parts[0] == "layers" {
                let k = parts[1];
                let rest = parts[2..].join(".");
                let mapped = match rest.as_str() {
                    "input_layernorm.weight" => "attn_norm.weight",
                    "post_attention_layernorm.weight" => "ffn_norm.weight",
                    "self_attn.q_proj.weight" => "attn_q.weight",
                    "self_attn.k_proj.weight" => "attn_k.weight",
                    "self_attn.v_proj.weight" => "attn_v.weight",
                    "self_attn.o_proj.weight" => "attn_output.weight",
                    "self_attn.q_norm.weight" => "attn_q_norm.weight",
                    "self_attn.k_norm.weight" => "attn_k_norm.weight",
                    "mlp.gate_proj.weight" => "ffn_gate.weight",
                    "mlp.up_proj.weight" => "ffn_up.weight",
                    "mlp.down_proj.weight" => "ffn_down.weight",
                    _ => return None,
                };
                return Some(format!("blk.{k}.{mapped}"));
            }
            None
        };
        for (name, (data, shape)) in &st {
            let mapped = map(name).ok_or_else(|| {
                BackendError::Shape(format!("safetensors: незнакомый тензор {name}"))
            })?;
            let mut b = Vec::with_capacity(data.len() * 4);
            for v in data {
                b.extend_from_slice(&v.to_le_bytes());
            }
            meta.push((mapped, shape.iter().map(|&d| d as u64).collect(), b.len()));
            blobs.push(b);
        }
        let mut buf: Vec<u8> = Vec::with_capacity(blobs.iter().map(|b| b.len()).sum());
        let mut offs = Vec::with_capacity(blobs.len());
        for b in &blobs {
            offs.push(buf.len());
            buf.extend_from_slice(b);
        }
        let tensors = meta
            .into_iter()
            .zip(offs)
            .map(
                |((name, shape, len), off)| crate::weights_map::WeightTensor {
                    name,
                    shape,
                    format: crate::dtype::WeightFormat::Plain(crate::dtype::Dtype::F32),
                    offset: off as u64,
                    data: &buf[off..off + len],
                },
            )
            .collect::<Vec<_>>();
        let wm = crate::weights_map::WeightsMap {
            arch: "dflash".into(),
            tensors,
            storage: vec![crate::weights_map::Storage {
                offset: 0,
                data: &buf,
            }],
        };
        Self::load_with_selector(
            backend,
            &wm,
            cfg,
            block_size,
            0,
            1,
            target_layers,
            mask_token,
            None,
            None,
        )
    }

    /// Диагностическая загрузка с транспонированием 2D-весов (гипотеза
    /// «конвертер не транспонировал»): каждый 2D .weight деквантуется и
    /// читается как [out, in] вместо ggml [in, out]. Для сравнения
    /// предсказаний при сомнительном layout файла.
    pub fn load_gguf_transposed(
        backend: &dyn Backend,
        gf: &crate::gguf::GgufFile<'_>,
        wm: &WeightsMap<'_>,
        vocab_size: usize,
    ) -> Result<Self, BackendError> {
        use crate::dtype::{Dtype, WeightFormat};
        // данные собираем в отдельные буферы; offset'ы — кумулятивные
        let mut blobs: Vec<Vec<u8>> = Vec::new();
        let mut meta: Vec<(String, Vec<u64>, WeightFormat, usize)> = Vec::new();
        for t in wm.tensors() {
            let sh: Vec<usize> = t.shape.iter().map(|&d| d as usize).collect();
            let trans = t.name.ends_with(".weight")
                && !t.name.contains("norm")
                && sh.len() == 2
                && sh[0] > 1
                && sh[1] > 1;
            if trans {
                let f = dequant_f32(t);
                let (r, c) = (sh[0], sh[1]);
                let mut ft = vec![0f32; r * c];
                for i in 0..r {
                    for j in 0..c {
                        ft[j * r + i] = f[i * c + j];
                    }
                }
                let mut b = Vec::with_capacity(ft.len() * 4);
                for v in &ft {
                    b.extend_from_slice(&v.to_le_bytes());
                }
                meta.push((
                    t.name.clone(),
                    vec![sh[1] as u64, sh[0] as u64],
                    WeightFormat::Plain(Dtype::F32),
                    b.len(),
                ));
                blobs.push(b);
            } else {
                meta.push((t.name.clone(), t.shape.clone(), t.format, t.data.len()));
                blobs.push(t.data.to_vec());
            }
        }
        let total: usize = blobs.iter().map(|b| b.len()).sum();
        let mut buf: Vec<u8> = Vec::with_capacity(total);
        let mut offs = Vec::with_capacity(blobs.len());
        for b in &blobs {
            offs.push(buf.len());
            buf.extend_from_slice(b);
        }
        let tensors = meta
            .into_iter()
            .zip(offs)
            .map(
                |((name, shape, format, len), off)| crate::weights_map::WeightTensor {
                    name,
                    shape,
                    format,
                    offset: off as u64,
                    data: &buf[off..off + len],
                },
            )
            .collect::<Vec<_>>();
        let wm2 = crate::weights_map::WeightsMap {
            arch: "dflash".into(),
            tensors,
            storage: vec![crate::weights_map::Storage {
                offset: 0,
                data: &buf,
            }],
        };
        Self::load_gguf(backend, gf, &wm2, vocab_size)
    }

    /// Загрузка с явными параметрами селектора (load_gguf передаёт их из
    /// метаданных; None — дефолт (4, 10)).
    pub fn load_with_selector(
        backend: &dyn Backend,
        w: &WeightsMap<'_>,
        cfg: Qwen3Config,
        block_size: usize,
        conv_kernel: usize,
        conv_group: usize,
        extract_layers: Vec<usize>,
        mask_token: u32,
        sel_rank: Option<usize>,
        sel_top_k: Option<usize>,
    ) -> Result<Self, BackendError> {
        let mut m = Self::load(
            backend,
            w,
            cfg,
            block_size,
            conv_kernel,
            conv_group,
            extract_layers,
            mask_token,
        )?;
        // перезагрузить селектор с метаданными, если они есть
        if let (Some(rank), Some(top_k), Some(_)) =
            (sel_rank, sel_top_k, w.get("selector_predecessor"))
        {
            let up_f32 = |name: &str, shape: Vec<usize>| -> Result<Tensor, BackendError> {
                let f = dequant_f32(
                    w.get(name)
                        .ok_or_else(|| BackendError::Shape(format!("missing tensor {name}")))?,
                );
                backend.from_cpu_f32(&f, shape)
            };
            m.selector = Some(Selector {
                predecessor: up_f32("selector_predecessor", vec![m.cfg.vocab_size, rank])?,
                successor: up_f32("selector_successor", vec![m.cfg.vocab_size, rank])?,
                hidden: up_f32("selector_hidden", vec![rank, m.cfg.dim])?,
                rank,
                top_k,
            });
        }
        Ok(m)
    }

    /// Инжект фич таргета (спека: llama_encode fc+norm_enc, затем decoder
    /// проекцирует в k/v): features [n_tokens, n_target_layers·D] (конкат
    /// extract-слоёв таргета) → fc [D, ·] → enc_norm → на каждый слой:
    /// k/v = wk/wv(enc) → k_norm → rope → кэш. pos — позиции токенов таргета.
    pub fn inject_features(
        &mut self,
        backend: &dyn Backend,
        features: &Tensor,
        pos: usize,
    ) -> Result<(), BackendError> {
        backend.begin_batch()?;
        let r = (|| {
            let t = features.shape()[0];
            // encoder-модуль (v1): fc + norm_enc; у v2 фичи уже спроецированы
            let enc = match (&self.fc, &self.enc_norm) {
                (Some(fc), Some(norm)) => {
                    let d = fc.shape()[0];
                    let y = backend.zeros(vec![t, d])?;
                    backend.gemm_nt(features, fc, &y)?;
                    norm.forward(backend, &y.view(vec![1, t, d]))?
                }
                _ => features.clone(),
            };
            for (i, layer) in self.layers.iter().enumerate() {
                let kv_size = self.cfg.n_kv_heads * self.cfg.head_dim;
                let q_size = self.cfg.n_heads * self.cfg.head_dim;
                let qkv = crate::nn::Linear {
                    w: layer.dense.attn.wqkv.clone(),
                    b: None,
                }
                .forward(backend, &enc)?;
                let k = crate::nn::slice_last(backend, &qkv, kv_size, q_size)?;
                let v = crate::nn::slice_last(backend, &qkv, kv_size, q_size + kv_size)?;
                // k_norm (как в спеке: Kcur = k_norm(Kcur) перед rope)
                let k = if let Some(kw) = &layer.dense.attn.k_norm {
                    let y = backend.zeros(k.shape().to_vec())?;
                    backend.rmsnorm_head(&k, kw, &y, self.cfg.n_kv_heads, 1e-6)?;
                    y
                } else {
                    k
                };
                let k = crate::qwen3::rope_standard(
                    backend,
                    &k,
                    pos,
                    t,
                    self.cfg.head_dim,
                    self.cfg.rope_theta,
                )?;
                // кэш индексирован ПОЗИЦИЯМИ токенов (llama.cpp): инжект
                // пишет на абсолютные позиции pos..pos+t; перезапись
                // диффузионных копий при повторе.
                self.caches[i].update_at(backend, &k, &v, pos)?;
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
    /// `pos` — АБСОЛЮТНАЯ rope-позиция первого токена блока (позиция anchor
    /// в последовательности таргета; спека: inp_pos = позиции токенов).
    /// ВНИМАНИЕ: мутирует кэши (пишет k/v draft-токенов) — после шага
    /// вызывающий должен откатить кэши до инжектной длины.
    /// Диффузия token-режима (ОРИГИНАЛЬНЫЙ протокол z-lab modeling_dflash.py):
    /// mask-патч `ids` ([anchor, MASK...]) → dense-слои. Внимание — прямая
    /// инъекция фич (НЕ через кэш!): k_ctx/v_ctx = wkv(hidden_norm(fc(фичи)))
    /// поворачиваются ПОЗИЦИЯМИ НАЧАЛА БЛОКА [pos..pos+ctx) и дописываются в
    /// кэш ПОСЛЕ noise-строк (порядок k-строк не влияет — маски нет).
    /// Кэш: только noise-копии принятых токенов (как z-lab DynamicCache).
    /// `features` — фичи таргета [ctx, D] (конкат входов extract-слоёв
    /// последних ПРИНЯТЫХ токенов); None — без инъекции.
    /// Логиты — через lm_head ТАРГЕТА (общий), скрытые — свой out_norm.
    pub fn diffuse(
        &mut self,
        backend: &dyn Backend,
        ids: &[u32],
        lm_head: &Tensor,
        pos: usize,
        features: Option<&Tensor>,
    ) -> Result<DiffusionResult, BackendError> {
        backend.begin_batch()?;
        let r = (|| {
            let t = ids.len();
            let embed = self
                .target_embed
                .as_ref()
                .ok_or_else(|| {
                    BackendError::Shape(
                        "diffuse: не установлен target_embed — драфтер без общих эмбеддингов"
                            .into(),
                    )
                })?
                .clone();
            let mut x = backend.zeros(vec![1, t, self.cfg.dim])?;
            backend.embedding(ids, &embed, &mut x)?;
            // фичи → fc → hidden_norm (z-lab: target_hidden = hidden_norm(fc(...)))
            let enc = match (features, &self.fc, &self.enc_norm) {
                (Some(f), Some(fc), Some(norm)) => {
                    let d = fc.shape()[0];
                    let y = backend.zeros(vec![f.shape()[0], d])?;
                    backend.gemm_nt(f, fc, &y)?;
                    let enc = norm.forward(backend, &y.view(vec![1, f.shape()[0], d]))?;
                    if std::env::var("LLM_DBG_SPEC").is_ok() {
                        eprintln!(
                            "diffuse-enc: f={:?} fc={:?} y={:?} enc={:?}",
                            f.shape(),
                            fc.shape(),
                            y.shape(),
                            enc.shape()
                        );
                    }
                    Some(enc)
                }
                _ => None,
            };
            let ctx = enc.as_ref().map(|e| e.shape()[1]).unwrap_or(0);
            // кэш-индексы — непрерывно от past_len (длина кэша ДО диффузии;
            // z-lab: cache_position = position_ids от get_seq_length()); на
            // первом шаге past_len=0 — запись с нуля, БЕЗ дыр.
            let past_len = self.caches[0].pos;
            // z-lab: position_ids окна = [start−ctx .. start+q); k_ctx — при
            // ФАКТИЧЕСКИХ позициях принятых токенов [pos−ctx .. pos), k_noise
            // и q при [pos .. pos+t). Раньше k_ctx крутили на [pos..pos+ctx)
            // — сдвиг rope на ctx поворачивал инжект (только хвост маски
            // видел контекст) — приёмка деградировала.
            let kctx_pos = pos.saturating_sub(ctx);
            for (i, layer) in self.layers.iter().enumerate() {
                if std::env::var("LLM_DBG_SPEC").is_ok() && i == 0 {
                    eprintln!(
                        "diffuse: layer0 x={:?} enc={:?}",
                        x.shape(),
                        enc.as_ref().map(|e| e.shape().to_vec())
                    );
                }
                // ── attn-часть (свой путь: прямая инъекция k_ctx) ──
                let mut h = layer.dense.attn_norm.forward(backend, &x)?;
                let attn_dyn = match (&layer.attn_conv.proj, &layer.attn_conv.base) {
                    (Some(p), Some(_)) => Some(conv_dynamic(backend, p, &h)?),
                    _ => None,
                };
                if let (Some(d), Some(b)) = (&attn_dyn, &layer.attn_conv.base) {
                    h = conv_side(backend, &h, b, d, 0, self.block_size)?;
                }
                let attn = &layer.dense.attn;
                if std::env::var("LLM_DBG_SPEC").is_ok() && i == 0 {
                    eprintln!(
                        "diffuse: layer0 h={:?} wqkv={:?}",
                        h.shape(),
                        attn.wqkv.shape()
                    );
                }
                let q_size = attn.n_heads * attn.head_dim;
                let kv_size = attn.n_kv_heads * attn.head_dim;
                let qkv = crate::nn::Linear {
                    w: attn.wqkv.clone(),
                    b: None,
                }
                .forward(backend, &h)?;
                let q = crate::nn::slice_last(backend, &qkv, q_size, 0)?;
                let k = crate::nn::slice_last(backend, &qkv, kv_size, q_size)?;
                let v = crate::nn::slice_last(backend, &qkv, kv_size, q_size + kv_size)?;
                // fused qk_norm+rope (Ф8): noise-токены на [pos..pos+t)
                let q = match &attn.q_norm {
                    Some(w) => backend.rmsnorm_head_rope(
                        &q,
                        w,
                        pos,
                        attn.head_dim,
                        self.cfg.rope_theta,
                        attn.n_heads,
                        1e-6,
                    )?,
                    None => crate::qwen3::rope_standard(
                        backend,
                        &q,
                        pos,
                        t,
                        attn.head_dim,
                        self.cfg.rope_theta,
                    )?,
                };
                let k = match &attn.k_norm {
                    Some(w) => backend.rmsnorm_head_rope(
                        &k,
                        w,
                        pos,
                        attn.head_dim,
                        self.cfg.rope_theta,
                        attn.n_kv_heads,
                        1e-6,
                    )?,
                    None => crate::qwen3::rope_standard(
                        backend,
                        &k,
                        pos,
                        t,
                        attn.head_dim,
                        self.cfg.rope_theta,
                    )?,
                };
                // прямая инъекция фич (z-lab: concat([k_ctx, k_noise])) —
                // k_ctx на [pos..pos+ctx), noise следом — кэш непрерывен
                if let Some(e) = &enc {
                    let qkv2 = crate::nn::Linear {
                        w: attn.wqkv.clone(),
                        b: None,
                    }
                    .forward(backend, e)?;
                    let kc = crate::nn::slice_last(backend, &qkv2, kv_size, q_size)?;
                    let vc = crate::nn::slice_last(backend, &qkv2, kv_size, q_size + kv_size)?;
                    let kc = match &attn.k_norm {
                        Some(w) => backend.rmsnorm_head_rope(
                            &kc,
                            w,
                            kctx_pos,
                            attn.head_dim,
                            self.cfg.rope_theta,
                            attn.n_kv_heads,
                            1e-6,
                        )?,
                        None => crate::qwen3::rope_standard(
                            backend,
                            &kc,
                            kctx_pos,
                            ctx,
                            attn.head_dim,
                            self.cfg.rope_theta,
                        )?,
                    };
                    self.caches[i].update_at(backend, &kc, &vc, past_len)?;
                }
                self.caches[i].update_at(backend, &k, &v, past_len + ctx)?;
                let ck = self.caches[i].view();
                let cv = self.caches[i].view_v();
                let attn_out = backend.zeros(vec![1, t, q_size])?;
                backend.attn_ar(
                    &q,
                    &ck,
                    &cv,
                    None,
                    &attn_out,
                    attn.n_heads,
                    attn.n_kv_heads,
                    attn.head_dim,
                )?;
                let mut cur = crate::nn::Linear {
                    w: attn.wo.clone(),
                    b: None,
                }
                .forward(backend, &attn_out)?;
                if let (Some(d), Some(b)) = (&attn_dyn, &layer.attn_conv.base) {
                    cur = conv_side(backend, &cur, b, d, 1, self.block_size)?;
                }
                let ffn_inp = add(backend, &cur, &x)?;
                // ── ffn-часть ──
                let mut h = layer.dense.ffn_norm.forward(backend, &ffn_inp)?;
                let ffn_dyn = match (&layer.ffn_conv.proj, &layer.ffn_conv.base) {
                    (Some(p), Some(_)) => Some(conv_dynamic(backend, p, &h)?),
                    _ => None,
                };
                if let (Some(d), Some(b)) = (&ffn_dyn, &layer.ffn_conv.base) {
                    h = conv_side(backend, &h, b, d, 0, self.block_size)?;
                }
                let mut f = crate::qwen3::ffn_forward(backend, &h, &layer.dense.ffn)?;
                if let (Some(d), Some(b)) = (&ffn_dyn, &layer.ffn_conv.base) {
                    f = conv_side(backend, &f, b, d, 1, self.block_size)?;
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
    /// сохраняет только принятые токены). Раньше pos присваивался
    /// безусловно — inject_len БОЛЬШЕ текущего pos расширял кэш в
    /// незаписанную (нулевую) зону; truncate зажимает min-ом.
    pub fn truncate_caches(&mut self, inject_len: usize) {
        for c in &mut self.caches {
            c.truncate(inject_len);
        }
    }

    /// Селектор v2 (спека build_dflash2_selector): жадный путь через лэттис.
    /// logits [1, T, V] (позиция 0 — anchor, 1..T — MASK), hidden [1, T, D]
    /// (t_embd после out_norm). Возвращает draft-токены MASK-позиций и их
    /// честные q (softmax-вероятность по полному ряду logits).
    /// Лэттис-строка: [cand_ids(top_k), scores(top_k·top_k)]; score[k] для
    /// pred: Σ_r succ[r, cand[k]] · (prev[r, pred] · hidden[r, pos]) + unary[k].
    pub fn select(
        &self,
        backend: &dyn Backend,
        logits: &Tensor,
        hidden: &Tensor,
    ) -> Result<SelectResult, BackendError> {
        let sel = self
            .selector
            .as_ref()
            .ok_or_else(|| BackendError::Shape("select: у драфтера нет селектора (v1)".into()))?;
        let (t, vocab) = (logits.shape()[1], logits.shape()[2]);
        let (rank, top_k) = (sel.rank, sel.top_k.min(vocab));
        let lg = backend.to_cpu_f32(logits)?; // [T, V]
        let hd = backend.to_cpu_f32(hidden)?; // [T, D]
        // ggml-порядок: prev/next [vocab, rank], hidden [rank, D]
        let pd = backend.to_cpu_f32(&sel.predecessor)?; // [V, rank]
        let sd = backend.to_cpu_f32(&sel.successor)?;
        let hw = backend.to_cpu_f32(&sel.hidden)?; // [rank, D]
        // top-k кандидаты и unary-логиты по каждой позиции
        let mut cand = vec![vec![0u32; top_k]; t];
        let mut unary = vec![vec![0f32; top_k]; t];
        for pos in 0..t {
            let row = &lg[pos * vocab..(pos + 1) * vocab];
            let mut idx: Vec<usize> = (0..vocab).collect();
            idx.sort_by(|&a, &b| {
                row[b]
                    .partial_cmp(&row[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for k in 0..top_k {
                cand[pos][k] = idx[k] as u32;
                unary[pos][k] = row[idx[k]];
            }
        }
        // скрытые → ранг: hid[pos][r] = Σ_d hidden[pos,d]·hw[r,d]
        let mut hid = vec![vec![0f32; rank]; t];
        for pos in 0..t {
            for r in 0..rank {
                let mut acc = 0.0f32;
                for d in 0..self.cfg.dim {
                    acc += hd[pos * self.cfg.dim + d] * hw[r * self.cfg.dim + d];
                }
                hid[pos][r] = acc;
            }
        }
        // жадный путь: на позиции 1 предшественник — anchor (позиция 0),
        // на позициях i>1 — выбранный кандидат i−1.
        let mut tokens = Vec::with_capacity(t - 1);
        let mut probs = Vec::with_capacity(t - 1);
        let mut pred = cand[0][0];
        for pos in 1..t {
            let pred_col: Vec<f32> = (0..rank).map(|r| pd[pred as usize * rank + r]).collect();
            let mut best = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            for k in 0..top_k {
                let cid = cand[pos][k] as usize;
                let mut s = unary[pos][k];
                for r in 0..rank {
                    s += sd[cid * rank + r] * (pred_col[r] * hid[pos][r]);
                }
                if s > best_score {
                    best_score = s;
                    best = k;
                }
            }
            tokens.push(cand[pos][best]);
            probs.push(softmax_row_prob(&lg, pos, vocab, cand[pos][best]));
            pred = cand[pos][best];
        }
        Ok(SelectResult { tokens, probs })
    }
}

/// Результат селектора v2: draft-токены MASK-позиций + их q.
pub struct SelectResult {
    pub tokens: Vec<u32>,
    pub probs: Vec<f32>,
}

/// softmax-вероятность токена в ряду логгитов (честная q драфтера).
fn softmax_row_prob(lg: &[f32], pos: usize, vocab: usize, tok: u32) -> f32 {
    let row = &lg[pos * vocab..(pos + 1) * vocab];
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for &l in row {
        sum += (l - max).exp();
    }
    ((row[tok as usize] - max).exp()) / sum
}

/// DFlash-драфтер (Ф2.5): диффузия блока [anchor, MASK×…] → селектор v2
/// (или top-1 для v1) → γ draft-токенов с честными q. Кэш на каждый вызов
/// сбрасывается (инжект — Ф2.4: inject_features перед draft).
pub struct DFlashDrafter {
    pub model: DFlashModel,
    pub target_embed: Tensor,
    pub lm_head: Tensor,
    /// Фичи таргета для прямой инъекции (z-lab: k_ctx = wkv(hidden_norm(fc(features)))).
    /// Ставятся вызывающим перед draft (последние ПРИНЯТЫЕ токены).
    pub features: Option<Tensor>,
    /// Кольцо принятых токенов (последние block_size).
    pub committed: std::collections::VecDeque<u32>,
    /// Сид сэмплинга (v1-диффузия: позиции блока симметричны без conv —
    /// top-1 вырождается; спека: "sample every block position").
    pub rng: u64,
}

impl DFlashDrafter {
    pub fn new(mut model: DFlashModel, target_embed: Tensor, lm_head: Tensor) -> Self {
        model.set_target_embed(target_embed.clone());
        Self {
            model,
            target_embed,
            lm_head,
            features: None,
            committed: std::collections::VecDeque::new(),
            rng: 0x85ebca6b,
        }
    }

    /// Сэмпл из softmax-распределения строки (LCG, как sample_modified).
    /// Сэмпл из softmax-распределения строки (LCG, как sample_modified).
    /// LLM_TOP_K=N — сэмпл только из топ-N (эксперимент: совпадение с
    /// posterior чаще, если распределения совпадают на топах).
    fn sample_row(&mut self, row: &[f32]) -> u32 {
        let top_k = std::env::var("LLM_TOP_K")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut w: Vec<f32> = row.iter().map(|&l| (l - max).exp()).collect();
        if top_k > 0 && top_k < w.len() {
            // оставить топ-N по весу, остальные — 0
            let mut idx: Vec<usize> = (0..w.len()).collect();
            // N (iter-9): a NaN logit made partial_cmp return None → panic.
            idx.sort_by(|&a, &b| w[b].partial_cmp(&w[a]).unwrap_or(std::cmp::Ordering::Equal));
            for &i in &idx[top_k..] {
                w[i] = 0.0;
            }
        }
        let total: f32 = w.iter().sum();
        self.rng = self
            .rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut r = (self.rng >> 33) as f32 / (1u64 << 33) as f32 * total;
        for (i, v) in w.iter_mut().enumerate() {
            r -= *v;
            if r <= 0.0 {
                return i as u32;
            }
        }
        (w.len() - 1) as u32
    }

    /// Дописать принятые таргетом токены в committed-кольцо.
    pub fn commit(&mut self, tokens: &[u32]) {
        let cap = self.model.block_size;
        for &t in tokens {
            self.committed.push_back(t);
            while self.committed.len() > cap {
                self.committed.pop_front();
            }
        }
    }
}

impl crate::spec::Drafter for DFlashDrafter {
    fn target_layers(&self) -> &[usize] {
        &self.model.extract_layers
    }

    fn after_extract(
        &mut self,
        _backend: &dyn Backend,
        extract: &[Tensor],
        pos: usize,
        n_accepted: usize,
    ) -> Result<(), BackendError> {
        // z-lab (dflash-ref model.py): context = ПРИНЯТЫЕ строки верификации —
        // target_hidden = extract(...)[:, :produced, :]; конкат слоёв по
        // последней оси. Принятые = якорь + n_accepted-1 проверенных.
        if extract.is_empty() || self.model.extract_layers.is_empty() {
            return Ok(());
        }
        if std::env::var("LLM_DBG_SPEC").is_ok() {
            eprintln!(
                "after_extract: n={} shapes={:?} accepted={}",
                extract.len(),
                extract
                    .iter()
                    .map(|t| t.shape().to_vec())
                    .collect::<Vec<_>>(),
                n_accepted
            );
        }
        // extract-тензоры — [1, T, D] (batch=1): строки 0..n_accepted —
        // принятые токены; их позиции [pos+1-n_accepted .. pos+1).
        // ВАЖНО (4.2c): z-lab конкат по ПОСЛЕДНЕЙ оси (dim=-1) — фичи
        // строки t = [l0[t]; l1[t]; …; l4[t]] (токен-мажор!). Раньше
        // клали слой-мажор ([L0-все строки, L1-все…]) — fc получал
        // перепутанный вход и драфтер выдавал мусор (приёмка 0%).
        let d = extract[0].shape()[2];
        let t = extract[0].shape()[1];
        let rows = n_accepted.min(t).min(t);
        let n_layers = extract.len();
        let mut feat = vec![0f32; n_layers * rows * d];
        for (li, t_) in extract.iter().enumerate() {
            let td = t_.shape()[1];
            let rows_ = _backend.to_cpu_f32(t_)?;
            let n = rows.min(td);
            for r in 0..n {
                let src = r * d;
                let dst = r * n_layers * d + li * d;
                feat[dst..dst + d].copy_from_slice(&rows_[src..src + d]);
            }
        }
        let features = _backend.from_cpu_f32(&feat, vec![rows, extract.len() * d])?;
        self.features = Some(features);
        let _ = pos;
        Ok(())
    }

    fn draft(
        &mut self,
        backend: &dyn Backend,
        history: &[u32],
        gamma: usize,
    ) -> (Vec<u32>, Vec<f32>) {
        if gamma == 0 {
            return (vec![], vec![]);
        }
        let anchor = history
            .last()
            .copied()
            .or_else(|| self.committed.back().copied());
        let anchor = match anchor {
            Some(a) => a,
            None => return (vec![], vec![]), // нет контекста — таргет сам декодит
        };
        // блок: [anchor, MASK × (block−1)]; MASK-позиции диффундируются.
        // rope-позиции — по спеке (draft_dflash): batch.pos = n + i — блок
        // идёт на позициях [n..n+block). n = АБСОЛЮТНАЯ позиция якоря
        // (history.len()−1): target-verify (decode_spec) обрабатывает те же
        // строки на тех же позициях. Раньше pos = history.len() были все
        // строки блока сдвинуты на 1 — row0 (якорь) крутился на позиции
        // num+1 вместо num (у неё и verify).
        let block = self.model.block_size;
        let pos = history.len().saturating_sub(1);
        let mut ids = vec![anchor];
        ids.resize(block, self.model.mask_token);
        // кэш содержит только noise-копии [0..pos); после диффузии —
        // откат до pos (z-lab: past_key_values_draft.crop(start))
        if std::env::var("LLM_DBG_SPEC").is_ok() {
            eprintln!(
                "DFlashDrafter: cache={} pos={pos} block={block} ctx={:?}",
                self.model.caches[0].pos,
                self.features.as_ref().map(|f| f.shape()[0])
            );
        }
        if std::env::var("LLM_DBG_SPEC").is_ok() {
            eprintln!(
                "draft: ids={} block={} pos={pos} features={:?}",
                ids.len(),
                block,
                self.features.as_ref().map(|f| f.shape().to_vec())
            );
        }
        let res =
            match self
                .model
                .diffuse(backend, &ids, &self.lm_head, pos, self.features.as_ref())
            {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("DFlashDrafter: диффузия не удалась: {e}");
                    return (vec![], vec![]);
                }
            };
        let (tokens, probs) = if self.model.selector.is_some() {
            match self.model.select(backend, &res.logits, &res.hidden) {
                Ok(s) => (s.tokens, s.probs),
                Err(e) => {
                    eprintln!("DFlashDrafter: селектор не удался: {e}");
                    return (vec![], vec![]);
                }
            }
        } else {
            // v1: сэмпл каждой MASK-позиции (спека: "sample every block
            // position on the backend") + честные q (softmax-вероятность).
            // z-lab model.py: при temperature=0 черновики = ARGMAX строк
            // (dflash_generate: block_output_ids[:, 1:] = argmax(draft_logits));
            // случайный LCG-сэмпл из полного ряда даёт q≈0 и приёмку 0%
            // независимо от качества драфтера. LLM_DRAFT_ARGMX=1 включает
            // argmax-режим (жёсткая генерация бенча).
            let argmx = std::env::var("LLM_DRAFT_ARGMX")
                .map(|v| v == "1")
                .unwrap_or(false);
            let (t, vocab) = (res.logits.shape()[1], res.logits.shape()[2]);
            let lg = match backend.to_cpu_f32(&res.logits) {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("DFlashDrafter: logits недоступны: {e}");
                    return (vec![], vec![]);
                }
            };
            let mut tokens = Vec::with_capacity(t - 1);
            let mut probs = Vec::with_capacity(t - 1);
            // LM-конвенция: row i предсказывает токен i+1, якорь на позиции 0,
            // черновики = строки 1..t (проверено: direct-вариант хуже — 0.7%).
            for pos in 1..t {
                let row = &lg[pos * vocab..(pos + 1) * vocab];
                let tok = if argmx {
                    let mut best = (0usize, f32::NEG_INFINITY);
                    for (i, &l) in row.iter().enumerate() {
                        if l.is_finite() && l > best.1 {
                            best = (i, l);
                        }
                    }
                    best.0 as u32
                } else {
                    self.sample_row(row)
                };
                tokens.push(tok);
                probs.push(softmax_row_prob(&lg, pos, vocab, tok));
            }
            (tokens, probs)
        };
        // кэш после диффузии содержит k/v блока — откат до pos (crop)
        self.model.truncate_caches(pos);
        self.commit(&tokens);
        (
            tokens.into_iter().take(gamma).collect(),
            probs.into_iter().take(gamma).collect(),
        )
    }
}

/// Результат диффузии: логиты (lm_head таргета) + скрытые (свой out_norm).
pub struct DiffusionResult {
    pub logits: Tensor, // [1, T, V]
    pub hidden: Tensor, // [1, T, D]
}

/// conv-проекция шума: dynamic = x @ proj^T — [n_tokens, n_groups·k·2]
/// (ggml [proj, T] column-major = row-major [T, proj]; coeff-строка
/// r = g + n_groups·(tap + k·side) — колонка).
fn conv_dynamic(backend: &dyn Backend, proj: &Tensor, x: &Tensor) -> Result<Tensor, BackendError> {
    let (b, t, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    let x2 = x.view(vec![b * t, d]);
    let y = backend.zeros(vec![b * t, proj.shape()[0]])?;
    backend.gemm_nt(&x2, proj, &y)?;
    Ok(y)
}

/// Один tap-конв (side: 0 = вход, 1 = выход) по формуле спеки.
/// dynamic [n_tokens, n_groups·k·2] (row-major, из conv_dynamic).
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
    // N (iter-9): degenerate conv metadata (kernel 0 / groups 0) used to
    // div-by-zero — clean error instead.
    if kernel == 0 {
        return Err(BackendError::Shape(
            "dflash conv: kernel dimension is 0".into(),
        ));
    }
    let n_groups = dynamic.shape()[1] / (kernel * 2);
    if n_groups == 0 {
        return Err(BackendError::Shape(
            "dflash conv: dynamic width smaller than kernel*2 groups".into(),
        ));
    }
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
            WeightFormat::Q4_0 | WeightFormat::Q8_0 | WeightFormat::Q4_K | WeightFormat::Q6_K => {
                backend.from_quantized(tensor.raw(), shape, tensor.format)
            }
            _ => {
                let f = dequant_f32(tensor);
                upload_f16(backend, &f, shape)
            }
        }
    };
    // q/k/v fused (как Qwen3Model::load): wqkv [q_size + 2·kv_size, dim]
    let mut wqkv = dequant_f32(
        w.get(&p("attn_q.weight"))
            .ok_or_else(|| BackendError::Shape("attn_q.weight missing".into()))?,
    );
    wqkv.extend(dequant_f32(w.get(&p("attn_k.weight")).ok_or_else(
        || BackendError::Shape("attn_k.weight missing".into()),
    )?));
    wqkv.extend(dequant_f32(w.get(&p("attn_v.weight")).ok_or_else(
        || BackendError::Shape("attn_v.weight missing".into()),
    )?));
    let attn = Attention {
        wqkv: upload_f16(
            backend,
            &wqkv,
            vec![
                cfg.n_heads * cfg.head_dim + 2 * cfg.n_kv_heads * cfg.head_dim,
                cfg.dim,
            ],
        )?,
        wo: up(&p("attn_output.weight"), vec![cfg.dim, cfg.dim])?,
        qkv_bias: None,
        q_norm: Some(
            backend.from_cpu_f32(
                &dequant_f32(
                    w.get(&p("attn_q_norm.weight"))
                        .ok_or_else(|| BackendError::Shape("attn_q_norm.weight missing".into()))?,
                ),
                vec![cfg.head_dim],
            )?,
        ),
        k_norm: Some(
            backend.from_cpu_f32(
                &dequant_f32(
                    w.get(&p("attn_k_norm.weight"))
                        .ok_or_else(|| BackendError::Shape("attn_k_norm.weight missing".into()))?,
                ),
                vec![cfg.head_dim],
            )?,
        ),
        n_heads: cfg.n_heads,
        n_kv_heads: cfg.n_kv_heads,
        head_dim: cfg.head_dim,
    };
    // gate/up fused (как Qwen3Model::load): w13 [2·ffn, dim]
    let mut w13 = dequant_f32(
        w.get(&p("ffn_gate.weight"))
            .ok_or_else(|| BackendError::Shape("ffn_gate.weight missing".into()))?,
    );
    w13.extend(dequant_f32(w.get(&p("ffn_up.weight")).ok_or_else(
        || BackendError::Shape("ffn_up.weight missing".into()),
    )?));
    let ffn = SwiGLU {
        w13: upload_f16(backend, &w13, vec![2 * cfg.ffn_dim, cfg.dim])?,
        w2: up(&p("ffn_down.weight"), vec![cfg.dim, cfg.ffn_dim])?,
    };
    Ok(Qwen3Layer {
        attn_norm: RmsNorm::from_weights(
            backend,
            dequant_f32(
                w.get(&p("attn_norm.weight"))
                    .ok_or_else(|| BackendError::Shape("attn_norm.weight missing".into()))?,
            ),
            cfg.norm_eps,
        )?,
        attn,
        ffn_norm: RmsNorm::from_weights(
            backend,
            dequant_f32(
                w.get(&p("ffn_norm.weight"))
                    .ok_or_else(|| BackendError::Shape("ffn_norm.weight missing".into()))?,
            ),
            cfg.norm_eps,
        )?,
        ffn,
    })
}

/// f16-загрузка весов (как upload_w в qwen3.rs).
fn upload_f16(
    backend: &dyn Backend,
    data: &[f32],
    shape: Vec<usize>,
) -> Result<Tensor, BackendError> {
    if backend.supports_f16_weights() {
        let f16: Vec<half::f16> = data.iter().map(|&x| half::f16::from_f32(x)).collect();
        backend.from_cpu_f16(&f16, shape)
    } else {
        backend.from_cpu_f32(data, shape)
    }
}

/// N1 (build fix): the tests need both the Metal backend AND the CPU harness —
/// bare `#[cfg(test)]` broke `cargo test` on Linux (E0432).
#[cfg(all(test, feature = "metal"))]
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
                ("attn_conv_base", vec![2, 3, c.dim]),
                ("attn_conv_proj", vec![(c.dim / 4) * 3 * 2, c.dim]),
                ("ffn_conv_base", vec![2, 3, c.dim]),
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
                let mut v = vec![0f32; c.vocab_size * 4];
                rng.fill(&mut v);
                v
            },
            vec![c.vocab_size, 4],
        ));
        st.push((
            "selector_successor".into(),
            {
                let mut v = vec![0f32; c.vocab_size * 4];
                rng.fill(&mut v);
                v
            },
            vec![c.vocab_size, 4],
        ));
        st.push((
            "selector_hidden".into(),
            {
                let mut v = vec![0f32; 4 * c.dim];
                rng.fill(&mut v);
                v
            },
            vec![4, c.dim],
        ));
        st.push((
            "fc.weight".into(),
            {
                let mut v = vec![0f32; c.dim * c.dim];
                rng.fill(&mut v);
                v
            },
            vec![c.dim, c.dim],
        ));
        st.push((
            "enc.output_norm.weight".into(),
            {
                let mut v = vec![0f32; c.dim];
                rng.fill(&mut v);
                v
            },
            vec![c.dim],
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
    #[cfg(feature = "cpu")]
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
        let m = DFlashModel::load(&backend, &wm, c.clone(), 16, 3, 4, vec![0], 151850).unwrap();
        assert_eq!(m.layers.len(), 2);
        assert!(m.selector.is_some());
        assert_eq!(m.cfg.dim, 16);
        assert_eq!(m.conv_kernel, 3);
        assert_eq!(m.mask_token, 151850);
        // conv-тензоры загружены (f16 — проверяем shape; base в ggml-порядке)
        assert_eq!(
            m.layers[0].attn_conv.base.as_ref().unwrap().shape(),
            &[2, 3, 16]
        );
        assert_eq!(
            m.layers[0].attn_conv.proj.as_ref().unwrap().shape(),
            &[24, 16]
        );
        // селектор: [vocab, rank] кодбуки + hidden [rank, dim]
        let s = m.selector.as_ref().unwrap();
        assert_eq!(s.predecessor.shape(), &[8, 4]);
        assert_eq!(s.successor.shape(), &[8, 4]);
        assert_eq!(s.hidden.shape(), &[4, 16]);
        assert_eq!(s.rank, 4);
        assert_eq!(s.top_k, 10);
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
        let proj = n_groups * kernel * 2; // 24
        let mut rng = Lcg::new(7);
        let mut xd = vec![0f32; tokens * dim];
        let mut bd = vec![0f32; dim * kernel * 2]; // файл {n_embd, k, 2} → row-major [2, k, D]
        let mut dd = vec![0f32; tokens * proj]; // ggml {proj, T} → row-major [T, proj]
        rng.fill(&mut xd);
        rng.fill(&mut bd);
        rng.fill(&mut dd);
        let xt = backend.from_cpu_f32(&xd, vec![tokens, dim]).unwrap();
        let bt = backend.from_cpu_f32(&bd, vec![2, kernel, dim]).unwrap();
        let dt = backend.from_cpu_f32(&dd, vec![tokens, proj]).unwrap();
        let (xc, bc, dc) = (
            cpu.from_cpu_f32(&xd, vec![tokens, dim]).unwrap(),
            cpu.from_cpu_f32(&bd, vec![2, kernel, dim]).unwrap(),
            cpu.from_cpu_f32(&dd, vec![tokens, proj]).unwrap(),
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
        let mut dflash =
            DFlashModel::load(&backend, &wm, c.clone(), 16, 3, 4, vec![0], 151850).unwrap();
        assert!(dflash.layers[0].attn_conv.proj.is_none());
        // общие эмбеддинги (те же данные, что у qwen)
        let ew = dequant_f32(wm.get("token_embd.weight").unwrap());
        let embed = Embedding::from_weights(&backend, ew, vec![c.vocab_size, c.dim]).unwrap();
        dflash.set_target_embed(embed.table.clone());
        // lm_head таргета (f16, как Qwen3Model::load)
        let ow = dequant_f32(wm.get("output.weight").unwrap());
        let f16: Vec<half::f16> = ow.iter().map(|&x| half::f16::from_f32(x)).collect();
        let lm_head = backend
            .from_cpu_f16(&f16, vec![c.vocab_size, c.dim])
            .unwrap();

        // t=1: каузальная маска не строится ни в одном пути — пути тождественны
        let ids = [3u32];
        let h = qwen.forward_hidden(&backend, &ids).unwrap();
        let d = dflash.diffuse(&backend, &ids, &lm_head, 0, None).unwrap();
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
        let d2 = dflash.diffuse(&backend, &ids2, &lm_head, 0, None).unwrap();
        let lg = backend.to_cpu_f32(&d2.logits).unwrap();
        assert!(lg.iter().all(|v| v.is_finite()));
        dflash.truncate_caches(0);
    }

    /// Селектор v2: жадный путь через лэттис против независимого
    /// CPU-референса (та же формула спеки build_dflash2_selector).
    #[test]
    fn selector_v2_matches_cpu_reference() {
        use crate::harness::Lcg;
        let backend = MetalBackend::new().unwrap();
        let c = cfg();
        let st = storage(&c);
        let mut bytes: Vec<u8> = Vec::new();
        let mut layout: Vec<(String, Vec<u64>, u64, usize)> = Vec::new();
        for (name, data, shape) in &st {
            if name.ends_with("_conv_proj") {
                continue;
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
        let m = DFlashModel::load(&backend, &wm, c.clone(), 8, 3, 4, vec![0], 151850).unwrap();
        let sel = m.selector.as_ref().unwrap();
        assert_eq!((sel.rank, sel.top_k), (4, 10));
        // синтетические logits [1, T=4, V=8] и hidden [1, 4, D=16]
        let mut rng = Lcg::new(9);
        let mut lg = vec![0f32; 4 * 8];
        let mut hd = vec![0f32; 4 * 16];
        rng.fill(&mut lg);
        rng.fill(&mut hd);
        let lt = backend.from_cpu_f32(&lg, vec![1, 4, 8]).unwrap();
        let ht = backend.from_cpu_f32(&hd, vec![1, 4, 16]).unwrap();
        let res = m.select(&backend, &lt, &ht).unwrap();
        assert_eq!(res.tokens.len(), 3);
        assert_eq!(res.probs.len(), 3);
        // ── CPU-референс (независимый, по спеке, ggml-порядок) ──
        let pd = backend.to_cpu_f32(&sel.predecessor).unwrap(); // [V, 4]
        let sd = backend.to_cpu_f32(&sel.successor).unwrap();
        let hw = backend.to_cpu_f32(&sel.hidden).unwrap(); // [4, 16]
        let (t, vocab, rank) = (4usize, 8usize, sel.rank);
        let top_k = sel.top_k.min(vocab);
        let mut cand = vec![vec![0u32; top_k]; t];
        let mut unary = vec![vec![0f32; top_k]; t];
        for pos in 0..t {
            let row = &lg[pos * vocab..(pos + 1) * vocab];
            let mut idx: Vec<usize> = (0..vocab).collect();
            idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
            for k in 0..top_k {
                cand[pos][k] = idx[k] as u32;
                unary[pos][k] = row[idx[k]];
            }
        }
        let mut hid = vec![vec![0f32; rank]; t];
        for pos in 0..t {
            for r in 0..rank {
                hid[pos][r] = (0..16).map(|d| hd[pos * 16 + d] * hw[r * 16 + d]).sum();
            }
        }
        let mut exp_tok = Vec::new();
        let mut pred = cand[0][0];
        for pos in 1..t {
            let mut best = 0usize;
            let mut bs = f32::NEG_INFINITY;
            for k in 0..top_k {
                let cid = cand[pos][k] as usize;
                let mut s = unary[pos][k];
                for r in 0..rank {
                    s += sd[cid * rank + r] * (pd[pred as usize * rank + r] * hid[pos][r]);
                }
                if s > bs {
                    bs = s;
                    best = k;
                }
            }
            exp_tok.push(cand[pos][best]);
            pred = cand[pos][best];
        }
        assert_eq!(res.tokens, exp_tok, "селектор: жадный путь не совпал");
        // q — вероятности в (0, 1]
        for &q in &res.probs {
            assert!(q > 0.0 && q <= 1.0, "q={q} вне (0,1]");
        }
        let qmax = res.probs[0];
        assert!((qmax - 1.0).abs() <= 1e-5 || qmax < 1.0);
    }

    /// DFlashDrafter: блок [anchor, MASK×…] → диффузия → селектор/топ-1;
    /// γ токенов, честные q, кэш откатан после вызова.
    #[test]
    fn dflash_drafter_drafts_block() {
        use crate::nn::Embedding;
        let backend = MetalBackend::new().unwrap();
        let c = cfg();
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
        let m = DFlashModel::load(&backend, &wm, c.clone(), 8, 3, 4, vec![0], 5).unwrap();
        let ew = dequant_f32(wm.get("token_embd.weight").unwrap());
        let embed = Embedding::from_weights(&backend, ew, vec![c.vocab_size, c.dim]).unwrap();
        let ow = dequant_f32(wm.get("output.weight").unwrap());
        let f16: Vec<half::f16> = ow.iter().map(|&x| half::f16::from_f32(x)).collect();
        let lm_head = backend
            .from_cpu_f16(&f16, vec![c.vocab_size, c.dim])
            .unwrap();
        let mut drafter = DFlashDrafter::new(m, embed.table.clone(), lm_head);
        // без истории — пустой draft (таргет декодит сам)
        let (t0, q0) = crate::spec::Drafter::draft(&mut drafter, &backend, &[], 4);
        assert!(t0.is_empty() && q0.is_empty());
        // с anchor — блок диффундируется, γ токенов с честными q
        let (toks, probs) = crate::spec::Drafter::draft(&mut drafter, &backend, &[3], 4);
        assert!(!toks.is_empty());
        assert!(toks.len() <= 4);
        assert_eq!(toks.len(), probs.len());
        for &q in &probs {
            assert!(q > 0.0 && q <= 1.0, "q={q} вне (0,1]");
        }
        for &tok in &toks {
            assert!(tok < 8, "draft-токен вне вокабуляра");
        }
        // committed-кольцо пополнено, кэш откатан
        assert!(drafter.committed.len() >= toks.len());
        assert!(drafter.model.caches[0].pos <= 8);
        // ещё один шаг — блок от нового anchor
        let (toks2, _) = crate::spec::Drafter::draft(&mut drafter, &backend, &[3], 2);
        assert!(toks2.len() <= 2);
    }

    /// Инжект фич таргета реально меняет логиты диффузии (кэш учитывается).
    #[test]
    fn inject_changes_diffusion() {
        use crate::harness::Lcg;
        use crate::nn::Embedding;
        let backend = MetalBackend::new().unwrap();
        let c = cfg();
        let st = storage(&c);
        let mut bytes: Vec<u8> = Vec::new();
        let mut layout: Vec<(String, Vec<u64>, u64, usize)> = Vec::new();
        for (name, data, shape) in &st {
            if name.ends_with("_conv_proj") {
                continue; // conv выключен — чистая диффузия
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
        let mut m = DFlashModel::load(&backend, &wm, c.clone(), 8, 3, 4, vec![0], 5).unwrap();
        let ew = dequant_f32(wm.get("token_embd.weight").unwrap());
        let embed = Embedding::from_weights(&backend, ew, vec![c.vocab_size, c.dim]).unwrap();
        m.set_target_embed(embed.table.clone());
        let ow = dequant_f32(wm.get("output.weight").unwrap());
        let f16: Vec<half::f16> = ow.iter().map(|&x| half::f16::from_f32(x)).collect();
        let lm_head = backend
            .from_cpu_f16(&f16, vec![c.vocab_size, c.dim])
            .unwrap();
        let ids = [3u32, 5, 5, 5];
        // без инжекта
        let r0 = m.diffuse(&backend, &ids, &lm_head, 3, None).unwrap();
        let lg0 = backend.to_cpu_f32(&r0.logits).unwrap();
        // с инжектом: случайные фичи [2, 20480→dim·2] — у мини-модели fc нет,
        // фичи идут напрямую в wkv (размерность dim)
        let mut rng = Lcg::new(11);
        let mut fd = vec![0f32; 2 * c.dim];
        rng.fill(&mut fd);
        let ft = backend.from_cpu_f32(&fd, vec![2, c.dim]).unwrap();
        let r1 = m.diffuse(&backend, &ids, &lm_head, 3, Some(&ft)).unwrap();
        let lg1 = backend.to_cpu_f32(&r1.logits).unwrap();
        assert_ne!(lg0, lg1, "инжект не влияет на логиты диффузии");
        m.truncate_caches(0);
    }

    /// Диагностика: диффузия различает позиции (маска-патч, без инжекта).
    /// Не топ-1-равенство, а именно РАЗЛИЧИЕ логитов по позициям.
    #[test]
    #[cfg(feature = "cpu")]
    fn diffuse_positions_differ() {
        use crate::nn::Embedding;
        let backend = MetalBackend::new().unwrap();
        let c = cfg();
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
        let mut m = DFlashModel::load(&backend, &wm, c.clone(), 8, 3, 4, vec![0], 151850).unwrap();
        let ew = dequant_f32(wm.get("token_embd.weight").unwrap());
        let embed = Embedding::from_weights(&backend, ew, vec![c.vocab_size, c.dim]).unwrap();
        m.set_target_embed(embed.table.clone());
        let ow = dequant_f32(wm.get("output.weight").unwrap());
        let f16: Vec<half::f16> = ow.iter().map(|&x| half::f16::from_f32(x)).collect();
        let lm_head = backend
            .from_cpu_f16(&f16, vec![c.vocab_size, c.dim])
            .unwrap();
        let ids = [3u32, 5, 5, 5];
        let r = m.diffuse(&backend, &ids, &lm_head, 0, None).unwrap();
        let lg = backend.to_cpu_f32(&r.logits).unwrap();
        let v = c.vocab_size;
        let mut tops = Vec::new();
        for pos in 0..4 {
            let row = &lg[pos * v..(pos + 1) * v];
            let best = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            tops.push(best);
        }
        // позиции должны различаться (иначе диффузия вырождена). На
        // random-весах топ-1 логитов доминируется строкой lm_head — проверяем
        // различие СКРЫТЫХ (rope различает позиции до проекции).
        let hv = backend.to_cpu_f32(&r.hidden).unwrap();
        let mut rows_eq = true;
        for pos in 1..4 {
            if hv[pos * c.dim..(pos + 1) * c.dim] != hv[..c.dim] {
                rows_eq = false;
            }
        }
        assert!(
            !rows_eq,
            "Metal: hidden-строки позиций идентичны — диффузия вырождена"
        );
        // то же на CPU-бэкенде (отделяет логику от Metal-кернелов)
        let cpu = crate::cpu::CpuBackend::default();
        let wm_c = crate::weights_map::WeightsMap {
            arch: "dflash".into(),
            tensors: wm.tensors.clone(),
            storage: wm.storage.clone(),
        };
        let mut mc = DFlashModel::load(&cpu, &wm_c, c.clone(), 8, 3, 4, vec![0], 151850).unwrap();
        let ew_c = dequant_f32(wm_c.get("token_embd.weight").unwrap());
        let embed_c = Embedding::from_weights(&cpu, ew_c, vec![c.vocab_size, c.dim]).unwrap();
        mc.set_target_embed(embed_c.table.clone());
        let ow_c = dequant_f32(wm_c.get("output.weight").unwrap());
        let lm_c = cpu.from_cpu_f32(&ow_c, vec![c.vocab_size, c.dim]).unwrap();
        // эталон: обычный qwen3-префилл на тех же токенах различает позиции?
        let mut qw = crate::qwen3::Qwen3Model::load(&cpu, &wm_c, c.clone()).unwrap();
        let qh = qw.forward_hidden(&cpu, &ids).unwrap();
        let qhv = cpu.to_cpu_f32(&qh).unwrap();
        let mut tops_q = Vec::new();
        for pos in 0..4 {
            let row = &qhv[pos * c.dim..(pos + 1) * c.dim];
            let best = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            tops_q.push(best);
        }
        eprintln!("qwen-prefill tops: {tops_q:?}");
        let rc = mc.diffuse(&cpu, &ids, &lm_c, 0, None).unwrap();
        let lgc = cpu.to_cpu_f32(&rc.logits).unwrap();
        let mut tops_c = Vec::new();
        for pos in 0..4 {
            let row = &lgc[pos * v..(pos + 1) * v];
            let best = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            tops_c.push(best);
        }
        // CPU: проверяем различие hidden-строк (top-1 логитов доминируется lm_head)
        let hvc = cpu.to_cpu_f32(&rc.hidden).unwrap();
        let mut rows_eq_c = true;
        for pos in 1..4 {
            if hvc[pos * c.dim..(pos + 1) * c.dim] != hvc[..c.dim] {
                rows_eq_c = false;
            }
        }
        assert!(
            !rows_eq_c,
            "CPU: hidden-строки позиций идентичны — диффузия вырождена"
        );
    }
}

/// Загрузить DFlash-драфтер (паттерн dflash_probe): веса модели,
/// target_embed и lm_head — от таргета. `.safetensors` — оригинал z-lab
/// (config.json рядом); `.gguf` — сконвертированный квант (ВНИМАНИЕ: файлы
/// Q8_0.gguf в ~/models/dflash старше zlab-версии — другие веса, приёмка
/// на них не показательна). Ошибка — понятная (нет/не GGUF).
pub fn load_drafter_gguf(
    backend: &dyn crate::backend::Backend,
    path: &std::path::Path,
    target_embed: Tensor,
    lm_head: Tensor,
    vocab_size: usize,
) -> Result<DFlashDrafter, BackendError> {
    let model = if path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e == "safetensors")
        .unwrap_or(false)
    {
        DFlashModel::load_safetensors(backend, path, vocab_size)?
    } else {
        let file = std::fs::File::open(path)
            .map_err(|e| BackendError::Shape(format!("open drafter {}: {e}", path.display())))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| BackendError::Shape(format!("mmap drafter: {e}")))?;
        let gf =
            crate::gguf::parse(&mmap).map_err(|e| BackendError::Shape(format!("gguf parse: {e}")))?;
        let wm = crate::weights_map::WeightsMap::from_gguf(&gf)
            .map_err(|e| BackendError::Shape(format!("weights: {e}")))?;
        DFlashModel::load_gguf(backend, &gf, &wm, vocab_size)?
    };
    Ok(DFlashDrafter::new(model, target_embed, lm_head))
}

#[cfg(all(test, feature = "metal", feature = "cpu"))]
mod real_dump_tests {
    use super::*;
    use crate::dflash::DFlashDrafter;
    use crate::metal::MetalBackend;

    /// 4.2c-verify: первый блок диффузии на РЕАЛЬНЫХ моделях (8B-драфтер +
    /// Qwen3-8B-Base) — дампы features/embeds/logits для числового сравнения
    /// с эталоном dflash/model.py (numpy). Запуск:
    /// cargo test --features metal,cpu dflash_real_block_dump -- --ignored
    #[test]
    #[ignore = "реальные GGUF (~10 ГБ на GPU)"]
    fn dflash_real_block_dump() {
        use crate::gguf;
        use crate::gguf::GgufValue;
        let backend = MetalBackend::new().unwrap();
        let home = std::env::var("HOME").unwrap();
        // Модели переехали в метарепу: models/ (реестр + gguf). Инф-переменная
        // для кастомного пути.
        let mdir = std::env::var("INFER_MODELS_DIR")
            .unwrap_or_else(|_| format!("{home}/work/ai/inference/models"));
        let tp = format!("{mdir}/llm/Qwen3-8B-Base-Q8_0.gguf");
        // ОРИГИНАЛ z-lab (Q8_0.gguf старше — другие веса, приёмка 0 в
        // принципе). Требует config.json рядом в ~/models/dflash/.
        let dp = format!("{mdir}/dflash/qwen3-8b-dflash-zlab.safetensors");
        let tb = std::fs::read(&tp).unwrap();
        let tgf = gguf::parse(&tb).unwrap();
        let twm = crate::weights_map::WeightsMap::from_gguf(&tgf).unwrap();
        let voc = crate::tokenizer::Tokenizer::from_gguf(&tgf).unwrap();
        let vocab = voc.vocab_size();
        let tc = crate::qwen3::Qwen3Config::from_gguf(&tgf, vocab).unwrap();
        let mut target = crate::qwen3::Qwen3Model::load(&backend, &twm, tc.clone()).unwrap();
        // осмысленный промпт (tg-мусор из 'a' вырождал модель в
        // детерминированную петлю [220,170,222,107,...] — приёмка на нём
        // бессмысленна); фичи = ВСЕ строки префилла (z-lab первый блок).
        let ids: Vec<u32> = voc.encode(
            "Расскажи подробно, как работает спекулятивное декодирование в трансформерах: какие есть методы и в чём их отличия?",
        );
        let tlayers = vec![1, 9, 17, 25, 33];
        let (logits, extract) = target.prefill_extract(&backend, &ids, &tlayers).unwrap();
        let lv = backend.to_cpu_f32(&logits).unwrap();
        let v = tc.vocab_size;
        let t_prompt = ids.len();
        let anchor = lv
            .chunks(v)
            .last()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0 as u32
            })
            .unwrap();
        eprintln!("anchor={anchor} vocab={v} t_prompt={t_prompt}");
        let mut dr = crate::dflash::load_drafter_gguf(
            &backend,
            std::path::Path::new(&dp),
            target.token_embed_table().clone(),
            target.output_projection().clone(),
            vocab,
        )
        .expect("drafter load");
        // фичи (t_prompt строк × 5 слоёв, выходы слоёв) — ТОКЕН-МАЖОР
        // (z-lab: cat по последней оси): строка t = [l0[t]; l1[t]; …; l4[t]].
        let d = tc.dim;
        let n_layers = extract.len();
        let mut feat: Vec<f32> = vec![0f32; n_layers * t_prompt * d];
        for (li, t) in extract.iter().enumerate() {
            let rows = backend.to_cpu_f32(t).unwrap();
            for r in 0..t_prompt {
                let src = r * d;
                let dst = r * n_layers * d + li * d;
                feat[dst..dst + d].copy_from_slice(&rows[src..src + d]);
            }
        }
        let ft = backend.from_cpu_f32(&feat, vec![t_prompt, 5 * d]).unwrap();
        dr.features = Some(ft.clone());
        // блок: [anchor, MASK×15], anchor на позиции t_prompt
        let block = dr.model.block_size;
        let mask = dr.model.mask_token;
        let mut bids = vec![anchor];
        bids.resize(block, mask);
        let pos = t_prompt;
        let res = dr
            .model
            .diffuse(&backend, &bids, &dr.lm_head.clone(), pos, dr.features.as_ref())
            .expect("diffuse");
        // дампы
        let write_f32 = |path: &str, data: &[f32]| {
            let mut b: Vec<u8> = Vec::with_capacity(data.len() * 4);
            for x in data {
                b.extend_from_slice(&x.to_le_bytes());
            }
            std::fs::write(path, b).unwrap();
        };
        write_f32("/tmp/dump_features.bin", &feat);
        // эмбеддинги блока (строки target_embed)
        let emb = backend.to_cpu_f32(target.token_embed_table()).unwrap();
        let mut bte = Vec::with_capacity(block * d);
        for &id in &bids {
            bte.extend_from_slice(&emb[id as usize * d..(id as usize + 1) * d]);
        }
        write_f32("/tmp/dump_embeds.bin", &bte);
        let lg = backend.to_cpu_f32(&res.logits).unwrap();
        write_f32("/tmp/dump_logits.bin", &lg);
        // lm_head таргета в f32 — для python-эталона приёмки (top-K IOU):
        // dequant из GGUF (тот же путь, что и загрузка движка).
        if let Some(t) = twm.get("output.weight") {
            let hd = crate::weights_map::dequant_f32(t);
            if !hd.is_empty() {
                write_f32("/tmp/dump_head.bin", &hd);
                eprintln!("head dumped: {} f32", hd.len());
            }
        }
        // Веса драфтера Q8-деквант (слой 0 q/k + fc) — сверка против bf16.
        if let Some(t) = twm.get("blk.0.attn_q.weight") {
            write_f32("/tmp/dump_q_attn_q.bin", &crate::weights_map::dequant_f32(t));
        }
        if let Some(t) = twm.get("blk.0.attn_k.weight") {
            write_f32("/tmp/dump_q_attn_k.bin", &crate::weights_map::dequant_f32(t));
        }
        if let Some(t) = twm.get("fc.weight") {
            write_f32("/tmp/dump_q_fc.bin", &crate::weights_map::dequant_f32(t));
        }
        eprintln!("q8 weight dumps written");
        let hd = backend.to_cpu_f32(&res.hidden).unwrap();
        write_f32("/tmp/dump_hidden.bin", &hd);
        let meta = format!(
            "{{\"vocab\":{v},\"dim\":{d},\"block\":{block},\"pos\":{pos},\"mask\":{mask},\"anchor\":{anchor},\"n_layers\":5,\"tlayers\":[1,9,17,25,33],\"t_prompt\":{t_prompt}}}"
        );
        std::fs::write("/tmp/dump_meta.json", meta).unwrap();
        eprintln!("dumps written");
        // ── приёмка первого блока: черновик vs фактические токены таргета ──
        use crate::spec::Drafter;
        let mut lg = logits.clone();
        let mut actual = Vec::with_capacity(12);
        for _ in 0..12 {
            let rv = backend.to_cpu_f32(&lg).unwrap();
            let row = &rv[..v];
            let tok = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0 as u32;
            actual.push(tok);
            lg = target.decode(&backend, tok).unwrap();
        }
        eprintln!("anchor={anchor} actual(1..)={:?}", &actual[..7]);
        // свежий драфтер, тот же контекст: draft() работает по history
        let mut dr2 = crate::dflash::load_drafter_gguf(
            &backend,
            std::path::Path::new(&dp),
            target.token_embed_table().clone(),
            target.output_projection().clone(),
            vocab,
        )
        .expect("drafter2");
        dr2.features = Some(ft.clone());
        dr2.commit(&[anchor]);
        let mut history = ids.clone();
        history.push(anchor);
        let (tokens, probs) = dr2.draft(&backend, &history, 6);
        eprintln!("actual  = {:?}", &actual[..7]);
        eprintln!("draft   = {:?}", tokens);
        eprintln!("probs   = {:?}", probs.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>());
        let matched = tokens
            .iter()
            .zip(actual.iter().skip(1))
            .filter(|(a, b)| a == b)
            .count();
        eprintln!("first-block acceptance: {matched}/{} (с лагом 1, как z-lab)", tokens.len());
    }
}
