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
    /// Sliding-window attention (gemma-style); None = full causal context.
    pub swa_window: Option<usize>,
    /// MROPE-секции (Qwen3-VL, IMROPE): [t, h, w, e] в парах; текст ротирует
    /// t/h/w-пары позицией pos, e-пары — позицией 0 (не ротируются).
    pub mrope_sections: Option<[usize; 4]>,
    /// Текстовый deepstack (Qwen3-VL): после первых N слоёв к выходу
    /// добавляются входные эмбеддинги со сдвигом (il+1) токенов
    /// (llama.cpp qwen3vl.cpp: t_inp_embd view). None = нет.
    pub n_deepstack: Option<usize>,
}

impl Qwen3Config {
    pub fn from_gguf(file: &GgufFile<'_>, vocab_size: usize) -> Result<Self, BackendError> {
        // Qwen3 и Qwen3-VL (MROPE-текст = обычный rope с их freq_base).
        let p = |s: &str| -> String {
            if file.get("qwen3vl.block_count").is_some() {
                format!("qwen3vl.{s}")
            } else {
                format!("qwen3.{s}")
            }
        };
        let u = |k: &str| -> Result<usize, BackendError> {
            file.get(k)
                .and_then(GgufValue::as_u32)
                .map(|v| v as usize)
                .ok_or_else(|| BackendError::Shape(format!("missing {k}")))
        };
        let dim = u(&p("embedding_length"))?;
        let n_heads = u(&p("attention.head_count"))?;
        let n_kv_heads = u(&p("attention.head_count_kv"))?;
        let ffn_dim = u(&p("feed_forward_length"))?;
        let n_layers = u(&p("block_count"))?;
        // head_dim: llama.cpp хранит его отдельно (…attention.key_length);
        // fallback — dim/n_heads (classic).
        let head_dim = u(&p("attention.key_length")).unwrap_or(dim / n_heads);
        let theta = file
            .get(&p("rope.freq_base"))
            .or_else(|| file.get(&p("attention.rope_theta")))
            .and_then(|v| match v {
                GgufValue::F32(f) => Some(*f as f64),
                GgufValue::F64(f) => Some(*f),
                _ => None,
            })
            .unwrap_or(1_000_000.0) as f32;
        let eps = file
            .get(&p("attention.layer_norm_rms_epsilon"))
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
            swa_window: file
                .get(&p("attention.sliding_window"))
                .and_then(GgufValue::as_u32)
                .map(|v| v as usize),
            mrope_sections: file
                .get(&p("rope.dimension_sections"))
                .and_then(GgufValue::as_array)
                .map(|arr| {
                    let mut out = [0usize; 4];
                    for (i, v) in arr.iter().take(4).enumerate() {
                        out[i] = v.as_u32().unwrap_or(0) as usize;
                    }
                    out
                }),
            n_deepstack: file
                .get(&p("n_deepstack_layers"))
                .and_then(GgufValue::as_u32)
                .map(|v| v as usize),
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
    pub caches: Vec<KvCache>,
    /// Логическая rope-позиция следующего decode (M-RoPE: секция t). Для
    /// текста равна числу токенов в кэше, но после картинки (Qwen3-VL)
    /// позиция растёт на max(nx,ny)=24, а не на 2304 vision-токенов.
    pos_rope: usize,
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
            pos_rope: 0,
        })
    }

    /// Prefill: ids [T] → logits [1, T, V]; KV-кэши пополняются.
    /// Временный debug: эмбеддинги (сверка prefill vs decode).
    pub fn embed_forward_test(
        &self,
        backend: &dyn Backend,
        ids: &[u32],
    ) -> Result<Tensor, BackendError> {
        self.embed.forward(backend, ids, ids.len())
    }

    pub fn prefill(&mut self, backend: &dyn Backend, ids: &[u32]) -> Result<Tensor, BackendError> {
        let t = ids.len();
        let x = self.embed.forward(backend, ids, t)?;
        let logits = self.forward_layers(backend, x, t, None)?;
        self.pos_rope = self.caches[0].pos;
        Ok(logits)
    }

    /// Prefill с vision-эмбеддингами (Ф1.11): токены == `vision_token`
    /// заменяются строками vision-энкодера (по порядку появления) — всё на
    /// GPU (embedding_override). `pos4d` — M-RoPE позиции (t,h,w,e) на токен
    /// (len = t*4; текст: t=h=w=pos, e=0; картинка: t=pos_0, h=pos_0+i/nx,
    /// w=pos_0+i%nx, e=0). `pos_after` — логическая позиция (секция t)
    /// следующего decode (после картинки pos_0+24, а не +2304).
    pub fn prefill_with_vision(
        &mut self,
        backend: &dyn Backend,
        ids: &[u32],
        vision: &Tensor,
        vision_token: u32,
        pos4d: &[u32],
        pos_after: usize,
    ) -> Result<Tensor, BackendError> {
        let t = ids.len();
        assert_eq!(pos4d.len(), t * 4, "pos4d: {t} токенов × 4");
        let x = backend.zeros(vec![1, t, self.cfg.dim])?;
        backend.embedding_override(ids, &self.embed.table, vision, &x, vision_token)?;
        let logits = self.forward_layers(backend, x, t, Some(pos4d))?;
        self.pos_rope = pos_after;
        Ok(logits)
    }

    /// Общий путь слоёв после эмбеддинга (текст или vision-текст).
    fn forward_layers(
        &mut self,
        backend: &dyn Backend,
        mut x: Tensor,
        t: usize,
        mrope: Option<&[u32]>,
    ) -> Result<Tensor, BackendError> {
        let pos = if mrope.is_some() { 0 } else { self.pos_rope };
        // текстовый deepstack (Qwen3-VL): к слоям 0..N-1 добавляются входные
        // эмбеддинги со сдвигом (il+1) токенов (llama.cpp t_inp_embd view);
        // для последних токенов сдвиг выходит за границу — вклад нулевой.
        let embd_in = x.clone();
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
                mrope,
            )?;
            #[cfg(feature = "metal")]
            if std::env::var("Q3_DBG_X20").is_ok() && i == 0 {
                use std::sync::atomic::{AtomicUsize, Ordering};
                static XPTR_COUNT: AtomicUsize = AtomicUsize::new(0);
                let c = XPTR_COUNT.fetch_add(1, Ordering::Relaxed);
                if c == 0 || (c == 20 && t == 1) || (c == 0 && t > 1) {
                    eprintln!("Q3_DBG addptr x={} o={}",
                        x.as_gpu().buffer.contents() as usize,
                        h.as_gpu().buffer.contents() as usize);
                }
            }
            #[cfg(feature = "metal")]
            if std::env::var("Q3_DBG_X20").is_ok() && i == 0 {
                let xv = backend.to_cpu_f32(&x)?;
                let last = xv.len() - self.cfg.dim;
                eprintln!("Q3_DBG x20 вход2: {:?}", &xv[last..last + 4]);
            }
            #[cfg(feature = "metal")]
            if std::env::var("Q3_DBG_X20").is_ok() && i == 0 {
                let xv = backend.to_cpu_f32(&x)?;
                let last = xv.len() - self.cfg.dim;
                eprintln!("Q3_DBG x20 вход: {:?}", &xv[last..last + 8]);
            }
            x = backend_add(backend, &x, &h)?;
            #[cfg(feature = "metal")]
            if std::env::var("Q3_DBG_X20").is_ok() && i == 0 {
                let dv = backend.to_cpu_f32(&x)?;
                let last = dv.len() - self.cfg.dim;
                let mut s = String::new();
                for v in &dv[last..last + self.cfg.dim] {
                    s.push_str(&format!("{v:.9}\n"));
                }
                std::fs::write("/tmp/x20_a0.txt", s)?;
            }
            let h = layer.ffn_norm.forward(backend, &x)?;
            let f = ffn_forward(backend, &h, &layer.ffn)?;
            x = backend_add(backend, &x, &f)?;
            // deepstack: x[t] += embd_in[t + il + 1] — только в vision-пути
            // (llama.cpp: в text-only токен-пути расширенный входной буфер
            // пэддится нулями — вклад нулевой; смещения кладёт mtmd при
            // vision-входе). Временная CPU-проверка, потом — MSL-кернел.
            if mrope.is_some() && std::env::var("Q3_NO_DS").is_err() {
                if let Some(n_ds) = self.cfg.n_deepstack {
                    if i < n_ds {
                        let shift = i + 1;
                        if t > shift {
                            let xd = backend.to_cpu_f32(&x)?;
                            let ed = backend.to_cpu_f32(&embd_in)?;
                            let mut nd = xd.clone();
                            for tt in 0..t - shift {
                                let src = (tt + shift) * self.cfg.dim;
                                let dst = tt * self.cfg.dim;
                                for d in 0..self.cfg.dim {
                                    nd[dst + d] += ed[src + d];
                                }
                            }
                            x = backend.from_cpu_f32(&nd, vec![1, t, self.cfg.dim])?;
                        }
                    }
                }
            }
            #[cfg(feature = "metal")]
            if std::env::var("Q3_DBG_X20").is_ok() && i == 0 {
                let dv = backend.to_cpu_f32(&x)?;
                let last = dv.len() - self.cfg.dim;
                let mut s = String::new();
                for v in &dv[last..last + self.cfg.dim] {
                    s.push_str(&format!("{v:.9}\n"));
                }
                std::fs::write("/tmp/x20_f0.txt", s)?;
            }
        }
        let h = self.out_norm.forward(backend, &x)?;
        if std::env::var("Q3_DBG_HN").is_ok() {
            let dv = backend.to_cpu_f32(&h)?;
            let last = dv.len() - self.cfg.dim;
            let mut s = String::new();
            for v in &dv[last..last + self.cfg.dim] {
                s.push_str(&format!("{v:.9}\n"));
            }
            std::fs::write("/tmp/hn_dump.txt", s)?;
        }
        let (b, tq, d) = (h.shape()[0], h.shape()[1], h.shape()[2]);
        let h2 = h.view(vec![b * tq, d]);
        let logits = backend.zeros(vec![b * tq, self.cfg.vocab_size])?;
        backend.gemm_nt(&h2, &self.output, &logits)?;
        Ok(logits.view(vec![b, tq, self.cfg.vocab_size]))
    }

    /// Decode: один токен → logits [1, 1, V]; KV-кэш растёт.
    pub fn decode(&mut self, backend: &dyn Backend, id: u32) -> Result<Tensor, BackendError> {
        let x = self.embed.forward(backend, &[id], 1)?;
        // Qwen3 debug scaffold (pointer dumps) — Metal-only (as_gpu); other
        // backends must still compile (Linux cpu/cuda builds).
        #[cfg(feature = "metal")]
        if std::env::var("Q3_DBG_X20").is_ok() {
            let xv = backend.to_cpu_f32(&x)?;
            eprintln!("Q3_DBG embed_out id={id}: {:?}", &xv[..4]);
        }
        let logits = self.forward_layers(backend, x, 1, None)?;
        self.pos_rope += 1;
        Ok(logits)
    }

    /// Сброс KV-кэшей (новый контекст).
    pub fn reset_cache(&mut self, backend: &dyn Backend) -> Result<(), BackendError> {
        for c in &mut self.caches {
            c.reset();
            c.shrink_to_init(backend)?;
        }
        self.pos_rope = 0;
        Ok(())
    }

    /// Стандартный rotate-half RoPE (HF) на CPU-хелпере + Attention с
    /// identity-таблицами. `mrope` — per-token M-RoPE позиции (t,h,w,e на
    /// токен, len = t*4): картинка Qwen3-VL видит ВСЮ себя (все t равны),
    /// текст после картинки — каузально по t.
    fn rope_and_attn(
        cfg: &Qwen3Config,
        backend: &dyn Backend,
        h: &Tensor,
        attn: &Attention,
        cache: &mut KvCache,
        pos: usize,
        t: usize,
        mrope: Option<&[u32]>,
    ) -> Result<Tensor, BackendError> {
        // q/k из h через Attention — но Attention применяет interleaved-RoPE
        // сам. Делаем pre-rotation стандартным RoPE и передаём identity-таблицы:
        // для этого раскладываем h на q/k/v, поворачиваем q/k, собираем обратно
        // и отдаём Attention'у с no-op rope.
        let (b, tq, _d) = (h.shape()[0], h.shape()[1], h.shape()[2]);
        let q_size = attn.n_heads * attn.head_dim;
        let kv_size = attn.n_kv_heads * attn.head_dim;
        let _timing = std::env::var("Q3_TIMING").is_ok();
        macro_rules! tlog {
            ($name:expr, $start:expr) => {
                if _timing {
                    eprintln!("TIMING {} {:.1}ms", $name, $start.elapsed().as_secs_f64() * 1e3);
                }
            };
        }
        // Qwen3 debug scaffold (pointer dumps) — Metal-only (as_gpu); other
        // backends must still compile (Linux cpu/cuda builds).
        #[cfg(feature = "metal")]
        if std::env::var("Q3_DBG_X20").is_ok() {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static H_COUNT: AtomicUsize = AtomicUsize::new(0);
            let c = H_COUNT.fetch_add(1, Ordering::Relaxed);
            let is_target = if tq > 1 { c == 0 } else { c == 20 * 28 };
            if is_target {
                let hv = backend.to_cpu_f32(h)?;
                let row = if tq > 1 { (20.min(tq - 1)) * h.shape()[2] } else { 0 };
                eprintln!("Q3_DBG h_before[{}]: {:?}", row / h.shape()[2], &hv[row..row + 8]);
            }
        }
        let t0_qkv = std::time::Instant::now();
        let qkv = Linear {
            w: attn.wqkv.clone(),
            b: None,
        }
        .forward(backend, h)?;
        tlog!("qkv_gemm", t0_qkv);
        #[cfg(feature = "metal")]
        if std::env::var("Q3_DBG_X20").is_ok() && tq == 1 {
            eprintln!("Q3_DBG ptrs qkv={} h={} w={} ck={}", qkv.as_gpu().buffer.contents() as usize, h.as_gpu().buffer.contents() as usize, attn.wqkv.as_gpu().buffer.contents() as usize, cache.view().as_gpu().buffer.contents() as usize);
        }
        // Qwen3 debug scaffold (pointer dumps) — Metal-only (as_gpu); other
        // backends must still compile (Linux cpu/cuda builds).
        #[cfg(feature = "metal")]
        if std::env::var("Q3_DBG_X20").is_ok() {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static QKV_COUNT: AtomicUsize = AtomicUsize::new(0);
            let c = QKV_COUNT.fetch_add(1, Ordering::Relaxed);
            let is_target = if tq > 1 { c == 0 } else { c == 20 * 28 };
            if is_target {
                let qv = backend.to_cpu_f32(&qkv)?;
                let row = if tq > 1 { (20.min(tq - 1)) * (q_size + 2 * kv_size) } else { 0 };
                let mut s = String::new();
                for v in &qv[row..row + 16] {
                    s.push_str(&format!("{v:.9}\n"));
                }
                std::fs::write("/tmp/qkv_l0.txt", s)?;
            }
        }
        if std::env::var("Q3_DBG_X").is_ok() && tq > 16 {
            let qv = backend.to_cpu_f32(&qkv)?;
            eprintln!("Q3_DBG qkv[16][2048..2052]: {:?}", &qv[16 * (q_size + 2 * kv_size) + q_size..16 * (q_size + 2 * kv_size) + q_size + 4]);
        }
        let q = crate::nn::slice_last(backend, &qkv, q_size, 0)?;
        let k = crate::nn::slice_last(backend, &qkv, kv_size, q_size)?;
        let v = crate::nn::slice_last(backend, &qkv, kv_size, q_size + kv_size)?;
        if std::env::var("Q3_DBG_X").is_ok() && tq > 16 {
            let kv = backend.to_cpu_f32(&k)?;
            eprintln!("Q3_DBG slice_k[16][0..4]: {:?}", &kv[16 * kv_size..16 * kv_size + 4]);
        }
        let (q, k) = if attn.q_norm.is_some() && attn.k_norm.is_some() {
            (
                qk_norm(backend, &q, attn.q_norm.as_ref().unwrap(), attn.n_heads)?,
                qk_norm(backend, &k, attn.k_norm.as_ref().unwrap(), attn.n_kv_heads)?,
            )
        } else {
            (q, k)
        };
        if std::env::var("Q3_DBG_X").is_ok() && tq > 16 {
            let kv = backend.to_cpu_f32(&k)?;
            eprintln!("Q3_DBG qkn_k[16][0..4]: {:?}", &kv[16 * kv_size..16 * kv_size + 4]);
        }
        let (q, k) = if let Some(mrope) = mrope {
            let sects = cfg.mrope_sections.expect("mrope без sections");
            (
                backend.rope_mrope(&q, mrope, attn.head_dim, cfg.rope_theta, sects)?,
                backend.rope_mrope(&k, mrope, attn.head_dim, cfg.rope_theta, sects)?,
            )
        } else if let Some(sects) = cfg.mrope_sections {
            if std::env::var("Q3_NO_MROPE").is_ok() {
                // ТЕСТ: обычный rope без секций
                (
                    rope_standard(backend, &q, pos, t, attn.head_dim, cfg.rope_theta)?,
                    rope_standard(backend, &k, pos, t, attn.head_dim, cfg.rope_theta)?,
                )
            } else if std::env::var("Q3_HW_ZERO").is_ok() {
                // ТЕСТ: t=pos, h/w/e=0 (pos4d = (p,0,0,0))
                let mut pos4d = vec![0u32; t * 4];
                for tt in 0..t {
                    pos4d[tt * 4] = (pos + tt) as u32;
                }
                (
                    backend.rope_mrope(&q, &pos4d, attn.head_dim, cfg.rope_theta, sects)?,
                    backend.rope_mrope(&k, &pos4d, attn.head_dim, cfg.rope_theta, sects)?,
                )
            } else {
                #[cfg(feature = "metal")]
                if std::env::var("Q3_DBG_X20").is_ok() && cfg.mrope_sections.is_some() {
                    eprintln!("Q3_DBG rope_text q_buf={} (before) tq={} pos={}",
                        q.as_gpu().buffer.contents() as usize, t, pos);
                }
                let qr = backend.rope_mrope_text(&q, pos, attn.head_dim, cfg.rope_theta, sects)?;
                let kr = backend.rope_mrope_text(&k, pos, attn.head_dim, cfg.rope_theta, sects)?;
                // Qwen3 debug scaffold (pointer dumps) — Metal-only (as_gpu); other
        // backends must still compile (Linux cpu/cuda builds).
        #[cfg(feature = "metal")]
        if std::env::var("Q3_DBG_X20").is_ok() {
                    eprintln!("Q3_DBG rope_text q_after={} k_after={}",
                        qr.as_gpu().buffer.contents() as usize,
                        kr.as_gpu().buffer.contents() as usize);
                }
                (qr, kr)
            }
        } else {
            (
                rope_standard(backend, &q, pos, t, attn.head_dim, cfg.rope_theta)?,
                rope_standard(backend, &k, pos, t, attn.head_dim, cfg.rope_theta)?,
            )
        };
        // attention напрямую через attn_ar (q/k уже повёрнуты стандартным RoPE)
        // q/k/v 3D [1, T, D]; attn_ar сам разложит по головам
        let t0_ck = std::time::Instant::now();
        cache.update(backend, &k, &v)?;
        tlog!("cache_update", t0_ck);
        // Qwen3 debug scaffold (pointer dumps) — Metal-only (as_gpu); other
        // backends must still compile (Linux cpu/cuda builds).
        #[cfg(feature = "metal")]
        if std::env::var("Q3_DBG_X20").is_ok() {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static Q20_COUNT2: AtomicUsize = AtomicUsize::new(0);
            let c = Q20_COUNT2.fetch_add(1, Ordering::Relaxed);
            if tq == 1 {
                eprintln!("Q3_DBG decode pos={pos} c={c}");
            }
            let is_target = if tq > 1 { c == 0 } else { c == 20 * 28 };
            if is_target {
                let qv = backend.to_cpu_f32(&q)?;
                let row = if tq > 1 { (20.min(tq - 1)) * q_size } else { 0 };
                let mut s = String::new();
                for v in &qv[row..row + 16] {
                    s.push_str(&format!("{v:.9}\n"));
                }
                std::fs::write("/tmp/q_l0d.txt", s)?;
            }
        }
        let mask = if let Some(mrope) = mrope {
            if t > 1 {
                Some(build_mrope_mask(backend, t, cache.pos, mrope)?)
            } else {
                None
            }
        } else if t > 1 {
            Some(crate::nn::build_causal_mask(backend, t, cache.pos, pos, cfg.swa_window)?)
        } else {
            None
        };
        let ck = cache.view();
        let cv = cache.view_v();
        let attn_out = backend.zeros(vec![b, tq, q_size])?;
        // Qwen3 debug scaffold (pointer dumps) — Metal-only (as_gpu); other
        // backends must still compile (Linux cpu/cuda builds).
        #[cfg(feature = "metal")]
        if std::env::var("Q3_DBG_X20").is_ok() {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static AOP_COUNT: AtomicUsize = AtomicUsize::new(0);
            let c = AOP_COUNT.fetch_add(1, Ordering::Relaxed);
            if (tq > 1 && c == 0) || (tq == 1 && c == 20 * 28) {
                eprintln!("Q3_DBG attn_out={} h={}",
                    attn_out.as_gpu().buffer.contents() as usize,
                    h.as_gpu().buffer.contents() as usize);
            }
        }
        let t0_ar = std::time::Instant::now();
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
        tlog!("attn_ar", t0_ar);
        // Qwen3 debug scaffold (pointer dumps) — Metal-only (as_gpu); other
        // backends must still compile (Linux cpu/cuda builds).
        #[cfg(feature = "metal")]
        if std::env::var("Q3_DBG_X20").is_ok() {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static ATTN3_COUNT: AtomicUsize = AtomicUsize::new(0);
            let c = ATTN3_COUNT.fetch_add(1, Ordering::Relaxed);
            let is_target = if tq > 1 { c == 0 } else { c == 20 * 28 };
            if is_target {
                let av = backend.to_cpu_f32(&attn_out)?;
                let row = if tq > 1 { (20.min(tq - 1)) * q_size } else { 0 };
                let mut s = String::new();
                for v in &av[row..row + 16] {
                    s.push_str(&format!("{v:.9}\n"));
                }
                std::fs::write("/tmp/attn_l0c.txt", s)?;
            }
        }

        let out = Linear {
            w: attn.wo.clone(),
            b: None,
        }
        .forward(backend, &attn_out)?;
        // Qwen3 debug scaffold (pointer dumps) — Metal-only (as_gpu); other
        // backends must still compile (Linux cpu/cuda builds).
        #[cfg(feature = "metal")]
        if std::env::var("Q3_DBG_X20").is_ok() {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static O_COUNT3: AtomicUsize = AtomicUsize::new(0);
            let c = O_COUNT3.fetch_add(1, Ordering::Relaxed);
            let is_target = if tq > 1 { c == 0 } else { c == 20 * 28 };
            if is_target {
                let ov = backend.to_cpu_f32(&out)?;
                let row = if tq > 1 { (20.min(tq - 1)) * q_size } else { 0 };
                let mut s = String::new();
                for v in &ov[row..row + 16] {
                    s.push_str(&format!("{v:.9}\n"));
                }
                std::fs::write("/tmp/o_l0c.txt", s)?;
            }
        }
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

/// M-RoPE attention mask: каузальность по t-позициям (секция 0). Токены
/// картинки имеют равные t → видят всю картинку; текст после картинки
/// (t больше) видит всё до себя. tkv — позиция в кэше (cache.pos).
fn build_mrope_mask(
    backend: &dyn Backend,
    tq: usize,
    tkv: usize,
    pos4d: &[u32],
) -> Result<Tensor, BackendError> {
    let mut data = vec![0.0f32; tq * tkv];
    for i in 0..tq {
        let ti = pos4d[i * 4];
        for j in 0..tkv {
            let tj = if j < tq { pos4d[j * 4] } else { 0 };
            if tj > ti {
                data[i * tkv + j] = f32::NEG_INFINITY;
            }
        }
    }
    backend.from_cpu_f32(&data, vec![1, tq, tkv])
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
    // Qwen3-VL FFN (llama.cpp build_ffn, LLM_FFN_SILU + LLM_FFN_PAR):
    // silu применяется к GATE-проекции, up остаётся линейным:
    //   gm = silu(x @ ffn_gate) * (x @ ffn_up) → down.
    // В GGUF Qwen3-VL ffn_gate/ffn_up лежат именно в этом порядке
    // (проверено: python-эталон совпал с llama.cpp при silu(gate)*up, corr 1.0).
    let (b, t, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    let x2 = x.view(vec![b * t, d]);
    let ffn_dim = ffn.w1.shape()[0];
    let a = backend.zeros(vec![b * t, ffn_dim])?;
    let g = backend.zeros(vec![b * t, ffn_dim])?;
    let gm = backend.zeros(vec![b * t, ffn_dim])?;
    let t0 = std::time::Instant::now();
    backend.gemm_nt(&x2, &ffn.w1, &a)?; // a = gate (в silu)
    backend.gemm_nt(&x2, &ffn.w3, &g)?; // g = up (линейный)
    backend.silu(&a, &a)?;
    backend.mul(&a, &g, &gm)?;
    if std::env::var("Q3_TIMING").is_ok() {
        eprintln!("TIMING ffn_gemm {:.1}ms", t0.elapsed().as_secs_f64() * 1e3);
    }
    #[cfg(feature = "metal")]
    if std::env::var("Q3_DBG_X20").is_ok() && t > 1 {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static FFN_COUNT: AtomicUsize = AtomicUsize::new(0);
        if FFN_COUNT.fetch_add(1, Ordering::Relaxed) == 0 {
            let uv = backend.to_cpu_f32(&a)?;
            let gv = backend.to_cpu_f32(&g)?;
            let mv = backend.to_cpu_f32(&gm)?;
            let mut s = String::new();
            for j in 0..16 {
                s.push_str(&format!("u {} {:.9} g {} {:.9} m {} {:.9}\n", j, uv[j], j, gv[j], j, mv[j]));
            }
            std::fs::write("/tmp/ffn_mid.txt", s)?;
        }
    }
    let y = backend.zeros(vec![b * t, d])?;
    #[cfg(feature = "metal")]
    if std::env::var("Q3_DBG_X20").is_ok() && t > 1 {
        eprintln!("Q3_DBG ptr y={:?} x={:?} gm={:?}",
            y.as_gpu().buffer.contents() as usize,
            x.as_gpu().buffer.contents() as usize,
            gm.as_gpu().buffer.contents() as usize);
    }
    let t1 = std::time::Instant::now();
    backend.gemm_nt(&gm, &ffn.w2, &y)?;
    if std::env::var("Q3_TIMING").is_ok() {
        eprintln!("TIMING ffn_down {:.1}ms", t1.elapsed().as_secs_f64() * 1e3);
    }
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
            swa_window: None,
            mrope_sections: None,
            n_deepstack: None,
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
                let d = (got - want).abs();
                if d >= 1e-3 {
                    eprintln!("DBG pos {k}: got {got} want {want} delta {d}");
                }
                assert!(d < 1e-3, "pos {k}: got {got} want {want}");
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

    /// rope_mrope (per-token pos4d) эквивалентен rope_mrope_text для текста
    /// (t=h=w=pos, e=0) и отличается для картинки (h/w свои). Также проверяем
    /// мёртвую h-категорию: пара с sector%3==1 ротируется позицией h.
    #[test]
    fn rope_mrope_matches_text_and_h_sector() {
        let backend = MetalBackend::new().unwrap();
        // x [1, 2, 8]: 2 токена, 2 головы hd=4
        let x: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        let xt = backend.from_cpu_f32(&x, vec![1, 2, 8]).unwrap();
        let sects = [24usize, 20, 20, 0];
        // текст: pos4d = (p,p,p,0)
        let text_pos: Vec<u32> = vec![1, 1, 1, 0, 2, 2, 2, 0];
        let a = backend.rope_mrope(&xt, &text_pos, 4, 5_000_000.0, sects).unwrap();
        let b = backend.rope_mrope_text(&xt, 1, 4, 5_000_000.0, sects).unwrap();
        let av = backend.to_cpu_f32(&a).unwrap();
        let bv = backend.to_cpu_f32(&b).unwrap();
        for (i, (x, y)) in av.iter().zip(&bv).enumerate() {
            assert!(
                (x - y).abs() < 1e-5,
                "text rope_mrope[{i}]: {x} vs text {y}"
            );
        }
        // картинка: первый токен pos4d=(1, 2, 3, 0): пара 1 (sector%3==1) — h-позиция 2.
        // С мёртвой h-категорией (старый баг sections) пара 1 не ротировалась бы.
        let img_pos: Vec<u32> = vec![1, 2, 3, 0, 5, 6, 7, 0];
        let c = backend.rope_mrope(&xt, &img_pos, 4, 5_000_000.0, sects).unwrap();
        let cv = backend.to_cpu_f32(&c).unwrap();
        // пара 1 токена 0: x0=0.5, x1=1.5, угол p=2·inv, inv=theta^-0.5≈4.47e-4
        let inv = 5_000_000f32.powf(-0.5);
        let (c_, s_) = ((2.0f32 * inv).cos(), (2.0f32 * inv).sin());
        let want0 = 0.5 * c_ - 1.5 * s_;
        assert!(
            (cv[1] - want0).abs() < 1e-5,
            "h-sector rotated: got {} want {}",
            cv[1],
            want0
        );
    }
}

#[cfg(all(test, feature = "metal"))]
mod attn_tests {
    use super::*;
    use crate::metal::MetalBackend;

    fn rng_u64(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        }
    }

    #[test]
    fn attn_ar_matches_cpu_reference() {
        let backend = MetalBackend::new().unwrap();
        let mut rng = rng_u64(0x4174746e);
        let (b, tq, tkv, h, kvh, hd) = (1usize, 20usize, 20usize, 16usize, 8usize, 128usize);
        let d = h * hd;
        let kvd = kvh * hd;
        let qv: Vec<f32> = (0..b * tq * d).map(|_| (rng() >> 32) as f32 / (1u64 << 32) as f32 - 0.5).collect();
        let kv: Vec<f32> = (0..b * tkv * kvd).map(|_| (rng() >> 32) as f32 / (1u64 << 32) as f32 - 0.5).collect();
        let vv: Vec<f32> = (0..b * tkv * kvd).map(|_| (rng() >> 32) as f32 / (1u64 << 32) as f32 - 0.5).collect();
        // маска: каузальная: j > t → -inf
        let mut mv = vec![0.0f32; b * tq * tkv];
        for t in 0..tq {
            for j in 0..tkv {
                if j > t {
                    mv[t * tkv + j] = f32::NEG_INFINITY;
                }
            }
        }
        let q = backend.from_cpu_f32(&qv, vec![b, tq, d]).unwrap();
        let k = backend.from_cpu_f32(&kv, vec![b, tkv, kvd]).unwrap();
        let v = backend.from_cpu_f32(&vv, vec![b, tkv, kvd]).unwrap();
        let mask = backend.from_cpu_f32(&mv, vec![b, tq, tkv]).unwrap();
        let y = backend.zeros(vec![b, tq, d]).unwrap();
        backend.attn_ar(&q, &k, &v, Some(&mask), &y, h, kvh, hd).unwrap();
        let yv = backend.to_cpu_f32(&y).unwrap();
        // CPU-референс
        let scale = 1.0 / (hd as f32).sqrt();
        let n_rep = h / kvh;
        let mut want = vec![0.0f32; b * tq * d];
        for t in 0..tq {
            for hh in 0..h {
                let kvhi = hh / n_rep;
                let mut scores = vec![0.0f32; tkv];
                for j in 0..tkv {
                    let mut acc = 0.0f32;
                    for dd in 0..hd {
                        acc += qv[(t * d + hh * hd + dd)] * kv[(j * kvd + kvhi * hd + dd)];
                    }
                    scores[j] = acc * scale + mv[t * tkv + j];
                }
                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for j in 0..tkv {
                    sum += (scores[j] - mx).exp();
                }
                for dd in 0..hd {
                    let mut acc = 0.0f32;
                    for j in 0..tkv {
                        acc += ((scores[j] - mx).exp() / sum) * vv[(j * kvd + kvhi * hd + dd)];
                    }
                    want[t * d + hh * hd + dd] = acc;
                }
            }
        }
        for i in 0..yv.len() {
            let dlt = (yv[i] - want[i]).abs();
            assert!(dlt < 1e-4, "idx {i}: got {} want {} delta {}", yv[i], want[i], dlt);
        }
        eprintln!("attn_ar_matches_cpu_reference OK ({} elems)", yv.len());
    }
}
