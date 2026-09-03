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
    /// Текстовый deepstack (Qwen3-VL): после слоёв 0..N-1 к скрытым
    /// состояниям НА ПОЗИЦИЯХ image-токенов добавляются deepstack-фичи
    /// vision-энкодера (HF `_deepstack_process`: hidden[visual_pos] +=
    /// ds[layer]; в llama.cpp qwen3vl.cpp per-token компоненты
    /// [main|ds0|ds1|ds2] входного буфера для текстовых токенов нулевые).
    /// None = нет.
    pub n_deepstack: Option<usize>,
}

impl Qwen3Config {
    pub fn from_gguf(file: &GgufFile<'_>, vocab_size: usize) -> Result<Self, BackendError> {
        // Архитектура из метаданных: qwen3 / qwen3vl / qwen35 (M-RoPE) и т.п.
        let arch = file.architecture().unwrap_or("qwen3").to_string();
        let p = |s: &str| -> String { format!("{arch}.{s}") };
        let u = |k: &str| -> Result<usize, BackendError> {
            file.get(k)
                .and_then(GgufValue::as_u32)
                .map(|v| v as usize)
                .ok_or_else(|| BackendError::Shape(format!("missing {k}")))
        };
        let dim = u(&p("embedding_length"))?;
        let n_heads = u(&p("attention.head_count"))?;
        // N (iter-10): GGUF metadata is external input — head_count = 0
        // panicked in the dim / n_heads fallback below (the same class the
        // dflash guard closed; qwen3 was left unguarded).
        if n_heads == 0 {
            return Err(BackendError::Shape(
                format!("{arch}.attention.head_count must be >= 1, got 0"),
            ));
        }
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

#[derive(Debug, Clone)]
pub struct Qwen3Layer {
    pub attn_norm: RmsNorm,
    pub attn: Attention,
    pub ffn_norm: RmsNorm,
    pub ffn: SwiGLU,
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
    /// Таблица эмбеддингов (для DFlash-драфтера, 4.2b).
    pub fn token_embed_table(&self) -> &Tensor {
        &self.embed.table
    }

    /// Выходная проекция [V, D] (для DFlash-драфтера).
    pub fn output_projection(&self) -> &Tensor {
        &self.output
    }
}

/// Результат спекулятивного шага (Ф2.1).
pub struct SpecOutcome {
    /// Принятые токены (первый — гарантированно; остальные — draft).
    /// ВНИМАНИЕ: последний элемент при `n_draft_accepted < draft.len()` —
    /// пересэмпл, он в KV-кэше НЕ лежит (кэш откатывается до принятых) —
    /// следующий decode/decode_spec обязан прогнать его через forward.
    pub accepted: Vec<u32>,
    /// Логиты последней позиции (продолжение, если приняты все draft).
    pub logits: Tensor,
    /// Скрытые extract-слоёв таргета (по ВСЕМ позициям forward'а, без
    /// out_norm) — для DFlash-инжекта следующего шага (Ф2.4).
    pub extract: Vec<Tensor>,
    /// Сколько draft-токенов принято подряд (0..=draft.len()); если
    /// < draft.len() — последний элемент accepted — ПЕРЕСЭМПЛ (его фич
    /// в extract нет: строка соответствует draft, а не токену).
    pub n_draft_accepted: usize,
    /// Строгое число ВЕРИФИЦИРОВАННЫХ черновиков (без пересэмпла) —
    /// для честной метрики приёмки (4.2b): n_draft_accepted при отказе
    /// считал пересэмпл как принятый.
    pub n_draft_verified: usize,
}

impl Qwen3Model {
    #[allow(clippy::too_many_arguments)]
    /// Квант-осведомлённая загрузка веса: Q4_0/Q8_0/Q4_K/Q6_K — нативно
    /// (dequant-in-kernel GEMM/MMV), иначе dequant→f16 (как раньше).
    fn upload_q(
        backend: &dyn Backend,
        w: &WeightsMap<'_>,
        name: &str,
        shape: Vec<usize>,
    ) -> Result<Tensor, BackendError> {
        let tensor = w
            .get(name)
            .ok_or_else(|| BackendError::Shape(format!("missing tensor {name}")))?;
        match tensor.format {
            crate::dtype::WeightFormat::Q4_0
            | crate::dtype::WeightFormat::Q8_0
            | crate::dtype::WeightFormat::Q4_K
            | crate::dtype::WeightFormat::Q6_K => {
                backend.from_quantized(tensor.raw(), shape, tensor.format)
            }
            _ => upload_w(backend, &crate::weights_map::dequant_f32(tensor), shape),
        }
    }

    /// Fused-веса (q/k/v, gate/up) с нативным квантом, если ВСЕ части
    /// квантованы одинаково (GGUF-модели — всегда); иначе dequant→f16.
    fn upload_fused(
        backend: &dyn Backend,
        w: &WeightsMap<'_>,
        names: &[&str],
        shape: Vec<usize>,
    ) -> Result<Tensor, BackendError> {
        let tensors: Vec<&crate::weights_map::WeightTensor<'_>> = names
            .iter()
            .map(|n| {
                w.get(n)
                    .ok_or_else(|| BackendError::Shape(format!("missing tensor {n}")))
            })
            .collect::<Result<_, _>>()?;
        let q = tensors[0].format;
        let all_q = tensors.iter().all(|t| t.format == q)
            && matches!(
                q,
                crate::dtype::WeightFormat::Q4_0
                    | crate::dtype::WeightFormat::Q8_0
                    | crate::dtype::WeightFormat::Q4_K
                    | crate::dtype::WeightFormat::Q6_K
            );
        if all_q {
            let mut bytes = Vec::new();
            for t in &tensors {
                bytes.extend_from_slice(t.raw());
            }
            backend.from_quantized(&bytes, shape, q)
        } else {
            let mut f = Vec::new();
            for t in &tensors {
                f.extend(crate::weights_map::dequant_f32(t));
            }
            upload_w(backend, &f, shape)
        }
    }

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
            // fuse q/k/v → [3D, D] (нативный квант, если все Q8_0/Q4_0/...)
            let qn = p("attn_q.weight");
            let kn = p("attn_k.weight");
            let vn = p("attn_v.weight");
            let wqkv = Self::upload_fused(
                backend,
                w,
                &[&qn, &kn, &vn],
                vec![
                    cfg.n_heads * cfg.head_dim + 2 * cfg.n_kv_heads * cfg.head_dim,
                    cfg.dim,
                ],
            )?;
            // QK-norm опционален (Qwen3 — есть; Qwen2.5-Omni thinker — нет).
            let q_norm = if w.get(&p("attn_q_norm.weight")).is_some() {
                Some(backend.from_cpu_f32(&t(&p("attn_q_norm.weight"))?, vec![cfg.head_dim])?)
            } else {
                None
            };
            let k_norm = if w.get(&p("attn_k_norm.weight")).is_some() {
                Some(backend.from_cpu_f32(&t(&p("attn_k_norm.weight"))?, vec![cfg.head_dim])?)
            } else {
                None
            };
            let qkv_bias = if w.get(&p("attn_q.bias")).is_some() {
                let mut b = t(&p("attn_q.bias"))?;
                b.extend(t(&p("attn_k.bias"))?);
                b.extend(t(&p("attn_v.bias"))?);
                Some(b)
            } else {
                None
            };
            let wo = Self::upload_q(
                backend,
                w,
                &p("attn_output.weight"),
                vec![cfg.dim, cfg.n_heads * cfg.head_dim],
            )?;
            let attn = Attention {
                wqkv,
                wo,
                qkv_bias: qkv_bias
                    .map(|b| {
                        backend.from_cpu_f32(
                            &b,
                            vec![cfg.n_heads * cfg.head_dim + 2 * cfg.n_kv_heads * cfg.head_dim],
                        )
                    })
                    .transpose()?,
                q_norm,
                k_norm,
                n_heads: cfg.n_heads,
                n_kv_heads: cfg.n_kv_heads,
                head_dim: cfg.head_dim,
            };
            let attn_norm =
                RmsNorm::from_weights(backend, t(&p("attn_norm.weight"))?, cfg.norm_eps)?;
            let ffn_norm = RmsNorm::from_weights(backend, t(&p("ffn_norm.weight"))?, cfg.norm_eps)?;
            // ffn: gate/up fused [2·ffn, D] (нативный квант), down [D, ffn]
            let gn = p("ffn_gate.weight");
            let un = p("ffn_up.weight");
            let ffn = SwiGLU {
                w13: Self::upload_fused(backend, w, &[&gn, &un], vec![2 * cfg.ffn_dim, cfg.dim])?,
                w2: Self::upload_q(
                    backend,
                    w,
                    &p("ffn_down.weight"),
                    vec![cfg.dim, cfg.ffn_dim],
                )?,
            };
            layers.push(Qwen3Layer {
                attn_norm,
                attn,
                ffn_norm,
                ffn,
            });
            // N (iter-10): KV in f16 by default — halves the attention
            // DRAM traffic (the decode bottleneck) and the cache VRAM; the
            // f16 kv_update + attn_ar_v4_f16f16 kernels are tested
            // (kv_f16.rs). F32 stays available via KvFormat for reference
            // runs.
            caches.push(KvCache::new(
                backend,
                cfg.n_kv_heads,
                cfg.head_dim,
                crate::dtype::KvFormat::F16,
                crate::dtype::KvFormat::F16,
                cfg.rope_theta,
                cfg.mrope_sections,
            )?);
        }
        // N (iter-10): bound the KV to the model's context window — the
        // default (None) grew without limit on long conversations (or
        // errored mid-reply when an operator set max_cap manually). The
        // cap errors LOUDLY at the window edge (compaction would corrupt
        // absolute RoPE positions — documented in set_max_cap).
        for c in &mut caches {
            c.set_max_cap(Some(cfg.rope_max_seq));
        }

        let out_norm = RmsNorm::from_weights(backend, t("output_norm.weight")?, cfg.norm_eps)?;
        // tied: output.weight может делить стораж с token_embd
        let output = match w.get("output.weight") {
            Some(_) if !w.tied_aliases("output.weight").is_empty() => {
                upload_w(backend, &embed_w, vec![cfg.vocab_size, cfg.dim])?
            }
            Some(_) => Self::upload_q(backend, w, "output.weight", vec![cfg.vocab_size, cfg.dim])?,
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
        backend.begin_batch()?;
        let r = (|| {
            let t = ids.len();
            let x = self.embed.forward(backend, ids, t)?;
            let logits = self.forward_layers(backend, x, t, None, None)?;
            self.pos_rope = self.caches[0].pos;
            Ok(logits)
        })();
        backend.end_batch()?;
        r
    }

    /// Prefill + скрытые extract-слоёв (для DFlash-инжекта, Ф2.4): h_i —
    /// выход слоя i (attn+ffn+residual, без out_norm), по порядку `layers`.
    /// Один проход: кэши и логиты как в prefill, скрытые — побочный сбор.
    pub fn prefill_extract(
        &mut self,
        backend: &dyn Backend,
        ids: &[u32],
        layers: &[usize],
    ) -> Result<(Tensor, Vec<Tensor>), BackendError> {
        backend.begin_batch()?;
        let r = (|| {
            let t = ids.len();
            let mut x = self.embed.forward(backend, ids, t)?;
            let mut out = Vec::with_capacity(layers.len());
            let mut li = 0usize;
            let (mut p_attn, mut p_ffn) = (0.0f64, 0.0f64);
            let prof = std::env::var("LLM_BENCH").is_ok();
            // N (iter-10): one causal mask per forward — identical for every
            // layer (each cache's pos is the same at its own update).
            // N (plan 1.6): roll the window BEFORE the current token's rope
            // and the mask — the compact resets the absolute positions, and
            // the current token must be roped at its NEW position (the
            // update-time roll would rope it at the pre-roll position).
            if self.caches[0].roll_if_needed(backend, t)? {
                self.pos_rope = self.caches[0].pos;
            }
            let mask = if t > 1 {
                Some(crate::nn::build_causal_mask(
                    backend,
                    t,
                    self.caches[0].pos + t,
                    self.pos_rope,
                    self.cfg.swa_window,
                )?)
            } else {
                None
            };
            for (i, layer) in self.layers.iter().enumerate() {
                // фичи для DFlash-инжекта = ВХОД extract-слоя (llama.cpp
                // llama_get_embeddings_layer_inp: выход предыдущего слоя)
                if li < layers.len() && layers[li] == i {
                    out.push(x.clone());
                    li += 1;
                }
                let t0 = std::time::Instant::now();
                if prof {
                    backend.end_batch()?;
                    backend.begin_batch()?;
                }
                let h = layer.attn_norm.forward(backend, &x)?;
                let h = Self::rope_and_attn(
                    &self.cfg,
                    backend,
                    &h,
                    &layer.attn,
                    &mut self.caches[i],
                    self.pos_rope,
                    t,
                    None,
                    true,
                    None,
                    mask.as_ref(),
                )?;
                x = backend_add(backend, &x, &h)?;
                if prof {
                    backend.end_batch()?;
                    backend.begin_batch()?;
                }
                if prof {
                    p_attn += t0.elapsed().as_secs_f64();
                }
                let t1 = std::time::Instant::now();
                let h = layer.ffn_norm.forward(backend, &x)?;
                let f = ffn_forward(backend, &h, &layer.ffn)?;
                x = backend_add(backend, &x, &f)?;
                if prof {
                    backend.end_batch()?;
                    backend.begin_batch()?;
                }
                if prof {
                    p_ffn += t1.elapsed().as_secs_f64();
                }
            }
            if prof {
                eprintln!(
                    "PROF prefill_extract t={}: attn {:.1}ms ({:.2}/layer) ffn {:.1}ms ({:.2}/layer)",
                    t,
                    p_attn * 1e3,
                    p_attn * 1e3 / self.layers.len() as f64,
                    p_ffn * 1e3,
                    p_ffn * 1e3 / self.layers.len() as f64
                );
            }
            let (b, tq, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
            let h = self.out_norm.forward(backend, &x)?;
            let h2 = h.view(vec![b * tq, d]);
            let logits = backend.zeros(vec![b * tq, self.cfg.vocab_size])?;
            backend.gemm_nt(&h2, &self.output, &logits)?;
            self.pos_rope = self.caches[0].pos;
            Ok((logits.view(vec![b, tq, self.cfg.vocab_size]), out))
        })();
        backend.end_batch()?;
        r
    }

    /// Prefill с vision-эмбеддингами (Ф1.11): токены == `vision_token`
    /// заменяются строками vision-энкодера (по порядку появления) — всё на
    /// GPU (embedding_override). `deepstack` — блоки vision-энкодера
    /// (VisionQwen3Vl::forward, 3 × [1, n_img, dim]; пустой срез = нет);
    /// после слоёв 0..n_deepstack-1 они добавляются к скрытым состояниям
    /// image-позиций (HF _deepstack_process). `pos4d` — M-RoPE позиции
    /// (t,h,w,e) на токен (len = t*4; текст: t=h=w=pos, e=0; картинка:
    /// t=pos_0, h=pos_0+i/nx, w=pos_0+i%nx, e=0). `pos_after` — логическая
    /// позиция (секция t) следующего decode (после картинки pos_0+24,
    /// а не +2304).
    pub fn prefill_with_vision(
        &mut self,
        backend: &dyn Backend,
        ids: &[u32],
        vision: &Tensor,
        deepstack: &[Tensor],
        vision_token: u32,
        pos4d: &[u32],
        pos_after: usize,
    ) -> Result<Tensor, BackendError> {
        backend.begin_batch()?;
        let r = (|| {
            let t = ids.len();
            assert_eq!(pos4d.len(), t * 4, "pos4d: {t} токенов × 4");
            let x = backend.zeros(vec![1, t, self.cfg.dim])?;
            backend.embedding_override(ids, &self.embed.table, vision, &x, vision_token)?;
            // Выровнять deepstack-блоки по токен-позициям: фичи на строках
            // image-токенов, нули вне их — полное сложение x += aligned[il]
            // тогда совпадает с HF (на текстовые токены deepstack не
            // действует; llama.cpp кладёт в буфере нули).
            let ds_aligned = self.align_deepstack(backend, ids, deepstack, vision_token)?;
            let ds = ds_aligned.as_deref();
            let logits = self.forward_layers(backend, x, t, Some(pos4d), ds)?;
            self.pos_rope = pos_after;
            Ok(logits)
        })();
        backend.end_batch()?;
        r
    }

    /// Выровнять deepstack-блоки: блок [1, n_img, dim] → [1, T, dim], где
    /// строки image-токенов (ids == vision_token) содержат фичи блока (по
    /// порядку), остальные — нули. Размерность T нужна, чтобы одно сложение
    /// x += aligned шло по всем токен-позициям сразу.
    fn align_deepstack(
        &self,
        backend: &dyn Backend,
        ids: &[u32],
        deepstack: &[Tensor],
        vision_token: u32,
    ) -> Result<Option<Vec<Tensor>>, BackendError> {
        if deepstack.is_empty() {
            return Ok(None);
        }
        let dim = self.cfg.dim;
        let t = ids.len();
        let img_rows: Vec<usize> = (0..t)
            .filter(|&k| ids[k] == vision_token)
            .collect();
        if img_rows.is_empty() {
            return Err(BackendError::Shape(
                "deepstack: нет image-токенов в ids (vision_token не найден)".into(),
            ));
        }
        let n_img = img_rows.len();
        eprintln!("deepstack: img_rows={} t={} vt={}", img_rows.len(), t, vision_token);
        let mut out = Vec::with_capacity(deepstack.len());
        for (il, block) in deepstack.iter().enumerate() {
            // Блоки приходят 2D [nq, dim] либо 3D [1, nq, dim].
            let sh = block.shape();
            let (b, n, d) = match sh.len() {
                2 => (1usize, sh[0], sh[1]),
                3 => (sh[0], sh[1], sh[2]),
                _ => return Err(BackendError::Shape(format!(
                    "deepstack[{il}]: неожиданная форма {sh:?}"
                ))),
            };
            if d != dim {
                return Err(BackendError::Shape(format!(
                    "deepstack[{il}]: dim {d} != {dim}"
                )));
            }
            if b * n != n_img {
                return Err(BackendError::Shape(format!(
                    "deepstack[{il}]: {b}×{n} строк != n_image_tokens {n_img}"
                )));
            }
            let feats = backend.to_cpu_f32(block)?;
            let mut padded = vec![0.0f32; t * dim];
            for (k, &row) in img_rows.iter().enumerate() {
                padded[row * dim..(row + 1) * dim]
                    .copy_from_slice(&feats[k * dim..(k + 1) * dim]);
            }
            out.push(backend.from_cpu_f32(&padded, vec![1, t, dim])?);
        }
        Ok(Some(out))
    }

    /// Скрытые состояния после `upto` слоёв (None — все) + out_norm.
    /// Для DFlash-драфтеров (conditioning на hidden states) и сверки
    /// DraftModel-среза с полной моделью. `deepstack` — выровненные
    /// deepstack-блоки ([1, T, dim], Some только в vision-пути).
    fn forward_layers_hidden(
        &mut self,
        backend: &dyn Backend,
        mut x: Tensor,
        t: usize,
        mrope: Option<&[u32]>,
        upto: Option<usize>,
        deepstack: Option<&[Tensor]>,
    ) -> Result<Tensor, BackendError> {
        let pos = if mrope.is_some() { 0 } else { self.pos_rope };
        if cfg!(debug_assertions) && std::env::var("INFER_LLM_DEBUG").is_ok() {
            let xv = backend.to_cpu_f32(&x)?;
            eprintln!("INFER_DEBUG embed[0,:8]={:?}", &xv[..8]);
        }
        let mut prof_attn = 0.0f64;
        let mut prof_ffn = 0.0f64;
        let prof = std::env::var("LLM_BENCH").is_ok();
        // Конфиг обещает deepstack-слои (Qwen3-VL), а блоков нет (пустой
        // срез от caller'а / vision-энкодера) — молча пропущенное сложение
        // искажает hidden states image-позиций. Громкое предупреждение один
        // раз на forward (Q3_NO_DS — осознанное отключение, без варнинга).
        if mrope.is_some()
            && std::env::var("Q3_NO_DS").is_err()
            && deepstack.map_or(true, |d| d.is_empty())
        {
            if let Some(n) = self.cfg.n_deepstack {
                if n > 0 {
                    tracing::warn!(
                        n_deepstack = n,
                        "mrope/vision forward: config expects {n} deepstack blocks, \
                         none provided — vision hidden states will miss the deepstack adds"
                    );
                }
            }
        }
        // N (iter-10): one mask per forward (see the other loops).
        let mask = if let Some(mrope) = mrope {
            if t > 1 {
                Some(build_mrope_mask(backend, t, self.caches[0].pos + t, mrope)?)
            } else {
                None
            }
        } else if t > 1 {
            Some(crate::nn::build_causal_mask(
                backend,
                t,
                self.caches[0].pos + t,
                pos,
                self.cfg.swa_window,
            )?)
        } else {
            None
        };
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(u) = upto {
                if i >= u {
                    break;
                }
            }
            let t0 = std::time::Instant::now();
            let h = layer.attn_norm.forward(backend, &x)?;
            if std::env::var("INFER_LLM_DEBUG").is_ok() && i == 0 {
                let hv = backend.to_cpu_f32(&h)?;
                eprintln!("INFER_DEBUG norm[0,:8]={:?}", &hv[..8]);
            }
            let h = Self::rope_and_attn(
                &self.cfg,
                backend,
                &h,
                &layer.attn,
                &mut self.caches[i],
                pos,
                t,
                mrope,
                true,
                None,
                mask.as_ref(),
            )?;
            x = backend_add(backend, &x, &h)?;
            if std::env::var("INFER_LLM_DEBUG").is_ok() && i == 0 {
                let xv = backend.to_cpu_f32(&x)?;
                let first = xv.len() - t * self.cfg.dim;
                eprintln!(
                    "INFER_DEBUG t={t} after_attn_layer0[0,:8]={:?}",
                    &xv[first..first + 8]
                );
                eprintln!(
                    "INFER_DEBUG t={t} after_attn_layer0[-1,:8]={:?}",
                    &xv[xv.len() - self.cfg.dim..xv.len() - self.cfg.dim + 8]
                );
            }
            if prof {
                prof_attn += t0.elapsed().as_secs_f64();
            }
            if std::env::var("OMNI_REF").is_ok() && i == 0 {
                let xv = backend.to_cpu_f32(&x)?;
                let last = xv.len() - self.cfg.dim;
                eprintln!(
                    "OMNI_REF after_attn_layer0[-1,:8]={:?}",
                    &xv[last..last + 8]
                );
            }
            let t1 = std::time::Instant::now();
            let h = layer.ffn_norm.forward(backend, &x)?;
            let f = ffn_forward(backend, &h, &layer.ffn)?;
            x = backend_add(backend, &x, &f)?;
            if prof {
                prof_ffn += t1.elapsed().as_secs_f64();
            }
            if std::env::var("OMNI_REF").is_ok() && i == 0 {
                let xv = backend.to_cpu_f32(&x)?;
                let last = xv.len() - self.cfg.dim;
                eprintln!("OMNI_REF after_ffn_layer0[-1,:8]={:?}", &xv[last..last + 8]);
            }
            // deepstack (Qwen3-VL): после слоя il < N к скрытым состояниям
            // image-позиций добавляется блок il (HF _deepstack_process:
            // hidden[visual_pos] += ds[layer]; llama.cpp qwen3vl.cpp: cur +=
            // view t_inp_embd — per-token компоненты [main|ds0|ds1|ds2], для
            // текстовых токенов нулевые). Блоки выровнены по токен-позициям
            // в prefill_with_vision (нули вне image-строк) — полное сложение.
            // Старый вариант `x[t] += embd_in[t + il + 1]` сдвигал на il+1
            // ТОКЕН эмбеддингов текста — не deepstack-фичи (ошибка).
            if mrope.is_some() && std::env::var("Q3_NO_DS").is_err() {
                if let Some(n_ds) = self.cfg.n_deepstack {
                    if i < n_ds {
                        if let Some(ds) = deepstack.and_then(|d| d.get(i)) {
                            x = backend.deepstack(&x, ds, 0)?;
                        }
                    }
                }
            }
        }
        if prof && t > 1 {
            eprintln!(
                "PROF layers={} t={}: attn {:.1}ms ffn {:.1}ms (per-layer {:.2}/{:.2}ms)",
                self.layers.len(),
                t,
                prof_attn * 1e3,
                prof_ffn * 1e3,
                prof_attn * 1e3 / self.layers.len() as f64,
                prof_ffn * 1e3 / self.layers.len() as f64
            );
        }
        let h = self.out_norm.forward(backend, &x)?;
        Ok(h)
    }

    /// Общий путь слоёв после эмбеддинга (текст или vision-текст).
    /// `deepstack` — выровненные deepstack-блоки (только vision-путь).
    fn forward_layers(
        &mut self,
        backend: &dyn Backend,
        x: Tensor,
        t: usize,
        mrope: Option<&[u32]>,
        deepstack: Option<&[Tensor]>,
    ) -> Result<Tensor, BackendError> {
        let (b, tq, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let h = self.forward_layers_hidden(backend, x, t, mrope, None, deepstack)?;
        let h2 = h.view(vec![b * tq, d]);
        let logits = backend.zeros(vec![b * tq, self.cfg.vocab_size])?;
        backend.gemm_nt(&h2, &self.output, &logits)?;
        Ok(logits.view(vec![b, tq, self.cfg.vocab_size]))
    }

    /// Скрытые состояния после всех слоёв (без lm_head) — для драфтеров.
    pub fn forward_hidden(
        &mut self,
        backend: &dyn Backend,
        ids: &[u32],
    ) -> Result<Tensor, BackendError> {
        backend.begin_batch()?;
        let r = (|| {
            let t = ids.len();
            let x = self.embed.forward(backend, ids, t)?;
            let h = self.forward_layers_hidden(backend, x, t, None, None, None)?;
            Ok(h)
        })();
        backend.end_batch()?;
        r
    }

    /// Скрытые после указанных слоёв (для DFlash-инжекта, Ф2.4): h_i — выход
    /// слоя i (attn+ffn+residual, БЕЗ out_norm), по порядку `layers`.
    /// Кэши пополняются как при обычном префилле.
    pub fn forward_extract(
        &mut self,
        backend: &dyn Backend,
        ids: &[u32],
        layers: &[usize],
    ) -> Result<Vec<Tensor>, BackendError> {
        backend.begin_batch()?;
        let r = (|| {
            let t = ids.len();
            let mut x = self.embed.forward(backend, ids, t)?;
            let mut out = Vec::with_capacity(layers.len());
            let mut li = 0usize;
            // N (iter-10): one causal mask per forward (see the other loops).
            // N (plan 1.6): roll the window BEFORE the current token's rope
            // and the mask — the compact resets the absolute positions, and
            // the current token must be roped at its NEW position (the
            // update-time roll would rope it at the pre-roll position).
            if self.caches[0].roll_if_needed(backend, t)? {
                self.pos_rope = self.caches[0].pos;
            }
            let mask = if t > 1 {
                Some(crate::nn::build_causal_mask(
                    backend,
                    t,
                    self.caches[0].pos + t,
                    self.pos_rope,
                    self.cfg.swa_window,
                )?)
            } else {
                None
            };
            for (i, layer) in self.layers.iter().enumerate() {
                // фичи для DFlash-инжекта = ВХОД extract-слоя (llama.cpp
                // llama_get_embeddings_layer_inp: выход предыдущего слоя)
                if li < layers.len() && layers[li] == i {
                    out.push(x.clone());
                    li += 1;
                }
                let h = layer.attn_norm.forward(backend, &x)?;
                let h = Self::rope_and_attn(
                    &self.cfg,
                    backend,
                    &h,
                    &layer.attn,
                    &mut self.caches[i],
                    self.pos_rope,
                    t,
                    None,
                    true,
                    None,
                    mask.as_ref(),
                )?;
                x = backend_add(backend, &x, &h)?;
                let h = layer.ffn_norm.forward(backend, &x)?;
                let f = ffn_forward(backend, &h, &layer.ffn)?;
                x = backend_add(backend, &x, &f)?;
            }
            Ok(out)
        })();
        backend.end_batch()?;
        r
    }

    /// Decode: один токен → logits [1, 1, V]; KV-кэш растёт.
    /// Вероятность токена в softmax-распределении строки.
    fn softmax_prob(row: &[f32], tok: u32) -> f32 {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for &l in row {
            sum += (l - max).exp();
        }
        ((row[tok as usize] - max).exp()) / sum
    }

    /// Честное rejection-sampling (Leviathan et al.): сэмпл из (p − q)+,
    /// где q — масса драфтера на draft-токене.
    fn sample_modified(row: &[f32], draft_tok: u32, q: f32, rng: &mut Option<u64>) -> u32 {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut w: Vec<f32> = row.iter().map(|&l| (l - max).exp()).collect();
        let sum_exp: f32 = w.iter().sum();
        // вычесть q-массу (в exp-масштабе) с draft-позиции; p' = (p − q)+
        w[draft_tok as usize] -= q * sum_exp;
        let mut total = 0.0f32;
        for v in w.iter_mut() {
            *v = v.max(0.0);
            total += *v;
        }
        if total <= 0.0 {
            return draft_tok; // вырожденный случай — масса целиком в draft
        }
        let mut r = Self::uniform01(rng) * total;
        for (i, &v) in w.iter().enumerate() {
            r -= v;
            if r <= 0.0 {
                return i as u32;
            }
        }
        (w.len() - 1) as u32
    }

    /// Равномерное U(0,1) — тот же LCG, что в inference::sampling
    /// (state = None → сид из часов, пишется обратно в state).
    fn uniform01(rng: &mut Option<u64>) -> f32 {
        match rng {
            Some(state) => {
                *state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (*state >> 33) as f32 / (1u64 << 33) as f32
            }
            None => {
                let nanos = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .subsec_nanos();
                let mut st = nanos as u64;
                st = st
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *rng = Some(st);
                (st >> 33) as f32 / (1u64 << 33) as f32
            }
        }
    }

    /// Спекулятивный decode (Ф2.1 каркас): батч-верификация draft одним
    /// forward на [id, draft...], честное rejection-sampling, KV-откат
    /// непринятого хвоста. `draft_probs` — вероятности драфтера q_i.
    /// `extract_layers` — собрать скрытые этих слоёв (DFlash-инжект Ф2.4;
    /// пустой — без сбора).
    ///
    /// Верификация draft[i] — по распределению на позиции ПЕРЕД ним (row i:
    /// row 0 — после id, row k — после draft[k-1]); при отклонении пересэмпл
    /// из той же строки. Возвращаемые логиты — row accepted_len-1: строка,
    /// соответствующая состоянию кэша после отката (при полном принятии —
    /// последняя, продолжение).
    pub fn decode_spec(
        &mut self,
        backend: &dyn Backend,
        id: u32,
        draft: &[u32],
        draft_probs: &[f32],
        rng: &mut Option<u64>,
        extract_layers: &[usize],
    ) -> Result<SpecOutcome, BackendError> {
        assert_eq!(draft.len(), draft_probs.len());
        let mut ids = Vec::with_capacity(1 + draft.len());
        ids.push(id);
        ids.extend_from_slice(draft);
        // 1. ОДИН forward: кэш пополняется всеми (1+γ) токенами.
        let (logits, extract) = self.prefill_extract(backend, &ids, extract_layers)?; // [1, T, V]
        let v = logits.shape()[2];
        let ld = backend.to_cpu_f32(&logits)?;
        // 2. Честное rejection-sampling (Leviathan et al., алгоритм 2):
        // draft[i] принимается с вероятностью min(1, p/q), где p — softmax
        // таргета на draft[i], q — вероятность драфтера. Раньше сравнение
        // было детерминированным (q <= p → принять) — не сэмплирование.
        let mut accepted: Vec<u32> = Vec::new();
        let mut accepted_len = 1usize;
        let mut n_verified = 0usize;
        for i in 0..draft.len() {
            let row = &ld[i * v..(i + 1) * v];
            let p = Self::softmax_prob(row, draft[i]);
            let q = draft_probs[i];
            let u = Self::uniform01(rng);
            if q <= 0.0 || u * q <= p {
                accepted.push(draft[i]);
                accepted_len += 1;
                n_verified += 1;
            } else {
                accepted.push(Self::sample_modified(row, draft[i], q, rng));
                break;
            }
        }
        // 3. KV-откат непринятого хвоста: остаются prefill-база + id + принятые.
        if accepted_len < ids.len() {
            let base = self.caches[0].pos - ids.len();
            for c in &mut self.caches {
                c.truncate(base + accepted_len);
            }
            self.pos_rope = base + accepted_len;
        }
        // Логиты позиции accepted_len-1 — СООТВЕТСТВУЮТ откаченному кэшу
        // (распределение после последнего принятого токена; при полном
        // принятии — последняя строка, продолжение после всех draft).
        let row_idx = accepted_len - 1;
        let row = &ld[row_idx * v..(row_idx + 1) * v];
        let logits = backend.from_cpu_f32(row, vec![1, 1, v])?;
        let n_draft_accepted = accepted_len.saturating_sub(1);
        Ok(SpecOutcome {
            accepted,
            logits,
            extract,
            n_draft_accepted,
            n_draft_verified: n_verified,
        })
    }

    pub fn decode(&mut self, backend: &dyn Backend, id: u32) -> Result<Tensor, BackendError> {
        // Все ~290 кернелов шага — в ОДИН command buffer (1 commit+wait
        // вместо 290 синхронных dispatch — decode был в ~70× медленнее
        // bandwidth-потолка из-за per-op sync).
        backend.begin_batch()?;
        let r = (|| {
            let x = self.embed.forward(backend, &[id], 1)?;
            let logits = self.forward_layers(backend, x, 1, None, None)?;
            self.pos_rope += 1;
            Ok(logits)
        })();
        backend.end_batch()?;
        r
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
    /// текст после картинки — каузально по t. `causal` — false для диффузии
    /// DFlash (кэш и draft-токены видят друг друга, маски нет).
    /// `cache_at` — запись k/v в кэш на АБСОЛЮТНУЮ позицию (DFlash: кэш
    /// индексирован позициями токенов); None — append (cache.pos).
    pub fn rope_and_attn(
        cfg: &Qwen3Config,
        backend: &dyn Backend,
        h: &Tensor,
        attn: &Attention,
        cache: &mut KvCache,
        pos: usize,
        t: usize,
        mrope: Option<&[u32]>,
        causal: bool,
        cache_at: Option<usize>,
        // N (iter-10): the causal/mrope mask is IDENTICAL for every layer
        // (each layer's cache.pos equals the same value at its own update),
        // so the caller builds it ONCE per forward — the old per-layer build
        // re-filled and re-uploaded the same mask 36× (a 2048-token prefill
        // = 36 × 16 MB H2D).
        mask: Option<&Tensor>,
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
        let qkv = if let Some(bias) = &attn.qkv_bias {
            // Qwen2.5-Omni thinker: q/k/v проекции с bias (fused [3D']).
            if cfg!(debug_assertions) && std::env::var("OMNI_REF").is_ok() {
                let bv = backend.to_cpu_f32(bias)?;
                eprintln!("OMNI_REF our_q_bias[:8]={:?}", &bv[..8.min(bv.len())]);
                let qv = backend.to_cpu_f32(&qkv)?;
                eprintln!("OMNI_REF qkv_before_bias[:8]={:?}", &qv[..8.min(qv.len())]);
            }
            backend.add_bias(&qkv, bias, &qkv)?;
            if cfg!(debug_assertions) && std::env::var("OMNI_REF").is_ok() {
                let qv = backend.to_cpu_f32(&qkv)?;
                let qs = q_size + 2 * kv_size;
                let last = qv.len() - qs;
                eprintln!("OMNI_REF qkv_after[-1,:8]={:?}", &qv[last..last + 8]);
            }
            qkv
        } else {
            if cfg!(debug_assertions) && std::env::var("OMNI_REF").is_ok() {
                eprintln!("OMNI_REF qkv_bias=None!!!");
            }
            qkv
        };
        let q = crate::nn::slice_last(backend, &qkv, q_size, 0)?;
        let k = crate::nn::slice_last(backend, &qkv, kv_size, q_size)?;
        let v = crate::nn::slice_last(backend, &qkv, kv_size, q_size + kv_size)?;
        if cfg!(debug_assertions) && std::env::var("INFER_LLM_DEBUG").is_ok() {
            for (tag, t) in [("q", &q), ("k", &k), ("v", &v)] {
                let tv = backend.to_cpu_f32(t)?;
                eprintln!("INFER_DEBUG {tag}_slice[:8]={:?}", &tv[..8.min(tv.len())]);
            }
        }
        if cfg!(debug_assertions) && std::env::var("OMNI_REF").is_ok() {
            let qv = backend.to_cpu_f32(&q)?;
            let kvv = backend.to_cpu_f32(&k)?;
            let hv = backend.to_cpu_f32(h)?;
            let t = tq;
            eprintln!(
                "OMNI_REF our_ln[-1,:8]={:?}",
                &hv[(t - 1) * q_size..(t - 1) * q_size + 8]
            );
            eprintln!(
                "OMNI_REF our_q0[-1,:8]={:?}",
                &qv[(t - 1) * q_size..(t - 1) * q_size + 8]
            );
            eprintln!(
                "OMNI_REF our_k0[-1,:8]={:?}",
                &kvv[(t - 1) * kv_size..(t - 1) * kv_size + 8]
            );
        }
        // НОРМАЛИЗАЦИЯ — ТОЛЬКО ОДНА, В ВЕТКАХ НИЖЕ. Раньше здесь стоял
        // пред-норм qk_norm, а затем fused rmsnorm_head_rope / qk_norm_pub
        // нормировали q/k ВТОРОЙ раз: 2× норм на реальном GGUF (Qwen3
        // 0.6B/8B) давал вырожденный вывод (повтор «|وووو…»). Ветки
        // сами нормируют: else — rmsnorm_head_rope, mrope — qk_norm_pub.
        let (q, k) = if let Some(mrope) = mrope {
            let q = qk_norm_pub(backend, &q, attn.q_norm.as_ref(), attn.n_heads, 1e-6)?;
            let k = qk_norm_pub(backend, &k, attn.k_norm.as_ref(), attn.n_kv_heads, 1e-6)?;
            let sects = cfg.mrope_sections.expect("mrope без sections");
            (
                backend.rope_mrope(&q, mrope, attn.head_dim, cfg.rope_theta, sects)?,
                backend.rope_mrope(&k, mrope, attn.head_dim, cfg.rope_theta, sects)?,
            )
        } else if let Some(sects) = cfg.mrope_sections {
            let q = qk_norm_pub(backend, &q, attn.q_norm.as_ref(), attn.n_heads, 1e-6)?;
            let k = qk_norm_pub(backend, &k, attn.k_norm.as_ref(), attn.n_kv_heads, 1e-6)?;
            (
                backend.rope_mrope_text(&q, pos, attn.head_dim, cfg.rope_theta, sects)?,
                backend.rope_mrope_text(&k, pos, attn.head_dim, cfg.rope_theta, sects)?,
            )
        } else {
            // fused qk_norm+rope (Ф8): 2 кернела на слой вместо 4; дефолт
            // (CPU/CUDA) — два шага
            let q = match &attn.q_norm {
                Some(w) => {
                    let q = backend.rmsnorm_head_rope(
                        &q,
                        w,
                        pos,
                        attn.head_dim,
                        cfg.rope_theta,
                        attn.n_heads,
                        1e-6,
                    )?;
                    if cfg!(debug_assertions) && std::env::var("INFER_LLM_DEBUG").is_ok() {
                        let qv = backend.to_cpu_f32(&q)?;
                        eprintln!("INFER_DEBUG q0_normed[:8]={:?}", &qv[..8.min(qv.len())]);
                    }
                    q
                }
                None => rope_standard(backend, &q, pos, t, attn.head_dim, cfg.rope_theta)?,
            };
            let k = match &attn.k_norm {
                Some(w) => backend.rmsnorm_head_rope(
                    &k,
                    w,
                    pos,
                    attn.head_dim,
                    cfg.rope_theta,
                    attn.n_kv_heads,
                    1e-6,
                )?,
                None => rope_standard(backend, &k, pos, t, attn.head_dim, cfg.rope_theta)?,
            };
            (q, k)
        };
        // attention напрямую через attn_ar (q/k уже повёрнуты стандартным RoPE)
        // q/k/v 3D [1, T, D]; attn_ar сам разложит по головам
        let t_upd = std::time::Instant::now();
        match cache_at {
            Some(at) => cache.update_at(backend, &k, &v, at)?,
            None => cache.update(backend, &k, &v)?,
        }
        let t_upd_d = t_upd.elapsed().as_secs_f64();
        // N (iter-10): mask passed in from the caller (built once per
        // forward); the `causal`/`mrope` arguments are kept for the call
        // sites' None-vs-Some decision, but the build itself moved out.
        let mask = mask.map(|m| m.clone());
        let ck = cache.view();
        let cv = cache.view_v();
        let attn_out = backend.zeros(vec![b, tq, q_size])?;
        let t_at = std::time::Instant::now();
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
        if cfg!(debug_assertions) && std::env::var("LLM_BENCH").is_ok() && tq == 1 {
            eprintln!(
                "  attn_ar {:.2}ms kv_update {:.2}ms",
                t_at.elapsed().as_secs_f64() * 1e3,
                t_upd_d * 1e3
            );
        }

        let out = Linear {
            w: attn.wo.clone(),
            b: None,
        }
        .forward(backend, &attn_out)?;
        Ok(out)
    }
}

/// QK-norm: per-head RMSNorm (w [dim], применяется по hd каждой головы).
pub fn qk_norm_pub(
    backend: &dyn Backend,
    x: &Tensor,
    w: Option<&Tensor>,
    heads: usize,
    eps: f32,
) -> Result<Tensor, BackendError> {
    let y = backend.zeros(x.shape().to_vec())?;
    match w {
        Some(w) => backend.rmsnorm_head(x, w, &y, heads, eps)?,
        None => return Ok(x.clone()),
    }
    Ok(y)
}

/// M-RoPE attention mask: каузальность по t-позициям (секция 0). Токены
/// картинки имеют равные t → видят всю картинку; текст после картинки
/// (t больше) видит всё до себя. tkv — размер кэша (cache.pos, ПОСЛЕ
/// записи текущих токенов); `tq` последних слотов — текущие токены,
/// их t-значения — pos4d. Слоты ДО них (прошлые ходы, j < tkv - tq) —
/// старый контекст: их сохранённые t недоступны, поэтому безопасный
/// фолбэк — считать их текстом (t = 0, секция 0): 0 ≤ ti для любых
/// текущих ti → старые слоты видны всем текущим токенам (каузальность
/// по порядку следования сохраняется).
fn build_mrope_mask(
    backend: &dyn Backend,
    tq: usize,
    tkv: usize,
    pos4d: &[u32],
) -> Result<Tensor, BackendError> {
    let cache_start = tkv.saturating_sub(tq); // слот первого текущего токена
    let mut data = vec![0.0f32; tq * tkv];
    for i in 0..tq {
        let ti = pos4d[i * 4];
        for j in 0..tkv {
            // j-й слот кэша: текущий токен — свой t из pos4d; старый — 0.
            let tj = if j >= cache_start {
                pos4d[(j - cache_start) * 4]
            } else {
                0
            };
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
            return Err(BackendError::Shape(format!(
                "upload_w: numel mismatch shape={shape:?} data={}",
                data.len()
            )));
        }
        let f16: Vec<half::f16> = data.iter().map(|&x| half::f16::from_f32(x)).collect();
        backend.from_cpu_f16(&f16, shape)
    } else {
        backend.from_cpu_f32(data, shape)
    }
}

/// Стандартный rotate-half RoPE (HF/Qwen3) через backend-оп
/// (Metal — кернел rope_std_f32; CPU — референс-вычисление).
pub fn rope_standard(
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

/// FFN (Qwen3-VL llama.cpp build_ffn, LLM_FFN_SILU + LLM_FFN_PAR):
/// silu применяется к GATE-проекции, up остаётся линейным:
///   gm = silu(x @ ffn_gate) * (x @ ffn_up) → down.
/// В GGUF Qwen3-VL ffn_gate/ffn_up лежат именно в этом порядке
/// (проверено: python-эталон совпал с llama.cpp при silu(gate)*up, corr 1.0).
pub fn ffn_forward(
    backend: &dyn Backend,
    x: &Tensor,
    ffn: &SwiGLU,
) -> Result<Tensor, BackendError> {
    let (b, t, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    let x2 = x.view(vec![b * t, d]);
    let ffn_dim = ffn.w2.shape()[1];
    let f = backend.zeros(vec![b * t, 2 * ffn_dim])?;
    let gm = backend.zeros(vec![b * t, ffn_dim])?;
    // fused gate|up GEMM + silu_mul: 2 кернела вместо 4 (w1+w3 gemm, silu, mul)
    backend.gemm_nt(&x2, &ffn.w13, &f)?;
    backend.silu_mul_fused(&f, &gm, ffn_dim)?;
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
            swa_window: None,
            mrope_sections: None,
            n_deepstack: None,
        }
    }

    /// Мини-Qwen3: случайные веса; WeightsMap строится поверх owned-буферов.
    fn mini_model() -> (MetalBackend, Qwen3Model) {
        let backend = MetalBackend::new().expect("metal backend");
        let c = cfg();
        mini_model_with_cfg(c)
    }

    fn mini_model_with_cfg(c: Qwen3Config) -> (MetalBackend, Qwen3Model) {
        let backend = MetalBackend::new().expect("metal backend");
        let cfg = c.clone(); // входной конфиг (n_layers среза!) — не переопределять cfg()
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
    /// Ф2.1: spec с пустым draft (γ=0) == обычный decode (бит-в-бит).
    #[test]
    fn decode_spec_empty_matches_decode() {
        let (backend, mut model) = mini_model();
        let ids = [1u32, 2, 3, 4];
        model.prefill(&backend, &ids).unwrap();
        let d1 = model.decode(&backend, 5).unwrap();

        model.reset_cache(&backend).unwrap();
        model.prefill(&backend, &ids).unwrap();
        let mut rng = None;
        let out = model
            .decode_spec(&backend, 5, &[], &[], &mut rng, &[])
            .unwrap();
        assert!(out.accepted.is_empty(), "γ=0: draft-токенов нет");
        let a = backend.to_cpu_f32(&d1).unwrap();
        let b = backend.to_cpu_f32(&out.logits).unwrap();
        assert_eq!(a, b, "spec(γ=0) должен совпадать с decode");
    }

    /// Ф2.1: честное rejection-sampling — draft отклоняется, когда случайный
    /// u > p/q (сид выбран так, что первый u > p токена 3 при q=1), KV
    /// откатывается — следующий decode совпадает с обычным decode после
    /// того же prefill.
    #[test]
    fn decode_spec_rejects_and_rolls_back() {
        let (backend, mut model) = mini_model();
        let ids = [1u32, 2, 3, 4];
        // p3 — вероятность draft-токена 3 в распределении проверки (row 0:
        // после id=5; γ=0-spec возвращает именно эту строку логитов).
        model.prefill(&backend, &ids).unwrap();
        let mut rng = None;
        let probe = model
            .decode_spec(&backend, 5, &[], &[], &mut rng, &[])
            .unwrap();
        let pv = backend.to_cpu_f32(&probe.logits).unwrap();
        let p3 = Qwen3Model::softmax_prob(&pv, 3);
        // Детерминированный сид: первый draw LCG > p3 → u·q = u > p —
        // гарантированное отклонение (при q=1 принятие ⇔ u ≤ p).
        let seed = (1u64..)
            .find(|&s| {
                let st = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (st >> 33) as f32 / (1u64 << 33) as f32 > p3
            })
            .unwrap();

        model.reset_cache(&backend).unwrap();
        model.prefill(&backend, &ids).unwrap();
        let l1 = model.decode(&backend, 5).unwrap();
        let l2 = model.decode(&backend, 6).unwrap();

        model.reset_cache(&backend).unwrap();
        model.prefill(&backend, &ids).unwrap();
        let mut rng = Some(seed);
        let out = model
            .decode_spec(&backend, 5, &[3], &[1.0], &mut rng, &[])
            .unwrap();
        assert_eq!(out.accepted.len(), 1, "мусорный draft отклонён");
        assert_ne!(
            out.accepted[0], 3,
            "отклонён → пересэмпл ≠ draft (q=1 вычитает его массу)"
        );
        // KV откачен (остался id=5) — decode(6) даёт обычный результат.
        let l2s = model.decode(&backend, 6).unwrap();
        let a = backend.to_cpu_f32(&l2).unwrap();
        let b = backend.to_cpu_f32(&l2s).unwrap();
        assert_eq!(a, b, "после отката decode должен совпадать с обычным");
        let _ = l1;
    }

    /// Честное RS: при q < p принятие детерминировано (min(1, p/q) > 1) —
    /// draft-токен принимается при любом u, кэш не откатывается.
    #[test]
    fn decode_spec_accepts_when_q_below_p() {
        let (backend, mut model) = mini_model();
        let ids = [1u32, 2, 3, 4];
        model.prefill(&backend, &ids).unwrap();
        let mut rng = None;
        let probe = model
            .decode_spec(&backend, 5, &[], &[], &mut rng, &[])
            .unwrap();
        let pv = backend.to_cpu_f32(&probe.logits).unwrap();
        let a = pv
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        let p = Qwen3Model::softmax_prob(&pv, a);
        let q = p * 0.5; // min(1, p/q) = 2 → принятие при любом u < 1

        model.reset_cache(&backend).unwrap();
        model.prefill(&backend, &ids).unwrap();
        let mut rng = Some(0xDEAD_BEEF);
        let out = model
            .decode_spec(&backend, 5, &[a], &[q], &mut rng, &[])
            .unwrap();
        assert_eq!(out.accepted, vec![a], "q < p → гарантированное принятие");
        assert_eq!(out.n_draft_accepted, 1);
        // Полное принятие: кэш не откатывался (база + id + draft).
        assert_eq!(model.caches[0].pos, ids.len() + 2);
    }

    /// Ф2.2: DraftModel = Qwen3Model с n_layers=k — его forward_hidden
    /// бит-в-бит совпадает со скрытыми полной модели после k слоёв.
    #[test]
    fn draft_slice_matches_full_hidden() {
        let (backend, full) = mini_model();
        let k = 2usize; // срез первых k слоёв (cfg() даёт 2 слоя)
        let mut cfg = full.cfg.clone();
        cfg.n_layers = k;
        let (_, mut draft) = mini_model_with_cfg(cfg);
        let ids = [1u32, 2, 3, 4];
        // полная модель — скрытые после k слоёв (кэш пуст — как у DraftModel;
        // через batch-путь, как forward_hidden — исключить sync-vs-batch)
        let h_full = {
            let mut m = full;
            let x = m.embed_forward_test(&backend, &ids).unwrap();
            let t = ids.len();
            backend.begin_batch().unwrap();
            let r = m
                .forward_layers_hidden(&backend, x, t, None, Some(k), None)
                .unwrap();
            backend.end_batch().unwrap();
            r
        };
        // DraftModel(k) — скрытые после k слоёв
        let h_draft = {
            let mut m = draft;
            m.forward_hidden(&backend, &ids).unwrap()
        };
        let a = backend.to_cpu_f32(&h_full).unwrap();
        let b = backend.to_cpu_f32(&h_draft).unwrap();
        assert_eq!(a, b, "срез k слоёв должен совпадать с полной моделью");
    }

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
        let a = backend
            .rope_mrope(&xt, &text_pos, 4, 5_000_000.0, sects)
            .unwrap();
        let b = backend
            .rope_mrope_text(&xt, 1, 4, 5_000_000.0, sects)
            .unwrap();
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
        let c = backend
            .rope_mrope(&xt, &img_pos, 4, 5_000_000.0, sects)
            .unwrap();
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
        let qv: Vec<f32> = (0..b * tq * d)
            .map(|_| (rng() >> 32) as f32 / (1u64 << 32) as f32 - 0.5)
            .collect();
        let kv: Vec<f32> = (0..b * tkv * kvd)
            .map(|_| (rng() >> 32) as f32 / (1u64 << 32) as f32 - 0.5)
            .collect();
        let vv: Vec<f32> = (0..b * tkv * kvd)
            .map(|_| (rng() >> 32) as f32 / (1u64 << 32) as f32 - 0.5)
            .collect();
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
        backend
            .attn_ar(&q, &k, &v, Some(&mask), &y, h, kvh, hd)
            .unwrap();
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
            assert!(
                dlt < 1e-3,
                "idx {i}: got {} want {} delta {}",
                yv[i],
                want[i],
                dlt
            );
        }
        eprintln!("attn_ar_matches_cpu_reference OK ({} elems)", yv.len());
    }
}
