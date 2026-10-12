//! GLM-4 (`glm4`) — волна A2 плана архитектур: порт llama.cpp
//! `src/models/glm4.cpp` (эталонный коммит 5266f24da).
//!
//! Отличия от Qwen3 (`qwen3.rs`), которые и делают граф другим:
//! 1. **Частичный RoPE**: `glm4.rope.dimension_count` (64 при head_dim 128) —
//!    вращаются только первые n_rot компонент головы
//!    (`Backend::rope_standard_partial`), хвост копируется;
//! 2. **Биасы q/k/v** (`attn_{q,k,v}.bias`, F32) — добавляются после проекции;
//! 3. **Sandwich-нормы**: `post_attention_norm` применяется к выходу внимания
//!    ДО прибавления residual, `post_ffw_norm` — к выходу FFN до residual;
//! 4. **Fused `ffn_up` [n_embd, 2·n_ff]**: первая половина — gate (к ней
//!    silu), вторая — up. Это ровно наш `nn::SwiGLU.w13` (тая же раскладка:
//!    `silu(X[i])·X[ffn+i]`), поэтому вес берётся как есть — без разрезания и
//!    без потери нативного кванта;
//! 5. Выходная проекция без tied-веса: `output.weight` есть в файле, но при
//!    его отсутствии берём таблицу эмбеддингов (как llama.cpp).
//!
//! Граф слоя (glm4.cpp): attn_norm → qkv → rope → attention → wo →
//! post_attention_norm → +residual → ffn_norm → swiglu → post_ffw_norm →
//! +residual. Затем output_norm → output.

use crate::backend::{Backend, Tensor};
use crate::error::BackendError;
use crate::gguf::{GgufFile, GgufValue};
use crate::inference::generate::CausalLm;
use crate::nn::{KvCache, Linear, RmsNorm, SwiGLU, build_causal_mask};
use crate::weights_map::{WeightsMap, dequant_f32};

/// Шаг текущего forward для GLM4_DUMP (0 = префилл, 1 = декод).
static DUMP_STEP: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

/// Конфигурация GLM-4.
#[derive(Debug, Clone)]
pub struct Glm4Config {
    pub arch: String,
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// Сколько компонент головы вращает RoPE (`rope.dimension_count`).
    pub n_rot: usize,
    pub ffn_dim: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
    pub eps: f32,
    pub ctx: usize,
}

impl Glm4Config {
    /// Архитектура семейства, которую читает этот модуль. Легаси `chatglm`
    /// (ChatGLM-3) намеренно НЕ принимаем: там fused qkv/up — другая
    /// раскладка весов, и молчаливо прочитанная как glm4 модель дала бы
    /// мусор; `unsupported_arch` говорит об этом явно.
    pub fn is_glm4_arch(arch: &str) -> bool {
        arch == "glm4"
    }

    pub fn from_gguf(file: &GgufFile<'_>, vocab_size: usize) -> Result<Self, BackendError> {
        let arch = file.architecture().unwrap_or("glm4").to_string();
        if !Self::is_glm4_arch(&arch) {
            return Err(crate::arch::unsupported_arch(&arch));
        }
        let p = |s: &str| format!("{arch}.{s}");
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
        let head_dim = u(&p("attention.key_length")).unwrap_or(dim / n_heads.max(1));
        // Частичный RoPE: без ключа — полный (n_rot = head_dim).
        let n_rot = file
            .get(&p("rope.dimension_count"))
            .and_then(GgufValue::as_u32)
            .map(|v| v as usize)
            .unwrap_or(head_dim);
        if n_rot > head_dim || !n_rot.is_multiple_of(2) {
            return Err(BackendError::Shape(format!(
                "glm4: rope.dimension_count {n_rot} несогласован с head_dim {head_dim}"
            )));
        }
        let rope_theta = file
            .get(&p("rope.freq_base"))
            .and_then(|v| match v {
                GgufValue::F32(f) => Some(*f),
                GgufValue::F64(f) => Some(*f as f32),
                _ => None,
            })
            .unwrap_or(10_000.0);
        let eps = file
            .get(&p("attention.layer_norm_rms_epsilon"))
            .and_then(|v| match v {
                GgufValue::F32(f) => Some(*f),
                GgufValue::F64(f) => Some(*f as f32),
                _ => None,
            })
            .unwrap_or(1e-5);
        let ctx = file
            .get(&p("context_length"))
            .and_then(GgufValue::as_u32)
            .map(|v| v as usize)
            .unwrap_or(8192);
        Ok(Self {
            arch,
            dim,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            n_rot,
            ffn_dim,
            vocab_size,
            rope_theta,
            eps,
            ctx,
        })
    }

    /// KV-формат: как у Qwen3 — f16 по умолчанию, `INFER_KV_F32` возвращает f32.
    pub fn kv_format(&self) -> crate::dtype::KvFormat {
        if std::env::var("INFER_KV_F32").is_ok() {
            crate::dtype::KvFormat::F32
        } else {
            crate::dtype::KvFormat::F16
        }
    }
}

struct Glm4Layer {
    attn_norm: RmsNorm,
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    wo: Tensor,
    q_bias: Option<Tensor>,
    k_bias: Option<Tensor>,
    v_bias: Option<Tensor>,
    post_attn_norm: RmsNorm,
    ffn_norm: RmsNorm,
    /// w13 = fused `ffn_up` [gate; up] как есть; w2 = `ffn_down`.
    ffn: SwiGLU,
    post_ffw_norm: RmsNorm,
}

/// GLM-4 (dense): attention с частичным RoPE и биасами + sandwich-нормы.
pub struct Glm4Model {
    pub cfg: Glm4Config,
    /// Таблица эмбеддингов: квантованная (Q4_K/Q6_K/…) — тогда поиск идёт
    /// через `embedding_quant` (dequant-in-kernel), иначе f16/f32 и
    /// обычный `embedding`. У GLM-4-9B таблица в Q4_K: `nn::Embedding`
    /// (float-путь) читал бы её как float и отдавал нули.
    embed: Tensor,
    embed_quant: bool,
    /// f32-копия таблицы эмбеддингов — по требованию (драфтер/PP читают
    /// f32-таблицу; квантованную он прочитать не может).
    embed_f32: Option<Tensor>,
    layers: Vec<Glm4Layer>,
    out_norm: RmsNorm,
    output: Tensor,
    pub caches: Vec<KvCache>,
}

/// Векторные веса (биасы q/k/v): грузим как **f32**, не f16 — кернел
/// `add_bias_f32` читает буфер как `float*`, и f16-загрузка (которую делает
/// `upload_w` на бэкендах с f16-весами) давала мусор вместо биаса.
fn t_f32(wm: &WeightsMap<'_>, backend: &dyn Backend, name: &str) -> Result<Tensor, BackendError> {
    let wt = wm
        .get(name)
        .ok_or_else(|| BackendError::Shape(format!("missing tensor {name}")))?;
    let shape: Vec<usize> = wt.shape.iter().map(|&s| s as usize).collect();
    backend.from_cpu_f32(&dequant_f32(wt), shape)
}

impl Glm4Model {
    pub fn load(
        backend: &dyn Backend,
        wm: &WeightsMap<'_>,
        cfg: Glm4Config,
    ) -> Result<Self, BackendError> {
        let dim = cfg.dim;
        let kd = cfg.n_kv_heads * cfg.head_dim;
        let qd = cfg.n_heads * cfg.head_dim;
        let t_q = |name: &str| -> Result<Tensor, BackendError> {
            crate::qwen3::Qwen3Model::upload_q(backend, wm, name, vec![qd, dim])
        };
        let t_kv = |name: &str| -> Result<Tensor, BackendError> {
            crate::qwen3::Qwen3Model::upload_q(backend, wm, name, vec![kd, dim])
        };
        let embed_quantized = wm
            .get("token_embd.weight")
            .map(|t| t.format.is_quantized())
            .unwrap_or(false);
        let embed = if embed_quantized {
            crate::qwen3::Qwen3Model::upload_q(
                backend,
                wm,
                "token_embd.weight",
                vec![cfg.vocab_size, dim],
            )?
        } else {
            t_f32(wm, backend, "token_embd.weight")?
        };
        // tied-output: отдельного output.weight может не быть (llama.cpp
        // подставляет таблицу эмбеддингов).
        let output = match wm.get("output.weight") {
            Some(_) => crate::qwen3::Qwen3Model::upload_q(
                backend,
                wm,
                "output.weight",
                vec![cfg.vocab_size, dim],
            )?,
            None => t_f32(wm, backend, "token_embd.weight")?,
        };
        let out_norm = RmsNorm::from_weights(backend, dequant_f32(wm.get("output_norm.weight").ok_or_else(
            || BackendError::Shape("missing output_norm.weight".into()),
        )?), cfg.eps)?;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        let mut caches = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = |s: &str| format!("blk.{i}.{s}");
            let norm = |name: &str| -> Result<RmsNorm, BackendError> {
                RmsNorm::from_weights(backend, dequant_f32(wm.get(name).ok_or_else(|| {
                    BackendError::Shape(format!("missing tensor {name}"))
                })?), cfg.eps)
            };
            let bias = |name: &str, n: usize| -> Result<Option<Tensor>, BackendError> {
                match wm.get(name) {
                    Some(_) => Ok(Some(t_f32(wm, backend, name)?)),
                    None => {
                        let _ = n;
                        Ok(None)
                    }
                }
            };
            // fused ffn_up: [dim, 2·ffn] → наш w13 [2·ffn, dim] (тая же раскладка
            // «первая половина = gate»), вес берётся нативным квантом.
            let w13 = crate::qwen3::Qwen3Model::upload_q(
                backend,
                wm,
                &p("ffn_up.weight"),
                vec![2 * cfg.ffn_dim, dim],
            )?;
            let w2 = crate::qwen3::Qwen3Model::upload_q(
                backend,
                wm,
                &p("ffn_down.weight"),
                vec![dim, cfg.ffn_dim],
            )?;
            layers.push(Glm4Layer {
                attn_norm: norm(&p("attn_norm.weight"))?,
                wq: t_q(&p("attn_q.weight"))?,
                wk: t_kv(&p("attn_k.weight"))?,
                wv: t_kv(&p("attn_v.weight"))?,
                wo: t_q(&p("attn_output.weight"))?,
                q_bias: bias(&p("attn_q.bias"), qd)?,
                k_bias: bias(&p("attn_k.bias"), kd)?,
                v_bias: bias(&p("attn_v.bias"), kd)?,
                post_attn_norm: norm(&p("post_attention_norm.weight"))?,
                ffn_norm: norm(&p("ffn_norm.weight"))?,
                ffn: SwiGLU { w13, w2 },
                post_ffw_norm: norm(&p("post_ffw_norm.weight"))?,
            });
            caches.push(KvCache::new(
                backend,
                cfg.n_kv_heads,
                cfg.head_dim,
                cfg.kv_format(),
                cfg.kv_format(),
                cfg.rope_theta,
                None,
            )?);
        }
        Ok(Self {
            cfg,
            embed,
            embed_quant: embed_quantized,
            embed_f32: None,
            layers,
            out_norm,
            output,
            caches,
        })
    }

    /// Таблицы, которые делит с таргетом DFlash-драфтер: (эмбеддинги f32,
    /// выходная проекция). У GLM таблица эмбеддингов квантована (Q4_K у
    /// GLM-4-9B) — драфтер читает f32-таблицу, поэтому собираем копию по
    /// требованию и кэшируем.
    pub fn shared_tables(
        &mut self,
        backend: &dyn Backend,
        wm: &WeightsMap<'_>,
    ) -> Result<(Tensor, Tensor), BackendError> {
        if self.embed_f32.is_none() {
            let emb = wm
                .get("token_embd.weight")
                .ok_or_else(|| BackendError::Shape("missing tensor token_embd.weight".into()))?;
            let rows = emb.shape.first().copied().unwrap_or(0) as usize;
            self.embed_f32 = Some(backend.from_cpu_f32(
                &dequant_f32(emb),
                vec![rows, self.cfg.dim],
            )?);
        }
        let embed = self.embed_f32.clone().expect("embed_f32 собран выше");
        Ok((embed, self.output.clone()))
    }

    fn add_bias_opt(
        backend: &dyn Backend,
        x: &Tensor,
        bias: Option<&Tensor>,
    ) -> Result<Tensor, BackendError> {
        match bias {
            None => Ok(x.clone()),
            Some(b) => {
                let y = backend.zeros(x.shape().to_vec())?;
                backend.add_bias(x, b, &y)?;
                Ok(y)
            }
        }
    }

    fn dbg_on() -> bool {
        std::env::var("GLM4_DEBUG").is_ok()
    }

    /// Побитовой дамп чекпоинтов слоя 0 в формате oracle_dump.cpp
    /// (`GLM4_DUMP=<file>`, только m=1): имена — как в графе эталона
    /// (attn_norm-0, post_attn_norm-0, ffn_inp-0, ffn_norm-0, ffn_up-0,
    /// ffn_swiglu-0, ffn_out-0, post_mlp_norm-0).
    fn dump_path() -> Option<String> {
        std::env::var("GLM4_DUMP").ok()
    }

    fn dump(backend: &dyn Backend, path: &str, name: &str, x: &Tensor) -> Result<(), BackendError> {
        let step = DUMP_STEP.load(std::sync::atomic::Ordering::Relaxed);
        // GLM4_DUMP_WATCH — фильтр подстрок (через запятую): на длинном m
        // полный дамп 321 чекпоинта это гигабайты; бисект пишет только нужное.
        if let Ok(w) = std::env::var("GLM4_DUMP_WATCH") {
            if !w.split(',').any(|s| !s.is_empty() && name.contains(s)) {
                return Ok(());
            }
        }
        let v = backend.to_cpu_f32(&x.view_restricted(vec![x.numel()]))?;
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| BackendError::Native(format!("GLM4_DUMP: {e}")))?;
        writeln!(f, "STEP {step} TENSOR {name} ne=[{},1,1,1]", v.len())
            .map_err(|e| BackendError::Native(format!("GLM4_DUMP: {e}")))?;
        for (i, val) in v.iter().enumerate() {
            write!(f, "{:08x}{}", val.to_bits(), if i % 8 == 7 { "\n" } else { " " })
                .map_err(|e| BackendError::Native(format!("GLM4_DUMP: {e}")))?;
        }
        writeln!(f).map_err(|e| BackendError::Native(format!("GLM4_DUMP: {e}")))?;
        Ok(())
    }

    /// Печать первых значений тензора (отладка постадийная, GLM4_DEBUG=1).
    fn dbg(backend: &dyn Backend, tag: &str, x: &Tensor) -> Result<(), BackendError> {
        if !Self::dbg_on() {
            return Ok(());
        }
        let v = backend.to_cpu_f32(&x.view_restricted(vec![8.min(x.numel())]))?;
        eprintln!("GLM4 {tag}: {:?}", &v[..8.min(v.len())]);
        Ok(())
    }

    /// Полный forward: ids → логиты [1, T, V].
    /// `last_tail` — как оркал (llama.cpp 0.4.0, GLM-4-9B m=12): последний
    /// слой срезается на последний токен СРАЗУ после внимания (kqv), дальше
    /// wo/FFN/norm считаются при m=1 (mv-путь). Без среза хвост идёт
    /// mm-кернелом на m=T и логиты не равны побитово. Вызывается из
    /// prefill_last; обычный prefill — false (все строки).
    fn forward(
        &mut self,
        backend: &dyn Backend,
        ids: &[u32],
        last_tail: bool,
    ) -> Result<Tensor, BackendError> {
        let t = ids.len();
        let pos = self.caches[0].pos;
        // Дамп чекпоинтов — только первого forward (шаг 0 эталона);
        // GLM4_DUMP_DECODE=1 добавляет шаг декода (STEP 1) — бисект Ф8.1g.
                DUMP_STEP.store(if pos == 0 { 0 } else { 1 }, std::sync::atomic::Ordering::Relaxed);
        let dump = Self::dump_path().filter(|_| {
            if std::env::var("GLM4_DUMP_DECODE").is_ok() {
                pos > 0
            } else {
                pos == 0
            }
        });
        let t_state = pos;
        let (nh, nkv, hd, dim) = (
            self.cfg.n_heads,
            self.cfg.n_kv_heads,
            self.cfg.head_dim,
            self.cfg.dim,
        );
        let (qd, kd) = (nh * hd, nkv * hd);
        // Плотная маска [Tq, Tkv]: как в qwen3.rs — у нас нет T-независимого
        // окна, маска строится на forward (блочный префилл — prefill_chunk).
        let mask = if t > 1 {
            Some(build_causal_mask(backend, t, pos + t, pos, None)?)
        } else {
            None
        };
        let mut x = backend.zeros(vec![1, t, dim])?;
        if self.embed_quant {
            backend.embedding_quant(ids, &self.embed, &x)?;
        } else {
            backend.embedding(ids, &self.embed, &x)?;
        }
        for (i, layer) in self.layers.iter().enumerate() {
            let inp = x.clone();
            if i == 0 {
                Self::dbg(backend, "embed", &x)?;
            }
            let h = layer.attn_norm.forward(backend, &x)?;
            if i == 0 {
                Self::dbg(backend, "attn_norm", &h)?;
            }
            if let Some(p) = &dump {
                Self::dump(backend, p, &format!("attn_norm-{i}"), &h)?;
            }
            let q = Linear {
                w: layer.wq.clone(),
                b: None,
            }
            .forward(backend, &h)?;
            let k = Linear {
                w: layer.wk.clone(),
                b: None,
            }
            .forward(backend, &h)?;
            let v = Linear {
                w: layer.wv.clone(),
                b: None,
            }
            .forward(backend, &h)?;
            if i == 0 {
                Self::dbg(backend, "q_pre", &q)?;
            }
            if let Some(p) = &dump {
                Self::dump(backend, p, "Qcur-0", &q)?;
            }
            let q = Self::add_bias_opt(backend, &q, layer.q_bias.as_ref())?;
            let k = Self::add_bias_opt(backend, &k, layer.k_bias.as_ref())?;
            let v = Self::add_bias_opt(backend, &v, layer.v_bias.as_ref())?;
            if i == 0 {
                Self::dbg(backend, "q_bias", &q)?;
                Self::dbg(backend, "k_bias", &k)?;
                Self::dbg(backend, "v_bias", &v)?;
            }
            debug_assert_eq!(q.shape(), &[1, t, qd]);
            debug_assert_eq!(k.shape(), &[1, t, kd]);
            // Частичный RoPE (n_rot = rope.dimension_count): вращаются первые
            // n_rot компонент каждой головы.
            // GLM-4 = llama.cpp ROPE_TYPE_NORM: пары СОСЕДНИЕ (2k, 2k+1),
            // частоты theta^(-2k/n_rot) — не halves, как у Qwen3.
            let q = backend.rope_partial(
                &q,
                pos,
                hd,
                self.cfg.n_rot,
                self.cfg.rope_theta,
                crate::backend::RopePairing::Adjacent,
            )?;
            let k = backend.rope_partial(
                &k,
                pos,
                hd,
                self.cfg.n_rot,
                self.cfg.rope_theta,
                crate::backend::RopePairing::Adjacent,
            )?;
            if i == 0 {
                Self::dbg(backend, "q_rope", &q)?;
                Self::dbg(backend, "k_rope", &k)?;
            }
            if let Some(p) = &dump {
                Self::dump(backend, p, "Kcur-0", &k)?;
            }
            self.caches[i].update(backend, &k, &v)?;
            if i == 0 && Self::dbg_on() {
                Self::dbg(backend, "cache_k", &self.caches[i].view())?;
                Self::dbg(backend, "cache_v", &self.caches[i].view_v())?;
                eprintln!(
                    "GLM4 cache: pos={} k_shape={:?} v_shape={:?} mask={:?}",
                    self.caches[i].pos,
                    self.caches[i].view().shape(),
                    self.caches[i].view_v().shape(),
                    mask.as_ref().map(|m| m.shape().to_vec())
                );
            }
            let attn = backend.zeros(vec![1, t, qd])?;
            backend.attn_ar(
                &q,
                &self.caches[i].view(),
                &self.caches[i].view_v(),
                mask.as_ref(),
                &attn,
                nh,
                nkv,
                hd,
            )?;
            if i == 0 {
                Self::dbg(backend, "attn_out", &attn)?;
            }
            if let Some(p) = &dump {
                Self::dump(backend, p, "kqv_out-0", &attn)?;
            }
            let o = Linear {
                w: layer.wo.clone(),
                b: None,
            }
            .forward(backend, &attn)?;
            if i == 0 {
                Self::dbg(backend, "after_wo", &o)?;
            }
            if let Some(p) = &dump {
                Self::dump(backend, p, "l_out-0", &o)?;
            }
            // Ф8.1e: срез хвоста последнего слоя (см. док forward). Оркал
            // режет ПОСЛЕ wo (l_out-39 = [4096,1], а wo ещё на m=12 mm):
            // post_attn_norm/FFN/norm — m=1, mv-путь.
            let slice_tail = last_tail && t > 1 && i + 1 == self.cfg.n_layers;
            let (o, inp) = if slice_tail {
                (
                    o.view(vec![1, t, dim]).view_offset(vec![1, 1, dim], (t - 1) * dim),
                    inp.view(vec![1, t, dim]).view_offset(vec![1, 1, dim], (t - 1) * dim),
                )
            } else {
                (o, inp)
            };
            let tt = if slice_tail { 1 } else { t };
            // Sandwich: post-attention norm ДО residual.
            let o = layer.post_attn_norm.forward(backend, &o)?;
            if let Some(p) = &dump {
                Self::dump(backend, p, &format!("post_attn_norm-{i}"), &o)?;
            }
            let y = backend.zeros(vec![1, tt, dim])?;
            backend.add(&o, &inp, &y)?;
            let f = layer.ffn_norm.forward(backend, &y)?;
            if let Some(p) = &dump {
                Self::dump(backend, p, &format!("ffn_inp-{i}"), &y)?;
                Self::dump(backend, p, &format!("ffn_norm-{i}"), &f)?;
            }
            let f = if let Some(p) = &dump {
                // Дамп-вариант ffn: те же три шага, что ffn_forward, с
                // промежуточными чекпоинтами эталона (up/swiglu/down).
                let (b, t2, d) = (f.shape()[0], f.shape()[1], f.shape()[2]);
                let x2 = f.view(vec![b * t2, d]);
                let ffn_dim = layer.ffn.w2.shape()[1];
                let fu = backend.zeros(vec![b * t2, 2 * ffn_dim])?;
                backend.gemm_nt(&x2, &layer.ffn.w13, &fu)?;
                Self::dump(backend, p, &format!("ffn_up-{i}"), &fu)?;
                let gm = backend.zeros(vec![b * t2, ffn_dim])?;
                backend.silu_mul_fused(&fu, &gm, ffn_dim)?;
                Self::dump(backend, p, &format!("ffn_swiglu-{i}"), &gm)?;
                let yo = backend.zeros(vec![b * t2, d])?;
                backend.gemm_nt(&gm, &layer.ffn.w2, &yo)?;
                Self::dump(backend, p, &format!("ffn_out-{i}"), &yo)?;
                yo.view(vec![b, t2, d])
            } else {
                crate::qwen3::ffn_forward(backend, &f, &layer.ffn)?
            };
            let f = layer.post_ffw_norm.forward(backend, &f)?;
            if i == 0 {
                Self::dbg(backend, "ffn_out", &f)?;
            }
            if let Some(p) = &dump {
                Self::dump(backend, p, &format!("post_mlp_norm-{i}"), &f)?;
            }
            let z = backend.zeros(vec![1, tt, dim])?;
            backend.add(&f, &y, &z)?;
            x = z;
            if std::env::var("GLM4_DEBUG_L").is_ok() {
                // Первые 4 значения ПОСЛЕДНЕЙ строки (позиция pos+t-1) — для
                // сверки с постадийным эталоном (glm_probe).
                let n: usize = x.numel();
                let view = x.view_restricted(vec![n]).view_offset(vec![4], n - dim);
                let v = backend.to_cpu_f32(&view)?;
                eprintln!("OURSL x[{}][{i}]: {:?}", t_state + t - 1, v);
            }
        }
        let h = self.out_norm.forward(backend, &x)?;
        if let Some(p) = &dump {
            Self::dump(backend, p, "result_norm", &h)?;
        }
        // При срезе хвоста (last_tail) x имеет 1 строку — логиты тоже [1,1,V].
        let tx = x.shape()[1];
        let h2 = h.view(vec![tx, dim]);
        let logits = backend.zeros(vec![tx, self.cfg.vocab_size])?;
        backend.gemm_nt(&h2, &self.output, &logits)?;
        Ok(logits.view(vec![1, tx, self.cfg.vocab_size]))
    }

    /// Размер блока префилла — как у Qwen3 (маска плотная [Tq, Tkv]).
    pub(crate) fn prefill_chunk() -> usize {
        crate::qwen3::Qwen3Model::prefill_chunk()
    }
}

impl CausalLm for Glm4Model {
    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }
    fn n_layers(&self) -> usize {
        self.cfg.n_layers
    }
    fn arch(&self) -> &str {
        &self.cfg.arch
    }
    fn context_len(&self) -> usize {
        self.caches.first().map(|c| c.pos).unwrap_or(0)
    }
    fn reset_cache(&mut self, backend: &dyn Backend) -> Result<(), BackendError> {
        for c in &mut self.caches {
            c.reset();
        }
        let _ = backend;
        Ok(())
    }

    fn prefill(&mut self, backend: &dyn Backend, ids: &[u32]) -> Result<Tensor, BackendError> {
        backend.begin_batch()?;
        let r = self.forward(backend, ids, false);
        backend.end_batch()?;
        r
    }

    fn prefill_last(&mut self, backend: &dyn Backend, ids: &[u32]) -> Result<Tensor, BackendError> {
        // Блочный префилл с логитами только последней позиции (как qwen3).
        // last_tail=true — хвост последнего слоя срезается на последний
        // токен (паритет с оркалом: wo/FFN при m=1, mv-путь).
        let chunk = Self::prefill_chunk();
        let mut done = 0usize;
        let mut last: Option<Tensor> = None;
        while done < ids.len() {
            let n = chunk.min(ids.len() - done);
            let part = &ids[done..done + n];
            backend.begin_batch()?;
            let logits = self.forward(backend, part, true);
            backend.end_batch()?;
            let logits = logits?;
            if done + n == ids.len() {
                last = Some(self.slice_last_row(backend, &logits)?);
            }
            done += n;
        }
        last.ok_or_else(|| BackendError::Shape("prefill_last: пустой промпт".into()))
    }

    fn prefill_extract(
        &mut self,
        backend: &dyn Backend,
        ids: &[u32],
        layers: &[usize],
    ) -> Result<(Tensor, Vec<Tensor>), BackendError> {
        // Пустой список слоёв — обычный промпт (generate зовёт этот метод
        // всегда, а extract-слои нужны только DFlash-драфтеру). Для glm4
        // драфтера нет: непустой список — явная ошибка, а не пустые фичи.
        if layers.is_empty() {
            let logits = self.prefill_last(backend, ids)?;
            return Ok((logits, Vec::new()));
        }
        Err(BackendError::UnsupportedFormat(format!(
            "{}: extract-слои (DFlash-инжест) не поддержаны — драфтера для glm4 нет",
            self.cfg.arch
        )))
    }

    fn decode(&mut self, backend: &dyn Backend, id: u32) -> Result<Tensor, BackendError> {
        backend.begin_batch()?;
        let r = self.forward(backend, &[id], false);
        backend.end_batch()?;
        r
    }

    fn decode_spec(
        &mut self,
        backend: &dyn Backend,
        id: u32,
        draft: &[u32],
        _draft_probs: &[f32],
        _rng: &mut Option<u64>,
        _extract_layers: &[usize],
        greedy: bool,
        _hist: &[u32],
        _repeat_penalty: f32,
    ) -> Result<crate::qwen3::SpecOutcome, BackendError> {
        // Оконной верификации (B8/DFlash) для glm4 пока нет: проверяем
        // черновики ПОСЛЕДОВАТЕЛЬНО (γ декодов). Текст идентичен жадному,
        // выигрыша по скорости нет — честный фолбэк вместо ошибки.
        if !greedy {
            return Err(BackendError::UnsupportedFormat(
                "glm4: спек-верификация поддержана только для жадного режима".into(),
            ));
        }
        let mut tokens = Vec::with_capacity(draft.len() + 1);
        let mut ids = vec![id];
        ids.extend_from_slice(draft);
        let mut accepted = 0usize;
        let mut outcome_logits = None;
        let mut truncated_at = None;
        for (i, &tok) in ids.iter().enumerate() {
            let logits = self.decode(backend, tok)?;
            let want = draft.get(i);
            let (argmax, _probs) = backend.argmax_rows(&logits, None)?;
            let got = argmax.first().copied().unwrap_or(0);
            match want {
                Some(w) if w == &got => {
                    accepted += 1;
                    tokens.push(*w);
                }
                _ => {
                    tokens.push(got);
                    outcome_logits = Some(logits);
                    truncated_at = Some(self.context_len());
                    break;
                }
            }
        }
        // Откат хвоста: принятые черновики остаются в кэше, отвергнутый —
        // нет (его мы уже посчитали, поэтому усекаем до позиции приёма).
        if let Some(p) = truncated_at {
            for c in &mut self.caches {
                c.truncate(p);
            }
        }
        let logits = outcome_logits.ok_or_else(|| {
            BackendError::Shape("glm4 spec: все черновики приняты (ре-сэмпл не нужен) — \
                                кэш уже содержит принятые токены".into())
        })?;
        Ok(crate::qwen3::SpecOutcome {
            accepted: tokens,
            logits,
            extract: Vec::new(),
            n_draft_accepted: accepted,
            n_draft_verified: accepted,
        })
    }

    fn lm_head(&self, backend: &dyn Backend, h: &Tensor) -> Result<Tensor, BackendError> {
        let (b, tq, d) = (h.shape()[0], h.shape()[1], h.shape()[2]);
        let h2 = h.view(vec![b * tq, d]);
        let logits = backend.zeros(vec![b * tq, self.cfg.vocab_size])?;
        backend.gemm_nt(&h2, &self.output, &logits)?;
        Ok(logits.view(vec![b, tq, self.cfg.vocab_size]))
    }
}

impl Glm4Model {
    /// Логиты только последней строки ([1, 1, V]).
    fn slice_last_row(&self, backend: &dyn Backend, logits: &Tensor) -> Result<Tensor, BackendError> {
        let t = logits.shape()[1];
        let v = logits.shape()[2];
        let view = logits.view_restricted(vec![t * v]);
        let rows = backend.to_cpu_f32(&view)?;
        let last = &rows[(t - 1) * v..];
        if std::env::var("GLM4_TOPK").is_ok() {
            let mut idx: Vec<usize> = (0..v).collect();
            idx.sort_unstable_by(|&a, &b| last[b].partial_cmp(&last[a]).unwrap());
            let top: Vec<(usize, f32)> = idx[..5].iter().map(|&i| (i, last[i])).collect();
            eprintln!("GLM4 top5 логитов: {top:?}");
        }
        Ok(backend.from_cpu_f32(last, vec![1, 1, v])?)
    }
}
