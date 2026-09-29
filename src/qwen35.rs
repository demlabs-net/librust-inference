//! Qwen3.5 / Qwen3.8 — гибридная архитектура (Ф9.4): слои линейного
//! внимания Gated DeltaNet чередуются с полным gated-attention.
//!
//! Эталон — llama.cpp `src/models/qwen35.cpp` + `delta-net-base.cpp` (и CPU-опы
//! ggml `ssm_conv` / `gated_delta_net`). Слой `il` рекуррентный, если
//! `(il + 1) % full_attention_interval != 0` (или по явному массиву
//! `attention.recurrent_layers`); у Qwen3.5-4B полное внимание — слои 3, 7, …, 31.
//!
//! Блок: `x → attn_norm → (GDN | attention) → +x → post_attention_norm →
//! SwiGLU → +residual`. Полное внимание: q-проекция выдаёт на голову
//! `[q (hd) | gate (hd)]`, q/k нормируются по голове, IMROPE с частичной
//! ротацией (64 из 256), выход attention умножается на `sigmoid(gate)`.
//! GDN: `attn_qkv` → causal conv1d (K=4, стейт K-1 токенов) → silu → q/k/v;
//! q/k — L2-норма; гейты `g = softplus(alpha + dt_bias)·ssm_a`,
//! `beta = sigmoid(beta)`; рекуррентный delta rule со стейтом `[Hv, S, S]`;
//! гейтовая RMS-норма с `z = attn_gate·x`; `ssm_out`.
//!
//! Состояние сессии — KV-кэши полных слоёв + conv/ssm-стейты рекуррентных.
//! В отличие от KV, рекуррентный стейт НЕ откатывается усечением: операции,
//! которым нужен откат (спекулятивная верификация, PP-стадии, перенос
//! KV-объекта), для гибрида явно отвергаются вызывающим кодом.

use crate::backend::{Backend, Tensor};
use crate::error::BackendError;
use crate::gguf::{GgufFile, GgufValue};
use crate::nn::{KvCache, Linear, RmsNorm, SwiGLU};
use crate::qwen3::{Qwen3Model, ffn_forward};
use crate::weights_map::WeightsMap;

/// Конфигурация гибрида из GGUF-метадаты (`qwen35.*` / `qwen35moe.*` без MoE).
#[derive(Debug, Clone)]
pub struct Qwen35Config {
    pub arch: String,
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
    /// Частичная ротация: число вращаемых измерений головы (64 из 256).
    pub n_rot: usize,
    pub rope_sections: [usize; 4],
    pub norm_eps: f32,
    pub max_seq: usize,
    /// GDN: ядро conv1d (4), размер стейта головы (128), k-голов (group_count),
    /// v-голов (time_step_rank), ширина v (inner_size).
    pub conv_kernel: usize,
    pub ssm_state: usize,
    pub ssm_k_heads: usize,
    pub ssm_v_heads: usize,
    pub ssm_inner: usize,
    /// По слою: true — линейное внимание (GDN), false — полное.
    pub recurrent: Vec<bool>,
}

impl Qwen35Config {
    /// Архитектуры, которые этот модуль исполняет.
    pub fn is_hybrid_arch(arch: &str) -> bool {
        matches!(arch, "qwen35" | "qwen3next")
    }

    pub fn from_gguf(file: &GgufFile<'_>, vocab_size: usize) -> Result<Self, BackendError> {
        let arch = file.architecture().unwrap_or("").to_string();
        if !Self::is_hybrid_arch(&arch) {
            return Err(BackendError::Shape(format!(
                "Qwen3.5-гибрид: архитектура {arch:?} не qwen35/qwen3next"
            )));
        }
        let p = |s: &str| format!("{arch}.{s}");
        let u = |k: &str| -> Result<usize, BackendError> {
            file.get(k)
                .and_then(GgufValue::as_u32)
                .map(|v| v as usize)
                .ok_or_else(|| BackendError::Shape(format!("missing {k}")))
        };
        let f = |k: &str| -> Option<f32> {
            file.get(k).and_then(|v| match v {
                GgufValue::F32(x) => Some(*x),
                GgufValue::F64(x) => Some(*x as f32),
                _ => None,
            })
        };
        let dim = u(&p("embedding_length"))?;
        let n_heads = u(&p("attention.head_count"))?;
        let n_kv_heads = u(&p("attention.head_count_kv"))?;
        if n_heads == 0 || n_kv_heads == 0 || !n_heads.is_multiple_of(n_kv_heads) {
            return Err(BackendError::Shape(format!(
                "{arch}: головы {n_heads}/{n_kv_heads} некорректны"
            )));
        }
        let head_dim = u(&p("attention.key_length")).unwrap_or(dim / n_heads);
        // MTP/NextN-слои (nextn_predict_layers) лежат в конце block_count и в
        // основном проходе не исполняются (llama.cpp: n_layer = all − nextn).
        let n_all = u(&p("block_count"))?;
        let n_nextn = u(&p("nextn_predict_layers")).unwrap_or(0);
        let n_layers = n_all.saturating_sub(n_nextn);
        let recurrent = match file
            .get(&p("attention.recurrent_layers"))
            .and_then(GgufValue::as_array)
        {
            Some(arr) => arr
                .iter()
                .take(n_layers)
                .map(|v| match v {
                    GgufValue::Bool(b) => *b,
                    other => other.as_u32().unwrap_or(0) != 0,
                })
                .collect::<Vec<_>>(),
            None => {
                let interval = u(&p("full_attention_interval")).unwrap_or(4).max(1);
                (0..n_layers).map(|i| (i + 1) % interval != 0).collect()
            }
        };
        if recurrent.len() != n_layers {
            return Err(BackendError::Shape(format!(
                "{arch}: recurrent_layers {} != слоёв {n_layers}",
                recurrent.len()
            )));
        }
        let mut rope_sections = [0usize; 4];
        if let Some(arr) = file
            .get(&p("rope.dimension_sections"))
            .and_then(GgufValue::as_array)
        {
            for (i, v) in arr.iter().take(4).enumerate() {
                rope_sections[i] = v.as_u32().unwrap_or(0) as usize;
            }
        }
        let n_rot = u(&p("rope.dimension_count")).unwrap_or(head_dim);
        if n_rot == 0 || n_rot > head_dim || n_rot % 2 != 0 {
            return Err(BackendError::Shape(format!(
                "{arch}: rope.dimension_count {n_rot} при head_dim {head_dim}"
            )));
        }
        let ssm_state = u(&p("ssm.state_size"))?;
        let ssm_k_heads = u(&p("ssm.group_count"))?;
        let ssm_v_heads = u(&p("ssm.time_step_rank"))?;
        let ssm_inner = u(&p("ssm.inner_size"))?;
        if ssm_k_heads == 0 || ssm_v_heads % ssm_k_heads != 0 {
            return Err(BackendError::Shape(format!(
                "{arch}: v-голов {ssm_v_heads} не кратно k-головам {ssm_k_heads}"
            )));
        }
        if ssm_inner != ssm_v_heads * ssm_state {
            // head_v_dim = inner / v_heads обязан совпадать с head_k_dim:
            // delta rule ядра и стейт [Hv, S, S] квадратные (ggml assert S_k == S_v).
            return Err(BackendError::Shape(format!(
                "{arch}: inner_size {ssm_inner} != v_heads {ssm_v_heads} × state {ssm_state}"
            )));
        }
        if ssm_state > 1024 {
            return Err(BackendError::Shape(format!(
                "{arch}: ssm.state_size {ssm_state} > 1024 (threadgroup delta rule)"
            )));
        }
        Ok(Self {
            arch: arch.clone(),
            dim,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            ffn_dim: u(&p("feed_forward_length"))?,
            vocab_size,
            rope_theta: f(&p("rope.freq_base")).unwrap_or(10_000_000.0),
            n_rot,
            rope_sections,
            norm_eps: f(&p("attention.layer_norm_rms_epsilon")).unwrap_or(1e-6),
            // Окно контекста движка (как у Qwen3Model): KV растёт до него,
            // дальше — громкая ошибка (у GDN-стейта окна нет вовсе).
            max_seq: 8192,
            conv_kernel: u(&p("ssm.conv_kernel"))?,
            ssm_state,
            ssm_k_heads,
            ssm_v_heads,
            ssm_inner,
            recurrent,
        })
    }

    /// Каналы conv1d: q (Hk·S) | k (Hk·S) | v (inner).
    pub fn conv_channels(&self) -> usize {
        2 * self.ssm_k_heads * self.ssm_state + self.ssm_inner
    }
}

/// Полное gated-внимание (Qwen3.5): q-проекция отдаёт на голову [q | gate].
struct FullAttn {
    /// [n_heads·hd·2 + 2·n_kv·hd, dim] — q(+gate), k, v одним GEMM.
    wqkv: Tensor,
    wo: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
}

/// Линейное внимание Gated DeltaNet.
struct Gdn {
    /// [conv_channels + inner (z) + 2·Hv (beta, alpha), dim] — все входные
    /// проекции слоя одним GEMM.
    w_in: Tensor,
    conv_w: Tensor,
    dt_bias: Tensor,
    a: Tensor,
    norm: Tensor,
    w_out: Tensor,
}

enum Mixer {
    Attn(FullAttn),
    Gdn(Gdn),
}

struct Layer {
    attn_norm: RmsNorm,
    post_norm: RmsNorm,
    mixer: Mixer,
    ffn: SwiGLU,
}

/// Рекуррентный стейт GDN-слоя (per-sequence).
struct GdnState {
    /// [C, K-1] — хвост conv-входа.
    conv: Tensor,
    /// [Hv, S, S] — транспонированный стейт delta rule (как в ggml).
    ssm: Tensor,
}

pub struct Qwen35Model {
    pub cfg: Qwen35Config,
    /// Таблица эмбеддингов в родном формате GGUF (деквант строки на лету);
    /// при tied-весах та же таблица служит lm_head.
    embed: Tensor,
    /// f32-копия эмбеддингов для драфтера (по требованию, shared_tables).
    embed_f32: Option<Tensor>,
    output: Tensor,
    out_norm: RmsNorm,
    layers: Vec<Layer>,
    /// KV-кэши полных слоёв (None у рекуррентных).
    caches: Vec<Option<KvCache>>,
    states: Vec<Option<GdnState>>,
    /// Позиция следующего токена (число токенов в контексте).
    pos: usize,
    /// forward считает логиты только последней позиции (prefill_last).
    last_only: bool,
}

impl Qwen35Model {
    pub fn load(
        backend: &dyn Backend,
        w: &WeightsMap<'_>,
        cfg: Qwen35Config,
    ) -> Result<Self, BackendError> {
        let get = |name: &str| {
            w.get(name)
                .ok_or_else(|| BackendError::Shape(format!("missing tensor {name}")))
        };
        let f32v = |name: &str| -> Result<Vec<f32>, BackendError> {
            Ok(crate::weights_map::dequant_f32(get(name)?))
        };
        // Эмбеддинги — в родном формате (Q8_0 248K×2560 = 0.68 ГБ вместо
        // 2.5 ГБ f32); lm_head при отсутствии output.weight — та же таблица.
        let emb = get("token_embd.weight")?;
        let emb_rows = emb.shape.first().copied().unwrap_or(0) as usize;
        if emb_rows < cfg.vocab_size {
            return Err(BackendError::Shape(format!(
                "token_embd: строк {emb_rows} < словаря {}",
                cfg.vocab_size
            )));
        }
        let embed = Qwen3Model::upload_q(backend, w, "token_embd.weight", vec![emb_rows, cfg.dim])?;
        let output = match w.get("output.weight") {
            Some(_) if w.tied_aliases("output.weight").is_empty() => {
                Qwen3Model::upload_q(backend, w, "output.weight", vec![cfg.vocab_size, cfg.dim])?
            }
            _ => {
                if emb_rows == cfg.vocab_size {
                    embed.clone()
                } else {
                    // lm_head обязан выдавать ровно vocab строк (сэмплер
                    // читает последние vocab логитов) — отдельная загрузка.
                    let data = emb.raw();
                    let row = data.len() / emb_rows;
                    backend.from_quantized(
                        &data[..row * cfg.vocab_size],
                        vec![cfg.vocab_size, cfg.dim],
                        emb.format,
                    )?
                }
            }
        };
        let out_norm = RmsNorm::from_weights(backend, f32v("output_norm.weight")?, cfg.norm_eps)?;

        let hd = cfg.head_dim;
        let s = cfg.ssm_state;
        let c = cfg.conv_channels();
        let mut layers = Vec::with_capacity(cfg.n_layers);
        let mut caches = Vec::with_capacity(cfg.n_layers);
        let mut states = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = |x: &str| format!("blk.{i}.{x}");
            let attn_norm = RmsNorm::from_weights(backend, f32v(&p("attn_norm.weight"))?, cfg.norm_eps)?;
            let post_norm = RmsNorm::from_weights(
                backend,
                f32v(&p("post_attention_norm.weight"))?,
                cfg.norm_eps,
            )?;
            let ffn = SwiGLU {
                w13: Qwen3Model::upload_fused(
                    backend,
                    w,
                    &[&p("ffn_gate.weight"), &p("ffn_up.weight")],
                    vec![2 * cfg.ffn_dim, cfg.dim],
                )?,
                w2: Qwen3Model::upload_q(
                    backend,
                    w,
                    &p("ffn_down.weight"),
                    vec![cfg.dim, cfg.ffn_dim],
                )?,
            };
            let mixer = if cfg.recurrent[i] {
                let hv = cfg.ssm_v_heads;
                let conv = f32v(&p("ssm_conv1d.weight"))?;
                if conv.len() != c * cfg.conv_kernel {
                    return Err(BackendError::Shape(format!(
                        "blk.{i}.ssm_conv1d: {} != C {c} × K {}",
                        conv.len(),
                        cfg.conv_kernel
                    )));
                }
                Mixer::Gdn(Gdn {
                    w_in: Qwen3Model::upload_fused(
                        backend,
                        w,
                        &[
                            &p("attn_qkv.weight"),
                            &p("attn_gate.weight"),
                            &p("ssm_beta.weight"),
                            &p("ssm_alpha.weight"),
                        ],
                        vec![c + cfg.ssm_inner + 2 * hv, cfg.dim],
                    )?,
                    conv_w: backend.from_cpu_f32(&conv, vec![c, cfg.conv_kernel])?,
                    dt_bias: backend.from_cpu_f32(&f32v(&p("ssm_dt.bias"))?, vec![hv])?,
                    a: backend.from_cpu_f32(&f32v(&p("ssm_a"))?, vec![hv])?,
                    norm: backend.from_cpu_f32(&f32v(&p("ssm_norm.weight"))?, vec![s])?,
                    w_out: Qwen3Model::upload_q(
                        backend,
                        w,
                        &p("ssm_out.weight"),
                        vec![cfg.dim, cfg.ssm_inner],
                    )?,
                })
            } else {
                let q_out = 2 * cfg.n_heads * hd;
                let kv_out = cfg.n_kv_heads * hd;
                Mixer::Attn(FullAttn {
                    wqkv: Qwen3Model::upload_fused(
                        backend,
                        w,
                        &[&p("attn_q.weight"), &p("attn_k.weight"), &p("attn_v.weight")],
                        vec![q_out + 2 * kv_out, cfg.dim],
                    )?,
                    wo: Qwen3Model::upload_q(
                        backend,
                        w,
                        &p("attn_output.weight"),
                        vec![cfg.dim, cfg.n_heads * hd],
                    )?,
                    q_norm: backend.from_cpu_f32(&f32v(&p("attn_q_norm.weight"))?, vec![hd])?,
                    k_norm: backend.from_cpu_f32(&f32v(&p("attn_k_norm.weight"))?, vec![hd])?,
                })
            };
            if cfg.recurrent[i] {
                caches.push(None);
                states.push(Some(Self::fresh_state(backend, &cfg)?));
            } else {
                let mut kv = KvCache::new(
                    backend,
                    cfg.n_kv_heads,
                    hd,
                    crate::dtype::KvFormat::F16,
                    crate::dtype::KvFormat::F16,
                    cfg.rope_theta,
                    Some(cfg.rope_sections),
                )?;
                kv.set_max_cap(Some(cfg.max_seq));
                caches.push(Some(kv));
                states.push(None);
            }
            layers.push(Layer {
                attn_norm,
                post_norm,
                mixer,
                ffn,
            });
        }
        Ok(Self {
            cfg,
            embed,
            embed_f32: None,
            output,
            out_norm,
            layers,
            caches,
            states,
            pos: 0,
            last_only: false,
        })
    }

    fn fresh_state(backend: &dyn Backend, cfg: &Qwen35Config) -> Result<GdnState, BackendError> {
        let s = cfg.ssm_state;
        Ok(GdnState {
            conv: backend.zeros(vec![cfg.conv_channels(), cfg.conv_kernel - 1])?,
            ssm: backend.zeros(vec![cfg.ssm_v_heads, s, s])?,
        })
    }

    /// Число токенов в контексте.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Таблицы, которые драфтер делит с таргетом: (эмбеддинги f32, lm_head).
    /// Эмбеддинги модели лежат в родном формате GGUF (деквант строки на
    /// лету), а драфтер читает таблицу обычным f32-эмбеддингом — сырые
    /// Q8_0-блоки в f32-кернеле давали драфтеру мусор (кейс 7.9). f32-копия
    /// (2.5 ГБ для 248K×2560) собирается по требованию — только когда к
    /// модели подключают драфтер; lm_head — тот же тензор, что у модели.
    pub fn shared_tables(
        &mut self,
        backend: &dyn Backend,
        w: &WeightsMap<'_>,
    ) -> Result<(Tensor, Tensor), BackendError> {
        if self.embed_f32.is_none() {
            let emb = w
                .get("token_embd.weight")
                .ok_or_else(|| BackendError::Shape("missing tensor token_embd.weight".into()))?;
            let rows = emb.shape.first().copied().unwrap_or(0) as usize;
            self.embed_f32 = Some(
                backend.from_cpu_f32(&crate::weights_map::dequant_f32(emb), vec![rows, self.cfg.dim])?,
            );
        }
        let embed = self.embed_f32.clone().expect("embed_f32 собран выше");
        Ok((embed, self.output.clone()))
    }

    /// Новая последовательность: KV пусты, рекуррентные стейты обнулены.
    pub fn reset_cache(&mut self, backend: &dyn Backend) -> Result<(), BackendError> {
        for c in self.caches.iter_mut().flatten() {
            c.reset();
            c.shrink_to_init(backend)?;
        }
        for st in self.states.iter_mut().flatten() {
            *st = Self::fresh_state(backend, &self.cfg)?;
        }
        self.pos = 0;
        Ok(())
    }

    /// Prefill: ids [T] → логиты [1, T, V]; контекст пополняется.
    pub fn prefill(&mut self, backend: &dyn Backend, ids: &[u32]) -> Result<Tensor, BackendError> {
        Ok(self.prefill_extract(backend, ids, &[])?.0)
    }

    /// Prefill + выходы слоёв `layers` (по возрастанию; без out_norm) —
    /// фичи таргета для DFlash-драфтера (его target_layers — индексы слоёв
    /// ЭТОЙ модели). Один проход: логиты и контекст — как у prefill.
    pub fn prefill_extract(
        &mut self,
        backend: &dyn Backend,
        ids: &[u32],
        layers: &[usize],
    ) -> Result<(Tensor, Vec<Tensor>), BackendError> {
        if ids.is_empty() {
            return Err(BackendError::Shape("qwen35 prefill: пустой промпт".into()));
        }
        if let Some(&bad) = layers.iter().find(|&&l| l >= self.cfg.n_layers) {
            return Err(BackendError::Shape(format!(
                "qwen35 extract: слой {bad} >= {}",
                self.cfg.n_layers
            )));
        }
        backend.begin_batch()?;
        let mut extract = Vec::with_capacity(layers.len());
        let r = self.forward(backend, ids, layers, &mut extract);
        backend.end_batch()?;
        Ok((r?, extract))
    }

    /// Один токен (decode-шаг) → логиты [1, 1, V].
    pub fn decode(&mut self, backend: &dyn Backend, id: u32) -> Result<Tensor, BackendError> {
        backend.begin_batch()?;
        let r = self.forward(backend, &[id], &[], &mut Vec::new());
        backend.end_batch()?;
        r
    }

    /// Prefill с логитами только последней позиции ([1, 1, V]) — генерация
    /// читает лишь их; lm_head (V=248K строк!) идёт по одной строке вместо T.
    pub fn prefill_last(&mut self, backend: &dyn Backend, ids: &[u32]) -> Result<Tensor, BackendError> {
        if ids.is_empty() {
            return Err(BackendError::Shape("qwen35 prefill: пустой промпт".into()));
        }
        backend.begin_batch()?;
        self.last_only = true;
        let r = self.forward(backend, ids, &[], &mut Vec::new());
        self.last_only = false;
        backend.end_batch()?;
        r
    }

    /// Логиты lm_head из нормированных скрытых (общий путь с forward).
    fn head_rows(&self, backend: &dyn Backend, h: &Tensor, t: usize) -> Result<Tensor, BackendError> {
        let d = self.cfg.dim;
        let logits = backend.zeros(vec![t, self.cfg.vocab_size])?;
        backend.gemm_nt(&h.view(vec![t, d]), &self.output, &logits)?;
        Ok(logits.view(vec![1, t, self.cfg.vocab_size]))
    }

    /// Снимок рекуррентных стейтов (копии на устройстве): delta rule обновляет
    /// стейт на месте, поэтому ссылка на старый тензор снимком не является.
    fn snapshot_states(&self, backend: &dyn Backend) -> Result<Vec<Option<(Tensor, Tensor)>>, BackendError> {
        let copy = |t: &Tensor| -> Result<Tensor, BackendError> {
            let y = backend.zeros(t.shape().to_vec())?;
            backend.scale(t, &y, 1.0)?;
            Ok(y)
        };
        self.states
            .iter()
            .map(|s| match s {
                Some(st) => Ok(Some((copy(&st.conv)?, copy(&st.ssm)?))),
                None => Ok(None),
            })
            .collect()
    }

    /// Спекулятивный шаг (Ф9.4): верификация [id, draft…] ОДНИМ forward'ом.
    /// KV полных слоёв откатывается усечением, а рекуррентный стейт GDN
    /// усечь нельзя (он уже впитал отвергнутые токены) — при неполном
    /// принятии стейты восстанавливаются из снимка ДО шага, и принятый
    /// префикс прогоняется заново (тот же кернельный путь, что у decode —
    /// траектория совпадает с обычной генерацией). При полном принятии
    /// реплея нет. Возврат — как у [`crate::qwen3::Qwen3Model::decode_spec`]:
    /// extract — выходы слоёв `extract_layers` по всем строкам окна
    /// верификации (DFlash берёт из них только принятые строки).
    #[allow(clippy::too_many_arguments)]
    pub fn decode_spec(
        &mut self,
        backend: &dyn Backend,
        id: u32,
        draft: &[u32],
        draft_probs: &[f32],
        rng: &mut Option<u64>,
        extract_layers: &[usize],
        greedy: bool,
        hist: &[u32],
        repeat_penalty: f32,
    ) -> Result<crate::qwen3::SpecOutcome, BackendError> {
        if draft.len() != draft_probs.len() {
            return Err(BackendError::Shape(format!(
                "decode_spec: черновиков {} != вероятностей {}",
                draft.len(),
                draft_probs.len()
            )));
        }
        let base = self.pos;
        let snap = self.snapshot_states(backend)?;
        let mut ids = Vec::with_capacity(1 + draft.len());
        ids.push(id);
        ids.extend_from_slice(draft);
        let (logits, extract) = self.prefill_extract(backend, &ids, extract_layers)?;
        let v = self.cfg.vocab_size;
        let ld = backend.to_cpu_f32(&logits)?;
        let (accepted, accepted_len, n_verified) = crate::qwen3::verify_draft(
            &ld,
            v,
            id,
            draft,
            draft_probs,
            rng,
            greedy,
            hist,
            repeat_penalty,
        );
        let row_idx = accepted_len - 1;
        if accepted_len < ids.len() {
            // Откат: KV — усечением до базы, стейты — из снимка; затем
            // реплей принятых (id + принятые черновики) — стейт и KV снова
            // описывают ровно принятый префикс.
            for c in self.caches.iter_mut().flatten() {
                c.truncate(base);
            }
            for (st, s) in self.states.iter_mut().zip(snap) {
                if let (Some(st), Some((conv, ssm))) = (st.as_mut(), s) {
                    st.conv = conv;
                    st.ssm = ssm;
                }
            }
            self.pos = base;
            let _ = self.prefill(backend, &ids[..accepted_len])?;
        }
        let row = &ld[row_idx * v..(row_idx + 1) * v];
        let out_logits = backend.from_cpu_f32(row, vec![1, 1, v])?;
        Ok(crate::qwen3::SpecOutcome {
            accepted,
            logits: out_logits,
            extract,
            n_draft_accepted: accepted_len - 1,
            n_draft_verified: n_verified,
        })
    }

    fn forward(
        &mut self,
        backend: &dyn Backend,
        ids: &[u32],
        extract_layers: &[usize],
        extract_out: &mut Vec<Tensor>,
    ) -> Result<Tensor, BackendError> {
        let t = ids.len();
        if self.pos + t > self.cfg.max_seq {
            return Err(BackendError::Shape(format!(
                "qwen35: контекст {} + {t} > окна {}",
                self.pos, self.cfg.max_seq
            )));
        }
        let d = self.cfg.dim;
        let mut x = backend.zeros(vec![1, t, d])?;
        backend.embedding_quant(ids, &self.embed, &x)?;
        // Одна маска на forward (все полные слои видят одни и те же позиции).
        let mask = if t > 1 {
            Some(crate::nn::build_causal_mask(backend, t, self.pos + t, self.pos, None)?)
        } else {
            None
        };
        let dbg = std::env::var("QWEN35_DEBUG").is_ok();
        // QWEN35_PROF=1: время по типам блоков (синхронизация после каждого
        // блока — только для диагностики, в штатном режиме батч не рвётся).
        let prof = std::env::var("QWEN35_PROF").is_ok();
        let sync = |t: &Tensor| -> Result<(), BackendError> {
            if prof {
                backend.to_cpu_f32(&t.view_restricted(vec![1]))?;
            }
            Ok(())
        };
        let (mut t_attn, mut t_gdn, mut t_ffn) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..self.layers.len() {
            let layer = &self.layers[i];
            let t0 = std::time::Instant::now();
            let h = layer.attn_norm.forward(backend, &x)?;
            let m = match &layer.mixer {
                Mixer::Attn(a) => {
                    let cache = self.caches[i]
                        .as_mut()
                        .ok_or_else(|| BackendError::Shape(format!("слой {i}: нет KV")))?;
                    Self::attn_forward(&self.cfg, backend, &h, a, cache, self.pos, mask.as_ref())?
                }
                Mixer::Gdn(g) => {
                    let st = self.states[i]
                        .as_mut()
                        .ok_or_else(|| BackendError::Shape(format!("слой {i}: нет стейта")))?;
                    Self::gdn_forward(&self.cfg, backend, &h, g, st)?
                }
            };
            let x1 = backend.zeros(x.shape().to_vec())?;
            backend.add(&x, &m, &x1)?;
            sync(&x1)?;
            let t1 = std::time::Instant::now();
            if self.cfg.recurrent[i] {
                t_gdn += (t1 - t0).as_secs_f64();
            } else {
                t_attn += (t1 - t0).as_secs_f64();
            }
            let h2 = layer.post_norm.forward(backend, &x1)?;
            let f = ffn_forward(backend, &h2, &layer.ffn)?;
            let x2 = backend.zeros(x.shape().to_vec())?;
            backend.add(&x1, &f, &x2)?;
            sync(&x2)?;
            t_ffn += t1.elapsed().as_secs_f64();
            x = x2;
            if extract_layers.contains(&i) {
                extract_out.push(x.clone());
            }
            if dbg {
                let v = backend.to_cpu_f32(&x)?;
                let last = &v[v.len() - d..];
                let rms = (last.iter().map(|a| a * a).sum::<f32>() / d as f32).sqrt();
                eprintln!(
                    "QWEN35 layer {i:>2} ({}) out[-1] rms={rms:.4} [:4]={:?}",
                    if self.cfg.recurrent[i] { "gdn " } else { "attn" },
                    &last[..4]
                );
            }
        }
        self.pos += t;
        let t_head = std::time::Instant::now();
        let (x, t) = if self.last_only && t > 1 {
            (x.view(vec![t, d]).view_offset(vec![1, 1, d], (t - 1) * d), 1)
        } else {
            (x, t)
        };
        let h = self.out_norm.forward(backend, &x)?;
        let logits = self.head_rows(backend, &h, t)?;
        if prof {
            sync(&logits)?;
            eprintln!(
                "QWEN35_PROF t={t}: attn {:.1}ms gdn {:.1}ms ffn {:.1}ms head {:.1}ms",
                t_attn * 1e3,
                t_gdn * 1e3,
                t_ffn * 1e3,
                t_head.elapsed().as_secs_f64() * 1e3
            );
        }
        Ok(logits)
    }

    /// Полное gated-внимание (qwen35.cpp build_layer_attn).
    fn attn_forward(
        cfg: &Qwen35Config,
        backend: &dyn Backend,
        h: &Tensor,
        a: &FullAttn,
        cache: &mut KvCache,
        pos: usize,
        mask: Option<&Tensor>,
    ) -> Result<Tensor, BackendError> {
        let t = h.shape()[1];
        let (nh, nkv, hd) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
        let qg_size = 2 * nh * hd;
        let kv_size = nkv * hd;
        let qkv = Linear {
            w: a.wqkv.clone(),
            b: None,
        }
        .forward(backend, h)?;
        // q-проекция на голову: [q (hd) | gate (hd)] — разбор через
        // [T·nh, 2·hd]-вид (строка = голова токена).
        let qg = crate::nn::slice_last(backend, &qkv, qg_size, 0)?;
        let k = crate::nn::slice_last(backend, &qkv, kv_size, qg_size)?;
        let v = crate::nn::slice_last(backend, &qkv, kv_size, qg_size + kv_size)?;
        let qg_rows = qg.view(vec![1, t * nh, 2 * hd]);
        let q = crate::nn::slice_last(backend, &qg_rows, hd, 0)?.view(vec![1, t, nh * hd]);
        let gate = crate::nn::slice_last(backend, &qg_rows, hd, hd)?.view(vec![1, t, nh * hd]);
        let qn = backend.zeros(vec![1, t, nh * hd])?;
        backend.rmsnorm_head(&q, &a.q_norm, &qn, nh, cfg.norm_eps)?;
        let kn = backend.zeros(vec![1, t, kv_size])?;
        backend.rmsnorm_head(&k, &a.k_norm, &kn, nkv, cfg.norm_eps)?;
        let qr = backend.rope_imrope_text(&qn, pos, hd, cfg.rope_theta, cfg.rope_sections, cfg.n_rot)?;
        let kr = backend.rope_imrope_text(&kn, pos, hd, cfg.rope_theta, cfg.rope_sections, cfg.n_rot)?;
        cache.update(backend, &kr, &v)?;
        let out = backend.zeros(vec![1, t, nh * hd])?;
        backend.attn_ar(&qr, &cache.view(), &cache.view_v(), mask, &out, nh, nkv, hd)?;
        let sg = backend.zeros(vec![1, t, nh * hd])?;
        backend.sigmoid(&gate, &sg)?;
        let gated = backend.zeros(vec![1, t, nh * hd])?;
        backend.mul(&out, &sg, &gated)?;
        Linear {
            w: a.wo.clone(),
            b: None,
        }
        .forward(backend, &gated)
    }

    /// Линейное внимание Gated DeltaNet (qwen35.cpp build_layer_attn_linear).
    fn gdn_forward(
        cfg: &Qwen35Config,
        backend: &dyn Backend,
        h: &Tensor,
        g: &Gdn,
        st: &mut GdnState,
    ) -> Result<Tensor, BackendError> {
        let t = h.shape()[1];
        let (s, hk, hv) = (cfg.ssm_state, cfg.ssm_k_heads, cfg.ssm_v_heads);
        let c = cfg.conv_channels();
        let inner = cfg.ssm_inner;
        let proj = Linear {
            w: g.w_in.clone(),
            b: None,
        }
        .forward(backend, h)?;
        let qkv = crate::nn::slice_last(backend, &proj, c, 0)?;
        let z = crate::nn::slice_last(backend, &proj, inner, c)?;
        let beta = crate::nn::slice_last(backend, &proj, hv, c + inner)?;
        let alpha = crate::nn::slice_last(backend, &proj, hv, c + inner + hv)?;
        let (gate, b) = backend.gdn_gates(&alpha, &beta, &g.dt_bias, &g.a)?;
        let (conv_out, new_conv) = backend.gdn_conv(&qkv, &g.conv_w, &st.conv)?;
        st.conv = new_conv;
        let act = backend.zeros(conv_out.shape().to_vec())?;
        backend.silu(&conv_out, &act)?;
        let qk_size = hk * s;
        let q = crate::nn::slice_last(backend, &act, qk_size, 0)?;
        let k = crate::nn::slice_last(backend, &act, qk_size, qk_size)?;
        let v = crate::nn::slice_last(backend, &act, inner, 2 * qk_size)?;
        let qn = backend.gdn_l2norm(&q, s, cfg.norm_eps)?;
        let kn = backend.gdn_l2norm(&k, s, cfg.norm_eps)?;
        let (o, new_ssm) = backend.gdn_delta_rule(&qn, &kn, &v, &gate, &b, &st.ssm, hk)?;
        st.ssm = new_ssm;
        let normed = backend.gdn_norm_gated(&o.view(vec![1, t, inner]), &z, &g.norm, cfg.norm_eps)?;
        Linear {
            w: g.w_out.clone(),
            b: None,
        }
        .forward(backend, &normed)
    }
}

/// Загрузка гибрида из GGUF-файла целиком (токенайзерный размер словаря —
/// от вызывающего: он же режет логиты).
pub fn load_gguf(
    backend: &dyn Backend,
    file: &GgufFile<'_>,
    w: &WeightsMap<'_>,
    vocab_size: usize,
) -> Result<Qwen35Model, BackendError> {
    let cfg = Qwen35Config::from_gguf(file, vocab_size)?;
    Qwen35Model::load(backend, w, cfg)
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use crate::backend::Backend;
    use crate::metal::MetalBackend;

    fn rng(seed: u64) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    fn rand(n: usize, r: &mut impl FnMut() -> f32) -> Vec<f32> {
        (0..n).map(|_| r()).collect()
    }

    fn close(a: &[f32], b: &[f32], tol: f32, what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: длины");
        let mut worst = 0.0f32;
        for (x, y) in a.iter().zip(b) {
            worst = worst.max((x - y).abs() / (1.0 + y.abs()));
        }
        assert!(worst < tol, "{what}: max rel err {worst}");
    }

    #[test]
    fn gdn_kernels_match_reference() {
        let be = MetalBackend::new().unwrap();
        let mut r = rng(0x5EED);
        let (t, c, k) = (5usize, 40usize, 4usize);
        let x = rand(t * c, &mut r);
        let w = rand(c * k, &mut r);
        let st = rand(c * (k - 1), &mut r);
        let xt = be.from_cpu_f32(&x, vec![1, t, c]).unwrap();
        let wt = be.from_cpu_f32(&w, vec![c, k]).unwrap();
        let stt = be.from_cpu_f32(&st, vec![c, k - 1]).unwrap();
        let (y, ns) = be.gdn_conv(&xt, &wt, &stt).unwrap();
        // эталон: прямое вычисление по определению ggml_ssm_conv
        let at = |ch: usize, p: usize| if p < k - 1 { st[ch * (k - 1) + p] } else { x[(p - (k - 1)) * c + ch] };
        let mut ry = vec![0.0f32; t * c];
        for tt in 0..t {
            for ch in 0..c {
                ry[tt * c + ch] = (0..k).map(|j| w[ch * k + j] * at(ch, tt + j)).sum();
            }
        }
        let mut rs = vec![0.0f32; c * (k - 1)];
        for ch in 0..c {
            for j in 0..k - 1 {
                rs[ch * (k - 1) + j] = at(ch, t + j);
            }
        }
        close(&be.to_cpu_f32(&y).unwrap(), &ry, 1e-5, "conv");
        close(&be.to_cpu_f32(&ns).unwrap(), &rs, 1e-6, "conv state");
    }

    #[test]
    fn delta_rule_kernel_matches_reference() {
        // S=16 — поточный кернел; S=128 (как у Qwen3.5) — simdgroup-вариант.
        delta_rule_case(3, 2, 4, 16);
        delta_rule_case(5, 2, 4, 128);
    }

    fn delta_rule_case(t: usize, hk: usize, hv: usize, s: usize) {
        let be = MetalBackend::new().unwrap();
        let mut r = rng(0xD17A ^ s as u64);
        let q = rand(t * hk * s, &mut r);
        let k = rand(t * hk * s, &mut r);
        let v = rand(t * hv * s, &mut r);
        let g: Vec<f32> = rand(t * hv, &mut r).iter().map(|x| -x.abs()).collect();
        let b: Vec<f32> = rand(t * hv, &mut r).iter().map(|x| 0.5 + 0.4 * x).collect();
        let st0 = rand(hv * s * s, &mut r);
        let up = |d: &[f32], sh: Vec<usize>| be.from_cpu_f32(d, sh).unwrap();
        let state = up(&st0, vec![hv, s, s]);
        let (out, ns) = be
            .gdn_delta_rule(
                &up(&q, vec![t, hk, s]),
                &up(&k, vec![t, hk, s]),
                &up(&v, vec![t, hv, s]),
                &up(&g, vec![t, hv]),
                &up(&b, vec![t, hv]),
                &state,
                hk,
            )
            .unwrap();
        // эталон: ggml gated_delta_net one_chunk (транспонированный стейт)
        let mut st = st0.clone();
        let mut ro = vec![0.0f32; t * hv * s];
        for h in 0..hv {
            let hq = h % hk;
            let m = &mut st[h * s * s..(h + 1) * s * s];
            for tt in 0..t {
                let qd = &q[(tt * hk + hq) * s..][..s];
                let kd = &k[(tt * hk + hq) * s..][..s];
                let vd = &v[(tt * hv + h) * s..][..s];
                let eg = g[tt * hv + h].exp();
                m.iter_mut().for_each(|x| *x *= eg);
                let mut delta = vec![0.0f32; s];
                for j in 0..s {
                    let dot: f32 = (0..s).map(|i| m[j * s + i] * kd[i]).sum();
                    delta[j] = (vd[j] - dot) * b[tt * hv + h];
                }
                for j in 0..s {
                    for i in 0..s {
                        m[j * s + i] += kd[i] * delta[j];
                    }
                }
                for j in 0..s {
                    let dot: f32 = (0..s).map(|i| m[j * s + i] * qd[i]).sum();
                    ro[(tt * hv + h) * s + j] = dot / (s as f32).sqrt();
                }
            }
        }
        close(&be.to_cpu_f32(&out).unwrap(), &ro, 1e-4, "delta out");
        close(&be.to_cpu_f32(&ns).unwrap(), &st, 1e-4, "delta state");
    }

    #[test]
    fn imrope_partial_matches_reference() {
        let be = MetalBackend::new().unwrap();
        let mut r = rng(0x40BE);
        let (t, heads, hd, n_rot, pos) = (3usize, 2usize, 32usize, 8usize, 7usize);
        let sections = [2usize, 1, 1, 0];
        let x = rand(t * heads * hd, &mut r);
        let y = be
            .rope_imrope_text(
                &be.from_cpu_f32(&x, vec![1, t, heads * hd]).unwrap(),
                pos,
                hd,
                10_000.0,
                sections,
                n_rot,
            )
            .unwrap();
        let yv = be.to_cpu_f32(&y).unwrap();
        // эталон: ggml_mrope_cache_init (IMROPE) + rotate_pairs(n_dims, n_dims/2)
        let d = heads * hd;
        let half = n_rot / 2;
        let sect = 4usize;
        let mut ry = x.clone();
        for tt in 0..t {
            for h in 0..heads {
                for i in 0..half {
                    let sector = i % sect;
                    let rot = (sector % 3 == 0 && sector < 3 * sections[0])
                        || (sector % 3 == 1 && sector < 3 * sections[1])
                        || (sector % 3 == 2 && sector < 3 * sections[2]);
                    let p = if rot { (pos + tt) as f32 } else { 0.0 };
                    let th = p * 10_000f32.powf(-2.0 * i as f32 / n_rot as f32);
                    let b = tt * d + h * hd;
                    let (x0, x1) = (x[b + i], x[b + i + half]);
                    ry[b + i] = x0 * th.cos() - x1 * th.sin();
                    ry[b + i + half] = x0 * th.sin() + x1 * th.cos();
                }
            }
        }
        close(&yv, &ry, 1e-4, "imrope");
    }

    /// Attention prefill-пути (Tq > 1): контекст в кэше (Tkv > Tq, причинная
    /// маска со смещением), hd 128 и 256, GQA 2:1 и 4:1, Tq не кратно 32.
    #[test]
    fn prefill_attention_matches_reference() {
        let be = MetalBackend::new().unwrap();
        for &(tq, past, nh, nkv, hd) in &[
            (37usize, 11usize, 4usize, 2usize, 128usize),
            (19, 40, 4, 1, 128),
            (19, 40, 4, 2, 128),
            (19, 11, 4, 1, 256),
            (19, 40, 4, 1, 256),
        ] {
            let tkv = past + tq;
            let mut r = rng((tq * 1000 + hd) as u64);
            let q = rand(tq * nh * hd, &mut r);
            let k = rand(tkv * nkv * hd, &mut r);
            let v = rand(tkv * nkv * hd, &mut r);
            let mask = crate::nn::build_causal_mask(&be, tq, tkv, past, None).unwrap();
            let out = be.zeros(vec![1, tq, nh * hd]).unwrap();
            be.attn_ar(
                &be.from_cpu_f32(&q, vec![1, tq, nh * hd]).unwrap(),
                &be.from_cpu_f32(&k, vec![1, tkv, nkv * hd]).unwrap(),
                &be.from_cpu_f32(&v, vec![1, tkv, nkv * hd]).unwrap(),
                Some(&mask),
                &out,
                nh,
                nkv,
                hd,
            )
            .unwrap();
            let mut ro = vec![0.0f32; tq * nh * hd];
            for t in 0..tq {
                for h in 0..nh {
                    let kh = h / (nh / nkv);
                    let qr = &q[(t * nh + h) * hd..][..hd];
                    let lim = past + t + 1; // причинно: ключи 0..=past+t
                    let sc: Vec<f32> = (0..lim)
                        .map(|j| {
                            let kr = &k[(j * nkv + kh) * hd..][..hd];
                            qr.iter().zip(kr).map(|(a, b)| a * b).sum::<f32>() / (hd as f32).sqrt()
                        })
                        .collect();
                    let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let e: Vec<f32> = sc.iter().map(|s| (s - mx).exp()).collect();
                    let sum: f32 = e.iter().sum();
                    for d in 0..hd {
                        ro[(t * nh + h) * hd + d] =
                            (0..lim).map(|j| e[j] / sum * v[(j * nkv + kh) * hd + d]).sum();
                    }
                }
            }
            close(
                &be.to_cpu_f32(&out).unwrap(),
                &ro,
                1e-4,
                &format!("prefill attn tq={tq} past={past} kv={nkv} hd={hd}"),
            );
        }
    }

    #[test]
    fn attention_wide_heads_match_reference() {
        // hd = 256 (Qwen3.5): лейн ведёт два float4-среза — раньше половина
        // измерений головы выпадала из скора и выхода.
        let be = MetalBackend::new().unwrap();
        let mut r = rng(0xA77);
        let (tq, tkv, nh, nkv, hd) = (2usize, 5usize, 4usize, 2usize, 256usize);
        let q = rand(tq * nh * hd, &mut r);
        let k = rand(tkv * nkv * hd, &mut r);
        let v = rand(tkv * nkv * hd, &mut r);
        let out = be.zeros(vec![1, tq, nh * hd]).unwrap();
        be.attn_ar(
            &be.from_cpu_f32(&q, vec![1, tq, nh * hd]).unwrap(),
            &be.from_cpu_f32(&k, vec![1, tkv, nkv * hd]).unwrap(),
            &be.from_cpu_f32(&v, vec![1, tkv, nkv * hd]).unwrap(),
            None,
            &out,
            nh,
            nkv,
            hd,
        )
        .unwrap();
        let mut ro = vec![0.0f32; tq * nh * hd];
        for t in 0..tq {
            for h in 0..nh {
                let kh = h / (nh / nkv);
                let qr = &q[(t * nh + h) * hd..][..hd];
                let sc: Vec<f32> = (0..tkv)
                    .map(|j| {
                        let kr = &k[(j * nkv + kh) * hd..][..hd];
                        qr.iter().zip(kr).map(|(a, b)| a * b).sum::<f32>() / (hd as f32).sqrt()
                    })
                    .collect();
                let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = sc.iter().map(|s| (s - mx).exp()).collect();
                let sum: f32 = e.iter().sum();
                for d in 0..hd {
                    ro[(t * nh + h) * hd + d] =
                        (0..tkv).map(|j| e[j] / sum * v[(j * nkv + kh) * hd + d]).sum();
                }
            }
        }
        close(&be.to_cpu_f32(&out).unwrap(), &ro, 1e-4, "attn hd=256");
    }

    #[test]
    fn quant_embedding_matches_dequant() {
        // Q8_0-таблица: строки на лету == dequant_q8_0 (эталон weights_map).
        let be = MetalBackend::new().unwrap();
        let (n, d) = (6usize, 64usize);
        let mut bytes = Vec::new();
        let mut r = rng(0xE3B);
        for _ in 0..n * d / 32 {
            let scale = half::f16::from_f32(0.01 + r().abs() * 0.05);
            bytes.extend_from_slice(&scale.to_le_bytes());
            for _ in 0..32 {
                bytes.push(((r() * 127.0) as i8) as u8);
            }
        }
        let table = be
            .from_quantized(&bytes, vec![n, d], crate::dtype::WeightFormat::Q8_0)
            .unwrap();
        let ids = [5u32, 0, 3];
        let y = be.zeros(vec![1, ids.len(), d]).unwrap();
        be.embedding_quant(&ids, &table, &y).unwrap();
        let full = crate::weights_map::dequant_q8_0(&bytes);
        let want: Vec<f32> = ids
            .iter()
            .flat_map(|&i| full[i as usize * d..(i as usize + 1) * d].to_vec())
            .collect();
        close(&be.to_cpu_f32(&y).unwrap(), &want, 1e-6, "embedding q8_0");
    }
}
