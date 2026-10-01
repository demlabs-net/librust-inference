//! Ф1.9 Decode loop: prefill → sample → decode → stop conditions, with a
//! streaming event interface. The loop is backend-agnostic (CPU logits
//! vectors through [`crate::inference::sampling`]); KV state lives in the
//! model.

use crate::backend::Backend;
use crate::error::BackendError;
use crate::inference::reasoning_budget::ReasoningBudget;
use crate::inference::sampling::{SamplingParams, top_p_sample_vec_history};
use crate::qwen3::Qwen3Model;
use crate::tokenizer::{IncrementalDecoder, Tokenizer};

/// Почему завершилась генерация.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    /// EOG-токен (eos/im_end и т.п.).
    Eog,
    /// Достигнут `max_tokens`.
    MaxTokens,
    /// Найден один из `stop_strings` в накопленном тексте.
    StopString,
    /// Останов по запросу (отмена генерации, Этап 2.2): клиент запросил стоп.
    Cancelled,
}

/// Событие стриминга.
#[derive(Debug, Clone)]
pub enum GenerateEvent {
    /// Новый фрагмент текста (уже склеенный по границам UTF-8).
    Delta(String),
    /// Сырой токен (для метрик/логов).
    Token(u32),
    /// Спек-статистика (4.2b) — при включённом драфтере.
    SpecStats(SpecStats),
    /// Времена префилла и декода раздельно (перед `Done`).
    Timings(Timings),
    /// Генерация завершена.
    Done(FinishReason),
}

/// Времена генерации — как `timings` llama-server (server_slot_stats):
/// префилл — от начала до логитов последней строки промпта (первый токен
/// берётся из них и в декод не входит); декод — от конца префилла до
/// последнего сэмплированного токена, скорость — по шагам декода
/// (`predicted_n - 1`). `predicted_n` — как n_decoded llama-server: включая
/// токен конца генерации (он сэмплирован из логитов последнего шага), хотя
/// в текст и в возвращаемые токены он не попадает.
#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Timings {
    pub prompt_n: usize,
    pub prompt_ms: f64,
    pub predicted_n: usize,
    pub predicted_ms: f64,
}

impl Timings {
    /// Шаги декода: первый токен «бесплатный» — из логитов префилла.
    pub fn decode_steps(&self) -> usize {
        self.predicted_n.saturating_sub(1)
    }

    pub fn prompt_per_second(&self) -> f64 {
        if self.prompt_ms > 0.0 { self.prompt_n as f64 * 1e3 / self.prompt_ms } else { 0.0 }
    }

    pub fn predicted_per_second(&self) -> f64 {
        if self.predicted_ms > 0.0 {
            self.decode_steps() as f64 * 1e3 / self.predicted_ms
        } else {
            0.0
        }
    }
}

/// Статистика спек-декодирования (Этап 4.2b).
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct SpecStats {
    pub drafts: u64,
    pub accepted: u64,
    pub rejected: u64,
}

impl SpecStats {
    pub fn acceptance(&self) -> Option<f64> {
        let total = self.accepted + self.rejected;
        if total == 0 {
            None
        } else {
            Some(self.accepted as f64 / total as f64 * 100.0)
        }
    }
}

/// Мультимодальный промпт (2.4b): ids с image_pad + фичи картинки.
pub struct VisionPrompt {
    pub ids: Vec<u32>,
    pub vemb: crate::backend::Tensor,
    pub vds: Vec<crate::backend::Tensor>,
    pub pad_token: u32,
    pub vision_token: u32,
    pub pos4d: Vec<u32>,
    pub pos_after: usize,
}

/// Транспорт PP-стадии (7.5): координатор отдаёт соседу активации [t, dim] и
/// получает выход его диапазона слоёв. Реализацию даёт сервис (кластерный
/// канал с TLS); librust-inference знает только этот контракт.
pub trait StageTransport: Send + Sync {
    /// `reset` — начало новой последовательности (стадия сбрасывает свои KV:
    /// её кэш мог остаться от прогрева/прошлой генерации), `pos` — позиция
    /// первого токена запроса (стадия сверяет её со своим кэшем).
    fn forward_stage(
        &self,
        from: usize,
        upto: Option<usize>,
        norm_out: bool,
        reset: bool,
        pos: usize,
        t: usize,
        hidden: Vec<f32>,
    ) -> Result<Vec<f32>, String>;
}

/// Разбиение модели для PP (7.5): локально считаются слои [0..from), сосед —
/// [from..] + out_norm. Эмбеддинг и lm_head всегда у координатора.
#[derive(Clone)]
pub struct PpSplit {
    pub from: usize,
    pub transport: std::sync::Arc<dyn StageTransport>,
    /// Первый вызов стадии в этой генерации (шлётся как reset: у соседа
    /// кэш мог остаться от прогрева/прошлой генерации).
    pub first: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

pub struct GenerateParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    /// min-p (llama.cpp): отсечь токены с p < min_p · p_max. 0 — выключено.
    pub min_p: f32,
    pub repeat_penalty: f32,
    /// Сид сэмплинга: `Some` — детерминированный поток (материализуется через
    /// [`crate::inference::sampling::state_from_seed`]), `None` — от часов.
    pub seed: Option<u64>,
    /// Текстовые стоп-строки (проверяются по накопленному выводу).
    pub stop_strings: Vec<String>,
    /// Спек-декодинг (4.2b): драфтер + число драфт-токенов (0 = выкл).
    pub drafter: Option<std::sync::Arc<std::sync::Mutex<dyn crate::spec::Drafter>>>,
    pub spec_gamma: usize,
    /// Vision (2.4b): картинка в промпте — prefill_with_vision вместо prefill.
    pub vision: Option<VisionPrompt>,
    /// PP-разбиение (7.5): None — обычная локальная генерация.
    pub pp: Option<PpSplit>,
    /// Бюджет рассуждений (Anthropic `thinking.budget_tokens`, llama.cpp
    /// `reasoning_budget_tokens`): автомат [`ReasoningBudget`], заранее
    /// прогнанный по генерационному префиксу промпта. По исчерпании петля
    /// вставляет его принудительную последовательность (сообщение +
    /// `</think>`), и модель переходит к ответу. None — без ограничения.
    pub reasoning_budget: Option<ReasoningBudget>,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.8,
            top_p: 0.8,
            top_k: 0,
            min_p: 0.0,
            repeat_penalty: 1.0,
            seed: None,
            stop_strings: Vec::new(),
            drafter: None,
            spec_gamma: 0,
            vision: None,
            pp: None,
            reasoning_budget: None,
        }
    }
}

/// Модель, которую умеет вести петля генерации (Ф9.4): классический Qwen3
/// (чистое внимание, KV) и гибрид Qwen3.5/3.8 (Gated DeltaNet + внимание).
/// Обязательное — prefill/decode/спек-верификация; PP-стадии, vision и сбор
/// скрытых для удалённого драфтера — возможности, которые есть не у всех
/// архитектур: по умолчанию они отвечают ЯВНОЙ ошибкой, а не молча
/// деградируют (рекуррентный стейт гибрида нельзя усечь или разрезать по
/// нодам тем же способом, что KV).
pub trait CausalLm {
    /// Размер словаря (ширина строки логитов).
    fn vocab_size(&self) -> usize;
    /// Число слоёв (для PP-границ и extract-слоёв).
    fn n_layers(&self) -> usize;
    /// Архитектура (диагностика и тексты ошибок).
    fn arch(&self) -> &str;
    /// Число токенов в контексте (позиция следующего).
    fn context_len(&self) -> usize;
    /// Новая последовательность: пустой контекст.
    fn reset_cache(&mut self, backend: &dyn Backend) -> Result<(), BackendError>;
    /// Prefill ids → логиты [1, T, V].
    fn prefill(&mut self, backend: &dyn Backend, ids: &[u32]) -> Result<crate::backend::Tensor, BackendError>;
    /// Prefill → логиты ТОЛЬКО последней позиции [1, 1, V] (генерации нужны
    /// лишь они; lm_head по одной строке вместо T).
    fn prefill_last(&mut self, backend: &dyn Backend, ids: &[u32]) -> Result<crate::backend::Tensor, BackendError> {
        self.prefill(backend, ids)
    }
    /// Prefill + выходы слоёв `layers` (DFlash-инжект).
    fn prefill_extract(
        &mut self,
        backend: &dyn Backend,
        ids: &[u32],
        layers: &[usize],
    ) -> Result<(crate::backend::Tensor, Vec<crate::backend::Tensor>), BackendError>;
    /// Один токен → логиты [1, 1, V].
    fn decode(&mut self, backend: &dyn Backend, id: u32) -> Result<crate::backend::Tensor, BackendError>;
    /// Спек-шаг: верификация [id, draft…] одним forward'ом + откат хвоста.
    #[allow(clippy::too_many_arguments)]
    fn decode_spec(
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
    ) -> Result<crate::qwen3::SpecOutcome, BackendError>;
    /// Prefill с картинкой (Qwen3-VL). По умолчанию — не поддержано.
    #[allow(clippy::too_many_arguments)]
    fn prefill_with_vision(
        &mut self,
        _backend: &dyn Backend,
        _ids: &[u32],
        _vision: &crate::backend::Tensor,
        _deepstack: &[crate::backend::Tensor],
        _vision_token: u32,
        _pos4d: &[u32],
        _pos_after: usize,
    ) -> Result<crate::backend::Tensor, BackendError> {
        Err(BackendError::UnsupportedFormat(format!(
            "vision-промпт для архитектуры {} не поддержан",
            self.arch()
        )))
    }
    /// PP-стадия: эмбеддинги координатора.
    fn embed_ids(&self, _backend: &dyn Backend, _ids: &[u32]) -> Result<crate::backend::Tensor, BackendError> {
        Err(self.pp_unsupported())
    }
    /// PP-стадия: слои [from..upto) на присланных скрытых.
    #[allow(clippy::too_many_arguments)]
    fn forward_layers_range(
        &mut self,
        _backend: &dyn Backend,
        _x: crate::backend::Tensor,
        _t: usize,
        _upto: Option<usize>,
        _from: usize,
        _norm_out: bool,
    ) -> Result<crate::backend::Tensor, BackendError> {
        Err(self.pp_unsupported())
    }
    /// PP-стадия: lm_head по нормированным скрытым.
    fn lm_head(&self, _backend: &dyn Backend, _h: &crate::backend::Tensor) -> Result<crate::backend::Tensor, BackendError> {
        Err(self.pp_unsupported())
    }
    /// PP-стадия: позиция кэша первого слоя диапазона.
    fn stage_pos(&self, _from: usize) -> usize {
        self.context_len()
    }
    /// PP-стадия: откат локальных слоёв до позиции (транзакция шага).
    fn truncate_to(&mut self, _pos: usize) -> Result<(), BackendError> {
        Err(self.pp_unsupported())
    }
    /// PP-стадия: синхронизировать rope-позицию с кэшем слоя `from`.
    fn sync_rope_to_cache(&mut self, _from: usize) {}
    /// Ошибка «PP не поддержан» — общий текст.
    fn pp_unsupported(&self) -> BackendError {
        BackendError::UnsupportedFormat(format!(
            "PP-разбиение по нодам для архитектуры {} не поддержано \
             (рекуррентный стейт слоёв не передаётся стадиям)",
            self.arch()
        ))
    }
}

impl CausalLm for crate::qwen3::Qwen3Model {
    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }
    fn n_layers(&self) -> usize {
        self.cfg.n_layers
    }
    fn arch(&self) -> &str {
        "qwen3"
    }
    fn context_len(&self) -> usize {
        self.caches.first().map(|c| c.pos).unwrap_or(0)
    }
    fn reset_cache(&mut self, backend: &dyn Backend) -> Result<(), BackendError> {
        Qwen3Model::reset_cache(self, backend)
    }
    fn prefill(&mut self, backend: &dyn Backend, ids: &[u32]) -> Result<crate::backend::Tensor, BackendError> {
        Qwen3Model::prefill(self, backend, ids)
    }
    fn prefill_last(&mut self, backend: &dyn Backend, ids: &[u32]) -> Result<crate::backend::Tensor, BackendError> {
        Qwen3Model::prefill_last(self, backend, ids)
    }
    fn prefill_extract(
        &mut self,
        backend: &dyn Backend,
        ids: &[u32],
        layers: &[usize],
    ) -> Result<(crate::backend::Tensor, Vec<crate::backend::Tensor>), BackendError> {
        Qwen3Model::prefill_extract(self, backend, ids, layers)
    }
    fn decode(&mut self, backend: &dyn Backend, id: u32) -> Result<crate::backend::Tensor, BackendError> {
        Qwen3Model::decode(self, backend, id)
    }
    fn decode_spec(
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
        Qwen3Model::decode_spec(
            self,
            backend,
            id,
            draft,
            draft_probs,
            rng,
            extract_layers,
            greedy,
            hist,
            repeat_penalty,
        )
    }
    fn prefill_with_vision(
        &mut self,
        backend: &dyn Backend,
        ids: &[u32],
        vision: &crate::backend::Tensor,
        deepstack: &[crate::backend::Tensor],
        vision_token: u32,
        pos4d: &[u32],
        pos_after: usize,
    ) -> Result<crate::backend::Tensor, BackendError> {
        Qwen3Model::prefill_with_vision(
            self,
            backend,
            ids,
            vision,
            deepstack,
            vision_token,
            pos4d,
            pos_after,
        )
    }
    fn embed_ids(&self, backend: &dyn Backend, ids: &[u32]) -> Result<crate::backend::Tensor, BackendError> {
        Qwen3Model::embed_ids(self, backend, ids)
    }
    fn forward_layers_range(
        &mut self,
        backend: &dyn Backend,
        x: crate::backend::Tensor,
        t: usize,
        upto: Option<usize>,
        from: usize,
        norm_out: bool,
    ) -> Result<crate::backend::Tensor, BackendError> {
        Qwen3Model::forward_layers_range(self, backend, x, t, None, upto, from, None, norm_out)
    }
    fn lm_head(&self, backend: &dyn Backend, h: &crate::backend::Tensor) -> Result<crate::backend::Tensor, BackendError> {
        Qwen3Model::lm_head(self, backend, h)
    }
    fn stage_pos(&self, from: usize) -> usize {
        self.caches[from.min(self.caches.len() - 1)].pos
    }
    fn truncate_to(&mut self, pos: usize) -> Result<(), BackendError> {
        for c in self.caches.iter_mut() {
            c.truncate(pos);
        }
        Ok(())
    }
    fn sync_rope_to_cache(&mut self, from: usize) {
        Qwen3Model::sync_rope_to_cache(self, from)
    }
}

impl CausalLm for crate::qwen35::Qwen35Model {
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
        self.pos()
    }
    fn reset_cache(&mut self, backend: &dyn Backend) -> Result<(), BackendError> {
        crate::qwen35::Qwen35Model::reset_cache(self, backend)
    }
    fn prefill(&mut self, backend: &dyn Backend, ids: &[u32]) -> Result<crate::backend::Tensor, BackendError> {
        crate::qwen35::Qwen35Model::prefill(self, backend, ids)
    }
    fn prefill_last(&mut self, backend: &dyn Backend, ids: &[u32]) -> Result<crate::backend::Tensor, BackendError> {
        crate::qwen35::Qwen35Model::prefill_last(self, backend, ids)
    }
    fn prefill_extract(
        &mut self,
        backend: &dyn Backend,
        ids: &[u32],
        layers: &[usize],
    ) -> Result<(crate::backend::Tensor, Vec<crate::backend::Tensor>), BackendError> {
        crate::qwen35::Qwen35Model::prefill_extract(self, backend, ids, layers)
    }
    fn decode(&mut self, backend: &dyn Backend, id: u32) -> Result<crate::backend::Tensor, BackendError> {
        crate::qwen35::Qwen35Model::decode(self, backend, id)
    }
    fn decode_spec(
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
        crate::qwen35::Qwen35Model::decode_spec(
            self,
            backend,
            id,
            draft,
            draft_probs,
            rng,
            extract_layers,
            greedy,
            hist,
            repeat_penalty,
        )
    }
}

/// Стоп-решение одного шага: EOG-токен, стоп-строка в накопленном тексте
/// или None (продолжать). Вынесено отдельно — детерминированный юнит.
pub fn check_stop(
    next: u32,
    is_eog: impl Fn(u32) -> bool,
    acc: &str,
    stop_strings: &[String],
) -> Option<FinishReason> {
    if is_eog(next) {
        return Some(FinishReason::Eog);
    }
    if stop_strings.iter().any(|s| acc.contains(s)) {
        return Some(FinishReason::StopString);
    }
    None
}

/// Один forward с PP-разбиением (7.5): эмбеддинг и lm_head — локально,
/// слои [0..from) — локально, [from..] + out_norm — на соседней ноде.
/// Позицию стадии ведёт локальный кэш [0..from) (sync_rope_to_cache) — то же
/// правило, что у prefill/decode без PP: префилл +t, декод-итерация +1.
fn pp_forward<M: CausalLm + ?Sized>(
    model: &mut M,
    backend: &dyn Backend,
    pp: &PpSplit,
    ids: &[u32],
) -> Result<crate::backend::Tensor, BackendError> {
    let t = ids.len();
    // Позиция ДО локального forward = стартовая позиция последовательности
    // для стадии (её кэш должен стоять там же). Запоминается и для ОТКАТА:
    // если стадия не приняла строки, локальные слои обязаны вернуться сюда,
    // иначе стороны разъезжаются на токен и все следующие вызовы падают на
    // проверке позиции (наблюдалось в кросс-машинном прогоне).
    let start_pos = model.stage_pos(0);
    let reset = pp
        .first
        .swap(false, std::sync::atomic::Ordering::Relaxed);
    let x = model.embed_ids(backend, ids)?;
    let dim = x.shape()[2];
    let mid = model.forward_layers_range(backend, x, t, Some(pp.from), 0, false)?;
    let hidden = backend.to_cpu_f32(&mid)?;
    let out = match pp.transport.forward_stage(pp.from, None, true, reset, start_pos, t, hidden) {
        Ok(v) => v,
        Err(e) => {
            // Транзакционный откат: строки, которые стадия не приняла,
            // снимаются и с локальных слоёв координатора.
            model.truncate_to(start_pos)?;
            return Err(BackendError::Shape(format!("PP-стадия: {e}")));
        }
    };
    if out.len() != t * dim {
        return Err(BackendError::Shape(format!(
            "PP-стадия: вернула {} элементов, ждали t*dim = {t}*{dim}",
            out.len()
        )));
    }
    let h = backend.from_cpu_f32(&out, vec![1, t, dim])?;
    // Локальный кэш [0..from) — источник истины по позиции: префилл даёт +t,
    // каждая декод-итерация +1 (как prefill/decode без PP).
    model.sync_rope_to_cache(0);
    model.lm_head(backend, &h)
}

/// Восстановление после сбоя стадии (7.11): обе стороны переигрывают ВСЮ
/// последовательность с чистого состояния — локальный сброс у координатора и
/// `reset` у соседа. Иначе позиции кэшей расходятся (координатор пересеял
/// последовательность локально, а у соседа остался старый хвост) и стадия
/// честно отвечает «позиция N != запрошенной M». Возвращает логиты последней
/// позиции; если стадия снова не сработала — ошибка (вызывающий уходит в
/// локальный откат).
fn pp_recover<M: CausalLm + ?Sized>(
    model: &mut M,
    backend: &dyn Backend,
    pp: &PpSplit,
    seq: &[u32],
) -> Result<crate::backend::Tensor, BackendError> {
    model.reset_cache(backend)?;
    // Следующая стадия обязана начать последовательность заново.
    pp.first.store(true, std::sync::atomic::Ordering::Relaxed);
    pp_forward(model, backend, pp, seq)
}

/// Локальный forward после отказа PP (7.10). `seq` — ВСЯ обработанная
/// последовательность (промпт + порождённое + текущий токен): в PP-режиме
/// локально жили только слои [0..from), поэтому продолжать с текущим кэшем
/// нельзя — нужен полный локальный префилл с чистого кэша (логиты последней
/// строки — то, что вернул бы decode).
fn local_fallback<M: CausalLm + ?Sized>(
    model: &mut M,
    backend: &dyn Backend,
    seq: &[u32],
) -> Result<crate::backend::Tensor, BackendError> {
    model.reset_cache(backend)?;
    if seq.len() > 1 {
        model.prefill_last(backend, seq)
    } else {
        model.decode(backend, seq[0])
    }
}

/// Выбор следующего токена по строке логитов: жадный (со штрафом за
/// повторы — единая реализация `sampling::greedy_next`, её же использует
/// жадная верификация спека) или сэмплинг top-p/top-k.
fn sample_next(
    backend: &dyn Backend,
    logits: &crate::backend::Tensor,
    vocab: usize,
    params: &GenerateParams,
    history: &[u32],
    rng_state: &mut Option<u64>,
) -> Result<u32, BackendError> {
    let v = backend.to_cpu_f32(logits)?;
    if v.len() < vocab || vocab == 0 {
        return Err(BackendError::Shape("sample: empty logits row".into()));
    }
    let last = &v[v.len() - vocab..];
    Ok(if params.temperature <= 0.0 {
        crate::inference::sampling::greedy_next(last, history, params.repeat_penalty)
    } else {
        let sp = SamplingParams {
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k,
            repetition_penalty: params.repeat_penalty,
            // Поток несёт `rng_state` (сид уже материализован) — второй
            // источник сида дал бы двойной шаг LCG.
            seed: None,
            min_p: params.min_p,
        };
        top_p_sample_vec_history(last, &sp, history, rng_state)
    })
}

enum Emit {
    Ok,
    Stop,
}

/// Отдать токен наружу: дельта текста (склейка UTF-8) + сырой токен.
/// Stop — потребитель попросил остановиться.
fn emit(
    t: u32,
    tok: &Tokenizer,
    decoder: &mut IncrementalDecoder,
    acc: &mut String,
    on_event: &mut impl FnMut(GenerateEvent) -> bool,
) -> Emit {
    // Сырые байты: токен может нести обрывок многобайтного символа —
    // decode_piece превратил бы его в U+FFFD до декодера.
    let delta = decoder.push(&tok.piece_bytes(t));
    if !delta.is_empty() {
        acc.push_str(&delta);
        if !on_event(GenerateEvent::Delta(delta)) {
            return Emit::Stop;
        }
    }
    if !on_event(GenerateEvent::Token(t)) {
        return Emit::Stop;
    }
    Emit::Ok
}

/// Сгенерировать продолжение после `ids` (полный префилл выполняется
/// здесь). Возвращает сгенерированные токены; события стримятся в
/// `on_event`. Greedy при temperature <= 0.
pub fn generate<M: CausalLm + ?Sized>(
    model: &mut M,
    backend: &dyn Backend,
    tok: &Tokenizer,
    ids: &[u32],
    params: &GenerateParams,
    mut on_event: impl FnMut(GenerateEvent) -> bool,
) -> Result<Vec<u32>, BackendError> {
    let t_start = std::time::Instant::now();
    let mut spec_stats = SpecStats::default();
    // Vision-префилл (2.4b): картинка уже посчитана — идём через M-RoPE.
    let mut logits = if let Some(pp) = params.pp.clone() {
        // PP (7.5): средние слои — на соседе. Vision-путь PP пока не
        // поддержан (M-RoPE/позиции картинки требуют передачи метаданных).
        if params.vision.is_some() {
            return Err(BackendError::Shape(
                "PP-разбиение не поддержано вместе с vision-промптом".into(),
            ));
        }
        // Архитектура без PP (гибрид) — явная ошибка ДО сети: иначе отказ
        // эмбеддингов ушёл бы в «сосед недоступен → локальный откат» и
        // молча подменил распределённый режим локальным.
        model.embed_ids(backend, &ids[..1])?;
        match pp_forward(model, backend, &pp, ids) {
            Ok(lg) => lg,
            Err(e) => {
                // 7.10: сосед недоступен/ошибся — деградируем в локальную
                // генерацию (модель у координатора полная), а не рвём запрос.
                tracing::warn!(error = %e, "PP-стадия недоступна — локальная генерация");
                local_fallback(model, backend, ids)?
            }
        }
    } else if let Some(v) = &params.vision {
        model.prefill_with_vision(
            backend,
            &v.ids,
            &v.vemb,
            &v.vds,
            v.vision_token,
            &v.pos4d,
            v.pos_after,
        )?
    } else if ids.len() > 1 {
        // DFlash (4.2.c, z-lab): первый блок контекстуализируется НА ВСЁМ
        // промпте — extract префилла отдаём драфтеру сразу (иначе первый
        // block только «видит» past_len=0 и не принимает ничего).
        let tlayers: Option<Vec<usize>> = params
            .drafter
            .as_ref()
            .map(|d| d.lock().unwrap().target_layers().to_vec());
        match tlayers {
            Some(tl) if !tl.is_empty() => {
                let (lg, extract) = model.prefill_extract(backend, ids, &tl)?;
                let n = ids.len();
                if let Some(dr) = params.drafter.as_ref() {
                    let mut dr = dr.lock().unwrap();
                    let _ = dr.after_extract(backend, &extract, n, n);
                }
                lg
            }
            _ => model.prefill_last(backend, ids)?,
        }
    } else {
        model.decode(backend, ids[0])?
    };
    let vocab = model.vocab_size();
    // Логиты префилла — на GPU, их чтение (первый сэмплинг) уже часть декода;
    // сам префилл синхронен (forward ждёт command buffer).
    let t_prompt = t_start.elapsed();
    let mut t_last_token = t_prompt;
    let mut eog_sampled = 0usize;

    // Диагностика параметров (LLM_DBG_PARAMS=1): расхождения спек/жадной
    // генерации на живом сервисе искали именно сравнением эффективных
    // параметров — печатаем те, что реально доехали до петли.
    if std::env::var("LLM_DBG_PARAMS").is_ok() {
        eprintln!(
            "PARAMS temp={} top_p={} top_k={} rep={} spec_gamma={} seed={:?} max_tokens={} t={} ids={:?}",
            params.temperature,
            params.top_p,
            params.top_k,
            params.repeat_penalty,
            params.spec_gamma,
            params.seed,
            params.max_tokens,
            ids.len(),
            &ids[..ids.len().min(12)],
        );
    }
    let mut generated: Vec<u32> = Vec::new();
    // Сид запроса материализуется в состояние один раз на генерацию: и
    // спекулятивный, и обычный путь сэмплят из этого одного потока.
    let mut rng_state: Option<u64> = params
        .seed
        .map(crate::inference::sampling::state_from_seed);
    let mut decoder = IncrementalDecoder::new();
    let mut acc = String::new(); // накопленный текст (для stop-строк)
    // Черновики выбираются так же, как их проверяет таргет: жадная проверка
    // ⇒ argmax-черновики, иначе — то же распределение (температура, top-k).
    let draft_sampling = if params.temperature <= 0.0 {
        crate::spec::DraftSampling::Greedy
    } else {
        crate::spec::DraftSampling::Sample {
            temperature: params.temperature,
            top_k: params.top_k,
        }
    };
    // История для repetition penalty: промпт + уже порождённые токены
    // (раньше params.repeat_penalty молча игнорировался).
    let mut history: Vec<u32> = ids.to_vec();
    let mut finish: Option<FinishReason> = None;

    // Якорь следующего спек-окна: при отказе черновика пересэмпл НЕ
    // прогоняется отдельным decode — он становится первым токеном следующей
    // верификации (z-lab dflash_generate, llama.cpp speculative: posterior =
    // якорь). Отдельный decode стоил лишний forward таргета на каждый отказ
    // и оставлял DFlash-драфтер без фич этого токена (дыра в контексте).
    let mut pending: Option<u32> = None;

    // Бюджет рассуждений (как сэмплер reasoning-budget llama.cpp): в
    // FORCING следующий токен — принудительный (проходит через forward,
    // попадает в вывод и историю как обычный); каждый выданный токен
    // учитывается автоматом.
    let mut budget = params.reasoning_budget.clone();

    // Лимит — по числу ВЫДАННЫХ токенов, а не итераций: спек-шаг выдаёт
    // до 1+γ токенов за итерацию, и счёт итераций пропускал лимит в разы
    // (64 запрошено — 98 выдано на Qwen3.5 с N-грамм-драфтером).
    while generated.len() < params.max_tokens {
        // Якорь от отказа спек-окна — выбор модели, сделанный ДО того, как
        // автомат потребовал вставку (окно обрезается по draft_room), так что
        // принуждение его не отменяет.
        let next = match pending.take() {
            Some(p) => p,
            None => match budget.as_ref().and_then(|b| b.forced_token()) {
                Some(f) => f,
                None => sample_next(backend, &logits, vocab, params, &history, &mut rng_state)?,
            },
        };
        if let Some(b) = budget.as_mut() {
            b.accept_piece(next, &tok.piece_bytes(next));
        }
        if let Some(r) = check_stop(next, |t| tok.is_eog(t), &acc, &params.stop_strings) {
            if r == FinishReason::Eog {
                eog_sampled = 1;
                t_last_token = t_start.elapsed();
            }
            finish = Some(r);
            break;
        }
        match emit(next, tok, &mut decoder, &mut acc, &mut on_event) {
            Emit::Stop => return Ok(generated),
            Emit::Ok => {}
        }
        generated.push(next);
        history.push(next);
        t_last_token = t_start.elapsed();
        if generated.len() >= params.max_tokens {
            break;
        }

        // Спек-декодинг (4.2b): проверка черновиков ОДНИМ forward'ом
        // (decode_spec: rejection sampling по Leviathan). PP (7.5) и
        // спек-декод несовместимы: верификация гоняет forward ЦЕЛИКОМ по
        // полному локальному KV, а в PP-режиме локально живут только слои
        // [0..from) — черновики проверялись бы по пустому кэшу.
        // Бюджет рассуждений: черновики проверяются только до места, где
        // автомат потребует вставку (принятые шли бы мимо неё); пока идёт
        // сама вставка, окно не открывается.
        let room = budget.as_ref().map(|b| b.forced_token().is_none());
        if let (Some(drafter), true) = (
            &params.drafter,
            params.spec_gamma > 0 && params.pp.is_none() && room != Some(false),
        ) {
            let (mut draft_tokens, mut draft_probs) = {
                let mut dr = drafter.lock().unwrap();
                dr.set_sampling(draft_sampling);
                dr.draft(backend, &history, params.spec_gamma)
            };
            if let Some(b) = budget.as_ref() {
                let k = b.draft_room(&draft_tokens);
                draft_tokens.truncate(k);
                draft_probs.truncate(k);
            }
            if !draft_tokens.is_empty() {
                let tlayers: Vec<usize> = {
                    let dr = drafter.lock().unwrap();
                    dr.target_layers().to_vec()
                };
                let out = model.decode_spec(
                    backend,
                    next,
                    &draft_tokens,
                    &draft_probs,
                    &mut rng_state,
                    &tlayers,
                    // temperature <= 0: та же жадная траектория, что у decode
                    // (включая штраф за повторы — верификация без него
                    // принимала бы токены, которых жадный путь не выбрал бы).
                    params.temperature <= 0.0,
                    &history,
                    params.repeat_penalty,
                )?;
                // Проверенные черновики (ровно те, что лежат в кэше таргета).
                let n_ok = out.n_draft_accepted + 1; // якорь + проверенные
                if !tlayers.is_empty() {
                    let pos = model.context_len();
                    let mut dr = drafter.lock().unwrap();
                    let _ = dr.after_extract(backend, &out.extract, pos, n_ok);
                }
                spec_stats.drafts += draft_tokens.len() as u64;
                let nv = out.n_draft_verified;
                spec_stats.accepted += nv as u64;
                spec_stats.rejected += (draft_tokens.len() - nv) as u64;
                // Принятые черновики — настоящие токены: стримим их (не
                // дальше лимита max_tokens).
                for &t in &out.accepted[1..n_ok] {
                    if generated.len() >= params.max_tokens {
                        break;
                    }
                    if let Some(b) = budget.as_mut() {
                        b.accept_piece(t, &tok.piece_bytes(t));
                    }
                    if let Some(r) = check_stop(t, |x| tok.is_eog(x), &acc, &params.stop_strings) {
                        if r == FinishReason::Eog {
                            eog_sampled = 1;
                            t_last_token = t_start.elapsed();
                        }
                        finish = Some(r);
                        break;
                    }
                    match emit(t, tok, &mut decoder, &mut acc, &mut on_event) {
                        Emit::Stop => return Ok(generated),
                        Emit::Ok => {}
                    }
                    generated.push(t);
                    history.push(t);
                    t_last_token = t_start.elapsed();
                }
                if finish.is_some() {
                    break;
                }
                // Окно закрылось отказом: пересэмпл (последний элемент
                // accepted) в кэше НЕ лежит — он якорь следующей итерации.
                // Полное принятие: логиты последней строки — продолжение.
                if n_ok < out.accepted.len() {
                    pending = out.accepted.last().copied();
                } else {
                    logits = out.logits;
                }
                continue;
            }
        }
        logits = if let Some(pp) = params.pp.as_ref() {
            match pp_forward(model, backend, pp, &[next]) {
                Ok(lg) => lg,
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "PP-стадия не сработала — переигрываем последовательность"
                    );
                    match pp_recover(model, backend, pp, &history) {
                        Ok(lg) => lg,
                        Err(e2) => {
                            tracing::warn!(error = %e2, "PP-стадия недоступна — локальный decode");
                            local_fallback(model, backend, &history)?
                        }
                    }
                }
            }
        } else {
            model.decode(backend, next)?
        };
    }
    if finish.is_none() {
        finish = Some(FinishReason::MaxTokens);
    }
    let tail = decoder.finish();
    if !tail.is_empty() && !on_event(GenerateEvent::Delta(tail)) {
        return Ok(generated);
    }
    if params.spec_gamma > 0 && params.drafter.is_some() {
        // Диагностика приёмки (LLM_DBG_PARAMS=1): без неё «спек не дал выигрыша»
        // неотличим от «черновики не предлагались» — а это разные дефекты.
        if std::env::var("LLM_DBG_PARAMS").is_ok() {
            eprintln!(
                "SPEC drafts={} accepted={} rejected={}",
                spec_stats.drafts, spec_stats.accepted, spec_stats.rejected
            );
        }
        on_event(GenerateEvent::SpecStats(spec_stats));
    }
    on_event(GenerateEvent::Timings(Timings {
        prompt_n: ids.len(),
        prompt_ms: t_prompt.as_secs_f64() * 1e3,
        predicted_n: generated.len() + eog_sampled,
        predicted_ms: (t_last_token.saturating_sub(t_prompt)).as_secs_f64() * 1e3,
    }));
    on_event(GenerateEvent::Done(finish.unwrap()));
    Ok(generated)
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use crate::metal::MetalBackend;
    use crate::qwen3::{Qwen3Config, Qwen3Model};
    use crate::weights_map::{WeightTensor, WeightsMap};

    fn mini_model() -> (MetalBackend, Qwen3Model) {
        let backend = MetalBackend::new().unwrap();
        let c = Qwen3Config {
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
        };
        let mut rng: u64 = 0xC0FFEE;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut rand_vec = |n: usize| {
            (0..n)
                .map(|_| ((next() >> 32) as f32 / (1u64 << 32) as f32) * 2.0 - 1.0)
                .collect::<Vec<f32>>()
        };
        let mut storage: Vec<(String, Vec<f32>, Vec<usize>)> = vec![(
            "token_embd.weight".into(),
            rand_vec(c.vocab_size * c.dim),
            vec![c.vocab_size, c.dim],
        )];
        for i in 0..c.n_layers {
            let p = |s: &str| format!("blk.{i}.{s}");
            for (n, shape) in [
                ("attn_norm.weight", vec![c.dim]),
                ("attn_q.weight", vec![c.dim, c.dim]),
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
                storage.push((p(n), rand_vec(numel), shape));
            }
        }
        storage.push(("output_norm.weight".into(), rand_vec(c.dim), vec![c.dim]));
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
        let model = Qwen3Model::load(&backend, &wm, c).unwrap();
        (backend, model)
    }

    fn tiny_tokenizer() -> crate::tokenizer::Tokenizer {
        crate::tokenizer::Tokenizer::from_parts(
            (0..8).map(|i| vec![b'a' + i as u8]).collect(),
            vec![],
            None,
            None,
            None,
        )
    }

    #[test]
    fn greedy_generates_until_max_tokens() {
        let (backend, mut model) = mini_model();
        let mut deltas = 0usize;
        let mut done = None;
        let params = GenerateParams {
            max_tokens: 3,
            temperature: 0.0,
            ..Default::default()
        };
        let gen_tokens = generate(
            &mut model,
            &backend,
            &tiny_tokenizer(),
            &[1, 2],
            &params,
            |ev| {
                match ev {
                    GenerateEvent::Delta(_) => deltas += 1,
                    GenerateEvent::Done(r) => done = Some(r),
                    _ => {}
                }
                true
            },
        )
        .unwrap();
        assert_eq!(gen_tokens.len(), 3, "max_tokens reached");
        assert_eq!(done, Some(FinishReason::MaxTokens));
        assert!(deltas > 0, "дельты текста не пришли");
    }

    /// Бюджет рассуждений: рассуждение открыто промптом (генпрефикс
    /// прогнан через автомат), после N токенов петля вставляет
    /// принудительную последовательность автомата (сообщение + «</think>»);
    /// дальше — обычный ответ модели.
    #[test]
    fn reasoning_budget_forces_close() {
        use crate::inference::reasoning_budget::ReasoningBudget;
        let (backend, mut model) = mini_model();
        // словарь 8 токенов: 0..4 — буквы, 5 — «<think>», 6 — «</think>», 7 — «\n»
        let mut vocab: Vec<Vec<u8>> = (0..5).map(|i| vec![b'a' + i as u8]).collect();
        vocab.push(b"<think>".to_vec());
        vocab.push(b"</think>".to_vec());
        vocab.push("\u{010A}".as_bytes().to_vec()); // byte-level «\n»
        let mut tok = crate::tokenizer::Tokenizer::from_parts(vocab, vec![], None, None, None);
        tok.add_special(b"<think>".to_vec(), 5);
        tok.add_special(b"</think>".to_vec(), 6);
        let mut budget = ReasoningBudget::for_tokenizer(&tok, "<think>", &["</think>"], "\n", 3).unwrap();
        budget.prefill(&tok, &tok.encode("<think>\n"));
        let params = GenerateParams {
            max_tokens: 12,
            temperature: 0.0,
            reasoning_budget: Some(budget),
            ..Default::default()
        };
        let out = generate(&mut model, &backend, &tok, &[1, 2], &params, |_| true).unwrap();
        // «\n» генпрефикса после <think> уже съел 1 из 3 (как в llama.cpp:
        // токены префикса проходят через автомат) — модели остаётся 2, затем
        // «\n» «</think>» (случайная мини-модель сама не закроет), после —
        // снова генерация модели
        assert_eq!(&out[2..4], &[7, 6], "вставка после 2 токенов: {out:?}");
        assert!(!out[..2].contains(&6), "{out:?}");
        assert_eq!(out.len(), 12, "после вставки генерация продолжается: {out:?}");
    }

    #[test]
    fn check_stop_eog_and_stop_string() {
        assert_eq!(check_stop(2, |t| t == 2, "", &[]), Some(FinishReason::Eog));
        assert_eq!(check_stop(1, |t| t == 2, "", &[]), None);
        assert_eq!(
            check_stop(7, |_| false, "hello world", &["world".into()]),
            Some(FinishReason::StopString)
        );
        assert_eq!(check_stop(7, |_| false, "hello", &["world".into()]), None);
    }

    /// Timings: приходит перед Done; prompt_n — длина промпта; predicted_n
    /// — выданные токены (+ сэмплированный EOG, как n_decoded llama-server);
    /// декод не включает префилл.
    #[test]
    fn timings_split_prefill_and_decode() {
        let (backend, mut model) = mini_model();
        let mut events = Vec::new();
        let params = GenerateParams { max_tokens: 5, temperature: 0.0, ..Default::default() };
        let out = generate(&mut model, &backend, &tiny_tokenizer(), &[1, 2, 3], &params, |ev| {
            events.push(ev);
            true
        })
        .unwrap();
        let pos_t = events.iter().position(|e| matches!(e, GenerateEvent::Timings(_))).expect("Timings");
        let pos_d = events.iter().position(|e| matches!(e, GenerateEvent::Done(_))).expect("Done");
        assert!(pos_t < pos_d, "Timings до Done");
        let GenerateEvent::Timings(t) = events[pos_t] else { unreachable!() };
        assert_eq!(t.prompt_n, 3);
        assert_eq!(t.predicted_n, out.len(), "без EOG: predicted_n = выданные токены");
        assert!(t.prompt_ms > 0.0 && t.predicted_ms > 0.0, "{t:?}");
        assert_eq!(t.decode_steps(), out.len() - 1);

        // EOG сэмплирован: в текст не попадает, но в predicted_n считается
        // (жадный первый токен того же промпта объявлен eos)
        let tok = crate::tokenizer::Tokenizer::from_parts(
            (0..8).map(|i| vec![b'a' + i as u8]).collect(),
            vec![],
            None,
            Some(out[0]),
            None,
        );
        model.reset_cache(&backend).unwrap();
        let mut timings = None;
        let out2 = generate(&mut model, &backend, &tok, &[1, 2, 3], &params, |ev| {
            if let GenerateEvent::Timings(t) = ev {
                timings = Some(t);
            }
            true
        })
        .unwrap();
        assert!(out2.is_empty(), "первый же токен — EOG: {out2:?}");
        assert_eq!(timings.unwrap().predicted_n, 1);
    }

    #[test]
    fn stream_emits_deltas_and_final_done() {
        let (backend, mut model) = mini_model();
        let mut events = Vec::new();
        let params = GenerateParams {
            max_tokens: 2,
            temperature: 0.0,
            ..Default::default()
        };
        generate(
            &mut model,
            &backend,
            &tiny_tokenizer(),
            &[1],
            &params,
            |ev| {
                events.push(ev);
                true
            },
        )
        .unwrap();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, GenerateEvent::Done(FinishReason::MaxTokens)))
        );
        assert!(events.iter().any(|e| matches!(e, GenerateEvent::Token(_))));
    }
}
