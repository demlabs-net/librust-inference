//! Драйвер пакетной генерации по слотам: несколько бесед идут ОДНИМ пакетным
//! шагом (строка на беседу), веса читаются один раз на пакет.
//!
//! Разделение ответственности: драйвер владеет состоянием (KV + рекуррентные
//! стейты GDN в слотах) и считает forward; ПОЛИТИКА (сэмплинг, стоп-условия,
//! штрафы, EOG) — у вызывающего: он получает логиты каждой беседы и решает,
//! какой токен подать следующим или что беседа закончена.
//!
//! ```ignore
//! let mut eng = BatchEngine::new(&mut model, &backend, 4)?;
//! let (slot, logits) = eng.submit(&backend, &prompt_ids)?;
//! let mut next = vec![(slot, argmax(&logits))];
//! loop {
//!     let rows = eng.step(&backend, &next)?;   // один forward на все беседы
//!     next = rows.iter().map(|(s, lg)| (*s, argmax(lg))).collect();
//! }
//! ```
//!
//! Пакет из одной беседы идёт одиночным путём (см.
//! [`Qwen35Model::decode_batch`]) — биты совпадают с обычным декодом.

use std::collections::VecDeque;
use std::time::Instant;

use crate::backend::Backend;
use crate::error::BackendError;
use crate::inference::generate::{
    Emit, FinishReason, GenerateEvent, GenerateParams, Timings, check_stop, emit,
};
use crate::inference::sampling::state_from_seed;
use crate::qwen35::{Qwen35Model, Qwen35Slot};
use crate::tokenizer::{IncrementalDecoder, Tokenizer};

/// Учёт беседы в слоте (позиция живёт в самом слоте).
#[derive(Debug, Clone, Copy)]
pub struct SlotMeta {
    /// Длина поданного промпта, токенов.
    pub prompt_len: usize,
    /// Сколько токенов уже сгенерировано пакетными шагами.
    pub tokens_out: usize,
}

/// Пакетный движок: модель + фиксированный набор слотов.
pub struct BatchEngine<'m> {
    model: &'m mut Qwen35Model,
    slots: Vec<Qwen35Slot>,
    meta: Vec<Option<SlotMeta>>,
}

impl<'m> BatchEngine<'m> {
    /// Создать движок на `capacity` мест. Состояние каждого слота (KV + GDN)
    /// выделяется сразу: для 4B это ~50 МБ, для 27B — сотни МБ на слот.
    pub fn new(
        model: &'m mut Qwen35Model,
        backend: &dyn Backend,
        capacity: usize,
    ) -> Result<Self, BackendError> {
        if capacity == 0 {
            return Err(BackendError::Shape("BatchEngine: capacity = 0".into()));
        }
        let mut slots = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            slots.push(model.new_slot(backend)?);
        }
        Ok(Self {
            model,
            slots,
            meta: vec![None; capacity],
        })
    }

    /// Движок на ГОТОВЫХ слотах: сервис держит их между пакетами — состояние
    /// последовательностей выделяется один раз, а не на каждый пакет.
    pub fn with_slots(model: &'m mut Qwen35Model, slots: Vec<Qwen35Slot>) -> Self {
        let meta = vec![None; slots.len()];
        Self { model, slots, meta }
    }

    /// Забрать слоты обратно (состояние бесед уже закрыто/освобождено).
    pub fn into_slots(self) -> Vec<Qwen35Slot> {
        self.slots
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Сколько бесед сейчас в работе.
    pub fn active(&self) -> usize {
        self.meta.iter().filter(|m| m.is_some()).count()
    }

    /// Индексы занятых слотов по возрастанию (порядок строк пакета в [`Self::step`]
    /// задаёт сам вызывающий, но этот список удобен для сборки `next`).
    pub fn active_slots(&self) -> Vec<usize> {
        self.meta
            .iter()
            .enumerate()
            .filter_map(|(i, m)| m.map(|_| i))
            .collect()
    }

    /// Предел контекста модели (окно, из метаданных GGUF).
    pub fn max_seq(&self) -> usize {
        self.model.cfg.max_seq
    }

    pub fn meta(&self, slot: usize) -> Option<SlotMeta> {
        self.meta.get(slot).copied().flatten()
    }

    /// Позиция (длина контекста) беседы в слоте.
    pub fn pos(&self, slot: usize) -> Option<usize> {
        self.meta.get(slot).and_then(|m| {
            m.map(|_| {
                // Позиция — в самом слоте; при его отсутствии (out of bounds) None.
                self.slots[slot].pos()
            })
        })
    }

    /// Принять беседу: prefill в свободный слот, отдать логиты последней
    /// позиции ([V], на хосте). Свободных слотов нет — внятная ошибка
    /// (очередь держит вызывающий).
    pub fn submit(
        &mut self,
        backend: &dyn Backend,
        ids: &[u32],
    ) -> Result<(usize, Vec<f32>), BackendError> {
        if ids.is_empty() {
            return Err(BackendError::Shape(
                "BatchEngine::submit: пустой промпт".into(),
            ));
        }
        let slot =
            self.meta.iter().position(|m| m.is_none()).ok_or_else(|| {
                BackendError::Shape("BatchEngine::submit: все слоты заняты".into())
            })?;
        let logits = self
            .model
            .prefill_slot_last(backend, &mut self.slots[slot], ids)?;
        let host = backend.to_cpu_f32(&logits)?;
        self.meta[slot] = Some(SlotMeta {
            prompt_len: ids.len(),
            tokens_out: 0,
        });
        Ok((slot, host))
    }

    /// Пакетный шаг: по токену на активную беседу. `next` покрывает РОВНО
    /// активные слоты (пары (слот, токен)); порядок пар задаёт порядок строк
    /// результата. Логиты возвращаются на хосте, по строке [V] на беседу.
    pub fn step(
        &mut self,
        backend: &dyn Backend,
        next: &[(usize, u32)],
    ) -> Result<Vec<(usize, Vec<f32>)>, BackendError> {
        if next.is_empty() {
            return Err(BackendError::Shape(
                "BatchEngine::step: пустой пакет".into(),
            ));
        }
        if next.len() != self.active() {
            return Err(BackendError::Shape(format!(
                "BatchEngine::step: {} пар против {} активных бесед",
                next.len(),
                self.active()
            )));
        }
        let mut seen = vec![false; self.slots.len()];
        let mut active = Vec::with_capacity(next.len());
        let mut ids = Vec::with_capacity(next.len());
        for &(slot, id) in next {
            if slot >= self.slots.len() {
                return Err(BackendError::Shape(format!(
                    "BatchEngine::step: нет слота {slot} (мест {})",
                    self.slots.len()
                )));
            }
            if self.meta[slot].is_none() {
                return Err(BackendError::Shape(format!(
                    "BatchEngine::step: слот {slot} свободен"
                )));
            }
            if seen[slot] {
                return Err(BackendError::Shape(format!(
                    "BatchEngine::step: слот {slot} дважды в пакете"
                )));
            }
            seen[slot] = true;
            active.push(slot);
            ids.push(id);
        }
        let logits = self
            .model
            .decode_batch_active(backend, &mut self.slots, &active, &ids)?;
        let host = backend.to_cpu_f32(&logits)?;
        let v = self.model.cfg.vocab_size;
        let mut out = Vec::with_capacity(active.len());
        for (row, &slot) in active.iter().enumerate() {
            let from = row * v;
            out.push((slot, host[from..from + v].to_vec()));
            if let Some(m) = self.meta[slot].as_mut() {
                m.tokens_out += 1;
            }
        }
        Ok(out)
    }

    /// Окно верификации ОДНОЙ беседы: `decode_spec` на её состоянии
    /// (`with_slot`, копий нет). Откат KV, переигрыш GDN-стейта принятыми
    /// рядами и проверка черновиков — внутри движка; здесь только подстановка
    /// состояния нужного слота.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_spec_slot(
        &mut self,
        backend: &dyn Backend,
        slot: usize,
        anchor: u32,
        draft: &[u32],
        draft_probs: &[f32],
        greedy: bool,
        hist: &[u32],
        sp: &crate::inference::sampling::SamplingParams,
    ) -> Result<crate::qwen3::SpecOutcome, BackendError> {
        if slot >= self.slots.len() {
            return Err(BackendError::Shape(format!(
                "decode_spec_slot: нет слота {slot} (мест {})",
                self.slots.len()
            )));
        }
        let m = &mut *self.model;
        let s = &mut self.slots[slot];
        m.with_slot(s, |m| {
            m.decode_spec(backend, anchor, draft, draft_probs, &mut None, &[], greedy, hist, sp)
        })
    }

    /// Один AR-шаг ОДНОЙ беседы на её состоянии (диагностика/сверка окон).
    pub fn decode_step_slot(
        &mut self,
        backend: &dyn Backend,
        slot: usize,
        id: u32,
    ) -> Result<crate::backend::Tensor, BackendError> {
        if slot >= self.slots.len() {
            return Err(BackendError::Shape(format!(
                "decode_step_slot: нет слота {slot} (мест {})",
                self.slots.len()
            )));
        }
        let m = &mut *self.model;
        let s = &mut self.slots[slot];
        m.with_slot(s, |m| m.decode(backend, id))
    }

    /// Освободить слот: состояние беседы выбрасывается, слот готов принять
    /// следующую (тензоры состояния уходят в пул бэкенда).
    pub fn finish(&mut self, slot: usize, backend: &dyn Backend) -> Result<(), BackendError> {
        if slot >= self.slots.len() {
            return Err(BackendError::Shape(format!(
                "BatchEngine::finish: нет слота {slot} (мест {})",
                self.slots.len()
            )));
        }
        self.meta[slot] = None;
        self.slots[slot] = self.model.new_slot(backend)?;
        Ok(())
    }
}

/// Запрос пакетной генерации: промпт + параметры.
///
/// Пакетный путь — обычный AR-декод по одному токену на беседу. Спекулятивный
/// драфтер, распределённый PP и vision-префилл в пакете не поддержаны
/// (у каждого — своё состояние/транспорт на последовательность) и дают явную
/// ошибку, а не тихую подмену.
pub struct BatchRequest {
    pub ids: Vec<u32>,
    pub params: GenerateParams,
}

/// Состояние одной беседы в пакете (сэмплинг, стоп, декодер UTF-8).
struct Stream {
    req: usize,
    decoder: IncrementalDecoder,
    acc: String,
    history: Vec<u32>,
    rng: Option<u64>,
    generated: Vec<u32>,
    budget: Option<crate::inference::reasoning_budget::ReasoningBudget>,
    /// Токен, который надо подать следующим шагом.
    next: u32,
    /// Логиты, УЖЕ полученные окном верификации (все черновики приняты):
    /// следующий токен сэмплится прямо из них, forward не нужен.
    pending: Option<Vec<f32>>,
    t_prompt: f64,
    t_start: Instant,
    predicted_n: usize,
}

/// Сгенерировать продолжения для нескольких промптов ПАКЕТОМ: по токену на
/// беседу за шаг, веса читаются один раз на пакет, `capacity` бесед идут
/// одновременно (остальные ждут в очереди и занимают место по мере
/// завершения). Сэмплинг/стоп-условия/EOG/штрафы — на КАЖДУЮ беседу свои,
/// ровно тот же каскад, что у [`crate::inference::generate::generate`].
///
/// События стримятся с индексом запроса: `on_event(req_index, event)`;
/// `false` из колбэка — отмена ЭТОЙ беседы (остальные продолжают).
///
/// Возвращает токены каждого запроса (в порядке `reqs`).
pub fn generate_batch(
    engine: &mut BatchEngine<'_>,
    backend: &dyn Backend,
    tok: &Tokenizer,
    reqs: &[BatchRequest],
    on_event: &mut impl FnMut(usize, GenerateEvent) -> bool,
) -> Result<Vec<Vec<u32>>, BackendError> {
    if reqs.is_empty() {
        return Ok(Vec::new());
    }
    for (i, r) in reqs.iter().enumerate() {
        if r.ids.is_empty() {
            return Err(BackendError::Shape(format!(
                "generate_batch: запрос {i}: пустой промпт"
            )));
        }
        if r.params.drafter.is_some() || r.params.pp.is_some() || r.params.vision.is_some() {
            return Err(BackendError::Shape(format!(
                "generate_batch: запрос {i}: спек/PP/vision в пакетном пути не поддержаны"
            )));
        }
    }
    let mut pending: VecDeque<usize> = (0..reqs.len()).collect();
    let mut streams: Vec<Option<Stream>> = (0..engine.capacity()).map(|_| None).collect();
    let mut out: Vec<Vec<u32>> = vec![Vec::new(); reqs.len()];
    let mut queue: Vec<(usize, u32)> = Vec::new();

    // Принять беседы в свободные места: prefill + первый токен.
    loop {
        while let Some(&q) = pending.front() {
            if !streams.iter().any(|s| s.is_none()) {
                break;
            }
            pending.pop_front();
            let t_prompt = Instant::now();
            let (slot, logits) = engine.submit(backend, &reqs[q].ids)?;
            let st = Stream {
                req: q,
                decoder: IncrementalDecoder::new(),
                acc: String::new(),
                history: reqs[q].ids.clone(),
                rng: reqs[q].params.seed.map(state_from_seed),
                generated: Vec::new(),
                budget: reqs[q].params.reasoning_budget.clone(),
                next: 0,
                pending: None,
                t_prompt: t_prompt.elapsed().as_secs_f64() * 1e3,
                t_start: Instant::now(),
                predicted_n: 0,
            };
            streams[slot] = Some(st);
            let done = advance(
                streams[slot].as_mut().expect("слот только что занят"),
                &logits,
                &reqs[q].params,
                tok,
                &mut out[q],
                on_event,
            );
            match done {
                None => queue.push((slot, streams[slot].as_ref().unwrap().next)),
                Some(reason) => {
                    close_stream(streams[slot].as_mut().unwrap(), reason, on_event);
                    engine.finish(slot, backend)?;
                    streams[slot] = None;
                }
            }
        }
        if queue.is_empty() {
            break;
        }
        // Контекст беседы кончился — закрываем ЕЁ (как llama.cpp на переполнении
        // окна), а не роняем весь пакет: остальные беседы продолжают идти.
        let max_seq = engine.max_seq();
        let mut ready: Vec<(usize, u32)> = Vec::with_capacity(queue.len());
        for &(slot, tok_) in queue.iter() {
            let full = engine.pos(slot).map(|p| p + 1 > max_seq).unwrap_or(true);
            if full {
                if let Some(st) = streams[slot].as_mut() {
                    close_stream(st, FinishReason::MaxTokens, on_event);
                }
                engine.finish(slot, backend)?;
                streams[slot] = None;
            } else {
                ready.push((slot, tok_));
            }
        }
        queue = ready;
        // Спец (INFER_BATCH_SPEC=1, только жадные беседы): у кого есть
        // вербатимное продолжение в истории — окно верификации через
        // decode_spec, остальные идут пакетом. Путь аддитивный: без черновиков
        // всё как раньше.
        // Спец по беседам (INFER_BATCH_SPEC=1): окно верификации через
        // decode_spec на состоянии ОДНОГО слота, черновики — prompt-lookup.
        // Лосслессно для жадных бесед: приёмка сверяет токены с одиночным AR
        // (cap 3 и 6, γ=2/4/8). Остальные беседы в том же такте идут пакетом.
        // ВАЖНО: параметры сэмплинга для окна берутся из запроса — дефолтный
        // SamplingParams (repeat 1.1) выключал быструю GPU-проверку argmax.
        let spec_on = std::env::var("INFER_BATCH_SPEC").is_ok();
        let gamma: usize = std::env::var("INFER_BATCH_SPEC_GAMMA")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8);
        let mut still_ar: Vec<(usize, u32)> = Vec::with_capacity(queue.len());
        if spec_on {
            for &(slot, tok_) in queue.iter() {
                let st = streams[slot].as_mut().expect("шаг по занятому слоту");
                let q = st.req;
                let greedy = reqs[q].params.temperature <= 0.0;
                let forced = st.budget.as_ref().and_then(|b| b.forced_token()).is_some();
                if !greedy || forced {
                    still_ar.push((slot, tok_));
                    continue;
                }
                let (drafts, probs) = crate::spec::ngram_lookup_min(&st.history, gamma, 4, 4);
                if drafts.len() < 2 {
                    still_ar.push((slot, tok_));
                    continue;
                }
                let hist = st.history.clone();
                // Параметры — РОВНО те же, что у одиночного пути: иначе
                // SamplingParams::default() (repeat 1.1) выключает быструю
                // GPU-проверку argmax и подменяет приёмку.
                let p = &reqs[q].params;
                let sp = crate::inference::sampling::SamplingParams {
                    temperature: p.temperature,
                    top_p: p.top_p,
                    top_k: p.top_k,
                    repetition_penalty: p.repeat_penalty,
                    seed: None,
                    min_p: p.min_p,
                };
                let o = match engine
                    .decode_spec_slot(backend, slot, tok_, &drafts, &probs, true, &hist, &sp)
                {
                    Ok(o) => o,
                    // Окно не удалось — беседа честно идёт AR-шагом.
                    Err(_) => {
                        still_ar.push((slot, tok_));
                        continue;
                    }
                };
                // Конвенция SpecOutcome (как в одиночном generate):
                //   n_ok = n_draft_accepted + 1  — якорь + принятые черновики;
                //   accepted[0] — ЯКОРЬ (уже выдан), наружу идут [1..n_ok];
                //   accepted[n_ok] при отказе — пересэмпл, В KV ЕГО НЕТ.
                let n_ok = o.n_draft_accepted + 1;
                // Диагностика приёмки: без неё «спек не дал выигрыша»
                // неотличим от «черновики не предлагались».
                on_event(
                    q,
                    crate::inference::generate::GenerateEvent::SpecStats(
                        crate::inference::generate::SpecStats {
                            drafts: drafts.len() as u64,
                            accepted: o.n_draft_accepted as u64,
                            rejected: (drafts.len() - o.n_draft_accepted) as u64,
                        },
                    ),
                );
                let all = n_ok == o.accepted.len();
                let row = if all {
                    Some(backend.to_cpu_f32(&o.logits)?)
                } else {
                    None
                };
                let mut closed = None;
                for t in o.accepted[1..n_ok].iter().copied() {
                    match check_stop(t, |x| tok.is_eog(x), &st.acc, &reqs[q].params.stop_strings) {
                        Some(r) => {
                            closed = Some(r);
                            break;
                        }
                        None => {}
                    }
                    match emit(t, tok, &mut st.decoder, &mut st.acc, &mut |ev| on_event(q, ev)) {
                        Emit::Stop => {
                            closed = Some(FinishReason::Cancelled);
                            break;
                        }
                        Emit::Ok => {}
                    }
                    st.generated.push(t);
                    out[q].push(t);
                    st.history.push(t);
                    st.predicted_n += 1;
                    if st.generated.len() >= reqs[q].params.max_tokens {
                        closed = Some(FinishReason::MaxTokens);
                        break;
                    }
                }
                match closed {
                    Some(r) => {
                        close_stream(st, r, on_event);
                        engine.finish(slot, backend)?;
                        streams[slot] = None;
                    }
                    None => {
                        // Полная приёмка: логиты продолжения уже есть; иначе
                        // последний принятый — пересэмпл, НЕ в KV, его кормим.
                        // ВАЖНО: пересэмпл — обычный следующий токен беседы,
                        // его надо ВЫДАТЬ (в одиночном пути он приходит из
                        // `pending` и печатается на следующей итерации), а не
                        // только запомнить как якорь — иначе выход короче и
                        // история/выдача разъезжаются с AR.
                        st.next = *o.accepted.last().expect("окно вернуло токены");
                        if n_ok < o.accepted.len() {
                            let rs = st.next;
                            let mut stop_now = None;
                            if let Some(r) = check_stop(
                                rs,
                                |x| tok.is_eog(x),
                                &st.acc,
                                &reqs[q].params.stop_strings,
                            ) {
                                stop_now = Some(r);
                            } else {
                                match emit(rs, tok, &mut st.decoder, &mut st.acc, &mut |ev| {
                                    on_event(q, ev)
                                }) {
                                    Emit::Stop => stop_now = Some(FinishReason::Cancelled),
                                    Emit::Ok => {}
                                }
                                if stop_now.is_none() {
                                    st.generated.push(rs);
                                    out[q].push(rs);
                                    st.history.push(rs);
                                    st.predicted_n += 1;
                                    if st.generated.len() >= reqs[q].params.max_tokens {
                                        stop_now = Some(FinishReason::MaxTokens);
                                    }
                                }
                            }
                            if let Some(r) = stop_now {
                                close_stream(st, r, on_event);
                                engine.finish(slot, backend)?;
                                streams[slot] = None;
                                continue;
                            }
                        }
                        st.pending = row;
                        match st.pending.take() {
                            // Готовые логиты: сэмплим из них, forward не нужен.
                            Some(row) => match advance(
                                st,
                                &row,
                                &reqs[q].params,
                                tok,
                                &mut out[q],
                                on_event,
                            ) {
                                None => still_ar.push((slot, st.next)),
                                Some(r) => {
                                    close_stream(st, r, on_event);
                                    engine.finish(slot, backend)?;
                                    streams[slot] = None;
                                }
                            },
                            None => still_ar.push((slot, st.next)),
                        }
                    }
                }
            }
        } else {
            still_ar = std::mem::take(&mut queue);
        }
        queue = still_ar;
        if queue.is_empty() {
            continue;
        }
        let rows = engine.step(backend, &queue)?;
        queue.clear();
        for (slot, row) in rows {
            let st = streams[slot].as_mut().expect("шаг по занятому слоту");
            let q = st.req;
            let done = advance(st, &row, &reqs[q].params, tok, &mut out[q], on_event);
            match done {
                None => queue.push((slot, streams[slot].as_ref().unwrap().next)),
                Some(reason) => {
                    close_stream(streams[slot].as_mut().unwrap(), reason, on_event);
                    engine.finish(slot, backend)?;
                    streams[slot] = None;
                }
            }
        }
    }
    Ok(out)
}

/// Один шаг беседы по строке логитов: сэмплинг → стоп → эмиссия → учёт.
/// `None` — продолжаем (`st.next` обновлён), `Some(reason)` — беседа
/// завершена (выход и последний токен уже учтены).
fn advance(
    st: &mut Stream,
    row: &[f32],
    params: &GenerateParams,
    tok: &Tokenizer,
    out: &mut Vec<u32>,
    on_event: &mut impl FnMut(usize, GenerateEvent) -> bool,
) -> Option<FinishReason> {
    let next = match st.budget.as_ref().and_then(|b| b.forced_token()) {
        Some(f) => f,
        None => crate::inference::generate::sample_row(row, params, &st.history, &mut st.rng),
    };
    if let Some(b) = st.budget.as_mut() {
        b.accept_piece(next, &tok.piece_bytes(next));
    }
    if let Some(r) = check_stop(next, |t| tok.is_eog(t), &st.acc, &params.stop_strings) {
        return Some(r);
    }
    let q = st.req;
    match emit(next, tok, &mut st.decoder, &mut st.acc, &mut |ev| {
        on_event(q, ev)
    }) {
        Emit::Stop => return Some(FinishReason::Cancelled),
        Emit::Ok => {}
    }
    st.generated.push(next);
    out.push(next);
    st.history.push(next);
    st.predicted_n += 1;
    if st.generated.len() >= params.max_tokens {
        return Some(FinishReason::MaxTokens);
    }
    st.next = next;
    None
}

/// Хвост завершённой беседы: невыданный остаток UTF-8, времена, Done.
fn close_stream(
    st: &mut Stream,
    reason: FinishReason,
    on_event: &mut impl FnMut(usize, GenerateEvent) -> bool,
) {
    let q = st.req;
    let tail = st.decoder.finish();
    if !tail.is_empty() {
        on_event(q, GenerateEvent::Delta(tail));
    }
    let prompt_n = st.history.len() - st.generated.len();
    on_event(
        q,
        GenerateEvent::Timings(Timings {
            prompt_n,
            prompt_ms: st.t_prompt,
            predicted_n: st.predicted_n,
            predicted_ms: st.t_start.elapsed().as_secs_f64() * 1e3,
        }),
    );
    on_event(q, GenerateEvent::Done(reason));
}
