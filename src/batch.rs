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

use crate::backend::{Backend, Tensor};
use crate::error::BackendError;
use crate::inference::generate::{
    Emit, FinishReason, GenerateEvent, GenerateParams, Timings, check_stop, emit,
};
use crate::inference::sampling::state_from_seed;
use crate::tokenizer::{IncrementalDecoder, Tokenizer};

/// Состояние беседы в слоте: собственная позиция и свои KV/рекуррентные
/// стейты. Пакетный драйвер о внутренностях не знает.
pub trait BatchSlot {
    fn pos(&self) -> usize;
    /// Память состояния слота (KV + рекуррентные стейты), байт.
    fn bytes(&self) -> u64;
}

/// Модель, умеющая пакетный шаг. Реализуется КАЖДЫМ движком (гибрид qwen35,
/// плотная qwen3, …), поэтому пакетный режим перестал быть свойством одного
/// архитектного варианта.
pub trait BatchModel {
    type Slot: BatchSlot;
    type Snap;

    fn dim(&self) -> usize;
    fn max_seq(&self) -> usize;
    fn vocab_size(&self) -> usize;
    fn new_slot(&self, backend: &dyn Backend) -> Result<Self::Slot, BackendError>;
    fn slot_snapshot(
        &self,
        backend: &dyn Backend,
        slot: &Self::Slot,
    ) -> Result<Self::Snap, BackendError>;
    fn slot_restore(&mut self, slot: &mut Self::Slot, snap: &Self::Snap);
    fn prefill_slot_chunk_ext(
        &mut self,
        backend: &dyn Backend,
        slot: &mut Self::Slot,
        ids: &[u32],
        want_logits: bool,
        want_hidden: bool,
    ) -> Result<(Option<Tensor>, Option<Tensor>), BackendError>;
    #[allow(clippy::too_many_arguments)]
    fn decode_batch_hidden(
        &mut self,
        backend: &dyn Backend,
        slots: &mut [Self::Slot],
        active: &[usize],
        ids: &[u32],
    ) -> Result<(Tensor, Tensor), BackendError>;

    /// Логиты только последней позиции промпта в слот (короткий ход).
    fn prefill_slot_last(
        &mut self,
        backend: &dyn Backend,
        slot: &mut Self::Slot,
        ids: &[u32],
    ) -> Result<Tensor, BackendError>;

    /// Пакетный шаг с логитами по каждой активной беседе ([b, V]).
    fn decode_batch_active(
        &mut self,
        backend: &dyn Backend,
        slots: &mut [Self::Slot],
        active: &[usize],
        ids: &[u32],
    ) -> Result<Tensor, BackendError>;

    /// Один AR-шаг одной беседы на ЕЁ состоянии (диагностика/сверка окон).
    fn decode_step(
        &mut self,
        backend: &dyn Backend,
        slot: &mut Self::Slot,
        id: u32,
    ) -> Result<Tensor, BackendError>;

    /// Спекулятивная верификация в слоте. Гибрид умеет (окно на GDN-стейтах);
    /// прочие движки честно отказывают — вызывающий идёт AR-путём.
    #[allow(clippy::too_many_arguments)]
    fn decode_spec_in_slot(
        &mut self,
        backend: &dyn Backend,
        slot: &mut Self::Slot,
        anchor: u32,
        draft: &[u32],
        draft_probs: &[f32],
        greedy: bool,
        hist: &[u32],
        sp: &crate::inference::sampling::SamplingParams,
        rng: &mut Option<u64>,
        layers: &[usize],
    ) -> Result<crate::qwen3::SpecOutcome, BackendError> {
        let _ = (
            backend,
            slot,
            anchor,
            draft,
            draft_probs,
            greedy,
            hist,
            sp,
            rng,
            layers,
        );
        Err(BackendError::UnsupportedFormat(
            "спекуляция в пакетном слоте этим движком не поддержана".into(),
        ))
    }
}

/// Слоты пакета вместе с кэшем префикса (переезжают между пакетами).
pub type BatchCache<M> = (
    Vec<<M as BatchModel>::Slot>,
    Vec<Vec<u32>>,
    Vec<Option<(Vec<u32>, <M as BatchModel>::Snap)>>,
);

/// Учёт беседы в слоте (позиция живёт в самом слоте).
#[derive(Debug, Clone, Copy)]
pub struct SlotMeta {
    /// Длина поданного промпта, токенов.
    pub prompt_len: usize,
    /// Сколько токенов уже сгенерировано пакетными шагами.
    pub tokens_out: usize,
}

/// Пакетный движок: модель + фиксированный набор слотов.
pub struct BatchEngine<'m, M: BatchModel> {
    model: &'m mut M,
    slots: Vec<M::Slot>,
    meta: Vec<Option<SlotMeta>>,
    /// КЭШ ПРЕФИКСА: история, лежащая в KV слота после `finish_keep`. Следующий
    /// ход той же беседы префиллит ТОЛЬКО хвост, если новый промпт начинается
    /// с этой истории (чат-шаблон перерисовывает всю переписку — префикс
    /// совпадает).
    cached: Vec<Vec<u32>>,
    /// Снимок состояния на КОНЦЕ ПРЕФИЛЛА: промпт + стейты. По нему идёт
    /// подхват префикса (в отличие от `cached`, не требует, чтобы клиент
    /// вернул сгенерированные токены — достаточно того же промпта).
    prefixed: Vec<Option<(Vec<u32>, M::Snap)>>,
}

impl<'m, M: BatchModel> BatchEngine<'m, M> {
    /// Создать движок на `capacity` мест. Состояние каждого слота (KV + GDN)
    /// выделяется сразу: для 4B это ~50 МБ, для 27B — сотни МБ на слот.
    pub fn new(
        model: &'m mut M,
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
            cached: vec![Vec::new(); capacity],
            prefixed: (0..capacity).map(|_| None).collect(),
        })
    }

    /// Движок на ГОТОВЫХ слотах: сервис держит их между пакетами — состояние
    /// последовательностей выделяется один раз, а не на каждый пакет.
    pub fn with_slots(model: &'m mut M, slots: Vec<M::Slot>) -> Self {
        let meta = vec![None; slots.len()];
        let cached = vec![Vec::new(); slots.len()];
        let prefixed = (0..slots.len()).map(|_| None).collect();
        Self {
            model,
            slots,
            meta,
            cached,
            prefixed,
        }
    }

    /// Забрать слоты обратно (состояние бесед уже закрыто/освобождено).
    pub fn into_slots(self) -> Vec<M::Slot> {
        self.slots
    }

    /// Забрать слоты ВМЕСТЕ С КЭШЕМ префикса: движок создаёт `BatchEngine` на
    /// каждый пакет, поэтому `cached`/`prefixed` обязаны переезжать вместе со
    /// слотами — иначе кэш теряется между вызовами.
    pub fn into_parts(
        self,
    ) -> (
        Vec<M::Slot>,
        Vec<Vec<u32>>,
        Vec<Option<(Vec<u32>, M::Snap)>>,
    ) {
        (self.slots, self.cached, self.prefixed)
    }

    /// Движок на готовых слотах + сохранённом кэше префикса (см.
    /// [`Self::into_parts`]).
    pub fn with_cache(
        model: &'m mut M,
        slots: Vec<M::Slot>,
        cached: Vec<Vec<u32>>,
        prefixed: Vec<Option<(Vec<u32>, M::Snap)>>,
    ) -> Self {
        let n = slots.len();
        let meta = vec![None; n];
        Self {
            model,
            slots,
            meta,
            cached,
            prefixed,
        }
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

    /// Память слотов (KV + рекуррентные стейты), включая сохранённые
    /// под кэш префикса: статус обязан показывать её, иначе «модель занимает
    /// N ГБ» расходится с реальностью.
    pub fn slots_bytes(&self) -> u64 {
        self.slots.iter().map(|s| s.bytes()).sum()
    }

    /// Размерность скрытого (для примирения драфтера).
    pub fn dim(&self) -> usize {
        self.model.dim()
    }

    /// Предел контекста модели (окно, из метаданных GGUF).
    pub fn max_seq(&self) -> usize {
        self.model.max_seq()
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

    /// Начать приём беседы БЕЗ префилла: слот резервируется под промпт, сам
    /// префилл идёт кусками через [`Self::prefill_chunk`] — так длинный
    /// промпт не замораживает остальные беседы пакета.
    pub fn begin_prefill(
        &mut self,
        backend: &dyn Backend,
        ids: &[u32],
    ) -> Result<usize, BackendError> {
        if ids.is_empty() {
            return Err(BackendError::Shape("begin_prefill: пустой промпт".into()));
        }
        let slot = self
            .meta
            .iter()
            .position(|m| m.is_none())
            .ok_or_else(|| BackendError::Shape("begin_prefill: все слоты заняты".into()))?;
        // НОВАЯ последовательность: слот мог остаться с кэшем префикса от
        // прошлой беседы — стираем состояние, иначе префилл дописался бы
        // поверх чужого KV (позиция не с нуля). Подхват кэша — только через
        // resume_begin, который сюда не попадает.
        self.slots[slot] = self.model.new_slot(backend)?;
        self.cached[slot].clear();
        self.prefixed[slot] = None;
        self.meta[slot] = Some(SlotMeta {
            prompt_len: ids.len(),
            tokens_out: 0,
        });
        Ok(slot)
    }

    /// Кусок префилла в слот. `None` — кусок промежуточный (head не считался);
    /// `Some(логиты [V])` — промпт подан целиком, это логиты последней
    /// позиции (из них сэмплится первый токен).
    pub fn prefill_chunk(
        &mut self,
        backend: &dyn Backend,
        slot: usize,
        ids: &[u32],
        last: bool,
    ) -> Result<Option<Vec<f32>>, BackendError> {
        Ok(self
            .prefill_chunk_hidden(backend, slot, ids, last, false)?
            .0)
    }

    /// Кусок префилла + скрытое по всем строкам куска (для примирения MTP:
    /// драфтеру нужны строки h всего промпта).
    pub fn prefill_chunk_hidden(
        &mut self,
        backend: &dyn Backend,
        slot: usize,
        ids: &[u32],
        last: bool,
        want_hidden: bool,
    ) -> Result<(Option<Vec<f32>>, Option<crate::backend::Tensor>), BackendError> {
        if slot >= self.slots.len() || self.meta[slot].is_none() {
            return Err(BackendError::Shape(format!(
                "prefill_chunk: слот {slot} вне или свободен"
            )));
        }
        // Копий состояния нет: поля модели и слота заимствуются раздельно.
        let m = &mut *self.model;
        let s = &mut self.slots[slot];
        let (lg, h) = m.prefill_slot_chunk_ext(backend, s, ids, last, want_hidden)?;
        Ok((
            match lg {
                Some(lg) => Some(backend.to_cpu_f32(&lg)?),
                None => None,
            },
            h,
        ))
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

    /// Пакетный шаг + СКРЫТОЕ последней позиции каждой беседы (строка [d] на
    /// слот) — вход MTP-черновиков на слот. Скрытое всё равно считается для
    /// логитов, лишней работы нет.
    pub fn step_hidden(
        &mut self,
        backend: &dyn Backend,
        next: &[(usize, u32)],
    ) -> Result<(Vec<(usize, Vec<f32>, Vec<f32>)>, Tensor), BackendError> {
        if next.is_empty() {
            return Err(BackendError::Shape("step_hidden: пустой пакет".into()));
        }
        let mut active = Vec::with_capacity(next.len());
        let mut ids = Vec::with_capacity(next.len());
        for &(slot, id) in next {
            if slot >= self.slots.len() || self.meta[slot].is_none() {
                return Err(BackendError::Shape(format!(
                    "step_hidden: слот {slot} вне или свободен"
                )));
            }
            active.push(slot);
            ids.push(id);
        }
        let (logits, hidden) =
            self.model
                .decode_batch_hidden(backend, &mut self.slots, &active, &ids)?;
        let host = backend.to_cpu_f32(&logits)?;
        let hhost = backend.to_cpu_f32(&hidden)?;
        let v = self.model.vocab_size();
        let d = self.model.dim();
        let mut out = Vec::with_capacity(active.len());
        for (row, &slot) in active.iter().enumerate() {
            out.push((
                slot,
                host[row * v..(row + 1) * v].to_vec(),
                hhost[row * d..(row + 1) * d].to_vec(),
            ));
            if let Some(m) = self.meta[slot].as_mut() {
                m.tokens_out += 1;
            }
        }
        Ok((out, hidden))
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
        let v = self.model.vocab_size();
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
        rng: &mut Option<u64>,
        layers: &[usize],
    ) -> Result<crate::qwen3::SpecOutcome, BackendError> {
        if slot >= self.slots.len() {
            return Err(BackendError::Shape(format!(
                "decode_spec_slot: нет слота {slot} (мест {})",
                self.slots.len()
            )));
        }
        let m = &mut *self.model;
        let s = &mut self.slots[slot];
        m.decode_spec_in_slot(
            backend,
            s,
            anchor,
            draft,
            draft_probs,
            greedy,
            hist,
            sp,
            rng,
            layers,
        )
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
        m.decode_step(backend, s, id)
    }

    /// Освободить слот, СОХРАНИВ KV и историю — кэш префикса для следующего
    /// хода той же беседы. Слот считается свободным, но если новый промпт
    /// начинается с сохранённой истории, префилл пойдёт только с хвоста.
    pub fn finish_keep(&mut self, slot: usize, history: &[u32]) -> Result<(), BackendError> {
        if slot >= self.slots.len() || self.meta[slot].is_none() {
            return Err(BackendError::Shape(format!(
                "finish_keep: слот {slot} вне или свободен"
            )));
        }
        self.meta[slot] = None;
        // В KV лежит ровно `pos` токенов: последний сэмплированный токен в
        // кэш ещё НЕ подан (он станет первым токеном следующего шага).
        let pos = self.slots[slot].pos();
        self.cached[slot] = history[..history.len().min(pos)].to_vec();
        Ok(())
    }

    /// Запечатать префикс: снять снимок состояния СЕЙЧАС (вызывается сразу
    /// после последнего куска префилла) и запомнить сами токены промпта.
    /// Дальше по этому снимку можно откатиться и подать только хвост.
    pub fn seal_prefix(
        &mut self,
        backend: &dyn Backend,
        slot: usize,
        prompt: &[u32],
    ) -> Result<(), BackendError> {
        if slot >= self.slots.len() || self.meta[slot].is_none() {
            return Err(BackendError::Shape(format!(
                "seal_prefix: слот {slot} вне или свободен"
            )));
        }
        let snap = self.model.slot_snapshot(backend, &self.slots[slot])?;
        self.prefixed[slot] = Some((prompt.to_vec(), snap));
        Ok(())
    }

    /// Сколько токенов запечатанного промпта совпадает с `ids` (диагностика;
    /// Some(0) — снимок есть, но промпты разошлись с первого токена).
    pub fn prefix_match_len(&self, slot: usize, ids: &[u32]) -> Option<usize> {
        let (p, _) = self.prefixed.get(slot)?.as_ref()?;
        Some(p.iter().zip(ids.iter()).take_while(|(a, b)| a == b).count())
    }

    /// Свободный слот, чей ЗАПЕЧАТАННЫЙ промпт — префикс `ids` (длина ≥ min).
    pub fn best_prefix_slot(&self, ids: &[u32], min: usize) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize)> = None;
        for slot in 0..self.slots.len() {
            if self.meta[slot].is_some() {
                continue;
            }
            let Some((p, _)) = self.prefixed[slot].as_ref() else {
                continue;
            };
            if p.len() < min.max(1) || p.len() > ids.len() {
                continue;
            }
            if p.iter().zip(ids.iter()).all(|(a, b)| a == b)
                && best.map(|(_, bk)| p.len() > bk).unwrap_or(true)
            {
                best = Some((slot, p.len()));
            }
        }
        best
    }

    /// Подхватить префикс по СНИМКУ: откатить слот к концу промпта и отметить
    /// занятым. Хвост `ids[k..]` подаёт вызывающий кусками префилла.
    pub fn resume_from_seal(&mut self, slot: usize, ids: &[u32]) -> Result<usize, BackendError> {
        if slot >= self.slots.len() || self.meta[slot].is_some() {
            return Err(BackendError::Shape(format!(
                "resume_from_seal: слот {slot} вне или занят"
            )));
        }
        let Some((p, snap)) = self.prefixed[slot].take() else {
            return Err(BackendError::Shape(format!(
                "resume_from_seal: слот {slot} без снимка префикса"
            )));
        };
        if p.len() > ids.len() || p.iter().zip(ids.iter()).any(|(a, b)| a != b) {
            self.prefixed[slot] = Some((p, snap));
            return Err(BackendError::Shape(
                "resume_from_seal: промпт не является префиксом".into(),
            ));
        }
        let k = p.len();
        self.model.slot_restore(&mut self.slots[slot], &snap);
        self.meta[slot] = Some(SlotMeta {
            prompt_len: ids.len(),
            tokens_out: 0,
        });
        Ok(k)
    }

    /// Сколько токенов из `ids` уже лежит в KV слота (общий префикс с кэшем).
    pub fn cached_prefix(&self, slot: usize, ids: &[u32]) -> usize {
        let Some(c) = self.cached.get(slot) else {
            return 0;
        };
        let n = c.len().min(ids.len());
        c.iter()
            .zip(ids.iter())
            .take_while(|(a, b)| a == b)
            .count()
            .min(n)
    }

    /// Свободный слот с самым длинным общим префиксом (0 — кэша нет).
    pub fn best_cached_slot(&self, ids: &[u32]) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize)> = None;
        for slot in 0..self.slots.len() {
            if self.meta[slot].is_some() {
                continue;
            }
            let k = self.cached_prefix(slot, ids);
            if k > 0 && best.map(|(_, bk)| k > bk).unwrap_or(true) {
                best = Some((slot, k));
            }
        }
        best
    }

    /// Подхватить кэш префикса без префилла: слот занимается, возвращается
    /// длина уже лежащего префикса. Хвост `ids[common..]` вызывающий подаёт
    /// кусками через [`Self::prefill_chunk`] — как обычный префилл.
    pub fn resume_begin(&mut self, slot: usize, ids: &[u32]) -> Result<usize, BackendError> {
        if slot >= self.slots.len() || self.meta[slot].is_some() {
            return Err(BackendError::Shape(format!(
                "resume_begin: слот {slot} вне или занят"
            )));
        }
        let common = self.cached_prefix(slot, ids);
        if common == 0 || common != self.slots[slot].pos() {
            return Err(BackendError::Shape(format!(
                "resume_begin: кэш {common} токенов, в KV {} — нельзя подхватить",
                self.slots[slot].pos()
            )));
        }
        self.meta[slot] = Some(SlotMeta {
            prompt_len: ids.len(),
            tokens_out: 0,
        });
        Ok(common)
    }

    /// Продолжить беседу в слоте с кэшем: подать только `ids[common..]`.
    /// Возвращает (сколько токенов уже было, логиты последней позиции).
    pub fn resume_prefix(
        &mut self,
        backend: &dyn Backend,
        slot: usize,
        ids: &[u32],
    ) -> Result<(usize, Option<Vec<f32>>), BackendError> {
        if slot >= self.slots.len() || self.meta[slot].is_some() {
            return Err(BackendError::Shape(format!(
                "resume_prefix: слот {slot} вне или занят"
            )));
        }
        let common = self.cached_prefix(slot, ids);
        if common == 0 {
            return Err(BackendError::Shape(
                "resume_prefix: общего префикса нет — нужен полный префилл".into(),
            ));
        }
        let pos = self.slots[slot].pos();
        if common != pos {
            return Err(BackendError::Shape(format!(
                "resume_prefix: кэш {common} токенов, в KV {pos} — рассинхрон"
            )));
        }
        self.meta[slot] = Some(SlotMeta {
            prompt_len: ids.len(),
            tokens_out: 0,
        });
        let tail = &ids[common..];
        if tail.is_empty() {
            return Ok((common, None));
        }
        let lg = self.prefill_chunk(backend, slot, tail, true)?;
        Ok((common, lg))
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
        self.cached[slot].clear();
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
    /// Где снять снимок префикса (конец ПЕРЕПИСКИ, без генерационного
    /// префикса шаблона). Следующий ход с тем же началом подхватит его и
    /// префиллит только хвост. None — не сохранять.
    pub prefix_seal_at: Option<usize>,
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
    /// Остаток промпта, который ещё не подан (префилл кусками), и сколько
    /// уже подано. None — префилл завершён.
    prefill: Option<Vec<u32>>,
    prefill_done: usize,
    /// Скрытое по строкам промпта (хост, [rows, d]) — для примирения MTP.
    prime_h: Vec<f32>,
    /// Абсолютная позиция снимка префикса (в токенах промпта) и снят ли он.
    seal_at: Option<usize>,
    sealed: bool,
    /// Скрытое последней позиции (out_norm) — вход MTP-черновиков на слот.
    /// Пусто, пока шаг не вернул его (после окна — устарело).
    hidden: Vec<f32>,
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
pub fn generate_batch<M: BatchModel>(
    engine: &mut BatchEngine<'_, M>,
    backend: &dyn Backend,
    tok: &Tokenizer,
    reqs: &[BatchRequest],
    on_event: &mut impl FnMut(usize, GenerateEvent) -> bool,
) -> Result<Vec<Vec<u32>>, BackendError> {
    generate_batch_with(engine, backend, tok, reqs, None, on_event)
}

/// Источник черновиков на беседу: (слот, история, скрытое последней позиции)
/// → (черновики, вероятности). Возвращает None — беседа идёт AR-шагом.
pub type BatchDraftFn<'a> = dyn FnMut(usize, &[u32], &[f32]) -> Option<BatchDraft> + 'a;

/// Черновики беседы: токены, вероятности и СЛОИ, скрытое которых нужно
/// захватить при верификации (для примирения драфтера после окна).
pub struct BatchDraft {
    pub tokens: Vec<u32>,
    pub probs: Vec<f32>,
    pub layers: Vec<usize>,
}

/// После окна верификации: (слот, скрытое строк окна [rows, d], позиция
/// ПОСЛЕ окна, принято токенов). Драфтер по этому вызову пересобирает KV —
/// без него он рассинхронизируется после первого же окна.
pub type BatchAfterWindowFn<'a> = dyn FnMut(usize, &crate::backend::Tensor, usize, usize) + 'a;

/// Примирение внешнего драфтера с промптом беседы: (слот, скрытое по ВСЕМ
/// строкам промпта [T, d], длина промпта). По этому вызову драфтер выстраивает
/// свой KV на промпт (контракт MTP: pos = n = длина промпта).
pub type BatchPrimeFn<'a> = dyn FnMut(usize, &crate::backend::Tensor, usize) -> bool + 'a;

/// Как [`generate_batch`], но черновики даёт ВНЕШНИЙ драфтер на слот (MTP,
/// обученный nextn-модуль таргета). Без него черновики берутся из
/// prompt-lookup, как в [`generate_batch`].
pub fn generate_batch_with<M: BatchModel>(
    engine: &mut BatchEngine<'_, M>,
    backend: &dyn Backend,
    tok: &Tokenizer,
    reqs: &[BatchRequest],
    draft_fn: Option<&mut BatchDraftFn<'_>>,
    on_event: &mut impl FnMut(usize, GenerateEvent) -> bool,
) -> Result<Vec<Vec<u32>>, BackendError> {
    generate_batch_full(engine, backend, tok, reqs, draft_fn, None, None, on_event)
}

/// То же, но с ПРИМИРЕНИЕМ драфтера: `prime_fn` получает скрытое по всем
/// строкам промпта сразу после префилла (MTP без этого «слеп»).
pub fn generate_batch_full<M: BatchModel>(
    engine: &mut BatchEngine<'_, M>,
    backend: &dyn Backend,
    tok: &Tokenizer,
    reqs: &[BatchRequest],
    mut draft_fn: Option<&mut BatchDraftFn<'_>>,
    mut prime_fn: Option<&mut BatchPrimeFn<'_>>,
    mut after_fn: Option<&mut BatchAfterWindowFn<'_>>,
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

    // Кусок префилла за такт: длинный промпт не замораживает остальные
    // беседы — между кусками они получают свои токены.
    // Кэш префикса имеет смысл от заметного куска: мельче — только трата
    // проверки. INFER_PREFIX_CACHE_MIN=0 выключает подхват (A/B).
    let prefix_cache_min: usize = std::env::var("INFER_PREFIX_CACHE_MIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    // ПРЕДЕЛ РАЗМЕРА КЭША: слот с сохранённым KV не освобождает память, пока
    // его не займёт другая беседа. Длинная беседа оставила бы KV на десятки
    // тысяч токенов (и снимок стейтов) навсегда. Сверх лимита слот
    // ОСВОБОЖДАЕТСЯ полностью — префикс пересчитается при следующем ходе.
    let prefix_cache_max: usize = std::env::var("INFER_PREFIX_CACHE_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096);
    // Закрытие беседы: кэш префикса — только если история в пределах лимита
    // (иначе слот освобождается целиком, иначе KV/снимок висели бы вечно).
    let keep_or_finish =
        |eng: &mut BatchEngine<'_, M>, slot: usize, hist: &[u32]| -> Result<(), BackendError> {
            if hist.len() <= prefix_cache_max {
                eng.finish_keep(slot, hist)
            } else {
                eng.finish(slot, backend)
            }
        };
    let pchunk: usize = std::env::var("INFER_BATCH_PREFILL_CHUNK")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|v: &usize| *v > 0)
        .unwrap_or(256);

    loop {
        // Принять новые беседы в свободные СЛОТЫ (префилл — ниже, кусками).
        while let Some(&q) = pending.front() {
            if !streams.iter().any(|s| s.is_none()) {
                break;
            }
            pending.pop_front();
            // КЭШ ПРЕФИКСА: если чья-то история (в свободном слоте) — префикс
            // нового промпта, подаём только хвост. Порог отсекает мелочь.
            let ids = &reqs[q].ids;
            let reuse = engine.best_prefix_slot(ids, prefix_cache_min).or_else(|| {
                engine
                    .best_cached_slot(ids)
                    .filter(|(_, k)| *k >= prefix_cache_min && prefix_cache_min > 0)
            });
            if reuse.is_none() && std::env::var("INFER_BATCH_LOG").is_ok() {
                for slot in 0..engine.capacity() {
                    if let Some(k) = engine.prefix_match_len(slot, ids) {
                        eprintln!(
                            "prefix-cache: слот {slot} совпал на {k} из {} (порог {prefix_cache_min})",
                            ids.len()
                        );
                    }
                }
            }
            let log = std::env::var("INFER_BATCH_LOG").is_ok();
            // Хвост должен быть НЕПУСТЫМ: при полностью совпавшем промпте
            // (клиент повторил тот же запрос) префиллить нечего, а пустой
            // кусок префилла — ошибка. Такой случай идёт полным префиллом.
            let reuse = reuse.filter(|(_, k)| *k < ids.len());
            let picked = match reuse {
                Some((slot, k)) => match engine.resume_from_seal(slot, ids) {
                    Ok(_) => Some((slot, k, "снимок")),
                    Err(e1) => match engine.resume_begin(slot, ids) {
                        Ok(_) => Some((slot, k, "точный")),
                        Err(e2) => {
                            if log {
                                eprintln!(
                                    "prefix-cache: слот {slot} k={k} не подхвачен: снимок: {e1}; точный: {e2}"
                                );
                            }
                            None
                        }
                    },
                },
                None => None,
            };
            let (slot, tail_from) = match picked {
                Some((slot, k, how)) => {
                    if log {
                        eprintln!(
                            "prefix-cache: слот {slot} отдал {k} из {} по «{how}»",
                            ids.len()
                        );
                    }
                    (slot, k)
                }
                None => (engine.begin_prefill(backend, ids)?, 0),
            };
            streams[slot] = Some(Stream {
                req: q,
                decoder: IncrementalDecoder::new(),
                acc: String::new(),
                history: reqs[q].ids.clone(),
                rng: reqs[q].params.seed.map(state_from_seed),
                generated: Vec::new(),
                budget: reqs[q].params.reasoning_budget.clone(),
                next: 0,
                prefill: Some(reqs[q].ids[tail_from..].to_vec()),
                prefill_done: 0,
                seal_at: reqs[q]
                    .prefix_seal_at
                    .filter(|sa| *sa > 0 && *sa <= reqs[q].ids.len()),
                sealed: false,
                prime_h: Vec::new(),
                hidden: Vec::new(),
                pending: None,
                t_prompt: 0.0,
                t_start: Instant::now(),
                predicted_n: 0,
            });
        }
        // Один кусок префилла на беседу; последний даёт первый токен.
        let mut prefilling = false;
        for slot in 0..streams.len() {
            let Some(st) = streams[slot].as_mut() else {
                continue;
            };
            let Some(p) = st.prefill.as_ref() else {
                continue;
            };
            prefilling = true;
            let base = reqs[st.req].ids.len() - p.len();
            let a = st.prefill_done;
            let mut b = (a + pchunk).min(p.len());
            // Не перескакиваем точку снимка: иначе снимок пришлось бы снимать
            // не на границе переписки (а GDN-стейт назад не откатить).
            let mut seal_here = false;
            if let (Some(sa), false) = (st.seal_at, st.sealed)
                && base + b > sa
                && base + a < sa
            {
                b = sa - base;
                seal_here = true;
            }
            let last = b == p.len();
            let piece: Vec<u32> = p[a..b].to_vec();
            st.prefill_done = b;
            let q = st.req;
            let t0 = Instant::now();
            let (lg, hchunk) =
                engine.prefill_chunk_hidden(backend, slot, &piece, last, prime_fn.is_some())?;
            if let Some(h) = hchunk {
                let host = backend.to_cpu_f32(&h)?;
                st.prime_h.extend_from_slice(&host);
            }
            st.t_prompt += t0.elapsed().as_secs_f64() * 1e3;
            if seal_here || (st.seal_at == Some(base + b) && !st.sealed) {
                let upto = st.seal_at.expect("seal_here");
                let cut = &reqs[q].ids[..upto];
                if let Err(e) = engine.seal_prefix(backend, slot, cut)
                    && std::env::var("INFER_BATCH_LOG").is_ok()
                {
                    eprintln!("prefix-cache: снимок не снят ({e})");
                }
                st.sealed = true;
            }
            if last {
                st.prefill = None;
                st.t_start = Instant::now();
                if let Some(pf) = prime_fn.as_deref_mut() {
                    let d = engine.dim();
                    let rows = st.prime_h.len() / d.max(1);
                    if rows > 0 {
                        let ht = backend.from_cpu_f32(&st.prime_h, vec![rows, d])?;
                        if !pf(slot, &ht, rows) && std::env::var("INFER_BATCH_LOG").is_ok() {
                            eprintln!("mtp: примирение слота {slot} не удалось");
                        }
                        // Скрытое последней строки промпта — вход ПЕРВОГО
                        // черновика: драфтер ждёт вызов ровно при
                        // history = committed + 1 (сразу после префилла).
                        let n = st.prime_h.len();
                        st.hidden = st.prime_h[n - d..].to_vec();
                    }
                    st.prime_h = Vec::new();
                }
            }
            if let Some(row) = lg {
                let done = advance(st, &row, &reqs[q].params, tok, &mut out[q], on_event);
                match done {
                    None => {
                        let nx = streams[slot].as_ref().unwrap().next;
                        queue.push((slot, nx));
                    }
                    Some(reason) => {
                        let hist = streams[slot].as_ref().unwrap().history.clone();
                        close_stream(streams[slot].as_mut().unwrap(), reason, on_event);
                        keep_or_finish(engine, slot, &hist)?;
                        streams[slot] = None;
                    }
                }
            }
        }
        if queue.is_empty() {
            if !prefilling && pending.is_empty() {
                break;
            }
            continue;
        }
        // Контекст беседы кончился — закрываем ЕЁ (как llama.cpp на переполнении
        // окна), а не роняем весь пакет: остальные беседы продолжают идти.
        let max_seq = engine.max_seq();
        let mut ready: Vec<(usize, u32)> = Vec::with_capacity(queue.len());
        for &(slot, tok_) in queue.iter() {
            let full = engine.pos(slot).map(|p| p + 1 > max_seq).unwrap_or(true);
            if full {
                let hist = streams[slot]
                    .as_ref()
                    .map(|st| st.history.clone())
                    .unwrap_or_default();
                if let Some(st) = streams[slot].as_mut() {
                    close_stream(st, FinishReason::MaxTokens, on_event);
                }
                keep_or_finish(engine, slot, &hist)?;
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
        // Спец ВКЛЮЧЁН по умолчанию (лосслессно: жадные — биты как AR,
        // сэмплирующие — тот же каскад, что одиночный спек). Отключение:
        // INFER_BATCH_SPEC=0.
        let spec_on = std::env::var("INFER_BATCH_SPEC")
            .map(|v| v != "0")
            .unwrap_or(true);
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
                // Бюджет рассуждений вставляет свои токены — окно его обошло
                // бы, поэтому такие беседы идут AR-шагом. Сэмплирующие беседы
                // идут окном через лосслесс-каскад (q — дельта, как в MTP).
                if forced {
                    still_ar.push((slot, tok_));
                    continue;
                }
                // Черновики: внешний драфтер (MTP на слот) — только если
                // скрытое СВЕЖЕЕ (шаг, а не окно); иначе prompt-lookup.
                let (drafts, probs, layers) = match draft_fn.as_deref_mut() {
                    Some(f) if !st.hidden.is_empty() => match f(slot, &st.history, &st.hidden) {
                        Some(d) => (d.tokens, d.probs, d.layers),
                        None => (Vec::new(), Vec::new(), Vec::new()),
                    },
                    _ => {
                        let (t, p) = crate::spec::ngram_lookup_min(&st.history, gamma, 4, 4);
                        (t, p, Vec::new())
                    }
                };
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
                let o = match engine.decode_spec_slot(
                    backend,
                    slot,
                    tok_,
                    &drafts,
                    &probs,
                    greedy,
                    &hist,
                    &sp,
                    &mut st.rng,
                    &layers,
                ) {
                    Ok(o) => o,
                    // Окно не удалось — беседа честно идёт AR-шагом.
                    Err(_) => {
                        still_ar.push((slot, tok_));
                        continue;
                    }
                };
                if !layers.is_empty()
                    && let Some(af) = after_fn.as_deref_mut()
                    && let Some(ext) = o.extract.first()
                {
                    let n_ok_now = o.n_draft_accepted + 1;
                    let pos_now = engine.pos(slot).unwrap_or(0);
                    af(slot, ext, pos_now, n_ok_now);
                }
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
                    if let Some(r) =
                        check_stop(t, |x| tok.is_eog(x), &st.acc, &reqs[q].params.stop_strings)
                    {
                        closed = Some(r);
                        break;
                    }
                    match emit(t, tok, &mut st.decoder, &mut st.acc, &mut |ev| {
                        on_event(q, ev)
                    }) {
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
                        let hist = st.history.clone();
                        close_stream(st, r, on_event);
                        keep_or_finish(engine, slot, &hist)?;
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
                                let hist = st.history.clone();
                                close_stream(st, r, on_event);
                                keep_or_finish(engine, slot, &hist)?;
                                streams[slot] = None;
                                continue;
                            }
                        }
                        st.pending = row;
                        match st.pending.take() {
                            // Готовые логиты: сэмплим из них, forward не нужен.
                            Some(row) => {
                                match advance(st, &row, &reqs[q].params, tok, &mut out[q], on_event)
                                {
                                    None => still_ar.push((slot, st.next)),
                                    Some(r) => {
                                        let hist = st.history.clone();
                                        close_stream(st, r, on_event);
                                        keep_or_finish(engine, slot, &hist)?;
                                        streams[slot] = None;
                                    }
                                }
                            }
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
        let (rows, h_all) = engine.step_hidden(backend, &queue)?;
        let b_rows = rows.len();
        let mut queue_next: Vec<(usize, u32)> = Vec::with_capacity(b_rows);
        for (r, (slot, row, hid)) in rows.into_iter().enumerate() {
            // Каждый поданный токен — в драфтер (контракт MTP: history =
            // committed + 1; иначе драфтер отстаёт на токен и молчит).
            if let Some(af) = after_fn.as_deref_mut() {
                let d = engine.dim();
                let hv = backend.row_view(&h_all, b_rows, d, r)?;
                let pos = engine.pos(slot).unwrap_or(0);
                af(slot, &hv, pos, 1);
            }
            let st = streams[slot].as_mut().expect("шаг по занятому слоту");
            // Скрытое последней позиции — только что посчитано этим шагом:
            // годный вход MTP-черновиков для СЛЕДУЮЩЕГО окна.
            st.hidden = hid;
            let q = st.req;
            let done = advance(st, &row, &reqs[q].params, tok, &mut out[q], on_event);
            match done {
                None => queue_next.push((slot, streams[slot].as_ref().unwrap().next)),
                Some(reason) => {
                    let hist = st.history.clone();
                    close_stream(st, reason, on_event);
                    // Кэш префикса: KV и история остаются для следующего хода.
                    keep_or_finish(engine, slot, &hist)?;
                    streams[slot] = None;
                }
            }
        }
        queue = queue_next;
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
