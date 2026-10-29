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

use crate::backend::Backend;
use crate::error::BackendError;
use crate::qwen35::{Qwen35Model, Qwen35Slot};

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
