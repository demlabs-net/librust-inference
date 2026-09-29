//! Ф2.1: спекулятивный декодинг — драфтеры.
//!
//! [`Drafter`] — интерфейс генерации draft-токенов с честными вероятностями
//! q_i (rejection-sampling в [`crate::qwen3::Qwen3Model::decode_spec`]
//! сохраняет целевое распределение при ЛЮБОЙ q; acceptance падает при
//! неточных q). [`NGramDrafter`] — суффиксный lookup по истории (n-граммы),
//! q=0.5 консервативно.

/// Версия драфтера (Ф2.5 профили: v1 — AR-блок 16, 5 слоёв; v2 — блок 8,
/// 7 draft-токенов + selector/conv).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrafterVersion {
    V1,
    V2,
}

/// Манифест драфтера (Ф2.2): версия, таргет (версионная сверка!), топология.
#[derive(Debug, Clone)]
pub struct DrafterManifest {
    pub version: DrafterVersion,
    /// Идентификатор таргета (например "qwen3-8b") — драфтеры
    /// версио-чувствительны: несовпадение → отказ от загрузки.
    pub target: String,
    /// Число слоёв драфтера (v1: 5, Qwen3-Coder: 8).
    pub n_layers: usize,
    /// Размер блока AR (v1: 16).
    pub block: usize,
    /// Top-K кандидатов селектора.
    pub top_k: usize,
    /// Число draft-токенов на шаг (v2: 7).
    pub n_draft: usize,
}

impl DrafterManifest {
    pub fn v1(target: &str, block: usize, top_k: usize) -> Self {
        Self {
            version: DrafterVersion::V1,
            target: target.into(),
            n_layers: 5,
            block,
            top_k,
            n_draft: block,
        }
    }

    pub fn v2(target: &str) -> Self {
        Self {
            version: DrafterVersion::V2,
            target: target.into(),
            n_layers: 8,
            block: 8,
            top_k: 10,
            n_draft: 7,
        }
    }
}

/// Транспорт черновиков (7.9): координатор просит СОСЕДНЮЮ ноду предложить
/// `gamma` токенов по истории. Контракт намеренно узкий (токены, не фичи):
/// верификация черновиков всегда ЛОКАЛЬНАЯ (rejection-sampling целевой
/// моделью), поэтому неверный/пустой ответ соседа не портит распределение —
/// он лишь снижает приёмку до нуля.
pub trait DraftTransport: Send + Sync {
    fn remote_draft(&self, history: &[u32], gamma: usize) -> Result<(Vec<u32>, Vec<f32>), String>;

    /// Скрытые таргета для удалённого графического драфтера (DFlash): без них
    /// он «слеп» (приёмка ~0). `features` — по слоям `target_layers()`,
    /// каждая [t, dim]; `pos`/`n_accepted` — как в `Drafter::after_extract`.
    /// Дефолт — транспорт фичи не умеет (N-грамный драфтер их не требует).
    fn remote_features(
        &self,
        _features: &[(Vec<usize>, Vec<f32>)],
        _pos: usize,
        _n_accepted: usize,
    ) -> Result<(), String> {
        Err("транспорт черновиков не умеет передавать фичи таргета".into())
    }
}

/// Драфтер на соседней ноде (7.9, «спекуляция через сеть»): сам ничего не
/// считает — пересылает историю и возвращает чужие черновики. У соседа
/// собственный KV/драфтер; координатору не нужно держать драфтер в памяти.
pub struct RemoteDrafter {
    pub transport: std::sync::Arc<dyn DraftTransport>,
    /// Последняя ошибка транспорта (диагностика/метрики): пустой ответ
    /// означает нулевую приёмку, и без этого поля причина терялась бы.
    pub last_error: Option<String>,
}

impl RemoteDrafter {
    pub fn new(transport: std::sync::Arc<dyn DraftTransport>) -> Self {
        Self {
            transport,
            last_error: None,
        }
    }
}

impl Drafter for RemoteDrafter {
    fn after_extract(
        &mut self,
        backend: &dyn crate::backend::Backend,
        extract: &[crate::backend::Tensor],
        pos: usize,
        n_accepted: usize,
    ) -> Result<(), crate::error::BackendError> {
        if extract.is_empty() {
            return Ok(());
        }
        let mut feats = Vec::with_capacity(extract.len());
        for t in extract {
            let shape = t.shape().to_vec();
            let data = backend.to_cpu_f32(t)?;
            feats.push((shape, data));
        }
        match self.transport.remote_features(&feats, pos, n_accepted) {
            Ok(()) => {
                self.last_error = None;
                Ok(())
            }
            Err(e) => {
                // Сосед не принял фичи: черновики будут слепыми (приёмка ~0),
                // но генерация не ломается — фиксируем причину.
                self.last_error = Some(e);
                Ok(())
            }
        }
    }

    fn draft(
        &mut self,
        _backend: &dyn crate::backend::Backend,
        history: &[u32],
        gamma: usize,
    ) -> (Vec<u32>, Vec<f32>) {
        match self.transport.remote_draft(history, gamma) {
            Ok(v) => {
                self.last_error = None;
                v
            }
            Err(e) => {
                // Сосед недоступен/ошибся — черновиков нет, генерация идёт
                // обычным decode (верификация не нужна).
                self.last_error = Some(e);
                (Vec::new(), Vec::new())
            }
        }
    }
}

/// Генератор draft-токенов. `backend` нужен графическим драфтерам
/// (DFlash — диффузия на GPU); n-граммовый его игнорирует.
pub trait Drafter {
    /// Следующие `gamma` токенов и их вероятности q_i (для rejection).
    fn draft(
        &mut self,
        backend: &dyn crate::backend::Backend,
        history: &[u32],
        gamma: usize,
    ) -> (Vec<u32>, Vec<f32>);

    /// Слои таргета, скрытые которых нужны для инжекта (DFlash: 5 слоёв;
    /// N-грам — пусто).
    fn target_layers(&self) -> &[usize] {
        &[]
    }

    /// Скрытые таргета по extract-слоям (после prefill_extract/DECODE_spec);
    /// `n_accepted` — число ПРИНЯТЫХ строк окна верификации (якорь +
    /// проверенные черновики; z-lab: context = только принятые, rows [0..n)),
    /// позиция последнего принятого токена. DFlash-инжект: фичи таргета
    /// → fc+enc_norm → k/v кэша драфтера (иначе драфтер «слеп» — 2% приёмка).
    fn after_extract(
        &mut self,
        _backend: &dyn crate::backend::Backend,
        _extract: &[crate::backend::Tensor],
        _pos: usize,
        _n_accepted: usize,
    ) -> Result<(), crate::error::BackendError> {
        Ok(())
    }
}

/// Суффиксный n-граммовый драфтер: самый длинный суффикс истории с
/// предыдущим вхождением → продолжение (максимальный порядок — 4).
/// N (plan 1.5): q больше НЕ хардкод 0.5 — это эмпирическая частота
/// выбранного продолжения среди всех вхождений суффикса (честная оценка
/// для rejection sampling: при q=1 драфтер уверен, при редких суффиксах
/// q низкое → принятие по p/q ≥ 1 — целевое распределение сохраняется).
#[derive(Debug, Default)]
pub struct NGramDrafter {
    max_order: usize,
}

impl NGramDrafter {
    pub fn new(max_order: usize) -> Self {
        Self {
            max_order: max_order.max(1),
        }
    }
}

impl Drafter for NGramDrafter {
    fn draft(
        &mut self,
        _backend: &dyn crate::backend::Backend,
        history: &[u32],
        gamma: usize,
    ) -> (Vec<u32>, Vec<f32>) {
        ngram_lookup(history, gamma, self.max_order)
    }
}

/// N-грамный lookup БЕЗ бэкенда (7.9): та же логика, что у
/// [`NGramDrafter`], но вызываемая там, где графа нет вовсе — например на
/// ноде-драфтере, обслуживающей черновики соседей по сети.
pub fn ngram_lookup(
    history: &[u32],
    gamma: usize,
    max_order: usize,
) -> (Vec<u32>, Vec<f32>) {
    let max_order = max_order.max(1);
    {
        let mut drafts: Vec<u32> = Vec::new();
        let mut probs: Vec<f32> = Vec::new();
        let mut hist: Vec<u32> = history.to_vec();
        for _ in 0..gamma {
            let mut next: Option<(u32, f32)> = None;
            let mut order = hist.len().min(max_order);
            while order >= 1 && next.is_none() {
                if hist.len() > order {
                    let suffix = &hist[hist.len() - order..];
                    // N (plan 1.5): count ALL suffix occurrences and the
                    // continuation frequencies — the most common
                    // continuation wins with q = its empirical frequency.
                    let mut counts: std::collections::HashMap<u32, u32> =
                        std::collections::HashMap::new();
                    let mut total = 0u32;
                    for i in (0..hist.len() - order).rev() {
                        if &hist[i..i + order] == suffix {
                            *counts.entry(hist[i + order]).or_insert(0) += 1;
                            total += 1;
                        }
                    }
                    if total > 0 {
                        let (tok, cnt) = counts.iter().max_by_key(|(_, c)| **c).unwrap();
                        next = Some((*tok, *cnt as f32 / total as f32));
                    }
                }
                order -= 1;
            }
            match next {
                Some((t, q)) => {
                    drafts.push(t);
                    probs.push(q);
                    hist.push(t); // симулируем принятие для следующей позиции
                }
                None => break,
            }
        }
        (drafts, probs)
    }
}

/// N1 (build fix): these tests use the CPU backend — gate on the cpu feature
/// so bare `cargo test` on non-cpu builds does not fail to compile.
#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;

    #[test]
    fn ngram_finds_repetition() {
        let mut d = NGramDrafter::new(4);
        // история ... 7 8 9 — суффикс [7,8,9] встречался ранее → продолжение 10
        let history: Vec<u32> = vec![1, 2, 3, 7, 8, 9, 4, 5, 6, 7, 8, 9];
        let (drafts, probs) = d.draft(&crate::cpu::CpuBackend, &history, 3);
        assert_eq!(drafts[0], 4, "продолжение [7,8,9] → 4");
        assert_eq!(probs.len(), drafts.len());
    }

    #[test]
    fn manifests_v1_v2() {
        let v1 = DrafterManifest::v1("qwen3-8b", 16, 8);
        assert_eq!(v1.version, DrafterVersion::V1);
        assert_eq!(v1.n_layers, 5);
        assert_eq!(v1.block, 16);
        assert_eq!(v1.n_draft, 16);
        let v2 = DrafterManifest::v2("qwen3-8b");
        assert_eq!(v2.version, DrafterVersion::V2);
        assert_eq!(v2.n_layers, 8);
        assert_eq!(v2.block, 8);
        assert_eq!(v2.n_draft, 7);
    }

    #[test]
    fn ngram_short_history() {
        let mut d = NGramDrafter::new(4);
        let (drafts, _) = d.draft(&crate::cpu::CpuBackend, &[1, 2], 4);
        assert!(drafts.is_empty(), "нет повторений — пустой draft");
    }
}
