//! Ф2.1: спекулятивный декодинг — драфтеры.
//!
//! [`Drafter`] — интерфейс генерации draft-токенов с вероятностями q_i.
//! Интерфейс передаёт СКАЛЯРНУЮ q (массу на выбранном токене), а остаток
//! rejection-sampling (p−q)⁺ требует всей раскладки q — значит скаляр честен
//! ТОЛЬКО как дельта: 1.0 для детерминированного черновика. При q=1 приём
//! идёт с вероятностью p(x), отказ — ресемпл из (p−δ_x)⁺; это лосслессно
//! при ЛЮБОМ поведении драфтера (сверка-семантика оракула). Любое q<1 без
//! полной раскладки смещает выход к черновику (ревью Ф2.9).
//! [`NGramDrafter`] — суффиксный lookup по истории (n-граммы), q=1.0 (дельта).

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

    /// Какие слои таргета нужны удалённому драфтеру (DFlash). КРИТИЧНО:
    /// координатор собирает extract-скрытые ТОЛЬКО для этих слоёв, и при
    /// пустом ответе не собирает их вовсе — драфтер остаётся слепым, выдаёт
    /// вырожденные черновики (последний id словаря, NaN-вероятности) и приёмка
    /// равна нулю. Дефолт — пусто (N-граму фичи не нужны).
    fn remote_target_layers(&self) -> Result<Vec<usize>, String> {
        Ok(Vec::new())
    }

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
    /// Слои таргета, запрошенные у соседа: кэшируется И отказ — иначе каждая
    /// итерация окна лезла в сеть за одним и тем же «нет» (ревью Ф2.9).
    layers: std::sync::OnceLock<Result<Vec<usize>, ()>>,
}

impl RemoteDrafter {
    pub fn new(transport: std::sync::Arc<dyn DraftTransport>) -> Self {
        Self {
            transport,
            last_error: None,
            layers: std::sync::OnceLock::new(),
        }
    }
}

impl Drafter for RemoteDrafter {
    fn target_layers(&self) -> &[usize] {
        // Слои приезжают от соседа: без них координатор не соберёт extract и
        // драфтер будет слепым (наблюдалось 0% приёмки и вырожденные черновики).
        let entry = self.layers.get_or_init(|| {
            self.transport.remote_target_layers().map_err(|e| {
                tracing::warn!("RemoteDrafter: сосед не отдал target_layers ({e}) — фич не будет");
            })
        });
        match entry {
            Ok(l) => l.as_slice(),
            Err(()) => &[],
        }
    }

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

/// Как драфтер выбирает черновики. Обязан совпадать с режимом проверки:
/// при жадной проверке (argmax таргета) сэмплированный черновик почти
/// никогда не совпадает с argmax — приёмка падает в разы (DFlash: 7–12%
/// вместо ~55% у llama.cpp на тех же весах).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DraftSampling {
    /// Жадная генерация: черновик = argmax строки драфтера.
    Greedy,
    /// Сэмплирующая: черновик из распределения драфтера с температурой и
    /// top-k запроса (0 — без отсечения); q остаётся дельтой (1.0) —
    /// скалярный интерфейс не перевозит полную раскладку (см. док модуля).
    Sample { temperature: f32, top_k: usize },
}

/// Генератор draft-токенов. `backend` нужен графическим драфтерам
/// (DFlash — диффузия на GPU); n-граммовый его игнорирует.
pub trait Drafter {
    /// Режим выбора черновиков на эту генерацию (вызывается петлёй
    /// `generate` перед первым окном). Драфтеры без распределения
    /// (n-граммы, удалённые) режим не используют.
    fn set_sampling(&mut self, _mode: DraftSampling) {}

    /// Забыть контекст прошлой генерации (кэш, фичи таргета, историю) —
    /// перед новой последовательностью с той же моделью.
    fn reset(&mut self) {}

    /// Следующие `gamma` токенов и их вероятности q_i (для rejection).
    /// Скалярная q честна только как дельта (1.0) — см. док модуля.
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
/// Выбор детерминированный (argmax частоты, тай-брейк — минимальный id),
/// поэтому q = 1.0 — дельта-предложение (см. док модуля: скалярная q<1 без
/// полной раскладки нечестна и смещает выход).
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
        // Строгий контекст: 1-2-грамматические совпадения на обычном тексте
        // случайны и только тратят окно (замер 27B: 12.7 против 14.6 tok/s).
        // Для цитат/правок/кода контекст повторяется целиком, порог не мешает.
        ngram_lookup_min(
            history,
            gamma,
            self.max_order,
            self.max_order.saturating_sub(1).max(2),
        )
    }
}

/// N-грамный lookup БЕЗ бэкенда (7.9): та же логика, что у
/// [`NGramDrafter`], но вызываемая там, где графа нет вовсе — например на
/// ноде-драфтере, обслуживающей черновики соседей по сети.
pub fn ngram_lookup(history: &[u32], gamma: usize, max_order: usize) -> (Vec<u32>, Vec<f32>) {
    ngram_lookup_min(history, gamma, max_order, 1)
}

/// То же, но совпадение засчитывается только с контекста длины `min_order`:
/// на обычном тексте короткие n-граммы повторяются случайно, и черновик
/// тратит окно впустую (замер 27B: 12.7 против 14.6 tok/s у чистого MTP);
/// для цитат/правок нужен длинный вербатимный контекст.
pub fn ngram_lookup_min(
    history: &[u32],
    gamma: usize,
    max_order: usize,
    min_order: usize,
) -> (Vec<u32>, Vec<f32>) {
    let max_order = max_order.max(1);
    let min_order = min_order.clamp(1, max_order);
    {
        let mut drafts: Vec<u32> = Vec::new();
        let mut probs: Vec<f32> = Vec::new();
        let mut hist: Vec<u32> = history.to_vec();
        for _ in 0..gamma {
            let mut next: Option<(u32, f32)> = None;
            let mut order = hist.len().min(max_order);
            while order >= min_order && next.is_none() {
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
                        // Детерминированный выбор (argmax частот) = ДЕЛЬТА-
                        // предложение: честная скалярная q — 1.0. Прежняя
                        // эмпирическая частота (< 1) завышала min(1, p/q) —
                        // over-acceptance и смещение выхода (ревью Ф2.9).
                        // Тай-брейк — минимальный id: max_by_key по HashMap
                        // недетерминирован между процессами (RandomState).
                        let (tok, _cnt) = counts
                            .iter()
                            .min_by(|(ka, a), (kb, b)| b.cmp(a).then(ka.cmp(kb)))
                            .unwrap();
                        next = Some((*tok, 1.0));
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

    /// Строгий контекст (min_order): повтор короткого суффикса не даёт
    /// черновика — иначе на обычном тексте lookup тратит окно впустую
    /// (замер 27B: 12.7 против 14.6 tok/s).
    #[test]
    fn ngram_min_order_requires_long_context() {
        // Повтор только 2-граммы [7,8] → при min_order=4 черновиков нет.
        let history: Vec<u32> = vec![7, 8, 1, 2, 3, 7, 8];
        let (short, _) = ngram_lookup_min(&history, 4, 4, 4);
        assert!(short.is_empty(), "2-грамма при min_order=4: {short:?}");
        // Полный 4-грамматический повтор: [5,6,7,8] → продолжение 9,10.
        let history: Vec<u32> = vec![5, 6, 7, 8, 9, 10, 1, 2, 3, 5, 6, 7, 8];
        let (long, _) = ngram_lookup_min(&history, 2, 4, 4);
        assert_eq!(&long[..2], &[9, 10], "4-грамма → вербатимное продолжение");
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
