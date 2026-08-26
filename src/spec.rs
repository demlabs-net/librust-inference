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

/// Генератор draft-токенов. `backend` нужен графическим драфтерам
/// (DFlash — диффузия на GPU); n-граммовый его игнорирует.
pub trait Drafter {
    /// Следующие `gamma` токенов и их вероятности q_i (для rejection).
    fn draft(&mut self, backend: &dyn crate::backend::Backend, history: &[u32], gamma: usize)
        -> (Vec<u32>, Vec<f32>);
}

/// Суффиксный n-граммовый драфтер: самый длинный суффикс истории с
/// предыдущим вхождением → продолжение (максимальный порядок — 4).
/// q=0.5: консервативная честная оценка (принимается при p >= 0.5).
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
        let mut drafts: Vec<u32> = Vec::new();
        let mut hist: Vec<u32> = history.to_vec();
        for _ in 0..gamma {
            let mut next = None;
            let mut order = hist.len().min(self.max_order);
            while order >= 1 && next.is_none() {
                if hist.len() > order {
                    let suffix = &hist[hist.len() - order..];
                    for i in (0..hist.len() - order).rev() {
                        if &hist[i..i + order] == suffix {
                            next = Some(hist[i + order]);
                            break;
                        }
                    }
                }
                order -= 1;
            }
            match next {
                Some(t) => {
                    drafts.push(t);
                    hist.push(t); // симулируем принятие для следующей позиции
                }
                None => break,
            }
        }
        let probs = vec![0.5f32; drafts.len()];
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
        let (drafts, probs) = d.draft(&crate::cpu::CpuBackend::default(), &history, 3);
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
        let (drafts, _) = d.draft(&crate::cpu::CpuBackend::default(), &[1, 2], 4);
        assert!(drafts.is_empty(), "нет повторений — пустой draft");
    }
}
