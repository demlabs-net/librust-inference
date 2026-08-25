//! Ф2.1: спекулятивный декодинг — драфтеры.
//!
//! [`Drafter`] — интерфейс генерации draft-токенов с честными вероятностями
//! q_i (rejection-sampling в [`crate::qwen3::Qwen3Model::decode_spec`]
//! сохраняет целевое распределение при ЛЮБОЙ q; acceptance падает при
//! неточных q). [`NGramDrafter`] — суффиксный lookup по истории (n-граммы),
//! q=0.5 консервативно.

/// Генератор draft-токенов.
pub trait Drafter {
    /// Следующие `gamma` токенов и их вероятности q_i (для rejection).
    fn draft(&mut self, history: &[u32], gamma: usize) -> (Vec<u32>, Vec<f32>);
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
    fn draft(&mut self, history: &[u32], gamma: usize) -> (Vec<u32>, Vec<f32>) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ngram_finds_repetition() {
        let mut d = NGramDrafter::new(4);
        // история ... 7 8 9 — суффикс [7,8,9] встречался ранее → продолжение 10
        let history: Vec<u32> = vec![1, 2, 3, 7, 8, 9, 4, 5, 6, 7, 8, 9];
        let (drafts, probs) = d.draft(&history, 3);
        assert_eq!(drafts[0], 4, "продолжение [7,8,9] → 4");
        assert_eq!(probs.len(), drafts.len());
    }

    #[test]
    fn ngram_short_history() {
        let mut d = NGramDrafter::new(4);
        let (drafts, _) = d.draft(&[1, 2], 4);
        assert!(drafts.is_empty(), "нет повторений — пустой draft");
    }
}
