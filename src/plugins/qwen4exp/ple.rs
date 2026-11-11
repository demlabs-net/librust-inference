//! PLE — n-граммные эмбеддинги (`qwen4exp` / Flash-Next), шаг Э2.3.
//!
//! Порт `llm_graph_input_ple::set_input` из llama.cpp
//! (`src/models/qwen4exp.cpp`, эталон 5266f24da). На каждый токен считается
//! набор хэшей по окну из `ngram_size` предшественников; каждая пара
//! (длина n, голова g) даёт строку в общей таблице n-грамм (отдельный шард),
//! которые потом складываются с проекциями слоя PLE.
//!
//! Семантика окна (важна для паритета):
//! * EOS внутри окна «обрезает» всё, что было до него: такие позиции
//!   заменяются на EOS, а не отбрасываются;
//! * отсутствующий предшественник (начало последовательности) читается как EOS;
//! * собственный EOS токена своё окно НЕ обрезает — ctx[0] всегда сам токен.
//!
//! Хэш: `mixed = ctx[0]·m[0] ⊕ ctx[1]·m[1] ⊕ …` (u64, wrapping), строка =
//! `mixed % vocab[h] + offset[h]`. Множители и диапазоны голов — из
//! метаданных (`ple.layer_multipliers`, `ple.head_offsets`,
//! `ple.head_vocab_sizes`).

/// Предшественник, которого нет (начало последовательности).
pub const MISSING: u32 = u32::MAX;

/// Константы PLE из GGUF.
#[derive(Debug, Clone, PartialEq)]
pub struct PleCfg {
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    /// `(ngram_size - 1) · heads_per_ngram`.
    pub n_heads: usize,
    pub eos_token: u32,
    /// Токен-заглушка для позиций картинки (у Flash-Next = eos, если ключа нет).
    pub image_token: u32,
    pub layer_multipliers: Vec<u64>,
    pub head_offsets: Vec<u64>,
    pub head_vocab_sizes: Vec<u64>,
}

impl PleCfg {
    /// Собрать и проверить константы. Ошибки — громкие: молча взятая
    /// неполная таблица дала бы другие строки и тихо разошедшийся текст.
    pub fn new(
        ngram_size: usize,
        heads_per_ngram: usize,
        eos_token: u32,
        image_token: Option<u32>,
        layer_multipliers: Vec<u64>,
        head_offsets: Vec<u64>,
        head_vocab_sizes: Vec<u64>,
    ) -> Result<Self, String> {
        if ngram_size < 2 {
            return Err(format!("PLE: ngram_size {ngram_size} < 2"));
        }
        if heads_per_ngram == 0 {
            return Err("PLE: heads_per_ngram = 0".into());
        }
        let n_heads = (ngram_size - 1) * heads_per_ngram;
        if layer_multipliers.len() < ngram_size {
            return Err(format!(
                "PLE: множителей {} на окно {ngram_size}",
                layer_multipliers.len()
            ));
        }
        if head_offsets.len() < n_heads || head_vocab_sizes.len() < n_heads {
            return Err(format!(
                "PLE: голов {n_heads}, а диапазонов {}/{}",
                head_offsets.len(),
                head_vocab_sizes.len()
            ));
        }
        if let Some(h) = head_vocab_sizes.iter().take(n_heads).position(|v| *v == 0) {
            return Err(format!("PLE: у головы {h} нулевой словарь"));
        }
        Ok(Self {
            ngram_size,
            heads_per_ngram,
            n_heads,
            eos_token,
            image_token: image_token.unwrap_or(eos_token),
            layer_multipliers,
            head_offsets,
            head_vocab_sizes,
        })
    }

    pub fn n_prev(&self) -> usize {
        self.ngram_size - 1
    }

    /// Строки таблицы n-грамм для пачки токенов: `[n_tokens · n_heads]`,
    /// по `n_heads` подряд на токен (порядок голов — как у эталона).
    ///
    /// `prev` — предшественники, плоско `[n_tokens · n_prev]`, «старые
    /// первыми»: предшественник `s` позиций назад лежит по индексу
    /// `n_prev - s`. Отсутствие — [`MISSING`].
    pub fn rows(&self, tokens: &[u32], prev: &[u32]) -> Result<Vec<u32>, String> {
        let n_prev = self.n_prev();
        if prev.len() != tokens.len() * n_prev {
            return Err(format!(
                "PLE: предшественников {} на {} токенов × {n_prev}",
                prev.len(),
                tokens.len()
            ));
        }
        let mut out = vec![0u32; tokens.len() * self.n_heads];
        let mut ctx = vec![0u32; self.ngram_size];
        for (i, &tok) in tokens.iter().enumerate() {
            ctx[0] = tok;
            let mut cut = false;
            for s in 1..self.ngram_size {
                let t = if cut {
                    MISSING
                } else {
                    prev[i * n_prev + (n_prev - s)]
                };
                cut = cut || t == MISSING || t == self.eos_token;
                ctx[s] = if cut { self.eos_token } else { t };
            }
            for n in 2..=self.ngram_size {
                let mut mixed = (ctx[0] as u64).wrapping_mul(self.layer_multipliers[0]);
                for j in 1..n {
                    mixed ^= (ctx[j] as u64).wrapping_mul(self.layer_multipliers[j]);
                }
                let base = (n - 2) * self.heads_per_ngram;
                for g in 0..self.heads_per_ngram {
                    let h = base + g;
                    out[i * self.n_heads + h] =
                        (mixed % self.head_vocab_sizes[h] + self.head_offsets[h]) as u32;
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Окно 3, две головы на длину → четыре головы. Значения посчитаны
    /// руками по формуле эталона.
    fn cfg() -> PleCfg {
        PleCfg::new(
            3,
            2,
            7,
            None,
            vec![1, 3, 5],
            vec![100, 200, 300, 400],
            vec![10, 10, 10, 10],
        )
        .unwrap()
    }

    #[test]
    fn rows_match_hand_computed_hashes() {
        let c = cfg();
        // Токен 2, предшественники: 1 назад — 4, 2 назад — 3.
        // n=2: 2·1 ⊕ 4·3 = 2 ⊕ 12 = 14 → 14%10+{100,200} = 104, 204
        // n=3: 14 ⊕ 3·5 = 14 ⊕ 15 = 1 → 1%10+{300,400} = 301, 401
        let rows = c.rows(&[2], &[3, 4]).unwrap();
        assert_eq!(rows, vec![104, 204, 301, 401]);
    }

    /// EOS в окне обрезает всё, что было до него: дальние позиции читаются
    /// как EOS, а не выбрасываются.
    #[test]
    fn eos_cuts_the_window() {
        let c = cfg();
        // 2 назад — 7 (EOS), 1 назад — 4.
        // n=2: 2 ⊕ 12 = 14 → 104, 204
        // n=3: 14 ⊕ 7·5=35 → 14 ⊕ 35 = 45 → 45%10 = 5 → 305, 405
        let rows = c.rows(&[2], &[7, 4]).unwrap();
        assert_eq!(rows, vec![104, 204, 305, 405]);
    }

    /// Отсутствующий предшественник (начало последовательности) = EOS.
    #[test]
    fn missing_predecessor_reads_as_eos() {
        let c = cfg();
        let rows = c.rows(&[2], &[MISSING, 4]).unwrap();
        assert_eq!(rows, vec![104, 204, 305, 405]);
    }

    /// Собственный EOS токена своё окно не обрезает.
    #[test]
    fn own_eos_does_not_cut() {
        let c = cfg();
        // Токен 7 сам EOS, но окно строится как обычно (2 назад — 3, 1 назад — 4):
        // n=2: 7 ⊕ 12 = 11 → 101, 201; n=3: 11 ⊕ 15 = 4 → 304, 404.
        // Если бы окно обрезалось, было бы n=2: 7 ⊕ 7·3=21 → 18 → 108, 208.
        let rows = c.rows(&[7], &[3, 4]).unwrap();
        assert_eq!(rows, vec![101, 201, 304, 404]);
    }

    /// Два токена подряд: предшественники не перепутаны между токенами.
    #[test]
    fn multiple_tokens_use_their_own_window() {
        let c = cfg();
        // токен 2: окно [3, 4] → 104,204,301,401 (см. выше)
        // токен 5: n=2: 5 ⊕ 4·3=12 → 5⊕12=9 → 109,209; n=3: 9⊕15=6 → 306,406
        let rows = c.rows(&[2, 5], &[3, 4, 3, 4]).unwrap();
        assert_eq!(rows, vec![104, 204, 301, 401, 109, 209, 306, 406]);
    }

    #[test]
    fn rejects_incomplete_metadata() {
        assert!(PleCfg::new(1, 2, 0, None, vec![1], vec![0, 0], vec![1, 1]).is_err());
        assert!(
            PleCfg::new(
                3,
                2,
                0,
                None,
                vec![1, 2],
                vec![0, 0, 0, 0],
                vec![1, 1, 1, 1]
            )
            .is_err()
        );
        assert!(PleCfg::new(3, 2, 0, None, vec![1, 2, 3], vec![0, 0], vec![1, 1, 1, 1]).is_err());
        assert!(
            PleCfg::new(
                3,
                2,
                0,
                None,
                vec![1, 2, 3],
                vec![0, 0, 0, 0],
                vec![1, 0, 1, 1]
            )
            .is_err()
        );
    }
}
