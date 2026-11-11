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

// ─────────────────── вывод констант (порт ngram_hash.cpp) ───────────────────

const K_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const K_PRIME1: u64 = 10_007;

fn splitmix64(x: u64) -> u64 {
    let x = x.wrapping_add(K_GAMMA);
    let x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn is_prime(v: u64) -> bool {
    if v < 2 {
        return false;
    }
    if v.is_multiple_of(2) {
        return v == 2;
    }
    let mut d = 3u64;
    while d * d <= v {
        if v.is_multiple_of(d) {
            return false;
        }
        d += 2;
    }
    true
}

/// Константы хэша PLE, выведенные из конфига (как `NgramHash` в Infernix).
///
/// В GGUF они лежали бы массивами, но артефакт Infernix их не хранит —
/// значит обязаны совпасть бит-в-бит с этой формулой, иначе строки таблицы
/// n-грамм поедут и текст разойдётся.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PleHash {
    pub multipliers: Vec<u64>,
    pub offsets: Vec<u64>,
    pub sizes: Vec<u64>,
    /// Полное число строк таблицы, выровненное (`make_divisible`).
    pub padded_rows: u64,
}

/// Вывести константы. `ple_layer_index` — номер МОДУЛЯ PLE (не слоя модели):
/// у Flash-Next PLE один, и это 0 — проверено сверкой `padded_rows` с
/// `ngram_table.rows` живого артефакта (320 001 536).
#[allow(clippy::too_many_arguments)]
pub fn derive(
    seed: u64,
    vocab_size: u64,
    ngram_size: usize,
    heads_per_ngram: usize,
    ngram_vocab_size_base: u64,
    make_divisible: u64,
    ple_layer_index: u32,
) -> Result<PleHash, String> {
    if ngram_size < 2 || heads_per_ngram == 0 || vocab_size == 0 || ngram_vocab_size_base == 0 {
        return Err("PLE: неполный конфиг хэша".into());
    }
    if make_divisible == 0 {
        return Err("PLE: make_divisible = 0".into());
    }
    let max_long = i64::MAX as u64;
    let half_bound = (max_long / vocab_size / 2).max(1);
    let base = seed.wrapping_add(K_PRIME1.wrapping_mul(ple_layer_index as u64));
    let mut multipliers = Vec::with_capacity(ngram_size);
    for i in 0..ngram_size as u64 {
        multipliers.push(
            2 * (splitmix64(base.wrapping_add(K_GAMMA.wrapping_mul(i + 1))) % half_bound) + 1,
        );
    }

    let heads = (ngram_size - 1) * heads_per_ngram;
    let mut prime = ngram_vocab_size_base - 1;
    let mut total = 0u64;
    let mut sizes = Vec::with_capacity(heads);
    let mut offsets = Vec::with_capacity(heads);
    let first = ple_layer_index as u64 * heads as u64;
    let last = (ple_layer_index as u64 + 1) * heads as u64;
    for i in 0..last {
        loop {
            prime += 1;
            if is_prime(prime) {
                break;
            }
        }
        if i >= first {
            sizes.push(prime);
            offsets.push(total);
            total += prime;
        }
    }
    let padded_rows = total.div_ceil(make_divisible) * make_divisible;
    Ok(PleHash {
        multipliers,
        offsets,
        sizes,
        padded_rows,
    })
}

impl PleHash {
    /// Собрать `PleCfg` для слоя: множители/диапазоны — из вывода, EOS — из конфига.
    pub fn into_cfg(
        self,
        ngram_size: usize,
        heads_per_ngram: usize,
        eos: u32,
    ) -> Result<PleCfg, String> {
        PleCfg::new(
            ngram_size,
            heads_per_ngram,
            eos,
            None,
            self.multipliers,
            self.offsets,
            self.sizes,
        )
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

    /// Вывод констант сверен с живым артефактом: у Flash-Next таблица
    /// n-грамм ровно 320 001 536 строк (это `padded_rows` при layer_index 0).
    #[test]
    fn derived_hash_matches_live_ngram_table() {
        let h = derive(1234, 248_320, 3, 8, 20_000_000, 128, 0).unwrap();
        assert_eq!(h.sizes.len(), 16, "голов = (3-1)·8");
        assert_eq!(h.multipliers.len(), 3);
        assert_eq!(h.sizes[0], 20_000_003);
        assert_eq!(h.offsets[0], 0);
        assert_eq!(h.padded_rows, 320_001_536, "ngram_table.rows артефакта");
        // множители нечётные (как в эталоне: 2·r+1)
        assert!(h.multipliers.iter().all(|m| m % 2 == 1));
        // смещения — префиксные суммы размеров
        let mut acc = 0u64;
        for (o, sz) in h.offsets.iter().zip(h.sizes.iter()) {
            assert_eq!(*o, acc);
            acc += sz;
        }
        // другой индекс модуля PLE даёт другой набор — проверка, что параметр жив
        let h1 = derive(1234, 248_320, 3, 8, 20_000_000, 128, 1).unwrap();
        assert_ne!(h.sizes, h1.sizes);
        assert_ne!(h.multipliers, h1.multipliers);
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
