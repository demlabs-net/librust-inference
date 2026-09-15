//! Top-p, temperature, repetition penalty, RAS sampling (CPU vectors).

/// The project-wide 64-bit LCG step (Knuth MMIX constants). Every stochastic
/// path in the workspace shares this one step so that a single change to the
/// generator cannot leave half the code on the old distribution.
#[inline]
pub fn lcg_next(state: u64) -> u64 {
    state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

/// Uniform `f32` in `[0, 1)` derived from a raw LCG state.
///
/// Uses the top 24 bits, which `f32` represents exactly: the largest value is
/// `(2^24 - 1) / 2^24 < 1.0`. Deriving from fewer bits (the previous
/// `state >> 33`) silently produced `[0, 0.25)`, and dividing a wider value by
/// `2^32` could round up to exactly `1.0`.
#[inline]
pub fn lcg_unit_f32(state: u64) -> f32 {
    ((state >> 40) as f32) * (1.0 / (1u64 << 24) as f32)
}

/// Draw a uniform `f32` from a mutable LCG state, seeding from the clock on the
/// first call (`None`). The advanced state is written back, so a caller that
/// keeps `Some(state)` gets a deterministic stream.
pub fn next_unit_f32(state: &mut Option<u64>) -> f32 {
    let next = match state {
        Some(s) => lcg_next(*s),
        None => lcg_next(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos() as u64,
        ),
    };
    *state = Some(next);
    lcg_unit_f32(next)
}

/// Sampling parameters.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    /// Top-k cutoff: keep only the k highest-probability tokens before applying
    /// top-p (0 = no limit). fishaudio S2-Pro reference default is 30.
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_p: 0.8,
            top_k: 0,
            repetition_penalty: 1.1,
            seed: None,
        }
    }
}

/// Top-p sample over a CPU logits vector (temperature + repetition penalty
/// applied inside). `history` — токены уже в контексте (промпт + порождённые):
/// их логиты штрафуются `repetition_penalty` (раньше параметр был no-op).
/// История ДЕДУПЛИЦИРУЕТСЯ (уникальные токены — как в HF); порядок HF:
/// сначала штраф по сырым логитам, потом температура.
pub fn top_p_sample_vec_history(
    logits: &[f32],
    params: &SamplingParams,
    history: &[u32],
    rng_state: &mut Option<u64>,
) -> u32 {
    let mut logits_vec = logits.to_vec();
    apply_repetition_penalty_vec_unique(&mut logits_vec, history, params.repetition_penalty);
    if params.temperature != 1.0 {
        for l in logits_vec.iter_mut() {
            *l /= params.temperature;
        }
    }
    let max_logit = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = logits_vec.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum_exp: f32 = exp_vals.iter().sum();
    let probs_vec: Vec<f32> = if sum_exp.is_finite() && sum_exp > 0.0 {
        exp_vals.iter().map(|&e| e / sum_exp).collect()
    } else {
        let n = logits_vec.len() as f32;
        vec![1.0 / n; logits_vec.len()]
    };

    let mut indexed: Vec<(usize, f32)> =
        probs_vec.iter().enumerate().map(|(i, &p)| (i, p)).collect();

    let k = if params.top_k > 0 {
        params.top_k.min(indexed.len())
    } else {
        indexed.len()
    };
    // M15: with top_k set (the TTS path uses 30), a full sort of the whole
    // vocab (155k entries ≈ 2-6 ms) per token is wasted work. select_nth
    // isolates the k largest in O(V), then only that window is sorted.
    //
    // NOTE: select_nth_unstable_by(n) puts the n-th element into the PIVOT —
    // head has length n, so the k largest are head + pivot (the old code
    // sliced head[..k], which panicked: "range end index out of range").
    if k < indexed.len() {
        let (head, pivot, _) = indexed.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
        let mut window: Vec<(usize, f32)> = head.to_vec();
        window.push(*pivot);
        window.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        let mut cumsum = 0.0f32;
        let mut cutoff = k;
        for (i, (_, p)) in window.iter().enumerate() {
            cumsum += p;
            if cumsum >= params.top_p {
                cutoff = i + 1;
                break;
            }
        }
        let top_tokens: Vec<(usize, f32)> = window[..cutoff].to_vec();
        return sample_from(&top_tokens, rng_state);
    }

    indexed.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

    let mut cumsum = 0.0f32;
    let mut cutoff = k;
    for (i, (_, p)) in indexed[..k].iter().enumerate() {
        cumsum += p;
        if cumsum >= params.top_p {
            cutoff = i + 1;
            break;
        }
    }

    let top_tokens: Vec<(usize, f32)> = indexed[..cutoff].to_vec();
    sample_from(&top_tokens, rng_state)
}

/// Старое API (без истории) — speech-крейт и прежние вызовы. Без истории
/// repetition penalty не применяется; для штрафа — top_p_sample_vec_history.
pub fn top_p_sample_vec(
    logits: &[f32],
    params: &SamplingParams,
    rng_state: &mut Option<u64>,
) -> u32 {
    top_p_sample_vec_history(logits, params, &[], rng_state)
}

/// Sample one token from a small (already top-p filtered) distribution.
fn sample_from(top_tokens: &[(usize, f32)], rng_state: &mut Option<u64>) -> u32 {
    let total: f32 = top_tokens.iter().map(|(_, p)| p).sum();

    let r = next_unit_f32(rng_state);

    let threshold = r * total;
    let mut cumsum = 0.0f32;
    for (idx, p) in top_tokens {
        cumsum += p;
        if cumsum >= threshold {
            return *idx as u32;
        }
    }
    top_tokens[0].0 as u32
}

/// Repetition penalty over a CPU logits vector (in place). Standard
/// behaviour (HF): every UNIQUE token of the history with nonzero logit is
/// penalised once — logit > 0 divided by the penalty, otherwise multiplied.
///
/// Wave-8 note: this fn is per-OCCURRENCE — the speech crate's TTS calls it
/// directly with the full history each step, and the previous dedup here
/// silently changed TTS penalty semantics (compounding per repeat). The
/// once-per-unique-token behaviour lives in
/// [`apply_repetition_penalty_vec_unique`] / [`top_p_sample_vec_history`].
pub fn apply_repetition_penalty_vec(logits: &mut [f32], history: &[u32], penalty: f32) {
    if penalty == 1.0 || history.is_empty() {
        return;
    }
    for &token_id in history {
        let idx = token_id as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

/// Dedup-вариант для LLM-генерации ([`top_p_sample_vec_history`],
/// greedy в [`crate::inference::generate`]): штраф ОДИН раз на уникальный
/// токен истории — повтор в истории не усиливает штраф (HF
/// RepetitionPenaltyLogitsProcessor).
pub fn apply_repetition_penalty_vec_unique(logits: &mut [f32], history: &[u32], penalty: f32) {
    if penalty == 1.0 || history.is_empty() {
        return;
    }
    let mut seen: Vec<u32> = history.to_vec();
    seen.sort_unstable();
    seen.dedup();
    apply_repetition_penalty_vec(logits, &seen, penalty);
}

/// RAS penalty over a CPU logits vector (in place).
pub fn ras_penalty_vec(logits: &mut [f32], history: &[u32], window: usize, penalty: f32) {
    if history.len() < 2 || penalty == 1.0 {
        return;
    }
    let window_start = history.len().saturating_sub(window);
    let recent = &history[window_start..];
    let mut counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for &t in recent {
        *counts.entry(t).or_insert(0) += 1;
    }
    for (&token, &count) in &counts {
        if count > 1 {
            let idx = token as usize;
            if idx < logits.len() {
                logits[idx] -= penalty * (count - 1) as f32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// M15 regression: the top_k path sliced head[..k] where head is k-1
    /// long (select_nth puts the k-th element in the pivot) — k=1 panicked
    /// with "range end index out of range".
    #[test]
    fn top_k_one_does_not_panic() {
        let logits = [1.0f32, 2.0, 3.0, 0.5, -1.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 1,
            repetition_penalty: 1.0,
            seed: Some(42),
        };
        let mut rng = None;
        // Must return the argmax (index 2) and not panic.
        let t = top_p_sample_vec_history(&logits, &params, &[], &mut rng);
        assert_eq!(t, 2, "top_k=1 must yield the argmax");
    }

    #[test]
    fn top_k_window_matches_full_sort() {
        let logits = [0.1f32, 0.9, 0.3, 0.7, 0.5, -0.2, 0.8, 0.4];
        let mk = |top_k: usize| SamplingParams {
            temperature: 1.0,
            top_p: 0.99,
            top_k,
            repetition_penalty: 1.0,
            seed: Some(7),
        };
        // Three largest logits are 0.9 (id 1), 0.8 (id 6), 0.7 (id 3).
        const TOP3: [usize; 3] = [1, 6, 3];
        // NOTE: with top_p=0.99 the cumulative mass reaches the cutoff only on
        // the LAST of the 8 tokens, so the unrestricted path legitimately may
        // return any of them. The earlier version of this test asserted that
        // the unrestricted draw always landed in the top-3 — it only ever held
        // because the RNG could not produce a draw above 0.25.
        let mut rng_b = None;
        let mut outside_top3 = 0;
        let mut rng_a = None;
        for _ in 0..200 {
            let ta = top_p_sample_vec_history(&logits, &mk(3), &[], &mut rng_a) as usize;
            assert!(
                TOP3.contains(&ta),
                "top_k=3 must stay in the top-3 window, got {ta}"
            );
            let tb = top_p_sample_vec_history(&logits, &mk(0), &[], &mut rng_b) as usize;
            if !TOP3.contains(&tb) {
                outside_top3 += 1;
            }
        }
        assert!(
            outside_top3 > 0,
            "an unrestricted draw never left the top-3 window — the RNG is truncated"
        );
    }

    /// Repetition penalty больше не no-op: токен из истории выталкивается
    /// из top-1 окна; повтор токена в истории не усиливает штраф
    /// (уникальные токены, как в HF).
    #[test]
    fn repetition_penalty_demotes_history_tokens() {
        let logits = [2.0f32, 1.5, 0.9, 0.1];
        let mk = |rp: f32| SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 1,
            repetition_penalty: rp,
            seed: Some(11),
        };
        // Без истории: argmax = 0 (top_k=1 — только он в окне).
        let mut rng = None;
        assert_eq!(
            top_p_sample_vec_history(&logits, &mk(1.0), &[], &mut rng),
            0
        );
        // Токен 0 в истории: 2.0/2 = 1.0 < 1.5 → argmax = 1 → top_k=1 окно
        // не содержит 0 → ни один draw не вернёт 0.
        let mut rng = None;
        for _ in 0..10 {
            assert_ne!(
                top_p_sample_vec_history(&logits, &mk(2.0), &[0], &mut rng),
                0,
                "penalised token must leave the top-1 window"
            );
        }
        // Повтор в истории не удваивает штраф: [0,0] → по-прежнему 2.0/2 =
        // 1.0 (без dedup было бы 0.5 → argmax 2).
        let mut rng = None;
        assert_eq!(
            top_p_sample_vec_history(&logits, &mk(2.0), &[0, 0], &mut rng),
            1
        );
        // penalty == 1.0 — no-op.
        let mut rng = None;
        assert_eq!(
            top_p_sample_vec_history(&logits, &mk(1.0), &[0], &mut rng),
            0
        );
    }

    /// Wave-8 revert: direct `apply_repetition_penalty_vec` is per-OCCURRENCE
    /// (the speech crate's TTS calls it with the full history each step —
    /// the dedup here had silently changed TTS penalty semantics). The dedup
    /// lives in the history path (`top_p_sample_vec_history` / `_unique`).
    #[test]
    fn direct_penalty_compounds_per_occurrence() {
        let mut lg = vec![2.0f32, 1.5, 0.9, 0.1];
        apply_repetition_penalty_vec(&mut lg, &[0, 0], 2.0);
        // 2.0 /2 /2 = 0.5 < 0.9 — повтор УСИЛИЛ штраф (как до wave-8).
        assert!((lg[0] - 0.5).abs() < 1e-6, "per-occurrence: {}", lg[0]);
        // Dedup-вариант: [0,0] → один штраф → 1.0.
        let mut lg = vec![2.0f32, 1.5, 0.9, 0.1];
        apply_repetition_penalty_vec_unique(&mut lg, &[0, 0], 2.0);
        assert!((lg[0] - 1.0).abs() < 1e-6, "unique: {}", lg[0]);
    }

    /// Штраф применяется по сырым логитам, температура — после (HF-порядок;
    /// для скалярных делений результат эквивалентен — фиксируем контракт).
    #[test]
    fn penalty_before_temperature() {
        let logits = [2.0f32, 1.5, 0.9, 0.1];
        let params = SamplingParams {
            temperature: 0.5,
            top_p: 1.0,
            top_k: 1,
            repetition_penalty: 2.0,
            seed: Some(3),
        };
        let mut rng = None;
        // Штраф: [1.0, 1.5, 0.9, 0.1]; температура: [2.0, 3.0, 1.8, 0.2]
        // → argmax = 1.
        assert_eq!(
            top_p_sample_vec_history(&logits, &params, &[0], &mut rng),
            1
        );
    }

    /// The old `(state >> 33) as f32 / 2^33` derivation could only return
    /// [0, 0.25): the top of the range was unreachable and the mean sat at
    /// 0.125. Guard the full [0, 1) span and the mean.
    #[test]
    fn unit_draw_spans_full_range() {
        let mut st = Some(1u64);
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        let mut sum = 0.0f64;
        const N: usize = 100_000;
        for _ in 0..N {
            let v = next_unit_f32(&mut st);
            assert!((0.0..1.0).contains(&v), "draw outside [0,1): {v}");
            lo = lo.min(v);
            hi = hi.max(v);
            sum += v as f64;
        }
        assert!(hi > 0.99, "upper range unreachable, max = {hi}");
        assert!(lo < 0.01, "lower range unreachable, min = {lo}");
        let mean = sum / N as f64;
        assert!((mean - 0.5).abs() < 0.01, "mean {mean} is not 0.5");
    }

    /// Regression for the truncated draw: two EQUAL logits must be selected
    /// about equally often. Under the old formula the first candidate owned
    /// the whole threshold range, so the second was never reachable.
    #[test]
    fn equal_tokens_are_selected_evenly() {
        let logits = [0.0f32, 0.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            seed: None,
        };
        let mut st = Some(987_654_321u64);
        let (mut first, mut second) = (0u32, 0u32);
        for _ in 0..10_000 {
            if top_p_sample_vec(&logits, &params, &mut st) == 0 {
                first += 1;
            } else {
                second += 1;
            }
        }
        assert!(
            first > 4_000 && second > 4_000,
            "biased selection: token0={first}, token1={second}"
        );
    }
}
