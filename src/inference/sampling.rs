//! Top-p, temperature, repetition penalty, RAS sampling (CPU vectors).

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

/// Top-p sample over a CPU logits vector (temperature applied inside).
pub fn top_p_sample_vec(
    logits: &[f32],
    params: &SamplingParams,
    rng_state: &mut Option<u64>,
) -> u32 {
    let logits_vec: Vec<f32> = if params.temperature != 1.0 {
        logits.iter().map(|&l| l / params.temperature).collect()
    } else {
        logits.to_vec()
    };
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

/// Sample one token from a small (already top-p filtered) distribution.
fn sample_from(top_tokens: &[(usize, f32)], rng_state: &mut Option<u64>) -> u32 {
    let total: f32 = top_tokens.iter().map(|(_, p)| p).sum();

    let r = match rng_state {
        Some(state) => {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (*state >> 33) as f32 / (1u64 << 33) as f32
        }
        None => {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos();
            let mut state = nanos as u64;
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *rng_state = Some(state);
            (state >> 33) as f32 / (1u64 << 33) as f32
        }
    };

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

/// Repetition penalty over a CPU logits vector (in place).
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
        let t = top_p_sample_vec(&logits, &params, &mut rng);
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
        // With top_p near 1.0 both paths effectively sample from the same
        // distribution: repeated draws must stay inside the top_k window.
        let mut rng_a = None;
        let mut rng_b = None;
        for _ in 0..50 {
            let ta = top_p_sample_vec(&logits, &mk(3), &mut rng_a);
            let tb = top_p_sample_vec(&logits, &mk(0), &mut rng_b);
            assert!(ta < 3 || ta == 2, "top_k=3 must stay in the top-3 window, got {ta}");
            // The full-sort path returns one of the 3 largest (0.9, 0.8, 0.7).
            assert!([1usize, 6, 3].contains(&(tb as usize)), "unexpected {tb}");
        }
    }
}
