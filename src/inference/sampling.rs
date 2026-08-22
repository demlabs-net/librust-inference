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
    indexed.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

    let k = if params.top_k > 0 {
        params.top_k.min(indexed.len())
    } else {
        indexed.len()
    };
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
    for (idx, p) in &top_tokens {
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
