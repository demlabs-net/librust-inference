//! Top-p, temperature, repetition penalty, RAS sampling.

#[cfg(feature = "candle")]
use candle_core::{DType, Device, Result, Tensor};

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

/// Top-p (nucleus) sampling with persistent RNG state.
///
/// `rng_state` is mutated to advance the PRNG across calls.
#[cfg(feature = "candle")]
pub fn top_p_sample(logits: &Tensor, params: &SamplingParams, rng_state: &mut Option<u64>) -> candle_core::Result<u32> {
    let device = logits.device();

    // Apply temperature — use the f64 scalar operator (dtype-safe in candle 0.11,
    // keeps the logits dtype, e.g. BF16).
    let logits = if params.temperature != 1.0 {
        (logits * (1.0f64 / params.temperature as f64))?
    } else {
        logits.clone()
    };

    // Convert to probabilities — manual softmax (candle 0.11 lacks CUDA softmax_last_dim).
    // Bring logits to F32 for the CPU-side softmax (they may be BF16 on GPU).
    let logits_vec: Vec<f32> = logits.to_dtype(DType::F32)?.to_vec1()?;
    let max_logit = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = logits_vec.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum_exp: f32 = exp_vals.iter().sum();
    let probs_vec: Vec<f32> = if sum_exp.is_finite() && sum_exp > 0.0 {
        exp_vals.iter().map(|&e| e / sum_exp).collect()
    } else {
        // Degenerate logits (all -inf → exp(NaN), or a NaN slipped in):
        // fall back to a uniform distribution instead of propagating NaN
        // (partial_cmp on NaN used to panic here).
        let n = logits_vec.len() as f32;
        vec![1.0 / n; logits_vec.len()]
    };

    // Sort by probability descending. total_cmp is NaN-safe — a NaN never
    // panics the sort (partial_cmp().unwrap() did).
    let mut indexed: Vec<(usize, f32)> = probs_vec.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

    // Top-k: restrict the candidate set to the k highest-probability tokens
    // first (0 = no limit). Top-p then narrows further within that window.
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

    // Sample from top-p tokens using persistent RNG state
    let top_tokens: Vec<(usize, f32)> = indexed[..cutoff].to_vec();
    let total: f32 = top_tokens.iter().map(|(_, p)| p).sum();

    // FIX B-6: Advance PRNG state across calls instead of resetting every time
    let r = match rng_state {
        Some(state) => {
            // LCG step: advances state
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (*state >> 33) as f32 / (1u64 << 33) as f32
        }
        None => {
            // Non-deterministic: mix time + counter for better entropy
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos();
            let mut state = nanos as u64;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *rng_state = Some(state); // Save state for next call
            (state >> 33) as f32 / (1u64 << 33) as f32
        }
    };

    let threshold = r * total;
    let mut cumsum = 0.0f32;
    for (idx, p) in &top_tokens {
        cumsum += p;
        if cumsum >= threshold {
            return Ok(*idx as u32);
        }
    }

    // Fallback: return the most likely token
    Ok(top_tokens[0].0 as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampling_params_default() {
        let p = SamplingParams::default();
        assert!((p.temperature - 0.8).abs() < 1e-6);
        assert!((p.top_p - 0.8).abs() < 1e-6);
        assert!((p.repetition_penalty - 1.1).abs() < 1e-6);
    }

    #[test]
    fn test_top_p_sample_deterministic() {
        let dev = crate::backend::default_device(0).expect("device").0;
        let mut logits = vec![-1.0f32; 4096];
        logits[100] = 10.0;
        let tensor = Tensor::new(logits.as_slice(), &dev).unwrap();
        let params = SamplingParams { seed: Some(42), ..Default::default() };
        let mut rng = params.seed;
        let token = top_p_sample(&tensor, &params, &mut rng).unwrap();
        assert_eq!(token, 100);
    }

    #[test]
    fn test_repetition_penalty() {
        let dev = crate::backend::default_device(0).expect("device").0;
        let logits = Tensor::new(&[1.0f32, 2.0, 3.0], &dev).unwrap();
        let history = vec![2u32]; // Token 2 was used
        let penalized = apply_repetition_penalty(&logits, &history, 1.5).unwrap();
        let vals: Vec<f32> = penalized.to_vec1().unwrap();
        // Token 2 (index 2) should be penalized: 3.0 / 1.5 = 2.0
        assert!((vals[2] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_ras_penalty() {
        let dev = crate::backend::default_device(0).expect("device").0;
        let logits = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &dev).unwrap();
        let history = vec![1u32, 1u32]; // Token 1 repeated
        let penalized = ras_penalty(&logits, &history, 10, 0.5).unwrap();
        let vals: Vec<f32> = penalized.to_vec1().unwrap();
        // Token 1 should be penalized (count > 1)
        assert!(vals[1] < 2.0);
    }
}

/// Apply repetition penalty to logits.
///
/// For tokens in history: divide logits by penalty (if logit > 0)
/// or multiply by penalty (if logit < 0).
#[cfg(feature = "candle")]
pub fn apply_repetition_penalty(
    logits: &Tensor,
    history: &[u32],
    penalty: f32,
) -> Result<Tensor> {
    if penalty == 1.0 || history.is_empty() {
        return Ok(logits.clone());
    }

    let orig_dtype = logits.dtype();
    let mut logits_vec: Vec<f32> = logits.to_dtype(DType::F32)?.to_vec1()?;
    for &token_id in history {
        let idx = token_id as usize;
        if idx < logits_vec.len() {
            if logits_vec[idx] > 0.0 {
                logits_vec[idx] /= penalty;
            } else {
                logits_vec[idx] *= penalty;
            }
        }
    }

    // Restore the original dtype (e.g. BF16) so downstream ops don't mismatch.
    Tensor::new(logits_vec.as_slice(), logits.device())?.to_dtype(orig_dtype)
}

/// Repetition Aware Sampling (RAS): check for repetitions in window.
///
/// If any token appears more than once in the last `window` tokens,
/// apply extra penalty to those tokens.
#[cfg(feature = "candle")]
pub fn ras_penalty(
    logits: &Tensor,
    history: &[u32],
    window: usize,
    penalty: f32,
) -> Result<Tensor> {
    if history.len() < 2 || penalty == 1.0 {
        return Ok(logits.clone());
    }

    let window_start = history.len().saturating_sub(window);
    let recent = &history[window_start..];

    // Count token occurrences in window
    let mut counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for &t in recent {
        *counts.entry(t).or_insert(0) += 1;
    }

    let orig_dtype = logits.dtype();
    let mut logits_vec: Vec<f32> = logits.to_dtype(DType::F32)?.to_vec1()?;
    for (&token, &count) in &counts {
        if count > 1 {
            let idx = token as usize;
            if idx < logits_vec.len() {
                logits_vec[idx] -= penalty * (count - 1) as f32;
            }
        }
    }

    // Restore the original dtype (e.g. BF16) so downstream ops don't mismatch.
    Tensor::new(logits_vec.as_slice(), logits.device())?.to_dtype(orig_dtype)

}
    /// Regression: all-(-inf) logits produced exp(NaN) probabilities and
    /// partial_cmp().unwrap() panicked in the sort — a degenerate model
    /// state crashed the process mid-generation. Must return SOME token.
    #[test]
    fn top_p_sample_survives_all_negative_inf_logits() {
        let device = candle_core::Device::Cpu;
        let logits = candle_core::Tensor::from_vec(
            vec![f32::NEG_INFINITY; 16],
            (16,),
            &device,
        )
        .unwrap();
        let params = SamplingParams::default();
        let mut rng = Some(42u64);
        let tok = top_p_sample(&logits, &params, &mut rng).unwrap();
        assert!((tok as usize) < 16, "token in range: {tok}");
        // A second call with advanced RNG also works.
        let tok2 = top_p_sample(&logits, &params, &mut rng).unwrap();
        assert!((tok2 as usize) < 16, "token in range: {tok2}");
    }

// ── CPU (Vec) variants for the Metal engine ─────────────────────────────

/// Top-p sample over a CPU logits vector (temperature applied inside).
/// Mirrors `top_p_sample` but takes `&[f32]` (Metal engine path).
pub fn top_p_sample_vec(logits: &[f32], params: &SamplingParams, rng_state: &mut Option<u64>) -> u32 {
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

    let mut indexed: Vec<(usize, f32)> = probs_vec.iter().enumerate().map(|(i, &p)| (i, p)).collect();
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
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (*state >> 33) as f32 / (1u64 << 33) as f32
        }
        None => {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos();
            let mut state = nanos as u64;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
