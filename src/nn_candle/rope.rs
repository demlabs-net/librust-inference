//! Rotary Position Embedding (RoPE) — interleaved convention.
//!
//! Python reference (llama.py:1004-1038) uses interleaved pairs (x[2k], x[2k+1]),
//! NOT halved pairs (x[k], x[k+half]). This implementation matches the reference.
//!
//! freq = 1.0 / (base^(2i/head_dim)) for i in 0..head_dim/2
//! Stored as complex pairs [cos, sin] of shape (max_seq_len, head_dim//2, 2)

use candle_core::{DType, Device, Result, Tensor};

/// Precomputed RoPE frequency table.
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    /// Precomputed cos values: (max_seq_len, head_dim//2)
    cos: Tensor,
    /// Precomputed sin values: (max_seq_len, head_dim//2)
    sin: Tensor,
}

impl RotaryEmbedding {
    pub fn new(head_dim: usize, max_seq_len: usize, base: f64, device: &Device, dtype: DType) -> Result<Self> {
        let half_dim = head_dim / 2;

        // Compute frequencies in f64 for precision, then cast to f32
        let freqs: Vec<f32> = (0..half_dim)
            .map(|i| (1.0f64 / base.powf(2.0 * i as f64 / head_dim as f64)) as f32)
            .collect();
        let freqs = Tensor::new(freqs.as_slice(), device)?;

        // t = [0, 1, 2, ..., max_seq_len-1]
        let t = Tensor::arange(0u32, max_seq_len as u32, device)?.to_dtype(DType::F32)?;

        // outer(t, freqs): (max_seq_len, half_dim)
        let freqs = t.unsqueeze(1)?.broadcast_mul(&freqs.unsqueeze(0)?)?;

        // Store the tables in the model dtype so they match the Q/K activations.
        Ok(Self {
            cos: freqs.cos()?.to_dtype(dtype)?,
            sin: freqs.sin()?.to_dtype(dtype)?,
        })
    }

    /// Apply rotary embedding to Q and K tensors.
    ///
    /// Input shape: (batch, n_heads, seq_len, head_dim)
    /// Output shape: same
    ///
    /// Uses interleaved convention matching Python reference:
    ///   For each pair (x[2k], x[2k+1]):
    ///     x'[2k]   = x[2k] * cos[k] - x[2k+1] * sin[k]
    ///     x'[2k+1] = x[2k] * sin[k] + x[2k+1] * cos[k]
    pub fn apply(&self, q: &Tensor, k: &Tensor, start_pos: usize) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;

        // Slice cos/sin for the current positions: (seq_len, half_dim)
        let cos = self.cos.narrow(0, start_pos, seq_len)?;
        let sin = self.sin.narrow(0, start_pos, seq_len)?;

        let q_rot = apply_rotary_interleaved(q, &cos, &sin)?;
        let k_rot = apply_rotary_interleaved(k, &cos, &sin)?;
        Ok((q_rot, k_rot))
    }

    /// Get the precomputed cos table (for external use / debugging).
    pub fn cos(&self) -> &Tensor { &self.cos }
    pub fn sin(&self) -> &Tensor { &self.sin }
}

/// Apply rotary embedding using interleaved pairs.
///
/// Input x: (batch, n_heads, seq_len, head_dim)
/// cos/sin: (seq_len, head_dim//2) — will be broadcast to (1, 1, seq_len, head_dim//2)
fn apply_rotary_interleaved(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let head_dim = x.dim(3)?;
    let half = head_dim / 2;

    // Reshape x to (..., seq_len, half, 2) to extract interleaved pairs
    let x_shape = x.dims();
    let x_reshaped = x.reshape((x_shape[0], x_shape[1], x_shape[2], half, 2))?;

    // x_even = x[..., 0] (elements 0, 2, 4, ...)
    // x_odd  = x[..., 1] (elements 1, 3, 5, ...)
    let x_even = x_reshaped.narrow(4, 0, 1)?.squeeze(4)?;
    let x_odd = x_reshaped.narrow(4, 1, 1)?.squeeze(4)?;

    // Broadcast cos/sin: (seq_len, half) -> (1, 1, seq_len, half)
    let cos_b = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin_b = sin.unsqueeze(0)?.unsqueeze(0)?;

    // Interleaved rotation:
    //   out_even = x_even * cos - x_odd * sin
    //   out_odd  = x_even * sin + x_odd * cos
    let out_even = (x_even.broadcast_mul(&cos_b)? - x_odd.broadcast_mul(&sin_b)?)?;
    let out_odd = (x_even.broadcast_mul(&sin_b)? + x_odd.broadcast_mul(&cos_b)?)?;

    // Stack back to interleaved: (..., seq_len, half, 2)
    let out_stacked = Tensor::stack(&[&out_even, &out_odd], 4)?;

    // Flatten back to (..., seq_len, head_dim)
    out_stacked.reshape(x_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_precompute_shape() {
        let dev = crate::backend::default_device(0).expect("device").0;
        let rope = RotaryEmbedding::new(128, 32768, 1_000_000.0, &dev, DType::F32).unwrap();
        assert_eq!(rope.cos().shape().dims(), &[32768, 64]);
        assert_eq!(rope.sin().shape().dims(), &[32768, 64]);
    }

    #[test]
    fn test_rope_apply_rotation() {
        let dev = crate::backend::default_device(0).expect("device").0;
        let rope = RotaryEmbedding::new(64, 100, 10000.0, &dev, DType::F32).unwrap();
        let q = Tensor::rand(0.0f32, 1.0f32, (1, 4, 10, 64), &dev).unwrap();
        let k = Tensor::rand(0.0f32, 1.0f32, (1, 4, 10, 64), &dev).unwrap();
        let (q_rot, k_rot) = rope.apply(&q, &k, 0).unwrap();
        assert_eq!(q_rot.shape().dims(), q.shape().dims());
        assert_eq!(k_rot.shape().dims(), k.shape().dims());
    }

    #[test]
    fn test_rope_fast_ar() {
        let dev = crate::backend::default_device(0).expect("device").0;
        let rope = RotaryEmbedding::new(128, 10, 1_000_000.0, &dev, DType::F32).unwrap();
        assert_eq!(rope.cos().shape().dims(), &[10, 64]);
    }

    /// Regression test: verify interleaved convention produces different output
    /// than the old halved convention for a known input.
    #[test]
    fn test_rope_interleaved_vs_halved() {
        let dev = crate::backend::default_device(0).expect("device").0;
        // head_dim=4, single position, base=1.0 so freq=1.0 for all
        let _rope = RotaryEmbedding::new(4, 1, 1.0, &dev, DType::F32).unwrap();
        // cos=[cos(0)]=1, sin=[sin(0)]=0 for position 0
        // Actually freq = 1/(1^(2i/4)) = 1 for all i, so angles = [0, 0]
        // cos=[1,1], sin=[0,0] → rotation is identity at position 0

        // For position 1: angles = [1, 1], cos=[cos(1), cos(1)]
        let rope2 = RotaryEmbedding::new(4, 2, 1.0, &dev, DType::F32).unwrap();
        // Input: [a, b, c, d] with head_dim=4
        let q = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &dev).unwrap()
            .reshape((1, 1, 1, 4)).unwrap();

        let (q_rot, _) = rope2.apply(&q, &q, 1).unwrap();
        let vals: Vec<f32> = q_rot.flatten_all().unwrap().to_vec1().unwrap();

        // Interleaved: pairs (1,2) and (3,4) rotated by angle=1
        // out[0] = 1*cos(1) - 2*sin(1)
        // out[1] = 1*sin(1) + 2*cos(1)
        let (c, s) = (1.0f32.cos(), 1.0f32.sin());
        assert!((vals[0] - (1.0*c - 2.0*s)).abs() < 0.01, "out[0]={} expected {}", vals[0], 1.0*c - 2.0*s);
        assert!((vals[1] - (1.0*s + 2.0*c)).abs() < 0.01, "out[1]={} expected {}", vals[1], 1.0*s + 2.0*c);
    }

    #[test]
    fn test_rope_position_0_is_identity() {
        let dev = crate::backend::default_device(0).expect("device").0;
        let rope = RotaryEmbedding::new(8, 10, 10000.0, &dev, DType::F32).unwrap();
        let q = Tensor::rand(0.0f32, 1.0f32, (1, 2, 1, 8), &dev).unwrap();
        let (q_rot, _) = rope.apply(&q, &q, 0).unwrap();
        // At position 0: cos=1, sin=0 → identity
        let orig: Vec<f32> = q.flatten_all().unwrap().to_vec1().unwrap();
        let rot: Vec<f32> = q_rot.flatten_all().unwrap().to_vec1().unwrap();
        for (a, b) in orig.iter().zip(rot.iter()) {
            assert!((a - b).abs() < 1e-5, "Position 0 should be identity: {} vs {}", a, b);
        }
    }
}
