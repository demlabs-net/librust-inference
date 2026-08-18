//! GQA Attention with optional QK-norm, causal mask, and KV-cache.
//!
//! Slow AR: 32 Q heads, 8 KV heads (GQA 4:1), QK-norm enabled, head_dim=128
//! Fast AR: 32 Q heads, 8 KV heads (GQA 4:1), QK-norm disabled, head_dim=128

use candle_core::{DType, Device, Module, Result, Tensor};


use crate::nn_candle::linear::QLinear;
use crate::nn_candle::rmsnorm::RmsNorm;
use crate::nn_candle::rope::RotaryEmbedding;

/// Initial capacity (positions) of a KV cache. Caches GROW on demand
/// (doubling) during generation instead of pre-allocating the model's
/// max_seq_len: S2-Pro's slow AR has max_seq_len=32768, and allocating K+V
/// for all 36 layers up front costs ~4.8 GB VRAM PER ENGINE while a typical
/// utterance uses a few hundred tokens.
const KV_CACHE_INIT_CAPACITY: usize = 256;

/// KV-cache for autoregressive generation.
///
/// Allocates a small buffer and grows it (doubling) as generation proceeds —
/// memory tracks the real sequence length, not the theoretical maximum.
#[derive(Debug)]
pub struct KvCache {
    k: Tensor,
    v: Tensor,
    pos: usize,
    capacity: usize,
}

impl KvCache {
    /// Create a KV-cache with the given dtype. `max_seq` is only an upper
    /// bound: the buffer starts small (`KV_CACHE_INIT_CAPACITY` positions)
    /// and grows on demand, so an engine costs ~40 MB idle instead of
    /// max_seq_len worth of VRAM.
    pub fn new(batch: usize, n_kv_heads: usize, max_seq: usize, head_dim: usize, dtype: DType, device: &Device) -> Result<Self> {
        let capacity = max_seq.min(KV_CACHE_INIT_CAPACITY).max(1);
        let k = Tensor::zeros((batch, n_kv_heads, capacity, head_dim), dtype, device)?;
        let v = Tensor::zeros((batch, n_kv_heads, capacity, head_dim), dtype, device)?;
        Ok(Self { k, v, pos: 0, capacity })
    }

    /// Grow the buffer (doubling) until it fits `needed` positions, copying
    /// the used prefix. Rare (logarithmic in sequence length) and cheap for
    /// small caches.
    fn ensure_capacity(&mut self, needed: usize) -> Result<()> {
        if needed <= self.capacity {
            return Ok(());
        }
        let mut new_cap = self.capacity;
        while new_cap < needed {
            new_cap = new_cap.saturating_mul(2);
        }
        let (batch, n_kv_heads, _seq, head_dim) = self.k.dims4()?;
        let k = Tensor::zeros((batch, n_kv_heads, new_cap, head_dim), self.k.dtype(), self.k.device())?;
        let v = Tensor::zeros((batch, n_kv_heads, new_cap, head_dim), self.v.dtype(), self.v.device())?;
        if self.pos > 0 {
            // narrow on a non-last dim yields a non-contiguous view; the
            // slice_set below requires a contiguous source.
            k.slice_set(&self.k.narrow(2, 0, self.pos)?.contiguous()?, 2, 0)?;
            v.slice_set(&self.v.narrow(2, 0, self.pos)?.contiguous()?, 2, 0)?;
        }
        self.k = k;
        self.v = v;
        self.capacity = new_cap;
        Ok(())
    }

    /// Update cache at current position with new k, v.
    pub fn update(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        let seq_len = k.dim(2)?;
        // Grow before writing — never slice_set past the buffer end.
        self.ensure_capacity(self.pos + seq_len)?;
        // Ensure contiguity for slice_set
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        self.k.slice_set(&k, 2, self.pos)?;
        self.v.slice_set(&v, 2, self.pos)?;
        self.pos += seq_len;
        Ok(())
    }

    /// Get all cached k, v up to current position.
    pub fn get(&self) -> Result<(Tensor, Tensor)> {
        let k = self.k.narrow(2, 0, self.pos)?;
        let v = self.v.narrow(2, 0, self.pos)?;
        Ok((k, v))
    }

    /// Reset cache position.
    pub fn reset(&mut self) {
        self.pos = 0;
    }

    pub fn pos(&self) -> usize { self.pos }
}

/// GQA Attention layer (quantizable projections — see [QLinear]).
#[derive(Clone)]
pub struct Attention {
    wqkv: QLinear,
    wo: QLinear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_rep: usize,
}

impl Attention {
    pub fn new(
        wqkv: QLinear, wo: QLinear,
        q_norm: Option<RmsNorm>, k_norm: Option<RmsNorm>,
        n_heads: usize, n_kv_heads: usize, head_dim: usize,
    ) -> Self {
        Self {
            wqkv, wo, q_norm, k_norm,
            n_heads, n_kv_heads, head_dim,
            n_rep: n_heads / n_kv_heads,
        }
    }

    /// Forward pass with KV-cache.
    ///
    /// - `x`: input (batch, seq_len, dim)
    /// - `rope`: rotary embedding
    /// - `kv_cache`: mutable KV-cache
    /// - `start_pos`: position for RoPE offset (0 for prefill)
    pub fn forward(
        &self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        kv_cache: &mut KvCache,
        start_pos: usize,
    ) -> Result<Tensor> {
        let (batch, seq_len, _dim) = x.dims3()?;

        // Fused QKV projection
        let qkv = self.wqkv.forward(x)?;
        let q_size = self.n_heads * self.head_dim;
        let kv_size = self.n_kv_heads * self.head_dim;

        let q = qkv.narrow(2, 0, q_size)?;
        let k = qkv.narrow(2, q_size, kv_size)?;
        let v = qkv.narrow(2, q_size + kv_size, kv_size)?;

        // Reshape: (batch, seq, heads, head_dim) -> (batch, heads, seq, head_dim)
        let q = q.reshape((batch, seq_len, self.n_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((batch, seq_len, self.n_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((batch, seq_len, self.n_kv_heads, self.head_dim))?.transpose(1, 2)?;

        // QK-norm (Slow AR only)
        let q = if let Some(ref qn) = self.q_norm { qn.forward(&q)? } else { q };
        let k = if let Some(ref kn) = self.k_norm { kn.forward(&k)? } else { k };

        // Apply RoPE
        let (q, k) = rope.apply(&q, &k, start_pos)?;

        // Update KV-cache
        kv_cache.update(&k, &v)?;
        let (cached_k, cached_v) = kv_cache.get()?;

        // GQA: expand K, V to match Q head count
        let k = repeat_kv(&cached_k, self.n_rep)?;
        let v = repeat_kv(&cached_v, self.n_rep)?;

        // Scaled dot-product attention
        // scale = 1/sqrt(head_dim) — use f64 operator overload for dtype safety
        let scale = 1.0f64 / (self.head_dim as f64).sqrt();
        let attn = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        // Apply causal mask for prefill (seq_len > 1); the mask covers the
        // FULL key sequence (cache + new keys) — see build_causal_mask.
        let attn = if seq_len > 1 {
            let mask = build_causal_mask(seq_len, start_pos, x.device(), attn.dtype())?;
            // attn shape: (batch, heads, seq_len, cached_len)
            // mask shape: (1, 1, seq_len, cached_len)
            attn.broadcast_add(&mask)?
        } else {
            attn
        };

        let attn = crate::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;

        // Reshape back
        let out = out.transpose(1, 2)?.reshape((batch, seq_len, self.n_heads * self.head_dim))?;

        // Output projection
        self.wo.forward(&out)
    }
}

/// Build a causal mask for the CURRENT prompt over the FULL key sequence
/// (cached keys + new keys). Shape: (1, 1, seq_len, cached_len) where
/// cached_len = start_pos + seq_len. Keys already in the cache (j <
/// start_pos) are past context — fully visible; only the new positions are
/// causally masked: query i sees keys j <= start_pos + i. Built in the
/// attention dtype (BF16) so the broadcast_add does not hit a dtype
/// mismatch.
fn build_causal_mask(seq_len: usize, start_pos: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let cached_len = start_pos + seq_len;
    let mut mask = vec![f32::NEG_INFINITY; seq_len * cached_len];
    for i in 0..seq_len {
        let visible = start_pos + i;
        for j in 0..=visible.min(cached_len - 1) {
            mask[i * cached_len + j] = 0.0;
        }
    }
    let mask = Tensor::new(mask.as_slice(), device)?
        .to_dtype(dtype)?
        .reshape((1, 1, seq_len, cached_len))?;
    Ok(mask)
}

/// Expand K/V heads for GQA: repeat each KV head `n_rep` times.
fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (batch, n_kv, seq, dim) = x.dims4()?;
    let x = x.unsqueeze(2)?.expand((batch, n_kv, n_rep, seq, dim))?;
    x.reshape((batch, n_kv * n_rep, seq, dim))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_new() {
        let dev = crate::backend::default_device(0).expect("device").0;
        let cache = KvCache::new(1, 8, 32768, 128, DType::F32, &dev).unwrap();
        assert_eq!(cache.pos(), 0);
    }

    #[test]
    fn test_kv_cache_update_and_get() {
        let dev = crate::backend::default_device(0).expect("device").0;
        let mut cache = KvCache::new(1, 8, 100, 64, DType::F32, &dev).unwrap();
        let k = Tensor::ones((1, 8, 1, 64), DType::F32, &dev).unwrap();
        let v = Tensor::ones((1, 8, 1, 64), DType::F32, &dev).unwrap();
        cache.update(&k, &v).unwrap();
        assert_eq!(cache.pos(), 1);
        let (ck, _cv) = cache.get().unwrap();
        assert_eq!(ck.shape().dims(), &[1, 8, 1, 64]);
    }

    #[test]
    fn test_kv_cache_reset() {
        let dev = crate::backend::default_device(0).expect("device").0;
        let mut cache = KvCache::new(1, 8, 100, 64, DType::F32, &dev).unwrap();
        let k = Tensor::ones((1, 8, 1, 64), DType::F32, &dev).unwrap();
        let v = Tensor::ones((1, 8, 1, 64), DType::F32, &dev).unwrap();
        cache.update(&k, &v).unwrap();
        cache.reset();
        assert_eq!(cache.pos(), 0);
    }

    #[test]
    fn test_repeat_kv() {
        let dev = crate::backend::default_device(0).expect("device").0;
        let x = Tensor::rand(0.0f32, 1.0f32, (1, 8, 10, 128), &dev).unwrap();
        let expanded = repeat_kv(&x, 4).unwrap();
        assert_eq!(expanded.shape().dims(), &[1, 32, 10, 128]);
    }

    #[test]
    fn test_causal_mask_shape() {
        let dev = crate::backend::default_device(0).expect("device").0;
        let mask = build_causal_mask(5, 0, &dev, DType::F32).unwrap();
        assert_eq!(mask.shape().dims(), &[1, 1, 5, 5]);
        // Upper triangle should be -inf, diagonal and below should be 0
        let vals: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(vals[0], 0.0); // (0,0)
        assert!(vals[1].is_infinite() && vals[1] < 0.0); // (0,1) = -inf
        assert_eq!(vals[6], 0.0); // (1,1) diagonal
    }
}
