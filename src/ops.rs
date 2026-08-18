//! Tensor ops missing a CUDA kernel in candle 0.11.
//!
//! candle-core 0.11 has no CUDA implementation of `softmax_last_dim`, so it
//! is rebuilt here from primitive ops that DO have CUDA kernels
//! (max_keepdim / broadcast_sub / exp / sum_keepdim / broadcast_div).

use candle_core::{D, Result, Tensor};

/// Numerically-stable softmax over the last dimension, CUDA-friendly.
/// Equivalent to `candle_nn::ops::softmax_last_dim` but works on GPU.
pub fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    let max = x.max_keepdim(D::Minus1)?;
    let shifted = x.broadcast_sub(&max)?;
    let exp = shifted.exp()?;
    let sum = exp.sum_keepdim(D::Minus1)?;
    exp.broadcast_div(&sum)
}
