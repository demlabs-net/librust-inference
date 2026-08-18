//! ConvNeXtBlock: depthwise + LayerNorm + GELU + layer scale.
//!
//! Used in RVQ downsample/upsample paths.
//! Architecture:
//!   h = DepthwiseConv(k=7, groups=dim)
//!   h = LayerNorm(h)  (in N,L,C layout)
//!   h = Linear(dim, 4*dim)(h)
//!   h = GELU(h)
//!   h = Linear(4*dim, dim)(h)
//!   h = gamma * h  (layer scale)
//!   out = residual + h

use candle_core::{D, Device, Module, Result, Tensor};
use candle_nn::Linear;

use super::causal_conv::CausalConv1d;

pub struct ConvNeXtBlock {
    depthwise: CausalConv1d,
    norm_weight: Tensor,
    norm_bias: Tensor,
    expand: Linear,
    project: Linear,
    gamma: Tensor, // layer scale, init=1e-6
}

impl ConvNeXtBlock {
    pub fn new(
        dim: usize, _device: &Device,
        dw_wv: Tensor, dw_wg: Tensor, dw_bias: Option<Tensor>,
        norm_weight: Tensor, norm_bias: Tensor,
        expand_weight: Tensor, expand_bias: Option<Tensor>,
        project_weight: Tensor, project_bias: Option<Tensor>,
        gamma: Tensor,
    ) -> Self {
        Self {
            depthwise: CausalConv1d::from_loaded(dw_wv, dw_wg, dw_bias, 1, 1, dim, 7),
            norm_weight,
            norm_bias,
            expand: Linear::new(expand_weight, expand_bias),
            project: Linear::new(project_weight, project_bias),
            gamma,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();

        // Depthwise conv (groups=dim)
        let h = self.depthwise.forward(x)?;

        // LayerNorm in (N, L, C) layout — biased variance (divide by N, not N-1)
        let h = h.transpose(1, 2)?; // -> (N, L, C)
        let mean = h.mean_keepdim(D::Minus1)?;
        let diff = h.broadcast_sub(&mean)?;
        let var = diff.sqr()?.mean_keepdim(D::Minus1)?; // FIX: biased variance (N, not N-1)
        let eps = Tensor::new(1e-6f32, h.device())?.to_dtype(h.dtype())?;
        let h = diff.broadcast_div(&(var.broadcast_add(&eps))?.sqrt()?)?;
        let h = h.broadcast_mul(&self.norm_weight)?;
        let h = h.broadcast_add(&self.norm_bias)?;
        let h = h.transpose(1, 2)?; // -> (N, C, L)

        // MLP: expand → GELU → project
        let h = h.transpose(1, 2)?; // -> (N, L, C) for Linear
        let h = self.expand.forward(&h)?;
        // GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
        let h = gelu(&h)?;
        let h = self.project.forward(&h)?;
        let h = h.transpose(1, 2)?; // -> (N, C, L)

        // Layer scale + residual
        let h = self.gamma.broadcast_mul(&h)?;
        residual + h
    }
}

/// GELU activation: x * 0.5 * (1 + erf(x / sqrt(2)))
fn gelu(x: &Tensor) -> Result<Tensor> {
    // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // All constants must match the input dtype (BF16 for model activations).
    let dt = x.dtype();
    let sqrt_2_over_pi = Tensor::new(0.7978845608028654f32, x.device())?.to_dtype(dt)?;
    let coeff = Tensor::new(0.044715f32, x.device())?.to_dtype(dt)?;
    let one = Tensor::new(1.0f32, x.device())?.to_dtype(dt)?;
    let half = Tensor::new(0.5f32, x.device())?.to_dtype(dt)?;

    let x3 = x.sqr()?.broadcast_mul(x)?;
    let inner = x.broadcast_add(&x3.broadcast_mul(&coeff)?)?;
    let inner = inner.broadcast_mul(&sqrt_2_over_pi)?;
    let tanh = inner.tanh()?;
    let tanh_plus1 = tanh.broadcast_add(&one)?;
    let result = x.broadcast_mul(&half)?.broadcast_mul(&tanh_plus1)?;
    Ok(result)
}
