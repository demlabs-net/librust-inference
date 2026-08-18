//! CausalConvNet and CausalTransConvNet with weight normalization.
//!
//! Weight normalization: weight = weight_g * weight_v / ||weight_v||_per_output_channel
//! Causal: left-padding only, so output depends only on past inputs.

use candle_core::{Result, Tensor};

/// Weight-normalized Conv1d with causal padding.
pub struct CausalConv1d {
    weight_v: Tensor,
    weight_g: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    dilation: usize,
    groups: usize,
    padding: usize,
}

impl CausalConv1d {
    pub fn from_loaded(
        weight_v: Tensor, weight_g: Tensor, bias: Option<Tensor>,
        stride: usize, dilation: usize, groups: usize, kernel_size: usize,
    ) -> Self {
        let effective_kernel = (kernel_size - 1) * dilation + 1;
        let padding = effective_kernel.saturating_sub(stride);
        Self { weight_v, weight_g, bias, stride, dilation, groups, padding }
    }

    fn normalized_weight(&self) -> Result<Tensor> {
        let norm = self.weight_v.sqr()?
            .sum_keepdim(candle_core::D::Minus1)?
            .sum_keepdim(1)?
            .sqrt()?;
        let eps = Tensor::new(1e-8f32, norm.device())?.to_dtype(norm.dtype())?;
        let norm = norm.broadcast_add(&eps)?;
        let normalized = self.weight_v.broadcast_div(&norm)?;
        self.weight_g.broadcast_mul(&normalized)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let weight = self.normalized_weight()?;
        let x = if self.padding > 0 {
            x.pad_with_zeros(2, self.padding, 0)?
        } else {
            x.clone()
        };
        let x = x.conv1d(&weight, 0, self.stride, self.dilation, self.groups)?;
        if let Some(ref bias) = self.bias {
            let b = bias.unsqueeze(0)?.unsqueeze(2)?;
            x.broadcast_add(&b)
        } else {
            Ok(x)
        }
    }
}

/// Weight-normalized ConvTranspose1d with causal unpadding.
/// FIX: Now correctly applies bias.
pub struct CausalConvTranspose1d {
    weight_v: Tensor,
    weight_g: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    unpad: usize,
}

impl CausalConvTranspose1d {
    pub fn from_loaded(
        weight_v: Tensor, weight_g: Tensor, bias: Option<Tensor>,
        stride: usize, kernel_size: usize,
    ) -> Self {
        let unpad = kernel_size.saturating_sub(stride);
        Self { weight_v, weight_g, bias, stride, unpad }
    }

    fn normalized_weight(&self) -> Result<Tensor> {
        let norm = self.weight_v.sqr()?
            .sum_keepdim(candle_core::D::Minus1)?
            .sum_keepdim(1)?
            .sqrt()?;
        let eps = Tensor::new(1e-8f32, norm.device())?.to_dtype(norm.dtype())?;
        let norm = norm.broadcast_add(&eps)?;
        let normalized = self.weight_v.broadcast_div(&norm)?;
        self.weight_g.broadcast_mul(&normalized)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let weight = self.normalized_weight()?;
        let mut x = x.conv_transpose1d(&weight, 0, 0, self.stride, 1, 1)?;

        // FIX: Apply bias (was missing before — BUG #12)
        if let Some(ref bias) = self.bias {
            let b = bias.unsqueeze(0)?.unsqueeze(2)?;
            x = x.broadcast_add(&b)?;
        }

        // Remove right padding for causality
        let len = x.dim(2)?;
        if self.unpad > 0 && len > self.unpad {
            x.narrow(2, 0, len - self.unpad)
        } else {
            Ok(x)
        }
    }
}
