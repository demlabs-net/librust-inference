//! ResidualUnit, EncoderBlock, DecoderBlock.
//!
//! ResidualUnit: Snake1d → WNConv1d(k=7,dil=d) → Snake1d → WNConv1d(k=1) → residual
//! EncoderBlock: ResidualUnit×3 → Snake1d → WNConv1d(stride)
//! DecoderBlock: Snake1d → WNConvTranspose1d(stride) → ResidualUnit×3

use candle_core::{Result, Tensor};

use super::snake::Snake1d;
use super::causal_conv::{CausalConv1d, CausalConvTranspose1d};

/// ResidualUnit: Snake1d → WNConv1d(k=7,dil=d) → Snake1d → WNConv1d(k=1) → residual
pub struct ResidualUnit {
    snake1: Snake1d,
    conv1: CausalConv1d,  // k=7, dilation=d
    snake2: Snake1d,
    conv2: CausalConv1d,  // k=1, dilation=1
}

impl ResidualUnit {
    pub fn new(
        _channels: usize, dilation: usize, _device: &candle_core::Device,
        // Conv1 weights (k=7, dilation=d)
        wv1: Tensor, wg1: Tensor, b1: Option<Tensor>, a1: Tensor,
        // Conv2 weights (k=1, dilation=1)
        wv2: Tensor, wg2: Tensor, b2: Option<Tensor>, a2: Tensor,
    ) -> Self {
        Self {
            snake1: Snake1d::from_tensor(a1),
            // Full convolution (groups=1): the stored kernel is [C, C, 7].
            conv1: CausalConv1d::from_loaded(wv1, wg1, b1, 1, dilation, 1, 7),
            snake2: Snake1d::from_tensor(a2),
            conv2: CausalConv1d::from_loaded(wv2, wg2, b2, 1, 1, 1, 1),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.snake1.forward(x)?;
        let y = self.conv1.forward(&y)?;
        let y = self.snake2.forward(&y)?;
        let y = self.conv2.forward(&y)?;
        x + y
    }
}

/// EncoderBlock: ResidualUnit×3 → Snake1d → WNConv1d(stride)
pub struct EncoderBlock {
    ru1: ResidualUnit,
    ru2: ResidualUnit,
    ru3: ResidualUnit,
    snake: Snake1d,
    downsample: CausalConv1d,
}

impl EncoderBlock {
    pub fn new(
        ru1: ResidualUnit, ru2: ResidualUnit, ru3: ResidualUnit,
        snake_alpha: Tensor,
        ds_wv: Tensor, ds_wg: Tensor, ds_bias: Option<Tensor>,
        _in_ch: usize, _out_ch: usize, stride: usize,
    ) -> Self {
        Self {
            ru1, ru2, ru3,
            snake: Snake1d::from_tensor(snake_alpha),
            downsample: CausalConv1d::from_loaded(ds_wv, ds_wg, ds_bias, stride, 1, 1, 2 * stride),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.ru1.forward(x)?;
        let x = self.ru2.forward(&x)?;
        let x = self.ru3.forward(&x)?;
        let x = self.snake.forward(&x)?;
        self.downsample.forward(&x)
    }
}

/// DecoderBlock: Snake1d → WNConvTranspose1d(stride) → ResidualUnit×3
pub struct DecoderBlock {
    snake: Snake1d,
    upsample: CausalConvTranspose1d,
    ru1: ResidualUnit,
    ru2: ResidualUnit,
    ru3: ResidualUnit,
}

impl DecoderBlock {
    pub fn new(
        snake_alpha: Tensor,
        up_wv: Tensor, up_wg: Tensor, up_bias: Option<Tensor>,
        stride: usize, kernel_size: usize,
        ru1: ResidualUnit, ru2: ResidualUnit, ru3: ResidualUnit,
    ) -> Self {
        Self {
            snake: Snake1d::from_tensor(snake_alpha),
            upsample: CausalConvTranspose1d::from_loaded(up_wv, up_wg, up_bias, stride, kernel_size),
            ru1, ru2, ru3,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.snake.forward(x)?;
        let x = self.upsample.forward(&x)?;
        let x = self.ru1.forward(&x)?;
        let x = self.ru2.forward(&x)?;
        self.ru3.forward(&x)
    }
}
