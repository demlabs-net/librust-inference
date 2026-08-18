//! Snake1d activation: y = x + (1/alpha) * sin(alpha * x)^2

use candle_core::{DType, Result, Tensor};

#[derive(Debug, Clone)]
pub struct Snake1d {
    alpha: Tensor,
}

impl Snake1d {
    pub fn from_tensor(alpha: Tensor) -> Self {
        Self { alpha }
    }

    pub fn new(channels: usize, device: &candle_core::Device) -> Result<Self> {
        let alpha = Tensor::ones((1, channels, 1), DType::F32, device)?;
        Ok(Self { alpha })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let ax = self.alpha.broadcast_mul(x)?;
        let sin_ax = ax.sin()?;
        let sin_sq = sin_ax.sqr()?;
        let inv_alpha = self.alpha.recip()?;
        let correction = inv_alpha.broadcast_mul(&sin_sq)?;
        x.broadcast_add(&correction)
    }

    pub fn alpha(&self) -> &Tensor { &self.alpha }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snake1d_output_shape() {
        let dev = candle_core::Device::Cpu;
        let snake = Snake1d::new(64, &dev).unwrap();
        let x = Tensor::zeros((1, 64, 1000), DType::F32, &dev).unwrap();
        let out = snake.forward(&x).unwrap();
        assert_eq!(out.shape().dims(), &[1, 64, 1000]);
    }

    #[test]
    fn test_snake1d_alpha_init() {
        let dev = candle_core::Device::Cpu;
        let snake = Snake1d::new(64, &dev).unwrap();
        let alpha: Vec<f32> = snake.alpha().flatten_all().unwrap().to_vec1().unwrap();
        for a in alpha {
            assert!((a - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_snake1d_zeros() {
        let dev = candle_core::Device::Cpu;
        let snake = Snake1d::new(16, &dev).unwrap();
        let x = Tensor::zeros((1, 16, 100), DType::F32, &dev).unwrap();
        let out = snake.forward(&x).unwrap();
        // Snake(0) = 0 + (1/1) * sin(1*0)^2 = 0
        let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        for v in vals {
            assert!((v - 0.0).abs() < 1e-6);
        }
    }
}
