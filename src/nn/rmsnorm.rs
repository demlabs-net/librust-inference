//! RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
//!
//! Computed in float32 for numerical stability, cast back to input dtype.

use candle_core::{DType, D, Result, Tensor};

/// RMSNorm layer.
#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    /// Forward: x * rsqrt(mean(x^2, dim=-1) + eps) * weight
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let input_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let rms = (variance + self.eps)?.sqrt()?;
        let normed = x_f32.broadcast_div(&rms)?;
        let weight_f32 = self.weight.to_dtype(DType::F32)?;
        let result = normed.broadcast_mul(&weight_f32)?;
        result.to_dtype(input_dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmsnorm_ones() {
        let dev = crate::backend::default_device(0).expect("device").0;
        let weight = Tensor::ones((2560,), DType::F32, &dev).unwrap();
        let rms = RmsNorm::new(weight, 1e-6);
        let x = Tensor::ones((1, 10, 2560), DType::F32, &dev).unwrap();
        let out = rms.forward(&x).unwrap();
        assert_eq!(out.shape().dims(), &[1, 10, 2560]);
    }

    #[test]
    fn test_rmsnorm_f16_cast() {
        let dev = crate::backend::default_device(0).expect("device").0;
        let weight = Tensor::ones((2560,), DType::F32, &dev).unwrap();
        let rms = RmsNorm::new(weight, 1e-6);
        let x = Tensor::ones((1, 5, 2560), DType::F16, &dev).unwrap();
        let out = rms.forward(&x).unwrap();
        assert_eq!(out.dtype(), DType::F16);
        assert_eq!(out.shape().dims(), &[1, 5, 2560]);
    }

    #[test]
    fn test_rmsnorm_zero_input() {
        let dev = crate::backend::default_device(0).expect("device").0;
        let weight = Tensor::ones((2560,), DType::F32, &dev).unwrap();
        let rms = RmsNorm::new(weight, 1e-6);
        let x = Tensor::zeros((1, 3, 2560), DType::F32, &dev).unwrap();
        let out = rms.forward(&x).unwrap();
        // Should be all zeros
        let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        for v in vals { assert!((v - 0.0).abs() < 1e-6); }
    }
}
