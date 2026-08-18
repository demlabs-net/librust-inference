//! SwiGLU Feed-Forward Network.
//!
//! output = w2(silu(w1(x)) * w3(x))
//! w1: gate projection (dim -> intermediate)
//! w3: up projection   (dim -> intermediate)
//! w2: down projection (intermediate -> dim)

use candle_core::{Module, Result, Tensor};

use crate::nn::linear::QLinear;

/// SwiGLU FFN layer (quantizable — see [QLinear]).
#[derive(Clone)]
pub struct SwiGlu {
    w1: QLinear,  // gate
    w3: QLinear,  // up
    w2: QLinear,  // down
}

impl SwiGlu {
    pub fn new(w1: QLinear, w3: QLinear, w2: QLinear) -> Self {
        Self { w1, w3, w2 }
    }

    /// Forward: w2(silu(w1(x)) * w3(x))
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.w1.forward(x)?;
        let up = self.w3.forward(x)?;
        // SiLU activation: x * sigmoid(x)
        let gate = silu(&gate)?;
        let intermediate = gate.broadcast_mul(&up)?;
        self.w2.forward(&intermediate)
    }
}

/// SiLU (Swish) activation: x * sigmoid(x)
/// sigmoid(x) = 1 / (1 + exp(-x))
/// Manual impl: candle 0.11 has no CUDA sigmoid kernel.
fn silu(x: &Tensor) -> Result<Tensor> {
    let neg_x = x.neg()?;
    let exp_neg = neg_x.exp()?;
    // The constant must match the input dtype (BF16 for all model activations).
    let one = Tensor::new(1.0f32, x.device())?.to_dtype(x.dtype())?;
    let denom = one.broadcast_add(&exp_neg)?;
    let sigmoid = one.broadcast_div(&denom)?;
    x.broadcast_mul(&sigmoid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use candle_nn::Linear;

    #[test]
    fn test_swiglu_shape() {
        let dev = crate::backend::default_device(0).expect("device").0;
        let w1 = Linear::new(Tensor::rand(0.0f32, 0.02f32, (9728, 2560), &dev).unwrap(), None);
        let w3 = Linear::new(Tensor::rand(0.0f32, 0.02f32, (9728, 2560), &dev).unwrap(), None);
        let w2 = Linear::new(Tensor::rand(0.0f32, 0.02f32, (2560, 9728), &dev).unwrap(), None);
        let swiglu = SwiGlu::new(QLinear::plain(w1), QLinear::plain(w3), QLinear::plain(w2));
        let x = Tensor::ones((1, 1, 2560), DType::F32, &dev).unwrap();
        let out = swiglu.forward(&x).unwrap();
        assert_eq!(out.shape().dims(), &[1, 1, 2560]);
    }
}
