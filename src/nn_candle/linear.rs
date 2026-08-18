//! Quantized linear layer.
//!
//! `QLinear` wraps either a Q8_0-quantized `QMatMul` (weights stored as
//! int8 blocks of 32 elements with a per-block scale — the quant scales) or
//! a plain `candle_nn::Linear`. Which path is used is decided at BUILD time
//! by the caller (e.g. from the `TTS_QUANTIZATION` env level), so switching
//! between int8 / fp16 / fp32 requires no code changes — just an env var.
//!
//! Q8_0 quantization keeps the weights at ~8 bits (4× less VRAM than fp16)
//! at near-fp16 quality; activations stay in F16.
//!
//! The internals are `Arc`'d so whole models can be cloned cheaply and
//! SHARED across pool engines: cloning a `QLinear` reuses the same weight
//! tensors in VRAM (only per-engine state like KV caches is duplicated).

use std::sync::Arc;

use candle_core::quantized::{QMatMul, QTensor};
use candle_core::quantized::GgmlDType;
use candle_core::{Module, Result, Tensor};
use candle_nn::Linear;

/// Linear layer: quantized (Q8_0) or plain. Cheap to clone — the weights are
/// shared (Arc), so N pool engines built from one model reuse the same VRAM.
#[derive(Clone)]
pub enum QLinear {
    Quantized(Arc<QMatMul>, Option<Arc<Tensor>>),
    Plain(Arc<Linear>),
}

impl QLinear {
    /// Build a linear layer. When `quantize` is true the weight is converted
    /// to Q8_0 (per-block 32 scale) and used through the quantized matmul
    /// kernel; otherwise the weight is used as-is (dtype decided upstream,
    /// e.g. bf16/fp32 from the quantization level).
    ///
    /// Memory: QTensor::quantize internally casts to F32 — do NOT pre-cast
    /// (an F16/F32 copy on top of the loaded weights would double their VRAM
    /// during load; casting one tensor at a time keeps the peak at
    /// loaded-weights + one transient copy).
    pub fn new(weight: Tensor, bias: Option<Tensor>, quantize: bool) -> Result<Self> {
        if quantize {
            let qt = QTensor::quantize(&weight, GgmlDType::Q8_0)?;
            Ok(Self::Quantized(Arc::new(QMatMul::from_qtensor(qt)?), bias.map(Arc::new)))
        } else {
            Ok(Self::Plain(Arc::new(Linear::new(weight, bias))))
        }
    }

    /// Wrap an existing plain linear (e.g. from shared helpers) — non-quantized.
    pub fn plain(linear: Linear) -> Self {
        Self::Plain(Arc::new(linear))
    }
}

impl Module for QLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Quantized(q, bias) => {
                let y = q.forward(x)?;
                match bias {
                    Some(b) => y.broadcast_add(b),
                    None => Ok(y),
                }
            }
            Self::Plain(l) => l.forward(x),
        }
    }
}
