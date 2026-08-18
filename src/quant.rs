//! Model quantization: FP32, BF16, FP16, INT8, INT6, INT4, NF4.
//!
//! Canonical home of `QuantizationLevel` — consolidated from
//! fishspeech-core / tts-core so every crate (TTS, STT, codec) shares ONE
//! definition. tts-core re-exports this module (`model::quantize`).
//!
//! ⚠️ HONESTY NOTE: the INT8/INT6/INT4/NF4 paths here are SIMULATED
//! quantization — each weight is quantized and immediately dequantized back
//! to a full-size F16 tensor, so they save NO memory (weights occupy the
//! same VRAM as plain FP16) and no packed tensors/scales are produced.
//! The REAL packed weight type is `nn::linear::QLinear` (Q8_0) used by the
//! TTS S2-Pro path — that one does deliver the memory savings. The
//! simulated levels exist for quality validation (round-trip error
//! measurements) and should not be advertised as memory-saving.
//!
//! Quantization is applied at weight-loading time to keep runtime fast.
//! Activations stay in f16/f32; only weights are quantized.

use candle_core::{DType, Result, Tensor};

/// Supported quantization levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum QuantizationLevel {
    /// Full precision float32 (~22GB VRAM).
    /// Reference mode for quality validation.
    FP32,
    /// Brain floating-point 16-bit (~11GB VRAM).
    /// Native dtype of S2-Pro safetensors. Default.
    BF16,
    /// IEEE 754 half-precision (~11GB VRAM).
    /// Wider GPU support (pre-Ampere).
    FP16,
    /// 8-bit integer dynamic quantization (~6GB VRAM with real packing;
    /// simulated here → weights stay F16, no savings).
    /// Weights quantized to int8, activations in f16.
    INT8,
    /// 6-bit integer weight-only quantization (~4GB VRAM with real packing;
    /// simulated here → weights stay F16, no savings).
    /// 4 values packed in 3 bytes (4×6=24 bits).
    INT6,
    /// 4-bit integer weight-only quantization (~2.5GB VRAM with real
    /// packing; simulated here → weights stay F16, no savings).
    /// Group-wise: 128 elements per group, quantization scales per group.
    INT4,
    /// NormalFloat 4-bit quantization (~2.5GB VRAM with real packing;
    /// simulated here → weights stay F16, no savings).
    /// Optimized for normally-distributed weights.
    NF4,
}

impl QuantizationLevel {
    /// Convert to candle DType for weight loading.
    pub fn to_dtype(self) -> DType {
        match self {
            Self::FP32 => DType::F32,
            Self::BF16 => DType::BF16,
            Self::FP16 | Self::INT8 | Self::INT6 | Self::INT4 | Self::NF4 => DType::F16,
        }
    }

    /// Whether this level requires quantization (beyond dtype conversion).
    pub fn is_quantized(self) -> bool {
        matches!(self, Self::INT8 | Self::INT6 | Self::INT4 | Self::NF4)
    }

    /// Expected VRAM usage in GB (approximate). For the SIMULATED int levels
    /// the weights are stored as F16 — same footprint as FP16.
    pub fn vram_gb(self) -> f32 {
        match self {
            Self::FP32 => 22.0,
            Self::BF16 | Self::FP16 | Self::INT8 | Self::INT6 | Self::INT4 | Self::NF4 => 11.0,
        }
    }

    /// Group size for group-wise quantization.
    pub fn group_size(self) -> usize {
        match self {
            Self::INT4 | Self::NF4 => 128,
            Self::INT6 => 64,
            _ => 1,
        }
    }

    /// Bits per weight element.
    pub fn bits(self) -> u32 {
        match self {
            Self::FP32 => 32,
            Self::BF16 | Self::FP16 => 16,
            Self::INT8 => 8,
            Self::INT6 => 6,
            Self::INT4 | Self::NF4 => 4,
        }
    }

    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            Self::FP32 => "fp32",
            Self::BF16 => "bf16",
            Self::FP16 => "fp16",
            Self::INT8 => "int8",
            Self::INT6 => "int6",
            Self::INT4 => "int4",
            Self::NF4 => "nf4",
        }
    }
}

impl std::str::FromStr for QuantizationLevel {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "fp32" => Ok(Self::FP32),
            "bf16" => Ok(Self::BF16),
            "fp16" => Ok(Self::FP16),
            "int8" => Ok(Self::INT8),
            "int6" => Ok(Self::INT6),
            "int4" => Ok(Self::INT4),
            "nf4" => Ok(Self::NF4),
            other => Err(format!("Unknown quantization: {}", other)),
        }
    }
}

impl Default for QuantizationLevel {
    fn default() -> Self {
        Self::BF16 // Native dtype of S2-Pro
    }
}

/// Quantize a weight tensor according to the quantization level.
///
/// For fp32/bf16/fp16: just cast to the target dtype.
/// For int8/int6/int4/nf4: SIMULATED quantization (see module docs) — the
/// weight is quantized and dequantized back to a full-size F16 tensor, no
/// packed representation or scales are produced.
pub fn quantize_weight(weight: &Tensor, level: QuantizationLevel) -> Result<Tensor> {
    if !level.is_quantized() {
        return weight.to_dtype(level.to_dtype());
    }

    quantize_to_int(weight, level)
}

/// Simulated integer quantization of a Linear weight matrix (see module
/// docs — NOT packed, weights come back as full-size F16).
fn quantize_to_int(weight: &Tensor, level: QuantizationLevel) -> Result<Tensor> {
    let device = weight.device();
    let _dtype = weight.dtype();
    let (out_dim, in_dim) = weight.dims2()?;
    let group_size = level.group_size();
    let _bits = level.bits();

    // Convert to f32 for quantization math
    let w_f32 = weight.to_dtype(DType::F32)?;
    let w_vec: Vec<f32> = w_f32.to_vec2()?.into_iter().flatten().collect();

    match level {
        QuantizationLevel::INT8 => quantize_int8(&w_vec, out_dim, in_dim, device),
        QuantizationLevel::INT4 => quantize_int4(&w_vec, out_dim, in_dim, group_size, device),
        QuantizationLevel::INT6 => quantize_int6(&w_vec, out_dim, in_dim, group_size, device),
        QuantizationLevel::NF4 => quantize_nf4(&w_vec, out_dim, in_dim, group_size, device),
        _ => unreachable!(),
    }
}

/// INT8 dynamic quantization: per-tensor symmetric quantization.
fn quantize_int8(w: &[f32], out_dim: usize, in_dim: usize, device: &candle_core::Device) -> Result<Tensor> {
    let max_val = w.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
    // All-zero weights: scale 0 would divide by zero → NaN weights.
    let scale = if max_val > 0.0 { max_val / 127.0f32 } else { 1.0f32 };

    let quantized: Vec<f32> = w.iter()
        .map(|&x| ((x / scale).round().clamp(-127.0, 127.0) * scale) as f32)
        .collect();

    Tensor::new(quantized.as_slice(), device)?.reshape((out_dim, in_dim))?.to_dtype(DType::F16)
}

/// INT4 group-wise quantization: 128 elements per group, 2 values per byte.
fn quantize_int4(w: &[f32], out_dim: usize, in_dim: usize, group_size: usize, device: &candle_core::Device) -> Result<Tensor> {
    let num_groups = (w.len() + group_size - 1) / group_size;
    let mut result = vec![0.0f32; w.len()];

    for g in 0..num_groups {
        let start = g * group_size;
        let end = (start + group_size).min(w.len());
        let group = &w[start..end];
        let max_val = group.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        // INT4: 16 levels (signed: -7..7, symmetric with scale=max/7 so the
        // dequantized values stay within [-max, max]); guard all-zero groups
        // (scale 0 → NaN).
        let scale = if max_val > 0.0 { max_val / 7.0f32 } else { 1.0f32 };

        for (i, &val) in group.iter().enumerate() {
            let q = (val / scale).round().clamp(-7.0, 7.0);
            result[start + i] = (q * scale) as f32;
        }
    }

    Tensor::new(result.as_slice(), device)?.reshape((out_dim, in_dim))?.to_dtype(DType::F16)
}

/// INT6 group-wise quantization: 64 elements per group.
fn quantize_int6(w: &[f32], out_dim: usize, in_dim: usize, group_size: usize, device: &candle_core::Device) -> Result<Tensor> {
    let num_groups = (w.len() + group_size - 1) / group_size;
    let mut result = vec![0.0f32; w.len()];

    for g in 0..num_groups {
        let start = g * group_size;
        let end = (start + group_size).min(w.len());
        let group = &w[start..end];
        let max_val = group.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        // INT6: 64 levels (signed: -32..31); guard all-zero groups.
        let scale = if max_val > 0.0 { max_val / 31.0f32 } else { 1.0f32 };

        for (i, &val) in group.iter().enumerate() {
            let q = (val / scale).round().clamp(-32.0, 31.0);
            result[start + i] = (q * scale) as f32;
        }
    }

    Tensor::new(result.as_slice(), device)?.reshape((out_dim, in_dim))?.to_dtype(DType::F16)
}

/// NF4 (NormalFloat4) quantization: optimized for normally-distributed weights.
fn quantize_nf4(w: &[f32], out_dim: usize, in_dim: usize, group_size: usize, device: &candle_core::Device) -> Result<Tensor> {
    // NF4 quantization levels (16 values, symmetric around zero)
    // These are the optimal quantization levels for N(0,1) distribution
    let nf4_levels: [f32; 16] = [
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
    ];

    let num_groups = (w.len() + group_size - 1) / group_size;
    let mut result = vec![0.0f32; w.len()];

    for g in 0..num_groups {
        let start = g * group_size;
        let end = (start + group_size).min(w.len());
        let group = &w[start..end];

        // Compute std of the group (assumes zero mean)
        let variance: f32 = group.iter().map(|&x| x * x).sum::<f32>() / group.len() as f32;
        let std = variance.sqrt().max(1e-8);

        for (i, &val) in group.iter().enumerate() {
            // Normalize to N(0,1)
            let normalized = val / std;
            // Find nearest NF4 level
            let mut best_dist = f32::MAX;
            let mut best_level = 0.0f32;
            for &level in &nf4_levels {
                let dist = (normalized - level).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_level = level;
                }
            }
            // Denormalize
            result[start + i] = best_level * std;
        }
    }

    Tensor::new(result.as_slice(), device)?.reshape((out_dim, in_dim))?.to_dtype(DType::F16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_level_default() {
        assert_eq!(QuantizationLevel::default(), QuantizationLevel::BF16);
    }

    #[test]
    fn test_parse_quantization() {
        assert_eq!("bf16".parse::<QuantizationLevel>().unwrap(), QuantizationLevel::BF16);
        assert_eq!("INT8".parse::<QuantizationLevel>().unwrap(), QuantizationLevel::INT8);
        assert_eq!("nf4".parse::<QuantizationLevel>().unwrap(), QuantizationLevel::NF4);
        assert!("xyz".parse::<QuantizationLevel>().is_err());
    }

    #[test]
    fn test_vram_estimation() {
        assert!((QuantizationLevel::BF16.vram_gb() - 11.0).abs() < 0.1);
        // Simulated int levels store weights as F16 — same footprint.
        assert!((QuantizationLevel::INT8.vram_gb() - 11.0).abs() < 0.1);
        assert!((QuantizationLevel::INT4.vram_gb() - 11.0).abs() < 0.1);
    }

    #[test]
    fn test_int8_quantize() {
        let device = crate::backend::default_device(0).expect("device").0;
        let w: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
        let result = quantize_int8(&w, 16, 16, &device).unwrap();
        assert_eq!(result.shape().dims(), &[16, 16]);
    }

    #[test]
    fn test_int4_quantize() {
        let device = crate::backend::default_device(0).expect("device").0;
        let w = vec![0.5f32, -0.3, 0.8, -0.1, 0.2, -0.9, 0.4, -0.6];
        let result = quantize_int4(&w, 2, 4, 128, &device).unwrap();
        assert_eq!(result.shape().dims(), &[2, 4]);
    }

    #[test]
    fn test_int6_quantize() {
        let device = crate::backend::default_device(0).expect("device").0;
        let w = vec![0.5f32, -0.3, 0.8, -0.1, 0.2, -0.9];
        let result = quantize_int6(&w, 2, 3, 64, &device).unwrap();
        assert_eq!(result.shape().dims(), &[2, 3]);
    }

    #[test]
    fn test_nf4_quantize() {
        let device = crate::backend::default_device(0).expect("device").0;
        let w = vec![0.5f32, -0.3, 0.8, -0.1, 0.2, -0.9, 0.4, -0.6];
        let result = quantize_nf4(&w, 2, 4, 128, &device).unwrap();
        assert_eq!(result.shape().dims(), &[2, 4]);
    }
}
