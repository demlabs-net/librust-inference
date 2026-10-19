//! Dtype and quantized-format model (Phase 1.2 of the backend plan).
//!
//! Three independent axes per component:
//! - **activation** — compute dtype of activations (GEMM accumulates in f32);
//! - **weights** — storage format of Linear/GEMM weights, either plain
//!   (f32/bf16/f16) or packed block-quantized (Q8_0, Q6_K, Int4, Nf4) with
//!   dequant inside the fused GEMM kernel;
//! - **kv** — KV-cache storage (f32/f16/q8_0/q4_0), K and V configured
//!   separately at the call site; quantization happens at `kv_update` write,
//!   inline dequant inside the fused attention kernel.
//!
//! [`validate`] rejects incompatible combinations at load time (fail fast —
//! the candle-era lesson "matmul lhs F16 rhs BF16"), never as a kernel crash.
//!
//! The block-quantized weight/KV layouts (Q8_0, Q6_K, q4_0) follow the
//! canonical llama.cpp formats (MIT License, https://github.com/ggml-org/llama.cpp)
//! — reimplemented here, not vendored.

use crate::error::BackendError;

/// Compute dtype of activations. GEMM accumulates in f32 regardless.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    F32,
    BF16,
    F16,
    I8,
}

impl Dtype {
    pub fn bytes(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::BF16 | Dtype::F16 => 2,
            Dtype::I8 => 1,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Dtype::F32 => "f32",
            Dtype::BF16 => "bf16",
            Dtype::F16 => "f16",
            Dtype::I8 => "i8",
        }
    }
}

/// Storage format of Linear/GEMM weights.
///
/// Plain formats store raw values in the given dtype. Block-quantized formats
/// pack weights (Q8_0: 32-elem blocks, int8 + f32 scale; Q6_K: 256-elem
/// super-blocks, 6.5625 bpw, k-quants; Int4/Nf4: group-quantized, 4.5 bpw
/// nominal) and are dequantized inside the fused GEMM kernel — the point is
/// real VRAM/bandwidth savings, not fake-quantized f16 tensors.
///
/// Variant names keep the canonical llama.cpp format names (Q8_0, Q6_K,
/// Q4_0, Q4_K — последний входит в схему Q4_K_M на уровне тензоров).
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightFormat {
    Plain(Dtype),
    Q4_0,
    Q8_0,
    Q4_K,
    Q6_K,
    Q5_K,
    Q5_0,
    /// IQ4_XS: i-квант (A4) — 256-эл. суперблок, 136 Б, 4.25 бита; ниббл —
    /// индекс в нелинейном кодбуке `kvalues_iq4nl` (ggml), плюс 6-битные
    /// масштабы подблоков (ls - 32, знак возможен).
    Iq4Xs,
    /// IQ4_NL: 32-эл. блок, 18 Б (f16 d + 16 байт нибблов), 4.5 бита;
    /// ниббл — индекс в том же нелинейном кодбуке kvalues_iq4nl, что у
    /// IQ4_XS, но без супер-блочных 6-битных масштабов (d на весь блок).
    Iq4Nl,
    /// MXFP4 (OCP): 32-эл. блок, 17 Б (E8M0-скейл + 16 байт нибблов E2M1),
    /// 4.25 бита. Значения — kvalues_fp4 («удвоенные» E2M1), d = 2^(e−128).
    Mxfp4,
    /// NVFP4: 64-эл. блок, 36 Б (4 UE4M3-скейла на подблоки по 16 + 32 Б
    /// нибблов E2M1), 4.5 бита. См. ggml block_nvfp4 / dequantize_row_nvfp4.
    Nvfp4,
    /// Q2_K: 256-эл. суперблок, 84 Б (scales[16] 4-бит scale/min, qs[64]
    /// по 2 бита, d, dmin), 2.625 бита.
    Q2_K,
    /// Q3_K: 256-эл. суперблок, 110 Б (hmask[32], qs[64] по 2 бита,
    /// scales[12] — 6-бит со знаком через −32, d), 3.4375 бита.
    Q3_K,
    /// Q4_1: 32-эл. блок, 20 Б (d, m, 16 Б нибблов), значение d·q + m.
    Q4_1,
    /// Q5_1: 32-эл. блок, 24 Б (d, m, qh[4], 16 Б нибблов), 5-й бит в qh.
    Q5_1,
    /// Q8_1: 32-эл. блок, 40 Б (d, s, int8[32]) — активационный формат.
    Q8_1,
    /// Q8_K: 256-эл. суперблок, 292 Б (f32 d, int8[256], int16 bsums[16]).
    Q8_K,
    /// Q1_0: 128-эл. блок, 18 Б (d + 16 Б битов), значения ±d.
    Q1_0,
    /// Q2_0: 64-эл. блок, 18 Б (d + 16 Б по 2 бита), значения (q−1)·d.
    Q2_0,
    /// TQ1_0: 256-эл. суперблок, 54 Б (qs[48] по 5 троичных в байте через
    /// 3^n, qh[4] по 4, d), 1.6875 бита.
    TQ1_0,
    /// TQ2_0: 256-эл. суперблок, 66 Б (qs[64] по 4 двухбитных, d), 2.0625.
    TQ2_0,
    /// IQ2_XXS: 256-эл. суперблок, 66 Б (d + qs[32] u16), 2.0625 бита;
    /// 2-битные коды — индексы в сетке iq2xxs_grid, знаки — ksigns_iq2xs.
    Iq2Xxs,
    Int4,
    Nf4,
}

impl WeightFormat {
    /// Bits per weight, storage layout (including per-block scales).
    pub fn bits_per_weight(self) -> f32 {
        match self {
            WeightFormat::Plain(d) => (d.bytes() * 8) as f32,
            WeightFormat::Q4_0 => 4.5, // 32 × 4 bit + f16 scale over 32 elems
            WeightFormat::Q8_0 => 8.5, // 32 × 8 bit + 32-bit scale over 32 elems
            // N (iter-9): 144 B × 8 / 256 = 4.5 (the old 4.5625 matched
            // neither the old 148 B nor the canonical 144 B layout).
            WeightFormat::Q4_K => 4.5, // 256-elem super-block: 144 B (canonical ggml)
            WeightFormat::Q6_K => 6.5625, // 256-elem super-block: 210 B (canonical ggml)
            WeightFormat::Q5_K => 5.5, // 256-elem super-block: 176 B (canonical ggml)
            WeightFormat::Q5_0 => 5.5, // 32-elem block: 22 B (canonical ggml)
            WeightFormat::Iq4Xs => 4.25, // 256-elem super-block: 136 B (ggml)
            WeightFormat::Iq4Nl => 4.5, // 32-elem block: 18 B (ggml block_iq4_nl)
            WeightFormat::Mxfp4 => 4.25, // 32-elem block: 17 B (ggml block_mxfp4)
            WeightFormat::Nvfp4 => 4.5, // 64-elem block: 36 B (ggml block_nvfp4)
            WeightFormat::Q2_K => 2.625, // 256-elem super-block: 84 B
            WeightFormat::Q3_K => 3.4375, // 256-elem super-block: 110 B
            WeightFormat::Q4_1 => 5.0, // 32-elem block: 20 B
            WeightFormat::Q5_1 => 6.0, // 32-elem block: 24 B
            WeightFormat::Q8_1 => 9.0, // 32-elem block: 36 B (d+s+int8[32])
            WeightFormat::Q8_K => 9.125, // 256-elem super-block: 292 B
            WeightFormat::Q1_0 => 1.125, // 128-elem block: 18 B
            WeightFormat::Q2_0 => 2.25, // 64-elem block: 18 B
            WeightFormat::TQ1_0 => 1.6875, // 256-elem super-block: 54 B
            WeightFormat::TQ2_0 => 2.0625, // 256-elem super-block: 66 B
            WeightFormat::Iq2Xxs => 2.0625, // 256-elem super-block: 66 B (i-quant)
            WeightFormat::Int4 => 4.5, // group 128: 4 bit + f32 scale/offset
            WeightFormat::Nf4 => 4.5,
        }
    }

    pub fn is_quantized(self) -> bool {
        !matches!(self, WeightFormat::Plain(_))
    }

    /// (элементов в блоке, байт в блоке) для блочно-квантованных форматов
    /// (раскладки ggml); None — формат не блочный (Plain/Int4/Nf4).
    ///
    /// Нужно там, где тензор режется ПО СТРОКАМ без копии (эксперты MoE):
    /// смещение среза считается в блоках, а не в элементах.
    pub fn block(self) -> Option<(usize, usize)> {
        match self {
            WeightFormat::Plain(_) | WeightFormat::Int4 | WeightFormat::Nf4 => None,
            WeightFormat::Q4_0 => Some((32, 18)),
            WeightFormat::Q5_0 => Some((32, 22)),
            WeightFormat::Iq4Nl => Some((32, 18)),
            WeightFormat::Mxfp4 => Some((32, 17)),
            WeightFormat::Nvfp4 => Some((64, 36)),
            WeightFormat::Q2_K => Some((256, 84)),
            WeightFormat::Q3_K => Some((256, 110)),
            WeightFormat::Q4_1 => Some((32, 20)),
            WeightFormat::Q5_1 => Some((32, 24)),
            WeightFormat::Q8_1 => Some((32, 36)),
            WeightFormat::Q8_K => Some((256, 292)),
            WeightFormat::Q1_0 => Some((128, 18)),
            WeightFormat::Q2_0 => Some((64, 18)),
            WeightFormat::TQ1_0 => Some((256, 54)),
            WeightFormat::TQ2_0 => Some((256, 66)),
            WeightFormat::Iq2Xxs => Some((256, 66)),
            WeightFormat::Iq4Xs => Some((256, 136)),
            WeightFormat::Q8_0 => Some((32, 34)),
            WeightFormat::Q4_K => Some((256, 144)),
            WeightFormat::Q5_K => Some((256, 176)),
            WeightFormat::Q6_K => Some((256, 210)),
        }
    }

    /// Байт в строке из `cols` элементов для блочного формата; None — не блочный.
    pub fn block_row_bytes(self, cols: usize) -> Option<usize> {
        self.block().map(|(be, bb)| cols.div_ceil(be) * bb)
    }

    pub fn name(self) -> String {
        match self {
            WeightFormat::Plain(d) => d.name().to_string(),
            WeightFormat::Q4_0 => "q4_0".into(),
            WeightFormat::Q8_0 => "q8_0".into(),
            WeightFormat::Q4_K => "q4_k".into(),
            WeightFormat::Q6_K => "q6_k".into(),
            WeightFormat::Q5_K => "q5_k".into(),
            WeightFormat::Q5_0 => "q5_0".into(),
            WeightFormat::Iq4Xs => "iq4_xs".into(),
            WeightFormat::Iq4Nl => "iq4_nl".into(),
            WeightFormat::Mxfp4 => "mxfp4".into(),
            WeightFormat::Nvfp4 => "nvfp4".into(),
            WeightFormat::Q2_K => "q2_k".into(),
            WeightFormat::Q3_K => "q3_k".into(),
            WeightFormat::Q4_1 => "q4_1".into(),
            WeightFormat::Q5_1 => "q5_1".into(),
            WeightFormat::Q8_1 => "q8_1".into(),
            WeightFormat::Q8_K => "q8_k".into(),
            WeightFormat::Q1_0 => "q1_0".into(),
            WeightFormat::Q2_0 => "q2_0".into(),
            WeightFormat::TQ1_0 => "tq1_0".into(),
            WeightFormat::TQ2_0 => "tq2_0".into(),
            WeightFormat::Iq2Xxs => "iq2_xxs".into(),
            WeightFormat::Int4 => "int4".into(),
            WeightFormat::Nf4 => "nf4".into(),
        }
    }
}

/// KV-cache storage format. K and V may differ; `kv_update` quantizes on
/// write, fused attention dequantizes inline on read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvFormat {
    F32,
    F16,
    Q8_0,
}

impl KvFormat {
    /// Bytes per cached element, storage layout.
    pub fn bytes_per_elem(self) -> f32 {
        match self {
            KvFormat::F32 => 4.0,
            KvFormat::F16 => 2.0,
            // K48: our Q8_0 KV uses an f32 scale per 32-elem block (36
            // bytes/block); the old comment said 34/32 — that would be a
            // llama.cpp-style f16 scale, which this backend does not store.
            KvFormat::Q8_0 => 36.0 / 32.0,
        }
    }
    pub fn is_quantized(self) -> bool {
        matches!(self, KvFormat::Q8_0)
    }
    pub fn name(self) -> &'static str {
        match self {
            KvFormat::F32 => "f32",
            KvFormat::F16 => "f16",
            KvFormat::Q8_0 => "q8_0",
        }
    }
}

/// One component's full quantization configuration (three independent axes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QuantConfig {
    pub activation: Dtype,
    pub weights: WeightFormat,
    pub kv: KvFormat,
}

impl QuantConfig {
    pub fn fp32_reference() -> Self {
        Self {
            activation: Dtype::F32,
            weights: WeightFormat::Plain(Dtype::F32),
            kv: KvFormat::F32,
        }
    }

    /// The user-scenario preset: activations f16, weights and KV in Q8.
    pub fn balanced_f16_q8() -> Self {
        Self {
            activation: Dtype::F16,
            weights: WeightFormat::Q8_0,
            kv: KvFormat::Q8_0,
        }
    }

    /// Max capacity preset (Phase 7.3 candidates; needs quality-gate).
    pub fn max_capacity() -> Self {
        Self {
            activation: Dtype::F16,
            weights: WeightFormat::Int4,
            kv: KvFormat::Q8_0,
        }
    }
}

/// Structural validation of a component configuration — fails fast at load
/// time, never in a kernel.
///
/// Rules (mirror the candle-era dtype lesson):
/// - plain f32 weights: any activation dtype (weights upcast into the gemm);
/// - plain bf16 weights: activations f32 or bf16 (f16 lhs × bf16 rhs breaks);
/// - plain f16 weights: activations f32 or f16;
/// - quantized weights: dequantized to f32 inside the kernel → any activation;
/// - KV formats combine freely with the above (attention dequantizes inline).
pub fn validate(cfg: QuantConfig) -> Result<(), BackendError> {
    let bad = |w: WeightFormat, a: Dtype| {
        BackendError::UnsupportedFormat(format!(
            "weights {}/activation {} combination not supported (GEMM dtype mismatch)",
            w.name(),
            a.name()
        ))
    };
    match cfg.weights {
        WeightFormat::Plain(Dtype::BF16) if cfg.activation == Dtype::F16 => {
            Err(bad(cfg.weights, cfg.activation))
        }
        WeightFormat::Plain(Dtype::F16) if cfg.activation == Dtype::BF16 => {
            Err(bad(cfg.weights, cfg.activation))
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_per_weight_known() {
        assert_eq!(WeightFormat::Plain(Dtype::BF16).bits_per_weight(), 16.0);
        assert_eq!(WeightFormat::Q8_0.bits_per_weight(), 8.5);
        assert!((WeightFormat::Q6_K.bits_per_weight() - 6.5625).abs() < 1e-6);
        assert!((WeightFormat::Q4_0.bits_per_weight() - 4.5).abs() < 1e-6);
        assert!((WeightFormat::Q4_K.bits_per_weight() - 4.5).abs() < 1e-6);
        assert!(WeightFormat::Q8_0.is_quantized());
        assert!(WeightFormat::Q4_0.is_quantized());
        assert!(WeightFormat::Q4_K.is_quantized());
        assert!(!WeightFormat::Plain(Dtype::F16).is_quantized());
    }

    #[test]
    fn kv_bytes_per_elem() {
        assert_eq!(KvFormat::F32.bytes_per_elem(), 4.0);
        assert_eq!(KvFormat::F16.bytes_per_elem(), 2.0);
        // K48: f32 scale per 32-elem block → 36/32 = 1.125.
        assert!((KvFormat::Q8_0.bytes_per_elem() - 1.125).abs() < 1e-6);
        assert!(KvFormat::Q8_0.is_quantized());
    }

    #[test]
    fn validate_ok_combinations() {
        for act in [Dtype::F32, Dtype::BF16, Dtype::F16] {
            validate(QuantConfig {
                activation: act,
                weights: WeightFormat::Q8_0,
                kv: KvFormat::Q8_0,
            })
            .unwrap();
            validate(QuantConfig {
                activation: act,
                weights: WeightFormat::Plain(Dtype::F32),
                kv: KvFormat::F16,
            })
            .unwrap();
        }
        validate(QuantConfig {
            activation: Dtype::BF16,
            weights: WeightFormat::Plain(Dtype::BF16),
            kv: KvFormat::F32,
        })
        .unwrap();
        validate(QuantConfig {
            activation: Dtype::F16,
            weights: WeightFormat::Plain(Dtype::F16),
            kv: KvFormat::Q8_0,
        })
        .unwrap();
    }

    #[test]
    fn validate_rejects_mismatched_plain() {
        assert!(
            validate(QuantConfig {
                activation: Dtype::F16,
                weights: WeightFormat::Plain(Dtype::BF16),
                kv: KvFormat::F32,
            })
            .is_err()
        );
        assert!(
            validate(QuantConfig {
                activation: Dtype::BF16,
                weights: WeightFormat::Plain(Dtype::F16),
                kv: KvFormat::F32,
            })
            .is_err()
        );
    }

    #[test]
    fn presets_validate() {
        validate(QuantConfig::fp32_reference()).unwrap();
        validate(QuantConfig::balanced_f16_q8()).unwrap();
        validate(QuantConfig::max_capacity()).unwrap();
    }
}
