//! Квантование и конверсия моделей (Ф4.12, Ф6.3): safetensors → GGUF и
//! GGUF → GGUF (Q4_0/Q5_0/Q8_0/Q4_K_M/Q5_K/Q6_K/BF16/F16/F32).
//!
//! Раскладки блоков — канонический ggml (дешифраторы совместимы с реальными
//! llama.cpp-файлами):
//! Q4_0 {f16 d, 16×u8 нибблы} = 18 Б/32; Q5_0 {f16 d, u8 qh[4], u8 qs[16]} =
//! 22 Б/32; Q8_0 {f16 d, 32×i8} = 34 Б/32;
//! Q4_K {f16 d, f16 dmin, u8 scales[12], u8 qs[128]} = 144 Б/256;
//! Q5_K {f16 d, f16 dmin, u8 scales[12], u8 qh[32], u8 qs[128]} = 176 Б/256
//! (упаковка scales — get_scale_min_k4: y = d·sc·q − dmin·m);
//! Q6_K {u8 ql[128], u8 qh[64], i8 scales[16], f16 d} = 210 Б/256
//! (d в КОНЦЕ блока, байты 208..210; y = d·sc·(q − 32)).
//!
//! Энкодеры — построчные порты reference-реализаций ggml (`_ref`,
//! llama.cpp 5266f24da = сборка локального llama-quantize; сверено с master
//! 2026-10). Совпадение с `llama-quantize` **побитовое** на реальных моделях
//! (Qwen3-0.6B во всех шести форматах, Qwen3.5-4B Q4_K_M), поэтому порядок
//! f32-операций, реципрок вместо деления и FMA-контракция clang
//! (`sum += w*x` → `mul_add`) менять НЕЛЬЗЯ — это ломает паритет.
//!
//! `quantize_gguf` дополнительно применяет политику типов llama.cpp
//! (`llama_tensor_get_type_impl`): 1-D тензоры/нормы остаются как есть,
//! выход и связанный эмбеддинг — Q6_K/Q8_0, ffn_down/attn_v «важных» слоёв —
//! Q6_K, служебные (ssm_conv1d, *_norm и т.д.) не квантуются.

use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::error::BackendError;
use crate::gguf::{GgmlType, GgufValue, parse};
use crate::weights::{
    read_safetensors_tensor_f32, safetensors_tensor_names, safetensors_tensor_numel,
};
use crate::weights_map::dequant_gguf;

/// Форматы вывода конверсии (имена — как в llama.cpp).
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormat {
    F32,
    F16,
    Bf16,
    Q4_0,
    Q8_0,
    /// Q4_K-схема (Q4_K_M в llama.cpp-терминах).
    Q4_K_M,
    Q6_K,
    Q5_0,
    /// Q5_K-схема (Q5_K_M в llama.cpp-терминах).
    Q5_K,
    /// IQ4_XS (ggml type 23): супер-блок 256 весов = 136 Б (4.25 bpw),
    /// 6-битные под-блоки scale по 32. Пишется БЕЗ imatrix (в отличие от
    /// IQ1/IQ2/IQ3_XXS — `tensor_requires_imatrix` в llama-quant.cpp его не
    /// требует); с imatrix был бы точнее — это опция v0.2 (Ф4.12).
    Iq4Xs,
}

impl QuantFormat {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_uppercase().as_str() {
            "F32" => Some(Self::F32),
            "F16" => Some(Self::F16),
            "BF16" => Some(Self::Bf16),
            "Q4_0" => Some(Self::Q4_0),
            "Q8_0" => Some(Self::Q8_0),
            "Q4_K_M" | "Q4_K" => Some(Self::Q4_K_M),
            "Q6_K" => Some(Self::Q6_K),
            "Q5_0" => Some(Self::Q5_0),
            "Q5_K" | "Q5_K_M" => Some(Self::Q5_K),
            "IQ4_XS" => Some(Self::Iq4Xs),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::Q4_0 => "Q4_0",
            Self::Q8_0 => "Q8_0",
            Self::Q4_K_M => "Q4_K_M",
            Self::Q6_K => "Q6_K",
            Self::Q5_0 => "Q5_0",
            Self::Q5_K => "Q5_K",
            Self::Iq4Xs => "IQ4_XS",
        }
    }

    fn ggml_type(self) -> GgmlType {
        match self {
            Self::F32 => GgmlType::F32,
            Self::F16 => GgmlType::F16,
            Self::Bf16 => GgmlType::Bf16,
            Self::Q4_0 => GgmlType::Q4_0,
            Self::Q8_0 => GgmlType::Q8_0,
            Self::Q4_K_M => GgmlType::Q4K,
            Self::Q6_K => GgmlType::Q6K,
            Self::Q5_0 => GgmlType::Q5_0,
            Self::Q5_K => GgmlType::Q5K,
            Self::Iq4Xs => GgmlType::Iq4Xs,
        }
    }

    /// Блокировать ли данные (для конверсии: только квантованные форматы
    /// требуют кратности блоку — F32/F16/BF16 — нет).
    fn needs_block_pad(self) -> bool {
        matches!(
            self,
            Self::Q4_0
                | Self::Q8_0
                | Self::Q4_K_M
                | Self::Q6_K
                | Self::Q5_0
                | Self::Q5_K
                | Self::Iq4Xs
        )
    }

    /// Бит на вес с учётом служебных полей блока (для оценки размера в UI).
    pub fn bits_per_weight(self) -> f32 {
        match self {
            Self::F32 => 32.0,
            Self::F16 | Self::Bf16 => 16.0,
            Self::Q4_0 | Self::Q4_K_M => 4.5,
            Self::Q5_0 | Self::Q5_K => 5.5,
            Self::Q6_K => 6.5625,
            Self::Q8_0 => 8.5,
            Self::Iq4Xs => 4.25,
        }
    }

    /// Формат по ggml-типу (обратный к `ggml_type`); `None` — тип, который
    /// движок писать не умеет (IQ4_NL/IQ1–IQ3 — чужие форматы; IQ4_XS умеет).
    pub fn from_ggml_type(t: GgmlType) -> Option<Self> {
        Some(match t {
            GgmlType::F32 => Self::F32,
            GgmlType::F16 => Self::F16,
            GgmlType::Bf16 => Self::Bf16,
            GgmlType::Q4_0 => Self::Q4_0,
            GgmlType::Q8_0 => Self::Q8_0,
            GgmlType::Q4K => Self::Q4_K_M,
            GgmlType::Q6K => Self::Q6_K,
            GgmlType::Q5_0 => Self::Q5_0,
            GgmlType::Q5K => Self::Q5_K,
            GgmlType::Iq4Xs => Self::Iq4Xs,
            _ => return None,
        })
    }
}

impl std::fmt::Display for QuantFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ── битовые конвертеры ──────────────────────────────────────────────────────

/// f32 → f16 с округлением к ближайшему чётному — ровно как
/// `GGML_FP32_TO_FP16` в ggml (RNE; переполнение → inf, субнормали).
/// Раньше здесь было УСЕЧЕНИЕ мантиссы: из-за него ни один блок квантов не
/// совпадал с llama-quantize (d округлялся вниз), а F16-конверсия отличалась
/// от эталона на 1 ulp.
pub fn f16_bits(x: f32) -> [u8; 2] {
    half::f16::from_f32(x).to_bits().to_le_bytes()
}

/// f16 → f32 (обратно, для чтения уже записанных блоков).
pub fn f16_bits_to_f32(b: [u8; 2]) -> f32 {
    half::f16::from_bits(u16::from_le_bytes(b)).to_f32()
}

pub fn bf16_bits(x: f32) -> [u8; 2] {
    // round-to-nearest-even на 16 бит
    let b = x.to_bits();
    let rounding = ((b >> 16) & 1) + 0x7FFF;
    let bytes = ((b + rounding) >> 16).to_le_bytes();
    [bytes[0], bytes[1]]
}

// ── квант-энкодеры (f32 → байты формата) ───────────────────────────────────
//
// Все четыре энкодера — построчный порт reference-реализаций ggml
// (`quantize_row_q*_ref` в ggml-quants.c, коммит 5266f24da; сверено с master
// 2026-10: функции побитово те же). Байты совпадают с `llama-quantize`
// на同等 входе — это и есть критерий приёмки. Порядок арифметики (f32,
// реципрок вместо деления, RNE через nearest_int/магическую константу)
// менять нельзя: любое «упрощение» ломает побитовое равенство.

/// `nearest_int` из ggml-quants.c: округление к ближайшему чётному (RNE)
/// через магическую константу 1.5·2²³. Точная копия — от неё зависит
/// побитовое совпадение K-квантов и Q6_K.
#[inline]
fn nearest_int(fval: f32) -> i32 {
    debug_assert!(fval.abs() <= 4194303.0, "nearest_int: |{fval}| > 2^22-1");
    let val = fval + 12582912.0f32; // 1.5 * 2^23
    let i = val.to_bits() as i32;
    (i & 0x007f_ffff) - 0x0040_0000
}

/// Групповой порог «все нули» из ggml (GROUP_MAX_EPS).
const GROUP_MAX_EPS: f32 = 1e-15;

/// Q4_0 canonical ggml layout: блок 32 → f16 d + 16 байт нибблов;
/// y = (q − 8)·d. Каноническая парность НИББЛОВ (ggml dequantize_row_
/// q4_0): байт j держит веса j и j+16 — low ниббл = w[j], high = w[j+16].
/// `d = max/−8` — ЗНАКОВЫЙ максимум (d бывает отрицательным, тогда q(max) = 0);
/// квантование — trunc(x·(1/d) + 8.5) с клампом 15.
pub fn quant_q4_0(weights: &[f32]) -> Vec<u8> {
    assert!(
        weights.len().is_multiple_of(32),
        "Q4_0: len % 32 == 0, got {}",
        weights.len()
    );
    let mut out = Vec::with_capacity(weights.len() / 32 * 18);
    for blk in weights.chunks_exact(32) {
        let mut amax = 0f32;
        let mut max = 0f32;
        for &v in blk {
            let av = v.abs();
            if amax < av {
                amax = av;
                max = v;
            }
        }
        let d = max / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f16_bits(d));
        for j in 0..16 {
            // C: MIN(15, (int8_t)(x*id + 8.5f)) — усечение к нулю.
            let x0 = blk[j] * id + 8.5;
            let x1 = blk[j + 16] * id + 8.5;
            let xi0 = (x0 as i32).min(15) as u8;
            let xi1 = (x1 as i32).min(15) as u8;
            out.push((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
        }
    }
    out
}

/// Q8_0: блок 32 → f16 d + 32 i8; y = q·d.
/// `d = amax/127`, `qs = roundf(x·(1/d))` (roundf — половина ОТ нуля).
pub fn quant_q8_0(weights: &[f32]) -> Vec<u8> {
    assert!(weights.len().is_multiple_of(32), "Q8_0: len % 32 == 0");
    let mut out = Vec::with_capacity(weights.len() / 32 * 34);
    for blk in weights.chunks_exact(32) {
        let amax = blk.iter().fold(0.0f32, |m, w| m.max(w.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f16_bits(d));
        for w in blk {
            out.push((w * id).round() as i8 as u8);
        }
    }
    out
}

/// Q5_0 (ggml block_q5_0): 22 байта = f16 d + qh[4] (5-е биты) + qs[16]
/// (нибблы, парность j / j+16). y = ((q & 0xF | 5-й бит << 4) − 16)·d;
/// d = max/−16 (знаковый максимум), квант — trunc(x·(1/d) + 16.5), кламп 31.
pub fn quant_q5_0(weights: &[f32]) -> Vec<u8> {
    assert!(weights.len().is_multiple_of(32), "Q5_0: len % 32 == 0");
    let mut out = Vec::with_capacity(weights.len() / 32 * 22);
    for blk in weights.chunks_exact(32) {
        let mut amax = 0f32;
        let mut max = 0f32;
        for &v in blk {
            let av = v.abs();
            if amax < av {
                amax = av;
                max = v;
            }
        }
        let d = max / -16.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let mut qh = 0u32;
        let mut qs = [0u8; 16];
        for j in 0..16 {
            let xi0 = ((blk[j] * id + 16.5) as i32).min(31) as u8;
            let xi1 = ((blk[j + 16] * id + 16.5) as i32).min(31) as u8;
            qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
            qh |= ((xi0 & 0x10) as u32 >> 4) << j;
            qh |= ((xi1 & 0x10) as u32 >> 4) << (j + 16);
        }
        out.extend_from_slice(&f16_bits(d));
        out.extend_from_slice(&qh.to_le_bytes());
        out.extend_from_slice(&qs);
    }
    out
}

/// `get_scale_min_k4` из ggml (распаковка пары scale/min подблока j).
#[inline]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// `make_qx_quants` из ggml (rmse_type = 1 для Q6_K: вес = x²).
/// Итеративное уточнение scale по ±9 сдвигам — порт один-в-один.
fn make_qx_quants(n: usize, nmax: i32, x: &[f32], l: &mut [i8], rmse_type: i32) -> f32 {
    let mut max = 0f32;
    let mut amax = 0f32;
    for &v in x.iter().take(n) {
        let ax = v.abs();
        if ax > amax {
            amax = ax;
            max = v;
        }
    }
    if amax < GROUP_MAX_EPS {
        for v in l.iter_mut().take(n) {
            *v = 0;
        }
        return 0.0;
    }
    let mut iscale = -(nmax as f32) / max;
    let weight = |i: usize, rmse: i32| -> f32 {
        match rmse {
            1 => x[i] * x[i],
            2 => 1.0,
            3 => x[i].abs(),
            _ => x[i].abs().sqrt(),
        }
    };
    if rmse_type == 0 {
        for i in 0..n {
            let li = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
            l[i] = (nmax + li) as i8;
        }
        return 1.0 / iscale;
    }
    let mut rmse_type = rmse_type;
    let return_early = rmse_type < 0;
    if return_early {
        rmse_type = -rmse_type;
    }
    let mut sumlx = 0f32;
    let mut suml2 = 0f32;
    for i in 0..n {
        let li = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
        l[i] = (nmax + li) as i8;
        let w = weight(i, rmse_type);
        // FMA: clang контрактит `sum += w*x*l` в fma((w*x), l, sum) при
        // -ffp-contract=on (умолчание для C). В knife-edge случаях это
        // решает, какой `is` победит в петле ниже — без FMA байты К-квантов
        // расходятся с llama-quantize (~0.06% байт).
        sumlx = (w * x[i]).mul_add(li as f32, sumlx);
        suml2 = (w * li as f32).mul_add(li as f32, suml2);
    }
    let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
    if return_early {
        return if suml2 > 0.0 {
            0.5 * (scale + 1.0 / iscale)
        } else {
            1.0 / iscale
        };
    }
    let mut best = scale * sumlx;
    for is in -9..=9 {
        if is == 0 {
            continue;
        }
        iscale = -(nmax as f32 + 0.1 * is as f32) / max;
        sumlx = 0.0;
        suml2 = 0.0;
        for i in 0..n {
            let li = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
            let w = weight(i, rmse_type);
            sumlx = (w * x[i]).mul_add(li as f32, sumlx);
            suml2 = (w * li as f32).mul_add(li as f32, suml2);
        }
        if suml2 > 0.0 && sumlx * sumlx > best * suml2 {
            for i in 0..n {
                let li = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
                l[i] = (nmax + li) as i8;
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    scale
}

/// `make_qkx2_quants` из ggml (Q4_K): итеративное уточнение scale+min
/// (nstep шагов по rmin..rmin+rdelta·nstep). Возвращает scale, пишет
/// `l` (0..nmax) и `the_min` (= −min, как хранит формат).
#[allow(clippy::too_many_arguments)]
fn make_qkx2_quants(
    n: usize,
    nmax: i32,
    x: &[f32],
    weights: &[f32],
    l: &mut [u8],
    the_min: &mut f32,
    laux: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    use_mad: bool,
) -> f32 {
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
        let w = weights[i];
        sum_w += w;
        sum_x = w.mul_add(x[i], sum_x); // C: sum_x += w*x[i] (FMA clang)
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        for v in l.iter_mut().take(n) {
            *v = 0;
        }
        *the_min = -min;
        return 0.0;
    }
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_error = 0f32;
    for i in 0..n {
        let li = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
        l[i] = li as u8;
        let mut diff = scale.mul_add(li as f32, min) - x[i]; // C: scale*l + min - x[i]
        diff = if use_mad { diff.abs() } else { diff * diff };
        best_error = weights[i].mul_add(diff, best_error);
    }
    if nstep < 1 {
        *the_min = -min;
        return scale;
    }
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let mut sum_l = 0f32;
        let mut sum_l2 = 0f32;
        let mut sum_xl = 0f32;
        for i in 0..n {
            let li = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
            laux[i] = li as u8;
            let w = weights[i];
            // FMA как в C-бинарнике (clang): sum += w*l, w*l*l, w*l*x.
            sum_l = w.mul_add(li as f32, sum_l);
            sum_l2 = (w * li as f32).mul_add(li as f32, sum_l2);
            sum_xl = (w * li as f32).mul_add(x[i], sum_xl);
        }
        // FMA: clang контрактит `sum_w*sum_l2 - sum_l*sum_l` в
        // fma(sum_w, sum_l2, -(sum_l*sum_l)) — без этого 6-битные
        // scale/min К-квантов расходятся с llama-quantize.
        let d = sum_w.mul_add(sum_l2, -(sum_l * sum_l));
        if d > 0.0 {
            // FMA в числителях (clang: fma(a, b, -(c*d))) — как D выше.
            let mut this_scale = sum_w.mul_add(sum_xl, -(sum_x * sum_l)) / d;
            let mut this_min = sum_l2.mul_add(sum_x, -(sum_l * sum_xl)) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut cur_error = 0f32;
            for i in 0..n {
                let mut diff = this_scale.mul_add(laux[i] as f32, this_min) - x[i];
                diff = if use_mad { diff.abs() } else { diff * diff };
                cur_error = weights[i].mul_add(diff, cur_error);
            }
            if cur_error < best_error {
                l[..n].copy_from_slice(&laux[..n]);
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    scale
}

/// Канонический Q4_K (ggml block_q4_K): блок 256 → 144 байта:
///   d (f16, 0..2), dmin (f16, 2..4), scales[12] (4..16 — 6-битные
///   scale/min 8 подблоков по 32, упаковка get_scale_min_k4), qs[128]
///   (16..144 — 4-битные значения, чанк-мажор: байт 32c+l держит
///   {low: 64c+l, high: 64c+32+l}, 4 чанка по 64).
/// Деквант: y = d·sc·q − dmin·m. Энкодер — порт quantize_row_q4_K_ref:
/// подблоковые scale/min ищутся make_qkx2_quants с весами
/// (av_x + |x|), затем подблоки доквантовываются по f16-округлённым d/dmin.
pub fn quant_q4_k(weights: &[f32]) -> Vec<u8> {
    assert!(weights.len().is_multiple_of(256), "Q4_K: len % 256 == 0");
    let mut out = Vec::with_capacity(weights.len() / 256 * 144);
    let mut l = [0u8; 256];
    let mut laux = [0u8; 32];
    let mut weights_sub = [0f32; 32];
    for blk in weights.chunks_exact(256) {
        let mut mins = [0f32; 8];
        let mut scales = [0f32; 8];
        let mut max_scale = 0f32;
        let mut max_min = 0f32;
        for j in 0..8 {
            let sub = &blk[32 * j..32 * j + 32];
            // FMA-аккумуляция как в C: sum_x2 += x*x.
            let mut sum_x2 = 0f32;
            for &v in sub.iter() {
                sum_x2 = v.mul_add(v, sum_x2);
            }
            let av_x = (sum_x2 / 32.0).sqrt();
            for (l_i, &v) in weights_sub.iter_mut().zip(sub) {
                *l_i = av_x + v.abs();
            }
            scales[j] = make_qkx2_quants(
                32,
                15,
                sub,
                &weights_sub,
                &mut l[32 * j..32 * j + 32],
                &mut mins[j],
                &mut laux,
                -1.0,
                0.1,
                20,
                false,
            );
            if scales[j] > max_scale {
                max_scale = scales[j];
            }
            if mins[j] > max_min {
                max_min = mins[j];
            }
        }
        let inv_scale = if max_scale > 0.0 {
            63.0 / max_scale
        } else {
            0.0
        };
        let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
        let mut scales_packed = [0u8; 12];
        for j in 0..8 {
            let ls = nearest_int(inv_scale * scales[j]).min(63) as u8;
            let lm = nearest_int(inv_min * mins[j]).min(63) as u8;
            if j < 4 {
                scales_packed[j] = ls;
                scales_packed[j + 4] = lm;
            } else {
                scales_packed[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                scales_packed[j - 4] |= (ls >> 4) << 6;
                scales_packed[j] |= (lm >> 4) << 6;
            }
        }
        let d_bytes = f16_bits(max_scale / 63.0);
        let dmin_bytes = f16_bits(max_min / 63.0);
        let d_f32 = f16_bits_to_f32(d_bytes);
        let dmin_f32 = f16_bits_to_f32(dmin_bytes);
        out.extend_from_slice(&d_bytes);
        out.extend_from_slice(&dmin_bytes);
        out.extend_from_slice(&scales_packed);
        for j in 0..8 {
            let (sc, m) = get_scale_min_k4(j, &scales_packed);
            let d = d_f32 * sc as f32;
            if d == 0.0 {
                continue;
            }
            let dm = dmin_f32 * m as f32;
            for ii in 0..32 {
                let li = nearest_int((blk[32 * j + ii] + dm) / d).clamp(0, 15);
                l[32 * j + ii] = li as u8;
            }
        }
        let mut qs = [0u8; 128];
        for (c, chunk) in qs.chunks_exact_mut(32).enumerate() {
            for li in 0..32 {
                let lo = l[64 * c + li] & 0x0F;
                let hi = (l[64 * c + 32 + li] & 0x0F) << 4;
                chunk[li] = lo | hi;
            }
        }
        out.extend_from_slice(&qs);
    }
    out
}

/// Канонический Q5_K (ggml block_q5_K): блок 256 → 176 байт:
///   d (f16), dmin (f16), scales[12] (упаковка get_scale_min_k4), qh[32]
///   (5-е биты; для чанка c биты m1 = 1<<(2c), m2 = 2<<(2c)), qs[128]
///   (низкие 4 бита, чанк-мажор). Деквант: y = d·sc·q − dmin·m, q ∈ 0..31.
/// Энкодер — порт quantize_row_q5_K_ref: make_qkx2_quants(32, 31, rmin −0.5,
/// nstep 15) + доквантовка по f16-округлённым d/dmin.
pub fn quant_q5_k(weights: &[f32]) -> Vec<u8> {
    assert!(weights.len().is_multiple_of(256), "Q5_K: len % 256 == 0");
    let mut out = Vec::with_capacity(weights.len() / 256 * 176);
    let mut l = [0u8; 256];
    let mut laux = [0u8; 32];
    let mut weights_sub = [0f32; 32];
    for blk in weights.chunks_exact(256) {
        let mut mins = [0f32; 8];
        let mut scales = [0f32; 8];
        let mut max_scale = 0f32;
        let mut max_min = 0f32;
        for j in 0..8 {
            let sub = &blk[32 * j..32 * j + 32];
            let mut sum_x2 = 0f32;
            for &v in sub.iter() {
                sum_x2 = v.mul_add(v, sum_x2);
            }
            let av_x = (sum_x2 / 32.0).sqrt();
            for (w, &v) in weights_sub.iter_mut().zip(sub) {
                *w = av_x + v.abs();
            }
            scales[j] = make_qkx2_quants(
                32,
                31,
                sub,
                &weights_sub,
                &mut l[32 * j..32 * j + 32],
                &mut mins[j],
                &mut laux,
                -0.5,
                0.1,
                15,
                false,
            );
            if scales[j] > max_scale {
                max_scale = scales[j];
            }
            if mins[j] > max_min {
                max_min = mins[j];
            }
        }
        let inv_scale = if max_scale > 0.0 {
            63.0 / max_scale
        } else {
            0.0
        };
        let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
        let mut scales_packed = [0u8; 12];
        for j in 0..8 {
            let ls = nearest_int(inv_scale * scales[j]).min(63) as u8;
            let lm = nearest_int(inv_min * mins[j]).min(63) as u8;
            if j < 4 {
                scales_packed[j] = ls;
                scales_packed[j + 4] = lm;
            } else {
                scales_packed[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                scales_packed[j - 4] |= (ls >> 4) << 6;
                scales_packed[j] |= (lm >> 4) << 6;
            }
        }
        let d_bytes = f16_bits(max_scale / 63.0);
        let dmin_bytes = f16_bits(max_min / 63.0);
        let d_f32 = f16_bits_to_f32(d_bytes);
        let dmin_f32 = f16_bits_to_f32(dmin_bytes);
        out.extend_from_slice(&d_bytes);
        out.extend_from_slice(&dmin_bytes);
        out.extend_from_slice(&scales_packed);
        for j in 0..8 {
            let (sc, m) = get_scale_min_k4(j, &scales_packed);
            let d = d_f32 * sc as f32;
            if d == 0.0 {
                continue;
            }
            let dm = dmin_f32 * m as f32;
            for ii in 0..32 {
                let li = nearest_int((blk[32 * j + ii] + dm) / d).clamp(0, 31);
                l[32 * j + ii] = li as u8;
            }
        }
        let mut qh = [0u8; 32];
        let mut qs = [0u8; 128];
        for (ci, chunk) in qs.chunks_exact_mut(32).enumerate() {
            let n = ci * 64;
            let (m1, m2) = (1u8 << (2 * ci), 2u8 << (2 * ci));
            for j in 0..32 {
                let mut l1 = l[n + j];
                if l1 > 15 {
                    l1 -= 16;
                    qh[j] |= m1;
                }
                let mut l2 = l[n + j + 32];
                if l2 > 15 {
                    l2 -= 16;
                    qh[j] |= m2;
                }
                chunk[j] = l1 | (l2 << 4);
            }
        }
        out.extend_from_slice(&qh);
        out.extend_from_slice(&qs);
    }
    out
}

/// Кодбук IQ4_NL/IQ4_XS (ggml `kvalues_iq4nl`), отсортирован — бинарный
/// поиск `best_index_int8` полагается на порядок.
const KVALUES_IQ4NL: [f32; 16] = [
    -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0, 69.0,
    89.0, 113.0,
];

/// `best_index_int8` из ggml-quants.c: ближайший индекс в отсортированном
/// кодбуке (бинарный поиск, при равенстве — больший индекс).
fn best_index_int8(val: &[f32; 16], x: f32) -> usize {
    if x <= val[0] {
        return 0;
    }
    if x >= val[15] {
        return 15;
    }
    let mut ml = 0usize;
    let mut mu = 15usize;
    while mu - ml > 1 {
        let mav = (ml + mu) / 2;
        if x < val[mav] {
            mu = mav;
        } else {
            ml = mav;
        }
    }
    if x - val[mu - 1] < val[mu] - x {
        mu - 1
    } else {
        mu
    }
}

/// IQ4_XS (ggml type 23): супер-блок 256 весов → 136 байт:
/// d(f16)@0, scales_h(u16)@2, scales_l[4]@4, qs[128]@8.
/// Деквант подблока ib: y = f16(d)·(ls−32)·kvalues[l], ls — 6 бит
/// (4 из scales_l + 2 из scales_h).
///
/// Энкодер — порт `quantize_iq4_xs`/`quantize_row_iq4_nl_impl`
/// (ggml-quants.c, коммит 5266f24da) по пути БЕЗ imatrix
/// (`quant_weights == NULL` → вес = x²; `tensor_requires_imatrix` не
/// требует imatrix для IQ4_XS). Порядок f32-операций и FMA-контракция
/// (`sumqx += w*q*x` → `fma(w*q, x, sumqx)`) — как у clang-сборки эталона:
/// байты обязаны совпадать с `llama-quantize` на том же входе.
pub fn quant_iq4_xs(weights: &[f32]) -> Vec<u8> {
    assert!(weights.len().is_multiple_of(256), "IQ4_XS: len % 256 == 0");
    let mut out = Vec::with_capacity(weights.len() / 256 * 136);
    let mut l = [0u8; 256];
    let mut scales = [0f32; 8];
    for blk in weights.chunks_exact(256) {
        let mut q4 = [0u8; 128]; // memset(q4, 0, 128) — как в ggml
        let mut scales_h = 0u16;
        let mut scales_l = [0u8; 4];
        let mut max_scale = 0f32;
        let mut amax_scale = 0f32;
        for ib in 0..8usize {
            let xb = &blk[32 * ib..32 * ib + 32];
            let lb = &mut l[32 * ib..32 * ib + 32];
            // вес без imatrix: w[j] = x[j]²
            let mut amax = 0f32;
            let mut max = 0f32;
            for &v in xb.iter() {
                let ax = v.abs();
                if ax > amax {
                    amax = ax;
                    max = v;
                }
            }
            if amax < GROUP_MAX_EPS {
                scales[ib] = 0.0;
                continue; // Lb перезапишется во втором проходе (idl=0 → 8)
            }
            let w = |j: usize| {
                let v = xb[j];
                v * v
            };
            let mut d = -max / KVALUES_IQ4NL[0]; // ntry=7>0
            let id = 1.0 / d;
            let mut sumqx = 0f32;
            let mut sumq2 = 0f32;
            for j in 0..32 {
                let al = id * xb[j];
                let li = best_index_int8(&KVALUES_IQ4NL, al);
                lb[j] = li as u8;
                let q = KVALUES_IQ4NL[li];
                let wj = w(j);
                // clang: sumqx + (w*q)*x → fma(w*q, x, sumqx)
                sumqx = (wj * q).mul_add(xb[j], sumqx);
                sumq2 = (wj * q).mul_add(q, sumq2);
            }
            d = if sumq2 > 0.0 { sumqx / sumq2 } else { 0.0 };
            let mut best = d * sumqx;
            for itry in -7i32..=7 {
                // C: (itry + values[0]) — целочисленное, затем /max
                let id = ((itry as f32) + KVALUES_IQ4NL[0]) / max;
                sumqx = 0.0;
                sumq2 = 0.0;
                for j in 0..32 {
                    let al = id * xb[j];
                    let li = best_index_int8(&KVALUES_IQ4NL, al);
                    let q = KVALUES_IQ4NL[li];
                    let wj = w(j);
                    sumqx = (wj * q).mul_add(xb[j], sumqx);
                    sumq2 = (wj * q).mul_add(q, sumq2);
                }
                if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                    d = sumqx / sumq2;
                    best = d * sumqx;
                }
            }
            scales[ib] = d;
            let abs_d = d.abs();
            if abs_d > amax_scale {
                amax_scale = abs_d;
                max_scale = d;
            }
        }
        // супер-блок: общий d, 6-битные ls на подблок (4 бита scales_l +
        // 2 бита scales_h), переквантование L по dl. ВНИМАНИЕ: id/dl в ggml
        // считаются от СЫРОГО f32 d (f16 округляется только записываемый dh).
        let d = -max_scale / 32.0;
        let dh_bytes = f16_bits(d);
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        for ib in 0..8usize {
            let mut li = nearest_int(id * scales[ib]);
            li = li.max(-32).min(31);
            let dl = d * li as f32;
            let idl = if dl != 0.0 { 1.0 / dl } else { 0.0 };
            let lb = &mut l[32 * ib..32 * ib + 32];
            let xb = &blk[32 * ib..32 * ib + 32];
            for j in 0..32 {
                lb[j] = best_index_int8(&KVALUES_IQ4NL, idl * xb[j]) as u8;
            }
            let ls = li + 32; // 0..63
            let l_l = (ls & 0xf) as u8;
            let l_h = ((ls >> 4) & 0x3) as u16;
            if ib % 2 == 0 {
                scales_l[ib / 2] = l_l;
            } else {
                scales_l[ib / 2] |= l_l << 4;
            }
            scales_h |= l_h << (2 * (ib % 8));
        }
        // упаковка нибблов: байт j — пары (j, j+16) подблока
        for i in 0..8usize {
            for j in 0..16 {
                q4[16 * i + j] = l[32 * i + j] | (l[32 * i + 16 + j] << 4);
            }
        }
        out.extend_from_slice(&dh_bytes);
        out.extend_from_slice(&scales_h.to_le_bytes());
        out.extend_from_slice(&scales_l);
        out.extend_from_slice(&q4);
    }
    out
}

/// Канонический Q6_K (ggml block_q6_K): блок 256 → 210 байт: ql[128]
/// (низкие 4 бита), qh[64] (старшие 2), scales[16] (int8 на 16-элементные
/// подблоки), d (f16) в КОНЦЕ (208..210). Деквант: y = d·sc·(q − 32).
/// Энкодер — порт quantize_row_q6_K_ref: подблоковые scale через
/// make_qx_quants(16, 32, rmse_type=1), затем q по f16-округлённому d·sc.
pub fn quant_q6_k(weights: &[f32]) -> Vec<u8> {
    assert!(weights.len().is_multiple_of(256), "Q6_K: len % 256 == 0");
    let mut out = Vec::with_capacity(weights.len() / 256 * 210);
    for blk in weights.chunks_exact(256) {
        let mut l = [0i8; 256];
        let mut scales = [0f32; 16];
        let mut max_scale = 0f32;
        let mut max_abs_scale = 0f32;
        for ib in 0..16 {
            let scale = make_qx_quants(
                16,
                32,
                &blk[16 * ib..16 * ib + 16],
                &mut l[16 * ib..16 * ib + 16],
                1,
            );
            scales[ib] = scale;
            let abs = scale.abs();
            if abs > max_abs_scale {
                max_abs_scale = abs;
                max_scale = scale;
            }
        }
        if max_abs_scale < GROUP_MAX_EPS {
            // Нулевой блок: d = 0, ql/qh/scales остаются нулями (210 байт).
            out.extend_from_slice(&[0u8; 210]);
            continue;
        }
        let iscale = -128.0 / max_scale;
        let d_bytes = f16_bits(1.0 / iscale);
        let d_f32 = f16_bits_to_f32(d_bytes);
        let mut block = [0u8; 210];
        block[208..210].copy_from_slice(&d_bytes);
        for (j, &scale) in scales.iter().enumerate() {
            let sc = nearest_int(iscale * scale).min(127) as i8;
            block[192 + j] = sc as u8;
            let dj = d_f32 * sc as f32;
            if dj == 0.0 {
                continue;
            }
            for ii in 0..16 {
                let q = nearest_int(blk[16 * j + ii] / dj).clamp(-32, 31);
                l[16 * j + ii] = (q + 32) as i8;
            }
        }
        for half in 0..2 {
            let b = half * 128;
            for k in 0..32 {
                let q1 = (l[b + k] & 0xF) as u8;
                let q2 = (l[b + k + 32] & 0xF) as u8;
                let q3 = (l[b + k + 64] & 0xF) as u8;
                let q4 = (l[b + k + 96] & 0xF) as u8;
                block[half * 64 + k] = q1 | (q3 << 4);
                block[half * 64 + k + 32] = q2 | (q4 << 4);
                block[128 + half * 32 + k] = (((l[b + k] >> 4) & 0x3) as u8)
                    | ((((l[b + k + 32] >> 4) & 0x3) as u8) << 2)
                    | ((((l[b + k + 64] >> 4) & 0x3) as u8) << 4)
                    | ((((l[b + k + 96] >> 4) & 0x3) as u8) << 6);
            }
        }
        out.extend_from_slice(&block);
    }
    out
}

/// Квантовать тензор в формат (f32 → байты ggml-типа).
pub fn quantize(weights: &[f32], fmt: QuantFormat) -> Vec<u8> {
    match fmt {
        QuantFormat::F32 => weights.iter().flat_map(|w| w.to_le_bytes()).collect(),
        QuantFormat::F16 => weights.iter().flat_map(|w| f16_bits(*w)).collect(),
        QuantFormat::Bf16 => weights.iter().flat_map(|w| bf16_bits(*w)).collect(),
        QuantFormat::Q4_0 => quant_q4_0(weights),
        QuantFormat::Q8_0 => quant_q8_0(weights),
        QuantFormat::Q4_K_M => quant_q4_k(weights),
        QuantFormat::Q6_K => quant_q6_k(weights),
        QuantFormat::Q5_0 => quant_q5_0(weights),
        QuantFormat::Q5_K => quant_q5_k(weights),
        QuantFormat::Iq4Xs => quant_iq4_xs(weights),
    }
}

// ── GGUF-райтер ─────────────────────────────────────────────────────────────

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF"
const GGUF_VERSION: u32 = 3;
/// Выравнивание секции данных по умолчанию (совпадает с ggml).
const TENSOR_ALIGN: u64 = 32;
// Типы GGUF v3 (как в парсере gguf.rs / llama.cpp): U8=0, I8=1, U16=2,
// I16=3, U32=4, I32=5, F32=6, BOOL=7, STRING=8, ARRAY=9, U64=10, I64=11,
// F64=12.
const GGUF_TYPE_U8: u32 = 0;
const GGUF_TYPE_I8: u32 = 1;
const GGUF_TYPE_U16: u32 = 2;
const GGUF_TYPE_I16: u32 = 3;
const GGUF_TYPE_U32: u32 = 4;
const GGUF_TYPE_I32: u32 = 5;
const GGUF_TYPE_F32: u32 = 6;
const GGUF_TYPE_BOOL: u32 = 7;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;
const GGUF_TYPE_U64: u32 = 10;
const GGUF_TYPE_I64: u32 = 11;
const GGUF_TYPE_F64: u32 = 12;

/// Значение метаданных для записи: те же варианты, что читает парсер
/// (`gguf::GgufValue`), плюс массивы — GGUF-значения переносятся из файла
/// в файл как есть (реальные модели содержат U32/I32/Bool/F32/строки и
/// массивы чисел: token_type, scores и т.д.).
pub use crate::gguf::GgufValue as MetaValue;

fn write_string(w: &mut Vec<u8>, s: &str) {
    w.extend_from_slice(&(s.len() as u64).to_le_bytes());
    w.extend_from_slice(s.as_bytes());
}

/// Тип GGUF-значения (для заголовка массива) + запись его байт.
fn value_bytes(v: &GgufValue, out: &mut Vec<u8>) -> Result<u32, BackendError> {
    Ok(match v {
        GgufValue::U8(x) => {
            out.push(*x);
            GGUF_TYPE_U8
        }
        GgufValue::I8(x) => {
            out.push(*x as u8);
            GGUF_TYPE_I8
        }
        GgufValue::U16(x) => {
            out.extend_from_slice(&x.to_le_bytes());
            GGUF_TYPE_U16
        }
        GgufValue::I16(x) => {
            out.extend_from_slice(&x.to_le_bytes());
            GGUF_TYPE_I16
        }
        GgufValue::U32(x) => {
            out.extend_from_slice(&x.to_le_bytes());
            GGUF_TYPE_U32
        }
        GgufValue::I32(x) => {
            out.extend_from_slice(&x.to_le_bytes());
            GGUF_TYPE_I32
        }
        GgufValue::F32(x) => {
            out.extend_from_slice(&x.to_le_bytes());
            GGUF_TYPE_F32
        }
        GgufValue::Bool(x) => {
            out.push(*x as u8);
            GGUF_TYPE_BOOL
        }
        GgufValue::String(s) => {
            write_string(out, s);
            GGUF_TYPE_STRING
        }
        GgufValue::U64(x) => {
            out.extend_from_slice(&x.to_le_bytes());
            GGUF_TYPE_U64
        }
        GgufValue::I64(x) => {
            out.extend_from_slice(&x.to_le_bytes());
            GGUF_TYPE_I64
        }
        GgufValue::F64(x) => {
            out.extend_from_slice(&x.to_le_bytes());
            GGUF_TYPE_F64
        }
        GgufValue::Array(items) => {
            // GGUF-массивы однородны; раскладка: тип элемента (u32), число
            // элементов (u64), затем сами элементы. Пустой массив — строковый
            // (как пишет llama.cpp для пустых списков).
            let elem_type = match items.first() {
                Some(first) => {
                    let mut probe = Vec::new();
                    value_bytes(first, &mut probe)?
                }
                None => GGUF_TYPE_STRING,
            };
            out.extend_from_slice(&elem_type.to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for it in items {
                let t = value_bytes(it, out)?;
                if t != elem_type {
                    return Err(BackendError::Shape(format!(
                        "gguf metadata: неоднородный массив ({t} != {elem_type})"
                    )));
                }
            }
            GGUF_TYPE_ARRAY
        }
    })
}

/// Описание выходного тензора (данные пишутся потоково, а не держатся в RAM).
#[derive(Debug, Clone)]
pub struct OutTensor {
    pub name: String,
    /// ne-порядок (fastest dim первый) — как в GGUF-файле.
    pub ne: Vec<u64>,
    pub ggml_type: GgmlType,
    /// Размер данных в байтах (должен совпадать с `ggml_type.bytes_per_element(numel)`).
    pub size: u64,
}

/// Тензор для записи: имя, dims (ne-порядок: fastest first), тип, данные.
pub struct QuantTensor {
    pub name: String,
    /// ne-порядок (fastest dim первый) — как в GGUF-файле.
    pub ne: Vec<u64>,
    pub ggml_type: GgmlType,
    pub data: Vec<u8>,
}

impl QuantTensor {
    fn out(&self) -> OutTensor {
        OutTensor {
            name: self.name.clone(),
            ne: self.ne.clone(),
            ggml_type: self.ggml_type,
            size: self.data.len() as u64,
        }
    }
}

/// Выравнивание данных: `general.alignment` из метаданных (степень двойки)
/// или 32 — как ggml. Файл, объявивший 64, обязан писать с 64.
pub fn alignment_of(metadata: &[(&str, &MetaValue)]) -> u64 {
    metadata
        .iter()
        .find(|(k, _)| *k == "general.alignment")
        .and_then(|(_, v)| v.as_u32())
        .map(u64::from)
        .filter(|a| *a > 0 && a.is_power_of_two())
        .unwrap_or(TENSOR_ALIGN)
}

/// Записать шапку (header + infos + паддинг до данных) и вернуть общее
/// число байт сверх шапки (сумма данных с межтензорными паддингами).
fn write_gguf_prefix<W: Write>(
    w: &mut W,
    metadata: &[(&str, &MetaValue)],
    tensors: &[OutTensor],
) -> Result<u64, BackendError> {
    let align = alignment_of(metadata);
    let mut header = Vec::new();
    header.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    header.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    header.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    header.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    let mut scratch = Vec::new();
    for (k, v) in metadata {
        write_string(&mut header, k);
        scratch.clear();
        let vt = value_bytes(v, &mut scratch)?;
        header.extend_from_slice(&vt.to_le_bytes());
        header.extend_from_slice(&scratch);
    }

    let mut infos = Vec::new();
    let mut offsets = Vec::with_capacity(tensors.len());
    let mut cur = 0u64;
    for t in tensors {
        while !cur.is_multiple_of(align) {
            cur += 1;
        }
        offsets.push(cur);
        cur += t.size;
    }
    let total = cur;
    for (t, off) in tensors.iter().zip(&offsets) {
        write_string(&mut infos, &t.name);
        infos.extend_from_slice(&(t.ne.len() as u32).to_le_bytes());
        for d in &t.ne {
            infos.extend_from_slice(&d.to_le_bytes());
        }
        infos.extend_from_slice(&t.ggml_type.raw().to_le_bytes());
        infos.extend_from_slice(&off.to_le_bytes());
    }

    let prefix_len = header.len() + infos.len();
    let mut data_start = prefix_len as u64;
    while !data_start.is_multiple_of(align) {
        data_start += 1;
    }
    w.write_all(&header)
        .and_then(|_| w.write_all(&infos))
        .map_err(|e| BackendError::Native(format!("gguf write: {e}")))?;
    let pad = (data_start as usize).saturating_sub(prefix_len);
    if pad > 0 {
        w.write_all(&vec![0u8; pad])
            .map_err(|e| BackendError::Native(format!("gguf write: {e}")))?;
    }
    Ok(total)
}

/// Считающий адаптер: ловит расхождение «сколько записали» с объявленным
/// размером тензора (молчаливый сдвиг всех последующих оффсетов — порча
/// файла; у llama.cpp это `GGML_ASSERT(gguf_get_tensor_size == new_size)`).
struct CountingWriter<'a, W: Write> {
    inner: &'a mut W,
    written: u64,
}

impl<W: Write> Write for CountingWriter<'_, W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.written += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Записать GGUF v3 потоково: данные тензоров отдаёт `write_data(i, w)`
/// (по тензору за вызов — модель любого размера не держится в RAM).
pub fn write_gguf_stream<W: Write>(
    w: &mut W,
    metadata: &[(&str, &MetaValue)],
    tensors: &[OutTensor],
    mut write_data: impl FnMut(usize, &mut dyn Write) -> Result<(), BackendError>,
    progress: &mut dyn FnMut(u64, u64),
) -> Result<(), BackendError> {
    let align = alignment_of(metadata);
    let total = write_gguf_prefix(w, metadata, tensors)?;
    let mut done = 0u64;
    for (i, t) in tensors.iter().enumerate() {
        while !done.is_multiple_of(align) {
            w.write_all(&[0u8])
                .map_err(|e| BackendError::Native(format!("gguf write: {e}")))?;
            done += 1;
        }
        let mut cw = CountingWriter {
            inner: w,
            written: 0,
        };
        write_data(i, &mut cw)?;
        if cw.written != t.size {
            return Err(BackendError::Shape(format!(
                "gguf: тензор {} записан {} байт, объявлено {} — оффсеты сдвинулись бы",
                t.name, cw.written, t.size
            )));
        }
        done += t.size;
        progress(done, total);
    }
    Ok(())
}

/// Записать GGUF v3 (данные уже в памяти): обёртка над потоковой версией.
pub fn write_gguf<W: Write>(
    w: &mut W,
    metadata: &[(&str, MetaValue)],
    tensors: &[QuantTensor],
    progress: &mut dyn FnMut(u64, u64),
) -> Result<(), BackendError> {
    let desc: Vec<OutTensor> = tensors.iter().map(QuantTensor::out).collect();
    let refs: Vec<(&str, &MetaValue)> = metadata.iter().map(|(k, v)| (*k, v)).collect();
    write_gguf_stream(
        w,
        &refs,
        &desc,
        |i, w| {
            w.write_all(&tensors[i].data)
                .map_err(|e| BackendError::Native(format!("gguf write: {e}")))
        },
        progress,
    )
}

// ── конверсия safetensors → GGUF ────────────────────────────────────────────

/// Конвертировать каталог модели (model.safetensors + config.json) в GGUF.
/// Память: по одному тензору (f32 читается с диска по оффсету), данные
/// квантуются и пишутся сразу. Токенайзер не переносится (v1 — веса;
/// vocab — отдельная подфаза Ф1.5-конвертера).
pub fn safetensors_to_gguf(
    model_dir: &Path,
    out: &Path,
    fmt: QuantFormat,
    progress: &mut dyn FnMut(u64, u64),
) -> Result<QuantReport, BackendError> {
    let st_path = model_dir.join("model.safetensors");
    let names =
        safetensors_tensor_names(&st_path).map_err(|e| BackendError::Native(e.to_string()))?;
    let (arch, name, ctx) = read_config(model_dir);
    let ggml_type = fmt.ggml_type();

    let meta: Vec<(&str, MetaValue)> = vec![
        ("general.architecture", MetaValue::String(arch.clone())),
        ("general.name", MetaValue::String(name.clone())),
        ("context_length", MetaValue::I32(ctx as i32)),
        ("general.file_type", MetaValue::I32(file_type_of(fmt))),
    ];

    // собираем тензоры по одному: f32 (с диска) → квант
    let mut tensors: Vec<QuantTensor> = Vec::with_capacity(names.len());
    let total_elems: u64 = names
        .iter()
        .map(|n| safetensors_tensor_numel(&st_path, n).unwrap_or(0) as u64)
        .sum();
    for n in &names {
        let (data, shape) = read_safetensors_tensor_f32(&st_path, n)
            .map_err(|e| BackendError::Native(e.to_string()))?;
        // ne = реверс shape (последняя dim — самая медленная в safetensors)
        let ne: Vec<u64> = shape.iter().rev().map(|d| *d as u64).collect();
        let numel: usize = shape.iter().product();
        // кратность блоку для квантов (fail fast на ровном)
        if fmt.needs_block_pad() {
            let block = ggml_type.block_elems() as usize;
            if !numel.is_multiple_of(block) && numel > block {
                return Err(BackendError::Shape(format!(
                    "{n}: numel {numel} не кратен блоку {block} для {fmt:?}"
                )));
            }
        }
        let q = quantize(&data, fmt);
        tensors.push(QuantTensor {
            name: n.clone(),
            ne,
            ggml_type,
            data: q,
        });
    }
    let mut file = std::fs::File::create(out)
        .map_err(|e| BackendError::Native(format!("create {}: {e}", out.display())))?;
    let total_bytes: u64 = tensors.iter().map(|t| t.data.len() as u64).sum();
    let mut p = |d: u64, _t: u64| {
        let frac = if total_bytes > 0 {
            d as f64 / total_bytes as f64
        } else {
            0.0
        };
        progress((frac * total_elems as f64) as u64, total_elems);
    };
    write_gguf(&mut file, &meta, &tensors, &mut p)?;
    Ok(QuantReport {
        arch,
        name,
        tensors: tensors.len(),
        bytes: total_bytes,
        format: fmt.name(),
        src_bytes: std::fs::metadata(&st_path).map(|m| m.len()).unwrap_or(0),
        by_type: vec![(fmt.name(), tensors.len(), total_bytes)],
        warnings: Vec::new(),
    })
}

fn file_type_of(fmt: QuantFormat) -> i32 {
    match fmt {
        QuantFormat::F32 => 0,
        QuantFormat::F16 => 1,
        QuantFormat::Q4_0 => 2,
        QuantFormat::Q8_0 => 7,
        QuantFormat::Q4_K_M => 15,
        QuantFormat::Q6_K => 18,
        QuantFormat::Q5_0 => 8,
        QuantFormat::Q5_K => 17,
        QuantFormat::Bf16 => 29,
        QuantFormat::Iq4Xs => 30, // LLAMA_FTYPE_MOSTLY_IQ4_XS
    }
}

/// Минимальная конфигурация из config.json (arch/name/context_length).
fn read_config(model_dir: &Path) -> (String, String, usize) {
    let cfg = model_dir.join("config.json");
    let mut arch = "unknown".to_string();
    let mut name = model_dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".into());
    let mut ctx = 4096usize;
    if let Some(v) = std::fs::read_to_string(&cfg)
        .ok()
        .and_then(|c| serde_json::from_str::<serde_json::Value>(&c).ok())
    {
        if let Some(a) = v
            .get("architectures")
            .and_then(|x| x.as_array())
            .and_then(|a| a.first())
            .and_then(|x| x.as_str())
        {
            arch = a.to_lowercase();
        }
        if let Some(n) = v.get("_name_or_path").and_then(|x| x.as_str()) {
            name = n.rsplit('/').next().unwrap_or(n).to_string();
        }
        if let Some(c) = v.get("max_position_embeddings").and_then(|x| x.as_u64()) {
            ctx = c as usize;
        }
    }
    (arch, name, ctx)
}

// ── политика типов при GGUF → GGUF квантовании (порт llama-quant.cpp) ───────
//
// llama-quantize квантует НЕ всё подряд: 1-D тензоры (нормы) и часть
// служебных остаются в исходном типе, эмбеддинги/выход и «чувствительные»
// слои (ffn_down/attn_v в начале, конце и каждом третьем слое) получают
// Q6_K. Файл, у которого всё подряд Q4_K, — не Q4_K_M. Здесь — построчный
// порт `tensor_allows_quantization`, `llama_tensor_get_type_impl` и
// `tensor_type_fallback` (llama.cpp 5266f24da = сборочный коммит
// llama-quantize на этой машине; сверено с master 2026-10: отличия только
// для архитектуры GLM5).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TensorCategory {
    Output,
    TokenEmbd,
    AttnQkv,
    AttnKvB,
    AttnV,
    AttnK,
    AttnQ,
    AttnOutput,
    FfnUp,
    FfnGate,
    FfnDown,
    Other,
}

fn tensor_category(name: &str) -> TensorCategory {
    use TensorCategory as C;
    // Порядок проверок — как в llama.cpp (важен: "attn_qkv" раньше "attn_v",
    // "attn_output" — после "attn_q").
    if name == "output.weight" {
        return C::Output;
    }
    if name == "token_embd.weight" || name == "per_layer_token_embd.weight" {
        return C::TokenEmbd;
    }
    if name.contains("attn_qkv.weight") {
        return C::AttnQkv;
    }
    if name.contains("attn_kv_b.weight") {
        return C::AttnKvB;
    }
    if name.contains("attn_v.weight") {
        return C::AttnV;
    }
    if name.contains("attn_k.weight") {
        return C::AttnK;
    }
    if name.contains("attn_q.weight") {
        return C::AttnQ;
    }
    if name.contains("attn_output.weight") {
        return C::AttnOutput;
    }
    if name.contains("ffn_up") {
        return C::FfnUp;
    }
    if name.contains("ffn_gate") {
        return C::FfnGate;
    }
    if name.contains("ffn_down") {
        return C::FfnDown;
    }
    C::Other
}

fn category_is_attn_v(cat: TensorCategory) -> bool {
    matches!(
        cat,
        TensorCategory::AttnV | TensorCategory::AttnQkv | TensorCategory::AttnKvB
    )
}

/// Можно ли вообще квантовать тензор (llama.cpp: только 2-D/3-D `*.weight`,
/// не нормы и не служебные таблицы). Остальные остаются в исходном типе.
fn tensor_allows_quantization(name: &str, n_dims: usize) -> bool {
    if n_dims < 2 {
        return false;
    }
    if !name.ends_with("weight") {
        return false;
    }
    if name.contains("_norm.weight") {
        return false;
    }
    if name.contains("ffn_gate_inp.weight") || name.contains("ffn_gate_tid2eid.weight") {
        return false;
    }
    if name.contains("altup") || name.contains("laurel") || name.contains("per_layer_model_proj") {
        return false;
    }
    for pat in [
        "ssm_conv1d",
        "shortconv.conv.weight",
        "indexer.k_proj.weight",
        "indexer.q_proj.weight",
        "attn_rel_b.weight",
        ".position_embd",
        "sam.pos_embd",
        "sam.neck.",
        "sam.net_",
        ".rel_pos",
        ".patch_embd",
        ".patch_merger",
        "a.rvq.codebook",
        "mm.a.code_embd",
        "time_mix_",
    ] {
        if name.contains(pat) {
            return false;
        }
    }
    true
}

/// Состояние политики: счётчики слоёв в порядке перечисления тензоров
/// (в llama.cpp — порядок тензоров модели, у нас — порядок в файле GGUF).
struct TypePolicy {
    fmt: QuantFormat,
    n_attention_wv: usize,
    i_attention_wv: usize,
    n_layer: usize,
    i_ffn_down: usize,
    has_tied_embeddings: bool,
    /// `hparams.n_gqa()` = n_head/n_head_kv (из форм тензоров attn_q/attn_k);
    /// 0 — неизвестно. Правило IQ4_XS/IQ4_NL: attn_v при n_gqa ≥ 4 → Q5_K.
    n_gqa: usize,
    pub fallbacks: Vec<String>,
}

impl TypePolicy {
    fn new(fmt: QuantFormat, names: &[(String, usize)], n_layer: usize, n_gqa: usize) -> Self {
        let mut n_attention_wv = 0;
        let mut has_tied_embeddings = true;
        for (name, _) in names {
            if category_is_attn_v(tensor_category(name)) {
                n_attention_wv += 1;
            }
            if tensor_category(name) == TensorCategory::Output {
                has_tied_embeddings = false;
            }
        }
        Self {
            fmt,
            n_attention_wv,
            i_attention_wv: 0,
            n_layer,
            i_ffn_down: 0,
            has_tied_embeddings,
            n_gqa,
            fallbacks: Vec::new(),
        }
    }

    /// `use_more_bits` из llama.cpp: «важные» слои (первые/последние 1/8 и
    /// каждый третий внутри) получают больше бит.
    fn use_more_bits(i_layer: usize, n_layers: usize) -> bool {
        if n_layers == 0 {
            return false;
        }
        i_layer < n_layers / 8 || i_layer >= 7 * n_layers / 8 || (i_layer - n_layers / 8) % 3 == 2
    }

    fn default_type(fmt: QuantFormat) -> GgmlType {
        match fmt {
            QuantFormat::F32 => GgmlType::F32,
            QuantFormat::F16 => GgmlType::F16,
            QuantFormat::Bf16 => GgmlType::Bf16,
            QuantFormat::Q4_0 => GgmlType::Q4_0,
            QuantFormat::Q8_0 => GgmlType::Q8_0,
            QuantFormat::Q4_K_M => GgmlType::Q4K,
            QuantFormat::Q6_K => GgmlType::Q6K,
            QuantFormat::Q5_0 => GgmlType::Q5_0,
            QuantFormat::Q5_K => GgmlType::Q5K,
            QuantFormat::Iq4Xs => GgmlType::Iq4Xs,
        }
    }

    /// `llama_tensor_get_type_impl`: правила по категории, ftype и слою.
    fn impl_type(
        &mut self,
        mut new_type: GgmlType,
        name: &str,
        ne0: u64,
        cat: TensorCategory,
    ) -> GgmlType {
        let qk_k = new_type.block_elems() as u64;
        let ftype = self.fmt;
        if cat == TensorCategory::Output
            || (self.has_tied_embeddings && cat == TensorCategory::TokenEmbd)
        {
            // Выход (и связанный с ним эмбеддинг): Q8_0 при несовместимой
            // форме, иначе Q6_K (для Q8_0-файла — остаётся Q8_0).
            if new_type == GgmlType::Q8_0 || !ne0.is_multiple_of(qk_k) {
                new_type = GgmlType::Q8_0;
            } else {
                new_type = GgmlType::Q6K;
            }
        } else if cat == TensorCategory::TokenEmbd {
            // Отдельный (не связанный) эмбеддинг — как решено для ftype.
        } else if category_is_attn_v(cat) {
            if matches!(ftype, QuantFormat::Q4_K_M | QuantFormat::Q5_K)
                && Self::use_more_bits(self.i_attention_wv, self.n_attention_wv)
            {
                new_type = GgmlType::Q6K;
            }
            // llama-quant.cpp:553: IQ4_NL/IQ4_XS + n_gqa ≥ 4 → attn_v в Q5_K
            // (у Qwen3-8B n_gqa=4 — потому в эталонном IQ4_XS-файле attn_v и
            // есть Q5_K). Только чистый v: у fused qkv своя категория.
            if ftype == QuantFormat::Iq4Xs && cat == TensorCategory::AttnV && self.n_gqa >= 4 {
                new_type = GgmlType::Q5K;
            }
            self.i_attention_wv += 1;
        } else if cat == TensorCategory::AttnK || cat == TensorCategory::AttnQ {
            // Для наших ftype без изменений.
        } else if cat == TensorCategory::FfnDown {
            if matches!(ftype, QuantFormat::Q4_K_M | QuantFormat::Q5_K)
                && Self::use_more_bits(self.i_ffn_down, self.n_layer)
            {
                new_type = GgmlType::Q6K;
            }
            // llama-quant.cpp:620: IQ4_NL/IQ4_XS БЕЗ imatrix: первые 1/8
            // ffn_down → Q5_K (защита первых слоёв). Наш энкодер imatrix не
            // использует — правило применяется всегда.
            if ftype == QuantFormat::Iq4Xs && self.i_ffn_down < self.n_layer / 8 {
                new_type = GgmlType::Q5K;
            }
            self.i_ffn_down += 1;
        } else if cat == TensorCategory::AttnOutput {
            // Для наших ftype без изменений.
        } else if cat == TensorCategory::AttnQkv {
            // Fused qkv (Qwen3.5 и родственные): Q4_K_M → Q5_K.
            if ftype == QuantFormat::Q4_K_M {
                new_type = GgmlType::Q5K;
            }
        } else if cat == TensorCategory::FfnGate || cat == TensorCategory::FfnUp {
            // Без изменений для наших ftype.
        }
        let _ = name;
        new_type
    }

    /// `tensor_type_fallback`: несовместимая форма (ne[0] не кратен блоку).
    fn fallback(&mut self, name: &str, ne0: u64, target: GgmlType) -> GgmlType {
        let qk = target.block_elems() as u64;
        if ne0.is_multiple_of(qk) {
            return target;
        }
        let demoted = match target {
            // Q4_K → Q5_0 (правило llama.cpp); Q6_K → Q8_0; Q5_K → Q5_1
            // (движок его не пишет — ближайший поддерживаемый Q5_0).
            // IQ4_XS → llama даёт IQ4_NL (блок 32); движок IQ4_NL не пишет,
            // поэтому ниже он деградирует в F16 (для форм Qwen3 не срабатывает:
            // ne[0] кратен 256).
            GgmlType::Q4K => GgmlType::Q5_0,
            GgmlType::Q6K => GgmlType::Q8_0,
            GgmlType::Q5K => GgmlType::Q5_0,
            other => other,
        };
        let demoted = if ne0.is_multiple_of(demoted.block_elems() as u64) {
            demoted
        } else {
            GgmlType::F16
        };
        self.fallbacks.push(format!(
            "{name}: ne[0] {ne0} не кратен {qk} для {target:?} → {demoted:?}"
        ));
        demoted
    }

    /// Итоговый тип тензора.
    fn target_type(&mut self, name: &str, ne: &[u64], src_type: GgmlType) -> GgmlType {
        if !tensor_allows_quantization(name, ne.len()) {
            return src_type;
        }
        let cat = tensor_category(name);
        let new_type = Self::default_type(self.fmt);
        let ne0 = ne.first().copied().unwrap_or(0);
        let new_type = self.impl_type(new_type, name, ne0, cat);
        self.fallback(name, ne0, new_type)
    }
}

/// Число слоёв модели: `<arch>.block_count` или по именам blk.N.
fn block_count_of(file: &crate::gguf::GgufFile<'_>, arch: &str) -> usize {
    if let Some(n) = file
        .get(&format!("{arch}.block_count"))
        .and_then(GgufValue::as_u32)
    {
        return n as usize;
    }
    let mut max_layer = 0usize;
    let mut seen = false;
    for t in &file.tensors {
        if let Some(rest) = t.name.strip_prefix("blk.")
            && let Some(num) = rest.split('.').next()
            && let Ok(n) = num.parse::<usize>()
        {
            seen = true;
            max_layer = max_layer.max(n + 1);
        }
    }
    if seen { max_layer } else { 0 }
}

// ── GGUF → GGUF квантование ─────────────────────────────────────────────────

/// Отчёт квантования.
#[derive(Debug, Clone)]
pub struct QuantReport {
    pub arch: String,
    pub name: String,
    pub tensors: usize,
    pub bytes: u64,
    pub format: &'static str,
    /// Размер исходного файла (для «8.0 → 2.2 ГБ» в панели).
    pub src_bytes: u64,
    /// Гистограмма (тип, тензоров, байт).
    pub by_type: Vec<(&'static str, usize, u64)>,
    /// Отклонения от llama.cpp (пусто = политика совпала).
    pub warnings: Vec<String>,
}

/// Квантовать GGUF в другой формат: потоково (файл любого размера не
/// читается в RAM целиком — mmap + слабы по строкам), с политикой типов
/// llama.cpp для норм/эмбеддингов/слоёв и с отменой.
///
/// Совместимость байт: для Q4_0/Q8_0 результат совпадает с `llama-quantize`
/// побитово (та же reference-арифметика ggml); для Q4_K_M/Q6_K — тоже,
/// пока форма тензоров совместима, а `fallbacks`/`warnings` пусты.
pub fn quantize_gguf(
    src: &Path,
    dst: &Path,
    fmt: QuantFormat,
    progress: &mut dyn FnMut(u64, u64),
    cancel: Option<&AtomicBool>,
) -> Result<QuantReport, BackendError> {
    let src_file = std::fs::File::open(src)
        .map_err(|e| BackendError::Native(format!("open {}: {e}", src.display())))?;
    let mmap = unsafe { memmap2::Mmap::map(&src_file) }
        .map_err(|e| BackendError::Native(format!("mmap {}: {e}", src.display())))?;
    let gf = parse(&mmap).map_err(|e| BackendError::Native(format!("gguf parse: {e}")))?;
    let arch = gf.architecture().unwrap_or("unknown").to_string();
    let name = gf
        .get("general.name")
        .and_then(|v| v.as_str())
        .unwrap_or("model")
        .to_string();

    // план: тип каждого тензора по политике llama.cpp
    let names: Vec<(String, usize)> = gf
        .tensors
        .iter()
        .map(|t| (t.name.clone(), t.dims.len()))
        .collect();
    // n_gqa = ne1(attn_q)/ne1(attn_k): в GGUF ne[0] — ВХОД (n_embd, у q и k
    // одинаковый), а размер выхода (n_head·head_dim / n_kv·head_dim) — ne[1].
    // Нужно правилу IQ4_XS для attn_v; 0, если тензоров нет.
    let n_gqa = {
        let out_dim = |pat: &str| {
            gf.tensors
                .iter()
                .find(|t| t.name.contains(pat))
                .and_then(|t| t.dims.get(1))
                .copied()
        };
        match (out_dim("attn_q.weight"), out_dim("attn_k.weight")) {
            (Some(q), Some(k)) if k > 0 && q % k == 0 => (q / k) as usize,
            _ => 0,
        }
    };
    let mut policy = TypePolicy::new(fmt, &names, block_count_of(&gf, &arch), n_gqa);
    let mut desc = Vec::with_capacity(gf.tensors.len());
    let mut plan: Vec<(GgmlType, bool)> = Vec::with_capacity(gf.tensors.len());
    for t in &gf.tensors {
        let target = policy.target_type(&t.name, &t.dims, t.ggml_type);
        if fmt.needs_block_pad() && target == TypePolicy::default_type(fmt) {
            let block = target.block_elems() as u64;
            let ne0 = t.dims.first().copied().unwrap_or(0);
            if ne0 % block != 0 && ne0 > block {
                return Err(BackendError::Shape(format!(
                    "{}: ne[0] {ne0} не кратен блоку {block} для {fmt}",
                    t.name
                )));
            }
        }
        let numel = t.numel();
        desc.push(OutTensor {
            name: t.name.clone(),
            ne: t.dims.clone(),
            ggml_type: target,
            size: target.bytes_per_element(numel),
        });
        let same = target == t.ggml_type;
        plan.push((target, same));
    }

    // метаданные: переносим всё, кроме split.*; file_type и
    // quantization_version — как llama-quantize.
    let mut meta: Vec<(String, GgufValue)> = Vec::with_capacity(gf.metadata.len() + 2);
    for (k, v) in &gf.metadata {
        if k == "split.no" || k == "split.count" || k == "split.tensors_count" {
            continue;
        }
        if k == "general.file_type" || k == "general.quantization_version" {
            continue;
        }
        meta.push((k.clone(), v.clone()));
    }
    meta.push((
        "general.quantization_version".to_string(),
        GgufValue::U32(2), // GGML_QNT_VERSION
    ));
    meta.push((
        "general.file_type".to_string(),
        GgufValue::U32(file_type_of(fmt) as u32),
    ));
    let meta_refs: Vec<(&str, &MetaValue)> = meta.iter().map(|(k, v)| (k.as_str(), v)).collect();

    // запись: шапка, затем тензоры потоково (по слабам строк)
    let mut out = std::fs::File::create(dst)
        .map_err(|e| BackendError::Native(format!("create {}: {e}", dst.display())))?;
    let total_src: u64 = gf.tensors.iter().map(|t| t.data.len() as u64).sum();
    let mut written: u64 = 0;
    let mut hist: Vec<(&'static str, usize, u64)> = Vec::new();
    for (t, d) in gf.tensors.iter().zip(&desc) {
        hist_push(&mut hist, d.ggml_type.name(), d.size);
        let _ = t;
    }
    // Слаб по строкам: ~16 МБ источника на итерацию (0.6B/8B/… не читаются
    // в RAM целиком). Внутри тензора прогресс считается по исходным байтам.
    const SLAB_BYTES: usize = 16 << 20;
    let check_cancel = || -> Result<(), BackendError> {
        if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            Err(BackendError::Cancelled)
        } else {
            Ok(())
        }
    };
    let result = write_gguf_stream(
        &mut out,
        &meta_refs,
        &desc,
        |i, w| {
            check_cancel()?;
            let t = &gf.tensors[i];
            let (target, same) = plan[i];
            if same {
                // Тип не меняется — копируем байты как есть (llama.cpp так же).
                w.write_all(t.data)
                    .map_err(|e| BackendError::Native(format!("gguf write: {e}")))?;
                written += t.data.len() as u64;
                progress(written, total_src);
                return Ok(());
            }
            let ne0 = t.dims.first().copied().unwrap_or(0) as usize;
            let src_row = t.ggml_type.bytes_per_element(ne0 as u64) as usize;
            let dst_row = target.bytes_per_element(ne0 as u64) as usize;
            // Энкодер — по ЦЕЛЕВОМУ типу (не по формату файла: у эмбеддингов
            // и выхода тип выше — Q6_K/Q8_0).
            let enc = QuantFormat::from_ggml_type(target).ok_or_else(|| {
                BackendError::Shape(format!(
                    "{}: тип {} не поддержан движком",
                    t.name,
                    target.name()
                ))
            })?;
            let rows_total = t.data.len().checked_div(src_row.max(1)).unwrap_or(0);
            let rows_per_slab = (SLAB_BYTES / src_row.max(1)).max(1);
            let mut row = 0usize;
            while row < rows_total {
                check_cancel()?;
                let n = rows_per_slab.min(rows_total - row);
                let src = &t.data[row * src_row..(row + n) * src_row];
                let w32 = dequant_gguf(t.ggml_type, src)?;
                let q = quantize(&w32, enc);
                debug_assert_eq!(q.len(), n * dst_row, "{}: слab размер", t.name);
                w.write_all(&q)
                    .map_err(|e| BackendError::Native(format!("gguf write: {e}")))?;
                row += n;
                written += (n * src_row) as u64;
                progress(written, total_src);
            }
            Ok(())
        },
        &mut |_, _| {},
    );
    if let Err(e) = result {
        drop(out);
        let _ = std::fs::remove_file(dst); // не оставляем обрубок
        return Err(e);
    }
    let bytes = desc.iter().map(|t| t.size).sum();
    Ok(QuantReport {
        arch,
        name,
        tensors: gf.tensors.len(),
        bytes,
        format: fmt.name(),
        src_bytes: mmap.len() as u64,
        by_type: hist,
        warnings: policy.fallbacks.clone(),
    })
}

fn hist_push(hist: &mut Vec<(&'static str, usize, u64)>, ty: &'static str, bytes: u64) {
    if let Some(e) = hist.iter_mut().find(|e| e.0 == ty) {
        e.1 += 1;
        e.2 += bytes;
    } else {
        hist.push((ty, 1, bytes));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < tol, "{x} vs {y}");
        }
    }

    fn dequant_q4_0(data: &[u8], n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n];
        for (bi, blk) in data.chunks_exact(18).enumerate() {
            let d = f16::from_le_bytes([blk[0], blk[1]]);
            // Canonical ggml pairing: byte j = w[j] (low) | w[j+16] << 4.
            for (j, byte) in blk[2..].iter().enumerate() {
                let base = bi * 32 + j;
                out[base] = ((byte & 0x0F) as i8 - 8) as f32 * d;
                out[base + 16] = ((byte >> 4) as i8 - 8) as f32 * d;
            }
        }
        out.truncate(n);
        out
    }

    /// N (iter-9): a REAL llama.cpp Q4_0 block (canonical pairing) must
    /// decode to the expected values — guards the mirrored-bug class that
    /// shipped the old dialect (encoder+decoder both wrong still round-trip).
    #[test]
    fn q4_0_canonical_layout_vector() {
        // Hand-crafted canonical block: d = 1.0, byte 0 = 0x12 → w[0] =
        // 2-8 = -6 (low nibble), w[16] = 1-8 = -7 (high nibble); all other
        // bytes 0 → w[j] = -8.
        let mut data = vec![0u8; 18];
        data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        data[2] = 0x12;
        let v = dequant_q4_0(&data, 32);
        assert_eq!(v.len(), 32);
        assert!((v[0] - (-6.0)).abs() < 1e-4, "w[0] = {}", v[0]);
        assert!((v[16] - (-7.0)).abs() < 1e-4, "w[16] = {}", v[16]);
        for j in [1usize, 15, 17, 31] {
            assert!((v[j] - (-8.0)).abs() < 1e-4, "w[{j}] = {}", v[j]);
        }
    }

    #[test]
    fn q4_0_roundtrip() {
        let w: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 8.0).collect();
        let enc = quant_q4_0(&w);
        // Q4_0: шаг кванта d = max_abs/8 — ошибка до ~d/2 + clamp
        approx(&dequant_q4_0(&enc, 64), &w, 0.5);
    }

    #[test]
    fn q8_0_roundtrip() {
        let w: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 4.0).collect();
        let enc = quant_q8_0(&w);
        // Q8_0: f16 d + 32 i8
        let mut out = Vec::new();
        for blk in enc.chunks_exact(34) {
            let d = f16::from_le_bytes([blk[0], blk[1]]);
            for q in &blk[2..] {
                out.push((*q as i8) as f32 * d);
            }
        }
        approx(&out, &w, 0.05);
    }

    #[test]
    fn q4_k_size_and_roundtrip() {
        let w: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 64.0).collect();
        let enc = quant_q4_k(&w);
        // Канонический блок: 2+2+12+128 = 144 байта.
        assert_eq!(enc.len(), 144);
        // Полный roundtrip через канонический декодер (weights_map).
        let dec = crate::weights_map::dequant_q4_k(&enc);
        // Ошибка ≤ dl/2 + f16-округление d/dmin. Подблоки с ПОЛОЖИТЕЛЬНЫМ
        // min кодируются от нуля (формат не хранит m < 0) — шаг вдвое
        // грубее: dl = 2/15 на крайнем правом подблоке.
        approx(&dec, &w, 0.08);
    }

    /// Q4_K-декодер читает КАНОНИЧЕСКУЮ раскладку (сверено с ggml
    /// dequantize_row_q4_K): d, dmin, scales[12] (get_scale_min_k4), qs
    /// чанк-мажор. Ручной блок с известными коэффициентами — контроль
    /// раскладки независимо от энкодера.
    #[test]
    fn q4_k_canonical_layout_vector() {
        // d = 0.5, dmin = 0.25; подблок 0: sc = 2, m = 1 → y = 0.5·2·q − 0.25.
        // Чанк 0: значения 0..31 (low-nibbles) → подблок is=0, значения
        // 32..63 (high-nibbles) → подблок is=1 (тот же (2,1) для простоты).
        let mut blk = vec![0u8; 144];
        blk[0..2].copy_from_slice(&f16_bits(0.5));
        blk[2..4].copy_from_slice(&f16_bits(0.25));
        // get_scale_min_k4(0) = (scales[0], scales[4]); get_scale_min_k4(1) =
        // (scales[1], scales[5]) — все (2, 1).
        for j in [0usize, 1] {
            blk[4 + j] = 2;
            blk[8 + j] = 1;
        }
        // qs чанк-мажор: значение q(i) в nibble байта 32c+l.
        // q = 0 → y = −0.25; q = 15 → y = 15·1 − 0.25 = 14.75.
        blk[16] = 0xF0; // value 0 = low nibble 0, value 32 = high nibble 15
        let y = crate::weights_map::dequant_q4_k(&blk);
        assert_eq!(y.len(), 256);
        assert_eq!(y[0], 0.5 * 2.0 * 0.0 - 0.25);
        assert_eq!(y[32], 0.5 * 2.0 * 15.0 - 0.25);
        // Остальные подблоки: sc/m = 0 → y = 0.
        assert_eq!(y[64], 0.0);
    }

    /// Q5_K/Q5_0: round-trip через CPU-деквант (та же раскладка, что в
    /// Metal-ядрах dot_q5_k/dot_q5_0 и в embedding-ветках).
    #[test]
    fn q5_roundtrip() {
        let w: Vec<f32> = (0..512).map(|i| (i as f32 * 0.37).sin() * 0.5).collect();
        let enc = quant_q5_k(&w);
        assert_eq!(enc.len(), 512 / 256 * 176, "Q5_K: 176 Б на 256");
        let dec = crate::weights_map::dequant_q5_k(&enc);
        approx(&dec, &w, 0.06);
        let enc0 = quant_q5_0(&w);
        assert_eq!(enc0.len(), 512 / 32 * 22, "Q5_0: 22 Б на 32");
        let dec0 = crate::weights_map::dequant_q5_0(&enc0);
        approx(&dec0, &w, 0.06);
    }

    #[test]
    fn q6_k_size_and_roundtrip() {
        let w: Vec<f32> = (0..256).map(|i| ((i as f32 * 1.7).sin()) * 0.3).collect();
        let enc = quant_q6_k(&w);
        // Канонический блок: 128+64+16+2 = 210 байт, d в хвосте.
        assert_eq!(enc.len(), 210);
        let d = f16::from_le_bytes([enc[208], enc[209]]);
        assert!(d != 0.0, "super-block scale must sit at bytes 208..210");
        // Roundtrip через канонический декодер (weights_map).
        let dec = crate::weights_map::dequant_q6_k(&enc);
        let scale = w.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        approx(&dec, &w, 0.06 * scale.max(1e-9));
    }

    /// Отмена: `quantize_gguf` по флагу возвращает Cancelled и НЕ оставляет
    /// файл-обрубок (панель/сервис полагаются на это при «Стоп»).
    #[test]
    fn quantize_gguf_cancel_leaves_no_file() {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
        let src = dir.join("mini-f16.gguf");
        let dst = std::env::temp_dir().join(format!(
            "quant-cancel-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let cancel = std::sync::atomic::AtomicBool::new(true);
        let mut progress = |_d: u64, _t: u64| {};
        let err = quantize_gguf(&src, &dst, QuantFormat::Q4_0, &mut progress, Some(&cancel))
            .expect_err("отмена должна вернуть ошибку");
        assert!(
            matches!(err, crate::error::BackendError::Cancelled),
            "ожидали Cancelled, получили {err:?}"
        );
        assert!(!dst.exists(), "обрубок {dst:?} не удалён");

        // Контроль: без отмены тот же путь даёт валидный GGUF и отчёт.
        let rep = quantize_gguf(&src, &dst, QuantFormat::Q4_0, &mut progress, None)
            .expect("квантование мини-фикстуры");
        assert!(dst.exists());
        assert!(rep.tensors > 0 && rep.bytes > 0);
        assert_eq!(rep.format, "Q4_0");
        // У мини-фикстуры token_embd [32, 64] — связанный эмбеддинг, который
        // политика хочет в Q6_K, но ne[0]=32 не кратен 256 → штатный
        // fallback Q8_0 (как у llama.cpp); норма (1-D) остаётся F16.
        assert_eq!(rep.warnings.len(), 1, "отклонения: {:?}", rep.warnings);
        assert!(
            rep.warnings[0].contains("не кратен 256"),
            "неожиданное отклонение: {:?}",
            rep.warnings
        );
        let mut hist: Vec<&str> = rep.by_type.iter().map(|(t, _, _)| *t).collect();
        hist.sort_unstable();
        assert_eq!(hist, vec!["Q4_0", "Q8_0"], "гистограмма {hist:?}");
        let bytes = std::fs::read(&dst).unwrap();
        let f = crate::gguf::parse(&bytes).unwrap();
        assert_eq!(f.tensors.len(), rep.tensors);
        assert_eq!(
            f.get("general.file_type").and_then(GgufValue::as_u32),
            Some(2)
        );
        let _ = std::fs::remove_file(&dst);
    }

    #[test]
    fn gguf_write_and_reparse() {
        let w: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let q = quantize(&w, QuantFormat::Q8_0);
        let mut buf = Vec::new();
        let mut prog = |_d: u64, _t: u64| {};
        write_gguf(
            &mut buf,
            &[
                ("general.architecture", MetaValue::String("test".into())),
                ("context_length", MetaValue::I32(128)),
                (
                    "tokenizer.ggml.tokens",
                    MetaValue::Array(vec![
                        GgufValue::String("<pad>".into()),
                        GgufValue::String("hello".into()),
                    ]),
                ),
            ],
            &[QuantTensor {
                name: "blk.0.attn_q.weight".to_string(),
                ne: vec![32, 2],
                ggml_type: GgmlType::Q8_0,
                data: q,
            }],
            &mut prog,
        )
        .unwrap();
        let file = parse(&buf).unwrap();
        assert_eq!(file.architecture(), Some("test"));
        assert_eq!(file.tensors.len(), 1);
        assert_eq!(file.tensors[0].name, "blk.0.attn_q.weight");
        assert_eq!(file.tensors[0].dims, vec![32, 2]);
        assert_eq!(file.tensors[0].data.len(), 68); // 64/32*34
        let w2 = dequant_gguf(file.tensors[0].ggml_type, file.tensors[0].data).unwrap();
        approx(&w2, &w, 0.3);
    }

    #[test]
    fn format_parse() {
        assert_eq!(QuantFormat::parse("q8_0"), Some(QuantFormat::Q8_0));
        assert_eq!(QuantFormat::parse("Q4_K_M"), Some(QuantFormat::Q4_K_M));
        assert_eq!(QuantFormat::parse("bf16"), Some(QuantFormat::Bf16));
        assert_eq!(QuantFormat::parse("iq4_xs"), Some(QuantFormat::Iq4Xs));
        assert_eq!(QuantFormat::parse("IQ4_XS"), Some(QuantFormat::Iq4Xs));
        assert_eq!(QuantFormat::parse("nope"), None);
    }

    /// IQ4_XS round-trip: энкодер → `dequant_iq4_xs` (хост-деквант из
    /// weights_map, сверенный побитово с ggml) → сравнение с входом.
    /// Ошибка кванта: шаг подблока ≈ max/32·(соседние кодовые точки), для
    /// плавного входа — доли от амплитуды.
    #[test]
    fn iq4_xs_roundtrip() {
        let w: Vec<f32> = (0..256)
            .map(|i| ((i as f32) * 0.07).sin() * 3.0 + ((i as f32) * 1.3).cos())
            .collect();
        let enc = quant_iq4_xs(&w);
        assert_eq!(enc.len(), 136);
        let deq = crate::weights_map::dequant_iq4_xs(&enc);
        assert_eq!(deq.len(), 256);
        let max_abs = w.iter().fold(0f32, |m, v| m.max(v.abs()));
        for (i, (a, b)) in deq.iter().zip(&w).enumerate() {
            // Нелинейный кодбук + 6-битная квантизация самого масштаба
            // подблока: у краевых точек ошибка до ~15% амплитуды (IQ4_XS
            // оптимизирован под гауссовы веса, не под синус). Побитовая
            // сверка с llama-quantize — ниже в скрипте приёмки.
            let tol = max_abs * 0.15 + 1e-3;
            assert!((a - b).abs() <= tol, "элемент {i}: {a} vs {b} (tol {tol})");
        }
    }

    /// Нулевой супер-блок: amax < GROUP_MAX_EPS → все scales=0, d = −0/32 =
    /// −0.0 (f16 0x8000 — ровно как у ggml, dh пишется из `-max_scale/32`),
    /// ls=32 → множитель 0 → деквант в нули.
    #[test]
    fn iq4_xs_zero_block() {
        let w = vec![0.0f32; 256];
        let enc = quant_iq4_xs(&w);
        assert_eq!(
            &enc[0..2],
            &[0x00, 0x80],
            "d = −0.0 (f16 0x8000), как у ggml"
        );
        let deq = crate::weights_map::dequant_iq4_xs(&enc);
        assert!(
            deq.iter().all(|v| *v == 0.0),
            "нулевой блок должен деквантоваться в нули"
        );
    }

    /// Layout-вектор: вручную собранный блок читается деквантером именно так
    /// (страховка от «оба конца одинаково неверны», ср. q4_0_canonical_layout).
    #[test]
    fn iq4_xs_layout_vector() {
        // d = 1.0. Подблок 0: ls = (scales_l[0]&0xF) | ((scales_h>>0)&3)<<4
        // = 1 | 16 = 17 → множитель d·(17−32) = −15; L=8 → kvalue 1 → y=−15.
        // Подблоки 1..7: ls = 32 (верхние 2 бита = 2) → множитель 0 → y=0.
        let mut data = vec![0u8; 136];
        data[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        data[4] = 0x01; // scales_l[0]: низшая четверка (ib=0) = 1
        let scales_h: u16 = 0b01
            | (0b10 << 2)
            | (0b10 << 4)
            | (0b10 << 6)
            | (0b10 << 8)
            | (0b10 << 10)
            | (0b10 << 12)
            | (0b10 << 14);
        data[2..4].copy_from_slice(&scales_h.to_le_bytes());
        for j in 0..16 {
            data[8 + j] = 8 | (8 << 4); // элементы 0..15 и 16..31 → L=8
        }
        let deq = crate::weights_map::dequant_iq4_xs(&data);
        for i in 0..32 {
            assert!((deq[i] - (-15.0)).abs() < 1e-4, "элемент {i}: {}", deq[i]);
        }
        assert!(deq[32..].iter().all(|v| *v == 0.0), "ls=32 → нули");
    }

    /// Кодбук и бинарный поиск: best_index_int8 на границах.
    #[test]
    fn iq4_xs_best_index() {
        assert_eq!(best_index_int8(&KVALUES_IQ4NL, -200.0), 0);
        assert_eq!(best_index_int8(&KVALUES_IQ4NL, 200.0), 15);
        assert_eq!(best_index_int8(&KVALUES_IQ4NL, 0.0), 8); // 0 ближе к 1, чем к −10
        assert_eq!(best_index_int8(&KVALUES_IQ4NL, -10.0), 7);
        assert_eq!(best_index_int8(&KVALUES_IQ4NL, 113.0), 15);
    }
}

// f16-битовая интерпретация для тестов (не тянем half в проде тестов)
#[cfg(test)]
mod f16 {
    pub fn from_le_bytes(b: [u8; 2]) -> f32 {
        let h = u16::from_le_bytes(b);
        let sign = ((h >> 15) & 1) as u32;
        let exp = ((h >> 10) & 0x1F) as u32;
        let frac = (h & 0x3FF) as u32;
        let bits = if exp == 0 {
            if frac == 0 {
                sign << 31
            } else {
                let mut e = 127i32 - 15 + 1;
                let mut f = frac;
                while f & 0x400 == 0 {
                    f <<= 1;
                    e -= 1;
                }
                (sign << 31) | ((e as u32) << 23) | ((f & 0x3FF) << 13)
            }
        } else if exp == 31 {
            (sign << 31) | 0x7F80_0000 | (frac << 13)
        } else {
            (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
        };
        f32::from_bits(bits)
    }
}
