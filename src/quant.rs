//! Квантование и конверсия моделей (Ф4.12): safetensors → GGUF во всех
//! наших форматах (Q4_0/Q8_0/Q4_K_M/Q6_K/BF16/F16/F32) + GGUF re-quant.
//!
//! Раскладки блоков — канонический ggml (дешифраторы совместимы с реальными
//! llama.cpp-файлами):
//! Q4_0 {f16 d, 16×u8 нибблы} = 18 Б/32; Q8_0 {f16 d, 32×i8} = 34 Б/32;
//! Q4_K {f16 d, f16 dmin, u8 scales[12], u8 qs[128]} = 144 Б/256
//! (упаковка scales — get_scale_min_k4, как в Q5_K; y = d·sc·q − dmin·m);
//! Q6_K {u8 ql[128], u8 qh[64], i8 scales[16], f16 d} = 210 Б/256
//! (d в КОНЦЕ блока, байты 208..210; y = d·sc·(q − 32)).

use std::io::Write;
use std::path::Path;

use crate::error::BackendError;
use crate::gguf::{GgmlType, GgufValue, parse};
use crate::weights::{
    read_safetensors_tensor_f32, safetensors_tensor_names, safetensors_tensor_numel,
};
use crate::weights_map::dequant_gguf;

/// Форматы вывода конверсии.
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
        }
    }

    /// Блокировать ли данные (для конверсии: только квантованные форматы
    /// требуют кратности блоку — F32/F16/BF16 — нет).
    fn needs_block_pad(self) -> bool {
        matches!(self, Self::Q4_0 | Self::Q8_0 | Self::Q4_K_M)
    }
}

// ── битовые конвертеры ──────────────────────────────────────────────────────

pub fn f16_bits(x: f32) -> [u8; 2] {
    let b = x.to_bits();
    let sign = (b >> 16) & 0x8000;
    let exp = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = (b >> 13) & 0x3FF;
    let h = if exp <= 0 {
        sign
    } else if exp >= 31 {
        sign | 0x7C00
    } else {
        sign | ((exp as u32) << 10) | mant
    };
    let bytes = h.to_le_bytes();
    [bytes[0], bytes[1]]
}

pub fn bf16_bits(x: f32) -> [u8; 2] {
    // round-to-nearest-even на 16 бит
    let b = x.to_bits();
    let rounding = ((b >> 16) & 1) + 0x7FFF;
    let bytes = ((b + rounding) >> 16).to_le_bytes();
    [bytes[0], bytes[1]]
}

// ── квант-энкодеры (f32 → байты формата) ───────────────────────────────────

/// Q4_0: блок 32 → f16 d + 16 байт нибблов; y = (q - 8) * d.
pub fn quant_q4_0(weights: &[f32]) -> Vec<u8> {
    assert!(
        weights.len().is_multiple_of(32),
        "Q4_0: len % 32 == 0, got {}",
        weights.len()
    );
    let mut out = Vec::with_capacity(weights.len() / 32 * 18);
    for blk in weights.chunks_exact(32) {
        let max_abs = blk.iter().fold(0.0f32, |m, w| m.max(w.abs()));
        let d = if max_abs > 0.0 { max_abs / 8.0 } else { 0.0 };
        out.extend_from_slice(&f16_bits(d));
        for pair in blk.chunks_exact(2) {
            let q0 = ((pair[0] / d).round() as i32 + 8).clamp(0, 15) as u8;
            let q1 = ((pair[1] / d).round() as i32 + 8).clamp(0, 15) as u8;
            out.push((q0 & 0x0F) | ((q1 & 0x0F) << 4));
        }
    }
    out
}

/// Q8_0: блок 32 → f16 d + 32 i8; y = q * d.
pub fn quant_q8_0(weights: &[f32]) -> Vec<u8> {
    assert!(weights.len().is_multiple_of(32), "Q8_0: len % 32 == 0");
    let mut out = Vec::with_capacity(weights.len() / 32 * 34);
    for blk in weights.chunks_exact(32) {
        let max_abs = blk.iter().fold(0.0f32, |m, w| m.max(w.abs()));
        let d = if max_abs > 0.0 { max_abs / 127.0 } else { 0.0 };
        out.extend_from_slice(&f16_bits(d));
        for w in blk {
            out.push((w / d).round().clamp(-127.0, 127.0) as i8 as u8);
        }
    }
    out
}

/// Канонический Q4_K (ggml block_q4_K): блок 256 → 144 байта:
///   d (f16, 0..2), dmin (f16, 2..4), scales[12] (4..16 — 6-битные
///   scale/min 8 подблоков по 32, упаковка get_scale_min_k4 — та же, что в
///   Q5_K), qs[128] (16..144 — 4-битные значения, чанк-мажор: байт 32c+l
///   держит {low: 64c+l, high: 64c+32+l}, 4 чанка по 64).
/// Деквант (ggml dequantize_row_q4_K): y = d·sc·q − dmin·m, q ∈ 0..15.
/// Схема квантования — упрощённый ggml (без итерационного уточнения
/// make_qkx2_quants): на подблок raw_sc = (max−min(min,0))/15, глобальные
/// d = max(raw_sc)/63, dmin = max(−min(min,0))/63; значение q = round((x+ml)/dl),
/// где dl = d·sc, ml = dmin·m. Энкодер не побитово равен llama.cpp, но
/// РАСКЛАДКА каноническая — блок декодируется любым ggml-ридером.
pub fn quant_q4_k(weights: &[f32]) -> Vec<u8> {
    assert!(weights.len().is_multiple_of(256), "Q4_K: len % 256 == 0");
    let mut out = Vec::with_capacity(weights.len() / 256 * 144);
    for blk in weights.chunks_exact(256) {
        let mut raw_sc = [0f32; 8];
        let mut raw_min = [0f32; 8];
        for b in 0..8 {
            let sub = &blk[b * 32..(b + 1) * 32];
            let max = sub.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min = sub.iter().cloned().fold(f32::INFINITY, f32::min);
            // Низ диапазона: min(min, 0). В формате m ≥ 0 (деквант вычитает
            // dmin·m ≥ 0), поэтому подблок с ПОЛОЖИТЕЛЬНЫМ min кодируется от
            // нуля — иначе весь DC подблока теряется (q клипается в 15).
            let low = min.min(0.0);
            raw_sc[b] = (max - low) / 15.0;
            raw_min[b] = low;
        }
        let mut d = raw_sc.iter().cloned().fold(0.0f32, f32::max) / 63.0;
        let mut dmin = raw_min.iter().map(|m| m.abs()).fold(0.0f32, f32::max) / 63.0;
        // Нулевой блок: d = 0 — деквант даёт 0 при любых q (мусор в битах не
        // важен); крошечный знаменатель ниже исключает деление на 0.
        if d == 0.0 {
            d = 1e-8;
        }
        if dmin == 0.0 {
            dmin = 1e-8;
        }
        out.extend_from_slice(&f16_bits(d));
        out.extend_from_slice(&f16_bits(dmin));
        let mut sc6 = [0u8; 8];
        let mut mn6 = [0u8; 8];
        for j in 0..8 {
            sc6[j] = ((raw_sc[j] / d).round() as i32).clamp(0, 63) as u8;
            // ggml хранит min КАК −min (положительное); деквант ВЫЧИТАЕТ
            // dmin·m. Для подблока с min > 0 код mn6 = 0 (сатурация — та же
            // особенность, что у Q5_K/q5k_encode: верх диапазона клипается).
            mn6[j] = ((-raw_min[j] / dmin).round() as i32).clamp(0, 63) as u8;
        }
        // scales[12]: упаковка get_scale_min_k4 (та же, что в Q5_K):
        //   j<4:  scales[j] = ls;          scales[j+4] = lm
        //   j>=4: scales[j+4] = (ls&0xF) | ((lm&0xF)<<4);
        //         scales[j-4] |= (ls>>4)<<6;  scales[j] |= (lm>>4)<<6
        let mut scales = [0u8; 12];
        for j in 0..8 {
            let ls = sc6[j];
            let lm = mn6[j];
            if j < 4 {
                scales[j] = ls & 63;
                scales[j + 4] = lm & 63;
            } else {
                scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                scales[j - 4] |= (ls >> 4) << 6;
                scales[j] |= (lm >> 4) << 6;
            }
        }
        out.extend_from_slice(&scales);
        // qs[128]: q = round((x + ml)/dl), dl = d·sc6[b], ml = dmin·mn6[b];
        // чанк-мажор: байт 32c+l = {low: значение 64c+l, high: 64c+32+l}.
        let mut qv = [0u8; 256];
        for b in 0..8 {
            let sub = &blk[b * 32..(b + 1) * 32];
            let dl = d * sc6[b] as f32;
            let ml = dmin * mn6[b] as f32;
            for (i, &x) in sub.iter().enumerate() {
                qv[b * 32 + i] = if dl > 0.0 {
                    ((x + ml) / dl).round().clamp(0.0, 15.0) as u8
                } else {
                    0
                };
            }
        }
        for c in 0..4 {
            for l in 0..32 {
                let lo = qv[64 * c + l] & 0x0F;
                let hi = (qv[64 * c + 32 + l] & 0x0F) << 4;
                out.push(lo | hi);
            }
        }
    }
    out
}

/// Канонический Q6_K (ggml block_q6_K): блок 256 → 210 байт: ql[128]
/// (низкие 4 бита, по байту l: {low: q(l), high: q(l+64)} и {low: q(l+32),
/// high: q(l+96)}), qh[64] (старшие 2 бита, байт l: q(l)|q(l+32)<<2|
/// q(l+64)<<4|q(l+96)<<6 на каждую 128-половину), scales[16] (int8 на
/// 16-элементные подблоки), d (f16) в КОНЦЕ, байты 208..210. Деквант:
/// y = d·sc·(q − 32), q ∈ 0..63. Схема — как в speech-крейте
/// (librust-inference-speech quant::quant_q6k_block): симметрично, без
/// zero-point, 6-битное сведение подблоковых scale к d.
pub fn quant_q6_k(weights: &[f32]) -> Vec<u8> {
    assert!(weights.len().is_multiple_of(256), "Q6_K: len % 256 == 0");
    let mut out = Vec::with_capacity(weights.len() / 256 * 210);
    for blk in weights.chunks_exact(256) {
        // Подблоковые scale (16×16): L-стейджинг [0, 63] как в make_qx_quants.
        let mut l = [0i8; 256];
        let mut scales = [0f32; 16];
        let mut max_abs_scale = 0f32;
        let mut max_scale = 0f32;
        for ib in 0..16 {
            let sub = &blk[ib * 16..(ib + 1) * 16];
            let mut amax = 0f32;
            let mut max = 0f32;
            for &v in sub {
                let av = v.abs();
                if av > amax {
                    amax = av;
                    max = v;
                }
            }
            let scale = if amax < 1e-15 {
                0.0
            } else {
                let iscale = -32.0 / max;
                for (i, &v) in sub.iter().enumerate() {
                    let q = (iscale * v).round_ties_even().clamp(-32.0, 31.0) as i8;
                    l[ib * 16 + i] = q + 32;
                }
                -max / 32.0
            };
            scales[ib] = scale;
            if scale.abs() > max_abs_scale {
                max_abs_scale = scale.abs();
                max_scale = scale;
            }
        }
        if max_abs_scale < 1e-15 {
            // Нулевой блок: d = 0, ql/qh/scales остаются нулями (210 байт).
            out.extend_from_slice(&[0u8; 210]);
            continue;
        }
        let iscale = -128.0 / max_scale;
        let d = 1.0 / iscale;
        out.resize(out.len() + 208, 0);
        let base = out.len() - 208;
        out.extend_from_slice(&f16_bits(d));
        for j in 0..16 {
            let sc = (iscale * scales[j]).round_ties_even() as i32;
            out[base + 192 + j] = (sc.min(127) as i8) as u8;
            let dj = d * sc as f32;
            if dj == 0.0 {
                continue;
            }
            for ii in 0..16 {
                let q = (blk[j * 16 + ii] / dj).round_ties_even().clamp(-32.0, 31.0) as i8;
                l[j * 16 + ii] = q + 32;
            }
        }
        // Упаковка ql/qh (две 128-половины).
        for half in 0..2 {
            let b = half * 128;
            for k in 0..32 {
                let q1 = l[b + k] & 0xF;
                let q2 = l[b + k + 32] & 0xF;
                let q3 = l[b + k + 64] & 0xF;
                let q4 = l[b + k + 96] & 0xF;
                out[base + half * 64 + k] = (q1 | (q3 << 4)) as u8;
                out[base + half * 64 + k + 32] = (q2 | (q4 << 4)) as u8;
                out[base + 128 + half * 32 + k] = (((l[b + k] >> 4) & 0x3)
                    | (((l[b + k + 32] >> 4) & 0x3) << 2)
                    | (((l[b + k + 64] >> 4) & 0x3) << 4)
                    | (((l[b + k + 96] >> 4) & 0x3) << 6)) as u8;
            }
        }
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
    }
}

// ── GGUF-райтер ─────────────────────────────────────────────────────────────

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF"
const GGUF_VERSION: u32 = 3;
const TENSOR_ALIGN: u64 = 32;
// Типы GGUF v3 (как в парсере gguf.rs / llama.cpp): U8=0, I8=1, U16=2,
// I16=3, U32=4, I32=5, F32=6, BOOL=7, STRING=8, ARRAY=9, U64=10, I64=11,
// F64=12.
const GGUF_TYPE_U8: u32 = 0;
const GGUF_TYPE_INT32: u32 = 5;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_STRING: u32 = 8;
const GGUF_TYPE_ARRAY: u32 = 9;

/// Значение метаданных для записи (v1: U8/I32/F32/STRING/ARRAY-STRING).
#[derive(Debug, Clone)]
pub enum MetaValue {
    U8(u8),
    I32(i32),
    F32(f32),
    Str(String),
    StrArray(Vec<String>),
}

fn write_string(w: &mut Vec<u8>, s: &str) {
    w.extend_from_slice(&(s.len() as u64).to_le_bytes());
    w.extend_from_slice(s.as_bytes());
}

fn meta_bytes(v: &MetaValue) -> (u32, Vec<u8>) {
    match v {
        MetaValue::U8(x) => (GGUF_TYPE_U8, vec![*x]),
        MetaValue::I32(x) => (GGUF_TYPE_INT32, x.to_le_bytes().to_vec()),
        MetaValue::F32(x) => (GGUF_TYPE_FLOAT32, x.to_le_bytes().to_vec()),
        MetaValue::Str(s) => {
            let mut b = Vec::new();
            write_string(&mut b, s);
            (GGUF_TYPE_STRING, b)
        }
        MetaValue::StrArray(a) => {
            let mut b = Vec::new();
            b.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
            b.extend_from_slice(&(a.len() as u64).to_le_bytes());
            for s in a {
                write_string(&mut b, s);
            }
            (GGUF_TYPE_ARRAY, b)
        }
    }
}

/// Тензор для записи: имя, dims (ne-порядок: fastest first), тип, данные.
pub struct QuantTensor {
    pub name: String,
    /// ne-порядок (fastest dim первый) — как в GGUF-файле.
    pub ne: Vec<u64>,
    pub ggml_type: GgmlType,
    pub data: Vec<u8>,
}

/// Записать GGUF v3: header + metadata + infos + данные (с прогрессом).
/// Данные пишутся одним проходом после расчёта оффсетов (без хранения
/// второй копии секций).
pub fn write_gguf<W: Write>(
    w: &mut W,
    metadata: &[(&str, MetaValue)],
    tensors: &[QuantTensor],
    progress: &mut dyn FnMut(u64, u64),
) -> Result<(), BackendError> {
    let mut header = Vec::new();
    header.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    header.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    header.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    header.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for (k, v) in metadata {
        write_string(&mut header, k);
        let (vt, vb) = meta_bytes(v);
        header.extend_from_slice(&vt.to_le_bytes());
        header.extend_from_slice(&vb);
    }

    // infos: размер фиксирован (не зависит от оффсетов) — считаем дважды
    let mut infos = Vec::new();
    for t in tensors {
        write_string(&mut infos, &t.name);
        infos.extend_from_slice(&(t.ne.len() as u32).to_le_bytes());
        for d in &t.ne {
            infos.extend_from_slice(&d.to_le_bytes());
        }
        infos.extend_from_slice(&t.ggml_type.raw().to_le_bytes());
        infos.extend_from_slice(&0u64.to_le_bytes()); // placeholder
    }

    let prefix_len = header.len() + infos.len();
    let mut data_start = prefix_len as u64;
    while !data_start.is_multiple_of(TENSOR_ALIGN) {
        data_start += 1;
    }
    // реальные оффсеты (от начала секции данных)
    let mut offsets = Vec::with_capacity(tensors.len());
    let mut cur = 0u64;
    for t in tensors {
        while !cur.is_multiple_of(TENSOR_ALIGN) {
            cur += 1;
        }
        offsets.push(cur);
        cur += t.data.len() as u64;
    }
    let total = cur;

    // infos с реальными оффсетами
    let mut infos2 = Vec::new();
    for (t, off) in tensors.iter().zip(&offsets) {
        write_string(&mut infos2, &t.name);
        infos2.extend_from_slice(&(t.ne.len() as u32).to_le_bytes());
        for d in &t.ne {
            infos2.extend_from_slice(&d.to_le_bytes());
        }
        infos2.extend_from_slice(&t.ggml_type.raw().to_le_bytes());
        infos2.extend_from_slice(&off.to_le_bytes());
    }
    debug_assert_eq!(infos2.len(), infos.len(), "infos size must be stable");

    w.write_all(&header)
        .and_then(|_| w.write_all(&infos2))
        .map_err(|e| BackendError::Native(format!("gguf write: {e}")))?;
    let pad = (data_start as usize).saturating_sub(prefix_len);
    if pad > 0 {
        w.write_all(&vec![0u8; pad])
            .map_err(|e| BackendError::Native(format!("gguf write: {e}")))?;
    }
    // данные с межтензорными паддингами (оффсеты считались с выравниванием 32)
    let mut done = 0u64;
    for t in tensors {
        while !done.is_multiple_of(TENSOR_ALIGN) {
            w.write_all(&[0u8])
                .map_err(|e| BackendError::Native(format!("gguf write: {e}")))?;
            done += 1;
        }
        w.write_all(&t.data)
            .map_err(|e| BackendError::Native(format!("gguf write: {e}")))?;
        done += t.data.len() as u64;
        progress(done, total);
    }
    Ok(())
}

// ── конверсия safetensors → GGUF ────────────────────────────────────────────

/// Отчёт конверсии.
#[derive(Debug, Clone)]
pub struct QuantReport {
    pub arch: String,
    pub name: String,
    pub tensors: usize,
    pub bytes: u64,
    pub format: &'static str,
}

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
        ("general.architecture", MetaValue::Str(arch.clone())),
        ("general.name", MetaValue::Str(name.clone())),
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
    })
}

fn file_type_of(fmt: QuantFormat) -> i32 {
    match fmt {
        QuantFormat::F32 => 0,
        QuantFormat::F16 => 1,
        QuantFormat::Q4_0 => 2,
        QuantFormat::Q8_0 => 7,
        QuantFormat::Q4_K_M => 15,
        QuantFormat::Bf16 => 29,
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

// ── GGUF re-quant ───────────────────────────────────────────────────────────

/// Переквантовать GGUF в другой формат (те же метаданные и имена).
pub fn requant_gguf(
    src: &Path,
    dst: &Path,
    fmt: QuantFormat,
    progress: &mut dyn FnMut(u64, u64),
) -> Result<QuantReport, BackendError> {
    let bytes = std::fs::read(src)
        .map_err(|e| BackendError::Native(format!("read {}: {e}", src.display())))?;
    let file = parse(&bytes).map_err(|e| BackendError::Native(format!("gguf parse: {e}")))?;
    let arch = file.architecture().unwrap_or("unknown").to_string();
    let ggml_type = fmt.ggml_type();

    let mut tensors: Vec<QuantTensor> = Vec::with_capacity(file.tensors.len());
    let total_elems: u64 = file.tensors.iter().map(|t| t.numel()).sum();
    for t in &file.tensors {
        let w = dequant_gguf(t.ggml_type, t.data)?;
        let numel = w.len();
        if fmt.needs_block_pad() {
            let block = ggml_type.block_elems() as usize;
            if !numel.is_multiple_of(block) && numel > block {
                return Err(BackendError::Shape(format!(
                    "{}: numel {numel} не кратен блоку {block} для {fmt:?}",
                    t.name
                )));
            }
        }
        let q = quantize(&w, fmt);
        tensors.push(QuantTensor {
            name: t.name.clone(),
            ne: t.dims.clone(),
            ggml_type,
            data: q,
        });
    }
    // метаданные переносим как есть (строки/числа — через GgufValue)
    let mut meta: Vec<(&str, MetaValue)> = Vec::new();
    for (k, v) in &file.metadata {
        let mv = match v {
            GgufValue::String(s) => MetaValue::Str(s.clone()),
            GgufValue::U8(x) => MetaValue::U8(*x),
            GgufValue::I32(x) => MetaValue::I32(*x),
            GgufValue::F32(x) => MetaValue::F32(*x),
            GgufValue::Array(a) => {
                let strs: Option<Vec<String>> =
                    a.iter().map(|x| x.as_str().map(String::from)).collect();
                match strs {
                    Some(s) => MetaValue::StrArray(s),
                    None => continue, // нестроковые массивы пропускаем (v1)
                }
            }
            _ => continue,
        };
        meta.push((k, mv));
    }
    let mut file_out = std::fs::File::create(dst)
        .map_err(|e| BackendError::Native(format!("create {}: {e}", dst.display())))?;
    let total_bytes: u64 = tensors.iter().map(|t| t.data.len() as u64).sum();
    let mut p = |d: u64, _t: u64| {
        let frac = if total_bytes > 0 {
            d as f64 / total_bytes as f64
        } else {
            0.0
        };
        progress((frac * total_elems as f64) as u64, total_elems);
    };
    write_gguf(&mut file_out, &meta, &tensors, &mut p)?;
    let name = file
        .get("general.name")
        .and_then(|v| v.as_str())
        .unwrap_or("model")
        .to_string();
    Ok(QuantReport {
        arch,
        name,
        tensors: tensors.len(),
        bytes: total_bytes,
        format: fmt.name(),
    })
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
        let mut out = Vec::with_capacity(n);
        for blk in data.chunks_exact(18) {
            let d = f32::from(f16::from_le_bytes([blk[0], blk[1]]));
            for byte in &blk[2..] {
                let lo = byte & 0x0F;
                let hi = byte >> 4;
                out.push((lo as i8 - 8) as f32 * d);
                out.push((hi as i8 - 8) as f32 * d);
            }
        }
        out.truncate(n);
        out
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
            let d = f32::from(f16::from_le_bytes([blk[0], blk[1]]));
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

    #[test]
    fn q6_k_size_and_roundtrip() {
        let w: Vec<f32> = (0..256).map(|i| ((i as f32 * 1.7).sin()) * 0.3).collect();
        let enc = quant_q6_k(&w);
        // Канонический блок: 128+64+16+2 = 210 байт, d в хвосте.
        assert_eq!(enc.len(), 210);
        let d = f32::from(f16::from_le_bytes([enc[208], enc[209]]));
        assert!(d != 0.0, "super-block scale must sit at bytes 208..210");
        // Roundtrip через канонический декодер (weights_map).
        let dec = crate::weights_map::dequant_q6_k(&enc);
        let scale = w.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        approx(&dec, &w, 0.06 * scale.max(1e-9));
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
                ("general.architecture", MetaValue::Str("test".into())),
                ("context_length", MetaValue::I32(128)),
                (
                    "tokenizer.ggml.tokens",
                    MetaValue::StrArray(vec!["<pad>".into(), "hello".into()]),
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
        let _ = approx(&w2, &w, 0.3);
    }

    #[test]
    fn format_parse() {
        assert_eq!(QuantFormat::parse("q8_0"), Some(QuantFormat::Q8_0));
        assert_eq!(QuantFormat::parse("Q4_K_M"), Some(QuantFormat::Q4_K_M));
        assert_eq!(QuantFormat::parse("bf16"), Some(QuantFormat::Bf16));
        assert_eq!(QuantFormat::parse("nope"), None);
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
