//! GGUF-контейнер (v2/v3): парсер metadata + тензорной таблицы (Ф1.1).
//!
//! Читает из `&[u8]` без копий (caller mmap'ит файл) — парсер держит только
//! слайсы. Поддерживаются все квант-типы, нужные движку: F32/F16/BF16,
//! Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1, Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K.
//!
//! ВАЖНО: размеры K-блоков — канонические ggml-раскладки (совместимы с
//! реальными llama.cpp-файлами): Q4_K = {f16 d, f16 dmin, scales[12]
//! (get_scale_min_k4), qs[128]} = 144 Б/256; Q6_K = {ql[128], qh[64],
//! scales[16] i8, f16 d (в хвосте)} = 210 Б/256. (До волны ревью 8 Q4_K
//! был кастомным 148-байтным диалектом — не читался llama.cpp.)

use std::fmt;

/// Магическая сигнатура "GGUF" (LE u32).
pub const GGUF_MAGIC: u32 = 0x4655_4747;

/// Ошибка парсинга GGUF. Все ошибки — `Err`, паник на битых файлах нет.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GgufError {
    BadMagic,
    UnsupportedVersion(u32),
    Truncated(&'static str),
    InvalidValueType(u32),
    InvalidTensorType(u32),
    TensorCountTooLarge(u64),
    MetadataCountTooLarge(u64),
    ArrayTooLarge(u64),
    DimsTooLarge(u32),
    NumelOverflow(String),
    UnalignedTensor {
        name: String,
        offset: u64,
    },
    TensorOverflow {
        name: String,
        offset: u64,
        size: u64,
    },
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic => write!(f, "not a GGUF file (bad magic)"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported GGUF version {v}"),
            Self::Truncated(what) => write!(f, "truncated file while reading {what}"),
            Self::InvalidValueType(t) => write!(f, "invalid GGUF value type {t}"),
            Self::InvalidTensorType(t) => write!(f, "invalid GGML tensor type {t}"),
            Self::TensorCountTooLarge(n) => write!(f, "unreasonable tensor count {n}"),
            Self::MetadataCountTooLarge(n) => write!(f, "unreasonable metadata kv count {n}"),
            Self::ArrayTooLarge(n) => write!(f, "unreasonable array length {n}"),
            Self::DimsTooLarge(n) => write!(f, "unreasonable n_dims {n}"),
            Self::NumelOverflow(name) => write!(f, "numel overflow in tensor `{name}`"),
            Self::UnalignedTensor { name, offset } => {
                write!(f, "tensor `{name}` data offset {offset} not 32-aligned")
            }
            Self::TensorOverflow { name, offset, size } => {
                write!(
                    f,
                    "tensor `{name}` data [{offset}, {}) overflows file",
                    offset + size
                )
            }
        }
    }
}

impl std::error::Error for GgufError {}

/// Разумные лимиты (защита от мусорных счётчиков в битых файлах).
const MAX_TENSORS: u64 = 1 << 20;
const MAX_KV: u64 = 1 << 24;
const MAX_ARRAY_ITEMS: u64 = 1 << 24;
const MAX_DIMS: u32 = 8;

/// Тип значения metadata (GGUF value types).
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::U32(v) => Some(*v),
            Self::I32(v) => Some(*v as u32),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            Self::Array(a) => Some(a),
            _ => None,
        }
    }
}

/// Тип тензора (GGML_TYPE).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlType {
    F32,
    F16,
    Bf16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    /// i-кванты (A4): пока только IQ4_XS — код 23 в GGUF (ggml.h эталонного
    /// коммита). 256 элементов на суперблок, 136 байт, 4.25 бита на вес.
    Iq4Xs,
}

impl GgmlType {
    pub fn from_raw(raw: u32) -> Option<Self> {
        Some(match raw {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            23 => Self::Iq4Xs,
            30 => Self::Bf16,
            _ => return None,
        })
    }

    /// Код типа в GGUF-файле (обратный к `from_raw`; дискриминанты enum
    /// НЕ совпадают с кодами — для записи использовать этот метод).
    pub fn raw(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Q8K => 15,
            Self::Iq4Xs => 23,
            // BF16 = 30 по ggml.h эталонного коммита (27 — это I64);
            // сверено с реальным llama.cpp-файлом (huihui-ai BF16-шейды).
            Self::Bf16 => 30,
        }
    }

    /// Элементов в блоке (32 для Q-блоков, 256 для K-блоков, 1 для float).
    pub fn block_elems(self) -> u32 {
        match self {
            Self::F32 | Self::F16 | Self::Bf16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K | Self::Iq4Xs => 256,
        }
    }

    /// Байт в блоке (см. док-комментарий модуля: K-размеры — по ggml-раскладкам,
    /// golden-сверка в Ф1.1).
    pub fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 40,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 292,
            // IQ4_XS: f16-масштаб d + 2 Б (scales_h) + 4 Б (scales_l) + 128 Б
            // нибблов = 136 Б на 256 весов (4.25 бита); раскладка ggml
            // block_iq4_xs, сверка с llama-quantize/dequant — в тестах.
            Self::Iq4Xs => 136,
        }
    }

    /// Байт на один элемент (для размера данных тензора).
    pub fn bytes_per_element(self, numel: u64) -> u64 {
        let elems = self.block_elems() as u64;
        let bytes = self.block_bytes() as u64;
        numel.div_ceil(elems).saturating_mul(bytes)
    }

    /// Имя типа в нотации ggml (для отчётов и логов: "Q4_K", "Q6_K", …).
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8K => "Q8_K",
            Self::Iq4Xs => "IQ4_XS",
        }
    }
}

/// Тензор: имя, dims (ne-порядок, fastest first), тип, слайс данных.
#[derive(Debug, Clone, PartialEq)]
pub struct GgufTensor<'a> {
    pub name: String,
    /// dims в GGUF-порядке (ne: fastest dim первый).
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    /// Смещение данных от начала файла.
    pub offset: u64,
    pub data: &'a [u8],
}

impl<'a> GgufTensor<'a> {
    /// Число элементов (произведение dims).
    pub fn numel(&self) -> u64 {
        self.dims.iter().product()
    }
    /// Размер данных в байтах.
    pub fn data_bytes(&self) -> u64 {
        self.ggml_type.bytes_per_element(self.numel())
    }
}

/// Разобранный GGUF: metadata + тензоры.
#[derive(Debug, Clone, PartialEq)]
pub struct GgufFile<'a> {
    pub version: u32,
    pub metadata: Vec<(String, GgufValue)>,
    pub tensors: Vec<GgufTensor<'a>>,
}

impl<'a> GgufFile<'a> {
    /// Значение metadata по ключу.
    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Архитектура модели (`general.architecture`).
    pub fn architecture(&self) -> Option<&str> {
        self.get("general.architecture").and_then(GgufValue::as_str)
    }
}

// ── Парсер ──────────────────────────────────────────────────────────────────

const V2: u32 = 2;
const V3: u32 = 3;
const TENSOR_ALIGN: u64 = 32;

fn read_u32(buf: &[u8], pos: &mut usize) -> Result<u32, GgufError> {
    let end = pos.checked_add(4).ok_or(GgufError::Truncated("u32"))?;
    let bytes = buf.get(*pos..end).ok_or(GgufError::Truncated("u32"))?;
    *pos = end;
    Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
}

fn read_u64(buf: &[u8], pos: &mut usize) -> Result<u64, GgufError> {
    let end = pos.checked_add(8).ok_or(GgufError::Truncated("u64"))?;
    let bytes = buf.get(*pos..end).ok_or(GgufError::Truncated("u64"))?;
    *pos = end;
    Ok(u64::from_le_bytes(bytes.try_into().unwrap()))
}

fn read_string(buf: &[u8], pos: &mut usize) -> Result<String, GgufError> {
    let len = read_u64(buf, pos)? as usize;
    let end = pos.checked_add(len).ok_or(GgufError::Truncated("string"))?;
    let bytes = buf.get(*pos..end).ok_or(GgufError::Truncated("string"))?;
    *pos = end;
    Ok(String::from_utf8_lossy(bytes).into_owned())
}

fn read_value(buf: &[u8], pos: &mut usize, vtype: u32) -> Result<GgufValue, GgufError> {
    match vtype {
        0 => {
            // P18b: U8 must advance the position like every other type —
            // the old code read the byte but did NOT `*pos += 1`, so any
            // file with a U8 metadata value desynced the whole metadata
            // block (wrong keys/values or a bogus Truncated).
            let v = buf.get(*pos..*pos + 1).ok_or(GgufError::Truncated("u8"))?[0];
            *pos += 1;
            Ok(GgufValue::U8(v))
        }
        1 => {
            let v = buf.get(*pos..*pos + 1).ok_or(GgufError::Truncated("i8"))?[0] as i8;
            *pos += 1;
            Ok(GgufValue::I8(v))
        }
        2 => {
            let v = u16::from_le_bytes(
                buf.get(*pos..*pos + 2)
                    .ok_or(GgufError::Truncated("u16"))?
                    .try_into()
                    .unwrap(),
            );
            *pos += 2;
            Ok(GgufValue::U16(v))
        }
        3 => {
            let v = i16::from_le_bytes(
                buf.get(*pos..*pos + 2)
                    .ok_or(GgufError::Truncated("i16"))?
                    .try_into()
                    .unwrap(),
            );
            *pos += 2;
            Ok(GgufValue::I16(v))
        }
        4 => Ok(GgufValue::U32(read_u32(buf, pos)?)),
        5 => Ok(GgufValue::I32(read_u32(buf, pos)? as i32)),
        6 => Ok(GgufValue::F32(f32::from_bits(read_u32(buf, pos)?))),
        7 => {
            let v = buf
                .get(*pos..*pos + 1)
                .ok_or(GgufError::Truncated("bool"))?[0];
            *pos += 1;
            Ok(GgufValue::Bool(v != 0))
        }
        8 => Ok(GgufValue::String(read_string(buf, pos)?)),
        9 => {
            let elem_type = read_u32(buf, pos)?;
            let count = read_u64(buf, pos)?;
            if count > MAX_ARRAY_ITEMS {
                return Err(GgufError::ArrayTooLarge(count));
            }
            let mut items = Vec::with_capacity(count.min(4096) as usize);
            for _ in 0..count {
                items.push(read_value(buf, pos, elem_type)?);
            }
            Ok(GgufValue::Array(items))
        }
        10 => Ok(GgufValue::U64(read_u64(buf, pos)?)),
        11 => Ok(GgufValue::I64(read_u64(buf, pos)? as i64)),
        12 => {
            let lo = read_u32(buf, pos)?;
            let hi = read_u32(buf, pos)?;
            Ok(GgufValue::F64(f64::from_bits(
                ((hi as u64) << 32) | lo as u64,
            )))
        }
        other => Err(GgufError::InvalidValueType(other)),
    }
}

/// Разобрать GGUF из слайса байт (caller: mmap файла).
pub fn parse(buf: &[u8]) -> Result<GgufFile<'_>, GgufError> {
    let mut pos = 0usize;
    let magic = read_u32(buf, &mut pos)?;
    if magic != GGUF_MAGIC {
        return Err(GgufError::BadMagic);
    }
    let version = read_u32(buf, &mut pos)?;
    if !(V2..=V3).contains(&version) {
        return Err(GgufError::UnsupportedVersion(version));
    }
    let tensor_count = read_u64(buf, &mut pos)?;
    if tensor_count > MAX_TENSORS {
        return Err(GgufError::TensorCountTooLarge(tensor_count));
    }
    let kv_count = read_u64(buf, &mut pos)?;
    if kv_count > MAX_KV {
        return Err(GgufError::MetadataCountTooLarge(kv_count));
    }

    let mut metadata = Vec::with_capacity(kv_count.min(4096) as usize);
    for _ in 0..kv_count {
        let key = read_string(buf, &mut pos)?;
        let vtype = read_u32(buf, &mut pos)?;
        let value = read_value(buf, &mut pos, vtype)?;
        metadata.push((key, value));
    }

    // Tensor infos: оффсеты в GGUF ОТНОСИТЕЛЬНЫ начала секции данных
    // (llama.cpp: data_offset + tensor.offset). Секция данных начинается
    // сразу после таблицы тензоров, с выравниванием `general.alignment`
    // (32 по умолчанию). Файл, собранный с 64, читался со смещением: парсер
    // отдавал zero-padding вместо весов, и модель деградировала молча.
    let align = metadata
        .iter()
        .find(|(k, _)| k == "general.alignment")
        .and_then(|(_, v)| v.as_u32())
        .map(u64::from)
        .filter(|a| *a > 0 && a.is_power_of_two())
        .unwrap_or(TENSOR_ALIGN);
    struct RawInfo {
        name: String,
        dims: Vec<u64>,
        ggml_type: GgmlType,
        rel_offset: u64,
    }
    let mut raw = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = read_string(buf, &mut pos)?;
        let n_dims = read_u32(buf, &mut pos)?;
        if n_dims > MAX_DIMS {
            return Err(GgufError::DimsTooLarge(n_dims));
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(read_u64(buf, &mut pos)?);
        }
        let raw_type = read_u32(buf, &mut pos)?;
        let ggml_type =
            GgmlType::from_raw(raw_type).ok_or(GgufError::InvalidTensorType(raw_type))?;
        let rel_offset = read_u64(buf, &mut pos)?;
        raw.push(RawInfo {
            name,
            dims,
            ggml_type,
            rel_offset,
        });
    }
    let mut data_base = pos as u64;
    while !data_base.is_multiple_of(align) {
        data_base += 1;
    }

    let mut tensors = Vec::with_capacity(raw.len());
    for info in raw {
        let RawInfo {
            name,
            dims,
            ggml_type,
            rel_offset,
        } = info;
        let offset = data_base
            .checked_add(rel_offset)
            .ok_or(GgufError::TensorOverflow {
                name: name.clone(),
                offset: rel_offset,
                size: 0,
            })?;
        let numel: u64 = dims
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| GgufError::NumelOverflow(name.clone()))?;
        let size = ggml_type.bytes_per_element(numel);
        let end = offset.checked_add(size).ok_or(GgufError::TensorOverflow {
            name: name.clone(),
            offset,
            size,
        })?;
        let end_usize = usize::try_from(end).map_err(|_| GgufError::TensorOverflow {
            name: name.clone(),
            offset,
            size,
        })?;
        let data = buf
            .get(offset as usize..end_usize)
            .ok_or(GgufError::TensorOverflow {
                name: name.clone(),
                offset,
                size,
            })?;
        // The RELATIVE offset must respect the SAME alignment the data section
        // uses (it used to hard-code 32, so a 64-aligned file was rejected).
        if rel_offset % align != 0 {
            return Err(GgufError::UnalignedTensor { name, offset });
        }
        tensors.push(GgufTensor {
            name,
            dims,
            ggml_type,
            offset,
            data,
        });
    }

    Ok(GgufFile {
        version,
        metadata,
        tensors,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(name: &str) -> Vec<u8> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join(name);
        std::fs::read(&path).unwrap_or_else(|e| panic!("fixture {name}: {e}"))
    }

    #[test]
    fn parse_all_quant_fixtures() {
        for name in [
            "mini-q4_0.gguf",
            "mini-q8_0.gguf",
            "mini-f16.gguf",
            "mini-bf16.gguf",
            "mini-q4_k.gguf",
            "mini-q6_k.gguf",
            "mini-q4_k_m.gguf",
        ] {
            let buf = fixture(name);
            let f = parse(&buf).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(f.architecture(), Some("qwen3"), "{name}");
            assert_eq!(
                f.get("context_length").and_then(GgufValue::as_u32),
                Some(128)
            );
            let tokens = f
                .get("tokenizer.ggml.tokens")
                .and_then(GgufValue::as_array)
                .unwrap();
            assert_eq!(tokens.len(), 3, "{name}");
            assert_eq!(f.tensors.len(), 2, "{name}");
            let t0 = &f.tensors[0];
            assert_eq!(t0.name, "token_embd.weight");
            // dims в ne-порядке: [cols, rows]
            assert_eq!(t0.dims, vec![32, 64]);
            assert_eq!(t0.offset % 32, 0);
            assert_eq!(t0.data.len() as u64, t0.data_bytes());
            // данные последнего тензора упираются в конец файла (нет хвостового
            // паддинга) — ловит класс багов «оффсеты относительно блока данных»
            let last = f.tensors.last().unwrap();
            assert_eq!(
                last.offset + last.data.len() as u64,
                buf.len() as u64,
                "{name}"
            );
        }
    }

    #[test]
    fn expected_byte_sizes() {
        // размеры данных фикстур (совпадают с fixture-gen)
        assert_eq!(
            GgmlType::Q4_0.bytes_per_element(64 * 32),
            18 * (64 * 32 / 32)
        );
        assert_eq!(
            GgmlType::Q8_0.bytes_per_element(64 * 32),
            34 * (64 * 32 / 32)
        );
        assert_eq!(
            GgmlType::Q4K.bytes_per_element(64 * 32),
            144 * (64 * 32 / 256)
        );
        assert_eq!(
            GgmlType::Q6K.bytes_per_element(64 * 32),
            210 * (64 * 32 / 256)
        );
        assert_eq!(GgmlType::F16.bytes_per_element(64 * 32), 2 * 64 * 32);
        assert_eq!(GgmlType::Bf16.bytes_per_element(64 * 32), 2 * 64 * 32);
    }

    #[test]
    fn tensor_type_matches_fixture() {
        let buf = fixture("mini-q4_k.gguf");
        let f = parse(&buf).unwrap();
        assert_eq!(f.tensors[0].ggml_type, GgmlType::Q4K);
        let buf = fixture("mini-bf16.gguf");
        let f = parse(&buf).unwrap();
        assert_eq!(f.tensors[0].ggml_type, GgmlType::Bf16);
        let buf = fixture("mini-q4_k_m.gguf");
        let f = parse(&buf).unwrap();
        assert_eq!(f.tensors[0].ggml_type, GgmlType::Q4K);
        assert_eq!(f.tensors[1].ggml_type, GgmlType::Q6K);
    }

    /// Коды типов ПРИБИТЫ к ggml.h эталонного коммита 5266f24da75d
    /// (F32=0 F16=1 Q4_0=2 Q8_0=8 Q4_K=12 Q5_K=13 Q6_K=14 IQ4_XS=23 BF16=30).
    /// BF16=27 был нашей самописной ошибкой (27 — I64): реальные llama.cpp
    /// BF16-файлы (huihui-ai) пишут 30. Менять код можно ТОЛЬКО вместе со
    /// сменой эталонного коммита и перегенерацией фикстур.
    #[test]
    fn raw_codes_pinned_to_reference() {
        assert_eq!(GgmlType::F32.raw(), 0);
        assert_eq!(GgmlType::F16.raw(), 1);
        assert_eq!(GgmlType::Bf16.raw(), 30);
        assert_eq!(GgmlType::Q4_0.raw(), 2);
        assert_eq!(GgmlType::Q8_0.raw(), 8);
        assert_eq!(GgmlType::Q4K.raw(), 12);
        assert_eq!(GgmlType::Q5K.raw(), 13);
        assert_eq!(GgmlType::Q6K.raw(), 14);
        assert_eq!(GgmlType::Iq4Xs.raw(), 23);
        for t in [
            GgmlType::F32,
            GgmlType::F16,
            GgmlType::Bf16,
            GgmlType::Q4_0,
            GgmlType::Q8_0,
            GgmlType::Q4K,
            GgmlType::Q5K,
            GgmlType::Q6K,
            GgmlType::Iq4Xs,
        ] {
            assert_eq!(GgmlType::from_raw(t.raw()), Some(t));
        }
        assert_eq!(GgmlType::from_raw(27), None); // I64 — не поддерживаем
    }

    #[test]
    fn bad_magic_rejected() {
        assert_eq!(parse(b"NOTGGUF...."), Err(GgufError::BadMagic));
    }

    #[test]
    fn truncation_rejected() {
        let buf = fixture("mini-f16.gguf");
        for cut in [1usize, 8, 16, buf.len() / 2, buf.len() - 1] {
            let err = parse(&buf[..cut]);
            assert!(err.is_err(), "truncation at {cut} must fail");
        }
    }

    /// Детерминированный «фаззинг»: мутации байтов не должны вызывать панику.
    #[test]
    fn mutation_fuzz_no_panic() {
        let buf = fixture("mini-q4_k.gguf");
        let mut rng = 0x5EEDu64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for round in 0..500 {
            let mut mutated = buf.clone();
            let n = 1 + (next() % 8) as usize;
            for _ in 0..n {
                let i = (next() as usize) % mutated.len();
                mutated[i] ^= (next() & 0xFF) as u8;
            }
            // допускаются и Ok, и Err — главное, без паники
            let _ = parse(&mutated);
            let _ = round;
        }
    }
}

// ── Painter (1.6b): минимальный GGUF v2 writer ─────────────────────────────

/// Тензор для записи: имя, размеры, тип, сырые данные (ggml-порядок).
pub struct GgufWriteTensor {
    pub name: String,
    pub dims: Vec<u32>,
    pub ggml_type: GgmlType,
    pub data: Vec<u8>,
}

fn w_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn w_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn w_str(out: &mut Vec<u8>, v: &str) {
    w_u64(out, v.len() as u64);
    out.extend_from_slice(v.as_bytes());
}

fn write_value(out: &mut Vec<u8>, v: &GgufValue) -> Result<(), GgufError> {
    match v {
        GgufValue::String(s) => w_str(out, s),
        GgufValue::U32(x) => w_u32(out, *x),
        GgufValue::F32(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::Bool(b) => out.push(u8::from(*b)),
        GgufValue::U64(x) => w_u64(out, *x),
        GgufValue::I32(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::I64(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::F64(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::Array(items) => {
            w_u32(out, items.first().map(gguf_type_of).unwrap_or(0));
            w_u64(out, items.len() as u64);
            for it in items {
                write_value(out, it)?;
            }
        }
        GgufValue::U8(x) => out.push(*x),
        GgufValue::I8(x) => out.push(*x as u8),
        GgufValue::U16(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::I16(x) => out.extend_from_slice(&x.to_le_bytes()),
    }
    Ok(())
}

fn gguf_type_of(v: &GgufValue) -> u32 {
    match v {
        GgufValue::U8(_) => 0,
        GgufValue::I8(_) => 1,
        GgufValue::U16(_) => 2,
        GgufValue::I16(_) => 3,
        GgufValue::U32(_) => 4,
        GgufValue::I32(_) => 5,
        GgufValue::F32(_) => 6,
        GgufValue::Bool(_) => 7,
        GgufValue::String(_) => 8,
        GgufValue::Array(_) => 9,
        GgufValue::U64(_) => 10,
        GgufValue::I64(_) => 11,
        GgufValue::F64(_) => 12,
    }
}

/// Записать GGUF v2: [magic, version, counts, KV, тензоры-таблица, данные].
pub fn write_gguf(
    out: &mut Vec<u8>,
    metadata: &[(String, GgufValue)],
    tensors: &[GgufWriteTensor],
) -> Result<(), GgufError> {
    out.clear();
    w_u32(out, GGUF_MAGIC);
    w_u32(out, V2);
    w_u64(out, tensors.len() as u64);
    w_u64(out, metadata.len() as u64);
    for (k, v) in metadata {
        w_str(out, k);
        w_u32(out, gguf_type_of(v));
        write_value(out, v)?;
    }
    // Таблица тензоров. Offsets заполним после (патчинг по сохранённым
    // позициям), т.к. данные идут ПОСЛЕ таблицы.
    let mut patch_pos = Vec::with_capacity(tensors.len());
    for t in tensors {
        w_str(out, &t.name);
        w_u32(out, t.dims.len() as u32);
        for d in &t.dims {
            w_u64(out, *d as u64);
        }
        w_u32(out, t.ggml_type.raw());
        patch_pos.push(out.len());
        w_u64(out, 0); // offset-плейсхолдер (RELATIVE к началу данных)
    }
    // Данные: выравнивание берётся из `general.alignment` (32 по умолчанию) —
    // ровно то, что использует парсер. Жёсткая 32 при ключе 64 в metadata
    // делала файл нечитаемым собственным round-trip'ом. Оффсеты в таблице —
    // относительно начала секции данных (как в llama.cpp), не абсолютные.
    let align = metadata
        .iter()
        .find(|(k, _)| k == "general.alignment")
        .and_then(|(_, v)| v.as_u32())
        .map(|a| a as usize)
        .filter(|a| *a > 0 && a.is_power_of_two())
        .unwrap_or(32);
    let mut pos = (out.len() + align - 1) & !(align - 1);
    let data_base = pos;
    for (i, t) in tensors.iter().enumerate() {
        let aligned = (pos + align - 1) & !(align - 1);
        let off_bytes = ((aligned - data_base) as u64).to_le_bytes();
        let pp = patch_pos[i];
        out[pp..pp + 8].copy_from_slice(&off_bytes);
        // выравнивание + данные
        out.resize(aligned, 0);
        out.extend_from_slice(&t.data);
        pos = aligned + t.data.len();
    }
    Ok(())
}

#[cfg(test)]
mod tests51 {
    use super::*;

    /// Builds a minimal GGUFv2 by hand so the data section can be aligned to an
    /// arbitrary `general.alignment` (the writer's own default is 32).
    fn hand_rolled_gguf(align: u64, payload: &[u8]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        b.extend_from_slice(&V2.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes()); // tensor count
        b.extend_from_slice(&2u64.to_le_bytes()); // kv count
        w_str(&mut b, "general.architecture");
        w_u32(&mut b, gguf_type_of(&GgufValue::String(String::new())));
        w_str(&mut b, "test");
        w_str(&mut b, "general.alignment");
        w_u32(&mut b, gguf_type_of(&GgufValue::U32(0)));
        w_u32(&mut b, align as u32);
        // tensor info
        w_str(&mut b, "w");
        w_u32(&mut b, 2);
        w_u64(&mut b, 4);
        w_u64(&mut b, 1);
        w_u32(&mut b, GgmlType::F32.raw());
        w_u64(&mut b, 0); // relative offset
        while !(b.len() as u64).is_multiple_of(align) {
            b.push(0);
        }
        b.extend_from_slice(payload);
        b
    }

    /// A file whose data section is aligned to 64 must read back EXACTLY: the
    /// parser hard-coded a 32-byte alignment, so it started reading 32 bytes of
    /// zero padding early and every weight came out as zeros (silently).
    #[test]
    fn non_default_alignment_is_honoured() {
        let data: Vec<f32> = vec![1.5, -2.25, 3.75, 0.5];
        let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
        let buf = hand_rolled_gguf(64, &bytes);
        let f = parse(&buf).unwrap();
        let t = &f.tensors[0];
        let back: Vec<f32> = t
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(back, data, "weights must survive a 64-byte alignment");
    }

    /// The writer must agree with the parser: a file written with an explicit
    /// `general.alignment` has to be readable by the same crate.
    #[test]
    fn writer_honours_declared_alignment() {
        let data: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0];
        let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
        let mut buf = Vec::new();
        write_gguf(
            &mut buf,
            &[
                (
                    "general.architecture".to_string(),
                    GgufValue::String("qwen3".into()),
                ),
                ("general.alignment".to_string(), GgufValue::U32(64)),
            ],
            &[GgufWriteTensor {
                name: "w".into(),
                dims: vec![2, 2],
                ggml_type: GgmlType::F32,
                data: bytes,
            }],
        )
        .unwrap();
        let f = parse(&buf).unwrap();
        let back: Vec<f32> = f.tensors[0]
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(back, data);
    }

    #[test]
    fn writer_roundtrip_v2() {
        let mut buf = Vec::new();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
        write_gguf(
            &mut buf,
            &[
                (
                    "general.architecture".to_string(),
                    GgufValue::String("qwen3".into()),
                ),
                ("qwen3.vocab_size".to_string(), GgufValue::U32(151936)),
                (
                    "misc.array".to_string(),
                    GgufValue::Array(vec![GgufValue::U32(1), GgufValue::U32(2)]),
                ),
            ],
            &[GgufWriteTensor {
                name: "token_embd.weight".into(),
                dims: vec![2, 2],
                ggml_type: GgmlType::F32,
                data: bytes,
            }],
        )
        .unwrap();
        // read-back
        let f = parse(&buf).unwrap();
        assert_eq!(f.architecture(), Some("qwen3"));
        assert_eq!(
            f.get("qwen3.vocab_size").and_then(GgufValue::as_u32),
            Some(151936)
        );
        assert_eq!(f.tensors.len(), 1);
        let t = &f.tensors[0];
        assert_eq!(t.name, "token_embd.weight");
        assert_eq!(t.numel(), 4);
        let raw = t.data;
        assert_eq!(raw.len(), 16);
        let back: Vec<f32> = raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(back, vec![1.0, 2.0, 3.0, 4.0]);
    }

    /// Мульти-тензор: проверка патчинга оффсетов при невыровненных данных,
    /// массивы строк и все скалярные типы значений метаданных.
    #[test]
    fn writer_roundtrip_multi_and_value_types() {
        let mut buf = Vec::new();
        let f16_bytes: Vec<u8> = (0u16..6).flat_map(|x| x.to_le_bytes()).collect();
        let f32_bytes: Vec<u8> = [1.0f32, -2.0]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        write_gguf(
            &mut buf,
            &[
                ("a.u8".into(), GgufValue::U8(7)),
                ("a.i8".into(), GgufValue::I8(-3)),
                ("a.u16".into(), GgufValue::U16(513)),
                ("a.i16".into(), GgufValue::I16(-513)),
                ("a.i32".into(), GgufValue::I32(-42)),
                ("a.f32".into(), GgufValue::F32(0.25)),
                ("a.bool".into(), GgufValue::Bool(true)),
                ("a.u64".into(), GgufValue::U64(u64::MAX / 3)),
                ("a.i64".into(), GgufValue::I64(-(i64::MAX / 3))),
                ("a.f64".into(), GgufValue::F64(1.5)),
                (
                    "a.strs".into(),
                    GgufValue::Array(vec![
                        GgufValue::String("x".into()),
                        GgufValue::String("yz".into()),
                    ]),
                ),
            ],
            &[
                GgufWriteTensor {
                    name: "first.weight".into(),
                    // 3 элемента f16 — 6 байт: первый тензор невыровнен по 32
                    dims: vec![3],
                    ggml_type: GgmlType::F16,
                    data: f16_bytes,
                },
                GgufWriteTensor {
                    name: "second.bias".into(),
                    dims: vec![2],
                    ggml_type: GgmlType::F32,
                    data: f32_bytes,
                },
            ],
        )
        .unwrap();

        let f = parse(&buf).unwrap();
        assert_eq!(f.get("a.u8"), Some(&GgufValue::U8(7)));
        assert_eq!(f.get("a.i8"), Some(&GgufValue::I8(-3)));
        assert_eq!(f.get("a.u16"), Some(&GgufValue::U16(513)));
        assert_eq!(f.get("a.i16"), Some(&GgufValue::I16(-513)));
        assert_eq!(f.get("a.i32"), Some(&GgufValue::I32(-42)));
        assert_eq!(f.get("a.f32"), Some(&GgufValue::F32(0.25)));
        assert_eq!(f.get("a.bool"), Some(&GgufValue::Bool(true)));
        assert_eq!(f.get("a.u64"), Some(&GgufValue::U64(u64::MAX / 3)));
        assert_eq!(f.get("a.i64"), Some(&GgufValue::I64(-(i64::MAX / 3))));
        assert_eq!(f.get("a.f64"), Some(&GgufValue::F64(1.5)));
        assert!(matches!(
            f.get("a.strs"),
            Some(GgufValue::Array(items)) if items.len() == 2
        ));

        assert_eq!(f.tensors.len(), 2);
        let t0 = &f.tensors[0];
        assert_eq!(t0.name, "first.weight");
        assert_eq!(t0.numel(), 3);
        assert_eq!(t0.data.len(), 6);
        // выравнивание: данные тензоров лежат на границах 32
        assert_eq!(t0.offset % 32, 0);
        assert_eq!(f.tensors[1].offset % 32, 0);
        let b: Vec<f32> = f.tensors[1]
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(b, vec![1.0, -2.0]);
    }
}
