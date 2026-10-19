//! Нормализация весов GGUF → внутренний формат (Ф1.2).
//!
//! [`WeightsMap`] — «модель весов» поверх разобранного [`crate::gguf::GgufFile`]:
//! каждый тензор получает логический shape (Rust-порядок, реверс ne-dims),
//! формат [`WeightFormat`] и lazy-доступ к сырым данным (без копий — caller
//! mmap'ит файл). Tied-веса (embedding/output на одном сторадже) дедуплицируются
//! по (offset, size): одно хранилище — несколько имён.
//!
//! [`dequant_f32`] — CPU-референс деквантования для тестов квант-GEMM/MMV
//! (Ф1.3/1.4) и проверки форматов; раскладки блоков по ggml (см. fixture-gen).

use crate::dtype::{Dtype, WeightFormat};
use crate::gguf::{GgmlType, GgufError, GgufFile};

/// Сырое хранилище тензора (после tied-дедупликации).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Storage<'a> {
    pub(crate) offset: u64,
    pub(crate) data: &'a [u8],
}

/// Один тензор модели в нормализованном виде.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightTensor<'a> {
    pub name: String,
    /// Логический shape в Rust-порядке (последняя dim — самая медленная в
    /// GGUF-файле; для матриц: [rows, cols]).
    pub shape: Vec<u64>,
    pub format: WeightFormat,
    pub(crate) offset: u64,
    pub(crate) data: &'a [u8],
}

impl<'a> WeightTensor<'a> {
    pub fn numel(&self) -> u64 {
        self.shape.iter().product()
    }
    /// Сырые байты стораджа (для GPU-загрузки / CPU-декаунта).
    pub fn raw(&self) -> &'a [u8] {
        self.data
    }
    /// N (iter-10): публичный конструктор — тесты (gguf_packed_gemm) строят
    /// CPU-оракул из GGUF-байтов без доступа к приватным полям.
    pub fn from_parts(
        name: impl Into<String>,
        shape: Vec<u64>,
        format: WeightFormat,
        data: &'a [u8],
    ) -> Self {
        Self {
            name: name.into(),
            shape,
            format,
            offset: 0,
            data,
        }
    }
}

/// Карта весов модели: тензоры по именам + tied-алиасы.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightsMap<'a> {
    pub arch: String,
    pub(crate) tensors: Vec<WeightTensor<'a>>,
    /// Стораж-индекс для tied-дедупликации (по (offset, size)).
    pub(crate) storage: Vec<Storage<'a>>,
}

impl<'a> WeightsMap<'a> {
    /// Построить карту из GGUF. Поддерживаемые форматы — целевая матрица
    /// движка: F32/F16/BF16, Q4_0, Q8_0, Q4_K, Q6_K (Q4_K_M на уровне схемы).
    /// Прочие GGML-типы — `UnsupportedFormat` (fail fast).
    pub fn from_gguf(file: &'a GgufFile<'a>) -> Result<Self, GgufError> {
        let arch = file.architecture().unwrap_or("unknown").to_string();
        let mut tensors = Vec::with_capacity(file.tensors.len());
        let mut storage: Vec<Storage<'a>> = Vec::new();

        for t in &file.tensors {
            let format = match t.ggml_type {
                GgmlType::F32 => WeightFormat::Plain(Dtype::F32),
                GgmlType::F16 => WeightFormat::Plain(Dtype::F16),
                GgmlType::Bf16 => WeightFormat::Plain(Dtype::BF16),
                GgmlType::Q4_0 => WeightFormat::Q4_0,
                GgmlType::Q8_0 => WeightFormat::Q8_0,
                GgmlType::Q4K => WeightFormat::Q4_K,
                GgmlType::Q6K => WeightFormat::Q6_K,
                GgmlType::Q5K => WeightFormat::Q5_K,
                GgmlType::Q5_0 => WeightFormat::Q5_0,
                GgmlType::Iq4Xs => WeightFormat::Iq4Xs,
                GgmlType::Iq4Nl => WeightFormat::Iq4Nl,
                GgmlType::Mxfp4 => WeightFormat::Mxfp4,
                GgmlType::Nvfp4 => WeightFormat::Nvfp4,
                GgmlType::Q2K => WeightFormat::Q2_K,
                GgmlType::Q3K => WeightFormat::Q3_K,
                GgmlType::Q4_1 => WeightFormat::Q4_1,
                GgmlType::Q5_1 => WeightFormat::Q5_1,
                GgmlType::Q8_1 => WeightFormat::Q8_1,
                GgmlType::Q8K => WeightFormat::Q8_K,
                GgmlType::Q1_0 => WeightFormat::Q1_0,
                GgmlType::Q2_0 => WeightFormat::Q2_0,
                GgmlType::Tq1_0 => WeightFormat::TQ1_0,
                GgmlType::Tq2_0 => WeightFormat::TQ2_0,
                other => {
                    return Err(GgufError::InvalidTensorType(other as u32));
                }
            };
            // логический shape: реверс ne-dims
            let shape: Vec<u64> = t.dims.iter().rev().copied().collect();
            // tied-дедупликация по (offset, size)
            let size = t.data.len() as u64;
            let storage_idx = storage
                .iter()
                .position(|s| s.offset == t.offset && s.data.len() as u64 == size)
                .unwrap_or_else(|| {
                    storage.push(Storage {
                        offset: t.offset,
                        data: t.data,
                    });
                    storage.len() - 1
                });
            tensors.push(WeightTensor {
                name: t.name.clone(),
                shape,
                format,
                offset: storage[storage_idx].offset,
                data: storage[storage_idx].data,
            });
        }

        Ok(Self {
            arch,
            tensors,
            storage,
        })
    }

    pub fn tensors(&self) -> &[WeightTensor<'a>] {
        &self.tensors
    }

    pub fn get(&self, name: &str) -> Option<&WeightTensor<'a>> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Имена тензоров, разделяющих то же хранилище, что и `name` (tied).
    pub fn tied_aliases(&self, name: &str) -> Vec<&str> {
        let Some(t) = self.get(name) else {
            return Vec::new();
        };
        self.tensors
            .iter()
            .filter(|o| o.offset == t.offset && o.data.len() == t.data.len() && o.name != name)
            .map(|o| o.name.as_str())
            .collect()
    }
}

// ── CPU-референс деквантования ──────────────────────────────────────────────

fn f16_to_f32(b: [u8; 2]) -> f32 {
    let h = u16::from_le_bytes(b);
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    let f = if exp == 0 {
        (mant as f32) * 2f32.powi(-24)
    } else if exp == 31 {
        f32::INFINITY
    } else {
        ((1u32 << 10) | mant) as f32 * 2f32.powi(exp as i32 - 25)
    };
    if sign == 1 { -f } else { f }
}

fn bf16_to_f32(b: [u8; 2]) -> f32 {
    f32::from_bits((u16::from_le_bytes(b) as u32) << 16)
}

pub(crate) fn dequant_q4_0(data: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() / 18 * 32);
    for blk in data.chunks_exact(18) {
        let d = f16_to_f32([blk[0], blk[1]]);
        // N (iter-9): canonical ggml pairing — byte j holds weights
        // j (low nibble) and j+16 (high nibble). Positions are INDEXED
        // (j and j+16) — the old consecutive push interleaved them as
        // (2j, 2j+1), silently misparsing every real llama.cpp Q4_0 GGUF.
        let qs = &blk[2..];
        for j in 0..16 {
            out.push(((qs[j] & 0x0F) as f32 - 8.0) * d); // w[j]
        }
        for j in 0..16 {
            out.push(((qs[j] >> 4) as f32 - 8.0) * d); // w[j+16]
        }
    }
    out
}

pub(crate) fn dequant_q8_0(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(34)
        .flat_map(|blk| {
            let d = f16_to_f32([blk[0], blk[1]]);
            blk[2..].iter().map(move |&b| (b as i8 as f32) * d)
        })
        .collect()
}

/// Канонический Q4_K (ggml block_q4_K, 144 Б/256): d (f16, 0..2), dmin
/// (f16, 2..4), scales[12] (4..16 — 6-битные scale/min 8 подблоков по 32,
/// упаковка get_scale_min_k4), qs[128] (16..144 — чанк-мажор: байт 32c+l
/// = {low: 64c+l, high: 64c+32+l}). Деквант (ggml dequantize_row_q4_K):
/// y = d·sc·q − dmin·m, q ∈ 0..15. Совместимо с реальными llama.cpp-файлами
/// (старая 148-байтная раскладка была кастомным диалектом — волна ревью 8).
pub(crate) fn dequant_q4_k(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(144)
        .flat_map(|sb| {
            let d = f16_to_f32([sb[0], sb[1]]);
            let dmin = f16_to_f32([sb[2], sb[3]]);
            let scales = &sb[4..16];
            let qs = &sb[16..144];
            let mut y = Vec::with_capacity(256);
            let mut is = 0usize;
            // 4 чанка по 64: low-nibbles → подблок is, high-nibbles → is+1.
            for c in 0..4 {
                let (sc, m) = scale_min_k4(is, scales);
                let d1 = d * sc as f32;
                let m1 = dmin * m as f32;
                let (sc2, m2) = scale_min_k4(is + 1, scales);
                let d2 = d * sc2 as f32;
                let m2 = dmin * m2 as f32;
                for l in 0..32 {
                    y.push(d1 * (qs[32 * c + l] & 0x0F) as f32 - m1);
                }
                for l in 0..32 {
                    y.push(d2 * (qs[32 * c + l] >> 4) as f32 - m2);
                }
                is += 2;
            }
            y
        })
        .collect()
}

/// ggml get_scale_min_k4: 12 байт scales → (scale, min) 6-битные коды
/// подблока j (та же упаковка, что у Q5_K).
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (sc, m)
    }
}

/// Канонический Q6_K (ggml block_q6_K, 210 Б/256): ql[128] (низкие 4 бита),
/// qh[64] (старшие 2 бита), scales[16] (int8 на 16-элементные подблоки),
/// d (f16) в КОНЦЕ блока (байты 208..210). Деквант: y = d·sc·(q − 32).
/// Бит-упаковка qh: на каждую 128-половину байт qh[lane] держит старшие
/// 2 бита значений lane|lane+32<<2|lane+64<<4|lane+96<<6 (сверено с CUDA
/// q6k_weight и speech-квантайзером — bit-exact).
pub(crate) fn dequant_q6_k(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(210)
        .flat_map(|sb| {
            let d = f16_to_f32([sb[208], sb[209]]);
            let ql = &sb[0..128];
            let qh = &sb[128..192];
            let scales = &sb[192..208];
            let mut y = vec![0f32; 256];
            for half in 0..2 {
                let qlh = &ql[half * 64..half * 64 + 64];
                let qhh = &qh[half * 32..half * 32 + 32];
                let sch = &scales[half * 8..half * 8 + 8];
                for lane in 0..32 {
                    let ql0 = qlh[lane];
                    let ql1 = qlh[lane + 32];
                    let qhv = qhh[lane];
                    let base = half * 128 + lane;
                    let s = |off: usize| sch[(lane >> 4) + off] as i8 as f32;
                    let q = |v: u8| (v as i32 - 32) as f32;
                    y[base] = d * s(0) * q((ql0 & 0xF) | ((qhv & 3) << 4));
                    y[base + 32] = d * s(2) * q((ql1 & 0xF) | ((qhv >> 2 & 3) << 4));
                    y[base + 64] = d * s(4) * q((ql0 >> 4) | ((qhv >> 4 & 3) << 4));
                    y[base + 96] = d * s(6) * q((ql1 >> 4) | ((qhv >> 6 & 3) << 4));
                }
            }
            y
        })
        .collect()
}

/// IQ4_XS (i-квант, ggml `dequantize_row_iq4_xs`, 136 Б/256): d(f16) @0,
/// scales_h (u16 LE) @2, scales_l[4] @4, qs[128] @8. Масштаб подблока
/// ib (0..8): 6 бит = (scales_l[ib/2] >> (4·(ib%2)) & 0xF) | ((scales_h >> 2·ib) & 3) << 4,
/// реальный множитель — d·(ls − 32) (знак возможен!). Ниббл — индекс в
/// нелинейном кодбуке kvalues_iq4nl: элементы подблока идут парами
/// (j — младший ниббл байта j, j+16 — старший).
pub fn dequant_iq4_xs(data: &[u8]) -> Vec<f32> {
    const KV: [f32; 16] = [
        -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0,
        69.0, 89.0, 113.0,
    ];
    data.chunks_exact(136)
        .flat_map(|sb| {
            let d = f16_to_f32([sb[0], sb[1]]);
            let scales_h = u16::from_le_bytes([sb[2], sb[3]]);
            let scales_l = &sb[4..8];
            let qs = &sb[8..136];
            let mut y = vec![0f32; 256];
            for ib in 0..8usize {
                let ls = (((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) as u32
                    | (((scales_h >> (2 * ib)) & 3) as u32) << 4) as i32;
                let dl = d * (ls - 32) as f32;
                let q = &qs[ib * 16..ib * 16 + 16];
                for j in 0..16usize {
                    y[ib * 32 + j] = dl * KV[(q[j] & 0xF) as usize];
                    y[ib * 32 + 16 + j] = dl * KV[(q[j] >> 4) as usize];
                }
            }
            y
        })
        .collect()
}

/// Q1_0 (18 Б/128): d(f16) @0, qs[16] @2; бит j → +d, иначе −d.
pub(crate) fn dequant_q1_0(data: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() / 18 * 128);
    for blk in data.chunks_exact(18) {
        let d = f16_to_f32([blk[0], blk[1]]);
        for j in 0..128usize {
            let bit = (blk[2 + j / 8] >> (j % 8)) & 1;
            out.push(if bit != 0 { d } else { -d });
        }
    }
    out
}

/// Q2_0 (18 Б/64): d(f16) @0, qs[16] @2 по 2 бита (младшие вперёд);
/// значение (q − 1)·d (00=−1, 01=0, 10=+1, 11=+2).
pub(crate) fn dequant_q2_0(data: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() / 18 * 64);
    for blk in data.chunks_exact(18) {
        let d = f16_to_f32([blk[0], blk[1]]);
        for j in 0..64usize {
            let q = (blk[2 + j / 4] >> (2 * (j % 4))) & 3;
            out.push((q as f32 - 1.0) * d);
        }
    }
    out
}

/// TQ1_0 (54 Б/256): qs[48] @0 (5 троичных значений в байте через 3^n),
/// qh[4] @48 (4 значения), d @52. Порядок эталона: 32 байта × 5 (m-major),
/// затем 16 байт × 5, затем 4 байта qh × 4; value = (xi − 1)·d, где
/// xi = ((byte·3^n)·3) >> 8.
pub(crate) fn dequant_tq1_0(data: &[u8]) -> Vec<f32> {
    const POW3: [u16; 5] = [1, 3, 9, 27, 81];
    let mut out = Vec::with_capacity(data.len() / 54 * 256);
    for blk in data.chunks_exact(54) {
        let d = f16_to_f32([blk[52], blk[53]]);
        let qs = &blk[0..48];
        let qh = &blk[48..52];
        // ВНИМАНИЕ: в эталоне q объявлен uint8_t — произведение берётся
        // по модулю 256 (переполнение), затем xi = (q·3) >> 8.
        let push = |byte: u8, n: usize, out: &mut Vec<f32>| {
            let q = ((byte as u32 * POW3[n] as u32) & 0xFF) as u32;
            let xi = ((q * 3) >> 8) as i32;
            out.push((xi - 1) as f32 * d);
        };
        for n in 0..5 {
            for m in 0..32 {
                push(qs[m], n, &mut out);
            }
        }
        for n in 0..5 {
            for m in 0..16 {
                push(qs[32 + m], n, &mut out);
            }
        }
        for n in 0..4 {
            for j in 0..4 {
                push(qh[j], n, &mut out);
            }
        }
    }
    out
}

/// TQ2_0 (66 Б/256): qs[64] @0, d(f16) @64; для позиции p: chunk = p/128,
/// l = (p%128)/32, m = p%32 → q = (qs[chunk·32 + m] >> 2l) & 3,
/// значение (q − 1)·d.
pub(crate) fn dequant_tq2_0(data: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() / 66 * 256);
    for blk in data.chunks_exact(66) {
        let d = f16_to_f32([blk[64], blk[65]]);
        for p in 0..256usize {
            let chunk = p / 128;
            let l = (p % 128) / 32;
            let m = p % 32;
            let q = (blk[chunk * 32 + m] >> (2 * l)) & 3;
            out.push((q as f32 - 1.0) * d);
        }
    }
    out
}

/// Q4_1 (ggml `dequantize_row_q4_1`, 20 Б/32): d(f16) @0, m(f16) @2,
/// qs[16] @4; значение d·q + m (без сдвига −8, в отличие от Q4_0).
pub(crate) fn dequant_q4_1(data: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() / 20 * 32);
    for blk in data.chunks_exact(20) {
        let d = f16_to_f32([blk[0], blk[1]]);
        let m = f16_to_f32([blk[2], blk[3]]);
        let mut row = [0f32; 32];
        for j in 0..16usize {
            row[j] = (blk[4 + j] & 0xF) as f32 * d + m;
            row[j + 16] = (blk[4 + j] >> 4) as f32 * d + m;
        }
        out.extend_from_slice(&row);
    }
    out
}

/// Q5_1 (ggml `dequantize_row_q5_1`, 24 Б/32): d @0, m @2, qh[4] @4,
/// qs[16] @8; 5-й бит значения j — бит j qh (j+16 — бит j+16), d·q + m.
pub(crate) fn dequant_q5_1(data: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() / 24 * 32);
    for blk in data.chunks_exact(24) {
        let d = f16_to_f32([blk[0], blk[1]]);
        let m = f16_to_f32([blk[2], blk[3]]);
        let qh = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
        let mut row = [0f32; 32];
        for j in 0..16usize {
            let lo = ((blk[8 + j] & 0xF) as u32 | ((qh >> j) & 1) << 4) as f32;
            let hi = ((blk[8 + j] >> 4) as u32 | ((qh >> (j + 16)) & 1) << 4) as f32;
            row[j] = lo * d + m;
            row[j + 16] = hi * d + m;
        }
        out.extend_from_slice(&row);
    }
    out
}

/// Q8_1 (36 Б/32): d(f16) @0, s(f16) @2 (сумма — для активаций), int8[32]
/// @4; деквант весов — d·q (s не используется).
pub(crate) fn dequant_q8_1(data: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() / 36 * 32);
    for blk in data.chunks_exact(36) {
        let d = f16_to_f32([blk[0], blk[1]]);
        for j in 0..32usize {
            out.push(blk[4 + j] as i8 as f32 * d);
        }
    }
    out
}

/// Q8_K (292 Б/256): f32 d @0, int8 qs[256] @4, int16 bsums[16] @260;
/// деквант весов — d·q.
pub(crate) fn dequant_q8_k(data: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() / 292 * 256);
    for blk in data.chunks_exact(292) {
        let d = f32::from_le_bytes([blk[0], blk[1], blk[2], blk[3]]);
        for j in 0..256usize {
            out.push(blk[4 + j] as i8 as f32 * d);
        }
    }
    out
}

/// Q2_K (ggml `dequantize_row_q2_K`, 84 Б/256): scales[16] @0 (4 бита
/// scale, 4 бита min), qs[64] @16, d @80, dmin @82. 16 подблоков по 16
/// значений: подблок s — iter = s/8 (q += 32), j = (s%8)/2 (сдвиг 2j),
/// half = s%2 (байты l + 16·half); значение dl·((q>>shift)&3) − ml.
pub(crate) fn dequant_q2_k(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(84)
        .flat_map(|sb| {
            let qs = &sb[16..80];
            let d = f16_to_f32([sb[80], sb[81]]);
            let dmin = f16_to_f32([sb[82], sb[83]]);
            let mut y = vec![0f32; 256];
            for s in 0..16usize {
                let sc = sb[s];
                let dl = d * (sc & 0xF) as f32;
                let ml = dmin * (sc >> 4) as f32;
                let shift = 2 * ((s % 8) / 2) as u32;
                let half = s % 2;
                let qb = &qs[(s / 8) * 32 + half * 16..];
                for l in 0..16usize {
                    let v = ((qb[l] >> shift) & 3) as i8;
                    y[s * 16 + l] = dl * v as f32 - ml;
                }
            }
            y
        })
        .collect()
}

/// Q3_K (ggml `dequantize_row_q3_K`, 110 Б/256): hmask[32] @0, qs[64] @32,
/// scales[12] @96, d @108. 6-битные scale со знаком (scales[i] − 32)
/// собираются из 12 байт (старшие 4 бита — в scales[8..12]); значение
/// dl·(((q>>shift)&3) − (hm & bit ? 0 : 4)).
pub(crate) fn dequant_q3_k(data: &[u8]) -> Vec<f32> {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0F0F_0F0F;
    data.chunks_exact(110)
        .flat_map(|sb| {
            let hm = &sb[0..32];
            let qs = &sb[32..96];
            let mut aux = [0u32; 4];
            for i in 0..3 {
                aux[i] = u32::from_le_bytes([
                    sb[96 + 4 * i],
                    sb[97 + 4 * i],
                    sb[98 + 4 * i],
                    sb[99 + 4 * i],
                ]);
            }
            let tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
            aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
            aux[0] = (aux[0] & KMASK2) | (((tmp >> 0) & KMASK1) << 4);
            aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
            let scales: [i8; 16] = {
                let mut sc = [0i8; 16];
                for i in 0..16 {
                    sc[i] = ((aux[i / 4] >> (8 * (i % 4))) & 0xFF) as u8 as i8;
                }
                sc
            };
            let d = f16_to_f32([sb[108], sb[109]]);
            let mut y = vec![0f32; 256];
            for s in 0..16usize {
                let dl = d * (scales[s] as f32 - 32.0);
                let shift = 2 * ((s % 8) / 2) as u32;
                let half = s % 2;
                let qb = &qs[(s / 8) * 32 + half * 16..];
                let hb = &hm[half * 16..];
                let bit = 1u8 << (s / 2);
                for l in 0..16usize {
                    let q = ((qb[l] >> shift) & 3) as i8;
                    let sub = if hb[l] & bit != 0 { 0 } else { 4 };
                    y[s * 16 + l] = dl * (q - sub) as f32;
                }
            }
            y
        })
        .collect()
}

/// IQ4_NL (ggml `dequantize_row_iq4_nl`, 18 Б/32): d(f16) @0, qs[16] @2.
/// Раскладка как у Q4_0: байт j — значение j (младший ниббл) и j+16
/// (старший); ниббл — индекс в нелинейном кодбуке kvalues_iq4nl (тот же,
/// что у IQ4_XS), масштаб d — на весь 32-эл. блок.
pub(crate) fn dequant_iq4_nl(data: &[u8]) -> Vec<f32> {
    const KV: [f32; 16] = [
        -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0,
        69.0, 89.0, 113.0,
    ];
    let mut out = Vec::with_capacity(data.len() / 18 * 32);
    for blk in data.chunks_exact(18) {
        let d = f16_to_f32([blk[0], blk[1]]);
        let mut row = [0f32; 32];
        for j in 0..16usize {
            row[j] = d * KV[(blk[2 + j] & 0xF) as usize];
            row[j + 16] = d * KV[(blk[2 + j] >> 4) as usize];
        }
        out.extend_from_slice(&row);
    }
    out
}

/// E8M0-скейл MXFP4 → float (ggml ggml_e8m0_to_fp32_half): d = 0.5·2^(e−127),
/// т.е. 2^(e−128); e<2 — субнормальные степени через биты 0x00200000<<e.
fn e8m0_to_fp32_half(e: u8) -> f32 {
    let bits: u32 = if e < 2 {
        0x0020_0000u32 << e
    } else {
        ((e as u32) - 1) << 23
    };
    f32::from_bits(bits)
}

/// UE4M3-скейл NVFP4 → float (порт ggml_ue4m3_to_fp32): unsigned, 4 бита
/// экспоненты (bias 7), 3 мантиссы; x==0 и x==0x7F — ноль; возвращает
/// значение·0.5 (компенсация «удвоенных» kvalues_fp4).
fn ue4m3_to_fp32(x: u8) -> f32 {
    if x == 0 || x == 0x7F {
        return 0.0;
    }
    let exp = ((x >> 3) & 0xF) as i32;
    let man = (x & 0x7) as f32;
    let raw = if exp == 0 {
        man * 2f32.powi(-9)
    } else {
        (1.0 + man / 8.0) * 2f32.powi(exp - 7)
    };
    raw * 0.5
}

/// NVFP4 (ggml `dequantize_row_nvfp4`, 36 Б/64): d[4] (UE4M3, по одному на
/// 16-эл. подблок) @0, qs[32] @4. Для подблока s: байт qs[s*8+j] — значения
/// s*16+j (младший ниббл) и s*16+j+8 (старший), умноженные на d_s.
pub(crate) fn dequant_nvfp4(data: &[u8]) -> Vec<f32> {
    const KV: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
    let mut out = Vec::with_capacity(data.len() / 36 * 64);
    for blk in data.chunks_exact(36) {
        let mut row = [0f32; 64];
        for s in 0..4usize {
            let d = ue4m3_to_fp32(blk[s]);
            for j in 0..8usize {
                let b = blk[4 + s * 8 + j];
                row[s * 16 + j] = KV[(b & 0xF) as usize] as f32 * d;
                row[s * 16 + j + 8] = KV[(b >> 4) as usize] as f32 * d;
            }
        }
        out.extend_from_slice(&row);
    }
    out
}

/// MXFP4 (ggml `dequantize_row_mxfp4`, 17 Б/32): E8M0-скейл @0, qs[16] @1.
/// Значения — kvalues_fp4 («удвоенные» E2M1: 0,1,2,3,4,6,8,12,0,-1,…,-12),
/// порядок как у Q4_0 (байт j — значения j и j+16).
pub(crate) fn dequant_mxfp4(data: &[u8]) -> Vec<f32> {
    const KV: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
    let mut out = Vec::with_capacity(data.len() / 17 * 32);
    for blk in data.chunks_exact(17) {
        let d = e8m0_to_fp32_half(blk[0]);
        let mut row = [0f32; 32];
        for j in 0..16usize {
            row[j] = KV[(blk[1 + j] & 0xF) as usize] as f32 * d;
            row[j + 16] = KV[(blk[1 + j] >> 4) as usize] as f32 * d;
        }
        out.extend_from_slice(&row);
    }
    out
}

/// Q5_0 (канонический ggml, 22 Б/32): d(f16) @0, qh[4] (u32 LE) @2,
/// qs[16] @6; парность j (младший ниббл) / j+16 (старший), 5-е биты из qh.
pub fn dequant_q5_0(data: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() / 22 * 32);
    for blk in data.chunks_exact(22) {
        let d = f16_to_f32([blk[0], blk[1]]);
        let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
        // Позиции элементов: j (младший ниббл) и j+16 (старший), как в
        // dequantize_row_q5_0; порядок вывода — по индексу, не «подряд».
        let mut row = [0f32; 32];
        for j in 0..16usize {
            let lo = ((blk[6 + j] & 0x0F) as u32 | (((qh >> j) & 1) << 4)) as i32 - 16;
            let hi = ((blk[6 + j] >> 4) as u32 | (((qh >> (j + 16)) & 1) << 4)) as i32 - 16;
            row[j] = lo as f32 * d;
            row[j + 16] = hi as f32 * d;
        }
        out.extend_from_slice(&row);
    }
    out
}

/// Q5_K (канонический ggml, 176 Б/256): d @0, dmin @2, scales[12] @4
/// (get_scale_min_k4), qh[32] @16 (5-е биты: маски 1<<2c / 2<<2c на чанк),
/// qs[128] @48 (низкие 4 бита, чанк-мажор). y = d·sc·q − dmin·m, q ∈ 0..31.
pub fn dequant_q5_k(data: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() / 176 * 256);
    for blk in data.chunks_exact(176) {
        let d = f16_to_f32([blk[0], blk[1]]);
        let dmin = f16_to_f32([blk[2], blk[3]]);
        let (mut u1, mut u2) = (1u8, 2u8);
        for c in 0..4usize {
            let (sc1, m1) = scale_min_k4(2 * c, &blk[4..16]);
            let (sc2, m2) = scale_min_k4(2 * c + 1, &blk[4..16]);
            let (dd1, mm1) = (d * sc1 as f32, dmin * m1 as f32);
            let (dd2, mm2) = (d * sc2 as f32, dmin * m2 as f32);
            // Порядок эталона: СНАЧАЛА 32 элемента низких нибблов, затем 32
            // высоких (не вперемешку — иначе все чётные/нечётные значения
            // меняются местами).
            for l in 0..32usize {
                let b = blk[48 + c * 32 + l];
                let qh = blk[16 + l];
                let lo = (b & 0x0F) as u32 + if qh & u1 != 0 { 16 } else { 0 };
                out.push(dd1 * lo as f32 - mm1);
            }
            for l in 0..32usize {
                let b = blk[48 + c * 32 + l];
                let qh = blk[16 + l];
                let hi = (b >> 4) as u32 + if qh & u2 != 0 { 16 } else { 0 };
                out.push(dd2 * hi as f32 - mm2);
            }
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    out
}

/// Деквантовать GGUF-тензор (сырой ggml-тип) в f32 — для re-quant (Ф4.12).
pub fn dequant_gguf(
    ggml_type: crate::gguf::GgmlType,
    data: &[u8],
) -> Result<Vec<f32>, crate::error::BackendError> {
    use crate::error::BackendError;
    use crate::gguf::GgmlType as T;
    Ok(match ggml_type {
        T::F32 => data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        T::F16 => data
            .chunks_exact(2)
            .map(|b| f16_to_f32([b[0], b[1]]))
            .collect(),
        T::Bf16 => data
            .chunks_exact(2)
            .map(|b| bf16_to_f32([b[0], b[1]]))
            .collect(),
        T::Q4_0 => dequant_q4_0(data),
        T::Q8_0 => dequant_q8_0(data),
        T::Q4K => dequant_q4_k(data),
        T::Q6K => dequant_q6_k(data),
        T::Q5K => dequant_q5_k(data),
        T::Q5_0 => dequant_q5_0(data),
        T::Iq4Xs => dequant_iq4_xs(data),
        T::Iq4Nl => dequant_iq4_nl(data),
        T::Mxfp4 => dequant_mxfp4(data),
        T::Nvfp4 => dequant_nvfp4(data),
        T::Q2K => dequant_q2_k(data),
        T::Q3K => dequant_q3_k(data),
        T::Q4_1 => dequant_q4_1(data),
        T::Q5_1 => dequant_q5_1(data),
        T::Q8_1 => dequant_q8_1(data),
        T::Q8K => dequant_q8_k(data),
        T::Q1_0 => dequant_q1_0(data),
        T::Q2_0 => dequant_q2_0(data),
        T::Tq1_0 => dequant_tq1_0(data),
        T::Tq2_0 => dequant_tq2_0(data),
        // re-quant из прочих форматов — по мере добавления (Ф4.12)
        other => {
            return Err(BackendError::UnsupportedFormat(format!(
                "dequant {other:?} (re-quant из этого формата — следующая итерация)"
            )));
        }
    })
}

/// Деквантовать тензор в f32 (CPU-референс; длина == numel).
/// Q26: the Int4/Nf4 arm must NOT panic (it is reachable via
/// `Backend::from_quantized` from a malformed checkpoint); the prod path
/// rejects those formats with an Err BEFORE calling this (backend.rs), and
/// here the fallback is an empty vec + a loud marker — the test harness
/// asserts len == numel and fails loudly.
pub fn dequant_f32(t: &WeightTensor<'_>) -> Vec<f32> {
    match t.format {
        WeightFormat::Plain(Dtype::F32) => t
            .data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        WeightFormat::Plain(Dtype::F16) => t
            .data
            .chunks_exact(2)
            .map(|b| f16_to_f32([b[0], b[1]]))
            .collect(),
        WeightFormat::Plain(Dtype::BF16) => t
            .data
            .chunks_exact(2)
            .map(|b| bf16_to_f32([b[0], b[1]]))
            .collect(),
        WeightFormat::Q4_0 => dequant_q4_0(t.data),
        WeightFormat::Q8_0 => dequant_q8_0(t.data),
        WeightFormat::Q4_K => dequant_q4_k(t.data),
        WeightFormat::Q6_K => dequant_q6_k(t.data),
        WeightFormat::Q5_K => dequant_q5_k(t.data),
        WeightFormat::Q5_0 => dequant_q5_0(t.data),
        WeightFormat::Iq4Xs => dequant_iq4_xs(t.data),
        WeightFormat::Iq4Nl => dequant_iq4_nl(t.data),
        WeightFormat::Mxfp4 => dequant_mxfp4(t.data),
        WeightFormat::Nvfp4 => dequant_nvfp4(t.data),
        WeightFormat::Q2_K => dequant_q2_k(t.data),
        WeightFormat::Q3_K => dequant_q3_k(t.data),
        WeightFormat::Q4_1 => dequant_q4_1(t.data),
        WeightFormat::Q5_1 => dequant_q5_1(t.data),
        WeightFormat::Q8_1 => dequant_q8_1(t.data),
        WeightFormat::Q8_K => dequant_q8_k(t.data),
        WeightFormat::Q1_0 => dequant_q1_0(t.data),
        WeightFormat::Q2_0 => dequant_q2_0(t.data),
        WeightFormat::TQ1_0 => dequant_tq1_0(t.data),
        WeightFormat::TQ2_0 => dequant_tq2_0(t.data),
        // N (plan 1.1): a stub that returned EMPTY weights (with an
        // eprintln) silently corrupted any model reaching this path —
        // fail loudly instead.
        WeightFormat::Int4 | WeightFormat::Nf4 => {
            unreachable!("dequant_f32: Int4/Nf4 is rejected by from_quantized before upload")
        }
        WeightFormat::Plain(_) => unreachable!("I8 weights not produced by GGUF normalization"),
    }
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

    fn map(name: &str) -> WeightsMap<'static> {
        // fixture buffer и разобранный файл живут вечно (leak) — файлы малые,
        // тесты детерминированны; так `WeightsMap<'static>` возвращается чисто.
        let buf: &'static [u8] = Box::leak(fixture(name).into_boxed_slice());
        let f: &'static crate::gguf::GgufFile<'static> =
            Box::leak(Box::new(crate::gguf::parse(buf).unwrap()));
        WeightsMap::from_gguf(f).unwrap()
    }

    fn approx(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() <= tol, "dequant mismatch: {x} vs {y}");
        }
    }

    #[test]
    fn formats_map_correctly() {
        for (name, want) in [
            ("mini-q4_0.gguf", WeightFormat::Q4_0),
            ("mini-q8_0.gguf", WeightFormat::Q8_0),
            ("mini-f16.gguf", WeightFormat::Plain(Dtype::F16)),
            ("mini-bf16.gguf", WeightFormat::Plain(Dtype::BF16)),
            ("mini-q4_k.gguf", WeightFormat::Q4_K),
            ("mini-q6_k.gguf", WeightFormat::Q6_K),
        ] {
            let m = map(name);
            assert_eq!(m.get("token_embd.weight").unwrap().format, want, "{name}");
            assert_eq!(m.arch, "qwen3");
        }
        // Q4_K_M-схема: смесь Q4_K + Q6_K по тензорам
        let m = map("mini-q4_k_m.gguf");
        assert_eq!(
            m.get("token_embd.weight").unwrap().format,
            WeightFormat::Q4_K
        );
        assert_eq!(
            m.get("blk.0.attn_q.weight").unwrap().format,
            WeightFormat::Q6_K
        );
    }

    #[test]
    fn shapes_are_logical_order() {
        let m = map("mini-q4_0.gguf");
        // GGUF ne-dims [32, 64] → логический [64, 32] (rows, cols)
        assert_eq!(m.get("token_embd.weight").unwrap().shape, vec![64, 32]);
        assert_eq!(m.get("blk.0.attn_q.weight").unwrap().shape, vec![32, 32]);
    }

    /// Кросс-форматная проверка: все кванты представляют ОДНИ и те же веса
    /// (одинаковые сиды fixture-gen), поэтому деквант ≈ деквант f16.
    #[test]
    fn dequant_matches_f16_reference() {
        let f16 = dequant_f32(map("mini-f16.gguf").get("token_embd.weight").unwrap());
        for (name, tol) in [
            ("mini-q4_0.gguf", 0.15),
            ("mini-q8_0.gguf", 0.02),
            ("mini-q4_k.gguf", 0.15),
            ("mini-q6_k.gguf", 0.06),
        ] {
            let q = dequant_f32(map(name).get("token_embd.weight").unwrap());
            approx(&q, &f16, tol);
        }
    }

    #[test]
    fn dequant_lengths_match_numel() {
        let m = map("mini-q4_k.gguf");
        for t in m.tensors() {
            assert_eq!(dequant_f32(t).len() as u64, t.numel(), "{}", t.name);
        }
    }

    /// Tied-веса: два тензора на одном сторадже → алиасы, одно хранилище.
    #[test]
    fn tied_weights_dedup() {
        // Собираем синтетический GGUF: header + 0 KV + 2 тензора на одном offset
        let mut buf = Vec::new();
        buf.extend_from_slice(&crate::gguf::GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&2u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&0u64.to_le_bytes()); // kv_count
        let mut infos = Vec::new();
        for name in ["token_embd.weight", "output.weight"] {
            // name string
            infos.extend_from_slice(&(name.len() as u64).to_le_bytes());
            infos.extend_from_slice(name.as_bytes());
            infos.extend_from_slice(&1u32.to_le_bytes()); // n_dims
            infos.extend_from_slice(&16u64.to_le_bytes()); // dims[0] = 16
            infos.extend_from_slice(&0u32.to_le_bytes()); // F32
            infos.extend_from_slice(&0u64.to_le_bytes()); // offset (tied)
        }
        let data: Vec<u8> = (0..64).flat_map(|i| (i as f32).to_le_bytes()).collect();
        buf.extend_from_slice(&infos);
        buf.extend_from_slice(&data);

        let f = crate::gguf::parse(&buf).unwrap();
        let m = WeightsMap::from_gguf(&f).unwrap();
        let t0 = m.get("token_embd.weight").unwrap();
        let t1 = m.get("output.weight").unwrap();
        assert_eq!(t0.format, WeightFormat::Plain(Dtype::F32));
        assert_eq!(
            t0.data.as_ptr(),
            t1.data.as_ptr(),
            "tied storage must be shared"
        );
        assert_eq!(m.tied_aliases("token_embd.weight"), vec!["output.weight"]);
        assert_eq!(m.tied_aliases("output.weight"), vec!["token_embd.weight"]);
        assert_eq!(dequant_f32(t0), dequant_f32(t1));
    }
}
