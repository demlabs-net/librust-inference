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
    data.chunks_exact(18)
        .flat_map(|blk| {
            let d = f16_to_f32([blk[0], blk[1]]);
            blk[2..]
                .iter()
                .flat_map(move |&b| [((b & 0x0F) as f32 - 8.0) * d, ((b >> 4) as f32 - 8.0) * d])
        })
        .collect()
}

pub(crate) fn dequant_q8_0(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(34)
        .flat_map(|blk| {
            let d = f16_to_f32([blk[0], blk[1]]);
            blk[2..].iter().map(move |&b| (b as i8 as f32) * d)
        })
        .collect()
}

pub(crate) fn dequant_q4_k(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(148)
        .flat_map(|sb| {
            let d = f16_to_f32([sb[0], sb[1]]);
            let dmin = f16_to_f32([sb[2], sb[3]]);
            let min = d * dmin;
            let mut sc = [0u8; 16];
            for j in 0..16 {
                sc[j] = (sb[4 + j / 2] >> ((j % 2) * 4)) & 0x0F;
            }
            let qs = &sb[20..148];
            (0..8).flat_map(move |b| {
                let s = sc[b] as f32;
                qs[b * 16..b * 16 + 16].iter().flat_map(move |&q| {
                    [
                        min + d * s * ((q & 0x0F) as f32 - 8.0),
                        min + d * s * ((q >> 4) as f32 - 8.0),
                    ]
                })
            })
        })
        .collect()
}

pub(crate) fn dequant_q6_k(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(210)
        .flat_map(|sb| {
            let d = f16_to_f32([sb[0], sb[1]]);
            let ql = &sb[2..130];
            let qh = &sb[130..194];
            let scales = &sb[194..210];
            (0..8).flat_map(move |b| {
                let sc = ((scales[b] & 0x0F) | ((scales[b + 8] & 0x0F) << 4)) as f32;
                (0..32).map(move |j| {
                    let idx = b * 32 + j;
                    let lo = if j % 2 == 0 {
                        ql[idx / 2] & 0x0F
                    } else {
                        ql[idx / 2] >> 4
                    };
                    let hi = (qh[idx / 4] >> ((idx % 4) * 2)) & 0x03;
                    let q = (lo | (hi << 4)) as f32;
                    d * sc * (q - 32.0)
                })
            })
        })
        .collect()
}

/// Деквантовать GGUF-тензор (сырой ggml-тип) в f32 — для re-quant (Ф4.12).
pub(crate) fn dequant_gguf(
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
        WeightFormat::Int4 | WeightFormat::Nf4 => {
            eprintln!(
                "dequant_f32: {} dequant not implemented — returning EMPTY weights (caller must reject the format)",
                match t.format {
                    WeightFormat::Int4 => "Int4",
                    _ => "Nf4",
                }
            );
            Vec::new()
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
