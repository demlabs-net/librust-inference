//! Multi-file safetensors index loading + weight-norm helpers.
use anyhow::Context;
use std::collections::HashMap;
use std::path::Path;

pub fn load_safetensors_index<P: AsRef<Path>>(
    index_path: P,
) -> anyhow::Result<HashMap<String, Vec<String>>> {
    let content = std::fs::read_to_string(&index_path)
        .with_context(|| format!("reading {}", index_path.as_ref().display()))?;
    let index: serde_json::Value = serde_json::from_str(&content)?;
    let weight_map = index
        .get("weight_map")
        .and_then(|m| m.as_object())
        .ok_or_else(|| anyhow::anyhow!("missing weight_map"))?;
    let mut result: HashMap<String, Vec<String>> = HashMap::new();
    for (tensor_name, file_name) in weight_map {
        // A non-string entry is a malformed index — fail loudly instead of
        // silently mapping the tensor to the "" file (obscure "file not
        // found" later).
        let file = file_name
            .as_str()
            .ok_or_else(|| {
                anyhow::anyhow!("weight_map[{tensor_name}]: expected string, got {file_name}")
            })?
            .to_string();
        result.entry(file).or_default().push(tensor_name.clone());
    }
    Ok(result)
}

// ─────────────────────── safetensors (fp32 reader) ───────────────────────

/// Результат чтения safetensors: имя тензора → (данные f32, shape).
pub type F32Tensors = HashMap<String, (Vec<f32>, Vec<usize>)>;

/// Read all fp32 tensors from a safetensors file: name → (data, shape).
/// Bounds-checked: a truncated/corrupt file (partial download, cut-off
/// write) yields an error instead of a slicing panic inside a plugin.
/// F32 native; F16/BF16 (e.g. the DAC codec checkpoint) are widened to f32
/// here so every consumer sees fp32 host data.
pub fn read_safetensors_f32(path: &Path) -> anyhow::Result<F32Tensors> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 8 {
        return Err(anyhow::anyhow!(
            "safetensors: file too short ({} bytes)",
            bytes.len()
        ));
    }
    let header_len = u64::from_le_bytes(bytes[..8].try_into()?) as usize;
    let header_end = 8usize
        .checked_add(header_len)
        .filter(|&e| e <= bytes.len())
        .ok_or_else(|| anyhow::anyhow!("safetensors: header len {header_len} exceeds file size"))?;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..header_end])?;
    let weight_map = header
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("invalid safetensors header"))?;
    let mut out = HashMap::new();
    for (name, info) in weight_map {
        if name == "__metadata__" {
            continue;
        }
        let offsets = info
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow::anyhow!("missing data_offsets for {name}"))?;
        let off = |i: usize| -> anyhow::Result<usize> {
            let o = offsets
                .get(i)
                .and_then(|v| v.as_u64())
                .ok_or_else(|| anyhow::anyhow!("bad data_offsets for {name}"))?
                as usize;
            o.checked_add(8)
                .and_then(|b| b.checked_add(header_len))
                .ok_or_else(|| anyhow::anyhow!("data offset overflow for {name}"))
        };
        let start = off(0)?;
        let end = off(1)?;
        let shape: Vec<usize> = info
            .get("shape")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_u64().map(|n| n as usize))
                    .collect()
            })
            .unwrap_or_default();
        let dtype = info.get("dtype").and_then(|v| v.as_str()).unwrap_or("F32");
        if shape.is_empty() {
            continue;
        }
        let data = bytes.get(start..end).ok_or_else(|| {
            anyhow::anyhow!("tensor {name}: byte range {start}..{end} out of bounds")
        })?;
        let n = match dtype {
            "F32" | "U32" => data.len() / 4,
            "F16" | "BF16" => data.len() / 2,
            // I64 — константы (position_ids и т.п.); widening до f32
            "I64" => data.len() / 8,
            other => {
                return Err(anyhow::anyhow!(
                    "tensor {name}: unsupported dtype {other} (F32/F16/BF16 only)"
                ));
            }
        };
        let mut vec = Vec::with_capacity(n);
        if dtype == "F32" {
            for chunk in data.chunks_exact(4) {
                vec.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        } else if dtype == "U32" {
            // U32 → f32 как числа (индексы/кванты; widening для совместимости)
            for chunk in data.chunks_exact(4) {
                vec.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32);
            }
        } else if dtype == "I64" {
            // I64 → f32 (position_ids и др. целые константы; widening)
            for chunk in data.chunks_exact(8) {
                let mut b = [0u8; 8];
                b.copy_from_slice(chunk);
                vec.push(i64::from_le_bytes(b) as f32);
            }
        } else if dtype == "F16" {
            for chunk in data.chunks_exact(2) {
                let h = u16::from_le_bytes([chunk[0], chunk[1]]);
                vec.push(f16_to_f32(h));
            }
        } else {
            for chunk in data.chunks_exact(2) {
                let h = u16::from_le_bytes([chunk[0], chunk[1]]);
                vec.push(f32::from_bits((h as u32) << 16));
            }
        }
        out.insert(name.clone(), (vec, shape));
    }
    Ok(out)
}

/// IEEE-754 half → f32 (bit-level, no `half` crate needed).
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let frac = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            // subnormal: normalize until the hidden bit appears
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
        (sign << 31) | ((exp + 112) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

// ─────────────────── по-тензорный доступ (конверсия Ф4.12) ───────────────────

/// Строка заголовка safetensors: (имя, dtype, shape, начало и КОНЕЦ диапазона
/// в файле). Конец обязателен: без него читатель доверял только shape и мог
/// молча выйти за пределы своего тензора.
type StTensorInfo = (String, String, Vec<usize>, usize, usize);

/// Парсинг заголовка safetensors (имена + оффсеты, без данных).
fn safetensors_header(path: &Path) -> anyhow::Result<Vec<StTensorInfo>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 8 {
        anyhow::bail!("safetensors: file too short ({} bytes)", bytes.len());
    }
    let header_len = u64::from_le_bytes(bytes[..8].try_into()?) as usize;
    let header_end = 8usize
        .checked_add(header_len)
        .filter(|&e| e <= bytes.len())
        .ok_or_else(|| anyhow::anyhow!("safetensors: header len {header_len} exceeds file size"))?;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..header_end])?;
    let map = header
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("invalid safetensors header"))?;
    let mut out = Vec::with_capacity(map.len());
    for (name, info) in map {
        if name == "__metadata__" {
            continue;
        }
        let dtype = info
            .get("dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("F32")
            .to_string();
        let shape: Vec<usize> = info
            .get("shape")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_u64().map(|n| n as usize))
                    .collect()
            })
            .unwrap_or_default();
        let offsets = info
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow::anyhow!("missing data_offsets for {name}"))?;
        // checked-арифметика как в read_safetensors_f32: битый заголовок
        // (обрезанный файл) — ошибка, а не переполнение/паника.
        let raw = offsets
            .first()
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow::anyhow!("bad data_offsets for {name}"))? as usize;
        let start = raw
            .checked_add(8)
            .and_then(|s| s.checked_add(header_len))
            .ok_or_else(|| anyhow::anyhow!("data offset overflow for {name}"))?;
        // Конец диапазона тоже нужен: без него читатель брал размер из shape
        // и не сверял его с заголовком, поэтому лживый/битый заголовок
        // приводил к молчаливому чтению ЧУЖИХ байтов (соседнего тензора).
        let raw_end = offsets
            .get(1)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow::anyhow!("bad data_offsets[1] for {name}"))? as usize;
        let end = raw_end
            .checked_add(8)
            .and_then(|e| e.checked_add(header_len))
            .ok_or_else(|| anyhow::anyhow!("data offset overflow for {name}"))?;
        if end < start {
            anyhow::bail!("tensor {name}: data_offsets[1] < data_offsets[0]");
        }
        out.push((name.clone(), dtype, shape, start, end));
    }
    Ok(out)
}

/// Имена тензоров safetensors (без чтения данных).
pub fn safetensors_tensor_names(path: &Path) -> anyhow::Result<Vec<String>> {
    Ok(safetensors_header(path)?
        .into_iter()
        .map(|(n, _, _, _, _)| n)
        .collect())
}

/// Число элементов тензора по имени.
pub fn safetensors_tensor_numel(path: &Path, name: &str) -> anyhow::Result<usize> {
    let h = safetensors_header(path)?;
    h.into_iter()
        .find(|(n, _, _, _, _)| n == name)
        .map(|(_, _, shape, _, _)| {
            shape
                .iter()
                .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                .ok_or_else(|| anyhow::anyhow!("tensor {name}: numel overflow"))
        })
        .transpose()?
        .ok_or_else(|| anyhow::anyhow!("tensor {name} not found"))
}

/// Прочитать ОДИН тензор с диска (f32, widened из F16/BF16/I64/U32).
/// Для конверсий: память = один тензор, а не весь файл.
pub fn read_safetensors_tensor_f32(
    path: &Path,
    name: &str,
) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
    let h = safetensors_header(path)?;
    let (_, dtype, shape, start, declared_end) = h
        .into_iter()
        .find(|(n, _, _, _, _)| n == name)
        .ok_or_else(|| anyhow::anyhow!("tensor {name} not found"))?;
    let bytes = std::fs::read(path)?;
    // checked product (битый shape в заголовке → чистый Err, не wrap).
    let numel: usize = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| anyhow::anyhow!("tensor {name}: numel overflow"))?;
    let elem = match dtype.as_str() {
        "F32" | "U32" => 4,
        "F16" | "BF16" => 2,
        "I64" => 8,
        other => anyhow::bail!("tensor {name}: unsupported dtype {other}"),
    };
    // checked (как в стриминг-ридере): битый shape → ошибка, не wrap.
    let nbytes = numel
        .checked_mul(elem)
        .ok_or_else(|| anyhow::anyhow!("tensor {name}: size overflow"))?;
    let end = start
        .checked_add(nbytes)
        .ok_or_else(|| anyhow::anyhow!("overflow for {name}"))?;
    // Сверяем вычисленный размер с границей из заголовка: shape/dtype могут
    // не соответствовать data_offsets, и тогда чтение ушло бы в соседний
    // тензор, не вызвав ни одной ошибки.
    if end > declared_end {
        anyhow::bail!(
            "tensor {name}: {nbytes} B по shape/dtype не влезает в диапазон заголовка              ({} B)",
            declared_end.saturating_sub(start)
        );
    }
    let data = bytes
        .get(start..end)
        .ok_or_else(|| anyhow::anyhow!("tensor {name}: byte range out of bounds"))?;
    let mut vec = Vec::with_capacity(numel);
    match dtype.as_str() {
        "F32" => {
            for chunk in data.chunks_exact(4) {
                vec.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        }
        "U32" => {
            for chunk in data.chunks_exact(4) {
                vec.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32);
            }
        }
        "I64" => {
            for chunk in data.chunks_exact(8) {
                let mut b = [0u8; 8];
                b.copy_from_slice(chunk);
                vec.push(i64::from_le_bytes(b) as f32);
            }
        }
        "F16" => {
            for chunk in data.chunks_exact(2) {
                let h = u16::from_le_bytes([chunk[0], chunk[1]]);
                vec.push(f16_to_f32(h));
            }
        }
        _ => {
            for chunk in data.chunks_exact(2) {
                let h = u16::from_le_bytes([chunk[0], chunk[1]]);
                vec.push(f32::from_bits((h as u32) << 16));
            }
        }
    }
    Ok((vec, shape))
}
