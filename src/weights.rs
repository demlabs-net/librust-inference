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
