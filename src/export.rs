//! 5.6 — экспорт модели: база (GGUF) + LoRA-адаптер → merged GGUF
//! (f32/f16/bf16/Q4_0/Q8_0/Q4_K_M) нашим writer'ом (1.6b).
//!
//! Формат адаптера — тот же, что экспортирует `train_qwen3`:
//! `adapter.json` c {model, trainer, rank, alpha, base, adapters: {имя
//! тензора: {a, b}}}. Merge: W' = W + α·B·A (α = alpha/rank; при хранении
//! в обучающих адаптерах α = 1.0 с alpha = rank).

use std::path::Path;

use crate::gguf::{GgmlType, GgufValue, GgufWriteTensor, write_gguf};
use crate::quant::{QuantFormat, quantize};
use crate::train_qwen3::AdapterFile;
use crate::weights_map::{WeightsMap, dequant_f32};

/// Маппинг целевого формата → GGML-тип writer'а.
pub fn ggml_type_of(fmt: QuantFormat) -> GgmlType {
    match fmt {
        QuantFormat::F32 => GgmlType::F32,
        QuantFormat::F16 => GgmlType::F16,
        QuantFormat::Bf16 => GgmlType::Bf16,
        QuantFormat::Q4_0 => GgmlType::Q4_0,
        QuantFormat::Q8_0 => GgmlType::Q8_0,
        QuantFormat::Q4_K_M => GgmlType::Q4K,
    }
}

pub fn parse_quant_format(s: &str) -> Option<QuantFormat> {
    match s.to_ascii_lowercase().as_str() {
        "f32" | "fp32" => Some(QuantFormat::F32),
        "f16" | "fp16" => Some(QuantFormat::F16),
        "bf16" => Some(QuantFormat::Bf16),
        "q4_0" | "q40" => Some(QuantFormat::Q4_0),
        "q8_0" | "q80" => Some(QuantFormat::Q8_0),
        "q4_k" | "q4_k_m" | "q4km" => Some(QuantFormat::Q4_K_M),
        _ => None,
    }
}

/// Применить адаптер к весам: delta[out, in] = α·B·A (B [out,r], A [r,in]).
fn apply_adapter(
    w: &mut [f32],
    a: &[f32],
    b: &[f32],
    out: usize,
    inp: usize,
    rank: usize,
    alpha: f32,
) {
    for l in 0..out {
        for k in 0..inp {
            let mut acc = 0.0f32;
            for r in 0..rank {
                acc += b[l * rank + r] * a[r * inp + k];
            }
            w[l * inp + k] += alpha * acc;
        }
    }
}

/// Merged-экспорт. Память: файл + по одному развёрнутому тензору за раз
/// (для 8B-подобных файлов — считайте по VRAM/RAM; стриминговый райтер — v2).
#[allow(clippy::too_many_arguments)]
pub fn export_merged_gguf(
    base_path: &Path,
    adapter_path: &Path,
    out_path: &Path,
    fmt: QuantFormat,
) -> anyhow::Result<()> {
    let base_bytes = std::fs::read(base_path)
        .map_err(|e| anyhow::anyhow!("read {}: {e}", base_path.display()))?;
    let file = crate::gguf::parse(&base_bytes)?;
    let wm = WeightsMap::from_gguf(&file)?;

    let adapter_text = std::fs::read_to_string(adapter_path)
        .map_err(|e| anyhow::anyhow!("read {}: {e}", adapter_path.display()))?;
    let adapter: AdapterFile = serde_json::from_str(&adapter_text)
        .map_err(|e| anyhow::anyhow!("adapter.json {}: {e}", adapter_path.display()))?;
    let alpha = (adapter.alpha / adapter.rank as f64) as f32;
    if !alpha.is_finite() || alpha <= 0.0 {
        anyhow::bail!("adapter.json: невалидная alpha/rank");
    }

    let ggml_type = ggml_type_of(fmt);
    let n_tensors = file.tensors.len();
    let mut writes = Vec::with_capacity(n_tensors);
    let mut total_bytes = 0u64;
    let mut with_adapter = 0usize;
    for t in &file.tensors {
        let wname = &t.name;
        let wt = wm
            .get(wname)
            .ok_or_else(|| anyhow::anyhow!("база без тензора {} (несовместимый файл)", wname))?;
        // логический размер: [out, in] (внутренний порядок WeightsMap)
        let shape: Vec<usize> = wt.shape.iter().map(|&s| s as usize).collect();
        let mut w = dequant_f32(wt);
        if let Some(pair) = adapter.adapters.get(wname) {
            // [out, in]: a [r, inp], b [out, r]
            let rank = adapter.rank;
            let (out, inp) = (pair.b.len() / rank, pair.a.len() / rank);
            if rank == 0
                || pair.a.len() % rank != 0
                || pair.b.len() % rank != 0
                || out * inp != w.len()
            {
                anyhow::bail!(
                    "adapter {}: размеры a/b не совпадают с весами {:?}",
                    wname,
                    shape
                );
            }
            apply_adapter(&mut w, &pair.a, &pair.b, out, inp, rank, alpha);
            with_adapter += 1;
        }
        let data = quantize(&w, fmt);
        total_bytes += data.len() as u64;
        writes.push(GgufWriteTensor {
            name: wname.clone(),
            dims: t.dims.iter().map(|&d| d as u32).collect(),
            ggml_type,
            data,
        });
    }

    // Метаданные базы — как есть (архитектура/токенайзер/конфиг) плюс
    // пометка об экспорте.
    let mut metadata = file.metadata.clone();
    metadata.push((
        "inference.merged".to_string(),
        GgufValue::String(adapter.model.clone()),
    ));
    metadata.push((
        "inference.lora_rank".to_string(),
        GgufValue::U32(adapter.rank as u32),
    ));

    let mut out = Vec::with_capacity(total_bytes as usize);
    write_gguf(&mut out, &metadata, &writes)?;
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(out_path, &out)?;
    tracing::info!(
        tensors = n_tensors,
        adapted = with_adapter,
        bytes = total_bytes,
        fmt = ?fmt,
        "merged export done"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GgufValue, parse};

    fn tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("export-{}-{}", std::process::id(), name))
    }

    fn write_tiny_base(path: &Path) {
        let mut buf = Vec::new();
        let w1: Vec<f32> = (0..12).map(|i| i as f32 + 0.5).collect();
        let w2: Vec<f32> = (0..6).map(|i| i as f32 * 2.0).collect();
        let bytes1: Vec<u8> = w1.iter().flat_map(|x| x.to_le_bytes()).collect();
        let bytes2: Vec<u8> = w2.iter().flat_map(|x| x.to_le_bytes()).collect();
        write_gguf(
            &mut buf,
            &[
                (
                    "general.architecture".into(),
                    GgufValue::String("tiny".into()),
                ),
                ("tiny.vocab_size".into(), GgufValue::U32(17)),
            ],
            &[
                GgufWriteTensor {
                    name: "w1.weight".into(),
                    dims: vec![4, 3],
                    ggml_type: GgmlType::F32,
                    data: bytes1,
                },
                GgufWriteTensor {
                    name: "w2.weight".into(),
                    dims: vec![3, 2],
                    ggml_type: GgmlType::F32,
                    data: bytes2,
                },
            ],
        )
        .unwrap();
        std::fs::write(path, &buf).unwrap();
    }

    #[test]
    fn merged_export_applies_adapter_and_preserves_others() {
        let base = tmp("base.gguf");
        let ad = tmp("adapter.json");
        let out = tmp("merged.gguf");
        write_tiny_base(&base);

        // адаптер только на w1: rank=2, alpha=2 (α=1.0)
        let file = AdapterFile {
            model: "tiny-model".into(),
            trainer: "qwen3-lora".into(),
            rank: 2,
            alpha: 2.0,
            base: None,
            adapters: {
                let mut m = std::collections::BTreeMap::new();
                // a [2, 4] = [[1,0,0,0],[0,1,0,0]]; b [3, 2] = [[1,0],[0,1],[1,1]]
                m.insert(
                    "w1.weight".into(),
                    crate::train_qwen3::AdapterPair {
                        a: vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        b: vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                    },
                );
                m
            },
        };
        let mut f = std::fs::File::create(&ad).unwrap();
        serde_json::to_writer_pretty(&mut f, &file).unwrap();

        export_merged_gguf(&base, &ad, &out, QuantFormat::F32).unwrap();

        let out_bytes = std::fs::read(&out).unwrap();
        let f = parse(&out_bytes).unwrap();
        assert_eq!(f.architecture(), Some("tiny"));
        assert_eq!(
            f.get("tiny.vocab_size").and_then(GgufValue::as_u32),
            Some(17)
        );
        assert_eq!(
            f.get("inference.lora_rank").and_then(GgufValue::as_u32),
            Some(2)
        );
        assert_eq!(f.tensors.len(), 2);
        // w1: W + α·B·A → row0 += [1,0,0,0]·1, row1 += [0,1,0,0], row2 += [1,0]+[0,1]
        let t1 = f.tensors.iter().find(|t| t.name == "w1.weight").unwrap();
        let vals: Vec<f32> = t1
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(vals.len(), 12);
        // base w1: [0.5,1.5,2.5, 3.5,4.5,5.5, 6.5,7.5,8.5, 9.5,10.5,11.5]
        // lodka: row0 (k0)=+1 → 1.5; row1 (k1)=+1 → 5.5; row2 (k0,k1)=+1 → 7.5/8.5
        assert!((vals[0] - 1.5).abs() < 1e-6, "{vals:?}");
        assert!((vals[5] - 6.5).abs() < 1e-6, "{vals:?}");
        assert!((vals[6] - 6.5).abs() < 1e-6, "{vals:?}");
        assert!((vals[7] - 7.5).abs() < 1e-6, "{vals:?}");
        assert!((vals[8] - 9.5).abs() < 1e-6, "{vals:?}");
        assert!((vals[9] - 10.5).abs() < 1e-6, "{vals:?}");
        // неадаптируемый тензор — без изменений
        let t2 = f.tensors.iter().find(|t| t.name == "w2.weight").unwrap();
        let vals2: Vec<f32> = t2
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(vals2, vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);

        // f16-экспорт: парсится, тип F16, размеры байт × 1/2
        let out16 = tmp("merged16.gguf");
        export_merged_gguf(&base, &ad, &out16, QuantFormat::F16).unwrap();
        let bytes16 = std::fs::read(&out16).unwrap();
        let f16 = parse(&bytes16).unwrap();
        let t1_16 = f16.tensors.iter().find(|t| t.name == "w1.weight").unwrap();
        assert_eq!(t1_16.ggml_type, GgmlType::F16);
        assert_eq!(t1_16.data.len(), 24);

        let _ = std::fs::remove_file(&base);
        let _ = std::fs::remove_file(&ad);
        let _ = std::fs::remove_file(&out);
        let _ = std::fs::remove_file(&out16);
    }
}
