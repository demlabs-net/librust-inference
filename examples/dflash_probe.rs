//! Диагностика DFlash: диффузия t=1 (AR-шаг драфтера) vs таргет-декод.
//! Использование: dflash_probe <target.gguf> <draft.gguf|.safetensors>
#![cfg(feature = "metal")]
use rust_inference::backend::Backend as _;
use rust_inference::dflash::DFlashModel;
use rust_inference::gguf;
use rust_inference::metal::MetalBackend;
use rust_inference::qwen3::{Qwen3Config, Qwen3Model};
use rust_inference::weights_map::{WeightsMap, dequant_f32};
use std::error::Error;
use std::fs::File;

fn top5(row: &[f32]) -> Vec<(usize, f32)> {
    let mut v: Vec<(usize, f32)> = row.iter().enumerate().map(|(i, &x)| (i, x)).collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    v.into_iter().take(5).collect()
}

// Дополнительно: --convert-zlab-q8 — переквантование актуального zlab
// драфтера (safetensors) в Q8_0 GGUF (старый Q8_0.gguf — ДРУГИЕ веса).

#[allow(dead_code)]
pub fn convert_zlab_q8() -> Result<(), Box<dyn Error>> {
    use rust_inference::gguf::{self, GgufValue, GgufWriteTensor, GgmlType, write_gguf};
    use rust_inference::quant::quant_q8_0;
    let home = std::env::var("HOME").unwrap();
    let old = format!("{home}/work/ai/inference/models/dflash/qwen3-8b-dflash-Q8_0.gguf");
    let st = format!("{home}/work/ai/inference/models/dflash/qwen3-8b-dflash-zlab.safetensors");
    let out = format!("{home}/work/ai/inference/models/dflash/qwen3-8b-dflash-zlab-Q8_0.gguf");
    let buf = std::fs::read(&old)?;
    let gf = gguf::parse(&buf)?;
    let metadata: Vec<(String, GgufValue)> = gf
        .metadata
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let st_w = rust_inference::weights::read_safetensors_f32(std::path::Path::new(&st))?;
    // та же карта имён, что в DFlashModel::load_safetensors
    let map = |name: &str| -> Option<String> {
        if name == "fc.weight" {
            return Some("fc.weight".into());
        }
        if name == "norm.weight" {
            return Some("output_norm.weight".into());
        }
        if name == "hidden_norm.weight" {
            return Some("enc.output_norm.weight".into());
        }
        let parts: Vec<&str> = name.split('.').collect();
        if parts.len() >= 4 && parts[0] == "layers" {
            let rest = parts[2..].join(".");
            let m = match rest.as_str() {
                "input_layernorm.weight" => "attn_norm.weight",
                "post_attention_layernorm.weight" => "ffn_norm.weight",
                "self_attn.q_proj.weight" => "attn_q.weight",
                "self_attn.k_proj.weight" => "attn_k.weight",
                "self_attn.v_proj.weight" => "attn_v.weight",
                "self_attn.o_proj.weight" => "attn_output.weight",
                "self_attn.q_norm.weight" => "attn_q_norm.weight",
                "self_attn.k_norm.weight" => "attn_k_norm.weight",
                "mlp.gate_proj.weight" => "ffn_gate.weight",
                "mlp.up_proj.weight" => "ffn_up.weight",
                "mlp.down_proj.weight" => "ffn_down.weight",
                _ => return None,
            };
            return Some(format!("blk.{}.{}", parts[1], m));
        }
        None
    };
    let mut tensors = Vec::new();
    for (name, (data, shape)) in &st_w {
        let mapped = map(name).ok_or_else(|| format!("незнакомый тензор {name}"))?;
        tensors.push(GgufWriteTensor {
            name: mapped,
            dims: shape.iter().map(|&d| d as u32).collect(),
            ggml_type: GgmlType::Q8_0,
            data: quant_q8_0(data),
        });
    }
    let mut outbuf = Vec::new();
    write_gguf(&mut outbuf, &metadata, &tensors)?;
    std::fs::write(&out, &outbuf)?;
    eprintln!("записан {out} ({} байт, {} тензоров)", outbuf.len(), tensors.len());
    Ok(())
}

#[allow(dead_code)]
pub fn verify_zlab_q8() -> Result<(), Box<dyn Error>> {
    use rust_inference::weights_map::{WeightsMap, dequant_f32};
    let home = std::env::var("HOME").unwrap();
    let p = format!("{home}/work/ai/inference/models/dflash/qwen3-8b-dflash-zlab-Q8_0.gguf");
    let buf = std::fs::read(&p)?;
    let gf = gguf::parse(&buf)?;
    let wm = WeightsMap::from_gguf(&gf)?;
    let qk = dequant_f32(wm.get("blk.0.attn_k.weight").unwrap());
    let st = rust_inference::weights::read_safetensors_f32(
        &std::path::PathBuf::from(format!("{home}/work/ai/inference/models/dflash/qwen3-8b-dflash-zlab.safetensors")),
    )?;
    let ref_data = &st["layers.0.self_attn.k_proj.weight"];
    let mean_sq = ref_data.0.iter().map(|x| x * x).sum::<f32>() / ref_data.0.len() as f32;
    let mut max_d = 0.0f32;
    let mut sum_d = 0.0f32;
    for (a, b) in qk.iter().zip(ref_data.0.iter()) {
        let d = (a - b).abs();
        max_d = max_d.max(d);
        sum_d += d;
    }
    eprintln!("Q8-vs-zlab: max {max_d:.4} mean {} ref-rms {}", sum_d / ref_data.0.len() as f32, mean_sq.sqrt());
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(|a| a == "--convert-zlab-q8").unwrap_or(false) {
        return convert_zlab_q8();
    }
    if args.get(1).map(|a| a == "--verify-zlab-q8").unwrap_or(false) {
        return verify_zlab_q8();
    }
    let (tgt, drf) = (&args[1], &args[2]);
    let backend = MetalBackend::new()?;
    let mmap_t = unsafe { memmap2::Mmap::map(&File::open(tgt)?)? };
    let gf_t = gguf::parse(&mmap_t)?;
    let wm_t = WeightsMap::from_gguf(&gf_t)?;
    let vocab = wm_t.get("token_embd.weight").unwrap().shape[0] as usize;
    let cfg_t = Qwen3Config::from_gguf(&gf_t, vocab)?;
    let mut tgt_model = Qwen3Model::load(&backend, &wm_t, cfg_t.clone())?;

    let mut dmod = if drf.ends_with(".safetensors") {
        DFlashModel::load_safetensors(&backend, std::path::Path::new(drf), vocab)?
    } else {
        let mmap_d = unsafe { memmap2::Mmap::map(&File::open(drf)?)? };
        let gf_d = gguf::parse(&mmap_d)?;
        let wm_d = WeightsMap::from_gguf(&gf_d)?;
        DFlashModel::load_gguf(&backend, &gf_d, &wm_d, vocab)?
    };
    eprintln!(
        "drafter: layers={} dim={} block={} extract={:?} mask={}",
        dmod.cfg.n_layers, dmod.cfg.dim, dmod.block_size, dmod.extract_layers, dmod.mask_token
    );

    let ew = dequant_f32(wm_t.get("token_embd.weight").unwrap());
    let embed_t = backend.from_cpu_f32(&ew, vec![vocab, cfg_t.dim])?;
    let ow = dequant_f32(
        wm_t.get("output.weight")
            .unwrap_or(wm_t.get("token_embd.weight").unwrap()),
    );
    let f16: Vec<half::f16> = ow.iter().map(|&x| half::f16::from_f32(x)).collect();
    let lm_head = backend.from_cpu_f16(&f16, vec![vocab, cfg_t.dim])?;
    dmod.set_target_embed(embed_t);

    // токен для пробы: <|im_start|> (151644) — известный контекст
    let id = 151644u32;
    let lg_t = tgt_model.decode(&backend, id)?;
    let v_t = backend.to_cpu_f32(&lg_t)?;
    let n = cfg_t.vocab_size;
    let last = &v_t[v_t.len() - n..];
    eprintln!("TARGET decode({id}): top5={:?}", top5(last));
    let r = dmod.diffuse(&backend, &[id], &lm_head, 1, None)?;
    let v_d = backend.to_cpu_f32(&r.logits)?;
    eprintln!("DRAFT  diffuse([{id}]): top5={:?}", top5(&v_d[..n]));
    // блок-диффузия: [id, MASK×15] — первая MASK-позиция vs таргет
    {
        let mut ids = vec![id];
        ids.resize(16, dmod.mask_token);
        let r = dmod.diffuse(&backend, &ids, &lm_head, 1, None)?;
        let v = backend.to_cpu_f32(&r.logits)?;
        let row0 = &v[..n];
        let row1 = &v[n..2 * n];
        eprintln!("DRAFT block pos0: top5={:?}", top5(row0));
        eprintln!("DRAFT block pos1: top5={:?}", top5(row1));
    }
    // диффузия на разных id (контекст влияет?)
    for id2 in [151645u32, 1234, 5] {
        let r = dmod.diffuse(&backend, &[id2], &lm_head, 1, None)?;
        let v = backend.to_cpu_f32(&r.logits)?;
        eprintln!("DRAFT diffuse([{id2}]): top3={:?}", &top5(&v[..n])[..3]);
    }
    // куски токенов
    let tokz = rust_inference::tokenizer::Tokenizer::from_gguf(&gf_t)?;
    for t in [11u32, 13, 198, 220, 67] {
        eprintln!("piece({t}) = {:?}", String::from_utf8_lossy(tokz.piece(t)));
    }
    Ok(())
}
