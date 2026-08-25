//! Диагностика DFlash: диффузия t=1 (AR-шаг драфтера) vs таргет-декод.
//! Использование: dflash_probe <target.gguf> <draft.gguf|.safetensors>
#![cfg(feature = "metal")]
use std::error::Error;
use std::fs::File;
use rust_inference::backend::Backend as _;
use rust_inference::dflash::DFlashModel;
use rust_inference::gguf;
use rust_inference::metal::MetalBackend;
use rust_inference::qwen3::{Qwen3Config, Qwen3Model};
use rust_inference::weights_map::{dequant_f32, WeightsMap};

fn top5(row: &[f32]) -> Vec<(usize, f32)> {
    let mut v: Vec<(usize, f32)> = row.iter().enumerate().map(|(i, &x)| (i, x)).collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    v.into_iter().take(5).collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
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
    let ow = dequant_f32(wm_t.get("output.weight").unwrap_or(wm_t.get("token_embd.weight").unwrap()));
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
