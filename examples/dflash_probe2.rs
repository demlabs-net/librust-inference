//! Диагностика v2: диффузия с РЕАЛЬНЫМИ фичами таргета vs таргет-декод.
//! Использование: dflash_probe2 <target.gguf> <draft.safetensors>
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
    let ew = dequant_f32(wm_t.get("token_embd.weight").unwrap());
    let embed_t = backend.from_cpu_f32(&ew, vec![vocab, cfg_t.dim])?;
    let ow = dequant_f32(
        wm_t.get("output.weight")
            .unwrap_or(wm_t.get("token_embd.weight").unwrap()),
    );
    let f16: Vec<half::f16> = ow.iter().map(|&x| half::f16::from_f32(x)).collect();
    let lm_head = backend.from_cpu_f16(&f16, vec![vocab, cfg_t.dim])?;
    dmod.set_target_embed(embed_t);

    // фраза: таргет-префилл, фичи extract-слоёв (входы [1,9,17,25,33])
    let tokz = rust_inference::tokenizer::Tokenizer::from_gguf(&gf_t)?;
    let phrase = "The capital of France is";
    let ids: Vec<u32> = tokz.encode(phrase);
    eprintln!("phrase tokens: {:?}", ids);
    let n = cfg_t.vocab_size;
    // 1) таргет-префилл фразы → последний токен предсказывает " Paris"
    let lg = tgt_model.prefill(&backend, &ids)?;
    let v = backend.to_cpu_f32(&lg)?;
    let last = &v[v.len() - n..];
    eprintln!("TARGET after {:?}: top5={:?}", phrase, top5(last));
    // 2) фичи extract-слоёв (входы [1,9,17,25,33]) для последних 4 токенов
    let tl = dmod.extract_layers.clone();
    let n_ctx = ids.len(); // вся фраза
    let sub = &ids[..];
    let feats = tgt_model.forward_extract(&backend, sub, &tl)?;
    let dd = dmod.cfg.dim;
    let nl = feats.len();
    let mut flat = vec![0f32; n_ctx * nl * dd];
    for (li, h) in feats.iter().enumerate() {
        let hv = backend.to_cpu_f32(h)?;
        for tok in 0..n_ctx {
            flat[tok * nl * dd + li * dd..tok * nl * dd + (li + 1) * dd]
                .copy_from_slice(&hv[tok * dd..(tok + 1) * dd]);
        }
    }
    let ft = backend.from_cpu_f32(&flat, vec![n_ctx, nl * dd])?;
    // 3) диффузия [последний-токен, MASK×15] с фичами → первая позиция
    let anchor = *ids.last().unwrap();
    let pos = ids.len(); // z-lab: блок на позициях [n..n+16)
    let mut bid = vec![anchor];
    bid.resize(16, dmod.mask_token);
    let r = dmod.diffuse(&backend, &bid, &lm_head, pos, Some(&ft))?;
    let vd = backend.to_cpu_f32(&r.logits)?;
    let row0 = &vd[..n];
    eprintln!("DRAFT block pos0 (ctx={n_ctx}): top5={:?}", top5(row0));
    // 4) без фич — сравнение
    let r2 = dmod.diffuse(&backend, &bid, &lm_head, pos, None)?;
    let v2 = backend.to_cpu_f32(&r2.logits)?;
    eprintln!("DRAFT block pos0 (no-ctx): top5={:?}", top5(&v2[..n]));
    Ok(())
}
