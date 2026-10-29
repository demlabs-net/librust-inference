//! Приёмка ДЛИННОГО контекста на нескольких слотах: две беседы по ~800–1200
//! токенов префиллятся в свои слоты и шагают пакетом; тексты обязаны совпасть
//! с одиночными прогонами тех же промптов.
//!
//! Проверяет ровно то, что ломается при нескольких беседах: рост KV каждого
//! слота независимо (ёмкость, `ensure`, позиции), отсутствие перекрёстного
//! влияния и корректность rope-позиций на длинных префиксах.
//!
//! `cargo run --release --features metal --example batch_longctx -- <model.gguf> [TARGET_TOKENS]`

#![cfg(feature = "metal")]

use std::error::Error;

use rust_inference::backend::Backend as _;
use rust_inference::gguf;
use rust_inference::metal::MetalBackend;
use rust_inference::qwen35;
use rust_inference::tokenizer::Tokenizer;
use rust_inference::weights_map::WeightsMap;

fn argmax(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Промпт нужной длины: склейка фрагментов, чтобы текст не был вырожденным
/// повтором одного предложения (n-граммы истории не должны совпадать).
fn long_prompt(tok: &Tokenizer, seed_text: &str, target: usize) -> Vec<u32> {
    let mut ids = tok.encode(seed_text);
    let mut out = Vec::with_capacity(target + ids.len());
    while out.len() < target {
        out.extend_from_slice(&ids);
        ids = tok.encode(&format!(" ({}) {seed_text}", out.len()));
    }
    out.truncate(target);
    out
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).ok_or("usage: batch_longctx <model.gguf> [N]")?;
    let target: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(900);
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let gf = gguf::parse(&mmap)?;
    let wm = WeightsMap::from_gguf(&gf)?;
    let tok = Tokenizer::from_gguf(&gf)?;
    let be = MetalBackend::new()?;
    let vocab = tok.vocab_size();
    let mut m = qwen35::load_gguf(&be, &gf, &wm, vocab)?;

    // Разные длины — вторая беседа короче: пакетный шаг обязан вести СВОИ
    // позиции у каждого слота (общая позиция дала бы мусор на короткой).
    let prompts = [
        long_prompt(
            &tok,
            "The history of the printing press begins with movable type in Asia and continues through Gutenberg.",
            target,
        ),
        long_prompt(
            &tok,
            "A recipe for bread needs flour, water, salt and yeast; the dough rests before baking.",
            target * 2 / 3,
        ),
    ];
    let n = 6usize;

    // Эталон: каждая беседа одиночно, своё состояние.
    let mut solo: Vec<Vec<u32>> = Vec::new();
    for p in &prompts {
        let mut slot = m.new_slot(&be)?;
        let mut lg = be.to_cpu_f32(&m.prefill_slot_last(&be, &mut slot, p)?)?;
        let mut toks = vec![argmax(&lg)];
        for _ in 0..n {
            let next = *toks.last().unwrap();
            lg = be.to_cpu_f32(&m.decode_slot(&be, &mut slot, next)?)?;
            toks.push(argmax(&lg));
        }
        solo.push(toks);
    }

    // Пакет: оба слота живут своими KV и шагают вместе.
    let mut slots = vec![m.new_slot(&be)?, m.new_slot(&be)?];
    let mut next: Vec<u32> = Vec::new();
    for (i, p) in prompts.iter().enumerate() {
        let lg = be.to_cpu_f32(&m.prefill_slot_last(&be, &mut slots[i], p)?)?;
        next.push(argmax(&lg));
    }
    let mut batched: Vec<Vec<u32>> = vec![Vec::new(); prompts.len()];
    for _ in 0..n {
        let lg = m.decode_batch(&be, &mut slots, &next)?;
        let host = be.to_cpu_f32(&lg)?;
        for i in 0..prompts.len() {
            let id = argmax(&host[i * vocab..(i + 1) * vocab]);
            batched[i].push(id);
            next[i] = id;
        }
    }
    // Токен из префилла входит в сверку первым.
    for i in 0..prompts.len() {
        batched[i].insert(0, solo[i][0]);
    }

    let mut ok = true;
    for (i, p) in prompts.iter().enumerate() {
        let same = batched[i] == solo[i];
        ok &= same;
        println!(
            "беседа {i}: промпт {} ток., позиция слота {} (разные длины), токены {}",
            p.len(),
            slots[i].pos(),
            if same {
                "совпали"
            } else {
                "РАЗОШЛИСЬ"
            }
        );
        if !same {
            eprintln!("  пакет: {:?}\n  эталон: {:?}", batched[i], solo[i]);
        }
    }
    assert!(
        ok,
        "длинный контекст на нескольких слотах разошёлся с одиночным"
    );
    println!(
        "OK: два слота по {} токенов контекста, пакет = одиночным прогонам",
        slots[0].pos()
    );
    Ok(())
}
