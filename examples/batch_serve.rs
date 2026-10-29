//! Приёмка драйвера пакетной генерации ([`rust_inference::batch::BatchEngine`]):
//! 6 бесед через 3 слота с ПЕРЕИСПОЛЬЗОВАНИЕМ мест — обязаны дать те же
//! жадные токены, что одиночные прогоны.
//!
//! `cargo run --release --features metal --example batch_serve -- <model.gguf> [CAP]`

#![cfg(feature = "metal")]

use std::collections::VecDeque;
use std::error::Error;

use rust_inference::backend::Backend as _;
use rust_inference::batch::BatchEngine;
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

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).ok_or("usage: batch_serve <model.gguf> [CAP]")?;
    let cap: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(3);
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let gf = gguf::parse(&mmap)?;
    let wm = WeightsMap::from_gguf(&gf)?;
    let tok = Tokenizer::from_gguf(&gf)?;
    let be = MetalBackend::new()?;
    let vocab = tok.vocab_size();
    let mut m = qwen35::load_gguf(&be, &gf, &wm, vocab)?;

    let prompts = [
        "The capital of France is Paris and the capital of Italy is",
        "2 + 2 = 4 and 3 + 3 = 6 and 4 + 4 =",
        "Once upon a time, in a small village by the sea, there lived a",
        "Water boils at 100 degrees Celsius and freezes at",
        "The quick brown fox jumps over the lazy dog. The dog was not amused, so the fox",
        "In 1969, humans first walked on the Moon during the Apollo",
    ];
    let max_new = 8usize;
    let ids: Vec<Vec<u32>> = prompts.iter().map(|p| tok.encode(p)).collect();

    // Эталон: каждая беседа одиночно, тем же жадным правилом.
    let mut solo: Vec<Vec<u32>> = Vec::new();
    for p in &ids {
        let mut slot = m.new_slot(&be)?;
        let mut lg = be.to_cpu_f32(&m.prefill_slot_last(&be, &mut slot, p)?)?;
        let mut toks = vec![argmax(&lg)];
        while toks.len() < max_new + 1 {
            let next = *toks.last().unwrap();
            lg = be.to_cpu_f32(&m.decode_slot(&be, &mut slot, next)?)?;
            toks.push(argmax(&lg));
        }
        solo.push(toks);
    }

    // Пакет: очередь бесед через `cap` слотов.
    let mut eng = BatchEngine::new(&mut m, &be, cap)?;
    let mut pending: VecDeque<usize> = (0..prompts.len()).collect();
    let mut owner: Vec<Option<usize>> = vec![None; eng.capacity()];
    let mut out: Vec<Vec<u32>> = vec![Vec::new(); prompts.len()];
    // Логиты последней позиции prefill'а — первый токен решает вызывающий.
    let mut first_logits: Vec<Vec<f32>> = vec![Vec::new(); prompts.len()];
    let mut next: Vec<(usize, u32)> = Vec::new();

    let mut fill = |eng: &mut BatchEngine<'_>,
                    be: &MetalBackend,
                    pending: &mut VecDeque<usize>,
                    owner: &mut Vec<Option<usize>>,
                    out: &mut Vec<Vec<u32>>,
                    next: &mut Vec<(usize, u32)>|
     -> Result<(), Box<dyn Error>> {
        while let Some(&p) = pending.front() {
            if eng.active() >= eng.capacity() {
                break;
            }
            let (slot, logits) = eng.submit(be, &ids[p])?;
            pending.pop_front();
            owner[slot] = Some(p);
            out[p].push(argmax(&logits));
            first_logits[p] = logits;
            next.push((slot, *out[p].last().unwrap()));
        }
        Ok(())
    };
    fill(&mut eng, &be, &mut pending, &mut owner, &mut out, &mut next)?;

    let mut steps = 0usize;
    while !next.is_empty() {
        let rows = eng.step(&be, &next)?;
        steps += 1;
        let mut follow: Vec<(usize, u32)> = Vec::new();
        for (slot, logits) in rows {
            let p = owner[slot].expect("у занятого слота есть беседа");
            out[p].push(argmax(&logits));
            if out[p].len() >= max_new + 1 {
                // Беседа набрала max_new+1 токен — место свободно.
                eng.finish(slot, &be)?;
                owner[slot] = None;
            } else {
                follow.push((slot, *out[p].last().unwrap()));
            }
        }
        next = follow;
        fill(&mut eng, &be, &mut pending, &mut owner, &mut out, &mut next)?;
    }

    let mut ok = true;
    for (p, prompt) in prompts.iter().enumerate() {
        let same = out[p] == solo[p];
        ok &= same;
        println!(
            "беседа {p} (\"{}…\"): {} ({} ток., контекст {}, шагов пакета всего {steps})",
            &prompt[..prompt.len().min(24)],
            if same { "совпала" } else { "РАЗОШЛАСЬ" },
            out[p].len(),
            ids[p].len() + out[p].len(),
        );
        if !same {
            eprintln!("  пакет: {:?}\n  эталон: {:?}", out[p], solo[p]);
        }
    }
    assert!(ok, "слоты с переиспользованием разошлись с одиночными прогонами");
    println!("OK: {cap} слотов обслужили {} бесед с переиспользованием мест", prompts.len());
    Ok(())
}
