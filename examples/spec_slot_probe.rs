//! Зонд спекуляции на ОДНОЙ беседе: локализует расхождение окна с AR до
//! конвенции `SpecOutcome` (без пакета вовсе).
//!
//! Печатает по шагам: якорь, что вернуло окно (`accepted`,
//! `n_draft_accepted`), что выдано, `pos` слота — и первое расхождение с
//! чистым AR-прогоном того же промпта.
//!
//! `cargo run --release --features metal --example spec_slot_probe -- <model.gguf> [GAMMA]`

#![cfg(feature = "metal")]

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
    let path = args.get(1).ok_or("usage: spec_slot_probe <model.gguf> [GAMMA]")?;
    let gamma: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(8);
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let gf = gguf::parse(&mmap)?;
    let wm = WeightsMap::from_gguf(&gf)?;
    let tok = Tokenizer::from_gguf(&gf)?;
    let be = MetalBackend::new()?;
    let vocab = tok.vocab_size();
    let mut m = qwen35::load_gguf(&be, &gf, &wm, vocab)?;

    // Промпт с ЯВНЫМ повтором: продолжение есть в истории -> lookup срабатывает.
    let prompt = tok.encode(
        "Repeat exactly this sentence: the quick brown fox jumps over the lazy dog. \
         Again: the quick brown fox jumps over the",
    );
    let n = 12usize;

    // 1) Чистый AR: эталон токенов и позиций.
    let mut eng = BatchEngine::new(&mut m, &be, 1)?;
    let (slot, lg) = eng.submit(&be, &prompt)?;
    let mut next = argmax(&lg);
    let mut ar = vec![next];
    for _ in 0..n {
        let rows = eng.step(&be, &[(slot, next)])?;
        next = argmax(&rows[0].1);
        ar.push(next);
    }
    let ar_pos = eng.pos(slot).unwrap();
    println!("AR: {} токенов, pos={ar_pos}", ar.len());
    drop(eng);

    // 2) Окно: та же беседа, но продолжение берём через decode_spec_slot.
    let mut eng = BatchEngine::new(&mut m, &be, 1)?;
    let (slot, lg) = eng.submit(&be, &prompt)?;
    let mut next = argmax(&lg);
    let mut hist: Vec<u32> = prompt.clone();
    let mut out = vec![next];
    hist.push(next);
    let mut step = 0usize;
    let mut diverged = None;
    while out.len() < n + 1 {
        let (drafts, probs) = rust_inference::spec::ngram_lookup_min(&hist, 1, 4, 4);
        if drafts.is_empty() {
            let rows = eng.step(&be, &[(slot, next)])?;
            next = argmax(&rows[0].1);
            out.push(next);
            hist.push(next);
            step += 1;
            continue;
        }
        let o = eng.decode_spec_slot(&be, slot, next, &drafts, &probs, true, &hist)?;
        let n_ok = o.n_draft_accepted + 1;
        println!(
            "шаг {step}: якорь {next}, 1 черновик {drafts:?}, accepted={:?}, n_draft={}, n_ok={n_ok}, pos={:?}",
            o.accepted,
            o.n_draft_accepted,
            eng.pos(slot)
        );
        // Конвенция одиночного пути: наружу идут accepted[1..n_ok].
        for &t in &o.accepted[1..n_ok] {
            out.push(t);
            hist.push(t);
        }
        if n_ok < o.accepted.len() {
            // Пересэмпл НЕ в KV — якорь следующего шага.
            next = *o.accepted.last().unwrap();
        } else {
            // Полная приёмка: логиты продолжения на руках.
            let row = be.to_cpu_f32(&o.logits)?;
            next = argmax(&row[..vocab]);
            out.push(next);
            hist.push(next);
        }
        for (i, (a, b)) in out.iter().zip(ar.iter()).enumerate() {
            if a != b && diverged.is_none() {
                diverged = Some((i, *a, *b));
            }
        }
        step += 1;
        if step > 40 {
            break;
        }
    }
    let sx_pos = eng.pos(slot).unwrap();
    println!("окно: {} токенов, pos={sx_pos}", out.len());
    println!("AR:   {ar:?}");
    println!("окно: {out:?}");
    match diverged {
        None => {
            let same = out.len() >= ar.len() && out[..ar.len()] == ar[..];
            println!(
                "{}",
                if same {
                    "OK: окно = AR"
                } else {
                    "хвост отличается (длина)"
                }
            );
        }
        Some((i, a, b)) => println!("РАСХОЖДЕНИЕ на токене {i}: окно {a}, AR {b}"),
    }
    Ok(())
}
