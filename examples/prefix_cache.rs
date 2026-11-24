//! Приёмка КЭША ПРЕФИКСА между ходами беседы: второй ход префиллит только
//! хвост, а токены обязаны совпасть с полным префиллом всей переписки.
//!
//! `cargo run --release --features metal --example prefix_cache -- <model.gguf>`

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
    let path = args.get(1).ok_or("usage: prefix_cache <model.gguf>")?;
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let gf = gguf::parse(&mmap)?;
    let wm = WeightsMap::from_gguf(&gf)?;
    let tok = Tokenizer::from_gguf(&gf)?;
    let be = MetalBackend::new()?;
    let vocab = tok.vocab_size();
    let mut m = qwen35::load_gguf(&be, &gf, &wm, vocab)?;

    let n = 8usize;
    let turn1 = tok.encode("Explain in one short sentence why the sky appears blue.");
    let follow = tok.encode(" Now say it even shorter.");

    // Ход 1: полный префилл + n жадных токенов.
    let mut eng = BatchEngine::new(&mut m, &be, 2)?;
    let slot_a = eng.begin_prefill(&be, &turn1)?;
    let mut lg = eng
        .prefill_chunk(&be, slot_a, &turn1, true)?
        .expect("логиты");
    let mut gen1 = Vec::new();
    let mut next = argmax(&lg);
    gen1.push(next);
    for _ in 1..n {
        lg = be.to_cpu_f32(&eng.decode_step_slot(&be, slot_a, next)?)?;
        next = argmax(&lg);
        gen1.push(next);
    }
    let mut history = turn1.clone();
    history.extend_from_slice(&gen1);

    // Ход 2: та же беседа продолжается — полный промпт = ход 1 + ответ + новая
    // реплика. Кэш слота должен покрыть ровно `history`.
    let mut turn2 = history.clone();
    turn2.extend_from_slice(&follow);

    // Эталон: ПОЛНЫЙ префилл всей переписки во втором слоте.
    let slot_b = eng.begin_prefill(&be, &turn2)?;
    let mut lg2 = eng
        .prefill_chunk(&be, slot_b, &turn2, true)?
        .expect("логиты");
    let mut ref_tokens = Vec::new();
    let mut nx = argmax(&lg2);
    ref_tokens.push(nx);
    for _ in 1..n {
        lg2 = be.to_cpu_f32(&eng.decode_step_slot(&be, slot_b, nx)?)?;
        nx = argmax(&lg2);
        ref_tokens.push(nx);
    }
    eng.finish(slot_b, &be)?;

    // Кэш: ход 1 уходит в кэш, ход 2 подхватывает префикс.
    eng.finish_keep(slot_a, &history)?;
    let (slot, common) = eng
        .best_cached_slot(&turn2)
        .ok_or("кэш префикса не найден")?;
    // В KV хода 1 лежит `history` без последнего сэмплированного токена.
    assert_eq!(
        common,
        history.len() - 1,
        "кэш обязан покрыть историю хода 1 (кроме не поданного последнего токена)"
    );
    let (common, lg3) = eng.resume_prefix(&be, slot, &turn2)?;
    assert_eq!(common, history.len() - 1);
    let mut lg3 = lg3.expect("логиты хвоста");
    let mut got = Vec::new();
    let mut n3 = argmax(&lg3);
    got.push(n3);
    for _ in 1..n {
        lg3 = be.to_cpu_f32(&eng.decode_step_slot(&be, slot, n3)?)?;
        n3 = argmax(&lg3);
        got.push(n3);
    }

    println!(
        "ход 2: промпт {} токенов, из кэша {} (префилл {} вместо {})",
        turn2.len(),
        common,
        turn2.len() - common,
        turn2.len()
    );
    println!("эталон: {ref_tokens:?}");
    println!("кэш:    {got:?}");
    assert_eq!(got, ref_tokens, "кэш префикса разошёлся с полным префиллом");
    println!(
        "OK: кэш префикса даёт те же токены, экономя {} токенов префилла ({:.0}%)",
        common,
        100.0 * common as f64 / turn2.len() as f64
    );
    Ok(())
}
