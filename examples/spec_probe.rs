//! Диагностика окна верификации для ОДНОЙ беседы (локализация расхождения
//! batch-spec). Никакого пакета: один слот, AR-прогон против прогона с окнами
//! `decode_spec_slot`, пошаговый дамп.
//!
//! `cargo run --release --features metal --example spec_probe -- <model.gguf>`

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
    let path = args.get(1).ok_or("usage: spec_probe <model.gguf>")?;
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let gf = gguf::parse(&mmap)?;
    let wm = WeightsMap::from_gguf(&gf)?;
    let tok = Tokenizer::from_gguf(&gf)?;
    let be = MetalBackend::new()?;
    let vocab = tok.vocab_size();
    let mut m = qwen35::load_gguf(&be, &gf, &wm, vocab)?;

    // Промпт с повтором: lookup обязан находить вербатимное продолжение.
    let prompt = tok.encode(
        "Repeat exactly: the quick brown fox jumps over the lazy dog. \
         the quick brown fox jumps over the lazy dog. the quick brown fox",
    );
    let n = 10usize;
    let gamma = 8usize;

    // ── AR-эталон ────────────────────────────────────────────────────────
    let mut slot = m.new_slot(&be)?;
    let mut lg = be.to_cpu_f32(&m.prefill_slot_last(&be, &mut slot, &prompt)?)?;
    let mut ar: Vec<u32> = vec![argmax(&lg)];
    for _ in 0..n {
        let next = *ar.last().unwrap();
        lg = be.to_cpu_f32(&m.decode_slot(&be, &mut slot, next)?)?;
        ar.push(argmax(&lg));
    }
    let ar_pos = slot.pos();

    // ── Прогон с окнами (та же логика, что в generate_batch) ─────────────
    let mut eng = BatchEngine::new(&mut m, &be, 1)?;
    let (s, lg0) = eng.submit(&be, &prompt)?;
    assert_eq!(s, 0);
    let mut history = prompt.clone();
    let mut spec: Vec<u32> = vec![argmax(&lg0)];
    history.push(*spec.last().unwrap());
    let mut pending: Option<Vec<f32>> = None;
    let mut steps = 0usize;
    let mut accepted_total = 0usize;

    while spec.len() <= n {
        steps += 1;
        // 1) Есть готовые логиты (полная приёмка прошлого окна) — сэмплим.
        if let Some(row) = pending.take() {
            let t = argmax(&row);
            println!("шаг {steps}: из логитов окна -> {t}");
            spec.push(t);
            history.push(t);
            continue;
        }
        // 2) Якорь — следующий токен; пробуем окно.
        let anchor = *spec.last().unwrap();
        let (drafts, probs) = rust_inference::spec::ngram_lookup_min(&history, gamma, 4, 4);
        if drafts.len() >= 2 {
            let o = eng.decode_spec_slot(&be, 0, anchor, &drafts, &probs, true, &history)?;
            let n_ok = o.n_draft_accepted + 1;
            println!(
                "шаг {steps}: черновиков {}, принято {}, accepted={:?} (якорь {anchor})",
                drafts.len(),
                o.n_draft_accepted,
                o.accepted
            );
            accepted_total += o.n_draft_accepted;
            for &t in &o.accepted[1..n_ok] {
                if spec.len() <= n {
                    spec.push(t);
                    history.push(t);
                }
            }
            if n_ok < o.accepted.len() {
                // пересэмпл не в KV — якорь следующего шага
                let t = *o.accepted.last().unwrap();
                if spec.len() <= n {
                    spec.push(t);
                    history.push(t);
                }
            } else {
                pending = Some(be.to_cpu_f32(&o.logits)?);
            }
        } else {
            // 3) Черновиков нет — обычный шаг по слоту.
            let t = anchor;
            let lg = eng.decode_step_slot(&be, 0, t)?;
            let row = be.to_cpu_f32(&lg)?;
            let next = argmax(&row);
            println!("шаг {steps}: AR-шаг по слоту");
            spec.push(next);
            history.push(next);
        }
    }
    spec.truncate(n + 1);
    ar.truncate(n + 1);
    let same = spec == ar;
    println!("\nAR      ({} шагов, pos {}): {:?}", n, ar_pos, ar);
    println!(
        "с окнами ({} шагов, принято {}): {:?}",
        steps, accepted_total, spec
    );
    println!(
        "первое расхождение: {:?}",
        spec.iter()
            .zip(&ar)
            .position(|(a, b)| a != b)
            .map(|i| (i, spec[i], ar[i]))
    );
    assert!(same, "окна разошлись с AR");
    println!("OK: окна дают те же токены, что AR");
    Ok(())
}
