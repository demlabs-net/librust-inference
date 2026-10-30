//! Приёмка пакетной ГЕНЕРАЦИИ ([`rust_inference::batch::generate_batch`]):
//! очередь из 6 бесед через 2 слота, сэмплинг/стоп/UTF-8 — на каждую свою.
//! Жадные беседы обязаны дать те же токены, что одиночный путь
//! `inference::generate::generate` на той же модели.
//!
//! `cargo run --release --features metal --example batch_generate -- <model.gguf> [CAP]`

#![cfg(feature = "metal")]

use std::error::Error;

use rust_inference::batch::{BatchEngine, BatchRequest, generate_batch};
use rust_inference::gguf;
use rust_inference::inference::generate::{FinishReason, GenerateEvent, GenerateParams, generate};
use rust_inference::metal::MetalBackend;
use rust_inference::qwen35;
use rust_inference::tokenizer::Tokenizer;
use rust_inference::weights_map::WeightsMap;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .ok_or("usage: batch_generate <model.gguf> [CAP]")?;
    let cap: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(2);
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
    let max_tokens = 8usize;
    let params = || GenerateParams {
        max_tokens,
        temperature: 0.0,
        ..Default::default()
    };

    // Эталон: одиночный путь на той же модели (сброс кэша как в сервисе).
    let mut solo: Vec<Vec<u32>> = Vec::new();
    for p in prompts {
        let ids = tok.encode(p);
        m.reset_cache(&be)?;
        let mut done = None;
        let toks = generate(&mut m, &be, &tok, &ids, &params(), |ev| {
            if let GenerateEvent::Done(r) = ev {
                done = Some(r);
            }
            true
        })?;
        assert_eq!(
            done,
            Some(FinishReason::MaxTokens),
            "одиночный путь должен упереться в max_tokens"
        );
        solo.push(toks);
    }

    // Пакет: 6 бесед через `cap` слота (очередь + переиспользование).
    let reqs: Vec<BatchRequest> = prompts
        .iter()
        .map(|p| BatchRequest {
            ids: tok.encode(p),
            params: params(),
        })
        .collect();
    let mut deltas = vec![0usize; reqs.len()];
    let mut dones: Vec<Option<FinishReason>> = vec![None; reqs.len()];
    let out = {
        let mut eng = BatchEngine::new(&mut m, &be, cap)?;
        generate_batch(&mut eng, &be, &tok, &reqs, &mut |q, ev| {
            match ev {
                GenerateEvent::Delta(_) => deltas[q] += 1,
                GenerateEvent::Done(r) => dones[q] = Some(r),
                _ => {}
            }
            true
        })?
    };

    let mut ok = true;
    for (i, p) in prompts.iter().enumerate() {
        let same = out[i] == solo[i];
        ok &= same;
        println!(
            "беседа {i} (\"{}…\", промпт {}): {} ток., дельт {}, финал {:?}{}",
            &p[..p.len().min(24)],
            reqs[i].ids.len(),
            out[i].len(),
            deltas[i],
            dones[i],
            if same {
                ""
            } else {
                " — РАСХОЖДЕНИЕ"
            }
        );
        if !same {
            eprintln!("  пакет: {:?}\n  эталон: {:?}", out[i], solo[i]);
        }
    }
    assert!(ok, "пакетная генерация разошлась с одиночной");

    // Сэмплинг с температурой: тот же сид, тот же каскад. Токены СВЕРЯЕМ, но
    // не требуем равенства: m=1 и m=B идут разными кернелами (разница ~1e-5),
    // и выбор из почти равных кандидатов может легально отличаться.
    let samp = || GenerateParams {
        max_tokens: 6,
        temperature: 0.8,
        top_p: 0.95,
        top_k: 40,
        seed: Some(1234),
        ..Default::default()
    };
    let two: Vec<&str> = vec![prompts[0], prompts[3]];
    let mut t_solo = Vec::new();
    for p in &two {
        m.reset_cache(&be)?;
        t_solo.push(generate(
            &mut m,
            &be,
            &tok,
            &tok.encode(p),
            &samp(),
            |_| true,
        )?);
    }
    let t_reqs: Vec<BatchRequest> = two
        .iter()
        .map(|p| BatchRequest {
            ids: tok.encode(p),
            params: samp(),
        })
        .collect();
    let t_out = {
        let mut eng = BatchEngine::new(&mut m, &be, 2)?;
        generate_batch(&mut eng, &be, &tok, &t_reqs, &mut |_, _| true)?
    };
    for (i, toks) in t_out.iter().enumerate() {
        assert_eq!(toks.len(), 6, "беседа {i}: длина при температуре");
        println!(
            "сэмплинг {i}: {} ток., с одиночным {}",
            toks.len(),
            if *toks == t_solo[i] {
                "совпал"
            } else {
                "разошёлся в выборе"
            }
        );
    }
    let all_done = dones.iter().all(|d| *d == Some(FinishReason::MaxTokens));
    assert!(all_done, "не все беседы закрылись Done(MaxTokens)");

    // Сэмплирующие беседы на ЦИТАТАХ: пакетный спек обязан совпасть с
    // ОДИНОЧНЫМ спеком (тот же лосслесс-каскад и сид), а не с чистым AR:
    // при температуре окно меняет поток, оставаясь лосслессным.
    let qprompts = [
        "Repeat exactly: the quick brown fox jumps over the lazy dog. Again: the quick brown fox jumps over the",
        "Copy this: one two three four five one two three four five one two three four",
    ];
    let sparam = || GenerateParams {
        max_tokens: 10,
        temperature: 0.8,
        top_p: 0.95,
        top_k: 40,
        seed: Some(7),
        spec_gamma: 8,
        ..Default::default()
    };
    let mut qsolo = Vec::new();
    for p in qprompts {
        m.reset_cache(&be)?;
        let mut pr = sparam();
        pr.drafter = Some(std::sync::Arc::new(std::sync::Mutex::new(
            rust_inference::spec::NGramDrafter::new(4),
        )));
        qsolo.push(generate(&mut m, &be, &tok, &tok.encode(p), &pr, |_| true)?);
    }
    let qreqs: Vec<BatchRequest> = qprompts
        .iter()
        .map(|p| BatchRequest {
            ids: tok.encode(p),
            params: GenerateParams {
                drafter: None,
                ..sparam()
            },
        })
        .collect();
    let qout = {
        let mut eng = BatchEngine::new(&mut m, &be, 2)?;
        generate_batch(&mut eng, &be, &tok, &qreqs, &mut |_, _| true)?
    };
    for (i, got) in qout.iter().enumerate() {
        println!(
            "сэмплинг-цитата {i}: {} ток., с одиночным спеком {}",
            got.len(),
            if *got == qsolo[i] {
                "совпал"
            } else {
                "разошёлся"
            }
        );
        assert_eq!(
            *got, qsolo[i],
            "сэмплирующий спек в пакете должен совпадать с одиночным спеком"
        );
    }
    println!(
        "OK: {cap} слота обслужили {} бесед, токены = одиночным прогонам",
        reqs.len()
    );
    Ok(())
}
