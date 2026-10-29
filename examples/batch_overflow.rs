//! Приёмка переполнения окна в ПАКЕТЕ: беседа, упёршаяся в контекст,
//! закрывается САМА, а пакет продолжает идти — остальные беседы добирают
//! свои токены до конца, ошибки нет.
//!
//! `INFER_LLM_CTX` задаёт окно (например 40 токенов), промпты разной длины.
//! `cargo run --release --features metal --example batch_overflow -- <model.gguf>`

#![cfg(feature = "metal")]

use std::error::Error;

use rust_inference::backend::Backend as _;
use rust_inference::batch::{BatchEngine, BatchRequest, generate_batch};
use rust_inference::gguf;
use rust_inference::inference::generate::{FinishReason, GenerateEvent, GenerateParams};
use rust_inference::metal::MetalBackend;
use rust_inference::qwen35;
use rust_inference::tokenizer::Tokenizer;
use rust_inference::weights_map::WeightsMap;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).ok_or("usage: batch_overflow <model.gguf>")?;
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let gf = gguf::parse(&mmap)?;
    let wm = WeightsMap::from_gguf(&gf)?;
    let tok = Tokenizer::from_gguf(&gf)?;
    let be = MetalBackend::new()?;
    let vocab = tok.vocab_size();
    let mut m = qwen35::load_gguf(&be, &gf, &wm, vocab)?;
    let max_seq = m.cfg.max_seq;
    println!("окно контекста: {max_seq} токенов (INFER_LLM_CTX)");

    // Промпты РАЗНОЙ длины: короткий влезает с запасом, длинный упирается.
    let base = tok.encode("The history of printing begins with movable type in Asia.");
    let mk = |n: usize| -> Vec<u32> {
        let mut v = Vec::with_capacity(n);
        while v.len() < n {
            v.extend_from_slice(&base);
        }
        v.truncate(n);
        v
    };
    let lens = [8usize, max_seq / 2, max_seq - 6];
    let reqs: Vec<BatchRequest> = lens
        .iter()
        .map(|&n| BatchRequest {
            ids: mk(n),
            params: GenerateParams {
                max_tokens: 20,
                temperature: 0.0,
                ..Default::default()
            },
        })
        .collect();

    let mut dones: Vec<Option<FinishReason>> = vec![None; reqs.len()];
    let out = {
        let mut eng = BatchEngine::new(&mut m, &be, reqs.len())?;
        generate_batch(&mut eng, &be, &tok, &reqs, &mut |q, ev| {
            if let GenerateEvent::Done(r) = ev {
                dones[q] = Some(r);
            }
            true
        })?
    };

    let mut ok = true;
    for (i, n) in lens.iter().enumerate() {
        let got = out[i].len();
        let finished = dones[i].is_some();
        // Длинная беседа обязана упереться в окно, короткие — дойти до max_tokens.
        let expect_full = *n + 20 <= max_seq;
        let fits = if expect_full { got == 20 } else { got < 20 };
        ok &= finished && fits;
        println!(
            "беседа {i}: промпт {n} ток., выдано {got} из 20, финал {:?} — {}",
            dones[i],
            if finished && fits {
                "как ожидалось"
            } else {
                "НЕ ТАК"
            }
        );
    }
    assert!(ok, "переполнение окна обработано неверно");
    println!("OK: упёршаяся в окно беседа закрыта, остальные дошли до max_tokens, ошибки нет");
    Ok(())
}
