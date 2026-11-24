//! Оракул: одиночный `generate()` с NGramDrafter и γ=1 обязан дать ТЕ ЖЕ
//! токены, что чистый AR (жадная приёмка лосслессна). Если он совпадает, а
//! мой пакетный зонд нет — ошибка в конвенции выдачи драйвера, а не в
//! `decode_spec`.
//!
//! `cargo run --release --features metal --example spec_oracle_probe -- <model.gguf> [GAMMA]`

#![cfg(feature = "metal")]

use std::error::Error;
use std::sync::{Arc, Mutex};

use rust_inference::backend::Backend as _;
use rust_inference::gguf;
use rust_inference::inference::generate::{GenerateParams, SpecStats, generate};
use rust_inference::metal::MetalBackend;
use rust_inference::qwen35;
use rust_inference::spec::Drafter;
use rust_inference::tokenizer::Tokenizer;
use rust_inference::weights_map::WeightsMap;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .ok_or("usage: spec_oracle_probe <model.gguf> [GAMMA]")?;
    let gamma: u32 = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(1);
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let gf = gguf::parse(&mmap)?;
    let wm = WeightsMap::from_gguf(&gf)?;
    let tok = Tokenizer::from_gguf(&gf)?;
    let be = MetalBackend::new()?;
    let vocab = tok.vocab_size();
    let mut m = qwen35::load_gguf(&be, &gf, &wm, vocab)?;

    let prompt = tok.encode(
        "Repeat exactly this sentence: the quick brown fox jumps over the lazy dog. \
         Again: the quick brown fox jumps over the",
    );
    let n = 12usize;
    let base = |max_tokens: usize| GenerateParams {
        max_tokens,
        temperature: 0.0,
        ..Default::default()
    };

    // 1) Чистый AR.
    m.reset_cache(&be)?;
    let ar = generate(&mut m, &be, &tok, &prompt, &base(n), |_| true)?;

    // 2) Спекуляция: N-грамм-драфтер, γ задан.
    let drafter: Arc<Mutex<dyn Drafter>> =
        Arc::new(Mutex::new(rust_inference::spec::NGramDrafter::new(4)));
    let mut p = base(n);
    p.drafter = Some(drafter);
    p.spec_gamma = gamma as usize;
    m.reset_cache(&be)?;
    let mut stats = SpecStats::default();
    let spec = generate(&mut m, &be, &tok, &prompt, &p, |ev| {
        if let rust_inference::inference::generate::GenerateEvent::SpecStats(s) = ev {
            stats = s;
        }
        true
    })?;

    println!("AR:   {ar:?}");
    println!("спек: {spec:?}");
    println!(
        "черновиков {}, принято {}, отклонено {}",
        stats.drafts, stats.accepted, stats.rejected
    );
    if ar == spec {
        println!("OK: одиночный спек == AR (движок верен, искать в драйвере)");
    } else {
        let i = ar.iter().zip(spec.iter()).position(|(a, b)| a != b);
        println!(
            "РАСХОЖДЕНИЕ на токене {i:?}: AR {:?}, спек {:?}",
            i.map(|i| ar[i]),
            i.map(|i| spec[i])
        );
    }
    Ok(())
}
