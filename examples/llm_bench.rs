//! Бенч pp/tg (Ф8.2) — та же методика, что `llama-bench -p 128 -n 64`:
//! прогрев (компиляция пайплайнов, аллокации пула) вне замера; pp — prefill
//! 128 токенов с пустого контекста; tg — 64 decode-шага; по `-r` повторов,
//! среднее. Архитектура — по GGUF (Qwen3 / гибрид Qwen3.5/3.8).
//!
//! `cargo run --release --features metal --example llm_bench -- <model.gguf> [pp] [tg] [reps]`

#![cfg(feature = "metal")]

use std::error::Error;
use std::time::Instant;

use rust_inference::backend::Backend as _;
use rust_inference::gguf;
use rust_inference::inference::generate::CausalLm;
use rust_inference::metal::MetalBackend;
use rust_inference::qwen3::{Qwen3Config, Qwen3Model};
use rust_inference::qwen35::{Qwen35Config, Qwen35Model};
use rust_inference::tokenizer::Tokenizer;
use rust_inference::weights_map::WeightsMap;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .ok_or("usage: llm_bench <model.gguf> [pp] [tg] [reps]")?;
    let pp: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(128);
    let tg: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(64);
    let reps: usize = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(3);
    // ctx — длина контекста ПЕРЕД замером tg (как llama-bench -p N -n M:
    // декод с длинным KV). 0 — без контекста.
    let ctx: usize = std::env::var("LLM_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let file = std::fs::File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let gf = gguf::parse(&mmap)?;
    let wm = WeightsMap::from_gguf(&gf)?;
    let tok = Tokenizer::from_gguf(&gf)?;
    let be = MetalBackend::new()?;
    let arch = gf.architecture().unwrap_or("qwen3").to_string();
    let mut model: Box<dyn CausalLm> = if Qwen35Config::is_hybrid_arch(&arch) {
        Box::new(Qwen35Model::load(
            &be,
            &wm,
            Qwen35Config::from_gguf(&gf, tok.vocab_size())?,
        )?)
    } else {
        Box::new(Qwen3Model::load(
            &be,
            &wm,
            Qwen3Config::from_gguf(&gf, tok.vocab_size())?,
        )?)
    };
    let vocab = model.vocab_size() as u32;
    // Детерминированные «случайные» токены (как llama-bench: не текст).
    let mut s = 0x9E3779B97F4A7C15u64;
    let mut next_id = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s % (vocab as u64).min(100_000)) as u32
    };
    let prompt: Vec<u32> = (0..pp).map(|_| next_id()).collect();

    // Прогрев: все кернелы prefill/decode путей компилируются здесь.
    model.reset_cache(&be)?;
    let lg = model.prefill(&be, &prompt[..pp.min(16)])?;
    let _ = be.to_cpu_f32(&lg)?;
    for _ in 0..4 {
        let lg = model.decode(&be, next_id())?;
        let _ = be.to_cpu_f32(&lg)?;
    }

    let mut pp_ts = Vec::new();
    let mut tg_ts = Vec::new();
    for _ in 0..reps {
        model.reset_cache(&be)?;
        let t0 = Instant::now();
        // Как генерация: логиты только последней позиции (llama-bench тоже
        // выводит лишь их).
        let lg = model.prefill_last(&be, &prompt)?;
        let _ = be.to_cpu_f32(&lg)?;
        pp_ts.push(pp as f64 / t0.elapsed().as_secs_f64());
        // Контекст для декода (llama-bench -p N -n M): префилл ctx токенов.
        if ctx > 0 {
            let long: Vec<u32> = (0..ctx).map(|_| next_id()).collect();
            let lg = model.prefill_last(&be, &long)?;
            let _ = be.to_cpu_f32(&lg)?;
        }

        let t0 = Instant::now();
        let mut id = next_id();
        for _ in 0..tg {
            let lg = model.decode(&be, id)?;
            // Как в реальной генерации: логиты на хост каждый шаг.
            let v = be.to_cpu_f32(&lg)?;
            id = (v.len() as u32 + id) % vocab;
        }
        tg_ts.push(tg as f64 / t0.elapsed().as_secs_f64());
    }
    let avg = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let name = std::path::Path::new(path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    println!("{name} pp{pp} {:.1} tok/s", avg(&pp_ts));
    if ctx > 0 {
        println!("{name} tg{tg}@{ctx} {:.1} tok/s", avg(&tg_ts));
    } else {
        println!("{name} tg{tg} {:.1} tok/s", avg(&tg_ts));
    }
    Ok(())
}
