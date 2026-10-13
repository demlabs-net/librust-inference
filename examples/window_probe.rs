//! Зонд батч-пути гибрида (Ф8.2): стоимость forward при m=1..16 на реальных
//! весах. Запуск: `cargo run --release --features metal --example window_probe
//! -- models/llm/Qwen3.8-27B-HauhauCS-IQ4_XS-MTP.gguf`.
//! Каждый размер прогоняется 5 раз после прогрева; между размерами кэш
//! сбрасывается (позиция не влияет на GDN/FFN-стоимость).

#![cfg(feature = "metal")]

use std::error::Error;
use std::time::Instant;

use rust_inference::backend::Backend as _;
use rust_inference::gguf;
use rust_inference::metal::MetalBackend;
use rust_inference::qwen35;
use rust_inference::tokenizer::Tokenizer;
use rust_inference::weights_map::WeightsMap;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).ok_or("usage: window_probe <model.gguf>")?;
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let gf = gguf::parse(&mmap)?;
    let wm = WeightsMap::from_gguf(&gf)?;
    let tok = Tokenizer::from_gguf(&gf)?;
    let backend = MetalBackend::new()?;
    let mut model = qwen35::load_gguf(&backend, &gf, &wm, tok.vocab_size())?;
    let ids: Vec<u32> = (0..16u32).map(|i| 100 + i).collect();
    // Прогрев кернелов всех путей.
    model.reset_cache(&backend)?;
    model.prefill(&backend, &ids)?;
    for m in [1usize, 2, 4, 8, 16] {
        let mut best = f64::MAX;
        for _ in 0..5 {
            model.reset_cache(&backend)?;
            let t0 = Instant::now();
            model.prefill(&backend, &ids[..m])?;
            let _ = backend.to_cpu_f32(&backend.zeros(vec![1])?)?; // sync
            best = best.min(t0.elapsed().as_secs_f64());
        }
        eprintln!("m={m:2}: {:.1} мс ({:.1} мс/строку)", best * 1e3, best * 1e3 / m as f64);
    }
    Ok(())
}
