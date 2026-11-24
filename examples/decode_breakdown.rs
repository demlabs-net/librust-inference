//! Где время decode-шага (Ф8.2): wall-время шага целиком vs сумма
//! GPU-времени — через MTLCommandBuffer GPUStartTime/GPUEndTime нельзя
//! (батч = один CB), поэтому сравниваем: (а) шаг с логитами на хост,
//! (б) N шагов подряд в ОДНОМ внешнем батче (без синхронизаций между
//! шагами) — разница = стоимость синхронизации/хоста на шаг.
#![cfg(feature = "metal")]
use rust_inference::backend::Backend as _;
use rust_inference::inference::generate::CausalLm;
use rust_inference::metal::MetalBackend;
use rust_inference::qwen3::{Qwen3Config, Qwen3Model};
use rust_inference::tokenizer::Tokenizer;
use rust_inference::weights_map::WeightsMap;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("model")?;
    let file = std::fs::File::open(&path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let gf = rust_inference::gguf::parse(&mmap)?;
    let wm = WeightsMap::from_gguf(&gf)?;
    let tok = Tokenizer::from_gguf(&gf)?;
    let be = MetalBackend::new()?;
    let mut m = Qwen3Model::load(&be, &wm, Qwen3Config::from_gguf(&gf, tok.vocab_size())?)?;
    let n = 32;
    // Контекст для декода: DEC_CTX токенов (по умолчанию 4).
    let ctx: usize = std::env::var("DEC_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let prompt: Vec<u32> = (0..ctx as u32).map(|i| 100 + (i * 7) % 5000).collect();
    let tl = std::time::Instant::now();
    let _ = CausalLm::prefill_last(&mut m, &be, &prompt)?;
    eprintln!("префилл {ctx} ток.: {:.2} с", tl.elapsed().as_secs_f64());
    for _ in 0..4 {
        let l = CausalLm::decode(&mut m, &be, 5)?;
        be.to_cpu_f32(&l)?;
    }
    // (а) обычный decode: логиты на хост каждый шаг
    let t0 = Instant::now();
    for _ in 0..n {
        let l = CausalLm::decode(&mut m, &be, 7)?;
        be.to_cpu_f32(&l)?;
    }
    let a = t0.elapsed().as_secs_f64() / n as f64;
    // (б) без чтения логитов, шаги в одном внешнем батче
    let t0 = Instant::now();
    be.begin_batch()?;
    for _ in 0..n {
        let _ = CausalLm::decode(&mut m, &be, 7)?;
    }
    be.end_batch()?;
    let b = t0.elapsed().as_secs_f64() / n as f64;
    // (в) только CPU-сторона кодирования: begin → decode×n (кернелы пишутся
    // в энкодер, GPU ещё не запущен) → замер ДО end_batch.
    be.begin_batch()?;
    let t0 = Instant::now();
    for _ in 0..n {
        let _ = CausalLm::decode(&mut m, &be, 7)?;
    }
    let c = t0.elapsed().as_secs_f64() / n as f64;
    be.end_batch()?;
    println!("CPU-кодирование шага:  {:.2} мс", c * 1e3);
    println!(
        "шаг с логитами на хост: {:.2} мс ({:.1} tok/s)",
        a * 1e3,
        1.0 / a
    );
    println!(
        "шаг в общем батче:      {:.2} мс ({:.1} tok/s)",
        b * 1e3,
        1.0 / b
    );
    Ok(())
}
