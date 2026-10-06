//! Разбивка prefill по стадиям слоя Qwen3 (Ф8.2b): синхронизация после
//! каждой стадии (только диагностика), T токенов, суммы по всем слоям.
#![cfg(feature = "metal")]
use rust_inference::backend::Backend;
use rust_inference::metal::MetalBackend;
use rust_inference::qwen3::{Qwen3Config, Qwen3Model};
use rust_inference::tokenizer::Tokenizer;
use rust_inference::weights_map::WeightsMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("model")?;
    let t: usize = std::env::args().nth(2).and_then(|v| v.parse().ok()).unwrap_or(128);
    let file = std::fs::File::open(&path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let gf = rust_inference::gguf::parse(&mmap)?;
    let wm = WeightsMap::from_gguf(&gf)?;
    let tok = Tokenizer::from_gguf(&gf)?;
    let be = MetalBackend::new()?;
    let mut m = Qwen3Model::load(&be, &wm, Qwen3Config::from_gguf(&gf, tok.vocab_size())?)?;
    let ids: Vec<u32> = (0..t as u32).map(|i| 100 + i * 7).collect();
    let _ = m.prefill_last(&be, &ids)?;
    m.reset_cache(&be)?;
    let mb = |be: &MetalBackend| be.ctx.device.current_allocated_size() as f64 / 1048576.0;
    // (1) CPU-кодирование prefill: внешний батч, замер ДО его закрытия
    // (GPU до коммита не стартует) + полное время с исполнением.
    let a0 = mb(&be);
    be.begin_batch()?;
    let t0 = std::time::Instant::now();
    let l = m.prefill_last(&be, &ids)?;
    let enc = t0.elapsed().as_secs_f64();
    let a1 = mb(&be);
    be.end_batch()?;
    be.to_cpu_f32(&l)?;
    let all = t0.elapsed().as_secs_f64();
    eprintln!(
        "prefill_last {t}: CPU-кодирование {:.1} мс, всего {:.1} мс; память устройства {a0:.0} → {a1:.0} МБ за батч",
        enc * 1e3,
        all * 1e3
    );
    m.reset_cache(&be)?;
    // SAFETY: только диагностика — переменная читается внутри forward.
    unsafe { std::env::set_var("LLM_BENCH", "1") };
    let t0 = std::time::Instant::now();
    let l = m.prefill_last(&be, &ids)?;
    be.to_cpu_f32(&l)?;
    eprintln!("prefill_last {t} (PROF): {:.1} мс", t0.elapsed().as_secs_f64() * 1e3);
    // FFN одного слоя на реальных весах модели, в одном батче ×10 —
    // сверка с синтетическим prefill_probe (тот же путь ffn_forward).
    let d = m.cfg.dim;
    let x = be.from_cpu_f32(&vec![0.01f32; t * d], vec![1, t, d])?;
    let ffn = m
        .debug_ffn(0)
        .ok_or("prefill_breakdown: модель без плотного FFN (MoE)")?;
    be.begin_batch()?;
    let _ = rust_inference::qwen3::ffn_forward(&be, &x, ffn)?;
    be.end_batch()?;
    let t1 = std::time::Instant::now();
    be.begin_batch()?;
    for _ in 0..10 {
        let _ = rust_inference::qwen3::ffn_forward(&be, &x, ffn)?;
    }
    be.end_batch()?;
    eprintln!("ffn слоя 0 (реальные веса): {:.2} мс", t1.elapsed().as_secs_f64() * 1e3 / 10.0);
    // Все слои по разу (как в prefill: каждый раз новые веса, холодный кэш).
    let t2 = std::time::Instant::now();
    be.begin_batch()?;
    for i in 0..m.cfg.n_layers {
        let _ = rust_inference::qwen3::ffn_forward(
            &be,
            &x,
            m.debug_ffn(i)
                .ok_or("prefill_breakdown: модель без плотного FFN (MoE)")?,
        )?;
    }
    be.end_batch()?;
    eprintln!(
        "ffn всех {} слоёв в одном батче: {:.2} мс/слой",
        m.cfg.n_layers,
        t2.elapsed().as_secs_f64() * 1e3 / m.cfg.n_layers as f64
    );
    // По слою с синхронизацией (как PROF в prefill).
    let t3 = std::time::Instant::now();
    for i in 0..m.cfg.n_layers {
        be.begin_batch()?;
        let _ = rust_inference::qwen3::ffn_forward(
            &be,
            &x,
            m.debug_ffn(i)
                .ok_or("prefill_breakdown: модель без плотного FFN (MoE)")?,
        )?;
        be.end_batch()?;
    }
    eprintln!(
        "ffn по слою с синхронизацией: {:.2} мс/слой",
        t3.elapsed().as_secs_f64() * 1e3 / m.cfg.n_layers as f64
    );
    Ok(())
}
