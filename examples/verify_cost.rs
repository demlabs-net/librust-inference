//! Стоимость окна верификации спека (Ф3): forward T строк (якорь + γ
//! черновиков) против одного шага декода на реальной модели. Спек выгоден,
//! только если forward 1+γ строк стоит немногим больше декода (матмулы —
//! mv_q8_0_mb, внимание — attn_split_f16_q1; до них ≈3× декода при любом T).
//!
//! `verify_cost <model.gguf>` печатает для T = 1..8: мс на forward T строк
//! (prefill_extract с 5 extract-слоями, как DFlash, и без них) и отношение к
//! декоду. Контекст — VERIFY_CTX токенов (по умолчанию 256).
#![cfg(feature = "metal")]
use rust_inference::backend::Backend as _;
use rust_inference::metal::MetalBackend;
use rust_inference::qwen3::{Qwen3Config, Qwen3Model};
use rust_inference::tokenizer::Tokenizer;
use rust_inference::weights_map::WeightsMap;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: verify_cost <model.gguf>")?;
    let file = std::fs::File::open(&path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let gf = rust_inference::gguf::parse(&mmap)?;
    let wm = WeightsMap::from_gguf(&gf)?;
    let tok = Tokenizer::from_gguf(&gf)?;
    let be = MetalBackend::new()?;
    let cfg = Qwen3Config::from_gguf(&gf, tok.vocab_size())?;
    let n_layers = cfg.n_layers;
    let mut m = Qwen3Model::load(&be, &wm, cfg)?;
    let ctx: usize = std::env::var("VERIFY_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    let prompt: Vec<u32> = (0..ctx as u32).map(|i| 100 + (i * 7) % 5000).collect();
    let extract: Vec<usize> = (0..5).map(|i| 1 + i * (n_layers - 2) / 4).collect();
    let reps = 12;

    // Каждый замер: кэш = промпт, затем forward, затем откат кэша к промпту.
    let mut run = |t: usize, layers: &[usize]| -> Result<f64, Box<dyn std::error::Error>> {
        m.reset_cache(&be)?;
        let lg = m.prefill_last(&be, &prompt)?;
        be.to_cpu_f32(&lg)?;
        let ids: Vec<u32> = (0..t as u32).map(|i| 200 + i).collect();
        let mut best = f64::MAX;
        for _ in 0..reps {
            let t0 = Instant::now();
            let lg = if t == 1 && layers.is_empty() {
                m.decode(&be, ids[0])?
            } else {
                m.prefill_extract(&be, &ids, layers)?.0
            };
            // логиты на хост — как в петле (verify читает все строки)
            be.to_cpu_f32(&lg)?;
            best = best.min(t0.elapsed().as_secs_f64() * 1e3);
            for c in &mut m.caches {
                c.truncate(ctx);
            }
            m.sync_rope_to_cache(0);
        }
        Ok(best)
    };
    // VERIFY_BREAKDOWN=T: разложение forward'а 2 и T строк (LLM_BENCH-профиль
    // слоёв: attn/ffn с синхронизацией; LLM_PROF_ATTN=1 — шаги внимания).
    if let Some(t) = std::env::var("VERIFY_BREAKDOWN")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        for tt in [2usize, t] {
            m.reset_cache(&be)?;
            let lg = m.prefill_last(&be, &prompt)?;
            be.to_cpu_f32(&lg)?;
            let ids: Vec<u32> = (0..tt as u32).map(|i| 200 + i).collect();
            // SAFETY: тот же поток; переменная читается внутри forward
            unsafe { std::env::set_var("LLM_BENCH", "1") };
            let t0 = Instant::now();
            let lg = m.prefill_extract(&be, &ids, &[])?.0;
            be.to_cpu_f32(&lg)?;
            unsafe { std::env::remove_var("LLM_BENCH") };
            eprintln!(
                "T={tt}: forward с синхронизацией по слоям {:.1} мс",
                t0.elapsed().as_secs_f64() * 1e3
            );
        }
        return Ok(());
    }
    let dec = run(1, &[])?;
    eprintln!("контекст {ctx}, extract {extract:?}");
    eprintln!("декод (1 строка): {dec:.2} мс");
    for t in 1..=8 {
        let plain = run(t, &[])?;
        let ex = run(t, &extract)?;
        eprintln!(
            "T={t}: forward {plain:6.2} мс ({:.2}× декода) · с extract {ex:6.2} мс ({:.2}×)",
            plain / dec,
            ex / dec
        );
    }
    Ok(())
}
