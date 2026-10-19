//! Жадная генерация на гибриде Qwen3.5/3.8 (Ф9.4) — сверка с llama.cpp.
//!
//! Запуск: `cargo run --release --features metal --example qwen35_chat -- \
//!   models/llm/Qwen3.5-4B-Q8_0.gguf "The capital of France is" [N]`
//!
//! Промпт подаётся как есть (без чат-шаблона) — тот же вход, что у
//! `llama-simple`, чтобы жадные тексты сравнивались один в один.
//! `QWEN35_DEBUG=1` печатает rms выхода каждого слоя на prefill.

#![cfg(feature = "metal")]

use std::error::Error;
use std::io::Write;
use std::time::Instant;

use rust_inference::backend::Backend as _;
use rust_inference::gguf;
use rust_inference::metal::MetalBackend;
use rust_inference::qwen35;
use rust_inference::tokenizer::{IncrementalDecoder, Tokenizer};
use rust_inference::weights_map::WeightsMap;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .ok_or("usage: qwen35_chat <model.gguf> [prompt] [n]")?;
    let prompt = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "The capital of France is".into());
    let n: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(32);

    let file = std::fs::File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let gf = gguf::parse(&mmap)?;
    let wm = WeightsMap::from_gguf(&gf)?;
    let tok = Tokenizer::from_gguf(&gf)?;
    let backend = MetalBackend::new()?;
    let t0 = Instant::now();
    let mut model = qwen35::load_gguf(&backend, &gf, &wm, tok.vocab_size())?;
    let c = &model.cfg;
    eprintln!(
        "{}: dim={} layers={} (gdn {}), heads={}/{} hd={} n_rot={} ssm S={} Hk={} Hv={} — загрузка {:.1}s",
        c.arch,
        c.dim,
        c.n_layers,
        c.recurrent.iter().filter(|r| **r).count(),
        c.n_heads,
        c.n_kv_heads,
        c.head_dim,
        c.n_rot,
        c.ssm_state,
        c.ssm_k_heads,
        c.ssm_v_heads,
        t0.elapsed().as_secs_f32()
    );
    let ids = tok.encode(&prompt);
    eprintln!("ids={ids:?}");
    // QWEN35_SPEC=ngram|mtp|<dflash.gguf>: та же жадная генерация через
    // inference::generate со спек-декодингом — текст обязан совпасть с
    // обычным жадным (откат рекуррентного стейта при отказе черновика).
    if let Ok(spec) = std::env::var("QWEN35_SPEC") {
        use rust_inference::inference::generate::{GenerateEvent, GenerateParams, generate};
        let drafter: std::sync::Arc<std::sync::Mutex<dyn rust_inference::spec::Drafter>> = if spec
            == "ngram"
        {
            std::sync::Arc::new(std::sync::Mutex::new(
                rust_inference::spec::NGramDrafter::new(4),
            ))
        } else if spec == "mtp" {
            // Ф2.6: нативный MTP — nextn-блок из того же файла
            // (blk.{n_layers}.nextn.*); таблицы эмбеддингов/LM-хед общие
            // с таргетом (как в эталоне: nextn.embed_tokens/shared_head_head
            // в этих чекпойнтах нет).
            let (embed, output) = model.head_tables();
            std::sync::Arc::new(std::sync::Mutex::new(
                rust_inference::mtp::MtpDrafter::load(&backend, &wm, &model.cfg, embed, output)?,
            ))
        } else {
            let (embed, lm_head) = model.shared_tables(&backend, &wm)?;
            std::sync::Arc::new(std::sync::Mutex::new(
                rust_inference::dflash::load_drafter_gguf(
                    &backend,
                    std::path::Path::new(&spec),
                    embed,
                    lm_head,
                    model.cfg.vocab_size,
                )?,
            ))
        };
        let params = GenerateParams {
            max_tokens: n,
            temperature: 0.0,
            repeat_penalty: 1.0,
            drafter: Some(drafter),
            spec_gamma: std::env::var("QWEN35_GAMMA")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(4),
            ..Default::default()
        };
        // Прогрев тем же путём (пайплайны таргета и драфтера, подкачка
        // весов после загрузки драфтера): замер — по тёплому движку.
        {
            let warm = GenerateParams {
                max_tokens: 4,
                temperature: 0.0,
                drafter: params.drafter.clone(),
                spec_gamma: params.spec_gamma,
                ..Default::default()
            };
            generate(&mut model, &backend, &tok, &ids, &warm, |_| true)?;
            model.reset_cache(&backend)?;
            if let Some(d) = &params.drafter {
                d.lock().unwrap().reset();
            }
        }
        print!("{prompt}");
        let mut stats = None;
        let mut timings = None;
        let out = generate(&mut model, &backend, &tok, &ids, &params, |ev| {
            match ev {
                GenerateEvent::Delta(s) => {
                    print!("{s}");
                    let _ = std::io::stdout().flush();
                }
                GenerateEvent::SpecStats(st) => stats = Some(st),
                GenerateEvent::Timings(t) => timings = Some(t),
                _ => {}
            }
            true
        })?;
        println!();
        let t = timings.ok_or("generate не прислал Timings")?;
        eprintln!(
            "spec {spec}: префилл {} ток. за {:.2}s ({:.1} tok/s); декод {} ток. ({} шагов) за {:.2}s ({:.1} tok/s), {:?}",
            t.prompt_n,
            t.prompt_ms / 1e3,
            t.prompt_per_second(),
            out.len(),
            t.decode_steps(),
            t.predicted_ms / 1e3,
            t.predicted_per_second(),
            stats
        );
        return Ok(());
    }
    let vocab = model.cfg.vocab_size;
    let t1 = Instant::now();
    let mut logits = model.prefill_last(&backend, &ids)?;
    eprintln!(
        "prefill {} ток. за {:.2}s",
        ids.len(),
        t1.elapsed().as_secs_f32()
    );
    let mut dec = IncrementalDecoder::new();
    print!("{prompt}");
    let t2 = Instant::now();
    let mut produced = 0usize;
    for step in 0..n {
        let v = backend.to_cpu_f32(&logits)?;
        let last = &v[v.len() - vocab..];
        if step == 0 {
            let mut top: Vec<(usize, f32)> = last.iter().cloned().enumerate().collect();
            top.sort_by(|a, b| b.1.total_cmp(&a.1));
            for (id, s) in top.iter().take(5) {
                eprintln!("  top {id:>6} {s:>8.3} {:?}", tok.decode_piece(*id as u32));
            }
        }
        let next = last
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .ok_or("пустые логиты")?;
        if tok.is_eog(next) {
            break;
        }
        print!("{}", dec.push(&tok.piece_bytes(next)));
        std::io::stdout().flush()?;
        produced += 1;
        logits = model.decode(&backend, next)?;
    }
    println!("{}", dec.finish());
    let dt = t2.elapsed().as_secs_f32();
    eprintln!(
        "decode {produced} ток. за {dt:.2}s ({:.1} tok/s)",
        produced as f32 / dt.max(1e-6)
    );
    Ok(())
}
