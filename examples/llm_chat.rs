//! e2e-прогон Qwen3-8B на Metal (Ф1.7, веха M1).
//!
//! Запуск: `cargo run --release --features metal --example llm_chat -- \
//!   ~/.vassista-models/llm/Qwen3-8B-Q8_0.gguf "Привет, расскажи о себе"`
//!
//! Пайплайн: mmap GGUF → конфиг/токенайзер/чат-шаблон из метадаты → модель
//! (f16-веса) → prefill + greedy decode → печать текста + метрики (tok/s,
//! TTFT). Качество валидируется вручную; перф-цели — в бенч-гарнесе (Ф1.8).

#![cfg(feature = "metal")]

use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use rust_inference::backend::Backend as _;
use rust_inference::chat_template::{Template, TemplateContext};
use rust_inference::gguf;
use rust_inference::metal::MetalBackend;
use rust_inference::qwen3::{Qwen3Config, Qwen3Model};
use rust_inference::tokenizer::{IncrementalDecoder, Tokenizer};
use rust_inference::weights_map::WeightsMap;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).ok_or("usage: llm_chat <model.gguf> [prompt]")?;
    let prompt = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "Привет! Коротко: кто ты?".into());

    let file = File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let gf = gguf::parse(&mmap)?;
    let wm = WeightsMap::from_gguf(&gf)?;

    let vocab_size = wm
        .get("token_embd.weight")
        .ok_or("token_embd.weight missing")?
        .shape[0] as usize;
    let cfg = Qwen3Config::from_gguf(&gf, vocab_size)?;
    let tok = Tokenizer::from_gguf(&gf)?;

    // чат-шаблон из метадаты (реальный Qwen3 — с trim-маркерами)
    let tpl_src = gf
        .get("tokenizer.chat_template")
        .and_then(|v| v.as_str())
        .ok_or("tokenizer.chat_template missing")?;
    let mut special = std::collections::HashMap::new();
    if let Some(eos) = tok.eos {
        special.insert(
            "eos_token".to_string(),
            String::from_utf8_lossy(tok.piece(eos)).into_owned(),
        );
    }
    if let Some(bos) = tok.bos {
        special.insert(
            "bos_token".to_string(),
            String::from_utf8_lossy(tok.piece(bos)).into_owned(),
        );
    }
    let tpl = Template::compile(tpl_src, special)?;
    let ctx = TemplateContext {
        messages: vec![std::collections::HashMap::from([
            ("role".to_string(), "user".to_string()),
            ("content".to_string(), prompt.clone()),
        ])],
        extra: std::collections::HashMap::new(),
    }
    .add_generation_prompt(true);
    let prompt_text = tpl.render(&ctx)?;
    let ids = tok.encode(&prompt_text);
    eprintln!(
        "cfg: dim={} layers={} heads={}/{} ffn={} vocab={}",
        cfg.dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.ffn_dim, cfg.vocab_size
    );
    eprintln!("prompt tokens: {} ({} chars)", ids.len(), prompt_text.len());

    let backend = MetalBackend::new()?;
    let mut model = Qwen3Model::load(&backend, &wm, cfg.clone())?;

    // prefill + TTFT
    let t0 = Instant::now();
    let mut logits = model.prefill(&backend, &ids)?;
    let ttft = t0.elapsed();
    eprintln!(
        "prefill {} tokens in {:.2}s (TTFT)",
        ids.len(),
        ttft.as_secs_f32()
    );

    // greedy decode
    let mut decoder = IncrementalDecoder::new();
    let mut generated: usize = 0;
    let d0 = Instant::now();
    let mut out = String::new();
    loop {
        let v = backend.to_cpu_f32(&logits)?;
        let n = cfg.vocab_size;
        let last = &v[v.len() - n..];
        let next = last
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        if tok.eos == Some(next) || generated >= 256 {
            break;
        }
        let piece = tok.piece(next).to_vec();
        let text = decoder.push(&piece);
        out.push_str(&text);
        print!("{text}");
        std::io::stdout().flush()?;
        logits = model.decode(&backend, next)?;
        generated += 1;
    }
    out.push_str(&decoder.finish());
    print!("{}", decoder.finish());
    std::io::stdout().flush()?;
    let dt = d0.elapsed();
    eprintln!();
    eprintln!(
        "generated {generated} tokens in {:.2}s → {:.1} tok/s (decode)",
        dt.as_secs_f32(),
        generated as f32 / dt.as_secs_f32()
    );
    eprintln!("--- output ---\n{out}");
    Ok(())
}
