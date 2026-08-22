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
    let prompt_text = if std::env::var("LLM_RAW_PROMPT").is_ok() {
        // базовые модели не следуют чат-шаблону — чистое продолжение текста
        prompt.clone()
    } else {
        match Template::compile(tpl_src, special) {
            Ok(tpl) => {
                let ctx = TemplateContext {
                    messages: vec![std::collections::HashMap::from([
                        ("role".to_string(), "user".to_string()),
                        ("content".to_string(), prompt.clone()),
                    ])],
                    extra: std::collections::HashMap::new(),
                }
                .add_generation_prompt(true);
                tpl.render(&ctx)?
            }
            Err(rust_inference::chat_template::TemplateError::Unsupported(_)) => {
                // Реальный Qwen3-шаблон использует {% set ns = namespace(...) %}
                // (tool-calling-обвязка) — вне minja-подмножества v1; классический
                // im_start-формат эквивалентен для валидации движка (Ф1.7).
                format!(
                    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                )
            }
            Err(e) => return Err(e.into()),
        }
    };
    let ids = tok.encode(&prompt_text);
    eprintln!(
        "cfg: dim={} layers={} heads={}/{} ffn={} vocab={}",
        cfg.dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.ffn_dim, cfg.vocab_size
    );
    eprintln!("prompt tokens: {} ({} chars)", ids.len(), prompt_text.len());
    if std::env::var("LLM_DBG_TOKENS").is_ok() {
        for (i, &id) in ids.iter().enumerate().take(12) {
            eprintln!("  tok[{i}]={id} {:?}", String::from_utf8_lossy(tok.piece(id)));
        }
        eprintln!("  roundtrip: {:?}", tok.decode(&ids));
    }
    for n in [
        "blk.0.attn_q_norm.weight",
        "blk.0.attn_k_norm.weight",
        "blk.0.attn_norm.weight",
        "blk.0.ffn_norm.weight",
        "output_norm.weight",
    ] {
        if let Some(t) = wm.get(n) {
            eprintln!("SHAPE {n}: {:?}", t.shape);
        }
    }
    if std::env::var("LLM_DBG_TOKENS").is_ok() {
        let specials = [tok.bos, tok.eos, tok.pad];
        eprintln!("special ids: {:?}", specials);
        for n in ["blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight", "blk.0.attn_output.weight", "blk.0.ffn_gate.weight", "blk.3.attn_q.weight"] {
            if let Some(t) = wm.get(n) {
                let d = rust_inference::weights_map::dequant_f32(t);
                let rms = (d.iter().map(|v| (v * v) as f64).sum::<f64>() / d.len() as f64).sqrt();
                eprintln!("DBG {n}: shape={:?} rms={rms:.5}", t.shape);
            }
        }
        let mut hits: Vec<(u32, String)> = Vec::new();
        for id in 0..vocab_size as u32 {
            let piece = String::from_utf8_lossy(tok.piece(id)).into_owned();
            if piece.contains("think") || piece.contains("<|") || piece.starts_with('<') {
                hits.push((id, piece));
            }
        }
        let think_hits: Vec<_> = hits
            .iter()
            .filter(|(_, p)| p.contains("think") || p.contains("<|"))
            .collect();
        eprintln!("think/special tokens ({}):", think_hits.len());
        for (id, p) in think_hits {
            eprintln!("  {id:>6} {:?}", p);
        }
    }

    let backend: Box<dyn rust_inference::backend::Backend> =
        if std::env::var("BACKEND").as_deref() == Ok("cpu") {
            Box::new(rust_inference::cpu::CpuBackend)
        } else {
            Box::new(MetalBackend::new()?)
        };
    let mut model = Qwen3Model::load(backend.as_ref(), &wm, cfg.clone())?;

    // prefill + TTFT
    let t0 = Instant::now();
    let mut logits = if std::env::var("LLM_BATCH").is_ok() {
        model.prefill(&*backend, &ids)?
    } else {
        let mut l = model.decode(&*backend, ids[0])?;
        for &id in &ids[1..] {
            l = model.decode(&*backend, id)?;
        }
        l
    };
    let ttft = t0.elapsed();
    eprintln!(
        "prefill {} tokens in {:.2}s (TTFT)",
        ids.len(),
        ttft.as_secs_f32()
    );

    // top-20 первого шага
    let v0 = backend.to_cpu_f32(&logits)?;
    let n = cfg.vocab_size;
    let last = &v0[v0.len() - n..];
    let mut top: Vec<(u32, f32)> = last
        .iter()
        .enumerate()
        .map(|(i, &x)| (i as u32, x))
        .collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("top-20 first step:");
    for (id, s) in top.iter().take(20) {
        eprintln!(
            "  {:>6} {:.4} {:?}",
            id,
            s,
            String::from_utf8_lossy(tok.piece(*id))
        );
    }
    // думающий канал Qwen3: <think>=151667, </think>=151668 — где они в наших логитах?
    for probe in [151667u32, 151668, 27, 29, 26865, 151645] {
        let rank = top.iter().position(|(id, _)| *id == probe);
        let score = top
            .iter()
            .find(|(id, _)| *id == probe)
            .map(|(_, s)| *s)
            .unwrap_or(f32::NAN);
        eprintln!(
            "  probe {probe} ({:?}): rank={:?} score={:.4}",
            String::from_utf8_lossy(tok.piece(probe)),
            rank,
            score
        );
    }

    // greedy decode
    let mut decoder = IncrementalDecoder::new();
    let mut generated: usize = 0;
    let d0 = Instant::now();
    let mut out = String::new();
    if std::env::var("LLM_DUMP_LOGITS").is_ok() {
        let vd = backend.to_cpu_f32(&logits)?;
        let n = cfg.vocab_size;
        let last = &vd[vd.len() - n..];
        for (i, s) in last.iter().enumerate() {
            println!("LOGIT {i} {s:.6}");
        }
    }
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
        if std::env::var("LLM_DBG_TOKENS").is_ok() && generated % 5 == 0 {
            let mut top: Vec<(u32, f32)> = last
                .iter()
                .enumerate()
                .map(|(i, &x)| (i as u32, x))
                .collect();
            top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            eprintln!("top-5 step {generated}:");
            for (id, s) in top.iter().take(5) {
                eprintln!(
                    "  {:>6} {:.4} {:?}",
                    id,
                    s,
                    String::from_utf8_lossy(tok.piece(*id))
                );
            }
        }
        let max_tokens: usize = std::env::var("LLM_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256);
        if tok.eos == Some(next) || generated >= max_tokens {
            break;
        }
        let text = decoder.push(tok.decode_piece(next).as_bytes());
        out.push_str(&text);
        print!("{text}");
        std::io::stdout().flush()?;
        logits = model.decode(&*backend, next)?;
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
