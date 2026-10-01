//! e2e-прогон Qwen3-8B на Metal (Ф1.7, веха M1).
//!
//! Запуск: `cargo run --release --features metal --example llm_chat -- \
//!   ~/.vassista-models/llm/Qwen3-8B-Q8_0.gguf "Привет, расскажи о себе"`
//!
//! Пайплайн: mmap GGUF → конфиг/токенайзер/чат-шаблон из метадаты → модель
//! → generate (батч-префилл + декод) → печать текста + метрики: префилл и
//! декод РАЗДЕЛЬНО, по событию Timings движка (семантика `timings`
//! llama-server). Перед замером — прогрев на том же промпте (префилл + шаг
//! декода, как прогрев llama-cli/llama-bench): первый forward процесса
//! платит за создание Metal-пайплайнов и пул буферов (на 8B ≈1.7 с против
//! 0.09 с тёплого префилла), это время печатается отдельно; `LLM_NO_WARMUP=1`
//! — замер холодного старта. Диагностика первого шага (top-20, пробы
//! токенов) — `LLM_TOP=1`; потокенный префилл вместо батча —
//! `LLM_PREFILL_PER_TOKEN=1` (отладка кернелов декода).

#![cfg(feature = "metal")]

use std::error::Error;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use rust_inference::chat_template::{Template, TemplateContext};
use rust_inference::gguf;
use rust_inference::metal::MetalBackend;
use rust_inference::qwen3::{Qwen3Config, Qwen3Model};
use rust_inference::spec::NGramDrafter;
use rust_inference::tokenizer::Tokenizer;
use rust_inference::weights_map::WeightsMap;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).ok_or("usage: llm_chat <model.gguf> [prompt]")?;
    // Промпт: аргумент или файл (LLM_PROMPT_FILE) — длинные контексты
    // (тысячи токенов) в argv неудобны.
    let prompt = match std::env::var("LLM_PROMPT_FILE") {
        Ok(f) if std::path::Path::new(&f).is_file() => {
            std::fs::read_to_string(&f).map_err(|e| format!("prompt file: {e}"))?
        }
        _ => args
            .get(2)
            .cloned()
            .unwrap_or_else(|| "Привет! Коротко: кто ты?".into()),
    };

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
        // Шаблон модели из GGUF; LLM_SYSTEM — системное сообщение,
        // LLM_THINK=0/1 — enable_thinking шаблонов Qwen3/3.5.
        let tpl = Template::compile(tpl_src, special)?;
        let mut pairs: Vec<(&str, &str)> = Vec::new();
        let system = std::env::var("LLM_SYSTEM").ok();
        if let Some(sys) = system.as_deref() {
            pairs.push(("system", sys));
        }
        pairs.push(("user", prompt.as_str()));
        let mut ctx = TemplateContext::from_pairs(&pairs).add_generation_prompt(true);
        ctx.enable_thinking = std::env::var("LLM_THINK").map(|v| v != "0").unwrap_or(true);
        tpl.render(&ctx)?
    };
    let ids = tok.encode(&prompt_text);
    eprintln!(
        "cfg: dim={} layers={} heads={}/{} ffn={} vocab={}",
        cfg.dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.ffn_dim, cfg.vocab_size
    );
    eprintln!("prompt tokens: {} ({} chars)", ids.len(), prompt_text.len());
    if std::env::var("LLM_DBG_TOKENS").is_ok() {
        for (i, &id) in ids.iter().enumerate().take(12) {
            eprintln!(
                "  tok[{i}]={id} {:?}",
                String::from_utf8_lossy(tok.piece(id))
            );
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
        for n in [
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_gate.weight",
            "blk.3.attn_q.weight",
        ] {
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

    #[cfg(feature = "cpu")]
    let backend: Box<dyn rust_inference::backend::Backend> =
        if std::env::var("BACKEND").as_deref() == Ok("cpu") {
            Box::new(rust_inference::cpu::CpuBackend)
        } else {
            Box::new(MetalBackend::new()?)
        };
    #[cfg(not(feature = "cpu"))]
    let backend: Box<dyn rust_inference::backend::Backend> = Box::new(MetalBackend::new()?);
    let mut model = Qwen3Model::load(backend.as_ref(), &wm, cfg.clone())?;

    if std::env::var("INFER_LLM_DEBUG").is_ok() {
        eprintln!("INFER_DEBUG ids={ids:?}");
    }
    if std::env::var("LLM_TOP").is_ok() {
        print_first_step(&mut model, backend.as_ref(), &tok, &ids, cfg.vocab_size)?;
        model.reset_cache(backend.as_ref())?;
    } else if std::env::var("LLM_NO_WARMUP").is_err() {
        let t0 = Instant::now();
        let logits = model.prefill_last(&*backend, &ids)?;
        backend.to_cpu_f32(&logits)?;
        model.decode(&*backend, ids[ids.len() - 1])?;
        model.reset_cache(backend.as_ref())?;
        eprintln!("прогрев (пайплайны, буферы): {:.2}s", t0.elapsed().as_secs_f32());
    }

    // Генерация — через БИБЛИОТЕЧНУЮ петлю inference::generate (та же, что
    // у сервиса): жадная при LLM_TEMP<=0 (по умолчанию), спек-режимы —
    // LLM_SPEC=1 (N-грамм) или LLM_SPEC_DFLASH=<draft.gguf|.safetensors>.
    // Раньше пример держал СОБСТВЕННУЮ копию спек-логики и сам дублировал
    // якорь каждого окна в выводе («E E=mcmc2..») — расхождение с
    // библиотечным путём, которое маскировало реальную приёмку.
    let max_tokens: usize = std::env::var("LLM_MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    let temperature: f32 = std::env::var("LLM_TEMP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let gamma: usize = std::env::var("LLM_GAMMA")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let drafter: Option<std::sync::Arc<std::sync::Mutex<dyn rust_inference::spec::Drafter>>> =
        if let Ok(p) = std::env::var("LLM_SPEC_DFLASH") {
            let ew = rust_inference::weights_map::dequant_f32(
                wm.get("token_embd.weight").ok_or("token_embd.weight missing")?,
            );
            let embed_t = backend.from_cpu_f32(&ew, vec![cfg.vocab_size, cfg.dim])?;
            let d = rust_inference::dflash::load_drafter_gguf(
                backend.as_ref(),
                std::path::Path::new(&p),
                embed_t,
                model.output_projection().clone(),
                cfg.vocab_size,
            )?;
            eprintln!(
                "dflash: layers={} dim={} block={} extract={:?} mask={}",
                d.model.cfg.n_layers,
                d.model.cfg.dim,
                d.model.block_size,
                d.model.extract_layers,
                d.model.mask_token
            );
            Some(std::sync::Arc::new(std::sync::Mutex::new(d)))
        } else if std::env::var("LLM_SPEC").is_ok() {
            Some(std::sync::Arc::new(std::sync::Mutex::new(NGramDrafter::new(4))))
        } else {
            None
        };
    // Потокенный префилл (отладка): кэш заполняется декодом по одному
    // токену, generate получает только последний токен промпта.
    let per_token = std::env::var("LLM_PREFILL_PER_TOKEN").is_ok();
    let gen_ids: Vec<u32> = if per_token && ids.len() > 1 {
        let t0 = Instant::now();
        for &id in &ids[..ids.len() - 1] {
            model.decode(&*backend, id)?;
        }
        eprintln!(
            "потокенный префилл {} токенов: {:.2}s",
            ids.len() - 1,
            t0.elapsed().as_secs_f32()
        );
        vec![ids[ids.len() - 1]]
    } else {
        ids.clone()
    };
    let params = rust_inference::inference::generate::GenerateParams {
        max_tokens,
        temperature,
        top_p: 0.9,
        top_k: 40,
        repeat_penalty: 1.0,
        seed: Some(0x5EED),
        drafter: drafter.clone(),
        spec_gamma: if drafter.is_some() { gamma } else { 0 },
        ..Default::default()
    };
    let mut out = String::new();
    let mut stats = None;
    let mut timings = None;
    let toks = rust_inference::inference::generate::generate(
        &mut model,
        backend.as_ref(),
        &tok,
        &gen_ids,
        &params,
        |ev| {
            use rust_inference::inference::generate::GenerateEvent;
            match ev {
                GenerateEvent::Delta(s) => {
                    print!("{s}");
                    let _ = std::io::stdout().flush();
                    out.push_str(&s);
                }
                GenerateEvent::SpecStats(st) => stats = Some(st),
                GenerateEvent::Timings(t) => timings = Some(t),
                _ => {}
            }
            true
        },
    )?;
    eprintln!();
    if let Some(st) = stats {
        eprintln!(
            "spec: черновиков {} принято {} ({:.1}%)",
            st.drafts,
            st.accepted,
            st.acceptance().unwrap_or(0.0)
        );
    }
    let t = timings.ok_or("generate не прислал Timings")?;
    eprintln!(
        "prefill {} tokens in {:.2}s → {:.1} tok/s (TTFT {:.0} ms)",
        t.prompt_n,
        t.prompt_ms / 1e3,
        t.prompt_per_second(),
        t.prompt_ms
    );
    eprintln!(
        "decode {} tokens ({} steps) in {:.2}s → {:.1} tok/s",
        toks.len(),
        t.decode_steps(),
        t.predicted_ms / 1e3,
        t.predicted_per_second()
    );
    eprintln!("--- output ---\n{out}");
    Ok(())
}

/// Диагностика первого шага (LLM_TOP=1): top-20 логитов после префилла и
/// ранги служебных токенов Qwen3 (<think>, </think>, <|im_end|> …).
fn print_first_step(
    model: &mut Qwen3Model,
    backend: &dyn rust_inference::backend::Backend,
    tok: &Tokenizer,
    ids: &[u32],
    vocab: usize,
) -> Result<(), Box<dyn Error>> {
    let logits = model.prefill_last(backend, ids)?;
    let v0 = backend.to_cpu_f32(&logits)?;
    let n = vocab;
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
    Ok(())
}
