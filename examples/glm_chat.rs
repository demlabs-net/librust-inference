//! GLM-4-чат: префилл + жадный декод через библиотечную петлю generate —
//! раннер для паритета с llama.cpp (`scripts/text_parity.py --impl glm_chat`).
//!
//! Запуск: glm_chat <model.gguf> "<промпт>" [max_tokens]
//! Env: LLM_SYSTEM (системное сообщение), LLM_RAW_PROMPT (без шаблона),
//!      INFER_KV_F32 (A/B-рычаг KV).
#![cfg(feature = "metal")]
use rust_inference::chat_template::{Template, TemplateContext};
use rust_inference::glm4::{Glm4Config, Glm4Model};
use rust_inference::inference::generate::CausalLm;
use rust_inference::metal::MetalBackend;
use rust_inference::tokenizer::Tokenizer;
use rust_inference::weights_map::WeightsMap;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: glm_chat <model.gguf> \"<prompt>\" [max_tokens]");
        std::process::exit(2);
    }
    let max_tokens: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);
    let prompt = args[2].clone();

    let bytes = std::fs::read(&args[1])?;
    let gf = rust_inference::gguf::parse(&bytes)?;
    let wm = WeightsMap::from_gguf(&gf)?;
    let tok = Tokenizer::from_gguf(&gf)?;
    let cfg = Glm4Config::from_gguf(&gf, tok.vocab_size())?;
    let backend: Box<dyn rust_inference::backend::Backend> = Box::new(MetalBackend::new()?);
    let mut model = Glm4Model::load(backend.as_ref(), &wm, cfg.clone())?;

    // Промпт: шаблон модели из GGUF (у glm4-конвертеров это пресет-имя —
    // chat_template.rs подставляет формат llama-chat.cpp).
    let prompt_text = if std::env::var("LLM_RAW_PROMPT").is_ok() {
        prompt.clone()
    } else {
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
        let system = std::env::var("LLM_SYSTEM").ok();
        let mut pairs: Vec<(&str, &str)> = Vec::new();
        if let Some(sys) = system.as_deref() {
            pairs.push(("system", sys));
        }
        pairs.push(("user", prompt.as_str()));
        let ctx = TemplateContext::from_pairs(&pairs).add_generation_prompt(true);
        tpl.render(&ctx)?
    };
    if std::env::var("LLM_DUMP_PROMPT").is_ok() {
        eprintln!("PROMPT={prompt_text:?}");
    }
    let ids = tok.encode(&prompt_text);
    eprintln!(
        "cfg: {} dim={} layers={} heads={}/{} head_dim={} n_rot={} ffn={} vocab={}",
        cfg.arch,
        cfg.dim,
        cfg.n_layers,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
        cfg.n_rot,
        cfg.ffn_dim,
        cfg.vocab_size
    );
    eprintln!("prompt tokens: {} ({} chars)", ids.len(), prompt_text.len());

    // Диагностика паритета: логиты послед позиции префилла в формате дампа
    // оркала (oracle_dump.cpp) — %08x-биты, сравнение побитовое.
    if std::env::var("LLM_DUMP_LOGITS").is_ok() {
        model.reset_cache(backend.as_ref())?;
        let lg = model.prefill_last(backend.as_ref(), &ids)?;
        let v = backend.to_cpu_f32(&lg)?;
        let out = std::env::var("LLM_DUMP_LOGITS").unwrap();
        let mut s = String::new();
        s.push_str("STEP 0 TENSOR result_output ne=[");
        s.push_str(&format!("{},1,1,1]\n", cfg.vocab_size));
        let last = &v[v.len() - cfg.vocab_size..];
        for (i, x) in last.iter().enumerate() {
            s.push_str(&format!(
                "{:08x}{}",
                x.to_bits(),
                if i % 8 == 7 { "\n" } else { " " }
            ));
        }
        s.push('\n');
        std::fs::write(&out, s)?;
        eprintln!("logits dumped");
        // Шаг декода (KV = t+1): оркал n_steps=1 жадно декодит argmax шага 0.
        if std::env::var("LLM_DUMP_DECODE").is_ok() {
            let v0 = backend.to_cpu_f32(&lg)?;
            let row = &v0[v0.len() - cfg.vocab_size..];
            let tok = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .unwrap();
            let lg2 = model.decode(backend.as_ref(), tok)?;
            let v2 = backend.to_cpu_f32(&lg2)?;
            let mut s = String::new();
            s.push_str("STEP 1 TENSOR result_output ne=[");
            s.push_str(&format!("{},1,1,1]\n", cfg.vocab_size));
            let last = &v2[v2.len() - cfg.vocab_size..];
            for (i, x) in last.iter().enumerate() {
                s.push_str(&format!(
                    "{:08x}{}",
                    x.to_bits(),
                    if i % 8 == 7 { "\n" } else { " " }
                ));
            }
            s.push('\n');
            std::fs::write(std::env::var("LLM_DUMP_DECODE").unwrap(), s)?;
            eprintln!("decode logits dumped (tok {tok})");
        }
    }

    let params = rust_inference::inference::generate::GenerateParams {
        max_tokens,
        temperature: 0.0, // жадный — как llama.cpp при сравнении текстов
        top_p: 0.9,
        top_k: 40,
        repeat_penalty: 1.0,
        seed: Some(0x5EED),
        ..Default::default()
    };
    let mut timings = None;
    rust_inference::inference::generate::generate(
        &mut model,
        backend.as_ref(),
        &tok,
        &ids,
        &params,
        |ev| {
            use rust_inference::inference::generate::GenerateEvent;
            match ev {
                GenerateEvent::Delta(s) => {
                    print!("{s}");
                    let _ = std::io::stdout().flush();
                }
                GenerateEvent::Timings(t) => timings = Some(t),
                _ => {}
            }
            true
        },
    )?;
    println!();
    if let Some(t) = timings {
        eprintln!(
            "prefill {} ток. за {:.2}s ({:.1} tok/s); decode {} ток. ({:.1} tok/s)",
            t.prompt_n,
            t.prompt_ms / 1000.0,
            t.prompt_per_second(),
            t.predicted_n,
            t.predicted_per_second()
        );
    }
    Ok(())
}
