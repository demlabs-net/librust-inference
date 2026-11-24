//! Чат-шаблоны против эталона llama.cpp (`llama-server --jinja`,
//! `/apply-template`): наш рендер `tokenizer.chat_template` из GGUF обязан
//! совпасть побайтно на диалогах с system, многоходовостью и `<think>` в
//! истории, `reasoning_content`, tools/tool_calls/tool и enable_thinking.
//!
//! Эталон — tests/data/template_golden.json (scripts/template_golden.py).
//! Нужны GGUF (только метаданные): `INFER_MODELS_DIR`, по умолчанию models/
//! метарепо. Нет ни одной модели — тест падает (молча зелёный хуже красного).

use std::collections::HashMap;
use std::path::PathBuf;

use rust_inference::chat_template::{Template, TemplateContext};
use rust_inference::gguf::{self, GgufValue};

fn models_dir() -> PathBuf {
    std::env::var("INFER_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models"))
}

/// Тело запроса /apply-template → наш контекст (те же правила, что у
/// llama.cpp: tools — только если непусты, chat_template_kwargs — переменные).
fn context(body: &serde_json::Value) -> TemplateContext {
    let mut ctx = TemplateContext {
        messages: body["messages"].as_array().cloned().unwrap_or_default(),
        tools: body["tools"].as_array().cloned().unwrap_or_default(),
        add_generation_prompt: true,
        ..Default::default()
    };
    if let Some(kw) = body["chat_template_kwargs"].as_object() {
        for (k, v) in kw {
            if k == "enable_thinking" {
                ctx.enable_thinking = v.as_bool().unwrap_or(true);
            } else {
                ctx.extra.insert(k.clone(), v.clone());
            }
        }
    }
    ctx
}

#[test]
fn chat_templates_match_llama_cpp() {
    let raw = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/template_golden.json"),
    )
    .expect("tests/data/template_golden.json");
    let golden: serde_json::Value = serde_json::from_str(&raw).expect("json");
    let cases = golden["cases"].as_object().expect("cases");
    let mut checked = 0usize;
    for (model, rendered) in golden["models"].as_object().expect("models") {
        let path = models_dir().join("llm").join(model);
        let Ok(file) = std::fs::File::open(&path) else {
            eprintln!("нет {path:?} — пропуск модели");
            continue;
        };
        let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap") };
        let gf = gguf::parse(&mmap).expect("gguf");
        let src = gf
            .get("tokenizer.chat_template")
            .and_then(GgufValue::as_str)
            .expect("chat_template");
        let tpl = Template::compile(src, HashMap::new()).unwrap_or_else(|e| panic!("{model}: {e}"));
        for (case, want) in rendered.as_object().expect("rendered") {
            let ctx = context(&cases[case]);
            let got = tpl.render(&ctx);
            match (want.get("prompt").and_then(|p| p.as_str()), got) {
                (Some(w), Ok(g)) => {
                    if g != w {
                        let d = g.chars().zip(w.chars()).take_while(|(a, b)| a == b).count();
                        let tail = |s: &str| {
                            s.chars()
                                .skip(d.saturating_sub(30))
                                .take(90)
                                .collect::<String>()
                        };
                        panic!(
                            "{model}/{case}: расхождение с символа {d}\n  наш:   {:?}\n  llama: {:?}",
                            tail(&g),
                            tail(w)
                        );
                    }
                }
                (None, Err(_)) => {} // эталон тоже отверг диалог
                (Some(_), Err(e)) => panic!("{model}/{case}: наш рендер упал: {e}"),
                (None, Ok(g)) => panic!("{model}/{case}: эталон отверг, а мы отрендерили {g:?}"),
            }
            checked += 1;
        }
    }
    assert!(checked > 0, "ни одной модели в {:?}", models_dir());
    eprintln!("совпало с llama.cpp: {checked} рендеров");
}
