//! Ручная проверка реального Qwen3 GGUF на эталонном CPU-бэкенде:
//! if CPU текст правильный → проблемы в Metal-кернелах; если мусор —
//! в загрузке/промпте. Запуск: cargo test --features cpu --test real_qwen_cpu -- --ignored --nocapture

use rust_inference::backend::Backend;
use rust_inference::gguf;
use rust_inference::inference::generate::{GenerateEvent, GenerateParams, generate};
use rust_inference::qwen3::{Qwen3Config, Qwen3Model};
use rust_inference::tokenizer::Tokenizer;
use rust_inference::weights_map::WeightsMap;

#[test]
#[ignore]
fn real_qwen_cpu_generation_is_sane() {
    let path = std::env::var("QWEN_GGUF").expect("QWEN_GGUF");
    let file = std::fs::File::open(&path).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let parsed = gguf::parse(&mmap).unwrap();
    let tok = Tokenizer::from_gguf(&parsed).unwrap();
    let wm = WeightsMap::from_gguf(&parsed).unwrap();
    let cfg = Qwen3Config::from_gguf(&parsed, tok.vocab_size()).unwrap();
    let backend = rust_inference::cpu::CpuBackend;
    let mut model = Qwen3Model::load(&backend, &wm, cfg).unwrap();
    let ids = tok.encode("Hello, who are you?");
    let mut out = String::new();
    generate(
        &mut model,
        &backend,
        &tok,
        &ids,
        &GenerateParams {
            max_tokens: 24,
            temperature: 0.0,
            ..GenerateParams::default()
        },
        |ev| {
            if let GenerateEvent::Delta(t) = ev {
                out.push_str(&t);
            }
        },
    )
    .unwrap();
    println!("CPU text: {out:?}");
    assert!(
        !out.chars().all(|c| c == '*' || c == '\n'),
        "вырожденный вывод: {out:?}"
    );
}
