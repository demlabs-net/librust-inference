//! Токенизатор против эталона llama.cpp (`llama-tokenize --ids --no-bos
//! --no-escape`, build 10809 / 5266f24da): id совпадают побайтно на корпусе
//! RU/EN/код/цифры/пробелы/эмодзи/диакритика/спец-токены — для всех типов
//! пред-токенизатора наших моделей (qwen2, qwen35).
//!
//! Эталон — `tests/data/tokenizer_golden.json` (scripts/tokenizer_golden.py
//! --emit). Нужны сами GGUF (токенизатор берётся из их метаданных, веса не
//! грузятся): каталог `INFER_MODELS_DIR` (по умолчанию models/ метарепо).
//! Модели нет — тест сообщает и пропускает её, но если не нашлось НИ ОДНОЙ —
//! падает: молча зелёный тест без проверки хуже красного.

use std::path::PathBuf;
use std::time::Instant;

use rust_inference::gguf;
use rust_inference::tokenizer::Tokenizer;

fn models_dir() -> PathBuf {
    std::env::var("INFER_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../models"))
}

fn load(name: &str) -> Option<(memmap2::Mmap, PathBuf)> {
    let path = models_dir().join("llm").join(name);
    let file = std::fs::File::open(&path).ok()?;
    let mmap = unsafe { memmap2::Mmap::map(&file).ok()? };
    Some((mmap, path))
}

#[test]
fn encode_matches_llama_cpp_golden() {
    let raw = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/tokenizer_golden.json"),
    )
    .expect("tests/data/tokenizer_golden.json");
    let golden: serde_json::Value = serde_json::from_str(&raw).expect("golden json");
    let mut checked = 0usize;
    for (model, cases) in golden.as_object().expect("object") {
        let Some((mmap, path)) = load(model) else {
            eprintln!("нет {model} в {:?} — пропуск модели", models_dir());
            continue;
        };
        let gf = gguf::parse(&mmap).expect("gguf");
        let tok = Tokenizer::from_gguf(&gf).unwrap_or_else(|e| panic!("{path:?}: {e}"));
        for (case, c) in cases.as_object().expect("cases") {
            let text = c["text"].as_str().expect("text");
            let want: Vec<u32> = c["ids"]
                .as_array()
                .expect("ids")
                .iter()
                .map(|v| v.as_u64().expect("id") as u32)
                .collect();
            let got = tok.encode(text);
            if got != want {
                let d = got
                    .iter()
                    .zip(&want)
                    .position(|(a, b)| a != b)
                    .unwrap_or(got.len().min(want.len()));
                panic!(
                    "{model}/{case}: расхождение с позиции {d}: наш {:?} / llama.cpp {:?}",
                    &got[d.saturating_sub(2)..(d + 4).min(got.len())],
                    &want[d.saturating_sub(2)..(d + 4).min(want.len())]
                );
            }
            // decode — обратная операция (спец-токены декодируются в свой текст)
            assert_eq!(
                tok.decode(&got),
                text,
                "{model}/{case}: decode(encode(x)) != x"
            );
            checked += 1;
        }
    }
    assert!(
        checked > 0,
        "ни одной модели в {:?}: проверка не выполнена",
        models_dir()
    );
    eprintln!("совпало с llama.cpp: {checked} текстов");
}

/// Линейная сложность: 1 МБ текста кодируется за секунды, а не часы (раньше
/// BPE шёл по всему тексту одним списком — 30 тыс. символов > 20 минут).
#[test]
fn encode_is_fast_on_long_text() {
    let Some((mmap, _)) = load("Qwen3-0.6B-Instruct-Q8_0.gguf") else {
        eprintln!("нет Qwen3-0.6B-Instruct-Q8_0.gguf — пропуск");
        return;
    };
    let gf = gguf::parse(&mmap).expect("gguf");
    let tok = Tokenizer::from_gguf(&gf).expect("tokenizer");
    let para = "Локальный инференс — это запуск моделей на своём железе. Local inference \
                runs models on your own hardware: 1234, 3.14, «кавычки», code_like(x) {}.\n";
    let text: String = para.repeat(1_000_000 / para.len() + 1);
    let t0 = Instant::now();
    let ids = tok.encode(&text);
    let dt = t0.elapsed();
    eprintln!(
        "{} байт → {} токенов за {:.0} мс",
        text.len(),
        ids.len(),
        dt.as_secs_f64() * 1e3
    );
    assert!(dt.as_secs_f64() < 2.0, "1 МБ кодировался {dt:?}");
    let short = &text[..text.char_indices().nth(30_000).unwrap().0];
    let t1 = Instant::now();
    tok.encode(short);
    assert!(
        t1.elapsed().as_secs_f64() < 0.1,
        "30 тыс. символов: {:?}",
        t1.elapsed()
    );
}

/// Набор EOG — как `special_eog_ids` llama.cpp: у Qwen3 eos = `<|im_end|>`,
/// но и `<|endoftext|>` (им заканчивает текст base-модель), и FIM-
/// разделители завершают генерацию. Раньше проверялся только eos — base-
/// модель после `<|endoftext|>` продолжала писать мусор.
#[test]
fn eog_set_matches_llama_cpp_rules() {
    let Some((mmap, _)) =
        load("Qwen3-8B-Base-Q8_0.gguf").or_else(|| load("Qwen3-0.6B-Instruct-Q8_0.gguf"))
    else {
        eprintln!("нет Qwen3 GGUF — пропуск");
        return;
    };
    let gf = gguf::parse(&mmap).expect("gguf");
    let tok = Tokenizer::from_gguf(&gf).expect("tokenizer");
    let id = |s: &str| -> u32 {
        let ids = tok.encode(s);
        assert_eq!(ids.len(), 1, "{s} обязан быть одним спец-токеном: {ids:?}");
        ids[0]
    };
    for s in [
        "<|im_end|>",
        "<|endoftext|>",
        "<|fim_pad|>",
        "<|repo_name|>",
        "<|file_sep|>",
    ] {
        assert!(tok.is_eog(id(s)), "{s} обязан завершать генерацию");
    }
    for s in ["<|im_start|>", "<think>", "<|vision_start|>"] {
        assert!(!tok.is_eog(id(s)), "{s} не должен завершать генерацию");
    }
    if let Some(eos) = tok.eos {
        assert!(tok.is_eog(eos), "eos всегда в EOG-наборе");
    }
}

/// Стриминг по токенам (piece_bytes → IncrementalDecoder) собирает ровно
/// исходный текст, даже когда символ разбит на несколько токенов (эмодзи,
/// редкие иероглифы): раньше кусок токена проходил через from_utf8_lossy
/// ДО декодера, и клиент получал «���» вместо 🦀.
#[test]
fn streaming_decode_keeps_split_characters() {
    let Some((mmap, _)) =
        load("Qwen3-0.6B-Instruct-Q8_0.gguf").or_else(|| load("Qwen3-8B-Q8_0.gguf"))
    else {
        eprintln!("нет Qwen3 GGUF — пропуск");
        return;
    };
    let gf = gguf::parse(&mmap).expect("gguf");
    let tok = Tokenizer::from_gguf(&gf).expect("tokenizer");
    let text = "Краб 🦀 и 𝔘 и 龘, эмодзи-флаг 🇷🇺 и 👩‍💻.";
    let ids = tok.encode(text);
    assert!(
        ids.iter()
            .any(|&i| std::str::from_utf8(&tok.piece_bytes(i)).is_err()),
        "в тексте должен быть символ, разбитый на токены"
    );
    let mut dec = rust_inference::tokenizer::IncrementalDecoder::new();
    let mut out = String::new();
    for &i in &ids {
        out.push_str(&dec.push(&tok.piece_bytes(i)));
    }
    out.push_str(&dec.finish());
    assert_eq!(out, text);
}
