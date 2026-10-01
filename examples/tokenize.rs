//! Токенизация текста токенизатором из GGUF — сверка с `llama-tokenize`.
//!
//! `tokenize <model.gguf> <файл>` печатает id через запятую в квадратных
//! скобках (тот же формат, что `llama-tokenize --ids --no-bos`) и время
//! кодирования в stderr. Читает только метаданные — веса не грузятся.

use std::error::Error;
use std::time::Instant;

use rust_inference::gguf;
use rust_inference::tokenizer::Tokenizer;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).ok_or("usage: tokenize <model.gguf> <text-file>")?;
    let text_path = args.get(2).ok_or("usage: tokenize <model.gguf> <text-file>")?;
    let text = std::fs::read_to_string(text_path)?;
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let gf = gguf::parse(&mmap)?;
    let tok = Tokenizer::from_gguf(&gf)?;
    let t0 = Instant::now();
    let ids = tok.encode(&text);
    let dt = t0.elapsed();
    eprintln!(
        "{} символов → {} токенов за {:.1} мс",
        text.chars().count(),
        ids.len(),
        dt.as_secs_f64() * 1e3
    );
    let body: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
    println!("[{}]", body.join(", "));
    Ok(())
}
