//! Проверка нативного MTP в GGUF: есть ли nextn-тензоры и загружается ли
//! драфтер (без запуска самой модели — нужны только nextn-веса и таблицы).
//!
//! `cargo run --release --features metal --example mtp_check -- <model.gguf>`

#![cfg(feature = "metal")]

use std::error::Error;

use rust_inference::backend::Backend as _;
use rust_inference::gguf;
use rust_inference::metal::MetalBackend;
use rust_inference::qwen35::{self, Qwen35Config};
use rust_inference::spec::Drafter as _;
use rust_inference::tokenizer::Tokenizer;
use rust_inference::weights_map::WeightsMap;

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args().nth(1).ok_or("usage: mtp_check <model.gguf>")?;
    let file = std::fs::File::open(&path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let gf = gguf::parse(&mmap)?;
    let wm = WeightsMap::from_gguf(&gf)?;
    let tok = Tokenizer::from_gguf(&gf)?;
    let be = MetalBackend::new()?;
    let cfg = Qwen35Config::from_gguf(&gf, tok.vocab_size())?;
    println!(
        "архитектура {}: слоёв {}, dim {}, nextn: {}",
        cfg.arch,
        cfg.n_layers,
        cfg.dim,
        rust_inference::mtp::has_mtp(&wm, &cfg)
    );
    if !rust_inference::mtp::has_mtp(&wm, &cfg) {
        println!("MTP нет — пакетная спека останется на prompt-lookup");
        return Ok(());
    }
    let mut m = qwen35::load_gguf(&be, &gf, &wm, tok.vocab_size())?;
    let (embed, output) = m.head_tables();
    let mut dr = rust_inference::mtp::MtpDrafter::load(&be, &wm, &cfg, embed, output)?;
    let hist = tok.encode("Repeat exactly: alpha beta gamma delta alpha beta gamma");
    let (drafts, probs) = dr.draft(&be, &hist, 4);
    println!(
        "драфтер загружен: {} черновиков {:?} (вероятности {})",
        drafts.len(),
        drafts,
        probs.len()
    );
    println!("OK: нативный MTP доступен");
    Ok(())
}
