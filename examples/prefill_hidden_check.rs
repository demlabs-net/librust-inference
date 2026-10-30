//! Проверка: пакетный префилл отдаёт скрытое по всем строкам (вход MTP).
#![cfg(feature = "metal")]
use rust_inference::batch::BatchEngine;
use rust_inference::{
    gguf, metal::MetalBackend, qwen35, tokenizer::Tokenizer, weights_map::WeightsMap,
};
use std::error::Error;
fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: prefill_hidden_check <model.gguf>")?;
    let f = std::fs::File::open(&path)?;
    let mmap = unsafe { memmap2::Mmap::map(&f)? };
    let gf = gguf::parse(&mmap)?;
    let wm = WeightsMap::from_gguf(&gf)?;
    let tok = Tokenizer::from_gguf(&gf)?;
    let be = MetalBackend::new()?;
    let mut m = qwen35::load_gguf(&be, &gf, &wm, tok.vocab_size())?;
    let mut eng = BatchEngine::new(&mut m, &be, 1)?;
    let ids = tok.encode("The quick brown fox jumps over the lazy dog, again and again.");
    let slot = eng.begin_prefill(&be, &ids)?;
    let d = eng.dim();
    let (lg, h) = eng.prefill_chunk_hidden(&be, slot, &ids, true, true)?;
    println!(
        "dim={d} логиты={} скрытое={}",
        lg.is_some(),
        h.as_ref()
            .map(|t| format!("{:?}", t.shape()))
            .unwrap_or_default()
    );
    let (lg2, h2) = eng.prefill_chunk_hidden(&be, slot, &ids[..4], false, true)?;
    println!(
        "промежуточный кусок: логиты={} скрытое={:?}",
        lg2.is_some(),
        h2.map(|t| t.shape().to_vec())
    );
    Ok(())
}
