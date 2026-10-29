//! Приёмка фундамента слотов: две беседы вперемешку обязаны дать РОВНО те же
//! логиты, что их одиночные прогоны (состояние слота переезжает без копий).
//!
//! `cargo run --release --features metal --example slot_switch -- <model.gguf>`
#![cfg(feature = "metal")]

use std::error::Error;

use rust_inference::backend::Backend as _;
use rust_inference::inference::generate::CausalLm;
use rust_inference::metal::MetalBackend;
use rust_inference::{Tensor, gguf, qwen35, tokenizer::Tokenizer, weights_map::WeightsMap};

fn pick(be: &MetalBackend, vocab: usize, logits: &Tensor) -> u32 {
    let v = be.to_cpu_f32(logits).unwrap();
    let last = &v[v.len() - vocab..];
    last.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap()
}

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: slot_switch <model.gguf>")?;
    let file = std::fs::File::open(&path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let gf = gguf::parse(&mmap)?;
    let wm = WeightsMap::from_gguf(&gf)?;
    let tok = Tokenizer::from_gguf(&gf)?;
    let be = MetalBackend::new()?;
    let vocab = tok.vocab_size();
    let mut m = qwen35::load_gguf(&be, &gf, &wm, vocab)?;

    let a = tok.encode("The capital of France is Paris and the capital of Italy is");
    let b = tok.encode("2 + 2 = 4 and 3 + 3 = 6 and 4 + 4 =");
    const N: usize = 4;

    // Эталон: A в одиночку. prefill + N декодов → токен N+1 → два шага логитов.
    m.reset_cache(&be)?;
    let mut lg = m.prefill_last(&be, &a)?;
    let mut solo_tokens = Vec::new();
    for _ in 0..N {
        let id = pick(&be, vocab, &lg);
        solo_tokens.push(id);
        lg = m.decode(&be, id)?;
    }
    let next_solo = pick(&be, vocab, &lg);
    let lg_solo = m.decode(&be, next_solo)?;
    let logits_solo = be.to_cpu_f32(&lg_solo)?;
    let lg_solo2 = m.decode(&be, next_solo)?;
    let logits_solo2 = be.to_cpu_f32(&lg_solo2)?;

    // Вперемешку: A (N декодов) → парк → B → активация A → тот же N+1 токен.
    m.reset_cache(&be)?;
    let mut lg_a = m.prefill_last(&be, &a)?;
    let mut inter_tokens = Vec::new();
    for _ in 0..N {
        let id = pick(&be, vocab, &lg_a);
        inter_tokens.push(id);
        lg_a = m.decode(&be, id)?;
    }
    assert_eq!(inter_tokens, solo_tokens, "A до парковки");
    let pos_a = m.context_len();
    let slot = m.park_slot(&be)?;
    assert_eq!(m.context_len(), 0, "активный контекст после парковки пуст");

    let lg_b = m.prefill_last(&be, &b)?;
    let mut id_b = pick(&be, vocab, &lg_b);
    for _ in 0..3 {
        id_b = pick(&be, vocab, &m.decode(&be, id_b)?);
    }
    let slot_b = m.park_slot(&be)?;
    let _ = slot_b;

    m.activate_slot(slot)?;
    assert_eq!(m.context_len(), pos_a, "позиция слота A восстановлена");
    let next_inter = pick(&be, vocab, &lg_a);
    assert_eq!(next_inter, next_solo, "следующий токен A совпал");
    let lg_inter = m.decode(&be, next_inter)?;
    let logits_inter = be.to_cpu_f32(&lg_inter)?;
    let lg_inter2 = m.decode(&be, next_inter)?;
    let logits_inter2 = be.to_cpu_f32(&lg_inter2)?;

    let same = |x: &[f32], y: &[f32]| -> bool {
        x.len() == y.len() && x.iter().zip(y).all(|(a, b)| a.to_bits() == b.to_bits())
    };
    let step1 = same(&logits_solo, &logits_inter);
    let step2 = same(&logits_solo2, &logits_inter2);
    println!(
        "слот A: pos={pos_a}, токен {next_inter} (эталон {next_solo}); логиты шагов: {} / {}",
        if step1 {
            "равны"
        } else {
            "РАЗОШЛИСЬ"
        },
        if step2 {
            "равны"
        } else {
            "РАЗОШЛИСЬ"
        },
    );
    assert!(step1 && step2, "логиты слота после переключения разошлись");
    println!("OK: слоты переключаются без потери состояния (побитово)");
    Ok(())
}
