//! Ф1.12 e2e: Qwen2.5-Omni-3B «аудио-вход» — аудио → текст на Metal.
//!
//! Запуск: `cargo run --release --features metal --example audio_chat -- \
//!   ~/.vassista-models/omni3b <вопрос> <audio.wav>`
//!
//! Пайплайн: wav (PCM16 mono 16k) → mel (Slaney 128, Whisper-стиль) →
//! OmniAudioTower (conv1/2 + 32 слоя + avg-pool + proj) → аудио-токены
//! вставляются на `<|audio|>` в input_embeds → Thinker (Qwen3-архитектура,
//! M-RoPE t=h=w=pos) → greedy decode текстового ответа.
//!
//! Env: AUDIO_CHAT_MAX_TOKENS (default 128), AUDIO_CHAT_DUMP_IDS,
//! AUDIO_CHAT_QUESTION (default «Опиши, что ты слышишь»).

#![cfg(feature = "metal")]

use std::error::Error;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use rust_inference::audio;
use rust_inference::backend::Backend as _;
use rust_inference::metal::MetalBackend;
use rust_inference::omni::{self, OmniModel};
use rust_inference::tokenizer::IncrementalDecoder;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let dir = args
        .get(1)
        .ok_or("usage: audio_chat <omni3b-dir> [question] [audio.wav]")?;
    let question = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| std::env::var("AUDIO_CHAT_QUESTION").unwrap_or_else(|_| "Опиши, что ты слышишь".into()));
    let wav_path = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| format!("{dir}/../test.wav"));

    let (wav, sr) = audio::read_wav_pcm16(Path::new(&wav_path))?;
    eprintln!("wav: {} samples ({:.2}s) @ {sr} Hz", wav.len(), wav.len() as f32 / sr as f32);
    if sr != 16000 {
        eprintln!("WARN: ожидается 16 kHz, получили {sr}");
    }

    let backend = MetalBackend::new()?;
    let t0 = Instant::now();
    let mut model = OmniModel::load(&backend, Path::new(dir))?;
    eprintln!("model loaded in {:.1}s", t0.elapsed().as_secs_f32());
    let tok = omni::tokenizer_from_json(Path::new(dir))?;
    eprintln!("tokenizer: vocab {}", tok.vocab_size());

    // аудио → эмбеддинги
    let t1 = Instant::now();
    let (ids, audio_features, pos4d, pos_after) = model.build_audio_input(&backend, &wav, &question, &tok)?;
    eprintln!("audio features: {:?} in {:.1}s", audio_features.shape(), t1.elapsed().as_secs_f32());
    eprintln!("prompt tokens: {}", ids.len());

    // prefill (текст + аудио)
    let t2 = Instant::now();
    let mut logits = model.thinker.prefill_with_vision(
        &backend,
        &ids,
        &audio_features,
        model.cfg.thinker.audio_token_id,
        &pos4d,
        pos_after,
    )?;
    eprintln!("prefill {} tokens in {:.2}s", ids.len(), t2.elapsed().as_secs_f32());

    // топ-5 первого шага
    let v0 = backend.to_cpu_f32(&logits)?;
    let n = model.cfg.thinker.vocab_size;
    let last = &v0[v0.len() - n..];
    let mut top: Vec<(u32, f32)> = last.iter().enumerate().map(|(i, &x)| (i as u32, x)).collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("top-5 first step:");
    for (id, s) in top.iter().take(5) {
        eprintln!("  {:>6} {:.4} {:?}", id, s, String::from_utf8_lossy(tok.piece(*id)));
    }

    // greedy decode
    let max_tokens: usize = std::env::var("AUDIO_CHAT_MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);
    let mut decoder = IncrementalDecoder::new();
    let mut out = String::new();
    let d0 = Instant::now();
    for _ in 0..max_tokens {
        let v = backend.to_cpu_f32(&logits)?;
        let last = &v[v.len() - n..];
        let next = last
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        if tok.eos == Some(next) {
            break;
        }
        let text = decoder.push(tok.decode_piece(next).as_bytes());
        out.push_str(&text);
        print!("{text}");
        std::io::stdout().flush()?;
        logits = model.thinker.decode(&backend, next)?;
    }
    eprintln!("\ngenerated {} tokens in {:.2}s → {:.1} tok/s",
        out.split_whitespace().count(),
        d0.elapsed().as_secs_f32(),
        max_tokens as f32 / d0.elapsed().as_secs_f32());
    eprintln!("--- output ---\n{out}");
    Ok(())
}
