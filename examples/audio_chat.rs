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
    let question = args.get(2).cloned().unwrap_or_else(|| {
        std::env::var("AUDIO_CHAT_QUESTION").unwrap_or_else(|_| "Опиши, что ты слышишь".into())
    });
    let wav_path = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| format!("{dir}/../test.wav"));

    let (wav, sr) = audio::read_wav_pcm16(Path::new(&wav_path))?;
    eprintln!(
        "wav: {} samples ({:.2}s) @ {sr} Hz",
        wav.len(),
        wav.len() as f32 / sr as f32
    );
    if sr != 16000 {
        eprintln!("WARN: ожидается 16 kHz, получили {sr}");
    }

    let backend = MetalBackend::new()?;
    let t0 = Instant::now();
    let mut model = OmniModel::load(&backend, Path::new(dir))?;
    eprintln!("model loaded in {:.1}s", t0.elapsed().as_secs_f32());
    if std::env::var("OMNI_REF").is_ok() {
        let cw0 = backend.to_cpu_f32(&model.tower.conv1_w)?;
        eprintln!(
            "OMNI_REF after_load conv1_w[1,5,:3]={:?}",
            conv3(&cw0, 1, 5)
        );
    }
    let tok = omni::tokenizer_from_json(Path::new(dir))?;
    eprintln!("tokenizer: vocab {}", tok.vocab_size());

    // аудио → эмбеддинги
    let t1 = Instant::now();
    let (ids, audio_features, pos4d, pos_after) = if std::env::var("AUDIO_CHAT_NO_AUDIO").is_ok() {
        // text-only: без аудио-части (локализация: префилл vs аудио-энкодер)
        let ids = tok.encode(&format!(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
             <|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        ));
        let pos4d: Vec<u32> = (0..ids.len() as u32).flat_map(|p| [p, p, p, 0]).collect();
        let n = ids.len();
        (ids, backend.zeros(vec![0, 2048])?, pos4d, n)
    } else {
        model.build_audio_input(&backend, &wav, &question, &tok)?
    };
    eprintln!(
        "audio features: {:?} in {:.1}s",
        audio_features.shape(),
        t1.elapsed().as_secs_f32()
    );
    if std::env::var("OMNI_REF").is_ok() {
        let cw1 = backend.to_cpu_f32(&model.tower.conv1_w)?;
        eprintln!(
            "OMNI_REF after_tower conv1_w[1,5,:3]={:?}",
            conv3(&cw1, 1, 5)
        );
        eprintln!("OMNI_REF after_tower conv1_w[0,0,:3]={:?}", &cw1[..3]);
        eprintln!(
            "OMNI_REF after_tower conv1_w[5,0,:3]={:?}",
            conv3(&cw1, 5, 0)
        );
    }
    if std::env::var("OMNI_REF").is_ok() && !std::env::var("AUDIO_CHAT_NO_AUDIO").is_ok() {
        let (mel_c, t_mel) = rust_inference::audio::log_mel_spectrogram(&wav);
        eprintln!("OMNI_REF our_mel[0,:8]={:?}", &mel_c[..8.min(mel_c.len())]);
        eprintln!(
            "OMNI_REF our_mel[:,0][:8]={:?}",
            (0..8).map(|m| mel_c[m * t_mel]).collect::<Vec<_>>()
        );
        eprintln!("OMNI_REF our_mel[127,0]={}", mel_c[127 * t_mel]);
    }
    if std::env::var("OMNI_REF").is_ok() && !std::env::var("AUDIO_CHAT_NO_AUDIO").is_ok() {
        let cw = backend.to_cpu_f32(&model.tower.conv1_w)?;
        eprintln!("OMNI_REF our_conv1_w[0,0,:8]={:?}", &cw[..3.min(cw.len())]);
        let (mel_c, t_c) = rust_inference::audio::log_mel_spectrogram(&wav);
        let mut tm = vec![0.0f32; t_c * 128];
        for i in 0..t_c {
            for m in 0..128 {
                tm[i * 128 + m] = mel_c[m * t_c + i];
            }
        }
        eprintln!("OMNI_REF our_x[0,:8]={:?}", &tm[..8.min(tm.len())]);
        let cb = backend.to_cpu_f32(&model.tower.conv1_b)?;
        eprintln!("OMNI_REF our_conv1_b[:8]={:?}", &cb[..8.min(cb.len())]);
        let w1: f64 = (0..128)
            .map(|ci| {
                conv3(&cw, 1, ci)
                    .iter()
                    .map(|&v| v as f64)
                    .sum::<f64>()
            })
            .sum();
        let w7: f64 = (0..128)
            .map(|ci| {
                conv3(&cw, 7, ci)
                    .iter()
                    .map(|&v| v as f64)
                    .sum::<f64>()
            })
            .sum();
        eprintln!("OMNI_REF our_w1_sum={w1} our_w7_sum={w7}");
        let m30 = &mel_c[30..36];
        eprintln!("OMNI_REF mel[0][30..36]={m30:?}");
        eprintln!(
            "OMNI_REF our_w[5,0,:3]={:?}",
            conv3(&cw, 5, 0)
        );
        eprintln!("OMNI_REF after_prefill conv1_w[0,0,:3]={:?}", &cw[..3]);
        let af = backend.to_cpu_f32(&audio_features)?;
        let d = model.cfg.thinker.dim;
        eprintln!("OMNI_REF our_tower[0,:8]={:?}", &af[..8.min(af.len())]);
        let last = af.len() - d;
        eprintln!("OMNI_REF our_tower[99,:8]={:?}", &af[last..last + 8]);
        let s: f64 = af.iter().map(|&v| v as f64).sum();
        eprintln!("OMNI_REF our_tower_sum={s}");
    }
    eprintln!("prompt tokens: {}", ids.len());
    if std::env::var("AUDIO_CHAT_DUMP_IDS").is_ok() {
        for (i, &id) in ids.iter().enumerate() {
            eprintln!(
                "  tok[{i}]={id} {:?}",
                String::from_utf8_lossy(tok.piece(id))
            );
        }
    }

    // prefill (текст + аудио)
    let t2 = Instant::now();
    if std::env::var("OMNI_REF").is_ok() && std::env::var("AUDIO_CHAT_NO_AUDIO").is_ok() {
        let e = model.thinker.embed_forward_test(&backend, &ids)?;
        let ev = backend.to_cpu_f32(&e)?;
        let last = ev.len() - model.cfg.thinker.dim;
        eprintln!("OMNI_REF our_emb_out[-1,:8]={:?}", &ev[last..last + 8]);
    }
    let mut logits = if std::env::var("AUDIO_CHAT_NO_AUDIO").is_ok() {
        model.thinker.prefill(&backend, &ids)?
    } else {
        model.thinker.prefill_with_vision(
            &backend,
            &ids,
            &audio_features,
            &[], // deepstack: у Omni-аудио его нет
            model.cfg.thinker.audio_token_id,
            &pos4d,
            pos_after,
        )?
    };
    eprintln!(
        "prefill {} tokens in {:.2}s",
        ids.len(),
        t2.elapsed().as_secs_f32()
    );

    // топ-5 первого шага
    let v0 = backend.to_cpu_f32(&logits)?;
    let n = model.cfg.thinker.vocab_size;
    let last = &v0[v0.len() - n..];
    let mut top: Vec<(u32, f32)> = last
        .iter()
        .enumerate()
        .map(|(i, &x)| (i as u32, x))
        .collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("top-5 first step:");
    for (id, s) in top.iter().take(5) {
        eprintln!(
            "  {:>6} {:.4} {:?}",
            id,
            s,
            String::from_utf8_lossy(tok.piece(*id))
        );
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
        if tok.is_eog(next) {
            break;
        }
        let text = decoder.push(tok.decode_piece(next).as_bytes());
        out.push_str(&text);
        print!("{text}");
        std::io::stdout().flush()?;
        logits = model.thinker.decode(&backend, next)?;
    }
    eprintln!(
        "\ngenerated {} tokens in {:.2}s → {:.1} tok/s",
        out.split_whitespace().count(),
        d0.elapsed().as_secs_f32(),
        max_tokens as f32 / d0.elapsed().as_secs_f32()
    );
    eprintln!("--- output ---\n{out}");
    Ok(())
}

/// Срез conv1-весов [out, in=128, k=3] в точке (out, in): три коэффициента ядра.
fn conv3(w: &[f32], out: usize, ci: usize) -> &[f32] {
    let base = (out * 128 + ci) * 3;
    &w[base..base + 3]
}
