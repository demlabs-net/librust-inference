//! Ф1.11 e2e: Qwen3-VL-2B «опиши картинку» — полный стек на Metal.
//!
//! Запуск: `cargo run --release --features metal --example vision_chat -- \
//!   ~/.vassista-models/llm/Qwen3VL-2B-Q8_0.gguf \
//!   ~/.vassista-models/llm/mmproj-Qwen3VL-2B-F16.gguf`
//!
//! Пайплайн: текст+vision prefill (image_pad → vision-эмбеддинги, M-RoPE
//! per-token позиции по llama.cpp mtmd: картинка t=pos_0, h=pos_0+i/24,
//! w=pos_0+i%24, e=0; текст t=h=w=pos) → greedy decode.

#![cfg(feature = "metal")]

use std::error::Error;
use std::io::Write;
use std::time::Instant;

use rust_inference::backend::Backend as _;
use rust_inference::gguf;
use rust_inference::metal::MetalBackend;
use rust_inference::qwen3::{Qwen3Config, Qwen3Model};
use rust_inference::tokenizer::{IncrementalDecoder, Tokenizer};
use rust_inference::vision_qwen3vl::VisionQwen3Vl;
use rust_inference::weights_map::WeightsMap;

const IMAGE_PAD: u32 = 151655; // <|image_pad|>
const VISION_NX: usize = 24; // (768/16)/2 — токенов по x после merge 2×2

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args
        .get(1)
        .ok_or("usage: vision_chat <model.gguf> <mmproj.gguf>")?;
    let mmproj_path = args
        .get(2)
        .ok_or("usage: vision_chat <model.gguf> <mmproj.gguf>")?;
    let prompt = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "Опиши эту картинку".into());

    let file = std::fs::File::open(model_path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let gf = gguf::parse(&mmap)?;
    let wm = WeightsMap::from_gguf(&gf)?;
    let vocab_size = wm
        .get("token_embd.weight")
        .ok_or("token_embd.weight missing")?
        .shape[0] as usize;
    let cfg = Qwen3Config::from_gguf(&gf, vocab_size)?;
    let tok = Tokenizer::from_gguf(&gf)?;
    eprintln!(
        "cfg: dim={} layers={} heads={}/{} ffn={} vocab={} mrope={:?}",
        cfg.dim,
        cfg.n_layers,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.ffn_dim,
        cfg.vocab_size,
        cfg.mrope_sections
    );

    let text_only = std::env::var("VISION_CHAT_TEXT_ONLY").is_ok();
    let backend = MetalBackend::new()?;
    let vision = if !text_only {
        // vision-энкодер (mmproj)
        let buf = std::fs::read(mmproj_path)?;
        let gfv = gguf::parse(&buf)?;
        let wmv = WeightsMap::from_gguf(&gfv)?;
        let n_layers = gfv
            .get("clip.vision.block_count")
            .and_then(gguf::GgufValue::as_u32)
            .unwrap_or(24) as usize;
        let v = VisionQwen3Vl::load(&backend, &wmv, n_layers)?;
        eprintln!(
            "vision: n_embd={} ffn={} layers={} proj={} tokens={}",
            v.n_embd,
            v.ffn_dim,
            v.n_layers,
            v.projection_dim,
            v.n_tokens()
        );

        // тестовая картинка: диагональный градиент 768×768 RGB
        let (h, w) = (768usize, 768usize);
        let img: Vec<f32> = (0..h * w * 3)
            .map(|i| {
                let px = (i / 3) % w;
                let py = (i / 3) / w;
                let x = px as f32 / w as f32;
                let y = py as f32 / h as f32;
                [x * 255.0, y * 255.0, ((x + y) / 2.0) * 255.0][i % 3]
            })
            .collect();
        let t0 = Instant::now();
        let (vemb, vds) = v.forward(&backend, &img, h, w)?;
        eprintln!(
            "vision forward: main {:?} + {} deepstack-блока {:?} в {:.1}s",
            vemb.shape(),
            vds.len(),
            vds.first().map(|d| d.shape()),
            t0.elapsed().as_secs_f32()
        );
        Some((v, vemb, vds))
    } else {
        eprintln!("vision: text-only (mmproj не грузим)");
        None
    };
    let n_img = vision.as_ref().map(|(v, _, _)| v.n_tokens()).unwrap_or(576);

    let mut ids: Vec<u32> = if text_only {
        tok.encode(&format!(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
             <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        ))
    } else {
        tok.encode(&format!(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
             <|im_start|>user\n<|vision_start|>"
        ))
    };
    let img_start = ids.len();
    if !text_only {
        ids.extend(std::iter::repeat(IMAGE_PAD).take(n_img));
        ids.extend(tok.encode(&format!(
            "<|vision_end|>\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )));
    }
    eprintln!(
        "prompt tokens: {} (image: {n_img} at {img_start})",
        ids.len()
    );
    if std::env::var("VISION_CHAT_DUMP_IDS").is_ok() {
        for id in &ids {
            println!("ID {id}");
        }
    }
    if std::env::var("VISION_CHAT_DBG_TOKENS").is_ok() {
        for (i, &id) in ids.iter().enumerate() {
            eprintln!(
                "  tok[{i}]={id} {:?}",
                String::from_utf8_lossy(tok.piece(id))
            );
        }
    }

    // M-RoPE позиции (llama.cpp mtmd): текст t=h=w=pos, e=0; картинка —
    // t=pos_0 (равна позиции первого image_pad), h=pos_0+i/24, w=pos_0+i%24.
    // Текст после картинки продолжается с pos_0 + max(nx,ny) = pos_0+24
    // (а не +2304) — логические позиции, не слоты KV.
    let img_end = img_start + n_img;
    let t_img = img_start as u32;
    let mut pos4d = vec![0u32; ids.len() * 4];
    for (i, &id) in ids.iter().enumerate() {
        if !text_only && id == IMAGE_PAD {
            let k = (i - img_start) as u32;
            pos4d[i * 4] = t_img;
            pos4d[i * 4 + 1] = t_img + k / VISION_NX as u32;
            pos4d[i * 4 + 2] = t_img + k % VISION_NX as u32;
            pos4d[i * 4 + 3] = 0;
        } else {
            let p = if i < img_start || text_only {
                i as u32
            } else {
                t_img + VISION_NX as u32 + (i - img_end) as u32
            };
            pos4d[i * 4] = p;
            pos4d[i * 4 + 1] = p;
            pos4d[i * 4 + 2] = p;
            pos4d[i * 4 + 3] = 0;
        }
    }
    // логическая позиция следующего decode (pos_0 + 24 + текст после)
    let pos_after = (t_img + VISION_NX as u32 + (ids.len() - img_end) as u32) as usize;

    let mut model = Qwen3Model::load(&backend, &wm, cfg.clone())?;
    let t0 = Instant::now();
    let mut logits = if let Some((_, vemb, vds)) = &vision {
        model.prefill_with_vision(
            &backend,
            &ids,
            vemb,
            vds,
            IMAGE_PAD,
            &pos4d,
            pos_after,
        )?
    } else {
        model.prefill(&backend, &ids)?
    };
    eprintln!(
        "prefill {} tokens in {:.2}s (TTFT)",
        ids.len(),
        t0.elapsed().as_secs_f32()
    );

    // топ-20 первого шага
    let v0 = backend.to_cpu_f32(&logits)?;
    let n = cfg.vocab_size;
    let last = &v0[v0.len() - n..];
    let mut top: Vec<(u32, f32)> = last
        .iter()
        .enumerate()
        .map(|(i, &x)| (i as u32, x))
        .collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("top-20 first step:");
    for (id, s) in top.iter().take(20) {
        eprintln!(
            "  {:>6} {:.4} {:?}",
            id,
            s,
            String::from_utf8_lossy(tok.piece(*id))
        );
    }
    if std::env::var("VISION_CHAT_DUMP_LOGITS").is_ok() {
        for (i, s) in last.iter().enumerate() {
            println!("LOGIT {i} {s:.6}");
        }
    }

    // greedy decode
    let mut decoder = IncrementalDecoder::new();
    let mut generated: usize = 0;
    let d0 = Instant::now();
    let mut out = String::new();
    loop {
        let v = backend.to_cpu_f32(&logits)?;
        let last = &v[v.len() - n..];
        let next = last
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        let max_tokens: usize = std::env::var("VISION_CHAT_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(128);
        if tok.eos == Some(next) || generated >= max_tokens {
            break;
        }
        let text = decoder.push(tok.decode_piece(next).as_bytes());
        out.push_str(&text);
        print!("{text}");
        std::io::stdout().flush()?;
        logits = model.decode(&backend, next)?;
        generated += 1;
    }
    out.push_str(&decoder.finish());
    print!("{}", decoder.finish());
    std::io::stdout().flush()?;
    let dt = d0.elapsed();
    eprintln!();
    eprintln!(
        "generated {generated} tokens in {:.2}s → {:.1} tok/s (decode)",
        dt.as_secs_f32(),
        generated as f32 / dt.as_secs_f32()
    );
    eprintln!("--- output ---\n{out}");
    Ok(())
}
