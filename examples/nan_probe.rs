//! NaN-звонд: forward_hidden для триггерных токенов.
use rust_inference::backend::Backend as _;
use rust_inference::gguf;
use rust_inference::metal::MetalBackend;
use rust_inference::qwen3::{Qwen3Config, Qwen3Model};
use rust_inference::tokenizer::Tokenizer;
use rust_inference::weights_map::WeightsMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let be = MetalBackend::new()?;
    let path = std::path::PathBuf::from(
        "/Users/dmitriygerasimov/work/ai/inference/models/llm/Qwen3-0.6B-Instruct-Q8_0-lmkit.gguf",
    );
    let buf = std::fs::read(&path)?;
    let gf = gguf::parse(&buf)?;
    let wm = WeightsMap::from_gguf(&gf)?;
    let tok = Tokenizer::from_gguf(&gf)?;
    let cfg = Qwen3Config::from_gguf(&gf, tok.vocab_size())?;
    let dim = cfg.dim;
    let mut model = Qwen3Model::load(&be, &wm, cfg.clone())?;
    use rust_inference::inference::generate::{GenerateEvent, GenerateParams, generate};
    for text in [
        "<|im_start|>user\nСкажи<|im_end|>\n<|im_start|>assistant\n",
        "Скажи",
    ] {
        let ids = tok.encode(text);
        model.reset_cache(&be)?;
        // полный generate как в движке: greedy, тот же промпт
        {
            let mut out = String::new();
            let r = generate(
                &mut model,
                &be,
                &tok,
                &ids,
                &GenerateParams {
                    max_tokens: 4,
                    temperature: 0.0,
                    top_p: 1.0,
                    top_k: 0,
                    repeat_penalty: 1.0,
                    ..Default::default()
                },
                |ev| {
                    if let GenerateEvent::Delta(t) = ev {
                        out.push_str(&t);
                    }
                    true
                },
            );
            eprintln!("generate(fresh) for {text:?}: {out:?} err={r:?}");
        }

        let lg = model.prefill(&be, &ids)?;
        let lv = be.to_cpu_f32(&lg)?;
        let n = ids.len();
        let mut nan_logits = 0;
        for x in &lv {
            if x.is_nan() {
                nan_logits += 1;
            }
        }
        let vv = tok.vocab_size();
        let last_row = &lv[lv.len() - vv..];
        let mut best = (0usize, f32::NEG_INFINITY);
        for (i, x) in last_row.iter().enumerate() {
            if x.is_finite() && *x > best.1 {
                best = (i, *x);
            }
        }
        eprintln!(
            "{text:?}: prefill-logits nan={nan_logits} argmax={} ({:?})",
            best.0,
            tok.decode_piece(best.0 as u32).chars().take(12).collect::<String>()
        );
        let h = model.forward_hidden(&be, &ids)?;
        let v = be.to_cpu_f32(&h)?;
        let mut nan_rows = 0;
        for r in 0..n {
            let row = &v[r * dim..(r + 1) * dim];
            if row.iter().any(|x| x.is_nan()) {
                nan_rows += 1;
            }
        }
        let last = &v[v.len() - dim..v.len()];
        eprintln!(
            "{text:?} ids={ids:?} rows={n} nan_rows={nan_rows} last_rms={:.4}",
            (last.iter().map(|x| x * x).sum::<f32>() / dim as f32).sqrt()
        );
    }
    Ok(())
}
