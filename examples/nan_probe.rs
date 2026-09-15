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
    for text in ["<|im_start|>user\nмассив<|im_end|>\n<|im_start|>assistant\n", "<|im_start|>user\nСкажи<|im_end|>\n<|im_start|>assistant\n"] {
        let ids = tok.encode(text);
        model.reset_cache(&be)?;


        let lg = model.prefill(&be, &ids)?;
        let lv = be.to_cpu_f32(&lg)?;
        let n = ids.len();
        let mut nan_logits = 0;
        for x in &lv {
            if x.is_nan() {
                nan_logits += 1;
            }
        }
        // последний hidden → out_norm → скаляр с головой: изолировать шаг
        let _ = model.reset_cache(&be);
        let _ = model.reset_cache(&be);
        if let Ok((ln, _, _)) = model.debug_layer_split(&be, &ids, 2) {
            let vl = be.to_cpu_f32(&ln)?;
            let nn = vl.iter().filter(|x| x.is_nan()).count();
            let mx = vl.iter().cloned().fold(0.0f32, |m, x| m.max(if x.is_finite() { x.abs() } else { 0.0 }));
            eprintln!("   layer2-normed: nan={nn} max_abs={mx:.3} first8={:?}", &vl[..8.min(vl.len())]);
        }
        // бисект: префиксы
        // строка эмбеддингов токена-триггера
        let tabs = be.to_cpu_f32(model.token_embed_table())?;
        let ew = &tabs[33995 * 1024..(33995 + 1) * 1024];
        let nn = ew.iter().filter(|x| x.is_nan()).count();
        let inf = ew.iter().filter(|x| x.is_infinite()).count();
        eprintln!("   emb[33995]: nan={nn} inf={inf} rms={:.4}", (ew.iter().map(|x| x*x).sum::<f32>()/1024.0).sqrt());
        let e2 = &tabs[19311 * 1024..(19311 + 1) * 1024];
        let nn2 = e2.iter().filter(|x| x.is_nan()).count();
        eprintln!("   emb[19311]: nan={nn2}");
        let teo = &tok;
        for (take, _) in [(1usize, 0u32), (5, 0), (10, 0), (20, 0), (40, 0)] {
            let p = ids[..take.min(ids.len())].to_vec();
            let _ = model.reset_cache(&be);
            if let Ok(lp) = model.prefill(&be, &p) {
                if let Ok(vp) = be.to_cpu_f32(&lp) {
                    let nn = vp.iter().filter(|x| x.is_nan()).count();
                    eprintln!("   prefix take={take}: nan={nn}");
                }
            }
        }
        let _ = teo;
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
