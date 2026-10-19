#![allow(clippy::needless_range_loop, clippy::explicit_counter_loop)]
//! Host-эталон слоя 0 GLM-4 (независимая реализация тех же формул) — для
//! постадийной сверки с обучающим/инференс-путём через `GLM4_DEBUG=1`.
//!
//! Печатает те же теги, что и glm4.rs (embed, attn_norm, q_bias, ..., ffn_out)
//! для ПЕРВОГО токена промпта. Расхождение локализует баг стадии.
//!
//! Запуск: glm_probe <model.gguf> "<промпт>"
use rust_inference::glm4::Glm4Config;
use rust_inference::tokenizer::Tokenizer;
use rust_inference::weights_map::{WeightsMap, dequant_f32};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&args[1])?;
    let gf = rust_inference::gguf::parse(&bytes)?;
    let wm = WeightsMap::from_gguf(&gf)?;
    let tok = Tokenizer::from_gguf(&gf)?;
    let cfg = Glm4Config::from_gguf(&gf, tok.vocab_size())?;
    let ids = tok.encode(args.get(2).map(|s| s.as_str()).unwrap_or("Привет"));
    if let Ok(id) = std::env::var("PRINT_PIECE") {
        let id: u32 = id.parse().unwrap();
        eprintln!("piece({id}) = {:?}", tok.decode_piece(id));
    }
    let first = ids[0];
    eprintln!("prompt ids: {ids:?} (первый {first})");

    let dim = cfg.dim;
    let hd = cfg.head_dim;
    let (nh, nkv) = (cfg.n_heads, cfg.n_kv_heads);
    let (qd, kd) = (nh * hd, nkv * hd);
    let get = |name: &str| -> Vec<f32> {
        let t = wm.get(name).unwrap_or_else(|| panic!("нет тензора {name}"));
        dequant_f32(t)
    };
    let norm = |x: &[f32], w: &[f32], eps: f32| -> Vec<f32> {
        let ss: f64 = x.iter().map(|v| (*v as f64) * (*v as f64)).sum();
        let s = (1.0 / (ss / x.len() as f64 + eps as f64).sqrt()) as f32;
        x.iter().zip(w).map(|(a, b)| a * s * b).collect()
    };
    let matvec = |w: &[f32], out: usize, inp: usize, x: &[f32]| -> Vec<f32> {
        (0..out)
            .map(|o| {
                let row = &w[o * inp..(o + 1) * inp];
                row.iter().zip(x).map(|(a, b)| a * b).sum()
            })
            .collect()
    };
    let show = |tag: &str, v: &[f32]| {
        eprintln!("HOST {tag}: {:?}", &v[..8.min(v.len())]);
    };

    // эмбеддинг + attn_norm
    let emb_all = get("token_embd.weight");
    let emb = emb_all[first as usize * dim..(first as usize + 1) * dim].to_vec();
    show("embed", &emb);
    let h = norm(&emb, &get("blk.0.attn_norm.weight"), cfg.eps);
    show("attn_norm", &h);

    // q/k/v + bias
    let mut q = matvec(&get("blk.0.attn_q.weight"), qd, dim, &h);
    show("q_pre", &q);
    let mut k = matvec(&get("blk.0.attn_k.weight"), kd, dim, &h);
    let v = matvec(&get("blk.0.attn_v.weight"), kd, dim, &h);
    let qb = get("blk.0.attn_q.bias");
    let kb = get("blk.0.attn_k.bias");
    let vb = get("blk.0.attn_v.bias");
    for (x, b) in q.iter_mut().zip(&qb) {
        *x += b;
    }
    for (x, b) in k.iter_mut().zip(&kb) {
        *x += b;
    }
    let v: Vec<f32> = v.iter().zip(&vb).map(|(a, b)| a + b).collect();
    show("q_bias", &q);
    show("k_bias", &k);
    show("v_bias", &v);

    // частичный RoPE: GLM-4 = ROPE_TYPE_NORM (соседние пары), pos = 0 → identity,
    // но проверим формулу на pos=1 (первый токен промпта у нас в позиции 0).
    let rope_adj = |x: &[f32], pos: usize| -> Vec<f32> {
        let mut y = x.to_vec();
        let half = cfg.n_rot / 2;
        for hh in 0..x.len() / hd {
            let base = hh * hd;
            for kk in 0..half {
                let inv = 1.0 / cfg.rope_theta.powf(2.0 * kk as f32 / cfg.n_rot as f32);
                let (c, s) = ((pos as f32 * inv).cos(), (pos as f32 * inv).sin());
                let (i0, i1) = (base + 2 * kk, base + 2 * kk + 1);
                let (x0, x1) = (x[i0], x[i1]);
                y[i0] = x0 * c - x1 * s;
                y[i1] = x0 * s + x1 * c;
            }
        }
        y
    };
    q = rope_adj(&q, 0);
    k = rope_adj(&k, 0);
    show("q_rope", &q);
    show("k_rope", &k);

    // внимание при t=1: выход = v своей kv-головы
    let mut av = vec![0f32; qd];
    for hh in 0..nh {
        let kh = hh / (nh / nkv);
        av[hh * hd..(hh + 1) * hd].copy_from_slice(&v[kh * hd..(kh + 1) * hd]);
    }
    show("attn_out", &av);

    // ── ПОЛНЫЙ forward: все слои, потоково (веса слоя живут один слой) ──
    // Декод по одному токену с host-KV (f32) — независимый эталон того же
    // графа. Печатает топ-1 логитов и (при GLM4_DEBUG_L) x после каждого слоя.
    let full = std::env::var("GLM4_FULL").is_ok();
    if !full {
        return Ok(());
    }
    let steps: usize = std::env::var("GLM4_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let eos = tok.eos;
    let out_norm_w = get("output_norm.weight");
    let out_w = get("output.weight");
    // host-KV: k/v по слоям [pos][kd]
    let mut kv: Vec<(Vec<f32>, Vec<f32>)> = vec![(Vec::new(), Vec::new()); cfg.n_layers];
    let mut pos = 0usize;
    let mut x = vec![0f32; dim];
    let mut next = first;
    let mut text = String::new();
    // Сначала ВЕСЬ промпт (позиции 0..len), затем генерация — иначе x[i]
    // эталона и движка соответствовали бы разным токенам.
    for step in 0..(ids.len() + steps) {
        let tok_in = if step < ids.len() { ids[step] } else { next };
        if step >= ids.len() {
            if Some(tok_in) == eos {
                break;
            }
            text.push_str(&tok.decode_piece(tok_in));
        }
        // эмбеддинг
        x.copy_from_slice(&emb_all[tok_in as usize * dim..(tok_in as usize + 1) * dim]);
        for il in 0..cfg.n_layers {
            let p = |s: &str| format!("blk.{il}.{s}");
            let inp = x.clone();
            let h = norm(&x, &get(&p("attn_norm.weight")), cfg.eps);
            let mut q = matvec(&get(&p("attn_q.weight")), qd, dim, &h);
            let mut k = matvec(&get(&p("attn_k.weight")), kd, dim, &h);
            let mut v = matvec(&get(&p("attn_v.weight")), kd, dim, &h);
            for (y, b) in q.iter_mut().zip(&get(&p("attn_q.bias"))) {
                *y += b;
            }
            for (y, b) in k.iter_mut().zip(&get(&p("attn_k.bias"))) {
                *y += b;
            }
            for (y, b) in v.iter_mut().zip(&get(&p("attn_v.bias"))) {
                *y += b;
            }
            q = rope_adj(&q, pos);
            k = rope_adj(&k, pos);
            kv[il].0.extend_from_slice(&k);
            kv[il].1.extend_from_slice(&v);
            // внимание: causal softmax по всем tkv ключам, GQA h/(nh/nkv)
            let tkv = pos + 1;
            let mut attn = vec![0f32; qd];
            let scale = 1.0 / (hd as f32).sqrt();
            for hh in 0..nh {
                let kh = hh / (nh / nkv);
                let qrow = &q[hh * hd..(hh + 1) * hd];
                let mut scores = vec![0f32; tkv];
                let mut mx = f32::NEG_INFINITY;
                for t in 0..tkv {
                    let krow = &kv[il].0[t * kd + kh * hd..t * kd + (kh + 1) * hd];
                    let dot: f32 = qrow.iter().zip(krow).map(|(a, b)| a * b).sum();
                    scores[t] = dot * scale;
                    mx = mx.max(scores[t]);
                }
                let mut sum = 0f32;
                for s0 in scores.iter_mut() {
                    *s0 = (*s0 - mx).exp();
                    sum += *s0;
                }
                for (t, s0) in scores.iter().enumerate() {
                    let w = s0 / sum;
                    let vrow = &kv[il].1[t * kd + kh * hd..t * kd + (kh + 1) * hd];
                    for dd in 0..hd {
                        attn[hh * hd + dd] += w * vrow[dd];
                    }
                }
            }
            let o = matvec(&get(&p("attn_output.weight")), dim, qd, &attn);
            let o = norm(&o, &get(&p("post_attention_norm.weight")), cfg.eps);
            let x1: Vec<f32> = inp.iter().zip(&o).map(|(a, b)| a + b).collect();
            let f = norm(&x1, &get(&p("ffn_norm.weight")), cfg.eps);
            let up_all = matvec(&get(&p("ffn_up.weight")), 2 * cfg.ffn_dim, dim, &f);
            let (gate, up) = up_all.split_at(cfg.ffn_dim);
            let act: Vec<f32> = gate
                .iter()
                .zip(up)
                .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
                .collect();
            let down = matvec(&get(&p("ffn_down.weight")), dim, cfg.ffn_dim, &act);
            let down = norm(&down, &get(&p("post_ffw_norm.weight")), cfg.eps);
            x = x1.iter().zip(&down).map(|(a, b)| a + b).collect();
            if std::env::var("GLM4_DEBUG_L").is_ok() {
                eprintln!("HOSTL x[{step}][{il}]: {:?}", &x[..4]);
            }
        }
        let h = norm(&x, &out_norm_w, cfg.eps);
        let logits = matvec(&out_w, cfg.vocab_size, dim, &h);
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
        if step + 1 == ids.len() {
            eprintln!(
                "HOST argmax после промпта: {:?} (logit {:.3})",
                tok.decode_piece(idx[0] as u32),
                logits[idx[0]]
            );
            let top: Vec<(usize, f32)> = idx[..5].iter().map(|&i| (i, logits[i])).collect();
            eprintln!("HOST top5 логитов: {top:?}");
        }
        next = idx[0] as u32;
        pos += 1;
    }
    eprintln!("HOST текст: {text:?}");
    Ok(())
}
