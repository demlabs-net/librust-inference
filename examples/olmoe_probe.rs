//! Host-эталон OLMoE (все слои, f32) — независимая реализация того же графа,
//! что и в движке (qwen3.rs с MoE-ветвью). Нужен для постадийной сверки:
//! HOST-теги должны совпасть с MOE_DEBUG-тегами движка.
//!
//! Запуск: olmoe_probe <model.gguf> "<промпт>" [слой]
//! Env: MOE_PROBE_L (слой для дампов, по умолчанию 0), MOE_PROBE_TOPK (сколько
//! верхних токенов логитов печатать, по умолчанию 5).
use rust_inference::tokenizer::Tokenizer;
use rust_inference::weights_map::{WeightsMap, dequant_f32};

#[derive(Clone, Copy)]
struct Cfg {
    dim: usize,
    n_layers: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    ffn: usize,
    n_expert: usize,
    n_used: usize,
    eps: f32,
    theta: f32,
    vocab: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let bytes = std::fs::read(&args[1])?;
    let gf = rust_inference::gguf::parse(&bytes)?;
    let wm = WeightsMap::from_gguf(&gf)?;
    let tok = Tokenizer::from_gguf(&gf)?;
    let u = |k: &str| -> usize {
        gf.get(k).and_then(|v| v.as_u32()).unwrap_or(0) as usize
    };
    let f = |k: &str, d: f32| -> f32 {
        gf.get(k)
            .and_then(|v| match v {
                rust_inference::gguf::GgufValue::F32(x) => Some(*x),
                rust_inference::gguf::GgufValue::F64(x) => Some(*x as f32),
                _ => None,
            })
            .unwrap_or(d)
    };
    let cfg = Cfg {
        dim: u("olmoe.embedding_length"),
        n_layers: u("olmoe.block_count"),
        nh: u("olmoe.attention.head_count"),
        nkv: u("olmoe.attention.head_count_kv"),
        hd: u("olmoe.embedding_length") / u("olmoe.attention.head_count").max(1),
        ffn: u("olmoe.feed_forward_length"),
        n_expert: u("olmoe.expert_count"),
        n_used: u("olmoe.expert_used_count"),
        eps: f("olmoe.attention.layer_norm_rms_epsilon", 1e-5),
        theta: f("olmoe.rope.freq_base", 10_000.0),
        vocab: tok.vocab_size(),
    };
    let probe_l: usize = std::env::var("MOE_PROBE_L")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let topk: usize = std::env::var("MOE_PROBE_TOPK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let prompt = args.get(2).map(|s| s.as_str()).unwrap_or("Hello");
    let ids = tok.encode(prompt);
    eprintln!(
        "cfg: dim={} layers={} heads={}/{} hd={} ffn={} E={} k={} eps={}",
        cfg.dim, cfg.n_layers, cfg.nh, cfg.nkv, cfg.hd, cfg.ffn, cfg.n_expert, cfg.n_used, cfg.eps
    );
    eprintln!("ids: {ids:?}");

    let get = |name: &str| -> Vec<f32> {
        let t = wm
            .get(name)
            .unwrap_or_else(|| panic!("нет тензора {name}"));
        dequant_f32(t)
    };
    let rmsnorm = |x: &[f32], w: &[f32], eps: f32| -> Vec<f32> {
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

    // RoPE (halves, NEOX/HF rotate_half), как в движке: пары (i, i+hd/2).
    let rope = |x: &mut [f32], pos: usize, hd: usize, theta: f32| {
        for h in x.chunks_mut(hd) {
            for i in 0..hd / 2 {
                let (a, b) = (h[i], h[i + hd / 2]);
                let ang = pos as f32 / theta.powf(2.0 * i as f32 / hd as f32);
                let (s, c) = ang.sin_cos();
                h[i] = a * c - b * s;
                h[i + hd / 2] = a * s + b * c;
            }
        }
    };

    let t = ids.len();
    let d = cfg.dim;
    let emb_all = get("token_embd.weight");
    let mut x: Vec<f32> = Vec::with_capacity(t * d);
    for &id in &ids {
        let o = id as usize * d;
        x.extend_from_slice(&emb_all[o..o + d]);
    }
    if probe_l == 0 {
        show("embed_last", &x[(t - 1) * d..t * d]);
    }

    for il in 0..cfg.n_layers {
        let p = |s: &str| format!("blk.{il}.{s}");
        let qw = get(&p("attn_q.weight"));
        let kw = get(&p("attn_k.weight"));
        let vw = get(&p("attn_v.weight"));
        let ow = get(&p("attn_output.weight"));
        let qn = get(&p("attn_q_norm.weight"));
        let kn = get(&p("attn_k_norm.weight"));
        let an = get(&p("attn_norm.weight"));
        let ffn_n = get(&p("ffn_norm.weight"));
        let gate_inp = get(&p("ffn_gate_inp.weight"));
        let up = get(&p("ffn_up_exps.weight"));
        let gate = get(&p("ffn_gate_exps.weight"));
        let down = get(&p("ffn_down_exps.weight"));

        let (qd, kd) = (cfg.nh * cfg.hd, cfg.nkv * cfg.hd);
        let mut qq: Vec<Vec<f32>> = Vec::with_capacity(t);
        let mut kk: Vec<Vec<f32>> = Vec::with_capacity(t);
        let mut vv: Vec<Vec<f32>> = Vec::with_capacity(t);
        for ti in 0..t {
            let h = rmsnorm(&x[ti * d..(ti + 1) * d], &an, cfg.eps);
            if il == probe_l && ti == t - 1 {
                show("attn_norm_last", &h);
            }
            let mut q = matvec(&qw, qd, d, &h);
            let mut k = matvec(&kw, kd, d, &h);
            let v = matvec(&vw, kd, d, &h);
            // OLMoE: QK-нормы по ВСЕЙ проекции (не по голове)
            q = rmsnorm(&q, &qn, cfg.eps);
            k = rmsnorm(&k, &kn, cfg.eps);
            if il == probe_l && ti == t - 1 {
                show("q_normed_last", &q);
            }
            rope(&mut q, ti, cfg.hd, cfg.theta);
            rope(&mut k, ti, cfg.hd, cfg.theta);
            qq.push(q);
            kk.push(k);
            vv.push(v);
        }
        // Причинное внимание по головам (kv-головы повторяются).
        let scale = 1.0 / (cfg.hd as f32).sqrt();
        let rep = cfg.nh / cfg.nkv.max(1);
        let mut attn_out: Vec<f32> = Vec::with_capacity(t * d);
        for ti in 0..t {
            let mut row = vec![0f32; qd];
            for h in 0..cfg.nh {
                let kh = h / rep.max(1);
                let qh = &qq[ti][h * cfg.hd..(h + 1) * cfg.hd];
                let mut scores = Vec::with_capacity(ti + 1);
                for tj in 0..=ti {
                    let kj = &kk[tj][kh * cfg.hd..(kh + 1) * cfg.hd];
                    let dot: f32 = qh.iter().zip(kj).map(|(a, b)| a * b).sum();
                    scores.push(dot * scale);
                }
                let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0f32;
                for s in scores.iter_mut() {
                    *s = (*s - m).exp();
                    sum += *s;
                }
                for (tj, s) in scores.iter().enumerate() {
                    let w = s / sum;
                    let vj = &vv[tj][kh * cfg.hd..(kh + 1) * cfg.hd];
                    for i in 0..cfg.hd {
                        row[h * cfg.hd + i] += w * vj[i];
                    }
                }
            }
            attn_out.extend_from_slice(&row);
        }
        // wo + residual
        for ti in 0..t {
            let o = matvec(&ow, d, qd, &attn_out[ti * qd..(ti + 1) * qd]);
            for i in 0..d {
                x[ti * d + i] += o[i];
            }
        }
        if il == probe_l {
            show("after_attn_last", &x[(t - 1) * d..t * d]);
        }

        // MoE
        let mut routed: Vec<(Vec<u32>, Vec<f32>)> = Vec::with_capacity(t);
        for ti in 0..t {
            let h = rmsnorm(&x[ti * d..(ti + 1) * d], &ffn_n, cfg.eps);
            let logits = matvec(&gate_inp, cfg.n_expert, d, &h);
            let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|v| (v - m).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();
            let mut idx: Vec<u32> = (0..cfg.n_expert as u32).collect();
            idx.sort_by(|&a, &b| {
                probs[b as usize]
                    .total_cmp(&probs[a as usize])
                    .then(a.cmp(&b))
            });
            idx.truncate(cfg.n_used);
            let ws: Vec<f32> = idx.iter().map(|&i| probs[i as usize]).collect();
            if il == probe_l && ti == t - 1 {
                eprintln!("HOST router_last ids={idx:?} w={ws:?}");
                show("ffn_norm_last", &h);
            }
            routed.push((idx, ws));
        }
        for ti in 0..t {
            let h = rmsnorm(&x[ti * d..(ti + 1) * d], &ffn_n, cfg.eps);
            let (idx, ws) = &routed[ti];
            let mut acc = vec![0f32; d];
            for (&e, &w) in idx.iter().zip(ws) {
                let e = e as usize;
                let up_e = &up[e * cfg.ffn * d..(e + 1) * cfg.ffn * d];
                let gate_e = &gate[e * cfg.ffn * d..(e + 1) * cfg.ffn * d];
                let down_e = &down[e * d * cfg.ffn..(e + 1) * d * cfg.ffn];
                let u = matvec(up_e, cfg.ffn, d, &h);
                let g = matvec(gate_e, cfg.ffn, d, &h);
                let act: Vec<f32> = g
                    .iter()
                    .zip(&u)
                    .map(|(gv, uv)| (gv / (1.0 + (-gv).exp())) * uv)
                    .collect();
                let o = matvec(down_e, d, cfg.ffn, &act);
                for i in 0..d {
                    acc[i] += w * o[i];
                }
            }
            if il == probe_l && ti == t - 1 {
                show("moe_out_last", &acc);
            }
            for i in 0..d {
                x[ti * d + i] += acc[i];
            }
        }
        if il == probe_l {
            show("after_ffn_last", &x[(t - 1) * d..t * d]);
        }
    }

    // Финальная норма + логиты
    let on = get("output_norm.weight");
    let ow = get("output.weight");
    let h = rmsnorm(&x[(t - 1) * d..t * d], &on, cfg.eps);
    let logits = matvec(&ow, cfg.vocab, d, &h);
    let mut idx: Vec<u32> = (0..cfg.vocab as u32).collect();
    idx.sort_by(|&a, &b| {
        logits[b as usize]
            .total_cmp(&logits[a as usize])
            .then(a.cmp(&b))
    });
    for &i in idx.iter().take(topk) {
        eprintln!(
            "HOST top: {i} {:?} {:.4}",
            String::from_utf8_lossy(&tok.piece(i)),
            logits[i as usize]
        );
    }
    Ok(())
}
