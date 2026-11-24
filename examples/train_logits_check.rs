//! Сверка лосса/логитов обучающего пути (`train_qwen3::forward_seq`) с
//! инференс-путём (`Qwen3Model`): расхождение = баг в тренировочном форварде
//! (после него LoRA учится не на том распределении, а perplexity врёт на
//! порядки). Печатает top-5 токенов обеих сторон и потери по одному и тому
//! же тексту.
//!
//! Запуск: INFER_MODELS_DIR=…/models cargo run --release --features metal \
//!   --example train_logits_check -- <model.gguf>
#![cfg(feature = "metal")]
use rust_inference::backend::Backend as _;
use rust_inference::metal::MetalBackend;
use rust_inference::qwen3::{Qwen3Config, Qwen3Model};
use rust_inference::tokenizer::Tokenizer;
use rust_inference::train_qwen3::{forward_seq, init_lora, load_base, param_states};
use rust_inference::weights_map::WeightsMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: train_logits_check <model.gguf>")?;
    let text = std::env::var("CHECK_TEXT").unwrap_or_else(|_| "Столица Франции — это".to_string());
    let bytes = std::fs::read(&path)?;
    let gf = rust_inference::gguf::parse(&bytes)?;
    let wm = WeightsMap::from_gguf(&gf)?;
    let tok = Tokenizer::from_gguf(&gf)?;
    let cfg = Qwen3Config::from_gguf(&gf, tok.vocab_size())?;
    let ids = tok.encode(&text);
    println!("текст: {text:?} → {} токенов", ids.len());
    let be = MetalBackend::new()?;

    // 2) обучающий путь при разных префиксах: сверка top-1 по позициям.
    let base = load_base(&be, &wm, &cfg)?;
    let states = param_states(&be, &init_lora(&be, &cfg, 8)?)?; // B = 0 → no-op
    let mut model2 = Qwen3Model::load(&be, &wm, cfg.clone())?;
    for n in 1..=ids.len() {
        // инференс: скармливаем токены по одному, берём логиты последней позиции.
        let lg = model2.decode(&be, ids[n - 1])?;
        let row_inf = be.to_cpu_f32(&lg)?;
        let labels = vec![-1i32; n];
        let fwd = forward_seq(&be, &base, &states, &ids[..n], &labels)?;
        let lg_tr = be.to_cpu_f32(&fwd.logits)?;
        let v = cfg.vocab_size;
        let row_tr = &lg_tr[(n - 1) * v..n * v];
        let t1_inf = argmax(&row_inf[..v]);
        let t1_tr = argmax(row_tr);
        let diff = row_inf[..v]
            .iter()
            .zip(row_tr)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        if n == 1 {
            println!("── постадийная сверка (t=1) ──");
            host_stages(&wm, &cfg, ids[0], &fwd);
        }
        println!(
            "n={n:2}: инференс {:>8?} | обучение {:>8?} | |Δ|max {diff:8.4} {}",
            tok.decode_piece(t1_inf as u32),
            tok.decode_piece(t1_tr as u32),
            if t1_inf == t1_tr {
                "OK"
            } else {
                "РАСХОЖДЕНИЕ"
            }
        );
    }
    Ok(())
}

// ── Постадийная сверка обучающего форварда с независимым host-эталоном ──────
// (t=1: внимание тривиально, rope в pos=0 — тождество; любой расхождение на
//  этих стадиях локализует баг в конкретном месте пайплайна).
fn host_stages(
    wm: &WeightsMap<'_>,
    cfg: &Qwen3Config,
    tok_id: u32,
    tr: &rust_inference::train_qwen3::FwdState,
) {
    use rust_inference::weights_map::dequant_f32;
    let dim = cfg.dim;
    let hd = cfg.head_dim;
    let nh = cfg.n_heads;
    let nkv = cfg.n_kv_heads;
    let eps = cfg.norm_eps;
    let get = |name: &str| -> Vec<f32> {
        let t = wm.get(name).unwrap_or_else(|| panic!("нет тензора {name}"));
        dequant_f32(t)
    };
    let norm = |x: &[f32], w: &[f32]| -> Vec<f32> {
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

    // эмбеддинг
    let emb_all = get("token_embd.weight");
    let emb = emb_all[tok_id as usize * dim..(tok_id as usize + 1) * dim].to_vec();
    let l0 = &tr.layers[0];
    let cmp = |tag: &str, a: &[f32], b: &[f32], n: usize| {
        let m = a.len().min(b.len()).min(n);
        let mut mx = 0f32;
        let mut bi = 0usize;
        for i in 0..m {
            let d = (a[i] - b[i]).abs();
            if d > mx {
                mx = d;
                bi = i;
            }
        }
        println!(
            "  {tag:22}: |Δ|max {mx:8.5} @{bi} (эталон {:.5}, обучающий {:.5})",
            a[bi], b[bi]
        );
        assert!(
            mx <= 1e-3,
            "{tag}: обучающий форвард расходится с эталоном (|Δ| {mx} @{bi}) — \
             проверь раскладку (GQA-группировка, нормы, rope)"
        );
    };
    cmp("embedding", &emb, &l0.x_h[..dim], 16);
    let x1 = norm(&emb, &get("blk.0.attn_norm.weight"));
    let q = matvec(&get("blk.0.attn_q.weight"), nh * hd, dim, &x1);
    cmp("q (до norm/rope)", &q, &l0.q_h[..nh * hd], 16);
    let k = matvec(&get("blk.0.attn_k.weight"), nkv * hd, dim, &x1);
    cmp("k (до norm/rope)", &k, &l0.k_h[..nkv * hd], 16);
    let v = matvec(&get("blk.0.attn_v.weight"), nkv * hd, dim, &x1);
    cmp("v", &v, &l0.v_h[..nkv * hd], 16);
    // q/k после per-head rmsnorm (rope при pos=0 — тождество)
    let qn_w = get("blk.0.attn_q_norm.weight");
    let kn_w = get("blk.0.attn_k_norm.weight");
    let mut q_n = vec![0f32; nh * hd];
    for h in 0..nh {
        let row = &q[h * hd..(h + 1) * hd];
        q_n[h * hd..(h + 1) * hd].copy_from_slice(&norm(row, &qn_w));
    }
    cmp("q_norm (rope pos=0)", &q_n, &l0.qr_h[..nh * hd], nh * hd);
    let mut k_n = vec![0f32; nkv * hd];
    for h in 0..nkv {
        let row = &k[h * hd..(h + 1) * hd];
        k_n[h * hd..(h + 1) * hd].copy_from_slice(&norm(row, &kn_w));
    }
    cmp("k_norm (rope pos=0)", &k_n, &l0.kr_h[..nkv * hd], nkv * hd);
    // выход внимания при t=1 = v, разложенный по головам запроса (o_h — ДО W_o)
    let mut av = vec![0f32; nh * hd];
    for h in 0..nh {
        // GQA-группировка h / (nh/nkv) — как в движке; `h % nkv` (чередование)
        // читал бы ЧУЖИЕ k/v (именно этот баг был в обучающем форварде).
        let kh = h / (nh / nkv);
        av[h * hd..(h + 1) * hd].copy_from_slice(&v[kh * hd..(kh + 1) * hd]);
    }
    cmp("attn out (t=1 = v)", &av, &l0.o_h, nh * hd);
    let o = matvec(&get("blk.0.attn_output.weight"), dim, nh * hd, &av);
    let x2: Vec<f32> = emb.iter().zip(&o).map(|(a, b)| a + b).collect();
    // x2_h в LayerCache — ВЫХОД ffn_norm (не сырой residual): сверяем
    // нормированный вход FFN (однозначная величина).
    let x3 = norm(&x2, &get("blk.0.ffn_norm.weight"));
    cmp("ffn_norm (FFN-вход)", &x3, &l0.x2_h[..dim], dim);
    let gate = matvec(&get("blk.0.ffn_gate.weight"), cfg.ffn_dim, dim, &x3);
    cmp("gate", &gate, &l0.gate_h, cfg.ffn_dim);
    let up = matvec(&get("blk.0.ffn_up.weight"), cfg.ffn_dim, dim, &x3);
    cmp("up", &up, &l0.up_h, cfg.ffn_dim);
    let act: Vec<f32> = gate
        .iter()
        .map(|g| {
            let s = 1.0 / (1.0 + (-g).exp());
            g * s
        })
        .collect();
    cmp("silu(gate)", &act, &l0.act_h, cfg.ffn_dim);
}

fn argmax(row: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &x) in row.iter().enumerate() {
        if x > row[best] {
            best = i;
        }
    }
    best
}
