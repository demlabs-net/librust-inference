//! Приёмка пакетного шага (батч-слоты, шаг 2).
//!
//! Три беседы префиллятся в СВОИ слоты, дальше по одному токену на слот за
//! шаг: строки всех слотов идут одним matmul'ом (веса читаются раз). Проверка:
//! жадные токены каждого слота шаг-в-шаг равны одиночным прогонам той же
//! беседы. Побитового равенства логитов тут не требуем — m=1 и m=B идут
//! разными кернелами (mv против mb/mm), норма батча та же, что у llama.cpp -np.
//!
//! `cargo run --release --features metal --example batch_step -- <model.gguf> [N]`

#![cfg(feature = "metal")]

use std::error::Error;
use std::time::Instant;

use rust_inference::backend::Backend as _;
use rust_inference::gguf;
use rust_inference::metal::MetalBackend;
use rust_inference::qwen35;
use rust_inference::tokenizer::Tokenizer;
use rust_inference::weights_map::WeightsMap;

fn argmax(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Жадный токен из логитов [.., V] (берём последнюю строку).
fn pick(
    be: &MetalBackend,
    vocab: usize,
    logits: &rust_inference::Tensor,
) -> Result<u32, Box<dyn Error>> {
    let v = be.to_cpu_f32(logits)?;
    let last = &v[v.len() - vocab..];
    Ok(argmax(last))
}

struct Outcome {
    /// Токены: [0] — из prefill, дальше по шагу декода (N+1 штук).
    tokens: Vec<Vec<u32>>,
    /// Логиты первого декод-шага (сверка расхождения с пакетом).
    first_logits: Vec<Vec<f32>>,
    secs: f64,
    /// Время каждого декод-шага, мс.
    steps_ms: Vec<f64>,
}

fn run_solo(
    m: &mut qwen35::Qwen35Model,
    be: &MetalBackend,
    tok: &Tokenizer,
    prompts: &[&str],
    n: usize,
) -> Result<Outcome, Box<dyn Error>> {
    let vocab = tok.vocab_size();
    let mut tokens = Vec::new();
    let mut first_logits = Vec::new();
    let mut steps_ms: Vec<f64> = Vec::new();
    let t0 = Instant::now();
    for p in prompts {
        let ids = tok.encode(p);
        let mut slot = m.new_slot(be)?;
        let mut lg = m.prefill_slot(be, &mut slot, &ids)?;
        let mut toks = vec![pick(be, vocab, &lg)?];
        for _ in 0..n {
            let ts = Instant::now();
            lg = m.decode_slot(be, &mut slot, *toks.last().unwrap())?;
            if toks.len() == 1 {
                first_logits.push(be.to_cpu_f32(&lg)?);
            }
            toks.push(pick(be, vocab, &lg)?);
            steps_ms.push(ts.elapsed().as_secs_f64() * 1e3);
        }
        tokens.push(toks);
    }
    Ok(Outcome {
        tokens,
        first_logits,
        secs: t0.elapsed().as_secs_f64(),
        steps_ms,
    })
}

fn run_batch(
    m: &mut qwen35::Qwen35Model,
    be: &MetalBackend,
    tok: &Tokenizer,
    prompts: &[&str],
    n: usize,
) -> Result<Outcome, Box<dyn Error>> {
    let vocab = tok.vocab_size();
    let b = prompts.len();
    let mut slots = Vec::with_capacity(b);
    let mut next = Vec::with_capacity(b);
    let mut tokens: Vec<Vec<u32>> = vec![Vec::new(); b];
    let mut first_logits = Vec::new();
    let mut steps_ms: Vec<f64> = Vec::new();
    let t0 = Instant::now();
    for prompt in prompts {
        let ids = tok.encode(prompt);
        let mut slot = m.new_slot(be)?;
        let lg = m.prefill_slot(be, &mut slot, &ids)?;
        next.push(pick(be, vocab, &lg)?);
        slots.push(slot);
    }
    for _ in 0..n {
        let ts = Instant::now();
        let lg = m.decode_batch(be, &mut slots, &next)?;
        let host = be.to_cpu_f32(&lg)?;
        steps_ms.push(ts.elapsed().as_secs_f64() * 1e3);
        for p in 0..b {
            let row = &host[p * vocab..(p + 1) * vocab];
            let id = argmax(row);
            if first_logits.len() < b {
                first_logits.push(row.to_vec());
            }
            tokens[p].push(id);
            next[p] = id;
        }
    }
    Ok(Outcome {
        tokens,
        first_logits,
        secs: t0.elapsed().as_secs_f64(),
        steps_ms,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).ok_or("usage: batch_step <model.gguf> [N]")?;
    let n: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(8);
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let gf = gguf::parse(&mmap)?;
    let wm = WeightsMap::from_gguf(&gf)?;
    let tok = Tokenizer::from_gguf(&gf)?;
    let be = MetalBackend::new()?;
    let vocab = tok.vocab_size();
    let mut m = qwen35::load_gguf(&be, &gf, &wm, vocab)?;

    let all_prompts = [
        "The capital of France is Paris and the capital of Italy is",
        "2 + 2 = 4 and 3 + 3 = 6 and 4 + 4 =",
        "Once upon a time, in a small village by the sea, there lived a",
        "Water boils at 100 degrees Celsius and freezes at",
        "The quick brown fox jumps over the lazy dog. The dog was not amused, so the fox",
        "In 1969, humans first walked on the Moon during the Apollo",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
        "The mitochondria is the powerhouse of the cell because it produces",
    ];
    // BATCH_B=N — первые N бесед (1,2,3,4).
    let b: usize = std::env::var("BATCH_B")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let prompts = &all_prompts[..b.min(all_prompts.len())];

    // Первый проход — правильность (он же прогрев), второй — время.
    eprintln!("phase: solo");
    let solo1 = run_solo(&mut m, &be, &tok, prompts, n)?;
    eprintln!("phase: batch");
    let batch1 = run_batch(&mut m, &be, &tok, prompts, n)?;
    eprintln!("phase: checks");
    assert_eq!(
        solo1.tokens.len(),
        prompts.len(),
        "число бесед в одиночном прогоне"
    );

    let mut ok = true;
    for (p, prompt) in prompts.iter().enumerate() {
        let solo = &solo1.tokens[p];
        let bat = &batch1.tokens[p];
        let mut first_diff = None;
        for s in 0..bat.len() {
            if bat[s] != solo[s + 1] {
                first_diff = Some(s);
                break;
            }
        }
        let drift = solo1.first_logits[p]
            .iter()
            .zip(&batch1.first_logits[p])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let same_argmax = argmax(&solo1.first_logits[p]) == argmax(&batch1.first_logits[p]);
        println!(
            "беседа {p} (\"{}…\", {} ток.): {}; max|Δлогит| = {drift:.3e}, argmax {}{}",
            &prompt[..prompt.len().min(28)],
            tok.encode(prompt).len(),
            match first_diff {
                None => format!("все {n} шагов совпали"),
                Some(s) => format!("РАСХОЖДЕНИЕ на шаге {s}"),
            },
            if same_argmax {
                "совпал"
            } else {
                "разошёлся"
            },
            if drift.is_finite() { "" } else { " (nan!)" },
        );
        ok &= first_diff.is_none();
    }

    // Второй проход: время (первый прогрел буферы/mmap). ORDER=batch_first —
    // обратный порядок (проверка зависимости от прогрева/троттлинга).
    be.ctx.ops_snapshot(true);
    let batch_first = std::env::var("ORDER")
        .map(|v| v == "batch_first")
        .unwrap_or(false);
    let (solo2, batch2, solo_ops, solo_ns, batch_ops, batch_ns) = if batch_first {
        let b2 = run_batch(&mut m, &be, &tok, prompts, n)?;
        let (bops, bns) = be.ctx.ops_snapshot(true);
        let s2 = run_solo(&mut m, &be, &tok, prompts, n)?;
        let (sops, sns) = be.ctx.ops_snapshot(true);
        (s2, b2, sops, sns, bops, bns)
    } else {
        let s2 = run_solo(&mut m, &be, &tok, prompts, n)?;
        let (sops, sns) = be.ctx.ops_snapshot(true);
        let b2 = run_batch(&mut m, &be, &tok, prompts, n)?;
        let (bops, bns) = be.ctx.ops_snapshot(true);
        (s2, b2, sops, sns, bops, bns)
    };
    eprintln!(
        "ops: одиночно {solo_ops} dispatch'ей ({:.1} ms хоста), пакетом {batch_ops} ({:.1} ms хоста)",
        solo_ns as f64 / 1e6,
        batch_ns as f64 / 1e6
    );
    let fmt = |v: &[f64]| -> String {
        v.iter()
            .map(|x| format!("{x:.0}"))
            .collect::<Vec<_>>()
            .join(" ")
    };
    eprintln!("шаги solo, мс:  {}", fmt(&solo2.steps_ms));
    eprintln!("шаги batch, мс: {}", fmt(&batch2.steps_ms));
    assert_eq!(solo2.tokens, solo1.tokens, "одиночный прогон нестабилен");
    assert_eq!(batch2.tokens, batch1.tokens, "пакетный прогон нестабилен");

    let solo_tps = (b * n) as f64 / solo2.secs;
    let batch_tps = (b * n) as f64 / batch2.secs;
    println!(
        "\nN={n} на беседу, B={b}: одиночно {solo_tps:.2} tok/s суммарно ({:.1} мс/шаг), пакетом {batch_tps:.2} tok/s ({:.1} мс/шаг на пакет), ×{:.2}",
        solo2.secs * 1e3 / (b * n) as f64,
        batch2.secs * 1e3 / n as f64,
        batch_tps / solo_tps,
    );
    // GPU-пол: один внешний батч на серию шагов — хост не участвует в ожидании,
    // время/шаг отражает работу GPU. Разделяет «медленный код» и «хост».
    if std::env::var("LOOP_STEPS").is_ok() {
        use rust_inference::backend::Backend;
        let nl = 16usize;
        let ids = tok.encode(prompts[0]);
        let mut slot = m.new_slot(&be)?;
        m.prefill_slot(&be, &mut slot, &ids)?;
        be.begin_batch()?;
        let t = Instant::now();
        for _ in 0..nl {
            let _ = m.decode_slot(&be, &mut slot, 13)?;
        }
        let solo_enc = t.elapsed().as_secs_f64() * 1e3 / nl as f64;
        be.end_batch()?;
        let solo_loop = t.elapsed().as_secs_f64() * 1e3 / nl as f64;
        let mut slots = vec![m.new_slot(&be)?];
        let ids = tok.encode(prompts[0]);
        m.prefill_slot(&be, &mut slots[0], &ids)?;
        be.begin_batch()?;
        let t = Instant::now();
        for _ in 0..nl {
            let _ = m.decode_batch(&be, &mut slots, &[13])?;
        }
        let batch_enc = t.elapsed().as_secs_f64() * 1e3 / nl as f64;
        be.end_batch()?;
        let batch_loop = t.elapsed().as_secs_f64() * 1e3 / nl as f64;
        println!(
            "GPU-пол: одиночно {solo_loop:.1} мс/шаг (кодирование хоста {solo_enc:.1}), пакетом {batch_loop:.1} (кодирование {batch_enc:.1})"
        );
    }

    assert!(ok, "пакет разошёлся с одиночными прогонами");
    println!("OK: пакетный шаг даёт тексты своих одиночных прогонов");
    Ok(())
}
