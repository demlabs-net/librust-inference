//! Сверка схемы Flash-Next с GGUF-файлом (вслепую, без весов).
//!
//! `cargo run --release --example qwen4exp_check -- <file.gguf>`
//!
//! Печатает сводку конфига и расхождения плана тензоров с файлом. Требует
//! ПОЛНЫЙ файл (парсер держит срезы данных); для проверки одного заголовка
//! используйте `scripts/gguf_meta.py`.
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: qwen4exp_check <file.gguf>")?;
    let file = std::fs::File::open(&path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let gf = rust_inference::gguf::parse(&mmap)?;
    let arch = gf.architecture().unwrap_or("").to_string();
    if !rust_inference::qwen4exp::Qwen4ExpConfig::is_arch(&arch) {
        println!("архитектура {arch:?} — не qwen4exp");
        return Ok(());
    }
    let vocab = gf
        .get("tokenizer.ggml.tokens")
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    let cfg = rust_inference::qwen4exp::Qwen4ExpConfig::from_gguf(&gf, vocab)
        .map_err(|e| format!("конфиг: {e}"))?;
    println!(
        "qwen4exp: dim={} слоёв {} (GDN {}, attention {}), головы {}/{}, hd={}, n_rot={}",
        cfg.dim,
        cfg.n_layers,
        cfg.n_recurrent(),
        cfg.n_full(),
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
        cfg.n_rot
    );
    println!(
        "MoE: {} экспертов top-{}, ff {}; shared ff {}; hc {}×dim, low-rank {}",
        cfg.n_experts,
        cfg.n_experts_used,
        cfg.expert_ff,
        cfg.shared_ff,
        cfg.hc_count,
        cfg.hc_lowrank
    );
    println!(
        "GDN: conv {} state {} k/v головы {}/{} inner {}; QSA top-k {}; PLE слои {:?} (n-грамма {})",
        cfg.conv_kernel,
        cfg.ssm_state,
        cfg.ssm_k_heads,
        cfg.ssm_v_heads,
        cfg.ssm_inner,
        cfg.indexer_top_k,
        cfg.ple_layers,
        cfg.ngram_size
    );
    let plan = cfg.tensor_plan(vocab);
    println!(
        "план: {} тензоров, в файле {}",
        plan.len(),
        gf.tensors.len()
    );
    let problems = rust_inference::qwen4exp::check_file(&cfg, &gf, vocab);
    if problems.is_empty() {
        println!("СХЕМА СОВПАЛА");
    } else {
        println!("расхождений {}:", problems.len());
        for p in problems.iter().take(20) {
            println!("  {p}");
        }
    }
    Ok(())
}
