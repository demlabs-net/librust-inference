//! Квантование GGUF (Ф6.3): движковый `quantize_gguf` без сервиса и панели —
//! для скриптов приёмки (сверка с `llama-quantize`) и ручных прогонов.
//!
//! `gguf_quantize <src.gguf> <dst.gguf> <FMT>` где FMT ∈ F32/F16/BF16/Q4_0/
//! Q8_0/Q4_K_M/Q6_K. Печатает отчёт JSON-строкой (единица для скриптов) и
//! строку прогресса; `gguf_quantize` использует ту же политику типов, что
//! сервис (нормы/эмбеддинги/слои — как llama-quantize).
use rust_inference::quant::{QuantFormat, quantize_gguf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: gguf_quantize <src.gguf> <dst.gguf> <FMT>");
        std::process::exit(2);
    }
    let (src, dst) = (args[1].clone(), args[2].clone());
    let fmt = QuantFormat::parse(&args[3]).ok_or_else(|| {
        format!(
            "неизвестный формат {} (F32/F16/BF16/Q4_0/Q8_0/Q4_K_M/Q6_K)",
            args[3]
        )
    })?;

    let t0 = std::time::Instant::now();
    let mut last_line = std::time::Instant::now();
    let mut progress = |done: u64, total: u64| {
        if last_line.elapsed().as_millis() >= 500 {
            last_line = std::time::Instant::now();
            let pct = if total > 0 {
                done as f64 * 100.0 / total as f64
            } else {
                0.0
            };
            eprintln!("  {pct:5.1}%  {:.2} ГБ", done as f64 / (1 << 30) as f64);
        }
    };
    let rep = quantize_gguf(
        std::path::Path::new(&src),
        std::path::Path::new(&dst),
        fmt,
        &mut progress,
        None,
    )?;
    let mut hist: Vec<String> = rep
        .by_type
        .iter()
        .map(|(t, n, b)| format!("\"{t}\":{{\"tensors\":{n},\"bytes\":{b}}}"))
        .collect();
    hist.sort();
    println!(
        "{{\"src\":\"{src}\",\"dst\":\"{dst}\",\"format\":\"{}\",\"arch\":\"{}\",\"tensors\":{},\
         \"src_bytes\":{},\"bytes\":{},\"secs\":{:.1},\"by_type\":{{{}}},\"warnings\":[{}]}}",
        rep.format,
        rep.arch,
        rep.tensors,
        rep.src_bytes,
        rep.bytes,
        t0.elapsed().as_secs_f64(),
        hist.join(","),
        rep.warnings
            .iter()
            .map(|w| format!("\"{}\"", w.replace('"', "'")))
            .collect::<Vec<_>>()
            .join(",")
    );
    Ok(())
}
