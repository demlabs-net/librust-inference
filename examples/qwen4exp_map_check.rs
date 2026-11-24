//! Сверка карты весов Flash-Next с контейнером NINFER.
//!
//! Читаются только заголовки частей: ни одного байта весов не тянется, поэтому
//! проверка мгновенная и не требует GPU.
//!
//! ```sh
//! cargo run --release --example qwen4exp_map_check -- /path/to/model.infernix
//! ```

use rust_inference::ninfer_layout::Variant;
use rust_inference::plugins::qwen4exp::{load::Weights, model};

fn main() {
    let path = match std::env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: qwen4exp_map_check <container.infernix>");
            std::process::exit(2);
        }
    };
    let w = match Weights::open(std::path::Path::new(&path), Variant::TileMajor, Variant::Identity) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("открыть: {e}");
            std::process::exit(1);
        }
    };
    println!("артефакт: {}", w.artifact());
    let cfg = match w.config() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("конфиг: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "конфиг: dim {} · {} слоёв ({} GDN / {} full) · MoE {}/top-{} · hc {}×{} · PLE {:?}",
        cfg.dim,
        cfg.n_layers,
        cfg.n_recurrent(),
        cfg.n_full(),
        cfg.n_experts,
        cfg.n_experts_used,
        cfg.hc_count,
        cfg.hc_lowrank,
        cfg.ple_layers
    );

    let problems = match model::validate(&w, &cfg) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("сверка: {e}");
            std::process::exit(1);
        }
    };
    println!("расхождений по карте: {}", problems.len());
    for p in problems.iter().take(25) {
        println!("  ✗ {p}");
    }

    // Банки экспертов: проверяем размеры по спецификации для каждого слоя.
    let mut bank_bad = 0;
    for layer in 0..cfg.n_layers {
        match w.expert_bank(layer) {
            Ok(b) => {
                if layer == 0 || layer == cfg.n_layers - 1 {
                    println!(
                        "  bank[{layer}]: {} экспертов, record {} → stride {}, всего {} МиБ",
                        b.experts,
                        b.record_bytes(),
                        b.record_stride(),
                        b.payload_bytes() / (1 << 20)
                    );
                }
            }
            Err(e) => {
                bank_bad += 1;
                if bank_bad <= 3 {
                    println!("  ✗ bank[{layer}]: {e}");
                }
            }
        }
    }
    println!("банков с ошибкой: {bank_bad} из {}", cfg.n_layers);
    if !problems.is_empty() || bank_bad > 0 {
        std::process::exit(1);
    }
    println!("карта сходится с контейнером");
}
