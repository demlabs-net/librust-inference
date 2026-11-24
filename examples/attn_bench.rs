//! Время декод-внимания по геометриям: hd=256 (Qwen3.5/3.8 27B, Flash-Next) и
//! hd=128 (Qwen3) при разных Tkv. Отвечает на вопрос, во что обходится
//! внимание на длинном контексте и как оно масштабируется.
//!
//! `cargo run --release --features metal --example attn_bench -- [reps]`
#![cfg(feature = "metal")]

use std::time::Instant;

use rust_inference::backend::Backend as _;
use rust_inference::metal::MetalBackend;

fn bench(be: &MetalBackend, nh: usize, nkv: usize, hd: usize, tkv: usize, reps: usize) {
    let q_data: Vec<f32> = (0..nh * hd)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
        .collect();
    let q = be.from_cpu_f32(&q_data, vec![1, 1, nh * hd]).unwrap();
    let kv_n = tkv * nkv * hd;
    let k_data: Vec<half::f16> = (0..kv_n)
        .map(|i| half::f16::from_f32(((i % 13) as f32 - 6.0) * 0.02))
        .collect();
    let v_data: Vec<half::f16> = (0..kv_n)
        .map(|i| half::f16::from_f32(((i % 11) as f32 - 5.0) * 0.03))
        .collect();
    let k = be.from_cpu_f16(&k_data, vec![1, tkv, nkv * hd]).unwrap();
    let v = be.from_cpu_f16(&v_data, vec![1, tkv, nkv * hd]).unwrap();
    let y = be.zeros(vec![1, 1, nh * hd]).unwrap();
    // прогрев (компиляция пайплайна)
    be.attn_ar(&q, &k, &v, None, &y, nh, nkv, hd).unwrap();
    let _ = be.to_cpu_f32(&y.view_restricted(vec![1])).unwrap();
    let t0 = Instant::now();
    be.begin_batch().unwrap();
    for _ in 0..reps {
        be.attn_ar(&q, &k, &v, None, &y, nh, nkv, hd).unwrap();
    }
    be.end_batch().unwrap();
    let dt = t0.elapsed().as_secs_f64() / reps as f64;
    let kv_bytes = 2.0 * kv_n as f64 * 2.0; // K+V, f16
    println!(
        "nh={nh} nkv={nkv} hd={hd} tkv={tkv}: {:8.3} мс  {:6.1} ГБ/с KV  {:6.2} Гфлоп",
        dt * 1e3,
        kv_bytes / dt / 1e9,
        2.0 * (nh as f64) * (tkv as f64) * (hd as f64) / dt / 1e9
    );
}

fn main() {
    let reps: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let be = MetalBackend::new().expect("MetalBackend");
    // 27B-гибрид: 24 головы, 4 kv, hd 256; Qwen3-8B: 32/8 hd 128.
    for (nh, nkv, hd) in [(24usize, 4usize, 256usize), (32, 8, 128)] {
        for tkv in [128usize, 512, 1440, 4096] {
            bench(&be, nh, nkv, hd, tkv, reps);
        }
    }
}
