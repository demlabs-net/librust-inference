//! Q8_0 GEMM при малом M (окно верификации спека: 1+γ строк) против
//! matvec декода на формах Qwen3-8B. Матричный GEMM спроектирован под
//! префилл (тайл 32 строки): при M ≤ 8 он читает веса так же, как при 32,
//! но считает 4–32× лишнего, и forward 2..8 строк стоит ≈3× декода — спек
//! с любым драфтером медленнее обычной генерации. Цель — время M строк
//! ≈ времени одной (веса читаются один раз, как в kernel_mul_mv_ext ggml).
#![cfg(feature = "metal")]

use rust_inference::backend::Backend;
use rust_inference::dtype::WeightFormat;
use rust_inference::metal::MetalBackend;
use std::time::Instant;

fn q8_0_bytes(n: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n * k / 32 * 34);
    for _ in 0..n * k / 32 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        out.extend_from_slice(&half::f16::from_f32(0.001 + (s % 1000) as f32 * 1e-5).to_le_bytes());
        for j in 0..32 {
            out.push(((s >> (j % 56)) as u8) ^ (j as u8));
        }
    }
    out
}

fn main() {
    let be = MetalBackend::new().unwrap();
    let iters = 50;
    // GEMM_MS=2,5 — только эти M (для перебора конфигураций)
    let ms: Vec<usize> = std::env::var("GEMM_MS")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 2, 3, 4, 5, 8, 16, 32]);
    let mut total = vec![0f64; ms.len()];
    for (name, n, k, mult) in [
        ("wqkv", 6144usize, 4096usize, 36.0f64),
        ("wo", 4096, 4096, 36.0),
        ("w13", 24576, 4096, 36.0),
        ("w2", 4096, 12288, 36.0),
        ("lm_head", 151936, 4096, 1.0),
    ] {
        let w = be.from_quantized(&q8_0_bytes(n, k, 7), vec![n, k], WeightFormat::Q8_0).unwrap();
        let mut line = format!("{name:8} [{n}x{k}]");
        let mut base = 0.0;
        for (mi, &m) in ms.iter().enumerate() {
            let a = be
                .from_cpu_f32(&(0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect::<Vec<_>>(), vec![m, k])
                .unwrap();
            let c = be.zeros(vec![m, n]).unwrap();
            be.begin_batch().unwrap();
            be.gemm_nt(&a, &w, &c).unwrap();
            be.end_batch().unwrap();
            let t0 = Instant::now();
            be.begin_batch().unwrap();
            for _ in 0..iters {
                be.gemm_nt(&a, &w, &c).unwrap();
            }
            be.end_batch().unwrap();
            let dt = t0.elapsed().as_secs_f64() / iters as f64 * 1e3;
            if mi == 0 {
                base = dt;
            }
            total[mi] += dt * mult;
            line.push_str(&format!("  M={m}: {dt:.3} ({:.1}×)", dt / base));
        }
        println!("{line}");
    }
    // Сумма матмулов forward'а 8B (36 слоёв + lm_head), мс
    let tl: Vec<String> = ms.iter().zip(&total).map(|(m, t)| format!("M={m}: {t:.1}")).collect();
    println!("forward-матмулы 8B: {}", tl.join("  "));
}
