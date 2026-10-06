//! Микробенч matvec-кернелов decode (Ф8.2): эффективная пропускная
//! способность весов (ГБ/с) для форм Qwen3-8B, против потолка памяти.
#![cfg(feature = "metal")]

use rust_inference::backend::Backend;
use rust_inference::dtype::WeightFormat;
use rust_inference::metal::MetalBackend;
use std::time::Instant;

fn q8_0_bytes(n: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n * k / 32 * 34);
    for _ in 0..n * k / 32 {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        out.extend_from_slice(&half::f16::from_f32(0.001 + (s % 1000) as f32 * 1e-5).to_le_bytes());
        for j in 0..32 {
            out.push(((s >> (j % 56)) as u8) ^ (j as u8));
        }
    }
    out
}

fn main() {
    let be = MetalBackend::new().unwrap();
    let iters = 200;
    for (name, n, k) in [("wqkv", 6144usize, 4096usize), ("w13", 24576, 4096), ("w2", 4096, 12288), ("lm_head", 151936, 4096)] {
        let w = be.from_quantized(&q8_0_bytes(n, k, 7), vec![n, k], WeightFormat::Q8_0).unwrap();
        let a = be.from_cpu_f32(&(0..k).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect::<Vec<_>>(), vec![1, k]).unwrap();
        let c = be.zeros(vec![1, n]).unwrap();
        be.begin_batch().unwrap();
        be.gemm_nt(&a, &w, &c).unwrap();
        be.end_batch().unwrap();
        let t0 = Instant::now();
        be.begin_batch().unwrap();
        for _ in 0..iters {
            be.gemm_nt(&a, &w, &c).unwrap();
        }
        be.end_batch().unwrap();
        let dt = t0.elapsed().as_secs_f64() / iters as f64;
        let bytes = (n * k / 32 * 34) as f64;
        println!("{name:8} [{n}x{k}] {:.3} ms/вызов  {:.0} ГБ/с", dt * 1e3, bytes / dt / 1e9);
    }
    // K-кванты (Q4_K/Q6_K): те же формы, что у плотных моделей и у
    // экспертов MoE. Сравнение с INFER_MV_KS_OFF=1 — «поток на строку».
    let q4k_bytes = |n: usize, k: usize, seed: u64| -> Vec<u8> {
        let mut s = seed;
        let mut out = Vec::with_capacity(n * k / 256 * 144);
        for _ in 0..n * k / 256 {
            for _ in 0..144 {
                s ^= s << 13; s ^= s >> 7; s ^= s << 17;
                out.push((s & 0xFF) as u8);
            }
        }
        out
    };
    let q6k_bytes = |n: usize, k: usize, seed: u64| -> Vec<u8> {
        let mut s = seed;
        let mut out = Vec::with_capacity(n * k / 256 * 210);
        for _ in 0..n * k / 256 {
            for _ in 0..210 {
                s ^= s << 13; s ^= s >> 7; s ^= s << 17;
                out.push((s & 0xFF) as u8);
            }
        }
        out
    };
    for (name, n, k, fmt, bytes) in [
        ("q4k_ffn", 13696usize, 4096usize, WeightFormat::Q4_K, q4k_bytes(13696, 4096, 11)),
        ("q4k_attn", 4096, 4096, WeightFormat::Q4_K, q4k_bytes(4096, 4096, 12)),
        ("q6k_out", 151936, 4096, WeightFormat::Q6_K, q6k_bytes(151936, 4096, 13)),
        ("q4k_exp", 1024, 2048, WeightFormat::Q4_K, q4k_bytes(1024, 2048, 14)),
        ("q6k_exp", 1024, 2048, WeightFormat::Q6_K, q6k_bytes(1024, 2048, 15)),
    ] {
        let w = be.from_quantized(&bytes, vec![n, k], fmt).unwrap();
        let a = be
            .from_cpu_f32(
                &(0..k).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect::<Vec<_>>(),
                vec![1, k],
            )
            .unwrap();
        let c = be.zeros(vec![1, n]).unwrap();
        be.begin_batch().unwrap();
        be.gemm_nt(&a, &w, &c).unwrap();
        be.end_batch().unwrap();
        let t0 = Instant::now();
        be.begin_batch().unwrap();
        for _ in 0..iters {
            be.gemm_nt(&a, &w, &c).unwrap();
        }
        be.end_batch().unwrap();
        let dt = t0.elapsed().as_secs_f64() / iters as f64;
        println!(
            "{name:8} [{n}x{k}] {:.3} ms/вызов  {:.0} ГБ/с",
            dt * 1e3,
            bytes.len() as f64 / dt / 1e9
        );
    }
    // Потолок оверхеда dispatch: пустой по сути кернел (add 16 элементов)
    // в одном батче — сколько стоит сам вызов.
    let x = be.zeros(vec![16]).unwrap();
    be.begin_batch().unwrap();
    be.add(&x, &x, &x).unwrap();
    be.end_batch().unwrap();
    let t0 = Instant::now();
    be.begin_batch().unwrap();
    for _ in 0..1000 {
        be.add(&x, &x, &x).unwrap();
    }
    be.end_batch().unwrap();
    println!("dispatch-оверхед: {:.1} мкс/кернел", t0.elapsed().as_secs_f64() * 1e6 / 1000.0);
}
