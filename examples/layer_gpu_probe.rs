//! GPU-время каждой операции слоя Qwen3 (Ф8.2) при заданных формах: каждая
//! операция повторяется подряд в ОДНОМ батче — CPU-кодирование перекрывается
//! исполнением, замер ≈ чистое GPU-время вызова. Сумма по слою × число слоёв
//! сравнивается с реальным prefill/decode — остаток = накладные расходы вне
//! кернелов.
//!
//! `layer_gpu_probe [dim ffn heads kv_heads head_dim T ctx]`
//! (по умолчанию Qwen3-8B, T=1 — decode; T=128 — prefill; ctx — длина KV)
#![cfg(feature = "metal")]
use rust_inference::backend::Backend;
use rust_inference::dtype::WeightFormat;
use rust_inference::metal::MetalBackend;
use std::time::Instant;

fn q8(n: usize, k: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * k / 32 * 34);
    for i in 0..n * k / 32 {
        out.extend_from_slice(&half::f16::from_f32(0.01).to_le_bytes());
        for j in 0..32 {
            out.push(((i * 31 + j * 7) % 255) as u8);
        }
    }
    out
}

fn time<F: FnMut()>(be: &MetalBackend, name: &str, reps: usize, mut f: F) -> f64 {
    for _ in 0..2 {
        be.begin_batch().unwrap();
        f();
        be.end_batch().unwrap();
    }
    let t0 = Instant::now();
    be.begin_batch().unwrap();
    for _ in 0..reps {
        f();
    }
    be.end_batch().unwrap();
    let us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;
    println!("{name:34} {us:9.1} мкс");
    us
}

fn main() {
    let a: Vec<usize> = std::env::args().skip(1).filter_map(|v| v.parse().ok()).collect();
    let g = |i: usize, d: usize| a.get(i).copied().unwrap_or(d);
    let (d, ffn, nh, nkv, hd, t, ctx) = (g(0, 4096), g(1, 12288), g(2, 32), g(3, 8), g(4, 128), g(5, 1), g(6, 64));
    let reps = if t == 1 { 64 } else { 16 };
    let be = MetalBackend::new().unwrap();
    let (qs, kvs) = (nh * hd, nkv * hd);
    let wqkv = be.from_quantized(&q8(qs + 2 * kvs, d), vec![qs + 2 * kvs, d], WeightFormat::Q8_0).unwrap();
    let wo = be.from_quantized(&q8(d, qs), vec![d, qs], WeightFormat::Q8_0).unwrap();
    let w13 = be.from_quantized(&q8(2 * ffn, d), vec![2 * ffn, d], WeightFormat::Q8_0).unwrap();
    let w2 = be.from_quantized(&q8(d, ffn), vec![d, ffn], WeightFormat::Q8_0).unwrap();
    let x = be.from_cpu_f32(&vec![0.01f32; t * d], vec![t, d]).unwrap();
    let x3 = x.view(vec![1, t, d]);
    let y = be.zeros(vec![t, d]).unwrap();
    let qkv = be.zeros(vec![t, qs + 2 * kvs]).unwrap();
    let f = be.zeros(vec![t, 2 * ffn]).unwrap();
    let gm = be.zeros(vec![t, ffn]).unwrap();
    let w = be.from_cpu_f32(&vec![1.0f32; d], vec![d]).unwrap();
    let qw = be.from_cpu_f32(&vec![1.0f32; hd], vec![hd]).unwrap();
    let q = be.zeros(vec![1, t, qs]).unwrap();
    let k1 = be.zeros(vec![1, t, kvs]).unwrap();
    let kv = be.zeros(vec![1, ctx + t, kvs]).unwrap();
    let o = be.zeros(vec![1, t, qs]).unwrap();
    let mask = if t > 1 {
        Some(rust_inference::nn::build_causal_mask(&be, t, ctx + t, ctx, None).unwrap())
    } else {
        None
    };
    let buf = be.zeros(vec![1, ctx + t + 256, kvs]).unwrap();
    let qkv3 = qkv.view(vec![1, t, qs + 2 * kvs]);
    println!("слой: dim {d} ffn {ffn} heads {nh}/{nkv}×{hd}, T={t}, ctx={ctx}");
    let mut total = 0.0;
    total += 2.0 * time(&be, "rmsnorm ×2", reps, || be.rmsnorm(&x, &w, &y, 1e-6).unwrap());
    total += time(&be, "gemm wqkv", reps, || be.gemm_nt(&x, &wqkv, &qkv).unwrap());
    total += 3.0 * time(&be, "slice_last ×3", reps, || {
        let _ = rust_inference::nn::slice_last(&be, &qkv3, qs, 0).unwrap();
    });
    total += time(&be, "norm+rope q", reps, || {
        let _ = be.rmsnorm_head_rope(&q, &qw, ctx, hd, 1e6, nh, 1e-6, None).unwrap();
    });
    total += time(&be, "norm+rope k", reps, || {
        let _ = be.rmsnorm_head_rope(&k1, &qw, ctx, hd, 1e6, nkv, 1e-6, None).unwrap();
    });
    total += 2.0 * time(&be, "kv_update ×2", reps, || be.kv_update(&buf, &k1, ctx).unwrap());
    total += time(&be, &format!("attn_ar tkv={}", ctx + t), reps, || {
        be.attn_ar(&q, &kv, &kv, mask.as_ref(), &o, nh, nkv, hd).unwrap()
    });
    let o2 = o.view(vec![t, qs]);
    total += time(&be, "gemm wo", reps, || be.gemm_nt(&o2, &wo, &y).unwrap());
    total += 2.0 * time(&be, "add ×2", reps, || be.add(&x3, &x3, &x3).unwrap());
    total += time(&be, "gemm w13", reps, || be.gemm_nt(&x, &w13, &f).unwrap());
    total += time(&be, "silu_mul", reps, || be.silu_mul_fused(&f, &gm, ffn).unwrap());
    total += time(&be, "gemm w2", reps, || be.gemm_nt(&gm, &w2, &y).unwrap());
    println!("сумма на слой: {total:.0} мкс");
}
