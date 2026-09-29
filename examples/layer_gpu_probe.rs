//! GPU-время компонентов decode-слоя Qwen3-8B (Ф8.2): каждый шаг
//! прогоняется 64 раза подряд в ОДНОМ батче — CPU-кодирование перекрывается
//! исполнением, замер ≈ чистое GPU-время на вызов.
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

fn time<F: FnMut()>(be: &MetalBackend, name: &str, mut f: F) -> f64 {
    for _ in 0..2 {
        be.begin_batch().unwrap();
        f();
        be.end_batch().unwrap();
    }
    let n = 64;
    let t0 = Instant::now();
    be.begin_batch().unwrap();
    for _ in 0..n {
        f();
    }
    be.end_batch().unwrap();
    let us = t0.elapsed().as_secs_f64() * 1e6 / n as f64;
    println!("{name:34} {us:8.1} мкс");
    us
}

fn main() {
    let be = MetalBackend::new().unwrap();
    let d = 4096;
    let wqkv = be.from_quantized(&q8(6144, d), vec![6144, d], WeightFormat::Q8_0).unwrap();
    let wo = be.from_quantized(&q8(d, d), vec![d, d], WeightFormat::Q8_0).unwrap();
    let w13 = be.from_quantized(&q8(24576, d), vec![24576, d], WeightFormat::Q8_0).unwrap();
    let w2 = be.from_quantized(&q8(d, 12288), vec![d, 12288], WeightFormat::Q8_0).unwrap();
    let x = be.from_cpu_f32(&vec![0.01f32; d], vec![1, d]).unwrap();
    let qkv = be.zeros(vec![1, 6144]).unwrap();
    let f = be.zeros(vec![1, 24576]).unwrap();
    let g = be.zeros(vec![1, 12288]).unwrap();
    let y = be.zeros(vec![1, d]).unwrap();
    let mut total = 0.0;
    total += time(&be, "mv wqkv 6144x4096", || be.gemm_nt(&x, &wqkv, &qkv).unwrap());
    total += time(&be, "mv wo 4096x4096", || be.gemm_nt(&x, &wo, &y).unwrap());
    total += time(&be, "mv w13 24576x4096", || be.gemm_nt(&x, &w13, &f).unwrap());
    total += time(&be, "silu_mul 12288", || be.silu_mul_fused(&f, &g, 12288).unwrap());
    total += time(&be, "mv w2 4096x12288", || be.gemm_nt(&g, &w2, &y).unwrap());
    let w = be.from_cpu_f32(&vec![1.0f32; d], vec![d]).unwrap();
    total += 2.0 * time(&be, "rmsnorm x2", || be.rmsnorm(&x, &w, &y, 1e-6).unwrap());
    let q = be.zeros(vec![1, 1, 4096]).unwrap();
    let qw = be.from_cpu_f32(&vec![1.0f32; 128], vec![128]).unwrap();
    let k1 = be.zeros(vec![1, 1, 1024]).unwrap();
    total += time(&be, "norm_rope q", || {
        let _ = be.rmsnorm_head_rope(&q, &qw, 50, 128, 1e6, 32, 1e-6).unwrap();
    });
    total += time(&be, "norm_rope k", || {
        let _ = be.rmsnorm_head_rope(&k1, &qw, 50, 128, 1e6, 8, 1e-6).unwrap();
    });
    let kv = be.zeros(vec![1, 64, 1024]).unwrap();
    let o = be.zeros(vec![1, 1, 4096]).unwrap();
    total += time(&be, "attn_ar tkv=64", || be.attn_ar(&q, &kv, &kv, None, &o, 32, 8, 128).unwrap());
    let buf = be.zeros(vec![1, 4096, 1024]).unwrap();
    total += 2.0 * time(&be, "kv_update x2", || be.kv_update(&buf, &k1, 10).unwrap());
    let q3 = be.zeros(vec![1, 1, 6144]).unwrap();
    total += 3.0 * time(&be, "slice_last x3", || {
        let _ = rust_inference::nn::slice_last(&be, &q3, 1024, 0).unwrap();
    });
    total += 2.0 * time(&be, "add x2", || be.add(&x, &x, &y).unwrap());
    let bytes = (6144 + 4096 + 24576) as f64 * 4096.0 * 34.0 / 32.0 + 4096.0 * 12288.0 * 34.0 / 32.0;
    println!("сумма на слой: {total:.0} мкс; веса слоя {:.1} МБ → потолок ~{:.0} мкс при 300 ГБ/с; 36 слоёв ≈ {:.1} мс",
        bytes / 1e6, bytes / 300e3, total * 36.0 / 1e3);
}
