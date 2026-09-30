//! Разбивка prefill (Ф8.2): время одного вызова GEMM/attention при M=128
//! токенах для форм Qwen3-8B (Q8_0), батч из 20 повторов.
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

fn time<F: FnMut()>(be: &MetalBackend, name: &str, flops: f64, mut f: F) {
    // 200 повторов: одиночный вызов (0.5–2 мс) тонет в шуме фоновой нагрузки.
    be.begin_batch().unwrap();
    f();
    be.end_batch().unwrap();
    let n = 200;
    let t0 = Instant::now();
    be.begin_batch().unwrap();
    for _ in 0..n {
        f();
    }
    be.end_batch().unwrap();
    let dt = t0.elapsed().as_secs_f64() / n as f64;
    println!("{name:34} {:8.2} мс  {:6.2} TFLOPS", dt * 1e3, flops / dt / 1e12);
}

fn main() {
    let be = MetalBackend::new().unwrap();
    let m = 128usize;
    for (name, n, k) in [("wqkv", 6144usize, 4096usize), ("w13", 24576, 4096), ("w2", 4096, 12288)] {
        let w = be.from_quantized(&q8(n, k), vec![n, k], WeightFormat::Q8_0).unwrap();
        let a = be.from_cpu_f32(&vec![0.01f32; m * k], vec![m, k]).unwrap();
        let c = be.zeros(vec![m, n]).unwrap();
        time(&be, &format!("gemm_nt {name} M={m} [{n}x{k}]"), 2.0 * (m * n * k) as f64, || {
            be.gemm_nt(&a, &w, &c).unwrap()
        });
    }
    // FFN Qwen3-8B целиком (w13 → silu_mul → w2) — сверка суммы с PROF.
    {
        let w13 = be.from_quantized(&q8(24576, 4096), vec![24576, 4096], WeightFormat::Q8_0).unwrap();
        let w2 = be.from_quantized(&q8(4096, 12288), vec![4096, 12288], WeightFormat::Q8_0).unwrap();
        let x = be.from_cpu_f32(&vec![0.01f32; m * 4096], vec![m, 4096]).unwrap();
        let f = be.zeros(vec![m, 24576]).unwrap();
        let g = be.zeros(vec![m, 12288]).unwrap();
        let y = be.zeros(vec![m, 4096]).unwrap();
        time(&be, "ffn целиком M=128", 2.0 * (m * 4096 * 36864) as f64, || {
            be.gemm_nt(&x, &w13, &f).unwrap();
            be.silu_mul_fused(&f, &g, 12288).unwrap();
            be.gemm_nt(&g, &w2, &y).unwrap();
        });
    }
    // GDN Qwen3.5-4B: delta rule на T=128 (32 v-головы, S=128)
    {
        let (t, hk, hv, s) = (m, 16usize, 32usize, 128usize);
        let q = be.zeros(vec![t, hk, s]).unwrap();
        let v = be.zeros(vec![t, hv, s]).unwrap();
        let g = be.zeros(vec![t, hv]).unwrap();
        let st = be.zeros(vec![hv, s, s]).unwrap();
        time(&be, "gdn_delta_rule T=128", 6.0 * (t * hv * s * s) as f64, || {
            let _ = be.gdn_delta_rule(&q, &q, &v, &g, &g, &st, hk).unwrap();
        });
        let x = be.zeros(vec![1, t, 8192]).unwrap();
        let w = be.zeros(vec![8192, 4]).unwrap();
        let cs = be.zeros(vec![8192, 3]).unwrap();
        time(&be, "gdn_conv T=128 C=8192", 2.0 * (t * 8192 * 4) as f64, || {
            let _ = be.gdn_conv(&x, &w, &cs).unwrap();
        });
    }
    // Внимание промпта для двух размеров головы (Qwen3: 128, Qwen3.5: 256)
    for (nh, nkv, hd) in [(32usize, 8usize, 128usize), (16, 4, 256)] {
        let q = be.zeros(vec![1, m, nh * hd]).unwrap();
        let kv = be.zeros(vec![1, m, nkv * hd]).unwrap();
        let o = be.zeros(vec![1, m, nh * hd]).unwrap();
        let mask = rust_inference::nn::build_causal_mask(&be, m, m, 0, None).unwrap();
        time(&be, &format!("attn T={m} nh={nh} hd={hd}"), 4.0 * (m * m * nh * hd) as f64, || {
            be.attn_ar(&q, &kv, &kv, Some(&mask), &o, nh, nkv, hd).unwrap()
        });
    }
}
