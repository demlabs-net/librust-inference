//! GEMM benchmark: cuBLAS vs our naive PTX kernel, on TTS/STT-relevant
//! shapes. `cargo test --release --features "cuda cpu" --test gemm_bench
//! -- --nocapture` (GPU host, VRAM free of other consumers).

#![cfg(all(feature = "cuda", feature = "cpu"))]

use std::time::Instant;

use rust_inference::cuda::CudaBackend;
use rust_inference::harness::Lcg;
use rust_inference::Backend;

fn bench_nt(gpu: &CudaBackend, m: usize, k: usize, n: usize, reps: usize) {
    let a = gpu.from_cpu_f32(&vec![0.5f32; m * k], vec![m, k]).unwrap();
    let b = gpu.from_cpu_f32(&vec![0.25f32; n * k], vec![n, k]).unwrap();
    let c = gpu.zeros(vec![m, n]).unwrap();

    // Warmup.
    gpu.gemm_nt(&a, &b, &c).unwrap();
    gpu.synchronize().unwrap();

    let t0 = Instant::now();
    for _ in 0..reps {
        gpu.gemm_nt(&a, &b, &c).unwrap();
    }
    gpu.synchronize().unwrap();
    let dt = t0.elapsed().as_secs_f64() / reps as f64;
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    println!(
        "gemm_nt {:>5}x{:>5}x{:>5}: {:>9.2} us  {:>7.1} GFLOP/s",
        m, n, k,
        dt * 1e6,
        flops / dt / 1e9
    );
}

#[test]
fn gemm_bench_suite() {
    let gpu = CudaBackend::new(0).unwrap();
    println!("\n=== gemm_nt (A[m,k]·B[n,k]ᵀ) — {} ===", if std::env::var_os("CUDA_NO_CUBLAS").is_some() { "NAIVE PTX" } else { "cuBLAS" });
    // TTS shapes: 2048/1024 dense, 128-ish per-token, codec 64/128.
    bench_nt(&gpu, 64, 64, 64, 2000);
    bench_nt(&gpu, 128, 128, 128, 1000);
    bench_nt(&gpu, 256, 256, 256, 500);
    bench_nt(&gpu, 512, 512, 512, 200);
    bench_nt(&gpu, 1024, 1024, 1024, 100);
    bench_nt(&gpu, 2048, 2048, 2048, 50);
    bench_nt(&gpu, 4096, 4096, 4096, 20);
    // STT shapes (per-frame linear): [33|66, 384] x [384, 384] and reduction.
    bench_nt(&gpu, 33, 384, 384, 1000);
    bench_nt(&gpu, 66, 384, 384, 500);
    bench_nt(&gpu, 66, 1536, 384, 500);
    // TTS decode (m=1) shapes.
    bench_nt(&gpu, 1, 2048, 2048, 500);
    bench_nt(&gpu, 1, 2048, 6144, 200);   // qkv
    bench_nt(&gpu, 1, 2048, 8192, 200);   // ffn up (w1/w3)
    bench_nt(&gpu, 1, 8192, 2048, 200);   // ffn down (w2)
    bench_nt(&gpu, 1, 2048, 1024, 500);   // logits head
    bench_nt(&gpu, 4, 2048, 2048, 500);
    bench_nt(&gpu, 4, 2048, 6144, 200);
    bench_nt(&gpu, 4, 8192, 2048, 200);
    let _ = Lcg::new(1);
}

fn bench_nt_i8(gpu: &CudaBackend, m: usize, k: usize, n: usize, reps: usize) {
    // Q8_0-style int8 weights: quantize a random fp32 matrix per 32-block.
    let mut b = vec![0f32; n * k];
    for v in b.iter_mut() {
        *v = (Lcg::new(1).next_f32() - 0.5) * 2.0;
    }
    let nblk = k / 32;
    let mut q = vec![0i8; n * k];
    let mut sc = vec![0f32; n * nblk];
    for r in 0..n {
        for blk in 0..nblk {
            let base = r * k + blk * 32;
            let mx = b[base..base + 32].iter().fold(0f32, |m, &v| m.max(v.abs()));
            let s = if mx > 0.0 { mx / 127.0 } else { 1.0 };
            sc[r * nblk + blk] = s;
            for j in 0..32 {
                q[base + j] = (b[base + j] / s).round().clamp(-128.0, 127.0) as i8;
            }
        }
    }
    // Upload transposed (k, n) — matches upload_quant / the CUDA kernels.
    let mut qt = vec![0i8; n * k];
    for r in 0..n {
        for kk in 0..k {
            qt[kk * n + r] = q[r * k + kk];
        }
    }
    let bw = gpu.from_cpu_i8(&qt, &sc, vec![n, k]).unwrap();
    let a = gpu.from_cpu_f32(&vec![0.5f32; m * k], vec![m, k]).unwrap();
    let c = gpu.zeros(vec![m, n]).unwrap();

    gpu.gemm_nt(&a, &bw, &c).unwrap();
    gpu.synchronize().unwrap();

    let t0 = Instant::now();
    for _ in 0..reps {
        gpu.gemm_nt(&a, &bw, &c).unwrap();
    }
    gpu.synchronize().unwrap();
    let dt = t0.elapsed().as_secs_f64() / reps as f64;
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    println!(
        "gemm_nt int8 {:>4}x{:>5}x{:>5}: {:>9.2} us  {:>7.1} GFLOP/s",
        m, n, k,
        dt * 1e6,
        flops / dt / 1e9
    );
}

#[test]
fn gemm_bench_i8_suite() {
    let gpu = CudaBackend::new(0).unwrap();
    println!("\n=== gemm_nt int8 (Q8_0-style) ===");
    bench_nt_i8(&gpu, 1, 2048, 6144, 200);   // qkv decode
    bench_nt_i8(&gpu, 1, 2048, 8192, 200);   // ffn up decode
    bench_nt_i8(&gpu, 1, 8192, 2048, 200);   // ffn down decode
    bench_nt_i8(&gpu, 1, 2048, 1024, 500);   // logits head
    bench_nt_i8(&gpu, 32, 2048, 6144, 100);  // qkv fill
    bench_nt_i8(&gpu, 32, 8192, 2048, 100);  // ffn down fill
}
