//! GEMM correctness across the shapes the TTS/STT models actually use
//! (cuBLAS path). `cargo test --release --features "cuda cpu" --test gemm_shapes`.

#![cfg(all(feature = "cuda", feature = "cpu"))]

use rust_inference::cpu::CpuBackend;
use rust_inference::cuda::CudaBackend;
use rust_inference::harness::Lcg;
use rust_inference::Backend;

fn check(gpu: &CudaBackend, cpu: &CpuBackend, m: usize, k: usize, n: usize) {
    let seed = (m * 7919 + k * 104729 + n) as u64;
    let a = {
        let mut d = vec![0.0f32; m * k];
        Lcg::new(seed).fill(&mut d);
        gpu.from_cpu_f32(&d, vec![m, k]).unwrap()
    };
    let ac = {
        let mut d = vec![0.0f32; m * k];
        Lcg::new(seed).fill(&mut d);
        cpu.from_cpu_f32(&d, vec![m, k]).unwrap()
    };
    let b = {
        let mut d = vec![0.0f32; n * k];
        Lcg::new(seed + 1).fill(&mut d);
        gpu.from_cpu_f32(&d, vec![n, k]).unwrap()
    };
    let bc = {
        let mut d = vec![0.0f32; n * k];
        Lcg::new(seed + 1).fill(&mut d);
        cpu.from_cpu_f32(&d, vec![n, k]).unwrap()
    };
    let y = gpu.zeros(vec![m, n]).unwrap();
    let yc = cpu.zeros(vec![m, n]).unwrap();
    gpu.gemm_nt(&a, &b, &y).unwrap();
    gpu.synchronize().unwrap();
    cpu.gemm_nt(&ac, &bc, &yc).unwrap();
    let g = gpu.to_cpu_f32(&y).unwrap();
    let c = cpu.to_cpu_f32(&yc).unwrap();
    for i in 0..g.len() {
        let ok = (g[i] - c[i]).abs() <= 2e-3 + 2e-3 * c[i].abs();
        assert!(ok, "gemm_nt [{m}x{k}x{n}] [{i}]: gpu {} cpu {}", g[i], c[i]);
    }
    println!("gemm_nt {m:>4}x{k:>4}x{n:>4} ok");
}

#[test]
fn gemm_nt_shapes() {
    let gpu = CudaBackend::new(0).unwrap();
    let cpu = CpuBackend;
    // TTS slow/fast AR: logits, qkv (3·d), attention out, FFN up/down.
    check(&gpu, &cpu, 1, 2048, 400);   // logits: (1,1,d) x (vocab,d)
    check(&gpu, &cpu, 1, 2048, 2048);  // single-token dense
    check(&gpu, &cpu, 4, 2048, 6144);  // qkv, 4 tokens
    check(&gpu, &cpu, 8, 2048, 6144);  // qkv, 8 tokens
    check(&gpu, &cpu, 8, 2048, 2048);  // attn out
    check(&gpu, &cpu, 8, 2048, 8192);  // FFN up
    check(&gpu, &cpu, 8, 8192, 2048);  // FFN down
    check(&gpu, &cpu, 64, 2048, 2048);
    check(&gpu, &cpu, 1, 1024, 1024);
    check(&gpu, &cpu, 8, 1024, 4096);
    // DAC codec: in/out projections.
    check(&gpu, &cpu, 1, 128, 1024);
    check(&gpu, &cpu, 1, 1024, 128);
    check(&gpu, &cpu, 1, 8, 1024);
    check(&gpu, &cpu, 64, 128, 1024);
    check(&gpu, &cpu, 64, 1024, 128);
    // STT t-one: per-frame linear, reduction pw.
    check(&gpu, &cpu, 33, 384, 384);
    check(&gpu, &cpu, 66, 384, 384);
    check(&gpu, &cpu, 66, 1536, 384);
    check(&gpu, &cpu, 470, 384, 384);
    check(&gpu, &cpu, 470, 1536, 384);
}
