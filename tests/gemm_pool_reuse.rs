//! Pool-reuse regression gate: many consecutive half-weight GEMMs share
//! the cpart scratch buffers. The tiled kernels write only a SUBSET of
#![cfg(feature = "cuda")]
//! the 16 z-slices (k not divisible by 16·nkz); reducing ALL 16 summed
//! stale pool contents from a previous GEMM in the same bucket — a 100%+
//! error that silently broke every quantized STT engine (gigaam/t-one).
use rust_inference::backend::Backend;
use rust_inference::cpu::CpuBackend;
use rust_inference::cuda::CudaBackend;

fn genr(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 32) as f32 / (1u64 << 32) as f32) * 2.0 - 1.0
        })
        .collect()
}

#[test]
fn sequential_halfw_gemms() {
    let cuda = CudaBackend::new(0).expect("cuda");
    let cpu = CpuBackend;
    // Two alternating real shapes (gigaam layer: linear_q [768,768] and
    // ffn1 [3072,768] with m=~90) — 32 GEMMs, like a 16-layer encoder.
    let shapes = [(90usize, 768usize, 768usize), (90, 768, 3072), (90, 3072, 768)];
    for round in 0..3 {
        let mut worst_rel = 0.0f32;
        for i in 0..32 {
            let (m, k, n) = shapes[(round * 32 + i) % 3];
            let a = genr(1000 + i as u64, m * k);
            let b = genr(2000 + i as u64, n * k);
            let bh: Vec<half::bf16> = b.iter().map(|&v| half::bf16::from_f32(v)).collect();
            let at = cuda.from_cpu_f32(&a, vec![m, k]).unwrap();
            let bt = cuda.from_cpu_bf16(&bh, vec![n, k]).unwrap();
            let ct = cuda.zeros(vec![m, n]).unwrap();
            cuda.gemm_nt(&at, &bt, &ct).unwrap();
            let got = cuda.to_cpu_f32(&ct).unwrap();
            let ac = cpu.from_cpu_f32(&a, vec![m, k]).unwrap();
            let bc = cpu.from_cpu_f32(&b, vec![n, k]).unwrap();
            let cc = cpu.zeros(vec![m, n]).unwrap();
            cpu.gemm_nt(&ac, &bc, &cc).unwrap();
            let want = cpu.to_cpu_f32(&cc).unwrap();
            let mut worst = 0.0f32;
            let mut scale = 0.0f32;
            for (g, w) in got.iter().zip(want.iter()) {
                worst = worst.max((g - w).abs());
                scale = scale.max(w.abs());
            }
            let rel = worst / scale.max(1.0);
            if rel > 0.05 || i < 3 {
                println!("  gemm {i}: ({m},{k},{n}) rel {rel:.4}");
            }
            worst_rel = worst_rel.max(rel);
        }
        assert!(worst_rel < 0.05, "round {round}: sequential halfw GEMMs diverged (pool-reuse bug): {worst_rel:.4}");
    }
}
