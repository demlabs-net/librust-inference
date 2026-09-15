//! N (iter-10): GGUF Q6_K/Q4_K tensors must upload PACKED (kb-major, no
//! dequant) and run through the fused dequant-in-kernel GEMMs with the same
//! numerics as the CPU reference — the old path dequantized to f32 (5-7x
//! DRAM traffic). End-to-end: canonical encoder → from_quantized (with the
//! row-major→kb-major transpose) → gemm_nt vs CPU oracle.
//! `cargo test --release --features "cuda cpu" --test gguf_packed_gemm`

#![cfg(all(feature = "cuda", feature = "cpu"))]

use rust_inference::backend::Backend;
use rust_inference::cpu::CpuBackend;
use rust_inference::cuda::CudaBackend;
use rust_inference::dtype::WeightFormat;
use rust_inference::quant::{quant_q4_k, quant_q6_k};

fn make_gpu() -> Option<CudaBackend> {
    std::panic::catch_unwind(|| CudaBackend::new(0))
        .ok()
        .and_then(|r| r.ok())
}

fn approx(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len());
    let mut worst = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        let e = (x - y).abs();
        if e > worst {
            worst = e;
        }
        assert!(e <= tol, "mismatch: {x} vs {y} (worst {worst})");
    }
}

/// Encode [rows·cols] f32 weights with a canonical ggml encoder into the
/// GGUF row-major super-block stream (what from_quantized receives).
fn gguf_stream(weights: &[f32], _cols: usize, fmt: WeightFormat) -> Vec<u8> {
    match fmt {
        WeightFormat::Q6_K => quant_q6_k(weights),
        WeightFormat::Q4_K => quant_q4_k(weights),
        _ => unreachable!(),
    }
}

fn run_case(fmt: WeightFormat, _blk: usize, tol: f32) {
    let Some(gpu) = make_gpu() else {
        eprintln!("no CUDA — skipping");
        return;
    };
    let cpu = CpuBackend;
    // gemm_nt: c[m,n] = a[m,k] · b[n,k]^T — the packed weight is (n,k).
    let (m, n, k) = (16usize, 8usize, 512usize); // k % 256 == 0
    // Random-ish weights in [-2, 2].
    let w: Vec<f32> = (0..n * k)
        .map(|i| ((((i as u64) * 2654435761u64) as f64 * 1e-9).sin() * 2.0) as f32)
        .collect();
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((((i as u64) * 40503u64) as f64 * 1e-9).cos() * 0.5) as f32)
        .collect();

    // GGUF row-major stream → from_quantized (transpose + packed upload).
    let stream = gguf_stream(&w, k, fmt);
    let b_gpu = gpu.from_quantized(&stream, vec![n, k], fmt).unwrap();
    let b_cpu = cpu
        .from_cpu_f32(
            &rust_inference::weights_map::dequant_f32(
                &rust_inference::weights_map::WeightTensor::from_parts(
                    "b",
                    vec![n as u64, k as u64],
                    fmt,
                    &stream,
                ),
            ),
            vec![n, k],
        )
        .unwrap();

    // CPU oracle: C = A @ B^T (a: [m, k], b: [n, k]).
    let a_cpu = cpu.from_cpu_f32(&a, vec![m, k]).unwrap();
    let mut c_cpu = cpu.zeros(vec![m, n]).unwrap();
    cpu.gemm_nt(&a_cpu, &b_cpu, &c_cpu).unwrap();

    let a_gpu = gpu.from_cpu_f32(&a, vec![m, k]).unwrap();
    let mut c_gpu = gpu.zeros(vec![m, n]).unwrap();
    gpu.gemm_nt(&a_gpu, &b_gpu, &c_gpu).unwrap();

    approx(
        &gpu.to_cpu_f32(&c_gpu).unwrap(),
        &cpu.to_cpu_f32(&c_cpu).unwrap(),
        tol,
    );
    // Sanity: the packed path must NOT be a silent f32 fallback — the
    // packed uploads carry dtype I8 (the f32 fallback would be F32).
    assert_eq!(
        b_gpu.dtype(),
        rust_inference::dtype::Dtype::I8,
        "packed upload expected"
    );
}

#[test]
fn gguf_q6k_packed_gemm_matches_cpu() {
    run_case(WeightFormat::Q6_K, 210, 1e-2);
}

#[test]
fn gguf_q4k_packed_gemm_matches_cpu() {
    run_case(WeightFormat::Q4_K, 144, 1e-2);
}
