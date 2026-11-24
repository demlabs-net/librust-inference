//! GPU↔CPU parity for the fused SwiGLU gate (silu_mul_fused): x is
//! (rows, 2·ffn) row-major with PER-ROW gate|up halves — the kernel must
//! index row = i/ffn, col = i%ffn and read x[row·2·ffn + col] /
//! x[row·2·ffn + ffn + col], exactly like the CPU oracle.
//!
//! Would have failed before the row-offset fix: the flat
//! y[i] = silu(x[i]) · x[ffn + i] was only correct for row 0 — every row ≥ 1
//! read across the row boundary (garbage FFN during prefill of every
//! LLM/omni/vision model).
//!
//! `cargo test --release --features "cuda cpu" --test silu_mul_parity -- --nocapture`

#![cfg(all(feature = "cuda", feature = "cpu"))]

use rust_inference::Backend;
use rust_inference::cpu::CpuBackend;
use rust_inference::cuda::CudaBackend;
use rust_inference::harness::Lcg;

fn make_gpu() -> Option<CudaBackend> {
    let r = std::panic::catch_unwind(|| CudaBackend::new(0));
    let err = match r {
        Ok(Ok(b)) => return Some(b),
        Ok(Err(e)) => e.to_string(),
        Err(p) => p
            .downcast_ref::<&str>()
            .copied()
            .unwrap_or("panic")
            .to_string(),
    };
    if std::env::var("CUDA_SMOKE_REQUIRED").is_ok() {
        panic!("CUDA device unavailable (CUDA_SMOKE_REQUIRED=1): {err}");
    }
    eprintln!("CUDA device unavailable, skipping: {err}");
    None
}

fn silu_mul_reference(x: &[f32], rows: usize, ffn: usize) -> Vec<f32> {
    // The CPU oracle's exact formula (cpu.rs silu_mul_fused).
    let mut y = vec![0f32; rows * ffn];
    for i in 0..rows * ffn {
        let (row, col) = (i / ffn, i % ffn);
        let g = x[row * 2 * ffn + col];
        let up = x[row * 2 * ffn + ffn + col];
        y[i] = (g / (1.0 + (-g).exp())) * up;
    }
    y
}

fn parity_case(rows: usize, ffn: usize, seed: u64) {
    let Some(gpu) = make_gpu() else { return };
    let cpu = CpuBackend;

    let mut x = vec![0f32; rows * 2 * ffn];
    Lcg::new(seed).fill(&mut x);

    // GPU path through the public backend method.
    let x_g = gpu.from_cpu_f32(&x, vec![rows, 2 * ffn]).unwrap();
    let y_g = gpu.zeros(vec![rows, ffn]).unwrap();
    gpu.silu_mul_fused(&x_g, &y_g, ffn).unwrap();
    gpu.synchronize().unwrap();
    let got = gpu.to_cpu_f32(&y_g).unwrap();

    // Host reference with the oracle's row-major formula.
    let want = silu_mul_reference(&x, rows, ffn);

    // expf (GPU) vs f32::exp (host) differ by a few ulp — compare with a
    // small absolute/relative tolerance instead of bit equality.
    let scale = want.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let tol = 1e-6 + 1e-6 * scale;
    let mut bad = 0usize;
    for i in 0..got.len() {
        let ok = got[i].is_finite() && (got[i] - want[i]).abs() <= tol;
        if !ok {
            bad += 1;
            if bad < 5 {
                eprintln!(
                    "r={rows} ffn={ffn} [{i}]: gpu {} != reference {}",
                    got[i], want[i]
                );
            }
        }
    }
    assert_eq!(
        bad, 0,
        "silu_mul_fused {rows}x{ffn}: {bad} elements off (tol {tol})"
    );

    // And the CPU backend itself must agree with the same input (both
    // backends share the trait contract).
    let x_c = cpu.from_cpu_f32(&x, vec![rows, 2 * ffn]).unwrap();
    let y_c = cpu.zeros(vec![rows, ffn]).unwrap();
    cpu.silu_mul_fused(&x_c, &y_c, ffn).unwrap();
    let cpu_out = cpu.to_cpu_f32(&y_c).unwrap();
    assert_eq!(
        cpu_out, want,
        "cpu backend diverged from the row-major formula"
    );

    println!(
        "silu_mul_fused rows={rows} ffn={ffn}: gpu == reference ({} elems)",
        got.len()
    );
}

#[test]
fn silu_mul_fused_matches_cpu() {
    // rows=4, ffn=64 — multi-row so the per-row offset is exercised
    // (the old flat kernel produced row 0 correctly and rows ≥ 1 garbage).
    parity_case(4, 64, 71);
    // A real caller shape: LLM prefill FFN (rows > 1, ffn a GEMM width).
    parity_case(7, 256, 72);
}

#[test]
fn silu_mul_fused_rejects_bad_ffn() {
    // ffn=0 / non-multiple numel must Err (mirror of the CPU oracle's
    // guard), not divide by zero or mis-slice the last row in the kernel.
    let Some(gpu) = make_gpu() else { return };
    let x_g = gpu.from_cpu_f32(&[0f32; 8], vec![2, 4]).unwrap();
    let y_g = gpu.zeros(vec![2, 2]).unwrap();
    assert!(gpu.silu_mul_fused(&x_g, &y_g, 0).is_err());
    assert!(gpu.silu_mul_fused(&x_g, &y_g, 3).is_err());
}
