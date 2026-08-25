//! Build script: compiles the CUDA kernels to PTX (compute_86 — one PTX JITs
//! onto sm_86/89/120) and to a native sm_120 CUBIN (Blackwell 5090).
//!
//! Policy (plan): the committed `src/cuda/kernels.ptx` and
//! `src/cuda/kernels.sm120.cubin` are the source of truth for CI and hosts
//! without the CUDA toolkit; when nvcc is available both are regenerated.
//! The CUBIN matters: the mma.sync/wmma tensor-core fragment layouts of a
//! compute_86 PTX are MISCOMPILED by the sm_120 JIT (measured: non-doc C
//! fragment order, ~8% errors on the first tile rows) — the native CUBIN
//! keeps the documented layouts. The backend loads the CUBIN on sm_120 and
//! falls back to the PTX elsewhere.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let kernels_cu = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/cuda/kernels.cu");
    let committed_ptx = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/cuda/kernels.ptx");
    let committed_cubin =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/cuda/kernels.sm120.cubin");
    let target_ptx = out_dir.join("kernels.ptx");
    let target_cubin = out_dir.join("kernels.sm120.cubin");

    println!("cargo:rerun-if-changed=src/cuda/kernels.cu");
    println!("cargo:rerun-if-env-changed=NVCC");

    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "nvcc".into());
    let ptx_ok = Command::new(&nvcc)
        .args(["-arch=compute_86", "-ptx", "-o"])
        .arg(&target_ptx)
        .arg(&kernels_cu)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ptx_ok {
        // Fallback: the committed PTX is the source of truth.
        println!("cargo:warning=nvcc unavailable — using committed kernels.ptx");
        std::fs::copy(&committed_ptx, &target_ptx)
            .expect("committed kernels.ptx missing — run: nvcc -arch=compute_86 -ptx -o src/cuda/kernels.ptx src/cuda/kernels.cu");
    }
    let cubin_ok = Command::new(&nvcc)
        .args(["-arch=sm_120", "-cubin", "-o"])
        .arg(&target_cubin)
        .arg(&kernels_cu)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !cubin_ok {
        // Fallback: the committed CUBIN is the source of truth.
        println!("cargo:warning=nvcc unavailable — using committed kernels.sm120.cubin");
        std::fs::copy(&committed_cubin, &target_cubin).expect(
            "committed kernels.sm120.cubin missing — run: nvcc -arch=sm_120 -cubin -o src/cuda/kernels.sm120.cubin src/cuda/kernels.cu",
        );
    }
}
