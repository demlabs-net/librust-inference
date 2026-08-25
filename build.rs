//! Build script: compiles the CUDA kernels to PTX (compute_86 — one PTX JITs
//! onto sm_86/89/120).
//!
//! Policy (plan): the committed `src/cuda/kernels.ptx` is the source of
//! truth for CI and hosts without the CUDA toolkit; when nvcc is available
//! it is regenerated. The sm_120 CUBIN and the mma.sync tensor-core kernels
//! were removed (2026-08-25) after the investigation: the ptxas miscompiles
//! the mma.sync fragment loads on sm_120 (out-of-bounds shared reads), the
//! wmma API is gone from CUDA 13, and tcgen05 is datacenter-only (sm_100a) —
//! the f32-accumulating tiled kernels are the production path.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let kernels_cu = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/cuda/kernels.cu");
    let committed_ptx = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/cuda/kernels.ptx");
    let target_ptx = out_dir.join("kernels.ptx");

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
}
