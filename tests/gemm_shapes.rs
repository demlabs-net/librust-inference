//! GEMM correctness across the shapes the TTS/STT models actually use
//! (cuBLAS path). `cargo test --release --features "cuda cpu" --test gemm_shapes`.

#![cfg(all(feature = "cuda", feature = "cpu"))]

use rust_inference::Backend;
use rust_inference::cpu::CpuBackend;
use rust_inference::cuda::CudaBackend;
use rust_inference::harness::Lcg;

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
    check(&gpu, &cpu, 1, 2048, 400); // logits: (1,1,d) x (vocab,d)
    check(&gpu, &cpu, 1, 2048, 2048); // single-token dense
    check(&gpu, &cpu, 4, 2048, 6144); // qkv, 4 tokens
    check(&gpu, &cpu, 8, 2048, 6144); // qkv, 8 tokens
    check(&gpu, &cpu, 8, 2048, 2048); // attn out
    check(&gpu, &cpu, 8, 2048, 8192); // FFN up
    check(&gpu, &cpu, 8, 8192, 2048); // FFN down
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

/// int8 (Q8_0-style) GEMM correctness vs the fp32 CPU reference. Weights are
/// quantized per 32-element blocks; the reference compares against the
/// dequantized approximation, so the tolerance is the int8 quantization
/// error (Lcg values in [-0.5, 0.5), ~127 levels → <1e-2 absolute).
fn check_i8(gpu: &CudaBackend, cpu: &CpuBackend, m: usize, k: usize, n: usize) {
    assert_eq!(k % 32, 0, "k must be a multiple of 32");
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
    // Quantize weights per 32-block on the CPU (same as upload_quant).
    let nblk = k / 32;
    let mut b = vec![0.0f32; n * k];
    Lcg::new(seed + 1).fill(&mut b);
    let mut q = vec![0i8; n * k];
    let mut sc = vec![0f32; n * nblk];
    for r in 0..n {
        for blk in 0..nblk {
            let base = r * k + blk * 32;
            let mx = b[base..base + 32]
                .iter()
                .fold(0f32, |mm, &v| mm.max(v.abs()));
            let s = if mx > 0.0 { mx / 127.0 } else { 1.0 };
            sc[r * nblk + blk] = s;
            for j in 0..32 {
                q[base + j] = (b[base + j] / s).round().clamp(-128.0, 127.0) as i8;
            }
        }
    }
    // Upload kb-major (block (kb, col) = 32 contiguous i8 at
    // (kb·rows + col)·32, scale at (kb·rows + col)) — matches upload_quant
    // / the fused gemm_q8k_splitk kernels.
    let mut qt = vec![0i8; n * k];
    for r in 0..n {
        for b in 0..nblk {
            for j in 0..32 {
                qt[(b * n + r) * 32 + j] = q[r * k + b * 32 + j];
            }
        }
    }
    let mut sct = vec![0f32; n * nblk];
    for r in 0..n {
        for b in 0..nblk {
            sct[b * n + r] = sc[r * nblk + b];
        }
    }
    let bw = gpu.from_cpu_i8(&qt, &sct, vec![n, k]).unwrap();
    // CPU reference: dequantized weight matrix → plain fp32 GEMM.
    let mut bd = vec![0.0f32; n * k];
    for r in 0..n {
        for blk in 0..nblk {
            let base = r * k + blk * 32;
            for j in 0..32 {
                bd[base + j] = q[base + j] as f32 * sc[r * nblk + blk];
            }
        }
    }
    let bdc = cpu.from_cpu_f32(&bd, vec![n, k]).unwrap();

    let y = gpu.zeros(vec![m, n]).unwrap();
    let yc = cpu.zeros(vec![m, n]).unwrap();
    gpu.gemm_nt(&a, &bw, &y).unwrap();
    gpu.synchronize().unwrap();
    cpu.gemm_nt(&ac, &bdc, &yc).unwrap();
    let g = gpu.to_cpu_f32(&y).unwrap();
    let c = cpu.to_cpu_f32(&yc).unwrap();
    let tol = 2e-2 + 1e-2 * c.iter().fold(0f32, |mm, &v| mm.max(v.abs()));
    let mut bad = 0;
    for i in 0..g.len() {
        let ok = (g[i] - c[i]).abs() <= tol;
        if !ok && bad < 8 {
            eprintln!(
                "int8 [{m}x{k}x{n}] row {} col {}: gpu {} cpu {}",
                i / n,
                i % n,
                g[i],
                c[i]
            );
            bad += 1;
        }
        assert!(
            ok,
            "int8 gemm_nt [{m}x{k}x{n}] [{i}]: gpu {} cpu {}",
            g[i], c[i]
        );
    }
    println!("gemm_nt int8 {m:>4}x{k:>4}x{n:>4} ok");
}

#[test]
fn gemm_nt_i8_shapes() {
    let gpu = CudaBackend::new(0).unwrap();
    let cpu = CpuBackend;
    // TTS decode (m=1) + fill (m>1) shapes, incl. non-square FFN.
    check_i8(&gpu, &cpu, 1, 2048, 6144);
    check_i8(&gpu, &cpu, 1, 2048, 8192);
    check_i8(&gpu, &cpu, 1, 8192, 2048);
    check_i8(&gpu, &cpu, 1, 2048, 1024);
    check_i8(&gpu, &cpu, 1, 2048, 400);
    check_i8(&gpu, &cpu, 8, 2048, 6144);
    check_i8(&gpu, &cpu, 8, 2048, 8192);
    check_i8(&gpu, &cpu, 8, 8192, 2048);
    check_i8(&gpu, &cpu, 33, 2048, 2048);
    // Prefill-shaped (TTS fill / STT one-shot frames) — the tiled m>1 path.
    check_i8(&gpu, &cpu, 64, 2560, 3840);
    check_i8(&gpu, &cpu, 470, 1536, 384);
    // Edge: m not a multiple of 64 / n not a multiple of 64 (tiled masks).
    check_i8(&gpu, &cpu, 65, 2048, 65);
    check_i8(&gpu, &cpu, 129, 1024, 400);
}

/// Half-weight (fp16/bf16) GEMM correctness vs the fp32 CPU reference —
/// the own split-k kernels (gemm_halfw_splitk_*) replace cuBLAS gemm_ex
/// for half weights, which refuses f32-A × half-B. The A operand follows
/// the activation axis (fp32 default, fp16/bf16 cast the rows); the split
/// factor S is picked per shape (grid cap ≤512, divisibility of k/32).
/// Weights are rounded to the half dtype on the CPU and the reference runs
/// against those exact rounded values — the tolerance covers only the
/// operand rounding and the fp32 accumulation.
fn check_half(
    gpu: &CudaBackend,
    cpu: &CpuBackend,
    m: usize,
    k: usize,
    n: usize,
    wf16: bool,
    act: rust_inference::dtype::Dtype,
) {
    // k < 32 shapes (e.g. the DAC codec's 1×8×1024) run the naive fallback.
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
    let mut b = vec![0.0f32; n * k];
    Lcg::new(seed + 1).fill(&mut b);
    let bw = if wf16 {
        let h: Vec<half::f16> = b.iter().map(|&x| half::f16::from_f32(x)).collect();
        b = h.iter().map(|v| v.to_f32()).collect();
        gpu.from_cpu_f16(&h, vec![n, k]).unwrap()
    } else {
        let h: Vec<half::bf16> = b.iter().map(|&x| half::bf16::from_f32(x)).collect();
        b = h.iter().map(|v| v.to_f32()).collect();
        gpu.from_cpu_bf16(&h, vec![n, k]).unwrap()
    };
    let bdc = cpu.from_cpu_f32(&b, vec![n, k]).unwrap();
    let y = gpu.zeros(vec![m, n]).unwrap();
    let yc = cpu.zeros(vec![m, n]).unwrap();
    gpu.set_activation_dtype(act).unwrap();
    gpu.gemm_nt(&a, &bw, &y).unwrap();
    gpu.synchronize().unwrap();
    cpu.gemm_nt(&ac, &bdc, &yc).unwrap();
    let g = gpu.to_cpu_f32(&y).unwrap();
    let c = cpu.to_cpu_f32(&yc).unwrap();
    let tol = 2e-2 + 2e-2 * c.iter().fold(0f32, |mm, &v| mm.max(v.abs()));
    let mut bad = 0;
    for i in 0..g.len() {
        let ok = (g[i] - c[i]).abs() <= tol;
        if !ok && bad < 8 {
            eprintln!(
                "half w={} act={:?} [{m}x{k}x{n}] row {} col {}: gpu {} cpu {}",
                if wf16 { "f16" } else { "bf16" },
                act,
                i / n,
                i % n,
                g[i],
                c[i]
            );
            bad += 1;
        }
        assert!(
            ok,
            "half gemm_nt [{m}x{k}x{n}] w={} act={:?} [{i}]: gpu {} cpu {}",
            if wf16 { "f16" } else { "bf16" },
            act,
            g[i],
            c[i]
        );
    }
    println!(
        "gemm_nt half w={} act={:?} {m:>4}x{k:>4}x{n:>4} ok",
        if wf16 { "f16" } else { "bf16" },
        act
    );
}

#[test]
fn gemm_nt_half_shapes() {
    use rust_inference::dtype::Dtype;
    let gpu = CudaBackend::new(0).unwrap();
    let cpu = CpuBackend;
    // Decode (m=1), fill (m>1) and guard-edge shapes: n not a multiple of
    // 256 (col < n), grid-cap forced S=2 (m=65), small k (S=4).
    let shapes = [
        (1usize, 2048usize, 6144usize),
        (1, 2048, 400),
        (8, 8192, 2048),
        (65, 1024, 400),
        (1, 128, 300),
        (4, 1280, 3840),
        // Prefill-shaped (TTS fill / STT one-shot) — the tiled m>1 path.
        (64, 2560, 3840),
        (470, 1536, 384),
        // DAC codec k<32 (raw row-major half weights — the naive fallback).
        (1, 8, 1024),
    ];
    for &(m, k, n) in &shapes {
        for wf16 in [false, true] {
            for act in [Dtype::F32, Dtype::F16, Dtype::BF16] {
                check_half(&gpu, &cpu, m, k, n, wf16, act);
            }
        }
    }
}
