//! N (iter-10): CUDA rope/rmsnorm_head_rope overrides must match the CPU
//! reference exactly (the trait defaults used to run host round-trips).
//! `cargo test --release --features "cuda cpu" --test rope_parity`

#![cfg(all(feature = "cuda", feature = "cpu"))]

use rust_inference::Backend;
use rust_inference::cpu::CpuBackend;
use rust_inference::cuda::CudaBackend;

fn make_gpu() -> Option<CudaBackend> {
    std::panic::catch_unwind(|| CudaBackend::new(0))
        .ok()
        .and_then(|r| r.ok())
}

fn approx(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() <= tol, "mismatch: {x} vs {y}");
    }
}

#[test]
fn rope_std_cuda_matches_cpu() {
    let Some(gpu) = make_gpu() else {
        eprintln!("no CUDA — skipping");
        return;
    };
    let cpu = CpuBackend;
    let (tq, d, hd, pos, theta) = (7usize, 1024usize, 128usize, 5usize, 1e4f32);
    let x: Vec<f32> = (0..tq * d)
        .map(|i| ((i * 2654435761) as f32 * 1e-9).sin() * 0.7)
        .collect();
    let xt = cpu.from_cpu_f32(&x, vec![1, tq, d]).unwrap();
    let yc = cpu.rope_standard(&xt, pos, hd, theta).unwrap();
    let xg = gpu.from_cpu_f32(&x, vec![1, tq, d]).unwrap();
    let yg = gpu.rope_standard(&xg, pos, hd, theta).unwrap();
    approx(
        &gpu.to_cpu_f32(&yg).unwrap(),
        &cpu.to_cpu_f32(&yc).unwrap(),
        1e-4,
    );
}

#[test]
fn rmsnorm_head_rope_cuda_matches_cpu() {
    let Some(gpu) = make_gpu() else {
        eprintln!("no CUDA — skipping");
        return;
    };
    let cpu = CpuBackend;
    let (tq, h, hd, pos, theta, eps) = (9usize, 8usize, 128usize, 3usize, 1e4f32, 1e-5f32);
    let d = h * hd;
    let x: Vec<f32> = (0..tq * d)
        .map(|i| ((i * 1103515245) as f32 * 1e-9).cos() * 0.9)
        .collect();
    let w: Vec<f32> = (0..hd)
        .map(|i| 0.5 + 0.5 * (i as f32 / hd as f32))
        .collect();
    let xt = cpu.from_cpu_f32(&x, vec![1, tq, d]).unwrap();
    let wt = cpu.from_cpu_f32(&w, vec![hd]).unwrap();
    let yc = cpu
        .rmsnorm_head_rope(&xt, &wt, pos, hd, theta, h, eps)
        .unwrap();
    let xg = gpu.from_cpu_f32(&x, vec![1, tq, d]).unwrap();
    let wg = gpu.from_cpu_f32(&w, vec![hd]).unwrap();
    let yg = gpu
        .rmsnorm_head_rope(&xg, &wg, pos, hd, theta, h, eps)
        .unwrap();
    approx(
        &gpu.to_cpu_f32(&yg).unwrap(),
        &cpu.to_cpu_f32(&yc).unwrap(),
        2e-4,
    );
}

#[test]
fn rope_mrope_text_cuda_matches_cpu() {
    let Some(gpu) = make_gpu() else {
        eprintln!("no CUDA — skipping");
        return;
    };
    let cpu = CpuBackend;
    let (tq, d, hd, pos, theta) = (5usize, 512usize, 64usize, 11usize, 1e4f32);
    let sections = [8usize, 4, 4, 0];
    let x: Vec<f32> = (0..tq * d)
        .map(|i| ((i * 97) as f32 * 1e-3).sin())
        .collect();
    let xt = cpu.from_cpu_f32(&x, vec![1, tq, d]).unwrap();
    let yc = cpu.rope_mrope_text(&xt, pos, hd, theta, sections).unwrap();
    let xg = gpu.from_cpu_f32(&x, vec![1, tq, d]).unwrap();
    let yg = gpu.rope_mrope_text(&xg, pos, hd, theta, sections).unwrap();
    approx(
        &gpu.to_cpu_f32(&yg).unwrap(),
        &cpu.to_cpu_f32(&yc).unwrap(),
        1e-4,
    );
}

#[test]
fn rope_mrope_4d_cuda_matches_cpu() {
    let Some(gpu) = make_gpu() else {
        eprintln!("no CUDA — skipping");
        return;
    };
    let cpu = CpuBackend;
    let (tq, d, hd, theta) = (6usize, 512usize, 64usize, 1e4f32);
    let sections = [8usize, 4, 4, 0];
    let pos4d: Vec<u32> = (0..tq * 4).map(|i| (i * 7) as u32).collect();
    let x: Vec<f32> = (0..tq * d)
        .map(|i| ((i * 31) as f32 * 1e-3).cos())
        .collect();
    let xt = cpu.from_cpu_f32(&x, vec![1, tq, d]).unwrap();
    let yc = cpu.rope_mrope(&xt, &pos4d, hd, theta, sections).unwrap();
    let xg = gpu.from_cpu_f32(&x, vec![1, tq, d]).unwrap();
    let yg = gpu.rope_mrope(&xg, &pos4d, hd, theta, sections).unwrap();
    approx(
        &gpu.to_cpu_f32(&yg).unwrap(),
        &cpu.to_cpu_f32(&yc).unwrap(),
        1e-4,
    );
}

// ── N (iter-10): penalized_topk CPU↔CUDA parity (TTS sampler pipeline) ──

#[test]
fn penalized_topk_cuda_matches_cpu() {
    let Some(gpu) = make_gpu() else {
        eprintln!("no CUDA — skipping");
        return;
    };
    let cpu = CpuBackend;
    let v = 20000usize; // not a full vocab, but the same pipeline
    // A peaked distribution + a repeated-history token that must be pushed
    // down by the per-occurrence penalty.
    let logits: Vec<f32> = (0..v)
        .map(|i| {
            let center = ((i as i64) - 12345).abs() as f32;
            -center * 0.02 + ((i as f32) * 0.13).sin()
        })
        .collect();
    let mut bias = vec![0.0f32; v];
    bias[7000] = 200.0; // strong bias for a low-ranked token
    bias[12345] = -3.0; // suppress the peak
    let history: Vec<u32> = vec![42, 7000, 42, 42, 999, 7000, 42, 7000];
    let lg = cpu.from_cpu_f32(&logits, vec![1, 1, v]).unwrap();
    let bg = cpu.from_cpu_f32(&bias, vec![v]).unwrap();
    let cpu_res = cpu
        .penalized_topk(&lg, &bg, &history, 1.2, 0.1, 10, 1.0, 32)
        .unwrap();
    let lg2 = gpu.from_cpu_f32(&logits, vec![1, 1, v]).unwrap();
    let bg2 = gpu.from_cpu_f32(&bias, vec![v]).unwrap();
    let gpu_res = gpu
        .penalized_topk(&lg2, &bg2, &history, 1.2, 0.1, 10, 1.0, 32)
        .unwrap();
    assert_eq!(cpu_res.len(), gpu_res.len());
    let tol = 1e-3;
    let mut worst = 0.0f32;
    for (a, b) in cpu_res.iter().zip(&gpu_res) {
        assert_eq!(a.0, b.0, "index mismatch: {} vs {}", a.0, b.0);
        worst = worst.max((a.1 - b.1).abs());
        assert!(
            (a.1 - b.1).abs() <= tol,
            "prob mismatch: {} vs {}",
            a.1,
            b.1
        );
    }
    eprintln!("penalized_topk parity: worst Δprob = {worst}");
    // The repeated token 42 must NOT be in the top window (penalized down),
    // and the biased 7000 must be present.
    let idxs: Vec<u32> = gpu_res.iter().map(|(i, _)| *i).collect();
    assert!(
        !idxs.contains(&42),
        "repeated token not suppressed: {idxs:?}"
    );
    assert!(idxs.contains(&7000), "biased token missing: {idxs:?}");
}
