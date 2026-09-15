//! KV-f16/q8 correctness: attn_ar_v2_f16 and kv_update_f16 vs the CPU f32
#![cfg(all(feature = "cuda", feature = "cpu"))]
//! oracle. `cargo test --release --features "cuda cpu" --test kv_f16 -- --nocapture`.

use rust_inference::Backend;
use rust_inference::cpu::CpuBackend;
use rust_inference::cuda::CudaBackend;
use rust_inference::harness::Lcg;

/// CPU q8_0 quantize — mirrors kv_update_q8 exactly: per-32-block
/// amax/127 scale, round-to-nearest-even (`__float2int_rn`).
fn quantize_q8(src: &[f32], d: usize) -> (Vec<i8>, Vec<f32>) {
    let nb = d / 32;
    let mut q = vec![0i8; src.len()];
    let mut sc = vec![0f32; src.len() / 32];
    for r in 0..src.len() / d {
        for blk in 0..nb {
            let mut amax = 0f32;
            for j in 0..32 {
                amax = amax.max(src[r * d + blk * 32 + j].abs());
            }
            sc[r * nb + blk] = amax / 127.0;
            if amax == 0.0 {
                continue;
            }
            let inv = 127.0 / amax;
            for j in 0..32 {
                q[r * d + blk * 32 + j] = (src[r * d + blk * 32 + j] * inv).round_ties_even() as i8;
            }
        }
    }
    (q, sc)
}

fn dequant_q8(q: &[i8], sc: &[f32]) -> Vec<f32> {
    q.iter()
        .enumerate()
        .map(|(i, &v)| f32::from(v) * sc[i / 32])
        .collect()
}

#[test]
fn attn_ar_f16_matches_f32() {
    let gpu = CudaBackend::new(0).unwrap();
    let cpu = CpuBackend;

    let b = 1usize;
    let tq = 3usize;
    let tkv = 5usize;
    let heads = 2usize;
    let kv_heads = 1usize;
    let hd = 4usize;

    let mut q = vec![0f32; b * tq * heads * hd];
    Lcg::new(1).fill(&mut q);
    let mut kv = vec![0f32; b * tkv * kv_heads * hd];
    Lcg::new(2).fill(&mut kv);
    let mask: Vec<f32> = (0..tq * tkv)
        .map(|i| if i % tkv <= i / tkv { 0.0 } else { -10000.0 })
        .collect();

    // CUDA f16 path: kv stored as half.
    let q_g = gpu.from_cpu_f32(&q, vec![b, tq, heads * hd]).unwrap();
    let kv_h: Vec<half::f16> = kv.iter().map(|&v| half::f16::from_f32(v)).collect();
    let k_g = gpu
        .from_cpu_f16(&kv_h, vec![b, tkv, kv_heads * hd])
        .unwrap();
    let v_g = gpu
        .from_cpu_f16(&kv_h, vec![b, tkv, kv_heads * hd])
        .unwrap();
    let m_g = gpu.from_cpu_f32(&mask, vec![tq, tkv]).unwrap();
    let y_g = gpu.zeros(vec![b, tq, heads * hd]).unwrap();
    gpu.attn_ar(&q_g, &k_g, &v_g, Some(&m_g), &y_g, heads, kv_heads, hd)
        .unwrap();
    gpu.synchronize().unwrap();
    let g = gpu.to_cpu_f32(&y_g).unwrap();

    // CPU f32 oracle with half-rounded kv (same quantization error).
    let kv_r: Vec<f32> = kv_h.iter().map(|&v| f32::from(v)).collect();
    let q_c = cpu.from_cpu_f32(&q, vec![b, tq, heads * hd]).unwrap();
    let k_c = cpu
        .from_cpu_f32(&kv_r, vec![b, tkv, kv_heads * hd])
        .unwrap();
    let v_c = cpu
        .from_cpu_f32(&kv_r, vec![b, tkv, kv_heads * hd])
        .unwrap();
    let m_c = cpu.from_cpu_f32(&mask, vec![tq, tkv]).unwrap();
    let y_c = cpu.zeros(vec![b, tq, heads * hd]).unwrap();
    cpu.attn_ar(&q_c, &k_c, &v_c, Some(&m_c), &y_c, heads, kv_heads, hd)
        .unwrap();
    let c = cpu.to_cpu_f32(&y_c).unwrap();

    let tol = 2e-3 + 2e-3 * c.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let mut bad = 0;
    for i in 0..g.len() {
        if (g[i] - c[i]).abs() > tol {
            bad += 1;
            if bad < 5 {
                eprintln!("attn_f16 [{i}] gpu {} cpu {}", g[i], c[i]);
            }
        }
    }
    assert_eq!(
        bad,
        0,
        "attn_ar_v2_f16 mismatch: {bad}/{}, tol {tol}",
        g.len()
    );
    println!("attn_ar f16 vs f32: ok ({} elems, tol {tol:.4})", g.len());
}

/// One attn_ar run for arbitrary K/V storage pairs; returns (gpu_out, cpu_out).
fn attn_ar_pair(
    k_store: &dyn Fn(&CudaBackend, &[f32], Vec<usize>) -> rust_inference::Tensor,
    v_store: &dyn Fn(&CudaBackend, &[f32], Vec<usize>) -> rust_inference::Tensor,
    k_cpu: &[f32],
    v_cpu: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let gpu = CudaBackend::new(0).unwrap();
    let cpu = CpuBackend;
    let b = 1usize;
    let tq = 3usize;
    let tkv = 5usize;
    let heads = 2usize;
    let kv_heads = 1usize;
    let hd = 64usize; // d must be % 32 == 0 for q8_0

    let mut q = vec![0f32; b * tq * heads * hd];
    Lcg::new(1).fill(&mut q);
    let mask: Vec<f32> = (0..tq * tkv)
        .map(|i| if i % tkv <= i / tkv { 0.0 } else { -10000.0 })
        .collect();

    let q_g = gpu.from_cpu_f32(&q, vec![b, tq, heads * hd]).unwrap();
    let k_g = k_store(&gpu, k_cpu, vec![b, tkv, kv_heads * hd]);
    let v_g = v_store(&gpu, v_cpu, vec![b, tkv, kv_heads * hd]);
    let m_g = gpu.from_cpu_f32(&mask, vec![tq, tkv]).unwrap();
    let y_g = gpu.zeros(vec![b, tq, heads * hd]).unwrap();
    gpu.attn_ar(&q_g, &k_g, &v_g, Some(&m_g), &y_g, heads, kv_heads, hd)
        .unwrap();
    gpu.synchronize().unwrap();
    let g = gpu.to_cpu_f32(&y_g).unwrap();

    let q_c = cpu.from_cpu_f32(&q, vec![b, tq, heads * hd]).unwrap();
    let k_c = cpu
        .from_cpu_f32(k_cpu, vec![b, tkv, kv_heads * hd])
        .unwrap();
    let v_c = cpu
        .from_cpu_f32(v_cpu, vec![b, tkv, kv_heads * hd])
        .unwrap();
    let m_c = cpu.from_cpu_f32(&mask, vec![tq, tkv]).unwrap();
    let y_c = cpu.zeros(vec![b, tq, heads * hd]).unwrap();
    cpu.attn_ar(&q_c, &k_c, &v_c, Some(&m_c), &y_c, heads, kv_heads, hd)
        .unwrap();
    let c = cpu.to_cpu_f32(&y_c).unwrap();
    (g, c)
}

fn assert_close(g: &[f32], c: &[f32], what: &str) {
    let tol = 2e-3 + 2e-3 * c.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let mut bad = 0;
    for i in 0..g.len() {
        // NaN always fails the comparison — count it explicitly so a
        // broken kernel can't pass vacuously.
        let ok = g[i].is_finite() && c[i].is_finite() && (g[i] - c[i]).abs() <= tol;
        if !ok {
            bad += 1;
            if bad < 5 {
                eprintln!("{what} [{i}] gpu {} cpu {}", g[i], c[i]);
            }
        }
    }
    assert_eq!(bad, 0, "{what} mismatch: {bad}/{}, tol {tol}", g.len());
    println!("{what}: ok ({} elems, tol {tol:.4})", g.len());
}

#[test]
fn attn_ar_q8q8_matches_f32() {
    let mut kv = vec![0f32; 5 * 64];
    Lcg::new(2).fill(&mut kv);
    let (kq, ksc) = quantize_q8(&kv, 64);
    let (vq, vsc) = quantize_q8(&kv, 64);
    let (g, c) = attn_ar_pair(
        &|gpu, _, sh| gpu.from_cpu_i8(&kq, &ksc, sh.clone()).unwrap(),
        &|gpu, _, sh| gpu.from_cpu_i8(&vq, &vsc, sh.clone()).unwrap(),
        &dequant_q8(&kq, &ksc),
        &dequant_q8(&vq, &vsc),
    );
    assert_close(&g, &c, "attn_ar q8q8 vs f32");
}

#[test]
fn attn_ar_q8f16_matches_f32() {
    let mut kv = vec![0f32; 5 * 64];
    Lcg::new(2).fill(&mut kv);
    let (kq, ksc) = quantize_q8(&kv, 64);
    let vh: Vec<half::f16> = kv.iter().map(|&v| half::f16::from_f32(v)).collect();
    let vr: Vec<f32> = vh.iter().map(|&v| f32::from(v)).collect();
    let (g, c) = attn_ar_pair(
        &|gpu, _, sh| gpu.from_cpu_i8(&kq, &ksc, sh.clone()).unwrap(),
        &|gpu, _, sh| gpu.from_cpu_f16(&vh, sh.clone()).unwrap(),
        &dequant_q8(&kq, &ksc),
        &vr,
    );
    assert_close(&g, &c, "attn_ar q8(f16) vs f32");
}

#[test]
fn attn_ar_f16q8_matches_f32() {
    let mut kv = vec![0f32; 5 * 64];
    Lcg::new(2).fill(&mut kv);
    let (vq, vsc) = quantize_q8(&kv, 64);
    let kh: Vec<half::f16> = kv.iter().map(|&v| half::f16::from_f32(v)).collect();
    let kr: Vec<f32> = kh.iter().map(|&v| f32::from(v)).collect();
    let (g, c) = attn_ar_pair(
        &|gpu, _, sh| gpu.from_cpu_f16(&kh, sh.clone()).unwrap(),
        &|gpu, _, sh| gpu.from_cpu_i8(&vq, &vsc, sh.clone()).unwrap(),
        &kr,
        &dequant_q8(&vq, &vsc),
    );
    assert_close(&g, &c, "attn_ar f16(q8) vs f32");
}

#[test]
fn kv_update_q8_roundtrip() {
    let gpu = CudaBackend::new(0).unwrap();
    let b = 1usize;
    let tq = 4usize;
    let d = 64usize;
    let cap = 8usize;
    let pos = 3usize;

    let mut src = vec![0f32; b * tq * d];
    Lcg::new(3).fill(&mut src);
    let s_g = gpu.from_cpu_f32(&src, vec![b, tq, d]).unwrap();
    let zeros = vec![0i8; b * cap * d];
    let sc0 = vec![1.0f32; b * cap * (d / 32)];
    let dst = gpu.from_cpu_i8(&zeros, &sc0, vec![b, cap, d]).unwrap();
    gpu.kv_update(&dst, &s_g, pos).unwrap();
    gpu.synchronize().unwrap();
    let got = gpu.to_cpu_f32(&dst).unwrap();

    let (q, qsc) = quantize_q8(&src, d);
    for ti in 0..tq {
        for j in 0..d {
            let want = f32::from(q[ti * d + j]) * qsc[ti * (d / 32) + j / 32];
            let gv = got[(pos + ti) * d + j];
            assert!(
                (gv - want).abs() < 1e-6,
                "kv_update_q8 row {} col {}: got {gv}, want {want}",
                pos + ti,
                j
            );
        }
    }
    println!("kv_update q8: ok");
}

#[test]
fn kv_update_q8_grow_copy() {
    let gpu = CudaBackend::new(0).unwrap();
    let b = 1usize;
    let tq = 4usize;
    let d = 64usize;
    let cap = 8usize;
    let pos = 2usize;

    let mut src = vec![0f32; b * tq * d];
    Lcg::new(4).fill(&mut src);
    let (sq, ssc) = quantize_q8(&src, d);
    let s_g = gpu.from_cpu_i8(&sq, &ssc, vec![b, tq, d]).unwrap();
    let zeros = vec![0i8; b * cap * d];
    let sc0 = vec![0.0f32; b * cap * (d / 32)];
    let dst = gpu.from_cpu_i8(&zeros, &sc0, vec![b, cap, d]).unwrap();
    gpu.kv_update(&dst, &s_g, pos).unwrap();
    gpu.synchronize().unwrap();
    let got = gpu.to_cpu_f32(&dst).unwrap();
    let want = dequant_q8(&sq, &ssc);
    for ti in 0..tq {
        for j in 0..d {
            let gv = got[(pos + ti) * d + j];
            let w = want[ti * d + j];
            assert!(
                (gv - w).abs() < 1e-6,
                "kv_update_q8_q8 row {} col {}: got {gv}, want {w}",
                pos + ti,
                j
            );
        }
    }
    println!("kv_update q8→q8 grow: ok");
}

/// Long-window attention across chunk boundaries (the v2 kernels capped at
/// 4096 shared scores; v4 chunks — this also validates the cross-chunk
/// online-softmax combine).
#[test]
fn attn_ar_long_window() {
    let gpu = CudaBackend::new(0).unwrap();
    let cpu = CpuBackend;
    let b = 1usize;
    let tq = 2usize;
    let tkv = 5000usize; // 79 chunks of 64
    let heads = 2usize;
    let kv_heads = 1usize;
    let hd = 64usize;

    let mut q = vec![0f32; b * tq * heads * hd];
    Lcg::new(9).fill(&mut q);
    let mut kv = vec![0f32; b * tkv * kv_heads * hd];
    Lcg::new(10).fill(&mut kv);
    let mask: Vec<f32> = (0..tq * tkv)
        .map(|i| if i % tkv <= i / tkv { 0.0 } else { -10000.0 })
        .collect();

    let q_g = gpu.from_cpu_f32(&q, vec![b, tq, heads * hd]).unwrap();
    let k_g = gpu.from_cpu_f32(&kv, vec![b, tkv, kv_heads * hd]).unwrap();
    let m_g = gpu.from_cpu_f32(&mask, vec![tq, tkv]).unwrap();
    let y_g = gpu.zeros(vec![b, tq, heads * hd]).unwrap();
    gpu.attn_ar(&q_g, &k_g, &k_g, Some(&m_g), &y_g, heads, kv_heads, hd)
        .unwrap();
    gpu.synchronize().unwrap();
    let g = gpu.to_cpu_f32(&y_g).unwrap();

    let q_c = cpu.from_cpu_f32(&q, vec![b, tq, heads * hd]).unwrap();
    let k_c = cpu.from_cpu_f32(&kv, vec![b, tkv, kv_heads * hd]).unwrap();
    let m_c = cpu.from_cpu_f32(&mask, vec![tq, tkv]).unwrap();
    let y_c = cpu.zeros(vec![b, tq, heads * hd]).unwrap();
    cpu.attn_ar(&q_c, &k_c, &k_c, Some(&m_c), &y_c, heads, kv_heads, hd)
        .unwrap();
    let c = cpu.to_cpu_f32(&y_c).unwrap();

    let tol = 2e-3 + 2e-3 * c.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let mut bad = 0;
    for i in 0..g.len() {
        let ok = g[i].is_finite() && c[i].is_finite() && (g[i] - c[i]).abs() <= tol;
        if !ok {
            bad += 1;
            if bad < 5 {
                eprintln!("attn_long [{i}] gpu {} cpu {}", g[i], c[i]);
            }
        }
    }
    assert_eq!(
        bad,
        0,
        "attn_ar long window mismatch: {bad}/{}, tol {tol}",
        g.len()
    );
    println!(
        "attn_ar long window (tkv={tkv}): ok ({} elems, tol {tol:.4})",
        g.len()
    );
}

#[test]
fn kv_update_f16_roundtrip() {
    let gpu = CudaBackend::new(0).unwrap();
    let b = 1usize;
    let tq = 4usize;
    let d = 6usize;
    let cap = 8usize;
    let pos = 3usize;

    let mut src = vec![0f32; b * tq * d];
    Lcg::new(3).fill(&mut src);
    let s_g = gpu.from_cpu_f32(&src, vec![b, tq, d]).unwrap();
    // f16 dst: create via from_cpu_f16 zeros.
    let zeros = vec![half::f16::from_f32(0.0); b * cap * d];
    let dst = gpu.from_cpu_f16(&zeros, vec![b, cap, d]).unwrap();
    gpu.kv_update(&dst, &s_g, pos).unwrap();
    gpu.synchronize().unwrap();
    let got = gpu.to_cpu_f32(&dst).unwrap();
    // Expect src in rows pos..pos+tq, quantized to f16.
    for ti in 0..tq {
        for j in 0..d {
            let want = half::f16::from_f32(src[ti * d + j]);
            let gv = got[(pos + ti) * d + j];
            let ok = (gv - f32::from(want)).abs() < 1e-6;
            assert!(
                ok,
                "kv_update_f16 row {} col {}: got {gv}, want {}",
                pos + ti,
                j,
                f32::from(want)
            );
        }
    }
    println!("kv_update f16: ok");
}
