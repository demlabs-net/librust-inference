//! KV-f16 correctness: attn_ar_v2_f16 and kv_update_f16 vs the CPU f32
//! oracle. `cargo test --release --features "cuda cpu" --test kv_f16 -- --nocapture`.

use rust_inference::Backend;
use rust_inference::cpu::CpuBackend;
use rust_inference::cuda::CudaBackend;
use rust_inference::harness::Lcg;

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
