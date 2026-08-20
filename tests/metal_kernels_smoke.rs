//! Smoke test for the Metal engine kernels (macOS only).
#![cfg(target_os = "macos")]

use rust_inference::metal::{GpuTensor, MetalCtx};

#[test]
fn gelu_tanh_writes_output() {
    let ctx = MetalCtx::new().unwrap();
    let x = GpuTensor::from_cpu_f32(&ctx.device, &[1.0, 2.0, 3.0, 10.0], vec![4]).unwrap();
    let y = GpuTensor::zeros(&ctx.device, vec![4]).unwrap();
    ctx.gelu_tanh_f32(&x, &y).unwrap();
    let v = y.to_cpu_f32(&ctx).unwrap();
    eprintln!("gelu out: {v:?}");
    assert!(v.iter().all(|a| !a.is_nan()), "gelu produced NaN: {v:?}");
    assert!(v[3] > 9.0, "gelu(10) expected ~10, got {}", v[3]);
}

#[test]
fn tanh_writes_output() {
    let ctx = MetalCtx::new().unwrap();
    let x = GpuTensor::from_cpu_f32(&ctx.device, &[0.5, 44.45, -3.0], vec![3]).unwrap();
    let y = GpuTensor::zeros(&ctx.device, vec![3]).unwrap();
    ctx.tanh_f32(&x, &y).unwrap();
    let v = y.to_cpu_f32(&ctx).unwrap();
    eprintln!("tanh out: {v:?}");
    assert!(!v[0].is_nan());
    assert!((v[0] - 0.4621).abs() < 1e-3, "tanh(0.5) = {}", v[0]);
}

#[test]
fn snake_writes_output() {
    let ctx = MetalCtx::new().unwrap();
    let x = GpuTensor::from_cpu_f32(&ctx.device, &[1.0, 2.0], vec![1, 2, 1]).unwrap();
    let a = GpuTensor::from_cpu_f32(&ctx.device, &[2.0], vec![1]).unwrap();
    let y = GpuTensor::zeros(&ctx.device, vec![1, 2, 1]).unwrap();
    ctx.snake_f32(&x, &a, &y).unwrap();
    let v = y.to_cpu_f32(&ctx).unwrap();
    eprintln!("snake out: {v:?}");
    assert!(v.iter().all(|x| !x.is_nan()));
}

#[test]
fn conv_transpose_smoke() {
    let ctx = MetalCtx::new().unwrap();
    // x (1, 2, 2): [ [1,2], [3,4] ]; w (2, 2, 2) identity-ish; stride 2
    let x = GpuTensor::from_cpu_f32(&ctx.device, &[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]).unwrap();
    let w = GpuTensor::from_cpu_f32(&ctx.device, &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], vec![2, 2, 2]).unwrap();
    let y = GpuTensor::zeros(&ctx.device, vec![1, 4, 2]).unwrap();
    ctx.conv_transpose1d_f32(&x, &w, None, 2, &y).unwrap();
    let v = y.to_cpu_f32(&ctx).unwrap();
    eprintln!("convT out: {v:?}");
    assert!(v.iter().all(|x| !x.is_nan()));
}

#[test]
fn embedding_and_rope_smoke() {
    let ctx = MetalCtx::new().unwrap();
    let table = GpuTensor::from_cpu_f32(
        &ctx.device,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        vec![4, 2],
    )
    .unwrap();
    let y = GpuTensor::zeros(&ctx.device, vec![1, 3, 2]).unwrap();
    ctx.embedding_f32(&[0, 2, 3], &table, &y).unwrap();
    let v = y.to_cpu_f32(&ctx).unwrap();
    eprintln!("embed out: {v:?}");
    assert_eq!(&v[..2], &[1.0, 2.0]);
    assert_eq!(&v[2..4], &[5.0, 6.0]);

    // rope interleaved: x [1,1,4], cos/sin [1,2]
    let x = GpuTensor::from_cpu_f32(&ctx.device, &[1.0, 2.0, 3.0, 4.0], vec![1, 1, 4]).unwrap();
    let cos = GpuTensor::from_cpu_f32(&ctx.device, &[1.0, 0.0], vec![1, 2]).unwrap();
    let sin = GpuTensor::from_cpu_f32(&ctx.device, &[0.0, 1.0], vec![1, 2]).unwrap();
    let yr = GpuTensor::zeros(&ctx.device, vec![1, 1, 4]).unwrap();
    ctx.rope_interleaved_f32(&x, &cos, &sin, &yr, 4, 0).unwrap();
    let vr = yr.to_cpu_f32(&ctx).unwrap();
    eprintln!("rope out: {vr:?}");
    // pair (1,2): (1*c - 2*s, 1*s + 2*c) = (1, 2); pair (3,4): (3*0-4*1, 3*1+4*0) = (-4, 3)
    assert_eq!(&vr[..2], &[1.0, 2.0]);
    assert_eq!(&vr[2..], &[-4.0, 3.0]);
}

/// Quantized GEMM (fp32 activations × f16/bf16 weights) matches the fp32
/// reference within the narrow-format tolerance. Weights here stay exactly
/// representable, so the tolerance is tight (1e-3 guards fp32 accumulation
/// ordering).
#[test]
fn gemm_nt_quant_smoke() {
    let ctx = MetalCtx::new().unwrap();
    let m = 8usize;
    let k = 16usize;
    let n = 12usize;
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect();
    let b: Vec<f32> = (0..n * k).map(|i| ((i % 11) as f32 - 5.0) * 0.5).collect();
    // reference: C = A · Bᵀ
    let mut ref_c = vec![0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0f32;
            for ki in 0..k {
                acc += a[mi * k + ki] * b[ni * k + ki];
            }
            ref_c[mi * n + ni] = acc;
        }
    }
    let a_t = GpuTensor::from_cpu_f32(&ctx.device, &a, vec![m, k]).unwrap();
    let c = GpuTensor::zeros(&ctx.device, vec![m, n]).unwrap();

    for (name, w) in [
        ("f16", half::f16::from_f32),
        // bf16 rounds mantissa to 8 bits — compare with looser bound
    ] {
        let _ = name;
        let w16: Vec<half::f16> = b.iter().map(|&v| w(v)).collect();
        let b_t = GpuTensor::from_cpu_f16(&ctx.device, &w16, vec![n, k]).unwrap();
        assert_eq!(b_t.dtype, rust_inference::metal::GpuDtype::F16);
        ctx.gemm_nt_f32(&a_t, &b_t, &c).unwrap();
        let got = c.to_cpu_f32(&ctx).unwrap();
        for (gi, gv) in got.iter().enumerate() {
            assert!(
                (gv - ref_c[gi]).abs() < 1e-2,
                "gemm_nt_f16 mismatch at {gi}: {gv} vs {}",
                ref_c[gi]
            );
        }
    }

    // bf16: convert explicitly, allow the 8-bit-mantissa rounding
    let wbf: Vec<half::bf16> = b.iter().map(|&v| half::bf16::from_f32(v)).collect();
    let b_t = GpuTensor::from_cpu_bf16(&ctx.device, &wbf, vec![n, k]).unwrap();
    assert_eq!(b_t.dtype, rust_inference::metal::GpuDtype::BF16);
    ctx.gemm_nt_f32(&a_t, &b_t, &c).unwrap();
    let got = c.to_cpu_f32(&ctx).unwrap();
    for (gi, gv) in got.iter().enumerate() {
        assert!(
            (gv - ref_c[gi]).abs() < 0.5,
            "gemm_nt_bf16 mismatch at {gi}: {gv} vs {}",
            ref_c[gi]
        );
    }

    // plain gemm (B row-major K,N) with f16 weights
    let bn: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32 - 6.0) * 0.5).collect();
    let mut ref_c2 = vec![0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0f32;
            for ki in 0..k {
                acc += a[mi * k + ki] * bn[ki * n + ni];
            }
            ref_c2[mi * n + ni] = acc;
        }
    }
    let w16: Vec<half::f16> = bn.iter().map(|&v| half::f16::from_f32(v)).collect();
    let b_row = GpuTensor::from_cpu_f16(&ctx.device, &w16, vec![k, n]).unwrap();
    ctx.gemm_f32(&a_t, &b_row, &c).unwrap();
    let got = c.to_cpu_f32(&ctx).unwrap();
    for (gi, gv) in got.iter().enumerate() {
        assert!(
            (gv - ref_c2[gi]).abs() < 1e-2,
            "gemm_f16 mismatch at {gi}: {gv} vs {}",
            ref_c2[gi]
        );
    }
}
