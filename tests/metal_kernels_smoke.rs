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
