//! Metal engine microbenchmark: our fused kernels vs candle's per-op path.
//!
//! Run: `cargo run --release --example metal_bench` (macOS).
//!
//! Answers the question that started the engine: is candle-Metal's per-op
//! command-buffer overhead really the bottleneck for conv-heavy models?

use candle_core::{Device, Tensor};
use rust_inference::metal::{Conv1dParams, GpuTensor, MetalCtx};
use std::time::Instant;

/// CPU reference for conv1d (naive triple loop) — correctness oracle.
fn conv1d_cpu_ref(
    x: &[f32],
    b: usize,
    t: usize,
    cin: usize,
    w: &[f32],
    cout: usize,
    k: usize,
    bias: &[f32],
    stride: usize,
    pad: usize,
) -> Vec<f32> {
    let t_out = (t + 2 * pad - (k - 1) - 1) / stride + 1;
    let mut y = vec![0f32; b * t_out * cout];
    for bb in 0..b {
        for tt in 0..t_out {
            for co in 0..cout {
                let mut acc = bias[co];
                for ci in 0..cin {
                    for kk in 0..k {
                        let t_in = tt as isize * stride as isize - pad as isize + kk as isize;
                        if t_in >= 0 && (t_in as usize) < t {
                            acc += x[(bb * t + t_in as usize) * cin + ci]
                                * w[(co * cin + ci) * k + kk];
                        }
                    }
                }
                y[(bb * t_out + tt) * cout + co] = acc;
            }
        }
    }
    y
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = MetalCtx::new()?;
    let candle_dev = Device::new_metal(0)?;
    println!("=== Vassista Metal engine vs candle-Metal (M3 Max) ===\n");

    // t-one-like shapes: X [1, 300, 256], W [256, 256, 7], bias [256]
    let (b, t, cin) = (1usize, 300usize, 256usize);
    let (cout, k) = (256usize, 7usize);
    let x_data: Vec<f32> = (0..b * t * cin).map(|i| (i as f32 * 0.001).sin()).collect();
    let w_data: Vec<f32> = (0..cout * cin * k)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
        .collect();
    let bias_data: Vec<f32> = (0..cout).map(|i| (i as f32 - 128.0) * 0.001).collect();

    // ── correctness: ours vs CPU reference ──
    let gx = GpuTensor::from_cpu_f32(&ctx.device, &x_data, vec![b, t, cin])?;
    let gw = GpuTensor::from_cpu_f32(&ctx.device, &w_data, vec![cout, cin, k])?;
    let gb = GpuTensor::from_cpu_f32(&ctx.device, &bias_data, vec![cout])?;
    let t_out = t; // stride=1, pad=3, k=7 → same length
    let gy = GpuTensor::zeros(&ctx.device, vec![b, t_out, cout])?;
    let params = Conv1dParams {
        stride: 1,
        pad: 3,
        dilation: 1,
    };
    ctx.conv1d_f32(&gx, &gw, Some(&gb), params, &gy)?;
    let ours = gy.to_cpu_f32(&ctx).unwrap();
    let reference = conv1d_cpu_ref(&x_data, b, t, cin, &w_data, cout, k, &bias_data, 1, 3);
    let diff = max_abs_diff(&ours, &reference);
    println!(
        "conv1d correctness: max|ours − ref| = {diff:.2e} {}",
        if diff < 1e-3 { "✓" } else { "✗ FAIL" }
    );

    // ── conv2d correctness vs candle ──
    {
        use candle_nn::Conv2dConfig;
        let x2: Vec<f32> = (0..1 * 1 * 50 * 64).map(|i| (i as f32 * 0.01).sin()).collect();
        let w2: Vec<f32> = (0..32 * 1 * 11 * 21).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
        let b2: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.001).collect();
        let gx2 = GpuTensor::from_cpu_f32(&ctx.device, &x2, vec![1, 1, 50, 64])?;
        let gw2 = GpuTensor::from_cpu_f32(&ctx.device, &w2, vec![32, 1, 11, 21])?;
        let gb2 = GpuTensor::from_cpu_f32(&ctx.device, &b2, vec![32])?;
        let gy2 = GpuTensor::zeros(&ctx.device, vec![1, 32, 40, 44])?;
        ctx.conv2d_f32(&gx2, &gw2, Some(&gb2), 1, 1, 0, false, &gy2)?;
        let xc2 = Tensor::from_vec(x2.clone(), (1, 1, 50, 64), &candle_dev)?;
        let wc2 = Tensor::from_vec(w2.clone(), (32, 1, 11, 21), &candle_dev)?;
        let bc2 = Tensor::from_vec(b2.clone(), 32, &candle_dev)?;
        let yc2 = xc2.conv2d(&wc2, 0, 1, 1, 1)?; // pad=0, stride=1
        let yc2 = yc2.broadcast_add(&bc2.reshape((1, 32, 1, 1))?)?;
        let yc2v: Vec<f32> = yc2.flatten_all()?.to_vec1()?;
        let d2 = max_abs_diff(&gy2.to_cpu_f32(&ctx).unwrap(), &yc2v);
        println!("conv2d correctness: max|ours − candle| = {d2:.2e} {}", if d2 < 1e-3 { "✓" } else { "✗ FAIL" });
    }

    // ── conv2d with LEFT pad (padT=10) vs candle manual left-pad + conv ──
    {
        let x2: Vec<f32> = (0..1 * 1 * 30 * 64).map(|i| (i as f32 * 0.01).sin()).collect();
        let w2: Vec<f32> = (0..32 * 1 * 11 * 21).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
        let b2: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.001).collect();
        let gx2 = GpuTensor::from_cpu_f32(&ctx.device, &x2, vec![1, 1, 30, 64])?;
        let gw2 = GpuTensor::from_cpu_f32(&ctx.device, &w2, vec![32, 1, 11, 21])?;
        let gb2 = GpuTensor::from_cpu_f32(&ctx.device, &b2, vec![32])?;
        let gy2 = GpuTensor::zeros(&ctx.device, vec![1, 32, 30, 44])?;
        ctx.conv2d_f32(&gx2, &gw2, Some(&gb2), 1, 1, 10, false, &gy2)?;
        let xc2 = Tensor::from_vec(x2.clone(), (1, 1, 30, 64), &candle_dev)?;
        let wc2 = Tensor::from_vec(w2.clone(), (32, 1, 11, 21), &candle_dev)?;
        let bc2 = Tensor::from_vec(b2.clone(), 32, &candle_dev)?;
        let pad = Tensor::zeros((1, 1, 10, 64), candle_core::DType::F32, &candle_dev)?;
        let xp = Tensor::cat(&[&pad, &xc2], 2)?;
        let yc2 = xp.conv2d(&wc2, 0, 1, 1, 1)?;
        let yc2 = yc2.broadcast_add(&bc2.reshape((1, 32, 1, 1))?)?;
        let yc2v: Vec<f32> = yc2.flatten_all()?.to_vec1()?;
        let d2 = max_abs_diff(&gy2.to_cpu_f32(&ctx).unwrap(), &yc2v);
        println!("conv2d-pad correctness: max|ours − candle| = {d2:.2e} {}", if d2 < 1e-3 { "✓" } else { "✗ FAIL" });
    }

    // ── BN folding: folded conv+bias vs candle (conv + manual BN eval) ──
    {
        let (cout, cin, kt, kf) = (32usize, 1usize, 11usize, 21usize);
        let mut w = vec![0f32; cout * cin * kt * kf];
        let mut bias = vec![0f32; cout];
        for i in 0..w.len() {
            w[i] = ((i % 17) as f32 - 8.0) * 0.01;
        }
        for i in 0..cout {
            bias[i] = (i as f32 - 16.0) * 0.01;
        }
        let bn_w: Vec<f32> = (0..cout).map(|i| 0.5 + (i as f32) * 0.01).collect();
        let bn_b: Vec<f32> = (0..cout).map(|i| -0.1 + (i as f32) * 0.005).collect();
        let bn_m: Vec<f32> = (0..cout).map(|i| (i as f32) * 0.02).collect();
        let bn_v: Vec<f32> = (0..cout).map(|i| 0.8 + (i as f32) * 0.01).collect();
        let eps = 1e-5f64;
        // fold (same as fold_all_bn):
        let mut wf = w.clone();
        let mut bf = bias.clone();
        for co in 0..cout {
            let scale = bn_w[co] as f64 / (bn_v[co] as f64 + eps).sqrt();
            for i in 0..cin * kt * kf {
                wf[co * cin * kt * kf + i] = (wf[co * cin * kt * kf + i] as f64 * scale) as f32;
            }
            bf[co] = ((bf[co] as f64 - bn_m[co] as f64) * scale + bn_b[co] as f64) as f32;
        }
        let x2: Vec<f32> = (0..1 * 1 * 30 * 64).map(|i| (i as f32 * 0.01).sin()).collect();
        let gx2 = GpuTensor::from_cpu_f32(&ctx.device, &x2, vec![1, 1, 30, 64])?;
        let gw2 = GpuTensor::from_cpu_f32(&ctx.device, &wf, vec![cout, cin, kt, kf])?;
        let gb2 = GpuTensor::from_cpu_f32(&ctx.device, &bf, vec![cout])?;
        let gy2 = GpuTensor::zeros(&ctx.device, vec![1, cout, 30, 44])?;
        ctx.conv2d_f32(&gx2, &gw2, Some(&gb2), 1, 1, 10, false, &gy2)?;
        // candle reference: conv + BN eval
        let xc2 = Tensor::from_vec(x2.clone(), (1, 1, 30, 64), &candle_dev)?;
        let pad = Tensor::zeros((1, 1, 10, 64), candle_core::DType::F32, &candle_dev)?;
        let xp = Tensor::cat(&[&pad, &xc2], 2)?;
        let wc2 = Tensor::from_vec(w.clone(), (cout, cin, kt, kf), &candle_dev)?;
        let yc2 = xp.conv2d(&wc2, 0, 1, 1, 1)?;
        let yc2 = yc2.broadcast_add(&Tensor::from_vec(bias.clone(), cout, &candle_dev)?.reshape((1, cout, 1, 1))?)?;
        let scale = Tensor::from_vec(bn_w.clone(), cout, &candle_dev)?
            .broadcast_div(&(Tensor::from_vec(bn_v.clone(), cout, &candle_dev)?.affine(1.0, eps))?.sqrt()?)?;
        let yc2 = yc2.broadcast_sub(&Tensor::from_vec(bn_m.clone(), cout, &candle_dev)?.reshape((1, cout, 1, 1))?)?
            .broadcast_mul(&scale.reshape((1, cout, 1, 1))?)?
            .broadcast_add(&Tensor::from_vec(bn_b.clone(), cout, &candle_dev)?.reshape((1, cout, 1, 1))?)?;
        let yc2v: Vec<f32> = yc2.flatten_all()?.to_vec1()?;
        let d2 = max_abs_diff(&gy2.to_cpu_f32(&ctx).unwrap(), &yc2v);
        println!("bn-fold correctness: max|ours − candle| = {d2:.2e} {}", if d2 < 1e-3 { "✓" } else { "✗ FAIL" });
    }

    // ── rmsnorm correctness vs candle ──
    {
        let xr: Vec<f32> = (0..40 * 384).map(|i| (i as f32 * 0.007).cos()).collect();
        let wr: Vec<f32> = (0..384).map(|i| 0.5 + (i as f32) * 0.001).collect();
        let gxr = GpuTensor::from_cpu_f32(&ctx.device, &xr, vec![40, 384])?;
        let gwr = GpuTensor::from_cpu_f32(&ctx.device, &wr, vec![384])?;
        let gyr = GpuTensor::zeros(&ctx.device, vec![40, 384])?;
        ctx.rmsnorm_f32(&gxr, &gwr, &gyr, 1e-8)?;
        let xcr = Tensor::from_vec(xr.clone(), (40, 384), &candle_dev)?;
        let wcr = Tensor::from_vec(wr.clone(), 384, &candle_dev)?;
        let mean = xcr.sqr()?.mean_keepdim(1)?;
        let normed = xcr.broadcast_div(&(mean + 1e-8)?.sqrt()?)?;
        let ycr = normed.broadcast_mul(&wcr)?;
        let ycrv: Vec<f32> = ycr.flatten_all()?.to_vec1()?;
        let dr = max_abs_diff(&gyr.to_cpu_f32(&ctx).unwrap(), &ycrv);
        println!("rmsnorm correctness: max|ours − candle| = {dr:.2e} {}", if dr < 1e-3 { "✓" } else { "✗ FAIL" });
    }

    // ── gemm_nt correctness vs candle ──
    {
        let xg: Vec<f32> = (0..40 * 2176).map(|i| (i as f32 * 0.003).sin()).collect();
        let wg: Vec<f32> = (0..384 * 2176).map(|i| ((i % 11) as f32 - 5.0) * 0.002).collect();
        let gxg = GpuTensor::from_cpu_f32(&ctx.device, &xg, vec![40, 2176])?;
        let gwg = GpuTensor::from_cpu_f32(&ctx.device, &wg, vec![384, 2176])?;
        let gyg = GpuTensor::zeros(&ctx.device, vec![40, 384])?;
        ctx.gemm_nt_f32(&gxg, &gwg, &gyg)?;
        let xcg = Tensor::from_vec(xg.clone(), (40, 2176), &candle_dev)?;
        let wcg = Tensor::from_vec(wg.clone(), (384, 2176), &candle_dev)?;
        let ycg = xcg.matmul(&wcg.t()?)?;
        let ycgv: Vec<f32> = ycg.flatten_all()?.to_vec1()?;
        let dg = max_abs_diff(&gyg.to_cpu_f32(&ctx).unwrap(), &ycgv);
        println!("gemm_nt correctness: max|ours − candle| = {dg:.2e} {}", if dg < 1e-3 { "✓" } else { "✗ FAIL" });
    }

    // ── glu vs candle ──
    {
        let rows = 40usize;
        let d = 384usize;
        let xg: Vec<f32> = (0..rows * 2 * d).map(|i| (i as f32 * 0.011).sin()).collect();
        let gx = GpuTensor::from_cpu_f32(&ctx.device, &xg, vec![rows, 2 * d])?;
        let gy = GpuTensor::zeros(&ctx.device, vec![rows, d])?;
        ctx.glu_f32(&gx, &gy)?;
        let xc = Tensor::from_vec(xg.clone(), (rows, 2 * d), &candle_dev)?;
        let gate = xc.narrow(1, 0, d)?;
        let value = xc.narrow(1, d, d)?;
        let sig = value.affine(0.5, 0.0)?.tanh()?.affine(0.5, 0.5)?;
        let yc = gate.broadcast_mul(&sig)?;
        let dg = max_abs_diff(&gy.to_cpu_f32(&ctx).unwrap(), &yc.flatten_all()?.to_vec1()?);
        println!("glu correctness: max|ours − candle| = {dg:.2e} {}", if dg < 1e-3 { "✓" } else { "✗ FAIL" });
    }

    // ── conv1d_groups (depthwise, time-major) vs candle ──
    {
        let (b, t, d, k) = (1usize, 40usize, 384usize, 31usize);
        let xg: Vec<f32> = (0..b * t * d).map(|i| (i as f32 * 0.013).cos()).collect();
        let wg: Vec<f32> = (0..d * k).map(|i| ((i % 9) as f32 - 4.0) * 0.01).collect();
        let bg: Vec<f32> = (0..d).map(|i| (i as f32 - 192.0) * 0.001).collect();
        let gx = GpuTensor::from_cpu_f32(&ctx.device, &xg, vec![b, t, d])?;
        let gw = GpuTensor::from_cpu_f32(&ctx.device, &wg, vec![d, 1, k])?;
        let gb = GpuTensor::from_cpu_f32(&ctx.device, &bg, vec![d])?;
        let gy = GpuTensor::zeros(&ctx.device, vec![b, t, d])?;
        ctx.conv1d_groups_f32(&gx, &gw, Some(&gb), Conv1dParams { stride: 1, pad: 30, dilation: 1 }, 0, true, true, &gy)?;
        // candle expects channel-major [B, D, T] — transpose the time-major data
        let mut xc_data = vec![0f32; b * t * d];
        for tt in 0..t {
            for dd in 0..d {
                xc_data[dd * t + tt] = xg[tt * d + dd];
            }
        }
        let xc = Tensor::from_vec(xc_data, (b, d, t), &candle_dev)?;
        let wc = Tensor::from_vec(wg.clone(), (d, 1, k), &candle_dev)?;
        let pad = Tensor::zeros((b, d, 30), candle_core::DType::F32, &candle_dev)?;
        let xp = Tensor::cat(&[&pad, &xc], 2)?;
        let yc = xp.conv1d(&wc, 0, 1, 1, d)?; // groups=d
        let yc = yc.broadcast_add(&Tensor::from_vec(bg.clone(), d, &candle_dev)?.reshape((1, d, 1))?)?;
        // my output is time-major [B,T,D]; candle's is [B,D,T] — transpose for compare
        let yc_t = yc.transpose(1, 2)?;
        let dg = max_abs_diff(&gy.to_cpu_f32(&ctx).unwrap(), &yc_t.flatten_all()?.to_vec1()?);
        println!("depthwise correctness: max|ours − candle| = {dg:.2e} {}", if dg < 1e-3 { "✓" } else { "✗ FAIL" });
    }

    // ── mhsa leader (no mask) vs candle ──
    {
        let (b, t, d, h) = (1usize, 16usize, 384usize, 8usize);
        let hd = d / h;
        let qv: Vec<f32> = (0..b * t * d).map(|i| (i as f32 * 0.017).sin()).collect();
        let kv: Vec<f32> = (0..b * t * d).map(|i| (i as f32 * 0.019).cos()).collect();
        let vv: Vec<f32> = (0..b * t * d).map(|i| (i as f32 * 0.023).sin()).collect();
        let gq = GpuTensor::from_cpu_f32(&ctx.device, &qv, vec![b, t, d])?;
        let gk = GpuTensor::from_cpu_f32(&ctx.device, &kv, vec![b, t, d])?;
        let gv = GpuTensor::from_cpu_f32(&ctx.device, &vv, vec![b, t, d])?;
        let gsc = GpuTensor::zeros(&ctx.device, vec![b, h, t, t])?;
        let gy = GpuTensor::zeros(&ctx.device, vec![b, t, d])?;
        ctx.mhsa_f32(&gq, &gk, &gv, &gsc, None, 0, &gy)?;
        // candle: reshape to heads, QK^T / sqrt(hd), softmax, ×V, reshape
        let qc = Tensor::from_vec(qv.clone(), (b, t, h, hd), &candle_dev)?.permute((0, 2, 1, 3))?.contiguous()?;
        let kc = Tensor::from_vec(kv.clone(), (b, t, h, hd), &candle_dev)?.permute((0, 2, 1, 3))?.contiguous()?;
        let vc = Tensor::from_vec(vv.clone(), (b, t, h, hd), &candle_dev)?.permute((0, 2, 1, 3))?.contiguous()?;
        let scale = (hd as f64).sqrt();
        let scores = qc.matmul(&kc.transpose(2, 3)?)?;
        let scores = scores.affine(1.0 / scale, 0.0)?;
        let attn = candle_nn::ops::softmax(&scores, 3)?;
        let out = attn.matmul(&vc)?.permute((0, 2, 1, 3))?.contiguous()?.reshape((b, t, d))?;
        let dg = max_abs_diff(&gy.to_cpu_f32(&ctx).unwrap(), &out.flatten_all()?.to_vec1()?);
        println!("mhsa correctness: max|ours − candle| = {dg:.2e} {}", if dg < 1e-3 { "✓" } else { "✗ FAIL" });
    }

    // ── layernorm_head vs candle ──
    {
        let (rows, h, hd) = (40usize, 8usize, 48usize);
        let xg: Vec<f32> = (0..rows * h * hd).map(|i| (i as f32 * 0.003).sin()).collect();
        let wg: Vec<f32> = (0..hd).map(|i| 0.5 + (i as f32) * 0.01).collect();
        let bg: Vec<f32> = (0..hd).map(|i| -0.2 + (i as f32) * 0.004).collect();
        let gx = GpuTensor::from_cpu_f32(&ctx.device, &xg, vec![rows, h * hd])?;
        let gw = GpuTensor::from_cpu_f32(&ctx.device, &wg, vec![hd])?;
        let gb = GpuTensor::from_cpu_f32(&ctx.device, &bg, vec![hd])?;
        let gy = GpuTensor::zeros(&ctx.device, vec![rows, h * hd])?;
        ctx.layernorm_head_f32(&gx, &gw, &gb, &gy, h, 1e-5)?;
        // candle per-head layernorm: reshape (rows, h, hd), normalize over last
        let xc = Tensor::from_vec(xg.clone(), (rows, h, hd), &candle_dev)?;
        let mean = xc.mean_keepdim(2)?;
        let centered = xc.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim(2)?;
        let normed = centered.broadcast_div(&(var.affine(1.0, 1e-5))?.sqrt()?)?;
        let yc = normed.broadcast_mul(&Tensor::from_vec(wg.clone(), hd, &candle_dev)?.reshape((1, 1, hd))?)?
            .broadcast_add(&Tensor::from_vec(bg.clone(), hd, &candle_dev)?.reshape((1, 1, hd))?)?;
        let dg = max_abs_diff(&gy.to_cpu_f32(&ctx).unwrap(), &yc.flatten_all()?.to_vec1()?);
        println!("layernorm_head correctness: max|ours − candle| = {dg:.2e} {}", if dg < 1e-3 { "✓" } else { "✗ FAIL" });
    }

    // ── rope_head vs candle ──
    {
        let (t, d, h, rope_dim) = (16usize, 384usize, 8usize, 32usize);
        let xg: Vec<f32> = (0..t * d).map(|i| (i as f32 * 0.007).cos()).collect();
        let hd = d / h;
        let half = rope_dim / 2;
        let mut cos = vec![0f32; t * half];
        let mut sin = vec![0f32; t * half];
        for n in 0..t {
            for i in 0..half {
                let f = n as f64 * (1.0 / 10_000f64.powf(2.0 * i as f64 / rope_dim as f64));
                cos[n * half + i] = f.cos() as f32;
                sin[n * half + i] = f.sin() as f32;
            }
        }
        let gx = GpuTensor::from_cpu_f32(&ctx.device, &xg, vec![t, d])?;
        let gc = GpuTensor::from_cpu_f32(&ctx.device, &cos, vec![t, half])?;
        let gs = GpuTensor::from_cpu_f32(&ctx.device, &sin, vec![t, half])?;
        let gy = GpuTensor::zeros(&ctx.device, vec![t, d])?;
        ctx.rope_head_f32(&gx, &gc, &gs, &gy, h, rope_dim)?;
        // candle reference: per head, rotate first rope_dim
        let xc = Tensor::from_vec(xg.clone(), (t, h, hd), &candle_dev)?;
        let x_rope = xc.narrow(2, 0, rope_dim)?;
        let x_pass = xc.narrow(2, rope_dim, hd - rope_dim)?;
        let x1 = x_rope.narrow(2, 0, half)?;
        let x2 = x_rope.narrow(2, half, half)?;
        let cos_t = Tensor::from_vec(cos.clone(), (t, 1, half), &candle_dev)?;
        let sin_t = Tensor::from_vec(sin.clone(), (t, 1, half), &candle_dev)?;
        let out_rope = x_rope.broadcast_mul(&Tensor::cat(&[&cos_t, &cos_t], 2)?)?
            .broadcast_add(&Tensor::cat(&[&x2.neg()?, &x1], 2)?.broadcast_mul(&Tensor::cat(&[&sin_t, &sin_t], 2)?)?)?;
        let yc = Tensor::cat(&[&out_rope, &x_pass], 2)?.reshape((t, d))?;
        let dg = max_abs_diff(&gy.to_cpu_f32(&ctx).unwrap(), &yc.flatten_all()?.to_vec1()?);
        println!("rope_head correctness: max|ours − candle| = {dg:.2e} {}", if dg < 1e-3 { "✓" } else { "✗ FAIL" });
    }

    // ── timing: candle (im2col + gemm + transpose per call) ──
    // candle conv1d expects (B, C_in, L) — same data, conv-convention shape
    let xc = Tensor::from_vec(x_data.clone(), (b, cin, t), &candle_dev)?;
    let wc = Tensor::from_vec(w_data.clone(), (cout, cin, k), &candle_dev)?;
    let bc = Tensor::from_vec(bias_data.clone(), cout, &candle_dev)?;
    let yc = xc
        .conv1d(&wc, 3, 1, 1, 1)?
        .broadcast_add(&bc.reshape((1, cout, 1))?)?;
    yc.to_vec3::<f32>()?; // force completion

    let iters = 10;
    eprintln!("timing candle-Metal ({iters} iters, this is the slow path)…");
    let start = Instant::now();
    for _ in 0..iters {
        let y = xc
            .conv1d(&wc, 3, 1, 1, 1)?
            .broadcast_add(&bc.reshape((1, cout, 1))?)?;
        y.to_vec3::<f32>()?;
    }
    let candle_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // ── timing: our fused kernel (one commit per call) ──
    eprintln!("timing our fused kernel…");
    let start = Instant::now();
    for _ in 0..iters {
        ctx.conv1d_f32(&gx, &gw, Some(&gb), params, &gy)?;
    }
    let ours_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // ── timing: our fused kernel inside ONE command buffer (100 calls batched) ──
    let start = Instant::now();
    for _ in 0..iters {
        rust_inference::metal::batch(&ctx, |batch| {
            for _ in 0..100 {
                batch.run(
                    "conv1d_direct_f32",
                    metal::MTLSize::new(b as u64, t_out as u64, cout as u64),
                    metal::MTLSize::new(1, 4, 4),
                    |enc| {
                        enc.set_buffer(0, Some(&gx.buffer), 0);
                        enc.set_buffer(1, Some(&gw.buffer), 0);
                        enc.set_buffer(2, Some(&gb.buffer), 0);
                        enc.set_buffer(3, Some(&gy.buffer), 0);
                        for (i, v) in [t, cin, cout, k].iter().enumerate() {
                            enc.set_bytes((4 + i) as u64, 8, (v as *const usize).cast());
                        }
                        for (i, v) in [1usize, 3usize, 1usize].iter().enumerate() {
                            enc.set_bytes((8 + i) as u64, 8, (v as *const usize).cast());
                        }
                    },
                )?;
            }
            Ok(())
        })?;
    }
    let batched_ms = start.elapsed().as_secs_f64() * 1000.0 / (iters as f64 * 100.0);

    println!(
        "conv1d [1,300,256]×[256,256,7]: candle-Metal {candle_ms:.2} ms/op | ours {ours_ms:.2} ms/op ({:.1}× faster) | ours batched-in-one-buffer {batched_ms:.4} ms/op",
        candle_ms / ours_ms
    );
    println!(
        "⇒ per-op CPU sync costs candle {:.2} ms/op of overhead; batching removes it.",
        candle_ms - batched_ms
    );
    Ok(())
}
