//! CPU reference backend (`cpu` feature) — the per-op test oracle.
//!
//! A deliberately naive f32 implementation of the whole [`Backend`] contract
//! over plain `Vec<f32>` storage. It is NOT a production path: no SIMD, no
//! batching tricks, single-threaded, and only usable via an explicit runtime
//! opt-in (never an implicit fallback — GPU-only rule).
//!
//! Role in the plan:
//! - unit oracle for every op (tolerance-based cross-backend comparison);
//! - CI runs of nn/speech layer tests on small fixtures without a GPU;
//! - golden test-vectors are generated from this backend (integer-exact
//!   dequant ops stay bit-exact across backends, GEMM/reductions are
//!   tolerance-compared).
//!
//! Reductions here sum in ascending index order — the canonical order that
//! GPU kernels must reproduce for deterministic same-backend behavior.

use crate::backend::{Backend, Tensor};
use crate::error::BackendError;

/// Naive f32 CPU backend (test oracle). Stateless; tensors carry their data.
#[derive(Debug, Clone, Default)]
pub struct CpuBackend;

fn data<'a>(t: &'a Tensor) -> parking_lot::RwLockReadGuard<'a, Vec<f32>> {
    t.cpu_read()
}

fn data_mut<'a>(t: &'a Tensor) -> parking_lot::RwLockWriteGuard<'a, Vec<f32>> {
    t.cpu_write()
}

fn check_len(t: &Tensor, expected: usize) -> Result<(), BackendError> {
    if t.numel() != expected {
        return Err(BackendError::Shape(format!(
            "tensor numel {} != expected {} (shape {:?})",
            t.numel(),
            expected,
            t.shape()
        )));
    }
    Ok(())
}

fn elemwise<F: Fn(f32) -> f32>(x: &Tensor, y: &Tensor, f: F) -> Result<(), BackendError> {
    check_len(x, y.numel())?;
    if x.cpu_same_storage(y) {
        // In-place (e.g. silu(&a, &a)): single write guard, no self-deadlock.
        let mut yd = data_mut(y);
        for yi in yd.iter_mut() {
            *yi = f(*yi);
        }
        return Ok(());
    }
    let xd: Vec<f32> = data(x).to_vec();
    let mut yd = data_mut(y);
    for (xi, yi) in xd.iter().zip(yd.iter_mut()) {
        *yi = f(*xi);
    }
    Ok(())
}

fn elemwise2<F: Fn(f32, f32) -> f32>(
    a: &Tensor,
    b: &Tensor,
    y: &Tensor,
    f: F,
) -> Result<(), BackendError> {
    check_len(a, y.numel())?;
    check_len(b, y.numel())?;
    if y.cpu_same_storage(a) || y.cpu_same_storage(b) {
        // Output aliases an input: snapshot inputs first, then write.
        let ad: Vec<f32> = data(a).to_vec();
        let bd: Vec<f32> = data(b).to_vec();
        let mut yd = data_mut(y);
        for ((ai, bi), yi) in ad.iter().zip(bd.iter()).zip(yd.iter_mut()) {
            *yi = f(*ai, *bi);
        }
        return Ok(());
    }
    let ad = data(a);
    let bd = data(b);
    let mut yd = data_mut(y);
    for ((ai, bi), yi) in ad.iter().zip(bd.iter()).zip(yd.iter_mut()) {
        *yi = f(*ai, *bi);
    }
    Ok(())
}

impl Backend for CpuBackend {
    fn zeros(&self, shape: Vec<usize>) -> Result<Tensor, BackendError> {
        let n: usize = shape.iter().product();
        Ok(Tensor::from_cpu(vec![0.0; n], shape))
    }

    fn from_cpu_f32(&self, d: &[f32], shape: Vec<usize>) -> Result<Tensor, BackendError> {
        let n: usize = shape.iter().product();
        if d.len() != n {
            return Err(BackendError::Shape(format!(
                "from_cpu_f32: {} elements for shape {shape:?} (needs {n})",
                d.len()
            )));
        }
        Ok(Tensor::from_cpu(d.to_vec(), shape))
    }

    fn to_cpu_f32(&self, t: &Tensor) -> Result<Vec<f32>, BackendError> {
        Ok(data(t).to_vec())
    }

    fn gemm(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<(), BackendError> {
        // C (M,N) = A (M,K) · B (K,N), row-major.
        let (m, k) = (a.shape()[0], a.shape()[1]);
        let (k2, n) = (b.shape()[0], b.shape()[1]);
        if k != k2 || c.shape() != [m, n] {
            return Err(BackendError::Shape(format!(
                "gemm shapes A{:?} B{:?} C{:?}",
                a.shape(),
                b.shape(),
                c.shape()
            )));
        }
        let (ad, bd) = (data(a).to_vec(), data(b).to_vec());
        let mut cd = data_mut(c);
        cd.fill(0.0);
        for i in 0..m {
            for l in 0..k {
                let av = ad[i * k + l];
                if av == 0.0 {
                    continue;
                }
                for j in 0..n {
                    cd[i * n + j] += av * bd[l * n + j];
                }
            }
        }
        Ok(())
    }

    fn gemm_nt(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<(), BackendError> {
        // C (M,N) = A (M,K) · Bᵀ (N,K): b row-major (N,K).
        let (m, k) = (a.shape()[0], a.shape()[1]);
        let (n, k2) = (b.shape()[0], b.shape()[1]);
        if k != k2 || c.shape() != [m, n] {
            return Err(BackendError::Shape(format!(
                "gemm_nt shapes A{:?} B{:?} C{:?}",
                a.shape(),
                b.shape(),
                c.shape()
            )));
        }
        let (ad, bd) = (data(a).to_vec(), data(b).to_vec());
        let mut cd = data_mut(c);
        cd.fill(0.0);
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for l in 0..k {
                    acc += ad[i * k + l] * bd[j * k + l];
                }
                cd[i * n + j] = acc;
            }
        }
        Ok(())
    }

    fn mul(&self, a: &Tensor, b: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        elemwise2(a, b, y, |x, z| x * z)
    }
    fn add(&self, a: &Tensor, b: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        elemwise2(a, b, y, |x, z| x + z)
    }
    fn sub(&self, a: &Tensor, b: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        elemwise2(a, b, y, |x, z| x - z)
    }
    fn div(&self, a: &Tensor, b: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        elemwise2(a, b, y, |x, z| x / z)
    }
    fn add_scalar(&self, x: &Tensor, y: &Tensor, c: f32) -> Result<(), BackendError> {
        elemwise(x, y, |v| v + c)
    }
    fn scale(&self, x: &Tensor, y: &Tensor, c: f32) -> Result<(), BackendError> {
        elemwise(x, y, |v| v * c)
    }
    fn add_bias(&self, x: &Tensor, bias: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        // bias is (out,) and broadcasts over the last dim of x.
        let bd = data(bias);
        let cols = bias.numel();
        let xd: Vec<f32> = data(x).to_vec();
        check_len(x, y.numel())?;
        if cols == 0 || x.numel() % cols != 0 {
            return Err(BackendError::Shape(format!(
                "add_bias: bias {} does not divide x {}",
                cols,
                x.numel()
            )));
        }
        let mut yd = data_mut(y);
        for (i, (xi, yi)) in xd.iter().zip(yd.iter_mut()).enumerate() {
            *yi = *xi + bd[i % cols];
        }
        Ok(())
    }
    fn sqrt(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        elemwise(x, y, f32::sqrt)
    }
    fn log(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        elemwise(x, y, f32::ln)
    }
    fn max_scalar(&self, x: &Tensor, y: &Tensor, floor: f32) -> Result<(), BackendError> {
        elemwise(x, y, |v| v.max(floor))
    }

    fn relu(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        elemwise(x, y, |v| v.max(0.0))
    }
    fn leaky_relu_clip(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        elemwise(x, y, |v| v.max(0.0) + 0.01 * v.min(0.0))
    }
    fn sigmoid(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        elemwise(x, y, |v| 1.0 / (1.0 + (-v).exp()))
    }
    fn tanh(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        elemwise(x, y, f32::tanh)
    }
    fn silu(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        elemwise(x, y, |v| v / (1.0 + (-v).exp()))
    }
    fn silu_mul_fused(&self, x: &Tensor, y: &Tensor, ffn: usize) -> Result<(), BackendError> {
        let (n, d) = (y.numel(), y.shape().last().copied().unwrap_or(0));
        let xr = x.cpu_read();
        let mut yr = y.cpu_write();
        for i in 0..n {
            let row = i / d;
            let g = xr[row * 2 * d + i % d];
            let up = xr[row * 2 * d + d + i % d];
            yr[i] = (g / (1.0 + (-g).exp())) * up;
        }
        let _ = ffn;
        Ok(())
    }
    fn dflash2_conv(
        &self,
        x: &Tensor,
        base: &Tensor,
        dynamic: &Tensor,
        y: &Tensor,
        kernel: usize,
        group_size: usize,
        block_size: usize,
        side: usize,
    ) -> Result<(), BackendError> {
        // CPU-референс спеки PR #27342 (ggml-порядок тензоров):
        // y[t,i] = Σ_tap (coeff[r, t] + BASE[side,tap,i])·x[t−tap],
        // r = g + n_groups·(tap + kernel·side), g = i/group_size,
        // DYN [n_tokens, n_groups·kernel·2] (row-major: t·proj + r),
        // BASE [2, kernel, n_embd] (side·k·D + tap·D + i), сдвиг в блоке.
        let (n_tokens, n_embd) = (y.shape()[0], y.shape()[1]);
        let xr = x.cpu_read();
        let br = base.cpu_read();
        let dr = dynamic.cpu_read();
        let mut yr = y.cpu_write();
        let n_groups = n_embd / group_size;
        let proj = n_groups * kernel * 2;
        for t in 0..n_tokens {
            let blk = t / block_size;
            let pos = t % block_size;
            for i in 0..n_embd {
                let g = i / group_size;
                let mut acc = 0.0f32;
                for tap in 0..kernel {
                    let v = if pos >= tap {
                        xr[(blk * block_size + (pos - tap)) * n_embd + i]
                    } else {
                        0.0
                    };
                    let r = g + n_groups * (tap + kernel * side);
                    let coeff = dr[t * proj + r];
                    let b = br[side * kernel * n_embd + tap * n_embd + i];
                    acc += (coeff + b) * v;
                }
                yr[t * n_embd + i] = acc;
            }
        }
        Ok(())
    }
    fn gelu_tanh(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        // tanh approximation (GPT-2 style), f32 compute.
        elemwise(x, y, |v| {
            0.5 * v * (1.0 + (0.7978845608 * (v + 0.044715 * v * v * v)).tanh())
        })
    }
    fn gelu_erf(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        // Точный GELU (erf), A&S 7.1.26 — та же формула, что в Metal-кернеле.
        elemwise(x, y, |v| {
            0.5 * v * (1.0 + crate::backend::erf_approx(v / std::f32::consts::SQRT_2))
        })
    }
    fn snake(&self, x: &Tensor, a: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        // y = x + (1/a)·sin(a·x)² — a PER-CHANNEL (Metal semantics):
        // A (C,), X (…, C) flattened; element i uses A[i % C].
        check_len(x, y.numel())?;
        let c = a.numel();
        let xd: Vec<f32> = data(x).to_vec();
        let ad = data(a);
        let mut yd = data_mut(y);
        for (i, xi) in xd.iter().enumerate() {
            let av = ad[i % c];
            let s = (av * *xi).sin();
            yd[i] = *xi + s * s / av;
        }
        Ok(())
    }

    fn rmsnorm(&self, x: &Tensor, w: &Tensor, y: &Tensor, eps: f32) -> Result<(), BackendError> {
        let cols = w.numel();
        let xd: Vec<f32> = data(x).to_vec();
        check_len(x, y.numel())?;
        if cols == 0 || x.numel() % cols != 0 {
            return Err(BackendError::Shape(
                "rmsnorm: weight size does not divide x".into(),
            ));
        }
        let wd = data(w);
        let mut yd = data_mut(y);
        for (r, row) in xd.chunks_exact(cols).enumerate() {
            let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / cols as f32;
            let inv = 1.0 / (mean_sq + eps).sqrt();
            for (j, (&xv, yv)) in row
                .iter()
                .zip(yd[r * cols..(r + 1) * cols].iter_mut())
                .enumerate()
            {
                *yv = xv * inv * wd[j];
            }
        }
        Ok(())
    }

    fn layernorm(
        &self,
        x: &Tensor,
        w: &Tensor,
        b: &Tensor,
        y: &Tensor,
        eps: f32,
    ) -> Result<(), BackendError> {
        let cols = w.numel();
        let xd: Vec<f32> = data(x).to_vec();
        check_len(x, y.numel())?;
        if cols == 0 || x.numel() % cols != 0 {
            return Err(BackendError::Shape(
                "layernorm: weight size does not divide x".into(),
            ));
        }
        let wd = data(w);
        let bd = data(b);
        let mut yd = data_mut(y);
        for (r, row) in xd.chunks_exact(cols).enumerate() {
            let mean = row.iter().sum::<f32>() / cols as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for (j, (&xv, yv)) in row
                .iter()
                .zip(yd[r * cols..(r + 1) * cols].iter_mut())
                .enumerate()
            {
                *yv = (xv - mean) * inv * wd[j] + bd[j];
            }
        }
        Ok(())
    }

    fn conv1d(
        &self,
        x: &Tensor,
        w: &Tensor,
        bias: Option<&Tensor>,
        stride: usize,
        pad_l: usize,
        pad_r: usize,
        dil: usize,
        y: &Tensor,
    ) -> Result<(), BackendError> {
        // Dense (groups=1) wrapper over the grouped implementation.
        self.conv1d_groups(x, w, bias, stride, pad_l, pad_r, dil, 1, y)
    }

    fn conv1d_groups(
        &self,
        x: &Tensor,
        w: &Tensor,
        bias: Option<&Tensor>,
        stride: usize,
        pad_l: usize,
        pad_r: usize,
        dil: usize,
        groups: usize,
        y: &Tensor,
    ) -> Result<(), BackendError> {
        // X (B,T,Cin), W (Cout,Cin_g,K), Y (B,Tout,Cout). Left/right padding;
        // out channel co reads input channels [g·Cin_g, (g+1)·Cin_g) with
        // g = co / Cin_g (depthwise when Cin_g == 1).
        let (b, t, cin) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let (cout, cin_g, k) = (w.shape()[0], w.shape()[1], w.shape()[2]);
        if cin % groups != 0 || cin_g != cin / groups {
            return Err(BackendError::Shape("conv1d_groups: group mismatch".into()));
        }
        let tout = (t + pad_l + pad_r - dil * (k - 1) - 1) / stride + 1;
        if y.shape() != &[b, tout, cout] {
            return Err(BackendError::Shape(format!(
                "conv1d: expected Y {:?}, got {:?}",
                vec![b, tout, cout],
                y.shape()
            )));
        }
        let (xd, wd) = (data(x).to_vec(), data(w).to_vec());
        let bd: Vec<f32> = match bias {
            Some(bi) => data(bi).to_vec(),
            None => vec![0.0; cout],
        };
        let mut yd = data_mut(y);
        yd.fill(0.0);
        let per = cin / groups; // input channels per group == cin_g
        for bi in 0..b {
            for to in 0..tout {
                for co in 0..cout {
                    let mut acc = bd[co];
                    let g = co / (cout / groups);
                    let ci0 = g * per;
                    for ci in ci0..ci0 + per {
                        for kk in 0..k {
                            let ti = to * stride + kk * dil;
                            let src = ti as isize - pad_l as isize;
                            if src < 0 || src >= t as isize {
                                continue;
                            }
                            // Weight rows are per OUTPUT channel (Metal
                            // convention): W[co * Cin_g + local_ci] — NOT
                            // group-major ((co/per)*per is wrong for dense
                            // when cin != cout).
                            acc += xd[(bi * t + src as usize) * cin + ci]
                                * wd[(co * cin_g + (ci - ci0)) * k + kk];
                        }
                    }
                    yd[(bi * tout + to) * cout + co] = acc;
                }
            }
        }
        Ok(())
    }

    fn conv2d_nchw(
        &self,
        x: &Tensor,
        w: &Tensor,
        bias: Option<&Tensor>,
        pad_h: usize,
        pad_w: usize,
        stride_t: usize,
        stride_f: usize,
        time_major_out: bool,
        y: &Tensor,
    ) -> Result<(), BackendError> {
        // X (N,Ci,H,W), W (Co,Ci,KH,KW); Y NCHW (N,Co,OH,OW) or time-major
        // (N, OH, Co·OW). dil=1.
        let (n, ci, h, wdim) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
        let (co, ci2, kh, kw) = (w.shape()[0], w.shape()[1], w.shape()[2], w.shape()[3]);
        if ci != ci2 {
            return Err(BackendError::Shape("conv2d: channel mismatch".into()));
        }
        // LEFT padding on both axes (Metal conv2d_f32 semantics — the tone
        // pre-encode convs left-pad time by pad_t; symmetric padding would
        // shift the output by pad_h).
        let (oh, ow) = (
            (h + pad_h - kh) / stride_t + 1,
            (wdim + pad_w - kw) / stride_f + 1,
        );
        if !time_major_out && y.shape() != &[n, co, oh, ow] {
            return Err(BackendError::Shape("conv2d: output shape mismatch".into()));
        }
        if time_major_out && y.shape() != &[n, oh, co * ow] {
            return Err(BackendError::Shape(
                "conv2d: time-major output shape mismatch".into(),
            ));
        }
        let (xd, wd) = (data(x).to_vec(), data(w).to_vec());
        let bd: Vec<f32> = match bias {
            Some(bi) => data(bi).to_vec(),
            None => vec![0.0; co],
        };
        let mut yd = data_mut(y);
        yd.fill(0.0);
        for ni in 0..n {
            for oci in 0..oh {
                for owi in 0..ow {
                    for coi in 0..co {
                        let mut acc = bd[coi];
                        for ci3 in 0..ci {
                            for khh in 0..kh {
                                for kww in 0..kw {
                                    // Left-pad semantics: negative positions
                                    // wrap to huge values and are skipped.
                                    let hi = (oci * stride_t + khh).wrapping_sub(pad_h);
                                    let wi = (owi * stride_f + kww).wrapping_sub(pad_w);
                                    if hi >= h || wi >= wdim {
                                        continue;
                                    }
                                    acc += xd[((ni * ci + ci3) * h + hi) * wdim + wi]
                                        * wd[((coi * ci + ci3) * kh + khh) * kw + kww];
                                }
                            }
                        }
                        if time_major_out {
                            yd[(ni * oh + oci) * (co * ow) + coi * ow + owi] = acc;
                        } else {
                            yd[((ni * co + coi) * oh + oci) * ow + owi] = acc;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn maxpool2d(&self, x: &Tensor, y: &Tensor, kh: usize, kw: usize) -> Result<(), BackendError> {
        // kernel == stride; X/Y (N,C,H,W).
        let (n, c, h, w) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
        let (oh, ow) = (h / kh, w / kw);
        if y.shape() != &[n, c, oh, ow] {
            return Err(BackendError::Shape(
                "maxpool2d: output shape mismatch".into(),
            ));
        }
        let xd: Vec<f32> = data(x).to_vec();
        let mut yd = data_mut(y);
        for ni in 0..n {
            for ci in 0..c {
                for oi in 0..oh {
                    for oj in 0..ow {
                        let mut best = f32::NEG_INFINITY;
                        for hi in 0..kh {
                            for wj in 0..kw {
                                let v = xd[((ni * c + ci) * h + oi * kh + hi) * w + oj * kw + wj];
                                best = best.max(v);
                            }
                        }
                        yd[((ni * c + ci) * oh + oi) * ow + oj] = best;
                    }
                }
            }
        }
        Ok(())
    }

    fn conv_transpose1d(
        &self,
        x: &Tensor,
        w: &Tensor,
        bias: Option<&Tensor>,
        stride: usize,
        y: &Tensor,
    ) -> Result<(), BackendError> {
        // W (Cin, Cout, K), X (B,T,Cin), Y (B,Tout,Cout); dil=1, pad=0.
        let (b, t, cin) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let (cin2, cout, k) = (w.shape()[0], w.shape()[1], w.shape()[2]);
        if cin != cin2 {
            return Err(BackendError::Shape(
                "conv_transpose1d: channel mismatch".into(),
            ));
        }
        let tout = (t - 1) * stride + k;
        if y.shape() != &[b, tout, cout] {
            return Err(BackendError::Shape(
                "conv_transpose1d: output shape mismatch".into(),
            ));
        }
        let (xd, wd) = (data(x).to_vec(), data(w).to_vec());
        let bd: Vec<f32> = match bias {
            Some(bi) => data(bi).to_vec(),
            None => vec![0.0; cout],
        };
        let mut yd = data_mut(y);
        yd.fill(0.0);
        for bi in 0..b {
            for ti in 0..t {
                for ci in 0..cin {
                    let xv = xd[(bi * t + ti) * cin + ci];
                    if xv == 0.0 {
                        continue;
                    }
                    for co in 0..cout {
                        for kk in 0..k {
                            let to = ti * stride + kk;
                            yd[(bi * tout + to) * cout + co] += xv * wd[(ci * cout + co) * k + kk];
                        }
                    }
                }
            }
        }
        // bias
        for bi in 0..b {
            for to in 0..tout {
                for co in 0..cout {
                    yd[(bi * tout + to) * cout + co] += bd[co];
                }
            }
        }
        Ok(())
    }

    fn im2col1d(
        &self,
        x: &Tensor,
        xc: &Tensor,
        k: usize,
        stride: usize,
        pad_l: usize,
        dil: usize,
        tout: usize,
    ) -> Result<(), BackendError> {
        let (b, t, cin) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        if xc.shape() != &[b, tout, cin * k] {
            return Err(BackendError::Shape(
                "im2col1d: output shape mismatch".into(),
            ));
        }
        let xd = data(x);
        let mut xcd = data_mut(xc);
        let per = cin * k;
        for bi in 0..b {
            for to in 0..tout {
                for ci in 0..cin {
                    for kk in 0..k {
                        // Signed arithmetic: pad_l can exceed the source
                        // index (causal padding) — in usize that underflows
                        // (panic in debug builds); `src >= 0` on usize was
                        // also a useless comparison.
                        let src = to as isize * stride as isize
                            + kk as isize * dil as isize
                            - pad_l as isize;
                        let v = if src >= 0 && (src as usize) < t {
                            xd[(bi * t + src as usize) * cin + ci]
                        } else {
                            0.0
                        };
                        xcd[(bi * tout + to) * per + ci * k + kk] = v;
                    }
                }
            }
        }
        Ok(())
    }

    fn im2col_t1d(
        &self,
        x: &Tensor,
        xc: &Tensor,
        k: usize,
        stride: usize,
        tout: usize,
    ) -> Result<(), BackendError> {
        let (b, tin, cin) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        if xc.shape() != &[b, tout, cin * k] {
            return Err(BackendError::Shape(
                "im2col_t1d: output shape mismatch".into(),
            ));
        }
        let xd = data(x);
        let mut xcd = data_mut(xc);
        let per = cin * k;
        for bi in 0..b {
            for to in 0..tout {
                for ci in 0..cin {
                    for kk in 0..k {
                        let num = to as isize - kk as isize;
                        let v = if num >= 0
                            && num % stride as isize == 0
                            && (num / stride as isize) < tin as isize
                        {
                            xd[(bi * tin + (num / stride as isize) as usize) * cin + ci]
                        } else {
                            0.0
                        };
                        xcd[(bi * tout + to) * per + ci * k + kk] = v;
                    }
                }
            }
        }
        Ok(())
    }

    fn slice_last(&self, x: &Tensor, y: &Tensor, off: usize) -> Result<(), BackendError> {
        // X (rows, D) → Y (rows, d0): columns [off, off+d0); d0 comes from Y.
        let (rows, d) = (x.shape()[0], x.shape()[1]);
        let d0 = y.shape()[1];
        if off + d0 > d || y.shape() != &[rows, d0] {
            return Err(BackendError::Shape(format!(
                "slice_last: x {:?} y {:?} off {off}",
                x.shape(),
                y.shape()
            )));
        }
        let xd: Vec<f32> = data(x).to_vec();
        let mut yd = data_mut(y);
        for r in 0..rows {
            yd[r * d0..(r + 1) * d0].copy_from_slice(&xd[r * d + off..r * d + off + d0]);
        }
        Ok(())
    }

    fn slice_time(&self, x: &Tensor, y: &Tensor, u: usize) -> Result<(), BackendError> {
        // X (B,T,C) → Y (B,T-u,C): drop the last u frames.
        let (b, t, c) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        if y.shape() != &[b, t - u, c] {
            return Err(BackendError::Shape(
                "slice_time: output shape mismatch".into(),
            ));
        }
        let xd: Vec<f32> = data(x).to_vec();
        let mut yd = data_mut(y);
        let keep = t - u;
        for bi in 0..b {
            yd[bi * keep * c..(bi + 1) * keep * c]
                .copy_from_slice(&xd[bi * t * c..bi * t * c + keep * c]);
        }
        Ok(())
    }

    fn softmax_last_dim(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        let (rows, d) = (x.shape()[0], x.shape()[1]);
        check_len(x, y.numel())?;
        let xd: Vec<f32> = data(x).to_vec();
        let mut yd = data_mut(y);
        for r in 0..rows {
            let row = &xd[r * d..(r + 1) * d];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|v| (v - max).exp()).sum();
            for (j, (&xv, yv)) in row
                .iter()
                .zip(yd[r * d..(r + 1) * d].iter_mut())
                .enumerate()
            {
                let _ = j;
                *yv = (xv - max).exp() / sum;
            }
        }
        Ok(())
    }

    fn embedding(&self, idx: &[u32], table: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        let (v, d) = (table.shape()[0], table.shape()[1]);
        // The Metal kernel writes Y as (1, len(idx), D); the CPU oracle
        // accepts any row-major layout with len(idx)·D elements.
        if y.numel() != idx.len() * d {
            return Err(BackendError::Shape(format!(
                "embedding: y numel {} != {} (idx {} × D {})",
                y.numel(),
                idx.len() * d,
                idx.len(),
                d
            )));
        }
        let td = data(table).to_vec();
        let mut yd = data_mut(y);
        for (i, &ix) in idx.iter().enumerate() {
            if ix as usize >= v {
                return Err(BackendError::Shape(format!(
                    "embedding: index {ix} >= vocab {v}"
                )));
            }
            if std::env::var("LLM_DBG_ATTN").is_ok() && i == 0 {
                eprintln!(
                    "CPU_EMB ix={ix} v={v} d={d} table[..8]: {:?} table[ix*d..]: {:?}",
                    &td[..8],
                    &td[ix as usize * d..ix as usize * d + 8]
                );
            }
            yd[i * d..(i + 1) * d].copy_from_slice(&td[ix as usize * d..(ix as usize + 1) * d]);
        }
        Ok(())
    }

    fn embedding_override(
        &self,
        idx: &[u32],
        table: &Tensor,
        vision: &Tensor,
        y: &Tensor,
        override_id: u32,
    ) -> Result<(), BackendError> {
        let (v, d) = (table.shape()[0], table.shape()[1]);
        if y.numel() != idx.len() * d {
            return Err(BackendError::Shape(format!(
                "embedding_override: y numel {} != {}",
                y.numel(),
                idx.len() * d
            )));
        }
        let (td, vd) = (data(table).to_vec(), data(vision).to_vec());
        let mut yd = data_mut(y);
        let nv = vision.shape()[0];
        let mut vi = 0usize;
        for (i, &id) in idx.iter().enumerate() {
            let dst = &mut yd[i * d..(i + 1) * d];
            if id == override_id {
                if vi < nv {
                    dst.copy_from_slice(&vd[vi * d..(vi + 1) * d]);
                } else {
                    dst.fill(0.0);
                }
                vi += 1;
            } else {
                let row = (id as usize).min(v - 1);
                dst.copy_from_slice(&td[row * d..(row + 1) * d]);
            }
        }
        Ok(())
    }

    fn rope(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        y: &Tensor,
        rope_dim: usize,
    ) -> Result<(), BackendError> {
        // Halved RoPE: X (T, head_dim), pairs (i, i + rope_dim/2).
        let (t, hd) = (x.shape()[0], x.shape()[1]);
        check_len(x, y.numel())?;
        let half = rope_dim / 2;
        let (xd, cd, sd) = (data(x).to_vec(), data(cos).to_vec(), data(sin).to_vec());
        let mut yd = data_mut(y);
        yd.copy_from_slice(&xd);
        for ti in 0..t {
            for i in 0..half {
                let (x0, x1) = (xd[ti * hd + i], xd[ti * hd + i + half]);
                let (c, s) = (cd[ti * half + i], sd[ti * half + i]);
                yd[ti * hd + i] = x0 * c - x1 * s;
                yd[ti * hd + i + half] = x0 * s + x1 * c;
            }
        }
        Ok(())
    }

    fn rope_interleaved(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        y: &Tensor,
        rope_dim: usize,
        start: usize,
    ) -> Result<(), BackendError> {
        // Interleaved (fishaudio), per-head (Metal semantics): x (B, T, H·hd)
        // with rope_dim = hd; pairs (2i, 2i+1) inside each head; cos/sin at
        // absolute position (t + start), shared across heads.
        let (b, t, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let hd = rope_dim;
        let h = d / hd;
        check_len(x, y.numel())?;
        let half = hd / 2;
        let (xd, cd, sd) = (data(x).to_vec(), data(cos).to_vec(), data(sin).to_vec());
        let mut yd = data_mut(y);
        yd.copy_from_slice(&xd);
        for bi in 0..b {
            for ti in 0..t {
                let ci = start + ti;
                for hh in 0..h {
                    for i in 0..half {
                        let idx = (bi * t + ti) * d + hh * hd + 2 * i;
                        let (x0, x1) = (xd[idx], xd[idx + 1]);
                        let (c, s) = (cd[ci * half + i], sd[ci * half + i]);
                        yd[idx] = x0 * c - x1 * s;
                        yd[idx + 1] = x0 * s + x1 * c;
                    }
                }
            }
        }
        Ok(())
    }

    fn rope_head(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        y: &Tensor,
        h: usize,
        rope_dim: usize,
    ) -> Result<(), BackendError> {
        // X (B, T, h·hd): halved RoPE inside the first rope_dim cols of each head.
        let (b, t, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        check_len(x, y.numel())?;
        let hd = d / h;
        let half = rope_dim / 2;
        let (xd, cd, sd) = (data(x).to_vec(), data(cos).to_vec(), data(sin).to_vec());
        let mut yd = data_mut(y);
        yd.copy_from_slice(&xd);
        for bi in 0..b {
            for ti in 0..t {
                for hh in 0..h {
                    let base = (bi * t + ti) * d + hh * hd;
                    for i in 0..half {
                        let (x0, x1) = (xd[base + i], xd[base + i + half]);
                        let (c, s) = (cd[ti * half + i], sd[ti * half + i]);
                        yd[base + i] = x0 * c - x1 * s;
                        yd[base + i + half] = x0 * s + x1 * c;
                    }
                }
            }
        }
        Ok(())
    }

    fn rmsnorm_head(
        &self,
        x: &Tensor,
        w: &Tensor,
        y: &Tensor,
        h: usize,
        eps: f32,
    ) -> Result<(), BackendError> {
        // X (rows, h·hd), w (hd) shared across heads. X may be 2D/3D
        // (batched tokens); rows = all leading dims flattened.
        let last = x.shape().len() - 1;
        let hd = x.shape()[last] / h;
        let rows = x.numel() / (h * hd);
        check_len(x, y.numel())?;
        let (xd, wd) = (data(x).to_vec(), data(w).to_vec());
        let mut yd = data_mut(y);
        for r in 0..rows {
            for hh in 0..h {
                let base = r * (h * hd) + hh * hd;
                let mean_sq = (0..hd).map(|j| xd[base + j] * xd[base + j]).sum::<f32>() / hd as f32;
                let inv = 1.0 / (mean_sq + eps).sqrt();
                for j in 0..hd {
                    yd[base + j] = xd[base + j] * inv * wd[j];
                }
            }
        }
        Ok(())
    }

    fn layernorm_head(
        &self,
        x: &Tensor,
        w: &Tensor,
        b: &Tensor,
        y: &Tensor,
        h: usize,
        eps: f32,
    ) -> Result<(), BackendError> {
        let hd = x.shape()[x.shape().len() - 1] / h;
        let rows = x.numel() / (h * hd);
        check_len(x, y.numel())?;
        let (xd, wd, bd) = (data(x), data(w), data(b));
        let mut yd = data_mut(y);
        for r in 0..rows {
            for hh in 0..h {
                let base = r * (h * hd) + hh * hd;
                let mean = (0..hd).map(|j| xd[base + j]).sum::<f32>() / hd as f32;
                let var = (0..hd)
                    .map(|j| (xd[base + j] - mean) * (xd[base + j] - mean))
                    .sum::<f32>()
                    / hd as f32;
                let inv = 1.0 / (var + eps).sqrt();
                for j in 0..hd {
                    yd[base + j] = (xd[base + j] - mean) * inv * wd[j] + bd[j];
                }
            }
        }
        Ok(())
    }

    fn mul_rows(&self, x: &Tensor, w: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        let (rows, d) = (x.shape()[0], x.shape()[1]);
        check_len(x, y.numel())?;
        let (xd, wd) = (data(x).to_vec(), data(w).to_vec());
        let mut yd = data_mut(y);
        for r in 0..rows {
            let s = wd[r];
            for j in 0..d {
                yd[r * d + j] = xd[r * d + j] * s;
            }
        }
        Ok(())
    }

    fn mul_vec(&self, x: &Tensor, w: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        let (rows, cols) = (x.shape()[0], x.shape()[1]);
        check_len(x, y.numel())?;
        let (xd, wd) = (data(x).to_vec(), data(w).to_vec());
        let mut yd = data_mut(y);
        for r in 0..rows {
            for j in 0..cols {
                yd[r * cols + j] = xd[r * cols + j] * wd[j];
            }
        }
        Ok(())
    }

    fn glu(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        let (rows, two_d) = (x.shape()[0], x.shape()[1]);
        let d = two_d / 2;
        if y.numel() != rows * d {
            return Err(BackendError::Shape(format!(
                "glu: x {:?} → y numel {} != {}",
                x.shape(),
                y.numel(),
                rows * d
            )));
        }
        let xd: Vec<f32> = data(x).to_vec();
        let mut yd = data_mut(y);
        for r in 0..rows {
            for j in 0..d {
                let a = xd[r * two_d + j];
                let bb = xd[r * two_d + d + j];
                yd[r * d + j] = a / (1.0 + (-bb).exp());
            }
        }
        Ok(())
    }

    fn repeat(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        // ×2 time upsampling: y[b, 2t + {0,1}, d] = x[b, t, d].
        let (b, t, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        if y.shape() != &[b, 2 * t, d] {
            return Err(BackendError::Shape("repeat: expected Y (B, 2T, D)".into()));
        }
        let xd: Vec<f32> = data(x).to_vec();
        let mut yd = data_mut(y);
        for bi in 0..b {
            for ti in 0..t {
                let src = &xd[(bi * t + ti) * d..(bi * t + ti + 1) * d];
                yd[(bi * 2 * t + 2 * ti) * d..(bi * 2 * t + 2 * ti + 1) * d].copy_from_slice(src);
                yd[(bi * 2 * t + 2 * ti + 1) * d..(bi * 2 * t + 2 * ti + 2) * d]
                    .copy_from_slice(src);
            }
        }
        Ok(())
    }

    fn add_scaled(
        &self,
        a: &Tensor,
        b: &Tensor,
        y: &Tensor,
        scale: f32,
    ) -> Result<(), BackendError> {
        elemwise2(a, b, y, |x, z| x + scale * z)
    }

    fn add_bias2d(&self, x: &Tensor, b: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        // Elementwise (Metal semantics): Y = X + B, all same shape.
        if b.shape() != x.shape() || y.shape() != x.shape() {
            return Err(BackendError::Shape(format!(
                "add_bias2d: shapes {:?} + {:?} -> {:?} (elementwise)",
                x.shape(),
                b.shape(),
                y.shape()
            )));
        }
        let xd: Vec<f32> = data(x).to_vec();
        let bd = data(b);
        let mut yd = data_mut(y);
        for i in 0..xd.len() {
            yd[i] = xd[i] + bd[i];
        }
        Ok(())
    }

    fn attn_ar(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        y: &Tensor,
        n_heads: usize,
        kv_heads: usize,
        hd: usize,
    ) -> Result<(), BackendError> {
        // Naive decomposition of the fused AR attention (tolerance-compared).
        // GQA: k/v depth is kv_heads·hd (not q depth n_heads·hd).
        let (b, tq, d) = (q.shape()[0], q.shape()[1], q.shape()[2]);
        let tkv = k.shape()[1];
        let kv_d = k.shape()[2];
        if y.shape() != &[b, tq, d] || kv_d != kv_heads * hd || v.shape()[2] != kv_d {
            return Err(BackendError::Shape("attn_ar: shape mismatch".into()));
        }
        // D8: the mask is indexed as [tq, tkv] below (m[t * tkv + tk]) — a
        // too-small mask would read out of bounds (panic) or wrong values.
        if let Some(m) = mask {
            if m.numel() < tq * tkv {
                return Err(BackendError::Shape(format!(
                    "attn_ar: mask numel {} < tq·tkv {tq}·{tkv}",
                    m.numel()
                )));
            }
        }
        let n_rep = n_heads / kv_heads;
        let (qd, kd, vd) = (data(q).to_vec(), data(k).to_vec(), data(v).to_vec());
        let md: Option<Vec<f32>> = mask.map(|m| data(m).to_vec());
        let mut yd = data_mut(y);
        let scale = 1.0 / (hd as f32).sqrt();
        for bi in 0..b {
            for t in 0..tq {
                for hh in 0..n_heads {
                    let kvh = hh / n_rep;
                    let out_off = (bi * tq + t) * d + hh * hd;
                    // scores
                    let mut scores = Vec::with_capacity(tkv);
                    for tk in 0..tkv {
                        let q_off = (bi * tq + t) * d + hh * hd;
                        let k_off = (bi * tkv + tk) * kv_d + kvh * hd;
                        let mut acc = 0.0f32;
                        for j in 0..hd {
                            acc += qd[q_off + j] * kd[k_off + j];
                        }
                        let mut s = acc * scale;
                        if let Some(m) = &md {
                            s += m[t * tkv + tk];
                        }
                        scores.push(s);
                    }
                    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for s in &scores {
                        sum += (s - max).exp();
                    }
                    for j in 0..hd {
                        let mut acc = 0.0f32;
                        for tk in 0..tkv {
                            let w = (scores[tk] - max).exp() / sum;
                            let v_off = (bi * tkv + tk) * kv_d + kvh * hd;
                            acc += w * vd[v_off + j];
                        }
                        yd[out_off + j] = acc;
                    }
                }
            }
        }
        Ok(())
    }

    fn mhsa(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        sc: &Tensor,
        mask: Option<&Tensor>,
        mode: u32,
        y: &Tensor,
    ) -> Result<(), BackendError> {
        // Naive MHSA (t-one style): Q/K/V (B,T,D), D = H·HD, H = sc.shape[1].
        // mode 0/1 — same math (scores computed locally; SC workspace ignored
        // in the naive oracle — leader writes, non-leader would read; the
        // CPU oracle recomputes for both); mode 2 — uniform attention.
        let (b, t, d) = (q.shape()[0], q.shape()[1], q.shape()[2]);
        let h = sc.shape()[1];
        let hd = d / h;
        if y.shape() != &[b, t, d] {
            return Err(BackendError::Shape("mhsa: output shape mismatch".into()));
        }
        // D8: the mask is indexed as [t, t] below (m[tq * t + tk]) — a
        // too-small mask would read out of bounds (panic) or wrong values.
        if let Some(m) = mask {
            if m.numel() < t * t {
                return Err(BackendError::Shape(format!(
                    "mhsa: mask numel {} < t·t {t}·{t}",
                    m.numel()
                )));
            }
        }
        if mode == 2 {
            // uniform: out = mean over time of V per head? The uniform kernel
            // averages the time axis of V (B,T,D) → Y (B,T,D) broadcast.
            let vd: Vec<f32> = data(v).to_vec();
            let mut yd = data_mut(y);
            for bi in 0..b {
                for j in 0..d {
                    let mean = (0..t).map(|ti| vd[(bi * t + ti) * d + j]).sum::<f32>() / t as f32;
                    for ti in 0..t {
                        yd[(bi * t + ti) * d + j] = mean;
                    }
                }
            }
            return Ok(());
        }
        let (qd, kd, vd) = (data(q).to_vec(), data(k).to_vec(), data(v).to_vec());
        let md: Option<Vec<f32>> = mask.map(|m| data(m).to_vec());
        let mut yd = data_mut(y);
        let scale = 1.0 / (hd as f32).sqrt();
        for bi in 0..b {
            for tq in 0..t {
                for hh in 0..h {
                    let off = |tt: usize, head: usize| (bi * t + tt) * d + head * hd;
                    let mut scores = Vec::with_capacity(t);
                    for tk in 0..t {
                        let mut acc = 0.0f32;
                        for j in 0..hd {
                            acc += qd[off(tq, hh) + j] * kd[off(tk, hh) + j];
                        }
                        let mut s = acc * scale;
                        if let Some(m) = &md {
                            s += m[tq * t + tk];
                        }
                        scores.push(s);
                    }
                    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    for s in &scores {
                        sum += (s - max).exp();
                    }
                    for j in 0..hd {
                        let mut acc = 0.0f32;
                        for tk in 0..t {
                            let w = (scores[tk] - max).exp() / sum;
                            acc += w * vd[off(tk, hh) + j];
                        }
                        yd[off(tq, hh) + j] = acc;
                    }
                }
            }
        }
        Ok(())
    }

    /// Relative-position MHSA (NeMo rel_pos / zipformer) — the CPU oracle:
    /// per-head scores ((q+u)·kᵀ + shift((q+v)·pᵀ))/√dk, softmax, ·v.
    fn mhsa_rel_pos(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        qp: &Tensor,
        p: &Tensor,
        u: &Tensor,
        vb: &Tensor,
        y: &Tensor,
        wfull: Option<&Tensor>,
        dk: usize,
        dp: usize,
        mode: u32,
    ) -> Result<(), BackendError> {
        let (t, d) = (q.shape()[0], q.shape()[1]);
        let h = d / dk;
        if y.shape() != &[t, d] || p.shape()[0] != 2 * t - 1 {
            return Err(BackendError::Shape("mhsa_rel_pos: shape mismatch".into()));
        }
        let (qd, kd, vd, qpd, pd) = (
            data(q).to_vec(), data(k).to_vec(), data(v).to_vec(),
            data(qp).to_vec(), data(p).to_vec(),
        );
        let (ud, vbd) = (data(u).to_vec(), data(vb).to_vec());
        let mut yd = data_mut(y);
        yd.fill(0.0);
        let mut wfd: Option<Vec<f32>> = wfull.map(|_| vec![0.0f32; h * t * t]);
        let scale = 1.0 / (dk as f32).sqrt();
        for hh in 0..h {
            let (qo, ko, vo, po) = (hh * dk, hh * dk, hh * dk, hh * dp);
            let (uo, vo_b) = (hh * dk, hh * dk);
            for qt in 0..t {
                let mut scores = vec![0.0f32; t];
                for tk in 0..t {
                    let mut ac = 0.0f32;
                    for j in 0..dk {
                        ac += (qd[qt * d + qo + j] + ud[uo + j]) * kd[tk * d + ko + j];
                    }
                    let mut bd = 0.0f32;
                    if mode == 0 {
                        // XL shift: position term reads query row
                        // ((tk+1)·T + qt)/(2T) and p offset ((tk+1)·T + qt) % (2T) − 1.
                        let flat = (tk + 1) * t + qt;
                        let qrow = flat / (2 * t);
                        let poff = flat % (2 * t);
                        if poff > 0 {
                            for j in 0..dp {
                                bd += (qd[qrow * d + qo + j] + vbd[vo_b + j]) * pd[(poff - 1) * (h * dp) + po + j];
                            }
                        }
                    } else {
                        let poff = t - 1 + tk - qt;
                        for j in 0..dp {
                            bd += (qpd[qt * (h * dp) + hh * dp + j] + vbd[hh * dp + j]) * pd[poff * (h * dp) + po + j];
                        }
                    }
                    scores[tk] = (ac + bd) * scale;
                }
                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in scores.iter_mut() {
                    let e = (*s - mx).exp();
                    *s = e;
                    sum += e;
                }
                if let Some(w) = wfd.as_mut() {
                    for (tk, &e) in scores.iter().enumerate() {
                        w[hh * t * t + qt * t + tk] = e / sum;
                    }
                }
                for (tk, &e) in scores.iter().enumerate() {
                    for j in 0..dk {
                        yd[qt * d + qo + j] += e * vd[tk * d + vo + j];
                    }
                }
                for j in 0..dk {
                    yd[qt * d + qo + j] /= sum;
                }
            }
        }
        if let (Some(wt), Some(w)) = (wfull, wfd) {
            let mut wd = data_mut(wt);
            wd.copy_from_slice(&w);
        }
        Ok(())
    }

    /// Per-head attention application over time (zipformer SelfAttention).
    fn mhsa_apply_weights(
        &self,
        w: &Tensor,
        x: &Tensor,
        y: &Tensor,
        vh: usize,
    ) -> Result<(), BackendError> {
        let (t, d) = (x.shape()[0], x.shape()[1]);
        let h = d / vh;
        let (wd, xd) = (data(w).to_vec(), data(x).to_vec());
        let mut yd = data_mut(y);
        for hh in 0..h {
            for qt in 0..t {
                for j in 0..vh {
                    let mut acc = 0.0f32;
                    for tk in 0..t {
                        acc += wd[hh * t * t + qt * t + tk] * xd[tk * d + hh * vh + j];
                    }
                    yd[qt * d + hh * vh + j] = acc;
                }
            }
        }
        Ok(())
    }

    fn uniform_attn(&self, v: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        // Y (B,C) = mean over time of V (B,T,C).
        let (b, t, c) = (v.shape()[0], v.shape()[1], v.shape()[2]);
        if y.shape() != &[b, c] {
            return Err(BackendError::Shape(
                "uniform_attn: output shape mismatch".into(),
            ));
        }
        let vd: Vec<f32> = data(v).to_vec();
        let mut yd = data_mut(y);
        for bi in 0..b {
            for j in 0..c {
                yd[bi * c + j] = (0..t).map(|ti| vd[(bi * t + ti) * c + j]).sum::<f32>() / t as f32;
            }
        }
        Ok(())
    }

    fn kv_update(&self, dst: &Tensor, src: &Tensor, pos: usize) -> Result<(), BackendError> {
        // dst (B, cap, D), src (B, Tq, D): write at [pos, pos+Tq).
        let (b, tq, d) = (src.shape()[0], src.shape()[1], src.shape()[2]);
        let cap = dst.shape()[1];
        if dst.shape()[2] != d || pos + tq > cap {
            return Err(BackendError::Shape("kv_update: capacity exceeded".into()));
        }
        let s = data(src);
        let mut dd = data_mut(dst);
        for bi in 0..b {
            for ti in 0..tq {
                let src_off = (bi * tq + ti) * d;
                let dst_off = (bi * cap + pos + ti) * d;
                dd[dst_off..dst_off + d].copy_from_slice(&s[src_off..src_off + d]);
            }
        }
        Ok(())
    }

    fn codebook_search(
        &self,
        x: &Tensor,
        table: &Tensor,
        idx: &mut [u32],
    ) -> Result<(), BackendError> {
        // Nearest L2 row of table (V, D) for each row of X (rows, D).
        let rows = x.shape()[0] * x.shape()[1];
        let d = x.shape()[2];
        let (v, td) = (table.shape()[0], table.shape()[1]);
        if td != d || idx.len() != rows {
            return Err(BackendError::Shape(
                "codebook_search: shape mismatch".into(),
            ));
        }
        let xd: Vec<f32> = data(x).to_vec();
        let tbl = data(table);
        for r in 0..rows {
            let mut best = f32::INFINITY;
            let mut best_i = 0u32;
            for vi in 0..v {
                let mut acc = 0.0f32;
                for j in 0..d {
                    let diff = xd[r * d + j] - tbl[vi * d + j];
                    acc += diff * diff;
                }
                if acc < best {
                    best = acc;
                    best_i = vi as u32;
                }
            }
            idx[r] = best_i;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn gemm_2x2() {
        let b = CpuBackend;
        let a = b.from_cpu_f32(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let bb = b.from_cpu_f32(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let c = b.zeros(vec![2, 2]).unwrap();
        b.gemm(&a, &bb, &c).unwrap();
        let got = b.to_cpu_f32(&c).unwrap();
        assert!(approx(got[0], 19.0, 1e-5) && approx(got[1], 22.0, 1e-5));
        assert!(approx(got[2], 43.0, 1e-5) && approx(got[3], 50.0, 1e-5));
    }

    #[test]
    fn gemm_nt() {
        let b = CpuBackend;
        let a = b.from_cpu_f32(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let bb = b.from_cpu_f32(&[1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap(); // identity (row-major N,K)
        let c = b.zeros(vec![2, 2]).unwrap();
        b.gemm_nt(&a, &bb, &c).unwrap();
        let got = b.to_cpu_f32(&c).unwrap();
        assert!(approx(got[0], 1.0, 1e-5) && approx(got[1], 2.0, 1e-5));
        assert!(approx(got[2], 3.0, 1e-5) && approx(got[3], 4.0, 1e-5));
    }

    #[test]
    fn rmsnorm_small() {
        let b = CpuBackend;
        let x = b.from_cpu_f32(&[1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let w = b.from_cpu_f32(&[1.0, 1.0, 1.0, 1.0], vec![4]).unwrap();
        let y = b.zeros(vec![1, 4]).unwrap();
        b.rmsnorm(&x, &w, &y, 1e-5).unwrap();
        let got = b.to_cpu_f32(&y).unwrap();
        let msq = (1.0f32 + 4.0 + 9.0 + 16.0) / 4.0;
        let inv = 1.0 / msq.sqrt();
        assert!(approx(got[0], inv, 1e-5));
        assert!(approx(got[3], 4.0 * inv, 1e-5));
    }

    #[test]
    fn conv1d_small() {
        let b = CpuBackend;
        // X (1,4,1) = [1,2,3,4]; W (1,1,3) = [1,1,1]; stride 1, LEFT pad 1, dil 1
        // → causal: Tout = (4 + 1 - 2 - 1)/1 + 1 = 3
        let x = b
            .from_cpu_f32(&[1.0, 2.0, 3.0, 4.0], vec![1, 4, 1])
            .unwrap();
        let w = b.from_cpu_f32(&[1.0, 1.0, 1.0], vec![1, 1, 3]).unwrap();
        let y = b.zeros(vec![1, 3, 1]).unwrap();
        b.conv1d(&x, &w, None, 1, 1, 0, 1, &y).unwrap();
        let got = b.to_cpu_f32(&y).unwrap();
        // y0 = x[-1]+x[0]+x[1] = 3; y1 = x[0]+x[1]+x[2] = 6; y2 = x[1]+x[2]+x[3] = 9
        assert!(approx(got[0], 3.0, 1e-5) && approx(got[1], 6.0, 1e-5));
        assert!(approx(got[2], 9.0, 1e-5));
    }

    #[test]
    fn slice_time_last() {
        let b = CpuBackend;
        let x = b
            .from_cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2])
            .unwrap();
        let y = b.zeros(vec![1, 2, 2]).unwrap();
        b.slice_time(&x, &y, 1).unwrap();
        let got = b.to_cpu_f32(&y).unwrap();
        assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0]);

        // slice_last: X (3,2) → Y (3,1), off=1: second column of each row.
        let x2 = b
            .from_cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2])
            .unwrap();
        let y2 = b.zeros(vec![3, 1]).unwrap();
        b.slice_last(&x2, &y2, 1).unwrap();
        let got2 = b.to_cpu_f32(&y2).unwrap();
        assert_eq!(got2, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn activations() {
        let b = CpuBackend;
        let x = b.from_cpu_f32(&[-1.0, 0.0, 1.0], vec![3]).unwrap();
        let y = b.zeros(vec![3]).unwrap();
        b.relu(&x, &y).unwrap();
        assert_eq!(b.to_cpu_f32(&y).unwrap(), vec![0.0, 0.0, 1.0]);
        b.silu(&x, &y).unwrap();
        let got = b.to_cpu_f32(&y).unwrap();
        assert!(approx(got[0], -1.0 / (1.0 + 1.0f32.exp()), 1e-6));
        assert!(approx(got[1], 0.0, 1e-6));
        assert!(approx(got[2], 1.0 / (1.0 + (-1.0f32).exp()), 1e-6));
    }

    #[test]
    fn snake_and_maxpool() {
        let b = CpuBackend;
        // snake: y = x + sin(a x)^2 / a
        let x = b.from_cpu_f32(&[0.5, 1.0], vec![2]).unwrap();
        let a = b.from_cpu_f32(&[2.0, 1.0], vec![2]).unwrap();
        let y = b.zeros(vec![2]).unwrap();
        b.snake(&x, &a, &y).unwrap();
        let got = b.to_cpu_f32(&y).unwrap();
        assert!(approx(got[0], 0.5 + (1.0f32).sin().powi(2) / 2.0, 1e-6));
        assert!(approx(got[1], 1.0 + (1.0f32).sin().powi(2), 1e-6));

        // maxpool2d (2,2,4,4) → (2,2,2,2)
        let m = b
            .from_cpu_f32(
                &(0..64).map(|v| v as f32).collect::<Vec<_>>(),
                vec![2, 2, 4, 4],
            )
            .unwrap();
        let p = b.zeros(vec![2, 2, 2, 2]).unwrap();
        b.maxpool2d(&m, &p, 2, 2).unwrap();
        let got = b.to_cpu_f32(&p).unwrap();
        assert!(approx(got[0], 5.0, 1e-5)); // row0 block max
        assert!(approx(got[3], 15.0, 1e-5));
    }

    #[test]
    fn softmax_and_glu() {
        let b = CpuBackend;
        let x = b.from_cpu_f32(&[1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let y = b.zeros(vec![1, 3]).unwrap();
        b.softmax_last_dim(&x, &y).unwrap();
        let got = b.to_cpu_f32(&y).unwrap();
        let sum = (-2.0f32).exp() + (-1.0f32).exp() + 1.0;
        assert!(approx(got[0], (-2.0f32).exp() / sum, 1e-6));
        assert!(approx(got[2], 1.0 / sum, 1e-6));

        // glu: x=[1,2,3,4], d=2 → y0 = 1·sigmoid(3), y1 = 2·sigmoid(4)
        let g = b.from_cpu_f32(&[1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let gy = b.zeros(vec![1, 2]).unwrap();
        b.glu(&g, &gy).unwrap();
        let got = b.to_cpu_f32(&gy).unwrap();
        assert!(approx(got[0], 1.0 / (1.0 + (-3.0f32).exp()), 1e-6));
        assert!(approx(got[1], 2.0 / (1.0 + (-4.0f32).exp()), 1e-6));
    }

    #[test]
    fn rope_forms() {
        let b = CpuBackend;
        // identity cos=1, sin=0 → output == input for halved and interleaved
        let x = b.from_cpu_f32(&[1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let cos = b.from_cpu_f32(&[1.0, 1.0], vec![1, 2]).unwrap();
        let sin = b.from_cpu_f32(&[0.0, 0.0], vec![1, 2]).unwrap();
        let y = b.zeros(vec![1, 4]).unwrap();
        b.rope(&x, &cos, &sin, &y, 4).unwrap();
        assert_eq!(b.to_cpu_f32(&y).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        // interleaved is 3D (B, T, H·hd) — single row (1, 1, 4)
        let x3 = b
            .from_cpu_f32(&[1.0, 2.0, 3.0, 4.0], vec![1, 1, 4])
            .unwrap();
        let y3 = b.zeros(vec![1, 1, 4]).unwrap();
        b.rope_interleaved(&x3, &cos, &sin, &y3, 4, 0).unwrap();
        assert_eq!(b.to_cpu_f32(&y3).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

        // non-trivial halved: cos=[1,0.5], sin=[0,0.5], rope_dim=4, half=2:
        // pairs (i, i+half): (0,2) and (1,3)
        // y0 = 1·1-3·0 = 1; y2 = 1·0+3·1 = 3; y1 = 2·0.5-4·0.5 = -1.0; y3 = 2·0.5+4·0.5 = 3.0
        let cos2 = b.from_cpu_f32(&[1.0, 0.5], vec![1, 2]).unwrap();
        let sin2 = b.from_cpu_f32(&[0.0, 0.5], vec![1, 2]).unwrap();
        b.rope(&x, &cos2, &sin2, &y, 4).unwrap();
        let got = b.to_cpu_f32(&y).unwrap();
        assert!(approx(got[0], 1.0, 1e-6) && approx(got[1], -1.0, 1e-6));
        assert!(approx(got[2], 3.0, 1e-6) && approx(got[3], 3.0, 1e-6));

        // rope_head: (1,1,4), h=2, hd=2, rope_dim=2; cos/sin indexed globally
        // by t only (same rotation for every head) → with cos=[1,0.5]/sin=[0,0.5]
        // at t=0 both heads use (c=1, s=0) → identity.
        let xh = b
            .from_cpu_f32(&[1.0, 2.0, 3.0, 4.0], vec![1, 1, 4])
            .unwrap();
        let yh = b.zeros(vec![1, 1, 4]).unwrap();
        b.rope_head(&xh, &cos2, &sin2, &yh, 2, 2).unwrap();
        let got = b.to_cpu_f32(&yh).unwrap();
        assert!(approx(got[0], 1.0, 1e-6) && approx(got[1], 2.0, 1e-6));
        assert!(approx(got[2], 3.0, 1e-6) && approx(got[3], 4.0, 1e-6));
    }

    #[test]
    fn embedding_glu_repeat_kv() {
        let b = CpuBackend;
        let table = b
            .from_cpu_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2])
            .unwrap();
        let y = b.zeros(vec![2, 2]).unwrap();
        b.embedding(&[2, 0], &table, &y).unwrap();
        assert_eq!(b.to_cpu_f32(&y).unwrap(), vec![5.0, 6.0, 1.0, 2.0]);

        // repeat ×2: (1,2,1) → (1,4,1)
        let x = b.from_cpu_f32(&[1.0, 2.0], vec![1, 2, 1]).unwrap();
        let y = b.zeros(vec![1, 4, 1]).unwrap();
        b.repeat(&x, &y).unwrap();
        assert_eq!(b.to_cpu_f32(&y).unwrap(), vec![1.0, 1.0, 2.0, 2.0]);

        // kv_update: dst (1,4,1) zeros, src (1,2,1)=[5,6], pos=1 → [0,5,6,0]
        let dst = b.zeros(vec![1, 4, 1]).unwrap();
        let src = b.from_cpu_f32(&[5.0, 6.0], vec![1, 2, 1]).unwrap();
        b.kv_update(&dst, &src, 1).unwrap();
        assert_eq!(b.to_cpu_f32(&dst).unwrap(), vec![0.0, 5.0, 6.0, 0.0]);
    }

    #[test]
    fn row_vec_and_head_norms() {
        let b = CpuBackend;
        let x = b.from_cpu_f32(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let w = b.from_cpu_f32(&[2.0, 3.0], vec![2]).unwrap();
        let y = b.zeros(vec![2, 2]).unwrap();
        b.mul_rows(&x, &w, &y).unwrap();
        assert_eq!(b.to_cpu_f32(&y).unwrap(), vec![2.0, 4.0, 9.0, 12.0]);
        b.mul_vec(&x, &w, &y).unwrap();
        assert_eq!(b.to_cpu_f32(&y).unwrap(), vec![2.0, 6.0, 6.0, 12.0]);

        // rmsnorm_head: (1,4), h=2, hd=2, w=[1,1] → per-head rmsnorm
        let xh = b.from_cpu_f32(&[1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let w1 = b.from_cpu_f32(&[1.0, 1.0], vec![2]).unwrap();
        let yh = b.zeros(vec![1, 4]).unwrap();
        b.rmsnorm_head(&xh, &w1, &yh, 2, 1e-5).unwrap();
        let got = b.to_cpu_f32(&yh).unwrap();
        let inv0 = 1.0 / (2.5f32).sqrt();
        let inv1 = 1.0 / (12.5f32).sqrt();
        assert!(approx(got[0], inv0, 1e-5) && approx(got[1], 2.0 * inv0, 1e-5));
        assert!(approx(got[2], 3.0 * inv1, 1e-5) && approx(got[3], 4.0 * inv1, 1e-5));

        // add_bias2d: elementwise (same shapes)
        let x2 = b
            .from_cpu_f32(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2])
            .unwrap();
        let bd = b
            .from_cpu_f32(&[10.0, 20.0, 10.0, 20.0], vec![1, 2, 2])
            .unwrap();
        let y2 = b.zeros(vec![1, 2, 2]).unwrap();
        b.add_bias2d(&x2, &bd, &y2).unwrap();
        assert_eq!(b.to_cpu_f32(&y2).unwrap(), vec![11.0, 22.0, 13.0, 24.0]);
    }

    #[test]
    fn uniform_codebook_attn() {
        let b = CpuBackend;
        let v = b
            .from_cpu_f32(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2])
            .unwrap();
        let y = b.zeros(vec![1, 2]).unwrap();
        b.uniform_attn(&v, &y).unwrap();
        assert_eq!(b.to_cpu_f32(&y).unwrap(), vec![2.0, 3.0]);

        // codebook: nearest row of [[0,0],[1,1]] to [0.1,0.2] → 0
        let x = b.from_cpu_f32(&[0.1, 0.2], vec![1, 1, 2]).unwrap();
        let table = b.from_cpu_f32(&[0.0, 0.0, 1.0, 1.0], vec![2, 2]).unwrap();
        let mut idx = [9u32; 1];
        b.codebook_search(&x, &table, &mut idx).unwrap();
        assert_eq!(idx, [0]);

        // attn_ar single token: q=k=[1,0], v=[5,6] → softmax of one → y=v
        let q = b.from_cpu_f32(&[1.0, 0.0], vec![1, 1, 2]).unwrap();
        let k = b.from_cpu_f32(&[1.0, 0.0], vec![1, 1, 2]).unwrap();
        let vv = b.from_cpu_f32(&[5.0, 6.0], vec![1, 1, 2]).unwrap();
        let ya = b.zeros(vec![1, 1, 2]).unwrap();
        b.attn_ar(&q, &k, &vv, None, &ya, 1, 1, 2).unwrap();
        let got = b.to_cpu_f32(&ya).unwrap();
        assert!(approx(got[0], 5.0, 1e-5) && approx(got[1], 6.0, 1e-5));

        // mhsa mode 2 (uniform): y = mean over time of v, broadcast per frame
        let qm = b
            .from_cpu_f32(&[1.0, 0.0, 0.0, 1.0], vec![1, 2, 2])
            .unwrap();
        let km = b
            .from_cpu_f32(&[1.0, 0.0, 0.0, 1.0], vec![1, 2, 2])
            .unwrap();
        let vm = b
            .from_cpu_f32(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2])
            .unwrap();
        let sc = b.zeros(vec![1, 1, 2, 2]).unwrap();
        let ym = b.zeros(vec![1, 2, 2]).unwrap();
        b.mhsa(&qm, &km, &vm, &sc, None, 2, &ym).unwrap();
        let got = b.to_cpu_f32(&ym).unwrap();
        assert!(approx(got[0], 2.0, 1e-5) && approx(got[1], 3.0, 1e-5));
        assert!(approx(got[2], 2.0, 1e-5) && approx(got[3], 3.0, 1e-5));

        // mhsa mode 0 with causal mask: single query, single key → y=v
        let q2 = b.from_cpu_f32(&[1.0, 0.0], vec![1, 1, 2]).unwrap();
        let k2 = b.from_cpu_f32(&[1.0, 0.0], vec![1, 1, 2]).unwrap();
        let mask = b.from_cpu_f32(&[0.0f32], vec![1, 1]).unwrap();
        let ym2 = b.zeros(vec![1, 1, 2]).unwrap();
        b.mhsa(&q2, &k2, &vv, &sc, Some(&mask), 0, &ym2).unwrap();
        let got = b.to_cpu_f32(&ym2).unwrap();
        assert!(approx(got[0], 5.0, 1e-5) && approx(got[1], 6.0, 1e-5));
    }
}
