//! Native NN layers on the [`Backend`] contract — backend-agnostic.
//!
//! Every layer holds its weights as [`Tensor`]s and dispatches through
//! `&dyn Backend`; each forward is one op call per kernel (batch fusion lands
//! with the model ports). The same layer code runs on the CPU reference
//! backend (unit tests / CI), the Metal backend (macOS, e2e oracle) and any
//! future CUDA/Vulkan/WebGPU backend.
//!
//! `KvCache` keeps Metal-compatible `*_metal` methods behind the `metal`
//! feature as a migration bridge for the speech crate; new code should use
//! the backend-based methods.

use crate::backend::{Backend, Tensor};
use crate::error::BackendError;

#[cfg(feature = "metal")]
use crate::metal::{GpuTensor, MetalCtx, MetalError};

/// 3D (B,T,D) → 2D (B*T, D) view over the same buffer (row-major).
fn view2(x: &Tensor) -> Tensor {
    let b = x.shape()[0] * x.shape()[1];
    let d = x.shape()[2];
    x.view(vec![b, d])
}

/// 2D → 3D (B,T,D) view.
fn view3(x: &Tensor, b: usize, t: usize, d: usize) -> Tensor {
    debug_assert_eq!(x.shape(), &[b * t, d]);
    x.view(vec![b, t, d])
}

/// Linear: y = x @ Wᵀ + b. W (out, in), x (B,T,in) → y (B,T,out).
pub struct Linear {
    pub w: Tensor,
    pub b: Option<Tensor>,
}

impl Linear {
    pub fn from_weights(
        backend: &dyn Backend,
        w: Vec<f32>,
        shape: Vec<usize>,
        bias: Option<Vec<f32>>,
    ) -> Result<Self, BackendError> {
        let out = shape[0];
        let w = backend.from_cpu_f32(&w, shape)?;
        let b = bias
            .map(|b| backend.from_cpu_f32(&b, vec![out]))
            .transpose()?;
        Ok(Self { w, b })
    }

    pub fn forward(&self, backend: &dyn Backend, x: &Tensor) -> Result<Tensor, BackendError> {
        let (b, t, _) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let out = self.w.shape()[0];
        let x2 = view2(x);
        let y = backend.zeros(vec![b * t, out])?;
        backend.gemm_nt(&x2, &self.w, &y)?;
        if let Some(bias) = &self.b {
            backend.add_bias(&y, bias, &y)?;
        }
        Ok(view3(&y, b, t, out))
    }
}

/// RMSNorm over the last dim. W (D).
pub struct RmsNorm {
    pub w: Tensor,
    pub eps: f32,
}

impl RmsNorm {
    pub fn from_weights(backend: &dyn Backend, w: Vec<f32>, eps: f32) -> Result<Self, BackendError> {
        let d = w.len();
        Ok(Self { w: backend.from_cpu_f32(&w, vec![d])?, eps })
    }

    pub fn forward(&self, backend: &dyn Backend, x: &Tensor) -> Result<Tensor, BackendError> {
        let (b, t, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let y = backend.zeros(vec![b * t, d])?;
        backend.rmsnorm(&view2(x), &self.w, &y, self.eps)?;
        Ok(view3(&y, b, t, d))
    }
}

/// SwiGLU FFN: y = (silu(x·W1ᵀ) ⊙ (x·W3ᵀ)) · W2ᵀ. All (D,D).
pub struct SwiGLU {
    pub w1: Tensor,
    pub w3: Tensor,
    pub w2: Tensor,
}

impl SwiGLU {
    pub fn from_weights(
        backend: &dyn Backend,
        w1: Vec<f32>,
        w3: Vec<f32>,
        w2: Vec<f32>,
        dim: usize,
    ) -> Result<Self, BackendError> {
        Ok(Self {
            w1: backend.from_cpu_f32(&w1, vec![dim, dim])?,
            w3: backend.from_cpu_f32(&w3, vec![dim, dim])?,
            w2: backend.from_cpu_f32(&w2, vec![dim, dim])?,
        })
    }

    pub fn forward(&self, backend: &dyn Backend, x: &Tensor) -> Result<Tensor, BackendError> {
        let (b, t, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let x2 = view2(x);
        let a = backend.zeros(vec![b * t, d])?;
        let g = backend.zeros(vec![b * t, d])?;
        let gm = backend.zeros(vec![b * t, d])?;
        backend.gemm_nt(&x2, &self.w1, &a)?;
        backend.gemm_nt(&x2, &self.w3, &g)?;
        backend.silu(&a, &a)?;
        backend.mul(&a, &g, &gm)?;
        let y = backend.zeros(vec![b * t, d])?;
        backend.gemm_nt(&gm, &self.w2, &y)?;
        Ok(view3(&y, b, t, d))
    }
}

/// Snake activation: y = x + sin²(a·x)/a. A (D).
pub struct Snake {
    pub a: Tensor,
}

impl Snake {
    pub fn from_weights(backend: &dyn Backend, a: Vec<f32>) -> Result<Self, BackendError> {
        let d = a.len();
        Ok(Self { a: backend.from_cpu_f32(&a, vec![d])? })
    }

    pub fn forward(&self, backend: &dyn Backend, x: &Tensor) -> Result<Tensor, BackendError> {
        let y = backend.zeros(x.shape().to_vec())?;
        backend.snake(x, &self.a, &y)?;
        Ok(y)
    }
}

/// Embedding lookup (batch fixed to 1 in the kernel). Table (N,D).
pub struct Embedding {
    pub table: Tensor,
}

impl Embedding {
    pub fn from_weights(backend: &dyn Backend, table: Vec<f32>, shape: Vec<usize>) -> Result<Self, BackendError> {
        Ok(Self { table: backend.from_cpu_f32(&table, shape)? })
    }

    pub fn forward(&self, backend: &dyn Backend, idx: &[u32], t: usize) -> Result<Tensor, BackendError> {
        let d = self.table.shape()[1];
        let y = backend.zeros(vec![1, t, d])?;
        backend.embedding(idx, &self.table, &y)?;
        Ok(y)
    }
}

/// Causal 1D convolution (groups = in channels): y[t] = Σ_k x[t - k·dil]·w[k].
/// W (out, in, k), pad = (k-1)·dil (left-padded, causal).
pub struct CausalConv1d {
    pub w: Tensor,
    pub b: Option<Tensor>,
    pub dil: usize,
}

impl CausalConv1d {
    pub fn from_weights(
        backend: &dyn Backend,
        w: Vec<f32>,
        shape: Vec<usize>,
        bias: Option<Vec<f32>>,
        dil: usize,
    ) -> Result<Self, BackendError> {
        let out = shape[0];
        let w = backend.from_cpu_f32(&w, shape)?;
        let b = bias
            .map(|b| backend.from_cpu_f32(&b, vec![out]))
            .transpose()?;
        Ok(Self { w, b, dil })
    }

    pub fn forward(&self, backend: &dyn Backend, x: &Tensor) -> Result<Tensor, BackendError> {
        let (b, t, _cin) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let (cout, _, k) = (self.w.shape()[0], self.w.shape()[1], self.w.shape()[2]);
        let pad = (k - 1) * self.dil;
        let t_out = t + pad - self.dil * (k - 1);
        let y = backend.zeros(vec![b, t_out, cout])?;
        backend.conv1d(x, &self.w, self.b.as_ref(), 1, pad, 0, self.dil, &y)?;
        Ok(y)
    }
}

/// Causal transposed 1D convolution (stride upsample), drops the last
/// (k - stride) frames. W (in, out, k).
pub struct ConvTranspose1d {
    pub w: Tensor,
    pub b: Option<Tensor>,
    pub stride: usize,
}

impl ConvTranspose1d {
    pub fn from_weights(
        backend: &dyn Backend,
        w: Vec<f32>,
        shape: Vec<usize>,
        bias: Option<Vec<f32>>,
        stride: usize,
    ) -> Result<Self, BackendError> {
        let cout = shape[1];
        let w = backend.from_cpu_f32(&w, shape)?;
        let b = bias
            .map(|b| backend.from_cpu_f32(&b, vec![cout]))
            .transpose()?;
        Ok(Self { w, b, stride })
    }

    pub fn forward(&self, backend: &dyn Backend, x: &Tensor) -> Result<Tensor, BackendError> {
        let (b, t, _) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let (_, cout, k) = (self.w.shape()[0], self.w.shape()[1], self.w.shape()[2]);
        let t_out = (t - 1) * self.stride + k;
        let y = backend.zeros(vec![b, t_out, cout])?;
        backend.conv_transpose1d(x, &self.w, self.b.as_ref(), self.stride, &y)?;
        let unpad = k.saturating_sub(self.stride);
        if unpad > 0 {
            let kept = backend.zeros(vec![b, t_out - unpad, cout])?;
            backend.slice_time(&y, &kept, unpad)?;
            Ok(kept)
        } else {
            Ok(y)
        }
    }
}

/// GQA attention + KV cache for the slow/fast AR paths.
/// wqkv (3D, D), wo (D, D), optional QK-norms (D).
pub struct Attention {
    pub wqkv: Tensor,
    pub wo: Tensor,
    pub q_norm: Option<Tensor>,
    pub k_norm: Option<Tensor>,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    pub fn from_weights(
        backend: &dyn Backend,
        wqkv: Vec<f32>,
        wo: Vec<f32>,
        q_norm: Option<Vec<f32>>,
        k_norm: Option<Vec<f32>>,
        dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, BackendError> {
        Ok(Self {
            wqkv: backend.from_cpu_f32(&wqkv, vec![3 * dim, dim])?,
            wo: backend.from_cpu_f32(&wo, vec![dim, dim])?,
            q_norm: q_norm
                .map(|w| backend.from_cpu_f32(&w, vec![dim]))
                .transpose()?,
            k_norm: k_norm
                .map(|w| backend.from_cpu_f32(&w, vec![dim]))
                .transpose()?,
            n_heads,
            n_kv_heads,
            head_dim,
        })
    }

    /// Forward one attention step: x (1, Tq, D) → attn out (1, Tq, D).
    /// `pos` — absolute RoPE offset of the FIRST query token; `t` — query
    /// length (t > 1 → causal mask over the whole cached KV).
    pub fn forward(
        &self,
        backend: &dyn Backend,
        x: &Tensor,
        cache: &mut KvCache,
        pos: usize,
        t: usize,
        rope: &(Tensor, Tensor),
    ) -> Result<Tensor, BackendError> {
        let (b, tq, _) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let q_size = self.n_heads * self.head_dim;
        let kv_size = self.n_kv_heads * self.head_dim;
        let qkv = Linear { w: self.wqkv.clone(), b: None }.forward(backend, x)?;
        let q = slice_last(backend, &qkv, q_size, 0)?;
        let k = slice_last(backend, &qkv, kv_size, q_size)?;
        let v = slice_last(backend, &qkv, kv_size, q_size + kv_size)?;
        let (q, k) = if let (Some(qn), Some(kn)) = (&self.q_norm, &self.k_norm) {
            (rmsnorm_head(backend, &q, qn, self.n_heads)?, rmsnorm_head(backend, &k, kn, self.n_kv_heads)?)
        } else {
            (q, k)
        };
        let q = rope_apply(backend, &q, rope, pos, self.head_dim)?;
        let k = rope_apply(backend, &k, rope, pos, self.head_dim)?;
        cache.update(backend, &k, &v)?;
        let ck = cache.view();
        let cv = cache.view_v();
        let mask = if t > 1 {
            Some(build_causal_mask(backend, t, cache.pos, pos)?)
        } else {
            None
        };
        let attn = backend.zeros(vec![b, tq, q_size])?;
        backend.attn_ar(&q, &ck, &cv, mask.as_ref(), &attn, self.n_heads, self.n_kv_heads, self.head_dim)?;
        let out = Linear { w: self.wo.clone(), b: None }.forward(backend, &attn)?;
        Ok(out)
    }
}

/// Growing KV cache (K/V rows (1, cap, D)); `pos` advances on update.
pub struct KvCache {
    k: Tensor,
    v: Tensor,
    pub pos: usize,
    cap: usize,
}

const KV_CACHE_INIT_CAPACITY: usize = 256;

impl KvCache {
    /// Create with the initial capacity; grows geometrically on demand.
    pub fn new(backend: &dyn Backend, n_kv_heads: usize, head_dim: usize) -> Result<Self, BackendError> {
        let cap = KV_CACHE_INIT_CAPACITY;
        let d = n_kv_heads * head_dim;
        Ok(Self {
            k: backend.zeros(vec![1, cap, d])?,
            v: backend.zeros(vec![1, cap, d])?,
            pos: 0,
            cap,
        })
    }

    fn ensure(&mut self, backend: &dyn Backend, needed: usize) -> Result<(), BackendError> {
        if needed <= self.cap {
            return Ok(());
        }
        let mut new_cap = self.cap;
        while new_cap < needed {
            new_cap = new_cap.saturating_mul(2);
        }
        let d = self.k.shape()[2];
        let old_k = std::mem::replace(&mut self.k, backend.zeros(vec![1, new_cap, d])?);
        let old_v = std::mem::replace(&mut self.v, backend.zeros(vec![1, new_cap, d])?);
        if self.pos > 0 {
            backend.kv_update(&self.k, &old_k, 0)?;
            backend.kv_update(&self.v, &old_v, 0)?;
        }
        self.cap = new_cap;
        Ok(())
    }

    /// Insert k/v (1, Tq, D) at the current position; advances `pos`.
    /// Quantized KV formats quantize on write inside the backend op.
    pub fn update(&mut self, backend: &dyn Backend, k: &Tensor, v: &Tensor) -> Result<(), BackendError> {
        let tq = k.shape()[1];
        self.ensure(backend, self.pos + tq)?;
        backend.kv_update(&self.k, k, self.pos)?;
        backend.kv_update(&self.v, v, self.pos)?;
        self.pos += tq;
        Ok(())
    }

    /// View of the first `pos` rows (same buffer, restricted shape).
    pub fn view(&self) -> Tensor {
        let d = self.k.shape()[2];
        self.k.view_restricted(vec![1, self.pos, d])
    }
    pub fn view_v(&self) -> Tensor {
        let d = self.v.shape()[2];
        self.v.view_restricted(vec![1, self.pos, d])
    }

    pub fn reset(&mut self) {
        self.pos = 0;
    }

    // ── Metal migration bridge (speech crate; remove after 1.6) ──

    #[cfg(feature = "metal")]
    pub fn new_metal(ctx: &MetalCtx, n_kv_heads: usize, head_dim: usize) -> Result<Self, MetalError> {
        let cap = KV_CACHE_INIT_CAPACITY;
        let d = n_kv_heads * head_dim;
        Ok(Self {
            k: Tensor::from_gpu(GpuTensor::zeros(&ctx.device, vec![1, cap, d])?),
            v: Tensor::from_gpu(GpuTensor::zeros(&ctx.device, vec![1, cap, d])?),
            pos: 0,
            cap,
        })
    }

    #[cfg(feature = "metal")]
    fn ensure_metal(&mut self, ctx: &MetalCtx, needed: usize) -> Result<(), MetalError> {
        if needed <= self.cap {
            return Ok(());
        }
        let mut new_cap = self.cap;
        while new_cap < needed {
            new_cap = new_cap.saturating_mul(2);
        }
        let d = self.k.shape()[2];
        let old_k = std::mem::replace(&mut self.k, Tensor::from_gpu(GpuTensor::zeros(&ctx.device, vec![1, new_cap, d])?));
        let old_v = std::mem::replace(&mut self.v, Tensor::from_gpu(GpuTensor::zeros(&ctx.device, vec![1, new_cap, d])?));
        if self.pos > 0 {
            ctx.kv_update_f32(self.k.as_gpu(), old_k.as_gpu(), 0)?;
            ctx.kv_update_f32(self.v.as_gpu(), old_v.as_gpu(), 0)?;
        }
        self.cap = new_cap;
        Ok(())
    }

    #[cfg(feature = "metal")]
    pub fn update_metal(&mut self, ctx: &MetalCtx, k: &GpuTensor, v: &GpuTensor) -> Result<(), MetalError> {
        let tq = k.shape[1];
        self.ensure_metal(ctx, self.pos + tq)?;
        ctx.kv_update_f32(self.k.as_gpu(), k, self.pos)?;
        ctx.kv_update_f32(self.v.as_gpu(), v, self.pos)?;
        self.pos += tq;
        Ok(())
    }

    #[cfg(feature = "metal")]
    pub fn view_metal(&self) -> GpuTensor {
        GpuTensor { buffer: self.k.as_gpu().buffer.clone(), shape: vec![1, self.pos, self.k.shape()[2]] }
    }

    #[cfg(feature = "metal")]
    pub fn view_v_metal(&self) -> GpuTensor {
        GpuTensor { buffer: self.v.as_gpu().buffer.clone(), shape: vec![1, self.pos, self.v.shape()[2]] }
    }
}

// ─────────────────────── operator helpers ───────────────────────

/// Slice the last dim: X (rows, D) → Y (rows, d0), starting at `off`.
pub fn slice_last(backend: &dyn Backend, x: &Tensor, d0: usize, off: usize) -> Result<Tensor, BackendError> {
    let (b, t, _d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    let rows = b * t;
    let y = backend.zeros(vec![rows, d0])?;
    backend.slice_last(&view2(x), &y, off)?;
    Ok(view3(&y, b, t, d0))
}

/// RMSNorm per head: X (B,T,H*HD), W (H*HD) → Y (B,T,H*HD).
fn rmsnorm_head(backend: &dyn Backend, x: &Tensor, w: &Tensor, n_heads: usize) -> Result<Tensor, BackendError> {
    let y = backend.zeros(x.shape().to_vec())?;
    backend.rmsnorm_head(x, w, &y, n_heads, 1e-6)?;
    Ok(y)
}

/// Interleaved RoPE at absolute position `pos` (cos/sin (max_seq, HD/2)).
fn rope_apply(
    backend: &dyn Backend,
    x: &Tensor,
    rope: &(Tensor, Tensor),
    pos: usize,
    hd: usize,
) -> Result<Tensor, BackendError> {
    let y = backend.zeros(x.shape().to_vec())?;
    backend.rope_interleaved(x, &rope.0, &rope.1, &y, hd, pos)?;
    Ok(y)
}

/// Causal attention mask: (1, Tq, Tkv), 0.0 for j ≤ pos + i, -inf elsewhere.
pub fn build_causal_mask(
    backend: &dyn Backend,
    tq: usize,
    tkv: usize,
    pos: usize,
) -> Result<Tensor, BackendError> {
    let mut data = vec![0.0f32; tq * tkv];
    for i in 0..tq {
        for j in 0..tkv {
            if j as isize > (pos + i) as isize {
                data[i * tkv + j] = f32::NEG_INFINITY;
            }
        }
    }
    backend.from_cpu_f32(&data, vec![1, tq, tkv])
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    use crate::cpu::CpuBackend;

    fn backend() -> CpuBackend {
        CpuBackend
    }

    #[test]
    fn linear_forward() {
        let b = backend();
        // W = identity (2x2), x = [[1,2],[3,4]] (B=1,T=2,D=2) → y = x
        let lin = Linear::from_weights(&b, vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], Some(vec![0.0, 0.0])).unwrap();
        let x = b.from_cpu_f32(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]).unwrap();
        let y = lin.forward(&b, &x).unwrap();
        let got = b.to_cpu_f32(&y).unwrap();
        assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0]);

        // bias shifts
        let lin = Linear::from_weights(&b, vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], Some(vec![10.0, 20.0])).unwrap();
        let y = lin.forward(&b, &x).unwrap();
        let got = b.to_cpu_f32(&y).unwrap();
        assert_eq!(got, vec![11.0, 22.0, 13.0, 24.0]);
    }

    #[test]
    fn rmsnorm_layer() {
        let b = backend();
        let rn = RmsNorm::from_weights(&b, vec![1.0, 1.0], 1e-5).unwrap();
        let x = b.from_cpu_f32(&[1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]).unwrap();
        let y = rn.forward(&b, &x).unwrap();
        let got = b.to_cpu_f32(&y).unwrap();
        let inv = 1.0 / (2.5f32).sqrt();
        assert!((got[0] - inv).abs() < 1e-5 && (got[1] - 2.0 * inv).abs() < 1e-5);
    }

    #[test]
    fn swiglu_snake_embedding() {
        let b = backend();
        let sg = SwiGLU::from_weights(&b, vec![1.0, 0.0, 0.0, 1.0], vec![1.0, 0.0, 0.0, 1.0], vec![1.0, 0.0, 0.0, 1.0], 2).unwrap();
        let x = b.from_cpu_f32(&[1.0, 2.0], vec![1, 1, 2]).unwrap();
        let y = sg.forward(&b, &x).unwrap();
        let got = b.to_cpu_f32(&y).unwrap();
        // a = silu(x) = x·sigmoid(x), g = x → y = silu(x)⊙x
        let sig = 1.0 / (1.0 + (-1.0f32).exp());
        assert!((got[0] - sig).abs() < 1e-5); // silu(1)·1
        assert!((got[1] - 2.0 * (2.0 / (1.0 + (-2.0f32).exp()))).abs() < 1e-5);

        let sn = Snake::from_weights(&b, vec![1.0, 1.0]).unwrap();
        let y = sn.forward(&b, &x).unwrap();
        let got = b.to_cpu_f32(&y).unwrap();
        assert!((got[0] - (1.0 + (1.0f32).sin().powi(2))).abs() < 1e-5);

        let emb = Embedding::from_weights(&b, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
        let y = emb.forward(&b, &[2, 0], 2).unwrap();
        let got = b.to_cpu_f32(&y).unwrap();
        assert_eq!(got, vec![5.0, 6.0, 1.0, 2.0]);
    }

    #[test]
    fn conv_layers() {
        let b = backend();
        // causal conv: k=3, dil=1 → pad=2, T_out = T
        let cc = CausalConv1d::from_weights(&b, vec![1.0, 1.0, 1.0], vec![1, 1, 3], None, 1).unwrap();
        let x = b.from_cpu_f32(&[1.0, 2.0, 3.0, 4.0], vec![1, 4, 1]).unwrap();
        let y = cc.forward(&b, &x).unwrap();
        let got = b.to_cpu_f32(&y).unwrap();
        // causal: y[t] = x[t] + x[t-1] + x[t-2] (left-padded)
        assert_eq!(got, vec![1.0, 3.0, 6.0, 9.0]);

        let ct = ConvTranspose1d::from_weights(&b, vec![1.0], vec![1, 1, 1], None, 2).unwrap();
        let y = ct.forward(&b, &x).unwrap();
        let got = b.to_cpu_f32(&y).unwrap();
        // sparse transposed conv (stride 2, k=1): y[2t] = x[t], y[2t+1] = 0
        assert_eq!(got, vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0]);
    }

    #[test]
    fn attention_with_kv_cache() {
        let b = backend();
        // single head, hd=2, dim=2; identity projections.
        let attn = Attention::from_weights(
            &b,
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], // wqkv (6,2): Q=K=V=identity
            vec![1.0, 0.0, 0.0, 1.0], // wo identity
            None,
            None,
            2,
            1,
            1,
            2,
        )
        .unwrap();
        let rope = (
            b.from_cpu_f32(&[1.0, 1.0, 1.0, 1.0], vec![4, 1]).unwrap(), // cos = 1 (max_seq, hd/2)
            b.from_cpu_f32(&[0.0, 0.0, 0.0, 0.0], vec![4, 1]).unwrap(), // sin = 0
        );
        let mut cache = KvCache::new(&b, 1, 2).unwrap();
        // step 1: x=[1,0] → attention over itself → y = x
        let x1 = b.from_cpu_f32(&[1.0, 0.0], vec![1, 1, 2]).unwrap();
        let y1 = attn.forward(&b, &x1, &mut cache, 0, 1, &rope).unwrap();
        let got1 = b.to_cpu_f32(&y1).unwrap();
        assert!((got1[0] - 1.0).abs() < 1e-5 && (got1[1] - 0.0).abs() < 1e-5);

        // step 2: x=[0,1] → q=[0,1]; cached K/V = [[1,0],[0,1]]; causal mask
        // opens BOTH positions (j ≤ pos+i = 1). scores = [0, 1]/√2 →
        // softmax weights [0.3303, 0.6697] over V rows → out = [0.3303, 0.6697].
        let x2 = b.from_cpu_f32(&[0.0, 1.0], vec![1, 1, 2]).unwrap();
        let y2 = attn.forward(&b, &x2, &mut cache, 1, 1, &rope).unwrap();
        let got2 = b.to_cpu_f32(&y2).unwrap();
        let w0 = 1.0 / (1.0 + (1.0f32 / 2.0f32.sqrt()).exp());
        let w1 = 1.0 - w0;
        assert!((got2[0] - w0).abs() < 1e-5 && (got2[1] - w1).abs() < 1e-5);

        // kv cache content: pos 2, k contains [1,0],[0,1]
        assert_eq!(cache.pos, 2);
        let kv = b.to_cpu_f32(&cache.view()).unwrap();
        assert_eq!(&kv[..4], &[1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn kv_cache_growth() {
        let b = backend();
        let mut cache = KvCache::new(&b, 1, 2).unwrap();
        for i in 0..300u32 {
            let v = i as f32;
            let k = b.from_cpu_f32(&[v, 0.0], vec![1, 1, 2]).unwrap();
            cache.update(&b, &k, &k).unwrap();
        }
        assert_eq!(cache.pos, 300);
        assert!(cache.cap >= 300);
        let kv = b.to_cpu_f32(&cache.view()).unwrap();
        assert_eq!(kv[0], 0.0);
        assert_eq!(kv[(299 * 2) as usize], 299.0);
    }
}
