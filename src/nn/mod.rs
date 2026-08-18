//! Native NN layers on the Vassista Metal engine — the candle-replacement
//! layer API (macOS). Every layer holds its weights as [`GpuTensor`]s and
//! dispatches its kernels through [`MetalCtx`]; each forward is one command
//! buffer per kernel call (batch fusion lands together with the model ports).
//!
//! The candle-based layer wrappers (for the CUDA/Linux path) live in
//! [`crate::nn_candle`], gated behind the `candle` feature.

use crate::metal::{GpuTensor, MetalCtx, MetalError};

/// 3D (B,T,D) → 2D (B*T, D) view over the same buffer (row-major).
fn view2(x: &GpuTensor) -> GpuTensor {
    GpuTensor {
        buffer: x.buffer.clone(),
        shape: vec![x.shape[0] * x.shape[1], x.shape[2]],
    }
}

/// 2D → 3D (B,T,D) view.
fn view3(x: &GpuTensor, b: usize, t: usize, d: usize) -> GpuTensor {
    debug_assert_eq!(x.shape, vec![b * t, d]);
    GpuTensor { buffer: x.buffer.clone(), shape: vec![b, t, d] }
}

/// Linear: y = x @ Wᵀ + b. W (out, in), x (B,T,in) → y (B,T,out).
pub struct Linear {
    pub w: GpuTensor,
    pub b: Option<GpuTensor>,
}

impl Linear {
    pub fn from_weights(
        ctx: &MetalCtx,
        w: Vec<f32>,
        shape: Vec<usize>,
        bias: Option<Vec<f32>>,
    ) -> Result<Self, MetalError> {
        let w = GpuTensor::from_cpu_f32(&ctx.device, &w, shape)?;
        let b = bias
            .map(|b| GpuTensor::from_cpu_f32(&ctx.device, &b, vec![w.shape[0]]))
            .transpose()?;
        Ok(Self { w, b })
    }

    pub fn forward(&self, ctx: &MetalCtx, x: &GpuTensor) -> Result<GpuTensor, MetalError> {
        let (b, t, _) = (x.shape[0], x.shape[1], x.shape[2]);
        let out = self.w.shape[0];
        let x2 = view2(x);
        let y = GpuTensor::zeros(&ctx.device, vec![b * t, out])?;
        ctx.gemm_nt_f32(&x2, &self.w, &y)?;
        if let Some(bias) = &self.b {
            ctx.add_bias_f32(&y, bias, &y)?;
        }
        Ok(view3(&y, b, t, out))
    }
}

/// RMSNorm over the last dim. W (D).
pub struct RmsNorm {
    pub w: GpuTensor,
    pub eps: f32,
}

impl RmsNorm {
    pub fn from_weights(ctx: &MetalCtx, w: Vec<f32>, eps: f32) -> Result<Self, MetalError> {
        Ok(Self { w: GpuTensor::from_cpu_f32(&ctx.device, &w, vec![w.len()])?, eps })
    }

    pub fn forward(&self, ctx: &MetalCtx, x: &GpuTensor) -> Result<GpuTensor, MetalError> {
        let rows = x.shape[0] * x.shape[1];
        let d = x.shape[2];
        let y = GpuTensor::zeros(&ctx.device, vec![rows, d])?;
        ctx.rmsnorm_f32(&view2(x), &self.w, &y, self.eps)?;
        Ok(view3(&y, x.shape[0], x.shape[1], d))
    }
}

/// SwiGLU FFN: y = (silu(x·W1ᵀ) ⊙ (x·W3ᵀ)) · W2ᵀ. All (D,D).
pub struct SwiGLU {
    pub w1: GpuTensor,
    pub w3: GpuTensor,
    pub w2: GpuTensor,
}

impl SwiGLU {
    pub fn from_weights(
        ctx: &MetalCtx,
        w1: Vec<f32>,
        w3: Vec<f32>,
        w2: Vec<f32>,
        dim: usize,
    ) -> Result<Self, MetalError> {
        let mk = |w: Vec<f32>| GpuTensor::from_cpu_f32(&ctx.device, &w, vec![dim, dim]);
        Ok(Self { w1: mk(w1)?, w3: mk(w3)?, w2: mk(w2)? })
    }

    pub fn forward(&self, ctx: &MetalCtx, x: &GpuTensor) -> Result<GpuTensor, MetalError> {
        let (b, t, d) = (x.shape[0], x.shape[1], x.shape[2]);
        let x2 = view2(x);
        let a = GpuTensor::zeros(&ctx.device, vec![b * t, d])?;
        let g = GpuTensor::zeros(&ctx.device, vec![b * t, d])?;
        let gm = GpuTensor::zeros(&ctx.device, vec![b * t, d])?;
        ctx.gemm_nt_f32(&x2, &self.w1, &a)?;
        ctx.gemm_nt_f32(&x2, &self.w3, &g)?;
        ctx.silu_f32(&a, &a)?;
        ctx.mul_f32(&a, &g, &gm)?;
        let y = GpuTensor::zeros(&ctx.device, vec![b * t, d])?;
        ctx.gemm_nt_f32(&gm, &self.w2, &y)?;
        Ok(view3(&y, b, t, d))
    }
}

/// Snake activation: y = x + sin²(a·x)/a. A (D).
pub struct Snake {
    pub a: GpuTensor,
}

impl Snake {
    pub fn from_weights(ctx: &MetalCtx, a: Vec<f32>) -> Result<Self, MetalError> {
        Ok(Self { a: GpuTensor::from_cpu_f32(&ctx.device, &a, vec![a.len()])? })
    }

    pub fn forward(&self, ctx: &MetalCtx, x: &GpuTensor) -> Result<GpuTensor, MetalError> {
        let y = GpuTensor::zeros(&ctx.device, x.shape.clone())?;
        ctx.snake_f32(x, &self.a, &y)?;
        Ok(y)
    }
}

/// Embedding lookup (batch fixed to 1 in the kernel). Table (N,D).
pub struct Embedding {
    pub table: GpuTensor,
}

impl Embedding {
    pub fn from_weights(ctx: &MetalCtx, table: Vec<f32>, shape: Vec<usize>) -> Result<Self, MetalError> {
        Ok(Self { table: GpuTensor::from_cpu_f32(&ctx.device, &table, shape)? })
    }

    pub fn forward(&self, ctx: &MetalCtx, idx: &[u32], t: usize) -> Result<GpuTensor, MetalError> {
        let d = self.table.shape[1];
        let y = GpuTensor::zeros(&ctx.device, vec![1, t, d])?;
        ctx.embedding_f32(idx, &self.table, &y)?;
        Ok(y)
    }
}

/// Causal 1D convolution (groups = in channels): y[t] = Σ_k x[t - k·dil]·w[k].
/// W (out, in, k), pad = (k-1)·dil (left-padded, causal).
pub struct CausalConv1d {
    pub w: GpuTensor,
    pub b: Option<GpuTensor>,
    pub dil: usize,
}

impl CausalConv1d {
    pub fn from_weights(
        ctx: &MetalCtx,
        w: Vec<f32>,
        shape: Vec<usize>,
        bias: Option<Vec<f32>>,
        dil: usize,
    ) -> Result<Self, MetalError> {
        let w = GpuTensor::from_cpu_f32(&ctx.device, &w, shape)?;
        let b = bias
            .map(|b| GpuTensor::from_cpu_f32(&ctx.device, &b, vec![w.shape[0]]))
            .transpose()?;
        Ok(Self { w, b, dil })
    }

    pub fn forward(&self, ctx: &MetalCtx, x: &GpuTensor) -> Result<GpuTensor, MetalError> {
        let (b, t, cin) = (x.shape[0], x.shape[1], x.shape[2]);
        let (cout, _, k) = (self.w.shape[0], self.w.shape[1], self.w.shape[2]);
        let pad = (k - 1) * self.dil;
        let t_out = t + pad - self.dil * (k - 1);
        let y = GpuTensor::zeros(&ctx.device, vec![b, t_out, cout])?;
        let params = crate::metal::Conv1dParams { stride: 1, pad, dilation: self.dil };
        // Dense (groups=1): every out channel reads all `cin` input channels.
        let dense = self.w.shape[1] == cin;
        ctx.conv1d_groups_f32(x, &self.w, self.b.as_ref(), params, 0, true, true, dense, &y)?;
        Ok(y)
    }
}

/// Causal transposed 1D convolution (stride upsample), drops the last
/// (k - stride) frames. W (in, out, k).
pub struct ConvTranspose1d {
    pub w: GpuTensor,
    pub b: Option<GpuTensor>,
    pub stride: usize,
}

impl ConvTranspose1d {
    pub fn from_weights(
        ctx: &MetalCtx,
        w: Vec<f32>,
        shape: Vec<usize>,
        bias: Option<Vec<f32>>,
        stride: usize,
    ) -> Result<Self, MetalError> {
        let w = GpuTensor::from_cpu_f32(&ctx.device, &w, shape)?;
        let b = bias
            .map(|b| GpuTensor::from_cpu_f32(&ctx.device, &b, vec![w.shape[1]]))
            .transpose()?;
        Ok(Self { w, b, stride })
    }

    pub fn forward(&self, ctx: &MetalCtx, x: &GpuTensor) -> Result<GpuTensor, MetalError> {
        let (b, t, _) = (x.shape[0], x.shape[1], x.shape[2]);
        let (_, cout, k) = (self.w.shape[0], self.w.shape[1], self.w.shape[2]);
        let t_out = (t - 1) * self.stride + k;
        let y = GpuTensor::zeros(&ctx.device, vec![b, t_out, cout])?;
        ctx.conv_transpose1d_f32(x, &self.w, self.b.as_ref(), self.stride, &y)?;
        let unpad = k.saturating_sub(self.stride);
        if unpad > 0 {
            let kept = GpuTensor::zeros(&ctx.device, vec![b, t_out - unpad, cout])?;
            ctx.slice_time_f32(&y, &kept, unpad)?;
            Ok(kept)
        } else {
            Ok(y)
        }
    }
}

/// GQA attention + KV cache for the slow/fast AR paths.
/// wqkv (3D, D), wo (D, D), optional QK-norms (D).
pub struct Attention {
    pub wqkv: GpuTensor,
    pub wo: GpuTensor,
    pub q_norm: Option<GpuTensor>,
    pub k_norm: Option<GpuTensor>,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    pub fn from_weights(
        ctx: &MetalCtx,
        wqkv: Vec<f32>,
        wo: Vec<f32>,
        q_norm: Option<Vec<f32>>,
        k_norm: Option<Vec<f32>>,
        dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, MetalError> {
        Ok(Self {
            wqkv: GpuTensor::from_cpu_f32(&ctx.device, &wqkv, vec![3 * dim, dim])?,
            wo: GpuTensor::from_cpu_f32(&ctx.device, &wo, vec![dim, dim])?,
            q_norm: q_norm
                .map(|w| GpuTensor::from_cpu_f32(&ctx.device, &w, vec![dim]))
                .transpose()?,
            k_norm: k_norm
                .map(|w| GpuTensor::from_cpu_f32(&ctx.device, &w, vec![dim]))
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
        ctx: &MetalCtx,
        x: &GpuTensor,
        cache: &mut KvCache,
        pos: usize,
        t: usize,
        rope: &(GpuTensor, GpuTensor),
    ) -> Result<GpuTensor, MetalError> {
        let d = self.wqkv.shape[1];
        let (b, tq, _) = (x.shape[0], x.shape[1], x.shape[2]);
        let q_size = self.n_heads * self.head_dim;
        let kv_size = self.n_kv_heads * self.head_dim;
        let qkv = Linear { w: self.wqkv.clone(), b: None }.forward(ctx, x)?;
        let q = slice_last(ctx, &qkv, q_size, 0)?;
        let k = slice_last(ctx, &qkv, kv_size, q_size)?;
        let v = slice_last(ctx, &qkv, kv_size, q_size + kv_size)?;
        let (q, k) = if let (Some(qn), Some(kn)) = (&self.q_norm, &self.k_norm) {
            (rmsnorm_head(ctx, &q, qn, self.n_heads)?, rmsnorm_head(ctx, &k, kn, self.n_kv_heads)?)
        } else {
            (q, k)
        };
        let q = rope_apply(ctx, &q, rope, pos, self.head_dim)?;
        let k = rope_apply(ctx, &k, rope, pos, self.head_dim)?;
        cache.update(ctx, &k, &v)?;
        let ck = cache.view();
        let cv = cache.view_v();
        let mask = if t > 1 {
            Some(build_causal_mask(ctx, t, cache.pos, pos)?)
        } else {
            None
        };
        let attn = GpuTensor::zeros(&ctx.device, vec![b, tq, q_size])?;
        ctx.attn_ar_f32(&q, &ck, &cv, mask.as_ref(), &attn, self.n_heads, self.n_kv_heads, self.head_dim)?;
        let out = Linear { w: self.wo.clone(), b: None }.forward(ctx, &attn)?;
        let _ = d;
        Ok(out)
    }
}

/// Growing KV cache (K/V rows (1, cap, D)); `pos` advances on update.
pub struct KvCache {
    k: GpuTensor,
    v: GpuTensor,
    pub pos: usize,
    cap: usize,
}

const KV_CACHE_INIT_CAPACITY: usize = 256;

impl KvCache {
    pub fn new(ctx: &MetalCtx, n_kv_heads: usize, head_dim: usize) -> Result<Self, MetalError> {
        let cap = KV_CACHE_INIT_CAPACITY;
        Ok(Self {
            k: GpuTensor::zeros(&ctx.device, vec![1, cap, n_kv_heads * head_dim])?,
            v: GpuTensor::zeros(&ctx.device, vec![1, cap, n_kv_heads * head_dim])?,
            pos: 0,
            cap,
        })
    }

    fn ensure(&mut self, ctx: &MetalCtx, needed: usize) -> Result<(), MetalError> {
        if needed <= self.cap {
            return Ok(());
        }
        let mut new_cap = self.cap;
        while new_cap < needed {
            new_cap = new_cap.saturating_mul(2);
        }
        let d = self.k.shape[2];
        let old_k = std::mem::replace(&mut self.k, GpuTensor::zeros(&ctx.device, vec![1, new_cap, d])?);
        let old_v = std::mem::replace(&mut self.v, GpuTensor::zeros(&ctx.device, vec![1, new_cap, d])?);
        if self.pos > 0 {
            ctx.kv_update_f32(&self.k, &old_k, 0)?;
            ctx.kv_update_f32(&self.v, &old_v, 0)?;
        }
        self.cap = new_cap;
        Ok(())
    }

    /// Insert k/v (1, Tq, D) at the current position; advances `pos`.
    pub fn update(&mut self, ctx: &MetalCtx, k: &GpuTensor, v: &GpuTensor) -> Result<(), MetalError> {
        let tq = k.shape[1];
        self.ensure(ctx, self.pos + tq)?;
        ctx.kv_update_f32(&self.k, k, self.pos)?;
        ctx.kv_update_f32(&self.v, v, self.pos)?;
        self.pos += tq;
        Ok(())
    }

    /// View of the first `pos` rows (same buffer, restricted shape).
    pub fn view(&self) -> GpuTensor {
        GpuTensor { buffer: self.k.buffer.clone(), shape: vec![1, self.pos, self.k.shape[2]] }
    }
    pub fn view_v(&self) -> GpuTensor {
        GpuTensor { buffer: self.v.buffer.clone(), shape: vec![1, self.pos, self.v.shape[2]] }
    }

    pub fn reset(&mut self) {
        self.pos = 0;
    }
}

// ─────────────────────── operator helpers ───────────────────────

/// Slice the last dim: X (rows, D) → Y (rows, d0), starting at `off`.
pub fn slice_last(ctx: &MetalCtx, x: &GpuTensor, d0: usize, off: usize) -> Result<GpuTensor, MetalError> {
    let rows = x.shape[0] * x.shape[1];
    let y = GpuTensor::zeros(&ctx.device, vec![rows, d0])?;
    ctx.slice_last_f32(&view2(x), &y, off)?;
    Ok(view3(&y, x.shape[0], x.shape[1], d0))
}

/// RMSNorm per head: X (B,T,H*HD), W (H*HD) → Y (B,T,H*HD).
fn rmsnorm_head(ctx: &MetalCtx, x: &GpuTensor, w: &GpuTensor, n_heads: usize) -> Result<GpuTensor, MetalError> {
    let y = GpuTensor::zeros(&ctx.device, x.shape.clone())?;
    ctx.rmsnorm_head_f32(x, w, &y, n_heads, 1e-6)?;
    Ok(y)
}

/// Interleaved RoPE at absolute position `pos` (cos/sin (max_seq, HD/2)).
fn rope_apply(
    ctx: &MetalCtx,
    x: &GpuTensor,
    rope: &(GpuTensor, GpuTensor),
    pos: usize,
    hd: usize,
) -> Result<GpuTensor, MetalError> {
    let y = GpuTensor::zeros(&ctx.device, x.shape.clone())?;
    ctx.rope_interleaved_f32(x, &rope.0, &rope.1, &y, hd, pos)?;
    Ok(y)
}

/// Causal attention mask: (1, Tq, Tkv), 0.0 for j ≤ pos + i, -inf elsewhere.
pub fn build_causal_mask(
    ctx: &MetalCtx,
    tq: usize,
    tkv: usize,
    pos: usize,
) -> Result<GpuTensor, MetalError> {
    let mut data = vec![0.0f32; tq * tkv];
    for i in 0..tq {
        for j in 0..tkv {
            if j as isize > (pos + i) as isize {
                data[i * tkv + j] = f32::NEG_INFINITY;
            }
        }
    }
    GpuTensor::from_cpu_f32(&ctx.device, &data, vec![1, tq, tkv])
}
