//! Backend abstraction — the single seam between the models/trainers and the
//! concrete GPU runtimes.
//!
//! [`Tensor`] is an opaque host-agnostic tensor handle; [`Backend`] is the
//! operation surface (gemm, convs, activations, elementwise, norms). The
//! Metal implementation lives in [`crate::metal::MetalBackend`] (`metal`
//! feature); future CUDA/Vulkan/WebGPU backends implement the same trait and
//! every consumer (NN layers, speech models, trainers) runs unchanged.
//!
//! Consumers take `&dyn Backend` — they never touch a concrete runtime type.
//!
//! # dtype & formats (Phase 1.2)
//!
//! The contract is f32-shaped today; [`crate::Dtype`] / quantized weight
//! formats / KV formats land with the dtype design. Ops must compute their
//! reductions/norms/softmax statistics in f32 regardless of the tensor dtype
//! (the f32-compute list in the contract doc).

#[cfg(feature = "metal")]
use crate::metal::GpuTensor;

#[cfg(any(feature = "cpu", feature = "cuda"))]
use std::sync::Arc;

use crate::dtype::Dtype;
use crate::error::BackendError;

/// Backend-owned tensor storage. Each backend attaches its native buffer
/// under its feature; the shape/dtype metadata lives on [`Tensor`] itself.
///
/// The CPU variant uses `Arc<RwLock<Vec<f32>>>` — interior mutability, since
/// the [`Backend`] contract passes every tensor as `&Tensor` (outputs are
/// mutated in place; the Metal engine does the same through its buffers).
#[derive(Debug, Clone)]
pub(crate) enum TensorInner {
    #[cfg(feature = "metal")]
    Metal(GpuTensor),
    #[cfg(feature = "cpu")]
    Cpu(Arc<parking_lot::RwLock<Vec<f32>>>),
    #[cfg(feature = "cuda")]
    Cuda(Arc<crate::cuda::CudaBuffer>),
    /// fp16 weight storage (GEMM operands only — activations stay fp32).
    #[cfg(feature = "cuda")]
    CudaF16(cudarc::driver::CudaSlice<half::f16>),
    /// bf16 weight storage (GEMM operands only).
    #[cfg(feature = "cuda")]
    CudaBF16(cudarc::driver::CudaSlice<half::bf16>),
    /// int8 weights + per-channel fp32 scale (GEMM operands only).
    #[cfg(feature = "cuda")]
    CudaI8 {
        data: cudarc::driver::CudaSlice<i8>,
        scale: cudarc::driver::CudaSlice<f32>,
    },
    /// Backend-less builds: no storage exists (contract skeleton only). Keeps
    /// the enum non-empty so `Tensor` methods compile without warnings.
    #[cfg(not(any(feature = "metal", feature = "cpu", feature = "cuda")))]
    #[allow(dead_code)] // never constructed — exists to keep the enum inhabited
    Unavailable,
}

/// An opaque tensor handle owned by the backend. Shape/numel/dtype are always
/// available; data moves through the [`Backend`] operations.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub(crate) inner: TensorInner,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: Dtype,
}

impl Tensor {
    #[cfg(feature = "metal")]
    pub(crate) fn from_gpu(t: GpuTensor) -> Self {
        let shape = t.shape.clone();
        let dtype = match t.dtype {
            crate::metal::GpuDtype::F32 => Dtype::F32,
            crate::metal::GpuDtype::F16 => Dtype::F16,
            crate::metal::GpuDtype::BF16 => Dtype::BF16,
        };
        Self { inner: TensorInner::Metal(t), shape, dtype }
    }

    #[cfg(feature = "metal")]
    pub(crate) fn as_gpu(&self) -> &GpuTensor {
        // let-else keeps the match exhaustive when several storage features
        // are enabled together (metal+cuda dev builds) — listing the arms
        // here broke the combination.
        let TensorInner::Metal(g) = &self.inner else {
            panic!("as_gpu() on a non-Metal tensor");
        };
        g
    }

    /// CPU-reference storage (Phase 2 backend). The CPU backend owns its data
    /// as plain f32 — this is the test oracle, not a production path.
    #[cfg(feature = "cpu")]
    pub(crate) fn from_cpu(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { inner: TensorInner::Cpu(Arc::new(parking_lot::RwLock::new(data))), shape, dtype: Dtype::F32 }
    }

    /// CPU-reference read guard (test oracle).
    #[cfg(feature = "cpu")]
    pub(crate) fn cpu_read(&self) -> parking_lot::RwLockReadGuard<'_, Vec<f32>> {
        match &self.inner {
            TensorInner::Cpu(v) => v.read(),
            #[cfg(feature = "metal")]
            TensorInner::Metal(_) => panic!("cpu_read() on a non-CPU tensor"),
            #[cfg(feature = "cuda")]
            TensorInner::Cuda(_) | TensorInner::CudaF16(_) | TensorInner::CudaBF16(_) | TensorInner::CudaI8 { .. } => panic!("cpu_read() on a non-CPU tensor"),
        }
    }

    /// CPU-reference write guard (test oracle).
    #[cfg(feature = "cpu")]
    pub(crate) fn cpu_write(&self) -> parking_lot::RwLockWriteGuard<'_, Vec<f32>> {
        match &self.inner {
            TensorInner::Cpu(v) => v.write(),
            #[cfg(feature = "metal")]
            TensorInner::Metal(_) => panic!("cpu_write() on a non-CPU tensor"),
            #[cfg(feature = "cuda")]
            TensorInner::Cuda(_) | TensorInner::CudaF16(_) | TensorInner::CudaBF16(_) | TensorInner::CudaI8 { .. } => panic!("cpu_write() on a non-CPU tensor"),
        }
    }

    /// Whether both tensors share the same CPU storage (in-place aliasing).
    /// Ops like `silu(&a, &a)` or `add_bias(&y, bias, &y)` alias input and
    /// output — the CPU reference must not take a read+write lock on the
    /// same `Arc<RwLock>` (that would self-deadlock).
    #[cfg(feature = "cpu")]
    pub(crate) fn cpu_same_storage(&self, other: &Tensor) -> bool {
        match (&self.inner, &other.inner) {
            (TensorInner::Cpu(a), TensorInner::Cpu(b)) => Arc::ptr_eq(a, b),
            #[cfg(any(feature = "metal", feature = "cuda"))]
            _ => false,
        }
    }

    /// CUDA storage (Phase 3 backend). `CudaBuffer` is an Arc'd pooled buffer
    /// that returns itself to the backend pool on last-drop.
    #[cfg(feature = "cuda")]
    pub(crate) fn from_cuda(b: Arc<crate::cuda::CudaBuffer>, shape: Vec<usize>) -> Self {
        Self { inner: TensorInner::Cuda(b), shape, dtype: Dtype::F32 }
    }

    /// fp16 tensor from a CUDA buffer (weights for fp16 GEMMs).
    #[cfg(feature = "cuda")]
    pub(crate) fn from_cuda_f16(b: cudarc::driver::CudaSlice<half::f16>, shape: Vec<usize>) -> Self {
        Self { inner: TensorInner::CudaF16(b), shape, dtype: Dtype::F16 }
    }

    /// bf16 tensor from a CUDA buffer (weights for bf16 GEMMs).
    #[cfg(feature = "cuda")]
    pub(crate) fn from_cuda_bf16(b: cudarc::driver::CudaSlice<half::bf16>, shape: Vec<usize>) -> Self {
        Self { inner: TensorInner::CudaBF16(b), shape, dtype: Dtype::BF16 }
    }

    /// int8 tensor with a per-channel fp32 scale (weights for int8 GEMMs).
    #[cfg(feature = "cuda")]
    pub(crate) fn from_cuda_i8(
        data: cudarc::driver::CudaSlice<i8>,
        scale: cudarc::driver::CudaSlice<f32>,
        shape: Vec<usize>,
    ) -> Self {
        Self { inner: TensorInner::CudaI8 { data, scale }, shape, dtype: Dtype::I8 }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn as_cuda(&self) -> &cudarc::driver::CudaSlice<f32> {
        match &self.inner {
            TensorInner::Cuda(b) => b.as_slice(),
            TensorInner::CudaF16(_) | TensorInner::CudaBF16(_) | TensorInner::CudaI8 { .. } => panic!("as_cuda() on a half/int8 tensor"),
            #[cfg(feature = "metal")]
            TensorInner::Metal(_) => panic!("as_cuda() on a non-CUDA tensor"),
            #[cfg(feature = "cpu")]
            TensorInner::Cpu(_) => panic!("as_cuda() on a non-CUDA tensor"),
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn as_cuda_f16(&self) -> &cudarc::driver::CudaSlice<half::f16> {
        match &self.inner {
            TensorInner::CudaF16(b) => b,
            _ => panic!("as_cuda_f16() on a non-CudaF16 tensor"),
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn as_cuda_bf16(&self) -> &cudarc::driver::CudaSlice<half::bf16> {
        match &self.inner {
            TensorInner::CudaBF16(b) => b,
            _ => panic!("as_cuda_bf16() on a non-CudaBF16 tensor"),
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn as_cuda_i8(&self) -> (&cudarc::driver::CudaSlice<i8>, &cudarc::driver::CudaSlice<f32>) {
        match &self.inner {
            TensorInner::CudaI8 { data, scale } => (data, scale),
            _ => panic!("as_cuda_i8() on a non-CudaI8 tensor"),
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn as_cuda_mut(&self) -> &mut cudarc::driver::CudaSlice<f32> {
        // &mut through &self: the caller is the only holder of the backend
        // during an op (no concurrent mutation of this tensor's storage).
        match &self.inner {
            TensorInner::Cuda(b) => unsafe {
                &mut *Arc::as_ptr(b).cast_mut().as_mut().unwrap().as_slice_mut()
            },
            TensorInner::CudaF16(_) | TensorInner::CudaBF16(_) | TensorInner::CudaI8 { .. } => panic!("as_cuda_mut() on a half/int8 tensor"),
            #[cfg(feature = "metal")]
            TensorInner::Metal(_) => panic!("as_cuda_mut() on a non-CUDA tensor"),
            #[cfg(feature = "cpu")]
            TensorInner::Cpu(_) => panic!("as_cuda_mut() on a non-CUDA tensor"),
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// A zero-copy view with a new shape (same buffer). The caller must keep
    /// the total element count (row-major reinterpretation).
    ///
    /// Metal note: the inner `GpuTensor` carries its own shape which the
    /// Metal wrappers read — the view REBUILDS it so both stay in sync
    /// (a stale inner shape made every view-using gemm fail shape checks).
    pub fn view(&self, shape: Vec<usize>) -> Tensor {
        debug_assert_eq!(shape.iter().product::<usize>(), self.numel());
        self.view_inner(shape)
    }

    /// Restricted zero-copy view: a window over a LARGER buffer (e.g. the
    /// KV cache exposing only the first `pos` rows). No numel check — the
    /// caller guarantees `product(shape) ≤ numel()`.
    pub fn view_restricted(&self, shape: Vec<usize>) -> Tensor {
        debug_assert!(shape.iter().product::<usize>() <= self.numel());
        self.view_inner(shape)
    }

    fn view_inner(&self, shape: Vec<usize>) -> Tensor {
        let dtype = self.dtype;
        #[cfg(feature = "metal")]
        {
            if let TensorInner::Metal(g) = &self.inner {
                return Tensor {
                    inner: TensorInner::Metal(GpuTensor {
                        buffer: g.buffer.clone(),
                        shape: shape.clone(),
                        dtype: g.dtype,
                    }),
                    shape,
                    dtype,
                };
            }
        }
        #[allow(unreachable_code)]
        Tensor { inner: self.inner.clone(), shape, dtype }
    }
}

/// Compute-surface operations. Every method dispatches its kernels through
/// the backend; implementations are responsible for allocation and
/// synchronization semantics. Errors cross the boundary as [`BackendError`].
/// Create the platform-default backend (env `BACKEND` can override:
/// `cpu`/`cuda`/`metal` — the feature must be compiled in; the GPU-only rule
/// keeps CPU behind an explicit opt-in).
pub fn default_backend() -> Result<std::sync::Arc<dyn Backend>, BackendError> {
    let requested = std::env::var("BACKEND").ok();
    #[cfg(feature = "metal")]
    if requested.as_deref().map(|r| r == "metal").unwrap_or(true) {
        return Ok(std::sync::Arc::new(crate::metal::MetalBackend::new()?));
    }
    #[cfg(feature = "cuda")]
    if requested.as_deref().map(|r| r == "cuda").unwrap_or(true) {
        return Ok(std::sync::Arc::new(crate::cuda::CudaBackend::new(0)?));
    }
    #[cfg(feature = "cpu")]
    if requested.as_deref().map(|r| r == "cpu").unwrap_or(false) {
        return Ok(std::sync::Arc::new(crate::cpu::CpuBackend));
    }
    #[allow(unreachable_code)]
    Err(BackendError::BackendUnavailable(
        "no backend feature enabled; set BACKEND=cpu|cuda|metal and enable the feature",
    ))
}

pub trait Backend: Send + Sync {
    /// Allocate a zero-initialized tensor of `shape`.
    fn zeros(&self, shape: Vec<usize>) -> Result<Tensor, BackendError>;
    /// Upload host data (row-major) into a new tensor.
    fn from_cpu_f32(&self, data: &[f32], shape: Vec<usize>) -> Result<Tensor, BackendError>;

    /// Upload fp16 data (GEMM weight storage; activations stay fp32).
    /// Default: convert to f32 and behave like `from_cpu_f32` (CPU reference,
    /// backend-less builds); CUDA stores true fp16.
    fn from_cpu_f16(&self, data: &[half::f16], shape: Vec<usize>) -> Result<Tensor, BackendError> {
        let f32s: Vec<f32> = data.iter().map(|h| f32::from(*h)).collect();
        self.from_cpu_f32(&f32s, shape)
    }

    /// Upload bf16 data (GEMM weight storage). Default: convert to f32.
    fn from_cpu_bf16(&self, data: &[half::bf16], shape: Vec<usize>) -> Result<Tensor, BackendError> {
        let f32s: Vec<f32> = data.iter().map(|h| f32::from(*h)).collect();
        self.from_cpu_f32(&f32s, shape)
    }

    /// Upload int8 weights with a per-channel fp32 scale (GEMM storage).
    /// Default: dequantize to f32.
    fn from_cpu_i8(&self, data: &[i8], scale: &[f32], shape: Vec<usize>) -> Result<Tensor, BackendError> {
        let rows = shape[0];
        let mut f32s = Vec::with_capacity(data.len());
        for (i, &v) in data.iter().enumerate() {
            f32s.push(v as f32 * scale[i / (shape[1..].iter().product::<usize>())]);
        }
        let _ = rows;
        self.from_cpu_f32(&f32s, shape)
    }
    /// Download a tensor to host memory.
    fn to_cpu_f32(&self, t: &Tensor) -> Result<Vec<f32>, BackendError>;

    // ── batching ──
    /// Start accumulating GPU kernels into ONE command buffer: many small
    /// ops (wake word, VAD) pay per-op commit+wait sync otherwise, which
    /// makes them slower than the audio arrives (observed: ~97 ms per 80 ms
    /// wake window — the engine could never keep up with the mic). Nested
    /// calls are allowed; [`Self::end_batch`] must be paired per call.
    /// Defaults are no-ops — only backends with per-op GPU sync (Metal)
    /// implement this.
    fn begin_batch(&self) -> Result<(), BackendError> {
        Ok(())
    }
    /// Flush the pending batch (one commit+wait) and leave batch mode.
    fn end_batch(&self) -> Result<(), BackendError> {
        Ok(())
    }
    /// Diagnostics: how many batches were opened (tests assert that
    /// latency-sensitive paths — the wake word — actually batch).
    fn batch_enter_count(&self) -> u64 {
        0
    }

    // ── gemm ──
    /// C (M,N) = A (M,K) · B (K,N).
    fn gemm(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<(), BackendError>;
    /// C (M,N) = A (M,K) · Bᵀ (N,K).
    fn gemm_nt(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<(), BackendError>;

    // ── elementwise ──
    fn mul(&self, a: &Tensor, b: &Tensor, y: &Tensor) -> Result<(), BackendError>;
    fn add(&self, a: &Tensor, b: &Tensor, y: &Tensor) -> Result<(), BackendError>;
    fn sub(&self, a: &Tensor, b: &Tensor, y: &Tensor) -> Result<(), BackendError>;
    fn div(&self, a: &Tensor, b: &Tensor, y: &Tensor) -> Result<(), BackendError>;
    fn add_scalar(&self, x: &Tensor, y: &Tensor, c: f32) -> Result<(), BackendError>;
    fn scale(&self, x: &Tensor, y: &Tensor, c: f32) -> Result<(), BackendError>;
    fn add_bias(&self, x: &Tensor, bias: &Tensor, y: &Tensor) -> Result<(), BackendError>;
    fn sqrt(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError>;
    fn log(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError>;
    fn max_scalar(&self, x: &Tensor, y: &Tensor, floor: f32) -> Result<(), BackendError>;

    // ── activations ──
    fn relu(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError>;
    fn leaky_relu_clip(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError>;
    fn sigmoid(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError>;
    fn tanh(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError>;
    fn silu(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError>;
    fn gelu_tanh(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError>;
    fn snake(&self, x: &Tensor, a: &Tensor, y: &Tensor) -> Result<(), BackendError>;

    // ── norms ──
    fn rmsnorm(&self, x: &Tensor, w: &Tensor, y: &Tensor, eps: f32) -> Result<(), BackendError>;
    fn layernorm(&self, x: &Tensor, w: &Tensor, b: &Tensor, y: &Tensor, eps: f32) -> Result<(), BackendError>;

    // ── convs ──
    /// Dense 1D conv (groups=1): X (B,T,Cin), W (Cout,Cin,K), Y (B,Tout,Cout).
    /// `pad_l`/`pad_r` — left/right padding (causal = pad_r 0);
    /// `tout = (t + pad_l + pad_r − dil·(k−1) − 1)/stride + 1`.
    #[allow(clippy::too_many_arguments)]
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
    ) -> Result<(), BackendError>;
    /// Grouped 1D conv (depthwise when groups == Cin): W (Cout, Cin_g, K);
    /// out channel `co` reads input channels [g·Cin_g, (g+1)·Cin_g),
    /// g = co / Cin_g. Same padding semantics as [`Backend::conv1d`].
    #[allow(clippy::too_many_arguments)]
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
    ) -> Result<(), BackendError>;
    /// 2D conv (NCHW): X (N,Ci,H,W), W (Co,Ci,KH,KW), Y (N,Co,OH,OW).
    /// `stride_t`/`stride_f` — strides over H/W; `time_major_out` produces
    /// Y (N, OH, Co·OW) instead of NCHW (the t-one subsampling layout).
    #[allow(clippy::too_many_arguments)]
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
    ) -> Result<(), BackendError>;
    /// MaxPool2d (kernel == stride): X/Y (N,C,H,W).
    fn maxpool2d(&self, x: &Tensor, y: &Tensor, kh: usize, kw: usize) -> Result<(), BackendError>;
    /// Transposed 1D conv: W (Cin, Cout, K), X (B,T,Cin), Y (B,Tout,Cout).
    fn conv_transpose1d(
        &self,
        x: &Tensor,
        w: &Tensor,
        bias: Option<&Tensor>,
        stride: usize,
        y: &Tensor,
    ) -> Result<(), BackendError>;

    /// im2col for a dense 1D conv (groups=1): X (B,T,Cin) → Xc
    /// (B,Tout,Cin·K): Xc[t, ci·K+kk] = X[t·stride + kk·dil − pad_l, ci]
    /// (zero outside). Turns convs into GEMMs (cuBLAS fp16/bf16), which
    /// also lets conv WEIGHTS take the model quantization.
    #[allow(clippy::too_many_arguments)]
    fn im2col1d(
        &self,
        x: &Tensor,
        xc: &Tensor,
        k: usize,
        stride: usize,
        pad_l: usize,
        dil: usize,
        tout: usize,
    ) -> Result<(), BackendError>;
    /// im2col for a transposed 1D conv: X (B,Tin,Cin) → Xc (B,Tout,Cin·K)
    /// with Xc[t, ci·K+kk] = X[(t−kk)/stride, ci] when divisible, else 0.
    fn im2col_t1d(
        &self,
        x: &Tensor,
        xc: &Tensor,
        k: usize,
        stride: usize,
        tout: usize,
    ) -> Result<(), BackendError>;

    // ── slices ──
    /// Last-dim slice: X (rows, D) → Y (rows, d0), starting at `off`.
    fn slice_last(&self, x: &Tensor, y: &Tensor, off: usize) -> Result<(), BackendError>;
    /// Time slice: drops the LAST `u` frames: X (B,T,C) → Y (B,T-u,C).
    fn slice_time(&self, x: &Tensor, y: &Tensor, u: usize) -> Result<(), BackendError>;

    // ── softmax (f32-compute) ──
    /// Row-wise softmax over the last dim: X (rows, D) → Y (rows, D).
    /// Internal max/exp/sum statistics are ALWAYS computed in f32 regardless
    /// of the tensor dtype (f32-compute contract).
    fn softmax_last_dim(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError>;

    // ── embedding ──
    /// Gather rows of `table` (V, D) by `idx` → Y (len(idx), D). f32-compute.
    fn embedding(&self, idx: &[u32], table: &Tensor, y: &Tensor) -> Result<(), BackendError>;

    // ── RoPE ──
    /// Halved RoPE: X (T, head_dim) rotated by (cos, sin) buffers, applying to
    /// the first `rope_dim` columns. Angles computed in f32.
    fn rope(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, y: &Tensor, rope_dim: usize) -> Result<(), BackendError>;
    /// Interleaved RoPE (fishaudio convention): pairs (x0, x1) rotate together.
    /// `start` is the absolute position offset — cos/sin are indexed at
    /// (t + start) so cached tokens keep their original rotation.
    fn rope_interleaved(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, y: &Tensor, rope_dim: usize, start: usize) -> Result<(), BackendError>;
    /// Per-head RoPE: X (B, T, h·hd) with `h` heads of `hd` dims; rotates the
    /// first `rope_dim` columns of every head.
    fn rope_head(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, y: &Tensor, h: usize, rope_dim: usize) -> Result<(), BackendError>;

    // ── head norms (f32-compute) ──
    /// RMSNorm applied per head: X (rows, h·hd), w per head-dim (hd), h heads.
    fn rmsnorm_head(&self, x: &Tensor, w: &Tensor, y: &Tensor, h: usize, eps: f32) -> Result<(), BackendError>;
    /// LayerNorm applied per head: X (rows, h·hd), w/b per head-dim (hd).
    fn layernorm_head(&self, x: &Tensor, w: &Tensor, b: &Tensor, y: &Tensor, h: usize, eps: f32) -> Result<(), BackendError>;

    // ── row/vec products ──
    /// Row-wise scale: X (rows, D) · w (rows,) → Y (rows, D) (each row scaled
    /// by its scalar). Used for per-sample weights.
    fn mul_rows(&self, x: &Tensor, w: &Tensor, y: &Tensor) -> Result<(), BackendError>;
    /// Column-wise scale: X (rows, D) · w (D,) → Y (rows, D).
    fn mul_vec(&self, x: &Tensor, w: &Tensor, y: &Tensor) -> Result<(), BackendError>;

    // ── GLU ──
    /// Gated linear unit: X (rows, 2d) → Y (rows, d), y = a·sigmoid(b).
    fn glu(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError>;

    // ── repeat (broadcast in time) ──
    /// Time-axis ×2 upsampling: X (B, T, D) → Y (B, 2T, D), each frame
    /// duplicated: y[b, 2t + {0,1}, d] = x[b, t, d].
    fn repeat(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError>;

    // ── fused add forms ──
    /// Y = A + scale·B (elementwise, same shapes).
    fn add_scaled(&self, a: &Tensor, b: &Tensor, y: &Tensor, scale: f32) -> Result<(), BackendError>;
    /// 2D bias add — ELEMENTWISE (x + b, same shapes; legacy Metal name).
    /// Use [`Backend::add`] for the same semantics; kept for parity with the
    /// Metal engine's `add_bias2d_f32` kernel.
    fn add_bias2d(&self, x: &Tensor, b: &Tensor, y: &Tensor) -> Result<(), BackendError>;

    // ── fused attention kernels ──
    /// Autoregressive GQA attention (TTS DualAR), fused: softmax(QKᵀ/√hd)·V
    /// with optional causal mask. Q (B,Tq,D), K/V (B,Tkv,D), y (B,Tq,D);
    /// n_heads query heads, kv_heads key/value heads (GQA repeat), hd head dim.
    /// Softmax statistics in f32 (f32-compute).
    #[allow(clippy::too_many_arguments)]
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
    ) -> Result<(), BackendError>;
    /// Fused multi-head self-attention (STT Conformer): Q/K/V (B,T,D) with
    /// D = H·HD (H = sc.shape[1]). `sc` (B,H,T,T) is the scores workspace —
    /// the leader (mode 0) writes scores, non-leaders (mode 1) read them
    /// (pre-masked); mode 2 = uniform. `mask` (T,T) bias (0 / -10000) or
    /// null. Softmax statistics in f32 (f32-compute).
    #[allow(clippy::too_many_arguments)]
    fn mhsa(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        sc: &Tensor,
        mask: Option<&Tensor>,
        mode: u32,
        y: &Tensor,
    ) -> Result<(), BackendError>;
    /// Uniform attention (STT): Y = mean over the time axis of V (B,T,C) → (B,C).
    fn uniform_attn(&self, v: &Tensor, y: &Tensor) -> Result<(), BackendError>;

    // ── KV cache ──
    /// Write SRC (B, Tq, D) into DST (B, cap, D) starting at position `pos`.
    /// Quantized KV formats quantize on write here; fused attention
    /// dequantizes inline on read (per-component KV config).
    fn kv_update(&self, dst: &Tensor, src: &Tensor, pos: usize) -> Result<(), BackendError>;

    // ── codebook search (DAC) ──
    /// For every row of X (B·T, D) find the nearest row of `table` (V, D)
    /// (L2) and write its index into `idx` (host buffer, len B·T).
    fn codebook_search(&self, x: &Tensor, table: &Tensor, idx: &mut [u32]) -> Result<(), BackendError>;
}
