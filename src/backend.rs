//! Backend abstraction — the single seam between the models/trainers and the
//! concrete GPU runtimes.
//!
//! [`Tensor`] is an opaque host-agnostic tensor handle; [`Backend`] is the
//! operation surface (gemm, convs, activations, elementwise, norms). The
//! Metal implementation lives in [`crate::metal::MetalBackend`]; a future
//! CUDA/Vulkan backend implements the same trait and every consumer (NN
//! layers, speech models, trainers) runs unchanged.
//!
//! Consumers take `&dyn Backend` (or `&impl Backend`) — they never touch a
//! concrete runtime type.

use std::sync::Arc;

use crate::metal::{GpuTensor, MetalError};

/// An opaque tensor handle owned by the backend. Shape/numel are always
/// available; data moves through the [`Backend`] operations.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub(crate) inner: GpuTensor,
}

impl Tensor {
    pub(crate) fn from_gpu(t: GpuTensor) -> Self {
        Self { inner: t }
    }
    pub(crate) fn as_gpu(&self) -> &GpuTensor {
        &self.inner
    }

    pub fn shape(&self) -> &[usize] {
        &self.inner.shape
    }
    pub fn numel(&self) -> usize {
        self.inner.numel()
    }

    /// A zero-copy view with a new shape (same buffer). The caller must keep
    /// the total element count (row-major reinterpretation).
    pub fn view(&self, shape: Vec<usize>) -> Tensor {
        debug_assert_eq!(shape.iter().product::<usize>(), self.numel());
        Tensor { inner: GpuTensor { buffer: self.inner.buffer.clone(), shape } }
    }
}

/// Compute-surface operations. Every method dispatches its kernels through
/// the backend; implementations are responsible for allocation and
/// synchronization semantics.
pub trait Backend {
    /// Allocate a zero-initialized tensor of `shape`.
    fn zeros(&self, shape: Vec<usize>) -> Result<Tensor, MetalError>;
    /// Upload host data (row-major) into a new tensor.
    fn from_cpu_f32(&self, data: &[f32], shape: Vec<usize>) -> Result<Tensor, MetalError>;
    /// Download a tensor to host memory.
    fn to_cpu_f32(&self, t: &Tensor) -> Result<Vec<f32>, MetalError>;

    // ── gemm ──
    /// C (M,N) = A (M,K) · B (K,N).
    fn gemm(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<(), MetalError>;
    /// C (M,N) = A (M,K) · Bᵀ (N,K).
    fn gemm_nt(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<(), MetalError>;

    // ── elementwise ──
    fn mul(&self, a: &Tensor, b: &Tensor, y: &Tensor) -> Result<(), MetalError>;
    fn add(&self, a: &Tensor, b: &Tensor, y: &Tensor) -> Result<(), MetalError>;
    fn sub(&self, a: &Tensor, b: &Tensor, y: &Tensor) -> Result<(), MetalError>;
    fn div(&self, a: &Tensor, b: &Tensor, y: &Tensor) -> Result<(), MetalError>;
    fn add_scalar(&self, x: &Tensor, y: &Tensor, c: f32) -> Result<(), MetalError>;
    fn scale(&self, x: &Tensor, y: &Tensor, c: f32) -> Result<(), MetalError>;
    fn add_bias(&self, x: &Tensor, bias: &Tensor, y: &Tensor) -> Result<(), MetalError>;
    fn sqrt(&self, x: &Tensor, y: &Tensor) -> Result<(), MetalError>;
    fn log(&self, x: &Tensor, y: &Tensor) -> Result<(), MetalError>;
    fn max_scalar(&self, x: &Tensor, y: &Tensor, floor: f32) -> Result<(), MetalError>;

    // ── activations ──
    fn relu(&self, x: &Tensor, y: &Tensor) -> Result<(), MetalError>;
    fn leaky_relu_clip(&self, x: &Tensor, y: &Tensor) -> Result<(), MetalError>;
    fn sigmoid(&self, x: &Tensor, y: &Tensor) -> Result<(), MetalError>;
    fn tanh(&self, x: &Tensor, y: &Tensor) -> Result<(), MetalError>;
    fn silu(&self, x: &Tensor, y: &Tensor) -> Result<(), MetalError>;
    fn gelu_tanh(&self, x: &Tensor, y: &Tensor) -> Result<(), MetalError>;
    fn snake(&self, x: &Tensor, a: &Tensor, y: &Tensor) -> Result<(), MetalError>;

    // ── norms ──
    fn rmsnorm(&self, x: &Tensor, w: &Tensor, y: &Tensor, eps: f32) -> Result<(), MetalError>;
    fn layernorm(&self, x: &Tensor, w: &Tensor, b: &Tensor, y: &Tensor, eps: f32) -> Result<(), MetalError>;

    // ── convs ──
    /// Dense 1D conv (groups=1): X (B,T,Cin), W (Cout,Cin,K), Y (B,Tout,Cout).
    fn conv1d(
        &self,
        x: &Tensor,
        w: &Tensor,
        bias: Option<&Tensor>,
        stride: usize,
        pad_l: usize,
        dil: usize,
        y: &Tensor,
    ) -> Result<(), MetalError>;
    /// 2D conv (NCHW): X (N,Ci,H,W), W (Co,Ci,KH,KW), Y (N,Co,OH,OW).
    #[allow(clippy::too_many_arguments)]
    fn conv2d_nchw(
        &self,
        x: &Tensor,
        w: &Tensor,
        bias: Option<&Tensor>,
        pad_h: usize,
        pad_w: usize,
        y: &Tensor,
    ) -> Result<(), MetalError>;
    /// MaxPool2d (kernel == stride): X/Y (N,C,H,W).
    fn maxpool2d(&self, x: &Tensor, y: &Tensor, kh: usize, kw: usize) -> Result<(), MetalError>;
    /// Transposed 1D conv: W (Cin, Cout, K), X (B,T,Cin), Y (B,Tout,Cout).
    fn conv_transpose1d(
        &self,
        x: &Tensor,
        w: &Tensor,
        bias: Option<&Tensor>,
        stride: usize,
        y: &Tensor,
    ) -> Result<(), MetalError>;

    // ── slices ──
    /// Last-dim slice: X (rows, D) → Y (rows, d0), starting at `off`.
    fn slice_last(&self, x: &Tensor, y: &Tensor, off: usize) -> Result<(), MetalError>;
    /// Time slice: drops the LAST `u` frames: X (B,T,C) → Y (B,T-u,C).
    fn slice_time(&self, x: &Tensor, y: &Tensor, u: usize) -> Result<(), MetalError>;
}
