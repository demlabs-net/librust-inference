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
    /// Q6_K packed weights (GEMM operands only): canonical llama.cpp layout,
    /// 210 bytes per 256-weight super-block (ql[128] + qh[64] + scales[16] +
    /// f16 d) — dequantized inline in the fused GEMM kernel.
    #[cfg(feature = "cuda")]
    CudaQ6K(cudarc::driver::CudaSlice<u8>),
    /// Q5_K packed weights (GEMM operands only): canonical llama.cpp layout,
    /// 176 bytes per 256-weight super-block (d + dmin f16, scales[12], qh[32],
    /// qs[128]) — dequantized inline in the fused GEMM kernel.
    #[cfg(feature = "cuda")]
    CudaQ5K(cudarc::driver::CudaSlice<u8>),
    /// Q3_K packed weights (GEMM operands only): canonical llama.cpp layout,
    /// 110 bytes per 256-weight super-block (hmask[32] + qs[64] + scales[12] +
    /// f16 d) — dequantized inline in the fused GEMM kernel.
    #[cfg(feature = "cuda")]
    CudaQ3K(cudarc::driver::CudaSlice<u8>),
    /// Canonical Q4_K packed weights (GEMM operands only): llama.cpp
    /// block_q4_K, 144 bytes per 256-weight super-block (d + dmin f16,
    /// scales[12] get_scale_min_k4, chunk-major qs) — GGUF Q4_K tensors
    /// (N iter-10: previously dequantized to f32 on the host, ~7x DRAM).
    #[cfg(feature = "cuda")]
    CudaQ4KC(cudarc::driver::CudaSlice<u8>),
    /// Q4_0 packed weights (GEMM operands only): canonical llama.cpp layout,
    /// 18 bytes per 32-weight block (f16 d + 16 nibble bytes) — dequantized
    /// inline in the fused GEMM kernel.
    #[cfg(feature = "cuda")]
    CudaQ4K(cudarc::driver::CudaSlice<u8>),
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
            // Квантованные веса — сырое байтовое хранилище; используются
            // только через metal-специфичные quantized-API (не Backend-трейт).
            crate::metal::GpuDtype::Q4_0
            | crate::metal::GpuDtype::Q8_0
            | crate::metal::GpuDtype::Q4_K
            | crate::metal::GpuDtype::Q6_K => Dtype::I8,
        };
        Self {
            inner: TensorInner::Metal(t),
            shape,
            dtype,
        }
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
        Self {
            inner: TensorInner::Cpu(Arc::new(parking_lot::RwLock::new(data))),
            shape,
            dtype: Dtype::F32,
        }
    }

    /// CPU-reference read guard (test oracle).
    #[cfg(feature = "cpu")]
    pub(crate) fn cpu_read(&self) -> parking_lot::RwLockReadGuard<'_, Vec<f32>> {
        match &self.inner {
            TensorInner::Cpu(v) => v.read(),
            #[cfg(feature = "metal")]
            TensorInner::Metal(_) => panic!("cpu_read() on a non-CPU tensor"),
            #[cfg(feature = "cuda")]
            TensorInner::Cuda(_)
            | TensorInner::CudaF16(_)
            | TensorInner::CudaBF16(_)
            | TensorInner::CudaI8 { .. }
            | TensorInner::CudaQ6K(_)
            | TensorInner::CudaQ5K(_)
            | TensorInner::CudaQ3K(_)
            | TensorInner::CudaQ4KC(_)
            | TensorInner::CudaQ4K(_) => panic!("cpu_read() on a non-CPU tensor"),
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
            TensorInner::Cuda(_)
            | TensorInner::CudaF16(_)
            | TensorInner::CudaBF16(_)
            | TensorInner::CudaI8 { .. }
            | TensorInner::CudaQ6K(_)
            | TensorInner::CudaQ5K(_)
            | TensorInner::CudaQ3K(_)
            | TensorInner::CudaQ4KC(_)
            | TensorInner::CudaQ4K(_) => panic!("cpu_write() on a non-CPU tensor"),
        }
    }

    /// Доступ к Metal-тензору (для кернел-диспетчеров того же крейта).
    #[cfg(feature = "metal")]
    pub(crate) fn inner_metal(&self) -> &crate::metal::GpuTensor {
        match &self.inner {
            TensorInner::Metal(g) => g,
            _ => panic!("inner_metal() on a non-Metal tensor"),
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
        Self {
            inner: TensorInner::Cuda(b),
            shape,
            dtype: Dtype::F32,
        }
    }

    /// fp16 tensor from a CUDA buffer (weights for fp16 GEMMs).
    #[cfg(feature = "cuda")]
    pub(crate) fn from_cuda_f16(
        b: cudarc::driver::CudaSlice<half::f16>,
        shape: Vec<usize>,
    ) -> Self {
        Self {
            inner: TensorInner::CudaF16(b),
            shape,
            dtype: Dtype::F16,
        }
    }

    /// bf16 tensor from a CUDA buffer (weights for bf16 GEMMs).
    #[cfg(feature = "cuda")]
    pub(crate) fn from_cuda_bf16(
        b: cudarc::driver::CudaSlice<half::bf16>,
        shape: Vec<usize>,
    ) -> Self {
        Self {
            inner: TensorInner::CudaBF16(b),
            shape,
            dtype: Dtype::BF16,
        }
    }

    /// int8 tensor with a per-channel fp32 scale (weights for int8 GEMMs).
    #[cfg(feature = "cuda")]
    pub(crate) fn from_cuda_i8(
        data: cudarc::driver::CudaSlice<i8>,
        scale: cudarc::driver::CudaSlice<f32>,
        shape: Vec<usize>,
    ) -> Self {
        Self {
            inner: TensorInner::CudaI8 { data, scale },
            shape,
            dtype: Dtype::I8,
        }
    }

    /// Q6_K packed-weight tensor (canonical 210 B / 256-weight blocks).
    #[cfg(feature = "cuda")]
    pub(crate) fn from_cuda_q6k(data: cudarc::driver::CudaSlice<u8>, shape: Vec<usize>) -> Self {
        Self {
            inner: TensorInner::CudaQ6K(data),
            shape,
            dtype: Dtype::I8,
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn as_cuda_q6k(&self) -> &cudarc::driver::CudaSlice<u8> {
        match &self.inner {
            TensorInner::CudaQ6K(b) => b,
            _ => panic!("as_cuda_q6k() on a non-CudaQ6K tensor"),
        }
    }

    /// Q4_0 packed-weight tensor (canonical 18 B / 32-weight blocks).
    #[cfg(feature = "cuda")]
    pub(crate) fn from_cuda_q4k(data: cudarc::driver::CudaSlice<u8>, shape: Vec<usize>) -> Self {
        Self {
            inner: TensorInner::CudaQ4K(data),
            shape,
            dtype: Dtype::I8,
        }
    }

    /// Canonical Q4_K packed-weight tensor (144 B / 256-weight blocks).
    #[cfg(feature = "cuda")]
    pub(crate) fn from_cuda_q4kc(data: cudarc::driver::CudaSlice<u8>, shape: Vec<usize>) -> Self {
        Self {
            inner: TensorInner::CudaQ4KC(data),
            shape,
            dtype: Dtype::I8,
        }
    }

    /// Q5_K packed-weight tensor (canonical 176 B / 256-weight super-blocks).
    #[cfg(feature = "cuda")]
    pub(crate) fn from_cuda_q5k(data: cudarc::driver::CudaSlice<u8>, shape: Vec<usize>) -> Self {
        Self {
            inner: TensorInner::CudaQ5K(data),
            shape,
            dtype: Dtype::I8,
        }
    }

    /// Q3_K packed-weight tensor (canonical 110 B / 256-weight super-blocks).
    #[cfg(feature = "cuda")]
    pub(crate) fn from_cuda_q3k(data: cudarc::driver::CudaSlice<u8>, shape: Vec<usize>) -> Self {
        Self {
            inner: TensorInner::CudaQ3K(data),
            shape,
            dtype: Dtype::I8,
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn as_cuda_q4k(&self) -> &cudarc::driver::CudaSlice<u8> {
        match &self.inner {
            TensorInner::CudaQ4K(b) => b,
            _ => panic!("as_cuda_q4k() on a non-CudaQ4K tensor"),
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn as_cuda_q5k(&self) -> &cudarc::driver::CudaSlice<u8> {
        match &self.inner {
            TensorInner::CudaQ5K(b) => b,
            _ => panic!("as_cuda_q5k() on a non-CudaQ5K tensor"),
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn as_cuda_q3k(&self) -> &cudarc::driver::CudaSlice<u8> {
        match &self.inner {
            TensorInner::CudaQ3K(b) => b,
            _ => panic!("as_cuda_q3k() on a non-CudaQ3K tensor"),
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn as_cuda(&self) -> &cudarc::driver::CudaSlice<f32> {
        match &self.inner {
            TensorInner::Cuda(b) => b.as_slice(),
            TensorInner::CudaF16(_)
            | TensorInner::CudaBF16(_)
            | TensorInner::CudaI8 { .. }
            | TensorInner::CudaQ6K(_)
            | TensorInner::CudaQ5K(_)
            | TensorInner::CudaQ3K(_)
            | TensorInner::CudaQ4KC(_)
            | TensorInner::CudaQ4K(_) => panic!("as_cuda() on a half/int8/K-quant tensor"),
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
    pub(crate) fn as_cuda_mut(&self) -> &mut cudarc::driver::CudaSlice<f32> {
        // &mut through &self: the caller is the only holder of the backend
        // during an op (no concurrent mutation of this tensor's storage).
        match &self.inner {
            TensorInner::Cuda(b) => unsafe {
                &mut *Arc::as_ptr(b).cast_mut().as_mut().unwrap().as_slice_mut()
            },
            TensorInner::CudaF16(_)
            | TensorInner::CudaBF16(_)
            | TensorInner::CudaI8 { .. }
            | TensorInner::CudaQ6K(_)
            | TensorInner::CudaQ5K(_)
            | TensorInner::CudaQ3K(_)
            | TensorInner::CudaQ4KC(_)
            | TensorInner::CudaQ4K(_) => panic!("as_cuda_mut() on a half/int8/K-quant tensor"),
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
        // R30: a release build must not rely on debug_assert — an oversized
        // view silently corrupts downstream indexing. Panic with a message
        // (programming-error backstop, not a data path).
        let n = shape.iter().product::<usize>();
        assert!(n == self.numel(), "view: {n} != numel {}", self.numel());
        self.view_inner(shape)
    }

    /// Restricted zero-copy view: a window over a LARGER buffer (e.g. the
    /// KV cache exposing only the first `pos` rows). No numel check — the
    /// caller guarantees `product(shape) ≤ numel()`.
    pub fn view_restricted(&self, shape: Vec<usize>) -> Tensor {
        // R30: same as view — the debug_assert alone was a release-mode
        // silent-corruption hole.
        let n = shape.iter().product::<usize>();
        assert!(n <= self.numel(), "view_restricted: {n} > numel {}", self.numel());
        self.view_inner(shape)
    }

    /// View starting `elem_offset` elements into the buffer. Metal — zero-copy
    /// view (offset через set_buffer); CPU/CUDA — contiguous КОПИЯ со сдвигом
    /// (та же семантика результата, без разделяемого хранилища). Используется
    /// KV compaction (Metal) и q/k/v-разрезкой fused-qkv (без slice-кернелов).
    ///
    /// D7: the offset is a real semantic (the old CUDA/CPU path silently
    /// IGNORED it and returned the buffer from 0). Bounds are checked at
    /// runtime, not just by debug_assert — an out-of-range offset is a
    /// programming error and must fail loudly, never return wrong data.
    pub fn view_offset(&self, shape: Vec<usize>, elem_offset: usize) -> Tensor {
        let n: usize = shape.iter().product();
        assert!(
            elem_offset + n <= self.numel(),
            "view_offset out of bounds: offset {elem_offset} + len {n} > numel {}",
            self.numel()
        );
        #[cfg(feature = "metal")]
        {
            if let TensorInner::Metal(g) = &self.inner {
                return Tensor {
                    inner: TensorInner::Metal(g.view_offset(shape.clone(), elem_offset)),
                    shape,
                    dtype: self.dtype,
                };
            }
        }
        #[cfg(feature = "cpu")]
        {
            let data = self.cpu_read();
            let mut v = Vec::with_capacity(n);
            v.extend_from_slice(&data[elem_offset..elem_offset + n]);
            drop(data);
            return Tensor::from_cpu(v, shape);
        }
        // O8: the CUDA fallthrough below used to IGNORE elem_offset and
        // return a view of element 0 — KvCache::compact then copied the
        // STALE half of the cache instead of [shift..pos) (silent long-
        // context corruption on Qwen3 roll). A real device-side offset copy
        // needs backend/stream access (a CUDA-backend task); until then a
        // nonzero offset is a loud programming error, not wrong data.
        if elem_offset != 0 {
            panic!(
                "view_offset: nonzero elem_offset {elem_offset} on CUDA is not supported \
                 (the old code silently returned data from element 0)"
            );
        }
        #[allow(unreachable_code)]
        self.view_restricted(shape)
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
                        offset: g.offset,
                        pooled: g.pooled.clone(),
                    }),
                    shape,
                    dtype,
                };
            }
        }
        #[allow(unreachable_code)]
        Tensor {
            inner: self.inner.clone(),
            shape,
            dtype,
        }
    }
}

/// Compute-surface operations. Every method dispatches its kernels through
/// the backend; implementations are responsible for allocation and
/// synchronization semantics. Errors cross the boundary as [`BackendError`].

/// erf(x) — аппроксимация Abramowitz & Stegun 7.1.26 (max |ε| ≈ 1.5e-7),
/// та же формула, что в Metal-кернеле gelu_erf (CPU=Metal согласованы).
pub fn erf_approx(x: f32) -> f32 {
    const P: f32 = 0.3275911;
    const A1: f32 = 0.254829592;
    const A2: f32 = -0.284496736;
    const A3: f32 = 1.421413741;
    const A4: f32 = -1.453152027;
    const A5: f32 = 1.061405429;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + P * ax);
    let poly = A1 * t + A2 * t * t + A3 * t * t * t + A4 * t * t * t * t + A5 * t * t * t * t * t;
    sign * (1.0 - poly * (-ax * ax).exp())
}

/// Create the platform-default backend (env `BACKEND` can override:
/// `cpu`/`cuda`/`metal` — the feature must be compiled in; the GPU-only rule
/// keeps CPU behind an explicit opt-in).
pub fn default_backend() -> Result<std::sync::Arc<dyn Backend>, BackendError> {
    // Переменная используется только при включённых фичах GPU-бэкендов;
    // без фич сборка обязана упасть в BackendUnavailable (GPU-only правило).
    #[cfg_attr(
        not(any(feature = "metal", feature = "cuda", feature = "cpu")),
        allow(unused_variables)
    )]
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

/// Методы `from_*` принимают `&self` намеренно: контракт диспатчится через
/// `&dyn Backend`, а реализации аллоцируют в своём рантайме (входной массив
/// не конвертируется в Self). Отключаем педантичный wrong_self_convention.
#[allow(clippy::wrong_self_convention)]

/// Re-upload host f32 data into the SAME storage dtype as `src` (the
/// kv_shifted/kv_rerotated defaults must preserve the KV storage format).
/// Generic over `B: ?Sized` so the trait defaults can pass `&Self`.
fn reupload_in_dtype<B: Backend + ?Sized>(
    backend: &B,
    src: &Tensor,
    data: &[f32],
) -> Result<Tensor, BackendError> {
    use crate::dtype::Dtype;
    match src.dtype {
        Dtype::F32 => backend.from_cpu_f32(data, src.shape().to_vec()),
        Dtype::F16 => {
            let h: Vec<half::f16> = data.iter().map(|&v| half::f16::from_f32(v)).collect();
            backend.from_cpu_f16(&h, src.shape().to_vec())
        }
        // q8_0 KV: re-quantize per 32-element block (the format make_kv_tensor
        // creates: i8 data + one f32 scale per 32-block).
        Dtype::I8 => {
            let d = src.shape()[2];
            if d % 32 != 0 {
                return Err(BackendError::Shape(format!(
                    "reupload_same_dtype: q8_0 requires d % 32 == 0, got d={d}"
                )));
            }
            let blocks = d / 32;
            let rows = src.shape()[0] * src.shape()[1];
            let mut i8 = vec![0i8; data.len()];
            let mut sc = vec![1.0f32; rows * blocks];
            for r in 0..rows {
                for blk in 0..blocks {
                    let base = r * d + blk * 32;
                    let mut mx = 0.0f32;
                    for j in 0..32 {
                        mx = mx.max(data[base + j].abs());
                    }
                    let s = (mx / 127.0).max(1e-8);
                    sc[r * blocks + blk] = s;
                    for j in 0..32 {
                        i8[base + j] = (data[base + j] / s).round() as i8;
                    }
                }
            }
            backend.from_cpu_i8(&i8, &sc, src.shape().to_vec())
        }
        _ => Err(BackendError::Shape(format!(
            "reupload_same_dtype: unsupported storage dtype {:?}",
            src.dtype
        ))),
    }
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
    fn from_cpu_bf16(
        &self,
        data: &[half::bf16],
        shape: Vec<usize>,
    ) -> Result<Tensor, BackendError> {
        let f32s: Vec<f32> = data.iter().map(|h| f32::from(*h)).collect();
        self.from_cpu_f32(&f32s, shape)
    }

    /// Upload int8 weights with a per-channel fp32 scale (GEMM storage).
    /// Default: dequantize to f32.
    fn from_cpu_i8(
        &self,
        data: &[i8],
        scale: &[f32],
        shape: Vec<usize>,
    ) -> Result<Tensor, BackendError> {
        // The int8 upload (rust-inference-speech::quant) packs kb-major:
        // block (kb, col) = 32 contiguous i8 at (kb·rows + col)·32, its
        // per-32-block scale at (kb·rows + col). The default dequant
        // mirrors the packing back into the standard row-major layout.
        let rows = shape[0];
        let per = shape[1..].iter().product::<usize>();
        let nblk = per / 32;
        let mut f32s = vec![0.0f32; rows * per];
        for r in 0..rows {
            for b in 0..nblk {
                let s = scale[b * rows + r];
                for j in 0..32 {
                    f32s[r * per + b * 32 + j] = data[(b * rows + r) * 32 + j] as f32 * s;
                }
            }
        }
        self.from_cpu_f32(&f32s, shape)
    }

    /// Upload Q6_K packed weights (canonical llama.cpp layout — 210 bytes per
    /// 256-weight super-block; see `WeightFormat::Q6_K`). Only the CUDA
    /// backend stores the packed form; default: unsupported.
    fn from_cpu_q6k(&self, _data: &[u8], _shape: Vec<usize>) -> Result<Tensor, BackendError> {
        Err(BackendError::UnsupportedFormat(
            "from_cpu_q6k: only the CUDA backend stores packed Q6_K weights".into(),
        ))
    }

    /// Upload Q4_0 packed weights (canonical llama.cpp layout — 18 bytes per
    /// 32-weight block; see `WeightFormat::Int4`). Only the CUDA backend
    /// stores the packed form; default: unsupported.
    fn from_cpu_q4k(&self, _data: &[u8], _shape: Vec<usize>) -> Result<Tensor, BackendError> {
        Err(BackendError::UnsupportedFormat(
            "from_cpu_q4k: only the CUDA backend stores packed Q4_0 weights".into(),
        ))
    }

    /// Upload canonical Q4_K packed weights (llama.cpp block_q4_K — 144
    /// bytes per 256-weight super-block; GGUF Q4_K tensors). Only the CUDA
    /// backend stores the packed form; default: unsupported.
    fn from_cpu_q4kc(&self, _data: &[u8], _shape: Vec<usize>) -> Result<Tensor, BackendError> {
        Err(BackendError::UnsupportedFormat(
            "from_cpu_q4kc: only the CUDA backend stores packed Q4_K weights".into(),
        ))
    }

    /// Upload Q5_K packed weights (canonical llama.cpp layout — 176 bytes per
    /// 256-weight super-block). Only the CUDA backend stores the packed form.
    fn from_cpu_q5k(&self, _data: &[u8], _shape: Vec<usize>) -> Result<Tensor, BackendError> {
        Err(BackendError::UnsupportedFormat(
            "from_cpu_q5k: only the CUDA backend stores packed Q5_K weights".into(),
        ))
    }

    /// Upload Q3_K packed weights (canonical llama.cpp layout — 110 bytes per
    /// 256-weight super-block). Only the CUDA backend stores the packed form.
    fn from_cpu_q3k(&self, _data: &[u8], _shape: Vec<usize>) -> Result<Tensor, BackendError> {
        Err(BackendError::UnsupportedFormat(
            "from_cpu_q3k: only the CUDA backend stores packed Q3_K weights".into(),
        ))
    }

    /// Может ли бэкенд хранить GEMM-веса в f16 (halved VRAM/трафик)?
    /// Metal — да (кернелы gemm_nt_f16/bf16); CPU-референс — только f32.
    fn supports_f16_weights(&self) -> bool {
        false
    }
    /// Стандартный rotate-half RoPE (HF/Qwen3): X (1, T, D) → Y, позиции
    /// pos..pos+T. CPU-референс — прямое вычисление; Metal — кернел.
    fn rope_standard(
        &self,
        x: &Tensor,
        pos: usize,
        hd: usize,
        theta: f32,
    ) -> Result<Tensor, BackendError> {
        let (tq, d) = (x.shape()[1], x.shape()[2]);
        let xd = self.to_cpu_f32(x)?;
        let mut yd = vec![0.0f32; xd.len()];
        let half = hd / 2;
        for tt in 0..tq {
            let p = (pos + tt) as f32;
            for h in 0..d / hd {
                for i in 0..half {
                    let inv = 1.0 / theta.powf(2.0 * i as f32 / hd as f32);
                    let c = (p * inv).cos();
                    let s = (p * inv).sin();
                    // NEOX/halves (HF rotate-half): пары (i, i + hd/2)
                    let idx = (tt * d + h * hd + i) as usize;
                    let (x0, x1) = (xd[idx], xd[idx + hd / 2]);
                    yd[idx] = x0 * c - x1 * s;
                    yd[idx + hd / 2] = x1 * c + x0 * s;
                }
            }
        }
        self.from_cpu_f32(&yd, x.shape().to_vec())
    }
    /// M-RoPE text (Qwen3-VL IMROPE, ТОЛЬКО текст): пары HALVES (i, i+hd/2),
    /// но e-секция (4-я позиция = 0 для текста) не ротируется. Категория
    /// пары i по llama.cpp IMROPE: sector = i % sum(sections):
    ///   sector % 3 == 0 && sector < 3*s0 → t (pos)
    ///   sector % 3 == 1 && sector < 3*s1 → h (pos)
    ///   sector % 3 == 2 && sector < 3*s2 → w (pos)
    ///   иначе → e (позиция 0 — без ротации).
    /// Для текста t=h=w=pos, поэтому ротируются все пары кроме e-категории.
    /// Default — CPU-референс; Metal — кернел.
    fn rope_mrope_text(
        &self,
        x: &Tensor,
        pos: usize,
        hd: usize,
        theta: f32,
        sections: [usize; 4],
    ) -> Result<Tensor, BackendError> {
        let (tq, d) = (x.shape()[1], x.shape()[2]);
        let xd = self.to_cpu_f32(x)?;
        let mut yd = vec![0.0f32; xd.len()];
        let half = hd / 2;
        let sect_dims = sections.iter().sum::<usize>().max(1);
        let (s0, s1, s2) = (sections[0], sections[1], sections[2]);
        for tt in 0..tq {
            let p = (pos + tt) as f32;
            for h in 0..d / hd {
                for i in 0..half {
                    let sector = i % sect_dims;
                    let is_e = !((sector % 3 == 0 && sector < 3 * s0)
                        || (sector % 3 == 1 && sector < 3 * s1)
                        || (sector % 3 == 2 && sector < 3 * s2));
                    let inv = 1.0 / theta.powf(2.0 * i as f32 / hd as f32);
                    let idx = (tt * d + h * hd + i) as usize;
                    if is_e {
                        yd[idx] = xd[idx];
                        yd[idx + half] = xd[idx + half];
                    } else {
                        let (c, s) = ((p * inv).cos(), (p * inv).sin());
                        let (x0, x1) = (xd[idx], xd[idx + half]);
                        yd[idx] = x0 * c - x1 * s;
                        yd[idx + half] = x0 * s + x1 * c;
                    }
                }
            }
        }
        self.from_cpu_f32(&yd, x.shape().to_vec())
    }

    /// M-RoPE (Qwen3-VL IMROPE, произвольные позиции на токен): pos4d —
    /// массив [tq*4] значений (t,h,w,e на токен; для текста t=h=w=pos, e=0;
    /// для картинки по llama.cpp mtmd: t=pos_0, h=pos_0+i/nx, w=pos_0+i%nx).
    /// Категория пары i — как в rope_mrope_text (mod-3 сектор); e-категория
    /// не ротируется. Default — CPU-референс; Metal — кернел rope_mrope_f32.
    fn rope_mrope(
        &self,
        x: &Tensor,
        pos4d: &[u32],
        hd: usize,
        theta: f32,
        sections: [usize; 4],
    ) -> Result<Tensor, BackendError> {
        let (tq, d) = (x.shape()[1], x.shape()[2]);
        // Q27: release builds must not rely on debug_assert — a short pos4d
        // would index OOB below (panic inside the default impl).
        if pos4d.len() != tq * 4 {
            return Err(BackendError::Shape(format!(
                "rope_mrope: pos4d {} != tq·4 {}",
                pos4d.len(),
                tq * 4
            )));
        }
        let xd = self.to_cpu_f32(x)?;
        let mut yd = vec![0.0f32; xd.len()];
        let half = hd / 2;
        let sect_dims = sections.iter().sum::<usize>().max(1);
        let (s0, s1, s2) = (sections[0], sections[1], sections[2]);
        for tt in 0..tq {
            let (pt, ph, pw) = (
                pos4d[tt * 4] as f32,
                pos4d[tt * 4 + 1] as f32,
                pos4d[tt * 4 + 2] as f32,
            );
            for h in 0..d / hd {
                for i in 0..half {
                    let sector = i % sect_dims;
                    let (is_e, p) = if sector % 3 == 0 && sector < 3 * s0 {
                        (false, pt)
                    } else if sector % 3 == 1 && sector < 3 * s1 {
                        (false, ph)
                    } else if sector % 3 == 2 && sector < 3 * s2 {
                        (false, pw)
                    } else {
                        (true, 0.0)
                    };
                    let inv = 1.0 / theta.powf(2.0 * i as f32 / hd as f32);
                    let idx = (tt * d + h * hd + i) as usize;
                    if is_e {
                        yd[idx] = xd[idx];
                        yd[idx + half] = xd[idx + half];
                    } else {
                        let (c, s) = ((p * inv).cos(), (p * inv).sin());
                        let (x0, x1) = (xd[idx], xd[idx + half]);
                        yd[idx] = x0 * c - x1 * s;
                        yd[idx + half] = x0 * s + x1 * c;
                    }
                }
            }
        }
        self.from_cpu_f32(&yd, x.shape().to_vec())
    }

    /// Deepstack (Qwen3-VL): y[t] = x[t] + embd[t+shift] (для t+shift < T).
    fn deepstack(&self, x: &Tensor, embd: &Tensor, shift: usize) -> Result<Tensor, BackendError> {
        let (b, t, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let xv = self.to_cpu_f32(x)?;
        let ev = self.to_cpu_f32(embd)?;
        let mut y = xv.clone();
        for tt in 0..t {
            let src = tt + shift;
            if src < t {
                for dd in 0..d {
                    y[tt * d + dd] += ev[src * d + dd];
                }
            }
        }
        self.from_cpu_f32(&y, vec![b, t, d])
    }

    /// Download a tensor to host memory.
    fn to_cpu_f32(&self, t: &Tensor) -> Result<Vec<f32>, BackendError>;

    /// Activation compute dtype for GEMMs (the plan's activation axis:
    /// `TTS_ACTIVATION`/`STT_ACTIVATION` = fp32 | fp16 | bf16, default fp32).
    /// Backends that keep activations in a fixed dtype ignore it; the CUDA
    /// backend casts the GEMM A-operand accordingly (fp16/bf16: half-width
    /// A traffic; fp32: reference precision). Default: no-op.
    fn set_activation_dtype(&self, _d: Dtype) -> Result<(), BackendError> {
        Ok(())
    }

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

    /// N (iter-10): exact top-k over the FULL penalty pipeline of one logits
    /// row (the TTS slow-AR sampler): per-occurrence repetition penalty
    /// (logit>0 ? /=pen^count : *=pen^count), RAS window penalty (−ras·(cnt−1)
    /// for tokens repeated within the last `ras_window` of history),
    /// semantic bias add, temperature division, then the top-k (index,
    /// probability) pairs sorted descending — probabilities use the
    /// FULL-vocab softmax denominator, so the caller reproduces the
    /// reference `top_p_sample_vec` exactly with only the RNG draw over ≤ k
    /// candidates. `logits` is the last row of the LM head output (still on
    /// the device on CUDA — only the k window + denominator come back).
    /// Default: the exact host reference computation.
    #[allow(clippy::too_many_arguments)]
    fn penalized_topk(
        &self,
        logits: &Tensor,
        bias: &Tensor,
        history: &[u32],
        rep_penalty: f32,
        ras_penalty: f32,
        ras_window: usize,
        temperature: f32,
        k: usize,
    ) -> Result<Vec<(u32, f32)>, BackendError> {
        let v = logits.shape()[logits.shape().len() - 1];
        // N (plan 7.3): fused host path — the penalties mutate the
        // downloaded row IN PLACE (the old code cloned it into `pen`), and
        // the exp/sum/candidates build is ONE pass (the old code
        // materialized exp_vals, summed it, then built indexed). Same float
        // ops in the same order — bit-identical results, ~2 fewer full
        // passes and ~2 fewer 4·v allocations per token.
        let mut pen = self.to_cpu_f32(logits)?;
        let bv = self.to_cpu_f32(bias)?;
        if rep_penalty != 1.0 {
            for &t in history {
                let idx = t as usize;
                if idx < pen.len() {
                    if pen[idx] > 0.0 {
                        pen[idx] /= rep_penalty;
                    } else {
                        pen[idx] *= rep_penalty;
                    }
                }
            }
        }
        if ras_penalty != 1.0 && history.len() >= 2 {
            let ws = history.len().saturating_sub(ras_window);
            let mut counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
            for &t in &history[ws..] {
                *counts.entry(t).or_insert(0) += 1;
            }
            for (&t, &c) in &counts {
                if c > 1 {
                    let idx = t as usize;
                    if idx < pen.len() {
                        pen[idx] -= ras_penalty * (c - 1) as f32;
                    }
                }
            }
        }
        for i in 0..pen.len().min(bv.len()) {
            pen[i] += bv[i];
        }
        if temperature != 1.0 {
            for l in pen.iter_mut() {
                *l /= temperature;
            }
        }
        let max = pen.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        // Sequential exp + running sum (identical summation order to the old
        // exp_vals-then-sum, so the fallback trigger and probs are unchanged).
        let mut indexed: Vec<(usize, f32)> = Vec::with_capacity(v);
        indexed.extend(pen.iter().enumerate().map(|(i, &l)| {
            let e = (l - max).exp();
            sum += e;
            (i, e)
        }));
        if !sum.is_finite() || sum <= 0.0 {
            // Reference fallback: uniform over the vocab (unreachable with
            // the TTS temperature=1 path; kept for parity).
            let p = 1.0 / v.max(1) as f32;
            return Ok((0..k.min(v.max(1))).map(|i| (i as u32, p)).collect());
        }
        for (_, e) in indexed.iter_mut() {
            *e /= sum;
        }
        let kk = k.min(indexed.len());
        if kk < indexed.len() {
            let (head, pivot, _) =
                indexed.select_nth_unstable_by(kk - 1, |a, b| b.1.total_cmp(&a.1));
            let mut window: Vec<(usize, f32)> = head.to_vec();
            window.push(*pivot);
            window.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            Ok(window.into_iter().map(|(i, p)| (i as u32, p)).collect())
        } else {
            indexed.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            Ok(indexed.into_iter().map(|(i, p)| (i as u32, p)).collect())
        }
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

    /// Upload квантованных весов (Q4_0/Q8_0/Q4_K/Q6_K) из сырых GGUF-байтов.
    /// Default (CPU/CUDA-оракул): деквант → f32; Metal — нативный формат.
    fn from_quantized(
        &self,
        data: &[u8],
        shape: Vec<usize>,
        format: crate::dtype::WeightFormat,
    ) -> Result<Tensor, BackendError> {
        let tmp = crate::weights_map::WeightTensor {
            name: "q".into(),
            shape: shape.iter().map(|&d| d as u64).collect(),
            format,
            offset: 0,
            data,
        };
        // Q26: reject unsupported formats HERE with an Err — the old path
        // fell into dequant_f32's `unimplemented!()` (a panic from a pub
        // path reachable with a malformed checkpoint).
        if matches!(format, crate::dtype::WeightFormat::Int4 | crate::dtype::WeightFormat::Nf4) {
            return Err(BackendError::UnsupportedFormat(format!(
                "from_quantized: {format:?} dequant not implemented"
            )));
        }
        let f = crate::weights_map::dequant_f32(&tmp);
        self.from_cpu_f32(&f, shape)
    }

    /// Transpose a 2D tensor (row-major (M,N) → (N,M)). Default: host
    /// round-trip — only the LoRA trainer uses it, on tiny tensors, so the
    /// Metal/CUDA backends inherit this without a kernel.
    fn transpose(&self, x: &Tensor) -> Result<Tensor, BackendError> {
        if x.shape().len() != 2 {
            return Err(BackendError::Shape("transpose: 2D only".into()));
        }
        let (m, n) = (x.shape()[0], x.shape()[1]);
        let data = self.to_cpu_f32(x)?;
        let mut t = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                t[j * m + i] = data[i * n + j];
            }
        }
        self.from_cpu_f32(&t, vec![n, m])
    }

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
    /// Fused SwiGLU gate, row-major per-row gate|up halves: X is (rows, 2·ffn),
    /// and for row r / col c the op writes
    /// Y[r·ffn + c] = silu(X[r·2·ffn + c]) · X[r·2·ffn + ffn + c].
    /// (The gate and up halves are PER ROW, not one flat split of X.)
    fn silu_mul_fused(&self, x: &Tensor, y: &Tensor, ffn: usize) -> Result<(), BackendError>;

    /// DFlash2 two-tap dynamic conv (спека PR #27342):
    /// Y[t,i] = Σ_tap (DYN[(g·k+tap)·2+side, t] + BASE[i·k·2 + tap·2 + side]) · X[t−tap, i]
    /// где g = i/group_size, сдвиг ВНУТРИ блока (t−tap < 0 в блоке → 0).
    /// X/Y [n_tokens, n_embd], BASE [n_embd, kernel, 2], DYN [n_groups·kernel·2, n_tokens].
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
    ) -> Result<(), BackendError>;
    fn gelu_tanh(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError>;

    /// Точный GELU (erf): 0.5·x·(1+erf(x/√2)) — vision (clip/Qwen3-VL)
    /// и аудио-энкодер (Qwen2.5-Omni) используют erf-вариант, не tanh.
    /// Обязательный: дефолт не может записать результат в `y` (нет доступа
    /// к стораджу) — см. историю мёртвого `let _ = &yt`.
    fn gelu_erf(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError>;

    /// Complex magnitude: X [..., 2n] (re | im row-major halves) → new
    /// Y [..., n] with y[i] = sqrt(x[i]² + x[i+n]²) (silero STFT). The
    /// default is the exact host math (Metal/CPU parity); CUDA fuses it
    /// into one kernel.
    fn magnitude_sqrt(&self, x: &Tensor) -> Result<Tensor, BackendError> {
        let mut shape = x.shape().to_vec();
        let n = x.numel() / 2;
        let xh = self.to_cpu_f32(x)?;
        let mut yh = Vec::with_capacity(n);
        for i in 0..n {
            let (re, im) = (xh[i], xh[i + n]);
            yh.push((re * re + im * im).sqrt());
        }
        *shape.last_mut().unwrap() /= 2;
        self.from_cpu_f32(&yh, shape)
    }

    /// Fused LSTM cell (silero, PyTorch i/f/g/o gate order):
    /// gates = (x·W_ih + b_ih) + (h·W_hh + b_hh);
    /// c' = σ(f)·c + σ(i)·tanh(g); h' = σ(o)·tanh(c').
    /// The default composes existing ops (bit-identical to the old silero
    /// path — Metal/CPU parity); CUDA runs one fused kernel (plan 7.7).
    fn lstm_cell(
        &self,
        x: &Tensor,
        h: &Tensor,
        c: &Tensor,
        w_ih: &Tensor,
        w_hh: &Tensor,
        b_ih: &Tensor,
        b_hh: &Tensor,
    ) -> Result<(Tensor, Tensor), BackendError> {
        let hidden = x.shape()[x.shape().len() - 1];
        let gates = self.zeros(vec![1, 4 * hidden])?;
        self.gemm_nt(x, w_ih, &gates)?;
        self.add_bias(&gates, b_ih, &gates)?;
        let hg = self.zeros(vec![1, 4 * hidden])?;
        self.gemm_nt(h, w_hh, &hg)?;
        self.add_bias(&hg, b_hh, &hg)?;
        self.add(&gates, &hg, &gates)?;
        let slice = |off: usize| -> Result<Tensor, BackendError> {
            let s = self.zeros(vec![1, hidden])?;
            self.slice_last(&gates, &s, off)?;
            Ok(s)
        };
        let i = slice(0)?;
        let f = slice(hidden)?;
        let g = slice(2 * hidden)?;
        let o = slice(3 * hidden)?;
        let (i_sig, f_sig, o_sig, g_tanh) = (
            self.zeros(vec![1, hidden])?,
            self.zeros(vec![1, hidden])?,
            self.zeros(vec![1, hidden])?,
            self.zeros(vec![1, hidden])?,
        );
        self.sigmoid(&i, &i_sig)?;
        self.sigmoid(&f, &f_sig)?;
        self.sigmoid(&o, &o_sig)?;
        self.tanh(&g, &g_tanh)?;
        let fc = self.zeros(vec![1, hidden])?;
        let ig = self.zeros(vec![1, hidden])?;
        self.mul(&f_sig, c, &fc)?;
        self.mul(&i_sig, &g_tanh, &ig)?;
        let c_new = self.zeros(vec![1, hidden])?;
        self.add(&fc, &ig, &c_new)?;
        let c_tanh = self.zeros(vec![1, hidden])?;
        self.tanh(&c_new, &c_tanh)?;
        let h_new = self.zeros(vec![1, hidden])?;
        self.mul(&o_sig, &c_tanh, &h_new)?;
        Ok((h_new, c_new))
    }

    /// Zipformer SwooshL: log1p(exp(x−4)) − 0.08·x − 0.035 (icefall).
    /// Returns a NEW tensor (a write-into-`y` default is impossible — no
    /// storage access, see gelu_erf): the default is the exact host math
    /// (Metal/CPU parity), the CUDA backend fuses it into elementwise_f32.
    fn swoosh_l(&self, x: &Tensor) -> Result<Tensor, BackendError> {
        let xh = self.to_cpu_f32(x)?;
        let mut yh = Vec::with_capacity(xh.len());
        for &v in &xh {
            let z = v - 4.0;
            let sp = if z > 0.0 {
                z + (-z).exp().ln_1p()
            } else {
                (z.exp() + 1.0).ln()
            };
            yh.push(sp - 0.08 * v - 0.035);
        }
        self.from_cpu_f32(&yh, x.shape().to_vec())
    }

    /// Zipformer SwooshR: x + SwooshL(x). See [`Backend::swoosh_l`].
    fn swoosh_r(&self, x: &Tensor) -> Result<Tensor, BackendError> {
        let xh = self.to_cpu_f32(x)?;
        let mut yh = Vec::with_capacity(xh.len());
        for &v in &xh {
            let z = v - 4.0;
            let sp = if z > 0.0 {
                z + (-z).exp().ln_1p()
            } else {
                (z.exp() + 1.0).ln()
            };
            yh.push(v + sp - 0.08 * v - 0.035);
        }
        self.from_cpu_f32(&yh, x.shape().to_vec())
    }

    /// Zipformer BiasNorm: y[r,j] = x[r,j]·exp(log_scale) /
    /// sqrt(mean_j (x[r,j]−bias[j])²) with a 1e-5 floor on the mean.
    /// Returns a NEW tensor (see [`Backend::swoosh_l`]); the default is the
    /// exact host math (Metal/CPU parity), CUDA runs a per-row kernel.
    fn bias_norm_zipformer(
        &self,
        x: &Tensor,
        bias: &Tensor,
        log_scale: f32,
    ) -> Result<Tensor, BackendError> {
        let shape = x.shape().to_vec();
        let d = *shape.last().unwrap();
        let rows = shape.iter().take(shape.len() - 1).product::<usize>();
        let xh = self.to_cpu_f32(x)?;
        let bh = self.to_cpu_f32(bias)?;
        let scale0 = log_scale.exp();
        let mut yh = vec![0.0f32; xh.len()];
        for r in 0..rows {
            let mut s = 0.0f32;
            for j in 0..d {
                let v = xh[r * d + j] - bh[j];
                s += v * v;
            }
            let sc = scale0 / (s / d as f32).sqrt().max(1e-5);
            for j in 0..d {
                yh[r * d + j] = xh[r * d + j] * sc;
            }
        }
        self.from_cpu_f32(&yh, shape)
    }

    fn snake(&self, x: &Tensor, a: &Tensor, y: &Tensor) -> Result<(), BackendError>;

    /// Vision patch merge 2×2 + patch_bias + position_embd (Qwen3-VL):
    /// conv-сетка (per×per) → 4 токена на блок в порядке (c0a0,c1a0,c0a1,c1a1);
    /// pos_embd переставляется той же перестановкой. X/POS [per², d] → Y.
    /// CPU-референс в cpu.rs, Metal — кернел merge2x2_add_f32.
    fn merge2x2_add(
        &self,
        x: &Tensor,
        bias: &Tensor,
        pos: &Tensor,
        y: &Tensor,
        per: usize,
    ) -> Result<(), BackendError> {
        // O7: the old default COMPUTED the result and threw it away
        // (`let _ = &yt`) — silent garbage on any backend without a kernel
        // (CUDA!): Qwen3-VL vision inputs were zeros/stale pool data with
        // no error anywhere. Fail loudly until a real kernel exists.
        let _ = (x, bias, pos, y, per);
        Err(BackendError::UnsupportedFormat(
            "merge2x2_add: no kernel on this backend (CPU reference lives in cpu.rs, Metal has a kernel, CUDA does not)".into(),
        ))
    }

    /// Vision group 4 соседних токенов: [n, d] → [n/4, 4·d] (deepstack,
    /// post_ln). CPU-референс в cpu.rs; Metal — кернел group4_f32.
    fn group4(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        // O7: see merge2x2_add — the old default threw the result away.
        let _ = (x, y);
        Err(BackendError::UnsupportedFormat(
            "group4: no kernel on this backend (CPU reference lives in cpu.rs, Metal has a kernel, CUDA does not)".into(),
        ))
    }

    /// Vision merger concat: chunks ([nq, proj], ≤4) → [4·nq, proj], строки
    /// чередуются: out[q·4+j] = chunk_j[q]. CPU-референс в cpu.rs; Metal —
    /// кернел concat4_f32.
    fn concat4(&self, chunks: &[Tensor], y: &Tensor) -> Result<(), BackendError> {
        // O7: see merge2x2_add — the old default threw the result away.
        let _ = (chunks, y);
        Err(BackendError::UnsupportedFormat(
            "concat4: no kernel on this backend (CPU reference lives in cpu.rs, Metal has a kernel, CUDA does not)".into(),
        ))
    }

    // ── norms ──
    fn rmsnorm(&self, x: &Tensor, w: &Tensor, y: &Tensor, eps: f32) -> Result<(), BackendError>;
    fn layernorm(
        &self,
        x: &Tensor,
        w: &Tensor,
        b: &Tensor,
        y: &Tensor,
        eps: f32,
    ) -> Result<(), BackendError>;

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

    /// Embedding with vision-token override (Ф1.11): токены == override_id
    /// заменяются строками из `vision` (по порядку появления). Всё на GPU.
    fn embedding_override(
        &self,
        idx: &[u32],
        table: &Tensor,
        vision: &Tensor,
        y: &Tensor,
        override_id: u32,
    ) -> Result<(), BackendError> {
        // O7: the old default COMPUTED the result and threw it away
        // (`let _ = &out`) — silent garbage on CUDA (the CPU reference is
        // overridden in cpu.rs, Metal has a kernel). Fail loudly instead.
        let _ = (idx, table, vision, y, override_id);
        Err(BackendError::UnsupportedFormat(
            "embedding_override: no kernel on this backend (CPU reference lives in cpu.rs, Metal has a kernel, CUDA does not)".into(),
        ))
    }

    // ── RoPE ──
    /// Halved RoPE: X (T, head_dim) rotated by (cos, sin) buffers, applying to
    /// the first `rope_dim` columns. Angles computed in f32.
    fn rope(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        y: &Tensor,
        rope_dim: usize,
    ) -> Result<(), BackendError>;
    /// Interleaved RoPE (fishaudio convention): pairs (x0, x1) rotate together.
    /// `start` is the absolute position offset — cos/sin are indexed at
    /// (t + start) so cached tokens keep their original rotation.
    fn rope_interleaved(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        y: &Tensor,
        rope_dim: usize,
        start: usize,
    ) -> Result<(), BackendError>;
    /// Per-head RoPE: X (B, T, h·hd) with `h` heads of `hd` dims; rotates the
    /// first `rope_dim` columns of every head.
    fn rope_head(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        y: &Tensor,
        h: usize,
        rope_dim: usize,
    ) -> Result<(), BackendError>;

    // ── head norms (f32-compute) ──
    /// RMSNorm applied per head: X (rows, h·hd), w per head-dim (hd), h heads.
    fn rmsnorm_head(
        &self,
        x: &Tensor,
        w: &Tensor,
        y: &Tensor,
        h: usize,
        eps: f32,
    ) -> Result<(), BackendError>;
    /// Fused per-head RMSNorm + rotate-half RoPE (Ф8): Y = rope(rmsnorm_head(X)).
    /// X (1, T, h·hd), w [hd], позиции токенов pos..pos+T.
    /// Default — два шага (rmsnorm_head + rope_standard); Metal — fused-кернел
    /// (qk_norm+rope в attn — 4 кернела на слой → 2).
    fn rmsnorm_head_rope(
        &self,
        x: &Tensor,
        w: &Tensor,
        pos: usize,
        hd: usize,
        theta: f32,
        h: usize,
        eps: f32,
    ) -> Result<Tensor, BackendError> {
        let tmp = self.zeros(x.shape().to_vec())?;
        self.rmsnorm_head(x, w, &tmp, h, eps)?;
        self.rope_standard(&tmp, pos, hd, theta)
    }
    /// LayerNorm applied per head: X (rows, h·hd), w/b per head-dim (hd).
    fn layernorm_head(
        &self,
        x: &Tensor,
        w: &Tensor,
        b: &Tensor,
        y: &Tensor,
        h: usize,
        eps: f32,
    ) -> Result<(), BackendError>;

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
    fn add_scaled(
        &self,
        a: &Tensor,
        b: &Tensor,
        y: &Tensor,
        scale: f32,
    ) -> Result<(), BackendError>;
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

    /// Bidirectional MHSA with relative positions (NeMo conformer rel_pos /
    /// icefall zipformer): q/k/v (T, H·dk), p (2T−1, H·dp), u/vb (H·dk, the
    /// per-head pos biases flattened), y (T, H·dk).
    /// scores = ((q+u)·kᵀ + shift((q+v)·pᵀ)) / √dk; softmax over keys;
    /// y = softmax·v. `mode` 0 = NeMo Transformer-XL shift, 1 = zipformer
    /// direct shift. When `w0` is Some the head-0 post-softmax weights
    /// (T, T) are written there (zipformer nonlin attention). f32-compute.
    #[allow(clippy::too_many_arguments)]
    fn mhsa_rel_pos(
        &self,
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _qp: &Tensor,
        _p: &Tensor,
        _u: &Tensor,
        _vb: &Tensor,
        _y: &Tensor,
        _wfull: Option<&Tensor>,
        _dk: usize,
        _dp: usize,
        _mode: u32,
        _scale: f32,
    ) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedFormat(
            "mhsa_rel_pos: backend does not implement relative-position attention".into(),
        ))
    }

    /// Per-head attention application over time (zipformer SelfAttention):
    /// y (T, H·vh) = w (H, T, T) · x (T, H·vh). f32-compute.
    fn mhsa_apply_weights(
        &self,
        _w: &Tensor,
        _x: &Tensor,
        _y: &Tensor,
        _vh: usize,
    ) -> Result<(), BackendError> {
        Err(BackendError::UnsupportedFormat(
            "mhsa_apply_weights: backend does not implement attention-weight application".into(),
        ))
    }

    // ── KV cache ──
    /// Write SRC (B, Tq, D) into DST (B, cap, D) starting at position `pos`.
    /// Quantized KV formats quantize on write here; fused attention
    /// dequantizes inline on read (per-component KV config).
    fn kv_update(&self, dst: &Tensor, src: &Tensor, pos: usize) -> Result<(), BackendError>;

    /// KV compaction shift (plan 1.6): rows [shift..pos) of the (B, cap, D)
    /// buffer → a NEW tensor with them at [0..pos−shift) (the rest zeroed),
    /// storage-dtype-preserving (f32 / f16 / q8_0 with the per-32-block
    /// scales). The source/destination regions are disjoint by construction
    /// (shift = pos/2). Returns a new tensor (a write-into-`buf` default is
    /// impossible — no storage access); CUDA shifts in place.
    fn kv_shifted(&self, buf: &Tensor, shift: usize, pos: usize) -> Result<Tensor, BackendError> {
        let (b, cap, d) = (buf.shape()[0], buf.shape()[1], buf.shape()[2]);
        let xh = self.to_cpu_f32(buf)?;
        let n = (pos - shift) * d;
        let mut yh = vec![0.0f32; b * cap * d];
        for bi in 0..b {
            let src = bi * cap * d + shift * d;
            yh[bi * cap * d..bi * cap * d + n].copy_from_slice(&xh[src..src + n]);
        }
        reupload_in_dtype(self, buf, &yh)
    }

    /// KV re-rotation after a compaction shift (plan 1.6): the kept keys
    /// were stored rotated for their ORIGINAL absolute positions
    /// [shift..pos); after the shift they must carry the rotations of
    /// [0..pos−shift). Rotates every pair (2i, 2i+1) of every head by
    /// −shift·theta^(−2i/hd). Returns a new tensor; CUDA rotates in place.
    fn kv_rerotated(
        &self,
        k: &Tensor,
        shift: usize,
        head_dim: usize,
        theta: f32,
        // M-RoPE sections (mirror of rope_mrope_text): the pairs in the
        // "empty" (t) sector carry no position rotation; None = the plain
        // rope (all pairs rotate).
        mrope_sections: Option<[usize; 4]>,
    ) -> Result<Tensor, BackendError> {
        let (b, cap, d) = (k.shape()[0], k.shape()[1], k.shape()[2]);
        let nheads = d / head_dim;
        let pairs = head_dim / 2;
        let mut xh = self.to_cpu_f32(k)?;
        let sect_dims = mrope_sections.map(|s| s[0] + s[1] + s[2]).unwrap_or(0);
        let rotates = |pi: usize| -> bool {
            if sect_dims == 0 {
                return true;
            }
            let (s0, s1, s2) = {
                let s = mrope_sections.unwrap();
                (s[0], s[1], s[2])
            };
            let sector = pi % sect_dims;
            (sector % 3 == 0 && sector < 3 * s0)
                || (sector % 3 == 1 && sector < 3 * s1)
                || (sector % 3 == 2 && sector < 3 * s2)
        };
        for row in 0..b * cap {
            for hh in 0..nheads {
                let hbase = row * d + hh * head_dim;
                for pi in 0..pairs {
                    if !rotates(pi) {
                        continue;
                    }
                    let ang = -(shift as f32) * theta.powf(-2.0 * pi as f32 / head_dim as f32);
                    let (c, s) = (ang.cos(), ang.sin());
                    let (x, y) = (xh[hbase + 2 * pi], xh[hbase + 2 * pi + 1]);
                    xh[hbase + 2 * pi] = x * c - y * s;
                    xh[hbase + 2 * pi + 1] = x * s + y * c;
                }
            }
        }
        reupload_in_dtype(self, k, &xh)
    }

    // ── codebook search (DAC) ──
    /// For every row of X (B·T, D) find the nearest row of `table` (V, D)
    /// (L2) and write its index into `idx` (host buffer, len B·T).
    fn codebook_search(
        &self,
        x: &Tensor,
        table: &Tensor,
        idx: &mut [u32],
    ) -> Result<(), BackendError>;
}
