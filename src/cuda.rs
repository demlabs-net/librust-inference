//! CUDA backend (`cuda` feature, Phase 3) — cudarc over the CUDA driver API,
//! our own PTX kernels.
//!
//! Kernels live in `src/cuda/kernels.cu`, compiled to PTX by `build.rs`
//! (nvcc -arch=compute_86; committed `kernels.ptx` is the fallback/source of
//! truth). One PTX binary JITs onto sm_86 (RTX 3090), sm_89 (RTX 4090) and
//! sm_120 (RTX 5090).
//!
//! First-iteration scope: the memory-transfer + elementwise + naive-GEMM
//! surface (correctness over speed). The remaining contract methods return
//! [`BackendError::Kernel`] "not implemented yet" — they fill in iteratively
//! (the CPU reference + golden vectors are the acceptance oracle).
//!
//! Determinism: kernel reductions iterate in ascending order (same
//! convention as the CPU reference backend).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cudarc::cublas::result;
use cudarc::cublas::safe::GemmConfig;
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::sys::{cublasComputeType_t, cublasGemmAlgo_t, cudaDataType};
use cudarc::cublas::{CudaBlas, safe::Gemm};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr,
    LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;

use crate::backend::{Backend, Tensor};
use crate::error::BackendError;

/// Raw cuBLAS handle. cuBLAS handles are not Send/Sync by themselves; the
/// backend serializes every use behind a mutex, matching what cudarc's own
/// `CudaBlas` does for its handle field.
#[derive(Clone, Copy)]
pub(crate) struct RawCublas(pub(crate) cudarc::cublas::sys::cublasHandle_t);
unsafe impl Send for RawCublas {}
unsafe impl Sync for RawCublas {}

/// Elementwise op codes — mirror `elementwise_f32` switch in kernels.cu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
enum ElemOp {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    AddScalar = 4,
    Scale = 5,
    Sqrt = 6,
    Log = 7,
    MaxScalar = 8,
    Relu = 9,
    LeakyReluClip = 10,
    Sigmoid = 11,
    Tanh = 12,
    Silu = 13,
    GeluTanh = 14,
}

fn map_err(e: impl std::fmt::Display) -> BackendError {
    let s = e.to_string();
    if s.to_lowercase().contains("out of memory") {
        BackendError::OutOfMemory(s)
    } else {
        BackendError::native(e)
    }
}

/// Pooled CUDA buffer (Phase 3.5b): owns a `CudaSlice<f32>` and returns it
/// to the backend's size-keyed pool when the last `Arc` reference drops —
/// inference allocates hundreds of temporaries per token, so avoiding
/// `cuMemAlloc` per op is a launch-budget win.
#[derive(Debug)]
pub struct CudaBuffer {
    slice: Option<CudaSlice<f32>>,
    pool: Option<Arc<Mutex<HashMap<usize, Vec<CudaSlice<f32>>>>>>,
}

impl CudaBuffer {
    pub(crate) fn as_slice(&self) -> &CudaSlice<f32> {
        self.slice.as_ref().expect("CudaBuffer empty")
    }

    pub(crate) fn as_slice_mut(&mut self) -> &mut CudaSlice<f32> {
        self.slice.as_mut().expect("CudaBuffer empty")
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        if let (Some(s), Some(pool)) = (self.slice.take(), &self.pool) {
            pool.lock().unwrap().entry(s.len()).or_default().push(s);
        }
    }
}

/// CUDA backend over the own PTX kernels. One device, one stream, one
/// module; `&dyn Backend` consumers never touch this type.
pub struct CudaBackend {
    /// Owns the CUDA context — the stream/module borrow it, so this field
    /// must outlive them (kept for lifetime, not read directly).
    #[allow(dead_code)]
    dev: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    functions: Mutex<HashMap<&'static str, CudaFunction>>,
    /// Size-keyed buffer pool (elements) — see [`CudaBuffer`].
    pool: Arc<Mutex<HashMap<usize, Vec<CudaSlice<f32>>>>>,
    /// cuBLAS handle, lazily created on the first GEMM (Phase 3.5). If cuBLAS
    /// is unavailable (missing libcublas), GEMMs fall back to the naive PTX
    /// kernel — logged once, never fatal.
    cublas: Mutex<Option<Arc<CudaBlas>>>,
    /// fp16 scratch pool (activations cast for fp16-weight GEMMs).
    pool_f16: Arc<Mutex<HashMap<usize, Vec<cudarc::driver::CudaSlice<half::f16>>>>>,
    /// int8 scratch pool (activations quantized for int8-weight GEMMs) —
    /// raw cudaMalloc per quant call added ~10-20 us to every decode GEMM.
    pool_i8: Arc<Mutex<HashMap<usize, Vec<cudarc::driver::CudaSlice<i8>>>>>,
    /// Raw cuBLAS handle (fp16 gemm_ex path); destroyed on Drop.
    cublas_raw: Mutex<Option<RawCublas>>,
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        use cudarc::cublas::sys;
        if let Some(h) = *self.cublas_raw.lock().unwrap() {
            unsafe { sys::cublasDestroy_v2(h.0) };
        }
    }
}

impl CudaBackend {
    pub fn new(ordinal: usize) -> Result<Self, BackendError> {
        let dev = CudaContext::new(ordinal).map_err(map_err)?;
        let stream = dev.default_stream();
        // The PTX is embedded at compile time (include_str!): reading it from
        // the OUT_DIR path at runtime would break on any other machine — the
        // container has no such path (CUDA_ERROR_FILE_NOT_FOUND).
        let ptx = Ptx::from_src(include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx")));
        let module = dev.load_module(ptx).map_err(map_err)?;
        Ok(Self {
            dev,
            stream,
            module,
            functions: Mutex::new(HashMap::new()),
            pool: Arc::new(Mutex::new(HashMap::new())),
            cublas: Mutex::new(None),
            pool_f16: Arc::new(Mutex::new(HashMap::new())),
            pool_i8: Arc::new(Mutex::new(HashMap::new())),
            cublas_raw: Mutex::new(None),
        })
    }

    fn take_pooled_f16(&self, n: usize) -> Result<cudarc::driver::CudaSlice<half::f16>, BackendError> {
        let mut pool = self.pool_f16.lock().unwrap();
        if let Some(bucket) = pool.get_mut(&n) {
            if let Some(s) = bucket.pop() {
                return Ok(s);
            }
        }
        unsafe { self.stream.alloc::<half::f16>(n) }.map_err(map_err)
    }

    fn wrap_pooled_f16(&self, s: cudarc::driver::CudaSlice<half::f16>) -> cudarc::driver::CudaSlice<half::f16> {
        // Return to the pool on drop via a wrapper is not possible for a bare
        // slice; keep a Vec<CudaSlice> pool and hand slices back explicitly
        // in the f16 GEMM path below.
        let _ = &s;
        s
    }

    /// Raw cuBLAS handle bound to this backend's stream (gemm_ex fp16/bf16
    /// path — CudaBlas hides its handle and its safe Gemm<f16> forces fp16
    /// output). The returned guard MUST be held for the whole gemm_ex call:
    /// cuBLAS handles are NOT thread-safe, and the TTS pool drives GEMMs from
    /// several threads at once (generation + codec decoder) — a concurrent
    /// gemm_ex on one handle returned CUDA_ERROR_INVALID_VALUE mid-decode
    /// (seen live: "DAC decode failed" on bf16 codec weights).
    fn cublas_raw_guard(
        &self,
    ) -> Result<(cudarc::cublas::sys::cublasHandle_t, std::sync::MutexGuard<'_, Option<RawCublas>>), BackendError> {
        use cudarc::cublas::sys;
        let mut guard = self.cublas_raw.lock().unwrap();
        if let Some(h) = *guard {
            return Ok((h.0, guard));
        }
        let mut h = std::ptr::null_mut();
        let st = unsafe { sys::cublasCreate_v2(&mut h) };
        if st != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(map_err(format!("cublasCreate_v2: {st:?}")));
        }
        // No cublasSetStream: the backend uses the legacy (null) stream, which
        // is cuBLAS's default — same ordering as every other kernel launch.
        *guard = Some(RawCublas(h));
        Ok((h, guard))
    }

    /// Cast x (f32) into a bf16 scratch buffer of the same size.
    fn cast_bf16(&self, x: &Tensor) -> Result<cudarc::driver::CudaSlice<half::bf16>, BackendError> {
        let n = x.numel();
        let mut y = unsafe { self.stream.alloc::<half::bf16>(n) }.map_err(map_err)?;
        let f = self.function("cast_f32_bf16")?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(&mut y);
        let n32 = n as i32;
        la.arg(&n32);
        unsafe { la.launch(cfg) }.map_err(map_err)?;
        Ok(y)
    }

    /// Quantize activations f32 (m, k) → int8 + per-32-block scales (m, k/32).
    /// Q8_0 layout — matches `quant_f32_i8_blk` in kernels.cu. k must be a
    /// multiple of 32 (all TTS/STT GEMM dims are; callers pad otherwise).
    fn quant_i8_blk(&self, x: &Tensor) -> Result<(cudarc::driver::CudaSlice<i8>, cudarc::driver::CudaSlice<f32>), BackendError> {
        let (m, k) = (x.shape()[0], x.shape()[1]);
        if k % 32 != 0 {
            return Err(BackendError::Shape(format!(
                "quant_i8_blk: k={k} not a multiple of 32"
            )));
        }
        let nblk = k / 32;
        let q = {
            let mut pool = self.pool_i8.lock().unwrap();
            if let Some(bucket) = pool.get_mut(&(m * k)) {
                if let Some(s) = bucket.pop() {
                    s
                } else {
                    unsafe { self.stream.alloc::<i8>(m * k) }.map_err(map_err)?
                }
            } else {
                unsafe { self.stream.alloc::<i8>(m * k) }.map_err(map_err)?
            }
        };
        let sc = self.take_pooled(m * nblk)?;
        let f = self.function("quant_f32_i8_blk")?;
        let cfg = LaunchConfig::for_num_elems((m * nblk) as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(&q);
        la.arg(&sc);
        let m32 = m as i32;
        let k32 = k as i32;
        la.arg(&m32);
        la.arg(&k32);
        unsafe { la.launch(cfg) }.map_err(map_err)?;
        Ok((q, sc))
    }

    /// Cast x (f32) into an fp16 scratch buffer of the same size.
    fn cast_f16(&self, x: &Tensor) -> Result<cudarc::driver::CudaSlice<half::f16>, BackendError> {
        let n = x.numel();
        let mut y = self.take_pooled_f16(n)?;
        let f = self.function("cast_f32_f16")?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(&mut y);
        let n32 = n as i32;
        la.arg(&n32);
        unsafe { la.launch(cfg) }.map_err(map_err)?;
        Ok(y)
    }

    /// cuBLAS GEMM (row-major semantics, see the two call sites). The handle
    /// is lazily created under the same mutex that serializes every call —
    /// cuBLAS handles are NOT thread-safe, and the TTS pool can drive GEMMs
    /// from several threads at once (generation + top-up).
    /// `CUDA_NO_CUBLAS=1` forces the naive PTX path (benchmarks, debug).
    /// `x`/`y` are passed in cuBLAS argument order (first/second matrix).
    fn gemm_cublas(&self, cfg: GemmConfig<f32>, x: &Tensor, y: &Tensor, c: &Tensor) -> Result<bool, BackendError> {
        if std::env::var_os("CUDA_NO_CUBLAS").is_some() {
            return Ok(false);
        }
        let mut guard = self.cublas.lock().unwrap();
        if guard.is_none() {
            match CudaBlas::new(self.stream.clone()) {
                Ok(b) => *guard = Some(Arc::new(b)),
                Err(e) => {
                    eprintln!("[cuda] cuBLAS unavailable, GEMM falls back to naive PTX kernel: {e}");
                    return Ok(false);
                }
            }
        }
        let blas = guard.as_ref().unwrap().clone();
        unsafe { blas.gemm(cfg, x.as_cuda(), y.as_cuda(), c.as_cuda_mut()) }
            .map_err(|e| map_err(format!("cublasSgemm: {e}")))?;
        Ok(true)
    }

    fn wrap_pooled(&self, slice: CudaSlice<f32>) -> Arc<CudaBuffer> {
        Arc::new(CudaBuffer { slice: Some(slice), pool: Some(self.pool.clone()) })
    }

    /// Block until all queued work on the backend stream completes.
    pub fn synchronize(&self) -> Result<(), BackendError> {
        self.stream.synchronize().map_err(map_err)
    }

    fn take_pooled(&self, n: usize) -> Result<CudaSlice<f32>, BackendError> {
        let mut pool = self.pool.lock().unwrap();
        if let Some(bucket) = pool.get_mut(&n) {
            if let Some(s) = bucket.pop() {
                return Ok(s);
            }
        }
        self.stream.alloc_zeros::<f32>(n).map_err(map_err)
    }

    fn function(&self, name: &'static str) -> Result<CudaFunction, BackendError> {
        let mut cache = self.functions.lock().unwrap();
        if let Some(f) = cache.get(name) {
            return Ok(f.clone());
        }
        let f = self.module.load_function(name).map_err(map_err)?;
        cache.insert(name, f.clone());
        Ok(f)
    }

    /// One-shot elementwise (in-place safe: `out` may alias `x`).
    fn elementwise(
        &self,
        op: ElemOp,
        x: &Tensor,
        y: Option<&Tensor>,
        out: &Tensor,
        c: f32,
    ) -> Result<(), BackendError> {
        let n = x.numel();
        let xd = x.as_cuda();
        let yd = y.map(|t| t.as_cuda()).unwrap_or(xd);
        let od = out.as_cuda();
        let n32 = n as i32;
        let opc = op as i32;
        let f = self.function("elementwise_f32")?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(xd);
        la.arg(yd);
        la.arg(od);
        la.arg(&n32);
        la.arg(&opc);
        la.arg(&c);
        unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
    }
}

impl Backend for CudaBackend {
    fn zeros(&self, shape: Vec<usize>) -> Result<Tensor, BackendError> {
        let n: usize = shape.iter().product();
        let mut s = self.take_pooled(n)?;
        self.stream.memset_zeros(&mut s).map_err(map_err)?;
        Ok(Tensor::from_cuda(self.wrap_pooled(s), shape))
    }

    fn from_cpu_f32(&self, data: &[f32], shape: Vec<usize>) -> Result<Tensor, BackendError> {
        let n: usize = shape.iter().product();
        if data.len() != n {
            return Err(BackendError::Shape(format!(
                "from_cpu_f32: {} elements for shape {shape:?}",
                data.len()
            )));
        }
        let mut s = self.take_pooled(n)?;
        self.stream.memcpy_htod(data, &mut s).map_err(map_err)?;
        Ok(Tensor::from_cuda(self.wrap_pooled(s), shape))
    }

    fn from_cpu_f16(&self, data: &[half::f16], shape: Vec<usize>) -> Result<Tensor, BackendError> {
        let n: usize = shape.iter().product();
        if data.len() != n {
            return Err(BackendError::Shape(format!(
                "from_cpu_f16: {} elements for shape {shape:?}",
                data.len()
            )));
        }
        let mut s = unsafe { self.stream.alloc::<half::f16>(n) }.map_err(map_err)?;
        self.stream.memcpy_htod(data, &mut s).map_err(map_err)?;
        Ok(Tensor::from_cuda_f16(s, shape))
    }

    fn from_cpu_bf16(&self, data: &[half::bf16], shape: Vec<usize>) -> Result<Tensor, BackendError> {
        let n: usize = shape.iter().product();
        if data.len() != n {
            return Err(BackendError::Shape(format!(
                "from_cpu_bf16: {} elements for shape {shape:?}",
                data.len()
            )));
        }
        let mut s = unsafe { self.stream.alloc::<half::bf16>(n) }.map_err(map_err)?;
        self.stream.memcpy_htod(data, &mut s).map_err(map_err)?;
        Ok(Tensor::from_cuda_bf16(s, shape))
    }

    fn from_cpu_i8(&self, data: &[i8], scale: &[f32], shape: Vec<usize>) -> Result<Tensor, BackendError> {
        let n: usize = shape.iter().product();
        if data.len() != n {
            return Err(BackendError::Shape(format!(
                "from_cpu_i8: {} elements for shape {shape:?}",
                data.len()
            )));
        }
        let mut s = unsafe { self.stream.alloc::<i8>(n) }.map_err(map_err)?;
        self.stream.memcpy_htod(data, &mut s).map_err(map_err)?;
        let mut sc = unsafe { self.stream.alloc::<f32>(scale.len()) }.map_err(map_err)?;
        self.stream.memcpy_htod(scale, &mut sc).map_err(map_err)?;
        Ok(Tensor::from_cuda_i8(s, sc, shape))
    }

    fn to_cpu_f32(&self, t: &Tensor) -> Result<Vec<f32>, BackendError> {
        self.stream.clone_dtoh(t.as_cuda()).map_err(map_err)
    }

    fn gemm(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<(), BackendError> {
        let (m, k) = (a.shape()[0], a.shape()[1]);
        let n = b.shape()[1];
        // Row-major C[m,n] = A[m,k]·B[k,n]. A row-major buffer read as
        // column-major is its transpose, so feed B (col-major [n,k] = Bᵀ) and
        // A (col-major [k,m] = Aᵀ) untransposed: op = Bᵀ·Aᵀ = (A·B)ᵀ, which
        // lands in the row-major [m,n] buffer layout unchanged.
        let cfg = GemmConfig::<f32> {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: 1.0,
            lda: n as i32,
            ldb: k as i32,
            beta: 0.0,
            ldc: n as i32,
        };
        if self.gemm_cublas(cfg, b, a, c)? {
            return Ok(());
        }
        let (m32, k32, n32) = (m as i32, k as i32, n as i32);
        let f = self.function("gemm_naive_f32")?;
        let cfg = LaunchConfig::for_num_elems((m * n) as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(a.as_cuda());
        la.arg(b.as_cuda());
        la.arg(c.as_cuda());
        la.arg(&m32);
        la.arg(&k32);
        la.arg(&n32);
        unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
    }

    fn gemm_nt(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<(), BackendError> {
        let (m, k) = (a.shape()[0], a.shape()[1]);
        let n = b.shape()[0];
        // Weight formats: fp16/bf16 cast the activation row(s) and run cuBLAS
        // gemm_ex (16F×16F → 32F out, fp32 accumulate) — halves the weight
        // read traffic (the decode bottleneck) and the weight VRAM; int8 uses
        // our Q8_0-style kernels below.
        if let crate::backend::TensorInner::CudaI8 { data: b8, scale: wscale } = &b.inner {
            // Q8_0-style int8 GEMM: per-32-block weight + activation scales,
            // our own kernels (cuBLAS IGEMM was 10-50x slower than fp16 on
            // decode shapes and cannot express block scales anyway). m==1
            // (decode hot path) uses the split-k kernel (S slices over k —
            // m=1 alone cannot fill the GPU: the old 1-byte-load kernel
            // measured ~130 us vs 10.5 us fp16); m>1 uses the tiled one.
            let (q, ascale) = self.quant_i8_blk(a)?;
            let k32 = k as i32;
            let n32 = n as i32;
            let m32 = m as i32;
            let c = c.as_cuda_mut();
            let nblk = k / 32;
            let grid_x = (n + 255) / 256;
            // Split-k: largest power-of-2 slice count that keeps the grid
            // near the GPU's block capacity (m=1 leaves most SMs idle
            // otherwise — e.g. 8 blocks for n=2048; m≤64 already has m× the
            // row blocks, so the grid cap applies to m=1 only). Requires
            // 16-byte aligned columns and a k-slice fitting the shared A.
            let s = if m == 1 {
                [32usize, 16, 8, 4, 2, 1]
                    .iter()
                    .copied()
                    .find(|&s| nblk % s == 0 && (k / s) <= 2048 && grid_x * s <= 512)
                    .unwrap_or(1)
            } else {
                [32usize, 16, 8, 4, 2, 1]
                    .iter()
                    .copied()
                    .find(|&s| nblk % s == 0 && (k / s) <= 2048)
                    .unwrap_or(1)
            };
            let use_splitk = s > 1 && n % 16 == 0 && (k / s) <= 2048;
            if m <= 64 && use_splitk {
                // m≤64 (TTS fill/decode, m = tokens): one block per
                // (row, 256 columns), split over k. The split-k kernel
                // re-reads B from L2 once per output row — cheaper than the
                // tiled kernel's latency-bound small-m occupancy (166 us vs
                // 238 us at 32x6144x2048). m>64 (STT frames) uses tiled,
                // which reads B exactly once.
                let cpart = self.take_pooled(s * m * n)?;
                let f = self.function("gemm_i8_mn_splitk")?;
                let cfg = LaunchConfig {
                    grid_dim: ((grid_x * m) as u32, s as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut la = self.stream.launch_builder(&f);
                la.arg(&q);
                la.arg(&ascale);
                la.arg(b8);
                la.arg(wscale);
                la.arg(&cpart);
                la.arg(&k32);
                la.arg(&n32);
                let s32 = s as i32;
                la.arg(&s32);
                let m32i = m as i32;
                la.arg(&m32i);
                unsafe { la.launch(cfg) }.map_err(map_err)?;
                // Combine the S partials (stream-ordered after splitk).
                let f = self.function("gemm_i8_reduce")?;
                let cfg = LaunchConfig {
                    grid_dim: (((m * n + 255) / 256) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut la = self.stream.launch_builder(&f);
                la.arg(&cpart);
                la.arg(c);
                la.arg(&n32);
                la.arg(&s32);
                la.arg(&m32i);
                unsafe { la.launch(cfg) }.map_err(map_err)?;
                self.pool.lock().unwrap().entry(s * m * n).or_default().push(cpart);
            } else if m == 1 && !use_splitk {
                let f = self.function("gemm_i8_m1")?;
                let cfg = LaunchConfig {
                    grid_dim: (((n + 255) / 256) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut la = self.stream.launch_builder(&f);
                la.arg(&q);
                la.arg(&ascale);
                la.arg(b8);
                la.arg(wscale);
                la.arg(c);
                la.arg(&k32);
                la.arg(&n32);
                unsafe { la.launch(cfg) }.map_err(map_err)?;
            } else {
                let f = self.function("gemm_i8_tiled")?;
                let cfg = LaunchConfig {
                    grid_dim: (((n + 31) / 32) as u32, ((m + 31) / 32) as u32, 1),
                    block_dim: (16, 16, 1),
                    shared_mem_bytes: 0,
                };
                let mut la = self.stream.launch_builder(&f);
                la.arg(&q);
                la.arg(&ascale);
                la.arg(b8);
                la.arg(wscale);
                la.arg(c);
                la.arg(&m32);
                la.arg(&k32);
                la.arg(&n32);
                unsafe { la.launch(cfg) }.map_err(map_err)?;
            }
            // Return the quant scratch to the pools — stream-ordered: the
            // next quant/GEMM is queued after these kernels, so reuse is
            // safe without a sync.
            self.pool_i8.lock().unwrap().entry(q.len()).or_default().push(q);
            self.pool.lock().unwrap().entry(ascale.len()).or_default().push(ascale);
            return Ok(());
        }
        if let crate::backend::TensorInner::CudaBF16(b16) = &b.inner {
            let a16 = self.cast_bf16(a)?;
            let (handle, _raw_guard) = self.cublas_raw_guard()?;
            let alpha = 1.0f32;
            let beta = 0.0f32;
            let res = {
                let (bptr, _br) = b16.device_ptr(&self.stream);
                let (aptr, _ar) = a16.device_ptr(&self.stream);
                let (cptr, _cr) = c.as_cuda_mut().device_ptr(&self.stream);
                unsafe {
                    result::gemm_ex(
                        handle,
                        cublasOperation_t::CUBLAS_OP_T,
                        cublasOperation_t::CUBLAS_OP_N,
                        n as i32,
                        m as i32,
                        k as i32,
                        (&alpha as *const f32) as *const _,
                        bptr as *const _,
                        cudaDataType::CUDA_R_16BF,
                        k as i32,
                        aptr as *const _,
                        cudaDataType::CUDA_R_16BF,
                        k as i32,
                        (&beta as *const f32) as *const _,
                        cptr as *mut _,
                        cudaDataType::CUDA_R_32F,
                        n as i32,
                        cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                    )
                }
            };
            return res.map_err(|e| map_err(format!("cublasGemmEx(bf16) [{m}x{k}x{n}]: {e}")));
        }
        if let crate::backend::TensorInner::CudaF16(b16) = &b.inner {
            let a16 = self.cast_f16(a)?;
            let (handle, _raw_guard) = self.cublas_raw_guard()?;
            let alpha = 1.0f32;
            let beta = 0.0f32;
            let res = {
                let (bptr, _br) = b16.device_ptr(&self.stream);
                let (aptr, _ar) = a16.device_ptr(&self.stream);
                let (cptr, _cr) = c.as_cuda_mut().device_ptr(&self.stream);
                unsafe {
                    result::gemm_ex(
                        handle,
                        cublasOperation_t::CUBLAS_OP_T,
                        cublasOperation_t::CUBLAS_OP_N,
                        n as i32,
                        m as i32,
                        k as i32,
                        (&alpha as *const f32) as *const _,
                        bptr as *const _,
                        cudaDataType::CUDA_R_16F,
                        k as i32,
                        aptr as *const _,
                        cudaDataType::CUDA_R_16F,
                        k as i32,
                        (&beta as *const f32) as *const _,
                        cptr as *mut _,
                        cudaDataType::CUDA_R_32F,
                        n as i32,
                        cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                    )
                }
            };
            self.pool_f16.lock().unwrap().entry(a16.len()).or_default().push(a16);
            return res.map_err(|e| map_err(format!("cublasGemmEx(f16): {e}")));
        }
        // Row-major C[m,n] = A[m,k]·B[n,k]ᵀ. B read as column-major [k,n] is
        // Bᵀ → op(T) gives B back; A as column-major [k,m] is Aᵀ → op(N) keeps
        // Aᵀ: op = B·Aᵀ = (A·Bᵀ)ᵀ — row-major layout again.
        let cfg = GemmConfig::<f32> {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,
            n: m as i32,
            k: k as i32,
            alpha: 1.0,
            lda: k as i32,
            ldb: k as i32,
            beta: 0.0,
            ldc: n as i32,
        };
        if self.gemm_cublas(cfg, b, a, c)? {
            return Ok(());
        }
        let (m32, k32, n32) = (m as i32, k as i32, n as i32);
        let f = self.function("gemm_nt_naive_f32")?;
        let cfg = LaunchConfig::for_num_elems((m * n) as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(a.as_cuda());
        la.arg(b.as_cuda());
        la.arg(c.as_cuda());
        la.arg(&m32);
        la.arg(&k32);
        la.arg(&n32);
        unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
    }

    fn mul(&self, a: &Tensor, b: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        self.elementwise(ElemOp::Mul, a, Some(b), y, 0.0)
    }
    fn add(&self, a: &Tensor, b: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        self.elementwise(ElemOp::Add, a, Some(b), y, 0.0)
    }
    fn sub(&self, a: &Tensor, b: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        self.elementwise(ElemOp::Sub, a, Some(b), y, 0.0)
    }
    fn div(&self, a: &Tensor, b: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        self.elementwise(ElemOp::Div, a, Some(b), y, 0.0)
    }
    fn add_scalar(&self, x: &Tensor, y: &Tensor, c: f32) -> Result<(), BackendError> {
        self.elementwise(ElemOp::AddScalar, x, None, y, c)
    }
    fn scale(&self, x: &Tensor, y: &Tensor, c: f32) -> Result<(), BackendError> {
        self.elementwise(ElemOp::Scale, x, None, y, c)
    }
    fn add_bias(&self, x: &Tensor, bias: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        let n = x.numel();
        let cols = bias.numel() as i32;
        let n32 = n as i32;
        let f = self.function("add_bias_f32")?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(bias.as_cuda());
        la.arg(y.as_cuda());
        la.arg(&n32);
        la.arg(&cols);
        unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
    }
    fn sqrt(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        self.elementwise(ElemOp::Sqrt, x, None, y, 0.0)
    }
    fn log(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        self.elementwise(ElemOp::Log, x, None, y, 0.0)
    }
    fn max_scalar(&self, x: &Tensor, y: &Tensor, floor: f32) -> Result<(), BackendError> {
        self.elementwise(ElemOp::MaxScalar, x, None, y, floor)
    }

    fn relu(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        self.elementwise(ElemOp::Relu, x, None, y, 0.0)
    }
    fn leaky_relu_clip(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        self.elementwise(ElemOp::LeakyReluClip, x, None, y, 0.0)
    }
    fn sigmoid(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        self.elementwise(ElemOp::Sigmoid, x, None, y, 0.0)
    }
    fn tanh(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        self.elementwise(ElemOp::Tanh, x, None, y, 0.0)
    }
    fn silu(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        self.elementwise(ElemOp::Silu, x, None, y, 0.0)
    }
    fn gelu_tanh(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        self.elementwise(ElemOp::GeluTanh, x, None, y, 0.0)
    }
    fn snake(&self, x: &Tensor, a: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        // Per-channel (Metal semantics): A (C,), X flattened (n·c).
        let c = a.numel() as i32;
        let n = (x.numel() / c as usize) as i32;
        let f = self.function("snake_f32")?;
        let cfg = LaunchConfig::for_num_elems((n * c) as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(a.as_cuda());
        la.arg(y.as_cuda());
        let __a1 = n as i32;
        la.arg(&__a1);
        la.arg(&c);
        unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
    }

    fn rmsnorm(&self, x: &Tensor, w: &Tensor, y: &Tensor, eps: f32) -> Result<(), BackendError> {
        // Contract: X (rows, cols) — 2D (callers flatten (B,T) into rows).
        let rows = x.shape()[0];
        let cols = x.shape()[1] as i32;
        let f = self.function("rmsnorm_v2_f32")?;
        // One 256-thread block per row — block-reduce over cols.
        let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(w.as_cuda());
        la.arg(y.as_cuda());
        let __a2 = rows as i32;
        la.arg(&__a2);
        la.arg(&cols);
        la.arg(&eps);
        unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
    }
    fn layernorm(&self, x: &Tensor, w: &Tensor, b: &Tensor, y: &Tensor, eps: f32) -> Result<(), BackendError> {
        // Contract: X (rows, cols) — 2D (callers flatten (B,T) into rows).
        let rows = x.shape()[0];
        let cols = x.shape()[1] as i32;
        let f = self.function("layernorm_f32")?;
        // One block per row — the kernel indexes rows via blockIdx.x.
        let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 };
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(w.as_cuda());
        la.arg(b.as_cuda());
        la.arg(y.as_cuda());
        let __a3 = rows as i32;
        la.arg(&__a3);
        la.arg(&cols);
        la.arg(&eps);
        unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
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
        let (b, t, cin) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let (cout, _, k) = (w.shape()[0], w.shape()[1], w.shape()[2]);
        let tout = (t + pad_l + pad_r - dil * (k - 1) - 1) / stride + 1;
        let n = b * tout * cout;
        let f = self.function("conv1d_f32")?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(w.as_cuda());
        if let Some(bi) = bias {
            la.arg(bi.as_cuda());
        } else {
            la.arg(x.as_cuda()); // dummy — kernel guards via has_bias
        }
        la.arg(y.as_cuda());
        let b32 = b as i32;
        let t32 = t as i32;
        let cin32 = cin as i32;
        let cout32 = cout as i32;
        let cin_g32 = (cin / groups) as i32;
        let k32 = k as i32;
        let s32 = stride as i32;
        let pl32 = pad_l as i32;
        let pr32 = pad_r as i32;
        let dil32 = dil as i32;
        let tout32 = tout as i32;
        let has_bias = if bias.is_some() { 1i32 } else { 0i32 };
        let groups32 = groups as i32;
        la.arg(&b32);
        la.arg(&t32);
        la.arg(&cin32);
        la.arg(&cout32);
        la.arg(&cin_g32);
        la.arg(&k32);
        la.arg(&s32);
        la.arg(&pl32);
        la.arg(&pr32);
        la.arg(&dil32);
        la.arg(&tout32);
        la.arg(&has_bias);
        la.arg(&groups32);
        unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
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
        let (n, ci, h, wdim) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
        let (co, _, kh, kw) = (w.shape()[0], w.shape()[1], w.shape()[2], w.shape()[3]);
        // LEFT padding (Metal conv2d_f32 semantics) — symmetric padding here
        // would make the kernel write past the output buffer (oh too big).
        let (oh, ow) = ((h + pad_h - kh) / stride_t + 1, (wdim + pad_w - kw) / stride_f + 1);
        let n_elems = if time_major_out { n * oh * co * ow } else { n * co * oh * ow };
        let f = self.function("conv2d_nchw_f32")?;
        let cfg = LaunchConfig::for_num_elems(n_elems as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(w.as_cuda());
        if let Some(bi) = bias {
            la.arg(bi.as_cuda());
        } else {
            la.arg(x.as_cuda());
        }
        la.arg(y.as_cuda());
        let n32 = n as i32;
        let ci32 = ci as i32;
        let h32 = h as i32;
        let w32 = wdim as i32;
        let co32 = co as i32;
        let kh32 = kh as i32;
        let kw32 = kw as i32;
        let ph32 = pad_h as i32;
        let pw32 = pad_w as i32;
        let st32 = stride_t as i32;
        let sf32 = stride_f as i32;
        let oh32 = oh as i32;
        let ow32 = ow as i32;
        let has_bias = if bias.is_some() { 1i32 } else { 0i32 };
        let tmo = if time_major_out { 1i32 } else { 0i32 };
        la.arg(&n32);
        la.arg(&ci32);
        la.arg(&h32);
        la.arg(&w32);
        la.arg(&co32);
        la.arg(&kh32);
        la.arg(&kw32);
        la.arg(&ph32);
        la.arg(&pw32);
        la.arg(&st32);
        la.arg(&sf32);
        la.arg(&oh32);
        la.arg(&ow32);
        la.arg(&has_bias);
        la.arg(&tmo);
        unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
    }

    fn maxpool2d(&self, x: &Tensor, y: &Tensor, kh: usize, kw: usize) -> Result<(), BackendError> {
        let (n, c, h, wdim) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
        let (oh, ow) = (h / kh, wdim / kw);
        let n_elems = n * c * oh * ow;
        let f = self.function("maxpool2d_f32")?;
        let cfg = LaunchConfig::for_num_elems(n_elems as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(y.as_cuda());
        let __a24 = n as i32;
        la.arg(&__a24);
        let __a25 = c as i32;
        la.arg(&__a25);
        let __a26 = h as i32;
        la.arg(&__a26);
        let __a27 = wdim as i32;
        la.arg(&__a27);
        let __a28 = kh as i32;
        la.arg(&__a28);
        let __a29 = kw as i32;
        la.arg(&__a29);
        let __a30 = oh as i32;
        la.arg(&__a30);
        let __a31 = ow as i32;
        la.arg(&__a31);
        unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
    }

    fn conv_transpose1d(
        &self,
        x: &Tensor,
        w: &Tensor,
        bias: Option<&Tensor>,
        stride: usize,
        y: &Tensor,
    ) -> Result<(), BackendError> {
        let (b, t, cin) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let (_, cout, k) = (w.shape()[0], w.shape()[1], w.shape()[2]);
        let tout = (t - 1) * stride + k;
        // Zero the output in place first (scatter with atomics accumulates).
        self.stream.memset_zeros(y.as_cuda_mut()).map_err(map_err)?;
        let n = b * t * cin;
        let f = self.function("conv_transpose1d_f32")?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(w.as_cuda());
        la.arg(x.as_cuda()); // dummy — the kernel never reads bias (separate pass)
        la.arg(y.as_cuda());
        let __a32 = b as i32;
        la.arg(&__a32);
        let __a33 = t as i32;
        la.arg(&__a33);
        let __a34 = cin as i32;
        la.arg(&__a34);
        let __a35 = cout as i32;
        la.arg(&__a35);
        let __a36 = k as i32;
        la.arg(&__a36);
        let __a37 = stride as i32;
        la.arg(&__a37);
        let __a38 = tout as i32;
        la.arg(&__a38);
        unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)?;
        if let Some(bi) = bias {
            let nb = b * tout * cout;
            let f = self.function("conv_transpose_bias_f32")?;
            let cfg = LaunchConfig::for_num_elems(nb as u32);
            let mut la = self.stream.launch_builder(&f);
            la.arg(y.as_cuda());
            la.arg(bi.as_cuda());
            let __a39 = nb as i32;
        la.arg(&__a39);
            let __a40 = cout as i32;
        la.arg(&__a40);
            unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)?;
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
        let n = b * tout * cin * k;
        let f = self.function("im2col1d_f32")?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(xc.as_cuda());
        let __a = b as i32;
        la.arg(&__a);
        let __b = t as i32;
        la.arg(&__b);
        let __c = cin as i32;
        la.arg(&__c);
        let __d = k as i32;
        la.arg(&__d);
        let __e = stride as i32;
        la.arg(&__e);
        let __f = pad_l as i32;
        la.arg(&__f);
        let __g = dil as i32;
        la.arg(&__g);
        let __h = tout as i32;
        la.arg(&__h);
        unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
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
        let n = b * tout * cin * k;
        let f = self.function("im2col_t1d_f32")?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(xc.as_cuda());
        let __a = b as i32;
        la.arg(&__a);
        let __b = tin as i32;
        la.arg(&__b);
        let __c = cin as i32;
        la.arg(&__c);
        let __d = k as i32;
        la.arg(&__d);
        let __e = stride as i32;
        la.arg(&__e);
        let __f = tout as i32;
        la.arg(&__f);
        unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
    }

    fn slice_last(&self, x: &Tensor, y: &Tensor, off: usize) -> Result<(), BackendError> {
        // Contract: X (rows, D) → Y (rows, d0) — 2D, like Metal/CPU (callers
        // flatten (B,T) into rows via view).
        let rows = x.shape()[0];
        let d = x.shape()[1];
        let d0 = y.shape()[1];
        let n = rows * d0;
        let f = self.function("slice_last_f32")?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(y.as_cuda());
        let __a41 = rows as i32;
        la.arg(&__a41);
        let __a42 = d as i32;
        la.arg(&__a42);
        let __a43 = d0 as i32;
        la.arg(&__a43);
        let __a44 = off as i32;
        la.arg(&__a44);
        unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
    }

    fn slice_time(&self, x: &Tensor, y: &Tensor, u: usize) -> Result<(), BackendError> {
        let (b, t, c) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let keep = t - u;
        let n = b * keep * c;
        let f = self.function("slice_time_f32")?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(y.as_cuda());
        let __a45 = b as i32;
        la.arg(&__a45);
        let __a46 = t as i32;
        la.arg(&__a46);
        let __a47 = keep as i32;
        la.arg(&__a47);
        let __a48 = c as i32;
        la.arg(&__a48);
        unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
    }

    fn softmax_last_dim(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        let rows = x.shape()[0];
        let cols = x.shape()[1] as i32;
        let f = self.function("softmax_last_dim_f32")?;
        // One block per row — the kernel indexes rows via blockIdx.x.
        let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 };
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(y.as_cuda());
        let __a49 = rows as i32;
        la.arg(&__a49);
        la.arg(&cols);
        unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
    }

    fn embedding(&self, idx: &[u32], table: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        let n = idx.len();
        let d = table.shape()[1] as i32;
        let idx_dev = self.stream.clone_htod(idx).map_err(map_err)?;
        let f = self.function("embedding_f32")?;
        let cfg = LaunchConfig::for_num_elems((n * d as usize) as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(&idx_dev);
        la.arg(table.as_cuda());
        la.arg(y.as_cuda());
        let __a50 = n as i32;
        la.arg(&__a50);
        la.arg(&d);
        unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
    }

    fn rope(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, y: &Tensor, rope_dim: usize) -> Result<(), BackendError> {
        let (t, hd) = (x.shape()[0], x.shape()[1]);
        let n = t * hd;
        let f = self.function("rope_f32")?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(cos.as_cuda());
        la.arg(sin.as_cuda());
        la.arg(y.as_cuda());
        let __a51 = t as i32;
        la.arg(&__a51);
        let __a52 = hd as i32;
        la.arg(&__a52);
        let __a53 = rope_dim as i32;
        la.arg(&__a53);
        unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
    }

    fn rope_interleaved(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, y: &Tensor, rope_dim: usize, start: usize) -> Result<(), BackendError> {
        let (b, t, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let n = b * t * d;
        let f = self.function("rope_interleaved_f32")?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(cos.as_cuda());
        la.arg(sin.as_cuda());
        la.arg(y.as_cuda());
        let __a54 = b as i32;
        la.arg(&__a54);
        let __a55 = t as i32;
        la.arg(&__a55);
        let __a56 = d as i32;
        la.arg(&__a56);
        let __a57 = rope_dim as i32;
        la.arg(&__a57);
        let __a58 = start as i32;
        la.arg(&__a58);
        unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
    }

    fn rope_head(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, y: &Tensor, h: usize, rope_dim: usize) -> Result<(), BackendError> {
        let (b, t, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let hd = d / h;
        let n = b * t * d;
        let f = self.function("rope_head_f32")?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(cos.as_cuda());
        la.arg(sin.as_cuda());
        la.arg(y.as_cuda());
        let __a59 = b as i32;
        la.arg(&__a59);
        let __a60 = t as i32;
        la.arg(&__a60);
        let __a61 = h as i32;
        la.arg(&__a61);
        let __a62 = hd as i32;
        la.arg(&__a62);
        let __a63 = rope_dim as i32;
        la.arg(&__a63);
        unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
    }

    fn rmsnorm_head(&self, x: &Tensor, w: &Tensor, y: &Tensor, h: usize, eps: f32) -> Result<(), BackendError> {
        let rows = x.shape()[0];
        let hd = x.shape()[1] / h;
        let f = self.function("rmsnorm_head_f32")?;
        // One block per (row, head) — kernel indexes via blockIdx.(x, y).
        let cfg = LaunchConfig { grid_dim: (rows as u32, h as u32, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 };
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(w.as_cuda());
        la.arg(y.as_cuda());
        let __a64 = rows as i32;
        la.arg(&__a64);
        let __a65 = h as i32;
        la.arg(&__a65);
        let __a66 = hd as i32;
        la.arg(&__a66);
        la.arg(&eps);
        unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
    }

    fn layernorm_head(&self, x: &Tensor, w: &Tensor, b: &Tensor, y: &Tensor, h: usize, eps: f32) -> Result<(), BackendError> {
        let rows = x.shape()[0];
        let hd = x.shape()[1] / h;
        let f = self.function("layernorm_head_f32")?;
        // One block per (row, head) — kernel indexes via blockIdx.(x, y).
        let cfg = LaunchConfig { grid_dim: (rows as u32, h as u32, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 };
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(w.as_cuda());
        la.arg(b.as_cuda());
        la.arg(y.as_cuda());
        let __a67 = rows as i32;
        la.arg(&__a67);
        let __a68 = h as i32;
        la.arg(&__a68);
        let __a69 = hd as i32;
        la.arg(&__a69);
        la.arg(&eps);
        unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
    }

    fn mul_rows(&self, x: &Tensor, w: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        let (rows, d) = (x.shape()[0], x.shape()[1]);
        let n = rows * d;
        let f = self.function("mul_rows_f32")?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(w.as_cuda());
        la.arg(y.as_cuda());
        let __a70 = rows as i32;
        la.arg(&__a70);
        let __a71 = d as i32;
        la.arg(&__a71);
        unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
    }

    fn mul_vec(&self, x: &Tensor, w: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        let (rows, d) = (x.shape()[0], x.shape()[1]);
        let n = rows * d;
        let f = self.function("mul_vec_f32")?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(w.as_cuda());
        la.arg(y.as_cuda());
        let __a72 = rows as i32;
        la.arg(&__a72);
        let __a73 = d as i32;
        la.arg(&__a73);
        unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
    }

    fn glu(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        let (rows, d) = (x.shape()[0], x.shape()[1] / 2);
        let n = rows * d;
        let f = self.function("glu_f32")?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(y.as_cuda());
        let __a74 = rows as i32;
        la.arg(&__a74);
        let __a75 = d as i32;
        la.arg(&__a75);
        unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
    }

    fn repeat(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        let (b, t, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let t_out = y.shape()[1];
        let n = b * t * d;
        let f = self.function("repeat_f32")?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(y.as_cuda());
        let __a76 = b as i32;
        la.arg(&__a76);
        let __a77 = t as i32;
        la.arg(&__a77);
        let __a78 = t_out as i32;
        la.arg(&__a78);
        let __a79 = d as i32;
        la.arg(&__a79);
        unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
    }

    fn add_scaled(&self, a: &Tensor, b: &Tensor, y: &Tensor, scale: f32) -> Result<(), BackendError> {
        let n = a.numel();
        let f = self.function("add_scaled_f32")?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(a.as_cuda());
        la.arg(b.as_cuda());
        la.arg(y.as_cuda());
        let __a80 = n as i32;
        la.arg(&__a80);
        la.arg(&scale);
        unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
    }

    fn add_bias2d(&self, x: &Tensor, b: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        let n = x.numel();
        let f = self.function("add_bias2d_f32")?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(b.as_cuda());
        la.arg(y.as_cuda());
        let __a81 = n as i32;
        la.arg(&__a81);
        unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
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
        let (b, tq, _) = (q.shape()[0], q.shape()[1], q.shape()[2]);
        let tkv = k.shape()[1];
        let n_rep = n_heads / kv_heads;
        // Parallel kernel: one 256-thread block per (batch, head, query).
        let f = self.function("attn_ar_v2_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (tq as u32, n_heads as u32, b as u32),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut la = self.stream.launch_builder(&f);
        la.arg(q.as_cuda());
        la.arg(k.as_cuda());
        la.arg(v.as_cuda());
        if let Some(m) = mask {
            la.arg(m.as_cuda());
        } else {
            la.arg(q.as_cuda()); // dummy — kernel guards via has_mask
        }
        la.arg(y.as_cuda());
        let __a84 = b as i32;
        la.arg(&__a84);
        let __a85 = tq as i32;
        la.arg(&__a85);
        let __a86 = tkv as i32;
        la.arg(&__a86);
        let __a87 = n_heads as i32;
        la.arg(&__a87);
        let __a88 = kv_heads as i32;
        la.arg(&__a88);
        let __a89 = hd as i32;
        la.arg(&__a89);
        let __a90 = n_rep as i32;
        la.arg(&__a90);
        let has_mask = if mask.is_some() { 1i32 } else { 0i32 };
        la.arg(&has_mask);
        unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
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
        // mode 0 = leader (scores → SC + mask), 1 = non-leader (reuse SC),
        // 2 = uniform (mean V) — matches the Metal kernel.
        let (b, t, d) = (q.shape()[0], q.shape()[1], q.shape()[2]);
        let h = sc.shape()[1];
        let hd = d / h;
        let f = self.function("mhsa_f32")?;
        // 3D grid: (b, h, t) — one block per (batch, head, query).
        let cfg = LaunchConfig {
            grid_dim: (b as u32, h as u32, t as u32),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut la = self.stream.launch_builder(&f);
        la.arg(q.as_cuda());
        la.arg(k.as_cuda());
        la.arg(v.as_cuda());
        la.arg(sc.as_cuda());
        if let Some(m) = mask {
            la.arg(m.as_cuda());
        } else {
            la.arg(q.as_cuda()); // dummy — kernel guards via has_mask
        }
        la.arg(y.as_cuda());
        let __a91 = b as i32;
        la.arg(&__a91);
        let __a92 = t as i32;
        la.arg(&__a92);
        let __a93 = h as i32;
        la.arg(&__a93);
        let __a94 = hd as i32;
        la.arg(&__a94);
        let __a95 = mode as i32;
        la.arg(&__a95);
        let has_mask = if mask.is_some() { 1i32 } else { 0i32 };
        la.arg(&has_mask);
        unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
    }

    fn uniform_attn(&self, v: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        let (b, t, c) = (v.shape()[0], v.shape()[1], v.shape()[2]);
        let n = b * c;
        let f = self.function("uniform_attn_f32")?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(v.as_cuda());
        la.arg(y.as_cuda());
        let __a96 = b as i32;
        la.arg(&__a96);
        let __a97 = t as i32;
        la.arg(&__a97);
        let __a98 = c as i32;
        la.arg(&__a98);
        unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
    }

    fn kv_update(&self, dst: &Tensor, src: &Tensor, pos: usize) -> Result<(), BackendError> {
        let (b, tq, d) = (src.shape()[0], src.shape()[1], src.shape()[2]);
        let cap = dst.shape()[1];
        let n = b * tq * d;
        let f = self.function("kv_update_f32")?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(src.as_cuda());
        la.arg(dst.as_cuda());
        let __a99 = b as i32;
        la.arg(&__a99);
        let __a100 = tq as i32;
        la.arg(&__a100);
        let __a101 = pos as i32;
        la.arg(&__a101);
        let __a102 = d as i32;
        la.arg(&__a102);
        let __a103 = cap as i32;
        la.arg(&__a103);
        unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
    }

    fn codebook_search(&self, x: &Tensor, table: &Tensor, idx: &mut [u32]) -> Result<(), BackendError> {
        let rows = x.shape()[0] * x.shape()[1];
        let d = x.shape()[2];
        let v = table.shape()[0];
        let idx_dev = self.stream.alloc_zeros::<u32>(rows).map_err(map_err)?;
        let f = self.function("codebook_search_f32")?;
        // One block per row — the kernel indexes rows via blockIdx.x.
        let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 };
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(table.as_cuda());
        la.arg(&idx_dev);
        let __a104 = rows as i32;
        la.arg(&__a104);
        let __a105 = v as i32;
        la.arg(&__a105);
        let __a106 = d as i32;
        la.arg(&__a106);
        unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)?;
        let got = self.stream.clone_dtoh(&idx_dev).map_err(map_err)?;
        idx.copy_from_slice(&got);
        Ok(())
    }
}
