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

use cudarc::cublas::safe::GemmConfig;
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, safe::Gemm};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;

use crate::backend::{Backend, Tensor};
use crate::error::BackendError;

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
    GeluErf = 15,
    SwooshL = 16,
    SwooshR = 17,
}

/// The GEMM A-operand for the quantized-int8 kernels (the activation
/// dtype is configurable — fp32 stays f32, fp16/bf16 get a half-width cast).
enum KernA<'a> {
    F32(&'a Tensor),
    F16(cudarc::driver::CudaSlice<half::f16>),
    BF16(cudarc::driver::CudaSlice<half::bf16>),
}

fn map_err(e: impl std::fmt::Display) -> BackendError {
    let s = e.to_string();
    if s.to_lowercase().contains("out of memory") {
        BackendError::OutOfMemory(s)
    } else {
        BackendError::native(e)
    }
}

/// J7: never panic on a poisoned pool/cache mutex. A thread panicking while
/// it holds one of these locks (e.g. mid-GEMM in the TTS top-up thread)
/// poisons it; a second panic on the next take() would abort the whole
/// process (panic across the GStreamer FFI boundary). The guarded data is
/// always in a usable state — the lock is never held across a fallible
/// operation — so recovering the guard degrades the pool instead of killing
/// the bridge.
fn lock_poison_resilient<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
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
            let mut pool = lock_poison_resilient(pool);
            let bucket = pool.entry(s.len()).or_default();
            // Cap the pool per bucket AND by byte size: without the byte
            // cap, grown KV buffers (~16 MB+ per cache row) returned by
            // KvCache::shrink_to_init pile up in unique-size buckets and
            // never free (18 GB of dead KV after a few calls — the live
            // VRAM creep). Small hot buffers (activations, cpart) stay.
            const MAX_POOL_BYTES: usize = 8 * 1024 * 1024;
            const MAX_PER_BUCKET: usize = 8;
            if bucket.len() < MAX_PER_BUCKET && s.len() * 4 <= MAX_POOL_BYTES {
                bucket.push(s);
            }
            // else: drop — the cudaMalloc'd memory is released.
        }
    }
}

/// Pool-cap helper shared by the f32 / f16 / bf16 scratch pools (A1): same
/// per-bucket and byte limits as [`CudaBuffer`]'s Drop, parameterized by the
/// element size so a 2-byte type gets a byte-fair cap (32 MB of f16 elements
/// would otherwise be the f32-equivalent of 8 MB).
fn push_capped<T>(
    pool: &Arc<Mutex<HashMap<usize, Vec<cudarc::driver::CudaSlice<T>>>>>,
    sl: cudarc::driver::CudaSlice<T>,
    elem_bytes: usize,
) {
    const MAX_POOL_BYTES: usize = 8 * 1024 * 1024;
    const MAX_PER_BUCKET: usize = 8;
    let mut pool = lock_poison_resilient(pool);
    let bucket = pool.entry(sl.len()).or_default();
    if bucket.len() < MAX_PER_BUCKET && sl.len() * elem_bytes <= MAX_POOL_BYTES {
        bucket.push(sl);
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
    /// fp16 scratch pool (activations cast for fp16/bf16-weight GEMMs).
    pool_f16: Arc<Mutex<HashMap<usize, Vec<cudarc::driver::CudaSlice<half::f16>>>>>,
    /// bf16 scratch pool (activations cast for the bf16 activation axis).
    pool_bf16: Arc<Mutex<HashMap<usize, Vec<cudarc::driver::CudaSlice<half::bf16>>>>>,
    /// Activation compute dtype for GEMM A-operands (the plan's activation
    /// axis — `set_activation_dtype`; 0=f32, 1=f16, 2=bf16).
    activation: std::sync::atomic::AtomicU8,
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
            pool_bf16: Arc::new(Mutex::new(HashMap::new())),
            activation: std::sync::atomic::AtomicU8::new(0),
        })
    }

    fn activation_dtype(&self) -> crate::dtype::Dtype {
        match self.activation.load(std::sync::atomic::Ordering::Relaxed) {
            1 => crate::dtype::Dtype::F16,
            2 => crate::dtype::Dtype::BF16,
            _ => crate::dtype::Dtype::F32,
        }
    }

    fn take_pooled_f16(
        &self,
        n: usize,
    ) -> Result<cudarc::driver::CudaSlice<half::f16>, BackendError> {
        let mut pool = lock_poison_resilient(&self.pool_f16);
        if let Some(bucket) = pool.get_mut(&n)
            && let Some(s) = bucket.pop()
        {
            return Ok(s);
        }
        unsafe { self.stream.alloc::<half::f16>(n) }.map_err(map_err)
    }

    fn take_pooled_bf16(
        &self,
        n: usize,
    ) -> Result<cudarc::driver::CudaSlice<half::bf16>, BackendError> {
        let mut pool = lock_poison_resilient(&self.pool_bf16);
        if let Some(bucket) = pool.get_mut(&n)
            && let Some(s) = bucket.pop()
        {
            return Ok(s);
        }
        unsafe { self.stream.alloc::<half::bf16>(n) }.map_err(map_err)
    }

    /// Cast x (f32) into a bf16 scratch buffer of the same size (pooled).
    fn cast_bf16(&self, x: &Tensor) -> Result<cudarc::driver::CudaSlice<half::bf16>, BackendError> {
        let n = x.numel();
        let mut y = self.take_pooled_bf16(n)?;
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

    /// Split factor for the m=1 split-k: the LARGEST divisor of k/32 with
    /// at least 2 k-blocks per split (≤ nblk/2), partial buffer ≤16 MB and
    /// grid ≤8192 blocks. The power-of-two list left the w2 shape at 160
    /// blocks; the full-nblk grid (1 k-block per split) measured slightly
    /// worse — 2 k-blocks per split balances grid density vs per-block work.
    fn pick_split(&self, nblk: usize, m: usize, n: usize, grid_x: usize) -> usize {
        let cap_cpart = 16 * 1024 * 1024 / 4; // elements (f32)
        let cap_grid = 8192;
        (1..=nblk / 2)
            .rev()
            .find(|&s| {
                nblk.is_multiple_of(s) && s * m * n <= cap_cpart && grid_x * m * s <= cap_grid
            })
            .unwrap_or(1)
    }

    /// Number of z-slices the tiled kernels actually WRITE. The tiled GEMMs
    /// launch grid.z=16 slices; each covers nkz = ceil(nsup/16) k-blocks and
    /// exits without writing when kb0 >= nsup (k not divisible by 16·nkz).
    /// Reducing ALL 16 slices would add STALE POOL CONTENTS from an earlier
    /// GEMM that shared the cpart buffer (a different shape in the same
    /// bucket) — the classic pool-reuse garbage that broke every quantized
    /// STT engine (gigaam/t-one: k=768 → only 12 of 16 slices written).
    fn tiled_used_slices(k: usize, block_shift: u32) -> usize {
        let nsup = k >> block_shift;
        if nsup == 0 {
            // k < 256 (Q6_K/Q5_K/Q3_K) or k < 32 (Q4_0/halfw): no k-block at
            // all — the tiled kernels exit immediately for nsup==0, so the
            // reduce would fold a NEVER-WRITTEN (stale pool) slice. That
            // shape class is unreachable from from_cpu_q* (they enforce
            // k-block multiples), so this is only a guard against the
            // 0/0 panic in div_ceil below — it must NOT be treated as a
            // "reducing stale data is fine" license.
            return 1;
        }
        let nkz = nsup.div_ceil(16);
        nsup.div_ceil(nkz).max(1)
    }

    /// Combine the (S·m, n) partials into C (gemm_i8_reduce — fixed order,
    /// deterministic) and return the cpart buffer to the pool.
    fn reduce_and_pool(
        &self,
        cpart: CudaSlice<f32>,
        c: &mut CudaSlice<f32>,
        m: usize,
        n: usize,
        s: usize,
        // Named for the contract it documents: cpart was allocated with
        // exactly this many elements (take_pooled(pool_cap)), so push_capped
        // keys its bucket on cpart.len() == pool_cap — the ALLOCATION-size
        // key T5-8/R requires. Kept in the signature (read by none) so the
        // invariant stays explicit at every call site.
        _pool_cap: usize,
    ) -> Result<(), BackendError> {
        let n32 = n as i32;
        let m32 = m as i32;
        let s32 = s as i32;
        let f = self.function("gemm_i8_reduce")?;
        let cfg = LaunchConfig {
            grid_dim: ((m * n).div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut la = self.stream.launch_builder(&f);
        la.arg(&cpart);
        la.arg(c);
        la.arg(&n32);
        la.arg(&s32);
        la.arg(&m32);
        unsafe { la.launch(cfg) }.map_err(map_err)?;
        // J7: poison-resilient pool lock (a panic while the pool mutex was
        // held must not abort the next take()).
        // T5-8/R: the bucket key MUST be the cpart ALLOCATION size — the
        // tiled paths take 16·m·n (grid.z=16, only `used` slices reduced),
        // the m==1 splitk paths take s·m·n (s = split factor, usually < 16).
        // The wave-6 blanket 16·m·n key made every splitk buffer land in
        // the tiled bucket: the tiled take could then pop an s·m·n buffer
        // and write 16 slices into it (OOB), and every splitk call stranded
        // its buffer (the bucket grows unbounded).
        // A1: push THROUGH push_capped (same MAX_POOL_BYTES / MAX_PER_BUCKET
        // discipline as CudaBuffer::drop and pool_a_scratch) — the cpart of
        // a split-k GEMM with a varying m (STT prefill) would otherwise
        // strand one big buffer per unique shape forever. cpart came from
        // take_pooled(pool_cap), so cpart.len() == pool_cap and the bucket
        // key is unchanged.
        push_capped(&self.pool, cpart, std::mem::size_of::<f32>());
        Ok(())
    }

    /// Return the activation cast scratch (if any) to its pool, capped like
    /// the f32 pool (A1): same MAX_POOL_BYTES / MAX_PER_BUCKET limits, so
    /// TTS_ACTIVATION=fp16/bf16 cannot grow VRAM without bound.
    fn pool_a_scratch(&self, a_arg: KernA) {
        match a_arg {
            KernA::F16(sl) => {
                push_capped(&self.pool_f16, sl, std::mem::size_of::<half::f16>());
            }
            KernA::BF16(sl) => {
                push_capped(&self.pool_bf16, sl, std::mem::size_of::<half::bf16>());
            }
            KernA::F32(_) => {}
        }
    }

    /// Half-weight (fp16/bf16) GEMM on the own split-k kernels (see the
    /// DEFINE_HALFW_SPLITK comment). Split factor S: m=1 (decode) the
    /// largest divisor of k/32 within a 512-block grid (latency hiding);
    /// m>1 (prefill/fill/STT) the largest divisor of k/32 in [8,4,2,1] —
    /// more blocks than S=1, B re-reads bounded by the L2 (k-multiple
    /// traffic; measured vs the m=1-shaped fallback). k % 32 == 0.
    fn gemm_halfw_nt<T>(
        &self,
        a: &Tensor,
        b: &cudarc::driver::CudaSlice<T>,
        c: &Tensor,
        m: usize,
        k: usize,
        n: usize,
        b_suf: &'static str,
    ) -> Result<(), BackendError> {
        if !k.is_multiple_of(32) {
            // Shapes like the DAC codec's (1, 8, 1024) carry raw row-major
            // half weights (k%32!=0 tensors are not kb-major packed) — the
            // naive elementwise kernel covers any k (the split-k/tiled paths
            // require k ≡ 0 mod 32; before this fallback they computed a
            // silent zero for k<32).
            let fname: &'static str = match b_suf {
                "bf16" => "gemm_halfw_naive_bf16",
                "f16" => "gemm_halfw_naive_f16",
                _ => unreachable!(),
            };
            let f = self.function(fname)?;
            let cfg = LaunchConfig::for_num_elems((m * n) as u32);
            let mut la = self.stream.launch_builder(&f);
            la.arg(a.as_cuda());
            la.arg(b);
            la.arg(c.as_cuda_mut());
            let m32 = m as i32;
            let k32 = k as i32;
            let n32 = n as i32;
            la.arg(&m32);
            la.arg(&k32);
            la.arg(&n32);
            unsafe { la.launch(cfg) }.map_err(map_err)?;
            return Ok(());
        }
        let k32 = k as i32;
        let n32 = n as i32;
        let m32 = m as i32;
        let c = c.as_cuda_mut();
        let grid_x = n.div_ceil(256);
        let act = self.activation_dtype();
        let (tiled_name, splitk_name): (&'static str, &'static str) = match (b_suf, act) {
            ("bf16", crate::dtype::Dtype::F16) => {
                ("gemm_halfw_tiled_f16_bf16", "gemm_halfw_splitk_f16_bf16")
            }
            ("bf16", crate::dtype::Dtype::BF16) => {
                ("gemm_halfw_tiled_bf16_bf16", "gemm_halfw_splitk_bf16_bf16")
            }
            ("bf16", _) => ("gemm_halfw_tiled_f32_bf16", "gemm_halfw_splitk_f32_bf16"),
            ("f16", crate::dtype::Dtype::F16) => {
                ("gemm_halfw_tiled_f16_f16", "gemm_halfw_splitk_f16_f16")
            }
            ("f16", crate::dtype::Dtype::BF16) => {
                ("gemm_halfw_tiled_bf16_f16", "gemm_halfw_splitk_bf16_f16")
            }
            ("f16", _) => ("gemm_halfw_tiled_f32_f16", "gemm_halfw_splitk_f32_f16"),
            _ => unreachable!("gemm_halfw_nt: unknown weight suffix {b_suf}"),
        };
        let a_arg: KernA = match act {
            crate::dtype::Dtype::F16 => KernA::F16(self.cast_f16(a)?),
            crate::dtype::Dtype::BF16 => KernA::BF16(self.cast_bf16(a)?),
            _ => KernA::F32(a),
        };
        if m > 1 {
            // m>1: the tiled kernel — one block per (32 rows × 256 cols),
            // grid.z=16 splits k into (16·m, n) partials combined by
            // gemm_i8_reduce (deterministic fixed-order reduction).
            // NOTE: the mma.sync tensor-core path was removed after the
            // sm_120 investigation (2026-08-25): the ptxas miscompiles the
            // fragment loads (SASS: LDS.U16 at 0x4040 with 0x800 shared —
            // out-of-bounds reads), the wmma API is gone from CUDA 13, and
            // tcgen05 is datacenter-only (sm_100a). The f32-accumulating
            // tiled kernel is the production path — it beats cuBLAS on all
            // measured shapes (1.06–3.4×).
            let s16 = 16usize;
            let cpart = self.take_pooled(s16 * m * n)?;
            let f = self.function(tiled_name)?;
            let cfg = LaunchConfig {
                grid_dim: (grid_x as u32, m.div_ceil(32) as u32, 16),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut la = self.stream.launch_builder(&f);
            match &a_arg {
                KernA::F32(t) => la.arg(t.as_cuda()),
                KernA::F16(sl) => la.arg(sl),
                KernA::BF16(sl) => la.arg(sl),
            };
            la.arg(b);
            la.arg(&cpart);
            la.arg(&m32);
            la.arg(&k32);
            la.arg(&n32);
            unsafe { la.launch(cfg) }.map_err(map_err)?;
            let used = Self::tiled_used_slices(k, 5);
            self.reduce_and_pool(cpart, c, m, n, used, 16 * m * n)?;
            self.pool_a_scratch(a_arg);
            return Ok(());
        }
        let nblk = k / 32;
        // Half kernel: power-of-two split list — the measured optimum
        // (w2: 160 blocks beats the 3040-block grid; 4 uint4 per k-block
        // already hides the latency).
        let s = [32usize, 16, 8, 4, 2, 1]
            .into_iter()
            .find(|&s| nblk.is_multiple_of(s))
            .unwrap_or(1);
        let cpart = self.take_pooled(s * m * n)?;
        let f = self.function(splitk_name)?;
        let cfg = LaunchConfig {
            grid_dim: ((grid_x * m) as u32, s as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut la = self.stream.launch_builder(&f);
        match &a_arg {
            KernA::F32(t) => la.arg(t.as_cuda()),
            KernA::F16(sl) => la.arg(sl),
            KernA::BF16(sl) => la.arg(sl),
        };
        la.arg(b);
        la.arg(&cpart);
        la.arg(&k32);
        la.arg(&n32);
        let s32 = s as i32;
        la.arg(&s32);
        la.arg(&m32);
        unsafe { la.launch(cfg) }.map_err(map_err)?;
        // Combine the S partials (stream-ordered after splitk).
        self.reduce_and_pool(cpart, c, m, n, s, s * m * n)?;
        self.pool_a_scratch(a_arg);
        Ok(())
    }

    /// cuBLAS GEMM (row-major semantics, see the two call sites). The handle
    /// is lazily created under the same mutex that serializes every call —
    /// cuBLAS handles are NOT thread-safe, and the TTS pool can drive GEMMs
    /// from several threads at once (generation + top-up).
    /// `CUDA_NO_CUBLAS=1` forces the naive PTX path (benchmarks, debug).
    /// `x`/`y` are passed in cuBLAS argument order (first/second matrix).
    fn gemm_cublas(
        &self,
        cfg: GemmConfig<f32>,
        x: &Tensor,
        y: &Tensor,
        c: &Tensor,
    ) -> Result<bool, BackendError> {
        if std::env::var_os("CUDA_NO_CUBLAS").is_some() {
            return Ok(false);
        }
        // Reject aliased operands BEFORE creating the &mut below: C = A·B with
        // C sharing storage with A or B is not a valid GEMM, and the aliased
        // mutable reference would be undefined behaviour.
        if c.shares_buffer_with(x) || c.shares_buffer_with(y) {
            return Err(BackendError::Shape(
                "gemm: output buffer aliases an input; use a distinct output tensor".into(),
            ));
        }
        let mut guard = lock_poison_resilient(&self.cublas);
        if guard.is_none() {
            match CudaBlas::new(self.stream.clone()) {
                Ok(b) => *guard = Some(Arc::new(b)),
                Err(e) => {
                    eprintln!(
                        "[cuda] cuBLAS unavailable, GEMM falls back to naive PTX kernel: {e}"
                    );
                    return Ok(false);
                }
            }
        }
        // J2/C8: the handle MUST stay locked for the GEMM — cuBLAS handles
        // are not thread-safe. The old code cloned the Arc, dropped the
        // guard and called gemm() unlocked: the TTS top-up thread and the
        // generation thread raced the same handle (silent corruption under
        // concurrent calls). Serializing here is correct; per-thread handles
        // are the future optimization.
        let blas = guard.as_ref().unwrap();
        unsafe { blas.gemm(cfg, x.as_cuda(), y.as_cuda(), c.as_cuda_mut()) }
            .map_err(|e| map_err(format!("cublasSgemm: {e}")))?;
        Ok(true)
    }

    fn wrap_pooled(&self, slice: CudaSlice<f32>) -> Arc<CudaBuffer> {
        Arc::new(CudaBuffer {
            slice: Some(slice),
            pool: Some(self.pool.clone()),
        })
    }

    /// Block until all queued work on the backend stream completes.
    pub fn synchronize(&self) -> Result<(), BackendError> {
        self.stream.synchronize().map_err(map_err)
    }

    fn take_pooled(&self, n: usize) -> Result<CudaSlice<f32>, BackendError> {
        let mut pool = lock_poison_resilient(&self.pool);
        if let Some(bucket) = pool.get_mut(&n)
            && let Some(s) = bucket.pop()
        {
            return Ok(s);
        }
        self.stream.alloc_zeros::<f32>(n).map_err(map_err)
    }

    fn function(&self, name: &'static str) -> Result<CudaFunction, BackendError> {
        let mut cache = lock_poison_resilient(&self.functions);
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
        // Length guards: the kernel indexes `x[i]`, `y[i]` and writes `out[i]`
        // for i in 0..n, but `n` is taken from `x` alone. A short `out` (or a
        // short `y`) therefore produced a device out-of-bounds write/read —
        // reachable through the safe public API, e.g. scale(&x[4096], &out[1]).
        // Sizes must agree exactly; aliasing `out` with `x` stays allowed.
        if out.numel() != n {
            return Err(BackendError::Shape(format!(
                "elementwise: output has {} elements, input has {n}",
                out.numel()
            )));
        }
        if let Some(y) = y
            && y.numel() != n
        {
            return Err(BackendError::Shape(format!(
                "elementwise: second operand has {} elements, first has {n}",
                y.numel()
            )));
        }
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
    fn set_activation_dtype(&self, d: crate::dtype::Dtype) -> Result<(), BackendError> {
        self.activation.store(
            match d {
                crate::dtype::Dtype::F16 => 1,
                crate::dtype::Dtype::BF16 => 2,
                _ => 0,
            },
            std::sync::atomic::Ordering::Relaxed,
        );
        Ok(())
    }

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

    /// N (iter-10): GGUF Q6_K/Q4_K tensors now upload PACKED (kb-major, one
    /// host-side block transpose — no dequant) and run through the fused
    /// dequant-in-kernel GEMMs instead of the trait default (dequant→f32:
    /// 32 B/weight VRAM and DRAM traffic vs 6.56/4.5 bits — ~5-7x). Q4_0 and
    /// Q8_0 keep the f32 path (the speech-dialect pairing / scale-format
    /// mismatches would silently misdecode; documented).
    fn from_quantized(
        &self,
        data: &[u8],
        shape: Vec<usize>,
        format: crate::dtype::WeightFormat,
    ) -> Result<Tensor, BackendError> {
        let blk_bytes = match format {
            crate::dtype::WeightFormat::Q6_K => 210usize,
            crate::dtype::WeightFormat::Q4_K => 144usize,
            // GGUF Q8_0: 34 Б/32 (f16 scale + коды) → int8-ядро (коды + f32
            // scale). Раньше падал в f32-fallback: 4 байта/вес ломали
            // загрузку 27B Q4_K_M (7.2 ГБ Q8_0 → 28.6 ГБ VRAM, CUDA OOM).
            crate::dtype::WeightFormat::Q8_0 => {
                let n: usize = shape.iter().product();
                match crate::weights_map::q8_0_to_i8(data, n) {
                    Some((codes, scales)) => {
                        return self.from_cpu_i8(&codes, &scales, shape);
                    }
                    None => {
                        let tmp = crate::weights_map::WeightTensor {
                            name: "q".into(),
                            shape: shape.iter().map(|&d| d as u64).collect(),
                            format,
                            offset: 0,
                            data,
                        };
                        let f = crate::weights_map::dequant_f32(&tmp);
                        return self.from_cpu_f32(&f, shape);
                    }
                }
            }
            _ => {
                // Fall back to the reference path (dequant → f32).
                let tmp = crate::weights_map::WeightTensor {
                    name: "q".into(),
                    shape: shape.iter().map(|&d| d as u64).collect(),
                    format,
                    offset: 0,
                    data,
                };
                if matches!(
                    format,
                    crate::dtype::WeightFormat::Int4 | crate::dtype::WeightFormat::Nf4
                ) {
                    return Err(BackendError::UnsupportedFormat(format!(
                        "from_quantized: {format:?} dequant not implemented"
                    )));
                }
                let f = crate::weights_map::dequant_f32(&tmp);
                return self.from_cpu_f32(&f, shape);
            }
        };
        // 2D GEMM weights (incl. fused qkv/w13 — the concatenated sub-tensors
        // are row-major over the fused rows, so a single transpose holds).
        if shape.len() != 2 {
            return Err(BackendError::Shape(format!(
                "from_quantized packed: shape {shape:?} must be 2D"
            )));
        }
        let (rows, cols) = (shape[0], shape[1]);
        if cols % 256 != 0 {
            // Not super-block aligned — dequant fallback.
            let tmp = crate::weights_map::WeightTensor {
                name: "q".into(),
                shape: shape.iter().map(|&d| d as u64).collect(),
                format,
                offset: 0,
                data,
            };
            let f = crate::weights_map::dequant_f32(&tmp);
            return self.from_cpu_f32(&f, shape);
        }
        let nsuper = cols / 256;
        let expect = rows * nsuper * blk_bytes;
        if data.len() != expect {
            return Err(BackendError::Shape(format!(
                "from_quantized: {} bytes for shape {shape:?} (expected {expect})",
                data.len()
            )));
        }
        // GGUF row-major super-blocks (r·nsuper + kb) → kb-major
        // (kb·rows + r) — the packed kernels' layout.
        let mut kb_major = vec![0u8; data.len()];
        for r in 0..rows {
            for kb in 0..nsuper {
                let src = (r * nsuper + kb) * blk_bytes;
                let dst = (kb * rows + r) * blk_bytes;
                kb_major[dst..dst + blk_bytes].copy_from_slice(&data[src..src + blk_bytes]);
            }
        }
        match format {
            crate::dtype::WeightFormat::Q6_K => self.from_cpu_q6k(&kb_major, shape),
            crate::dtype::WeightFormat::Q4_K => self.from_cpu_q4kc(&kb_major, shape),
            _ => unreachable!(),
        }
    }

    fn from_cpu_f16(&self, data: &[half::f16], shape: Vec<usize>) -> Result<Tensor, BackendError> {
        let n: usize = shape.iter().product();
        if data.len() != n {
            return Err(BackendError::Shape(format!(
                "from_cpu_f16: {} elements for shape {shape:?}",
                data.len()
            )));
        }
        // Half WEIGHTS are stored kb-major — block (kb, col) = 32 contiguous
        // halves at (kb·rows + col)·32, the same packing as the int8 path:
        // 4 neighbouring columns share one 128 B L1 line per k-block, while
        // row-major pulls a separate line per column (measured 30-50% DRAM/
        // L2 efficiency on the big lmh/w12 forms vs 85%+ kb-major).
        // Non-2D tensors and k%32!=0 shapes (biases, non-GEMM uses) stay
        // raw row-major — consumers must agree on the layout.
        let packed: std::borrow::Cow<[half::f16]> =
            if shape.len() == 2 && n > 0 && shape[1].is_multiple_of(32) {
                let (rows, per) = (shape[0], shape[1]);
                let nblk = per / 32;
                let mut p = vec![half::f16::default(); n];
                for r in 0..rows {
                    for b in 0..nblk {
                        for j in 0..32 {
                            p[(b * rows + r) * 32 + j] = data[r * per + b * 32 + j];
                        }
                    }
                }
                std::borrow::Cow::Owned(p)
            } else {
                std::borrow::Cow::Borrowed(data)
            };
        let mut s = unsafe { self.stream.alloc::<half::f16>(n) }.map_err(map_err)?;
        self.stream
            .memcpy_htod(packed.as_ref(), &mut s)
            .map_err(map_err)?;
        Ok(Tensor::from_cuda_f16(s, shape))
    }

    fn from_cpu_bf16(
        &self,
        data: &[half::bf16],
        shape: Vec<usize>,
    ) -> Result<Tensor, BackendError> {
        let n: usize = shape.iter().product();
        if data.len() != n {
            return Err(BackendError::Shape(format!(
                "from_cpu_bf16: {} elements for shape {shape:?}",
                data.len()
            )));
        }
        let packed: std::borrow::Cow<[half::bf16]> =
            if shape.len() == 2 && n > 0 && shape[1].is_multiple_of(32) {
                let (rows, per) = (shape[0], shape[1]);
                let nblk = per / 32;
                let mut p = vec![half::bf16::default(); n];
                for r in 0..rows {
                    for b in 0..nblk {
                        for j in 0..32 {
                            p[(b * rows + r) * 32 + j] = data[r * per + b * 32 + j];
                        }
                    }
                }
                std::borrow::Cow::Owned(p)
            } else {
                std::borrow::Cow::Borrowed(data)
            };
        let mut s = unsafe { self.stream.alloc::<half::bf16>(n) }.map_err(map_err)?;
        self.stream
            .memcpy_htod(packed.as_ref(), &mut s)
            .map_err(map_err)?;
        Ok(Tensor::from_cuda_bf16(s, shape))
    }

    fn from_cpu_i8(
        &self,
        data: &[i8],
        scale: &[f32],
        shape: Vec<usize>,
    ) -> Result<Tensor, BackendError> {
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

    fn from_cpu_q6k(&self, data: &[u8], shape: Vec<usize>) -> Result<Tensor, BackendError> {
        let per: usize = shape[1..].iter().product();
        if !per.is_multiple_of(256) {
            return Err(BackendError::Shape(format!(
                "from_cpu_q6k: per-dim {per} not a multiple of 256 (Q6_K super-blocks)"
            )));
        }
        let expect = shape[0] * (per / 256) * 210;
        if data.len() != expect {
            return Err(BackendError::Shape(format!(
                "from_cpu_q6k: {} packed bytes for shape {shape:?} (expected {expect})",
                data.len()
            )));
        }
        let mut s = unsafe { self.stream.alloc::<u8>(data.len()) }.map_err(map_err)?;
        self.stream.memcpy_htod(data, &mut s).map_err(map_err)?;
        Ok(Tensor::from_cuda_q6k(s, shape))
    }

    fn from_cpu_q4k(&self, data: &[u8], shape: Vec<usize>) -> Result<Tensor, BackendError> {
        let per: usize = shape[1..].iter().product();
        if !per.is_multiple_of(32) {
            return Err(BackendError::Shape(format!(
                "from_cpu_q4k: per-dim {per} not a multiple of 32 (Q4_0 blocks)"
            )));
        }
        let expect = shape[0] * (per / 32) * 18;
        if data.len() != expect {
            return Err(BackendError::Shape(format!(
                "from_cpu_q4k: {} packed bytes for shape {shape:?} (expected {expect})",
                data.len()
            )));
        }
        let mut s = unsafe { self.stream.alloc::<u8>(data.len()) }.map_err(map_err)?;
        self.stream.memcpy_htod(data, &mut s).map_err(map_err)?;
        Ok(Tensor::from_cuda_q4k(s, shape))
    }

    fn from_cpu_q4kc(&self, data: &[u8], shape: Vec<usize>) -> Result<Tensor, BackendError> {
        let per: usize = shape[1..].iter().product();
        if !per.is_multiple_of(256) {
            return Err(BackendError::Shape(format!(
                "from_cpu_q4kc: per-dim {per} not a multiple of 256 (Q4_K super-blocks)"
            )));
        }
        let expect = shape[0] * (per / 256) * 144;
        if data.len() != expect {
            return Err(BackendError::Shape(format!(
                "from_cpu_q4kc: {} packed bytes for shape {shape:?} (expected {expect})",
                data.len()
            )));
        }
        let mut s = unsafe { self.stream.alloc::<u8>(data.len()) }.map_err(map_err)?;
        self.stream.memcpy_htod(data, &mut s).map_err(map_err)?;
        Ok(Tensor::from_cuda_q4kc(s, shape))
    }

    fn from_cpu_q5k(&self, data: &[u8], shape: Vec<usize>) -> Result<Tensor, BackendError> {
        let per: usize = shape[1..].iter().product();
        if !per.is_multiple_of(256) {
            return Err(BackendError::Shape(format!(
                "from_cpu_q5k: per-dim {per} not a multiple of 256 (Q5_K super-blocks)"
            )));
        }
        let expect = shape[0] * (per / 256) * 176;
        if data.len() != expect {
            return Err(BackendError::Shape(format!(
                "from_cpu_q5k: {} packed bytes for shape {shape:?} (expected {expect})",
                data.len()
            )));
        }
        let mut s = unsafe { self.stream.alloc::<u8>(data.len()) }.map_err(map_err)?;
        self.stream.memcpy_htod(data, &mut s).map_err(map_err)?;
        Ok(Tensor::from_cuda_q5k(s, shape))
    }

    fn from_cpu_q3k(&self, data: &[u8], shape: Vec<usize>) -> Result<Tensor, BackendError> {
        let per: usize = shape[1..].iter().product();
        if !per.is_multiple_of(256) {
            return Err(BackendError::Shape(format!(
                "from_cpu_q3k: per-dim {per} not a multiple of 256 (Q3_K super-blocks)"
            )));
        }
        let expect = shape[0] * (per / 256) * 110;
        if data.len() != expect {
            return Err(BackendError::Shape(format!(
                "from_cpu_q3k: {} packed bytes for shape {shape:?} (expected {expect})",
                data.len()
            )));
        }
        let mut s = unsafe { self.stream.alloc::<u8>(data.len()) }.map_err(map_err)?;
        self.stream.memcpy_htod(data, &mut s).map_err(map_err)?;
        Ok(Tensor::from_cuda_q3k(s, shape))
    }

    /// fp16 storage download (f16 KV caches, half weights). Signed-off
    /// counterpart of `to_cpu_f32`: the roll/parity diagnostics download KV in
    /// its storage dtype instead of widening, so a real f16 cache can be
    /// compared bit-for-bit with the reference.
    fn to_cpu_f16(&self, t: &Tensor) -> Result<Vec<half::f16>, BackendError> {
        if let crate::backend::TensorInner::CudaF16(b) = &t.inner {
            return self.stream.clone_dtoh(b).map_err(map_err);
        }
        let f = self.to_cpu_f32(t)?;
        Ok(f.into_iter().map(half::f16::from_f32).collect())
    }

    fn to_cpu_f32(&self, t: &Tensor) -> Result<Vec<f32>, BackendError> {
        // Half-precision storage (f16/bf16 KV, fp16 weights) downloads the
        // raw bits and widens on the CPU — same result as a device-side cast.
        // Half weights are kb-major (from_cpu_*) — un-packed back to the
        // standard row-major layout here. q8_0 (CudaI8) dequantizes with
        // its per-32-block scales.
        match &t.inner {
            crate::backend::TensorInner::CudaF16(b) => {
                let raw: Vec<half::f16> = self.stream.clone_dtoh(b).map_err(map_err)?;
                let shape = t.shape();
                if shape.len() == 2 && !raw.is_empty() && shape[1].is_multiple_of(32) {
                    let (rows, per) = (shape[0], shape[1]);
                    let nblk = per / 32;
                    let mut out = vec![0f32; raw.len()];
                    for r in 0..rows {
                        for b in 0..nblk {
                            for j in 0..32 {
                                out[r * per + b * 32 + j] = f32::from(raw[(b * rows + r) * 32 + j]);
                            }
                        }
                    }
                    Ok(out)
                } else {
                    Ok(raw.into_iter().map(f32::from).collect())
                }
            }
            crate::backend::TensorInner::CudaBF16(b) => {
                let raw: Vec<half::bf16> = self.stream.clone_dtoh(b).map_err(map_err)?;
                let shape = t.shape();
                if shape.len() == 2 && !raw.is_empty() && shape[1].is_multiple_of(32) {
                    let (rows, per) = (shape[0], shape[1]);
                    let nblk = per / 32;
                    let mut out = vec![0f32; raw.len()];
                    for r in 0..rows {
                        for b in 0..nblk {
                            for j in 0..32 {
                                out[r * per + b * 32 + j] = f32::from(raw[(b * rows + r) * 32 + j]);
                            }
                        }
                    }
                    Ok(out)
                } else {
                    Ok(raw.into_iter().map(f32::from).collect())
                }
            }
            crate::backend::TensorInner::CudaI8 { data, scale } => {
                let raw: Vec<i8> = self.stream.clone_dtoh(data).map_err(map_err)?;
                let sc: Vec<f32> = self.stream.clone_dtoh(scale).map_err(map_err)?;
                // kb-major packing: block (kb, col) = 32 i8 at
                // (kb·rows + col)·32, scale at (kb·rows + col).
                let rows = t.shape()[0];
                let per: usize = t.shape()[1..].iter().product();
                let nblk = per / 32;
                let mut out = vec![0f32; rows * per];
                for r in 0..rows {
                    for b in 0..nblk {
                        let s = sc[b * rows + r];
                        let blk = &raw[(b * rows + r) * 32..(b * rows + r + 1) * 32];
                        for j in 0..32 {
                            out[r * per + b * 32 + j] = f32::from(blk[j]) * s;
                        }
                    }
                }
                Ok(out)
            }
            crate::backend::TensorInner::CudaQ6K(data) => {
                let raw: Vec<u8> = self.stream.clone_dtoh(data).map_err(map_err)?;
                let per: usize = t.shape()[1..].iter().product();
                let rows = t.shape()[0];
                let nsuper = per / 256;
                let mut out = vec![0f32; rows * per];
                for r in 0..rows {
                    for sb in 0..nsuper {
                        // kb-major: block (sb, r) at (sb·rows + r)·210.
                        let blk = &raw[(sb * rows + r) * 210..(sb * rows + r + 1) * 210];
                        let d = f32::from(half::f16::from_bits(u16::from_le_bytes([
                            blk[208], blk[209],
                        ])));
                        for nchunk in 0..2 {
                            for lane in 0..32 {
                                let ql0 = blk[nchunk * 64 + lane];
                                let ql1 = blk[nchunk * 64 + lane + 32];
                                let qh = blk[128 + nchunk * 32 + lane];
                                let base = r * per + sb * 256 + nchunk * 128 + lane;
                                let sc0 = blk[192 + nchunk * 8 + (lane >> 4)] as i8 as f32;
                                let sc1 = blk[192 + nchunk * 8 + (lane >> 4) + 2] as i8 as f32;
                                let sc2 = blk[192 + nchunk * 8 + (lane >> 4) + 4] as i8 as f32;
                                let sc3 = blk[192 + nchunk * 8 + (lane >> 4) + 6] as i8 as f32;
                                let q1 = (((ql0 & 0xF) | ((qh & 3) << 4)) as i32 - 32) as f32;
                                let q2 =
                                    (((ql1 & 0xF) | (((qh >> 2) & 3) << 4)) as i32 - 32) as f32;
                                let q3 = ((((ql0 >> 4) & 0xF) | (((qh >> 4) & 3) << 4)) as i32 - 32)
                                    as f32;
                                let q4 = ((((ql1 >> 4) & 0xF) | (((qh >> 6) & 3) << 4)) as i32 - 32)
                                    as f32;
                                out[base] = d * sc0 * q1;
                                out[base + 32] = d * sc1 * q2;
                                out[base + 64] = d * sc2 * q3;
                                out[base + 96] = d * sc3 * q4;
                            }
                        }
                    }
                }
                Ok(out)
            }
            crate::backend::TensorInner::CudaQ5K(data) => {
                let raw: Vec<u8> = self.stream.clone_dtoh(data).map_err(map_err)?;
                let per: usize = t.shape()[1..].iter().product();
                let rows = t.shape()[0];
                let nsuper = per / 256;
                let mut out = vec![0f32; rows * per];
                for r in 0..rows {
                    for sb in 0..nsuper {
                        // kb-major: block (sb, r) at (sb·rows + r)·176.
                        let blk = &raw[(sb * rows + r) * 176..(sb * rows + r + 1) * 176];
                        let d =
                            f32::from(half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])));
                        let mn =
                            f32::from(half::f16::from_bits(u16::from_le_bytes([blk[2], blk[3]])));
                        for j in 0..256 {
                            let b = j >> 5;
                            let l = j & 31;
                            // Per-32-block scales — the odd block of a pair
                            // has its OWN scale (kernel parity; the old
                            // (b>>1)*2 applied the even block's scale to
                            // both halves).
                            let is = b;
                            let (sc, m) = if is < 4 {
                                (blk[4 + is] & 63, blk[4 + is + 4] & 63)
                            } else {
                                (
                                    (blk[4 + is + 4] & 0xF) | ((blk[4 + is - 4] >> 6) << 4),
                                    (blk[4 + is + 4] >> 4) | ((blk[4 + is] >> 6) << 4),
                                )
                            };
                            let dl = d * sc as f32;
                            let ml = mn * m as f32;
                            let half = b & 1;
                            // qs is chunk-major (16 + 32·c + l); the old
                            // `blk[16 + l]` read chunk 0's bytes for blocks
                            // 2..7 (kernel-parity fix).
                            let qsb = 32 * (b >> 1);
                            let mut q = if half == 0 {
                                blk[16 + qsb + l] & 0xF
                            } else {
                                blk[16 + qsb + l] >> 4
                            };
                            let u = 1 << (2 * (b >> 1) + half);
                            if blk[144 + l] & u != 0 {
                                q += 16;
                            }
                            out[r * per + sb * 256 + j] = dl * q as f32 - ml;
                        }
                    }
                }
                Ok(out)
            }
            crate::backend::TensorInner::CudaQ3K(data) => {
                let raw: Vec<u8> = self.stream.clone_dtoh(data).map_err(map_err)?;
                let per: usize = t.shape()[1..].iter().product();
                let rows = t.shape()[0];
                let nsuper = per / 256;
                let mut out = vec![0f32; rows * per];
                for r in 0..rows {
                    for sb in 0..nsuper {
                        // kb-major: block (sb, r) at (sb·rows + r)·110.
                        let blk = &raw[(sb * rows + r) * 110..(sb * rows + r + 1) * 110];
                        let d = f32::from(half::f16::from_bits(u16::from_le_bytes([
                            blk[108], blk[109],
                        ])));
                        // scales[12] → 16 int8 via the ggml aux permutation.
                        let a0 = u32::from_le_bytes([blk[96], blk[97], blk[98], blk[99]]);
                        let a1 = u32::from_le_bytes([blk[100], blk[101], blk[102], blk[103]]);
                        let tmp = u32::from_le_bytes([blk[104], blk[105], blk[106], blk[107]]);
                        let k1 = 0x03030303u32;
                        let k2 = 0x0f0f0f0fu32;
                        let b0 = (a0 & k2) | ((tmp & k1) << 4);
                        let b1 = (a1 & k2) | (((tmp >> 2) & k1) << 4);
                        let b2 = ((a0 >> 4) & k2) | (((tmp >> 4) & k1) << 4);
                        let b3 = ((a1 >> 4) & k2) | (((tmp >> 6) & k1) << 4);
                        let scs: [i8; 16] = [
                            b0 as i8,
                            (b0 >> 8) as i8,
                            (b0 >> 16) as i8,
                            (b0 >> 24) as i8,
                            b1 as i8,
                            (b1 >> 8) as i8,
                            (b1 >> 16) as i8,
                            (b1 >> 24) as i8,
                            b2 as i8,
                            (b2 >> 8) as i8,
                            (b2 >> 16) as i8,
                            (b2 >> 24) as i8,
                            b3 as i8,
                            (b3 >> 8) as i8,
                            (b3 >> 16) as i8,
                            (b3 >> 24) as i8,
                        ];
                        for j in 0..256 {
                            let s = j >> 4;
                            let l = j & 15;
                            let dl = d * (scs[s] as f32 - 32.0);
                            let qbase = (s % 2) * 16 + (s / 8) * 32;
                            let shift = ((s / 2) % 4) * 2;
                            let hbase = (s % 2) * 16;
                            // Canonical hmask bit s/2 (kernel + encoder
                            // parity — the old (s/2)%4 collided sub-blocks
                            // s and s+8 on the same bit).
                            let mbit = 1 << (s / 2);
                            let q2 = ((blk[32 + qbase + l] >> shift) & 3) as i32;
                            let h = if blk[hbase + l] & mbit != 0 { 0 } else { 4 };
                            out[r * per + sb * 256 + j] = dl * (q2 - h) as f32;
                        }
                    }
                }
                Ok(out)
            }
            crate::backend::TensorInner::CudaQ4K(data) => {
                // N (iter-9): this is the SPEECH-crate fused-kernel dialect
                // ((2j, 2j+1) pairing — from_cpu_q4k only receives the speech
                // quantizer's output), NOT the canonical ggml Q4_0 layout
                // (j, j+16) used by the GGUF path. GGUF Q4_0 never reaches
                // this branch (it flows dequant→f32 via from_quantized).
                let raw: Vec<u8> = self.stream.clone_dtoh(data).map_err(map_err)?;
                let per: usize = t.shape()[1..].iter().product();
                let rows = t.shape()[0];
                let nsup = per / 32;
                let mut out = vec![0f32; rows * per];
                for r in 0..rows {
                    for sb in 0..nsup {
                        // kb-major: block (sb, r) at (sb·rows + r)·18.
                        let blk = &raw[(sb * rows + r) * 18..(sb * rows + r + 1) * 18];
                        let d =
                            f32::from(half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])));
                        for j in 0..32 {
                            let nib = blk[2 + (j >> 1)];
                            let q = if j & 1 == 1 { nib >> 4 } else { nib & 0xF };
                            out[r * per + sb * 32 + j] = (q as i32 - 8) as f32 * d;
                        }
                    }
                }
                Ok(out)
            }
            _ => self.stream.clone_dtoh(t.as_cuda()).map_err(map_err),
        }
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
        // Q6_K / Q4_0 packed weights: fused-dequant GEMM (fp32 activations).
        // m≤64 (TTS decode/fill) uses split-k, k/S > 2048 falls back to the
        // no-shared kernel (the slow-AR w2 has k=9728), m>64 (STT one-shot)
        // uses the row-chunk kernel — the same policy as the int8 path.
        if let crate::backend::TensorInner::CudaQ6K(bpack) = &b.inner {
            return self.gemm_packed_nt(a, bpack, c, m, k, n, 8, 210, 32, "gemm_q6k");
        }
        if let crate::backend::TensorInner::CudaQ5K(bpack) = &b.inner {
            return self.gemm_packed_nt(a, bpack, c, m, k, n, 8, 176, 32, "gemm_q5k");
        }
        if let crate::backend::TensorInner::CudaQ3K(bpack) = &b.inner {
            return self.gemm_packed_nt(a, bpack, c, m, k, n, 8, 110, 32, "gemm_q3k");
        }
        if let crate::backend::TensorInner::CudaQ4KC(bpack) = &b.inner {
            // N (iter-10): canonical Q4_K (GGUF) — v2 dequant-amortized
            // kernels, same structure as q6k.
            return self.gemm_packed_nt(a, bpack, c, m, k, n, 8, 144, 32, "gemm_q4kc");
        }
        if let crate::backend::TensorInner::CudaQ4K(bpack) = &b.inner {
            return self.gemm_packed_nt(a, bpack, c, m, k, n, 5, 18, 256, "gemm_q4k");
        }
        // Plain fp16/bf16 weights: cast the activation row(s) and run cuBLAS
        // gemm_ex (16F×16F → 32F out, fp32 accumulate) — halves the weight
        // read traffic (the decode bottleneck) and the weight VRAM; int8 uses
        // our Q8_0-style kernels below.
        if let crate::backend::TensorInner::CudaI8 {
            data: b8,
            scale: wscale,
        } = &b.inner
        {
            if k % 32 != 0 {
                return Err(BackendError::Shape(format!(
                    "int8 gemm_nt: k={k} not a multiple of 32"
                )));
            }
            // Q8_0-style int8 GEMM: per-32-block weight scales, the A
            // operand follows the activation axis (f32 default, fp16/bf16
            // cast). kb-major packed weights feed the own kernels: m=1
            // (decode) the split-k (the largest S that divides k/32 within
            // a 512-block grid — latency hiding); m>1 (prefill/fill/STT)
            // the tiled kernel (32 rows × 256 cols per block, B staged in
            // shared once per k-block).
            let k32 = k as i32;
            let n32 = n as i32;
            let m32 = m as i32;
            let c = c.as_cuda_mut();
            let grid_x = n.div_ceil(256);
            let act = self.activation_dtype();
            let (tiled_name, splitk_name): (&'static str, &'static str) = match act {
                crate::dtype::Dtype::F16 => ("gemm_q8k_tiled_f16", "gemm_q8k_splitk_f16"),
                crate::dtype::Dtype::BF16 => ("gemm_q8k_tiled_bf16", "gemm_q8k_splitk_bf16"),
                _ => ("gemm_q8k_tiled_f32", "gemm_q8k_splitk_f32"),
            };
            let a_arg: KernA = match act {
                crate::dtype::Dtype::F16 => KernA::F16(self.cast_f16(a)?),
                crate::dtype::Dtype::BF16 => KernA::BF16(self.cast_bf16(a)?),
                _ => KernA::F32(a),
            };
            if m > 1 {
                let s16 = 16usize;
                let cpart = self.take_pooled(s16 * m * n)?;
                let f = self.function(tiled_name)?;
                let cfg = LaunchConfig {
                    grid_dim: (grid_x as u32, m.div_ceil(32) as u32, 16),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut la = self.stream.launch_builder(&f);
                match &a_arg {
                    KernA::F32(t) => la.arg(t.as_cuda()),
                    KernA::F16(sl) => la.arg(sl),
                    KernA::BF16(sl) => la.arg(sl),
                };
                la.arg(b8);
                la.arg(wscale);
                la.arg(&cpart);
                la.arg(&m32);
                la.arg(&k32);
                la.arg(&n32);
                unsafe { la.launch(cfg) }.map_err(map_err)?;
                let used = Self::tiled_used_slices(k, 5);
                self.reduce_and_pool(cpart, c, m, n, used, 16 * m * n)?;
                self.pool_a_scratch(a_arg);
                return Ok(());
            }
            let nblk = k / 32;
            let s = self.pick_split(nblk, m, n, grid_x);
            let cpart = self.take_pooled(s * m * n)?;
            let f = self.function(splitk_name)?;
            let cfg = LaunchConfig {
                grid_dim: ((grid_x * m) as u32, s as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut la = self.stream.launch_builder(&f);
            match &a_arg {
                KernA::F32(t) => la.arg(t.as_cuda()),
                KernA::F16(sl) => la.arg(sl),
                KernA::BF16(sl) => la.arg(sl),
            };
            la.arg(b8);
            la.arg(wscale);
            la.arg(&cpart);
            la.arg(&k32);
            la.arg(&n32);
            let s32 = s as i32;
            la.arg(&s32);
            la.arg(&m32);
            unsafe { la.launch(cfg) }.map_err(map_err)?;
            // Combine the S partials (stream-ordered after splitk).
            self.reduce_and_pool(cpart, c, m, n, s, s * m * n)?;
            self.pool_a_scratch(a_arg);
            return Ok(());
        }
        // fp16/bf16 weights: own split-k GEMM — cuBLAS gemm_ex refuses
        // f32-A × half-B (CUBLAS_STATUS_NOT_SUPPORTED) and cudarc's safe
        // Gemm forces fp16 output, so the half-weight paths never touch
        // cuBLAS. The A operand follows the activation axis (f32 default,
        // fp16/bf16 cast) — gemm_halfw_splitk_{f32,f16,bf16}_{f16,bf16}.
        if let crate::backend::TensorInner::CudaBF16(b16) = &b.inner {
            return self.gemm_halfw_nt(a, b16, c, m, k, n, "bf16");
        }
        if let crate::backend::TensorInner::CudaF16(b16) = &b.inner {
            return self.gemm_halfw_nt(a, b16, c, m, k, n, "f16");
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
        // Mirror of the CPU oracle's guard: cols == 0 divides by zero in the
        // kernel (i % cols) and a non-multiple n would bias the wrong
        // columns / run off the tail.
        if cols == 0 || !n.is_multiple_of(cols as usize) {
            return Err(BackendError::Shape(format!(
                "add_bias: bias {} does not divide x {n}",
                cols
            )));
        }
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
    fn swoosh_l(&self, x: &Tensor) -> Result<Tensor, BackendError> {
        let shape = x.shape().to_vec();
        let mut s = self.take_pooled(x.numel())?;
        let f = self.function("elementwise_f32")?;
        let cfg = LaunchConfig::for_num_elems(x.numel() as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(x.as_cuda());
        la.arg(&mut s);
        let n = x.numel() as i32;
        la.arg(&n);
        let op = ElemOp::SwooshL as i32;
        la.arg(&op);
        la.arg(&0.0f32);
        unsafe { la.launch(cfg).map_err(map_err)? };
        Ok(Tensor::from_cuda(self.wrap_pooled(s), shape))
    }
    fn swoosh_r(&self, x: &Tensor) -> Result<Tensor, BackendError> {
        let shape = x.shape().to_vec();
        let mut s = self.take_pooled(x.numel())?;
        let f = self.function("elementwise_f32")?;
        let cfg = LaunchConfig::for_num_elems(x.numel() as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(x.as_cuda());
        la.arg(&mut s);
        let n = x.numel() as i32;
        la.arg(&n);
        let op = ElemOp::SwooshR as i32;
        la.arg(&op);
        la.arg(&0.0f32);
        unsafe { la.launch(cfg).map_err(map_err)? };
        Ok(Tensor::from_cuda(self.wrap_pooled(s), shape))
    }
    // CUDA-кернел magnitude_sqrt_f32 переписан на построчную семантику
    // (см. kernels.cu), но PTX в репозитории собран из старой версии —
    // оверрайд снят до пересборки PTX на машине с nvcc: хостовая
    // реализация верна для любой раскладки, а тензор VAD мал (T×258).
    // Вернуть оверрайд после `cargo build --features cuda` с nvcc.

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
        let n = 4 * hidden;
        let mut h_new = self.take_pooled(hidden)?;
        let mut c_new = self.take_pooled(hidden)?;
        let f = self.function("lstm_cell_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (n as u32, 1, 1),
            shared_mem_bytes: (n * 4) as u32,
        };
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(h.as_cuda());
        la.arg(c.as_cuda());
        la.arg(w_ih.as_cuda());
        la.arg(w_hh.as_cuda());
        la.arg(b_ih.as_cuda());
        la.arg(b_hh.as_cuda());
        la.arg(&mut h_new);
        la.arg(&mut c_new);
        let hi = hidden as i32;
        la.arg(&hi);
        unsafe { la.launch(cfg).map_err(map_err)? };
        let shape = vec![1, hidden];
        Ok((
            Tensor::from_cuda(self.wrap_pooled(h_new), shape.clone()),
            Tensor::from_cuda(self.wrap_pooled(c_new), shape),
        ))
    }

    fn bias_norm_zipformer(
        &self,
        x: &Tensor,
        bias: &Tensor,
        log_scale: f32,
    ) -> Result<Tensor, BackendError> {
        let shape = x.shape().to_vec();
        // (1, t, d) or (t, d) — one row per (t) slice.
        let (t, d) = if shape.len() == 3 {
            (shape[1], shape[2])
        } else if shape.len() == 2 {
            (shape[0], shape[1])
        } else {
            return Err(BackendError::Shape(format!(
                "bias_norm_zipformer: x shape {shape:?} must be 2D or 3D"
            )));
        };
        let n: usize = shape.iter().product();
        let mut s = self.take_pooled(n)?;
        let f = self.function("bias_norm_zipformer_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (t as u32, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(bias.as_cuda());
        la.arg(&log_scale);
        la.arg(&mut s);
        let (ti, di) = (t as i32, d as i32);
        la.arg(&ti);
        la.arg(&di);
        unsafe { la.launch(cfg).map_err(map_err)? };
        Ok(Tensor::from_cuda(self.wrap_pooled(s), shape))
    }
    fn silu_mul_fused(&self, x: &Tensor, y: &Tensor, ffn: usize) -> Result<(), BackendError> {
        let numel = y.numel();
        // Mirror of the CPU oracle's guard: the row-major kernel computes
        // row = i / ffn, col = i % ffn — ffn == 0 divides by zero and a
        // non-multiple numel would mis-slice the last row.
        if ffn == 0 || !numel.is_multiple_of(ffn) {
            return Err(BackendError::Shape(format!(
                "silu_mul_fused: ffn={ffn} does not divide y numel {numel}"
            )));
        }
        let n = numel as i32;
        let ffn_i = ffn as i32;
        let xd = x.as_cuda();
        let od = y.as_cuda();
        let f = self.function("silu_mul_fused_f32")?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(xd);
        la.arg(od);
        la.arg(&n);
        la.arg(&ffn_i);
        unsafe { la.launch(cfg).map_err(map_err)? };
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
        let (n_tokens, n_embd) = (y.shape()[0] as i32, y.shape()[1] as i32);
        let n_groups = n_embd / group_size as i32;
        let f = self.function("dflash2_conv_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (
                (n_tokens as u32).div_ceil(32),
                (n_embd as u32).div_ceil(8),
                1,
            ),
            block_dim: (32, 8, 1),
            shared_mem_bytes: 0,
        };
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(base.as_cuda());
        la.arg(dynamic.as_cuda());
        la.arg(y.as_cuda());
        la.arg(&n_tokens);
        la.arg(&n_embd);
        // N1 (build fix): the `&(expr as i32)` temporaries were dropped while
        // borrowed (E0716); bind to locals so the launch builder can borrow them.
        let kernel32 = kernel as i32;
        let group32 = group_size as i32;
        let block32 = block_size as i32;
        la.arg(&kernel32);
        la.arg(&group32);
        la.arg(&block32);
        la.arg(&n_groups);
        let side32 = side as i32;
        la.arg(&side32);
        unsafe { la.launch(cfg).map_err(map_err)? };
        Ok(())
    }
    fn gelu_tanh(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        self.elementwise(ElemOp::GeluTanh, x, None, y, 0.0)
    }
    fn gelu_erf(&self, x: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        self.elementwise(ElemOp::GeluErf, x, None, y, 0.0)
    }
    fn snake(&self, x: &Tensor, a: &Tensor, y: &Tensor) -> Result<(), BackendError> {
        // Per-channel (Metal semantics): A (C,), X flattened (n·c).
        let c = a.numel() as i32;
        // Q21/R28 (mirror of the CPU reference): c == 0 divides by zero
        // (x.numel() / c — host and kernel); c not dividing x.numel() would
        // truncate the tail. (a == 0 is fine — the kernel treats it as the
        // sin²(ax)/a → 0 limit.)
        if c == 0 || !x.numel().is_multiple_of(c as usize) {
            return Err(BackendError::Shape(format!(
                "snake: channel count {c} does not divide x ({})",
                x.numel()
            )));
        }
        let n = (x.numel() / c as usize) as i32;
        let f = self.function("snake_f32")?;
        let cfg = LaunchConfig::for_num_elems((n * c) as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(a.as_cuda());
        la.arg(y.as_cuda());
        let __a1 = n;
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
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
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
    fn layernorm(
        &self,
        x: &Tensor,
        w: &Tensor,
        b: &Tensor,
        y: &Tensor,
        eps: f32,
    ) -> Result<(), BackendError> {
        // Contract: X (rows, cols) — 2D (callers flatten (B,T) into rows).
        let rows = x.shape()[0];
        let cols = x.shape()[1] as i32;
        let f = self.function("layernorm_f32")?;
        // One block per row — the kernel indexes rows via blockIdx.x.
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
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
        // Q22 (mirror of the CPU reference): groups == 0 divides by zero in
        // the kernel (`cin / groups`, `cout / groups`); groups > cin or a
        // cin not divisible by groups reads wrong weight rows; the weight
        // row count cin_g must equal cin / groups.
        let cin_g = w.shape()[1];
        if groups == 0
            || groups > cin
            || cin % groups != 0
            || cout % groups != 0
            || cin_g != cin / groups
        {
            return Err(BackendError::Shape(format!(
                "conv1d_groups: group mismatch (cout={cout}, cin={cin}, groups={groups}, cin_g={cin_g})"
            )));
        }
        // Q22b: stride == 0 divides by zero, and a padded input smaller
        // than the dilated kernel window underflowed the usize `tout`
        // (wrapped → nonsense shape / huge grid). Guard BEFORE computing
        // tout — the CPU oracle shares the same arithmetic.
        if stride == 0 {
            return Err(BackendError::Shape(format!(
                "conv1d_groups: stride 0 (t={t}, pad_l={pad_l}, pad_r={pad_r}, dil={dil}, k={k})"
            )));
        }
        if t + pad_l + pad_r < dil * (k - 1) + 1 {
            return Err(BackendError::Shape(format!(
                "conv1d_groups: padded input {} smaller than the dilated kernel window {} (t={t}, pad_l={pad_l}, pad_r={pad_r}, dil={dil}, k={k})",
                t + pad_l + pad_r,
                dil * (k - 1) + 1
            )));
        }
        let tout = (t + pad_l + pad_r - dil * (k - 1) - 1) / stride + 1;
        if tout == 0 {
            // Unreachable since the Q22b window guard (the CPU now returns
            // Err for this case as well) — kept as belt-and-braces against a
            // 0-element launch (invalid 0-block grid).
            self.stream.memset_zeros(y.as_cuda_mut()).map_err(map_err)?;
            return Ok(());
        }
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
        if ci != w.shape()[1] {
            return Err(BackendError::Shape("conv2d: channel mismatch".into()));
        }
        if stride_t == 0 || stride_f == 0 {
            return Err(BackendError::Shape("conv2d: stride 0".into()));
        }
        // R (iter-8 review): the same usize-underflow class conv1d guarded —
        // a window larger than the padded input wrapped oh/ow to a huge
        // value and the launch OOB-wrote y (no shape check here).
        if h + pad_h < kh || wdim + pad_w < kw {
            return Err(BackendError::Shape(format!(
                "conv2d: padded input {}x{} smaller than the kernel {}x{}",
                h + pad_h,
                wdim + pad_w,
                kh,
                kw
            )));
        }
        // LEFT padding (Metal conv2d_f32 semantics) — symmetric padding here
        // would make the kernel write past the output buffer (oh too big).
        let (oh, ow) = (
            (h + pad_h - kh) / stride_t + 1,
            (wdim + pad_w - kw) / stride_f + 1,
        );
        let n_elems = if time_major_out {
            n * oh * co * ow
        } else {
            n * co * oh * ow
        };
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
        // Детерминированный вариант (аудит 2026-10-13): поток на выход,
        // аккумулятор в регистре, bias слит — без memset и без atomicAdd.
        let n = b * tout * cout;
        let f = self.function("conv_transpose1d_f32")?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(w.as_cuda());
        let hb = bias.is_some() as i32;
        if let Some(bi) = bias {
            la.arg(bi.as_cuda());
        } else {
            la.arg(x.as_cuda()); // dummy — kernel guards via has_bias
        }
        la.arg(&hb);
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
        // Mirror of the CPU oracle's guard: off + d0 past D (or a mismatched
        // y) would read/write OOB device memory — the GPU has no panic to
        // catch it.
        if off + d0 > d || y.shape() != [rows, d0] {
            return Err(BackendError::Shape(format!(
                "slice_last: x {:?} y {:?} off {off}",
                x.shape(),
                y.shape()
            )));
        }
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
        // Mirror of the CPU oracle's guards (Q22): u > t underflows
        // `t - u` (wrap in release, panic in debug), and a mismatched
        // output shape would read/write past the buffers.
        if u > t {
            return Err(BackendError::Shape(format!("slice_time: u={u} > t={t}")));
        }
        if y.shape() != [b, t - u, c] {
            return Err(BackendError::Shape(
                "slice_time: output shape mismatch".into(),
            ));
        }
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
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
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
        // The CPU oracle rejects out-of-range indices (Err); the GPU kernels
        // read table[idx·d .. ] — guard BEFORE launch so a bad codebook/token
        // index cannot silently read OOB memory.
        let rows = table.shape()[0];
        if let Some(&bad) = idx.iter().find(|&&i| i as usize >= rows) {
            return Err(BackendError::Shape(format!(
                "embedding: index {bad} >= rows {rows}"
            )));
        }
        let idx_dev = self.stream.clone_htod(idx).map_err(map_err)?;
        let cfg = LaunchConfig::for_num_elems((n * d as usize) as u32);
        let n32 = n as i32;
        macro_rules! launch_embedding {
            ($fname:literal, $table:expr) => {{
                let f = self.function($fname)?;
                let mut la = self.stream.launch_builder(&f);
                la.arg(&idx_dev);
                la.arg($table);
                la.arg(y.as_cuda());
                la.arg(&n32);
                la.arg(&d);
                unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
            }};
        }
        // Half-precision tables (TTS embeddings): the dedicated kernels
        // convert to f32 on read — the output tensor stays f32.
        match &table.inner {
            crate::backend::TensorInner::CudaBF16(t) => launch_embedding!("embedding_bf16", t),
            crate::backend::TensorInner::CudaF16(t) => launch_embedding!("embedding_f16", t),
            _ => launch_embedding!("embedding_f32", table.as_cuda()),
        }
    }

    fn rope(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        y: &Tensor,
        rope_dim: usize,
    ) -> Result<(), BackendError> {
        let (t, hd) = (x.shape()[0], x.shape()[1]);
        let n = t * hd;
        // F16: cos/sin indexed [ti·half + i] — short tables (past max_seq_len)
        // read OOB on the GPU; fail loudly like the CPU oracle.
        let half = rope_dim / 2;
        if cos.numel() < t * half || sin.numel() < t * half {
            return Err(BackendError::Shape(format!(
                "rope: cos/sin {} elements < t·half {} (t={t}, rope_dim={rope_dim})",
                cos.numel().min(sin.numel()),
                t * half
            )));
        }
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

    fn rope_interleaved(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        y: &Tensor,
        rope_dim: usize,
        start: usize,
    ) -> Result<(), BackendError> {
        let (b, t, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let n = b * t * d;
        // F16: cos/sin indexed at absolute position (start + ti) — the table
        // must cover (start + t) positions.
        let half = rope_dim / 2;
        if cos.numel() < (start + t) * half || sin.numel() < (start + t) * half {
            return Err(BackendError::Shape(format!(
                "rope_interleaved: cos/sin {} < (start+t)·half {} (start={start}, t={t})",
                cos.numel().min(sin.numel()),
                (start + t) * half
            )));
        }
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

    fn rope_head(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        y: &Tensor,
        h: usize,
        rope_dim: usize,
    ) -> Result<(), BackendError> {
        let (b, t, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let hd = d / h;
        let n = b * t * d;
        let half = rope_dim / 2;
        if cos.numel() < t * half || sin.numel() < t * half {
            return Err(BackendError::Shape(format!(
                "rope_head: cos/sin {} elements < t·half {} (t={t}, rope_dim={rope_dim})",
                cos.numel().min(sin.numel()),
                t * half
            )));
        }
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

    fn rmsnorm_head(
        &self,
        x: &Tensor,
        w: &Tensor,
        y: &Tensor,
        h: usize,
        eps: f32,
    ) -> Result<(), BackendError> {
        let rows = x.shape()[0];
        // h == 0 divides by zero (x.shape()[1] / h) and would launch a
        // degenerate (rows, 0) grid; a non-multiple x dim silently truncates
        // hd and mis-addresses rows — strict divisibility, same as the CPU
        // reference.
        if h == 0 || !x.shape()[1].is_multiple_of(h) {
            return Err(BackendError::Shape(format!(
                "rmsnorm_head: head count h={h} does not divide x dim {}",
                x.shape()[1]
            )));
        }
        let hd = x.shape()[1] / h;
        let f = self.function("rmsnorm_head_f32")?;
        // One block per (row, head) — kernel indexes via blockIdx.(x, y).
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, h as u32, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
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

    // N (iter-10): GPU implementations of rope/rmsnorm_head_rope — the trait
    // defaults ran to_cpu_f32 → host sin/cos → from_cpu_f32 (2 full device
    // syncs × 2 per layer per decode token = 72 syncs/token for 36 layers,
    // the loop's dominant host cost). The kernels compute theta/cos/sin
    // inline with the EXACT CPU-reference semantics (halves pairing, e-
    // section skip, per-token 4D positions).

    fn rope_standard(
        &self,
        x: &Tensor,
        pos: usize,
        hd: usize,
        theta: f32,
        yarn: Option<&crate::rope::RopeYarn>,
    ) -> Result<Tensor, BackendError> {
        // YaRN в CUDA-кернеле — Ф9; честная ошибка вместо молчаливого игнора
        if let Some(y) = yarn {
            if !y.is_identity() {
                return Err(BackendError::Shape(
                    "rope_standard: YaRN/линейное скалирование в CUDA ещё не реализовано (Ф9)"
                        .into(),
                ));
            }
        }
        let (tq, d) = (x.shape()[1], x.shape()[2]);
        if d % hd != 0 || !hd.is_multiple_of(2) || hd == 0 {
            return Err(BackendError::Shape(format!(
                "rope_standard: hd={hd} does not divide d={d} (and must be even)"
            )));
        }
        let y = self.zeros(vec![1, tq, d])?;
        let n = tq * (d / hd) * (hd / 2);
        let f = self.function("rope_std_f32")?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(y.as_cuda());
        let (tq, d, hd, pos) = (tq as i32, d as i32, hd as i32, pos as i32);
        la.arg(&tq);
        la.arg(&d);
        la.arg(&hd);
        la.arg(&pos);
        la.arg(&theta);
        unsafe { la.launch(cfg) }.map_err(map_err)?;
        Ok(y)
    }

    fn rmsnorm_head_rope(
        &self,
        x: &Tensor,
        w: &Tensor,
        pos: usize,
        hd: usize,
        theta: f32,
        h: usize,
        eps: f32,
        yarn: Option<&crate::rope::RopeYarn>,
    ) -> Result<Tensor, BackendError> {
        // YaRN в CUDA-кернеле — Ф9; честная ошибка вместо молчаливого игнора
        if let Some(y) = yarn {
            if !y.is_identity() {
                return Err(BackendError::Shape(
                    "rmsnorm_head_rope: YaRN/линейное скалирование в CUDA ещё не реализовано (Ф9)"
                        .into(),
                ));
            }
        }
        let (tq, d) = (x.shape()[1], x.shape()[2]);
        if h == 0 || d != h * hd || !hd.is_multiple_of(2) {
            return Err(BackendError::Shape(format!(
                "rmsnorm_head_rope: h={h}, hd={hd} vs d={d}"
            )));
        }
        let y = self.zeros(vec![1, tq, d])?;
        let n = tq * h * (hd / 2);
        let f = self.function("rmsnorm_head_rope_f32")?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(w.as_cuda());
        la.arg(y.as_cuda());
        let (tq, h, hd, pos) = (tq as i32, h as i32, hd as i32, pos as i32);
        la.arg(&tq);
        la.arg(&h);
        la.arg(&hd);
        la.arg(&eps);
        la.arg(&pos);
        la.arg(&theta);
        unsafe { la.launch(cfg) }.map_err(map_err)?;
        Ok(y)
    }

    fn rope_mrope_text(
        &self,
        x: &Tensor,
        pos: usize,
        hd: usize,
        theta: f32,
        sections: [usize; 4],
    ) -> Result<Tensor, BackendError> {
        let (tq, d) = (x.shape()[1], x.shape()[2]);
        if d % hd != 0 || !hd.is_multiple_of(2) || hd == 0 {
            return Err(BackendError::Shape(format!(
                "rope_mrope_text: hd={hd} does not divide d={d}"
            )));
        }
        let sect_dims = sections.iter().sum::<usize>().max(1);
        let y = self.zeros(vec![1, tq, d])?;
        let n = tq * (d / hd) * (hd / 2);
        let f = self.function("rope_mrope_text_f32")?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(y.as_cuda());
        let (tq, d, hd, pos) = (tq as i32, d as i32, hd as i32, pos as i32);
        let (s0, s1, s2) = (sections[0] as i32, sections[1] as i32, sections[2] as i32);
        let sd = sect_dims as i32;
        la.arg(&tq);
        la.arg(&d);
        la.arg(&hd);
        la.arg(&pos);
        la.arg(&theta);
        la.arg(&s0);
        la.arg(&s1);
        la.arg(&s2);
        la.arg(&sd);
        unsafe { la.launch(cfg) }.map_err(map_err)?;
        Ok(y)
    }

    fn rope_mrope(
        &self,
        x: &Tensor,
        pos4d: &[u32],
        hd: usize,
        theta: f32,
        sections: [usize; 4],
    ) -> Result<Tensor, BackendError> {
        let (tq, d) = (x.shape()[1], x.shape()[2]);
        if pos4d.len() != tq * 4 {
            return Err(BackendError::Shape(format!(
                "rope_mrope: pos4d {} != tq·4 {}",
                pos4d.len(),
                tq * 4
            )));
        }
        if d % hd != 0 || !hd.is_multiple_of(2) || hd == 0 {
            return Err(BackendError::Shape(format!(
                "rope_mrope: hd={hd} does not divide d={d}"
            )));
        }
        let sect_dims = sections.iter().sum::<usize>().max(1);
        let y = self.zeros(vec![1, tq, d])?;
        // Positions as f32 (exact below 2^24) — avoids a u32 buffer type.
        let pos_f32: Vec<f32> = pos4d.iter().map(|&v| v as f32).collect();
        let pos_gpu = self.from_cpu_f32(&pos_f32, vec![tq * 4])?;
        let n = tq * (d / hd) * (hd / 2);
        let f = self.function("rope_mrope_f32")?;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(pos_gpu.as_cuda());
        la.arg(y.as_cuda());
        let (tq, d, hd) = (tq as i32, d as i32, hd as i32);
        let (s0, s1, s2) = (sections[0] as i32, sections[1] as i32, sections[2] as i32);
        let sd = sect_dims as i32;
        la.arg(&tq);
        la.arg(&d);
        la.arg(&hd);
        la.arg(&theta);
        la.arg(&s0);
        la.arg(&s1);
        la.arg(&s2);
        la.arg(&sd);
        unsafe { la.launch(cfg) }.map_err(map_err)?;
        Ok(y)
    }

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
        // N (iter-10): the full penalty pipeline + softmax denominator + top-k
        // run on the GPU (hist_counts_u32 + tts_topk_penalized_f32 +
        // tts_topk_merge_f32); only k candidates and the denominator come
        // back. The old path downloaded the whole vocab row and did 4 host
        // passes per token (the TTS sampler's dominant cost).
        let v = logits.shape()[logits.shape().len() - 1];
        let kk = k.clamp(1, 32);
        if v == 0 {
            return Err(BackendError::Shape(
                "penalized_topk: empty logits row".into(),
            ));
        }
        let h = history.len();
        // History + histogram counts (u32).
        let mut hist_g: Option<cudarc::driver::CudaSlice<u32>> = None;
        if h > 0 {
            let mut hg = unsafe { self.stream.alloc::<u32>(h) }.map_err(map_err)?;
            self.stream.memcpy_htod(history, &mut hg).map_err(map_err)?;
            hist_g = Some(hg);
        }
        let mut counts = unsafe { self.stream.alloc::<u32>(v) }.map_err(map_err)?;
        self.stream.memset_zeros(&mut counts).map_err(map_err)?;
        if let Some(hg) = &hist_g {
            let f = self.function("hist_counts_u32")?;
            let cfg = LaunchConfig::for_num_elems(h as u32);
            let mut la = self.stream.launch_builder(&f);
            la.arg(hg);
            la.arg(&counts);
            let hi = h as i32;
            la.arg(&hi);
            unsafe { la.launch(cfg) }.map_err(map_err)?;
        }
        // Scratch: per-thread heaps (256 × kk), max, sum.
        let mut heap_val = unsafe { self.stream.alloc::<f32>(256 * kk) }.map_err(map_err)?;
        let mut heap_idx = unsafe { self.stream.alloc::<u32>(256 * kk) }.map_err(map_err)?;
        let mut out_max = unsafe { self.stream.alloc::<f32>(1) }.map_err(map_err)?;
        let mut out_sum = unsafe { self.stream.alloc::<f32>(1) }.map_err(map_err)?;
        {
            let f = self.function("tts_topk_penalized_f32")?;
            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut la = self.stream.launch_builder(&f);
            la.arg(logits.as_cuda());
            la.arg(bias.as_cuda());
            la.arg(&counts);
            // The kernel only dereferences the history pointer when
            // ras_penalty applies AND h >= 2; pass a dummy when empty.
            let empty_hist: cudarc::driver::CudaSlice<u32> =
                unsafe { self.stream.alloc::<u32>(1) }.map_err(map_err)?;
            la.arg(hist_g.as_ref().unwrap_or(&empty_hist));
            let (vi, ki) = (v as i32, kk as i32);
            let hi = h as i32;
            let rw = ras_window as i32;
            la.arg(&hi);
            la.arg(&vi);
            la.arg(&ki);
            la.arg(&rep_penalty);
            la.arg(&ras_penalty);
            la.arg(&rw);
            la.arg(&temperature);
            la.arg(&mut heap_val);
            la.arg(&mut heap_idx);
            la.arg(&mut out_max);
            la.arg(&mut out_sum);
            unsafe { la.launch(cfg) }.map_err(map_err)?;
        }
        {
            let f = self.function("tts_topk_merge_f32")?;
            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let n = 256 * kk;
            let mut topk_val = unsafe { self.stream.alloc::<f32>(kk) }.map_err(map_err)?;
            let mut topk_idx = unsafe { self.stream.alloc::<u32>(kk) }.map_err(map_err)?;
            let mut la = self.stream.launch_builder(&f);
            la.arg(&heap_val);
            la.arg(&heap_idx);
            let ni = n as i32;
            la.arg(&ni);
            let ki = kk as i32;
            la.arg(&ki);
            la.arg(&mut topk_val);
            la.arg(&mut topk_idx);
            unsafe { la.launch(cfg) }.map_err(map_err)?;
            let vals = self.stream.clone_dtoh(&topk_val).map_err(map_err)?;
            let idxs = self.stream.clone_dtoh(&topk_idx).map_err(map_err)?;
            let max = self.stream.clone_dtoh(&out_max).map_err(map_err)?[0];
            let sum = self.stream.clone_dtoh(&out_sum).map_err(map_err)?[0];
            let mut out: Vec<(u32, f32)> = Vec::with_capacity(kk);
            if !sum.is_finite() || sum <= 0.0 {
                let p = 1.0 / v as f32;
                for i in 0..kk.min(v) {
                    out.push((i as u32, p));
                }
                return Ok(out);
            }
            for i in 0..kk {
                let p = ((vals[i] - max).exp() / sum).max(0.0);
                out.push((idxs[i], p));
            }
            Ok(out)
        }
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
        let rows = x.shape()[0];
        let hd = x.shape()[1] / h;
        let f = self.function("layernorm_head_f32")?;
        // One block per (row, head) — kernel indexes via blockIdx.(x, y).
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, h as u32, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
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

    fn add_scaled(
        &self,
        a: &Tensor,
        b: &Tensor,
        y: &Tensor,
        scale: f32,
    ) -> Result<(), BackendError> {
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
        // D8 (mirror of the CPU oracle): the mask is indexed as [tq, tkv] —
        // a too-small mask would read OOB device memory (garbage bias, no
        // panic to catch it on the GPU).
        if let Some(m) = mask
            && m.numel() < tq * tkv
        {
            return Err(BackendError::Shape(format!(
                "attn_ar: mask numel {} < tq·tkv {tq}·{tkv}",
                m.numel()
            )));
        }
        // Q20: the v4 kernel stages one warp per 32 columns in an 8-warp
        // block — hd > 256 would overflow sh_p[8][64] (and nwarps > 8).
        if hd > 256 {
            return Err(BackendError::UnsupportedFormat(format!(
                "attn_ar: head_dim {hd} > 256 not supported by attn_ar_v4"
            )));
        }
        if tkv == 0 {
            // Q24: zero the output — a stale pool buffer must not come back
            // as "attention result" for an empty cache.
            self.stream.memset_zeros(y.as_cuda_mut()).map_err(map_err)?;
            return Ok(());
        }
        // Two-kernel warp-chunked attention (attn_ar_v4): each (chunk, warp)
        // writes a 34-f32 partial (max, sum-exp, weighted-V[32]) with ZERO
        // block-level syncs in the hot loop; attn_ar_v4_combine merges them.
        let nwarps = hd.div_ceil(32);
        let nchunks = tkv.div_ceil(64).max(1);
        // CUDA grid.z is limited to 65535 — a huge batch × chunk product
        // would fail the launch with CUDA_ERROR_INVALID_VALUE (or be
        // silently truncated by a wrap in the cast) instead of Err.
        let grid_z = b * nchunks;
        if grid_z > 65535 {
            return Err(BackendError::Shape(format!(
                "attn_ar: grid.z {grid_z} (b={b} · nchunks={nchunks}) exceeds the CUDA limit 65535"
            )));
        }
        let scratch = self.take_pooled(b * tq * n_heads * nchunks * nwarps * 34)?;
        // K/V storage dtype: 0=f32, 1=f16/bf16, 2=q8_0 — the v4 kernels
        // dequantize inline on read (f16 → float, q8_0 block-scale).
        let kfmt = match &k.inner {
            crate::backend::TensorInner::Cuda(_) => 0,
            crate::backend::TensorInner::CudaF16(_) | crate::backend::TensorInner::CudaBF16(_) => 1,
            crate::backend::TensorInner::CudaI8 { .. } => 2,
            _ => unreachable!(),
        };
        let vfmt = match &v.inner {
            crate::backend::TensorInner::Cuda(_) => 0,
            crate::backend::TensorInner::CudaF16(_) | crate::backend::TensorInner::CudaBF16(_) => 1,
            crate::backend::TensorInner::CudaI8 { .. } => 2,
            _ => unreachable!(),
        };
        // P18b: the v4 kernels have f32/f16/q8_0 KV variants ONLY (KV_READ
        // decodes __half2float for fmt=1). A CudaBF16 K/V buffer must never
        // be passed to the f16 kernel — bf16 and f16 are different exponent
        // layouts, so every value would decode as garbage. There is no
        // bf16→f16 conversion kernel (cast_f32_f16/cast_f32_bf16 both read
        // f32) and the cache-creation path is f32/f16/q8_0 only, so a bf16
        // KV cache cannot exist; refuse loudly if one ever appears (same
        // contract as kv_update's P18a guard).
        if matches!(k.inner, crate::backend::TensorInner::CudaBF16(_))
            || matches!(v.inner, crate::backend::TensorInner::CudaBF16(_))
        {
            return Err(BackendError::UnsupportedFormat(
                "attn_ar: bf16 KV storage is not supported on CUDA (no bf16 kernel — the f16 kernels would decode bf16 bits as f16)".into(),
            ));
        }
        let fname: &'static str = match (kfmt, vfmt) {
            (0, 0) => "attn_ar_v4_f32f32",
            (0, 1) => "attn_ar_v4_f32f16",
            (0, 2) => "attn_ar_v4_f32q8",
            (1, 0) => "attn_ar_v4_f16f32",
            (1, 1) => "attn_ar_v4_f16f16",
            (1, 2) => "attn_ar_v4_f16q8",
            (2, 0) => "attn_ar_v4_q8f32",
            (2, 1) => "attn_ar_v4_q8f16",
            (2, 2) => "attn_ar_v4_q8q8",
            _ => unreachable!(),
        };
        let f = self.function(fname)?;
        // N (iter-10): size the block to the ACTIVE warps (nwarps = hd/32) —
        // the old fixed 256-thread block left 4 of 8 warps idle at hd=128
        // (the real qwen3/DFlash dim) — half the issue slots wasted on a
        // latency-bound kernel.
        let block_w = (nwarps * 32).max(32) as u32;
        let cfg = LaunchConfig {
            grid_dim: (tq as u32, n_heads as u32, (b * nchunks) as u32),
            block_dim: (block_w, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut la = self.stream.launch_builder(&f);
        la.arg(q.as_cuda());
        match &k.inner {
            crate::backend::TensorInner::Cuda(_) => {
                la.arg(k.as_cuda());
            }
            crate::backend::TensorInner::CudaF16(b) => {
                la.arg(b);
            }
            crate::backend::TensorInner::CudaBF16(b) => {
                la.arg(b);
            }
            crate::backend::TensorInner::CudaI8 { data, .. } => {
                la.arg(data);
            }
            _ => unreachable!(),
        }
        match &v.inner {
            crate::backend::TensorInner::Cuda(_) => {
                la.arg(v.as_cuda());
            }
            crate::backend::TensorInner::CudaF16(b) => {
                la.arg(b);
            }
            crate::backend::TensorInner::CudaBF16(b) => {
                la.arg(b);
            }
            crate::backend::TensorInner::CudaI8 { data, .. } => {
                la.arg(data);
            }
            _ => unreachable!(),
        }
        if let Some(m) = mask {
            la.arg(m.as_cuda());
        } else {
            la.arg(q.as_cuda()); // dummy — kernel guards via has_mask
        }
        la.arg(&scratch);
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
        let nw32 = nwarps as i32;
        la.arg(&nw32);
        let nc32 = nchunks as i32;
        la.arg(&nc32);
        // Block scales for q8_0 K/V (dummy f32 slice for non-q8 — unused;
        // a per-call alloc of 4 bytes is cheaper than a borrow juggle).
        let dummy = self.take_pooled(1)?;
        let dummy2 = dummy.clone();
        match &k.inner {
            crate::backend::TensorInner::CudaI8 { scale, .. } => {
                la.arg(scale);
            }
            _ => {
                la.arg(&dummy);
            }
        }
        match &v.inner {
            crate::backend::TensorInner::CudaI8 { scale, .. } => {
                la.arg(scale);
            }
            _ => {
                la.arg(&dummy2);
            }
        }
        unsafe { la.launch(cfg) }.map_err(map_err)?;
        // C6: return the dummy scales to the pool — they leaked per call.
        lock_poison_resilient(&self.pool)
            .entry(1)
            .or_default()
            .push(dummy);
        // Merge the partials into the output.
        let f = self.function("attn_ar_v4_combine")?;
        let cfg = LaunchConfig {
            grid_dim: (tq as u32, n_heads as u32, b as u32),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut la = self.stream.launch_builder(&f);
        la.arg(&scratch);
        la.arg(y.as_cuda_mut());
        la.arg(&__a84);
        la.arg(&__a85);
        let nh32 = n_heads as i32;
        la.arg(&nh32);
        let hd32 = hd as i32;
        la.arg(&hd32);
        la.arg(&nw32);
        la.arg(&nc32);
        unsafe { la.launch(cfg) }.map_err(map_err)?;
        // Q19: return the scratch to the pool WITH the byte/per-bucket caps
        // (the old code pushed unconditionally via `.unwrap()` — a poisoned
        // lock aborted the process, and unbounded buckets grew VRAM on long
        // generations: every new tkv created a new bucket entry forever).
        push_capped(&self.pool, scratch, 4);
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
        // mode 0 = leader (scores → SC + mask), 1 = non-leader (reuse SC),
        // 2 = uniform (mean V) — matches the Metal kernel.
        let (b, t, d) = (q.shape()[0], q.shape()[1], q.shape()[2]);
        let h = sc.shape()[1];
        let hd = d / h;
        // D8 (mirror of the CPU oracle): the mask is indexed as [t, t] — a
        // too-small mask would read OOB device memory.
        if let Some(m) = mask
            && m.numel() < t * t
        {
            return Err(BackendError::Shape(format!(
                "mhsa: mask numel {} < t·t {t}·{t}",
                m.numel()
            )));
        }
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
        // Block scales for q8_0 K/V (dummy f32 slice for non-q8 — unused;
        // a per-call alloc of 4 bytes is cheaper than a borrow juggle).
        let dummy = self.take_pooled(1)?;
        let dummy2 = dummy.clone();
        match &k.inner {
            crate::backend::TensorInner::CudaI8 { scale, .. } => {
                la.arg(scale);
            }
            _ => {
                la.arg(&dummy);
            }
        }
        match &v.inner {
            crate::backend::TensorInner::CudaI8 { scale, .. } => {
                la.arg(scale);
            }
            _ => {
                la.arg(&dummy2);
            }
        }
        unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)?;
        // C6: return the dummy scales to the pool (they leaked per call).
        lock_poison_resilient(&self.pool)
            .entry(1)
            .or_default()
            .push(dummy);
        Ok(())
    }

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
        scale: f32,
    ) -> Result<(), BackendError> {
        let (t, d) = (q.shape()[0], q.shape()[1]);
        let h = d / dk;
        // The kernel stages per-chunk score rows in
        // `__shared__ float scs[Q][2048]` — T > 2048 would index past the
        // shared array (OOB). Guard before launch (the CPU reference errors
        // on the same contract).
        if t > 2048 {
            return Err(BackendError::Shape(format!(
                "mhsa_rel_pos: T {t} exceeds the kernel's 2048-element shared-memory score limit"
            )));
        }
        // The kernel's weighted-V stage is one thread per column in a
        // 256-thread block (`if (tid < dk)`) — dk > 256 would SILENTLY DROP
        // the columns past 256 instead of failing. Real callers use
        // dk ≤ 128 (NeMo/zipformer); refuse anything larger loudly.
        if dk > 256 {
            return Err(BackendError::Shape(format!(
                "mhsa_rel_pos: dk {dk} exceeds the kernel's 256-thread weighted-V limit"
            )));
        }
        let f = self.function("mhsa_rel_pos_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (t.div_ceil(4) as u32, h as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut la = self.stream.launch_builder(&f);
        la.arg(q.as_cuda());
        la.arg(k.as_cuda());
        la.arg(v.as_cuda());
        la.arg(qp.as_cuda());
        la.arg(p.as_cuda());
        la.arg(u.as_cuda());
        la.arg(vb.as_cuda());
        la.arg(y.as_cuda_mut());
        if let Some(w) = wfull {
            la.arg(w.as_cuda_mut());
        } else {
            la.arg(y.as_cuda()); // dummy pointer — never NULL, see has_wfull below (O4)
        }
        let t32 = t as i32;
        la.arg(&t32);
        let dk32 = dk as i32;
        la.arg(&dk32);
        let dp32 = dp as i32;
        la.arg(&dp32);
        let h32 = h as i32;
        la.arg(&h32);
        let m32 = mode as i32;
        la.arg(&m32);
        la.arg(&scale);
        // O4: the kernel must know whether the wfull pointer is REAL — the
        // dummy above is never NULL, and the old `wfull != NULL` guard made
        // wfull=None (NeMo mode 0) write post-softmax weights into the
        // output buffer y (concurrent blocks raced it; OOB past y when T > d).
        let has_wfull = i32::from(wfull.is_some());
        la.arg(&has_wfull);
        unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
    }

    fn mhsa_apply_weights(
        &self,
        w: &Tensor,
        x: &Tensor,
        y: &Tensor,
        vh: usize,
    ) -> Result<(), BackendError> {
        let (t, d) = (x.shape()[0], x.shape()[1]);
        let h = d / vh;
        let f = self.function("mhsa_apply_weights_f32")?;
        let cfg = LaunchConfig {
            grid_dim: (1, h as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut la = self.stream.launch_builder(&f);
        la.arg(w.as_cuda());
        la.arg(x.as_cuda());
        la.arg(y.as_cuda_mut());
        let t32 = t as i32;
        la.arg(&t32);
        let vh32 = vh as i32;
        la.arg(&vh32);
        let h32 = h as i32;
        la.arg(&h32);
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

    fn kv_shifted(&self, buf: &Tensor, shift: usize, pos: usize) -> Result<Tensor, BackendError> {
        let (b, cap, d) = (buf.shape()[0], buf.shape()[1], buf.shape()[2]);
        let _shape = buf.shape().to_vec();
        let (bi, si, pi, di, ci) = (b as i32, shift as i32, pos as i32, d as i32, cap as i32);
        // In-place shift on the same buffer (the regions are disjoint;
        // the kernels write through the handle — the returned tensor is a
        // shallow clone of the mutated buffer).
        match &buf.inner {
            crate::backend::TensorInner::Cuda(_) => {
                let f = self.function("kv_shift_f32")?;
                let cfg = LaunchConfig::for_num_elems((b * (pos - shift) * d) as u32);
                let mut la = self.stream.launch_builder(&f);
                la.arg(buf.as_cuda());
                la.arg(&bi);
                la.arg(&si);
                la.arg(&pi);
                la.arg(&di);
                la.arg(&ci);
                unsafe { la.launch(cfg).map_err(map_err)? };
                Ok(buf.clone())
            }
            crate::backend::TensorInner::CudaF16(s) => {
                let f = self.function("kv_shift_f16")?;
                let cfg = LaunchConfig::for_num_elems((b * (pos - shift) * d) as u32);
                let mut la = self.stream.launch_builder(&f);
                la.arg(s);
                la.arg(&bi);
                la.arg(&si);
                la.arg(&pi);
                la.arg(&di);
                la.arg(&ci);
                unsafe { la.launch(cfg).map_err(map_err)? };
                Ok(buf.clone())
            }
            crate::backend::TensorInner::CudaI8 { data, scale } => {
                let f = self.function("kv_shift_q8")?;
                let cfg = LaunchConfig::for_num_elems((b * (pos - shift) * d) as u32);
                let mut la = self.stream.launch_builder(&f);
                la.arg(data);
                la.arg(scale);
                la.arg(&bi);
                la.arg(&si);
                la.arg(&pi);
                la.arg(&di);
                la.arg(&ci);
                unsafe { la.launch(cfg).map_err(map_err)? };
                Ok(buf.clone())
            }
            _ => Err(BackendError::Shape(format!(
                "kv_shifted: unsupported storage {:?}",
                buf.dtype
            ))),
        }
    }

    fn kv_rerotated(
        &self,
        k: &Tensor,
        shift: usize,
        head_dim: usize,
        theta: f32,
        mrope_sections: Option<[usize; 4]>,
    ) -> Result<Tensor, BackendError> {
        let (b, cap, d) = (k.shape()[0], k.shape()[1], k.shape()[2]);
        let nheads = d / head_dim;
        let pairs = head_dim / 2;
        let _shape = k.shape().to_vec();
        // The rotation table (hd/2 entries): angle_i = −shift·theta^(−2i/hd).
        let mut cos_tab = vec![0.0f32; pairs];
        let mut sin_tab = vec![0.0f32; pairs];
        for pi in 0..pairs {
            let ang = -(shift as f32) * theta.powf(-2.0 * pi as f32 / head_dim as f32);
            cos_tab[pi] = ang.cos();
            sin_tab[pi] = ang.sin();
        }
        let cos_g = self.from_cpu_f32(&cos_tab, vec![pairs])?;
        let sin_g = self.from_cpu_f32(&sin_tab, vec![pairs])?;
        let total = b * cap * nheads * pairs;
        let (bi, ci, hdi, nhi, si, pai) = (
            b as i32,
            cap as i32,
            head_dim as i32,
            nheads as i32,
            shift as i32,
            pairs as i32,
        );
        let (s0, s1, s2, sect_dims) = mrope_sections
            .map(|s| {
                (
                    s[0] as i32,
                    s[1] as i32,
                    s[2] as i32,
                    (s[0] + s[1] + s[2]) as i32,
                )
            })
            .unwrap_or((0, 0, 0, 0));
        match &k.inner {
            crate::backend::TensorInner::Cuda(_) => {
                let f = self.function("kv_rerotate_f32")?;
                let cfg = LaunchConfig::for_num_elems(total as u32);
                let mut la = self.stream.launch_builder(&f);
                la.arg(k.as_cuda());
                la.arg(cos_g.as_cuda());
                la.arg(sin_g.as_cuda());
                la.arg(&bi);
                la.arg(&ci);
                la.arg(&hdi);
                la.arg(&nhi);
                la.arg(&si);
                la.arg(&pai);
                la.arg(&s0);
                la.arg(&s1);
                la.arg(&s2);
                la.arg(&sect_dims);
                unsafe { la.launch(cfg).map_err(map_err)? };
                Ok(k.clone())
            }
            crate::backend::TensorInner::CudaF16(s) => {
                let f = self.function("kv_rerotate_f16")?;
                let cfg = LaunchConfig::for_num_elems(total as u32);
                let mut la = self.stream.launch_builder(&f);
                la.arg(s);
                la.arg(cos_g.as_cuda());
                la.arg(sin_g.as_cuda());
                la.arg(&bi);
                la.arg(&ci);
                la.arg(&hdi);
                la.arg(&nhi);
                la.arg(&si);
                la.arg(&pai);
                la.arg(&s0);
                la.arg(&s1);
                la.arg(&s2);
                la.arg(&sect_dims);
                unsafe { la.launch(cfg).map_err(map_err)? };
                Ok(k.clone())
            }
            crate::backend::TensorInner::CudaI8 { data, scale } => {
                let f = self.function("kv_rerotate_q8")?;
                let cfg = LaunchConfig::for_num_elems((b * cap * nheads) as u32);
                let mut la = self.stream.launch_builder(&f);
                la.arg(data);
                la.arg(scale);
                la.arg(cos_g.as_cuda());
                la.arg(sin_g.as_cuda());
                la.arg(&bi);
                la.arg(&ci);
                la.arg(&hdi);
                la.arg(&nhi);
                la.arg(&si);
                la.arg(&s0);
                la.arg(&s1);
                la.arg(&s2);
                la.arg(&sect_dims);
                unsafe { la.launch(cfg).map_err(map_err)? };
                Ok(k.clone())
            }
            _ => Err(BackendError::Shape(format!(
                "kv_rerotated: unsupported storage {:?}",
                k.dtype
            ))),
        }
    }

    /// CUDA has no nonzero-offset views (`view_offset` panics by design), so
    /// the row read/write pair copies device-side instead. Both are tiny
    /// (one head row per call) and only the offset differs from Metal.
    fn row_view(&self, x: &Tensor, rows: usize, d: usize, row: usize) -> Result<Tensor, BackendError> {
        let n: usize = x.shape().iter().product();
        if n != rows * d || row >= rows {
            return Err(BackendError::Shape(format!(
                "row_view: {:?} is not ({rows}, {d}) or row {row} out of range",
                x.shape()
            )));
        }
        let crate::backend::TensorInner::Cuda(_) = &x.inner else {
            return Err(BackendError::UnsupportedFormat(
                "row_view: non-CUDA tensor on the CUDA backend".into(),
            ));
        };
        let dst = self.take_pooled(d)?;
        let f = self.function("copy_tail_f32")?;
        let cfg = LaunchConfig::for_num_elems(d as u32);
        let off = (row * d) as i32;
        let d32 = d as i32;
        let mut la = self.stream.launch_builder(&f);
        la.arg(x.as_cuda());
        la.arg(&dst);
        la.arg(&off);
        la.arg(&d32);
        unsafe { la.launch(cfg) }.map_err(map_err)?;
        Ok(Tensor::from_cuda(self.wrap_pooled(dst), vec![1, 1, d]))
    }

    fn row_write(&self, dst: &Tensor, src: &Tensor, row: usize) -> Result<(), BackendError> {
        let d = dst.shape()[2];
        if src.shape() != [1, 1, d] {
            return Err(BackendError::Shape(format!(
                "row_write: src {:?} is not (1, 1, {d})",
                src.shape()
            )));
        }
        let cap = dst.shape()[1];
        if row >= cap {
            return Err(BackendError::Shape(format!(
                "row_write: row {row} outside (1, {cap}, {d})"
            )));
        }
        if let (crate::backend::TensorInner::Cuda(_), crate::backend::TensorInner::Cuda(_)) =
            (&dst.inner, &src.inner)
        {
            let f = self.function("copy_into_f32")?;
            let cfg = LaunchConfig::for_num_elems(d as u32);
            let off = (row * d) as i32;
            let d32 = d as i32;
            let mut la = self.stream.launch_builder(&f);
            la.arg(src.as_cuda());
            la.arg(dst.as_cuda());
            la.arg(&off);
            la.arg(&d32);
            return unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err);
        }
        Err(BackendError::UnsupportedFormat(
            "row_write: non-CUDA tensor on the CUDA backend".into(),
        ))
    }

    fn kv_update(&self, dst: &Tensor, src: &Tensor, pos: usize) -> Result<(), BackendError> {
        let (b, tq, d) = (src.shape()[0], src.shape()[1], src.shape()[2]);
        // K44: the q8_0 KV format stores one f32 scale per 32-element
        // block — a head_dim not divisible by 32 would silently drop the
        // tail elements in kv_update_q8 (the make_kv_tensor q8_0 branch
        // already rejects this; the update path must too).
        if matches!(dst.inner, crate::backend::TensorInner::CudaI8 { .. }) && d % 32 != 0 {
            return Err(BackendError::Shape(format!(
                "kv_update q8_0: head_dim {d} not a multiple of 32"
            )));
        }
        let cap = dst.shape()[1];
        // Mirror of the CPU reference guard: dst (B, cap, D), src (B, Tq, D)
        // — pos + tq must fit within the capacity (the kernel writes rows
        // [pos, pos+tq) of dst; past-cap rows would be OOB) and the batches
        // must match: a mismatched dst batch would let the kernel write past
        // the dst buffer (silent OOB on the GPU, where the CPU slice-copy
        // would panic).
        if dst.shape()[0] != b || dst.shape()[2] != d || pos + tq > cap {
            return Err(BackendError::Shape(format!(
                "kv_update: dst batch {} / dim {} != src batch {b} / dim {d}, or pos {pos} + tq {tq} > cap {cap}",
                dst.shape()[0],
                dst.shape()[2]
            )));
        }
        let n = b * tq * d;
        let cfg = LaunchConfig::for_num_elems(n as u32);
        let k32 = b as i32;
        let t32 = tq as i32;
        let p32 = pos as i32;
        let d32 = d as i32;
        let c32 = cap as i32;
        // P18a: bf16 KV storage has no real kernels — the old (f32→bf16)
        // arm ran the f16 kernel (f16 bits written into a bf16 buffer →
        // garbage on read), (bf16→f16) did a raw 16-bit copy (different
        // exponent layouts) and (bf16→bf16) fell into the generic Err.
        // Fail loudly until a proper __float2bfloat16 kernel exists.
        if matches!(src.inner, crate::backend::TensorInner::CudaBF16(_))
            || matches!(dst.inner, crate::backend::TensorInner::CudaBF16(_))
        {
            return Err(BackendError::UnsupportedFormat(
                "kv_update: bf16 KV storage is not supported on CUDA (no bf16 kernel — the f16 kernels corrupted cross-format data)".into(),
            ));
        }
        // Explicit per-pair branches: writes quantize (f32 → f16/q8_0),
        // grows copy in the same format (f16→f16, q8_0→q8_0).
        match (&src.inner, &dst.inner) {
            (crate::backend::TensorInner::Cuda(_), crate::backend::TensorInner::Cuda(_)) => {
                let f = self.function("kv_update_f32")?;
                let mut la = self.stream.launch_builder(&f);
                la.arg(src.as_cuda());
                la.arg(dst.as_cuda());
                la.arg(&k32);
                la.arg(&t32);
                la.arg(&p32);
                la.arg(&d32);
                la.arg(&c32);
                unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
            }
            (crate::backend::TensorInner::Cuda(_), crate::backend::TensorInner::CudaF16(b)) => {
                let f = self.function("kv_update_f16")?;
                let mut la = self.stream.launch_builder(&f);
                la.arg(src.as_cuda());
                la.arg(b);
                la.arg(&k32);
                la.arg(&t32);
                la.arg(&p32);
                la.arg(&d32);
                la.arg(&c32);
                unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
            }
            (
                crate::backend::TensorInner::Cuda(_),
                crate::backend::TensorInner::CudaI8 { data, scale },
            ) => {
                let f = self.function("kv_update_q8")?;
                let mut la = self.stream.launch_builder(&f);
                la.arg(src.as_cuda());
                la.arg(data);
                la.arg(scale);
                la.arg(&k32);
                la.arg(&t32);
                la.arg(&p32);
                la.arg(&d32);
                la.arg(&c32);
                unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
            }
            (crate::backend::TensorInner::CudaF16(a), crate::backend::TensorInner::CudaF16(b)) => {
                let f = self.function("kv_update_f16_f16")?;
                let mut la = self.stream.launch_builder(&f);
                la.arg(a);
                la.arg(b);
                la.arg(&k32);
                la.arg(&t32);
                la.arg(&p32);
                la.arg(&d32);
                la.arg(&c32);
                unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
            }
            (crate::backend::TensorInner::CudaF16(a), crate::backend::TensorInner::Cuda(_)) => {
                let f = self.function("kv_update_f16_f32")?;
                let mut la = self.stream.launch_builder(&f);
                la.arg(a);
                la.arg(dst.as_cuda());
                la.arg(&k32);
                la.arg(&t32);
                la.arg(&p32);
                la.arg(&d32);
                la.arg(&c32);
                unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
            }
            (
                crate::backend::TensorInner::CudaI8 {
                    data: sa,
                    scale: ss,
                },
                crate::backend::TensorInner::CudaI8 {
                    data: da,
                    scale: ds,
                },
            ) => {
                let f = self.function("kv_update_q8_q8")?;
                let mut la = self.stream.launch_builder(&f);
                la.arg(sa);
                la.arg(ss);
                la.arg(da);
                la.arg(ds);
                la.arg(&k32);
                la.arg(&t32);
                la.arg(&p32);
                la.arg(&d32);
                la.arg(&c32);
                unsafe { la.launch(cfg) }.map(|_| ()).map_err(map_err)
            }
            _ => Err(BackendError::Shape(
                "kv_update: unsupported format pair".to_string(),
            )),
        }
    }

    fn codebook_search(
        &self,
        x: &Tensor,
        table: &Tensor,
        idx: &mut [u32],
    ) -> Result<(), BackendError> {
        let rows = x.shape()[0] * x.shape()[1];
        let d = x.shape()[2];
        let v = table.shape()[0];
        // A2: validate before launching — copy_from_slice below panics on a
        // length mismatch, and a mismatched idx would tear the whole bridge
        // down (the CPU reference returns Err on the same contract). The
        // table width must also match d: the kernel indexes tbl[vi·d + j],
        // so a narrower/wider table reads OOB device memory.
        if table.shape()[1] != d {
            return Err(BackendError::Shape(format!(
                "codebook_search: table dim {} != x dim {d}",
                table.shape()[1]
            )));
        }
        if idx.len() != rows {
            return Err(BackendError::Shape(format!(
                "codebook_search: idx {} != rows {rows}",
                idx.len()
            )));
        }
        let idx_dev = self.stream.alloc_zeros::<u32>(rows).map_err(map_err)?;
        let f = self.function("codebook_search_f32")?;
        // One block per row — the kernel indexes rows via blockIdx.x.
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
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

impl CudaBackend {
    /// Fused-dequant GEMM for packed weights (Q6_K / Q4_0). `nsup_shift`:
    /// weights per packed block (8 → 256, 5 → 32); `blk_bytes`: 210 / 18;
    /// `bn`: column tile (32 Q6_K v2 — dequant-amortized f32 tile fits the
    /// 48 KB static-shared budget; 256 Q4_0).
    /// Kernel family `name` ∈ {gemm_q6k, gemm_q4k}.
    fn gemm_packed_nt(
        &self,
        a: &Tensor,
        bpack: &cudarc::driver::CudaSlice<u8>,
        c: &Tensor,
        m: usize,
        k: usize,
        n: usize,
        nsup_shift: u32,
        blk_bytes: usize,
        bn: usize,
        name: &str,
    ) -> Result<(), BackendError> {
        let k32 = k as i32;
        let n32 = n as i32;
        let m32 = m as i32;
        let c = c.as_cuda_mut();
        let nblk = k >> nsup_shift;
        let grid_x = n.div_ceil(bn);
        // Split-k for m == 1 (decode): the dequant-amortized f32-tile kernel.
        // For m > 1 the tiled kernel stages B once per 32-row chunk and the
        // activation tile in shared (the split-k re-reads B per row AND
        // re-dequants — measured 3-7× slower for prefill). S=1 writes
        // straight into C — no reduce pass.
        let s = if m == 1 {
            self.pick_split(nblk, m, n, grid_x)
        } else {
            1
        };
        // The m=1 split-k follows the activation axis (f16/bf16 cast the A
        // rows — half-width A shared tile and traffic); the m>1 tiled
        // kernel keeps the f32 activation tile (A is amortized in shared
        // there — the axis would buy nothing). The f16/bf16 splitk variants
        // exist only for q6k/q4k — q5k/q3k kernels are f32-only; casting and
        // handing the f16 pointer to the f32 kernel would read garbage (the
        // old dispatch fell through to the f32 arm). Decode A is 1×k, so the
        // cast buys nothing on q5k/q3k anyway.
        let act = self.activation_dtype();
        let a_arg: KernA = if m == 1 {
            match name {
                "gemm_q6k" | "gemm_q4k" | "gemm_q4kc" => match act {
                    crate::dtype::Dtype::F16 => KernA::F16(self.cast_f16(a)?),
                    crate::dtype::Dtype::BF16 => KernA::BF16(self.cast_bf16(a)?),
                    _ => KernA::F32(a),
                },
                _ => KernA::F32(a),
            }
        } else {
            KernA::F32(a)
        };
        let (splitk, tiled): (&'static str, &'static str) = match (&a_arg, name) {
            (KernA::F16(_), "gemm_q6k") => ("gemm_q6k_splitk_v2_f16", "gemm_q6k_tiled_v3"),
            (KernA::BF16(_), "gemm_q6k") => ("gemm_q6k_splitk_v2_bf16", "gemm_q6k_tiled_v3"),
            (KernA::F16(_), "gemm_q4k") => ("gemm_q4k_f16_splitk", "gemm_q4k_f16_tiled"),
            (KernA::BF16(_), "gemm_q4k") => ("gemm_q4k_bf16_splitk", "gemm_q4k_bf16_tiled"),
            (KernA::F16(_), "gemm_q4kc") => ("gemm_q4kc_splitk_v2_f16", "gemm_q4kc_tiled_v3"),
            (KernA::BF16(_), "gemm_q4kc") => ("gemm_q4kc_splitk_v2_bf16", "gemm_q4kc_tiled_v3"),
            (_, "gemm_q6k") => ("gemm_q6k_splitk_v2_f32", "gemm_q6k_tiled_v3"),
            (_, "gemm_q5k") => ("gemm_q5k_splitk_v2_f32", "gemm_q5k_tiled_v3"),
            (_, "gemm_q3k") => ("gemm_q3k_splitk_v2_f32", "gemm_q3k_tiled_v3"),
            (_, "gemm_q4k") => ("gemm_q4k_splitk", "gemm_q4k_tiled"),
            (_, "gemm_q4kc") => ("gemm_q4kc_splitk_v2_f32", "gemm_q4kc_tiled_v3"),
            _ => unreachable!("gemm_packed_nt: name/activation combo"),
        };
        if m == 1 {
            if s > 1 {
                let cpart = self.take_pooled(s * m * n)?;
                let f = self.function(splitk)?;
                let cfg = LaunchConfig {
                    grid_dim: ((grid_x * m) as u32, s as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut la = self.stream.launch_builder(&f);
                match &a_arg {
                    KernA::F32(t) => la.arg(t.as_cuda()),
                    KernA::F16(sl) => la.arg(sl),
                    KernA::BF16(sl) => la.arg(sl),
                };
                la.arg(bpack);
                la.arg(&cpart);
                la.arg(&k32);
                la.arg(&n32);
                let s32 = s as i32;
                la.arg(&s32);
                la.arg(&m32);
                unsafe { la.launch(cfg) }.map_err(map_err)?;
                self.reduce_and_pool(cpart, c, m, n, s, s * m * n)?;
            } else {
                let f = self.function(splitk)?;
                let cfg = LaunchConfig {
                    grid_dim: ((grid_x * m) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut la = self.stream.launch_builder(&f);
                match &a_arg {
                    KernA::F32(t) => la.arg(t.as_cuda()),
                    KernA::F16(sl) => la.arg(sl),
                    KernA::BF16(sl) => la.arg(sl),
                };
                la.arg(bpack);
                la.arg(c);
                la.arg(&k32);
                la.arg(&n32);
                let s32 = 1i32;
                la.arg(&s32);
                la.arg(&m32);
                unsafe { la.launch(cfg) }.map_err(map_err)?;
            }
        } else {
            let s16 = 16usize;
            let cpart = self.take_pooled(s16 * m * n)?;
            let f = self.function(tiled)?;
            let cfg = LaunchConfig {
                grid_dim: (grid_x as u32, m.div_ceil(32) as u32, 16),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut la = self.stream.launch_builder(&f);
            match &a_arg {
                KernA::F32(t) => la.arg(t.as_cuda()),
                KernA::F16(sl) => la.arg(sl),
                KernA::BF16(sl) => la.arg(sl),
            };
            la.arg(bpack);
            la.arg(&cpart);
            la.arg(&m32);
            la.arg(&k32);
            la.arg(&n32);
            unsafe { la.launch(cfg) }.map_err(map_err)?;
            let used = Self::tiled_used_slices(k, nsup_shift);
            self.reduce_and_pool(cpart, c, m, n, used, 16 * m * n)?;
        }
        self.pool_a_scratch(a_arg);
        let _ = blk_bytes; // layout constant — kept for the kernel contract
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tiled GEMMs write ceil(nsup/nkz) of the 16 z-slices; reducing
    /// more would sum stale pool contents (the STT-garbage bug).
    #[test]
    fn tiled_used_slices_math() {
        // k=768 (gigaam/t-one): nsup=24, nkz=2 → 12 slices written.
        assert_eq!(CudaBackend::tiled_used_slices(768, 5), 12);
        // k=3072: nsup=96, nkz=6 → all 16.
        assert_eq!(CudaBackend::tiled_used_slices(3072, 5), 16);
        // k=384: nsup=12, nkz=1 → 12.
        assert_eq!(CudaBackend::tiled_used_slices(384, 5), 12);
        // k=32: nsup=1, nkz=1 → 1.
        assert_eq!(CudaBackend::tiled_used_slices(32, 5), 1);
        // k=16384: nsup=512, nkz=32 → 16.
        assert_eq!(CudaBackend::tiled_used_slices(16384, 5), 16);
        // k=1056: nsup=33, nkz=3 → 11.
        assert_eq!(CudaBackend::tiled_used_slices(1056, 5), 11);
        // Q6_K block (256 elems): k=2048 → nsup=8, nkz=1 → 8.
        assert_eq!(CudaBackend::tiled_used_slices(2048, 8), 8);
        // Q4_0 block (64 elems): k=768 → nsup=12, nkz=1 → 12.
        assert_eq!(CudaBackend::tiled_used_slices(768, 6), 12);
        // nsup == 0 (k below one super-block): must return 1, not divide
        // by zero (the tiled kernels exit immediately; reduce is harmless).
        assert_eq!(CudaBackend::tiled_used_slices(128, 8), 1);
        assert_eq!(CudaBackend::tiled_used_slices(16, 5), 1);
        assert_eq!(CudaBackend::tiled_used_slices(0, 5), 1);
    }
}
