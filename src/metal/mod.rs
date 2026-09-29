//! **Metal engine** — our own Metal compute runtime.
//!
//! Why this exists: existing per-op GPU runtimes commit a separate command
//! buffer **per tensor op** (im2col + gemm + transpose for every conv1d) and
//! synchronise the CPU after each; their tensor buffers are private and
//! kernels are recompiled at runtime per process. For conv-heavy streaming
//! models (t-one conformer, S2-Pro DualAR, DAC codec) that makes real-time
//! inference on Metal impossible.
//!
//! This module is the replacement: our own MSL kernels, our own buffers
//! ([`GpuTensor`]), ONE commit per kernel call and — for a model step — a
//! single command buffer with zero CPU sync between kernels
//! (see [`CommandBatch`]). Everything is fp32 for now; fp16/bf16 kernels
//! land together with the model ports.
//!
//! Candle stays for the CUDA/Linux path and for host-side plumbing.

#![allow(clippy::too_many_arguments)]

use metal::*;
use objc::rc::autoreleasepool;

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Debug, thiserror::Error)]
pub enum MetalError {
    #[error("no Metal device available")]
    NoDevice,
    #[error("metal kernel library compile failed: {0}")]
    LibraryCompile(String),
    #[error("metal kernel `{0}` not found: {1}")]
    KernelNotFound(&'static str, String),
    #[error("metal pipeline `{0}` failed: {1}")]
    Pipeline(&'static str, String),
    #[error("buffer allocation failed")]
    BufferAlloc,
    #[error("shape mismatch: {0}")]
    Shape(String),
}

/// Storage dtype of a [`GpuTensor`]. Activations stay fp32; fp16/bf16 are
/// GEMM weight formats (halved VRAM + weight-read traffic, fp32 accumulate
/// in the kernels — the backend dtype contract). Q4_0/Q8_0/Q4_K/Q6_K —
/// block-quantized GEMM weights (Ф1.3/1.4): dequant inside the kernel, raw
/// block bytes in the buffer (ggml layouts).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuDtype {
    F32,
    F16,
    BF16,
    Q4_0,
    Q8_0,
    Q4_K,
    Q6_K,
}

impl GpuDtype {
    /// Элементов в квант-блоке (для проверки кратности K).
    pub fn block_elems(self) -> Option<usize> {
        match self {
            GpuDtype::Q4_0 | GpuDtype::Q8_0 => Some(32),
            GpuDtype::Q4_K | GpuDtype::Q6_K => Some(256),
            _ => None,
        }
    }
}

/// A contiguous tensor living in Metal memory (fp32 activations, or
/// fp16/bf16 GEMM weights — see [`GpuDtype`]).
/// Глобальный пул MTLBuffer (free-list по классам размера, 4KB-степени).
/// Буферы создаются ВНУТРИ autoreleasepool — autorelease-маркер снимается
/// сразу (refcount 1, Rust-владение), неявные пулы objc больше не могут
/// освободить живую память (источник недетерминизма промежуточных буферов).
pub struct BufferPoolState {
    free: Mutex<HashMap<usize, Vec<Buffer>>>,
    /// Байты в free-листе. Лист раньше рос без предела: в batch-режиме
    /// буферы не переиспользуются (acquire создаёт новые), а после батча все
    /// отложенные уходили в лист — долгая генерация копила память (Qwen3-0.6B:
    /// RSS 3.7 → 11.6 ГБ за 1500 токенов; стресс-приёмка Ф8.3).
    free_bytes: std::cell::Cell<usize>,
}

/// Порог dispatch'ей на command buffer в batch-режиме (см. BatchState).
/// `INFER_METAL_CB_CHUNK` переопределяет; 0 — без порционного коммита.
fn cb_chunk() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        match std::env::var("INFER_METAL_CB_CHUNK").ok().and_then(|v| v.parse::<usize>().ok()) {
            Some(0) => usize::MAX,
            Some(n) => n,
            // Замер (M3 Max, pp128/tg64): 8–32 — плато, 0 (без порций) —
            // prefill на треть медленнее; 16 — середина плато.
            None => 16,
        }
    })
}

/// Предел free-листа пула (байты): сверх него освобождённые буферы
/// отдаются системе, а не копятся. `INFER_METAL_POOL_MB` переопределяет.
fn pool_cap_bytes() -> usize {
    static CAP: OnceLock<usize> = OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("INFER_METAL_POOL_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1024)
            * 1024
            * 1024
    })
}

impl BufferPoolState {
    fn new() -> Self {
        Self {
            free: Mutex::new(HashMap::new()),
            free_bytes: std::cell::Cell::new(0),
        }
    }

    /// Положить буфер в free-лист с учётом предела: не влез — drop (память
    /// возвращается системе; единственный refcount — наш).
    fn put_free(&self, free: &mut HashMap<usize, Vec<Buffer>>, class: usize, buf: Buffer) {
        if self.free_bytes.get() + class > pool_cap_bytes() {
            return;
        }
        self.free_bytes.set(self.free_bytes.get() + class);
        free.entry(class).or_default().push(buf);
    }

    /// Взять буфер размера >= bytes (класс 4KB) или создать (маркер снят).
    fn acquire(&self, device: &Device, bytes: usize) -> Buffer {
        let class = (bytes + 4095) & !4095;
        let class = class.max(4096);
        // Free-лист содержит ТОЛЬКО простаивающие буферы: освобождённые
        // внутри батча уходят в DEFERRED и возвращаются в лист лишь после
        // ожидания завершения батча (end_batch), вне батча каждый кернел
        // ждётся синхронно. Поэтому выдача из листа безопасна и в
        // batch-режиме (CPU-запись в буфер не гоняется с GPU). Раньше в
        // батче пул не использовался вовсе — каждый промежуточный тензор
        // decode-шага был новым MTLBuffer (~4 мс CPU на шаг, больше
        // GPU-работы у малых моделей).
        if std::env::var("OMNI_NO_POOL").is_err()
            && let Some(buf) = self
                .free
                .lock()
                .unwrap()
                .get_mut(&class)
                .and_then(|v| v.pop())
        {
            self.free_bytes.set(self.free_bytes.get().saturating_sub(class));
            return buf;
        }
        // newBuffer* — семейство `new` (Cocoa ownership): объект приходит
        // С УЖЕ ВЗЯТЫМ владением (+1, retainCount 1, без autorelease) —
        // единственный `Buffer` и есть этот refcount, его drop освобождает
        // память. Прежний «трюк» clone + forget добавлял ВТОРОЙ retain,
        // который никто не отпускал: утекал КАЖДЫЙ буфер (стресс Ф8.3: 5 ГБ
        // устройства за 40 шагов; долгая генерация уводила машину в своп и
        // демон получал jetsam-SIGKILL). Проверка — tests/metal_buffer_leak.
        // Новые shared-буферы Metal отдаёт обнулёнными — копия нулей через
        // newBufferWithBytes не нужна.
        device.new_buffer(class as u64, MTLResourceOptions::StorageModeShared)
    }

    /// Вернуть буфер в пул (retain — пул держит единственный refcount).
    /// В batch-режиме буферы НЕ возвращаются в общий пул сразу: их может
    /// использовать НЕВЫПОЛНЕННЫЙ command buffer (в очереди GPU), а другой
    /// поток/тест переиспользовал бы буфер с гонкой CPU↔GPU. Откладываются
    /// в глобальный deferred-список, который возвращается в пул после
    /// закрытия последнего batch (все command buffers выполнены).
    fn push(&self, buf: &Buffer) {
        if batch_depth() > 0 {
            DEFERRED.with(|d| d.borrow_mut().push(buf.clone()));
            return;
        }
        let class = (buf.length() as usize + 4095) & !4095;
        let class = class.max(4096);
        let mut free = self.free.lock().unwrap();
        if std::env::var("OMNI_DUP").is_ok() {
            let p = buf.contents();
            let dup = free
                .get(&class)
                .map(|v| v.iter().filter(|b| b.contents() == p).count())
                .unwrap_or(0);
            if dup > 0 {
                eprintln!("OMNI_DUP: DOUBLE PUSH ptr={p:?} len={}", buf.length());
            }
        }
        self.put_free(&mut free, class, buf.clone());
    }
}

use std::cell::{Cell, RefCell};

thread_local! {
    /// Пул буферов — ПОТОКОЛОКАЛЬНЫЙ. Глобальный пул отдавал буфер,
    /// освобождённый ДРУГИМ потоком, а у потоков разные Metal-очереди: один
    /// поток мог писать (или ставить кернел) в буфер, который ещё
    /// использовался чужим незавершённым command buffer'ом. Прогон тестов в
    /// несколько потоков давал NaN/мусор в логитах (с `OMNI_NO_POOL=1` набор
    /// чистый). Кросс-поточная миграция буферов исключена целиком.
    static POOL_LOCAL: RefCell<BufferPoolState> = RefCell::new(BufferPoolState::new());

    /// Глубина batch-режима ЭТОГО потока: пока > 0 — acquire не
    /// переиспользует буферы пула, а push откладывает их в DEFERRED (они
    /// могут быть заняты невыполненными command buffers).
    static BATCH_DEPTH: Cell<usize> = const { Cell::new(0) };

    /// Буферы, освобождённые внутри batch — возвращаются в пул после
    /// закрытия последнего batch (все command buffers выполнены, wait сделан).
    static DEFERRED: RefCell<Vec<Buffer>> = const { RefCell::new(Vec::new()) };
}

fn with_pool<R>(f: impl FnOnce(&BufferPoolState) -> R) -> R {
    POOL_LOCAL.with(|p| f(&p.borrow()))
}

fn batch_depth() -> usize {
    BATCH_DEPTH.with(|d| d.get())
}

/// Маркер принадлежности буфера пулу: последний клон (все view-клоны
/// тензора дропнуты) возвращает буфер в пул при Drop.
#[derive(Debug, Clone)]
pub(crate) struct PooledToken;

#[derive(Debug, Clone)]
pub struct GpuTensor {
    pub buffer: Buffer,
    pub shape: Vec<usize>,
    pub dtype: GpuDtype,
    /// Byte offset into `buffer` (view into a larger allocation — KV
    /// compact / paged-KV read windows without copies).
    pub offset: usize,
    /// Some(пул) — буфер из пула; возвращается при drop последнего клона.
    /// pub(crate): view-конструкторы в backend.rs/nn (view_inner, view_metal)
    /// обязаны распространять маркер — иначе главный тензор дропнется раньше
    /// живого view, и пул переиспользует буфер под чужие данные.
    pub(crate) pooled: Option<Arc<PooledToken>>,
}

impl Drop for GpuTensor {
    fn drop(&mut self) {
        if let Some(tok) = self.pooled.as_ref() {
            if Arc::strong_count(tok) == 1 {
                with_pool(|p| p.push(&self.buffer));
            }
        }
    }
}

impl GpuTensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Bytes per element for this dtype (contiguous row-major storage).
    pub fn elem_size(&self) -> usize {
        match self.dtype {
            GpuDtype::F32 => 4,
            GpuDtype::F16 | GpuDtype::BF16 => 2,
            GpuDtype::Q8_0 | GpuDtype::Q4_0 => 1,
            GpuDtype::Q4_K | GpuDtype::Q6_K => 1,
        }
    }

    /// View into this tensor starting `elem_offset` elements in (same
    /// buffer, byte offset carried through every kernel dispatch).
    pub fn view_offset(&self, shape: Vec<usize>, elem_offset: usize) -> Self {
        Self {
            buffer: self.buffer.clone(),
            shape,
            dtype: self.dtype,
            offset: self.offset + elem_offset * self.elem_size(),
            pooled: self.pooled.clone(),
        }
    }

    /// Upload fp32 data into a new shared-mode buffer.
    pub fn from_cpu_f32(
        device: &Device,
        data: &[f32],
        shape: Vec<usize>,
    ) -> Result<Self, MetalError> {
        let expected: usize = shape.iter().product();
        assert_eq!(expected, data.len(), "GpuTensor data/shape mismatch");
        let buffer = with_pool(|p| p.acquire(device, data.len() * 4));
        if buffer.contents().is_null() {
            return Err(MetalError::BufferAlloc);
        }
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buffer.contents().cast(), data.len());
        }
        Ok(Self {
            buffer,
            shape,
            dtype: GpuDtype::F32,
            offset: 0,
            pooled: Some(Arc::new(PooledToken)),
        })
    }

    /// Zero-filled buffer of the given shape (fp32).
    pub fn zeros(device: &Device, shape: Vec<usize>) -> Result<Self, MetalError> {
        let bytes = shape.iter().product::<usize>() * 4;
        let buffer = with_pool(|p| p.acquire(device, bytes));
        if buffer.contents().is_null() {
            return Err(MetalError::BufferAlloc);
        }
        unsafe {
            std::ptr::write_bytes(buffer.contents().cast::<u8>(), 0, bytes);
        }
        Ok(Self {
            buffer,
            shape,
            dtype: GpuDtype::F32,
            offset: 0,
            pooled: Some(Arc::new(PooledToken)),
        })
    }

    /// Upload fp16 GEMM weights (2 bytes/element; the gemm kernels decode
    /// and accumulate in fp32).
    pub fn from_cpu_f16(
        device: &Device,
        data: &[half::f16],
        shape: Vec<usize>,
    ) -> Result<Self, MetalError> {
        let expected: usize = shape.iter().product();
        assert_eq!(expected, data.len(), "GpuTensor f16 data/shape mismatch");
        let buffer = with_pool(|p| p.acquire(device, data.len() * 2));
        if buffer.contents().is_null() {
            return Err(MetalError::BufferAlloc);
        }
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buffer.contents().cast(), data.len());
        }
        Ok(Self {
            buffer,
            shape,
            dtype: GpuDtype::F16,
            offset: 0,
            pooled: Some(Arc::new(PooledToken)),
        })
    }

    /// Upload bf16 GEMM weights (2 bytes/element, bit-pattern passthrough).
    pub fn from_cpu_bf16(
        device: &Device,
        data: &[half::bf16],
        shape: Vec<usize>,
    ) -> Result<Self, MetalError> {
        let expected: usize = shape.iter().product();
        assert_eq!(expected, data.len(), "GpuTensor bf16 data/shape mismatch");
        let buffer = with_pool(|p| p.acquire(device, data.len() * 2));
        if buffer.contents().is_null() {
            return Err(MetalError::BufferAlloc);
        }
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buffer.contents().cast(), data.len());
        }
        Ok(Self {
            buffer,
            shape,
            dtype: GpuDtype::BF16,
            offset: 0,
            pooled: Some(Arc::new(PooledToken)),
        })
    }

    /// Upload block-quantized GEMM weights (Ф1.3/1.4): raw ggml-блоки как есть
    /// (Q4_0/Q8_0: 18/34 Б на 32 элемента; Q4_K/Q6_K: 144/210 Б на 256) —
    /// деквантование внутри кернела. Размер данных проверяется по numel.
    ///
    /// Q4_K был 148 Б — это кастомный диалект, не совпадающий с GGUF, из-за
    /// чего КОРРЕКТНЫЕ Q4_K-тензоры отвергались с «quantized data N B,
    /// expected M B» ещё до загрузки в GPU.
    pub fn from_quantized(
        device: &Device,
        data: &[u8],
        shape: Vec<usize>,
        dtype: GpuDtype,
    ) -> Result<Self, MetalError> {
        let bytes_per_elem = match dtype {
            GpuDtype::Q4_0 => 18.0 / 32.0,
            GpuDtype::Q8_0 => 34.0 / 32.0,
            // Canonical ggml block_q4_K = 144 B / 256 weights (the 148 B
            // dialect was removed together with its in-kernel layout).
            GpuDtype::Q4_K => 144.0 / 256.0,
            GpuDtype::Q6_K => 210.0 / 256.0,
            other => {
                return Err(MetalError::Shape(format!(
                    "from_quantized: {other:?} is not a quantized dtype"
                )));
            }
        };
        let numel: usize = shape.iter().product();
        let expected = (numel as f32 * bytes_per_elem).ceil() as usize;
        if data.len() != expected {
            return Err(MetalError::Shape(format!(
                "quantized data {} B, expected {expected} B for {numel} elems ({dtype:?})",
                data.len()
            )));
        }
        let buffer = with_pool(|p| p.acquire(device, data.len()));
        if buffer.contents().is_null() {
            return Err(MetalError::BufferAlloc);
        }
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buffer.contents().cast(), data.len());
        }
        Ok(Self {
            buffer,
            shape,
            dtype,
            offset: 0,
            pooled: Some(Arc::new(PooledToken)),
        })
    }

    /// Download to CPU (shared memory). First commits an EMPTY command
    /// buffer on the same queue the kernels used and waits for it — a fence
    /// that makes every earlier GPU write visible before the copy. The old
    /// contract relied silently on callers having committed+waited; a
    /// missing wait returned stale/partial data.
    pub fn to_cpu_f32(&self, ctx: &MetalCtx) -> Result<Vec<f32>, MetalError> {
        if self.dtype != GpuDtype::F32 {
            return Err(MetalError::Shape(
                "to_cpu_f32 on a quantized (f16/bf16) tensor".into(),
            ));
        }
        // The producer may be inside a batch — the kernel writes are not
        // visible until the pending command buffer is committed+waited.
        ctx.flush_batch()?;
        let barrier = ctx.queue.new_command_buffer();
        barrier.commit();
        barrier.wait_until_completed();
        let ptr = self.buffer.contents();
        if ptr.is_null() {
            return Err(MetalError::BufferAlloc);
        }
        // `offset` — это view внутрь большего буфера (KV-кэш: компакция и
        // срезы передвигают окно внутри той же аллокации). Раньше чтение шло
        // от начала буфера, поэтому to_cpu_f32 на view возвращал ЧУЖИЕ
        // элементы — молча, без ошибки.
        let base = unsafe { (ptr as *const u8).add(self.offset) } as *const f32;
        let mut out = vec![0f32; self.numel()];
        unsafe {
            std::ptr::copy_nonoverlapping(base, out.as_mut_ptr(), out.len());
        }
        Ok(out)
    }

    /// Download TRUE fp16 storage (the f16 KV cache). Mirror of
    /// [`GpuTensor::to_cpu_f32`] with the matching dtype guard: the buffer
    /// holds `__fp16`, so f32 reads would be garbage — the KV roll that needs
    /// the stored keys back re-rotates them through this.
    pub fn to_cpu_f16(&self, ctx: &MetalCtx) -> Result<Vec<half::f16>, MetalError> {
        if self.dtype != GpuDtype::F16 {
            return Err(MetalError::Shape(
                "to_cpu_f16 on a non-f16 tensor".into(),
            ));
        }
        ctx.flush_batch()?;
        let barrier = ctx.queue.new_command_buffer();
        barrier.commit();
        barrier.wait_until_completed();
        let ptr = self.buffer.contents();
        if ptr.is_null() {
            return Err(MetalError::BufferAlloc);
        }
        let base = unsafe { (ptr as *const u8).add(self.offset) } as *const u16;
        let mut out = vec![half::f16::from_bits(0); self.numel()];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = half::f16::from_bits(unsafe { *base.add(i) });
        }
        Ok(out)
    }
}

/// 1D convolution parameters (stride/pad/dilation).
#[derive(Debug, Clone, Copy)]
pub struct Conv1dParams {
    pub stride: usize,
    pub pad: usize,
    pub dilation: usize,
}

/// The Metal context: device, command queue, cached pipelines.
pub struct MetalCtx {
    pub device: Device,
    queue: CommandQueue,
    pipelines: Mutex<HashMap<&'static str, ComputePipelineState>>,
    /// Компилированная библиотека — ОДНА на процесс. Раньше `pipeline(name)`
    /// на каждый новый кернел заново компилировал kernels.metal (290
    /// кернелов — минуты на первый вызов); здесь one-shot: первое имя
    /// компилирует, остальные пайплайны создаются мгновенно.
    library: OnceLock<Library>,
    /// Persistent batch state: while `depth > 0`, every kernel dispatch
    /// goes into ONE command buffer (no per-op commit+wait) and a single
    /// [`MetalCtx::flush_batch`] syncs the GPU. The wake/VAD paths run a
    /// whole window this way — per-op sync made a wake window (~60 small
    /// ops) cost ~97 ms against an 80 ms real-time budget, so the engine
    /// fell behind the mic and the wake word never fired live.
    batch: Mutex<BatchState>,
}

#[derive(Default)]
struct BatchState {
    depth: usize,
    /// Command buffer батча и его ЕДИНСТВЕННЫЙ compute-энкодер (serial
    /// dispatch): все кернелы шага пишутся в один энкодер, коммит — один
    /// на батч (flush/end_batch). Serial-энкодер исполняет dispatch'и по
    /// порядку, а hazard tracking буферов упорядочивает зависимые — ни
    /// барьеров, ни per-kernel command buffer'ов не нужно.
    cmd: Option<CommandBuffer>,
    enc: Option<ComputeCommandEncoder>,
    /// Dispatch'ей в открытом энкодере: по достижении порога (см.
    /// `cb_chunk`) command buffer коммитится без ожидания и открывается
    /// следующий — GPU начинает исполнять шаг, пока CPU кодирует остаток.
    in_enc: usize,
    /// Последний закоммиченный command buffer (flush без ожидания не
    /// делается — поле для совместимости ожидания в end_batch).
    last: Option<CommandBuffer>,
    /// Diagnostics: how many times a batch was opened (tests assert the
    /// wake path actually batches — a regression that drops begin_batch
    /// reintroduces the per-op sync stall).
    enters: u64,
}

impl BatchState {
    /// Закрыть энкодер и закоммитить command buffer батча (если есть);
    /// коммит становится `last` — его ждёт вызывающий.
    fn commit_open(&mut self) {
        self.in_enc = 0;
        if let Some(enc) = self.enc.take() {
            enc.end_encoding();
        }
        if let Some(cmd) = self.cmd.take() {
            cmd.commit();
            self.last = Some(cmd);
        }
    }

    /// Дождаться последнего коммита и проверить статус (Metal может упасть
    /// молча: wait возвращается, а в буферах мусор).
    fn wait_last(&mut self) {
        if let Some(cmd) = self.last.take() {
            cmd.wait_until_completed();
            let status = cmd.status();
            if status != metal::MTLCommandBufferStatus::Completed {
                tracing::error!(
                    status = ?status,
                    "Metal batch command buffer failed — results are garbage"
                );
            }
        }
    }
}

impl Drop for BatchState {
    fn drop(&mut self) {
        // Энкодер, отпущенный без endEncoding, — assert Metal (аварийное
        // завершение процесса): закрываем, даже если батч брошен ошибкой.
        if let Some(enc) = self.enc.take() {
            enc.end_encoding();
        }
    }
}

impl MetalCtx {
    pub fn new() -> Result<Self, MetalError> {
        let device = Device::system_default().ok_or(MetalError::NoDevice)?;
        let queue = device.new_command_queue();
        let ctx = Self {
            device,
            queue,
            pipelines: Mutex::new(HashMap::new()),
            library: OnceLock::new(),
            batch: Mutex::new(BatchState::default()),
        };
        // Compile the whole kernel library once; pipelines are built lazily
        // and cached.
        for name in [
            "gemm_f32",
            "gemm_nt_f32",
            // quantized GEMM (fp32 activations × f16/bf16 weights, f32 acc)
            "gemm_f16",
            "gemm_bf16",
            "gemm_nt_f16",
            "gemm_nt_bf16",
            "conv1d_groups_f32",
            "conv2d_f32",
            "conv1d_direct_f32",
            "mul_f32",
            "repeat_f32",
            "mhsa_f32",
            "rope_head_f32",
            "layernorm_head_f32",
            "softmax_last_dim_f32",
            "rmsnorm_f32",
            "layernorm_f32",
            "silu_f32",
            "relu_f32",
            "sigmoid_f32",
            "add_f32",
            "add_scalar_f32",
            "sub_f32",
            "div_f32",
            "sqrt_f32",
            "log_f32",
            "max_scalar_f32",
            "leaky_relu_clip_f32",
            "conv2d_nchw_f32",
            "maxpool2d_f32",
            "glu_f32",
            "add_bias_f32",
            "add_bias2d_f32",
            "add_scaled_f32",
            "scale_f32",
            "uniform_attn_f32",
            "rope_f32",
            // DAC codec path (fp32 decode/encode)
            "conv_transpose1d_f32",
            "snake_f32",
            "gelu_tanh_f32",
            "embedding_f32",
            "rope_interleaved_f32",
            "mul_vec_f32",
            "tanh_f32",
            "slice_last_f32",
            "slice_time_f32",
            "codebook_search_f32",
            "deepstack_add_f32",
            "attn_ar_f32",
            "kv_update_f32",
            "rmsnorm_head_f32",
            "mul_rows_f32",
        ] {
            ctx.pipeline(name)?;
        }
        Ok(ctx)
    }

    fn pipeline(&self, name: &'static str) -> Result<ComputePipelineState, MetalError> {
        if let Some(p) = self.pipelines.lock().unwrap().get(name) {
            return Ok(p.clone());
        }
        // Библиотека — ОДИН раз на процесс (раньше: компиляция всего
        // kernels.metal на каждый новый кернел → минуты холодного старта).
        let library = self.library.get_or_init(|| {
            let options = CompileOptions::new();
            self.device
                .new_library_with_source(include_str!("kernels.metal"), &options)
                .unwrap_or_else(|e| panic!("metal kernel library compile failed: {e}"))
        });
        let func = library
            .get_function(name, None)
            .map_err(|e| MetalError::KernelNotFound(name, e))?;
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| MetalError::Pipeline(name, e))?;
        self.pipelines
            .lock()
            .unwrap()
            .insert(name, pipeline.clone());
        Ok(pipeline)
    }

    /// Run one kernel with a grid; single commit + wait, inside an autorelease
    /// pool so the borrowed Metal objects are released deterministically.
    /// In batch mode (see [`Self::begin_batch`]) the kernel is appended to
    /// the pending command buffer WITHOUT commit/wait.
    fn run(
        &self,
        kernel: &'static str,
        grid: MTLSize,
        group: MTLSize,
        setup: impl FnOnce(&ComputeCommandEncoderRef),
    ) -> Result<(), MetalError> {
        // Batch path: ВСЕ кернелы шага — в ОДИН serial compute-энкодер
        // одного command buffer'а; коммит+ожидание — один раз в конце батча
        // (или в flush перед чтением на хосте). Раньше каждый кернел
        // получал СВОЙ command buffer (commit на кернел): ~300 коммитов на
        // decode-шаг — оверхед драйвера съедал больше, чем GPU-работа
        // (decode Qwen3-8B 6 tok/s при bandwidth-потолке ~35). Пул отложен
        // (DEFERRED) до закрытия последнего батча — буферы невыполненного
        // command buffer'а не переиспользуются.
        {
            let mut batch = self.batch.lock().unwrap();
            if batch.depth > 0 {
                // Fetch the pipeline BEFORE touching the encoder: a pipeline
                // error must not leave a half-configured dispatch.
                let pipeline = self.pipeline(kernel)?;
                if batch.enc.is_none() {
                    // autoreleasepool: авто-освобождаемые ссылки на новые
                    // объекты сливаются сразу; владение держат to_owned.
                    let (cmd, enc) = autoreleasepool(|| {
                        let cmd = self.queue.new_command_buffer().to_owned();
                        let enc = cmd
                            .compute_command_encoder_with_dispatch_type(MTLDispatchType::Serial)
                            .to_owned();
                        (cmd, enc)
                    });
                    batch.cmd = Some(cmd);
                    batch.enc = Some(enc);
                }
                let enc = batch.enc.as_ref().expect("encoder opened above");
                enc.set_compute_pipeline_state(&pipeline);
                setup(enc);
                enc.dispatch_threads(grid, group);
                batch.in_enc += 1;
                // Порционный коммит: с одним command buffer'ом на батч GPU
                // простаивал, пока CPU кодировал ВЕСЬ шаг (prefill Qwen3-8B —
                // 50 мс CPU из 375). Очередь исполняет command buffer'ы по
                // порядку коммита, отслеживание зависимостей буферов сохраняет
                // корректность между ними.
                if batch.in_enc >= cb_chunk() {
                    batch.commit_open();
                }
                return Ok(());
            }
        }
        autoreleasepool(|| {
            let pipeline = self.pipeline(kernel)?;
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            setup(enc);
            enc.dispatch_threads(grid, group);
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            let status = cmd.status();
            if status != metal::MTLCommandBufferStatus::Completed {
                tracing::error!(kernel, status = ?status, "Metal kernel failed");
            }
            Ok(())
        })
    }

    /// Enter batch mode (nestable). Ops accumulate until the matching
    /// [`Self::end_batch`] — or an explicit [`Self::flush_batch`].
    pub fn begin_batch(&self) {
        let mut batch = self.batch.lock().unwrap();
        batch.depth += 1;
        batch.enters += 1;
        BATCH_DEPTH.with(|d| d.set(d.get() + 1));
    }

    /// How many batches were opened since the context was created
    /// (diagnostics/tests).
    pub fn batch_enter_count(&self) -> u64 {
        self.batch.lock().unwrap().enters
    }

    /// Leave batch mode; on the outermost exit, commit+wait the pending
    /// command buffer once.
    pub fn end_batch(&self) -> Result<(), MetalError> {
        let mut batch = self.batch.lock().unwrap();
        if batch.depth > 0 {
            batch.depth -= 1;
            BATCH_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
        }
        if batch.depth == 0 {
            batch.commit_open();
            batch.wait_last();
            // Последний batch закрыт — все command buffers выполнены:
            // возвращаем отложенные буферы в пул СВОЕГО потока
            // (см. BufferPoolState::push).
            if batch_depth() == 0 {
                let deferred = DEFERRED.with(|d| std::mem::take(&mut *d.borrow_mut()));
                with_pool(|p| {
                    let mut free = p.free.lock().unwrap();
                    for buf in deferred {
                        let class = (buf.length() as usize + 4095) & !4095;
                        let class = class.max(4096);
                        p.put_free(&mut free, class, buf);
                    }
                });
            }
        }
        Ok(())
    }

    /// Commit+wait the pending batch WITHOUT leaving batch mode. Required
    /// before any host-side read ([`GpuTensor::to_cpu_f32`]) — the shared
    /// buffer's contents are only valid after the GPU finished.
    ///
    /// A Metal command buffer can fail SILENTLY: waitUntilCompleted returns
    /// even when the buffer's status is Error, and the buffers hold stale
    /// garbage. The wake engine's batch runs ~60 kernels — one failed kernel
    /// poisons the whole window's mel/embedding/phrase, which shows up as a
    /// CONSTANT near-zero phrase probability on any input (the live symptom
    /// 2026-08-22: offline 0.35 on the same audio, daemon 0.0001). Log the
    /// status + error when it happens instead of shipping the garbage.
    pub fn flush_batch(&self) -> Result<(), MetalError> {
        let mut batch = self.batch.lock().unwrap();
        batch.commit_open();
        batch.wait_last();
        Ok(())
    }

    // ── ops ──

    /// C = A·B (all row-major). Shapes: A (M,K), B (K,N), C (M,N).
    /// B may carry fp16/bf16 GEMM weights — the kernel narrows the weight
    /// reads to 2 bytes/element and accumulates in fp32 (A/C stay fp32).
    pub fn gemm_f32(&self, a: &GpuTensor, b: &GpuTensor, c: &GpuTensor) -> Result<(), MetalError> {
        let m = a.shape[0];
        let k = a.shape[1];
        let n = b.shape[1];
        if a.shape[1] != b.shape[0] || c.shape != [m, n] {
            return Err(MetalError::Shape(format!(
                "gemm {:?}x{:?}->{:?}",
                a.shape, b.shape, c.shape
            )));
        }
        let name = match b.dtype {
            GpuDtype::F32 => "gemm_f32",
            GpuDtype::F16 => "gemm_f16",
            GpuDtype::BF16 => "gemm_bf16",
            // Квантованные веса — только через gemm_nt_quantized/mv_quantized.
            GpuDtype::Q4_0 | GpuDtype::Q8_0 | GpuDtype::Q4_K | GpuDtype::Q6_K => {
                return Err(MetalError::Shape(
                    "gemm_f32: quantized weights not supported (use gemm_nt_quantized)".into(),
                ));
            }
        };
        // decode (m=1): векторизованный weight-stationary кернел (float4
        // по n, half4-чтения) — в разы быстрее полного gemm с grid (1, N).
        if m == 1 && b.dtype == GpuDtype::F16 {
            let n4 = n.div_ceil(4);
            let grid = MTLSize::new(1, n4 as u64, 1);
            let group = MTLSize::new(1, 64, 1);
            self.run("mv_f16_f4", grid, group, |enc| {
                enc.set_buffer(0, Some(&a.buffer), a.offset as u64);
                enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
                enc.set_buffer(2, Some(&c.buffer), c.offset as u64);
                enc.set_bytes(3, 8, (&m as *const usize).cast());
                enc.set_bytes(4, 8, (&k as *const usize).cast());
                enc.set_bytes(5, 8, (&n as *const usize).cast());
            })?;
            return Ok(());
        }
        if b.dtype == GpuDtype::Q8_0 && m > 1 {
            let grid = MTLSize::new(n as u64, m.div_ceil(8) as u64, 1);
            let group = MTLSize::new(256, 1, 1);
            return self.run("gemm_nt_q8_0_m8", grid, group, |enc| {
                enc.set_buffer(0, Some(&a.buffer), a.offset as u64);
                enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
                enc.set_buffer(2, Some(&c.buffer), c.offset as u64);
                enc.set_bytes(3, 8, (&m as *const usize).cast());
                enc.set_bytes(4, 8, (&k as *const usize).cast());
                enc.set_bytes(5, 8, (&n as *const usize).cast());
            });
        }
        let grid = MTLSize::new(m as u64, n as u64, 1);
        let group = MTLSize::new(8, 8, 1);
        self.run(name, grid, group, |enc| {
            enc.set_buffer(0, Some(&a.buffer), a.offset as u64);
            enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
            enc.set_buffer(2, Some(&c.buffer), c.offset as u64);
            enc.set_bytes(3, 8, (&m as *const usize).cast());
            enc.set_bytes(4, 8, (&k as *const usize).cast());
            enc.set_bytes(5, 8, (&n as *const usize).cast());
        })
    }

    /// Direct 1D convolution. X (B,T,Cin), W (Cout,Cin,K), bias (Cout) or None,
    /// Y (B,Tout,Cout).
    pub fn conv1d_f32(
        &self,
        x: &GpuTensor,
        w: &GpuTensor,
        bias: Option<&GpuTensor>,
        params: Conv1dParams,
        y: &GpuTensor,
    ) -> Result<(), MetalError> {
        let b = x.shape[0];
        let t = x.shape[1];
        let cin = x.shape[2];
        let cout = w.shape[0];
        let k = w.shape[2];
        let t_out = (t + 2 * params.pad - params.dilation * (k - 1) - 1) / params.stride + 1;
        if y.shape != [b, t_out, cout] {
            return Err(MetalError::Shape(format!(
                "conv1d out {:?} != [{b},{t_out},{cout}]",
                y.shape
            )));
        }
        let grid = MTLSize::new(b as u64, t_out as u64, cout as u64);
        let group = MTLSize::new(1, 4, 4);
        self.run("conv1d_direct_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&w.buffer), w.offset as u64);
            enc.set_buffer(
                2,
                bias.map(|b| &*b.buffer),
                bias.map(|b| b.offset as u64).unwrap_or(0),
            );
            enc.set_buffer(3, Some(&y.buffer), y.offset as u64);
            for (i, v) in [t, cin, cout, k].iter().enumerate() {
                enc.set_bytes((4 + i) as u64, 8, (v as *const usize).cast());
            }
            for (i, v) in [params.stride, params.pad, params.dilation]
                .iter()
                .enumerate()
            {
                enc.set_bytes((8 + i) as u64, 8, (v as *const usize).cast());
            }
        })
    }

    /// im2col for a dense 1D conv (groups=1): X (B,T,Cin) → Xc (B,Tout,Cin·K),
    /// Xc[t, ci·K+kk] = X[t·stride + kk·dil − pad_l, ci] (zero outside).
    pub fn im2col1d_f32(
        &self,
        x: &GpuTensor,
        xc: &GpuTensor,
        k: usize,
        stride: usize,
        pad_l: usize,
        dil: usize,
        tout: usize,
    ) -> Result<(), MetalError> {
        let b = x.shape[0];
        let t = x.shape[1];
        let cin = x.shape[2];
        if xc.shape != [b, tout, cin * k] {
            return Err(MetalError::Shape(format!(
                "im2col1d out {:?} != [{b},{tout},{}]",
                xc.shape,
                cin * k
            )));
        }
        let grid = MTLSize::new(b as u64, tout as u64, (cin * k) as u64);
        let group = MTLSize::new(1, 1, 1);
        self.run("im2col1d_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&xc.buffer), xc.offset as u64);
            for (i, v) in [t, cin, k].iter().enumerate() {
                enc.set_bytes((2 + i) as u64, 8, (v as *const usize).cast());
            }
            for (i, v) in [stride, pad_l, dil, tout].iter().enumerate() {
                enc.set_bytes((5 + i) as u64, 8, (v as *const usize).cast());
            }
        })
    }

    /// im2col for a transposed 1D conv: X (B,Tin,Cin) → Xc (B,Tout,Cin·K),
    /// Xc[t, ci·K+kk] = X[(t−kk)/stride, ci] when divisible, else 0.
    pub fn im2col_t1d_f32(
        &self,
        x: &GpuTensor,
        xc: &GpuTensor,
        k: usize,
        stride: usize,
        tout: usize,
    ) -> Result<(), MetalError> {
        let b = x.shape[0];
        let tin = x.shape[1];
        let cin = x.shape[2];
        if xc.shape != [b, tout, cin * k] {
            return Err(MetalError::Shape(format!(
                "im2col_t1d out {:?} != [{b},{tout},{}]",
                xc.shape,
                cin * k
            )));
        }
        let grid = MTLSize::new(b as u64, tout as u64, (cin * k) as u64);
        let group = MTLSize::new(1, 1, 1);
        self.run("im2col_t1d_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&xc.buffer), xc.offset as u64);
            for (i, v) in [tin, cin, k].iter().enumerate() {
                enc.set_bytes((2 + i) as u64, 8, (v as *const usize).cast());
            }
            for (i, v) in [stride, tout].iter().enumerate() {
                enc.set_bytes((5 + i) as u64, 8, (v as *const usize).cast());
            }
        })
    }

    /// Softmax over the last dimension. X/Y same shape (rows, cols).
    pub fn softmax_last_dim_f32(&self, x: &GpuTensor, y: &GpuTensor) -> Result<(), MetalError> {
        let rows = x.shape[0];
        let cols = x.shape[1];
        let grid = MTLSize::new(rows as u64, 1, 1);
        let group = MTLSize::new(1, 1, 1);
        self.run("softmax_last_dim_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&y.buffer), y.offset as u64);
            enc.set_bytes(2, 8, (&rows as *const usize).cast());
            enc.set_bytes(3, 8, (&cols as *const usize).cast());
        })
    }

    /// SiLU: y[i] = silu(x[i]) (same shape, may alias).
    pub fn silu_f32(&self, x: &GpuTensor, y: &GpuTensor) -> Result<(), MetalError> {
        let n = x.numel();
        let grid = MTLSize::new(n as u64, 1, 1);
        let group = MTLSize::new(256, 1, 1);
        self.run("silu_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&y.buffer), y.offset as u64);
            enc.set_bytes(2, 8, (&n as *const usize).cast());
        })
    }

    /// Fused SwiGLU gate: Y[i] = silu(X[i]) * X[FFN + i]; X [rows, 2*FFN].
    pub fn silu_mul_fused_f32(
        &self,
        x: &GpuTensor,
        y: &GpuTensor,
        ffn: usize,
    ) -> Result<(), MetalError> {
        let n = y.numel();
        let grid = MTLSize::new(n as u64, 1, 1);
        let group = MTLSize::new(256, 1, 1);
        self.run("silu_mul_fused_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&y.buffer), y.offset as u64);
            enc.set_bytes(2, 8, (&n as *const usize).cast());
            enc.set_bytes(3, 8, (&ffn as *const usize).cast());
        })
    }

    /// DFlash2 two-tap dynamic conv (спека PR #27342): см. backend.rs.
    /// Grid (t, i) — каждый поток считает свой выходной элемент.
    pub fn dflash2_conv_f32(
        &self,
        x: &GpuTensor,
        base: &GpuTensor,
        dynamic: &GpuTensor,
        y: &GpuTensor,
        kernel: usize,
        group_size: usize,
        block_size: usize,
        side: usize,
    ) -> Result<(), MetalError> {
        let (n_tokens, n_embd) = (y.shape[0], y.shape[1]);
        let n_groups = n_embd / group_size;
        let grid = MTLSize::new(n_tokens as u64, n_embd as u64, 1);
        let group = MTLSize::new(1, 256, 1);
        self.run("dflash2_conv_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&base.buffer), base.offset as u64);
            enc.set_buffer(2, Some(&dynamic.buffer), dynamic.offset as u64);
            enc.set_buffer(3, Some(&y.buffer), y.offset as u64);
            enc.set_bytes(4, 8, (&n_tokens as *const usize).cast());
            enc.set_bytes(5, 8, (&n_embd as *const usize).cast());
            enc.set_bytes(6, 8, (&kernel as *const usize).cast());
            enc.set_bytes(7, 8, (&group_size as *const usize).cast());
            enc.set_bytes(8, 8, (&block_size as *const usize).cast());
            enc.set_bytes(9, 8, (&n_groups as *const usize).cast());
            enc.set_bytes(10, 8, (&side as *const usize).cast());
        })
    }

    /// ReLU: y[i] = max(x[i], 0) (same shape, may alias).
    pub fn relu_f32(&self, x: &GpuTensor, y: &GpuTensor) -> Result<(), MetalError> {
        let n = x.numel();
        let grid = MTLSize::new(n as u64, 1, 1);
        let group = MTLSize::new(256, 1, 1);
        self.run("relu_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&y.buffer), y.offset as u64);
            enc.set_bytes(2, 8, (&n as *const usize).cast());
        })
    }

    /// Sigmoid: y[i] = 1/(1+exp(-x[i])) (same shape, may alias).
    pub fn sigmoid_f32(&self, x: &GpuTensor, y: &GpuTensor) -> Result<(), MetalError> {
        let n = x.numel();
        let grid = MTLSize::new(n as u64, 1, 1);
        let group = MTLSize::new(256, 1, 1);
        self.run("sigmoid_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&y.buffer), y.offset as u64);
            enc.set_bytes(2, 8, (&n as *const usize).cast());
        })
    }

    /// Elementwise add: Y = A + B (same shape; may alias A or B).
    pub fn add_f32(&self, a: &GpuTensor, b: &GpuTensor, y: &GpuTensor) -> Result<(), MetalError> {
        let n = a.numel();
        let grid = MTLSize::new(n as u64, 1, 1);
        let group = MTLSize::new(256, 1, 1);
        self.run("add_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&a.buffer), a.offset as u64);
            enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
            enc.set_buffer(2, Some(&y.buffer), y.offset as u64);
            enc.set_bytes(3, 8, (&n as *const usize).cast());
        })
    }

    /// Elementwise add scalar: Y = X + c (same shape, may alias).
    pub fn add_scalar_f32(&self, x: &GpuTensor, y: &GpuTensor, c: f32) -> Result<(), MetalError> {
        let n = x.numel();
        let grid = MTLSize::new(n as u64, 1, 1);
        let group = MTLSize::new(256, 1, 1);
        self.run("add_scalar_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&y.buffer), y.offset as u64);
            enc.set_bytes(2, 8, (&n as *const usize).cast());
            enc.set_bytes(3, 4, (&c as *const f32).cast());
        })
    }

    /// Elementwise subtract: Y = A - B (same shape; may alias A or B).
    pub fn sub_f32(&self, a: &GpuTensor, b: &GpuTensor, y: &GpuTensor) -> Result<(), MetalError> {
        let n = a.numel();
        let grid = MTLSize::new(n as u64, 1, 1);
        let group = MTLSize::new(256, 1, 1);
        self.run("sub_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&a.buffer), a.offset as u64);
            enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
            enc.set_buffer(2, Some(&y.buffer), y.offset as u64);
            enc.set_bytes(3, 8, (&n as *const usize).cast());
        })
    }

    /// Elementwise divide: Y = A / B (same shape; may alias A or B).
    pub fn div_f32(&self, a: &GpuTensor, b: &GpuTensor, y: &GpuTensor) -> Result<(), MetalError> {
        let n = a.numel();
        let grid = MTLSize::new(n as u64, 1, 1);
        let group = MTLSize::new(256, 1, 1);
        self.run("div_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&a.buffer), a.offset as u64);
            enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
            enc.set_buffer(2, Some(&y.buffer), y.offset as u64);
            enc.set_bytes(3, 8, (&n as *const usize).cast());
        })
    }

    /// Elementwise sqrt (same shape, may alias).
    pub fn sqrt_f32(&self, x: &GpuTensor, y: &GpuTensor) -> Result<(), MetalError> {
        let n = x.numel();
        let grid = MTLSize::new(n as u64, 1, 1);
        let group = MTLSize::new(256, 1, 1);
        self.run("sqrt_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&y.buffer), y.offset as u64);
            enc.set_bytes(2, 8, (&n as *const usize).cast());
        })
    }

    /// Elementwise log (same shape, may alias).
    pub fn log_f32(&self, x: &GpuTensor, y: &GpuTensor) -> Result<(), MetalError> {
        let n = x.numel();
        let grid = MTLSize::new(n as u64, 1, 1);
        let group = MTLSize::new(256, 1, 1);
        self.run("log_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&y.buffer), y.offset as u64);
            enc.set_bytes(2, 8, (&n as *const usize).cast());
        })
    }

    /// Elementwise max with a scalar floor: Y = max(X, floor).
    pub fn max_scalar_f32(
        &self,
        x: &GpuTensor,
        y: &GpuTensor,
        floor: f32,
    ) -> Result<(), MetalError> {
        let n = x.numel();
        let grid = MTLSize::new(n as u64, 1, 1);
        let group = MTLSize::new(256, 1, 1);
        self.run("max_scalar_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&y.buffer), y.offset as u64);
            enc.set_bytes(2, 8, (&n as *const usize).cast());
            enc.set_bytes(3, 4, (&floor as *const f32).cast());
        })
    }

    /// Clipped leaky ReLU: Y = max(leaky(x, 0.2), -0.4).
    pub fn leaky_relu_clip_f32(&self, x: &GpuTensor, y: &GpuTensor) -> Result<(), MetalError> {
        let n = x.numel();
        let grid = MTLSize::new(n as u64, 1, 1);
        let group = MTLSize::new(256, 1, 1);
        self.run("leaky_relu_clip_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&y.buffer), y.offset as u64);
            enc.set_bytes(2, 8, (&n as *const usize).cast());
        })
    }

    /// 2D convolution (NCHW). X (N,Ci,Hi,Wi), W (Co,Ci,KH,KW), bias (Co),
    /// 2D conv (NCHW) with strides; `time_major_out` produces Y (N,OH,Co·OW)
    /// — the t-one subsampling layout. LEFT padding (CUDA/conv1d convention):
    /// OH = (Hi + pad_h − KH)/stride_t + 1.
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_nchw_f32(
        &self,
        x: &GpuTensor,
        w: &GpuTensor,
        bias: Option<&GpuTensor>,
        pad_h: usize,
        pad_w: usize,
        stride_t: usize,
        stride_f: usize,
        time_major_out: bool,
        y: &GpuTensor,
    ) -> Result<(), MetalError> {
        let (n, ci, hi, wi) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
        let (co, _, kh, kw) = (w.shape[0], w.shape[1], w.shape[2], w.shape[3]);
        let oh = (hi + pad_h - kh) / stride_t + 1;
        let ow = (wi + pad_w - kw) / stride_f + 1;
        let grid = MTLSize::new((oh * ow) as u64, n as u64, co as u64);
        let group = MTLSize::new(1, 1, 1);
        self.run("conv2d_nchw_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&w.buffer), w.offset as u64);
            enc.set_buffer(
                2,
                bias.map(|b| &*b.buffer),
                bias.map(|b| b.offset as u64).unwrap_or(0),
            );
            enc.set_buffer(3, Some(&y.buffer), y.offset as u64);
            let consts = [
                n,
                ci,
                hi,
                wi,
                co,
                kh,
                kw,
                oh,
                ow,
                pad_h,
                pad_w,
                stride_t,
                stride_f,
                time_major_out as usize,
            ];
            for (i, v) in consts.iter().enumerate() {
                enc.set_bytes((4 + i) as u64, 8, (v as *const usize).cast());
            }
        })
    }

    /// MaxPool2d (kernel == stride, no padding). X/Y (N,C,H,W).
    pub fn maxpool2d_f32(
        &self,
        x: &GpuTensor,
        y: &GpuTensor,
        kh: usize,
        kw: usize,
    ) -> Result<(), MetalError> {
        let (n, c, hi, wi) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
        let (oh, ow) = (y.shape[2], y.shape[3]);
        let grid = MTLSize::new((oh * ow) as u64, n as u64, c as u64);
        let group = MTLSize::new(1, 1, 1);
        self.run("maxpool2d_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&y.buffer), y.offset as u64);
            for (i, v) in [n, c, hi, wi, oh, ow, kh, kw].iter().enumerate() {
                enc.set_bytes((2 + i) as u64, 8, (v as *const usize).cast());
            }
        })
    }

    /// Broadcast-add bias on the last dim: Y (B,n) = X (B,n) + bias (n).
    pub fn add_bias_f32(
        &self,
        x: &GpuTensor,
        bias: &GpuTensor,
        y: &GpuTensor,
    ) -> Result<(), MetalError> {
        // Последняя ось = cols (2D (rows,cols) или 3D (B,T,D) — row-major
        // идентичен (rows,cols)); rows = numel / cols (тот же класс бага,
        // что layernorm_f32 — 3D обрабатывал только первую строку).
        let cols = x.shape[x.shape.len() - 1];
        let rows = x.numel() / cols;
        let grid = MTLSize::new(cols as u64, rows as u64, 1);
        let group = MTLSize::new(8, 8, 1);
        self.run("add_bias_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&bias.buffer), bias.offset as u64);
            enc.set_buffer(2, Some(&y.buffer), y.offset as u64);
            enc.set_bytes(3, 8, (&cols as *const usize).cast());
        })
    }

    /// Rotary embedding (halved, with pass-through): X (T, head_dim),
    /// cos/sin (T, rope_dim/2); only the first `rope_dim` dims are rotated.
    pub fn rope_f32(
        &self,
        x: &GpuTensor,
        cos_buf: &GpuTensor,
        sin_buf: &GpuTensor,
        y: &GpuTensor,
        rope_dim: usize,
    ) -> Result<(), MetalError> {
        let t = x.shape[0];
        let head_dim = x.shape[1];
        let grid = MTLSize::new(t as u64, head_dim as u64, 1);
        let group = MTLSize::new(1, 8, 1);
        self.run("rope_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&cos_buf.buffer), cos_buf.offset as u64);
            enc.set_buffer(2, Some(&sin_buf.buffer), sin_buf.offset as u64);
            enc.set_buffer(3, Some(&y.buffer), y.offset as u64);
            enc.set_bytes(4, 8, (&t as *const usize).cast());
            enc.set_bytes(5, 8, (&head_dim as *const usize).cast());
            enc.set_bytes(6, 8, (&rope_dim as *const usize).cast());
        })
    }

    /// C = A·Bᵀ: A (M,K), B (N,K), C (M,N) — for y = x @ Wᵀ with W stored
    /// row-major [N,K].
    pub fn gemm_nt_f32(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        c: &GpuTensor,
    ) -> Result<(), MetalError> {
        let m = a.shape[0];
        let k = a.shape[1];
        let n = b.shape[0];
        if a.shape[1] != b.shape[1] || c.shape != [m, n] {
            return Err(MetalError::Shape(format!(
                "gemm_nt {:?}x{:?}->{:?}",
                a.shape, b.shape, c.shape
            )));
        }
        // B (the weight) may be stored fp16/bf16 — narrowed reads, fp32
        // accumulate; A/C stay fp32.
        let name = match b.dtype {
            GpuDtype::F32 => "gemm_nt_f32",
            GpuDtype::F16 => "gemm_nt_f16",
            GpuDtype::BF16 => "gemm_nt_bf16",
            // Квантованные веса — только через gemm_nt_quantized/mv_quantized.
            GpuDtype::Q4_0 | GpuDtype::Q8_0 | GpuDtype::Q4_K | GpuDtype::Q6_K => {
                return Err(MetalError::Shape(
                    "gemm_nt_f32: quantized weights not supported (use gemm_nt_quantized)".into(),
                ));
            }
        };
        let grid = MTLSize::new(m as u64, n as u64, 1);
        let group = MTLSize::new(8, 8, 1);
        self.run(name, grid, group, |enc| {
            enc.set_buffer(0, Some(&a.buffer), a.offset as u64);
            enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
            enc.set_buffer(2, Some(&c.buffer), c.offset as u64);
            enc.set_bytes(3, 8, (&m as *const usize).cast());
            enc.set_bytes(4, 8, (&k as *const usize).cast());
            enc.set_bytes(5, 8, (&n as *const usize).cast());
        })
    }

    /// C = A·Bᵀ с КВАНТОВАННЫМИ весами B [N, K] (Ф1.3): dequant-in-kernel,
    /// fp32-аккумуляция. Требует K, кратного блоку формата (32/256).
    pub fn gemm_nt_quantized(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        c: &GpuTensor,
    ) -> Result<(), MetalError> {
        let m = a.shape[0];
        let k = a.shape[1];
        let n = b.shape[0];
        if a.shape[1] != b.shape[1] || c.shape != [m, n] {
            return Err(MetalError::Shape(format!(
                "gemm_nt_quantized {:?}x{:?}->{:?}",
                a.shape, b.shape, c.shape
            )));
        }
        let name = match b.dtype {
            GpuDtype::Q4_0 => "gemm_nt_q4_0",
            GpuDtype::Q8_0 => "gemm_nt_q8_0",
            GpuDtype::Q4_K => "gemm_nt_q4_k",
            GpuDtype::Q6_K => "gemm_nt_q6_k",
            other => {
                return Err(MetalError::Shape(format!(
                    "gemm_nt_quantized: B dtype {other:?} is not quantized"
                )));
            }
        };
        let block = b
            .dtype
            .block_elems()
            .ok_or_else(|| MetalError::Shape("quantized block size unavailable".into()))?;
        if !k.is_multiple_of(block) {
            return Err(MetalError::Shape(format!(
                "gemm_nt_quantized: K={k} not a multiple of block {block} ({:?})",
                b.dtype
            )));
        }
        // m>1, Q8_0: тайловый GEMM на матричных блоках simdgroup (prefill и
        // верификация спека). INFER_GEMM_Q8_LEGACY=1 — прежний m8f4 (сверка).
        if m > 1 && b.dtype == GpuDtype::Q8_0 && std::env::var("INFER_GEMM_Q8_LEGACY").is_err() {
            let (m32, k32, n32) = (m as u32, k as u32, n as u32);
            // Схема ggml kernel_mul_mm: threadgroup 128 потоков на тайл
            // 32 (M) × 64 (N); grid — (M-тайлы, N-тайлы) threadgroup'ов.
            let grid = MTLSize::new((m.div_ceil(32) * 128) as u64, n.div_ceil(64) as u64, 1);
            let group = MTLSize::new(128, 1, 1);
            return self.run("gemm_nt_q8_0_mm", grid, group, |enc| {
                enc.set_buffer(0, Some(&a.buffer), a.offset as u64);
                enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
                enc.set_buffer(2, Some(&c.buffer), c.offset as u64);
                enc.set_bytes(3, 4, (&m32 as *const u32).cast());
                enc.set_bytes(4, 4, (&k32 as *const u32).cast());
                enc.set_bytes(5, 4, (&n32 as *const u32).cast());
                // 8 КБ: тайлы весов/активаций (4+2 КБ) и staging хвостового
                // тайла выхода (32×64 float).
                enc.set_threadgroup_memory_length(0, 8192);
            });
        }
        // m>1: m8-кернелы (n-столбец × 8 m-строк, явные аккумуляторы) —
        // B читается 1 раз на 8 строк; F16 — half4-чтения.
        if m > 1 {
            let (kname, is_f16) = match b.dtype {
                GpuDtype::Q8_0 => ("gemm_nt_q8_0_m8f4", false),
                GpuDtype::F16 => ("gemm_nt_f16_m8", true),
                _ => ("", false),
            };
            if !kname.is_empty() {
                let grid = MTLSize::new(n as u64, m.div_ceil(8) as u64, 1);
                let group = MTLSize::new(256, 1, 1);
                return self.run(kname, grid, group, |enc| {
                    enc.set_buffer(0, Some(&a.buffer), a.offset as u64);
                    enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
                    enc.set_buffer(2, Some(&c.buffer), c.offset as u64);
                    enc.set_bytes(3, 8, (&m as *const usize).cast());
                    enc.set_bytes(4, 8, (&k as *const usize).cast());
                    enc.set_bytes(5, 8, (&n as *const usize).cast());
                });
            }
        }
        let grid = MTLSize::new(m as u64, n as u64, 1);
        let group = MTLSize::new(8, 8, 1);
        self.run(name, grid, group, |enc| {
            enc.set_buffer(0, Some(&a.buffer), a.offset as u64);
            enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
            enc.set_buffer(2, Some(&c.buffer), c.offset as u64);
            enc.set_bytes(3, 8, (&m as *const usize).cast());
            enc.set_bytes(4, 8, (&k as *const usize).cast());
            enc.set_bytes(5, 8, (&n as *const usize).cast());
        })
    }

    /// MV (decode, batch=1, Ф1.4): C[n] = Σ_k A[k]·B[n][k], B квантованный
    /// [N, K]; A — вектор [K]. thread на выходной n.
    pub fn mv_quantized(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        c: &GpuTensor,
    ) -> Result<(), MetalError> {
        let k = a.shape[0];
        let n = b.shape[0];
        if a.shape.len() != 1 || b.shape.len() != 2 || b.shape[1] != k || c.shape != [n] {
            return Err(MetalError::Shape(format!(
                "mv_quantized {:?}x{:?}->{:?}",
                a.shape, b.shape, c.shape
            )));
        }
        let name = match b.dtype {
            GpuDtype::Q4_0 => "mv_q4_0",
            GpuDtype::Q8_0 => "mv_q8_0",
            GpuDtype::Q4_K => "mv_q4_k",
            GpuDtype::Q6_K => "mv_q6_k",
            other => {
                return Err(MetalError::Shape(format!(
                    "mv_quantized: B dtype {other:?} is not quantized"
                )));
            }
        };
        let block = b
            .dtype
            .block_elems()
            .ok_or_else(|| MetalError::Shape("quantized block size unavailable".into()))?;
        if !k.is_multiple_of(block) {
            return Err(MetalError::Shape(format!(
                "mv_quantized: K={k} not a multiple of block {block} ({:?})",
                b.dtype
            )));
        }
        // Q8_0 (основной квант LLM): simdgroup-кооперативный кернел — лейны
        // читают соседние байты строки (коалесцированно), 8 строк на
        // threadgroup из 64 потоков. Прочие кванты — поток на строку.
        if b.dtype == GpuDtype::Q8_0 && std::env::var("INFER_MV_Q8_LEGACY").is_err() {
            let (k32, n32) = (k as u32, n as u32);
            // 2 строки на threadgroup из 4 simdgroup'ов (K делится между ними).
            let grid = MTLSize::new((n.div_ceil(2) * 128) as u64, 1, 1);
            let group = MTLSize::new(128, 1, 1);
            return self.run("mv_q8_0_ks", grid, group, |enc| {
                enc.set_buffer(0, Some(&a.buffer), a.offset as u64);
                enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
                enc.set_buffer(2, Some(&c.buffer), c.offset as u64);
                enc.set_bytes(3, 4, (&k32 as *const u32).cast());
                enc.set_bytes(4, 4, (&n32 as *const u32).cast());
            });
        }
        let grid = MTLSize::new(n as u64, 1, 1);
        let group = MTLSize::new(64, 1, 1);
        self.run(name, grid, group, |enc| {
            enc.set_buffer(0, Some(&a.buffer), a.offset as u64);
            enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
            enc.set_buffer(2, Some(&c.buffer), c.offset as u64);
            enc.set_bytes(3, 8, (&k as *const usize).cast());
            enc.set_bytes(4, 8, (&n as *const usize).cast());
        })
    }

    /// Grouped 1D convolution. X (B,Cin,T), W (Cout,Cin_g,K), bias (Cout) or
    /// None, Y (B,Cout,Tout). groups implied by Cin_g: out channel co reads
    /// input channels [g*Cin_g, (g+1)*Cin_g) with g = co / Cin_g.
    #[allow(clippy::too_many_arguments)]
    pub fn conv1d_groups_f32(
        &self,
        x: &GpuTensor,
        w: &GpuTensor,
        bias: Option<&GpuTensor>,
        params: Conv1dParams,
        pad_r: usize,
        time_major: bool,
        time_major_out: bool,
        dense: bool,
        y: &GpuTensor,
    ) -> Result<(), MetalError> {
        let b = x.shape[0];
        let t = x.shape[1];
        let cin = x.shape[2];
        let cout = w.shape[0];
        let cin_g = w.shape[1];
        let k = w.shape[2];
        let t_out = (t + params.pad + pad_r - params.dilation * (k - 1) - 1) / params.stride + 1;
        let expected = if time_major_out {
            vec![b, t_out, cout]
        } else {
            vec![b, cout, t_out]
        };
        if y.shape != expected {
            return Err(MetalError::Shape(format!(
                "conv1d_groups out {:?} != {expected:?}",
                y.shape
            )));
        }
        let grid = MTLSize::new(b as u64, t_out as u64, cout as u64);
        let group = MTLSize::new(1, 4, 4);
        self.run("conv1d_groups_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&w.buffer), w.offset as u64);
            enc.set_buffer(
                2,
                bias.map(|b| &*b.buffer),
                bias.map(|b| b.offset as u64).unwrap_or(0),
            );
            enc.set_buffer(3, Some(&y.buffer), y.offset as u64);
            for (i, v) in [t, cin, cout, k, cin_g].iter().enumerate() {
                enc.set_bytes((4 + i) as u64, 8, (v as *const usize).cast());
            }
            for (i, v) in [params.stride, params.pad, pad_r, params.dilation]
                .iter()
                .enumerate()
            {
                enc.set_bytes((9 + i) as u64, 8, (v as *const usize).cast());
            }
            let tm = time_major as usize;
            enc.set_bytes(13, 8, (&tm as *const usize).cast());
            let tmo = time_major_out as usize;
            enc.set_bytes(14, 8, (&tmo as *const usize).cast());
            let dn = dense as usize;
            enc.set_bytes(15, 8, (&dn as *const usize).cast());
        })
    }

    /// Direct 2D convolution (subsampling): X (B,Cin,T,F), W (Cout,Cin,kT,kF),
    /// bias (Cout, BN-folded) or None, Y (B,Cout,Tout,Fout). Left-pads time by
    /// `pad_t`; strides (s_t, s_f).
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_f32(
        &self,
        x: &GpuTensor,
        w: &GpuTensor,
        bias: Option<&GpuTensor>,
        s_t: usize,
        s_f: usize,
        pad_t: usize,
        time_major_out: bool,
        y: &GpuTensor,
    ) -> Result<(), MetalError> {
        let b = x.shape[0];
        let cin = x.shape[1];
        let t = x.shape[2];
        let f = x.shape[3];
        let cout = w.shape[0];
        let kt = w.shape[2];
        let kf = w.shape[3];
        let t_out = (t + pad_t - kt) / s_t + 1;
        let f_out = (f - kf) / s_f + 1;
        if y.shape != [b, cout, t_out, f_out] {
            return Err(MetalError::Shape(format!(
                "conv2d out {:?} != [{b},{cout},{t_out},{f_out}]",
                y.shape
            )));
        }
        let grid = MTLSize::new(b as u64, cout as u64, (t_out * f_out) as u64);
        let group = MTLSize::new(1, 1, 8);
        self.run("conv2d_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&w.buffer), w.offset as u64);
            enc.set_buffer(
                2,
                bias.map(|b| &*b.buffer),
                bias.map(|b| b.offset as u64).unwrap_or(0),
            );
            enc.set_buffer(3, Some(&y.buffer), y.offset as u64);
            for (i, v) in [cin, t, f, cout, kt, kf].iter().enumerate() {
                enc.set_bytes((4 + i) as u64, 8, (v as *const usize).cast());
            }
            for (i, v) in [s_t, s_f, pad_t].iter().enumerate() {
                enc.set_bytes((10 + i) as u64, 8, (v as *const usize).cast());
            }
            let tm = time_major_out as usize;
            enc.set_bytes(13, 8, (&tm as *const usize).cast());
        })
    }

    /// RMSNorm over the last dim: X (rows, D), W (D), Y (rows, D).
    pub fn rmsnorm_f32(
        &self,
        x: &GpuTensor,
        w: &GpuTensor,
        y: &GpuTensor,
        eps: f32,
    ) -> Result<(), MetalError> {
        let d = x.shape[x.shape.len() - 1];
        let rows = x.numel() / d;
        // threadgroup на строку (см. rmsnorm_row_f32): 256 потоков, для
        // коротких строк — до ближайшего кратного 32.
        let tpg = d.clamp(32, 256).div_ceil(32) * 32;
        let grid = MTLSize::new((rows * tpg) as u64, 1, 1);
        let group = MTLSize::new(tpg as u64, 1, 1);
        let d32 = d as u32;
        self.run("rmsnorm_row_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&w.buffer), w.offset as u64);
            enc.set_buffer(2, Some(&y.buffer), y.offset as u64);
            enc.set_bytes(3, 4, (&d32 as *const u32).cast());
            enc.set_bytes(4, 4, (&eps as *const f32).cast());
        })
    }

    /// LayerNorm over the last dim: X (rows, D), W/B (D), Y (rows, D).
    pub fn layernorm_f32(
        &self,
        x: &GpuTensor,
        w: &GpuTensor,
        b: &GpuTensor,
        y: &GpuTensor,
        eps: f32,
    ) -> Result<(), MetalError> {
        // Последняя ось = d (2D (rows,d) или 3D (B,T,d) — row-major
        // идентичен (rows,d)); rows = numel / d.
        let d = x.shape[x.shape.len() - 1];
        let rows = x.numel() / d;
        let grid = MTLSize::new(rows as u64, 1, 1);
        let group = MTLSize::new(1, 1, 1);
        self.run("layernorm_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&w.buffer), w.offset as u64);
            enc.set_buffer(2, Some(&b.buffer), b.offset as u64);
            enc.set_buffer(3, Some(&y.buffer), y.offset as u64);
            enc.set_bytes(4, 8, (&rows as *const usize).cast());
            enc.set_bytes(5, 8, (&d as *const usize).cast());
            enc.set_bytes(6, 4, (&eps as *const f32).cast());
        })
    }

    /// GLU: X (rows, 2D) → Y (rows, D): gate * sigmoid(value).
    pub fn glu_f32(&self, x: &GpuTensor, y: &GpuTensor) -> Result<(), MetalError> {
        let rows = x.shape[0];
        let d = x.shape[1] / 2;
        let grid = MTLSize::new(rows as u64, d as u64, 1);
        let group = MTLSize::new(4, 8, 1);
        self.run("glu_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&y.buffer), y.offset as u64);
            enc.set_bytes(2, 8, (&rows as *const usize).cast());
            enc.set_bytes(3, 8, (&d as *const usize).cast());
        })
    }

    /// Y = X + B elementwise (same shape; mask bias).
    pub fn add_bias2d_f32(
        &self,
        x: &GpuTensor,
        b: &GpuTensor,
        y: &GpuTensor,
    ) -> Result<(), MetalError> {
        let n = x.numel();
        let grid = MTLSize::new(n as u64, 1, 1);
        let group = MTLSize::new(256, 1, 1);
        self.run("add_bias2d_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
            enc.set_buffer(2, Some(&y.buffer), y.offset as u64);
            enc.set_bytes(3, 8, (&n as *const usize).cast());
        })
    }

    /// Y = A + B*scale (elementwise; residual add with half-step scaling).
    pub fn add_scaled_f32(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        y: &GpuTensor,
        scale: f32,
    ) -> Result<(), MetalError> {
        let n = a.numel();
        let grid = MTLSize::new(n as u64, 1, 1);
        let group = MTLSize::new(256, 1, 1);
        self.run("add_scaled_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&a.buffer), a.offset as u64);
            enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
            enc.set_buffer(2, Some(&y.buffer), y.offset as u64);
            enc.set_bytes(3, 8, (&n as *const usize).cast());
            enc.set_bytes(4, 4, (&scale as *const f32).cast());
        })
    }

    /// Y = X * scale.
    pub fn scale_f32(&self, x: &GpuTensor, y: &GpuTensor, scale: f32) -> Result<(), MetalError> {
        let n = x.numel();
        let grid = MTLSize::new(n as u64, 1, 1);
        let group = MTLSize::new(256, 1, 1);
        self.run("scale_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&y.buffer), y.offset as u64);
            enc.set_bytes(2, 8, (&n as *const usize).cast());
            enc.set_bytes(3, 4, (&scale as *const f32).cast());
        })
    }

    /// Head-aware LayerNorm (t-one q_ln/k_ln): X (rows, H*HD) → Y, normalized
    /// per (row, head); W/B (HD).
    pub fn layernorm_head_f32(
        &self,
        x: &GpuTensor,
        w: &GpuTensor,
        b: &GpuTensor,
        y: &GpuTensor,
        h: usize,
        eps: f32,
    ) -> Result<(), MetalError> {
        let rows = x.shape[0];
        let hd = x.shape[1] / h;
        let grid = MTLSize::new(rows as u64, h as u64, hd as u64);
        let group = MTLSize::new(1, 1, 8);
        self.run("layernorm_head_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&w.buffer), w.offset as u64);
            enc.set_buffer(2, Some(&b.buffer), b.offset as u64);
            enc.set_buffer(3, Some(&y.buffer), y.offset as u64);
            for (i, v) in [rows, h, hd].iter().enumerate() {
                enc.set_bytes((4 + i) as u64, 8, (v as *const usize).cast());
            }
            enc.set_bytes(7, 4, (&eps as *const f32).cast());
        })
    }

    /// Uniform attention fallback: V (B,H,KV,HD) → Y (B,H,T,HD) where each
    /// row is the mean over KV.
    pub fn uniform_attn_f32(&self, v: &GpuTensor, y: &GpuTensor) -> Result<(), MetalError> {
        let (b, h, kv, hd) = (v.shape[0], v.shape[1], v.shape[2], v.shape[3]);
        let t = y.shape[2];
        let grid = MTLSize::new(b as u64, h as u64, (t * hd) as u64);
        let group = MTLSize::new(1, 1, 8);
        self.run("uniform_attn_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&v.buffer), v.offset as u64);
            enc.set_buffer(1, Some(&y.buffer), y.offset as u64);
            for (i, v) in [b, h, t, kv, hd].iter().enumerate() {
                enc.set_bytes((2 + i) as u64, 8, (v as *const usize).cast());
            }
        })
    }

    /// Y = A * B (elementwise).
    pub fn mul_f32(&self, a: &GpuTensor, b: &GpuTensor, y: &GpuTensor) -> Result<(), MetalError> {
        let n = a.numel();
        let grid = MTLSize::new(n as u64, 1, 1);
        let group = MTLSize::new(256, 1, 1);
        self.run("mul_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&a.buffer), a.offset as u64);
            enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
            enc.set_buffer(2, Some(&y.buffer), y.offset as u64);
            enc.set_bytes(3, 8, (&n as *const usize).cast());
        })
    }

    /// Repeat frames ×2 with truncating write: X (B,T,D) → Y (B,T_OUT,D).
    pub fn repeat_f32(&self, x: &GpuTensor, y: &GpuTensor) -> Result<(), MetalError> {
        let (b, t, d) = (x.shape[0], x.shape[1], x.shape[2]);
        let t_out = y.shape[1];
        let grid = MTLSize::new(b as u64, t as u64, d as u64);
        let group = MTLSize::new(1, 1, 8);
        self.run("repeat_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&y.buffer), y.offset as u64);
            for (i, v) in [b, t, t_out, d].iter().enumerate() {
                enc.set_bytes((2 + i) as u64, 8, (v as *const usize).cast());
            }
        })
    }

    /// Fused multi-head self-attention. Q/K/V (B,T,D) time-major; SC
    /// (B,H,T,T); MASK (T,T) bias or None; MODE: 0 leader, 1 non-leader, 2
    /// uniform. Y (B,T,D).
    pub fn mhsa_f32(
        &self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        sc: &GpuTensor,
        mask: Option<&GpuTensor>,
        mode: u32,
        y: &GpuTensor,
    ) -> Result<(), MetalError> {
        let (b, t, d) = (q.shape[0], q.shape[1], q.shape[2]);
        let h = sc.shape[1];
        let hd = d / h;
        let grid = MTLSize::new(b as u64, h as u64, t as u64);
        let group = MTLSize::new(1, 1, 1);
        self.run("mhsa_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&q.buffer), q.offset as u64);
            enc.set_buffer(1, Some(&k.buffer), k.offset as u64);
            enc.set_buffer(2, Some(&v.buffer), v.offset as u64);
            enc.set_buffer(3, Some(&sc.buffer), sc.offset as u64);
            enc.set_buffer(
                4,
                mask.map(|m| &*m.buffer),
                mask.as_ref().map(|m| m.offset as u64).unwrap_or(0),
            );
            enc.set_buffer(5, Some(&y.buffer), y.offset as u64);
            for (i, v) in [b, t, h, hd].iter().enumerate() {
                enc.set_bytes((6 + i) as u64, 8, (v as *const usize).cast());
            }
            let m = mode as usize;
            enc.set_bytes(10, 8, (&m as *const usize).cast());
        })
    }

    /// Head-aware RoPE: X (B,T,D), cos/sin (T, rope_dim/2), Y (B,T,D).
    pub fn rope_head_f32(
        &self,
        x: &GpuTensor,
        cos_buf: &GpuTensor,
        sin_buf: &GpuTensor,
        y: &GpuTensor,
        h: usize,
        rope_dim: usize,
    ) -> Result<(), MetalError> {
        let (b, t, d) = (x.shape[0], x.shape[1], x.shape[2]);
        let hd = d / h;
        let grid = MTLSize::new((b * h) as u64, t as u64, d as u64);
        let group = MTLSize::new(1, 1, 8);
        self.run("rope_head_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&cos_buf.buffer), cos_buf.offset as u64);
            enc.set_buffer(2, Some(&sin_buf.buffer), sin_buf.offset as u64);
            enc.set_buffer(3, Some(&y.buffer), y.offset as u64);
            for (i, v) in [b, t, h, hd, rope_dim].iter().enumerate() {
                enc.set_bytes((4 + i) as u64, 8, (v as *const usize).cast());
            }
        })
    }

    // ── DAC codec ops (fp32) ──

    /// Transposed 1D convolution. W is (Cin, Cout, K) — the
    /// conv_transpose1d layout (in_channels, out_channels, k). X (B,T,Cin),
    /// bias (Cout) or None, Y (B,Tout,Cout) with Tout = (T-1)*stride + K.
    pub fn conv_transpose1d_f32(
        &self,
        x: &GpuTensor,
        w: &GpuTensor,
        bias: Option<&GpuTensor>,
        stride: usize,
        y: &GpuTensor,
    ) -> Result<(), MetalError> {
        let (b, t, cin) = (x.shape[0], x.shape[1], x.shape[2]);
        let (_, cout, k) = (w.shape[0], w.shape[1], w.shape[2]);
        let t_out = (t - 1) * stride + k;
        if y.shape != [b, t_out, cout] {
            return Err(MetalError::Shape(format!(
                "conv_transpose1d out {:?} != [{b},{t_out},{cout}]",
                y.shape
            )));
        }
        let grid = MTLSize::new(b as u64, t_out as u64, cout as u64);
        let group = MTLSize::new(1, 4, 4);
        self.run("conv_transpose1d_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&w.buffer), w.offset as u64);
            enc.set_buffer(
                2,
                bias.map(|b| &*b.buffer),
                bias.map(|b| b.offset as u64).unwrap_or(0),
            );
            enc.set_buffer(3, Some(&y.buffer), y.offset as u64);
            for (i, v) in [t, cin, cout, k].iter().enumerate() {
                enc.set_bytes((4 + i) as u64, 8, (v as *const usize).cast());
            }
            enc.set_bytes(8, 8, (&stride as *const usize).cast());
        })
    }

    /// Snake activation: X (B,T,C), alpha (C), Y (B,T,C).
    pub fn snake_f32(
        &self,
        x: &GpuTensor,
        alpha: &GpuTensor,
        y: &GpuTensor,
    ) -> Result<(), MetalError> {
        let rows = x.shape[0] * x.shape[1];
        let c = x.shape[2];
        let grid = MTLSize::new(rows as u64, c as u64, 1);
        let group = MTLSize::new(64, 8, 1);
        self.run("snake_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&alpha.buffer), alpha.offset as u64);
            enc.set_buffer(2, Some(&y.buffer), y.offset as u64);
            enc.set_bytes(3, 8, (&rows as *const usize).cast());
            enc.set_bytes(4, 8, (&c as *const usize).cast());
        })
    }

    /// GELU (tanh approx), elementwise, same shape.
    pub fn gelu_tanh_f32(&self, x: &GpuTensor, y: &GpuTensor) -> Result<(), MetalError> {
        let n = x.numel();
        let grid = MTLSize::new(n as u64, 1, 1);
        let group = MTLSize::new(256, 1, 1);
        self.run("gelu_tanh_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&y.buffer), y.offset as u64);
            enc.set_bytes(2, 8, (&n as *const usize).cast());
        })
    }

    /// GELU (erf) — vision (clip/Qwen3-VL): 0.5·x·(1+erf(x/√2)).
    pub fn gelu_erf_f32(&self, x: &GpuTensor, y: &GpuTensor) -> Result<(), MetalError> {
        let n = x.numel();
        let grid = MTLSize::new(n as u64, 1, 1);
        let group = MTLSize::new(256, 1, 1);
        self.run("gelu_erf_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&y.buffer), y.offset as u64);
            enc.set_bytes(2, 8, (&n as *const usize).cast());
        })
    }

    /// Vision patch layout: merge 2×2 blocks + patch_bias + position_embd.
    /// X/POS [per², d] → Y [per², d]; one thread per (block-row, token-in-block).
    pub fn merge2x2_add_f32(
        &self,
        x: &GpuTensor,
        bias: &GpuTensor,
        pos: &GpuTensor,
        y: &GpuTensor,
        per: usize,
    ) -> Result<(), MetalError> {
        let d = x.shape[1];
        let grid = MTLSize::new((per * per / 4) as u64, 4, 1);
        let group = MTLSize::new(64, 1, 1);
        self.run("merge2x2_add_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&bias.buffer), bias.offset as u64);
            enc.set_buffer(2, Some(&pos.buffer), pos.offset as u64);
            enc.set_buffer(3, Some(&y.buffer), y.offset as u64);
            enc.set_bytes(4, 8, (&per as *const usize).cast());
            enc.set_bytes(5, 8, (&d as *const usize).cast());
        })
    }

    /// Vision deepstack/post_ln group: [n, d] → [n/4, 4·d].
    pub fn group4_f32(&self, x: &GpuTensor, y: &GpuTensor) -> Result<(), MetalError> {
        let n = x.shape[0];
        let d = x.shape[1];
        let total = (n / 4) * 4 * d;
        let grid = MTLSize::new(total as u64, 1, 1);
        let group = MTLSize::new(256, 1, 1);
        self.run("group4_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&y.buffer), y.offset as u64);
            enc.set_bytes(2, 8, (&n as *const usize).cast());
            enc.set_bytes(3, 8, (&d as *const usize).cast());
        })
    }

    /// Vision merger concat: chunks ([nq, proj] ×nchunks, ≤4) → [4·nq, proj],
    /// rows interleaved: out[q·4+j] = chunk_j[q].
    pub fn concat4_f32(&self, chunks: &[GpuTensor], y: &GpuTensor) -> Result<(), MetalError> {
        let nq = y.shape[0] / 4;
        let proj = y.shape[1];
        let total = nq * 4 * proj;
        let grid = MTLSize::new(total as u64, 1, 1);
        let group = MTLSize::new(256, 1, 1);
        self.run("concat4_f32", grid, group, |enc| {
            for (i, c) in chunks.iter().take(4).enumerate() {
                enc.set_buffer(i as u64, Some(&c.buffer), c.offset as u64);
            }
            enc.set_buffer(4, Some(&y.buffer), y.offset as u64);
            let nchunks = chunks.len().min(4);
            enc.set_bytes(5, 8, (&nq as *const usize).cast());
            enc.set_bytes(6, 8, (&proj as *const usize).cast());
            enc.set_bytes(7, 8, (&nchunks as *const usize).cast());
        })
    }

    /// Embedding lookup: idx (B,T) u32, table (N,D) → Y (B,T,D). Batch fixed
    /// to 1 in the kernel (codec decode path). The idx buffer is uploaded
    /// per call (small: T×4 bytes).
    pub fn embedding_f32(
        &self,
        idx: &[u32],
        table: &GpuTensor,
        y: &GpuTensor,
    ) -> Result<(), MetalError> {
        let (t, d) = (y.shape[1], y.shape[2]);
        let n = table.shape[0];
        if idx.len() != t || y.shape[0] != 1 {
            return Err(MetalError::Shape(format!(
                "embedding: idx {} != T {t} or batch != 1",
                idx.len()
            )));
        }
        let idx_buf = with_pool(|p| p.acquire(&self.device, t * 4));
        if idx_buf.contents().is_null() {
            return Err(MetalError::BufferAlloc);
        }
        unsafe {
            std::ptr::copy_nonoverlapping(idx.as_ptr(), idx_buf.contents().cast(), t);
        }
        let grid = MTLSize::new(1, t as u64, d as u64);
        let group = MTLSize::new(1, 4, 8);
        let res = self.run("embedding_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&idx_buf), 0);
            enc.set_buffer(1, Some(&table.buffer), table.offset as u64);
            enc.set_buffer(2, Some(&y.buffer), y.offset as u64);
            for (i, v) in [t, n, d].iter().enumerate() {
                enc.set_bytes((3 + i) as u64, 8, (v as *const usize).cast());
            }
        });
        with_pool(|p| p.push(&idx_buf));
        res
    }

    /// Эмбеддинги из таблицы в родном формате (f16/Q8_0/Q4_0/Q4_K/Q6_K):
    /// деквант строки на лету — огромный словарь (Qwen3.5: 248K) не
    /// раздувается в f32, а tied-таблица одновременно служит lm_head.
    pub fn embedding_quant_f32(
        &self,
        idx: &[u32],
        table: &GpuTensor,
        y: &GpuTensor,
    ) -> Result<(), MetalError> {
        let (t, d) = (y.shape[1], y.shape[2]);
        let n = table.shape[0];
        if idx.len() != t || y.shape[0] != 1 || table.shape.get(1) != Some(&d) {
            return Err(MetalError::Shape(format!(
                "embedding_quant: idx {} / T {t} / table {:?} / y {:?}",
                idx.len(),
                table.shape,
                y.shape
            )));
        }
        let fmt: u32 = match table.dtype {
            GpuDtype::F16 => 0,
            GpuDtype::Q8_0 => 1,
            GpuDtype::Q4_0 => 2,
            GpuDtype::Q4_K => 3,
            GpuDtype::Q6_K => 4,
            other => {
                return Err(MetalError::Shape(format!(
                    "embedding_quant: таблица {other:?} не поддержана"
                )));
            }
        };
        if let Some(block) = table.dtype.block_elems()
            && !d.is_multiple_of(block)
        {
            return Err(MetalError::Shape(format!(
                "embedding_quant: D={d} не кратно блоку {block} ({:?})",
                table.dtype
            )));
        }
        let idx_buf = with_pool(|p| p.acquire(&self.device, t * 4));
        if idx_buf.contents().is_null() {
            return Err(MetalError::BufferAlloc);
        }
        unsafe {
            std::ptr::copy_nonoverlapping(idx.as_ptr(), idx_buf.contents().cast(), t);
        }
        let (t32, n32, d32) = (t as u32, n as u32, d as u32);
        let res = self.run(
            "embedding_quant_f32",
            MTLSize::new(t as u64, d as u64, 1),
            MTLSize::new(1, 64.min(d) as u64, 1),
            |enc| {
                enc.set_buffer(0, Some(&idx_buf), 0);
                enc.set_buffer(1, Some(&table.buffer), table.offset as u64);
                enc.set_buffer(2, Some(&y.buffer), y.offset as u64);
                enc.set_bytes(3, 4, (&t32 as *const u32).cast());
                enc.set_bytes(4, 4, (&n32 as *const u32).cast());
                enc.set_bytes(5, 4, (&d32 as *const u32).cast());
                enc.set_bytes(6, 4, (&fmt as *const u32).cast());
            },
        );
        with_pool(|p| p.push(&idx_buf));
        res
    }

    /// RoPE with interleaved pairs per head: X (B,T,D=H*HD),
    /// cos/sin (max_seq,HD/2), START absolute position offset.
    pub fn rope_interleaved_f32(
        &self,
        x: &GpuTensor,
        cos_buf: &GpuTensor,
        sin_buf: &GpuTensor,
        y: &GpuTensor,
        hd: usize,
        start: usize,
    ) -> Result<(), MetalError> {
        let (b, t, d) = (x.shape[0], x.shape[1], x.shape[2]);
        let grid = MTLSize::new(b as u64, t as u64, d as u64);
        let group = MTLSize::new(1, 4, 8);
        self.run("rope_interleaved_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&cos_buf.buffer), cos_buf.offset as u64);
            enc.set_buffer(2, Some(&sin_buf.buffer), sin_buf.offset as u64);
            enc.set_buffer(3, Some(&y.buffer), y.offset as u64);
            for (i, v) in [b, t, d, hd, start].iter().enumerate() {
                enc.set_bytes((4 + i) as u64, 8, (v as *const usize).cast());
            }
        })
    }

    /// Broadcast-multiply by a per-column vector (LayerScale): X (rows,cols),
    /// W (cols) → Y (rows,cols).
    pub fn mul_vec_f32(
        &self,
        x: &GpuTensor,
        w: &GpuTensor,
        y: &GpuTensor,
    ) -> Result<(), MetalError> {
        let (rows, cols) = (x.shape[0], x.shape[1]);
        let grid = MTLSize::new(rows as u64, cols as u64, 1);
        let group = MTLSize::new(8, 8, 1);
        self.run("mul_vec_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&w.buffer), w.offset as u64);
            enc.set_buffer(2, Some(&y.buffer), y.offset as u64);
            enc.set_bytes(3, 8, (&rows as *const usize).cast());
            enc.set_bytes(4, 8, (&cols as *const usize).cast());
        })
    }

    /// tanh elementwise.
    pub fn tanh_f32(&self, x: &GpuTensor, y: &GpuTensor) -> Result<(), MetalError> {
        let n = x.numel();
        let grid = MTLSize::new(n as u64, 1, 1);
        let group = MTLSize::new(256, 1, 1);
        self.run("tanh_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&y.buffer), y.offset as u64);
            enc.set_bytes(2, 8, (&n as *const usize).cast());
        })
    }

    /// Nearest-codebook search (RVQ encode): X (rows, 8), table (N, 8) →
    /// argmin L2 indices written back into `idx` (rows).
    pub fn codebook_search_f32(
        &self,
        x: &GpuTensor,
        table: &GpuTensor,
        idx: &mut [u32],
    ) -> Result<(), MetalError> {
        let rows = x.shape[0] * x.shape[1]; // batch × time
        let n = table.shape[0];
        if idx.len() != rows {
            return Err(MetalError::Shape(format!(
                "codebook_search: idx {} != rows {rows}",
                idx.len()
            )));
        }
        let idx_buf = with_pool(|p| p.acquire(&self.device, rows * 4));
        if idx_buf.contents().is_null() {
            return Err(MetalError::BufferAlloc);
        }
        unsafe {
            std::ptr::copy_nonoverlapping(idx.as_ptr(), idx_buf.contents().cast(), rows);
        }
        let grid = MTLSize::new(rows as u64, 1, 1);
        let group = MTLSize::new(256, 1, 1);
        let res = (|| -> Result<(), MetalError> {
            self.run("codebook_search_f32", grid, group, |enc| {
                enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
                enc.set_buffer(1, Some(&table.buffer), table.offset as u64);
                enc.set_buffer(2, Some(&idx_buf), 0);
                for (i, v) in [rows, n].iter().enumerate() {
                    enc.set_bytes((3 + i) as u64, 8, (v as *const usize).cast());
                }
            })?;
            // In batch mode the kernel has NOT executed yet — the host copy
            // below would read the previous indices. Flush like to_cpu_f32 does.
            self.flush_batch()?;
            let ptr = idx_buf.contents();
            if ptr.is_null() {
                return Err(MetalError::BufferAlloc);
            }
            unsafe {
                std::ptr::copy_nonoverlapping(ptr as *const u32, idx.as_mut_ptr(), rows);
            }
            Ok(())
        })();
        with_pool(|p| p.push(&idx_buf));
        res
    }

    /// GQA attention with KV cache: q (B,Tq,H*HD), k/v (B,Tkv,KVH*HD),
    /// Deepstack: y[t] = x[t] + embd[t+shift] (vision-путь Qwen3-VL).
    pub fn deepstack_add_f32(
        &self,
        x: &GpuTensor,
        embd: &GpuTensor,
        shift: usize,
        y: &GpuTensor,
    ) -> Result<(), MetalError> {
        let (t, d) = (x.shape[1], x.shape[2]);
        let grid = MTLSize::new(t as u64, d as u64, 1);
        self.run("deepstack_add_f32", grid, MTLSize::new(16, 8, 1), |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&embd.buffer), embd.offset as u64);
            enc.set_buffer(2, Some(&y.buffer), y.offset as u64);
            for (i, v) in [t, shift, d].iter().enumerate() {
                enc.set_bytes((3 + i) as u64, 8, (v as *const usize).cast());
            }
        })
    }

    /// y (B,Tq,H*HD). Mask (B,Tq,Tkv) or None (generation, causal).
    #[allow(clippy::too_many_arguments)]
    pub fn attn_ar_f32(
        &self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        mask: Option<&GpuTensor>,
        y: &GpuTensor,
        n_heads: usize,
        kv_heads: usize,
        hd: usize,
    ) -> Result<(), MetalError> {
        let (b, tq, _d) = (q.shape[0], q.shape[1], q.shape[2]);
        let tkv = k.shape[1];
        let n_rep = n_heads / kv_heads;
        // Decode (один запрос, без маски): голова = threadgroup из 8
        // simdgroup'ов, ключи делятся между ними (flash-decoding).
        if b == 1
            && tq == 1
            && mask.is_none()
            && hd.is_multiple_of(4)
            && hd <= 256
            && n_heads.is_multiple_of(kv_heads)
            && std::env::var("INFER_ATTN_LEGACY").is_err()
        {
            let (tkv32, h32, kvh32, hd32) = (tkv as u32, n_heads as u32, kv_heads as u32, hd as u32);
            let bytes = (8 * (2 + hd) * 4).div_ceil(16) * 16;
            let grid = MTLSize::new((n_heads * 256) as u64, 1, 1);
            let group = MTLSize::new(256, 1, 1);
            return self.run("attn_decode_split_f32", grid, group, |enc| {
                enc.set_buffer(0, Some(&q.buffer), q.offset as u64);
                enc.set_buffer(1, Some(&k.buffer), k.offset as u64);
                enc.set_buffer(2, Some(&v.buffer), v.offset as u64);
                enc.set_buffer(3, Some(&y.buffer), y.offset as u64);
                enc.set_bytes(4, 4, (&tkv32 as *const u32).cast());
                enc.set_bytes(5, 4, (&h32 as *const u32).cast());
                enc.set_bytes(6, 4, (&kvh32 as *const u32).cast());
                enc.set_bytes(7, 4, (&hd32 as *const u32).cast());
                enc.set_threadgroup_memory_length(0, bytes as u64);
            });
        }
        let grid = MTLSize::new(b as u64, (n_heads * 32) as u64, tq as u64);
        let group = MTLSize::new(1, 32, 1);
        self.run("attn_ar_f32_fused", grid, group, |enc| {
            enc.set_buffer(0, Some(&q.buffer), q.offset as u64);
            enc.set_buffer(1, Some(&k.buffer), k.offset as u64);
            enc.set_buffer(2, Some(&v.buffer), v.offset as u64);
            enc.set_buffer(
                3,
                mask.map(|m| &*m.buffer),
                mask.as_ref().map(|m| m.offset as u64).unwrap_or(0),
            );
            enc.set_buffer(4, Some(&y.buffer), y.offset as u64);
            for (i, v) in [b, tq, tkv, n_heads, kv_heads, hd, n_rep]
                .iter()
                .enumerate()
            {
                enc.set_bytes((5 + i) as u64, 8, (v as *const usize).cast());
            }
        })?;
        Ok(())
    }

    // ── Ф9.4: Qwen3.5/3.8 (гибрид Gated DeltaNet) ──────────────────────────

    /// Causal depthwise conv1d со стейтом (ggml_ssm_conv на [state | x]).
    /// x [T, C], w [C, K], state [C, K-1] → y [T, C].
    pub fn gdn_conv1d_f32(
        &self,
        x: &GpuTensor,
        w: &GpuTensor,
        state: &GpuTensor,
        y: &GpuTensor,
        t: usize,
        c: usize,
        k: usize,
    ) -> Result<(), MetalError> {
        let (t32, c32, k32) = (t as u32, c as u32, k as u32);
        self.run(
            "gdn_conv1d_f32",
            MTLSize::new(t as u64, c as u64, 1),
            MTLSize::new(1, 64.min(c) as u64, 1),
            |enc| {
                enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
                enc.set_buffer(1, Some(&w.buffer), w.offset as u64);
                enc.set_buffer(2, Some(&state.buffer), state.offset as u64);
                enc.set_buffer(3, Some(&y.buffer), y.offset as u64);
                enc.set_bytes(4, 4, (&t32 as *const u32).cast());
                enc.set_bytes(5, 4, (&c32 as *const u32).cast());
                enc.set_bytes(6, 4, (&k32 as *const u32).cast());
            },
        )
    }

    /// Новый conv-стейт: последние K-1 позиций конкатенации [state | x].
    pub fn gdn_conv_state_f32(
        &self,
        x: &GpuTensor,
        state: &GpuTensor,
        new_state: &GpuTensor,
        t: usize,
        c: usize,
        k: usize,
    ) -> Result<(), MetalError> {
        let (t32, c32, k32) = (t as u32, c as u32, k as u32);
        self.run(
            "gdn_conv_state_f32",
            MTLSize::new((k - 1) as u64, c as u64, 1),
            MTLSize::new(1, 64.min(c) as u64, 1),
            |enc| {
                enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
                enc.set_buffer(1, Some(&state.buffer), state.offset as u64);
                enc.set_buffer(2, Some(&new_state.buffer), new_state.offset as u64);
                enc.set_bytes(3, 4, (&t32 as *const u32).cast());
                enc.set_bytes(4, 4, (&c32 as *const u32).cast());
                enc.set_bytes(5, 4, (&k32 as *const u32).cast());
            },
        )
    }

    /// L2-нормировка строк-голов: y = x / sqrt(Σx² + eps). x/y [rows, hd].
    pub fn gdn_l2norm_f32(
        &self,
        x: &GpuTensor,
        y: &GpuTensor,
        rows: usize,
        hd: usize,
        eps: f32,
    ) -> Result<(), MetalError> {
        let (r32, h32) = (rows as u32, hd as u32);
        self.run(
            "gdn_l2norm_f32",
            MTLSize::new(rows as u64, 1, 1),
            MTLSize::new(32.min(rows) as u64, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
                enc.set_buffer(1, Some(&y.buffer), y.offset as u64);
                enc.set_bytes(2, 4, (&r32 as *const u32).cast());
                enc.set_bytes(3, 4, (&h32 as *const u32).cast());
                enc.set_bytes(4, 4, (&eps as *const f32).cast());
            },
        )
    }

    /// Гейты GDN: g = softplus(alpha + dt_bias)·a, beta = sigmoid(beta_raw).
    /// alpha/beta/g/b [n = T·H], dt_bias/a [H].
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_gates_f32(
        &self,
        alpha: &GpuTensor,
        beta: &GpuTensor,
        dt_bias: &GpuTensor,
        a: &GpuTensor,
        g: &GpuTensor,
        b: &GpuTensor,
        n: usize,
        h: usize,
    ) -> Result<(), MetalError> {
        let (n32, h32) = (n as u32, h as u32);
        self.run(
            "gdn_gates_f32",
            MTLSize::new(n as u64, 1, 1),
            MTLSize::new(64.min(n) as u64, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&alpha.buffer), alpha.offset as u64);
                enc.set_buffer(1, Some(&beta.buffer), beta.offset as u64);
                enc.set_buffer(2, Some(&dt_bias.buffer), dt_bias.offset as u64);
                enc.set_buffer(3, Some(&a.buffer), a.offset as u64);
                enc.set_buffer(4, Some(&g.buffer), g.offset as u64);
                enc.set_buffer(5, Some(&b.buffer), b.offset as u64);
                enc.set_bytes(6, 4, (&n32 as *const u32).cast());
                enc.set_bytes(7, 4, (&h32 as *const u32).cast());
            },
        )
    }

    /// Рекуррентный gated delta rule (ggml gated_delta_net, K=1): по
    /// threadgroup на v-голову, поток на столбец стейта. Стейт [Hv, S, S]
    /// обновляется на месте; out [T, Hv, S].
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_delta_rule_f32(
        &self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        g: &GpuTensor,
        b: &GpuTensor,
        state: &GpuTensor,
        out: &GpuTensor,
        t: usize,
        hk: usize,
        hv: usize,
        s: usize,
    ) -> Result<(), MetalError> {
        let (t32, hk32, hv32, s32) = (t as u32, hk as u32, hv as u32, s as u32);
        // S кратно 32 и ≤ 256 — simdgroup-параллельный вариант (4 строки
        // стейта на threadgroup, строка на simdgroup); иначе — поток на строку.
        let (kernel, grid, group) = if s.is_multiple_of(32) && s <= 256 && (hv * s).is_multiple_of(4) {
            (
                "gdn_delta_rule_sg_f32",
                MTLSize::new((hv * s / 4 * 128) as u64, 1, 1),
                MTLSize::new(128, 1, 1),
            )
        } else {
            (
                "gdn_delta_rule_f32",
                MTLSize::new((hv * s) as u64, 1, 1),
                MTLSize::new(s as u64, 1, 1),
            )
        };
        self.run(
            kernel,
            grid,
            group,
            |enc| {
                enc.set_buffer(0, Some(&q.buffer), q.offset as u64);
                enc.set_buffer(1, Some(&k.buffer), k.offset as u64);
                enc.set_buffer(2, Some(&v.buffer), v.offset as u64);
                enc.set_buffer(3, Some(&g.buffer), g.offset as u64);
                enc.set_buffer(4, Some(&b.buffer), b.offset as u64);
                enc.set_buffer(5, Some(&state.buffer), state.offset as u64);
                enc.set_buffer(6, Some(&out.buffer), out.offset as u64);
                enc.set_bytes(7, 4, (&t32 as *const u32).cast());
                enc.set_bytes(8, 4, (&hk32 as *const u32).cast());
                enc.set_bytes(9, 4, (&hv32 as *const u32).cast());
                enc.set_bytes(10, 4, (&s32 as *const u32).cast());
            },
        )
    }

    /// Гейтовая RMS-норма GDN: y = rmsnorm(x)·w · silu(z). x/z/y [rows, hd].
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_norm_gated_f32(
        &self,
        x: &GpuTensor,
        z: &GpuTensor,
        w: &GpuTensor,
        y: &GpuTensor,
        rows: usize,
        hd: usize,
        eps: f32,
    ) -> Result<(), MetalError> {
        let (r32, h32) = (rows as u32, hd as u32);
        self.run(
            "gdn_norm_gated_f32",
            MTLSize::new(rows as u64, 1, 1),
            MTLSize::new(32.min(rows) as u64, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
                enc.set_buffer(1, Some(&z.buffer), z.offset as u64);
                enc.set_buffer(2, Some(&w.buffer), w.offset as u64);
                enc.set_buffer(3, Some(&y.buffer), y.offset as u64);
                enc.set_bytes(4, 4, (&r32 as *const u32).cast());
                enc.set_bytes(5, 4, (&h32 as *const u32).cast());
                enc.set_bytes(6, 4, (&eps as *const f32).cast());
            },
        )
    }

    /// IMROPE для текста с частичной ротацией (первые n_rot измерений головы).
    #[allow(clippy::too_many_arguments)]
    pub fn rope_imrope_text_f32(
        &self,
        x: &GpuTensor,
        y: &GpuTensor,
        tq: usize,
        d: usize,
        hd: usize,
        pos: usize,
        theta: f32,
        sections: [usize; 4],
        n_rot: usize,
    ) -> Result<(), MetalError> {
        let (tq32, d32, hd32, pos32, nr32) =
            (tq as u32, d as u32, hd as u32, pos as u32, n_rot as u32);
        let sects: [u32; 4] = [
            sections[0] as u32,
            sections[1] as u32,
            sections[2] as u32,
            sections[3] as u32,
        ];
        self.run(
            "rope_imrope_text_f32",
            MTLSize::new(tq as u64, (d / hd) as u64, hd as u64),
            MTLSize::new(1, 1, 64.min(hd) as u64),
            |enc| {
                enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
                enc.set_buffer(1, Some(&y.buffer), y.offset as u64);
                enc.set_bytes(2, 4, (&tq32 as *const u32).cast());
                enc.set_bytes(3, 4, (&d32 as *const u32).cast());
                enc.set_bytes(4, 4, (&hd32 as *const u32).cast());
                enc.set_bytes(5, 4, (&pos32 as *const u32).cast());
                enc.set_bytes(6, 4, (&theta as *const f32).cast());
                enc.set_bytes(7, 16, (sects.as_ptr() as *const u8).cast());
                enc.set_bytes(8, 4, (&nr32 as *const u32).cast());
            },
        )
    }

    /// KV-cache update: dst (B,cap,D)[b, pos+t, d] = src (B,Tq,D)[b, t, d].
    pub fn kv_update_f32(
        &self,
        dst: &GpuTensor,
        src: &GpuTensor,
        pos: usize,
    ) -> Result<(), MetalError> {
        let (b, tq, d) = (src.shape[0], src.shape[1], src.shape[2]);
        let cap = dst.shape[1];
        let grid = MTLSize::new(b as u64, tq as u64, d as u64);
        let group = MTLSize::new(1, 4, 4);
        self.run("kv_update_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&dst.buffer), dst.offset as u64);
            enc.set_buffer(1, Some(&src.buffer), src.offset as u64);
            for (i, v) in [b, tq, pos, d, cap].iter().enumerate() {
                enc.set_bytes((2 + i) as u64, 8, (v as *const usize).cast());
            }
        })
    }

    pub fn embedding_override_f32(
        &self,
        idx: &[u32],
        table: &GpuTensor,
        vision: &GpuTensor,
        y: &GpuTensor,
        override_id: u32,
    ) -> Result<(), MetalError> {
        let t = y.shape[1];
        let n = table.shape[0];
        let d = table.shape[1];
        // число vision-строк: vision может быть [V, D] или [1, V, D]
        let v = vision.shape.iter().product::<usize>() / d;
        if idx.len() != t {
            return Err(MetalError::Shape(format!(
                "embedding_override: idx {} != T {t}",
                idx.len()
            )));
        }
        let idx_buf = with_pool(|p| p.acquire(&self.device, t * 4));
        if idx_buf.contents().is_null() {
            return Err(MetalError::BufferAlloc);
        }
        unsafe {
            std::ptr::copy_nonoverlapping(idx.as_ptr(), idx_buf.contents().cast(), t);
        }
        let grid = MTLSize::new(1, t as u64, d as u64);
        let group = MTLSize::new(1, 4, 8);
        let res = self.run("embedding_override_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&idx_buf), 0);
            enc.set_buffer(1, Some(&table.buffer), table.offset as u64);
            enc.set_buffer(2, Some(&vision.buffer), vision.offset as u64);
            enc.set_buffer(3, Some(&y.buffer), y.offset as u64);
            for (i, val) in [t, n, d, override_id as usize, v].iter().enumerate() {
                enc.set_bytes((4 + i) as u64, 8, (val as *const usize).cast());
            }
        });
        with_pool(|p| p.push(&idx_buf));
        res
    }

    /// RMSNorm per head (QK-norm): X (rows, H*HD) → Y, W (HD).
    pub fn rmsnorm_head_f32(
        &self,
        x: &GpuTensor,
        w: &GpuTensor,
        y: &GpuTensor,
        h: usize,
        eps: f32,
    ) -> Result<(), MetalError> {
        // X (B, T, H·HD) or (rows, H·HD); rows = all leading dims flattened
        // (batched tokens must ALL be normalized — shape[0] alone is wrong
        // for prefill batches). HD lives on the LAST axis.
        let hd = x.shape[x.shape.len() - 1] / h;
        let rows = x.shape.iter().product::<usize>() / (h * hd);
        let nh = (rows * h) as u32;
        let hd32 = hd as u32;
        let grid = MTLSize::new(rows as u64 * h as u64 * 32, 1, 1);
        let group = MTLSize::new(64, 1, 1);
        self.run("rmsnorm_head_sg_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&w.buffer), w.offset as u64);
            enc.set_buffer(2, Some(&y.buffer), y.offset as u64);
            enc.set_bytes(3, 4, (&nh as *const u32).cast());
            enc.set_bytes(4, 4, (&hd32 as *const u32).cast());
            enc.set_bytes(5, 4, (&eps as *const f32).cast());
        })
    }

    /// Per-row scalar multiply: X (rows, D), W (rows) → Y.
    pub fn mul_rows_f32(
        &self,
        x: &GpuTensor,
        w: &GpuTensor,
        y: &GpuTensor,
    ) -> Result<(), MetalError> {
        let (rows, d) = (x.shape[0], x.shape[1]);
        let grid = MTLSize::new(rows as u64, d as u64, 1);
        let group = MTLSize::new(8, 8, 1);
        self.run("mul_rows_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&w.buffer), w.offset as u64);
            enc.set_buffer(2, Some(&y.buffer), y.offset as u64);
            for (i, v) in [rows, d].iter().enumerate() {
                enc.set_bytes((3 + i) as u64, 8, (v as *const usize).cast());
            }
        })
    }

    /// Slice along the last dim: X (rows, D) → Y (rows, d0) starting at `off`.
    pub fn slice_last_f32(
        &self,
        x: &GpuTensor,
        y: &GpuTensor,
        off: usize,
    ) -> Result<(), MetalError> {
        let rows = x.shape[0];
        let d = x.shape[1];
        let d0 = y.shape[1];
        let grid = MTLSize::new(rows as u64, d0 as u64, 1);
        let group = MTLSize::new(8, 8, 1);
        self.run("slice_last_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&y.buffer), y.offset as u64);
            for (i, v) in [rows, d, off, d0].iter().enumerate() {
                enc.set_bytes((2 + i) as u64, 8, (v as *const usize).cast());
            }
        })
    }

    /// Causal unpad along time: X (B,T,C) → Y (B,T-U,C), Y = X[:, U:, :].
    pub fn slice_time_f32(&self, x: &GpuTensor, y: &GpuTensor, u: usize) -> Result<(), MetalError> {
        let (b, t, c) = (x.shape[0], x.shape[1], x.shape[2]);
        if y.shape != [b, t - u, c] {
            return Err(MetalError::Shape(format!(
                "slice_time out {:?} != [{b},{},{c}]",
                y.shape,
                t - u
            )));
        }
        let grid = MTLSize::new(b as u64, (t - u) as u64, c as u64);
        let group = MTLSize::new(1, 4, 4);
        self.run("slice_time_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&y.buffer), y.offset as u64);
            for (i, v) in [b, t, u, c].iter().enumerate() {
                enc.set_bytes((2 + i) as u64, 8, (v as *const usize).cast());
            }
        })
    }
}

/// A single-command-buffer session: build it, dispatch kernels, commit once,
/// wait once. This is the "no per-op CPU sync" primitive — a whole model step
/// goes in here. The command buffer is borrowed from an enclosing
/// autorelease pool, so the session only exists inside [`batch`].
pub struct BatchSession<'a> {
    ctx: &'a MetalCtx,
    cmd: &'a CommandBufferRef,
}

impl BatchSession<'_> {
    /// Dispatch one kernel into the current batch (no commit, no wait).
    pub fn run(
        &self,
        kernel: &'static str,
        grid: MTLSize,
        group: MTLSize,
        setup: impl FnOnce(&ComputeCommandEncoderRef),
    ) -> Result<(), MetalError> {
        let pipeline = self.ctx.pipeline(kernel)?;
        let enc = self.cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        setup(enc);
        enc.dispatch_threads(grid, group);
        enc.end_encoding();
        Ok(())
    }
}

/// Run a batch of kernels in ONE command buffer: dispatch inside `f`, then a
/// single commit + wait. The command buffer lives for the duration of `f`
/// (autorelease pool encloses it), so no dangling references.
pub fn batch(
    ctx: &MetalCtx,
    f: impl FnOnce(&BatchSession) -> Result<(), MetalError>,
) -> Result<(), MetalError> {
    autoreleasepool(|| {
        let cmd = ctx.queue.new_command_buffer();
        let session = BatchSession { ctx, cmd };
        f(&session)?;
        cmd.commit();
        cmd.wait_until_completed();
        Ok(())
    })
}

/// The Metal backend — the [`crate::backend::Backend`] implementation over
/// the own engine. Consumers use `&dyn Backend` and never touch this type.
pub struct MetalBackend {
    pub ctx: MetalCtx,
}

impl MetalBackend {
    pub fn new() -> Result<Self, MetalError> {
        Ok(Self {
            ctx: MetalCtx::new()?,
        })
    }
}

impl crate::backend::Backend for MetalBackend {
    fn supports_f16_weights(&self) -> bool {
        true
    }

    fn rope_standard(
        &self,
        x: &crate::backend::Tensor,
        pos: usize,
        hd: usize,
        theta: f32,
    ) -> Result<crate::backend::Tensor, crate::error::BackendError> {
        let (tq, d) = (x.shape()[1], x.shape()[2]);
        let y = self.zeros(vec![1, tq, d])?;
        let xg = x.inner_metal();
        let yg = y.inner_metal();
        let grid = metal::MTLSize::new(tq as u64, (d / hd) as u64, (hd / 2) as u64);
        let group = metal::MTLSize::new(1, 4, 4);
        self.ctx.run("rope_std_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&xg.buffer), xg.offset as u64);
            enc.set_buffer(1, Some(&yg.buffer), yg.offset as u64);
            enc.set_bytes(2, 8, (&tq as *const usize).cast());
            enc.set_bytes(3, 8, (&d as *const usize).cast());
            enc.set_bytes(4, 8, (&hd as *const usize).cast());
            enc.set_bytes(5, 8, (&pos as *const usize).cast());
            enc.set_bytes(6, 4, (&theta as *const f32).cast());
        })?;
        Ok(y)
    }

    fn rmsnorm_head_rope(
        &self,
        x: &crate::backend::Tensor,
        w: &crate::backend::Tensor,
        pos: usize,
        hd: usize,
        theta: f32,
        h: usize,
        eps: f32,
    ) -> Result<crate::backend::Tensor, crate::error::BackendError> {
        let (tq, d) = (x.shape()[1], x.shape()[2]);
        let y = self.zeros(vec![1, tq, d])?;
        let xg = x.inner_metal();
        let wg = w.inner_metal();
        let yg = y.inner_metal();
        // simdgroup на (токен, голову) — см. rmsnorm_head_rope_sg_f32.
        let (h32, nh32, hd32, pos32) = (h as u32, (tq * h) as u32, hd as u32, pos as u32);
        let grid = metal::MTLSize::new((tq * h * 32) as u64, 1, 1);
        let group = metal::MTLSize::new(64, 1, 1);
        self.ctx.run("rmsnorm_head_rope_sg_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&xg.buffer), xg.offset as u64);
            enc.set_buffer(1, Some(&wg.buffer), wg.offset as u64);
            enc.set_buffer(2, Some(&yg.buffer), yg.offset as u64);
            enc.set_bytes(3, 4, (&h32 as *const u32).cast());
            enc.set_bytes(4, 4, (&nh32 as *const u32).cast());
            enc.set_bytes(5, 4, (&hd32 as *const u32).cast());
            enc.set_bytes(6, 4, (&eps as *const f32).cast());
            enc.set_bytes(7, 4, (&pos32 as *const u32).cast());
            enc.set_bytes(8, 4, (&theta as *const f32).cast());
        })?;
        Ok(y)
    }

    fn qkv_norm_rope_kv(
        &self,
        qkv: &crate::backend::Tensor,
        q_norm: &crate::backend::Tensor,
        k_norm: &crate::backend::Tensor,
        kbuf: &crate::backend::Tensor,
        vbuf: &crate::backend::Tensor,
        h: usize,
        kvh: usize,
        hd: usize,
        rope_pos: usize,
        pos_kv: usize,
        theta: f32,
        eps: f32,
    ) -> Result<crate::backend::Tensor, crate::error::BackendError> {
        let t = qkv.numel() / ((h + 2 * kvh) * hd);
        let (kg, vg) = (kbuf.as_gpu(), vbuf.as_gpu());
        // KV-буферы обязаны быть f32 и вмещать строки [pos_kv, pos_kv + t).
        let cap = kg.shape[kg.shape.len() - 2];
        if kg.dtype != GpuDtype::F32 || vg.dtype != GpuDtype::F32 || pos_kv + t > cap {
            return Err(crate::error::BackendError::Shape(format!(
                "qkv_norm_rope_kv: KV {:?}/{:?} cap {cap}, нужны строки {pos_kv}..{}",
                kg.dtype,
                vg.dtype,
                pos_kv + t
            )));
        }
        let q = self.zeros(vec![1, t, h * hd])?;
        let (t32, h32, kvh32, hd32, rp32, pk32) =
            (t as u32, h as u32, kvh as u32, hd as u32, rope_pos as u32, pos_kv as u32);
        let grid = MTLSize::new((t * (h + 2 * kvh) * 32) as u64, 1, 1);
        let group = MTLSize::new(64, 1, 1);
        let (xg, wq, wk, qg) = (qkv.as_gpu(), q_norm.as_gpu(), k_norm.as_gpu(), q.as_gpu());
        self.ctx.run("qkv_norm_rope_kv_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&xg.buffer), xg.offset as u64);
            enc.set_buffer(1, Some(&wq.buffer), wq.offset as u64);
            enc.set_buffer(2, Some(&wk.buffer), wk.offset as u64);
            enc.set_buffer(3, Some(&qg.buffer), qg.offset as u64);
            enc.set_buffer(4, Some(&kg.buffer), kg.offset as u64);
            enc.set_buffer(5, Some(&vg.buffer), vg.offset as u64);
            enc.set_bytes(6, 4, (&t32 as *const u32).cast());
            enc.set_bytes(7, 4, (&h32 as *const u32).cast());
            enc.set_bytes(8, 4, (&kvh32 as *const u32).cast());
            enc.set_bytes(9, 4, (&hd32 as *const u32).cast());
            enc.set_bytes(10, 4, (&rp32 as *const u32).cast());
            enc.set_bytes(11, 4, (&pk32 as *const u32).cast());
            enc.set_bytes(12, 4, (&theta as *const f32).cast());
            enc.set_bytes(13, 4, (&eps as *const f32).cast());
        })?;
        Ok(q)
    }

    fn rope_mrope_text(
        &self,
        x: &crate::backend::Tensor,
        pos: usize,
        hd: usize,
        theta: f32,
        sections: [usize; 4],
    ) -> Result<crate::backend::Tensor, crate::error::BackendError> {
        let g = x.as_gpu();
        let (tq, d) = (g.shape[1], g.shape[2]);
        let y = GpuTensor::zeros(&self.ctx.device, vec![1, tq, d])?;
        let grid = MTLSize::new(tq as u64, (d / hd) as u64, (hd / 2) as u64);
        let group = MTLSize::new(1, 1, 8);
        self.ctx.run("rope_mrope_text_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&g.buffer), g.offset as u64);
            enc.set_buffer(1, Some(&y.buffer), y.offset as u64);
            enc.set_bytes(2, 8, (&tq as *const usize).cast());
            enc.set_bytes(3, 8, (&d as *const usize).cast());
            enc.set_bytes(4, 8, (&hd as *const usize).cast());
            enc.set_bytes(5, 8, (&pos as *const usize).cast());
            enc.set_bytes(6, 4, (&theta as *const f32).cast());
            let sects32: [u32; 4] = [
                sections[0] as u32,
                sections[1] as u32,
                sections[2] as u32,
                sections[3] as u32,
            ];
            enc.set_bytes(7, 16, (sects32.as_ptr() as *const u8).cast());
        })?;
        Ok(crate::backend::Tensor::from_gpu(y))
    }

    fn rope_mrope(
        &self,
        x: &crate::backend::Tensor,
        pos4d: &[u32],
        hd: usize,
        theta: f32,
        sections: [usize; 4],
    ) -> Result<crate::backend::Tensor, crate::error::BackendError> {
        let g = x.as_gpu();
        let (tq, d) = (g.shape[1], g.shape[2]);
        let y = GpuTensor::zeros(&self.ctx.device, vec![1, tq, d])?;
        let grid = MTLSize::new(tq as u64, (d / hd) as u64, (hd / 2) as u64);
        let group = MTLSize::new(1, 1, 8);
        let pos_buf = with_pool(|p| p.acquire(&self.ctx.device, pos4d.len() * 4));
        if pos_buf.contents().is_null() {
            return Err(MetalError::BufferAlloc.into());
        }
        unsafe {
            std::ptr::copy_nonoverlapping(pos4d.as_ptr(), pos_buf.contents().cast(), pos4d.len());
        }
        let res = self.ctx.run("rope_mrope_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&g.buffer), g.offset as u64);
            enc.set_buffer(1, Some(&y.buffer), y.offset as u64);
            enc.set_bytes(2, 8, (&tq as *const usize).cast());
            enc.set_bytes(3, 8, (&d as *const usize).cast());
            enc.set_bytes(4, 8, (&hd as *const usize).cast());
            enc.set_buffer(5, Some(&pos_buf), 0);
            enc.set_bytes(6, 4, (&theta as *const f32).cast());
            let sects32: [u32; 4] = [
                sections[0] as u32,
                sections[1] as u32,
                sections[2] as u32,
                sections[3] as u32,
            ];
            enc.set_bytes(7, 16, (sects32.as_ptr() as *const u8).cast());
        });
        with_pool(|p| p.push(&pos_buf));
        res?;
        Ok(crate::backend::Tensor::from_gpu(y))
    }

    // ── Ф9.4: гибрид Qwen3.5/3.8 (Gated DeltaNet) — кернелы ──

    fn gdn_conv(
        &self,
        x: &crate::backend::Tensor,
        w: &crate::backend::Tensor,
        state: &crate::backend::Tensor,
    ) -> Result<(crate::backend::Tensor, crate::backend::Tensor), crate::error::BackendError>
    {
        let (c, k) = (w.shape()[0], w.shape()[1]);
        let t = x.numel() / c;
        let y = self.zeros(x.shape().to_vec())?;
        let ns = self.zeros(vec![c, k - 1])?;
        self.ctx
            .gdn_conv1d_f32(x.as_gpu(), w.as_gpu(), state.as_gpu(), y.as_gpu(), t, c, k)?;
        self.ctx
            .gdn_conv_state_f32(x.as_gpu(), state.as_gpu(), ns.as_gpu(), t, c, k)?;
        Ok((y, ns))
    }

    fn gdn_l2norm(
        &self,
        x: &crate::backend::Tensor,
        hd: usize,
        eps: f32,
    ) -> Result<crate::backend::Tensor, crate::error::BackendError> {
        let y = self.zeros(x.shape().to_vec())?;
        self.ctx
            .gdn_l2norm_f32(x.as_gpu(), y.as_gpu(), x.numel() / hd, hd, eps)?;
        Ok(y)
    }

    fn gdn_gates(
        &self,
        alpha: &crate::backend::Tensor,
        beta: &crate::backend::Tensor,
        dt_bias: &crate::backend::Tensor,
        a: &crate::backend::Tensor,
    ) -> Result<(crate::backend::Tensor, crate::backend::Tensor), crate::error::BackendError>
    {
        let g = self.zeros(alpha.shape().to_vec())?;
        let b = self.zeros(beta.shape().to_vec())?;
        self.ctx.gdn_gates_f32(
            alpha.as_gpu(),
            beta.as_gpu(),
            dt_bias.as_gpu(),
            a.as_gpu(),
            g.as_gpu(),
            b.as_gpu(),
            alpha.numel(),
            dt_bias.numel(),
        )?;
        Ok((g, b))
    }

    fn gdn_delta_rule(
        &self,
        q: &crate::backend::Tensor,
        k: &crate::backend::Tensor,
        v: &crate::backend::Tensor,
        g: &crate::backend::Tensor,
        b: &crate::backend::Tensor,
        state: &crate::backend::Tensor,
        hk: usize,
    ) -> Result<(crate::backend::Tensor, crate::backend::Tensor), crate::error::BackendError>
    {
        let (hv, s) = (state.shape()[0], state.shape()[1]);
        let t = v.numel() / (hv * s);
        let out = self.zeros(vec![t, hv, s])?;
        // Стейт обновляется НА МЕСТЕ: одна threadgroup на v-голову проходит
        // все T токенов последовательно (рекуррентность по времени).
        self.ctx.gdn_delta_rule_f32(
            q.as_gpu(),
            k.as_gpu(),
            v.as_gpu(),
            g.as_gpu(),
            b.as_gpu(),
            state.as_gpu(),
            out.as_gpu(),
            t,
            hk,
            hv,
            s,
        )?;
        Ok((out, state.clone()))
    }

    fn gdn_norm_gated(
        &self,
        x: &crate::backend::Tensor,
        z: &crate::backend::Tensor,
        w: &crate::backend::Tensor,
        eps: f32,
    ) -> Result<crate::backend::Tensor, crate::error::BackendError> {
        let hd = w.numel();
        let y = self.zeros(x.shape().to_vec())?;
        self.ctx.gdn_norm_gated_f32(
            x.as_gpu(),
            z.as_gpu(),
            w.as_gpu(),
            y.as_gpu(),
            x.numel() / hd,
            hd,
            eps,
        )?;
        Ok(y)
    }

    fn rope_imrope_text(
        &self,
        x: &crate::backend::Tensor,
        pos: usize,
        hd: usize,
        theta: f32,
        sections: [usize; 4],
        n_rot: usize,
    ) -> Result<crate::backend::Tensor, crate::error::BackendError> {
        let (tq, d) = (x.shape()[1], x.shape()[2]);
        let y = self.zeros(vec![1, tq, d])?;
        self.ctx.rope_imrope_text_f32(
            x.as_gpu(),
            y.as_gpu(),
            tq,
            d,
            hd,
            pos,
            theta,
            sections,
            n_rot,
        )?;
        Ok(y)
    }

    fn zeros(
        &self,
        shape: Vec<usize>,
    ) -> Result<crate::backend::Tensor, crate::error::BackendError> {
        Ok(crate::backend::Tensor::from_gpu(GpuTensor::zeros(
            &self.ctx.device,
            shape,
        )?))
    }

    fn from_cpu_f32(
        &self,
        data: &[f32],
        shape: Vec<usize>,
    ) -> Result<crate::backend::Tensor, crate::error::BackendError> {
        Ok(crate::backend::Tensor::from_gpu(GpuTensor::from_cpu_f32(
            &self.ctx.device,
            data,
            shape,
        )?))
    }

    /// fp16 GEMM weights: stored as true fp16 (2 bytes/element) — the gemm
    /// kernels read them narrowed and accumulate in fp32.
    fn from_cpu_f16(
        &self,
        data: &[half::f16],
        shape: Vec<usize>,
    ) -> Result<crate::backend::Tensor, crate::error::BackendError> {
        Ok(crate::backend::Tensor::from_gpu(GpuTensor::from_cpu_f16(
            &self.ctx.device,
            data,
            shape,
        )?))
    }

    /// bf16 GEMM weights: stored as true bf16 (bit-pattern passthrough).
    fn from_cpu_bf16(
        &self,
        data: &[half::bf16],
        shape: Vec<usize>,
    ) -> Result<crate::backend::Tensor, crate::error::BackendError> {
        Ok(crate::backend::Tensor::from_gpu(GpuTensor::from_cpu_bf16(
            &self.ctx.device,
            data,
            shape,
        )?))
    }

    fn to_cpu_f32(
        &self,
        t: &crate::backend::Tensor,
    ) -> Result<Vec<f32>, crate::error::BackendError> {
        Ok(t.as_gpu().to_cpu_f32(&self.ctx)?)
    }

    fn to_cpu_f16(
        &self,
        t: &crate::backend::Tensor,
    ) -> Result<Vec<half::f16>, crate::error::BackendError> {
        Ok(t.as_gpu().to_cpu_f16(&self.ctx)?)
    }

    fn deepstack(
        &self,
        x: &crate::backend::Tensor,
        embd: &crate::backend::Tensor,
        shift: usize,
    ) -> Result<crate::backend::Tensor, crate::error::BackendError> {
        let y = GpuTensor::zeros(&self.ctx.device, x.as_gpu().shape.clone())?;
        self.ctx
            .deepstack_add_f32(x.as_gpu(), embd.as_gpu(), shift, &y)?;
        Ok(crate::backend::Tensor::from_gpu(y))
    }

    /// Metal's KV kernels are f32-only (`kv_update_f32`/`attn_ar_f32`; the
    /// true f16 KV kernels — `attn_ar_v4_f16f16` — are CUDA's). An f16-typed
    /// KV allocation therefore stores f32 values with half the bytes they
    /// need: the attention ran past the end of the buffer from ~cap/2 rows on
    /// (garbage/NaN in long contexts) and every host-side read of the cache
    /// (the window roll) either refused or read halfs of f32 words. Coerce to
    /// f32 until real f16 KV kernels exist here.
    fn supported_kv_format(&self, _fmt: crate::dtype::KvFormat) -> crate::dtype::KvFormat {
        crate::dtype::KvFormat::F32
    }

    fn begin_batch(&self) -> Result<(), crate::error::BackendError> {
        self.ctx.begin_batch();
        Ok(())
    }

    fn end_batch(&self) -> Result<(), crate::error::BackendError> {
        self.ctx.end_batch()?;
        Ok(())
    }

    fn batch_enter_count(&self) -> u64 {
        self.ctx.batch_enter_count()
    }

    fn gemm(
        &self,
        a: &crate::backend::Tensor,
        b: &crate::backend::Tensor,
        c: &crate::backend::Tensor,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.gemm_f32(a.as_gpu(), b.as_gpu(), c.as_gpu())?;
        Ok(())
    }

    fn gemm_nt(
        &self,
        a: &crate::backend::Tensor,
        b: &crate::backend::Tensor,
        c: &crate::backend::Tensor,
    ) -> Result<(), crate::error::BackendError> {
        let bg = b.as_gpu();
        // Квантованные веса — через dequant-in-kernel GEMM/MMV (Ф1.3/1.4).
        if matches!(
            bg.dtype,
            GpuDtype::Q4_0 | GpuDtype::Q8_0 | GpuDtype::Q4_K | GpuDtype::Q6_K
        ) {
            if a.as_gpu().shape[0] == 1 {
                // mv_quantized — MMV-примитив: a 1D [k], c 1D [n] (тот же буфер).
                let k = a.as_gpu().shape[1];
                let n = bg.shape[0];
                let av = a.view(vec![k]);
                let cv = c.view(vec![n]);
                self.ctx.mv_quantized(av.as_gpu(), bg, cv.as_gpu())?;
            } else {
                self.ctx.gemm_nt_quantized(a.as_gpu(), bg, c.as_gpu())?;
            }
            return Ok(());
        }
        self.ctx.gemm_nt_f32(a.as_gpu(), bg, c.as_gpu())?;
        Ok(())
    }

    fn from_quantized(
        &self,
        data: &[u8],
        shape: Vec<usize>,
        format: crate::dtype::WeightFormat,
    ) -> Result<crate::backend::Tensor, crate::error::BackendError> {
        let dtype = match format {
            crate::dtype::WeightFormat::Q4_0 => GpuDtype::Q4_0,
            crate::dtype::WeightFormat::Q8_0 => GpuDtype::Q8_0,
            crate::dtype::WeightFormat::Q4_K => GpuDtype::Q4_K,
            crate::dtype::WeightFormat::Q6_K => GpuDtype::Q6_K,
            other => {
                return Err(crate::error::BackendError::Shape(format!(
                    "from_quantized: {other:?} is not quantized"
                )));
            }
        };
        let g = GpuTensor::from_quantized(&self.ctx.device, data, shape.clone(), dtype)?;
        Ok(crate::backend::Tensor::from_gpu(g))
    }

    fn mul(
        &self,
        a: &crate::backend::Tensor,
        b: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.mul_f32(a.as_gpu(), b.as_gpu(), y.as_gpu())?;
        Ok(())
    }

    fn add(
        &self,
        a: &crate::backend::Tensor,
        b: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.add_f32(a.as_gpu(), b.as_gpu(), y.as_gpu())?;
        Ok(())
    }

    fn sub(
        &self,
        a: &crate::backend::Tensor,
        b: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.sub_f32(a.as_gpu(), b.as_gpu(), y.as_gpu())?;
        Ok(())
    }

    fn div(
        &self,
        a: &crate::backend::Tensor,
        b: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.div_f32(a.as_gpu(), b.as_gpu(), y.as_gpu())?;
        Ok(())
    }

    fn add_scalar(
        &self,
        x: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
        c: f32,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.add_scalar_f32(x.as_gpu(), y.as_gpu(), c)?;
        Ok(())
    }

    fn scale(
        &self,
        x: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
        c: f32,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.scale_f32(x.as_gpu(), y.as_gpu(), c)?;
        Ok(())
    }

    fn add_bias(
        &self,
        x: &crate::backend::Tensor,
        bias: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx
            .add_bias_f32(x.as_gpu(), bias.as_gpu(), y.as_gpu())?;
        Ok(())
    }

    fn sqrt(
        &self,
        x: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.sqrt_f32(x.as_gpu(), y.as_gpu())?;
        Ok(())
    }

    fn log(
        &self,
        x: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.log_f32(x.as_gpu(), y.as_gpu())?;
        Ok(())
    }

    fn max_scalar(
        &self,
        x: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
        floor: f32,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.max_scalar_f32(x.as_gpu(), y.as_gpu(), floor)?;
        Ok(())
    }

    fn relu(
        &self,
        x: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.relu_f32(x.as_gpu(), y.as_gpu())?;
        Ok(())
    }

    fn leaky_relu_clip(
        &self,
        x: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.leaky_relu_clip_f32(x.as_gpu(), y.as_gpu())?;
        Ok(())
    }

    fn sigmoid(
        &self,
        x: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.sigmoid_f32(x.as_gpu(), y.as_gpu())?;
        Ok(())
    }

    fn tanh(
        &self,
        x: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.tanh_f32(x.as_gpu(), y.as_gpu())?;
        Ok(())
    }

    fn silu(
        &self,
        x: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.silu_f32(x.as_gpu(), y.as_gpu())?;
        Ok(())
    }

    fn silu_mul_fused(
        &self,
        x: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
        ffn: usize,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.silu_mul_fused_f32(x.as_gpu(), y.as_gpu(), ffn)?;
        Ok(())
    }

    fn dflash2_conv(
        &self,
        x: &crate::backend::Tensor,
        base: &crate::backend::Tensor,
        dynamic: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
        kernel: usize,
        group_size: usize,
        block_size: usize,
        side: usize,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.dflash2_conv_f32(
            x.as_gpu(),
            base.as_gpu(),
            dynamic.as_gpu(),
            y.as_gpu(),
            kernel,
            group_size,
            block_size,
            side,
        )?;
        Ok(())
    }

    fn gelu_tanh(
        &self,
        x: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.gelu_tanh_f32(x.as_gpu(), y.as_gpu())?;
        Ok(())
    }

    fn gelu_erf(
        &self,
        x: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.gelu_erf_f32(x.as_gpu(), y.as_gpu())?;
        Ok(())
    }

    fn merge2x2_add(
        &self,
        x: &crate::backend::Tensor,
        bias: &crate::backend::Tensor,
        pos: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
        per: usize,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx
            .merge2x2_add_f32(x.as_gpu(), bias.as_gpu(), pos.as_gpu(), y.as_gpu(), per)?;
        Ok(())
    }

    fn group4(
        &self,
        x: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.group4_f32(x.as_gpu(), y.as_gpu())?;
        Ok(())
    }

    fn concat4(
        &self,
        chunks: &[crate::backend::Tensor],
        y: &crate::backend::Tensor,
    ) -> Result<(), crate::error::BackendError> {
        let gpus: Vec<_> = chunks.iter().map(|c| c.as_gpu().clone()).collect();
        self.ctx.concat4_f32(&gpus, y.as_gpu())?;
        Ok(())
    }

    fn snake(
        &self,
        x: &crate::backend::Tensor,
        a: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.snake_f32(x.as_gpu(), a.as_gpu(), y.as_gpu())?;
        Ok(())
    }

    fn rmsnorm(
        &self,
        x: &crate::backend::Tensor,
        w: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
        eps: f32,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx
            .rmsnorm_f32(x.as_gpu(), w.as_gpu(), y.as_gpu(), eps)?;
        Ok(())
    }

    fn layernorm(
        &self,
        x: &crate::backend::Tensor,
        w: &crate::backend::Tensor,
        b: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
        eps: f32,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx
            .layernorm_f32(x.as_gpu(), w.as_gpu(), b.as_gpu(), y.as_gpu(), eps)?;
        Ok(())
    }

    fn conv1d(
        &self,
        x: &crate::backend::Tensor,
        w: &crate::backend::Tensor,
        bias: Option<&crate::backend::Tensor>,
        stride: usize,
        pad_l: usize,
        pad_r: usize,
        dil: usize,
        y: &crate::backend::Tensor,
    ) -> Result<(), crate::error::BackendError> {
        let params = Conv1dParams {
            stride,
            pad: pad_l,
            dilation: dil,
        };
        let dense = w.as_gpu().shape[1] == x.as_gpu().shape[2];
        self.ctx.conv1d_groups_f32(
            x.as_gpu(),
            w.as_gpu(),
            bias.map(|b| b.as_gpu()),
            params,
            pad_r,
            true,
            true,
            dense,
            y.as_gpu(),
        )?;
        Ok(())
    }

    fn conv1d_groups(
        &self,
        x: &crate::backend::Tensor,
        w: &crate::backend::Tensor,
        bias: Option<&crate::backend::Tensor>,
        stride: usize,
        pad_l: usize,
        pad_r: usize,
        dil: usize,
        groups: usize,
        y: &crate::backend::Tensor,
    ) -> Result<(), crate::error::BackendError> {
        // The Metal kernel derives Cin_g from W (Cout, Cin_g, K); dense is
        // true when every out channel reads all Cin input channels.
        let _ = groups; // implied by W's Cin_g dim (see the kernel)
        let params = Conv1dParams {
            stride,
            pad: pad_l,
            dilation: dil,
        };
        let dense = w.as_gpu().shape[1] == x.as_gpu().shape[2];
        self.ctx.conv1d_groups_f32(
            x.as_gpu(),
            w.as_gpu(),
            bias.map(|b| b.as_gpu()),
            params,
            pad_r,
            true,
            true,
            dense,
            y.as_gpu(),
        )?;
        Ok(())
    }

    fn conv2d_nchw(
        &self,
        x: &crate::backend::Tensor,
        w: &crate::backend::Tensor,
        bias: Option<&crate::backend::Tensor>,
        pad_h: usize,
        pad_w: usize,
        stride_t: usize,
        stride_f: usize,
        time_major_out: bool,
        y: &crate::backend::Tensor,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.conv2d_nchw_f32(
            x.as_gpu(),
            w.as_gpu(),
            bias.map(|b| b.as_gpu()),
            pad_h,
            pad_w,
            stride_t,
            stride_f,
            time_major_out,
            y.as_gpu(),
        )?;
        Ok(())
    }

    fn maxpool2d(
        &self,
        x: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
        kh: usize,
        kw: usize,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.maxpool2d_f32(x.as_gpu(), y.as_gpu(), kh, kw)?;
        Ok(())
    }

    fn conv_transpose1d(
        &self,
        x: &crate::backend::Tensor,
        w: &crate::backend::Tensor,
        bias: Option<&crate::backend::Tensor>,
        stride: usize,
        y: &crate::backend::Tensor,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.conv_transpose1d_f32(
            x.as_gpu(),
            w.as_gpu(),
            bias.map(|b| b.as_gpu()),
            stride,
            y.as_gpu(),
        )?;
        Ok(())
    }

    fn im2col1d(
        &self,
        x: &crate::backend::Tensor,
        xc: &crate::backend::Tensor,
        k: usize,
        stride: usize,
        pad_l: usize,
        dil: usize,
        tout: usize,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx
            .im2col1d_f32(x.as_gpu(), xc.as_gpu(), k, stride, pad_l, dil, tout)?;
        Ok(())
    }

    fn im2col_t1d(
        &self,
        x: &crate::backend::Tensor,
        xc: &crate::backend::Tensor,
        k: usize,
        stride: usize,
        tout: usize,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx
            .im2col_t1d_f32(x.as_gpu(), xc.as_gpu(), k, stride, tout)?;
        Ok(())
    }

    fn slice_last(
        &self,
        x: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
        off: usize,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.slice_last_f32(x.as_gpu(), y.as_gpu(), off)?;
        Ok(())
    }

    fn slice_time(
        &self,
        x: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
        u: usize,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.slice_time_f32(x.as_gpu(), y.as_gpu(), u)?;
        Ok(())
    }

    fn softmax_last_dim(
        &self,
        x: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.softmax_last_dim_f32(x.as_gpu(), y.as_gpu())?;
        Ok(())
    }

    fn embedding(
        &self,
        idx: &[u32],
        table: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.embedding_f32(idx, table.as_gpu(), y.as_gpu())?;
        Ok(())
    }

    fn embedding_quant(
        &self,
        idx: &[u32],
        table: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
    ) -> Result<(), crate::error::BackendError> {
        if table.as_gpu().dtype == GpuDtype::F32 {
            self.ctx.embedding_f32(idx, table.as_gpu(), y.as_gpu())?;
        } else {
            self.ctx.embedding_quant_f32(idx, table.as_gpu(), y.as_gpu())?;
        }
        Ok(())
    }

    fn embedding_override(
        &self,
        idx: &[u32],
        table: &crate::backend::Tensor,
        vision: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
        override_id: u32,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.embedding_override_f32(
            idx,
            table.as_gpu(),
            vision.as_gpu(),
            y.as_gpu(),
            override_id,
        )?;
        Ok(())
    }

    fn rope(
        &self,
        x: &crate::backend::Tensor,
        cos: &crate::backend::Tensor,
        sin: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
        rope_dim: usize,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx
            .rope_f32(x.as_gpu(), cos.as_gpu(), sin.as_gpu(), y.as_gpu(), rope_dim)?;
        Ok(())
    }

    fn rope_interleaved(
        &self,
        x: &crate::backend::Tensor,
        cos: &crate::backend::Tensor,
        sin: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
        rope_dim: usize,
        start: usize,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.rope_interleaved_f32(
            x.as_gpu(),
            cos.as_gpu(),
            sin.as_gpu(),
            y.as_gpu(),
            rope_dim,
            start,
        )?;
        Ok(())
    }

    fn rope_head(
        &self,
        x: &crate::backend::Tensor,
        cos: &crate::backend::Tensor,
        sin: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
        h: usize,
        rope_dim: usize,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.rope_head_f32(
            x.as_gpu(),
            cos.as_gpu(),
            sin.as_gpu(),
            y.as_gpu(),
            h,
            rope_dim,
        )?;
        Ok(())
    }

    fn rmsnorm_head(
        &self,
        x: &crate::backend::Tensor,
        w: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
        h: usize,
        eps: f32,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx
            .rmsnorm_head_f32(x.as_gpu(), w.as_gpu(), y.as_gpu(), h, eps)?;
        Ok(())
    }

    fn layernorm_head(
        &self,
        x: &crate::backend::Tensor,
        w: &crate::backend::Tensor,
        b: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
        h: usize,
        eps: f32,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx
            .layernorm_head_f32(x.as_gpu(), w.as_gpu(), b.as_gpu(), y.as_gpu(), h, eps)?;
        Ok(())
    }

    fn mul_rows(
        &self,
        x: &crate::backend::Tensor,
        w: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.mul_rows_f32(x.as_gpu(), w.as_gpu(), y.as_gpu())?;
        Ok(())
    }

    fn mul_vec(
        &self,
        x: &crate::backend::Tensor,
        w: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.mul_vec_f32(x.as_gpu(), w.as_gpu(), y.as_gpu())?;
        Ok(())
    }

    fn glu(
        &self,
        x: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.glu_f32(x.as_gpu(), y.as_gpu())?;
        Ok(())
    }

    fn repeat(
        &self,
        x: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.repeat_f32(x.as_gpu(), y.as_gpu())?;
        Ok(())
    }

    fn add_scaled(
        &self,
        a: &crate::backend::Tensor,
        b: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
        scale: f32,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx
            .add_scaled_f32(a.as_gpu(), b.as_gpu(), y.as_gpu(), scale)?;
        Ok(())
    }

    fn add_bias2d(
        &self,
        x: &crate::backend::Tensor,
        b: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx
            .add_bias2d_f32(x.as_gpu(), b.as_gpu(), y.as_gpu())?;
        Ok(())
    }

    fn attn_ar(
        &self,
        q: &crate::backend::Tensor,
        k: &crate::backend::Tensor,
        v: &crate::backend::Tensor,
        mask: Option<&crate::backend::Tensor>,
        y: &crate::backend::Tensor,
        n_heads: usize,
        kv_heads: usize,
        hd: usize,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.attn_ar_f32(
            q.as_gpu(),
            k.as_gpu(),
            v.as_gpu(),
            mask.map(|m| m.as_gpu()),
            y.as_gpu(),
            n_heads,
            kv_heads,
            hd,
        )?;
        Ok(())
    }

    fn mhsa(
        &self,
        q: &crate::backend::Tensor,
        k: &crate::backend::Tensor,
        v: &crate::backend::Tensor,
        sc: &crate::backend::Tensor,
        mask: Option<&crate::backend::Tensor>,
        mode: u32,
        y: &crate::backend::Tensor,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.mhsa_f32(
            q.as_gpu(),
            k.as_gpu(),
            v.as_gpu(),
            sc.as_gpu(),
            mask.map(|m| m.as_gpu()),
            mode,
            y.as_gpu(),
        )?;
        Ok(())
    }

    fn uniform_attn(
        &self,
        v: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.uniform_attn_f32(v.as_gpu(), y.as_gpu())?;
        Ok(())
    }

    fn kv_update(
        &self,
        dst: &crate::backend::Tensor,
        src: &crate::backend::Tensor,
        pos: usize,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.kv_update_f32(dst.as_gpu(), src.as_gpu(), pos)?;
        Ok(())
    }

    fn codebook_search(
        &self,
        x: &crate::backend::Tensor,
        table: &crate::backend::Tensor,
        idx: &mut [u32],
    ) -> Result<(), crate::error::BackendError> {
        self.ctx
            .codebook_search_f32(x.as_gpu(), table.as_gpu(), idx)?;
        Ok(())
    }
}

// ── Квант-GEMM/MV: сверка с CPU-референсом (Ф1.3/1.4) ────────────────────────

#[cfg(all(test, feature = "metal"))]
mod quant_gemm_tests {
    use super::*;
    use crate::gguf;
    use crate::weights_map::{WeightsMap, dequant_f32};
    use std::path::PathBuf;

    fn fixture(name: &str) -> Vec<u8> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join(name);
        std::fs::read(&path).unwrap_or_else(|e| panic!("fixture {name}: {e}"))
    }

    /// f32-веса источника (f16-фикстура — те же сиды, что у всех квантов).
    fn source_f32(name: &str) -> Vec<f32> {
        let buf = fixture(name);
        let f = gguf::parse(&buf).unwrap();
        let m = WeightsMap::from_gguf(&f).unwrap();
        dequant_f32(m.get("token_embd.weight").unwrap())
    }

    // Тестовые энкодеры Q4_K/Q6_K (K=256-синтетика; логика как в fixture-gen).
    /// Канонический Q4_K (ggml block_q4_K, 144 Б/256): d (f16), dmin (f16),
    /// scales[12] (6-битные scale/min 8 подблоков, get_scale_min_k4),
    /// qs[128] (чанк-мажор: байт 32c+l = {low: 64c+l, high: 64c+32+l}).
    /// Старая фикстура писала 148 Б (кастомный диалект scales[16]) — загрузчик
    /// давно читает канонические 144 Б, и тест падал на size-check.
    /// min-член не используем (dmin = 0): кодируются модули — тест сравнивает
    /// Metal с CPU-деквантом ТЕХ ЖЕ байт.
    fn enc_q4_k(w: &[f32]) -> Vec<u8> {
        let mut out = Vec::new();
        for sb in w.chunks_exact(256) {
            let smax = sb.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            let d = if smax > 0.0 { smax / 15.0 } else { 0.0 };
            out.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes()); // dmin = 0
            let mut sub_sc = [1u8; 8];
            for (j, blk) in sb.chunks_exact(32).enumerate() {
                let bmax = blk.iter().fold(0.0f32, |m, x| m.max(x.abs()));
                sub_sc[j] = if d > 0.0 && bmax > 0.0 {
                    ((bmax / d).round() as u8).clamp(1, 15)
                } else {
                    1
                };
            }
            // scales[0..4] — sc подблоков 0..3 (старшие 2 бита не нужны при
            // sc ≤ 15), scales[4..8] — mins подблоков 0..3 = 0,
            // scales[8..12] — sc подблоков 4..7 в младшем ниббле (mins = 0).
            let mut scales = [0u8; 12];
            scales[0..4].copy_from_slice(&sub_sc[0..4]);
            scales[8..12].copy_from_slice(&sub_sc[4..8]);
            let mut qs = [0u8; 128];
            for c in 0..4 {
                let (s_lo, s_hi) = (sub_sc[2 * c] as f32, sub_sc[2 * c + 1] as f32);
                for l in 0..32 {
                    let q_lo =
                        ((sb[64 * c + l].abs() / (d * s_lo)).round() as i32).clamp(0, 15) as u8;
                    let q_hi = ((sb[64 * c + 32 + l].abs() / (d * s_hi)).round() as i32)
                        .clamp(0, 15) as u8;
                    qs[32 * c + l] = (q_lo & 0x0F) | (q_hi << 4);
                }
            }
            out.extend_from_slice(&scales);
            out.extend_from_slice(&qs);
        }
        out
    }

    /// Канонический Q6_K (ggml block_q6_K, 210 Б/256): ql[128] (младшие
    /// 4 бита), qh[64] (старшие 2 бита), scales[16] (int8 на 16-элементные
    /// подблоки), d (f16) в КОНЦЕ блока. y = d·sc·(q − 32).
    /// Прежняя фикстура кодировала кастомный диалект (d в начале, 8 × 8-бит
    /// масштабов) — он совпадал со старым кернелом, но не с GGUF/эталоном.
    fn enc_q6_k(w: &[f32]) -> Vec<u8> {
        let mut out = Vec::new();
        for sb in w.chunks_exact(256) {
            let smax = sb.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            // q − 32 ∈ [−32, 31]: d·s ≈ bmax/31 даёт полный диапазон.
            let d = if smax > 0.0 { smax / (31.0 * 127.0) } else { 0.0 };
            let mut ql = [0u8; 128];
            let mut qh = [0u8; 64];
            let mut scales = [0u8; 16];
            for half in 0..2usize {
                let base = half * 128;
                for l in 0..32usize {
                    let is = l / 16;
                    for (k, off) in [0usize, 32, 64, 96].into_iter().enumerate() {
                        let e = base + l + off;
                        let blk16 = base + is * 16;
                        let bmax = sb[blk16..blk16 + 16]
                            .iter()
                            .fold(0.0f32, |m, x| m.max(x.abs()));
                        let sidx = half * 8 + is + 2 * k;
                        let sc = if d > 0.0 && bmax > 0.0 {
                            ((bmax / (31.0 * d)).round() as i32).clamp(1, 127) as u8
                        } else {
                            1
                        };
                        scales[sidx] = sc;
                        let q = ((sb[e] / (d * sc as f32)).round() as i32 + 32).clamp(0, 63) as u8;
                        let lo = q & 0x0F;
                        let hi2 = (q >> 4) & 0x03;
                        // низкие нибблы: q, l+32 → ql[l], ql[l+32];
                        // высокие нибблы: q, l+32 → ql[l], ql[l+32] (старшие 4)
                        let (qi, sh) = ((l + off / 32 * 32), 0u32);
                        let _ = qi;
                        let _ = sh;
                        if off == 0 || off == 32 {
                            let idx = half * 64 + l + off;
                            ql[idx] |= lo;
                        } else {
                            let idx = half * 64 + l + (off - 64);
                            ql[idx] |= lo << 4;
                        }
                        qh[half * 32 + l] |= hi2 << (2 * k);
                    }
                }
            }
            out.extend_from_slice(&ql);
            out.extend_from_slice(&qh);
            out.extend_from_slice(&scales);
            out.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        }
        out
    }

    /// (bytes, shape[N,K], dtype, per-elem tol) — источники для GPU-теста.
    fn cases() -> Vec<(Vec<u8>, Vec<usize>, GpuDtype, f32)> {
        let src: Vec<f32> = (0..8 * 256)
            .map(|i| ((i as f32 * 7919.0).sin() * 0.5))
            .collect();
        // Q8_0 128×96 (синтетический): полные тайлы tiled-GEMM (64 по N,
        // 32 по M) и K из нескольких блоков — фикстура 64×32 их не задевает.
        let q8_big: Vec<u8> = (0..128 * 96 / 32)
            .flat_map(|b| {
                let mut blk = half::f16::from_f32(0.01 + (b % 7) as f32 * 0.003)
                    .to_bits()
                    .to_le_bytes()
                    .to_vec();
                blk.extend((0..32).map(|j| (((b * 37 + j * 11) % 255) as i32 - 127) as i8 as u8));
                blk
            })
            .collect();
        vec![
            (q8_big, vec![128, 96], GpuDtype::Q8_0, 0.05),
            (enc_q4_k(&src), vec![8, 256], GpuDtype::Q4_K, 0.15),
            (enc_q6_k(&src), vec![8, 256], GpuDtype::Q6_K, 0.06),
            // Q4_0/Q8_0 — прямо из фикстур (K=32, те же веса, что у f16)
            (
                {
                    let buf = fixture("mini-q4_0.gguf");
                    let f = gguf::parse(&buf).unwrap();
                    let m = WeightsMap::from_gguf(&f).unwrap();
                    m.get("token_embd.weight").unwrap().raw().to_vec()
                },
                vec![64, 32],
                GpuDtype::Q4_0,
                0.15,
            ),
            (
                {
                    let buf = fixture("mini-q8_0.gguf");
                    let f = gguf::parse(&buf).unwrap();
                    let m = WeightsMap::from_gguf(&f).unwrap();
                    m.get("token_embd.weight").unwrap().raw().to_vec()
                },
                vec![64, 32],
                GpuDtype::Q8_0,
                0.02,
            ),
        ]
    }

    /// CPU-референс: деквант → C[m,n] = Σ A[m,k]·Bf[n,k].
    fn cpu_ref(a: &[f32], m: usize, k: usize, bf: &[f32], n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for mm in 0..m {
            for nn in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += a[mm * k + kk] * bf[nn * k + kk];
                }
                c[mm * n + nn] = acc;
            }
        }
        c
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn quantized_gemm_matches_cpu() {
        let ctx = MetalCtx::new().expect("MetalCtx");
        let dev = &ctx.device;
        // m: хвостовой тайл (3), ровно тайл (32), несколько тайлов с
        // хвостом (77) — у Q8_0 разные пути выгрузки tiled-GEMM.
        for ((bytes, shape, dtype, tol), m) in cases()
            .into_iter()
            .flat_map(|c| [3usize, 32, 77].map(move |m| (c.clone(), m)))
        {
            let n = shape[0];
            let k = shape[1];
            let b = GpuTensor::from_quantized(dev, &bytes, shape.clone(), dtype).unwrap();
            // CPU-референс: деквант тех же байт
            let tmp = WeightsMap {
                arch: "t".into(),
                tensors: vec![crate::weights_map::WeightTensor {
                    name: "w".into(),
                    shape: shape.iter().map(|&d| d as u64).collect(),
                    format: match dtype {
                        GpuDtype::Q4_0 => crate::dtype::WeightFormat::Q4_0,
                        GpuDtype::Q8_0 => crate::dtype::WeightFormat::Q8_0,
                        GpuDtype::Q4_K => crate::dtype::WeightFormat::Q4_K,
                        GpuDtype::Q6_K => crate::dtype::WeightFormat::Q6_K,
                        _ => unreachable!(),
                    },
                    offset: 0,
                    data: &bytes,
                }],
                storage: vec![],
            };
            let bf = dequant_f32(&tmp.tensors[0]);
            let a_data: Vec<f32> = (0..m * k).map(|i| ((i as f32 * 104729.0).sin())).collect();
            let c_ref = cpu_ref(&a_data, m, k, &bf, n);

            let a = GpuTensor::from_cpu_f32(dev, &a_data, vec![m, k]).unwrap();
            let c = GpuTensor::zeros(dev, vec![m, n]).unwrap();
            ctx.gemm_nt_quantized(&a, &b, &c).unwrap();
            let got = c.to_cpu_f32(&ctx).unwrap();
            let bound = tol * k as f32 * 1.5;
            let diff = max_diff(&got, &c_ref);
            assert!(diff <= bound, "{dtype:?}: |Δ| max {diff} > {bound}");
        }
    }

    #[test]
    fn quantized_mv_matches_cpu() {
        let ctx = MetalCtx::new().expect("MetalCtx");
        let dev = &ctx.device;
        for (bytes, shape, dtype, tol) in cases() {
            let n = shape[0];
            let k = shape[1];
            let b = GpuTensor::from_quantized(dev, &bytes, shape.clone(), dtype).unwrap();
            let tmp = WeightsMap {
                arch: "t".into(),
                tensors: vec![crate::weights_map::WeightTensor {
                    name: "w".into(),
                    shape: shape.iter().map(|&d| d as u64).collect(),
                    format: match dtype {
                        GpuDtype::Q4_0 => crate::dtype::WeightFormat::Q4_0,
                        GpuDtype::Q8_0 => crate::dtype::WeightFormat::Q8_0,
                        GpuDtype::Q4_K => crate::dtype::WeightFormat::Q4_K,
                        GpuDtype::Q6_K => crate::dtype::WeightFormat::Q6_K,
                        _ => unreachable!(),
                    },
                    offset: 0,
                    data: &bytes,
                }],
                storage: vec![],
            };
            let bf = dequant_f32(&tmp.tensors[0]);
            let a_data: Vec<f32> = (0..k).map(|i| ((i as f32 * 15485863.0).sin())).collect();
            let c_ref = cpu_ref(&a_data, 1, k, &bf, n);

            let a = GpuTensor::from_cpu_f32(dev, &a_data, vec![k]).unwrap();
            let c = GpuTensor::zeros(dev, vec![n]).unwrap();
            ctx.mv_quantized(&a, &b, &c).unwrap();
            let got = c.to_cpu_f32(&ctx).unwrap();
            let bound = tol * k as f32 * 1.5;
            let diff = max_diff(&got, &c_ref);
            assert!(diff <= bound, "{dtype:?} mv: |Δ| max {diff} > {bound}");
        }
    }
}

#[cfg(test)]
mod batch_debug {
    use super::*;
    fn bytes_rep(n: usize, k: usize) -> Vec<u8> {
        let mut b = Vec::new();
        let half_one = 0x3C00u16.to_le_bytes();
        for _ in 0..n * (k / 32) {
            b.extend_from_slice(&half_one);
            for i in 0..32 {
                b.push((i % 64) as u8);
            }
        }
        b
    }
    #[test]
    fn q8_batch_vs_naive() {
        let ctx = MetalCtx::new().unwrap();
        // Q8_0: n=2, k=32: scale=1.0 (f16-le), q = [0..31], [31..0]
        let mut bytes = Vec::new();
        let half_one = 0x3C00u16.to_le_bytes();
        bytes.extend_from_slice(&half_one);
        for i in 0..32 {
            bytes.push(i as u8);
        }
        bytes.extend_from_slice(&half_one);
        for i in 0..32 {
            bytes.push((31 - i) as u8);
        }
        let b =
            GpuTensor::from_quantized(&ctx.device, &bytes, vec![2, 32], GpuDtype::Q8_0).unwrap();
        // A: m=2, k=32: a0 = [1,0,0,...], a1 = [0..31]
        let mut ad = vec![0f32; 2 * 32];
        ad[0] = 1.0;
        for i in 0..32 {
            ad[32 + i] = i as f32;
        }
        let a = GpuTensor::from_cpu_f32(&ctx.device, &ad, vec![2, 32]).unwrap();
        let c = GpuTensor::zeros(&ctx.device, vec![2, 2]).unwrap();
        // barrier-проверка
        {
            let c4 = GpuTensor::zeros(&ctx.device, vec![32]).unwrap();
            ctx.run(
                "tg_barrier_test",
                metal::MTLSize::new(1, 1, 32),
                metal::MTLSize::new(1, 1, 32),
                |enc| {
                    enc.set_buffer(0, Some(&c4.buffer), c4.offset as u64);
                },
            )
            .unwrap();
            let v4 = c4.to_cpu_f32(&ctx).unwrap();
            assert!(v4.iter().all(|&x| x == 42.0), "barrier сломан: {v4:?}");
        }
        ctx.gemm_nt_quantized(&a, &b, &c).unwrap();
        let cv = c.to_cpu_f32(&ctx).unwrap();
        // микро-бенч: attn_ar 9 токенов × 36 кэш × 32 головы
        {
            let q9 =
                GpuTensor::from_cpu_f32(&ctx.device, &vec![0.1f32; 1 * 9 * 4096], vec![1, 9, 4096])
                    .unwrap();
            let k36 = GpuTensor::from_cpu_f32(
                &ctx.device,
                &vec![0.2f32; 1 * 36 * 1024],
                vec![1, 36, 1024],
            )
            .unwrap();
            let v36 = GpuTensor::from_cpu_f32(
                &ctx.device,
                &vec![0.3f32; 1 * 36 * 1024],
                vec![1, 36, 1024],
            )
            .unwrap();
            let y9 = GpuTensor::zeros(&ctx.device, vec![1, 9, 4096]).unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..10 {
                ctx.attn_ar_f32(&q9, &k36, &v36, None, &y9, 32, 8, 128)
                    .unwrap();
            }
            eprintln!(
                "micro: attn_ar 9x36x32 = {:.2}ms",
                t0.elapsed().as_secs_f64() * 1e3 / 10.0
            );
            // rmsnorm_head 9 токенов
            let w128 = GpuTensor::from_cpu_f32(&ctx.device, &vec![0.5f32; 128], vec![128]).unwrap();
            let yn = GpuTensor::zeros(&ctx.device, vec![1, 9, 4096]).unwrap();
            let t1 = std::time::Instant::now();
            for _ in 0..10 {
                ctx.rmsnorm_head_f32(&q9, &w128, &yn, 32, 1e-6).unwrap();
            }
            eprintln!(
                "micro: rmsnorm_head 9t = {:.2}ms",
                t1.elapsed().as_secs_f64() * 1e3 / 10.0
            );
            // rope 9 токенов (напрямую через run)
            let yr = GpuTensor::zeros(&ctx.device, vec![1, 9, 4096]).unwrap();
            let t2 = std::time::Instant::now();
            for _ in 0..10 {
                let grid = metal::MTLSize::new(9, 32, 64);
                let group = metal::MTLSize::new(1, 4, 4);
                ctx.run("rope_std_f32", grid, group, |enc| {
                    enc.set_buffer(0, Some(&q9.buffer), q9.offset as u64);
                    enc.set_buffer(1, Some(&yr.buffer), yr.offset as u64);
                    for (i, v) in [9usize, 4096, 128, 0usize].iter().enumerate() {
                        enc.set_bytes((2 + i) as u64, 8, (v as *const usize).cast());
                    }
                    enc.set_bytes(6, 4, (&1e6f32 as *const f32).cast());
                })
                .unwrap();
            }
            eprintln!(
                "micro: rope 9t = {:.2}ms",
                t2.elapsed().as_secs_f64() * 1e3 / 10.0
            );
        }
        // микро-бенч: gemm 16×4096×4096 Q8_0 (как prefill-верификация)
        {
            let m16 = 16usize;
            let k16 = 4096usize;
            let n16 = 4096usize;
            let b16 = GpuTensor::from_quantized(
                &ctx.device,
                &bytes_rep(n16, k16),
                vec![n16, k16],
                GpuDtype::Q8_0,
            )
            .unwrap();
            let a16 =
                GpuTensor::from_cpu_f32(&ctx.device, &vec![0.5f32; m16 * k16], vec![m16, k16])
                    .unwrap();
            let c16 = GpuTensor::zeros(&ctx.device, vec![m16, n16]).unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..3 {
                ctx.gemm_nt_quantized(&a16, &b16, &c16).unwrap();
            }
            eprintln!(
                "micro: gemm 16x4096x4096 Q8_0 = {:.1}ms",
                t0.elapsed().as_secs_f64() * 1e3 / 3.0
            );
        }
        // CPU: C[0][0] = 0 (a0·q0: 1*0=0); C[0][1] = 31; C[1][0] = Σi*i = 9455
        let exp = vec![
            0.0,
            31.0,
            10416.0,
            (0..32).map(|i| (i * (31 - i)) as f32).sum(),
        ];
        assert_eq!(cv, exp, "batch Q8_0 m=2: got {cv:?} expected {exp:?}");
        // m=1: только строка 0
        let a1 = GpuTensor::from_cpu_f32(&ctx.device, &ad[..32], vec![1, 32]).unwrap();
        let c1 = GpuTensor::zeros(&ctx.device, vec![1, 2]).unwrap();
        ctx.gemm_nt_quantized(&a1, &b, &c1).unwrap();
        let cv1 = c1.to_cpu_f32(&ctx).unwrap();
        assert_eq!(cv1, vec![0.0, 31.0], "batch Q8_0 m=1: got {cv1:?}");
        // единица на j=31 в строке 1 (лишний член?)
        let mut ad2 = vec![0f32; 2 * 32];
        ad2[32 + 31] = 1.0;
        let a2 = GpuTensor::from_cpu_f32(&ctx.device, &ad2, vec![2, 32]).unwrap();
        let c2 = GpuTensor::zeros(&ctx.device, vec![2, 2]).unwrap();
        ctx.gemm_nt_quantized(&a2, &b, &c2).unwrap();
        let cv2 = c2.to_cpu_f32(&ctx).unwrap();
        assert_eq!(cv2, vec![0.0, 0.0, 31.0, 0.0], "единица j=31: got {cv2:?}");
    }
}
