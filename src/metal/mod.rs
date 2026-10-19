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

thread_local! {
    /// Принудительный выбор пути внимания предфилла для ТЕКУЩЕГО потока
    /// (тесты): переменные окружения общие на процесс и в параллельных
    /// тестах переключали бы путь соседям.
    static ATTN_PATH: std::cell::Cell<Option<&'static str>> = const { std::cell::Cell::new(None) };
}

/// Выполнить `f`, принудительно выбрав путь внимания предфилла: `"tiles"`
/// (INFER_ATTN_TILES), `"split"` (INFER_ATTN_SPLIT) или `"legacy"`
/// (INFER_ATTN_LEGACY) — только в этом потоке.
pub fn with_attn_path<R>(path: &'static str, f: impl FnOnce() -> R) -> R {
    let prev = ATTN_PATH.with(|c| c.replace(Some(path)));
    let r = f();
    ATTN_PATH.with(|c| c.set(prev));
    r
}

/// Число simdgroup'ов split-K внимания по ключам: одна группа на ~64 ключа,
/// не больше 32 и не больше, чем помещается в threadgroup-память слияния
/// (`nsg·(2+hd)·4` байт; предел устройства — 32 КБ). При hd = 256 32 группы
/// требовали 33 КБ: на контексте > 1984 токенов (Qwen3.5) Metal отбрасывал
/// команду — выход внимания оставался мусором без ошибки.
fn split_attn_groups(tkv: usize, hd: usize) -> usize {
    const TG_MEM: usize = 32 * 1024;
    let by_mem = (TG_MEM / ((2 + hd) * 4)).max(1);
    tkv.div_ceil(64).clamp(1, 32).min(by_mem)
}

/// Включён ли вариант внимания: переопределение потока или переменная
/// окружения (`INFER_ATTN_<NAME>=1`; для legacy — сам факт наличия).
fn attn_opt(name: &'static str) -> bool {
    if let Some(p) = ATTN_PATH.with(|c| c.get()) {
        return p == name;
    }
    match name {
        "legacy" => std::env::var("INFER_ATTN_LEGACY").is_ok(),
        "split" => std::env::var("INFER_ATTN_SPLIT").as_deref() == Ok("1"),
        "tiles" => std::env::var("INFER_ATTN_TILES").as_deref() == Ok("1"),
        _ => false,
    }
}

/// Storage dtype of a [`GpuTensor`]. Activations stay fp32; fp16/bf16 are
/// GEMM weight formats (halved VRAM + weight-read traffic, fp32 accumulate
/// in the kernels — the backend dtype contract). Q4_0/Q8_0/Q4_K/Q6_K —
/// block-quantized GEMM weights (Ф1.3/1.4): dequant inside the kernel, raw
/// block bytes in the buffer (ggml layouts).
// Q4_K/Q6_K — имена форматов ggml (как в GGUF), не CamelCase.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuDtype {
    F32,
    F16,
    BF16,
    Q4_0,
    Q8_0,
    Q4_K,
    Q6_K,
    Q5_K,
    Q5_0,
    /// IQ4_XS: 256-эл. суперблок, 136 Б (кодбук kvalues_iq4nl).
    Iq4Xs,
    /// IQ4_NL: 32-эл. блок, 18 Б (f16 d + 16 Б нибблов), кодбук kvalues_iq4nl.
    Iq4Nl,
    /// MXFP4 (OCP): 32-эл. блок, 17 Б (E8M0 + 16 Б нибблов E2M1).
    Mxfp4,
    /// NVFP4: 64-эл. блок, 36 Б (4 UE4M3-скейла на подблоки 16 + 32 Б нибблов).
    Nvfp4,
    /// Q2_K: 256-эл. суперблок, 84 Б (scales[16], qs[64], d, dmin).
    Q2_K,
    /// Q3_K: 256-эл. суперблок, 110 Б (hmask[32], qs[64], scales[12], d).
    Q3_K,
    /// Q4_1: 32-эл. блок, 20 Б (d, m, нибблы), значение d·q + m.
    Q4_1,
    /// Q5_1: 32-эл. блок, 24 Б (d, m, qh[4], нибблы).
    Q5_1,
    /// Q8_1: 32-эл. блок, 40 Б (d, s, int8[32]).
    Q8_1,
    /// Q8_K: 256-эл. суперблок, 292 Б (f32 d, int8[256], bsums[16]).
    Q8_K,
    /// Q1_0: 128-эл. блок, 18 Б (d + биты), значения ±d.
    Q1_0,
    /// Q2_0: 64-эл. блок, 18 Б (d + 2-битные), значения (q−1)·d.
    Q2_0,
    /// TQ1_0: 256-эл. суперблок, 54 Б (qs[48] по 3^n, qh[4], d).
    TQ1_0,
    /// TQ2_0: 256-эл. суперблок, 66 Б (qs[64] по 2 бита, d).
    TQ2_0,
    /// IQ2_XXS: 256-эл. суперблок, 66 Б (сетка iq2xxs_grid + ksigns).
    Iq2Xxs,
    /// IQ3_XXS: 256-эл. суперблок, 98 Б (сетка iq3xxs_grid + знаки).
    Iq3Xxs,
    /// IQ2_XS: 74 Б (сетка iq2xs_grid), IQ2_S: 82 Б (iq2s_grid),
    /// IQ3_S: 110 Б (iq3s_grid + signs), IQ1_S: 50 Б (iq1s_grid),
    /// IQ1_M: 56 Б (iq1s_grid, 3-битные scale).
    Iq2Xs,
    Iq2S,
    Iq3S,
    Iq1S,
    Iq1M,
}

impl GpuDtype {
    /// Элементов в квант-блоке (для проверки кратности K).
    pub fn block_elems(self) -> Option<usize> {
        match self {
            GpuDtype::Q4_0 | GpuDtype::Q8_0 | GpuDtype::Iq4Nl | GpuDtype::Mxfp4 => Some(32),
            GpuDtype::Nvfp4 => Some(64),
            GpuDtype::Q4_1 | GpuDtype::Q5_1 | GpuDtype::Q8_1 => Some(32),
            GpuDtype::Q8_K
            | GpuDtype::TQ1_0
            | GpuDtype::TQ2_0
            | GpuDtype::Iq2Xxs
            | GpuDtype::Iq3Xxs
            | GpuDtype::Iq2Xs
            | GpuDtype::Iq2S
            | GpuDtype::Iq3S
            | GpuDtype::Iq1S
            | GpuDtype::Iq1M => Some(256),
            GpuDtype::Q1_0 => Some(128),
            GpuDtype::Q2_0 => Some(64),
            GpuDtype::Q4_K
            | GpuDtype::Q6_K
            | GpuDtype::Q5_K
            | GpuDtype::Iq4Xs
            | GpuDtype::Q2_K
            | GpuDtype::Q3_K => Some(256),
            GpuDtype::Q5_0 => Some(32),
            _ => None,
        }
    }

    /// Байт в квант-блоке (раскладки ggml-common.h: Q4_0 18, Q5_0 22,
    /// Q8_0 34, Q4_K 144, Q5_K 176, Q6_K 210, IQ4_XS 136). Q4_K — именно 144
    /// (2·half + scales[12] + qs[128]); 148 был кастомным диалектом до
    /// волны ревью 8.
    pub fn block_bytes(self) -> Option<usize> {
        match self {
            GpuDtype::Q4_0 => Some(18),
            GpuDtype::Q5_0 => Some(22),
            GpuDtype::Q8_0 => Some(34),
            GpuDtype::Q4_K => Some(144),
            GpuDtype::Q5_K => Some(176),
            GpuDtype::Q6_K => Some(210),
            GpuDtype::Iq4Xs => Some(136),
            GpuDtype::Iq4Nl => Some(18),
            GpuDtype::Mxfp4 => Some(17),
            GpuDtype::Nvfp4 => Some(36),
            GpuDtype::Q2_K => Some(84),
            GpuDtype::Q3_K => Some(110),
            GpuDtype::Q4_1 => Some(20),
            GpuDtype::Q5_1 => Some(24),
            GpuDtype::Q8_1 => Some(36),
            GpuDtype::Q8_K => Some(292),
            GpuDtype::Q1_0 | GpuDtype::Q2_0 => Some(18),
            GpuDtype::TQ1_0 => Some(54),
            GpuDtype::TQ2_0 => Some(66),
            GpuDtype::Iq2Xxs => Some(66),
            GpuDtype::Iq3Xxs => Some(98),
            GpuDtype::Iq2Xs => Some(74),
            GpuDtype::Iq2S => Some(82),
            GpuDtype::Iq3S => Some(110),
            GpuDtype::Iq1S => Some(50),
            GpuDtype::Iq1M => Some(56),
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
        match std::env::var("INFER_METAL_CB_CHUNK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            Some(0) => usize::MAX,
            Some(n) => n,
            // Замер (M3 Max, pp128/tg64): 8–32 — плато, 0 (без порций) —
            // prefill на треть медленнее; 16 — середина плато.
            None => 16,
        }
    })
}

/// Диагностика хостового пути (Ф8.2g): INFER_METAL_OPS=1 включает счёт
/// dispatch'ей и суммарное время, проведённое в [`MetalCtx::run`] на хосте
/// (кодирование энкодера, без GPU-ожидания). Выключено по умолчанию:
/// два атомика на dispatch — заметная доля стоимости мелкого оверхеда,
/// который эти счётчики и измеряют.
fn ops_prof() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("INFER_METAL_OPS").is_ok())
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
            self.free_bytes
                .set(self.free_bytes.get().saturating_sub(class));
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
            DEFERRED_BYTES.with(|b| b.set(b.get() + buf.length() as usize));
            // Давление: один длинный префилл освобождает сотни мегабайт за
            // слой, а end_batch будет только в конце — без сброса машина
            // уходит в своп (watchdog-паника 2026-10-12). Commit+wait +
            // возврат в пул прямо сейчас.
            if DEFERRED_BYTES.with(|b| b.get()) > deferred_cap_bytes() {
                drain_deferred_now();
            }
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

    /// Байты в DEFERRED (для сброса по давлению — см. BufferPoolState::push).
    static DEFERRED_BYTES: Cell<usize> = const { Cell::new(0) };

    /// Batch-состояние текущего потока (ставится в begin_batch): нужно,
    /// чтобы push() мог сделать commit+wait и вернуть DEFERRED в пул, не
    /// дожидаясь end_batch — иначе один длинный префилл копит гигабайты
    /// живых command-buffer'ом занятых буферов и уводит машину в своп
    /// (watchdog-паника 2026-10-12 на префилле 4k).
    static BATCH_BS: RefCell<Option<Arc<Mutex<BatchState>>>> = const { RefCell::new(None) };
}

/// Предел DEFERRED до принудительного commit+wait-слияния (МБ).
fn deferred_cap_bytes() -> usize {
    static CAP: OnceLock<usize> = OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("INFER_METAL_DEFERRED_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1536)
            * 1024
            * 1024
    })
}

/// commit+wait открытого батча и возврат DEFERRED в пул. Вызывается из
/// push() по давлению и из end_batch. try_lock: если батч держит другой
/// фрейм — пропустить (попробуем на следующем push).
fn drain_deferred_now() {
    let bs = BATCH_BS.with(|c| c.borrow().clone());
    let Some(bs) = bs else { return };
    let Ok(mut batch) = bs.try_lock() else { return };
    if batch.depth == 0 {
        return;
    }
    batch.commit_open();
    batch.wait_last();
    drop(batch);
    let deferred = DEFERRED.with(|d| std::mem::take(&mut *d.borrow_mut()));
    DEFERRED_BYTES.with(|b| b.set(0));
    with_pool(|p| {
        let mut free = p.free.lock().unwrap();
        for buf in deferred {
            let class = (buf.length() as usize + 4095) & !4095;
            let class = class.max(4096);
            p.put_free(&mut free, class, buf);
        }
    });
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
        if let Some(tok) = self.pooled.as_ref()
            && Arc::strong_count(tok) == 1
        {
            with_pool(|p| p.push(&self.buffer));
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
            GpuDtype::Q4_K
            | GpuDtype::Q6_K
            | GpuDtype::Q5_K
            | GpuDtype::Q5_0
            | GpuDtype::Iq4Xs
            | GpuDtype::Iq4Nl
            | GpuDtype::Mxfp4
            | GpuDtype::Nvfp4
            | GpuDtype::Q2_K
            | GpuDtype::Q3_K
            | GpuDtype::Q4_1
            | GpuDtype::Q5_1
            | GpuDtype::Q8_1
            | GpuDtype::Q8_K
            | GpuDtype::Q1_0
            | GpuDtype::Q2_0
            | GpuDtype::TQ1_0
            | GpuDtype::TQ2_0
            | GpuDtype::Iq2Xxs
            | GpuDtype::Iq3Xxs
            | GpuDtype::Iq2Xs
            | GpuDtype::Iq2S
            | GpuDtype::Iq3S
            | GpuDtype::Iq1S
            | GpuDtype::Iq1M => 1,
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

    /// Zero-filled f16 buffer (f16-KV/активации, Ф8.2f).
    /// Zero-filled f32 tensor (маленькие буферы: q во внимании с f16-KV).
    pub fn zeros_f32(device: &Device, shape: Vec<usize>) -> Result<Self, MetalError> {
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

    pub fn zeros_f16(device: &Device, shape: Vec<usize>) -> Result<Self, MetalError> {
        let bytes = shape.iter().product::<usize>() * 2;
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
            dtype: GpuDtype::F16,
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
            GpuDtype::Q5_K => 176.0 / 256.0,
            GpuDtype::Q5_0 => 22.0 / 32.0,
            GpuDtype::Iq4Xs => 136.0 / 256.0,
            GpuDtype::Iq4Nl => 18.0 / 32.0,
            GpuDtype::Mxfp4 => 17.0 / 32.0,
            GpuDtype::Nvfp4 => 36.0 / 64.0,
            GpuDtype::Q2_K => 84.0 / 256.0,
            GpuDtype::Q3_K => 110.0 / 256.0,
            GpuDtype::Q4_1 => 20.0 / 32.0,
            GpuDtype::Q5_1 => 24.0 / 32.0,
            GpuDtype::Q8_1 => 36.0 / 32.0,
            GpuDtype::Q8_K => 292.0 / 256.0,
            GpuDtype::Q1_0 => 18.0 / 128.0,
            GpuDtype::Q2_0 => 18.0 / 64.0,
            GpuDtype::TQ1_0 => 54.0 / 256.0,
            GpuDtype::TQ2_0 => 66.0 / 256.0,
            GpuDtype::Iq2Xxs => 66.0 / 256.0,
            GpuDtype::Iq3Xxs => 98.0 / 256.0,
            GpuDtype::Iq2Xs => 74.0 / 256.0,
            GpuDtype::Iq2S => 82.0 / 256.0,
            GpuDtype::Iq3S => 110.0 / 256.0,
            GpuDtype::Iq1S => 50.0 / 256.0,
            GpuDtype::Iq1M => 56.0 / 256.0,
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
            return Err(MetalError::Shape("to_cpu_f16 on a non-f16 tensor".into()));
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
    batch: Arc<Mutex<BatchState>>,
    /// Диагностика хостового оверхеда (INFER_METAL_OPS): dispatch'ей и
    /// наносекунд хоста в `run`. Атомики, чтобы читаться из примеров/тестов
    /// без &mut; на нулевые счётчики никто не смотрит, когда профиль выключен.
    ops_count: std::sync::atomic::AtomicU64,
    ops_host_ns: std::sync::atomic::AtomicU64,
    /// Порог малого-m для батчевых путей квант-GEMM (0 — брать из
    /// INFER_QUANT_MM_MIN, дефолт 4). Инстанс-переопределение для тестов:
    /// env-рычаг глобальный, и тесты с разными порогами в одном процессе
    /// перетирали друг друга (флейк iq4_xs_matvec_batch_consistency).
    quant_mm_min: std::sync::atomic::AtomicUsize,
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
            batch: Arc::new(Mutex::new(BatchState::default())),
            ops_count: std::sync::atomic::AtomicU64::new(0),
            ops_host_ns: std::sync::atomic::AtomicU64::new(0),
            quant_mm_min: std::sync::atomic::AtomicUsize::new(0),
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
            "copy_rows_f32",
            "argmax_lse_rows_f32",
            "attn_prefill_mma_f16_nf16",
            "attn_prefill_mma_f16_nf32",
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
        // Исходник лежит в kernels/ тематическими файлами (5700+ строк в
        // одном файле стали god object); конкатенация эквивалентна одному
        // исходнику: общий заголовок (include/using/хелперы) идёт первым,
        // кернелы между собой не перекрёстно-зависимы.
        let library = self.library.get_or_init(|| {
            const SOURCE: &str = concat!(
                include_str!("kernels/common.metal"),
                include_str!("kernels/gemm.metal"),
                include_str!("kernels/mul_mv.metal"),
                include_str!("kernels/attention.metal"),
                include_str!("kernels/gdn.metal"),
                include_str!("kernels/rope.metal"),
                include_str!("kernels/norm.metal"),
                include_str!("kernels/conv.metal"),
                include_str!("kernels/embed.metal"),
                include_str!("kernels/moe.metal"),
                include_str!("kernels/elementwise.metal"),
            );
            let options = CompileOptions::new();
            self.device
                .new_library_with_source(SOURCE, &options)
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
    ///
    /// INFER_METAL_OPS=1 (см. [`ops_prof`]) добавляет счётчик dispatch'ей и
    /// суммарное хостовое время — сколько окно/шаг стоит ДО GPU.
    fn run(
        &self,
        kernel: &'static str,
        grid: MTLSize,
        group: MTLSize,
        setup: impl FnOnce(&ComputeCommandEncoderRef),
    ) -> Result<(), MetalError> {
        if !ops_prof() {
            return self.run_inner(kernel, grid, group, setup);
        }
        let t = std::time::Instant::now();
        let r = self.run_inner(kernel, grid, group, setup);
        self.ops_host_ns.fetch_add(
            t.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        self.ops_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        r
    }

    /// (число dispatch'ей, наносекунды хоста в `run`) — и сброс, если `reset`.
    /// Счётчики живут только под INFER_METAL_OPS=1.
    pub fn ops_snapshot(&self, reset: bool) -> (u64, u64) {
        use std::sync::atomic::Ordering::Relaxed;
        if reset {
            (
                self.ops_count.swap(0, Relaxed),
                self.ops_host_ns.swap(0, Relaxed),
            )
        } else {
            (self.ops_count.load(Relaxed), self.ops_host_ns.load(Relaxed))
        }
    }

    /// Переопределить порог малого-m батчевых путей на этом инстансе
    /// (тесты; 0 — вернуть env-поведение). См. поле `quant_mm_min`.
    pub fn set_quant_mm_min(&self, v: usize) {
        self.quant_mm_min
            .store(v, std::sync::atomic::Ordering::Relaxed);
    }

    /// Порог малого-m: инстанс-переопределение, иначе INFER_QUANT_MM_MIN,
    /// иначе 4. Env читается на каждом вызове (рычаг сверки живой).
    pub fn quant_mm_min(&self) -> usize {
        let v = self.quant_mm_min.load(std::sync::atomic::Ordering::Relaxed);
        if v > 0 {
            return v;
        }
        std::env::var("INFER_QUANT_MM_MIN")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(4)
    }

    fn run_inner(
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
        drop(batch);
        BATCH_DEPTH.with(|d| d.set(d.get() + 1));
        BATCH_BS.with(|c| *c.borrow_mut() = Some(self.batch.clone()));
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
                DEFERRED_BYTES.with(|b| b.set(0));
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
            GpuDtype::Q4_0
            | GpuDtype::Q8_0
            | GpuDtype::Q4_K
            | GpuDtype::Q6_K
            | GpuDtype::Q5_K
            | GpuDtype::Q5_0
            | GpuDtype::Iq4Xs
            | GpuDtype::Iq4Nl
            | GpuDtype::Mxfp4
            | GpuDtype::Nvfp4
            | GpuDtype::Q2_K
            | GpuDtype::Q3_K
            | GpuDtype::Q4_1
            | GpuDtype::Q5_1
            | GpuDtype::Q8_1
            | GpuDtype::Q8_K
            | GpuDtype::Q1_0
            | GpuDtype::Q2_0
            | GpuDtype::TQ1_0
            | GpuDtype::TQ2_0
            | GpuDtype::Iq2Xxs
            | GpuDtype::Iq3Xxs
            | GpuDtype::Iq2Xs
            | GpuDtype::Iq2S
            | GpuDtype::Iq3S
            | GpuDtype::Iq1S
            | GpuDtype::Iq1M => {
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

    pub fn swiglu_f32(
        &self,
        x: &GpuTensor,
        u: &GpuTensor,
        y: &GpuTensor,
    ) -> Result<(), MetalError> {
        let n = x.numel();
        let grid = MTLSize::new(n as u64, 1, 1);
        let group = MTLSize::new(256, 1, 1);
        self.run("swiglu_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
            enc.set_buffer(1, Some(&u.buffer), u.offset as u64);
            enc.set_buffer(2, Some(&y.buffer), y.offset as u64);
            enc.set_bytes(3, 8, (&n as *const usize).cast());
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
        // Эталон: F32/F16-веса при m∈[2,8] идут mul_mv_ext (не поэлементным
        // GEMM и не classic mv) — воспроизводим выбор; фоллбэк INFER_NO_MV_EXT=1.
        if matches!(b.dtype, GpuDtype::F32 | GpuDtype::F16)
            && (2..=8).contains(&m)
            && std::env::var("INFER_NO_MV_EXT").is_err()
            && self.mv_ext(a, b, c)?
        {
            return Ok(());
        }
        // f16-веса при m > 1 (префилл, LoRA-merged f16-модели): m8-кернел —
        // столбец весов читается один раз на 8 строк активаций (half4), в
        // 1.7× быстрее поэлементного gemm_nt_f16 при m ≥ 128 (замер
        // 4096×4096: 3.4 против 5.9 мс). Требует K, кратного 4.
        if b.dtype == GpuDtype::F16 && m > 1 && k.is_multiple_of(4) {
            let grid = MTLSize::new(n as u64, m.div_ceil(8) as u64, 1);
            let group = MTLSize::new(256, 1, 1);
            return self.run("gemm_nt_f16_m8", grid, group, |enc| {
                enc.set_buffer(0, Some(&a.buffer), a.offset as u64);
                enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
                enc.set_buffer(2, Some(&c.buffer), c.offset as u64);
                enc.set_bytes(3, 8, (&m as *const usize).cast());
                enc.set_bytes(4, 8, (&k as *const usize).cast());
                enc.set_bytes(5, 8, (&n as *const usize).cast());
            });
        }
        // m==1 F32 — simdgroup-матвек: v4 даёт по одному потоку на выход,
        // при n=96 (GDN beta/alpha) это 96 потоков на всю GPU — 16 ГБ/с.
        // Фоллбэк INFER_F32_NO_MV=1.
        if b.dtype == GpuDtype::F32
            && m == 1
            && k.is_multiple_of(4)
            && a.offset.is_multiple_of(16)
            && b.offset.is_multiple_of(16)
            && std::env::var("INFER_F32_NO_MV").is_err()
        {
            return self.run(
                "gemm_nt_f32_mv",
                MTLSize::new((n * 32) as u64, 1, 1),
                MTLSize::new(32, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&a.buffer), a.offset as u64);
                    enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
                    enc.set_buffer(2, Some(&c.buffer), c.offset as u64);
                    enc.set_bytes(3, 8, (&m as *const usize).cast());
                    enc.set_bytes(4, 8, (&k as *const usize).cast());
                    enc.set_bytes(5, 8, (&n as *const usize).cast());
                },
            );
        }
        // Эталонный выбор (ggml_metal_op_mul_mat_use_mm): f32-веса при
        // ne00>=64 && ne11>8 идут kernel_mul_mm_f32_f32 (MMA-порядок). Без
        // этого роутер OLMoE (gate_inp f32, m=префилл) расходится с оркалом
        // на ~1e-4, веса слотов — и весь длинный префилл (Ф8.1h).
        if b.dtype == GpuDtype::F32
            && k >= 64
            && m > 8
            && k.is_multiple_of(32)
            && std::env::var("INFER_NO_F32_MM").is_err()
        {
            let grid = MTLSize::new((m.div_ceil(32) * 128) as u64, n.div_ceil(64) as u64, 1);
            let group = MTLSize::new(128, 1, 1);
            return self.run("gemm_nt_f32_mm", grid, group, |enc| {
                enc.set_buffer(0, Some(&a.buffer), a.offset as u64);
                enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
                enc.set_buffer(2, Some(&c.buffer), c.offset as u64);
                enc.set_bytes(3, 8, (&m as *const usize).cast());
                enc.set_bytes(4, 8, (&k as *const usize).cast());
                enc.set_bytes(5, 8, (&n as *const usize).cast());
                enc.set_threadgroup_memory_length(0, 8192);
            });
        }
        let name = match b.dtype {
            // F32 при 16-байтном выравнивании и K%4==0 — v4-вариант:
            // float4-загрузки при том же порядке накопления (побитово равен
            // скалярному), ~4× меньше латентности загрузок на поток.
            GpuDtype::F32 => {
                if k.is_multiple_of(4) && a.offset.is_multiple_of(16) && b.offset.is_multiple_of(16) {
                    "gemm_nt_f32_v4"
                } else {
                    "gemm_nt_f32"
                }
            }
            GpuDtype::F16 => "gemm_nt_f16",
            GpuDtype::BF16 => "gemm_nt_bf16",
            // Квантованные веса — только через gemm_nt_quantized/mv_quantized.
            GpuDtype::Q4_0
            | GpuDtype::Q8_0
            | GpuDtype::Q4_K
            | GpuDtype::Q6_K
            | GpuDtype::Q5_K
            | GpuDtype::Q5_0
            | GpuDtype::Iq4Xs
            | GpuDtype::Iq4Nl
            | GpuDtype::Mxfp4
            | GpuDtype::Nvfp4
            | GpuDtype::Q2_K
            | GpuDtype::Q3_K
            | GpuDtype::Q4_1
            | GpuDtype::Q5_1
            | GpuDtype::Q8_1
            | GpuDtype::Q8_K
            | GpuDtype::Q1_0
            | GpuDtype::Q2_0
            | GpuDtype::TQ1_0
            | GpuDtype::TQ2_0
            | GpuDtype::Iq2Xxs
            | GpuDtype::Iq3Xxs
            | GpuDtype::Iq2Xs
            | GpuDtype::Iq2S
            | GpuDtype::Iq3S
            | GpuDtype::Iq1S
            | GpuDtype::Iq1M => {
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
            // Q4_0: (а) окно верификации m 2..8 — батчевый matvec (прямые
            // вызовы ctx; трейт ловит этот диапазон раньше, поэтому здесь
            // дубль нужен только тестам/прямым вызовам); (б) m>8 — тайловой
            // sgemm (порт схемы ggml kernel_mul_mm_q4_0: m>8 в эталоне идёт
            // по mul_mm). До этого обе ветки отсутствовали: окна читали веса
            // построчно, префилл шёл скалярным gemm_nt_q4_0 (каждый поток —
            // вся строка весов; 8B-матмулы M=16: 442 против 85 мс у Q8_0).
            GpuDtype::Q4_0 => {
                let mm_min = self.quant_mm_min();
                if (2..=8).contains(&m)
                    && mm_min <= 8
                    && k.is_multiple_of(32)
                    && a.offset.is_multiple_of(16)
                    && std::env::var("INFER_Q4_0_NO_MB").is_err()
                {
                    self.mv_q4_0_mb(a, b, c)?;
                    return Ok(());
                }
                if m > 8 && k.is_multiple_of(32) && std::env::var("INFER_Q4_0_NO_MM").is_err() {
                    let (m32, k32, n32) = (m as u32, k as u32, n as u32);
                    // Тайл 32 (M) × 64 (N), threadgroup 128 потоков —
                    // схема kernel_mul_mm (как у Q8_0/Q6_K).
                    let grid =
                        MTLSize::new((m.div_ceil(32) * 128) as u64, n.div_ceil(64) as u64, 1);
                    let group = MTLSize::new(128, 1, 1);
                    return self.run("gemm_nt_q4_0_mm", grid, group, |enc| {
                        enc.set_buffer(0, Some(&a.buffer), a.offset as u64);
                        enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
                        enc.set_buffer(2, Some(&c.buffer), c.offset as u64);
                        enc.set_bytes(3, 4, (&m32 as *const u32).cast());
                        enc.set_bytes(4, 4, (&k32 as *const u32).cast());
                        enc.set_bytes(5, 4, (&n32 as *const u32).cast());
                        // 8 КБ: тайлы весов/активаций (4+2 КБ) и staging
                        // хвостового тайла выхода (32×64 float).
                        enc.set_threadgroup_memory_length(0, 8192);
                    });
                }
                "gemm_nt_q4_0"
            }
            GpuDtype::Q8_0 => "gemm_nt_q8_0",
            // Q4_K при m > 8 — тайловой sgemm: ДОСЛОВНЫЙ порт
            // kernel_mul_mm_q4_K_f32 (dequantize_q4_K + get_scale_min_k4_just2,
            // gemm.metal). Паритет: при m > 8 эталон считает q/k/v и FFN
            // ЭТИМ кернелом (use_mm), и его порядок суммирования отличается
            // и от matvec'а, и от поэлементного GEMM (OLMoE p2).
            // Долг «НЕ побитово с эталонным mm» ЗАКРЫТ: загрузчик весов
            // сверен с эталонной формулой ПОБИТОВО на 39424 half-весах
            // (тест q4_k_mm_weights_bitwise_match_oracle_dequant — единичные
            // активации извлекают каждый вес из mm). Причина былого
            // расхождения: в эталонной формуле нечётных подблоков d делится
            // на 16, ниббл маскируется 0xF0 — порядок half-округления другой,
            // чем у «d·sc·q_hi», поэтому рукописный порт и не совпадал.
            // INFER_Q4K_NO_MM=1 — фоллбэк на скалярный gemm_nt_q4_k.
            // Порог m>8 и K%256==0.
            GpuDtype::Q4_K => {
                let mm_min = std::env::var("INFER_QUANT_MM_MIN")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(2);
                if m >= mm_min && m > 1 && k.is_multiple_of(256) && std::env::var("INFER_Q4K_NO_MM").is_err()
                {
                    let (m32, k32, n32) = (m as u32, k as u32, n as u32);
                    let grid =
                        MTLSize::new((m.div_ceil(32) * 128) as u64, n.div_ceil(64) as u64, 1);
                    let group = MTLSize::new(128, 1, 1);
                    return self.run("gemm_nt_q4_k_mm", grid, group, |enc| {
                        enc.set_buffer(0, Some(&a.buffer), a.offset as u64);
                        enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
                        enc.set_buffer(2, Some(&c.buffer), c.offset as u64);
                        enc.set_bytes(3, 4, (&m32 as *const u32).cast());
                        enc.set_bytes(4, 4, (&k32 as *const u32).cast());
                        enc.set_bytes(5, 4, (&n32 as *const u32).cast());
                        enc.set_threadgroup_memory_length(0, 8192);
                    });
                }
                "gemm_nt_q4_k"
            }
            // Q6_K при m > 8 — тайловой sgemm (порт kernel_mul_mm_q6_K_f32,
            // каркас как у Q5_K mm): скалярный gemm_nt_q6_k перечитывает веса
            // на каждую строку активаций (lm_head 27B m=128 — 423 мс в
            // бенче). Срабатывает на верификации DFlash (блок 16) и при
            // draft_tokens > 7; префилл — last_only (m=1, matvec). Порог
            // m>8 и K%256==0 — use_mm llama.cpp. Фоллбэк INFER_Q6K_NO_MM.
            GpuDtype::Q6_K => {
                let mm_min = std::env::var("INFER_QUANT_MM_MIN")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(2);
                if m >= mm_min && m > 1 && k.is_multiple_of(256) && std::env::var("INFER_Q6K_NO_MM").is_err()
                {
                    let (m32, k32, n32) = (m as u32, k as u32, n as u32);
                    let grid =
                        MTLSize::new((m.div_ceil(32) * 128) as u64, n.div_ceil(64) as u64, 1);
                    let group = MTLSize::new(128, 1, 1);
                    return self.run("gemm_nt_q6_k_mm", grid, group, |enc| {
                        enc.set_buffer(0, Some(&a.buffer), a.offset as u64);
                        enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
                        enc.set_buffer(2, Some(&c.buffer), c.offset as u64);
                        enc.set_bytes(3, 4, (&m32 as *const u32).cast());
                        enc.set_bytes(4, 4, (&k32 as *const u32).cast());
                        enc.set_bytes(5, 4, (&n32 as *const u32).cast());
                        enc.set_threadgroup_memory_length(0, 8192);
                    });
                }
                "gemm_nt_q6_k"
            }
            // Q5_K при m > 8 — тайловый sgemm (порт kernel_mul_mm_q5_K_f32):
            // скалярный gemm_nt_q5_k на префилле 27B-гибрида давал 0.97
            // TFLOPS против 7 у mm-пути IQ4_XS (gdn.qkv/w_out — 22 из 23.6
            // мс GDN-слоя). Порог m>8 и требование K%256==0 — как у IQ4_XS
            // (порог use_mm llama.cpp). Фоллбэк INFER_Q5K_NO_MM.
            GpuDtype::Q5_K => {
                let mm_min = std::env::var("INFER_QUANT_MM_MIN")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(2);
                if m >= mm_min && m > 1 && k.is_multiple_of(256) && std::env::var("INFER_Q5K_NO_MM").is_err()
                {
                    let (m32, k32, n32) = (m as u32, k as u32, n as u32);
                    let grid =
                        MTLSize::new((m.div_ceil(32) * 128) as u64, n.div_ceil(64) as u64, 1);
                    let group = MTLSize::new(128, 1, 1);
                    return self.run("gemm_nt_q5_k_mm", grid, group, |enc| {
                        enc.set_buffer(0, Some(&a.buffer), a.offset as u64);
                        enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
                        enc.set_buffer(2, Some(&c.buffer), c.offset as u64);
                        enc.set_bytes(3, 4, (&m32 as *const u32).cast());
                        enc.set_bytes(4, 4, (&k32 as *const u32).cast());
                        enc.set_bytes(5, 4, (&n32 as *const u32).cast());
                        enc.set_threadgroup_memory_length(0, 8192);
                    });
                }
                "gemm_nt_q5_k"
            }
            // Q5_0 при m > 8 — тайловой sgemm (Ф8.1e): оркальный use_mm —
            // ne00>=64 && ne11>8 БЕЗ K%256; ffn_down GLM-4-9B (K=13696) —
            // Q5_0, скалярный gemm_nt_q5_0 расходился на ~1e-3 (ffn_out-5).
            // Порог K%32 — шаг mul_mm-тайла; фоллбэк INFER_Q50_NO_MM=1.
            GpuDtype::Q5_0 => {
                let mm_min = std::env::var("INFER_QUANT_MM_MIN")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(2);
                if m >= mm_min
                    && m > 1
                    && k.is_multiple_of(32)
                    && std::env::var("INFER_Q50_NO_MM").is_err()
                {
                    let (m32, k32, n32) = (m as u32, k as u32, n as u32);
                    let grid =
                        MTLSize::new((m.div_ceil(32) * 128) as u64, n.div_ceil(64) as u64, 1);
                    let group = MTLSize::new(128, 1, 1);
                    return self.run("gemm_nt_q5_0_mm", grid, group, |enc| {
                        enc.set_buffer(0, Some(&a.buffer), a.offset as u64);
                        enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
                        enc.set_buffer(2, Some(&c.buffer), c.offset as u64);
                        enc.set_bytes(3, 4, (&m32 as *const u32).cast());
                        enc.set_bytes(4, 4, (&k32 as *const u32).cast());
                        enc.set_bytes(5, 4, (&n32 as *const u32).cast());
                        enc.set_threadgroup_memory_length(0, 8192);
                    });
                }
                "gemm_nt_q5_0"
            }
            // IQ4_XS: при m > 8 эталон идёт по kernel_mul_mm (порог use_mm:
            // ne11 > 8), при m ≤ 8 — по mul_mv; mul_mm требует K%256.
            // Замеренная матрица ОКОН (8B-матмулы, мс, чередующиеся пары):
            // mb 28/42/61/86 против mm 87-92 на m=2..5; на m=6..8 наоборот —
            // mm 87-92 против mb 122/102/122. Прежний порядок (mm первым при
            // дефолтном пороге 2) гнал окна γ=1..4 (m=2..5) через тайл 32
            // строк: m=2 стоил 87 мс против 28 у mb. Теперь mb для 2..5,
            // mm для 6..8 и m>8. mb побитово равен mv (тест
            // iq4_xs_matvec_batch_consistency) — порядок ещё и паритет-
            // положительный; mm при m 2..8 НЕ побитово mv, паритет
            // возвращается INFER_IQ4XS_MM_MIN/INFER_QUANT_MM_MIN=9 (m 6..8
            // тогда уходят в mb — битово mv).
            // IQ4_NL / MXFP4: окна m 2..8 — батчевый matvec (веса 1× на m
            // строк), m>8 — тайловой sgemm (порт схемы mul_mm, загрузчик
            // весов с кодбуком/E8M0). Блок 32 элемента, K%32.
            // Q1_0 / Q2_0 / TQ1_0 / TQ2_0: окна m 2..8 — точечный mb,
            // m>8 — mm-тайл (блоки 128/64/256 кратны 32).
            GpuDtype::Q1_0
            | GpuDtype::Q2_0
            | GpuDtype::TQ1_0
            | GpuDtype::TQ2_0
            | GpuDtype::Iq2Xxs
            | GpuDtype::Iq3Xxs
            | GpuDtype::Iq2Xs
            | GpuDtype::Iq2S
            | GpuDtype::Iq3S
            | GpuDtype::Iq1S
            | GpuDtype::Iq1M => {
                let mb_off = match b.dtype {
                    GpuDtype::Q1_0 => std::env::var("INFER_Q1_0_NO_MB").is_ok(),
                    GpuDtype::Q2_0 => std::env::var("INFER_Q2_0_NO_MB").is_ok(),
                    GpuDtype::TQ1_0 => std::env::var("INFER_TQ1_0_NO_MB").is_ok(),
                    GpuDtype::TQ2_0 => std::env::var("INFER_TQ2_0_NO_MB").is_ok(),
                    GpuDtype::Iq2Xxs => std::env::var("INFER_IQ2XXS_NO_MB").is_ok(),
                    GpuDtype::Iq3Xxs => std::env::var("INFER_IQ3XXS_NO_MB").is_ok(),
                    GpuDtype::Iq2Xs => std::env::var("INFER_IQ2XS_NO_MB").is_ok(),
                    GpuDtype::Iq2S => std::env::var("INFER_IQ2S_NO_MB").is_ok(),
                    GpuDtype::Iq3S => std::env::var("INFER_IQ3S_NO_MB").is_ok(),
                    GpuDtype::Iq1S => std::env::var("INFER_IQ1S_NO_MB").is_ok(),
                    _ => std::env::var("INFER_IQ1M_NO_MB").is_ok(),
                };
                if m <= 8 && !mb_off {
                    if m == 1 {
                        let av = a.view_offset(vec![k], 0);
                        let cv = c.view_offset(vec![n], 0);
                        return self.mv_quantized(&av, b, &cv);
                    }
                    return self.mv_tern_mb(a, b, c);
                }
                let blk = match b.dtype {
                    GpuDtype::Q1_0 => 128usize,
                    GpuDtype::Q2_0 => 64,
                    _ => 256,
                };
                if m > 8 && k.is_multiple_of(blk) {
                    let name = match b.dtype {
                        GpuDtype::Q1_0 => "gemm_nt_q1_0_mm",
                        GpuDtype::Q2_0 => "gemm_nt_q2_0_mm",
                        GpuDtype::TQ1_0 => "gemm_nt_tq1_0_mm",
                        GpuDtype::TQ2_0 => "gemm_nt_tq2_0_mm",
                        GpuDtype::Iq2Xxs => "gemm_nt_iq2_xxs_mm",
                        GpuDtype::Iq3Xxs => "gemm_nt_iq3_xxs_mm",
                        GpuDtype::Iq2Xs => "gemm_nt_iq2_xs_mm",
                        GpuDtype::Iq2S => "gemm_nt_iq2_s_mm",
                        GpuDtype::Iq3S => "gemm_nt_iq3_s_mm",
                        GpuDtype::Iq1S => "gemm_nt_iq1_s_mm",
                        _ => "gemm_nt_iq1_m_mm",
                    };
                    let (m32, k32, n32) = (m as u32, k as u32, n as u32);
                    let grid =
                        MTLSize::new((m.div_ceil(32) * 128) as u64, n.div_ceil(64) as u64, 1);
                    let group = MTLSize::new(128, 1, 1);
                    return self.run(name, grid, group, |enc| {
                        enc.set_buffer(0, Some(&a.buffer), a.offset as u64);
                        enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
                        enc.set_buffer(2, Some(&c.buffer), c.offset as u64);
                        enc.set_bytes(3, 4, (&m32 as *const u32).cast());
                        enc.set_bytes(4, 4, (&k32 as *const u32).cast());
                        enc.set_bytes(5, 4, (&n32 as *const u32).cast());
                        enc.set_threadgroup_memory_length(0, 8192);
                    });
                }
                let av = a.view_offset(vec![k], 0);
                let cv = c.view_offset(vec![n], 0);
                return self.mv_quantized(&av, b, &cv);
            }
            // Q4_1 / Q5_1 / Q8_1 / Q8_K: окна m 2..8 — батчевый matvec,
            // m>8 — mm-тайл. Q4_1/Q5_1/Q8_1 — блок 32, Q8_K — 256.
            GpuDtype::Q4_1 | GpuDtype::Q5_1 | GpuDtype::Q8_1 | GpuDtype::Q8_K => {
                let mb_off = match b.dtype {
                    GpuDtype::Q4_1 => std::env::var("INFER_Q4_1_NO_MB").is_ok(),
                    GpuDtype::Q5_1 => std::env::var("INFER_Q5_1_NO_MB").is_ok(),
                    GpuDtype::Q8_1 => std::env::var("INFER_Q8_1_NO_MB").is_ok(),
                    _ => std::env::var("INFER_Q8K_NO_MB").is_ok(),
                };
                if m <= 8 && !mb_off {
                    if m == 1 {
                        let av = a.view_offset(vec![k], 0);
                        let cv = c.view_offset(vec![n], 0);
                        return self.mv_quantized(&av, b, &cv);
                    }
                    if matches!(b.dtype, GpuDtype::Q8_1 | GpuDtype::Q8_K) {
                        return self.mv_q8_1_q8_k_mb(a, b, c);
                    }
                    return self.mv_q4_1_q5_1_mb(a, b, c);
                }
                let blk = if b.dtype == GpuDtype::Q8_K { 256 } else { 32 };
                if m > 8 && k.is_multiple_of(blk) {
                    let name = match b.dtype {
                        GpuDtype::Q4_1 => "gemm_nt_q4_1_mm",
                        GpuDtype::Q5_1 => "gemm_nt_q5_1_mm",
                        GpuDtype::Q8_1 => "gemm_nt_q8_1_mm",
                        _ => "gemm_nt_q8_k_mm",
                    };
                    let (m32, k32, n32) = (m as u32, k as u32, n as u32);
                    let grid =
                        MTLSize::new((m.div_ceil(32) * 128) as u64, n.div_ceil(64) as u64, 1);
                    let group = MTLSize::new(128, 1, 1);
                    return self.run(name, grid, group, |enc| {
                        enc.set_buffer(0, Some(&a.buffer), a.offset as u64);
                        enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
                        enc.set_buffer(2, Some(&c.buffer), c.offset as u64);
                        enc.set_bytes(3, 4, (&m32 as *const u32).cast());
                        enc.set_bytes(4, 4, (&k32 as *const u32).cast());
                        enc.set_bytes(5, 4, (&n32 as *const u32).cast());
                        enc.set_threadgroup_memory_length(0, 8192);
                    });
                }
                let av = a.view_offset(vec![k], 0);
                let cv = c.view_offset(vec![n], 0);
                return self.mv_quantized(&av, b, &cv);
            }
            // Q2_K / Q3_K: окна m 2..8 — батчевый matvec (2 лейна на
            // подблок), m>8 — mm-тайл (шаг 32 = 2 подблока). Блок 256.
            GpuDtype::Q2_K | GpuDtype::Q3_K => {
                let mb_off = if b.dtype == GpuDtype::Q2_K {
                    std::env::var("INFER_Q2K_NO_MB").is_ok()
                } else {
                    std::env::var("INFER_Q3K_NO_MB").is_ok()
                };
                if m <= 8 && !mb_off {
                    if m == 1 {
                        let av = a.view_offset(vec![k], 0);
                        let cv = c.view_offset(vec![n], 0);
                        return self.mv_quantized(&av, b, &cv);
                    }
                    return self.mv_q2_k_q3_k_mb(a, b, c);
                }
                if m > 8 && k.is_multiple_of(256) {
                    let name = if b.dtype == GpuDtype::Q2_K {
                        "gemm_nt_q2_k_mm"
                    } else {
                        "gemm_nt_q3_k_mm"
                    };
                    let (m32, k32, n32) = (m as u32, k as u32, n as u32);
                    let grid =
                        MTLSize::new((m.div_ceil(32) * 128) as u64, n.div_ceil(64) as u64, 1);
                    let group = MTLSize::new(128, 1, 1);
                    return self.run(name, grid, group, |enc| {
                        enc.set_buffer(0, Some(&a.buffer), a.offset as u64);
                        enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
                        enc.set_buffer(2, Some(&c.buffer), c.offset as u64);
                        enc.set_bytes(3, 4, (&m32 as *const u32).cast());
                        enc.set_bytes(4, 4, (&k32 as *const u32).cast());
                        enc.set_bytes(5, 4, (&n32 as *const u32).cast());
                        enc.set_threadgroup_memory_length(0, 8192);
                    });
                }
                let av = a.view_offset(vec![k], 0);
                let cv = c.view_offset(vec![n], 0);
                return self.mv_quantized(&av, b, &cv);
            }
            GpuDtype::Iq4Nl | GpuDtype::Mxfp4 | GpuDtype::Nvfp4 => {
                let mb_off = match b.dtype {
                    GpuDtype::Iq4Nl => std::env::var("INFER_IQ4NL_NO_MB").is_ok(),
                    GpuDtype::Mxfp4 => std::env::var("INFER_MXFP4_NO_MB").is_ok(),
                    _ => std::env::var("INFER_NVFP4_NO_MB").is_ok(),
                };
                if m <= 8 && !mb_off {
                    if m == 1 {
                        // Замер mb1 vs скалярного mv на m=1: mb1 ХУЖЕ на
                        // IQ4_NL (79 vs 40 мс), MXFP4 (120 vs 36), Q3_K
                        // (133 vs 47), остальные — паритет. Оставлен
                        // скалярный mv (порядок суммирования точный).
                        let av = a.view_offset(vec![k], 0);
                        let cv = c.view_offset(vec![n], 0);
                        return self.mv_quantized(&av, b, &cv);
                    }
                    if b.dtype == GpuDtype::Nvfp4 {
                        return self.mv_nvfp4_mb(a, b, c);
                    }
                    return self.mv_iq4_nl_mxfp4_mb(a, b, c);
                }
                let mm_ok = if b.dtype == GpuDtype::Nvfp4 {
                    k.is_multiple_of(64)
                } else {
                    k.is_multiple_of(32)
                };
                if m > 8 && mm_ok {
                    let name = match b.dtype {
                        GpuDtype::Iq4Nl => "gemm_nt_iq4_nl_mm",
                        GpuDtype::Mxfp4 => "gemm_nt_mxfp4_mm",
                        _ => "gemm_nt_nvfp4_mm",
                    };
                    let (m32, k32, n32) = (m as u32, k as u32, n as u32);
                    let grid =
                        MTLSize::new((m.div_ceil(32) * 128) as u64, n.div_ceil(64) as u64, 1);
                    let group = MTLSize::new(128, 1, 1);
                    return self.run(name, grid, group, |enc| {
                        enc.set_buffer(0, Some(&a.buffer), a.offset as u64);
                        enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
                        enc.set_buffer(2, Some(&c.buffer), c.offset as u64);
                        enc.set_bytes(3, 4, (&m32 as *const u32).cast());
                        enc.set_bytes(4, 4, (&k32 as *const u32).cast());
                        enc.set_bytes(5, 4, (&n32 as *const u32).cast());
                        enc.set_threadgroup_memory_length(0, 8192);
                    });
                }
                // m=1 без mb-пути (рычаг): скалярный mv.
                let av = a.view_offset(vec![k], 0);
                let cv = c.view_offset(vec![n], 0);
                return self.mv_quantized(&av, b, &cv);
            }
            GpuDtype::Iq4Xs => {
                let mm_min = std::env::var("INFER_QUANT_MM_MIN")
                    .or_else(|_| std::env::var("INFER_IQ4XS_MM_MIN"))
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(2);
                let mb_on = k.is_multiple_of(256) && std::env::var("INFER_IQ4XS_NO_MB").is_err();
                if mb_on && (2..=5).contains(&m) {
                    return self.mv_iq4_xs_mb(a, b, c, k, n);
                }
                if m >= mm_min
                    && m > 1
                    && k.is_multiple_of(256)
                    && std::env::var("INFER_IQ4XS_NO_MM").is_err()
                {
                    let (m32, k32, n32) = (m as u32, k as u32, n as u32);
                    let grid =
                        MTLSize::new((m.div_ceil(32) * 128) as u64, n.div_ceil(64) as u64, 1);
                    let group = MTLSize::new(128, 1, 1);
                    return self.run("gemm_nt_iq4_xs_mm", grid, group, |enc| {
                        enc.set_buffer(0, Some(&a.buffer), a.offset as u64);
                        enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
                        enc.set_buffer(2, Some(&c.buffer), c.offset as u64);
                        enc.set_bytes(3, 4, (&m32 as *const u32).cast());
                        enc.set_bytes(4, 4, (&k32 as *const u32).cast());
                        enc.set_bytes(5, 4, (&n32 as *const u32).cast());
                        enc.set_threadgroup_memory_length(0, 8192);
                    });
                }
                // m 6..8 (и паритет-режим при mm_min>8): батчевый matvec;
                // m=1 — r2-декод внутри dispatch.
                if mb_on && m <= 8 {
                    return self.mv_iq4_xs_mb(a, b, c, k, n);
                }
                return self.mv_iq4_xs_dispatch(a, b, c, k, n);
            }
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
        // 1 < m ≤ 8, Q8_0 (окно верификации спека, черновой блок DFlash):
        // matvec с батчем (mv_q8_0_mb) — веса читаются один раз на m строк
        // активаций. Тайловый GEMM ниже при таком m стоил ≈3.5× matvec'а —
        // forward 1+γ строк был дороже трёх шагов декода, и спек проигрывал
        // при любой приёмке. m = 7, 8 — два прохода (4 + остаток). Активации
        // читаются float4: нужен 16-байтовый сдвиг. INFER_GEMM_Q8_SMALL_MM=1
        // — прежний тайловый путь (сверка).
        if m > 1
            && m <= 8
            && b.dtype == GpuDtype::Q8_0
            && a.offset.is_multiple_of(16)
            && std::env::var("INFER_GEMM_Q8_SMALL_MM").is_err()
        {
            let (k32, n32) = (k as u32, n as u32);
            let pass = |rows: usize, row0: usize| -> Result<(), MetalError> {
                let (name, nr) = match rows {
                    2 => ("mv_q8_0_mb2", 4usize),
                    3 => ("mv_q8_0_mb3", 4),
                    4 => ("mv_q8_0_mb4", 4),
                    5 => ("mv_q8_0_mb5", 8),
                    _ => ("mv_q8_0_mb6", 8),
                };
                // 2 simdgroup'а на threadgroup, NR строк весов на simdgroup
                let grid = MTLSize::new((n.div_ceil(nr * 2) * 64) as u64, 1, 1);
                let group = MTLSize::new(64, 1, 1);
                let a_off = (a.offset + row0 * k * 4) as u64;
                let c_off = (c.offset + row0 * n * 4) as u64;
                self.run(name, grid, group, |enc| {
                    enc.set_buffer(0, Some(&a.buffer), a_off);
                    enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
                    enc.set_buffer(2, Some(&c.buffer), c_off);
                    enc.set_bytes(3, 4, (&k32 as *const u32).cast());
                    enc.set_bytes(4, 4, (&n32 as *const u32).cast());
                })
            };
            return match m {
                7 | 8 => {
                    pass(4, 0)?;
                    pass(m - 4, 4)
                }
                _ => pass(m, 0),
            };
        }
        // m>1, Q8_0: тайловый GEMM на матричных блоках simdgroup (prefill и
        // верификация спека). INFER_GEMM_Q8_LEGACY=1 — прежний m8f4 (сверка).
        if m > 1 && b.dtype == GpuDtype::Q8_0 && std::env::var("INFER_GEMM_Q8_LEGACY").is_err() {
            let (m32, k32, n32) = (m as u32, k as u32, n as u32);
            // Схема ggml kernel_mul_mm: threadgroup 128 потоков на тайл
            // 32 (M) × 64 (N); grid — (M-тайлы, N-тайлы) threadgroup'ов.
            // Тайл 64×128 пробовался (вчетверо меньше перечитывания
            // активаций) — в пределах шума: 5.1–6.7 против 5.7–6.7 TFLOPS,
            // при M=256 даже хуже (меньше threadgroup'ов в полёте).
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
        // m>1: m8-кернел (n-столбец × 8 m-строк, явные аккумуляторы) —
        // B читается 1 раз на 8 строк. f16-веса сюда не попадают (они идут
        // через gemm_nt_f32 → gemm_nt_f16_m8).
        if m > 1 {
            let kname = match b.dtype {
                GpuDtype::Q8_0 => "gemm_nt_q8_0_m8f4",
                _ => "",
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
    /// Выбор ext-кернела зеркалом эталонного диспатча
    /// (ggml_metal_op_mul_mat, 5266f24da): src1 F32, K%128==0;
    /// F32/F16/Q4_0/Q5_0/Q8_0 — m∈[2,8]; Q4_K/Q5_K/Q6_K — m∈[4,8];
    /// IQ4_XS — в ext-списке эталона нет (всегда classic mul_mv при m≤8).
    /// nxpsg = (K%256==0 && m<3) ? 16 : 8; r1ptg: 2→2, 3/6→3, 4/7/8→4, 5→5.
    fn mv_ext_kernel(dtype: GpuDtype, m: usize, k: usize) -> Option<(&'static str, usize, usize)> {
        if !(2..=8).contains(&m) || !k.is_multiple_of(128) {
            return None;
        }
        let (x8r2, x8r3, x8r4, x8r5, x16r2): (&str, &str, &str, &str, &str) = match dtype {
            GpuDtype::F32 => (
                "mv_ext_f32_x8_r2",
                "mv_ext_f32_x8_r3",
                "mv_ext_f32_x8_r4",
                "mv_ext_f32_x8_r5",
                "mv_ext_f32_x16_r2",
            ),
            GpuDtype::F16 => (
                "mv_ext_f16_x8_r2",
                "mv_ext_f16_x8_r3",
                "mv_ext_f16_x8_r4",
                "mv_ext_f16_x8_r5",
                "mv_ext_f16_x16_r2",
            ),
            GpuDtype::Q4_0 => (
                "mv_ext_q4_0_x8_r2",
                "mv_ext_q4_0_x8_r3",
                "mv_ext_q4_0_x8_r4",
                "mv_ext_q4_0_x8_r5",
                "mv_ext_q4_0_x16_r2",
            ),
            GpuDtype::Q5_0 => (
                "mv_ext_q5_0_x8_r2",
                "mv_ext_q5_0_x8_r3",
                "mv_ext_q5_0_x8_r4",
                "mv_ext_q5_0_x8_r5",
                "mv_ext_q5_0_x16_r2",
            ),
            GpuDtype::Q8_0 => (
                "mv_ext_q8_0_x8_r2",
                "mv_ext_q8_0_x8_r3",
                "mv_ext_q8_0_x8_r4",
                "mv_ext_q8_0_x8_r5",
                "mv_ext_q8_0_x16_r2",
            ),
            GpuDtype::Q4_K => (
                "",
                "mv_ext_q4_k_x8_r3",
                "mv_ext_q4_k_x8_r4",
                "mv_ext_q4_k_x8_r5",
                "",
            ),
            GpuDtype::Q5_K => (
                "",
                "mv_ext_q5_k_x8_r3",
                "mv_ext_q5_k_x8_r4",
                "mv_ext_q5_k_x8_r5",
                "",
            ),
            GpuDtype::Q6_K => (
                "",
                "mv_ext_q6_k_x8_r3",
                "mv_ext_q6_k_x8_r4",
                "mv_ext_q6_k_x8_r5",
                "",
            ),
            GpuDtype::BF16
            | GpuDtype::Iq4Xs
            | GpuDtype::Iq4Nl
            | GpuDtype::Mxfp4
            | GpuDtype::Nvfp4
            | GpuDtype::Q2_K
            | GpuDtype::Q3_K
            | GpuDtype::Q4_1
            | GpuDtype::Q5_1
            | GpuDtype::Q8_1
            | GpuDtype::Q8_K
            | GpuDtype::Q1_0
            | GpuDtype::Q2_0
            | GpuDtype::TQ1_0
            | GpuDtype::TQ2_0
            | GpuDtype::Iq2Xxs
            | GpuDtype::Iq3Xxs
            | GpuDtype::Iq2Xs
            | GpuDtype::Iq2S
            | GpuDtype::Iq3S
            | GpuDtype::Iq1S
            | GpuDtype::Iq1M => return None,
        };
        let lo = if matches!(dtype, GpuDtype::Q4_K | GpuDtype::Q5_K | GpuDtype::Q6_K) {
            4
        } else {
            2
        };
        if m < lo {
            return None;
        }
        let nxpsg = if k.is_multiple_of(256) && m < 3 {
            16
        } else {
            8
        };
        let r1ptg = match m {
            2 => 2,
            3 | 6 => 3,
            4 | 7 | 8 => 4,
            5 => 5,
            _ => return None,
        };
        let name = match (nxpsg, r1ptg) {
            (16, 2) => x16r2,
            (8, 2) => x8r2,
            (8, 3) => x8r3,
            (8, 4) => x8r4,
            (8, 5) => x8r5,
            _ => return None,
        };
        if name.is_empty() {
            return None;
        }
        Some((name, nxpsg, r1ptg))
    }

    /// mul_mv_ext — малобатчевый matvec (порт эталонного kernel_mul_mv_ext,
    /// см. ext-секцию mul_mv.metal). Ok(true) — кернел запущен; Ok(false) —
    /// случай вне ext-окна/раскладки, вызывающий идёт своим путём (как
    /// эталон вне ext).
    pub fn mv_ext(&self, a: &GpuTensor, b: &GpuTensor, c: &GpuTensor) -> Result<bool, MetalError> {
        let m = a.shape[0];
        let k = a.shape[1];
        let n = b.shape[0];
        let Some((name, nxpsg, r1ptg)) = Self::mv_ext_kernel(b.dtype, m, k) else {
            return Ok(false);
        };
        // float4-чтения активаций — 16-байтное выравнивание; широкие чтения
        // весов: F32 — float4 (16), F16 — half4 (8), квант-блоки — char4/
        // uint16/uint32 внутри строки (4; строки кратны 4 при K%128==0).
        let b_align = match b.dtype {
            GpuDtype::F32 => 16,
            GpuDtype::F16 => 8,
            _ => 4,
        };
        if !a.offset.is_multiple_of(16) || !b.offset.is_multiple_of(b_align) {
            return Ok(false);
        }
        let nb01: u64 = match b.dtype {
            GpuDtype::F32 => (k * 4) as u64,
            GpuDtype::F16 => (k * 2) as u64,
            q => ((k / q.block_elems().unwrap()) * q.block_bytes().unwrap()) as u64,
        };
        let nb11: u64 = (k * 4) as u64;
        if std::env::var("INFER_EXT_DBG").is_ok() {
            eprintln!(
                "mv_ext {name}: m={m} n={n} k={k} nb01={nb01} nb11={nb11} a.off={} b.off={} ntg=({},{})",
                a.offset,
                b.offset,
                n.div_ceil(32 / nxpsg * 2),
                m.div_ceil(r1ptg)
            );
        }
        let (ne00, ne01, ne11, ne0) = (k as i32, n as i32, m as i32, n as i32);
        // Диспатч эталона: threadgroups (ceil(ne01/r0ptg), ceil(ne11/r1ptg), 1),
        // r0ptg = (32/nxpsg)*nsg, nsg=2, 64 потока в threadgroup'е.
        let r0ptg = (32 / nxpsg) * 2;
        let ntg_x = n.div_ceil(r0ptg);
        let ntg_y = m.div_ceil(r1ptg);
        let grid = MTLSize::new((ntg_x * 64) as u64, ntg_y as u64, 1);
        let group = MTLSize::new(64, 1, 1);
        self.run(name, grid, group, |enc| {
            enc.set_buffer(0, Some(&a.buffer), a.offset as u64);
            enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
            enc.set_buffer(2, Some(&c.buffer), c.offset as u64);
            enc.set_bytes(3, 4, (&ne00 as *const i32).cast());
            enc.set_bytes(4, 4, (&ne01 as *const i32).cast());
            enc.set_bytes(5, 4, (&ne11 as *const i32).cast());
            enc.set_bytes(6, 8, (&nb01 as *const u64).cast());
            enc.set_bytes(7, 8, (&nb11 as *const u64).cast());
            enc.set_bytes(8, 4, (&ne0 as *const i32).cast());
        })?;
        Ok(true)
    }

    /// Q4_0 матвек с батчем строк активаций (окно верификации 1 < m ≤ 8):
    /// веса читаются 1× на m строк активаций — аналог mv_q8_0_mb для Q4_0.
    /// До этого у Q4_0 батчевого пути не было вовсе: окна шли mv_ext'ом
    /// построчно (веса m раз), у mm-тайла Q4_0 тоже нет — m>8 идёт скалярным
    /// gemm_nt_q4_0. m 7/8 — два прохода (4+остаток). Требования: K % 32,
    /// A сдвинут на 16 (float2-загрузки; 8-байтового хватило бы, но все
    /// тензоры движка выровнены по 4096 — единый гейт проще).
    pub fn mv_q4_0_mb(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        c: &GpuTensor,
    ) -> Result<(), MetalError> {
        let k = a.shape[1];
        let n = b.shape[0];
        let m = a.shape[0];
        if b.shape != [n, k] || c.shape != [m, n] {
            return Err(MetalError::Shape(format!(
                "mv_q4_0_mb {:?}x{:?}->{:?}",
                a.shape, b.shape, c.shape
            )));
        }
        if b.dtype != GpuDtype::Q4_0 {
            return Err(MetalError::Shape(format!(
                "mv_q4_0_mb: dtype {:?} не Q4_0",
                b.dtype
            )));
        }
        if !k.is_multiple_of(32) {
            return Err(MetalError::Shape(format!(
                "mv_q4_0_mb: K={k} не кратен 32 (хвост блока не обрабатывается)"
            )));
        }
        if !a.offset.is_multiple_of(16) {
            return Err(MetalError::Shape(format!(
                "mv_q4_0_mb: A.offset={} не 16-выровнен (float2-загрузки)",
                a.offset
            )));
        }
        let (k32, n32) = (k as u32, n as u32);
        let pass = |rows: usize, row0: usize| -> Result<(), MetalError> {
            let (name, nr) = match rows {
                2 => ("mv_q4_0_mb2", 4usize),
                3 => ("mv_q4_0_mb3", 4),
                4 => ("mv_q4_0_mb4", 4),
                5 => ("mv_q4_0_mb5", 8),
                _ => ("mv_q4_0_mb6", 8),
            };
            // 2 simdgroup'а на threadgroup, NR строк весов на simdgroup.
            let grid = MTLSize::new((n.div_ceil(nr * 2) * 64) as u64, 1, 1);
            let group = MTLSize::new(64, 1, 1);
            let a_off = (a.offset + row0 * k * 4) as u64;
            let c_off = (c.offset + row0 * n * 4) as u64;
            self.run(name, grid, group, |enc| {
                enc.set_buffer(0, Some(&a.buffer), a_off);
                enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
                enc.set_buffer(2, Some(&c.buffer), c_off);
                enc.set_bytes(3, 4, (&k32 as *const u32).cast());
                enc.set_bytes(4, 4, (&n32 as *const u32).cast());
            })
        };
        match m {
            7 | 8 => {
                pass(4, 0)?;
                pass(m - 4, 4)
            }
            _ => pass(m, 0),
        }
    }

    /// Q1_0 / Q2_0 / TQ1_0 / TQ2_0 матвек с батчем строк активаций (окно
    /// 1 < m ≤ 8): 32 лейна на строку, лейн — per/32 значений подряд
    /// (Q1_0 4, Q2_0 2, TQ 8); значение берётся точечной функцией по позиции
    /// в блоке. Требования: K%(128|64|256).
    pub fn mv_tern_mb(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        c: &GpuTensor,
    ) -> Result<(), MetalError> {
        let k = a.shape[1];
        let n = b.shape[0];
        let m = a.shape[0];
        if b.shape != [n, k] || c.shape != [m, n] {
            return Err(MetalError::Shape(format!(
                "mv_tern_mb {:?}x{:?}->{:?}",
                a.shape, b.shape, c.shape
            )));
        }
        let (stem, blk) = match b.dtype {
            GpuDtype::Q1_0 => ("mv_q1_0_mb", 128usize),
            GpuDtype::Q2_0 => ("mv_q2_0_mb", 64),
            GpuDtype::TQ1_0 => ("mv_tq1_0_mb", 256),
            GpuDtype::TQ2_0 => ("mv_tq2_0_mb", 256),
            GpuDtype::Iq2Xxs => ("mv_iq2_xxs_mb", 256),
            GpuDtype::Iq3Xxs => ("mv_iq3_xxs_mb", 256),
            GpuDtype::Iq2Xs => ("mv_iq2_xs_mb", 256),
            GpuDtype::Iq2S => ("mv_iq2_s_mb", 256),
            GpuDtype::Iq3S => ("mv_iq3_s_mb", 256),
            GpuDtype::Iq1S => ("mv_iq1_s_mb", 256),
            GpuDtype::Iq1M => ("mv_iq1_m_mb", 256),
            other => {
                return Err(MetalError::Shape(format!(
                    "mv_tern_mb: dtype {other:?} не Q1_0/Q2_0/TQ1_0/TQ2_0"
                )));
            }
        };
        if !k.is_multiple_of(blk) {
            return Err(MetalError::Shape(format!(
                "mv_tern_mb: K={k} не кратен {blk}"
            )));
        }
        let (k32, n32) = (k as u32, n as u32);
        let pass = |rows: usize, row0: usize| -> Result<(), MetalError> {
            let (name, nr): (&'static str, usize) = match (stem, rows) {
                ("mv_q1_0_mb", 2) => ("mv_q1_0_mb2", 4),
                ("mv_q1_0_mb", 3) => ("mv_q1_0_mb3", 4),
                ("mv_q1_0_mb", 4) => ("mv_q1_0_mb4", 4),
                ("mv_q1_0_mb", 5) => ("mv_q1_0_mb5", 8),
                ("mv_q1_0_mb", _) => ("mv_q1_0_mb6", 8),
                ("mv_q2_0_mb", 2) => ("mv_q2_0_mb2", 4),
                ("mv_q2_0_mb", 3) => ("mv_q2_0_mb3", 4),
                ("mv_q2_0_mb", 4) => ("mv_q2_0_mb4", 4),
                ("mv_q2_0_mb", 5) => ("mv_q2_0_mb5", 8),
                ("mv_q2_0_mb", _) => ("mv_q2_0_mb6", 8),
                ("mv_tq1_0_mb", 2) => ("mv_tq1_0_mb2", 4),
                ("mv_tq1_0_mb", 3) => ("mv_tq1_0_mb3", 4),
                ("mv_tq1_0_mb", 4) => ("mv_tq1_0_mb4", 4),
                ("mv_tq1_0_mb", 5) => ("mv_tq1_0_mb5", 8),
                ("mv_tq1_0_mb", _) => ("mv_tq1_0_mb6", 8),
                ("mv_tq2_0_mb", 2) => ("mv_tq2_0_mb2", 4),
                ("mv_iq2_xxs_mb", 2) => ("mv_iq2_xxs_mb2", 4),
                ("mv_iq3_xxs_mb", 2) => ("mv_iq3_xxs_mb2", 4),
                ("mv_tq2_0_mb", 3) => ("mv_tq2_0_mb3", 4),
                ("mv_iq2_xxs_mb", 3) => ("mv_iq2_xxs_mb3", 4),
                ("mv_iq3_xxs_mb", 3) => ("mv_iq3_xxs_mb3", 4),
                ("mv_tq2_0_mb", 4) => ("mv_tq2_0_mb4", 4),
                ("mv_iq2_xxs_mb", 4) => ("mv_iq2_xxs_mb4", 4),
                ("mv_iq3_xxs_mb", 4) => ("mv_iq3_xxs_mb4", 4),
                ("mv_tq2_0_mb", 5) => ("mv_tq2_0_mb5", 8),
                ("mv_iq2_xxs_mb", 5) => ("mv_iq2_xxs_mb5", 8),
                ("mv_iq3_xxs_mb", 5) => ("mv_iq3_xxs_mb5", 8),
                ("mv_tq2_0_mb", _) => ("mv_tq2_0_mb6", 8),
                ("mv_iq2_xxs_mb", _) => ("mv_iq2_xxs_mb6", 8),
                ("mv_iq2_xs_mb", 2) => ("mv_iq2_xs_mb2", 4),
                ("mv_iq2_xs_mb", 3) => ("mv_iq2_xs_mb3", 4),
                ("mv_iq2_xs_mb", 4) => ("mv_iq2_xs_mb4", 4),
                ("mv_iq2_xs_mb", 5) => ("mv_iq2_xs_mb5", 8),
                ("mv_iq2_xs_mb", _) => ("mv_iq2_xs_mb6", 8),
                ("mv_iq2_s_mb", 2) => ("mv_iq2_s_mb2", 4),
                ("mv_iq2_s_mb", 3) => ("mv_iq2_s_mb3", 4),
                ("mv_iq2_s_mb", 4) => ("mv_iq2_s_mb4", 4),
                ("mv_iq2_s_mb", 5) => ("mv_iq2_s_mb5", 8),
                ("mv_iq2_s_mb", _) => ("mv_iq2_s_mb6", 8),
                ("mv_iq3_s_mb", 2) => ("mv_iq3_s_mb2", 4),
                ("mv_iq3_s_mb", 3) => ("mv_iq3_s_mb3", 4),
                ("mv_iq3_s_mb", 4) => ("mv_iq3_s_mb4", 4),
                ("mv_iq3_s_mb", 5) => ("mv_iq3_s_mb5", 8),
                ("mv_iq3_s_mb", _) => ("mv_iq3_s_mb6", 8),
                ("mv_iq1_s_mb", 2) => ("mv_iq1_s_mb2", 4),
                ("mv_iq1_s_mb", 3) => ("mv_iq1_s_mb3", 4),
                ("mv_iq1_s_mb", 4) => ("mv_iq1_s_mb4", 4),
                ("mv_iq1_s_mb", 5) => ("mv_iq1_s_mb5", 8),
                ("mv_iq1_s_mb", _) => ("mv_iq1_s_mb6", 8),
                ("mv_iq1_m_mb", 2) => ("mv_iq1_m_mb2", 4),
                ("mv_iq1_m_mb", 3) => ("mv_iq1_m_mb3", 4),
                ("mv_iq1_m_mb", 4) => ("mv_iq1_m_mb4", 4),
                ("mv_iq1_m_mb", 5) => ("mv_iq1_m_mb5", 8),
                ("mv_iq1_m_mb", _) => ("mv_iq1_m_mb6", 8),
                _ => ("mv_iq3_xxs_mb6", 8),
            };
            let grid = MTLSize::new((n.div_ceil(nr * 2) * 64) as u64, 1, 1);
            let group = MTLSize::new(64, 1, 1);
            let a_off = (a.offset + row0 * k * 4) as u64;
            let c_off = (c.offset + row0 * n * 4) as u64;
            self.run(name, grid, group, |enc| {
                enc.set_buffer(0, Some(&a.buffer), a_off);
                enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
                enc.set_buffer(2, Some(&c.buffer), c_off);
                enc.set_bytes(3, 4, (&k32 as *const u32).cast());
                enc.set_bytes(4, 4, (&n32 as *const u32).cast());
            })
        };
        match m {
            7 | 8 => {
                pass(4, 0)?;
                pass(m - 4, 4)
            }
            _ => pass(m, 0),
        }
    }

    /// Q4_1 / Q5_1 матвек с батчем строк активаций (окно 1 < m ≤ 8):
    /// раскладка как у Q4_0/Q5_0 (8 лейнов на блок, 2 байта = 4 значения;
    /// Q5_1 — 5-й бит из qh), значение q·d + m. Требования: K%32, A 16-выровнен.
    pub fn mv_q4_1_q5_1_mb(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        c: &GpuTensor,
    ) -> Result<(), MetalError> {
        let k = a.shape[1];
        let n = b.shape[0];
        let m = a.shape[0];
        if b.shape != [n, k] || c.shape != [m, n] {
            return Err(MetalError::Shape(format!(
                "mv_q4_1_q5_1_mb {:?}x{:?}->{:?}",
                a.shape, b.shape, c.shape
            )));
        }
        let stem = match b.dtype {
            GpuDtype::Q4_1 => "mv_q4_1_mb",
            GpuDtype::Q5_1 => "mv_q5_1_mb",
            other => {
                return Err(MetalError::Shape(format!(
                    "mv_q4_1_q5_1_mb: dtype {other:?} не Q4_1/Q5_1"
                )));
            }
        };
        if !k.is_multiple_of(32) {
            return Err(MetalError::Shape(format!(
                "mv_q4_1_q5_1_mb: K={k} не кратен 32"
            )));
        }
        if !a.offset.is_multiple_of(16) {
            return Err(MetalError::Shape(format!(
                "mv_q4_1_q5_1_mb: A.offset={} не 16-выровнен",
                a.offset
            )));
        }
        let (k32, n32) = (k as u32, n as u32);
        let pass = |rows: usize, row0: usize| -> Result<(), MetalError> {
            let (name, nr): (&'static str, usize) = match (stem, rows) {
                ("mv_q4_1_mb", 2) => ("mv_q4_1_mb2", 4),
                ("mv_q4_1_mb", 3) => ("mv_q4_1_mb3", 4),
                ("mv_q4_1_mb", 4) => ("mv_q4_1_mb4", 4),
                ("mv_q4_1_mb", 5) => ("mv_q4_1_mb5", 8),
                ("mv_q4_1_mb", _) => ("mv_q4_1_mb6", 8),
                ("mv_q5_1_mb", 2) => ("mv_q5_1_mb2", 4),
                ("mv_q5_1_mb", 3) => ("mv_q5_1_mb3", 4),
                ("mv_q5_1_mb", 4) => ("mv_q5_1_mb4", 4),
                ("mv_q5_1_mb", 5) => ("mv_q5_1_mb5", 8),
                _ => ("mv_q5_1_mb6", 8),
            };
            let grid = MTLSize::new((n.div_ceil(nr * 2) * 64) as u64, 1, 1);
            let group = MTLSize::new(64, 1, 1);
            let a_off = (a.offset + row0 * k * 4) as u64;
            let c_off = (c.offset + row0 * n * 4) as u64;
            self.run(name, grid, group, |enc| {
                enc.set_buffer(0, Some(&a.buffer), a_off);
                enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
                enc.set_buffer(2, Some(&c.buffer), c_off);
                enc.set_bytes(3, 4, (&k32 as *const u32).cast());
                enc.set_bytes(4, 4, (&n32 as *const u32).cast());
            })
        };
        match m {
            7 | 8 => {
                pass(4, 0)?;
                pass(m - 4, 4)
            }
            _ => pass(m, 0),
        }
    }

    /// Q8_1 / Q8_K матвек с батчем строк активаций (окно 1 < m ≤ 8):
    /// Q8_1 — 8 лейнов на 32-эл. блок по 4 значения; Q8_K — 32 лейна на
    /// строку по 8 значений из 256. Значение d·q. Требования: K%(32|256).
    pub fn mv_q8_1_q8_k_mb(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        c: &GpuTensor,
    ) -> Result<(), MetalError> {
        let k = a.shape[1];
        let n = b.shape[0];
        let m = a.shape[0];
        if b.shape != [n, k] || c.shape != [m, n] {
            return Err(MetalError::Shape(format!(
                "mv_q8_1_q8_k_mb {:?}x{:?}->{:?}",
                a.shape, b.shape, c.shape
            )));
        }
        let (stem, blk) = match b.dtype {
            GpuDtype::Q8_1 => ("mv_q8_1_mb", 32usize),
            GpuDtype::Q8_K => ("mv_q8_k_mb", 256),
            other => {
                return Err(MetalError::Shape(format!(
                    "mv_q8_1_q8_k_mb: dtype {other:?} не Q8_1/Q8_K"
                )));
            }
        };
        if !k.is_multiple_of(blk) {
            return Err(MetalError::Shape(format!(
                "mv_q8_1_q8_k_mb: K={k} не кратен {blk}"
            )));
        }
        let (k32, n32) = (k as u32, n as u32);
        let pass = |rows: usize, row0: usize| -> Result<(), MetalError> {
            let (name, nr): (&'static str, usize) = match (stem, rows) {
                ("mv_q8_1_mb", 2) => ("mv_q8_1_mb2", 4),
                ("mv_q8_1_mb", 3) => ("mv_q8_1_mb3", 4),
                ("mv_q8_1_mb", 4) => ("mv_q8_1_mb4", 4),
                ("mv_q8_1_mb", 5) => ("mv_q8_1_mb5", 8),
                ("mv_q8_1_mb", _) => ("mv_q8_1_mb6", 8),
                ("mv_q8_k_mb", 2) => ("mv_q8_k_mb2", 4),
                ("mv_q8_k_mb", 3) => ("mv_q8_k_mb3", 4),
                ("mv_q8_k_mb", 4) => ("mv_q8_k_mb4", 4),
                ("mv_q8_k_mb", 5) => ("mv_q8_k_mb5", 8),
                _ => ("mv_q8_k_mb6", 8),
            };
            let grid = MTLSize::new((n.div_ceil(nr * 2) * 64) as u64, 1, 1);
            let group = MTLSize::new(64, 1, 1);
            let a_off = (a.offset + row0 * k * 4) as u64;
            let c_off = (c.offset + row0 * n * 4) as u64;
            self.run(name, grid, group, |enc| {
                enc.set_buffer(0, Some(&a.buffer), a_off);
                enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
                enc.set_buffer(2, Some(&c.buffer), c_off);
                enc.set_bytes(3, 4, (&k32 as *const u32).cast());
                enc.set_bytes(4, 4, (&n32 as *const u32).cast());
            })
        };
        match m {
            7 | 8 => {
                pass(4, 0)?;
                pass(m - 4, 4)
            }
            _ => pass(m, 0),
        }
    }

    /// Q2_K / Q3_K матвек с батчем строк активаций (окно 1 < m ≤ 8): 32
    /// лейна на строку, 2 лейна на подблок (16 значений → по 8 байт).
    /// Блок 256: Q2_K 84 Б (scales[16], qs[64], d, dmin), Q3_K 110 Б
    /// (hmask[32], qs[64], scales[12], d). m 7/8 — два прохода.
    /// Требования: K%256; A читается скалярно (выравнивание не нужно).
    pub fn mv_q2_k_q3_k_mb(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        c: &GpuTensor,
    ) -> Result<(), MetalError> {
        let k = a.shape[1];
        let n = b.shape[0];
        let m = a.shape[0];
        if b.shape != [n, k] || c.shape != [m, n] {
            return Err(MetalError::Shape(format!(
                "mv_q2_k_q3_k_mb {:?}x{:?}->{:?}",
                a.shape, b.shape, c.shape
            )));
        }
        let stem = match b.dtype {
            GpuDtype::Q2_K => "mv_q2_k_mb",
            GpuDtype::Q3_K => "mv_q3_k_mb",
            other => {
                return Err(MetalError::Shape(format!(
                    "mv_q2_k_q3_k_mb: dtype {other:?} не Q2_K/Q3_K"
                )));
            }
        };
        if !k.is_multiple_of(256) {
            return Err(MetalError::Shape(format!(
                "mv_q2_k_q3_k_mb: K={k} не кратен 256"
            )));
        }
        let (k32, n32) = (k as u32, n as u32);
        let pass = |rows: usize, row0: usize| -> Result<(), MetalError> {
            let name: &'static str = match (stem, rows) {
                ("mv_q2_k_mb", 2) => "mv_q2_k_mb2",
                ("mv_q2_k_mb", 3) => "mv_q2_k_mb3",
                ("mv_q2_k_mb", 4) => "mv_q2_k_mb4",
                ("mv_q2_k_mb", 5) => "mv_q2_k_mb5",
                ("mv_q2_k_mb", _) => "mv_q2_k_mb6",
                ("mv_q3_k_mb", 2) => "mv_q3_k_mb2",
                ("mv_q3_k_mb", 3) => "mv_q3_k_mb3",
                ("mv_q3_k_mb", 4) => "mv_q3_k_mb4",
                ("mv_q3_k_mb", 5) => "mv_q3_k_mb5",
                _ => "mv_q3_k_mb6",
            };
            // grid = ПОТОКИ: 32 потока (один simdgroup) на строку весов.
            let grid = MTLSize::new(32, n as u64, 1);
            let group = MTLSize::new(32, 1, 1);
            let a_off = (a.offset + row0 * k * 4) as u64;
            let c_off = (c.offset + row0 * n * 4) as u64;
            self.run(name, grid, group, |enc| {
                enc.set_buffer(0, Some(&a.buffer), a_off);
                enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
                enc.set_buffer(2, Some(&c.buffer), c_off);
                enc.set_bytes(3, 4, (&k32 as *const u32).cast());
                enc.set_bytes(4, 4, (&n32 as *const u32).cast());
            })
        };
        match m {
            7 | 8 => {
                pass(4, 0)?;
                pass(m - 4, 4)
            }
            _ => pass(m, 0),
        }
    }

    /// NVFP4 матвек с батчем строк активаций (окно 1 < m ≤ 8): 32 лейна
    /// делят блок 64 на 4 подблока по 8 лейнов (лейн — 1 байт = 2 значения
    /// j и j+8), веса 1× на m строк. Блок 36 Б: d[4] UE4M3 + qs[32].
    /// m 7/8 — два прохода. Требования: K%64 (блоки), A — скалярно.
    pub fn mv_nvfp4_mb(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        c: &GpuTensor,
    ) -> Result<(), MetalError> {
        let k = a.shape[1];
        let n = b.shape[0];
        let m = a.shape[0];
        if b.shape != [n, k] || c.shape != [m, n] {
            return Err(MetalError::Shape(format!(
                "mv_nvfp4_mb {:?}x{:?}->{:?}",
                a.shape, b.shape, c.shape
            )));
        }
        if b.dtype != GpuDtype::Nvfp4 {
            return Err(MetalError::Shape(format!(
                "mv_nvfp4_mb: dtype {:?} не NVFP4",
                b.dtype
            )));
        }
        if !k.is_multiple_of(64) {
            return Err(MetalError::Shape(format!(
                "mv_nvfp4_mb: K={k} не кратен 64 (блок NVFP4)"
            )));
        }
        let (k32, n32) = (k as u32, n as u32);
        let pass = |rows: usize, row0: usize| -> Result<(), MetalError> {
            let (name, nr) = match rows {
                2 => ("mv_nvfp4_mb2", 4),
                3 => ("mv_nvfp4_mb3", 4),
                4 => ("mv_nvfp4_mb4", 4),
                5 => ("mv_nvfp4_mb5", 8),
                _ => ("mv_nvfp4_mb6", 8),
            };
            let grid = MTLSize::new((n.div_ceil(nr * 2) * 64) as u64, 1, 1);
            let group = MTLSize::new(64, 1, 1);
            let a_off = (a.offset + row0 * k * 4) as u64;
            let c_off = (c.offset + row0 * n * 4) as u64;
            self.run(name, grid, group, |enc| {
                enc.set_buffer(0, Some(&a.buffer), a_off);
                enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
                enc.set_buffer(2, Some(&c.buffer), c_off);
                enc.set_bytes(3, 4, (&k32 as *const u32).cast());
                enc.set_bytes(4, 4, (&n32 as *const u32).cast());
            })
        };
        match m {
            7 | 8 => {
                pass(4, 0)?;
                pass(m - 4, 4)
            }
            _ => pass(m, 0),
        }
    }

    /// IQ4_NL / MXFP4 матвек с батчем строк активаций (окно 1 < m ≤ 8):
    /// веса читаются 1× на m строк. Раскладка нибблов как у Q4_0; блок
    /// 18 Б (IQ4_NL: f16 d + 16 Б, d·KV_IQ4) или 17 Б (MXFP4: E8M0 + 16 Б,
    /// KV_FP4·2^(e−128)). m 7/8 — два прохода. Требования: K%32, A 16-выровнен.
    pub fn mv_iq4_nl_mxfp4_mb(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        c: &GpuTensor,
    ) -> Result<(), MetalError> {
        let k = a.shape[1];
        let n = b.shape[0];
        let m = a.shape[0];
        if b.shape != [n, k] || c.shape != [m, n] {
            return Err(MetalError::Shape(format!(
                "mv_iq4_nl_mxfp4_mb {:?}x{:?}->{:?}",
                a.shape, b.shape, c.shape
            )));
        }
        let stem = match b.dtype {
            GpuDtype::Iq4Nl => "mv_iq4_nl_mb",
            GpuDtype::Mxfp4 => "mv_mxfp4_mb",
            other => {
                return Err(MetalError::Shape(format!(
                    "mv_iq4_nl_mxfp4_mb: dtype {other:?} не IQ4_NL/MXFP4"
                )));
            }
        };
        if !k.is_multiple_of(32) {
            return Err(MetalError::Shape(format!(
                "mv_iq4_nl_mxfp4_mb: K={k} не кратен 32"
            )));
        }
        if !a.offset.is_multiple_of(16) {
            return Err(MetalError::Shape(format!(
                "mv_iq4_nl_mxfp4_mb: A.offset={} не 16-выровнен (float2-загрузки)",
                a.offset
            )));
        }
        let (k32, n32) = (k as u32, n as u32);
        let pass = |rows: usize, row0: usize| -> Result<(), MetalError> {
            let (suffix, nr) = match rows {
                2 => ("2", 4usize),
                3 => ("3", 4),
                4 => ("4", 4),
                5 => ("5", 8),
                _ => ("6", 8),
            };
            let name: &'static str = match (stem, suffix) {
                ("mv_iq4_nl_mb", "2") => "mv_iq4_nl_mb2",
                ("mv_iq4_nl_mb", "3") => "mv_iq4_nl_mb3",
                ("mv_iq4_nl_mb", "4") => "mv_iq4_nl_mb4",
                ("mv_iq4_nl_mb", "5") => "mv_iq4_nl_mb5",
                ("mv_iq4_nl_mb", _) => "mv_iq4_nl_mb6",
                ("mv_mxfp4_mb", "2") => "mv_mxfp4_mb2",
                ("mv_mxfp4_mb", "3") => "mv_mxfp4_mb3",
                ("mv_mxfp4_mb", "4") => "mv_mxfp4_mb4",
                ("mv_mxfp4_mb", "5") => "mv_mxfp4_mb5",
                _ => "mv_mxfp4_mb6",
            };
            let grid = MTLSize::new((n.div_ceil(nr * 2) * 64) as u64, 1, 1);
            let group = MTLSize::new(64, 1, 1);
            let a_off = (a.offset + row0 * k * 4) as u64;
            let c_off = (c.offset + row0 * n * 4) as u64;
            self.run(name, grid, group, |enc| {
                enc.set_buffer(0, Some(&a.buffer), a_off);
                enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
                enc.set_buffer(2, Some(&c.buffer), c_off);
                enc.set_bytes(3, 4, (&k32 as *const u32).cast());
                enc.set_bytes(4, 4, (&n32 as *const u32).cast());
            })
        };
        match m {
            7 | 8 => {
                pass(4, 0)?;
                pass(m - 4, 4)
            }
            _ => pass(m, 0),
        }
    }

    /// Q5_K матвек с батчем строк активаций (окно спек-верификации 1 < m ≤ 8).
    /// Кернел mv_q5_k_mb v2: 32 потока на строку весов, 16 лейнов делят
    /// суперблок, A читается float4 — веса 1× на m строк (v1 с суперблоком
    /// на лейн был ×3 к mm-тайлу; отклонён по замерам). m 7/8 — два прохода
    /// (4+остаток). Требования: K % 256, выравнивание A по 4 (позиции кратны
    /// 8·(lane%4) при базе, кратной 256 → всегда 4-выровнены).
    pub fn mv_q5_k_mb(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        c: &GpuTensor,
    ) -> Result<(), MetalError> {
        let k = a.shape[1];
        let n = b.shape[0];
        let m = a.shape[0];
        if a.shape != [m, k] || b.shape != [n, k] || c.shape != [m, n] {
            return Err(MetalError::Shape(format!(
                "mv_q5_k_mb {:?}x{:?}->{:?}",
                a.shape, b.shape, c.shape
            )));
        }
        // Тихая вера предусловиям недопустима: K без остатка на 256 (иначе
        // хвост молча не считается), чужой формат и невыровненный A
        // (float4-загрузки в кернеле) — явные ошибки.
        if b.dtype != GpuDtype::Q5_K {
            return Err(MetalError::Shape(format!(
                "mv_q5_k_mb: dtype {:?} не Q5_K",
                b.dtype
            )));
        }
        if !k.is_multiple_of(256) {
            return Err(MetalError::Shape(format!(
                "mv_q5_k_mb: K={k} не кратен 256 (хвост суперблока не обрабатывается)"
            )));
        }
        if !a.offset.is_multiple_of(16) {
            return Err(MetalError::Shape(format!(
                "mv_q5_k_mb: A.offset={} не 16-выровнен (float4-загрузки)",
                a.offset
            )));
        }
        let (k32, n32) = (k as u32, n as u32);
        let pass = |rows: usize, row0: usize| -> Result<(), MetalError> {
            let name = match rows {
                2 => "mv_q5_k_mb2",
                3 => "mv_q5_k_mb3",
                4 => "mv_q5_k_mb4",
                5 => "mv_q5_k_mb5",
                _ => "mv_q5_k_mb6",
            };
            // grid = ПОТОКИ (dispatch_threads): 32 потока на строку весов.
            let grid = MTLSize::new(32, n as u64, 1);
            let group = MTLSize::new(32, 1, 1);
            let a_off = (a.offset + row0 * k * 4) as u64;
            let c_off = (c.offset + row0 * n * 4) as u64;
            self.run(name, grid, group, |enc| {
                enc.set_buffer(0, Some(&a.buffer), a_off);
                enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
                enc.set_buffer(2, Some(&c.buffer), c_off);
                enc.set_bytes(3, 4, (&k32 as *const u32).cast());
                enc.set_bytes(4, 4, (&n32 as *const u32).cast());
            })
        };
        match m {
            7 | 8 => {
                pass(4, 0)?;
                pass(m - 4, 4)
            }
            2..=6 => pass(m, 0),
            other => Err(MetalError::Shape(format!(
                "mv_q5_k_mb: m={other} вне 2..=8"
            ))),
        }
    }

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
            GpuDtype::Q5_K => "mv_q5_k",
            GpuDtype::Q5_0 => "mv_q5_0",
            // IQ4_XS: один кернел и на декод, и на префилл (сетка (N, M)) —
            // тайлового GEMM для i-квантов пока нет, но копий и лишних
            // проходов тоже: см. mv_iq4_xs.
            GpuDtype::Iq4Xs => {
                return self.mv_iq4_xs_dispatch(a, b, c, k, n);
            }
            // IQ4_NL/MXFP4: один шаблонный кернел (host_name по формату),
            // последовательное накопление по 32-эл. блокам.
            GpuDtype::Iq4Nl => "mv_iq4_nl",
            GpuDtype::Mxfp4 => "mv_mxfp4",
            GpuDtype::Nvfp4 => "mv_nvfp4",
            GpuDtype::Q2_K => "mv_q2_k",
            GpuDtype::Q3_K => "mv_q3_k",
            GpuDtype::Q4_1 => "mv_q4_1",
            GpuDtype::Q5_1 => "mv_q5_1",
            GpuDtype::Q8_1 => "mv_q8_1",
            GpuDtype::Q8_K => "mv_q8_k",
            GpuDtype::Q1_0 => "mv_q1_0",
            GpuDtype::Q2_0 => "mv_q2_0",
            GpuDtype::TQ1_0 => "mv_tq1_0",
            GpuDtype::TQ2_0 => "mv_tq2_0",
            GpuDtype::Iq2Xxs => "mv_iq2_xxs",
            GpuDtype::Iq3Xxs => "mv_iq3_xxs",
            GpuDtype::Iq2Xs => "mv_iq2_xs",
            GpuDtype::Iq2S => "mv_iq2_s",
            GpuDtype::Iq3S => "mv_iq3_s",
            GpuDtype::Iq1S => "mv_iq1_s",
            GpuDtype::Iq1M => "mv_iq1_m",
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
        // Q4_K matvec: порт эталонного mul_mv (32 лейна на строку, пара строк
        // на simdgroup, y один раз на пару). Чтения побайтовые — выравнивание
        // не требуется. Порядок суммирования отличается от скалярного
        // dot_q4_k — класс MV_KS, приёмка парными text_parity на Q4_K-моделях
        // (GLM-4-9B, OLMoE); фоллбэк INFER_Q4K_NO_R2.
        if b.dtype == GpuDtype::Q4_K && std::env::var("INFER_Q4K_NO_R2").is_err() {
            let (k32, n32) = (k as u32, n as u32);
            let grid = MTLSize::new(32, n.div_ceil(2) as u64, 1);
            let group = MTLSize::new(32, 1, 1);
            return self.run("mv_q4_k_r2", grid, group, |enc| {
                enc.set_buffer(0, Some(&a.buffer), a.offset as u64);
                enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
                enc.set_buffer(2, Some(&c.buffer), c.offset as u64);
                enc.set_bytes(3, 4, (&k32 as *const u32).cast());
                enc.set_bytes(4, 4, (&n32 as *const u32).cast());
            });
        }
        // Q5_K matvec: тот же порт эталонного mul_mv (8 лейнов на строку,
        // пара строк на simdgroup, y один раз на пару). uint16-чтение scales
        // требует чётного смещения; фоллбэк INFER_Q5K_NO_R2.
        if b.dtype == GpuDtype::Q5_K
            && b.offset.is_multiple_of(2)
            && std::env::var("INFER_Q5K_NO_R2").is_err()
        {
            let (k32, n32) = (k as u32, n as u32);
            let grid = MTLSize::new(32, n.div_ceil(2) as u64, 1);
            let group = MTLSize::new(32, 1, 1);
            return self.run("mv_q5_k_r2", grid, group, |enc| {
                enc.set_buffer(0, Some(&a.buffer), a.offset as u64);
                enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
                enc.set_buffer(2, Some(&c.buffer), c.offset as u64);
                enc.set_bytes(3, 4, (&k32 as *const u32).cast());
                enc.set_bytes(4, 4, (&n32 as *const u32).cast());
            });
        }
        // Q6_K matvec: порт эталонного mul_mv (пара строк на simdgroup, y
        // один раз на пару, 16 лейнов на строку). Порядок суммирования по K
        // отличается от mv_q6_k — приёмка парными text_parity; фоллбэк
        // INFER_Q6K_NO_R2.
        if b.dtype == GpuDtype::Q6_K && std::env::var("INFER_Q6K_NO_R2").is_err() {
            let (k32, n32) = (k as u32, n as u32);
            let grid = MTLSize::new(32, n.div_ceil(2) as u64, 1);
            let group = MTLSize::new(32, 1, 1);
            return self.run("mv_q6_k_r2", grid, group, |enc| {
                enc.set_buffer(0, Some(&a.buffer), a.offset as u64);
                enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
                enc.set_buffer(2, Some(&c.buffer), c.offset as u64);
                enc.set_bytes(3, 4, (&k32 as *const u32).cast());
                enc.set_bytes(4, 4, (&n32 as *const u32).cast());
            });
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
        // Q5_0 (ffn_down в GLM-4-9B Q4_K_M — каждый третий слой): побитовый
        // порт оркального kernel_mul_mv_q5_0_f32 (mul_vec_q_n_f32_impl):
        // 4 строки на simdgroup, K между 32 лейнами (блок на 16, половина
        // блока на лейн), финал simd_sum. Скалярный mv_q5_0 суммирует по K
        // иначе и логи GLM расходятся на 1 ulp. Фоллбэк INFER_Q50_LEGACY.
        if b.dtype == GpuDtype::Q5_0 && std::env::var("INFER_Q50_LEGACY").is_err() {
            let (k32, n32) = (k as u32, n as u32);
            // 16 строк на threadgroup (4 sgitg × NR0=4).
            let grid = MTLSize::new((n.div_ceil(16) * 128) as u64, 1, 1);
            let group = MTLSize::new(128, 1, 1);
            return self.run("mv_q5_0_sg", grid, group, |enc| {
                enc.set_buffer(0, Some(&a.buffer), a.offset as u64);
                enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
                enc.set_buffer(2, Some(&c.buffer), c.offset as u64);
                enc.set_bytes(3, 4, (&k32 as *const u32).cast());
                enc.set_bytes(4, 4, (&n32 as *const u32).cast());
            });
        }
        // Q4_K/Q6_K: K-расщеплённый кернел (строка весов на simdgroup) —
        // «поток на строку» читал суперблоки последовательно и упирался в
        // латентность (MoE-эксперты: 17 ГБ/с против ~94 у llama.cpp).
        // K-расщеплённый путь — ОПТ-ИН (INFER_MV_KS=1), по замеру он быстрее на
        // «узких» матрицах: эксперты MoE [1024×2048] 56 против 28 ГБ/с,
        // [4096×4096] 145 против 90; на широких проигрывает ([13696×4096]
        // 140 против 161, [151936×4096] 94 против 108) и там не берётся.
        //
        // Почему не по умолчанию: порядок суммирования по K другой, и на
        // чувствительных моделях это меняет исход близких логитов — у OLMoE
        // (64-экспертный роутер) text_parity расходился на 2 промптах из 3,
        // тогда как путь «поток на строку» даёт байт-в-байт. Паритет первичен,
        // скорость — рычагом (решение зафиксировано в плане A3.4).
        let ks_ok = n <= 4096 && std::env::var("INFER_MV_KS").is_ok();
        if ks_ok
            && let Some(kname) = match b.dtype {
                GpuDtype::Q4_K => Some("mv_ks_q4_k"),
                GpuDtype::Q6_K => Some("mv_ks_q6_k"),
                _ => None,
            } {
                let (k32, n32) = (k as u32, n as u32);
                let grid = MTLSize::new((n.div_ceil(4) * 128) as u64, 1, 1);
                let group = MTLSize::new(128, 1, 1);
                return self.run(kname, grid, group, |enc| {
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

    /// IQ4_XS-проекция: C[M,N] = A[M,K] · W[N,K]ᵀ — сетка (N, M), поток на
    /// (токен, строку весов). Один кернел на декод и префилл.
    /// IQ4_XS матвек с батчем строк активаций (окно m 2..8): квант-блок
    /// строки весов читается ОДИН раз на m строк, побитово равен mv
    /// (тест iq4_xs_matvec_batch_consistency). m = 7, 8 — два прохода
    /// (4 + остаток). grid = ПОТОКИ (run → dispatch_threads), не
    /// threadgroup'ы: 32 потока на строку весов = один simdgroup (lane делит
    /// суперблоки, simd_sum). «Убрать избыточный grid.x=32» — ошибка чтения
    /// (Ф8.2g): с grid.x=1 тридцать два потока размазывались по 32 РАЗНЫМ
    /// строкам (lane указывал на чужие) и mb1 врал на 21% — поймано тестом
    /// iq4_xs_matvec_batch_consistency. Требует K%256 (гейт у вызывающих);
    /// A читается скалярно, выравнивание не нужно.
    fn mv_iq4_xs_mb(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        c: &GpuTensor,
        k: usize,
        n: usize,
    ) -> Result<(), MetalError> {
        let (k32, n32) = (k as u32, n as u32);
        let pass = |rows: usize, row0: usize| -> Result<(), MetalError> {
            let name = match rows {
                1 => "mv_iq4_xs_mb1",
                2 => "mv_iq4_xs_mb2",
                3 => "mv_iq4_xs_mb3",
                4 => "mv_iq4_xs_mb4",
                5 => "mv_iq4_xs_mb5",
                _ => "mv_iq4_xs_mb6",
            };
            let grid = MTLSize::new(32, n as u64, 1);
            let group = MTLSize::new(32, 1, 1);
            let a_off = (a.offset + row0 * k * 4) as u64;
            let c_off = (c.offset + row0 * n * 4) as u64;
            self.run(name, grid, group, |enc| {
                enc.set_buffer(0, Some(&a.buffer), a_off);
                enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
                enc.set_buffer(2, Some(&c.buffer), c_off);
                enc.set_bytes(3, 4, (&k32 as *const u32).cast());
                enc.set_bytes(4, 4, (&n32 as *const u32).cast());
            })
        };
        let m = a.shape[0];
        match m {
            7 | 8 => {
                pass(4, 0)?;
                pass(m - 4, 4)
            }
            _ => pass(m, 0),
        }
    }

    fn mv_iq4_xs_dispatch(
        &self,
        a: &GpuTensor,
        b: &GpuTensor,
        c: &GpuTensor,
        k: usize,
        n: usize,
    ) -> Result<(), MetalError> {
        // M — из ФОРМЫ ВЫХОДА C (матvec: [N] → 1 строка; GEMM: [M,N]).
        // Раньше m = a.shape[0]: в matvec-пути A — 1-D вектор [K], и m
        // становилось равно K — сетка (N, K), кернел писал C[m·N+n] далеко
        // за концом буфера выхода: молчаливое OOB-поражение соседних
        // аллокаций (флейк q8_0_small_batch/embedding_quant в параллельном
        // прогоне тестов; поименовано по реестру аллокаций).
        let m = if c.shape.len() == 2 { c.shape[0] } else { 1 };
        if a.numel() != m * k || c.numel() != m * n {
            return Err(MetalError::Shape(format!(
                "mv_iq4_xs: A {:?} / C {:?} не согласованы с M={m}, K={k}, N={n}",
                a.shape, c.shape
            )));
        }
        let (k32, n32) = (k as u32, n as u32);
        // Декод (m = 1) — порт эталонного mul_mv: один simdgroup на ПАРУ
        // строк весов, y один раз на пару (float4), кванты u32-парами, две
        // цепи ILP. Побитово == mv_iq4_xs (тест iq4_xs_r2_matches_rowwise);
        // требует 16-байтового сдвига A и 4-байтового B. INFER_IQ4XS_NO_R2=1
        // — прежний путь (сверка).
        if m == 1
            && a.offset.is_multiple_of(16)
            && b.offset.is_multiple_of(4)
            && std::env::var("INFER_IQ4XS_NO_R2").is_err()
        {
            let grid = MTLSize::new(32, n.div_ceil(2) as u64, 1);
            let group = MTLSize::new(32, 1, 1);
            return self.run("mv_iq4_xs_r2", grid, group, |enc| {
                enc.set_buffer(0, Some(&a.buffer), a.offset as u64);
                enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
                enc.set_buffer(2, Some(&c.buffer), c.offset as u64);
                enc.set_bytes(3, 4, (&k32 as *const u32).cast());
                enc.set_bytes(4, 4, (&n32 as *const u32).cast());
            });
        }
        // Строку весов считает один simdgroup (32 лейна) — сетка (32, N, M).
        let grid = MTLSize::new(32, n as u64, m as u64);
        let group = MTLSize::new(32, 1, 1);
        self.run("mv_iq4_xs", grid, group, |enc| {
            enc.set_buffer(0, Some(&a.buffer), a.offset as u64);
            enc.set_buffer(1, Some(&b.buffer), b.offset as u64);
            enc.set_buffer(2, Some(&c.buffer), c.offset as u64);
            enc.set_bytes(3, 4, (&k32 as *const u32).cast());
            enc.set_bytes(4, 4, (&n32 as *const u32).cast());
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
        // Побитовый порт оркального launch (ggml_metal_op_norm): ne00_t = d/4
        // (float4) либо d; nth = 32, удваивать пока nth < ne00_t и < 1024,
        // затем min(1024), min(ceil(ne00_t/32)*32).
        let vec = d.is_multiple_of(4) && x.offset.is_multiple_of(16) && w.offset.is_multiple_of(16) && y.offset.is_multiple_of(16);
        let nt = if vec { d / 4 } else { d };
        let mut nth = 32usize;
        while nth < nt && nth < 1024 {
            nth *= 2;
        }
        let nth = nth.min(1024).min(nt.div_ceil(32) * 32);
        let grid = MTLSize::new((rows * nth) as u64, 1, 1);
        let group = MTLSize::new(nth as u64, 1, 1);
        let d32 = d as u32;
        self.run(
            if vec {
                "rmsnorm_row_f32"
            } else {
                "rmsnorm_row_scalar_f32"
            },
            grid,
            group,
            |enc| {
                enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
                enc.set_buffer(1, Some(&w.buffer), w.offset as u64);
                enc.set_buffer(2, Some(&y.buffer), y.offset as u64);
                enc.set_bytes(3, 4, (&d32 as *const u32).cast());
                enc.set_bytes(4, 4, (&eps as *const f32).cast());
            },
        )
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

    /// Аргмакс по строкам логитов [T, V] + logsumexp строки (см.
    /// [`Backend::argmax_rows`]). `factors` — множители штрафа [V] (1.0 вне
    /// истории) либо None. Возврат: (ids [T], lse [T]) на хосте.
    pub fn argmax_lse_rows_f32(
        &self,
        logits: &GpuTensor,
        factors: Option<&GpuTensor>,
        t: usize,
        v: usize,
    ) -> Result<(Vec<u32>, Vec<f32>), MetalError> {
        // id пишутся как float (V < 2^24 — точность f32 полная): отдельный
        // целочисленный тензор не заводим
        let ids = GpuTensor::zeros(&self.device, vec![t])?;
        let lse = GpuTensor::zeros(&self.device, vec![t])?;
        let (t32, v32) = (t as u32, v as u32);
        let has = factors.is_some() as u32;
        let group = 256u64;
        let grid = MTLSize::new(t as u64 * group, 1, 1);
        // Нет factors — буфер не читается, но обязан быть привязан
        let fallback = match factors {
            Some(_) => None,
            None => Some(logits.clone()),
        };
        let factors = factors.unwrap_or_else(|| fallback.as_ref().unwrap());
        self.run(
            "argmax_lse_rows_f32",
            grid,
            MTLSize::new(group, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&logits.buffer), logits.offset as u64);
                enc.set_buffer(1, Some(&factors.buffer), factors.offset as u64);
                enc.set_buffer(2, Some(&ids.buffer), ids.offset as u64);
                enc.set_buffer(3, Some(&lse.buffer), lse.offset as u64);
                enc.set_bytes(4, 4, (&t32 as *const u32).cast());
                enc.set_bytes(5, 4, (&v32 as *const u32).cast());
                enc.set_bytes(6, 4, (&has as *const u32).cast());
            },
        )?;
        let idv = ids.to_cpu_f32(self)?;
        let lsev = lse.to_cpu_f32(self)?;
        Ok((idv.into_iter().map(|x| x as u32).collect(), lsev))
    }

    /// Построчная копия fp32 со сдвигом/шагом (см. [`Backend::copy_rows_into`]).
    #[allow(clippy::too_many_arguments)]
    pub fn copy_rows_f32(
        &self,
        src: &GpuTensor,
        rows: usize,
        cols: usize,
        dst: &GpuTensor,
        dst_row0: usize,
        dst_col0: usize,
        dst_stride: usize,
    ) -> Result<(), MetalError> {
        let (rows32, cols32) = (rows as u32, cols as u32);
        let (r0_32, c0_32, st32) = (dst_row0 as u32, dst_col0 as u32, dst_stride as u32);
        let grid = MTLSize::new(cols as u64, rows as u64, 1);
        let group = MTLSize::new(32, 4, 1);
        self.run("copy_rows_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&src.buffer), src.offset as u64);
            enc.set_buffer(1, Some(&dst.buffer), dst.offset as u64);
            enc.set_bytes(2, 4, (&rows32 as *const u32).cast());
            enc.set_bytes(3, 4, (&cols32 as *const u32).cast());
            enc.set_bytes(4, 4, (&r0_32 as *const u32).cast());
            enc.set_bytes(5, 4, (&c0_32 as *const u32).cast());
            enc.set_bytes(6, 4, (&st32 as *const u32).cast());
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
            GpuDtype::Q5_K => 5,
            GpuDtype::Q5_0 => 6,
            GpuDtype::Iq4Xs => 7,
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
    /// pipeline() требует &'static str (кэш по имени); имена tuned-vec
    /// собираются из частей — интерним уникальные строки (набор конечен:
    /// 18 баз × 4 суффикса, утечка ограничена этим).
    fn static_kernel_name(name: String) -> &'static str {
        static CACHE: OnceLock<Mutex<std::collections::HashMap<String, &'static str>>> =
            OnceLock::new();
        let m = CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
        let mut g = m.lock().unwrap();
        if let Some(s) = g.get(&name) {
            return s;
        }
        let s: &'static str = Box::leak(name.clone().into_boxed_str());
        g.insert(name, s);
        s
    }

    /// Точная копия ggml_metal_tuning::fa_vec_pick для SKU M3 Max и f16 KV
    /// (строки таблицы ggml-metal-tuning.cpp: device M3_MAX, type F16).
    /// ne11 < 1024 → baseline; далее exact-бакет (ne11_b, ne01_b) → domain
    /// default (ne11_b = −1, decode/batch) → baseline. Бакеты:
    /// ne11 ∈ {1024, 4096, 16384}, ne01 ∈ {2, 3, 4, 5}.
    fn fa_vec_pick(dk: usize, dv: usize, ne11: usize, ne01: usize) -> (usize, usize) {
        let baseline_ne: usize = match (dk, dv) {
            (64, 64) => 2,
            (128, 128) => 1,
            (256, 256) => 1,
            _ => 4,
        };
        let ne11_b = if ne11 < 1024 {
            0
        } else if ne11 < 4096 {
            1
        } else if ne11 < 16384 {
            2
        } else {
            3
        };
        if ne11_b == 0 {
            return (1, baseline_ne);
        }
        let ne01_b = if ne01 < 2 {
            0
        } else if ne01 < 3 {
            1
        } else if ne01 < 4 {
            2
        } else if ne01 < 5 {
            3
        } else {
            4
        };
        let exact = match (dk as i32, dv as i32, ne11_b, ne01_b) {
            (128, 128, 1, 4) => Some((1, 1)),
            (128, 128, 2, 4) => Some((1, 1)),
            (64, 64, 2, 0) => Some((1, 4)),
            (64, 64, 1, 2) => Some((1, 2)),
            (64, 64, 2, 2) => Some((1, 4)),
            (64, 64, 3, 1) => Some((1, 4)),
            (64, 64, 3, 2) => Some((4, 4)),
            _ => None,
        };
        if let Some(q) = exact {
            return q;
        }
        let domain = if ne01_b == 0 { 0 } else { 1 };
        match (dk as i32, dv as i32, domain) {
            (128, 128, 1) => (2, 2),
            (64, 64, 1) => (2, 4),
            _ => (1, baseline_ne),
        }
    }

    /// Внимание по f16-кэшу (Ф8.2f). Требует f16 у всех трёх входов: fused
    /// `qkv_norm_rope_kv` пишет Q сразу в half, поэтому смешанных форматов
    /// на этом пути нет (не-fused ветки модель для f16-кэша не выбирает).
    /// Ф8.1c: побитов путь эталонного FA VEC (pad → vec → reduce) для f16 KV,
    /// hd ∈ {64,128}, ne01 < 20. Ф8.1g: tuned Q/NE по таблице M3 Max при
    /// ne11 ≥ 1024 + nsg ∈ {1,2,4} от ne11 (ggml-metal-ops.cpp).
    #[allow(clippy::too_many_arguments)]
    pub fn attn_fa_vec_f16(
        &self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        mask: Option<&GpuTensor>,
        y: &GpuTensor,
        n_heads: usize,
        kv_heads: usize,
        hd: usize,
        tq: usize,
        tkv: usize,
    ) -> Result<(), MetalError> {
        #[repr(C)]
        struct FaVecArgs {
            ne01: i32,
            ne02: i32,
            ne03: i32,
            nb01: u64,
            nb02: u64,
            nb03: u64,
            ne11: i32,
            ne_12_2: i32,
            ne_12_3: i32,
            ns10: i32,
            nb11: u64,
            nb12: u64,
            nb13: u64,
            ns20: i32,
            nb21: u64,
            nb22: u64,
            nb23: u64,
            ne31: i32,
            ne32: i32,
            ne33: i32,
            nb31: u64,
            nb32: u64,
            nb33: u64,
            ne1: i32,
            ne2: i32,
            ne3: i32,
            scale: f32,
        }
        #[repr(C)]
        struct FaPadArgs {
            ne11: i32,
            ne_12_2: i32,
            ne_12_3: i32,
            nb11: u64,
            nb12: u64,
            nb13: u64,
            nb21: u64,
            nb22: u64,
            nb23: u64,
            ne31: i32,
            ne32: i32,
            ne33: i32,
            nb31: u64,
            nb32: u64,
            nb33: u64,
        }

        let scale = 1.0f32 / (hd as f32).sqrt();
        let qd = n_heads * hd;
        let kd = kv_heads * hd;
        let nb11 = (kd * 2) as u64; // байт на ключ (наш layout: key-major)
        let nb12 = (hd * 2) as u64; // байт на kv-голову
        let nb31 = (tkv * 2) as u64; // байт на строку маски

        // маска f32 → f16 (0/-inf точны; оркал получает маску уже в half)
        let mask16 = match mask {
            Some(m) => {
                let n = tq * tkv;
                let mt = GpuTensor::zeros_f16(&self.device, vec![n])?;
                let n32 = n as u32;
                self.run(
                    "cvt_f32_f16",
                    MTLSize::new(n as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                    |enc| {
                        enc.set_buffer(0, Some(&m.buffer), m.offset as u64);
                        enc.set_buffer(1, Some(&mt.buffer), mt.offset as u64);
                        enc.set_bytes(2, 4, (&n32 as *const u32).cast());
                    },
                )?;
                Some(mt)
            }
            None => None,
        };
        let dummy = q;

        if std::env::var("INFER_FA_VEC_DEBUG").is_ok() {
            eprintln!(
                "fa_vec dbg tq={tq} tkv={tkv} nh={n_heads} nkv={kv_heads} hd={hd} \
                 nb01={} nb02={} nb11={nb11} nb12={nb12} nb31={nb31} kvpad={}",
                qd * 4,
                hd * 4,
                !tkv.is_multiple_of(32)
            );
            if let Some(m16) = &mask16 {
                let mv = m16.to_cpu_f16(self)?;
                for i in 0..tq.min(3) {
                    let row: Vec<f32> = mv[i * tkv..(i + 1) * tkv]
                        .iter()
                        .map(|x| x.to_f32())
                        .collect();
                    eprintln!("  mask16 r{i}: {row:?}");
                }
            }
            let qv = q.to_cpu_f32(self)?;
            let kv = k.to_cpu_f16(self)?;
            if tq > 1 && tkv > 1 {
                eprintln!("  q r0: {:?} r1: {:?}", &qv[0..4], &qv[qd..qd + 4]);
                eprintln!("  k r0: {:?} r1: {:?}", &kv[0..4], &kv[kd..kd + 4]);
            }
            if let Ok(path) = std::env::var("INFER_FA_VEC_DUMP_DEC") {
                // дамп входов ШАГА ДЕКОДА (tq==1), первый вызов
                if tq == 1 && !std::path::Path::new(&path).exists() {
                    use std::io::Write;
                    let mut f = std::fs::File::create(&path).unwrap();
                    let n = tq * n_heads * hd;
                    let kn = tkv * kv_heads * hd;
                    for x in &qv[..n] {
                        f.write_all(&x.to_le_bytes()).unwrap();
                    }
                    let kvv = v.to_cpu_f16(self)?;
                    for x in &kv[..kn] {
                        f.write_all(&x.to_f32().to_le_bytes()).unwrap();
                    }
                    for x in &kvv[..kn] {
                        f.write_all(&x.to_f32().to_le_bytes()).unwrap();
                    }
                    eprintln!("  fa_vec DEC dumped tkv={tkv}");
                }
            }
            if let Ok(path) = std::env::var("INFER_FA_VEC_DUMP") {
                // только слой 0 (первый вызов)
                if !std::path::Path::new(&path).exists() {
                    use std::io::Write;
                    let mut f = std::fs::File::create(&path).unwrap();
                    let n = tq * n_heads * hd;
                    f.write_all(qd.to_be_bytes().as_slice()).unwrap();
                    f.write_all(tq.to_be_bytes().as_slice()).unwrap();
                    for x in &qv[..n] {
                        f.write_all(&x.to_be_bytes()).unwrap();
                    }
                    let kn = tkv * kv_heads * hd;
                    for x in &kv[..kn] {
                        f.write_all(&x.to_f32().to_be_bytes()).unwrap();
                    }
                    let vv = v.to_cpu_f16(self)?;
                    for x in &vv[..kn] {
                        f.write_all(&x.to_f32().to_be_bytes()).unwrap();
                    }
                    eprintln!("  fa_vec dumped to {path}");
                }
            }
        }

        // pad-буфер хвостового чанка (K, V, маска) — при ne11 % 32 != 0
        let has_kvpad = !tkv.is_multiple_of(32);
        let pad = if has_kvpad {
            let bytes =
                nb11 * 32 * kv_heads as u64 + nb11 * 32 * kv_heads as u64 + 32 * tq as u64 * 2;
            Some(GpuTensor::zeros(
                &self.device,
                vec![bytes.div_ceil(4) as usize],
            )?)
        } else {
            None
        };

        if has_kvpad {
            let p = pad.as_ref().unwrap();
            let a = FaPadArgs {
                ne11: tkv as i32,
                ne_12_2: kv_heads as i32,
                ne_12_3: 1,
                nb11,
                nb12,
                nb13: nb12,
                nb21: nb11,
                nb22: nb12,
                nb23: nb12,
                ne31: tq as i32,
                ne32: 1,
                ne33: 1,
                nb31,
                nb32: (tq * tkv * 2) as u64,
                nb33: (tq * tkv * 2) as u64,
            };
            // threadgroups (32, nkv, 1) × (32,1,1)
            self.run(
                "fa_vec_pad",
                MTLSize::new(32 * 32, kv_heads as u64, 1),
                MTLSize::new(32, 1, 1),
                |enc| {
                    enc.set_bytes(
                        0,
                        std::mem::size_of::<FaPadArgs>() as u64,
                        (&a as *const FaPadArgs).cast(),
                    );
                    enc.set_buffer(1, Some(&k.buffer), k.offset as u64);
                    enc.set_buffer(2, Some(&v.buffer), v.offset as u64);
                    enc.set_buffer(
                        3,
                        Some(&mask16.as_ref().unwrap_or(dummy).buffer),
                        mask16.as_ref().unwrap_or(dummy).offset as u64,
                    );
                    enc.set_buffer(4, Some(&p.buffer), p.offset as u64);
                },
            )?;
        }

        // tmp: nwg частичных результатов + S/M в хвосте
        let nrows = tq * n_heads;
        let tmp = GpuTensor::zeros(&self.device, vec![nrows * (hd * 32 + 64)])?;

        let a = FaVecArgs {
            ne01: tq as i32,
            ne02: n_heads as i32,
            ne03: 1,
            nb01: (qd * 4) as u64,
            nb02: (hd * 4) as u64,
            nb03: (tq * qd * 4) as u64,
            ne11: tkv as i32,
            ne_12_2: kv_heads as i32,
            ne_12_3: 1,
            // ns10 = nb11/nb10: элементы на ключ (у оракула function constant;
            // в его раскладке K [D,T,H] совпадает с DK, в нашей interleaved —
            // kv_heads*hd). Кернел шагает ключи этим stride'ом.
            ns10: (nb11 / 2) as i32,
            nb11,
            nb12,
            nb13: nb12,
            ns20: (nb11 / 2) as i32,
            nb21: nb11,
            nb22: nb12,
            nb23: nb12,
            ne31: tq as i32,
            ne32: 1,
            ne33: 1,
            nb31,
            nb32: (tq * tkv * 2) as u64,
            nb33: (tq * tkv * 2) as u64,
            ne1: n_heads as i32,
            ne2: tq as i32,
            ne3: 1,
            scale,
        };
        // Ф8.1g: tuned Q/NE (таблица M3 Max) + nsg от ne11 — ровно как оркал:
        // nsg = 1 при ne11 < 2048, 2 при < 4096, иначе 4; grid.x = ⌈ne01/Q⌉
        // threadgroup'ов по 32*nsg потоков.
        let (q_q, q_ne) = Self::fa_vec_pick(hd, hd, tkv, tq);
        let nsg = if tkv < 2048 {
            1
        } else if tkv < 4096 {
            2
        } else {
            4
        };
        let base = match (hd, q_q, q_ne, nsg) {
            (128, 1, 1, 1) => "fa_vec_dk128",
            (128, 1, 1, 2) => "fa_vec_dk128_q1ne1_nsg2",
            (128, 1, 1, 4) => "fa_vec_dk128_q1ne1_nsg4",
            (128, 2, 2, 1) => "fa_vec_dk128_q2ne2",
            (128, 2, 2, 2) => "fa_vec_dk128_q2ne2_nsg2",
            (128, 2, 2, 4) => "fa_vec_dk128_q2ne2_nsg4",
            (64, 1, 2, 1) => "fa_vec_dk64",
            (64, 1, 2, 2) => "fa_vec_dk64_q1ne2_nsg2",
            (64, 1, 2, 4) => "fa_vec_dk64_q1ne2_nsg4",
            (64, 1, 4, 1) => "fa_vec_dk64_q1ne4",
            (64, 1, 4, 2) => "fa_vec_dk64_q1ne4_nsg2",
            (64, 1, 4, 4) => "fa_vec_dk64_q1ne4_nsg4",
            (64, 2, 4, 1) => "fa_vec_dk64_q2ne4",
            (64, 2, 4, 2) => "fa_vec_dk64_q2ne4_nsg2",
            (64, 2, 4, 4) => "fa_vec_dk64_q2ne4_nsg4",
            (64, 4, 4, 1) => "fa_vec_dk64_q4ne4",
            (64, 4, 4, 2) => "fa_vec_dk64_q4ne4_nsg2",
            (64, 4, 4, 4) => "fa_vec_dk64_q4ne4_nsg4",
            _ => {
                return Err(MetalError::Shape(format!(
                    "fa_vec: нет инстанциации Q={q_q} NE={q_ne} nsg={nsg} hd={hd}"
                )));
            }
        };
        let name = Self::static_kernel_name(format!(
            "{}{}{}",
            base,
            if mask.is_some() { "_mask" } else { "" },
            if has_kvpad { "_pad" } else { "" }
        ));
        let smem = ((hd.max(128) + 4 * 32 + 2 * hd.max(128)) * nsg * q_q * 2).div_ceil(16) * 16;
        if std::env::var("INFER_FA_VEC_ARGS").is_ok() {
            eprintln!(
                "fa_vec ARGS name={name} smem={smem} grid=({},{},{}) q.off={} k.off={} v.off={} \
                nb01={} nb02={} nb11={} nb12={} ns10={} nb31={} ne31={} scale={:08x}",
                (tq.div_ceil(q_q) * 32 * nsg),
                n_heads,
                32,
                q.offset,
                k.offset,
                v.offset,
                a.nb01,
                a.nb02,
                a.nb11,
                a.nb12,
                a.ns10,
                a.nb31,
                a.ne31,
                a.scale.to_bits()
            );
        }
        self.run(
            name,
            MTLSize::new((tq.div_ceil(q_q) * 32 * nsg) as u64, n_heads as u64, 32),
            MTLSize::new((32 * nsg) as u64, 1, 1),
            |enc| {
                enc.set_bytes(
                    0,
                    std::mem::size_of::<FaVecArgs>() as u64,
                    (&a as *const FaVecArgs).cast(),
                );
                enc.set_buffer(1, Some(&q.buffer), q.offset as u64);
                enc.set_buffer(2, Some(&k.buffer), k.offset as u64);
                enc.set_buffer(3, Some(&v.buffer), v.offset as u64);
                enc.set_buffer(
                    4,
                    Some(&mask16.as_ref().unwrap_or(dummy).buffer),
                    mask16.as_ref().unwrap_or(dummy).offset as u64,
                );
                enc.set_buffer(5, Some(&q.buffer), q.offset as u64);
                enc.set_buffer(
                    6,
                    Some(&pad.as_ref().unwrap_or(dummy).buffer),
                    pad.as_ref().unwrap_or(dummy).offset as u64,
                );
                enc.set_buffer(7, Some(&tmp.buffer), tmp.offset as u64);
                enc.set_threadgroup_memory_length(0, smem as u64);
            },
        )?;

        let nrows_i = nrows as i32;
        let rname = if hd == 128 {
            "fa_vec_reduce_dv128"
        } else {
            "fa_vec_reduce_dv64"
        };
        self.run(
            rname,
            MTLSize::new((nrows * 1024) as u64, 1, 1),
            MTLSize::new(1024, 1, 1),
            |enc| {
                enc.set_bytes(0, 4, (&nrows_i as *const i32).cast());
                enc.set_buffer(1, Some(&tmp.buffer), tmp.offset as u64);
                enc.set_buffer(2, Some(&y.buffer), y.offset as u64);
            },
        )?;
        Ok(())
    }

    /// Ф8.1f: побитов путь эталонного FA TILE (kernel_flash_attn_ext_blk/pad
    /// и kernel_flash_attn_ext, Q=8, C=64, NSG=4, f16 KV, dense-маска).
    /// Оркал берёт его при ne01 >= 20. Требует маску (наши префиллы её имеют).
    #[allow(clippy::too_many_arguments)]
    pub fn attn_fa_tile_f16(
        &self,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        mask: Option<&GpuTensor>,
        y: &GpuTensor,
        n_heads: usize,
        kv_heads: usize,
        hd: usize,
        tq: usize,
        tkv: usize,
    ) -> Result<(), MetalError> {
        #[repr(C)]
        struct FaTileArgs {
            ne01: i32,
            ne02: i32,
            ne03: i32,
            nb01: u64,
            nb02: u64,
            nb03: u64,
            ne11: i32,
            ne_12_2: i32,
            ne_12_3: i32,
            ns10: i32,
            nb11: u64,
            nb12: u64,
            nb13: u64,
            ns20: i32,
            nb21: u64,
            nb22: u64,
            nb23: u64,
            ne31: i32,
            ne32: i32,
            ne33: i32,
            nb31: u64,
            nb32: u64,
            nb33: u64,
            ne1: i32,
            ne2: i32,
            ne3: i32,
            scale: f32,
        }
        #[repr(C)]
        struct FaPadArgs {
            ne11: i32,
            ne_12_2: i32,
            ne_12_3: i32,
            nb11: u64,
            nb12: u64,
            nb13: u64,
            nb21: u64,
            nb22: u64,
            nb23: u64,
            ne31: i32,
            ne32: i32,
            ne33: i32,
            nb31: u64,
            nb32: u64,
            nb33: u64,
        }
        #[repr(C)]
        struct FaBlkArgs {
            ne01: i32,
            ne30: i32,
            ne31: i32,
            ne32: i32,
            ne33: i32,
            nb31: u64,
            nb32: u64,
            nb33: u64,
        }

        let scale = 1.0f32 / (hd as f32).sqrt();
        let qd = n_heads * hd;
        let kd = kv_heads * hd;
        let nb11 = (kd * 2) as u64;
        let nb12 = (hd * 2) as u64;
        let nb31 = (tkv * 2) as u64;
        if std::env::var("INFER_FA_TILE_DEBUG").is_ok() {
            eprintln!(
                "fa_tile ENTER tq={tq} tkv={tkv} nh={n_heads} nkv={kv_heads} hd={hd} mask={}",
                mask.is_some()
            );
        }
        if let Ok(path) = std::env::var("INFER_FA_TILE_DUMP")
            && !std::path::Path::new(&path).exists() {
                use std::io::Write;
                let qv = q.to_cpu_f32(self)?;
                let kv = k.to_cpu_f16(self)?;
                let vv = v.to_cpu_f16(self)?;
                let mut f = std::fs::File::create(&path).unwrap();
                let n = tq * n_heads * hd;
                let kn = tkv * kv_heads * hd;
                for x in &qv[..n] {
                    f.write_all(&x.to_be_bytes()).unwrap();
                }
                for x in &kv[..kn] {
                    f.write_all(&x.to_f32().to_be_bytes()).unwrap();
                }
                for x in &vv[..kn] {
                    f.write_all(&x.to_f32().to_be_bytes()).unwrap();
                }
                eprintln!("fa_tile dumped to {path}");
            }

        let mask16 = match mask {
            Some(m) => {
                let n = tq * tkv;
                let mt = GpuTensor::zeros_f16(&self.device, vec![n])?;
                let n32 = n as u32;
                self.run(
                    "cvt_f32_f16",
                    MTLSize::new(n as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                    |enc| {
                        enc.set_buffer(0, Some(&m.buffer), m.offset as u64);
                        enc.set_buffer(1, Some(&mt.buffer), mt.offset as u64);
                        enc.set_bytes(2, 4, (&n32 as *const u32).cast());
                    },
                )?;
                Some(mt)
            }
            None => None,
        };
        let dummy = q;

        // pad хвостового чанка C=64
        let has_kvpad = !tkv.is_multiple_of(64);
        let pad = if has_kvpad {
            let bytes = nb11 * 64 * kv_heads as u64 * 2 + 64 * tq as u64 * 2;
            Some(GpuTensor::zeros(
                &self.device,
                vec![bytes.div_ceil(4) as usize],
            )?)
        } else {
            None
        };
        if has_kvpad {
            let p = pad.as_ref().unwrap();
            let a = FaPadArgs {
                ne11: tkv as i32,
                ne_12_2: kv_heads as i32,
                ne_12_3: 1,
                nb11,
                nb12,
                nb13: nb12,
                nb21: nb11,
                nb22: nb12,
                nb23: nb12,
                ne31: tq as i32,
                ne32: 1,
                ne33: 1,
                nb31,
                nb32: (tq * tkv * 2) as u64,
                nb33: (tq * tkv * 2) as u64,
            };
            // threadgroups (64, nkv, 1) × (32,1,1)
            self.run(
                "fa_tile_pad",
                MTLSize::new(64 * 32, kv_heads as u64, 1),
                MTLSize::new(32, 1, 1),
                |enc| {
                    enc.set_bytes(
                        0,
                        std::mem::size_of::<FaPadArgs>() as u64,
                        (&a as *const FaPadArgs).cast(),
                    );
                    enc.set_buffer(1, Some(&k.buffer), k.offset as u64);
                    enc.set_buffer(2, Some(&v.buffer), v.offset as u64);
                    enc.set_buffer(
                        3,
                        Some(&mask16.as_ref().unwrap_or(dummy).buffer),
                        mask16.as_ref().unwrap_or(dummy).offset as u64,
                    );
                    enc.set_buffer(4, Some(&p.buffer), p.offset as u64);
                },
            )?;
        }

        // скан блоков маски (0 = весь -INF, 1 = проверять, 2 = все нули)
        let nblk1 = tq.div_ceil(8);
        let nblk0 = tkv.div_ceil(64);
        let blk = GpuTensor::zeros(&self.device, vec![(nblk1 * nblk0 * kv_heads).div_ceil(4)])?;
        let ba = FaBlkArgs {
            ne01: tq as i32,
            ne30: tkv as i32,
            ne31: tq as i32,
            ne32: 1,
            ne33: 1,
            nb31,
            nb32: (tq * tkv * 2) as u64,
            nb33: (tq * tkv * 2) as u64,
        };
        self.run(
            "fa_tile_blk",
            MTLSize::new((nblk0 * 32) as u64, nblk1 as u64, 1),
            MTLSize::new(32, 1, 1),
            |enc| {
                enc.set_bytes(
                    0,
                    std::mem::size_of::<FaBlkArgs>() as u64,
                    (&ba as *const FaBlkArgs).cast(),
                );
                enc.set_buffer(
                    1,
                    Some(&mask16.as_ref().unwrap_or(dummy).buffer),
                    mask16.as_ref().unwrap_or(dummy).offset as u64,
                );
                enc.set_buffer(2, Some(&blk.buffer), blk.offset as u64);
            },
        )?;

        let a = FaTileArgs {
            ne01: tq as i32,
            ne02: n_heads as i32,
            ne03: 1,
            nb01: (qd * 4) as u64,
            nb02: (hd * 4) as u64,
            nb03: (tq * qd * 4) as u64,
            ne11: tkv as i32,
            ne_12_2: kv_heads as i32,
            ne_12_3: 1,
            ns10: (nb11 / 2) as i32,
            nb11,
            nb12,
            nb13: nb12,
            ns20: (nb11 / 2) as i32,
            nb21: nb11,
            nb22: nb12,
            nb23: nb12,
            ne31: tq as i32,
            ne32: 1,
            ne33: 1,
            nb31,
            nb32: (tq * tkv * 2) as u64,
            nb33: (tq * tkv * 2) as u64,
            ne1: n_heads as i32,
            ne2: tq as i32,
            ne3: 1,
            scale,
        };
        let bcm = !tq.is_multiple_of(8);
        let name = match (hd, has_kvpad, bcm) {
            (128, true, true) => "fa_tile_dk128_mask_pad_bcm",
            (128, true, false) => "fa_tile_dk128_mask_pad",
            (128, false, true) => "fa_tile_dk128_mask_bcm",
            (128, false, false) => "fa_tile_dk128_mask",
            (64, true, true) => "fa_tile_dk64_mask_pad_bcm",
            (64, true, false) => "fa_tile_dk64_mask_pad",
            (64, false, true) => "fa_tile_dk64_mask_bcm",
            _ => "fa_tile_dk64_mask",
        };
        // FATTN_SMEM: PAD2(Q*(DK + 2*PAD2(DV,64) + 4*C) * 2, 16)
        let pv = hd.max(64).div_ceil(64) * 64;
        let smem = (8 * (hd + 2 * pv + 4 * 64) * 2).div_ceil(16) * 16;
        self.run(
            name,
            MTLSize::new((nblk1 * 128) as u64, n_heads as u64, 1),
            MTLSize::new(128, 1, 1),
            |enc| {
                enc.set_bytes(
                    0,
                    std::mem::size_of::<FaTileArgs>() as u64,
                    (&a as *const FaTileArgs).cast(),
                );
                enc.set_buffer(1, Some(&q.buffer), q.offset as u64);
                enc.set_buffer(2, Some(&k.buffer), k.offset as u64);
                enc.set_buffer(3, Some(&v.buffer), v.offset as u64);
                enc.set_buffer(
                    4,
                    Some(&mask16.as_ref().unwrap_or(dummy).buffer),
                    mask16.as_ref().unwrap_or(dummy).offset as u64,
                );
                enc.set_buffer(5, Some(&q.buffer), q.offset as u64);
                enc.set_buffer(
                    6,
                    Some(&pad.as_ref().unwrap_or(dummy).buffer),
                    pad.as_ref().unwrap_or(dummy).offset as u64,
                );
                enc.set_buffer(7, Some(&blk.buffer), blk.offset as u64);
                enc.set_buffer(8, Some(&y.buffer), y.offset as u64);
                enc.set_threadgroup_memory_length(0, smem as u64);
            },
        )?;
        Ok(())
    }

    fn attn_ar_f16(
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
        if v.dtype != GpuDtype::F16 {
            return Err(MetalError::Shape(format!(
                "attn_ar: V не f16 при f16-K: q {:?}, k {:?}, v {:?}",
                q.dtype, k.dtype, v.dtype
            )));
        }
        // Q — f32 во ВСЕХ кернелах f16-внимания (кроме mma-предфилла, где
        // половина нужна формату фрагментов). Так делает и llama.cpp: KV
        // хранится в half, а запрос не округляется — округление q теряло
        // ~11 бит и на близких логитах меняло выбранный токен через десятки
        // шагов генерации (OLMoE, 64-экспертный роутер — самый чувствительный).
        if q.dtype != GpuDtype::F32 {
            return Err(MetalError::Shape(format!(
                "attn_ar f16: Q обязан быть f32, получен {:?}",
                q.dtype
            )));
        }
        if !n_heads.is_multiple_of(kv_heads) || !hd.is_multiple_of(4) {
            return Err(MetalError::Shape(format!(
                "attn_ar f16: H={n_heads} не делится на KVH={kv_heads} или HD={hd} не кратен 4"
            )));
        }
        // Ф8.1c: побитово воспроизведённый эталонный FA VEC (fa.metal
        // kernel_flash_attn_ext_{pad,vec,vec_reduce}). Оркал берёт его при
        // ne01 < 20 && ne00 % 32 == 0 — без ограничения на ne11: при
        // ne11 < 1024 тюнинг fa_vec_pick даёт baseline Q=1, NE=baseline_ne
        // (128→1, 64→2), при ne11 ≥ 1024 — tuned Q/NE из таблицы SKU
        // (Ф8.1g, M3 Max) плюс nsg ∈ {1,2,4} от ne11.
        // Фоллбэк INFER_NO_FA_VEC=1.
        if b == 1
            && tq <= 19
            && (hd == 64 || hd == 128)
            && std::env::var("INFER_NO_FA_VEC").is_err()
        {
            return self.attn_fa_vec_f16(q, k, v, mask, y, n_heads, kv_heads, hd, tq, tkv);
        }
        // Ф8.1f: tile-путь (побитовый порт kernel_flash_attn_ext) — оркал
        // берёт его при ne01 >= 20. Требует маску (префиллы её имеют) и
        // hd ∈ {64,128}. Фоллбэк INFER_NO_FA_TILE=1 — прежний mma-путь.
        if b == 1
            && tq >= 20
            && mask.is_some()
            && (hd == 64 || hd == 128)
            && std::env::var("INFER_NO_FA_TILE").is_err()
        {
            return self.attn_fa_tile_f16(q, k, v, mask, y, n_heads, kv_heads, hd, tq, tkv);
        }
        // Предфилл: half×half матричные блоки (16 запросов/threadgroup).
        // tkv ≥ 8: хвостовой фрагмент ключей сдвигается к Tkv-8 (см. кернел).
        // HD до 256 (Qwen3.5): фрагментов головы вдвое больше (nf = HD/8).
        if b == 1 && tq >= 16 && tkv >= 8 && hd.is_multiple_of(8) && hd <= 256 {
            let (tq32, tkv32, h32, kvh32, hd32) = (
                tq as u32,
                tkv as u32,
                n_heads as u32,
                kv_heads as u32,
                hd as u32,
            );
            // S float 2×8×32 + P half 2×8×32 + диаг float 2×64 + 1/l 2×8
            // + Q half 16×HD (стейджинг с построчной защитой хвоста).
            let bytes: u64 =
                (2 * 8 * 32 * 4 + 2 * 8 * 32 * 2 + 2 * 64 * 4 + 2 * 8 * 4 + 16 * hd * 2) as u64;
            // Инстанциация по числу фрагментов головы (HD/8): 16 при HD ≤ 128,
            // 32 при HD = 256 — регистры при HD=128 не растут.
            // mma-предфилл работает half-фрагментами (как llama.cpp для
            // префилла) — здесь переводим q в f16 отдельным проходом.
            let n: usize = q.shape.iter().product();
            if !n.is_multiple_of(4) {
                return Err(MetalError::Shape(format!(
                    "attn_ar f16: число элементов Q ({n}) не кратно 4"
                )));
            }
            let qh = GpuTensor::zeros_f16(&self.device, q.shape.clone())?;
            let n4 = (n / 4) as u32;
            self.run(
                "cast_f32_f16",
                MTLSize::new((n / 4) as u64, 1, 1),
                MTLSize::new(64, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&q.buffer), q.offset as u64);
                    enc.set_buffer(1, Some(&qh.buffer), qh.offset as u64);
                    enc.set_bytes(2, 4, (&n4 as *const u32).cast());
                },
            )?;
            let q = &qh;
            let name = if hd <= 128 {
                "attn_prefill_mma_f16_nf16"
            } else {
                "attn_prefill_mma_f16_nf32"
            };
            let grid = MTLSize::new((tq.div_ceil(16) * 64) as u64, n_heads as u64, 1);
            let group = MTLSize::new(64, 1, 1);
            return self.run(name, grid, group, |enc| {
                enc.set_buffer(0, Some(&q.buffer), q.offset as u64);
                enc.set_buffer(1, Some(&k.buffer), k.offset as u64);
                enc.set_buffer(2, Some(&v.buffer), v.offset as u64);
                enc.set_buffer(
                    3,
                    mask.map(|m| &*m.buffer),
                    mask.as_ref().map(|m| m.offset as u64).unwrap_or(0),
                );
                enc.set_buffer(4, Some(&y.buffer), y.offset as u64);
                enc.set_bytes(5, 4, (&tq32 as *const u32).cast());
                enc.set_bytes(6, 4, (&tkv32 as *const u32).cast());
                enc.set_bytes(7, 4, (&h32 as *const u32).cast());
                enc.set_bytes(8, 4, (&kvh32 as *const u32).cast());
                enc.set_bytes(9, 4, (&hd32 as *const u32).cast());
                enc.set_threadgroup_memory_length(0, bytes);
            });
        }
        // Декод (один запрос, без маски): split-K, half4-чтения.
        if b == 1 && tq == 1 && mask.is_none() && hd <= 256 {
            let (tkv32, h32, kvh32, hd32) =
                (tkv as u32, n_heads as u32, kv_heads as u32, hd as u32);
            let nsg = split_attn_groups(tkv, hd);
            let nsg32 = nsg as u32;
            let bytes = (nsg * (2 + hd) * 4).div_ceil(16) * 16;
            let grid = MTLSize::new((n_heads * nsg * 32) as u64, 1, 1);
            let group = MTLSize::new((nsg * 32) as u64, 1, 1);
            return self.run("attn_decode_split_f16", grid, group, |enc| {
                enc.set_buffer(0, Some(&q.buffer), q.offset as u64);
                enc.set_buffer(1, Some(&k.buffer), k.offset as u64);
                enc.set_buffer(2, Some(&v.buffer), v.offset as u64);
                enc.set_buffer(3, Some(&y.buffer), y.offset as u64);
                enc.set_bytes(4, 4, (&tkv32 as *const u32).cast());
                enc.set_bytes(5, 4, (&h32 as *const u32).cast());
                enc.set_bytes(6, 4, (&kvh32 as *const u32).cast());
                enc.set_bytes(7, 4, (&hd32 as *const u32).cast());
                enc.set_bytes(8, 4, (&nsg32 as *const u32).cast());
                enc.set_threadgroup_memory_length(0, bytes as u64);
            });
        }
        // Малый батч запросов (окно верификации спека, блок драфтера):
        // split-K по ключам, threadgroup на пару (голова, запрос); маска —
        // та же, что у общего пути. INFER_ATTN_MB_OFF=1 — общий путь (сверка).
        // ВНИМАНИЕ: ни этот, ни общий f16-кернел НЕ побитово равны оркальному
        // kernel_flash_attn_ext_vec (его окно — ne01 < 20) — префилл-логи
        // расходятся на 1e-3; порт vec-кернела — отдельная задача.
        if b == 1
            && (2..=8).contains(&tq)
            && hd <= 256
            && std::env::var("INFER_ATTN_MB_OFF").is_err()
        {
            let (tkv32, h32, kvh32, hd32, tq32) = (
                tkv as u32,
                n_heads as u32,
                kv_heads as u32,
                hd as u32,
                tq as u32,
            );
            let has_mask = mask.is_some() as u32;
            let nsg = split_attn_groups(tkv, hd);
            let nsg32 = nsg as u32;
            let bytes = (nsg * (2 + hd) * 4).div_ceil(16) * 16;
            let grid = MTLSize::new((n_heads * nsg * 32) as u64, tq as u64, 1);
            let group = MTLSize::new((nsg * 32) as u64, 1, 1);
            return self.run("attn_split_f16_q1", grid, group, |enc| {
                enc.set_buffer(0, Some(&q.buffer), q.offset as u64);
                enc.set_buffer(1, Some(&k.buffer), k.offset as u64);
                enc.set_buffer(2, Some(&v.buffer), v.offset as u64);
                enc.set_buffer(3, Some(&y.buffer), y.offset as u64);
                enc.set_bytes(4, 4, (&tkv32 as *const u32).cast());
                enc.set_bytes(5, 4, (&h32 as *const u32).cast());
                enc.set_bytes(6, 4, (&kvh32 as *const u32).cast());
                enc.set_bytes(7, 4, (&hd32 as *const u32).cast());
                enc.set_bytes(8, 4, (&nsg32 as *const u32).cast());
                enc.set_bytes(9, 4, (&tq32 as *const u32).cast());
                match mask {
                    Some(m) => enc.set_buffer(10, Some(&m.buffer), m.offset as u64),
                    // буфер обязан быть привязан; без маски не читается
                    None => enc.set_buffer(10, Some(&q.buffer), q.offset as u64),
                }
                enc.set_bytes(11, 4, (&has_mask as *const u32).cast());
                enc.set_threadgroup_memory_length(0, bytes as u64);
            });
        }
        // Общий путь: любые Tq (в т.ч. хвостовой чанк), маска, hd до 512.
        let n_rep = n_heads / kv_heads;
        let grid = MTLSize::new(b as u64, (n_heads * 32) as u64, tq as u64);
        let group = MTLSize::new(1, 32, 1);
        self.run("attn_ar_f16_fused", grid, group, |enc| {
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
        })
    }

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
        // f16-KV (Ф8.2f): K/V (а с fused-путём и Q) — half. Диспетчер:
        // предфилл — матричные half-блоки, декод — split-K по half4, прочее
        // (хвосты чанков Tq<16, короткие промпты) — общий f16-кернел.
        if k.dtype == GpuDtype::F16 {
            return self.attn_ar_f16(q, k, v, mask, y, n_heads, kv_heads, hd);
        }
        let (b, tq, _d) = (q.shape[0], q.shape[1], q.shape[2]);
        let tkv = k.shape[1];
        let n_rep = n_heads / kv_heads;
        // Промпт (Tq ≥ 16, одна последовательность): flash-attention на
        // матричных блоках simdgroup — K/V фрагментами прямо из device-памяти,
        // 16 запросов одной головы на threadgroup. INFER_ATTN_LEGACY=1 —
        // прежний кернел для сверки.
        // Матричные блоки окупаются при hd ≤ 128 (1.5× на Qwen3); при hd=256
        // (Qwen3.5, 16 голов) фрагментов O вдвое больше, а threadgroup'ов
        // вдвое меньше — прежний кернел быстрее (замер: 0.44 против 0.34 мс).
        // Сплит-K (Ф8.2e) — ВЫКЛЮЧЕН по умолчанию: верен (покрыт тестом
        // prefill_attention_matches_reference), но медленнее прежнего пути
        // (43 против 28 мс на слой при Tq=512, контексте 4.6k). Причина:
        // частичные softmax весят столько же, сколько выход, то есть при
        // SPLIT=16 трафик частичных (запись+чтение) в 32 раза больше выхода
        // и перекрывает выигрыш от укорочения цепочки ключей. Задел на
        // случай, когда частичные можно будет держать в разделяемой памяти
        // (при SPLIT=4 и hd=128 это 4×(2+128)×8 запросов ≈ 16 КБ).
        // Включается INFER_ATTN_SPLIT=1.
        if b == 1
            && tq >= 32
            && tkv >= 8
            && hd.is_multiple_of(8)
            && hd <= 128
            && n_heads.is_multiple_of(kv_heads)
            && !attn_opt("legacy")
            && attn_opt("split")
        {
            let split = tkv.div_ceil(128).clamp(1, 16);
            if split >= 2 {
                let (tq32, tkv32, h32, kvh32, hd32, sp32) = (
                    tq as u32,
                    tkv as u32,
                    n_heads as u32,
                    kv_heads as u32,
                    hd as u32,
                    split as u32,
                );
                let part = GpuTensor::zeros(&self.device, vec![tq, n_heads, split, 2 + hd])?;
                let (pg, qg, kg, vg, yg) = (&part, q, k, v, y);
                // частичные: (запросы/16) × голов × splits, threadgroup 64
                self.run(
                    "attn_prefill_partial_f32",
                    MTLSize::new((tq.div_ceil(16) * 64) as u64, n_heads as u64, split as u64),
                    MTLSize::new(64, 1, 1),
                    |enc| {
                        enc.set_buffer(0, Some(&qg.buffer), qg.offset as u64);
                        enc.set_buffer(1, Some(&kg.buffer), kg.offset as u64);
                        enc.set_buffer(2, Some(&vg.buffer), vg.offset as u64);
                        enc.set_buffer(
                            3,
                            mask.map(|m| &*m.buffer),
                            mask.as_ref().map(|m| m.offset as u64).unwrap_or(0),
                        );
                        enc.set_buffer(4, Some(&pg.buffer), pg.offset as u64);
                        enc.set_bytes(5, 4, (&tq32 as *const u32).cast());
                        enc.set_bytes(6, 4, (&tkv32 as *const u32).cast());
                        enc.set_bytes(7, 4, (&h32 as *const u32).cast());
                        enc.set_bytes(8, 4, (&kvh32 as *const u32).cast());
                        enc.set_bytes(9, 4, (&hd32 as *const u32).cast());
                        enc.set_bytes(10, 4, (&sp32 as *const u32).cast());
                    },
                )?;
                // слияние: threadgroup на строку (запрос, голова)
                let (tqh32, hd32b) = ((tq * n_heads) as u32, hd as u32);
                self.run(
                    "attn_prefill_combine_f32",
                    MTLSize::new((tq * n_heads * 32) as u64, 1, 1),
                    MTLSize::new(32, 1, 1),
                    |enc| {
                        enc.set_buffer(0, Some(&pg.buffer), pg.offset as u64);
                        enc.set_buffer(1, Some(&yg.buffer), yg.offset as u64);
                        enc.set_bytes(2, 4, (&tqh32 as *const u32).cast());
                        enc.set_bytes(3, 4, (&hd32b as *const u32).cast());
                        enc.set_bytes(4, 4, (&sp32 as *const u32).cast());
                    },
                )?;
                return Ok(());
            }
        }
        // Полу-тайловый путь (Ф8.2e) — ВЫКЛЮЧЕН по умолчанию: он верен
        // (тесты prefill_attention_matches_reference и
        // tiled_attention_layout_probe), но на M3 медленнее варианта с
        // фрагментами из device-памяти: группа просит 16–31 КБ разделяемой
        // памяти, а на ядро её 32 КБ — в полёте вдвое-вчетверо меньше групп
        // и задержки не перекрываются (замер: 38 против 27 мс на слой при
        // Tq=512/контексте 4.6k, 0.36 против 0.25 мс при Tq=128). Включается
        // INFER_ATTN_TILES=1 — задел под конвейеризацию (двойная буферизация
        // тайлов и асинхронные копии), где запас есть.
        if b == 1
            && tq >= 32
            && tkv >= 8
            && hd.is_multiple_of(8)
            && hd <= 128
            && n_heads.is_multiple_of(kv_heads)
            && !attn_opt("legacy")
            && attn_opt("tiles")
        {
            let (tq32, tkv32, h32, kvh32, hd32) = (
                tq as u32,
                tkv as u32,
                n_heads as u32,
                kv_heads as u32,
                hd as u32,
            );
            // Q half 32×HD + K/V half 16×HD + S float 4×8×16 + P half 4×8×16
            // + диагонали 1 КБ + 1/l ≈ 16.1 КБ при hd=128 — две группы на ядро.
            let bytes = (32 * hd * 2
                + 2 * 16 * hd * 2
                + 4 * 8 * 16 * 4
                + 4 * 8 * 16 * 2
                + 4 * 64 * 4
                + 32 * 4) as u64;
            let grid = MTLSize::new((tq.div_ceil(32) * 128) as u64, n_heads as u64, 1);
            let group = MTLSize::new(128, 1, 1);
            return self.run("attn_prefill_fa4_f32", grid, group, |enc| {
                enc.set_buffer(0, Some(&q.buffer), q.offset as u64);
                enc.set_buffer(1, Some(&k.buffer), k.offset as u64);
                enc.set_buffer(2, Some(&v.buffer), v.offset as u64);
                enc.set_buffer(
                    3,
                    mask.map(|m| &*m.buffer),
                    mask.as_ref().map(|m| m.offset as u64).unwrap_or(0),
                );
                enc.set_buffer(4, Some(&y.buffer), y.offset as u64);
                enc.set_bytes(5, 4, (&tq32 as *const u32).cast());
                enc.set_bytes(6, 4, (&tkv32 as *const u32).cast());
                enc.set_bytes(7, 4, (&h32 as *const u32).cast());
                enc.set_bytes(8, 4, (&kvh32 as *const u32).cast());
                enc.set_bytes(9, 4, (&hd32 as *const u32).cast());
                enc.set_threadgroup_memory_length(0, bytes);
            });
        }
        // Матричные блоки окупаются при hd ≤ 128 (1.5× на Qwen3); при hd=256
        // (Qwen3.5, 16 голов) фрагментов O вдвое больше, а threadgroup'ов
        // вдвое меньше — прежний кернел быстрее (замер: 0.44 против 0.34 мс).
        if b == 1
            && tq >= 16
            && tkv >= 8
            && hd.is_multiple_of(8)
            && hd <= 128
            && n_heads.is_multiple_of(kv_heads)
            && !attn_opt("legacy")
        {
            let (tq32, tkv32, h32, kvh32, hd32) = (
                tq as u32,
                tkv as u32,
                n_heads as u32,
                kv_heads as u32,
                hd as u32,
            );
            // [16][HD] Q (он же выход) + [16][32] S + [16][8] диаг + [16] 1/l
            let floats = 16 * hd + 16 * 32 + 16 * 8 + 16;
            let bytes = (floats * 4).div_ceil(16) * 16;
            let grid = MTLSize::new((tq.div_ceil(16) * 64) as u64, n_heads as u64, 1);
            let group = MTLSize::new(64, 1, 1);
            return self.run("attn_prefill_mma_f32", grid, group, |enc| {
                enc.set_buffer(0, Some(&q.buffer), q.offset as u64);
                enc.set_buffer(1, Some(&k.buffer), k.offset as u64);
                enc.set_buffer(2, Some(&v.buffer), v.offset as u64);
                enc.set_buffer(
                    3,
                    mask.map(|m| &*m.buffer),
                    mask.as_ref().map(|m| m.offset as u64).unwrap_or(0),
                );
                enc.set_buffer(4, Some(&y.buffer), y.offset as u64);
                enc.set_bytes(5, 4, (&tq32 as *const u32).cast());
                enc.set_bytes(6, 4, (&tkv32 as *const u32).cast());
                enc.set_bytes(7, 4, (&h32 as *const u32).cast());
                enc.set_bytes(8, 4, (&kvh32 as *const u32).cast());
                enc.set_bytes(9, 4, (&hd32 as *const u32).cast());
                enc.set_threadgroup_memory_length(0, bytes as u64);
            });
        }
        // Decode (один запрос, без маски): голова = threadgroup из 8
        // simdgroup'ов, ключи делятся между ними (flash-decoding).
        if b == 1
            && tq == 1
            && mask.is_none()
            && hd.is_multiple_of(4)
            && hd <= 256
            && n_heads.is_multiple_of(kv_heads)
            && !attn_opt("legacy")
        {
            let (tkv32, h32, kvh32, hd32) =
                (tkv as u32, n_heads as u32, kv_heads as u32, hd as u32);
            // Разбиение ключей: на длинном контексте 8 групп оставляли всего
            // 32 threadgroup'а — каждый лейн тянул цепочку из сотен ключей
            // последовательно (декод на 4k контекста падал до 1.3 ток/с).
            // Число групп растёт с длиной KV (см. split_attn_groups).
            let nsg = split_attn_groups(tkv, hd);
            let nsg32 = nsg as u32;
            let bytes = (nsg * (2 + hd) * 4).div_ceil(16) * 16;
            let grid = MTLSize::new((n_heads * nsg * 32) as u64, 1, 1);
            let group = MTLSize::new((nsg * 32) as u64, 1, 1);
            return self.run("attn_decode_split_f32", grid, group, |enc| {
                enc.set_buffer(0, Some(&q.buffer), q.offset as u64);
                enc.set_buffer(1, Some(&k.buffer), k.offset as u64);
                enc.set_buffer(2, Some(&v.buffer), v.offset as u64);
                enc.set_buffer(3, Some(&y.buffer), y.offset as u64);
                enc.set_bytes(4, 4, (&tkv32 as *const u32).cast());
                enc.set_bytes(5, 4, (&h32 as *const u32).cast());
                enc.set_bytes(6, 4, (&kvh32 as *const u32).cast());
                enc.set_bytes(7, 4, (&hd32 as *const u32).cast());
                enc.set_bytes(8, 4, (&nsg32 as *const u32).cast());
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

    /// Fused GDN-хвост decode-шага (t=1): conv+silu+split+l2norm+сдвиг стейта
    /// одним кернелом. Одна threadgroup из `hd` потоков на голову; grid —
    /// ВСЕГО потоков (heads·hd), см. ловушку dispatch_threads.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_conv_silu_split_norm_f32(
        &self,
        x: &GpuTensor,
        w: &GpuTensor,
        state: &GpuTensor,
        y: &GpuTensor,
        new_state: &GpuTensor,
        c: usize,
        k: usize,
        hd: usize,
        qk: usize,
        eps: f32,
    ) -> Result<(), MetalError> {
        let (c32, k32, hd32, qk32) = (c as u32, k as u32, hd as u32, qk as u32);
        let heads = c.div_ceil(hd);
        self.run(
            "gdn_conv_silu_split_norm_f32",
            MTLSize::new((heads * hd) as u64, 1, 1),
            MTLSize::new(hd as u64, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
                enc.set_buffer(1, Some(&w.buffer), w.offset as u64);
                enc.set_buffer(2, Some(&state.buffer), state.offset as u64);
                enc.set_buffer(3, Some(&y.buffer), y.offset as u64);
                enc.set_buffer(4, Some(&new_state.buffer), new_state.offset as u64);
                enc.set_bytes(5, 4, (&c32 as *const u32).cast());
                enc.set_bytes(6, 4, (&k32 as *const u32).cast());
                enc.set_bytes(7, 4, (&hd32 as *const u32).cast());
                enc.set_bytes(8, 4, (&qk32 as *const u32).cast());
                enc.set_bytes(9, 4, (&eps as *const f32).cast());
            },
        )
    }

    /// Гейты GDN: g = softplus(alpha + dt_bias)·a, beta = sigmoid(beta_raw).
    /// alpha/beta/g/b [n = T·H], dt_bias/a [H].
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_gates_f32(
        &self,
        ba: &GpuTensor,
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
            MTLSize::new(64.min(n.max(1)) as u64, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&ba.buffer), ba.offset as u64);
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
        // S = 128: блок токенов в threadgroup-память — k/q/v блока читаются
        // один раз на threadgroup (а не на каждый токен каждой группой строк).
        // INFER_GDN_LEGACY=1 — прежний путь для сверки.
        let (kernel, grid, group) = if s == 128 && std::env::var("INFER_GDN_LEGACY").is_err() {
            (
                "gdn_delta_rule_chunk_f32",
                MTLSize::new((s / 16 * 32) as u64, (hv * 16) as u64, 1),
                MTLSize::new(32, 16, 1),
            )
        } else if s.is_multiple_of(32) && s <= 256 && (hv * s).is_multiple_of(4) {
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
        self.run(kernel, grid, group, |enc| {
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
        })
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
        // Σx² по всем лейнам (simd_sum) вместо последовательного накопления
        // одним: на декоде 27B (rows=1, HD=6144) 0.16 мс/слой → ~0.01.
        // HD%128==0 — 4 simdgroup на строку (одной мало при rows=1).
        // Фоллбэк на прежний порядок — INFER_NG_SERIAL=1.
        let (kernel, nth) = if hd.is_multiple_of(128) && std::env::var("INFER_NG_SERIAL").is_err() {
            ("gdn_norm_gated_par4_f32", 128u64)
        } else if hd.is_multiple_of(32) && std::env::var("INFER_NG_SERIAL").is_err() {
            ("gdn_norm_gated_par_f32", 32u64)
        } else {
            ("gdn_norm_gated_f32", 32u64)
        };
        self.run(
            kernel,
            // dispatch_threads — ВСЕГО потоков: nth лейнов на строку.
            MTLSize::new((rows as u64) * nth, 1, 1),
            MTLSize::new(nth, 1, 1),
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
        // f16-хранилище — свой кернел (квантизация на записи); f32→f32 и
        // f16→f16 (рост кэша копией) обслуживаются дальше по типу.
        if dst.dtype == GpuDtype::F16 && src.dtype == GpuDtype::F32 {
            let (b, tq, d) = (src.shape[0], src.shape[1], src.shape[2]);
            let cap = dst.shape[1];
            let grid = MTLSize::new(b as u64, tq as u64, d as u64);
            let group = MTLSize::new(1, 4, 4);
            return self.run("kv_update_f16_f32", grid, group, |enc| {
                enc.set_buffer(0, Some(&dst.buffer), dst.offset as u64);
                enc.set_buffer(1, Some(&src.buffer), src.offset as u64);
                for (i, v) in [b, tq, pos, d, cap].iter().enumerate() {
                    enc.set_bytes((2 + i) as u64, 8, (v as *const usize).cast());
                }
            });
        }
        if dst.dtype == GpuDtype::F16 && src.dtype == GpuDtype::F16 {
            let (b, tq, d) = (src.shape[0], src.shape[1], src.shape[2]);
            let cap = dst.shape[1];
            let grid = MTLSize::new(b as u64, tq as u64, d as u64);
            let group = MTLSize::new(1, 4, 4);
            return self.run("kv_copy_f16", grid, group, |enc| {
                enc.set_buffer(0, Some(&dst.buffer), dst.offset as u64);
                enc.set_buffer(1, Some(&src.buffer), src.offset as u64);
                for (i, v) in [b, tq, pos, d, cap].iter().enumerate() {
                    enc.set_bytes((2 + i) as u64, 8, (v as *const usize).cast());
                }
            });
        }
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
        let vec = hd.is_multiple_of(4) && x.offset.is_multiple_of(16) && w.offset.is_multiple_of(16) && y.offset.is_multiple_of(16);
        self.run(
            if vec {
                "rmsnorm_head_sg_f32"
            } else {
                "rmsnorm_head_scalar_f32"
            },
            grid,
            group,
            |enc| {
                enc.set_buffer(0, Some(&x.buffer), x.offset as u64);
                enc.set_buffer(1, Some(&w.buffer), w.offset as u64);
                enc.set_buffer(2, Some(&y.buffer), y.offset as u64);
                enc.set_bytes(3, 4, (&nh as *const u32).cast());
                enc.set_bytes(4, 4, (&hd32 as *const u32).cast());
                enc.set_bytes(5, 4, (&eps as *const f32).cast());
            },
        )
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
        yarn: Option<&crate::rope::RopeYarn>,
    ) -> Result<crate::backend::Tensor, crate::error::BackendError> {
        let (tq, d) = (x.shape()[1], x.shape()[2]);
        let y = self.zeros(vec![1, tq, d])?;
        let xg = x.inner_metal();
        let yg = y.inner_metal();
        let yarn = yarn.copied().unwrap_or(crate::rope::RopeYarn::IDENTITY);
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
            enc.set_bytes(7, 24, (&yarn as *const crate::rope::RopeYarn).cast());
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
        yarn: Option<&crate::rope::RopeYarn>,
    ) -> Result<crate::backend::Tensor, crate::error::BackendError> {
        let (tq, d) = (x.shape()[1], x.shape()[2]);
        let y = self.zeros(vec![1, tq, d])?;
        let xg = x.inner_metal();
        let wg = w.inner_metal();
        let yg = y.inner_metal();
        let yarn = yarn.copied().unwrap_or(crate::rope::RopeYarn::IDENTITY);
        // simdgroup на (токен, голову) — см. rmsnorm_head_rope_sg_f32.
        let (h32, nh32, hd32, pos32) = (h as u32, (tq * h) as u32, hd as u32, pos as u32);
        let grid = metal::MTLSize::new((tq * h * 32) as u64, 1, 1);
        let group = metal::MTLSize::new(64, 1, 1);
        let vec = hd.is_multiple_of(4) && xg.offset.is_multiple_of(16);
        self.ctx.run(
            if vec {
                "rmsnorm_head_rope_sg_f32"
            } else {
                "rmsnorm_head_rope_scalar_f32"
            },
            grid,
            group,
            |enc| {
                enc.set_buffer(0, Some(&xg.buffer), xg.offset as u64);
                enc.set_buffer(1, Some(&wg.buffer), wg.offset as u64);
                enc.set_buffer(2, Some(&yg.buffer), yg.offset as u64);
                enc.set_bytes(3, 4, (&h32 as *const u32).cast());
                enc.set_bytes(4, 4, (&nh32 as *const u32).cast());
                enc.set_bytes(5, 4, (&hd32 as *const u32).cast());
                enc.set_bytes(6, 4, (&eps as *const f32).cast());
                enc.set_bytes(7, 4, (&pos32 as *const u32).cast());
                enc.set_bytes(8, 4, (&theta as *const f32).cast());
                enc.set_bytes(9, 24, (&yarn as *const crate::rope::RopeYarn).cast());
            },
        )?;
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
        yarn: Option<&crate::rope::RopeYarn>,
    ) -> Result<crate::backend::Tensor, crate::error::BackendError> {
        let t = qkv.numel() / ((h + 2 * kvh) * hd);
        let yarn = yarn.copied().unwrap_or(crate::rope::RopeYarn::IDENTITY);
        // f16-KV (Ф8.2f): K/V пишутся в half (вдвое меньше трафика внимания),
        // а Q остаётся f32 — как в llama.cpp: округление запроса до half
        // теряло точность, и текст расходился с эталоном через десятки
        // идентичных токенов (замер: OLMoE-1B-7B, промпт про Париж).
        if kbuf.dtype() == crate::dtype::Dtype::F16 {
            if vbuf.dtype() != crate::dtype::Dtype::F16 {
                return Err(crate::error::BackendError::Shape(
                    "qkv_norm_rope_kv: k-буфер f16, v-буфер не f16".into(),
                ));
            }
            let (kg, vg) = (kbuf.as_gpu(), vbuf.as_gpu());
            let qh = GpuTensor::zeros_f32(&self.ctx.device, vec![1, t, h * hd])?;
            let (xg, wq, wk) = (qkv.as_gpu(), q_norm.as_gpu(), k_norm.as_gpu());
            let (t32, h32, kvh32, hd32, rp32, pk32) = (
                t as u32,
                h as u32,
                kvh as u32,
                hd as u32,
                rope_pos as u32,
                pos_kv as u32,
            );
            let grid = MTLSize::new((t * (h + 2 * kvh) * 32) as u64, 1, 1);
            self.ctx.run(
                "qkv_norm_rope_kv_f16_f32",
                grid,
                MTLSize::new(64, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&xg.buffer), xg.offset as u64);
                    enc.set_buffer(1, Some(&wq.buffer), wq.offset as u64);
                    enc.set_buffer(2, Some(&wk.buffer), wk.offset as u64);
                    enc.set_buffer(3, Some(&qh.buffer), qh.offset as u64);
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
                    enc.set_bytes(14, 24, (&yarn as *const crate::rope::RopeYarn).cast());
                },
            )?;
            return Ok(crate::backend::Tensor::from_gpu(qh));
        }
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
        let (t32, h32, kvh32, hd32, rp32, pk32) = (
            t as u32,
            h as u32,
            kvh as u32,
            hd as u32,
            rope_pos as u32,
            pos_kv as u32,
        );
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
            enc.set_bytes(14, 24, (&yarn as *const crate::rope::RopeYarn).cast());
        })?;
        Ok(q)
    }

    fn rope_partial(
        &self,
        x: &crate::backend::Tensor,
        pos: usize,
        hd: usize,
        n_rot: usize,
        theta: f32,
        pairing: crate::backend::RopePairing,
    ) -> Result<crate::backend::Tensor, crate::error::BackendError> {
        if pairing == crate::backend::RopePairing::Halves && (n_rot == 0 || n_rot == hd) {
            return self.rope_standard(x, pos, hd, theta, None);
        }
        let (tq, d) = (x.shape()[1], x.shape()[2]);
        if n_rot > hd || !n_rot.is_multiple_of(2) || !hd.is_multiple_of(2) || d % hd != 0 {
            return Err(crate::error::BackendError::Shape(format!(
                "rope_standard_partial: n_rot={n_rot} / hd={hd} / d={d}"
            )));
        }
        let y = self.zeros(vec![1, tq, d])?;
        let xg = x.inner_metal();
        let yg = y.inner_metal();
        // partial rope (GLM) с YaRN пока не встречается — тождественные параметры
        let yarn = crate::rope::RopeYarn::IDENTITY;
        let grid = metal::MTLSize::new(tq as u64, (d / hd) as u64, hd as u64);
        let group = metal::MTLSize::new(1, 4, 8);
        self.ctx.run("rope_std_part_f32", grid, group, |enc| {
            enc.set_buffer(0, Some(&xg.buffer), xg.offset as u64);
            enc.set_buffer(1, Some(&yg.buffer), yg.offset as u64);
            enc.set_bytes(2, 8, (&tq as *const usize).cast());
            enc.set_bytes(3, 8, (&d as *const usize).cast());
            enc.set_bytes(4, 8, (&hd as *const usize).cast());
            enc.set_bytes(5, 8, (&n_rot as *const usize).cast());
            enc.set_bytes(6, 8, (&pos as *const usize).cast());
            enc.set_bytes(7, 4, (&theta as *const f32).cast());
            let pairing: usize = match pairing {
                crate::backend::RopePairing::Halves => 0,
                crate::backend::RopePairing::Adjacent => 1,
            };
            enc.set_bytes(8, 8, (&pairing as *const usize).cast());
            enc.set_bytes(9, 24, (&yarn as *const crate::rope::RopeYarn).cast());
        })?;
        Ok(y)
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
    ) -> Result<(crate::backend::Tensor, crate::backend::Tensor), crate::error::BackendError> {
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

    #[allow(clippy::too_many_arguments)]
    fn gdn_conv_silu_split_norm(
        &self,
        x: &crate::backend::Tensor,
        w: &crate::backend::Tensor,
        state: &crate::backend::Tensor,
        c: usize,
        k: usize,
        hd: usize,
        qk: usize,
        eps: f32,
    ) -> Result<(crate::backend::Tensor, crate::backend::Tensor), crate::error::BackendError> {
        let y = self.zeros(vec![c])?;
        let ns = self.zeros(vec![c, k - 1])?;
        self.ctx.gdn_conv_silu_split_norm_f32(
            x.as_gpu(),
            w.as_gpu(),
            state.as_gpu(),
            y.as_gpu(),
            ns.as_gpu(),
            c,
            k,
            hd,
            qk,
            eps,
        )?;
        Ok((y, ns))
    }

    fn gdn_gates(
        &self,
        ba: &crate::backend::Tensor,
        dt_bias: &crate::backend::Tensor,
        a: &crate::backend::Tensor,
    ) -> Result<(crate::backend::Tensor, crate::backend::Tensor), crate::error::BackendError> {
        let h = dt_bias.numel();
        let n = ba.numel() / 2;
        let mut gs = ba.shape().to_vec();
        let l = gs.len();
        gs[l - 1] = h;
        let g = self.zeros(gs.clone())?;
        let b = self.zeros(gs)?;
        self.ctx.gdn_gates_f32(
            ba.as_gpu(),
            dt_bias.as_gpu(),
            a.as_gpu(),
            g.as_gpu(),
            b.as_gpu(),
            n,
            h,
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
    ) -> Result<(crate::backend::Tensor, crate::backend::Tensor), crate::error::BackendError> {
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

    /// f16-KV (Ф8.2f): запись — `kv_update_f16_f32` (квантизация на записи),
    /// внимание — f16-варианты кернелов (см. attn_*_f16). Формат запрашивает
    /// модель (hd ≤ 128: у кернелов для hd=256 f16-варианта нет); при
    /// хранилище f32 всё идёт прежним путём.
    fn supported_kv_format(&self, fmt: crate::dtype::KvFormat) -> crate::dtype::KvFormat {
        fmt
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
            GpuDtype::Q4_0
                | GpuDtype::Q8_0
                | GpuDtype::Q4_K
                | GpuDtype::Q6_K
                | GpuDtype::Q5_K
                | GpuDtype::Q5_0
                | GpuDtype::Iq4Xs
                | GpuDtype::Iq4Nl
                | GpuDtype::Mxfp4
                | GpuDtype::Nvfp4
                | GpuDtype::Q2_K
                | GpuDtype::Q3_K
                | GpuDtype::Q4_1
                | GpuDtype::Q5_1
                | GpuDtype::Q8_1
                | GpuDtype::Q8_K
                | GpuDtype::Q1_0
                | GpuDtype::Q2_0
                | GpuDtype::TQ1_0
                | GpuDtype::TQ2_0
                | GpuDtype::Iq2Xxs
                | GpuDtype::Iq3Xxs
                | GpuDtype::Iq2Xs
                | GpuDtype::Iq2S
                | GpuDtype::Iq3S
                | GpuDtype::Iq1S
                | GpuDtype::Iq1M
        ) {
            let m = a.as_gpu().shape[0];
            if m <= 8 {
                // Эталонный ggml-metal берёт mul_mm ТОЛЬКО при m > 8
                // (ggml_metal_op_mul_mat_use_mm), иначе — mul_mv для любого
                // m. Быстрый mul_mv копит неизмасштабированные нибблы
                // (промежуточные acc в ~256× крупнее точного декванта), и его
                // округление отличается от точного dequant-GEMM на ~5e-5 на
                // скалярное произведение — через RMS-norm это разыгрывается в
                // расхождение текстов (OLMoE p2, символ 106). Значит при
                // m ≤ 8 идём matvec'ом, как эталон; по столбцам — каждый
                // выход независим и побитово равен эталонному запуску с
                // grid.z = m. mv_quantized — MMV-примитив: a 1D [k], c 1D [n].
                let k = a.as_gpu().shape[1];
                let n = bg.shape[0];
                // ОПТИМИЗАЦИЯ спек-окон (Ф8.2, 27B): окна верификации
                // (t = 1+γ ≤ 8) тут идут mv-путём — веса перечитываются с
                // каждой строкой, окно γ+1 строк дороже γ+1 AR-шагов, и спек
                // на Metal всегда проигрывал жадному декоду (Ф2.9/Ф8.2).
                // Рычаг INFER_QUANT_MM_MIN (дефолт 4) переводит окна 4..8 на
                // gemm_nt_quantized (mm-тайл 32 строк читает веса ОДИН раз):
                // window_probe 27B — окно 8 строк 520→280 мс (1.9×). mm при
                // m=2..3 ХУЖЕ mv-батча (холостые строки тайла), дефолт 4 —
                // баланс. Числа НЕ равны mv побитово (накопление нибблов в
                // быстром mv; у IQ4_XS различия — только знаки нулей, |Δ|=0;
                // Q5_K — побитово равен) — паритет с эталоном возвращается
                // INFER_QUANT_MM_MIN=9; тексты спека проверяются приёмкой
                // (лосслесс-семантика argmax-сверки).
                let mm_min = self.ctx.quant_mm_min();
                // K-кванты, окна m 4..6: mv_ext (ПОРТ эталона) где он реально
                // быстрее — матрица по формату×m из чередующихся прогонов
                // (8B-матмулы, минимум из 3, мс): Q4_K m=4 69 против mm 93
                // (−27%), m=5 ничья (98≈96), m=6 mm лучше (114 против 99);
                // Q5_K m=4..6 74/99/115 против mb 98/131/194 (−24/−25/−42%);
                // Q6_K только m=4 87 против mm 96 (−10%). Поэтому — по-форматная
                // матрица, а не «4..6 для всех»: широкое правило сначала было
                // ошибочным (Q4_K m=6 и Q6_K m=5..6 становились хуже).
                // m=2..3 сюда не входят: у K-квантов ext-кернела для них нет
                // (mv_ext_kernel: lo=4), там построчный mv. IQ4_XS — без ext
                // вовсе (окна: mb 2..3, mm 4..8).
                let k_ext_m = match bg.dtype {
                    GpuDtype::Q4_K => m == 4,
                    GpuDtype::Q5_K => (4..=6).contains(&m),
                    GpuDtype::Q6_K => m == 4,
                    _ => false,
                };
                let k_batched_ext = k_ext_m && std::env::var("INFER_NO_MV_EXT").is_err();
                if k_batched_ext && self.ctx.mv_ext(a.as_gpu(), bg, c.as_gpu())? {
                    return Ok(());
                }
                // Q5_K (gdn.qkv/w_out 27B): mb v2 — веса 1× на m строк БЕЗ
                // холостых строк mm-тайла. Замеренная матрица (8B, мс):
                // mb 47.5/68.4 (m=2,3) против mm 54.7/71.8 и ext 51.4/71.6 —
                // mb только для m 2..3; m 4..6 забирает ext выше; m=8 mb
                // гонит ДВА прохода и читает веса 2× — mm быстрее (~102 vs
                // 104). INFER_Q5K_MB=1 — форсаж mb и на 7..8 (сверка);
                // INFER_Q5K_NO_MB=1 — отказ от mb (сверки/замеры). Кернел
                // грузит A через float4 — гейт 16-байтного выравнивания
                // (misaligned vector load на Metal = мусор/фолт).
                let q5k_mb = if m >= 7 {
                    std::env::var("INFER_Q5K_MB").is_ok()
                } else {
                    std::env::var("INFER_Q5K_NO_MB").is_err()
                };
                if bg.dtype == GpuDtype::Q5_K
                    && q5k_mb
                    && mm_min <= 8
                    && m > 1
                    && m <= 8
                    && k.is_multiple_of(256)
                    && a.as_gpu().offset.is_multiple_of(16)
                {
                    self.ctx.mv_q5_k_mb(a.as_gpu(), bg, c.as_gpu())?;
                    return Ok(());
                }
                let mm_capable = matches!(
                    bg.dtype,
                    GpuDtype::Q4_K
                        | GpuDtype::Q5_K
                        | GpuDtype::Q6_K
                        | GpuDtype::Q5_0
                        | GpuDtype::Iq4Xs
                );
                let kmod = if bg.dtype == GpuDtype::Q5_0 { 32 } else { 256 };
                if m >= mm_min && m > 1 && mm_capable && k.is_multiple_of(kmod) {
                    self.ctx.gemm_nt_quantized(a.as_gpu(), bg, c.as_gpu())?;
                    return Ok(());
                }
                // Q8_0 (окно верификации Q8_0-моделей: MTP γ≤7, DFlash2-блок):
                // батчевый matvec mv_q8_0_mb — веса читаются 1× на m строк,
                // m=7..8 — два прохода (4+остаток). ЭТО БЫЛ основной путь окон
                // с 7247854, но Ф8.1 (b7d8d4e) завёл для паритета MoE-префилла
                // ветку «m≤8 → mv_ext» для ВСЕХ квантов, и Q8_0-окна молча
                // вернулись к чтению весов построчно (регресс без замеров).
                // Порог mb-пути — m≥2 (у mb нет холостых строк тайла; замер
                // 8B: M=2 41.3→32.0, M=3 37.6→32.3, M=4 54.3→33.1 мс), т.е.
                // НЕ mm_min: четвёрка в INFER_QUANT_MM_MIN — свойство mm-тайла.
                // Паритет-режим (INFER_QUANT_MM_MIN>8) и INFER_Q8_NO_MB=1
                // возвращают путь эталона. Гейты: 16-байтовый сдвиг A
                // (float4-загрузки) и K%32 (блок Q8_0) — иначе mv_ext.
                if bg.dtype == GpuDtype::Q8_0
                    && mm_min <= 8
                    && (2..=8).contains(&m)
                    && k.is_multiple_of(32)
                    && a.as_gpu().offset.is_multiple_of(16)
                    && std::env::var("INFER_Q8_NO_MB").is_err()
                {
                    self.ctx.gemm_nt_quantized(a.as_gpu(), bg, c.as_gpu())?;
                    return Ok(());
                }
                // Q4_0 (saiga-Q4_0, MTP/mmproj-веса Q4_0): тот же класс окон, что
                // у Q8_0, но батчевого пути не было ВООБЩЕ (ни mb, ни mm-тайла) —
                // окна m 2..8 читали веса m раз построчным mv_ext. mv_q4_0_mb
                // закрывает окна (m 7/8 — два прохода). Порог m≥2 — как у Q8_0
                // (у mb нет холостых строк), паритет-режим mm_min>8 и
                // INFER_Q4_0_NO_MB=1 — откат. Гейты: K%32 и 16Б-сдвиг A.
                if bg.dtype == GpuDtype::Q4_0
                    && mm_min <= 8
                    && (2..=8).contains(&m)
                    && k.is_multiple_of(32)
                    && a.as_gpu().offset.is_multiple_of(16)
                    && std::env::var("INFER_Q4_0_NO_MB").is_err()
                {
                    self.ctx.mv_q4_0_mb(a.as_gpu(), bg, c.as_gpu())?;
                    return Ok(());
                }
                // IQ4_NL / MXFP4 (новые форматы): окна m 2..8 — батчевый
                // matvec в gemm_nt_quantized (веса 1× на m строк), m>8 — mm;
                // m=1 идёт сюда же (arm отдаст скалярный mv). Гейты: K%32 и
                // 16-байтный сдвиг A (float2/float4-загрузки).
                let new_fmt_blk = match bg.dtype {
                    GpuDtype::Iq4Nl | GpuDtype::Mxfp4 => Some(32usize),
                    GpuDtype::Nvfp4 => Some(64),
                    GpuDtype::Q2_K
                    | GpuDtype::Q3_K
                    | GpuDtype::Q8_K
                    | GpuDtype::TQ1_0
                    | GpuDtype::TQ2_0
                    | GpuDtype::Iq2Xxs
                    | GpuDtype::Iq3Xxs
                    | GpuDtype::Iq2Xs
                    | GpuDtype::Iq2S
                    | GpuDtype::Iq3S
                    | GpuDtype::Iq1S
                    | GpuDtype::Iq1M => Some(256),
                    GpuDtype::Q4_1 | GpuDtype::Q5_1 | GpuDtype::Q8_1 => Some(32),
                    GpuDtype::Q1_0 => Some(128),
                    GpuDtype::Q2_0 => Some(64),
                    _ => None,
                };
                if new_fmt_blk.is_some_and(|b| k.is_multiple_of(b))
                    && mm_min <= 8
                    && (2..=8).contains(&m)
                    && a.as_gpu().offset.is_multiple_of(16)
                {
                    self.ctx.gemm_nt_quantized(a.as_gpu(), bg, c.as_gpu())?;
                    return Ok(());
                }
                // IQ4_XS, окна m 2..3 (γ=1,2): батчевый матvec уже выбран в
                // gemm_nt_quantized для m ≤ 8, но trait доходил туда только
                // при m ≥ mm_min (дефолт 4) — окна γ=1,2 читали веса 2-3
                // раза построчным mv. Маршрут паритет-НЕЙТРАЛЕН: mb
                // побитово равен построчному mv (тест
                // iq4_xs_matvec_batch_consistency), поэтому выигрыш здесь
                // только скоростной. A читается скалярно — гейт выравнивания
                // не нужен; K%256 — из раскладки суперблоков.
                if bg.dtype == GpuDtype::Iq4Xs
                    && mm_min <= 8
                    && (2..=3).contains(&m)
                    && k.is_multiple_of(256)
                    && std::env::var("INFER_IQ4XS_NO_MB").is_err()
                {
                    self.ctx.gemm_nt_quantized(a.as_gpu(), bg, c.as_gpu())?;
                    return Ok(());
                }
                // Внутри ext-окна эталон берёт НЕ classic mul_mv, а
                // mul_mv_ext (свой порядок суммирования) — воспроизводим
                // выбор точно; вне окна mv_ext возвращает false.
                // Фоллбэк INFER_NO_MV_EXT=1.
                if std::env::var("INFER_NO_MV_EXT").is_err()
                    && self.ctx.mv_ext(a.as_gpu(), bg, c.as_gpu())?
                {
                    return Ok(());
                }
                for j in 0..m {
                    let av = a.view_offset(vec![k], j * k);
                    let cv = c.view_offset(vec![n], j * n);
                    self.ctx.mv_quantized(av.as_gpu(), bg, cv.as_gpu())?;
                }
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
            crate::dtype::WeightFormat::Q5_K => GpuDtype::Q5_K,
            crate::dtype::WeightFormat::Q5_0 => GpuDtype::Q5_0,
            crate::dtype::WeightFormat::Iq4Xs => GpuDtype::Iq4Xs,
            crate::dtype::WeightFormat::Iq4Nl => GpuDtype::Iq4Nl,
            crate::dtype::WeightFormat::Mxfp4 => GpuDtype::Mxfp4,
            crate::dtype::WeightFormat::Nvfp4 => GpuDtype::Nvfp4,
            crate::dtype::WeightFormat::Q2_K => GpuDtype::Q2_K,
            crate::dtype::WeightFormat::Q3_K => GpuDtype::Q3_K,
            crate::dtype::WeightFormat::Q4_1 => GpuDtype::Q4_1,
            crate::dtype::WeightFormat::Q5_1 => GpuDtype::Q5_1,
            crate::dtype::WeightFormat::Q8_1 => GpuDtype::Q8_1,
            crate::dtype::WeightFormat::Q8_K => GpuDtype::Q8_K,
            crate::dtype::WeightFormat::Q1_0 => GpuDtype::Q1_0,
            crate::dtype::WeightFormat::Q2_0 => GpuDtype::Q2_0,
            crate::dtype::WeightFormat::TQ1_0 => GpuDtype::TQ1_0,
            crate::dtype::WeightFormat::TQ2_0 => GpuDtype::TQ2_0,
            crate::dtype::WeightFormat::Iq2Xxs => GpuDtype::Iq2Xxs,
            crate::dtype::WeightFormat::Iq3Xxs => GpuDtype::Iq3Xxs,
            crate::dtype::WeightFormat::Iq2Xs => GpuDtype::Iq2Xs,
            crate::dtype::WeightFormat::Iq2S => GpuDtype::Iq2S,
            crate::dtype::WeightFormat::Iq3S => GpuDtype::Iq3S,
            crate::dtype::WeightFormat::Iq1S => GpuDtype::Iq1S,
            crate::dtype::WeightFormat::Iq1M => GpuDtype::Iq1M,
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

    fn argmax_rows(
        &self,
        logits: &crate::backend::Tensor,
        history: Option<(&[u32], f32)>,
    ) -> Result<(Vec<u32>, Vec<f32>), crate::error::BackendError> {
        let shape = logits.shape().to_vec();
        let v = *shape.last().unwrap_or(&0);
        if v == 0 {
            return Err(crate::error::BackendError::Shape(
                "argmax_rows: пустой словарь".into(),
            ));
        }
        let t = logits.numel() / v;
        let factors = match history {
            Some((hist, penalty)) if penalty != 1.0 && !hist.is_empty() => {
                // Множитель штрафа на токен: 1.0 вне уникальной истории.
                // Само деление/умножение — в ядре (знак логита).
                let mut f = vec![1.0f32; v];
                let mut seen: Vec<u32> = hist.to_vec();
                seen.sort_unstable();
                seen.dedup();
                for &tok in &seen {
                    if let Some(slot) = f.get_mut(tok as usize) {
                        *slot = penalty;
                    }
                }
                Some(self.from_cpu_f32(&f, vec![v])?)
            }
            _ => None,
        };
        let r = self.ctx.argmax_lse_rows_f32(
            logits.as_gpu(),
            factors.as_ref().map(|t| t.as_gpu()),
            t,
            v,
        )?;
        Ok(r)
    }

    #[allow(clippy::too_many_arguments)]
    fn copy_rows_into(
        &self,
        src: &crate::backend::Tensor,
        rows: usize,
        cols: usize,
        dst: crate::backend::Tensor,
        dst_row0: usize,
        dst_col0: usize,
        dst_stride: usize,
    ) -> Result<crate::backend::Tensor, crate::error::BackendError> {
        self.ctx.copy_rows_f32(
            src.as_gpu(),
            rows,
            cols,
            dst.as_gpu(),
            dst_row0,
            dst_col0,
            dst_stride,
        )?;
        Ok(dst)
    }

    fn add_bias(
        &self,
        x: &crate::backend::Tensor,
        bias: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
    ) -> Result<(), crate::error::BackendError> {
        // Кернел читает bias как float*: f16-буфер (upload_w на Metal грузит
        // f16!) молча давал мусор вместо биаса — ловим явно.
        if bias.as_gpu().dtype != GpuDtype::F32 || x.as_gpu().dtype != GpuDtype::F32 {
            return Err(crate::error::BackendError::Shape(format!(
                "add_bias: нужны f32-тензоры, получено x={:?} bias={:?}",
                x.as_gpu().dtype,
                bias.as_gpu().dtype
            )));
        }
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

    fn swiglu(
        &self,
        x: &crate::backend::Tensor,
        u: &crate::backend::Tensor,
        y: &crate::backend::Tensor,
    ) -> Result<(), crate::error::BackendError> {
        self.ctx.swiglu_f32(x.as_gpu(), u.as_gpu(), y.as_gpu())?;
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
            self.ctx
                .embedding_quant_f32(idx, table.as_gpu(), y.as_gpu())?;
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

    fn moe_experts_mv(
        &self,
        a: &crate::backend::Tensor,
        ids: &crate::backend::Tensor,
        weights: &crate::backend::Tensor,
        b: &crate::backend::Tensor,
        c: &crate::backend::Tensor,
        n_rows: usize,
        per_slot: bool,
    ) -> Result<(), crate::error::BackendError> {
        use crate::error::BackendError;
        let (ag, bg, cg) = (a.inner_metal(), b.inner_metal(), c.inner_metal());
        let k = a.shape()[a.shape().len() - 1];
        let kslots = ids.numel();
        if kslots == 0 || weights.numel() != kslots {
            return Err(BackendError::Shape(format!(
                "moe_experts_mv: ids {} / weights {}",
                kslots,
                weights.numel()
            )));
        }
        let (ig, wg) = (ids.inner_metal(), weights.inner_metal());
        let name = match (bg.dtype, per_slot) {
            (GpuDtype::Q4_K, false) => "mv_id_q4_k",
            (GpuDtype::Q4_K, true) => "mv_id_q4_k_dpsl",
            (GpuDtype::Q6_K, false) => "mv_id_q6_k",
            (GpuDtype::Q6_K, true) => "mv_id_q6_k_dpsl",
            (other, _) => {
                return Err(BackendError::UnsupportedFormat(format!(
                    "moe_experts_mv: квант {other:?} не поддержан (Q4_K/Q6_K)"
                )));
            }
        };
        let block = bg.dtype.block_elems().unwrap_or(0);
        if block == 0 || !k.is_multiple_of(block) {
            return Err(BackendError::Shape(format!(
                "moe_experts_mv: K={k} не кратен блоку {block}"
            )));
        }
        let (k32, n32, ks32) = (k as u32, n_rows as u32, kslots as u32);
        // 8 строк весов на threadgroup (4 simdgroup × пара строк, 32 лейна на
        // строку — r2-раскладка), слоты — цикл.
        let grid = MTLSize::new((n_rows.div_ceil(8) * 128) as u64, 1, 1);
        let group = MTLSize::new(128, 1, 1);
        self.ctx
            .run(name, grid, group, |enc| {
                enc.set_buffer(0, Some(&ag.buffer), ag.offset as u64);
                enc.set_buffer(1, Some(&ig.buffer), ig.offset as u64);
                enc.set_buffer(2, Some(&wg.buffer), wg.offset as u64);
                enc.set_buffer(3, Some(&bg.buffer), bg.offset as u64);
                enc.set_buffer(4, Some(&cg.buffer), cg.offset as u64);
                enc.set_bytes(5, 4, (&k32 as *const u32).cast());
                enc.set_bytes(6, 4, (&n32 as *const u32).cast());
                enc.set_bytes(7, 4, (&ks32 as *const u32).cast());
            })
            .map_err(|e| BackendError::Shape(format!("moe_experts_mv: {e}")))
    }

    #[allow(clippy::too_many_arguments)]
    fn moe_experts_mm(
        &self,
        a: &crate::backend::Tensor,
        ids: &crate::backend::Tensor,
        b: &crate::backend::Tensor,
        c: &crate::backend::Tensor,
        n_rows: usize,
        n_expert: usize,
        t: usize,
        k: usize,
        ne11: usize,
        nb11: u64,
        nb12: u64,
    ) -> Result<(), crate::error::BackendError> {
        use crate::error::BackendError;
        // Дословные копии ggml_metal_kargs_mul_mm_id_{map0,} (порядок и
        // выравнивание полей = C-layout, repr(C) совпадает с MSL).
        #[repr(C)]
        struct MmIdMap0Args {
            ne02: i32,
            ne10: i32,
            ne11: i32,
            nb11: u64,
            nb12: u64,
            ne21: i32,
            ne20: i32,
            nb21: u64,
        }
        #[repr(C)]
        struct MmIdArgs {
            ne00: i32,
            ne02: i32,
            nb01: u64,
            nb02: u64,
            nb03: u64,
            ne11: i32,
            nb10: u64,
            nb11: u64,
            nb12: u64,
            nb13: u64,
            ne20: i32,
            ne21: i32,
            ne0: i32,
            ne1: i32,
        }
        let (ag, bg, cg, ig) = (
            a.inner_metal(),
            b.inner_metal(),
            c.inner_metal(),
            ids.inner_metal(),
        );
        let kk = a.shape()[a.shape().len() - 1];
        let name = match bg.dtype {
            GpuDtype::Q4_K => "mm_id_q4_k",
            GpuDtype::Q6_K => "mm_id_q6_k",
            other => {
                return Err(BackendError::UnsupportedFormat(format!(
                    "moe_experts_mm: квант {other:?} не поддержан (Q4_K/Q6_K)"
                )));
            }
        };
        let blk = bg.dtype.block_bytes().unwrap_or(0) as u64;
        if blk == 0 || !kk.is_multiple_of(256) || n_expert == 0 || t == 0 || k == 0 {
            return Err(BackendError::Shape(format!(
                "moe_experts_mm: K={kk} E={n_expert} t={t} k={k}"
            )));
        }
        let nb01 = (kk as u64 / 256) * blk;
        // scratch: tpe [E] u32 + hids [E*t] i32 (у оркала — хвост dst-буфера)
        let scratch = self.ctx.device.new_buffer(
            ((n_expert * (1 + t)) * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let map0_name = MetalCtx::static_kernel_name(format!("mm_id_map0_ne20_{k}"));
        let m0 = MmIdMap0Args {
            ne02: n_expert as i32,
            ne10: 0,
            ne11: 0,
            nb11: 0,
            nb12: 0,
            ne21: t as i32,
            ne20: k as i32,
            nb21: (k * 4) as u64,
        };
        let smem0 = ((n_expert * k * 2).div_ceil(16) * 16) as u64;
        self.ctx
            .run(
                map0_name,
                MTLSize::new(n_expert as u64, 1, 1),
                MTLSize::new(n_expert as u64, 1, 1),
                |enc| {
                    enc.set_bytes(
                        0,
                        std::mem::size_of::<MmIdMap0Args>() as u64,
                        (&m0 as *const MmIdMap0Args).cast(),
                    );
                    enc.set_buffer(1, Some(&ig.buffer), ig.offset as u64);
                    enc.set_buffer(2, Some(&scratch), 0);
                    enc.set_buffer(3, Some(&scratch), (n_expert * 4) as u64);
                    enc.set_threadgroup_memory_length(0, smem0);
                },
            )
            .map_err(|e| BackendError::Shape(format!("moe_experts_mm map0: {e}")))?;
        let args = MmIdArgs {
            ne00: kk as i32,
            ne02: n_expert as i32,
            nb01,
            nb02: nb01 * n_rows as u64,
            nb03: 0,
            ne11: ne11 as i32,
            nb10: 4,
            nb11,
            nb12,
            nb13: 0,
            ne20: k as i32,
            ne21: t as i32,
            ne0: n_rows as i32,
            ne1: k as i32,
        };
        // dispatch_threadgroups ((t+31)/32, (N+63)/64, E) × 128; run() мыслит
        // потоками: grid.x = тайлов × 128.
        let grid = MTLSize::new(
            (t.div_ceil(32) * 128) as u64,
            n_rows.div_ceil(64) as u64,
            n_expert as u64,
        );
        self.ctx
            .run(name, grid, MTLSize::new(128, 1, 1), |enc| {
                enc.set_bytes(
                    0,
                    std::mem::size_of::<MmIdArgs>() as u64,
                    (&args as *const MmIdArgs).cast(),
                );
                enc.set_buffer(1, Some(&bg.buffer), bg.offset as u64);
                enc.set_buffer(2, Some(&ag.buffer), ag.offset as u64);
                enc.set_buffer(3, Some(&scratch), 0);
                enc.set_buffer(4, Some(&scratch), (n_expert * 4) as u64);
                enc.set_buffer(5, Some(&cg.buffer), cg.offset as u64);
                enc.set_threadgroup_memory_length(0, 8192);
            })
            .map_err(|e| BackendError::Shape(format!("moe_experts_mm: {e}")))
    }

    fn moe_acc_slots(
        &self,
        src: &crate::backend::Tensor,
        w: &crate::backend::Tensor,
        acc: &crate::backend::Tensor,
        t: usize,
        k: usize,
    ) -> Result<(), crate::error::BackendError> {
        use crate::error::BackendError;
        let (sg, wg, agt) = (src.inner_metal(), w.inner_metal(), acc.inner_metal());
        let d = acc.shape()[acc.shape().len() - 1];
        let (t32, k32, d32) = (t as u32, k as u32, d as u32);
        let grid = MTLSize::new((t * d) as u64, 1, 1);
        self.ctx
            .run("moe_acc_slots", grid, MTLSize::new(64, 1, 1), |enc| {
                enc.set_buffer(0, Some(&sg.buffer), sg.offset as u64);
                enc.set_buffer(1, Some(&wg.buffer), wg.offset as u64);
                enc.set_buffer(2, Some(&agt.buffer), agt.offset as u64);
                enc.set_bytes(3, 4, (&t32 as *const u32).cast());
                enc.set_bytes(4, 4, (&k32 as *const u32).cast());
                enc.set_bytes(5, 4, (&d32 as *const u32).cast());
            })
            .map_err(|e| BackendError::Shape(format!("moe_acc_slots: {e}")))
    }

    fn from_cpu_u32(
        &self,
        data: &[u32],
        shape: Vec<usize>,
    ) -> Result<crate::backend::Tensor, crate::error::BackendError> {
        let n: usize = shape.iter().product();
        if n != data.len() {
            return Err(crate::error::BackendError::Shape(format!(
                "from_cpu_u32: shape {shape:?} против {} элементов",
                data.len()
            )));
        }
        let mut t = GpuTensor::zeros_f32(&self.ctx.device, vec![n])?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr().cast::<u8>(),
                t.buffer.contents().cast::<u8>(),
                n * 4,
            );
        }
        t.shape = shape;
        Ok(crate::backend::Tensor::from_gpu(t))
    }

    fn to_cpu_u32(
        &self,
        t: &crate::backend::Tensor,
    ) -> Result<Vec<u32>, crate::error::BackendError> {
        let g = t.inner_metal();
        let n = t.numel();
        // Как to_cpu_f32: сначала слить батч (запись кернела не видна до
        // commit+wait), затем скопировать n×4 байта с учётом offset вью.
        self.ctx
            .flush_batch()
            .map_err(|e| crate::error::BackendError::Shape(format!("to_cpu_u32: {e}")))?;
        let barrier = self.ctx.queue.new_command_buffer();
        barrier.commit();
        barrier.wait_until_completed();
        let ptr = g.buffer.contents();
        if ptr.is_null() {
            return Err(crate::error::BackendError::Shape(
                "to_cpu_u32: нулевой contents".into(),
            ));
        }
        let base = unsafe { (ptr as *const u8).add(g.offset) };
        let mut out = vec![0u32; n];
        unsafe {
            std::ptr::copy_nonoverlapping(base, out.as_mut_ptr().cast::<u8>(), n * 4);
        }
        Ok(out)
    }

    fn moe_route(
        &self,
        logits: &crate::backend::Tensor,
        ids: &crate::backend::Tensor,
        weights: &crate::backend::Tensor,
        k: usize,
        norm: bool,
        scale: f32,
    ) -> Result<(), crate::error::BackendError> {
        use crate::error::BackendError;
        let (lg, ig, wg) = (
            logits.inner_metal(),
            ids.inner_metal(),
            weights.inner_metal(),
        );
        let shp = logits.shape();
        if shp.len() != 2 {
            return Err(BackendError::Shape(format!(
                "moe_route: логиты {shp:?}, ждали [T, E]"
            )));
        }
        let (t, e) = (shp[0], shp[1]);
        if e > 4096 || k == 0 || k > e {
            return Err(BackendError::Shape(format!(
                "moe_route: E={e}, k={k} вне пределов (E ≤ 4096, 1 ≤ k ≤ E)"
            )));
        }
        let (e32, k32) = (e as u32, k as u32);
        let norm32 = norm as u32;
        // dispatch_threads: grid — ЧИСЛО ПОТОКОВ (не threadgroup'ов!), т.е.
        // 128 потоков на токен. С `grid = T` запускался один неполный
        // threadgroup, и роутер считал только первый токен (тихо).
        let grid = MTLSize::new((t * 128) as u64, 1, 1);
        let group = MTLSize::new(128, 1, 1);
        self.ctx
            .run("moe_route_topk", grid, group, |enc| {
                enc.set_buffer(0, Some(&lg.buffer), lg.offset as u64);
                enc.set_buffer(1, Some(&ig.buffer), ig.offset as u64);
                enc.set_buffer(2, Some(&wg.buffer), wg.offset as u64);
                enc.set_bytes(3, 4, (&e32 as *const u32).cast());
                enc.set_bytes(4, 4, (&k32 as *const u32).cast());
                enc.set_bytes(5, 4, (&norm32 as *const u32).cast());
                enc.set_bytes(6, 4, (&scale as *const f32).cast());
                enc.set_threadgroup_memory_length(0, (e * 4) as u64);
                enc.set_threadgroup_memory_length(1, 512);
                enc.set_threadgroup_memory_length(2, 512);
            })
            .map_err(|err| BackendError::Shape(format!("moe_route: {err}")))
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

    /// IQ4_XS (волна A4): эталон — порт `dequantize_row_iq4_xs` из ggml.
    /// Синтетические ВАЛИДНЫЕ блоки (случайные байты дали бы NaN в f16-масштабе):
    /// d = 1, scales_l/scales_h задают ls 0..63, нибблы — индексы кодбука.
    /// Сверяем и хост-деквант, и Metal-matvec (mv_iq4_xs) с ним.
    #[test]
    fn iq4_xs_mv_matches_host_dequant() {
        use crate::backend::Backend as _;
        let be = MetalBackend::new().unwrap();
        let (n, k) = (5usize, 512usize); // 5 строк по 2 суперблока
        let mut seed = 0x9E37_79B9u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut raw = Vec::with_capacity(n * k / 256 * 136);
        for _ in 0..n * k / 256 {
            raw.extend_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes()); // d
            raw.extend_from_slice(&(next() as u16).to_le_bytes()); // scales_h
            for _ in 0..4 {
                raw.push((next() & 0xFF) as u8); // scales_l
            }
            for _ in 0..128 {
                raw.push((next() & 0xFF) as u8); // нибблы
            }
        }
        let w = be
            .from_quantized(&raw, vec![n, k], crate::dtype::WeightFormat::Iq4Xs)
            .unwrap();
        let a: Vec<f32> = (0..k).map(|i| ((i % 29) as f32 - 14.0) * 0.03).collect();
        let av = be.from_cpu_f32(&a, vec![1, k]).unwrap();
        let c = be.zeros(vec![1, n]).unwrap();
        let a1 = av.view(vec![k]);
        let c1 = c.view(vec![n]);
        be.ctx
            .mv_quantized(a1.inner_metal(), w.inner_metal(), c1.inner_metal())
            .unwrap();
        let got = be.to_cpu_f32(&c).unwrap();
        // эталон: деквант строки (тот же алгоритм, что у деquant_iq4_xs) + dot
        for row in 0..n {
            let bytes = &raw[row * (k / 256) * 136..(row + 1) * (k / 256) * 136];
            let y = crate::weights_map::dequant_iq4_xs(bytes);
            let want: f64 = y
                .iter()
                .zip(&a)
                .map(|(w, x)| (*w as f64) * (*x as f64))
                .sum();
            let d = (got[row] as f64 - want).abs();
            assert!(
                d <= 1e-4 * (1.0 + want.abs()),
                "строка {row}: Metal {} против хоста {want} (Δ {d})",
                got[row]
            );
        }
    }

    /// IQ4_XS matvec: строка при M>1 должна давать тот же результат, что та
    /// же строка при M=1 (per-(row,token) редукция не зависит от M). Ловит баги
    /// сетки (32,N,M) / указателя a=A+m·K при пакетном префилле.
    #[test]
    fn iq4_xs_matvec_batch_consistency() {
        use crate::backend::Backend as _;
        let be = MetalBackend::new().unwrap();
        let (n, k) = (40usize, 512usize);
        let mut seed = 0xABCD_1234u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut raw = Vec::new();
        for _ in 0..n * k / 256 {
            raw.extend_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
            raw.extend_from_slice(&(next() as u16).to_le_bytes());
            for _ in 0..4 {
                raw.push((next() & 0xFF) as u8);
            }
            for _ in 0..128 {
                raw.push((next() & 0xFF) as u8);
            }
        }
        let w = be
            .from_quantized(&raw, vec![n, k], crate::dtype::WeightFormat::Iq4Xs)
            .unwrap();
        let rows = 5usize;
        let a: Vec<f32> = (0..rows * k)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.03 + (i / k) as f32 * 0.01)
            .collect();
        // M=5 пачкой. Ф8.2: при дефолтном пороге (INFER_QUANT_MM_MIN=4) M≥4
        // идёт mm-кернелом — его числа НЕ равны mb/mv побитово (накопление
        // нибблов), расхождение ~1e-3 относительных — это ОЖИДАЕМО. Поэтому
        // побитовая сверка mb==mv — под паритет-порогом 9, а мм-путь — с
        // rel-допуском против построчного mv.
        unsafe { std::env::set_var("INFER_QUANT_MM_MIN", "9") };
        let av5 = be.from_cpu_f32(&a, vec![rows, k]).unwrap();
        let c5 = be.zeros(vec![rows, n]).unwrap();
        be.ctx
            .gemm_nt_quantized(av5.inner_metal(), w.inner_metal(), c5.inner_metal())
            .unwrap();
        let got5 = be.to_cpu_f32(&c5).unwrap();
        // каждая строка по-одиночке (M=1)
        for r in 0..rows {
            let av1 = be.from_cpu_f32(&a[r * k..r * k + k], vec![1, k]).unwrap();
            let c1 = be.zeros(vec![1, n]).unwrap();
            be.ctx
                .gemm_nt_quantized(av1.inner_metal(), w.inner_metal(), c1.inner_metal())
                .unwrap();
            let got1 = be.to_cpu_f32(&c1).unwrap();
            for ni in 0..n {
                let b = got5[r * n + ni];
                let s = got1[ni];
                assert_eq!(
                    b.to_bits(),
                    s.to_bits(),
                    "M=5 row={r} n={ni}: {b} vs M=1 {s} (не побитово)"
                );
            }
        }
        // мм-путь (дефолтный порог 4 → M=5 идёт gemm_nt_iq4_xs_mm): сверка
        // ОБОИХ путей (mm и построчный mv) с хост-деквантом — численная
        // истина; mm vs mv напрямую имеет длинный хвост расхождений
        // (накопление нибблов в быстром mv), но оба близки к точному.
        unsafe { std::env::remove_var("INFER_QUANT_MM_MIN") };
        let cmm = be.zeros(vec![rows, n]).unwrap();
        be.ctx
            .gemm_nt_quantized(av5.inner_metal(), w.inner_metal(), cmm.inner_metal())
            .unwrap();
        let gotm = be.to_cpu_f32(&cmm).unwrap();
        let wt_f = crate::weights_map::dequant_iq4_xs(&raw);
        for r in 0..rows {
            let x = &a[r * k..r * k + k];
            let host: Vec<f32> = (0..n)
                .map(|ni| (0..k).map(|ki| wt_f[ni * k + ki] * x[ki]).sum())
                .collect();
            let av1 = be.from_cpu_f32(x, vec![1, k]).unwrap();
            let c1 = be.zeros(vec![1, n]).unwrap();
            be.ctx
                .gemm_nt_quantized(av1.inner_metal(), w.inner_metal(), c1.inner_metal())
                .unwrap();
            let got1 = be.to_cpu_f32(&c1).unwrap();
            for ni in 0..n {
                let h = host[ni];
                let bm = gotm[r * n + ni];
                let bv = got1[ni];
                let scale = 1.0 + h.abs();
                let relm = (bm - h).abs() / scale;
                let relv = (bv - h).abs() / scale;
                // Оба пути — порты llama.cpp (mul_mm/mul_mv) с накоплением
                // нескалированных нибблов: отклонение от ТОЧНОГО декванта
                // до ~3% rel на крупновесной синтетике (d=1) — свойство
                // кернелов эталона, не баг раскладки (грубый баг дал бы
                // O(1)); побитовый паритет с оракулом — на совпадении путей
                // (см. oracle-сверки A4).
                assert!(
                    relm < 5e-2,
                    "mm row={r} n={ni}: {bm} vs host {h} (rel {relm})"
                );
                assert!(
                    relv < 5e-2,
                    "mv row={r} n={ni}: {bv} vs host {h} (rel {relv})"
                );
            }
        }
    }

    /// Q5_K mb: matvec с батчем строк против точного хост-декванта.
    /// Формы: m 2/3/5 (один проход) и 7/8 (два прохода 4+остаток); K 512 и
    /// 768 (nb=2/3 — хвостовые лейны и неполные суперблоки в simd_sum).
    /// Допуск rel 1e-3: mb деквантует поэлементно (как хост), отличается
    /// только порядок fp32-суммирования (simd_sum по лейнам) — ожидаемо
    /// ~1e-5..1e-6; 1e-3 — с запасом, грубые сдвиги раскладки дают O(1).
    #[test]
    fn q5_k_mb_matches_host_dequant() {
        use crate::backend::Backend as _;
        let be = MetalBackend::new().unwrap();
        for (n, k, m) in [
            (4usize, 512usize, 5usize),
            (4, 512, 2),
            (3, 768, 3),
            (4, 512, 7),
            (4, 768, 8),
        ] {
            let mut seed = 0x5EED_1A2Bu64 ^ ((k as u64) << 8) ^ (m as u64);
            let mut next = || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed
            };
            // Q5_K: 176 Б/256 = d | dmin | scales[12] | qh[32] | qs[128]
            let mut raw = Vec::with_capacity(n * k / 256 * 176);
            for _ in 0..n * k / 256 {
                raw.extend_from_slice(&half::f16::from_f32(0.02).to_bits().to_le_bytes());
                raw.extend_from_slice(&half::f16::from_f32(0.005).to_bits().to_le_bytes());
                for _ in 0..12 {
                    raw.push((next() & 0xFF) as u8);
                }
                for _ in 0..32 {
                    raw.push((next() & 0xFF) as u8);
                }
                for _ in 0..128 {
                    raw.push((next() & 0xFF) as u8);
                }
            }
            let w = be
                .from_quantized(&raw, vec![n, k], crate::dtype::WeightFormat::Q5_K)
                .unwrap();
            let a: Vec<f32> = (0..m * k)
                .map(|i| ((i % 31) as f32 - 15.0) * 0.02)
                .collect();
            let av = be.from_cpu_f32(&a, vec![m, k]).unwrap();
            let c = be.zeros(vec![m, n]).unwrap();
            be.ctx
                .mv_q5_k_mb(av.inner_metal(), w.inner_metal(), c.inner_metal())
                .unwrap();
            let got = be.to_cpu_f32(&c).unwrap();
            for row in 0..n {
                let bytes = &raw[row * (k / 256) * 176..(row + 1) * (k / 256) * 176];
                let y = crate::weights_map::dequant_q5_k(bytes);
                for mi in 0..m {
                    let x = &a[mi * k..(mi + 1) * k];
                    let want: f64 = y
                        .iter()
                        .zip(x)
                        .map(|(w, x)| (*w as f64) * (*x as f64))
                        .sum();
                    let d = (got[mi * n + row] as f64 - want).abs() / (1.0 + want.abs());
                    assert!(
                        d < 1e-3,
                        "n={n} k={k} m={m}: mb[{mi},{row}]={} vs host {want} (rel {d})",
                        got[mi * n + row]
                    );
                }
            }
        }
    }

    /// Маршрут окон K-квантов по m — замеренная матрица, а не «ext для всех»:
    /// Q4_K — m=4 ext, m=6/7 тайл (на 6 ext ХУЖЕ: 114 против 99 мс); Q5_K —
    /// m=4..6 ext, m=8 тайл; Q6_K — m=4 ext, m=5 тайл. Пин сравнивает
    /// трейтовый gemm_nt с ПРЯМЫМ ctx.mv_ext побитово: удаление стадии
    /// ext-предпочтения или расширение её на чужой m уронят тест.
    #[test]
    fn trait_gemm_nt_k_quant_windows_route_by_m() {
        use crate::backend::Backend as _;
        let be = MetalBackend::new().expect("MetalBackend");
        let (n, k) = (77usize, 512usize);
        let mut seed = 0xC0DE_5150u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        // Q4_K: 144 Б = d | dmin | scales[12] | qs[128]
        let mut q4k = Vec::with_capacity(n * k / 256 * 144);
        for _ in 0..n * k / 256 {
            q4k.extend_from_slice(&half::f16::from_f32(0.02).to_bits().to_le_bytes());
            q4k.extend_from_slice(&half::f16::from_f32(0.005).to_bits().to_le_bytes());
            for _ in 0..12 {
                q4k.push((next() & 0xFF) as u8);
            }
            for _ in 0..128 {
                q4k.push((next() & 0xFF) as u8);
            }
        }
        // Q5_K: 176 Б = d | dmin | scales[12] | qh[32] | qs[128]
        let mut q5k = Vec::with_capacity(n * k / 256 * 176);
        for _ in 0..n * k / 256 {
            q5k.extend_from_slice(&half::f16::from_f32(0.02).to_bits().to_le_bytes());
            q5k.extend_from_slice(&half::f16::from_f32(0.005).to_bits().to_le_bytes());
            for _ in 0..12 {
                q5k.push((next() & 0xFF) as u8);
            }
            for _ in 0..32 {
                q5k.push((next() & 0xFF) as u8);
            }
            for _ in 0..128 {
                q5k.push((next() & 0xFF) as u8);
            }
        }
        // Q6_K: 210 Б = ql[128] | qh[64] | scales[16] | d
        let mut q6k = Vec::with_capacity(n * k / 256 * 210);
        for _ in 0..n * k / 256 {
            for _ in 0..192 {
                q6k.push((next() & 0xFF) as u8);
            }
            for _ in 0..16 {
                q6k.push(((next() & 0x3F) as i8 - 32) as u8);
            }
            q6k.extend_from_slice(&half::f16::from_f32(0.01).to_bits().to_le_bytes());
        }
        let w4k = be
            .from_quantized(&q4k, vec![n, k], crate::dtype::WeightFormat::Q4_K)
            .unwrap();
        let w5k = be
            .from_quantized(&q5k, vec![n, k], crate::dtype::WeightFormat::Q5_K)
            .unwrap();
        let w6k = be
            .from_quantized(&q6k, vec![n, k], crate::dtype::WeightFormat::Q6_K)
            .unwrap();
        // (формат, m, ожидается ли ext). У K-квантов ext существует только
        // при m ≥ 4 (mv_ext_kernel: lo=4 для K-квантов), поэтому m=2..3 сюда
        // не входят — там построчный mv.
        for (w, tag, m, want_ext) in [
            (&w4k, "Q4_K", 4usize, true),
            (&w4k, "Q4_K", 6, false),
            (&w4k, "Q4_K", 7, false),
            (&w5k, "Q5_K", 4, true),
            (&w5k, "Q5_K", 6, true),
            (&w5k, "Q5_K", 8, false),
            (&w6k, "Q6_K", 4, true),
            (&w6k, "Q6_K", 5, false),
        ] {
            let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 7919.0).sin()).collect();
            let a = be.from_cpu_f32(&a_data, vec![m, k]).unwrap();
            be.ctx.set_quant_mm_min(4);
            let c_new = be.zeros(vec![m, n]).unwrap();
            be.gemm_nt(&a, w, &c_new).unwrap();
            let got = be.to_cpu_f32(&c_new).unwrap();
            let c_ext = be.zeros(vec![m, n]).unwrap();
            let used = be
                .ctx
                .mv_ext(a.inner_metal(), w.inner_metal(), c_ext.inner_metal())
                .unwrap();
            assert!(used, "{tag} m={m}: ext не применим — тест недействителен");
            let ext = be.to_cpu_f32(&c_ext).unwrap();
            let same = got
                .iter()
                .zip(ext.iter())
                .all(|(g, e)| g.to_bits() == e.to_bits());
            assert_eq!(
                same,
                want_ext,
                "{tag} m={m}: ожидался {}ext-путь",
                if want_ext { "" } else { "НЕ " }
            );
        }
    }

    /// Q4_K mm (m>8, префилл) против ТОЧНОЙ формулы эталонного
    /// dequantize_q4_K (dequantize.h:431, kernel_mul_mm_q4_K_f32): для
    /// нечётных подблоков эталон считает `(d/16) * sc * (byte & 0xF0) - ml`,
    /// а не `d * sc * q_hi - ml` — математически равно, но half-округление
    /// веса идёт от другой величины. Наш q4k_dequantize_mm — дословный порт.
    ///
    /// ПРОВЕРКА ПОБИТОВАЯ: активации — единичная матрица (A[i][i]=1), тогда
    /// C[k][n] = ровно float(half-вес w[n,k]) (одно слагаемое 1.0*half_w,
    /// остальные нули — float-сумма точна). Сравниваем биты float с
    /// эталонной half-формулой: расхождение хотя бы на 1 ulp half падает.
    /// Долговая запись «НЕ побитово с эталонным mm» относилась к рукописному
    /// порту ДО дословного (gemm.metal: get_scale_min_k4_just2).
    #[test]
    fn q4_k_mm_weights_bitwise_match_oracle_dequant() {
        use crate::backend::Backend as _;
        let be = MetalBackend::new().unwrap();
        let (n, k) = (77usize, 512usize);
        let mut seed = 0x44_4B5E_ED01u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        // Q4_K: d | dmin | scales[12] | qs[128]
        let mut raw = Vec::with_capacity(n * k / 256 * 144);
        for _ in 0..n * k / 256 {
            raw.extend_from_slice(&half::f16::from_f32(0.02).to_bits().to_le_bytes());
            raw.extend_from_slice(&half::f16::from_f32(0.005).to_bits().to_le_bytes());
            for _ in 0..12 {
                raw.push((next() & 0xFF) as u8);
            }
            for _ in 0..128 {
                raw.push((next() & 0xFF) as u8);
            }
        }
        let w = be
            .from_quantized(&raw, vec![n, k], crate::dtype::WeightFormat::Q4_K)
            .unwrap();
        // Единичная A: m = k = 512 (m>8 → mm-тайл), A[i][j] = i==j
        let mut a = vec![0f32; k * k];
        for i in 0..k {
            a[i * k + i] = 1.0;
        }
        let av = be.from_cpu_f32(&a, vec![k, k]).unwrap();
        let c = be.zeros(vec![k, n]).unwrap();
        be.ctx
            .gemm_nt_quantized(av.inner_metal(), w.inner_metal(), c.inner_metal())
            .unwrap();
        let got = be.to_cpu_f32(&c).unwrap();
        // Эталонный dequantize_q4_K: just2 + «d/16, mask 0xF0» для нечётных
        // подблоков; округление веса в half.
        let mut checked = 0usize;
        let mut zeros = 0usize;
        for r in 0..n {
            let sb0 = r * (k / 256) * 144;
            for sb in 0..k / 256 {
                let b = &raw[sb0 + sb * 144..sb0 + (sb + 1) * 144];
                let d = half::f16::from_le_bytes([b[0], b[1]]);
                let dmin = half::f16::from_le_bytes([b[2], b[3]]);
                let scales = &b[4..16];
                let qs = &b[16..144];
                let just2 = |j: usize, kk: usize| -> (f32, f32) {
                    if j < 4 {
                        (
                            (scales[j + kk] & 63) as f32,
                            (scales[j + 4 + kk] & 63) as f32,
                        )
                    } else {
                        (
                            ((scales[j + 4 + kk] & 0xF) | ((scales[j - 4 + kk] & 0xC0) >> 2))
                                as f32,
                            ((scales[j + 4 + kk] >> 4) | ((scales[j + kk] & 0xC0) >> 2)) as f32,
                        )
                    }
                };
                for il in 0..16usize {
                    let is = (il / 4) * 2;
                    let part = il & 1;
                    let c4 = il / 4;
                    let (sc, mn) = just2(is, (il & 3) / 2);
                    let d_eff = if (il & 3) < 2 {
                        half::f16::to_f32(d)
                    } else {
                        half::f16::to_f32(half::f16::from_f32(half::f16::to_f32(d) / 16.0))
                    };
                    let mask: u8 = if (il & 3) < 2 { 0x0F } else { 0xF0 };
                    let dl = d_eff * sc;
                    let ml = half::f16::to_f32(dmin) * mn;
                    for i in 0..16usize {
                        let byte = qs[c4 * 32 + 16 * part + i];
                        let pos = sb * 256 + (il / 2) * 32 + 16 * part + i;
                        let want = half::f16::from_f32(dl * (byte & mask) as f32 - ml);
                        let g = got[pos * n + r];
                        if want.to_bits() & 0x7FFF == 0 {
                            zeros += 1;
                        }
                        assert_eq!(
                            g.to_bits(),
                            half::f16::to_f32(want).to_bits(),
                            "r={r} k={pos}: mm {} vs эталонный half {} (ila={il})",
                            g,
                            half::f16::to_f32(want)
                        );
                        checked += 1;
                    }
                }
            }
        }
        eprintln!("q4_k_mm: побитово сверено {checked} весов (из них нулей {zeros})");
    }

    /// Q5_K mm (m>8; gdn.qkv/w_out 27B) — ПОБИТОВАЯ сверка с эталонным
    /// dequantize_q5_K (dequantize.h:450): та же just2-паковка scales, но
    /// деквант иной — помимо d/16/0xF0 для нечётных подблоков добавляются
    /// qh-биты: `dl*((q & mask) + (qh & ul ? 16|256 : 0)) - ml`, ul = 1<<(il/2).
    /// Единичные активации извлекают каждый half-вес из mm-тайла побитово.
    #[test]
    fn q5_k_mm_weights_bitwise_match_oracle_dequant() {
        use crate::backend::Backend as _;
        let be = MetalBackend::new().unwrap();
        let (n, k) = (77usize, 512usize);
        let mut seed = 0x55_4B5E_ED02u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        // Q5_K: d | dmin | scales[12] | qh[32] | qs[128]
        let mut raw = Vec::with_capacity(n * k / 256 * 176);
        for _ in 0..n * k / 256 {
            raw.extend_from_slice(&half::f16::from_f32(0.02).to_bits().to_le_bytes());
            raw.extend_from_slice(&half::f16::from_f32(0.005).to_bits().to_le_bytes());
            for _ in 0..12 {
                raw.push((next() & 0xFF) as u8);
            }
            for _ in 0..32 {
                raw.push((next() & 0xFF) as u8);
            }
            for _ in 0..128 {
                raw.push((next() & 0xFF) as u8);
            }
        }
        let w = be
            .from_quantized(&raw, vec![n, k], crate::dtype::WeightFormat::Q5_K)
            .unwrap();
        let mut a = vec![0f32; k * k];
        for i in 0..k {
            a[i * k + i] = 1.0;
        }
        let av = be.from_cpu_f32(&a, vec![k, k]).unwrap();
        let c = be.zeros(vec![k, n]).unwrap();
        be.ctx
            .gemm_nt_quantized(av.inner_metal(), w.inner_metal(), c.inner_metal())
            .unwrap();
        let got = be.to_cpu_f32(&c).unwrap();
        let mut checked = 0usize;
        for r in 0..n {
            let sb0 = r * (k / 256) * 176;
            for sb in 0..k / 256 {
                let b = &raw[sb0 + sb * 176..sb0 + (sb + 1) * 176];
                let d = half::f16::from_le_bytes([b[0], b[1]]);
                let dmin = half::f16::from_le_bytes([b[2], b[3]]);
                let scales = &b[4..16];
                let qh_base = &b[16..48];
                let qs = &b[48..176];
                let just2 = |j: usize, kk: usize| -> (f32, f32) {
                    if j < 4 {
                        (
                            (scales[j + kk] & 63) as f32,
                            (scales[j + 4 + kk] & 63) as f32,
                        )
                    } else {
                        (
                            ((scales[j + 4 + kk] & 0xF) | ((scales[j - 4 + kk] & 0xC0) >> 2))
                                as f32,
                            ((scales[j + 4 + kk] >> 4) | ((scales[j + kk] & 0xC0) >> 2)) as f32,
                        )
                    }
                };
                for il in 0..16usize {
                    let is = (il / 4) * 2;
                    let part = il & 1;
                    let c4 = il / 4;
                    let i3 = il & 3;
                    let ul = 1u8 << (il / 2);
                    let (sc, mn) = just2(is, i3 / 2);
                    let dh = half::f16::to_f32(d);
                    let d_eff = if i3 < 2 { dh } else { dh / 16.0 };
                    let mask: u8 = if i3 < 2 { 0x0F } else { 0xF0 };
                    let qh_val = if i3 < 2 { 16f32 } else { 256f32 };
                    let dl = d_eff * sc;
                    let ml = half::f16::to_f32(dmin) * mn;
                    for i in 0..16usize {
                        let qb = qs[c4 * 32 + 16 * part + i];
                        let hb = qh_base[16 * part + i];
                        let add = if hb & ul != 0 { qh_val } else { 0.0 };
                        let pos = sb * 256 + (il / 2) * 32 + 16 * part + i;
                        let want = half::f16::from_f32(dl * ((qb & mask) as f32 + add) - ml);
                        let g = got[pos * n + r];
                        assert_eq!(
                            g.to_bits(),
                            half::f16::to_f32(want).to_bits(),
                            "r={r} k={pos}: q5_k mm {} vs эталонный half {} (il={il})",
                            g,
                            half::f16::to_f32(want)
                        );
                        checked += 1;
                    }
                }
            }
        }
        eprintln!("q5_k_mm: побитово сверено {checked} весов");
    }

    /// Q6_K mm (m>8; lm_head 27B) — ПОБИТОВАЯ сверка с эталонным
    /// dequantize_q6_K (dequantize.h:474): ql/qh-битовая сборка (kmask1/2,
    /// shr/shl) и масштабы dl0..dl3 = dl0/256^k; блок 210 Б
    /// (ql[128] | qh[64] | scales[16 int8] | d). Единичные активации
    /// извлекают каждый half-вес из mm-тайла побитово.
    #[test]
    fn q6_k_mm_weights_bitwise_match_oracle_dequant() {
        use crate::backend::Backend as _;
        let be = MetalBackend::new().unwrap();
        let (n, k) = (77usize, 512usize);
        let mut seed = 0x66_4B5E_ED03u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        // Q6_K: ql[128] | qh[64] | scales[16 int8] | d(f16)
        let mut raw = Vec::with_capacity(n * k / 256 * 210);
        for _ in 0..n * k / 256 {
            for _ in 0..192 {
                raw.push((next() & 0xFF) as u8);
            }
            for _ in 0..16 {
                raw.push(((next() & 0x3F) as i8 - 32) as u8);
            }
            raw.extend_from_slice(&half::f16::from_f32(0.01).to_bits().to_le_bytes());
        }
        let w = be
            .from_quantized(&raw, vec![n, k], crate::dtype::WeightFormat::Q6_K)
            .unwrap();
        let mut a = vec![0f32; k * k];
        for i in 0..k {
            a[i * k + i] = 1.0;
        }
        let av = be.from_cpu_f32(&a, vec![k, k]).unwrap();
        let c = be.zeros(vec![k, n]).unwrap();
        be.ctx
            .gemm_nt_quantized(av.inner_metal(), w.inner_metal(), c.inner_metal())
            .unwrap();
        let got = be.to_cpu_f32(&c).unwrap();
        let rd_u16 = |b: &[u8], off: usize| -> u32 { (b[off] as u32) | ((b[off + 1] as u32) << 8) };
        let mut checked = 0usize;
        for r in 0..n {
            let sb0 = r * (k / 256) * 210;
            for sb in 0..k / 256 {
                let b = &raw[sb0 + sb * 210..sb0 + (sb + 1) * 210];
                let d_all = half::f16::to_f32(half::f16::from_le_bytes([b[208], b[209]]));
                for il in 0..16usize {
                    // эталонные uint16-смещения → байтовые
                    let ql_off = 64 * (il / 8) + 32 * ((il / 2) & 1) + 16 * (il & 1);
                    let qh_off = 128 + 32 * (il / 8) + 16 * (il & 1);
                    let sc = (b[192 + (il % 2) + 2 * (il / 2)] as i8) as f32;
                    let i2 = (il / 2) & 3;
                    let kmask1: u32 = if i2 > 1 {
                        if i2 > 2 { 0xC0C0_C0C0 } else { 0x3030_3030 }
                    } else if i2 > 0 {
                        0x0C0C_0C0C
                    } else {
                        0x0303_0303
                    };
                    let kmask2: u32 = if i2 > 1 { 0xF0F0_F0F0 } else { 0x0F0F_0F0F };
                    let ml = d_all * sc * 32.0;
                    let dl0 = d_all * sc;
                    let dl1 = dl0 / 256.0;
                    let dl2 = dl0 / (256.0 * 256.0);
                    let dl3 = dl0 / (256.0 * 256.0 * 256.0);
                    let shr_h: u32 = if i2 > 2 { 2 } else { 0 };
                    let shl_h: u32 = if i2 > 1 {
                        0
                    } else if i2 > 0 {
                        2
                    } else {
                        4
                    };
                    let shr_l: u32 = if i2 > 1 { 4 } else { 0 };
                    for i in 0..4usize {
                        let low = (rd_u16(b, ql_off + 4 * i)
                            | (rd_u16(b, ql_off + 4 * i + 2) << 16))
                            & kmask2;
                        let high = (rd_u16(b, qh_off + 4 * i)
                            | (rd_u16(b, qh_off + 4 * i + 2) << 16))
                            & kmask1;
                        let q = ((high << shl_h) >> shr_h) | (low >> shr_l);
                        let dl = [dl0, dl1, dl2, dl3];
                        for (j, dlj) in dl.iter().enumerate() {
                            let masked = (q & (0xFFu32 << (8 * j))) as f32;
                            let want = half::f16::from_f32(dlj * masked - ml);
                            let pos = sb * 256 + (il / 2) * 32 + 16 * (il & 1) + 4 * i + j;
                            let g = got[pos * n + r];
                            assert_eq!(
                                g.to_bits(),
                                half::f16::to_f32(want).to_bits(),
                                "r={r} k={pos}: q6_k mm {} vs эталонный half {} (il={il}, i={i}, j={j})",
                                g,
                                half::f16::to_f32(want)
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
        eprintln!("q6_k_mm: побитово сверено {checked} весов");
    }

    /// IQ4_XS mm (m>8; осн. веса 27B) — ПОБИТОВАЯ сверка с эталонным
    /// dequantize_iq4_xs (dequantize.h:699): блок 136 Б
    /// (d | scales_h | scales_l[4] | qs[128]), 6-битный ls → d·(ls−32),
    /// кодбук kvalues_iq4nl (16 нелинейных значений), ниббл il2-половины.
    #[test]
    fn iq4_xs_mm_weights_bitwise_match_oracle_dequant() {
        use crate::backend::Backend as _;
        let be = MetalBackend::new().unwrap();
        let (n, k) = (77usize, 512usize);
        let mut seed = 0x1945_EED1u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        // IQ4_XS: d(f16) | scales_h(u16) | scales_l[4] | qs[128]
        let mut raw = Vec::with_capacity(n * k / 256 * 136);
        for _ in 0..n * k / 256 {
            raw.extend_from_slice(&half::f16::from_f32(0.98).to_bits().to_le_bytes());
            raw.extend_from_slice(&(next() as u16).to_le_bytes());
            for _ in 0..4 {
                raw.push((next() & 0xFF) as u8);
            }
            for _ in 0..128 {
                raw.push((next() & 0xFF) as u8);
            }
        }
        let w = be
            .from_quantized(&raw, vec![n, k], crate::dtype::WeightFormat::Iq4Xs)
            .unwrap();
        let mut a = vec![0f32; k * k];
        for i in 0..k {
            a[i * k + i] = 1.0;
        }
        let av = be.from_cpu_f32(&a, vec![k, k]).unwrap();
        let c = be.zeros(vec![k, n]).unwrap();
        be.ctx
            .gemm_nt_quantized(av.inner_metal(), w.inner_metal(), c.inner_metal())
            .unwrap();
        let got = be.to_cpu_f32(&c).unwrap();
        const KV: [f32; 16] = [
            -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0,
            69.0, 89.0, 113.0,
        ];
        let mut checked = 0usize;
        for r in 0..n {
            let sb0 = r * (k / 256) * 136;
            for sb in 0..k / 256 {
                let b = &raw[sb0 + sb * 136..sb0 + (sb + 1) * 136];
                let d = half::f16::to_f32(half::f16::from_le_bytes([b[0], b[1]]));
                let scales_h = (b[2] as u32) | ((b[3] as u32) << 8);
                for ib32 in 0..8usize {
                    let ls = ((((b[4 + ib32 / 2] >> (4 * (ib32 % 2))) & 0xF) as u32)
                        | (((scales_h >> (2 * ib32)) & 3) << 4))
                        as i32
                        - 32;
                    let dm = d * ls as f32;
                    for off in 0..32usize {
                        let il2 = off / 16;
                        let ii = (off % 16) / 4;
                        let j = off % 4;
                        let qoff = 8 + 16 * ib32 + 4 * ii;
                        let q4 = (b[qoff] as u32)
                            | ((b[qoff + 1] as u32) << 8)
                            | ((b[qoff + 2] as u32) << 16)
                            | ((b[qoff + 3] as u32) << 24);
                        let aux = (q4 >> (4 * il2 as u32)) & 0x0f0f_0f0f;
                        let nib = (aux >> (8 * j as u32)) & 0xF;
                        let want = half::f16::from_f32(dm * KV[nib as usize]);
                        let pos = sb * 256 + ib32 * 32 + off;
                        let g = got[pos * n + r];
                        // Знак нуля после mma не определён (−0 + +0 = +0):
                        // у IQ4_XS ls=0 даёт весь подблок ±0, сравниваем
                        // нули по модулю, остальное — побитово.
                        let wf = half::f16::to_f32(want);
                        let ok = if wf == 0.0 && g == 0.0 {
                            true
                        } else {
                            g.to_bits() == wf.to_bits()
                        };
                        assert!(
                            ok,
                            "r={r} k={pos}: iq4_xs mm {} vs эталонный half {} (ib32={ib32}, off={off})",
                            g, wf
                        );
                        checked += 1;
                    }
                }
            }
        }
        eprintln!("iq4_xs_mm: побитово сверено {checked} весов");
    }

    /// Приоритет путей окна IQ4_XS по m: m=2..5 — БАТЧЕВЫЙ matvec (mb),
    /// m=6..8 — mm-тайл. Замеры 8B (мс): mb 29/44/64/87 против mm 87 на
    /// m=2..5; на m=6..8 mm 87-89 против mb 122/102/122. Ошибка этого
    /// захода: до перестановки mm стоял первым с порогом по умолчанию 2, и
    /// окна γ=1..4 (m=2..5) шли тайлом 32 строк (m=2 стоил 3×). Пин: трейт
    /// побитово равен прямому mv_iq4_xs_mb на m 2..5 и НЕ равен на m 6..8.
    #[test]
    fn trait_gemm_nt_iq4_xs_window_routes_by_m() {
        use crate::backend::Backend as _;
        let be = MetalBackend::new().expect("MetalBackend");
        let (n, k) = (77usize, 512usize);
        let mut seed = 0x1A4B_5150u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut raw = Vec::with_capacity(n * k / 256 * 136);
        for _ in 0..n * k / 256 {
            raw.extend_from_slice(&half::f16::from_f32(0.98).to_bits().to_le_bytes());
            raw.extend_from_slice(&(next() as u16).to_le_bytes());
            for _ in 0..4 {
                raw.push((next() & 0xFF) as u8);
            }
            for _ in 0..128 {
                raw.push((next() & 0xFF) as u8);
            }
        }
        let w = be
            .from_quantized(&raw, vec![n, k], crate::dtype::WeightFormat::Iq4Xs)
            .unwrap();
        for (m, want_mb) in [(2usize, true), (4, true), (5, true), (6, false), (8, false)] {
            let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 7919.0).sin()).collect();
            let a = be.from_cpu_f32(&a_data, vec![m, k]).unwrap();
            be.ctx.set_quant_mm_min(2);
            let c_new = be.zeros(vec![m, n]).unwrap();
            be.gemm_nt(&a, &w, &c_new).unwrap();
            let got = be.to_cpu_f32(&c_new).unwrap();
            let c_mb = be.zeros(vec![m, n]).unwrap();
            be.ctx
                .mv_iq4_xs_mb(a.inner_metal(), w.inner_metal(), c_mb.inner_metal(), k, n)
                .unwrap();
            let mb = be.to_cpu_f32(&c_mb).unwrap();
            let same = got
                .iter()
                .zip(mb.iter())
                .all(|(g, m)| g.to_bits() == m.to_bits());
            assert_eq!(
                same,
                want_mb,
                "m={m}: ожидался {}путь окна IQ4_XS",
                if want_mb { "mb-" } else { "НЕ mb (mm) " }
            );
        }
    }

    /// IQ2_XS / IQ2_S / IQ3_S / IQ1_S / IQ1_M: канонические нулевые блоки,
    /// mb vs хост (m 2..8), mm побитово (единичная A).
    #[test]
    fn iq_rest_paths_match_host() {
        use crate::backend::Backend as _;
        // Канонические: нулевые блоки с d=1 (кроме IQ1_M — там d в scales).
        let mut b = half::f16::from_f32(1.0).to_bits().to_le_bytes().to_vec();
        b.extend(std::iter::repeat_n(0u8, 72));
        let d = crate::weights_map::dequant_iq2_xs(&b);
        assert!(d.iter().all(|v| *v == 1.0), "IQ2_XS: нулевой блок");
        let mut b = half::f16::from_f32(1.0).to_bits().to_le_bytes().to_vec();
        b.extend(std::iter::repeat_n(0u8, 80));
        let d = crate::weights_map::dequant_iq2_s(&b);
        assert!(d.iter().all(|v| *v == 1.0), "IQ2_S: нулевой блок");
        let mut b = half::f16::from_f32(1.0).to_bits().to_le_bytes().to_vec();
        b.extend(std::iter::repeat_n(0u8, 108));
        let d = crate::weights_map::dequant_iq3_s(&b);
        assert!(d.iter().all(|v| *v == 1.0), "IQ3_S: нулевой блок");
        // IQ1_S: q=0 → dl=1, delta=+0.125, grid[0] байт 0xFF = int8 −1 → −0.875
        let mut b = half::f16::from_f32(1.0).to_bits().to_le_bytes().to_vec();
        b.extend(std::iter::repeat_n(0u8, 48));
        let d = crate::weights_map::dequant_iq1_s(&b);
        assert!(d.iter().all(|v| *v == -0.875), "IQ1_S: −1 + 0.125 = −0.875");
        // IQ1_M: scale_u16 = 0 → d = 0 → все нули
        let b = vec![0u8; 56];
        let d = crate::weights_map::dequant_iq1_m(&b);
        assert!(d.iter().all(|v| *v == 0.0), "IQ1_M: d=0");

        let be = MetalBackend::new().expect("MetalBackend");
        let mut seed = 0x44_5853_0003u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let cases = [
            (crate::dtype::WeightFormat::Iq2Xs, 74usize, 0usize),
            (crate::dtype::WeightFormat::Iq2S, 82, 0),
            (crate::dtype::WeightFormat::Iq3S, 110, 0),
            (crate::dtype::WeightFormat::Iq1S, 50, 0),
            (crate::dtype::WeightFormat::Iq1M, 56, 48),
        ];
        for (fmt, bb, scale_off) in cases {
            let (n, k) = (77usize, 256usize);
            let mut raw = Vec::with_capacity(n * k / 256 * bb);
            for _ in 0..n * k / 256 {
                for _ in 0..bb {
                    raw.push((next() & 0xFF) as u8);
                }
                let base = raw.len() - bb;
                if fmt == crate::dtype::WeightFormat::Iq1M {
                    // scale_u16 собирается из всех 8 байт scales — случайные
                    // s1..s3 давали f16-NaN. s0 = 0xC000 → scale_u16 = 0xC000
                    // (d = −2), поля sc = 0 → dl = d.
                    for b in 0..8 {
                        raw[base + 48 + b] = 0;
                    }
                    raw[base + 49] = 0xC0;
                } else {
                    raw[base..base + 2]
                        .copy_from_slice(&half::f16::from_f32(0.02).to_bits().to_le_bytes());
                }
                let _ = scale_off;
            }
            let w = be.from_quantized(&raw, vec![n, k], fmt).unwrap();
            let bf = crate::weights_map::dequant_f32(&crate::weights_map::WeightTensor {
                name: "w".into(),
                shape: vec![n as u64, k as u64],
                format: fmt,
                offset: 0,
                data: &raw,
            });
            for m in 2..=8usize {
                let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 104729.0).sin()).collect();
                let av = be.from_cpu_f32(&a, vec![m, k]).unwrap();
                let c = be.zeros(vec![m, n]).unwrap();
                be.ctx
                    .mv_tern_mb(av.inner_metal(), w.inner_metal(), c.inner_metal())
                    .unwrap();
                let got = be.to_cpu_f32(&c).unwrap();
                for r in 0..n {
                    for mi in 0..m {
                        let want: f64 = bf[r * k..(r + 1) * k]
                            .iter()
                            .zip(&a[mi * k..(mi + 1) * k])
                            .map(|(w, x)| (*w as f64) * (*x as f64))
                            .sum();
                        let rel = (got[mi * n + r] as f64 - want).abs() / (1.0 + want.abs());
                        assert!(rel <= 1e-3, "{fmt:?} mb m={m} r={r}: Δ {rel}");
                    }
                }
            }
            let mut a = vec![0f32; k * k];
            for i in 0..k {
                a[i * k + i] = 1.0;
            }
            let av = be.from_cpu_f32(&a, vec![k, k]).unwrap();
            let c = be.zeros(vec![k, n]).unwrap();
            be.ctx
                .gemm_nt_quantized(av.inner_metal(), w.inner_metal(), c.inner_metal())
                .unwrap();
            let got = be.to_cpu_f32(&c).unwrap();
            for r in 0..n {
                for kk in 0..k {
                    let want = half::f16::from_f32(bf[r * k + kk]);
                    let g = got[kk * n + r];
                    let wf = half::f16::to_f32(want);
                    let ok = if wf == 0.0 && g == 0.0 {
                        true
                    } else {
                        g.to_bits() == wf.to_bits()
                    };
                    assert!(ok, "{fmt:?} mm r={r} k={kk}: {g} vs {wf}");
                }
            }
        }
    }

    /// IQ3_XXS (98 Б/256): канонический нулевой блок (grid[0]=0x04040404,
    /// signs=0, db=0.25 → 1.0), mb vs хост, mm побитово, маршрут.
    #[test]
    fn iq3_xxs_paths_match_host() {
        use crate::backend::Backend as _;
        let mut b = half::f16::from_f32(1.0).to_bits().to_le_bytes().to_vec();
        b.extend(std::iter::repeat_n(0u8, 96));
        assert_eq!(b.len(), 98);
        let d = crate::weights_map::dequant_iq3_xxs(&b);
        assert_eq!(d.len(), 256);
        // db = d·(0.5 + 0)·0.5 = 0.25, grid[0] = 4 → 1.0
        assert!(
            d.iter().all(|v| *v == 1.0),
            "нулевой блок IQ3_XXS должен быть 1.0"
        );

        let be = MetalBackend::new().expect("MetalBackend");
        let (n, k) = (77usize, 256usize);
        let mut seed = 0x33_5853_0002u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut raw = Vec::with_capacity(n * k / 256 * 98);
        for _ in 0..n * k / 256 {
            for _ in 0..98 {
                raw.push((next() & 0xFF) as u8);
            }
            let base = raw.len() - 98;
            raw[base..base + 2].copy_from_slice(&half::f16::from_f32(0.02).to_bits().to_le_bytes());
        }
        let w = be
            .from_quantized(&raw, vec![n, k], crate::dtype::WeightFormat::Iq3Xxs)
            .unwrap();
        let bf = crate::weights_map::dequant_f32(&crate::weights_map::WeightTensor {
            name: "w".into(),
            shape: vec![n as u64, k as u64],
            format: crate::dtype::WeightFormat::Iq3Xxs,
            offset: 0,
            data: &raw,
        });
        for m in 2..=8usize {
            let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 104729.0).sin()).collect();
            let av = be.from_cpu_f32(&a, vec![m, k]).unwrap();
            let c = be.zeros(vec![m, n]).unwrap();
            be.ctx
                .mv_tern_mb(av.inner_metal(), w.inner_metal(), c.inner_metal())
                .unwrap();
            let got = be.to_cpu_f32(&c).unwrap();
            for r in 0..n {
                for mi in 0..m {
                    let want: f64 = bf[r * k..(r + 1) * k]
                        .iter()
                        .zip(&a[mi * k..(mi + 1) * k])
                        .map(|(w, x)| (*w as f64) * (*x as f64))
                        .sum();
                    let rel = (got[mi * n + r] as f64 - want).abs() / (1.0 + want.abs());
                    assert!(rel <= 1e-3, "iq3_xxs mb m={m} r={r}: Δ {rel}");
                }
            }
        }
        let mut a = vec![0f32; k * k];
        for i in 0..k {
            a[i * k + i] = 1.0;
        }
        let av = be.from_cpu_f32(&a, vec![k, k]).unwrap();
        let c = be.zeros(vec![k, n]).unwrap();
        be.ctx
            .gemm_nt_quantized(av.inner_metal(), w.inner_metal(), c.inner_metal())
            .unwrap();
        let got = be.to_cpu_f32(&c).unwrap();
        for r in 0..n {
            for kk in 0..k {
                let want = half::f16::from_f32(bf[r * k + kk]);
                let g = got[kk * n + r];
                let wf = half::f16::to_f32(want);
                let ok = if wf == 0.0 && g == 0.0 {
                    true
                } else {
                    g.to_bits() == wf.to_bits()
                };
                assert!(ok, "iq3_xxs mm r={r} k={kk}: {g} vs {wf}");
            }
        }
    }

    /// IQ2_XXS (66 Б/256): канонический блок, mb vs хост, mm побитово,
    /// маршрут. Нулевые qs → aux32 = 0 → db = d·0.25, grid[0] = 8·0x08,
    /// signs = 0 → все значения = d·2 (при d=1 → 1.0).
    #[test]
    fn iq2_xxs_paths_match_host() {
        use crate::backend::Backend as _;
        let mut b = half::f16::from_f32(1.0).to_bits().to_le_bytes().to_vec();
        b.extend(std::iter::repeat_n(0u8, 64));
        assert_eq!(b.len(), 66);
        let d = crate::weights_map::dequant_iq2_xxs(&b);
        assert_eq!(d.len(), 256);
        assert!(
            d.iter().all(|v| *v == 1.0),
            "нулевой блок IQ2_XXS должен быть 1.0"
        );

        let be = MetalBackend::new().expect("MetalBackend");
        let (n, k) = (77usize, 256usize);
        let mut seed = 0x22_5853_0001u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut raw = Vec::with_capacity(n * k / 256 * 66);
        for _ in 0..n * k / 256 {
            for _ in 0..66 {
                raw.push((next() & 0xFF) as u8);
            }
            let base = raw.len() - 66;
            raw[base..base + 2].copy_from_slice(&half::f16::from_f32(0.02).to_bits().to_le_bytes());
        }
        let w = be
            .from_quantized(&raw, vec![n, k], crate::dtype::WeightFormat::Iq2Xxs)
            .unwrap();
        let bf = crate::weights_map::dequant_f32(&crate::weights_map::WeightTensor {
            name: "w".into(),
            shape: vec![n as u64, k as u64],
            format: crate::dtype::WeightFormat::Iq2Xxs,
            offset: 0,
            data: &raw,
        });
        for m in 2..=8usize {
            let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 104729.0).sin()).collect();
            let av = be.from_cpu_f32(&a, vec![m, k]).unwrap();
            let c = be.zeros(vec![m, n]).unwrap();
            be.ctx
                .mv_tern_mb(av.inner_metal(), w.inner_metal(), c.inner_metal())
                .unwrap();
            let got = be.to_cpu_f32(&c).unwrap();
            for r in 0..n {
                for mi in 0..m {
                    let want: f64 = bf[r * k..(r + 1) * k]
                        .iter()
                        .zip(&a[mi * k..(mi + 1) * k])
                        .map(|(w, x)| (*w as f64) * (*x as f64))
                        .sum();
                    let rel = (got[mi * n + r] as f64 - want).abs() / (1.0 + want.abs());
                    assert!(rel <= 1e-3, "iq2_xxs mb m={m} r={r}: Δ {rel}");
                }
            }
        }
        let mut a = vec![0f32; k * k];
        for i in 0..k {
            a[i * k + i] = 1.0;
        }
        let av = be.from_cpu_f32(&a, vec![k, k]).unwrap();
        let c = be.zeros(vec![k, n]).unwrap();
        be.ctx
            .gemm_nt_quantized(av.inner_metal(), w.inner_metal(), c.inner_metal())
            .unwrap();
        let got = be.to_cpu_f32(&c).unwrap();
        for r in 0..n {
            for kk in 0..k {
                let want = half::f16::from_f32(bf[r * k + kk]);
                let g = got[kk * n + r];
                let wf = half::f16::to_f32(want);
                let ok = if wf == 0.0 && g == 0.0 {
                    true
                } else {
                    g.to_bits() == wf.to_bits()
                };
                assert!(ok, "iq2_xxs mm r={r} k={kk}: {g} vs {wf}");
            }
        }
        // маршруты
        for m in [4usize, 16] {
            let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 7919.0).sin()).collect();
            let av = be.from_cpu_f32(&a, vec![m, k]).unwrap();
            let c = be.zeros(vec![m, n]).unwrap();
            be.gemm_nt(&av, &w, &c).unwrap();
            let got = be.to_cpu_f32(&c).unwrap();
            let cd = be.zeros(vec![m, n]).unwrap();
            if m <= 8 {
                be.ctx
                    .mv_tern_mb(av.inner_metal(), w.inner_metal(), cd.inner_metal())
                    .unwrap();
            } else {
                be.ctx
                    .gemm_nt_quantized(av.inner_metal(), w.inner_metal(), cd.inner_metal())
                    .unwrap();
            }
            let want = be.to_cpu_f32(&cd).unwrap();
            assert!(
                got.iter()
                    .zip(want.iter())
                    .all(|(g, w)| g.to_bits() == w.to_bits()),
                "iq2_xxs m={m}: маршрут сломан"
            );
        }
    }

    /// Q1_0 / Q2_0 / TQ1_0 / TQ2_0: канонические блоки, mb vs хост,
    /// mm побитово (единичная A), маршруты.
    #[test]
    fn q1_0_q2_0_tq_paths_match_host() {
        use crate::backend::Backend as _;
        // Q1_0: d=1, qs[0]=1 → y[0]=+1, y[1]=−1
        let mut b = half::f16::from_f32(1.0).to_bits().to_le_bytes().to_vec();
        let mut qs = vec![0u8; 16];
        qs[0] = 1;
        b.extend_from_slice(&qs);
        let d = crate::weights_map::dequant_q1_0(&b);
        assert_eq!(d[0], 1.0);
        assert_eq!(d[1], -1.0);
        // Q2_0: qs[0]=0b11100100 → q 0..3 → −1,0,1,2
        let mut b = half::f16::from_f32(1.0).to_bits().to_le_bytes().to_vec();
        let mut qs = vec![0u8; 16];
        qs[0] = 0b1110_0100;
        b.extend_from_slice(&qs);
        let d = crate::weights_map::dequant_q2_0(&b);
        assert_eq!(&d[0..4], &[-1.0, 0.0, 1.0, 2.0]);
        // TQ2_0: d=1, qs[0]=0b11100100 → позиции 0/32/64/96 = −1/0/1/2
        let mut b = vec![0u8; 64];
        b[0] = 0b1110_0100;
        b.extend_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        let d = crate::weights_map::dequant_tq2_0(&b);
        assert_eq!(d[0], -1.0);
        assert_eq!(d[32], 0.0);
        assert_eq!(d[64], 1.0);
        assert_eq!(d[96], 2.0);
        // TQ1_0: d=1, qs[0]=128 → n=0: q=128, xi=(384)>>8=1 → 0;
        // n=1 (позиция 32): q=(128·3)&0xFF=128, xi=1 → 0.
        let mut b = vec![0u8; 52];
        b[0] = 128;
        b.extend_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        assert_eq!(b.len(), 54);
        let d = crate::weights_map::dequant_tq1_0(&b);
        assert_eq!(d[0], 0.0);
        assert_eq!(d[32], 0.0);

        let be = MetalBackend::new().expect("MetalBackend");
        let mut seed = 0x71_71_71_71u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for (fmt, bb, blk) in [
            (crate::dtype::WeightFormat::Q1_0, 18usize, 128usize),
            (crate::dtype::WeightFormat::Q2_0, 18, 64),
            (crate::dtype::WeightFormat::TQ1_0, 54, 256),
            (crate::dtype::WeightFormat::TQ2_0, 66, 256),
        ] {
            let (n, k) = (77usize, blk);
            let mut raw = Vec::with_capacity(n * k / blk * bb);
            for _ in 0..n * k / blk {
                for _ in 0..bb {
                    raw.push((next() & 0xFF) as u8);
                }
                let base = raw.len() - bb;
                let doff = match fmt {
                    crate::dtype::WeightFormat::Q1_0 | crate::dtype::WeightFormat::Q2_0 => 0,
                    crate::dtype::WeightFormat::TQ1_0 => 52,
                    _ => 64,
                };
                raw[base + doff..base + doff + 2]
                    .copy_from_slice(&half::f16::from_f32(0.02).to_bits().to_le_bytes());
            }
            let w = be.from_quantized(&raw, vec![n, k], fmt).unwrap();
            let bf = crate::weights_map::dequant_f32(&crate::weights_map::WeightTensor {
                name: "w".into(),
                shape: vec![n as u64, k as u64],
                format: fmt,
                offset: 0,
                data: &raw,
            });
            for m in 2..=8usize {
                let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 104729.0).sin()).collect();
                let av = be.from_cpu_f32(&a, vec![m, k]).unwrap();
                let c = be.zeros(vec![m, n]).unwrap();
                be.ctx
                    .mv_tern_mb(av.inner_metal(), w.inner_metal(), c.inner_metal())
                    .unwrap();
                let got = be.to_cpu_f32(&c).unwrap();
                for r in 0..n {
                    for mi in 0..m {
                        let want: f64 = bf[r * k..(r + 1) * k]
                            .iter()
                            .zip(&a[mi * k..(mi + 1) * k])
                            .map(|(w, x)| (*w as f64) * (*x as f64))
                            .sum();
                        let rel = (got[mi * n + r] as f64 - want).abs() / (1.0 + want.abs());
                        assert!(rel <= 1e-3, "{fmt:?} mb m={m} r={r}: Δ {rel}");
                    }
                }
            }
            let mut a = vec![0f32; k * k];
            for i in 0..k {
                a[i * k + i] = 1.0;
            }
            let av = be.from_cpu_f32(&a, vec![k, k]).unwrap();
            let c = be.zeros(vec![k, n]).unwrap();
            be.ctx
                .gemm_nt_quantized(av.inner_metal(), w.inner_metal(), c.inner_metal())
                .unwrap();
            let got = be.to_cpu_f32(&c).unwrap();
            for r in 0..n {
                for kk in 0..k {
                    let want = half::f16::from_f32(bf[r * k + kk]);
                    let g = got[kk * n + r];
                    let wf = half::f16::to_f32(want);
                    let ok = if wf == 0.0 && g == 0.0 {
                        true
                    } else {
                        g.to_bits() == wf.to_bits()
                    };
                    assert!(ok, "{fmt:?} mm r={r} k={kk}: {g} vs {wf}");
                }
            }
        }
    }

    /// Q4_1 / Q5_1 / Q8_1 / Q8_K: канонические блоки, mb vs хост, mm
    /// побитово (единичная A), маршруты.
    #[test]
    fn q4_1_q5_1_q8_1_q8_k_paths_match_host() {
        use crate::backend::Backend as _;
        // Q4_1: d=1, m=0.5, qs[0]=0x21 → y[0]=2.5, y[16]=1.5
        let mut b = half::f16::from_f32(1.0).to_bits().to_le_bytes().to_vec();
        b.extend_from_slice(&half::f16::from_f32(0.5).to_bits().to_le_bytes());
        let mut qs = vec![0u8; 16];
        qs[0] = 0x21;
        b.extend_from_slice(&qs);
        let d = crate::weights_map::dequant_q4_1(&b);
        assert_eq!(d[0], 1.5); // младший ниббл 1 → 1·d + m
        assert_eq!(d[16], 2.5); // старший ниббл 2
        // Q5_1: qh=1 (бит0) → y[0]=(1+16)+0.5=17.5, y[16]=2+0.5=2.5
        let mut b = half::f16::from_f32(1.0).to_bits().to_le_bytes().to_vec();
        b.extend_from_slice(&half::f16::from_f32(0.5).to_bits().to_le_bytes());
        b.extend_from_slice(&1u32.to_le_bytes());
        let mut qs = vec![0u8; 16];
        qs[0] = 0x21;
        b.extend_from_slice(&qs);
        let d = crate::weights_map::dequant_q5_1(&b);
        assert_eq!(d[0], 17.5);
        assert_eq!(d[16], 2.5);
        // Q8_1: d=1, qs[0]=-3 → -3
        let mut b = half::f16::from_f32(1.0).to_bits().to_le_bytes().to_vec();
        b.extend_from_slice(&half::f16::from_f32(0.0).to_bits().to_le_bytes());
        let mut qs = vec![0u8; 32];
        qs[0] = (-3i8) as u8;
        b.extend_from_slice(&qs);
        assert_eq!(b.len(), 36);
        let d = crate::weights_map::dequant_q8_1(&b);
        assert_eq!(d[0], -3.0);
        // Q8_K: d=0.5, qs[0]=-4 → -2
        let mut b = 0.5f32.to_le_bytes().to_vec();
        let mut qs = vec![0u8; 256];
        qs[0] = (-4i8) as u8;
        b.extend_from_slice(&qs);
        b.extend_from_slice(&[0u8; 32]); // bsums[16] int16
        assert_eq!(b.len(), 292);
        let d = crate::weights_map::dequant_q8_k(&b);
        assert_eq!(d[0], -2.0);

        let be = MetalBackend::new().expect("MetalBackend");
        let mut seed = 0x51_51_51_51u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for (fmt, bb, blk) in [
            (crate::dtype::WeightFormat::Q4_1, 20usize, 32usize),
            (crate::dtype::WeightFormat::Q5_1, 24, 32),
            (crate::dtype::WeightFormat::Q8_1, 36, 32),
            (crate::dtype::WeightFormat::Q8_K, 292, 256),
        ] {
            let (n, k) = (77usize, blk);
            let mut raw = Vec::with_capacity(n * k / blk * bb);
            for _ in 0..n * k / blk {
                for _ in 0..bb {
                    raw.push((next() & 0xFF) as u8);
                }
                let base = raw.len() - bb;
                match fmt {
                    crate::dtype::WeightFormat::Q4_1 | crate::dtype::WeightFormat::Q5_1 => {
                        raw[base..base + 2]
                            .copy_from_slice(&half::f16::from_f32(0.02).to_bits().to_le_bytes());
                        raw[base + 2..base + 4]
                            .copy_from_slice(&half::f16::from_f32(0.005).to_bits().to_le_bytes());
                    }
                    crate::dtype::WeightFormat::Q8_1 => {
                        raw[base..base + 2]
                            .copy_from_slice(&half::f16::from_f32(0.02).to_bits().to_le_bytes());
                        raw[base + 2..base + 4]
                            .copy_from_slice(&half::f16::from_f32(0.0).to_bits().to_le_bytes());
                    }
                    _ => {
                        raw[base..base + 4].copy_from_slice(&0.02f32.to_le_bytes());
                        // bsums не трогаем — деквант их не читает
                    }
                }
            }
            let w = be.from_quantized(&raw, vec![n, k], fmt).unwrap();
            let bf = crate::weights_map::dequant_f32(&crate::weights_map::WeightTensor {
                name: "w".into(),
                shape: vec![n as u64, k as u64],
                format: fmt,
                offset: 0,
                data: &raw,
            });
            for m in 2..=8usize {
                let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 104729.0).sin()).collect();
                let av = be.from_cpu_f32(&a, vec![m, k]).unwrap();
                let c = be.zeros(vec![m, n]).unwrap();
                if fmt == crate::dtype::WeightFormat::Q8_1
                    || fmt == crate::dtype::WeightFormat::Q8_K
                {
                    be.ctx
                        .mv_q8_1_q8_k_mb(av.inner_metal(), w.inner_metal(), c.inner_metal())
                        .unwrap();
                } else {
                    be.ctx
                        .mv_q4_1_q5_1_mb(av.inner_metal(), w.inner_metal(), c.inner_metal())
                        .unwrap();
                }
                let got = be.to_cpu_f32(&c).unwrap();
                for r in 0..n {
                    for mi in 0..m {
                        let want: f64 = bf[r * k..(r + 1) * k]
                            .iter()
                            .zip(&a[mi * k..(mi + 1) * k])
                            .map(|(w, x)| (*w as f64) * (*x as f64))
                            .sum();
                        let rel = (got[mi * n + r] as f64 - want).abs() / (1.0 + want.abs());
                        assert!(rel <= 1e-3, "{fmt:?} mb m={m} r={r}: Δ {rel}");
                    }
                }
            }
            // mm побитово
            let mut a = vec![0f32; k * k];
            for i in 0..k {
                a[i * k + i] = 1.0;
            }
            let av = be.from_cpu_f32(&a, vec![k, k]).unwrap();
            let c = be.zeros(vec![k, n]).unwrap();
            be.ctx
                .gemm_nt_quantized(av.inner_metal(), w.inner_metal(), c.inner_metal())
                .unwrap();
            let got = be.to_cpu_f32(&c).unwrap();
            for r in 0..n {
                for kk in 0..k {
                    let want = half::f16::from_f32(bf[r * k + kk]);
                    let g = got[kk * n + r];
                    let wf = half::f16::to_f32(want);
                    let ok = if wf == 0.0 && g == 0.0 {
                        true
                    } else {
                        g.to_bits() == wf.to_bits()
                    };
                    assert!(ok, "{fmt:?} mm r={r} k={kk}: {g} vs {wf}");
                }
            }
        }
    }

    /// Q2_K / Q3_K: канонические блоки, mb vs хост (m 2..8), mm побитово
    /// (единичная A), маршруты. Раскладка: 16 подблоков по 16; shift =
    /// 2·((s%8)/2), half = s%2, q += 32 на 8 подблоков.
    #[test]
    fn q2_k_q3_k_paths_match_host() {
        use crate::backend::Backend as _;
        // Q2_K: d=1.0, dmin=0.5, scales[0]=0x10 (dl=0, ml=0.5), qs[0]=3 →
        // y[0] = 0·3 − 0.5 = −0.5; подблок 1 (scales[1]=0) → −0.0.
        let mut b2 = vec![0u8; 16];
        b2[0] = 0x10;
        b2.extend(std::iter::repeat_n(0u8, 64));
        b2.extend_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        b2.extend_from_slice(&half::f16::from_f32(0.5).to_bits().to_le_bytes());
        b2[16] = 3; // qs[0]
        assert_eq!(b2.len(), 84);
        let d = crate::weights_map::dequant_q2_k(&b2);
        assert_eq!(d[0], -0.5);
        assert_eq!(d[15], -0.5);
        assert_eq!(d[16], -0.0);
        // Q3_K: d=1.0, scales[0]=1 (dl = 1·(1−32) = −31), hmask=0 → sub=4,
        // qs[0]=1 → y[0] = −31·(1−4) = 93; qs[1]=0 → 124.
        let mut b3 = vec![0u8; 110];
        b3[96] = 1; // scales[0]
        b3[32] = 1; // qs[0]
        b3[108..110].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        let d = crate::weights_map::dequant_q3_k(&b3);
        assert_eq!(d[0], 93.0);
        assert_eq!(d[1], 124.0);

        let be = MetalBackend::new().expect("MetalBackend");
        let (n, k) = (77usize, 256usize);
        let mut seed = 0x22_4B51_4B33u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for (fmt, bb) in [
            (crate::dtype::WeightFormat::Q2_K, 84usize),
            (crate::dtype::WeightFormat::Q3_K, 110usize),
        ] {
            let mut raw = Vec::with_capacity(n * k / 256 * bb);
            for _ in 0..n * k / 256 {
                for _ in 0..bb {
                    raw.push((next() & 0xFF) as u8);
                }
                // sane d/dmin (иначе f16-мусор даёт NaN/Inf и тест теряет
                // смысл): Q2_K — d@80, dmin@82; Q3_K — d@108 (scales[12]@96).
                let base = raw.len() - bb;
                if bb == 84 {
                    raw[base + 80..base + 82]
                        .copy_from_slice(&half::f16::from_f32(0.02).to_bits().to_le_bytes());
                    raw[base + 82..base + 84]
                        .copy_from_slice(&half::f16::from_f32(0.005).to_bits().to_le_bytes());
                } else {
                    raw[base + 108..base + 110]
                        .copy_from_slice(&half::f16::from_f32(0.02).to_bits().to_le_bytes());
                }
            }
            let w = be.from_quantized(&raw, vec![n, k], fmt).unwrap();
            let bf = crate::weights_map::dequant_f32(&crate::weights_map::WeightTensor {
                name: "w".into(),
                shape: vec![n as u64, k as u64],
                format: fmt,
                offset: 0,
                data: &raw,
            });
            for m in 2..=8usize {
                let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 104729.0).sin()).collect();
                let av = be.from_cpu_f32(&a, vec![m, k]).unwrap();
                let c = be.zeros(vec![m, n]).unwrap();
                be.ctx
                    .mv_q2_k_q3_k_mb(av.inner_metal(), w.inner_metal(), c.inner_metal())
                    .unwrap();
                let got = be.to_cpu_f32(&c).unwrap();
                for r in 0..n {
                    for mi in 0..m {
                        let want: f64 = bf[r * k..(r + 1) * k]
                            .iter()
                            .zip(&a[mi * k..(mi + 1) * k])
                            .map(|(w, x)| (*w as f64) * (*x as f64))
                            .sum();
                        let rel = (got[mi * n + r] as f64 - want).abs() / (1.0 + want.abs());
                        assert!(rel <= 1e-3, "{fmt:?} mb m={m} r={r}: Δ {rel}");
                    }
                }
            }
            // mm побитово
            let mut a = vec![0f32; k * k];
            for i in 0..k {
                a[i * k + i] = 1.0;
            }
            let av = be.from_cpu_f32(&a, vec![k, k]).unwrap();
            let c = be.zeros(vec![k, n]).unwrap();
            be.ctx
                .gemm_nt_quantized(av.inner_metal(), w.inner_metal(), c.inner_metal())
                .unwrap();
            let got = be.to_cpu_f32(&c).unwrap();
            for r in 0..n {
                for kk in 0..k {
                    let want = half::f16::from_f32(bf[r * k + kk]);
                    let g = got[kk * n + r];
                    let wf = half::f16::to_f32(want);
                    let ok = if wf == 0.0 && g == 0.0 {
                        true
                    } else {
                        g.to_bits() == wf.to_bits()
                    };
                    assert!(ok, "{fmt:?} mm r={r} k={kk}: {g} vs {wf}");
                }
            }
            // маршруты
            for m in [4usize, 16] {
                let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 7919.0).sin()).collect();
                let av = be.from_cpu_f32(&a, vec![m, k]).unwrap();
                let c = be.zeros(vec![m, n]).unwrap();
                be.gemm_nt(&av, &w, &c).unwrap();
                let got = be.to_cpu_f32(&c).unwrap();
                let cd = be.zeros(vec![m, n]).unwrap();
                if m <= 8 {
                    be.ctx
                        .mv_q2_k_q3_k_mb(av.inner_metal(), w.inner_metal(), cd.inner_metal())
                        .unwrap();
                } else {
                    be.ctx
                        .gemm_nt_quantized(av.inner_metal(), w.inner_metal(), cd.inner_metal())
                        .unwrap();
                }
                let want = be.to_cpu_f32(&cd).unwrap();
                assert!(
                    got.iter()
                        .zip(want.iter())
                        .all(|(g, w)| g.to_bits() == w.to_bits()),
                    "{fmt:?} m={m}: маршрут сломан"
                );
            }
        }
    }

    /// NVFP4 (блок 64: 4 UE4M3-скейла на подблоки 16 + qs[32], E2M1):
    /// канонический блок, mb vs хост-деквант, mm побитово (единичная A),
    /// маршруты. Ниже — IQ4_NL/MXFP4 тест того же вида.
    #[test]
    fn nvfp4_paths_match_host() {
        use crate::backend::Backend as _;
        // UE4M3 0x40 → 1.0; qs[0]=0x21 → позиции 0 и 8 подблока 0.
        let mut blk = vec![0x40u8, 0, 0, 0];
        let mut qs = vec![0u8; 32];
        qs[0] = 0x21;
        blk.extend_from_slice(&qs);
        let d = crate::weights_map::dequant_nvfp4(&blk);
        assert_eq!(d[0], 1.0);
        assert_eq!(d[8], 2.0);
        assert_eq!(d[1], 0.0);

        let be = MetalBackend::new().expect("MetalBackend");
        let fmt = crate::dtype::WeightFormat::Nvfp4;
        let (n, k) = (77usize, 128usize);
        let mut seed = 0x4E56_4650u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut raw = Vec::with_capacity(n * k / 64 * 36);
        for _ in 0..n * k / 64 {
            for _ in 0..4 {
                raw.push(120 + (next() % 40) as u8); // UE4M3 в разумном диапазоне
            }
            for _ in 0..32 {
                raw.push((next() & 0xFF) as u8);
            }
        }
        let w = be.from_quantized(&raw, vec![n, k], fmt).unwrap();
        let bf = crate::weights_map::dequant_f32(&crate::weights_map::WeightTensor {
            name: "w".into(),
            shape: vec![n as u64, k as u64],
            format: fmt,
            offset: 0,
            data: &raw,
        });
        // mb vs точный хост-деквант
        for m in 2..=8usize {
            let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 104729.0).sin()).collect();
            let av = be.from_cpu_f32(&a, vec![m, k]).unwrap();
            let c = be.zeros(vec![m, n]).unwrap();
            be.ctx
                .mv_nvfp4_mb(av.inner_metal(), w.inner_metal(), c.inner_metal())
                .unwrap();
            let got = be.to_cpu_f32(&c).unwrap();
            for r in 0..n {
                for mi in 0..m {
                    let want: f64 = bf[r * k..(r + 1) * k]
                        .iter()
                        .zip(&a[mi * k..(mi + 1) * k])
                        .map(|(w, x)| (*w as f64) * (*x as f64))
                        .sum();
                    let rel = (got[mi * n + r] as f64 - want).abs() / (1.0 + want.abs());
                    assert!(rel <= 1e-3, "nvfp4 mb m={m} r={r}: Δ {rel}");
                }
            }
        }
        // mm побитово (единичная A)
        let mut a = vec![0f32; k * k];
        for i in 0..k {
            a[i * k + i] = 1.0;
        }
        let av = be.from_cpu_f32(&a, vec![k, k]).unwrap();
        let c = be.zeros(vec![k, n]).unwrap();
        be.ctx
            .gemm_nt_quantized(av.inner_metal(), w.inner_metal(), c.inner_metal())
            .unwrap();
        let got = be.to_cpu_f32(&c).unwrap();
        for r in 0..n {
            for kk in 0..k {
                let want = half::f16::from_f32(bf[r * k + kk]);
                let g = got[kk * n + r];
                let wf = half::f16::to_f32(want);
                let ok = if wf == 0.0 && g == 0.0 {
                    true
                } else {
                    g.to_bits() == wf.to_bits()
                };
                assert!(ok, "nvfp4 mm r={r} k={kk}: {g} vs {wf}");
            }
        }
        // маршруты: m=4 → mb, m=16 → mm
        for m in [4usize, 16] {
            let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 7919.0).sin()).collect();
            let av = be.from_cpu_f32(&a, vec![m, k]).unwrap();
            let c = be.zeros(vec![m, n]).unwrap();
            be.gemm_nt(&av, &w, &c).unwrap();
            let got = be.to_cpu_f32(&c).unwrap();
            let cd = be.zeros(vec![m, n]).unwrap();
            if m <= 8 {
                be.ctx
                    .mv_nvfp4_mb(av.inner_metal(), w.inner_metal(), cd.inner_metal())
                    .unwrap();
            } else {
                be.ctx
                    .gemm_nt_quantized(av.inner_metal(), w.inner_metal(), cd.inner_metal())
                    .unwrap();
            }
            let want = be.to_cpu_f32(&cd).unwrap();
            assert!(
                got.iter()
                    .zip(want.iter())
                    .all(|(g, w)| g.to_bits() == w.to_bits()),
                "nvfp4 m={m}: маршрут сломан"
            );
        }
    }

    /// IQ4_NL / MXFP4: форматы добавлены 2026-10-18 (полная матрица квантов).
    /// Блоки: IQ4_NL 18 Б (f16 d + 16 Б нибблов, d·KV_IQ4), MXFP4 17 Б
    /// (E8M0 + 16 Б, KV_FP4·2^(e−128)), раскладка нибблов как у Q4_0.
    /// Проверки: (1) канонические ручные блоки; (2) mb (окна) vs точный
    /// хост-деквант; (3) mm (m>8) ПОБИТОВО через единичную A; (4) маршрут.
    #[test]
    fn iq4_nl_mxfp4_paths_match_host() {
        use crate::backend::Backend as _;
        // (1) Канонический IQ4_NL: d=1.0, байт0=0x12 → w[0]=KV[2]=-83,
        // w[16]=KV[1]=-104. MXFP4: e=127 (d=2^-1), qs[0]=0x21 → w[0]=KV[1]*0.5=0.5,
        // w[16]=KV[2]*0.5=1.0.
        let mut nl = half::f16::from_f32(1.0).to_bits().to_le_bytes().to_vec();
        let mut qs = vec![0u8; 16];
        qs[0] = 0x12;
        nl.extend_from_slice(&qs);
        let d = crate::weights_map::dequant_iq4_nl(&nl);
        assert_eq!(d[0], -83.0);
        assert_eq!(d[16], -104.0);
        let mut fp = vec![127u8];
        let mut qs = vec![0u8; 16];
        qs[0] = 0x21;
        fp.extend_from_slice(&qs);
        let d = crate::weights_map::dequant_mxfp4(&fp);
        assert_eq!(d[0], 0.5);
        assert_eq!(d[16], 1.0);

        let be = MetalBackend::new().expect("MetalBackend");
        let (n, k) = (77usize, 128usize);
        let mut seed = 0x4E4C_5150u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for (fmt, bb) in [
            (crate::dtype::WeightFormat::Iq4Nl, 18usize),
            (crate::dtype::WeightFormat::Mxfp4, 17),
        ] {
            let mut raw = Vec::with_capacity(n * k / 32 * bb);
            for _ in 0..n * k / 32 {
                if bb == 18 {
                    raw.extend_from_slice(&half::f16::from_f32(0.02).to_bits().to_le_bytes());
                } else {
                    raw.push(120 + (next() % 16) as u8);
                }
                for _ in 0..16 {
                    raw.push((next() & 0xFF) as u8);
                }
            }
            let w = be.from_quantized(&raw, vec![n, k], fmt).unwrap();
            let bf = crate::weights_map::dequant_f32(&crate::weights_map::WeightTensor {
                name: "w".into(),
                shape: vec![n as u64, k as u64],
                format: fmt,
                offset: 0,
                data: &raw,
            });
            // (2) mb (окна) vs точный хост-деквант
            for m in 2..=8usize {
                let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 104729.0).sin()).collect();
                let av = be.from_cpu_f32(&a, vec![m, k]).unwrap();
                let c = be.zeros(vec![m, n]).unwrap();
                be.ctx
                    .mv_iq4_nl_mxfp4_mb(av.inner_metal(), w.inner_metal(), c.inner_metal())
                    .unwrap();
                let got = be.to_cpu_f32(&c).unwrap();
                for r in 0..n {
                    for mi in 0..m {
                        let want: f64 = bf[r * k..(r + 1) * k]
                            .iter()
                            .zip(&a[mi * k..(mi + 1) * k])
                            .map(|(w, x)| (*w as f64) * (*x as f64))
                            .sum();
                        let rel = (got[mi * n + r] as f64 - want).abs() / (1.0 + want.abs());
                        assert!(rel <= 1e-3, "{fmt:?} mb m={m} r={r}: Δ {rel}");
                    }
                }
            }
            // (3) mm (m>8) побитово: единичная A извлекает half-веса
            let mut a = vec![0f32; k * k];
            for i in 0..k {
                a[i * k + i] = 1.0;
            }
            let av = be.from_cpu_f32(&a, vec![k, k]).unwrap();
            let c = be.zeros(vec![k, n]).unwrap();
            be.ctx
                .gemm_nt_quantized(av.inner_metal(), w.inner_metal(), c.inner_metal())
                .unwrap();
            let got = be.to_cpu_f32(&c).unwrap();
            for r in 0..n {
                for kk in 0..k {
                    let want = half::f16::from_f32(bf[r * k + kk]);
                    let g = got[kk * n + r];
                    // знак нуля после mma не определён
                    let wf = half::f16::to_f32(want);
                    let ok = if wf == 0.0 && g == 0.0 {
                        true
                    } else {
                        g.to_bits() == wf.to_bits()
                    };
                    assert!(ok, "{fmt:?} mm r={r} k={kk}: {g} vs {wf}");
                }
            }
            // (4) маршрут: m=4 → mb (не mm), m=16 → mm
            for (m, want_mb) in [(4usize, true), (16, false)] {
                let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 7919.0).sin()).collect();
                let av = be.from_cpu_f32(&a, vec![m, k]).unwrap();
                let c = be.zeros(vec![m, n]).unwrap();
                be.gemm_nt(&av, &w, &c).unwrap();
                let got = be.to_cpu_f32(&c).unwrap();
                let cm = be.zeros(vec![m, n]).unwrap();
                if m <= 8 {
                    be.ctx
                        .mv_iq4_nl_mxfp4_mb(av.inner_metal(), w.inner_metal(), cm.inner_metal())
                        .unwrap();
                    let want = be.to_cpu_f32(&cm).unwrap();
                    assert_eq!(
                        got.iter()
                            .zip(want.iter())
                            .all(|(g, w)| g.to_bits() == w.to_bits()),
                        want_mb,
                        "{fmt:?} m={m}: маршрут mb сломан"
                    );
                } else {
                    let am = av.inner_metal();
                    let _ = am;
                    // mm: сравнение с прямым gemm_nt_quantized
                    let cd = be.zeros(vec![m, n]).unwrap();
                    be.ctx
                        .gemm_nt_quantized(av.inner_metal(), w.inner_metal(), cd.inner_metal())
                        .unwrap();
                    let want = be.to_cpu_f32(&cd).unwrap();
                    assert!(
                        got.iter()
                            .zip(want.iter())
                            .all(|(g, w)| g.to_bits() == w.to_bits()),
                        "{fmt:?} m={m}: трейт не совпал с quantized-путём"
                    );
                }
            }
        }
    }

    /// IQ4_XS mul_mm (m>8): сверка тайлового GEMM с хост-деквантом. Ловит
    /// раскладку il/x по суперблокам и попадание весов в K-позиции тайла.
    #[test]
    fn iq4_xs_mm_matches_host_dequant() {
        use crate::backend::Backend as _;
        let be = MetalBackend::new().unwrap();
        let (m, n, k) = (16usize, 64usize, 512usize); // m>8 → gemm_nt_iq4_xs_mm
        let mut seed = 0x1234_5678u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut raw = Vec::with_capacity(n * k / 256 * 136);
        for _ in 0..n * k / 256 {
            raw.extend_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
            raw.extend_from_slice(&(next() as u16).to_le_bytes());
            for _ in 0..4 {
                raw.push((next() & 0xFF) as u8);
            }
            for _ in 0..128 {
                raw.push((next() & 0xFF) as u8);
            }
        }
        let w = be
            .from_quantized(&raw, vec![n, k], crate::dtype::WeightFormat::Iq4Xs)
            .unwrap();
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.03)
            .collect();
        // активации в тайле mul_mm — тоже half
        let ah: Vec<f32> = a
            .iter()
            .map(|&x| half::f16::to_f32(half::f16::from_f32(x)))
            .collect();
        let av = be.from_cpu_f32(&a, vec![m, k]).unwrap();
        let c = be.zeros(vec![m, n]).unwrap();
        be.ctx
            .gemm_nt_quantized(av.inner_metal(), w.inner_metal(), c.inner_metal())
            .unwrap();
        let got = be.to_cpu_f32(&c).unwrap();
        // Эталон — как mul_mm: деквант f32 → округление веса в half (тайл
        // sa хранит half) → f64-dot. Тогда остаётся лишь разница порядка
        // накопления; структурный баг раскладки il/x дал бы ≫1e-4.
        let rows: Vec<Vec<f32>> = (0..n)
            .map(|r| {
                crate::weights_map::dequant_iq4_xs(
                    &raw[r * (k / 256) * 136..(r + 1) * (k / 256) * 136],
                )
                .into_iter()
                .map(|x| half::f16::to_f32(half::f16::from_f32(x)))
                .collect()
            })
            .collect();
        let mut worst = 0f64;
        for mi in 0..m {
            for ni in 0..n {
                let want: f64 = rows[ni]
                    .iter()
                    .zip(&ah[mi * k..mi * k + k])
                    .map(|(w, x)| (*w as f64) * (*x as f64))
                    .sum();
                let d = (got[mi * n + ni] as f64 - want).abs();
                let rel = d / (1.0 + want.abs());
                if rel > worst {
                    worst = rel;
                }
                assert!(
                    rel <= 2e-4,
                    "m={mi} n={ni}: {} vs {want} (Δ {d})",
                    got[mi * n + ni]
                );
            }
        }
        eprintln!("iq4_xs_mm worst rel {worst:e}");
    }

    /// m>8 → gemm_nt_q6_k_mm (порт kernel_mul_mm_q6_K_f32). Эталон — как
    /// mul_mm: деквант f32 → округление веса и активаций в half (тайлы
    /// sa/sb) → f64-dot; остаётся разница порядка накопления. Раскладку
    /// il/x (210-байтный блок, ql/qh/scales/d@208) баг сдвинул бы на ≫1e-4.
    #[test]
    fn q6_k_mm_matches_host_dequant() {
        use crate::backend::Backend as _;
        let be = MetalBackend::new().unwrap();
        let (m, n, k) = (16usize, 64usize, 512usize); // m>8 → mm-путь
        let src: Vec<f32> = (0..n * k)
            .map(|i| (i as f64 * 7919.0_f64).sin() as f32 * 0.5)
            .collect();
        let raw = enc_q6_k(&src);
        let w = be
            .from_quantized(&raw, vec![n, k], crate::dtype::WeightFormat::Q6_K)
            .unwrap();
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.03)
            .collect();
        let ah: Vec<f32> = a
            .iter()
            .map(|&x| half::f16::to_f32(half::f16::from_f32(x)))
            .collect();
        let av = be.from_cpu_f32(&a, vec![m, k]).unwrap();
        let c = be.zeros(vec![m, n]).unwrap();
        be.ctx
            .gemm_nt_quantized(av.inner_metal(), w.inner_metal(), c.inner_metal())
            .unwrap();
        let got = be.to_cpu_f32(&c).unwrap();
        let rows: Vec<Vec<f32>> = (0..n)
            .map(|r| {
                crate::weights_map::dequant_q6_k(
                    &raw[r * (k / 256) * 210..(r + 1) * (k / 256) * 210],
                )
                .into_iter()
                .map(|x| half::f16::to_f32(half::f16::from_f32(x)))
                .collect()
            })
            .collect();
        let mut worst = 0f64;
        for mi in 0..m {
            for ni in 0..n {
                let want: f64 = rows[ni]
                    .iter()
                    .zip(&ah[mi * k..mi * k + k])
                    .map(|(w, x)| (*w as f64) * (*x as f64))
                    .sum();
                let d = (got[mi * n + ni] as f64 - want).abs();
                let rel = d / (1.0 + want.abs());
                if rel > worst {
                    worst = rel;
                }
                assert!(
                    rel <= 2e-4,
                    "m={mi} n={ni}: {} vs {want} (Δ {d})",
                    got[mi * n + ni]
                );
            }
        }
        eprintln!("q6_k_mm worst rel {worst:e}");
    }

    fn fixture(name: &str) -> Vec<u8> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join(name);
        std::fs::read(&path).unwrap_or_else(|e| panic!("fixture {name}: {e}"))
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
            let d = if smax > 0.0 {
                smax / (31.0 * 127.0)
            } else {
                0.0
            };
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
            .map(|i| (i as f32 * 7919.0).sin() * 0.5)
            .collect();
        // Q8_0 128×96 (синтетический): полные тайлы tiled-GEMM (64 по N,
        // 32 по M) и K из нескольких блоков — фикстура 64×32 их не задевает.
        let q8_big: Vec<u8> = (0..128 * 96 / 32)
            .flat_map(|b| {
                let mut blk = half::f16::from_f32(0.01 + (b % 7) as f32 * 0.003)
                    .to_bits()
                    .to_le_bytes()
                    .to_vec();
                blk.extend((0..32).map(|j| {
                    let q = ((b * 37 + j * 11) % 255) as u8;
                    q.wrapping_sub(127) as i8 as u8
                }));
                blk
            })
            .collect();
        vec![
            (q8_big, vec![128, 96], GpuDtype::Q8_0, 0.05),
            (enc_q4_k(&src), vec![8, 256], GpuDtype::Q4_K, 0.15),
            (enc_q6_k(&src), vec![8, 256], GpuDtype::Q6_K, 0.06),
            // Q5_K/Q5_0 — наш энкодер (порт ref llama.cpp) + CPU-деквант;
            // сверка GPU-ядла с dequant_f32 в quantized_mv/gemm.
            (
                crate::quant::quantize(&src, crate::quant::QuantFormat::Q5_K),
                vec![8, 256],
                GpuDtype::Q5_K,
                0.12,
            ),
            (
                crate::quant::quantize(&src, crate::quant::QuantFormat::Q5_0),
                vec![8, 256],
                GpuDtype::Q5_0,
                0.12,
            ),
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
                        GpuDtype::Q5_K => crate::dtype::WeightFormat::Q5_K,
                        GpuDtype::Q5_0 => crate::dtype::WeightFormat::Q5_0,
                        _ => unreachable!(),
                    },
                    offset: 0,
                    data: &bytes,
                }],
                storage: vec![],
            };
            let bf = dequant_f32(&tmp.tensors[0]);
            let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 104729.0).sin()).collect();
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

    /// Память слияния split-K внимания помещается в threadgroup (32 КБ)
    /// при любой длине контекста и hd до 512: превышение Metal без слоя
    /// валидации молча принимал (MTL_DEBUG_LAYER=1 — assertion), это UB.
    #[test]
    fn split_attn_groups_fit_threadgroup_memory() {
        for hd in [64usize, 128, 256, 512] {
            for tkv in [1usize, 63, 64, 1984, 2048, 4096, 100_000] {
                let n = split_attn_groups(tkv, hd);
                assert!(n >= 1);
                assert!(
                    n * (2 + hd) * 4 <= 32 * 1024,
                    "hd={hd} tkv={tkv}: {n} групп"
                );
            }
        }
        assert_eq!(split_attn_groups(4096, 128), 32, "hd=128 не урезается");
    }

    /// GPU-аргмакс строк (окно спека) == CPU greedy_next: включая ничьи
    /// (первый максимум), NaN/−∞ и штраф за повторы.
    #[test]
    fn argmax_rows_matches_cpu() {
        use crate::backend::Backend as _;
        let be = MetalBackend::new().expect("MetalBackend");
        let (t, v) = (5usize, 999usize);
        let mut data: Vec<f32> = (0..t * v)
            .map(|i| ((i as f32 * 0.017).sin() * 3.0).round() / 2.0)
            .collect();
        // Ничья на первом месте: равные максимумы в первой строке
        data[10] = 5.0;
        data[500] = 5.0;
        // NaN и −∞ не должны побеждать
        data[v + 3] = f32::NAN;
        data[2 * v + 7] = f32::NEG_INFINITY;
        let logits = be.from_cpu_f32(&data, vec![t, v]).unwrap();
        for penalty in [1.0f32, 1.1] {
            for hist in [vec![], vec![10u32, 10, 500, 900]] {
                let h = (!hist.is_empty()).then_some((hist.as_slice(), penalty));
                let (ids, lse) = be.argmax_rows(&logits, h).unwrap();
                for r in 0..t {
                    let row = &data[r * v..(r + 1) * v];
                    let want = crate::inference::sampling::greedy_next(row, &hist, penalty);
                    assert_eq!(ids[r], want, "p={penalty} r={r}");
                    // lse — против прямого расчёта по PENALIZED ряду
                    let mut pen = row.to_vec();
                    if let Some((hh, pp)) = h {
                        crate::inference::sampling::apply_repetition_penalty_vec_unique(
                            &mut pen, hh, pp,
                        );
                    }
                    let mx = pen.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let sum: f32 = pen
                        .iter()
                        .filter(|x| x.is_finite())
                        .map(|x| (x - mx).exp())
                        .sum();
                    let want_lse = if mx > f32::NEG_INFINITY {
                        sum.ln()
                    } else {
                        f32::NEG_INFINITY
                    };
                    assert!(
                        (lse[r] - want_lse).abs() < 1e-3
                            || (lse[r].is_infinite() && want_lse.is_infinite()),
                        "lse p={penalty} r={r}: {} vs {want_lse}",
                        lse[r]
                    );
                }
            }
        }
    }

    /// Малый батч Q8_0 (окно верификации спека): M = 2…8 — mv_q8_0_mb
    /// (M = 7, 8 — двумя проходами), N не кратно числу строк на simdgroup,
    /// K — 3 блока (не кратно шагу лейнов). Сверка с CPU и с тайловым путём.
    #[test]
    fn q8_0_small_batch_matches_cpu() {
        let ctx = MetalCtx::new().expect("MetalCtx");
        let dev = &ctx.device;
        let (n, k) = (77usize, 96usize);
        let bytes: Vec<u8> = (0..n * k / 32)
            .flat_map(|b| {
                let mut blk = half::f16::from_f32(0.01 + (b % 7) as f32 * 0.003)
                    .to_bits()
                    .to_le_bytes()
                    .to_vec();
                blk.extend((0..32).map(|j| {
                    let q = ((b * 37 + j * 11) % 255) as u8;
                    q.wrapping_sub(127) as i8 as u8
                }));
                blk
            })
            .collect();
        let b = GpuTensor::from_quantized(dev, &bytes, vec![n, k], GpuDtype::Q8_0).unwrap();
        let tmp = WeightsMap {
            arch: "t".into(),
            tensors: vec![crate::weights_map::WeightTensor {
                name: "w".into(),
                shape: vec![n as u64, k as u64],
                format: crate::dtype::WeightFormat::Q8_0,
                offset: 0,
                data: &bytes,
            }],
            storage: vec![],
        };
        let bf = dequant_f32(&tmp.tensors[0]);
        for m in 2..=8usize {
            let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 104729.0).sin()).collect();
            let c_ref = cpu_ref(&a_data, m, k, &bf, n);
            let a = GpuTensor::from_cpu_f32(dev, &a_data, vec![m, k]).unwrap();
            let c = GpuTensor::zeros(dev, vec![m, n]).unwrap();
            ctx.gemm_nt_quantized(&a, &b, &c).unwrap();
            let got = c.to_cpu_f32(&ctx).unwrap();
            let diff = max_diff(&got, &c_ref);
            assert!(diff <= 1e-3, "M={m}: |Δ| max {diff}");
        }
    }

    /// Маршрут трейтового gemm_nt для Q8_0 при m ∈ [2,8] (окно верификации
    /// MTP/DFlash на Q8_0-моделях) — БАТЧЕВЫЙ mv_q8_0_mb (веса 1× на m строк),
    /// а не построчный mv_ext. Регресс: Ф8.1 (b7d8d4e) завёл ветку «m≤8 →
    /// mv_ext» для всех квантов и молча снял Q8_0-окна с mb-пути (7247854);
    /// замер 4B-Q8_0+DFlash: γ=2 49.7→55.8, γ=3 44.3→61.1 tok/s; приёмка и
    /// текст SPEC==AR не изменились. Пин маршрута: результат трейта равен
    /// ПРЯМОМУ gemm_nt_quantized (его гейт Q8_0-mb не зависит от рычага и
    /// покрыт q8_0_small_batch_matches_cpu против CPU) и ОТЛИЧАЕТСЯ от
    /// mv_ext-варианта (порог 9, как до правки) — иначе тест вырожден.
    #[test]
    fn trait_gemm_nt_q8_0_small_m_goes_batched() {
        use crate::backend::Backend as _;
        let be = MetalBackend::new().expect("MetalBackend");
        // Порог — ИНСТАНС-переопределение: env-рычаг глобальный, и тест с
        // другой его настройкой (iq4_xs: 9) в том же процессе перетирал бы
        // этот маршрут (и наоборот — этот тест ломал бы тот).
        be.ctx.set_quant_mm_min(4);
        let (n, k) = (77usize, 128usize); // k%128==0 — mv_ext вообще применим
        let bytes: Vec<u8> = (0..n * k / 32)
            .flat_map(|b| {
                let mut blk = half::f16::from_f32(0.01 + (b % 7) as f32 * 0.003)
                    .to_bits()
                    .to_le_bytes()
                    .to_vec();
                blk.extend((0..32).map(|j| {
                    let q = ((b * 37 + j * 11) % 255) as u8;
                    q.wrapping_sub(127) as i8 as u8
                }));
                blk
            })
            .collect();
        let w = be
            .from_quantized(&bytes, vec![n, k], crate::dtype::WeightFormat::Q8_0)
            .unwrap();
        for m in 2..=8usize {
            let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 7919.0).sin()).collect();
            let a = be.from_cpu_f32(&a_data, vec![m, k]).unwrap();
            be.ctx.set_quant_mm_min(4);
            let c_new = be.zeros(vec![m, n]).unwrap();
            be.gemm_nt(&a, &w, &c_new).unwrap();
            let got = be.to_cpu_f32(&c_new).unwrap();
            // Прямой вызов gemm_nt_quantized — безусловно mb-путь Q8_0.
            let c_mb = be.zeros(vec![m, n]).unwrap();
            be.ctx
                .gemm_nt_quantized(a.inner_metal(), w.inner_metal(), c_mb.inner_metal())
                .unwrap();
            let want = be.to_cpu_f32(&c_mb).unwrap();
            let mut same = true;
            for (g, wnt) in got.iter().zip(want.iter()) {
                if g.to_bits() != wnt.to_bits() {
                    same = false;
                    break;
                }
            }
            assert!(
                same,
                "M={m}: трейтовый gemm_nt не равен mb-пути (маршрут сломан)"
            );
            // Контроль вырожденности: паритет-порог 9 (окна идут mv_ext,
            // как до правки) обязан дать ДРУГИЕ числа.
            be.ctx.set_quant_mm_min(9);
            let c_old = be.zeros(vec![m, n]).unwrap();
            be.gemm_nt(&a, &w, &c_old).unwrap();
            let old = be.to_cpu_f32(&c_old).unwrap();
            be.ctx.set_quant_mm_min(4);
            let differs = got
                .iter()
                .zip(old.iter())
                .any(|(g, o)| g.to_bits() != o.to_bits());
            assert!(
                differs,
                "M={m}: mb и mv_ext совпали побитово — пин маршрута недействителен"
            );
        }
    }

    /// Генератор Q4_0-блоков (18 Б: f16 d + 16 байт нибблов) для тестов окон.
    fn q4_0_bytes(n: usize, k: usize) -> Vec<u8> {
        (0..n * k / 32)
            .flat_map(|b| {
                let mut blk = half::f16::from_f32(0.01 + (b % 7) as f32 * 0.003)
                    .to_bits()
                    .to_le_bytes()
                    .to_vec();
                blk.extend((0..16).map(|j| {
                    let lo = ((b * 37 + j * 11) % 15) as u8;
                    let hi = ((b * 53 + j * 7) % 15) as u8;
                    lo | (hi << 4)
                }));
                blk
            })
            .collect()
    }

    /// Батчевый матвек Q4_0 (mv_q4_0_mb, окно m=2..8) против ТОЧНОГО
    /// хост-декванта: N не кратно строкам на simdgroup, K=96 (3 блока,
    /// не кратно шагу 4 блоков). m 7/8 — два прохода (4+остаток).
    #[test]
    fn q4_0_small_batch_matches_cpu() {
        let ctx = MetalCtx::new().expect("MetalCtx");
        let dev = &ctx.device;
        let (n, k) = (77usize, 96usize);
        let bytes = q4_0_bytes(n, k);
        let b = GpuTensor::from_quantized(dev, &bytes, vec![n, k], GpuDtype::Q4_0).unwrap();
        let tmp = WeightsMap {
            arch: "t".into(),
            tensors: vec![crate::weights_map::WeightTensor {
                name: "w".into(),
                shape: vec![n as u64, k as u64],
                format: crate::dtype::WeightFormat::Q4_0,
                offset: 0,
                data: &bytes,
            }],
            storage: vec![],
        };
        let bf = dequant_f32(&tmp.tensors[0]);
        for m in 2..=8usize {
            let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 104729.0).sin()).collect();
            let c_ref = cpu_ref(&a_data, m, k, &bf, n);
            let a = GpuTensor::from_cpu_f32(dev, &a_data, vec![m, k]).unwrap();
            let c = GpuTensor::zeros(dev, vec![m, n]).unwrap();
            ctx.mv_q4_0_mb(&a, &b, &c).unwrap();
            let got = c.to_cpu_f32(&ctx).unwrap();
            let diff = max_diff(&got, &c_ref);
            assert!(diff <= 1e-3, "M={m}: |Δ| max {diff}");
        }
    }

    /// Маршрут трейтового gemm_nt для Q4_0 при m ∈ [2,8] — mv_q4_0_mb (веса
    /// 1× на m строк), а не построчный mv_ext: у Q4_0 батчевого пути не было
    /// вовсе (ни mb, ни mm-тайла — m>8 идёт скалярным gemm_nt_q4_0), поэтому
    /// окна Q4_0-моделей читали веса m раз. Пин: трейт == прямой mv_q4_0_mb
    /// (тот покрыт q4_0_small_batch_matches_cpu против CPU) и ≠ mv_ext-пути
    /// (порог 9). Инстанс-порог — без гонки за общим env.
    #[test]
    fn trait_gemm_nt_q4_0_small_m_goes_batched() {
        use crate::backend::Backend as _;
        let be = MetalBackend::new().expect("MetalBackend");
        be.ctx.set_quant_mm_min(4);
        let (n, k) = (77usize, 128usize); // k%128==0 — mv_ext вообще применим
        let bytes = q4_0_bytes(n, k);
        let w = be
            .from_quantized(&bytes, vec![n, k], crate::dtype::WeightFormat::Q4_0)
            .unwrap();
        for m in 2..=8usize {
            let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 7919.0).sin()).collect();
            let a = be.from_cpu_f32(&a_data, vec![m, k]).unwrap();
            be.ctx.set_quant_mm_min(4);
            let c_new = be.zeros(vec![m, n]).unwrap();
            be.gemm_nt(&a, &w, &c_new).unwrap();
            let got = be.to_cpu_f32(&c_new).unwrap();
            let c_mb = be.zeros(vec![m, n]).unwrap();
            be.ctx
                .mv_q4_0_mb(a.inner_metal(), w.inner_metal(), c_mb.inner_metal())
                .unwrap();
            let want = be.to_cpu_f32(&c_mb).unwrap();
            assert!(
                got.iter()
                    .zip(want.iter())
                    .all(|(g, w)| g.to_bits() == w.to_bits()),
                "M={m}: трейтовый gemm_nt не равен mv_q4_0_mb (маршрут сломан)"
            );
            be.ctx.set_quant_mm_min(9);
            let c_old = be.zeros(vec![m, n]).unwrap();
            be.gemm_nt(&a, &w, &c_old).unwrap();
            let old = be.to_cpu_f32(&c_old).unwrap();
            be.ctx.set_quant_mm_min(4);
            assert!(
                got.iter()
                    .zip(old.iter())
                    .any(|(g, o)| g.to_bits() != o.to_bits()),
                "M={m}: mb и mv_ext совпали побитово — пин маршрута недействителен"
            );
        }
    }

    /// m>8 → gemm_nt_q4_0_mm (порт схемы ggml kernel_mul_mm_q4_0; до этого
    /// m>8 шло скалярным gemm_nt_q4_0). Эталон — как mul_mm: деквант f32 →
    /// округление веса и активаций в half (тайлы sa/sb) → f64-dot; остаётся
    /// только разница порядка накопления. Раскладка нибблов (каноническая:
    /// значение j = младший ниббл байта j, j+16 = старший) сдвинула бы
    /// результат на ≫1e-4. m=33/n=77 — хвостовые тайлы M и N (staging-путь).
    #[test]
    fn q4_0_mm_matches_host_dequant() {
        use crate::backend::Backend as _;
        let be = MetalBackend::new().unwrap();
        let (m, n, k) = (33usize, 77usize, 512usize);
        let raw = q4_0_bytes(n, k);
        let w = be
            .from_quantized(&raw, vec![n, k], crate::dtype::WeightFormat::Q4_0)
            .unwrap();
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.03)
            .collect();
        let ah: Vec<f32> = a
            .iter()
            .map(|&x| half::f16::to_f32(half::f16::from_f32(x)))
            .collect();
        let av = be.from_cpu_f32(&a, vec![m, k]).unwrap();
        let c = be.zeros(vec![m, n]).unwrap();
        be.ctx
            .gemm_nt_quantized(av.inner_metal(), w.inner_metal(), c.inner_metal())
            .unwrap();
        let got = be.to_cpu_f32(&c).unwrap();
        let rows: Vec<Vec<f32>> = (0..n)
            .map(|r| {
                crate::weights_map::dequant_q4_0(&raw[r * (k / 32) * 18..(r + 1) * (k / 32) * 18])
                    .into_iter()
                    .map(|x| half::f16::to_f32(half::f16::from_f32(x)))
                    .collect()
            })
            .collect();
        let mut worst = 0f64;
        for mi in 0..m {
            for ni in 0..n {
                let want: f64 = rows[ni]
                    .iter()
                    .zip(&ah[mi * k..mi * k + k])
                    .map(|(w, x)| (*w as f64) * (*x as f64))
                    .sum();
                let d = (got[mi * n + ni] as f64 - want).abs();
                let rel = d / (1.0 + want.abs());
                if rel > worst {
                    worst = rel;
                }
                assert!(
                    rel <= 2e-4,
                    "m={mi} n={ni}: {} vs {want} (Δ {d})",
                    got[mi * n + ni]
                );
            }
        }
        eprintln!("q4_0_mm worst rel {worst:e}");
    }

    /// embedding_quant по всем квант-таблицам (вкл. Q5_K/Q5_0): GPU-декод
    /// строки == CPU dequant_f32 с точностью до f32-умножения.
    #[test]
    fn embedding_quant_matches_cpu() {
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
                        GpuDtype::Q5_K => crate::dtype::WeightFormat::Q5_K,
                        GpuDtype::Q5_0 => crate::dtype::WeightFormat::Q5_0,
                        _ => unreachable!(),
                    },
                    offset: 0,
                    data: &bytes,
                }],
                storage: vec![],
            };
            let bf = dequant_f32(&tmp.tensors[0]);
            let idx = [0u32, (n / 2) as u32, (n - 1) as u32];
            let y = GpuTensor::zeros(dev, vec![1, idx.len(), k]).unwrap();
            ctx.embedding_quant_f32(&idx, &b, &y).unwrap();
            for (t, &row) in idx.iter().enumerate() {
                for d in 0..k {
                    let got = {
                        let cpu = y.to_cpu_f32(&ctx).unwrap();
                        cpu[t * k + d]
                    };
                    let want = bf[row as usize * k + d];
                    let diff = (got - want).abs();
                    let bound = tol * 8.0;
                    assert!(
                        diff <= bound,
                        "{dtype:?} row {row} d {d}: {got} vs {want} (|Δ| {diff} > {bound})"
                    );
                }
            }
        }
    }

    /// Частичный RoPE (GLM-4): кернел `rope_std_part_f32` == host-референс
    /// (Backend::rope_standard_partial по умолчанию) при n_rot < hd, включая
    /// случай n_rot = hd (полный поворот) и хвост без поворота.
    #[test]
    fn rope_partial_matches_host() {
        use crate::backend::{Backend as _, RopePairing};
        let be = MetalBackend::new().expect("MetalBackend");
        let (tq, d, hd) = (3usize, 8usize, 8usize); // 2 головы по 8
        let data: Vec<f32> = (0..tq * d).map(|i| 0.1 + (i as f32) * 0.017).collect();
        let x = be.from_cpu_f32(&data, vec![1, tq, d]).unwrap();
        let theta = 10_000.0f32;
        for pairing in [RopePairing::Halves, RopePairing::Adjacent] {
            for n_rot in [2usize, 4, 8] {
                let got = be
                    .to_cpu_f32(&be.rope_partial(&x, 5, hd, n_rot, theta, pairing).unwrap())
                    .unwrap();
                // host-референс: та же математика независимо от кернела
                let half = n_rot / 2;
                let mut want = data.clone();
                for tt in 0..tq {
                    let p = (5 + tt) as f32;
                    for h in 0..d / hd {
                        let base = tt * d + h * hd;
                        for k in 0..half {
                            let inv = 1.0 / theta.powf(2.0 * k as f32 / n_rot as f32);
                            let (c, sv) = ((p * inv).cos(), (p * inv).sin());
                            let (i0, i1) = match pairing {
                                RopePairing::Halves => (k, k + half),
                                RopePairing::Adjacent => (2 * k, 2 * k + 1),
                            };
                            let (x0, x1) = (data[base + i0], data[base + i1]);
                            want[base + i0] = x0 * c - x1 * sv;
                            want[base + i1] = x0 * sv + x1 * c;
                        }
                    }
                }
                let diff = max_diff(&got, &want);
                assert!(diff <= 1e-5, "{pairing:?} n_rot={n_rot}: |Δ| max {diff}");
            }
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
                        GpuDtype::Q5_K => crate::dtype::WeightFormat::Q5_K,
                        GpuDtype::Q5_0 => crate::dtype::WeightFormat::Q5_0,
                        _ => unreachable!(),
                    },
                    offset: 0,
                    data: &bytes,
                }],
                storage: vec![],
            };
            let bf = dequant_f32(&tmp.tensors[0]);
            let a_data: Vec<f32> = (0..k).map(|i| (i as f32 * 15485863.0).sin()).collect();
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
                GpuTensor::from_cpu_f32(&ctx.device, &vec![0.1f32; 9 * 4096], vec![1, 9, 4096])
                    .unwrap();
            let k36 =
                GpuTensor::from_cpu_f32(&ctx.device, &vec![0.2f32; 36 * 1024], vec![1, 36, 1024])
                    .unwrap();
            let v36 =
                GpuTensor::from_cpu_f32(&ctx.device, &vec![0.3f32; 36 * 1024], vec![1, 36, 1024])
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
                let yarn = crate::rope::RopeYarn::IDENTITY;
                ctx.run("rope_std_f32", grid, group, |enc| {
                    enc.set_buffer(0, Some(&q9.buffer), q9.offset as u64);
                    enc.set_buffer(1, Some(&yr.buffer), yr.offset as u64);
                    for (i, v) in [9usize, 4096, 128, 0usize].iter().enumerate() {
                        enc.set_bytes((2 + i) as u64, 8, (v as *const usize).cast());
                    }
                    enc.set_bytes(6, 4, (&1e6f32 as *const f32).cast());
                    enc.set_bytes(7, 24, (&yarn as *const crate::rope::RopeYarn).cast());
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

#[cfg(test)]
mod glm_attention_tests {
    use super::*;
    use crate::backend::Backend as _;

    /// GQA с БОЛЬШИМ отношением: nh=32 / nkv=2 (hd=128) — геометрия GLM-4.
    /// Проверяем `attn_ar` (Metal) против CPU-референса при tq = 1, 7, 16:
    /// существующие тесты покрывали отношения до 8, а n_rep=16 не встречалось.
    /// KV-кэш f16 (как у модели) с kv_dim = 2·128 = 256: запись и чтение
    /// через view — регрессия на NaN в f16-пути при этой геометрии.
    #[test]
    fn kv_cache_f16_small_kv_dim_roundtrip() {
        use crate::backend::Backend as _;
        let metal = MetalBackend::new().expect("MetalBackend");
        let (nkv, hd) = (2usize, 128usize);
        let kvd = nkv * hd;
        for t in [1usize, 2, 7, 16] {
            let mut cache = crate::nn::KvCache::new(
                &metal,
                nkv,
                hd,
                crate::dtype::KvFormat::F16,
                crate::dtype::KvFormat::F16,
                10_000.0,
                None,
            )
            .unwrap();
            let k_v: Vec<f32> = (0..t * kvd).map(|i| ((i as f32) * 0.011).sin()).collect();
            let v_v: Vec<f32> = (0..t * kvd).map(|i| ((i as f32) * 0.013).cos()).collect();
            let k = metal.from_cpu_f32(&k_v, vec![1, t, kvd]).unwrap();
            let v = metal.from_cpu_f32(&v_v, vec![1, t, kvd]).unwrap();
            cache.update(&metal, &k, &v).unwrap();
            let ck = cache.view();
            let cv = cache.view_v();
            assert_eq!(ck.shape(), vec![1, t, kvd], "t={t}: форма K-вида");
            let ckv = metal.to_cpu_f16(&ck).unwrap();
            let cvv = metal.to_cpu_f16(&cv).unwrap();
            let nf = ckv
                .iter()
                .chain(cvv.iter())
                .filter(|x| !x.is_finite())
                .count();
            assert_eq!(nf, 0, "t={t}: NaN/Inf в f16-кэше ({nf})");
            for i in 0..t * kvd {
                let dk = (ckv[i].to_f32() - k_v[i]).abs();
                let dv = (cvv[i].to_f32() - v_v[i]).abs();
                assert!(dk <= 2e-3 && dv <= 2e-3, "t={t} i={i}: |Δk|={dk} |Δv|={dv}");
            }
        }
    }

    #[test]
    fn attn_ar_large_gqa_ratio_matches_cpu() {
        let metal = MetalBackend::new().expect("MetalBackend");
        let cpu = crate::cpu::CpuBackend;
        let (nh, nkv, hd) = (32usize, 2usize, 128usize);
        let (qd, kvd) = (nh * hd, nkv * hd);
        for tq in [1usize, 7, 16] {
            let tkv = tq;
            let mk = |n: usize, seed: f32| -> Vec<f32> {
                (0..n)
                    .map(|i| ((i as f32 * 0.37 + seed).sin()) * 0.5)
                    .collect()
            };
            let qd_v = mk(tq * qd, 0.1);
            let kd_v = mk(tkv * kvd, 0.7);
            let vd_v = mk(tkv * kvd, 1.3);
            let mask_v: Vec<f32> = (0..tq * tkv)
                .map(|i| {
                    let (i, j) = (i / tkv, i % tkv);
                    if j > i { f32::NEG_INFINITY } else { 0.0 }
                })
                .collect();
            let q = metal.from_cpu_f32(&qd_v, vec![1, tq, qd]).unwrap();
            let k = metal.from_cpu_f32(&kd_v, vec![1, tkv, kvd]).unwrap();
            let v = metal.from_cpu_f32(&vd_v, vec![1, tkv, kvd]).unwrap();
            let m = metal.from_cpu_f32(&mask_v, vec![1, tq, tkv]).unwrap();
            let y = metal.zeros(vec![1, tq, qd]).unwrap();
            metal
                .attn_ar(&q, &k, &v, Some(&m), &y, nh, nkv, hd)
                .unwrap();
            let got = metal.to_cpu_f32(&y).unwrap();
            let qc = cpu.from_cpu_f32(&qd_v, vec![1, tq, qd]).unwrap();
            let kc = cpu.from_cpu_f32(&kd_v, vec![1, tkv, kvd]).unwrap();
            let vc = cpu.from_cpu_f32(&vd_v, vec![1, tkv, kvd]).unwrap();
            let mc = cpu.from_cpu_f32(&mask_v, vec![1, tq, tkv]).unwrap();
            let yc = cpu.zeros(vec![1, tq, qd]).unwrap();
            cpu.attn_ar(&qc, &kc, &vc, Some(&mc), &yc, nh, nkv, hd)
                .unwrap();
            let want = cpu.to_cpu_f32(&yc).unwrap();
            let nf = got.iter().filter(|x| !x.is_finite()).count();
            assert_eq!(nf, 0, "tq={tq}: NaN/Inf в выходе Metal ({nf})");
            let diff = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(diff <= 1e-4, "tq={tq}: |Δ| max {diff}");
            // f16-KV (как в кэше модели): K/V — f16-буферы, тот же attn_ar.
            let k16: Vec<half::f16> = kd_v.iter().map(|&x| half::f16::from_f32(x)).collect();
            let v16: Vec<half::f16> = vd_v.iter().map(|&x| half::f16::from_f32(x)).collect();
            let kf = metal.from_cpu_f16(&k16, vec![1, tkv, kvd]).unwrap();
            let vf = metal.from_cpu_f16(&v16, vec![1, tkv, kvd]).unwrap();
            let yf = metal.zeros(vec![1, tq, qd]).unwrap();
            metal
                .attn_ar(&q, &kf, &vf, Some(&m), &yf, nh, nkv, hd)
                .unwrap();
            let got16 = metal.to_cpu_f32(&yf).unwrap();
            let nf16 = got16.iter().filter(|x| !x.is_finite()).count();
            assert_eq!(nf16, 0, "tq={tq}: NaN/Inf в f16-KV выходе ({nf16})");
            let diff16 = got16
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(diff16 <= 2e-2, "tq={tq}: f16-KV |Δ| max {diff16}");
        }
    }
}
