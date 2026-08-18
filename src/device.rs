//! Low-level per-backend device constructors + VRAM diagnostics.
//!
//! Backend SELECTION (which one to use) lives in [`crate::backend`] — chosen at
//! build time by target platform. This module only holds the constructors the
//! selector calls, plus the VRAM diagnostics the engine pool consults for
//! warm-spare top-up gating.
//!
//! The per-backend constructors are `cfg`-gated: `cuda_device` exists only on
//! non-macOS builds (cuda feature), `metal_device` only on macOS (metal
//! feature). Production code reaches them via `backend::default_device`; they
//! are kept `pub` for tests and direct lower-level use.
//!
//! # VRAM probes
//!
//! [`vram_free_bytes`] / [`vram_total_bytes`] / [`log_pool_stats`] gate the
//! engine pools' warm-spare top-up. `None` means "unknown" and callers MUST
//! fall back to a permissive policy (attempt the build) rather than refuse —
//! that is exactly how `gst-plugin-tts` consumes them.
//!
//! - macOS: total = unified-memory size (`hw.memsize`); free is not exposed
//!   (no per-process free-VRAM API on Metal).
//! - elsewhere: CUDA driver API (`cuMemGetInfo` via cudarc) — real free memory
//!   including the stream pool's free list; NVML probe of device 0 (behind the
//!   `nvml` feature) as fallback when the driver API is unreachable.

use candle_core::Device;

/// CUDA device constructor — non-macOS builds only (cuda feature).
#[cfg(not(target_os = "macos"))]
pub fn cuda_device(ordinal: usize) -> anyhow::Result<Device> {
    Device::new_cuda(ordinal).map_err(|e| anyhow::anyhow!("CUDA device {ordinal}: {e}"))
}

/// Metal device constructor — macOS builds only (metal feature).
#[cfg(target_os = "macos")]
pub fn metal_device(ordinal: usize) -> anyhow::Result<Device> {
    Device::new_metal(ordinal).map_err(|e| anyhow::anyhow!("Metal device {ordinal}: {e}"))
}

/// CPU device. Always available (`Device::Cpu` needs no feature). Used for
/// weight loading and tests only — never as an inference fallback (the crate
/// rule is GPU-only inference).
pub fn cpu_device() -> Device {
    Device::Cpu
}

/// VRAM available for new allocations, if queryable on this backend.
///
/// Non-macOS: uses `cuMemGetInfo` (total free on device) — the REAL free
/// memory that the CUDA driver can give out, INCLUDING memory currently held
/// in the stream pool's free list. The pool's cached chunks ARE available for
/// reuse — `cuMemGetInfo` correctly accounts for them as "free" because
/// they're not pinned; only memory actually in use by live tensors reduces
/// the free count. So the result is accurate for pool-sizing decisions even
/// with warm chunks cached (nvidia-smi instead reports the driver's committed
/// pages, which include pool reservations). Falls back to an NVML probe
/// (`nvml` feature) when the driver API is unreachable.
///
/// macOS: not exposed (no per-process free-VRAM API on Metal) → `None`.
pub fn vram_free_bytes() -> Option<usize> {
    #[cfg(target_os = "macos")]
    {
        // TODO(Metal): query `MTLDevice` — `recommended_max_working_set_size`
        // minus `currentAllocatedSize` (unified memory on Apple Silicon).
        // Needs the candle `MetalDevice` handle accessor, to be confirmed
        // against a running device.
        None
    }
    #[cfg(not(target_os = "macos"))]
    {
        cudarc::driver::result::mem_get_info()
            .ok()
            .map(|(free, _total)| free)
            .or_else(nvml_free_bytes)
    }
}

/// Total device memory in bytes, if queryable. macOS reports the unified
/// memory size; elsewhere the CUDA driver API (CUDA primary context) with an
/// NVML fallback.
pub fn vram_total_bytes() -> Option<usize> {
    #[cfg(target_os = "macos")]
    {
        macos_unified_memory_bytes()
    }
    #[cfg(not(target_os = "macos"))]
    {
        cudarc::driver::result::mem_get_info()
            .ok()
            .map(|(_free, total)| total)
            .or_else(nvml_total_bytes)
    }
}

// ── CUDA memory pool management (non-macOS only; driver API via cudarc) ──────

/// CUDA memory pool statistics for diagnostics.
///
/// - `reserved`: physical bytes held by the pool (includes free-list)
/// - `used`: bytes currently allocated to live tensors
/// - `free_in_pool`: reserved - used = cached chunks available for reuse
#[cfg(not(target_os = "macos"))]
pub struct PoolStats {
    pub reserved: usize,
    pub used: usize,
    pub free_in_pool: usize,
}

/// Get the CUDA memory pool statistics. Returns None if the driver API is
/// unavailable or the pool can't be queried.
#[cfg(not(target_os = "macos"))]
pub fn pool_stats() -> Option<PoolStats> {
    unsafe {
        let mut dev: cudarc::driver::sys::CUdevice = 0;
        if cudarc::driver::sys::cuDeviceGet(&mut dev, 0)
            != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS
        {
            return None;
        }
        let mut pool: cudarc::driver::sys::CUmemoryPool = std::ptr::null_mut();
        if cudarc::driver::sys::cuDeviceGetDefaultMemPool(&mut pool, dev)
            != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS
            || pool.is_null()
        {
            return None;
        }
        let mut reserved: u64 = 0;
        let mut used: u64 = 0;
        cudarc::driver::sys::cuMemPoolGetAttribute(
            pool,
            cudarc::driver::sys::CUmemPool_attribute_enum::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
            &mut reserved as *mut u64 as *mut std::ffi::c_void,
        );
        cudarc::driver::sys::cuMemPoolGetAttribute(
            pool,
            cudarc::driver::sys::CUmemPool_attribute_enum::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
            &mut used as *mut u64 as *mut std::ffi::c_void,
        );
        Some(PoolStats {
            reserved: reserved as usize,
            used: used as usize,
            free_in_pool: reserved.saturating_sub(used) as usize,
        })
    }
}

/// Log device/pool diagnostics. Best-effort, never panics.
///
/// Called by the TTS engine around warmup; records which backend is active,
/// the VRAM snapshot, and (on CUDA) the memory-pool state.
pub fn log_pool_stats() {
    #[cfg(target_os = "macos")]
    {
        let backend = crate::backend::COMPILED_BACKEND;
        match vram_total_bytes() {
            Some(total) => tracing::info!(
                ?backend, total,
                "pool stats: {:.2} GB total (unified memory; free unknown on Metal)",
                total as f64 / 1e9,
            ),
            None => tracing::info!(
                ?backend,
                "pool stats: VRAM query not wired for this backend (returning None)"
            ),
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        use std::sync::atomic::{AtomicBool, Ordering};
        static CHECKED: AtomicBool = AtomicBool::new(false);
        if CHECKED.swap(true, Ordering::Relaxed) {
            return;
        }
        if let Some(stats) = pool_stats() {
            tracing::info!(
                "CUDA pool: reserved {:.1} GB, used {:.1} GB, free-in-pool {:.1} GB (cached chunks for reuse)",
                stats.reserved as f64 / 1e9,
                stats.used as f64 / 1e9,
                stats.free_in_pool as f64 / 1e9,
            );
        }
        if let (Some(free), Some(total)) = (vram_free_bytes(), vram_total_bytes()) {
            tracing::info!(
                "VRAM: {:.1} GB free / {:.1} GB total",
                free as f64 / 1e9, total as f64 / 1e9,
            );
        }
    }
}

/// Trim the CUDA memory pool: release ALL unused physical memory back to the
/// OS. Call after pool growth (top-up) to keep nvidia-smi honest.
///
/// Uses `cuMemPoolTrimTo(pool, 0)` from the CUDA driver API (CUDA 11.2+).
#[cfg(not(target_os = "macos"))]
pub fn trim_cuda_memory_pool() {
    let result = unsafe {
        let mut dev: cudarc::driver::sys::CUdevice = 0;
        if cudarc::driver::sys::cuDeviceGet(&mut dev, 0)
            != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS
        {
            return;
        }
        let mut pool: cudarc::driver::sys::CUmemoryPool = std::ptr::null_mut();
        if cudarc::driver::sys::cuDeviceGetDefaultMemPool(&mut pool, dev)
            != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS
            || pool.is_null()
        {
            return;
        }
        cudarc::driver::sys::cuMemPoolTrimTo(pool, 0)
    };
    if result == cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
        if let Some((free, total)) = vram_free_bytes().zip(vram_total_bytes()) {
            tracing::debug!(
                "CUDA pool trimmed — {:.1} GB free / {:.1} GB total",
                free as f64 / 1e9, total as f64 / 1e9
            );
        }
    }
}

/// Set the CUDA memory pool release threshold to 0 — freed memory is returned
/// to the OS immediately instead of being cached. DISABLES buffer reuse.
///
/// ⚠️ Only call when nvidia-smi accuracy is critical. For normal operation
/// with warmup-based buffer reuse, do NOT call this — the pool's default
/// retention keeps warm chunks for instant reuse by subsequent inferences.
#[cfg(not(target_os = "macos"))]
#[allow(dead_code)]
pub fn set_pool_release_threshold_zero() {
    let result = unsafe {
        let mut dev: cudarc::driver::sys::CUdevice = 0;
        if cudarc::driver::sys::cuDeviceGet(&mut dev, 0)
            != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS
        {
            return;
        }
        let mut pool: cudarc::driver::sys::CUmemoryPool = std::ptr::null_mut();
        if cudarc::driver::sys::cuDeviceGetDefaultMemPool(&mut pool, dev)
            != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS
            || pool.is_null()
        {
            return;
        }
        // CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = 4, value is CUmemPoolPtrToValue
        // (a u64: 0 = release immediately).
        let threshold: u64 = 0;
        cudarc::driver::sys::cuMemPoolSetAttribute(
            pool,
            cudarc::driver::sys::CUmemPool_attribute_enum::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            &threshold as *const u64 as *mut std::ffi::c_void,
        )
    };
    if result == cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
        tracing::info!("CUDA pool release threshold set to 0 — freed memory returns to OS immediately");
    } else {
        tracing::warn!("Failed to set CUDA pool release threshold (error {result:?}) — pool will retain memory");
    }
}

/// Log a one-time diagnostic about CUDA memory pool retention.
///
/// cudarc/candle use the CUDA DRIVER API (`cuMemAllocAsync`/`cuMemFreeAsync`),
/// not the runtime API. Memory freed via `cuMemFreeAsync` returns to the
/// stream's memory pool, not the OS. `nvidia-smi` shows the pool's *reserved*
/// bytes. Call `trim_cuda_memory_pool()` to release unused memory back to the
/// OS after large frees.
#[cfg(not(target_os = "macos"))]
pub fn check_cuda_allocator() {
    use std::sync::atomic::{AtomicBool, Ordering};
    static CHECKED: AtomicBool = AtomicBool::new(false);
    if CHECKED.swap(true, Ordering::Relaxed) {
        return;
    }
    if let Some((free, total)) = vram_free_bytes().zip(vram_total_bytes()) {
        tracing::info!(
            "CUDA driver API memory pool active ({:.1} GB free / {:.1} GB total). \
             nvidia-smi shows reserved pool bytes — call trim_cuda_memory_pool() \
             after large frees for accurate reporting.",
            free as f64 / 1e9, total as f64 / 1e9
        );
    }
}

// ── macOS helper ─────────────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
fn macos_unified_memory_bytes() -> Option<usize> {
    let out = std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout).ok()?.trim().parse::<usize>().ok()
}

// ── NVML fallback (feature `nvml`, non-macOS) ────────────────────────────────

#[cfg(not(target_os = "macos"))]
fn nvml_total_bytes() -> Option<usize> {
    #[cfg(feature = "nvml")]
    {
        nvml_probe::memory_info().map(|m| m.total as usize)
    }
    #[cfg(not(feature = "nvml"))]
    {
        None
    }
}

#[cfg(not(target_os = "macos"))]
fn nvml_free_bytes() -> Option<usize> {
    #[cfg(feature = "nvml")]
    {
        nvml_probe::memory_info().map(|m| m.free as usize)
    }
    #[cfg(not(feature = "nvml"))]
    {
        None
    }
}

#[cfg(all(not(target_os = "macos"), feature = "nvml"))]
mod nvml_probe {
    //! Lazy, fallible NVML binding. NVML is initialised at most once and shared;
    //! any failure (no driver, no GPU, library missing) collapses to `None` so
    //! every probe degrades gracefully on non-CUDA hosts.
    use nvml_wrapper::Nvml;
    use std::sync::OnceLock;

    static NVML: OnceLock<Option<Nvml>> = OnceLock::new();

    fn nvml() -> Option<&'static Nvml> {
        NVML.get_or_init(|| Nvml::init().ok()).as_ref()
    }

    pub(super) fn memory_info() -> Option<nvml_wrapper::MemoryInfo> {
        let nvml = nvml()?;
        let dev = nvml.device_by_index(0).ok()?;
        dev.memory_info().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probes_fall_back_gracefully_without_nvml() {
        #[cfg(target_os = "macos")]
        {
            // Unified memory is always reported via sysctl on macOS.
            assert!(vram_total_bytes().is_some());
            assert_eq!(vram_free_bytes(), None);
        }
        // Must not panic in any configuration (on CUDA hosts the cudarc probe
        // may legitimately report real numbers — no hard asserts here):
        log_pool_stats();
    }

    #[test]
    fn cpu_device_is_cpu() {
        assert!(matches!(cpu_device(), Device::Cpu));
    }
}
