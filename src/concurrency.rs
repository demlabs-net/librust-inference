//! Metal GPU concurrency guard.
//!
//! The Metal device can panic when multiple command buffers are committed
//! concurrently from different threads (a known limitation of the shared
//! Metal device). The engine pools keep N warm engines to serve parallel
//! calls; without gating, several concurrent Metal forward passes risk the
//! "committing a command buffer while an encoder is still active" panic.
//!
//! [`metal_inference_guard()`] returns an RAII guard holding a permit from a
//! process-wide counting semaphore. Wrap the top-level GPU call of each engine
//! (TTS `generate_chunk`, STT `transcribe`) so at most `METAL_CONCURRENCY`
//! (env, default 2) engines run inference at the same time. The pool, warmup,
//! and parallel-data capability are preserved — only true concurrency is
//! capped. If panics appear at the default, lower `METAL_CONCURRENCY` to 1.
//!
//! On non-Metal builds the guard is a zero-sized no-op.

/// Simple counting semaphore (sync) over std Mutex + Condvar. macOS-only —
/// the CUDA/Linux guard is a no-op and never touches it.
#[cfg(feature = "metal")]
use once_cell::sync::Lazy;
#[cfg(feature = "metal")]
use std::sync::{Condvar, Mutex};

#[cfg(feature = "metal")]
struct Semaphore {
    count: Mutex<usize>,
    cv: Condvar,
}

#[cfg(feature = "metal")]
impl Semaphore {
    fn new(count: usize) -> Self {
        Self {
            count: Mutex::new(count),
            cv: Condvar::new(),
        }
    }

    fn acquire(&self) {
        let mut c = self.count.lock().unwrap();
        while *c == 0 {
            c = self.cv.wait(c).unwrap();
        }
        *c -= 1;
    }

    fn release(&self) {
        let mut c = self.count.lock().unwrap();
        *c += 1;
        self.cv.notify_one();
    }
}

/// Default permits (parallel, below the observed Metal panic threshold).
/// Override with the `METAL_CONCURRENCY` env var (must be > 0).
#[cfg(feature = "metal")]
fn configured_permits() -> usize {
    std::env::var("METAL_CONCURRENCY")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(2)
}

#[cfg(feature = "metal")]
static METAL_SEM: Lazy<Semaphore> = Lazy::new(|| Semaphore::new(configured_permits()));

/// RAII inference guard. On Metal it holds a semaphore permit for its lifetime;
/// on other backends it is a zero-sized no-op.
pub struct InferenceGuard {
    #[cfg(feature = "metal")]
    _permit: Permit,
}

#[cfg(feature = "metal")]
struct Permit;
#[cfg(feature = "metal")]
impl Drop for Permit {
    fn drop(&mut self) {
        METAL_SEM.release();
    }
}

/// Acquire a Metal inference permit for the duration of the returned guard.
///
/// Place at the top of a GPU forward pass:
/// ```ignore
/// let _g = rust_inference::concurrency::metal_inference_guard();
/// ```
#[cfg(feature = "metal")]
pub fn metal_inference_guard() -> InferenceGuard {
    METAL_SEM.acquire();
    tracing::trace!("metal inference permit acquired");
    InferenceGuard { _permit: Permit }
}

/// No-op guard on non-Metal builds (CUDA / CPU): no command-buffer contention.
#[cfg(not(feature = "metal"))]
pub fn metal_inference_guard() -> InferenceGuard {
    InferenceGuard {}
}
