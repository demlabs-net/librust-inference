//! CUDA Graph capture for Fast AR inference acceleration.
//!
//! Fast AR has fixed shape (batch=1, seq=11) — ideal for CUDA graph.
//! When available, captures the forward pass on first call and replays
//! on all subsequent calls, eliminating kernel launch overhead (~3x speedup).
//!
//! NOTE: candle 0.11 does not expose CUDA graph capture API.
//! This module is a placeholder — implementation deferred to Phase 8
//! when candle adds CUDA graph support or when we switch to cudarc directly.

use candle_core::Result;

/// CUDA Graph capture wrapper for Fast AR.
/// Currently a no-op — returns false from `is_available()`.
pub struct CudaGraph {
    _captured: bool,
}

impl CudaGraph {
    pub fn new() -> Self {
        Self { _captured: false }
    }

    /// CUDA graph capture is not available in candle 0.11.
    pub fn is_available(&self) -> bool {
        false
    }

    /// Attempt to capture — returns Err if not supported.
    pub fn capture<F: FnOnce() -> Result<candle_core::Tensor>>(&mut self, _f: F) -> Result<candle_core::Tensor> {
        Err(candle_core::Error::Msg("CUDA graph capture not supported in candle 0.11".into()))
    }
}
