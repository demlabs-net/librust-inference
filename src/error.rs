//! Backend-agnostic error type — the single error surface of the
//! [`crate::backend::Backend`] contract.
//!
//! Every backend maps its native error into [`BackendError`] at the trait
//! boundary (`From` impls under their features). Callers (NN layers, speech
//! models, pools) match on the typed variants instead of stringly errors —
//! e.g. an allocation failure must surface as [`BackendError::OutOfMemory`]
//! so the engine pools can map it to the NO_RESOURCES/Waiting semantics
//! instead of a panic.

/// All errors that can cross a backend boundary.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    /// A backend that is not compiled into this build was requested
    /// (feature not enabled).
    #[error("backend not compiled in: {0}")]
    BackendUnavailable(&'static str),

    /// The device/context could not be created (no GPU, driver missing,
    /// feature not enabled, …).
    #[error("device init failed: {0}")]
    DeviceInit(String),

    /// Операция остановлена по явному запросу (флаг отмены: таймаут TTS,
    /// стоп-кнопка). Не ошибка устройства — вызывающий сам решил прервать.
    #[error("cancelled")]
    Cancelled,

    /// GPU memory allocation failed. Pools must translate this into the
    /// NO_RESOURCES / Waiting semantics, never panic.
    #[error("out of memory: {0}")]
    OutOfMemory(String),

    /// Shape/rank mismatch between tensors in an op.
    #[error("shape mismatch: {0}")]
    Shape(String),

    /// A dtype or quantized-format combination is not supported by this
    /// backend/op (the config validator catches these at load time; this is
    /// the runtime backstop).
    #[error("unsupported dtype/format combination: {0}")]
    UnsupportedFormat(String),

    /// A kernel failed to compile/load (WGSL/SPIR-V/PTX/MSL).
    #[error("kernel error: {0}")]
    Kernel(String),

    /// The native backend reported a generic failure.
    #[error("backend error: {0}")]
    Native(String),

    /// Host-side I/O (weights, cache files, …).
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

impl BackendError {
    pub fn native(e: impl std::fmt::Display) -> Self {
        BackendError::Native(e.to_string())
    }
}

#[cfg(feature = "metal")]
impl From<crate::metal::MetalError> for BackendError {
    fn from(e: crate::metal::MetalError) -> Self {
        // MetalError carries no structured OOM info today; wrap generically.
        // When the Metal engine gains typed errors, match here.
        BackendError::native(e)
    }
}
