//! Backend abstraction — the single chokepoint for device selection.
//!
//! The compute backend is fixed at BUILD TIME by target platform:
//! - macOS → [`Backend::Metal`] (GPU, via `candle-core`'s `metal` feature
//!   and the crate's own MSL engine)
//! - Linux → [`Backend::Cuda`]  (GPU, via `candle-core`'s `cuda` feature)
//! - Windows → [`Backend::DirectML`] by policy (any vendor), with `CUDA`
//!   available on NVIDIA via the `cuda` cargo feature; the DirectML runtime
//!   itself is not integrated yet — building without `cuda` fails with a
//!   clear `compile_error!` instead of a broken CUDA build
//!
//! Every consumer must go through [`default_device`] — never call
//! `Device::new_cuda` / `Device::new_metal` directly. Centralising the choice
//! here is also the seam where the DirectML / Vulkan / OpenCL backends slot
//! in: add the runtime behind its cargo feature + an arm in
//! [`default_device`] and declare the deps in `Cargo.toml`.
//!
//! See also the crate-level rule: **no CPU fallback** — inference and the DAC
//! codec run entirely on the GPU backend selected here.

use crate::quant::QuantizationLevel;
use candle_core::Device;

/// The compute backend compiled into this build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// NVIDIA CUDA (Linux production, Windows with the `cuda` feature).
    Cuda,
    /// Apple Metal (macOS).
    Metal,
    /// Microsoft DirectML (Windows, any vendor) — policy default on
    /// Windows; the integrated runtime is not implemented yet.
    DirectML,
    /// Vulkan Compute (Linux AMD/Intel/RPi, Windows) — not implemented.
    Vulkan,
    /// OpenCL — placeholder. Not implemented (candle has no OpenCL device);
    /// exists so the abstraction can grow without touching call sites.
    OpenCl,
}

/// The backend this binary was compiled for (target-platform-driven).
#[cfg(target_os = "macos")]
pub const COMPILED_BACKEND: Backend = Backend::Metal;

/// The backend this binary was compiled for (target-platform-driven).
#[cfg(all(unix, not(target_os = "macos")))]
pub const COMPILED_BACKEND: Backend = Backend::Cuda;

/// The backend this binary was compiled for (target-platform-driven).
/// Windows policy default is DirectML; the `cuda` feature (NVIDIA) wins.
#[cfg(all(target_os = "windows", feature = "cuda"))]
pub const COMPILED_BACKEND: Backend = Backend::Cuda;

/// The backend this binary was compiled for (target-platform-driven).
#[cfg(all(target_os = "windows", not(feature = "cuda")))]
pub const COMPILED_BACKEND: Backend = Backend::DirectML;

/// Create the default device for this build's backend.
///
/// `ordinal` selects the physical device index (0 = first GPU). Returns the
/// device together with the backend it was created on, so callers can branch
/// on backend-specific behaviour (e.g. quantization defaults).
///
/// On failure the error is human-readable; callers in the engine pool log it
/// and degrade gracefully (warm spares simply aren't built).
pub fn default_device(ordinal: usize) -> anyhow::Result<(Device, Backend)> {
    #[cfg(target_os = "macos")]
    {
        let dev = crate::device::metal_device(ordinal)?;
        Ok((dev, Backend::Metal))
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let dev = crate::device::cuda_device(ordinal)?;
        Ok((dev, Backend::Cuda))
    }
    #[cfg(all(target_os = "windows", feature = "cuda"))]
    {
        let dev = crate::device::cuda_device(ordinal)?;
        Ok((dev, Backend::Cuda))
    }
    #[cfg(all(target_os = "windows", not(feature = "cuda")))]
    {
        // DirectML policy default, but the runtime is not integrated yet.
        // Fail loudly and actionably instead of silently falling back.
        anyhow::bail!(
            "DirectML device not integrated yet on Windows — \
             build with `--features cuda` on NVIDIA hardware"
        )
    }
}

/// Platform-aware default quantization.
///
/// `BF16` is the native dtype of the S2-Pro safetensors, but Metal's matmul
/// does not support bf16 — on Metal we default to `FP16`. CUDA keeps `BF16`.
/// DirectML / Vulkan / OpenCL default to `FP16` too (the integrated runtimes
/// are not in place yet; the default will be re-measured when they land).
/// An explicit env override (`TTS_QUANTIZATION` / `STT_QUANTIZATION`) always
/// wins over this default; this is only the fallback when the env is unset.
pub fn default_quantization(b: Backend) -> QuantizationLevel {
    match b {
        Backend::Cuda => QuantizationLevel::BF16,
        Backend::Metal | Backend::DirectML | Backend::Vulkan | Backend::OpenCl => {
            QuantizationLevel::FP16
        }
    }
}
