//! Shared inference infrastructure — the replacement for candle.
//!
//! The general-purpose inference stack: the own Metal engine, native NN
//! layers, CPU sampling, safetensors weights. Serves TTS, STT and future
//! LLM ports alike. No candle anywhere.
//!
//! ## Modules
//!
//! - [`nn`] — Native NN layers on the Metal engine (Linear, RMSNorm,
//!   RoPE, GQA Attention + KV cache, SwiGLU, CausalConv1d/ConvTranspose,
//!   ConvNeXt, Snake, Embedding) — the candle-replacement layer API.
//! - [`metal`] — The Metal engine: own MSL kernels + buffers
//!   (macOS-only), the candle-replacement compute runtime.
//! - [`inference`] — Sampling (top-p, repetition penalty, RAS).
//! - [`concurrency`] — Cross-engine inference guards (Metal command-buffer serialization).
//! - [`weights`] — Multi-file safetensors index loading + fp32 reader.

pub mod backend;
pub mod nn;
pub mod concurrency;
pub mod inference;
#[cfg(target_os = "macos")]
pub mod metal;
pub mod weights;

pub use backend::{Backend, Tensor};

/// GPU-only rule: a build without an integrated GPU backend must not silently
/// fall back to CPU (there is no CPU path). The Metal engine covers macOS; a
/// future CUDA/Vulkan backend will cover Linux/Windows.
#[cfg(all(target_os = "windows", not(target_os = "macos")))]
compile_error!(
    "rust-inference: no integrated GPU backend for this target yet — \
     the Metal engine covers macOS; CUDA/Vulkan land later."
);
