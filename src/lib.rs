//! Shared inference infrastructure — the replacement for candle.
//!
//! Extracted from `librust-speech-inference` so the general-purpose inference
//! stack (NN primitives, the own Metal engine, device/backend management,
//! sampling, weights) can serve TTS, STT and future LLM ports alike.
//!
//! ## Modules
//!
//! - [`nn`] — Native NN layers on the Metal engine (Linear, RMSNorm,
//!   RoPE, GQA Attention + KV cache, SwiGLU, CausalConv1d/ConvTranspose,
//!   ConvNeXt, Snake, Embedding) — the candle-replacement layer API.
//! - [`metal`] — The Metal engine: own MSL kernels + buffers
//!   (macOS-only), the candle-replacement compute runtime.
//! - [`inference`] — Sampling (top-p, repetition penalty, RAS), CUDA graph stub.
//! - [`ops`] — Tensor ops missing a kernel in candle 0.11 (softmax_last_dim).
//! - [`backend`] — Build-time backend selection (Metal on macOS, CUDA on Linux) + default device.
//! - [`device`] — Low-level per-backend device constructors + VRAM diagnostics.
//! - [`concurrency`] — Cross-engine inference guards (Metal command-buffer serialization).
//! - [`quant`] — Quantization levels (FP32/BF16/FP16/INT8/INT6/INT4/NF4).
//! - [`weights`] — Multi-file safetensors index loading + fp32 reader + weight-norm helpers.
//!
//! The candle-based pieces (`nn_candle`, `ops`, `quant`, `backend`, `device`,
//! `cuda_graph`, tensor sampling) are gated behind the `candle` feature
//! (NOT enabled by default) — the Metal engine does not need candle
//! at all; it is only pulled in by the CUDA/Linux path and the golden
//! reference tests.

pub mod nn;
#[cfg(feature = "candle")]
pub mod nn_candle;
#[cfg(feature = "candle")]
pub mod backend;
pub mod concurrency;
#[cfg(feature = "candle")]
pub mod device;
pub mod inference;
#[cfg(target_os = "macos")]
pub mod metal;
#[cfg(feature = "candle")]
pub mod ops;
#[cfg(feature = "candle")]
pub mod quant;
pub mod weights;

/// GPU-only rule: a Windows build without an integrated GPU backend must not
/// silently fall back to CPU (there is no CPU path). The DirectML runtime is
/// the policy default but is not integrated yet — until it lands, NVIDIA
/// Windows builds use `--features cuda` and anything else fails here.
#[cfg(all(target_os = "windows", not(feature = "cuda")))]
compile_error!(
    "rust-inference on Windows requires a GPU backend: \
     build with `--features cuda` on NVIDIA hardware. \
     The DirectML runtime (Windows policy default) is not integrated yet — \
     see src/backend.rs."
);
