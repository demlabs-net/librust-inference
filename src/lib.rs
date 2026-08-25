//! Shared inference infrastructure — the replacement for candle.
//!
//! The general-purpose inference stack: the own Metal engine, native NN
//! layers, CPU sampling, safetensors weights. Serves TTS, STT and future
//! LLM ports alike. No candle anywhere.
//!
//! ## Modules
//!
//! - [`nn`] — Native NN layers (Linear, RMSNorm, RoPE, GQA Attention + KV
//!   cache, SwiGLU, CausalConv1d/ConvTranspose, ConvNeXt, Snake, Embedding).
//!   Currently Metal-backed (`metal` feature); being moved onto the
//!   [`backend::Backend`] contract.
//! - [`metal`] — The Metal engine: own MSL kernels + buffers (`metal`
//!   feature, macOS), the candle-replacement compute runtime.
//! - [`inference`] — Sampling (top-p, repetition penalty, RAS).
//! - [`concurrency`] — Cross-engine inference guards (Metal command-buffer
//!   serialization; no-op on other backends).
//! - [`weights`] — Multi-file safetensors index loading + fp32 reader.

pub mod backend;
pub mod audio;
pub mod chat_template;
pub mod concurrency;
#[cfg(feature = "cpu")]
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod dflash;
pub mod dtype;
pub mod error;
pub mod gguf;
#[cfg(feature = "cpu")]
pub mod harness;
pub mod inference;
#[cfg(feature = "metal")]
pub mod metal;
pub mod nn;
pub mod omni;
pub mod qwen3;
pub mod spec;
pub mod train;
pub mod vision;
pub mod vision_qwen3vl;
pub mod tokenizer;
pub mod weights;
pub mod weights_map;

pub use backend::{Backend, Tensor};
pub use dtype::{Dtype, KvFormat, QuantConfig, WeightFormat};
pub use error::BackendError;

// GPU-only rule: a build without an enabled GPU backend must not silently fall
// back to CPU (there is no implicit CPU path; `cpu` is the test-only reference
// backend behind an explicit runtime opt-in). Backend selection is a build-time
// feature: `metal` (macOS), future `cuda`/`vulkan`/`wasm-webgpu`.
