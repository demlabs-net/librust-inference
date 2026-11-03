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
//! - [`plugin`] — Плагинный слой движков: контракт `EnginePlugin`,
//!   версионирование, способности; нативные (qwen) и внешние адаптеры.
//! - [`concurrency`] — Cross-engine inference guards (Metal command-buffer
//!   serialization; no-op on other backends).
//! - [`weights`] — Multi-file safetensors index loading + fp32 reader.

// Осознанные исключения для педантичных линтов (не «глушение под ноль», а
// контрактные особенности этого крейта):
// - from_* принимают &self НАМЕРЕННО: методы диспатчатся через &dyn Backend и
//   аллоцируют буфер в рантайме реализации (см. комментарий в backend.rs);
// - too_many_arguments/type_complexity: сигнатуры обёрток повторяют раскладку
//   GPU-операций (gemm/attn/kv) — группировка в структуру добавила бы слой
//   без пользы для читателя;
// - needless_range_loop: циклы индексируют несколько массивов одновременно.
#![allow(
    clippy::wrong_self_convention,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::needless_range_loop
)]

/// Каталог моделей: `INFER_MODELS_DIR` → `~/.infer-models` → устаревший
/// `VASSISTA_MODELS_DIR`/`~/.vassista-models` (с предупреждением). Та же
/// политика, что в `infer_core::models_dir()` сервиса: тесты/примеры не
/// должны знать про старые пути (раскладка подкаталогов: llm/stt/tts/vad/vlm).
pub fn models_dir() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("INFER_MODELS_DIR") {
        return std::path::PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    let new = std::path::PathBuf::from(&home).join(".infer-models");
    if new.exists() {
        return new;
    }
    if let Ok(p) = std::env::var("VASSISTA_MODELS_DIR") {
        eprintln!("models: VASSISTA_MODELS_DIR устарела — переименуйте в INFER_MODELS_DIR");
        return std::path::PathBuf::from(p);
    }
    let legacy = std::path::PathBuf::from(&home).join(".vassista-models");
    if legacy.exists() {
        eprintln!(
            "models: найден устаревший {} — используйте INFER_MODELS_DIR (раскладка та же)",
            legacy.display()
        );
        return legacy;
    }
    new
}

pub mod arch;
pub mod audio;
pub mod backend;
pub mod batch;
pub mod bert;
pub mod chat_template;
pub mod concurrency;
#[cfg(feature = "cpu")]
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod datasets;
pub mod dflash;
pub mod dtype;
pub mod error;
pub mod export;
pub mod gguf;
pub mod glm4;
#[cfg(feature = "cpu")]
pub mod harness;
pub mod inference;
#[cfg(feature = "metal")]
pub mod metal;
pub mod moe;
pub mod mtp;
pub mod nn;
pub mod omni;
pub mod plugin;
pub mod quant;
pub mod qwen3;
pub mod qwen35;
/// Qwen3.8-Flash-Next (`qwen4exp`): конфиг + схема файла (forward — A5).
pub mod qwen4exp;
pub mod rope;
pub mod spec;
pub mod tokenizer;
pub mod tp;
pub mod train;
pub mod train_qwen3;
mod unicode_data;
pub mod vision;
pub mod vision_qwen3vl;
pub mod weights;
pub mod weights_map;

pub use backend::{Backend, Tensor};
pub use dtype::{Dtype, KvFormat, QuantConfig, WeightFormat};
pub use error::BackendError;
pub use plugin::{
    Capabilities, EnginePlugin, ExternalEngine, ModelVersion, Offload, PluginKind, QwenPlugin,
    QwenVersion, SpecBackend, Version, NATIVE_REGISTRY, QWEN, lookup_arch, qwen_version_for_arch,
};

// GPU-only rule: a build without an enabled GPU backend must not silently fall
// back to CPU (there is no implicit CPU path; `cpu` is the test-only reference
// backend behind an explicit runtime opt-in). Backend selection is a build-time
// feature: `metal` (macOS), future `cuda`/`vulkan`/`wasm-webgpu`.
