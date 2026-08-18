//! Inference utilities shared between TTS and STT.
#[cfg(feature = "candle")]
pub mod cuda_graph;
pub mod sampling;
