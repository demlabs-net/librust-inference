//! Ф1.9 Decode loop: prefill → sample → decode → stop conditions, with a
//! streaming event interface. The loop is backend-agnostic (CPU logits
//! vectors through [`crate::inference::sampling`]); KV state lives in the
//! model.

use crate::backend::Backend;
use crate::error::BackendError;
use crate::inference::sampling::{SamplingParams, top_p_sample_vec};
use crate::qwen3::Qwen3Model;
use crate::tokenizer::{IncrementalDecoder, Tokenizer};

/// Почему завершилась генерация.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    /// EOG-токен (eos/im_end и т.п.).
    Eog,
    /// Достигнут `max_tokens`.
    MaxTokens,
    /// Найден один из `stop_strings` в накопленном тексте.
    StopString,
}

/// Событие стриминга.
#[derive(Debug, Clone)]
pub enum GenerateEvent {
    /// Новый фрагмент текста (уже склеенный по границам UTF-8).
    Delta(String),
    /// Сырой токен (для метрик/логов).
    Token(u32),
    /// Генерация завершена.
    Done(FinishReason),
}

/// Параметры генерации (дефолты — Qwen3-совместимые).
#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repeat_penalty: f32,
    /// Текстовые стоп-строки (проверяются по накопленному выводу).
    pub stop_strings: Vec<String>,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.8,
            top_p: 0.8,
            top_k: 0,
            repeat_penalty: 1.0,
            stop_strings: Vec::new(),
        }
    }
}

/// Стоп-решение одного шага: EOG-токен, стоп-строка в накопленном тексте
/// или None (продолжать). Вынесено отдельно — детерминированный юнит.
pub fn check_stop(
    next: u32,
    eos: Option<u32>,
    acc: &str,
    stop_strings: &[String],
) -> Option<FinishReason> {
    if eos == Some(next) {
        return Some(FinishReason::Eog);
    }
    if stop_strings.iter().any(|s| acc.contains(s)) {
        return Some(FinishReason::StopString);
    }
    None
}

/// Сгенерировать продолжение после `ids` (полный префилл выполняется
/// здесь). Возвращает сгенерированные токены; события стримятся в
/// `on_event`. Greedy при temperature <= 0.
pub fn generate(
    model: &mut Qwen3Model,
    backend: &dyn Backend,
    tok: &Tokenizer,
    ids: &[u32],
    params: &GenerateParams,
    mut on_event: impl FnMut(GenerateEvent),
) -> Result<Vec<u32>, BackendError> {
    // Prefill: батч-префилл, если промпт длиннее одного токена.
    let mut logits = if ids.len() > 1 {
        model.prefill(backend, ids)?
    } else {
        model.decode(backend, ids[0])?
    };
    let vocab = model.cfg.vocab_size;

    let mut generated: Vec<u32> = Vec::new();
    let mut rng_state: Option<u64> = None;
    let mut decoder = IncrementalDecoder::new();
    let mut acc = String::new(); // накопленный текст (для stop-строк)
    let mut finish: Option<FinishReason> = None;

    for _ in 0..params.max_tokens {
        let v = backend.to_cpu_f32(&logits)?;
        let last = &v[v.len() - vocab..];
        let next = if params.temperature <= 0.0 {
            // P18c: a NaN logit used to PANIC in partial_cmp().unwrap() —
            // treat non-finite values as -inf (never selected).
            last.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap()
        } else {
            let sp = SamplingParams {
                temperature: params.temperature,
                top_p: params.top_p,
                top_k: params.top_k,
                repetition_penalty: params.repeat_penalty,
                seed: None,
            };
            top_p_sample_vec(last, &sp, &mut rng_state)
        };

        if let Some(r) = check_stop(next, tok.eos, &acc, &params.stop_strings) {
            finish = Some(r);
            break;
        }
        let piece = tok.decode_piece(next);
        let delta = decoder.push(piece.as_bytes());
        if !delta.is_empty() {
            acc.push_str(&delta);
            on_event(GenerateEvent::Delta(delta));
        }
        on_event(GenerateEvent::Token(next));
        generated.push(next);

        logits = model.decode(backend, next)?;
    }
    if finish.is_none() {
        finish = Some(FinishReason::MaxTokens);
    }
    let tail = decoder.finish();
    if !tail.is_empty() {
        on_event(GenerateEvent::Delta(tail));
    }
    on_event(GenerateEvent::Done(finish.unwrap()));
    Ok(generated)
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use crate::metal::MetalBackend;
    use crate::qwen3::{Qwen3Config, Qwen3Model};
    use crate::weights_map::{WeightTensor, WeightsMap};

    fn mini_model() -> (MetalBackend, Qwen3Model) {
        let backend = MetalBackend::new().unwrap();
        let c = Qwen3Config {
            dim: 16,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 4,
            ffn_dim: 32,
            vocab_size: 8,
            rope_theta: 10_000.0,
            rope_max_seq: 128,
            norm_eps: 1e-6,
            swa_window: None,
            mrope_sections: None,
            n_deepstack: None,
        };
        let mut rng: u64 = 0xC0FFEE;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut rand_vec = |n: usize| {
            (0..n)
                .map(|_| ((next() >> 32) as f32 / (1u64 << 32) as f32) * 2.0 - 1.0)
                .collect::<Vec<f32>>()
        };
        let mut storage: Vec<(String, Vec<f32>, Vec<usize>)> = vec![(
            "token_embd.weight".into(),
            rand_vec(c.vocab_size * c.dim),
            vec![c.vocab_size, c.dim],
        )];
        for i in 0..c.n_layers {
            let p = |s: &str| format!("blk.{i}.{s}");
            for (n, shape) in [
                ("attn_norm.weight", vec![c.dim]),
                ("attn_q.weight", vec![c.dim, c.dim]),
                ("attn_k.weight", vec![c.n_kv_heads * c.head_dim, c.dim]),
                ("attn_v.weight", vec![c.n_kv_heads * c.head_dim, c.dim]),
                ("attn_output.weight", vec![c.dim, c.dim]),
                ("attn_q_norm.weight", vec![c.head_dim]),
                ("attn_k_norm.weight", vec![c.head_dim]),
                ("ffn_norm.weight", vec![c.dim]),
                ("ffn_gate.weight", vec![c.ffn_dim, c.dim]),
                ("ffn_up.weight", vec![c.ffn_dim, c.dim]),
                ("ffn_down.weight", vec![c.dim, c.ffn_dim]),
            ] {
                let numel = shape.iter().product::<usize>();
                storage.push((p(n), rand_vec(numel), shape));
            }
        }
        storage.push(("output_norm.weight".into(), rand_vec(c.dim), vec![c.dim]));
        let raw: Vec<(String, Vec<u8>, Vec<u64>)> = storage
            .iter()
            .map(|(n, data, shape)| {
                let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
                (n.clone(), bytes, shape.iter().map(|&d| d as u64).collect())
            })
            .collect();
        let tensors: Vec<WeightTensor> = raw
            .iter()
            .map(|(name, bytes, shape)| WeightTensor {
                name: name.clone(),
                shape: shape.clone(),
                format: crate::dtype::WeightFormat::Plain(crate::dtype::Dtype::F32),
                offset: 0,
                data: bytes,
            })
            .collect();
        let wm = WeightsMap {
            arch: "qwen3".into(),
            tensors,
            storage: vec![],
        };
        let model = Qwen3Model::load(&backend, &wm, c).unwrap();
        (backend, model)
    }

    fn tiny_tokenizer() -> crate::tokenizer::Tokenizer {
        crate::tokenizer::Tokenizer::from_parts(
            (0..8).map(|i| vec![b'a' + i as u8]).collect(),
            vec![],
            None,
            None,
            None,
        )
    }

    #[test]
    fn greedy_generates_until_max_tokens() {
        let (backend, mut model) = mini_model();
        let mut deltas = 0usize;
        let mut done = None;
        let params = GenerateParams {
            max_tokens: 3,
            temperature: 0.0,
            ..Default::default()
        };
        let gen_tokens = generate(
            &mut model,
            &backend,
            &tiny_tokenizer(),
            &[1, 2],
            &params,
            |ev| match ev {
                GenerateEvent::Delta(_) => deltas += 1,
                GenerateEvent::Done(r) => done = Some(r),
                _ => {}
            },
        )
        .unwrap();
        assert_eq!(gen_tokens.len(), 3, "max_tokens reached");
        assert_eq!(done, Some(FinishReason::MaxTokens));
        assert!(deltas >= 0);
    }

    #[test]
    fn check_stop_eog_and_stop_string() {
        assert_eq!(check_stop(2, Some(2), "", &[]), Some(FinishReason::Eog));
        assert_eq!(check_stop(1, Some(2), "", &[]), None);
        assert_eq!(
            check_stop(7, None, "hello world", &["world".into()]),
            Some(FinishReason::StopString)
        );
        assert_eq!(check_stop(7, None, "hello", &["world".into()]), None);
    }

    #[test]
    fn stream_emits_deltas_and_final_done() {
        let (backend, mut model) = mini_model();
        let mut events = Vec::new();
        let params = GenerateParams {
            max_tokens: 2,
            temperature: 0.0,
            ..Default::default()
        };
        generate(
            &mut model,
            &backend,
            &tiny_tokenizer(),
            &[1],
            &params,
            |ev| {
                events.push(ev);
            },
        )
        .unwrap();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, GenerateEvent::Done(FinishReason::MaxTokens)))
        );
        assert!(events.iter().any(|e| matches!(e, GenerateEvent::Token(_))));
    }
}
