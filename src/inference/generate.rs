//! Ф1.9 Decode loop: prefill → sample → decode → stop conditions, with a
//! streaming event interface. The loop is backend-agnostic (CPU logits
//! vectors through [`crate::inference::sampling`]); KV state lives in the
//! model.

use crate::backend::Backend;
use crate::error::BackendError;
use crate::inference::sampling::{
    SamplingParams, apply_repetition_penalty_vec_unique, top_p_sample_vec_history,
};
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
    /// Останов по запросу (отмена генерации, Этап 2.2): клиент запросил стоп.
    Cancelled,
}

/// Событие стриминга.
#[derive(Debug, Clone)]
pub enum GenerateEvent {
    /// Новый фрагмент текста (уже склеенный по границам UTF-8).
    Delta(String),
    /// Сырой токен (для метрик/логов).
    Token(u32),
    /// Спек-статистика (4.2b) — при включённом драфтере.
    SpecStats(SpecStats),
    /// Генерация завершена.
    Done(FinishReason),
}

/// Статистика спек-декодирования (Этап 4.2b).
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct SpecStats {
    pub drafts: u64,
    pub accepted: u64,
    pub rejected: u64,
}

impl SpecStats {
    pub fn acceptance(&self) -> Option<f64> {
        let total = self.accepted + self.rejected;
        if total == 0 {
            None
        } else {
            Some(self.accepted as f64 / total as f64 * 100.0)
        }
    }
}

/// Мультимодальный промпт (2.4b): ids с image_pad + фичи картинки.
pub struct VisionPrompt {
    pub ids: Vec<u32>,
    pub vemb: crate::backend::Tensor,
    pub vds: Vec<crate::backend::Tensor>,
    pub pad_token: u32,
    pub vision_token: u32,
    pub pos4d: Vec<u32>,
    pub pos_after: usize,
}

pub struct GenerateParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repeat_penalty: f32,
    /// Сид сэмплинга: `Some` — детерминированный поток (материализуется через
    /// [`crate::inference::sampling::state_from_seed`]), `None` — от часов.
    pub seed: Option<u64>,
    /// Текстовые стоп-строки (проверяются по накопленному выводу).
    pub stop_strings: Vec<String>,
    /// Спек-декодинг (4.2b): драфтер + число драфт-токенов (0 = выкл).
    pub drafter: Option<std::sync::Arc<std::sync::Mutex<dyn crate::spec::Drafter>>>,
    pub spec_gamma: usize,
    /// Vision (2.4b): картинка в промпте — prefill_with_vision вместо prefill.
    pub vision: Option<VisionPrompt>,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.8,
            top_p: 0.8,
            top_k: 0,
            repeat_penalty: 1.0,
            seed: None,
            stop_strings: Vec::new(),
            drafter: None,
            spec_gamma: 0,
            vision: None,
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
    mut on_event: impl FnMut(GenerateEvent) -> bool,
) -> Result<Vec<u32>, BackendError> {
    let mut spec_stats = SpecStats::default();
    // Vision-префилл (2.4b): картинка уже посчитана — идём через M-RoPE.
    let mut logits = if let Some(v) = &params.vision {
        model.prefill_with_vision(
            backend,
            &v.ids,
            &v.vemb,
            &v.vds,
            v.vision_token,
            &v.pos4d,
            v.pos_after,
        )?
    } else if ids.len() > 1 {
        // DFlash (4.2.c, z-lab): первый блок контекстуализируется НА ВСЁМ
        // промпте — extract префилла отдаём драфтеру сразу (иначе первый
        // block только «видит» past_len=0 и не принимает ничего).
        let tlayers: Option<Vec<usize>> = params
            .drafter
            .as_ref()
            .map(|d| d.lock().unwrap().target_layers().to_vec());
        match tlayers {
            Some(tl) if !tl.is_empty() => {
                let (lg, extract) = model.prefill_extract(backend, ids, &tl)?;
                let n = ids.len();
                if let Some(dr) = params.drafter.as_ref() {
                    let mut dr = dr.lock().unwrap();
                    let _ = dr.after_extract(backend, &extract, n, n);
                }
                lg
            }
            _ => model.prefill(backend, ids)?,
        }
    } else {
        model.decode(backend, ids[0])?
    };
    let vocab = model.cfg.vocab_size;

    let mut generated: Vec<u32> = Vec::new();
    // Сид запроса материализуется в состояние один раз на генерацию: и
    // спекулятивный, и обычный путь сэмплят из этого одного потока.
    let mut rng_state: Option<u64> = params
        .seed
        .map(crate::inference::sampling::state_from_seed);
    let mut decoder = IncrementalDecoder::new();
    let mut acc = String::new(); // накопленный текст (для stop-строк)
    // История для repetition penalty: промпт + уже порождённые токены
    // (раньше params.repeat_penalty молча игнорировался).
    let mut history: Vec<u32> = ids.to_vec();
    let mut finish: Option<FinishReason> = None;

    for _ in 0..params.max_tokens {
        let v = backend.to_cpu_f32(&logits)?;
        let last = &v[v.len() - vocab..];
        let next = if params.temperature <= 0.0 {
            // Greedy: ТОТ ЖЕ repetition penalty, что и в сэмплирующей ветке
            // (раньше greedy игнорировал history-штраф). Копия — last
            // остаётся нетронутым для следующих итераций.
            let mut penalized = last.to_vec();
            apply_repetition_penalty_vec_unique(&mut penalized, &history, params.repeat_penalty);
            // P18c: a NaN logit used to PANIC in partial_cmp().unwrap().
            // N (iter-9): the claim "treat non-finite as -inf" is now real —
            // the old max_by kept a NaN at index 0 as the running max
            // (partial_cmp(NaN, x) == None → Equal → accumulator survives).
            let mut best: Option<(usize, f32)> = None;
            for (i, l) in penalized.iter().enumerate() {
                let l = if l.is_finite() { *l } else { f32::NEG_INFINITY };
                if best.is_none_or(|(_, bl)| l > bl) {
                    best = Some((i, l));
                }
            }
            best.map(|(i, _)| i as u32)
                .ok_or_else(|| BackendError::Shape("greedy: empty logits row".into()))?
        } else {
            let sp = SamplingParams {
                temperature: params.temperature,
                top_p: params.top_p,
                top_k: params.top_k,
                repetition_penalty: params.repeat_penalty,
                // Поток несёт `rng_state` (сид уже материализован выше) —
                // второй источник сида здесь дал бы двойной шаг LCG.
                seed: None,
            };
            top_p_sample_vec_history(last, &sp, &history, &mut rng_state)
        };

        if let Some(r) = check_stop(next, tok.eos, &acc, &params.stop_strings) {
            finish = Some(r);
            break;
        }
        let piece = tok.decode_piece(next);
        let delta = decoder.push(piece.as_bytes());
        if !delta.is_empty() {
            acc.push_str(&delta);
            if !on_event(GenerateEvent::Delta(delta)) {
                return Ok(generated);
            }
        }
        if !on_event(GenerateEvent::Token(next)) {
            return Ok(generated);
        }
        generated.push(next);
        history.push(next);

        // Спек-декодинг (4.2b): проверка черновиков ОДНИМ forward'ом
        // (decode_spec: rejection sampling по Leviathan). Раньше здесь был
        // только счётчик с off-by-one (сравнение черновика следующей
        // позиции с уже выпущенным токеном) — 0% приёмки на любом драфтере.
        if let (Some(drafter), true) = (&params.drafter, params.spec_gamma > 0) {
            let (draft_tokens, draft_probs) = {
                let mut dr = drafter.lock().unwrap();
                dr.draft(backend, &history, params.spec_gamma)
            };
            if !draft_tokens.is_empty() {
                let tlayers: Vec<usize> = {
                    let dr = drafter.lock().unwrap();
                    dr.target_layers().to_vec()
                };
                let out = model.decode_spec(
                    backend,
                    next,
                    &draft_tokens,
                    &draft_probs,
                    &mut rng_state,
                    &tlayers,
                )?;
                if !tlayers.is_empty() {
                    let pos = model.caches[0].pos;
                    let n_ok = out.n_draft_accepted + 1; // якорь + проверенные
                    let mut dr = drafter.lock().unwrap();
                    let _ = dr.after_extract(backend, &out.extract, pos, n_ok);
                }
                spec_stats.drafts += params.spec_gamma as u64;
                let nv = out.n_draft_verified;
                spec_stats.accepted += nv as u64;
                spec_stats.rejected += (params.spec_gamma - nv) as u64;
                // Проверенные черновики — настоящие токены: стримим их.
                for &t in &out.accepted[1..] {
                    if let Some(r) = check_stop(t, tok.eos, &acc, &params.stop_strings) {
                        finish = Some(r);
                        break;
                    }
                    let piece = tok.decode_piece(t);
                    let delta = decoder.push(piece.as_bytes());
                    if !delta.is_empty() {
                        acc.push_str(&delta);
                        if !on_event(GenerateEvent::Delta(delta)) {
                            return Ok(generated);
                        }
                    }
                    if !on_event(GenerateEvent::Token(t)) {
                        return Ok(generated);
                    }
                    generated.push(t);
                    history.push(t);
                }
                logits = out.logits;
                if finish.is_some() {
                    break;
                }
            } else {
                logits = model.decode(backend, next)?;
            }
        } else {
            logits = model.decode(backend, next)?;
        }
    }
    if finish.is_none() {
        finish = Some(FinishReason::MaxTokens);
    }
    let tail = decoder.finish();
    if !tail.is_empty() && !on_event(GenerateEvent::Delta(tail)) {
        return Ok(generated);
    }
    if params.spec_gamma > 0 && params.drafter.is_some() {
        on_event(GenerateEvent::SpecStats(spec_stats));
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
            |ev| {
                match ev {
                    GenerateEvent::Delta(_) => deltas += 1,
                    GenerateEvent::Done(r) => done = Some(r),
                    _ => {}
                }
                true
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
                true
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
