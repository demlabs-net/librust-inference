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

/// Транспорт PP-стадии (7.5): координатор отдаёт соседу активации [t, dim] и
/// получает выход его диапазона слоёв. Реализацию даёт сервис (кластерный
/// канал с TLS); librust-inference знает только этот контракт.
pub trait StageTransport: Send + Sync {
    /// `reset` — начало новой последовательности (стадия сбрасывает свои KV:
    /// её кэш мог остаться от прогрева/прошлой генерации), `pos` — позиция
    /// первого токена запроса (стадия сверяет её со своим кэшем).
    fn forward_stage(
        &self,
        from: usize,
        upto: Option<usize>,
        norm_out: bool,
        reset: bool,
        pos: usize,
        t: usize,
        hidden: Vec<f32>,
    ) -> Result<Vec<f32>, String>;
}

/// Разбиение модели для PP (7.5): локально считаются слои [0..from), сосед —
/// [from..] + out_norm. Эмбеддинг и lm_head всегда у координатора.
#[derive(Clone)]
pub struct PpSplit {
    pub from: usize,
    pub transport: std::sync::Arc<dyn StageTransport>,
    /// Первый вызов стадии в этой генерации (шлётся как reset: у соседа
    /// кэш мог остаться от прогрева/прошлой генерации).
    pub first: std::sync::Arc<std::sync::atomic::AtomicBool>,
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
    /// PP-разбиение (7.5): None — обычная локальная генерация.
    pub pp: Option<PpSplit>,
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
            pp: None,
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

/// Один forward с PP-разбиением (7.5): эмбеддинг и lm_head — локально,
/// слои [0..from) — локально, [from..] + out_norm — на соседней ноде.
/// Позицию стадии ведёт локальный кэш [0..from) (sync_rope_to_cache) — то же
/// правило, что у prefill/decode без PP: префилл +t, декод-итерация +1.
fn pp_forward(
    model: &mut Qwen3Model,
    backend: &dyn Backend,
    pp: &PpSplit,
    ids: &[u32],
) -> Result<crate::backend::Tensor, BackendError> {
    let t = ids.len();
    let dim = model.cfg.dim;
    // Позиция ДО локального forward = стартовая позиция последовательности
    // для стадии (её кэш должен стоять там же). Запоминается и для ОТКАТА:
    // если стадия не приняла строки, локальные слои обязаны вернуться сюда,
    // иначе стороны разъезжаются на токен и все следующие вызовы падают на
    // проверке позиции (наблюдалось в кросс-машинном прогоне).
    let start_pos = model.caches[0].pos;
    let reset = pp
        .first
        .swap(false, std::sync::atomic::Ordering::Relaxed);
    let x = model.embed_ids(backend, ids)?;
    let mid = model.forward_layers_range(backend, x, t, None, Some(pp.from), 0, None, false)?;
    let hidden = backend.to_cpu_f32(&mid)?;
    let out = match pp.transport.forward_stage(pp.from, None, true, reset, start_pos, t, hidden) {
        Ok(v) => v,
        Err(e) => {
            // Транзакционный откат: строки, которые стадия не приняла,
            // снимаются и с локальных слоёв координатора.
            for c in model.caches.iter_mut() {
                c.truncate(start_pos);
            }
            return Err(BackendError::Shape(format!("PP-стадия: {e}")));
        }
    };
    if out.len() != t * dim {
        return Err(BackendError::Shape(format!(
            "PP-стадия: вернула {} элементов, ждали t*dim = {t}*{dim}",
            out.len()
        )));
    }
    let h = backend.from_cpu_f32(&out, vec![1, t, dim])?;
    // Локальный кэш [0..from) — источник истины по позиции: префилл даёт +t,
    // каждая декод-итерация +1 (как prefill/decode без PP).
    model.sync_rope_to_cache(0);
    model.lm_head(backend, &h)
}

/// Восстановление после сбоя стадии (7.11): обе стороны переигрывают ВСЮ
/// последовательность с чистого состояния — локальный сброс у координатора и
/// `reset` у соседа. Иначе позиции кэшей расходятся (координатор пересеял
/// последовательность локально, а у соседа остался старый хвост) и стадия
/// честно отвечает «позиция N != запрошенной M». Возвращает логиты последней
/// позиции; если стадия снова не сработала — ошибка (вызывающий уходит в
/// локальный откат).
fn pp_recover(
    model: &mut Qwen3Model,
    backend: &dyn Backend,
    pp: &PpSplit,
    seq: &[u32],
) -> Result<crate::backend::Tensor, BackendError> {
    model.reset_cache(backend)?;
    // Следующая стадия обязана начать последовательность заново.
    pp.first.store(true, std::sync::atomic::Ordering::Relaxed);
    pp_forward(model, backend, pp, seq)
}

/// Локальный forward после отказа PP (7.10). `seq` — ВСЯ обработанная
/// последовательность (промпт + порождённое + текущий токен): в PP-режиме
/// локально жили только слои [0..from), поэтому продолжать с текущим кэшем
/// нельзя — нужен полный локальный префилл с чистого кэша (логиты последней
/// строки — то, что вернул бы decode).
fn local_fallback(
    model: &mut Qwen3Model,
    backend: &dyn Backend,
    seq: &[u32],
) -> Result<crate::backend::Tensor, BackendError> {
    model.reset_cache(backend)?;
    if seq.len() > 1 {
        model.prefill(backend, seq)
    } else {
        model.decode(backend, seq[0])
    }
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
    let mut logits = if let Some(pp) = params.pp.clone() {
        // PP (7.5): средние слои — на соседе. Vision-путь PP пока не
        // поддержан (M-RoPE/позиции картинки требуют передачи метаданных).
        if params.vision.is_some() {
            return Err(BackendError::Shape(
                "PP-разбиение не поддержано вместе с vision-промптом".into(),
            ));
        }
        match pp_forward(model, backend, &pp, ids) {
            Ok(lg) => lg,
            Err(e) => {
                // 7.10: сосед недоступен/ошибся — деградируем в локальную
                // генерацию (модель у координатора полная), а не рвём запрос.
                tracing::warn!(error = %e, "PP-стадия недоступна — локальная генерация");
                local_fallback(model, backend, ids)?
            }
        }
    } else if let Some(v) = &params.vision {
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

    // Диагностика параметров (LLM_DBG_PARAMS=1): расхождения спек/жадной
    // генерации на живом сервисе искали именно сравнением эффективных
    // параметров — печатаем те, что реально доехали до петли.
    if std::env::var("LLM_DBG_PARAMS").is_ok() {
        eprintln!(
            "PARAMS temp={} top_p={} top_k={} rep={} spec_gamma={} seed={:?} max_tokens={} t={} ids={:?}",
            params.temperature,
            params.top_p,
            params.top_k,
            params.repeat_penalty,
            params.spec_gamma,
            params.seed,
            params.max_tokens,
            ids.len(),
            &ids[..ids.len().min(12)],
        );
    }
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
            // Единственная реализация жадного шага (sampling::greedy_next):
            // её же использует жадная верификация спека — расхождение
            // (штраф в одном месте и отсутствие в другом) давало разные
            // траектории спек- и обычной генерации.
            if last.is_empty() {
                return Err(BackendError::Shape("greedy: empty logits row".into()));
            }
            crate::inference::sampling::greedy_next(last, &history, params.repeat_penalty)
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
        // PP (7.5) и спек-декод несовместимы: верификация черновиков гоняет
        // forward ЦЕЛИКОМ и опирается на полный локальный KV, а в PP-режиме
        // локально живут только слои [0..from). Молча включать нельзя —
        // получились бы черновики, проверенные по пустому кэшу.
        if let (Some(drafter), true) = (
            &params.drafter,
            params.spec_gamma > 0 && params.pp.is_none(),
        ) {
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
                    // temperature <= 0: та же жадная траектория, что у decode
                    // (включая штраф за повторы — верификация без него
                    // принимала бы токены, которых жадный путь не выбрал бы).
                    params.temperature <= 0.0,
                    &history,
                    params.repeat_penalty,
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
                // Окно закрылось ПЕРЕСЭМПЛОМ (n_draft_accepted < γ): этот
                // токен в KV-кэше НЕ лежит (SpecOutcome), а строка из
                // `out.logits` описывает контекст с ОТКЛОНЁННЫМ черновиком —
                // сэмплирование из неё повторяло пересэмпл в выводе. Доводим
                // пересэмпл через forward: кэш и логиты снова согласованы.
                let resampled = out.n_draft_accepted < draft_tokens.len();
                logits = if resampled {
                    match out.accepted.last() {
                        Some(&last) => model.decode(backend, last)?,
                        None => out.logits,
                    }
                } else {
                    out.logits
                };
                if finish.is_some() {
                    break;
                }
            } else if let Some(pp) = params.pp.as_ref() {
                logits = match pp_forward(model, backend, pp, &[next]) {
                    Ok(lg) => lg,
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            "PP-стадия не сработала — переигрываем последовательность"
                        );
                        match pp_recover(model, backend, pp, &history) {
                            Ok(lg) => lg,
                            Err(e2) => {
                                tracing::warn!(error = %e2, "PP-стадия недоступна — локальный decode");
                                local_fallback(model, backend, &history)?
                            }
                        }
                    }
                };
            } else {
                logits = model.decode(backend, next)?;
            }
        } else if let Some(pp) = params.pp.as_ref() {
            logits = match pp_forward(model, backend, pp, &[next]) {
                Ok(lg) => lg,
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "PP-стадия не сработала — переигрываем последовательность"
                    );
                    match pp_recover(model, backend, pp, &history) {
                        Ok(lg) => lg,
                        Err(e2) => {
                            tracing::warn!(error = %e2, "PP-стадия недоступна — локальный decode");
                            local_fallback(model, backend, &history)?
                        }
                    }
                }
            };
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
