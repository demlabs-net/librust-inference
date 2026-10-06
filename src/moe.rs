//! MoE-каркас (A3 плана «архитектуры + тиринг»): роутер и экспертный FFN.
//!
//! Порт llama.cpp `build_moe_ffn` (`src/llama-graph.cpp`, эталон 5266f24da) для
//! схем без общих экспертов и без группового отбора (Qwen3-MoE, OLMoE):
//!
//! ```text
//! logits = gate_inp · x                     [n_expert, T]
//! probs  = гейтинг(logits)                  (softmax по ВСЕМ экспертам)
//! topk   = argsort_top_k(probs, n_used)     (по убыванию)
//! w      = probs[topk]                      (несмещённые!)
//! w     /= Σw                               (norm_topk_prob, clamp 6.1e-5)
//! w     *= expert_weights_scale             (если не 0 и не 1)
//! out    = Σ_j w_j · down_j( silu(gate_j x) · up_j x )
//! ```
//!
//! Гейтинг и top-k считаются на ХОСТЕ: логиты роутера — это T×E чисел
//! (OLMoE: 64, Qwen3-MoE: 128), перенос копеечный, а на хосте видна точная
//! семантика (`ggml_argsort_top_k` — по убыванию, ничьи по возрастанию
//! индекса), которую иначе пришлось бы повторять кернелом.
//!
//! Экспертные веса — 3-Д тензоры GGUF `[n_expert, rows, cols]`; срез одного
//! эксперта берётся БЕЗ копии (`view_offset`/`view_offset_bytes` для
//! блочно-квантованных), поэтому нативный квант (Q4_K/Q6_K/…) сохраняется:
//! никакого «распакуй все эксперты в f16».
//!
//! Что НЕ поддержано этим модулем (и почему это громкая ошибка, а не тихая
//! подмена): общие эксперты (`ffn_*_shexp`), биасы роутера/экспертов, отбор
//! по группам (DeepSeek), латентная проекция экспертов. Всё это — следующие
//! шаги волны A3, и до них такие модели обязаны падать с объяснением.

use crate::backend::{Backend, Tensor};
use crate::error::BackendError;
use crate::gguf::{GgufFile, GgufValue};
use crate::weights_map::WeightsMap;

/// Функция гейтинга — llama.cpp `llama_expert_gating_func_type` (значения
/// совпадают с метаданными GGUF `<arch>.expert_gating_func`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gating {
    /// softmax по всем экспертам (Qwen3-MoE, OLMoE, Mixtral).
    Softmax = 1,
    /// sigmoid (DeepSeek-V3, Llama-4).
    Sigmoid = 2,
    /// softmax применяется к ВЕСАМ top-k, а не к логитам (Grok).
    SoftmaxWeight = 3,
    /// sqrt(softplus(x)) (Nemotron).
    SqrtSoftplus = 4,
}

impl Gating {
    fn from_code(code: u32) -> Option<Self> {
        match code {
            1 => Some(Gating::Softmax),
            2 => Some(Gating::Sigmoid),
            3 => Some(Gating::SoftmaxWeight),
            4 => Some(Gating::SqrtSoftplus),
            _ => None,
        }
    }
}

/// Конфиг MoE-ветви из метаданных GGUF (`<arch>.expert_*`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoeConfig {
    /// Всего экспертов (n_expert).
    pub n_expert: usize,
    /// Сколько экспертов на токен (n_expert_used; top-k).
    pub n_expert_used: usize,
    /// Длина FFN одного эксперта (`expert_feed_forward_length`).
    pub ffn_dim: usize,
    pub gating: Gating,
    /// Нормировать веса выбранных экспертов к сумме 1 (Qwen3-MoE: да).
    pub norm_topk_prob: bool,
    /// Масштаб весов (0 и 1 = не применять — как llama.cpp).
    pub expert_weights_scale: f32,
    /// Число групп отбора и сколько групп берётся (DeepSeek: 8/4);
    /// 0/1 — группового отбора нет.
    pub n_expert_groups: usize,
    pub n_group_used: usize,
}

/// Ключи hparams, общие для всех MoE-архитектур llama.cpp
/// (`LLM_KV_EXPERT_*`, src/llama-arch.cpp).
const KEYS: &[(&str, &str)] = &[
    ("count", "expert_count"),
    ("used", "expert_used_count"),
    ("ffn", "expert_feed_forward_length"),
    ("gating", "expert_gating_func"),
    ("scale", "expert_weights_scale"),
    ("norm", "expert_weights_norm"),
    ("groups", "expert_group_count"),
    ("groups_used", "expert_group_used_count"),
];

/// Разбор значения "число или массив из n_layers": llama.cpp читает
/// `expert_used_count` через `get_key_or_arr`. Массив с РАЗНЫМИ значениями мы
/// не поддержали бы (роутер и граф у нас общие на слои) — лучше громкая
/// ошибка, чем молча взятое первое число.
fn scalar_u32(
    file: &GgufFile<'_>,
    key: &str,
    n_layers: usize,
) -> Result<Option<u32>, BackendError> {
    let Some(v) = file.get(key) else {
        return Ok(None);
    };
    if let Some(x) = v.as_u32() {
        return Ok(Some(x));
    }
    if let Some(arr) = v.as_array() {
        let vals: Vec<u32> = arr.iter().filter_map(GgufValue::as_u32).collect();
        if vals.len() == n_layers && !vals.is_empty() {
            let first = vals[0];
            if vals.iter().all(|&x| x == first) {
                return Ok(Some(first));
            }
            return Err(BackendError::UnsupportedFormat(format!(
                "{key}: значения по слоям различаются ({vals:?}) — не поддержано \
                 (роутер у нас общий на слои)"
            )));
        }
    }
    Err(BackendError::Shape(format!(
        "{key}: ожидалось число или массив из {n_layers} чисел"
    )))
}

fn as_f32(v: &GgufValue) -> Option<f32> {
    match v {
        GgufValue::F32(f) => Some(*f),
        GgufValue::F64(f) => Some(*f as f32),
        GgufValue::U32(x) => Some(*x as f32),
        GgufValue::I32(x) => Some(*x as f32),
        GgufValue::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

impl MoeConfig {
    /// Прочитать MoE-параметры архитектуры `arch`; None — модель не MoE.
    pub fn from_gguf(
        file: &GgufFile<'_>,
        arch: &str,
        n_layers: usize,
        dense_ffn: usize,
    ) -> Result<Option<Self>, BackendError> {
        let p = |k: &str| format!("{arch}.{k}");
        let Some(n_expert) = scalar_u32(file, &p(KEYS[0].1), n_layers)?.map(|v| v as usize) else {
            return Ok(None);
        };
        if n_expert == 0 {
            return Ok(None);
        }
        let n_expert_used = scalar_u32(file, &p(KEYS[1].1), n_layers)?
            .map(|v| v as usize)
            .ok_or_else(|| BackendError::Shape(format!("missing {}", p(KEYS[1].1))))?;
        let ffn_dim = scalar_u32(file, &p(KEYS[2].1), n_layers)?
            .map(|v| v as usize)
            .unwrap_or(dense_ffn);
        // Гейтинг: у большинства арх ключа нет — llama.cpp подставляет функцию
        // в графе (для qwen3moe/olmoe это softmax).
        let gating = match file.get(&p(KEYS[3].1)).and_then(GgufValue::as_u32) {
            Some(code) => Gating::from_code(code).ok_or_else(|| {
                BackendError::UnsupportedFormat(format!("{}: неизвестный код {code}", p(KEYS[3].1)))
            })?,
            None => Gating::Softmax,
        };
        let expert_weights_scale = file
            .get(&p(KEYS[4].1))
            .and_then(as_f32)
            .unwrap_or(0.0);
        let norm_topk_prob = file
            .get(&p(KEYS[5].1))
            .and_then(as_f32)
            .map(|v| v != 0.0)
            .unwrap_or(false);
        let n_expert_groups = scalar_u32(file, &p(KEYS[6].1), n_layers)?
            .map(|v| v as usize)
            .unwrap_or(0);
        let n_group_used = scalar_u32(file, &p(KEYS[7].1), n_layers)?
            .map(|v| v as usize)
            .unwrap_or(0);
        Ok(Some(Self {
            n_expert,
            n_expert_used,
            ffn_dim,
            gating,
            norm_topk_prob,
            expert_weights_scale,
            n_expert_groups,
            n_group_used,
        }))
    }

    /// Отбор экспертов и веса для ОДНОГО токена: логиты [E] → ids[k], веса[k].
    /// Порт `build_moe_ffn` (без групп): гейтинг → top-k → нормировка → масштаб.
    pub fn route_row(&self, logits: &[f32], ids: &mut Vec<u32>, weights: &mut Vec<f32>) {
        let e = self.n_expert;
        debug_assert_eq!(logits.len(), e);
        let mut probs = vec![0f32; e];
        match self.gating {
            Gating::Softmax => softmax_into(&logits[..e], &mut probs),
            Gating::Sigmoid => {
                for (o, x) in probs.iter_mut().zip(&logits[..e]) {
                    *o = 1.0 / (1.0 + (-*x).exp());
                }
            }
            // Сырые логиты: softmax применяется ниже, к весам top-k.
            Gating::SoftmaxWeight => probs.copy_from_slice(&logits[..e]),
            Gating::SqrtSoftplus => {
                for (o, x) in probs.iter_mut().zip(&logits[..e]) {
                    let sp = if *x > 20.0 { *x } else { (1.0 + x.exp()).ln() };
                    *o = sp.sqrt();
                }
            }
        }
        // top-k: по убыванию значения, при равенстве — меньший индекс первым
        // (устойчивая сортировка; ggml сортирует биты значения — эквивалентно).
        let k = self.n_expert_used.clamp(1, e);
        ids.clear();
        ids.extend(0..e as u32);
        ids.sort_by(|&a, &b| {
            probs[b as usize]
                .total_cmp(&probs[a as usize])
                .then(a.cmp(&b))
        });
        ids.truncate(k);
        weights.clear();
        weights.extend(ids.iter().map(|&i| probs[i as usize]));
        if matches!(self.gating, Gating::SoftmaxWeight) {
            let w = weights.clone();
            softmax_into(&w, weights);
        }
        if self.norm_topk_prob {
            let sum = weights.iter().sum::<f32>().max(6.103515625e-5);
            for w in weights.iter_mut() {
                *w /= sum;
            }
        }
        if self.expert_weights_scale != 0.0 && self.expert_weights_scale != 1.0 {
            for w in weights.iter_mut() {
                *w *= self.expert_weights_scale;
            }
        }
    }
}

/// softmax с вычитанием максимума (как `ggml_soft_max`).
fn softmax_into(x: &[f32], out: &mut [f32]) {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for (o, v) in out.iter_mut().zip(x) {
        let e = (v - m).exp();
        *o = e;
        sum += e;
    }
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for o in out.iter_mut() {
        *o *= inv;
    }
}

/// Экспертный тензор GGUF `[n_expert, rows, cols]`: эксперты лежат подряд,
/// срез одного эксперта — это `rows` подряд идущих строк.
#[derive(Debug, Clone)]
pub struct ExpertTensor {
    pub w: Tensor,
    pub rows: usize,
    pub cols: usize,
    /// Байт в строке для блочно-квантованного хранения (None — срез в
    /// элементах; на CPU-бэкенде веса лежат в f32, и блочного среза нет).
    row_bytes: Option<usize>,
}

impl ExpertTensor {
    /// Загрузить 3-Д экспертный тензор (нативным квантом, без копий и
    /// без распаковки). `shape3` — [n_expert, rows, cols] из модели.
    pub fn load(
        backend: &dyn Backend,
        wm: &WeightsMap<'_>,
        name: &str,
        n_expert: usize,
        rows: usize,
        cols: usize,
    ) -> Result<Self, BackendError> {
        let wt = wm
            .get(name)
            .ok_or_else(|| BackendError::Shape(format!("missing tensor {name}")))?;
        let expect = [n_expert as u64, rows as u64, cols as u64];
        if wt.shape.as_slice() != expect {
            return Err(BackendError::Shape(format!(
                "{name}: ожидался 3-Д [{n_expert}, {rows}, {cols}], в файле {:?}",
                wt.shape
            )));
        }
        let w = crate::qwen3::Qwen3Model::upload_q(
            backend,
            wm,
            name,
            vec![n_expert * rows, cols],
        )?;
        // Блочный срез — только если бэкенд РЕАЛЬНО держит тензор в packed
        // виде: CPU грузит f32, и байтовое смещение там было бы в 4 раза
        // меньше нужного.
        let row_bytes = if w.is_block_quantized() {
            let rb = wt.format.block_row_bytes(cols);
            debug_assert!(rb.is_some(), "{name}: квант без блочной раскладки?");
            rb
        } else {
            None
        };
        Ok(Self {
            w,
            rows,
            cols,
            row_bytes,
        })
    }

    /// Срез эксперта `e` — [rows, cols], без копии.
    pub fn expert(&self, e: usize) -> Tensor {
        match self.row_bytes {
            Some(rb) => self.w.view_offset_bytes(
                vec![self.rows, self.cols],
                e * self.rows * rb,
            ),
            None => self
                .w
                .view_offset(vec![self.rows, self.cols], e * self.rows * self.cols),
        }
    }
}

/// Ветвь MoE слоя: роутер + эксперты (up/gate/down).
#[derive(Debug, Clone)]
pub struct MoeLayer {
    pub cfg: MoeConfig,
    pub dim: usize,
    /// Роутер `ffn_gate_inp` [n_expert, dim].
    pub gate_inp: Tensor,
    /// `ffn_up_exps` [E, ffn, dim].
    pub up: ExpertTensor,
    /// `ffn_gate_exps` [E, ffn, dim].
    pub gate: ExpertTensor,
    /// `ffn_down_exps` [E, dim, ffn].
    pub down: ExpertTensor,
}

impl MoeLayer {
    /// Собрать ветвь слоя `prefix` (например `blk.3.`) из весов; отсутствие
    /// обязательного тензора — ошибка. Незнакомые части MoE (общие эксперты,
    /// биасы роутера/экспертов) отвергаются явно: молча пропущенный тензор
    /// дал бы другой ответ вместо ошибки.
    pub fn load(
        backend: &dyn Backend,
        wm: &WeightsMap<'_>,
        prefix: &str,
        cfg: MoeConfig,
        dim: usize,
        dense_ffn: usize,
    ) -> Result<Self, BackendError> {
        for suffix in [
            "ffn_gate_shexp.weight",
            "ffn_up_shexp.weight",
            "ffn_down_shexp.weight",
            "ffn_gate_inp.bias",
            "ffn_gate_exps.bias",
            "ffn_up_exps.bias",
            "ffn_down_exps.bias",
        ] {
            if wm.get(&format!("{prefix}{suffix}")).is_some() {
                return Err(BackendError::UnsupportedFormat(format!(
                    "MoE {prefix}: {suffix} не поддержан (общие эксперты/биасы — \
                     следующий шаг A3)"
                )));
            }
        }
        if cfg.n_expert_groups > 1 {
            return Err(BackendError::UnsupportedFormat(format!(
                "MoE {prefix}: групповой отбор экспертов ({} групп, берётся {}) \
                 не поддержан — шаг A3",
                cfg.n_expert_groups, cfg.n_group_used
            )));
        }
        let ffn = if cfg.ffn_dim == 0 { dense_ffn } else { cfg.ffn_dim };
        let cfg = MoeConfig { ffn_dim: ffn, ..cfg };
        let n = cfg.n_expert;
        let t = |s: &str| format!("{prefix}{s}");
        Ok(Self {
            cfg,
            dim,
            // Роутер — ВСЕГДА f32: от его логитов зависит ВЫБОР экспертов, а
            // llama.cpp держит gate_inp как есть (в GGUF — F32) и ничего не
            // округляет. Наша f16-загрузка (upload_w) теряла биты, и на
            // близких значениях логитов выбирался ДРУГОЙ эксперт — текст
            // расходился через десятки идентичных токенов.
            gate_inp: {
                let wt = wm
                    .get(&t("ffn_gate_inp.weight"))
                    .ok_or_else(|| BackendError::Shape("missing ffn_gate_inp.weight".into()))?;
                backend.from_cpu_f32(&crate::weights_map::dequant_f32(wt), vec![n, dim])?
            },
            up: ExpertTensor::load(backend, wm, &t("ffn_up_exps.weight"), n, ffn, dim)?,
            gate: ExpertTensor::load(backend, wm, &t("ffn_gate_exps.weight"), n, ffn, dim)?,
            down: ExpertTensor::load(backend, wm, &t("ffn_down_exps.weight"), n, dim, ffn)?,
        })
    }

    /// Forward: x [1, T, dim] → y [1, T, dim].
    ///
    /// Пока ПОТОКЕННО (T×k×3 матвектора). Это то же самое множество
    /// произведений, что у `mul_mat_id` llama.cpp (группировка токенов по
    /// эксперту меняет только порядок вызовов, не математику); группировка —
    /// оптимизация префилла, шаг A3.4.
    pub fn forward(
        &self,
        backend: &dyn Backend,
        x: &Tensor,
    ) -> Result<Tensor, BackendError> {
        let (b, t, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        if b != 1 || d != self.dim {
            return Err(BackendError::Shape(format!(
                "MoeLayer: ожидался вход [1, T, {d}], получен {b}x{t}x{d}"
            )));
        }
        let e = self.cfg.n_expert;
        let ffn = self.cfg.ffn_dim;
        let x2 = x.view(vec![t, d]);

        // 1. Логиты роутера [T, E] (gemm_nt: x[T,dim] × gate_inp[E,dim]ᵀ).
        let logits = backend.zeros(vec![t, e])?;
        backend.gemm_nt(&x2, &self.gate_inp, &logits)?;
        let host = backend.to_cpu_f32(&logits)?;

        // 2. Маршрутизация на хосте.
        let mut routed: Vec<(Vec<u32>, Vec<f32>)> = Vec::with_capacity(t);
        let mut ids = Vec::with_capacity(self.cfg.n_expert_used);
        let mut ws = Vec::with_capacity(self.cfg.n_expert_used);
        let dbg = std::env::var("MOE_DEBUG").is_ok();
        for ti in 0..t {
            self.cfg.route_row(&host[ti * e..(ti + 1) * e], &mut ids, &mut ws);
            if dbg {
                eprintln!(
                    "ENG router[t={ti}] ids={ids:?} w={ws:?}",
                );
            }
            routed.push((ids.clone(), ws.clone()));
        }

        // 3. Эксперты: по токену, по выбранному эксперту — срезы без копий.
        let acc = backend.zeros(vec![t, d])?;
        for (ti, (ids, ws)) in routed.iter().enumerate() {
            let x_t = x2.view_offset(vec![1, d], ti * d);
            let acc_t = acc.view_offset(vec![1, d], ti * d);
            for (&eid, &w) in ids.iter().zip(ws.iter()) {
                if w == 0.0 {
                    continue;
                }
                let y_up = backend.zeros(vec![1, ffn])?;
                backend.gemm_nt(&x_t, &self.up.expert(eid as usize), &y_up)?;
                let g = backend.zeros(vec![1, ffn])?;
                backend.gemm_nt(&x_t, &self.gate.expert(eid as usize), &g)?;
                let act = backend.zeros(vec![1, ffn])?;
                backend.silu(&g, &act)?;
                let prod = backend.zeros(vec![1, ffn])?;
                backend.mul(&act, &y_up, &prod)?;
                let y_dn = backend.zeros(vec![1, d])?;
                backend.gemm_nt(&prod, &self.down.expert(eid as usize), &y_dn)?;
                backend.add_scaled(&acc_t, &y_dn, &acc_t, w)?;
            }
        }
        Ok(acc.view(vec![1, t, d]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(k: usize, norm: bool) -> MoeConfig {
        MoeConfig {
            n_expert: 4,
            n_expert_used: k,
            ffn_dim: 8,
            gating: Gating::Softmax,
            norm_topk_prob: norm,
            expert_weights_scale: 0.0,
            n_expert_groups: 0,
            n_group_used: 0,
        }
    }

    /// Роутер: softmax по всем, top-k по убыванию, веса — ИЗ НЕСМЕЩЁННЫХ probs.
    #[test]
    fn route_softmax_topk_weights() {
        let c = cfg(2, false);
        let mut ids = Vec::new();
        let mut w = Vec::new();
        c.route_row(&[1.0, 3.0, 0.0, -1.0], &mut ids, &mut w);
        assert_eq!(ids, vec![1, 0]);
        // softmax: e^3/(e^3+e^1+e^0+e^-1), e^1/(...)
        let den = 3f32.exp() + 1f32.exp() + 1.0 + (-1f32).exp();
        assert!((w[0] - 3f32.exp() / den).abs() < 1e-6, "{w:?}");
        assert!((w[1] - 1f32.exp() / den).abs() < 1e-6, "{w:?}");
        assert!((w.iter().sum::<f32>() - (3f32.exp() + 1f32.exp()) / den).abs() < 1e-6);
    }

    /// norm_topk_prob (Qwen3-MoE): веса делятся на свою сумму → сумма 1.
    #[test]
    fn route_normalizes_topk() {
        let c = cfg(2, true);
        let mut ids = Vec::new();
        let mut w = Vec::new();
        c.route_row(&[1.0, 3.0, 0.0, -1.0], &mut ids, &mut w);
        assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6, "{w:?}");
        // Отношение весов сохранилось: exp(3)/exp(1) = e²
        assert!((w[0] / w[1] - 2f32.exp()).abs() < 1e-4, "{w:?}");
    }

    /// Ничьи: меньший индекс первым (устойчиво), k ограничен числом экспертов.
    #[test]
    fn route_ties_and_k_clamp() {
        let mut c = cfg(8, false);
        let mut ids = Vec::new();
        let mut w = Vec::new();
        c.route_row(&[2.0, 2.0, 2.0, 2.0], &mut ids, &mut w);
        assert_eq!(ids, vec![0, 1, 2, 3], "все четыре равны — порядок по индексу");
        assert_eq!(w.len(), 4);
        // k = 0 не должен вернуть пустой список (иначе токен выпадет из FFN)
        c.n_expert_used = 0;
        c.route_row(&[1.0, 0.0, 0.0, 0.0], &mut ids, &mut w);
        assert_eq!(ids, vec![0]);
    }

    /// Масштаб весов (deepseek-подобные): применяется после нормировки.
    #[test]
    fn route_applies_weights_scale() {
        let mut c = cfg(2, true);
        c.expert_weights_scale = 0.5;
        let mut ids = Vec::new();
        let mut w = Vec::new();
        c.route_row(&[1.0, 3.0, 0.0, -1.0], &mut ids, &mut w);
        assert!((w.iter().sum::<f32>() - 0.5).abs() < 1e-6, "{w:?}");
    }

    /// sigmoid-гейтинг (DeepSeek-V3-подобные): веса — сигмоиды, не softmax.
    #[test]
    fn route_sigmoid() {
        let mut c = cfg(2, false);
        c.gating = Gating::Sigmoid;
        let mut ids = Vec::new();
        let mut w = Vec::new();
        c.route_row(&[0.0, 1.0, -1.0, 0.5], &mut ids, &mut w);
        assert_eq!(ids, vec![1, 3]);
        assert!((w[0] - 1.0 / (1.0 + (-1f32).exp())).abs() < 1e-6, "{w:?}");
    }
}
