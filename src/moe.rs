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
        let expert_weights_scale = file.get(&p(KEYS[4].1)).and_then(as_f32).unwrap_or(0.0);
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
            let sum = weights.iter().sum::<f32>().max(6.103_515_6e-5);
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
        let w = crate::qwen3::Qwen3Model::upload_q(backend, wm, name, vec![n_expert * rows, cols])?;
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

    /// Срез эксперта `e` — [rows, cols], без копии. Горячий путь MoE идёт
    /// через [`ExpertTensor::w`] целиком (`moe_experts_mv`), а срез нужен
    /// сверкам/будущим путям (смена резидентности экспертов в Части B).
    pub fn expert(&self, e: usize) -> Tensor {
        match self.row_bytes {
            Some(rb) => self
                .w
                .view_offset_bytes(vec![self.rows, self.cols], e * self.rows * rb),
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
        let ffn = if cfg.ffn_dim == 0 {
            dense_ffn
        } else {
            cfg.ffn_dim
        };
        let cfg = MoeConfig {
            ffn_dim: ffn,
            ..cfg
        };
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
    pub fn forward(&self, backend: &dyn Backend, x: &Tensor) -> Result<Tensor, BackendError> {
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
            self.cfg
                .route_row(&host[ti * e..(ti + 1) * e], &mut ids, &mut ws);
            if dbg {
                eprintln!("ENG router[t={ti}] ids={ids:?} w={ws:?}",);
            }
            routed.push((ids.clone(), ws.clone()));
        }

        // INFER_DUMP_MOE=<файл> — побитовый дамп роутер-логитов и весов для
        // сверки с scripts/oracle_dump.cpp (тот же формат: заголовок TENSOR +
        // hex-биты по 8 на строку). Вызовы нумеруются глобально: 16 MoE-слоёв
        // на forward, порядок вызовов = порядок слоёв.
        if let Ok(path) = std::env::var("INFER_DUMP_MOE") {
            use std::io::Write as _;
            static CALL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let call = CALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
            {
                let mut dump = |name: &str, ne: &str, vals: &[f32]| {
                    let _ = writeln!(f, "STEP {call} TENSOR {name} ne=[{ne}]");
                    for (i, v) in vals.iter().enumerate() {
                        let _ = write!(
                            f,
                            "{:08x}{}",
                            v.to_bits(),
                            if i % 8 == 7 { "\n" } else { " " }
                        );
                    }
                    let _ = writeln!(f);
                };
                // Логиты в эталонном порядке [E, t] (токами по столбцам).
                let mut lg = Vec::with_capacity(t * e);
                for ti in 0..t {
                    lg.extend_from_slice(&host[ti * e..(ti + 1) * e]);
                }
                dump("eng_logits", &format!("{e},{t},1,1"), &lg);
                let mut wt: Vec<f32> = Vec::with_capacity(t * self.cfg.n_expert_used);
                let mut idv: Vec<f32> = Vec::with_capacity(wt.capacity());
                for (eid, w) in routed.iter().flat_map(|(i, w)| i.iter().zip(w.iter())) {
                    idv.push(*eid as f32);
                    wt.push(*w);
                }
                dump(
                    "eng_topk_ids",
                    &format!("{},1,1,1", t * self.cfg.n_expert_used),
                    &idv,
                );
                dump(
                    "eng_weights",
                    &format!("1,{},1,1", self.cfg.n_expert_used),
                    &wt,
                );
            }
        }

        // 3. Проекции экспертов. Id-кернелы (mul_mat_id) — ПУТЬ ПО УМОЛЧАНИЮ:
        // с портированием mv_id на r2-раскладку эталонного mul_mv они считают
        // ровно тот же порядок по K, что по-слотовый matvec (тест
        // id_kernel_matches_per_expert_matvec — up/gate побитово; e2e OLMoE
        // INFER_MOE_NO_ID=1 против id-пути — байт-в-байт), и на порядок
        // дешевле по вызовам/аллокациям (3·k вызовов на токен свёрнуты в 3).
        // Прежний скалярный порядок mv_id (ломал паритет на OLMoE) устранён.
        // INFER_MOE_NO_ID=1 — по-слотовый путь (A/B и бэкенды без mul_mat_id;
        // на UnsupportedFormat откат происходит сам).
        let id_path = std::env::var("INFER_MOE_NO_ID").is_err();
        if !id_path {
            return self.ffn_per_slot(backend, &x2, &routed, t, d, ffn);
        }
        match self.ffn_id(backend, &x2, &logits, &routed, t, d, ffn, e) {
            Err(BackendError::UnsupportedFormat(_)) => {
                self.ffn_per_slot(backend, &x2, &routed, t, d, ffn)
            }
            r => r,
        }
    }

    /// Страховочный путь: по одному matvec на (токен, слот) без id-кернелов.
    fn ffn_per_slot(
        &self,
        backend: &dyn Backend,
        x2: &Tensor,
        routed: &[(Vec<u32>, Vec<f32>)],
        t: usize,
        d: usize,
        ffn: usize,
    ) -> Result<Tensor, BackendError> {
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

    /// Путь по умолчанию: ОДИН вызов mul_mat_id на проекцию на токен —
    /// внутри кернела перебираются слоты top-k. Раньше на токен слоя
    /// приходилось 3·k вызовов и столько же аллокаций буферов (384 на токен
    /// у OLMoE) — это и было узкое место декода (40 против 208 tok/s).
    fn ffn_id(
        &self,
        backend: &dyn Backend,
        x2: &Tensor,
        logits: &Tensor,
        routed: &[(Vec<u32>, Vec<f32>)],
        t: usize,
        d: usize,
        ffn: usize,
        e: usize,
    ) -> Result<Tensor, BackendError> {
        let k = self.cfg.n_expert_used.min(e);
        // Роутер на устройстве: без чтения логитов на CPU (host-путь стоил
        // синхронизацию GPU→CPU на слой на токен — 16 на токен у OLMoE).
        // Доступен только для softmax-гейтинга без групп; остальные схемы
        // идут host-путём (routed уже посчитан выше).
        let device_ids = backend.zeros(vec![t * k]).ok();
        let device_wts = backend.zeros(vec![t * k]).ok();
        let mut device_route = false;
        if let (Some(di), Some(dw)) = (&device_ids, &device_wts)
            && matches!(self.cfg.gating, Gating::Softmax)
            && self.cfg.n_expert_groups <= 1
        {
            device_route = backend
                .moe_route(
                    logits,
                    di,
                    dw,
                    k,
                    self.cfg.norm_topk_prob,
                    self.cfg.expert_weights_scale,
                )
                .is_ok();
        }
        // Эталон при ≥32 токенах ubatch считает mul_mat_id кернелом mm_id
        // (ggml_metal_op_mul_mat_id_use_mm), у которого свой порядок
        // суммирования — без него префилл MoE не битуется. Пакетный путь на
        // весь чанк; UnsupportedFormat (чужой квант/бэкенд) → по-слотовый цикл.
        if t >= 32 && std::env::var("INFER_MOE_NO_MM").is_err() {
            match self.ffn_id_mm(
                backend,
                x2,
                routed,
                t,
                d,
                ffn,
                e,
                k,
                device_route,
                device_ids.as_ref(),
                device_wts.as_ref(),
            ) {
                Err(BackendError::UnsupportedFormat(_)) => {}
                r => return r,
            }
        }
        let acc = backend.zeros(vec![t, d])?;
        for ti in 0..t {
            let x_t = x2.view_offset(vec![1, d], ti * d);
            // ids/weights: с устройства (роутер посчитал) или с хоста (fallback).
            let (ids_t, wts_t) = if device_route {
                (
                    device_ids
                        .as_ref()
                        .expect("device_ids есть")
                        .view_offset(vec![k], ti * k),
                    device_wts
                        .as_ref()
                        .expect("device_wts есть")
                        .view_offset(vec![k], ti * k),
                )
            } else {
                let (ids, ws) = &routed[ti];
                (
                    backend.from_cpu_u32(ids, vec![k])?,
                    backend.from_cpu_f32(ws, vec![k])?,
                )
            };
            // up/gate: вход общий (строка токена), выход — по слотам [k, ffn].
            let up = backend.zeros(vec![k, ffn])?;
            backend.moe_experts_mv(&x_t, &ids_t, &wts_t, &self.up.w, &up, ffn, false)?;
            let gate = backend.zeros(vec![k, ffn])?;
            backend.moe_experts_mv(&x_t, &ids_t, &wts_t, &self.gate.w, &gate, ffn, false)?;
            // silu(gate)·up поэлементно по всем слотам сразу.
            // swiglu_split оркала — ОДИН кернел (см. moe::ffn_id_mm).
            let prod = backend.zeros(vec![k, ffn])?;
            backend.swiglu(&gate, &up, &prod)?;
            // down: у каждого слота свой вход, выход — взвешенная сумма [dim].
            let out_t = acc.view_offset(vec![1, d], ti * d);
            backend.moe_experts_mv(&prod, &ids_t, &wts_t, &self.down.w, &out_t, d, true)?;
        }
        Ok(acc.view(vec![1, t, d]))
    }

    /// Пакетный mul_mat_id на весь чанк (путь оркала mm_id): три вызова на
    /// слой вместо 3·t, ids [t*k] одним куском, веса применяются в
    /// moe_acc_slots в том же порядке слотов, что по-слотовый add_scaled.
    #[allow(clippy::too_many_arguments)]
    fn ffn_id_mm(
        &self,
        backend: &dyn Backend,
        x2: &Tensor,
        routed: &[(Vec<u32>, Vec<f32>)],
        t: usize,
        d: usize,
        ffn: usize,
        e: usize,
        k: usize,
        device_route: bool,
        device_ids: Option<&Tensor>,
        device_wts: Option<&Tensor>,
    ) -> Result<Tensor, BackendError> {
        // ids/weights одним куском: [t*k], слот быстрее токена (= оркальный
        // ids [ne20=k, ne21=t] с nb21=k).
        let (ids_t, wts_t) = if device_route {
            (
                device_ids.expect("device_ids есть").view(vec![t * k]),
                device_wts.expect("device_wts есть").view(vec![t * k]),
            )
        } else {
            let mut idv = Vec::with_capacity(t * k);
            let mut wv = Vec::with_capacity(t * k);
            for (ids, ws) in routed.iter().take(t) {
                idv.extend_from_slice(ids);
                wv.extend_from_slice(ws);
            }
            (
                backend.from_cpu_u32(&idv, vec![t * k])?,
                backend.from_cpu_f32(&wv, vec![t * k])?,
            )
        };
        // gate/up: src1 = строка токена, ne11=1 (вещание по слотам).
        let up = backend.zeros(vec![t, k, ffn])?;
        backend.moe_experts_mm(
            &x2.view(vec![t, d]),
            &ids_t,
            &self.up.w,
            &up,
            ffn,
            e,
            t,
            k,
            1,
            0,
            (d * 4) as u64,
        )?;
        let gate = backend.zeros(vec![t, k, ffn])?;
        backend.moe_experts_mm(
            &x2.view(vec![t, d]),
            &ids_t,
            &self.gate.w,
            &gate,
            ffn,
            e,
            t,
            k,
            1,
            0,
            (d * 4) as u64,
        )?;
        let prod = backend.zeros(vec![t, k, ffn])?;
        backend.swiglu(&gate, &up, &prod)?;
        // down: src1 [ffn, k, t] (слот-за-слотом, ne11=k), выход [t, k, d].
        let dn = backend.zeros(vec![t, k, d])?;
        backend.moe_experts_mm(
            &prod,
            &ids_t,
            &self.down.w,
            &dn,
            d,
            e,
            t,
            k,
            k,
            (ffn * 4) as u64,
            (ffn * k * 4) as u64,
        )?;
        // INFER_MOE_DUMP=<файл> — побитовые дампы выходов mm_id (первый
        // вызов) для сверки с узлами оркального графа ffn_moe_{up,gate,down}.
        if let Ok(path) = std::env::var("INFER_MOE_DUMP") {
            use std::io::Write as _;
            static CALL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let call = CALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if call < 16
                && let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                {
                    let _ = writeln!(&mut f, "# call {call}");
                    for (name, tt, ne) in [
                        ("up", &up, format!("{ffn},{k},{t},1")),
                        ("gate", &gate, format!("{ffn},{k},{t},1")),
                        ("prod", &prod, format!("{ffn},{k},{t},1")),
                        ("dn", &dn, format!("{d},{k},{t},1")),
                    ] {
                        let v = backend.to_cpu_f32(tt)?;
                        writeln!(f, "STEP {call} TENSOR {name} ne=[{ne}]")?;
                        for (i, x) in v.iter().enumerate() {
                            write!(
                                f,
                                "{:08x}{}",
                                x.to_bits(),
                                if i % 8 == 7 { "\n" } else { " " }
                            )?;
                        }
                        writeln!(f)?;
                    }
                }
        }
        let acc = backend.zeros(vec![t, d])?;
        backend.moe_acc_slots(&dn, &wts_t, &acc, t, k)?;
        if std::env::var("INFER_MOE_DUMP_ACC").is_ok() {
            use std::io::Write as _;
            static ACALL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let call = ACALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if call >= 16 {
                return Ok(acc.view(vec![1, t, d]));
            }
            let v = backend.to_cpu_f32(&acc)?;
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(
                std::env::var("INFER_MOE_DUMP")
                    .unwrap_or_else(|_| "/tmp/llama_ref/olm_acc.txt".into()),
            )?;
            writeln!(f, "STEP {call} TENSOR acc ne=[{d},{t},1,1]")?;
            for (i, x) in v.iter().enumerate() {
                write!(
                    f,
                    "{:08x}{}",
                    x.to_bits(),
                    if i % 8 == 7 { "\n" } else { " " }
                )?;
            }
            writeln!(f)?;
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
        assert_eq!(
            ids,
            vec![0, 1, 2, 3],
            "все четыре равны — порядок по индексу"
        );
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

    /// id-кернелы (mul_mat_id) дают ТЕ ЖЕ числа, что matvec по срезу эксперта:
    /// различие — только порядок суммирования, поэтому допуск — 1e-5·‖x‖.
    /// Проверяются оба режима: up/gate (вход общий, выход по слотам) и down
    /// (вход по слотам, взвешенная сумма).
    #[cfg(feature = "metal")]
    #[test]
    fn id_kernel_matches_per_expert_matvec() {
        use crate::backend::Backend as _;
        use crate::dtype::WeightFormat;
        let be = crate::metal::MetalBackend::new().unwrap();
        // Размеры кратны блоку Q4_K (256) — как у реальных моделей
        // (OLMoE: k=2048, ffn=1024; здесь уменьшено для скорости теста).
        let (e, k, n, ffn) = (8usize, 512usize, 256usize, 256usize);
        // Валидные Q4_K-суперблоки: d=1, dmin=0.5, масштабы 1..8, нибблы случайны.
        let mut seed = 0x1234_5678u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut q4k = |rows: usize, cols: usize| -> Vec<u8> {
            let mut out = Vec::with_capacity(rows * cols / 256 * 144);
            for _ in 0..rows * cols / 256 {
                out.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
                out.extend_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
                for i in 0..12 {
                    out.push(((next() >> (i % 8)) & 0xFF) as u8);
                }
                for _ in 0..128 {
                    out.push((next() & 0xFF) as u8);
                }
            }
            out
        };
        let up_w = be
            .from_quantized(&q4k(e * ffn, k), vec![e * ffn, k], WeightFormat::Q4_K)
            .unwrap();
        let down_w = be
            .from_quantized(&q4k(e * n, ffn), vec![e * n, ffn], WeightFormat::Q4_K)
            .unwrap();
        let ids: Vec<u32> = (0..4).map(|j| ((j * 3) % e) as u32).collect();
        let ws: Vec<f32> = vec![0.4, 0.3, 0.2, 0.1];
        let x = be
            .from_cpu_f32(
                &(0..k)
                    .map(|i| ((i % 13) as f32 - 6.0) * 0.05)
                    .collect::<Vec<_>>(),
                vec![1, k],
            )
            .unwrap();

        // up/gate: выход по слотам [4, ffn]
        let y_id = be.zeros(vec![4, ffn]).unwrap();
        let ids_t = be.from_cpu_u32(&ids, vec![ids.len()]).unwrap();
        let ws_t = be.from_cpu_f32(&ws, vec![ws.len()]).unwrap();
        be.moe_experts_mv(&x, &ids_t, &ws_t, &up_w, &y_id, ffn, false)
            .unwrap();
        let id_host = be.to_cpu_f32(&y_id).unwrap();
        for (j, &eid) in ids.iter().enumerate() {
            let slice = ExpertTensor {
                w: up_w.clone(),
                rows: ffn,
                cols: k,
                row_bytes: Some(144 * k / 256),
            }
            .expert(eid as usize);
            let y = be.zeros(vec![1, ffn]).unwrap();
            be.gemm_nt(&x, &slice, &y).unwrap();
            let host = be.to_cpu_f32(&y).unwrap();
            for i in 0..ffn {
                let got = id_host[j * ffn + i];
                let want = host[i];
                // mv_id и по-слотовый matvec считают одну и ту же r2-раскладку
                // (хелперы mv_q4k_* в common.metal) — ПОБИТОВОЕ равенство и
                // есть контракт (эталонный mv_id оборачивает ту же impl).
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "слот {j} (эксперт {eid}) [{i}]: id {got} против среза {want}"
                );
            }
        }

        // down: вход по слотам [4, ffn] → взвешенная сумма [n]
        let prod = be
            .from_cpu_f32(
                &(0..4 * ffn)
                    .map(|i| ((i % 11) as f32 - 5.0) * 0.03)
                    .collect::<Vec<_>>(),
                vec![4, ffn],
            )
            .unwrap();
        let y_ids = be.zeros(vec![n]).unwrap();
        be.moe_experts_mv(&prod, &ids_t, &ws_t, &down_w, &y_ids, n, true)
            .unwrap();
        let sum_id = be.to_cpu_f32(&y_ids).unwrap();
        let mut want = vec![0f32; n];
        for (j, &eid) in ids.iter().enumerate() {
            let slice = ExpertTensor {
                w: down_w.clone(),
                rows: n,
                cols: ffn,
                row_bytes: Some(144 * ffn / 256),
            }
            .expert(eid as usize);
            let row = prod.view_offset(vec![1, ffn], j * ffn);
            let y = be.zeros(vec![1, n]).unwrap();
            be.gemm_nt(&row, &slice, &y).unwrap();
            let host = be.to_cpu_f32(&y).unwrap();
            for i in 0..n {
                want[i] += ws[j] * host[i];
            }
        }
        for i in 0..n {
            // Побитово и для down: кернел суммирует слоты БЕЗ контракции
            // (#pragma clang fp contract(off): p = w·v округляется, затем
            // acc + p) — ровно как host-эталон и как эталонный граф llama.cpp
            // (ggml_mul затем ggml_add — два отдельных кернела).
            assert_eq!(
                sum_id[i].to_bits(),
                want[i].to_bits(),
                "down [{i}]: id {} против суммы срезов {}",
                sum_id[i],
                want[i]
            );
        }
    }

    /// Роутер на устройстве (moe_route) даёт те же id и веса, что host-путь:
    /// softmax, top-k (ничьи — меньший индекс), нормировка, масштаб.
    #[cfg(feature = "metal")]
    #[test]
    fn device_router_matches_host() {
        use crate::backend::Backend as _;
        let be = crate::metal::MetalBackend::new().unwrap();
        let (t, e, k) = (3usize, 128usize, 4usize);
        // Логиты с ровными значениями (ничьи) и без них.
        let mut seed = 0xDEAD_BEEFu64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed >> 11) as f32 / (1u64 << 53) as f32 - 0.5) * 6.0
        };
        let mut logits = Vec::with_capacity(t * e);
        for ti in 0..t {
            for i in 0..e {
                logits.push(if ti == 1 && i < 8 { 3.0 } else { next() });
            }
        }
        let lg = be.from_cpu_f32(&logits, vec![t, e]).unwrap();
        for norm in [false, true] {
            let c = MoeConfig {
                n_expert: e,
                n_expert_used: k,
                ffn_dim: 1,
                gating: Gating::Softmax,
                norm_topk_prob: norm,
                expert_weights_scale: 1.5,
                n_expert_groups: 0,
                n_group_used: 0,
            };
            let ids_t = be.zeros(vec![t * k]).unwrap();
            let ws_t = be.zeros(vec![t * k]).unwrap();
            be.moe_route(&lg, &ids_t, &ws_t, k, norm, c.expert_weights_scale)
                .unwrap();
            let got_ids = be.to_cpu_u32(&ids_t).unwrap();
            let got_ws = be.to_cpu_f32(&ws_t).unwrap();
            let mut want_ids = Vec::new();
            let mut want_ws = Vec::new();
            for ti in 0..t {
                let (mut i, mut w) = (Vec::new(), Vec::new());
                c.route_row(&logits[ti * e..(ti + 1) * e], &mut i, &mut w);
                want_ids.push(i);
                want_ws.push(w);
            }
            if std::env::var("DBG_ROUTE").is_ok() {
                eprintln!("DBG norm={norm} device ids={got_ids:?} host ids={want_ids:?}");
                eprintln!("DBG device ws={got_ws:?}");
                eprintln!("DBG host ws={want_ws:?}");
            }
            for ti in 0..t {
                for j in 0..k {
                    assert_eq!(
                        got_ids[ti * k + j],
                        want_ids[ti][j],
                        "токен {ti} слот {j}: id (norm={norm})"
                    );
                    let (g, w) = (got_ws[ti * k + j], want_ws[ti][j]);
                    assert!(
                        (g - w).abs() <= 1e-6 * (1.0 + w.abs()),
                        "токен {ti} слот {j}: вес {g} против {w} (norm={norm})"
                    );
                }
            }
        }
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
