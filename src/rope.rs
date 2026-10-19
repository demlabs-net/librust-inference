//! YaRN/линейное скалирование RoPE — зеркало ggml `rope_yarn`/
//! `ggml_rope_cache_init` (ggml-cpu/ops.cpp) и вывода параметров из
//! `llama-context.cpp` эталонной сборки.
//!
//! Зачем: модели с `rope.scaling.type=yarn` (например, Qwen3-8B-abliterated
//! IQ4_XS: factor=4, orig_ctx=32768) в эталоне поворачивают q/k на
//! СКОРРЕКТИРОВАННЫЕ частоты и масштабируют cos/sin на mscale. Без этого
//! логиты расходятся с эталоном тем сильнее, чем больше позиций участвует
//! в attention (на позиции 0 поворот — тождество, поэтому одно-токенный
//! промпт паритетен, а с T≥2 расхождение растёт).
//!
//! Формула на пару k (0-индексация пар) при позиции p:
//!   theta   = p · theta_scale^k            (мультипликативное накопление,
//!                                           порядок перемножений как в ggml)
//!   interp  = freq_scale · theta
//!   ramp    = 1 − clamp((k − corr_lo)/(corr_hi − corr_lo), 0, 1)
//!   ang     = interp·(1 − ramp·ext) + theta·(ramp·ext)
//!   ms      = attn_factor · (1 + 0.1·ln(1/freq_scale))   [только ext≠0]
//!   cos = cosf(ang)·ms;  sin = sinf(ang)·ms
//!
//! `ext_factor == 0 && freq_scale == 1 && attn_factor == 1` — обычный RoPE,
//! кернелы в этом случае идут прежним путём (численно неотличимым от
//! эталона для нескалированных моделей).

/// Параметры коррекции, выводимые из метаданных GGUF (см. `from_meta`).
/// Передаются в кернелы одним `repr(C)`-структурным аргументом.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeYarn {
    /// 1/factor (ggml `freq_scale`); 1.0 — без сжатия частот.
    pub freq_scale: f32,
    /// 1.0 для yarn, 0.0 для linear/none (ggml `ext_factor`).
    pub ext_factor: f32,
    /// ggml `attn_factor` (op-параметр) — уже с выводом llama-context.cpp.
    pub attn_factor: f32,
    /// corr_dims[0] = max(0, floor(corr_dim(beta_fast))).
    pub corr_lo: f32,
    /// corr_dims[1] = min(n_dims−1, ceil(corr_dim(beta_slow))).
    pub corr_hi: f32,
    /// powf(freq_base, −2/n_dims) — множитель theta на пару.
    pub theta_scale: f32,
}

impl RopeYarn {
    /// Тождественные параметры: кернелы идут старым путём.
    pub const IDENTITY: RopeYarn = RopeYarn {
        freq_scale: 1.0,
        ext_factor: 0.0,
        attn_factor: 1.0,
        corr_lo: 0.0,
        corr_hi: 0.0,
        theta_scale: 0.0,
    };

    pub fn is_identity(&self) -> bool {
        self.ext_factor == 0.0 && self.freq_scale == 1.0 && self.attn_factor == 1.0
    }

    /// Вывод из метаданных — зеркало `llama_model::load_hparams` +
    /// `llama_context` (коммит 5266f24da75d). Ключи — с префиксом архитектуры
    /// (`qwen3.rope.scaling.type` …). Возвращает `None`, когда скалирования
    /// нет (тип none/отсутствует и factor не задан) — старый путь.
    ///
    /// ВАЖНО (паритет с этой сборкой эталона): `rope.scaling.yarn_attn_factor`
    /// эталоном НЕ читается — только `rope.scaling.attn_factor`
    /// (`hparams.rope_attn_factor`, по умолчанию 1.0).
    pub fn from_meta(
        get_f32: &dyn Fn(&str) -> Option<f32>,
        get_str: &dyn Fn(&str) -> Option<String>,
        get_u32: &dyn Fn(&str) -> Option<u32>,
        prefix: &str,
        freq_base: f32,
        n_dims: usize,
        n_ctx_train: u32,
    ) -> Option<RopeYarn> {
        let key = |k: &str| format!("{prefix}.rope.scaling.{k}");
        let scaling = get_str(&key("type")).unwrap_or_else(|| "none".into());
        let factor = get_f32(&key("factor")).unwrap_or(0.0);
        // rope_freq_scale_train: 1/factor (0 → 1.0), как в load_hparams
        let freq_scale = if factor == 0.0 { 1.0 } else { 1.0 / factor };
        let ext_factor = match scaling.as_str() {
            "yarn" => 1.0f32,
            _ => 0.0f32,
        };
        let freq_scale = if scaling == "none" { 1.0 } else { freq_scale };
        if ext_factor == 0.0 && freq_scale == 1.0 {
            return None; // ни линейного сжатия, ни yarn — старый путь
        }
        let meta_attn = get_f32(&key("attn_factor")).unwrap_or(1.0);
        // llama-context.cpp: при ext≠0 attn_factor = get_mscale(factor,1) ·
        // 1/(1+0.1·ln factor) · hparams.rope_attn_factor. Порядок операций —
        // тот же, что у эталона (f32).
        let mut attn_factor = meta_attn;
        if ext_factor != 0.0 {
            let f = if factor == 0.0 { 1.0 } else { factor };
            let mscale = if f <= 1.0 { 1.0 } else { 0.1 * f.ln() + 1.0 }; // get_mscale(f, 1.0)
            attn_factor = mscale * (1.0 / (1.0 + 0.1 * f.ln()));
            attn_factor *= meta_attn;
        }
        let n_ctx_orig = get_u32(&key("original_context_length")).unwrap_or(n_ctx_train);
        let beta_fast = get_f32(&key("yarn_beta_fast")).unwrap_or(32.0);
        let beta_slow = get_f32(&key("yarn_beta_slow")).unwrap_or(1.0);
        let (corr_lo, corr_hi) = Self::corr_dims(
            n_dims as f32,
            n_ctx_orig as f32,
            freq_base,
            beta_fast,
            beta_slow,
        );
        let theta_scale = freq_base.powf(-2.0 / n_dims as f32);
        Some(RopeYarn {
            freq_scale,
            ext_factor,
            attn_factor,
            corr_lo,
            corr_hi,
            theta_scale,
        })
    }

    /// ggml_rope_yarn_corr_dims: [max(0,floor(c(beta_fast))), min(n−1,ceil(c(beta_slow)))].
    fn corr_dims(
        n_dims: f32,
        n_ctx_orig: f32,
        base: f32,
        beta_fast: f32,
        beta_slow: f32,
    ) -> (f32, f32) {
        let c = |n_rot: f32| {
            n_dims * (n_ctx_orig / (n_rot * 2.0 * std::f32::consts::PI)).ln() / (2.0 * base.ln())
        };
        let start = c(beta_fast).floor();
        let end = c(beta_slow).ceil();
        (0.0f32.max(start), (n_dims - 1.0).min(end))
    }

    /// cos/sin для пары `k` на позиции `p` — CPU-зеркало `rope_yarn`
    /// (тот же порядок f32-операций, включая накопление theta).
    pub fn cos_sin(&self, p: f32, k: usize) -> (f32, f32) {
        let mut theta = p;
        for _ in 0..k {
            theta *= self.theta_scale;
        }
        let theta_interp = self.freq_scale * theta;
        let mut ang = theta_interp;
        let mut ms = self.attn_factor;
        if self.ext_factor != 0.0 {
            // rope_yarn_ramp(low, high, i0): y = (i0/2 − low)/max(0.001, high−low),
            // ramp = 1 − MIN(1, MAX(0, y)); ramp_mix = ramp·ext_factor
            let kf = k as f32;
            let y = (kf - self.corr_lo) / (self.corr_hi - self.corr_lo).max(0.001);
            let ramp_mix = (1.0 - 1.0f32.min(0.0f32.max(y))) * self.ext_factor;
            ang = theta_interp * (1.0 - ramp_mix) + theta * ramp_mix;
            ms *= 1.0 + 0.1 * (1.0 / self.freq_scale).ln();
        }
        (ang.cos() * ms, ang.sin() * ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Параметры Qwen3-8B-abliterated-IQ4_XS: yarn factor=4, orig=32768,
    /// base=1e6, n_dims=128, attn_factor по умолчанию (1.0).
    fn abliterated() -> RopeYarn {
        let get_f32: &dyn Fn(&str) -> Option<f32> = &|k| {
            if k.ends_with("rope.scaling.factor") {
                Some(4.0)
            } else {
                None
            }
        };
        let get_str: &dyn Fn(&str) -> Option<String> = &|k| {
            if k.ends_with("rope.scaling.type") {
                Some("yarn".into())
            } else {
                None
            }
        };
        let get_u32: &dyn Fn(&str) -> Option<u32> = &|k| {
            if k.ends_with("original_context_length") {
                Some(32768)
            } else {
                None
            }
        };
        RopeYarn::from_meta(get_f32, get_str, get_u32, "qwen3", 1e6, 128, 131072).unwrap()
    }

    #[test]
    fn yarn_derivation_matches_reference() {
        let y = abliterated();
        assert_eq!(y.freq_scale, 0.25);
        assert_eq!(y.ext_factor, 1.0);
        // get_mscale(4,1)·1/(1+0.1·ln4)·1.0 == 1.0 (с точностью f32)
        assert!((y.attn_factor - 1.0).abs() < 1e-6, "{}", y.attn_factor);
        // corr_dims: 128·ln(32768/(32·2π))/(2·ln 1e6) = 23.59 → 23;
        //            128·ln(32768/(1·2π))/(2·ln 1e6)  = 39.66 → 40
        assert_eq!(y.corr_lo, 23.0);
        assert_eq!(y.corr_hi, 40.0);
        // theta_scale = 1e6^(−2/128)
        assert!((y.theta_scale - 1e6f32.powf(-2.0 / 128.0)).abs() < 1e-9);
    }

    #[test]
    fn position_zero_is_identity_rotation() {
        let y = abliterated();
        let (c, s) = y.cos_sin(0.0, 5);
        assert_eq!((c, s), (y.attn_factor * (1.0 + 0.1f32 * 4.0f32.ln()), 0.0));
        // mscale при p=0: cos(0)·ms = ms, sin(0)·ms = 0
    }

    #[test]
    fn no_scaling_returns_none() {
        let get_f32: &dyn Fn(&str) -> Option<f32> = &|_| None;
        let get_str: &dyn Fn(&str) -> Option<String> = &|_| None;
        let get_u32: &dyn Fn(&str) -> Option<u32> = &|_| None;
        assert!(
            RopeYarn::from_meta(get_f32, get_str, get_u32, "qwen3", 1e6, 128, 131072).is_none()
        );
    }

    #[test]
    fn linear_scaling_has_no_ramp() {
        let get_f32: &dyn Fn(&str) -> Option<f32> = &|k| {
            if k.ends_with("rope.scaling.factor") {
                Some(2.0)
            } else {
                None
            }
        };
        let get_str: &dyn Fn(&str) -> Option<String> = &|k| {
            if k.ends_with("rope.scaling.type") {
                Some("linear".into())
            } else {
                None
            }
        };
        let get_u32: &dyn Fn(&str) -> Option<u32> = &|_| None;
        let y = RopeYarn::from_meta(get_f32, get_str, get_u32, "qwen3", 1e6, 128, 131072).unwrap();
        assert_eq!(y.freq_scale, 0.5);
        assert_eq!(y.ext_factor, 0.0);
        // ext=0 → mscale не домножается, угол = freq_scale·theta
        let (c, s) = y.cos_sin(1.0, 0);
        assert_eq!(c, 0.5f32.cos());
        assert_eq!(s, 0.5f32.sin());
    }
}
