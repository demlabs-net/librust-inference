//! Hyper-connections (`qwen4exp` / Qwen3.8-Flash-Next) — шаг Э2.2.
//!
//! Порт `build_hc_mix` / `build_hc_combine` из llama.cpp
//! (`src/models/qwen4exp.cpp`, эталон 5266f24da). У Flash-Next нет
//! LayerNorm между блоками: остаток — это **hc параллельных потоков**
//! `[T, hc, dim]`, а «норма» блока — обучаемый миксер, который
//!  1) нормирует каждый поток (RMSNorm по dim) и масштабирует своей гаммой
//!     (`gamma[c]`, конвертер GGUF сложил её в `1 + w`);
//!  2) low-rank гейтом `down → silu(·/hc) → up` считает sigmoid-гейт на
//!     каждый (поток, канал);
//!  3) сворачивает потоки в один `mixed [T, dim]` их СРЕДНИМ;
//!  4) отдельно отдаёт `inject [T, hc]` — веса, с которыми выход блока
//!     разложится обратно по потокам.
//!
//! Обратное разложение (`combine`): каждый поток получает свой
//! `2·sigmoid(inject/hc)` — центр в 1, поэтому нулевой inject = обычный
//! residual-add. Финальный миксер (`hc_head`) заменяет `output_norm`.
//!
//! Раскладка: канонически остаток живёт как `[T, hc·dim]` (потоки внутри
//! токена подряд — так `W_down`/`W_up`/`W_inject` применяются ОДНИМ gemm на
//! весь батч). Для RMSNorm по потоку тот же буфер читается как `[T·hc, dim]`.
//!
//! Что не поддержано: full-rank вариант (у DeepSeek-V4 `low_rank`
//! отсутствует). У Flash-Next low-rank есть всегда; при его отсутствии
//! модуль честно отказывает, а не подставляет догадку.

use crate::backend::{Backend, Tensor};
use crate::error::BackendError;
use crate::weights_map::WeightsMap;

/// Один HC-миксер: перед вниманием, перед MoE или финальный (head).
#[derive(Debug, Clone)]
pub struct HyperConn {
    pub hc: usize,
    pub dim: usize,
    pub lr: usize,
    eps: f32,
    /// Гамма по потокам: `[hc·dim]`, плоский вид `[hc, dim]`.
    norm: Tensor,
    /// `[lr, hc·dim]`.
    down: Tensor,
    /// `[hc·dim, lr]`.
    up: Tensor,
    /// `[hc, hc·dim]`; у финального миксера отсутствует.
    inject: Option<Tensor>,
    /// Единицы для RMSNorm (нормировка без своей гаммы: гамма применяется
    /// следующим `mul_vec` по потокам).
    ones: Tensor,
}

impl HyperConn {
    /// Загрузить миксер `prefix` (`blk.3.hc_attn_`, `blk.3.hc_ffn_` или
    /// `hc_head_`). `dim` — ширина одного потока, `hc` — число потоков.
    pub fn load(
        backend: &dyn Backend,
        wm: &WeightsMap<'_>,
        prefix: &str,
        hc: usize,
        dim: usize,
        eps: f32,
    ) -> Result<Self, BackendError> {
        if hc < 2 {
            return Err(BackendError::UnsupportedFormat(format!(
                "{prefix}: hyper-connections требуют hc > 1, а в модели {hc}"
            )));
        }
        let hc_dim = hc * dim;
        let t = |s: &str| format!("{prefix}{s}");
        let get = |name: &str| {
            wm.get(name)
                .ok_or_else(|| BackendError::Shape(format!("missing tensor {name}")))
        };
        let check = |name: &str, wt: &crate::weights_map::WeightTensor<'_>, want: [usize; 2]| {
            let want64 = [want[0] as u64, want[1] as u64];
            if wt.shape.as_slice() != want64 {
                return Err(BackendError::Shape(format!(
                    "{name}: ожидался {:?} (GGUF-порядок), в файле {:?}",
                    want, wt.shape
                )));
            }
            Ok(())
        };

        // gamma: в файле [dim, hc] → память [hc][dim] (ne0 быстрее) — ровно
        // плоский вид «поток за потоком», которого ждёт mul_vec по [T, hc·dim].
        let norm_wt = get(&t("norm.weight"))?;
        check(&t("norm.weight"), norm_wt, [dim, hc])?;

        let down_wt = get(&t("down.weight"))?;
        // В файле [hc·dim, lr] → ранг = вторая размерность.
        let lr = down_wt.shape[1] as usize;
        check(&t("down.weight"), down_wt, [hc_dim, lr])?;
        let up_wt = get(&t("up.weight"))?;
        check(&t("up.weight"), up_wt, [lr, hc_dim])?;

        let norm = backend.from_cpu_f32(
            &crate::weights_map::dequant_f32(norm_wt),
            vec![hc_dim],
        )?;
        let down = crate::qwen3::Qwen3Model::upload_q(backend, wm, &t("down.weight"), vec![lr, hc_dim])?;
        let up = crate::qwen3::Qwen3Model::upload_q(backend, wm, &t("up.weight"), vec![hc_dim, lr])?;

        let inject = match wm.get(&t("inject.weight")) {
            Some(wt) => {
                check(&t("inject.weight"), wt, [hc_dim, hc])?;
                Some(crate::qwen3::Qwen3Model::upload_q(
                    backend,
                    wm,
                    &t("inject.weight"),
                    vec![hc, hc_dim],
                )?)
            }
            None => None,
        };

        let ones = backend.from_cpu_f32(&vec![1.0f32; dim], vec![dim])?;
        Ok(Self {
            hc,
            dim,
            lr,
            eps,
            norm,
            down,
            up,
            inject,
            ones,
        })
    }

    /// Миксер блока: `res [T, hc·dim]` → (`mixed [T, dim]`, `inject [T, hc]`).
    /// `inject` — None только у head-миксера.
    pub fn mix(
        &self,
        backend: &dyn Backend,
        res: &Tensor,
        t: usize,
    ) -> Result<(Tensor, Option<Tensor>), BackendError> {
        let (hc, dim, hc_dim) = (self.hc, self.dim, self.hc * self.dim);

        // 1. RMSNorm каждого потока: тот же буфер, но как [T·hc, dim].
        let rows = backend.zeros(vec![t * hc, dim])?;
        backend.rmsnorm(&res.view(vec![t * hc, dim]), &self.ones, &rows, self.eps)?;
        // 2. Гамма по потокам (плоский [hc·dim]).
        let xn = backend.zeros(vec![t, hc_dim])?;
        backend.mul_vec(&rows.view(vec![t, hc_dim]), &self.norm, &xn)?;

        // 3. Low-rank гейт: lo = silu(W_down·xn / hc), gate = W_up·lo.
        let lo = backend.zeros(vec![t, self.lr])?;
        backend.gemm_nt(&xn, &self.down, &lo)?;
        let lo_s = backend.zeros(vec![t, self.lr])?;
        backend.scale(&lo, &lo_s, 1.0 / hc as f32)?;
        let act = backend.zeros(vec![t, self.lr])?;
        backend.silu(&lo_s, &act)?;
        let gate = backend.zeros(vec![t, hc_dim])?;
        backend.gemm_nt(&act, &self.up, &gate)?;

        // 4. gated = xn · sigmoid(gate); mixed = среднее по потокам.
        let sg = backend.zeros(vec![t, hc_dim])?;
        backend.sigmoid(&gate, &sg)?;
        let gated = backend.zeros(vec![t, hc_dim])?;
        backend.mul(&xn, &sg, &gated)?;
        let summed = self.sum_streams(backend, &gated, t)?;
        let mixed = backend.zeros(vec![t, dim])?;
        backend.scale(&summed, &mixed, 1.0 / hc as f32)?;

        // 5. inject = W_inject · xn — веса раскладки выхода блока по потокам.
        let inject = match &self.inject {
            Some(w) => {
                let y = backend.zeros(vec![t, hc])?;
                backend.gemm_nt(&xn, w, &y)?;
                Some(y)
            }
            None => None,
        };
        Ok((mixed, inject))
    }

    /// Разложить выход блока по потокам: `res' = res + block ⊗ 2·sigmoid(inject/hc)`.
    pub fn combine(
        &self,
        backend: &dyn Backend,
        res: &Tensor,
        block: &Tensor,
        inject: &Tensor,
        t: usize,
    ) -> Result<Tensor, BackendError> {
        let hc = self.hc;
        let half = backend.zeros(vec![t, hc])?;
        backend.scale(inject, &half, 1.0 / hc as f32)?;
        let sg = backend.zeros(vec![t, hc])?;
        backend.sigmoid(&half, &sg)?;
        let w = backend.zeros(vec![t, hc])?;
        backend.scale(&sg, &w, 2.0)?;

        let delta = self.scatter(backend, block, &w, t)?;
        let out = backend.zeros(vec![t, hc * self.dim])?;
        backend.add(res, &delta, &out)?;
        Ok(out)
    }

    /// `Σ` по потокам; ядро бэкенда, иначе честный host-фолбэк (пока
    /// fused-HC-ядра не написаны — Э3.3).
    fn sum_streams(&self, backend: &dyn Backend, gated: &Tensor, t: usize) -> Result<Tensor, BackendError> {
        let (hc, dim) = (self.hc, self.dim);
        let y = backend.zeros(vec![t, dim])?;
        match backend.sum_rows_grouped(&gated.view(vec![t * hc, dim]), &y, hc) {
            Ok(()) => Ok(y),
            Err(BackendError::UnsupportedFormat(_)) => {
                let src = backend.to_cpu_f32(&gated.view(vec![t * hc, dim]))?;
                let mut out = vec![0f32; t * dim];
                for ti in 0..t {
                    for c in 0..hc {
                        for k in 0..dim {
                            out[ti * dim + k] += src[(ti * hc + c) * dim + k];
                        }
                    }
                }
                backend.from_cpu_f32(&out, vec![t, dim])
            }
            Err(e) => Err(e),
        }
    }

    /// `block [T, dim] ⊗ w [T, hc] → [T, hc·dim]`; host-фолбэк, если ядра нет.
    fn scatter(
        &self,
        backend: &dyn Backend,
        block: &Tensor,
        w: &Tensor,
        t: usize,
    ) -> Result<Tensor, BackendError> {
        let (hc, dim) = (self.hc, self.dim);
        let y = backend.zeros(vec![t, hc * dim])?;
        match backend.hc_scatter(block, w, &y) {
            Ok(()) => Ok(y),
            Err(BackendError::UnsupportedFormat(_)) => {
                let b = backend.to_cpu_f32(block)?;
                let s = backend.to_cpu_f32(w)?;
                let mut out = vec![0f32; t * hc * dim];
                for ti in 0..t {
                    for c in 0..hc {
                        let scale = s[ti * hc + c];
                        for k in 0..dim {
                            out[(ti * hc + c) * dim + k] = b[ti * dim + k] * scale;
                        }
                    }
                }
                backend.from_cpu_f32(&out, vec![t, hc * dim])
            }
            Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Параметры tiny-конфига: hc=2, dim=2, low-rank=2.
    fn tiny(be: &dyn Backend) -> (HyperConn, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let (hc, dim, lr) = (2usize, 2usize, 2usize);
        let hc_dim = hc * dim;
        // Детерминированные значения.
        let norm: Vec<f32> = (0..hc_dim).map(|i| 1.0 + 0.1 * i as f32).collect();
        let down: Vec<f32> = (0..lr * hc_dim).map(|i| 0.05 * (i as f32 + 1.0)).collect();
        let up: Vec<f32> = (0..hc_dim * lr).map(|i| -0.07 * (i as f32 + 1.0)).collect();
        let inj: Vec<f32> = (0..hc * hc_dim).map(|i| 0.03 * (i as f32) - 0.1).collect();
        let hc_struct = HyperConn {
            hc,
            dim,
            lr,
            eps: 1e-6,
            norm: be.from_cpu_f32(&norm, vec![hc_dim]).unwrap(),
            down: be.from_cpu_f32(&down, vec![lr, hc_dim]).unwrap(),
            up: be.from_cpu_f32(&up, vec![hc_dim, lr]).unwrap(),
            inject: Some(be.from_cpu_f32(&inj, vec![hc, hc_dim]).unwrap()),
            ones: be.from_cpu_f32(&vec![1.0; dim], vec![dim]).unwrap(),
        };
        (hc_struct, norm, down, up, inj)
    }

    /// mix/combine против ручного счёта по формулам llama.cpp.
    #[test]
    fn mix_and_combine_match_host_math() {
        use crate::backend::Backend as _;
        let be = crate::cpu::CpuBackend;
        let (hcmod, norm, down, up, inj) = tiny(&be);
        let (hc, dim, lr) = (2usize, 2usize, 2usize);
        let t = 2usize;
        let x: Vec<f32> = vec![
            0.5, -1.0, 0.25, 0.75, // токен 0: потоки 0,1
            -0.5, 0.5, 1.0, -0.25, // токен 1
        ];
        let xt = be.from_cpu_f32(&x, vec![t, hc * dim]).unwrap();
        let (mixed, inject) = hcmod.mix(&be, &xt, t).unwrap();
        let inject = inject.expect("inject есть");
        let got_mixed = be.to_cpu_f32(&mixed).unwrap();
        let got_inject = be.to_cpu_f32(&inject).unwrap();

        let mut want_mixed = vec![0f32; t * dim];
        let mut want_inject = vec![0f32; t * hc];
        for ti in 0..t {
            // xn[t, c, k] = gamma[c,k] · x / rms(x_c)
            let mut xn = vec![0f32; hc * dim];
            for c in 0..hc {
                let base = ti * hc * dim + c * dim;
                let ms: f32 = (0..dim).map(|k| x[base + k] * x[base + k]).sum::<f32>() / dim as f32;
                let inv = 1.0 / (ms + 1e-6).sqrt();
                for k in 0..dim {
                    xn[c * dim + k] = x[base + k] * inv * norm[c * dim + k];
                }
            }
            // lo / gate / gated
            let mut lo = vec![0f32; lr];
            for j in 0..lr {
                lo[j] = (0..hc * dim).map(|i| down[j * hc * dim + i] * xn[i]).sum();
                let s = lo[j] / hc as f32;
                lo[j] = s / (1.0 + (-s).exp());
            }
            let mut gated = vec![0f32; hc * dim];
            for c in 0..hc {
                for k in 0..dim {
                    let gate: f32 = (0..lr).map(|j| up[(c * dim + k) * lr + j] * lo[j]).sum();
                    gated[c * dim + k] = xn[c * dim + k] / (1.0 + (-gate).exp());
                }
            }
            for k in 0..dim {
                want_mixed[ti * dim + k] =
                    (0..hc).map(|c| gated[c * dim + k]).sum::<f32>() / hc as f32;
            }
            for c in 0..hc {
                want_inject[ti * hc + c] = (0..hc * dim)
                    .map(|i| inj[c * hc * dim + i] * xn[i])
                    .sum();
            }
        }
        for i in 0..t * dim {
            assert!(
                (got_mixed[i] - want_mixed[i]).abs() <= 1e-5 * (1.0 + want_mixed[i].abs()),
                "mixed[{i}]: {} против {}",
                got_mixed[i],
                want_mixed[i]
            );
        }
        for i in 0..t * hc {
            assert!(
                (got_inject[i] - want_inject[i]).abs() <= 1e-5 * (1.0 + want_inject[i].abs()),
                "inject[{i}]: {} против {}",
                got_inject[i],
                want_inject[i]
            );
        }

        // combine: res + block ⊗ 2·sigmoid(inject/hc)
        let block = vec![1.0f32, -2.0, 0.5, 3.0];
        let bt = be.from_cpu_f32(&block, vec![t, dim]).unwrap();
        let out = hcmod.combine(&be, &xt, &bt, &inject, t).unwrap();
        let got = be.to_cpu_f32(&out).unwrap();
        for ti in 0..t {
            for c in 0..hc {
                let wgt = 2.0 / (1.0 + (-(want_inject[ti * hc + c] / hc as f32)).exp());
                for k in 0..dim {
                    let want = x[(ti * hc + c) * dim + k] + block[ti * dim + k] * wgt;
                    let i = (ti * hc + c) * dim + k;
                    assert!(
                        (got[i] - want).abs() <= 1e-5 * (1.0 + want.abs()),
                        "combine[{i}]: {} против {want}",
                        got[i]
                    );
                }
            }
        }
    }

    /// Head-миксер без inject: mix работает, combine недоступен.
    #[test]
    fn head_mixer_has_no_inject() {
        use crate::backend::Backend as _;
        let be = crate::cpu::CpuBackend;
        let (mut m, ..) = tiny(&be);
        m.inject = None;
        let xt = be.from_cpu_f32(&[1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let (_mixed, inject) = m.mix(&be, &xt, 1).unwrap();
        assert!(inject.is_none(), "у head-миксера inject отсутствует");
    }
}
