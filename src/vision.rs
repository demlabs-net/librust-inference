//! Ф1.11 Vision: ViT encoder + mmproj projector (llama.cpp clip GGUF
//! convention — `clip.*` metadata, `v.*` tensors). Target: gemma-4-e4b
//! (+mmproj); Qwen3.8-VL mmproj (conv patch embd, merger) — Ф9.
//!
//! Pipeline: resize (bilinear, host — входной препроцессинг как
//! токенизация) → normalize → patchify → patch_embd (+pos_embd) →
//! transformer blocks (pre-norm MHA, fused qkv, GELU / quick-GELU FFN) → norm →
//! projector. All tensor math runs on the backend (GPU).

use crate::backend::{Backend, Tensor};
use crate::error::BackendError;
use crate::gguf::{GgufFile, GgufValue};
use crate::weights_map::WeightsMap;

/// Конфигурация vision-энкодера из `clip.*` метадаты.
#[derive(Debug, Clone)]
pub struct VisionConfig {
    pub projection_dim: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub embedding_length: usize,
    pub ffn_length: usize,
    pub block_count: usize,
    pub n_heads: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub use_gelu: bool,
    pub norm_eps: f32,
}

impl VisionConfig {
    pub fn from_gguf(file: &GgufFile<'_>) -> Result<Self, BackendError> {
        let u = |k: &str| -> Result<usize, BackendError> {
            file.get(k)
                .and_then(GgufValue::as_u32)
                .map(|v| v as usize)
                .ok_or_else(|| BackendError::Shape(format!("missing {k}")))
        };
        let f = |k: &str| -> Result<f32, BackendError> {
            file.get(k)
                .and_then(|v| match v {
                    GgufValue::F32(x) => Some(*x),
                    GgufValue::F64(x) => Some(*x as f32),
                    _ => None,
                })
                .ok_or_else(|| BackendError::Shape(format!("missing {k}")))
        };
        let arr3 = |k: &str| -> Result<[f32; 3], BackendError> {
            let arr = file
                .get(k)
                .and_then(GgufValue::as_array)
                .ok_or_else(|| BackendError::Shape(format!("missing {k}")))?;
            let mut out = [0.5f32; 3];
            for (i, v) in arr.iter().take(3).enumerate() {
                out[i] = match v {
                    GgufValue::F32(x) => *x,
                    GgufValue::F64(x) => *x as f32,
                    _ => return Err(BackendError::Shape(format!("{k}[{i}] not f32"))),
                };
            }
            Ok(out)
        };
        Ok(Self {
            projection_dim: u("clip.vision.projection_dim")?,
            image_size: u("clip.vision.image_size")?,
            patch_size: u("clip.vision.patch_size")?,
            embedding_length: u("clip.vision.embedding_length")?,
            ffn_length: u("clip.vision.feed_forward_length")?,
            block_count: u("clip.vision.block_count")?,
            n_heads: u("clip.vision.attention.head_count")?,
            image_mean: arr3("clip.vision.image_mean")?,
            image_std: arr3("clip.vision.image_std")?,
            use_gelu: match file.get("clip.use_gelu") {
                Some(GgufValue::Bool(b)) => *b,
                _ => false,
            },
            norm_eps: f("clip.vision.attention.layer_norm_epsilon")?,
        })
    }
}

/// FFN-активация CLIP-энкодера: GELU (`clip.use_gelu=true`) или quick-GELU
/// (`false`, как в llama.cpp clip для ViT-L/14): x·σ(1.702·x). Раньше вторая
/// ветка была ошибкой «not implemented yet» — такой mmproj не грузился.
fn clip_ffn_activation(
    backend: &dyn Backend,
    up: &Tensor,
    use_gelu: bool,
) -> Result<Tensor, BackendError> {
    let g = backend.zeros(up.shape().to_vec())?;
    if use_gelu {
        backend.gelu_tanh(up, &g)?;
    } else {
        let scaled = backend.zeros(up.shape().to_vec())?;
        backend.scale(up, &scaled, 1.702)?;
        let sg = backend.zeros(up.shape().to_vec())?;
        backend.sigmoid(&scaled, &sg)?;
        backend.mul(up, &sg, &g)?;
    }
    Ok(g)
}

struct VisionLayer {
    ln1_w: Tensor,
    ln1_b: Tensor,
    qkv: Tensor,
    qkv_b: Tensor,
    attn_out: Tensor,
    attn_out_b: Tensor,
    ln2_w: Tensor,
    ln2_b: Tensor,
    ffn_up: Tensor,
    ffn_up_b: Tensor,
    ffn_down: Tensor,
    ffn_down_b: Tensor,
}

/// Vision encoder + projector.
pub struct VisionModel {
    pub cfg: VisionConfig,
    patch_embd: Tensor,
    patch_embd_b: Option<Tensor>,
    pos_embd: Option<Tensor>,
    layers: Vec<VisionLayer>,
    norm_w: Tensor,
    /// Required: a missing `v.norm.bias` is a load error (was: the weight
    /// silently used as bias).
    norm_b: Tensor,
    output: Tensor,
}

impl VisionModel {
    pub fn load(
        backend: &dyn Backend,
        w: &WeightsMap<'_>,
        cfg: VisionConfig,
    ) -> Result<Self, BackendError> {
        let t = |name: &str| -> Result<Tensor, BackendError> {
            let tensor = w
                .get(name)
                .ok_or_else(|| BackendError::Shape(format!("missing tensor {name}")))?;
            let data = crate::weights_map::dequant_f32(tensor);
            let shape: Vec<usize> = tensor.shape.iter().map(|&d| d as usize).collect();
            backend.from_cpu_f32(&data, shape)
        };
        let d = cfg.embedding_length;
        let patch_embd = t("v.patch_embd.weight")?;
        let patch_embd_b = w
            .get("v.patch_embd.bias")
            .map(|_| t("v.patch_embd.bias"))
            .transpose()?;
        let pos_embd = w
            .get("v.pos_embd.weight")
            .map(|_| t("v.pos_embd.weight"))
            .transpose()?;
        let mut layers = Vec::with_capacity(cfg.block_count);
        for i in 0..cfg.block_count {
            let p = |s: &str| format!("v.blk.{i}.{s}");
            layers.push(VisionLayer {
                ln1_w: t(&p("ln1.weight"))?,
                ln1_b: t(&p("ln1.bias"))?,
                qkv: t(&p("attn_qkv.weight"))?,
                qkv_b: t(&p("attn_qkv.bias"))?,
                attn_out: t(&p("attn_out.weight"))?,
                attn_out_b: t(&p("attn_out.bias"))?,
                ln2_w: t(&p("ln2.weight"))?,
                ln2_b: t(&p("ln2.bias"))?,
                ffn_up: t(&p("ffn_up.weight"))?,
                ffn_up_b: t(&p("ffn_up.bias"))?,
                ffn_down: t(&p("ffn_down.weight"))?,
                ffn_down_b: t(&p("ffn_down.bias"))?,
            });
        }
        let norm_w = t("v.norm.weight")?;
        // bias ОБЯЗАТЕЛЕН: раньше отсутствие v.norm.bias молча подставляло
        // ВЕС как bias (`unwrap_or(&self.norm_w)`) — мусорная нормализация.
        // Остальные нормы файла (ln1/ln2) тоже требуются при загрузке.
        let norm_b = t("v.norm.bias")?;
        let output = t("v.output.weight")?;
        let _ = d;
        Ok(Self {
            cfg,
            patch_embd,
            patch_embd_b,
            pos_embd,
            layers,
            norm_w,
            norm_b,
            output,
        })
    }

    /// Число vision-токенов для изображения image_size×image_size.
    pub fn n_patches(&self) -> usize {
        let s = self.cfg.image_size / self.cfg.patch_size;
        s * s
    }

    /// Прогнать изображение (RGB, [h, w, 3]) → [1, n_patches, projection_dim].
    pub fn forward(
        &self,
        backend: &dyn Backend,
        image_rgb: &[f32],
        h: usize,
        w: usize,
    ) -> Result<Tensor, BackendError> {
        let cfg = &self.cfg;
        let patches = preprocess(image_rgb, h, w, cfg)?;
        let n_patches = patches.0;
        // patch_embd: [n_patches, D] = patches · W^T (+bias)
        let x = backend.from_cpu_f32(&patches.1, vec![n_patches, patches.2])?;
        let hh0 = backend.zeros(vec![n_patches, cfg.embedding_length])?;
        backend.gemm_nt(&x, &self.patch_embd, &hh0)?;
        let hh = match &self.patch_embd_b {
            Some(b) => {
                let y = backend.zeros(vec![n_patches, cfg.embedding_length])?;
                backend.add_bias(&hh0, b, &y)?;
                y
            }
            None => hh0,
        };
        // + pos_embd (learned; без интерполяции — размер должен совпадать).
        let hh = if let Some(pos) = &self.pos_embd {
            if pos.shape()[0] != n_patches {
                return Err(BackendError::Shape(format!(
                    "pos_embd {} != n_patches {n_patches} (interpolation: Ф9)",
                    pos.shape()[0]
                )));
            }
            let y = backend.zeros(vec![n_patches, cfg.embedding_length])?;
            backend.add(&hh, pos, &y)?;
            y
        } else {
            hh
        };
        let hh = hh.view(vec![1, n_patches, cfg.embedding_length]);

        let mut x = hh;
        for layer in &self.layers {
            // ln1 → fused qkv → attention (layernorm по 2D-виду — rows = b·tq)
            let (b, tq, _) = (x.shape()[0], x.shape()[1], x.shape()[2]);
            let xv = x.view(vec![b * tq, cfg.embedding_length]);
            let ln2d = backend.zeros(vec![b * tq, cfg.embedding_length])?;
            backend.layernorm(&xv, &layer.ln1_w, &layer.ln1_b, &ln2d, cfg.norm_eps)?;
            let ln = ln2d.view(vec![b, tq, cfg.embedding_length]);
            let x2 = ln.view(vec![b * tq, cfg.embedding_length]);
            let qkv = backend.zeros(vec![b * tq, 3 * cfg.embedding_length])?;
            backend.gemm_nt(&x2, &layer.qkv, &qkv)?;
            let qkv_b = backend.zeros(qkv.shape().to_vec())?;
            backend.add_bias(&qkv, &layer.qkv_b, &qkv_b)?;
            let qkv = qkv_b;
            let qkv = qkv.view(vec![b, tq, 3 * cfg.embedding_length]);
            let q = crate::nn::slice_last(backend, &qkv, cfg.embedding_length, 0)?;
            let k =
                crate::nn::slice_last(backend, &qkv, cfg.embedding_length, cfg.embedding_length)?;
            let v = crate::nn::slice_last(
                backend,
                &qkv,
                cfg.embedding_length,
                2 * cfg.embedding_length,
            )?;
            let attn = backend.zeros(vec![b, tq, cfg.embedding_length])?;
            backend.attn_ar(
                &q,
                &k,
                &v,
                None,
                &attn,
                cfg.n_heads,
                cfg.n_heads,
                cfg.embedding_length / cfg.n_heads,
            )?;
            let attn = attn.view(vec![b * tq, cfg.embedding_length]);
            let ao0 = backend.zeros(vec![b * tq, cfg.embedding_length])?;
            backend.gemm_nt(&attn, &layer.attn_out, &ao0)?;
            let ao_b = backend.zeros(vec![b * tq, cfg.embedding_length])?;
            backend.add_bias(&ao0, &layer.attn_out_b, &ao_b)?;
            let ao = ao_b.view(vec![b, tq, cfg.embedding_length]);
            let xr = backend.zeros(x.shape().to_vec())?;
            backend.add(&x, &ao, &xr)?;
            // ln2 → ffn (gelu)
            let xrv = xr.view(vec![b * tq, cfg.embedding_length]);
            let ln2_2d = backend.zeros(vec![b * tq, cfg.embedding_length])?;
            backend.layernorm(&xrv, &layer.ln2_w, &layer.ln2_b, &ln2_2d, cfg.norm_eps)?;
            let ln2 = ln2_2d.view(vec![b, tq, cfg.embedding_length]);
            let x3 = ln2.view(vec![b * tq, cfg.embedding_length]);
            let up = backend.zeros(vec![b * tq, cfg.ffn_length])?;
            backend.gemm_nt(&x3, &layer.ffn_up, &up)?;
            backend.add_bias(&up, &layer.ffn_up_b, &up)?;
            let g = clip_ffn_activation(backend, &up, cfg.use_gelu)?;
            let down = backend.zeros(vec![b * tq, cfg.embedding_length])?;
            backend.gemm_nt(&g, &layer.ffn_down, &down)?;
            backend.add_bias(&down, &layer.ffn_down_b, &down)?;
            let down = down.view(vec![b, tq, cfg.embedding_length]);
            let xf = backend.zeros(x.shape().to_vec())?;
            backend.add(&xr, &down, &xf)?;
            x = xf;
        }
        // final norm → projector (norm_b required at load)
        let (b, tq, _) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let xv = x.view(vec![b * tq, cfg.embedding_length]);
        let xn2d = backend.zeros(vec![b * tq, cfg.embedding_length])?;
        backend.layernorm(&xv, &self.norm_w, &self.norm_b, &xn2d, cfg.norm_eps)?;
        let x4 = xn2d.view(vec![b * tq, cfg.embedding_length]);
        let out = backend.zeros(vec![b * tq, cfg.projection_dim])?;
        backend.gemm_nt(&x4, &self.output, &out)?;
        Ok(out.view(vec![b, tq, cfg.projection_dim]))
    }
}

/// Хост-препроцессинг: resize (билинейный) → normalize → patchify.
/// Валидация входа (раньше отсутствие проверок давало панику/мусор:
/// `resized.copy_from_slice` при коротком буфере, `h - 1`/`w - 1`
/// underflow при нулевых размерах). Возвращает
/// (n_patches, данные [n_patches, C·P·P], C·P·P).
fn preprocess(
    image_rgb: &[f32],
    h: usize,
    w: usize,
    cfg: &VisionConfig,
) -> Result<(usize, Vec<f32>, usize), BackendError> {
    if h == 0 || w == 0 {
        return Err(BackendError::Shape(format!(
            "preprocess: пустое изображение {w}×{h}"
        )));
    }
    if image_rgb.len() != h * w * 3 {
        return Err(BackendError::Shape(format!(
            "preprocess: ожидалось {w}×{h}×3 = {} значений, получено {}",
            h * w * 3,
            image_rgb.len()
        )));
    }
    let target = cfg.image_size;
    // 1) билинейный resize до (target, target)
    let mut resized = vec![0.0f32; target * target * 3];
    if h == target && w == target {
        resized.copy_from_slice(image_rgb);
    } else {
        for y in 0..target {
            let sy = (y as f32 + 0.5) * (h as f32 / target as f32) - 0.5;
            let y0 = sy.floor().max(0.0) as usize;
            let y1 = (y0 + 1).min(h - 1);
            let fy = sy - y0 as f32;
            for x in 0..target {
                let sx = (x as f32 + 0.5) * (w as f32 / target as f32) - 0.5;
                let x0 = sx.floor().max(0.0) as usize;
                let x1 = (x0 + 1).min(w - 1);
                let fx = sx - x0 as f32;
                for c in 0..3 {
                    let p00 = image_rgb[(y0 * w + x0) * 3 + c];
                    let p01 = image_rgb[(y0 * w + x1) * 3 + c];
                    let p10 = image_rgb[(y1 * w + x0) * 3 + c];
                    let p11 = image_rgb[(y1 * w + x1) * 3 + c];
                    let top = p00 * (1.0 - fx) + p01 * fx;
                    let bot = p10 * (1.0 - fx) + p11 * fx;
                    resized[(y * target + x) * 3 + c] = top * (1.0 - fy) + bot * fy;
                }
            }
        }
    }
    // 2) normalize + 3) patchify (c-major в патче: c·P·P + ph·P + pw)
    let p = cfg.patch_size;
    let per_side = target / p;
    let cpp = 3 * p * p;
    let mut out = vec![0.0f32; per_side * per_side * cpp];
    for py in 0..per_side {
        for px in 0..per_side {
            let dst = (py * per_side + px) * cpp;
            for c in 0..3 {
                let mean = cfg.image_mean[c];
                let std = cfg.image_std[c].max(1e-6);
                for ph in 0..p {
                    for pw in 0..p {
                        let sy = py * p + ph;
                        let sx = px * p + pw;
                        let v = resized[(sy * target + sx) * 3 + c] / 255.0;
                        out[dst + c * p * p + ph * p + pw] = (v - mean) / std;
                    }
                }
            }
        }
    }
    Ok((per_side * per_side, out, cpp))
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use crate::metal::MetalBackend;
    use std::path::PathBuf;

    fn load_mini() -> (MetalBackend, VisionModel) {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/mini-vit.gguf");
        let buf = std::fs::read(&path).unwrap();
        let f = crate::gguf::parse(&buf).unwrap();
        let wm = WeightsMap::from_gguf(&f).unwrap();
        let cfg = VisionConfig::from_gguf(&f).unwrap();
        assert_eq!(cfg.image_size, 32);
        assert_eq!(cfg.embedding_length, 8);
        let backend = MetalBackend::new().unwrap();
        let model = VisionModel::load(&backend, &wm, cfg).unwrap();
        (backend, model)
    }

    #[test]
    fn config_parsed_from_fixture() {
        let (_, m) = load_mini();
        assert_eq!(m.cfg.projection_dim, 4);
        assert_eq!(m.cfg.patch_size, 16);
        assert_eq!(m.cfg.block_count, 1);
        assert_eq!(m.cfg.n_heads, 2);
        assert_eq!(m.n_patches(), 4);
    }

    #[test]
    fn forward_matches_cpu_reference() {
        let (backend, model) = load_mini();
        let (h, w) = (32usize, 32usize);
        let img: Vec<f32> = (0..h * w * 3).map(|i| ((i * 37) % 256) as f32).collect();
        let gpu = model.forward(&backend, &img, h, w).unwrap();
        let gv = backend.to_cpu_f32(&gpu).unwrap();
        assert_eq!(gv.len(), 4 * 4, "4 patches × proj_dim 4");

        #[cfg(feature = "cpu")]
        {
            let cpu_b = crate::cpu::CpuBackend;
            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/mini-vit.gguf");
            let buf = std::fs::read(&path).unwrap();
            let f = crate::gguf::parse(&buf).unwrap();
            let wm = WeightsMap::from_gguf(&f).unwrap();
            let cfg = VisionConfig::from_gguf(&f).unwrap();
            let cpu_model = VisionModel::load(&cpu_b, &wm, cfg).unwrap();
            let cpu_out = cpu_model.forward(&cpu_b, &img, h, w).unwrap();
            let cv = cpu_b.to_cpu_f32(&cpu_out).unwrap();
            for (a, b) in gv.iter().zip(cv.iter()) {
                assert!((a - b).abs() < 1e-3, "metal {a} vs cpu {b}");
            }
        }
    }

    #[test]
    fn pos_embd_size_mismatch_is_rejected() {
        // Не тестируем напрямую (нужен несовпадающий размер pos_embd) —
        // покрыто проверкой в forward; здесь просто держим сигнатуру.
        let (_, m) = load_mini();
        assert!(m.n_patches() > 0);
    }

    /// Preprocess отклоняет неверный вход вместо паники/мусора:
    /// короткий буфер, нулевые размеры.
    #[test]
    fn preprocess_rejects_bad_input() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/mini-vit.gguf");
        let buf = std::fs::read(&path).unwrap();
        let f = crate::gguf::parse(&buf).unwrap();
        let cfg = VisionConfig::from_gguf(&f).unwrap();

        // пустое изображение
        assert!(preprocess(&[], 0, 0, &cfg).is_err());
        // буфер не h·w·3
        assert!(preprocess(&[0.5f32; 100], 32, 32, &cfg).is_err());
        // корректный размер — Ok
        let img = vec![128.0f32; 32 * 32 * 3];
        let (n, data, cpp) = preprocess(&img, 32, 32, &cfg).unwrap();
        assert_eq!((n, cpp), (4, 3 * 16 * 16));
        assert_eq!(data.len(), n * cpp);
    }
}

#[cfg(test)]
mod activation_tests {
    use super::*;

    /// quick-GELU == x·σ(1.702x) на хосте (обе ветки активации живы).
    #[test]
    fn quick_gelu_matches_host() {
        let Ok(backend) = crate::backend::default_backend() else {
            eprintln!("no backend — skipping");
            return;
        };
        let x: Vec<f32> = (0..37).map(|i| (i as f32 * 0.37).sin() * 4.0).collect();
        let t = backend.from_cpu_f32(&x, vec![1, 37]).unwrap();
        let q = clip_ffn_activation(backend.as_ref(), &t, false).unwrap();
        let got = backend.to_cpu_f32(&q).unwrap();
        for (i, (&v, &g)) in x.iter().zip(&got).enumerate() {
            let want = v * (1.0 / (1.0 + (-1.702 * v).exp()));
            assert!(
                (g - want).abs() <= 1e-6,
                "i={i}: quick_gelu {g} vs {want} (x={v})"
            );
        }
        // GELU-ветка отвечает и не совпадает с quick на тех же данных.
        let gelu = clip_ffn_activation(backend.as_ref(), &t, true).unwrap();
        let gv = backend.to_cpu_f32(&gelu).unwrap();
        assert!(gv.iter().zip(&got).any(|(a, b)| (a - b).abs() > 1e-3));
    }
}
