//! Ф1.11 Qwen3-VL vision (спец-граф llama.cpp qwen3vl.cpp): conv-patch ×2
//! (сумма), spatial merge 2×2, position_embd, 24 блока БЕЗ GQA (fused qkv,
//! hd=64, FFN up/down + ERF-GELU, без gate), deepstack-ветки (слои 5/11/17),
//! post_ln, merger mm.0→gelu→mm.2 → main [576, 2048]; deepstack-блоки
//! (3 × [576, 2048]) возвращаются РАЗДЕЛЬНО — их добавляет к скрытым
//! состояниям image-позиций LLM (Qwen3Model::prefill_with_vision).
//!
//! M-RoPE vision для ОДНОЙ картинки — identity (t=0; роуп нужен только
//! видео); позиции даёт learned position_embd.

use crate::backend::{Backend, Tensor};
use crate::error::BackendError;
use crate::gguf::{GgufFile, GgufValue};
use crate::weights_map::WeightsMap;

pub const VISION_HEADS: usize = 16;
pub const VISION_HD: usize = 64;
pub const DEEPSTACK_LAYERS: [usize; 3] = [5, 11, 17];

struct DeepStack {
    norm_w: Tensor,
    norm_b: Tensor,
    fc1: Tensor,
    fc1_b: Tensor,
    fc2: Tensor,
    fc2_b: Tensor,
}

struct VLayer {
    ln1_w: Tensor,
    ln1_b: Tensor,
    qkv: Tensor,
    qkv_b: Tensor,
    o: Tensor,
    o_b: Tensor,
    ln2_w: Tensor,
    ln2_b: Tensor,
    ffn_up: Tensor,
    ffn_up_b: Tensor,
    ffn_down: Tensor,
    ffn_down_b: Tensor,
    deepstack: Option<DeepStack>,
}

/// Qwen3-VL vision encoder + merger.
pub struct VisionQwen3Vl {
    pub image_size: usize,
    pub patch_size: usize,
    pub n_embd: usize,
    pub ffn_dim: usize,
    pub n_layers: usize,
    pub projection_dim: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    patch_w0: Tensor,
    patch_w1: Tensor,
    patch_bias: Tensor,
    pos_embd: Tensor, // [2304, n_embd], УЖЕ в merge-порядке
    layers: Vec<VLayer>,
    post_ln_w: Tensor,
    post_ln_b: Tensor,
    mm0: Tensor,
    mm0_b: Tensor,
    mm2: Tensor,
    mm2_b: Tensor,
}

impl VisionQwen3Vl {
    pub fn load(
        backend: &dyn Backend,
        w: &WeightsMap<'_>,
        n_layers: usize,
    ) -> Result<Self, BackendError> {
        let t = |name: &str| -> Result<Tensor, BackendError> {
            let tensor = w
                .get(name)
                .ok_or_else(|| BackendError::Shape(format!("missing tensor {name}")))?;
            let data = crate::weights_map::dequant_f32(tensor);
            let shape: Vec<usize> = tensor.shape.iter().map(|&d| d as usize).collect();
            Ok(backend.from_cpu_f32(&data, shape)?)
        };
        // patch_embd.weight: GGUF dims (16,16,3,1024) = kernel (ph,pw,c,out),
        // файловый flat — ne0(ph) быстрейший. Нам нужен [out, c·P·P] c-major
        // (наш patchify: c·P·P + ph·P + pw). Перестановка при загрузке.
        let patch = |name: &str| -> Result<Tensor, BackendError> {
            let tensor = w
                .get(name)
                .ok_or_else(|| BackendError::Shape(format!("missing tensor {name}")))?;
            let raw = crate::weights_map::dequant_f32(tensor);
            let (p, c, out) = (16usize, 3usize, 1024usize);
            let mut flat = vec![0.0f32; out * c * p * p];
            for o in 0..out {
                for cidx in 0..c {
                    for ph in 0..p {
                        for pw in 0..p {
                            // файловый flat: ne0=ph быстрейший → idx = ph + p·pw + p²·c + p²·3·o
                            let src = ph + p * pw + p * p * cidx + p * p * c * o;
                            let dst = o * c * p * p + cidx * p * p + ph * p + pw;
                            flat[dst] = raw[src];
                        }
                    }
                }
            }
            Ok(backend.from_cpu_f32(&flat, vec![out, c * p * p])?)
        };
        let p0 = patch("v.patch_embd.weight")?;
        let p1 = patch("v.patch_embd.weight.1")?;
        let patch_bias = t("v.patch_embd.bias")?;
        let pos_embd = t("v.position_embd.weight")?; // [2304, 1024]
        let n_embd = pos_embd.shape()[1];
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = |s: &str| format!("v.blk.{i}.{s}");
            let deepstack = if DEEPSTACK_LAYERS.contains(&i) {
                let ds = |s: &str| format!("v.deepstack.{i}.{s}");
                Some(DeepStack {
                    norm_w: t(&ds("norm.weight"))?,
                    norm_b: t(&ds("norm.bias"))?,
                    fc1: t(&ds("fc1.weight"))?,
                    fc1_b: t(&ds("fc1.bias"))?,
                    fc2: t(&ds("fc2.weight"))?,
                    fc2_b: t(&ds("fc2.bias"))?,
                })
            } else {
                None
            };
            layers.push(VLayer {
                ln1_w: t(&p("ln1.weight"))?,
                ln1_b: t(&p("ln1.bias"))?,
                qkv: t(&p("attn_qkv.weight"))?,
                qkv_b: t(&p("attn_qkv.bias"))?,
                o: t(&p("attn_out.weight"))?,
                o_b: t(&p("attn_out.bias"))?,
                ln2_w: t(&p("ln2.weight"))?,
                ln2_b: t(&p("ln2.bias"))?,
                ffn_up: t(&p("ffn_up.weight"))?,
                ffn_up_b: t(&p("ffn_up.bias"))?,
                ffn_down: t(&p("ffn_down.weight"))?,
                ffn_down_b: t(&p("ffn_down.bias"))?,
                deepstack,
            });
        }
        let post_ln_w = t("v.post_ln.weight")?;
        let post_ln_b = t("v.post_ln.bias")?;
        let mm0 = t("mm.0.weight")?;
        let mm0_b = t("mm.0.bias")?;
        let mm2 = t("mm.2.weight")?;
        let mm2_b = t("mm.2.bias")?;
        let _ = p0; // размеры проверяются в forward
        let _ = p1;
        Ok(Self {
            image_size: 768,
            patch_size: 16,
            n_embd,
            ffn_dim: 4096,
            n_layers,
            projection_dim: mm2.shape()[0],
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
            patch_w0: p0,
            patch_w1: p1,
            patch_bias,
            pos_embd,
            layers,
            post_ln_w,
            post_ln_b,
            mm0,
            mm0_b,
            mm2,
            mm2_b,
        })
    }

    /// Число vision-токенов: (image/32)² — ОДИН токен на 2×2 блок патчей
    /// (spatial merge 2×2): 768² → 24² = 576. Совпадает с числом строк
    /// main-выхода forward (nq).
    pub fn n_tokens(&self) -> usize {
        let per = self.image_size / (self.patch_size * 2); // 24
        per * per // 576
    }

    /// Прогнать изображение → (main, deepstack):
    /// main — входные эмбеддинги LLM [1, nq, projection_dim] (nq = 576 для
    /// 768²), deepstack — 3 блока (слои 5/11/17) по [1, nq, projection_dim].
    /// Deepstack НЕ конкатенируются с main как отдельные токены — они
    /// добавляются к скрытым состояниям image-позиций внутри первых слоёв
    /// LLM (Qwen3Model::prefill_with_vision; HF _deepstack_process).
    pub fn forward(
        &self,
        backend: &dyn Backend,
        image_rgb: &[f32],
        h: usize,
        w: usize,
    ) -> Result<(Tensor, Vec<Tensor>), BackendError> {
        let n_embd = self.n_embd;
        let p = self.patch_size;
        let per = self.image_size / p; // 48 патчей по стороне
        if h != self.image_size || w != self.image_size {
            return Err(BackendError::Shape(
                "Qwen3-VL: вход приводится к 768×768 (resize в клиенте)".into(),
            ));
        }
        // 1) патчи [per·per, 3·p·p] (c-major) + нормализация
        let cpp = 3 * p * p;
        let mut patches = vec![0.0f32; per * per * cpp];
        for py in 0..per {
            for px in 0..per {
                let dst = (py * per + px) * cpp;
                for c in 0..3 {
                    let mean = self.image_mean[c];
                    let std = self.image_std[c].max(1e-6);
                    for ph in 0..p {
                        for pw in 0..p {
                            let v = image_rgb
                                [((py * p + ph) * self.image_size + px * p + pw) * 3 + c]
                                / 255.0;
                            patches[dst + c * p * p + ph * p + pw] = (v - mean) / std;
                        }
                    }
                }
            }
        }
        // 2) conv ×2 (линейный на патчах) + сумма
        let xp = backend.from_cpu_f32(&patches, vec![per * per, cpp])?;
        let c0 = backend.zeros(vec![per * per, n_embd])?;
        backend.gemm_nt(&xp, &self.patch_w0, &c0)?;
        let c1 = backend.zeros(vec![per * per, n_embd])?;
        backend.gemm_nt(&xp, &self.patch_w1, &c1)?;
        let conv = backend.zeros(vec![per * per, n_embd])?;
        backend.add(&c0, &c1, &conv)?;
        if std::env::var("VISION_DBG").is_ok() {
            let dv = backend.to_cpu_f32(&conv)?;
            eprintln!("DBG conv row0[0..4]: {:?}", &dv[..4]);
        }
        // 3) merge 2×2 + patch_bias + position_embd — одним кернелом (GPU)
        let per = per; // сетка патчей (per² = 2304)
        let mut x = backend.zeros(vec![per * per, n_embd])?;
        backend.merge2x2_add(&conv, &self.patch_bias, &self.pos_embd, &mut x, per)?;
        if std::env::var("VISION_DBG").is_ok() {
            let dv = backend.to_cpu_f32(&x)?;
            eprintln!("DBG after merge+pos row0[0..4]: {:?}", &dv[..4]);
        }
        // 5) блоки (M-RoPE vision для картинки — identity: t=0)
        let n_pos = per * per;
        let mut deepstack_out: Vec<Tensor> = Vec::new();
        for (il, layer) in self.layers.iter().enumerate() {
            let x2 = x.view(vec![n_pos, n_embd]);
            let ln = backend.zeros(vec![n_pos, n_embd])?;
            backend.layernorm(&x2, &layer.ln1_w, &layer.ln1_b, &ln, 1e-6)?;
            if std::env::var("VISION_DBG").is_ok() && il == 0 {
                let dv = backend.to_cpu_f32(&ln)?;
                eprintln!("DBG ln1 row0[0..4]: {:?}", &dv[..4]);
            }
            let qkv = backend.zeros(vec![n_pos, 3 * n_embd])?;
            backend.gemm_nt(&ln, &layer.qkv, &qkv)?;
            if std::env::var("VISION_DBG").is_ok() && il == 0 {
                let dv = backend.to_cpu_f32(&qkv)?;
                eprintln!("DBG qkv row0[0..4]: {:?}", &dv[..4]);
            }
            let qkv_b = backend.zeros(vec![n_pos, 3 * n_embd])?;
            backend.add_bias(&qkv, &layer.qkv_b, &qkv_b)?;
            let qkv3 = qkv_b.view(vec![1, n_pos, 3 * n_embd]);
            let q = crate::nn::slice_last(backend, &qkv3, n_embd, 0)?;
            let k = crate::nn::slice_last(backend, &qkv3, n_embd, n_embd)?;
            let v = crate::nn::slice_last(backend, &qkv3, n_embd, 2 * n_embd)?;
            let attn = backend.zeros(vec![1, n_pos, n_embd])?;
            backend.attn_ar(
                &q,
                &k,
                &v,
                None,
                &attn,
                VISION_HEADS,
                VISION_HEADS,
                VISION_HD,
            )?;
            if std::env::var("VISION_DBG").is_ok() && il == 0 {
                let dv = backend.to_cpu_f32(&attn)?;
                eprintln!("DBG attn row0[0..4]: {:?}", &dv[..4]);
            }
            let attn2 = attn.view(vec![n_pos, n_embd]);
            let o0 = backend.zeros(vec![n_pos, n_embd])?;
            backend.gemm_nt(&attn2, &layer.o, &o0)?;
            let o = backend.zeros(vec![n_pos, n_embd])?;
            backend.add_bias(&o0, &layer.o_b, &o)?;
            let xr = backend.zeros(vec![n_pos, n_embd])?;
            backend.add(&x, &o, &xr)?;
            // ffn: up → gelu(erf) → down
            let x3 = xr.view(vec![n_pos, n_embd]);
            let ln2 = backend.zeros(vec![n_pos, n_embd])?;
            backend.layernorm(&x3, &layer.ln2_w, &layer.ln2_b, &ln2, 1e-6)?;
            let up = backend.zeros(vec![n_pos, self.ffn_dim])?;
            backend.gemm_nt(&ln2, &layer.ffn_up, &up)?;
            backend.add_bias(&up, &layer.ffn_up_b, &up)?;
            let g = backend.zeros(vec![n_pos, self.ffn_dim])?;
            backend.gelu_erf(&up, &g)?;
            let down = backend.zeros(vec![n_pos, n_embd])?;
            backend.gemm_nt(&g, &layer.ffn_down, &down)?;
            backend.add_bias(&down, &layer.ffn_down_b, &down)?;
            let xf = backend.zeros(vec![n_pos, n_embd])?;
            backend.add(&xr, &down, &xf)?;
            x = xf;
            if std::env::var("VISION_DBG").is_ok() && (il == 0 || il == 5) {
                let dv = backend.to_cpu_f32(&x)?;
                eprintln!("DBG after block {il} row0[0..4]: {:?}", &dv[..4]);
            }
            // deepstack: [n_pos, n_embd] → [n_pos/4, 4·n_embd] (merge 4 соседних)
            if let Some(ds) = &layer.deepstack {
                let nq = n_pos / 4;
                let g4 = backend.zeros(vec![nq, 4 * n_embd])?;
                backend.group4(&x, &g4)?;
                let ln = backend.zeros(vec![nq, 4 * n_embd])?;
                backend.layernorm(&g4, &ds.norm_w, &ds.norm_b, &ln, 1e-6)?;
                let f1 = backend.zeros(vec![nq, 4 * n_embd])?;
                backend.gemm_nt(&ln, &ds.fc1, &f1)?;
                backend.add_bias(&f1, &ds.fc1_b, &f1)?;
                let g1 = backend.zeros(vec![nq, 4 * n_embd])?;
                backend.gelu_erf(&f1, &g1)?;
                let f2 = backend.zeros(vec![nq, self.projection_dim])?;
                backend.gemm_nt(&g1, &ds.fc2, &f2)?;
                backend.add_bias(&f2, &ds.fc2_b, &f2)?;
                deepstack_out.push(f2);
            }
        }
        // 6) post_ln (к 1024-мерным) → reshape [n_pos/4, 4·n_embd] → merger
        let nq = n_pos / 4;
        let ln0 = backend.zeros(vec![n_pos, n_embd])?;
        backend.layernorm(&x, &self.post_ln_w, &self.post_ln_b, &ln0, 1e-6)?;
        let g4 = backend.zeros(vec![nq, 4 * n_embd])?;
        backend.group4(&ln0, &g4)?;
        let m0 = backend.zeros(vec![nq, 4 * n_embd])?;
        backend.gemm_nt(&g4, &self.mm0, &m0)?;
        backend.add_bias(&m0, &self.mm0_b, &m0)?;
        let g0 = backend.zeros(vec![nq, 4 * n_embd])?;
        backend.gelu_erf(&m0, &g0)?;
        let main = backend.zeros(vec![nq, self.projection_dim])?;
        backend.gemm_nt(&g0, &self.mm2, &main)?;
        backend.add_bias(&main, &self.mm2_b, &main)?;
        if std::env::var("VISION_DBG").is_ok() {
            let dv = backend.to_cpu_f32(&main)?;
            eprintln!("DBG main row0[0..4]: {:?}", &dv[..4]);
        }
        // 7) main + deepstack — РАЗДЕЛЬНО: main [1, nq, proj] — входные
        // эмбеддинги LLM; три deepstack-блока [1, nq, proj] добавляются
        // к скрытым состояниям image-позиций после слоёв 0..2 LLM
        // (см. Qwen3Model::prefill_with_vision). Конкатенация блоков как
        // «дополнительных токенов» (4·nq строк) была ошибкой: deepstack —
        // НЕ токены, а добавки к позициям.
        let main = main.view(vec![1, nq, self.projection_dim]);
        Ok((main, deepstack_out))
    }
}

/// Перестановка 2×2-блоков: вход (rows, d) сетка per×per → выход (rows, d),
/// где 4 соседних патча блока (by,bx) становятся последовательными в порядке
/// (c0a0, c1a0, c0a1, c1a1) — c=по x (2bx, 2bx+1), a=по y (2by, 2by+1).
pub fn merge_2x2(data: &[f32], per: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; per * per * d];
    let half = per / 2;
    for by in 0..half {
        for bx in 0..half {
            let dst_block = (by * half + bx) * 4 * d;
            // (c0a0): патч (2by, 2bx)
            let src = (2 * by * per + 2 * bx) * d;
            out[dst_block..dst_block + d].copy_from_slice(&data[src..src + d]);
            // (c1a0): (2by, 2bx+1)
            let src = (2 * by * per + 2 * bx + 1) * d;
            out[dst_block + d..dst_block + 2 * d].copy_from_slice(&data[src..src + d]);
            // (c0a1): (2by+1, 2bx)
            let src = ((2 * by + 1) * per + 2 * bx) * d;
            out[dst_block + 2 * d..dst_block + 3 * d].copy_from_slice(&data[src..src + d]);
            // (c1a1): (2by+1, 2bx+1)
            let src = ((2 * by + 1) * per + 2 * bx + 1) * d;
            out[dst_block + 3 * d..dst_block + 4 * d].copy_from_slice(&data[src..src + d]);
        }
    }
    out
}

/// Группировка 4 соседних токенов в один: [n, d] → [n/4, 4·d].
pub fn group4(data: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; (n / 4) * 4 * d];
    for i in 0..n / 4 {
        for j in 0..4 {
            let src = (i * 4 + j) * d;
            let dst = (i * 4 * d) + j * d;
            out[dst..dst + d].copy_from_slice(&data[src..src + d]);
        }
    }
    out
}

/// Конфиг из clip.* метадаты (image_size/patch) — для проверки.
pub fn config_from_gguf(file: &GgufFile<'_>) -> (usize, usize) {
    let u = |k: &str| file.get(k).and_then(GgufValue::as_u32).map(|v| v as usize);
    (
        u("clip.vision.image_size").unwrap_or(768),
        u("clip.vision.patch_size").unwrap_or(16),
    )
}
