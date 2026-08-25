//! Ф1.12: Qwen2.5-Omni — аудио-вход (полудуплекс: аудио → текст).
//!
//! Пайплайн (бит-в-бит transformers `Qwen2_5OmniThinkerForConditionalGeneration`):
//! 1. `audio::log_mel_spectrogram` — mel 128×T (WhisperFeatureExtractor).
//! 2. `OmniAudioTower` — conv1(128→1280,s1) → GELU → conv2(s2) → GELU →
//!    синусоиды → 32× (LayerNorm+self-attn → FFN GELU) → ln_post →
//!    avg-pool парами → proj(1280→2048). Окна по 200 кадров (n_window·2),
//!    внимание внутри окна полное (mask=None).
//! 3. Аудио-токены вставляются на позиции `<|audio|>` (151646) в
//!    `input_embeds`, M-RoPE позиции t=h=w=pos по порядку; Thinker —
//!    та же Qwen3-архитектура (переиспользует `Qwen3Model`), mrope_sections
//!    [16,24,24,0], q/k/v-проекции С bias (Qwen2.5), QK-norm отсутствует.

use std::collections::HashMap;
use std::path::Path;

use crate::audio;
use crate::backend::{Backend, Tensor};
use crate::error::BackendError;
use crate::nn::Linear;
use crate::qwen3::{Qwen3Config, Qwen3Model};
use crate::weights::{read_safetensors_f32, F32Tensors};
use crate::weights_map::{WeightTensor, WeightsMap};
use crate::dtype::{Dtype, WeightFormat};

// ─────────────────────────── конфиг ───────────────────────────

#[derive(Debug, Clone)]
pub struct OmniAudioConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub ffn_dim: usize,
    pub n_mel: usize,
    pub output_dim: usize,
    /// Полу-окно: чанки по n_window·2 кадров.
    pub n_window: usize,
    pub max_source_positions: usize,
}

#[derive(Debug, Clone)]
pub struct OmniThinkerConfig {
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
    pub audio_bos_id: u32,
    pub audio_eos_id: u32,
    pub audio_token_id: u32,
    pub position_id_per_seconds: usize,
}

#[derive(Debug, Clone)]
pub struct OmniConfig {
    pub audio: OmniAudioConfig,
    pub thinker: OmniThinkerConfig,
}

fn get<'a>(v: &'a serde_json::Value, path: &[&str]) -> Option<&'a serde_json::Value> {
    let mut cur = v;
    for p in path {
        cur = cur.get(*p)?;
    }
    Some(cur)
}

fn get_u64(v: &serde_json::Value, path: &[&str]) -> Result<usize, BackendError> {
    get(v, path)
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .ok_or_else(|| BackendError::Shape(format!("config: missing {:?}", path)))
}

fn get_f64(v: &serde_json::Value, path: &[&str]) -> Result<f32, BackendError> {
    get(v, path)
        .and_then(|x| x.as_f64())
        .map(|x| x as f32)
        .ok_or_else(|| BackendError::Shape(format!("config: missing {:?}", path)))
}

/// Читает config.json Qwen2.5-Omni-3B (thinker_config.text_config /
/// audio_config + спец-токены).
pub fn config_from_json(dir: &Path) -> Result<OmniConfig, BackendError> {
    let raw = std::fs::read_to_string(dir.join("config.json"))
        .map_err(|e| BackendError::Shape(format!("config.json: {e}")))?;
    let v: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| BackendError::Shape(format!("config.json: {e}")))?;
    let thinker = get(&v, &["thinker_config"])
        .ok_or_else(|| BackendError::Shape("no thinker_config".into()))?;
    let text = get(thinker, &["text_config"])
        .ok_or_else(|| BackendError::Shape("no text_config".into()))?;
    let acfg = get(thinker, &["audio_config"])
        .ok_or_else(|| BackendError::Shape("no audio_config".into()))?;
    let mrope = get(text, &["rope_scaling", "mrope_section"])
        .and_then(|x| x.as_array())
        .map(|a| {
            let mut out = [0usize; 4];
            for (i, x) in a.iter().take(4).enumerate() {
                out[i] = x.as_u64().unwrap_or(0) as usize;
            }
            out
        })
        .ok_or_else(|| BackendError::Shape("mrope_section missing".into()))?;
    let audio = OmniAudioConfig {
        d_model: get_u64(acfg, &["d_model"])?,
        n_layers: get_u64(acfg, &["encoder_layers"])?,
        n_heads: get_u64(acfg, &["encoder_attention_heads"])?,
        ffn_dim: get_u64(acfg, &["encoder_ffn_dim"])?,
        n_mel: get_u64(acfg, &["num_mel_bins"])?,
        output_dim: get_u64(acfg, &["output_dim"])?,
        n_window: get_u64(acfg, &["n_window"])?,
        max_source_positions: get_u64(acfg, &["max_source_positions"])?,
    };
    let thinker = OmniThinkerConfig {
        dim: get_u64(text, &["hidden_size"])?,
        n_layers: get_u64(text, &["num_hidden_layers"])?,
        n_heads: get_u64(text, &["num_attention_heads"])?,
        n_kv_heads: get_u64(text, &["num_key_value_heads"])?,
        head_dim: get_u64(text, &["head_dim"]).unwrap_or_else(|_| {
            get_u64(text, &["hidden_size"]).unwrap() / get_u64(text, &["num_attention_heads"]).unwrap()
        }),
        ffn_dim: get_u64(text, &["intermediate_size"])?,
        vocab_size: get_u64(text, &["vocab_size"])?,
        rope_theta: get_f64(text, &["rope_theta"])?,
        norm_eps: get_f64(text, &["rms_norm_eps"])?,
        audio_bos_id: get_u64(thinker, &["audio_start_token_id"]).map(|x| x as u32)?,
        audio_eos_id: get_u64(thinker, &["audio_end_token_id"]).map(|x| x as u32)?,
        audio_token_id: get_u64(thinker, &["audio_token_index"]).map(|x| x as u32)?,
        position_id_per_seconds: get_u64(thinker, &["position_id_per_seconds"])?,
    };
    let _ = mrope; // фиксируем [16,24,24] — совпадает с GGUF-проверкой
    Ok(OmniConfig { audio, thinker })
}

// ─────────────────────── аудио-энкодер ───────────────────────

struct AudioAttn {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
}

struct AudioLayer {
    attn_ln_w: Tensor,
    attn_ln_b: Tensor,
    attn: AudioAttn,
    ffn_ln_w: Tensor,
    ffn_ln_b: Tensor,
    fc1: Linear,
    fc2: Linear,
}

/// Whisper-стиль аудио-энкодер Qwen2.5-Omni (d_model=1280, 32 слоя,
/// 20 голов full attention, GELU, bias во всех проекциях кроме k).
pub struct OmniAudioTower {
    pub cfg: OmniAudioConfig,
    pub conv1_w: Tensor,
    pub conv1_b: Tensor,
    conv2_w: Tensor,
    conv2_b: Tensor,
    pos_emb: Tensor, // [1, max_source_positions, d_model]
    layers: Vec<AudioLayer>,
    ln_post_w: Tensor,
    ln_post_b: Tensor,
    proj_w: Tensor,
    proj_b: Tensor,
}

impl OmniAudioTower {
    pub fn load(
        backend: &dyn Backend,
        t: &F32Tensors,
        cfg: &OmniAudioConfig,
    ) -> Result<Self, BackendError> {
        let g = |name: &str| -> Result<Vec<f32>, BackendError> {
            t.get(&format!("thinker.audio_tower.{name}"))
                .map(|(d, _)| d.clone())
                .ok_or_else(|| BackendError::Shape(format!("audio_tower.{name} missing")))
        };
        let up = |backend: &dyn Backend,
                  data: &[f32],
                  shape: Vec<usize>|
         -> Result<Tensor, BackendError> { backend.from_cpu_f32(data, shape) };
        let ln = |backend: &dyn Backend, name: &str, d: usize| -> Result<(Tensor, Tensor), BackendError> {
            Ok((
                up(backend, &g(&format!("{name}.weight"))?, vec![d])?,
                up(backend, &g(&format!("{name}.bias"))?, vec![d])?,
            ))
        };
        let lin = |backend: &dyn Backend,
                   name: &str,
                   rows: usize,
                   cols: usize,
                   has_bias: bool|
         -> Result<Linear, BackendError> {
            let w = g(&format!("{name}.weight"))?;
            let b = if has_bias {
                Some(g(&format!("{name}.bias"))?)
            } else {
                None
            };
            Linear::from_weights(backend, w, vec![rows, cols], b)
        };

        let d = cfg.d_model;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = format!("layers.{i}.");
            let (attn_ln_w, attn_ln_b) = ln(backend, &format!("{p}self_attn_layer_norm"), d)?;
            let (ffn_ln_w, ffn_ln_b) = ln(backend, &format!("{p}final_layer_norm"), d)?;
            layers.push(AudioLayer {
                attn_ln_w,
                attn_ln_b,
                attn: AudioAttn {
                    q: lin(backend, &format!("{p}self_attn.q_proj"), d, d, true)?,
                    k: lin(backend, &format!("{p}self_attn.k_proj"), d, d, false)?,
                    v: lin(backend, &format!("{p}self_attn.v_proj"), d, d, true)?,
                    o: lin(backend, &format!("{p}self_attn.out_proj"), d, d, true)?,
                },
                ffn_ln_w,
                ffn_ln_b,
                fc1: lin(backend, &format!("{p}fc1"), cfg.ffn_dim, d, true)?,
                fc2: lin(backend, &format!("{p}fc2"), d, cfg.ffn_dim, true)?,
            });
        }
        let (ln_post_w, ln_post_b) = ln(backend, "ln_post", d)?;
        let conv1_w = g("conv1.weight")?;
        let conv2_w = g("conv2.weight")?;
        let pos = audio::sinusoids(cfg.max_source_positions, d, 10000.0);
        Ok(Self {
            cfg: cfg.clone(),
            conv1_w: up(backend, &conv1_w, vec![d, cfg.n_mel, 3])?,
            conv1_b: up(backend, &g("conv1.bias")?, vec![d])?,
            conv2_w: up(backend, &conv2_w, vec![d, d, 3])?,
            conv2_b: up(backend, &g("conv2.bias")?, vec![d])?,
            pos_emb: up(backend, &pos, vec![1, cfg.max_source_positions, d])?,
            layers,
            ln_post_w,
            ln_post_b,
            proj_w: up(backend, &g("proj.weight")?, vec![cfg.output_dim, d])?,
            proj_b: up(backend, &g("proj.bias")?, vec![cfg.output_dim])?,
        })
    }

    /// mel [1, 128, T] (уже log-нормализованный) → аудио-токены [N, output_dim].
    /// mel [1, T, n_mel] (уже log-нормализованный, time-major как conv1d)
    /// → аудио-токены [N, output_dim].
    /// Окна по n_window·2 кадров (внимание внутри окна — полное, без маски).
    pub fn forward(&self, backend: &dyn Backend, mel: &Tensor) -> Result<Tensor, BackendError> {
        let cfg = &self.cfg;
        let (_, t, n_mel) = (mel.shape()[0], mel.shape()[1], mel.shape()[2]);
        if n_mel != cfg.n_mel {
            return Err(BackendError::Shape(format!("mel bins {n_mel} != {}", cfg.n_mel)));
        }
        let mel_cpu = backend.to_cpu_f32(mel)?;
        let win = cfg.n_window * 2;
        let mut out_rows: Vec<f32> = Vec::new();
        let mut start = 0usize;
        while start < t {
            let len = (t - start).min(win);
            // срез mel [1, len, n_mel] (строки start..start+len, time-major)
            let mut data = vec![0.0f32; len * n_mel];
            for i in 0..len {
                for m in 0..n_mel {
                    data[i * n_mel + m] = mel_cpu[(start + i) * n_mel + m];
                }
            }
            let x = backend.from_cpu_f32(&data, vec![1, len, n_mel])?;
            let tokens = self.forward_window(backend, &x)?;
            out_rows.extend(backend.to_cpu_f32(&tokens)?);
            start += len;
        }
        backend.from_cpu_f32(&out_rows, vec![out_rows.len() / cfg.output_dim, cfg.output_dim])
    }

    /// Одно окно (≤ 200 кадров): conv → pos → слои → ln_post → avg-pool → proj.
    /// x — time-major [1, T, n_mel] (как conv1d).
    pub(crate) fn forward_window(&self, backend: &dyn Backend, x: &Tensor) -> Result<Tensor, BackendError> {
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let len = x.shape()[1];

        // conv1 (stride 1) → GELU
        let h1 = backend.zeros(vec![1, len, d])?;
        backend.conv1d(x, &self.conv1_w, Some(&self.conv1_b), 1, 1, 1, 1, &h1)?;
        if std::env::var("OMNI_REF").is_ok() {
            let h1v = backend.to_cpu_f32(&h1)?;
            eprintln!("OMNI_REF our_conv1_raw[0,:8]={:?}", &h1v[..8.min(h1v.len())]);
        }
        backend.gelu_erf(&h1, &h1)?;
        // conv2 (stride 2) → GELU; l2 = (len - 1)/2 + 1
        let l2 = (len - 1) / 2 + 1;
        let mut h = backend.zeros(vec![1, l2, d])?;
        backend.conv1d(&h1, &self.conv2_w, Some(&self.conv2_b), 2, 1, 1, 1, &h)?;
        backend.gelu_erf(&h, &h)?;
        if std::env::var("OMNI_REF").is_ok() {
            let h1v = backend.to_cpu_f32(&h1)?;
            let h2v = backend.to_cpu_f32(&h)?;
            eprintln!("OMNI_REF our_conv1[0,:8]={:?}", &h1v[..8.min(h1v.len())]);
            eprintln!("OMNI_REF our_conv2[0,:8]={:?}", &h2v[..8.min(h2v.len())]);
        }
        // + синусоидальные позиции [0..l2)
        let pos = self.pos_emb.view_offset(vec![1, l2, d], 0);
        let mut h = {
            let mut y = backend.zeros(vec![1, l2, d])?;
            backend.add(&h, &pos, &mut y)?;
            y
        };
        if std::env::var("OMNI_REF").is_ok() {
            let hv = backend.to_cpu_f32(&h)?;
            eprintln!("OMNI_REF our_pos[0,:8]={:?}", &hv[..8.min(hv.len())]);
        }

        for (li, layer) in self.layers.iter().enumerate() {
            // self-attn (полный, mask=None)
            let hn = backend.zeros(vec![1, l2, d])?;
            backend.layernorm(&h, &layer.attn_ln_w, &layer.attn_ln_b, &hn, 1e-5)?;
            let q = layer.attn.q.forward(backend, &hn)?;
            let k = layer.attn.k.forward(backend, &hn)?;
            let v = layer.attn.v.forward(backend, &hn)?;
            let attn_out = backend.zeros(vec![1, l2, d])?;
            backend.attn_ar(&q, &k, &v, None, &attn_out, cfg.n_heads, cfg.n_heads, d / cfg.n_heads)?;
            let o = layer.attn.o.forward(backend, &attn_out)?;
            // attn-выход: h = h + o (переприсваивание ВНЕШНЕЙ mut h —
            // `let h =` в цикле создал бы тень и ПОТЕРЯЛ результат между итерациями)
            h = {
                let mut y = backend.zeros(vec![1, l2, d])?;
                backend.add(&h, &o, &mut y)?;
                y
            };
            if std::env::var("OMNI_REF").is_ok() {
                let hv = backend.to_cpu_f32(&h)?;
                eprintln!("OMNI_REF t_after_attn[0,:8]={:?}", &hv[..8.min(hv.len())]);
            }
            // FFN
            let hn2 = backend.zeros(vec![1, l2, d])?;
            backend.layernorm(&h, &layer.ffn_ln_w, &layer.ffn_ln_b, &hn2, 1e-5)?;
            if std::env::var("OMNI_REF").is_ok() {
                let hv = backend.to_cpu_f32(&hn2)?;
                eprintln!("OMNI_REF t_hn2[0,:4]={:?}", &hv[..4.min(hv.len())]);
            }
            let f = layer.fc1.forward(backend, &hn2)?;
            backend.gelu_erf(&f, &f)?;
            let f = layer.fc2.forward(backend, &f)?;
            if std::env::var("OMNI_REF").is_ok() {
                let hv = backend.to_cpu_f32(&f)?;
                eprintln!("OMNI_REF t_after_ffn[0,:8]={:?}", &hv[..8.min(hv.len())]);
            }
            h = {
                let mut y = backend.zeros(vec![1, l2, d])?;
                backend.add(&h, &f, &mut y)?;
                y
            };
        }

        // ln_post → avg-pool парами → proj
        let hp = backend.zeros(vec![1, l2, d])?;
        backend.layernorm(&h, &self.ln_post_w, &self.ln_post_b, &hp, 1e-5)?;
        if std::env::var("OMNI_REF").is_ok() {
            let hv = backend.to_cpu_f32(&hp)?;
            eprintln!("OMNI_REF t_ln_post[0,:8]={:?}", &hv[..8.min(hv.len())]);
        }
        let n_pool = (l2 - 2) / 2 + 1;
        let hp_cpu = backend.to_cpu_f32(&hp)?;
        let mut pooled = vec![0.0f32; n_pool * d];
        for i in 0..n_pool {
            for j in 0..d {
                pooled[i * d + j] = 0.5 * (hp_cpu[(2 * i) * d + j] + hp_cpu[(2 * i + 1) * d + j]);
            }
        }
        let p = backend.from_cpu_f32(&pooled, vec![1, n_pool, d])?;
        let y = Linear {
            w: self.proj_w.clone(),
            b: Some(self.proj_b.clone()),
        }
        .forward(backend, &p)?;
        Ok(y.view(vec![n_pool, cfg.output_dim]))
    }
}

// ─────────────────────── загрузка модели ───────────────────────

pub struct OmniModel {
    pub cfg: OmniConfig,
    pub tower: OmniAudioTower,
    pub thinker: Qwen3Model,
}

/// Переименование тензоров thinker → имена Qwen3Model::load
/// (blk.{i}.attn_q / ffn_gate / ...), отбрасывая bias-тензоры mlp и
/// не-думательные части (talker/token2wav/visual/audio_tower).
fn map_thinker_name(name: &str) -> Option<String> {
    let rest = name.strip_prefix("thinker.")?;
    let mapped = match rest {
        "model.embed_tokens.weight" => "token_embd.weight".to_string(),
        "model.norm.weight" => "output_norm.weight".to_string(),
        "lm_head.weight" => "output.weight".to_string(),
        r if r.starts_with("model.layers.") => {
            let parts: Vec<&str> = r.split('.').collect();
            // model.layers.{i}.{module}.{proj}.{kind} (6) ИЛИ
            // model.layers.{i}.{module}.{kind} (5 — input_layernorm и т.п.)
            let (i, module, proj, kind) = match parts.as_slice() {
                [_, _, i, module, proj, kind] => (i, *module, Some(*proj), *kind),
                [_, _, i, module, kind] => (i, *module, None, *kind),
                _ => return None,
            };
            let key = match (module, proj) {
                ("self_attn", Some("q_proj")) => "attn_q",
                ("self_attn", Some("k_proj")) => "attn_k",
                ("self_attn", Some("v_proj")) => "attn_v",
                ("self_attn", Some("o_proj")) => "attn_output",
                ("self_attn", Some("q_norm")) | ("self_attn", Some("k_norm")) => return None,
                ("input_layernorm", _) => "attn_norm",
                ("post_attention_layernorm", _) => "ffn_norm",
                ("mlp", Some("gate_proj")) => "ffn_gate",
                ("mlp", Some("up_proj")) => "ffn_up",
                ("mlp", Some("down_proj")) => "ffn_down",
                _ => return None,
            };
            format!("blk.{i}.{key}.{kind}")
        }
        _ => return None,
    };
    Some(mapped)
}

/// Собрать WeightsMap (f32) из прочитанных safetensors-тензоров thinker'а и
/// загрузить Qwen3Model. raw-хранилище живёт в этой функции — Qwen3Model::load
/// копирует всё в GPU до возврата.
fn load_thinker(
    backend: &dyn Backend,
    t: &F32Tensors,
    cfg: &OmniThinkerConfig,
) -> Result<Qwen3Model, BackendError> {
    let mut raw: Vec<(String, Vec<u8>, Vec<u64>)> = Vec::new();
    for (name, (data, shape)) in t {
        if let Some(mapped) = map_thinker_name(name) {
            // attn_q/k/v.bias обрабатывает Qwen3Model::load (qkv_bias);
            // остальные bias в Qwen3-архитектуре не используются.
            let is_attn_bias = mapped.ends_with(".bias")
                && (mapped.contains("attn_q.") || mapped.contains("attn_k.") || mapped.contains("attn_v."));
            if mapped.ends_with(".bias") && !is_attn_bias {
                continue;
            }
            let mut bytes = Vec::with_capacity(data.len() * 4);
            for v in data {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            raw.push((mapped, bytes, shape.iter().map(|&d| d as u64).collect()));
        }
    }
    // Реальные (накопительные) офсеты: tied_aliases в Qwen3Model::load
    // сравнивает offset+len — при offset=0 у всех тензоров output.weight
    // ложно «связывался» с token_embd.weight (одинаковые длины) и выходная
    // проекция подменялась эмбеддингами (мусорные логиты на спец-токенах).
    let mut off = 0u64;
    let tensors: Vec<WeightTensor> = raw
        .iter()
        .map(|(n, bytes, shape)| {
            let t = WeightTensor {
                name: n.clone(),
                shape: shape.clone(),
                format: WeightFormat::Plain(Dtype::F32),
                offset: off,
                data: bytes,
            };
            off += bytes.len() as u64;
            t
        })
        .collect();
    let wm = WeightsMap {
        arch: "qwen3".into(),
        tensors,
        storage: vec![],
    };
    eprintln!("thinker weights: {} tensors", wm.tensors.len());
    let thinker_cfg = Qwen3Config {
        dim: cfg.dim,
        n_layers: cfg.n_layers,
        n_heads: cfg.n_heads,
        n_kv_heads: cfg.n_kv_heads,
        head_dim: cfg.head_dim,
        ffn_dim: cfg.ffn_dim,
        vocab_size: cfg.vocab_size,
        rope_theta: cfg.rope_theta,
        rope_max_seq: 32768,
        norm_eps: cfg.norm_eps,
        swa_window: None,
        mrope_sections: Some([16, 24, 24, 0]),
        n_deepstack: None,
    };
    Qwen3Model::load(backend, &wm, thinker_cfg)
}

impl OmniModel {
    /// dir — каталог с config.json + model-0000{i}-of-00003.safetensors.
    pub fn load(backend: &dyn Backend, dir: &Path) -> Result<Self, BackendError> {
        let cfg = config_from_json(dir)?;
        let mut all: F32Tensors = HashMap::new();
        let mut i = 1;
        loop {
            let p = dir.join(format!("model-{i:05}-of-00003.safetensors"));
            if !p.exists() {
                break;
            }
            let loaded = read_safetensors_f32(&p)
                .map_err(|e| BackendError::Shape(format!("{}: {e}", p.display())))?;
            let n = loaded.len();
            all.extend(loaded);
            eprintln!("loaded shard {i}: {n} tensors");
            i += 1;
        }
        if i == 1 {
            return Err(BackendError::Shape(format!(
                "no safetensors shards in {}",
                dir.display()
            )));
        }
        let tower = OmniAudioTower::load(backend, &all, &cfg.audio)?;
        let thinker = load_thinker(backend, &all, &cfg.thinker)?;
        Ok(Self { cfg, tower, thinker })
    }

    /// Собрать вход: ids с audio_bos + N×audio_token + audio_eos,
    /// аудио-эмбеддинги [N, 2048] и M-RoPE позиции (t=h=w=pos).
    /// Возвращает (ids, audio_features, pos4d, pos_after).
    pub fn build_audio_input(
        &self,
        backend: &dyn Backend,
        wav: &[f32],
        text_after: &str,
        tokenizer: &crate::tokenizer::Tokenizer,
    ) -> Result<(Vec<u32>, Tensor, Vec<u32>, usize), BackendError> {
        let (mel, t) = audio::log_mel_spectrogram(wav);
        // mel — [128, T] row-major (m*T+i); tower ждёт time-major [T, 128] —
        // транспонируем (перестановка, не reshape!)
        let mut mel_tm = vec![0.0f32; t * 128];
        for i in 0..t {
            for m in 0..128 {
                mel_tm[i * 128 + m] = mel[m * t + i];
            }
        }
        let mel_t = backend.from_cpu_f32(&mel_tm, vec![1, t, 128])?;
        let audio_tokens = self.tower.forward(backend, &mel_t)?;
        let n = audio_tokens.shape()[0];
        let ids = {
            let mut ids = tokenizer.encode(
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n\
                 <|im_start|>user\n",
            );
            ids.push(self.cfg.thinker.audio_bos_id);
            ids.extend(std::iter::repeat(self.cfg.thinker.audio_token_id).take(n));
            ids.push(self.cfg.thinker.audio_eos_id);
            ids.extend(tokenizer.encode(&format!(
                "{text_after}<|im_end|>\n<|im_start|>assistant\n"
            )));
            ids
        };
        let pos4d: Vec<u32> = (0..ids.len() as u32).flat_map(|p| [p, p, p, 0]).collect();
        let n_ids = ids.len();
        Ok((ids, audio_tokens, pos4d, n_ids))
    }
}

/// Парсер HF tokenizer.json (BPE + byte_fallback + added_tokens) —
/// Qwen2.5-Omni. Vocab: строки с «Ġ» (U+0120) и байт-токены «<0xXX>»;
/// претокенизация в нашем Tokenizer tiktoken-стиля (пробел→Ġ, \n→Ċ) —
/// для словаря Qwen совпадает с эталонным разбиением (проверено на Qwen3).
pub fn tokenizer_from_json(dir: &Path) -> Result<crate::tokenizer::Tokenizer, BackendError> {
    let raw = std::fs::read_to_string(dir.join("tokenizer.json"))
        .map_err(|e| BackendError::Shape(format!("tokenizer.json: {e}")))?;
    let v: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| BackendError::Shape(format!("tokenizer.json: {e}")))?;
    let model = get(&v, &["model"]).ok_or_else(|| BackendError::Shape("no model".into()))?;

    let norm = |s: &str| -> Vec<u8> {
        // HF byte_fallback: «<0xXX>» → сырой байт
        let b = s.as_bytes();
        if b.len() == 6 && b[0] == b'<' && b[1] == b'0' && b[2] == b'x' && b[5] == b'>' {
            if let Ok(n) = u8::from_str_radix(&s[3..5], 16) {
                return vec![n];
            }
        }
        s.as_bytes().to_vec()
    };

    let mut vocab: Vec<(Vec<u8>, u32)> = model["vocab"]
        .as_object()
        .ok_or_else(|| BackendError::Shape("vocab not object".into()))?
        .iter()
        .map(|(k, idv)| (norm(k), idv.as_u64().unwrap_or(0) as u32))
        .collect();
    vocab.sort_by_key(|(_, id)| *id);
    let pieces: Vec<Vec<u8>> = vocab.into_iter().map(|(p, _)| p).collect();

    let mut id_of: std::collections::HashMap<Vec<u8>, u32> = std::collections::HashMap::new();
    for (i, p) in pieces.iter().enumerate() {
        id_of.insert(p.clone(), i as u32);
    }

    let mut merges = Vec::new();
    if let Some(arr) = model["merges"].as_array() {
        for m in arr {
            // два формата: строка "a b" ИЛИ массив [a, b]
            let (l, r) = if let Some(s) = m.as_str() {
                match s.split_once(' ') {
                    Some(pair) => pair,
                    None => {
                        return Err(BackendError::Shape(format!("bad merge {s}")));
                    }
                }
            } else if let Some(pair) = m.as_array() {
                let l = pair
                    .first()
                    .and_then(|x| x.as_str())
                    .ok_or_else(|| BackendError::Shape("merge left not string".into()))?;
                let r = pair
                    .get(1)
                    .and_then(|x| x.as_str())
                    .ok_or_else(|| BackendError::Shape("merge right not string".into()))?;
                (l, r)
            } else {
                return Err(BackendError::Shape("merge not string/array".into()));
            };
            let li = *id_of
                .get(&norm(l))
                .ok_or_else(|| BackendError::Shape(format!("merge left {l} missing")))?;
            let ri = *id_of
                .get(&norm(r))
                .ok_or_else(|| BackendError::Shape(format!("merge right {r} missing")))?;
            let mut merged = norm(l);
            merged.extend_from_slice(&norm(r));
            let mi = *id_of
                .get(&merged)
                .ok_or_else(|| BackendError::Shape(format!("merge result {l}+{r} missing")))?;
            merges.push(((li, ri), mi));
        }
    }

    let mut tok = crate::tokenizer::Tokenizer::from_parts(pieces, merges, None, None, None);
    // спец-токены (added_tokens special=true) — едиными кусками до BPE
    if let Some(arr) = v["added_tokens"].as_array() {
        for t in arr {
            if t["special"].as_bool().unwrap_or(false) {
                let id = t["id"].as_u64().unwrap_or(0) as u32;
                let content = t["content"].as_str().unwrap_or("");
                tok.add_special(norm(content), id);
            }
        }
    }
    Ok(tok)
}

#[cfg(all(test, feature = "cpu", feature = "metal"))]
mod tests {
    use super::*;
    use crate::cpu::CpuBackend;
    use crate::metal::MetalBackend;

    fn rng_u64(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        }
    }

    fn rand_vec(n: usize, rng: &mut impl FnMut() -> u64) -> Vec<f32> {
        (0..n)
            .map(|_| ((rng() >> 32) as f32 / (1u64 << 32) as f32) * 2.0 - 1.0)
            .collect()
    }

    fn mini_weights(cfg: &OmniAudioConfig) -> F32Tensors {
        let mut rng = rng_u64(42);
        let mut t = F32Tensors::new();
        let mut put = |name: &str, shape: Vec<usize>| {
            let n = shape.iter().product();
            t.insert(
                format!("thinker.audio_tower.{name}"),
                (rand_vec(n, &mut rng), shape),
            );
        };
        put("conv1.weight", vec![cfg.d_model, cfg.n_mel, 3]);
        put("conv1.bias", vec![cfg.d_model]);
        put("conv2.weight", vec![cfg.d_model, cfg.d_model, 3]);
        put("conv2.bias", vec![cfg.d_model]);
        for i in 0..cfg.n_layers {
            let p = format!("layers.{i}.");
            for (proj, bias) in [
                ("self_attn.q_proj", true),
                ("self_attn.k_proj", false),
                ("self_attn.v_proj", true),
                ("self_attn.out_proj", true),
            ] {
                put(&format!("{p}{proj}.weight"), vec![cfg.d_model, cfg.d_model]);
                if bias {
                    put(&format!("{p}{proj}.bias"), vec![cfg.d_model]);
                }
            }
            put(&format!("{p}self_attn_layer_norm.weight"), vec![cfg.d_model]);
            put(&format!("{p}self_attn_layer_norm.bias"), vec![cfg.d_model]);
            put(&format!("{p}fc1.weight"), vec![cfg.ffn_dim, cfg.d_model]);
            put(&format!("{p}fc1.bias"), vec![cfg.ffn_dim]);
            put(&format!("{p}fc2.weight"), vec![cfg.d_model, cfg.ffn_dim]);
            put(&format!("{p}fc2.bias"), vec![cfg.d_model]);
            put(&format!("{p}final_layer_norm.weight"), vec![cfg.d_model]);
            put(&format!("{p}final_layer_norm.bias"), vec![cfg.d_model]);
        }
        put("ln_post.weight", vec![cfg.d_model]);
        put("ln_post.bias", vec![cfg.d_model]);
        put("proj.weight", vec![cfg.output_dim, cfg.d_model]);
        put("proj.bias", vec![cfg.output_dim]);
        t
    }

    /// Metal-энкодер численно совпадает с CPU-референсом (тот же trait).
    #[test]
    fn tower_metal_matches_cpu() {
        let cfg = OmniAudioConfig {
            d_model: 16,
            n_layers: 2,
            n_heads: 2,
            ffn_dim: 32,
            n_mel: 8,
            output_dim: 24,
            n_window: 10,
            max_source_positions: 64,
        };
        let t = mini_weights(&cfg);


        let mb = MetalBackend::new().expect("metal backend");
        let m_tower = OmniAudioTower::load(&mb, &t, &cfg).unwrap();
        let cpu = CpuBackend;
        let c_tower = OmniAudioTower::load(&cpu, &t, &cfg).unwrap();


        // синтетический mel: 100 кадров (log-mel диапазон ~[-1, 1])
        let mut rng = rng_u64(7);
        let mel: Vec<f32> = rand_vec(100 * 8, &mut rng);

        // одно окно (20 кадров) — локализация расхождения
        let win: Vec<f32> = mel[..20 * 8].to_vec();
        let m_x = mb.from_cpu_f32(&win, vec![1, 20, 8]).unwrap();
        let c_x = cpu.from_cpu_f32(&win, vec![1, 20, 8]).unwrap();
        let m_w = m_tower.forward_window(&mb, &m_x).unwrap();
        let c_w = c_tower.forward_window(&cpu, &c_x).unwrap();
        let mv = mb.to_cpu_f32(&m_w).unwrap();
        let cv = cpu.to_cpu_f32(&c_w).unwrap();
        let mut max_delta = 0.0f32;
        for (a, b) in mv.iter().zip(&cv) {
            max_delta = max_delta.max((a - b).abs());
        }
        assert!(max_delta < 1e-3, "forward_window max delta {max_delta}");

        // полный forward (5 окон)
        let mel_t = cpu.from_cpu_f32(&mel, vec![1, 100, 8]).unwrap();
        let m_out = m_tower
            .forward(&mb, &mb.from_cpu_f32(&mel, vec![1, 100, 8]).unwrap())
            .unwrap();
        let c_out = c_tower.forward(&cpu, &mel_t).unwrap();
        assert_eq!(m_out.shape(), c_out.shape());
        let mv = mb.to_cpu_f32(&m_out).unwrap();
        let cv = cpu.to_cpu_f32(&c_out).unwrap();
        let mut max_delta = 0.0f32;
        for (a, b) in mv.iter().zip(&cv) {
            max_delta = max_delta.max((a - b).abs());
        }
        assert!(max_delta < 1e-3, "tower max delta {max_delta}");
    }

    /// attn_ar с mask=None (полное внимание) и hd=64 — аудио-путь.
    #[test]
    fn attn_ar_nomask_hd64_matches_cpu() {
        let mut rng = rng_u64(11);
        let tq = 10;
        let hd = 8;
        let heads = 2;
        let d = heads * hd;
        let q: Vec<f32> = rand_vec(1 * tq * d, &mut rng);
        let k: Vec<f32> = rand_vec(1 * tq * d, &mut rng);
        let v: Vec<f32> = rand_vec(1 * tq * d, &mut rng);

        let mb = MetalBackend::new().expect("metal backend");
        let cpu = CpuBackend;
        let m_q = mb.from_cpu_f32(&q, vec![1, tq, d]).unwrap();
        let m_k = mb.from_cpu_f32(&k, vec![1, tq, d]).unwrap();
        let m_v = mb.from_cpu_f32(&v, vec![1, tq, d]).unwrap();
        let m_y = mb.zeros(vec![1, tq, d]).unwrap();
        mb.attn_ar(&m_q, &m_k, &m_v, None, &m_y, heads, heads, hd).unwrap();

        let c_q = cpu.from_cpu_f32(&q, vec![1, tq, d]).unwrap();
        let c_k = cpu.from_cpu_f32(&k, vec![1, tq, d]).unwrap();
        let c_v = cpu.from_cpu_f32(&v, vec![1, tq, d]).unwrap();
        let c_y = cpu.zeros(vec![1, tq, d]).unwrap();
        cpu.attn_ar(&c_q, &c_k, &c_v, None, &c_y, heads, heads, hd).unwrap();

        let mv = mb.to_cpu_f32(&m_y).unwrap();
        let cv = cpu.to_cpu_f32(&c_y).unwrap();
        let mut max_delta = 0.0f32;
        for (a, b) in mv.iter().zip(&cv) {
            max_delta = max_delta.max((a - b).abs());
        }
        assert!(max_delta < 1e-4, "attn_ar nomask hd64 max delta {max_delta}");
    }

    /// layernorm (w/b/eps) — метал vs cpu.
    #[test]
    fn layernorm_matches_cpu() {
        let mut rng = rng_u64(13);
        let d = 16;
        let x: Vec<f32> = rand_vec(10 * d, &mut rng);
        let w: Vec<f32> = rand_vec(d, &mut rng);
        let b: Vec<f32> = rand_vec(d, &mut rng);

        let mb = MetalBackend::new().expect("metal backend");
        let cpu = CpuBackend;
        let m_x = mb.from_cpu_f32(&x, vec![10, d]).unwrap();
        let m_w = mb.from_cpu_f32(&w, vec![d]).unwrap();
        let m_b = mb.from_cpu_f32(&b, vec![d]).unwrap();
        let m_y = mb.zeros(vec![10, d]).unwrap();
        mb.layernorm(&m_x, &m_w, &m_b, &m_y, 1e-5).unwrap();

        let c_x = cpu.from_cpu_f32(&x, vec![10, d]).unwrap();
        let c_w = cpu.from_cpu_f32(&w, vec![d]).unwrap();
        let c_b = cpu.from_cpu_f32(&b, vec![d]).unwrap();
        let c_y = cpu.zeros(vec![10, d]).unwrap();
        cpu.layernorm(&c_x, &c_w, &c_b, &c_y, 1e-5).unwrap();

        let mv = mb.to_cpu_f32(&m_y).unwrap();
        let cv = cpu.to_cpu_f32(&c_y).unwrap();
        let mut max_delta = 0.0f32;
        for (a, b) in mv.iter().zip(&cv) {
            max_delta = max_delta.max((a - b).abs());
        }
        assert!(max_delta < 1e-4, "layernorm max delta {max_delta}");
    }

    /// Linear с bias (gemm_nt + add_bias) — метал vs cpu.
    #[test]
    fn linear_bias_matches_cpu() {
        let mut rng = rng_u64(17);
        let (rows, d, out) = (10, 16, 32);
        let x: Vec<f32> = rand_vec(rows * d, &mut rng);
        let w: Vec<f32> = rand_vec(out * d, &mut rng);
        let b: Vec<f32> = rand_vec(out, &mut rng);

        let mb = MetalBackend::new().expect("metal backend");
        let cpu = CpuBackend;
        let m_lin = crate::nn::Linear::from_weights(&mb, w.clone(), vec![out, d], Some(b.clone())).unwrap();
        let c_lin = crate::nn::Linear::from_weights(&cpu, w, vec![out, d], Some(b)).unwrap();
        let m_x = mb.from_cpu_f32(&x, vec![1, rows, d]).unwrap();
        let c_x = cpu.from_cpu_f32(&x, vec![1, rows, d]).unwrap();
        let m_y = m_lin.forward(&mb, &m_x).unwrap();
        let c_y = c_lin.forward(&cpu, &c_x).unwrap();
        let mv = mb.to_cpu_f32(&m_y).unwrap();
        let cv = cpu.to_cpu_f32(&c_y).unwrap();
        let mut max_delta = 0.0f32;
        for (a, b) in mv.iter().zip(&cv) {
            max_delta = max_delta.max((a - b).abs());
        }
        assert!(max_delta < 1e-3, "linear+bias max delta {max_delta}");
    }

    /// conv1d с РЕАЛЬНЫМИ весами (Qwen2.5-Omni conv1: [1280,128,3]) и
    /// реальным mel-входом (vassista_live.wav) — метал vs CPU.
    #[test]
    fn conv1d_real_weights_matches_cpu() {
        let dir = std::path::Path::new("/Users/dmitriygerasimov/.vassista-models/omni3b");
        if !dir.join("config.json").exists() {
            eprintln!("skip: omni3b не скачан");
            return;
        }
        let cfg = config_from_json(dir).unwrap();
        let mut all: F32Tensors = std::collections::HashMap::new();
        for i in 1..=3 {
            let p = dir.join(format!("model-{i:05}-of-00003.safetensors"));
            let t = read_safetensors_f32(&p).unwrap();
            all.extend(t);
        }
        let mb = MetalBackend::new().expect("metal backend");
        let cpu = CpuBackend;
        let m_tower = OmniAudioTower::load(&mb, &all, &cfg.audio).unwrap();
        let c_tower = OmniAudioTower::load(&cpu, &all, &cfg.audio).unwrap();

        let wav = audio::read_wav_pcm16(std::path::Path::new(
            "/Users/dmitriygerasimov/work/ai/vassista/crates/vs-wake/tests/fixtures/vassista_live.wav",
        ))
        .unwrap()
        .0;
        let (mel, t) = audio::log_mel_spectrogram(&wav);
        let n = t.min(20);
        let mut tm = vec![0.0f32; n * 128];
        for i in 0..n {
            for m in 0..128 {
                tm[i * 128 + m] = mel[m * t + i];
            }
        }
        let m_x = mb.from_cpu_f32(&tm, vec![1, n, 128]).unwrap();
        let c_x = cpu.from_cpu_f32(&tm, vec![1, n, 128]).unwrap();
        let cwv = mb.to_cpu_f32(&m_tower.conv1_w).unwrap();
        eprintln!("conv1d real: w[1,5,:3]={:?}", &cwv[(1 * 128 + 5) * 3..(1 * 128 + 5) * 3 + 3]);
        eprintln!("conv1d real: w[0,0,:3]={:?}", &cwv[..3]);


        let lnw = mb.to_cpu_f32(&m_tower.layers[0].ffn_ln_w).unwrap();
        let lnb = mb.to_cpu_f32(&m_tower.layers[0].ffn_ln_b).unwrap();
        eprintln!("conv1d real: ffn_ln w[:4]={:?} b[:4]={:?}", &lnw[..4], &lnb[..4]);
        let q1v = mb.to_cpu_f32(&m_tower.layers[1].attn.q.w).unwrap();
        eprintln!("conv1d real: layer1 q[0,:4]={:?}", &q1v[..4.min(q1v.len())]);
        // layernorm 3D [1, 20, 1280] метал vs CPU
        let ln_x: Vec<f32> = (0..20 * 1280).map(|i| ((i * 7919) % 1000) as f32 / 1000.0 - 0.5).collect();
        let ln_w: Vec<f32> = (0..1280).map(|i| ((i * 104729) % 100) as f32 / 1000.0).collect();
        let ln_b: Vec<f32> = (0..1280).map(|i| (((i as u64 * 15485863) % 100) as f32) / 10000.0).collect();
        let m_lnx = mb.from_cpu_f32(&ln_x, vec![1, 20, 1280]).unwrap();
        let m_lnw = mb.from_cpu_f32(&ln_w, vec![1280]).unwrap();
        let m_lnb = mb.from_cpu_f32(&ln_b, vec![1280]).unwrap();
        let m_lny = mb.zeros(vec![1, 20, 1280]).unwrap();
        mb.layernorm(&m_lnx, &m_lnw, &m_lnb, &m_lny, 1e-5).unwrap();
        let c_lnx = cpu.from_cpu_f32(&ln_x, vec![1, 20, 1280]).unwrap();
        let c_lnw = cpu.from_cpu_f32(&ln_w, vec![1280]).unwrap();
        let c_lnb = cpu.from_cpu_f32(&ln_b, vec![1280]).unwrap();
        let c_lny = cpu.zeros(vec![1, 20, 1280]).unwrap();
        cpu.layernorm(&c_lnx, &c_lnw, &c_lnb, &c_lny, 1e-5).unwrap();
        let mlv = mb.to_cpu_f32(&m_lny).unwrap();
        let clv = cpu.to_cpu_f32(&c_lny).unwrap();
        let mut ln_delta = 0.0f32;
        for (a, b) in mlv.iter().zip(&clv) {
            ln_delta = ln_delta.max((a - b).abs());
        }
        eprintln!("conv1d real: layernorm3d delta={ln_delta} m[0,:4]={:?} c[0,:4]={:?}",
            &mlv[..4], &clv[..4]);
        let fc1v = mb.to_cpu_f32(&m_tower.layers[0].fc1.w).unwrap();
        eprintln!("conv1d real: fc1[0,:4]={:?}", &fc1v[..4.min(fc1v.len())]);
        let m_h1 = mb.zeros(vec![1, n, 1280]).unwrap();
        let c_h1 = cpu.zeros(vec![1, n, 1280]).unwrap();
        mb.conv1d(&m_x, &m_tower.conv1_w, Some(&m_tower.conv1_b), 1, 1, 1, 1, &m_h1).unwrap();
        cpu.conv1d(&c_x, &c_tower.conv1_w, Some(&c_tower.conv1_b), 1, 1, 1, 1, &c_h1).unwrap();
        let m_out = mb.to_cpu_f32(&m_h1).unwrap();
        eprintln!("conv1d real: out[0,:4]={:?}", &m_out[..4.min(m_out.len())]);
        // полный forward_window (20 кадров, 2 слоя) vs python-эталон
        let fw = m_tower.forward_window(&mb, &m_x).unwrap();
        let fwv = mb.to_cpu_f32(&fw).unwrap();
        eprintln!("conv1d real: fw[0,:4]={:?}", &fwv[..4.min(fwv.len())]);
        let cfw = c_tower.forward_window(&cpu, &c_x).unwrap();
        let cfwv = cpu.to_cpu_f32(&cfw).unwrap();
        let mut fw_delta = 0.0f32;
        for (a, b) in fwv.iter().zip(&cfwv) {
            fw_delta = fw_delta.max((a - b).abs());
        }
        eprintln!("conv1d real: fw delta={fw_delta} cfw[0,:4]={:?}", &cfwv[..4.min(cfwv.len())]);
        let mv = mb.to_cpu_f32(&m_h1).unwrap();
        let cv = cpu.to_cpu_f32(&c_h1).unwrap();
        let mut max_delta = 0.0f32;
        let mut worst = 0usize;
        for (i, (a, b)) in mv.iter().zip(&cv).enumerate() {
            let d = (a - b).abs();
            if d > max_delta {
                max_delta = d;
                worst = i;
            }
        }
        eprintln!("conv1d real: delta {max_delta} at {worst} (m {} c {})", mv[worst], cv[worst]);
        assert!(max_delta < 1e-3, "conv1d real max delta {max_delta}");
    }

    /// Локализация расхождения: conv1d напрямую (metal vs cpu), тот же формат.
    #[test]
    fn conv1d_metal_matches_cpu() {
        let mut rng = rng_u64(3);
        let x: Vec<f32> = rand_vec(1 * 20 * 8, &mut rng);
        let w: Vec<f32> = rand_vec(16 * 8 * 3, &mut rng);
        let b: Vec<f32> = rand_vec(16, &mut rng);

        let mb = MetalBackend::new().expect("metal backend");
        let cpu = CpuBackend;
        let m_x = mb.from_cpu_f32(&x, vec![1, 20, 8]).unwrap();
        let m_w = mb.from_cpu_f32(&w, vec![16, 8, 3]).unwrap();
        let m_b = mb.from_cpu_f32(&b, vec![16]).unwrap();
        let m_y = mb.zeros(vec![1, 20, 16]).unwrap();
        mb.conv1d(&m_x, &m_w, Some(&m_b), 1, 1, 1, 1, &m_y).unwrap();

        let c_x = cpu.from_cpu_f32(&x, vec![1, 20, 8]).unwrap();
        let c_w = cpu.from_cpu_f32(&w, vec![16, 8, 3]).unwrap();
        let c_b = cpu.from_cpu_f32(&b, vec![16]).unwrap();
        let c_y = cpu.zeros(vec![1, 20, 16]).unwrap();
        cpu.conv1d(&c_x, &c_w, Some(&c_b), 1, 1, 1, 1, &c_y).unwrap();

        let mv = mb.to_cpu_f32(&m_y).unwrap();
        let cv = cpu.to_cpu_f32(&c_y).unwrap();
        let mut max_delta = 0.0f32;
        for (a, b) in mv.iter().zip(&cv) {
            max_delta = max_delta.max((a - b).abs());
        }
        assert!(max_delta < 1e-4, "conv1d max delta {max_delta}");
    }
}
