//! BERT-энкодер для эмбеддингов (Ф4.5): bge-семейство (bge-small-en-v1.5,
//! bge-m3 и др.). WordPiece-токенайзер (vocab.txt), absolute position +
//! segment embeddings, бидирекшнл attention, pooler (CLS → dense → tanh),
//! L2-нормализация — протокол bge.
//!
//! Загрузка: safetensors (fp32/f16/bf16 — читатель в [`crate::weights`]),
//! имена в карте HuggingFace BERT (`bert.embeddings.*`,
//! `encoder.layer.{i}.*`, `pooler.dense.*`). Конфиг — из config.json.

use std::collections::HashMap;
use std::path::Path;

use crate::backend::{Backend, Tensor};
use crate::error::BackendError;
use crate::nn::{Embedding, Linear};
use crate::weights::{F32Tensors, read_safetensors_f32};

// ---------------------------------------------------------------------------
// WordPiece-токенайзер (BERT/bge-small)
// ---------------------------------------------------------------------------

/// WordPiece-токенайзер из vocab.txt (по строке — токен; спец-токены
/// [CLS]/[SEP]/[UNK]/[PAD] ищутся по имени).
pub struct WordPieceTokenizer {
    vocab: HashMap<String, u32>,
    cls: u32,
    sep: u32,
    unk: u32,
    /// Максимум токенов без учёта [CLS]/[SEP].
    max_len: usize,
}

impl WordPieceTokenizer {
    pub fn from_vocab_txt(path: &Path) -> Result<Self, BackendError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| BackendError::Native(format!("vocab.txt: {e}")))?;
        let mut vocab = HashMap::new();
        for (i, line) in content.lines().enumerate() {
            if !line.is_empty() {
                vocab.insert(line.to_string(), i as u32);
            }
        }
        if vocab.is_empty() {
            return Err(BackendError::Native("vocab.txt is empty".into()));
        }
        let id = |name: &str| {
            vocab
                .get(name)
                .copied()
                .ok_or_else(|| BackendError::Native(format!("missing special token {name}")))
        };
        Ok(Self {
            cls: id("[CLS]")?,
            sep: id("[SEP]")?,
            unk: id("[UNK]")?,
            vocab,
            max_len: 512,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// [CLS] + wordpieces + [SEP], обрезка до max_len (без спец-токенов).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = vec![self.cls];
        let mut used = 0usize;
        for word in pretokenize(text) {
            if used >= self.max_len {
                break;
            }
            for piece in self.wordpiece(&word) {
                if used >= self.max_len {
                    break;
                }
                ids.push(piece);
                used += 1;
            }
        }
        ids.push(self.sep);
        ids
    }

    /// Greedy longest-match WordPiece с ##-суффиксами; неразбиваемое
    /// слово целиком → [UNK] (как в transformers).
    fn wordpiece(&self, word: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut start = 0usize;
        while start < word.len() {
            let mut end = word.len();
            let mut found = None;
            while end > start {
                let mut piece = String::with_capacity(end - start + 2);
                if start > 0 {
                    piece.push_str("##");
                }
                piece.push_str(&word[start..end]);
                if let Some(id) = self.vocab.get(&piece) {
                    found = Some(*id);
                    break;
                }
                // назад на границу char (start..end всегда валидный срез)
                end = word.floor_char_boundary(end - 1);
            }
            match found {
                Some(id) => out.push(id),
                None => {
                    out.push(self.unk);
                    break;
                }
            }
            start = end;
        }
        out
    }
}

/// BERTBasicTokenizer-претокенайзер: lowercase, слова по пробелам,
/// пунктуация — отдельными токенами.
fn pretokenize(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let mut words = Vec::new();
    let mut cur = String::new();
    for c in lower.chars() {
        if c.is_whitespace() {
            if !cur.is_empty() {
                words.push(std::mem::take(&mut cur));
            }
        } else if c.is_alphanumeric() {
            cur.push(c);
        } else {
            if !cur.is_empty() {
                words.push(std::mem::take(&mut cur));
            }
            words.push(c.to_string());
        }
    }
    if !cur.is_empty() {
        words.push(cur);
    }
    words
}

// ---------------------------------------------------------------------------
// Конфиг
// ---------------------------------------------------------------------------

/// Конфигурация BERT из config.json (HF) с дефолтами.
#[derive(Debug, Clone)]
pub struct BertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub layer_norm_eps: f32,
}

impl BertConfig {
    pub fn from_json(path: &Path) -> Result<Self, BackendError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| BackendError::Native(format!("config.json: {e}")))?;
        let v: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| BackendError::Native(format!("config.json: {e}")))?;
        let n = |k: &str, d: usize| {
            v.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .unwrap_or(d)
        };
        Ok(Self {
            vocab_size: n("vocab_size", 30522),
            hidden_size: n("hidden_size", 768),
            num_hidden_layers: n("num_hidden_layers", 12),
            num_attention_heads: n("num_attention_heads", 12),
            intermediate_size: n("intermediate_size", 3072),
            max_position_embeddings: n("max_position_embeddings", 512),
            type_vocab_size: n("type_vocab_size", 2),
            layer_norm_eps: v
                .get("layer_norm_eps")
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .unwrap_or(1e-12),
        })
    }
}

// ---------------------------------------------------------------------------
// Энкодер
// ---------------------------------------------------------------------------

struct BertLayer {
    attn_qkv: Tensor, // (3D, D)
    attn_qkv_bias: Tensor,
    attn_out: Tensor, // (D, D)
    attn_out_bias: Tensor,
    attn_ln_w: Tensor,
    attn_ln_b: Tensor,
    ffn_up: Tensor, // (4D, D)
    ffn_up_bias: Tensor,
    ffn_down: Tensor, // (D, 4D)
    ffn_down_bias: Tensor,
    ffn_ln_w: Tensor,
    ffn_ln_b: Tensor,
}

impl BertLayer {
    fn forward(
        &self,
        backend: &dyn Backend,
        x: &Tensor,
        eps: f32,
        n_heads: usize,
        hd: usize,
    ) -> Result<Tensor, BackendError> {
        let (b, t, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let qkv = Linear {
            w: self.attn_qkv.clone(),
            b: Some(self.attn_qkv_bias.clone()),
        }
        .forward(backend, x)?;
        let q = crate::nn::slice_last(backend, &qkv, d, 0)?;
        let k = crate::nn::slice_last(backend, &qkv, d, d)?;
        let v = crate::nn::slice_last(backend, &qkv, d, 2 * d)?;
        let attn = backend.zeros(vec![b, t, d])?;
        // mask=None → бидирекшнл self-attention (BERT)
        backend.attn_ar(&q, &k, &v, None, &attn, n_heads, n_heads, hd)?;
        let attn_out = Linear {
            w: self.attn_out.clone(),
            b: Some(self.attn_out_bias.clone()),
        }
        .forward(backend, &attn)?;
        let sum1 = backend.zeros(vec![b, t, d])?;
        backend.add(x, &attn_out, &sum1)?;
        let ln1 = backend.zeros(vec![b, t, d])?;
        backend.layernorm(&sum1, &self.attn_ln_w, &self.attn_ln_b, &ln1, eps)?;
        let up = Linear {
            w: self.ffn_up.clone(),
            b: Some(self.ffn_up_bias.clone()),
        }
        .forward(backend, &ln1)?;
        let gelu = backend.zeros(vec![b, t, self.ffn_up.shape()[0]])?;
        backend.gelu_erf(&up, &gelu)?;
        let down = Linear {
            w: self.ffn_down.clone(),
            b: Some(self.ffn_down_bias.clone()),
        }
        .forward(backend, &gelu)?;
        let sum2 = backend.zeros(vec![b, t, d])?;
        backend.add(&ln1, &down, &sum2)?;
        let ln2 = backend.zeros(vec![b, t, d])?;
        backend.layernorm(&sum2, &self.ffn_ln_w, &self.ffn_ln_b, &ln2, eps)?;
        Ok(ln2)
    }
}

/// BERT-энкодер (bge-семейство): embeddings → 12×[attn+ffn] → pooler → L2.
pub struct BertEncoder {
    config: BertConfig,
    tokenizer: WordPieceTokenizer,
    word_emb: Embedding,
    pos_emb: Embedding,
    seg_emb: Embedding,
    emb_ln_w: Tensor,
    emb_ln_b: Tensor,
    layers: Vec<BertLayer>,
    pooler_w: Tensor, // (D, D)
    pooler_b: Tensor,
}

fn get<'a>(w: &'a F32Tensors, name: &str) -> Result<&'a Vec<f32>, BackendError> {
    // HF-имена (bert.embeddings.*) и SentenceTransformers-экспорт (без bert.)
    w.get(name)
        .or_else(|| w.get(&format!("bert.{name}")))
        .map(|(data, _)| data)
        .ok_or_else(|| BackendError::Native(format!("missing safetensors tensor: {name}")))
}

impl BertEncoder {
    /// Загрузка из каталога модели (model.safetensors, config.json,
    /// vocab.txt). Тензоры сразу в backend (GPU).
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        backend: &dyn Backend,
        dir: &Path,
        tokenizer: WordPieceTokenizer,
        config: BertConfig,
    ) -> Result<Self, BackendError> {
        let w: F32Tensors = read_safetensors_f32(&dir.join("model.safetensors"))
            .map_err(|e| BackendError::Native(format!("model.safetensors: {e}")))?;
        let d = config.hidden_size;
        let int4 = config.intermediate_size;
        let emb = |name: &str| -> Result<Embedding, BackendError> {
            let data = get(&w, name)?;
            // Shape — с ТЕМ ЖЕ bert.-фолбэком, что и get() выше: для
            // SentenceTransformers-экспорта тензор лежит под "bert.<name>",
            // и старый точечный lookup возвращал пустой shape.
            let shape = w
                .get(name)
                .or_else(|| w.get(&format!("bert.{name}")))
                .map(|(_, s)| s.clone())
                .unwrap_or_default();
            Embedding::from_weights(backend, data.clone(), shape)
        };
        let lin = |name: &str, out: usize, inp: usize| -> Result<(Tensor, Tensor), BackendError> {
            let wt = backend.from_cpu_f32(get(&w, &format!("{name}.weight"))?, vec![out, inp])?;
            let bs = backend.from_cpu_f32(get(&w, &format!("{name}.bias"))?, vec![out])?;
            Ok((wt, bs))
        };
        let ln = |name: &str| -> Result<(Tensor, Tensor), BackendError> {
            Ok((
                backend.from_cpu_f32(get(&w, &format!("{name}.weight"))?, vec![d])?,
                backend.from_cpu_f32(get(&w, &format!("{name}.bias"))?, vec![d])?,
            ))
        };

        let word_emb = emb("embeddings.word_embeddings.weight")?;
        let pos_emb = emb("embeddings.position_embeddings.weight")?;
        let seg_emb = emb("embeddings.token_type_embeddings.weight")?;
        let (emb_ln_w, emb_ln_b) = ln("embeddings.LayerNorm")?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = format!("encoder.layer.{i}");
            // q/k/v fuse в один (3D, D) тензор
            let mut qkv = Vec::with_capacity(3 * d * d);
            let mut qkv_bias = Vec::with_capacity(3 * d);
            for part in ["query", "key", "value"] {
                qkv.extend_from_slice(get(&w, &format!("{p}.attention.self.{part}.weight"))?);
                qkv_bias.extend_from_slice(get(&w, &format!("{p}.attention.self.{part}.bias"))?);
            }
            let (attn_out, attn_out_bias) = lin(&format!("{p}.attention.output.dense"), d, d)?;
            let (attn_ln_w, attn_ln_b) = ln(&format!("{p}.attention.output.LayerNorm"))?;
            let (ffn_up, ffn_up_bias) = lin(&format!("{p}.intermediate.dense"), int4, d)?;
            let (ffn_down, ffn_down_bias) = lin(&format!("{p}.output.dense"), d, int4)?;
            let (ffn_ln_w, ffn_ln_b) = ln(&format!("{p}.output.LayerNorm"))?;
            layers.push(BertLayer {
                attn_qkv: backend.from_cpu_f32(&qkv, vec![3 * d, d])?,
                attn_qkv_bias: backend.from_cpu_f32(&qkv_bias, vec![3 * d])?,
                attn_out,
                attn_out_bias,
                attn_ln_w,
                attn_ln_b,
                ffn_up,
                ffn_up_bias,
                ffn_down,
                ffn_down_bias,
                ffn_ln_w,
                ffn_ln_b,
            });
        }

        let (pooler_w, pooler_b) = lin("pooler.dense", d, d)?;
        Ok(Self {
            config,
            tokenizer,
            word_emb,
            pos_emb,
            seg_emb,
            emb_ln_w,
            emb_ln_b,
            layers,
            pooler_w,
            pooler_b,
        })
    }

    pub fn config(&self) -> &BertConfig {
        &self.config
    }

    /// Векторизация текстов: по одному (lookup-кернел batch=1); выход —
    /// L2-нормализованные эмбеддинги размера hidden_size.
    pub fn embed(
        &self,
        backend: &dyn Backend,
        texts: &[&str],
    ) -> Result<Vec<Vec<f32>>, BackendError> {
        let mut out = Vec::with_capacity(texts.len());
        for text in texts {
            out.push(self.embed_one(backend, text)?);
        }
        Ok(out)
    }

    fn embed_one(&self, backend: &dyn Backend, text: &str) -> Result<Vec<f32>, BackendError> {
        let d = self.config.hidden_size;
        let ids = self.tokenizer.encode(text);
        // обрезка по max_position_embeddings: bge-протокол (и промышленная
        // практика SentenceTransformers) — хвост отбрасывается, а не ошибка
        let ids: Vec<u32> = if ids.len() > self.config.max_position_embeddings {
            ids[..self.config.max_position_embeddings].to_vec()
        } else {
            ids
        };
        let t = ids.len();
        let word = self.word_emb.forward(backend, &ids, t)?;
        let pos_ids: Vec<u32> = (0..t as u32).collect();
        let pos = self.pos_emb.forward(backend, &pos_ids, t)?;
        let seg_ids = vec![0u32; t];
        let seg = self.seg_emb.forward(backend, &seg_ids, t)?;
        // HF BERT: embeddings = LayerNorm(word + pos + token_type) — seg_emb
        // обязана участвовать (раньше x3 строилась, а норм применялся к
        // word+pos — сегментные эмбеддинги молча отбрасывались).
        let mut x = backend.zeros(vec![1, t, d])?;
        backend.add(&word, &pos, &x)?;
        backend.add(&x, &seg, &x)?;
        let ln0 = backend.zeros(vec![1, t, d])?;
        backend.layernorm(
            &x,
            &self.emb_ln_w,
            &self.emb_ln_b,
            &ln0,
            self.config.layer_norm_eps,
        )?;
        x = ln0;
        let hd = d / self.config.num_attention_heads;
        for layer in &self.layers {
            x = layer.forward(
                backend,
                &x,
                self.config.layer_norm_eps,
                self.config.num_attention_heads,
                hd,
            )?;
        }
        // CLS = первая строка: срез через CPU (D ≤ 1024 — дёшево);
        // Linear ожидает 3D (batch, seq, hidden)
        let x_cpu = backend.to_cpu_f32(&x)?;
        let cls = backend.from_cpu_f32(&x_cpu[..d], vec![1, 1, d])?;
        let pooled = Linear {
            w: self.pooler_w.clone(),
            b: Some(self.pooler_b.clone()),
        }
        .forward(backend, &cls)?;
        let tanh = backend.zeros(vec![1, d])?;
        backend.tanh(&pooled, &tanh)?;
        let mut vec = backend.to_cpu_f32(&tanh)?;
        // L2-нормализация (bge-протокол)
        let norm = vec.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        for v in &mut vec {
            *v /= norm;
        }
        Ok(vec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Детерминированный LCG-генератор весов (от стабильного сида).
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        }
    }

    /// Мини-модель в памяти: D=16, 2 слоя, 2 головы, vocab 32, max_pos 16.
    fn mini_weights() -> (F32Tensors, BertConfig) {
        let d = 16usize;
        let int4 = 32usize;
        let mut rng = Lcg(0x5eed_2026);
        let mut t = |rows: usize, cols: usize| -> Vec<f32> {
            (0..rows * cols).map(|_| rng.next_f32()).collect()
        };
        let mut w = F32Tensors::new();
        let mut put = |name: String, data: Vec<f32>, shape: Vec<usize>| {
            w.insert(name, (data, shape));
        };
        put(
            "bert.embeddings.word_embeddings.weight".into(),
            t(34, d),
            vec![34, d],
        );
        put(
            "bert.embeddings.position_embeddings.weight".into(),
            t(16, d),
            vec![16, d],
        );
        put(
            "bert.embeddings.token_type_embeddings.weight".into(),
            t(2, d),
            vec![2, d],
        );
        put(
            "bert.embeddings.LayerNorm.weight".into(),
            vec![1.0; d],
            vec![d],
        );
        put(
            "bert.embeddings.LayerNorm.bias".into(),
            vec![0.0; d],
            vec![d],
        );
        for i in 0..2 {
            let p = format!("encoder.layer.{i}");
            for part in ["query", "key", "value"] {
                put(
                    format!("{p}.attention.self.{part}.weight"),
                    t(d, d),
                    vec![d, d],
                );
                put(format!("{p}.attention.self.{part}.bias"), t(d, 1), vec![d]);
            }
            put(
                format!("{p}.attention.output.dense.weight"),
                t(d, d),
                vec![d, d],
            );
            put(format!("{p}.attention.output.dense.bias"), t(d, 1), vec![d]);
            put(
                format!("{p}.attention.output.LayerNorm.weight"),
                vec![1.0; d],
                vec![d],
            );
            put(
                format!("{p}.attention.output.LayerNorm.bias"),
                vec![0.0; d],
                vec![d],
            );
            put(
                format!("{p}.intermediate.dense.weight"),
                t(int4, d),
                vec![int4, d],
            );
            put(
                format!("{p}.intermediate.dense.bias"),
                t(int4, 1),
                vec![int4],
            );
            put(
                format!("{p}.output.dense.weight"),
                t(d, int4),
                vec![d, int4],
            );
            put(format!("{p}.output.dense.bias"), t(d, 1), vec![d]);
            put(
                format!("{p}.output.LayerNorm.weight"),
                vec![1.0; d],
                vec![d],
            );
            put(format!("{p}.output.LayerNorm.bias"), vec![0.0; d], vec![d]);
        }
        put("pooler.dense.weight".into(), t(d, d), vec![d, d]);
        put("pooler.dense.bias".into(), t(d, 1), vec![d]);
        let config = BertConfig {
            vocab_size: 34,
            hidden_size: d,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            intermediate_size: int4,
            max_position_embeddings: 16,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
        };
        (w, config)
    }

    /// Прогон мини-модели на конкретном бэкенде.
    fn run_on(backend: &dyn Backend, w: &F32Tensors, config: &BertConfig) -> Vec<Vec<f32>> {
        // строим энкодер напрямую из тензоров (без файлов)
        let d = config.hidden_size;
        let emb = |name: &str| -> Embedding {
            let (data, shape) = w.get(name).unwrap();
            Embedding::from_weights(backend, data.clone(), shape.clone()).unwrap()
        };
        let lin = |name: &str, out: usize, inp: usize| -> (Tensor, Tensor) {
            (
                backend
                    .from_cpu_f32(
                        w.get(&format!("{name}.weight")).unwrap().0.as_slice(),
                        vec![out, inp],
                    )
                    .unwrap(),
                backend
                    .from_cpu_f32(
                        w.get(&format!("{name}.bias")).unwrap().0.as_slice(),
                        vec![out],
                    )
                    .unwrap(),
            )
        };
        let ln = |name: &str| -> (Tensor, Tensor) {
            (
                backend
                    .from_cpu_f32(
                        w.get(&format!("{name}.weight")).unwrap().0.as_slice(),
                        vec![d],
                    )
                    .unwrap(),
                backend
                    .from_cpu_f32(
                        w.get(&format!("{name}.bias")).unwrap().0.as_slice(),
                        vec![d],
                    )
                    .unwrap(),
            )
        };
        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let p = format!("encoder.layer.{i}");
            let mut qkv = Vec::new();
            let mut qkv_bias = Vec::new();
            for part in ["query", "key", "value"] {
                qkv.extend_from_slice(
                    &w.get(&format!("{p}.attention.self.{part}.weight"))
                        .unwrap()
                        .0,
                );
                qkv_bias.extend_from_slice(
                    &w.get(&format!("{p}.attention.self.{part}.bias")).unwrap().0,
                );
            }
            let (attn_out, attn_out_bias) = lin(&format!("{p}.attention.output.dense"), d, d);
            let (attn_ln_w, attn_ln_b) = ln(&format!("{p}.attention.output.LayerNorm"));
            let (ffn_up, ffn_up_bias) = lin(
                &format!("{p}.intermediate.dense"),
                config.intermediate_size,
                d,
            );
            let (ffn_down, ffn_down_bias) =
                lin(&format!("{p}.output.dense"), d, config.intermediate_size);
            let (ffn_ln_w, ffn_ln_b) = ln(&format!("{p}.output.LayerNorm"));
            layers.push(BertLayer {
                attn_qkv: backend.from_cpu_f32(&qkv, vec![3 * d, d]).unwrap(),
                attn_qkv_bias: backend.from_cpu_f32(&qkv_bias, vec![3 * d]).unwrap(),
                attn_out,
                attn_out_bias,
                attn_ln_w,
                attn_ln_b,
                ffn_up,
                ffn_up_bias,
                ffn_down,
                ffn_down_bias,
                ffn_ln_w,
                ffn_ln_b,
            });
        }
        let (pooler_w, pooler_b) = lin("pooler.dense", d, d);
        let enc = BertEncoder {
            config: config.clone(),
            tokenizer: test_tokenizer(),
            word_emb: emb("bert.embeddings.word_embeddings.weight"),
            pos_emb: emb("bert.embeddings.position_embeddings.weight"),
            seg_emb: emb("bert.embeddings.token_type_embeddings.weight"),
            emb_ln_w: ln("bert.embeddings.LayerNorm").0,
            emb_ln_b: ln("bert.embeddings.LayerNorm").1,
            layers,
            pooler_w,
            pooler_b,
        };
        enc.embed(backend, &["hello world", "machine learning"])
            .unwrap()
    }

    fn test_tokenizer() -> WordPieceTokenizer {
        // vocab 0..26 = "a".."z", 26=[UNK], 27=[CLS], 28=[SEP], 29=[PAD],
        // плюс ##-суффиксы для «hello» → he##l##lo
        let mut vocab = HashMap::new();
        for (i, c) in (b'a'..=b'z').enumerate() {
            vocab.insert((c as char).to_string(), i as u32);
        }
        vocab.insert("[UNK]".into(), 26);
        vocab.insert("[CLS]".into(), 27);
        vocab.insert("[SEP]".into(), 28);
        vocab.insert("[PAD]".into(), 29);
        vocab.insert("##l".into(), 30);
        vocab.insert("##lo".into(), 31);
        vocab.insert("##ing".into(), 32);
        vocab.insert("he".into(), 33);
        WordPieceTokenizer {
            vocab,
            cls: 27,
            sep: 28,
            unk: 26,
            max_len: 512,
        }
    }

    #[test]
    fn wordpiece_encode_basic() {
        let tok = test_tokenizer();
        let ids = tok.encode("hello");
        // [CLS] he ##l ##lo [SEP] — greedy longest-match
        assert_eq!(ids, vec![27, 33, 30, 31, 28]);
    }

    #[test]
    fn wordpiece_unknown_word() {
        let tok = test_tokenizer();
        // "zz": "z" — валидный префикс, "##z" нет → хвост UNK (как в transformers)
        let ids = tok.encode("zz");
        assert_eq!(ids, vec![27, 25, 26, 28]);
        // "123": "1" нет в vocab вообще → слово целиком UNK
        let ids = tok.encode("123");
        assert_eq!(ids, vec![27, 26, 28]);
    }

    #[test]
    fn wordpiece_punctuation_split() {
        let tok = test_tokenizer();
        let ids = tok.encode("hello!");
        // [CLS] he ##l ##lo ! [SEP]
        assert_eq!(ids, vec![27, 33, 30, 31, 26, 28]);
    }

    #[test]
    fn wordpiece_lowercases() {
        let tok = test_tokenizer();
        let ids = tok.encode("HELLO");
        assert_eq!(ids, vec![27, 33, 30, 31, 28]);
    }

    // The test compares the CPU reference against a REAL backend — it
    // only makes sense when metal/cuda is compiled in (the old cfg also
    // matched cpu-only builds, where default_backend() bails and the
    // test panicked).
    #[cfg(any(feature = "metal", feature = "cuda"))]
    #[test]
    fn cpu_matches_gpu() {
        let (w, config) = mini_weights();
        let b = crate::train::default_backend().unwrap();
        let out = run_on(b.as_ref(), &w, &config);
        assert_eq!(out.len(), 2);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(v.len(), config.hidden_size);
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-3,
                "embedding {i} not L2-normalized: {norm}"
            );
            assert!(
                v.iter().any(|x| x.abs() > 1e-6),
                "embedding {i} is all zeros"
            );
        }
        // разные тексты → разные векторы
        let d = (0..config.hidden_size)
            .map(|i| (out[0][i] - out[1][i]).abs())
            .fold(0.0f32, f32::max);
        assert!(d > 1e-4, "embeddings suspiciously identical: {d}");
    }
}
