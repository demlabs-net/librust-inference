//! Ф2.6: нативный MTP (NextN) — третий драфтер каркаса спекулятивного
//! декодинга, для гибридов `qwen35` с `nextn_predict_layers = 1`.
//!
//! Эталон — llama.cpp 5266f24da75d: `src/models/qwen35.cpp` (`graph_mtp`)
//! и `common/speculative.cpp` (`common_speculative_impl_draft_mtp`).
//!
//! NextN-блок лежит в `blk.{n_layers}.*` (сразу за трейком) и состоит из
//! адаптера (`nextn.{eh_proj,enorm,hnorm,shared_head_norm}`) и одного
//! полновнимательного декодер-блока (attn_q с гейтом на голову, q/k-norm,
//! IMROPE, SwiGLU). Пара на позиции q — `(t_q, h_{q−1})`, где h — скрытое
//! трейка ПОСЛЕ `output_norm` (эталонный `process()` сдвигает строки h
//! вправо: `embd[k] = h_tgt[k−1]`, первая строка — `pending_h` прошлого
//! батча). Выход блока после `shared_head_norm` — это `t_h_nextn`: он же
//! лезит в голову (тэйден `model.output`), он же становится h следующего
//! шага цепочки.
//!
//! Траектория драфта (qwen35 — одна обученная голова, без chain_heads):
//!   seed  (id_last, pending_h) на позиции committed;
//!   шаг i (t̂_i, h'_i) на позиции committed + i + 1, h' — выход блока.
//! После верификации таргетом MTP-KV пересобирается ПРИНЯТЫМИ строками
//! окна с h трейка (replay в [`MtpDrafter::after_extract`]) — черновые
//! строки поверх принятой границы отбрасываются (в эталоне — seq_rm от
//! чекпойнта; здесь — truncate + перезапись).

use crate::backend::{Backend, Tensor};
use crate::error::BackendError;
use crate::nn::{KvCache, Linear, SwiGLU};
use crate::qwen3::Qwen3Model;
use crate::qwen35::{FullAttn, HIDDEN_LAYER, Qwen35Config, Qwen35Model};
use crate::spec::{DraftSampling, Drafter};
use crate::weights_map::WeightsMap;

/// Нативный MTP-драфтер: nextn-блок + собственный KV-кэш + состояние
/// границы (`pending_h`, `committed`), зеркалящее драйвер эталона.
pub struct MtpDrafter {
    cfg: Qwen35Config,
    /// Таблица эмбеддингов трейка в родном формате GGUF (эталон:
    /// `model.tok_embd`, nextn.embed_tokens в наших файлах нет).
    embed: Tensor,
    /// LM-хед трейка (эталон: `model.output`, shared_head_head нет).
    output: Tensor,
    attn: FullAttn,
    ffn: SwiGLU,
    attn_norm: Tensor,
    post_norm: Tensor,
    enorm: Tensor,
    hnorm: Tensor,
    shnorm: Tensor,
    /// [dim, 2·dim] — GGUF ne = [2·dim, dim] (ne0 = вход).
    eh_proj: Tensor,
    kv: KvCache,
    /// Число токенов, для которых KV nextn уже пересобран ПРИНЯТЫМ
    /// префиксом (= model.context_len() на момент последнего after_extract).
    committed: usize,
    /// h трейка (после out_norm) на последней принятой позиции — вторая
    /// половина пары seed-строки; None до первого after_extract.
    pending_h: Option<Tensor>,
    /// Окно последнего draft(): [anchor, черновики…] — replay в
    /// after_extract берёт из него токены принятых строк.
    window: Vec<u32>,
    /// Строки h префилла, переигрывание которых отложено до первого
    /// draft(): после префилла токены промпта есть только в history,
    /// который драфтер видит лишь в draft(). (pos = ids.len(),
    /// n_accepted = ids.len() — весь промпт принят.)
    pending_rows: Option<Tensor>,
    sampling: DraftSampling,
    rng: u64,
}

/// Есть ли у файла nextn-веса (blk.{n_layers}.nextn.eh_proj.weight).
pub fn has_mtp(w: &WeightsMap<'_>, cfg: &Qwen35Config) -> bool {
    w.get(&format!("blk.{}.nextn.eh_proj.weight", cfg.n_layers))
        .is_some()
}

impl MtpDrafter {
    /// Загрузка nextn-блока из тех же весов, что и таргет. `embed`/`output`
    /// — тензоры таргета ([`Qwen35Model::head_tables`]) — драфтер делит их
    /// с моделью, как эталон (nextn.embed_tokens/shared_head_head в этих
    /// чекпойнтах нет → `model.tok_embd`/`model.output`).
    pub fn load(
        backend: &dyn Backend,
        w: &WeightsMap<'_>,
        cfg: &Qwen35Config,
        embed: Tensor,
        output: Tensor,
    ) -> Result<Self, BackendError> {
        let il = cfg.n_layers;
        let p = |x: &str| format!("blk.{il}.{x}");
        if w.get(&p("nextn.eh_proj.weight")).is_none() {
            return Err(BackendError::Shape(format!(
                "MTP: в blk.{il} нет nextn.* — у модели нативного MTP нет"
            )));
        }
        let f32v = |name: &str| -> Result<Vec<f32>, BackendError> {
            let t = w
                .get(name)
                .ok_or_else(|| BackendError::Shape(format!("missing tensor {name}")))?;
            Ok(crate::weights_map::dequant_f32(t))
        };
        let hd = cfg.head_dim;
        let q_out = 2 * cfg.n_heads * hd;
        let kv_out = cfg.n_kv_heads * hd;
        // Внимательная часть — те же разделённые проекции в родных квантах,
        // что у полновнимательных слоёв трейка (q на голову [q | gate],
        // затем k, v). Здесь q/k — Q4_K, v — Q5_K: склейка в один GEMM
        // заставила бы деквантовать всё в F32 (Ф8.2g).
        let attn = FullAttn {
            wq: Qwen3Model::upload_q(backend, w, &p("attn_q.weight"), vec![q_out, cfg.dim])?,
            wk: Qwen3Model::upload_q(backend, w, &p("attn_k.weight"), vec![kv_out, cfg.dim])?,
            wv: Qwen3Model::upload_q(backend, w, &p("attn_v.weight"), vec![kv_out, cfg.dim])?,
            wo: Qwen3Model::upload_q(
                backend,
                w,
                &p("attn_output.weight"),
                vec![cfg.dim, cfg.n_heads * hd],
            )?,
            q_norm: backend.from_cpu_f32(&f32v(&p("attn_q_norm.weight"))?, vec![hd])?,
            k_norm: backend.from_cpu_f32(&f32v(&p("attn_k_norm.weight"))?, vec![hd])?,
        };
        let ffn = SwiGLU {
            w13: Qwen3Model::upload_fused(
                backend,
                w,
                &[&p("ffn_gate.weight"), &p("ffn_up.weight")],
                vec![2 * cfg.ffn_dim, cfg.dim],
            )?,
            w2: Qwen3Model::upload_q(
                backend,
                w,
                &p("ffn_down.weight"),
                vec![cfg.dim, cfg.ffn_dim],
            )?,
        };
        let d = cfg.dim;
        let nextn_norm = |name: &str| -> Result<Tensor, BackendError> {
            backend.from_cpu_f32(&f32v(&p(name))?, vec![d])
        };
        // shared_head_norm обязателен: fallback на output_norm (эталонная
        // тернарная ветка) не реализован — в наших файлах он есть.
        let shnorm = nextn_norm("nextn.shared_head_norm.weight")?;
        let fmt = if std::env::var("INFER_KV_F32").is_ok() {
            crate::dtype::KvFormat::F32
        } else {
            crate::dtype::KvFormat::F16
        };
        let mut kv = KvCache::new(
            backend,
            cfg.n_kv_heads,
            hd,
            fmt,
            fmt,
            cfg.rope_theta,
            Some(cfg.rope_sections),
        )?;
        kv.set_max_cap(Some(cfg.max_seq));
        Ok(Self {
            cfg: cfg.clone(),
            embed,
            output,
            attn,
            ffn,
            attn_norm: nextn_norm("attn_norm.weight")?,
            post_norm: nextn_norm("post_attention_norm.weight")?,
            enorm: nextn_norm("nextn.enorm.weight")?,
            hnorm: nextn_norm("nextn.hnorm.weight")?,
            shnorm,
            eh_proj: Qwen3Model::upload_q(backend, w, &p("nextn.eh_proj.weight"), vec![d, 2 * d])?,
            kv,
            committed: 0,
            pending_h: None,
            window: Vec::new(),
            pending_rows: None,
            sampling: DraftSampling::Greedy,
            rng: 0x85ebca6b,
        })
    }

    /// Одна строка nextn-блока: пара (h_prev [1,1,d], token) на позиции
    /// `pos` → (логиты [1,1,V] или None, h_out [1,1,d] после
    /// shared_head_norm = t_h_nextn). KV-кэш блока дописывается на pos.
    fn step(
        &mut self,
        backend: &dyn Backend,
        h: &Tensor,
        token: u32,
        pos: usize,
        want_logits: bool,
    ) -> Result<(Option<Tensor>, Tensor), BackendError> {
        if self.kv.pos != pos {
            return Err(BackendError::Shape(format!(
                "MTP: KV на {}, строка на {pos}",
                self.kv.pos
            )));
        }
        let d = self.cfg.dim;
        let eps = self.cfg.norm_eps;
        // e_norm(emb(t_q)) | h_norm(h_{q−1}) — порядок конкатенации эталонный
        // (ggml_concat(e_norm, h_norm, dim=0): эмбеддинг ПЕРВЫМ).
        let e = backend.zeros(vec![1, 1, d])?;
        backend.embedding_quant(&[token], &self.embed, &e)?;
        let e_n = backend.zeros(vec![1, 1, d])?;
        backend.rmsnorm(&e, &self.enorm, &e_n, eps)?;
        let h_n = backend.zeros(vec![1, 1, d])?;
        backend.rmsnorm(h, &self.hnorm, &h_n, eps)?;
        let cat = backend.zeros(vec![1, 2 * d])?;
        let cat = backend.copy_rows_into(&e_n.view(vec![1, d]), 1, d, cat, 0, 0, 2 * d)?;
        let cat = backend.copy_rows_into(&h_n.view(vec![1, d]), 1, d, cat, 0, d, 2 * d)?;
        let x = Linear {
            w: self.eh_proj.clone(),
            b: None,
        }
        .forward(backend, &cat.view(vec![1, 1, 2 * d]))?;
        let x = x.view(vec![1, 1, d]);
        // Декодер-блок — тот же путь, что у полновнимательного слоя трейка
        // (attn_forward: q|gate, q/k-norm, IMROPE, sigmoid-gate, wo).
        let an = backend.zeros(vec![1, 1, d])?;
        backend.rmsnorm(&x, &self.attn_norm, &an, eps)?;
        let m = Qwen35Model::attn_forward(
            &self.cfg,
            backend,
            &an,
            &self.attn,
            &mut self.kv,
            pos,
            None,
        )?;
        let x1 = backend.zeros(vec![1, 1, d])?;
        backend.add(&x, &m, &x1)?;
        let pn = backend.zeros(vec![1, 1, d])?;
        backend.rmsnorm(&x1, &self.post_norm, &pn, eps)?;
        let f = crate::qwen3::ffn_forward(backend, &pn, &self.ffn)?;
        let x2 = backend.zeros(vec![1, 1, d])?;
        backend.add(&x1, &f, &x2)?;
        let h_out = backend.zeros(vec![1, 1, d])?;
        backend.rmsnorm(&x2, &self.shnorm, &h_out, eps)?;
        let logits = if want_logits {
            let lg = backend.zeros(vec![1, self.cfg.vocab_size])?;
            backend.gemm_nt(&h_out.view(vec![1, d]), &self.output, &lg)?;
            Some(lg.view(vec![1, 1, self.cfg.vocab_size]))
        } else {
            None
        };
        Ok((logits, h_out))
    }

    /// Выбор черновика из строки логитов [1,1,V] + честная q.
    fn pick(&mut self, backend: &dyn Backend, logits: &Tensor) -> Result<(u32, f32), BackendError> {
        let (argmx, temp, top_k) = match self.sampling {
            DraftSampling::Greedy => (true, 1.0, 0),
            DraftSampling::Sample { temperature, top_k } => (false, temperature.max(1e-6), top_k),
        };
        if argmx {
            let (ids, _lse) = backend.argmax_rows(logits, None)?;
            return Ok((ids[0], 1.0));
        }
        let v = self.cfg.vocab_size;
        let row = backend.to_cpu_f32(logits)?;
        let id = self.sample_row(&row[..v], temp, top_k);
        // Скалярная q честна только как дельта (1.0) — sample_row усещает
        // top-k, а softmax_row_prob_t считал бы вероятность полного softmax
        // (over-acceptance, ревью Ф2.9).
        Ok((id, 1.0))
    }

    /// Сэмпл из softmax(row/temperature) с топ-K (LCG, как у DFlash).
    fn sample_row(&mut self, row: &[f32], temperature: f32, top_k: usize) -> u32 {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut w: Vec<f32> = row
            .iter()
            .map(|&l| ((l - max) / temperature).exp())
            .collect();
        if top_k > 0 && top_k < w.len() {
            let mut idx: Vec<usize> = (0..w.len()).collect();
            idx.sort_by(|&a, &b| w[b].partial_cmp(&w[a]).unwrap_or(std::cmp::Ordering::Equal));
            for &i in &idx[top_k..] {
                w[i] = 0.0;
            }
        }
        let total: f32 = w.iter().sum();
        self.rng = crate::inference::sampling::lcg_next(self.rng);
        let mut r = crate::inference::sampling::lcg_unit_f32(self.rng) * total;
        for (i, v) in w.iter_mut().enumerate() {
            r -= *v;
            if r <= 0.0 {
                return i as u32;
            }
        }
        (w.len() - 1) as u32
    }

    /// Число пересобранных принятым префиксом позиций nextn-KV.
    pub fn committed(&self) -> usize {
        self.committed
    }
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    use crate::cpu::CpuBackend;

    fn rng(seed: u64) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s as f32 / u32::MAX as f32) - 0.5
        }
    }

    fn rand(n: usize, r: &mut impl FnMut() -> f32) -> Vec<f32> {
        (0..n).map(|_| r()).collect()
    }

    fn close(a: &[f32], b: &[f32], tol: f32, what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: len {} != {}", a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x - y).abs() <= tol * (1.0 + x.abs().max(y.abs())),
                "{what}[{i}]: {x} != {y}"
            );
        }
    }

    fn rms(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
        let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        x.iter().zip(w).map(|(v, g)| v * inv * g).collect()
    }

    /// w — [out, in] строками (движковая конвенция Linear/gemm_nt).
    fn matvec(w: &[f32], out: usize, inp: usize, x: &[f32]) -> Vec<f32> {
        (0..out)
            .map(|o| (0..inp).map(|i| w[o * inp + i] * x[i]).sum())
            .collect()
    }

    /// IMROPE-референс (тот же, что qwen35::tests::imrope_partial), t=1.
    fn imrope(
        x: &mut [f32],
        heads: usize,
        hd: usize,
        n_rot: usize,
        pos: usize,
        sections: [usize; 4],
        theta: f32,
    ) {
        let half = n_rot / 2;
        for h in 0..heads {
            for i in 0..half {
                let sector = i % 4;
                let rot = (sector.is_multiple_of(3) && sector < 3 * sections[0])
                    || (sector % 3 == 1 && sector < 3 * sections[1])
                    || (sector % 3 == 2 && sector < 3 * sections[2]);
                if !rot {
                    continue;
                }
                let th = pos as f32 * theta.powf(-2.0 * i as f32 / n_rot as f32);
                let b = h * hd;
                let (x0, x1) = (x[b + i], x[b + i + half]);
                x[b + i] = x0 * th.cos() - x1 * th.sin();
                x[b + i + half] = x0 * th.sin() + x1 * th.cos();
            }
        }
    }

    struct Fix {
        cfg: Qwen35Config,
        embed: Vec<f32>,
        output: Vec<f32>,
        wqkv: Vec<f32>,
        wo: Vec<f32>,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
        w13: Vec<f32>,
        w2: Vec<f32>,
        attn_norm: Vec<f32>,
        post_norm: Vec<f32>,
        enorm: Vec<f32>,
        hnorm: Vec<f32>,
        shnorm: Vec<f32>,
        eh_proj: Vec<f32>,
    }

    fn fixture(seed: u64) -> Fix {
        let (dim, nh, nkv, hd, ffn, vocab) = (8usize, 2usize, 1usize, 4usize, 16usize, 7usize);
        let mut r = rng(seed);
        let cfg = Qwen35Config {
            arch: "qwen35".into(),
            dim,
            n_layers: 2,
            n_heads: nh,
            n_kv_heads: nkv,
            head_dim: hd,
            ffn_dim: ffn,
            vocab_size: vocab,
            rope_theta: 10_000.0,
            n_rot: 4,
            rope_sections: [1, 1, 0, 0],
            norm_eps: 1e-6,
            max_seq: 64,
            conv_kernel: 4,
            ssm_state: 8,
            ssm_k_heads: 1,
            ssm_v_heads: 1,
            ssm_inner: 8,
            recurrent: vec![true, false],
        };
        Fix {
            cfg,
            embed: rand(vocab * dim, &mut r),
            output: rand(vocab * dim, &mut r),
            wqkv: rand((2 * nh * hd + 2 * nkv * hd) * dim, &mut r),
            wo: rand(dim * nh * hd, &mut r),
            q_norm: rand(hd, &mut r),
            k_norm: rand(hd, &mut r),
            w13: rand(2 * ffn * dim, &mut r),
            w2: rand(dim * ffn, &mut r),
            attn_norm: rand(dim, &mut r),
            post_norm: rand(dim, &mut r),
            enorm: rand(dim, &mut r),
            hnorm: rand(dim, &mut r),
            shnorm: rand(dim, &mut r),
            eh_proj: rand(dim * 2 * dim, &mut r),
        }
    }

    fn drafter(be: &dyn Backend, f: &Fix) -> MtpDrafter {
        let d = f.cfg.dim;
        let t = |v: &[f32], sh: Vec<usize>| be.from_cpu_f32(v, sh).unwrap();
        let kv_out = f.cfg.n_kv_heads * f.cfg.head_dim;
        let q_out = 2 * f.cfg.n_heads * f.cfg.head_dim;
        let mut kv = KvCache::new(
            be,
            f.cfg.n_kv_heads,
            f.cfg.head_dim,
            crate::dtype::KvFormat::F32,
            crate::dtype::KvFormat::F32,
            f.cfg.rope_theta,
            Some(f.cfg.rope_sections),
        )
        .unwrap();
        kv.set_max_cap(Some(f.cfg.max_seq));
        MtpDrafter {
            cfg: f.cfg.clone(),
            embed: t(&f.embed, vec![f.cfg.vocab_size, d]),
            output: t(&f.output, vec![f.cfg.vocab_size, d]),
            attn: FullAttn {
                wq: t(&f.wqkv[..q_out * d], vec![q_out, d]),
                wk: t(&f.wqkv[q_out * d..(q_out + kv_out) * d], vec![kv_out, d]),
                wv: t(&f.wqkv[(q_out + kv_out) * d..], vec![kv_out, d]),
                wo: t(&f.wo, vec![d, f.cfg.n_heads * f.cfg.head_dim]),
                q_norm: t(&f.q_norm, vec![f.cfg.head_dim]),
                k_norm: t(&f.k_norm, vec![f.cfg.head_dim]),
            },
            ffn: SwiGLU {
                w13: t(&f.w13, vec![2 * f.cfg.ffn_dim, d]),
                w2: t(&f.w2, vec![d, f.cfg.ffn_dim]),
            },
            attn_norm: t(&f.attn_norm, vec![d]),
            post_norm: t(&f.post_norm, vec![d]),
            enorm: t(&f.enorm, vec![d]),
            hnorm: t(&f.hnorm, vec![d]),
            shnorm: t(&f.shnorm, vec![d]),
            eh_proj: t(&f.eh_proj, vec![d, 2 * d]),
            kv,
            committed: 0,
            pending_h: None,
            pending_rows: None,
            window: Vec::new(),
            sampling: DraftSampling::Greedy,
            rng: 1,
        }
    }

    /// Наивный f32-референс graph_mtp (qwen35.cpp:539-640) для ОДНОЙ строки;
    /// k/v-история — строки предыдущих шагов (k уже повёрнутые).
    #[allow(clippy::too_many_arguments)]
    fn ref_step(
        f: &Fix,
        h_in: &[f32],
        token: u32,
        pos: usize,
        hist_k: &[Vec<f32>],
        hist_v: &[Vec<f32>],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let c = &f.cfg;
        let (d, nh, nkv, hd) = (c.dim, c.n_heads, c.n_kv_heads, c.head_dim);
        let eps = c.norm_eps;
        let e = &f.embed[token as usize * d..][..d];
        let e_n = rms(e, &f.enorm, eps);
        let h_n = rms(h_in, &f.hnorm, eps);
        let cat: Vec<f32> = e_n.iter().chain(h_n.iter()).copied().collect();
        let x = matvec(&f.eh_proj, d, 2 * d, &cat);
        let an = rms(&x, &f.attn_norm, eps);
        let qkv = matvec(&f.wqkv, 2 * nh * hd + 2 * nkv * hd, d, &an);
        let qg = &qkv[..2 * nh * hd];
        let krow: Vec<f32> = qkv[2 * nh * hd..2 * nh * hd + nkv * hd].to_vec();
        let vrow: Vec<f32> = qkv[2 * nh * hd + nkv * hd..].to_vec();
        // q/k per-head rmsnorm
        let mut q = vec![0.0f32; nh * hd];
        for hh in 0..nh {
            let r = rms(&qg[hh * 2 * hd..hh * 2 * hd + hd], &f.q_norm, eps);
            q[hh * hd..][..hd].copy_from_slice(&r);
        }
        let mut k = vec![0.0f32; nkv * hd];
        for hh in 0..nkv {
            let r = rms(&krow[hh * hd..hh * hd + hd], &f.k_norm, eps);
            k[hh * hd..][..hd].copy_from_slice(&r);
        }
        imrope(&mut q, nh, hd, c.n_rot, pos, c.rope_sections, c.rope_theta);
        imrope(&mut k, nkv, hd, c.n_rot, pos, c.rope_sections, c.rope_theta);
        // внимание по истории + текущей строке
        let ks: Vec<&Vec<f32>> = hist_k.iter().chain(std::iter::once(&k)).collect();
        let vs: Vec<&Vec<f32>> = hist_v.iter().chain(std::iter::once(&vrow)).collect();
        let mut gated = vec![0.0f32; nh * hd];
        for hh in 0..nh {
            let kh = hh / (nh / nkv);
            let qr = &q[hh * hd..][..hd];
            let sc: Vec<f32> = ks
                .iter()
                .map(|kk| {
                    (0..hd).map(|i| qr[i] * kk[kh * hd + i]).sum::<f32>() / (hd as f32).sqrt()
                })
                .collect();
            let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let ex: Vec<f32> = sc.iter().map(|s| (s - mx).exp()).collect();
            let sum: f32 = ex.iter().sum();
            for i in 0..hd {
                let o: f32 = (0..ks.len())
                    .map(|j| ex[j] / sum * vs[j][kh * hd + i])
                    .sum();
                let g = qg[hh * 2 * hd + hd + i];
                gated[hh * hd + i] = o / (1.0 + (-g).exp());
            }
        }
        let m = matvec(&f.wo, d, nh * hd, &gated);
        let x1: Vec<f32> = x.iter().zip(&m).map(|(a, b)| a + b).collect();
        let pn = rms(&x1, &f.post_norm, eps);
        let gu = matvec(&f.w13, 2 * c.ffn_dim, d, &pn);
        let act: Vec<f32> = (0..c.ffn_dim)
            .map(|i| {
                let g = gu[i];
                let u = gu[c.ffn_dim + i];
                g / (1.0 + (-g).exp()) * u
            })
            .collect();
        let ff = matvec(&f.w2, d, c.ffn_dim, &act);
        let x2: Vec<f32> = x1.iter().zip(&ff).map(|(a, b)| a + b).collect();
        let h_out = rms(&x2, &f.shnorm, eps);
        let logits = matvec(&f.output, c.vocab_size, d, &h_out);
        (h_out, logits, k, vrow)
    }

    #[test]
    fn mtp_step_matches_reference() {
        let be = CpuBackend;
        let f = fixture(0xA11CE);
        let mut dr = drafter(&be, &f);
        let d = f.cfg.dim;
        let h0 = vec![0.1f32; d];
        let mut hist_k: Vec<Vec<f32>> = Vec::new();
        let mut hist_v: Vec<Vec<f32>> = Vec::new();
        let mut h = h0.clone();
        for (step, &tok) in [3u32, 5].iter().enumerate() {
            let pos = step;
            let (lg, h_out) = dr
                .step(
                    &be,
                    &be.from_cpu_f32(&h, vec![1, 1, d]).unwrap(),
                    tok,
                    pos,
                    true,
                )
                .unwrap();
            let (rh, rl, rk, rv) = ref_step(&f, &h, tok, pos, &hist_k, &hist_v);
            hist_k.push(rk);
            hist_v.push(rv);
            let hv = be.to_cpu_f32(&h_out).unwrap();
            close(&hv, &rh, 2e-4, &format!("step {step} h_out"));
            let lv = be.to_cpu_f32(&lg.unwrap()).unwrap();
            close(&lv, &rl, 2e-4, &format!("step {step} logits"));
            h = rh;
        }
        // KV-кэш блока обязан совпасть с историей референса (k — повёрнутый).
        let kv = be.to_cpu_f32(&dr.kv.view()).unwrap();
        let nkv = f.cfg.n_kv_heads * f.cfg.head_dim;
        for (j, row) in hist_k.iter().enumerate() {
            close(&kv[j * nkv..][..nkv], row, 2e-4, &format!("kv.k row {j}"));
        }
    }

    /// Replay обязан восстановить ровно тот же KV, что нативная цепочка
    /// draft-шагов (те же пары (t_q, h_{q−1})). Путь — отложенный префилл:
    /// after_extract с pos == rows откладывает строки, первый draft()
    /// переигрывает их по history.
    #[test]
    fn mtp_replay_rebuilds_chain_kv() {
        let be = CpuBackend::default();
        let f = fixture(0xBEEF);
        let d = f.cfg.dim;
        // seed-строка последовательности: pending_h до первого префилла —
        // нули (эталон: pending_h инициализирован нулями), replay q=0 — тоже
        // нули.
        let h0 = vec![0.0f32; d];
        let toks = [1u32, 4, 2];

        // 1) цепочка: три шага, h следующего = h_out предыдущего.
        let mut chain = drafter(&be, &f);
        chain.pending_h = Some(be.from_cpu_f32(&h0, vec![1, 1, d]).unwrap());
        let mut h_rows: Vec<f32> = Vec::new();
        let mut h = chain.pending_h.clone().unwrap();
        for (i, &tok) in toks.iter().enumerate() {
            let (_, h_out) = chain.step(&be, &h, tok, i, false).unwrap();
            let hv = be.to_cpu_f32(&h_out).unwrap();
            h_rows.extend_from_slice(&hv);
            h = h_out;
        }
        let chain_kv = be.to_cpu_f32(&chain.kv.view()).unwrap();
        let chain_v = be.to_cpu_f32(&chain.kv.view_v()).unwrap();

        // 2) отложенный replay: after_extract (префилл) + draft с history.
        let mut rep = drafter(&be, &f);
        rep.pending_h = Some(be.from_cpu_f32(&h0, vec![1, 1, d]).unwrap());
        let extract = be.from_cpu_f32(&h_rows, vec![toks.len(), d]).unwrap();
        rep.after_extract(&be, &[extract.clone()], toks.len(), toks.len())
            .unwrap();
        assert!(rep.pending_rows.is_some(), "префилл отложен до draft()");
        // history = промпт + anchor; anchor — любой (черновиков gamma=0).
        let mut history = toks.to_vec();
        history.push(0);
        let (t, p) = rep.draft(&be, &history, 0);
        assert!(t.is_empty() && p.is_empty(), "gamma=0 — пустые черновики");
        assert_eq!(rep.committed(), toks.len());
        let kv = be.to_cpu_f32(&rep.kv.view()).unwrap();
        let v = be.to_cpu_f32(&rep.kv.view_v()).unwrap();
        close(&kv, &chain_kv, 1e-6, "replay k");
        close(&v, &chain_v, 1e-6, "replay v");

        // 3) повторный префилл (повторная попытка сервиса) при непустом
        //    окне прошлой последовательности — обязан сбросить границу.
        rep.window = vec![9, 9];
        rep.after_extract(&be, &[extract.clone()], toks.len(), toks.len())
            .unwrap();
        assert_eq!(rep.committed(), 0);
        assert!(rep.pending_rows.is_some());
    }

    /// Границы draft(): рассинхрон history/committed и отсутствие pending_h
    /// обязаны давать пустые черновики (не панику).
    #[test]
    fn mtp_draft_guards() {
        let be = CpuBackend;
        let f = fixture(0xC0FFEE);
        let mut dr = drafter(&be, &f);
        assert_eq!(dr.draft(&be, &[1, 2, 3], 4).0.len(), 0, "нет pending_h");
        dr.pending_h = Some(be.zeros(vec![1, 1, f.cfg.dim]).unwrap());
        dr.committed = 9;
        assert_eq!(
            dr.draft(&be, &[1, 2, 3], 4).0.len(),
            0,
            "history != committed+1"
        );
    }
}

impl Drafter for MtpDrafter {
    fn set_sampling(&mut self, mode: DraftSampling) {
        self.sampling = mode;
    }

    fn reset(&mut self) {
        self.kv.reset();
        self.committed = 0;
        self.pending_h = None;
        self.pending_rows = None;
        self.window.clear();
        // Сид черновой LCG — тоже часть состояния последовательности:
        // без сброса два запроса с одним seed давали разные черновики в
        // зависимости от того, сколько генераций жил движок (воспроизводимость
        // спек-против-спека, ревью Ф2.9).
        self.rng = 0x85ebca6b;
    }

    fn target_layers(&self) -> &[usize] {
        &[HIDDEN_LAYER]
    }

    fn draft(
        &mut self,
        backend: &dyn Backend,
        history: &[u32],
        gamma: usize,
    ) -> (Vec<u32>, Vec<f32>) {
        let Some(anchor) = history.last().copied() else {
            return (Vec::new(), Vec::new());
        };
        // Отложенное переигрывание префилла: строки h промпта лежат в
        // pending_rows, токены — в начале history. Длину проверяем ДО take():
        // забирая строки и выходя по короткой history, префилл терялся
        // безвозвратно (драфтер мёртв до конца генерации, ревью Ф2.9).
        if self.committed == 0 && self.pending_rows.is_some() {
            let t = self.pending_rows.as_ref().unwrap().shape()[0];
            if history.len() < t + 1 {
                return (Vec::new(), Vec::new());
            }
            let rows = self.pending_rows.take().unwrap();
            let d = self.cfg.dim;
            let r = (|| -> Result<(), BackendError> {
                for q in 0..t {
                    let h_prev = if q == 0 {
                        backend.zeros(vec![1, 1, d])?
                    } else {
                        rows.view_offset(vec![1, 1, d], (q - 1) * d)
                    };
                    self.step(backend, &h_prev, history[q], q, false)?;
                }
                Ok(())
            })();
            if let Err(e) = r {
                eprintln!("MtpDrafter: replay префилла: {e}");
                return (Vec::new(), Vec::new());
            }
            self.committed = t;
        }
        // history = принятые + anchor (anchor ещё не в модели): ровно на 1
        // больше границы. Расхождение = драфтер рассинхронизирован с таргетом
        // — черновиков не даём. Драйвер теперь ведёт каждый токен через
        // after_extract (в т.ч. при пустом черновике), так что рассинхрон
        // возможен только после ошибки инжекта — фиксируем причину в логе.
        if gamma == 0 {
            return (Vec::new(), Vec::new());
        }
        if history.len() != self.committed + 1 {
            // eprintln, а не tracing::warn: в примерах/CLI подписчика нет —
            // рассинхрон границы молча терялся, и «drafter предлагает 1-2
            // черновика вместо γ» был неотличим от нормальной работы.
            eprintln!(
                "MtpDrafter: рассинхрон границы: history={} committed={} — черновиков нет",
                history.len(),
                self.committed
            );
            return (Vec::new(), Vec::new());
        }
        let Some(mut h) = self.pending_h.clone() else {
            return (Vec::new(), Vec::new());
        };
        // Хвост черновой цепочки прошлого окна (если after_extract не
        // добрался до него) — дописывать будем с границы committed.
        self.kv.truncate(self.committed);
        let mut tokens = Vec::with_capacity(gamma);
        let mut probs = Vec::with_capacity(gamma);
        let mut tok = anchor;
        for i in 0..gamma {
            let step = self.step(backend, &h, tok, self.committed + i, true);
            let (lg, h_out) = match step {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("MtpDrafter: шаг {i}: {e}");
                    // Что уже насэмплили — валидные черновики (их KV в кэше).
                    break;
                }
            };
            let Ok((id, q)) = self.pick(backend, lg.as_ref().unwrap()) else {
                break;
            };
            tokens.push(id);
            probs.push(q);
            h = h_out;
            tok = id;
        }
        self.window = Vec::with_capacity(tokens.len() + 1);
        self.window.push(anchor);
        self.window.extend_from_slice(&tokens);
        if std::env::var("LLM_DBG_MTP").is_ok() {
            eprintln!("MTP draft: {} черновиков", tokens.len());
        }
        (tokens, probs)
    }

    fn after_extract(
        &mut self,
        backend: &dyn Backend,
        extract: &[Tensor],
        pos: usize,
        n_accepted: usize,
    ) -> Result<(), BackendError> {
        if extract.is_empty() || n_accepted == 0 {
            return Ok(());
        }
        let h = &extract[0];
        let d = self.cfg.dim;
        // decode_spec отдаёт [1, rows, d], prefill_extract_last — [rows, d].
        let rows = if h.shape().len() == 3 {
            h.shape()[1]
        } else {
            h.shape()[0]
        };
        let n = n_accepted.min(rows);
        let dbg_ac = std::env::var("LLM_DBG_MTP").is_ok();
        if dbg_ac {
            eprintln!(
                "MTP after_extract: pos={pos} n_accepted={n_accepted} rows={rows} window={} committed={} -> n={}",
                self.window.len(),
                self.committed,
                n.min(self.window.len().max(1))
            );
        }
        // Префилл узнаётся по pos == n_accepted: префилл-вызов идёт с
        // pos = n = длина промпта (позиции [0..n)), а окно верификации —
        // pos = base + n_accepted при base ≥ 1, т.е. pos > n_accepted
        // ВСЕГДА. Прежний признак pos == rows (rows — строки extract,
        // 1+γ в окне) ложно срабатывал на коротких окнах отказа:
        // base + n_ok == 1+γ (маленький промпт + все черновики отвергнуты)
        // — nextn-KV сбрасывался посреди генерации (ревью Ф2.9).
        // (Признак по пустому window не годится: повторная попытка после
        // вырождения в сервисе — префилл при непустом окне прошлой
        // последовательности.)
        if pos == n_accepted {
            self.kv.reset();
            self.committed = 0;
            self.window.clear();
            // Токены промпта драфтер увидит только в первом draft() (history)
            // — строки h откладываем туда.
            self.pending_rows = Some(h.view(vec![rows, d]));
            self.pending_h = Some(h.view_offset(vec![1, 1, d], (rows - 1) * d));
            return Ok(());
        }
        let n = n.min(self.window.len());
        if n == 0 {
            return Ok(());
        }
        let base = pos.saturating_sub(n);
        // Черновые строки поверх принятой границы — долой; принятые строки
        // окна пересобираются с h ТРЕЙКА (не цепочки) — инвариант KV
        // «строка q = (t_q, h_{q−1})» восстанавливается.
        self.kv.truncate(base);
        for i in 0..n {
            let h_prev = if i == 0 {
                match &self.pending_h {
                    Some(p) => p.clone(),
                    None => backend.zeros(vec![1, 1, d])?,
                }
            } else {
                h.view_offset(vec![1, 1, d], (i - 1) * d)
            };
            self.step(backend, &h_prev, self.window[i], base + i, false)?;
        }
        self.pending_h = Some(h.view_offset(vec![1, 1, d], (n - 1) * d));
        self.committed = base + n;
        Ok(())
    }
}
