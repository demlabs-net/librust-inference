//! Qwen3.8-Flash-Next (`qwen4exp`) — конфиг и СХЕМА ФАЙЛА.
//!
//! Архитектура сильно отличается от qwen35 (27B): к продукту добавляются
//! 4-поточные **hyper-connections** (остаточный поток = 4 × dim), MoE на
//! 512 экспертов top-10 + shared-expert, разреженное внимание с **QSA-
//! индексером** (top-k 2048 по сжатым ключам), **PLE** — n-граммные
//! эмбеддинги (16 голов × ~20M словарей, отдельный шард) и MTP отдельным
//! файлом. Точные имена/размерности сняты с реального заголовка
//! `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (IQ2_XS, shard 1).
//!
//! Что здесь есть: разбор метаданных (`from_gguf`) и полный план тензоров
//! (`tensor_plan`) — по нему загрузчик и `qwen4exp_check` сверяют файл.
//! Чего ещё нет: forward (математика hyper-connections/QSA/PLE/MoE в
//! работе) — волна A5, следующий шаг; конфиг и схема — фундамент.
//!
//! Порядок размерностей в плане — GGUF (`ne`, ne0 первым, как печатает
//! `scripts/gguf_meta.py`). Загрузчики весов принимают обратный порядок
//! (`vec![out, in]`), как в qwen3/qwen35.

use crate::arch;
use crate::gguf::{GgufFile, GgufValue};

/// Архитектура в `general.architecture`.
pub const ARCH: &str = "qwen4exp";

/// Гипер-параметры модели из `qwen4exp.*` GGUF-метаданных.
#[derive(Debug, Clone)]
pub struct Qwen4ExpConfig {
    pub arch: String,
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// Частичная ротация (64 из 256) и mrope-секции [11,11,10,0].
    pub n_rot: usize,
    pub rope_sections: [usize; 4],
    pub rope_theta: f32,
    pub norm_eps: f32,
    pub max_seq: usize,
    pub vocab_size: usize,
    /// MoE: 512 экспертов, top-10, ff 640; shared-expert ff 640.
    pub n_experts: usize,
    pub n_experts_used: usize,
    pub expert_ff: usize,
    pub shared_ff: usize,
    /// GDN-миксер: conv 4, state 128, k-головы 16, v-головы 48, inner 6144.
    pub conv_kernel: usize,
    pub ssm_state: usize,
    pub ssm_k_heads: usize,
    pub ssm_v_heads: usize,
    pub ssm_inner: usize,
    /// Каждый `full_attention_interval`-й слой — полное внимание.
    pub full_attn_interval: usize,
    /// Сжатие ключей индексером по слоям (0 — слой без индексера).
    pub compress_ratios: Vec<usize>,
    pub indexer_heads: usize,
    pub indexer_head_dim: usize,
    pub indexer_top_k: usize,
    /// Hyper-connections: 4 потока, low-rank 320.
    pub hc_count: usize,
    pub hc_lowrank: usize,
    /// PLE: номера слоёв (0-based), n-граммы, голов на n-грамму, conv.
    pub ple_layers: Vec<usize>,
    pub ngram_size: usize,
    pub ple_heads_per_ngram: usize,
    pub ple_conv_kernel: usize,
    pub ple_eos: u32,
    /// Ширина строки n-граммной таблицы (ple_embed_dim / голов).
    pub ple_row_width: usize,
    pub ple_layer_multipliers: Vec<u64>,
    pub ple_head_offsets: Vec<u64>,
    pub ple_head_vocab_sizes: Vec<u64>,
    /// По слою: true — линейное внимание (GDN), false — полное.
    pub recurrent: Vec<bool>,
}

impl Qwen4ExpConfig {
    pub fn is_arch(arch: &str) -> bool {
        arch == ARCH
    }

    /// Разбор метаданных из GGUF.
    pub fn from_gguf(file: &GgufFile<'_>, vocab_size: usize) -> Result<Self, String> {
        Self::from_lookup(file.architecture().unwrap_or(""), vocab_size, &|k| {
            file.get(k).cloned()
        })
    }

    /// Разбор из произвольного источника ключей (тесты без файла).
    pub fn from_lookup(
        arch: &str,
        vocab_size: usize,
        get: &dyn Fn(&str) -> Option<GgufValue>,
    ) -> Result<Self, String> {
        if !Self::is_arch(arch) {
            return Err(format!("qwen4exp: архитектура {arch:?} не {ARCH}"));
        }
        let p = |s: &str| format!("{ARCH}.{s}");
        let u = |k: &str| -> Result<usize, String> {
            get(&p(k))
                .and_then(|v| as_u64_any(&v))
                .map(|v| v as usize)
                .ok_or_else(|| format!("qwen4exp: нет {k}"))
        };
        let f = |k: &str| -> Option<f32> {
            match get(&p(k))? {
                GgufValue::F32(x) => Some(x),
                GgufValue::F64(x) => Some(x as f32),
                GgufValue::U32(x) => Some(x as f32),
                _ => None,
            }
        };
        let arr_u64 = |k: &str| -> Vec<u64> {
            get(&p(k))
                .and_then(|v| {
                    v.as_array()
                        .map(|a| a.iter().filter_map(as_u64_any).collect())
                })
                .unwrap_or_default()
        };
        let arr_usize =
            |k: &str| -> Vec<usize> { arr_u64(k).into_iter().map(|v| v as usize).collect() };

        let dim = u("embedding_length")?;
        let n_layers = u("block_count")?;
        let n_heads = u("attention.head_count")?;
        let n_kv_heads = u("attention.head_count_kv")?;
        let head_dim = u("attention.key_length").unwrap_or(dim / n_heads);
        let n_rot = u("rope.dimension_count").unwrap_or(head_dim);
        let mut rope_sections = [0usize; 4];
        let secs = arr_usize("rope.dimension_sections");
        for (i, s) in secs.iter().take(4).enumerate() {
            rope_sections[i] = *s;
        }
        let interval = u("full_attention_interval").unwrap_or(4);
        let recurrent: Vec<bool> = (0..n_layers).map(|i| (i + 1) % interval != 0).collect();
        let mut compress_ratios = arr_usize("attention.compress_ratios");
        compress_ratios.resize(n_layers, 0);

        let ple_row_width = match u("embedding_length_per_layer_input") {
            Ok(v) if v > 0 => v,
            _ => dim,
        };

        Ok(Self {
            arch: arch.to_string(),
            dim,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            n_rot,
            rope_sections,
            rope_theta: f("rope.freq_base").unwrap_or(10_000.0),
            norm_eps: f("attention.layer_norm_rms_epsilon").unwrap_or(1e-6),
            max_seq: u("context_length").unwrap_or(4096),
            vocab_size,
            n_experts: u("expert_count").unwrap_or(0),
            n_experts_used: u("expert_used_count").unwrap_or(0),
            expert_ff: u("expert_feed_forward_length").unwrap_or(0),
            shared_ff: u("expert_shared_feed_forward_length").unwrap_or(0),
            conv_kernel: u("ssm.conv_kernel")?,
            ssm_state: u("ssm.state_size")?,
            ssm_k_heads: u("ssm.group_count")?,
            ssm_v_heads: u("ssm.time_step_rank")?,
            ssm_inner: u("ssm.inner_size")?,
            full_attn_interval: interval,
            compress_ratios,
            indexer_heads: u("attention.indexer.head_count").unwrap_or(0),
            indexer_head_dim: u("attention.indexer.key_length").unwrap_or(0),
            indexer_top_k: u("attention.indexer.top_k").unwrap_or(0),
            hc_count: u("hyper_connection.count").unwrap_or(1),
            hc_lowrank: u("hyper_connection.low_rank").unwrap_or(0),
            ple_layers: arr_usize("ple.layers"),
            ngram_size: u("ple.ngram_size").unwrap_or(0),
            ple_heads_per_ngram: u("ple.heads_per_ngram").unwrap_or(0),
            ple_conv_kernel: u("ple.conv_kernel").unwrap_or(0),
            ple_eos: u("ple.eos_token_id").unwrap_or(0) as u32,
            ple_row_width,
            ple_layer_multipliers: arr_u64("ple.layer_multipliers"),
            ple_head_offsets: arr_u64("ple.head_offsets"),
            ple_head_vocab_sizes: arr_u64("ple.head_vocab_sizes"),
            recurrent,
        })
    }

    /// Число GDN-слоёв / слоёв полного внимания.
    pub fn n_recurrent(&self) -> usize {
        self.recurrent.iter().filter(|r| **r).count()
    }

    pub fn n_full(&self) -> usize {
        self.n_layers - self.n_recurrent()
    }

    /// Ширина 4-поточного остаточного потока (hyper-connections).
    pub fn hc_hidden(&self) -> usize {
        self.dim * self.hc_count
    }

    /// GDN: conv_dim = 2·key_dim + value_dim.
    pub fn gdn_conv_dim(&self) -> usize {
        2 * self.ssm_k_heads * self.ssm_state + self.ssm_inner
    }

    /// План тензоров файла: имена и `ne` (GGUF-порядок). `n_vocab` — из
    /// токенизатора (в метаданных `qwen4exp` его нет).
    ///
    /// PLE-таблицы n-грамм (16 голов × ~20M) лежат ОТДЕЛЬНЫМ шардом и в план
    /// не входят: здесь только проекции PLE-слоя.
    pub fn tensor_plan(&self, n_vocab: usize) -> Vec<TensorSpec> {
        let mut out = Vec::new();
        let dim = self.dim;
        let hc = self.hc_hidden();
        let mut add = |name: String, ne: Vec<usize>| {
            out.push(TensorSpec { name, ne });
        };
        add("token_embd.weight".into(), vec![dim, n_vocab]);
        add("output.weight".into(), vec![dim, n_vocab]);
        add("output_hc_down.weight".into(), vec![hc, self.hc_lowrank]);
        add("output_hc_norm.weight".into(), vec![hc]);
        add("output_hc_up.weight".into(), vec![self.hc_lowrank, hc]);
        for i in 0..self.n_layers {
            let b = |s: &str| format!("blk.{i}.{s}");
            if self.recurrent[i] {
                add(b("attn_qkv.weight"), vec![dim, self.gdn_conv_dim()]);
                add(b("attn_gate.weight"), vec![dim, self.ssm_inner]);
                add(b("ssm_a"), vec![self.ssm_v_heads]);
                add(b("ssm_dt.bias"), vec![self.ssm_v_heads]);
                add(b("ssm_alpha.weight"), vec![dim, self.ssm_v_heads]);
                add(b("ssm_beta.weight"), vec![dim, self.ssm_v_heads]);
                add(
                    b("ssm_conv1d.weight"),
                    vec![self.conv_kernel, self.gdn_conv_dim()],
                );
                add(b("ssm_norm.weight"), vec![self.ssm_state]);
                add(b("ssm_out.weight"), vec![self.ssm_inner, dim]);
            } else {
                add(
                    b("attn_q.weight"),
                    vec![dim, 2 * self.n_heads * self.head_dim],
                );
                add(
                    b("attn_k.weight"),
                    vec![dim, self.n_kv_heads * self.head_dim],
                );
                add(
                    b("attn_v.weight"),
                    vec![dim, self.n_kv_heads * self.head_dim],
                );
                add(
                    b("attn_output.weight"),
                    vec![self.n_heads * self.head_dim, dim],
                );
                add(b("attn_q_norm.weight"), vec![self.head_dim]);
                add(b("attn_k_norm.weight"), vec![self.head_dim]);
                add(
                    b("indexer.q_proj.weight"),
                    vec![dim, self.indexer_heads * self.indexer_head_dim],
                );
                add(b("indexer.k_proj.weight"), vec![dim, self.indexer_head_dim]);
                add(b("indexer.q_norm.weight"), vec![self.indexer_head_dim]);
                add(b("indexer.k_norm.weight"), vec![self.indexer_head_dim]);
            }
            // MoE: 512 экспертов top-10 + shared-expert.
            add(b("ffn_gate_inp.weight"), vec![dim, self.n_experts]);
            add(b("ffn_gate_inp_shexp.weight"), vec![dim]);
            add(b("ffn_gate_shexp.weight"), vec![dim, self.shared_ff]);
            add(b("ffn_up_shexp.weight"), vec![dim, self.shared_ff]);
            add(b("ffn_down_shexp.weight"), vec![self.shared_ff, dim]);
            add(
                b("ffn_gate_exps.weight"),
                vec![dim, self.expert_ff, self.n_experts],
            );
            add(
                b("ffn_up_exps.weight"),
                vec![dim, self.expert_ff, self.n_experts],
            );
            add(
                b("ffn_down_exps.weight"),
                vec![self.expert_ff, dim, self.n_experts],
            );
            // Hyper-connections: ветка внимания и ветка FFN.
            for branch in ["hc_attn", "hc_ffn"] {
                add(
                    b(&format!("{branch}_down.weight")),
                    vec![hc, self.hc_lowrank],
                );
                add(b(&format!("{branch}_norm.weight")), vec![hc]);
                add(b(&format!("{branch}_up.weight")), vec![self.hc_lowrank, hc]);
                add(
                    b(&format!("{branch}_inject.weight")),
                    vec![hc, self.hc_count],
                );
            }
            // PLE-слой (по метаданным) — проекции; таблицы n-грамм в шарде 2.
            if self.ple_layers.contains(&i) {
                add(b("ple_conv1d.weight"), vec![self.ple_conv_kernel, hc]);
                add(b("ple_key.weight"), vec![self.ple_row_width, hc]);
                add(b("ple_value.weight"), vec![self.ple_row_width, dim]);
                add(b("ple_norm_conv.weight"), vec![hc]);
                add(b("ple_norm_key.weight"), vec![hc]);
                add(b("ple_norm_query.weight"), vec![hc]);
            }
        }
        out
    }
}

fn as_u64_any(v: &GgufValue) -> Option<u64> {
    match v {
        GgufValue::U32(x) => Some(*x as u64),
        GgufValue::I32(x) => Some(*x as u64),
        GgufValue::U64(x) => Some(*x),
        GgufValue::I64(x) => Some(*x as u64),
        GgufValue::U16(x) => Some(*x as u64),
        GgufValue::I16(x) => Some(*x as u64),
        GgufValue::U8(x) => Some(*x as u64),
        GgufValue::I8(x) => Some(*x as u64),
        _ => None,
    }
}

/// Ожидаемый тензор: имя + размерности в GGUF-порядке (`ne0` первым).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: String,
    pub ne: Vec<usize>,
}

/// Сверка плана с файлом: чего нет, что лишнее по размерностям.
/// Возвращает текст для человека; пусто — файл соответствует схеме.
pub fn check_file(cfg: &Qwen4ExpConfig, file: &GgufFile<'_>, n_vocab: usize) -> Vec<String> {
    let plan = cfg.tensor_plan(n_vocab);
    let mut problems = Vec::new();
    for spec in &plan {
        match file.tensors.iter().find(|t| t.name == spec.name) {
            None => problems.push(format!("нет тензора {}", spec.name)),
            Some(t) => {
                let ne: Vec<usize> = t.dims.iter().map(|d| *d as usize).collect();
                if ne != spec.ne {
                    problems.push(format!(
                        "{}: dims {:?}, в плане {:?}",
                        spec.name, ne, spec.ne
                    ));
                }
            }
        }
    }
    problems
}

/// Статус архитектуры для реестра (`arch.rs`).
pub fn arch_entry() -> arch::ArchInfo {
    // Пока — Planned: конфиг и схема сверены с реальным файлом, forward
    // (hyper-connections/QSA/PLE/MoE) — волна A5.
    arch::ArchInfo {
        name: ARCH,
        family: arch::Family::Moe,
        status: arch::Support::Planned,
        note: "Qwen3.8-Flash-Next: конфиг+схема сверены с GGUF; forward — A5",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Метаданные реального файла (IQ2_XS shard 1): значения сняты из
    /// заголовка ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF.
    fn real_meta() -> HashMap<String, GgufValue> {
        let mut m = HashMap::new();
        let u = |v: u32| GgufValue::U32(v);
        m.insert("qwen4exp.block_count".into(), u(48));
        m.insert("qwen4exp.context_length".into(), u(262144));
        m.insert("qwen4exp.embedding_length".into(), u(2560));
        m.insert("qwen4exp.attention.head_count".into(), u(24));
        m.insert("qwen4exp.attention.head_count_kv".into(), u(2));
        m.insert("qwen4exp.attention.key_length".into(), u(256));
        m.insert("qwen4exp.rope.dimension_count".into(), u(64));
        m.insert("qwen4exp.rope.freq_base".into(), GgufValue::F32(1e7));
        m.insert(
            "qwen4exp.rope.dimension_sections".into(),
            GgufValue::Array(vec![u(11), u(11), u(10), u(0)]),
        );
        m.insert(
            "qwen4exp.attention.layer_norm_rms_epsilon".into(),
            GgufValue::F32(1e-6),
        );
        m.insert("qwen4exp.expert_count".into(), u(512));
        m.insert("qwen4exp.expert_used_count".into(), u(10));
        m.insert("qwen4exp.expert_feed_forward_length".into(), u(640));
        m.insert("qwen4exp.expert_shared_feed_forward_length".into(), u(640));
        m.insert("qwen4exp.ssm.conv_kernel".into(), u(4));
        m.insert("qwen4exp.ssm.state_size".into(), u(128));
        m.insert("qwen4exp.ssm.group_count".into(), u(16));
        m.insert("qwen4exp.ssm.time_step_rank".into(), u(48));
        m.insert("qwen4exp.ssm.inner_size".into(), u(6144));
        m.insert("qwen4exp.full_attention_interval".into(), u(4));
        m.insert("qwen4exp.hyper_connection.count".into(), u(4));
        m.insert("qwen4exp.hyper_connection.low_rank".into(), u(320));
        m.insert("qwen4exp.attention.indexer.head_count".into(), u(4));
        m.insert("qwen4exp.attention.indexer.key_length".into(), u(128));
        m.insert("qwen4exp.attention.indexer.top_k".into(), u(2048));
        let mut ratios = vec![u(0); 48];
        for i in (3..48).step_by(4) {
            ratios[i] = u(4);
        }
        m.insert(
            "qwen4exp.attention.compress_ratios".into(),
            GgufValue::Array(ratios),
        );
        m.insert("qwen4exp.ple.layers".into(), GgufValue::Array(vec![u(1)]));
        m.insert("qwen4exp.ple.ngram_size".into(), u(3));
        m.insert("qwen4exp.ple.heads_per_ngram".into(), u(8));
        m.insert("qwen4exp.ple.conv_kernel".into(), u(4));
        m.insert("qwen4exp.ple.eos_token_id".into(), u(248044));
        m.insert("qwen4exp.embedding_length_per_layer_input".into(), u(160));
        m.insert(
            "qwen4exp.ple.layer_multipliers".into(),
            GgufValue::Array(vec![
                GgufValue::U64(23703573157769),
                GgufValue::U64(20109073645365),
                GgufValue::U64(8052911324071),
            ]),
        );
        let offsets: Vec<GgufValue> = [
            0u64, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
            160000406, 180000449, 200000514, 220000595, 240000646, 260000741, 280000824, 300000925,
        ]
        .into_iter()
        .map(GgufValue::U64)
        .collect();
        m.insert(
            "qwen4exp.ple.head_offsets".into(),
            GgufValue::Array(offsets),
        );
        let sizes: Vec<GgufValue> = [
            20000003u64,
            20000023,
            20000033,
            20000047,
            20000059,
            20000063,
            20000069,
            20000077,
            20000079,
            20000083,
            20000087,
            20000089,
            20000097,
            20000101,
            20000107,
            20000109,
        ]
        .into_iter()
        .map(GgufValue::U64)
        .collect();
        m.insert(
            "qwen4exp.ple.head_vocab_sizes".into(),
            GgufValue::Array(sizes),
        );
        m
    }

    fn real_cfg() -> Qwen4ExpConfig {
        let m = real_meta();
        Qwen4ExpConfig::from_lookup(ARCH, 248320, &|k| m.get(k).cloned()).expect("конфиг")
    }

    #[test]
    fn parses_real_header() {
        let c = real_cfg();
        assert_eq!((c.dim, c.n_layers, c.head_dim), (2560, 48, 256));
        assert_eq!((c.n_heads, c.n_kv_heads), (24, 2));
        assert_eq!((c.n_rot, c.rope_sections), (64, [11, 11, 10, 0]));
        assert_eq!((c.n_experts, c.n_experts_used), (512, 10));
        assert_eq!((c.expert_ff, c.shared_ff), (640, 640));
        assert_eq!((c.hc_count, c.hc_lowrank), (4, 320));
        assert_eq!(c.hc_hidden(), 10240);
        assert_eq!(c.gdn_conv_dim(), 10240);
        assert_eq!((c.ssm_k_heads, c.ssm_v_heads, c.ssm_inner), (16, 48, 6144));
        assert_eq!(
            (c.indexer_heads, c.indexer_head_dim, c.indexer_top_k),
            (4, 128, 2048)
        );
        assert_eq!(c.ple_layers, vec![1]);
        assert_eq!(c.ple_row_width, 160);
        assert_eq!(c.ple_layer_multipliers.len(), 3);
        assert_eq!((c.n_recurrent(), c.n_full()), (36, 12));
        assert!(c.recurrent[0] && c.recurrent[1] && c.recurrent[2] && !c.recurrent[3]);
    }

    /// План обязан совпадать с реальными именами/размерностями shard 1.
    #[test]
    fn plan_matches_real_file_shapes() {
        let c = real_cfg();
        let plan = c.tensor_plan(248320);
        let find = |n: &str| plan.iter().find(|s| s.name == n).unwrap().ne.clone();
        assert_eq!(find("token_embd.weight"), vec![2560, 248320]);
        assert_eq!(find("output_hc_down.weight"), vec![10240, 320]);
        assert_eq!(find("output_hc_up.weight"), vec![320, 10240]);
        // GDN-слой: qkv 2·16·128 + 6144 = 10240, gate = value_dim.
        assert_eq!(find("blk.0.attn_qkv.weight"), vec![2560, 10240]);
        assert_eq!(find("blk.0.attn_gate.weight"), vec![2560, 6144]);
        assert_eq!(find("blk.0.ssm_conv1d.weight"), vec![4, 10240]);
        assert_eq!(find("blk.0.ssm_out.weight"), vec![6144, 2560]);
        // Полное внимание: q = 2·24·256, k/v = 2·256.
        assert_eq!(find("blk.3.attn_q.weight"), vec![2560, 12288]);
        assert_eq!(find("blk.3.attn_k.weight"), vec![2560, 512]);
        assert_eq!(find("blk.3.attn_output.weight"), vec![6144, 2560]);
        assert_eq!(find("blk.3.indexer.q_proj.weight"), vec![2560, 512]);
        assert_eq!(find("blk.3.indexer.k_proj.weight"), vec![2560, 128]);
        // MoE: 3D-эксперты [dim, 640, 512].
        assert_eq!(find("blk.0.ffn_gate_exps.weight"), vec![2560, 640, 512]);
        assert_eq!(find("blk.0.ffn_down_exps.weight"), vec![640, 2560, 512]);
        assert_eq!(find("blk.0.ffn_gate_inp.weight"), vec![2560, 512]);
        // PLE только у слоя 1.
        assert_eq!(find("blk.1.ple_key.weight"), vec![160, 10240]);
        assert!(plan.iter().all(|s| !s.name.starts_with("blk.2.ple_")));
        // Счётчик сверен с реальным заголовком: 1223 тензора в shard 1.
        // 48·(8 MoE + 8 hc) + 36·9 GDN + 12·10 attention + 5 плотных + 6 PLE.
        assert_eq!(plan.len(), 1223, "план должен совпадать с shard 1");
    }
}
