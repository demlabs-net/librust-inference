//! Карта весов Flash-Next: bindings NINFER → поля модели.
//!
//! Здесь нет математики forward — только «что где лежит и какой формы».
//! Смысл: сверка карты с конфигом ДО загрузки байтов. Если имя пропало или
//! форма не та, ошибка всплывает здесь, а не через час на первый токен.
//!
//! Имена слева — bindings артефакта (`text/layers/3/attention/key`), справа —
//! роль в модели (см. `storage-layouts.md` и `qwen3_8-flash-next-design.md`).

use crate::plugins::qwen4exp::config::Qwen4ExpConfig;
use crate::plugins::qwen4exp::load::Weights;

/// Роль тензора в слое — по ней forward выберет путь.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// hyper-connections: гамма `[hc·dim]`, up `[hc·dim, lr]`, down+inject.
    HcNorm,
    HcUp,
    HcDownInject,
    /// GDN: fused q|k, value, z (гейт), output, conv, a, dt, norm, alpha|beta.
    GdnQk,
    GdnValue,
    GdnGate,
    GdnOut,
    GdnConv,
    GdnA,
    GdnDt,
    GdnNorm,
    GdnAlphaBeta,
    /// Полное внимание: q|gate, k, v, out, нормы, индексер.
    AttnQGate,
    AttnK,
    AttnV,
    AttnOut,
    AttnQNorm,
    AttnKNorm,
    IdxQ,
    IdxK,
    IdxQNorm,
    IdxKNorm,
    /// MoE: роутер, гейт общего эксперта, общий эксперт, банк экспертов.
    MoeRouterScore,
    MoeSharedGate,
    MoeSharedUp,
    MoeSharedDown,
    MoeExperts,
    /// PLE-слой.
    PleKey,
    PleValue,
    PleNormKey,
    PleNormQuery,
    PleNormConv,
    PleConv,
}

/// Ожидаемый тензор: bindings-имя, роль, форма `[rows, cols]` (None у 3-Д).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expect {
    pub name: String,
    pub role: Role,
    pub shape: Option<(usize, usize)>,
}

/// Ветка HC: `attn_hc` (перед миксером внимания) или `mlp_hc` (перед MoE).
fn hc(cfg: &Qwen4ExpConfig, layer: usize, branch: &str) -> Vec<Expect> {
    let hc_dim = cfg.hc_hidden();
    let lr = cfg.hc_lowrank;
    let p = |s: &str| format!("text/layers/{layer}/{branch}/{s}");
    vec![
        Expect { name: p("norm"), role: Role::HcNorm, shape: Some((1, hc_dim)) },
        // В манифесте `up` = [lr, hc·dim], `down` = [hc·dim, lr] (проверено
        // сверкой с артефактом); семантика (какая матрица что умножает)
        // уточняется по forward-коду, здесь важна форма файла.
        Expect { name: p("up"), role: Role::HcUp, shape: Some((lr, hc_dim)) },
        // down и inject — части одного объекта: [lr, hc_dim] + [hc, hc_dim]
        Expect {
            name: p("down"),
            role: Role::HcDownInject,
            shape: Some((hc_dim, lr)),
        },
        Expect {
            name: p("inject"),
            role: Role::HcDownInject,
            shape: Some((cfg.hc_count, hc_dim)),
        },
    ]
}

/// Все тензоры слоя по конфигу. Порядок — как в слое forward.
pub fn layer_tensors(cfg: &Qwen4ExpConfig, layer: usize) -> Vec<Expect> {
    let mut out = hc(cfg, layer, "attn_hc");
    if cfg.recurrent[layer] {
        let key_dim = cfg.ssm_k_heads * cfg.ssm_state;
        let value_dim = cfg.ssm_inner;
        let p = |s: &str| format!("text/layers/{layer}/gdn/{s}");
        out.extend([
            // q|k — две части одного объекта: [key_dim, dim] каждая
            Expect { name: p("query"), role: Role::GdnQk, shape: Some((key_dim, cfg.dim)) },
            Expect { name: p("key"), role: Role::GdnQk, shape: Some((key_dim, cfg.dim)) },
            Expect { name: p("value"), role: Role::GdnValue, shape: Some((value_dim, cfg.dim)) },
            Expect { name: p("z"), role: Role::GdnGate, shape: Some((value_dim, cfg.dim)) },
            Expect { name: p("output"), role: Role::GdnOut, shape: Some((cfg.dim, value_dim)) },
            Expect { name: p("convolution"), role: Role::GdnConv, shape: Some((cfg.conv_kernel, 2 * key_dim + value_dim)) },
            Expect { name: p("a"), role: Role::GdnA, shape: Some((1, cfg.ssm_v_heads)) },
            Expect { name: p("dt_bias"), role: Role::GdnDt, shape: Some((1, cfg.ssm_v_heads)) },
            Expect { name: p("norm"), role: Role::GdnNorm, shape: Some((1, cfg.ssm_state)) },
            // alpha|beta — тоже части одного объекта
            Expect { name: p("a_projection"), role: Role::GdnAlphaBeta, shape: Some((cfg.ssm_v_heads, cfg.dim)) },
            Expect { name: p("b_projection"), role: Role::GdnAlphaBeta, shape: Some((cfg.ssm_v_heads, cfg.dim)) },
        ]);
    } else {
        let q_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let idx = cfg.indexer_heads * cfg.indexer_head_dim;
        let p = |s: &str| format!("text/layers/{layer}/attention/{s}");
        out.extend([
            Expect { name: p("query"), role: Role::AttnQGate, shape: Some((q_dim, cfg.dim)) },
            Expect { name: p("gate"), role: Role::AttnQGate, shape: Some((q_dim, cfg.dim)) },
            Expect { name: p("key"), role: Role::AttnK, shape: Some((kv_dim, cfg.dim)) },
            Expect { name: p("value"), role: Role::AttnV, shape: Some((kv_dim, cfg.dim)) },
            Expect { name: p("output"), role: Role::AttnOut, shape: Some((cfg.dim, q_dim)) },
            Expect { name: p("query_norm"), role: Role::AttnQNorm, shape: Some((1, cfg.head_dim)) },
            Expect { name: p("key_norm"), role: Role::AttnKNorm, shape: Some((1, cfg.head_dim)) },
            Expect { name: p("index_query"), role: Role::IdxQ, shape: Some((idx, cfg.dim)) },
            Expect { name: p("index_key"), role: Role::IdxK, shape: Some((cfg.indexer_head_dim, cfg.dim)) },
            Expect { name: p("index_query_norm"), role: Role::IdxQNorm, shape: Some((1, cfg.indexer_head_dim)) },
            Expect { name: p("index_key_norm"), role: Role::IdxKNorm, shape: Some((1, cfg.indexer_head_dim)) },
        ]);
    }
    out.extend(hc(cfg, layer, "mlp_hc"));

    let p = |s: &str| format!("text/layers/{layer}/moe/{s}");
    out.extend([
        // router|shared_score — части одного объекта: [E, dim] + [1, dim]
        Expect { name: p("router"), role: Role::MoeRouterScore, shape: Some((cfg.n_experts, cfg.dim)) },
        Expect { name: p("shared_score"), role: Role::MoeRouterScore, shape: Some((1, cfg.dim)) },
        Expect { name: p("shared/gate"), role: Role::MoeSharedGate, shape: Some((cfg.shared_ff, cfg.dim)) },
        Expect { name: p("shared/up"), role: Role::MoeSharedUp, shape: Some((cfg.shared_ff, cfg.dim)) },
        Expect { name: p("shared/down"), role: Role::MoeSharedDown, shape: Some((cfg.dim, cfg.shared_ff)) },
        // банк экспертов — 3-Д, форма проверяется отдельно (ExpertBank)
        Expect { name: p("experts"), role: Role::MoeExperts, shape: None },
    ]);

    if cfg.ple_layers.contains(&layer) {
        let p = |s: &str| format!("text/layers/{layer}/{s}");
        out.extend([
            Expect { name: p("ple_key"), role: Role::PleKey, shape: Some((cfg.ple_row_width, cfg.hc_hidden())) },
            Expect { name: p("ple_value"), role: Role::PleValue, shape: Some((cfg.ple_row_width, cfg.hc_hidden())) },
            Expect { name: p("ple_norm_key"), role: Role::PleNormKey, shape: Some((cfg.hc_count, cfg.dim)) },
            Expect { name: p("ple_norm_query"), role: Role::PleNormQuery, shape: Some((cfg.hc_count, cfg.dim)) },
            Expect { name: p("ple_norm_conv"), role: Role::PleNormConv, shape: Some((cfg.hc_count, cfg.dim)) },
            Expect { name: p("ple_conv1d"), role: Role::PleConv, shape: Some((cfg.hc_count, cfg.dim)) },
        ]);
    }
    out
}

/// Тензоры верхнего уровня: эмбеддинг, голова, финальный миксер.
pub fn top_tensors(cfg: &Qwen4ExpConfig) -> Vec<Expect> {
    let hc_dim = cfg.hc_hidden();
    vec![
        Expect { name: "text/token_embedding".into(), role: Role::HcNorm, shape: Some((cfg.vocab_size, cfg.dim)) },
        Expect { name: "text/output_head".into(), role: Role::HcNorm, shape: Some((cfg.vocab_size, cfg.dim)) },
        Expect { name: "text/final_mixer/norm".into(), role: Role::HcNorm, shape: Some((1, hc_dim)) },
        Expect { name: "text/final_mixer/up".into(), role: Role::HcUp, shape: Some((cfg.hc_lowrank, hc_dim)) },
        Expect { name: "text/final_mixer/down".into(), role: Role::HcDownInject, shape: Some((hc_dim, cfg.hc_lowrank)) },
    ]
}

/// Сверить карту с контейнером: список расхождений (пусто — всё на месте).
///
/// Читаются только заголовки частей, байты весов не тянутся.
pub fn validate(weights: &Weights, cfg: &Qwen4ExpConfig) -> Result<Vec<String>, String> {
    let mut problems = Vec::new();
    let mut check = |e: &Expect| {
        match weights.shape(&e.name) {
            Ok((rows, cols, _format)) => {
                if let Some((want_rows, want_cols)) = e.shape
                    && (rows, cols) != (want_rows, want_cols)
                {
                    problems.push(format!(
                        "{}: {:?} {:?}, ожидалось {:?}",
                        e.name,
                        rows,
                        cols,
                        (want_rows, want_cols)
                    ));
                }
            }
            Err(err) => problems.push(format!("{}: {err}", e.name)),
        }
    };
    for e in top_tensors(cfg) {
        check(&e);
    }
    for layer in 0..cfg.n_layers {
        for e in layer_tensors(cfg, layer) {
            // 3-Д банк экспертов сверяется ExpertBank-ом, не здесь
            if e.shape.is_some() {
                check(&e);
            }
        }
    }
    Ok(problems)
}
