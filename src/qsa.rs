//! QSA-индексер (`qwen4exp` / Flash-Next) — шаг Э2.3.
//!
//! Порт `build_qsa_top_k` из llama.cpp (`src/models/qwen4exp.cpp`,
//! эталон 5266f24da). У Flash-Next разреженное внимание не по всем ячейкам
//! KV: индексер держит СВОЙ низкоразмерный кэш ключей (одна голова,
//! `indexer_head_size`), ключи которого
//!
//! 1. пишутся сырыми (без норм и поворота — они накладываются позже, уже
//!    на усреднённый блок);
//! 2. **усредняются блоками** по `compress_ratio` ячеек;
//! 3. нормируются RMSNorm и поворачиваются RoPE по позиции БЛОКА;
//! 4. скорируются запросом `index_q_proj` (несколько голов): скор головы —
//!    `relu(q·k)`, скоры голов складываются;
//! 5. скор блока раскладывается на все его ячейки, затем берётся
//!    `top_k + ratio − 1` ячеек — бюджет в ЦЕЛЫХ блоках плюс хвост.
//!
//! Здесь живёт арифметика шагов 2, 4 (свёртка голов) и 5 (выбор ячеек).
//! Нормы и повороты вызывающий делает существующими оп-ами (`rmsnorm`,
//! `rope`), а кэш ключей — его забота: этот модуль принимает готовые
//! (уже повёрнутые) ключи и запросы, поэтому тестируется без модели.

/// Константы индексер-слоя из GGUF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QsaCfg {
    /// Ширина ключа/запроса индексёра (`attention.indexer.key_length`).
    pub idx_dim: usize,
    /// Голов запроса индексёра (`attention.indexer.head_count`).
    pub n_heads: usize,
    /// Бюджет в ячейках (`attention.indexer.top_k`).
    pub top_k: usize,
    /// Блок сжатия этого слоя (`attention.compress_ratios[layer]`).
    pub ratio: usize,
}

impl QsaCfg {
    pub fn validate(&self) -> Result<(), String> {
        if self.idx_dim == 0 || self.n_heads == 0 {
            return Err("QSA: idx_dim/n_heads нулевые".into());
        }
        if self.top_k == 0 {
            return Err("QSA: top_k = 0".into());
        }
        if self.ratio == 0 {
            return Err("QSA: ratio = 0 (слой без сжатия должен идти плотным путём)".into());
        }
        Ok(())
    }

    /// Ширина top-k: бюджет в ячейках плюс хвост неполного блока
    /// (`min(n_kv, top_k + ratio − 1)`), как в эталоне.
    pub fn width(&self, n_kv: usize) -> usize {
        (self.top_k + self.ratio - 1).min(n_kv)
    }

    pub fn n_blocks(&self, n_kv: usize) -> usize {
        n_kv.div_ceil(self.ratio)
    }
}

/// Ячейки по блокам: `cells[blk*ratio + i]` — индекс KV-ячейки `i`-го
/// члена блока `blk`. Неполный хвост повторяет последнюю валидную ячейку,
/// иначе среднее блока разбавилось бы нулями (в эталоне это делает
/// memory-слой, здесь — явно и проверяемо).
pub fn tail_cells(n_kv: usize, ratio: usize) -> Vec<u32> {
    let blocks = n_kv.div_ceil(ratio);
    let mut cells = Vec::with_capacity(blocks * ratio);
    for b in 0..blocks {
        for i in 0..ratio {
            let cell = (b * ratio + i).min(n_kv.saturating_sub(1));
            cells.push(cell as u32);
        }
    }
    cells
}

/// Среднее ключей блока: `keys [n_kv, idx_dim]`, `cells [n_blocks·ratio]`
/// → `pooled [n_blocks, idx_dim]`. Сумма ровно `ratio` членов с делением на
/// `ratio` — как `pooled = Σ members / r` в эталоне.
pub fn pool_blocks(
    keys: &[f32],
    cells: &[u32],
    ratio: usize,
    idx_dim: usize,
) -> Result<Vec<f32>, String> {
    if ratio == 0 || idx_dim == 0 {
        return Err("QSA: pool при нулевом ratio/idx_dim".into());
    }
    if cells.len() % ratio != 0 {
        return Err(format!(
            "QSA: ячеек {} не кратно блоку {ratio}",
            cells.len()
        ));
    }
    let blocks = cells.len() / ratio;
    let n_kv = keys.len() / idx_dim;
    if keys.len() % idx_dim != 0 {
        return Err(format!("QSA: ключи {} не кратны {idx_dim}", keys.len()));
    }
    let mut out = vec![0f32; blocks * idx_dim];
    for b in 0..blocks {
        for i in 0..ratio {
            let cell = cells[b * ratio + i] as usize;
            if cell >= n_kv {
                return Err(format!("QSA: ячейка {cell} вне кэша ({n_kv})"));
            }
            let src = &keys[cell * idx_dim..(cell + 1) * idx_dim];
            let dst = &mut out[b * idx_dim..(b + 1) * idx_dim];
            for k in 0..idx_dim {
                dst[k] += src[k];
            }
        }
        let dst = &mut out[b * idx_dim..(b + 1) * idx_dim];
        for k in 0..idx_dim {
            dst[k] /= ratio as f32;
        }
    }
    Ok(out)
}

/// Скор блоков: `q [n_tokens, n_heads·idx_dim]` и `pooled [n_blocks, idx_dim]`
/// → `scores [n_tokens, n_blocks]`.
///
/// Скор головы — `relu(q·k)` (rectified dot, как в lightning indexer
/// DeepSeek), скоры голов складываются. `relu` считается ДО суммы: отрицательные
/// головы не гасят положительные.
pub fn block_scores(
    q: &[f32],
    pooled: &[f32],
    n_tokens: usize,
    cfg: QsaCfg,
) -> Result<Vec<f32>, String> {
    cfg.validate()?;
    let (d, h) = (cfg.idx_dim, cfg.n_heads);
    if q.len() != n_tokens * h * d {
        return Err(format!(
            "QSA: запрос {} ≠ {n_tokens}·{h}·{d}",
            q.len()
        ));
    }
    if pooled.len() % d != 0 {
        return Err(format!("QSA: pooled {} не кратен {d}", pooled.len()));
    }
    let blocks = pooled.len() / d;
    let mut out = vec![0f32; n_tokens * blocks];
    for t in 0..n_tokens {
        for b in 0..blocks {
            let mut s = 0f32;
            for head in 0..h {
                let qh = &q[(t * h + head) * d..(t * h + head + 1) * d];
                let kb = &pooled[b * d..(b + 1) * d];
                let dot: f32 = qh.iter().zip(kb.iter()).map(|(a, c)| a * c).sum();
                s += dot.max(0.0);
            }
            out[t * blocks + b] = s;
        }
    }
    Ok(out)
}

/// Скор блока на каждую его ячейку + `top_k` выбор.
///
/// `block_scores [n_tokens, n_blocks]`, `cells [n_blocks·ratio]`,
/// `bias` — необязательная поправка на ячейку (маска/ALiBi; None — нули).
/// Возвращает `[n_tokens, width]` индексов ячеек, отсортированных по скору
/// по убыванию (ничьи — меньший индекс первым), как `ggml_top_k` + наш
/// роутерный порядок.
pub fn top_k_cells(
    block_scores: &[f32],
    cells: &[u32],
    cfg: QsaCfg,
    n_tokens: usize,
    bias: Option<&[f32]>,
) -> Result<Vec<u32>, String> {
    cfg.validate()?;
    let n_blocks = cells.len() / cfg.ratio;
    if block_scores.len() != n_tokens * n_blocks {
        return Err(format!(
            "QSA: скоров {} ≠ {n_tokens}·{n_blocks}",
            block_scores.len()
        ));
    }
    // Число ячеек, по которым идёт выбор: максимум индекс ячейки + 1
    // (хвост повторяет последнюю, поэтому это и есть n_kv).
    let n_kv = cells.iter().copied().max().unwrap_or(0) as usize + 1;
    let width = cfg.width(n_kv);
    let mut out = Vec::with_capacity(n_tokens * width);
    for t in 0..n_tokens {
        // score ячейки = score её блока (+ bias)
        let mut per_cell: Vec<(f32, u32)> = Vec::with_capacity(n_blocks * cfg.ratio);
        for b in 0..n_blocks {
            let s = block_scores[t * n_blocks + b];
            for i in 0..cfg.ratio {
                let cell = cells[b * cfg.ratio + i];
                let add = bias.map(|v| v.get(cell as usize).copied().unwrap_or(0.0)).unwrap_or(0.0);
                per_cell.push((s + add, cell));
            }
        }
        // Берём уникальные ячейки: хвост блока повторяет последнюю ячейку,
        // и без дедупликации она заняла бы несколько мест в top-k.
        per_cell.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        let mut seen = vec![false; n_kv];
        let mut taken = 0;
        for (_, cell) in per_cell {
            if seen[cell as usize] {
                continue;
            }
            seen[cell as usize] = true;
            out.push(cell);
            taken += 1;
            if taken == width {
                break;
            }
        }
        // Меньше width возможно только если ячеек физически меньше.
        while taken < width {
            out.push(0);
            taken += 1;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Блок 2, две головы, dim 1 — считается в уме.
    fn cfg() -> QsaCfg {
        QsaCfg {
            idx_dim: 1,
            n_heads: 2,
            top_k: 2,
            ratio: 2,
        }
    }

    /// Хвост неполного блока повторяет последнюю ячейку.
    #[test]
    fn tail_cells_repeat_last() {
        // n_kv = 5, ratio 2 → блоки [0,1] [2,3] [4,4]
        assert_eq!(tail_cells(5, 2), vec![0, 1, 2, 3, 4, 4]);
        assert_eq!(tail_cells(4, 2), vec![0, 1, 2, 3]);
        assert_eq!(tail_cells(1, 4), vec![0, 0, 0, 0]);
    }

    /// Среднее блока — ровно сумма ratio членов / ratio.
    #[test]
    fn pool_is_mean_of_block_members() {
        // keys: 4 ячейки по 2 числа; блок 2
        let keys = vec![
            1.0, 2.0, // 0
            3.0, 4.0, // 1
            5.0, 6.0, // 2
            7.0, 8.0, // 3
        ];
        let cells = tail_cells(4, 2);
        let pooled = pool_blocks(&keys, &cells, 2, 2).unwrap();
        assert_eq!(pooled, vec![2.0, 3.0, 6.0, 7.0]);
    }

    /// Реакция на неполный хвост: последний блок усредняет повтор,
    /// то есть равен значению последней ячейки, а не (x+0)/2.
    #[test]
    fn pool_tail_block_uses_repeated_last_cell() {
        let keys = vec![10.0, 20.0, 30.0]; // dim 1, 3 ячейки
        let cells = tail_cells(3, 2); // [0,1] [2,2]
        let pooled = pool_blocks(&keys, &cells, 2, 1).unwrap();
        assert_eq!(pooled, vec![15.0, 30.0]);
    }

    /// Скор: relu по голове ДО суммы голов. Два блока (dim 1), две головы:
    /// q = [3, −5], ключи блоков [1, −1] →
    /// блок 0: relu(3·1) + relu(−5·1) = 3 + 0 = 3
    /// блок 1: relu(3·−1) + relu(−5·−1) = 0 + 5 = 5
    /// (sum-then-relu дал бы 0 и 2 — эти числа и ловят порядок операций).
    #[test]
    fn scores_rectify_per_head_before_sum() {
        let c = cfg();
        let q = vec![3.0, -5.0];
        let pooled = vec![1.0, -1.0];
        let s = block_scores(&q, &pooled, 1, c).unwrap();
        assert_eq!(s, vec![3.0, 5.0], "relu до суммы голов");
    }

    /// top-k выбирает ячейки по скору блока, ширина = top_k + ratio − 1,
    /// ничьи — меньший индекс первым, повтор хвоста не дублируется.
    #[test]
    fn top_k_expands_blocks_and_dedups_tail() {
        let c = cfg(); // ratio 2, top_k 2 → width = min(n_kv, 3)
        // n_kv = 4 → блоки [0,1] и [2,3]; блоки: 1.0 и 5.0
        let block_scores = vec![1.0, 5.0];
        let cells = tail_cells(4, 2);
        let idx = top_k_cells(&block_scores, &cells, c, 1, None).unwrap();
        // ширина 3: две ячейки лучшего блока (2, 3) + одна следующего (0)
        assert_eq!(idx, vec![2, 3, 0]);
    }

    /// Неполный хвост: повтор последней ячейки не занимает два места.
    #[test]
    fn top_k_does_not_duplicate_repeated_tail_cell() {
        let c = cfg();
        // n_kv = 3 → блоки [0,1] и [2,2]; лучший — второй
        let block_scores = vec![1.0, 9.0];
        let cells = tail_cells(3, 2);
        let idx = top_k_cells(&block_scores, &cells, c, 1, None).unwrap();
        // width = min(3, 3) = 3: ячейка 2 один раз, затем 0 и 1
        assert_eq!(idx, vec![2, 0, 1]);
    }

    /// Bias прибавляется к скору ячейки (маска/ALiBi) и может переставить выбор.
    #[test]
    fn bias_can_change_selection() {
        let c = cfg();
        let block_scores = vec![5.0, 1.0];
        let cells = tail_cells(4, 2);
        let idx = top_k_cells(&block_scores, &cells, c, 1, None).unwrap();
        assert_eq!(idx, vec![0, 1, 2]);
        // штраф первым двум ячейкам: лучший блок теперь второй
        let bias = vec![-10.0, -10.0, 0.0, 0.0];
        let idx = top_k_cells(&block_scores, &cells, c, 1, Some(&bias)).unwrap();
        assert_eq!(idx, vec![2, 3, 0]);
    }
}
