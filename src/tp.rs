//! TP-задел (7.6): шардирование линейных слоёв между нодами.
//!
//! Тензорный параллелизм делит ВЕСА одного слоя, а не слои целиком (в
//! отличие от PP, 7.5). Две раскладки:
//!
//! * **колоночная** (по выходным каналам): `W` режется по строкам, каждая
//!   нода считает свою часть выходных фич (`Y[:, shard]`), связи в forward
//!   НЕТ — части собираются конкатенацией (all-gather) на выходе слоя;
//! * **строчная** (по входным каналам): `W` режется по столбцам, и каждая
//!   нода считает ЧАСТИЧНУЮ сумму полного `Y` — на выходе нужен all-reduce.
//!
//! Здесь — только примитивы нарезки и проверка численной эквивалентности
//! (сама сеть all-reduce — следующий шаг: кластерное сообщение + движок).

use crate::backend::{Backend, Tensor};
use crate::error::BackendError;

/// Нарезка весов по СТРОКАМ (колоночный параллелизм: выходные каналы).
/// Возвращает [rows/n, cols] — кусок, дающий свою часть выходных фич.
pub fn split_rows(
    _backend: &dyn Backend,
    w: &Tensor,
    shard: usize,
    n_shards: usize,
) -> Result<Tensor, BackendError> {
    let (rows, cols) = (w.shape()[0], w.shape()[1]);
    if n_shards == 0 || shard >= n_shards {
        return Err(BackendError::Shape(format!(
            "split_rows: шард {shard} из {n_shards}"
        )));
    }
    if !rows.is_multiple_of(n_shards) {
        return Err(BackendError::Shape(format!(
            "split_rows: {rows} строк не делится на {n_shards}"
        )));
    }
    let part = rows / n_shards;
    // Строки лежат подряд — это непрерывный кусок того же буфера
    // (Metal: view без копии; CPU/CUDA: копия того же диапазона).
    Ok(w.view_offset(vec![part, cols], shard * part * cols))
}

/// Нарезка весов по СТОЛБЦАМ (строчный параллелизм: входные каналы).
/// Возвращает [rows, cols/n] — кусок, дающий ЧАСТИЧНУЮ сумму полного Y.
///
/// Столбцы в row-major не непрерывны, поэтому идёт host-перекладка: это
/// подготовка весов (один раз на загрузке/шардировании), не горячий путь.
pub fn split_cols(
    backend: &dyn Backend,
    w: &Tensor,
    shard: usize,
    n_shards: usize,
) -> Result<Tensor, BackendError> {
    let (rows, cols) = (w.shape()[0], w.shape()[1]);
    if n_shards == 0 || shard >= n_shards {
        return Err(BackendError::Shape(format!(
            "split_cols: шард {shard} из {n_shards}"
        )));
    }
    if !cols.is_multiple_of(n_shards) {
        return Err(BackendError::Shape(format!(
            "split_cols: {cols} столбцов не делится на {n_shards}"
        )));
    }
    let part = cols / n_shards;
    let src = backend.to_cpu_f32(w)?;
    let mut out = vec![0.0f32; rows * part];
    for r in 0..rows {
        let s = r * cols + shard * part;
        out[r * part..(r + 1) * part].copy_from_slice(&src[s..s + part]);
    }
    backend.from_cpu_f32(&out, vec![rows, part])
}

/// Собрать части выходных фич (all-gather колоночного шарда) в один
/// [rows, cols] — конкатенация по последней оси.
pub fn gather_cols(backend: &dyn Backend, parts: &[Tensor]) -> Result<Tensor, BackendError> {
    let Some(first) = parts.first() else {
        return Err(BackendError::Shape("gather_cols: пустой список".into()));
    };
    let (rows, part_cols) = (first.shape()[0], first.shape()[1]);
    let cols = part_cols * parts.len();
    let mut out = vec![0.0f32; rows * cols];
    for (i, p) in parts.iter().enumerate() {
        if p.shape() != [rows, part_cols] {
            return Err(BackendError::Shape(format!(
                "gather_cols: часть {i} формы {:?}, ждали [{rows}, {part_cols}]",
                p.shape()
            )));
        }
        let v = backend.to_cpu_f32(p)?;
        for r in 0..rows {
            let dst = r * cols + i * part_cols;
            out[dst..dst + part_cols].copy_from_slice(&v[r * part_cols..(r + 1) * part_cols]);
        }
    }
    backend.from_cpu_f32(&out, vec![rows, cols])
}

/// Сложить частичные суммы строчного шарда (all-reduce, покомпонентно) —
/// локальная половина операции: у себя частичные суммы уже есть, чужие
/// приезжают по сети и складываются сюда.
pub fn reduce_sum(
    backend: &dyn Backend,
    mine: &Tensor,
    theirs: &Tensor,
) -> Result<Tensor, BackendError> {
    if mine.shape() != theirs.shape() {
        return Err(BackendError::Shape(format!(
            "reduce_sum: формы {:?} и {:?}",
            mine.shape(),
            theirs.shape()
        )));
    }
    let a = backend.to_cpu_f32(mine)?;
    let b = backend.to_cpu_f32(theirs)?;
    let sum: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    backend.from_cpu_f32(&sum, mine.shape().to_vec())
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use crate::metal::MetalBackend;

    fn be() -> MetalBackend {
        MetalBackend::new().unwrap()
    }

    fn mk(backend: &dyn Backend, rows: usize, cols: usize, seed: f32) -> Tensor {
        let v: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32 * seed).sin()) * 0.5 + (i % 7) as f32 * 0.01)
            .collect();
        backend.from_cpu_f32(&v, vec![rows, cols]).unwrap()
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    /// Колоночный шард: каждая нода считает свою часть выходных фич, сборка
    /// частей == полный GEMM (бит-в-бит: K-цикл у частей тот же).
    #[test]
    fn column_shard_gather_equals_full_gemm() {
        let b = be();
        let backend: &dyn Backend = &b;
        let (m, k, n) = (4usize, 32usize, 64usize);
        let x = mk(backend, m, k, 0.317);
        let w = mk(backend, n, k, 0.113);

        let y_full = backend.zeros(vec![m, n]).unwrap();
        backend.gemm_nt(&x, &w, &y_full).unwrap();
        let full = backend.to_cpu_f32(&y_full).unwrap();

        let mut parts = Vec::new();
        for s in 0..2 {
            let ws = split_rows(backend, &w, s, 2).unwrap();
            assert_eq!(ws.shape(), [n / 2, k]);
            let ys = backend.zeros(vec![m, n / 2]).unwrap();
            backend.gemm_nt(&x, &ws, &ys).unwrap();
            parts.push(ys);
        }
        let gathered = gather_cols(backend, &parts).unwrap();
        let got = backend.to_cpu_f32(&gathered).unwrap();
        assert_eq!(got, full, "колоночный шард: сборка != полный GEMM");
    }

    /// Строчный шард: частичные суммы по входным каналам; их сложение
    /// (all-reduce) даёт полный GEMM. Порядок суммирования другой, поэтому
    /// сверка с допуском, а не бит-в-бит.
    #[test]
    fn row_shard_reduce_equals_full_gemm() {
        let b = be();
        let backend: &dyn Backend = &b;
        let (m, k, n) = (4usize, 32usize, 64usize);
        let x = mk(backend, m, k, 0.271);
        let w = mk(backend, n, k, 0.197);

        let y_full = backend.zeros(vec![m, n]).unwrap();
        backend.gemm_nt(&x, &w, &y_full).unwrap();
        let full = backend.to_cpu_f32(&y_full).unwrap();

        // x режется по входным каналам так же, как w по столбцам.
        let xv = backend.to_cpu_f32(&x).unwrap();
        let half = k / 2;
        let mut partial = Vec::new();
        for s in 0..2 {
            let ws = split_cols(backend, &w, s, 2).unwrap();
            assert_eq!(ws.shape(), [n, half]);
            let mut xs = vec![0.0f32; m * half];
            for r in 0..m {
                xs[r * half..(r + 1) * half]
                    .copy_from_slice(&xv[r * k + s * half..r * k + (s + 1) * half]);
            }
            let xt = backend.from_cpu_f32(&xs, vec![m, half]).unwrap();
            let ys = backend.zeros(vec![m, n]).unwrap();
            backend.gemm_nt(&xt, &ws, &ys).unwrap();
            partial.push(ys);
        }
        let sum = reduce_sum(backend, &partial[0], &partial[1]).unwrap();
        let got = backend.to_cpu_f32(&sum).unwrap();
        let d = max_diff(&got, &full);
        assert!(d < 1e-4, "строчный шард + all-reduce: |Δ| {d} > 1e-4");
    }

    /// Неверная нарезка — громкая ошибка (а не молча укороченный тензор).
    #[test]
    fn split_rejects_indivisible_shards() {
        let b = be();
        let backend: &dyn Backend = &b;
        let w = mk(backend, 6, 32, 0.11);
        assert!(split_rows(backend, &w, 0, 4).is_err(), "6 строк на 4 шарда");
        assert!(split_cols(backend, &w, 4, 4).is_err(), "шард вне диапазона");
        let w2 = mk(backend, 8, 30, 0.11);
        assert!(split_cols(backend, &w2, 0, 4).is_err(), "30 столбцов на 4");
    }
}
