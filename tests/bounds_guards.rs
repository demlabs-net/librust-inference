//! F16-extension guards: rope tables and embedding indices must fail loudly
//! on out-of-bounds access (the GPU kernels used to read OOB silently).
#![cfg(feature = "cpu")]
use rust_inference::Backend;
use rust_inference::cpu::CpuBackend;

#[test]
fn rope_short_table_returns_err() {
    let b = CpuBackend;
    let x = b.from_cpu_f32(&[0.5f32; 4 * 8], vec![4, 8]).unwrap();
    let y = b.zeros(vec![4, 8]).unwrap();
    // cos/sin cover only 2 positions, x has 4 → must Err, not OOB.
    let cos = b.from_cpu_f32(&[1.0f32; 2 * 4], vec![2, 4]).unwrap();
    let sin = b.from_cpu_f32(&[0.0f32; 2 * 4], vec![2, 4]).unwrap();
    let r = b.rope(&x, &cos, &sin, &y, 8);
    assert!(r.is_err(), "rope must reject short cos/sin");
}

#[test]
fn rope_head_short_table_returns_err() {
    let b = CpuBackend;
    let x = b.from_cpu_f32(&[0.5f32; 4 * 8], vec![1, 4, 8]).unwrap();
    let y = b.zeros(vec![1, 4, 8]).unwrap();
    let cos = b.from_cpu_f32(&[1.0f32; 2 * 4], vec![2, 4]).unwrap();
    let sin = b.from_cpu_f32(&[0.0f32; 2 * 4], vec![2, 4]).unwrap();
    assert!(b.rope_head(&x, &cos, &sin, &y, 1, 8).is_err());
}

#[test]
fn embedding_oob_index_returns_err() {
    let b = CpuBackend;
    let table = b.from_cpu_f32(&[0.25f32; 10 * 4], vec![10, 4]).unwrap();
    let y = b.zeros(vec![3, 4]).unwrap();
    // index 10 >= rows 10.
    let r = b.embedding(&[0u32, 10, 2], &table, &y);
    assert!(r.is_err(), "embedding must reject an out-of-range index");
    // in-range works.
    let y2 = b.zeros(vec![2, 4]).unwrap();
    b.embedding(&[0u32, 9], &table, &y2).unwrap();
}
