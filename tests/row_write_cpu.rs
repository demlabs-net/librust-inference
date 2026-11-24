//! Примитив сборки батч-строк `Backend::row_write` (пакетный шаг по слотам).
//!
//! Смысл: строка `row` приёмника (1, cap, D) ← источник (1, 1, D). На CPU
//! `view_offset` КОПИРУЕТ, поэтому наивная запись «scale в вид» теряется —
//! оп обязан писать в само хранилище.
//!
//! `cargo test --features cpu --test row_write_cpu`

#![cfg(feature = "cpu")]

use rust_inference::Backend;
use rust_inference::cpu::CpuBackend;

#[test]
fn row_write_lands_in_named_row_only() {
    let be = CpuBackend;
    let dst = be.zeros(vec![1, 3, 4]).unwrap();
    let src = be
        .from_cpu_f32(&[7.0, 8.0, 9.0, 10.0], vec![1, 1, 4])
        .unwrap();
    be.row_write(&dst, &src, 1).unwrap();
    let got = be.to_cpu_f32(&dst).unwrap();
    assert_eq!(
        got,
        vec![0.0, 0.0, 0.0, 0.0, 7.0, 8.0, 9.0, 10.0, 0.0, 0.0, 0.0, 0.0]
    );
}

#[test]
fn row_write_last_row_and_overwrite() {
    let be = CpuBackend;
    let dst = be.zeros(vec![1, 2, 3]).unwrap();
    let a = be.from_cpu_f32(&[1.0, 2.0, 3.0], vec![1, 1, 3]).unwrap();
    let b = be.from_cpu_f32(&[4.0, 5.0, 6.0], vec![1, 1, 3]).unwrap();
    be.row_write(&dst, &a, 1).unwrap();
    be.row_write(&dst, &b, 1).unwrap();
    let got = be.to_cpu_f32(&dst).unwrap();
    assert_eq!(got, vec![0.0, 0.0, 0.0, 4.0, 5.0, 6.0]);
}

#[test]
fn row_write_rejects_bad_shapes() {
    let be = CpuBackend;
    let dst = be.zeros(vec![1, 2, 4]).unwrap();
    let wrong_width = be.from_cpu_f32(&[1.0, 2.0, 3.0], vec![1, 1, 3]).unwrap();
    assert!(be.row_write(&dst, &wrong_width, 0).is_err());
    let ok = be
        .from_cpu_f32(&[1.0, 2.0, 3.0, 4.0], vec![1, 1, 4])
        .unwrap();
    assert!(be.row_write(&dst, &ok, 2).is_err(), "строка за пределами");
}
