//! Ф1.11 vision layout ops on the Metal backend: merge2x2_add, group4,
//! concat4 — каждый кернел сверяется с CPU-референсом (same math, host
//! side). GPU-only правило: Metal обязателен, CPU — только эталон.
//!
//! `cargo test --release --features metal --test vision_ops`

#![cfg(feature = "metal")]

use rust_inference::backend::{Backend, Tensor};
use rust_inference::metal::MetalBackend;

fn backend() -> MetalBackend {
    MetalBackend::new().expect("metal backend")
}

fn cpu_merge2x2_add(x: &[f32], bias: &[f32], pos: &[f32], per: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; per * per * d];
    let hp = per / 2;
    for by in 0..hp {
        for bx in 0..hp {
            let dst_block = (by * hp + bx) * 4 * d;
            let srcs = [
                (2 * by * per + 2 * bx) * d,
                (2 * by * per + 2 * bx + 1) * d,
                ((2 * by + 1) * per + 2 * bx) * d,
                ((2 * by + 1) * per + 2 * bx + 1) * d,
            ];
            for (j, &src) in srcs.iter().enumerate() {
                for k in 0..d {
                    out[dst_block + j * d + k] = x[src + k] + bias[k] + pos[src + k];
                }
            }
        }
    }
    out
}

#[test]
fn merge2x2_add_matches_cpu() {
    let bk = backend();
    let (per, d) = (8usize, 16usize);
    let n = per * per;
    let x: Vec<f32> = (0..n * d).map(|i| (i % 31) as f32 * 0.5 - 3.0).collect();
    let bias: Vec<f32> = (0..d).map(|i| i as f32 * 0.25 - 2.0).collect();
    let pos: Vec<f32> = (0..n * d).map(|i| ((i * 7) % 29) as f32 * 0.125 - 1.0).collect();
    let want = cpu_merge2x2_add(&x, &bias, &pos, per, d);

    let xt = bk.from_cpu_f32(&x, vec![n, d]).unwrap();
    let bt = bk.from_cpu_f32(&bias, vec![d]).unwrap();
    let pt = bk.from_cpu_f32(&pos, vec![n, d]).unwrap();
    let yt = bk.zeros(vec![n, d]).unwrap();
    bk.merge2x2_add(&xt, &bt, &pt, &yt, per).unwrap();
    let got = bk.to_cpu_f32(&yt).unwrap();

    for i in 0..want.len() {
        assert!(
            (got[i] - want[i]).abs() < 1e-4,
            "merge2x2_add[{i}]: got {} want {}",
            got[i],
            want[i]
        );
    }
}

fn cpu_group4(x: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; (n / 4) * 4 * d];
    for i in 0..n / 4 {
        for j in 0..4 {
            let src = (i * 4 + j) * d;
            let dst = i * 4 * d + j * d;
            out[dst..dst + d].copy_from_slice(&x[src..src + d]);
        }
    }
    out
}

#[test]
fn group4_matches_cpu() {
    let bk = backend();
    let (n, d) = (24usize, 8usize);
    let x: Vec<f32> = (0..n * d).map(|i| (i % 13) as f32 * 0.75 - 4.0).collect();
    let want = cpu_group4(&x, n, d);

    let xt = bk.from_cpu_f32(&x, vec![n, d]).unwrap();
    let yt = bk.zeros(vec![n / 4, 4 * d]).unwrap();
    bk.group4(&xt, &yt).unwrap();
    let got = bk.to_cpu_f32(&yt).unwrap();

    assert_eq!(got.len(), want.len());
    for i in 0..want.len() {
        assert!((got[i] - want[i]).abs() < 1e-4, "group4[{i}]: got {} want {}", got[i], want[i]);
    }
}

fn cpu_concat4(chunks: &[&[f32]], nq: usize, proj: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; nq * 4 * proj];
    for (j, c) in chunks.iter().enumerate().take(4) {
        for q in 0..nq {
            let src = q * proj;
            let dst = (q * 4 + j) * proj;
            out[dst..dst + proj].copy_from_slice(&c[src..src + proj]);
        }
    }
    out
}

#[test]
fn concat4_matches_cpu() {
    let bk = backend();
    let (nq, proj) = (6usize, 8usize);
    let c0: Vec<f32> = (0..nq * proj).map(|i| i as f32).collect();
    let c1: Vec<f32> = (0..nq * proj).map(|i| 100.0 + i as f32).collect();
    let c2: Vec<f32> = (0..nq * proj).map(|i| 200.0 + i as f32).collect();
    let c3: Vec<f32> = (0..nq * proj).map(|i| 300.0 + i as f32).collect();
    let want = cpu_concat4(&[&c0, &c1, &c2, &c3], nq, proj);

    let ts: Vec<Tensor> = [c0, c1, c2, c3]
        .iter()
        .map(|c| bk.from_cpu_f32(c, vec![nq, proj]).unwrap())
        .collect();
    let yt = bk.zeros(vec![nq * 4, proj]).unwrap();
    bk.concat4(&ts, &yt).unwrap();
    let got = bk.to_cpu_f32(&yt).unwrap();

    for i in 0..want.len() {
        assert!(
            (got[i] - want[i]).abs() < 1e-4,
            "concat4[{i}]: got {} want {}",
            got[i],
            want[i]
        );
    }
}

#[test]
fn concat4_three_chunks_matches_cpu() {
    let bk = backend();
    let (nq, proj) = (4usize, 6usize);
    let c0: Vec<f32> = (0..nq * proj).map(|i| i as f32).collect();
    let c1: Vec<f32> = (0..nq * proj).map(|i| 10.0 + i as f32).collect();
    let c2: Vec<f32> = (0..nq * proj).map(|i| 20.0 + i as f32).collect();
    // 3 чанка: строка (q*4+3) остаётся нулевой (не тронута кернелом)
    let want = cpu_concat4(&[&c0, &c1, &c2], nq, proj);

    let ts: Vec<Tensor> = [c0, c1, c2]
        .iter()
        .map(|c| bk.from_cpu_f32(c, vec![nq, proj]).unwrap())
        .collect();
    let yt = bk.zeros(vec![nq * 4, proj]).unwrap();
    bk.concat4(&ts, &yt).unwrap();
    let got = bk.to_cpu_f32(&yt).unwrap();

    for i in 0..want.len() {
        assert!(
            (got[i] - want[i]).abs() < 1e-4,
            "concat4-3[{i}]: got {} want {}",
            got[i],
            want[i]
        );
    }
}
