//! rmsnorm (обычный): rows=21 vs rows=1 — сверка строки.
#![cfg(feature = "metal")]
use rust_inference::backend::Backend;
use rust_inference::metal::MetalBackend;

fn genr(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 32) as f32 / (1u64 << 32) as f32) * 2.0 - 1.0
        })
        .collect()
}

#[test]
fn rmsnorm_rows21_matches_rows1() {
    let bk = MetalBackend::new().unwrap();
    let d = 2048usize;
    let x21 = genr(1, 21 * d);
    let w = genr(2, d);
    let xt = bk.from_cpu_f32(&x21, vec![21, d]).unwrap();
    let wt = bk.from_cpu_f32(&w, vec![d]).unwrap();
    let yt = bk.zeros(vec![21, d]).unwrap();
    bk.rmsnorm(&xt, &wt, &yt, 1e-6).unwrap();
    let got21 = bk.to_cpu_f32(&yt).unwrap();
    let row = &x21[20 * d..21 * d];
    let xt1 = bk.from_cpu_f32(row, vec![1, d]).unwrap();
    let yt1 = bk.zeros(vec![1, d]).unwrap();
    bk.rmsnorm(&xt1, &wt, &yt1, 1e-6).unwrap();
    let got1 = bk.to_cpu_f32(&yt1).unwrap();
    let mut maxd = 0.0f32;
    for i in 0..d {
        let diff = (got21[20 * d + i] - got1[i]).abs();
        if diff > maxd { maxd = diff; }
    }
    assert!(maxd < 1e-4, "rmsnorm rows21 vs rows1: max diff {maxd}");
}
