//! Ф1.8 KV-cache: context-window roll (GPU compact via offset views) and
//! SWA masks — Metal backend (dev machine GPU-only rule).
//!
//! `cargo test --release --features metal --test kv_compact`

use rust_inference::backend::{Backend, Tensor};
use rust_inference::metal::MetalBackend;
use rust_inference::nn::{build_causal_mask, KvCache};

fn backend() -> MetalBackend {
    MetalBackend::new().expect("metal backend")
}

fn fill(bk: &dyn Backend, base: f32, n: usize, d: usize) -> Tensor {
    let data: Vec<f32> = (0..n * d).map(|i| base + (i % 7) as f32 * 0.25).collect();
    bk.from_cpu_f32(&data, vec![1, n, d]).unwrap()
}

#[test]
fn compact_drops_oldest_half_on_overflow() {
    let bk = backend();
    let mut cache = KvCache::new(&bk, 2, 4, rust_inference::dtype::KvFormat::F32, rust_inference::dtype::KvFormat::F32)
        .expect("cache");
    cache.set_max_cap(Some(8));
    let d = 8usize;
    // 10 single-token updates; max_cap 8 → compact fires at the 9th write
    // (pos 8 → keep 4), then tokens 8,9 append → pos 6, contents = tokens 4..9.
    for i in 0..10 {
        let k = fill(&bk, i as f32 * 10.0, 1, d);
        let v = fill(&bk, i as f32 * 100.0, 1, d);
        cache.update(&bk, &k, &v).expect("update");
    }
    assert_eq!(cache.pos, 6, "pos after compact");
    let kd = bk.to_cpu_f32(&cache.view()).unwrap();
    assert_eq!(kd.len(), 6 * d);
    // row 0 must be token 4: base 40.0
    let expected: Vec<f32> = (0..d).map(|i| 40.0 + (i % 7) as f32 * 0.25).collect();
    for (got, want) in kd[..d].iter().zip(&expected) {
        assert!((got - want).abs() < 1e-4, "k row0: got {got} want {want}");
    }
    // last row must be token 9: base 90.0
    let expected9: Vec<f32> = (0..d).map(|i| 90.0 + (i % 7) as f32 * 0.25).collect();
    for (got, want) in kd[5 * d..].iter().zip(&expected9) {
        assert!((got - want).abs() < 1e-4, "k row5: got {got} want {want}");
    }
    // v must roll the same way (row0 = token 4, base 400.0)
    let vd = bk.to_cpu_f32(&cache.view_v()).unwrap();
    let expected_v: Vec<f32> = (0..d).map(|i| 400.0 + (i % 7) as f32 * 0.25).collect();
    for (got, want) in vd[..d].iter().zip(&expected_v) {
        assert!((got - want).abs() < 1e-4, "v row0: got {got} want {want}");
    }
}

#[test]
fn no_compact_below_max_cap() {
    let bk = backend();
    let mut cache = KvCache::new(&bk, 1, 4, rust_inference::dtype::KvFormat::F32, rust_inference::dtype::KvFormat::F32)
        .expect("cache");
    cache.set_max_cap(Some(16));
    for i in 0..10 {
        cache
            .update(&bk, &fill(&bk, i as f32, 1, 4), &fill(&bk, i as f32 * 2.0, 1, 4))
            .unwrap();
    }
    assert_eq!(cache.pos, 10, "no roll below max_cap");
}

#[test]
fn swa_mask_hides_outside_window() {
    let bk = backend();
    let m = build_causal_mask(&bk, 3, 6, 0, Some(4)).unwrap();
    let d = bk.to_cpu_f32(&m).unwrap();
    let neg = f32::NEG_INFINITY;
    // tq=3, tkv=6, pos=0, window=4: j > i → -inf; j < i+1-4 → -inf.
    // row i=0: visible j=0 only (j>0 hidden) → [0, -inf, -inf, -inf, -inf, -inf]
    assert_eq!(d[0], 0.0);
    for j in 1..6 {
        assert_eq!(d[j], neg, "row0 col{j}");
    }
    // row i=1: j<=1 visible → [0, 0, -inf, -inf, -inf, -inf]
    assert_eq!(d[6], 0.0);
    assert_eq!(d[7], 0.0);
    assert_eq!(d[8], neg);
    // row i=2: window 4 keeps j >= 2+1-4 = -1 → j<=2 visible
    assert_eq!(d[12], 0.0);
    assert_eq!(d[13], 0.0);
    assert_eq!(d[14], 0.0);
    assert_eq!(d[15], neg);
}

#[test]
fn swa_mask_with_offset_pos() {
    let bk = backend();
    // pos=5 (continuation): token i sees j <= 5+i and j >= 5+i+1-4 = i+2.
    let m = build_causal_mask(&bk, 2, 8, 5, Some(4)).unwrap();
    let d = bk.to_cpu_f32(&m).unwrap();
    let neg = f32::NEG_INFINITY;
    // row0 (pos 5): visible j in [2..=5] → cols 0,1,6,7 = -inf
    for j in [0usize, 1, 6, 7] {
        assert_eq!(d[j], neg, "row0 col{j}");
    }
    assert_eq!(d[2], 0.0);
    assert_eq!(d[5], 0.0);
    // row1 (pos 6): visible j in [3..=6]
    assert_eq!(d[8 + 2], neg);
    assert_eq!(d[8 + 3], 0.0);
    assert_eq!(d[8 + 6], 0.0);
    assert_eq!(d[8 + 7], neg);
}
