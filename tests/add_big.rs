//! add_f32 для больших тензоров (21×2048) — сверка с CPU.
#![cfg(feature = "metal")]
use rust_inference::backend::Backend;
use rust_inference::metal::MetalBackend;

#[test]
fn add_big_matches_cpu() {
    let bk = MetalBackend::new().unwrap();
    let n = 21 * 2048usize;
    let a: Vec<f32> = (0..n).map(|i| (i % 997) as f32 * 0.001 - 0.5).collect();
    let b: Vec<f32> = (0..n).map(|i| (i % 503) as f32 * 0.002 - 1.0).collect();
    let at = bk.from_cpu_f32(&a, vec![1, 21, 2048]).unwrap();
    let bt = bk.from_cpu_f32(&b, vec![1, 21, 2048]).unwrap();
    let yt = bk.zeros(vec![1, 21, 2048]).unwrap();
    bk.add(&at, &bt, &yt).unwrap();
    let got = bk.to_cpu_f32(&yt).unwrap();
    let mut maxd = 0.0f32;
    let mut bad = 0;
    for i in 0..n {
        let want = a[i] + b[i];
        let d = (got[i] - want).abs();
        if d > maxd { maxd = d; }
        if d > 1e-4 && bad < 5 { bad += 1; eprintln!("add[{i}]: got {} want {}", got[i], want); }
    }
    assert!(maxd < 1e-3, "add max diff {maxd}");
}
