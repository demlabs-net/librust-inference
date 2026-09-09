//! slice_last для qkv [21, 4096]: q (0, 2048), k (2048, 1024), v (3072, 1024).
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
fn slice_last_qkv_matches_cpu() {
    let bk = MetalBackend::new().unwrap();
    let (t, d) = (21usize, 4096usize);
    let x = genr(1, t * d);
    let xt = bk.from_cpu_f32(&x, vec![1, t, d]).unwrap();
    for (off, d0) in [(0usize, 2048usize), (2048, 1024), (3072, 1024)] {
        let y = rust_inference::nn::slice_last(&bk, &xt, d0, off).unwrap();
        let got = bk.to_cpu_f32(&y).unwrap();
        let mut maxd = 0.0f32;
        for r in 0..t {
            for i in 0..d0 {
                let want = x[r * d + off + i];
                let diff = (got[r * d0 + i] - want).abs();
                if diff > maxd {
                    maxd = diff;
                }
            }
        }
        assert!(maxd < 1e-4, "slice off={off} d0={d0}: max diff {maxd}");
    }
}
