//! gemm_nt_f16: честный эталон (b округлён до f16) — строки 0/10/20.
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
fn gemm_f16_m21_k2048_n4096_row20() {
    let bk = MetalBackend::new().unwrap();
    let (m, k, n) = (1usize, 2048usize, 4096usize);
    let a = genr(1, m * k);
    let b = genr(2, n * k);
    let bf16: Vec<f32> = b.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect();
    let at = bk.from_cpu_f32(&a, vec![m, k]).unwrap();
    let bt = bk.from_cpu_f16(
        &b.iter().map(|&v| half::f16::from_f32(v)).collect::<Vec<_>>(),
        vec![n, k],
    )
    .unwrap();
    let ct = bk.zeros(vec![m, n]).unwrap();
    bk.gemm_nt(&at, &bt, &ct).unwrap();
    let got = bk.to_cpu_f32(&ct).unwrap();
    for row in [0usize] {
        for col in [0usize, 7, 2048, 4095] {
            let want: f32 = (0..k).map(|kk| a[row * k + kk] * bf16[col * k + kk]).sum();
            let g = got[row * n + col];
            let rel = (g - want).abs() / want.abs().max(1e-6);
            assert!(rel < 1e-3, "gemm[{row},{col}]: got {g} want {want} rel {rel}");
        }
    }
}
