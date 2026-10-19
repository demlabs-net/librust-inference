//! Синтетика для mv_q5_0_sg: K=32 (один блок), N=16. Дельта-входы по
//! элементам — сверка деквант-маппинга поштучно.
use rust_inference::backend::Backend as _;

fn main() {
    let be = rust_inference::metal::MetalBackend::new().unwrap();
    let (k, n) = (32usize, 16usize);
    // Блок row: d = 1.0 (half), qh = row-й паттерн, qs = 0x10 * 16 →
    // lo nibble 0, hi nibble 1. qh = 0x0000FFFF → элементы 0..15 получают
    // +16 (bit4), 16..31 без. Тогда deq(e<16) = (0|16)-16 = 0;
    // deq(e>=16) = (1|0)-16 = -15.
    let mut w = Vec::new();
    for row in 0..n {
        w.extend_from_slice(&[0x00u8, 0x3c]); // d = 1.0f16
        let qh: u32 = if row % 2 == 0 {
            0x0000_FFFF
        } else {
            0xFFFF_0000
        };
        w.extend_from_slice(&qh.to_le_bytes());
        for _ in 0..16 {
            w.push(0x10);
        }
    }
    let wt = be
        .from_quantized(&w, vec![n, k], rust_inference::dtype::WeightFormat::Q5_0)
        .unwrap();
    for e in [0usize, 1, 15, 16, 17, 31] {
        let mut x = vec![0f32; k];
        x[e] = 1.0;
        let xt = be.from_cpu_f32(&x, vec![1, k]).unwrap();
        let c = be.zeros(vec![1, n]).unwrap();
        be.gemm_nt(&xt, &wt, &c).unwrap();
        let got = be.to_cpu_f32(&c).unwrap();
        println!(
            "e={e:2}: {:?}",
            got.iter().take(4).map(|v| *v).collect::<Vec<_>>()
        );
    }
    // Ожидаем для e<16 row0: (0|16 bit? lo nibble=0, qh bit e=1 → 16)-16 = 0;
    // e>=16: hi nibble=1, qh bit=0 → 1-16 = -15.
}
