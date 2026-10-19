//! Побитовая сверка mm- и mv/mb-путей IQ4_XS при m=2..8 (окно спек-верификации).
//! INFER_IQ4XS_MM_MIN управляет порогом mm в gemm_nt_quantized: тест гоняет
//! каждую строку через mv (m=1, порог не срабатывает) и целиком через mm
//! (порог=2), сравнивая биты и |Δ|. Запуск: cargo run --release --features
//! metal --example iq4_mm_check
use rust_inference::backend::Backend as _;
use rust_inference::dtype::WeightFormat;
use rust_inference::metal::MetalBackend;

/// IQ4_XS: 136 Б/256 (d f16 | scales_h u16 | scales[4] | qs[128]) — синтетика.
fn iq4_xs_bytes(n: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n * k / 256 * 136);
    for _ in 0..n * k / 256 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        out.extend_from_slice(&half::f16::from_f32(0.004 + (s % 100) as f32 * 1e-4).to_le_bytes());
        out.extend_from_slice(&((s >> 3) as u16).to_le_bytes());
        for _ in 0..4 {
            out.push((s >> 11) as u8);
        }
        for j in 0..128 {
            out.push(((s >> (j % 48)) ^ (j as u64 * 7)) as u8);
        }
    }
    out
}

/// Q5_K: 176 Б/256 — d, dmin, scales[12], qh[32], qs[128].
fn q5_k_bytes(n: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n * k / 256 * 176);
    for _ in 0..n * k / 256 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        out.extend_from_slice(&half::f16::from_f32(0.004 + (s % 100) as f32 * 1e-4).to_le_bytes());
        out.extend_from_slice(&half::f16::from_f32(0.001).to_le_bytes());
        for j in 0..12 {
            out.push(((s >> (j % 40)) ^ (j as u64)) as u8);
        }
        for j in 0..32 {
            out.push(((s >> (j % 48)) ^ (j as u64 * 3)) as u8);
        }
        for j in 0..128 {
            out.push(((s >> (j % 50)) ^ (j as u64 * 7)) as u8);
        }
    }
    out
}

fn main() {
    let be = MetalBackend::new().unwrap();
    let mut seed: u64 = 0x9E3779B97F4A7C15;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 33) as f32) * 2.0 - 1.0
    };
    let (k, n) = (768usize, 512usize); // k % 256 == 0
    for (fmt_name, fmt, bytes) in [
        ("IQ4_XS", WeightFormat::Iq4Xs, iq4_xs_bytes(n, k, 0xBEEF)),
        ("Q5_K", WeightFormat::Q5_K, q5_k_bytes(n, k, 0x5EED)),
    ] {
        let wt = be.from_quantized(&bytes, vec![n, k], fmt).unwrap();
        for m in 2..=8usize {
            let x: Vec<f32> = (0..m * k).map(|_| rnd()).collect();
            let a = be.from_cpu_f32(&x, vec![m, k]).unwrap();
            // mv/mb-путь (дефолтный порог 9 не задет m≤8)
            // SAFETY: однопоточный бенч до старта потоков бэкенда
            unsafe { std::env::set_var("INFER_IQ4XS_MM_MIN", "9") };
            let cm = be.zeros(vec![m, n]).unwrap();
            be.gemm_nt(&a, &wt, &cm).unwrap();
            let ref_c = be.to_cpu_f32(&cm).unwrap();
            // mm-путь (порог 2)
            unsafe { std::env::set_var("INFER_IQ4XS_MM_MIN", "2") };
            let c2 = be.zeros(vec![m, n]).unwrap();
            be.gemm_nt(&a, &wt, &c2).unwrap();
            let got = be.to_cpu_f32(&c2).unwrap();
            unsafe { std::env::remove_var("INFER_IQ4XS_MM_MIN") };
            let bit_eq = got
                .iter()
                .zip(&ref_c)
                .all(|(g, r)| g.to_bits() == r.to_bits());
            let diffs = got
                .iter()
                .zip(&ref_c)
                .filter(|(g, r)| g.to_bits() != r.to_bits())
                .count();
            let max_abs = got
                .iter()
                .zip(&ref_c)
                .map(|(g, r)| (g - r).abs())
                .fold(0f32, f32::max);
            println!(
                "{fmt_name} m={m}: bit_eq={bit_eq} diffs={diffs}/{} max|Δ|={max_abs:.3e}",
                m * n
            );
        }
    }
}
