//! Декод-матvec IQ4_XS (m=1) и окно спека (m=2..8) на формах 27B-гибрида:
//! сколько ГБ/с реально выжимают mv_iq4_xs / mv_iq4_xs_mb против потолка
//! памяти. M3 Max ≈ 400 ГБ/с теория, llama.cpp на IQ4_XS 27B даёт ~230.
#![cfg(feature = "metal")]

use rust_inference::backend::Backend;
use rust_inference::dtype::WeightFormat;
use rust_inference::metal::MetalBackend;
use std::time::Instant;

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

/// Q6_K: 210 Б/256 — ql[128], qh[64], scales[16], d (в конце).
/// Q4_K: 144 Б/256 — d, dmin, scales[12], qs[128].
fn q4_k_bytes(n: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n * k / 256 * 144);
    for _ in 0..n * k / 256 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        out.extend_from_slice(&half::f16::from_f32(0.004 + (s % 100) as f32 * 1e-4).to_le_bytes());
        out.extend_from_slice(&half::f16::from_f32(0.001).to_le_bytes());
        for j in 0..12 {
            out.push(((s >> (j % 40)) ^ (j as u64)) as u8);
        }
        for j in 0..128 {
            out.push(((s >> (j % 48)) ^ (j as u64 * 7)) as u8);
        }
    }
    out
}

fn q6_k_bytes(n: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n * k / 256 * 210);
    for _ in 0..n * k / 256 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        for j in 0..128 {
            out.push(((s >> (j % 50)) ^ (j as u64 * 7)) as u8);
        }
        for j in 0..64 {
            out.push(((s >> (j % 40)) ^ (j as u64 * 3)) as u8);
        }
        for j in 0..16 {
            out.push(((s >> (j % 30)) ^ (j as u64)) as u8);
        }
        out.extend_from_slice(&half::f16::from_f32(0.004 + (s % 100) as f32 * 1e-4).to_le_bytes());
    }
    out
}

fn main() {
    let be = MetalBackend::new().unwrap();
    let iters = 30;
    // Формы 27B-гибрида (dim=5120): ffn w13 [2·17408? — берём фактические],
    // GDN w_in/w_out, lm_head — из гистограммы файла HauhauCS.
    for (name, n, k, fmt) in [
        ("ffn.w13", 34816usize, 5120usize, WeightFormat::Iq4Xs),
        ("ffn.w2", 5120, 17408, WeightFormat::Iq4Xs),
        ("gdn.qkv", 10240, 5120, WeightFormat::Q5_K),
        ("gdn.gate", 6144, 5120, WeightFormat::Iq4Xs),
        ("gdn.w_out", 5120, 6144, WeightFormat::Q5_K),
        ("attn.q", 12288, 5120, WeightFormat::Iq4Xs),
        ("lm_head", 248320, 5120, WeightFormat::Q6_K),
        // Формы класса Q4_K_M 8–9B (GLM-4-9B: dim 4096, ffn 13696).
        ("k.ffn.w13", 27392usize, 4096usize, WeightFormat::Q4_K),
        ("k.ffn.w2", 13696, 4096, WeightFormat::Q4_K),
        ("k.attn.qkv", 6144, 4096, WeightFormat::Q4_K),
    ] {
        let raw = match fmt {
            WeightFormat::Q5_K => q5_k_bytes(n, k, 7),
            WeightFormat::Q6_K => q6_k_bytes(n, k, 7),
            WeightFormat::Q4_K => q4_k_bytes(n, k, 7),
            _ => iq4_xs_bytes(n, k, 7),
        };
        let w = be.from_quantized(&raw, vec![n, k], fmt).unwrap();
        let wbytes = raw.len();
        let mut line = format!("{name:9} [{n}x{k}] {:5.1} МБ", wbytes as f64 / 1e6);
        for m in [1usize, 2, 3, 4, 5, 6, 7, 8] {
            let a = be
                .from_cpu_f32(
                    &(0..m * k)
                        .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
                        .collect::<Vec<_>>(),
                    vec![m, k],
                )
                .unwrap();
            let c = be.zeros(vec![m, n]).unwrap();
            be.begin_batch().unwrap();
            for _ in 0..3 {
                be.gemm_nt(&a, &w, &c).unwrap();
            }
            be.end_batch().unwrap();
            let t0 = Instant::now();
            be.begin_batch().unwrap();
            for _ in 0..iters {
                be.gemm_nt(&a, &w, &c).unwrap();
            }
            be.end_batch().unwrap();
            let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
            // m ≤ 8 — mv/batch-mv путь: веса читаются на каждую строку
            // (k-расщеплённые кернелы переиспользуют их в L2, но считаем
            // пессимистично m раз). m > 8 — тайловый mul_mm: веса ОДИН раз,
            // плюс активации и выход.
            let traffic = if m > 8 {
                wbytes + m * k * 4 + m * n * 4
            } else {
                wbytes * m
            };
            let gbs = traffic as f64 / ms / 1e6;
            line.push_str(&format!(" | m={m}: {:6.2} мс {:5.0} ГБ/с", ms, gbs));
        }
        println!("{line}");
    }
}
