//! Q8_0/Q4_0 GEMM при малом M (окно верификации спека: 1+γ строк) против
//! matvec декода на формах Qwen3-8B. Матричный GEMM спроектирован под
//! префилл (тайл 32 строки): при M ≤ 8 он читает веса так же, как при 32,
//! но считает 4–32× лишнего, и forward 2..8 строк стоит ≈3× декода — спек
//! с любым драфтером медленнее обычной генерации. Цель — время M строк
//! ≈ времени одной (веса читаются один раз, как в kernel_mul_mv_ext ggml).
//! GEMM_DTYPE=q4_0 — тот же прогон на Q4_0 (у него mb- путь и mm-тайл
//! появились позже; M>8 там пока скалярный gemm_nt_q4_0).
#![cfg(feature = "metal")]

use rust_inference::backend::Backend;
use rust_inference::dtype::WeightFormat;
use rust_inference::metal::MetalBackend;
use std::time::Instant;

fn q8_0_bytes(n: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n * k / 32 * 34);
    for _ in 0..n * k / 32 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        out.extend_from_slice(&half::f16::from_f32(0.001 + (s % 1000) as f32 * 1e-5).to_le_bytes());
        for j in 0..32 {
            out.push(((s >> (j % 56)) as u8) ^ (j as u8));
        }
    }
    out
}

/// Q4_0: 18 Б на 32 значения (f16 d + 16 байт нибблов, каноническая
/// раскладка ggml: байт j — значения j (младший) и j+16 (старший)).
fn q4_0_bytes(n: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n * k / 32 * 18);
    for _ in 0..n * k / 32 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        out.extend_from_slice(&half::f16::from_f32(0.001 + (s % 1000) as f32 * 1e-5).to_le_bytes());
        for j in 0..16 {
            let lo = ((s >> (j % 56)) & 0x0F) as u8 ^ j as u8;
            let hi = ((s >> ((j + 3) % 56)) & 0x0F) as u8 ^ (j as u8);
            out.push((lo & 0x0F) | (hi << 4));
        }
    }
    out
}

/// Q4_K: 144 Б на 256 значений (d | dmin | scales[12] | qs[128]); d/dmin —
/// sane f16, остальное — псевдослучайные байты (перф-замер, не точность).
fn q4_k_bytes(n: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n * k / 256 * 144);
    for _ in 0..n * k / 256 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        out.extend_from_slice(&half::f16::from_f32(0.02).to_bits().to_le_bytes());
        out.extend_from_slice(&half::f16::from_f32(0.005).to_bits().to_le_bytes());
        for _ in 0..12 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            out.push((s & 0xFF) as u8);
        }
        for _ in 0..128 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            out.push((s & 0xFF) as u8);
        }
    }
    out
}

/// Q5_K: 176 Б (d | dmin | scales[12] | qh[32] | qs[128]).
fn q5_k_bytes(n: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n * k / 256 * 176);
    for _ in 0..n * k / 256 {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        out.extend_from_slice(&half::f16::from_f32(0.02).to_bits().to_le_bytes());
        out.extend_from_slice(&half::f16::from_f32(0.005).to_bits().to_le_bytes());
        for _ in 0..12 {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            out.push((s & 0xFF) as u8);
        }
        for _ in 0..32 {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            out.push((s & 0xFF) as u8);
        }
        for _ in 0..128 {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            out.push((s & 0xFF) as u8);
        }
    }
    out
}

/// Q6_K: 210 Б (ql[128] | qh[64] | scales[16] | d f16).
fn q6_k_bytes(n: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n * k / 256 * 210);
    for _ in 0..n * k / 256 {
        for _ in 0..192 {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            out.push((s & 0xFF) as u8);
        }
        for _ in 0..16 {
            out.push(((s & 0x3F) as i8 - 32) as u8);
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        }
        out.extend_from_slice(&half::f16::from_f32(0.01).to_bits().to_le_bytes());
    }
    out
}

/// IQ4_XS: 136 Б (d f16 | scales_h u16 | scales_l[4] | qs[128]).
fn iq4_xs_bytes(n: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n * k / 256 * 136);
    for _ in 0..n * k / 256 {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        out.extend_from_slice(&half::f16::from_f32(0.98).to_bits().to_le_bytes());
        out.extend_from_slice(&(s as u16).to_le_bytes());
        for _ in 0..4 {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            out.push((s & 0xFF) as u8);
        }
        for _ in 0..128 {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            out.push((s & 0xFF) as u8);
        }
    }
    out
}

/// NVFP4: 36 Б на 64 значения (4 UE4M3-скейла + qs[32]).
fn nvfp4_bytes(n: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n * k / 64 * 36);
    for _ in 0..n * k / 64 {
        for _ in 0..4 {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            out.push(120 + (s % 40) as u8);
        }
        for _ in 0..32 {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            out.push((s & 0xFF) as u8);
        }
    }
    out
}

/// IQ4_NL: 18 Б (f16 d + 16 байт нибблов).
fn iq4_nl_bytes(n: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n * k / 32 * 18);
    for _ in 0..n * k / 32 {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        out.extend_from_slice(&half::f16::from_f32(0.02).to_bits().to_le_bytes());
        for _ in 0..16 {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            out.push((s & 0xFF) as u8);
        }
    }
    out
}

/// MXFP4: 17 Б (E8M0 + 16 байт нибблов).
fn mxfp4_bytes(n: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut out = Vec::with_capacity(n * k / 32 * 17);
    for _ in 0..n * k / 32 {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        out.push(120 + (s % 16) as u8);
        for _ in 0..16 {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            out.push((s & 0xFF) as u8);
        }
    }
    out
}

fn main() {
    let be = MetalBackend::new().unwrap();
    let iters = 50;
    // GEMM_MS=2,5 — только эти M (для перебора конфигураций)
    let ms: Vec<usize> = std::env::var("GEMM_MS")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 2, 3, 4, 5, 8, 16, 32]);
    let mut total = vec![0f64; ms.len()];
    let dtype = std::env::var("GEMM_DTYPE").unwrap_or_else(|_| "q8_0".into());
    let (make_w, fmt): (fn(usize, usize, u64) -> Vec<u8>, WeightFormat) = match dtype.as_str() {
        "q4_0" => (q4_0_bytes, WeightFormat::Q4_0),
        "q4_k" => (q4_k_bytes, WeightFormat::Q4_K),
        "q5_k" => (q5_k_bytes, WeightFormat::Q5_K),
        "q6_k" => (q6_k_bytes, WeightFormat::Q6_K),
        "iq4_xs" => (iq4_xs_bytes, WeightFormat::Iq4Xs),
        "iq4_nl" => (iq4_nl_bytes, WeightFormat::Iq4Nl),
        "mxfp4" => (mxfp4_bytes, WeightFormat::Mxfp4),
        "nvfp4" => (nvfp4_bytes, WeightFormat::Nvfp4),
        _ => (q8_0_bytes, WeightFormat::Q8_0),
    };
    if dtype != "q8_0" {
        println!("{dtype}");
    }
    for (name, n, k, mult) in [
        ("wqkv", 6144usize, 4096usize, 36.0f64),
        ("wo", 4096, 4096, 36.0),
        ("w13", 24576, 4096, 36.0),
        ("w2", 4096, 12288, 36.0),
        ("lm_head", 151936, 4096, 1.0),
    ] {
        let w = be.from_quantized(&make_w(n, k, 7), vec![n, k], fmt).unwrap();
        let mut line = format!("{name:8} [{n}x{k}]");
        let mut base = 0.0;
        for (mi, &m) in ms.iter().enumerate() {
            let a = be
                .from_cpu_f32(&(0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect::<Vec<_>>(), vec![m, k])
                .unwrap();
            let c = be.zeros(vec![m, n]).unwrap();
            be.begin_batch().unwrap();
            be.gemm_nt(&a, &w, &c).unwrap();
            be.end_batch().unwrap();
            let t0 = Instant::now();
            be.begin_batch().unwrap();
            for _ in 0..iters {
                be.gemm_nt(&a, &w, &c).unwrap();
            }
            be.end_batch().unwrap();
            let dt = t0.elapsed().as_secs_f64() / iters as f64 * 1e3;
            if mi == 0 {
                base = dt;
            }
            total[mi] += dt * mult;
            line.push_str(&format!("  M={m}: {dt:.3} ({:.1}×)", dt / base));
        }
        println!("{line}");
    }
    // Сумма матмулов forward'а 8B (36 слоёв + lm_head), мс
    let tl: Vec<String> = ms.iter().zip(&total).map(|(m, t)| format!("M={m}: {t:.1}")).collect();
    println!("forward-матмулы 8B: {}", tl.join("  "));
}
