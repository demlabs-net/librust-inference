//! Синтетика для attn_fa_vec_f16: точная форма GLM-4-9B (tq=13, nh=32, nkv=2,
//! hd=128), причинная маска; сверка с numpy-эталоном (f16-скругления как в
//! кернеле). nkv=2 принципиален: K/V лежат interleaved [T, NKV*HD], stride
//! ключа = NKV*HD ≠ HD — при nkv=1 этот баг не воспроизводится.
use rust_inference::backend::Backend as _;
use rust_inference::metal::MetalBackend;

fn main() {
    let be = MetalBackend::new().unwrap();
    let (tq, nh, nkv, hd) = (13usize, 32usize, 2usize, 128usize);
    let tkv = 13usize;
    let mut q = vec![0f32; tq * nh * hd];
    let mut k = vec![half::f16::from_f32(0.0f32); tkv * nkv * hd];
    let mut v = vec![half::f16::from_f32(0.0f32); tkv * nkv * hd];
    let mut rng = 0x1234u32;
    let mut rnd = || {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        ((rng >> 8) as f32 / 8388608.0) - 1.0
    };
    for x in q.iter_mut() {
        *x = rnd();
    }
    for x in k.iter_mut() {
        *x = half::f16::from_f32(rnd());
    }
    for x in v.iter_mut() {
        *x = half::f16::from_f32(rnd());
    }
    let mut mask = vec![0.0f32; tq * tkv];
    for i in 0..tq { for j in 0..tkv { if j > i { mask[i * tkv + j] = f32::NEG_INFINITY; } } }
    let qt = be.from_cpu_f32(&q, vec![1, tq, nh * hd]).unwrap();
    let kt = be.from_cpu_f16(&k, vec![1, tkv, nkv * hd]).unwrap();
    let vt = be.from_cpu_f16(&v, vec![1, tkv, nkv * hd]).unwrap();
    let mt = be.from_cpu_f32(&mask, vec![1, tq, tkv]).unwrap();
    let y = be.zeros(vec![1, tq, nh * hd]).unwrap();
    use rust_inference::backend::Backend;
    be.attn_ar(&qt, &kt, &vt, Some(&mt), &y, nh, nkv, hd).unwrap();
    let got = be.to_cpu_f32(&y).unwrap();
    // эталон: для каждого токена и головы — softmax по разрешённым ключам
    let scale = 1.0f32 / (hd as f32).sqrt();
    let mut bad = 0usize;
    let mut worst = 0f32;
    for t in 0..tq {
        for h in 0..nh {
            let kvh = h / (nh / nkv);
            let mut scores = Vec::new();
            for j in 0..tkv {
                if mask[t * tkv + j] == 0.0 {
                    let s: f32 = (0..hd)
                        .map(|d| q[t * nh * hd + h * hd + d] * f32::from(k[j * nkv * hd + kvh * hd + d]))
                        .sum();
                    scores.push((j, s * scale));
                }
            }
            let m = scores.iter().map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = scores.iter().map(|(_, s)| (s - m).exp()).sum();
            for d in 0..hd {
                let e: f32 = scores
                    .iter()
                    .map(|(j, s)| (s - m).exp() * f32::from(v[j * nkv * hd + kvh * hd + d]))
                    .sum::<f32>()
                    / sum;
                let g = got[t * nh * hd + h * hd + d];
                let diff = (g - e).abs();
                if diff > 1e-4 {
                    bad += 1;
                    if diff > worst {
                        worst = diff;
                        if bad < 4 {
                            println!("t={t} h={h} d={d}: got={g} want={e}");
                        }
                    }
                }
            }
        }
    }
    println!("bad={bad} worst={worst}");
}
