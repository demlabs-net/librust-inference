//! Решающий тест Ф8.1g: точные входы оракула (K/V-кэш и post-rope Q с шага
//! декода tkv=1034) → наш attn_ar → сверка kqv_out побитово. Разделяет
//! «баг кернела» от «наши входы отличаются».
use rust_inference::backend::Backend as _;
use rust_inference::metal::MetalBackend;

fn rd(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap();
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn ulp_hist(got: &[f32], want: &[f32]) {
    let mut hist = std::collections::BTreeMap::new();
    for (g, w) in got.iter().zip(want) {
        if g.to_bits() != w.to_bits() {
            let d = g.to_bits() as i64 - w.to_bits() as i64;
            *hist.entry(d).or_insert(0) += 1;
        }
    }
    println!("ulp_hist: {:?}", hist);
}

fn bad_positions(got: &[f32], want: &[f32]) {
    let v: Vec<(usize, usize)> = (0..want.len())
        .filter(|&i| got[i].to_bits() != want[i].to_bits())
        .map(|i| (i / 128, i % 128))
        .collect();
    println!("bad_positions: {:?}", v);
}

fn main() {
    let be = MetalBackend::new().unwrap();
    let dir = std::env::var("FA_DIR").unwrap_or_else(|_| "/tmp/llama_ref".into());
    let q = rd(&format!("{dir}/dec_q.bin"));
    let k = rd(&format!("{dir}/dec_k.bin"));
    let v = rd(&format!("{dir}/dec_v.bin"));
    let ywant = rd(&format!("{dir}/dec_y.bin"));
    let (nh, nkv, hd) = (32usize, 2usize, 128usize);
    let tq = 1usize;
    let tkv_cap: usize = std::env::var("FA_TKV_CAP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);
    let tkv = (k.len() / (nkv * hd)).min(tkv_cap);
    eprintln!("tkv={tkv} q={} k={} v={}", q.len(), k.len(), v.len());
    let kh: Vec<half::f16> = k.iter().map(|x| half::f16::from_f32(*x)).collect();
    let vh: Vec<half::f16> = v.iter().map(|x| half::f16::from_f32(*x)).collect();
    let qt = be.from_cpu_f32(&q, vec![1, tq, nh * hd]).unwrap();
    let kt = be.from_cpu_f16(&kh, vec![1, tkv, nkv * hd]).unwrap();
    let vt = be.from_cpu_f16(&vh, vec![1, tkv, nkv * hd]).unwrap();
    let y = be.zeros(vec![1, tq, nh * hd]).unwrap();
    // два прогона: без маски (наш декод) и с нулевой маской (как у оракула)
    for (label, mask) in [
        ("nomask", None),
        (
            "zeromask",
            Some(
                be.from_cpu_f32(&vec![0.0f32; tq * tkv], vec![1, tq, tkv])
                    .unwrap(),
            ),
        ),
    ] {
        if std::env::var("FA_BATCH").is_ok() {
            be.begin_batch().unwrap();
        }
        be.attn_ar(&qt, &kt, &vt, mask.as_ref(), &y, nh, nkv, hd)
            .unwrap();
        if std::env::var("FA_BATCH").is_ok() {
            be.end_batch().unwrap();
        }
        let got = be.to_cpu_f32(&y).unwrap();
        let bad = (0..ywant.len())
            .filter(|i| got[*i].to_bits() != ywant[*i].to_bits())
            .count();
        let worst = (0..ywant.len())
            .map(|i| (got[i] - ywant[i]).abs())
            .fold(0f32, f32::max);
        println!("{label}: bad={bad}/{} worst={worst:e}", ywant.len());
        if std::env::var("FA_SAVE").is_ok() && label == "nomask" {
            let bytes: Vec<u8> = got.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write("/tmp/llama_ref/iso_y.bin", bytes).unwrap();
        }
        if std::env::var("FA_ULP").is_ok() {
            ulp_hist(&got, &ywant);
        }
        if std::env::var("FA_POS").is_ok() {
            bad_positions(&got, &ywant);
        }
        if std::env::var("FA_DIAG").is_ok() {
            let mut byh = vec![0usize; nh];
            for i in 0..ywant.len() {
                if got[i].to_bits() != ywant[i].to_bits() {
                    byh[i / hd] += 1;
                }
            }
            println!(
                "by head: {:?}",
                byh.iter()
                    .enumerate()
                    .filter(|(_, c)| **c > 0)
                    .collect::<Vec<_>>()
            );
        }
    }
}
// (diag appended below via env FA_DIAG)
