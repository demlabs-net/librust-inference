//! Изоляция Ф8.1c: прогон attn_fa_vec_f16 на ВХОДАХ оракула (Qcur/Kcur/Vcur
//! слоя 0 из дампа llama.cpp, m=12) и побитовая сверка с kqv_out оракула.
//! Если совпадает — кернел верен, расхождение в наших K/V (кэш); если нет —
//! баг в порте.
use rust_inference::backend::Backend as _;
use rust_inference::metal::MetalBackend;

fn load(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap();
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn main() {
    let be = MetalBackend::new().unwrap();
    let (tq, nh, nkv, hd) = (23usize, 32usize, 2usize, 128usize);
    let tkv = 23usize;
    let q = load("/tmp/hq.bin");
    let k = load("/tmp/hk.bin");
    let v = load("/tmp/hv.bin");
    let want = load("/tmp/hkqv.bin");
    let kh: Vec<half::f16> = k.iter().map(|x| half::f16::from_f32(*x)).collect();
    let vh: Vec<half::f16> = v.iter().map(|x| half::f16::from_f32(*x)).collect();
    let mut mask = vec![0.0f32; tq * tkv];
    for i in 0..tq {
        for j in 0..tkv {
            if j > i {
                mask[i * tkv + j] = f32::NEG_INFINITY;
            }
        }
    }
    let qt = be.from_cpu_f32(&q, vec![1, tq, nh * hd]).unwrap();
    let kt = be.from_cpu_f16(&kh, vec![1, tkv, nkv * hd]).unwrap();
    let vt = be.from_cpu_f16(&vh, vec![1, tkv, nkv * hd]).unwrap();
    let mt = be.from_cpu_f32(&mask, vec![1, tq, tkv]).unwrap();
    let y = be.zeros(vec![1, tq, nh * hd]).unwrap();
    be.attn_ar(&qt, &kt, &vt, Some(&mt), &y, nh, nkv, hd)
        .unwrap();
    let got = be.to_cpu_f32(&y).unwrap();
    let mut bad = 0usize;
    let mut worst = 0f32;
    for (a, b) in want.iter().zip(&got) {
        if a != b {
            bad += 1;
            let d = (a - b).abs();
            if d > worst {
                worst = d;
            }
        }
    }
    println!("bad={bad}/{} worst={worst:.2e}", want.len());
    for r in 0..tq {
        let d = want[r * 4096..(r + 1) * 4096]
            .iter()
            .zip(&got[r * 4096..(r + 1) * 4096])
            .filter(|(a, b)| a != b)
            .count();
        println!("row{r}: diffs={d}");
    }
}
