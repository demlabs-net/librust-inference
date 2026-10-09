//! Где время в GDN-слое (Ф8.1): изолированные замеры стадий decode-шага
//! 27B-гибрида — четыре проекции (Q5_K/IQ4_XS/F32×2) и элементная цепочка
//! (gates, conv, silu, slice, l2norm, delta_rule, norm_gated). Прогрев +
//! N повторов без синхронизаций между итерациями, wall/N на стадию.
#![cfg(feature = "metal")]

use rust_inference::backend::Backend as _;
use rust_inference::dtype::WeightFormat;
use rust_inference::metal::MetalBackend;
use std::time::Instant;

fn bench(be: &MetalBackend, name: &str, bytes: usize, reps: usize, mut f: impl FnMut()) {
    f(); // прогрев
    // Повторы — в ОДНОМ command buffer (begin_batch/end_batch): без этого
    // каждый вызов мерит commit+wait-пол (~0.2 мс), а не кернел.
    let t0 = Instant::now();
    be.begin_batch().unwrap();
    for _ in 0..reps {
        f();
    }
    be.end_batch().unwrap();
    let s = t0.elapsed().as_secs_f64() / reps as f64;
    println!(
        "{name:14} {:6.1} МБ | {:7.3} мс | {:6.1} ГБ/с",
        bytes as f64 / 1e6,
        s * 1e3,
        bytes as f64 / s / 1e9
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let be = MetalBackend::new()?;
    let reps: usize = 50;
    // Геометрия 27B-гибрида.
    let (dim, qkv_out, ssm_inner, hk, s, hv) =
        (5120usize, 10240usize, 6144usize, 16usize, 128usize, 48usize);
    let qk = hk * s; // 2048

    let h2 = be.from_cpu_f32(
        &(0..dim).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect::<Vec<_>>(),
        vec![1, dim],
    )?;

    // ── Проекции ───────────────────────────────────────────────────────────
    // Q5_K веса [10240,5120]: 176 Б/256.
    let mut b5 = Vec::with_capacity(qkv_out * dim / 256 * 176);
    for i in 0..qkv_out * dim / 256 {
        b5.extend_from_slice(&half::f16::from_f32(0.01).to_le_bytes());
        b5.extend_from_slice(&half::f16::from_f32(0.001).to_le_bytes());
        for j in 0..12 {
            b5.push(((i + j) % 61) as u8);
        }
        for j in 0..160 {
            b5.push(((i * 7 + j) % 251) as u8);
        }
    }
    let w_qkv = be.from_quantized(&b5, vec![qkv_out, dim], WeightFormat::Q5_K)?;
    // IQ4_XS z [6144,5120]: 136 Б/256.
    let mut b4 = Vec::with_capacity(ssm_inner * dim / 256 * 136);
    for i in 0..ssm_inner * dim / 256 {
        b4.extend_from_slice(&half::f16::from_f32(0.01).to_le_bytes());
        b4.extend_from_slice(&((i * 13) as u16).to_le_bytes());
        for j in 0..4 {
            b4.push(((i * 5 + j) % 61) as u8);
        }
        for j in 0..128 {
            b4.push(((i * 5 + j) % 253) as u8);
        }
    }
    let w_gate = be.from_quantized(&b4, vec![ssm_inner, dim], WeightFormat::Iq4Xs)?;
    // F32 beta/alpha [48,5120] и склеенные [96,5120].
    let w_beta = be.from_cpu_f32(&vec![0.001f32; hv * dim], vec![hv, dim])?;
    let w_alpha = be.from_cpu_f32(&vec![0.001f32; hv * dim], vec![hv, dim])?;
    let w_ba = be.from_cpu_f32(&vec![0.001f32; 2 * hv * dim], vec![2 * hv, dim])?;

    let y_qkv = be.zeros(vec![1, qkv_out])?;
    let y_z = be.zeros(vec![1, ssm_inner])?;
    let y_ba = be.zeros(vec![1, hv])?;
    let y_ba2 = be.zeros(vec![1, 2 * hv])?;
    bench(&be, "qkv Q5_K", b5.len(), reps, || {
        be.gemm_nt(&h2, &w_qkv, &y_qkv).unwrap();
    });
    bench(&be, "gate IQ4XS", b4.len(), reps, || {
        be.gemm_nt(&h2, &w_gate, &y_z).unwrap();
    });
    bench(&be, "beta F32", hv * dim * 4, reps, || {
        be.gemm_nt(&h2, &w_beta, &y_ba).unwrap();
    });
    bench(&be, "alpha F32", hv * dim * 4, reps, || {
        be.gemm_nt(&h2, &w_alpha, &y_ba).unwrap();
    });
    bench(&be, "ba F32 ×2", 2 * hv * dim * 4, reps, || {
        be.gemm_nt(&h2, &w_ba, &y_ba2).unwrap();
    });

    // ── Элементная цепочка ────────────────────────────────────────────────
    let qkv = be.zeros(vec![1, 1, qkv_out])?;
    let z = be.zeros(vec![1, 1, ssm_inner])?;
    let ba = be.zeros(vec![1, 1, 2 * hv])?;
    let dt_bias = be.zeros(vec![hv])?;
    let a = be.zeros(vec![hv])?;
    let conv_w = be.zeros(vec![qkv_out, 4])?;
    let conv_st = be.zeros(vec![qkv_out, 3])?;
    let norm_w = be.zeros(vec![ssm_inner])?;

    bench(&be, "gates", 0, reps, || {
        be.gdn_gates(&ba, &dt_bias, &a).unwrap();
    });
    // Корректность gates: ba = alpha|beta с известными значениями.
    let mut bav = vec![0.0f32; 2 * hv];
    for h in 0..hv {
        bav[h] = h as f32 * 0.1 - 1.0; // alpha
        bav[hv + h] = -(h as f32) * 0.1; // beta
    }
    let ba2 = be.from_cpu_f32(&bav, vec![1, 1, 2 * hv])?;
    let dtb = be.from_cpu_f32(&vec![0.5f32; hv], vec![hv])?;
    let av = be.from_cpu_f32(&vec![2.0f32; hv], vec![hv])?;
    let (gg, bb) = be.gdn_gates(&ba2, &dtb, &av)?;
    let gv = be.to_cpu_f32(&gg)?;
    let bv = be.to_cpu_f32(&bb)?;
    let mut maxd = 0.0f32;
    for h in 0..hv {
        let x = bav[h] + 0.5;
        let sp = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
        maxd = maxd.max((gv[h] - sp * 2.0).abs());
        maxd = maxd.max((bv[h] - 1.0 / (1.0 + (-bav[hv + h]).exp())).abs());
    }
    println!("gates контроль: |Δ| max {maxd}");
    assert!(maxd < 1e-5, "gates неверны");
    bench(&be, "conv", qkv_out * 4 * 2, reps, || {
        be.gdn_conv(&qkv, &conv_w, &conv_st).unwrap();
    });
    let (conv_out, _cst) = be.gdn_conv(&qkv, &conv_w, &conv_st)?;
    let act = be.zeros(conv_out.shape().to_vec())?;
    bench(&be, "silu", qkv_out * 4 * 2, reps, || {
        be.silu(&conv_out, &act).unwrap();
    });
    bench(&be, "slice q", qk * 4, reps, || {
        rust_inference::nn::slice_last(&be, &act, qk, 0).unwrap();
    });
    let q = rust_inference::nn::slice_last(&be, &act, qk, 0)?;
    let k = rust_inference::nn::slice_last(&be, &act, qk, qk)?;
    let v = rust_inference::nn::slice_last(&be, &act, ssm_inner, 2 * qk)?;
    bench(&be, "l2norm", qk * 4 * 2, reps, || {
        be.gdn_l2norm(&q, s, 1e-6).unwrap();
    });
    let gate = be.zeros(vec![1, 1, hv])?;
    let b2 = be.zeros(vec![1, 1, hv])?;
    let ssm = be.zeros(vec![hv, s, s])?;
    bench(&be, "delta_rule", hv * s * s * 4 * 2, reps, || {
        be.gdn_delta_rule(&q, &k, &v, &gate, &b2, &ssm, hk).unwrap();
    });
    let o = be.zeros(vec![1, 1, ssm_inner])?;
    bench(&be, "norm_gated", ssm_inner * 4 * 2, reps, || {
        be.gdn_norm_gated(&o, &z, &norm_w, 1e-6).unwrap();
    });
    // Корректность fused-хвоста (conv+silu+split+l2norm+стейт) против
    // хост-цепочки с тем же порядком арифметики.
    let cq = 2 * qk + ssm_inner;
    let mut rng = 0x243F6A8885A308D3u64;
    let mut rnd = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                ((rng >> 40) as f32 / 1e7) - 0.5
            })
            .collect()
    };
    let xf = rnd(cq);
    let wf = rnd(cq * 4);
    let sf = rnd(cq * 3);
    let xt = be.from_cpu_f32(&xf, vec![cq])?;
    let wt = be.from_cpu_f32(&wf, vec![cq, 4])?;
    let stt = be.from_cpu_f32(&sf, vec![cq, 3])?;
    let (yf, nsf) = be.gdn_conv_silu_split_norm(&xt, &wt, &stt, cq, 4, s, qk, 1e-6)?;
    let yfv = be.to_cpu_f32(&yf)?;
    let nsfv = be.to_cpu_f32(&nsf)?;
    let mut ywant = vec![0.0f32; cq];
    let mut nswant = vec![0.0f32; cq * 3];
    for ch in 0..cq {
        let mut acc = 0.0f32;
        for j in 0..4 {
            let v = if j < 3 { sf[ch * 3 + j] } else { xf[ch] };
            acc += wf[ch * 4 + j] * v;
        }
        ywant[ch] = acc / (1.0 + (-acc).exp());
        for j in 0..3 {
            nswant[ch * 3 + j] = if j + 1 < 3 { sf[ch * 3 + j + 1] } else { xf[ch] };
        }
    }
    for hbase in (0..2 * qk).step_by(s) {
        let mut ss = 0.0f32;
        for j in 0..s {
            ss += ywant[hbase + j] * ywant[hbase + j];
        }
        let inv = 1.0 / (ss + 1e-6).sqrt();
        for j in 0..s {
            ywant[hbase + j] *= inv;
        }
    }
    let maxd = yfv
        .iter()
        .zip(&ywant)
        .map(|(a, b)| (a - b).abs())
        .chain(nsfv.iter().zip(&nswant).map(|(a, b)| (a - b).abs()))
        .fold(0.0f32, f32::max);
    println!("fused-хвост контроль: |Δ| max {maxd}");
    assert!(maxd < 1e-5, "fused-хвост неверен");
    let xq = be.from_cpu_f32(&xf, vec![1, 1, cq])?;
    bench(&be, "fused tail", cq * 4 * 3, reps, || {
        be.gdn_conv_silu_split_norm(&xq, &wt, &stt, cq, 4, s, qk, 1e-6).unwrap();
    });
    // Корректность norm_gated: x,z,w с известными значениями.
    let xv: Vec<f32> = (0..ssm_inner).map(|i| ((i % 37) as f32 - 18.0) * 0.05).collect();
    let zv: Vec<f32> = (0..ssm_inner).map(|i| ((i % 23) as f32 - 11.0) * 0.2).collect();
    let wv: Vec<f32> = (0..ssm_inner).map(|i| 1.0 + (i % 7) as f32 * 0.1).collect();
    let xt = be.from_cpu_f32(&xv, vec![1, 1, ssm_inner])?;
    let zt = be.from_cpu_f32(&zv, vec![1, 1, ssm_inner])?;
    let wt = be.from_cpu_f32(&wv, vec![ssm_inner])?;
    let yg = be.gdn_norm_gated(&xt, &zt, &wt, 1e-6)?;
    let ygv = be.to_cpu_f32(&yg)?;
    let ss: f32 = xv.iter().map(|v| v * v).sum();
    let inv = 1.0 / (ss / ssm_inner as f32 + 1e-6).sqrt();
    let mut maxd = 0.0f32;
    for i in 0..ssm_inner {
        let zz = zv[i];
        let want = xv[i] * inv * wv[i] * (zz / (1.0 + (-zz).exp()));
        maxd = maxd.max((ygv[i] - want).abs());
    }
    println!("norm_gated контроль: |Δ| max {maxd}");
    println!("  got[0..4]: {:?} got[32]: {}", &ygv[0..4], ygv[32]);
    let ss2: f32 = xv.iter().map(|v| v * v).sum();
    let inv2 = 1.0 / (ss2 / ssm_inner as f32 + 1e-6).sqrt();
    println!("  want[0..4]: {:?}", (0..4).map(|i| { let zz = zv[i]; xv[i] * inv2 * wv[i] * (zz / (1.0 + (-zz).exp())) }).collect::<Vec<f32>>());
    assert!(maxd < 1e-4, "norm_gated неверен");
    // Корректность v4-gemm: F32 [8,512] против хоста.
    let kk = 512usize;
    let a8: Vec<f32> = (0..kk).map(|i| ((i % 31) as f32 - 15.0) * 0.02).collect();
    let b8: Vec<f32> = (0..8 * kk).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
    let at = be.from_cpu_f32(&a8, vec![1, kk])?;
    let bt = be.from_cpu_f32(&b8, vec![8, kk])?;
    let ct = be.zeros(vec![1, 8])?;
    be.gemm_nt(&at, &bt, &ct)?;
    let cv = be.to_cpu_f32(&ct)?;
    let mut maxd = 0.0f32;
    for n in 0..8 {
        let want: f32 = (0..kk).map(|k| a8[k] * b8[n * kk + k]).sum();
        maxd = maxd.max((cv[n] - want).abs());
    }
    println!("gemm v4 контроль: |Δ| max {maxd}");
    assert!(maxd < 1e-5, "gemm v4 неверен");
    Ok(())
}
