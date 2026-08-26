//! mhsa_rel_pos / mhsa_apply_weights vs the CPU oracle (both shift modes).
#![cfg(all(feature = "cuda", feature = "cpu"))]
//! `cargo test --release --features "cuda cpu" --test mhsa_rel_pos -- --nocapture`

use rust_inference::Backend;
use rust_inference::cpu::CpuBackend;
use rust_inference::cuda::CudaBackend;
use rust_inference::harness::Lcg;

fn run_case(mode: u32, t: usize, dk: usize, dp: usize, h: usize, seed: u64) {
    let gpu = CudaBackend::new(0).unwrap();
    let cpu = CpuBackend;
    let d = h * dk;
    let mut q = vec![0f32; t * d];
    Lcg::new(seed).fill(&mut q);
    let mut k = vec![0f32; t * d];
    Lcg::new(seed + 1).fill(&mut k);
    let mut v = vec![0f32; t * d];
    Lcg::new(seed + 2).fill(&mut v);
    let mut p = vec![0f32; (2 * t - 1) * h * dp];
    Lcg::new(seed + 3).fill(&mut p);
    // The position-query tensor (zipformer p-slice layout: [T, H·dp]).
    let mut qp = vec![0f32; t * h * dp];
    Lcg::new(seed + 6).fill(&mut qp);
    let mut u = vec![0f32; h * dk];
    Lcg::new(seed + 4).fill(&mut u);
    for x in u.iter_mut() {
        *x *= 0.01;
    }
    let mut vb = vec![0f32; h * dp];
    Lcg::new(seed + 5).fill(&mut vb);
    for x in vb.iter_mut() {
        *x *= 0.01;
    }

    let q_g = gpu.from_cpu_f32(&q, vec![t, d]).unwrap();
    let k_g = gpu.from_cpu_f32(&k, vec![t, d]).unwrap();
    let v_g = gpu.from_cpu_f32(&v, vec![t, d]).unwrap();
    let qp_g = gpu.from_cpu_f32(&qp, vec![t, h * dp]).unwrap();
    let p_g = gpu.from_cpu_f32(&p, vec![2 * t - 1, h * dp]).unwrap();
    let u_g = gpu.from_cpu_f32(&u, vec![h * dk]).unwrap();
    let vb_g = gpu.from_cpu_f32(&vb, vec![h * dp]).unwrap();
    let y_g = gpu.zeros(vec![t, d]).unwrap();
    let w_g = gpu.zeros(vec![h, t, t]).unwrap();
    gpu.mhsa_rel_pos(
        &q_g,
        &k_g,
        &v_g,
        &qp_g,
        &p_g,
        &u_g,
        &vb_g,
        &y_g,
        Some(&w_g),
        dk,
        dp,
        mode,
    )
    .unwrap();
    gpu.synchronize().unwrap();
    let got = gpu.to_cpu_f32(&y_g).unwrap();
    let wgot = gpu.to_cpu_f32(&w_g).unwrap();

    let q_c = cpu.from_cpu_f32(&q, vec![t, d]).unwrap();
    let k_c = cpu.from_cpu_f32(&k, vec![t, d]).unwrap();
    let v_c = cpu.from_cpu_f32(&v, vec![t, d]).unwrap();
    let qp_c = cpu.from_cpu_f32(&qp, vec![t, h * dp]).unwrap();
    let p_c = cpu.from_cpu_f32(&p, vec![2 * t - 1, h * dp]).unwrap();
    let u_c = cpu.from_cpu_f32(&u, vec![h * dk]).unwrap();
    let vb_c = cpu.from_cpu_f32(&vb, vec![h * dp]).unwrap();
    let y_c = cpu.zeros(vec![t, d]).unwrap();
    let w_c = cpu.zeros(vec![h, t, t]).unwrap();
    cpu.mhsa_rel_pos(
        &q_c,
        &k_c,
        &v_c,
        &qp_c,
        &p_c,
        &u_c,
        &vb_c,
        &y_c,
        Some(&w_c),
        dk,
        dp,
        mode,
    )
    .unwrap();
    let want = cpu.to_cpu_f32(&y_c).unwrap();
    let wwant = cpu.to_cpu_f32(&w_c).unwrap();

    let tol = 2e-3 + 2e-3 * want.iter().fold(0f32, |m, &x| m.max(x.abs()));
    let mut bad = 0;
    for i in 0..got.len() {
        let ok = got[i].is_finite() && want[i].is_finite() && (got[i] - want[i]).abs() <= tol;
        if !ok {
            bad += 1;
            if bad < 5 {
                eprintln!("mode {mode} y [{i}] gpu {} cpu {}", got[i], want[i]);
            }
        }
    }
    for i in 0..wgot.len() {
        let ok = wgot[i].is_finite() && wwant[i].is_finite() && (wgot[i] - wwant[i]).abs() <= tol;
        if !ok {
            bad += 1;
            if bad < 5 {
                eprintln!("mode {mode} w [{i}] gpu {} cpu {}", wgot[i], wwant[i]);
            }
        }
    }
    assert_eq!(
        bad, 0,
        "mhsa_rel_pos mode {mode} mismatch: {bad}, tol {tol}"
    );
    println!("mhsa_rel_pos mode {mode} (t={t} dk={dk} dp={dp} h={h}): ok");
}

#[test]
fn mhsa_rel_pos_xl_matches_cpu() {
    run_case(0, 17, 8, 8, 2, 41);
    run_case(0, 65, 64, 64, 4, 42);
}

#[test]
fn mhsa_rel_pos_zipformer_matches_cpu() {
    run_case(1, 17, 32, 4, 4, 43);
    run_case(1, 65, 24, 4, 8, 44);
}

#[test]
fn mhsa_apply_weights_matches_cpu() {
    let gpu = CudaBackend::new(0).unwrap();
    let cpu = CpuBackend;
    let (t, h, vh) = (17usize, 4usize, 6usize);
    let mut w = vec![0f32; h * t * t];
    Lcg::new(51).fill(&mut w);
    // row-normalize to look like softmax weights
    for hh in 0..h {
        for r in 0..t {
            let s: f32 = w[hh * t * t + r * t..hh * t * t + (r + 1) * t]
                .iter()
                .sum::<f32>()
                .abs()
                + 1.0;
            for c in 0..t {
                w[hh * t * t + r * t + c] /= s;
            }
        }
    }
    let mut x = vec![0f32; t * h * vh];
    Lcg::new(52).fill(&mut x);
    let w_g = gpu.from_cpu_f32(&w, vec![h, t, t]).unwrap();
    let x_g = gpu.from_cpu_f32(&x, vec![t, h * vh]).unwrap();
    let y_g = gpu.zeros(vec![t, h * vh]).unwrap();
    gpu.mhsa_apply_weights(&w_g, &x_g, &y_g, vh).unwrap();
    gpu.synchronize().unwrap();
    let got = gpu.to_cpu_f32(&y_g).unwrap();
    let w_c = cpu.from_cpu_f32(&w, vec![h, t, t]).unwrap();
    let x_c = cpu.from_cpu_f32(&x, vec![t, h * vh]).unwrap();
    let y_c = cpu.zeros(vec![t, h * vh]).unwrap();
    cpu.mhsa_apply_weights(&w_c, &x_c, &y_c, vh).unwrap();
    let want = cpu.to_cpu_f32(&y_c).unwrap();
    let tol = 2e-3 + 2e-3 * want.iter().fold(0f32, |m, &x| m.max(x.abs()));
    let mut bad = 0;
    for i in 0..got.len() {
        let ok = got[i].is_finite() && want[i].is_finite() && (got[i] - want[i]).abs() <= tol;
        if !ok {
            bad += 1;
            if bad < 5 {
                eprintln!("apply [{i}] gpu {} cpu {}", got[i], want[i]);
            }
        }
    }
    assert_eq!(bad, 0, "mhsa_apply_weights mismatch: {bad}, tol {tol}");
    println!("mhsa_apply_weights: ok");
}
