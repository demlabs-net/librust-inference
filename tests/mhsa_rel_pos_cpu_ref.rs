//! P17: mhsa_rel_pos mode 0 (GigaAM/NeMo Transformer-XL position shift)
//! vs the SEMANTIC reference pos[q][k] = bd[q][k−q+t−1] (0 outside the
//! band). The formula was verified bit-exact against gigaam_encoder.py
//! rel_shift (pad(bd,(1,0)) → view(2t,t) → drop row 0 → view(t,2t−1) →
//! [:,:,:,:t]) for t ∈ {2,3,5}. The old kernel/CPU indexing
//! (flat=(tk+1)·t+qt, divisor 2t) read the TRANSPOSED index — every pair
//! got the wrong p row.
#![cfg(feature = "cpu")]

use rust_inference::Backend;
use rust_inference::cpu::CpuBackend;
use rust_inference::harness::Lcg;

#[test]
fn mode0_matches_semantic_reference() {
    let cpu = CpuBackend;
    for t in [2usize, 3, 5, 8] {
        let dk = 3;
        let dp = 4;
        let h = 2;
        let d = h * dk;
        let mut q = vec![0f32; t * d];
        Lcg::new(11).fill(&mut q);
        let mut k = vec![0f32; t * d];
        Lcg::new(12).fill(&mut k);
        let mut v = vec![0f32; t * d];
        Lcg::new(13).fill(&mut v);
        let mut p = vec![0f32; (2 * t - 1) * h * dp];
        Lcg::new(14).fill(&mut p);
        let mut u = vec![0f32; h * dk];
        Lcg::new(15).fill(&mut u);
        let mut vb = vec![0f32; h * dp];
        Lcg::new(16).fill(&mut vb);
        for x in u.iter_mut() {
            *x *= 0.01;
        }
        for x in vb.iter_mut() {
            *x *= 0.01;
        }
        let scale = 1.0 / (dk as f32).sqrt();

        let mut qp = vec![0f32; t * h * dp];
        Lcg::new(17).fill(&mut qp);
        let q_t = cpu.from_cpu_f32(&q, vec![t, d]).unwrap();
        let k_t = cpu.from_cpu_f32(&k, vec![t, d]).unwrap();
        let v_t = cpu.from_cpu_f32(&v, vec![t, d]).unwrap();
        let qp_t = cpu.from_cpu_f32(&qp, vec![t, h * dp]).unwrap();
        let p_t = cpu.from_cpu_f32(&p, vec![2 * t - 1, h * dp]).unwrap();
        let u_t = cpu.from_cpu_f32(&u, vec![h * dk]).unwrap();
        let vb_t = cpu.from_cpu_f32(&vb, vec![h * dp]).unwrap();
        let y_t = cpu.zeros(vec![t, d]).unwrap();
        cpu.mhsa_rel_pos(
            &q_t, &k_t, &v_t, &qp_t, &p_t, &u_t, &vb_t, &y_t,
            None, dk, dp, 0, scale,
        )
        .unwrap();
        let y = cpu.to_cpu_f32(&y_t).unwrap();

        // Semantic reference (verified against gigaam_encoder.py):
        //   bd[q][k] = (q[q]+vb) · p[k−q+t−1]  (0 outside the band)
        //   scores = ((q+u)·kᵀ + bd) · scale;  y = softmax(scores)·v
        let mut want = vec![0f32; t * d];
        for hh in 0..h {
            for qt in 0..t {
                let mut scores = Vec::with_capacity(t);
                for tk in 0..t {
                    let mut ac = 0.0f32;
                    for j in 0..dk {
                        ac += (q[qt * d + hh * dk + j] + u[hh * dk + j])
                            * k[tk * d + hh * dk + j];
                    }
                    let off = tk as isize - qt as isize + t as isize - 1;
                    let mut bd = 0.0f32;
                    if off >= 0 && (off as usize) < 2 * t - 1 {
                        // min(dk, dp): q rows stride dk, p rows stride dp.
                        for j in 0..dk.min(dp) {
                            bd += (q[qt * d + hh * dk + j] + vb[hh * dp + j])
                                * p[off as usize * (h * dp) + hh * dp + j];
                        }
                    }
                    scores.push((ac + bd) * scale);
                }
                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = scores.iter().map(|s| (s - mx).exp()).sum();
                for tk in 0..t {
                    let w = (scores[tk] - mx).exp() / sum;
                    for j in 0..dk {
                        want[qt * d + hh * dk + j] += w * v[tk * d + hh * dk + j];
                    }
                }
            }
        }
        let tol = 5e-4 + 1e-4 * want.iter().fold(0f32, |m, &x| m.max(x.abs()));
        let mut bad = 0;
        for i in 0..y.len() {
            if (y[i] - want[i]).abs() > tol {
                bad += 1;
                if bad < 5 {
                    eprintln!("t={t} [{i}]: got {} want {}", y[i], want[i]);
                }
            }
        }
        assert_eq!(bad, 0, "mode-0 mismatch at t={t}: {bad} elements");
    }
}
