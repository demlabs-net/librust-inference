//! GPU↔CPU parity for the packed Q5_K_M / Q3_K_S upload path: upload the
//! weights via the speech crate's `upload_quant`, download the dequantized
//! values via `gpu.to_cpu_f32` and compare against the speech crate's
//! `q5k_decode`/`q3k_decode` of the SAME packed bytes. `upload_quant` packs
//! every 256-weight super-block with `q5k_encode`/`q3k_encode` (kb-major),
//! so re-encoding the same source rows reproduces the exact byte stream the
//! GPU tensor holds — the CUDA dequant mirror in `to_cpu_f32` must decode
//! it identically.
//!
//! Would have failed before the wave-6.5 dequant fixes: wrong per-block
//! scale reads, chunk-major qs indexing and hmask-bit collisions all decode
//! to different values than the reference.
//! `cargo test --release --features "cuda cpu" --test quant_kquants_parity -- --nocapture`

#![cfg(all(feature = "cuda", feature = "cpu"))]

use rust_inference::Backend;
use rust_inference::cuda::CudaBackend;
use rust_inference::harness::Lcg;
use rust_inference_speech::quant::k_quants;
use rust_inference_speech::quant::{WeightQuant, upload_quant};

fn make_gpu() -> Option<CudaBackend> {
    let r = std::panic::catch_unwind(|| CudaBackend::new(0));
    let err = match r {
        Ok(Ok(b)) => return Some(b),
        Ok(Err(e)) => e.to_string(),
        Err(p) => p
            .downcast_ref::<&str>()
            .copied()
            .unwrap_or("panic")
            .to_string(),
    };
    if std::env::var("CUDA_SMOKE_REQUIRED").is_ok() {
        panic!("CUDA device unavailable (CUDA_SMOKE_REQUIRED=1): {err}");
    }
    eprintln!("CUDA device unavailable, skipping: {err}");
    None
}

fn parity_case(quant: WeightQuant, rows: usize, per: usize, seed: u64) {
    let Some(gpu) = make_gpu() else { return };
    assert_eq!(per % 256, 0, "super-block count must be integral");

    // Lcg [-1, 1) data — exercises the negative-min Q5_K path and the
    // Q3_K hmask/sign path (all-positive data would saturate the min codes).
    let mut w = vec![0f32; rows * per];
    Lcg::new(seed).fill(&mut w);

    // The speech cache keys on (name, shape) WITHOUT the quant — distinct
    // names keep a future same-shape cross-quant case from colliding with
    // this one's cached tensor.
    let name = match quant {
        WeightQuant::Q5KM => "w_q5k",
        WeightQuant::Q3KS => "w_q3k",
        _ => unreachable!("parity_case supports Q5KM/Q3KS only"),
    };
    let b_g = upload_quant(&gpu, quant, name, &w, &[rows, per]).unwrap();
    gpu.synchronize().unwrap();
    let got = gpu.to_cpu_f32(&b_g).unwrap();
    assert_eq!(got.len(), rows * per);

    // Reference: decode the SAME packed bytes the upload packed (per
    // super-block, in source order — upload_quant's kb-major layout keeps
    // block (sb, r) contiguous, the dequant mirror indexes it back).
    let nsuper = per / 256;
    let mut bad = 0usize;
    for r in 0..rows {
        for sb in 0..nsuper {
            let src = &w[r * per + sb * 256..r * per + (sb + 1) * 256];
            let packed = match quant {
                WeightQuant::Q5KM => k_quants::q5k_encode(src),
                WeightQuant::Q3KS => k_quants::q3k_encode(src),
                _ => unreachable!("parity_case supports Q5KM/Q3KS only"),
            };
            let dec = match quant {
                WeightQuant::Q5KM => k_quants::q5k_decode(&packed),
                WeightQuant::Q3KS => k_quants::q3k_decode(&packed),
                _ => unreachable!("parity_case supports Q5KM/Q3KS only"),
            };
            for j in 0..256 {
                let g = got[r * per + sb * 256 + j];
                let c = dec[j];
                if g != c {
                    bad += 1;
                    if bad < 5 {
                        eprintln!(
                            "{quant:?} r={r} sb={sb} j={j}: gpu {g} != reference {c}",
                        );
                    }
                }
            }
        }
    }
    assert_eq!(
        bad, 0,
        "{quant:?} {rows}x{per}: gpu to_cpu_f32 != reference decode ({bad} elements)"
    );
    println!(
        "{quant:?} {rows}x{per}: gpu to_cpu_f32 == reference decode ({} elems)",
        rows * per
    );
}

#[test]
fn q5k_upload_dequant_matches_reference() {
    // k=768 like the STT GEMMs (3 super-blocks/row) and a wider row count
    // to cover the kb-major stride (rows are non-contiguous in the upload).
    parity_case(WeightQuant::Q5KM, 4, 768, 61);
    parity_case(WeightQuant::Q5KM, 17, 1024, 62);
}

#[test]
fn q3k_upload_dequant_matches_reference() {
    parity_case(WeightQuant::Q3KS, 4, 768, 63);
    parity_case(WeightQuant::Q3KS, 17, 1024, 64);
}
