//! Golden-vector harness (Phase 2.2): every op of the contract runs on the
//! CPU reference backend against committed golden fixtures.
//!
//! Run with `REGEN_GOLDEN=1` to generate `tests/golden/*.json` from the
//! CPU oracle (existing files are overwritten only with `REGEN_GOLDEN_FORCE=1`);
//! the normal run compares. Once GPU backends land, they reuse
//! the same committed inputs/outputs (tolerance-based comparison).
#![cfg(feature = "cpu")]

use rust_inference::Backend;
use rust_inference::harness::{check_deterministic, check_op, upload};

#[test]
fn golden_gemm() {
    check_op(
        "gemm",
        1,
        &[("a", vec![4, 8]), ("b", vec![8, 4]), ("c", vec![4, 4])],
        |b, t| {
            b.gemm(&t[0], &t[1], &t[2]).unwrap();
            vec![("c".into(), vec![4, 4], b.to_cpu_f32(&t[2]).unwrap())]
        },
    );
}

#[test]
fn golden_gemm_nt() {
    check_op(
        "gemm_nt",
        2,
        &[("a", vec![4, 8]), ("b", vec![4, 8]), ("c", vec![4, 4])],
        |b, t| {
            b.gemm_nt(&t[0], &t[1], &t[2]).unwrap();
            vec![("c".into(), vec![4, 4], b.to_cpu_f32(&t[2]).unwrap())]
        },
    );
}

#[test]
fn golden_elementwise() {
    check_op(
        "add",
        3,
        &[("a", vec![2, 3]), ("b", vec![2, 3]), ("y", vec![2, 3])],
        |b, t| {
            b.add(&t[0], &t[1], &t[2]).unwrap();
            vec![("y".into(), vec![2, 3], b.to_cpu_f32(&t[2]).unwrap())]
        },
    );
    check_op(
        "mul",
        4,
        &[("a", vec![2, 3]), ("b", vec![2, 3]), ("y", vec![2, 3])],
        |b, t| {
            b.mul(&t[0], &t[1], &t[2]).unwrap();
            vec![("y".into(), vec![2, 3], b.to_cpu_f32(&t[2]).unwrap())]
        },
    );
    check_op(
        "sub",
        5,
        &[("a", vec![2, 3]), ("b", vec![2, 3]), ("y", vec![2, 3])],
        |b, t| {
            b.sub(&t[0], &t[1], &t[2]).unwrap();
            vec![("y".into(), vec![2, 3], b.to_cpu_f32(&t[2]).unwrap())]
        },
    );
    check_op(
        "div",
        6,
        &[("a", vec![2, 3]), ("b", vec![2, 3]), ("y", vec![2, 3])],
        |b, t| {
            // avoid div-by-zero: shift b into [0.5, 1.5)
            let bd = b.to_cpu_f32(&t[1]).unwrap();
            let shifted: Vec<f32> = bd
                .iter()
                .map(|v| (v * 0.25 + 1.0).abs().max(0.25))
                .collect();
            let b2 = upload(b, &shifted, vec![2, 3]);
            b.div(&t[0], &b2, &t[2]).unwrap();
            vec![("y".into(), vec![2, 3], b.to_cpu_f32(&t[2]).unwrap())]
        },
    );
}

#[test]
fn golden_scalar_and_bias() {
    check_op(
        "add_scalar",
        7,
        &[("x", vec![2, 3]), ("y", vec![2, 3])],
        |b, t| {
            b.add_scalar(&t[0], &t[1], 1.5).unwrap();
            vec![("y".into(), vec![2, 3], b.to_cpu_f32(&t[1]).unwrap())]
        },
    );
    check_op(
        "scale",
        8,
        &[("x", vec![2, 3]), ("y", vec![2, 3])],
        |b, t| {
            b.scale(&t[0], &t[1], 0.75).unwrap();
            vec![("y".into(), vec![2, 3], b.to_cpu_f32(&t[1]).unwrap())]
        },
    );
    check_op(
        "add_bias",
        9,
        &[("x", vec![2, 3]), ("bias", vec![3]), ("y", vec![2, 3])],
        |b, t| {
            b.add_bias(&t[0], &t[1], &t[2]).unwrap();
            vec![("y".into(), vec![2, 3], b.to_cpu_f32(&t[2]).unwrap())]
        },
    );
    check_op(
        "add_scaled",
        10,
        &[("a", vec![2, 3]), ("b", vec![2, 3]), ("y", vec![2, 3])],
        |b, t| {
            b.add_scaled(&t[0], &t[1], &t[2], 0.5).unwrap();
            vec![("y".into(), vec![2, 3], b.to_cpu_f32(&t[2]).unwrap())]
        },
    );
    check_op(
        "add_bias2d",
        11,
        &[
            ("x", vec![2, 4, 3]),
            ("b", vec![2, 4, 3]),
            ("y", vec![2, 4, 3]),
        ],
        |b, t| {
            b.add_bias2d(&t[0], &t[1], &t[2]).unwrap();
            vec![("y".into(), vec![2, 4, 3], b.to_cpu_f32(&t[2]).unwrap())]
        },
    );
}

#[test]
fn golden_math_ops() {
    check_op(
        "sqrt_log_max",
        12,
        &[("x", vec![2, 3]), ("y", vec![2, 3])],
        |b, t| {
            let xd = b.to_cpu_f32(&t[0]).unwrap();
            let pos: Vec<f32> = xd.iter().map(|v| v.abs() + 0.5).collect();
            let x2 = upload(b, &pos, vec![2, 3]);
            b.sqrt(&x2, &t[1]).unwrap();
            let sq = b.to_cpu_f32(&t[1]).unwrap();
            b.log(&x2, &t[1]).unwrap();
            let lg = b.to_cpu_f32(&t[1]).unwrap();
            b.max_scalar(&x2, &t[1], 1.0).unwrap();
            let mx = b.to_cpu_f32(&t[1]).unwrap();
            vec![
                ("sqrt".into(), vec![2, 3], sq),
                ("log".into(), vec![2, 3], lg),
                ("max".into(), vec![2, 3], mx),
            ]
        },
    );
}

#[test]
fn golden_activations() {
    check_op(
        "activations",
        13,
        &[("x", vec![2, 3]), ("a", vec![2, 3]), ("y", vec![2, 3])],
        |b, t| {
            let mut out = Vec::new();
            for (name, res) in [
                (
                    "relu",
                    b.relu(&t[0], &t[2]).and_then(|_| b.to_cpu_f32(&t[2])),
                ),
                (
                    "sigmoid",
                    b.sigmoid(&t[0], &t[2]).and_then(|_| b.to_cpu_f32(&t[2])),
                ),
                (
                    "tanh",
                    b.tanh(&t[0], &t[2]).and_then(|_| b.to_cpu_f32(&t[2])),
                ),
                (
                    "silu",
                    b.silu(&t[0], &t[2]).and_then(|_| b.to_cpu_f32(&t[2])),
                ),
                (
                    "gelu",
                    b.gelu_tanh(&t[0], &t[2]).and_then(|_| b.to_cpu_f32(&t[2])),
                ),
                (
                    "snake",
                    b.snake(&t[0], &t[1], &t[2])
                        .and_then(|_| b.to_cpu_f32(&t[2])),
                ),
            ] {
                out.push((name.to_string(), vec![2, 3], res.unwrap()));
            }
            out
        },
    );
}

#[test]
fn golden_norms() {
    check_op(
        "norms",
        14,
        &[
            ("x", vec![4, 6]),
            ("w", vec![6]),
            ("b", vec![6]),
            ("y", vec![4, 6]),
        ],
        |b, t| {
            b.rmsnorm(&t[0], &t[1], &t[3], 1e-5).unwrap();
            let r = b.to_cpu_f32(&t[3]).unwrap();
            b.layernorm(&t[0], &t[1], &t[2], &t[3], 1e-5).unwrap();
            let l = b.to_cpu_f32(&t[3]).unwrap();
            vec![
                ("rmsnorm".into(), vec![4, 6], r),
                ("layernorm".into(), vec![4, 6], l),
            ]
        },
    );
}

#[test]
fn golden_head_norms() {
    check_op(
        "head_norms",
        15,
        &[
            ("x", vec![4, 8]),
            ("w", vec![4]),
            ("b", vec![4]),
            ("y", vec![4, 8]),
        ],
        |b, t| {
            b.rmsnorm_head(&t[0], &t[1], &t[3], 2, 1e-5).unwrap();
            let r = b.to_cpu_f32(&t[3]).unwrap();
            b.layernorm_head(&t[0], &t[1], &t[2], &t[3], 2, 1e-5)
                .unwrap();
            let l = b.to_cpu_f32(&t[3]).unwrap();
            vec![
                ("rmsnorm_head".into(), vec![4, 8], r),
                ("layernorm_head".into(), vec![4, 8], l),
            ]
        },
    );
}

#[test]
fn golden_convs() {
    check_op(
        "conv1d",
        16,
        &[
            ("x", vec![2, 8, 4]),
            ("w", vec![6, 4, 3]),
            ("y", vec![2, 7, 6]),
        ],
        |b, t| {
            b.conv1d(&t[0], &t[1], None, 1, 1, 0, 1, &t[2]).unwrap();
            vec![("y".into(), vec![2, 7, 6], b.to_cpu_f32(&t[2]).unwrap())]
        },
    );
    check_op(
        "conv1d_stride",
        17,
        &[
            ("x", vec![2, 8, 4]),
            ("w", vec![6, 4, 3]),
            ("y", vec![2, 4, 6]),
        ],
        |b, t| {
            b.conv1d(&t[0], &t[1], None, 2, 1, 0, 1, &t[2]).unwrap();
            vec![("y".into(), vec![2, 4, 6], b.to_cpu_f32(&t[2]).unwrap())]
        },
    );
    check_op(
        "conv2d",
        18,
        &[
            ("x", vec![1, 2, 6, 6]),
            ("w", vec![3, 2, 3, 3]),
            ("y", vec![1, 3, 4, 4]),
        ],
        |b, t| {
            b.conv2d_nchw(&t[0], &t[1], None, 0, 0, 1, 1, false, &t[2])
                .unwrap();
            vec![("y".into(), vec![1, 3, 4, 4], b.to_cpu_f32(&t[2]).unwrap())]
        },
    );
    check_op(
        "maxpool",
        19,
        &[("x", vec![1, 2, 8, 8]), ("y", vec![1, 2, 4, 4])],
        |b, t| {
            b.maxpool2d(&t[0], &t[1], 2, 2).unwrap();
            vec![("y".into(), vec![1, 2, 4, 4], b.to_cpu_f32(&t[1]).unwrap())]
        },
    );
    check_op(
        "conv_transpose1d",
        20,
        &[
            ("x", vec![2, 4, 3]),
            ("w", vec![3, 5, 3]),
            ("y", vec![2, 9, 5]),
        ],
        |b, t| {
            b.conv_transpose1d(&t[0], &t[1], None, 2, &t[2]).unwrap();
            vec![("y".into(), vec![2, 9, 5], b.to_cpu_f32(&t[2]).unwrap())]
        },
    );
}

#[test]
fn golden_slices() {
    check_op(
        "slices",
        21,
        &[
            ("x", vec![3, 4]),
            ("yl", vec![3, 2]),
            ("xt", vec![2, 4, 3]),
            ("yt", vec![2, 3, 3]),
        ],
        |b, t| {
            b.slice_last(&t[0], &t[1], 2).unwrap();
            let sl = b.to_cpu_f32(&t[1]).unwrap();
            b.slice_time(&t[2], &t[3], 1).unwrap();
            let st = b.to_cpu_f32(&t[3]).unwrap();
            vec![
                ("slice_last".into(), vec![3, 2], sl),
                ("slice_time".into(), vec![2, 3, 3], st),
            ]
        },
    );
}

#[test]
fn golden_softmax_glu() {
    check_op(
        "softmax_glu",
        22,
        &[
            ("x", vec![4, 8]),
            ("y", vec![4, 8]),
            ("g", vec![4, 8]),
            ("gy", vec![4, 4]),
        ],
        |b, t| {
            b.softmax_last_dim(&t[0], &t[1]).unwrap();
            let s = b.to_cpu_f32(&t[1]).unwrap();
            b.glu(&t[2], &t[3]).unwrap();
            let g = b.to_cpu_f32(&t[3]).unwrap();
            vec![
                ("softmax".into(), vec![4, 8], s),
                ("glu".into(), vec![4, 4], g),
            ]
        },
    );
}

#[test]
fn golden_rope() {
    check_op(
        "rope",
        23,
        &[
            ("x", vec![6, 8]),
            ("cos", vec![6, 4]),
            ("sin", vec![6, 4]),
            ("y", vec![6, 8]),
        ],
        |b, t| {
            b.rope(&t[0], &t[1], &t[2], &t[3], 8).unwrap();
            vec![("rope".into(), vec![6, 8], b.to_cpu_f32(&t[3]).unwrap())]
        },
    );
    check_op(
        "rope_interleaved",
        24,
        &[
            ("x", vec![1, 6, 8]),
            ("cos", vec![6, 4]),
            ("sin", vec![6, 4]),
            ("y", vec![1, 6, 8]),
        ],
        |b, t| {
            b.rope_interleaved(&t[0], &t[1], &t[2], &t[3], 8, 0)
                .unwrap();
            vec![(
                "rope_interleaved".into(),
                vec![1, 6, 8],
                b.to_cpu_f32(&t[3]).unwrap(),
            )]
        },
    );
    check_op(
        "rope_head",
        25,
        &[
            ("x", vec![2, 6, 8]),
            ("cos", vec![6, 2]),
            ("sin", vec![6, 2]),
            ("y", vec![2, 6, 8]),
        ],
        |b, t| {
            b.rope_head(&t[0], &t[1], &t[2], &t[3], 2, 4).unwrap();
            vec![(
                "rope_head".into(),
                vec![2, 6, 8],
                b.to_cpu_f32(&t[3]).unwrap(),
            )]
        },
    );
}

#[test]
fn golden_mul_rows_vec_repeat() {
    check_op(
        "mul_rows",
        26,
        &[("x", vec![4, 5]), ("w", vec![4]), ("y", vec![4, 5])],
        |b, t| {
            b.mul_rows(&t[0], &t[1], &t[2]).unwrap();
            vec![("y".into(), vec![4, 5], b.to_cpu_f32(&t[2]).unwrap())]
        },
    );
    check_op(
        "mul_vec",
        27,
        &[("x", vec![4, 5]), ("w", vec![5]), ("y", vec![4, 5])],
        |b, t| {
            b.mul_vec(&t[0], &t[1], &t[2]).unwrap();
            vec![("y".into(), vec![4, 5], b.to_cpu_f32(&t[2]).unwrap())]
        },
    );
    check_op(
        "repeat",
        28,
        &[("x", vec![2, 3, 4]), ("y", vec![2, 6, 4])],
        |b, t| {
            b.repeat(&t[0], &t[1]).unwrap();
            vec![("y".into(), vec![2, 6, 4], b.to_cpu_f32(&t[1]).unwrap())]
        },
    );
}

#[test]
fn golden_attn_ar() {
    check_op(
        "attn_ar",
        29,
        &[
            ("q", vec![1, 4, 8]),
            ("k", vec![1, 6, 4]),
            ("v", vec![1, 6, 4]),
            ("y", vec![1, 4, 8]),
        ],
        |b, t| {
            b.attn_ar(&t[0], &t[1], &t[2], None, &t[3], 4, 2, 2)
                .unwrap();
            vec![("y".into(), vec![1, 4, 8], b.to_cpu_f32(&t[3]).unwrap())]
        },
    );
}

#[test]
fn golden_mhsa_uniform() {
    check_op(
        "mhsa",
        30,
        &[
            ("q", vec![1, 4, 8]),
            ("k", vec![1, 4, 8]),
            ("v", vec![1, 4, 8]),
            ("sc", vec![1, 2, 4, 4]),
            ("y", vec![1, 4, 8]),
        ],
        |b, t| {
            b.mhsa(&t[0], &t[1], &t[2], &t[3], None, 0, &t[4]).unwrap();
            vec![("y".into(), vec![1, 4, 8], b.to_cpu_f32(&t[4]).unwrap())]
        },
    );
    check_op(
        "uniform_attn",
        31,
        &[("v", vec![1, 4, 8]), ("y", vec![1, 8])],
        |b, t| {
            b.uniform_attn(&t[0], &t[1]).unwrap();
            vec![("y".into(), vec![1, 8], b.to_cpu_f32(&t[1]).unwrap())]
        },
    );
}

#[test]
fn golden_kv_update() {
    check_op(
        "kv_update",
        32,
        &[("dst", vec![1, 8, 4]), ("src", vec![1, 3, 4])],
        |b, t| {
            b.kv_update(&t[0], &t[1], 2).unwrap();
            vec![("dst".into(), vec![1, 8, 4], b.to_cpu_f32(&t[0]).unwrap())]
        },
    );
}

#[test]
fn golden_embedding() {
    check_op(
        "embedding",
        33,
        &[("table", vec![16, 6]), ("y", vec![5, 6])],
        |b, t| {
            let idx: Vec<u32> = (0..5).map(|i| (i * 3) % 16).collect();
            b.embedding(&idx, &t[0], &t[1]).unwrap();
            vec![("y".into(), vec![5, 6], b.to_cpu_f32(&t[1]).unwrap())]
        },
    );
}

#[test]
fn determinism_all_ops() {
    // Same-backend, same-seed → bit-identical output for every op family.
    check_deterministic(
        "gemm",
        40,
        &[("a", vec![4, 8]), ("b", vec![8, 4]), ("c", vec![4, 4])],
        |b, t| {
            b.gemm(&t[0], &t[1], &t[2]).unwrap();
            vec![("c".into(), vec![4, 4], b.to_cpu_f32(&t[2]).unwrap())]
        },
    );
    check_deterministic(
        "attn_ar",
        41,
        &[
            ("q", vec![1, 4, 8]),
            ("k", vec![1, 6, 4]),
            ("v", vec![1, 6, 4]),
            ("y", vec![1, 4, 8]),
        ],
        |b, t| {
            b.attn_ar(&t[0], &t[1], &t[2], None, &t[3], 4, 2, 2)
                .unwrap();
            vec![("y".into(), vec![1, 4, 8], b.to_cpu_f32(&t[3]).unwrap())]
        },
    );
    check_deterministic(
        "softmax",
        42,
        &[("x", vec![4, 8]), ("y", vec![4, 8])],
        |b, t| {
            b.softmax_last_dim(&t[0], &t[1]).unwrap();
            vec![("y".into(), vec![4, 8], b.to_cpu_f32(&t[1]).unwrap())]
        },
    );
    check_deterministic(
        "convs",
        43,
        &[
            ("x", vec![2, 8, 4]),
            ("w", vec![6, 4, 3]),
            ("y", vec![2, 7, 6]),
        ],
        |b, t| {
            b.conv1d(&t[0], &t[1], None, 1, 1, 0, 1, &t[2]).unwrap();
            vec![("y".into(), vec![2, 7, 6], b.to_cpu_f32(&t[2]).unwrap())]
        },
    );
}
