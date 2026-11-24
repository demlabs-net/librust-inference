//! Ф1.8/1.6 KV-cache: context-window roll — REAL compaction (the data moves
//! down and the kept keys are re-rotated for their new absolute positions).
//!
//! `BACKEND=cuda|cpu|metal cargo test --release --test kv_compact`

use rust_inference::backend::default_backend;
use rust_inference::backend::{Backend, Tensor};
use rust_inference::nn::KvCache;

/// Skip gracefully when no backend feature/env is enabled (the plain
/// cargo test must not fail on a machine without BACKEND set).
fn backend() -> Option<std::sync::Arc<dyn Backend>> {
    default_backend().ok()
}

/// Чтение хранилища KV в f32 независимо от формата: Metal с Ф8.2f держит
/// f16-кэш НАСТОЯЩИМ half (to_cpu_f32 такой тензор отвергает — читаем
/// to_cpu_f16 и расширяем). Формат — по типу тензора, не по запрошенному:
/// CPU-бэкенд, например, приводил бы f16-запрос к f32-хранилищу.
fn read_kv_f32(b: &dyn Backend, t: &Tensor) -> Vec<f32> {
    match t.dtype() {
        rust_inference::dtype::Dtype::F16 => b
            .to_cpu_f16(t)
            .unwrap()
            .iter()
            .map(|&h| f32::from(h))
            .collect(),
        other => {
            assert_eq!(
                other,
                rust_inference::dtype::Dtype::F32,
                "KV dtype {other:?}"
            );
            b.to_cpu_f32(t).unwrap()
        }
    }
}

/// Standard rope for one (1, 1, D) tensor at an absolute position.
fn rope(b: &dyn Backend, x: &Tensor, pos: usize, hd: usize, theta: f32) -> Tensor {
    b.rope_standard(x, pos, hd, theta, None).expect("rope")
}

fn tok(b: &dyn Backend, i: usize, d: usize, base: f32) -> Tensor {
    let data: Vec<f32> = (0..d)
        .map(|j| base + (i * 13 + j * 7) as f32 * 0.03125)
        .collect();
    b.from_cpu_f32(&data, vec![1, 1, d]).unwrap()
}

/// The compaction must (a) move the newest half down, (b) re-rotate the
/// kept keys for their NEW absolute positions, and (c) keep the attention
/// over the rolled window EXACTLY equal to the un-rolled reference.
#[test]
fn compact_rolls_and_rerotates() {
    let Some(b) = backend() else {
        eprintln!("no backend (set BACKEND=cuda|cpu|metal) — skipping");
        return;
    };
    let (kv_heads, hd, theta) = (2usize, 8usize, 10000.0f32);
    let d = kv_heads * hd;
    let mut a = KvCache::new(
        b.as_ref(),
        kv_heads,
        hd,
        rust_inference::dtype::KvFormat::F32,
        rust_inference::dtype::KvFormat::F32,
        theta,
        None,
    )
    .expect("cache");
    a.set_max_cap(Some(8));
    let mut r = KvCache::new(
        b.as_ref(),
        kv_heads,
        hd,
        rust_inference::dtype::KvFormat::F32,
        rust_inference::dtype::KvFormat::F32,
        theta,
        None,
    )
    .expect("cache");

    // 12 tokens: the windowed cache rolls at the 9th, the reference grows.
    // The windowed cache's absolute positions reset at the roll — its
    // updates use the CACHE's current pos (the caller's pos_rope), the
    // reference keeps the absolute token index.
    for i in 0..12 {
        // caller-driven roll (the qwen3 flow): the rope position is read
        // AFTER the roll, so the token that triggers the compact is roped
        // at its NEW position.
        a.roll_if_needed(b.as_ref(), 1).expect("roll");
        let pos = a.pos;
        let k = rope(b.as_ref(), &tok(b.as_ref(), i, d, 1.0), pos, hd, theta);
        let v = tok(b.as_ref(), i, d, 10.0);
        a.update(b.as_ref(), &k, &v).expect("a update");
        let kr = rope(b.as_ref(), &tok(b.as_ref(), i, d, 1.0), i, hd, theta);
        r.update(b.as_ref(), &kr, &v).expect("r update");
    }
    // After the roll the window holds tokens 4..12 at slots [0..8).
    assert_eq!(a.pos, 8, "window pos after the roll");
    let ak = b.to_cpu_f32(&a.view()).unwrap();
    let av = b.to_cpu_f32(&a.view_v()).unwrap();
    let rk = b.to_cpu_f32(&r.view()).unwrap();
    let rv = b.to_cpu_f32(&r.view_v()).unwrap();
    // The VALUES are a pure move: slot j == reference slot j+4.
    for j in 0..8 {
        let s = &av[j * d..(j + 1) * d];
        let t = &rv[(j + 4) * d..(j + 5) * d];
        assert_eq!(s, t, "value move at slot {j}");
    }
    // The KEYS are the reference keys RE-ROTATED by −4 steps: the slot j
    // holds the token #(4+j) rotated for position j; the reference slot
    // j+4 holds the same token rotated for position j+4. Rotating the
    // reference slice by −4 must recover the compacted slot.
    for j in 0..8 {
        let s = &ak[j * d..(j + 1) * d];
        let t0 = &rk[(j + 4) * d..(j + 5) * d];
        let mut t = t0.to_vec();
        // The rope pair layout is INTERLEAVED HALVES (like the kernels):
        // pair pi = (t[pi], t[pi + hd/2]) within each head.
        for hh in 0..kv_heads {
            for pi in 0..hd / 2 {
                let ang = -4.0f32 * theta.powf(-2.0 * pi as f32 / hd as f32);
                let (c, sn) = (ang.cos(), ang.sin());
                let base = hh * hd;
                let (x, y) = (t[base + pi], t[base + pi + hd / 2]);
                t[base + pi] = x * c - y * sn;
                t[base + pi + hd / 2] = x * sn + y * c;
            }
        }
        let max_d = s
            .iter()
            .zip(&t)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(max_d < 1e-4, "key re-rotation at slot {j}: {max_d}");
    }

    // Attention equivalence: the same query attends the rolled window (A,
    // positions reset, mask over [0..8)) and the reference window (B, the
    // last 8 slots, the first 4 masked). The score for the token #(4+j) is
    // R(8)·R(j) in A and R(12)·R(4+j) in B — equal by the RoPE identity.
    let qbase = tok(b.as_ref(), 99, d, 3.0);
    let qa = rope(b.as_ref(), &qbase, 8, hd, theta);
    let qb = rope(b.as_ref(), &qbase, 12, hd, theta);
    let mask_a = b.from_cpu_f32(&[0.0f32; 8], vec![1, 1, 8]).unwrap();
    let mut mb = vec![f32::NEG_INFINITY; 12];
    mb[4..].fill(0.0);
    let mask_b = b.from_cpu_f32(&mb, vec![1, 1, 12]).unwrap();
    let ya = b.zeros(vec![1, 1, d]).unwrap();
    let yb = b.zeros(vec![1, 1, d]).unwrap();
    b.attn_ar(
        &qa,
        &a.view(),
        &a.view_v(),
        Some(&mask_a),
        &ya,
        kv_heads,
        kv_heads,
        hd,
    )
    .expect("attn a");
    b.attn_ar(
        &qb,
        &r.view(),
        &r.view_v(),
        Some(&mask_b),
        &yb,
        kv_heads,
        kv_heads,
        hd,
    )
    .expect("attn b");
    let ga = b.to_cpu_f32(&ya).unwrap();
    let gb = b.to_cpu_f32(&yb).unwrap();
    let max_d = ga
        .iter()
        .zip(&gb)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_d < 1e-5,
        "attention over the rolled window diverges: {max_d}"
    );
}

/// The roll must work for the quantized KV storages too (f16, q8_0): the
/// values move exactly (a pure copy), and the attention over the rolled
/// window stays close to the f32 reference (the storage precision).
#[test]
fn compact_rolls_quantized_formats() {
    let Some(b) = backend() else {
        eprintln!("no backend (set BACKEND=cuda|cpu|metal) — skipping");
        return;
    };
    // d must be a multiple of 32 for the q8_0 storage, and the head dim
    // must be >= 32 (the q8 block lives inside one head): 2 heads × 32.
    let (kv_heads, hd, theta) = (2usize, 32usize, 10000.0f32);
    let d = kv_heads * hd;
    // q8_0 is lossy at write (i8 + per-block scale) — its value tolerance
    // covers the quantization error; the shift itself is an exact copy.
    for (name, kf, vf, vtol) in [
        (
            "f16",
            rust_inference::dtype::KvFormat::F16,
            rust_inference::dtype::KvFormat::F16,
            1e-2f32,
        ),
        (
            "q8",
            rust_inference::dtype::KvFormat::Q8_0,
            rust_inference::dtype::KvFormat::Q8_0,
            0.2f32,
        ),
    ] {
        let mut a = KvCache::new(b.as_ref(), kv_heads, hd, kf, vf, theta, None).expect("cache");
        a.set_max_cap(Some(8));
        for i in 0..12 {
            a.roll_if_needed(b.as_ref(), 1).expect("roll");
            let pos = a.pos;
            let k = rope(b.as_ref(), &tok(b.as_ref(), i, d, 1.0), pos, hd, theta);
            a.update(b.as_ref(), &k, &tok(b.as_ref(), i, d, 10.0))
                .expect("update");
        }
        assert_eq!(a.pos, 8, "{name}: window pos after the roll");
        let av = read_kv_f32(b.as_ref(), &a.view_v());
        for j in 0..8 {
            // The values are a pure move: slot j == slot j+4 of the ORIGINAL
            // sequence — the exactness holds across the storage formats (the
            // shift copies the quantized blocks verbatim).
            let expect = tok(b.as_ref(), j + 4, d, 10.0);
            let got = &av[j * d..(j + 1) * d];
            let e = b.to_cpu_f32(&expect).unwrap();
            let max_d = got
                .iter()
                .zip(&e)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            assert!(max_d < vtol, "{name}: value move at slot {j}: {max_d}");
        }
        // The attention over the rolled window must be close to the f32
        // reference (the storage precision, not the compaction, dominates).
        let qa = rope(b.as_ref(), &tok(b.as_ref(), 99, d, 3.0), 8, hd, theta);
        let mask = b.from_cpu_f32(&[0.0f32; 8], vec![1, 1, 8]).unwrap();
        let ya = b.zeros(vec![1, 1, d]).unwrap();
        b.attn_ar(
            &qa,
            &a.view(),
            &a.view_v(),
            Some(&mask),
            &ya,
            kv_heads,
            kv_heads,
            hd,
        )
        .expect("attn");
        // The reference: the same 8 tokens in a fresh f32 cache at [0..8).
        let mut r = KvCache::new(
            b.as_ref(),
            kv_heads,
            hd,
            rust_inference::dtype::KvFormat::F32,
            rust_inference::dtype::KvFormat::F32,
            theta,
            None,
        )
        .expect("cache");
        for i in 4..12 {
            let k = rope(b.as_ref(), &tok(b.as_ref(), i, d, 1.0), i - 4, hd, theta);
            r.update(b.as_ref(), &k, &tok(b.as_ref(), i, d, 10.0))
                .expect("update");
        }
        let qb = rope(b.as_ref(), &tok(b.as_ref(), 99, d, 3.0), 8, hd, theta);
        let yb = b.zeros(vec![1, 1, d]).unwrap();
        b.attn_ar(
            &qb,
            &r.view(),
            &r.view_v(),
            Some(&mask),
            &yb,
            kv_heads,
            kv_heads,
            hd,
        )
        .expect("attn ref");
        let ga = b.to_cpu_f32(&ya).unwrap();
        let gb = b.to_cpu_f32(&yb).unwrap();
        let max_d = ga
            .iter()
            .zip(&gb)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_d < 0.5,
            "{name}: attention over the rolled window diverges: {max_d}"
        );
    }
}
