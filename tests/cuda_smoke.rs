//! CUDA backend smoke tests (`cuda` + `cpu` features): real GPU round-trips
//! vs the CPU reference oracle across the whole contract surface. Skipped
//! (with a message) when no CUDA device is available on the host — set
//! `CUDA_SMOKE_REQUIRED=1` to make a missing/broken device FAIL instead
//! (verification runs; plain `cargo test` on GPU-less CI keeps skipping).

#![cfg(all(feature = "cuda", feature = "cpu"))]

use rust_inference::Backend;
use rust_inference::cpu::CpuBackend;
use rust_inference::cuda::CudaBackend;
use rust_inference::harness::Lcg;

fn make_gpu() -> Option<CudaBackend> {
    match CudaBackend::new(0) {
        Ok(b) => Some(b),
        Err(e) => {
            if std::env::var("CUDA_SMOKE_REQUIRED").is_ok() {
                panic!("CUDA device unavailable (CUDA_SMOKE_REQUIRED=1): {e}");
            }
            eprintln!("CUDA device unavailable, skipping: {e}");
            None
        }
    }
}

/// Run `f` on both backends and compare with rtol/atol.
fn both<F>(gpu: &CudaBackend, cpu: &CpuBackend, f: F, rtol: f32, atol: f32, what: &str)
where
    F: Fn(&CudaBackend) -> Vec<f32> + Fn(&CpuBackend) -> Vec<f32>,
{
    let g = f(gpu);
    let c = f(cpu);
    assert_eq!(g.len(), c.len(), "{what}: length mismatch");
    for (i, (a, b)) in g.iter().zip(c.iter()).enumerate() {
        let ok = (a - b).abs() <= atol + rtol * b.abs();
        assert!(ok, "{what}[{i}]: gpu {a} != cpu {b}");
    }
}

fn lcg_tensor<B: Backend>(b: &B, shape: Vec<usize>, seed: u64) -> rust_inference::Tensor {
    let n: usize = shape.iter().product();
    let mut data = vec![0.0f32; n];
    Lcg::new(seed).fill(&mut data);
    b.from_cpu_f32(&data, shape).unwrap()
}

#[test]
fn cuda_elementwise_vs_cpu() {
    let Some(gpu) = make_gpu() else { return };
    let cpu = CpuBackend;
    let n = 257usize; // not a multiple of block size
    let x = lcg_tensor(&gpu, vec![n], 1);
    let xc = lcg_tensor(&cpu, vec![n], 1);
    let y = lcg_tensor(&gpu, vec![n], 2);
    let yc = lcg_tensor(&cpu, vec![n], 2);

    let go = gpu.zeros(vec![n]).unwrap();
    let co = cpu.zeros(vec![n]).unwrap();

    gpu.add(&x, &y, &go).unwrap();
    cpu.add(&xc, &yc, &co).unwrap();
    assert!(close(
        &gpu.to_cpu_f32(&go).unwrap(),
        &cpu.to_cpu_f32(&co).unwrap()
    ));

    gpu.scale(&x, &go, 0.75).unwrap();
    cpu.scale(&xc, &co, 0.75).unwrap();
    assert!(close(
        &gpu.to_cpu_f32(&go).unwrap(),
        &cpu.to_cpu_f32(&co).unwrap()
    ));

    gpu.silu(&x, &go).unwrap();
    cpu.silu(&xc, &co).unwrap();
    assert!(close(
        &gpu.to_cpu_f32(&go).unwrap(),
        &cpu.to_cpu_f32(&co).unwrap()
    ));

    gpu.gelu_tanh(&x, &go).unwrap();
    cpu.gelu_tanh(&xc, &co).unwrap();
    assert!(close(
        &gpu.to_cpu_f32(&go).unwrap(),
        &cpu.to_cpu_f32(&co).unwrap()
    ));
}

#[test]
fn cuda_norms_softmax_vs_cpu() {
    let Some(gpu) = make_gpu() else { return };
    let cpu = CpuBackend;
    let x = lcg_tensor(&gpu, vec![4, 8], 3);
    let xc = lcg_tensor(&cpu, vec![4, 8], 3);
    let w = lcg_tensor(&gpu, vec![8], 4);
    let wc = lcg_tensor(&cpu, vec![8], 4);
    let b = lcg_tensor(&gpu, vec![8], 5);
    let bc = lcg_tensor(&cpu, vec![8], 5);

    let go = gpu.zeros(vec![4, 8]).unwrap();
    let co = cpu.zeros(vec![4, 8]).unwrap();
    gpu.rmsnorm(&x, &w, &go, 1e-5).unwrap();
    cpu.rmsnorm(&xc, &wc, &co, 1e-5).unwrap();
    assert!(close(
        &gpu.to_cpu_f32(&go).unwrap(),
        &cpu.to_cpu_f32(&co).unwrap()
    ));
    gpu.layernorm(&x, &w, &b, &go, 1e-5).unwrap();
    cpu.layernorm(&xc, &wc, &bc, &co, 1e-5).unwrap();
    assert!(close(
        &gpu.to_cpu_f32(&go).unwrap(),
        &cpu.to_cpu_f32(&co).unwrap()
    ));
    gpu.softmax_last_dim(&x, &go).unwrap();
    cpu.softmax_last_dim(&xc, &co).unwrap();
    assert!(close(
        &gpu.to_cpu_f32(&go).unwrap(),
        &cpu.to_cpu_f32(&co).unwrap()
    ));

    // head norms: (rows, h·hd) with h=2, hd=4
    let xh = lcg_tensor(&gpu, vec![4, 8], 6);
    let xhc = lcg_tensor(&cpu, vec![4, 8], 6);
    let wh = lcg_tensor(&gpu, vec![4], 7);
    let whc = lcg_tensor(&cpu, vec![4], 7);
    let bh = lcg_tensor(&gpu, vec![4], 8);
    let bhc = lcg_tensor(&cpu, vec![4], 8);
    gpu.rmsnorm_head(&xh, &wh, &go, 2, 1e-5).unwrap();
    cpu.rmsnorm_head(&xhc, &whc, &co, 2, 1e-5).unwrap();
    assert!(close(
        &gpu.to_cpu_f32(&go).unwrap(),
        &cpu.to_cpu_f32(&co).unwrap()
    ));
    gpu.layernorm_head(&xh, &wh, &bh, &go, 2, 1e-5).unwrap();
    cpu.layernorm_head(&xhc, &whc, &bhc, &co, 2, 1e-5).unwrap();
    assert!(close(
        &gpu.to_cpu_f32(&go).unwrap(),
        &cpu.to_cpu_f32(&co).unwrap()
    ));
}

#[test]
fn cuda_convs_slices_vs_cpu() {
    let Some(gpu) = make_gpu() else { return };
    let cpu = CpuBackend;
    // conv1d causal: x (1,6,4), w (5,4,3), pad 1 → tout = 6+1-2-1+1 = 5
    let x = lcg_tensor(&gpu, vec![1, 6, 4], 9);
    let xc = lcg_tensor(&cpu, vec![1, 6, 4], 9);
    let w = lcg_tensor(&gpu, vec![5, 4, 3], 10);
    let wc = lcg_tensor(&cpu, vec![5, 4, 3], 10);
    let go = gpu.zeros(vec![1, 5, 5]).unwrap();
    let co = cpu.zeros(vec![1, 5, 5]).unwrap();
    gpu.conv1d(&x, &w, None, 1, 1, 0, 1, &go).unwrap();
    cpu.conv1d(&xc, &wc, None, 1, 1, 0, 1, &co).unwrap();
    assert!(
        close(&gpu.to_cpu_f32(&go).unwrap(), &cpu.to_cpu_f32(&co).unwrap()),
        "conv1d"
    );

    // conv_transpose1d: x (1,4,2), w (2,3,3), stride 2 → tout = (4-1)*2+3 = 9
    let xt = lcg_tensor(&gpu, vec![1, 4, 2], 11);
    let xtc = lcg_tensor(&cpu, vec![1, 4, 2], 11);
    let wt = lcg_tensor(&gpu, vec![2, 3, 3], 12);
    let wtc = lcg_tensor(&cpu, vec![2, 3, 3], 12);
    let gct = gpu.zeros(vec![1, 9, 3]).unwrap();
    let cct = cpu.zeros(vec![1, 9, 3]).unwrap();
    gpu.conv_transpose1d(&xt, &wt, None, 2, &gct).unwrap();
    cpu.conv_transpose1d(&xtc, &wtc, None, 2, &cct).unwrap();
    assert!(
        close(
            &gpu.to_cpu_f32(&gct).unwrap(),
            &cpu.to_cpu_f32(&cct).unwrap()
        ),
        "conv_transpose1d"
    );

    // slices — contract is 2D (rows, D) → (rows, d0); 3D callers flatten
    let xs = lcg_tensor(&gpu, vec![6, 4], 13);
    let xsc = lcg_tensor(&cpu, vec![6, 4], 13);
    let gs = gpu.zeros(vec![6, 2]).unwrap();
    let cs = cpu.zeros(vec![6, 2]).unwrap();
    gpu.slice_last(&xs, &gs, 2).unwrap();
    cpu.slice_last(&xsc, &cs, 2).unwrap();
    assert!(
        close(&gpu.to_cpu_f32(&gs).unwrap(), &cpu.to_cpu_f32(&cs).unwrap()),
        "slice_last"
    );
    let xs3 = lcg_tensor(&gpu, vec![2, 3, 4], 13);
    let xs3c = lcg_tensor(&cpu, vec![2, 3, 4], 13);
    let gs2 = gpu.zeros(vec![2, 2, 4]).unwrap();
    let cs2 = cpu.zeros(vec![2, 2, 4]).unwrap();
    gpu.slice_time(&xs3, &gs2, 1).unwrap();
    cpu.slice_time(&xs3c, &cs2, 1).unwrap();
    assert!(
        close(
            &gpu.to_cpu_f32(&gs2).unwrap(),
            &cpu.to_cpu_f32(&cs2).unwrap()
        ),
        "slice_time"
    );
}

#[test]
fn cuda_rope_glu_repeat_kv_vs_cpu() {
    let Some(gpu) = make_gpu() else { return };
    let cpu = CpuBackend;
    // rope interleaved with start offset
    let x = lcg_tensor(&gpu, vec![1, 4, 8], 14);
    let xc = lcg_tensor(&cpu, vec![1, 4, 8], 14);
    let cos = lcg_tensor(&gpu, vec![8, 4], 15);
    let cosc = lcg_tensor(&cpu, vec![8, 4], 15);
    let sin = lcg_tensor(&gpu, vec![8, 4], 16);
    let sinc = lcg_tensor(&cpu, vec![8, 4], 16);
    let go = gpu.zeros(vec![1, 4, 8]).unwrap();
    let co = cpu.zeros(vec![1, 4, 8]).unwrap();
    gpu.rope_interleaved(&x, &cos, &sin, &go, 8, 3).unwrap();
    cpu.rope_interleaved(&xc, &cosc, &sinc, &co, 8, 3).unwrap();
    assert!(
        close(&gpu.to_cpu_f32(&go).unwrap(), &cpu.to_cpu_f32(&co).unwrap()),
        "rope_interleaved"
    );

    // glu
    let gx = lcg_tensor(&gpu, vec![2, 8], 17);
    let gxc = lcg_tensor(&cpu, vec![2, 8], 17);
    let gg = gpu.zeros(vec![2, 4]).unwrap();
    let cg = cpu.zeros(vec![2, 4]).unwrap();
    gpu.glu(&gx, &gg).unwrap();
    cpu.glu(&gxc, &cg).unwrap();
    assert!(
        close(&gpu.to_cpu_f32(&gg).unwrap(), &cpu.to_cpu_f32(&cg).unwrap()),
        "glu"
    );

    // repeat ×2
    let rx = lcg_tensor(&gpu, vec![1, 3, 4], 18);
    let rxc = lcg_tensor(&cpu, vec![1, 3, 4], 18);
    let gr = gpu.zeros(vec![1, 6, 4]).unwrap();
    let cr = cpu.zeros(vec![1, 6, 4]).unwrap();
    gpu.repeat(&rx, &gr).unwrap();
    cpu.repeat(&rxc, &cr).unwrap();
    assert!(
        close(&gpu.to_cpu_f32(&gr).unwrap(), &cpu.to_cpu_f32(&cr).unwrap()),
        "repeat"
    );

    // kv_update
    let dst = gpu.zeros(vec![1, 8, 4]).unwrap();
    let dstc = cpu.zeros(vec![1, 8, 4]).unwrap();
    let src = lcg_tensor(&gpu, vec![1, 3, 4], 19);
    let srcc = lcg_tensor(&cpu, vec![1, 3, 4], 19);
    gpu.kv_update(&dst, &src, 2).unwrap();
    cpu.kv_update(&dstc, &srcc, 2).unwrap();
    assert!(
        close(
            &gpu.to_cpu_f32(&dst).unwrap(),
            &cpu.to_cpu_f32(&dstc).unwrap()
        ),
        "kv_update"
    );

    // embedding
    let table = lcg_tensor(&gpu, vec![16, 4], 20);
    let tablec = lcg_tensor(&cpu, vec![16, 4], 20);
    let idx: Vec<u32> = (0..5).map(|i| (i * 3) % 16).collect();
    let ge = gpu.zeros(vec![1, 5, 4]).unwrap();
    let ce = cpu.zeros(vec![1, 5, 4]).unwrap();
    gpu.embedding(&idx, &table, &ge).unwrap();
    cpu.embedding(&idx, &tablec, &ce).unwrap();
    assert!(
        close(&gpu.to_cpu_f32(&ge).unwrap(), &cpu.to_cpu_f32(&ce).unwrap()),
        "embedding"
    );

    // uniform_attn
    let v = lcg_tensor(&gpu, vec![1, 4, 6], 21);
    let vc = lcg_tensor(&cpu, vec![1, 4, 6], 21);
    let gu = gpu.zeros(vec![1, 6]).unwrap();
    let cu = cpu.zeros(vec![1, 6]).unwrap();
    gpu.uniform_attn(&v, &gu).unwrap();
    cpu.uniform_attn(&vc, &cu).unwrap();
    assert!(
        close(&gpu.to_cpu_f32(&gu).unwrap(), &cpu.to_cpu_f32(&cu).unwrap()),
        "uniform_attn"
    );
}

#[test]
fn cuda_attn_ar_vs_cpu() {
    let Some(gpu) = make_gpu() else { return };
    let cpu = CpuBackend;
    // GQA: 4 heads, 2 kv heads, hd=2; tq=3, tkv=4 (cached), causal mask
    let q = lcg_tensor(&gpu, vec![1, 3, 8], 22);
    let qc = lcg_tensor(&cpu, vec![1, 3, 8], 22);
    let k = lcg_tensor(&gpu, vec![1, 4, 4], 23);
    let kc = lcg_tensor(&cpu, vec![1, 4, 4], 23);
    let v = lcg_tensor(&gpu, vec![1, 4, 4], 24);
    let vc = lcg_tensor(&cpu, vec![1, 4, 4], 24);
    // causal mask (tq=3, tkv=4): j <= i allowed
    let mut mask_data = vec![f32::NEG_INFINITY; 12];
    for i in 0..3usize {
        for j in 0..4usize {
            if j <= i {
                mask_data[i * 4 + j] = 0.0;
            }
        }
    }
    let mask = gpu.from_cpu_f32(&mask_data, vec![3, 4]).unwrap();
    let maskc = cpu.from_cpu_f32(&mask_data, vec![3, 4]).unwrap();
    let go = gpu.zeros(vec![1, 3, 8]).unwrap();
    let co = cpu.zeros(vec![1, 3, 8]).unwrap();
    gpu.attn_ar(&q, &k, &v, Some(&mask), &go, 4, 2, 2).unwrap();
    cpu.attn_ar(&qc, &kc, &vc, Some(&maskc), &co, 4, 2, 2)
        .unwrap();
    assert!(
        close(&gpu.to_cpu_f32(&go).unwrap(), &cpu.to_cpu_f32(&co).unwrap()),
        "attn_ar"
    );

    // codebook_search
    let x = gpu
        .from_cpu_f32(&[0.1, 0.2, 0.9, 0.8], vec![1, 2, 2])
        .unwrap();
    let xc = cpu
        .from_cpu_f32(&[0.1, 0.2, 0.9, 0.8], vec![1, 2, 2])
        .unwrap();
    let table = gpu.from_cpu_f32(&[0.0, 0.0, 1.0, 1.0], vec![2, 2]).unwrap();
    let tablec = cpu.from_cpu_f32(&[0.0, 0.0, 1.0, 1.0], vec![2, 2]).unwrap();
    let mut gi = [0u32; 2];
    let mut ci = [0u32; 2];
    gpu.codebook_search(&x, &table, &mut gi).unwrap();
    cpu.codebook_search(&xc, &tablec, &mut ci).unwrap();
    assert_eq!(gi, ci, "codebook_search");
}

#[test]
fn cuda_misc_vs_cpu() {
    let Some(gpu) = make_gpu() else { return };
    let cpu = CpuBackend;
    // mhsa mode 0 (single head, causal-free; naive recompute on both sides)
    let q = lcg_tensor(&gpu, vec![1, 3, 4], 30);
    let qc = lcg_tensor(&cpu, vec![1, 3, 4], 30);
    let k = lcg_tensor(&gpu, vec![1, 3, 4], 31);
    let kc = lcg_tensor(&cpu, vec![1, 3, 4], 31);
    let v = lcg_tensor(&gpu, vec![1, 3, 4], 32);
    let vc = lcg_tensor(&cpu, vec![1, 3, 4], 32);
    let sc = gpu.zeros(vec![1, 1, 3, 3]).unwrap();
    let scc = cpu.zeros(vec![1, 1, 3, 3]).unwrap();
    let go = gpu.zeros(vec![1, 3, 4]).unwrap();
    let co = cpu.zeros(vec![1, 3, 4]).unwrap();
    gpu.mhsa(&q, &k, &v, &sc, None, 0, &go).unwrap();
    cpu.mhsa(&qc, &kc, &vc, &scc, None, 0, &co).unwrap();
    assert!(
        close(&gpu.to_cpu_f32(&go).unwrap(), &cpu.to_cpu_f32(&co).unwrap()),
        "mhsa"
    );

    // conv2d_nchw: x (1,2,5,5), w (3,2,3,3), pad 0 → (1,3,3,3)
    let x = lcg_tensor(&gpu, vec![1, 2, 5, 5], 33);
    let xc = lcg_tensor(&cpu, vec![1, 2, 5, 5], 33);
    let w = lcg_tensor(&gpu, vec![3, 2, 3, 3], 34);
    let wc = lcg_tensor(&cpu, vec![3, 2, 3, 3], 34);
    let g2 = gpu.zeros(vec![1, 3, 3, 3]).unwrap();
    let c2 = cpu.zeros(vec![1, 3, 3, 3]).unwrap();
    gpu.conv2d_nchw(&x, &w, None, 0, 0, 1, 1, false, &g2)
        .unwrap();
    cpu.conv2d_nchw(&xc, &wc, None, 0, 0, 1, 1, false, &c2)
        .unwrap();
    assert!(
        close(&gpu.to_cpu_f32(&g2).unwrap(), &cpu.to_cpu_f32(&c2).unwrap()),
        "conv2d_nchw"
    );

    // maxpool2d: (1,1,4,4) k2 → (1,1,2,2)
    let m = lcg_tensor(&gpu, vec![1, 1, 4, 4], 35);
    let mc = lcg_tensor(&cpu, vec![1, 1, 4, 4], 35);
    let gm = gpu.zeros(vec![1, 1, 2, 2]).unwrap();
    let cm = cpu.zeros(vec![1, 1, 2, 2]).unwrap();
    gpu.maxpool2d(&m, &gm, 2, 2).unwrap();
    cpu.maxpool2d(&mc, &cm, 2, 2).unwrap();
    assert!(
        close(&gpu.to_cpu_f32(&gm).unwrap(), &cpu.to_cpu_f32(&cm).unwrap()),
        "maxpool2d"
    );

    // add_bias2d: x (1,3,4) + b (1,4)
    let xb = lcg_tensor(&gpu, vec![1, 3, 4], 36);
    let xbc = lcg_tensor(&cpu, vec![1, 3, 4], 36);
    let bb = lcg_tensor(&gpu, vec![1, 3, 4], 37);
    let bbc = lcg_tensor(&cpu, vec![1, 3, 4], 37);
    let gb = gpu.zeros(vec![1, 3, 4]).unwrap();
    let cb = cpu.zeros(vec![1, 3, 4]).unwrap();
    gpu.add_bias2d(&xb, &bb, &gb).unwrap();
    cpu.add_bias2d(&xbc, &bbc, &cb).unwrap();
    assert!(
        close(&gpu.to_cpu_f32(&gb).unwrap(), &cpu.to_cpu_f32(&cb).unwrap()),
        "add_bias2d"
    );

    // mul_rows / mul_vec / snake / rope_head
    let xr = lcg_tensor(&gpu, vec![3, 4], 38);
    let xrc = lcg_tensor(&cpu, vec![3, 4], 38);
    let wr = lcg_tensor(&gpu, vec![3], 39);
    let wrc = lcg_tensor(&cpu, vec![3], 39);
    let gr = gpu.zeros(vec![3, 4]).unwrap();
    let cr = cpu.zeros(vec![3, 4]).unwrap();
    gpu.mul_rows(&xr, &wr, &gr).unwrap();
    cpu.mul_rows(&xrc, &wrc, &cr).unwrap();
    assert!(
        close(&gpu.to_cpu_f32(&gr).unwrap(), &cpu.to_cpu_f32(&cr).unwrap()),
        "mul_rows"
    );

    let wv = lcg_tensor(&gpu, vec![4], 40);
    let wvc = lcg_tensor(&cpu, vec![4], 40);
    gpu.mul_vec(&xr, &wv, &gr).unwrap();
    cpu.mul_vec(&xrc, &wvc, &cr).unwrap();
    assert!(
        close(&gpu.to_cpu_f32(&gr).unwrap(), &cpu.to_cpu_f32(&cr).unwrap()),
        "mul_vec"
    );

    let sa = lcg_tensor(&gpu, vec![4], 41);
    let sac = lcg_tensor(&cpu, vec![4], 41);
    gpu.snake(&xr, &sa, &gr).unwrap();
    cpu.snake(&xrc, &sac, &cr).unwrap();
    assert!(
        close(&gpu.to_cpu_f32(&gr).unwrap(), &cpu.to_cpu_f32(&cr).unwrap()),
        "snake"
    );

    let rh = lcg_tensor(&gpu, vec![1, 3, 8], 42);
    let rhc = lcg_tensor(&cpu, vec![1, 3, 8], 42);
    let rcos = lcg_tensor(&gpu, vec![3, 2], 43);
    let rcosc = lcg_tensor(&cpu, vec![3, 2], 43);
    let rsin = lcg_tensor(&gpu, vec![3, 2], 44);
    let rsinc = lcg_tensor(&cpu, vec![3, 2], 44);
    let grh = gpu.zeros(vec![1, 3, 8]).unwrap();
    let crh = cpu.zeros(vec![1, 3, 8]).unwrap();
    gpu.rope_head(&rh, &rcos, &rsin, &grh, 2, 4).unwrap();
    cpu.rope_head(&rhc, &rcosc, &rsinc, &crh, 2, 4).unwrap();
    assert!(
        close(
            &gpu.to_cpu_f32(&grh).unwrap(),
            &cpu.to_cpu_f32(&crh).unwrap()
        ),
        "rope_head"
    );
}

fn close(a: &[f32], b: &[f32]) -> bool {
    const RTOL: f32 = 2e-3; // naive kernels: fused ops get a wider band
    const ATOL: f32 = 1e-4;
    let ok = a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| (x - y).abs() <= ATOL + RTOL * y.abs());
    if !ok && std::env::var("CUDA_SMOKE_VERBOSE").is_ok() {
        let mut shown = 0;
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            if (x - y).abs() > ATOL + RTOL * y.abs() && shown < 8 {
                eprintln!("  mismatch[{i}]: gpu {x} vs cpu {y} (diff {})", x - y);
                shown += 1;
            }
        }
    }
    ok
}

/// Length guards on the elementwise kernels: `n` is taken from the FIRST
/// operand, so a short `out` (or short second operand) used to launch a kernel
/// that wrote/read past the end of a device allocation — reachable from safe
/// code, e.g. `scale(&x[4096], &out[1], 2.0)`. The call must now fail cleanly.
#[test]
fn elementwise_rejects_length_mismatch() {
    let Some(gpu) = make_gpu() else { return };
    let long = gpu.from_cpu_f32(&[1.0f32; 64], vec![64]).unwrap();
    let short = gpu.from_cpu_f32(&[0.0f32; 1], vec![1]).unwrap();
    let same = gpu.from_cpu_f32(&[2.0f32; 64], vec![64]).unwrap();

    // out too short (the old code wrote 64 elements into a 1-element buffer)
    assert!(
        gpu.scale(&long, &short, 2.0).is_err(),
        "scale into a short output must be rejected"
    );
    // second operand too short
    assert!(
        gpu.add(&long, &short, &same).is_err(),
        "add with a short second operand must be rejected"
    );
    // matching lengths still work
    assert!(gpu.scale(&long, &same, 2.0).is_ok());
}

/// GEMM with an output aliasing an input is not a valid GEMM: it used to build
/// a `&mut` to storage that another live reference pointed at (UB). Now: error.
#[test]
fn gemm_rejects_aliased_output() {
    let Some(gpu) = make_gpu() else { return };
    let a = gpu.from_cpu_f32(&[1.0f32; 16], vec![4, 4]).unwrap();
    let b = gpu.from_cpu_f32(&[1.0f32; 16], vec![4, 4]).unwrap();
    let c = gpu.from_cpu_f32(&[0.0f32; 16], vec![4, 4]).unwrap();
    assert!(gpu.gemm(&a, &b, &c).is_ok(), "distinct buffers must work");
    let aliased = a.clone();
    assert!(
        gpu.gemm(&a, &b, &aliased).is_err(),
        "gemm with c aliasing a must be rejected"
    );
    assert!(
        gpu.gemm(&c, &b, &c).is_err(),
        "gemm with c aliasing an input tensor must be rejected"
    );
}
