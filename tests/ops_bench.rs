//! Micro-benchmarks of the non-GEMM ops on TTS decode shapes.
//! `cargo test --release --features "cuda cpu" --test ops_bench -- --nocapture`
#![cfg(all(feature = "cuda", feature = "cpu"))]
use rust_inference::Backend;
use rust_inference::cuda::CudaBackend;
use std::time::Instant;

fn bench<F: Fn(&CudaBackend)>(gpu: &CudaBackend, name: &str, reps: usize, f: F) {
    f(gpu); // warmup
    gpu.synchronize().unwrap();
    let t0 = Instant::now();
    for _ in 0..reps {
        f(gpu);
    }
    gpu.synchronize().unwrap();
    let dt = t0.elapsed().as_secs_f64() / reps as f64;
    println!("{name:>34}: {:>8.2} us", dt * 1e6);
}

#[test]
#[ignore = "K28: GPU benchmark — run with --ignored on a GPU host"]
fn ops_bench_suite() {
    let gpu = CudaBackend::new(0).unwrap();
    let w = gpu.from_cpu_f32(&vec![0.1f32; 2048], vec![2048]).unwrap();
    let w8 = gpu.from_cpu_f32(&vec![0.1f32; 8192], vec![8192]).unwrap();
    let x1 = gpu.zeros(vec![1, 2048]).unwrap();
    let x8 = gpu.zeros(vec![1, 8192]).unwrap();
    let x3 = gpu.zeros(vec![1, 6144]).unwrap();
    let y = gpu.zeros(vec![1, 2048]).unwrap();
    let y8 = gpu.zeros(vec![1, 8192]).unwrap();
    let y3 = gpu.zeros(vec![1, 6144]).unwrap();

    bench(&gpu, "rmsnorm [1,2048]", 2000, |g| {
        g.rmsnorm(&x1, &w, &y, 1e-6).unwrap();
    });
    bench(&gpu, "silu [1,2048]", 2000, |g| {
        g.silu(&x1, &y).unwrap();
    });
    bench(&gpu, "silu [1,8192]", 2000, |g| {
        g.silu(&x8, &y8).unwrap();
    });
    bench(&gpu, "mul [1,8192]", 2000, |g| {
        g.mul(&x8, &y8, &y8).unwrap();
    });
    bench(&gpu, "add_bias [1,6144]", 2000, |g| {
        g.add_bias(&x3, &y3, &y3).unwrap();
    });
    bench(&gpu, "zeros [1,2048]", 2000, |g| {
        g.zeros(vec![1, 2048]).unwrap();
    });
    bench(&gpu, "zeros [1,8192]", 2000, |g| {
        g.zeros(vec![1, 8192]).unwrap();
    });
    bench(&gpu, "slice_last [1,6144]->[1,2048]", 2000, |g| {
        g.slice_last(&x3, &y, 0).unwrap();
    });
    bench(&gpu, "rope [1,1,2048]", 1000, |g| {
        let cos = g.zeros(vec![1, 16]).unwrap();
        let sin = g.zeros(vec![1, 16]).unwrap();
        g.rope_interleaved(
            &x1.view(vec![1, 1, 2048]),
            &cos,
            &sin,
            &y.view(vec![1, 1, 2048]),
            8,
            0,
        )
        .unwrap();
    });
    // attn_ar with growing kv
    for tkv in [16usize, 64, 273] {
        let q = gpu.zeros(vec![1, 1, 2048]).unwrap();
        let k = gpu.zeros(vec![1, tkv, 1024]).unwrap();
        let v = gpu.zeros(vec![1, tkv, 1024]).unwrap();
        let o = gpu.zeros(vec![1, 1, 2048]).unwrap();
        let reps = 200;
        let t0 = Instant::now();
        for _ in 0..reps {
            gpu.attn_ar(&q, &k, &v, None, &o, 8, 4, 256).unwrap();
        }
        gpu.synchronize().unwrap();
        let dt = t0.elapsed().as_secs_f64() / reps as f64;
        println!("attn_ar tkv={tkv:>3} (hd=256, h=8): {:>8.2} us", dt * 1e6);
    }
}
