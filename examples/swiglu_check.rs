//! Изоляция Ф8.1h: наш silu+mul против оркального kernel_swiglu на одних
//! входах (дампы gate/up/swiglu слоя 0 OLMoE).
use rust_inference::backend::Backend as _;
use rust_inference::metal::MetalBackend;

fn rd(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap();
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() {
    let be = MetalBackend::new().unwrap();
    let g = rd("/tmp/llama_ref/sw_gate.bin");
    let u = rd("/tmp/llama_ref/sw_up.bin");
    let want = rd("/tmp/llama_ref/sw_swiglu.bin");
    let n = g.len();
    let gt = be.from_cpu_f32(&g, vec![n]).unwrap();
    let ut = be.from_cpu_f32(&u, vec![n]).unwrap();
    let act = be.zeros(vec![n]).unwrap();
    be.silu(&gt, &act).unwrap();
    let prod = be.zeros(vec![n]).unwrap();
    be.mul(&act, &ut, &prod).unwrap();
    let got = be.to_cpu_f32(&prod).unwrap();
    let fused = be.zeros(vec![n]).unwrap();
    be.swiglu(&gt, &ut, &fused).unwrap();
    let gotf = be.to_cpu_f32(&fused).unwrap();
    let badf = (0..n)
        .filter(|i| gotf[*i].to_bits() != want[*i].to_bits())
        .count();
    println!("swiglu_fused vs oracle swiglu: bad={badf}/{n}");
    let bad = (0..n)
        .filter(|i| got[*i].to_bits() != want[*i].to_bits())
        .count();
    println!("silu+mul vs oracle swiglu: bad={bad}/{n}");
    // и сам silu: сравним с «эталонным» silu, восстановленным из swiglu/up?
    // нельзя (up≠1). Проверим детерминизм: второй прогон.
    let act2 = be.zeros(vec![n]).unwrap();
    be.silu(&gt, &act2).unwrap();
    let prod2 = be.zeros(vec![n]).unwrap();
    be.mul(&act2, &ut, &prod2).unwrap();
    let got2 = be.to_cpu_f32(&prod2).unwrap();
    println!(
        "детерминизм: {}",
        (0..n)
            .filter(|i| got[*i].to_bits() != got2[*i].to_bits())
            .count()
    );
}
