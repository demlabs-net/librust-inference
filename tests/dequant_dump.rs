//! Сверка деквиант Q8_0 на реальном файле (против python-разбора вручную).
//! Запуск: QWEN_GGUF=path cargo test --features cpu --test dequant_dump -- --ignored --nocapture

use rust_inference::gguf;
use rust_inference::weights_map::WeightsMap;

#[test]
#[ignore]
fn dump_q8_first_block() {
    let path = std::env::var("QWEN_GGUF").expect("QWEN_GGUF");
    let file = std::fs::File::open(&path).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let parsed = gguf::parse(&mmap).unwrap();
    let wm = WeightsMap::from_gguf(&parsed).unwrap();
    let t = wm.get("token_embd.weight").expect("token_embd");
    println!("shape={:?} numel={}", t.shape, t.numel());
    let raw = t.raw();
    println!("raw_len={} first4bytes={:?}", raw.len(), &raw[..4]);
    // первый блок Q8_0: [scale f16][32×i8]
    let d = i16::from_le_bytes([raw[0], raw[1]]);
    let d = (d as f32) * 2f32.powi(-16); // f16→f32
    let vals: Vec<f32> = raw[2..34].iter().map(|&b| (b as i8) as f32 * d).collect();
    println!("d={d:.8} vals={vals:?}");
}
