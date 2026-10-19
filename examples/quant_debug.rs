//! Отладка квантования: первые байты блока Q4_0/Q8_0 выбранного тензора.
//! `quant_debug <src.gguf> <tensor_name> [n]`
use rust_inference::gguf::parse;
use rust_inference::quant::{quant_q4_0, quant_q8_0};
use rust_inference::weights_map::dequant_gguf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let want = &args[2];
    let n: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8);
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let gf = parse(&mmap)?;
    let t = gf
        .tensors
        .iter()
        .find(|t| &t.name == want)
        .ok_or("тензор не найден")?;
    println!(
        "{}: type {:?} dims {:?} offset {} len {}",
        t.name,
        t.ggml_type,
        t.dims,
        t.offset,
        t.data.len()
    );
    let w = dequant_gguf(
        t.ggml_type,
        &t.data[..t.ggml_type.bytes_per_element(32) as usize],
    )?;
    println!("первые {n} f32 из источника: {:?}", &w[..n]);
    let q4 = quant_q4_0(&w);
    println!("наш Q4_0 блок: {}", hex(&q4));
    let q8 = quant_q8_0(&w);
    println!("наш Q8_0 блок: {}", hex(&q8));
    Ok(())
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
