//! Диагностика контейнера NINFER: манифест, тома, байты одного тензора.
//!
//! Запуск (путь к контейнеру передаётся аргументом, ничего не хардкодим):
//! ```sh
//! cargo run --release --example ninfer_probe -- /path/to/model.infernix [имя-bindings]
//! ```

use std::collections::BTreeMap;

use rust_inference::ninfer::Ninfer;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = match args.next() {
        Some(p) => p,
        None => {
            eprintln!("usage: ninfer_probe <container.infernix> [bindings-name]");
            std::process::exit(2);
        }
    };
    let n = Ninfer::open(std::path::Path::new(&path)).expect("открыть контейнер");
    println!("артефакт: {:?}", n.name);
    println!("объектов: {}", n.objects().len());

    let mut layouts: BTreeMap<&str, usize> = BTreeMap::new();
    let mut formats: BTreeMap<&str, usize> = BTreeMap::new();
    let mut payload = 0u64;
    for o in n.objects().iter().filter(|o| o.is_tensor()) {
        *layouts.entry(o.layout.as_str()).or_default() += 1;
        *formats.entry(o.format.as_str()).or_default() += 1;
        payload += o.bytes;
    }
    println!("payload: {:.1} GiB", payload as f64 / (1u64 << 30) as f64);
    println!("раскладки: {layouts:?}");
    println!("форматы: {formats:?}");

    let names = n.names_with_prefix("text/layers/0/");
    println!("bindings слоя 0 ({}):", names.len());
    for name in names.iter().take(20) {
        let o = n.object_by_name(name).unwrap();
        println!(
            "  {name} → {} {:?} {} {} bytes={}",
            o.id, o.shape, o.format, o.layout, o.bytes
        );
    }

    let want = args
        .next()
        .unwrap_or_else(|| "text/layers/0/attn_hc/norm".to_string());
    if let Some(o) = n.object_by_name(&want) {
        let bytes = n.read(o).expect("прочитать объект");
        println!(
            "\n{want} → {} {:?} {} {}",
            o.id, o.shape, o.format, o.layout
        );
        println!("прочитано {} байт (ожидалось {})", bytes.len(), o.bytes);
        // bf16 → f32 (старшие 16 бит f32)
        if o.format == "bf16" && bytes.len() >= 4 {
            let vals: Vec<f32> = bytes
                .chunks_exact(2)
                .map(|c| {
                    let hi = u16::from_le_bytes([c[0], c[1]]) as u32;
                    f32::from_bits(hi << 16)
                })
                .collect();
            let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mean = vals.iter().sum::<f32>() / vals.len() as f32;
            println!(
                "bf16: {} значений, min {min:.4} max {max:.4} mean {mean:.4}, первые {:?}",
                vals.len(),
                &vals[..vals.len().min(8)]
            );
        }
    } else {
        // имя может ссылаться на диапазоны внутри упакованного объекта
        match n.parts(&want) {
            Ok(parts) => {
                println!("\n{want}: упаковано в {} частей", parts.len());
                for (o, off, len) in &parts {
                    println!(
                        "  {} +{}..{} ({} байт) {} {}",
                        o.id,
                        off,
                        off + len,
                        len,
                        o.format,
                        o.layout
                    );
                }
                let bytes = n.read_name(&want).expect("прочитать по имени");
                println!("собрано {} байт", bytes.len());
            }
            Err(e) => println!("\nнет bindings-имени {want}: {e}"),
        }
    }
}
