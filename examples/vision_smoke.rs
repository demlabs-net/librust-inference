//! Smoke: Qwen3-VL vision (real mmproj) — shape и не-NaN.
//! cargo run --release --features metal,cpu --example vision_smoke

use rust_inference::backend::Backend;
use rust_inference::metal::MetalBackend;
use rust_inference::vision_qwen3vl::VisionQwen3Vl;
use rust_inference::weights_map::WeightsMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "/Users/dmitriygerasimov/.vassista-models/llm/mmproj-Qwen3VL-2B-F16.gguf".into()
    });
    let buf = std::fs::read(&path)?;
    let gf = rust_inference::gguf::parse(&buf)?;
    let wm = WeightsMap::from_gguf(&gf)?;
    let backend = MetalBackend::new()?;
    let (n_layers, _, _) = (
        rust_inference::gguf::parse(&buf)
            .ok()
            .and_then(|gf| {
                rust_inference::gguf::GgufValue::as_u32(gf.get("clip.vision.block_count")?)
            })
            .unwrap_or(24) as usize,
        0usize,
        0usize,
    );
    let model = VisionQwen3Vl::load(&backend, &wm, n_layers)?;
    println!(
        "cfg: n_embd={} ffn={} layers={} proj={} tokens={}",
        model.n_embd,
        model.ffn_dim,
        model.n_layers,
        model.projection_dim,
        model.n_tokens()
    );
    // тестовая картинка: диагональный градиент 768×768 RGB
    let (h, w) = (768usize, 768usize);
    let img: Vec<f32> = (0..h * w * 3)
        .map(|i| {
            let px = (i / 3) % w;
            let py = (i / 3) / w;
            let x = px as f32 / w as f32;
            let y = py as f32 / h as f32;
            [x * 255.0, y * 255.0, ((x + y) / 2.0) * 255.0][i % 3]
        })
        .collect();
    let t0 = std::time::Instant::now();
    let (out, ds) = model.forward(&backend, &img, h, w)?;
    let dt = t0.elapsed();
    let v = backend.to_cpu_f32(&out)?;
    let n_nan = v.iter().filter(|x| x.is_nan()).count();
    let n_inf = v.iter().filter(|x| x.is_infinite()).count();
    let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
    println!(
        "out: shape {:?} values={} nan={} inf={} mean={mean:.4} in {:.1}s (deepstack: {} блока {:?})",
        out.shape(),
        v.len(),
        n_nan,
        n_inf,
        dt.as_secs_f32(),
        ds.len(),
        ds.first().map(|d| d.shape()),
    );
    assert_eq!(
        out.shape(),
        &[1, 576, 2048],
        "expected 576 vision tokens (24×24 после merge 2×2) × 2048"
    );
    assert_eq!(n_nan, 0, "NaN in vision output");
    assert_eq!(n_inf, 0, "Inf in vision output");
    let row0 = &v[..16];
    let norm0 = (v[..2048].iter().map(|x| (x * x) as f64).sum::<f64>()).sqrt();
    println!("rust row0[0..16]: {row0:?}");
    println!("rust norm row0: {norm0:.4}");
    if std::env::var("VISION_DUMP").is_ok() {
        // полный выход для сверки с python-эталоном (vision_py.npy)
        let mut out = String::new();
        for val in v.iter().take(576 * 2048) {
            out.push_str(&format!("{val:.9}\n"));
        }
        std::fs::write("/tmp/vision_rust.txt", out)?;
        println!("dumped /tmp/vision_rust.txt");
    }
    println!("VISION SMOKE OK");
    let _ = rust_inference::vision_qwen3vl::merge_2x2(&[], 0, 0); // keep import
    let _ = rust_inference::vision_qwen3vl::group4(&[], 0, 0);
    Ok(())
}
