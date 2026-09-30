//! f16-KV (Ф8.2f): Metal реально хранит KV в half — память под контекст
//! вдвое меньше f32. Проверяем на уровне хранилища (`KvCache` + формат
//! тензора), а не по RSS: RSS прячет Metal-буферы (они не в куче процесса).
#![cfg(all(target_os = "macos", feature = "metal"))]

use rust_inference::backend::Backend;
use rust_inference::dtype::KvFormat;
use rust_inference::metal::MetalBackend;
use rust_inference::nn::KvCache;

#[test]
fn f16_kv_storage_is_half_of_f32() {
    let be = MetalBackend::new().unwrap();
    // Геометрия Qwen3-8B: 8 KV-голов × 128 = 1024 значения на строку.
    let rows = 4096usize;
    let mk = |fmt| -> KvCache {
        let mut c = KvCache::new(&be, 8, 128, fmt, fmt, 1_000_000.0, None).unwrap();
        c.ensure(&be, rows).unwrap();
        c
    };
    let c32 = mk(KvFormat::F32);
    let c16 = mk(KvFormat::F16);

    // Формат запрошенный = формат хранимый (Metal не подменяет f16 на f32).
    assert_eq!(c32.stored_formats(), (KvFormat::F32, KvFormat::F32));
    assert_eq!(c16.stored_formats(), (KvFormat::F16, KvFormat::F16));

    let (b32, b16) = (c32.storage_bytes(), c16.storage_bytes());
    assert!(b32 >= rows * 1024 * 4 * 2, "ёмкость f32-кэша: {b32} байт");
    assert_eq!(b16 * 2, b32, "f16-кэш обязан быть ровно вдвое меньше");

    // Для протокола: сколько это на реальных окнах (k+v на всех слоях, 36
    // слоёв × 1024 значения на строку; ёмкость растёт удвоением).
    let layers = 36usize;
    let per_token = (1024 * layers * 2) as f64; // K+V, все слои
    for ctx in [4096usize, 32768] {
        let cap = ctx.next_power_of_two();
        eprintln!(
            "ctx {ctx:>6} (ёмкость {cap:>6}): KV f32 {:>6.2} ГБ, f16 {:>6.2} ГБ",
            per_token * cap as f64 * 4.0 / 1e9,
            per_token * cap as f64 * 2.0 / 1e9
        );
    }
}
