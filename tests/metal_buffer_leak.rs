//! Утечка MTLBuffer (стресс Ф8.3): буферы, созданные пулом, обязаны
//! освобождаться. Счётчик — `MTLDevice.currentAllocatedSize` (сколько байт
//! реально держит устройство), а не RSS: страницы утёкших буферов ОС
//! выгружает в своп, и RSS «падает», пока своп растёт (так демон дошёл до
//! jetsam-SIGKILL на долгой генерации).
#![cfg(all(target_os = "macos", feature = "metal"))]

use rust_inference::backend::Backend;
use rust_inference::metal::MetalBackend;

fn allocated_mb(be: &MetalBackend) -> f64 {
    be.ctx.device.current_allocated_size() as f64 / (1024.0 * 1024.0)
}

#[test]
fn batch_mode_buffers_are_released() {
    let be = MetalBackend::new().unwrap();
    // «шаг модели»: много промежуточных тензоров в batch-режиме
    let step = |be: &MetalBackend| {
        be.begin_batch().unwrap();
        for _ in 0..64 {
            let a = be.zeros(vec![256, 1024]).unwrap(); // 1 МБ
            let b = be.zeros(vec![256, 1024]).unwrap();
            be.add(&a, &b, &a).unwrap();
        }
        be.end_batch().unwrap();
    };
    for _ in 0..3 {
        step(&be);
    }
    let base = allocated_mb(&be);
    for _ in 0..40 {
        step(&be);
    }
    let after = allocated_mb(&be);
    eprintln!("allocated: {base:.0} МБ → {after:.0} МБ после 40 шагов по ~128 МБ");
    // 40 шагов × 128 МБ = 5 ГБ временных буферов: при освобождении рост
    // ограничен пулом (≤ его предела), при утечке — гигабайты.
    assert!(
        after - base < 1200.0,
        "буферы не освобождаются: {base:.0} → {after:.0} МБ"
    );
}
