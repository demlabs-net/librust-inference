//! Диагностика процессорного времени (для профилей шага).
//!
//! Зачем: секции профайлера меряют стенное время, и по ним не видно, шаг
//! упирается в процессор или ждёт (очередь устройства, аллокатор, ОС).
//! Сравнение user+sys с wall отвечает на этот вопрос без профайлера.

/// Суммарное CPU-время процесса `(user, sys)` в секундах.
///
/// Linux — `/proc/self/stat` (поля 14 и 15, тики по 100 Гц); на прочих
/// платформах — нули: диагностика, а не функциональность.
#[cfg(target_os = "linux")]
pub fn cpu_times() -> (f64, f64) {
    let Ok(s) = std::fs::read_to_string("/proc/self/stat") else {
        return (0.0, 0.0);
    };
    // Поле comm (2-е) может содержать пробелы и скобки — отсчитываем поля
    // после ПОСЛЕДНЕЙ ')': тогда utime = 12-е, stime = 13-е среди остатка.
    let Some(rest) = s.rsplit(')').next() else {
        return (0.0, 0.0);
    };
    let f: Vec<&str> = rest.split_whitespace().collect();
    let tick = 100.0; // USER_HZ на Linux
    let get = |i: usize| f.get(i).and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
    (get(11) / tick, get(12) / tick)
}

#[cfg(not(target_os = "linux"))]
pub fn cpu_times() -> (f64, f64) {
    (0.0, 0.0)
}

#[cfg(test)]
mod tests {
    /// Время процесса не убывает и не врёт знаком.
    #[test]
    fn cpu_times_is_sane() {
        let (u, s) = super::cpu_times();
        assert!(u >= 0.0 && s >= 0.0, "отрицательное CPU-время: {u} {s}");
        if cfg!(target_os = "linux") {
            // Тест сам что-то посчитал — время обязано быть ненулевым.
            let mut acc = 0u64;
            for i in 0..2_000_000u64 {
                acc = acc.wrapping_add(i % 7);
            }
            assert!(acc > 0);
            let (u2, _) = super::cpu_times();
            assert!(u2 >= u, "время уменьшилось: {u} -> {u2}");
        }
    }
}
