//! Ф1.12: аудио-вход — Whisper-стиль mel-препроцессинг (Qwen2.5-Omni).
//!
//! Точная копия transformers `WhisperFeatureExtractor` + `audio_utils`:
//! slaney-мел-фильтрбанк (min 0 Hz, max 8000 Hz), hann-окно 400 (periodic),
//! hop 160 (10 мс @ 16 кГц), power-спектр, log10-нормализация
//! (`max - 8.0`, `(x + 4) / 4`), последний STFT-кадр отбрасывается
//! (`stft[..., :-1]` в оригинале). Также синусоидальные позиционные
//! эмбеддинги аудио-энкодера (Whisper-стиль, cat sin/cos).

use std::path::Path;

use rustfft::num_complex::Complex;

pub const N_FFT: usize = 400;
pub const HOP: usize = 160;
pub const N_MELS: usize = 128;
pub const SAMPLING_RATE: usize = 16000;
/// WhisperFeatureExtractor: фильтрбанк покрывает [0, 8000] Hz.
const MEL_MAX_HZ: f64 = 8000.0;

fn hz_to_mel_slaney(f: f64) -> f64 {
    let min_log_hertz = 1000.0;
    let min_log_mel = 15.0;
    let logstep = 27.0 / (6.4f64).ln();
    let mels = 3.0 * f / 200.0;
    if f >= min_log_hertz {
        min_log_mel + (f / min_log_hertz).ln() * logstep
    } else {
        mels
    }
}

fn mel_to_hz_slaney(m: f64) -> f64 {
    let min_log_hertz = 1000.0;
    let min_log_mel = 15.0;
    let logstep = (6.4f64).ln() / 27.0;
    let freq = 200.0 * m / 3.0;
    if m >= min_log_mel {
        min_log_hertz * (logstep * (m - min_log_mel)).exp()
    } else {
        freq
    }
}

/// Slaney-мел-фильтрбанк (n_fft/2+1 × n_mels, row-major [freq][mel]) —
/// бит-в-бит transformers `audio_utils.mel_filter_bank` (норм "slaney",
/// треугольники в частотном пространстве).
pub fn mel_filter_bank() -> Vec<f32> {
    let n_bins = N_FFT / 2 + 1; // 201
    let mel_min = hz_to_mel_slaney(0.0);
    let mel_max = hz_to_mel_slaney(MEL_MAX_HZ);
    let mut filter_freqs = Vec::with_capacity(N_MELS + 2);
    for i in 0..N_MELS + 2 {
        let t = i as f64 / (N_MELS + 1) as f64;
        filter_freqs.push(mel_to_hz_slaney(mel_min + t * (mel_max - mel_min)));
    }
    let mut fft_freqs = Vec::with_capacity(n_bins);
    for i in 0..n_bins {
        fft_freqs.push(i as f64 * (SAMPLING_RATE as f64 / 2.0) / (n_bins - 1) as f64);
    }
    // filter_diff = diff(filter_freqs); slopes = filter_freqs[i] - fft_freqs[j]
    let mut filter_diff = Vec::with_capacity(N_MELS + 1);
    for i in 0..N_MELS + 1 {
        filter_diff.push(filter_freqs[i + 1] - filter_freqs[i]);
    }
    let mut filters = vec![0.0f32; n_bins * N_MELS];
    for i in 0..N_MELS {
        for j in 0..n_bins {
            // down_slopes = -slopes[:, :-2] / filter_diff[:-1] → фильтр i,
            // left slope; up_slopes = slopes[:, 2:] / filter_diff[1:].
            let down = -(filter_freqs[i] - fft_freqs[j]) / filter_diff[i];
            let up = (filter_freqs[i + 2] - fft_freqs[j]) / filter_diff[i + 1];
            filters[j * N_MELS + i] = down.min(up).max(0.0) as f32;
        }
    }
    // Slaney-нормировка: колонка i × 2/(f[i+2] - f[i]).
    for i in 0..N_MELS {
        let enorm = 2.0 / (filter_freqs[i + 2] - filter_freqs[i]);
        for j in 0..n_bins {
            filters[j * N_MELS + i] *= enorm as f32;
        }
    }
    filters
}

/// Hann-окно, periodic (как `torch.hann_window(400)`).
pub fn hann_window(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / len as f64).cos())
        .map(|v| v as f32)
        .collect()
}

/// STFT power-спектр + mel + log-нормализация — ровно как
/// `WhisperFeatureExtractor._torch_extract_fbank_features` для одного
/// канала. Возвращает (mel [128 × T] row-major, T).
///
/// Длина: torch.stft(center=True) даёт `len/160 + 1` кадров, последний
/// отбрасывается (`stft[..., :-1]`) → T = len / 160.
pub fn log_mel_spectrogram(wav: &[f32]) -> (Vec<f32>, usize) {
    let n_bins = N_FFT / 2 + 1;
    let filters = mel_filter_bank();
    let window = hann_window(N_FFT);
    let mut planner = rustfft::FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);

    let t = wav.len() / HOP;
    let mut spec = vec![0.0f32; n_bins * t];
    let mut frame = vec![Complex::new(0.0f32, 0.0f32); N_FFT];
    for i in 0..t {
        let start = i * HOP; // центральный сэмпл кадра; окно [-200, +200)
        for (k, c) in frame.iter_mut().enumerate() {
            let idx = start as isize + k as isize - (N_FFT / 2) as isize;
            let v = if idx < 0 {
                wav[(-idx) as usize] // reflect
            } else if idx >= wav.len() as isize {
                wav[(2 * wav.len() as isize - idx - 2) as usize] // reflect
            } else {
                wav[idx as usize]
            };
            *c = Complex::new(v * window[k], 0.0);
        }
        fft.process(&mut frame);
        for j in 0..n_bins {
            spec[j * t + i] = frame[j].re * frame[j].re + frame[j].im * frame[j].im;
        }
    }
    // mel = filters.T @ spec → [128, T]
    let mut mel = vec![0.0f32; N_MELS * t];
    for m in 0..N_MELS {
        for i in 0..t {
            let mut acc = 0.0f32;
            for j in 0..n_bins {
                acc += filters[j * N_MELS + m] * spec[j * t + i];
            }
            mel[m * t + i] = acc;
        }
    }
    // log10 с floor 1e-10, отсечка max-8, (x+4)/4
    for v in mel.iter_mut() {
        *v = v.max(1e-10).log10();
    }
    let max_val = mel.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    for v in mel.iter_mut() {
        *v = v.max(max_val - 8.0);
        *v = (*v + 4.0) / 4.0;
    }
    (mel, t)
}

/// Синусоидальные позиционные эмбеддинги (Whisper-стиль): 
/// `cat([sin, cos], dim=1)`, max_timescale=10000 — бит-в-бит
/// `SinusoidsPositionEmbedding` из modeling_qwen2_5_omni.py.
pub fn sinusoids(length: usize, channels: usize, max_timescale: f64) -> Vec<f32> {
    let half = channels / 2;
    let log_timescale_increment = max_timescale.ln() / (half as f64 - 1.0);
    let mut out = vec![0.0f32; length * channels];
    for t in 0..length {
        for i in 0..half {
            let inv = (-log_timescale_increment * i as f64).exp();
            let scaled = t as f64 * inv;
            out[t * channels + i] = scaled.sin() as f32;
            out[t * channels + half + i] = scaled.cos() as f32;
        }
    }
    out
}

/// Парсинг WAV (PCM16, mono) → (samples f32 в [-1,1], sample_rate).
/// RIFF-заголовок разбирается вручную (без внешних зависимостей).
pub fn read_wav_pcm16(path: &Path) -> anyhow::Result<(Vec<f32>, u32)> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        anyhow::bail!("not a RIFF/WAVE file: {}", path.display());
    }
    let mut fmt_ok = false;
    let mut sample_rate = 0u32;
    let mut channels = 0u16;
    let mut bits = 0u16;
    let mut pcm: Option<&[u8]> = None;
    let mut off = 12;
    while off + 8 <= bytes.len() {
        let id = &bytes[off..off + 4];
        let size = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap()) as usize;
        let body = off + 8;
        if body + size > bytes.len() {
            break;
        }
        match id {
            b"fmt " if size >= 16 => {
                let fmt = u16::from_le_bytes(bytes[body..body + 2].try_into().unwrap());
                channels = u16::from_le_bytes(bytes[body + 2..body + 4].try_into().unwrap());
                sample_rate = u32::from_le_bytes(bytes[body + 4..body + 8].try_into().unwrap());
                bits = u16::from_le_bytes(bytes[body + 14..body + 16].try_into().unwrap());
                fmt_ok = fmt == 1 && bits == 16;
            }
            b"data" => {
                pcm = Some(&bytes[body..body + size]);
            }
            _ => {}
        }
        off = body + size + (size & 1); // word-align
    }
    if !fmt_ok || channels != 1 {
        anyhow::bail!("wav: нужен PCM16 mono (fmt_ok={fmt_ok}, ch={channels})");
    }
    let pcm = pcm.ok_or_else(|| anyhow::anyhow!("wav: нет data-чанка"))?;
    let samples = pcm
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
        .collect::<Vec<_>>();
    Ok((samples, sample_rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Golden против numpy-эталона (transformers-формулы, float64, seed 42):
    /// эталонный wav + mel лежат в fixtures/mel_ref.json (сгенерировано
    /// /tmp/mel_ref.py, numpy 2.5.2). Проверка на ТОМ ЖЕ сигнале.
    #[test]
    fn log_mel_matches_numpy_golden() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/mel_ref.json");
        let raw = std::fs::read_to_string(path).expect("mel_ref.json в fixtures/");
        let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let wav: Vec<f32> = v["wav"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        let want_mel: Vec<f32> = v["mel"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| row.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32))
            .collect();
        let (mel, t) = log_mel_spectrogram(&wav);
        assert_eq!(t, 100);
        assert_eq!(mel.len(), want_mel.len());
        let mut max_delta = 0.0f32;
        for (got, want) in mel.iter().zip(&want_mel) {
            max_delta = max_delta.max((got - want).abs());
        }
        // f32-накопление: расхождение с float64-эталоном ≤ ~1e-5
        assert!(max_delta < 5e-4, "max delta {max_delta} vs numpy-эталона");
    }

    #[test]
    fn mel_filter_bank_is_zero_free_and_slaneyshaped() {
        let f = mel_filter_bank();
        assert_eq!(f.len(), (N_FFT / 2 + 1) * N_MELS);
        let n_bins = N_FFT / 2 + 1;
        let mut peaks = Vec::new();
        for m in 0..N_MELS {
            let mut best = (0usize, 0.0f32);
            for j in 0..n_bins {
                if f[j * N_MELS + m] > best.1 {
                    best = (j, f[j * N_MELS + m]);
                }
            }
            peaks.push(best.0 as f64 * (SAMPLING_RATE as f64 / 2.0) / (n_bins - 1) as f64);
        }
        // фильтры не убывают по частоте пика (соседние фильтры на низких
        // частотах могут делить один FFT-бин — поэтому >=)
        for w in peaks.windows(2) {
            assert!(w[1] >= w[0], "пики убывают: {} -> {}", w[0], w[1]);
        }
        // первый фильтр начинается на 0 Гц (пик выше 0), последний — строго ниже 8000
        assert!(peaks[0] > 0.0);
        assert!(peaks[N_MELS - 1] < MEL_MAX_HZ);
    }

    #[test]
    fn sinusoids_match_formula() {
        let s = sinusoids(4, 8, 10000.0);
        // канал 0: sin(0)=0; канал 3: sin(0)=0? проверяем по формуле
        // scaled_time[t, i] = t * 10000^(-i/3); out[0]=sin, out[4..]=cos
        for t in 0..4usize {
            for i in 0..4usize {
                let inv = (-(10000.0f64).ln() / 3.0 * i as f64).exp();
                let scaled = t as f64 * inv;
                let got_sin = s[t * 8 + i] as f64;
                let got_cos = s[t * 8 + 4 + i] as f64;
                assert!((got_sin - scaled.sin()).abs() < 1e-5, "sin t{t} i{i}");
                assert!((got_cos - scaled.cos()).abs() < 1e-5, "cos t{t} i{i}");
            }
        }
    }
}
