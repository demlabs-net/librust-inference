//! LoRA-обучение (Ф3): Adam на [`Backend`], LoRA-адаптеры, `Trainer`-трейт,
//! `TrainJob`/`JobStore` — перенесено из vs-train (генерик, без vs-типов).
//!
//! Эталонный тренер — `demo-lora` (малый MLP 4→16→1 + синтетические данные):
//! прогоняет весь конвейер (жизненный цикл джобы, Adam, прогресс, экспорт
//! `adapter.json`) на любом бэкенде. Подключение реальных семейств моделей
//! на тот же интерфейс — следующий шаг (Ф3.5).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::backend::{Backend, Tensor};

/// Deterministic seed so runs are reproducible.
const SEED: u64 = 42;
/// Input/output dims of the demo model.
const IN_DIM: usize = 4;
const HID_DIM: usize = 16;
const OUT_DIM: usize = 1;
/// Demo dataset size (batches slice it).
const DATASET_N: usize = 512;

/// Жизненный цикл джобы обучения.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum JobStatus {
    Queued,
    Running {
        step: usize,
        total: usize,
        loss: Option<f64>,
    },
    Done,
    Cancelled,
    Failed(String),
}

impl std::fmt::Display for JobStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JobStatus::Queued => write!(f, "queued"),
            JobStatus::Running { step, total, loss } => {
                write!(f, "running {step}/{total}")?;
                if let Some(l) = loss {
                    write!(f, " loss={l:.4}")?;
                }
                Ok(())
            }
            JobStatus::Done => write!(f, "done"),
            JobStatus::Cancelled => write!(f, "cancelled"),
            JobStatus::Failed(e) => write!(f, "failed: {e}"),
        }
    }
}

/// Одна джоба LoRA-обучения.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainJob {
    pub id: String,
    /// Идентификатор базовой модели (строка; в Vassista — id манифеста
    /// vs-model-manager).
    pub model_id: String,
    /// Тренер: v1 — `demo-lora`.
    pub trainer: String,
    pub lora_rank: usize,
    pub epochs: usize,
    pub lr: f64,
    pub batch_size: usize,
    pub status: JobStatus,
    pub created_at: f64,
    /// Выходной КАТАЛОГ адаптеров; `output` (файл внутри) — при завершении.
    #[serde(default)]
    pub output_dir: Option<PathBuf>,
    pub output: Option<PathBuf>,
    /// Этап 5.1: early-stop — сколько шагов без улучшения (None = выкл).
    #[serde(default)]
    pub patience: Option<usize>,
    /// Этап 5.1: период чекпойнтов (шагов); чекпойнты — в output_dir/.
    #[serde(default)]
    pub checkpoint_every: usize,
    /// Этап 5.1: resume — восстанавливаться из чекпойнта (внутри output_dir).
    #[serde(default)]
    pub resume_from: Option<PathBuf>,
    /// Этап 5.3: датасет (файл) и его формат (sharegpt/alpaca/chatml/raw/dpo).
    #[serde(default)]
    pub dataset_path: Option<PathBuf>,
    #[serde(default)]
    pub dataset_format: Option<String>,
}

impl TrainJob {
    pub fn new(id: impl Into<String>, model_id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            model_id: model_id.into(),
            trainer: "demo-lora".into(),
            lora_rank: 4,
            epochs: 200,
            lr: 1e-3,
            batch_size: 32,
            status: JobStatus::Queued,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0),
            output_dir: None,
            output: None,
            patience: None,
            checkpoint_every: 0,
            resume_from: None,
            dataset_path: None,
            dataset_format: None,
        }
    }

    /// Валидация гиперпараметров ДО того, как джоба коснётся VRAM/GPU.
    /// Непроверенные значения ранее доходили до тренера как есть
    /// (batch_size=0 — деление на ноль, >512 — тихо обучало ничто).
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.batch_size == 0 || self.batch_size > DATASET_N {
            anyhow::bail!(
                "batch_size must be in 1..={DATASET_N} (got {}) — the demo dataset has {DATASET_N} samples",
                self.batch_size
            );
        }
        if self.epochs == 0 {
            anyhow::bail!("epochs must be >= 1 (got 0)");
        }
        if self.lora_rank == 0 {
            anyhow::bail!("lora_rank must be >= 1 (got 0)");
        }
        if !self.lr.is_finite() || self.lr <= 0.0 {
            anyhow::bail!("lr must be finite and > 0 (got {})", self.lr);
        }
        if let Some(p) = &self.dataset_path {
            if !p.is_file() {
                anyhow::bail!("dataset file not found: {}", p.display());
            }
            if let Some(f) = &self.dataset_format {
                if crate::datasets::DatasetFormat::parse(f).is_none() {
                    anyhow::bail!("unknown dataset format: {f}");
                }
            }
        }
        Ok(())
    }

    pub fn is_terminal(&self) -> bool {
        matches!(
            self.status,
            JobStatus::Done | JobStatus::Cancelled | JobStatus::Failed(_)
        )
    }
}

/// JSON-хранилище джоб: один файл на джобу в `jobs_dir`.
#[derive(Clone)]
pub struct JobStore {
    jobs_dir: PathBuf,
}

impl JobStore {
    pub fn new(jobs_dir: impl Into<PathBuf>) -> Self {
        Self {
            jobs_dir: jobs_dir.into(),
        }
    }

    /// Каталог хранилища (для спавна рабочих копий).
    pub fn dir(&self) -> &Path {
        &self.jobs_dir
    }

    fn path(&self, id: &str) -> anyhow::Result<PathBuf> {
        // `id` приходит из CLI — без проверки `id = "../../x"` читал бы/
        // писал/удалял файлы вне каталога хранилища.
        if id.is_empty()
            || !id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
        {
            anyhow::bail!("invalid job id `{id}` (allowed: [A-Za-z0-9._-])");
        }
        Ok(self.jobs_dir.join(format!("{id}.json")))
    }

    pub fn save(&self, job: &TrainJob) -> anyhow::Result<()> {
        std::fs::create_dir_all(&self.jobs_dir)?;
        let text = serde_json::to_string_pretty(job)?;
        // Атомарно: temp + rename (обрыв/параллельное сохранение прогресса
        // из воркера не оставит рваный JSON).
        let path = self.path(&job.id)?;
        let tmp = self.jobs_dir.join(format!("{}.tmp", job.id));
        std::fs::write(&tmp, text)?;
        std::fs::rename(&tmp, &path)?;
        Ok(())
    }

    pub fn load(&self, id: &str) -> anyhow::Result<TrainJob> {
        let text = std::fs::read_to_string(self.path(id)?)?;
        Ok(serde_json::from_str(&text)?)
    }

    pub fn list(&self) -> anyhow::Result<Vec<TrainJob>> {
        let mut jobs = Vec::new();
        if !self.jobs_dir.is_dir() {
            return Ok(jobs);
        }
        for entry in std::fs::read_dir(&self.jobs_dir)? {
            let entry = entry?;
            if entry.path().extension().and_then(|e| e.to_str()) == Some("json") {
                if let Ok(text) = std::fs::read_to_string(entry.path()) {
                    if let Ok(job) = serde_json::from_str::<TrainJob>(&text) {
                        jobs.push(job);
                    }
                }
            }
        }
        jobs.sort_by(|a, b| {
            a.created_at
                .partial_cmp(&b.created_at)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(jobs)
    }

    pub fn delete(&self, id: &str) -> anyhow::Result<()> {
        let path = self.path(id)?;
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }
}

/// Интерфейс тренера: `on_progress(step, total, loss)` может вызываться много
/// раз из потока тренера; `stop` — флаг кооперативной отмены (проверяется
/// между шагами). Возвращает путь к экспортированному адаптеру.
/// Прогресс-событие тренера (Этап 5.1): шаг, потери, прошедшее время
/// (tokens_per_sec/vram — когда бэкенд отдаёт, сейчас None).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TrainProgress {
    pub step: usize,
    pub total_steps: usize,
    pub loss: Option<f64>,
    pub elapsed_ms: f64,
    /// Опционально: ток-с обучения и VRAM МБ (бэкенд-зависимые).
    pub tokens_per_sec: Option<f64>,
    pub vram_mb: Option<u64>,
}

pub trait Trainer: Send + Sync {
    fn name(&self) -> &str;
    fn run(
        &self,
        job: &TrainJob,
        stop: &std::sync::atomic::AtomicBool,
        on_progress: &dyn Fn(&TrainProgress),
    ) -> anyhow::Result<PathBuf>;
}

/// Формат экспорта LoRA-адаптера (`adapter.json`).
#[derive(Serialize)]
pub struct AdapterExport {
    pub model: String,
    pub trainer: String,
    pub rank: usize,
    pub in_dim: usize,
    pub out_dim: usize,
    pub scale: f64,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
}

/// Конструктор бэкенда по умолчанию: Metal на macOS, CUDA — по фиче.
/// (Снят compile_error на не-macOS — выбор бэкенда теперь в либе, Ф3.2.)
#[cfg(feature = "metal")]
pub fn default_backend() -> anyhow::Result<Box<dyn Backend>> {
    Ok(Box::new(crate::metal::MetalBackend::new()?))
}

#[cfg(all(not(feature = "metal"), feature = "cuda"))]
pub fn default_backend() -> anyhow::Result<Box<dyn Backend>> {
    Ok(Box::new(crate::cuda::CudaBackend::new(0)?))
}

// Build shim for the cpu-only configuration (e.g. plain `cargo test` with
// default features): the demo trainer needs a real backend, but the crate
// must still COMPILE without metal/cuda — the run() fails with a clear
// message at runtime instead of a compile error.
#[cfg(all(not(feature = "metal"), not(feature = "cuda")))]
pub fn default_backend() -> anyhow::Result<Box<dyn Backend>> {
    anyhow::bail!("demo-lora needs a real backend (cuda or metal) — cpu-only build")
}

fn rng() -> rand::rngs::StdRng {
    rand::SeedableRng::seed_from_u64(SEED)
}

/// Синтетический датасет (на хосте): N×4 входов, полиномиальные цели.
fn dataset(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = rng();
    let x: Vec<f32> = (0..n * IN_DIM)
        .map(|_| rand::Rng::random_range(&mut rng, -1.0..1.0))
        .collect();
    let y: Vec<f32> = x
        .chunks_exact(IN_DIM)
        .map(|row| row[0] * row[1] + row[2] - row[3])
        .collect();
    (x, y)
}

/// Замороженные базовые веса (генерация на хосте): W1 (16×4), b1, W2, b2.
fn base_weights() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = rng();
    let w1: Vec<f32> = (0..HID_DIM * IN_DIM)
        .map(|_| rand::Rng::random_range(&mut rng, -0.5..0.5))
        .collect();
    let b1 = vec![0.0f32; HID_DIM];
    let w2: Vec<f32> = (0..OUT_DIM * HID_DIM)
        .map(|_| rand::Rng::random_range(&mut rng, -0.5..0.5))
        .collect();
    let b2 = vec![0.0f32; OUT_DIM];
    (w1, b1, w2, b2)
}

/// Кэш прямого прохода для обратного.
struct Cache {
    pre: Tensor,
    lora_b: Tensor,
}

/// Forward: `out = tanh(x·W1ᵀ + b1 + LoRA(x)) · W2ᵀ + b2`,
/// `LoRA(x) = (x·Aᵀ)·Bᵀ·scale`. Все операции — через [`Backend`].
#[allow(clippy::too_many_arguments)]
fn forward(
    bk: &dyn Backend,
    x: &Tensor,
    a: &Tensor,
    b: &Tensor,
    w1: &Tensor,
    b1: &Tensor,
    w2: &Tensor,
    b2: &Tensor,
    scale: f32,
    n: usize,
    r: usize,
) -> Result<(Tensor, Cache), crate::error::BackendError> {
    let lora_b = bk.zeros(vec![n, r])?;
    bk.gemm_nt(x, a, &lora_b)?;
    let lora = bk.zeros(vec![n, HID_DIM])?;
    bk.gemm_nt(&lora_b, b, &lora)?;
    let lora_s = bk.zeros(vec![n, HID_DIM])?;
    bk.scale(&lora, &lora_s, scale)?;
    let base = bk.zeros(vec![n, HID_DIM])?;
    bk.gemm_nt(x, w1, &base)?;
    bk.add_bias(&base, b1, &base)?;
    let pre = bk.zeros(vec![n, HID_DIM])?;
    bk.add(&base, &lora_s, &pre)?;
    let h = bk.zeros(vec![n, HID_DIM])?;
    bk.tanh(&pre, &h)?;
    let out = bk.zeros(vec![n, OUT_DIM])?;
    bk.gemm_nt(&h, w2, &out)?;
    bk.add_bias(&out, b2, &out)?;
    Ok((out, Cache { pre, lora_b }))
}

/// MSE-лосс + градиенты по A и B.
#[allow(clippy::too_many_arguments)]
fn backward(
    bk: &dyn Backend,
    x: &Tensor,
    y: &Tensor,
    out: &Tensor,
    b: &Tensor,
    w2: &Tensor,
    cache: &Cache,
    scale: f32,
    n: usize,
    r: usize,
) -> Result<(f64, Tensor, Tensor), crate::error::BackendError> {
    let diff = bk.zeros(vec![n, OUT_DIM])?;
    bk.sub(out, y, &diff)?;
    let diff_s = bk.zeros(vec![n, OUT_DIM])?;
    bk.scale(&diff, &diff_s, 2.0 / n as f32)?;
    // loss (скаляр на хосте — метрика, не инференс).
    let out_v = bk.to_cpu_f32(out)?;
    let y_v = bk.to_cpu_f32(y)?;
    let loss = out_v
        .iter()
        .zip(y_v.iter())
        .map(|(o, t)| ((o - t) as f64) * ((o - t) as f64) / n as f64)
        .sum();

    // d_lora (B,16) = d_out·W2 ⊙ (1 − pre²)
    let d_h = bk.zeros(vec![n, HID_DIM])?;
    bk.gemm(&diff_s, w2, &d_h)?;
    let pre2 = bk.zeros(vec![n, HID_DIM])?;
    bk.mul(&cache.pre, &cache.pre, &pre2)?;
    let one_m = bk.zeros(vec![n, HID_DIM])?;
    bk.scale(&pre2, &one_m, -1.0)?;
    bk.add_scalar(&one_m, &one_m, 1.0)?;
    let d_lora = bk.zeros(vec![n, HID_DIM])?;
    bk.mul(&d_h, &one_m, &d_lora)?;

    // g_b (16,r) = d_loraᵀ · (lora_b · scale)
    let lora_s = bk.zeros(vec![n, r])?;
    bk.scale(&cache.lora_b, &lora_s, scale)?;
    let g_b = bk.zeros(vec![HID_DIM, r])?;
    // Tensor::view — row-major RESHAPE, не транспонирование (исторический
    // баг: градиенты LoRA были неверны, smoke-тест их не ловил).
    let d_lora_t = bk.transpose(&d_lora)?;
    bk.gemm(&d_lora_t, &lora_s, &g_b)?;

    // g_a (r,4) = (d_lora·B·scale)ᵀ · x
    let d_b = bk.zeros(vec![n, r])?;
    bk.gemm(&d_lora, b, &d_b)?;
    let d_b_s = bk.zeros(vec![n, r])?;
    bk.scale(&d_b, &d_b_s, scale)?;
    let g_a = bk.zeros(vec![r, IN_DIM])?;
    let d_b_t = bk.transpose(&d_b_s)?;
    bk.gemm(&d_b_t, x, &g_a)?;

    Ok((loss, g_a, g_b))
}

/// Обычный Adam (без weight decay) для обучаемых адаптеров — генерик по
/// [`Backend`] (Ф3.1: перенесён из vs-train).
pub struct Adam {
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    t: usize,
    m: std::collections::HashMap<String, Tensor>,
    v: std::collections::HashMap<String, Tensor>,
}

impl Adam {
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
            m: Default::default(),
            v: Default::default(),
        }
    }

    /// Один шаг для параметра с его градиентом (in place, на бэкенде).
    pub fn step(
        &mut self,
        bk: &dyn Backend,
        name: &str,
        param: &mut Tensor,
        grad: &Tensor,
    ) -> Result<(), crate::error::BackendError> {
        self.t += 1;
        let shape = grad.shape().to_vec();
        let (b1, b2) = (self.beta1 as f32, self.beta2 as f32);
        let m = self
            .m
            .entry(name.to_string())
            .or_insert_with(|| bk.zeros(shape.clone()).unwrap());
        let v = self
            .v
            .entry(name.to_string())
            .or_insert_with(|| bk.zeros(shape.clone()).unwrap());
        let m_hat_c = (1.0 / (1.0 - self.beta1.powi(self.t as i32))) as f32;
        let v_hat_c = (1.0 / (1.0 - self.beta2.powi(self.t as i32))) as f32;

        let mut new_m = bk.zeros(shape.clone())?;
        bk.scale(m, &mut new_m, b1)?;
        let mut g1 = bk.zeros(shape.clone())?;
        bk.scale(grad, &mut g1, 1.0 - b1)?;
        bk.add(&new_m, &g1, m)?;
        let mut new_v = bk.zeros(shape.clone())?;
        bk.scale(v, &mut new_v, b2)?;
        let mut g2 = bk.zeros(shape.clone())?;
        bk.mul(grad, grad, &mut g2)?;
        let mut g2s = bk.zeros(shape.clone())?;
        bk.scale(&g2, &mut g2s, 1.0 - b2)?;
        bk.add(&new_v, &g2s, v)?;
        let mut m_hat = bk.zeros(shape.clone())?;
        bk.scale(m, &mut m_hat, m_hat_c)?;
        let mut v_hat = bk.zeros(shape.clone())?;
        bk.scale(v, &mut v_hat, v_hat_c)?;
        let mut vs = bk.zeros(shape.clone())?;
        bk.sqrt(&v_hat, &mut vs)?;
        let mut vse = bk.zeros(shape.clone())?;
        bk.add_scalar(&vs, &mut vse, self.eps as f32)?;
        let mut upd = bk.zeros(shape.clone())?;
        bk.div(&m_hat, &vse, &mut upd)?;
        let mut upds = bk.zeros(shape.clone())?;
        bk.scale(&upd, &mut upds, self.lr as f32)?;
        let mut next = bk.zeros(shape.clone())?;
        bk.sub(param, &upds, &mut next)?;
        *param = next;
        Ok(())
    }
}

/// Эталонный тренер v1: малый MLP + синтетика — референс конвейера
/// (жизненный цикл, Adam, прогресс, экспорт).
pub struct DemoLoraTrainer;

impl Trainer for DemoLoraTrainer {
    fn name(&self) -> &str {
        "demo-lora"
    }

    fn run(
        &self,
        job: &TrainJob,
        stop: &std::sync::atomic::AtomicBool,
        on_progress: &dyn Fn(&TrainProgress),
    ) -> anyhow::Result<PathBuf> {
        let r = job.lora_rank;
        let backend = default_backend()?;
        tracing::info!(rank = r, "demo-lora training start");

        let out_dir = job
            .output_dir
            .clone()
            .or_else(|| job.output.clone())
            .unwrap_or_else(|| std::env::temp_dir().join("vassista-train"));
        std::fs::create_dir_all(&out_dir)?;
        let checkpoint_path = job
            .resume_from
            .clone()
            .unwrap_or_else(|| out_dir.join("checkpoint.json"));

        let (w1, b1, w2, b2) = base_weights();
        let w1 = backend.from_cpu_f32(&w1, vec![HID_DIM, IN_DIM])?;
        let b1 = backend.from_cpu_f32(&b1, vec![HID_DIM])?;
        let w2 = backend.from_cpu_f32(&w2, vec![OUT_DIM, HID_DIM])?;
        let b2 = backend.from_cpu_f32(&b2, vec![OUT_DIM])?;

        // Этап 5.1: resume из чекпойнта (шаг + a/b) либо инициализация.
        let mut start_step = 0usize;
        let mut a_init: Vec<f32> = (0..r * IN_DIM)
            .map(|_| rand::Rng::random_range(&mut rng(), -0.1..0.1))
            .collect();
        let mut b_init: Vec<f32> = vec![0.0f32; HID_DIM * r];
        if checkpoint_path.exists() {
            if let Some(cp) = load_checkpoint(&checkpoint_path) {
                if cp.a.len() == a_init.len() && cp.b.len() == b_init.len() {
                    a_init = cp.a;
                    b_init = cp.b;
                    start_step = cp.step;
                    tracing::info!(step = cp.step, loss = ?cp.loss, "resume from checkpoint");
                }
            }
        }
        let mut a = backend.from_cpu_f32(&a_init, vec![r, IN_DIM])?;
        let mut b = backend.from_cpu_f32(&b_init, vec![HID_DIM, r])?;
        let scale = 1.0f32; // alpha == rank

        let (x, y) = dataset(DATASET_N);
        let batch = job.batch_size;
        let steps_per_epoch = DATASET_N / batch;
        let total_steps = steps_per_epoch * job.epochs;
        let checkpoint_every = job.checkpoint_every.max(10);
        let patience = job.patience;
        let mut adam = Adam::new(job.lr);
        let mut last_loss: Option<f64> = None;
        let mut best_loss: Option<f64> = None;
        let mut no_improve_steps = 0usize;
        let started = std::time::Instant::now();
        let mut progress = |step: usize, total_steps: usize, loss: Option<f64>| {
            on_progress(&TrainProgress {
                step,
                total_steps,
                loss,
                elapsed_ms: started.elapsed().as_millis() as f64,
                tokens_per_sec: None,
                vram_mb: None,
            });
        };

        for epoch in 0..job.epochs {
            if stop.load(std::sync::atomic::Ordering::Relaxed) {
                anyhow::bail!("cancelled by user");
            }
            for bi in 0..steps_per_epoch {
                let step = epoch * steps_per_epoch + bi + 1;
                if step <= start_step {
                    continue; // resume: шаги до чекпойнта уже выполнены
                }
                let start = bi * batch;
                let xb = backend.from_cpu_f32(
                    &x[start * IN_DIM..(start + batch) * IN_DIM],
                    vec![batch, IN_DIM],
                )?;
                let yb = backend.from_cpu_f32(&y[start..start + batch], vec![batch, OUT_DIM])?;
                let (out, cache) = forward(
                    backend.as_ref(),
                    &xb,
                    &a,
                    &b,
                    &w1,
                    &b1,
                    &w2,
                    &b2,
                    scale,
                    batch,
                    r,
                )?;
                let (loss, g_a, g_b) = backward(
                    backend.as_ref(),
                    &xb,
                    &yb,
                    &out,
                    &b,
                    &w2,
                    &cache,
                    scale,
                    batch,
                    r,
                )?;
                adam.step(backend.as_ref(), "a", &mut a, &g_a)?;
                adam.step(backend.as_ref(), "b", &mut b, &g_b)?;
                last_loss = Some(loss);

                // Чекпойнт: шаг + a/b — для resume/прерывания.
                if step % checkpoint_every == 0 {
                    let cp_a = backend.to_cpu_f32(&a)?;
                    let cp_b = backend.to_cpu_f32(&b)?;
                    save_checkpoint(&checkpoint_path, step, &cp_a, &cp_b, Some(loss))?;
                }

                // Early-stop по плато train-loss (val-loss — с 5.3/split).
                if let Some(p) = patience {
                    match best_loss {
                        None => best_loss = Some(loss),
                        Some(bl) if loss >= bl * 0.999 => {
                            no_improve_steps += 1;
                            if no_improve_steps >= p {
                                tracing::info!(step, loss, "early-stop (no improvement)");
                                progress(step, total_steps, last_loss);
                                return export_adapter(job, &out_dir, backend.as_ref(), &a, &b, r, scale);
                            }
                        }
                        Some(bl) => {
                            if loss < bl {
                                best_loss = Some(loss);
                                no_improve_steps = 0;
                            }
                        }
                    }
                }

                if step % 10 == 0 || step == total_steps {
                    progress(step, total_steps, last_loss);
                }
            }
        }
        progress(total_steps, total_steps, last_loss);
        tracing::info!(loss = ?last_loss, "demo-lora training done");
        export_adapter(job, &out_dir, backend.as_ref(), &a, &b, r, scale)
    }
}

/// Экспорт adapter.json (атомарно: tmp + rename).
fn export_adapter(
    job: &TrainJob,
    out_dir: &Path,
    backend: &dyn Backend,
    a: &Tensor,
    b: &Tensor,
    r: usize,
    scale: f32,
) -> anyhow::Result<PathBuf> {
    let export = AdapterExport {
        model: job.model_id.clone(),
        trainer: job.trainer.clone(),
        rank: r,
        in_dim: IN_DIM,
        out_dim: HID_DIM,
        scale: scale as f64,
        a: backend.to_cpu_f32(a)?,
        b: backend.to_cpu_f32(b)?,
    };
    let path = out_dir.join("adapter.json");
    let tmp = out_dir.join("adapter.json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(&export)?)?;
    std::fs::rename(&tmp, &path)?;
    Ok(path)
}

/// Чекпойнт: JSON {step, a, b, loss} — ф32-векторы на CPU.
#[derive(Serialize, Deserialize)]
pub struct TrainCheckpoint {
    pub step: usize,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub loss: Option<f64>,
}

fn save_checkpoint(
    path: &Path,
    step: usize,
    a: &[f32],
    b: &[f32],
    loss: Option<f64>,
) -> anyhow::Result<()> {
    let cp = TrainCheckpoint {
        step,
        a: a.to_vec(),
        b: b.to_vec(),
        loss,
    };
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(&cp)?)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

pub fn load_checkpoint(path: &Path) -> Option<TrainCheckpoint> {
    let raw = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&raw).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Полный smoke обучения: лосс падает, файл адаптера появляется.
    /// Требует реальный бэкенд (metal/cuda) — на cpu-only сборке
    /// default_backend() даёт осмысленную ошибку, тест не запускается.
    #[cfg(any(feature = "metal", feature = "cuda"))]
    #[test]
    fn demo_lora_loss_decreases_and_exports() {
        let mut job = TrainJob::new("test", "demo-model");
        job.epochs = 20;
        let out_dir = std::env::temp_dir().join(format!("train-test-{}", std::process::id()));
        job.output_dir = Some(out_dir.clone());

        let losses: std::sync::Mutex<Vec<f64>> = std::sync::Mutex::new(Vec::new());
        let trainer = DemoLoraTrainer;
        let stop = std::sync::atomic::AtomicBool::new(false);
        let path = trainer
            .run(&job, &stop, &|_step, _total, loss| {
                if let Some(l) = loss {
                    losses.lock().unwrap().push(l);
                }
            })
            .expect("training must run");

        let losses = losses.lock().unwrap();
        assert!(!losses.is_empty(), "progress must report losses");
        let first = losses.first().copied().unwrap();
        let last = losses.last().copied().unwrap();
        assert!(
            last < first,
            "loss must decrease: first={first:.4} last={last:.4}"
        );
        assert!(
            path.exists(),
            "adapter must be exported: {}",
            path.display()
        );
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("\"rank\""));
        let _ = std::fs::remove_dir_all(&out_dir);
    }

    /// JobStore: round-trip + list + валидация.
    #[test]
    fn job_store_roundtrip() {
        let dir = std::env::temp_dir().join(format!("train-store-{}", std::process::id()));
        let store = JobStore::new(&dir);
        let mut job = TrainJob::new("j1", "m1");
        job.status = JobStatus::Running {
            step: 1,
            total: 10,
            loss: Some(0.5),
        };
        store.save(&job).unwrap();
        let loaded = store.load("j1").unwrap();
        assert_eq!(loaded.id, "j1");
        assert!(matches!(loaded.status, JobStatus::Running { .. }));
        assert_eq!(store.list().unwrap().len(), 1);
        // path-валидация: traversal-ид отклоняется
        assert!(store.path("../../etc").is_err());
        store.delete("j1").unwrap();
        assert!(store.load("j1").is_err());
        assert!(job.validate().is_ok());
        job.batch_size = 0;
        assert!(job.validate().is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod tests51 {
    use super::*;

    #[test]
    fn checkpoint_roundtrip_and_resume_path() {
        let dir = std::env::temp_dir().join(format!("tr51-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("checkpoint.json");
        save_checkpoint(&path, 42, &[1.0, 2.0, 3.0], &[4.0, 5.0], Some(0.123)).unwrap();
        let cp = load_checkpoint(&path).unwrap();
        assert_eq!(cp.step, 42);
        assert_eq!(cp.a, vec![1.0, 2.0, 3.0]);
        assert_eq!(cp.b, vec![4.0, 5.0]);
        assert_eq!(cp.loss, Some(0.123));
        // resume_from: джоба с чекпойнтом читает его (путь существует)
        let mut job = TrainJob::new("j", "m");
        job.resume_from = Some(path.clone());
        assert!(job.resume_from.as_ref().unwrap().exists());
        assert!(load_checkpoint(&path).is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
