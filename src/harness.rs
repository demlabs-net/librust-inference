//! Test harness (Phase 2.2): deterministic input generation, golden-vector
//! persistence and tolerance-based comparison.
//!
//! Golden vectors are generated ONCE from the CPU reference backend
//! (`REGEN_GOLDEN=1 cargo test --features cpu`) and committed under
//! `tests/golden/`. Every backend then runs the same committed inputs and is
//! compared against the committed outputs with rtol/atol tolerance —
//! cross-backend GEMM/reduction orderings legitimately differ, so bit-exact
//! comparison is only used for same-backend determinism checks.
//!
//! Determinism: the LCG below mirrors `inference::sampling` (same constants)
//! so fixtures are reproducible across platforms.

use std::path::{Path, PathBuf};

use crate::backend::{Backend, Tensor};
use crate::cpu::CpuBackend;

/// LCG — same constants as `inference::sampling` (project convention).
#[derive(Debug, Clone)]
pub struct Lcg(pub u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    /// Advance the shared LCG step; see [`crate::inference::sampling`].
    pub fn next_u32(&mut self) -> u32 {
        self.0 = crate::inference::sampling::lcg_next(self.0);
        (self.0 >> 32) as u32
    }
    /// Uniform in [0, 1).
    ///
    /// Fixtures must span both signs: the previous version mixed the top 31
    /// bits down to 23 and divided by 2^24, so it returned [0, 0.5) and
    /// `next_signed` never produced a positive value — every generated weight,
    /// activation and golden input was negative.
    pub fn next_f32(&mut self) -> f32 {
        self.0 = crate::inference::sampling::lcg_next(self.0);
        crate::inference::sampling::lcg_unit_f32(self.0)
    }
    /// Uniform in [-1, 1).
    pub fn next_signed(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }
    pub fn fill(&mut self, out: &mut [f32]) {
        for v in out.iter_mut() {
            *v = self.next_signed();
        }
    }
}

/// A named tensor payload: data + shape.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TensorData {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// One op case: named inputs → named outputs.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GoldenCase {
    pub op: String,
    pub inputs: Vec<TensorData>,
    pub outputs: Vec<TensorData>,
}

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
}

pub fn golden_path(op: &str) -> PathBuf {
    golden_dir().join(format!("{op}.json"))
}

/// True when the golden files should be (re)generated instead of compared.
pub fn regen_enabled() -> bool {
    std::env::var("REGEN_GOLDEN")
        .map(|v| v == "1")
        .unwrap_or(false)
}

pub fn save_golden(case: &GoldenCase) -> std::io::Result<()> {
    let dir = golden_dir();
    std::fs::create_dir_all(&dir)?;
    let path = golden_path(&case.op);
    // K34: не перезаписываем существующий оракул молча — сбитый REGEN_GOLDEN=1
    // в окружении (CI/шелл) замаскировал бы регрессию. Поверх существующего
    // файла пишем только с явным REGEN_GOLDEN_FORCE=1.
    let force = std::env::var("REGEN_GOLDEN_FORCE")
        .map(|v| v == "1")
        .unwrap_or(false);
    if path.exists() && !force {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            format!(
                "{} already exists — refusing to silently overwrite; \
                 delete it or rerun with REGEN_GOLDEN_FORCE=1",
                path.display()
            ),
        ));
    }
    let json = serde_json::to_string_pretty(case).expect("serialize golden");
    std::fs::write(path, json)
}

pub fn load_golden(op: &str) -> std::io::Result<GoldenCase> {
    let json = std::fs::read_to_string(golden_path(op))?;
    serde_json::from_str(&json).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

/// rtol/atol comparison; on mismatch returns a description of the first
/// offending element. NaN propagates as a mismatch (never silently equal).
pub fn assert_close(actual: &[f32], expected: &[f32], rtol: f32, atol: f32) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!("len {} != {}", actual.len(), expected.len()));
    }
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let ok = if e.is_nan() || a.is_nan() {
            false
        } else {
            (a - e).abs() <= atol + rtol * e.abs()
        };
        if !ok {
            return Err(format!(
                "index {i}: got {a}, expected {e} (rtol {rtol}, atol {atol})"
            ));
        }
    }
    Ok(())
}

/// Standard tolerance for CPU↔GPU cross-backend comparison of one op.
pub fn standard_tolerance() -> (f32, f32) {
    (1e-4, 1e-5)
}

/// Wider tolerance for fused ops (naive CPU decomposition vs fused kernels).
pub fn fused_tolerance() -> (f32, f32) {
    (2e-3, 1e-4)
}

/// Run a golden case on the CPU reference backend and either compare against
/// the committed golden (default) or rewrite it (`REGEN_GOLDEN=1`).
///
/// `run` receives the input tensors (already uploaded to the backend) and
/// returns the produced output tensors (via `to_cpu_f32`).
pub fn check_op<F>(op: &str, seed: u64, shapes: &[(&str, Vec<usize>)], run: F)
where
    F: FnOnce(&CpuBackend, &[Tensor]) -> Vec<(String, Vec<usize>, Vec<f32>)>,
{
    let b = CpuBackend;
    // Deterministic inputs from the shared LCG — fixtures identical everywhere.
    let mut inputs = Vec::new();
    {
        let mut lcg = Lcg::new(seed);
        for (name, shape) in shapes {
            let n: usize = shape.iter().product();
            let mut data = vec![0.0f32; n];
            lcg.fill(&mut data);
            inputs.push(TensorData {
                name: name.to_string(),
                shape: shape.clone(),
                data,
            });
        }
    }

    let tensors: Vec<Tensor> = inputs
        .iter()
        .map(|t| {
            b.from_cpu_f32(&t.data, t.shape.clone())
                .expect("upload input")
        })
        .collect();

    let produced = run(&b, &tensors);
    let outputs: Vec<TensorData> = produced
        .into_iter()
        .map(|(name, shape, data)| TensorData { name, shape, data })
        .collect();

    let case = GoldenCase {
        op: op.to_string(),
        inputs,
        outputs,
    };

    if regen_enabled() {
        save_golden(&case).expect("write golden");
        return;
    }

    let golden = load_golden(op).unwrap_or_else(|e| {
        panic!("golden {op} not found ({e}) — run once with REGEN_GOLDEN=1 to generate")
    });
    let (rtol, atol) = standard_tolerance();
    for (got, exp) in case.outputs.iter().zip(golden.outputs.iter()) {
        assert_eq!(
            got.shape, exp.shape,
            "op {op}: shape mismatch for {}",
            got.name
        );
        assert_close(&got.data, &exp.data, rtol, atol)
            .unwrap_or_else(|e| panic!("op {op} tensor {}: {e}", got.name));
    }
}

/// Same-backend determinism: run the same op twice on the CPU backend and
/// require bit-identical output (fixed reduction orderings).
pub fn check_deterministic<F>(op: &str, seed: u64, shapes: &[(&str, Vec<usize>)], run: F)
where
    F: Fn(&CpuBackend, &[Tensor]) -> Vec<(String, Vec<usize>, Vec<f32>)>,
{
    let b = CpuBackend;
    let make = || {
        let mut lcg = Lcg::new(seed);
        let tensors: Vec<Tensor> = shapes
            .iter()
            .map(|(_name, shape)| {
                let n: usize = shape.iter().product();
                let mut data = vec![0.0f32; n];
                lcg.fill(&mut data);
                b.from_cpu_f32(&data, shape.clone()).expect("upload")
            })
            .collect();
        tensors
    };
    let first = run(&b, &make());
    let second = run(&b, &make());
    assert_eq!(first.len(), second.len(), "op {op}: output count changed");
    for ((n1, s1, d1), (n2, s2, d2)) in first.iter().zip(second.iter()) {
        assert_eq!(n1, n2, "op {op}: output name mismatch");
        assert_eq!(s1, s2, "op {op}: output shape mismatch for {n1}");
        assert_eq!(
            d1, d2,
            "op {op}: output NOT bit-identical for {n1} (determinism violated)"
        );
    }
}

/// Convenience: upload helper for a single tensor (used by op runners).
pub fn upload(b: &CpuBackend, data: &[f32], shape: Vec<usize>) -> Tensor {
    b.from_cpu_f32(data, shape).expect("upload")
}
