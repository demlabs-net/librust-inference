//! Датасеты файнтюнинга (Этап 5.3): парсеры форматов (sharegpt/alpaca/
//! chatml/raw/DPO-пары), профиль (длины/дубли/язык), split train/val/test.
//! Примеры — пары (prompt, completion) — вход для SFT-тренеров (5.2b).

use std::collections::BTreeMap;
use std::path::Path;

/// Один пример SFT: промпт → завершение.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Example {
    pub prompt: String,
    pub completion: String,
    /// DPO: отброшенный вариант — для пар (иначе None).
    #[serde(default)]
    pub rejected: Option<String>,
}

/// Формат набора (по имени файла/параметру).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetFormat {
    Sharegpt,
    Alpaca,
    Chatml,
    Raw,
    Dpo,
}

impl DatasetFormat {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "sharegpt" => Some(Self::Sharegpt),
            "alpaca" => Some(Self::Alpaca),
            "chatml" => Some(Self::Chatml),
            "raw" => Some(Self::Raw),
            "dpo" => Some(Self::Dpo),
            _ => None,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sharegpt => "sharegpt",
            Self::Alpaca => "alpaca",
            Self::Chatml => "chatml",
            Self::Raw => "raw",
            Self::Dpo => "dpo",
        }
    }
}

/// Профиль набора: статистика для UI/валидации.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DatasetProfile {
    pub format: String,
    pub count: usize,
    pub avg_prompt_chars: f64,
    pub max_prompt_chars: usize,
    pub avg_completion_chars: f64,
    pub exact_duplicates: usize,
    /// Примерная доля примеров с кириллицей (0..1).
    pub cyrillic_share: f64,
    pub total_chars: u64,
}

/// Разобрать файл в примеры (JSON или построчный JSONL; raw — по абзацам).
pub fn parse_dataset(path: &Path, format: DatasetFormat) -> anyhow::Result<Vec<Example>> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    let is_jsonl = trimmed.contains('\n') && trimmed.lines().all(|l| l.trim().starts_with('{'));
    let items: Vec<serde_json::Value> = if is_jsonl {
        trimmed
            .lines()
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l.trim()).ok())
            .collect()
    } else {
        match serde_json::from_str::<serde_json::Value>(trimmed) {
            Ok(v) => match v {
                serde_json::Value::Array(a) => a,
                other => vec![other],
            },
            Err(_) => return Ok(raw_to_examples(&raw, format)),
        }
    };
    Ok(items
        .iter()
        .filter_map(|v| json_to_example(v, format))
        .collect())
}

fn raw_to_examples(raw: &str, format: DatasetFormat) -> Vec<Example> {
    // raw: абзацы → completion; промпт = пустой (последовательное обучение).
    if format == DatasetFormat::Raw {
        return raw
            .split("\n\n")
            .filter(|p| p.trim().len() > 20)
            .map(|p| Example {
                prompt: String::new(),
                completion: p.trim().to_string(),
                rejected: None,
            })
            .collect();
    }
    Vec::new()
}

/// Текст из поля (строка или список {type,text}).
fn content_of(v: &serde_json::Value, key: &str) -> Option<String> {
    let f = v.get(key)?;
    if let Some(s) = f.as_str() {
        return Some(s.to_string());
    }
    if let Some(arr) = f.as_array() {
        let mut out = String::new();
        for part in arr {
            if let Some(t) = part.get("text").and_then(|x| x.as_str()) {
                out.push_str(t);
            }
        }
        return Some(out);
    }
    None
}

fn json_to_example(v: &serde_json::Value, format: DatasetFormat) -> Option<Example> {
    match format {
        DatasetFormat::Chatml => {
            let msgs = v.get("messages")?.as_array()?;
            let mut prompt = String::new();
            let mut completion = String::new();
            for (i, m) in msgs.iter().enumerate() {
                let role = m.get("role")?.as_str()?;
                let content = m.get("content")?.as_str()?;
                if role == "user" {
                    prompt.push_str(content);
                    prompt.push('\n');
                } else if role == "assistant" {
                    completion.push_str(content);
                    completion.push('\n');
                } else if role == "system" && i == 0 {
                    prompt = format!("Система: {content}\n");
                }
            }
            if completion.trim().is_empty() {
                None
            } else {
                Some(Example {
                    prompt,
                    completion: completion.trim_end().to_string(),
                    rejected: None,
                })
            }
        }
        DatasetFormat::Sharegpt => {
            let msgs = v.get("conversations")?;
            let mut prompt = String::new();
            let mut completion = String::new();
            for m in msgs.as_array()? {
                let from = m.get("from").and_then(|x| x.as_str())?;
                let value = m.get("value").and_then(|x| x.as_str())?;
                if from == "user" || from == "human" {
                    prompt.push_str(value);
                    prompt.push('\n');
                } else if from == "gpt" || from == "assistant" {
                    completion.push_str(value);
                    completion.push('\n');
                }
            }
            if completion.trim().is_empty() {
                None
            } else {
                Some(Example {
                    prompt,
                    completion: completion.trim_end().to_string(),
                    rejected: None,
                })
            }
        }
        DatasetFormat::Alpaca => {
            let mut prompt = v
                .get("instruction")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            if let Some(input) = v.get("input").and_then(|x| x.as_str())
                && !input.is_empty()
            {
                prompt.push('\n');
                prompt.push_str(input);
            }
            let completion = v.get("output").and_then(|x| x.as_str())?.to_string();
            Some(Example {
                prompt,
                completion,
                rejected: None,
            })
        }
        DatasetFormat::Dpo => {
            let prompt = content_of(v, "prompt")?;
            let chosen = content_of(v, "chosen")?;
            let rejected = content_of(v, "rejected");
            Some(Example {
                prompt,
                completion: chosen,
                rejected,
            })
        }
        DatasetFormat::Raw => None,
    }
}

/// Профиль набора: статистика.
pub fn profile(examples: &[Example], format: DatasetFormat) -> DatasetProfile {
    let mut dup = 0usize;
    let mut seen: BTreeMap<u32, usize> = BTreeMap::new();
    let mut total_chars = 0u64;
    let mut cyrillic = 0usize;
    let mut sum_p = 0.0f64;
    let mut sum_c = 0.0f64;
    let mut max_p = 0usize;
    for e in examples {
        let hash = simple_hash(&e.prompt) ^ simple_hash(&e.completion).rotate_left(11);
        *seen.entry(hash).or_insert(0) += 1;
        let plen = e.prompt.chars().count();
        let clen = e.completion.chars().count();
        sum_p += plen as f64;
        sum_c += clen as f64;
        max_p = max_p.max(plen);
        total_chars += (plen + clen) as u64;
        if e.prompt.chars().any(|c| ('а'..='я').contains(&c)) {
            cyrillic += 1;
        }
    }
    for n in seen.values() {
        if *n > 1 {
            dup += n - 1;
        }
    }
    DatasetProfile {
        format: format.as_str().to_string(),
        count: examples.len(),
        avg_prompt_chars: if examples.is_empty() {
            0.0
        } else {
            sum_p / examples.len() as f64
        },
        max_prompt_chars: max_p,
        avg_completion_chars: if examples.is_empty() {
            0.0
        } else {
            sum_c / examples.len() as f64
        },
        exact_duplicates: dup,
        cyrillic_share: if examples.is_empty() {
            0.0
        } else {
            cyrillic as f64 / examples.len() as f64
        },
        total_chars,
    }
}

/// Разбиение train/val/test (детерминированное — по seed).
pub fn split_examples(
    examples: &[Example],
    train: f64,
    val: f64,
    test: f64,
    seed: u64,
) -> (Vec<Example>, Vec<Example>, Vec<Example>) {
    let total = examples.len();
    if total == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let mut idx: Vec<usize> = (0..total).collect();
    let mut rng = 0x9e3779b97f4a7c15u64 ^ seed;
    for i in (1..total).rev() {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (rng % (i as u64 + 1)) as usize;
        idx.swap(i, j);
    }
    let mut n_train = ((total as f64) * train).round() as usize;
    let n_val = ((total as f64) * val).round() as usize;
    let n_test = ((total as f64) * test).round() as usize;
    let extra = total.saturating_sub(n_train + n_val + n_test);
    n_train += extra; // остаток — в train
    let mut pick = |n: usize| {
        idx.drain(0..n)
            .map(|i| examples[i].clone())
            .collect::<Vec<_>>()
    };
    (
        pick(n_train),
        if n_val > 0 { pick(n_val) } else { Vec::new() },
        if n_test > 0 { pick(n_test) } else { Vec::new() },
    )
}

fn simple_hash(s: &str) -> u32 {
    let mut h = 0x811c9dc5u32;
    for b in s.bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(0x01000193);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_alpaca_and_profile() {
        let dir = std::env::temp_dir().join(format!("ds-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let p = dir.join("a.json");
        std::fs::write(
            &p,
            serde_json::json!([
                {"instruction": "Скажи привет", "input": "", "output": "Привет!"},
                {"instruction": "Сложи", "input": "2 и 2", "output": "4"},
                {"instruction": "Дубль", "input": "", "output": "Ответ"}
            ])
            .to_string(),
        )
        .unwrap();
        let exs = parse_dataset(&p, DatasetFormat::Alpaca).unwrap();
        assert_eq!(exs.len(), 3);
        assert_eq!(exs[0].prompt, "Скажи привет");
        assert_eq!(exs[1].prompt, "Сложи\n2 и 2");
        let prof = profile(&exs, DatasetFormat::Alpaca);
        assert_eq!(prof.count, 3);
        assert!(
            prof.cyrillic_share > 0.9,
            "кириллица: {}",
            prof.cyrillic_share
        );
        let (tr, va, te) = split_examples(&exs, 0.8, 0.1, 0.1, 42);
        assert_eq!(tr.len() + va.len() + te.len(), 3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_sharegpt_jsonl_and_dpo() {
        let dir = std::env::temp_dir().join(format!("ds2-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let p = dir.join("s.jsonl");
        std::fs::write(
            &p,
            r#"{"conversations":[{"from":"user","value":"ча"}]}
{"conversations":[{"from":"user","value":"привет"},{"from":"gpt","value":"здравствуй"}]}
"#,
        )
        .unwrap();
        let exs = parse_dataset(&p, DatasetFormat::Sharegpt).unwrap();
        assert_eq!(exs.len(), 1, "первая строка без ответа отбрасывается");
        assert_eq!(exs[0].completion, "здравствуй");
        let p2 = dir.join("d.json");
        std::fs::write(
            &p2,
            serde_json::json!([
                {"prompt": "вопрос", "chosen": "хороший", "rejected": "плохой"}
            ])
            .to_string(),
        )
        .unwrap();
        let exs2 = parse_dataset(&p2, DatasetFormat::Dpo).unwrap();
        assert_eq!(exs2[0].completion, "хороший");
        assert_eq!(exs2[0].rejected.as_deref(), Some("плохой"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
