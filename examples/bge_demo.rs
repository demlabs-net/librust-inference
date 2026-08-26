//! BGE-демо (Ф4.5): эмбеддинги bge-моделью на своём BERT-энкодере.
//!
//! Использование: `bge_demo <каталог модели> [--cpu|--metal] [текст...]`
//! Каталог должен содержать model.safetensors, config.json, vocab.txt.
//! Печатает эмбеддинги (первые 4 значения) и попарный косинус.
//! Без текстов — встроенный набор семантических пар (проверка качества).

use std::path::PathBuf;

use rust_inference::backend::default_backend;
use rust_inference::bert::{BertConfig, BertEncoder, WordPieceTokenizer};

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = PathBuf::from(args.first().map(String::as_str).unwrap_or("."));
    let texts: Vec<String> = args
        .iter()
        .skip(1)
        .filter(|a| !a.starts_with("--"))
        .cloned()
        .collect();
    let texts = if texts.is_empty() {
        vec![
            "A man is eating food.".into(),
            "A man is eating a piece of bread.".into(),
            "The girl is carrying a baby.".into(),
            "A man is riding a horse.".into(),
            "A woman is playing violin.".into(),
            "Two men pushed carts through the woods.".into(),
        ]
    } else {
        texts
    };

    let backend = default_backend()?;
    let tokenizer = WordPieceTokenizer::from_vocab_txt(&dir.join("vocab.txt"))?;
    let config = BertConfig::from_json(&dir.join("config.json"))?;
    println!(
        "BERT {}: h={}, слоёв={}, голов={}, vocab={}",
        dir.display(),
        config.hidden_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        tokenizer.vocab_size()
    );

    let enc = BertEncoder::load(backend.as_ref(), &dir, tokenizer, config)?;
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let emb = enc.embed(backend.as_ref(), &refs)?;
    for (i, (text, e)) in texts.iter().zip(&emb).enumerate() {
        println!(
            "[{i}] {text:?} → [{:.3}, {:.3}, {:.3}, {:.3} …] (dim {})",
            e[0],
            e[1],
            e[2],
            e[3],
            e.len()
        );
    }
    println!("\nкосинусная похожесть:");
    for i in 0..emb.len() {
        for j in (i + 1)..emb.len() {
            println!("  {i}×{j}: {:.3}", cosine(&emb[i], &emb[j]));
        }
    }
    Ok(())
}
