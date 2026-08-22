//! BPE-токенайзер из GGUF-метадаты (Ф1.5).
//!
//! Vocab: `tokenizer.ggml.tokens` (array of string), merges:
//! `tokenizer.ggml.merges` (array of "a b"), спец-токены:
//! `tokenizer.ggml.{bos,eos,padding}_token_id` (u32).
//!
//! Encode — байтовый BPE: старт с байтовых токенов (если в словаре есть
//! одно-байтовые куски, иначе байт пропускается как неизвестный), затем
//! жадное применение merge-пар по порядку ранга (v1; точная Qwen/Gemma
//! pretokenizer-регулярка — golden-тесты с реальными моделями в Ф1.7+).
//! Decode — конкатенация кусков; [`IncrementalDecoder`] корректно склеивает
//! неполные UTF-8 последовательности на границах чанков стрима ([А-3]).

use crate::gguf::{GgufFile, GgufValue};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerError {
    MissingVocab,
    BadTokenEntry(usize),
    BadMerge(String),
    MissingTokenId(&'static str),
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingVocab => write!(f, "tokenizer.ggml.tokens missing"),
            Self::BadTokenEntry(i) => write!(f, "tokenizer.ggml.tokens[{i}] is not a string"),
            Self::BadMerge(m) => write!(f, "bad merge entry `{m}`"),
            Self::MissingTokenId(k) => write!(f, "{k} missing"),
        }
    }
}

impl std::error::Error for TokenizerError {}

/// BPE-токенайзер модели.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    /// id → кусок (сырые байты).
    vocab: Vec<Vec<u8>>,
    /// кусок → id.
    id_of: HashMap<Vec<u8>, u32>,
    /// merge (left_id, right_id) → merged_id; порядок вектора = ранг.
    merges: Vec<((u32, u32), u32)>,
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub pad: Option<u32>,
}

impl Tokenizer {
    /// Собрать из готовых частей (для тестов и не-GGUF источников).
    pub fn from_parts(
        vocab: Vec<Vec<u8>>,
        merges: Vec<((u32, u32), u32)>,
        bos: Option<u32>,
        eos: Option<u32>,
        pad: Option<u32>,
    ) -> Self {
        let id_of = vocab
            .iter()
            .enumerate()
            .map(|(i, p)| (p.clone(), i as u32))
            .collect();
        Self {
            vocab,
            id_of,
            merges,
            bos,
            eos,
            pad,
        }
    }

    /// Из GGUF-метадаты.
    pub fn from_gguf(file: &GgufFile<'_>) -> Result<Self, TokenizerError> {
        let tokens = file
            .get("tokenizer.ggml.tokens")
            .and_then(GgufValue::as_array)
            .ok_or(TokenizerError::MissingVocab)?;
        let mut vocab = Vec::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            vocab.push(
                t.as_str()
                    .ok_or(TokenizerError::BadTokenEntry(i))?
                    .as_bytes()
                    .to_vec(),
            );
        }

        let mut id_of = vocab
            .iter()
            .enumerate()
            .map(|(i, p)| (p.clone(), i as u32))
            .collect::<HashMap<_, _>>();

        let mut merges = Vec::new();
        if let Some(arr) = file
            .get("tokenizer.ggml.merges")
            .and_then(GgufValue::as_array)
        {
            for m in arr {
                let s = m
                    .as_str()
                    .ok_or_else(|| TokenizerError::BadMerge("non-string".into()))?;
                let (l, r) = s
                    .split_once(' ')
                    .ok_or_else(|| TokenizerError::BadMerge(s.into()))?;
                let li = *id_of
                    .get(l.as_bytes())
                    .ok_or_else(|| TokenizerError::BadMerge(s.into()))?;
                let ri = *id_of
                    .get(r.as_bytes())
                    .ok_or_else(|| TokenizerError::BadMerge(s.into()))?;
                // merged-кусок: l+r (должен присутствовать в словаре)
                let mut merged = l.as_bytes().to_vec();
                merged.extend_from_slice(r.as_bytes());
                let mi = *id_of
                    .get(&merged)
                    .ok_or_else(|| TokenizerError::BadMerge(s.into()))?;
                merges.push(((li, ri), mi));
            }
        }

        let tid = |key: &'static str| -> Result<Option<u32>, TokenizerError> {
            match file.get(key) {
                None => Ok(None),
                Some(v) => v
                    .as_u32()
                    .map(Some)
                    .ok_or(TokenizerError::MissingTokenId(key)),
            }
        };
        Ok(Self {
            vocab,
            id_of,
            merges,
            bos: tid("tokenizer.ggml.bos_token_id")?,
            eos: tid("tokenizer.ggml.eos_token_id")?,
            pad: tid("tokenizer.ggml.padding_token_id")?,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn piece(&self, id: u32) -> &[u8] {
        self.vocab
            .get(id as usize)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Encode: байты → начальные токены (по одно-байтовым кускам словаря) →
    /// жадный BPE по рангам merges. Неизвестные байты пропускаются (v1).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids: Vec<u32> = Vec::new();
        for &b in text.as_bytes() {
            if let Some(&id) = self.id_of.get(&[b][..]) {
                ids.push(id);
            }
        }
        // жадный проход: ищем merge с наименьшим рангом среди соседних пар
        loop {
            let mut best: Option<(usize, usize, u32)> = None; // (pos, rank, merged_id)
            for (rank, ((l, r), m)) in self.merges.iter().enumerate() {
                if ids.len() < 2 {
                    break;
                }
                for i in 0..ids.len() - 1 {
                    if ids[i] == *l && ids[i + 1] == *r {
                        best = Some((i, rank, *m));
                        break;
                    }
                }
                if best.is_some() {
                    break;
                }
            }
            match best {
                Some((i, _, m)) => {
                    ids[i] = m;
                    ids.remove(i + 1);
                }
                None => break,
            }
        }
        ids
    }

    /// Decode: конкатенация кусков (lossy для битых последовательностей).
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = Vec::new();
        for &id in ids {
            out.extend_from_slice(self.piece(id));
        }
        String::from_utf8_lossy(&out).into_owned()
    }
}

/// Инкрементальный декодер для стриминга: куски могут обрываться на границе
/// UTF-8-символа — хвост неполной последовательности удерживается до
/// следующего куска ([А-3], Ф1.5).
#[derive(Debug, Default)]
pub struct IncrementalDecoder {
    pending: Vec<u8>,
}

impl IncrementalDecoder {
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    /// Добавить кусок, вернуть валидный текст (неполный хвост удерживается).
    pub fn push(&mut self, piece: &[u8]) -> String {
        self.pending.extend_from_slice(piece);
        let mut out = String::new();
        let mut i = 0usize;
        while i < self.pending.len() {
            let b = self.pending[i];
            let len = utf8_char_len(b);
            if len == 0 || i + len > self.pending.len() {
                break; // неполный/невалидный хвост — держим в pending
            }
            let chunk = &self.pending[i..i + len];
            if std::str::from_utf8(chunk).is_err() {
                break; // битые байты — держим (потеря минимальна, стрим продолжается)
            }
            out.push_str(unsafe { std::str::from_utf8_unchecked(chunk) });
            i += len;
        }
        if i > 0 {
            self.pending.drain(..i);
        }
        out
    }

    /// Сбросить остаток (lossy).
    pub fn finish(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }
}

/// Длина UTF-8 символа по первому байту (0 = невалидный старт).
fn utf8_char_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b >= 0xC2 && b <= 0xDF {
        2
    } else if b >= 0xE0 && b <= 0xEF {
        3
    } else if b >= 0xF0 && b <= 0xF4 {
        4
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(name: &str) -> Vec<u8> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join(name);
        std::fs::read(&path).unwrap_or_else(|e| panic!("fixture {name}: {e}"))
    }

    /// Синтетический словарь: байты + полная merge-цепочка для "hello".
    fn synthetic() -> Tokenizer {
        let mut vocab: Vec<Vec<u8>> = (0..=255).map(|b| vec![b]).collect();
        let he: Vec<u8> = b"he".to_vec();
        let ll: Vec<u8> = b"ll".to_vec();
        let llo: Vec<u8> = b"llo".to_vec();
        let hello: Vec<u8> = b"hello".to_vec();
        vocab.push(he.clone());
        vocab.push(ll.clone());
        vocab.push(llo.clone());
        vocab.push(hello.clone());
        let ids =
            |p: &[u8]| -> u32 { vocab.iter().position(|v| v.as_slice() == p).unwrap() as u32 };
        let merges = vec![
            ((ids(b"h"), ids(b"e")), ids(&he)),
            ((ids(b"l"), ids(b"l")), ids(&ll)),
            ((ids(&ll), ids(b"o")), ids(&llo)),
            ((ids(&he), ids(&llo)), ids(&hello)),
        ];
        Tokenizer::from_parts(vocab, merges, None, Some(0), Some(0))
    }

    #[test]
    fn merge_applied_greedily() {
        let t = synthetic();
        let ids = t.encode("hello");
        // полная цепочка merges должна схлопнуть 5 байтов в один токен
        assert_eq!(
            ids,
            vec![t.id_of.get(b"hello".as_slice()).copied().unwrap()]
        );
        assert_eq!(t.decode(&ids), "hello");
    }

    #[test]
    fn encode_decode_roundtrip_ascii() {
        let t = synthetic();
        let text = "hello world hello";
        let ids = t.encode(text);
        assert_eq!(t.decode(&ids), text);
    }

    #[test]
    fn from_gguf_mini_fixture() {
        let buf = fixture("mini-f16.gguf");
        let f = crate::gguf::parse(&buf).unwrap();
        let t = Tokenizer::from_gguf(&f).unwrap();
        assert_eq!(t.vocab_size(), 3);
        assert_eq!(t.decode(&[0]), "<pad>");
        assert_eq!(t.decode(&[1]), " hello");
        assert_eq!(t.decode(&[2]), " world");
    }

    #[test]
    fn incremental_decode_utf8_boundaries() {
        let mut d = IncrementalDecoder::new();
        // «пр» = [0xD0 0xBF, 0xD1 0x80]; режем по байтам
        assert_eq!(d.push(&[0xD0]), "");
        assert_eq!(d.push(&[0xBF]), "п");
        assert_eq!(d.push(&[0xD1]), "");
        assert_eq!(d.push(&[0x80]), "р");
        assert_eq!(d.push(b"!"), "!");
        assert_eq!(d.finish(), "");
    }

    #[test]
    fn incremental_decode_ascii_stream() {
        let mut d = IncrementalDecoder::new();
        assert_eq!(d.push(b"hel"), "hel");
        assert_eq!(d.push(b"lo"), "lo");
        assert_eq!(d.finish(), "");
    }
}
