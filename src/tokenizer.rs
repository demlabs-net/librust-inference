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
    /// Спец-токены (CONTROL/USER_DEFINED из token_type), longest-first.
    special: Vec<(Vec<u8>, u32)>,
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub pad: Option<u32>,
}

impl Tokenizer {
    /// Добавить спец-токен единым куском (HF added_tokens special=true).
    /// Поддерживает longest-first порядок.
    pub fn add_special(&mut self, piece: Vec<u8>, id: u32) {
        self.special.push((piece, id));
        self.special.sort_by_key(|p| std::cmp::Reverse(p.0.len()));
    }
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
            special: Vec::new(),
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

        let id_of = vocab
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

        // Спец-токены из token_type (ggml): 1=NORMAL, 2=UNKNOWN, 3=CONTROL,
        // 4=USER_DEFINED, 5=UNUSED, 6=BYTE. CONTROL/USER_DEFINED вырезаются
        // едиными кусками до BPE (иначе `<|im_start|>` распадается на части —
        // модель видит не тот промпт).
        let mut special: Vec<(Vec<u8>, u32)> = Vec::new();
        if let Some(arr) = file
            .get("tokenizer.ggml.token_type")
            .and_then(GgufValue::as_array)
        {
            for (i, t) in arr.iter().enumerate() {
                if let Some(tt) = t.as_u32()
                    && (tt == 3 || tt == 4)
                    && vocab[i].len() > 1
                {
                    special.push((vocab[i].clone(), i as u32));
                }
            }
        }
        // longest-first: жадный матч берёт самый длинный спец-токен.
        special.sort_by_key(|p| std::cmp::Reverse(p.0.len()));

        Ok(Self {
            vocab,
            id_of,
            merges,
            special,
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

    /// Encode: GPT-2 byte-level (каждый UTF-8 байт → символ по таблице:
    /// 0x20→Ġ, 0x0A→Ċ, 0xD0→Ð, 0x82→Ĥ, ...), затем байты → начальные токены
    /// → жадный BPE по рангам merges. Неизвестные байты пропускаются (v1).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let (btu, _) = unicode_tables();
        let mut mapped = String::with_capacity(text.len());
        for &b in text.as_bytes() {
            mapped.push(btu[b as usize]);
        }
        // инициализация по СИМВОЛАМ (tiktoken-стиль): каждый символ — токен;
        // неизвестные символы — фоллбэк на байты (byte-fallback токены 0x00-0xFF).
        // Спец-токены (CONTROL/USER_DEFINED) вырезаются едиными кусками ДО BPE.
        let mapped_bytes = mapped.as_bytes();
        let mut ids: Vec<u32> = Vec::new();
        let mut i = 0usize;
        while i < mapped_bytes.len() {
            let mut matched: Option<(usize, u32)> = None;
            for (piece, id) in &self.special {
                if mapped_bytes[i..].starts_with(piece) {
                    matched = Some((piece.len(), *id));
                    break;
                }
            }
            if let Some((len, id)) = matched {
                ids.push(id);
                i += len;
                continue;
            }
            let ch_len = utf8_len(mapped_bytes[i]);
            let piece = &mapped_bytes[i..i + ch_len];
            if let Some(&id) = self.id_of.get(piece) {
                ids.push(id);
            } else {
                for &b in piece {
                    if let Some(&id) = self.id_of.get(&[b][..]) {
                        ids.push(id);
                    }
                }
            }
            i += ch_len;
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

    /// Decode: конкатенация кусков + обратный byte-level (Ġ→0x20, Ð→0xD0, ...).
    pub fn decode(&self, ids: &[u32]) -> String {
        let (_, utb) = unicode_tables();
        let mut bytes = Vec::new();
        for &id in ids {
            for c in String::from_utf8_lossy(self.piece(id)).chars() {
                if let Some(&b) = utb.get(&c) {
                    bytes.push(b);
                } else {
                    let mut buf = [0u8; 4];
                    bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Кусок токена как текст (для стриминга): обратный byte-level.
    pub fn decode_piece(&self, id: u32) -> String {
        let (_, utb) = unicode_tables();
        let mut bytes = Vec::new();
        for c in String::from_utf8_lossy(self.piece(id)).chars() {
            if let Some(&b) = utb.get(&c) {
                bytes.push(b);
            } else {
                let mut buf = [0u8; 4];
                bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
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
            // N (iter-9): an invalid head used to wedge the decoder FOREVER —
            // the scan restarted at byte 0 every time, so once the head was
            // permanently broken (a lone continuation byte, or a lead byte
            // whose continuation never arrived), every push broke at i == 0,
            // nothing was ever drained and `pending` grew by the whole
            // remaining stream (the comment's "stream continues" was wrong).
            if len == 0 {
                // Continuation byte at a char boundary — invalid: skip lossily.
                out.push(char::REPLACEMENT_CHARACTER);
                i += 1;
                continue;
            }
            if i + len > self.pending.len() {
                // The char is incomplete so far. If the byte right after the
                // lead can never continue it (not a continuation byte), the
                // char is broken — flush the lead lossily instead of holding
                // a tail that can never become valid.
                let broken = self.pending[i + 1..]
                    .first()
                    .map(|&n| !(0x80..=0xBF).contains(&n))
                    .unwrap_or(false);
                if broken {
                    out.push(char::REPLACEMENT_CHARACTER);
                    i += 1;
                    continue;
                }
                break; // неполный хвост — ждём продолжения
            }
            let chunk = &self.pending[i..i + len];
            if std::str::from_utf8(chunk).is_err() {
                // Битые байты: отдаём lossy и двигаемся (не застреваем).
                out.push_str(&String::from_utf8_lossy(chunk));
                i += len;
                continue;
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

/// Обратная tiktoken-маппинг: Ġ (U+0120)→пробел, Ċ (U+010A)→\n.
/// GPT-2 byte-level таблица (как `bytes_to_unicode` в transformers):
/// байты 0x21-0x7E, 0xA1-0xAC, 0xAE-0xFF — сами (latin-1); остальные
/// (0x00-0x20, 0x7F-0xA0, 0xAD) — U+0100+n в порядке байта. 0x20→Ġ, 0x0A→Ċ.
fn unicode_tables() -> &'static ([char; 256], std::collections::HashMap<char, u8>) {
    use std::collections::HashMap;
    use std::sync::OnceLock;
    static T: OnceLock<([char; 256], HashMap<char, u8>)> = OnceLock::new();
    T.get_or_init(|| {
        let mut tbl = [char::from(0u8); 256];
        let mut n = 0u32;
        for b in 0u8..=255u8 {
            let in_bs = (0x21..=0x7E).contains(&b)
                || (0xA1..=0xAC).contains(&b)
                || (0xAE..=0xFF).contains(&b);
            if in_bs {
                tbl[b as usize] = char::from_u32(b as u32).unwrap();
            } else {
                tbl[b as usize] = char::from_u32(0x100 + n).unwrap();
                n += 1;
            }
        }
        let rev = tbl.iter().enumerate().map(|(i, &c)| (c, i as u8)).collect();
        (tbl, rev)
    })
}

/// Длина UTF-8 символа по первому байту (0 = невалидный старт).
fn utf8_char_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if (0xC2..=0xDF).contains(&b) {
        2
    } else if (0xE0..=0xEF).contains(&b) {
        3
    } else if (0xF0..=0xF4).contains(&b) {
        4
    } else {
        0
    }
}

/// Число байт UTF-8-символа по первому байту.
fn utf8_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b >> 5 == 0b110 {
        2
    } else if b >> 4 == 0b1110 {
        3
    } else {
        4
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
        vocab.push("\u{0120}".as_bytes().to_vec()); // Ġ — byte-level пробел
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

    /// N (iter-9): a permanently broken head must not wedge the decoder —
    /// the old code broke at i == 0 forever and lost the whole stream.
    #[test]
    fn incremental_decoder_survives_broken_head() {
        // Lone continuation byte at the head (never valid).
        let mut d = IncrementalDecoder::new();
        assert_eq!(d.push(&[0x80]), "\u{FFFD}");
        assert_eq!(d.push(b"ab"), "ab");
        assert_eq!(d.push(&[0x80, 0x41]), "\u{FFFD}A");
        assert!(d.pending.is_empty(), "pending: {:?}", d.pending);

        // Lead byte whose continuation never arrives, then more text.
        let mut d2 = IncrementalDecoder::new();
        assert_eq!(d2.push(&[0xE2]), ""); // может стать символом — ждём
        assert_eq!(d2.push(b"x"), "\u{FFFD}x"); // 0xE2 + 'x' никогда не валиден
        assert_eq!(d2.push(&[0xE2, 0x82, 0xAC]), "\u{20AC}"); // € (3 байта)
        assert!(d2.pending.is_empty(), "pending: {:?}", d2.pending);
    }

    #[test]
    fn tiktoken_space_mapping() {
        // словарь: байтовые токены + "Ġ" (символ) + "he"/"llo"/"hello" + merge
        let mut vocab: Vec<Vec<u8>> = (0..=255).map(|b| vec![b]).collect();
        let g: Vec<u8> = "\u{0120}".as_bytes().to_vec();
        let he: Vec<u8> = b"he".to_vec();
        let llo: Vec<u8> = b"llo".to_vec();
        let hello: Vec<u8> = b"hello".to_vec();
        vocab.push(g.clone());
        vocab.push(he.clone());
        vocab.push(llo.clone());
        vocab.push(hello.clone());
        let ids_of = |p: &[u8]| -> u32 { vocab.iter().position(|v| v == p).unwrap() as u32 };
        let hello_id = ids_of(&hello);
        let merges = vec![((ids_of(&he), ids_of(&llo)), hello_id)];
        let t = Tokenizer::from_parts(vocab, merges, None, None, None);
        let ids = t.encode("hi hello");
        // пробел → символ Ġ (id 256); полная merge-цепочка — отдельный тест
        assert!(ids.contains(&256), "Ġ token missing: ids={ids:?}");
        let _ = hello_id;
        assert_eq!(t.decode(&ids), "hi hello");
        assert_eq!(t.decode_piece(256), " ");
    }

    #[test]
    fn incremental_decode_ascii_stream() {
        let mut d = IncrementalDecoder::new();
        assert_eq!(d.push(b"hel"), "hel");
        assert_eq!(d.push(b"lo"), "lo");
        assert_eq!(d.finish(), "");
    }
}
