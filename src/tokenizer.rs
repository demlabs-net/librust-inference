//! BPE-токенайзер из GGUF-метадаты (Ф1.5).
//!
//! Vocab: `tokenizer.ggml.tokens` (array of string), merges:
//! `tokenizer.ggml.merges` (array of "a b"), спец-токены:
//! `tokenizer.ggml.{bos,eos,padding}_token_id` (u32).
//!
//! Encode повторяет llama.cpp (`llm_tokenizer_bpe_session::tokenize`):
//! 1. спец-токены (CONTROL/USER_DEFINED) вырезаются из текста единым куском;
//! 2. остальной текст пред-токенизатор режет на «слова» по регулярке типа
//!    `tokenizer.ggml.pre` (порт `unicode_regex_split_custom_*`, категории
//!    символов — из таблиц того же коммита llama.cpp, см. unicode_data.rs);
//! 3. внутри слова — byte-level BPE: пары по рангу из приоритетной очереди
//!    (O(n log n) на слово), устаревшие пары отбрасываются при извлечении.
//!
//! Слияния через границы слов невозможны — как у эталона. Раньше пред-
//! токенизации не было, а лучшая пара искалась полным проходом по всем
//! merges на каждое слияние: 30 тыс. символов кодировались больше 20 минут,
//! и токены расходились с llama.cpp на границах слов.
//!
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
    /// `tokenizer.ggml.pre` без порта регулярки: молча резать «как-нибудь»
    /// нельзя — модель увидит не те токены.
    UnsupportedPreTokenizer(String),
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingVocab => write!(f, "tokenizer.ggml.tokens missing"),
            Self::BadTokenEntry(i) => write!(f, "tokenizer.ggml.tokens[{i}] is not a string"),
            Self::BadMerge(m) => write!(f, "bad merge entry `{m}`"),
            Self::MissingTokenId(k) => write!(f, "{k} missing"),
            Self::UnsupportedPreTokenizer(p) => {
                write!(f, "tokenizer.ggml.pre = `{p}`: пред-токенизатор не поддержан")
            }
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
    /// (left_id, right_id) → (ранг, merged_id); ранг = порядок в merges.
    merge_rank: HashMap<(u32, u32), (u32, u32)>,
    /// Спец-токены (CONTROL/USER_DEFINED из token_type), longest-first.
    special: Vec<(Vec<u8>, u32)>,
    /// Регулярка разбиения на слова (`tokenizer.ggml.pre`).
    pre: PreTokenizer,
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub pad: Option<u32>,
    /// Токены конца генерации (EOG): eos + все «конечные» спец-токены
    /// словаря — правило `special_eog_ids` llama.cpp. Одного eos мало: у
    /// Qwen3 eos = `<|im_end|>`, а base-модель заканчивает текст
    /// `<|endoftext|>` — генерация шла дальше него в мусор.
    eog: Vec<u32>,
}

/// Тексты токенов, завершающих генерацию (llama.cpp `special_eog_ids`,
/// 5266f24da) + FIM-разделители Qwen (fim_pad/fim_repo/file_sep и аналоги).
const EOG_TEXTS: &[&str] = &[
    "<|eot_id|>",
    "<|im_end|>",
    "<|end|>",
    "<|return|>",
    "<|call|>",
    "<|flush|>",
    "<|calls|>",
    "<end_of_turn>",
    "<|endoftext|>",
    "</s>",
    "<|eom_id|>",
    "<EOT>",
    "_<EOT>",
    "[EOT]",
    "[EOS]",
    "<|end_of_text|>",
    "<end_of_utterance>",
    "<eos>",
    "<turn|>",
    "<|tool_response>",
    "<｜end▁of▁sentence｜>",
    "[e~[",
    "<|fim_pad|>",
    "<fim-pad>",
    "<fim_pad>",
    "<PAD>",
    "[PAD]",
    "<|fim_repo|>",
    "<|repo_name|>",
    "<fim-repo>",
    "<REPO>",
    "<reponame>",
    "<|file_sep|>",
];

fn eog_set(id_of: &HashMap<Vec<u8>, u32>, eos: Option<u32>) -> Vec<u32> {
    let mut v: Vec<u32> = EOG_TEXTS
        .iter()
        .filter_map(|t| id_of.get(t.as_bytes()).copied())
        .collect();
    v.extend(eos);
    v.sort_unstable();
    v.dedup();
    v
}

/// Пред-токенизатор: на какие «слова» режется текст до BPE. Порт
/// `unicode_regex_split_custom_*` llama.cpp (5266f24da).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreTokenizer {
    /// Qwen2/Qwen3/Qwen3-VL (`qwen2`, `deepseek-r1-qwen`, …):
    /// `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
    ///  ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
    Qwen2,
    /// Qwen3.5 (`qwen35`): как Qwen2, но буквенные серии поглощают и
    /// диакритику `\p{M}`.
    Qwen35,
    /// GLM-4/ChatGLM (`glm4`, `chatglm-bpe`): как Qwen2, но цифры берутся
    /// сериями по 1-3 (`\p{N}{1,3}`), а не по одной.
    ChatGlm4,
}

impl PreTokenizer {
    /// Имя из `tokenizer.ggml.pre` → тип; таблица соответствий — как в
    /// `llama_vocab::impl::load` llama.cpp. Отсутствие ключа у GPT-2-словаря
    /// llama.cpp трактует как `default`, которого у нас нет: явная ошибка.
    pub fn from_gguf_name(name: &str) -> Result<Self, TokenizerError> {
        match name {
            "qwen2" | "deepseek-r1-qwen" | "kormo" | "f2llmv2" => Ok(Self::Qwen2),
            "qwen35" => Ok(Self::Qwen35),
            "glm4" | "chatglm-bpe" => Ok(Self::ChatGlm4),
            other => Err(TokenizerError::UnsupportedPreTokenizer(other.to_string())),
        }
    }
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
        let eog = eog_set(&id_of, eos);
        Self {
            vocab,
            id_of,
            merge_rank: rank_table(&merges),
            special: Vec::new(),
            pre: PreTokenizer::Qwen2,
            bos,
            eos,
            pad,
            eog,
        }
    }

    /// Токен завершает генерацию (EOG-набор, см. поле `eog`).
    pub fn is_eog(&self, id: u32) -> bool {
        self.eog.binary_search(&id).is_ok()
    }

    /// Сменить пред-токенизатор (источники не из GGUF: tokenizer.json Omni).
    pub fn with_pre(mut self, pre: PreTokenizer) -> Self {
        self.pre = pre;
        self
    }

    pub fn pre_tokenizer(&self) -> PreTokenizer {
        self.pre
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
        // 4=USER_DEFINED, 5=UNUSED, 6=BYTE. UNKNOWN/CONTROL/USER_DEFINED
        // вырезаются едиными кусками до BPE (иначе `<|im_start|>` распадается
        // на части — модель видит не тот промпт) — тот же набор, что
        // `cache_special_tokens` llama.cpp.
        let mut special: Vec<(Vec<u8>, u32)> = Vec::new();
        if let Some(arr) = file
            .get("tokenizer.ggml.token_type")
            .and_then(GgufValue::as_array)
        {
            for (i, t) in arr.iter().enumerate() {
                if let Some(tt) = t.as_u32()
                    && (tt == 2 || tt == 3 || tt == 4)
                    && !vocab[i].is_empty()
                {
                    special.push((vocab[i].clone(), i as u32));
                }
            }
        }
        // longest-first: жадный матч берёт самый длинный спец-токен
        // (стабильная сортировка — при равной длине порядок id, как у
        // llama.cpp `cache_special_tokens`).
        special.sort_by_key(|p| std::cmp::Reverse(p.0.len()));

        let pre = match file.get("tokenizer.ggml.pre").and_then(GgufValue::as_str) {
            Some(name) => PreTokenizer::from_gguf_name(name)?,
            None => {
                return Err(TokenizerError::UnsupportedPreTokenizer(
                    "(ключ tokenizer.ggml.pre отсутствует)".into(),
                ));
            }
        };

        let eos = tid("tokenizer.ggml.eos_token_id")?;
        // eot_id/eom_id из метаданных — тоже конец генерации (как llama.cpp)
        let mut eog = eog_set(&id_of, eos);
        for k in ["tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"] {
            if let Some(id) = tid(k)? {
                eog.push(id);
            }
        }
        eog.sort_unstable();
        eog.dedup();
        Ok(Self {
            vocab,
            id_of,
            merge_rank: rank_table(&merges),
            special,
            pre,
            bos: tid("tokenizer.ggml.bos_token_id")?,
            eos,
            pad: tid("tokenizer.ggml.padding_token_id")?,
            eog,
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

    /// Encode (см. шапку модуля): спец-токены → слова → BPE внутри слова.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::with_capacity(text.len() / 3 + 8);
        let mut cache: HashMap<&str, Vec<u32>> = HashMap::new();
        for frag in self.partition_special(text) {
            match frag {
                Fragment::Special(id) => ids.push(id),
                Fragment::Text(t) => {
                    for word in pre_split(t, self.pre) {
                        if let Some(hit) = cache.get(word) {
                            ids.extend_from_slice(hit);
                            continue;
                        }
                        let w = self.bpe_word(word);
                        ids.extend_from_slice(&w);
                        cache.insert(word, w);
                    }
                }
            }
        }
        ids
    }

    /// Вырезать спец-токены (порт `tokenizer_st_partition`, parse_special):
    /// для каждого спец-токена — longest-first — делим ещё не разобранные
    /// текстовые фрагменты по его вхождениям. Атрибутов lstrip/rstrip в
    /// GGUF нет — пробелы вокруг остаются тексту.
    fn partition_special<'t>(&self, text: &'t str) -> Vec<Fragment<'t>> {
        let mut frags = vec![Fragment::Text(text)];
        for (piece, id) in &self.special {
            // Текст спец-токена в GGUF — СЫРОЙ (added tokens не проходят
            // byte-level-отображение), как и в llama.cpp: ищем как есть.
            let Ok(needle) = std::str::from_utf8(piece) else {
                continue;
            };
            let needle = needle.to_string();
            if needle.is_empty()
                || !frags
                    .iter()
                    .any(|f| matches!(f, Fragment::Text(t) if t.contains(needle.as_str())))
            {
                continue;
            }
            let mut next = Vec::with_capacity(frags.len() + 4);
            for f in frags {
                match f {
                    Fragment::Text(t) if t.contains(needle.as_str()) => {
                        let mut rest = t;
                        while let Some(pos) = rest.find(needle.as_str()) {
                            if pos > 0 {
                                next.push(Fragment::Text(&rest[..pos]));
                            }
                            next.push(Fragment::Special(*id));
                            rest = &rest[pos + needle.len()..];
                        }
                        if !rest.is_empty() {
                            next.push(Fragment::Text(rest));
                        }
                    }
                    other => next.push(other),
                }
            }
            frags = next;
        }
        frags
    }

    /// Byte-level BPE одного слова: символы — byte-level-отображение байтов
    /// UTF-8; соседние пары сливаются по возрастанию ранга (при равном ранге
    /// — левая раньше), как `llm_bigram_bpe::comparator` llama.cpp.
    fn bpe_word(&self, word: &str) -> Vec<u32> {
        let (btu, _) = unicode_tables();
        // Символы слова: по одному на исходный UTF-8-символ (как в
        // llama.cpp: символ = UTF-8-последовательность УЖЕ отображённого
        // текста; отображение байт-в-символ, так что длина в символах =
        // числу байтов исходника).
        let mut sym_ids: Vec<Option<u32>> = Vec::with_capacity(word.len());
        let mut pieces: Vec<String> = Vec::with_capacity(word.len());
        for &b in word.as_bytes() {
            let c = btu[b as usize];
            let mut tmp = [0u8; 4];
            let s = c.encode_utf8(&mut tmp);
            pieces.push(s.to_string());
            sym_ids.push(self.id_of.get(s.as_bytes()).copied());
        }
        let n = pieces.len();
        if n == 0 {
            return Vec::new();
        }
        // Связный список символов.
        let mut prev: Vec<isize> = (0..n as isize).map(|i| i - 1).collect();
        let mut next: Vec<isize> = (0..n as isize)
            .map(|i| if i + 1 < n as isize { i + 1 } else { -1 })
            .collect();
        let mut alive = vec![true; n];
        // Куча: (ранг, левый индекс, id левого, id правого, merged).
        let mut heap: std::collections::BinaryHeap<std::cmp::Reverse<(u32, usize, u32, u32, u32)>> =
            std::collections::BinaryHeap::new();
        let push = |heap: &mut std::collections::BinaryHeap<_>, l: usize, r: usize, ids: &[Option<u32>]| {
            if let (Some(a), Some(b)) = (ids[l], ids[r])
                && let Some(&(rank, m)) = self.merge_rank.get(&(a, b))
            {
                heap.push(std::cmp::Reverse((rank, l, a, b, m)));
            }
        };
        for i in 0..n - 1 {
            push(&mut heap, i, i + 1, &sym_ids);
        }
        while let Some(std::cmp::Reverse((_rank, l, a, b, m))) = heap.pop() {
            if !alive[l] {
                continue;
            }
            let r = next[l];
            if r < 0 {
                continue;
            }
            let r = r as usize;
            // Пара устарела, если любой из символов уже стал другим токеном.
            if sym_ids[l] != Some(a) || sym_ids[r] != Some(b) {
                continue;
            }
            sym_ids[l] = Some(m);
            alive[r] = false;
            next[l] = next[r];
            if next[r] >= 0 {
                prev[next[r] as usize] = l as isize;
            }
            if prev[l] >= 0 {
                push(&mut heap, prev[l] as usize, l, &sym_ids);
            }
            if next[l] >= 0 {
                push(&mut heap, l, next[l] as usize, &sym_ids);
            }
        }
        let mut out = Vec::with_capacity(n);
        let mut i: isize = 0;
        while i >= 0 {
            let iu = i as usize;
            match sym_ids[iu] {
                Some(id) => out.push(id),
                // Символа нет в словаре (у byte-level-словарей не бывает:
                // все 256 отображённых байтов — токены): как llama.cpp —
                // по токену на каждый байт куска, если такой есть.
                None => {
                    for &byte in pieces[iu].as_bytes() {
                        if let Some(&id) = self.id_of.get(&[byte][..]) {
                            out.push(id);
                        }
                    }
                }
            }
            i = next[iu];
        }
        out
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

    /// Кусок токена как текст (обратный byte-level). Токен-обрывок
    /// многобайтного символа даёт U+FFFD — для стриминга брать
    /// [`Self::piece_bytes`] через `IncrementalDecoder`.
    pub fn decode_piece(&self, id: u32) -> String {
        String::from_utf8_lossy(&self.piece_bytes(id)).into_owned()
    }

    /// Сырые байты текста токена (обратный byte-level): у эмодзи и редких
    /// символов, разбитых на несколько токенов, это неполный UTF-8.
    pub fn piece_bytes(&self, id: u32) -> Vec<u8> {
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
        bytes
    }
}

/// Фрагмент текста после вырезания спец-токенов.
enum Fragment<'t> {
    Text(&'t str),
    Special(u32),
}

/// (left_id, right_id) → (ранг, merged); при повторе пары берётся первый
/// (минимальный) ранг, как в llama.cpp `bpe_ranks.emplace`.
fn rank_table(merges: &[((u32, u32), u32)]) -> HashMap<(u32, u32), (u32, u32)> {
    let mut t = HashMap::with_capacity(merges.len());
    for (rank, &((l, r), m)) in merges.iter().enumerate() {
        t.entry((l, r)).or_insert((rank as u32, m));
    }
    t
}

/// Флаги кодовой точки (биты как `unicode_cpt_flags` llama.cpp).
#[derive(Clone, Copy, Default)]
struct CptFlags(u16);

impl CptFlags {
    const UNDEFINED: u16 = 0x0001;
    const NUMBER: u16 = 0x0002;
    const LETTER: u16 = 0x0004;
    const ACCENT_MARK: u16 = 0x0010;
    const WHITESPACE: u16 = 0x0100;

    fn is_number(self) -> bool {
        self.0 & Self::NUMBER != 0
    }
    fn is_letter(self) -> bool {
        self.0 & Self::LETTER != 0
    }
    fn is_accent_mark(self) -> bool {
        self.0 & Self::ACCENT_MARK != 0
    }
    fn is_whitespace(self) -> bool {
        self.0 & Self::WHITESPACE != 0
    }
    /// `flags.as_uint() != 0` в llama.cpp: за пределами фрагмента флагов нет.
    fn any(self) -> bool {
        self.0 != 0
    }
}

fn cpt_flags(cpt: u32) -> CptFlags {
    use crate::unicode_data::{RANGES_FLAGS, WHITESPACE};
    if cpt >= 0x110000 {
        return CptFlags(CptFlags::UNDEFINED);
    }
    // последний диапазон, начало которого ≤ cpt
    let i = RANGES_FLAGS.partition_point(|&(start, _)| start <= cpt) - 1;
    let mut f = RANGES_FLAGS[i].1;
    if WHITESPACE.binary_search(&cpt).is_ok() {
        f |= CptFlags::WHITESPACE;
    }
    CptFlags(f)
}

fn cpt_tolower(cpt: u32) -> u32 {
    use crate::unicode_data::LOWERCASE;
    match LOWERCASE.binary_search_by_key(&cpt, |&(c, _)| c) {
        Ok(i) => LOWERCASE[i].1,
        Err(_) => cpt,
    }
}

/// Разбиение текста на слова — порт `unicode_regex_split_custom_qwen2`
/// (и `_qwen35`) llama.cpp. Работает по кодовым точкам; возвращает срезы
/// исходной строки.
fn pre_split(text: &str, pre: PreTokenizer) -> Vec<&str> {
    // кодовые точки и их байтовые смещения (невалидный UTF-8 у &str невозможен)
    let mut cpts: Vec<u32> = Vec::with_capacity(text.len());
    let mut offs: Vec<usize> = Vec::with_capacity(text.len() + 1);
    for (o, c) in text.char_indices() {
        cpts.push(c as u32);
        offs.push(o);
    }
    offs.push(text.len());
    let n = cpts.len();
    let flags: Vec<CptFlags> = cpts.iter().map(|&c| cpt_flags(c)).collect();
    const OUT: u32 = 0xFFFF_FFFF;
    let get_cpt = |p: usize| if p < n { cpts[p] } else { OUT };
    let get_flags = |p: usize| if p < n { flags[p] } else { CptFlags(0) };
    let marks = pre == PreTokenizer::Qwen35;
    // буква в смысле серии: \p{L} (qwen2) или [\p{L}\p{M}] (qwen35)
    let letterish = |f: CptFlags| f.is_letter() || (marks && f.is_accent_mark());

    // границы слов в кодовых точках; срезы строки — в конце
    let mut bounds: Vec<(usize, usize)> = Vec::new();
    let mut prev_end = 0usize;
    let mut add = |end: usize, bounds: &mut Vec<(usize, usize)>| -> usize {
        let len = end - prev_end;
        if len > 0 {
            bounds.push((prev_end, end));
        }
        prev_end = end;
        len
    };

    let mut pos = 0usize;
    while pos < n {
        let cpt = get_cpt(pos);
        let f = get_flags(pos);

        // (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if cpt == '\'' as u32 && pos + 1 < n {
            let c1 = cpt_tolower(get_cpt(pos + 1));
            if c1 == 's' as u32 || c1 == 't' as u32 || c1 == 'm' as u32 || c1 == 'd' as u32 {
                pos += add(pos + 2, &mut bounds);
                continue;
            }
            if pos + 2 < n {
                let c2 = cpt_tolower(get_cpt(pos + 2));
                if (c1 == 'r' as u32 && c2 == 'e' as u32)
                    || (c1 == 'v' as u32 && c2 == 'e' as u32)
                    || (c1 == 'l' as u32 && c2 == 'l' as u32)
                {
                    pos += add(pos + 3, &mut bounds);
                    continue;
                }
            }
        }

        // [^\r\n\p{L}\p{N}]?\p{L}+   (qwen35: [\p{L}\p{M}]+)
        if !(cpt == '\r' as u32 || cpt == '\n' as u32 || f.is_number())
            && (letterish(f) || letterish(get_flags(pos + 1)))
        {
            pos += 1;
            while letterish(get_flags(pos)) {
                pos += 1;
            }
            add(pos, &mut bounds);
            continue;
        }

        // \p{N} (qwen2/qwen35) или \p{N}{1,3} (glm4/chatglm): серия до трёх
        // цифр — один токен-слово, дальше регулярка начинает новый матч.
        if f.is_number() {
            let mut k = 1usize;
            if pre == PreTokenizer::ChatGlm4 {
                while k < 3 && get_flags(pos + k).is_number() {
                    k += 1;
                }
            }
            pos += k;
            add(pos, &mut bounds);
            continue;
        }

        // <space>?[^\s\p{L}\p{N}]+[\r\n]*   (qwen35: и без \p{M})
        let mut f2 = if cpt == ' ' as u32 { get_flags(pos + 1) } else { f };
        let is_other = |g: CptFlags| {
            !(g.is_whitespace() || g.is_letter() || g.is_number() || (marks && g.is_accent_mark()))
        };
        if is_other(f2) && f.any() {
            if cpt == ' ' as u32 {
                pos += 1;
            }
            while is_other(f2) && f2.any() {
                pos += 1;
                f2 = get_flags(pos);
            }
            let mut c2 = get_cpt(pos);
            while c2 == '\r' as u32 || c2 == '\n' as u32 {
                pos += 1;
                c2 = get_cpt(pos);
            }
            add(pos, &mut bounds);
            continue;
        }

        let mut num_ws = 0usize;
        let mut last_rn_end = 0usize;
        while get_flags(pos + num_ws).is_whitespace() {
            let c2 = get_cpt(pos + num_ws);
            if c2 == '\r' as u32 || c2 == '\n' as u32 {
                last_rn_end = pos + num_ws + 1;
            }
            num_ws += 1;
        }

        // \s*[\r\n]+
        if last_rn_end > 0 {
            pos = last_rn_end;
            add(pos, &mut bounds);
            continue;
        }

        // \s+(?!\S)
        if num_ws > 1 && get_cpt(pos + num_ws) != OUT {
            pos += num_ws - 1;
            add(pos, &mut bounds);
            continue;
        }

        // \s+
        if num_ws > 0 {
            pos += num_ws;
            add(pos, &mut bounds);
            continue;
        }

        // нет совпадений — один символ
        pos += 1;
        add(pos, &mut bounds);
    }
    bounds
        .into_iter()
        .map(|(a, b)| &text[offs[a]..offs[b]])
        .collect()
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


#[cfg(test)]
mod tests {
    /// GLM-4/chatglm pre: цифры сериями по 1-3, у qwen2 — по одной
    /// (референс: llama.cpp `unicode_regex_split_custom` для CHATGLM4,
    /// `\p{N}{1,3}` против `\p{N}`).
    #[test]
    fn pre_split_glm4_digit_runs() {
        let glm = PreTokenizer::ChatGlm4;
        let qwen = PreTokenizer::Qwen2;
        assert_eq!(pre_split("123", glm), vec!["123"]);
        assert_eq!(pre_split("1234", glm), vec!["123", "4"]);
        assert_eq!(pre_split("1234567", glm), vec!["123", "456", "7"]);
        assert_eq!(pre_split("12", glm), vec!["12"]);
        // qwen2 режет каждую цифру отдельно
        assert_eq!(pre_split("123", qwen), vec!["1", "2", "3"]);
        // смешанный текст: буквы и пунктуация — как у qwen2
        let t = "Ошибка 404: файл не найден (12.5%)";
        let gl = pre_split(t, glm);
        let qw = pre_split(t, qwen);
        assert!(gl.contains(&"404"), "{gl:?}");
        assert!(qw.contains(&"4") && qw.contains(&"0"), "{qw:?}");
        // склейка обратно даёт исходный текст (никаких потерь)
        for parts in [&gl, &qw] {
            assert_eq!(parts.concat(), t);
        }
    }

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
