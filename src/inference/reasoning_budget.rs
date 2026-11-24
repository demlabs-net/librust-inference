//! Бюджет рассуждений — порт `common/reasoning-budget.cpp` llama.cpp
//! (5266f24da): автомат над потоком ТОКЕНОВ, ограничивающий длину блока
//! `<think>…</think>`.
//!
//! IDLE → COUNTING → (WAITING_UTF8) → FORCING → DONE:
//! - IDLE: ждём стартовую последовательность (`<think>`);
//! - COUNTING: считаем токены рассуждения, ждём естественный конец (`</think>`,
//!   у шаблонов с `<tool_call>` — и его: вызов инструмента закрывает мысль);
//! - WAITING_UTF8: бюджет исчерпан, но последний токен оборвал UTF-8-символ —
//!   даём модели его дописать;
//! - FORCING: выдаём принудительную последовательность (сообщение + первый
//!   конечный тег) токен за токеном, остальные логиты = −∞;
//! - DONE: прозрачно; новый стартовый тег снова взводит бюджет (несколько
//!   блоков рассуждений в одном ответе).
//!
//! Генерационный префикс промпта (то, что шаблон дописал после истории:
//! `<|im_start|>assistant\n<think>\n` у Qwen3.5) прогоняется через автомат
//! заранее — так рассуждение, открытое промптом, считается с первого токена.

use crate::tokenizer::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetState {
    Idle,
    Counting,
    Forcing,
    WaitingUtf8,
    Done,
}

/// Поиск набора последовательностей в потоке токенов (Aho–Corasick, как
/// `token_matcher` llama.cpp): `advance` возвращает индекс самой длинной
/// последовательности, закончившейся на этом токене.
#[derive(Debug, Clone)]
struct TokenMatcher {
    /// Переходы бора: (узел, токен) → узел.
    next: Vec<Vec<(u32, usize)>>,
    fail: Vec<usize>,
    /// Длина самой длинной последовательности, кончающейся в узле (с
    /// учётом суффиксных ссылок), или None.
    out: Vec<Option<usize>>,
    state: usize,
}

impl TokenMatcher {
    fn new(seqs: &[Vec<u32>]) -> Self {
        let mut uniq: Vec<Vec<u32>> = Vec::new();
        for s in seqs {
            if !s.is_empty() && !uniq.contains(s) {
                uniq.push(s.clone());
            }
        }
        let mut next: Vec<Vec<(u32, usize)>> = vec![Vec::new()];
        let mut depth = vec![0usize];
        let mut term: Vec<Option<usize>> = vec![None];
        for s in &uniq {
            let mut node = 0usize;
            for &t in s {
                node = match next[node].iter().find(|(k, _)| *k == t) {
                    Some(&(_, n)) => n,
                    None => {
                        next.push(Vec::new());
                        depth.push(depth[node] + 1);
                        term.push(None);
                        let n = next.len() - 1;
                        next[node].push((t, n));
                        n
                    }
                };
            }
            term[node] = Some(s.len());
        }
        // Суффиксные ссылки обходом в ширину; out — самая длинная
        // последовательность из term узла и out его суффиксной ссылки.
        let n = next.len();
        let mut fail = vec![0usize; n];
        let mut out = term.clone();
        let mut queue = std::collections::VecDeque::new();
        for &(_, c) in &next[0] {
            queue.push_back(c);
        }
        while let Some(u) = queue.pop_front() {
            let children = next[u].clone();
            for (t, c) in children {
                let mut f = fail[u];
                fail[c] = loop {
                    if let Some(&(_, g)) = next[f].iter().find(|(k, _)| *k == t)
                        && g != c
                    {
                        break g;
                    }
                    if f == 0 {
                        break 0;
                    }
                    f = fail[f];
                };
                if out[c].is_none() {
                    out[c] = out[fail[c]];
                } else if let (Some(o), Some(own)) = (out[fail[c]], out[c]) {
                    // обе кончаются здесь: побеждает длиннейшая
                    out[c] = Some(o.max(own));
                }
                queue.push_back(c);
            }
        }
        Self {
            next,
            fail,
            out,
            state: 0,
        }
    }

    /// Учесть токен; Some(длина) — на нём закончилась последовательность.
    fn advance(&mut self, tok: u32) -> Option<usize> {
        let mut s = self.state;
        self.state = loop {
            if let Some(&(_, n)) = self.next[s].iter().find(|(k, _)| *k == tok) {
                break n;
            }
            if s == 0 {
                break 0;
            }
            s = self.fail[s];
        };
        let m = self.out[self.state];
        if m.is_some() {
            self.state = 0;
        }
        m
    }

    fn reset(&mut self) {
        self.state = 0;
    }
}

/// Кусок токена завершает UTF-8-символ (`common_utf8_is_complete`).
pub fn utf8_is_complete(s: &[u8]) -> bool {
    if s.is_empty() {
        return true;
    }
    for i in 1..=s.len().min(4) {
        let c = s[s.len() - i];
        if c & 0xC0 != 0x80 {
            let expected = if c >= 0xF0 {
                4
            } else if c >= 0xE0 {
                3
            } else if c >= 0xC0 {
                2
            } else {
                1
            };
            return i >= expected;
        }
    }
    false
}

/// Автомат бюджета рассуждений (одна генерация).
#[derive(Debug, Clone)]
pub struct ReasoningBudget {
    start: TokenMatcher,
    end: TokenMatcher,
    forced: Vec<u32>,
    budget: usize,
    remaining: usize,
    state: BudgetState,
    force_pos: usize,
}

impl ReasoningBudget {
    /// `start`/`end` — стартовые и конечные последовательности токенов;
    /// `forced` — что вставить по исчерпании (сообщение + первый конечный
    /// тег); `initial` — начальное состояние (COUNTING с бюджетом 0 сразу
    /// становится FORCING).
    pub fn new(
        start: &[Vec<u32>],
        end: &[Vec<u32>],
        forced: Vec<u32>,
        budget: usize,
        initial: BudgetState,
    ) -> Self {
        let state = if initial == BudgetState::Counting && budget == 0 {
            BudgetState::Forcing
        } else {
            initial
        };
        Self {
            start: TokenMatcher::new(start),
            end: TokenMatcher::new(end),
            forced,
            budget,
            remaining: budget,
            state,
            force_pos: 0,
        }
    }

    /// Автомат для модели: теги токенизируются её словарём, принудительная
    /// последовательность = `message` + первый конечный тег (как
    /// `reasoning_budget_forced` llama-server). Пустой стартовый или
    /// конечный тег — бюджет неприменим (None).
    pub fn for_tokenizer(
        tok: &Tokenizer,
        start_tag: &str,
        end_tags: &[&str],
        message: &str,
        budget: usize,
    ) -> Option<Self> {
        let start = tok.encode(start_tag);
        let end: Vec<Vec<u32>> = end_tags
            .iter()
            .filter(|t| !t.is_empty())
            .map(|t| tok.encode(t))
            .collect();
        if start.is_empty() || end.is_empty() {
            return None;
        }
        let mut forced = if message.is_empty() {
            Vec::new()
        } else {
            tok.encode(message)
        };
        forced.extend_from_slice(&end[0]);
        Some(Self::new(&[start], &end, forced, budget, BudgetState::Idle))
    }

    /// Прогнать токены генерационного префикса промпта (рассуждение,
    /// открытое шаблоном, начинает счёт ещё до генерации).
    pub fn prefill(&mut self, tok: &Tokenizer, ids: &[u32]) {
        for &t in ids {
            self.accept_piece(t, &tok.piece_bytes(t));
        }
    }

    pub fn state(&self) -> BudgetState {
        self.state
    }

    /// Токен, который обязан быть следующим (FORCING), иначе None.
    pub fn forced_token(&self) -> Option<u32> {
        (self.state == BudgetState::Forcing)
            .then(|| self.forced.get(self.force_pos).copied())
            .flatten()
    }

    /// Ручной переход в FORCING (только из COUNTING) — `common_reasoning_budget_force`.
    pub fn force(&mut self) -> bool {
        if self.state != BudgetState::Counting {
            return false;
        }
        self.state = BudgetState::Forcing;
        self.force_pos = 0;
        self.end.reset();
        true
    }

    /// Учесть токен без проверки UTF-8-границы (как llama.cpp без словаря).
    pub fn accept(&mut self, tok: u32) {
        self.accept_inner(tok, true);
    }

    /// Учесть выбранный (или вставленный) токен; `piece` — его сырые байты:
    /// бюджет, кончившийся посреди многобайтного символа, ждёт его конца.
    pub fn accept_piece(&mut self, tok: u32, piece: &[u8]) {
        self.accept_inner(tok, utf8_is_complete(piece));
    }

    fn accept_inner(&mut self, tok: u32, complete: bool) {
        match self.state {
            BudgetState::Idle => {
                if self.start.advance(tok).is_some() {
                    self.arm();
                }
            }
            BudgetState::Counting | BudgetState::WaitingUtf8 => {
                if self.end.advance(tok).is_some() {
                    self.state = BudgetState::Done;
                    return;
                }
                if self.state == BudgetState::WaitingUtf8 {
                    if complete {
                        self.state = BudgetState::Forcing;
                        self.force_pos = 0;
                        self.end.reset();
                    }
                } else {
                    self.remaining = self.remaining.saturating_sub(1);
                    if self.remaining == 0 {
                        self.state = if complete {
                            self.force_pos = 0;
                            BudgetState::Forcing
                        } else {
                            BudgetState::WaitingUtf8
                        };
                        self.end.reset();
                    }
                }
            }
            BudgetState::Forcing => {
                // Продвигает вставку только сам принудительный токен. У
                // llama.cpp сюда попадают и токены генпрефикса: при бюджете 0
                // «\n» после `<think>` Qwen3.5 засчитывался как вставленный
                // `</think>`, и рассуждение шло без ограничения.
                if self.forced.get(self.force_pos) != Some(&tok) {
                    return;
                }
                self.end.advance(tok);
                self.force_pos += 1;
                if self.force_pos >= self.forced.len() {
                    self.state = BudgetState::Done;
                }
            }
            BudgetState::Done => {
                if self.start.advance(tok).is_some() {
                    self.end.reset();
                    self.arm();
                }
            }
        }
    }

    fn arm(&mut self) {
        self.state = BudgetState::Counting;
        self.remaining = self.budget;
        if self.remaining == 0 {
            self.state = BudgetState::Forcing;
            self.force_pos = 0;
        }
    }

    /// Сколько первых черновиков спек-окна можно проверять (якорь окна уже
    /// учтён): перед каждым из `draft[..k]` автомат ещё не требует вставки,
    /// поэтому и пересэмпл при отказе внутри окна — законный выбор модели.
    /// Без проверки UTF-8 окно режется не позже настоящего автомата.
    /// Состояние не меняется.
    pub fn draft_room(&self, draft: &[u32]) -> usize {
        let mut sim = self.clone();
        let mut k = 0;
        for &t in draft {
            if sim.state == BudgetState::Forcing {
                break;
            }
            sim.accept(t);
            k += 1;
        }
        k
    }
}

/// Генерационный префикс промпта: часть, которую шаблон дописал при
/// `add_generation_prompt` (`common_chat_template_generation_prompt`):
/// общий префикс двух рендеров отрезается.
pub fn generation_prefix(with_prompt: &str, without_prompt: &str) -> String {
    let n = with_prompt
        .bytes()
        .zip(without_prompt.bytes())
        .take_while(|(a, b)| a == b)
        .count();
    // по границе символа: общий префикс байтов мог разрезать символ
    let mut n = n;
    while !with_prompt.is_char_boundary(n) {
        n -= 1;
    }
    with_prompt[n..].to_string()
}

/// Стартовый и конечные теги рассуждения для шаблона модели — как
/// `thinking_start_tag`/`thinking_end_tags` llama.cpp: у шаблонов с
/// XML-вызовами Qwen3-Coder/Qwen3.5 (`<tool_call>` + `<function=` +
/// `<parameter=`) мысль закрывает и `<tool_call>`; без `<think>` в
/// шаблоне бюджет неприменим.
pub fn thinking_tags(template_src: &str) -> Option<(&'static str, Vec<&'static str>)> {
    if !template_src.contains("<think>") {
        return None;
    }
    let qwen3_coder = template_src.contains("<tool_call>")
        && template_src.contains("<function=")
        && template_src.contains("<parameter=");
    let ends = if qwen3_coder {
        vec!["</think>", "<tool_call>"]
    } else {
        vec!["</think>"]
    };
    Some(("<think>", ends))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Порт `test_reasoning_budget` (tests/test-reasoning-budget.cpp):
    /// на каждом шаге — принуждает ли автомат токен; возвращает (первый,
    /// последний) шаг принуждения. Пока автомат принуждает, выдаётся его
    /// токен (так делает петля генерации; у llama.cpp сэмплер с маской −∞
    /// тоже не может выбрать другой), иначе — токен последовательности.
    fn run(
        seq: &[u32],
        start: &[Vec<u32>],
        end: &[Vec<u32>],
        forced: &[u32],
        budget: usize,
        initial: BudgetState,
    ) -> Option<(usize, usize)> {
        let mut b = ReasoningBudget::new(start, end, forced.to_vec(), budget, initial);
        let mut span: Option<(usize, usize)> = None;
        for (i, &t) in seq.iter().enumerate() {
            let forced = b.forced_token();
            let forcing = forced.is_some();
            b.accept(forced.unwrap_or(t));
            if forcing {
                span = Some((span.map(|s| s.0).unwrap_or(i), i));
            } else if span.is_some() {
                break;
            }
        }
        span
    }

    #[test]
    fn llama_cpp_budget_cases() {
        let s = vec![vec![100u32]];
        let e = vec![vec![101u32]];
        // 1: естественный конец раньше бюджета — принуждения нет
        assert_eq!(
            run(
                &[100, 50, 51, 101, 52],
                &s,
                &e,
                &[102],
                5,
                BudgetState::Idle
            ),
            None
        );
        // 2: бюджет 2 исчерпан на i=2 — принуждение на i=3..4
        assert_eq!(
            run(
                &[100, 50, 51, 52, 53],
                &s,
                &e,
                &[102, 101],
                2,
                BudgetState::Idle
            ),
            Some((3, 4))
        );
        // 3: COUNTING с бюджетом 0 — сразу FORCING
        assert_eq!(
            run(
                &[100, 50, 51, 52],
                &s,
                &e,
                &[102, 101],
                0,
                BudgetState::Counting
            ),
            Some((0, 1))
        );
        // 4: нет тегов — прозрачно
        assert_eq!(
            run(
                &[50, 51, 52, 53],
                &[vec![]],
                &[vec![]],
                &[102],
                2,
                BudgetState::Idle
            ),
            None
        );
        // 5: открыто сразу (COUNTING), бюджет 2
        assert_eq!(
            run(
                &[50, 51, 52, 53],
                &s,
                &e,
                &[102, 101],
                2,
                BudgetState::Counting
            ),
            Some((2, 3))
        );
        // 6: второй блок рассуждений снова получает бюджет
        assert_eq!(
            run(
                &[100, 50, 101, 100, 60, 61, 62, 63],
                &s,
                &e,
                &[102, 101],
                2,
                BudgetState::Idle
            ),
            Some((6, 7))
        );
        // 7: несколько стартовых последовательностей (многотокенная)
        assert_eq!(
            run(
                &[110, 111, 50, 51, 52, 53],
                &[vec![100], vec![110, 111]],
                &e,
                &[102, 101],
                2,
                BudgetState::Idle
            ),
            Some((4, 5))
        );
        // 8: естественный конец второй (многотокенной) конечной последовательностью
        assert_eq!(
            run(
                &[100, 50, 103, 104, 52],
                &s,
                &[vec![101], vec![103, 104]],
                &[102, 101],
                5,
                BudgetState::Idle
            ),
            None
        );
    }

    /// Бюджет 0 и рассуждение, открытое промптом (`<think>\n` у Qwen3.5):
    /// токен «\n» генпрефикса не засчитывается как вставка — первым
    /// сгенерированным токеном будет закрытие.
    #[test]
    fn zero_budget_with_prompt_opened_reasoning() {
        let mut b =
            ReasoningBudget::new(&[vec![100]], &[vec![101]], vec![101], 0, BudgetState::Idle);
        b.accept(100); // <think> генпрефикса
        b.accept(7); // «\n» генпрефикса
        assert_eq!(b.forced_token(), Some(101));
        b.accept(101);
        assert_eq!(b.state(), BudgetState::Done);
    }

    #[test]
    fn manual_force_only_from_counting() {
        let s = vec![vec![100u32]];
        let e = vec![vec![101u32]];
        let mut b = ReasoningBudget::new(&s, &e, vec![102, 101], 5, BudgetState::Idle);
        assert!(!b.force(), "из IDLE нельзя");
        b.accept(100);
        b.accept(50);
        assert!(b.force());
        assert_eq!(b.forced_token(), Some(102));
        b.accept(102);
        assert!(!b.force(), "из FORCING нельзя и позиция не откатывается");
        assert_eq!(b.forced_token(), Some(101));
        b.accept(101);
        assert_eq!(b.state(), BudgetState::Done);
        assert!(!b.force(), "из DONE нельзя");
    }

    #[test]
    fn overlapping_end_sequences() {
        // {104} и {103,104} кончаются на одном токене — естественный конец
        let mut b = ReasoningBudget::new(
            &[vec![100]],
            &[vec![104], vec![103, 104]],
            vec![102, 104],
            5,
            BudgetState::Idle,
        );
        for t in [100, 103, 104] {
            b.accept(t);
        }
        assert_eq!(b.state(), BudgetState::Done);
        let mut m = TokenMatcher::new(&[vec![104], vec![103, 104]]);
        assert_eq!(m.advance(103), None);
        assert_eq!(m.advance(104), Some(2), "побеждает длиннейшая");
    }

    #[test]
    fn utf8_boundaries() {
        for s in [
            &b"hello"[..],
            b"",
            b"\xC2\xA0",
            b"\xE2\x80\x9C",
            b"\xF0\x9F\x98\x80",
            b"abc\xC3\xA9",
        ] {
            assert!(utf8_is_complete(s), "{s:?}");
        }
        for s in [
            &b"\xC2"[..],
            b"\xE2\x80",
            b"\xE2",
            b"\xF0\x9F\x98",
            b"\xF0\x9F",
            b"\xF0",
            b"\x80",
            b"hello\xC3",
        ] {
            assert!(!utf8_is_complete(s), "{s:?}");
        }
    }

    /// Бюджет кончился посреди многобайтного символа — автомат ждёт, пока
    /// кусок очередного токена не закончится целым символом. Как в
    /// llama.cpp, проверяется кусок ОДНОГО токена: кусок из одних байтов-
    /// продолжений считается неполным, так что вставка идёт после
    /// следующего токена с целым символом на конце.
    #[test]
    fn waits_for_utf8_completion() {
        let mut b =
            ReasoningBudget::new(&[vec![100]], &[vec![101]], vec![101], 1, BudgetState::Idle);
        b.accept(100);
        b.accept_piece(60, &[0xF0, 0x9F]); // начало эмодзи
        assert_eq!(b.state(), BudgetState::WaitingUtf8);
        assert_eq!(b.forced_token(), None);
        b.accept_piece(61, &[0x98, 0x80]); // его конец — одни продолжения
        assert_eq!(b.state(), BudgetState::WaitingUtf8);
        b.accept_piece(62, b" x");
        assert_eq!(b.forced_token(), Some(101));
    }

    #[test]
    fn draft_room_stops_before_forcing() {
        let mut b = ReasoningBudget::new(
            &[vec![100]],
            &[vec![101]],
            vec![102, 101],
            3,
            BudgetState::Idle,
        );
        b.accept(100);
        b.accept(50); // якорь окна съел 1 из 3
        // черновики — ещё 2, третий уже под вставкой
        assert_eq!(b.draft_room(&[51, 52, 53, 54]), 2);
        // естественный конец в черновиках — окно не режется
        assert_eq!(b.draft_room(&[101, 52, 53]), 3);
        assert_eq!(
            b.state(),
            BudgetState::Counting,
            "симуляция не трогает автомат"
        );
        b.accept(51);
        b.accept(52);
        assert_eq!(b.draft_room(&[53]), 0, "вставка обязательна — окна нет");
    }

    #[test]
    fn generation_prefix_is_template_suffix() {
        assert_eq!(
            generation_prefix(
                "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n",
                "<|im_start|>user\nhi<|im_end|>\n"
            ),
            "<|im_start|>assistant\n<think>\n"
        );
        // общий префикс режет символ — отступаем к границе
        assert_eq!(generation_prefix("aяb", "aюc"), "яb");
    }

    #[test]
    fn tags_follow_template_family() {
        assert_eq!(thinking_tags("{{ x }}"), None);
        assert_eq!(
            thinking_tags("<think>"),
            Some(("<think>", vec!["</think>"]))
        );
        assert_eq!(
            thinking_tags("<think> <tool_call> <function= <parameter="),
            Some(("<think>", vec!["</think>", "<tool_call>"]))
        );
    }
}
