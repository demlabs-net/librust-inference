//! Разделение вывода модели на рассуждения и ответ: `<think>…</think>`
//! (Qwen3/Qwen3.5, DeepSeek-R1) → `reasoning_content` + `content`, как
//! `reasoning_format: deepseek` у llama-server.
//!
//! Рассуждение начинается либо тегом `<think>` в выводе, либо уже в промпте
//! (шаблон Qwen3.5 сам ставит `<think>\n` в генерационный префикс — тогда
//! вывод начинается сразу с текста рассуждения): это задаёт `start_open`.
//! Потоковый парсер терпим к фрагментам: тег может прийти по кускам.
//! Пробелы — как у llama-server: ведущие у рассуждения (после `<think>`) и у
//! ответа (после `</think>`) отбрасываются, хвостовые у рассуждения (перед
//! `</think>`) остаются (сверено scripts/service_parity.py).

/// Событие стрима: кусок рассуждения или ответа.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReasoningEvent {
    Reasoning(String),
    Content(String),
}

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Ещё не решено: ждём `<think>` в начале вывода (пробелы до него — пропуск).
    Start,
    /// Внутри рассуждения.
    Reasoning,
    /// Ответ (после `</think>` или если рассуждения не было).
    Content,
}

#[derive(Debug)]
pub struct ReasoningParser {
    state: State,
    /// Ещё не отданный хвост (может оказаться началом тега).
    pending: String,
    /// Отрезать ведущие пробелы у следующего куска текущей секции.
    trim_lead: bool,
}

impl ReasoningParser {
    /// `start_open` — рассуждение открыто промптом (шаблон закончил
    /// генерационный префикс на `<think>`).
    pub fn new(start_open: bool) -> Self {
        Self {
            state: if start_open {
                State::Reasoning
            } else {
                State::Start
            },
            pending: String::new(),
            trim_lead: true,
        }
    }

    /// Открыт ли тег рассуждения в конце промпта (без учёта пробелов).
    pub fn prompt_opens_reasoning(prompt: &str) -> bool {
        prompt.trim_end().ends_with(OPEN)
    }

    pub fn feed(&mut self, delta: &str) -> Vec<ReasoningEvent> {
        self.pending.push_str(delta);
        let mut out = Vec::new();
        self.scan(&mut out, false);
        out
    }

    /// Конец стрима: отдать удержанный хвост. Незакрытое рассуждение
    /// остаётся рассуждением (генерацию оборвали внутри него).
    pub fn finish(&mut self) -> Vec<ReasoningEvent> {
        let mut out = Vec::new();
        self.scan(&mut out, true);
        out
    }

    fn emit(&mut self, out: &mut Vec<ReasoningEvent>, mut text: String) {
        if self.trim_lead {
            let t = text.trim_start();
            if t.is_empty() {
                return;
            }
            text = t.to_string();
            self.trim_lead = false;
        }
        match self.state {
            State::Reasoning => out.push(ReasoningEvent::Reasoning(text)),
            _ => out.push(ReasoningEvent::Content(text)),
        }
    }

    fn scan(&mut self, out: &mut Vec<ReasoningEvent>, finish: bool) {
        loop {
            match self.state {
                State::Start => {
                    let t = self.pending.trim_start();
                    if t.is_empty() {
                        if finish {
                            self.pending.clear();
                        }
                        return;
                    }
                    if let Some(rest) = t.strip_prefix(OPEN) {
                        self.pending = rest.to_string();
                        self.state = State::Reasoning;
                        self.trim_lead = true;
                        continue;
                    }
                    if !finish && OPEN.starts_with(t) {
                        return; // начало тега пришло фрагментом
                    }
                    self.state = State::Content;
                    continue;
                }
                State::Reasoning => {
                    if let Some(pos) = self.pending.find(CLOSE) {
                        let body = self.pending[..pos].to_string();
                        self.pending = self.pending[pos + CLOSE.len()..].to_string();
                        if !body.is_empty() {
                            self.emit(out, body);
                        }
                        self.state = State::Content;
                        self.trim_lead = true;
                        continue;
                    }
                    let keep = if finish {
                        0
                    } else {
                        hold_back(&self.pending, CLOSE)
                    };
                    let n = self.pending.len() - keep;
                    if n > 0 {
                        let chunk: String = self.pending.drain(..n).collect();
                        self.emit(out, chunk);
                    }
                    return;
                }
                State::Content => {
                    if !self.pending.is_empty() {
                        let chunk = std::mem::take(&mut self.pending);
                        self.emit(out, chunk);
                    }
                    return;
                }
            }
        }
    }
}

/// Сколько байт в конце `s` удержать: самый длинный суффикс, являющийся
/// префиксом тега (тег может прийти по кускам).
fn hold_back(s: &str, tag: &str) -> usize {
    (1..=s.len().min(tag.len() - 1))
        .rev()
        .find(|&len| {
            let start = s.len() - len;
            s.is_char_boundary(start) && tag.starts_with(&s[start..])
        })
        .unwrap_or(0)
}

/// Разделить готовый текст целиком (не-стрим ответ).
pub fn split(text: &str, start_open: bool) -> (Option<String>, String) {
    let mut p = ReasoningParser::new(start_open);
    let mut reasoning = String::new();
    let mut content = String::new();
    let mut had_reasoning = false;
    for ev in p.feed(text).into_iter().chain(p.finish()) {
        match ev {
            ReasoningEvent::Reasoning(t) => {
                had_reasoning = true;
                reasoning.push_str(&t);
            }
            ReasoningEvent::Content(t) => content.push_str(&t),
        }
    }
    (had_reasoning.then_some(reasoning), content)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_think_block_and_trims_joints() {
        let (r, c) = split("<think>\nСкладываю.\n</think>\n\nЧетыре.", false);
        // хвостовой перевод строки рассуждения остаётся (как llama-server)
        assert_eq!(r.as_deref(), Some("Складываю.\n"));
        assert_eq!(c, "Четыре.");
    }

    #[test]
    fn prompt_opened_reasoning() {
        // Qwen3.5: шаблон закончил промпт на «<think>\n» — вывод начинается
        // сразу с рассуждения
        let (r, c) = split("Думаю...\n</think>\n\nОтвет", true);
        assert_eq!(r.as_deref(), Some("Думаю...\n"));
        assert_eq!(c, "Ответ");
        assert!(ReasoningParser::prompt_opens_reasoning(
            "<|im_start|>assistant\n<think>\n"
        ));
        assert!(!ReasoningParser::prompt_opens_reasoning(
            "<|im_start|>assistant\n"
        ));
    }

    #[test]
    fn no_reasoning_is_plain_content() {
        let (r, c) = split("Просто ответ.", false);
        assert_eq!(r, None);
        assert_eq!(c, "Просто ответ.");
        // пустой блок (enable_thinking=false у Qwen3) — рассуждения нет
        let (r, c) = split("<think>\n\n</think>\n\nОтвет", false);
        assert_eq!(r, None);
        assert_eq!(c, "Ответ");
    }

    #[test]
    fn unterminated_reasoning_stays_reasoning() {
        let (r, c) = split("<think>\nдумаю и думаю", false);
        assert_eq!(r.as_deref(), Some("думаю и думаю"));
        assert_eq!(c, "");
    }

    /// Поток по одному символу даёт то же, что разбор целиком.
    #[test]
    fn streaming_matches_whole_text() {
        for (text, open) in [
            ("<think>\nA b\n</think>\n\nC d", false),
            ("x\n</think>\n\ny", true),
            ("plain", false),
            ("<thi", false),
            ("<think>a</thin", false),
        ] {
            let mut p = ReasoningParser::new(open);
            let (mut r, mut c) = (String::new(), String::new());
            let mut evs = Vec::new();
            for ch in text.chars() {
                evs.extend(p.feed(&ch.to_string()));
            }
            evs.extend(p.finish());
            for ev in evs {
                match ev {
                    ReasoningEvent::Reasoning(t) => r.push_str(&t),
                    ReasoningEvent::Content(t) => c.push_str(&t),
                }
            }
            let (wr, wc) = split(text, open);
            assert_eq!(
                (r.as_str(), c.as_str()),
                (wr.as_deref().unwrap_or(""), wc.as_str()),
                "{text:?}"
            );
        }
    }
}
