//! Ф1.10 Tool-call parsing from the streaming text output (Qwen/Hermes
//! style): `<tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>`.
//!
//! The parser is fragment-tolerant: arguments arrive in arbitrary chunk
//! sizes (1-2 chars per Delta is a realistic worst case), so the tool-call
//! body is accumulated until the closing tag. Text before/after a call
//! passes through as [`ToolEvent::Text`]. A call left open at stream end
//! is dropped (generation was truncated).

use serde_json::Value;

/// Один вызов инструмента. `arguments` — сырое JSON-тело (объект), как в
/// стриме (без пере-сериализации — точность аргументов важнее).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: String,
}

/// Событие стрима с учётом tool-calls.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolEvent {
    Text(String),
    ToolCall(ToolCall),
}

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";

/// Stateful парсер tool-call'ов из текстового стрима.
#[derive(Debug, Default)]
pub struct ToolCallParser {
    /// Текст, который ещё не решено, куда отнести (обычный текст или
    /// начало tool-call).
    pending: String,
    /// Аккумулированное тело текущего (незакрытого) tool-call.
    call_body: Option<String>,
    /// Уже выданные события (feed может вернуть несколько).
    out: Vec<ToolEvent>,
}

impl ToolCallParser {
    pub fn new() -> Self {
        Self::default()
    }

    /// Подать очередной фрагмент текста; вернуть накопившиеся события.
    pub fn feed(&mut self, delta: &str) -> Vec<ToolEvent> {
        self.out.clear();
        self.pending.push_str(delta);
        self.scan();
        std::mem::take(&mut self.out)
    }

    /// Конец стрима: закрыть хвост (текст после последнего tool-call).
    /// Незакрытый tool-call отбрасывается (обрезанная генерация).
    pub fn finish(&mut self) -> Vec<ToolEvent> {
        self.out.clear();
        if self.call_body.is_some() {
            // Незакрытый вызов — мусор: дропаем и тело, и pending-текст
            // внутри него (он был частью сломанного tool-call).
            self.call_body = None;
            self.pending.clear();
        } else if !self.pending.is_empty() {
            self.out
                .push(ToolEvent::Text(std::mem::take(&mut self.pending)));
        }
        std::mem::take(&mut self.out)
    }

    fn scan(&mut self) {
        loop {
            if self.call_body.is_some() {
                // Новые фрагменты идут В ТЕЛО вызова, а не в pending.
                let body = self.call_body.as_mut().unwrap();
                body.push_str(&self.pending);
                self.pending.clear();
                // Ищем закрывающий тег в накопленном теле.
                if let Some(pos) = body.find(CLOSE) {
                    let json_part = &body[..pos];
                    let call = parse_tool_call(json_part);
                    if let Some(call) = call {
                        self.out.push(ToolEvent::ToolCall(call));
                    }
                    // Хвост после закрывающего тега — снова обычный текст.
                    let rest = body[pos + CLOSE.len()..].to_string();
                    self.call_body = None;
                    self.pending = rest;
                    continue;
                }
                // Незакрытый вызов: держим всё в теле, pending пуст.
                return;
            }

            // Не в tool-call: ищем открывающий тег.
            if let Some(pos) = self.pending.find(OPEN) {
                if pos > 0 {
                    self.out
                        .push(ToolEvent::Text(self.pending[..pos].to_string()));
                }
                // Всё после открывающего тега — тело вызова.
                let rest = self.pending[pos + OPEN.len()..].to_string();
                self.call_body = Some(rest);
                self.pending.clear();
                continue;
            }
            // Открывающего тега нет: НЕ выдаём хвост как текст целиком —
            // самый длинный СУФФИКС pending, являющийся префиксом OPEN, может
            // быть началом тега, пришедшего фрагментами (например, chunk
            // "text <tool" — раньше целиком утекал в текст, т.к. проверка
            // `OPEN.starts_with(pending)` работала только когда pending
            // НАЧИНАЛСЯ с тега). Префикс перед суффиксом — обычный текст.
            let keep = (1..=self.pending.len().min(OPEN.len() - 1))
                .rev()
                .find(|&len| {
                    let start = self.pending.len() - len;
                    self.pending.is_char_boundary(start)
                        && OPEN.starts_with(&self.pending[start..])
                })
                .unwrap_or(0);
            let text_len = self.pending.len() - keep;
            if text_len > 0 {
                let text = self.pending[..text_len].to_string();
                self.pending.drain(..text_len);
                self.out.push(ToolEvent::Text(text));
            }
            return;
        }
    }
}

/// Разобрать JSON-тело tool-call: `{"name": ..., "arguments": ...}`.
fn parse_tool_call(json_part: &str) -> Option<ToolCall> {
    let trimmed = json_part.trim();
    let value: Value = serde_json::from_str(trimmed).ok()?;
    let name = value.get("name")?.as_str()?.to_string();
    let args = value.get("arguments")?;
    let arguments = if args.is_string() {
        args.as_str().unwrap_or_default().to_string()
    } else {
        args.to_string()
    };
    Some(ToolCall { name, arguments })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_call_in_one_chunk() {
        let mut p = ToolCallParser::new();
        let ev = p.feed(
            r#"Sure!<tool_call>
{"name": "get_weather", "arguments": {"city": "Moscow"}}
</tool_call>Done"#,
        );
        assert_eq!(
            ev,
            vec![
                ToolEvent::Text("Sure!".into()),
                ToolEvent::ToolCall(ToolCall {
                    name: "get_weather".into(),
                    arguments: r#"{"city":"Moscow"}"#.into(),
                }),
                ToolEvent::Text("Done".into()),
            ]
        );
    }

    #[test]
    fn fragmented_by_single_chars() {
        let mut p = ToolCallParser::new();
        let input = r#"<tool_call>
{"name": "search", "arguments": {"q": "rust"}}
</tool_call>"#;
        let mut events = Vec::new();
        for ch in input.chars() {
            let mut s = String::new();
            s.push(ch);
            events.extend(p.feed(&s));
        }
        events.extend(p.finish());
        assert_eq!(
            events,
            vec![ToolEvent::ToolCall(ToolCall {
                name: "search".into(),
                arguments: r#"{"q":"rust"}"#.into(),
            })]
        );
    }

    #[test]
    fn open_tag_split_across_chunks() {
        let mut p = ToolCallParser::new();
        let mut events = Vec::new();
        events.extend(p.feed("<tool_"));
        events.extend(p.feed("call>{\"name\": \"a\", \"arguments\": {}}\n</tool_call>"));
        events.extend(p.finish());
        assert_eq!(
            events,
            vec![ToolEvent::ToolCall(ToolCall {
                name: "a".into(),
                arguments: "{}".into(),
            })]
        );
    }

    /// Частичный тег ПОСЛЕ обычного текста: раньше pending "text <tool"
    /// целиком утекал в текст (проверка `OPEN.starts_with(pending)` работала
    /// только когда pending начинался с тега) — тег разваливался, тело вызова
    /// приходило как текст.
    #[test]
    fn open_tag_split_after_text() {
        let mut p = ToolCallParser::new();
        let mut events = Vec::new();
        events.extend(p.feed("text <tool"));
        // Текст до тега выдан, частичный тег удержан.
        assert_eq!(events, vec![ToolEvent::Text("text ".into())]);
        events.extend(p.feed("_call>{\"name\": \"a\", \"arguments\": {}}</tool_call>"));
        events.extend(p.finish());
        assert_eq!(
            events,
            vec![
                ToolEvent::Text("text ".into()),
                ToolEvent::ToolCall(ToolCall {
                    name: "a".into(),
                    arguments: "{}".into(),
                }),
            ]
        );
    }

    /// Ложное совпадение суффикса: удержанный префикс тега, за которым тег
    /// НЕ продолжился, доезжает до потребителя как текст (в исходном порядке).
    #[test]
    fn held_back_prefix_flushed_when_tag_aborts() {
        let mut p = ToolCallParser::new();
        let mut events = Vec::new();
        events.extend(p.feed("abc <to"));
        assert_eq!(events, vec![ToolEvent::Text("abc ".into())]);
        events.extend(p.feed("ol of text"));
        events.extend(p.finish());
        assert_eq!(
            events,
            vec![
                ToolEvent::Text("abc ".into()),
                ToolEvent::Text("<tool of text".into()),
            ]
        );
    }

    #[test]
    fn truncated_call_is_dropped() {
        let mut p = ToolCallParser::new();
        let mut events = Vec::new();
        events.extend(p.feed("<tool_call>\n{\"name\": \"a\", \"arg"));
        events.extend(p.finish());
        assert!(events.is_empty(), "truncated call dropped: {events:?}");
    }

    #[test]
    fn arguments_as_string_are_kept() {
        let mut p = ToolCallParser::new();
        let ev = p.feed(r#"<tool_call>{"name": "f", "arguments": "{\"x\": 1}"}</tool_call>"#);
        assert_eq!(
            ev,
            vec![ToolEvent::ToolCall(ToolCall {
                name: "f".into(),
                arguments: r#"{"x": 1}"#.into(),
            })]
        );
    }

    #[test]
    fn text_only_stream_passes_through() {
        let mut p = ToolCallParser::new();
        let mut events = Vec::new();
        events.extend(p.feed("Hello"));
        events.extend(p.feed(" world"));
        events.extend(p.finish());
        // Стрим отдаёт куски как приходят (склейка — на стороне потребителя).
        assert_eq!(
            events,
            vec![
                ToolEvent::Text("Hello".into()),
                ToolEvent::Text(" world".into())
            ]
        );
    }
}
