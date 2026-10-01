//! Чат-шаблоны моделей (`tokenizer.chat_template` в GGUF) — Jinja на
//! minijinja с совместимостью Python/HF (minijinja-contrib pycompat: методы
//! строк `startswith`/`split`/`strip`…) и функциями, которых ждут шаблоны:
//! `raise_exception`, `strftime_now`, `tojson` с разделителями `json.dumps`.
//!
//! Раньше здесь было собственное подмножество Jinja: ни один реальный шаблон
//! (Qwen3, Qwen3.5, Qwen3-VL — `namespace`, `macro`, `loop.*`, срезы, методы
//! строк) не компилировался, и сервис молча подставлял свой ChatML для
//! ЛЮБОЙ модели. Эталон рендера — `llama-server --jinja /apply-template`
//! (tests/chat_template_golden.rs, scripts/template_golden.py).

use std::collections::HashMap;

use minijinja::value::{Kwargs, Value};
use minijinja::{Environment, Error, ErrorKind, UndefinedBehavior};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateError {
    /// Шаблон не компилируется (синтаксис).
    Compile(String),
    /// Ошибка рендера, в т.ч. `raise_exception` из самого шаблона (например
    /// «System message must be at the beginning»).
    Render(String),
}

impl std::fmt::Display for TemplateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Compile(s) => write!(f, "чат-шаблон не компилируется: {s}"),
            Self::Render(s) => write!(f, "чат-шаблон: {s}"),
        }
    }
}

impl std::error::Error for TemplateError {}

/// Контекст рендера. Сообщения — JSON-объекты как в OpenAI-запросе:
/// `role`, `content` (строка или массив частей), `reasoning_content`,
/// `tool_calls`, `tool_call_id`, `name`.
#[derive(Debug, Clone)]
pub struct TemplateContext {
    pub messages: Vec<serde_json::Value>,
    /// Описания инструментов (OpenAI `tools`); пусто — переменная не задаётся.
    pub tools: Vec<serde_json::Value>,
    pub add_generation_prompt: bool,
    /// `enable_thinking` шаблонов Qwen3/3.5. Передаётся ВСЕГДА, как в
    /// llama.cpp (по умолчанию true): шаблон Qwen3.5 без переменной считает
    /// рассуждения выключенными и вставляет пустой `<think></think>`.
    pub enable_thinking: bool,
    /// Прочие переменные (`chat_template_kwargs` запроса).
    pub extra: HashMap<String, serde_json::Value>,
}

impl Default for TemplateContext {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            tools: Vec::new(),
            add_generation_prompt: false,
            enable_thinking: true,
            extra: HashMap::new(),
        }
    }
}

impl TemplateContext {
    /// Сообщения из пар (роль, текст).
    pub fn from_pairs(pairs: &[(&str, &str)]) -> Self {
        Self {
            messages: pairs
                .iter()
                .map(|(r, c)| serde_json::json!({ "role": r, "content": c }))
                .collect(),
            ..Default::default()
        }
    }

    pub fn add_generation_prompt(mut self, v: bool) -> Self {
        self.add_generation_prompt = v;
        self
    }
}

/// Скомпилированный шаблон.
pub struct Template {
    env: Environment<'static>,
    /// Исходный текст шаблона: по нему определяется семейство (теги
    /// рассуждений, формат вызовов инструментов) — как в llama.cpp.
    source: String,
    /// `bos_token`/`eos_token` и т.п. — из метаданных GGUF.
    pub special_tokens: HashMap<String, String>,
}

impl std::fmt::Debug for Template {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Template")
            .field("special_tokens", &self.special_tokens)
            .finish_non_exhaustive()
    }
}

const NAME: &str = "chat";

impl Template {
    pub fn compile(
        src: &str,
        special_tokens: HashMap<String, String>,
    ) -> Result<Self, TemplateError> {
        let mut env = Environment::new();
        // Python-семантика: `x.startswith(...)`, `x.split(...)[-1]`,
        // `dict.items()` и т.п. — шаблоны HF написаны под Jinja2+Python.
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        // Как Jinja2 по умолчанию: неопределённое — пустое, а не ошибка
        // (шаблоны проверяют `x is defined`, но не везде).
        env.set_undefined_behavior(UndefinedBehavior::Lenient);
        // Текст шаблона — как в Jinja2 HF (trim_blocks=True, lstrip_blocks=True).
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_keep_trailing_newline(false);
        env.add_function("raise_exception", raise_exception);
        env.add_function("strftime_now", strftime_now);
        env.add_filter("tojson", tojson);
        env.add_template_owned(NAME, src.to_string())
            .map_err(|e| TemplateError::Compile(error_chain(&e)))?;
        Ok(Self { env, source: src.to_string(), special_tokens })
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn render(&self, ctx: &TemplateContext) -> Result<String, TemplateError> {
        let tpl = self
            .env
            .get_template(NAME)
            .map_err(|e| TemplateError::Render(error_chain(&e)))?;
        let mut vars: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
        for (k, v) in &self.special_tokens {
            vars.insert(k.clone(), serde_json::Value::String(v.clone()));
        }
        vars.insert("messages".into(), serde_json::Value::Array(ctx.messages.clone()));
        if !ctx.tools.is_empty() {
            vars.insert("tools".into(), serde_json::Value::Array(ctx.tools.clone()));
        }
        vars.insert("enable_thinking".into(), serde_json::Value::Bool(ctx.enable_thinking));
        for (k, v) in &ctx.extra {
            vars.insert(k.clone(), v.clone());
        }
        if ctx.add_generation_prompt {
            vars.insert("add_generation_prompt".into(), serde_json::Value::Bool(true));
        }
        tpl.render(Value::from_serialize(&vars))
            .map_err(|e| TemplateError::Render(error_chain(&e)))
    }
}

/// Сообщение ошибки с причинами (minijinja кладёт `raise_exception` в source).
fn error_chain(e: &Error) -> String {
    let mut s = e.to_string();
    let mut cur: Option<&(dyn std::error::Error + 'static)> = std::error::Error::source(e);
    while let Some(c) = cur {
        s.push_str(": ");
        s.push_str(&c.to_string());
        cur = c.source();
    }
    s
}

fn raise_exception(msg: String) -> Result<Value, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, msg))
}

/// `strftime_now(fmt)` — дата/время по формату strftime (шаблоны Llama/Gemma
/// вписывают текущую дату в system). Поддержаны %Y %m %d %B %b %H %M %S.
fn strftime_now(fmt: String) -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let (y, mo, d) = civil_from_days(secs.div_euclid(86_400));
    let sod = secs.rem_euclid(86_400);
    const MONTHS: [&str; 12] = [
        "January", "February", "March", "April", "May", "June", "July", "August", "September",
        "October", "November", "December",
    ];
    let mut out = String::new();
    let mut it = fmt.chars();
    while let Some(c) = it.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('Y') => out.push_str(&format!("{y:04}")),
            Some('m') => out.push_str(&format!("{mo:02}")),
            Some('d') => out.push_str(&format!("{d:02}")),
            Some('B') => out.push_str(MONTHS[(mo - 1) as usize]),
            Some('b') => out.push_str(&MONTHS[(mo - 1) as usize][..3]),
            Some('H') => out.push_str(&format!("{:02}", sod / 3600)),
            Some('M') => out.push_str(&format!("{:02}", sod % 3600 / 60)),
            Some('S') => out.push_str(&format!("{:02}", sod % 60)),
            Some('%') => out.push('%'),
            Some(o) => {
                out.push('%');
                out.push(o);
            }
            None => out.push('%'),
        }
    }
    out
}

/// Дни от эпохи → (год, месяц, день) по григорианскому календарю.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y, m, d)
}

/// `tojson` как `json.dumps` Python (так рендерят HF и llama.cpp):
/// разделители `", "` и `": "`, не-ASCII без экранирования, без HTML-escape
/// (встроенный фильтр minijinja экранирует `<>&'` и пишет компактно — блок
/// tools получался другим). `indent=N` — многострочный вывод.
fn tojson(value: Value, kwargs: Kwargs) -> Result<Value, Error> {
    let indent: Option<usize> = kwargs.get("indent")?;
    let _: Option<bool> = kwargs.get("ensure_ascii")?;
    let _: Option<bool> = kwargs.get("sort_keys")?;
    kwargs.assert_all_used()?;
    let json: serde_json::Value = serde_json::to_value(&value)
        .map_err(|e| Error::new(ErrorKind::InvalidOperation, format!("tojson: {e}")))?;
    let mut out = String::new();
    write_py_json(&json, indent, 0, &mut out);
    Ok(Value::from_safe_string(out))
}

fn write_py_json(v: &serde_json::Value, indent: Option<usize>, level: usize, out: &mut String) {
    use serde_json::Value as J;
    let nl = |out: &mut String, lvl: usize| {
        if let Some(n) = indent {
            out.push('\n');
            out.push_str(&" ".repeat(n * lvl));
        }
    };
    match v {
        J::Null => out.push_str("null"),
        J::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        J::Number(n) => out.push_str(&n.to_string()),
        J::String(s) => out.push_str(&serde_json::to_string(s).unwrap_or_default()),
        J::Array(a) => {
            if a.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                    if indent.is_none() {
                        out.push(' ');
                    }
                }
                nl(out, level + 1);
                write_py_json(x, indent, level + 1, out);
            }
            nl(out, level);
            out.push(']');
        }
        J::Object(o) => {
            if o.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push('{');
            for (i, (k, x)) in o.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                    if indent.is_none() {
                        out.push(' ');
                    }
                }
                nl(out, level + 1);
                out.push_str(&serde_json::to_string(k).unwrap_or_default());
                out.push_str(": ");
                write_py_json(x, indent, level + 1, out);
            }
            nl(out, level);
            out.push('}');
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compile(src: &str) -> Template {
        Template::compile(src, HashMap::new()).unwrap()
    }

    #[test]
    fn chatml_renders_chat_and_generation_prompt() {
        let t = compile(
            "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\\n'}}{% endif %}",
        );
        let ctx = TemplateContext::from_pairs(&[("user", "привет"), ("assistant", "здравствуй")]);
        assert_eq!(
            t.render(&ctx.clone().add_generation_prompt(true)).unwrap(),
            "<|im_start|>user\nпривет<|im_end|>\n<|im_start|>assistant\nздравствуй<|im_end|>\n<|im_start|>assistant\n"
        );
        assert!(!t.render(&ctx).unwrap().ends_with("<|im_start|>assistant\n"));
    }

    /// Конструкции реальных шаблонов, которых не умел прежний парсер.
    #[test]
    fn python_methods_namespace_loop_and_slices() {
        let t = compile(
            "{%- set ns = namespace(n=0) -%}\
             {%- for m in messages[::-1] -%}{%- set ns.n = ns.n + 1 -%}{%- endfor -%}\
             {{ ns.n }}|{{ 'a</think>b'.split('</think>')[-1] }}|{{ '\\nx\\n'.strip('\\n') }}|\
             {{ 'abc'.startswith('ab') }}|{% for m in messages %}{{ loop.last }}{% endfor %}",
        );
        let out = t.render(&TemplateContext::from_pairs(&[("user", "a"), ("assistant", "b")])).unwrap();
        // булевы — как в Python/Jinja2 (`True`/`False`)
        assert_eq!(out, "2|b|x|True|FalseTrue");
    }

    #[test]
    fn raise_exception_is_a_render_error() {
        let t = compile("{{ raise_exception('System message must be at the beginning.') }}");
        let e = t.render(&TemplateContext::default()).unwrap_err();
        assert!(e.to_string().contains("System message must be at the beginning."), "{e}");
    }

    #[test]
    fn tojson_matches_python_json_dumps() {
        let t = compile("{{ x | tojson }}");
        let mut ctx = TemplateContext::default();
        ctx.extra.insert(
            "x".into(),
            serde_json::json!({"name": "get_weather", "args": {"city": "Москва", "n": [1, 2]}, "html": "<a&b>"}),
        );
        assert_eq!(
            t.render(&ctx).unwrap(),
            r#"{"name": "get_weather", "args": {"city": "Москва", "n": [1, 2]}, "html": "<a&b>"}"#
        );
    }

    #[test]
    fn special_tokens_and_kwargs_are_variables() {
        let t = Template::compile(
            "{{ bos_token }}{% if enable_thinking is defined and enable_thinking is false %}off{% endif %}",
            HashMap::from([("bos_token".to_string(), "<s>".to_string())]),
        )
        .unwrap();
        let mut ctx = TemplateContext::default();
        assert_eq!(t.render(&ctx).unwrap(), "<s>");
        ctx.enable_thinking = false;
        assert_eq!(t.render(&ctx).unwrap(), "<s>off");
    }

    #[test]
    fn syntax_error_is_compile_error() {
        assert!(matches!(
            Template::compile("{% if %}", HashMap::new()),
            Err(TemplateError::Compile(_))
        ));
    }

    #[test]
    fn strftime_now_formats_date() {
        let s = strftime_now("%Y-%m-%d %B".into());
        assert!(s.len() >= 11, "{s}");
        assert!(s[..4].parse::<u32>().unwrap() >= 2024, "{s}");
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(20_362), (2025, 10, 1));
    }
}
