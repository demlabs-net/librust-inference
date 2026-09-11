//! Чат-шаблоны: minja-подмножество Jinja для реальных моделей (Ф1.6).
//!
//! Поддерживается то, что используют Qwen3/Gemma/Saiga: text, `{{ expr }}`,
//! `{% if/elif/else %}`, `{% for %}`, фильтр `trim`, вызовы `message['role']`,
//! `loop.index0`, конкатенация строк, `eos_token`/`bos_token`. Прочее (макросы,
//! наследование, сложные фильтры) — честная ошибка Unsupported, не молчаливая
//! порча вывода. Golden-сверка с реальными шаблонами моделей — в Ф1.7.

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateError {
    UnclosedTag(&'static str),
    Unsupported(String),
    UnknownFilter(String),
    NotString(String),
}

impl std::fmt::Display for TemplateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnclosedTag(t) => write!(f, "unclosed template tag `{t}`"),
            Self::Unsupported(s) => write!(f, "unsupported template construct: {s}"),
            Self::UnknownFilter(n) => write!(f, "unknown filter `{n}`"),
            Self::NotString(s) => write!(f, "expected string, got `{s}`"),
        }
    }
}

impl std::error::Error for TemplateError {}

type VResult = Result<String, TemplateError>;

#[derive(Debug, Clone)]
enum Node {
    Text(String),
    Expr(String),
    If {
        branches: Vec<(String, Vec<Node>)>,
        otherwise: Vec<Node>,
    },
    For {
        var: String,
        iter: String,
        body: Vec<Node>,
    },
}

#[derive(Debug, Clone, Default)]
pub struct TemplateContext {
    /// `messages`: [{role, content}].
    pub messages: Vec<HashMap<String, String>>,
    pub extra: HashMap<String, String>,
}

impl TemplateContext {
    pub fn add_generation_prompt(mut self, v: bool) -> Self {
        self.extra.insert(
            "add_generation_prompt".into(),
            if v { "true" } else { "false" }.into(),
        );
        self
    }
}

/// Скомпилированный шаблон.
#[derive(Debug, Clone)]
pub struct Template {
    nodes: Vec<Node>,
    /// Заполнители спец-токенов (`eos_token` и т.п.) — из GGUF-метадаты/манифеста.
    pub special_tokens: HashMap<String, String>,
}

impl Template {
    pub fn compile(
        src: &str,
        special_tokens: HashMap<String, String>,
    ) -> Result<Self, TemplateError> {
        let toks = tokenize(src)?;
        let mut pos = 0usize;
        let nodes = build(&toks, &mut pos)?;
        if pos != toks.len() {
            return Err(TemplateError::Unsupported(format!(
                "stray block terminator: {:?}",
                toks.get(pos)
            )));
        }
        Ok(Self {
            nodes,
            special_tokens,
        })
    }

    pub fn render(&self, ctx: &TemplateContext) -> VResult {
        render_nodes(&self.nodes, ctx, &self.special_tokens, 0)
    }
}

// ── Парсер ──────────────────────────────────────────────────────────────────
// Токенизатор {{ }}/{% %}/{# #} → рекурсивный билдер (вложенные if/for
// поддерживаются — нужны шаблонам Gemma/Saiga).

#[derive(Debug)]
enum Tok {
    Text(String),
    Expr(String),
    Tag(String),
}

#[derive(Clone, Copy, PartialEq)]
enum MarkerKind {
    Expr,
    Tag,
    Comment,
}

/// Найти ближайший закрывающий маркер, ПРОПУСКАЯ строковые литералы
/// ('...'/"...", с backslash-экранированием) — `{{ '}}' -}}` закрывается
/// `-}}`, а не внутренним `}}` (план 2.8; раньше контент обрезался до `'`).
fn find_close(after: &str, close_pats: &[&str]) -> Option<(usize, bool)> {
    let mut quote: Option<char> = None;
    let mut iter = after.char_indices().peekable();
    while let Some((i, ch)) = iter.next() {
        match quote {
            Some(q) => {
                if ch == '\\' {
                    // escaped char inside the literal — skip it
                    let _ = iter.next();
                } else if ch == q {
                    quote = None;
                }
            }
            None => {
                if ch == '\'' || ch == '"' {
                    quote = Some(ch);
                    continue;
                }
                for cp in close_pats {
                    if after[i..].starts_with(cp) {
                        return Some((i, cp.starts_with('-')));
                    }
                }
            }
        }
    }
    None
}

fn tokenize(src: &str) -> Result<Vec<Tok>, TemplateError> {
    let mut toks = Vec::new();
    let mut text = String::new();
    let mut trim_next = false;
    let mut rest = src;
    loop {
        // ближайший открывающий маркер ('{{-'/'{{'/'{%-'/'{%'/'{#-'/'{#';
        // '-' = trim)
        let mut best: Option<(usize, MarkerKind)> = None;
        for (pat, kind) in [
            ("{{-", MarkerKind::Expr),
            ("{{", MarkerKind::Expr),
            ("{%-", MarkerKind::Tag),
            ("{%", MarkerKind::Tag),
            // N (2.8): {# #} комментарии были задекларированы в доке, но
            // не реализованы — контент комментария утекал в вывод как текст.
            ("{#-", MarkerKind::Comment),
            ("{#", MarkerKind::Comment),
        ] {
            if let Some(p) = rest.find(pat)
                && best.is_none_or(|(bi, _)| p < bi)
            {
                best = Some((p, kind));
            }
        }
        let Some((idx, kind)) = best else {
            text.push_str(rest);
            break;
        };
        let trim_before = rest[idx..].starts_with("{{-")
            || rest[idx..].starts_with("{%-")
            || rest[idx..].starts_with("{#-");
        if trim_next {
            text = text.trim_start().to_string();
            // trim_next перезаписывается ниже (trim_after) — сброс не нужен.
        }
        text.push_str(&rest[..idx]);
        if trim_before {
            text = text.trim_end().to_string();
        }
        // закрытие: '-}}'/'}}' для expr, '-%}'/'%}' для tag, '-#}'/'#}'
        // для комментария
        let after = &rest[idx + 2 + usize::from(trim_before)..];
        let close_pats: &[&str] = match kind {
            MarkerKind::Expr => &["-}}", "}}"],
            MarkerKind::Tag => &["-%}", "%}"],
            MarkerKind::Comment => &["-#}", "#}"],
        };
        // N (iter-9 + 2.8): EARLIEST match, пропуская кавычки — старый
        // list-order `find` закрывал `{{ '}}' -}}` на внутреннем `}}` и
        // резал контент до `'`.
        let Some((p, trim_after)) = find_close(after, close_pats) else {
            return Err(TemplateError::UnclosedTag(match kind {
                MarkerKind::Expr => "{{",
                MarkerKind::Tag => "{%",
                MarkerKind::Comment => "{#",
            }));
        };
        if !text.is_empty() {
            toks.push(Tok::Text(std::mem::take(&mut text)));
        }
        if kind != MarkerKind::Comment {
            let content = after[..p].trim();
            toks.push(match kind {
                MarkerKind::Expr => Tok::Expr(content.to_string()),
                MarkerKind::Tag => Tok::Tag(content.to_string()),
                MarkerKind::Comment => unreachable!(),
            });
        }
        trim_next = trim_after;
        rest = &after[p + 2 + usize::from(trim_after)..];
    }
    if trim_next {
        text = text.trim_start().to_string();
    }
    if !text.is_empty() {
        toks.push(Tok::Text(text));
    }
    Ok(toks)
}

fn build(toks: &[Tok], pos: &mut usize) -> Result<Vec<Node>, TemplateError> {
    let mut nodes = Vec::new();
    while *pos < toks.len() {
        match &toks[*pos] {
            Tok::Text(t) => {
                if !t.is_empty() {
                    nodes.push(Node::Text(t.clone()));
                }
                *pos += 1;
            }
            Tok::Expr(e) => {
                nodes.push(Node::Expr(e.clone()));
                *pos += 1;
            }
            Tok::Tag(t) => {
                if let Some(cond) = t.strip_prefix("if ") {
                    *pos += 1;
                    let (branches, otherwise) = parse_if(toks, pos, cond.to_string())?;
                    nodes.push(Node::If {
                        branches,
                        otherwise,
                    });
                } else if let Some(expr) = t.strip_prefix("for ") {
                    let (var, iter) = expr
                        .split_once(" in ")
                        .ok_or_else(|| TemplateError::Unsupported(format!("for: {expr}")))?;
                    *pos += 1;
                    let body = build(toks, pos)?;
                    match toks.get(*pos) {
                        Some(Tok::Tag(tag)) if tag == "endfor" => {
                            *pos += 1;
                            nodes.push(Node::For {
                                var: var.trim().into(),
                                iter: iter.trim().into(),
                                body,
                            });
                        }
                        other => {
                            return Err(TemplateError::Unsupported(format!(
                                "expected endfor, got {other:?}"
                            )));
                        }
                    }
                } else {
                    // endif/else/elif — завершают блок, отдаём наружу
                    return Ok(nodes);
                }
            }
        }
    }
    Ok(nodes)
}

fn parse_if(
    toks: &[Tok],
    pos: &mut usize,
    first_cond: String,
) -> Result<(Vec<(String, Vec<Node>)>, Vec<Node>), TemplateError> {
    let mut branches = Vec::new();
    let mut cond = first_cond;
    loop {
        let body = build(toks, pos)?;
        match toks.get(*pos) {
            Some(Tok::Tag(tag)) if tag == "elif " => unreachable!(),
            Some(Tok::Tag(tag)) if tag.starts_with("elif ") => {
                branches.push((cond, body));
                cond = tag[5..].to_string();
                *pos += 1;
            }
            Some(Tok::Tag(tag)) if tag == "else" => {
                branches.push((cond, body));
                *pos += 1;
                let obody = build(toks, pos)?;
                match toks.get(*pos) {
                    Some(Tok::Tag(t)) if t == "endif" => {
                        *pos += 1;
                        return Ok((branches, obody));
                    }
                    other => {
                        return Err(TemplateError::Unsupported(format!(
                            "expected endif, got {other:?}"
                        )));
                    }
                }
            }
            Some(Tok::Tag(tag)) if tag == "endif" => {
                branches.push((cond, body));
                *pos += 1;
                return Ok((branches, Vec::new()));
            }
            other => {
                return Err(TemplateError::Unsupported(format!(
                    "expected endif, got {other:?}"
                )));
            }
        }
    }
}

// ── Интерпретатор ───────────────────────────────────────────────────────────

fn render_nodes(
    nodes: &[Node],
    ctx: &TemplateContext,
    st: &HashMap<String, String>,
    depth: usize,
) -> VResult {
    if depth > 64 {
        return Err(TemplateError::Unsupported(
            "template nesting too deep".into(),
        ));
    }
    let mut out = String::new();
    for n in nodes {
        match n {
            Node::Text(t) => out.push_str(t),
            Node::Expr(e) => out.push_str(&eval_expr(e, ctx, st)?),
            Node::If {
                branches,
                otherwise,
            } => {
                let mut matched = false;
                for (cond, body) in branches {
                    if eval_truthy(cond, ctx, st)? {
                        out.push_str(&render_nodes(body, ctx, st, depth + 1)?);
                        matched = true;
                        break;
                    }
                }
                if !matched {
                    out.push_str(&render_nodes(otherwise, ctx, st, depth + 1)?);
                }
            }
            Node::For { var, iter, body } => {
                let items = eval_list(iter, ctx, st)?;
                for (i, item) in items.iter().enumerate() {
                    let mut sub = ctx.clone();
                    sub.extra.insert(var.clone(), item.clone());
                    sub.extra.insert("loop.index0".into(), i.to_string());
                    out.push_str(&render_nodes(body, &sub, st, depth + 1)?);
                }
            }
        }
    }
    Ok(out)
}

fn eval_list(
    expr: &str,
    ctx: &TemplateContext,
    _st: &HashMap<String, String>,
) -> Result<Vec<String>, TemplateError> {
    match expr {
        "messages" => Ok(ctx.messages.iter().map(render_message).collect()),
        _ => Err(TemplateError::Unsupported(format!("iterable `{expr}`"))),
    }
}

fn render_message(m: &HashMap<String, String>) -> String {
    // сообщение сериализуется как `{'role': '...', 'content': '...'}`;
    // кавычка и backslash экранируются, иначе dict_get обрезает значение
    // на первой же апострофе («don't stop» → «don») — раньше содержимое
    // вставлялось raw.
    let esc = |s: &str| s.replace('\\', "\\\\").replace('\'', "\\'");
    let role = esc(m.get("role").map(String::as_str).unwrap_or(""));
    let content = esc(m.get("content").map(String::as_str).unwrap_or(""));
    format!("{{'role': '{role}', 'content': '{content}'}}")
}

/// Значение выражения как строка. Поддержка: конкатенация '+', литералы в
/// кавычках, индексация ['key'], спец-токены, add_generation_prompt,
/// loop.index0, фильтр |trim, сравнение == / != в условиях.
fn eval_expr(expr: &str, ctx: &TemplateContext, st: &HashMap<String, String>) -> VResult {
    let e = expr.trim();
    // фильтр
    let (base, filter) = match e.rsplit_once(" | ") {
        Some((b, f)) => (b, Some(f.trim())),
        None => (e, None),
    };
    let mut v = eval_value(base, ctx, st)?;
    if let Some(f) = filter {
        v = match f {
            "trim" => v.trim().to_string(),
            other => return Err(TemplateError::UnknownFilter(other.into())),
        };
    }
    Ok(v)
}

fn eval_value(expr: &str, ctx: &TemplateContext, st: &HashMap<String, String>) -> VResult {
    let e = expr.trim();
    // конкатенация '+'
    if let Some((l, r)) = split_plus(e) {
        return Ok(eval_value(l, ctx, st)? + &eval_value(r, ctx, st)?);
    }
    if let Some(inner) = e.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')) {
        return Ok(unescape_literal(inner));
    }
    if let Some(inner) = e.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
        return Ok(unescape_literal(inner));
    }
    // индексация: expr['key']
    if let Some(rest) = e.strip_suffix(']')
        && let Some((obj, key)) = rest.split_once("['")
    {
        let key = key
            .strip_suffix('\'')
            .ok_or_else(|| TemplateError::Unsupported(e.into()))?;
        return eval_index(obj.trim(), key, ctx, st);
    }
    // спец-токены и переменные контекста
    if let Some(v) = st.get(e) {
        return Ok(v.clone());
    }
    if let Some(v) = ctx.extra.get(e) {
        return Ok(v.clone());
    }
    match e {
        "true" => Ok("true".into()),
        "false" => Ok("false".into()),
        _ => Err(TemplateError::Unsupported(format!("expression `{e}`"))),
    }
}

fn eval_index(
    obj: &str,
    key: &str,
    ctx: &TemplateContext,
    _st: &HashMap<String, String>,
) -> VResult {
    match obj {
        "message" => {
            // переменная цикла (сериализованный словарь) имеет приоритет
            if let Some(v) = ctx.extra.get("message") {
                return dict_get(v, key);
            }
            let m = ctx
                .messages
                .last()
                .ok_or_else(|| TemplateError::NotString("no message".into()))?;
            m.get(key)
                .cloned()
                .ok_or_else(|| TemplateError::NotString(format!("message[{key}]")))
        }
        "messages" => Err(TemplateError::Unsupported("messages[i] indexing".into())),
        _ => {
            // цикл: переменная цикла — словарь-строка вида {'role': '...', 'content': '...'}
            let v = ctx
                .extra
                .get(obj)
                .ok_or_else(|| TemplateError::Unsupported(format!("object `{obj}`")))?;
            dict_get(v, key)
        }
    }
}

/// Достать значение из сериализованного словаря `{'role': '...', 'content':
/// '...'}`. Значение читается с учётом backslash-эскейпов (render_message
/// экранирует `\` и `'`): `\'` → `'`, `\\` → `\`; неэкранированная `'`
/// завершает значение.
fn dict_get(v: &str, key: &str) -> VResult {
    let pat = format!("'{key}': '");
    let start = v
        .find(&pat)
        .ok_or_else(|| TemplateError::NotString(format!("dict[{key}]")))?;
    let tail = &v[start + pat.len()..];
    let mut out = String::new();
    let mut escaped = false;
    let mut closed = false;
    for c in tail.chars() {
        if escaped {
            out.push(c); // \' → ', \\ → \ (второй символ экраНа — как есть)
            escaped = false;
        } else {
            match c {
                '\\' => escaped = true,
                '\'' => {
                    closed = true;
                    break;
                }
                other => out.push(other),
            }
        }
    }
    if !closed {
        return Err(TemplateError::NotString(format!("dict[{key}]")));
    }
    Ok(out)
}

fn eval_truthy(
    cond: &str,
    ctx: &TemplateContext,
    st: &HashMap<String, String>,
) -> Result<bool, TemplateError> {
    let c = cond.trim();
    for op in [" == ", " != "] {
        if let Some((l, r)) = c.split_once(op) {
            let lv = eval_value(l, ctx, st)?;
            let rv = eval_value(r, ctx, st)?;
            return Ok(if op.trim() == "==" {
                lv == rv
            } else {
                lv != rv
            });
        }
    }
    // простое значение; неопределённая переменная — falsy (семантика Jinja)
    match eval_value(c, ctx, st) {
        Ok(v) => match v.as_str() {
            "true" => Ok(true),
            "false" => Ok(false),
            other => Err(TemplateError::Unsupported(format!("condition `{other}`"))),
        },
        Err(TemplateError::Unsupported(_)) => Ok(false),
        Err(e) => Err(e),
    }
}

/// Эскейпы в строковых литералах шаблона (Jinja: `'\n'` = перевод строки).
fn unescape_literal(s: &str) -> String {
    s.replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\r", "\r")
        .replace("\\\\", "\\")
        .replace("\\'", "'")
        .replace("\\\"", "\"")
}

/// Разделить конкатенацию на верхнем уровне (вне кавычек).
fn split_plus(e: &str) -> Option<(&str, &str)> {
    let mut quote: Option<char> = None;
    let mut depth = 0usize;
    for (i, ch) in e.char_indices() {
        match quote {
            Some(q) if ch == q => quote = None,
            Some(_) => {}
            None => match ch {
                '\'' | '"' => quote = Some(ch),
                '(' | '[' => depth += 1,
                // N (iter-9): a malformed expression with `)` before any
                // `(` used to underflow usize (panic in debug, silently
                // wrong parse in release).
                ')' | ']' => depth = depth.saturating_sub(1),
                '+' if depth == 0 => return Some((&e[..i], &e[i + 1..])),
                _ => {}
            },
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen_template() -> Template {
        Template::compile(
            "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\\n'}}{% endif %}",
            HashMap::new(),
        )
        .unwrap()
    }

    fn ctx(messages: &[(&str, &str)]) -> TemplateContext {
        TemplateContext {
            messages: messages
                .iter()
                .map(|(r, c)| {
                    HashMap::from([
                        ("role".to_string(), r.to_string()),
                        ("content".to_string(), c.to_string()),
                    ])
                })
                .collect(),
            extra: HashMap::new(),
        }
    }

    #[test]
    fn qwen3_template_renders_chat() {
        let t = qwen_template();
        let out = t
            .render(
                &ctx(&[("user", "привет"), ("assistant", "здравствуй")])
                    .add_generation_prompt(true),
            )
            .unwrap();
        assert_eq!(
            out,
            "<|im_start|>user\nпривет<|im_end|>\n<|im_start|>assistant\nздравствуй<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn qwen3_no_generation_prompt() {
        let t = qwen_template();
        let out = t.render(&ctx(&[("user", "hi")])).unwrap();
        assert_eq!(out, "<|im_start|>user\nhi<|im_end|>\n");
    }

    #[test]
    fn gemma_template_roles() {
        let t = Template::compile(
            "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\\n' + message['content'] + '<end_of_turn>\\n<start_of_turn>model\\n' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + '<end_of_turn>\\n' }}{% endif %}{% endfor %}",
            HashMap::new(),
        )
        .unwrap();
        let out = t
            .render(&ctx(&[("user", "q"), ("assistant", "a")]))
            .unwrap();
        assert_eq!(
            out,
            "<start_of_turn>user\nq<end_of_turn>\n<start_of_turn>model\na<end_of_turn>\n"
        );
    }

    #[test]
    fn trim_filter_and_special_tokens() {
        let t = Template::compile("{{ '  x  ' | trim }}{{ eos_token }}", {
            let mut m = HashMap::new();
            m.insert("eos_token".into(), "<|end|>".into());
            m
        })
        .unwrap();
        assert_eq!(t.render(&TemplateContext::default()).unwrap(), "x<|end|>");
    }

    #[test]
    fn unsupported_construct_is_error_not_garbage() {
        let t = Template::compile("{% macro x() %}y{% endmacro %}", HashMap::new());
        assert!(t.is_err());
    }

    /// Nested-quote close markers (plan 2.8): `{{ '}}' -}}` closes at the
    /// trailing `-}}`, not at the `}}` INSIDE the string literal (the old
    /// tokenizer cut the content at the first `}}` → Expr("'")).
    #[test]
    fn quoted_close_marker_is_not_a_close() {
        let t = Template::compile("{{ '}}' -}}", HashMap::new()).unwrap();
        assert_eq!(t.render(&TemplateContext::default()).unwrap(), "}}");
        let t = Template::compile("{{ 'a}}b' }}", HashMap::new()).unwrap();
        assert_eq!(t.render(&TemplateContext::default()).unwrap(), "a}}b");
        // double-quoted with the escaped quote inside the literal — the
        // close scan must skip the escaped `'` too, and the eval unescapes.
        let t = Template::compile("{{ \"it\\'s }} ok\" }}", HashMap::new()).unwrap();
        assert_eq!(t.render(&TemplateContext::default()).unwrap(), "it's }} ok");
    }

    /// {# #} комментарии (задекларированы в доке токенизатора, но не
    /// реализованы): контент отбрасывается, trim-маркеры работают.
    #[test]
    fn comment_markers_are_dropped() {
        let t = Template::compile("a{# comment }} with 'quotes' #}b", HashMap::new()).unwrap();
        assert_eq!(t.render(&TemplateContext::default()).unwrap(), "ab");
        let t = Template::compile("a {#- c -#} b", HashMap::new()).unwrap();
        assert_eq!(t.render(&TemplateContext::default()).unwrap(), "ab");
        let t = Template::compile("{{ 'x' }}{# trailing #}", HashMap::new()).unwrap();
        assert_eq!(t.render(&TemplateContext::default()).unwrap(), "x");
    }

    /// Апостроф и backslash в content не должны обрезать значение:
    /// render_message экранирует их, dict_get разворачивает.
    #[test]
    fn apostrophe_in_content_is_not_truncated() {
        let t = Template::compile(
            "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}{% endfor %}",
            HashMap::new(),
        )
        .unwrap();
        let out = t
            .render(&ctx(&[("user", "don't stop, 100% \\ real")]))
            .unwrap();
        assert_eq!(out, "user: don't stop, 100% \\ real");
    }
}

#[cfg(test)]
mod trim_tests {
    use super::*;

    /// Jinja trim-маркеры: {%- -%} {{- -}} (использует реальный Qwen3-шаблон).
    #[test]
    fn trim_markers_remove_whitespace() {
        let t = Template::compile(
            "{%- for m in messages -%}\n  {{- m }}  \n{%- endfor -%}",
            HashMap::new(),
        )
        .unwrap();
        let ctx = TemplateContext {
            messages: vec![],
            extra: HashMap::from([("m".into(), "x".into()), ("loop.index0".into(), "0".into())]),
        };
        // цикл по messages пуст — но выражение m не используется; проверяем
        // что компиляция и рендер пустого тела не падают и текст тримится
        let out = t.render(&ctx).unwrap();
        assert_eq!(out, "");
    }

    #[test]
    fn qwen3_real_style_template() {
        // фрагмент реального Qwen3-шаблона (имитация с trim-маркерами)
        let t = Template::compile(
            "{%- for message in messages -%}{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' -}}{%- endfor -%}{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}",
            HashMap::new(),
        )
        .unwrap();
        let ctx = TemplateContext {
            messages: vec![HashMap::from([
                ("role".into(), "user".into()),
                ("content".into(), "привет".into()),
            ])],
            extra: HashMap::from([("add_generation_prompt".into(), "true".into())]),
        };
        let out = t.render(&ctx).unwrap();
        assert_eq!(
            out,
            "<|im_start|>user\nпривет<|im_end|>\n<|im_start|>assistant\n"
        );
    }
}
