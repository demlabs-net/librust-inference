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

fn tokenize(src: &str) -> Result<Vec<Tok>, TemplateError> {
    let mut toks = Vec::new();
    let mut rest = src;
    while !rest.is_empty() {
        let expr_at = rest.find("{{");
        let tag_at = rest.find("{%");
        let com_at = rest.find("{#");
        // ближайший маркер
        let mut best = rest.len();
        let mut kind = 0usize; // 0=none 1=expr 2=tag 3=comment
        for (at, k) in [(expr_at, 1usize), (tag_at, 2), (com_at, 3)] {
            if let Some(a) = at {
                if a < best {
                    best = a;
                    kind = k;
                }
            }
        }
        if kind == 0 {
            toks.push(Tok::Text(rest.to_string()));
            break;
        }
        if best > 0 {
            toks.push(Tok::Text(rest[..best].to_string()));
        }
        match kind {
            1 => {
                let end = rest[best + 2..]
                    .find("}}")
                    .ok_or(TemplateError::UnclosedTag("{{"))?;
                toks.push(Tok::Expr(rest[best + 2..best + 2 + end].trim().to_string()));
                rest = &rest[best + 2 + end + 2..];
            }
            2 => {
                let end = rest[best + 2..]
                    .find("%}")
                    .ok_or(TemplateError::UnclosedTag("{%"))?;
                toks.push(Tok::Tag(rest[best + 2..best + 2 + end].trim().to_string()));
                rest = &rest[best + 2 + end + 2..];
            }
            _ => {
                let end = rest[best + 2..]
                    .find("#}")
                    .ok_or(TemplateError::UnclosedTag("{#"))?;
                rest = &rest[best + 2 + end + 2..];
            }
        }
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
    // сообщение сериализуется как `{'role': '...', 'content': '...'}`
    let role = m.get("role").map(String::as_str).unwrap_or("");
    let content = m.get("content").map(String::as_str).unwrap_or("");
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
    if let Some(rest) = e.strip_suffix(']') {
        if let Some((obj, key)) = rest.split_once("['") {
            let key = key
                .strip_suffix('\'')
                .ok_or_else(|| TemplateError::Unsupported(e.into()))?;
            return eval_index(obj.trim(), key, ctx, st);
        }
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
    st: &HashMap<String, String>,
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

/// Достать значение из сериализованного словаря `{'role': '...', 'content': '...'}`.
fn dict_get(v: &str, key: &str) -> VResult {
    let pat = format!("'{key}': '");
    let start = v
        .find(&pat)
        .ok_or_else(|| TemplateError::NotString(format!("dict[{key}]")))?;
    let tail = &v[start + pat.len()..];
    let end = tail
        .find('\'')
        .ok_or_else(|| TemplateError::NotString(format!("dict[{key}]")))?;
    Ok(tail[..end].to_string())
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
                ')' | ']' => depth -= 1,
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
}
