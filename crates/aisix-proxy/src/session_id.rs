//! The session a request belongs to, as the client identifies it.
//!
//! Agent clients tag every turn of one conversation with a stable id, each in
//! its own place: a vendor header, Codex's unprefixed headers, the Anthropic
//! `metadata.user_id`, a W3C `baggage` entry. [`resolve`] reads them in one
//! fixed order, so any feature keyed on a session reads the same id.

use axum::http::HeaderMap;
use serde_json::Value;

/// The gateway's own header, read before any client's.
pub(crate) const SESSION_ID_HEADER: &str = "x-aisix-session-id";

/// Longer ids are treated as absent: a session id is a key, not a payload.
const MAX_SESSION_ID_BYTES: usize = 256;

/// Codex's unprefixed headers, session before thread; the underscore
/// spellings are what builds before its header rename sent.
const CODEX_SESSION_HEADERS: [&str; 4] =
    ["session-id", "session_id", "thread-id", "conversation_id"];

/// The session id of a request, or `None` when it carries none. First valid
/// value wins:
///
/// 1. `x-aisix-session-id`;
/// 2. any `x-<vendor>-session-id` header whose value is an id-shaped token
///    (8 or more of `[A-Za-z0-9_-]`), the lexicographically first header name
///    when several are;
/// 3. for a Codex caller (`User-Agent` starting `codex` followed by `-`, `_`,
///    space or `/`), `session-id`, `session_id`, `thread-id`,
///    `conversation_id`, id-shaped;
/// 4. `x-session-id`, id-shaped;
/// 5. the body's `metadata.user_id`: an object's `session_id`; a string
///    holding a JSON object's `session_id`; or a string's text after its last
///    `_session_` — each `[A-Za-z0-9_-]+`;
/// 6. the body's `metadata.session_id`, a non-empty string;
/// 7. the `session.id` entry of the `baggage` header.
///
/// A value longer than 256 bytes counts as absent wherever it is found.
pub(crate) fn resolve(headers: &HeaderMap, metadata: Option<&Value>) -> Option<String> {
    let header = |name: &str| headers.get(name).and_then(|v| v.to_str().ok());
    let usable = |v: &str| (!v.is_empty() && v.len() <= MAX_SESSION_ID_BYTES).then(|| v.to_owned());

    header(SESSION_ID_HEADER)
        .and_then(usable)
        .or_else(|| vendor_header(headers))
        .or_else(|| {
            is_codex_caller(header("user-agent")?)
                .then(|| {
                    CODEX_SESSION_HEADERS
                        .iter()
                        .filter_map(|name| header(name))
                        .find(|v| is_id_token(v))
                        .map(str::to_owned)
                })
                .flatten()
        })
        .or_else(|| {
            header("x-session-id")
                .filter(|v| is_id_token(v))
                .map(str::to_owned)
        })
        .or_else(|| metadata.and_then(user_id_session).and_then(|v| usable(&v)))
        .or_else(|| {
            metadata
                .and_then(|m| m.get("session_id"))
                .and_then(Value::as_str)
                .and_then(usable)
        })
        .or_else(|| {
            headers
                .get_all("baggage")
                .iter()
                .filter_map(|v| v.to_str().ok())
                .find_map(baggage_session_id)
                .and_then(usable)
        })
}

fn vendor_header(headers: &HeaderMap) -> Option<String> {
    headers
        .iter()
        .filter(|(name, _)| {
            let name = name.as_str();
            name != SESSION_ID_HEADER
                && name.starts_with("x-")
                && name.ends_with("-session-id")
                && name.len() > "x-".len() + "-session-id".len()
        })
        .filter_map(|(name, value)| Some((name.as_str(), value.to_str().ok()?)))
        .filter(|(_, value)| is_id_token(value))
        .min_by_key(|(name, _)| *name)
        .map(|(_, value)| value.to_owned())
}

fn is_codex_caller(user_agent: &str) -> bool {
    let bytes = user_agent.as_bytes();
    bytes.len() > 5
        && bytes[..5].eq_ignore_ascii_case(b"codex")
        && matches!(bytes[5], b'-' | b'_' | b' ' | b'/')
}

/// 8 or more of `[A-Za-z0-9_-]`, at most [`MAX_SESSION_ID_BYTES`].
fn is_id_token(value: &str) -> bool {
    (8..=MAX_SESSION_ID_BYTES).contains(&value.len()) && is_id_chars(value)
}

fn is_id_chars(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

/// The session inside an Anthropic `metadata.user_id`.
fn user_id_session(metadata: &Value) -> Option<String> {
    let object_session = |user_id: &Value| {
        user_id
            .get("session_id")
            .and_then(Value::as_str)
            .filter(|s| is_id_chars(s))
            .map(str::to_owned)
    };
    match metadata.get("user_id")? {
        user_id @ Value::Object(_) => object_session(user_id),
        Value::String(user_id) => {
            if user_id.trim_start().starts_with('{') {
                if let Ok(parsed @ Value::Object(_)) = serde_json::from_str::<Value>(user_id) {
                    return object_session(&parsed);
                }
            }
            let (_, session) = user_id.rsplit_once("_session_")?;
            is_id_chars(session).then(|| session.to_owned())
        }
        _ => None,
    }
}

/// `session.id` in a W3C Baggage header (`session.id=abc,user.id=42`).
fn baggage_session_id(baggage: &str) -> Option<&str> {
    baggage.split(',').find_map(|member| {
        let (key, value) = member.trim().split_once('=')?;
        let value = value.trim();
        (key.trim() == "session.id" && !value.is_empty()).then_some(value)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (name, value) in pairs {
            map.append(
                axum::http::HeaderName::from_bytes(name.as_bytes()).unwrap(),
                value.parse().unwrap(),
            );
        }
        map
    }

    const UUID: &str = "e96634a3-fa28-4083-b354-55542e2dca01";

    #[test]
    fn precedence_and_value_rules() {
        let long = "a".repeat(257);
        type Case<'a> = (
            &'a str,
            Vec<(&'a str, &'a str)>,
            Option<Value>,
            Option<&'a str>,
        );
        let cases: Vec<Case<'_>> = vec![
            ("nothing", vec![], None, None),
            (
                "own header beats every other source",
                vec![
                    ("x-aisix-session-id", "s1"),
                    ("x-claude-code-session-id", UUID),
                    ("x-session-id", "bare-session"),
                ],
                Some(json!({"session_id": "body-session"})),
                Some("s1"),
            ),
            (
                "own header over 256 bytes is absent",
                vec![("x-aisix-session-id", &long)],
                None,
                None,
            ),
            (
                "own header over 256 bytes falls through",
                vec![
                    ("x-aisix-session-id", &long),
                    ("x-claude-code-session-id", UUID),
                ],
                None,
                Some(UUID),
            ),
            (
                "vendor header",
                vec![("x-claude-code-session-id", UUID)],
                None,
                Some(UUID),
            ),
            (
                "x-litellm-session-id is a vendor header",
                vec![("x-litellm-session-id", "litellm-123")],
                None,
                Some("litellm-123"),
            ),
            (
                "vendor value too short",
                vec![("x-claude-code-session-id", "short")],
                None,
                None,
            ),
            (
                "vendor value with a dot",
                vec![("x-claude-code-session-id", "abc.defgh")],
                None,
                None,
            ),
            (
                "vendor headers: lexicographically first name wins",
                vec![
                    ("x-zed-session-id", "zzzzzzzz"),
                    ("x-acme-session-id", "aaaaaaaa"),
                ],
                None,
                Some("aaaaaaaa"),
            ),
            (
                "an invalid first vendor header does not hide a valid one",
                vec![
                    ("x-acme-session-id", "bad"),
                    ("x-zed-session-id", "zzzzzzzz"),
                ],
                None,
                Some("zzzzzzzz"),
            ),
            (
                "x-session-id is not a vendor header",
                vec![("x-session-id", "s")],
                None,
                None,
            ),
            (
                "codex headers need a codex caller",
                vec![("session-id", UUID)],
                None,
                None,
            ),
            (
                "codex session-id",
                vec![
                    ("user-agent", "codex_cli_rs/0.80.0 (Linux)"),
                    ("session-id", UUID),
                    ("thread-id", "thread-0001"),
                ],
                None,
                Some(UUID),
            ),
            (
                "codex thread-id when no session header",
                vec![
                    ("user-agent", "codex-tui/1.0"),
                    ("thread-id", "thread-0001"),
                ],
                None,
                Some("thread-0001"),
            ),
            (
                "codex legacy spellings",
                vec![("user-agent", "Codex/1"), ("conversation_id", "conv-00001")],
                None,
                Some("conv-00001"),
            ),
            (
                "codexx is not codex",
                vec![("user-agent", "codexx/1"), ("session-id", UUID)],
                None,
                None,
            ),
            (
                "vendor header beats codex",
                vec![
                    ("user-agent", "codex_cli_rs/1"),
                    ("session-id", UUID),
                    ("x-acme-session-id", "acme-session"),
                ],
                None,
                Some("acme-session"),
            ),
            (
                "bare x-session-id",
                vec![("x-session-id", "opencode-1")],
                None,
                Some("opencode-1"),
            ),
            (
                "bare x-session-id too short",
                vec![("x-session-id", "s")],
                None,
                None,
            ),
            (
                "bare header beats body",
                vec![("x-session-id", "opencode-1")],
                Some(json!({"session_id": "body-session"})),
                Some("opencode-1"),
            ),
            (
                "user_id object",
                vec![],
                Some(json!({"user_id": {"session_id": UUID}})),
                Some(UUID),
            ),
            (
                "user_id JSON string (current Claude Code)",
                vec![],
                Some(
                    json!({"user_id": format!("{{\"device_id\":\"d\",\"account_uuid\":\"\",\"session_id\":\"{UUID}\"}}")}),
                ),
                Some(UUID),
            ),
            (
                "user_id legacy string",
                vec![],
                Some(json!({"user_id": format!("user_abc_account__session_{UUID}")})),
                Some(UUID),
            ),
            (
                "user_id string without a session",
                vec![],
                Some(json!({"user_id": "alice"})),
                None,
            ),
            (
                "user_id beats metadata.session_id",
                vec![],
                Some(json!({"user_id": {"session_id": "from-user-id"}, "session_id": "plain"})),
                Some("from-user-id"),
            ),
            (
                "an unusable user_id falls through to metadata.session_id",
                vec![],
                Some(json!({"user_id": {"session_id": "has space"}, "session_id": "plain"})),
                Some("plain"),
            ),
            (
                "metadata.session_id",
                vec![],
                Some(json!({"session_id": "x"})),
                Some("x"),
            ),
            (
                "metadata.session_id empty",
                vec![],
                Some(json!({"session_id": ""})),
                None,
            ),
            (
                "metadata.session_id not a string",
                vec![],
                Some(json!({"session_id": 7})),
                None,
            ),
            (
                "metadata.session_id over 256 bytes",
                vec![],
                Some(json!({"session_id": long.clone()})),
                None,
            ),
            (
                "baggage",
                vec![("baggage", "user.id=42, session.id=bag-1")],
                None,
                Some("bag-1"),
            ),
            (
                "baggage without session",
                vec![("baggage", "user.id=42")],
                None,
                None,
            ),
            (
                "body beats baggage",
                vec![("baggage", "session.id=bag-1")],
                Some(json!({"session_id": "body"})),
                Some("body"),
            ),
        ];
        for (name, hs, metadata, want) in cases {
            assert_eq!(
                resolve(&headers(&hs), metadata.as_ref()).as_deref(),
                want,
                "{name}"
            );
        }
    }
}
