//! Tool-call handling every bridge shares.
//!
//! Gemini's thought signature rides in the tool-call id the gateway hands a
//! client (`<id>__thought__<signature>`, the signature URL-safe base64
//! without padding), because no client protocol has another field that
//! brings it back with the history. Only the Vertex Gemini bridge reads it.
//! Every other upstream must see the base id alone: OpenAI caps tool-call
//! ids at 64 characters and Bedrock also restricts them to
//! `[A-Za-z0-9_-]`, so a conversation that started on Gemini and moves to
//! another model would otherwise be rejected.

use std::borrow::Cow;

use serde_json::Value;

use crate::{BridgeError, ChatFormat};

/// Separates a tool-call id from the thought signature appended to it.
pub const THOUGHT_SIGNATURE_SEPARATOR: &str = "__thought__";

/// `id` split into its base id and the thought signature appended to it.
pub fn split_thought_signature(id: &str) -> (&str, Option<&str>) {
    match id.split_once(THOUGHT_SIGNATURE_SEPARATOR) {
        Some((base, signature)) => (base, Some(signature)),
        None => (id, None),
    }
}

/// `id` without a thought-signature suffix.
pub fn strip_thought_signature(id: &str) -> &str {
    split_thought_signature(id).0
}

fn has_signature(v: Option<&Value>) -> bool {
    v.and_then(Value::as_str)
        .is_some_and(|id| id.contains(THOUGHT_SIGNATURE_SEPARATOR))
}

fn strip_in_place(v: Option<&mut Value>) {
    if let Some(Value::String(id)) = v {
        if let Some((base, _)) = id.split_once(THOUGHT_SIGNATURE_SEPARATOR) {
            *id = base.to_string();
        }
    }
}

/// `req` with every tool-call id (assistant `tool_calls[].id`, tool
/// `tool_call_id`) reduced to its base id. Borrowed back unchanged when no
/// id carries a signature, which is every request that never touched a
/// Gemini model.
pub fn chat_without_thought_signatures(req: Cow<'_, ChatFormat>) -> Cow<'_, ChatFormat> {
    let signed = req.messages.iter().any(|m| {
        m.tool_call_id
            .as_deref()
            .is_some_and(|id| id.contains(THOUGHT_SIGNATURE_SEPARATOR))
            || m.extra
                .get("tool_calls")
                .and_then(Value::as_array)
                .is_some_and(|calls| calls.iter().any(|c| has_signature(c.get("id"))))
    });
    if !signed {
        return req;
    }
    let mut req = req.into_owned();
    for m in &mut req.messages {
        if let Some(id) = m.tool_call_id.as_mut() {
            let base = strip_thought_signature(id).len();
            id.truncate(base);
        }
        if let Some(calls) = m.extra.get_mut("tool_calls").and_then(Value::as_array_mut) {
            for call in calls {
                strip_in_place(call.get_mut("id"));
            }
        }
    }
    Cow::Owned(req)
}

/// Reduce every tool-call id in a raw request body (Anthropic Messages
/// `tool_use.id` / `tool_result.tool_use_id`, Responses `call_id`, chat
/// `tool_calls[].id` / `tool_call_id`) to its base id, wherever it sits.
/// A call's own arguments (`tool_use.input`, `arguments`) are the caller's
/// data and are left alone, whatever keys they hold.
pub fn strip_thought_signatures_in_body(body: &mut Value) {
    match body {
        Value::Object(obj) => {
            for (key, value) in obj.iter_mut() {
                match (key.as_str(), &*value) {
                    ("id" | "call_id" | "tool_call_id" | "tool_use_id", _) => {
                        strip_in_place(Some(value))
                    }
                    ("arguments", _) | ("input", Value::Object(_)) => {}
                    _ => strip_thought_signatures_in_body(value),
                }
            }
        }
        Value::Array(items) => items.iter_mut().for_each(strip_thought_signatures_in_body),
        _ => {}
    }
}

/// Refuse the deprecated `functions` / `function_call` request fields on a
/// bridge that builds its upstream body itself. Such a bridge has no
/// translation for them, and dropping them would answer a function-calling
/// request with prose.
pub fn reject_legacy_function_calling(req: &ChatFormat) -> Result<(), BridgeError> {
    match ["functions", "function_call"]
        .into_iter()
        .find(|k| req.extra.contains_key(*k))
    {
        Some(field) => Err(BridgeError::InvalidUpstreamConfig(format!(
            "`{field}` is not supported for this model; use `tools` and `tool_choice` instead"
        ))),
        None => Ok(()),
    }
}

/// The text a guardrail scans for a `tool_calls` array: every call as it
/// is, but with any thought signature cut off its `id`. The base id is
/// part of what the call carries — on the input side it is caller text an
/// upstream receives — so it is scanned like the rest of the call; the
/// signature is hundreds of bytes of opaque base64 a keyword rule could
/// match by chance, and no upstream but Gemini ever sees it.
pub fn tool_calls_scan_text(tool_calls: &Value) -> String {
    match tool_calls {
        Value::Array(calls) => blocks_scan_text(calls),
        other => other.to_string(),
    }
}

/// [`tool_calls_scan_text`] over any list of tool-call-bearing objects
/// (an Anthropic `content` array included): each one serialized with its
/// `id` reduced to the base id, as a JSON array.
pub fn blocks_scan_text<'a>(blocks: impl IntoIterator<Item = &'a Value>) -> String {
    struct BaseId<'a>(&'a Value);
    impl serde::Serialize for BaseId<'_> {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            use serde::ser::SerializeMap;
            match self.0 {
                Value::Object(obj) => {
                    let mut map = serializer.serialize_map(None)?;
                    for (k, v) in obj {
                        match (k.as_str(), v) {
                            ("id", Value::String(id)) => {
                                map.serialize_entry(k, strip_thought_signature(id))?
                            }
                            _ => map.serialize_entry(k, v)?,
                        }
                    }
                    map.end()
                }
                other => other.serialize(serializer),
            }
        }
    }
    let blocks: Vec<BaseId<'_>> = blocks.into_iter().map(BaseId).collect();
    serde_json::to_string(&blocks).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn signed_ids_are_reduced_to_their_base_everywhere() {
        let req: ChatFormat = serde_json::from_value(json!({
            "model": "m",
            "messages": [
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_a__thought__c2ln", "type": "function",
                     "function": {"name": "f", "arguments": "{}"}},
                    {"id": "call_b", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_a__thought__c2ln", "content": "ok"}
            ]
        }))
        .unwrap();
        let stripped = chat_without_thought_signatures(Cow::Borrowed(&req));
        assert!(matches!(stripped, Cow::Owned(_)));
        assert_eq!(stripped.messages[0].extra["tool_calls"][0]["id"], "call_a");
        assert_eq!(stripped.messages[0].extra["tool_calls"][1]["id"], "call_b");
        assert_eq!(stripped.messages[1].tool_call_id.as_deref(), Some("call_a"));
        assert!(matches!(
            chat_without_thought_signatures(Cow::Borrowed(&stripped)),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn raw_bodies_lose_the_suffix_on_every_id_field() {
        let mut body = json!({
            "messages": [{"role": "assistant", "content": [
                {"type": "tool_use", "id": "call_a__thought__c2ln", "name": "f",
                 "input": {"id": "kept__thought__as-is"}},
                {"type": "text", "text": "an id__thought__ in prose stays"}
            ]}, {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_a__thought__c2ln", "content": "ok"}
            ]}],
            "input": [{"type": "function_call", "call_id": "call_b__thought__eHl6"}]
        });
        strip_thought_signatures_in_body(&mut body);
        assert_eq!(body["messages"][0]["content"][0]["id"], "call_a");
        assert_eq!(
            body["messages"][0]["content"][0]["input"]["id"],
            "kept__thought__as-is"
        );
        assert_eq!(
            body["messages"][0]["content"][1]["text"],
            "an id__thought__ in prose stays"
        );
        assert_eq!(body["messages"][1]["content"][0]["tool_use_id"], "call_a");
        assert_eq!(body["input"][0]["call_id"], "call_b");
    }

    #[test]
    fn legacy_function_fields_are_refused() {
        let mut req = ChatFormat::new("m", vec![]);
        assert!(reject_legacy_function_calling(&req).is_ok());
        req.extra.insert("functions".into(), json!([]));
        assert!(matches!(
            reject_legacy_function_calling(&req),
            Err(BridgeError::InvalidUpstreamConfig(m)) if m.contains("tools")
        ));
    }

    #[test]
    fn scan_text_carries_the_base_id_without_its_signature() {
        let text = tool_calls_scan_text(&json!([
            {"id": "call_a__thought__c2ln", "type": "function",
             "function": {"name": "lookup", "arguments": "{\"q\":1}"}}
        ]));
        assert!(text.contains("\"call_a\""), "{text}");
        assert!(
            !text.contains("__thought__") && !text.contains("c2ln"),
            "{text}"
        );
        assert!(text.contains("lookup") && text.contains("{\\\"q\\\":1}"));
    }
}
