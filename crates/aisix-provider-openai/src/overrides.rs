//! Per-`ProviderKey` request/response override primitives.
//!
//! Skeleton for [api7/AISIX-Cloud#302](https://github.com/api7/AISIX-Cloud/issues/302)
//! Phase A. The issue's §5 RuntimeConfig adds a `request` block
//! (`param_renames` / `param_constraints` / `default_headers` /
//! `default_body_fields`) and a `response` block (`stream_done_marker` /
//! `content_list_to_string` / `error_envelope` / `reasoning_field`) so
//! cp-api can capture per-provider quirks without forking a Bridge.
//!
//! This module ships the primitive transforms. **Nothing in
//! [`OpenAiBridge`](crate::OpenAiBridge) wires them in yet** — Phase
//! A2.5 added the on-disk shape (`RequestOverrides` / `ResponseOverrides`
//! on [`aisix_core::ProviderKey`]); Phase D consumes the functions
//! here from inside the Bridge once the new contract cuts over. Until
//! then the public API is exercised by unit tests against
//! `serde_json::Value` / `http::HeaderMap` inputs.
//!
//! The closed schema types ([`ParamConstraints`], [`StreamDoneMarker`])
//! live in `aisix-core` so cp-api can write them straight into etcd
//! payloads; this module re-uses those types for its apply-function
//! signatures so cp-api and the DP agree on a single wire shape.
//! [`StreamDoneOutcome`] is purely a runtime evaluation result and
//! stays here — it never serializes.
//!
//! Reference implementations consulted:
//! - LiteLLM `convert_content_list_to_str` —
//!   <https://github.com/BerriAI/litellm/blob/main/litellm/litellm_core_utils/prompt_templates/common_utils.py>
//! - LiteLLM `convert_content_list_to_string` flag —
//!   <https://github.com/BerriAI/litellm/blob/main/litellm/llms/openai_like/dynamic_config.py>
//! - DeepSeek reasoning shape (`delta.reasoning_content`) —
//!   <https://api-docs.deepseek.com/guides/reasoning_model>
//! - OpenAI SSE `data: [DONE]` terminator —
//!   <https://platform.openai.com/docs/api-reference/chat/streaming>

use std::collections::HashMap;

use aisix_core::{ParamConstraints, StreamDoneMarker};
use http::{
    header::{HeaderName, HeaderValue},
    HeaderMap,
};
use serde_json::{Map, Value};

/// Outcome of evaluating an SSE stream against a
/// [`StreamDoneMarker`] policy. Runtime-only — never serialized to
/// etcd, so it stays in the provider crate rather than `aisix-core`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamDoneOutcome {
    /// Stream complied with the policy.
    Ok,
    /// Required terminator was missing.
    MissingDoneMarker,
    /// Terminator was forbidden but observed.
    UnexpectedDoneMarker,
}

/// Apply `request.param_renames` to a JSON request body.
///
/// Rewrites every present top-level key whose name matches a key in
/// `renames` to the renames-map's value. Absent source keys are
/// no-ops.
///
/// **Collision semantics: source wins.** When both source and target
/// are present in the body, the source value overwrites the target.
/// Matches LiteLLM's convention across `databricks` / `openai_like` /
/// `sambanova` / `dynamic_config` transformations — for the canonical
/// `max_completion_tokens → max_tokens` case the newer (source) value
/// is what the caller intended; an accidentally-leftover deprecated
/// `max_tokens` value should not shadow it.
///
/// Non-object bodies are no-ops by construction; the function does
/// not panic on `null` / array / scalar inputs.
pub fn apply_param_renames(body: &mut Value, renames: &HashMap<String, String>) {
    let Some(obj) = body.as_object_mut() else {
        return;
    };
    for (from, to) in renames {
        if from == to {
            continue;
        }
        let Some(value) = obj.remove(from) else {
            continue;
        };
        // Source wins: overwrite any pre-existing target value.
        obj.insert(to.clone(), value);
    }
}

/// Apply `request.param_constraints` to a JSON request body.
///
/// Phase A handles `temperature` only — other parameter clamps are
/// added when a real upstream needs them. If the body has no
/// `temperature` field, or its value is not a number, the function
/// is a no-op (the upstream itself will surface invalid types).
pub fn apply_param_constraints(body: &mut Value, constraints: &ParamConstraints) {
    let Some(obj) = body.as_object_mut() else {
        return;
    };
    let Some(temp) = obj.get_mut("temperature") else {
        return;
    };
    let Some(current) = temp.as_f64() else {
        return;
    };
    let mut next = current;
    if let Some(max) = constraints.temperature_max {
        if next > max {
            next = max;
        }
    }
    if let Some(min) = constraints.temperature_min {
        if next < min {
            next = min;
        }
    }
    if next != current {
        // serde_json::Number::from_f64 returns None for NaN/Inf —
        // those aren't valid JSON anyway, so leaving the field
        // unchanged on the conversion failure is correct.
        if let Some(num) = serde_json::Number::from_f64(next) {
            *temp = Value::Number(num);
        }
    }
}

/// Authentication-related headers that `apply_default_headers` will
/// never set, even if cp-api validation slips and allows them through.
/// Defense-in-depth: a misconfigured `default_headers` block must never
/// override the auth header the OpenAiBridge sets itself. cp-api SHOULD
/// reject these at write time (issue #302 §5 validation rules), but
/// the DP enforces it again at apply time.
///
/// `HeaderName` is case-insensitive so the lowercase form is the
/// canonical comparison key.
const RESERVED_DEFAULT_HEADERS: &[&str] = &[
    "authorization",        // OpenAI / Anthropic / Vertex Bearer
    "x-api-key",            // Anthropic raw, also OpenAI legacy proxies
    "x-goog-api-key",       // Gemini API key
    "api-key",              // Azure OpenAI key
    "x-amz-security-token", // AWS SigV4 session header (Bedrock)
    "x-amz-date",           // AWS SigV4 timestamp (Bedrock)
];

/// Apply `request.default_headers` to an outbound `HeaderMap`.
///
/// Headers already present on `headers` (case-insensitive, since
/// [`http::HeaderName`] is case-insensitive) are left alone — the
/// caller's explicit value always wins over a default. Names or
/// values that fail [`HeaderName`] / [`HeaderValue`] parsing are
/// silently skipped; the default block came from cp-api validation
/// and any unparseable entry is a config error one layer up, not a
/// runtime failure the dispatch should hard-fail on.
///
/// **Auth-header guard:** keys in [`RESERVED_DEFAULT_HEADERS`] are
/// dropped before insertion as defense-in-depth — a misconfigured
/// default_headers block must never inject `Authorization` or vendor
/// API-key headers that would override the Bridge's own auth.
pub fn apply_default_headers(headers: &mut HeaderMap, defaults: &HashMap<String, String>) {
    for (name, value) in defaults {
        let Ok(parsed_name) = name.parse::<HeaderName>() else {
            continue;
        };
        if RESERVED_DEFAULT_HEADERS.contains(&parsed_name.as_str()) {
            continue;
        }
        if headers.contains_key(&parsed_name) {
            continue;
        }
        let Ok(parsed_value) = HeaderValue::from_str(value) else {
            continue;
        };
        headers.insert(parsed_name, parsed_value);
    }
}

/// Apply `request.default_body_fields` to a JSON request body.
///
/// Adds each entry of `defaults` to the body when the key is absent.
/// Existing keys are left untouched (caller wins). Non-object bodies
/// are a no-op.
pub fn apply_default_body_fields(body: &mut Value, defaults: &Map<String, Value>) {
    let Some(obj) = body.as_object_mut() else {
        return;
    };
    for (key, value) in defaults {
        obj.entry(key.clone()).or_insert_with(|| value.clone());
    }
}

/// Evaluate an SSE stream's terminator against a policy.
///
/// `done_marker_seen` reports whether the stream actually emitted
/// `data: [DONE]`. Returns the resulting [`StreamDoneOutcome`] so
/// the caller decides whether to emit a synthesized terminator,
/// surface a wire-shape error, or accept silently.
pub fn apply_stream_done_marker_policy(
    policy: StreamDoneMarker,
    done_marker_seen: bool,
) -> StreamDoneOutcome {
    match (policy, done_marker_seen) {
        (StreamDoneMarker::Required, true) => StreamDoneOutcome::Ok,
        (StreamDoneMarker::Required, false) => StreamDoneOutcome::MissingDoneMarker,
        (StreamDoneMarker::Optional, _) => StreamDoneOutcome::Ok,
        (StreamDoneMarker::None, false) => StreamDoneOutcome::Ok,
        (StreamDoneMarker::None, true) => StreamDoneOutcome::UnexpectedDoneMarker,
    }
}

/// Apply `response.content_list_to_string` to a chat-completions
/// request body.
///
/// Some OpenAI-compat upstreams (Mistral on Azure AI, SambaNova,
/// Heroku) only accept `message.content` as a string. This walks
/// every entry in the top-level `messages` array and, when an
/// entry's `content` is an array of `{type:"text", text:"..."}`
/// blocks, replaces it with the concatenated text of those blocks.
///
/// Matches LiteLLM's `convert_content_list_to_str`: text blocks are
/// concatenated with no separator. Non-text blocks (image_url,
/// audio, etc.) are skipped — same behavior as the reference — and
/// if every block is non-text the field is left as-is, since
/// flattening a vision payload to an empty string would silently
/// drop user intent.
pub fn apply_content_list_to_string(body: &mut Value) {
    let Some(obj) = body.as_object_mut() else {
        return;
    };
    let Some(messages) = obj.get_mut("messages").and_then(|m| m.as_array_mut()) else {
        return;
    };
    for message in messages {
        let Some(message_obj) = message.as_object_mut() else {
            continue;
        };
        let Some(content) = message_obj.get_mut("content") else {
            continue;
        };
        let Some(blocks) = content.as_array() else {
            continue;
        };
        let mut texts = String::new();
        let mut saw_text = false;
        for block in blocks {
            let Some(block_obj) = block.as_object() else {
                continue;
            };
            if block_obj.get("type").and_then(|t| t.as_str()) != Some("text") {
                continue;
            }
            if let Some(text) = block_obj.get("text").and_then(|t| t.as_str()) {
                texts.push_str(text);
                saw_text = true;
            }
        }
        if saw_text {
            *content = Value::String(texts);
        }
    }
}

/// Lift a nested reasoning field on a streaming chunk up to the
/// canonical `delta.reasoning_content` slot.
///
/// `path` is a `.`-separated address inside `chunk` (e.g.
/// `"delta.reasoning_content"`). When the value at that path is a
/// non-empty string and lives on a `delta` object, we copy it onto
/// the same delta as `reasoning_content`. When the source path
/// itself is `delta.reasoning_content` the function is a no-op
/// (already canonical).
///
/// This intentionally does *not* mutate any other shape — the only
/// guarantee is "after a successful call, the canonical
/// `reasoning_content` slot reflects whatever the upstream put at
/// `path`, when the upstream put a string there". Reference:
/// LiteLLM normalizes vendor-specific reasoning fields onto
/// `reasoning_content` in [`litellm/llms/ovhcloud/chat/transformation.py`](https://github.com/BerriAI/litellm/blob/main/litellm/llms/ovhcloud/chat/transformation.py).
pub fn extract_reasoning_field(chunk: &mut Value, path: &str) {
    let segments: Vec<&str> = path.split('.').filter(|s| !s.is_empty()).collect();
    if segments.is_empty() {
        return;
    }

    // Walk every choice's `delta` for an OpenAI-style chunk:
    // `{ "choices": [{ "delta": { ... } }, ...] }`. The path
    // semantics from issue #302 §5 (`"delta.reasoning_content"`)
    // are relative to each entry's delta-bearing object, so we
    // require the first segment to be `delta`.
    if segments[0] != "delta" || segments.len() < 2 {
        return;
    }
    let tail = &segments[1..];

    let Some(choices) = chunk
        .as_object_mut()
        .and_then(|obj| obj.get_mut("choices"))
        .and_then(|c| c.as_array_mut())
    else {
        return;
    };

    for choice in choices {
        let Some(delta) = choice
            .as_object_mut()
            .and_then(|obj| obj.get_mut("delta"))
            .and_then(|d| d.as_object_mut())
        else {
            continue;
        };

        // Drill down `tail` inside delta to fetch the source string.
        let mut cursor: &Value = match delta.get(tail[0]) {
            Some(v) => v,
            None => continue,
        };
        for seg in &tail[1..] {
            cursor = match cursor.as_object().and_then(|o| o.get(*seg)) {
                Some(v) => v,
                None => {
                    cursor = &Value::Null;
                    break;
                }
            };
        }
        let Some(source_str) = cursor.as_str() else {
            continue;
        };
        if source_str.is_empty() {
            continue;
        }
        // Canonical slot is `delta.reasoning_content`. If the source
        // path *is* the canonical slot the entry already exists and
        // the insert is a no-op of the same value — cheaper to leave
        // it alone.
        if tail == ["reasoning_content"] {
            continue;
        }
        let owned = source_str.to_string();
        delta.insert("reasoning_content".to_string(), Value::String(owned));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // --- apply_param_renames --------------------------------------

    #[test]
    fn renames_known_param() {
        let mut body = json!({ "max_completion_tokens": 100, "temperature": 0.5 });
        let renames = HashMap::from([(
            "max_completion_tokens".to_string(),
            "max_tokens".to_string(),
        )]);
        apply_param_renames(&mut body, &renames);
        assert_eq!(body, json!({ "max_tokens": 100, "temperature": 0.5 }));
    }

    #[test]
    fn rename_absent_key_is_noop() {
        let mut body = json!({ "temperature": 0.5 });
        let renames = HashMap::from([(
            "max_completion_tokens".to_string(),
            "max_tokens".to_string(),
        )]);
        apply_param_renames(&mut body, &renames);
        assert_eq!(body, json!({ "temperature": 0.5 }));
    }

    #[test]
    fn rename_preserves_value_type() {
        // Source value type (array, object, bool, number, null) is
        // preserved unchanged after rename.
        let mut body = json!({ "stop": ["END"] });
        let renames = HashMap::from([("stop".to_string(), "stop_sequences".to_string())]);
        apply_param_renames(&mut body, &renames);
        assert_eq!(body, json!({ "stop_sequences": ["END"] }));
    }

    #[test]
    fn applies_multiple_renames() {
        let mut body = json!({
            "max_completion_tokens": 100,
            "stop": ["X"],
            "temperature": 0.5
        });
        let renames = HashMap::from([
            (
                "max_completion_tokens".to_string(),
                "max_tokens".to_string(),
            ),
            ("stop".to_string(), "stop_sequences".to_string()),
        ]);
        apply_param_renames(&mut body, &renames);
        assert_eq!(
            body,
            json!({
                "max_tokens": 100,
                "stop_sequences": ["X"],
                "temperature": 0.5
            })
        );
    }

    #[test]
    fn rename_source_wins_when_both_present() {
        // Both source and target keys are present in the body. Per
        // LiteLLM convention (source wins), the source's value should
        // overwrite the target's pre-existing value — for the canonical
        // max_completion_tokens → max_tokens case, the newer source
        // value 100 wins over the deprecated target value 50.
        let mut body = json!({ "max_completion_tokens": 100, "max_tokens": 50 });
        let renames = HashMap::from([(
            "max_completion_tokens".to_string(),
            "max_tokens".to_string(),
        )]);
        apply_param_renames(&mut body, &renames);
        assert_eq!(body, json!({ "max_tokens": 100 }));
    }

    #[test]
    fn rename_to_self_is_noop() {
        let mut body = json!({ "model": "gpt-4o" });
        let renames = HashMap::from([("model".to_string(), "model".to_string())]);
        apply_param_renames(&mut body, &renames);
        assert_eq!(body, json!({ "model": "gpt-4o" }));
    }

    #[test]
    fn renames_on_non_object_body_is_noop() {
        let mut body = json!(42);
        let renames = HashMap::from([("a".to_string(), "b".to_string())]);
        apply_param_renames(&mut body, &renames);
        assert_eq!(body, json!(42));
    }

    // --- apply_param_constraints ----------------------------------

    #[test]
    fn temperature_above_max_is_clamped() {
        let mut body = json!({ "temperature": 1.7 });
        let constraints = ParamConstraints {
            temperature_max: Some(1.0),
            temperature_min: None,
        };
        apply_param_constraints(&mut body, &constraints);
        assert_eq!(body["temperature"].as_f64(), Some(1.0));
    }

    #[test]
    fn temperature_within_range_is_untouched() {
        let mut body = json!({ "temperature": 0.7 });
        let constraints = ParamConstraints {
            temperature_max: Some(1.0),
            temperature_min: Some(0.0),
        };
        apply_param_constraints(&mut body, &constraints);
        assert_eq!(body["temperature"].as_f64(), Some(0.7));
    }

    #[test]
    fn temperature_below_min_is_clamped() {
        let mut body = json!({ "temperature": -0.2 });
        let constraints = ParamConstraints {
            temperature_max: None,
            temperature_min: Some(0.0),
        };
        apply_param_constraints(&mut body, &constraints);
        assert_eq!(body["temperature"].as_f64(), Some(0.0));
    }

    #[test]
    fn temperature_missing_is_noop() {
        let mut body = json!({ "model": "gpt-4o" });
        let constraints = ParamConstraints {
            temperature_max: Some(1.0),
            temperature_min: None,
        };
        apply_param_constraints(&mut body, &constraints);
        assert_eq!(body, json!({ "model": "gpt-4o" }));
    }

    #[test]
    fn temperature_non_number_is_noop() {
        // Garbage-typed temperature lets the upstream surface the
        // type error; the clamp doesn't try to coerce.
        let mut body = json!({ "temperature": "hot" });
        let constraints = ParamConstraints {
            temperature_max: Some(1.0),
            temperature_min: None,
        };
        apply_param_constraints(&mut body, &constraints);
        assert_eq!(body, json!({ "temperature": "hot" }));
    }

    #[test]
    fn empty_constraints_is_noop() {
        let mut body = json!({ "temperature": 2.0 });
        let constraints = ParamConstraints::default();
        apply_param_constraints(&mut body, &constraints);
        assert_eq!(body["temperature"].as_f64(), Some(2.0));
    }

    // --- apply_default_headers ------------------------------------

    #[test]
    fn adds_missing_default_header() {
        let mut headers = HeaderMap::new();
        let defaults = HashMap::from([("anthropic-version".to_string(), "2023-06-01".to_string())]);
        apply_default_headers(&mut headers, &defaults);
        assert_eq!(headers.get("anthropic-version").unwrap(), "2023-06-01");
    }

    #[test]
    fn does_not_overwrite_existing_header() {
        let mut headers = HeaderMap::new();
        headers.insert("x-foo", HeaderValue::from_static("caller-value"));
        let defaults = HashMap::from([("x-foo".to_string(), "default-value".to_string())]);
        apply_default_headers(&mut headers, &defaults);
        assert_eq!(headers.get("x-foo").unwrap(), "caller-value");
    }

    #[test]
    fn header_match_is_case_insensitive() {
        // Caller-set header in mixed case must still block a default
        // header keyed in lowercase — http::HeaderName is canonicalized.
        let mut headers = HeaderMap::new();
        headers.insert("X-Foo", HeaderValue::from_static("caller-value"));
        let defaults = HashMap::from([("x-foo".to_string(), "default-value".to_string())]);
        apply_default_headers(&mut headers, &defaults);
        assert_eq!(headers.get("x-foo").unwrap(), "caller-value");
    }

    #[test]
    fn skips_unparseable_header_name() {
        let mut headers = HeaderMap::new();
        let defaults = HashMap::from([
            ("not a valid name".to_string(), "v".to_string()),
            ("x-foo".to_string(), "ok".to_string()),
        ]);
        apply_default_headers(&mut headers, &defaults);
        assert!(headers.get("x-foo").is_some());
        assert_eq!(headers.len(), 1);
    }

    #[test]
    fn rejects_reserved_auth_headers() {
        // Defense-in-depth: even if cp-api validation slipped and shipped
        // a default_headers block containing Authorization / X-Api-Key /
        // X-Goog-Api-Key, DP must not inject them. Case-insensitive via
        // HeaderName canonicalization.
        let mut headers = HeaderMap::new();
        let defaults = HashMap::from([
            (
                "Authorization".to_string(),
                "Bearer attacker-token".to_string(),
            ),
            ("X-Api-Key".to_string(), "attacker-key".to_string()),
            ("X-API-KEY".to_string(), "attacker-key-2".to_string()),
            (
                "x-goog-api-key".to_string(),
                "attacker-google-key".to_string(),
            ),
            ("x-foo".to_string(), "ok-default".to_string()),
        ]);
        apply_default_headers(&mut headers, &defaults);
        assert!(
            headers.get("authorization").is_none(),
            "auth must be blocked"
        );
        assert!(
            headers.get("x-api-key").is_none(),
            "x-api-key must be blocked"
        );
        assert!(
            headers.get("x-goog-api-key").is_none(),
            "x-goog-api-key must be blocked"
        );
        assert_eq!(headers.get("x-foo").unwrap(), "ok-default");
        assert_eq!(headers.len(), 1, "only x-foo should have been inserted");
    }

    // --- apply_default_body_fields --------------------------------

    #[test]
    fn adds_missing_default_field() {
        let mut body = json!({ "model": "gpt-4o" });
        let defaults: Map<String, Value> =
            serde_json::from_value(json!({ "safe_prompt": true })).unwrap();
        apply_default_body_fields(&mut body, &defaults);
        assert_eq!(body, json!({ "model": "gpt-4o", "safe_prompt": true }));
    }

    #[test]
    fn does_not_overwrite_existing_field() {
        let mut body = json!({ "safe_prompt": false });
        let defaults: Map<String, Value> =
            serde_json::from_value(json!({ "safe_prompt": true })).unwrap();
        apply_default_body_fields(&mut body, &defaults);
        assert_eq!(body, json!({ "safe_prompt": false }));
    }

    #[test]
    fn default_body_fields_non_object_is_noop() {
        let mut body = json!([1, 2, 3]);
        let defaults: Map<String, Value> =
            serde_json::from_value(json!({ "safe_prompt": true })).unwrap();
        apply_default_body_fields(&mut body, &defaults);
        assert_eq!(body, json!([1, 2, 3]));
    }

    // --- apply_stream_done_marker_policy --------------------------

    #[test]
    fn done_required_and_seen_is_ok() {
        assert_eq!(
            apply_stream_done_marker_policy(StreamDoneMarker::Required, true),
            StreamDoneOutcome::Ok
        );
    }

    #[test]
    fn done_required_and_missing_is_error() {
        assert_eq!(
            apply_stream_done_marker_policy(StreamDoneMarker::Required, false),
            StreamDoneOutcome::MissingDoneMarker
        );
    }

    #[test]
    fn done_optional_accepts_either() {
        assert_eq!(
            apply_stream_done_marker_policy(StreamDoneMarker::Optional, true),
            StreamDoneOutcome::Ok
        );
        assert_eq!(
            apply_stream_done_marker_policy(StreamDoneMarker::Optional, false),
            StreamDoneOutcome::Ok
        );
    }

    #[test]
    fn done_none_accepts_absence() {
        assert_eq!(
            apply_stream_done_marker_policy(StreamDoneMarker::None, false),
            StreamDoneOutcome::Ok
        );
    }

    #[test]
    fn done_none_flags_unexpected_marker() {
        assert_eq!(
            apply_stream_done_marker_policy(StreamDoneMarker::None, true),
            StreamDoneOutcome::UnexpectedDoneMarker
        );
    }

    // --- apply_content_list_to_string -----------------------------

    #[test]
    fn flattens_two_text_blocks() {
        // Matches LiteLLM's convert_content_list_to_str: text fields
        // are concatenated with no separator.
        let mut body = json!({
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text", "text": "hi" },
                    { "type": "text", "text": "world" }
                ]
            }]
        });
        apply_content_list_to_string(&mut body);
        assert_eq!(body["messages"][0]["content"], json!("hiworld"));
    }

    #[test]
    fn leaves_string_content_alone() {
        let mut body = json!({
            "messages": [{ "role": "user", "content": "already a string" }]
        });
        apply_content_list_to_string(&mut body);
        assert_eq!(body["messages"][0]["content"], json!("already a string"));
    }

    #[test]
    fn preserves_image_only_blocks() {
        // Vision payloads with only image_url blocks: don't silently
        // drop them by flattening to "". Leave as-is and let the
        // upstream surface the unsupported wire.
        let original = json!({
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "image_url", "image_url": { "url": "https://x.example/a.png" } }
                ]
            }]
        });
        let mut body = original.clone();
        apply_content_list_to_string(&mut body);
        assert_eq!(body, original);
    }

    #[test]
    fn skips_non_text_blocks_but_flattens_text_neighbours() {
        let mut body = json!({
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text", "text": "describe " },
                    { "type": "image_url", "image_url": { "url": "https://x.example/a.png" } },
                    { "type": "text", "text": "this" }
                ]
            }]
        });
        apply_content_list_to_string(&mut body);
        assert_eq!(body["messages"][0]["content"], json!("describe this"));
    }

    #[test]
    fn no_messages_field_is_noop() {
        let mut body = json!({ "model": "gpt-4o" });
        apply_content_list_to_string(&mut body);
        assert_eq!(body, json!({ "model": "gpt-4o" }));
    }

    // --- extract_reasoning_field ----------------------------------

    #[test]
    fn extracts_deepseek_reasoning_to_canonical_slot() {
        // DeepSeek emits `reasoning_content` already on delta — the
        // canonical slot matches, so it's a structural no-op but
        // must not corrupt the chunk.
        let mut chunk = json!({
            "choices": [
                { "delta": { "reasoning_content": "thinking..." } }
            ]
        });
        let snapshot = chunk.clone();
        extract_reasoning_field(&mut chunk, "delta.reasoning_content");
        assert_eq!(chunk, snapshot);
    }

    #[test]
    fn lifts_nested_reasoning_to_canonical_slot() {
        // Hypothetical upstream nests reasoning inside an extra
        // object: lift it onto the canonical key on the same delta.
        let mut chunk = json!({
            "choices": [
                {
                    "delta": {
                        "vendor_extras": { "thoughts": "thinking..." }
                    }
                }
            ]
        });
        extract_reasoning_field(&mut chunk, "delta.vendor_extras.thoughts");
        assert_eq!(
            chunk["choices"][0]["delta"]["reasoning_content"],
            json!("thinking...")
        );
        // Source field is preserved — extraction is non-destructive.
        assert_eq!(
            chunk["choices"][0]["delta"]["vendor_extras"]["thoughts"],
            json!("thinking...")
        );
    }

    #[test]
    fn missing_source_path_is_noop() {
        let mut chunk = json!({
            "choices": [{ "delta": { "content": "hello" } }]
        });
        let snapshot = chunk.clone();
        extract_reasoning_field(&mut chunk, "delta.reasoning_content");
        assert_eq!(chunk, snapshot);
    }

    #[test]
    fn empty_string_source_is_noop() {
        // An empty reasoning_content string should not populate the
        // canonical slot — caller treats it the same as missing.
        let mut chunk = json!({
            "choices": [{ "delta": { "vendor_extras": { "thoughts": "" } } }]
        });
        extract_reasoning_field(&mut chunk, "delta.vendor_extras.thoughts");
        assert!(chunk["choices"][0]["delta"]
            .get("reasoning_content")
            .is_none());
    }

    #[test]
    fn non_delta_root_path_is_rejected() {
        // The contract is "lift from delta sub-path". A root that
        // isn't `delta` is treated as a no-op rather than silently
        // mutating an unrelated tree.
        let mut chunk = json!({
            "choices": [{ "delta": { "content": "hi" }, "extras": { "thoughts": "t" } }]
        });
        let snapshot = chunk.clone();
        extract_reasoning_field(&mut chunk, "extras.thoughts");
        assert_eq!(chunk, snapshot);
    }

    #[test]
    fn handles_multiple_choices() {
        let mut chunk = json!({
            "choices": [
                { "delta": { "vendor_extras": { "thoughts": "first" } } },
                { "delta": { "vendor_extras": { "thoughts": "second" } } }
            ]
        });
        extract_reasoning_field(&mut chunk, "delta.vendor_extras.thoughts");
        assert_eq!(
            chunk["choices"][0]["delta"]["reasoning_content"],
            json!("first")
        );
        assert_eq!(
            chunk["choices"][1]["delta"]["reasoning_content"],
            json!("second")
        );
    }
}
