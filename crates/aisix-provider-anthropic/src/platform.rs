//! An Anthropic Messages body sent to Claude on a cloud platform's own
//! Messages routes — Amazon Bedrock `InvokeModel` and Vertex AI
//! `rawPredict` — instead of being translated through the chat form.
//!
//! Both platforms take the Anthropic Messages body with a few differences,
//! which [`shape_request`] applies and nothing else:
//!
//! - the model is named by the URL, so `model` leaves the body, and the
//!   platform's own `anthropic_version` goes in;
//! - a closed set of top-level fields is accepted ([`allowed_fields`]),
//!   and `metadata` carries `user_id` only;
//! - `anthropic-beta` values are the ones the platform knows
//!   ([`platform_beta`]), mapped to its own names where they differ —
//!   Bedrock reads them from the body's `anthropic_beta`, Vertex AI from
//!   the header;
//! - a `cache_control` marker loses the sub-fields the platform refuses;
//! - neither platform reads media from a URL, so the gateway downloads it
//!   ([`media_urls`] / [`inline_media`]).
//!
//! Reference:
//! <https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages-request-response.html>,
//! <https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude/use-claude>.

use std::collections::BTreeSet;

use aisix_gateway::media_fetch::{FetchedMedia, MediaUrl};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    Bedrock,
    Vertex,
}

impl Platform {
    /// The upstream's name in a client-facing message.
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Bedrock => "Claude on Amazon Bedrock",
            Self::Vertex => "Claude on Vertex AI",
        }
    }

    fn anthropic_version(self) -> &'static str {
        match self {
            Self::Bedrock => "bedrock-2023-05-31",
            Self::Vertex => "vertex-2023-10-16",
        }
    }
}

/// The top-level request fields Bedrock `InvokeModel` accepts for Claude.
const BEDROCK_FIELDS: &[&str] = &[
    "anthropic_version",
    "anthropic_beta",
    "max_tokens",
    "messages",
    "system",
    "stop_sequences",
    "temperature",
    "top_p",
    "top_k",
    "tools",
    "tool_choice",
    "thinking",
    "metadata",
    "output_config",
    "safeguards",
    "context_management",
];

/// The top-level request fields forwarded to Vertex AI `rawPredict`: the
/// Anthropic Messages request fields, without `service_tier`, which
/// Vertex AI refuses.
const VERTEX_FIELDS: &[&str] = &[
    "anthropic_version",
    "max_tokens",
    "messages",
    "metadata",
    "stop_sequences",
    "stream",
    "system",
    "temperature",
    "thinking",
    "tool_choice",
    "tools",
    "top_k",
    "top_p",
    "inference_geo",
    "mcp_servers",
    "context_management",
    "compaction",
    "container",
    "output_config",
    "speed",
    "cache_control",
    "safeguards",
];

pub fn allowed_fields(platform: Platform) -> &'static [&'static str] {
    match platform {
        Platform::Bedrock => BEDROCK_FIELDS,
        Platform::Vertex => VERTEX_FIELDS,
    }
}

/// `anthropic-beta` values Bedrock accepts, keyed by the Anthropic name.
/// A value not listed is dropped: Bedrock answers an unknown one with 400.
const BEDROCK_BETAS: &[(&str, &str)] = &[
    (
        "advanced-tool-use-2025-11-20",
        "tool-search-tool-2025-10-19",
    ),
    ("compact-2026-01-12", "compact-2026-01-12"),
    ("computer-use-2025-01-24", "computer-use-2025-01-24"),
    ("computer-use-2025-11-24", "computer-use-2025-11-24"),
    ("context-1m-2025-08-07", "context-1m-2025-08-07"),
    (
        "context-management-2025-06-27",
        "context-management-2025-06-27",
    ),
    (
        "dangerous-tool-use-2026-09-03",
        "dangerous-tool-use-2026-09-03",
    ),
    ("effort-2025-11-24", "effort-2025-11-24"),
    (
        "fine-grained-tool-streaming-2025-05-14",
        "fine-grained-tool-streaming-2025-05-14",
    ),
    (
        "thinking-binding-controls-2026-08-01",
        "thinking-binding-controls-2026-08-01",
    ),
    ("tool-search-tool-2025-10-19", "tool-search-tool-2025-10-19"),
    (
        "mid-conversation-output-config-2026-07-01",
        "mid-conversation-output-config-2026-07-01",
    ),
    (
        "thinking-display-updates-2026-08-18",
        "thinking-display-updates-2026-08-18",
    ),
    (
        "mid-conversation-tool-changes-2026-07-01",
        "mid-conversation-tool-changes-2026-07-01",
    ),
];

/// `anthropic-beta` values Vertex AI accepts, keyed by the Anthropic name.
const VERTEX_BETAS: &[(&str, &str)] = &[
    (
        "advanced-tool-use-2025-11-20",
        "tool-search-tool-2025-10-19",
    ),
    ("compact-2026-01-12", "compact-2026-01-12"),
    ("computer-use-2025-01-24", "computer-use-2025-01-24"),
    ("computer-use-2025-11-24", "computer-use-2025-11-24"),
    ("context-1m-2025-08-07", "context-1m-2025-08-07"),
    (
        "context-management-2025-06-27",
        "context-management-2025-06-27",
    ),
    (
        "dangerous-tool-use-2026-09-03",
        "dangerous-tool-use-2026-09-03",
    ),
    (
        "interleaved-thinking-2025-05-14",
        "interleaved-thinking-2025-05-14",
    ),
    ("per-turn-control-2026-07-01", "per-turn-control-2026-07-01"),
    (
        "thinking-binding-controls-2026-08-01",
        "thinking-binding-controls-2026-08-01",
    ),
    (
        "thinking-display-updates-2026-08-18",
        "thinking-display-updates-2026-08-18",
    ),
    ("tool-search-tool-2025-10-19", "tool-search-tool-2025-10-19"),
    ("web-search-2025-03-05", "web-search-2025-03-05"),
    ("inline-tools-2026-09-15", "inline-tools-2026-09-15"),
];

/// The platform's name for an Anthropic `anthropic-beta` value, `None`
/// when the platform does not take it.
pub fn platform_beta(platform: Platform, beta: &str) -> Option<&'static str> {
    let table = match platform {
        Platform::Bedrock => BEDROCK_BETAS,
        Platform::Vertex => VERTEX_BETAS,
    };
    table.iter().find(|(k, _)| *k == beta).map(|(_, v)| *v)
}

const EFFORT_BETA: &str = "effort-2025-11-24";
const COMPACT_BETA: &str = "compact-2026-01-12";
const CONTEXT_MANAGEMENT_BETA: &str = "context-management-2025-06-27";
const TOOL_SEARCH_BETA: &str = "tool-search-tool-2025-10-19";
const WEB_SEARCH_BETA: &str = "web-search-2025-03-05";
const DANGEROUS_TOOL_USE_BETA: &str = "dangerous-tool-use-2026-09-03";
const PER_TURN_CONTROL_BETA: &str = "per-turn-control-2026-07-01";

/// The `context_management.edits` types Bedrock `InvokeModel` runs, with
/// the beta each needs. Any other edit is refused there with a 400.
const BEDROCK_CONTEXT_EDITS: &[(&str, &str)] = &[
    ("compact_20260112", COMPACT_BETA),
    ("clear_tool_uses_20250919", CONTEXT_MANAGEMENT_BETA),
];

/// The values of every `anthropic-beta` header in `headers`, comma lists
/// split.
pub fn header_betas(headers: Option<&http::HeaderMap>) -> Vec<String> {
    headers
        .into_iter()
        .flat_map(|h| h.get_all("anthropic-beta"))
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(','))
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_string)
        .collect()
}

/// The Claude model name inside a platform model id —
/// `claude-sonnet-4-5-20250929-v1:0` out of
/// `us.anthropic.claude-sonnet-4-5-20250929-v1:0` or an inference-profile
/// ARN ending in one, `claude-sonnet-4-5@20250929` as it is.
pub fn claude_name(upstream_id: &str) -> &str {
    let tail = upstream_id.rsplit('/').next().unwrap_or(upstream_id);
    match tail.to_ascii_lowercase().find("claude") {
        Some(at) => &tail[at..],
        None => tail,
    }
}

/// Adapt a caller's Anthropic Messages body for `platform` in place and
/// return the `anthropic-beta` values to send: on Bedrock they are also
/// written into the body, on Vertex AI the caller sends them as the
/// header. `betas` are the request's own (`anthropic-beta` headers).
pub fn shape_request(
    body: &mut Value,
    platform: Platform,
    upstream_id: &str,
    betas: Vec<String>,
) -> Vec<String> {
    let Some(obj) = body.as_object_mut() else {
        return Vec::new();
    };
    obj.remove("model");
    if platform == Platform::Bedrock {
        obj.remove("stream");
    }
    obj.insert(
        "anthropic_version".to_string(),
        Value::String(platform.anthropic_version().to_string()),
    );
    if let Some(metadata) = obj.remove("metadata") {
        let mut kept = Map::new();
        if let Some(user_id) = metadata.get("user_id").filter(|v| v.is_string()) {
            kept.insert("user_id".to_string(), user_id.clone());
        }
        if metadata.is_object() {
            obj.insert("metadata".to_string(), Value::Object(kept));
        }
    }
    fold_output_format(obj);

    let mut wanted: BTreeSet<String> = betas.into_iter().collect();
    match platform {
        Platform::Bedrock => {
            if effort_needs_beta(obj, upstream_id) {
                wanted.insert(EFFORT_BETA.to_string());
            }
            if uses_tool_search(obj) {
                wanted.insert(TOOL_SEARCH_BETA.to_string());
            }
            filter_bedrock_context_management(obj, &mut wanted);
        }
        Platform::Vertex => {
            if let Some(edits) = obj
                .get("context_management")
                .and_then(|c| c.get("edits"))
                .and_then(Value::as_array)
            {
                for edit in edits {
                    let beta =
                        if edit.get("type").and_then(Value::as_str) == Some("compact_20260112") {
                            COMPACT_BETA
                        } else {
                            CONTEXT_MANAGEMENT_BETA
                        };
                    wanted.insert(beta.to_string());
                }
            }
            if tools(obj).any(|t| {
                t.get("type")
                    .and_then(Value::as_str)
                    .is_some_and(|ty| ty.starts_with("web_search"))
            }) {
                wanted.insert(WEB_SEARCH_BETA.to_string());
            }
            if uses_tool_search(obj) {
                wanted.insert(TOOL_SEARCH_BETA.to_string());
            }
            if obj.get("safeguards").is_some_and(|v| !v.is_null()) {
                wanted.insert(DANGEROUS_TOOL_USE_BETA.to_string());
            }
            if obj
                .get("messages")
                .and_then(Value::as_array)
                .is_some_and(|ms| ms.iter().any(|m| m.get("output_config").is_some()))
            {
                wanted.insert(PER_TURN_CONTROL_BETA.to_string());
            }
        }
    }
    let sent: Vec<String> = wanted
        .iter()
        .filter_map(|b| platform_beta(platform, b))
        .map(str::to_string)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let dropped: Vec<&String> = wanted
        .iter()
        .filter(|b| platform_beta(platform, b).is_none())
        .collect();
    if !dropped.is_empty() {
        tracing::debug!(
            platform = platform.display_name(),
            dropped = ?dropped,
            "anthropic-beta values the platform does not take were not sent"
        );
    }
    if platform == Platform::Bedrock && !sent.is_empty() {
        obj.insert(
            "anthropic_beta".to_string(),
            Value::Array(sent.iter().cloned().map(Value::String).collect()),
        );
    }

    let allowed = allowed_fields(platform);
    obj.retain(|k, _| allowed.contains(&k.as_str()));

    match platform {
        Platform::Bedrock => {
            crate::wire::retain_cache_control_fields(body, |k| matches!(k, "type" | "ttl"))
        }
        Platform::Vertex => crate::wire::retain_cache_control_fields(body, |k| k != "scope"),
    }
    sent
}

/// The token-counting body for Bedrock `CountTokens` with the
/// `invokeModel` input: the body `/v1/messages` would send to InvokeModel
/// ([`shape_request`]), carrying the `max_tokens` the `InvokeModel` schema
/// requires — above any thinking budget, which InvokeModel validates and
/// which does not change the count. Returns `None` when the caller's
/// messages are all plain strings, which take the `converse` input
/// instead.
pub fn bedrock_count_tokens_invoke_body(body: &Value, upstream_id: &str) -> Option<Value> {
    let typed_blocks = body
        .get("messages")
        .and_then(Value::as_array)
        .is_none_or(|ms| {
            ms.iter().any(|m| {
                m.get("content")
                    .and_then(Value::as_array)
                    .is_some_and(|c| c.iter().any(|b| b.get("type").is_some()))
            })
        });
    if !typed_blocks {
        return None;
    }
    let mut out = body.clone();
    shape_request(&mut out, Platform::Bedrock, upstream_id, Vec::new());
    let budget = out
        .get("thinking")
        .and_then(|t| t.get("budget_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    if let Some(obj) = out.as_object_mut() {
        let max_tokens = obj
            .get("max_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(1024);
        obj.insert(
            "max_tokens".to_string(),
            Value::from(max_tokens.max(budget + 1)),
        );
    }
    Some(out)
}

/// The body for Vertex AI's count-tokens model: the fields, `metadata` and
/// `cache_control` adjustments `/v1/messages` gets there, with `model` —
/// which this route reads from the body — set to `model`.
pub fn vertex_count_tokens_body(body: &Value, model: &str) -> Value {
    let mut out = body.clone();
    shape_request(&mut out, Platform::Vertex, model, Vec::new());
    if let Some(obj) = out.as_object_mut() {
        obj.remove("anthropic_version");
        obj.remove("stream");
        obj.insert("model".to_string(), Value::String(model.to_string()));
    }
    out
}

/// Where each media URL in an Anthropic Messages body sits: `image` and
/// `document` blocks with a `url` source in a user turn, and the same
/// inside a `tool_result`.
pub fn media_urls(body: &Value) -> Vec<MediaUrl> {
    let mut out = Vec::new();
    let Some(messages) = body.get("messages").and_then(Value::as_array) else {
        return out;
    };
    for (i, m) in messages.iter().enumerate() {
        let Some(blocks) = m.get("content").and_then(Value::as_array) else {
            continue;
        };
        for (j, block) in blocks.iter().enumerate() {
            let at = format!("messages[{i}].content[{j}]");
            if let Some(url) = url_source(block) {
                out.push(MediaUrl {
                    uri: url.to_string(),
                    mime_type: None,
                    location: at.clone(),
                });
            }
            if block.get("type").and_then(Value::as_str) == Some("tool_result") {
                for (k, inner) in block
                    .get("content")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .enumerate()
                {
                    if let Some(url) = url_source(inner) {
                        out.push(MediaUrl {
                            uri: url.to_string(),
                            mime_type: None,
                            location: format!("{at}.content[{k}]"),
                        });
                    }
                }
            }
        }
    }
    out
}

fn url_source(block: &Value) -> Option<&str> {
    if !matches!(
        block.get("type").and_then(Value::as_str),
        Some("image" | "document")
    ) {
        return None;
    }
    let source = block.get("source")?;
    (source.get("type").and_then(Value::as_str) == Some("url"))
        .then(|| source.get("url").and_then(Value::as_str))
        .flatten()
}

/// Replace every URL source [`media_urls`] found with the downloaded
/// bytes. A URL that was not downloaded — `upstream.media_fetch` is off —
/// is refused, since the platform would refuse it too.
pub fn inline_media(
    body: &mut Value,
    platform: Platform,
    fetched: &FetchedMedia,
) -> Result<(), String> {
    let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) else {
        return Ok(());
    };
    for (i, m) in messages.iter_mut().enumerate() {
        let Some(blocks) = m.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        for (j, block) in blocks.iter_mut().enumerate() {
            let at = format!("messages[{i}].content[{j}]");
            inline_block(block, platform, fetched, &at)?;
            if block.get("type").and_then(Value::as_str) == Some("tool_result") {
                if let Some(inner) = block.get_mut("content").and_then(Value::as_array_mut) {
                    for (k, b) in inner.iter_mut().enumerate() {
                        inline_block(b, platform, fetched, &format!("{at}.content[{k}]"))?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn inline_block(
    block: &mut Value,
    platform: Platform,
    fetched: &FetchedMedia,
    at: &str,
) -> Result<(), String> {
    let Some(url) = url_source(block).map(str::to_string) else {
        return Ok(());
    };
    let Some(download) = fetched.take(&url) else {
        return Err(format!(
            "{at}: {} does not read media from a URL, and the gateway did not download {}; \
             send the content inline as base64",
            platform.display_name(),
            aisix_gateway::content_part::display_uri(&url),
        ));
    };
    let document = block.get("type").and_then(Value::as_str) == Some("document");
    let source = if document && download.mime_type.starts_with("text/") {
        let text = String::from_utf8(download.data.into_bytes())
            .map_err(|_| format!("{at}: the document at the URL is not UTF-8 text"))?;
        serde_json::json!({"type": "text", "media_type": "text/plain", "data": text})
    } else {
        serde_json::json!({
            "type": "base64",
            "media_type": download.mime_type,
            "data": download.data.into_base64(),
        })
    };
    block["source"] = source;
    Ok(())
}

/// One Anthropic SSE `error` frame.
pub fn error_frame(kind: &str, message: &str) -> bytes::Bytes {
    let data = serde_json::json!({
        "type": "error",
        "error": {"type": kind, "message": message},
    });
    bytes::Bytes::from(format!("event: error\ndata: {data}\n\n"))
}

/// One Anthropic SSE frame for an event payload, named by its `type`.
pub fn event_frame(payload: &[u8]) -> bytes::Bytes {
    let kind = serde_json::from_slice::<Value>(payload)
        .ok()
        .and_then(|v| v.get("type").and_then(Value::as_str).map(str::to_string));
    let mut out = Vec::with_capacity(payload.len() + 32);
    if let Some(kind) = kind {
        out.extend_from_slice(b"event: ");
        out.extend_from_slice(kind.as_bytes());
        out.push(b'\n');
    }
    out.extend_from_slice(b"data: ");
    out.extend_from_slice(payload);
    out.extend_from_slice(b"\n\n");
    bytes::Bytes::from(out)
}

fn tools(obj: &Map<String, Value>) -> impl Iterator<Item = &Value> {
    obj.get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
}

fn uses_tool_search(obj: &Map<String, Value>) -> bool {
    tools(obj).any(|t| {
        matches!(
            t.get("type").and_then(Value::as_str),
            Some("tool_search_tool_regex_20251119" | "tool_search_tool_bm25_20251119")
        )
    })
}

/// `output_config.effort` takes a beta before the families that made it
/// generally available (Claude 4.6). A model name the gateway cannot read
/// is treated as needing it.
fn effort_needs_beta(obj: &Map<String, Value>, upstream_id: &str) -> bool {
    let effort_set = obj
        .get("output_config")
        .and_then(|c| c.get("effort"))
        .and_then(Value::as_str)
        .is_some_and(|e| !e.is_empty());
    effort_set && crate::wire::claude_family(claude_name(upstream_id)).is_none_or(|v| v < (4, 6))
}

/// The legacy top-level `output_format` moves into `output_config.format`,
/// winning over one already there.
fn fold_output_format(obj: &mut Map<String, Value>) {
    let Some(format) = obj.remove("output_format") else {
        return;
    };
    if !format.is_object() {
        return;
    }
    match obj.get_mut("output_config").and_then(Value::as_object_mut) {
        Some(config) => {
            config.insert("format".to_string(), format);
        }
        None => {
            obj.insert(
                "output_config".to_string(),
                serde_json::json!({"format": format}),
            );
        }
    }
}

/// Keep the `context_management.edits` Bedrock runs and add the beta each
/// needs; drop the field when none is left.
fn filter_bedrock_context_management(obj: &mut Map<String, Value>, betas: &mut BTreeSet<String>) {
    let Some(cm) = obj.get_mut("context_management") else {
        return;
    };
    let Some(edits) = cm.get_mut("edits").and_then(Value::as_array_mut) else {
        obj.remove("context_management");
        return;
    };
    edits.retain(|e| {
        let beta = e.get("type").and_then(Value::as_str).and_then(|ty| {
            BEDROCK_CONTEXT_EDITS
                .iter()
                .find(|(t, _)| *t == ty)
                .map(|(_, b)| *b)
        });
        if let Some(beta) = beta {
            betas.insert(beta.to_string());
        }
        beta.is_some()
    });
    if edits.is_empty() {
        obj.remove("context_management");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn caller_body() -> Value {
        json!({
            "model": "alias",
            "stream": true,
            "max_tokens": 64,
            "metadata": {"user_id": "u-1", "session": "s"},
            "system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral", "ttl": "1h", "scope": "global"}}],
            "messages": [{"role": "user", "content": "hi"}],
            "context_management": {"edits": [
                {"type": "clear_thinking_20251015"},
                {"type": "clear_tool_uses_20250919"}
            ]},
            "output_config": {"effort": "high"},
            "service_tier": "auto",
            "unknown_future_field": 1
        })
    }

    #[test]
    fn bedrock_body_takes_its_fields_version_and_betas() {
        let mut body = caller_body();
        let betas = shape_request(
            &mut body,
            Platform::Bedrock,
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            vec![
                "context-1m-2025-08-07".into(),
                "interleaved-thinking-2025-05-14".into(),
            ],
        );
        assert_eq!(
            betas,
            [
                "context-1m-2025-08-07",
                "context-management-2025-06-27",
                "effort-2025-11-24"
            ]
        );
        assert_eq!(body["anthropic_beta"], json!(betas));
        assert_eq!(body["anthropic_version"], "bedrock-2023-05-31");
        assert_eq!(body["metadata"], json!({"user_id": "u-1"}));
        assert_eq!(
            body["context_management"],
            json!({"edits": [{"type": "clear_tool_uses_20250919"}]})
        );
        assert_eq!(
            body["system"][0]["cache_control"],
            json!({"type": "ephemeral", "ttl": "1h"})
        );
        for gone in ["model", "stream", "service_tier", "unknown_future_field"] {
            assert!(
                body.get(gone).is_none(),
                "{gone} must not reach InvokeModel"
            );
        }
    }

    #[test]
    fn vertex_body_keeps_stream_and_sends_betas_as_the_header() {
        let mut body = caller_body();
        let betas = shape_request(
            &mut body,
            Platform::Vertex,
            "claude-opus-4-6@default",
            vec![
                "effort-2025-11-24".into(),
                "interleaved-thinking-2025-05-14".into(),
            ],
        );
        assert_eq!(
            betas,
            [
                "context-management-2025-06-27",
                "interleaved-thinking-2025-05-14"
            ]
        );
        assert!(body.get("anthropic_beta").is_none());
        assert_eq!(body["anthropic_version"], "vertex-2023-10-16");
        assert_eq!(body["stream"], true);
        assert_eq!(
            body["system"][0]["cache_control"],
            json!({"type": "ephemeral", "ttl": "1h"})
        );
        for gone in ["model", "service_tier", "unknown_future_field"] {
            assert!(body.get(gone).is_none(), "{gone} must not reach rawPredict");
        }
    }

    #[test]
    fn effort_beta_only_before_claude_4_6() {
        let mut body = json!({"output_config": {"effort": "low"}, "messages": []});
        let betas = shape_request(
            &mut body,
            Platform::Bedrock,
            "global.anthropic.claude-opus-4-6-v1",
            Vec::new(),
        );
        assert!(betas.is_empty(), "{betas:?}");
    }

    #[test]
    fn media_urls_are_found_in_tool_results_too() {
        let body = json!({"messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "url", "url": "https://x/a.png"}},
            {"type": "tool_result", "tool_use_id": "t", "content": [
                {"type": "image", "source": {"type": "url", "url": "https://x/b.png"}}
            ]}
        ]}]});
        let found: Vec<(String, String)> = media_urls(&body)
            .into_iter()
            .map(|u| (u.uri, u.location))
            .collect();
        assert_eq!(
            found,
            [
                (
                    "https://x/a.png".to_string(),
                    "messages[0].content[0]".to_string()
                ),
                (
                    "https://x/b.png".to_string(),
                    "messages[0].content[1].content[0]".to_string()
                ),
            ]
        );
    }
}
