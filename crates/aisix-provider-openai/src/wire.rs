//! OpenAI chat-completions wire shapes.
//!
//! These mirror <https://platform.openai.com/docs/api-reference/chat>.
//! The gateway's own [`ChatFormat`](aisix_gateway::ChatFormat) is already
//! OpenAI-compatible, but we still serialise through a dedicated type so:
//!
//! 1. forward compatibility — OpenAI can add fields without forcing a
//!    breaking change on `ChatFormat`.
//! 2. deserialisation is strict where *we* read it (responses), loose
//!    where *they* read it (requests) — i.e. we accept extra fields
//!    from upstream but don't invent params.
//!
//! # Public surface
//!
//! The request/response/stream-chunk types and their conversion helpers
//! are `pub` (not `pub(crate)`) because **sibling provider crates** in
//! this workspace reuse them — Azure OpenAI Service's wire shape is
//! literally OpenAI chat-completions, so the
//! [`aisix-provider-azure-openai`](crate::aisix_provider_azure_openai)
//! crate parses Azure responses through these same types (Azure's
//! `prompt_filter_results` / `content_filter_results` extensions pass
//! through because none of the types set `deny_unknown_fields`). Future
//! OpenAI-compatible bridges (additional self-hosted endpoints, etc.)
//! follow the same pattern. The visibility is **not a public-SDK
//! stability promise**; it's an internal workspace contract. Embedding
//! types stay `pub(crate)` because they're scoped to
//! [`OpenAiBridge`](crate::OpenAiBridge) only until a sibling crate
//! needs them.

use aisix_gateway::{
    ChatChunk, ChatDelta, ChatFormat, ChatMessage, ChatResponse, EmbeddingObject, EmbeddingRequest,
    EmbeddingResponse, EmbeddingUsage, EmbeddingVector, FinishReason, Role, UsageStats,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiRequest<'a> {
    pub model: &'a str,
    pub messages: &'a [OpenAiMessage<'a>],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    pub stream: bool,
    #[serde(flatten)]
    pub extra: &'a serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OpenAiMessage<'a> {
    pub role: &'a str,
    /// `content` accepts string OR typed-block array per OpenAI's
    /// vision spec; we forward whichever the caller sent. The
    /// `untagged` enum makes `OpenAiContent::Blocks([...])`
    /// serialise as a bare array and `OpenAiContent::Text("...")`
    /// as a bare string — matching the OpenAI wire shape.
    pub content: OpenAiContent<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<&'a str>,
    /// Pass-through for OpenAI message-level fields the gateway
    /// doesn't model (`tool_calls`, `refusal`, `audio`, …). Mirrors
    /// `OpenAiRequest::extra` at the message level so conversation
    /// history with tool_call rounds round-trips to the upstream
    /// without the gateway having to model every OpenAI field.
    #[serde(flatten, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra: &'a serde_json::Map<String, serde_json::Value>,
}

/// Wire-shape of an OpenAI message's `content` field. OpenAI accepts
/// either a string (text-only message) or an array of typed content
/// blocks (vision / multimodal); we forward whichever the gateway's
/// [`ChatMessage`] carried in. See
/// <https://platform.openai.com/docs/guides/vision>.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum OpenAiContent<'a> {
    Text(&'a str),
    Blocks(&'a [serde_json::Value]),
}

/// Build the upstream request from the gateway's normalised format.
///
/// `upstream_model` is the part after the `<provider>/` prefix from the
/// Model entity (e.g. `"gpt-4o"`, not `"openai/gpt-4o"`).
pub fn build_request<'a>(
    req: &'a ChatFormat,
    upstream_model: &'a str,
    messages: &'a [OpenAiMessage<'a>],
    stream: bool,
) -> OpenAiRequest<'a> {
    OpenAiRequest {
        model: upstream_model,
        messages,
        temperature: req.temperature,
        top_p: req.top_p,
        max_tokens: req.max_tokens,
        stream,
        extra: &req.extra,
    }
}

pub fn messages_from(req: &ChatFormat) -> Vec<OpenAiMessage<'_>> {
    req.messages
        .iter()
        .map(|m| OpenAiMessage {
            role: role_str(m.role),
            // Vision / multimodal callers send `content` as an array
            // of typed blocks; OpenAI accepts the array form natively
            // (no translation needed for OpenAI / Gemini-OpenAI-compat /
            // DeepSeek-OpenAI-compat upstreams). When present forward
            // the raw blocks verbatim; otherwise forward the bare
            // string. See `ChatMessage::content_blocks` doc.
            content: match m.content_blocks.as_deref() {
                Some(blocks) => OpenAiContent::Blocks(blocks),
                None => OpenAiContent::Text(&m.content),
            },
            name: m.name.as_deref(),
            tool_call_id: m.tool_call_id.as_deref(),
            extra: &m.extra,
        })
        .collect()
}

fn role_str(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

fn role_from_str(s: &str) -> Role {
    match s {
        "system" => Role::System,
        "user" => Role::User,
        "tool" => Role::Tool,
        _ => Role::Assistant,
    }
}

#[derive(Debug, Deserialize)]
pub struct OpenAiResponse {
    pub id: String,
    pub model: String,
    pub choices: Vec<OpenAiChoice>,
    #[serde(default)]
    pub usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAiChoice {
    pub message: OpenAiResponseMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAiResponseMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    /// DeepSeek-reasoner (and other reasoning models) return the
    /// chain-of-thought text at `message.reasoning_content` on the
    /// non-streaming path (#466). Without this field it was silently
    /// dropped at deserialization — the streaming path already
    /// surfaces it via `extract_reasoning_field`, but non-streaming
    /// had no capture. Forwarded to the client verbatim via
    /// `ChatMessage.extra` so OpenAI-SDK clients see
    /// `choices[0].message.reasoning_content`.
    #[serde(default)]
    pub reasoning_content: Option<String>,
}

// `#[serde(default)]` at the container level so an upstream that omits
// any token counter (OpenAI-compatible providers vary on which fields
// they return) deserializes to 0 instead of failing the whole response
// body decode with `502 upstream_decode_error` (#474).
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct OpenAiUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    /// Present on responses that touched the prompt cache. Subset of
    /// `prompt_tokens`. Optional so older API versions / non-cached
    /// responses parse without the field.
    #[serde(default)]
    pub prompt_tokens_details: Option<OpenAiPromptDetails>,
    /// Present on responses from o1 / o3 reasoning models. Subset of
    /// `completion_tokens`.
    #[serde(default)]
    pub completion_tokens_details: Option<OpenAiCompletionDetails>,
    /// DeepSeek-native context-cache HIT count (#542). DeepSeek's
    /// OpenAI-compatible response puts the cache counters at the top
    /// level of `usage` (not nested under `prompt_tokens_details` like
    /// OpenAI). Optional so OpenAI / other compat upstreams parse
    /// without it.
    #[serde(default)]
    pub prompt_cache_hit_tokens: Option<u32>,
    /// DeepSeek-native context-cache MISS count (#542).
    #[serde(default)]
    pub prompt_cache_miss_tokens: Option<u32>,
}

#[derive(Debug, Default, Deserialize)]
pub struct OpenAiPromptDetails {
    /// Tokens served from the prompt cache (50% of prompt rate).
    #[serde(default)]
    pub cached_tokens: u32,
}

#[derive(Debug, Default, Deserialize)]
pub struct OpenAiCompletionDetails {
    /// o1/o3 reasoning tokens. Same rate as `completion_tokens`,
    /// surfaced separately so admins can see "of which N were
    /// reasoning" on the dashboard.
    #[serde(default)]
    pub reasoning_tokens: u32,
}

pub fn response_into_chat_response(mut raw: OpenAiResponse) -> ChatResponse {
    let first = raw.choices.drain(..).next();
    let (message, finish) = match first {
        Some(c) => {
            let mut extra = serde_json::Map::new();
            if let Some(tool_calls) = c.message.tool_calls {
                if !tool_calls.is_empty() {
                    extra.insert(
                        "tool_calls".to_string(),
                        serde_json::Value::Array(tool_calls),
                    );
                }
            }
            // #466: surface DeepSeek-reasoner's reasoning_content on the
            // non-streaming path. Stashed in `extra` so render.rs's
            // RenderedMessage (which flattens `extra`) emits it as
            // `message.reasoning_content` on the wire — matching the
            // streaming path's `delta.reasoning_content`. Skip empty
            // strings so a model that returns `""` doesn't add a noise
            // field.
            if let Some(reasoning) = c.message.reasoning_content {
                if !reasoning.is_empty() {
                    extra.insert(
                        "reasoning_content".to_string(),
                        serde_json::Value::String(reasoning),
                    );
                }
            }
            (
                ChatMessage {
                    role: role_from_str(&c.message.role),
                    content: c.message.content.unwrap_or_default(),
                    content_blocks: None,
                    name: None,
                    tool_call_id: None,
                    extra,
                },
                finish_reason(c.finish_reason.as_deref()),
            )
        }
        None => (ChatMessage::assistant(""), FinishReason::Stop),
    };

    let usage = raw.usage.map(into_usage).unwrap_or_default();

    ChatResponse {
        id: raw.id,
        model: raw.model,
        message,
        finish_reason: finish,
        usage,
    }
}

fn into_usage(u: OpenAiUsage) -> UsageStats {
    UsageStats {
        prompt_tokens: u.prompt_tokens,
        completion_tokens: u.completion_tokens,
        total_tokens: u.total_tokens,
        // Normalize the cache-hit count into the canonical field for
        // ALL providers (#542): OpenAI nests it under
        // `prompt_tokens_details.cached_tokens`; DeepSeek puts it at the
        // top level as `prompt_cache_hit_tokens`. Prefer the OpenAI
        // nested form when present, else fall back to DeepSeek's native
        // field. This is what surfaces as `prompt_tokens_details.cached_tokens`
        // on the client response regardless of upstream.
        cached_prompt_tokens: u
            .prompt_tokens_details
            .as_ref()
            .map(|d| d.cached_tokens)
            // A *zeroed* nested detail must not mask a real native
            // count (PR #442 audit MEDIUM-1): a hybrid OpenAI-compat
            // proxy could send `prompt_tokens_details:{cached_tokens:0}`
            // alongside a non-zero top-level `prompt_cache_hit_tokens`.
            // Treat nested-zero as "no signal" so the native count
            // wins; a genuine nested non-zero still takes precedence.
            .filter(|&n| n > 0)
            .or(u.prompt_cache_hit_tokens)
            .unwrap_or(0),
        reasoning_tokens: u
            .completion_tokens_details
            .as_ref()
            .map(|d| d.reasoning_tokens)
            .unwrap_or(0),
        // OpenAI doesn't currently expose Anthropic-style cache
        // creation/read counters; leave at 0 (cp-api falls back to
        // prompt rate for unset cache counters).
        cache_creation_tokens: 0,
        cache_read_tokens: 0,
        // Preserve DeepSeek's native counters verbatim for passthrough
        // (#542) so a client reading the native field names still works,
        // alongside the normalized `cached_prompt_tokens` above. `None`
        // for non-DeepSeek upstreams → the renderer omits the fields.
        prompt_cache_hit_tokens: u.prompt_cache_hit_tokens,
        prompt_cache_miss_tokens: u.prompt_cache_miss_tokens,
    }
}

fn finish_reason(raw: Option<&str>) -> FinishReason {
    match raw {
        Some("stop") | None => FinishReason::Stop,
        Some("length") => FinishReason::Length,
        Some("content_filter") => FinishReason::ContentFilter,
        Some("tool_calls") => FinishReason::ToolCalls,
        Some(other) => FinishReason::Other(other.to_string()),
    }
}

#[derive(Debug, Deserialize)]
pub struct OpenAiStreamChunk {
    pub id: String,
    pub model: String,
    pub choices: Vec<OpenAiStreamChoice>,
    #[serde(default)]
    pub usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAiStreamChoice {
    pub delta: OpenAiStreamDelta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAiStreamDelta {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    /// Canonical reasoning slot — DeepSeek emits this directly; for
    /// upstreams that put their reasoning text at a different path
    /// (issue #302 §5 `response.reasoning_field`) the bridge runs
    /// [`extract_reasoning_field`](crate::overrides::extract_reasoning_field)
    /// over the parsed [`Value`] before the typed parse so the lifted
    /// string lands here.
    #[serde(default)]
    pub reasoning_content: Option<String>,
}

pub fn stream_chunk_into_chat_chunk(mut raw: OpenAiStreamChunk) -> ChatChunk {
    let first = raw.choices.drain(..).next();
    let (delta, finish) = match first {
        Some(c) => (
            ChatDelta {
                role: c.delta.role.as_deref().map(role_from_str),
                content: c.delta.content,
                tool_calls: c.delta.tool_calls,
                reasoning_content: c.delta.reasoning_content,
            },
            c.finish_reason
                .as_deref()
                .map(|r| finish_reason(Some(r)))
                .filter(|_| c.finish_reason.is_some()),
        ),
        None => (ChatDelta::default(), None),
    };
    ChatChunk {
        id: raw.id,
        model: raw.model,
        delta,
        finish_reason: finish,
        usage: raw.usage.map(into_usage),
    }
}

// ─── Embeddings wire types ────────────────────────────────────────────────────

/// `input` field shape on the OpenAI `/v1/embeddings` request. Per
/// OpenAI's spec (<https://platform.openai.com/docs/api-reference/embeddings/create>)
/// the field accepts either a single string or an array — the gateway
/// preserves whichever the caller sent, per `docs/api-proxy.md` §4.4
/// "both pass through" and #162.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub(crate) enum OpenAiEmbedInput<'a> {
    /// Single-string form. Used when the caller's request body had
    /// `input: "..."` (string), preserving the wire shape on the
    /// upstream side.
    Single(&'a str),
    /// Array form. Used when the caller's request body had
    /// `input: ["...", ...]` (array) OR when the `input_was_single`
    /// signal is missing (round-tripped requests).
    Multi(&'a [String]),
}

/// Request body forwarded to OpenAI `/v1/embeddings`.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct OpenAiEmbedRequest<'a> {
    pub model: &'a str,
    pub input: OpenAiEmbedInput<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
}

/// One embedding object from OpenAI's response.
///
/// Issue #393: `embedding` is an `EmbeddingVector` (untagged enum
/// over `Vec<f32>` / `String`) so the deserializer accepts both
/// shapes OpenAI emits — `Vec<f32>` when the request carried
/// `encoding_format: "float"`, and base64 `String` when it carried
/// `encoding_format: "base64"` (the OpenAI SDK default). Pre-#393
/// this field was strict-typed `Vec<f32>` and every default-SDK
/// `embeddings.create({model, input})` call failed with
/// `502 upstream_decode_error` because the deserializer rejected
/// the string shape.
#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiEmbeddingObject {
    pub index: u32,
    pub object: String,
    pub embedding: EmbeddingVector,
}

/// Usage block from OpenAI's embeddings response. `#[serde(default)]`
/// so a provider that returns only `total_tokens` (e.g. Jina) — or omits
/// usage fields entirely — still deserializes instead of failing with
/// `502 upstream_decode_error` (#474).
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub(crate) struct OpenAiEmbedUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

/// Full response body from OpenAI `/v1/embeddings`.
#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiEmbedResponse {
    pub object: String,
    pub model: String,
    pub data: Vec<OpenAiEmbeddingObject>,
    #[serde(default)]
    pub usage: Option<OpenAiEmbedUsage>,
}

pub(crate) fn embed_request_body<'a>(
    req: &'a EmbeddingRequest,
    upstream_model: &'a str,
) -> OpenAiEmbedRequest<'a> {
    // Per #162: the upstream wire shape mirrors what the caller sent
    // when feasible. Single-string callers get `input: "text"`;
    // array-form callers get `input: ["text", ...]`. The
    // `input_was_single` flag from EmbeddingRequest is the load-
    // bearing signal — without it, callers reading docs §4.4 ("both
    // pass through") got always-array on the upstream wire,
    // contradicting the published contract.
    //
    // Defensive fallback: if `input_was_single` is true but the vec
    // is empty or has more than one element, drop back to array
    // form (the single-string shape is undefined for those cases).
    let input = if req.input_was_single && req.input.len() == 1 {
        OpenAiEmbedInput::Single(&req.input[0])
    } else {
        OpenAiEmbedInput::Multi(&req.input)
    };
    OpenAiEmbedRequest {
        model: upstream_model,
        input,
        encoding_format: req.encoding_format.as_deref(),
        dimensions: req.dimensions,
    }
}

pub(crate) fn embed_response_into(raw: OpenAiEmbedResponse) -> EmbeddingResponse {
    let usage = raw.usage.unwrap_or_default();
    EmbeddingResponse {
        object: raw.object,
        model: raw.model,
        data: raw
            .data
            .into_iter()
            .map(|e| EmbeddingObject {
                index: e.index,
                object: e.object,
                embedding: e.embedding,
            })
            .collect(),
        usage: EmbeddingUsage {
            prompt_tokens: usage.prompt_tokens,
            total_tokens: usage.total_tokens,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_streaming_response_parses_into_chat_response() {
        let body = r#"{
            "id": "cmpl-1",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hi there"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
        }"#;
        let raw: OpenAiResponse = serde_json::from_str(body).unwrap();
        let out = response_into_chat_response(raw);
        assert_eq!(out.id, "cmpl-1");
        assert_eq!(out.model, "gpt-4o");
        assert_eq!(out.message.role, Role::Assistant);
        assert_eq!(out.message.content, "hi there");
        assert_eq!(out.finish_reason, FinishReason::Stop);
        assert_eq!(out.usage.total_tokens, 6);
        // No cache / reasoning details on this minimal response →
        // counters stay at 0 (cp-api falls back to standard rates).
        assert_eq!(out.usage.cached_prompt_tokens, 0);
        assert_eq!(out.usage.reasoning_tokens, 0);
    }

    #[test]
    fn response_with_tool_calls_propagates_to_message_extra() {
        let body = r#"{
            "id": "cmpl-tc",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": "{\"tz\":\"UTC\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }"#;
        let raw: OpenAiResponse = serde_json::from_str(body).unwrap();
        let out = response_into_chat_response(raw);
        assert_eq!(out.finish_reason, FinishReason::ToolCalls);
        let tc = out
            .message
            .extra
            .get("tool_calls")
            .expect("tool_calls in extra")
            .as_array()
            .unwrap();
        assert_eq!(tc.len(), 1);
        assert_eq!(tc[0]["id"], "call_abc");
        assert_eq!(tc[0]["function"]["name"], "get_time");
    }

    #[test]
    fn cache_and_reasoning_details_populate_when_present() {
        // Verified shape from
        // https://platform.openai.com/docs/api-reference/chat/object#chat/object-usage
        // (prompt_tokens_details + completion_tokens_details).
        let body = r#"{
            "id": "cmpl-2",
            "object": "chat.completion",
            "model": "o1-2024-12-17",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
                "prompt_tokens_details": {"cached_tokens": 800},
                "completion_tokens_details": {"reasoning_tokens": 200}
            }
        }"#;
        let raw: OpenAiResponse = serde_json::from_str(body).unwrap();
        let out = response_into_chat_response(raw);
        assert_eq!(out.usage.prompt_tokens, 1000);
        assert_eq!(out.usage.cached_prompt_tokens, 800);
        assert_eq!(out.usage.completion_tokens, 500);
        assert_eq!(out.usage.reasoning_tokens, 200);
        // OpenAI upstreams don't emit DeepSeek's native top-level cache
        // counters — they stay None so the renderer omits them (#542).
        assert_eq!(out.usage.prompt_cache_hit_tokens, None);
        assert_eq!(out.usage.prompt_cache_miss_tokens, None);
    }

    /// Issue #542: DeepSeek's OpenAI-compatible response carries the
    /// context-cache counters at the TOP LEVEL of `usage`
    /// (`prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`), not
    /// nested under `prompt_tokens_details` like OpenAI. The bridge
    /// must (a) NORMALIZE the hit count into the canonical
    /// `cached_prompt_tokens` so it surfaces as OpenAI-shape
    /// `prompt_tokens_details.cached_tokens`, AND (b) PRESERVE the
    /// native fields verbatim for passthrough. Mirrors the ecosystem's
    /// hybrid (DeepSeek-native + OpenAI-canonical both present).
    #[test]
    fn deepseek_native_cache_counters_normalize_and_passthrough() {
        // Verified shape from https://api-docs.deepseek.com — DeepSeek's
        // usage object adds prompt_cache_hit_tokens / prompt_cache_miss_tokens.
        let body = r#"{
            "id": "cmpl-ds",
            "object": "chat.completion",
            "model": "deepseek-chat",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "total_tokens": 1100,
                "prompt_cache_hit_tokens": 768,
                "prompt_cache_miss_tokens": 232
            }
        }"#;
        let raw: OpenAiResponse = serde_json::from_str(body).unwrap();
        let out = response_into_chat_response(raw);
        // (a) normalized into the canonical cache-hit field
        assert_eq!(
            out.usage.cached_prompt_tokens, 768,
            "DeepSeek prompt_cache_hit_tokens must normalize into cached_prompt_tokens",
        );
        // (b) native fields preserved verbatim for passthrough
        assert_eq!(out.usage.prompt_cache_hit_tokens, Some(768));
        assert_eq!(out.usage.prompt_cache_miss_tokens, Some(232));
    }

    /// Issue #466: DeepSeek-reasoner returns `reasoning_content` on the
    /// non-streaming `choices[0].message`. The bridge must capture it
    /// (it was silently dropped pre-fix — the response message struct
    /// had no field for it) and surface it via `ChatMessage.extra` so
    /// render.rs flattens it back onto `message.reasoning_content` for
    /// the SDK client.
    #[test]
    fn non_streaming_reasoning_content_surfaces_via_extra() {
        let body = r#"{
            "id": "cmpl-r1",
            "object": "chat.completion",
            "model": "deepseek-reasoner",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The answer is 42.",
                    "reasoning_content": "Let me think step by step... 6 times 7 is 42."
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        }"#;
        let raw: OpenAiResponse = serde_json::from_str(body).unwrap();
        let out = response_into_chat_response(raw);
        assert_eq!(out.message.content, "The answer is 42.");
        let reasoning = out
            .message
            .extra
            .get("reasoning_content")
            .expect("reasoning_content must be captured into message.extra (#466)")
            .as_str()
            .unwrap();
        assert_eq!(reasoning, "Let me think step by step... 6 times 7 is 42.");
    }

    /// Issue #466 (audit LOW): a reasoning model that also calls a
    /// tool returns BOTH `tool_calls` and `reasoning_content` on the
    /// same message. Both must land in `extra` under their own keys
    /// (independent inserts, no collision).
    #[test]
    fn non_streaming_tool_calls_and_reasoning_content_coexist() {
        let body = r#"{
            "id": "cmpl-r1-tool",
            "object": "chat.completion",
            "model": "deepseek-reasoner",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "I should call the time tool.",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": "{}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 8, "total_tokens": 13}
        }"#;
        let raw: OpenAiResponse = serde_json::from_str(body).unwrap();
        let out = response_into_chat_response(raw);
        assert!(
            out.message.extra.contains_key("tool_calls"),
            "tool_calls must be preserved alongside reasoning_content",
        );
        assert_eq!(
            out.message.extra["reasoning_content"],
            "I should call the time tool.",
        );
    }

    /// Issue #466 companion: a response WITHOUT reasoning_content (the
    /// common non-reasoning model case) must not add a spurious empty
    /// field to extra.
    #[test]
    fn non_streaming_without_reasoning_content_adds_no_extra_field() {
        let body = r#"{
            "id": "cmpl-plain",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        }"#;
        let raw: OpenAiResponse = serde_json::from_str(body).unwrap();
        let out = response_into_chat_response(raw);
        assert!(
            !out.message.extra.contains_key("reasoning_content"),
            "no reasoning_content field when upstream didn't send one",
        );
    }

    /// PR #442 audit MEDIUM-1: a hybrid OpenAI-compat upstream that
    /// sends BOTH a nested `prompt_tokens_details.cached_tokens: 0`
    /// AND a non-zero top-level `prompt_cache_hit_tokens` must not let
    /// the zeroed nested detail mask the real native count. The
    /// normalized `cached_prompt_tokens` should take the native 700.
    #[test]
    fn zeroed_nested_cache_detail_does_not_mask_native_count() {
        let body = r#"{
            "id": "cmpl-hybrid",
            "object": "chat.completion",
            "model": "some-compat-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 10,
                "total_tokens": 1010,
                "prompt_tokens_details": {"cached_tokens": 0},
                "prompt_cache_hit_tokens": 700
            }
        }"#;
        let raw: OpenAiResponse = serde_json::from_str(body).unwrap();
        let out = response_into_chat_response(raw);
        assert_eq!(
            out.usage.cached_prompt_tokens, 700,
            "zeroed nested cached_tokens must not mask the non-zero native count",
        );
        assert_eq!(out.usage.prompt_cache_hit_tokens, Some(700));
    }

    #[test]
    fn missing_finish_reason_defaults_to_stop() {
        let body = r#"{
            "id": "cmpl-x",
            "model": "gpt-4o",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}]
        }"#;
        let raw: OpenAiResponse = serde_json::from_str(body).unwrap();
        let out = response_into_chat_response(raw);
        assert_eq!(out.finish_reason, FinishReason::Stop);
        assert_eq!(out.usage, UsageStats::default());
    }

    #[test]
    fn unknown_finish_reason_lands_in_other() {
        let body = r#"{
            "id": "cmpl-y",
            "model": "gpt-4o",
            "choices": [{
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "weird_reason"
            }]
        }"#;
        let raw: OpenAiResponse = serde_json::from_str(body).unwrap();
        let out = response_into_chat_response(raw);
        assert_eq!(
            out.finish_reason,
            FinishReason::Other("weird_reason".into())
        );
    }

    #[test]
    fn stream_chunk_maps_content_delta() {
        let body = r#"{
            "id": "cmpl-s",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "delta": {"content": "hel"},
                "finish_reason": null
            }]
        }"#;
        let raw: OpenAiStreamChunk = serde_json::from_str(body).unwrap();
        let chunk = stream_chunk_into_chat_chunk(raw);
        assert_eq!(chunk.delta.content.as_deref(), Some("hel"));
        assert!(chunk.finish_reason.is_none());
    }

    #[test]
    fn stream_chunk_terminal_emits_finish_reason() {
        let body = r#"{
            "id": "cmpl-s",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }"#;
        let raw: OpenAiStreamChunk = serde_json::from_str(body).unwrap();
        let chunk = stream_chunk_into_chat_chunk(raw);
        assert_eq!(chunk.finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn stream_chunk_with_tool_calls_propagates_to_delta() {
        let body = r#"{
            "id": "cmpl-t",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_abc",
                        "type": "function",
                        "function": { "name": "get_weather", "arguments": "" }
                    }]
                },
                "finish_reason": null
            }]
        }"#;
        let raw: OpenAiStreamChunk = serde_json::from_str(body).unwrap();
        let chunk = stream_chunk_into_chat_chunk(raw);
        let tc = chunk.delta.tool_calls.expect("tool_calls in delta");
        assert_eq!(tc.len(), 1);
        assert_eq!(tc[0]["id"], "call_abc");
        assert_eq!(tc[0]["function"]["name"], "get_weather");
    }

    #[test]
    fn build_request_sets_stream_flag_and_propagates_params() {
        let req = ChatFormat {
            model: "my-model".into(),
            messages: vec![ChatMessage::user("hi")],
            temperature: Some(0.4),
            top_p: Some(0.9),
            max_tokens: Some(64),
            stream: None,
            extra: serde_json::Map::new(),
        };
        let msgs = messages_from(&req);
        let built = build_request(&req, "gpt-4o", &msgs, true);
        let json = serde_json::to_value(&built).unwrap();
        assert_eq!(json["model"], "gpt-4o");
        // Temperature is f32 → serialises with f64-ish precision.
        let t = json["temperature"].as_f64().unwrap();
        assert!((t - 0.4).abs() < 1e-6, "temperature was {t}");
        let p = json["top_p"].as_f64().unwrap();
        assert!((p - 0.9).abs() < 1e-6, "top_p was {p}");
        assert_eq!(json["max_tokens"], 64);
        assert_eq!(json["stream"], true);
        assert_eq!(json["messages"][0]["role"], "user");
        assert_eq!(json["messages"][0]["content"], "hi");
    }

    #[test]
    fn extra_fields_flatten_into_request() {
        let mut extra = serde_json::Map::new();
        extra.insert("seed".into(), serde_json::json!(42));
        extra.insert("presence_penalty".into(), serde_json::json!(0.1));
        let req = ChatFormat {
            model: "m".into(),
            messages: vec![],
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: None,
            extra,
        };
        let msgs = messages_from(&req);
        let built = build_request(&req, "gpt-4o", &msgs, false);
        let json = serde_json::to_value(&built).unwrap();
        assert_eq!(json["seed"], 42);
        assert_eq!(json["presence_penalty"], 0.1);
    }

    /// Regression for issue #110: assistant tool_calls / refusal /
    /// audio fields on incoming messages must round-trip verbatim into
    /// the upstream OpenAI request, otherwise conversation-history
    /// replay reaches OpenAI without the previous tool round and the
    /// response is wrong.
    #[test]
    fn assistant_tool_calls_round_trip_into_upstream_request() {
        let history_json = r#"[
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": null,
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "w", "arguments": "{}"}}]},
            {"role": "tool", "content": "75F", "tool_call_id": "c1"},
            {"role": "user", "content": "tomorrow?"}
        ]"#;
        let messages: Vec<ChatMessage> = serde_json::from_str(history_json).unwrap();
        let req = ChatFormat {
            model: "my-model".into(),
            messages,
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: None,
            extra: serde_json::Map::new(),
        };
        let msgs = messages_from(&req);
        let built = build_request(&req, "gpt-4o", &msgs, false);
        let json = serde_json::to_value(&built).unwrap();
        let assistant = &json["messages"][1];
        // tool_calls preserved verbatim from input → upstream.
        let tool_calls = assistant["tool_calls"].as_array().expect(
            "assistant message to upstream OpenAI is missing tool_calls — \
             history replay would silently lose the previous tool round",
        );
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0]["id"], "c1");
        assert_eq!(tool_calls[0]["function"]["name"], "w");
        // tool message's tool_call_id reaches upstream.
        let tool_msg = &json["messages"][2];
        assert_eq!(tool_msg["tool_call_id"], "c1");
        assert_eq!(tool_msg["role"], "tool");
    }

    /// Issue #393: OpenAI's embeddings response carries `embedding` as
    /// either a JSON array (when the request set
    /// `encoding_format: "float"`) OR a base64 string (when the
    /// request set `encoding_format: "base64"`, which is the OpenAI
    /// SDK default). Pre-fix the deserializer rejected the string
    /// shape with `error decoding response body` → 502.
    #[test]
    fn embeddings_response_accepts_float_array() {
        let body = r#"{
            "object": "list",
            "model": "text-embedding-3-small",
            "data": [{
                "index": 0,
                "object": "embedding",
                "embedding": [0.1, -0.2, 0.3]
            }],
            "usage": { "prompt_tokens": 1, "total_tokens": 1 }
        }"#;
        let raw: OpenAiEmbedResponse =
            serde_json::from_str(body).expect("float-array embedding must deserialize");
        let resp = embed_response_into(raw);
        assert_eq!(resp.data.len(), 1);
        match &resp.data[0].embedding {
            EmbeddingVector::Float(v) => {
                assert_eq!(v.len(), 3);
                assert!((v[0] - 0.1).abs() < 1e-6);
            }
            EmbeddingVector::Base64(_) => {
                panic!("float request must deserialize as EmbeddingVector::Float")
            }
        }
    }

    #[test]
    fn embeddings_response_tolerates_missing_prompt_tokens_474() {
        // Jina's /v1/embeddings returns `usage.total_tokens` only,
        // omitting `prompt_tokens`. Pre-#474 the required field made the
        // whole body fail to deserialize -> `502 upstream_decode_error`.
        let body = r#"{
            "object": "list",
            "model": "jina-embeddings-v5-text-small",
            "data": [{
                "index": 0,
                "object": "embedding",
                "embedding": [0.1]
            }],
            "usage": { "total_tokens": 6 }
        }"#;
        let raw: OpenAiEmbedResponse =
            serde_json::from_str(body).expect("embeddings without prompt_tokens must deserialize");
        let resp = embed_response_into(raw);
        assert_eq!(resp.usage.prompt_tokens, 0);
        assert_eq!(resp.usage.total_tokens, 6);
    }

    #[test]
    fn chat_response_tolerates_partial_usage_474() {
        // Same bug class as embeddings: a chat usage object missing some
        // counters must not fail the response decode.
        let body = r#"{
            "id": "x",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop"
            }],
            "usage": { "total_tokens": 3 }
        }"#;
        let raw: OpenAiResponse =
            serde_json::from_str(body).expect("chat with partial usage must deserialize");
        let out = response_into_chat_response(raw);
        assert_eq!(out.usage.prompt_tokens, 0);
        assert_eq!(out.usage.completion_tokens, 0);
        assert_eq!(out.usage.total_tokens, 3);
    }

    #[test]
    fn embeddings_response_accepts_base64_string() {
        // OpenAI returns base64 as a quoted JSON string. Pre-#393
        // this was the bug — `Vec<f32>` deserializer rejected the
        // string shape with `error decoding response body`.
        let body = r#"{
            "object": "list",
            "model": "text-embedding-3-small",
            "data": [{
                "index": 0,
                "object": "embedding",
                "embedding": "ZAAAANSAAAA="
            }],
            "usage": { "prompt_tokens": 1, "total_tokens": 1 }
        }"#;
        let raw: OpenAiEmbedResponse =
            serde_json::from_str(body).expect("base64-string embedding must deserialize");
        let resp = embed_response_into(raw);
        assert_eq!(resp.data.len(), 1);
        match &resp.data[0].embedding {
            EmbeddingVector::Base64(s) => assert_eq!(s, "ZAAAANSAAAA="),
            EmbeddingVector::Float(_) => {
                panic!("base64 string must deserialize as EmbeddingVector::Base64")
            }
        }
    }

    #[test]
    fn embeddings_response_serializes_back_to_same_shape() {
        // The pass-through contract: the JSON serialization of the
        // gateway-internal EmbeddingObject MUST emit the same shape
        // the upstream returned. Float → array, base64 → string. An
        // SDK that asked for base64 sees a base64 string in the
        // gateway's response; an SDK that asked for float sees an
        // array. No format translation.
        let float_obj = EmbeddingObject {
            index: 0,
            object: "embedding".to_string(),
            embedding: EmbeddingVector::Float(vec![0.1, -0.2, 0.3]),
        };
        let s = serde_json::to_string(&float_obj).unwrap();
        assert!(
            s.contains(r#""embedding":[0.1,-0.2,0.3]"#),
            "float must serialize as JSON array; got {s}"
        );

        let b64_obj = EmbeddingObject {
            index: 0,
            object: "embedding".to_string(),
            embedding: EmbeddingVector::Base64("ZAAAANSAAAA=".to_string()),
        };
        let s = serde_json::to_string(&b64_obj).unwrap();
        assert!(
            s.contains(r#""embedding":"ZAAAANSAAAA=""#),
            "base64 must serialize as JSON string; got {s}"
        );
    }
}
