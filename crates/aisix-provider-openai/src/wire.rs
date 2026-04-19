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

use aisix_gateway::{
    ChatChunk, ChatDelta, ChatFormat, ChatMessage, ChatResponse, EmbeddingObject, EmbeddingRequest,
    EmbeddingResponse, EmbeddingUsage, FinishReason, Role, UsageStats,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize)]
pub(crate) struct OpenAiRequest<'a> {
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
pub(crate) struct OpenAiMessage<'a> {
    pub role: &'a str,
    pub content: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<&'a str>,
}

/// Build the upstream request from the gateway's normalised format.
///
/// `upstream_model` is the part after the `<provider>/` prefix from the
/// Model entity (e.g. `"gpt-4o"`, not `"openai/gpt-4o"`).
pub(crate) fn build_request<'a>(
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

pub(crate) fn messages_from(req: &ChatFormat) -> Vec<OpenAiMessage<'_>> {
    req.messages
        .iter()
        .map(|m| OpenAiMessage {
            role: role_str(m.role),
            content: &m.content,
            name: m.name.as_deref(),
            tool_call_id: m.tool_call_id.as_deref(),
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
pub(crate) struct OpenAiResponse {
    pub id: String,
    pub model: String,
    pub choices: Vec<OpenAiChoice>,
    #[serde(default)]
    pub usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiChoice {
    pub message: OpenAiResponseMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiResponseMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

pub(crate) fn response_into_chat_response(mut raw: OpenAiResponse) -> ChatResponse {
    let first = raw.choices.drain(..).next();
    let (message, finish) = match first {
        Some(c) => (
            ChatMessage {
                role: role_from_str(&c.message.role),
                content: c.message.content.unwrap_or_default(),
                name: None,
                tool_call_id: None,
            },
            finish_reason(c.finish_reason.as_deref()),
        ),
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
pub(crate) struct OpenAiStreamChunk {
    pub id: String,
    pub model: String,
    pub choices: Vec<OpenAiStreamChoice>,
    #[serde(default)]
    pub usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiStreamChoice {
    pub delta: OpenAiStreamDelta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiStreamDelta {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
}

pub(crate) fn stream_chunk_into_chat_chunk(mut raw: OpenAiStreamChunk) -> ChatChunk {
    let first = raw.choices.drain(..).next();
    let (delta, finish) = match first {
        Some(c) => (
            ChatDelta {
                role: c.delta.role.as_deref().map(role_from_str),
                content: c.delta.content,
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

/// Request body forwarded to OpenAI `/v1/embeddings`.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct OpenAiEmbedRequest<'a> {
    pub model: &'a str,
    pub input: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
}

/// One embedding object from OpenAI's response.
#[derive(Debug, Deserialize)]
pub(crate) struct OpenAiEmbeddingObject {
    pub index: u32,
    pub object: String,
    pub embedding: Vec<f32>,
}

/// Usage block from OpenAI's embeddings response.
#[derive(Debug, Default, Deserialize)]
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
    OpenAiEmbedRequest {
        model: upstream_model,
        input: &req.input,
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
}
