//! Render the gateway's normalised `ChatResponse` / `ChatChunk` into the
//! OpenAI wire shape that clients expect on `/v1/chat/completions`.
//!
//! The structure is intentionally independent from the provider crates'
//! upstream types — those describe what we *received*, while these
//! describe what we *emit*. Keeping them separate means a client-facing
//! schema change doesn't ripple into every provider adapter.

use aisix_gateway::{ChatChunk, ChatResponse, FinishReason, Role};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct ChatCompletion {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<NonStreamChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct NonStreamChoice {
    pub index: u32,
    pub message: RenderedMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct RenderedMessage {
    pub role: &'static str,
    pub content: String,
    /// Forward-compatible bag for OpenAI message-level fields the
    /// gateway doesn't model directly on `ChatMessage` (e.g.
    /// `tool_calls` for cross-provider tool-use translation,
    /// `refusal` for OpenAI's safety-classifier output, `audio` for
    /// realtime/4o audio models). Bridges populate this on the way
    /// back from the upstream; serde flatten emits each entry as a
    /// top-level field on the wire so OpenAI SDK clients see the
    /// standard shape.
    #[serde(flatten, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Serialize, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub struct StreamChoice {
    pub index: u32,
    pub delta: RenderedDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Default)]
pub struct RenderedDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    /// Reasoning text (DeepSeek-canonical `delta.reasoning_content`).
    /// Surfaced when the bridge applied
    /// [`response.reasoning_field`](aisix_core::ResponseOverrides::reasoning_field)
    /// — issue #302 §5.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

pub fn render_response(created_unix_ts: i64, resp: ChatResponse) -> ChatCompletion {
    ChatCompletion {
        id: resp.id,
        object: "chat.completion",
        created: created_unix_ts,
        model: resp.model,
        choices: vec![NonStreamChoice {
            index: 0,
            message: RenderedMessage {
                role: role_to_str(resp.message.role),
                content: resp.message.content,
                // Forward bridge-populated fields (`tool_calls`,
                // `refusal`, `audio`, …) through to the caller.
                extra: resp.message.extra,
            },
            finish_reason: finish_to_str(&resp.finish_reason).to_string(),
        }],
        usage: Usage {
            prompt_tokens: resp.usage.prompt_tokens,
            completion_tokens: resp.usage.completion_tokens,
            total_tokens: resp.usage.total_tokens,
        },
    }
}

pub fn render_chunk(created_unix_ts: i64, chunk: ChatChunk) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: chunk.id,
        object: "chat.completion.chunk",
        created: created_unix_ts,
        model: chunk.model,
        choices: vec![StreamChoice {
            index: 0,
            delta: RenderedDelta {
                role: chunk.delta.role.map(role_to_str),
                content: chunk.delta.content,
                tool_calls: chunk.delta.tool_calls,
                reasoning_content: chunk.delta.reasoning_content,
            },
            finish_reason: chunk
                .finish_reason
                .as_ref()
                .map(|f| finish_to_str(f).to_string()),
        }],
        usage: chunk.usage.map(|u| Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        }),
    }
}

/// Inject the `x-ratelimit-*` response headers that OpenAI SDK clients
/// read for back-pressure / progress reporting.
///
/// Only headers with a configured limit (non-`None`) are injected;
/// endpoints or keys that have no limit set don't emit anything — the
/// client should not assume absence means unlimited when it sees nothing.
pub fn inject_ratelimit_headers(
    response: &mut axum::response::Response,
    status: &aisix_ratelimit::RateLimitStatus,
) {
    use axum::http::HeaderValue;

    let headers = response.headers_mut();

    macro_rules! set_header {
        ($name:expr, $value:expr) => {
            if let Ok(v) = HeaderValue::try_from($value.to_string()) {
                headers.insert($name, v);
            }
        };
    }

    if let Some(lim) = status.rpm_limit {
        set_header!("x-ratelimit-limit-requests", lim);
        set_header!(
            "x-ratelimit-remaining-requests",
            status.rpm_remaining().unwrap_or(0)
        );
        set_header!(
            "x-ratelimit-reset-requests",
            format!("{}s", status.rpm_reset_secs)
        );
    }

    if let Some(lim) = status.tpm_limit {
        set_header!("x-ratelimit-limit-tokens", lim);
        set_header!(
            "x-ratelimit-remaining-tokens",
            status.tpm_remaining().unwrap_or(0)
        );
        set_header!(
            "x-ratelimit-reset-tokens",
            format!("{}s", status.tpm_reset_secs)
        );
    }

    if let Some(lim) = status.concurrency_limit {
        set_header!("x-ratelimit-limit-concurrent", lim);
        set_header!(
            "x-ratelimit-remaining-concurrent",
            lim.saturating_sub(status.in_flight)
        );
    }
}

fn role_to_str(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

fn finish_to_str(f: &FinishReason) -> &str {
    match f {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::ContentFilter => "content_filter",
        FinishReason::ToolCalls => "tool_calls",
        FinishReason::Other(s) => s.as_str(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_gateway::{ChatMessage, UsageStats};

    #[test]
    fn render_response_matches_openai_shape() {
        let r = ChatResponse {
            id: "cmpl-1".into(),
            model: "m".into(),
            message: ChatMessage::assistant("hello"),
            finish_reason: FinishReason::Stop,
            usage: UsageStats::new(3, 2),
        };
        let out = render_response(42, r);
        let json = serde_json::to_value(&out).unwrap();
        assert_eq!(json["object"], "chat.completion");
        assert_eq!(json["created"], 42);
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        assert_eq!(json["choices"][0]["message"]["role"], "assistant");
        assert_eq!(json["choices"][0]["message"]["content"], "hello");
        assert_eq!(json["usage"]["total_tokens"], 5);
    }

    #[test]
    fn render_chunk_omits_finish_reason_when_absent() {
        let chunk = ChatChunk {
            id: "c".into(),
            model: "m".into(),
            delta: aisix_gateway::ChatDelta {
                role: None,
                content: Some("hi".into()),
                tool_calls: None,
                reasoning_content: None,
            },
            finish_reason: None,
            usage: None,
        };
        let out = render_chunk(1, chunk);
        let json = serde_json::to_value(&out).unwrap();
        assert_eq!(json["object"], "chat.completion.chunk");
        assert_eq!(json["choices"][0]["delta"]["content"], "hi");
        // finish_reason / usage must be absent (not null).
        assert!(json["choices"][0].get("finish_reason").is_none());
        assert!(json.get("usage").is_none());
    }

    #[test]
    fn finish_reason_other_serialises_verbatim() {
        let r = ChatResponse {
            id: "cmpl".into(),
            model: "m".into(),
            message: ChatMessage::assistant(""),
            finish_reason: FinishReason::Other("weird".into()),
            usage: UsageStats::default(),
        };
        let out = render_response(0, r);
        let json = serde_json::to_value(&out).unwrap();
        assert_eq!(json["choices"][0]["finish_reason"], "weird");
    }
}
