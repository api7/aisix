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
