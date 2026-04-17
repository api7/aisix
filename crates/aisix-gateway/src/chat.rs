//! Provider-agnostic chat request / response types.
//!
//! The gateway normalises every client request into a [`ChatFormat`] and
//! hands it to whichever [`crate::bridge::Bridge`] implementation matches
//! the target provider. The response shape (either a full [`ChatResponse`]
//! or a stream of [`ChatChunk`]s) is symmetric: providers emit the normalised
//! form and the proxy layer re-encodes into whatever the client expects
//! (defaulting to OpenAI-compatible JSON).
//!
//! These types are deliberately a superset of OpenAI's shape because that
//! is the most permissive of the four providers we're targeting; fields
//! that don't map cleanly to a specific upstream become the provider's
//! responsibility to drop or translate.

use serde::{Deserialize, Serialize};

/// Role of a chat message. Matches OpenAI's taxonomy; providers that only
/// support system/user/assistant are expected to reject `Tool` at their
/// own boundary rather than silently collapsing roles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
            name: None,
            tool_call_id: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            name: None,
            tool_call_id: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            name: None,
            tool_call_id: None,
        }
    }
}

/// Normalised chat completion request.
///
/// `model` is the **public-facing** name from the Admin API (e.g.
/// `"my-gpt4"`), not the upstream model id. The gateway resolves this to
/// an `aisix_core::Model` before calling a Bridge; the Bridge receives
/// only the resolved [`crate::bridge::BridgeContext`] and translates the
/// `ChatFormat` to the provider's own request shape.
///
/// Unknown top-level fields are **not** rejected — OpenAI's API adds
/// params regularly (e.g. `top_k`, `seed`, `presence_penalty`), and each
/// Bridge is responsible for forwarding or ignoring them. Extras land in
/// the `extra` map via `#[serde(flatten)]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatFormat {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Free-form extra fields the client sent. We don't strip unknown
    /// params at the gateway — each Bridge decides what to forward.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty", flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl ChatFormat {
    pub fn new(model: impl Into<String>, messages: Vec<ChatMessage>) -> Self {
        Self {
            model: model.into(),
            messages,
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: None,
            extra: serde_json::Map::new(),
        }
    }

    pub fn is_streaming(&self) -> bool {
        self.stream.unwrap_or(false)
    }
}

/// Why a completion finished. Unknown upstream reasons collapse to
/// [`FinishReason::Other`] carrying the original string.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
    ToolCalls,
    Other(String),
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UsageStats {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl UsageStats {
    pub fn new(prompt: u32, completion: u32) -> Self {
        Self {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt.saturating_add(completion),
        }
    }
}

/// Full (non-streaming) chat response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatResponse {
    pub id: String,
    pub model: String,
    pub message: ChatMessage,
    pub finish_reason: FinishReason,
    pub usage: UsageStats,
}

/// One streamed delta event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatChunk {
    pub id: String,
    pub model: String,
    pub delta: ChatDelta,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<UsageStats>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatDelta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_format_round_trips_through_json() {
        let f = ChatFormat {
            model: "my-gpt4".into(),
            messages: vec![
                ChatMessage::system("you are helpful"),
                ChatMessage::user("hi"),
            ],
            temperature: Some(0.2),
            top_p: None,
            max_tokens: Some(100),
            stream: Some(true),
            extra: serde_json::Map::new(),
        };

        let json = serde_json::to_string(&f).unwrap();
        let back: ChatFormat = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model, "my-gpt4");
        assert_eq!(back.messages.len(), 2);
        assert_eq!(back.temperature, Some(0.2));
        assert!(back.is_streaming());
    }

    #[test]
    fn extras_capture_unknown_top_level_fields() {
        // `top_k` isn't a known field — it lands in `extra` so the Bridge
        // can decide whether to forward it to the upstream provider.
        let json = r#"{
            "model": "my-gpt4",
            "messages": [],
            "top_k": 40
        }"#;
        let f: ChatFormat = serde_json::from_str(json).unwrap();
        assert_eq!(f.extra.get("top_k").and_then(|v| v.as_u64()), Some(40));
    }

    #[test]
    fn is_streaming_defaults_to_false_when_unset() {
        let f = ChatFormat::new("m", vec![]);
        assert!(!f.is_streaming());
    }

    #[test]
    fn finish_reason_known_variants_are_snake_case() {
        let stop: FinishReason = serde_json::from_str(r#""stop""#).unwrap();
        let content_filter: FinishReason = serde_json::from_str(r#""content_filter""#).unwrap();
        assert_eq!(stop, FinishReason::Stop);
        assert_eq!(content_filter, FinishReason::ContentFilter);
    }

    #[test]
    fn usage_stats_saturates_total() {
        let u = UsageStats::new(u32::MAX, 10);
        assert_eq!(u.total_tokens, u32::MAX);
    }

    #[test]
    fn message_constructors_set_role() {
        assert_eq!(ChatMessage::system("x").role, Role::System);
        assert_eq!(ChatMessage::user("x").role, Role::User);
        assert_eq!(ChatMessage::assistant("x").role, Role::Assistant);
    }

    #[test]
    fn chat_chunk_omits_optional_fields_on_wire() {
        let chunk = ChatChunk {
            id: "cmpl-1".into(),
            model: "m".into(),
            delta: ChatDelta {
                role: None,
                content: Some("hello".into()),
            },
            finish_reason: None,
            usage: None,
        };
        let json = serde_json::to_string(&chunk).unwrap();
        assert!(!json.contains("\"finish_reason\""));
        assert!(!json.contains("\"usage\""));
        assert!(json.contains("\"content\":\"hello\""));
    }
}
