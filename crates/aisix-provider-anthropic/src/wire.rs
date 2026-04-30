//! Anthropic `/v1/messages` wire shapes.
//!
//! Reference: <https://docs.anthropic.com/en/api/messages>
//!
//! Key differences from OpenAI that this module handles:
//!
//! - System prompt is a top-level `system` field, not a message with
//!   `role: "system"` — we collapse all leading system messages into one
//!   string and forward it there.
//! - Only `user` and `assistant` roles on the wire. `tool` messages from
//!   ChatFormat are rejected at the bridge boundary rather than being
//!   silently re-classified.
//! - Content is an array of blocks — we emit a single `{"type":"text",…}`
//!   block per message and read the concatenation of text blocks on the
//!   way back.
//! - `max_tokens` is required by Anthropic. We default to a safe ceiling
//!   when the client didn't set one, but log the fallback so operators
//!   can tune the default if desired.
//! - Streaming events are typed (`message_start`, `content_block_delta`,
//!   …). We only emit a `ChatChunk` when a delta carries content or a
//!   stop reason — other events just advance internal state.

use aisix_gateway::{
    ChatChunk, ChatDelta, ChatFormat, ChatMessage, ChatResponse, FinishReason, Role, UsageStats,
};
use serde::{Deserialize, Serialize};

/// Anthropic requires a non-zero `max_tokens`. Clients that omit it get
/// this ceiling — generous enough to cover normal completions, conservative
/// enough that a runaway prompt doesn't burn tokens silently.
pub(crate) const DEFAULT_MAX_TOKENS: u32 = 4096;

#[derive(Debug, Clone, Serialize)]
pub(crate) struct AnthropicRequest<'a> {
    pub model: &'a str,
    pub messages: Vec<AnthropicMessage<'a>>,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    pub stream: bool,
    #[serde(flatten)]
    pub extra: &'a serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct AnthropicMessage<'a> {
    pub role: &'a str,
    pub content: Vec<AnthropicTextBlock<'a>>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct AnthropicTextBlock<'a> {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub text: &'a str,
}

#[derive(Debug, thiserror::Error)]
pub enum TranslateError {
    #[error("anthropic does not support role {role:?}")]
    UnsupportedRole { role: &'static str },
}

/// Split the gateway's flat ChatFormat into Anthropic's (system, messages)
/// shape. Consecutive system messages at the head are concatenated with
/// a blank line, matching how users typically compose multi-paragraph
/// system prompts in the OpenAI format.
pub(crate) fn split_system<'a>(
    req: &'a ChatFormat,
) -> Result<(Option<String>, Vec<AnthropicMessage<'a>>), TranslateError> {
    let mut system_parts: Vec<&'a str> = Vec::new();
    let mut messages: Vec<AnthropicMessage<'a>> = Vec::new();
    let mut seen_non_system = false;

    for m in &req.messages {
        match m.role {
            Role::System => {
                if seen_non_system {
                    // System messages interleaved with user/assistant
                    // turns don't map cleanly; append as a user turn to
                    // preserve semantics without silently dropping them.
                    messages.push(AnthropicMessage {
                        role: "user",
                        content: vec![AnthropicTextBlock {
                            kind: "text",
                            text: &m.content,
                        }],
                    });
                } else {
                    system_parts.push(&m.content);
                }
            }
            Role::User => {
                seen_non_system = true;
                messages.push(AnthropicMessage {
                    role: "user",
                    content: vec![AnthropicTextBlock {
                        kind: "text",
                        text: &m.content,
                    }],
                });
            }
            Role::Assistant => {
                seen_non_system = true;
                messages.push(AnthropicMessage {
                    role: "assistant",
                    content: vec![AnthropicTextBlock {
                        kind: "text",
                        text: &m.content,
                    }],
                });
            }
            Role::Tool => return Err(TranslateError::UnsupportedRole { role: "tool" }),
        }
    }

    let system = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n\n"))
    };
    Ok((system, messages))
}

pub(crate) fn build_request<'a>(
    req: &'a ChatFormat,
    upstream_model: &'a str,
    system: Option<String>,
    messages: Vec<AnthropicMessage<'a>>,
    stream: bool,
) -> AnthropicRequest<'a> {
    AnthropicRequest {
        model: upstream_model,
        messages,
        max_tokens: req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        system,
        temperature: req.temperature,
        top_p: req.top_p,
        stream,
        extra: &req.extra,
    }
}

/// Non-streaming response shape from `/v1/messages`.
#[derive(Debug, Deserialize)]
pub(crate) struct AnthropicResponse {
    pub id: String,
    pub model: String,
    #[serde(default)]
    pub content: Vec<AnthropicResponseBlock>,
    #[serde(default)]
    pub stop_reason: Option<String>,
    #[serde(default)]
    pub usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum AnthropicResponseBlock {
    #[serde(rename = "text")]
    Text { text: String },
    /// Tool-use and other block types are ignored for PR #8 — a later
    /// tools PR will surface them as typed `ChatChunk.tool_calls`.
    #[serde(other)]
    Other,
}

#[derive(Debug, Default, Deserialize)]
pub(crate) struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    /// Tokens written to the prompt cache (1.25× input rate). Optional
    /// — present only on requests with cache_control segments.
    #[serde(default)]
    pub cache_creation_input_tokens: u32,
    /// Tokens served from the prompt cache (0.10× input rate).
    #[serde(default)]
    pub cache_read_input_tokens: u32,
}

pub(crate) fn response_into_chat_response(raw: AnthropicResponse) -> ChatResponse {
    let text = raw
        .content
        .iter()
        .filter_map(|b| match b {
            AnthropicResponseBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("");

    let usage = raw
        .usage
        .map(|u| UsageStats {
            prompt_tokens: u.input_tokens,
            completion_tokens: u.output_tokens,
            total_tokens: u.input_tokens.saturating_add(u.output_tokens),
            cache_creation_tokens: u.cache_creation_input_tokens,
            cache_read_tokens: u.cache_read_input_tokens,
            // Anthropic doesn't use OpenAI's cached-prompt-tokens or
            // reasoning-tokens taxonomy; leave at 0.
            cached_prompt_tokens: 0,
            reasoning_tokens: 0,
        })
        .unwrap_or_default();

    ChatResponse {
        id: raw.id,
        model: raw.model,
        message: ChatMessage {
            role: Role::Assistant,
            content: text,
            name: None,
            tool_call_id: None,
        },
        finish_reason: map_stop_reason(raw.stop_reason.as_deref()),
        usage,
    }
}

fn map_stop_reason(raw: Option<&str>) -> FinishReason {
    match raw {
        Some("end_turn") | Some("stop_sequence") | None => FinishReason::Stop,
        Some("max_tokens") => FinishReason::Length,
        Some("tool_use") => FinishReason::ToolCalls,
        Some(other) => FinishReason::Other(other.to_string()),
    }
}

/// Streaming events from Anthropic. Only variants that can yield user-
/// visible output or terminate the stream are modeled here; the rest are
/// quietly dropped by the Bridge.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum AnthropicStreamEvent {
    #[serde(rename = "message_start")]
    MessageStart {
        message: AnthropicStreamStartMessage,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { delta: AnthropicStreamDelta },
    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: AnthropicStreamMessageDelta,
        #[serde(default)]
        usage: Option<AnthropicStreamUsage>,
    },
    #[serde(rename = "message_stop")]
    MessageStop,
    /// Catch-all for content_block_start / content_block_stop / ping /
    /// unknown event types — we don't need their state for chunk emission.
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
pub(crate) struct AnthropicStreamStartMessage {
    pub id: String,
    pub model: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum AnthropicStreamDelta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
pub(crate) struct AnthropicStreamMessageDelta {
    #[serde(default)]
    pub stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct AnthropicStreamUsage {
    #[serde(default)]
    pub output_tokens: Option<u32>,
}

/// Rolling state the Bridge carries across a stream so chunks can be
/// tagged with the message id/model even though only the first event
/// carries them.
#[derive(Debug, Default)]
pub(crate) struct StreamState {
    pub id: String,
    pub model: String,
}

impl StreamState {
    pub fn update(&mut self, event: &AnthropicStreamEvent) {
        if let AnthropicStreamEvent::MessageStart { message } = event {
            self.id = message.id.clone();
            self.model = message.model.clone();
        }
    }

    /// Translate one event into an optional chunk to yield upstream.
    pub fn to_chunk(&self, event: &AnthropicStreamEvent) -> Option<ChatChunk> {
        match event {
            AnthropicStreamEvent::ContentBlockDelta {
                delta: AnthropicStreamDelta::TextDelta { text },
            } => Some(ChatChunk {
                id: self.id.clone(),
                model: self.model.clone(),
                delta: ChatDelta {
                    role: None,
                    content: Some(text.clone()),
                },
                finish_reason: None,
                usage: None,
            }),
            AnthropicStreamEvent::MessageDelta { delta, usage } => {
                let finish = delta
                    .stop_reason
                    .as_deref()
                    .map(|r| map_stop_reason(Some(r)));
                let usage = usage
                    .as_ref()
                    .and_then(|u| u.output_tokens.map(|n| UsageStats::new(0, n)));
                if finish.is_none() && usage.is_none() {
                    return None;
                }
                Some(ChatChunk {
                    id: self.id.clone(),
                    model: self.model.clone(),
                    delta: ChatDelta::default(),
                    finish_reason: finish,
                    usage,
                })
            }
            _ => None,
        }
    }

    pub fn is_terminal(event: &AnthropicStreamEvent) -> bool {
        matches!(event, AnthropicStreamEvent::MessageStop)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_system_merges_leading_system_messages() {
        let req = ChatFormat::new(
            "claude",
            vec![
                ChatMessage::system("you are helpful"),
                ChatMessage::system("respond concisely"),
                ChatMessage::user("hi"),
            ],
        );
        let (system, msgs) = split_system(&req).unwrap();
        assert_eq!(
            system.as_deref(),
            Some("you are helpful\n\nrespond concisely")
        );
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, "user");
    }

    #[test]
    fn split_system_mid_conversation_becomes_user_turn() {
        let req = ChatFormat::new(
            "claude",
            vec![
                ChatMessage::user("hi"),
                ChatMessage::system("forget everything"),
                ChatMessage::assistant("ok"),
            ],
        );
        let (system, msgs) = split_system(&req).unwrap();
        assert!(system.is_none());
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[1].role, "user"); // former system message
    }

    #[test]
    fn split_system_rejects_tool_role() {
        let req = ChatFormat::new(
            "claude",
            vec![ChatMessage {
                role: Role::Tool,
                content: "x".into(),
                name: None,
                tool_call_id: None,
            }],
        );
        assert!(matches!(
            split_system(&req),
            Err(TranslateError::UnsupportedRole { role: "tool" })
        ));
    }

    #[test]
    fn build_request_applies_default_max_tokens_when_unset() {
        let req = ChatFormat::new("claude", vec![ChatMessage::user("hi")]);
        let (_system, messages) = split_system(&req).unwrap();
        let built = build_request(&req, "claude-sonnet-4-5", None, messages, false);
        assert_eq!(built.max_tokens, DEFAULT_MAX_TOKENS);

        let req = ChatFormat {
            max_tokens: Some(256),
            ..ChatFormat::new("claude", vec![ChatMessage::user("hi")])
        };
        let (_system, messages) = split_system(&req).unwrap();
        let built = build_request(&req, "claude-sonnet-4-5", None, messages, false);
        assert_eq!(built.max_tokens, 256);
    }

    #[test]
    fn non_streaming_response_concatenates_text_blocks() {
        let body = r#"{
            "id": "msg_01A",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [
                {"type": "text", "text": "hel"},
                {"type": "text", "text": "lo"}
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 2}
        }"#;
        let raw: AnthropicResponse = serde_json::from_str(body).unwrap();
        let out = response_into_chat_response(raw);
        assert_eq!(out.id, "msg_01A");
        assert_eq!(out.message.content, "hello");
        assert_eq!(out.finish_reason, FinishReason::Stop);
        assert_eq!(out.usage.total_tokens, 5);
    }

    #[test]
    fn cache_creation_and_read_counters_populate_when_present() {
        // Verified shape from
        // https://docs.anthropic.com/en/api/messages (usage object
        // with cache_creation_input_tokens + cache_read_input_tokens).
        let body = r#"{
            "id": "msg_cache_01",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 4,
                "cache_creation_input_tokens": 200,
                "cache_read_input_tokens": 800
            }
        }"#;
        let raw: AnthropicResponse = serde_json::from_str(body).unwrap();
        let out = response_into_chat_response(raw);
        assert_eq!(out.usage.prompt_tokens, 10);
        assert_eq!(out.usage.completion_tokens, 4);
        assert_eq!(out.usage.cache_creation_tokens, 200);
        assert_eq!(out.usage.cache_read_tokens, 800);
        // Anthropic doesn't use OpenAI's cached_prompt / reasoning
        // taxonomy — these stay 0.
        assert_eq!(out.usage.cached_prompt_tokens, 0);
        assert_eq!(out.usage.reasoning_tokens, 0);
    }

    #[test]
    fn stop_reason_mappings_match_spec() {
        assert_eq!(map_stop_reason(Some("end_turn")), FinishReason::Stop);
        assert_eq!(map_stop_reason(Some("max_tokens")), FinishReason::Length);
        assert_eq!(map_stop_reason(Some("tool_use")), FinishReason::ToolCalls);
        assert_eq!(
            map_stop_reason(Some("exotic_reason")),
            FinishReason::Other("exotic_reason".into())
        );
        assert_eq!(map_stop_reason(None), FinishReason::Stop);
    }

    #[test]
    fn content_blocks_other_than_text_are_skipped() {
        // Tool-use blocks on a completion we're treating as plain text
        // should not break parsing; they're simply not surfaced yet.
        let body = r#"{
            "id": "msg_02",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [
                {"type": "tool_use", "id": "tu_1", "name": "search", "input": {}},
                {"type": "text", "text": "done"}
            ]
        }"#;
        let raw: AnthropicResponse = serde_json::from_str(body).unwrap();
        let out = response_into_chat_response(raw);
        assert_eq!(out.message.content, "done");
    }

    #[test]
    fn stream_events_deserialise_into_typed_variants() {
        let start: AnthropicStreamEvent = serde_json::from_str(
            r#"{"type":"message_start","message":{"id":"msg_1","model":"claude","type":"message","role":"assistant","content":[],"stop_reason":null,"usage":{"input_tokens":1}}}"#,
        )
        .unwrap();
        assert!(matches!(start, AnthropicStreamEvent::MessageStart { .. }));

        let delta: AnthropicStreamEvent = serde_json::from_str(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#,
        )
        .unwrap();
        assert!(matches!(
            delta,
            AnthropicStreamEvent::ContentBlockDelta { .. }
        ));

        let msg_delta: AnthropicStreamEvent = serde_json::from_str(
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}"#,
        )
        .unwrap();
        assert!(matches!(
            msg_delta,
            AnthropicStreamEvent::MessageDelta { .. }
        ));

        let ping: AnthropicStreamEvent = serde_json::from_str(r#"{"type":"ping"}"#).unwrap();
        assert!(matches!(ping, AnthropicStreamEvent::Other));
    }

    #[test]
    fn stream_state_tracks_id_and_emits_text_delta() {
        let mut state = StreamState::default();
        let start: AnthropicStreamEvent = serde_json::from_str(
            r#"{"type":"message_start","message":{"id":"msg_9","model":"claude-sonnet-4-5","type":"message","role":"assistant","content":[],"stop_reason":null,"usage":{"input_tokens":1}}}"#,
        )
        .unwrap();
        state.update(&start);
        assert_eq!(state.id, "msg_9");
        assert!(state.to_chunk(&start).is_none());

        let delta: AnthropicStreamEvent = serde_json::from_str(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#,
        )
        .unwrap();
        let chunk = state.to_chunk(&delta).unwrap();
        assert_eq!(chunk.id, "msg_9");
        assert_eq!(chunk.delta.content.as_deref(), Some("hi"));
    }

    #[test]
    fn stream_state_emits_finish_on_message_delta() {
        let state = StreamState {
            id: "msg".into(),
            model: "claude".into(),
        };
        let end: AnthropicStreamEvent = serde_json::from_str(
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}"#,
        )
        .unwrap();
        let chunk = state.to_chunk(&end).unwrap();
        assert_eq!(chunk.finish_reason, Some(FinishReason::Stop));
        assert_eq!(chunk.usage.unwrap().completion_tokens, 3);
    }
}
