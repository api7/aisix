//! Anthropic Messages API format types.
//!
//! These types represent Anthropic's native `/v1/messages` API format.
//! They are used both for:
//! - **Inbound** requests to the gateway's `/v1/messages` endpoint (as a `ChatFormat`)
//! - **Native** Anthropic provider communication (via `NativeAnthropicMessagesSupport`)

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ── Request types ──

/// Anthropic Messages API request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessagesRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    pub max_tokens: u32,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemPrompt>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<AnthropicMetadata>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<AnthropicTool>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<AnthropicToolChoice>,
}

/// System prompt — either a plain string or structured content blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    Text(String),
    Blocks(Vec<SystemBlock>),
}

/// A system content block (text with optional cache_control).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SystemBlock {
    pub r#type: String,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

/// Cache control directive for prompt caching.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CacheControl {
    pub r#type: String,
}

/// Anthropic request metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}

/// A message in the Anthropic format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: AnthropicContent,
}

/// Message content — either a plain string or structured content blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

/// A content block in an Anthropic message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },

    #[serde(rename = "image")]
    Image { source: ImageSource },

    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },

    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<AnthropicContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

/// Image source for image content blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSource {
    pub r#type: String,
    pub media_type: String,
    pub data: String,
}

/// An Anthropic tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: Value,
}

/// Tool choice specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicToolChoice {
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "any")]
    Any,
    #[serde(rename = "tool")]
    Tool { name: String },
}

// ── Response types ──

/// Anthropic Messages API response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessagesResponse {
    pub id: String,
    pub r#type: String,
    pub role: String,
    pub content: Vec<AnthropicContentBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

/// Anthropic usage metrics.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(default)]
    pub cache_creation_input_tokens: u32,
    #[serde(default)]
    pub cache_read_input_tokens: u32,
}

// ── Streaming event types ──

/// Anthropic SSE stream event.
///
/// The Anthropic streaming protocol uses typed SSE events:
/// `message_start` → (`content_block_start` → `content_block_delta`* → `content_block_stop`)*
/// → `message_delta` → `message_stop`
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicStreamEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: MessageStartPayload },

    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: usize,
        content_block: AnthropicContentBlock,
    },

    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { index: usize, delta: ContentDelta },

    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: usize },

    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: MessageDelta,
        usage: DeltaUsage,
    },

    #[serde(rename = "message_stop")]
    MessageStop,

    #[serde(rename = "ping")]
    Ping,

    #[serde(rename = "error")]
    Error { error: AnthropicError },
}

impl AnthropicStreamEvent {
    /// Returns the SSE event type string for this event.
    pub fn event_type(&self) -> &'static str {
        match self {
            Self::MessageStart { .. } => "message_start",
            Self::ContentBlockStart { .. } => "content_block_start",
            Self::ContentBlockDelta { .. } => "content_block_delta",
            Self::ContentBlockStop { .. } => "content_block_stop",
            Self::MessageDelta { .. } => "message_delta",
            Self::MessageStop => "message_stop",
            Self::Ping => "ping",
            Self::Error { .. } => "error",
        }
    }
}

/// Payload of a `message_start` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageStartPayload {
    pub id: String,
    pub r#type: String,
    pub role: String,
    pub model: String,
    pub usage: InputUsage,
}

/// Input usage reported at message start.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InputUsage {
    pub input_tokens: u32,
}

/// Content delta within a `content_block_delta` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentDelta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },

    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
}

/// Delta in a `message_delta` event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageDelta {
    pub stop_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
}

/// Usage reported in `message_delta`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeltaUsage {
    pub output_tokens: u32,
    #[serde(default)]
    pub input_tokens: u32,
}

/// Anthropic API error body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicError {
    pub r#type: String,
    pub message: String,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn request_round_trip() {
        let json = json!({
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        });
        let req: AnthropicMessagesRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.model, "claude-3-5-sonnet-20241022");
        assert_eq!(req.max_tokens, 1024);
        assert!(req.system.is_none());
        assert!(req.tools.is_none());

        let serialized = serde_json::to_value(&req).unwrap();
        assert_eq!(serialized["model"], "claude-3-5-sonnet-20241022");
    }

    #[test]
    fn request_with_system_string() {
        let json = json!({
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Hi"}]
        });
        let req: AnthropicMessagesRequest = serde_json::from_value(json).unwrap();
        assert!(matches!(req.system, Some(SystemPrompt::Text(ref s)) if s == "You are helpful."));
    }

    #[test]
    fn request_with_system_blocks() {
        let json = json!({
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "system": [
                {"type": "text", "text": "You are helpful.", "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [{"role": "user", "content": "Hi"}]
        });
        let req: AnthropicMessagesRequest = serde_json::from_value(json).unwrap();
        if let Some(SystemPrompt::Blocks(blocks)) = &req.system {
            assert_eq!(blocks.len(), 1);
            assert!(blocks[0].cache_control.is_some());
        } else {
            panic!("Expected system blocks");
        }
    }

    #[test]
    fn request_with_tools() {
        let json = json!({
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [{
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {"type": "object", "properties": {}}
            }],
            "tool_choice": {"type": "auto"}
        });
        let req: AnthropicMessagesRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.tools.as_ref().unwrap().len(), 1);
        assert!(matches!(req.tool_choice, Some(AnthropicToolChoice::Auto)));
    }

    #[test]
    fn response_round_trip() {
        let json = json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0
            }
        });
        let resp: AnthropicMessagesResponse = serde_json::from_value(json).unwrap();
        assert_eq!(resp.id, "msg_123");
        assert_eq!(resp.content.len(), 1);
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp.usage.output_tokens, 20);
    }

    #[test]
    fn response_with_tool_use() {
        let json = json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "SF"}}
            ],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "tool_use",
            "stop_sequence": null,
            "usage": {"input_tokens": 10, "output_tokens": 50}
        });
        let resp: AnthropicMessagesResponse = serde_json::from_value(json).unwrap();
        assert_eq!(resp.content.len(), 2);
        assert!(
            matches!(&resp.content[1], AnthropicContentBlock::ToolUse { name, .. } if name == "get_weather")
        );
    }

    #[test]
    fn stream_event_message_start() {
        let json = json!({
            "type": "message_start",
            "message": {
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-sonnet-20241022",
                "usage": {"input_tokens": 25}
            }
        });
        let event: AnthropicStreamEvent = serde_json::from_value(json).unwrap();
        assert_eq!(event.event_type(), "message_start");
        if let AnthropicStreamEvent::MessageStart { message } = &event {
            assert_eq!(message.id, "msg_123");
            assert_eq!(message.usage.input_tokens, 25);
        } else {
            panic!("Expected MessageStart");
        }
    }

    #[test]
    fn stream_event_content_delta_text() {
        let json = json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"}
        });
        let event: AnthropicStreamEvent = serde_json::from_value(json).unwrap();
        assert_eq!(event.event_type(), "content_block_delta");
    }

    #[test]
    fn stream_event_content_delta_tool_json() {
        let json = json!({
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": "{\"city\":"}
        });
        let event: AnthropicStreamEvent = serde_json::from_value(json).unwrap();
        if let AnthropicStreamEvent::ContentBlockDelta { delta, index } = &event {
            assert_eq!(*index, 1);
            assert!(
                matches!(delta, ContentDelta::InputJsonDelta { partial_json } if partial_json == "{\"city\":")
            );
        } else {
            panic!("Expected ContentBlockDelta");
        }
    }

    #[test]
    fn stream_event_message_delta() {
        let json = json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 50, "input_tokens": 0}
        });
        let event: AnthropicStreamEvent = serde_json::from_value(json).unwrap();
        assert_eq!(event.event_type(), "message_delta");
        if let AnthropicStreamEvent::MessageDelta { delta, usage } = &event {
            assert_eq!(delta.stop_reason.as_deref(), Some("end_turn"));
            assert_eq!(usage.output_tokens, 50);
        } else {
            panic!("Expected MessageDelta");
        }
    }

    #[test]
    fn content_block_tool_result() {
        let json = json!({
            "type": "tool_result",
            "tool_use_id": "tu_1",
            "content": "72°F and sunny"
        });
        let block: AnthropicContentBlock = serde_json::from_value(json).unwrap();
        assert!(
            matches!(block, AnthropicContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == "tu_1")
        );
    }
}
