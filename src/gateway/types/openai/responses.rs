//! OpenAI Responses API format types.
//!
//! The Responses API is a higher-level abstraction over Chat Completions
//! that supports server-side conversation state (`previous_response_id`),
//! built-in tools (web search, file search), and a richer output format.
//!
//! In the gateway, requests to `/v1/responses` are bridged to the hub
//! (Chat Completions) format for cross-provider compatibility, with
//! `NativeOpenAIResponsesSupport` providing a bypass for providers that
//! natively support the Responses API (e.g., OpenAI).

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ── Request types ──

/// Responses API request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesApiRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_management: Option<Vec<ContextManagement>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation: Option<ConversationReference>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<String>>,

    pub model: String,
    pub input: ResponsesInput,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tool_calls: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ResponsesTool>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<ResponsePrompt>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_retention: Option<PromptCacheRetention>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<ResponsesStreamOptions>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<ResponseTextConfig>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>,

    /// Server-side state: chain to a previous response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<Truncation>,
}

/// Context management configuration for Responses requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextManagement {
    pub r#type: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub compact_threshold: Option<u32>,
}

/// Conversation reference for a Responses request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ConversationReference {
    Id(String),
    Descriptor { id: String },
}

/// Prompt template reference used by the Responses API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsePrompt {
    pub id: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub variables: Option<Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

/// Prompt cache retention policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PromptCacheRetention {
    #[serde(rename = "in-memory")]
    InMemory,
    #[serde(rename = "24h")]
    TwentyFourHours,
}

/// Reasoning controls for reasoning-capable models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub generate_summary: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

/// Streaming-specific configuration for the Responses API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesStreamOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_obfuscation: Option<bool>,
}

/// Text response configuration for the Responses API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseTextConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<String>,
}

/// Truncation mode for the Responses API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Truncation {
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "disabled")]
    Disabled,
}

/// Input to the Responses API — either a simple string or structured items.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<ResponsesInputItem>),
}

/// An input item in the Responses API.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponsesInputItem {
    #[serde(rename = "message")]
    Message {
        role: String,
        content: ResponsesContent,
    },

    #[serde(rename = "function_call_output")]
    FunctionCallOutput { call_id: String, output: String },
}

/// Content of a Responses API message — string or parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesContent {
    Text(String),
    Parts(Vec<ResponsesContentPart>),
}

/// A content part in a Responses API message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponsesContentPart {
    #[serde(rename = "input_text")]
    InputText { text: String },

    #[serde(rename = "input_image")]
    InputImage {
        #[serde(skip_serializing_if = "Option::is_none")]
        image_url: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
}

/// A tool definition in the Responses API.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponsesTool {
    #[serde(rename = "function")]
    Function {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        parameters: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },

    #[serde(rename = "web_search_preview")]
    WebSearch {
        #[serde(skip_serializing_if = "Option::is_none")]
        user_location: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        search_context_size: Option<String>,
    },

    #[serde(rename = "file_search")]
    FileSearch {
        vector_store_ids: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        max_num_results: Option<u32>,
    },
}

// ── Response types ──

/// Responses API response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesApiResponse {
    pub id: String,
    pub object: String,
    pub created_at: u64,
    pub model: String,
    pub output: Vec<ResponsesOutputItem>,
    pub status: String,
    pub usage: ResponsesUsage,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
}

/// An output item in the Responses API response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponsesOutputItem {
    #[serde(rename = "message")]
    Message {
        id: String,
        role: String,
        content: Vec<ResponsesOutputContent>,
        status: String,
    },

    #[serde(rename = "function_call")]
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
        status: String,
    },
}

/// Content within a Responses API output message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponsesOutputContent {
    #[serde(rename = "output_text")]
    OutputText { text: String },
}

/// Usage reported in the Responses API response.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResponsesUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
}

// ── Streaming event types ──

/// Responses API SSE stream event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponsesApiStreamEvent {
    #[serde(rename = "response.created")]
    ResponseCreated { response: ResponsesApiResponse },

    #[serde(rename = "response.in_progress")]
    ResponseInProgress { response: ResponsesApiResponse },

    #[serde(rename = "response.completed")]
    ResponseCompleted { response: ResponsesApiResponse },

    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        output_index: usize,
        item: ResponsesOutputItem,
    },

    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        output_index: usize,
        item: ResponsesOutputItem,
    },

    #[serde(rename = "response.content_part.added")]
    ContentPartAdded {
        output_index: usize,
        content_index: usize,
        part: ResponsesOutputContent,
    },

    #[serde(rename = "response.content_part.done")]
    ContentPartDone {
        output_index: usize,
        content_index: usize,
        part: ResponsesOutputContent,
    },

    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        output_index: usize,
        content_index: usize,
        delta: String,
    },

    #[serde(rename = "response.output_text.done")]
    OutputTextDone {
        output_index: usize,
        content_index: usize,
        text: String,
    },

    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta { output_index: usize, delta: String },

    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone {
        output_index: usize,
        arguments: String,
    },

    #[serde(rename = "error")]
    Error { message: String },
}

impl ResponsesApiStreamEvent {
    /// Returns the SSE event type string for this event.
    pub fn event_type(&self) -> &'static str {
        match self {
            Self::ResponseCreated { .. } => "response.created",
            Self::ResponseInProgress { .. } => "response.in_progress",
            Self::ResponseCompleted { .. } => "response.completed",
            Self::OutputItemAdded { .. } => "response.output_item.added",
            Self::OutputItemDone { .. } => "response.output_item.done",
            Self::ContentPartAdded { .. } => "response.content_part.added",
            Self::ContentPartDone { .. } => "response.content_part.done",
            Self::OutputTextDelta { .. } => "response.output_text.delta",
            Self::OutputTextDone { .. } => "response.output_text.done",
            Self::FunctionCallArgumentsDelta { .. } => "response.function_call_arguments.delta",
            Self::FunctionCallArgumentsDone { .. } => "response.function_call_arguments.done",
            Self::Error { .. } => "error",
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn request_text_input() {
        let json = json!({
            "model": "gpt-4.1",
            "input": "Hello"
        });
        let req: ResponsesApiRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.model, "gpt-4.1");
        assert!(matches!(req.input, ResponsesInput::Text(ref s) if s == "Hello"));
    }

    #[test]
    fn request_with_current_openai_fields() {
        let json = json!({
            "background": true,
            "context_management": [{"type": "compact", "compact_threshold": 4}],
            "conversation": {"id": "conv_123"},
            "include": [
                "message.input_image.image_url",
                "message.output_text.logprobs"
            ],
            "model": "gpt-5.4",
            "input": [{
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Describe this image"},
                    {"type": "input_image", "file_id": "file_123", "detail": "high"}
                ]
            }],
            "instructions": "Be brief",
            "max_output_tokens": 256,
            "max_tool_calls": 3,
            "temperature": 0.2,
            "top_p": 0.9,
            "parallel_tool_calls": true,
            "prompt": {
                "id": "pmpt_123",
                "variables": {"topic": "weather"},
                "version": "3"
            },
            "prompt_cache_key": "cache-key-1",
            "prompt_cache_retention": "in-memory",
            "reasoning": {
                "effort": "medium",
                "generate_summary": "auto"
            },
            "safety_identifier": "user_123",
            "service_tier": "auto",
            "stream": true,
            "stream_options": {"include_obfuscation": true},
            "metadata": {"trace_id": "abc"},
            "text": {
                "format": {"type": "json_schema"},
                "verbosity": "low"
            },
            "top_logprobs": 5,
            "store": true,
            "truncation": "auto"
        });

        let req: ResponsesApiRequest = serde_json::from_value(json).unwrap();
        assert_eq!(req.background, Some(true));
        assert_eq!(req.max_tool_calls, Some(3));
        assert!(
            matches!(req.conversation, Some(ConversationReference::Descriptor { ref id }) if id == "conv_123")
        );
        assert!(matches!(
            req.prompt_cache_retention,
            Some(PromptCacheRetention::InMemory)
        ));
        assert!(matches!(req.truncation, Some(Truncation::Auto)));
        assert_eq!(req.top_logprobs, Some(5));

        let serialized = serde_json::to_value(&req).unwrap();
        assert_eq!(serialized["prompt_cache_retention"], "in-memory");
        assert_eq!(serialized["truncation"], "auto");
        assert_eq!(serialized["stream_options"]["include_obfuscation"], true);
    }

    #[test]
    fn request_items_input() {
        let json = json!({
            "model": "gpt-4.1",
            "input": [
                {"type": "message", "role": "user", "content": "Hi"},
                {"type": "function_call_output", "call_id": "call_1", "output": "42"}
            ]
        });
        let req: ResponsesApiRequest = serde_json::from_value(json).unwrap();
        if let ResponsesInput::Items(items) = &req.input {
            assert_eq!(items.len(), 2);
            assert!(
                matches!(&items[0], ResponsesInputItem::Message { role, .. } if role == "user")
            );
            assert!(
                matches!(&items[1], ResponsesInputItem::FunctionCallOutput { call_id, .. } if call_id == "call_1")
            );
        } else {
            panic!("Expected Items input");
        }
    }

    #[test]
    fn request_with_tools() {
        let json = json!({
            "model": "gpt-4.1",
            "input": "Search for weather",
            "tools": [
                {"type": "function", "name": "get_weather", "parameters": {"type": "object"}},
                {"type": "web_search_preview"},
                {"type": "file_search", "vector_store_ids": ["vs_1"]}
            ]
        });
        let req: ResponsesApiRequest = serde_json::from_value(json).unwrap();
        let tools = req.tools.unwrap();
        assert_eq!(tools.len(), 3);
        assert!(matches!(&tools[0], ResponsesTool::Function { name, .. } if name == "get_weather"));
        assert!(matches!(&tools[1], ResponsesTool::WebSearch { .. }));
        assert!(
            matches!(&tools[2], ResponsesTool::FileSearch { vector_store_ids, .. } if vector_store_ids == &["vs_1"])
        );
    }

    #[test]
    fn response_round_trip() {
        let json = json!({
            "id": "resp_123",
            "object": "response",
            "created_at": 1700000000u64,
            "model": "gpt-4.1",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello!"}],
                    "status": "completed"
                }
            ],
            "status": "completed",
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        });
        let resp: ResponsesApiResponse = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(resp.id, "resp_123");
        assert_eq!(resp.output.len(), 1);
        assert_eq!(resp.usage.total_tokens, 15);

        let serialized = serde_json::to_value(&resp).unwrap();
        assert_eq!(serialized["id"], "resp_123");
    }

    #[test]
    fn response_with_function_call() {
        let json = json!({
            "id": "resp_123",
            "object": "response",
            "created_at": 1700000000u64,
            "model": "gpt-4.1",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": "{\"city\":\"SF\"}",
                    "status": "completed"
                }
            ],
            "status": "completed",
            "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
        });
        let resp: ResponsesApiResponse = serde_json::from_value(json).unwrap();
        assert!(
            matches!(&resp.output[0], ResponsesOutputItem::FunctionCall { name, .. } if name == "get_weather")
        );
    }

    #[test]
    fn stream_event_output_text_delta() {
        let json = json!({
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hello"
        });
        let event: ResponsesApiStreamEvent = serde_json::from_value(json).unwrap();
        assert_eq!(event.event_type(), "response.output_text.delta");
        if let ResponsesApiStreamEvent::OutputTextDelta {
            delta,
            output_index,
            content_index,
        } = &event
        {
            assert_eq!(delta, "Hello");
            assert_eq!(*output_index, 0);
            assert_eq!(*content_index, 0);
        } else {
            panic!("Expected OutputTextDelta");
        }
    }

    #[test]
    fn stream_event_function_call_arguments_delta() {
        let json = json!({
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "delta": "{\"city\":"
        });
        let event: ResponsesApiStreamEvent = serde_json::from_value(json).unwrap();
        assert_eq!(event.event_type(), "response.function_call_arguments.delta");
    }

    #[test]
    fn stream_event_error() {
        let json = json!({
            "type": "error",
            "message": "rate limit exceeded"
        });
        let event: ResponsesApiStreamEvent = serde_json::from_value(json).unwrap();
        assert_eq!(event.event_type(), "error");
    }

    #[test]
    fn stream_event_response_completed() {
        let response_json = json!({
            "id": "resp_123",
            "object": "response",
            "created_at": 1700000000u64,
            "model": "gpt-4.1",
            "output": [],
            "status": "completed",
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        });
        let json = json!({
            "type": "response.completed",
            "response": response_json
        });
        let event: ResponsesApiStreamEvent = serde_json::from_value(json).unwrap();
        assert_eq!(event.event_type(), "response.completed");
    }

    #[test]
    fn content_parts_multipart() {
        let json = json!({
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe this image"},
                {"type": "input_image", "image_url": "https://example.com/img.png", "detail": "high"}
            ]
        });
        let item: ResponsesInputItem = serde_json::from_value(json).unwrap();
        if let ResponsesInputItem::Message {
            content: ResponsesContent::Parts(parts),
            ..
        } = &item
        {
            assert_eq!(parts.len(), 2);
            assert!(
                matches!(&parts[0], ResponsesContentPart::InputText { text } if text == "Describe this image")
            );
            assert!(
                matches!(&parts[1], ResponsesContentPart::InputImage { image_url: Some(url), .. } if url == "https://example.com/img.png")
            );
        } else {
            panic!("Expected Message with Parts");
        }
    }

    #[test]
    fn input_image_supports_file_id() {
        let json = json!({
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_image", "file_id": "file_123", "detail": "high"}
            ]
        });

        let item: ResponsesInputItem = serde_json::from_value(json).unwrap();
        if let ResponsesInputItem::Message {
            content: ResponsesContent::Parts(parts),
            ..
        } = item
        {
            assert!(
                matches!(parts[0], ResponsesContentPart::InputImage { file_id: Some(ref file_id), .. } if file_id == "file_123")
            );
        } else {
            panic!("Expected Message with image part");
        }
    }
}
