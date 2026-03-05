use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::proxy::types::{
    ChatCompletionChoice, ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionChunkDelta,
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionUsage, ChatMessage,
};

#[derive(Debug, Serialize)]
pub struct AnthropicMessagesRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    pub max_tokens: u32,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<Vec<AnthropicSystemBlock>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<AnthropicMetadata>,
}

#[derive(Debug, PartialEq, Serialize)]
pub struct AnthropicSystemBlock {
    pub r#type: String,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct AnthropicMetadata {
    pub user_id: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicMessagesResponse {
    pub id: String,
    pub r#type: String,
    pub role: String,
    pub content: Vec<AnthropicContentBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
}

#[derive(Debug, Deserialize)]
pub struct AnthropicUsage {
    pub cache_creation_input_tokens: u32,
    pub cache_read_input_tokens: u32,
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicStreamEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: AnthropicMessageStart },

    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        #[allow(dead_code)]
        index: u32,
        #[allow(dead_code)]
        content_block: AnthropicContentBlock,
    },

    #[serde(rename = "content_block_delta")]
    ContentBlockDelta {
        #[allow(dead_code)]
        index: u32,
        delta: AnthropicDelta,
    },

    #[serde(rename = "content_block_stop")]
    ContentBlockStop {
        #[allow(dead_code)]
        index: u32,
    },

    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: AnthropicMessageDeltaBody,
        usage: AnthropicUsage,
    },

    #[serde(rename = "message_stop")]
    MessageStop,

    #[serde(rename = "ping")]
    Ping,

    #[serde(rename = "error")]
    Error { error: AnthropicErrorBody },
}

#[derive(Debug, Deserialize)]
pub struct AnthropicMessageStart {
    pub id: String,
    pub model: String,
    #[allow(dead_code)]
    pub role: String,
    #[allow(dead_code)]
    pub usage: AnthropicUsage,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicDelta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
}

#[derive(Debug, Deserialize)]
pub struct AnthropicMessageDeltaBody {
    pub stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicErrorBody {
    pub r#type: String,
    pub message: String,
}

impl From<ChatCompletionRequest> for AnthropicMessagesRequest {
    fn from(request: ChatCompletionRequest) -> Self {
        // Anthropic does not have a "system" role in messages.
        // Extract system messages into top-level `system` param as TextBlockParam array.
        let mut system_blocks: Vec<AnthropicSystemBlock> = Vec::new();
        let mut messages: Vec<AnthropicMessage> = Vec::new();

        for msg in request.messages {
            if msg.role == "system" {
                system_blocks.push(AnthropicSystemBlock {
                    r#type: "text".to_string(),
                    text: msg.content,
                });
            } else {
                messages.push(AnthropicMessage {
                    role: msg.role,
                    content: msg.content,
                });
            }
        }

        let system = if system_blocks.is_empty() {
            None
        } else {
            Some(system_blocks)
        };

        // Anthropic requires max_tokens; default to 4096 if not provided.
        let max_tokens = request.max_tokens.unwrap_or(4096);

        // Map OpenAI `user` to Anthropic `metadata.user_id`.
        let metadata = request.user.map(|user_id| AnthropicMetadata { user_id });

        AnthropicMessagesRequest {
            model: request.model,
            messages,
            max_tokens,
            system,
            temperature: request.temperature,
            top_p: request.top_p,
            stop_sequences: request.stop,
            stream: request.stream,
            metadata,
        }
    }
}

fn map_stop_reason(stop_reason: &Option<String>) -> Option<String> {
    stop_reason.as_ref().map(|reason| {
        match reason.as_str() {
            "end_turn" => "stop",
            "max_tokens" => "length",
            "stop_sequence" => "stop",
            other => other,
        }
        .to_string()
    })
}

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

impl From<AnthropicMessagesResponse> for ChatCompletionResponse {
    fn from(resp: AnthropicMessagesResponse) -> Self {
        let content = resp
            .content
            .iter()
            .map(|block| match block {
                AnthropicContentBlock::Text { text } => text.as_str(),
            })
            .collect::<Vec<_>>()
            .join("");

        // According to https://platform.claude.com/docs/en/api/messages/create#raw_message_delta_event.usage:
        // input_tokens includes all tokens in the input, including direct input, cache creation, and cache reads.
        let prompt_tokens = resp.usage.input_tokens
            + resp.usage.cache_creation_input_tokens
            + resp.usage.cache_read_input_tokens;
        let total_tokens = prompt_tokens + resp.usage.output_tokens;

        ChatCompletionResponse {
            id: resp.id,
            object: "chat.completion".to_string(),
            created: now_unix_secs(),
            model: resp.model,
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content,
                    name: None,
                },
                finish_reason: map_stop_reason(&resp.stop_reason),
            }],
            usage: ChatCompletionUsage {
                prompt_tokens,
                completion_tokens: resp.usage.output_tokens,
                total_tokens,
            },
        }
    }
}

pub struct StreamState {
    pub id: String,
    pub model: String,
    pub created: u64,
}

impl StreamState {
    pub fn new() -> Self {
        Self {
            id: String::new(),
            model: String::new(),
            created: now_unix_secs(),
        }
    }

    /// Returns `None` for events that don't map to a chunk (ping, content_block_start/stop, message_stop).
    pub fn process_event(&mut self, event: AnthropicStreamEvent) -> Option<ChatCompletionChunk> {
        match event {
            AnthropicStreamEvent::MessageStart { message } => {
                self.id = message.id;
                self.model = message.model.clone();

                Some(ChatCompletionChunk {
                    id: self.id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created: self.created,
                    model: message.model,
                    choices: vec![ChatCompletionChunkChoice {
                        index: 0,
                        delta: ChatCompletionChunkDelta {
                            role: Some("assistant".to_string()),
                            content: Some(String::new()),
                        },
                        finish_reason: None,
                    }],
                    usage: None,
                })
            }
            AnthropicStreamEvent::ContentBlockDelta { delta, .. } => {
                let text = match delta {
                    AnthropicDelta::TextDelta { text } => text,
                };
                Some(ChatCompletionChunk {
                    id: self.id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created: self.created,
                    model: self.model.clone(),
                    choices: vec![ChatCompletionChunkChoice {
                        index: 0,
                        delta: ChatCompletionChunkDelta {
                            role: None,
                            content: Some(text),
                        },
                        finish_reason: None,
                    }],
                    usage: None,
                })
            }
            AnthropicStreamEvent::MessageDelta { delta, usage } => {
                let prompt_tokens = usage.input_tokens
                    + usage.cache_creation_input_tokens
                    + usage.cache_read_input_tokens;
                let total_tokens = prompt_tokens + usage.output_tokens;
                Some(ChatCompletionChunk {
                    id: self.id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created: self.created,
                    model: self.model.clone(),
                    choices: vec![ChatCompletionChunkChoice {
                        index: 0,
                        delta: ChatCompletionChunkDelta {
                            role: None,
                            content: None,
                        },
                        finish_reason: map_stop_reason(&delta.stop_reason),
                    }],
                    usage: Some(ChatCompletionUsage {
                        prompt_tokens,
                        completion_tokens: usage.output_tokens,
                        total_tokens,
                    }),
                })
            }
            AnthropicStreamEvent::ContentBlockStart { .. }
            | AnthropicStreamEvent::ContentBlockStop { .. }
            | AnthropicStreamEvent::MessageStop
            | AnthropicStreamEvent::Ping => None,
            AnthropicStreamEvent::Error { error } => Some(ChatCompletionChunk {
                id: self.id.clone(),
                object: "chat.completion.chunk".to_string(),
                created: self.created,
                model: self.model.clone(),
                choices: vec![ChatCompletionChunkChoice {
                    index: 0,
                    delta: ChatCompletionChunkDelta {
                        role: None,
                        content: Some(format!(
                            "[Anthropic error: {} - {}]",
                            error.r#type, error.message
                        )),
                    },
                    finish_reason: Some("stop".to_string()),
                }],
                usage: None,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_conversion_extracts_system() {
        let request = ChatCompletionRequest {
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: "You are helpful.".to_string(),
                    name: None,
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: "Hello".to_string(),
                    name: None,
                },
            ],
            model: "claude-3-5-sonnet-20241022".to_string(),
            max_tokens: Some(1024),
            temperature: Some(0.7),
            top_p: None,
            stop: Some(vec!["END".to_string()]),
            stream: None,
            frequency_penalty: None,
            logprobs: None,
            n: None,
            presence_penalty: None,
            response_format: None,
            top_logprobs: None,
            user: None,
        };

        let anthropic_req = AnthropicMessagesRequest::from(request);
        let system = anthropic_req.system.expect("should have system blocks");
        assert_eq!(system.len(), 1);
        assert_eq!(system[0].r#type, "text");
        assert_eq!(system[0].text, "You are helpful.");
        assert_eq!(anthropic_req.messages.len(), 1);
        assert_eq!(anthropic_req.messages[0].role, "user");
        assert_eq!(anthropic_req.messages[0].content, "Hello");
        assert_eq!(anthropic_req.max_tokens, 1024);
        assert_eq!(anthropic_req.temperature, Some(0.7));
        assert_eq!(anthropic_req.stop_sequences, Some(vec!["END".to_string()]));
        assert!(anthropic_req.metadata.is_none());
    }

    #[test]
    fn test_request_conversion_no_system() {
        let request = ChatCompletionRequest {
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
                name: None,
            }],
            model: "claude-3-5-sonnet-20241022".to_string(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop: None,
            stream: None,
            frequency_penalty: None,
            logprobs: None,
            n: None,
            presence_penalty: None,
            response_format: None,
            top_logprobs: None,
            user: None,
        };

        let anthropic_req = AnthropicMessagesRequest::from(request);
        assert_eq!(anthropic_req.system, None);
        assert_eq!(anthropic_req.max_tokens, 4096); // default
    }

    #[test]
    fn test_response_conversion() {
        let resp = AnthropicMessagesResponse {
            id: "msg_123".to_string(),
            r#type: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![AnthropicContentBlock::Text {
                text: "Hello!".to_string(),
            }],
            model: "claude-3-5-sonnet-20241022".to_string(),
            stop_reason: Some("end_turn".to_string()),
            usage: AnthropicUsage {
                cache_creation_input_tokens: 7,
                cache_read_input_tokens: 8,
                input_tokens: 10,
                output_tokens: 5,
            },
        };

        let chat_resp = ChatCompletionResponse::from(resp);
        assert_eq!(chat_resp.id, "msg_123");
        assert_eq!(chat_resp.object, "chat.completion");
        assert_eq!(chat_resp.choices.len(), 1);
        assert_eq!(chat_resp.choices[0].message.content, "Hello!");
        assert_eq!(chat_resp.choices[0].message.role, "assistant");
        assert_eq!(chat_resp.choices[0].finish_reason, Some("stop".to_string()));
        assert_eq!(chat_resp.usage.prompt_tokens, 25);
        assert_eq!(chat_resp.usage.completion_tokens, 5);
        assert_eq!(chat_resp.usage.total_tokens, 30);
    }

    #[test]
    fn test_stop_reason_mapping() {
        assert_eq!(
            map_stop_reason(&Some("end_turn".to_string())),
            Some("stop".to_string())
        );
        assert_eq!(
            map_stop_reason(&Some("max_tokens".to_string())),
            Some("length".to_string())
        );
        assert_eq!(
            map_stop_reason(&Some("stop_sequence".to_string())),
            Some("stop".to_string())
        );
        assert_eq!(map_stop_reason(&None), None);
    }

    #[test]
    fn test_stream_state_message_start() {
        let mut state = StreamState::new();
        let event = AnthropicStreamEvent::MessageStart {
            message: AnthropicMessageStart {
                id: "msg_abc".to_string(),
                model: "claude-3-5-sonnet-20241022".to_string(),
                role: "assistant".to_string(),
                usage: AnthropicUsage {
                    cache_creation_input_tokens: 7,
                    cache_read_input_tokens: 8,
                    input_tokens: 25,
                    output_tokens: 1,
                },
            },
        };

        let chunk = state.process_event(event).expect("should produce chunk");
        assert_eq!(chunk.id, "msg_abc");
        assert_eq!(chunk.choices[0].delta.role, Some("assistant".to_string()));
        assert_eq!(state.id, "msg_abc");
        assert_eq!(state.model, "claude-3-5-sonnet-20241022");
    }

    #[test]
    fn test_stream_state_content_delta() {
        let mut state = StreamState::new();
        state.id = "msg_abc".to_string();
        state.model = "claude-3-5-sonnet-20241022".to_string();

        let event = AnthropicStreamEvent::ContentBlockDelta {
            index: 0,
            delta: AnthropicDelta::TextDelta {
                text: "Hello".to_string(),
            },
        };

        let chunk = state.process_event(event).expect("should produce chunk");
        assert_eq!(chunk.choices[0].delta.content, Some("Hello".to_string()));
        assert!(chunk.choices[0].delta.role.is_none());
    }

    #[test]
    fn test_stream_state_ping_returns_none() {
        let mut state = StreamState::new();
        assert!(state.process_event(AnthropicStreamEvent::Ping).is_none());
    }

    #[test]
    fn test_request_conversion_multiple_system_messages() {
        let request = ChatCompletionRequest {
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: "You are helpful.".to_string(),
                    name: None,
                },
                ChatMessage {
                    role: "system".to_string(),
                    content: "Be concise.".to_string(),
                    name: None,
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: "Hello".to_string(),
                    name: None,
                },
            ],
            model: "claude-3-5-sonnet-20241022".to_string(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop: None,
            stream: None,
            frequency_penalty: None,
            logprobs: None,
            n: None,
            presence_penalty: None,
            response_format: None,
            top_logprobs: None,
            user: None,
        };

        let anthropic_req = AnthropicMessagesRequest::from(request);
        let system = anthropic_req.system.expect("should have system blocks");
        assert_eq!(system.len(), 2);
        assert_eq!(system[0].text, "You are helpful.");
        assert_eq!(system[1].text, "Be concise.");
    }

    #[test]
    fn test_request_conversion_metadata_from_user() {
        let request = ChatCompletionRequest {
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
                name: None,
            }],
            model: "claude-3-5-sonnet-20241022".to_string(),
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop: None,
            stream: None,
            frequency_penalty: None,
            logprobs: None,
            n: None,
            presence_penalty: None,
            response_format: None,
            top_logprobs: None,
            user: Some("user-abc-123".to_string()),
        };

        let anthropic_req = AnthropicMessagesRequest::from(request);
        let metadata = anthropic_req.metadata.expect("should have metadata");
        assert_eq!(metadata.user_id, "user-abc-123");
    }

    #[test]
    fn test_system_block_serialization() {
        let block = AnthropicSystemBlock {
            r#type: "text".to_string(),
            text: "You are helpful.".to_string(),
        };
        let json = serde_json::to_value(&block).unwrap();
        assert_eq!(json["type"], "text");
        assert_eq!(json["text"], "You are helpful.");
    }
}
