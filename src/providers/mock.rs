use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    providers::{Provider, types::ProviderError},
    proxy::types::{
        ChatCompletionChoice, ChatCompletionChunk, ChatCompletionChunkChoice,
        ChatCompletionChunkDelta, ChatCompletionRequest, ChatCompletionResponse,
        ChatCompletionUsage, ChatMessage, EmbeddingRequest, EmbeddingResponse,
    },
};

pub const IDENTIFIER: &str = "mock";

#[derive(Debug, Default, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct MockProviderConfig {}

#[derive(Default)]
pub struct MockProvider {
    _config: MockProviderConfig,
}

#[async_trait]
impl Provider for MockProvider {
    #[fastrace::trace]
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, ProviderError> {
        let secs: u64 = rand::random_range(1..=5);
        tokio::time::sleep(Duration::from_secs(secs)).await;

        Ok(ChatCompletionResponse {
            id: "1332793c-46e5-4a4b-bbfd-479a3153b99d".to_string(),
            object: "chat.completion".to_string(),
            created: 1769256364,
            model: request.model.clone(),
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatMessage {
                    name: None,
                    role: "assistant".to_string(),
                    content: format!(
                        "Hello! 👋 Current time: {:?}",
                        epoch_duration().as_nanos() as u64
                    ),
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: ChatCompletionUsage {
                prompt_tokens: 5,
                completion_tokens: 33,
                total_tokens: 38,
            },
        })
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<BoxStream<'static, Result<ChatCompletionChunk, ProviderError>>, ProviderError> {
        let id = "ae343f5c-6383-4c33-90e3-26421324b5c5".to_string();
        let created = epoch_duration().as_secs();
        let model = request.model.clone();

        let mut chunks: Vec<ChatCompletionChunk> = Vec::new();

        // First chunk: role announcement
        chunks.push(ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.clone(),
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta: ChatCompletionChunkDelta {
                    role: Some("assistant".to_string()),
                    content: Some("".to_string()),
                },
                finish_reason: None,
            }],
            usage: None,
        });

        let time = format!("{:?}", epoch_duration().as_nanos() as u64);
        let latest_message = request
            .messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .map(|m| m.content.clone())
            .unwrap_or_default();

        // Content tokens
        let content_tokens = vec![
            "Hello",
            "!",
            " I",
            "'m",
            " here",
            " to",
            " help",
            ".",
            " Please",
            " go",
            " ahead",
            " and",
            " ask",
            " your",
            " question",
            " —",
            " I",
            "'ll",
            " do",
            " my",
            " best",
            " to",
            " assist",
            " you",
            ".",
            " Your",
            " message",
            " is",
            " \"",
            &latest_message,
            "\".",
            " Current",
            " time",
            ": ",
            &time,
            ". Have",
            " a",
            " great",
            " day",
            "!",
            " 👋",
        ];

        for token in content_tokens {
            chunks.push(ChatCompletionChunk {
                id: id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model.clone(),
                choices: vec![ChatCompletionChunkChoice {
                    index: 0,
                    delta: ChatCompletionChunkDelta {
                        role: None,
                        content: Some(token.to_string()),
                    },
                    finish_reason: None,
                }],
                usage: None,
            });
        }

        // Final chunk with finish_reason and usage
        chunks.push(ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.clone(),
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta: ChatCompletionChunkDelta {
                    role: None,
                    content: Some("".to_string()),
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(ChatCompletionUsage {
                prompt_tokens: 11,
                completion_tokens: 25,
                total_tokens: 36,
            }),
        });

        let stream = futures::stream::iter(chunks)
            .then(|chunk| async move {
                tokio::time::sleep(Duration::from_millis(100)).await;
                Ok::<ChatCompletionChunk, ProviderError>(chunk)
            })
            .boxed();

        Ok(stream)
    }

    async fn embedding(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, ProviderError> {
        let inputs = embedding_inputs(&request);
        let prompt_tokens = inputs
            .iter()
            .map(|input| estimated_tokens(input))
            .sum::<u32>();

        let data = inputs
            .iter()
            .enumerate()
            .map(|(index, input)| {
                json!({
                    "object": "embedding",
                    "embedding": deterministic_embedding(input, 16),
                    "index": index as i32,
                })
            })
            .collect::<Vec<_>>();

        let response = serde_json::from_value::<EmbeddingResponse>(json!({
            "object": "list",
            "data": data,
            "model": request.model,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens,
            }
        }))?;

        Ok(response)
    }
}

fn epoch_duration() -> Duration {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock must be after UNIX_EPOCH")
}

fn embedding_inputs(request: &EmbeddingRequest) -> Vec<String> {
    let value = serde_json::to_value(&request.input).unwrap_or(Value::Null);
    match value {
        Value::String(single) => vec![single],
        Value::Array(values) => values
            .into_iter()
            .filter_map(|item| item.as_str().map(str::to_owned))
            .collect(),
        _ => vec![],
    }
}

fn estimated_tokens(input: &str) -> u32 {
    let count = input.split_whitespace().count() as u32;
    if count == 0 { 1 } else { count }
}

fn deterministic_embedding(input: &str, dim: usize) -> Vec<f32> {
    let mut state: u64 = 0xcbf29ce484222325;
    for byte in input.bytes() {
        state ^= byte as u64;
        state = state.wrapping_mul(0x100000001b3);
    }

    (0..dim)
        .map(|idx| {
            let mixed = state.rotate_left((idx as u32 * 7) % 64)
                ^ (idx as u64 + 1).wrapping_mul(0x9e3779b97f4a7c15);
            ((mixed % 2000) as f32 / 1000.0) - 1.0
        })
        .collect()
}
