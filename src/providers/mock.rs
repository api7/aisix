use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};
use serde::{Deserialize, Serialize};

use crate::{
    providers::{Provider, types::ProviderError},
    proxy::types::{
        ChatCompletionChoice, ChatCompletionChunk, ChatCompletionChunkChoice,
        ChatCompletionChunkDelta, ChatCompletionRequest, ChatCompletionResponse,
        ChatCompletionUsage, ChatMessage,
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
    #[fastrace::trace(properties = { "request": "{request:?}" })]
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
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_nanos() as u64
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
        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
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
}
