use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{
    providers::{Provider, types::ProviderError},
    proxy::types::{
        ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse, ChatCompletionUsage,
        ChatMessage,
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
}
