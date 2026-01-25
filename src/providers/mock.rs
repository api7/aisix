use std::error::Error;

use async_trait::async_trait;

use crate::{
    handlers::chat::{ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse},
    providers::Provider,
};

#[derive(Default)]
pub struct MockProvider {}

#[async_trait]
impl Provider for MockProvider {
    #[fastrace::trace(properties = { "request": "{request:?}" })]
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, Box<dyn Error + Send + Sync>> {
        Ok(ChatCompletionResponse {
            id: "1332793c-46e5-4a4b-bbfd-479a3153b99d".to_string(),
            object: "chat.completion".to_string(),
            created: 1769256364,
            model: request.model.clone(),
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: crate::handlers::chat::ChatMessage {
                    name: None,
                    role: "assistant".to_string(),
                    content: format!(
                        "Hello！👋 Current time: {:?}",
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_nanos() as u64
                    ),
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: crate::handlers::chat::ChatCompletionUsage {
                prompt_tokens: 5,
                completion_tokens: 33,
                total_tokens: 38,
            },
        })
    }
}
