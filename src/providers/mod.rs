use async_trait::async_trait;
use std::error::Error;

use crate::handler::chat::{ChatCompletionRequest, ChatCompletionResponse};

#[async_trait]
pub trait Provider: Send + Sync {
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, Box<dyn Error + Send + Sync>>;

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<reqwest::Response, Box<dyn Error + Send + Sync>>;
}

pub mod deepseek;
