use async_trait::async_trait;
use futures::stream::BoxStream;
use std::error::Error;

use crate::handlers::chat::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse};

#[async_trait]
pub trait Provider: Send + Sync {
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, Box<dyn Error + Send + Sync>> {
        Err("Not implemented".into())
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<
        BoxStream<'static, Result<ChatCompletionChunk, Box<dyn Error + Send + Sync>>>,
        Box<dyn Error + Send + Sync>,
    > {
        Err("Not implemented".into())
    }
}

pub mod deepseek;
pub mod mock;
