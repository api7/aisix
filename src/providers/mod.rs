use async_trait::async_trait;
use futures::stream::BoxStream;
use std::{error::Error, sync::LazyLock};

use crate::{
    config::entities::models::ProviderConfig,
    handlers::chat::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse},
};

pub mod deepseek;
pub mod mock;

static REQWEST_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(|| reqwest::Client::new());

pub fn create_provider(config: &ProviderConfig) -> Box<dyn Provider> {
    match config {
        ProviderConfig::DeepSeek(deepseek_config) => Box::new(deepseek::DeepSeekProvider::new(
            REQWEST_CLIENT.clone(),
            deepseek_config.api_key.clone(),
        )),
        ProviderConfig::Mock => Box::new(mock::MockProvider::default()),
    }
}

#[async_trait]
pub trait Provider: Send + Sync {
    async fn chat_completion(
        &self,
        _request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, Box<dyn Error + Send + Sync>> {
        Err("Not implemented".into())
    }

    async fn chat_completion_stream(
        &self,
        _request: ChatCompletionRequest,
    ) -> Result<
        BoxStream<'static, Result<ChatCompletionChunk, Box<dyn Error + Send + Sync>>>,
        Box<dyn Error + Send + Sync>,
    > {
        Err("Not implemented".into())
    }
}
