use async_trait::async_trait;
use futures::stream::BoxStream;
use std::{error::Error, sync::LazyLock};

use crate::{
    config::entities::models::ProviderConfig,
    handlers::chat_completions::{
        ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse,
    },
};

mod deepseek;
mod gemini;
mod mock;
mod openai;
mod openai_compatible;

// Re-export identifiers
pub mod identifiers {
    use super::{deepseek, gemini, mock, openai};

    pub const DEEPSEEK: &str = deepseek::IDENTIFIER;
    pub const GEMINI: &str = gemini::IDENTIFIER;
    pub const MOCK: &str = mock::IDENTIFIER;
    pub const OPENAI: &str = openai::IDENTIFIER;
}

// Re-export provider config types
pub mod configs {
    pub use super::{
        deepseek::DeepSeekProviderConfig, gemini::GeminiProviderConfig,
        openai::OpenAIProviderConfig,
    };
}

static REQWEST_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(|| reqwest::Client::new());

pub fn create_provider(config: &ProviderConfig) -> Box<dyn Provider> {
    match config {
        ProviderConfig::OpenAI(config) => Box::new(openai::OpenAIProvider::new(
            REQWEST_CLIENT.clone(),
            config.api_key.clone(),
        )),
        ProviderConfig::DeepSeek(config) => Box::new(deepseek::DeepSeekProvider::new(
            REQWEST_CLIENT.clone(),
            config.api_key.clone(),
        )),
        ProviderConfig::Gemini(config) => Box::new(gemini::GeminiProvider::new(
            REQWEST_CLIENT.clone(),
            config.api_key.clone(),
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
