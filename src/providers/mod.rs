mod anthropic;
mod deepseek;
mod gemini;
mod mock;
mod openai;
mod openai_compatible;
mod types;

use std::sync::LazyLock;

use anyhow::Result;
use async_trait::async_trait;
use futures::stream::BoxStream;
pub use types::ProviderError;

use crate::{
    config::entities::models::ProviderConfig,
    proxy::types::{
        ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest,
        EmbeddingResponse,
    },
};

// Re-export identifiers
pub mod identifiers {
    use super::{anthropic, deepseek, gemini, mock, openai};

    pub const ANTHROPIC: &str = anthropic::IDENTIFIER;
    pub const DEEPSEEK: &str = deepseek::IDENTIFIER;
    pub const GEMINI: &str = gemini::IDENTIFIER;
    pub const MOCK: &str = mock::IDENTIFIER;
    pub const OPENAI: &str = openai::IDENTIFIER;
}

// Re-export provider config types
pub mod configs {
    pub use super::{
        anthropic::AnthropicProviderConfig, deepseek::DeepSeekProviderConfig,
        gemini::GeminiProviderConfig, mock::MockProviderConfig, openai::OpenAIProviderConfig,
    };
}

static REQWEST_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(reqwest::Client::new);

pub fn init_client() {
    let _ = REQWEST_CLIENT.clone();
}

#[fastrace::trace(short_name = true)]
pub fn create_provider(config: &ProviderConfig) -> Box<dyn Provider> {
    match config {
        ProviderConfig::Anthropic(config) => {
            let mut provider =
                anthropic::AnthropicProvider::new(REQWEST_CLIENT.clone(), config.api_key.clone());
            if let Some(api_base) = config.api_base.clone() {
                provider = provider.with_base_url(api_base);
            }
            Box::new(provider)
        }
        ProviderConfig::OpenAI(config) => {
            let mut provider =
                openai::OpenAIProvider::new(REQWEST_CLIENT.clone(), config.api_key.clone());
            if let Some(api_base) = config.api_base.clone() {
                provider = provider.with_base_url(api_base);
            }
            Box::new(provider)
        }
        ProviderConfig::DeepSeek(config) => {
            let mut provider =
                deepseek::DeepSeekProvider::new(REQWEST_CLIENT.clone(), config.api_key.clone());
            if let Some(api_base) = config.api_base.clone() {
                provider = provider.with_base_url(api_base);
            }
            Box::new(provider)
        }
        ProviderConfig::Gemini(config) => {
            let mut provider =
                gemini::GeminiProvider::new(REQWEST_CLIENT.clone(), config.api_key.clone());
            if let Some(api_base) = config.api_base.clone() {
                provider = provider.with_base_url(api_base);
            }
            Box::new(provider)
        }
        ProviderConfig::Mock(_config) => Box::new(mock::MockProvider::default()),
    }
}

#[async_trait]
pub trait Provider: Send + Sync {
    async fn chat_completion(
        &self,
        _request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, ProviderError> {
        Err(ProviderError::NotYetImplemented)
    }

    async fn chat_completion_stream(
        &self,
        _request: ChatCompletionRequest,
    ) -> Result<BoxStream<'static, Result<ChatCompletionChunk, ProviderError>>, ProviderError> {
        Err(ProviderError::NotYetImplemented)
    }

    async fn embedding(
        &self,
        _request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, ProviderError> {
        Err(ProviderError::NotYetImplemented)
    }
}

trait URLFormatter {
    fn format_url(&self, endpoint: &str) -> String;
}
