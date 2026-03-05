use async_trait::async_trait;
use futures::stream::BoxStream;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::{
    providers::{
        Provider, ProviderError, URLFormatter,
        openai_compatible::{chat_completion, chat_completion_stream, embedding},
    },
    proxy::types::{
        ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, EmbeddingRequest,
        EmbeddingResponse,
    },
};

pub const IDENTIFIER: &str = "openai";
const DEFAULT_API_BASE: &str = "https://api.openai.com/v1";

#[derive(Serialize)]
struct OpenAIChatCompletionRequestStreamOptions {
    include_usage: bool,
}

#[derive(Serialize)]
struct OpenAIChatCompletionRequest<T: Serialize> {
    #[serde(flatten)]
    inner: T,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<OpenAIChatCompletionRequestStreamOptions>,
}

impl From<ChatCompletionRequest> for OpenAIChatCompletionRequest<ChatCompletionRequest> {
    fn from(request: ChatCompletionRequest) -> Self {
        Self {
            inner: request,
            stream_options: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, utoipa::ToSchema)]
pub struct OpenAIProviderConfig {
    pub api_key: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_base: Option<String>,
}

pub struct OpenAIProvider {
    config: OpenAIProviderConfig,
    client: Client,
}

impl OpenAIProvider {
    pub fn new(client: Client, api_key: String) -> Self {
        Self {
            config: OpenAIProviderConfig {
                api_key: api_key.clone(),
                api_base: None,
            },
            client,
        }
    }

    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.config.api_base = Some(base_url);
        self
    }
}

impl URLFormatter for OpenAIProvider {
    fn format_url(&self, endpoint: &str) -> String {
        format!(
            "{}/{}",
            self.config.api_base.as_deref().unwrap_or(DEFAULT_API_BASE),
            endpoint
        )
    }
}

#[async_trait]
impl Provider for OpenAIProvider {
    #[fastrace::trace]
    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, ProviderError> {
        let url = self.format_url("chat/completions");
        chat_completion(
            self.client.clone(),
            &url,
            &self.config.api_key,
            OpenAIChatCompletionRequest::from(request),
        )
        .await
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<BoxStream<'static, Result<ChatCompletionChunk, ProviderError>>, ProviderError> {
        let url = self.format_url("chat/completions");

        let mut request = OpenAIChatCompletionRequest::from(request);
        request.inner.stream = Some(true);
        request.stream_options = Some(OpenAIChatCompletionRequestStreamOptions {
            include_usage: true,
        });
        chat_completion_stream(self.client.clone(), &url, &self.config.api_key, request).await
    }

    async fn embedding(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, ProviderError> {
        let url = self.format_url("embeddings");
        embedding(self.client.clone(), &url, &self.config.api_key, request).await
    }
}
