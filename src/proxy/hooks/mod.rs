pub mod authorization;
mod context;
pub mod observability;
pub mod rate_limit;

pub use context::RequestContext;

use crate::proxy::{
    handlers::embeddings::EmbeddingUsage,
    types::{ChatCompletionResponse, ChatCompletionUsage, EmbeddingResponse},
};

/// Response data wrapper for different response types
pub enum ResponseData {
    ChatCompletion(ChatCompletionResponse),
    Embedding(EmbeddingResponse),
}

impl ResponseData {
    pub fn token_usage(&self) -> TokenUsage {
        match self {
            Self::ChatCompletion(resp) => TokenUsage::from_chat_completion(&resp.usage),
            Self::Embedding(resp) => {
                // EmbeddingResponse.usage is Option<EmbeddingUsage>
                if let Some(ref usage) = resp.usage {
                    TokenUsage::from_embedding(usage)
                } else {
                    // If no usage info, return zero tokens
                    TokenUsage {
                        prompt_tokens: None,
                        completion_tokens: None,
                        total_tokens: 0,
                    }
                }
            }
        }
    }
}

/// Token usage statistics
#[derive(Debug, Clone)]
pub struct TokenUsage {
    /// Prompt tokens
    pub prompt_tokens: Option<u64>,
    /// Completion tokens (None for embeddings and other types that don't support it)
    pub completion_tokens: Option<u64>,
    /// Total tokens (always available)
    pub total_tokens: u64,
}

impl TokenUsage {
    /// Create from ChatCompletionResponse usage
    pub fn from_chat_completion(usage: &ChatCompletionUsage) -> Self {
        Self {
            prompt_tokens: Some(usage.prompt_tokens as u64),
            completion_tokens: Some(usage.completion_tokens as u64),
            total_tokens: usage.total_tokens as u64,
        }
    }

    /// Create from EmbeddingResponse usage
    pub fn from_embedding(usage: &EmbeddingUsage) -> Self {
        Self {
            prompt_tokens: Some(usage.prompt_tokens as u64),
            completion_tokens: None,
            total_tokens: usage.total_tokens as u64,
        }
    }
}
