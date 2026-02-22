mod auth;
mod metric;
mod rate_limit;
mod validate_model;

use std::sync::{Arc, LazyLock};

use anyhow::Result;
use async_trait::async_trait;
use axum::{extract::Request, http::HeaderMap, response::Response};

use crate::proxy::handlers::{
    chat_completions::{ChatCompletionChunk, ChatCompletionResponse, ChatCompletionUsage},
    embeddings::{EmbeddingResponse, EmbeddingUsage},
};

/// Hook context containing request metadata and state
pub type Context = http::Extensions;

/// Hook action result
pub enum HookAction {
    /// Continue normal execution
    Continue,
    /// Return early with custom response
    EarlyReturn(Response),
}

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
    /// Prompt tokens (None for embeddings and other types that don't support it)
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

    /// Create from EmbeddingResponse usage (only total tokens)
    pub fn from_embedding(usage: &EmbeddingUsage) -> Self {
        Self {
            prompt_tokens: None,
            completion_tokens: None,
            total_tokens: usage.total_tokens as u64,
        }
    }
}

/// Proxy hook trait for implementing custom hooks
#[async_trait]
pub trait ProxyHook: Send + Sync {
    /// Hook name for debugging/logging
    fn name(&self) -> &str;

    /// Pre-call hook: executed before provider call
    /// Can modify request or return early response
    async fn pre_call(&self, _ctx: &mut Context, _req: &mut Request) -> Result<HookAction> {
        Ok(HookAction::Continue)
    }

    /// Moderation hook: run parallel input checks (optional)
    async fn moderation(&self, _ctx: &Context) -> Result<()> {
        Ok(())
    }

    /// Post-call success hook: executed after successful provider response
    async fn post_call_success(&self, _ctx: &mut Context, _response: &ResponseData) -> Result<()> {
        Ok(())
    }

    /// Post-call streaming chunk hook: wrap the entire stream for real-time processing
    /// Used for guardrails and real-time content filtering on each chunk
    /// Default: return original stream without wrapping
    async fn post_call_streaming_chunk(
        &self,
        _ctx: &Context,
        stream: futures::stream::BoxStream<'static, Result<ChatCompletionChunk>>,
    ) -> Result<futures::stream::BoxStream<'static, Result<ChatCompletionChunk>>> {
        Ok(stream)
    }

    /// Post-call streaming hook: called once after stream ends
    /// Used for rate limit post-check and other operations requiring complete usage data
    async fn post_call_streaming(&self, _ctx: &mut Context, _usage: &TokenUsage) -> Result<()> {
        Ok(())
    }

    /// Post-call failure hook: executed when provider call fails
    /// Can transform error or return custom response
    async fn post_call_failure(
        &self,
        _ctx: &mut Context,
        _error: &anyhow::Error,
    ) -> Result<HookAction> {
        Ok(HookAction::Continue)
    }

    /// Post-call headers hook: add custom headers to response
    async fn post_call_headers(&self, _ctx: &mut Context, _headers: &mut HeaderMap) -> Result<()> {
        Ok(())
    }
}

/// Hook manager for registering and executing hooks
pub struct HookManager {
    pub hooks: Vec<Arc<dyn ProxyHook>>,
}

impl HookManager {
    pub fn new() -> Self {
        Self { hooks: vec![] }
    }

    pub fn register(&mut self, hook: Arc<dyn ProxyHook>) {
        self.hooks.push(hook);
    }

    /// Execute all pre_call hooks in order
    pub async fn execute_pre_call(
        &self,
        ctx: &mut Context,
        req: &mut Request,
    ) -> Result<HookAction> {
        for hook in &self.hooks {
            match hook.pre_call(ctx, req).await? {
                HookAction::Continue => continue,
                HookAction::EarlyReturn(resp) => return Ok(HookAction::EarlyReturn(resp)),
            }
        }
        Ok(HookAction::Continue)
    }

    /// Execute all post_call_success hooks in order
    pub async fn execute_post_call_success(
        &self,
        ctx: &mut Context,
        response: &ResponseData,
    ) -> Result<()> {
        for hook in &self.hooks {
            hook.post_call_success(ctx, response).await?;
        }
        Ok(())
    }

    /// Execute all post_call_streaming_chunk hooks (wrap stream)
    pub async fn execute_post_call_streaming_chunk(
        &self,
        ctx: &Context,
        mut stream: futures::stream::BoxStream<'static, Result<ChatCompletionChunk>>,
    ) -> Result<futures::stream::BoxStream<'static, Result<ChatCompletionChunk>>> {
        for hook in &self.hooks {
            stream = hook.post_call_streaming_chunk(ctx, stream).await?;
        }
        Ok(stream)
    }

    /// Execute all post_call_streaming hooks (after stream ends)
    pub async fn execute_post_call_streaming(
        &self,
        ctx: &mut Context,
        usage: &TokenUsage,
    ) -> Result<()> {
        for hook in &self.hooks {
            hook.post_call_streaming(ctx, usage).await?;
        }
        Ok(())
    }

    /// Execute all post_call_failure hooks in order
    pub async fn execute_post_call_failure(
        &self,
        ctx: &mut Context,
        error: &anyhow::Error,
    ) -> Result<HookAction> {
        for hook in &self.hooks {
            match hook.post_call_failure(ctx, error).await? {
                HookAction::Continue => continue,
                HookAction::EarlyReturn(resp) => return Ok(HookAction::EarlyReturn(resp)),
            }
        }
        Ok(HookAction::Continue)
    }

    /// Execute all post_call_headers hooks in order
    pub async fn execute_post_call_headers(
        &self,
        ctx: &mut Context,
        headers: &mut HeaderMap,
    ) -> Result<()> {
        for hook in &self.hooks {
            hook.post_call_headers(ctx, headers).await?;
        }
        Ok(())
    }
}

// Global hook manager, initialized at startup and never changed
// No RwLock needed since it's read-only after initialization
pub static HOOK_MANAGER: LazyLock<HookManager> = LazyLock::new(|| {
    let mut manager = HookManager::new();

    // Register built-in hooks
    manager.register(Arc::new(auth::AuthHook::new()));
    manager.register(Arc::new(validate_model::ValidateModelHook::new()));
    manager.register(Arc::new(rate_limit::RateLimitHook::new()));
    manager.register(Arc::new(metric::MetricHook::new()));

    manager
});
