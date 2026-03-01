mod metric;
mod rate_limit;
mod validate_model;

use std::{
    any::Any,
    ops::{Deref, DerefMut},
    sync::LazyLock,
};

use anyhow::Result;
use async_trait::async_trait;
use axum::{
    extract::{FromRequestParts, Request},
    http::HeaderMap,
    response::Response,
};
use http::request::Parts;

use crate::{
    config::entities::{ApiKey, ResourceEntry},
    proxy::{
        AppState,
        handlers::{
            chat_completions::{ChatCompletionChunk, ChatCompletionResponse, ChatCompletionUsage},
            embeddings::{EmbeddingResponse, EmbeddingUsage},
        },
    },
};

pub const HOOK_FILTER_ALL: fn(&Box<dyn ProxyHook>) -> bool = |_| true;
pub const HOOK_FILTER_NONE: fn(&Box<dyn ProxyHook>) -> bool = |_| false;

/// Hook context containing request metadata and state
pub struct HookContext(http::Extensions);

impl HookContext {
    pub fn new() -> Self {
        Self(http::Extensions::new())
    }
}

impl FromRequestParts<AppState> for HookContext {
    type Rejection = ();

    async fn from_request_parts(
        parts: &mut Parts,
        state: &AppState,
    ) -> Result<Self, Self::Rejection> {
        let mut ctx = http::Extensions::new();
        ctx.insert(state.clone());
        ctx.insert(parts.extensions.remove::<ResourceEntry<ApiKey>>().expect(
            "Authentication middleware should have inserted ApiKey into request extensions",
        ));
        Ok(Self(ctx))
    }
}

impl Deref for HookContext {
    type Target = http::Extensions;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for HookContext {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

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
#[allow(unused)]
pub trait ProxyHook: Any + Send + Sync {
    /// Hook name for debugging/logging
    fn name(&self) -> &str;

    /// Pre-call hook: executed before provider call
    /// Can modify request or return early response
    async fn pre_call(&self, _ctx: &mut HookContext, _req: &mut Request) -> Result<HookAction> {
        Ok(HookAction::Continue)
    }

    /// Moderation hook: run parallel input checks (optional)
    async fn moderation(&self, _ctx: &HookContext) -> Result<()> {
        Ok(())
    }

    /// Post-call success hook: executed after successful provider response
    async fn post_call_success(
        &self,
        _ctx: &mut HookContext,
        _response: &ResponseData,
    ) -> Result<()> {
        Ok(())
    }

    /// Post-call streaming chunk hook: wrap the entire stream for real-time processing
    /// Used for guardrails and real-time content filtering on each chunk
    /// Default: return original stream without wrapping
    async fn post_call_streaming_chunk(
        &self,
        _ctx: &HookContext,
        stream: futures::stream::BoxStream<'static, Result<ChatCompletionChunk>>,
    ) -> Result<futures::stream::BoxStream<'static, Result<ChatCompletionChunk>>> {
        Ok(stream)
    }

    /// Post-call streaming hook: called once after stream ends
    /// Used for rate limit post-check and other operations requiring complete usage data
    async fn post_call_streaming(&self, _ctx: &mut HookContext, _usage: &TokenUsage) -> Result<()> {
        Ok(())
    }

    /// Post-call failure hook: executed when provider call fails
    /// Can transform error or return custom response
    async fn post_call_failure(
        &self,
        _ctx: &mut HookContext,
        _error: &anyhow::Error,
    ) -> Result<HookAction> {
        Ok(HookAction::Continue)
    }

    /// Post-call headers hook: add custom headers to response
    async fn post_call_headers(
        &self,
        _ctx: &mut HookContext,
        _headers: &mut HeaderMap,
    ) -> Result<()> {
        Ok(())
    }
}

/// Hook manager for registering and executing hooks
pub struct HookManager {
    pub hooks: Vec<Box<dyn ProxyHook>>,
}

#[allow(unused)]
impl HookManager {
    pub fn new() -> Self {
        Self { hooks: vec![] }
    }

    pub fn register(&mut self, hook: Box<dyn ProxyHook>) -> &mut Self {
        self.hooks.push(hook);
        self
    }

    /// Execute all pre_call hooks in order
    pub async fn pre_call<F>(
        &self,
        ctx: &mut HookContext,
        req: &mut Request,
        filter: F,
    ) -> Result<HookAction>
    where
        F: Fn(&Box<dyn ProxyHook>) -> bool,
    {
        for hook in &self.hooks {
            if !filter(hook) {
                continue;
            }
            match hook.pre_call(ctx, req).await? {
                HookAction::Continue => continue,
                HookAction::EarlyReturn(resp) => return Ok(HookAction::EarlyReturn(resp)),
            }
        }
        Ok(HookAction::Continue)
    }

    /// Execute all post_call_success hooks in order
    pub async fn execute_post_call_success<F>(
        &self,
        ctx: &mut HookContext,
        response: &ResponseData,
        filter: F,
    ) -> Result<()>
    where
        F: Fn(&Box<dyn ProxyHook>) -> bool,
    {
        for hook in &self.hooks {
            if !filter(hook) {
                continue;
            }
            hook.post_call_success(ctx, response).await?;
        }
        Ok(())
    }

    /// Execute all post_call_streaming_chunk hooks (wrap stream)
    pub async fn execute_post_call_streaming_chunk<F>(
        &self,
        ctx: &HookContext,
        mut stream: futures::stream::BoxStream<'static, Result<ChatCompletionChunk>>,
        filter: F,
    ) -> Result<futures::stream::BoxStream<'static, Result<ChatCompletionChunk>>>
    where
        F: Fn(&Box<dyn ProxyHook>) -> bool,
    {
        for hook in &self.hooks {
            if !filter(hook) {
                continue;
            }
            stream = hook.post_call_streaming_chunk(ctx, stream).await?;
        }
        Ok(stream)
    }

    /// Execute all post_call_streaming hooks (after stream ends)
    pub async fn execute_post_call_streaming<F>(
        &self,
        ctx: &mut HookContext,
        usage: &TokenUsage,
        filter: F,
    ) -> Result<()>
    where
        F: Fn(&Box<dyn ProxyHook>) -> bool,
    {
        for hook in &self.hooks {
            if !filter(hook) {
                continue;
            }
            hook.post_call_streaming(ctx, usage).await?;
        }
        Ok(())
    }

    /// Execute all post_call_failure hooks in order
    pub async fn execute_post_call_failure<F>(
        &self,
        ctx: &mut HookContext,
        error: &anyhow::Error,
        filter: F,
    ) -> Result<HookAction>
    where
        F: Fn(&Box<dyn ProxyHook>) -> bool,
    {
        for hook in &self.hooks {
            if !filter(hook) {
                continue;
            }
            match hook.post_call_failure(ctx, error).await? {
                HookAction::Continue => continue,
                HookAction::EarlyReturn(resp) => return Ok(HookAction::EarlyReturn(resp)),
            }
        }
        Ok(HookAction::Continue)
    }

    /// Execute all post_call_headers hooks in order
    pub async fn execute_post_call_headers<F>(
        &self,
        ctx: &mut HookContext,
        headers: &mut HeaderMap,
        filter: F,
    ) -> Result<()>
    where
        F: Fn(&Box<dyn ProxyHook>) -> bool,
    {
        for hook in &self.hooks {
            if !filter(hook) {
                continue;
            }
            hook.post_call_headers(ctx, headers).await?;
        }
        Ok(())
    }
}

pub static HOOK_MANAGER: LazyLock<HookManager> = LazyLock::new(|| {
    let mut manager = HookManager::new();
    manager
        .register(Box::new(validate_model::ValidateModelHook))
        .register(Box::new(rate_limit::RateLimitHook))
        .register(Box::new(metric::MetricHook));
    manager
});
