use anyhow::Result;
use async_trait::async_trait;
use axum::{http::HeaderMap, response::IntoResponse};

use super::{HookAction, Context, ProxyHook, ResponseData, TokenUsage};
use crate::proxy::policies::rate_limit::{self, RateLimitError, RateLimitResponse};

/// Rate limit hook implementation
pub struct RateLimitHook;

impl RateLimitHook {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ProxyHook for RateLimitHook {
    fn name(&self) -> &str {
        "rate_limit"
    }

    async fn pre_call(&self, ctx: &mut Context) -> Result<HookAction> {
        // 1. Execute pre-check for API key
        match rate_limit::pre_check(&ctx.api_key).await {
            Ok(results) => {
                ctx.rate_limit_state.store_pre_check(results);
            }
            Err((metric, error)) => {
                // Return 429 error
                let response =
                    RateLimitResponse::new(ctx.api_key.id.clone(), metric, error).into_response();
                return Ok(HookAction::EarlyReturn(response));
            }
        }

        // 2. Execute pre-check for model
        match rate_limit::pre_check(&ctx.model).await {
            Ok(results) => {
                ctx.rate_limit_state.store_pre_check(results);
            }
            Err((metric, error)) => {
                let response =
                    RateLimitResponse::new(ctx.model.name.clone(), metric, error).into_response();
                return Ok(HookAction::EarlyReturn(response));
            }
        }

        Ok(HookAction::Continue)
    }

    async fn post_call_success(
        &self,
        ctx: &mut Context,
        response: &ResponseData,
    ) -> Result<()> {
        let usage = response.token_usage();

        // 1. Execute post-check for API key (using total_tokens)
        match rate_limit::post_check(&ctx.api_key, usage.total_tokens).await {
            Ok(results) => {
                ctx.rate_limit_state.store_post_check(results);
            }
            Err((metric, err)) => {
                if let RateLimitError::Internal(msg) = &err {
                    log::error!(
                        "Post-check error for api_key: metric={:?}, error={}",
                        metric,
                        msg
                    );
                }
            }
        }

        // 2. Execute post-check for model (using total_tokens)
        match rate_limit::post_check(&ctx.model, usage.total_tokens).await {
            Ok(results) => {
                ctx.rate_limit_state.store_post_check(results);
            }
            Err((metric, err)) => {
                if let RateLimitError::Internal(msg) = &err {
                    log::error!(
                        "Post-check error for model: metric={:?}, error={}",
                        metric,
                        msg
                    );
                }
            }
        }

        Ok(())
    }

    async fn post_call_streaming(&self, ctx: &mut Context, usage: &TokenUsage) -> Result<()> {
        // Streaming response post-check logic (called once after stream ends)
        // Uses usage information extracted from the last chunk

        // Execute post-check for API key and model (same as post_call_success)
        match rate_limit::post_check(&ctx.api_key, usage.total_tokens).await {
            Ok(results) => {
                ctx.rate_limit_state.store_post_check(results);
            }
            Err((metric, err)) => {
                if let RateLimitError::Internal(msg) = &err {
                    log::error!(
                        "Post-check error for api_key (streaming): metric={:?}, error={}",
                        metric,
                        msg
                    );
                }
            }
        }

        match rate_limit::post_check(&ctx.model, usage.total_tokens).await {
            Ok(results) => {
                ctx.rate_limit_state.store_post_check(results);
            }
            Err((metric, err)) => {
                if let RateLimitError::Internal(msg) = &err {
                    log::error!(
                        "Post-check error for model (streaming): metric={:?}, error={}",
                        metric,
                        msg
                    );
                }
            }
        }

        Ok(())
    }

    async fn post_call_headers(&self, ctx: &Context, headers: &mut HeaderMap) -> Result<()> {
        // Add rate limit headers
        ctx.rate_limit_state.add_headers(headers);
        Ok(())
    }
}
