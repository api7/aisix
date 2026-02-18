use anyhow::Result;
use async_trait::async_trait;
use axum::{extract::Request, http::HeaderMap, response::IntoResponse};

use super::{Context, HookAction, ProxyHook, ResponseData, TokenUsage};
use crate::{
    config::entities::{ApiKey, Model, ResourceEntry},
    proxy::policies::rate_limit::{self, RateLimitError, RateLimitResponse, RateLimitState},
};

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

    async fn pre_call(&self, ctx: &mut Context, _req: &mut Request) -> Result<HookAction> {
        //TODO: unwrap safely
        let api_key = ctx.get::<ResourceEntry<ApiKey>>().cloned().unwrap();
        let model = ctx.get::<ResourceEntry<Model>>().cloned().unwrap();
        let rate_limit_state = ctx.get_or_insert(RateLimitState::new());

        // 1. Execute pre-check for API key
        match rate_limit::pre_check(&api_key).await {
            Ok(results) => {
                rate_limit_state.store_pre_check(results);
            }
            Err((metric, error)) => {
                // Return 429 error
                let response =
                    RateLimitResponse::new(api_key.id.clone(), metric, error).into_response();
                return Ok(HookAction::EarlyReturn(response));
            }
        }

        // 2. Execute pre-check for model
        match rate_limit::pre_check(&model).await {
            Ok(results) => {
                rate_limit_state.store_pre_check(results);
            }
            Err((metric, error)) => {
                let response =
                    RateLimitResponse::new(model.id.clone(), metric, error).into_response();
                return Ok(HookAction::EarlyReturn(response));
            }
        }

        Ok(HookAction::Continue)
    }

    async fn post_call_success(&self, ctx: &mut Context, response: &ResponseData) -> Result<()> {
        //TODO: unwrap safely
        let api_key = ctx.get::<ResourceEntry<ApiKey>>().cloned().unwrap();
        let model = ctx.get::<ResourceEntry<Model>>().cloned().unwrap();
        let rate_limit_state = ctx.get_mut::<RateLimitState>().unwrap();

        //TODO: read from context
        let usage = response.token_usage();

        // 1. Execute post-check for API key (using total_tokens)
        match rate_limit::post_check(&api_key, usage.total_tokens).await {
            Ok(results) => {
                rate_limit_state.store_post_check(results);
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
        match rate_limit::post_check(&model, usage.total_tokens).await {
            Ok(results) => {
                rate_limit_state.store_post_check(results);
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
        //TODO: unwrap safely
        let api_key = ctx.get::<ResourceEntry<ApiKey>>().cloned().unwrap();
        let model = ctx.get::<ResourceEntry<Model>>().cloned().unwrap();
        let rate_limit_state = ctx.get_mut::<RateLimitState>().unwrap();

        // Streaming response post-check logic (called once after stream ends)
        // Uses usage information extracted from the last chunk

        // Execute post-check for API key and model (same as post_call_success)
        match rate_limit::post_check(&api_key, usage.total_tokens).await {
            Ok(results) => {
                rate_limit_state.store_post_check(results);
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

        match rate_limit::post_check(&model, usage.total_tokens).await {
            Ok(results) => {
                rate_limit_state.store_post_check(results);
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
        //TODO: unwrap safely
        let rate_limit_state = ctx.get::<RateLimitState>().cloned().unwrap();

        // Add rate limit headers
        //TODO: incorrect remain values
        rate_limit_state.add_headers(headers);
        Ok(())
    }
}
