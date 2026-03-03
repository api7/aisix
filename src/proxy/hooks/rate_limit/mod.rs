mod local;
mod types;
mod utils;

use anyhow::Result;
use async_trait::async_trait;
use axum::{extract::Request, http::HeaderMap, response::IntoResponse};

use crate::{
    config::entities::{ApiKey, Model, ResourceEntry},
    proxy::hooks::{HookContext, HookError, ProxyHook, ResponseData, TokenUsage},
};
use types::*;
use utils::{CheckPhase, RateLimitResponse, RateLimitState, run_check};

pub struct RateLimitHook;

impl RateLimitHook {
    fn get_resources(ctx: &mut HookContext) -> (ResourceEntry<ApiKey>, ResourceEntry<Model>) {
        let api_key = ctx
            .get::<ResourceEntry<ApiKey>>()
            .cloned()
            .expect("apikey should exist in context");
        let model = ctx
            .get::<ResourceEntry<Model>>()
            .cloned()
            .expect("model should exist in context");
        (api_key, model)
    }

    fn get_rate_limit_state(ctx: &mut HookContext) -> &mut RateLimitState {
        ctx.get_mut::<RateLimitState>()
            .expect("rate limit state should be initialized in context")
    }

    async fn run_post_check(&self, ctx: &mut HookContext, total_tokens: u64) {
        let (api_key, model) = Self::get_resources(ctx);
        let rate_limit_state = Self::get_rate_limit_state(ctx);
        Self::apply_post_check("api_key", &api_key, total_tokens, rate_limit_state).await;
        Self::apply_post_check("model", &model, total_tokens, rate_limit_state).await;
    }

    async fn apply_pre_check<T: crate::config::entities::types::HasRateLimit>(
        id: String,
        entity: &T,
        state: &mut RateLimitState,
    ) -> Option<axum::response::Response> {
        match run_check(entity, CheckPhase::Pre).await {
            Ok(results) => {
                state.store_pre_check(results);
                None
            }
            Err((metric, error)) => Some(RateLimitResponse::new(id, metric, error).into_response()),
        }
    }

    async fn apply_post_check<T: crate::config::entities::types::HasRateLimit>(
        name: &str,
        entity: &T,
        total_tokens: u64,
        state: &mut RateLimitState,
    ) {
        match run_check(entity, CheckPhase::Post(total_tokens)).await {
            Ok(results) => state.store_post_check(results),
            Err((metric, RateLimitError::Internal(msg))) => {
                log::error!(
                    "Post-check error for {}: metric={:?}, error={}",
                    name,
                    metric,
                    msg
                );
            }
            Err(_) => {}
        }
    }
}

#[async_trait]
impl ProxyHook for RateLimitHook {
    fn name(&self) -> &str {
        "rate_limit"
    }

    async fn pre_call(&self, ctx: &mut HookContext, _req: &mut Request) -> Result<(), HookError> {
        let (api_key, model) = Self::get_resources(ctx);
        let rate_limit_state = ctx.get_or_insert(RateLimitState::new());

        if let Some(resp) =
            Self::apply_pre_check(api_key.id.clone(), &api_key, rate_limit_state).await
        {
            return Err(HookError::RawResponse(resp));
        }
        if let Some(resp) = Self::apply_pre_check(model.id.clone(), &model, rate_limit_state).await
        {
            return Err(HookError::RawResponse(resp));
        }

        Ok(())
    }

    async fn post_call_success(
        &self,
        ctx: &mut HookContext,
        response: &ResponseData,
    ) -> Result<(), HookError> {
        let usage = response.token_usage();
        self.run_post_check(ctx, usage.total_tokens).await;
        Ok(())
    }

    async fn post_call_streaming(&self, ctx: &mut HookContext) -> Result<(), HookError> {
        let usage = ctx
            .get::<TokenUsage>()
            .expect("TokenUsage should be in context");

        self.run_post_check(ctx, usage.total_tokens).await;
        Ok(())
    }

    async fn post_call_headers(
        &self,
        ctx: &mut HookContext,
        headers: &mut HeaderMap,
    ) -> Result<(), HookError> {
        let rate_limit_state = Self::get_rate_limit_state(ctx);
        rate_limit_state.add_headers(headers);
        Ok(())
    }
}
