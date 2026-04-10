mod concurrent;
mod ratelimit;

use anyhow::Result;
use axum::{
    body::Body,
    response::{IntoResponse, Response},
};
use concurrent::{
    ConcurrencyPermit, ConcurrencyPermits,
    utils::{ConcurrencyLimitResponse, ConcurrencyState, run_concurrency_check},
};
use log::error;
use ratelimit::utils::{CheckPhase, RateLimitResponse, RateLimitState, run_check};
use thiserror::Error;

use crate::{
    config::entities::{ApiKey, Model, ResourceEntry},
    proxy::hooks::{
        RequestContext, ResponseData, TokenUsage, rate_limit::ratelimit::RateLimitError,
    },
};

#[derive(Debug, Error)]
pub enum RateLimitHookError {
    #[error("Rate limit exceeded")]
    Raw(Response<Body>),
}

async fn get_resources(ctx: &RequestContext) -> (ResourceEntry<ApiKey>, ResourceEntry<Model>) {
    let guard = ctx.extensions().await;
    let api_key = guard
        .get::<ResourceEntry<ApiKey>>()
        .cloned()
        .expect("apikey should exist in context");
    let model = guard
        .get::<ResourceEntry<Model>>()
        .cloned()
        .expect("model should exist in context");
    (api_key, model)
}

async fn run_post_check(ctx: &mut RequestContext, total_tokens: u64) {
    let (api_key, model) = get_resources(ctx).await;
    let mut guard = ctx.extensions_mut().await;
    let rate_limit_state = guard
        .get_mut::<RateLimitState>()
        .expect("rate limit state should be initialized in context");

    apply_post_check("api_key", &api_key, total_tokens, rate_limit_state).await;
    apply_post_check("model", &model, total_tokens, rate_limit_state).await;
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
            error!("Post-check error for {name}: metric={metric:?}, error={msg}");
        }
        Err(_) => {}
    }
}

/// Run concurrency check for an entity and collect the permit.
/// Returns an error response if the concurrency limit is exceeded.
async fn apply_concurrency_check<T: crate::config::entities::types::HasRateLimit>(
    id: String,
    entity: &T,
    permits: &mut Vec<ConcurrencyPermit>,
    concurrency_state: &mut ConcurrencyState,
) -> Option<axum::response::Response> {
    match run_concurrency_check(entity).await {
        None => None, // No concurrency limit configured
        Some(Ok(permit)) => {
            concurrency_state.store_check(permit.info.clone());
            permits.push(permit);
            None
        }
        Some(Err(error)) => Some(ConcurrencyLimitResponse::new(id, error).into_response()),
    }
}

#[fastrace::trace]
pub async fn pre_check(ctx: &mut RequestContext) -> Result<(), RateLimitHookError> {
    let (api_key, model) = get_resources(ctx).await;

    // --- Rate limit checks ---
    {
        let mut guard = ctx.extensions_mut().await;
        if guard.get::<RateLimitState>().is_none() {
            guard.insert(RateLimitState::new());
        }

        let rate_limit_state = guard
            .get_mut::<RateLimitState>()
            .expect("rate limit state should be initialized in context");

        if let Some(resp) = apply_pre_check(api_key.id.clone(), &api_key, rate_limit_state).await {
            return Err(RateLimitHookError::Raw(resp));
        }
        if let Some(resp) = apply_pre_check(model.id.clone(), &model, rate_limit_state).await {
            return Err(RateLimitHookError::Raw(resp));
        }
    }

    // --- Concurrency checks ---
    let mut permits = Vec::new();

    {
        let mut guard = ctx.extensions_mut().await;
        if guard.get::<ConcurrencyState>().is_none() {
            guard.insert(ConcurrencyState::new());
        }

        {
            let concurrency_state = guard
                .get_mut::<ConcurrencyState>()
                .expect("concurrency state should be initialized in context");

            if let Some(resp) = apply_concurrency_check(
                api_key.id.clone(),
                &api_key,
                &mut permits,
                concurrency_state,
            )
            .await
            {
                return Err(RateLimitHookError::Raw(resp));
            }

            if let Some(resp) =
                apply_concurrency_check(model.id.clone(), &model, &mut permits, concurrency_state)
                    .await
            {
                return Err(RateLimitHookError::Raw(resp));
            }
        }

        if !permits.is_empty() {
            guard.insert(ConcurrencyPermits(permits));
        }
    }

    Ok(())
}

#[fastrace::trace]
pub async fn post_check(ctx: &mut RequestContext, response: &ResponseData) -> Result<()> {
    let usage = response.token_usage();
    run_post_check(ctx, usage.total_tokens).await;
    Ok(())
}

#[fastrace::trace]
pub async fn post_check_streaming(ctx: &mut RequestContext) -> Result<()> {
    let total_tokens = ctx
        .extensions()
        .await
        .get::<TokenUsage>()
        .map(|usage| usage.total_tokens)
        .unwrap_or(0);

    run_post_check(ctx, total_tokens).await;
    Ok(())
}

pub async fn inject_response_headers(
    ctx: &mut RequestContext,
    headers: &mut axum::http::HeaderMap,
) {
    let mut guard = ctx.extensions_mut().await;

    if let Some(rate_limit_state) = guard.get_mut::<RateLimitState>() {
        rate_limit_state.add_headers(headers);
    }

    if let Some(concurrency_state) = guard.get_mut::<ConcurrencyState>() {
        concurrency_state.add_headers(headers);
    }
}
