mod concurrent;
mod ratelimit;

use anyhow::Result;
use axum::{
    body::Body,
    response::{IntoResponse, Response},
};
use concurrent::{
    ConcurrencyPermits,
    utils::{ConcurrencyLimitResponse, ConcurrencyState, run_concurrency_check},
};
use log::error;
use ratelimit::utils::{CheckPhase, RateLimitResponse, RateLimitState, run_check};
use thiserror::Error;

use crate::{
    config::entities::{ApiKey, Model, ResourceEntry},
    proxy::hooks::{
        RequestContext, ResponseData, TokenUsage,
        rate_limit::ratelimit::RateLimitError as RRateLimitError,
    },
};

#[derive(Debug, Error)]
pub enum RateLimitError {
    #[error("Rate limit exceeded")]
    Raw(Response<Body>),
}

impl IntoResponse for RateLimitError {
    fn into_response(self) -> Response {
        match self {
            RateLimitError::Raw(resp) => resp,
        }
    }
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
    let api_key_result = run_check(&api_key, CheckPhase::Post(total_tokens)).await;
    let model_result = run_check(&model, CheckPhase::Post(total_tokens)).await;

    let mut guard = ctx.extensions_mut().await;
    let rate_limit_state = guard
        .get_mut::<RateLimitState>()
        .expect("rate limit state should be initialized in context");

    match api_key_result {
        Ok(results) => rate_limit_state.store_post_check(results),
        Err((metric, RRateLimitError::Internal(msg))) => {
            error!("Post-check error for api_key: metric={metric:?}, error={msg}");
        }
        Err(_) => {}
    }

    match model_result {
        Ok(results) => rate_limit_state.store_post_check(results),
        Err((metric, RRateLimitError::Internal(msg))) => {
            error!("Post-check error for model: metric={metric:?}, error={msg}");
        }
        Err(_) => {}
    }
}

/// Performs pre-checks for rate limiting and concurrency limits.
/// Returns `Ok(())` if all checks pass, or `Err(RateLimitHookError)` if any check fails.
pub async fn pre_check(ctx: &mut RequestContext) -> Result<(), RateLimitError> {
    let (api_key, model) = get_resources(ctx).await;
    let api_key_rate_limit_result = run_check(&api_key, CheckPhase::Pre).await;
    let model_rate_limit_result = run_check(&model, CheckPhase::Pre).await;

    // --- Rate limit checks ---
    {
        let mut guard = ctx.extensions_mut().await;
        if guard.get::<RateLimitState>().is_none() {
            guard.insert(RateLimitState::new());
        }

        let rate_limit_state = guard
            .get_mut::<RateLimitState>()
            .expect("rate limit state should be initialized in context");

        match api_key_rate_limit_result {
            Ok(results) => rate_limit_state.store_pre_check(results),
            Err((metric, error)) => {
                return Err(RateLimitError::Raw(
                    RateLimitResponse::new(api_key.id.clone(), metric, error).into_response(),
                ));
            }
        }

        match model_rate_limit_result {
            Ok(results) => rate_limit_state.store_pre_check(results),
            Err((metric, error)) => {
                return Err(RateLimitError::Raw(
                    RateLimitResponse::new(model.id.clone(), metric, error).into_response(),
                ));
            }
        }
    }

    // --- Concurrency checks ---
    let api_key_concurrency_result = run_concurrency_check(&api_key).await;
    let model_concurrency_result = run_concurrency_check(&model).await;
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

            match api_key_concurrency_result {
                None => {}
                Some(Ok(permit)) => {
                    concurrency_state.store_check(permit.info.clone());
                    permits.push(permit);
                }
                Some(Err(error)) => {
                    return Err(RateLimitError::Raw(
                        ConcurrencyLimitResponse::new(api_key.id.clone(), error).into_response(),
                    ));
                }
            }

            match model_concurrency_result {
                None => {}
                Some(Ok(permit)) => {
                    concurrency_state.store_check(permit.info.clone());
                    permits.push(permit);
                }
                Some(Err(error)) => {
                    return Err(RateLimitError::Raw(
                        ConcurrencyLimitResponse::new(model.id.clone(), error).into_response(),
                    ));
                }
            }
        }

        if !permits.is_empty() {
            guard.insert(ConcurrencyPermits(permits));
        }
    }

    Ok(())
}

/// Performs post-checks for rate limiting after the response is generated.
/// It will record the total token usage and update the rate limit state accordingly.
/// This should be called for non-streaming responses.
pub async fn post_check(ctx: &mut RequestContext, response: &ResponseData) -> Result<()> {
    let usage = response.token_usage();
    run_post_check(ctx, usage.total_tokens).await;
    Ok(())
}

/// Performs post-checks for streaming responses after the stream is completed.
/// It will record the total token usage and update the rate limit state accordingly.
/// This should be called after the streaming usage is received.
pub async fn post_check_streaming(ctx: &mut RequestContext) -> Result<()> {
    let total_tokens = {
        match ctx.extensions().await.get::<TokenUsage>() {
            Some(usage) => usage.total_tokens,
            None => 0,
        }
    };

    run_post_check(ctx, total_tokens).await;
    Ok(())
}

/// Injects rate limit and concurrency limit headers into the response header.
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
