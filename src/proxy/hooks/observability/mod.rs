use std::time::Instant;

use log::error;
use metrics::{counter, histogram};

use crate::proxy::hooks::{RequestContext, ResponseData, TokenUsage, authorization::RequestModel};

#[derive(Clone)]
struct StartTime(Instant);

async fn get_start_time(ctx: &RequestContext) -> Instant {
    ctx.extensions()
        .await
        .get::<StartTime>()
        .expect("StartTime should be in context")
        .0
}

async fn get_request_model_name(ctx: &RequestContext) -> String {
    ctx.extensions()
        .await
        .get::<RequestModel>()
        .expect("RequestModel should be in context")
        .0
        .clone()
}

async fn record_llm_latency(ctx: &RequestContext, model_name: String) {
    histogram!(
        crate::utils::metrics::LLM_LATENCY_KEY,
        "model" => model_name,
    )
    .record(get_start_time(ctx).await.elapsed().as_millis() as f64);
}

fn record_token_usage(model_name: String, usage: &TokenUsage) {
    counter!(
        crate::utils::metrics::TOKEN_COUNT_KEY,
        "type" => "prompt",
        "model" => model_name.clone(),
    )
    .increment(usage.prompt_tokens.unwrap_or(0));

    counter!(
        crate::utils::metrics::TOKEN_COUNT_KEY,
        "type" => "completion",
        "model" => model_name.clone(),
    )
    .increment(usage.completion_tokens.unwrap_or(0));

    counter!(
        crate::utils::metrics::TOKEN_COUNT_KEY,
        "type" => "total",
        "model" => model_name,
    )
    .increment(usage.total_tokens);
}

/// Records the request start timestamp in the request context.
pub async fn record_start_time(ctx: &mut RequestContext) {
    ctx.extensions_mut().await.insert(StartTime(Instant::now()));
}

/// Records latency and token metrics for a non-streaming response.
pub async fn record_usage(ctx: &mut RequestContext, response: &ResponseData) {
    let model_name = get_request_model_name(ctx).await;
    record_llm_latency(ctx, model_name.clone()).await;
    record_token_usage(model_name, &response.token_usage());
}

/// Records first-token latency for a streaming response.
pub async fn record_first_token_latency(ctx: &mut RequestContext) {
    let model_name = get_request_model_name(ctx).await;

    histogram!(
        crate::utils::metrics::LLM_FIRST_TOKEN_LATENCY_KEY,
        "model" => model_name,
    )
    .record(get_start_time(ctx).await.elapsed().as_millis() as f64);
}

/// Records final latency and token metrics for a completed streaming response.
pub async fn record_streaming_usage(ctx: &mut RequestContext) {
    let model_name = get_request_model_name(ctx).await;

    record_llm_latency(ctx, model_name.clone()).await;

    // Note, to avoid holdind the guard and then across to await boundary
    let guard = ctx.extensions().await;
    let usage = match guard.get::<TokenUsage>() {
        Some(usage) => usage,
        None => {
            error!("Token usage not found in context for model {}", model_name);
            return;
        }
    };

    record_token_usage(model_name, usage);
}
