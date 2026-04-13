use std::time::Instant;

use metrics::{counter, histogram};

use crate::{
    gateway::types::common::Usage,
    proxy::hooks::{RequestContext, authorization::RequestModel},
};

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

fn record_token_usage(model_name: String, usage: &Usage) {
    counter!(
        crate::utils::metrics::TOKEN_COUNT_KEY,
        "type" => "prompt",
        "model" => model_name.clone(),
    )
    .increment(usage.input_tokens.unwrap_or(0) as u64);

    counter!(
        crate::utils::metrics::TOKEN_COUNT_KEY,
        "type" => "completion",
        "model" => model_name.clone(),
    )
    .increment(usage.output_tokens.unwrap_or(0) as u64);

    counter!(
        crate::utils::metrics::TOKEN_COUNT_KEY,
        "type" => "total",
        "model" => model_name,
    )
    .increment(usage.resolved_total_tokens().map(u64::from).unwrap_or(0));
}

/// Records the request start timestamp in the request context.
pub async fn record_start_time(ctx: &mut RequestContext) {
    ctx.extensions_mut().await.insert(StartTime(Instant::now()));
}

/// Records latency and token metrics for a non-streaming response.
pub async fn record_usage(ctx: &mut RequestContext, usage: &Usage) {
    let model_name = get_request_model_name(ctx).await;
    record_llm_latency(ctx, model_name.clone()).await;
    record_token_usage(model_name, usage);
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
pub async fn record_streaming_usage(ctx: &mut RequestContext, usage: &Usage) {
    let model_name = get_request_model_name(ctx).await;

    record_llm_latency(ctx, model_name.clone()).await;
    record_token_usage(model_name, usage);
}
