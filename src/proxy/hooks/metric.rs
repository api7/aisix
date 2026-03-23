use std::time::Instant;

use anyhow::Result;
use async_trait::async_trait;
use axum::extract::Request;
use log::error;
use metrics::{counter, histogram};

use crate::proxy::{
    hooks::{HookContext, HookError, ProxyHook, ResponseData, TokenUsage},
    middlewares::RequestModel,
    types::ChatCompletionChunk,
};

#[derive(Clone)]
pub struct StartTime(Instant);

pub struct MetricHook;

impl MetricHook {
    fn get_start_time(ctx: &HookContext) -> Instant {
        ctx.get::<StartTime>()
            .expect("StartTime should be in context")
            .0
    }

    fn get_request_model_name(ctx: &HookContext) -> String {
        ctx.get::<RequestModel>()
            .expect("RequestModel should be in context")
            .0
            .clone()
    }

    fn record_llm_latency(model_name: String, ctx: &HookContext) {
        histogram!(
            crate::utils::metrics::LLM_LATENCY_KEY,
            "model" => model_name,
        )
        .record(MetricHook::get_start_time(ctx).elapsed().as_millis() as f64);
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
}

#[async_trait]
impl ProxyHook for MetricHook {
    fn name(&self) -> &str {
        "metric"
    }

    async fn pre_call(
        &self,
        ctx: &mut HookContext,
        _request: &mut Request,
    ) -> Result<(), HookError> {
        ctx.insert(StartTime(Instant::now()));
        Ok(())
    }

    async fn post_call_success(
        &self,
        ctx: &mut HookContext,
        response: &ResponseData,
    ) -> Result<(), HookError> {
        let model_name = MetricHook::get_request_model_name(ctx);
        Self::record_llm_latency(model_name.clone(), ctx);
        Self::record_token_usage(model_name, &response.token_usage());

        Ok(())
    }

    async fn post_call_streaming_chunk(
        &self,
        ctx: &mut HookContext,
        _chunk: &ChatCompletionChunk,
        idx: i32,
    ) -> Result<(), HookError> {
        if idx == 0 {
            let model_name = MetricHook::get_request_model_name(ctx);

            histogram!(
                crate::utils::metrics::LLM_FIRST_TOKEN_LATENCY_KEY,
                "model" => model_name,
            )
            .record(MetricHook::get_start_time(ctx).elapsed().as_millis() as f64);
        }

        Ok(())
    }

    async fn post_call_streaming(&self, ctx: &mut HookContext) -> Result<(), HookError> {
        let model_name = MetricHook::get_request_model_name(ctx);

        Self::record_llm_latency(model_name.clone(), ctx);

        let usage = match ctx.get::<TokenUsage>() {
            Some(usage) => usage,
            None => {
                error!("Token usage not found in context for model {}", model_name);
                return Ok(());
            }
        };

        Self::record_token_usage(model_name, usage);

        Ok(())
    }
}
