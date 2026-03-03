use std::time::Instant;

use anyhow::Result;
use async_trait::async_trait;
use axum::extract::Request;
use log::error;

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
        crate::utils::metrics::METRIC_LLM_LATENCY.record(
            MetricHook::get_start_time(ctx).elapsed().as_millis() as u64,
            &[opentelemetry::KeyValue::new("model", model_name)],
        );
    }

    fn record_token_usage(model_name: String, usage: &TokenUsage) {
        crate::utils::metrics::METRIC_TOKEN_COUNT.add(
            usage.prompt_tokens.unwrap_or(0),
            &[
                opentelemetry::KeyValue::new("type", "prompt"),
                opentelemetry::KeyValue::new("model", model_name.clone()),
            ],
        );

        crate::utils::metrics::METRIC_TOKEN_COUNT.add(
            usage.completion_tokens.unwrap_or(0),
            &[
                opentelemetry::KeyValue::new("type", "completion"),
                opentelemetry::KeyValue::new("model", model_name.clone()),
            ],
        );

        crate::utils::metrics::METRIC_TOKEN_COUNT.add(
            usage.total_tokens,
            &[
                opentelemetry::KeyValue::new("type", "total"),
                opentelemetry::KeyValue::new("model", model_name),
            ],
        );
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

            crate::utils::metrics::METRIC_LLM_FIRST_TOKEN_LATENCY.record(
                MetricHook::get_start_time(ctx).elapsed().as_millis() as u64,
                &[opentelemetry::KeyValue::new("model", model_name)],
            );
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
