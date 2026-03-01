use anyhow::Result;
use async_trait::async_trait;

use crate::proxy::{
    hooks::{HookContext, ProxyHook, ResponseData},
    middlewares::RequestModel,
};

pub struct MetricHook;

#[async_trait]
impl ProxyHook for MetricHook {
    fn name(&self) -> &str {
        "metric"
    }

    async fn post_call_success(&self, ctx: &mut HookContext, response: &ResponseData) -> Result<()> {
        // token count
        let request_model = ctx
            .get::<RequestModel>()
            .expect("RequestModel not found in context");
        let model_name = request_model.0.clone();
        let usage = response.token_usage();

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

        Ok(())
    }
}
