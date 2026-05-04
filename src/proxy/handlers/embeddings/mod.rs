mod span_attributes;
mod types;

use std::time::Duration;

use axum::{
    Json,
    extract::State,
    response::{IntoResponse, Response},
};
use fastrace::prelude::*;
use log::error;
use span_attributes::{request_span_properties, response_span_properties};
pub use types::EmbeddingError;

use crate::{
    config::entities::{Model, ResourceEntry},
    gateway::{
        error::GatewayError,
        types::{
            common::Usage,
            embed::{EmbeddingRequest, EmbeddingResponse},
        },
    },
    proxy::{
        AppState,
        hooks::{self, RequestContext},
        provider::create_provider_instance,
        utils::trace::span_attributes::apply_span_properties,
    },
    utils::future::{WithSpan, maybe_timeout},
};

fn embedding_usage(response: &EmbeddingResponse) -> Usage {
    match &response.usage {
        Some(usage) => Usage {
            input_tokens: Some(usage.prompt_tokens),
            total_tokens: Some(usage.total_tokens),
            ..Default::default()
        },
        None => Usage::default(),
    }
}

pub async fn embeddings(
    State(state): State<AppState>,
    mut request_ctx: RequestContext,
    Json(mut request_data): Json<EmbeddingRequest>,
) -> Result<Response, EmbeddingError> {
    hooks::observability::record_start_time(&mut request_ctx).await;
    hooks::authorization::check(&mut request_ctx, request_data.model.clone()).await?;
    hooks::rate_limit::pre_check(&mut request_ctx).await?;

    let model = request_ctx
        .extensions()
        .await
        .get::<ResourceEntry<Model>>()
        .cloned()
        .ok_or(EmbeddingError::MissingModelInContext)?;

    let gateway = state.gateway();
    let resources = state.resources();
    let provider = model.provider(resources.as_ref()).ok_or_else(|| {
        GatewayError::Internal(format!("provider {} not found", model.provider_id))
    })?;
    let provider_instance = create_provider_instance(gateway.as_ref(), &provider)?;
    let provider_base_url = provider_instance.effective_base_url().ok();
    let timeout = model.timeout.map(Duration::from_millis);

    // Replace request model name with real model name
    request_data.model = model.model.clone();

    let span = Span::enter_with_local_parent("aisix.llm.embeddings");
    apply_span_properties(
        &span,
        request_span_properties(
            &request_data,
            provider_instance.def.as_ref(),
            provider_base_url.as_ref(),
        ),
    );

    let (response, span) = (WithSpan {
        inner: maybe_timeout(timeout, gateway.embed(&request_data, &provider_instance)),
        span: Some(span),
    })
    .await;

    match response {
        Ok(Ok(response)) => {
            let usage = embedding_usage(&response);
            span.add_properties(|| response_span_properties(&response, &usage));
            let mut resp = Json(response).into_response();
            if let Err(err) = hooks::rate_limit::post_check(&mut request_ctx, &usage).await {
                error!("Rate limit post_check error: {}", err);
            }
            hooks::observability::record_usage(&mut request_ctx, &usage).await;
            hooks::rate_limit::inject_response_headers(&mut request_ctx, resp.headers_mut()).await;

            Ok(resp)
        }
        Ok(Err(err)) => {
            span.add_property(|| ("error.type", "gateway_error"));
            error!("Error generating embeddings: {}", err);
            Err(EmbeddingError::GatewayError(err))
        }
        Err(err) => {
            span.add_property(|| ("error.type", "timeout"));
            Err(EmbeddingError::Timeout(err))
        }
    }
}
