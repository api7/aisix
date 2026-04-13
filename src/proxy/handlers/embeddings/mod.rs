mod types;

use std::time::Duration;

use axum::{
    Json,
    extract::State,
    response::{IntoResponse, Response},
};
use log::error;
pub use types::*;

use crate::{
    config::entities::{Model, ResourceEntry},
    gateway::types::common::Usage,
    proxy::{
        AppState,
        hooks::{self, RequestContext},
        provider::create_legacy_provider,
    },
    utils::future::maybe_timeout,
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

#[fastrace::trace]
pub async fn embeddings(
    State(_state): State<AppState>,
    mut request_ctx: RequestContext,
    Json(mut request_data): Json<EmbeddingRequest>,
) -> Result<Response, EmbeddingError> {
    hooks::observability::record_start_time(&mut request_ctx).await;
    hooks::authorization::check(&mut request_ctx, request_data.model.clone()).await?;
    hooks::rate_limit::pre_check(&mut request_ctx).await?;

    //TODO: safe unwrap
    let model = request_ctx
        .extensions()
        .await
        .get::<ResourceEntry<Model>>()
        .cloned()
        .unwrap();

    let provider = create_legacy_provider(&model);
    let timeout = model.timeout.map(Duration::from_millis);

    // Replace request model name with real model name
    request_data.model = model.model.name.clone();

    match maybe_timeout(timeout, provider.embedding(request_data)).await {
        Ok(Ok(response)) => {
            let usage = embedding_usage(&response);
            let mut resp = Json(response).into_response();
            if let Err(err) = hooks::rate_limit::post_check(&mut request_ctx, &usage).await {
                error!("Rate limit post_check error: {}", err);
            }
            hooks::observability::record_usage(&mut request_ctx, &usage).await;
            hooks::rate_limit::inject_response_headers(&mut request_ctx, resp.headers_mut()).await;

            Ok(resp)
        }
        Ok(Err(err)) => {
            error!("Error generating embeddings: {}", err);
            Err(EmbeddingError::ProviderError(err))
        }
        Err(err) => Err(EmbeddingError::Timeout(err)),
    }
}
