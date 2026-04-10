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
    providers::create_provider,
    proxy::{
        AppState,
        hooks::{self, RequestContext, ResponseData},
    },
    utils::future::maybe_timeout,
};

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

    let provider = create_provider(&model.provider_config);
    let timeout = model.timeout.map(Duration::from_millis);

    // Replace request model name with real model name
    request_data.model = model.model.name.clone();

    match maybe_timeout(timeout, provider.embedding(request_data)).await {
        Ok(Ok(response)) => {
            let response_data = ResponseData::Embedding(response.clone());
            let mut resp = Json(response).into_response();
            if let Err(err) = hooks::rate_limit::post_check(&mut request_ctx, &response_data).await
            {
                error!("Rate limit post_check error: {}", err);
            }
            hooks::observability::record_usage(&mut request_ctx, &response_data).await;
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
