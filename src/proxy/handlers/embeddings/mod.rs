mod types;

use std::time::Duration;

use axum::{
    Json,
    extract::{Extension, Request, State},
    response::{IntoResponse, Response},
};
use log::error;
pub use types::*;

use crate::{
    config::entities::{Model, ResourceEntry},
    providers::create_provider,
    proxy::{
        AppState,
        hooks::{HOOK_FILTER_ALL, HOOK_MANAGER, HookContext, ResponseData},
        middlewares::RequestModel,
    },
    utils::future::maybe_timeout,
};

pub async fn embeddings(
    State(_state): State<AppState>,
    Extension(mut request_data): Extension<EmbeddingRequest>,
    mut hook_ctx: HookContext,
    mut request: Request,
) -> Result<Response, EmbeddingError> {
    // PRE CALL HOOKS START
    hook_ctx.insert(RequestModel(request_data.model));

    HOOK_MANAGER
        .pre_call(&mut hook_ctx, &mut request, HOOK_FILTER_ALL)
        .await?;
    // PRE CALL HOOKS END

    //TODO: safe unwrap
    let model = hook_ctx.get::<ResourceEntry<Model>>().cloned().unwrap();

    let provider = create_provider(&model.provider_config);
    let timeout = model.timeout.map(Duration::from_millis);

    // Replace request model name with real model name
    request_data.model = model.model.name.clone();

    match maybe_timeout(timeout, provider.embedding(request_data)).await {
        Ok(Ok(mut response)) => {
            response.model = hook_ctx.get::<RequestModel>().cloned().unwrap().0; //TODO: safe unwrap

            // Execute post_call_success hooks
            let response_data = ResponseData::Embedding(response.clone());
            if let Err(err) = HOOK_MANAGER
                .post_call_success(&mut hook_ctx, &response_data, HOOK_FILTER_ALL)
                .await
            {
                error!("Hook post_call_success error: {}", err);
            }

            // Build response and add headers
            let mut resp = Json(response).into_response();
            if let Err(err) = HOOK_MANAGER
                .post_call_headers(&mut hook_ctx, resp.headers_mut(), HOOK_FILTER_ALL)
                .await
            {
                error!("Hook post_call_headers error: {}", err);
            }

            Ok(resp)
        }
        Ok(Err(err)) => {
            error!("Error generating embeddings: {}", err);
            Err(EmbeddingError::ProviderError(err))
        }
        Err(err) => Err(EmbeddingError::Timeout(err)),
    }
}
