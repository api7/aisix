mod types;

use axum::{
    Json,
    extract::{Extension, Request, State},
    response::{IntoResponse, Response},
};
use log::error;

use crate::{
    config::entities::{Model, ResourceEntry},
    providers::create_provider,
    proxy::{
        AppState,
        hooks::{HOOK_FILTER_ALL, HOOK_MANAGER, HookContext, ResponseData},
        middlewares::RequestModel,
    },
};

pub use types::*;

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

    // Replace request model name with real model name
    request_data.model = model.model.name.clone();

    match provider.embedding(request_data).await {
        Ok(mut response) => {
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
        Err(err) => {
            error!("Error generating embeddings: {}", err);
            let err: anyhow::Error = err.into();
            HOOK_MANAGER
                .post_call_failure(&mut hook_ctx, &err, HOOK_FILTER_ALL)
                .await?;
            Err(EmbeddingError::ProviderError(err.to_string()))
        }
    }
}
