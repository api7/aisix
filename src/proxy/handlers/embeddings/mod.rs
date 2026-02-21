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
        hooks::{Context, HOOK_MANAGER, HookAction, ResponseData},
        middlewares::RequestModel,
    },
};

pub use types::*;

pub async fn embeddings(
    State(state): State<AppState>,
    Extension(mut request_data): Extension<EmbeddingRequest>,
    mut request: Request,
) -> Response {
    // PRE CALL HOOKS START
    let mut hook_ctx = Context::new();

    hook_ctx.insert(state);
    hook_ctx.insert(RequestModel(request_data.model));

    let action = HOOK_MANAGER
        .execute_pre_call(&mut hook_ctx, &mut request)
        .await;

    match action {
        Ok(HookAction::EarlyReturn(response)) => {
            return response;
        }
        Err(err) => {
            error!("Hook pre_call error: {}", err);
            return (EmbeddingError::InternalError(err.to_string())).into_response();
        }
        _ => {}
    }

    // PRE CALL HOOKS END

    //TODO: safe unwrap
    let model = hook_ctx.get::<ResourceEntry<Model>>().cloned().unwrap();

    let provider = create_provider(&model.provider_config);

    // Replace request model name with real model name
    request_data.model = model.model.split("/").nth(1).unwrap().to_string();

    match provider.embedding(request_data).await {
        Ok(mut response) => {
            response.model = hook_ctx.get::<RequestModel>().cloned().unwrap().0; //TODO: safe unwrap

            // Execute post_call_success hooks
            let response_data = ResponseData::Embedding(response.clone());
            if let Err(err) = HOOK_MANAGER
                .execute_post_call_success(&mut hook_ctx, &response_data)
                .await
            {
                error!("Hook post_call_success error: {}", err);
            }

            // Build response and add headers
            let mut resp = Json(response).into_response();
            if let Err(err) = HOOK_MANAGER
                .execute_post_call_headers(&hook_ctx, resp.headers_mut())
                .await
            {
                error!("Hook post_call_headers error: {}", err);
            }

            resp
        }
        Err(err) => {
            error!("Error generating embeddings: {}", err);
            (EmbeddingError::ProviderError(err.to_string())).into_response()
        }
    }
}
