use std::time::SystemTime;

use axum::{
    Json,
    extract::State,
    response::{IntoResponse, Response},
};
use serde::Serialize;
use thiserror::Error;

use crate::{
    config::entities::{ApiKey, ResourceEntry},
    proxy::{AppState, hooks::RequestContext},
};

// Model structure representing a single model
#[derive(Serialize)]
struct Model {
    // [The model identifier, which can be referenced in the API endpoints.](https://platform.openai.com/docs/api-reference/models/object#models-object-id)
    id: String,
    // [The object type, which is always "model".](https://platform.openai.com/docs/api-reference/models/object#models-object-object)
    object: &'static str,
    // [The Unix timestamp (in seconds) when the model was created.](https://platform.openai.com/docs/api-reference/models/object#models-object-created)
    created: u64,
    // [The organization that owns the model.](https://platform.openai.com/docs/api-reference/models/object#models-object-owned_by)
    owned_by: &'static str,
}

// Response structure for listing models
#[derive(Serialize)]
pub struct ModelList {
    // [The object type, which is always "list".](https://platform.openai.com/docs/api-reference/models/list)
    object: &'static str,
    // [The list of models.](https://platform.openai.com/docs/api-reference/models/list)
    data: Vec<Model>,
}

#[derive(Debug, Error)]
pub enum ModelError {}

impl IntoResponse for ModelError {
    fn into_response(self) -> Response {
        match self {}
    }
}

#[fastrace::trace]
pub async fn list_models(
    State(state): State<AppState>,
    request_ctx: RequestContext,
) -> Result<Response, ModelError> {
    let api_key = request_ctx
        .extensions()
        .await
        .get::<ResourceEntry<ApiKey>>()
        .cloned()
        .expect("apikey should exist in context");

    let created = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_else(|_| std::time::Duration::new(0, 0))
        .as_secs();

    Ok(Json(ModelList {
        object: "list",
        data: state
            .resources()
            .models
            .list()
            .values()
            .filter_map(|model| {
                if api_key.allowed_models.contains(&model.name) {
                    Some(Model {
                        id: model.name.clone(),
                        object: "model",
                        created,
                        owned_by: "apisix",
                    })
                } else {
                    None
                }
            })
            .collect(),
    })
    .into_response())
}
