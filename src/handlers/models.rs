use std::time::SystemTime;

use axum::{Json, extract::State};
use serde::Serialize;

use crate::handlers::AppState;

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

#[fastrace::trace]
pub async fn list_models(State(state): State<AppState>) -> Json<ModelList> {
    let created = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_else(|_| std::time::Duration::new(0, 0))
        .as_secs();

    Json(ModelList {
        object: "list",
        data: state
            .resources()
            .models
            .list()
            .values()
            .map(|model| Model {
                id: model.name.clone(),
                object: "model",
                created,
                owned_by: "apisix",
            })
            .collect(),
    })
}
