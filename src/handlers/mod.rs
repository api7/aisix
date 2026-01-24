use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Request, State},
    routing::{get, post},
};
use fastrace::prelude::{SpanContext, SpanId, TraceId};
use serde_json::json;

pub mod chat;
mod extractors;
mod middlewares;

#[derive(Clone)]
pub struct AppState {
    config: Arc<crate::config::Config>,
    resources: Arc<crate::config::entities::ResourceRegistry>,
}

impl AppState {
    pub fn new(
        config: crate::config::Config,
        resources: crate::config::entities::ResourceRegistry,
    ) -> Self {
        let config = Arc::new(config);
        let resources = Arc::new(resources);
        Self { config, resources }
    }

    pub fn resources(&self) -> Arc<crate::config::entities::ResourceRegistry> {
        self.resources.clone()
    }
}

async fn trace_id_middleware(
    mut request: Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let traceparent =
        SpanContext::new(TraceId::random(), SpanId::random()).encode_w3c_traceparent();

    println!("Injecting traceparent header: {}", traceparent);

    request.headers_mut().insert(
        fastrace_axum::TRACEPARENT_HEADER,
        traceparent.parse().unwrap(),
    );
    next.run(request).await
}

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat::chat_completions))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            middlewares::auth::auth,
        ))
        .with_state(state.clone())
}

#[fastrace::trace]
async fn list_models(State(state): State<AppState>) -> Json<serde_json::Value> {
    let models = state.resources().models.list();
    /* {
      "object": "list",
      "data": [
        {
          "id": "model-id-0",
          "object": "model",
          "created": 1686935002,
          "owned_by": "organization-owner"
        }
      ],
      "object": "list"
    } */

    Json(
        json!({ "object": "list", "data": models.values().map(|model| {
        json!({ "id": model.name, "object": "model", "owned_by": "apisix" })  }).collect::<Vec<_>>() }),
    )
}
