use std::sync::Arc;

use axum::{
    Router,
    routing::{get, post},
};

use crate::proxy::middlewares;

pub mod chat_completions;
pub mod embeddings;
mod models;

#[derive(Clone)]
pub struct AppState {
    #[allow(dead_code)]
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

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .merge(Router::new().route("/v1/models", get(models::list_models)))
        .route(
            "/v1/chat/completions",
            post(chat_completions::chat_completions)
                .layer(axum::middleware::from_fn(middlewares::rate_limit_check))
                .layer(axum::middleware::from_fn_with_state(
                    state.clone(),
                    middlewares::validate_model::<chat_completions::ChatCompletionRequest>,
                ))
                .layer(axum::middleware::from_fn(
                    middlewares::parse_body::<chat_completions::ChatCompletionRequest>,
                )),
        )
        .route(
            "/v1/embeddings",
            post(embeddings::embeddings)
                .layer(axum::middleware::from_fn(middlewares::rate_limit_check))
                .layer(axum::middleware::from_fn_with_state(
                    state.clone(),
                    middlewares::validate_model::<embeddings::EmbeddingRequest>,
                ))
                .layer(axum::middleware::from_fn(
                    middlewares::parse_body::<embeddings::EmbeddingRequest>,
                )),
        )
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            middlewares::auth,
        ))
        .layer(axum::middleware::from_fn(middlewares::log))
        .layer(middlewares::TraceLayer)
        .with_state(state.clone())
}
