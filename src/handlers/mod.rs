use std::sync::Arc;

use axum::{
    Router,
    routing::{get, post},
};

pub mod chat_completions;
pub mod embeddings;
mod extractors;
mod middlewares;
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
            post(chat_completions::chat_completions),
        )
        .route("/v1/embeddings", post(embeddings::embeddings))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            middlewares::auth,
        ))
        .layer(axum::middleware::from_fn(middlewares::log))
        .layer(middlewares::TraceLayer)
        .with_state(state.clone())
}
