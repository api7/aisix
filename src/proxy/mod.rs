use std::sync::Arc;

use axum::{
    Router,
    middleware::from_fn,
    routing::{get, post},
};

mod handlers;
mod hooks;
mod middlewares;

// types
pub mod types {
    pub use super::handlers::{
        chat_completions::{
            ChatCompletionChoice, ChatCompletionChunk, ChatCompletionRequest,
            ChatCompletionResponse, ChatCompletionUsage, ChatMessage,
        },
        embeddings::{EmbeddingRequest, EmbeddingResponse},
    };
}

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
        .merge(Router::new().route("/v1/models", get(handlers::models::list_models)))
        .route(
            "/v1/chat/completions",
            post(handlers::chat_completions::chat_completions).layer(from_fn(
                middlewares::parse_body::<handlers::chat_completions::ChatCompletionRequest>,
            )),
        )
        .route(
            "/v1/embeddings",
            post(handlers::embeddings::embeddings).layer(from_fn(
                middlewares::parse_body::<handlers::embeddings::EmbeddingRequest>,
            )),
        )
        .layer(from_fn(middlewares::trace))
        .with_state(state)
}
