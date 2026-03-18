mod handlers;
pub mod hooks;
mod middlewares;

use std::sync::Arc;

use axum::{
    Router,
    middleware::{from_fn, from_fn_with_state},
    routing::{get, post},
};

use crate::config::{Config, entities::ResourceRegistry};

// types
pub mod types {
    pub use super::handlers::{
        chat_completions::{
            ChatCompletionChoice, ChatCompletionChunk, ChatCompletionChunkChoice,
            ChatCompletionChunkDelta, ChatCompletionRequest, ChatCompletionResponse,
            ChatCompletionUsage, ChatMessage,
        },
        embeddings::{EmbeddingRequest, EmbeddingResponse},
    };
}

#[derive(Clone)]
pub struct AppState {
    #[allow(dead_code)]
    config: Arc<Config>,
    resources: Arc<ResourceRegistry>,
}

impl AppState {
    pub fn new(config: Arc<Config>, resources: Arc<ResourceRegistry>) -> Self {
        Self { config, resources }
    }

    pub fn resources(&self) -> Arc<ResourceRegistry> {
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
        .layer(from_fn_with_state(state.clone(), middlewares::auth))
        .layer(from_fn(middlewares::trace))
        .with_state(state)
}
