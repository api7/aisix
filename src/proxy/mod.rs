mod handlers;
mod hooks;
mod middlewares;
mod provider;
pub mod types;

use std::sync::Arc;

use axum::{
    Router,
    extract::DefaultBodyLimit,
    middleware::{from_fn, from_fn_with_state},
    routing::{get, post},
};

use crate::{
    config::{Config, entities::ResourceRegistry},
    gateway::Gateway,
};

#[derive(Clone)]
pub struct AppState {
    #[allow(dead_code)]
    config: Arc<Config>,
    resources: Arc<ResourceRegistry>,
    gateway: Arc<Gateway>,
}

impl AppState {
    pub fn new(
        config: Arc<Config>,
        resources: Arc<ResourceRegistry>,
        gateway: Arc<Gateway>,
    ) -> Self {
        Self {
            config,
            resources,
            gateway,
        }
    }

    pub fn resources(&self) -> Arc<ResourceRegistry> {
        self.resources.clone()
    }

    pub fn gateway(&self) -> Arc<Gateway> {
        self.gateway.clone()
    }
}

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .merge(Router::new().route("/v1/models", get(handlers::models::list_models)))
        .route(
            "/v1/chat/completions",
            post(handlers::chat_completions::chat_completions),
        )
        .route(
            "/v1/messages",
            post(handlers::messages::messages).layer(DefaultBodyLimit::max(32 * 1024 * 1024)),
        )
        .route("/v1/embeddings", post(handlers::embeddings::embeddings))
        .layer(DefaultBodyLimit::max(10 * 1024 * 1024))
        .layer(from_fn_with_state(state.clone(), middlewares::auth))
        .layer(from_fn(middlewares::trace))
        .with_state(state)
}
