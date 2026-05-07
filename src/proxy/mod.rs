mod handlers;
mod hooks;
pub(crate) mod message_history;
mod middlewares;
mod provider;
mod utils;

use std::sync::Arc;

use axum::{
    Router,
    extract::DefaultBodyLimit,
    middleware::{from_fn, from_fn_with_state},
    routing::{get, post},
};

use self::message_history::MessageHistoryStorage;
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
    message_history_storage: Arc<dyn MessageHistoryStorage>,
}

impl AppState {
    pub fn new(
        config: Arc<Config>,
        resources: Arc<ResourceRegistry>,
        gateway: Arc<Gateway>,
        message_history_storage: Arc<dyn MessageHistoryStorage>,
    ) -> Self {
        Self {
            config,
            resources,
            gateway,
            message_history_storage,
        }
    }

    pub fn resources(&self) -> Arc<ResourceRegistry> {
        self.resources.clone()
    }

    pub fn gateway(&self) -> Arc<Gateway> {
        self.gateway.clone()
    }

    pub fn message_history_storage(&self) -> Arc<dyn MessageHistoryStorage> {
        self.message_history_storage.clone()
    }
}

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .merge(Router::new().route("/v1/models", get(handlers::models::list_models)))
        .route(
            "/v1/chat/completions",
            post(
                handlers::format_handler::format_handler::<
                    handlers::chat_completions::ChatCompletionsAdapter,
                >,
            ),
        )
        .route(
            "/v1/messages",
            post(handlers::format_handler::format_handler::<handlers::messages::MessagesAdapter>)
                .layer(DefaultBodyLimit::max(32 * 1024 * 1024)),
        )
        .route(
            "/v1/responses",
            post(handlers::format_handler::format_handler::<handlers::responses::ResponsesAdapter>),
        )
        .route("/v1/embeddings", post(handlers::embeddings::embeddings))
        .layer(DefaultBodyLimit::max(10 * 1024 * 1024))
        .layer(from_fn_with_state(state.clone(), middlewares::auth))
        .layer(from_fn(middlewares::trace))
        .with_state(state)
}
