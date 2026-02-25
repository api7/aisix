mod models;
mod types;

use std::sync::Arc;

use axum::{Router, routing::get};
use utoipa::OpenApi;
use utoipa_scalar::{Scalar, Servable as ScalarServable};

const PATH_PREFIX: &str = "/aisix/admin";

#[derive(OpenApi)]
#[openapi(info(description = "AI Gateway Admin API"))]
#[openapi(tags(
    (name = models::OPENAPI_TAG, description = "Admin API for managing AI models")
))]
#[openapi(paths(models::list, models::get, models::post, models::put, models::delete))]
struct ApiDoc;

#[derive(Clone)]
pub struct AppState {
    #[allow(dead_code)]
    config: Arc<crate::config::Config>,
    config_provider: Arc<dyn crate::config::ConfigProvider>,
}

impl AppState {
    pub fn new(
        config: crate::config::Config,
        config_provider: Arc<dyn crate::config::ConfigProvider>,
    ) -> Self {
        let config = Arc::new(config);
        Self {
            config,
            config_provider,
        }
    }
}

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .nest(
            PATH_PREFIX,
            Router::new().merge(
                Router::new()
                    .route("/models", get(models::list).post(models::post))
                    .route(
                        "/models/{id}",
                        get(models::get).put(models::put).delete(models::delete),
                    ),
            ),
        )
        .merge(Scalar::with_url("/openapi", ApiDoc::openapi()))
        .with_state(state)
}
