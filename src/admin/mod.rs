mod apikeys;
mod models;
mod types;

use std::sync::Arc;

use axum::{
    Router,
    extract::{Request, State},
    middleware::Next,
    response::{IntoResponse, Response},
    routing::get,
};
use utoipa::{
    Modify, OpenApi,
    openapi::security::{
        ApiKey as OASApiKey, ApiKeyValue as OASApiKeyValue, HttpAuthScheme,
        HttpBuilder as OASHttpBuilder, SecurityScheme,
    },
};
use utoipa_scalar::{Scalar, Servable as ScalarServable};

use crate::admin::types::AuthError;

const PATH_PREFIX: &str = "/aisix/admin";

#[derive(OpenApi)]
#[openapi(
    info(description = "AI Gateway Admin API"),
    modifiers(&SecurityAddon),
    tags(
        (name = models::OPENAPI_TAG, description = "Admin API for managing AI models"),
        (name = apikeys::OPENAPI_TAG, description = "Admin API for managing API keys")
    ),
    security(
        ("bearer" = []),
        ("api_key" = [])
    ),
    paths(
        models::list,
        models::get,
        models::post,
        models::put,
        models::delete,
        apikeys::list,
        apikeys::get,
        apikeys::post,
        apikeys::put,
        apikeys::delete,
    )
)]
struct ApiDoc;

struct SecurityAddon;

impl Modify for SecurityAddon {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        if let Some(components) = openapi.components.as_mut() {
            components.add_security_scheme(
                "bearer",
                SecurityScheme::Http(OASHttpBuilder::new().scheme(HttpAuthScheme::Bearer).build()),
            );
            components.add_security_scheme(
                "api_key",
                SecurityScheme::ApiKey(OASApiKey::Header(OASApiKeyValue::new("x-api-key"))),
            );
        }
    }
}

#[derive(Clone)]
pub struct AppState {
    #[allow(dead_code)]
    config: Arc<crate::config::Config>,
    config_provider: Arc<dyn crate::config::ConfigProvider>,
    resources: Arc<crate::config::entities::ResourceRegistry>,
}

impl AppState {
    pub fn new(
        config: crate::config::Config,
        config_provider: Arc<dyn crate::config::ConfigProvider>,
        resources: Arc<crate::config::entities::ResourceRegistry>,
    ) -> Self {
        let config = Arc::new(config);
        Self {
            config,
            config_provider,
            resources,
        }
    }
}

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .nest(
            PATH_PREFIX,
            Router::new()
                .merge(
                    Router::new()
                        .route("/models", get(models::list).post(models::post))
                        .route(
                            "/models/{id}",
                            get(models::get).put(models::put).delete(models::delete),
                        ),
                )
                .merge(
                    Router::new()
                        .route("/apikeys", get(apikeys::list).post(apikeys::post))
                        .route(
                            "/apikeys/{id}",
                            get(apikeys::get).put(apikeys::put).delete(apikeys::delete),
                        ),
                )
                .layer(axum::middleware::from_fn_with_state(state.clone(), auth)),
        )
        .merge(Scalar::with_url("/openapi", ApiDoc::openapi()))
        .with_state(state)
}

async fn auth(
    State(state): State<AppState>,
    req: Request,
    next: Next,
) -> Result<Response, Response> {
    let api_key = match req.headers().get(http::header::AUTHORIZATION) {
        Some(value) => {
            let header = value.to_str().unwrap_or("");
            let (prefix, rest) = header.split_at(7.min(header.len()));
            if prefix.eq_ignore_ascii_case("bearer ") {
                rest
            } else {
                header
            }
        }
        None => match req.headers().get("x-api-key") {
            Some(value) => value.to_str().unwrap_or(""),
            None => return Err(AuthError::MissingKey.into_response()),
        },
    };

    let admin_keys = match &state.config.deployment.admin {
        Some(admin) => match &admin.admin_key {
            Some(keys) => keys,
            None => return Err(AuthError::MissingKey.into_response()),
        },
        None => return Err(AuthError::MissingKey.into_response()),
    };

    if !admin_keys.iter().any(|item| item.key == api_key) {
        return Err(AuthError::InvalidKey.into_response());
    }

    Ok(next.run(req).await)
}
