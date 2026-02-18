use anyhow::Result;
use async_trait::async_trait;
use axum::{
    extract::Request,
    response::{IntoResponse, Response},
};

use crate::{
    config::entities::{ApiKey, ResourceEntry},
    proxy::{
        AppState,
        hooks::{Context, HookAction, ProxyHook},
    },
};

#[derive(Debug)]
pub enum AuthError {
    InvalidApiKey,
}

impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        (axum::http::StatusCode::UNAUTHORIZED, "Unauthorized").into_response()
    }
}

pub struct AuthHook;

impl AuthHook {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl ProxyHook for AuthHook {
    fn name(&self) -> &str {
        "auth"
    }

    async fn pre_call(&self, ctx: &mut Context, request: &mut Request) -> Result<HookAction> {
        let api_key = match request.headers().get(http::header::AUTHORIZATION) {
            Some(value) => {
                let header = value.to_str().unwrap_or("");
                let (prefix, rest) = header.split_at(7.min(header.len()));
                if prefix.eq_ignore_ascii_case("bearer ") {
                    rest
                } else {
                    header
                }
            }
            None => {
                return Ok(HookAction::EarlyReturn(
                    AuthError::InvalidApiKey.into_response(),
                ));
            }
        };

        let state = ctx
            .get::<AppState>()
            .cloned()
            .expect("AppState should be in context");

        let api_key = match state.resources().apikeys.get_by_key(api_key) {
            Some(api_key) => api_key,
            None => {
                return Ok(HookAction::EarlyReturn(
                    AuthError::InvalidApiKey.into_response(),
                ));
            }
        };

        ctx.insert::<ResourceEntry<ApiKey>>(api_key.1);

        Ok(HookAction::Continue)
    }
}
