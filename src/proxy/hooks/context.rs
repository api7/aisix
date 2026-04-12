use std::sync::Arc;

use axum::extract::FromRequestParts;
use http::request::Parts;
use tokio::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::{
    config::entities::{ApiKey, ResourceEntry},
    proxy::AppState,
};

struct RequestContextInner {
    app_state: AppState,
    extensions: RwLock<http::Extensions>,
}

#[derive(Clone)]
pub struct RequestContext {
    inner: Arc<RequestContextInner>,
}

impl FromRequestParts<AppState> for RequestContext {
    type Rejection = ();

    async fn from_request_parts(
        parts: &mut Parts,
        state: &AppState,
    ) -> Result<Self, Self::Rejection> {
        let mut ctx = http::Extensions::new();
        ctx.insert(parts.extensions.remove::<ResourceEntry<ApiKey>>().expect(
            "Authentication middleware should have inserted ApiKey into request extensions",
        ));
        Ok(Self {
            inner: Arc::new(RequestContextInner {
                app_state: state.clone(),
                extensions: RwLock::new(ctx),
            }),
        })
    }
}

impl RequestContext {
    pub fn app_state(&self) -> &AppState {
        &self.inner.app_state
    }

    pub async fn extensions(&self) -> RwLockReadGuard<'_, http::Extensions> {
        self.inner.extensions.read().await
    }

    pub async fn extensions_mut(&self) -> RwLockWriteGuard<'_, http::Extensions> {
        self.inner.extensions.write().await
    }
}
