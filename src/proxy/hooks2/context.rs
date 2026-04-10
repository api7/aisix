use std::ops::{Deref, DerefMut};

use axum::extract::FromRequestParts;
use http::request::Parts;

use crate::{
    config::entities::{ApiKey, ResourceEntry},
    proxy::AppState,
};

pub struct RequestContext {
    #[allow(unused)]
    app_state: AppState,
    extensions: http::Extensions,
}

impl FromRequestParts<AppState> for RequestContext {
    type Rejection = ();

    async fn from_request_parts(
        parts: &mut Parts,
        state: &AppState,
    ) -> Result<Self, Self::Rejection> {
        let mut ctx = http::Extensions::new();
        ctx.insert(
            parts
                .extensions
                .get::<ResourceEntry<ApiKey>>()
                .expect(
                    "Authentication middleware should have inserted ApiKey into request extensions",
                )
                .clone(),
        ); // TODO: remove instand of clone
        Ok(Self {
            app_state: state.clone(),
            extensions: ctx,
        })
    }
}

impl Deref for RequestContext {
    type Target = http::Extensions;

    fn deref(&self) -> &Self::Target {
        &self.extensions
    }
}

impl DerefMut for RequestContext {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.extensions
    }
}

impl RequestContext {
    pub fn app_state(&self) -> &AppState {
        &self.app_state
    }
}
