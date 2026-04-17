//! Bearer-token authentication for the proxy surface.
//!
//! The extractor [`AuthenticatedKey`] parses `Authorization: Bearer <key>`
//! (or `x-api-key: <key>` as a convenience alternative), looks the key
//! up in the current `AisixSnapshot`, and yields the matching `ApiKey`
//! entity. Handlers take `AuthenticatedKey` as an argument — if parsing
//! or lookup fails the request is short-circuited with a 401 envelope
//! before the handler runs.

use aisix_core::resource::ResourceEntry;
use aisix_core::ApiKey;
use axum::extract::{FromRef, FromRequestParts};
use axum::http::request::Parts;
use std::sync::Arc;

use crate::error::ProxyError;
use crate::state::ProxyState;

#[derive(Debug, Clone)]
pub struct AuthenticatedKey {
    pub entry: Arc<ResourceEntry<ApiKey>>,
}

impl AuthenticatedKey {
    pub fn key(&self) -> &ApiKey {
        &self.entry.value
    }
}

#[axum::async_trait]
impl<S> FromRequestParts<S> for AuthenticatedKey
where
    S: Send + Sync,
    ProxyState: FromRef<S>,
{
    type Rejection = ProxyError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let token = extract_bearer(parts)?;
        let proxy_state = ProxyState::from_ref(state);
        let snapshot = proxy_state.snapshot.load();
        let entry = snapshot
            .apikeys
            .get_by_name(&token)
            .ok_or(ProxyError::InvalidApiKey)?;
        Ok(AuthenticatedKey { entry })
    }
}

fn extract_bearer(parts: &Parts) -> Result<String, ProxyError> {
    if let Some(auth) = parts.headers.get(axum::http::header::AUTHORIZATION) {
        let s = auth.to_str().map_err(|_| ProxyError::MissingAuth)?;
        if let Some(rest) = s.strip_prefix("Bearer ") {
            let rest = rest.trim();
            if rest.is_empty() {
                return Err(ProxyError::MissingAuth);
            }
            return Ok(rest.to_string());
        }
        return Err(ProxyError::MissingAuth);
    }
    if let Some(raw) = parts.headers.get("x-api-key") {
        let s = raw.to_str().map_err(|_| ProxyError::MissingAuth)?;
        let s = s.trim();
        if s.is_empty() {
            return Err(ProxyError::MissingAuth);
        }
        return Ok(s.to_string());
    }
    Err(ProxyError::MissingAuth)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::{HeaderMap, HeaderValue, Request};

    fn parts_with(headers: HeaderMap) -> Parts {
        let mut req = Request::builder().uri("/").body(()).unwrap();
        *req.headers_mut() = headers;
        req.into_parts().0
    }

    #[test]
    fn extract_bearer_happy_path() {
        let mut h = HeaderMap::new();
        h.insert(
            axum::http::header::AUTHORIZATION,
            HeaderValue::from_static("Bearer sk-abc"),
        );
        let parts = parts_with(h);
        assert_eq!(extract_bearer(&parts).unwrap(), "sk-abc");
    }

    #[test]
    fn extract_bearer_accepts_x_api_key_as_alternative() {
        let mut h = HeaderMap::new();
        h.insert("x-api-key", HeaderValue::from_static("sk-abc"));
        let parts = parts_with(h);
        assert_eq!(extract_bearer(&parts).unwrap(), "sk-abc");
    }

    #[test]
    fn extract_bearer_rejects_missing_header() {
        let parts = parts_with(HeaderMap::new());
        assert!(matches!(
            extract_bearer(&parts),
            Err(ProxyError::MissingAuth)
        ));
    }

    #[test]
    fn extract_bearer_rejects_wrong_scheme() {
        let mut h = HeaderMap::new();
        h.insert(
            axum::http::header::AUTHORIZATION,
            HeaderValue::from_static("Basic dXNlcjpwdw=="),
        );
        let parts = parts_with(h);
        assert!(matches!(
            extract_bearer(&parts),
            Err(ProxyError::MissingAuth)
        ));
    }

    #[test]
    fn extract_bearer_rejects_empty_bearer() {
        let mut h = HeaderMap::new();
        h.insert(
            axum::http::header::AUTHORIZATION,
            HeaderValue::from_static("Bearer   "),
        );
        let parts = parts_with(h);
        assert!(matches!(
            extract_bearer(&parts),
            Err(ProxyError::MissingAuth)
        ));
    }
}
