//! Admin-key bearer auth. Distinct from the proxy's API-key auth:
//!
//! - Admin keys come from `config.admin.admin_keys` (static, bootstrap
//!   config), not the `ApiKey` table in etcd.
//! - Presentation matches OpenAI convention for symmetry —
//!   `Authorization: Bearer <key>` with an `x-api-key` fallback.
//!
//! This extractor short-circuits with an `AdminError::Unauthorized`
//! envelope before any handler runs.

use axum::extract::{FromRef, FromRequestParts};
use axum::http::request::Parts;

use crate::error::AdminError;
use crate::state::AdminState;

/// Marker yielded by the extractor once an admin key has been verified.
/// Handlers don't need the key itself — just proof that the caller
/// supplied a valid one — so the type is empty by design.
#[derive(Debug, Clone, Copy)]
pub struct AdminAuth;

#[axum::async_trait]
impl<S> FromRequestParts<S> for AdminAuth
where
    S: Send + Sync,
    AdminState: FromRef<S>,
{
    type Rejection = AdminError;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let token = extract_bearer(parts)?;
        let admin_state = AdminState::from_ref(state);
        let is_authorized = admin_state.admin_keys.iter().any(|k| k == &token);
        if !is_authorized {
            return Err(AdminError::Unauthorized);
        }
        Ok(AdminAuth)
    }
}

fn extract_bearer(parts: &Parts) -> Result<String, AdminError> {
    if let Some(auth) = parts.headers.get(axum::http::header::AUTHORIZATION) {
        let s = auth.to_str().map_err(|_| AdminError::Unauthorized)?;
        if let Some(rest) = s.strip_prefix("Bearer ") {
            let rest = rest.trim();
            if rest.is_empty() {
                return Err(AdminError::Unauthorized);
            }
            return Ok(rest.to_string());
        }
        return Err(AdminError::Unauthorized);
    }
    if let Some(raw) = parts.headers.get("x-api-key") {
        let s = raw.to_str().map_err(|_| AdminError::Unauthorized)?;
        let s = s.trim();
        if s.is_empty() {
            return Err(AdminError::Unauthorized);
        }
        return Ok(s.to_string());
    }
    Err(AdminError::Unauthorized)
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
    fn extract_bearer_reads_authorization_header() {
        let mut h = HeaderMap::new();
        h.insert(
            axum::http::header::AUTHORIZATION,
            HeaderValue::from_static("Bearer admin-secret"),
        );
        assert_eq!(extract_bearer(&parts_with(h)).unwrap(), "admin-secret");
    }

    #[test]
    fn extract_bearer_accepts_x_api_key_fallback() {
        let mut h = HeaderMap::new();
        h.insert("x-api-key", HeaderValue::from_static("admin-secret"));
        assert_eq!(extract_bearer(&parts_with(h)).unwrap(), "admin-secret");
    }

    #[test]
    fn extract_bearer_rejects_missing_and_wrong_scheme() {
        assert!(matches!(
            extract_bearer(&parts_with(HeaderMap::new())),
            Err(AdminError::Unauthorized)
        ));

        let mut h = HeaderMap::new();
        h.insert(
            axum::http::header::AUTHORIZATION,
            HeaderValue::from_static("Basic Zm9v"),
        );
        assert!(matches!(
            extract_bearer(&parts_with(h)),
            Err(AdminError::Unauthorized)
        ));
    }
}
