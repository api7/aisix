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
        let proxy_state = ProxyState::from_ref(state);
        let token = match extract_bearer(parts) {
            Ok(t) => t,
            Err(e) => {
                // No credential at all: metric only — logging every
                // scanner probe would be noise (AISIX-Cloud#1081).
                proxy_state
                    .metrics
                    .record_auth_decision("none", false, "missing_credentials");
                return Err(e);
            }
        };
        authenticate_token(&proxy_state, &token).await
    }
}

/// Authenticate a plaintext bearer and enforce key lifecycle. The single
/// auth choke point behind the [`AuthenticatedKey`] extractor; also
/// called directly by surfaces whose credentials arrive outside the
/// standard headers (WebSocket subprotocol auth on `/v1/realtime`).
///
/// A JWT-shaped bearer takes the OIDC path (`crate::jwt`) whenever the
/// environment has an enabled trust provider — with no fallback to the
/// key lookup on failure, so a rejected JWT can never be retried as a
/// key. Everything else is looked up as an API key by hash.
pub(crate) async fn authenticate_token(
    state: &ProxyState,
    token: &str,
) -> Result<AuthenticatedKey, ProxyError> {
    let snapshot = state.snapshot.load();
    if crate::jwt::looks_like_jwt(token) && crate::jwt::any_enabled_provider(&snapshot) {
        return crate::jwt::authenticate_jwt(state, &snapshot, token).await;
        // No enabled trust provider: fall through to the key path so a
        // custom-imported key that happens to look like a JWT keeps
        // authenticating on deployments that never enable JWT auth.
    }
    // Self-hosted CP (prd-09a §9A.7B.4): the snapshot stores
    // SHA-256 hashes of the plaintext bearer, never the plaintext
    // itself. Hash the incoming token via the canonical helper
    // and look up by the hex digest. cp-api hashes with the same
    // function before persistence, so the two sides agree byte
    // for byte.
    let Some(entry) = snapshot.apikeys.get_by_name(&ApiKey::hash_bearer(token)) else {
        return Err(deny_key(state, "unknown_key", ProxyError::InvalidApiKey));
    };
    // Lifecycle enforcement (#933): a known key that is disabled or
    // past its expiry deadline must be rejected here, at the single
    // auth choke point, so every proxy surface (chat, messages,
    // responses, embeddings, audio, passthrough, MCP, …) inherits
    // the same 401 without per-handler checks.
    if entry.value.disabled {
        return Err(deny_key(state, "key_disabled", ProxyError::ApiKeyDisabled));
    }
    if entry.value.is_expired_at(chrono::Utc::now()) {
        return Err(deny_key(state, "key_expired", ProxyError::ApiKeyExpired));
    }
    state.metrics.record_auth_decision("api_key", true, "");
    Ok(AuthenticatedKey { entry })
}

/// Record an API-key denial on the decision metric + log
/// (AISIX-Cloud#1081 — before this, a 401 was invisible: the extractor
/// short-circuits ahead of every handler, so neither the request
/// counter nor the access log ever fired). The token itself is never
/// logged.
///
/// `unknown_key` is the scanner-probe shape (`Bearer <junk>`) and
/// carries no operator signal beyond the metric — same reasoning as the
/// extractor's `missing_credentials`, which is also metric-only. It logs
/// at `debug` so an internet-facing DP is not flooded. `key_disabled` /
/// `key_expired` name a real key an operator provisioned, so they stay
/// at `warn`.
fn deny_key(state: &ProxyState, reason: &'static str, err: ProxyError) -> ProxyError {
    state.metrics.record_auth_decision("api_key", false, reason);
    if reason == "unknown_key" {
        tracing::debug!(
            target: "aisix::auth",
            method = "api_key",
            reason = %reason,
            "rejected inbound credential",
        );
    } else {
        tracing::warn!(
            target: "aisix::auth",
            method = "api_key",
            reason = %reason,
            "rejected inbound credential",
        );
    }
    err
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
