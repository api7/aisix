//! Terminal handling for requests rejected BEFORE dispatch.
//!
//! Every dispatching handler ends by emitting one access-log line plus the
//! request metrics for whatever it did. The paths that give up *before*
//! dispatch — the body-cap middleware's `Content-Length` short-circuit and
//! the body-extractor rejection each handler unwraps at its top — used to
//! `return` a bare response instead, so an oversize request was invisible:
//! a caller saw `413`, the operator saw nothing in the access log and no
//! `aisix_proxy_requests_total` sample. "Client reports 413, gateway has no
//! record of the request" was indistinguishable from the request never
//! arriving.
//!
//! Route every pre-dispatch rejection through [`reject_before_dispatch`] so
//! the family can't drift again: the rendered envelope and the telemetry are
//! produced by the same call.

use std::sync::Arc;
use std::time::Instant;

use aisix_core::{ApiKey, ResourceEntry};
use aisix_obs::{AccessLog, RequestOutcome};
use axum::extract::FromRequestParts;
use axum::http::request::Parts;
use axum::response::{IntoResponse, Response};
use serde::de::DeserializeOwned;

use crate::error::ProxyError;
use crate::request_id::{new_request_id, RequestId};
use crate::state::ProxyState;
use crate::usage_attr::UNRESOLVED_MODEL_LABEL;

/// Metric `provider` label for a rejection that never reached routing.
/// Matches what the handlers' own pre-dispatch error paths (auth, 404)
/// already record, so the series doesn't fork.
const UNRESOLVED_PROVIDER_LABEL: &str = "unknown";

/// Which wire envelope the caller expects. The Anthropic-protocol routes
/// (`/v1/messages`, `/v1/messages/count_tokens`) must answer in Anthropic
/// shape or the Claude SDK can't parse the error (#336) — the rejection
/// path is no exception.
#[derive(Clone, Copy)]
pub(crate) enum Envelope {
    OpenAi,
    Anthropic,
}

/// Emit the access log + request metrics for a request refused before
/// dispatch, and render `err` into the caller's envelope.
///
/// `api_key_id` is `None` for the middleware short-circuit, which runs
/// ahead of authentication — the request is refused on its declared size
/// alone, before any credential is read.
#[allow(clippy::too_many_arguments)]
pub(crate) fn reject_before_dispatch(
    state: &ProxyState,
    method: &str,
    path: &str,
    request_id: &str,
    api_key_id: Option<&str>,
    started: Instant,
    envelope: Envelope,
    err: ProxyError,
) -> Response {
    let status = err.status().as_u16();
    let elapsed = started.elapsed();
    let (error_kind, error) = crate::attempt::access_log_error(&err);
    AccessLog {
        method,
        path,
        status,
        latency: elapsed,
        // Nothing is resolved this early: no upstream was picked, and the
        // body naming the model is exactly what we refused to read.
        provider: None,
        model: None,
        api_key_id,
        prompt_tokens: None,
        completion_tokens: None,
        total_tokens: None,
        request_id,
        served_by_model: None,
        routing_attempt_count: None,
        routing_fallback_count: None,
        error_kind: Some(error_kind),
        error: Some(&error),
    }
    .emit();
    state.metrics.record_request(
        UNRESOLVED_PROVIDER_LABEL,
        UNRESOLVED_MODEL_LABEL,
        status,
        RequestOutcome::from_status(status),
        elapsed,
    );
    match envelope {
        Envelope::OpenAi => err.into_response(),
        Envelope::Anthropic => err.into_anthropic_response(),
    }
}

/// `axum::extract::Path` with the rejection routed through
/// [`reject_before_dispatch`].
///
/// A `:param` segment that fails extraction — invalid percent-encoding such
/// as `/v1/files/%ff` — otherwise answers axum's bare 400: no access log,
/// no request metrics, no caller envelope (#880, the same silent class
/// #863 collected for body rejections). Every handler on a `:param` route
/// takes this instead of `Path`, so the family can't drift back.
///
/// Declared after `auth: AuthenticatedKey` in handler signatures, like
/// `Path` was — extractors run in order, so authentication still precedes
/// the path parse (an unauthenticated caller gets 401, not a 400 that
/// confirms anything about the route) and the resolved key published by the
/// auth extractor attributes the rejection.
pub(crate) struct AisixPath<T>(pub(crate) T);

#[axum::async_trait]
impl<T> FromRequestParts<ProxyState> for AisixPath<T>
where
    T: DeserializeOwned + Send + 'static,
{
    type Rejection = Response;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &ProxyState,
    ) -> Result<Self, Self::Rejection> {
        match axum::extract::Path::<T>::from_request_parts(parts, state).await {
            Ok(axum::extract::Path(value)) => Ok(Self(value)),
            Err(_) => {
                let request_id = parts
                    .extensions
                    .get::<RequestId>()
                    .map(|r| r.0.clone())
                    .unwrap_or_else(new_request_id);
                let api_key_id = parts
                    .extensions
                    .get::<Arc<ResourceEntry<ApiKey>>>()
                    .map(|entry| entry.id.clone());
                // The raw path, not a route template: the malformed segment
                // IS the subject of this rejection, and the access log is
                // per-request (the bounded labels live in the metrics).
                Err(reject_before_dispatch(
                    state,
                    parts.method.as_str(),
                    parts.uri.path(),
                    &request_id,
                    api_key_id.as_deref(),
                    Instant::now(),
                    Envelope::OpenAi,
                    ProxyError::InvalidRequest("invalid path parameter".into()),
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use aisix_core::snapshot::SnapshotHandle;
    use aisix_core::{AisixSnapshot, ApiKey, ProxyConfig, ResourceEntry};
    use axum::body::Body;
    use axum::http::{Request as HttpRequest, StatusCode};
    use std::sync::Arc;
    use tower::ServiceExt;

    use crate::state::ProxyState;

    const TOKEN: &str = "sk-path-reject-test";

    fn router() -> axum::Router {
        let apikey: ApiKey = serde_json::from_value(serde_json::json!({
            "key_hash": ApiKey::hash_bearer(TOKEN),
            "allowed_models": ["*"],
        }))
        .expect("valid apikey");
        let snapshot = AisixSnapshot::new();
        snapshot
            .apikeys
            .insert(ResourceEntry::new("ak-1", apikey, 1));
        let cfg = ProxyConfig {
            addr: "127.0.0.1:0".into(),
            request_body_limit_bytes: 0,
            tls: None,
            real_ip: Default::default(),
            url_rewrites: Vec::new(),
        };
        let state = ProxyState::new(
            SnapshotHandle::new(snapshot),
            Arc::new(aisix_gateway::Hub::new()),
            &cfg,
        )
        .without_cache();
        crate::build_router(state)
    }

    async fn send(
        router: axum::Router,
        method: &str,
        path: &str,
        auth: bool,
    ) -> (StatusCode, String) {
        let mut builder = HttpRequest::builder().method(method).uri(path);
        if auth {
            builder = builder.header("authorization", format!("Bearer {TOKEN}"));
        }
        let response = router
            .oneshot(builder.body(Body::empty()).unwrap())
            .await
            .expect("router responds");
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 16)
            .await
            .unwrap_or_default();
        (status, String::from_utf8_lossy(&bytes).into_owned())
    }

    #[tokio::test]
    async fn malformed_path_params_answer_the_openai_envelope_across_the_family() {
        // `%ff` is valid percent-encoding but invalid UTF-8 after decoding —
        // the `Path` extractor rejects it. Pre-#880 that was axum's bare 400
        // text; every `:param` route must now answer the caller envelope
        // (and, mechanically via `reject_before_dispatch`, emit the access
        // log + metrics every other pre-dispatch rejection gets).
        let router = router();
        for (method, path) in [
            ("POST", "/a2a/%ff"),
            ("GET", "/a2a/%ff/.well-known/agent-card.json"),
            ("GET", "/mcp/%ff"),
            ("GET", "/v1/files/%ff"),
            ("GET", "/v1/files/%ff/content"),
            ("GET", "/v1/batches/%ff"),
            ("POST", "/v1/batches/%ff/cancel"),
            ("GET", "/v1/fine_tuning/jobs/%ff"),
            ("POST", "/v1/fine_tuning/jobs/%ff/cancel"),
            ("GET", "/v1/videos/%ff"),
            ("GET", "/v1/videos/%ff/content"),
            ("POST", "/passthrough/%ff/v1/chat"),
        ] {
            let (status, body) = send(router.clone(), method, path, true).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{method} {path}: {body}");
            assert!(
                body.contains("invalid_request_error"),
                "{method} {path} must answer the OpenAI envelope, got: {body}"
            );
        }
    }

    #[tokio::test]
    async fn auth_still_precedes_the_path_parse() {
        // Extractor order is unchanged: an unauthenticated caller gets 401,
        // not a 400 that reveals how the path would have parsed.
        let router = router();
        let (status, _) = send(router, "GET", "/v1/files/%ff", false).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }
}
