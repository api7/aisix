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
use aisix_obs::AccessLog;
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
        provider_request_id: None,
        served_by_model: None,
        routing_attempt_count: None,
        routing_fallback_count: None,
        error_kind: Some(error_kind),
        error: Some(&error),
    }
    .emit();
    // `path` must be normalized, not passed through: `AisixPath` below hands
    // this the RAW `parts.uri.path()` so the access log can name the
    // malformed segment, and that string is caller-controlled (#451).
    // `request_metrics` keys the LLM-vs-proxy split off the result, so a 413
    // on /v1/chat/completions lands in the same families as the
    // model-not-found the handler itself records.
    crate::request_metrics::record(
        state,
        crate::normalize_endpoint_label(path),
        crate::request_metrics::Caller::unattributed(api_key_id),
        crate::request_metrics::Upstream {
            provider: UNRESOLVED_PROVIDER_LABEL,
            model: UNRESOLVED_MODEL_LABEL,
            ..Default::default()
        },
        status,
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
            Err(rejection) => {
                // axum classifies wrong parameter arity / unsupported types
                // as 500 — a server wiring bug, not caller input. Keep that
                // loud and unenveloped; only caller-caused 400s are recorded
                // as refused requests below.
                let axum_response = rejection.into_response();
                if axum_response.status() != axum::http::StatusCode::BAD_REQUEST {
                    return Err(axum_response);
                }
                // Fallback mirrors the handlers' own idiom (see mcp.rs /
                // a2a.rs); unreachable in the real router, where
                // `ensure_request_id` runs before routing.
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

    fn state() -> ProxyState {
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
            request_id: Default::default(),
            thread_per_core: None,
            workers: None,
            url_rewrites: Vec::new(),
        };
        ProxyState::new(
            SnapshotHandle::new(snapshot),
            Arc::new(aisix_gateway::Hub::new()),
            &cfg,
        )
        .without_cache()
    }

    fn router() -> axum::Router {
        crate::build_router(state())
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
        let state = state();
        let router = crate::build_router(state.clone());
        for (method, path) in [
            ("POST", "/a2a/%ff"),
            ("GET", "/a2a/%ff/.well-known/agent-card.json"),
            ("GET", "/mcp/%ff"),
            ("GET", "/v1/files/%ff"),
            ("DELETE", "/v1/files/%ff"),
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

        // The refusals are RECORDED, not just enveloped: the chokepoint
        // counts them with the unresolved labels.
        let scrape = state.metrics.render();
        assert!(
            scrape.contains(r#"status="400""#) && scrape.contains(r#"provider="unknown""#),
            "the 400s must be counted with unresolved labels, got: {scrape}"
        );
    }

    #[tokio::test]
    async fn wiring_bugs_keep_their_500_instead_of_blaming_the_caller() {
        // A handler whose tuple arity doesn't match the route's captures is
        // a server wiring bug — axum classifies it 500, and the extractor
        // must pass that through rather than record a caller-caused 400.
        async fn miswired(
            crate::reject::AisixPath(_x): crate::reject::AisixPath<String>,
            axum::extract::State(_): axum::extract::State<ProxyState>,
        ) -> &'static str {
            "unreachable"
        }
        let router = axum::Router::new()
            .route("/wired/:a/:b", axum::routing::get(miswired))
            .with_state(state());
        let (status, body) = send(router, "GET", "/wired/x/y", false).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{body}");
        assert!(
            !body.contains("invalid_request_error"),
            "a wiring bug must not wear the caller envelope: {body}"
        );
    }

    #[tokio::test]
    async fn multipart_content_type_mismatch_answers_the_envelope() {
        // Sending JSON to a multipart endpoint is a common client mistake —
        // the same silent bare-400 class, one extractor over (#880 review
        // follow-up). Every multipart route must answer the envelope.
        let router = router();
        for path in [
            "/v1/audio/transcriptions",
            "/v1/audio/translations",
            "/v1/files",
        ] {
            let request = HttpRequest::post(path)
                .header("authorization", format!("Bearer {TOKEN}"))
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap();
            let response = router
                .clone()
                .oneshot(request)
                .await
                .expect("router responds");
            let status = response.status();
            let bytes = axum::body::to_bytes(response.into_body(), 1 << 16)
                .await
                .unwrap_or_default();
            let body = String::from_utf8_lossy(&bytes);
            assert_eq!(status, StatusCode::BAD_REQUEST, "{path}: {body}");
            assert!(
                body.contains("invalid_request_error"),
                "{path} must answer the OpenAI envelope, got: {body}"
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
