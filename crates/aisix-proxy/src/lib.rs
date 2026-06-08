//! aisix-proxy — client-facing proxy router (`:3000`).
//!
//! Mounts the OpenAI-compatible surface:
//! - `GET  /livez`
//! - `POST /v1/chat/completions` (streaming + non-streaming)
//!
//! Handlers run behind the [`AuthenticatedKey`] extractor which reads
//! the Bearer token (or `x-api-key` fallback) and looks the key up in
//! the current [`AisixSnapshot`]. Model authorisation is enforced per
//! request against `ApiKey::allowed_models`. Upstream calls are
//! dispatched through the [`aisix_gateway::Hub`] to the registered
//! `Bridge` for the Model's provider.
//!
//! Errors surface as OpenAI-style envelopes:
//!
//! ```json
//! {"error":{"message":"…","type":"…"}}
//! ```
//!
//! Status codes follow [`crate::error::ProxyError::status`] — spec §3 auth
//! rules (401/403), `Bridge` mapping preserves upstream 4xx and collapses
//! upstream 5xx to 502.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

mod attempt;
mod audio;
mod auth;
pub mod background;
pub mod budget;
mod chat;
mod client_ip;
mod completions;
pub(crate) mod cooldown;
mod count_tokens;
mod dispatch;
mod embeddings;
mod error;
mod error_translate;
pub mod health;
mod http_client;
mod images;
mod messages;
mod models;
mod passthrough;
mod quota;
mod render;
mod request_id;
mod rerank;
mod responses;
mod routing;
mod state;
mod util;

pub use auth::AuthenticatedKey;
pub use error::{ErrorEnvelope, ProxyError};
pub use health::{
    HealthTracker, LivezState, ModelRuntimeStatusTracker, RuntimeStatus, RuntimeStatusSnapshot,
};
pub use state::ProxyState;

use axum::extract::State;
use axum::http::{header, HeaderValue, Request};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get, post};
use axum::Router;
use tower_http::set_header::SetResponseHeaderLayer;

/// Product token emitted in the `Server` response header. Format follows
/// RFC 9110 §10.2.4 (`product/version`) and matches the convention used
/// by adjacent gateways (APISIX, nginx, kong). Version is the workspace
/// crate version, baked in at compile time.
const SERVER_HEADER_VALUE: &str = concat!("AISIX/", env!("CARGO_PKG_VERSION"));

/// Build the proxy router. Mounts `/livez` plus the
/// OpenAI-compatible proxy surface.
pub fn build_router(state: ProxyState) -> Router {
    let body_limit = state.request_body_limit_bytes;
    Router::new()
        .route("/livez", get(livez))
        .route("/v1/models", get(models::list_models))
        .route("/v1/chat/completions", post(chat::chat_completions))
        .route("/v1/completions", post(completions::completions))
        .route("/v1/embeddings", post(embeddings::embeddings))
        .route("/v1/images/generations", post(images::image_generations))
        .route("/v1/messages", post(messages::messages))
        .route(
            "/v1/messages/count_tokens",
            post(count_tokens::count_tokens),
        )
        .route("/v1/rerank", post(rerank::rerank))
        .route("/v1/responses", post(responses::responses))
        .route("/v1/audio/transcriptions", post(audio::transcriptions))
        .route("/v1/audio/translations", post(audio::translations))
        .route("/v1/audio/speech", post(audio::speech))
        .route(
            "/passthrough/:provider/*rest",
            any(passthrough::passthrough),
        )
        // Wire the configured cap into axum's request-body extractor
        // chain (`Json<T>` defers to `Bytes` which honors this layer).
        // Without this, axum 0.7's `DefaultBodyLimit` falls back to
        // its built-in 2 MiB default, which silently rejects bodies
        // in the 2 MiB-to-cap band with a stock `BytesRejection`
        // response (NOT the OpenAI envelope). The middleware below
        // catches the Content-Length-known case ahead of the
        // extractor; this layer catches chunked / size-mismatched
        // bodies once their actual byte count exceeds the cap.
        .layer(axum::extract::DefaultBodyLimit::max(body_limit))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            enforce_request_body_limit,
        ))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            record_in_flight_request,
        ))
        // Identify the data plane on every response, including error
        // envelopes and short-circuited responses from the layers
        // above. `overriding` (vs `if_not_present`) ensures the
        // gateway's identity is authoritative — any Server header set
        // by inner handlers or accidentally proxied from upstream is
        // replaced, so client-visible Server never leaks provider
        // identity.
        .layer(SetResponseHeaderLayer::overriding(
            header::SERVER,
            HeaderValue::from_static(SERVER_HEADER_VALUE),
        ))
        .with_state(state)
}

async fn record_in_flight_request(
    State(state): State<ProxyState>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    // Normalize to a bounded route template BEFORE using the path as a
    // metric label. This middleware runs before authentication and before
    // route matching, so the raw `request.uri().path()` is fully
    // attacker-controlled — the `/passthrough/:provider/*rest` wildcard
    // suffix (or any 404 path) would otherwise let an unauthenticated
    // caller mint unbounded Prometheus time series (#451).
    let endpoint = normalize_endpoint_label(request.uri().path());
    let inbound_protocol = inbound_protocol_for_endpoint(endpoint).to_string();
    let _guard = InFlightGuard::new(
        state.metrics.clone(),
        endpoint.to_string(),
        inbound_protocol,
    );
    next.run(request).await
}

/// Collapse a raw request path to a fixed route template so metric labels
/// stay bounded regardless of caller-supplied path segments. Keep this
/// allowlist in sync with the routes registered in `build_router`; any
/// unrecognized path (including unmatched 404s) maps to `"other"`.
fn normalize_endpoint_label(path: &str) -> &'static str {
    match path {
        "/livez" => "/livez",
        "/v1/models" => "/v1/models",
        "/v1/chat/completions" => "/v1/chat/completions",
        "/v1/completions" => "/v1/completions",
        "/v1/embeddings" => "/v1/embeddings",
        "/v1/images/generations" => "/v1/images/generations",
        "/v1/messages" => "/v1/messages",
        "/v1/messages/count_tokens" => "/v1/messages/count_tokens",
        "/v1/rerank" => "/v1/rerank",
        "/v1/responses" => "/v1/responses",
        "/v1/audio/transcriptions" => "/v1/audio/transcriptions",
        "/v1/audio/translations" => "/v1/audio/translations",
        "/v1/audio/speech" => "/v1/audio/speech",
        _ if path.starts_with("/passthrough/") => "/passthrough/:provider/*rest",
        _ => "other",
    }
}

fn inbound_protocol_for_endpoint(endpoint: &str) -> &'static str {
    if endpoint == "/v1/messages" || endpoint == "/v1/messages/count_tokens" {
        "anthropic"
    } else {
        "openai"
    }
}

struct InFlightGuard {
    metrics: std::sync::Arc<aisix_obs::Metrics>,
    endpoint: String,
    inbound_protocol: String,
}

impl InFlightGuard {
    fn new(
        metrics: std::sync::Arc<aisix_obs::Metrics>,
        endpoint: String,
        inbound_protocol: String,
    ) -> Self {
        metrics.increment_proxy_in_flight(&endpoint, &inbound_protocol);
        Self {
            metrics,
            endpoint,
            inbound_protocol,
        }
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.metrics
            .decrement_proxy_in_flight(&self.endpoint, &self.inbound_protocol);
    }
}

/// Per RFC 9110 §15.5.14, a request body that exceeds the gateway's
/// configured `request_body_limit_bytes` must surface as a clean
/// `413 Content Too Large` response — NOT an `ECONNRESET` from a
/// mid-write socket close. This middleware inspects the inbound
/// `Content-Length` header before any handler runs and short-circuits
/// with the OpenAI-shape error envelope when the declared size
/// exceeds the cap.
///
/// Bodies sent with chunked transfer encoding (no Content-Length)
/// fall through to handler-level body extraction, which still
/// enforces the limit but with the slower fail mode (the read errors
/// once the cap is hit). Catching the Content-Length-known case here
/// is the load-bearing user-visible win: the OpenAI Node SDK and
/// `fetch` both set Content-Length for non-streamed POSTs, and
/// without this middleware they see ECONNRESET (indistinguishable
/// from a network failure or a gateway crash) instead of 413.
async fn enforce_request_body_limit(
    State(state): State<ProxyState>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    // /v1/messages must emit the Anthropic-shape error envelope
    // (closes #336). The middleware runs BEFORE the handler so the
    // handler's `into_anthropic_response()` would never see the
    // rejection — capture the inbound path here and use it to pick
    // the envelope shape on the reject paths below. Captured as
    // `bool` rather than holding a borrow into `request`, so the
    // `request.into_body()` move on the drain path doesn't conflict
    // with the captured value (audit HIGH-3 follow-up).
    //
    // Audit LOW-A (3rd audit): `/v1/messages/` (trailing slash) also
    // routes to the Anthropic handler via axum's path normalization,
    // but an exact-match check would miss it. The official Anthropic
    // SDK never appends a trailing slash so real-world exposure is
    // near-zero, but non-SDK callers (curl, custom clients) could
    // hit it. Accept both forms.
    let path = request.uri().path();
    let is_anthropic_path =
        path == "/v1/messages" || path == "/v1/messages/" || path == "/v1/messages/count_tokens";
    let render = |e: ProxyError| -> Response {
        if is_anthropic_path {
            e.into_anthropic_response()
        } else {
            e.into_response()
        }
    };
    // RFC 9110 §8.6 — a server SHOULD reject a request that carries
    // duplicate or conflicting `Content-Length` values rather than
    // act on the first one (which is a request-smuggling vector).
    let mut content_lengths = request
        .headers()
        .get_all(axum::http::header::CONTENT_LENGTH)
        .iter();
    let first = content_lengths.next();
    if content_lengths.next().is_some() {
        return render(ProxyError::InvalidRequest(
            "conflicting Content-Length headers".into(),
        ));
    }
    if let Some(declared) = first
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<usize>().ok())
    {
        if declared > state.request_body_limit_bytes {
            // Drain the inbound body so hyper can flush the 413 response
            // on the same HTTP/1.1 connection. Without this, hyper closes
            // the socket while the client is still writing, and the client
            // sees EPIPE/ECONNRESET instead of the 413.
            drain_body(request.into_body()).await;
            return render(ProxyError::RequestTooLarge {
                limit_bytes: state.request_body_limit_bytes,
            });
        }
    }
    next.run(request).await
}

/// Read and discard the inbound body, bounded by both bytes and time.
///
/// Byte cap (32 MiB) prevents a huge `Content-Length` from consuming
/// unbounded memory.  Time cap (5 s) prevents a slowloris-style
/// client from holding the task indefinitely by dribbling data.
async fn drain_body(body: axum::body::Body) {
    use http_body_util::BodyExt;

    const DRAIN_CAP: usize = 32 * 1024 * 1024;
    const DRAIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

    let _ = tokio::time::timeout(DRAIN_TIMEOUT, async {
        let mut drained = 0usize;
        let mut body = body;
        while let Some(Ok(frame)) = body.frame().await {
            if let Some(data) = frame.data_ref() {
                drained += data.len();
                if drained >= DRAIN_CAP {
                    break;
                }
            }
        }
    })
    .await;
}

async fn livez(
    State(state): State<ProxyState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Response {
    crate::health::livez_response(&state.livez, params.contains_key("verbose"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_label_is_bounded_for_arbitrary_paths() {
        // Known routes pass through unchanged.
        assert_eq!(
            normalize_endpoint_label("/v1/chat/completions"),
            "/v1/chat/completions"
        );
        assert_eq!(normalize_endpoint_label("/v1/messages"), "/v1/messages");
        // Passthrough collapses regardless of provider/suffix — this is the
        // unbounded-cardinality vector from #451.
        assert_eq!(
            normalize_endpoint_label("/passthrough/openai/anything/unique-123"),
            "/passthrough/:provider/*rest"
        );
        assert_eq!(
            normalize_endpoint_label("/passthrough/openai/other-unique-456"),
            "/passthrough/:provider/*rest"
        );
        // Arbitrary unauthenticated paths bucket to a single label.
        assert_eq!(normalize_endpoint_label("/random/x"), "other");
        assert_eq!(normalize_endpoint_label("/random/y"), "other");
    }

    use aisix_core::resource::ResourceEntry;
    use aisix_core::snapshot::SnapshotHandle;
    use aisix_core::{AisixSnapshot, ApiKey, Model, ProxyConfig};
    use aisix_gateway::{Hub, SseDecoder, SseEvent};
    use aisix_provider_openai::OpenAiBridge;
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use futures::StreamExt;
    use reqwest::Client;
    use std::sync::Arc;
    use tower::ServiceExt;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn cfg() -> ProxyConfig {
        ProxyConfig {
            addr: "127.0.0.1:0".into(),
            request_body_limit_bytes: 1_048_576,
            real_ip: Default::default(),
            tls: None,
        }
    }

    fn openai_test_bridge() -> OpenAiBridge {
        let client = Client::builder()
            .user_agent("aisix-test/0.1")
            .no_proxy()
            .build()
            .unwrap();
        OpenAiBridge::with_client(client)
    }

    /// State used by the *existing* tests — cache disabled so the
    /// rate-limit / wiremock cases still see every request reach the
    /// upstream. Cache-specific tests build state with the default
    /// constructor (which keeps caching on) instead.
    fn build_state(snapshot: AisixSnapshot, hub: Arc<Hub>) -> ProxyState {
        let handle = SnapshotHandle::new(snapshot);
        ProxyState::new(handle, hub, &cfg()).without_cache()
    }

    fn build_state_with_cache(snapshot: AisixSnapshot, hub: Arc<Hub>) -> ProxyState {
        let handle = SnapshotHandle::new(snapshot);
        ProxyState::new(handle, hub, &cfg())
    }

    const PK_ID: &str = "11111111-1111-1111-1111-111111111111";

    fn model_entry(name: &str) -> ResourceEntry<Model> {
        let cfg = format!(
            r#"{{
                "display_name": "{name}",
                "provider": "openai",
                "model_name": "gpt-4o",
                "provider_key_id": "{PK_ID}"
            }}"#
        );
        let model: Model = serde_json::from_str(&cfg).unwrap();
        ResourceEntry::new("model-id-1", model, 1)
    }

    fn provider_key_entry(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        let cfg = format!(
            r#"{{"display_name":"openai-up","secret":"sk-upstream","api_base":"{api_base}","provider":"openai","adapter":"openai"}}"#
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&cfg).unwrap();
        ResourceEntry::new(PK_ID, pk, 1)
    }

    fn new_snap(api_base: &str) -> AisixSnapshot {
        let snap = AisixSnapshot::new();
        snap.provider_keys.insert(provider_key_entry(api_base));
        snap
    }

    fn apikey_entry(key: &str, allowed: &[&str]) -> ResourceEntry<ApiKey> {
        apikey_entry_with_limits(key, allowed, None)
    }

    fn apikey_entry_with_limits(
        key: &str,
        allowed: &[&str],
        rate_limit: Option<serde_json::Value>,
    ) -> ResourceEntry<ApiKey> {
        let allowed_json = serde_json::to_string(&allowed).unwrap();
        let rl_tail = match rate_limit {
            Some(v) => format!(", \"rate_limit\": {v}"),
            None => String::new(),
        };
        // Tests pass the plaintext bearer here (e.g. "sk-caller"); the
        // wire schema stores its SHA-256 (§9A.7B.4). Hash via the
        // canonical helper so request-side `Bearer <plaintext>` lookups
        // line up.
        let key_hash = ApiKey::hash_bearer(key);
        let cfg =
            format!(r#"{{"key_hash": "{key_hash}", "allowed_models": {allowed_json}{rl_tail}}}"#);
        let apikey: ApiKey = serde_json::from_str(&cfg).unwrap();
        ResourceEntry::new("key-id-1", apikey, 1)
    }

    fn seed_snapshot(model: &str, allowed: &[&str], api_base: &str) -> AisixSnapshot {
        let snap = new_snap(api_base);
        snap.models.insert(model_entry(model));
        snap.apikeys.insert(apikey_entry("sk-caller", allowed));
        snap
    }

    /// Seed an env-scope keyword guardrail into a live snapshot handle.
    ///
    /// Uses `handle.rcu()` to atomically replace the snapshot and bump the
    /// version counter. `LiveGuardrailIndex` compares versions on every
    /// `resolve()` call; without the bump it would return a stale (empty)
    /// index regardless of when this helper is called relative to
    /// `build_state`. With the bump the index rebuilds on the next request,
    /// making the call-order invariant.
    ///
    /// `guardrail_json` must be a valid inline `Guardrail` JSON payload
    /// (same wire shape as `/aisix/<env>/guardrails/<uuid>`).
    /// A single env-scope attachment is inserted alongside it so the
    /// guardrail fires on every request.
    fn seed_guardrail(
        handle: &SnapshotHandle<AisixSnapshot>,
        guardrail_id: &str,
        guardrail_json: &str,
    ) {
        use aisix_core::models::{Guardrail as DomainGuardrail, GuardrailAttachment};
        let gid = guardrail_id.to_string();
        let row: DomainGuardrail = serde_json::from_str(guardrail_json).unwrap();
        let att: GuardrailAttachment = serde_json::from_str(&format!(
            r#"{{"guardrail_id": "{gid}", "scope_type": "env", "priority": 50}}"#
        ))
        .unwrap();
        // rcu: load current snapshot → clone it → insert guardrail entries →
        // store the new snapshot and bump the version. The closure is
        // idempotent: re-inserting the same id merely overwrites with
        // identical data, so retries under contention are safe.
        handle.rcu(|snap| {
            let new_snap = snap.clone();
            new_snap
                .guardrails
                .insert(ResourceEntry::new(gid.clone(), row.clone(), 1));
            new_snap.guardrail_attachments.insert(ResourceEntry::new(
                format!("att-{gid}"),
                att.clone(),
                1,
            ));
            new_snap
        });
    }

    /// Insert a default-enabled cache policy on the snapshot so the
    /// proxy's cache gate (chat::dispatch) opens the lookup path.
    /// Stage 2 honors existence + `enabled`; Stage 3 honors
    /// `applies_to`. The default `applies_to=all` (set by serde
    /// when omitted) matches every request, so existing tests that
    /// seed a bare policy keep passing.
    fn seed_cache_policy(snap: &AisixSnapshot, name: &str) {
        seed_cache_policy_with_applies_to(snap, name, "all");
    }

    /// Like `seed_cache_policy` but with a specific `applies_to`
    /// clause — used by the Stage 3 tests that pin the matcher's
    /// behaviour on `model:<name>` / `api_key:<id>`.
    fn seed_cache_policy_with_applies_to(snap: &AisixSnapshot, name: &str, applies_to: &str) {
        let cfg =
            format!(r#"{{"name": "{name}", "backend": "memory", "applies_to": "{applies_to}"}}"#,);
        let policy: aisix_core::models::CachePolicy = serde_json::from_str(&cfg).unwrap();
        snap.cache_policies
            .insert(ResourceEntry::new(format!("cp-id-{name}"), policy, 1));
    }

    /// Disabled-policy seeder for #154 regression coverage. Posts a
    /// `CachePolicy{enabled: false, applies_to: "all"}` so the
    /// cache-gate predicate at chat.rs (`entry.value.enabled && ...`)
    /// must skip it.
    fn seed_cache_policy_disabled(snap: &AisixSnapshot, name: &str) {
        let cfg = format!(
            r#"{{"name": "{name}", "backend": "memory", "applies_to": "all", "enabled": false}}"#,
        );
        let policy: aisix_core::models::CachePolicy = serde_json::from_str(&cfg).unwrap();
        snap.cache_policies
            .insert(ResourceEntry::new(format!("cp-id-{name}"), policy, 1));
    }

    fn seed_snapshot_with_limits(
        model: &str,
        allowed: &[&str],
        api_base: &str,
        rate_limit: serde_json::Value,
    ) -> AisixSnapshot {
        let snap = new_snap(api_base);
        snap.models.insert(model_entry(model));
        snap.apikeys.insert(apikey_entry_with_limits(
            "sk-caller",
            allowed,
            Some(rate_limit),
        ));
        snap
    }

    async fn run(app: Router, req: Request<Body>) -> axum::http::Response<Body> {
        app.oneshot(req).await.unwrap()
    }

    #[tokio::test]
    async fn non_streaming_happy_path_returns_openai_shaped_json() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-upstream",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
            })))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hello"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["content"], "hi");
        assert_eq!(v["usage"]["total_tokens"], 3);
    }

    #[tokio::test]
    async fn missing_authorization_returns_401_envelope() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"my-gpt4","messages":[]}"#))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "invalid_api_key");
    }

    #[tokio::test]
    async fn livez_reports_plain_ok_by_default() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("GET")
            .uri("/livez")
            .body(Body::empty())
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        assert_eq!(std::str::from_utf8(&bytes).unwrap(), "ok");
    }

    /// Every response — including success bodies, error envelopes, and
    /// short-circuited middleware rejections — must carry the gateway's
    /// `Server` product token (`AISIX/<semver>`) so clients can identify
    /// the data plane without round-tripping to a status endpoint.
    #[tokio::test]
    async fn server_header_identifies_the_data_plane() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        // Success path — plain handler response.
        let ok_req = Request::builder()
            .method("GET")
            .uri("/livez")
            .body(Body::empty())
            .unwrap();
        let ok_resp = run(app.clone(), ok_req).await;
        let ok_server = ok_resp
            .headers()
            .get(axum::http::header::SERVER)
            .expect("success response must carry Server header")
            .to_str()
            .unwrap()
            .to_string();
        assert!(
            ok_server.starts_with("AISIX/") && ok_server.len() > "AISIX/".len(),
            "expected `AISIX/<version>`, got {ok_server:?}"
        );

        // Error path — auth failure envelope. Same Server header must
        // appear so error responses don't accidentally hide the gateway's
        // identity (and don't leak any upstream Server token).
        let unauth_req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"my-gpt4","messages":[]}"#))
            .unwrap();
        let unauth_resp = run(app, unauth_req).await;
        assert_eq!(unauth_resp.status(), StatusCode::UNAUTHORIZED);
        let err_server = unauth_resp
            .headers()
            .get(axum::http::header::SERVER)
            .expect("error response must carry Server header")
            .to_str()
            .unwrap();
        assert_eq!(err_server, ok_server);
    }

    /// The 413 short-circuit runs INSIDE `SetResponseHeaderLayer` —
    /// `enforce_request_body_limit` rejects the request before any
    /// handler executes. This pins layer ordering: a regression that
    /// moves `SetResponseHeaderLayer` inside the body-limit middleware
    /// (or anywhere "below" it in the stack) would silently strip the
    /// Server header from 413 responses while every existing test
    /// still passed.
    #[tokio::test]
    async fn server_header_present_on_413_short_circuit() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let oversized = 2 * 1024 * 1024; // 2 MiB > 1 MiB cap from cfg()
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .header("content-length", oversized.to_string())
            .body(Body::from(
                r#"{"model":"my-gpt4","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let server = resp
            .headers()
            .get(axum::http::header::SERVER)
            .expect("413 short-circuit must carry Server header")
            .to_str()
            .unwrap();
        assert!(
            server.starts_with("AISIX/"),
            "expected `AISIX/<version>`, got {server:?}"
        );
    }

    /// Security contract: the gateway must NEVER leak an upstream
    /// provider's `Server` token to the client. The passthrough handler
    /// copies upstream response headers wholesale (minus hop-by-hop),
    /// so an upstream like Cloudflare/nginx/gunicorn would surface its
    /// own Server unless `overriding` actually replaces it. A regression
    /// that swapped `overriding` → `if_not_present` would be a silent
    /// information-disclosure bug (provider fingerprinting via error
    /// envelopes) — this test locks the no-leak property.
    #[tokio::test]
    async fn server_header_overrides_upstream_provider_token_no_leak() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    // Upstream's identity — MUST NOT survive the round-trip.
                    .insert_header("server", "cloudflare-nginx/3.7-leakthis")
                    .set_body_json(serde_json::json!({
                        "id": "cmpl-upstream",
                        "model": "gpt-4o",
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": "x"},
                            "finish_reason": "stop"
                        }],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                    })),
            )
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"my-gpt4","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        // Exactly one Server value — `overriding` replaces, doesn't append.
        let all: Vec<_> = resp
            .headers()
            .get_all(axum::http::header::SERVER)
            .iter()
            .collect();
        assert_eq!(
            all.len(),
            1,
            "exactly one Server value expected; got {all:?}"
        );

        let server = all[0].to_str().unwrap();
        assert!(
            server.starts_with("AISIX/"),
            "Server must be the gateway identity; got {server:?}"
        );
        assert!(
            !server.contains("cloudflare") && !server.contains("nginx"),
            "Upstream Server token leaked through; got {server:?}"
        );
    }

    #[tokio::test]
    async fn livez_rejects_non_get_requests() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/livez")
            .body(Body::empty())
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::METHOD_NOT_ALLOWED);
    }

    #[tokio::test]
    async fn livez_returns_500_when_shutting_down() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let state = build_state(snap, hub);
        state.livez.mark_shutting_down();
        let app = build_router(state);

        let req = Request::builder()
            .method("GET")
            .uri("/livez")
            .body(Body::empty())
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let text = std::str::from_utf8(&bytes).unwrap();
        assert!(text.contains("livez check failed"));
    }

    #[tokio::test]
    async fn health_route_is_not_found() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("GET")
            .uri("/health")
            .body(Body::empty())
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn unknown_api_key_returns_401() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-does-not-exist")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"my-gpt4","messages":[]}"#))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn model_not_in_allowed_list_returns_403() {
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        // ApiKey allows only "other-model", the caller asks for "my-gpt4".
        let snap = seed_snapshot("my-gpt4", &["other-model"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"my-gpt4","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "permission_denied");
    }

    #[tokio::test]
    async fn unknown_model_returns_404_envelope() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["*"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"no-such-model","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "model_not_found");
    }

    #[tokio::test]
    async fn empty_messages_returns_400() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"my-gpt4","messages":[]}"#))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    /// Issue #324: missing required field on the chat-completion
    /// request body must surface as **400 Bad Request** per OpenAI's
    /// wire contract, not 422 Unprocessable Entity. SDKs branching
    /// on the status code see different semantics depending on
    /// which proxy they sit behind; a customer migrating between
    /// OpenAI direct and a gateway-fronted deployment needs the
    /// 400-vs-422 distinction to be wire-stable.
    ///
    /// Pre-fix: axum's `Json<ChatFormat>` extractor returned
    /// `JsonRejection::JsonDataError` → 422.
    /// Post-fix: the handler intercepts the JsonRejection and maps
    /// to `ProxyError::InvalidRequest` → 400.
    #[tokio::test]
    async fn missing_model_field_returns_400_not_422() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        // Valid JSON, valid `messages` field, but `model` omitted —
        // the OpenAI ChatCompletion contract requires it.
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "missing model field must surface as 400 per OpenAI wire contract — #324",
        );
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request_error");
    }

    /// Companion case: missing `messages` field must also surface
    /// as 400. Same OpenAI wire contract — `messages` is required.
    #[tokio::test]
    async fn missing_messages_field_returns_400_not_422() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"my-gpt4"}"#))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "missing messages field must surface as 400 per OpenAI wire contract — #324",
        );
        // Pin the envelope shape too — a future regression that
        // returned 400 with a non-OpenAI envelope (or empty body)
        // would otherwise pass on status alone. Per audit MEDIUM on
        // PR #400.
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request_error");
    }

    /// Companion case: malformed JSON (syntax error) also must
    /// surface as 400, not 422. Same handler path as #324 — the
    /// JsonRejection variants for syntax vs data error both map
    /// to InvalidRequest.
    #[tokio::test]
    async fn malformed_json_returns_400() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(r#"{not even valid json"#))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(
            resp.status(),
            StatusCode::BAD_REQUEST,
            "malformed JSON must surface as 400, not 422",
        );
        // Envelope-shape pin matching the sibling missing-field
        // tests — same JsonRejection → InvalidRequest path; the
        // envelope must stay OpenAI-shape on every variant.
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request_error");
    }

    /// Issue #159: a request body whose declared `Content-Length`
    /// exceeds the configured cap must surface as `413 Content Too
    /// Large` per RFC 9110 §15.5.14, NOT as `ECONNRESET` from a
    /// mid-write socket close. Regression: the handler-level body
    /// extractor's overflow path was racing the client write,
    /// surfacing as a network failure indistinguishable from a
    /// gateway crash. The new middleware short-circuits on the
    /// declared size before any handler runs.
    #[tokio::test]
    async fn oversize_body_returns_413_envelope_with_content_length_check() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        // Declare a Content-Length well over the test cfg's 1 MiB cap
        // but ship a tiny actual body — the middleware MUST reject
        // based on the declared size alone, before reading the body.
        // (A real caller's `JSON.stringify` would set Content-Length
        // matching the body size; the assertion is "we trust the
        // declared header for the early reject".)
        let oversized = 2 * 1024 * 1024; // 2 MiB > 1 MiB cap
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .header("content-length", oversized.to_string())
            .body(Body::from(
                r#"{"model":"my-gpt4","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        // OpenAI-shape envelope per docs/api-proxy.md §3:
        // `{ "error": { "message": ..., "type": "..." } }`
        assert_eq!(v["error"]["type"], "invalid_request_error");
        let message = v["error"]["message"].as_str().unwrap();
        assert!(
            message.contains("limit"),
            "413 message should reference the limit; got {message:?}"
        );
    }

    /// Audit HIGH-3 (#343): the body-limit middleware runs BEFORE
    /// the `/v1/messages` handler, so its rejection path must emit
    /// the Anthropic-shape envelope rather than the OpenAI-shape
    /// envelope — otherwise the Claude SDK's strict parser falls
    /// through to a generic exception. Same contract as the handler-
    /// side `into_anthropic_response()` for #336.
    #[tokio::test]
    async fn oversize_body_on_v1_messages_returns_anthropic_envelope_request_too_large() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let oversized = 2 * 1024 * 1024;
        let req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .header("content-length", oversized.to_string())
            .body(Body::from(
                r#"{"model":"my-gpt4","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        // Anthropic-shape envelope per docs.anthropic.com/en/api/errors:
        // `{ "type": "error", "error": { "type": "...", "message": "..." } }`.
        assert_eq!(v["type"], "error", "Anthropic top-level discriminator");
        assert_eq!(
            v["error"]["type"], "request_too_large",
            "413 → Anthropic-canonical request_too_large per status-to-type mapping",
        );
        // OpenAI-only fields must be absent.
        assert!(v["error"].get("code").is_none());
        assert!(v["error"].get("param").is_none());
    }

    /// Audit MEDIUM-A (3rd audit) on #343: when the caller streams an
    /// oversize body without a declared Content-Length, the
    /// `enforce_request_body_limit` middleware skips its early reject
    /// (no length to compare), and the `Json<Value>` extractor's
    /// `DefaultBodyLimit` cap fires during read. That produces a
    /// `JsonRejection::BytesRejection`, which the handler MUST map
    /// to `RequestTooLarge` (413 + `error.type=="request_too_large"`)
    /// rather than `InvalidRequest` (400 + `"invalid_request_error"`)
    /// — the Claude SDK branches on `error.type=="request_too_large"`
    /// to mark requests as "non-retriable cap exceeded"; folding it
    /// into 400 breaks the retry-policy signal.
    #[tokio::test]
    async fn streaming_oversize_body_on_v1_messages_returns_413_request_too_large() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        // Build a streaming body that yields > 1 MiB chunk-by-chunk
        // with NO upstream-set Content-Length. The middleware can't
        // decide on size and will pass through; the per-extractor
        // `DefaultBodyLimit` cap (set to `request_body_limit_bytes`
        // in `build_router`) fires on the read, surfacing as
        // `JsonRejection::BytesRejection`.
        let chunk = vec![b'x'; 200 * 1024]; // 200 KiB per chunk
        let stream =
            futures::stream::iter((0..10).map(move |_| Ok::<_, std::io::Error>(chunk.clone())));
        let body = Body::from_stream(stream);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            // Intentionally NO Content-Length.
            .body(body)
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(
            resp.status(),
            StatusCode::PAYLOAD_TOO_LARGE,
            "streaming-oversize must surface as 413 request_too_large, NOT 400 invalid_request_error",
        );
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["type"], "error");
        assert_eq!(
            v["error"]["type"], "request_too_large",
            "Anthropic-canonical 413 → request_too_large; mistakenly folding into invalid_request_error \
             breaks the Claude SDK's retry-policy branch",
        );
    }

    /// Audit LOW-A (3rd audit) on #343: the path-match guard in
    /// `enforce_request_body_limit` must accept both `/v1/messages`
    /// and `/v1/messages/` (trailing slash) — axum's path
    /// normalization routes both to the Anthropic handler, but a
    /// strict `==` check on the bare form would miss the trailing-
    /// slash variant. SDKs don't add the slash; non-SDK callers
    /// (curl, custom clients) might.
    #[tokio::test]
    async fn oversize_body_on_v1_messages_trailing_slash_still_anthropic_envelope() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let oversized = 2 * 1024 * 1024;
        let req = Request::builder()
            .method("POST")
            .uri("/v1/messages/")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .header("content-length", oversized.to_string())
            .body(Body::from(
                r#"{"model":"my-gpt4","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            v["type"], "error",
            "trailing-slash /v1/messages/ must still emit Anthropic envelope",
        );
        assert_eq!(v["error"]["type"], "request_too_large");
    }

    /// Companion to the above: duplicate Content-Length on /v1/messages
    /// also emits Anthropic envelope. Smuggling-rejection path runs
    /// in the same middleware as the body-limit reject.
    #[tokio::test]
    async fn duplicate_content_length_on_v1_messages_returns_anthropic_envelope() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let body =
            r#"{"model":"my-gpt4","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}"#;
        let mut req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();
        req.headers_mut().append(
            axum::http::header::CONTENT_LENGTH,
            axum::http::HeaderValue::from(body.len()),
        );
        req.headers_mut().append(
            axum::http::header::CONTENT_LENGTH,
            axum::http::HeaderValue::from(body.len() + 1),
        );
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["type"], "error");
        assert_eq!(v["error"]["type"], "invalid_request_error");
    }

    /// Issue #159 audit MEDIUM-3: duplicate `Content-Length` headers
    /// are a classic request-smuggling vector — a server that acts on
    /// the first value while a downstream peer acts on the second can
    /// be tricked into framing the body wrongly. Per RFC 9110 §8.6 a
    /// server SHOULD reject the request rather than disambiguate.
    #[tokio::test]
    async fn duplicate_content_length_headers_return_400_invalid_request() {
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let body = r#"{"model":"my-gpt4","messages":[{"role":"user","content":"hi"}]}"#;
        let mut req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body))
            .unwrap();
        // Inject TWO Content-Length headers (simulating a smuggling
        // attempt). axum's HeaderMap supports `append` for duplicate
        // names; the middleware must reject rather than read the
        // first value.
        req.headers_mut().append(
            axum::http::header::CONTENT_LENGTH,
            axum::http::HeaderValue::from(body.len()),
        );
        req.headers_mut().append(
            axum::http::header::CONTENT_LENGTH,
            axum::http::HeaderValue::from(body.len() + 1),
        );

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request_error");
        let message = v["error"]["message"].as_str().unwrap();
        assert!(
            message.contains("Content-Length"),
            "smuggling-rejection message should mention Content-Length; got {message:?}"
        );
    }

    /// Issue #159 companion: a body within the cap must NOT be
    /// rejected — the middleware short-circuits ONLY when the
    /// Content-Length exceeds the cap, leaving normal traffic
    /// untouched. Without this guard, a regression that always-
    /// rejected (e.g. comparing the wrong field) would be invisible
    /// since most existing tests don't set Content-Length.
    #[tokio::test]
    async fn within_limit_body_is_not_rejected_by_middleware() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-ok",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let app = build_router(build_state(snap, hub));

        let body = r#"{"model":"my-gpt4","messages":[{"role":"user","content":"hi"}]}"#;
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .header("content-length", body.len().to_string())
            .body(Body::from(body))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn upstream_429_passes_through_with_openai_envelope() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"my-gpt4","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "upstream_error");
    }

    /// Issues #322 + #327: when an OpenAI upstream returns a coded
    /// 4xx with the standard `{error:{message,type,code,param}}`
    /// envelope, the gateway:
    /// - preserves `message`, `code`, and `param` verbatim so SDK
    ///   retry logic that branches on `error.code` keeps working;
    /// - normalises `error.type` to the DP-stable token
    ///   `"upstream_error"`, hiding the upstream's private taxonomy
    ///   from the customer (the upstream `type` here —
    ///   `"upstream_test_fixture"` — is mock-llm's internal label
    ///   and must not bleed through).
    #[tokio::test]
    async fn upstream_openai_4xx_forwards_code_and_param_but_normalises_type_per_issue_327() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(429).set_body_raw(
                br#"{"error":{"message":"upstream forced 429","type":"upstream_test_fixture","code":"forced_429","param":"model"}}"#.as_slice(),
                "application/json",
            ))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"my-gpt4","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        let bytes = to_bytes(resp.into_body(), 2048).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["message"], "upstream forced 429");
        // Per #327: `error.type` is the DP-stable taxonomy, NOT the
        // upstream's `type`. The upstream's `upstream_test_fixture`
        // token must NOT leak to the customer envelope.
        assert_eq!(v["error"]["type"], "upstream_error");
        // Per #322: `error.code` and `error.param` ARE preserved so
        // SDK retry logic can branch on the granular code.
        assert_eq!(v["error"]["code"], "forced_429");
        assert_eq!(v["error"]["param"], "model");
    }

    /// Issue #322 fallback contract: when the upstream body is not a
    /// recognisable JSON envelope (HTML error page, garbled text), the
    /// gateway must NOT crash or surface raw bytes; it falls back to
    /// the generic `upstream_error` envelope with the truncated body
    /// as `message`. This pins the content-type guard.
    #[tokio::test]
    async fn upstream_4xx_non_json_body_falls_back_to_generic_envelope() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(400).set_body_raw(
                b"<html><body>403 Forbidden by WAF</body></html>".as_slice(),
                "text/html",
            ))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"my-gpt4","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = to_bytes(resp.into_body(), 2048).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        // Generic envelope: type=upstream_error, message contains the
        // truncated raw body (no JSON parse attempted on text/html).
        assert_eq!(v["error"]["type"], "upstream_error");
        assert!(v["error"].get("code").is_none() || v["error"]["code"].is_null());
    }

    /// Issue #322 sanity check on the 5xx branch: upstream 5xx still
    /// collapses to 502 with the generic envelope AND the upstream
    /// `error.message` is suppressed. Engine names / shard ids / queue
    /// depth routinely appear in upstream 5xx bodies (in this fixture:
    /// "engine offline shard 47") — those are operator-internal and
    /// must not bleed through to the customer envelope.
    #[tokio::test]
    async fn upstream_openai_5xx_with_json_envelope_collapses_and_redacts_message() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(503).set_body_raw(
                br#"{"error":{"message":"engine offline shard 47","type":"server_error","code":"engine_overloaded"}}"#.as_slice(),
                "application/json",
            ))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"my-gpt4","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        let bytes = to_bytes(resp.into_body(), 2048).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "upstream_error");
        let msg = v["error"]["message"].as_str().unwrap();
        assert!(
            !msg.contains("engine offline") && !msg.contains("shard 47"),
            "upstream 5xx `error.message` must NOT leak to customer; got: {msg:?}"
        );
        // Upstream `code` (engine_overloaded) must also not pass
        // through on 5xx.
        assert!(
            v["error"].get("code").is_none() || v["error"]["code"].is_null(),
            "upstream 5xx `error.code` must not pass through; got code={:?}",
            v["error"]["code"]
        );
    }

    /// Cross-provider contract: Anthropic upstream 5xx → client sees an
    /// OpenAI-shape envelope `{error:{type:"upstream_error",...}}` with
    /// status 502 (collapsed per `BridgeError::http_status`, see
    /// crates/aisix-gateway/src/bridge.rs).
    #[tokio::test]
    async fn upstream_anthropic_5xx_collapses_to_502_with_openai_envelope() {
        use aisix_provider_anthropic::AnthropicBridge;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(503).set_body_string(
                r#"{"type":"error","error":{"type":"overloaded_error","message":"upstream busy"}}"#,
            ))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(matrix_anthropic_pk(&upstream.uri()));
        snap.models.insert(anthropic_model_entry("my-claude"));
        snap.apikeys
            .insert(apikey_entry("sk-caller", &["my-claude"]));
        let hub = Arc::new(Hub::new());
        hub.register_specialized("anthropic", Arc::new(AnthropicBridge::new()));
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "my-claude",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        let v: serde_json::Value =
            serde_json::from_slice(&to_bytes(resp.into_body(), 1024).await.unwrap()).unwrap();
        // Anthropic-shape leaks must not bleed through to the client.
        assert_eq!(v["error"]["type"], "upstream_error");
        assert!(v["error"]["message"].is_string());
    }

    /// Cross-provider 4xx forwarding (issue #327): Anthropic upstream
    /// 400 reaches the OpenAI-client side with `error.type` normalised
    /// to the DP-stable `"upstream_error"` token — Anthropic's private
    /// taxonomy (`invalid_request_error`, `authentication_error`, etc.)
    /// must not bleed through.
    #[tokio::test]
    async fn upstream_anthropic_400_normalises_type_to_upstream_error() {
        use aisix_provider_anthropic::AnthropicBridge;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(400).set_body_raw(
                br#"{"type":"error","error":{"type":"invalid_request_error","message":"bad input"}}"#.as_slice(),
                "application/json",
            ))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(matrix_anthropic_pk(&upstream.uri()));
        snap.models.insert(anthropic_model_entry("my-claude"));
        snap.apikeys
            .insert(apikey_entry("sk-caller", &["my-claude"]));
        let hub = Arc::new(Hub::new());
        hub.register_specialized("anthropic", Arc::new(AnthropicBridge::new()));
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "my-claude",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v: serde_json::Value =
            serde_json::from_slice(&to_bytes(resp.into_body(), 1024).await.unwrap()).unwrap();
        assert_eq!(v["error"]["type"], "upstream_error");
        assert_eq!(v["error"]["message"], "bad input");
        // Anthropic `invalid_request_error` doesn't derive an OpenAI
        // string code — translation table emits `code: null`.
        assert!(v["error"].get("code").is_none() || v["error"]["code"].is_null());
    }

    /// Issue #322 + #327 cross-wire contract: Anthropic upstream
    /// `rate_limit_error` must derive OpenAI `error.code =
    /// rate_limit_exceeded` (so SDK retry logic that switches on
    /// `error.code` recognises the rate-limit failure regardless of
    /// upstream), while `error.type` stays as the DP-stable
    /// `"upstream_error"` (per #327, Anthropic's `rate_limit_error`
    /// token must not bleed through).
    #[tokio::test]
    async fn upstream_anthropic_rate_limit_derives_openai_rate_limit_exceeded_code() {
        use aisix_provider_anthropic::AnthropicBridge;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(429).set_body_raw(
                br#"{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}"#.as_slice(),
                "application/json",
            ))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(matrix_anthropic_pk(&upstream.uri()));
        snap.models.insert(anthropic_model_entry("my-claude"));
        snap.apikeys
            .insert(apikey_entry("sk-caller", &["my-claude"]));
        let hub = Arc::new(Hub::new());
        hub.register_specialized("anthropic", Arc::new(AnthropicBridge::new()));
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "my-claude",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        let v: serde_json::Value =
            serde_json::from_slice(&to_bytes(resp.into_body(), 1024).await.unwrap()).unwrap();
        // Per #327: `error.type` is the DP-stable token, never the
        // upstream's. Per #322: `error.code` is the derived OpenAI
        // string code so SDK retry logic fires correctly.
        assert_eq!(v["error"]["type"], "upstream_error");
        assert_eq!(v["error"]["code"], "rate_limit_exceeded");
        assert_eq!(v["error"]["message"], "slow down");
    }

    /// Garbage upstream body (200 + non-JSON) must surface as 502 with
    /// `error.type = "upstream_decode_error"` — distinct from the 4xx/5xx
    /// `upstream_error` token so dashboards can tell parsing failures
    /// apart from upstream errors.
    #[tokio::test]
    async fn upstream_unparseable_body_returns_502_decode_error_envelope() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_string("{not valid json"))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"my-gpt4","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        let v: serde_json::Value =
            serde_json::from_slice(&to_bytes(resp.into_body(), 1024).await.unwrap()).unwrap();
        assert_eq!(v["error"]["type"], "upstream_decode_error");
    }

    #[tokio::test]
    async fn provider_without_registered_bridge_returns_503() {
        // Snapshot has a Model targeting openai, but the Hub is empty.
        let hub = Arc::new(Hub::new());
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], "http://unused");
        let app = build_router(build_state(snap, hub));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"my-gpt4","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn streaming_response_emits_sse_then_done_sentinel() {
        let upstream = MockServer::start().await;
        let sse = "\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .contains("text/event-stream"));

        // Drain the body, decode SSE events, assert we got at least one
        // delta chunk plus the terminating [DONE].
        let mut body_stream = resp.into_body().into_data_stream();
        let mut decoder = SseDecoder::new();
        let mut events = Vec::new();
        while let Some(chunk) = body_stream.next().await {
            let bytes = chunk.unwrap();
            events.extend(decoder.feed(bytes.as_ref()));
        }
        assert!(events.contains(&SseEvent::Done), "missing [DONE] sentinel");
        let data_count = events
            .iter()
            .filter(|e| matches!(e, SseEvent::Data(_)))
            .count();
        assert!(
            data_count >= 2,
            "expected at least two chat chunks, got {data_count}"
        );
    }

    /// Issue #177: per docs/api-proxy.md §5, abnormal upstream
    /// stream termination must close the response WITHOUT `[DONE]`
    /// — SDK consumers that key off `[DONE]` for clean-completion
    /// signal need to detect truncation. The previous behavior
    /// emitted `[DONE]` after the SSE error event, masking
    /// truncated responses as complete.
    #[tokio::test]
    async fn streaming_response_omits_done_when_upstream_returns_invalid_json_mid_stream() {
        let upstream = MockServer::start().await;
        // Upstream emits two valid SSE chunks then a malformed JSON
        // payload. The malformed payload triggers `serde_json::from_str`
        // to fail in the bridge's `build_chunk_stream`, surfacing as
        // `BridgeError::UpstreamDecode` to `build_sse_stream`, which
        // emits an SSE `event: error` frame. After that frame the
        // proxy MUST NOT emit `[DONE]`.
        let sse = "\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial \"},\"finish_reason\":null}]}\n\n\
data: <not valid json>\n\n";
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        // Drain the wire bytes — at this layer we want byte-level
        // assertions, not just decoded SSE events, because the
        // contract being verified is "did `[DONE]` appear at all
        // on the wire".
        let mut body_stream = resp.into_body().into_data_stream();
        let mut wire = Vec::new();
        while let Some(chunk) = body_stream.next().await {
            wire.extend_from_slice(chunk.unwrap().as_ref());
        }
        let wire_str = String::from_utf8(wire).expect("SSE bytes are utf8");

        // Per docs §5: NO `[DONE]` after abnormal termination.
        assert!(
            !wire_str.contains("data: [DONE]"),
            "abnormal termination MUST close without [DONE]; got wire:\n{wire_str}"
        );
        // The error event MUST be emitted so SDK consumers see a
        // failure signal.
        assert!(
            wire_str.contains("event: error"),
            "abnormal termination MUST emit `event: error`; got wire:\n{wire_str}"
        );
        // The error payload MUST be valid OpenAI-envelope JSON
        // (the SDK does `JSON.parse(sse.data)` BEFORE checking
        // event type, so plain-string payloads yield a SyntaxError
        // instead of the typed APIError callers expect).
        let err_event_idx = wire_str.find("event: error\n").unwrap();
        let after_err = &wire_str[err_event_idx + "event: error\n".len()..];
        let data_line = after_err
            .lines()
            .find(|l| l.starts_with("data: "))
            .expect("error event followed by a data line");
        let json_payload = &data_line["data: ".len()..];
        let parsed: serde_json::Value = serde_json::from_str(json_payload)
            .expect("error frame data must be valid OpenAI-envelope JSON");
        assert!(
            parsed.get("error").is_some(),
            "error frame data must be `{{\"error\": {{...}}}}` shape; got {json_payload}"
        );
    }

    /// Issue #204: streaming responses MUST run output guardrails — pre-fix
    /// the streaming path skipped them, so a `kind: "keyword"` deny-list was
    /// bypassable by setting `stream: true`.
    ///
    /// Issue #466: output guardrails must also HOLD content back while
    /// streaming. `keyword` now inherits the default hold-back policy
    /// ([`StreamOutputPolicy::BufferFull`], fail-closed), so a blocked
    /// streaming response NEVER puts the forbidden content on the wire.
    /// This test pins:
    ///
    ///   - 200 OK + SSE wire shape (the request itself is well-formed)
    ///   - upstream IS hit (output guardrails run AFTER the upstream call)
    ///   - an SSE `event: error` frame with the OpenAI `content_filter` envelope
    ///   - NO terminal `[DONE]` (a guardrail block is an abnormal termination)
    ///   - the matched literal "secret-string" does NOT appear ANYWHERE on
    ///     the wire — the hold-back means the offending content is never
    ///     emitted (the #466 fix; previously the live-forwarded chunks
    ///     leaked it before the end-of-stream check).
    #[tokio::test]
    async fn streaming_output_guardrail_blocks_with_sse_error_event_and_no_done() {
        let upstream = MockServer::start().await;
        // Upstream emits 3 SSE chunks: role, then content containing
        // the forbidden literal, then the terminal stop. The full
        // assistant content the guardrail evaluates is "leak: secret-string"
        // which the keyword guardrail at "secret-string" must block.
        let sse = "\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"leak: secret-string\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let state = build_state(snap, hub);
        seed_guardrail(
            &state.snapshot,
            "g-stream-output",
            r#"{"name":"stream-output-guard","kind":"keyword","hook_point":"output","patterns":[{"kind":"literal","value":"secret-string"}]}"#,
        );
        let app = build_router(state);

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let mut body_stream = resp.into_body().into_data_stream();
        let mut wire = Vec::new();
        while let Some(chunk) = body_stream.next().await {
            wire.extend_from_slice(chunk.unwrap().as_ref());
        }
        let wire_str = String::from_utf8(wire).expect("SSE bytes are utf8");

        // #466: the default hold-back policy means the forbidden content is
        // never forwarded — the matched literal appears NOWHERE on the wire
        // (pre-fix the live-forwarded chunks leaked it before the check).
        assert!(
            !wire_str.contains("secret-string"),
            "hold-back guardrail leaked the matched content onto the wire; got:\n{wire_str}"
        );

        // Per docs §5 abnormal-termination contract (the guardrail
        // block is the streaming-equivalent of an abnormal close):
        // NO `[DONE]` after the error event. SDK consumers that key
        // off `[DONE]` need to detect the truncation.
        assert!(
            !wire_str.contains("data: [DONE]"),
            "blocked stream MUST close without [DONE]; got wire:\n{wire_str}"
        );
        // SSE `event: error` frame MUST appear so SDK consumers see
        // a failure signal.
        assert!(
            wire_str.contains("event: error"),
            "blocked stream MUST emit `event: error`; got wire:\n{wire_str}"
        );
        // The error frame's data MUST be valid OpenAI-envelope JSON
        // with `error.type: "content_filter"` (parallel to #153's
        // non-streaming contract).
        let err_event_idx = wire_str.find("event: error\n").unwrap();
        let after_err = &wire_str[err_event_idx + "event: error\n".len()..];
        let data_line = after_err
            .lines()
            .find(|l| l.starts_with("data: "))
            .expect("error event followed by a data line");
        let json_payload = &data_line["data: ".len()..];
        let parsed: serde_json::Value = serde_json::from_str(json_payload)
            .expect("error frame data must be valid OpenAI-envelope JSON");
        assert_eq!(
            parsed["error"]["type"], "content_filter",
            "error.type must mark the guardrail block; got {json_payload}"
        );
        // Per #153: the matched literal MUST NOT appear inside the
        // error frame envelope (the *error*, not the pre-emitted
        // chunks — those carry the partial content that buffer-
        // then-check accepts as a known trade-off).
        let error_message = parsed["error"]["message"].as_str().unwrap();
        assert!(
            !error_message.contains("secret-string"),
            "guardrail leaked the matched literal in the error envelope; got {error_message:?}"
        );
        assert_eq!(
            error_message, "response blocked by content policy",
            "wire-level message must use the redacted static string per #153"
        );
    }

    /// #448 parity (streaming): the chat streaming output guardrail buffered
    /// only `delta.content`, so a blocked literal in a tool-call's `arguments`
    /// leaked (chat non-streaming + /v1/messages streaming already scan tool
    /// calls). With the fix, tool-call name + arguments are scanned at
    /// end-of-stream; a blocked literal blocks the stream (error frame) and is
    /// held back (never on the wire) under the default BufferFull policy.
    #[tokio::test]
    async fn streaming_output_guardrail_blocks_tool_call_arguments() {
        let upstream = MockServer::start().await;
        let c1 = serde_json::json!({"id":"up-1","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]});
        // Tool-call delta carrying the forbidden literal in `arguments` — the
        // pre-fix path never scanned this.
        let c2 = serde_json::json!({"id":"up-1","model":"gpt-4o","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"secret-string\"}"}}]},"finish_reason":null}]});
        let c3 = serde_json::json!({"id":"up-1","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]});
        let sse = format!("data: {c1}\n\ndata: {c2}\n\ndata: {c3}\n\ndata: [DONE]\n\n");
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let state = build_state(snap, hub);
        seed_guardrail(
            &state.snapshot,
            "g-stream-tc",
            r#"{"name":"stream-tc-guard","kind":"keyword","hook_point":"output","patterns":[{"kind":"literal","value":"secret-string"}]}"#,
        );
        let app = build_router(state);

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let mut body_stream = resp.into_body().into_data_stream();
        let mut wire = Vec::new();
        while let Some(chunk) = body_stream.next().await {
            wire.extend_from_slice(chunk.unwrap().as_ref());
        }
        let wire_str = String::from_utf8(wire).expect("SSE bytes are utf8");

        assert!(
            !wire_str.contains("secret-string"),
            "tool-call arguments leaked despite output block; got:\n{wire_str}"
        );
        assert!(
            wire_str.contains("event: error"),
            "blocked stream must emit `event: error`; got:\n{wire_str}"
        );
        let idx = wire_str
            .find("event: error\n")
            .expect("error event present");
        let data_line = wire_str[idx..]
            .lines()
            .find(|l| l.starts_with("data: "))
            .expect("error event followed by a data line");
        let parsed: serde_json::Value =
            serde_json::from_str(&data_line["data: ".len()..]).expect("valid error envelope");
        assert_eq!(parsed["error"]["type"], "content_filter");
    }

    /// P2 (#379): like the keyword guardrail above (which now holds back by
    /// default, #466), `azure_content_safety_text_moderation` keeps offending
    /// content off the wire — here via the configurable `Window` policy
    /// rather than the default `BufferFull`. A blocked streaming response
    /// NEVER puts the offending content on the wire.
    #[tokio::test]
    async fn streaming_text_moderation_blocks_and_holds_content_back_no_leak() {
        let upstream = MockServer::start().await;
        let sse = "\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"this is harmful text\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&upstream)
            .await;

        // Azure Content Safety mock returns high severity → block.
        let acs = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/contentsafety/text:analyze"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "categoriesAnalysis": [{"category": "Hate", "severity": 6}],
                "blocklistsMatch": []
            })))
            .mount(&acs)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let state = build_state(snap, hub);
        seed_guardrail(
            &state.snapshot,
            "g-textmod-output",
            &format!(
                r#"{{"name":"textmod","kind":"azure_content_safety_text_moderation","hook_point":"output","endpoint":"{}","api_key":"k"}}"#,
                acs.uri()
            ),
        );
        let app = build_router(state);

        let body = serde_json::json!({"model":"my-gpt4","messages":[{"role":"user","content":"hi"}],"stream":true});
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let mut body_stream = resp.into_body().into_data_stream();
        let mut wire = Vec::new();
        while let Some(chunk) = body_stream.next().await {
            wire.extend_from_slice(chunk.unwrap().as_ref());
        }
        let wire_str = String::from_utf8(wire).expect("SSE bytes are utf8");

        // The hold-back guarantee: the harmful content NEVER reached the
        // wire (held in `pending`, dropped on block).
        assert!(
            !wire_str.contains("harmful text"),
            "hold-back must keep blocked content off the wire; got:\n{wire_str}"
        );
        assert!(
            wire_str.contains("event: error"),
            "blocked stream must emit `event: error`; got:\n{wire_str}"
        );
        assert!(
            !wire_str.contains("data: [DONE]"),
            "blocked stream must omit [DONE]; got:\n{wire_str}"
        );
        let idx = wire_str.find("event: error\n").unwrap();
        let data_line = wire_str[idx..]
            .lines()
            .find(|l| l.starts_with("data: "))
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&data_line["data: ".len()..]).unwrap();
        assert_eq!(parsed["error"]["type"], "content_filter");
    }

    /// Clean content is held back, scanned, then released in full + [DONE].
    #[tokio::test]
    async fn streaming_text_moderation_releases_clean_content() {
        let upstream = MockServer::start().await;
        let sse = "\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"perfectly fine answer\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&upstream)
            .await;
        let acs = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/contentsafety/text:analyze"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "categoriesAnalysis": [{"category": "Hate", "severity": 0}],
                "blocklistsMatch": []
            })))
            .mount(&acs)
            .await;
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let state = build_state(snap, hub);
        seed_guardrail(
            &state.snapshot,
            "g-textmod-clean",
            &format!(
                r#"{{"name":"textmod","kind":"azure_content_safety_text_moderation","hook_point":"output","endpoint":"{}","api_key":"k"}}"#,
                acs.uri()
            ),
        );
        let app = build_router(state);
        let body = serde_json::json!({"model":"my-gpt4","messages":[{"role":"user","content":"hi"}],"stream":true});
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let mut body_stream = resp.into_body().into_data_stream();
        let mut wire = Vec::new();
        while let Some(c) = body_stream.next().await {
            wire.extend_from_slice(c.unwrap().as_ref());
        }
        let wire_str = String::from_utf8(wire).expect("utf8");
        assert!(
            wire_str.contains("perfectly fine answer"),
            "clean content must be released after the scan; got:\n{wire_str}"
        );
        assert!(
            wire_str.contains("data: [DONE]"),
            "clean stream must end with [DONE]; got:\n{wire_str}"
        );
        assert!(
            !wire_str.contains("event: error"),
            "clean stream must not emit an error frame; got:\n{wire_str}"
        );
    }

    /// Drive a streaming chat through a seeded text-moderation guardrail
    /// and return the raw SSE wire bytes. `guardrail_cfg` is the full
    /// guardrail JSON (with the ACS mock endpoint already substituted).
    async fn run_textmod_stream(guardrail_cfg: &str, upstream_sse: &str) -> String {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(upstream_sse.to_owned()),
            )
            .mount(&upstream)
            .await;
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let state = build_state(snap, hub);
        seed_guardrail(&state.snapshot, "g-tm", guardrail_cfg);
        let app = build_router(state);
        let body = serde_json::json!({"model":"my-gpt4","messages":[{"role":"user","content":"hi"}],"stream":true});
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let mut body_stream = resp.into_body().into_data_stream();
        let mut wire = Vec::new();
        while let Some(c) = body_stream.next().await {
            wire.extend_from_slice(c.unwrap().as_ref());
        }
        String::from_utf8(wire).expect("utf8")
    }

    async fn acs_mock(severity: u8) -> MockServer {
        let acs = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/contentsafety/text:analyze"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "categoriesAnalysis": [{"category": "Hate", "severity": severity}],
                "blocklistsMatch": []
            })))
            .mount(&acs)
            .await;
        acs
    }

    fn two_content_chunks(a: &str, b: &str) -> String {
        format!(
            "data: {{\"id\":\"u\",\"model\":\"gpt-4o\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{a}\"}},\"finish_reason\":null}}]}}\n\n\
data: {{\"id\":\"u\",\"model\":\"gpt-4o\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{b}\"}},\"finish_reason\":null}}]}}\n\n\
data: {{\"id\":\"u\",\"model\":\"gpt-4o\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}]}}\n\n\
data: [DONE]\n\n"
        )
    }

    /// H1: Window mode blocks MID-STREAM (small window so the first window
    /// trips before end-of-stream) and leaks nothing.
    #[tokio::test]
    async fn streaming_text_moderation_window_blocks_mid_stream() {
        let acs = acs_mock(6).await;
        let cfg = format!(
            r#"{{"name":"tm","kind":"azure_content_safety_text_moderation","hook_point":"output","endpoint":"{}","api_key":"k","stream_processing_mode":"window","window_size":5,"window_overlap_size":1}}"#,
            acs.uri()
        );
        let wire = run_textmod_stream(&cfg, &two_content_chunks("hello ", "world!")).await;
        assert!(
            !wire.contains("hello"),
            "mid-stream block must not leak window content; got:\n{wire}"
        );
        assert!(
            !wire.contains("world"),
            "mid-stream block must not leak later content; got:\n{wire}"
        );
        assert!(
            wire.contains("event: error"),
            "expected content_filter frame; got:\n{wire}"
        );
        assert!(
            !wire.contains("data: [DONE]"),
            "blocked stream omits [DONE]; got:\n{wire}"
        );
    }

    /// H1: Window mode releases multiple clean windows (exercises the
    /// mid-stream flush + overlap retention), ending with [DONE].
    #[tokio::test]
    async fn streaming_text_moderation_window_releases_clean_multiwindow() {
        let acs = acs_mock(0).await;
        let cfg = format!(
            r#"{{"name":"tm","kind":"azure_content_safety_text_moderation","hook_point":"output","endpoint":"{}","api_key":"k","stream_processing_mode":"window","window_size":5,"window_overlap_size":2}}"#,
            acs.uri()
        );
        let wire = run_textmod_stream(&cfg, &two_content_chunks("hello ", "world!")).await;
        assert!(
            wire.contains("hello"),
            "clean windows must be released; got:\n{wire}"
        );
        assert!(
            wire.contains("world"),
            "all clean content must be released; got:\n{wire}"
        );
        assert!(
            wire.contains("data: [DONE]"),
            "clean stream ends with [DONE]; got:\n{wire}"
        );
        assert!(
            !wire.contains("event: error"),
            "clean stream emits no error; got:\n{wire}"
        );
    }

    /// H1: BufferFull cap exceeded with fail_closed → block, no leak.
    #[tokio::test]
    async fn streaming_text_moderation_buffer_full_cap_fail_closed_blocks() {
        let acs = acs_mock(0).await; // severity irrelevant — the cap trips first
        let cfg = format!(
            r#"{{"name":"tm","kind":"azure_content_safety_text_moderation","hook_point":"output","endpoint":"{}","api_key":"k","stream_processing_mode":"buffer_full","max_buffer_bytes":4,"on_buffer_exceeded":"fail_closed"}}"#,
            acs.uri()
        );
        let wire = run_textmod_stream(&cfg, &two_content_chunks("abcd", "efghij")).await;
        assert!(
            !wire.contains("abcd"),
            "fail-closed cap must not leak buffered content; got:\n{wire}"
        );
        assert!(
            wire.contains("event: error"),
            "cap fail-closed must emit content_filter; got:\n{wire}"
        );
        assert!(
            !wire.contains("data: [DONE]"),
            "cap-blocked stream omits [DONE]; got:\n{wire}"
        );
    }

    /// H1: BufferFull cap exceeded with fail_open → release held + forward
    /// the rest live, ending with [DONE].
    #[tokio::test]
    async fn streaming_text_moderation_buffer_full_cap_fail_open_releases() {
        let acs = acs_mock(0).await;
        let cfg = format!(
            r#"{{"name":"tm","kind":"azure_content_safety_text_moderation","hook_point":"output","endpoint":"{}","api_key":"k","stream_processing_mode":"buffer_full","max_buffer_bytes":4,"on_buffer_exceeded":"fail_open"}}"#,
            acs.uri()
        );
        let wire = run_textmod_stream(&cfg, &two_content_chunks("abcd", "efghij")).await;
        assert!(
            wire.contains("abcd"),
            "fail-open cap must release held content; got:\n{wire}"
        );
        assert!(
            wire.contains("data: [DONE]"),
            "fail-open released stream ends with [DONE]; got:\n{wire}"
        );
        assert!(
            !wire.contains("event: error"),
            "fail-open release emits no error; got:\n{wire}"
        );
    }

    // ---- regression coverage for issue #107 -------------------------
    // Pre-fix only /v1/chat/completions enforced rate-limit / budget;
    // every other LLM endpoint silently bypassed both. The test below
    // pins /v1/embeddings — representative of the class — to ensure
    // the gate fires after this PR. Adding the same coverage to every
    // endpoint would multiply the test surface without buying signal,
    // since the gate is centralised in `crate::quota::enforce`. If
    // any individual handler ever stops calling it, that handler's
    // own tests would still catch the breakage on the budget path
    // (BudgetExceeded surfaces as a 4xx the existing tests assert on).

    #[tokio::test]
    async fn rate_limit_rpm_applies_to_embeddings_endpoint_issue_107() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/embeddings"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "object": "list",
                "model": "text-embedding-3-small",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "usage": {"prompt_tokens": 5, "total_tokens": 5}
            })))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot_with_limits(
            "my-gpt4",
            &["my-gpt4"],
            &upstream.uri(),
            serde_json::json!({"rpm": 1}),
        );
        let state = build_state(snap, hub);
        let body = serde_json::json!({"model": "my-gpt4", "input": "hello"});
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };

        // First request consumes the only RPM slot.
        let resp = run(build_router(state.clone()), make_req()).await;
        assert_eq!(resp.status(), StatusCode::OK);

        // Pre-fix: this would also return 200 (the gate didn't run on
        // /v1/embeddings). Post-fix: 429 because the rpm=1 cap is now
        // enforced uniformly via crate::quota::enforce.
        let resp = run(build_router(state.clone()), make_req()).await;
        assert_eq!(
            resp.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "/v1/embeddings must enforce rate limits (issue #107); pre-fix it bypassed",
        );
        let body_bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(v["error"]["type"], "rate_limit_exceeded");
    }

    /// Regression for issue #108: streaming chat used to commit
    /// `0` tokens up front and never look at the upstream's terminal
    /// usage frame. TPM caps were silently bypassed for all streaming
    /// traffic. The fix: build_sse_stream now passes the largest
    /// `total_tokens` seen on any chunk to a callback that calls
    /// `Limiter::add_tokens_post_stream`, after the SSE stream
    /// completes. This test exercises that path end-to-end:
    ///
    /// 1. Issue one streaming request whose terminal SSE chunk
    ///    carries `usage.total_tokens = 1500`. Pre-fix this would
    ///    leave TPM at 0; post-fix TPM should be 1500.
    /// 2. Issue a second streaming request with the same key.
    ///    With TPM cap at 1000, this must 429 (not 200) — the
    ///    pre-emptive `tpm.is_exceeded` check on pre_commit catches
    ///    the over-shoot left by the previous request.
    #[tokio::test]
    async fn streaming_chat_tpm_cap_enforced_after_post_stream_commit_issue_108() {
        let upstream = MockServer::start().await;
        // Final SSE chunk carries usage. OpenAI emits this when the
        // client sets `stream_options.include_usage=true`; the proxy
        // doesn't yet add that on the streamed leg, but the OpenAI
        // bridge does parse `usage` off any chunk that has one — so
        // our mock can include it on the terminal chunk and the
        // bridge will surface it.
        let sse = "\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"up-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":500,\"completion_tokens\":1000,\"total_tokens\":1500}}\n\n\
data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot_with_limits(
            "my-gpt4",
            &["my-gpt4"],
            &upstream.uri(),
            serde_json::json!({"tpm": 1000}),
        );
        let state = build_state(snap, hub);
        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        });
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };

        // First streaming request succeeds. Drive the body to
        // completion so build_sse_stream's on_complete fires.
        let resp = run(build_router(state.clone()), make_req()).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let mut body_stream = resp.into_body().into_data_stream();
        while let Some(chunk) = body_stream.next().await {
            let _ = chunk.unwrap();
        }

        // Second request must 429 — TPM is now over-shot at 1500/1000.
        // Pre-fix TPM stayed at 0 and this would have been a 200.
        let resp = run(build_router(state.clone()), make_req()).await;
        assert_eq!(
            resp.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "streaming chat must commit upstream tokens to TPM (issue #108); \
             pre-fix this returned 200 and the cap was bypassed",
        );
        let body_bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(v["error"]["type"], "rate_limit_exceeded");
    }

    #[tokio::test]
    async fn rate_limit_rpm_returns_429_with_retry_after_header() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-up",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot_with_limits(
            "my-gpt4",
            &["my-gpt4"],
            &upstream.uri(),
            serde_json::json!({"rpm": 1}),
        );
        let state = build_state(snap, hub);
        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };

        // First request succeeds.
        let resp = run(build_router(state.clone()), make_req()).await;
        assert_eq!(resp.status(), StatusCode::OK);

        // Second request within the same minute trips rpm=1 → 429.
        let resp = run(build_router(state.clone()), make_req()).await;
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        let retry = resp
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok())
            .expect("missing or malformed retry-after header");
        assert!(retry >= 1);
        let body_bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(v["error"]["type"], "rate_limit_exceeded");
    }

    #[tokio::test]
    async fn rate_limit_tpm_blocks_after_token_commit_exhausts_window() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-up",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop"
                }],
                // Deliberately overshoot the TPM cap so the next
                // pre_commit observes an exhausted window.
                "usage": {"prompt_tokens": 10_000, "completion_tokens": 10_000, "total_tokens": 20_000}
            })))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot_with_limits(
            "my-gpt4",
            &["my-gpt4"],
            &upstream.uri(),
            serde_json::json!({"tpm": 1_000}),
        );
        let state = build_state(snap, hub);
        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };

        // First request goes through (pre-commit TPM is unchecked for an
        // empty bucket); usage counted at post-deduct overshoots the cap.
        let resp = run(build_router(state.clone()), make_req()).await;
        assert_eq!(resp.status(), StatusCode::OK);

        // Second request sees TPM > 1000 and rejects.
        let resp = run(build_router(state), make_req()).await;
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        let body_bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(v["error"]["type"], "rate_limit_exceeded");
    }

    #[tokio::test]
    async fn request_lifecycle_increments_metrics_counters() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-up",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
            })))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());

        let state = build_state(snap, hub);
        let metrics = state.metrics.clone();
        let app = build_router(state);

        // Pre-flight: counter family is absent until something writes.
        assert!(!metrics.render().contains("aisix_requests_total"));

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let rendered = metrics.render();
        assert!(rendered.contains("aisix_requests_total"));
        assert!(rendered.contains("provider=\"openai\""));
        assert!(rendered.contains("outcome=\"success\""));
        assert!(rendered.contains("aisix_tokens_consumed_total"));
        // 7 tokens were committed.
        assert!(
            rendered.contains("7"),
            "expected tokens counter at 7:\n{rendered}"
        );
    }

    #[tokio::test]
    async fn ratelimit_rejection_increments_ratelimit_counter() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-up",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot_with_limits(
            "my-gpt4",
            &["my-gpt4"],
            &upstream.uri(),
            serde_json::json!({"rpm": 1}),
        );
        let state = build_state(snap, hub);
        let metrics = state.metrics.clone();
        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };

        let _ = run(build_router(state.clone()), make_req()).await;
        let resp = run(build_router(state), make_req()).await;
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);

        let rendered = metrics.render();
        assert!(rendered.contains("aisix_ratelimit_rejections_total"));
        assert!(rendered.contains("scope=\"requests\""));
    }

    #[tokio::test]
    async fn cache_hit_short_circuits_upstream_and_sets_header() {
        // Wiremock that *only* satisfies one upstream call. If the cache
        // ever lets a second request through, the test fails with a 500.
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-up",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "cached"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .expect(1) // hard expectation: exactly one upstream hit
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        // Cache gate opens only when an enabled policy exists in
        // snapshot. Without this seed step the test would 200 but
        // the cache header would be absent (policy-disabled path).
        seed_cache_policy(&snap, "test-cache");
        // Cache enabled — uses the default constructor.
        let state = build_state_with_cache(snap, hub);
        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };

        // First request — miss.
        let resp = run(build_router(state.clone()), make_req()).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get("x-aisix-cache")
                .and_then(|v| v.to_str().ok()),
            Some("miss"),
        );

        // Second identical request — hit.
        let resp = run(build_router(state), make_req()).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get("x-aisix-cache")
                .and_then(|v| v.to_str().ok()),
            Some("hit"),
        );
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], "cached");
    }

    /// Regression for #88. On a cache hit the DP must surface the
    /// cached response's prompt + completion tokens on a dedicated
    /// `cache_hit_saved_*` pair so cp-api can multiply by its pricing
    /// catalog server-side and report `cost_saved_usd` on `/usage`.
    /// Miss rows must keep the saved counters at zero.
    #[tokio::test]
    async fn cache_hit_emits_saved_token_counters_on_telemetry_event() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-up",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "cached"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18}
            })))
            .expect(1)
            .mount(&upstream)
            .await;

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        seed_cache_policy(&snap, "test-cache");
        let state = build_state_with_cache(snap, hub).with_usage_sink(UsageSink::new(tx));

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };

        // Miss: saved counters must be zero (the request paid the upstream).
        let _ = run(build_router(state.clone()), make_req()).await;
        let miss_event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("miss event was never emitted")
            .expect("sender dropped");
        assert_eq!(miss_event.cache_status, "miss");
        assert_eq!(miss_event.cache_hit_saved_input_tokens, 0);
        assert_eq!(miss_event.cache_hit_saved_output_tokens, 0);

        // Hit: saved counters must mirror the cached response's usage.
        let _ = run(build_router(state), make_req()).await;
        let hit_event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("hit event was never emitted")
            .expect("sender dropped");
        assert_eq!(hit_event.cache_status, "hit");
        assert_eq!(hit_event.cache_hit_saved_input_tokens, 7);
        assert_eq!(hit_event.cache_hit_saved_output_tokens, 11);
        // `prompt_tokens` keeps mirroring the cached usage too — the
        // existing dashboard rollups stay correct. The new field is
        // additive, not a substitute.
        assert_eq!(hit_event.prompt_tokens, 7);
        assert_eq!(hit_event.completion_tokens, 11);
    }

    #[tokio::test]
    async fn cache_miss_when_request_payload_differs() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-up",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            // Two distinct payloads → expect exactly two upstream calls.
            .expect(2)
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        seed_cache_policy(&snap, "test-cache");
        let state = build_state_with_cache(snap, hub);

        let body_a = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "first"}]
        });
        let body_b = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "second"}]
        });
        let mk = |body: &serde_json::Value| {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };

        let r1 = run(build_router(state.clone()), mk(&body_a)).await;
        let r2 = run(build_router(state), mk(&body_b)).await;
        for r in [r1, r2] {
            assert_eq!(r.status(), StatusCode::OK);
            assert_eq!(
                r.headers()
                    .get("x-aisix-cache")
                    .and_then(|v| v.to_str().ok()),
                Some("miss"),
            );
        }
    }

    #[tokio::test]
    async fn applies_to_model_does_not_cache_unmatched_model() {
        // Stage 3 contract: a `cache_policy` with
        // `applies_to = "model:<other>"` must NOT enable the cache for
        // requests targeting a different model. Three identical
        // requests should hit the upstream three times — none of them
        // gets cached.
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-up",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "always-fresh"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .expect(3) // each call must reach the upstream
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        // The api key + model are named "my-gpt4"; the policy below
        // pins applies_to to a different model name so no request in
        // this test matches.
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        seed_cache_policy_with_applies_to(&snap, "scoped", "model:not-my-gpt4");
        let state = build_state_with_cache(snap, hub);

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };

        // All three calls: 200, no `x-aisix-cache` header (policy gate
        // closed for this model). The wiremock `.expect(3)` above is
        // the load-bearing assertion — it fails the test at server
        // teardown if any call short-circuited via the cache.
        for _ in 0..3 {
            let resp = run(build_router(state.clone()), make_req()).await;
            assert_eq!(resp.status(), StatusCode::OK);
            assert!(
                resp.headers().get("x-aisix-cache").is_none(),
                "policy-gate-closed responses must not carry x-aisix-cache",
            );
        }
    }

    /// Issue #154 regression: a CachePolicy with `enabled: false`
    /// must NOT cache. The disabled policy must be filtered out by
    /// the find-first-enabled predicate at the chat.rs cache gate;
    /// every identical request must reach the upstream and the
    /// response must NOT carry an `x-aisix-cache` header (per the
    /// "policy-gate-closed = no header" contract pinned by the
    /// `applies_to_filters_out_unmatched_model` test above).
    #[tokio::test]
    async fn disabled_cache_policy_does_not_cache_and_emits_no_header() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-uncached",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "always-fresh"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .expect(3) // each call must reach the upstream
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        // The disabled policy applies_to="all" (would match every
        // request), but `enabled: false` MUST cause the find-first-
        // enabled predicate to skip it.
        seed_cache_policy_disabled(&snap, "off-policy");
        let state = build_state_with_cache(snap, hub);

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };

        // All three calls: 200, no `x-aisix-cache` header (policy
        // is disabled). wiremock's `.expect(3)` fails the test if
        // any call short-circuited via cache (= the disable flag
        // wasn't honored).
        for _ in 0..3 {
            let resp = run(build_router(state.clone()), make_req()).await;
            assert_eq!(resp.status(), StatusCode::OK);
            assert!(
                resp.headers().get("x-aisix-cache").is_none(),
                "disabled cache_policy must not emit x-aisix-cache header"
            );
        }
    }

    #[tokio::test]
    async fn applies_to_model_caches_matched_model() {
        // Counterpart to the negative test above: when the policy
        // `applies_to` matches the request's model, the cache gate
        // opens and the second identical request hits.
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-up",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "matched"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .expect(1) // only one upstream hit; second call must come from cache
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        seed_cache_policy_with_applies_to(&snap, "scoped", "model:my-gpt4");
        let state = build_state_with_cache(snap, hub);
        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let make_req = || {
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("authorization", "Bearer sk-caller")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap()
        };

        let r1 = run(build_router(state.clone()), make_req()).await;
        assert_eq!(r1.status(), StatusCode::OK);
        assert_eq!(
            r1.headers()
                .get("x-aisix-cache")
                .and_then(|v| v.to_str().ok()),
            Some("miss"),
        );

        let r2 = run(build_router(state), make_req()).await;
        assert_eq!(r2.status(), StatusCode::OK);
        assert_eq!(
            r2.headers()
                .get("x-aisix-cache")
                .and_then(|v| v.to_str().ok()),
            Some("hit"),
        );
    }

    /// Build a `ResourceEntry<Model>` with a non-default id so the test
    /// can mount multiple Models in one snapshot. `pk_id` lets each
    /// model point at its own ProviderKey row — useful for routing
    /// tests that use multiple upstream MockServers.
    fn model_entry_with_id(id: &str, name: &str, pk_id: &str) -> ResourceEntry<Model> {
        let cfg = format!(
            r#"{{
                "display_name": "{name}",
                "provider": "openai",
                "model_name": "gpt-4o",
                "provider_key_id": "{pk_id}"
            }}"#
        );
        let model: Model = serde_json::from_str(&cfg).unwrap();
        ResourceEntry::new(id, model, 1)
    }

    fn pk_entry_with_id(pk_id: &str, api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        let cfg = format!(
            r#"{{"display_name":"openai-{pk_id}","secret":"sk-upstream","api_base":"{api_base}","provider":"openai","adapter":"openai"}}"#
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&cfg).unwrap();
        ResourceEntry::new(pk_id, pk, 1)
    }

    /// Build a virtual routing Model that points at `targets` (other
    /// Model.display_name values) using the given strategy.
    fn routing_entry(
        name: &str,
        strategy: &str,
        targets: &[&str],
        retries: Option<u32>,
        max_fallbacks: Option<u32>,
        retry_on_429: Option<bool>,
    ) -> ResourceEntry<Model> {
        let target_objs: Vec<serde_json::Value> = targets
            .iter()
            .map(|t| serde_json::json!({"model": t}))
            .collect();
        let cfg = serde_json::json!({
            "display_name": name,
            "routing": {
                "strategy": strategy,
                "targets": target_objs,
                "retries": retries,
                "max_fallbacks": max_fallbacks,
                "retry_on_429": retry_on_429,
            }
        });
        let model: Model = serde_json::from_value(cfg).unwrap();
        ResourceEntry::new(format!("router-{name}"), model, 1)
    }

    #[tokio::test]
    async fn routing_failover_retries_to_second_target_when_first_5xxs() {
        let bad_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(502).set_body_string("upstream down"))
            .mount(&bad_upstream)
            .await;

        let good_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-good",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "fallback worked"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .mount(&good_upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(pk_entry_with_id("pk-bad", &bad_upstream.uri()));
        snap.provider_keys
            .insert(pk_entry_with_id("pk-good", &good_upstream.uri()));
        snap.models
            .insert(model_entry_with_id("m-bad", "primary", "pk-bad"));
        snap.models
            .insert(model_entry_with_id("m-good", "secondary", "pk-good"));
        snap.models.insert(routing_entry(
            "smart",
            "failover",
            &["primary", "secondary"],
            None,
            None,
            None,
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["smart"]));

        let app = build_router(build_state(snap, hub));
        let body = serde_json::json!({
            "model": "smart",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        assert_eq!(
            status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&bytes)
        );
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], "fallback worked");
    }

    #[tokio::test]
    async fn routing_propagates_4xx_without_attempting_fallback() {
        // First target returns 400 — caller mistake, no point trying
        // the second target. We assert the request fails 400 *and* the
        // second wiremock never sees a request.
        let bad_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(400).set_body_string("invalid_request"))
            .expect(1)
            .mount(&bad_upstream)
            .await;

        let standby_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            // Should never be hit; expect(0) enforces it on Drop.
            .respond_with(ResponseTemplate::new(200))
            .expect(0)
            .mount(&standby_upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(pk_entry_with_id("pk-bad", &bad_upstream.uri()));
        snap.provider_keys
            .insert(pk_entry_with_id("pk-standby", &standby_upstream.uri()));
        snap.models
            .insert(model_entry_with_id("m-bad", "primary", "pk-bad"));
        snap.models
            .insert(model_entry_with_id("m-standby", "secondary", "pk-standby"));
        snap.models.insert(routing_entry(
            "smart",
            "failover",
            &["primary", "secondary"],
            None,
            None,
            None,
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["smart"]));

        let app = build_router(build_state(snap, hub));
        let body = serde_json::json!({
            "model": "smart",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "{}",
            String::from_utf8_lossy(&bytes)
        );
    }

    #[tokio::test]
    async fn routing_retries_current_target_before_failover() {
        use aisix_obs::UsageSink;

        let flaky_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(502).set_body_string("try again"))
            .expect(2)
            .mount(&flaky_upstream)
            .await;

        let good_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-good",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "after retries"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .expect(1)
            .mount(&good_upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(pk_entry_with_id("pk-flaky", &flaky_upstream.uri()));
        snap.provider_keys
            .insert(pk_entry_with_id("pk-good", &good_upstream.uri()));
        snap.models
            .insert(model_entry_with_id("m-flaky", "primary", "pk-flaky"));
        snap.models
            .insert(model_entry_with_id("m-good", "secondary", "pk-good"));
        snap.models.insert(routing_entry(
            "smart",
            "failover",
            &["primary", "secondary"],
            Some(1),
            Some(1),
            None,
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["smart"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_router(build_state(snap, hub).with_usage_sink(UsageSink::new(tx)));
        let body = serde_json::json!({
            "model": "smart",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        assert_eq!(
            status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&bytes)
        );
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], "after retries");

        // Per #655 each upstream attempt emits its own UsageEvent, all
        // sharing `request_id`. Here: primary fails (initial), primary
        // fails again (retry), secondary succeeds (fallback) — 3 events.
        let mut events = Vec::new();
        for _ in 0..3 {
            let ev = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
                .await
                .expect("usage event was never emitted")
                .expect("sender dropped");
            events.push(ev);
        }
        events.sort_by_key(|e| e.attempt_index);
        let rid = events[0].request_id.clone();
        assert!(
            !rid.is_empty() && events.iter().all(|e| e.request_id == rid),
            "all attempts share the request_id (trace key)"
        );

        // initial attempt on `primary` failed with the upstream's 502
        assert_eq!(events[0].attempt_index, 0);
        assert_eq!(events[0].attempt_kind, "initial");
        assert_eq!(events[0].attempt_model, "primary");
        assert_eq!(events[0].status_code, 502);
        assert_eq!(events[0].error_class, "upstream_status");
        assert_eq!(events[0].prompt_tokens, 0);
        assert_eq!(events[0].completion_tokens, 0);

        // retry on the SAME target, also failed
        assert_eq!(events[1].attempt_index, 1);
        assert_eq!(events[1].attempt_kind, "retry");
        assert_eq!(events[1].attempt_model, "primary");
        assert_eq!(events[1].status_code, 502);
        assert!(!events[1].error_class.is_empty());

        // fallback to `secondary` succeeded and carries the real tokens
        assert_eq!(events[2].attempt_index, 2);
        assert_eq!(events[2].attempt_kind, "fallback");
        assert_eq!(events[2].attempt_model, "secondary");
        assert_eq!(events[2].status_code, 200);
        assert_eq!(events[2].error_class, "");
        assert_eq!(events[2].prompt_tokens, 1);
        assert_eq!(events[2].completion_tokens, 1);
    }

    #[tokio::test]
    async fn streaming_routing_records_failed_initial_attempt() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(502).set_body_string("stream unavailable"))
            .expect(1)
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(pk_entry_with_id("pk-primary", &upstream.uri()));
        snap.models
            .insert(model_entry_with_id("m-primary", "primary", "pk-primary"));
        snap.models.insert(routing_entry(
            "smart",
            "failover",
            &["primary"],
            None,
            None,
            None,
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["smart"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_router(build_state(snap, hub).with_usage_sink(UsageSink::new(tx)));
        let body = serde_json::json!({
            "model": "smart",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);

        // Streaming attempts only the first target (#655): the single
        // failed initial attempt is emitted as one per-attempt event.
        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("usage event was never emitted")
            .expect("sender dropped");
        assert_eq!(event.attempt_index, 0);
        assert_eq!(event.attempt_kind, "initial");
        assert_eq!(event.attempt_model, "primary");
        assert_eq!(event.status_code, 502);
        assert_eq!(event.error_class, "upstream_status");
        assert_eq!(event.prompt_tokens, 0);
        // No further events for this single-attempt request.
        if let Ok(Some(extra)) =
            tokio::time::timeout(std::time::Duration::from_millis(100), rx.recv()).await
        {
            panic!(
                "unexpected 2nd event: idx={} kind={} model={} status={} err={}",
                extra.attempt_index,
                extra.attempt_kind,
                extra.attempt_model,
                extra.status_code,
                extra.error_class
            );
        }
    }

    #[tokio::test]
    async fn messages_routing_emits_per_attempt_events() {
        // Per #655 the /v1/messages family must emit one UsageEvent per
        // upstream attempt, just like /v1/chat/completions. A Model Group
        // whose primary 502s and secondary succeeds emits 2 events sharing
        // request_id, tagged inbound_protocol="anthropic".
        use aisix_obs::UsageSink;

        let bad_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(502).set_body_string("upstream down"))
            .expect(1)
            .mount(&bad_upstream)
            .await;

        let good_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-good",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "fallback worked"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .expect(1)
            .mount(&good_upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(pk_entry_with_id("pk-bad", &bad_upstream.uri()));
        snap.provider_keys
            .insert(pk_entry_with_id("pk-good", &good_upstream.uri()));
        snap.models
            .insert(model_entry_with_id("m-bad", "primary", "pk-bad"));
        snap.models
            .insert(model_entry_with_id("m-good", "secondary", "pk-good"));
        snap.models.insert(routing_entry(
            "smart",
            "failover",
            &["primary", "secondary"],
            None,
            None,
            None,
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["smart"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_router(build_state(snap, hub).with_usage_sink(UsageSink::new(tx)));
        let body = serde_json::json!({
            "model": "smart",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        assert_eq!(
            status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&bytes)
        );

        let mut events = Vec::new();
        for _ in 0..2 {
            let ev = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
                .await
                .expect("usage event was never emitted")
                .expect("sender dropped");
            events.push(ev);
        }
        events.sort_by_key(|e| e.attempt_index);
        let rid = events[0].request_id.clone();
        assert!(
            !rid.is_empty() && events.iter().all(|e| e.request_id == rid),
            "all attempts share the request_id (trace key)"
        );
        assert!(
            events.iter().all(|e| e.inbound_protocol == "anthropic"),
            "/v1/messages tags inbound_protocol=anthropic on every attempt"
        );

        // initial attempt on `primary` failed with the upstream's 502
        assert_eq!(events[0].attempt_index, 0);
        assert_eq!(events[0].attempt_kind, "initial");
        assert_eq!(events[0].attempt_model, "primary");
        assert_eq!(events[0].status_code, 502);
        assert_eq!(events[0].error_class, "upstream_status");
        assert_eq!(events[0].prompt_tokens, 0);
        assert_eq!(events[0].completion_tokens, 0);

        // fallback to `secondary` succeeded with real tokens
        assert_eq!(events[1].attempt_index, 1);
        assert_eq!(events[1].attempt_kind, "fallback");
        assert_eq!(events[1].attempt_model, "secondary");
        assert_eq!(events[1].status_code, 200);
        assert_eq!(events[1].error_class, "");
        assert_eq!(events[1].prompt_tokens, 1);
        assert_eq!(events[1].completion_tokens, 1);
    }

    #[tokio::test]
    async fn responses_routing_emits_per_attempt_events() {
        // Per #655 the /v1/responses family must emit one UsageEvent per
        // upstream attempt. A Model Group whose primary 502s and secondary
        // succeeds emits 2 events sharing request_id, inbound_protocol="openai".
        use aisix_obs::UsageSink;

        let bad_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(502).set_body_string("upstream down"))
            .expect(1)
            .mount(&bad_upstream)
            .await;

        let good_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "resp-good",
                "object": "response",
                "output": [],
                "usage": {"input_tokens": 1, "output_tokens": 1}
            })))
            .expect(1)
            .mount(&good_upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(pk_entry_with_id("pk-bad", &bad_upstream.uri()));
        snap.provider_keys
            .insert(pk_entry_with_id("pk-good", &good_upstream.uri()));
        snap.models
            .insert(model_entry_with_id("m-bad", "primary", "pk-bad"));
        snap.models
            .insert(model_entry_with_id("m-good", "secondary", "pk-good"));
        snap.models.insert(routing_entry(
            "smart",
            "failover",
            &["primary", "secondary"],
            None,
            None,
            None,
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["smart"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_router(build_state(snap, hub).with_usage_sink(UsageSink::new(tx)));
        let body = serde_json::json!({
            "model": "smart",
            "input": "hi"
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/responses")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        assert_eq!(
            status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&bytes)
        );

        let mut events = Vec::new();
        for _ in 0..2 {
            let ev = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
                .await
                .expect("usage event was never emitted")
                .expect("sender dropped");
            events.push(ev);
        }
        events.sort_by_key(|e| e.attempt_index);
        let rid = events[0].request_id.clone();
        assert!(
            !rid.is_empty() && events.iter().all(|e| e.request_id == rid),
            "all attempts share the request_id (trace key)"
        );
        assert!(
            events.iter().all(|e| e.inbound_protocol == "openai"),
            "/v1/responses tags inbound_protocol=openai on every attempt"
        );

        // initial attempt on `primary` failed with the upstream's 502
        assert_eq!(events[0].attempt_index, 0);
        assert_eq!(events[0].attempt_kind, "initial");
        assert_eq!(events[0].attempt_model, "primary");
        assert_eq!(events[0].status_code, 502);
        assert_eq!(events[0].error_class, "upstream_status");
        assert_eq!(events[0].prompt_tokens, 0);

        // fallback to `secondary` succeeded with real tokens
        assert_eq!(events[1].attempt_index, 1);
        assert_eq!(events[1].attempt_kind, "fallback");
        assert_eq!(events[1].attempt_model, "secondary");
        assert_eq!(events[1].status_code, 200);
        assert_eq!(events[1].error_class, "");
        assert_eq!(events[1].prompt_tokens, 1);
        assert_eq!(events[1].completion_tokens, 1);
    }

    #[tokio::test]
    async fn routing_can_retry_and_failover_on_429_when_enabled() {
        let ratelimited_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(429).set_body_string("slow down"))
            .expect(2)
            .mount(&ratelimited_upstream)
            .await;

        let good_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-good",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "429 fallback worked"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .expect(1)
            .mount(&good_upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(pk_entry_with_id("pk-429", &ratelimited_upstream.uri()));
        snap.provider_keys
            .insert(pk_entry_with_id("pk-good", &good_upstream.uri()));
        snap.models
            .insert(model_entry_with_id("m-429", "primary", "pk-429"));
        snap.models
            .insert(model_entry_with_id("m-good", "secondary", "pk-good"));
        snap.models.insert(routing_entry(
            "smart",
            "failover",
            &["primary", "secondary"],
            Some(1),
            Some(1),
            Some(true),
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["smart"]));

        let app = build_router(build_state(snap, hub));
        let body = serde_json::json!({
            "model": "smart",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        assert_eq!(
            status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&bytes)
        );
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], "429 fallback worked");
    }

    #[tokio::test]
    async fn routing_skips_target_in_runtime_cooldown() {
        let cooled_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-cooled",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "should not be called"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .expect(0)
            .mount(&cooled_upstream)
            .await;

        let good_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-good",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "cooldown skipped"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .expect(1)
            .mount(&good_upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(pk_entry_with_id("pk-cooled", &cooled_upstream.uri()));
        snap.provider_keys
            .insert(pk_entry_with_id("pk-good", &good_upstream.uri()));
        snap.models
            .insert(model_entry_with_id("m-cooled", "primary", "pk-cooled"));
        snap.models
            .insert(model_entry_with_id("m-good", "secondary", "pk-good"));
        snap.models.insert(routing_entry(
            "smart",
            "failover",
            &["primary", "secondary"],
            None,
            None,
            None,
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["smart"]));

        let state = build_state(snap, hub);
        state.runtime_status.mark_cooldown(
            "m-cooled",
            std::time::Duration::from_secs(30),
            "retryable_failure",
        );
        let app = build_router(state);
        let body = serde_json::json!({
            "model": "smart",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        assert_eq!(
            status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&bytes)
        );
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], "cooldown skipped");
    }

    #[tokio::test]
    async fn routing_ignores_cooldown_when_it_would_empty_all_candidates() {
        let primary_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-primary",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "cooldown fallback"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .expect(1)
            .mount(&primary_upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(pk_entry_with_id("pk-primary", &primary_upstream.uri()));
        snap.models
            .insert(model_entry_with_id("m-primary", "primary", "pk-primary"));
        snap.models.insert(routing_entry(
            "smart",
            "failover",
            &["primary"],
            None,
            None,
            None,
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["smart"]));

        let state = build_state(snap, hub);
        state.runtime_status.mark_cooldown(
            "m-primary",
            std::time::Duration::from_secs(30),
            "retryable_failure",
        );
        let app = build_router(state);
        let body = serde_json::json!({
            "model": "smart",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        assert_eq!(
            status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&bytes)
        );
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], "cooldown fallback");
    }

    #[tokio::test]
    async fn routing_retryable_failure_puts_target_into_cooldown_for_next_request() {
        let flaky_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(502).set_body_string("temporary upstream failure"))
            .expect(1)
            .mount(&flaky_upstream)
            .await;

        let stable_upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-stable",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "stable fallback"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .expect(2)
            .mount(&stable_upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(pk_entry_with_id("pk-flaky", &flaky_upstream.uri()));
        snap.provider_keys
            .insert(pk_entry_with_id("pk-stable", &stable_upstream.uri()));
        snap.models
            .insert(model_entry_with_id("m-flaky", "primary", "pk-flaky"));
        snap.models
            .insert(model_entry_with_id("m-stable", "secondary", "pk-stable"));
        snap.models.insert(routing_entry(
            "smart",
            "failover",
            &["primary", "secondary"],
            Some(0),
            Some(1),
            None,
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["smart"]));

        let state = build_state(snap, hub);
        let app = build_router(state.clone());
        let body = serde_json::json!({
            "model": "smart",
            "messages": [{"role": "user", "content": "first"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            state.runtime_status.status("m-flaky").status,
            RuntimeStatus::Cooldown
        );

        let app = build_router(state);
        let body = serde_json::json!({
            "model": "smart",
            "messages": [{"role": "user", "content": "second"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        assert_eq!(
            status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&bytes)
        );
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], "stable fallback");
    }

    #[tokio::test]
    async fn routing_to_missing_target_returns_400() {
        // Routing references a Model that isn't in the snapshot — this
        // is a misconfiguration and should surface as a clean 400.
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));

        let snap = AisixSnapshot::new();
        snap.models.insert(routing_entry(
            "smart",
            "failover",
            &["nonexistent"],
            None,
            None,
            None,
        ));
        snap.apikeys.insert(apikey_entry("sk-caller", &["smart"]));
        // No upstream provider_key needed — the routing target itself
        // is missing so dispatch fails before any provider lookup.

        let app = build_router(build_state(snap, hub));
        let body = serde_json::json!({
            "model": "smart",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn ratelimit_response_headers_are_injected_on_success() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-up",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot_with_limits(
            "my-gpt4",
            &["my-gpt4"],
            &upstream.uri(),
            serde_json::json!({"rpm": 100, "tpm": 50000}),
        );
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let headers = resp.headers();
        assert!(
            headers.contains_key("x-ratelimit-limit-requests"),
            "missing x-ratelimit-limit-requests"
        );
        assert_eq!(
            headers
                .get("x-ratelimit-limit-requests")
                .and_then(|v| v.to_str().ok()),
            Some("100"),
        );
        assert!(
            headers.contains_key("x-ratelimit-limit-tokens"),
            "missing x-ratelimit-limit-tokens"
        );
        assert_eq!(
            headers
                .get("x-ratelimit-limit-tokens")
                .and_then(|v| v.to_str().ok()),
            Some("50000"),
        );
        // Remaining should be limit - 1 (one request consumed).
        assert_eq!(
            headers
                .get("x-ratelimit-remaining-requests")
                .and_then(|v| v.to_str().ok()),
            Some("99"),
        );
    }

    #[tokio::test]
    async fn input_guardrail_block_returns_422_and_skips_upstream() {
        // wiremock that fails the test if it's hit at all.
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0) // hard expectation — guardrail must short-circuit
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let state = build_state(snap, hub);
        seed_guardrail(
            &state.snapshot,
            "g-input-block",
            r#"{"name":"input-guard","kind":"keyword","patterns":[{"kind":"literal","value":"forbidden-token"}]}"#,
        );
        let app = build_router(state);

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "say the forbidden-token please"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");
        // Per #153, the wire-level `error.message` MUST NOT carry
        // the matched-pattern detail. The previous assertion
        // `.contains("forbidden-token")` pinned the leaky behavior
        // (the literal value of the forbidden pattern showing up
        // in the caller-visible message). Redaction keeps the
        // matched literal in operator logs (`tracing`) only.
        let message = v["error"]["message"].as_str().unwrap();
        assert!(
            !message.contains("forbidden-token"),
            "wire-level error.message must not leak the matched literal; got {message:?}"
        );
        assert_eq!(message, "request blocked by content policy");
    }

    /// Regression: a guardrail-blocked request must record the resolved
    /// model_id on its telemetry event. Earlier the error path hard-coded
    /// model_id="" for every failure, which left the dashboard /logs
    /// "Guardrail blocks" tab showing an empty model column.
    #[tokio::test]
    async fn input_guardrail_block_records_resolved_model_id_in_telemetry() {
        use aisix_obs::UsageSink;

        // Capturing usage sink — we read the emitted event off the
        // receiver to assert telemetry shape, not just the HTTP response.
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0)
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let state = build_state(snap, hub);
        seed_guardrail(
            &state.snapshot,
            "g-input-block",
            r#"{"name":"input-guard","kind":"keyword","patterns":[{"kind":"literal","value":"forbidden-token"}]}"#,
        );
        let state = state.with_usage_sink(UsageSink::new(tx));
        let app = build_router(state);

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "say the forbidden-token please"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);

        // The seeded model_entry uses the literal id "model-id-1"
        // (see lib.rs::model_entry). Pinning the exact value catches
        // regressions where the id silently becomes empty.
        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("usage event was never emitted")
            .expect("sender dropped without sending");
        assert_eq!(event.model_id, "model-id-1");
        assert_eq!(event.status_code, 422);
        assert!(event.guardrail_blocked);
    }

    #[tokio::test]
    async fn output_guardrail_block_returns_422_after_upstream_runs() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-up",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "here is your secret-string"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            })))
            .expect(1) // upstream IS called; guardrail blocks the response
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let state = build_state(snap, hub);
        seed_guardrail(
            &state.snapshot,
            "g-output-block",
            r#"{"name":"output-guard","kind":"keyword","hook_point":"output","patterns":[{"kind":"literal","value":"secret-string"}]}"#,
        );
        let app = build_router(state);

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "anything"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");
        // Per #153, the matched literal from the model's response
        // ("secret-string" — what the upstream returned and the
        // guardrail matched) MUST NOT appear in the caller-visible
        // error envelope. Echoing it would defeat the entire point
        // of an output guardrail: anyone who can trigger the rule
        // could extract the model's forbidden output via the error
        // message. This is the most security-critical assertion
        // for the whole guardrail surface.
        let message = v["error"]["message"].as_str().unwrap();
        assert!(
            !message.contains("secret-string"),
            "output guardrail leaked the matched literal back to the caller; got {message:?}"
        );
        // The full error envelope (any field) must also be clean —
        // future regressions might leak via a different field
        // (param/code) so check the whole serialized blob.
        let blob = serde_json::to_string(&v).unwrap();
        assert!(
            !blob.contains("secret-string"),
            "output guardrail leaked the matched literal in the envelope; got {blob}"
        );
        assert_eq!(message, "response blocked by content policy");
    }

    /// Regression for #226: when an output-content-filter blocks a
    /// response that the upstream already produced, the telemetry event
    /// MUST carry the upstream-billed `prompt_tokens` /
    /// `completion_tokens` instead of zeroing them. Pre-fix the error
    /// path uniformly emitted zeros for every error variant — the
    /// "request never reached the upstream" assumption baked into the
    /// failure-path comment was wrong for the output-block case where
    /// the upstream HAS run and the provider has already charged.
    #[tokio::test]
    async fn output_guardrail_block_records_upstream_usage_in_telemetry() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-blocked-1",
                "model": "gpt-4o-2024-08-06",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "leak the secret-string"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
            })))
            .expect(1)
            .mount(&upstream)
            .await;

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let state = build_state(snap, hub);
        seed_guardrail(
            &state.snapshot,
            "g-output-block",
            r#"{"name":"output-guard","kind":"keyword","hook_point":"output","patterns":[{"kind":"literal","value":"secret-string"}]}"#,
        );
        let state = state.with_usage_sink(UsageSink::new(tx));
        let app = build_router(state);

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "anything"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("usage event was never emitted")
            .expect("sender dropped without sending");
        // The customer paid the provider for these tokens — telemetry
        // must reflect that, not silently drop them to 0.
        assert_eq!(
            event.prompt_tokens, 11,
            "output-block must preserve the upstream's prompt_tokens"
        );
        assert_eq!(
            event.completion_tokens, 7,
            "output-block must preserve the upstream's completion_tokens"
        );
        assert!(event.guardrail_blocked);
        assert_eq!(event.status_code, 422);
        assert_eq!(event.provider_request_id, "cmpl-blocked-1");
        assert_eq!(event.provider_model_version, "gpt-4o-2024-08-06");
        assert_eq!(event.finish_reason, "stop");
        // cache_status reflects the per-policy gate the request went
        // through; "disabled" here because the test seeds no
        // cache_policy. A regression that drops cache_status on the
        // output-block path would surface as empty-string here.
        assert_eq!(event.cache_status, "disabled");
    }

    /// Regression for ai-gateway#196 audit HIGH-1: streaming chat
    /// telemetry must fire even when the client disconnects mid-
    /// stream. Pre-fix, on_complete lived in a `yield`-following
    /// branch of the async_stream! body that only ran when the
    /// consumer pulled — a dropped consumer (axum aborting the
    /// response future) skipped it entirely, so the customer's
    /// upstream call billed but the gateway recorded zero events.
    /// Post-fix, on_complete fires from a Drop guard so cancellation
    /// still produces a usage_event (with whatever counts were
    /// captured up to disconnect, typically 0 if disconnect beat
    /// the upstream's `usage` chunk).
    #[tokio::test]
    async fn streaming_chat_telemetry_fires_on_client_disconnect() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        // Use a slow drip so we can disconnect before [DONE] arrives.
        let sse = "\
data: {\"id\":\"cmpl-cancel-1\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"cmpl-cancel-1\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"cmpl-cancel-1\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&upstream)
            .await;

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let state = build_state(snap, hub).with_usage_sink(UsageSink::new(tx));
        let app = build_router(state);

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        // Read ONE chunk then drop the response body — simulates a
        // client that hung up before the upstream's terminal chunk.
        // The Drop guard inside build_sse_stream must still fire
        // on_complete for this disconnected request.
        let mut body_stream = resp.into_body().into_data_stream();
        let _first = body_stream.next().await;
        drop(body_stream);

        let event = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
            .await
            .expect("usage event was never emitted (Drop guard regression)")
            .expect("sender dropped without sending");
        // The exact token counts depend on how many chunks reached
        // the guard before disconnect — could be 0 (disconnect beat
        // the upstream emission entirely) or more. The contract is
        // simply that an event fires; counts are best-effort. Pin
        // the structural fields to confirm we didn't grab some
        // unrelated event.
        assert_eq!(event.status_code, 200);
        assert!(!event.guardrail_blocked);
    }

    /// Regression for #225: streaming chat must read the terminal SSE
    /// chunk's `usage` block and forward those counts into the
    /// telemetry event. Pre-fix the streaming path captured only
    /// `total_tokens` (for rate-limit accounting) and dropped
    /// `prompt_tokens` / `completion_tokens` — telemetry recorded zero
    /// for every streamed request even though the DP had the real
    /// counts in hand.
    #[tokio::test]
    async fn streaming_chat_telemetry_records_usage_from_terminal_chunk() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        // OpenAI's stream_options.include_usage=true shape: the final
        // delta chunk before [DONE] carries a `usage` block.
        let sse = "\
data: {\"id\":\"cmpl-stream-1\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"cmpl-stream-1\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"cmpl-stream-1\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":13,\"completion_tokens\":4,\"total_tokens\":17}}\n\n\
data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&upstream)
            .await;

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let state = build_state(snap, hub).with_usage_sink(UsageSink::new(tx));
        let app = build_router(state);

        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "stream_options": {"include_usage": true}
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        // Drain the SSE body so build_sse_stream's on_complete fires.
        // Telemetry emission is wired to that callback; the channel
        // stays empty until the full stream has been consumed.
        let mut body_stream = resp.into_body().into_data_stream();
        while let Some(chunk) = body_stream.next().await {
            let _ = chunk.unwrap();
        }

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("usage event was never emitted")
            .expect("sender dropped without sending");
        assert_eq!(
            event.prompt_tokens, 13,
            "streaming telemetry must capture prompt_tokens from the terminal chunk's usage block"
        );
        assert_eq!(
            event.completion_tokens, 4,
            "streaming telemetry must capture completion_tokens from the terminal chunk"
        );
        assert_eq!(event.status_code, 200);
        assert!(!event.guardrail_blocked);
        assert_eq!(event.provider_request_id, "cmpl-stream-1");
        assert_eq!(event.provider_model_version, "gpt-4o-2024-08-06");
        assert_eq!(event.finish_reason, "stop");
    }

    #[tokio::test]
    async fn budget_exceeded_returns_429() {
        use crate::budget::BudgetClient;

        // cp-api stand-in: returns a deny decision for our key.
        // Wire shape mirrors cp-api's budgetCheckResponse — see
        // internal/cpapi/resources/budget_check.go (prd-09b rev 2 §5.5).
        let cp = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/dp/budget_check"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "allow": false,
                "fail_mode": "closed",
                "reason": {
                    "type": "billing_error",
                    "code": "budget_exceeded",
                    "message": "monthly cap exceeded",
                    "scope": "api_key",
                    "scope_ref": "ak-uuid",
                    "limit_usd": "10.00",
                    "spent_usd": "10.50",
                    "period": "month",
                    "period_resets_at": "2026-05-01T00:00:00Z",
                    "retry_after_seconds": 86400
                }
            })))
            .mount(&cp)
            .await;

        // Upstream chat endpoint must NOT be hit — the budget check
        // fires before dispatch.
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200))
            .expect(0)
            .mount(&upstream)
            .await;

        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(openai_test_bridge()));

        let snap = seed_snapshot("my-gpt4", &["my-gpt4"], &upstream.uri());
        let state = build_state(snap, hub).with_budget_client(Arc::new(BudgetClient::new(
            cp.uri(),
            reqwest::Client::new(),
        )));

        let app = build_router(state);
        let body = serde_json::json!({
            "model": "my-gpt4",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();

        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "billing_error");
        assert_eq!(v["error"]["code"], "budget_exceeded");
    }

    // ─── Cross-protocol × upstream matrix ─────────────────────────
    //
    // Closes the gap noted in earlier review: the per-bridge wiremock
    // tests prove each Bridge translates ChatFormat ↔ its wire shape,
    // and the proxy lib tests above prove `/v1/chat/completions` end-
    // to-end against an OpenAi upstream — but the *integration* of an
    // OpenAI-protocol inbound request hitting an Anthropic / Gemini /
    // DeepSeek upstream had zero coverage. These tests pin the full
    // path: client body parser → Hub.get(provider) → Bridge.chat[_stream]
    // → upstream → Bridge response decoder → renderer → wire bytes.

    const MATRIX_ANTHROPIC_PK_ID: &str = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";
    const MATRIX_GOOGLE_PK_ID: &str = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb";
    const MATRIX_DEEPSEEK_PK_ID: &str = "cccccccc-cccc-cccc-cccc-cccccccccccc";
    const MATRIX_COHERE_PK_ID: &str = "dddddddd-dddd-dddd-dddd-dddddddddddd";

    fn anthropic_model_entry(name: &str) -> ResourceEntry<Model> {
        let cfg = format!(
            r#"{{
                "display_name": "{name}",
                "provider": "anthropic",
                "model_name": "claude-3-5-haiku-20241022",
                "provider_key_id": "{MATRIX_ANTHROPIC_PK_ID}"
            }}"#
        );
        ResourceEntry::new("model-anthropic-1", serde_json::from_str(&cfg).unwrap(), 1)
    }

    fn gemini_model_entry(name: &str) -> ResourceEntry<Model> {
        let cfg = format!(
            r#"{{
                "display_name": "{name}",
                "provider": "google",
                "model_name": "gemini-2.0-flash",
                "provider_key_id": "{MATRIX_GOOGLE_PK_ID}"
            }}"#
        );
        ResourceEntry::new("model-gemini-1", serde_json::from_str(&cfg).unwrap(), 1)
    }

    fn deepseek_model_entry(name: &str) -> ResourceEntry<Model> {
        let cfg = format!(
            r#"{{
                "display_name": "{name}",
                "provider": "deepseek",
                "model_name": "deepseek-chat",
                "provider_key_id": "{MATRIX_DEEPSEEK_PK_ID}"
            }}"#
        );
        ResourceEntry::new("model-deepseek-1", serde_json::from_str(&cfg).unwrap(), 1)
    }

    fn cohere_model_entry(name: &str) -> ResourceEntry<Model> {
        let cfg = format!(
            r#"{{
                "display_name": "{name}",
                "provider": "cohere",
                "model_name": "command-r",
                "provider_key_id": "{MATRIX_COHERE_PK_ID}"
            }}"#
        );
        ResourceEntry::new("model-cohere-1", serde_json::from_str(&cfg).unwrap(), 1)
    }

    fn matrix_pk_entry(
        id: &'static str,
        secret: &str,
        api_base: &str,
        provider: &str,
        adapter: &str,
    ) -> ResourceEntry<aisix_core::ProviderKey> {
        let cfg = format!(
            r#"{{"display_name":"matrix-up","secret":"{secret}","api_base":"{api_base}","provider":"{provider}","adapter":"{adapter}"}}"#
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&cfg).unwrap();
        ResourceEntry::new(id, pk, 1)
    }

    fn matrix_anthropic_pk(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        matrix_pk_entry(
            MATRIX_ANTHROPIC_PK_ID,
            "sk-ant-test",
            api_base,
            "anthropic",
            "anthropic",
        )
    }

    fn matrix_gemini_pk(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        matrix_pk_entry(
            MATRIX_GOOGLE_PK_ID,
            "ya29-test",
            api_base,
            "google",
            "openai",
        )
    }

    fn matrix_deepseek_pk(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        matrix_pk_entry(
            MATRIX_DEEPSEEK_PK_ID,
            "sk-deepseek",
            api_base,
            "deepseek",
            "openai",
        )
    }

    fn matrix_cohere_pk(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        matrix_pk_entry(
            MATRIX_COHERE_PK_ID,
            "cohere-key",
            api_base,
            "cohere",
            "openai",
        )
    }

    /// (OpenAI inbound) × (Anthropic upstream) × (non-streaming).
    /// The most valuable cross-protocol cell — exercises real wire-shape
    /// translation in both directions inside `AnthropicBridge::chat`.
    #[tokio::test]
    async fn matrix_openai_in_anthropic_upstream_non_streaming() {
        use aisix_provider_anthropic::AnthropicBridge;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("x-api-key", "sk-ant-test"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "msg_01",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello from Claude!"}],
                "model": "claude-3-5-haiku-20241022",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 4}
            })))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(matrix_anthropic_pk(&upstream.uri()));
        snap.models.insert(anthropic_model_entry("my-claude"));
        snap.apikeys
            .insert(apikey_entry("sk-caller", &["my-claude"]));
        let hub = Arc::new(Hub::new());
        hub.register_specialized("anthropic", Arc::new(AnthropicBridge::new()));
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "my-claude",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value =
            serde_json::from_slice(&to_bytes(resp.into_body(), 65536).await.unwrap()).unwrap();
        // OpenAI-shape wire on the way out.
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["role"], "assistant");
        assert_eq!(v["choices"][0]["message"]["content"], "Hello from Claude!");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert_eq!(v["usage"]["prompt_tokens"], 5);
        assert_eq!(v["usage"]["completion_tokens"], 4);
    }

    /// (OpenAI inbound) × (Anthropic upstream) × (streaming).
    /// Pin the SSE event-stream translation: AnthropicBridge ingests
    /// typed Anthropic events (message_start / content_block_delta /
    /// message_delta / message_stop) and emits flat OpenAI deltas.
    #[tokio::test]
    async fn matrix_openai_in_anthropic_upstream_streaming() {
        use aisix_provider_anthropic::AnthropicBridge;

        let upstream = MockServer::start().await;
        let sse = "\
event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_01\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-3-5-haiku-20241022\",\"stop_reason\":null,\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}\n\n\
event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hel\"}}\n\n\
event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}\n\n\
event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":2}}\n\n\
event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(matrix_anthropic_pk(&upstream.uri()));
        snap.models.insert(anthropic_model_entry("my-claude"));
        snap.apikeys
            .insert(apikey_entry("sk-caller", &["my-claude"]));
        let hub = Arc::new(Hub::new());
        hub.register_specialized("anthropic", Arc::new(AnthropicBridge::new()));
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "my-claude",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get(axum::http::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("text/event-stream"),
        );
        let body =
            String::from_utf8(to_bytes(resp.into_body(), 65536).await.unwrap().to_vec()).unwrap();
        // OpenAI-shape SSE deltas on the way out.
        assert!(
            body.contains("\"object\":\"chat.completion.chunk\""),
            "missing OpenAI chunk envelope in:\n{body}"
        );
        assert!(body.contains("\"content\":\"hel\""));
        assert!(body.contains("\"content\":\"lo\""));
        assert!(body.contains("\"finish_reason\":\"stop\""));
        assert!(body.contains("data: [DONE]"));
    }

    /// (OpenAI inbound) × (Gemini upstream). Gemini is served by the
    /// `Adapter::Openai` family bridge — cp-api stores the Gemini PK
    /// with `adapter: "openai"` and `api_base` pointing at Google's
    /// `/v1beta/openai` compat endpoint. The integration test pins
    /// that an inbound OpenAI request resolves through the family
    /// bridge and round-trips Gemini's OpenAI-shape response.
    #[tokio::test]
    async fn matrix_openai_in_gemini_upstream_non_streaming() {
        use aisix_core::Adapter;
        use aisix_provider_openai::OpenAiBridge;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-gemini",
                "model": "gemini-2.0-flash",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from Gemini!"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9}
            })))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.provider_keys.insert(matrix_gemini_pk(&upstream.uri()));
        snap.models.insert(gemini_model_entry("my-gemini"));
        snap.apikeys
            .insert(apikey_entry("sk-caller", &["my-gemini"]));
        let hub = Arc::new(Hub::new());
        hub.register_family(Adapter::Openai, Arc::new(OpenAiBridge::new()));
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "my-gemini",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value =
            serde_json::from_slice(&to_bytes(resp.into_body(), 65536).await.unwrap()).unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], "Hello from Gemini!");
        assert_eq!(v["usage"]["total_tokens"], 9);
    }

    /// (OpenAI inbound) × (DeepSeek upstream). DeepSeek is served by
    /// the `Adapter::Openai` family bridge — cp-api stores the
    /// DeepSeek PK with `adapter: "openai"` and `api_base` pointing
    /// at `https://api.deepseek.com`. The integration test pins
    /// that an inbound OpenAI request resolves through the family
    /// bridge and round-trips DeepSeek's OpenAI-shape response.
    #[tokio::test]
    async fn matrix_openai_in_deepseek_upstream_non_streaming() {
        use aisix_core::Adapter;
        use aisix_provider_openai::OpenAiBridge;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer sk-deepseek"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-deepseek",
                "model": "deepseek-chat",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from DeepSeek!"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 6, "completion_tokens": 7, "total_tokens": 13}
            })))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(matrix_deepseek_pk(&upstream.uri()));
        snap.models.insert(deepseek_model_entry("my-deepseek"));
        snap.apikeys
            .insert(apikey_entry("sk-caller", &["my-deepseek"]));
        let hub = Arc::new(Hub::new());
        hub.register_family(Adapter::Openai, Arc::new(OpenAiBridge::new()));
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "my-deepseek",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value =
            serde_json::from_slice(&to_bytes(resp.into_body(), 65536).await.unwrap()).unwrap();
        assert_eq!(
            v["choices"][0]["message"]["content"],
            "Hello from DeepSeek!"
        );
        assert_eq!(v["usage"]["total_tokens"], 13);
    }

    /// (OpenAI inbound) × (Cohere chat-compat upstream). Cohere serves
    /// an OpenAI-shape envelope at `/compatibility/v1/chat/completions`
    /// per <https://docs.cohere.com/reference/chat>; cp-api stores the
    /// Cohere PK with `adapter: "openai"` and `api_base` pointing at
    /// `https://api.cohere.com/compatibility/v1`. The integration test
    /// pins that an inbound OpenAI request resolves through the
    /// `Adapter::Openai` family bridge (no specialized "cohere"
    /// registration in this Hub) and round-trips Cohere's OpenAI-shape
    /// response.
    ///
    /// Backfills coverage lost when the #379 clean cut deleted the
    /// `cohere_chat_compat_round_trips_openai_envelope` unit test
    /// (which exercised `OpenAiBridge::with_name("cohere")`, a code
    /// path that no longer exists).
    #[tokio::test]
    async fn matrix_openai_in_cohere_chat_compat_non_streaming() {
        use aisix_core::Adapter;
        use aisix_provider_openai::OpenAiBridge;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer cohere-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-cohere",
                "model": "command-r",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from Cohere!"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
            })))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        // PK `api_base` points at the wiremock root the way cp-api's
        // adapter_map points real Cohere PKs at `…/compatibility/v1`.
        snap.provider_keys.insert(matrix_cohere_pk(&upstream.uri()));
        snap.models.insert(cohere_model_entry("my-cohere"));
        snap.apikeys
            .insert(apikey_entry("sk-caller", &["my-cohere"]));
        let hub = Arc::new(Hub::new());
        // Family-only registration — NO `register_specialized("cohere", …)`.
        // The whole point of the test is to prove the family bridge
        // serves Cohere chat-compat without a vendor-specific entry.
        hub.register_family(Adapter::Openai, Arc::new(OpenAiBridge::new()));
        let app = build_router(build_state(snap, hub));

        let body = serde_json::json!({
            "model": "my-cohere",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = run(app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let v: serde_json::Value =
            serde_json::from_slice(&to_bytes(resp.into_body(), 65536).await.unwrap()).unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], "Hello from Cohere!");
        assert_eq!(v["usage"]["total_tokens"], 7);
    }
}
