//! `/a2a/:agent` — the downstream-facing A2A gateway endpoint.
//!
//! AISIX fronts each registered A2A agent: a caller reaches an agent through
//! `/a2a/<agent>`, and its card is served (with the service URL rewritten to
//! point back at the gateway) at `/a2a/<agent>/.well-known/agent-card.json`.
//! The caller authenticates with an AISIX API key — the [`AuthenticatedKey`]
//! extractor rejects a missing or invalid key with `401` before the request
//! reaches the agent. The endpoint is rebuilt from the current configuration
//! snapshot on each request, so it always reflects the live `a2a_agents` set.
//!
//! A `message/send` (and every other JSON-RPC call) is governed by the SAME
//! pipeline as an LLM request, keyed on the caller's API key: per-agent access
//! control (the key's `allowed_agents`), rate-limit + budget (`quota::enforce`),
//! and a usage event into the shared sink. The upstream credential is held
//! gateway-side and never reaches the caller. Guardrails over A2A message
//! content are a later step.
//!
//! The request body is forwarded verbatim to the upstream agent, so the caller
//! speaks whichever A2A wire version the agent is pinned to; the gateway does
//! not translate between the 0.3 and 1.0 formats here.

use std::time::{Duration, Instant};

use aisix_a2a::{upstream_from_a2a_agent, A2aBridge, A2aError, HttpBridge};
use aisix_obs::{AccessLog, RequestOutcome, UsageEvent};
use axum::body::to_bytes;
use axum::extract::{Path, Request, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;

use crate::auth::AuthenticatedKey;
use crate::request_id::new_request_id;
use crate::state::ProxyState;

/// Bounded `model` metric label for /a2a requests — A2A has no resolved model,
/// and the agent name is a path segment (bounded by the registered set, but
/// kept as a fixed label to match the /mcp convention and #451).
const A2A_MODEL_LABEL: &str = "a2a";

/// Just enough of a JSON-RPC request to record the method for usage and echo
/// the id back in a synthesized error. Unknown fields are ignored.
#[derive(Deserialize)]
struct JsonRpcPeek {
    method: Option<String>,
    id: Option<serde_json::Value>,
}

/// Serve a JSON-RPC request to `/a2a/:agent`. Authentication (`401`), per-agent
/// ACL (`403`), and rate-limit + budget (`429` / budget error) gate the call
/// before the request is forwarded to the upstream agent; a usage event is
/// emitted either way.
pub async fn a2a_endpoint(
    auth: AuthenticatedKey,
    Path(agent): Path<String>,
    State(state): State<ProxyState>,
    request: Request,
) -> Response {
    let started = Instant::now();
    let request_id = request
        .extensions()
        .get::<crate::request_id::RequestId>()
        .map(|r| r.0.clone())
        .unwrap_or_else(new_request_id);
    let api_key_id = auth.entry.id.clone();
    let http_method = request.method().clone();

    let response = dispatch(auth, &agent, &state, request, &request_id).await;

    let elapsed = started.elapsed();
    let status = response.status().as_u16();
    AccessLog {
        method: http_method.as_str(),
        path: "/a2a",
        status,
        latency: elapsed,
        provider: Some("a2a"),
        model: None,
        api_key_id: Some(&api_key_id),
        prompt_tokens: None,
        completion_tokens: None,
        total_tokens: None,
        request_id: &request_id,
        served_by_model: None,
        routing_attempt_count: None,
        routing_fallback_count: None,
    }
    .emit();
    state.metrics.record_request(
        "a2a",
        A2A_MODEL_LABEL,
        status,
        RequestOutcome::from_status(status),
        elapsed,
    );
    response
}

async fn dispatch(
    auth: AuthenticatedKey,
    agent: &str,
    state: &ProxyState,
    request: Request,
    request_id: &str,
) -> Response {
    // Resolve the agent from the live snapshot. A disabled agent is treated as
    // absent — not served, same as a missing one.
    let snapshot = state.snapshot.load();
    let entry = match snapshot.a2a_agents.get_by_name(agent) {
        Some(entry) if entry.value.enabled => entry,
        _ => return (StatusCode::NOT_FOUND, format!("unknown A2A agent: {agent}")).into_response(),
    };

    // Per-agent access control, keyed on the same API key object as LLM/MCP
    // access. A key with no `allowed_agents` reaches none (grant is explicit).
    if !auth.key().can_access_agent(agent) {
        return (
            StatusCode::FORBIDDEN,
            format!("this key may not reach A2A agent: {agent}"),
        )
            .into_response();
    }

    let upstream = match upstream_from_a2a_agent(&entry.value) {
        Ok(upstream) => upstream,
        // Currently only oauth2 upstream auth, which the runtime does not
        // implement yet — surface it as "not implemented".
        Err(err) => {
            emit_a2a_usage(
                state,
                &auth,
                request_id,
                agent,
                "",
                StatusCode::NOT_IMPLEMENTED.as_u16(),
                Duration::ZERO,
            );
            return (StatusCode::NOT_IMPLEMENTED, err.to_string()).into_response();
        }
    };

    let (_parts, body) = request.into_parts();
    let bytes = match to_bytes(body, state.request_body_limit_bytes).await {
        Ok(bytes) => bytes,
        Err(_) => return (StatusCode::BAD_REQUEST, "invalid request body").into_response(),
    };
    let value: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(_) => return (StatusCode::BAD_REQUEST, "invalid JSON-RPC body").into_response(),
    };
    let peek = serde_json::from_slice::<JsonRpcPeek>(&bytes).ok();
    let method = peek
        .as_ref()
        .and_then(|p| p.method.clone())
        .unwrap_or_default();
    let rpc_id = peek.and_then(|p| p.id);

    // Reuse the LLM path's rate-limit + budget gate. The reservation is held
    // for the call and dropped after (an A2A call carries no token cost yet).
    // On 429 / budget-exceeded this returns before the upstream is contacted.
    let _reservation = match crate::quota::enforce(state, &auth, None).await {
        Ok(reservation) => reservation,
        Err(err) => {
            let response = err.into_response();
            emit_a2a_usage(
                state,
                &auth,
                request_id,
                agent,
                &method,
                response.status().as_u16(),
                Duration::ZERO,
            );
            return response;
        }
    };

    let bridge = HttpBridge::new(upstream);
    let started = Instant::now();
    let result = bridge.send(&value).await;
    let latency = started.elapsed();

    match result {
        Ok(response_value) => {
            emit_a2a_usage(
                state,
                &auth,
                request_id,
                agent,
                &method,
                StatusCode::OK.as_u16(),
                latency,
            );
            axum::Json(response_value).into_response()
        }
        Err(err) => {
            let status = a2a_error_status(&err);
            tracing::warn!(agent = %agent, error = %err, "A2A upstream call failed");
            emit_a2a_usage(
                state,
                &auth,
                request_id,
                agent,
                &method,
                status.as_u16(),
                latency,
            );
            a2a_error_response(rpc_id, status, &err.to_string())
        }
    }
}

/// Serve the upstream agent's card at `/a2a/:agent/.well-known/agent-card.json`,
/// rewriting its advertised service `url` to point back at this gateway so
/// callers discover the agent through `/a2a/<agent>`.
pub async fn a2a_agent_card(
    auth: AuthenticatedKey,
    Path(agent): Path<String>,
    State(state): State<ProxyState>,
    headers: HeaderMap,
) -> Response {
    let snapshot = state.snapshot.load();
    let entry = match snapshot.a2a_agents.get_by_name(&agent) {
        Some(entry) if entry.value.enabled => entry,
        _ => return (StatusCode::NOT_FOUND, format!("unknown A2A agent: {agent}")).into_response(),
    };
    if !auth.key().can_access_agent(&agent) {
        return (
            StatusCode::FORBIDDEN,
            format!("this key may not reach A2A agent: {agent}"),
        )
            .into_response();
    }
    let upstream = match upstream_from_a2a_agent(&entry.value) {
        Ok(upstream) => upstream,
        Err(err) => return (StatusCode::NOT_IMPLEMENTED, err.to_string()).into_response(),
    };

    let bridge = HttpBridge::new(upstream);
    let mut card = match bridge.fetch_agent_card().await {
        Ok(card) => card,
        Err(err) => {
            tracing::warn!(agent = %agent, error = %err, "A2A agent card fetch failed");
            return (StatusCode::BAD_GATEWAY, err.to_string()).into_response();
        }
    };
    // Rewrite the advertised service endpoint to the gateway so downstream
    // callers route subsequent requests through `/a2a/<agent>`. Derived from
    // the request's Host header (and forwarded scheme) since the gateway's
    // public URL is not otherwise known here.
    if let Some(base) = gateway_base(&headers) {
        card.url = format!("{base}/a2a/{agent}");
    }
    axum::Json(card).into_response()
}

/// Reconstruct the gateway's public base (`scheme://host`) from request
/// headers: the `Host` header, and `X-Forwarded-Proto` when a proxy set it
/// (defaulting to `https`). Returns `None` when no Host header is present, in
/// which case the card's `url` is left as the upstream advertised it.
fn gateway_base(headers: &HeaderMap) -> Option<String> {
    let host = headers.get(header::HOST)?.to_str().ok()?;
    let scheme = headers
        .get("x-forwarded-proto")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("https");
    Some(format!("{scheme}://{host}"))
}

/// Map a bridge error to the client-visible HTTP status: an upstream that could
/// not be reached is a bad gateway; a not-yet-supported config is not
/// implemented; anything else from the call is a bad gateway too.
fn a2a_error_status(err: &A2aError) -> StatusCode {
    match err {
        A2aError::Unsupported(_) => StatusCode::NOT_IMPLEMENTED,
        A2aError::Connect(_) | A2aError::Request(_) => StatusCode::BAD_GATEWAY,
    }
}

/// Build a JSON-RPC error envelope for a gateway-side failure, echoing the
/// request id. A2A clients expect a JSON-RPC body, so the failure surfaces as
/// an error object they can handle rather than a bare HTTP error.
fn a2a_error_response(
    id: Option<serde_json::Value>,
    status: StatusCode,
    message: &str,
) -> Response {
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": id.unwrap_or(serde_json::Value::Null),
        "error": { "code": -32000, "message": message },
    });
    (status, axum::Json(body)).into_response()
}

/// Emit a usage event for a single A2A call into the same sink as LLM usage.
/// A2A calls carry no token cost yet, so token/cost fields stay zero; the event
/// records who called which agent with which method, the outcome, and latency.
fn emit_a2a_usage(
    state: &ProxyState,
    auth: &AuthenticatedKey,
    request_id: &str,
    agent: &str,
    method: &str,
    status_code: u16,
    latency: Duration,
) {
    let event = UsageEvent {
        request_id: request_id.to_string(),
        occurred_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        api_key_id: auth.entry.id.clone(),
        status_code,
        latency_ms: latency.as_millis().min(u32::MAX as u128) as u32,
        inbound_protocol: "a2a".to_string(),
        a2a_agent_name: agent.to_string(),
        a2a_method: method.to_string(),
        ..Default::default()
    };
    state.usage_sink.try_emit("a2a", event.clone());
    let snap = state.snapshot.load();
    let exporters = snap.observability_exporters.entries();
    state
        .otlp_fan_out
        .fan_out(&event, None, exporters.iter().map(|e| &e.value));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_status_maps_unsupported_to_501_and_transport_to_502() {
        assert_eq!(
            a2a_error_status(&A2aError::Unsupported("oauth2".into())),
            StatusCode::NOT_IMPLEMENTED
        );
        assert_eq!(
            a2a_error_status(&A2aError::Connect("dns".into())),
            StatusCode::BAD_GATEWAY
        );
        assert_eq!(
            a2a_error_status(&A2aError::Request("500".into())),
            StatusCode::BAD_GATEWAY
        );
    }

    #[test]
    fn gateway_base_uses_forwarded_proto_then_defaults_https() {
        let mut headers = HeaderMap::new();
        headers.insert(header::HOST, "gw.example.com".parse().unwrap());
        assert_eq!(
            gateway_base(&headers).as_deref(),
            Some("https://gw.example.com")
        );
        headers.insert("x-forwarded-proto", "http".parse().unwrap());
        assert_eq!(
            gateway_base(&headers).as_deref(),
            Some("http://gw.example.com")
        );
    }

    #[test]
    fn gateway_base_is_none_without_host() {
        assert_eq!(gateway_base(&HeaderMap::new()), None);
    }

    // ---- endpoint integration tests: drive the real router via oneshot ----
    use crate::build_router;
    use aisix_core::{A2aAgent, AisixSnapshot, ApiKey, ProxyConfig, ResourceEntry, SnapshotHandle};
    use axum::body::Body;
    use axum::http::Request as HttpRequest;
    use std::sync::Arc;
    use tower::ServiceExt;

    const TOKEN: &str = "sk-a2a-endpoint-test";

    fn proxy_cfg() -> ProxyConfig {
        ProxyConfig {
            addr: "127.0.0.1:0".into(),
            request_body_limit_bytes: 1_048_576,
            real_ip: Default::default(),
            tls: None,
        }
    }

    /// Snapshot with one API key (granting `allowed_agents`, or none when
    /// `allowed_agents` is `null`) and one `invoice` agent at `agent_url`.
    fn snapshot_with(
        agent_url: &str,
        enabled: bool,
        allowed_agents: serde_json::Value,
    ) -> AisixSnapshot {
        let mut key = serde_json::json!({
            "key_hash": ApiKey::hash_bearer(TOKEN),
            "allowed_models": ["*"],
        });
        if !allowed_agents.is_null() {
            key["allowed_agents"] = allowed_agents;
        }
        let apikey: ApiKey = serde_json::from_value(key).expect("valid apikey");
        let agent: A2aAgent = serde_json::from_value(serde_json::json!({
            "display_name": "invoice",
            "url": agent_url,
            "enabled": enabled,
        }))
        .expect("valid a2a agent");

        let snap = AisixSnapshot::new();
        snap.apikeys.insert(ResourceEntry::new("ak-1", apikey, 1));
        snap.a2a_agents.insert(ResourceEntry::new("ag-1", agent, 1));
        snap
    }

    fn router_with(snap: AisixSnapshot) -> axum::Router {
        let handle = SnapshotHandle::new(snap);
        let hub = Arc::new(aisix_gateway::Hub::new());
        build_router(ProxyState::new(handle, hub, &proxy_cfg()).without_cache())
    }

    fn a2a_post(agent: &str, auth: bool) -> HttpRequest<Body> {
        let body = serde_json::json!({"jsonrpc": "2.0", "id": 1, "method": "message/send"});
        let mut b = HttpRequest::post(format!("/a2a/{agent}"))
            .header("host", "a2a.aisix.example.com")
            .header("content-type", "application/json");
        if auth {
            b = b.header("authorization", format!("Bearer {TOKEN}"));
        }
        b.body(Body::from(body.to_string())).unwrap()
    }

    #[tokio::test]
    async fn endpoint_denies_key_without_allowed_agents_403() {
        // Unreachable upstream on purpose: the ACL must reject BEFORE any
        // upstream call is made.
        let app = router_with(snapshot_with(
            "http://127.0.0.1:1/a2a",
            true,
            serde_json::Value::Null,
        ));
        let resp = app.oneshot(a2a_post("invoice", true)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn endpoint_disabled_agent_is_404() {
        let app = router_with(snapshot_with(
            "http://127.0.0.1:1/a2a",
            false,
            serde_json::json!(["*"]),
        ));
        let resp = app.oneshot(a2a_post("invoice", true)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn endpoint_unknown_agent_is_404() {
        let app = router_with(snapshot_with(
            "http://127.0.0.1:1/a2a",
            true,
            serde_json::json!(["*"]),
        ));
        let resp = app.oneshot(a2a_post("does-not-exist", true)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn endpoint_missing_key_is_401() {
        let app = router_with(snapshot_with(
            "http://127.0.0.1:1/a2a",
            true,
            serde_json::json!(["*"]),
        ));
        let resp = app.oneshot(a2a_post("invoice", false)).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    /// A stub upstream that serves an agent card advertising an internal URL.
    async fn spawn_card_stub() -> std::net::SocketAddr {
        let app = axum::Router::new().route(
            "/.well-known/agent-card.json",
            axum::routing::get(|| async {
                axum::Json(serde_json::json!({
                    "name": "Invoice Agent",
                    "url": "https://upstream.internal/a2a",
                    "version": "2.1.0",
                    "skills": [{"id": "extract"}]
                }))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app.into_make_service())
                .await
                .unwrap();
        });
        addr
    }

    #[tokio::test]
    async fn endpoint_rewrites_agent_card_url_to_gateway() {
        let addr = spawn_card_stub().await;
        let app = router_with(snapshot_with(
            &format!("http://{addr}/a2a"),
            true,
            serde_json::json!(["*"]),
        ));
        let req = HttpRequest::get("/a2a/invoice/.well-known/agent-card.json")
            .header("host", "a2a.aisix.example.com")
            .header("authorization", format!("Bearer {TOKEN}"))
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), 1_048_576)
            .await
            .unwrap();
        let card: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        // The advertised service URL is rewritten to the gateway; the caller's
        // Host is reflected and every other card field is preserved.
        assert_eq!(card["url"], "https://a2a.aisix.example.com/a2a/invoice");
        assert_eq!(card["name"], "Invoice Agent");
        assert_eq!(card["version"], "2.1.0");
        assert_eq!(card["skills"][0]["id"], "extract");
    }
}
