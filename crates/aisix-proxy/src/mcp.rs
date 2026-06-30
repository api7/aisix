//! `/mcp` — the downstream-facing MCP gateway endpoint.
//!
//! AISIX presents as a single MCP server to a downstream agent: it aggregates
//! the tools of the registered `mcp_servers` and routes tool calls back to
//! them. The caller authenticates with an AISIX API key — the
//! [`AuthenticatedKey`] extractor rejects a missing or invalid key with `401`
//! before the request reaches the gateway. The gateway is rebuilt from the
//! current configuration snapshot on each request, so it always reflects the
//! live `mcp_servers` set.
//!
//! Per-tool access control, guardrail / quota reuse, and usage logging over MCP
//! traffic are layered on in subsequent steps; this step establishes the
//! authenticated, snapshot-sourced endpoint.

use std::time::{Duration, Instant};

use aisix_obs::UsageEvent;
use axum::body::{to_bytes, Body};
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use tower::ServiceExt;

use crate::auth::AuthenticatedKey;
use crate::state::ProxyState;

/// Just enough of a JSON-RPC request to tell a tool call apart from the MCP
/// handshake / discovery methods, and to recover the called tool's name.
/// Unknown fields are ignored.
#[derive(Deserialize)]
struct JsonRpcPeek {
    method: Option<String>,
    params: Option<PeekParams>,
}

#[derive(Deserialize)]
struct PeekParams {
    /// The namespaced `<server>__<tool>` name on a `tools/call`.
    name: Option<String>,
}

/// Serve a `/mcp` request. The [`AuthenticatedKey`] extractor enforces a valid
/// AISIX API key (responding `401` otherwise). A `tools/call` is then subject to
/// the same rate-limit and budget governance as an LLM request — keyed on the
/// caller's API key — before being handled by an MCP gateway built from the
/// current snapshot's `mcp_servers`, and a usage event is emitted into the same
/// pipeline as LLM calls. The `initialize` / `tools/list` handshake and discovery
/// methods pass through ungated and unmetered.
pub async fn mcp_endpoint(
    auth: AuthenticatedKey,
    State(state): State<ProxyState>,
    request: Request,
) -> Response {
    // Buffer the body so the JSON-RPC method can be inspected, then rebuilt for
    // the gateway. The global body-limit layer has already capped the size.
    let (parts, body) = request.into_parts();
    let bytes = match to_bytes(body, state.request_body_limit_bytes).await {
        Ok(bytes) => bytes,
        Err(_) => return (StatusCode::BAD_REQUEST, "invalid request body").into_response(),
    };

    let peek = serde_json::from_slice::<JsonRpcPeek>(&bytes).ok();
    let is_tool_call = peek.as_ref().and_then(|p| p.method.as_deref()) == Some("tools/call");
    // Split the namespaced tool name into (server, tool) up front, owned, so it
    // survives the body being consumed when the request is rebuilt.
    let (mcp_server, mcp_tool) = if is_tool_call {
        peek.as_ref()
            .and_then(|p| p.params.as_ref())
            .and_then(|p| p.name.as_deref())
            .and_then(|name| name.split_once(aisix_mcp::TOOL_NAMESPACE_SEPARATOR))
            .map(|(server, tool)| (server.to_string(), tool.to_string()))
            .unwrap_or_default()
    } else {
        (String::new(), String::new())
    };

    // Reuse the LLM path's rate-limit + budget gate on the unit of work. The
    // reservation is held for the duration of the call and dropped after (no
    // tokens to commit — a tool call carries no token cost), which releases the
    // concurrency slot. On 429 / budget-exceeded this returns before any
    // upstream is contacted — and the rejected call is still recorded.
    let _reservation = if is_tool_call {
        match crate::quota::enforce(&state, &auth, None).await {
            Ok(reservation) => Some(reservation),
            Err(err) => {
                let response = err.into_response();
                emit_tool_call_usage(
                    &state,
                    &auth,
                    &mcp_server,
                    &mcp_tool,
                    response.status().as_u16(),
                    Duration::ZERO,
                );
                return response;
            }
        }
    } else {
        None
    };

    let snapshot = state.snapshot.load();
    // Scope the gateway to the tools this caller's key permits, so MCP tool
    // access is governed by the same key object as LLM access.
    let acl = aisix_mcp::ToolAcl::from_allowed(auth.key().allowed_tools.as_deref());
    let gateway = aisix_mcp::McpGateway::from_snapshot(&snapshot).with_tool_acl(acl);
    let service = aisix_mcp::streamable_http_service(gateway);
    let request = Request::from_parts(parts, Body::from(bytes));
    // `StreamableHttpService` is a tower service that dispatches on method and
    // never fails (`Error = Infallible`); map its boxed body back to axum's.
    let started = Instant::now();
    let response = match service.oneshot(request).await {
        Ok(response) => response.map(Body::new),
        Err(infallible) => match infallible {},
    };

    if is_tool_call {
        emit_tool_call_usage(
            &state,
            &auth,
            &mcp_server,
            &mcp_tool,
            response.status().as_u16(),
            started.elapsed(),
        );
    }
    response
}

/// Emit a usage event for a single MCP tool call into the same sink as LLM
/// usage. MCP calls carry no token cost yet, so token/cost fields stay zero;
/// the event records who called which tool, the outcome, and the latency.
fn emit_tool_call_usage(
    state: &ProxyState,
    auth: &AuthenticatedKey,
    mcp_server: &str,
    mcp_tool: &str,
    status_code: u16,
    latency: Duration,
) {
    let event = UsageEvent {
        request_id: uuid::Uuid::new_v4().to_string(),
        occurred_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        api_key_id: auth.entry.id.clone(),
        status_code,
        latency_ms: latency.as_millis().min(u32::MAX as u128) as u32,
        inbound_protocol: "mcp".to_string(),
        mcp_server_name: mcp_server.to_string(),
        mcp_tool_name: mcp_tool.to_string(),
        ..Default::default()
    };
    state.usage_sink.try_emit("mcp", event);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_router;
    use aisix_core::{AisixSnapshot, ApiKey, ProxyConfig, ResourceEntry, SnapshotHandle};
    use axum::body::Body;
    use axum::http::{Request as HttpRequest, StatusCode};
    use std::sync::Arc;

    fn cfg() -> ProxyConfig {
        ProxyConfig {
            addr: "127.0.0.1:0".into(),
            request_body_limit_bytes: 1_048_576,
            real_ip: Default::default(),
            tls: None,
        }
    }

    const TOKEN: &str = "sk-mcp-endpoint-test";

    /// A snapshot carrying one valid API key (and no MCP servers — the MCP
    /// `initialize` handshake is answered by the gateway itself, no upstream
    /// needed).
    fn snapshot_with_key() -> AisixSnapshot {
        let key_hash = ApiKey::hash_bearer(TOKEN);
        let apikey: ApiKey = serde_json::from_value(serde_json::json!({
            "key_hash": key_hash,
            "allowed_models": ["*"],
        }))
        .expect("valid apikey");
        let snapshot = AisixSnapshot::new();
        snapshot
            .apikeys
            .insert(ResourceEntry::new("ak-1", apikey, 1));
        snapshot
    }

    fn router_with(snapshot: AisixSnapshot) -> axum::Router {
        let handle = SnapshotHandle::new(snapshot);
        let hub = Arc::new(aisix_gateway::Hub::new());
        build_router(ProxyState::new(handle, hub, &cfg()).without_cache())
    }

    /// A minimal MCP `initialize` request body + the headers the Streamable
    /// HTTP transport requires (Accept must list both content types).
    fn initialize_request(auth: Option<&str>) -> HttpRequest<Body> {
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": { "name": "endpoint-test", "version": "0.1" }
            }
        });
        // A non-loopback Host on purpose: proves the gateway accepts the
        // deployment's real DNS name (rmcp's default Host allowlist is disabled
        // for this key-authenticated endpoint).
        let mut builder = HttpRequest::post("/mcp")
            .header("host", "mcp.aisix.example.com")
            .header("content-type", "application/json")
            .header("accept", "application/json, text/event-stream");
        if let Some(token) = auth {
            builder = builder.header("authorization", format!("Bearer {token}"));
        }
        builder.body(Body::from(body.to_string())).unwrap()
    }

    /// A snapshot whose key carries an inline `rate_limit` of `rpm` requests
    /// per minute and may call every tool.
    fn snapshot_with_rate_limited_key(rpm: u32) -> AisixSnapshot {
        let key_hash = ApiKey::hash_bearer(TOKEN);
        let apikey: ApiKey = serde_json::from_value(serde_json::json!({
            "key_hash": key_hash,
            "allowed_models": ["*"],
            "allowed_tools": ["*"],
            "rate_limit": { "rpm": rpm },
        }))
        .expect("valid apikey");
        let snapshot = AisixSnapshot::new();
        snapshot
            .apikeys
            .insert(ResourceEntry::new("ak-1", apikey, 1));
        snapshot
    }

    /// A JSON-RPC request to `/mcp` for `method`, authenticated with `TOKEN`.
    fn mcp_request(method: &str, params: serde_json::Value) -> HttpRequest<Body> {
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        });
        HttpRequest::post("/mcp")
            .header("host", "mcp.aisix.example.com")
            .header("content-type", "application/json")
            .header("accept", "application/json, text/event-stream")
            .header("authorization", format!("Bearer {TOKEN}"))
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    fn tools_call_request() -> HttpRequest<Body> {
        mcp_request(
            "tools/call",
            serde_json::json!({ "name": "ghost__tool", "arguments": {} }),
        )
    }

    #[tokio::test]
    async fn rate_limit_applies_to_tool_calls_but_not_handshake() {
        // rpm=1: the key may make one tools/call per minute.
        let router = router_with(snapshot_with_rate_limited_key(1));

        // First tool call passes the rate gate (status is whatever the gateway
        // returns — there are no upstreams — but NOT 429).
        let first = router
            .clone()
            .oneshot(tools_call_request())
            .await
            .expect("router responds");
        assert_ne!(
            first.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "first tool call should pass the rate gate"
        );

        // Second tool call within the same minute is rate-limited.
        let second = router
            .clone()
            .oneshot(tools_call_request())
            .await
            .expect("router responds");
        assert_eq!(
            second.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "second tool call should be rate-limited"
        );

        // Neither handshake nor discovery is rate-limited, even with the key at
        // its tool-call limit — a client can always connect and enumerate.
        let handshake = router
            .clone()
            .oneshot(initialize_request(Some(TOKEN)))
            .await
            .expect("router responds");
        assert_ne!(
            handshake.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "initialize must not be rate-limited"
        );

        let listed = router
            .oneshot(mcp_request("tools/list", serde_json::json!({})))
            .await
            .expect("router responds");
        assert_ne!(
            listed.status(),
            StatusCode::TOO_MANY_REQUESTS,
            "tools/list must not be rate-limited"
        );
    }

    #[tokio::test]
    async fn rejects_request_without_api_key() {
        let router = router_with(snapshot_with_key());
        let resp = router
            .oneshot(initialize_request(None))
            .await
            .expect("router responds");
        assert_eq!(
            resp.status(),
            StatusCode::UNAUTHORIZED,
            "missing API key must be rejected at the /mcp edge"
        );
    }

    #[tokio::test]
    async fn rejects_request_with_invalid_api_key() {
        let router = router_with(snapshot_with_key());
        let resp = router
            .oneshot(initialize_request(Some("sk-wrong")))
            .await
            .expect("router responds");
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn auth_gates_non_post_methods() {
        // The route is `any(...)`, so every method must be auth-gated — a GET
        // with no key must 401 (not fall through to rmcp's 405).
        let router = router_with(snapshot_with_key());
        let req = HttpRequest::get("/mcp")
            .header("host", "mcp.aisix.example.com")
            .body(Body::empty())
            .unwrap();
        let resp = router.oneshot(req).await.expect("router responds");
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn trailing_slash_route_is_auth_gated() {
        let router = router_with(snapshot_with_key());
        let req = HttpRequest::post("/mcp/")
            .header("host", "mcp.aisix.example.com")
            .header("content-type", "application/json")
            .header("accept", "application/json, text/event-stream")
            .body(Body::from("{}"))
            .unwrap();
        let resp = router.oneshot(req).await.expect("router responds");
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn oversized_unauthenticated_body_is_limited_before_handler() {
        // A declared Content-Length over the cap is rejected (413) by the
        // body-limit layer, which wraps the route — before auth or the handler,
        // so an oversized unauthenticated body can't pin resources.
        let router = router_with(snapshot_with_key());
        let big = "a".repeat(1_048_577); // cfg() cap is 1 MiB
        let req = HttpRequest::post("/mcp")
            .header("host", "mcp.aisix.example.com")
            .header("content-type", "application/json")
            .header("content-length", big.len().to_string())
            .body(Body::from(big))
            .unwrap();
        let resp = router.oneshot(req).await.expect("router responds");
        let status = resp.status();
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "got {status}");
    }

    #[tokio::test]
    async fn authenticated_request_reaches_the_mcp_gateway() {
        let router = router_with(snapshot_with_key());
        let resp = router
            .oneshot(initialize_request(Some(TOKEN)))
            .await
            .expect("router responds");
        // Auth passed and the request was served by the MCP gateway (not a 401).
        let status = resp.status();
        let body = axum::body::to_bytes(resp.into_body(), 64 * 1024)
            .await
            .expect("read body");
        let text = String::from_utf8_lossy(&body);
        assert_eq!(
            status,
            StatusCode::OK,
            "a valid key should reach the gateway and complete the MCP initialize handshake; body: {text}"
        );
        assert!(
            text.contains("serverInfo") || text.contains("protocolVersion"),
            "initialize result should carry the server info, got: {text}"
        );
    }

    #[tokio::test]
    async fn emits_usage_event_for_tool_call_only() {
        use aisix_obs::{UsageEvent, UsageSink};

        let (tx, mut rx) = tokio::sync::mpsc::channel::<UsageEvent>(8);
        let handle = SnapshotHandle::new(snapshot_with_key());
        let hub = Arc::new(aisix_gateway::Hub::new());
        let state = ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let router = build_router(state);

        // A tools/call emits one usage event into the same sink as LLM calls,
        // carrying the MCP attribution (server + tool, parsed from the
        // namespaced name `ghost__tool`).
        let _ = router
            .clone()
            .oneshot(tools_call_request())
            .await
            .expect("router responds");
        let event = rx
            .try_recv()
            .expect("a usage event was emitted for the tool call");
        assert_eq!(event.inbound_protocol, "mcp");
        assert_eq!(event.mcp_server_name, "ghost");
        assert_eq!(event.mcp_tool_name, "tool");
        assert_eq!(event.api_key_id, "ak-1");
        assert_eq!(event.prompt_tokens, 0, "MCP calls carry no token cost");
        assert!(
            rx.try_recv().is_err(),
            "exactly one usage event per tool call"
        );

        // The handshake does NOT emit a usage event.
        let _ = router
            .oneshot(initialize_request(Some(TOKEN)))
            .await
            .expect("router responds");
        assert!(
            rx.try_recv().is_err(),
            "initialize must not emit a usage event"
        );
    }

    #[tokio::test]
    async fn rate_limited_tool_call_still_emits_usage_event() {
        use aisix_obs::{UsageEvent, UsageSink};

        // rpm=1: the second tool call is rate-limited (429) but still recorded —
        // the reject path emits before returning.
        let (tx, mut rx) = tokio::sync::mpsc::channel::<UsageEvent>(8);
        let handle = SnapshotHandle::new(snapshot_with_rate_limited_key(1));
        let hub = Arc::new(aisix_gateway::Hub::new());
        let state = ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let router = build_router(state);

        let _ = router
            .clone()
            .oneshot(tools_call_request())
            .await
            .expect("router responds");
        let _ = rx.try_recv().expect("first (allowed) call emits");

        let second = router
            .oneshot(tools_call_request())
            .await
            .expect("router responds");
        assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
        let event = rx
            .try_recv()
            .expect("the rate-limited call is still recorded");
        assert_eq!(event.status_code, 429);
        assert_eq!(event.inbound_protocol, "mcp");
        assert_eq!(event.mcp_server_name, "ghost");
        assert_eq!(event.mcp_tool_name, "tool");
    }
}
