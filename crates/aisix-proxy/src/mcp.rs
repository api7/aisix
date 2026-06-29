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

use axum::body::Body;
use axum::extract::{Request, State};
use axum::response::Response;
use tower::ServiceExt;

use crate::auth::AuthenticatedKey;
use crate::state::ProxyState;

/// Serve a `/mcp` request. The [`AuthenticatedKey`] extractor enforces a valid
/// AISIX API key (responding `401` otherwise); the request is then handled by an
/// MCP gateway built from the current snapshot's `mcp_servers`.
pub async fn mcp_endpoint(
    _auth: AuthenticatedKey,
    State(state): State<ProxyState>,
    request: Request,
) -> Response {
    let snapshot = state.snapshot.load();
    let gateway = aisix_mcp::McpGateway::from_snapshot(&snapshot);
    let service = aisix_mcp::streamable_http_service(gateway);
    // `StreamableHttpService` is a tower service that dispatches on method and
    // never fails (`Error = Infallible`); map its boxed body back to axum's.
    match service.oneshot(request).await {
        Ok(response) => response.map(Body::new),
        Err(infallible) => match infallible {},
    }
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
        let mut builder = HttpRequest::post("/mcp")
            .header("host", "localhost")
            .header("content-type", "application/json")
            .header("accept", "application/json, text/event-stream");
        if let Some(token) = auth {
            builder = builder.header("authorization", format!("Bearer {token}"));
        }
        builder.body(Body::from(body.to_string())).unwrap()
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
}
