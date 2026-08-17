//! The per-server `protocol_version` bridge contract (AISIX-Cloud#1151):
//!
//! - `protocol_version: "2026-07-28"` opens the upstream session with the
//!   handshake-free `server/discover` lifecycle — the only way to reach a
//!   server that no longer answers `initialize`.
//! - The selection is EXPLICIT in both directions: a modern-only server
//!   under the default (legacy handshake) config fails visibly, and a
//!   legacy server under the `2026-07-28` config fails visibly. No probe,
//!   no silent cross-generation fallback — automatic bridging is how
//!   version/session context ends up crossing the protocol boundary (the
//!   open bug class in auto-bridging gateways).
//! - Downstream context never crosses to the upstream: the bridge opens
//!   its own session, so the caller's `Authorization`, session id, and
//!   protocol version header are absent from upstream requests by
//!   construction — pinned here against a header-capturing upstream.
//! - A stateful legacy upstream receives the COMPLETE handshake sequence
//!   (`initialize` → `notifications/initialized` → operation) on every
//!   bridge session.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use aisix_mcp::{
    streamable_http_service, EphemeralBridge, McpBridge, McpGateway, McpProtocol, McpUpstream,
    RmcpBridge,
};
use axum::extract::State;
use axum::response::IntoResponse;

/// Which generation the stub upstream speaks.
#[derive(Clone, Copy, PartialEq)]
enum StubGeneration {
    /// Answers `initialize` (optionally minting a session id); rejects
    /// `server/discover` with JSON-RPC method-not-found.
    Legacy { stateful: bool },
    /// Answers `server/discover` and stateless operations; rejects
    /// `initialize` with JSON-RPC method-not-found.
    ModernOnly,
}

#[derive(Default)]
struct StubRecorder {
    initialize: AtomicUsize,
    discover: AtomicUsize,
    /// JSON-RPC method sequence, in arrival order.
    sequence: Mutex<Vec<String>>,
    /// Headers seen on the most recent `tools/call`.
    call_headers: Mutex<Vec<(String, String)>>,
}

#[derive(Clone)]
struct Stub {
    generation: StubGeneration,
    recorder: Arc<StubRecorder>,
}

async fn stub_mcp(
    State(stub): State<Stub>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> axum::response::Response {
    let message: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return (axum::http::StatusCode::BAD_REQUEST, "not json").into_response(),
    };
    let method = message["method"].as_str().unwrap_or_default().to_string();
    let id = message["id"].clone();
    stub.recorder
        .sequence
        .lock()
        .expect("sequence lock")
        .push(method.clone());
    let respond = |result: serde_json::Value| {
        (
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            serde_json::json!({ "jsonrpc": "2.0", "id": id.clone(), "result": result }).to_string(),
        )
            .into_response()
    };
    let method_not_found = |what: &str| {
        (
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            serde_json::json!({
                "jsonrpc": "2.0",
                "id": id.clone(),
                "error": { "code": -32601, "message": format!("method not found: {what}") }
            })
            .to_string(),
        )
            .into_response()
    };
    match method.as_str() {
        "initialize" => {
            stub.recorder.initialize.fetch_add(1, Ordering::SeqCst);
            match stub.generation {
                StubGeneration::ModernOnly => method_not_found("initialize"),
                StubGeneration::Legacy { stateful } => {
                    let result = serde_json::json!({
                        "protocolVersion": "2025-11-25",
                        "capabilities": { "tools": {} },
                        "serverInfo": { "name": "stub", "version": "0.0.0" },
                    });
                    if stateful {
                        (
                            [
                                (axum::http::header::CONTENT_TYPE, "application/json"),
                                (
                                    axum::http::HeaderName::from_static("mcp-session-id"),
                                    "sess-legacy-1",
                                ),
                            ],
                            serde_json::json!({ "jsonrpc": "2.0", "id": id, "result": result })
                                .to_string(),
                        )
                            .into_response()
                    } else {
                        respond(result)
                    }
                }
            }
        }
        "server/discover" => {
            stub.recorder.discover.fetch_add(1, Ordering::SeqCst);
            match stub.generation {
                StubGeneration::Legacy { .. } => method_not_found("server/discover"),
                StubGeneration::ModernOnly => respond(serde_json::json!({
                    "resultType": "complete",
                    "supportedVersions": ["2026-07-28"],
                    "capabilities": { "tools": {} },
                    "ttlMs": 0,
                    "cacheScope": "private",
                })),
            }
        }
        "notifications/initialized" => axum::http::StatusCode::ACCEPTED.into_response(),
        "tools/list" => respond(serde_json::json!({
            "tools": [{
                "name": "echo",
                "description": "echo",
                "inputSchema": { "type": "object" },
            }],
        })),
        "tools/call" => {
            let mut seen = stub
                .recorder
                .call_headers
                .lock()
                .expect("call_headers lock");
            *seen = headers
                .iter()
                .map(|(name, value)| {
                    (
                        name.as_str().to_ascii_lowercase(),
                        String::from_utf8_lossy(value.as_bytes()).into_owned(),
                    )
                })
                .collect();
            respond(serde_json::json!({
                "content": [{ "type": "text", "text": "ok" }],
                "isError": false,
            }))
        }
        other => (
            axum::http::StatusCode::BAD_REQUEST,
            format!("unexpected method {other}"),
        )
            .into_response(),
    }
}

async fn spawn_stub(generation: StubGeneration) -> (SocketAddr, Arc<StubRecorder>) {
    let recorder = Arc::new(StubRecorder::default());
    let stub = Stub {
        generation,
        recorder: Arc::clone(&recorder),
    };
    let app = axum::Router::new()
        .route("/mcp", axum::routing::post(stub_mcp))
        .with_state(stub);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve");
    });
    (addr, recorder)
}

/// `protocol_version: "2026-07-28"` reaches a modern-only upstream:
/// handshake-free startup, list + call work, and `initialize` is never
/// attempted.
#[tokio::test]
async fn modern_only_upstream_works_with_the_dated_protocol_version() {
    let (addr, recorder) = spawn_stub(StubGeneration::ModernOnly).await;
    let upstream =
        McpUpstream::new(format!("http://{addr}/mcp")).with_protocol(McpProtocol::V20260728);
    let bridge = RmcpBridge::connect(&upstream)
        .await
        .expect("discover-lifecycle connect");

    let tools = bridge.list_tools().await.expect("list tools");
    assert_eq!(tools[0].name, "echo");
    let result = bridge
        .call_tool("echo", serde_json::json!({}))
        .await
        .expect("call tool");
    assert!(!result.is_error);

    assert_eq!(
        recorder.initialize.load(Ordering::SeqCst),
        0,
        "a 2026-07-28 session must never attempt the legacy handshake"
    );
    assert!(recorder.discover.load(Ordering::SeqCst) >= 1);
}

/// A modern-only upstream under the DEFAULT (legacy handshake) config
/// fails visibly at connect — and the bridge does not secretly probe
/// `server/discover` to save it.
#[tokio::test]
async fn modern_only_upstream_fails_visibly_under_default_config() {
    let (addr, recorder) = spawn_stub(StubGeneration::ModernOnly).await;
    let result = RmcpBridge::connect(&McpUpstream::new(format!("http://{addr}/mcp"))).await;
    assert!(
        result.is_err(),
        "a modern-only server must fail the legacy handshake visibly"
    );
    assert_eq!(
        recorder.discover.load(Ordering::SeqCst),
        0,
        "the legacy lifecycle must not silently probe server/discover"
    );
}

/// A legacy upstream under the `2026-07-28` config fails visibly — no
/// silent fallback to the `initialize` handshake.
#[tokio::test]
async fn legacy_upstream_fails_visibly_under_modern_config_without_fallback() {
    let (addr, recorder) = spawn_stub(StubGeneration::Legacy { stateful: false }).await;
    let upstream =
        McpUpstream::new(format!("http://{addr}/mcp")).with_protocol(McpProtocol::V20260728);
    let result = RmcpBridge::connect(&upstream).await;
    assert!(
        result.is_err(),
        "a legacy server must fail the discover lifecycle visibly"
    );
    assert_eq!(
        recorder.initialize.load(Ordering::SeqCst),
        0,
        "the discover lifecycle must not silently fall back to initialize"
    );
}

/// Drive one modern downstream `tools/call` through the full gateway chain
/// and return the headers the upstream saw.
async fn upstream_headers_after_gateway_call(
    stub_addr: SocketAddr,
    recorder: &StubRecorder,
) -> Vec<(String, String)> {
    let bridge = EphemeralBridge::new(McpUpstream::new(format!("http://{stub_addr}/mcp")));
    let gateway = McpGateway::new([("alpha".to_string(), Arc::new(bridge) as Arc<dyn McpBridge>)]);
    let app = axum::Router::new().nest_service("/mcp", streamable_http_service(gateway, 0));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind gateway port");
    let gw_addr = listener.local_addr().expect("gateway addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve gateway");
    });

    // A modern (2026-07-28) downstream caller carrying context that must
    // NOT cross to the upstream: an API credential, a stale session id,
    // and its own protocol version header.
    let response = reqwest::Client::new()
        .post(format!("http://{gw_addr}/mcp"))
        .header("content-type", "application/json")
        .header("accept", "application/json, text/event-stream")
        .header("authorization", "Bearer sk-DOWNSTREAM-SECRET")
        .header("mcp-session-id", "downstream-session-1")
        .header("mcp-protocol-version", "2026-07-28")
        .header("mcp-method", "tools/call")
        .header("mcp-name", "alpha__echo")
        .body(
            serde_json::json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "alpha__echo",
                    "arguments": {},
                    "_meta": {
                        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
                        "io.modelcontextprotocol/clientInfo": {
                            "name": "ctx-test", "version": "0.0.0"
                        },
                        "io.modelcontextprotocol/clientCapabilities": {},
                    }
                }
            })
            .to_string(),
        )
        .send()
        .await
        .expect("send");
    assert_eq!(response.status(), 200, "{}", response.text().await.unwrap());

    recorder
        .call_headers
        .lock()
        .expect("call_headers lock")
        .clone()
}

/// Downstream `Authorization`, session id, and protocol version never
/// cross the protocol boundary: the bridge opens its own upstream session
/// with its own negotiated context.
#[tokio::test]
async fn downstream_context_never_crosses_to_the_upstream() {
    let (stub_addr, recorder) = spawn_stub(StubGeneration::Legacy { stateful: false }).await;
    let headers = upstream_headers_after_gateway_call(stub_addr, &recorder).await;

    let get = |name: &str| {
        headers
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_str())
    };
    assert_eq!(
        get("authorization"),
        None,
        "the caller's credential must never be forwarded (upstream auth is \
         gateway-held): {headers:?}"
    );
    assert_eq!(
        get("mcp-session-id"),
        None,
        "a downstream session id must never leak into the upstream session: {headers:?}"
    );
    let upstream_version = get("mcp-protocol-version");
    assert_ne!(
        upstream_version,
        Some("2026-07-28"),
        "the downstream protocol version must not be mirrored upstream — the \
         bridge negotiated its own (legacy) session: {headers:?}"
    );
}

/// A stateful legacy upstream sees the COMPLETE handshake sequence for the
/// bridge's session — never a bare `tools/call` riding on a session the
/// upstream doesn't have.
#[tokio::test]
async fn stateful_legacy_upstream_receives_the_full_handshake_sequence() {
    let (stub_addr, recorder) = spawn_stub(StubGeneration::Legacy { stateful: true }).await;
    let _ = upstream_headers_after_gateway_call(stub_addr, &recorder).await;

    let sequence = recorder.sequence.lock().expect("sequence lock").clone();
    assert_eq!(
        sequence,
        vec![
            "initialize".to_string(),
            "notifications/initialized".to_string(),
            "tools/call".to_string(),
        ],
        "every bridge session must run the full legacy lifecycle"
    );
}
