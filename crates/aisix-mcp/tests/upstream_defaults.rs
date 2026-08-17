//! Regression tests for the two rmcp 3.x client-side defaults that would
//! silently change the gateway's cost and isolation posture if left at their
//! SDK values (AISIX-Cloud#1144):
//!
//! 1. **MRTR auto-retry.** `RunningService::call_tool` drives SEP-2322
//!    multi-round-trip requests automatically — up to 10 upstream round
//!    trips for ONE inbound call. The bridge must send exactly one request
//!    per inbound `tools/call` and surface a non-final (`input_required`)
//!    answer as a clean error, never a hidden retry loop.
//! 2. **Client response cache.** rmcp enables a per-peer response cache by
//!    default, honoring the server's `ttlMs` hint. The bridge is shared
//!    across every AISIX caller reaching the same upstream, so a cached
//!    response would cross tenant boundaries. The bridge must disable it:
//!    two identical requests on ONE session must both reach the upstream,
//!    even when the upstream advertises a large `ttlMs`.
//!
//! The upstream here is a raw axum JSON-RPC stub, not an rmcp server: the
//! tests need full control of the wire (an `input_required` result, a large
//! `ttlMs`) and an exact request counter.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use aisix_mcp::{McpBridge, McpUpstream, RmcpBridge};
use axum::extract::State;
use axum::response::IntoResponse;

/// Counts upstream `tools/list` and `tools/call` requests, so the tests can
/// assert exactly how many times the bridge actually hit the wire.
#[derive(Default)]
struct StubCounters {
    list: AtomicUsize,
    call: AtomicUsize,
}

/// Behavior switch for the stub's `tools/call` answer.
#[derive(Clone, Copy)]
enum CallBehavior {
    /// Answer with a SEP-2322 `input_required` result (non-final).
    InputRequired,
}

#[derive(Clone)]
struct Stub {
    counters: Arc<StubCounters>,
    call_behavior: Option<CallBehavior>,
}

/// Minimal legacy Streamable HTTP endpoint: answers `initialize`, accepts
/// `notifications/initialized`, serves `tools/list` / `tools/call`.
async fn stub_mcp(State(stub): State<Stub>, body: axum::body::Bytes) -> axum::response::Response {
    let message: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return (axum::http::StatusCode::BAD_REQUEST, "not json").into_response(),
    };
    let method = message["method"].as_str().unwrap_or_default();
    let id = message["id"].clone();
    let respond = |result: serde_json::Value| {
        (
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            serde_json::json!({ "jsonrpc": "2.0", "id": id, "result": result }).to_string(),
        )
            .into_response()
    };
    match method {
        "initialize" => respond(serde_json::json!({
            "protocolVersion": "2025-11-25",
            "capabilities": { "tools": {} },
            "serverInfo": { "name": "stub", "version": "0.0.0" },
        })),
        "notifications/initialized" => axum::http::StatusCode::ACCEPTED.into_response(),
        "tools/list" => {
            // Distinct payload per hit + a huge freshness hint: if any layer
            // between the bridge and this stub cached the first answer, the
            // second request would never arrive (the counter stays at 1) and
            // the second result would repeat hit-1's description.
            let hit = stub.counters.list.fetch_add(1, Ordering::SeqCst) + 1;
            respond(serde_json::json!({
                "tools": [{
                    "name": "echo",
                    "description": format!("hit-{hit}"),
                    "inputSchema": { "type": "object" },
                }],
                "ttlMs": 3_600_000u64,
                "cacheScope": "public",
            }))
        }
        "tools/call" => {
            stub.counters.call.fetch_add(1, Ordering::SeqCst);
            match stub.call_behavior {
                // `resultType: "input_required"` alone is a valid
                // `InputRequiredResult` (both detail fields are optional) —
                // and it is the exact discriminator rmcp's untagged result
                // parsing keys on, so the shape cannot mis-parse as a
                // completed tool result.
                Some(CallBehavior::InputRequired) => respond(serde_json::json!({
                    "resultType": "input_required",
                    "requestState": "opaque-state",
                })),
                None => respond(serde_json::json!({
                    "content": [{ "type": "text", "text": "ok" }],
                    "isError": false,
                })),
            }
        }
        other => (
            axum::http::StatusCode::BAD_REQUEST,
            format!("unexpected method {other}"),
        )
            .into_response(),
    }
}

async fn spawn_stub(call_behavior: Option<CallBehavior>) -> (SocketAddr, Arc<StubCounters>) {
    let counters = Arc::new(StubCounters::default());
    let stub = Stub {
        counters: Arc::clone(&counters),
        call_behavior,
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
    (addr, counters)
}

/// One inbound `tools/call` = exactly ONE upstream request, even when the
/// upstream answers `input_required`. With rmcp's default `call_tool` the
/// SDK would re-send the request itself (up to 10 rounds) — every extra
/// round an unmetered upstream call outside AISIX's quota and budget
/// accounting.
#[tokio::test]
async fn input_required_is_one_round_trip_and_a_clean_error() {
    let (addr, counters) = spawn_stub(Some(CallBehavior::InputRequired)).await;
    let bridge = RmcpBridge::connect(&McpUpstream::new(format!("http://{addr}/mcp")))
        .await
        .expect("connect");

    let error = bridge
        .call_tool("echo", serde_json::json!({ "text": "x" }))
        .await
        .expect_err("a non-final result must surface as an error");
    assert!(
        error.to_string().contains("interactive input"),
        "error names the MRTR condition: {error}"
    );
    assert_eq!(
        counters.call.load(Ordering::SeqCst),
        1,
        "the bridge must not silently retry an input_required response"
    );
}

/// Two identical `tools/list` calls on ONE upstream session both reach the
/// upstream, despite a one-hour `ttlMs` hint: the SDK's default-enabled
/// response cache is disabled in the bridge. A cached second answer would
/// mean one AISIX caller could be served another caller's upstream view.
#[tokio::test]
async fn upstream_cache_hints_do_not_short_circuit_the_bridge() {
    let (addr, counters) = spawn_stub(None).await;
    let bridge = RmcpBridge::connect(&McpUpstream::new(format!("http://{addr}/mcp")))
        .await
        .expect("connect");

    let first = bridge.list_tools().await.expect("first list");
    assert_eq!(first[0].description.as_deref(), Some("hit-1"));

    let second = bridge.list_tools().await.expect("second list");
    assert_eq!(
        second[0].description.as_deref(),
        Some("hit-2"),
        "the second list must be the upstream's SECOND answer, not a replay"
    );
    assert_eq!(
        counters.list.load(Ordering::SeqCst),
        2,
        "both list calls must reach the upstream"
    );
}
