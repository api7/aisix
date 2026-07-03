//! Roundtrip against a real (locally spawned) upstream A2A agent.
//!
//! Proves the governed tunnel end to end over real HTTP — no mocked network:
//! the bridge discovers the agent card at the RFC 8615 well-known URI, forwards
//! a JSON-RPC `message/send`, and the gateway-held upstream credential reaches
//! the upstream (and only the upstream) while an unauthenticated bridge sends
//! no credential at all.

use std::net::SocketAddr;

use aisix_a2a::{A2aAuth, A2aBridge, A2aUpstream, HttpBridge, DEFAULT_UPSTREAM_TIMEOUT};
use axum::http::HeaderMap;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

/// A minimal upstream A2A agent: serves its card at the well-known URI and
/// answers JSON-RPC by echoing back the request id and the credentials it saw,
/// so the test can assert what the gateway forwarded.
async fn spawn_agent() -> SocketAddr {
    async fn card() -> Json<Value> {
        Json(json!({
            "name": "Test Agent",
            "url": "https://upstream.example.com/a2a",
            "version": "1.0.0",
            "skills": [{"id": "echo", "name": "Echo"}]
        }))
    }

    async fn rpc(headers: HeaderMap, Json(body): Json<Value>) -> Json<Value> {
        let auth = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        let api_key = headers
            .get("x-api-key")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        Json(json!({
            "jsonrpc": "2.0",
            "id": body["id"].clone(),
            "result": {
                "kind": "task",
                "id": "task-1",
                "status": {"state": "completed"},
                "echoed_auth": auth,
                "echoed_api_key": api_key,
            }
        }))
    }

    let app = Router::new()
        .route("/.well-known/agent-card.json", get(card))
        .route("/a2a", post(rpc));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app.into_make_service())
            .await
            .unwrap();
    });
    addr
}

fn message_send(id: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": "hello"}],
                "messageId": "m1"
            }
        }
    })
}

#[tokio::test]
async fn fetches_card_and_forwards_bearer() {
    let addr = spawn_agent().await;
    let bridge = HttpBridge::new(A2aUpstream {
        url: format!("http://{addr}/a2a"),
        auth: A2aAuth::Bearer("tok-123".into()),
        timeout: DEFAULT_UPSTREAM_TIMEOUT,
    });

    let card = bridge.fetch_agent_card().await.unwrap();
    assert_eq!(card.name, "Test Agent");
    // Unknown fields survive the round-trip (needed for later URL rewriting).
    assert_eq!(card.rest["version"], "1.0.0");

    let resp = bridge.send(&message_send("req-1")).await.unwrap();
    assert_eq!(resp["id"], "req-1", "JSON-RPC id must round-trip");
    assert_eq!(resp["result"]["id"], "task-1");
    // The gateway-held bearer reached the upstream.
    assert_eq!(resp["result"]["echoed_auth"], "Bearer tok-123");
}

#[tokio::test]
async fn forwards_api_key_header() {
    let addr = spawn_agent().await;
    let bridge = HttpBridge::new(A2aUpstream {
        url: format!("http://{addr}/a2a"),
        auth: A2aAuth::ApiKey("k-secret".into()),
        timeout: DEFAULT_UPSTREAM_TIMEOUT,
    });

    let resp = bridge.send(&message_send("req-2")).await.unwrap();
    assert_eq!(resp["result"]["echoed_api_key"], "k-secret");
    // api_key auth must not also mint an Authorization header.
    assert!(resp["result"]["echoed_auth"].is_null());
}

#[tokio::test]
async fn sends_no_credential_when_none() {
    let addr = spawn_agent().await;
    let bridge = HttpBridge::new(A2aUpstream {
        url: format!("http://{addr}/a2a"),
        auth: A2aAuth::None,
        timeout: DEFAULT_UPSTREAM_TIMEOUT,
    });

    let resp = bridge.send(&message_send("req-3")).await.unwrap();
    assert!(resp["result"]["echoed_auth"].is_null());
    assert!(resp["result"]["echoed_api_key"].is_null());
}
