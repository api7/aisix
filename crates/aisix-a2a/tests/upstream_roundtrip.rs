//! Roundtrip against a real (locally spawned) upstream A2A agent.
//!
//! Proves the governed tunnel end to end over real HTTP — no mocked network:
//! the bridge discovers the agent card at the RFC 8615 well-known URI, forwards
//! a JSON-RPC `message/send`, and the gateway-held upstream credential reaches
//! the upstream (and only the upstream) while an unauthenticated bridge sends
//! no credential at all.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use aisix_a2a::{A2aAuth, A2aBridge, A2aError, A2aUpstream, HttpBridge, DEFAULT_UPSTREAM_TIMEOUT};
use aisix_core::A2aProtocolVersion;
use axum::http::header::LOCATION;
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{any, get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

/// An upstream pinned to A2A 1.0 — the default for a registered agent.
fn upstream(url: String, auth: A2aAuth) -> A2aUpstream {
    A2aUpstream {
        url,
        auth,
        protocol_version: A2aProtocolVersion::V1_0,
        timeout: DEFAULT_UPSTREAM_TIMEOUT,
    }
}

/// Read back the `A2A-Version` an inbound request carried, or `null`.
fn seen_version(headers: &HeaderMap) -> Value {
    headers
        .get("a2a-version")
        .and_then(|v| v.to_str().ok())
        .map(|v| Value::String(v.to_string()))
        .unwrap_or(Value::Null)
}

/// A minimal upstream A2A agent: serves its card at the well-known URI and
/// answers JSON-RPC by echoing back the request id and the credentials it saw,
/// so the test can assert what the gateway forwarded.
async fn spawn_agent() -> SocketAddr {
    async fn card(headers: HeaderMap) -> Json<Value> {
        Json(json!({
            "name": "Test Agent",
            "url": "https://upstream.example.com/a2a",
            "version": "1.0.0",
            "skills": [{"id": "echo", "name": "Echo"}],
            "echoed_version": seen_version(&headers),
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
                "echoed_version": seen_version(&headers),
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
    let bridge = HttpBridge::new(upstream(
        format!("http://{addr}/a2a"),
        A2aAuth::Bearer("tok-123".into()),
    ));

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
    let bridge = HttpBridge::new(upstream(
        format!("http://{addr}/a2a"),
        A2aAuth::ApiKey("k-secret".into()),
    ));

    let resp = bridge.send(&message_send("req-2")).await.unwrap();
    assert_eq!(resp["result"]["echoed_api_key"], "k-secret");
    // api_key auth must not also mint an Authorization header.
    assert!(resp["result"]["echoed_auth"].is_null());
}

#[tokio::test]
async fn sends_no_credential_when_none() {
    let addr = spawn_agent().await;
    let bridge = HttpBridge::new(upstream(format!("http://{addr}/a2a"), A2aAuth::None));

    let resp = bridge.send(&message_send("req-3")).await.unwrap();
    assert!(resp["result"]["echoed_auth"].is_null());
    assert!(resp["result"]["echoed_api_key"].is_null());
}

/// An upstream that answers `/redirect` with `302 -> /secret` and counts hits
/// on `/secret`. Lets a test prove the gateway does NOT chase the redirect —
/// otherwise a compromised agent could pivot the VPC-internal DP into an SSRF.
async fn spawn_redirect_probe() -> (SocketAddr, Arc<AtomicUsize>) {
    let hits = Arc::new(AtomicUsize::new(0));
    let secret_hits = hits.clone();
    let app = Router::new()
        .route(
            "/redirect",
            post(|| async { (StatusCode::FOUND, [(LOCATION, "/secret")]).into_response() }),
        )
        .route(
            "/secret",
            get(move || {
                let h = secret_hits.clone();
                async move {
                    h.fetch_add(1, Ordering::SeqCst);
                    Json(json!({"jsonrpc": "2.0", "result": {"leaked": true}}))
                }
            }),
        );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app.into_make_service())
            .await
            .unwrap();
    });
    (addr, hits)
}

#[tokio::test]
async fn refuses_to_follow_upstream_redirect() {
    let (addr, secret_hits) = spawn_redirect_probe().await;
    let bridge = HttpBridge::new(upstream(format!("http://{addr}/redirect"), A2aAuth::None));

    let err = bridge.send(&message_send("r")).await.unwrap_err();
    // The 302 surfaces as a non-success status error — it is NOT followed.
    assert!(
        matches!(err, A2aError::Request(_)),
        "a redirect must surface as an error, got {err:?}"
    );
    assert_eq!(
        secret_hits.load(Ordering::SeqCst),
        0,
        "the redirect target must NOT be fetched — the gateway must not chase upstream redirects"
    );
}

/// An upstream that publishes its card ONLY under the agent's own path prefix
/// and answers every unrouted path with the catch-all `405` the real one
/// returns. This is the shape of any platform that multiplexes tenants under a
/// path, and of any self-hosted agent behind an ingress path.
async fn spawn_path_hosted_agent() -> SocketAddr {
    async fn card(headers: HeaderMap) -> Json<Value> {
        Json(json!({
            "name": "Path Hosted Agent",
            "url": "https://upstream.example.com/v3/a2a/serve/agent-42",
            "protocolVersion": "0.3.0",
            "echoed_version": seen_version(&headers),
        }))
    }

    async fn rpc(Json(body): Json<Value>) -> Json<Value> {
        Json(json!({
            "jsonrpc": "2.0",
            "id": body["id"].clone(),
            "result": {"kind": "task", "id": "task-path", "status": {"state": "completed"}}
        }))
    }

    let app = Router::new()
        .route(
            "/v3/a2a/serve/agent-42/.well-known/agent-card.json",
            get(card),
        )
        .route("/v3/a2a/serve/agent-42", post(rpc))
        .fallback(any(|| async { StatusCode::METHOD_NOT_ALLOWED }));
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
async fn discovers_a_card_hosted_under_the_agent_path() {
    // #913: the registered path used to be discarded, so the bridge asked the
    // ORIGIN for a card this agent publishes only under its prefix, and took
    // the catch-all 405 as the agent's answer. No configuration could unblock
    // it — the one `url` field feeds both the card fetch and the RPC endpoint.
    let addr = spawn_path_hosted_agent().await;
    let bridge = HttpBridge::new(upstream(
        format!("http://{addr}/v3/a2a/serve/agent-42"),
        A2aAuth::None,
    ));

    let card = bridge.fetch_agent_card().await.unwrap();
    assert_eq!(card.name, "Path Hosted Agent");
    // And the endpoint itself still resolves off the same registered URL.
    let resp = bridge.send(&message_send("p-1")).await.unwrap();
    assert_eq!(resp["result"]["id"], "task-path");
}

#[tokio::test]
async fn still_discovers_a_card_hosted_at_the_origin() {
    // The origin URI remains a candidate, so every agent registered while it
    // was the ONLY candidate keeps resolving. `spawn_agent` serves its card
    // there and nowhere else.
    let addr = spawn_agent().await;
    let bridge = HttpBridge::new(upstream(format!("http://{addr}/a2a"), A2aAuth::None));

    assert_eq!(bridge.fetch_agent_card().await.unwrap().name, "Test Agent");
}

#[tokio::test]
async fn reports_a_failure_when_no_candidate_serves_a_card() {
    let addr = spawn_path_hosted_agent().await;
    let bridge = HttpBridge::new(upstream(format!("http://{addr}/nope"), A2aAuth::None));

    let err = bridge.fetch_agent_card().await.unwrap_err();
    assert!(
        matches!(err, A2aError::Connect(_)),
        "exhausting every candidate must surface the upstream failure, got {err:?}"
    );
}

#[tokio::test]
async fn announces_the_pinned_wire_version_on_every_upstream_call() {
    // #911: the gateway sent no `A2A-Version` at all. The spec makes an agent
    // read an absent value as 0.3, so a 1.0-pinned agent rejected every call
    // with VersionNotSupportedError and `protocol_version` was inert.
    let addr = spawn_agent().await;

    let pinned_10 = HttpBridge::new(upstream(format!("http://{addr}/a2a"), A2aAuth::None));
    assert_eq!(
        pinned_10.fetch_agent_card().await.unwrap().rest["echoed_version"],
        "1.0",
        "the card fetch must announce the pinned version too"
    );
    assert_eq!(
        pinned_10.send(&message_send("v-1")).await.unwrap()["result"]["echoed_version"],
        "1.0"
    );

    let pinned_03 = HttpBridge::new(A2aUpstream {
        protocol_version: A2aProtocolVersion::V0_3,
        ..upstream(format!("http://{addr}/a2a"), A2aAuth::None)
    });
    assert_eq!(
        pinned_03.fetch_agent_card().await.unwrap().rest["echoed_version"],
        "0.3"
    );
    assert_eq!(
        pinned_03.send(&message_send("v-0")).await.unwrap()["result"]["echoed_version"],
        "0.3"
    );
}

/// An upstream that accepts the connection and never answers, so every
/// candidate URI burns its whole deadline.
async fn spawn_black_hole() -> SocketAddr {
    let app = Router::new().fallback(any(|| async {
        std::future::pending::<()>().await;
        StatusCode::OK
    }));
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
async fn the_card_fetch_deadline_covers_the_whole_candidate_walk() {
    // `timeout_ms` bounds the card fetch as ONE upstream operation. Walking
    // several candidates must not hand each a fresh deadline, or a hung agent
    // pins a gateway request for `candidates × timeout_ms`.
    let addr = spawn_black_hole().await;
    let bridge = HttpBridge::new(A2aUpstream {
        timeout: Duration::from_millis(600),
        ..upstream(format!("http://{addr}/a2a"), A2aAuth::None)
    });

    let started = Instant::now();
    let err = bridge.fetch_agent_card().await.unwrap_err();
    let elapsed = started.elapsed();

    assert!(matches!(err, A2aError::Connect(_)), "got {err:?}");
    // Four candidates would be 2.4s if each restarted the clock. The bound is
    // loose enough for a slow CI box while staying far below that.
    assert!(
        elapsed < Duration::from_millis(1500),
        "card fetch took {elapsed:?}; the candidate walk must share ONE deadline"
    );
}
