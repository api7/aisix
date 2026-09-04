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

/// Every header an inbound request carried, name -> the list of values sent
/// under it. A list rather than one value on purpose: a gateway that appended
/// its own credential behind a forwarded one would still read correctly under
/// `get`, and only the arity shows it.
fn header_dump(headers: &HeaderMap) -> Value {
    let mut out = serde_json::Map::new();
    for name in headers.keys() {
        let values: Vec<Value> = headers
            .get_all(name)
            .iter()
            .map(|v| Value::String(v.to_str().unwrap_or("<non-ascii>").to_string()))
            .collect();
        out.insert(name.as_str().to_string(), Value::Array(values));
    }
    Value::Object(out)
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
            "echoed_headers": header_dump(&headers),
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
                "echoed_headers": header_dump(&headers),
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

/// An upstream that answers `message/stream` with a real SSE body, written in
/// awkward chunks: two events in one write, an event split across two writes,
/// and comment / `event:` framing in between. Proves the reader reassembles
/// across chunk boundaries rather than assuming one chunk is one event.
async fn spawn_streaming_agent() -> SocketAddr {
    async fn stream(headers: HeaderMap) -> impl IntoResponse {
        let seen_headers = header_dump(&headers).to_string();
        let seen_version = seen_version(&headers).to_string();
        let accept = headers
            .get("accept")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let chunks: Vec<Result<String, std::convert::Infallible>> = vec![
            Ok(format!(
                ": open\ndata: {{\"jsonrpc\":\"2.0\",\"id\":\"s\",\"result\":{{\"seq\":1,\"version\":{seen_version},\"accept\":\"{accept}\",\"headers\":{seen_headers}}}}}\n\n\
                 event: status-update\ndata: {{\"jsonrpc\":\"2.0\",\"id\":\"s\",\"result\":{{\"seq\":2}}}}\n\n"
            )),
            Ok("data: {\"jsonrpc\":\"2.0\",\"id\":\"s\",\"resu".to_string()),
            Ok("lt\":{\"seq\":3,\"final\":true}}\n\n".to_string()),
        ];
        (
            [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
            axum::body::Body::from_stream(futures::stream::iter(chunks)),
        )
    }

    let app = Router::new().route("/a2a", post(stream));
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
async fn streams_events_as_they_arrive_across_chunk_boundaries() {
    use futures::StreamExt;

    let addr = spawn_streaming_agent().await;
    let bridge = HttpBridge::new(upstream(format!("http://{addr}/a2a"), A2aAuth::None));

    let events: Vec<Value> = bridge
        .send_stream(&json!({"jsonrpc":"2.0","id":"s","method":"message/stream"}))
        .await
        .expect("stream opens")
        .map(|e| e.expect("event parses"))
        .collect()
        .await;

    assert_eq!(events.len(), 3, "got {events:#?}");
    assert_eq!(events[0]["result"]["seq"], 1);
    assert_eq!(events[1]["result"]["seq"], 2);
    // Reassembled from two writes that split mid-JSON.
    assert_eq!(events[2]["result"]["seq"], 3);
    assert_eq!(events[2]["result"]["final"], true);
    // A streaming call is still an A2A call: it announces its version and asks
    // for the streaming content type.
    assert_eq!(events[0]["result"]["version"], "1.0");
    assert_eq!(events[0]["result"]["accept"], "text/event-stream");
}

#[tokio::test]
async fn a_refused_stream_surfaces_before_any_event() {
    let addr = spawn_path_hosted_agent().await;
    // `/nope` is the catch-all 405, so the upstream refuses the call outright.
    let bridge = HttpBridge::new(upstream(format!("http://{addr}/nope"), A2aAuth::None));

    let Err(err) = bridge
        .send_stream(&json!({"jsonrpc":"2.0","id":"s","method":"message/stream"}))
        .await
    else {
        panic!("a refused stream must not open");
    };
    assert!(matches!(err, A2aError::Request(_)), "got {err:?}");
}

/// An upstream that answers a streaming call the way an A2A agent refuses one:
/// HTTP 200 with a JSON-RPC error body, not SSE.
async fn spawn_json_refusing_agent() -> SocketAddr {
    let app = Router::new().route(
        "/a2a",
        post(|| async {
            Json(json!({
                "jsonrpc": "2.0",
                "id": "s",
                "error": {"code": -32601, "message": "streaming not supported"}
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
async fn a_json_rpc_error_at_http_200_reaches_the_caller() {
    use futures::StreamExt;

    // A JSON-RPC error is delivered at HTTP 200, so an agent refusing a
    // streaming call answers JSON rather than SSE. Handed to the SSE reader
    // that body has no `data:` line, so the caller would get an empty,
    // apparently successful stream and the refusal would vanish.
    let addr = spawn_json_refusing_agent().await;
    let bridge = HttpBridge::new(upstream(format!("http://{addr}/a2a"), A2aAuth::None));

    let events: Vec<Value> = bridge
        .send_stream(&json!({"jsonrpc":"2.0","id":"s","method":"message/stream"}))
        .await
        .expect("a 200 answer opens")
        .map(|e| e.expect("event parses"))
        .collect()
        .await;

    assert_eq!(events.len(), 1, "the refusal must reach the caller");
    assert_eq!(events[0]["error"]["code"], -32601);
}

#[tokio::test]
async fn opening_a_stream_is_bounded_even_though_reading_it_is_not() {
    // Until the response headers arrive there is no stream and no keep-alive,
    // so an upstream that accepts the connection and then says nothing would
    // pin this request — and the quota slot it holds — forever.
    let addr = spawn_black_hole().await;
    let bridge = HttpBridge::new(A2aUpstream {
        timeout: Duration::from_millis(400),
        ..upstream(format!("http://{addr}/a2a"), A2aAuth::None)
    });

    let started = Instant::now();
    let Err(err) = bridge
        .send_stream(&json!({"jsonrpc":"2.0","id":"s","method":"message/stream"}))
        .await
    else {
        panic!("opening must not hang on an upstream that never answers");
    };
    let elapsed = started.elapsed();

    assert!(matches!(err, A2aError::Connect(_)), "got {err:?}");
    assert!(
        elapsed < Duration::from_millis(2_000),
        "opening took {elapsed:?}; it must be bounded by timeout_ms"
    );
}

#[tokio::test]
async fn a_malformed_final_line_fails_the_stream() {
    use futures::StreamExt;

    // The body ends mid-event with no trailing newline. Treating that as a
    // clean end would let a truncated task read as a complete one.
    async fn truncated() -> impl IntoResponse {
        let chunks: Vec<Result<String, std::convert::Infallible>> = vec![Ok(
            "data: {\"jsonrpc\":\"2.0\",\"result\":{\"seq\":1}}\n\ndata: {\"jsonrpc\"".to_string(),
        )];
        (
            [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
            axum::body::Body::from_stream(futures::stream::iter(chunks)),
        )
    }
    let app = Router::new().route("/a2a", post(truncated));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app.into_make_service())
            .await
            .unwrap();
    });

    let bridge = HttpBridge::new(upstream(format!("http://{addr}/a2a"), A2aAuth::None));
    let events: Vec<Result<Value, A2aError>> = bridge
        .send_stream(&json!({"jsonrpc":"2.0","id":"s","method":"message/stream"}))
        .await
        .expect("stream opens")
        .collect()
        .await;

    assert_eq!(events.len(), 2, "got {events:#?}");
    assert!(events[0].is_ok());
    assert!(
        events[1].is_err(),
        "a truncated trailing event must fail the stream, not end it quietly"
    );
}

// ---------------------------------------------------------------------------
// `forward_client_headers` — the operator names inbound client headers that
// must reach this agent. The gateway rebuilds the outbound JSON-RPC message,
// so the shared resolver applies both blocking tiers; `a2a-version` is the one
// slot this surface owns on top of them.
// ---------------------------------------------------------------------------

/// A registered agent forwarding `patterns`.
fn agent_forwarding(patterns: &[&str]) -> aisix_core::A2aAgent {
    serde_json::from_value(json!({
        "name": "fwd",
        "url": "https://agents.example.com/a2a",
        "forward_client_headers": patterns,
    }))
    .expect("agent deserialises")
}

/// The headers a caller sent to the gateway.
fn client_sent(pairs: &[(&str, &str)]) -> HeaderMap {
    let mut map = HeaderMap::new();
    for (name, value) in pairs {
        map.insert(
            axum::http::HeaderName::try_from(*name).expect("header name"),
            axum::http::HeaderValue::from_str(value).expect("header value"),
        );
    }
    map
}

/// What the gateway resolves out of a caller's request for that agent — the
/// exact call the `/a2a` handlers make.
fn resolved(
    patterns: &[&str],
    sent: &[(&str, &str)],
) -> Vec<(axum::http::HeaderName, axum::http::HeaderValue)> {
    aisix_a2a::forwarded_client_headers(&agent_forwarding(patterns), Some(&client_sent(sent)))
}

/// Values the upstream saw under `name`, from an echoed header dump.
fn seen(dump: &Value, name: &str) -> Vec<String> {
    dump.get(name)
        .and_then(Value::as_array)
        .map(|vs| {
            vs.iter()
                .map(|v| v.as_str().unwrap_or_default().to_string())
                .collect()
        })
        .unwrap_or_default()
}

/// A forwarded client header reaches the upstream agent on a JSON-RPC call.
#[tokio::test]
async fn a_forwarded_client_header_reaches_the_upstream_agent() {
    let addr = spawn_agent().await;
    let bridge = HttpBridge::new(upstream(format!("http://{addr}/a2a"), A2aAuth::None))
        .with_forwarded_client_headers(resolved(
            &["x-user-jwt"],
            &[("x-user-jwt", "eyJhbGciOi.caller")],
        ));

    let resp = bridge.send(&message_send("f-1")).await.unwrap();
    assert_eq!(
        seen(&resp["result"]["echoed_headers"], "x-user-jwt"),
        vec!["eyJhbGciOi.caller"]
    );
}

/// Forwarding `authorization` hands the agent the caller's own credential
/// INSTEAD of the gateway-held bearer — exactly one credential on the wire.
#[tokio::test]
async fn a_forwarded_authorization_replaces_the_gateway_bearer() {
    let addr = spawn_agent().await;
    let bridge = HttpBridge::new(upstream(
        format!("http://{addr}/a2a"),
        A2aAuth::Bearer("gateway-held-secret".into()),
    ))
    .with_forwarded_client_headers(resolved(
        &["authorization"],
        &[("authorization", "Bearer caller-token")],
    ));

    let resp = bridge.send(&message_send("f-2")).await.unwrap();
    // The arity is the assertion: `reqwest`'s `header` appends, so a gateway
    // credential that failed to stand aside would ride behind the caller's and
    // read correctly under the first value alone.
    assert_eq!(
        seen(&resp["result"]["echoed_headers"], "authorization"),
        vec!["Bearer caller-token"]
    );
}

/// The same, in the other credential slot: `api_key` auth sends `x-api-key`,
/// and a forwarded copy displaces it rather than joining it.
#[tokio::test]
async fn a_forwarded_api_key_replaces_the_gateway_key() {
    let addr = spawn_agent().await;
    let bridge = HttpBridge::new(upstream(
        format!("http://{addr}/a2a"),
        A2aAuth::ApiKey("gateway-held-key".into()),
    ))
    .with_forwarded_client_headers(resolved(
        &["x-api-key"],
        &[("x-api-key", "callers-own-key")],
    ));

    let resp = bridge.send(&message_send("f-3")).await.unwrap();
    assert_eq!(
        seen(&resp["result"]["echoed_headers"], "x-api-key"),
        vec!["callers-own-key"]
    );
}

/// An agent forwarding nothing still gets the gateway's own credential and
/// nothing else — the default every registered agent keeps.
#[tokio::test]
async fn without_a_forward_the_gateway_bearer_is_still_the_only_credential() {
    let addr = spawn_agent().await;
    let bridge = HttpBridge::new(upstream(
        format!("http://{addr}/a2a"),
        A2aAuth::Bearer("gateway-held-secret".into()),
    ))
    .with_forwarded_client_headers(resolved(&[], &[("authorization", "Bearer caller-token")]));

    let resp = bridge.send(&message_send("f-4")).await.unwrap();
    assert_eq!(
        seen(&resp["result"]["echoed_headers"], "authorization"),
        vec!["Bearer gateway-held-secret"]
    );
}

/// A forwarded value never reaches a log through `Debug` — it may be the
/// caller's own credential, and the dispatch path formats the bridge on error.
#[test]
fn a_forwarded_value_is_redacted_in_debug() {
    let bridge = HttpBridge::new(upstream("http://x/a2a".into(), A2aAuth::None))
        .with_forwarded_client_headers(resolved(
            &["x-user-jwt"],
            &[("x-user-jwt", "eyJhbGciOi.caller")],
        ));
    let rendered = format!("{bridge:?}");
    assert!(
        rendered.contains("x-user-jwt"),
        "the slot stays diagnosable: {rendered}"
    );
    assert!(
        !rendered.contains("eyJhbGciOi.caller"),
        "the value must not print: {rendered}"
    );
}

/// `a2a-version` is the gateway's own announcement of the pinned wire version,
/// so a caller cannot claim it — not even under `["*"]`, which admits every
/// ordinary header beside it.
#[tokio::test]
async fn the_version_announcement_is_never_forwardable() {
    let sent = &[("a2a-version", "0.3"), ("x-plain", "kept")][..];
    let names: Vec<String> = resolved(&["*"], sent)
        .into_iter()
        .map(|(n, _)| n.as_str().to_string())
        .collect();
    assert_eq!(names, vec!["x-plain"], "a glob must not reach a2a-version");
    // Named exactly, it is still refused: unlike a credential slot, this one
    // is the gateway's own assertion rather than an identity the operator may
    // delegate.
    assert!(resolved(&["a2a-version"], sent).is_empty());

    // And on the wire the agent sees the pinned version, once.
    let addr = spawn_agent().await;
    let bridge = HttpBridge::new(upstream(format!("http://{addr}/a2a"), A2aAuth::None))
        .with_forwarded_client_headers(resolved(&["*"], sent));
    let resp = bridge.send(&message_send("f-5")).await.unwrap();
    assert_eq!(
        seen(&resp["result"]["echoed_headers"], "a2a-version"),
        vec!["1.0"]
    );
    assert_eq!(
        seen(&resp["result"]["echoed_headers"], "x-plain"),
        vec!["kept"]
    );
}

/// The card fetch is an upstream call like any other, so it forwards too — an
/// agent that gates card discovery on the end user's own credential is exactly
/// the deployment this capability exists for.
#[tokio::test]
async fn the_agent_card_fetch_forwards_too() {
    let addr = spawn_agent().await;
    let bridge = HttpBridge::new(upstream(format!("http://{addr}/a2a"), A2aAuth::None))
        .with_forwarded_client_headers(resolved(
            &["x-user-jwt"],
            &[("x-user-jwt", "eyJhbGciOi.caller")],
        ));

    let card = bridge.fetch_agent_card().await.unwrap();
    assert_eq!(
        seen(&card.rest["echoed_headers"], "x-user-jwt"),
        vec!["eyJhbGciOi.caller"]
    );
}

/// And so does the streaming path, which opens its request at a different call
/// site than the buffered one.
#[tokio::test]
async fn a_streaming_call_forwards_too() {
    use futures::StreamExt;

    let addr = spawn_streaming_agent().await;
    let bridge = HttpBridge::new(upstream(format!("http://{addr}/a2a"), A2aAuth::None))
        .with_forwarded_client_headers(resolved(
            &["x-user-jwt"],
            &[("x-user-jwt", "eyJhbGciOi.caller")],
        ));

    let events: Vec<Value> = bridge
        .send_stream(&json!({"jsonrpc":"2.0","id":"s","method":"message/stream"}))
        .await
        .expect("stream opens")
        .map(|e| e.expect("event parses"))
        .collect()
        .await;

    assert_eq!(
        seen(&events[0]["result"]["headers"], "x-user-jwt"),
        vec!["eyJhbGciOi.caller"]
    );
    // The streaming call still asks for SSE with its own `accept`, which the
    // forward's blocked set is what keeps a caller from overriding.
    assert_eq!(events[0]["result"]["accept"], "text/event-stream");
}
