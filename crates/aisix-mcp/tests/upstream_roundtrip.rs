//! End-to-end test of [`RmcpBridge`] against a *real* MCP server.
//!
//! No mock transport: we stand up an actual `rmcp` Streamable HTTP server
//! (an "echo" tool) on an ephemeral port, nested in axum, and drive it through
//! the public `McpBridge` surface over real HTTP — the same path a production
//! upstream takes. Two contracts are pinned:
//!   1. `initialize` → `tools/list` → `tools/call` round-trips correctly.
//!   2. The gateway-held Bearer is sent to the upstream (and its absence is
//!      rejected), per the MCP authorization no-passthrough model.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use aisix_mcp::{EphemeralBridge, McpBridge, McpUpstream, OAuthClientConfig, RmcpBridge};
use rmcp::model::{
    CallToolRequestParams, CallToolResult, Content, ListToolsResult, PaginatedRequestParams,
    ServerCapabilities, ServerInfo, Tool,
};
use rmcp::service::RequestContext;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
use rmcp::{ErrorData, RoleServer, ServerHandler};

/// A minimal real MCP server exposing one tool, `echo`, that returns its
/// `text` argument back as a text content block.
#[derive(Clone, Default)]
struct EchoServer;

impl ServerHandler for EchoServer {
    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "text": { "type": "string" } },
            "required": ["text"],
        });
        let schema_obj = schema.as_object().expect("schema is an object").clone();
        let tool = Tool::new("echo", "Echo back the provided text", schema_obj);
        Ok(ListToolsResult::with_all_items(vec![tool]))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        if request.name != "echo" {
            return Err(ErrorData::invalid_params(
                format!("unknown tool: {}", request.name),
                None,
            ));
        }
        let text = request
            .arguments
            .as_ref()
            .and_then(|m| m.get("text"))
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        // A `sleep` argument lets the timeout test drive a slow upstream.
        if text == "sleep" {
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        }
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info
    }
}

/// Reject any request whose `Authorization` header is not `Bearer <expected>`,
/// when an expected token is configured. Lets the test assert that the
/// gateway-held credential actually reaches the upstream on every request.
async fn require_bearer(
    axum::extract::State(expected): axum::extract::State<Option<String>>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Some(expected) = expected.as_deref() {
        let presented = request
            .headers()
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok());
        if presented != Some(&format!("Bearer {expected}")) {
            return (axum::http::StatusCode::UNAUTHORIZED, "missing bearer").into_response();
        }
    }
    next.run(request).await
}

/// Exact-header gate for the richer auth types: reject any request that does
/// not carry `name: expected` with a `401`. `www_authenticate` toggles the
/// `WWW-Authenticate` challenge on the rejection — the rmcp client maps a 401
/// with and without the challenge to two different error shapes, and the
/// token-invalidation path must handle both.
#[derive(Clone)]
struct RequiredHeader {
    name: &'static str,
    expected: String,
    www_authenticate: bool,
}

async fn require_exact_header(
    axum::extract::State(required): axum::extract::State<RequiredHeader>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    let presented = request
        .headers()
        .get(required.name)
        .and_then(|v| v.to_str().ok());
    if presented != Some(required.expected.as_str()) {
        let mut response =
            (axum::http::StatusCode::UNAUTHORIZED, "credential rejected").into_response();
        if required.www_authenticate {
            response.headers_mut().insert(
                axum::http::header::WWW_AUTHENTICATE,
                axum::http::HeaderValue::from_static("Bearer realm=\"mcp\""),
            );
        }
        return response;
    }
    next.run(request).await
}

/// Start an echo server that requires an exact header on every request.
async fn spawn_echo_server_requiring_header(required: RequiredHeader) -> SocketAddr {
    let service = StreamableHttpService::new(
        || Ok(EchoServer),
        Arc::new(LocalSessionManager::default()),
        StreamableHttpServerConfig::default(),
    );
    let app = axum::Router::new().nest_service("/mcp", service).layer(
        axum::middleware::from_fn_with_state(required, require_exact_header),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve");
    });
    addr
}

/// A minimal OAuth token endpoint minting `tok-<n>` (`n` = 1-based hit count)
/// with a long `expires_in`. Returns its URL and the hit counter, so tests can
/// assert exactly how many times a token was (re-)minted.
async fn spawn_token_endpoint() -> (String, Arc<AtomicUsize>) {
    use axum::response::IntoResponse;
    let hits = Arc::new(AtomicUsize::new(0));
    let counter = hits.clone();
    let app = axum::Router::new().route(
        "/oauth/token",
        axum::routing::post(move || {
            let counter = counter.clone();
            async move {
                let n = counter.fetch_add(1, Ordering::SeqCst) + 1;
                axum::Json(serde_json::json!({
                    "access_token": format!("tok-{n}"),
                    "token_type": "Bearer",
                    "expires_in": 3600,
                }))
                .into_response()
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve");
    });
    (format!("http://{addr}/oauth/token"), hits)
}

/// Start the echo server on an ephemeral port; return its bound address.
async fn spawn_echo_server(require_bearer_token: Option<&str>) -> SocketAddr {
    let service = StreamableHttpService::new(
        || Ok(EchoServer),
        Arc::new(LocalSessionManager::default()),
        StreamableHttpServerConfig::default(),
    );
    let expected = require_bearer_token.map(str::to_string);
    let app = axum::Router::new().nest_service("/mcp", service).layer(
        axum::middleware::from_fn_with_state(expected, require_bearer),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve");
    });
    addr
}

#[tokio::test]
async fn lists_and_calls_tools_over_streamable_http() {
    let addr = spawn_echo_server(None).await;
    let upstream = McpUpstream::new(format!("http://{addr}/mcp"));
    let bridge = RmcpBridge::connect(&upstream)
        .await
        .expect("connect to upstream MCP server");

    // tools/list surfaces the echo tool with its schema.
    let tools = bridge.list_tools().await.expect("list tools");
    assert_eq!(tools.len(), 1, "expected exactly one tool");
    let echo = &tools[0];
    assert_eq!(echo.name, "echo");
    assert_eq!(
        echo.description.as_deref(),
        Some("Echo back the provided text")
    );
    assert!(
        echo.input_schema["properties"]["text"].is_object(),
        "schema should describe the `text` argument, got: {}",
        echo.input_schema
    );

    // tools/call echoes the argument back.
    let result = bridge
        .call_tool("echo", serde_json::json!({ "text": "hello mcp" }))
        .await
        .expect("call echo tool");
    assert!(!result.is_error, "echo should not be a tool error");
    assert_eq!(
        result.content[0]["text"], "hello mcp",
        "echoed text block should equal the input, got: {}",
        result.content
    );

    // An unknown tool surfaces as an error, not a silent empty result.
    let unknown = bridge
        .call_tool("does_not_exist", serde_json::Value::Null)
        .await;
    assert!(unknown.is_err(), "unknown tool must error");
}

#[tokio::test]
async fn forwards_gateway_held_bearer_to_upstream() {
    let addr = spawn_echo_server(Some("s3cret-token")).await;
    let url = format!("http://{addr}/mcp");

    // Without the gateway-held credential, the upstream rejects the session.
    let unauth = RmcpBridge::connect(&McpUpstream::new(url.clone())).await;
    assert!(
        unauth.is_err(),
        "connect without bearer must fail against an auth-required upstream"
    );

    // With it, the session establishes and tools are reachable — proving the
    // gateway-held Bearer is forwarded to the upstream.
    let upstream = McpUpstream::new(url).with_bearer("s3cret-token");
    let bridge = RmcpBridge::connect(&upstream)
        .await
        .expect("connect with gateway-held bearer");
    let tools = bridge.list_tools().await.expect("list tools");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "echo");
}

#[tokio::test]
async fn forwards_gateway_held_api_key_to_upstream() {
    let addr = spawn_echo_server_requiring_header(RequiredHeader {
        name: "x-api-key",
        expected: "k-123".to_string(),
        www_authenticate: false,
    })
    .await;
    let url = format!("http://{addr}/mcp");

    // Without the gateway-held key, the upstream rejects the session.
    let unauth = RmcpBridge::connect(&McpUpstream::new(url.clone())).await;
    assert!(
        unauth.is_err(),
        "connect without the API key must fail against a key-required upstream"
    );

    // With it, the session establishes — proving `x-api-key` is sent on
    // every upstream request.
    let upstream = McpUpstream::new(url).with_api_key("k-123");
    let bridge = RmcpBridge::connect(&upstream)
        .await
        .expect("connect with gateway-held API key");
    let tools = bridge.list_tools().await.expect("list tools");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "echo");
}

#[tokio::test]
async fn api_key_with_invalid_header_bytes_fails_cleanly() {
    let addr = spawn_echo_server(None).await;
    // A newline is not a valid HTTP header byte: the connect must return a
    // clean config error (no panic) and must not echo the key material.
    let upstream = McpUpstream::new(format!("http://{addr}/mcp")).with_api_key("bad\nkey");
    let err = match RmcpBridge::connect(&upstream).await {
        Ok(_) => panic!("invalid header bytes must fail cleanly, not connect"),
        Err(err) => err,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("not a valid HTTP header value"),
        "expected the config-error message, got: {msg}"
    );
    assert!(!msg.contains("bad\n"), "key material must not leak: {msg}");
}

#[tokio::test]
async fn oauth2_mints_token_and_reuses_it_across_operations() {
    let (token_url, mints) = spawn_token_endpoint().await;
    // The upstream accepts exactly the first minted token — so a passing
    // list/call proves the gateway attached `Authorization: Bearer tok-1`.
    let addr = spawn_echo_server(Some("tok-1")).await;

    let upstream = McpUpstream::new(format!("http://{addr}/mcp")).with_oauth2(OAuthClientConfig {
        client_id: "cid-roundtrip".to_string(),
        client_secret: "cs".to_string(),
        token_url,
        scopes: Vec::new(),
    });

    // EphemeralBridge reconnects per operation: the second operation must
    // reuse the cached token instead of minting a new one.
    let bridge = EphemeralBridge::new(upstream);
    let tools = bridge.list_tools().await.expect("list via minted token");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "echo");
    let result = bridge
        .call_tool("echo", serde_json::json!({ "text": "hello oauth" }))
        .await
        .expect("call via cached token");
    assert_eq!(result.content[0]["text"], "hello oauth");
    assert_eq!(
        mints.load(Ordering::SeqCst),
        1,
        "the token must be minted once and then served from the cache"
    );
}

#[tokio::test]
async fn upstream_401_with_challenge_invalidates_the_cached_token() {
    let (token_url, mints) = spawn_token_endpoint().await;
    // This upstream never accepts our minted tokens: every request gets a
    // 401 WITH a `WWW-Authenticate` challenge (rmcp's `AuthRequired` shape).
    let addr = spawn_echo_server_requiring_header(RequiredHeader {
        name: "authorization",
        expected: "Bearer some-other-token".to_string(),
        www_authenticate: true,
    })
    .await;

    let upstream = McpUpstream::new(format!("http://{addr}/mcp")).with_oauth2(OAuthClientConfig {
        client_id: "cid-401-challenge".to_string(),
        client_secret: "cs".to_string(),
        token_url,
        scopes: Vec::new(),
    });
    let bridge = EphemeralBridge::new(upstream);

    assert!(bridge.list_tools().await.is_err());
    assert!(bridge.list_tools().await.is_err());
    assert_eq!(
        mints.load(Ordering::SeqCst),
        2,
        "each upstream 401 must invalidate the cached token so the next attempt re-mints"
    );
}

#[tokio::test]
async fn upstream_401_without_challenge_also_invalidates_the_cached_token() {
    let (token_url, mints) = spawn_token_endpoint().await;
    // Same rejection, but WITHOUT `WWW-Authenticate` — rmcp surfaces this as
    // a generic `HTTP 401` response error, the other shape the invalidation
    // path must recognise.
    let addr = spawn_echo_server_requiring_header(RequiredHeader {
        name: "authorization",
        expected: "Bearer some-other-token".to_string(),
        www_authenticate: false,
    })
    .await;

    let upstream = McpUpstream::new(format!("http://{addr}/mcp")).with_oauth2(OAuthClientConfig {
        client_id: "cid-401-bare".to_string(),
        client_secret: "cs".to_string(),
        token_url,
        scopes: Vec::new(),
    });
    let bridge = EphemeralBridge::new(upstream);

    assert!(bridge.list_tools().await.is_err());
    assert!(bridge.list_tools().await.is_err());
    assert_eq!(
        mints.load(Ordering::SeqCst),
        2,
        "a bare upstream 401 must also invalidate the cached token"
    );
}

#[tokio::test]
async fn misconfigured_oauth2_upstream_fails_cleanly_without_leaking() {
    let addr = spawn_echo_server(None).await;
    // `token_url` missing — the canonical mis-configured row the flat schema
    // deliberately lets through. The operation must fail with a clean error,
    // not a panic, and never surface the client secret.
    let upstream = McpUpstream::new(format!("http://{addr}/mcp")).with_oauth2(OAuthClientConfig {
        client_id: "cid-misconfig".to_string(),
        client_secret: "super-secret".to_string(),
        token_url: String::new(),
        scopes: Vec::new(),
    });
    let err = EphemeralBridge::new(upstream)
        .list_tools()
        .await
        .expect_err("a token fetch with no token_url must fail");
    let msg = err.to_string();
    assert!(
        msg.contains("oauth2"),
        "error should name the misconfiguration: {msg}"
    );
    assert!(
        !msg.contains("super-secret"),
        "the client secret must never leak: {msg}"
    );
}

#[tokio::test]
async fn upstream_call_times_out_instead_of_hanging() {
    let addr = spawn_echo_server(None).await;
    let upstream = McpUpstream::new(format!("http://{addr}/mcp"))
        .with_timeout(std::time::Duration::from_millis(200));
    let bridge = RmcpBridge::connect(&upstream).await.expect("connect");

    // The server sleeps 2s on this call; the 200ms deadline must fire first.
    let started = std::time::Instant::now();
    let result = bridge
        .call_tool("echo", serde_json::json!({ "text": "sleep" }))
        .await;
    assert!(result.is_err(), "a call exceeding the deadline must error");
    assert!(
        started.elapsed() < std::time::Duration::from_secs(1),
        "call should give up at the ~200ms deadline, not wait out the 2s server sleep"
    );
}
