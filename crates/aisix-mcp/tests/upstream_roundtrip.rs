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
use std::sync::Arc;

use aisix_mcp::{McpBridge, McpUpstream, RmcpBridge};
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
