//! End-to-end test of the dual-role gateway: AISIX as an MCP *server* to a
//! downstream agent, fronting two *real* upstream MCP servers.
//!
//! Topology, all real Streamable HTTP over ephemeral ports (no mock transport):
//!
//!   downstream rmcp client  ──►  McpGateway (/mcp)  ──►  upstream "alpha" (echo)
//!                                                   └──►  upstream "beta"  (echo)
//!
//! Each upstream labels its echo so routing is observable. Pins: aggregated +
//! namespaced `tools/list`, `tools/call` routes to the owning upstream, and
//! bad/prefixless names are rejected.

use std::net::SocketAddr;
use std::sync::Arc;

use aisix_mcp::{
    streamable_http_service, McpBridge, McpError, McpGateway, McpTool, McpToolResult, McpUpstream,
    RmcpBridge,
};
use rmcp::model::{
    CallToolRequestParams, CallToolResult, Content, ErrorData, ListToolsResult,
    PaginatedRequestParams, ServerCapabilities, ServerInfo, Tool,
};
use rmcp::service::RequestContext;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::{RoleServer, ServerHandler, ServiceExt};

/// A real upstream MCP server exposing one `echo` tool that prefixes its reply
/// with `label`, so the test can tell which upstream actually handled a call.
#[derive(Clone)]
struct LabeledEcho {
    label: &'static str,
}

impl ServerHandler for LabeledEcho {
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
        let tool = Tool::new(
            "echo",
            "Echo back the provided text",
            schema.as_object().expect("schema is an object").clone(),
        );
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
        // `fail` drives the tool-level-error path: a valid call whose tool
        // reports failure, returned as `Ok(CallToolResult::error(..))`.
        if text == "fail" {
            return Ok(CallToolResult::error(vec![Content::text("boom")]));
        }
        Ok(CallToolResult::success(vec![Content::text(format!(
            "{}:{text}",
            self.label
        ))]))
    }

    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info
    }
}

/// Start a labeled upstream echo server; return its bound address.
async fn spawn_upstream(label: &'static str) -> SocketAddr {
    let service = StreamableHttpService::new(
        move || Ok(LabeledEcho { label }),
        Arc::new(LocalSessionManager::default()),
        StreamableHttpServerConfig::default(),
    );
    serve(axum::Router::new().nest_service("/mcp", service)).await
}

/// Serve the gateway itself; return its bound address.
async fn spawn_gateway(gateway: McpGateway) -> SocketAddr {
    serve(axum::Router::new().nest_service("/mcp", streamable_http_service(gateway))).await
}

async fn serve(app: axum::Router) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve");
    });
    addr
}

/// A bridge whose every operation fails — stands in for an unreachable or
/// broken upstream so the graceful-skip path is deterministic.
struct FailingBridge;

#[async_trait::async_trait]
impl McpBridge for FailingBridge {
    async fn list_tools(&self) -> Result<Vec<McpTool>, McpError> {
        Err(McpError::Request("simulated upstream failure".into()))
    }
    async fn call_tool(
        &self,
        _name: &str,
        _arguments: serde_json::Value,
    ) -> Result<McpToolResult, McpError> {
        Err(McpError::Request("simulated upstream failure".into()))
    }
}

/// Connect a bridge to a freshly-spawned labeled upstream.
async fn bridge_to(label: &'static str) -> Arc<dyn McpBridge> {
    let addr = spawn_upstream(label).await;
    let bridge = RmcpBridge::connect(&McpUpstream::new(format!("http://{addr}/mcp")))
        .await
        .expect("connect upstream bridge");
    Arc::new(bridge)
}

/// Decode the first text content block of a tool result.
fn first_text(result: &CallToolResult) -> String {
    let value = serde_json::to_value(&result.content).expect("encode content");
    value[0]["text"].as_str().unwrap_or_default().to_string()
}

#[tokio::test]
async fn aggregates_and_routes_across_upstreams() {
    let gateway = McpGateway::new([
        ("alpha".to_string(), bridge_to("alpha").await),
        ("beta".to_string(), bridge_to("beta").await),
    ]);
    let gw_addr = spawn_gateway(gateway).await;

    // The downstream agent talks to AISIX as if it were a single MCP server.
    let client = ()
        .serve(StreamableHttpClientTransport::from_uri(format!(
            "http://{gw_addr}/mcp"
        )))
        .await
        .expect("downstream client connects to gateway");

    // tools/list is aggregated and namespaced `server__tool`.
    let tools = client.list_all_tools().await.expect("list tools");
    let names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
    assert_eq!(
        tools.len(),
        2,
        "both upstreams' tools should appear: {names:?}"
    );
    assert!(
        names.contains(&"alpha__echo"),
        "missing alpha tool: {names:?}"
    );
    assert!(
        names.contains(&"beta__echo"),
        "missing beta tool: {names:?}"
    );

    // tools/call routes to the owning upstream — proven by the label prefix.
    let from_alpha = client
        .call_tool(call("alpha__echo", "hi"))
        .await
        .expect("call alpha");
    assert_eq!(first_text(&from_alpha), "alpha:hi");

    let from_beta = client
        .call_tool(call("beta__echo", "hi"))
        .await
        .expect("call beta");
    assert_eq!(first_text(&from_beta), "beta:hi");

    // Unknown server and a prefixless name both error, not misroute.
    assert!(
        client.call_tool(call("ghost__echo", "x")).await.is_err(),
        "unknown server must error"
    );
    assert!(
        client.call_tool(call("echo", "x")).await.is_err(),
        "prefixless tool name must error"
    );
}

#[tokio::test]
async fn skips_failing_upstream_keeping_the_rest() {
    // One healthy upstream + one that fails every call. tools/list must still
    // return the healthy upstream's tools, not collapse to empty or error.
    let gateway = McpGateway::new([
        ("alpha".to_string(), bridge_to("alpha").await),
        (
            "down".to_string(),
            Arc::new(FailingBridge) as Arc<dyn McpBridge>,
        ),
    ]);
    let gw_addr = spawn_gateway(gateway).await;
    let client = ()
        .serve(StreamableHttpClientTransport::from_uri(format!(
            "http://{gw_addr}/mcp"
        )))
        .await
        .expect("connect");

    let tools = client.list_all_tools().await.expect("list tools");
    let names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
    assert_eq!(
        tools.len(),
        1,
        "failing upstream's tools dropped, healthy one kept: {names:?}"
    );
    assert!(
        names.contains(&"alpha__echo"),
        "alpha tool missing: {names:?}"
    );
}

#[tokio::test]
async fn propagates_tool_level_error_as_ok() {
    // An upstream tool that reports failure must reach the agent as a tool
    // result with is_error=true — NOT as a transport/protocol Err.
    let gateway = McpGateway::new([("alpha".to_string(), bridge_to("alpha").await)]);
    let gw_addr = spawn_gateway(gateway).await;
    let client = ()
        .serve(StreamableHttpClientTransport::from_uri(format!(
            "http://{gw_addr}/mcp"
        )))
        .await
        .expect("connect");

    let result = client
        .call_tool(call("alpha__echo", "fail"))
        .await
        .expect("tool-level error must be Ok(error_result), not a protocol Err");
    assert_eq!(result.is_error, Some(true), "tool-level error flag lost");
    assert_eq!(first_text(&result), "boom");
}

#[tokio::test]
async fn duplicate_upstream_name_keeps_first() {
    // Two registrations under the same name: the first wins, the second is
    // dropped — no duplicate tool names, all calls route to the first.
    let gateway = McpGateway::new([
        ("dup".to_string(), bridge_to("first").await),
        ("dup".to_string(), bridge_to("second").await),
    ]);
    let gw_addr = spawn_gateway(gateway).await;
    let client = ()
        .serve(StreamableHttpClientTransport::from_uri(format!(
            "http://{gw_addr}/mcp"
        )))
        .await
        .expect("connect");

    let tools = client.list_all_tools().await.expect("list tools");
    assert_eq!(tools.len(), 1, "duplicate name should yield one tool");
    assert_eq!(tools[0].name.as_ref(), "dup__echo");

    let result = client
        .call_tool(call("dup__echo", "hi"))
        .await
        .expect("call");
    assert_eq!(
        first_text(&result),
        "first:hi",
        "should route to first registration"
    );
}

/// Build a `tools/call` for `name` with a single `text` argument.
fn call(name: &'static str, text: &str) -> CallToolRequestParams {
    let args = serde_json::json!({ "text": text });
    CallToolRequestParams::new(name).with_arguments(args.as_object().unwrap().clone())
}
