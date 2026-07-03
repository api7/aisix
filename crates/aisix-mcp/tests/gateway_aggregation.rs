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

use aisix_core::{AisixSnapshot, McpServer, ResourceEntry};
use aisix_mcp::{
    streamable_http_service, upstream_from_mcp_server, McpAuth, McpBridge, McpError, McpGateway,
    McpTool, McpToolResult, McpUpstream, RmcpBridge, ToolAcl,
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

#[test]
fn upstream_from_mcp_server_maps_auth_and_timeout() {
    let server: McpServer = serde_json::from_value(serde_json::json!({
        "display_name": "gh",
        "url": "https://api.example.com/mcp",
        "auth_type": "bearer",
        "secret": "tok",
        "timeout_ms": 1234
    }))
    .unwrap();
    let upstream = upstream_from_mcp_server(&server);
    assert_eq!(upstream.url, "https://api.example.com/mcp");
    assert_eq!(upstream.timeout, std::time::Duration::from_millis(1234));
    assert!(matches!(upstream.auth, McpAuth::Bearer(ref t) if t == "tok"));

    // `none` auth and absent timeout fall back to defaults.
    let plain: McpServer = serde_json::from_value(serde_json::json!({
        "display_name": "x", "url": "https://x/mcp"
    }))
    .unwrap();
    assert!(matches!(
        upstream_from_mcp_server(&plain).auth,
        McpAuth::None
    ));
}

#[test]
fn upstream_from_mcp_server_maps_api_key_and_oauth2() {
    let api_key: McpServer = serde_json::from_value(serde_json::json!({
        "display_name": "gh",
        "url": "https://api.example.com/mcp",
        "auth_type": "api_key",
        "secret": "k-1"
    }))
    .unwrap();
    assert!(matches!(
        upstream_from_mcp_server(&api_key).auth,
        McpAuth::ApiKey(ref k) if k == "k-1"
    ));

    let oauth: McpServer = serde_json::from_value(serde_json::json!({
        "display_name": "gh2",
        "url": "https://api.example.com/mcp",
        "auth_type": "oauth2",
        "secret": "cs",
        "client_id": "cid",
        "token_url": "https://auth.example.com/token",
        "scopes": ["read", "write"]
    }))
    .unwrap();
    match upstream_from_mcp_server(&oauth).auth {
        McpAuth::OAuth2(cfg) => {
            assert_eq!(cfg.client_id, "cid");
            assert_eq!(cfg.client_secret, "cs");
            assert_eq!(cfg.token_url, "https://auth.example.com/token");
            assert_eq!(cfg.scopes, vec!["read".to_string(), "write".to_string()]);
        }
        other => panic!("expected OAuth2 auth, got {other:?}"),
    }

    // Missing oauth2 fields map to empty strings — the mapping never errors;
    // the token fetch fails cleanly at connect time instead.
    let incomplete: McpServer = serde_json::from_value(serde_json::json!({
        "display_name": "gh3",
        "url": "https://api.example.com/mcp",
        "auth_type": "oauth2"
    }))
    .unwrap();
    match upstream_from_mcp_server(&incomplete).auth {
        McpAuth::OAuth2(cfg) => {
            assert!(cfg.client_id.is_empty());
            assert!(cfg.client_secret.is_empty());
            assert!(cfg.token_url.is_empty());
            assert!(cfg.scopes.is_empty());
        }
        other => panic!("expected OAuth2 auth, got {other:?}"),
    }
}

/// Build a snapshot resource entry for an upstream at `addr`.
fn mcp_entry(id: &str, name: &str, addr: &SocketAddr, enabled: bool) -> ResourceEntry<McpServer> {
    let server: McpServer = serde_json::from_value(serde_json::json!({
        "display_name": name,
        "url": format!("http://{addr}/mcp"),
        "enabled": enabled
    }))
    .unwrap();
    ResourceEntry::new(id, server, 1)
}

#[tokio::test]
async fn from_snapshot_sources_only_enabled_upstreams() {
    let alpha = spawn_upstream("alpha").await;
    let beta = spawn_upstream("beta").await;

    let snapshot = AisixSnapshot::new();
    snapshot
        .mcp_servers
        .insert(mcp_entry("e1", "alpha", &alpha, true));
    snapshot
        .mcp_servers
        .insert(mcp_entry("e2", "beta", &beta, false)); // disabled

    let gw_addr = spawn_gateway(McpGateway::from_snapshot(&snapshot)).await;
    let client = ()
        .serve(StreamableHttpClientTransport::from_uri(format!(
            "http://{gw_addr}/mcp"
        )))
        .await
        .expect("connect");

    // Only the enabled server's tool is aggregated.
    let tools = client.list_all_tools().await.expect("list tools");
    let names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
    assert_eq!(
        tools.len(),
        1,
        "disabled upstream must be skipped: {names:?}"
    );
    assert_eq!(names[0], "alpha__echo");

    // And it routes correctly (ephemeral connect per call).
    let result = client
        .call_tool(call("alpha__echo", "hi"))
        .await
        .expect("call");
    assert_eq!(first_text(&result), "alpha:hi");
}

#[tokio::test]
async fn from_snapshot_degrades_misconfigured_oauth2_upstream_gracefully() {
    let alpha = spawn_upstream("alpha").await;

    let snapshot = AisixSnapshot::new();
    snapshot
        .mcp_servers
        .insert(mcp_entry("e1", "alpha", &alpha, true));
    // An `oauth2` server missing its `token_url`: schema-valid and loadable
    // (the flat schema stays permissive on credential coupling), but its
    // token fetch can never succeed.
    let broken: McpServer = serde_json::from_value(serde_json::json!({
        "display_name": "broken",
        "url": format!("http://{alpha}/mcp"),
        "auth_type": "oauth2",
        "client_id": "cid",
        "secret": "top-secret-cs",
        "enabled": true
    }))
    .unwrap();
    snapshot
        .mcp_servers
        .insert(ResourceEntry::new("e2", broken, 1));

    let gw_addr = spawn_gateway(McpGateway::from_snapshot(&snapshot)).await;
    let client = ()
        .serve(StreamableHttpClientTransport::from_uri(format!(
            "http://{gw_addr}/mcp"
        )))
        .await
        .expect("connect");

    // The mis-configured upstream's tools are simply absent (its failure is
    // logged server-side); the healthy upstream keeps serving.
    let tools = client.list_all_tools().await.expect("list tools");
    let names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
    assert_eq!(
        names,
        vec!["alpha__echo"],
        "misconfigured oauth2 upstream must be skipped, not fatal to the aggregate"
    );

    // Calling into it fails with the generic upstream-failure message — no
    // hang, and nothing about its credentials leaks to the agent.
    let err = client
        .call_tool(call("broken__echo", "hi"))
        .await
        .expect_err("a call through the misconfigured upstream must fail");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("failed to call tool"),
        "expected the generic upstream-failure error, got: {msg}"
    );
    assert!(
        !msg.contains("top-secret-cs") && !msg.contains("oauth"),
        "credential/config detail must not reach the agent: {msg}"
    );

    // The healthy upstream still routes.
    let ok = client
        .call_tool(call("alpha__echo", "hi"))
        .await
        .expect("healthy upstream keeps serving");
    assert_eq!(first_text(&ok), "alpha:hi");
}

#[test]
fn tool_acl_from_allowed_semantics() {
    // No allowed_tools / empty → deny all.
    assert!(matches!(ToolAcl::from_allowed(None), ToolAcl::Allow(ref s) if s.is_empty()));
    assert!(matches!(ToolAcl::from_allowed(Some(&[])), ToolAcl::Allow(ref s) if s.is_empty()));
    // Wildcard → allow all.
    assert!(matches!(
        ToolAcl::from_allowed(Some(&["*".to_string()])),
        ToolAcl::AllowAll
    ));
    // Exact set.
    assert!(matches!(
        ToolAcl::from_allowed(Some(&["a__b".to_string()])),
        ToolAcl::Allow(_)
    ));
    // A per-server wildcard is a scoped Allow, not AllowAll — only a bare
    // `"*"` opens everything.
    assert!(matches!(
        ToolAcl::from_allowed(Some(&["github__*".to_string()])),
        ToolAcl::Allow(_)
    ));
}

#[tokio::test]
async fn tool_acl_filters_list_and_rejects_calls() {
    // Two real upstreams; the key permits only alpha's tool.
    let gateway = McpGateway::new([
        ("alpha".to_string(), bridge_to("alpha").await),
        ("beta".to_string(), bridge_to("beta").await),
    ])
    .with_tool_acl(ToolAcl::from_allowed(Some(&["alpha__echo".to_string()])));
    let gw_addr = spawn_gateway(gateway).await;
    let client = ()
        .serve(StreamableHttpClientTransport::from_uri(format!(
            "http://{gw_addr}/mcp"
        )))
        .await
        .expect("connect");

    // tools/list exposes only the permitted tool — beta's is hidden.
    let tools = client.list_all_tools().await.expect("list tools");
    let names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
    assert_eq!(
        names,
        vec!["alpha__echo"],
        "ACL must hide non-permitted tools"
    );

    // The permitted tool is callable.
    let allowed = client
        .call_tool(call("alpha__echo", "hi"))
        .await
        .expect("permitted call");
    assert_eq!(first_text(&allowed), "alpha:hi");

    // A non-permitted tool is rejected (defense-in-depth), not routed upstream,
    // with a neutral message that doesn't reveal whether the tool/server exists.
    let err = client
        .call_tool(call("beta__echo", "hi"))
        .await
        .expect_err("non-permitted tool call must be rejected");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("not available"),
        "rejection should use the neutral message, got: {msg}"
    );
    assert!(
        !msg.contains("permitted") && !msg.contains("forbidden") && !msg.contains("exists"),
        "rejection must not reveal existence/permission detail, got: {msg}"
    );
}

#[tokio::test]
async fn tool_acl_per_server_wildcard_scopes_to_one_server() {
    // A `<server>__*` grant exposes every tool on that server and nothing
    // from any other — the granularity an operator gets without a live tool
    // list. Two real upstreams; the key is scoped to alpha only.
    let gateway = McpGateway::new([
        ("alpha".to_string(), bridge_to("alpha").await),
        ("beta".to_string(), bridge_to("beta").await),
    ])
    .with_tool_acl(ToolAcl::from_allowed(Some(&["alpha__*".to_string()])));
    let gw_addr = spawn_gateway(gateway).await;
    let client = ()
        .serve(StreamableHttpClientTransport::from_uri(format!(
            "http://{gw_addr}/mcp"
        )))
        .await
        .expect("connect");

    // Every alpha tool is exposed; nothing from beta.
    let tools = client.list_all_tools().await.expect("list tools");
    let names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
    assert!(
        names.iter().all(|n| n.starts_with("alpha__")),
        "per-server wildcard must expose only alpha's tools, got: {names:?}"
    );
    assert!(
        names.contains(&"alpha__echo"),
        "alpha's tool should be visible, got: {names:?}"
    );
    assert!(
        !names.iter().any(|n| n.starts_with("beta__")),
        "beta's tools must stay hidden, got: {names:?}"
    );

    // alpha is callable; beta is rejected by the same ACL (defense-in-depth).
    let allowed = client
        .call_tool(call("alpha__echo", "hi"))
        .await
        .expect("permitted call");
    assert_eq!(first_text(&allowed), "alpha:hi");
    client
        .call_tool(call("beta__echo", "hi"))
        .await
        .expect_err("a tool outside the granted server must be rejected");
}

/// Build a `tools/call` for `name` with a single `text` argument.
fn call(name: &'static str, text: &str) -> CallToolRequestParams {
    let args = serde_json::json!({ "text": text });
    CallToolRequestParams::new(name).with_arguments(args.as_object().unwrap().clone())
}
