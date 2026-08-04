//! End-to-end test of the **scoped** gateway (`/mcp/{server}`): AISIX as an
//! MCP server fronting exactly one registered upstream, serving its tools
//! under their original (un-namespaced) names.
//!
//! Topology, all real Streamable HTTP over ephemeral ports (no mock
//! transport):
//!
//!   downstream rmcp client  ──►  McpGateway (scoped "alpha")  ──►  upstream "alpha" (echo)
//!
//! Pins: `initialize` reports the scoped server's name; `tools/list` strips
//! the `alpha__` prefix; `tools/call` accepts both the bare and the
//! namespaced spelling; a foreign prefix stays pinned to the scoped upstream;
//! ACL patterns keep their namespaced meaning; unknown/disabled servers
//! resolve to no gateway at all.

use std::net::SocketAddr;
use std::sync::Arc;

use aisix_core::{AisixSnapshot, McpServer, ResourceEntry};
use aisix_mcp::{streamable_http_service, McpGateway, ToolAcl};
use rmcp::model::{
    CallToolRequestParams, CallToolResult, Content, ErrorData, ListToolsResult,
    PaginatedRequestParams, ServerCapabilities, ServerInfo, Tool,
};
use rmcp::service::RequestContext;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::{RoleServer, ServerHandler, ServiceExt};

/// A real upstream MCP server exposing one echo tool under `tool_name`,
/// prefixing its reply with `label` so routing is observable.
#[derive(Clone)]
struct LabeledEcho {
    label: &'static str,
    tool_name: &'static str,
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
            self.tool_name,
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
        if request.name != self.tool_name {
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
async fn spawn_upstream(label: &'static str, tool_name: &'static str) -> SocketAddr {
    let service = StreamableHttpService::new(
        move || Ok(LabeledEcho { label, tool_name }),
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

/// A snapshot with one enabled server `alpha` at `addr` and one disabled
/// server `dark`.
fn snapshot_with_alpha(addr: &SocketAddr) -> AisixSnapshot {
    let snapshot = AisixSnapshot::new();
    snapshot
        .mcp_servers
        .insert(mcp_entry("e1", "alpha", addr, true));
    snapshot
        .mcp_servers
        .insert(mcp_entry("e2", "dark", addr, false));
    snapshot
}

async fn connect(gw_addr: SocketAddr) -> rmcp::service::RunningService<rmcp::RoleClient, ()> {
    ().serve(StreamableHttpClientTransport::from_uri(format!(
        "http://{gw_addr}/mcp"
    )))
    .await
    .expect("downstream client connects to gateway")
}

fn call(name: &'static str, text: &str) -> CallToolRequestParams {
    let args = serde_json::json!({ "text": text });
    CallToolRequestParams::new(name).with_arguments(args.as_object().unwrap().clone())
}

/// Decode the first text content block of a tool result.
fn first_text(result: &CallToolResult) -> String {
    let value = serde_json::to_value(&result.content).expect("encode content");
    value[0]["text"].as_str().unwrap_or_default().to_string()
}

#[tokio::test]
async fn scoped_serves_original_names_and_accepts_both_call_forms() {
    let upstream = spawn_upstream("alpha", "echo").await;
    let snapshot = snapshot_with_alpha(&upstream);
    let gateway =
        McpGateway::from_snapshot_scoped(&snapshot, "alpha").expect("alpha is registered");
    let gw_addr = spawn_gateway(gateway).await;
    let client = connect(gw_addr).await;

    // `initialize` presents the scoped server, not the aggregate.
    let info = client.peer_info().expect("initialize completed");
    assert_eq!(info.server_info.name, "alpha");

    // tools/list carries the upstream's original names — no `alpha__` prefix.
    let tools = client.list_all_tools().await.expect("list tools");
    let names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
    assert_eq!(names, vec!["echo"], "original name only: {names:?}");

    // The bare (original) spelling calls through…
    let bare = client
        .call_tool(call("echo", "hi"))
        .await
        .expect("bare call");
    assert_eq!(first_text(&bare), "alpha:hi");

    // …and the namespaced spelling still works, so an aggregated-endpoint
    // client pointed at the scoped URL keeps functioning.
    let prefixed = client
        .call_tool(call("alpha__echo", "hi"))
        .await
        .expect("namespaced call");
    assert_eq!(first_text(&prefixed), "alpha:hi");
}

#[tokio::test]
async fn scoped_missing_or_disabled_server_resolves_to_none() {
    let upstream = spawn_upstream("alpha", "echo").await;
    let snapshot = snapshot_with_alpha(&upstream);

    assert!(
        McpGateway::from_snapshot_scoped(&snapshot, "ghost").is_none(),
        "unregistered server must not resolve"
    );
    assert!(
        McpGateway::from_snapshot_scoped(&snapshot, "dark").is_none(),
        "disabled server must resolve like a missing one"
    );
}

#[tokio::test]
async fn scoped_acl_keeps_namespaced_meaning() {
    let upstream = spawn_upstream("alpha", "echo").await;
    let snapshot = snapshot_with_alpha(&upstream);

    // A grant written against the aggregated form covers the scoped endpoint.
    let gateway = McpGateway::from_snapshot_scoped(&snapshot, "alpha")
        .expect("alpha is registered")
        .with_tool_acl(ToolAcl::from_allowed(Some(&["alpha__echo".to_string()])));
    let client = connect(spawn_gateway(gateway).await).await;
    let tools = client.list_all_tools().await.expect("list tools");
    assert_eq!(tools.len(), 1, "granted tool listed (bare)");
    assert_eq!(tools[0].name.as_ref(), "echo");
    let ok = client.call_tool(call("echo", "hi")).await.expect("allowed");
    assert_eq!(first_text(&ok), "alpha:hi");

    // A grant for a different server admits nothing here — the bare name is
    // re-namespaced before the check, so the scoped surface cannot widen a
    // key's grant.
    let gateway = McpGateway::from_snapshot_scoped(&snapshot, "alpha")
        .expect("alpha is registered")
        .with_tool_acl(ToolAcl::from_allowed(Some(&["beta__*".to_string()])));
    let client = connect(spawn_gateway(gateway).await).await;
    let tools = client.list_all_tools().await.expect("list tools");
    assert!(tools.is_empty(), "foreign grant must expose nothing");
    assert!(
        client.call_tool(call("echo", "hi")).await.is_err(),
        "foreign grant must not admit a bare-name call"
    );
}

#[tokio::test]
async fn scoped_registered_foreign_prefix_fails_closed() {
    // The upstream serves a tool literally named `beta__echo`, and `beta` IS
    // a registered, enabled server. On alpha's scoped gateway that spelling
    // must fail closed — a cross-server mistake, never silently served as a
    // bare name (which this upstream would happily answer).
    let upstream = spawn_upstream("alpha", "beta__echo").await;
    let snapshot = snapshot_with_alpha(&upstream);
    snapshot
        .mcp_servers
        .insert(mcp_entry("e3", "beta", &upstream, true));
    let gateway =
        McpGateway::from_snapshot_scoped(&snapshot, "alpha").expect("alpha is registered");
    let client = connect(spawn_gateway(gateway).await).await;

    // tools/list keeps the colliding literal name namespaced: advertising
    // the bare `beta__echo` would advertise a spelling `tools/call` rejects.
    let tools = client.list_all_tools().await.expect("list tools");
    let names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
    assert_eq!(names, vec!["alpha__beta__echo"], "list must round-trip");

    assert!(
        client.call_tool(call("beta__echo", "x")).await.is_err(),
        "a registered foreign prefix must fail closed, never route or serve"
    );

    // The advertised spelling reaches the literal tool: `alpha__beta__echo`
    // strips alpha's own prefix first.
    let escaped = client
        .call_tool(call("alpha__beta__echo", "hi"))
        .await
        .expect("the namespaced spelling reaches the literal tool");
    assert_eq!(first_text(&escaped), "alpha:hi");
}

#[tokio::test]
async fn scoped_server_name_ending_in_underscore_namespaces_cleanly() {
    // `data_` is a legal server name (only `__` inside a name is rejected).
    // Prefix parsing is whole-string based, not first-separator based, so
    // the namespaced spelling `data___query` (= `data_` + `__` + `query`)
    // resolves to `query` even while a server named `data` also exists.
    let upstream = spawn_upstream("data_", "query").await;
    let snapshot = AisixSnapshot::new();
    snapshot
        .mcp_servers
        .insert(mcp_entry("e1", "data_", &upstream, true));
    snapshot
        .mcp_servers
        .insert(mcp_entry("e2", "data", &upstream, true));
    let gateway =
        McpGateway::from_snapshot_scoped(&snapshot, "data_").expect("data_ is registered");
    let client = connect(spawn_gateway(gateway).await).await;

    let tools = client.list_all_tools().await.expect("list tools");
    let names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
    assert_eq!(names, vec!["query"]);

    let bare = client.call_tool(call("query", "hi")).await.expect("bare");
    assert_eq!(first_text(&bare), "data_:hi");
    let namespaced = client
        .call_tool(call("data___query", "hi"))
        .await
        .expect("namespaced spelling of a trailing-underscore server");
    assert_eq!(first_text(&namespaced), "data_:hi");
}

#[tokio::test]
async fn scoped_unregistered_prefix_is_a_bare_name() {
    // `ghost` is not a registered server, so `ghost__echo` is just a tool
    // name that happens to contain the separator — it must reach the scoped
    // upstream verbatim (which serves exactly that name here).
    let upstream = spawn_upstream("alpha", "ghost__echo").await;
    let snapshot = snapshot_with_alpha(&upstream);
    let gateway =
        McpGateway::from_snapshot_scoped(&snapshot, "alpha").expect("alpha is registered");
    let client = connect(spawn_gateway(gateway).await).await;

    let served = client
        .call_tool(call("ghost__echo", "hi"))
        .await
        .expect("an unregistered prefix stays a bare tool name");
    assert_eq!(first_text(&served), "alpha:hi");
}

#[tokio::test]
async fn scoped_upstream_tool_spelled_like_the_prefix_keeps_precedence() {
    // Pathological upstream: a tool literally named `alpha__echo` on server
    // `alpha`. Prefix-stripping takes precedence (documented in `call_tool`),
    // so the bare spelling would not round-trip — tools/list therefore keeps
    // the namespaced `alpha__alpha__echo`, which IS the callable spelling;
    // `alpha__echo` reduces to `echo`, which does not exist.
    let upstream = spawn_upstream("alpha", "alpha__echo").await;
    let snapshot = snapshot_with_alpha(&upstream);
    let gateway =
        McpGateway::from_snapshot_scoped(&snapshot, "alpha").expect("alpha is registered");
    let client = connect(spawn_gateway(gateway).await).await;

    let tools = client.list_all_tools().await.expect("list tools");
    let names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
    assert_eq!(
        names,
        vec!["alpha__alpha__echo"],
        "a non-round-tripping name stays namespaced: {names:?}"
    );

    let advertised = client
        .call_tool(call("alpha__alpha__echo", "hi"))
        .await
        .expect("the advertised spelling reaches the literal tool");
    assert_eq!(first_text(&advertised), "alpha:hi");

    assert!(
        client.call_tool(call("alpha__echo", "x")).await.is_err(),
        "the single-prefixed spelling reduces to `echo`, which must error"
    );
}
