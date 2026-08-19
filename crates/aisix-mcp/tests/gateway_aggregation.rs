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
    McpProtocol, McpTool, McpToolResult, McpUpstream, RmcpBridge, ToolAcl,
};
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, ErrorData,
    ListToolsResult, PaginatedRequestParams, ServerCapabilities, ServerInfo, Tool,
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
    ) -> Result<CallToolResponse, ErrorData> {
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
            return Ok(CallToolResult::error(vec![ContentBlock::text("boom")]).into());
        }
        Ok(
            CallToolResult::success(vec![ContentBlock::text(format!("{}:{text}", self.label))])
                .into(),
        )
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
    serve(axum::Router::new().nest_service("/mcp", streamable_http_service(gateway, 0))).await
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

/// The configured `protocol_version` — the wire value a control plane
/// writes — selects the bridge lifecycle, and its absence keeps the legacy
/// handshake. Pins the deserialize → upstream mapping itself, so reverting
/// the mapping (not just the lifecycle switch) fails a test.
#[test]
fn upstream_from_mcp_server_maps_protocol_version() {
    let modern: McpServer = serde_json::from_value(serde_json::json!({
        "display_name": "m",
        "url": "https://api.example.com/mcp",
        "protocol_version": "2026-07-28"
    }))
    .unwrap();
    assert_eq!(
        upstream_from_mcp_server(&modern).protocol,
        McpProtocol::V20260728
    );

    let dflt: McpServer = serde_json::from_value(serde_json::json!({
        "display_name": "d", "url": "https://api.example.com/mcp"
    }))
    .unwrap();
    assert_eq!(
        upstream_from_mcp_server(&dflt).protocol,
        McpProtocol::LegacyHandshake
    );

    // An unknown revision is a hard deserialization error, not a silent
    // fallback to some default lifecycle.
    let unknown = serde_json::from_value::<McpServer>(serde_json::json!({
        "display_name": "u",
        "url": "https://api.example.com/mcp",
        "protocol_version": "2099-01-01"
    }));
    assert!(unknown.is_err(), "unknown revisions must be rejected");
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
    // No list / empty → deny all.
    assert!(!ToolAcl::from_allowed(None).permits("github__create_issue"));
    assert!(!ToolAcl::from_allowed(Some(&[])).permits("github__create_issue"));
    // Wildcard → allow all.
    let all = ToolAcl::from_allowed(Some(&["*".to_string()]));
    assert!(all.permits("github__create_issue"));
    assert!(all.permits("anything__at_all"));
    // Exact set.
    let exact = ToolAcl::from_allowed(Some(&["a__b".to_string()]));
    assert!(exact.permits("a__b"));
    assert!(!exact.permits("a__c"));
    // A per-server wildcard is a scoped grant, not allow-all — only a bare
    // `"*"` opens everything.
    let scoped = ToolAcl::from_allowed(Some(&["github__*".to_string()]));
    assert!(scoped.permits("github__create_issue"));
    assert!(!scoped.permits("slack__post_message"));
}

// ---- layered effective-ACL resolution (`ToolAcl::resolve`) ----

/// Build a caller key from raw JSON (the etcd document shape).
fn acl_key(json: serde_json::Value) -> aisix_core::models::ApiKey {
    serde_json::from_value(json).unwrap()
}

/// Build a snapshot holding the given `mcp_policies` rows, each `(id, doc)`.
fn policy_snapshot(rows: &[(&str, serde_json::Value)]) -> AisixSnapshot {
    let snapshot = AisixSnapshot::new();
    for (id, doc) in rows {
        let policy: aisix_core::models::McpPolicy = serde_json::from_value(doc.clone()).unwrap();
        snapshot
            .mcp_policies
            .insert(ResourceEntry::new(*id, policy, 1));
    }
    snapshot
}

fn resolve_now(snapshot: &AisixSnapshot, key: &aisix_core::models::ApiKey) -> ToolAcl {
    ToolAcl::resolve(snapshot, key)
}

#[test]
fn resolve_grants_nothing_when_no_layer_is_configured() {
    // Deny-by-default: an empty conjunction must not mean "everything".
    let snapshot = AisixSnapshot::new();
    let key = acl_key(serde_json::json!({"key_hash":"h","allowed_models":[]}));
    assert!(!resolve_now(&snapshot, &key).permits("github__create_issue"));
}

#[test]
fn resolve_key_layer_alone_governs_when_no_policy_exists() {
    let snapshot = AisixSnapshot::new();
    let key = acl_key(serde_json::json!({
        "key_hash":"h","allowed_models":[],"mcp_access":{"allow":["github__*"]}
    }));
    let acl = resolve_now(&snapshot, &key);
    assert!(acl.permits("github__create_issue"));
    assert!(!acl.permits("slack__post_message"));
}

#[test]
fn resolve_env_layer_alone_governs_a_key_with_no_block() {
    // A key that adds no constraint of its own takes the env layer as-is.
    let snapshot = policy_snapshot(&[(
        "p-env",
        serde_json::json!({
            "scope":"env","allow":["github__*"],"deny":["github__delete_repo"]
        }),
    )]);
    let key = acl_key(serde_json::json!({"key_hash":"h","allowed_models":[]}));
    let acl = resolve_now(&snapshot, &key);
    assert!(acl.permits("github__create_issue"));
    assert!(!acl.permits("github__delete_repo"), "deny beats allow");
    assert!(!acl.permits("slack__post_message"));
}

#[test]
fn resolve_empty_allow_list_on_any_layer_grants_nothing() {
    // The explicit "block everything" spelling, replacing the removed
    // `mode: none` / `mode: deny`.
    let wide_env =
        policy_snapshot(&[("p-env", serde_json::json!({"scope":"env","allow":["*"]}))]);
    let blocked_key = acl_key(serde_json::json!({
        "key_hash":"h","allowed_models":[],"mcp_access":{"allow":[]}
    }));
    assert!(!resolve_now(&wide_env, &blocked_key).permits("github__create_issue"));

    let blocked_env = policy_snapshot(&[("p-env", serde_json::json!({"scope":"env","allow":[]}))]);
    let wide_key = acl_key(serde_json::json!({
        "key_hash":"h","allowed_models":[],"mcp_access":{"allow":["*"]}
    }));
    assert!(!resolve_now(&blocked_env, &wide_key).permits("github__create_issue"));
}

#[test]
fn resolve_key_layer_narrows_but_never_widens_the_policy_layers() {
    let snapshot = policy_snapshot(&[("p-env", serde_json::json!({"scope":"env","allow":["*"]}))]);
    let key = acl_key(serde_json::json!({
        "key_hash":"h","allowed_models":[],
        "mcp_access":{"allow":["github__*"],"deny":["github__delete_repo"]}
    }));
    let acl = resolve_now(&snapshot, &key);
    assert!(acl.permits("github__create_issue"));
    assert!(!acl.permits("slack__post_message"), "the key narrows `*`");
    assert!(!acl.permits("github__delete_repo"), "key deny subtracts");

    // The key cannot reach outside a narrower env layer either: the
    // intersection of [slack__*] and [github__*] is empty.
    let narrow_env = policy_snapshot(&[(
        "p-env",
        serde_json::json!({"scope":"env","allow":["slack__*"]}),
    )]);
    let acl = resolve_now(&narrow_env, &key);
    assert!(!acl.permits("github__create_issue"));
    assert!(!acl.permits("slack__post_message"));
}

#[test]
fn resolve_team_layer_intersects_with_env_rather_than_replacing_it() {
    let snapshot = policy_snapshot(&[
        (
            "p-env",
            serde_json::json!({"scope":"env","allow":["github__*","slack__*"]}),
        ),
        (
            "p-team",
            serde_json::json!({
                "scope":"team","scope_ref":"team-1","allow":["github__*","jira__*"]
            }),
        ),
    ]);

    // A key in team-1 gets exactly the overlap of the two layers: the team
    // layer narrows the environment and cannot add `jira__*` on its own.
    let member = acl_key(serde_json::json!({
        "key_hash":"h","allowed_models":[],"team_id":"team-1"
    }));
    let acl = resolve_now(&snapshot, &member);
    assert!(acl.permits("github__create_issue"));
    assert!(
        !acl.permits("slack__post_message"),
        "the team layer narrows the env layer"
    );
    assert!(
        !acl.permits("jira__create_ticket"),
        "a team layer must never widen beyond the env layer"
    );

    // A key outside any team sees the env layer alone.
    let unbound = acl_key(serde_json::json!({"key_hash":"h","allowed_models":[]}));
    let acl = resolve_now(&snapshot, &unbound);
    assert!(acl.permits("slack__post_message"));
    assert!(acl.permits("github__create_issue"));

    // A key in a team WITHOUT its own policy also sees the env layer alone.
    let other_team = acl_key(serde_json::json!({
        "key_hash":"h","allowed_models":[],"team_id":"team-2"
    }));
    assert!(resolve_now(&snapshot, &other_team).permits("slack__post_message"));
}

#[test]
fn resolve_all_three_layers_intersect() {
    let snapshot = policy_snapshot(&[
        (
            "p-env",
            serde_json::json!({"scope":"env","allow":["*"],"deny":["github__delete_repo"]}),
        ),
        (
            "p-team",
            serde_json::json!({"scope":"team","scope_ref":"team-1","allow":["github__*"]}),
        ),
    ]);
    let key = acl_key(serde_json::json!({
        "key_hash":"h","allowed_models":[],"team_id":"team-1",
        "mcp_access":{"allow":["github__create_issue","github__delete_repo"]}
    }));
    let acl = resolve_now(&snapshot, &key);
    assert!(acl.permits("github__create_issue"));
    assert!(
        !acl.permits("github__list_issues"),
        "the key layer narrows the team layer"
    );
    assert!(
        !acl.permits("github__delete_repo"),
        "an env deny survives allows on every other layer"
    );
}

#[test]
fn resolve_env_deny_survives_a_broader_team_layer() {
    // Deny patterns are a global union: an environment-level deny holds
    // however wide the team and key layers are.
    let snapshot = policy_snapshot(&[
        (
            "p-env",
            serde_json::json!({
                "scope":"env","allow":["*"],"deny":["github__delete_repo"]
            }),
        ),
        (
            "p-team",
            serde_json::json!({"scope":"team","scope_ref":"team-1","allow":["*"]}),
        ),
    ]);
    let key = acl_key(serde_json::json!({
        "key_hash":"h","allowed_models":[],"team_id":"team-1"
    }));
    let acl = resolve_now(&snapshot, &key);
    assert!(acl.permits("github__create_issue"));
    assert!(
        !acl.permits("github__delete_repo"),
        "an env-level deny must survive a wide-open team layer"
    );
}

#[test]
fn resolve_deny_only_layer_is_spelled_with_a_wildcard_allow() {
    // A layer that only means to subtract writes `allow: ["*"]`; forgetting
    // it is a schema error rather than a silent lockout (see
    // `mcp_policy_requires_an_explicit_allow_side` in aisix-core).
    let snapshot = policy_snapshot(&[(
        "p-env",
        serde_json::json!({"scope":"env","allow":["*"],"deny":["github__delete_repo"]}),
    )]);
    let key = acl_key(serde_json::json!({"key_hash":"h","allowed_models":[]}));
    let acl = resolve_now(&snapshot, &key);
    assert!(acl.permits("slack__post_message"));
    assert!(!acl.permits("github__delete_repo"));
}

#[test]
fn resolve_ignores_disabled_policies() {
    // Disabled team policy → the env layer is the only policy layer left.
    let snapshot = policy_snapshot(&[
        (
            "p-env",
            serde_json::json!({"scope":"env","allow":["slack__*"]}),
        ),
        (
            "p-team",
            serde_json::json!({
                "scope":"team","scope_ref":"team-1","allow":["github__*"],"enabled":false
            }),
        ),
    ]);
    let key = acl_key(serde_json::json!({
        "key_hash":"h","allowed_models":[],"team_id":"team-1"
    }));
    let acl = resolve_now(&snapshot, &key);
    assert!(acl.permits("slack__post_message"));
    assert!(!acl.permits("github__create_issue"));

    // Disabled env policy → neither its grant nor its deny applies, leaving
    // the key layer as the only layer.
    let snapshot = policy_snapshot(&[(
        "p-env",
        serde_json::json!({
            "scope":"env","allow":["*"],"deny":["slack__*"],"enabled":false
        }),
    )]);
    let scoped = acl_key(serde_json::json!({
        "key_hash":"h","allowed_models":[],"mcp_access":{"allow":["slack__*"]}
    }));
    assert!(
        !resolve_now(&snapshot, &key).permits("anything__at_all"),
        "with every policy disabled and no key block, nothing is granted"
    );
    assert!(
        resolve_now(&snapshot, &scoped).permits("slack__post_message"),
        "a disabled policy's deny must stop applying"
    );
}

#[test]
fn resolve_duplicate_scope_rows_pick_lowest_id() {
    // The writer enforces one row per scope; if duplicates ever appear the
    // outcome must at least be deterministic — lowest id governs.
    let snapshot = policy_snapshot(&[
        (
            "p-b",
            serde_json::json!({"scope":"env","allow":["beta__*"]}),
        ),
        (
            "p-a",
            serde_json::json!({"scope":"env","allow":["alpha__*"]}),
        ),
    ]);
    let key = acl_key(serde_json::json!({"key_hash":"h","allowed_models":[]}));
    let acl = resolve_now(&snapshot, &key);
    assert!(acl.permits("alpha__echo"));
    assert!(!acl.permits("beta__echo"));
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

#[test]
fn resolve_duplicate_scope_rows_deny_union_survives_tie_break() {
    // Grant side: lowest id governs. Deny side: the union of every
    // enabled applicable row — a deny must never vanish because its row
    // lost the duplicate tie-break.
    let snapshot = policy_snapshot(&[
        (
            "p-b",
            serde_json::json!({"scope":"env","allow":["*"],"deny":["beta__echo"]}),
        ),
        ("p-a", serde_json::json!({"scope":"env","allow":["*"]})),
    ]);
    let key = acl_key(serde_json::json!({"key_hash":"h","allowed_models":[]}));
    let acl = resolve_now(&snapshot, &key);
    assert!(acl.permits("alpha__echo"));
    assert!(
        !acl.permits("beta__echo"),
        "the losing duplicate's deny must still subtract"
    );
}

#[test]
fn resolve_key_block_without_an_allow_side_grants_nothing() {
    // A key block is the key's whole allow side; an empty one is the empty
    // intersection, whatever the policy layers grant. (The write path
    // requires the field, so this only guards the runtime loader.)
    let snapshot = policy_snapshot(&[("p-env", serde_json::json!({"scope":"env","allow":["*"]}))]);
    let key = acl_key(serde_json::json!({
        "key_hash":"h","allowed_models":[],"mcp_access":{"allow":[]}
    }));
    assert!(!resolve_now(&snapshot, &key).permits("alpha__echo"));
}

#[test]
fn resolve_team_row_without_scope_ref_matches_no_key() {
    // The schema gate requires scope_ref on team rows; if a malformed
    // row ever reaches the table anyway (this insert bypasses the
    // loader), it must neither govern any key nor shadow the env
    // default.
    let snapshot = policy_snapshot(&[
        (
            "p-env",
            serde_json::json!({"scope":"env","allow":["alpha__*"]}),
        ),
        ("p-bad", serde_json::json!({"scope":"team","allow":["*"]})),
    ]);
    let teamed = acl_key(serde_json::json!({
        "key_hash":"h","allowed_models":[],"team_id":"team-1"
    }));
    let acl = resolve_now(&snapshot, &teamed);
    assert!(
        acl.permits("alpha__echo"),
        "env default must keep governing"
    );
    assert!(
        !acl.permits("beta__echo"),
        "a scope_ref-less team row must not apply to any key"
    );
}

#[tokio::test]
async fn policy_resolved_acl_filters_list_and_rejects_calls() {
    // Full wiring: an env policy granting alpha only and a key that adds no
    // constraint of its own. beta stays hidden from tools/list and rejected
    // on tools/call.
    let snapshot = policy_snapshot(&[(
        "p-env",
        serde_json::json!({"scope":"env","allow":["alpha__*"]}),
    )]);
    let key = acl_key(serde_json::json!({"key_hash":"h","allowed_models":[]}));
    let acl = resolve_now(&snapshot, &key);

    let gateway = McpGateway::new([
        ("alpha".to_string(), bridge_to("alpha").await),
        ("beta".to_string(), bridge_to("beta").await),
    ])
    .with_tool_acl(acl);
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
        names,
        vec!["alpha__echo"],
        "policy-resolved ACL must hide non-granted tools"
    );

    let allowed = client
        .call_tool(call("alpha__echo", "hi"))
        .await
        .expect("granted call");
    assert_eq!(first_text(&allowed), "alpha:hi");

    let err = client
        .call_tool(call("beta__echo", "hi"))
        .await
        .expect_err("a non-granted tool call must be rejected");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("not available"),
        "rejection should use the neutral message, got: {msg}"
    );
}

/// Build a `tools/call` for `name` with a single `text` argument.
fn call(name: &'static str, text: &str) -> CallToolRequestParams {
    let args = serde_json::json!({ "text": text });
    CallToolRequestParams::new(name).with_arguments(args.as_object().unwrap().clone())
}
