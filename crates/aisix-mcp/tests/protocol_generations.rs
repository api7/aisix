//! Two-generation contract tests for the downstream `/mcp` surface after the
//! rmcp 3.x upgrade: the stateless MCP `2026-07-28` revision and the legacy
//! Streamable HTTP generations must BOTH work against the same endpoint, and
//! the version-negotiation behavior pinned by AISIX-Cloud#1144/#1148 must not
//! drift:
//!
//! - `initialize` echoes a supported requested version exactly, and falls
//!   back to `2025-11-25` (the endpoint's historical answer) for anything
//!   else — never to whatever the SDK's `LATEST` alias happens to be.
//! - `server/discover` advertises exactly `SUPPORTED_PROTOCOL_VERSION_NAMES`.
//! - Modern requests are served without any `Mcp-Session-Id`, carry
//!   `resultType: "complete"`, and get the honest do-not-cache hints
//!   (`ttlMs: 0`, `cacheScope: "private"`); legacy responses keep their
//!   historical shape (no `resultType`).
//! - The deliberate `allowed_hosts` opt-out survives the upgrade: a
//!   non-loopback `Host` is served, not 403'd (the AISIX API key is the real
//!   access control on this endpoint).
//!
//! Raw HTTP is used for the wire-shape pins (a real client hides headers and
//! envelope details); rmcp's own modern client covers the end-to-end modern
//! lifecycle.

use std::net::SocketAddr;
use std::sync::Arc;

use aisix_mcp::{streamable_http_service, McpBridge, McpError, McpGateway, McpTool, McpToolResult};
use rmcp::model::{CallToolRequestParams, ClientInfo, ProtocolVersion};
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::{ClientLifecycleMode, ClientServiceExt, ServiceExt};

/// A self-contained upstream: one `echo` tool, no network. The generation
/// tests exercise the DOWNSTREAM protocol surface; a live upstream session
/// would only add noise.
struct StaticEcho;

#[async_trait::async_trait]
impl McpBridge for StaticEcho {
    async fn list_tools(&self) -> Result<Vec<McpTool>, McpError> {
        Ok(vec![McpTool {
            name: "echo".to_string(),
            description: Some("echo back the text argument".to_string()),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": { "text": { "type": "string" } },
            }),
        }])
    }

    async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<McpToolResult, McpError> {
        assert_eq!(name, "echo");
        let text = arguments
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        Ok(McpToolResult {
            content: serde_json::json!([{ "type": "text", "text": text }]),
            structured_content: None,
            is_error: false,
        })
    }
}

async fn spawn_gateway() -> SocketAddr {
    let gateway = McpGateway::new([(
        "alpha".to_string(),
        Arc::new(StaticEcho) as Arc<dyn McpBridge>,
    )]);
    let app = axum::Router::new().nest_service("/mcp", streamable_http_service(gateway, 0));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve");
    });
    addr
}

/// POST one JSON-RPC message; return `(status, headers, parsed body)`.
async fn post_raw(
    addr: SocketAddr,
    extra_headers: &[(&str, &str)],
    body: serde_json::Value,
) -> (
    reqwest::StatusCode,
    reqwest::header::HeaderMap,
    serde_json::Value,
) {
    let client = reqwest::Client::new();
    let mut request = client
        .post(format!("http://{addr}/mcp"))
        .header("content-type", "application/json")
        .header("accept", "application/json, text/event-stream");
    for (name, value) in extra_headers {
        request = request.header(*name, *value);
    }
    let response = request.body(body.to_string()).send().await.expect("send");
    let status = response.status();
    let headers = response.headers().clone();
    let text = response.text().await.expect("read body");
    let parsed = serde_json::from_str(&text).unwrap_or(serde_json::Value::String(text));
    (status, headers, parsed)
}

fn initialize_body(protocol_version: &str) -> serde_json::Value {
    serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": protocol_version,
            "capabilities": {},
            "clientInfo": { "name": "generation-test", "version": "0.0.0" },
        }
    })
}

/// `initialize` echoes a requested version exactly when it is one we
/// support, for BOTH generations' Streamable HTTP clients.
#[tokio::test]
async fn initialize_echoes_supported_versions_exactly() {
    let addr = spawn_gateway().await;
    for version in aisix_mcp::SUPPORTED_PROTOCOL_VERSION_NAMES {
        let (status, headers, body) = post_raw(addr, &[], initialize_body(version)).await;
        assert_eq!(status, 200, "initialize {version}: {body}");
        assert_eq!(
            body["result"]["protocolVersion"], *version,
            "supported version must be echoed exactly: {body}"
        );
        // Stateless serving on every generation: no session is minted.
        assert!(
            headers.get("mcp-session-id").is_none(),
            "no Mcp-Session-Id may be issued ({version})"
        );
    }
}

/// A version we do NOT list falls back to the endpoint's historical answer,
/// `2025-11-25` — pinned so an SDK `LATEST` bump can never silently move it.
#[tokio::test]
async fn initialize_falls_back_to_2025_11_25() {
    let addr = spawn_gateway().await;
    for unsupported in ["2024-11-05", "9999-12-31"] {
        let (status, _headers, body) = post_raw(addr, &[], initialize_body(unsupported)).await;
        assert_eq!(status, 200, "initialize {unsupported}: {body}");
        assert_eq!(
            body["result"]["protocolVersion"], "2025-11-25",
            "unsupported request must fall back to the pinned version: {body}"
        );
    }
}

/// The deliberate DNS-rebinding-allowlist opt-out (`allowed_hosts` cleared)
/// survives the upgrade: a request whose `Host` is the deployment's real DNS
/// name is served, not 403'd. rmcp's DEFAULT config would reject this.
#[tokio::test]
async fn non_loopback_host_is_served() {
    let addr = spawn_gateway().await;
    let (status, _headers, body) = post_raw(
        addr,
        &[("host", "aisix.example.com")],
        initialize_body("2025-11-25"),
    )
    .await;
    assert_eq!(
        status, 200,
        "non-loopback Host must be accepted (API-key auth is the real gate): {body}"
    );
}

/// `server/discover` advertises exactly the supported list — the same list
/// `initialize` negotiates against and the proxy header gate enforces.
#[tokio::test]
async fn discover_advertises_the_supported_list() {
    let addr = spawn_gateway().await;
    let (status, headers, body) = post_raw(
        addr,
        &[
            ("mcp-protocol-version", "2026-07-28"),
            ("mcp-method", "server/discover"),
        ],
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "server/discover",
            "params": {
                "_meta": {
                    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
                    "io.modelcontextprotocol/clientInfo": {
                        "name": "generation-test", "version": "0.0.0"
                    },
                    "io.modelcontextprotocol/clientCapabilities": {},
                }
            }
        }),
    )
    .await;
    assert_eq!(status, 200, "server/discover: {body}");
    let advertised: Vec<&str> = body["result"]["supportedVersions"]
        .as_array()
        .unwrap_or_else(|| panic!("supportedVersions missing: {body}"))
        .iter()
        .filter_map(|v| v.as_str())
        .collect();
    assert_eq!(
        advertised,
        aisix_mcp::SUPPORTED_PROTOCOL_VERSION_NAMES.to_vec(),
        "discover must advertise the pinned list"
    );
    assert!(
        headers.get("mcp-session-id").is_none(),
        "discover is stateless"
    );
}

/// The modern stateless flow, raw: no handshake, per-request metadata, the
/// SEP-2243 mirrored headers. Pins the wire shape a `2026-07-28` client
/// observes: `resultType: "complete"`, the do-not-cache hints on
/// `tools/list`, and no session header anywhere.
#[tokio::test]
async fn modern_stateless_list_and_call_without_handshake() {
    let addr = spawn_gateway().await;
    let meta = serde_json::json!({
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientInfo": {
            "name": "generation-test", "version": "0.0.0"
        },
        "io.modelcontextprotocol/clientCapabilities": {},
    });

    let (status, headers, body) = post_raw(
        addr,
        &[
            ("mcp-protocol-version", "2026-07-28"),
            ("mcp-method", "tools/list"),
        ],
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": { "_meta": meta }
        }),
    )
    .await;
    assert_eq!(status, 200, "modern tools/list: {body}");
    assert!(
        headers.get("mcp-session-id").is_none(),
        "modern tools/list is stateless"
    );
    let result = &body["result"];
    assert_eq!(result["resultType"], "complete", "modern result: {body}");
    assert_eq!(
        result["ttlMs"], 0,
        "the ACL-filtered list must not advertise cacheability: {body}"
    );
    assert_eq!(result["cacheScope"], "private", "caller-specific: {body}");
    let names: Vec<&str> = result["tools"]
        .as_array()
        .unwrap_or_else(|| panic!("tools missing: {body}"))
        .iter()
        .filter_map(|t| t["name"].as_str())
        .collect();
    assert_eq!(names, ["alpha__echo"], "namespaced tool list");

    let (status, _headers, body) = post_raw(
        addr,
        &[
            ("mcp-protocol-version", "2026-07-28"),
            ("mcp-method", "tools/call"),
            ("mcp-name", "alpha__echo"),
        ],
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "alpha__echo",
                "arguments": { "text": "hello-modern" },
                "_meta": meta,
            }
        }),
    )
    .await;
    assert_eq!(status, 200, "modern tools/call: {body}");
    assert_eq!(body["result"]["resultType"], "complete", "{body}");
    assert_eq!(
        body["result"]["content"][0]["text"], "hello-modern",
        "tool routed and executed: {body}"
    );
}

/// SEP-2243: a mirrored header that contradicts the body is rejected before
/// dispatch — the mirror is only trustworthy because mismatches are fatal.
#[tokio::test]
async fn modern_header_body_mismatch_is_rejected() {
    let addr = spawn_gateway().await;
    let (status, _headers, body) = post_raw(
        addr,
        &[
            ("mcp-protocol-version", "2026-07-28"),
            ("mcp-method", "tools/call"),
            ("mcp-name", "alpha__echo"),
        ],
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/list",
            "params": {
                "_meta": {
                    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
                    "io.modelcontextprotocol/clientInfo": {
                        "name": "generation-test", "version": "0.0.0"
                    },
                    "io.modelcontextprotocol/clientCapabilities": {},
                }
            }
        }),
    )
    .await;
    assert_eq!(
        status, 400,
        "Mcp-Method header contradicting the body must be rejected: {body}"
    );
}

/// Legacy responses keep their historical wire shape across EVERY legacy
/// generation this endpoint serves, on both `tools/list` and `tools/call`:
/// no `resultType` discriminator, and no `2026-07-28` cache-hint fields
/// (rmcp strips only `resultType` for legacy peers, so the gateway
/// version-gates the hints itself).
#[tokio::test]
async fn legacy_results_keep_their_wire_shape() {
    let addr = spawn_gateway().await;
    let client = ()
        .serve(StreamableHttpClientTransport::from_uri(format!(
            "http://{addr}/mcp"
        )))
        .await
        .expect("legacy client connects");
    let tools = client.list_all_tools().await.expect("list tools");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name.as_ref(), "alpha__echo");

    for version in ["2025-03-26", "2025-06-18", "2025-11-25"] {
        let (status, _headers, body) = post_raw(
            addr,
            &[("mcp-protocol-version", version)],
            serde_json::json!({
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/list",
                "params": {}
            }),
        )
        .await;
        assert_eq!(status, 200, "legacy {version} tools/list: {body}");
        for field in ["resultType", "ttlMs", "cacheScope"] {
            assert!(
                body["result"].get(field).is_none(),
                "no {field} on a legacy {version} tools/list response: {body}"
            );
        }

        let (status, _headers, body) = post_raw(
            addr,
            &[("mcp-protocol-version", version)],
            serde_json::json!({
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": { "name": "alpha__echo", "arguments": { "text": "legacy" } }
            }),
        )
        .await;
        assert_eq!(status, 200, "legacy {version} tools/call: {body}");
        assert_eq!(
            body["result"]["content"][0]["text"], "legacy",
            "legacy {version} call executes: {body}"
        );
        assert!(
            body["result"].get("resultType").is_none(),
            "no resultType on a legacy {version} tools/call response: {body}"
        );
    }
}

/// The end-to-end modern lifecycle through rmcp's own client: `Discover`
/// startup (no initialize handshake), then list + call.
#[tokio::test]
async fn modern_rmcp_client_lifecycle_works_end_to_end() {
    let addr = spawn_gateway().await;
    let client = ClientInfo::default()
        .serve_with_lifecycle(
            StreamableHttpClientTransport::from_uri(format!("http://{addr}/mcp")),
            ClientLifecycleMode::Discover {
                preferred_versions: vec![ProtocolVersion::V_2026_07_28],
            },
        )
        .await
        .expect("modern client startup via server/discover");

    let tools = client.list_all_tools().await.expect("list tools");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name.as_ref(), "alpha__echo");

    let mut params = CallToolRequestParams::new("alpha__echo".to_string());
    params = params.with_arguments(
        serde_json::json!({ "text": "modern-lifecycle" })
            .as_object()
            .expect("object")
            .clone(),
    );
    let result = client.call_tool(params).await.expect("call tool");
    let content = serde_json::to_value(&result.content).expect("encode");
    assert_eq!(content[0]["text"], "modern-lifecycle");
}

/// A real MCP server for the full-chain test below: one `echo` tool served
/// over actual Streamable HTTP.
#[derive(Clone, Default)]
struct UpstreamEcho;

impl rmcp::ServerHandler for UpstreamEcho {
    async fn list_tools(
        &self,
        _request: Option<rmcp::model::PaginatedRequestParams>,
        _context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<rmcp::model::ListToolsResult, rmcp::ErrorData> {
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "text": { "type": "string" } },
        });
        let schema_obj = schema.as_object().expect("schema is an object").clone();
        let tool = rmcp::model::Tool::new("echo", "Echo the text argument", schema_obj);
        Ok(rmcp::model::ListToolsResult::with_all_items(vec![tool]))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<rmcp::model::CallToolResponse, rmcp::ErrorData> {
        let text = request
            .arguments
            .as_ref()
            .and_then(|m| m.get("text"))
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        Ok(
            rmcp::model::CallToolResult::success(vec![rmcp::model::ContentBlock::text(text)])
                .into(),
        )
    }

    fn get_info(&self) -> rmcp::model::ServerInfo {
        let mut info = rmcp::model::ServerInfo::default();
        info.capabilities = rmcp::model::ServerCapabilities::builder()
            .enable_tools()
            .build();
        info
    }
}

/// The full shipped chain under a modern client: `Discover`-lifecycle
/// client → gateway → `EphemeralBridge` (real connect/list/call per
/// operation, legacy handshake) → real upstream MCP server over HTTP. Pins
/// that the modern downstream surface composes with the production bridge —
/// not just with an in-process stub — including the cross-generation seam
/// (modern downstream, legacy upstream session).
#[tokio::test]
async fn modern_client_through_real_bridge_and_upstream() {
    use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
    use rmcp::transport::streamable_http_server::{
        StreamableHttpServerConfig, StreamableHttpService,
    };

    // Real upstream server on an ephemeral port.
    let upstream_service = StreamableHttpService::new(
        move || Ok(UpstreamEcho),
        Arc::new(LocalSessionManager::default()),
        StreamableHttpServerConfig::default(),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind upstream port");
    let upstream_addr = listener.local_addr().expect("upstream addr");
    tokio::spawn(async move {
        axum::serve(
            listener,
            axum::Router::new().nest_service("/mcp", upstream_service),
        )
        .await
        .expect("serve upstream");
    });

    // Gateway fronting it through the production bridge type.
    let bridge = aisix_mcp::EphemeralBridge::new(aisix_mcp::McpUpstream::new(format!(
        "http://{upstream_addr}/mcp"
    )));
    let gateway = McpGateway::new([("alpha".to_string(), Arc::new(bridge) as Arc<dyn McpBridge>)]);
    let app = axum::Router::new().nest_service("/mcp", streamable_http_service(gateway, 0));
    let gw_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind gateway port");
    let gw_addr = gw_listener.local_addr().expect("gateway addr");
    tokio::spawn(async move {
        axum::serve(gw_listener, app).await.expect("serve gateway");
    });

    let client = ClientInfo::default()
        .serve_with_lifecycle(
            StreamableHttpClientTransport::from_uri(format!("http://{gw_addr}/mcp")),
            ClientLifecycleMode::Discover {
                preferred_versions: vec![ProtocolVersion::V_2026_07_28],
            },
        )
        .await
        .expect("modern client startup against the real chain");

    let tools = client.list_all_tools().await.expect("list tools");
    let names: Vec<&str> = tools.iter().map(|t| t.name.as_ref()).collect();
    assert_eq!(names, ["alpha__echo"], "real upstream tool aggregated");

    let mut params = CallToolRequestParams::new("alpha__echo".to_string());
    params = params.with_arguments(
        serde_json::json!({ "text": "through-the-real-chain" })
            .as_object()
            .expect("object")
            .clone(),
    );
    let result = client.call_tool(params).await.expect("call tool");
    let content = serde_json::to_value(&result.content).expect("encode");
    assert_eq!(content[0]["text"], "through-the-real-chain");
}
