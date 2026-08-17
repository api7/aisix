//! Serve the MCP gateway on a local port so the OFFICIAL MCP conformance
//! suite can be run against the exact `/mcp/{server}` protocol surface the
//! data plane ships (AISIX-Cloud#1144 acceptance):
//!
//! ```text
//! cargo run -p aisix-mcp --example conformance_server
//! # optional: bind a different address (default 127.0.0.1:3111)
//! cargo run -p aisix-mcp --example conformance_server -- 127.0.0.1:4000
//! npx -y @modelcontextprotocol/conformance server --url http://127.0.0.1:3111/mcp
//! ```
//!
//! The chain is the production one end to end: the conformance client talks
//! to a SCOPED [`McpGateway`] (original tool names, as `/mcp/{server}`
//! serves them), which reaches an in-process upstream MCP server through the
//! real `EphemeralBridge` — connect, list, call, disconnect per operation.
//! The upstream implements the tool set the suite's `tools-call-*` scenarios
//! prescribe (`test_simple_text`, `test_image_content`, …), each returning
//! the exact content the scenario checks.
//!
//! Expected NON-passes, all deliberate surface decisions rather than bugs:
//! prompts/resources/completions scenarios (tools-only gateway),
//! sampling / elicitation / progress / logging scenarios (server-initiated
//! frames a cross-upstream aggregator does not relay; `logging/setLevel` is
//! also deleted by the 2026-07-28 revision), and the `Host`-allowlist half
//! of `dns-rebinding-protection` (deliberately disabled — the endpoint is
//! server-to-server and API-key-gated in production; see
//! `streamable_http_service`).
//!
//! Auth, quota, and guardrails live in the proxy layer above this service
//! and are deliberately absent here — the suite probes the protocol, not
//! AISIX governance.

use std::sync::Arc;

use aisix_mcp::{streamable_http_service, McpGateway};
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, ErrorData,
    ListToolsResult, PaginatedRequestParams, ServerCapabilities, ServerInfo, Tool,
};
use rmcp::service::RequestContext;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
use rmcp::{RoleServer, ServerHandler};

/// 1x1 red-pixel PNG.
const TEST_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
/// Minimal (headers-only) WAV.
const TEST_WAV_BASE64: &str = "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=";

/// The upstream: implements the conformance suite's prescribed test tools.
#[derive(Clone, Default)]
struct ConformanceUpstream;

fn tool(name: &str, description: &str) -> Tool {
    let schema = serde_json::json!({ "type": "object", "properties": {} });
    let schema_obj = schema.as_object().expect("schema is an object").clone();
    Tool::new(name.to_string(), description.to_string(), schema_obj)
}

/// The `(content, is_error)` each conformance tool must answer with.
fn conformance_result(name: &str) -> Option<(serde_json::Value, bool)> {
    match name {
        "test_simple_text" => Some((
            serde_json::json!([
                { "type": "text", "text": "This is a simple text response for testing." }
            ]),
            false,
        )),
        "test_image_content" => Some((
            serde_json::json!([
                { "type": "image", "data": TEST_PNG_BASE64, "mimeType": "image/png" }
            ]),
            false,
        )),
        "test_audio_content" => Some((
            serde_json::json!([
                { "type": "audio", "data": TEST_WAV_BASE64, "mimeType": "audio/wav" }
            ]),
            false,
        )),
        "test_embedded_resource" => Some((
            serde_json::json!([{
                "type": "resource",
                "resource": {
                    "uri": "test://embedded-resource",
                    "mimeType": "text/plain",
                    "text": "This is an embedded resource content.",
                }
            }]),
            false,
        )),
        "test_multiple_content_types" => Some((
            serde_json::json!([
                { "type": "text", "text": "Multiple content types test:" },
                { "type": "image", "data": TEST_PNG_BASE64, "mimeType": "image/png" },
                {
                    "type": "resource",
                    "resource": {
                        "uri": "test://mixed-content-resource",
                        "mimeType": "application/json",
                        "text": "{\"test\":\"data\",\"value\":123}",
                    }
                }
            ]),
            false,
        )),
        "test_error_handling" => Some((
            serde_json::json!([
                { "type": "text", "text": "This tool intentionally returns an error for testing" }
            ]),
            true,
        )),
        _ => None,
    }
}

impl ServerHandler for ConformanceUpstream {
    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        Ok(ListToolsResult::with_all_items(vec![
            tool("test_simple_text", "Returns simple text content"),
            tool("test_image_content", "Returns image content"),
            tool("test_audio_content", "Returns audio content"),
            tool("test_embedded_resource", "Returns an embedded resource"),
            tool("test_multiple_content_types", "Returns mixed content"),
            tool("test_error_handling", "Always returns a tool error"),
        ]))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        let (content, is_error) = conformance_result(request.name.as_ref()).ok_or_else(|| {
            ErrorData::invalid_params(format!("unknown tool: {}", request.name), None)
        })?;
        let blocks: Vec<ContentBlock> = serde_json::from_value(content)
            .map_err(|e| ErrorData::internal_error(format!("bad fixture: {e}"), None))?;
        let result = if is_error {
            CallToolResult::error(blocks)
        } else {
            CallToolResult::success(blocks)
        };
        Ok(result.into())
    }

    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info
    }
}

#[tokio::main]
async fn main() {
    // The in-process upstream the gateway bridges to, on an ephemeral port.
    let upstream_service = StreamableHttpService::new(
        move || Ok(ConformanceUpstream),
        Arc::new(LocalSessionManager::default()),
        StreamableHttpServerConfig::default(),
    );
    let upstream_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind upstream port");
    let upstream_addr = upstream_listener.local_addr().expect("upstream addr");
    tokio::spawn(async move {
        axum::serve(
            upstream_listener,
            axum::Router::new().nest_service("/mcp", upstream_service),
        )
        .await
        .expect("serve upstream");
    });

    // Register it as a snapshot `mcp_servers` row and serve the SCOPED
    // gateway — the same construction `/mcp/{server}` uses in production,
    // so the suite exercises the shipped bridge + gateway chain.
    let server: aisix_core::McpServer = serde_json::from_value(serde_json::json!({
        "display_name": "conformance",
        "url": format!("http://{upstream_addr}/mcp"),
        "auth_type": "none",
    }))
    .expect("valid mcp_servers row");
    let snapshot = aisix_core::AisixSnapshot::new();
    snapshot
        .mcp_servers
        .insert(aisix_core::ResourceEntry::new("mcp-conf", server, 1));
    let gateway = McpGateway::from_snapshot_scoped(&snapshot, "conformance")
        .expect("scoped gateway over the registered upstream");

    let addr = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "127.0.0.1:3111".to_string());
    let app = axum::Router::new().nest_service("/mcp", streamable_http_service(gateway, 0));
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| panic!("bind {addr}: {e}"));
    println!("upstream:           http://{upstream_addr}/mcp");
    println!("conformance target: http://{addr}/mcp");
    axum::serve(listener, app).await.expect("serve");
}
