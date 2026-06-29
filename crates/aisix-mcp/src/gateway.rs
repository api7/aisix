//! The downstream-facing MCP gateway endpoint.
//!
//! [`McpGateway`] makes AISIX look like a single MCP server to a downstream
//! agent while fronting N registered upstream servers (each an [`McpBridge`]).
//! It is the other half of the dual role: an MCP *client* to each upstream
//! (via [`crate::RmcpBridge`]) and an MCP *server* to the agent (this type,
//! served over Streamable HTTP by [`streamable_http_service`]).
//!
//! Two operations, mirroring the upstream surface:
//! - `tools/list` fans out across every upstream and returns one aggregated
//!   list, each tool namespaced `server<SEP>tool`. An upstream that fails to
//!   list is skipped (its tools are simply absent), so one bad upstream does
//!   not blind the agent to the rest.
//! - `tools/call` strips the namespace prefix and routes to the owning
//!   upstream.
//!
//! The aggregator holds no per-request or per-session state, so governance
//! never depends on a transport session — which keeps it aligned with the
//! stateless direction of the MCP 2026-07-28 revision.
//!
//! Wiring this endpoint behind the gateway's auth / per-tool ACL / quota /
//! observability pipeline (and sourcing upstreams from the resource snapshot)
//! is the next step; this type takes an explicit set of upstreams and is not
//! yet mounted on any production listener.

use std::borrow::Cow;
use std::sync::Arc;

use rmcp::model::{
    CallToolRequestParams, CallToolResult, Content, ErrorData, ListToolsResult,
    PaginatedRequestParams, ServerCapabilities, ServerInfo, Tool,
};
use rmcp::service::RequestContext;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
use rmcp::{RoleServer, ServerHandler};

use aisix_core::AisixSnapshot;

use crate::bridge::{upstream_from_mcp_server, EphemeralBridge, McpBridge};

/// Separator between an upstream server's registered name and a tool name in
/// the aggregated namespace, e.g. `github__create_issue`. Server names must
/// not contain it; tool names may (we split on the first occurrence).
pub const TOOL_NAMESPACE_SEPARATOR: &str = "__";

/// One registered upstream: its gateway-facing name and the live bridge to it.
struct NamedUpstream {
    name: String,
    bridge: Arc<dyn McpBridge>,
}

/// Aggregates N upstream MCP servers behind one downstream MCP server surface.
/// Cheap to clone (the upstream set is shared); the Streamable HTTP transport
/// clones it per session.
#[derive(Clone)]
pub struct McpGateway {
    upstreams: Arc<[NamedUpstream]>,
}

impl McpGateway {
    /// Build a gateway over the given `(server_name, bridge)` upstreams.
    /// Registration order is the order tools are listed in.
    ///
    /// A name may only register once: a duplicate is dropped (the first
    /// registration wins) with a warning, rather than silently shadowing the
    /// later one and emitting duplicate tool names on the wire. Server names
    /// must not contain [`TOOL_NAMESPACE_SEPARATOR`].
    pub fn new(upstreams: impl IntoIterator<Item = (String, Arc<dyn McpBridge>)>) -> Self {
        let mut seen = std::collections::HashSet::new();
        let mut deduped = Vec::new();
        for (name, bridge) in upstreams {
            debug_assert!(
                !name.contains(TOOL_NAMESPACE_SEPARATOR),
                "upstream server name `{name}` must not contain the namespace \
                 separator `{TOOL_NAMESPACE_SEPARATOR}`"
            );
            if !seen.insert(name.clone()) {
                tracing::warn!(
                    upstream = %name,
                    "duplicate MCP upstream name; keeping the first registration, dropping this one"
                );
                continue;
            }
            deduped.push(NamedUpstream { name, bridge });
        }
        Self {
            upstreams: deduped.into(),
        }
    }

    /// Build a gateway whose upstreams are the **enabled** `mcp_servers` in the
    /// snapshot, each reached through an [`EphemeralBridge`] (connect per
    /// request). Disabled servers are skipped. Registration order follows the
    /// snapshot's iteration order; duplicate display_names are deduped (first
    /// wins) by [`McpGateway::new`], though the Admin API already enforces
    /// uniqueness.
    pub fn from_snapshot(snapshot: &AisixSnapshot) -> Self {
        let upstreams = snapshot
            .mcp_servers
            .entries()
            .into_iter()
            .filter(|entry| entry.value.enabled)
            .map(|entry| {
                let upstream = upstream_from_mcp_server(&entry.value);
                let bridge: Arc<dyn McpBridge> = Arc::new(EphemeralBridge::new(upstream));
                (entry.value.display_name.clone(), bridge)
            });
        McpGateway::new(upstreams)
    }

    fn find(&self, server: &str) -> Option<&Arc<dyn McpBridge>> {
        self.upstreams
            .iter()
            .find(|u| u.name == server)
            .map(|u| &u.bridge)
    }
}

impl ServerHandler for McpGateway {
    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        // Fan out concurrently; each upstream call is already deadline-bounded
        // by its bridge, so a slow upstream cannot stall the aggregate.
        let listed = futures::future::join_all(
            self.upstreams
                .iter()
                .map(|u| async move { (u.name.as_str(), u.bridge.list_tools().await) }),
        )
        .await;

        let mut tools = Vec::new();
        for (server, result) in listed {
            match result {
                Ok(upstream_tools) => {
                    tools.extend(upstream_tools.into_iter().map(|t| prefixed_tool(server, t)));
                }
                Err(error) => {
                    // Degrade gracefully: drop this upstream's tools, keep the
                    // rest. Detail is logged server-side (never client-visible).
                    tracing::warn!(
                        upstream = server,
                        error = %error,
                        "skipping upstream in tools/list: list_tools failed"
                    );
                }
            }
        }
        Ok(ListToolsResult::with_all_items(tools))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        let (server, tool) = request
            .name
            .split_once(TOOL_NAMESPACE_SEPARATOR)
            .ok_or_else(|| {
                ErrorData::invalid_params(
                    format!(
                        "tool name '{}' is missing a 'server__tool' prefix",
                        request.name
                    ),
                    None,
                )
            })?;

        let bridge = self.find(server).ok_or_else(|| {
            ErrorData::invalid_params(format!("unknown MCP server '{server}'"), None)
        })?;

        let arguments = request
            .arguments
            .map(serde_json::Value::Object)
            .unwrap_or(serde_json::Value::Null);

        let result = bridge.call_tool(tool, arguments).await.map_err(|error| {
            // Generic client-facing message; the upstream's detail (which may
            // include its URL) is logged server-side, not surfaced to the agent.
            tracing::warn!(
                upstream = server,
                tool = tool,
                error = %error,
                "upstream tools/call failed"
            );
            ErrorData::internal_error(
                format!("upstream MCP server '{server}' failed to call tool"),
                None,
            )
        })?;

        into_call_tool_result(result)
    }

    fn get_info(&self) -> ServerInfo {
        let mut info = ServerInfo::default();
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info.instructions = Some(
            "AISIX MCP gateway: aggregates tools from registered upstream MCP \
             servers, namespaced as `server__tool`."
                .to_string(),
        );
        info
    }
}

/// Build the Streamable HTTP service for this gateway, ready to nest in axum
/// at `/mcp`.
///
/// Configured stateless (no sticky session, JSON responses): the aggregator
/// keeps no per-session state, so the endpoint can sit behind a plain load
/// balancer — matching the MCP 2026-07-28 transport direction.
pub fn streamable_http_service(
    gateway: McpGateway,
) -> StreamableHttpService<McpGateway, LocalSessionManager> {
    let mut config = StreamableHttpServerConfig::default();
    config.stateful_mode = false;
    config.json_response = true;
    // Disable rmcp's `Host`-header allowlist. Its default
    // (`localhost`/`127.0.0.1`/`::1`) is a DNS-rebinding guard for
    // browser-driven local servers — it 403s every request whose `Host` is
    // the deployment's real DNS name. This endpoint is not browser-driven: it
    // is reached server-to-server by agents and is gated by the AISIX API key,
    // which is the real access control. An empty allowlist accepts any `Host`;
    // the request is still authenticated upstream of this service. (Operators
    // who want Host pinning can layer it at their ingress.)
    config.allowed_hosts = Vec::new();
    StreamableHttpService::new(
        move || Ok(gateway.clone()),
        Arc::new(LocalSessionManager::default()),
        config,
    )
}

/// Namespace an upstream tool: `server<SEP>tool`, preserving its schema and
/// (optional) description.
fn prefixed_tool(server: &str, tool: crate::McpTool) -> Tool {
    let schema = match tool.input_schema {
        serde_json::Value::Object(map) => map,
        // A non-object schema is malformed per JSON Schema; advertise an empty
        // object rather than dropping the tool.
        _ => serde_json::Map::new(),
    };
    Tool::new_with_raw(
        format!("{server}{TOOL_NAMESPACE_SEPARATOR}{}", tool.name),
        tool.description.map(Cow::Owned),
        schema,
    )
}

/// Map our [`crate::McpToolResult`] back onto rmcp's `CallToolResult`,
/// preserving the upstream's tool-level error flag (a tool-level error is
/// propagated as `Ok(error_result)`, not turned into a protocol error).
fn into_call_tool_result(result: crate::McpToolResult) -> Result<CallToolResult, ErrorData> {
    let content: Vec<Content> = serde_json::from_value(result.content).map_err(|e| {
        ErrorData::internal_error(format!("malformed tool content from upstream: {e}"), None)
    })?;
    let mut call_result = if result.is_error {
        CallToolResult::error(content)
    } else {
        CallToolResult::success(content)
    };
    call_result.structured_content = result.structured_content;
    Ok(call_result)
}
