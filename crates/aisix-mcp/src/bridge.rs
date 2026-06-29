//! The upstream MCP client, behind the [`McpBridge`] trait.
//!
//! A bridge owns one live MCP session to a single upstream server (Streamable
//! HTTP transport) and exposes just the two operations the gateway needs in
//! this first cut: enumerate the server's tools, and invoke one. Aggregating
//! many bridges into the downstream-facing `/mcp` endpoint, tool namespacing,
//! and wiring into the shared guardrail/quota pipeline come in later steps —
//! this layer only proves a governed tunnel to one real upstream.
//!
//! All `rmcp` types are converted to this crate's own DTOs at the boundary so
//! the rest of the data plane never depends on the SDK directly. That keeps
//! rmcp's still-moving API contained to this file.

use async_trait::async_trait;
use rmcp::model::CallToolRequestParams;
use rmcp::service::{RoleClient, RunningService};
use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::ServiceExt;

use crate::error::McpError;

/// How the gateway authenticates to an upstream MCP server. The credential is
/// held here on the gateway side and is never exposed to the calling agent —
/// the agent presents only its AISIX key. The MCP authorization spec
/// (2025-11-25) also requires that a downstream client token is never passed
/// through to the upstream; a Bearer set here is a distinct, gateway-held
/// credential.
#[derive(Debug, Clone)]
pub enum McpAuth {
    /// No upstream auth — the server is reachable as-is.
    None,
    /// Send `Authorization: Bearer <token>` on every upstream request. The
    /// token is the raw value, without the `Bearer ` prefix.
    Bearer(String),
}

/// Connection parameters for a single upstream MCP server.
#[derive(Debug, Clone)]
pub struct McpUpstream {
    /// The server's Streamable HTTP MCP endpoint, e.g.
    /// `https://api.example.com/mcp`.
    pub url: String,
    /// Upstream authentication, held gateway-side.
    pub auth: McpAuth,
}

impl McpUpstream {
    /// Build an unauthenticated upstream.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            auth: McpAuth::None,
        }
    }

    /// Set Bearer auth (raw token, no `Bearer ` prefix).
    pub fn with_bearer(mut self, token: impl Into<String>) -> Self {
        self.auth = McpAuth::Bearer(token.into());
        self
    }
}

/// One tool advertised by an upstream server, normalised off the wire shape.
#[derive(Debug, Clone, PartialEq)]
pub struct McpTool {
    /// The tool's name, as the upstream advertises it (no gateway prefix yet).
    pub name: String,
    /// Human-readable description, if the server provides one.
    pub description: Option<String>,
    /// JSON Schema for the tool's arguments, as a JSON object.
    pub input_schema: serde_json::Value,
}

/// The outcome of a `tools/call`, normalised off the wire shape.
#[derive(Debug, Clone, PartialEq)]
pub struct McpToolResult {
    /// The content blocks the tool returned, as a JSON array (text, images,
    /// resource links, …). Left as raw JSON here; the downstream endpoint
    /// shapes it for the agent.
    pub content: serde_json::Value,
    /// Whether the upstream flagged this result as a tool-level error.
    pub is_error: bool,
}

/// The gateway's view of one upstream MCP server. Implemented by
/// [`RmcpBridge`]; kept as a trait so the rest of the data plane depends on
/// this surface rather than on `rmcp`, and so the upstream can be stubbed in
/// higher-layer tests.
#[async_trait]
pub trait McpBridge: Send + Sync {
    /// List the tools the upstream currently exposes.
    async fn list_tools(&self) -> Result<Vec<McpTool>, McpError>;

    /// Invoke a tool by name with the given JSON arguments. `arguments` must
    /// be a JSON object or `null` (no arguments); anything else is rejected.
    async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<McpToolResult, McpError>;
}

/// `rmcp`-backed [`McpBridge`]: holds one running client session to the
/// upstream. Dropping it tears the session down.
pub struct RmcpBridge {
    running: RunningService<RoleClient, ()>,
}

impl RmcpBridge {
    /// Open a session to `upstream`: build the Streamable HTTP transport
    /// (injecting gateway-held auth) and run the `initialize` handshake.
    pub async fn connect(upstream: &McpUpstream) -> Result<Self, McpError> {
        let transport = match &upstream.auth {
            McpAuth::None => StreamableHttpClientTransport::from_uri(upstream.url.clone()),
            McpAuth::Bearer(token) => StreamableHttpClientTransport::from_config(
                StreamableHttpClientTransportConfig::with_uri(upstream.url.clone())
                    .auth_header(token.clone()),
            ),
        };
        let running = ().serve(transport).await.map_err(|e| McpError::Connect(e.to_string()))?;
        Ok(Self { running })
    }
}

#[async_trait]
impl McpBridge for RmcpBridge {
    async fn list_tools(&self) -> Result<Vec<McpTool>, McpError> {
        let result = self
            .running
            .list_tools(None)
            .await
            .map_err(|e| McpError::Request(e.to_string()))?;
        Ok(result.tools.into_iter().map(into_mcp_tool).collect())
    }

    async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<McpToolResult, McpError> {
        let mut params = CallToolRequestParams::new(name.to_string());
        params = match arguments {
            serde_json::Value::Null => params,
            serde_json::Value::Object(map) => params.with_arguments(map),
            _ => {
                return Err(McpError::Request(
                    "tool arguments must be a JSON object or null".to_string(),
                ))
            }
        };
        let result = self
            .running
            .call_tool(params)
            .await
            .map_err(|e| McpError::Request(e.to_string()))?;
        let content = serde_json::to_value(&result.content)
            .map_err(|e| McpError::Request(format!("failed to encode tool result: {e}")))?;
        Ok(McpToolResult {
            content,
            is_error: result.is_error.unwrap_or(false),
        })
    }
}

/// Normalise an `rmcp` `Tool` into our [`McpTool`] DTO.
fn into_mcp_tool(tool: rmcp::model::Tool) -> McpTool {
    McpTool {
        name: tool.name.into_owned(),
        description: tool.description.map(|d| d.into_owned()),
        input_schema: serde_json::Value::Object((*tool.input_schema).clone()),
    }
}
