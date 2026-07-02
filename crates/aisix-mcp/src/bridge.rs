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

use std::collections::HashMap;
use std::time::Duration;

use aisix_core::{McpAuthType, McpServer};
use async_trait::async_trait;
use http::{HeaderName, HeaderValue};
use rmcp::model::CallToolRequestParams;
use rmcp::service::{ClientInitializeError, RoleClient, RunningService};
use rmcp::transport::streamable_http_client::{
    StreamableHttpClientTransportConfig, StreamableHttpError,
};
use rmcp::transport::StreamableHttpClientTransport;
use rmcp::ServiceExt;

use crate::error::McpError;

/// Default deadline for a single upstream operation (connect / list / call).
/// rmcp's high-level client sets no request timeout and reqwest has no default
/// one, so without this a hung or slow upstream pins the gateway request task
/// indefinitely. Overridable per upstream via [`McpUpstream::with_timeout`].
pub const DEFAULT_UPSTREAM_TIMEOUT: Duration = Duration::from_secs(30);

/// Header carrying the gateway-held key for `api_key` upstream auth.
const API_KEY_HEADER: &str = "x-api-key";

/// How the gateway authenticates to an upstream MCP server. The credential is
/// held here on the gateway side and is never exposed to the calling agent —
/// the agent presents only its AISIX key. The MCP authorization spec
/// (2025-11-25) also requires that a downstream client token is never passed
/// through to the upstream; every credential set here — a Bearer, an API key,
/// or an OAuth token the gateway mints itself — is a distinct, gateway-held
/// credential.
#[derive(Clone)]
pub enum McpAuth {
    /// No upstream auth — the server is reachable as-is.
    None,
    /// Send `Authorization: Bearer <token>` on every upstream request. The
    /// token is the raw value, without the `Bearer ` prefix.
    Bearer(String),
    /// Send `x-api-key: <key>` on every upstream request.
    ApiKey(String),
    /// OAuth 2.0 client credentials (RFC 6749 §4.4): mint an access token at
    /// the configured token endpoint and send it as `Authorization: Bearer
    /// <access_token>`. The token is gateway-minted and gateway-held — never
    /// the caller's credential (see [`crate::oauth`]).
    OAuth2(OAuthClientConfig),
}

/// Client-credentials parameters for [`McpAuth::OAuth2`].
#[derive(Clone)]
pub struct OAuthClientConfig {
    /// OAuth client identifier (non-secret).
    pub client_id: String,
    /// OAuth client secret. Redacted from `Debug` like every other
    /// gateway-held credential in this module.
    pub client_secret: String,
    /// Token endpoint URL the credentials are exchanged at (non-secret).
    pub token_url: String,
    /// Scopes to request, joined with spaces into the `scope` parameter.
    pub scopes: Vec<String>,
}

// Hand-written for the same reason as `McpAuth`'s: the client secret must
// never land in logs via `{:?}`. The non-secret fields stay visible — they
// are what an operator needs to identify the token exchange being logged.
impl std::fmt::Debug for OAuthClientConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OAuthClientConfig")
            .field("client_id", &self.client_id)
            .field("client_secret", &"***redacted***")
            .field("token_url", &self.token_url)
            .field("scopes", &self.scopes)
            .finish()
    }
}

// Hand-written so the gateway-held token never lands in logs via `{:?}`. This
// crate is the credential holder; a derived `Debug` would print the bearer in
// plaintext the moment any caller logs an upstream.
impl std::fmt::Debug for McpAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            McpAuth::None => f.write_str("None"),
            McpAuth::Bearer(_) => f.write_str("Bearer(***redacted***)"),
            McpAuth::ApiKey(_) => f.write_str("ApiKey(***redacted***)"),
            // Delegates to the redacting `OAuthClientConfig` impl above.
            McpAuth::OAuth2(cfg) => f.debug_tuple("OAuth2").field(cfg).finish(),
        }
    }
}

/// Connection parameters for a single upstream MCP server.
#[derive(Clone)]
pub struct McpUpstream {
    /// The server's Streamable HTTP MCP endpoint, e.g.
    /// `https://api.example.com/mcp`.
    pub url: String,
    /// Upstream authentication, held gateway-side.
    pub auth: McpAuth,
    /// Per-operation deadline. Defaults to [`DEFAULT_UPSTREAM_TIMEOUT`].
    pub timeout: Duration,
}

// Manual so a `Bearer` token cannot leak through `McpUpstream`'s `Debug`
// (delegates to the redacting `McpAuth` impl above).
impl std::fmt::Debug for McpUpstream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpUpstream")
            .field("url", &self.url)
            .field("auth", &self.auth)
            .field("timeout", &self.timeout)
            .finish()
    }
}

impl McpUpstream {
    /// Build an unauthenticated upstream with the default timeout.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            auth: McpAuth::None,
            timeout: DEFAULT_UPSTREAM_TIMEOUT,
        }
    }

    /// Set Bearer auth (raw token, no `Bearer ` prefix).
    pub fn with_bearer(mut self, token: impl Into<String>) -> Self {
        self.auth = McpAuth::Bearer(token.into());
        self
    }

    /// Set API-key auth (sent as `x-api-key: <key>`).
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.auth = McpAuth::ApiKey(key.into());
        self
    }

    /// Set OAuth 2.0 client-credentials auth.
    pub fn with_oauth2(mut self, config: OAuthClientConfig) -> Self {
        self.auth = McpAuth::OAuth2(config);
        self
    }

    /// Override the per-operation deadline.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

/// One tool advertised by an upstream server, normalised off the wire shape.
///
/// Minimal for this step: tool annotations (`readOnlyHint` / `destructiveHint`)
/// and `output_schema` are dropped here and will be carried when the per-tool
/// ACL / guardrail layer (DP-4) needs them.
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
    /// The tool's structured result, when it returns one (MCP `structuredContent`).
    /// A tool may return only structured content with an empty `content` array.
    pub structured_content: Option<serde_json::Value>,
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
    timeout: Duration,
}

impl RmcpBridge {
    /// Open a session to `upstream`: build the Streamable HTTP transport
    /// (injecting gateway-held auth — for `oauth2` this mints or reuses a
    /// cached access token first) and run the `initialize` handshake. The
    /// whole sequence, token minting included, is bounded by the upstream's
    /// timeout.
    pub async fn connect(upstream: &McpUpstream) -> Result<Self, McpError> {
        let establish = async {
            let transport = match &upstream.auth {
                McpAuth::None => StreamableHttpClientTransport::from_uri(upstream.url.clone()),
                McpAuth::Bearer(token) => StreamableHttpClientTransport::from_config(
                    StreamableHttpClientTransportConfig::with_uri(upstream.url.clone())
                        .auth_header(token.clone()),
                ),
                McpAuth::ApiKey(key) => {
                    // A key with non-header-safe bytes is a clean config error,
                    // not a panic — and the key itself never enters the message.
                    let mut value = HeaderValue::from_str(key).map_err(|_| {
                        McpError::Connect(
                            "upstream API key is not a valid HTTP header value".to_string(),
                        )
                    })?;
                    // Marks the value opaque to `Debug` formatting of the
                    // header map, mirroring this module's redaction posture.
                    value.set_sensitive(true);
                    let headers = HashMap::from([(HeaderName::from_static(API_KEY_HEADER), value)]);
                    StreamableHttpClientTransport::from_config(
                        StreamableHttpClientTransportConfig::with_uri(upstream.url.clone())
                            .custom_headers(headers),
                    )
                }
                McpAuth::OAuth2(cfg) => {
                    let token = crate::oauth::get_or_fetch(cfg).await?;
                    StreamableHttpClientTransport::from_config(
                        StreamableHttpClientTransportConfig::with_uri(upstream.url.clone())
                            .auth_header(token),
                    )
                }
            };
            ().serve(transport).await.map_err(|e| {
                // An upstream 401 against a minted token means the token was
                // revoked or expired earlier than promised: drop the cache
                // entry so the next attempt re-mints instead of replaying it.
                if let McpAuth::OAuth2(cfg) = &upstream.auth {
                    if init_error_is_unauthorized(&e) {
                        crate::oauth::invalidate(cfg);
                    }
                }
                McpError::Connect(e.to_string())
            })
        };
        let running = tokio::time::timeout(upstream.timeout, establish)
            .await
            .map_err(|_| McpError::Connect("upstream MCP connect timed out".to_string()))??;
        Ok(Self {
            running,
            timeout: upstream.timeout,
        })
    }
}

/// Whether a failed `initialize` handshake was an upstream `401 Unauthorized`.
///
/// The reqwest transport surfaces a 401 in one of two stable shapes (rmcp is
/// pinned exactly, so these cannot drift silently): a 401 carrying a
/// `WWW-Authenticate` header becomes `StreamableHttpError::AuthRequired`, and
/// any other non-success status becomes
/// `UnexpectedServerResponse("HTTP <status>: …")`. Both arrive here inside
/// `ClientInitializeError::TransportError` as the type-erased transport error;
/// the downcast names rmcp's own reqwest (`rmcp_reqwest`, the 0.13 line — not
/// the workspace 0.12) so the types match. Post-handshake operations don't
/// need this: [`EphemeralBridge`] reconnects per operation, so every request
/// replays the handshake and a rejected token always surfaces on this path.
fn init_error_is_unauthorized(error: &ClientInitializeError) -> bool {
    let ClientInitializeError::TransportError { error, .. } = error else {
        return false;
    };
    match error
        .error
        .downcast_ref::<StreamableHttpError<rmcp_reqwest::Error>>()
    {
        Some(StreamableHttpError::AuthRequired(_)) => true,
        Some(StreamableHttpError::UnexpectedServerResponse(message)) => {
            // Anchored to the prefix: the format is `HTTP <status>: <body>`,
            // so an upstream body can never fake a 401 here.
            message.starts_with("HTTP 401")
        }
        _ => false,
    }
}

#[async_trait]
impl McpBridge for RmcpBridge {
    async fn list_tools(&self) -> Result<Vec<McpTool>, McpError> {
        let result = tokio::time::timeout(self.timeout, self.running.list_tools(None))
            .await
            .map_err(|_| McpError::Request("upstream tools/list timed out".to_string()))?
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
        let result = tokio::time::timeout(self.timeout, self.running.call_tool(params))
            .await
            .map_err(|_| McpError::Request("upstream tools/call timed out".to_string()))?
            .map_err(|e| McpError::Request(e.to_string()))?;
        let content = serde_json::to_value(&result.content)
            .map_err(|e| McpError::Request(format!("failed to encode tool result: {e}")))?;
        Ok(McpToolResult {
            content,
            structured_content: result.structured_content,
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

/// Build the connection parameters for an upstream from its registered
/// [`McpServer`] resource: maps `auth_type` and its credential fields to
/// [`McpAuth`] and `timeout_ms` to the per-operation deadline.
///
/// Stays permissive on purpose: fields a mis-configured resource left unset
/// map to empty strings rather than erroring here. The credential exchange
/// then fails cleanly at connect time and that server degrades like any
/// unreachable upstream (its tools drop out of `tools/list`, the failure is
/// logged), instead of one bad row poisoning snapshot loading.
pub fn upstream_from_mcp_server(server: &McpServer) -> McpUpstream {
    let auth = match server.auth_type {
        McpAuthType::None => McpAuth::None,
        McpAuthType::Bearer => McpAuth::Bearer(server.secret.clone().unwrap_or_default()),
        McpAuthType::ApiKey => McpAuth::ApiKey(server.secret.clone().unwrap_or_default()),
        McpAuthType::OAuth2 => McpAuth::OAuth2(OAuthClientConfig {
            client_id: server.client_id.clone().unwrap_or_default(),
            client_secret: server.secret.clone().unwrap_or_default(),
            token_url: server.token_url.clone().unwrap_or_default(),
            scopes: server.scopes.clone().unwrap_or_default(),
        }),
    };
    let timeout = server
        .timeout_ms
        .map(Duration::from_millis)
        .unwrap_or(DEFAULT_UPSTREAM_TIMEOUT);
    McpUpstream {
        url: server.url.clone(),
        auth,
        timeout,
    }
}

/// An [`McpBridge`] that opens a fresh upstream session for each operation and
/// drops it when done.
///
/// The downstream `/mcp` endpoint is stateless, so the gateway holds no
/// long-lived upstream connections: every `tools/list` / `tools/call` connects,
/// runs, and disconnects. Connection pooling is a later optimization; this keeps
/// the snapshot-sourced gateway free of connection-lifecycle state, so a
/// configuration change is picked up on the next request with nothing to
/// reconcile.
pub struct EphemeralBridge {
    upstream: McpUpstream,
}

impl EphemeralBridge {
    pub fn new(upstream: McpUpstream) -> Self {
        Self { upstream }
    }
}

#[async_trait]
impl McpBridge for EphemeralBridge {
    async fn list_tools(&self) -> Result<Vec<McpTool>, McpError> {
        RmcpBridge::connect(&self.upstream)
            .await?
            .list_tools()
            .await
    }

    async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<McpToolResult, McpError> {
        RmcpBridge::connect(&self.upstream)
            .await?
            .call_tool(name, arguments)
            .await
    }
}
