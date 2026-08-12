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

/// Shared HTTP client for every rmcp streamable-http transport, with the
/// deployment's `upstream.*` connection settings applied.
///
/// rmcp pins its own reqwest line (0.13 — see Cargo.toml), so the shared
/// `aisix_gateway::client_builder()` (workspace reqwest) cannot be handed
/// to it directly; this is the sanctioned second construction site that
/// applies the SAME `upstream_http::config()` values to rmcp's reqwest.
///
/// What this changes vs rmcp's `default_http_client()`:
/// - a connect timeout exists at all (rmcp sets none, so a black-holed
///   upstream was bounded only by the coarse per-operation deadline —
///   the actual bug this fixes);
/// - connection POOLING turns ON. rmcp deliberately disables idle
///   pooling (`pool_max_idle_per_host(0)`) to dodge ~40 ms delayed-ACK
///   stalls on reused connections; here reuse wins — every other
///   outbound path pools under `upstream.*` management, and per-call
///   TCP+TLS handshakes to a remote MCP server cost far more than the
///   stall rmcp avoids. An operator can restore rmcp's behaviour with
///   `upstream.pool_max_idle_per_host: 0`;
/// - the TCP keepalive triple moves from reqwest's default 15 s/15 s/3
///   to the deployment's 60 s/30 s/5;
/// - the deployment's `upstream.tls` trust decision applies here too. An
///   MCP server behind an enterprise CA is exactly as common as a model
///   endpoint behind one, and the PEM has to be re-parsed rather than
///   reused because rmcp's `Certificate` is a different crate version's
///   type than the one `upstream_tls` caches for the workspace line;
/// - redirects are refused, as they are on every other client that
///   carries a caller's payload under a gateway-held credential. A
///   `tools/call` POSTs the caller's arguments under the MCP server's
///   `x-api-key` or bearer, and reqwest does not strip either on a hop
///   to whatever host a `Location` names.
///
/// One client for all MCP upstreams = one shared pool, matching how the
/// provider bridges share theirs (auth is injected per-request by the
/// transport, never client-wide).
fn shared_http_client() -> rmcp_reqwest::Client {
    static CLIENT: std::sync::OnceLock<rmcp_reqwest::Client> = std::sync::OnceLock::new();
    CLIENT
        .get_or_init(|| {
            let cfg = aisix_gateway::upstream_http::config();
            let mut b = rmcp_reqwest::Client::builder()
                .redirect(rmcp_reqwest::redirect::Policy::none())
                .pool_idle_timeout(cfg.pool_idle_timeout)
                .tcp_keepalive(cfg.tcp_keepalive);
            if let Some(d) = cfg.connect_timeout {
                b = b.connect_timeout(d);
            }
            if let Some(d) = cfg.tcp_keepalive_interval {
                b = b.tcp_keepalive_interval(d);
            }
            if let Some(n) = cfg.tcp_keepalive_retries {
                b = b.tcp_keepalive_retries(n);
            }
            if let Some(n) = cfg.pool_max_idle_per_host {
                b = b.pool_max_idle_per_host(n);
            }
            if let Some(pem) = &cfg.tls.extra_ca_pem {
                // Already validated at boot by `TlsSettings::load`; a
                // failure here would mean rmcp's reqwest disagrees with
                // ours about the same bytes, which is worth a loud line
                // rather than a silently untrusting client.
                match rmcp_reqwest::Certificate::from_pem_bundle(pem) {
                    Ok(roots) => {
                        for root in roots {
                            b = b.add_root_certificate(root);
                        }
                    }
                    Err(e) => tracing::error!(
                        error = %e,
                        "upstream.tls.ca_file not applied to MCP upstream connections"
                    ),
                }
            }
            if let Some(configured) = &cfg.tls.client_identity {
                match rmcp_reqwest::Identity::from_pem(&configured.joined()) {
                    Ok(identity) => b = b.identity(identity),
                    Err(e) => tracing::error!(
                        error = %e,
                        "upstream.tls client identity not applied to MCP upstream connections"
                    ),
                }
            }
            if !cfg.tls.verify {
                b = b.danger_accept_invalid_certs(true);
            }
            // The connection settings are lost if the build fails; the
            // redirect refusal must not be, which `Client::new()` would
            // silently give back.
            b.build().unwrap_or_else(|_| {
                rmcp_reqwest::Client::builder()
                    .redirect(rmcp_reqwest::redirect::Policy::none())
                    .build()
                    .expect("a client with only a redirect policy always builds")
            })
        })
        .clone()
}

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
            // Every arm goes through `with_client(shared_http_client(), ..)`
            // so the transport inherits the deployment's `upstream.*`
            // connection settings — `from_uri`/`from_config` would build
            // rmcp's own default client with none of them.
            let transport = match &upstream.auth {
                McpAuth::None => StreamableHttpClientTransport::with_client(
                    shared_http_client(),
                    StreamableHttpClientTransportConfig::with_uri(upstream.url.clone()),
                ),
                McpAuth::Bearer(token) => StreamableHttpClientTransport::with_client(
                    shared_http_client(),
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
                    StreamableHttpClientTransport::with_client(
                        shared_http_client(),
                        StreamableHttpClientTransportConfig::with_uri(upstream.url.clone())
                            .custom_headers(headers),
                    )
                }
                McpAuth::OAuth2(cfg) => {
                    let token = crate::oauth::get_or_fetch(cfg).await?;
                    StreamableHttpClientTransport::with_client(
                        shared_http_client(),
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
                // Bound + sanitize: a bare-401 shape embeds the upstream's
                // response body in the error text, which lands in gateway
                // logs. An upstream that reflects request headers into its
                // error body (or emits control characters for log injection)
                // must not get either past this point verbatim.
                McpError::Connect(sanitize_error_message(&e.to_string()))
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

/// Bound an upstream-derived error message for logging: control characters
/// (log-injection vectors) are stripped and the text is truncated, since a
/// bare non-success response embeds the upstream's body verbatim.
pub(crate) fn sanitize_error_message(message: &str) -> String {
    const MAX_LEN: usize = 256;
    let cleaned: String = message
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    if cleaned.chars().count() <= MAX_LEN {
        cleaned
    } else {
        let truncated: String = cleaned.chars().take(MAX_LEN).collect();
        format!("{truncated}…")
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
///
/// Known gap (availability-only): a bare 401 whose body parses as a JSON-RPC
/// error arrives as `ClientInitializeError::JsonRpcError` — not a transport
/// error — and is not recognized here, so a revoked token is replayed until
/// the cache's expiry skew retires it (at most ~59 minutes at the default
/// lifetime). Spec-conforming servers answer 401 with `WWW-Authenticate`,
/// which IS recognized.
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
/// Whether a URL sends its traffic over cleartext HTTP. Matches how the
/// HTTP stack will actually treat it (the URL parser lowercases the scheme
/// and strips leading whitespace), so `HTTP://` and `" http://"` are
/// flagged too; `https://` never is.
fn is_cleartext(url: &str) -> bool {
    let trimmed = url.trim_start();
    trimmed
        .get(..7)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("http://"))
}

/// The cleartext-credential findings for a registered server: which
/// gateway-held secret travels over plain HTTP, and to which URL. Applies
/// to every credentialed auth type against an `http://` server URL —
/// `type: mcp` and `type: openapi` rows share these fields — plus, for
/// `oauth2`, a cleartext `token_url` (it carries the client secret; a
/// distinct finding even when `token_url` equals `url`). Pure, so tests
/// pin the selection.
pub(crate) fn cleartext_findings(server: &McpServer) -> Vec<(&'static str, String)> {
    let mut findings = Vec::new();
    if server.auth_type != McpAuthType::None && is_cleartext(&server.url) {
        findings.push(("the gateway-held upstream credential", server.url.clone()));
    }
    if server.auth_type == McpAuthType::OAuth2 {
        let token_url = server.token_url.as_deref().unwrap_or_default();
        if is_cleartext(token_url) {
            findings.push(("the OAuth client secret", token_url.to_string()));
        }
    }
    findings
}

/// Warn — once per distinct finding per process — that a credential travels
/// unencrypted. Deliberately a warning, not a rejection: plain-HTTP
/// upstreams inside a private network are a lawful, common deployment, and
/// request behavior (including redirect handling) stays aligned with the
/// reference SDK baseline (#879). Deduped on the (server, finding, url)
/// tuple because the gateway is rebuilt from the snapshot on every request
/// — an undeduped warn would log per call.
pub(crate) fn warn_cleartext_credential(server: &McpServer) {
    use std::sync::{Mutex, OnceLock};
    type Warned = std::collections::HashSet<(String, &'static str, String)>;
    static WARNED: OnceLock<Mutex<Warned>> = OnceLock::new();
    for (what, url) in cleartext_findings(server) {
        let mut warned = WARNED
            .get_or_init(Default::default)
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if warned.insert((server.name.clone(), what, url.clone())) {
            tracing::warn!(
                server = %server.name,
                url = %url,
                "{what} is sent over cleartext http; anyone on the network path can read \
                 it — serve this upstream over https"
            );
        }
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cleartext_credential_detection() {
        let server = |auth_type: &str, url: &str, token_url: Option<&str>| -> McpServer {
            serde_json::from_value(serde_json::json!({
                "display_name": "s",
                "url": url,
                "auth_type": auth_type,
                "secret": "k",
                "client_id": "c",
                "token_url": token_url,
            }))
            .expect("valid server")
        };

        // A credentialed http:// URL is flagged; https and credential-less
        // http are not (https also starts with "http", so the scheme match
        // must include the separator). The URL parser lowercases the scheme
        // and strips leading whitespace, so those shapes flag too.
        for auth in ["bearer", "api_key"] {
            assert_eq!(
                cleartext_findings(&server(auth, "http://mcp.internal/mcp", None)).len(),
                1,
                "{auth} over http must be flagged"
            );
        }
        assert!(cleartext_findings(&server("bearer", "https://mcp.internal/mcp", None)).is_empty());
        assert!(cleartext_findings(&server("none", "http://mcp.internal/mcp", None)).is_empty());
        assert!(cleartext_findings(&server("bearer", "", None)).is_empty());
        assert_eq!(
            cleartext_findings(&server("bearer", "HTTP://mcp.internal/mcp", None)).len(),
            1
        );
        assert_eq!(
            cleartext_findings(&server("bearer", " http://mcp.internal/mcp", None)).len(),
            1
        );

        // oauth2: the server URL carries the minted token, the token URL the
        // client secret — two distinct findings, even at the same URL.
        let both = cleartext_findings(&server("oauth2", "http://x/mcp", Some("http://x/mcp")));
        assert_eq!(both.len(), 2, "token_url == url must still flag twice");
        let token_only = cleartext_findings(&server(
            "oauth2",
            "https://x/mcp",
            Some("http://idp.internal/token"),
        ));
        assert_eq!(token_only.len(), 1);
        assert_eq!(token_only[0].0, "the OAuth client secret");
    }

    /// The dial to an upstream that swallows SYNs must be cut by the
    /// shared client's `upstream.connect_timeout_ms` (default 5 s) —
    /// before the transport ran on rmcp's default reqwest, which has no
    /// connect timeout, so the dial hung until the OUTER
    /// `upstream.timeout` (12 s here; kernel SYN retries run ≈127 s).
    /// 203.0.113.1 (TEST-NET-3) is reserved and unrouted, the standard
    /// black-hole address; on networks that answer with an immediate
    /// "unreachable" the call fails fast either way — the assertion is
    /// the upper bound, which only the connect timeout guarantees.
    #[tokio::test]
    async fn connect_timeout_bounds_an_unreachable_upstream() {
        let mut upstream = McpUpstream::new("http://203.0.113.1:81/mcp");
        upstream.timeout = Duration::from_secs(12);
        let started = std::time::Instant::now();
        let err = RmcpBridge::connect(&upstream)
            .await
            .err()
            .expect("dial to a black-holed upstream must fail");
        assert!(
            started.elapsed() < Duration::from_secs(8),
            "dial was not bounded by connect_timeout: took {:?} ({err})",
            started.elapsed()
        );
    }

    /// The hand-written Debug impls are the only guard between a credential
    /// and the logs; pin them so a `#[derive(Debug)]` regression fails loudly
    /// for every secret-bearing variant.
    #[test]
    fn debug_redacts_every_credential_variant() {
        let oauth = McpAuth::OAuth2(OAuthClientConfig {
            client_id: "cid".into(),
            client_secret: "cs-LEAK".into(),
            token_url: "https://idp.example.com/token".into(),
            scopes: vec!["read".into()],
        });
        let rendered = format!(
            "{oauth:?} {:?} {:?}",
            McpAuth::ApiKey("key-LEAK".into()),
            McpAuth::Bearer("tok-LEAK".into())
        );
        assert!(
            !rendered.contains("LEAK"),
            "credential leaked into Debug output: {rendered}"
        );
        // The non-secret fields stay visible for operability.
        assert!(rendered.contains("idp.example.com"));
        assert!(rendered.contains("cid"));
    }

    #[test]
    fn sanitize_error_message_strips_controls_and_truncates() {
        let injected = "HTTP 401: bad\r\n[FAKE LOG LINE] evil";
        let cleaned = sanitize_error_message(injected);
        assert!(
            !cleaned.contains('\n') && !cleaned.contains('\r'),
            "{cleaned}"
        );
        assert!(cleaned.starts_with("HTTP 401"));

        let long = format!("HTTP 502: {}", "x".repeat(1000));
        let truncated = sanitize_error_message(&long);
        assert!(truncated.chars().count() <= 257, "bounded output");
        assert!(truncated.ends_with('…'));
    }
}
