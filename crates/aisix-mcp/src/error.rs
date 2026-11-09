//! Typed errors for the MCP upstream client.
//!
//! These are deliberately coarse for the first cut: the gateway only needs to
//! distinguish "could not establish the session" from "the session is up but
//! the RPC failed" so the proxy layer can pick a client-visible status. Finer
//! mapping (per-JSON-RPC-error-code) lands when the `/mcp` endpoint wires MCP
//! traffic into the shared error-translation path.

/// Error surfaced by an [`crate::McpBridge`].
#[derive(Debug, thiserror::Error)]
pub enum McpError {
    /// The upstream MCP session could not be established — DNS/TCP/TLS
    /// failure, the `initialize` handshake was rejected, or the URL/auth is
    /// wrong. Non-retryable without operator action.
    #[error("failed to connect to upstream MCP server: {0}")]
    Connect(String),

    /// The session is established but an MCP request (`tools/list`,
    /// `tools/call`) failed at the protocol layer.
    #[error("upstream MCP request failed: {0}")]
    Request(String),

    /// An `oauth2_token_exchange` server could not obtain an upstream token
    /// for the caller.
    #[error("upstream token exchange failed: {0}")]
    TokenExchange(TokenExchangeError),

    /// An `oauth2_authorization_code` server has no usable token for the
    /// caller.
    #[error("upstream user authorization failed: {0}")]
    UserOAuth(UserOAuthError),

    /// As [`McpError::Request`], for an upstream that answered `401` after the
    /// session opened, which is worth telling apart when the caller's token
    /// can be refreshed.
    #[error("upstream MCP request failed: {0}")]
    Unauthorized(String),

    /// An upstream failure with a known cause, reported on the request's
    /// telemetry as a [`ToolCallFailure`]. `detail` holds the whole text for
    /// the gateway's logs and stays there.
    #[error("{detail}")]
    Upstream {
        failure: ToolCallFailure,
        detail: String,
    },
}

/// Why a `tools/call` failed upstream, as the request's telemetry reports it.
/// The caller's response is not derived from this. The message is fixed text:
/// what a tool or upstream said never goes into it, since the guardrails and
/// the exporter's content mode only govern the captured tool result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallFailure {
    /// The tool answered with `isError: true`.
    ToolError,
    /// The upstream answered with this HTTP status.
    UpstreamStatus(u16),
    Timeout,
    /// The upstream could not be reached, or dropped the connection.
    Transport,
    /// The upstream answered with this JSON-RPC error code. The MCP client drops
    /// the HTTP status of such an answer, so a non-2xx one reads the same way.
    JsonRpc(i32),
    /// The upstream answered with something the MCP client could not read.
    Undecodable,
    /// The credential for the upstream could not be obtained.
    CredentialUnavailable,
    /// The caller's MCP access rules do not permit the tool.
    PermissionDenied,
    /// The server does not have the tool.
    ToolNotFound,
}

impl ToolCallFailure {
    pub fn error_class(self) -> &'static str {
        match self {
            Self::ToolError => "tool_error",
            Self::UpstreamStatus(_) => "upstream_status",
            Self::Timeout => "timeout",
            Self::Transport => "transport",
            Self::JsonRpc(_) => "upstream_in_band",
            Self::Undecodable => "upstream_decode",
            Self::CredentialUnavailable => "credential_error",
            Self::PermissionDenied => "permission_denied",
            Self::ToolNotFound => "tool_not_found",
        }
    }

    pub fn error_message(self) -> String {
        match self {
            Self::ToolError => "MCP tool returned isError".to_string(),
            Self::UpstreamStatus(status) => {
                format!("upstream MCP server returned HTTP {status}")
            }
            Self::Timeout => "upstream MCP server timed out".to_string(),
            Self::Transport => "upstream MCP server connection failed".to_string(),
            Self::JsonRpc(code) => {
                format!("upstream MCP server returned JSON-RPC error {code}")
            }
            Self::Undecodable => "upstream MCP server returned an undecodable response".to_string(),
            Self::CredentialUnavailable => "MCP upstream credential unavailable".to_string(),
            Self::PermissionDenied => "MCP tool access denied".to_string(),
            Self::ToolNotFound => "MCP tool not found".to_string(),
        }
    }
}

impl From<UserOAuthFailure> for McpError {
    fn from(failure: UserOAuthFailure) -> Self {
        McpError::UserOAuth(UserOAuthError {
            failure,
            reached_upstream: false,
        })
    }
}

impl From<TokenExchangeFailure> for McpError {
    fn from(failure: TokenExchangeFailure) -> Self {
        McpError::TokenExchange(TokenExchangeError {
            failure,
            reached_upstream: false,
        })
    }
}

impl McpError {
    /// The telemetry cause of this failure, when it has a known one.
    pub fn tool_call_failure(&self) -> Option<ToolCallFailure> {
        match self {
            McpError::Upstream { failure, .. } => Some(*failure),
            McpError::Unauthorized(_) => Some(ToolCallFailure::UpstreamStatus(401)),
            McpError::TokenExchange(_) | McpError::UserOAuth(_) => {
                Some(ToolCallFailure::CredentialUnavailable)
            }
            _ => None,
        }
    }

    /// Mark a connect-time failure as the credential fetch's rather than the
    /// upstream's: the MCP server was not contacted.
    pub(crate) fn from_credential_fetch(self) -> Self {
        match self {
            McpError::Connect(_) => McpError::Upstream {
                failure: ToolCallFailure::CredentialUnavailable,
                detail: self.to_string(),
            },
            other => other,
        }
    }

    /// Mark a token-exchange failure as one that followed upstream contact:
    /// the upstream rejected the previous token with a `401` and it was the
    /// re-exchange that failed.
    pub(crate) fn after_upstream_contact(self) -> Self {
        match self {
            McpError::TokenExchange(error) => McpError::TokenExchange(TokenExchangeError {
                reached_upstream: true,
                ..error
            }),
            other => other,
        }
    }
}

/// A failed token exchange and whether the operation that hit it had
/// already contacted the upstream server.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{failure}")]
pub struct TokenExchangeError {
    pub failure: TokenExchangeFailure,
    pub reached_upstream: bool,
}

/// Why an on-behalf-of token exchange produced no upstream token. Each
/// kind maps to one client-visible status on the single-server endpoint.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TokenExchangeFailure {
    /// The caller was not authenticated with a token the gateway verified
    /// through a JWT trust provider, so there is nothing to exchange.
    #[error("the caller presented no verified token to exchange")]
    NoSubject,
    /// The identity provider refused the caller's token. `claims` is the
    /// provider's step-up challenge when it sent one: what the caller must
    /// satisfy at the provider before retrying.
    #[error("the identity provider rejected the caller's token")]
    SubjectRejected { claims: Option<String> },
    /// The identity provider refused the gateway's own client registration
    /// or requested scope, or the server's exchange settings are incomplete.
    #[error("the gateway's token exchange configuration was not accepted")]
    Misconfigured,
    /// The identity provider could not be reached, failed, throttled or
    /// timed out the request, or answered without a usable access token.
    #[error("the identity provider did not return a usable token")]
    Unavailable,
}

/// A failed attempt to use a caller's stored upstream token, and whether the
/// operation that hit it had already contacted the upstream server.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{failure}")]
pub struct UserOAuthError {
    pub failure: UserOAuthFailure,
    pub reached_upstream: bool,
}

/// Why an `oauth2_authorization_code` server cannot be called for a caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum UserOAuthFailure {
    /// The caller's credential names no user to keep a token for.
    #[error("the caller's credential has no user")]
    NoSubject,
    /// The caller has no usable token, and signing in would give it one.
    #[error("the caller has not signed in to this server")]
    AuthorizationRequired,
    /// The gateway cannot run the flow for this server as configured.
    #[error("the server's OAuth settings cannot be used")]
    Misconfigured,
    /// Storage or the upstream authorization server could not answer. The
    /// caller's connection, if it has one, is intact.
    #[error("the caller's upstream token is temporarily unavailable")]
    Unavailable,
}
