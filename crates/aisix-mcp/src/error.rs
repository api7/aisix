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
