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
    TokenExchange(TokenExchangeFailure),
}

/// Why an on-behalf-of token exchange produced no upstream token. Each
/// kind maps to one client-visible status on the single-server endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TokenExchangeFailure {
    /// The caller was not authenticated with a token the gateway verified
    /// through a JWT trust provider, so there is nothing to exchange.
    #[error("the caller presented no verified token to exchange")]
    NoSubject,
    /// The identity provider refused the caller's token.
    #[error("the identity provider rejected the caller's token")]
    SubjectRejected,
    /// The identity provider refused the gateway's own client registration
    /// or requested scope, or the server's exchange settings are incomplete.
    #[error("the gateway's token exchange configuration was not accepted")]
    Misconfigured,
    /// The identity provider could not be reached, failed, or answered
    /// without a usable access token.
    #[error("the identity provider did not return a usable token")]
    Unavailable,
}
