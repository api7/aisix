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
}
