//! Typed errors for the A2A upstream client.
//!
//! Deliberately coarse for the first cut: the gateway only needs to
//! distinguish "could not reach the upstream agent" from "the request reached
//! it but failed" so the proxy layer can pick a client-visible status. Finer
//! mapping (per-JSON-RPC-error-code) lands when the `/a2a` endpoint wires A2A
//! traffic into the shared error-translation path.

/// Error surfaced by an [`crate::A2aBridge`].
#[derive(Debug, thiserror::Error)]
pub enum A2aError {
    /// The upstream agent could not be reached — DNS/TCP/TLS failure, timeout,
    /// or a non-success HTTP status when fetching the agent card. Non-retryable
    /// without operator action.
    #[error("failed to reach upstream A2A agent: {0}")]
    Connect(String),

    /// The upstream was reached but the request failed — a non-success HTTP
    /// status on the JSON-RPC call, or a malformed response body.
    #[error("upstream A2A request failed: {0}")]
    Request(String),
}
