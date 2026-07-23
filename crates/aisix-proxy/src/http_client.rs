//! Shared `reqwest::Client` for direct HTTP calls (messages, audio, etc.).
//!
//! Initialised lazily once and reused across all calls so the connection
//! pool is shared and we don't pay TLS handshake cost on every request.
//! Connection-layer settings come from `aisix_gateway::upstream_http`, the
//! same source the provider bridges use — this client talks to the same
//! upstreams, so it must expire pooled connections on the same schedule.

use reqwest::Client;
use std::sync::OnceLock;

/// Returns the process-wide shared HTTP client.
pub fn client() -> &'static Client {
    static CLIENT: OnceLock<Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        aisix_gateway::client_builder()
            .user_agent("aisix/0.1")
            .build()
            .unwrap_or_else(|_| Client::new())
    })
}
