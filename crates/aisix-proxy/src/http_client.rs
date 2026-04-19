//! Shared `reqwest::Client` for direct HTTP calls (messages, audio, etc.).
//!
//! Initialised lazily once and reused across all calls so the connection
//! pool is shared and we don't pay TLS handshake cost on every request.

use reqwest::Client;
use std::sync::OnceLock;

/// Returns the process-wide shared HTTP client.
pub fn client() -> &'static Client {
    static CLIENT: OnceLock<Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        Client::builder()
            .user_agent("aisix/0.1")
            .build()
            .unwrap_or_else(|_| Client::new())
    })
}
