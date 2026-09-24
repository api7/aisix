//! Shared `reqwest::Client` for direct HTTP calls (messages, audio, etc.).
//!
//! Initialised lazily once and reused across all calls so the connection
//! pool is shared and we don't pay TLS handshake cost on every request.
//! Connection-layer settings come from `aisix_gateway::upstream_http`, the
//! same source the provider bridges use — this client talks to the same
//! upstreams, so it must expire pooled connections on the same schedule.

use aisix_core::models::provider_key::UpstreamConnection;
use reqwest::Client;
use std::sync::OnceLock;

/// Returns the process-wide shared HTTP client.
pub fn client() -> &'static Client {
    static CLIENT: OnceLock<Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        aisix_gateway::client_builder()
            .build()
            .unwrap_or_else(|_| Client::new())
    })
}

/// The client for a call dispatched on behalf of one Provider Key.
///
/// Returns a clone of the shared client whenever the key sets no
/// connection override, which is every key that names neither a private
/// CA nor a resolution address — so the ordinary path keeps sharing one
/// connection pool.
///
/// Every passthrough surface goes through here rather than [`client`]:
/// a key configured with a private CA, or with an upstream reachable only
/// at a fixed address, has to reach its endpoint on `/v1/messages`,
/// `/v1/responses`, `/v1/audio/*`, `/v1/videos/*`, the jobs surface and
/// the raw tunnel, not only on the endpoints that run through a provider
/// bridge.
pub fn client_for(conn: Option<&UpstreamConnection>) -> Client {
    aisix_gateway::upstream_tls::client_for_provider_key(client(), conn)
}

#[cfg(test)]
mod tests {
    /// The passthrough surfaces are a family — `/v1/messages`,
    /// `/v1/responses`, `count_tokens`, rerank, audio, videos, the jobs
    /// surface, the raw tunnel — and a new one added on [`client`]
    /// instead of [`client_for`] fails in exactly one way: the Provider
    /// Key's private CA and resolution address are ignored on that
    /// endpoint only, while every other endpoint for the same key keeps
    /// working.
    ///
    /// (#471 and #715 are the same lesson twice: a per-request mechanism
    /// wired into one member of this family and silently missing from the
    /// rest.)
    #[test]
    fn no_dispatch_site_uses_the_shared_client_directly() {
        let src_dir = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src"));
        let mut offenders = Vec::new();
        for entry in std::fs::read_dir(src_dir).expect("read src") {
            let path = entry.expect("dir entry").path();
            if path.extension().is_none_or(|e| e != "rs") {
                continue;
            }
            // This module defines both, and `client_for` is built on top
            // of `client`.
            if path.file_name().is_some_and(|n| n == "http_client.rs") {
                continue;
            }
            let src = std::fs::read_to_string(&path).expect("read source");
            let production = match src.find("\n#[cfg(test)]\nmod ") {
                Some(i) => &src[..i],
                None => &src[..],
            };
            for (n, line) in production.lines().enumerate() {
                if line.contains("http_client::client()") {
                    offenders.push(format!("{}:{}", path.display(), n + 1));
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "these dispatch on the shared client, so the Provider Key's \
             connection overrides are ignored on that endpoint; use \
             `http_client::client_for(pk.upstream_connection().as_ref())`:\n{}",
            offenders.join("\n"),
        );
    }
}
