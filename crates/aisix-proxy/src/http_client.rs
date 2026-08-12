//! Shared `reqwest::Client` for direct HTTP calls (messages, audio, etc.).
//!
//! Initialised lazily once and reused across all calls so the connection
//! pool is shared and we don't pay TLS handshake cost on every request.
//! Connection-layer settings come from `aisix_gateway::upstream_http`, the
//! same source the provider bridges use — this client talks to the same
//! upstreams, so it must expire pooled connections on the same schedule.

use aisix_core::models::provider_key::ProviderKeyTls;
use reqwest::Client;
use std::sync::OnceLock;

/// Returns the process-wide shared HTTP client.
pub fn client() -> &'static Client {
    static CLIENT: OnceLock<Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        aisix_gateway::dispatch_client_builder()
            .user_agent("aisix/0.1")
            .build()
            .unwrap_or_else(|_| aisix_gateway::dispatch_client_fallback())
    })
}

/// The client for a call dispatched on behalf of one Provider Key.
///
/// Returns a clone of the shared client whenever the key sets no `tls`
/// override, which is every key that does not name a private CA — so the
/// ordinary path keeps sharing one connection pool.
///
/// Every passthrough surface goes through here rather than [`client`]:
/// a key configured with a private CA has to reach its endpoint on
/// `/v1/messages`, `/v1/responses`, `/v1/audio/*`, `/v1/videos/*`, the
/// jobs surface and the raw tunnel, not only on the endpoints that run
/// through a provider bridge.
pub fn client_for(tls: Option<&ProviderKeyTls>) -> Client {
    aisix_gateway::upstream_tls::client_for_provider_key(client(), tls)
}

#[cfg(test)]
mod tests {
    /// The passthrough surfaces are a family — `/v1/messages`,
    /// `/v1/responses`, `count_tokens`, rerank, audio, videos, the jobs
    /// surface, the raw tunnel — and a new one added on [`client`]
    /// instead of [`client_for`] fails in exactly one way: the Provider
    /// Key's private CA is ignored on that endpoint only, while every
    /// other endpoint for the same key keeps working.
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
            "these dispatch on the shared client, so `provider_key.tls` is \
             ignored on that endpoint; use `http_client::client_for(pk.tls.as_ref())`:\n{}",
            offenders.join("\n"),
        );
    }
}
