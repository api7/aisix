//! Inbound authority shared by URL rewriting and passthrough routing.

use axum::extract::Request;
use axum::http::header;

/// The URI authority takes precedence over `Host` for absolute-form requests
/// (RFC 9112 section 3.2.2). Origin-form requests use `Host`.
/// Lowercased, `:port` stripped.
pub(crate) fn inbound_host(req: &Request) -> Option<String> {
    let raw = req.uri().authority().map(|a| a.as_str()).or_else(|| {
        req.headers()
            .get(header::HOST)
            .and_then(|v| v.to_str().ok())
    })?;
    let no_port = raw.rsplit_once(':').map_or(raw, |(head, port)| {
        // Only treat the suffix as a port when it is all digits — an
        // IPv6 literal's last group would otherwise be truncated.
        if port.chars().all(|c| c.is_ascii_digit()) {
            head
        } else {
            raw
        }
    });
    Some(no_port.trim_end_matches('.').to_ascii_lowercase())
}
