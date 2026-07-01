//! Downstream client attribution for usage logs (#492).
//!
//! Resolves the real client IP and the `User-Agent` once per request and
//! exposes them via the [`ClientContext`] extractor — the same low-churn
//! `FromRequestParts` pattern handlers already use for `AuthenticatedKey`.
//!
//! IP resolution mirrors nginx `set_real_ip_from` + `real_ip_recursive`:
//! the immediate TCP peer is the client unless it's a configured trusted
//! proxy, in which case the configured forwarded header (default
//! `x-forwarded-for`) is walked to find the originating address. With no
//! trusted proxies configured (the default) the peer is always logged.

use std::net::{IpAddr, SocketAddr};

use aisix_core::config::RealIpConfig;
use axum::extract::{ConnectInfo, FromRef, FromRequestParts};
use axum::http::request::Parts;
use ipnet::IpNet;

use crate::state::ProxyState;

/// Pre-parsed `proxy.real_ip` config carried on [`ProxyState`] so the
/// per-request extractor doesn't re-parse CIDRs on the hot path.
#[derive(Debug, Clone, Default)]
pub struct ResolvedRealIp {
    pub trusted: Vec<IpNet>,
    pub recursive: bool,
    pub header: String,
}

impl ResolvedRealIp {
    /// Build from validated config. CIDRs are already validated at config
    /// load (`Config::validate`); a malformed entry here degrades to
    /// "trust nothing" rather than panicking on the request path.
    pub fn from_config(cfg: &RealIpConfig) -> Self {
        Self {
            trusted: cfg.parse_trusted().unwrap_or_default(),
            recursive: cfg.recursive,
            header: cfg.header.clone(),
        }
    }
}

/// nginx `set_real_ip_from` + `real_ip_recursive` equivalent.
///
/// - `peer`      – TCP peer address (from `ConnectInfo`).
/// - `forwarded` – parsed forwarded-header list, left-to-right as received.
/// - `trusted`   – pre-parsed trusted-proxy CIDRs.
/// - `recursive` – nginx `real_ip_recursive` on/off.
pub fn resolve_client_ip(
    peer: IpAddr,
    forwarded: &[IpAddr],
    trusted: &[IpNet],
    recursive: bool,
) -> IpAddr {
    let is_trusted = |ip: &IpAddr| trusted.iter().any(|n| n.contains(ip));
    // nginx only rewrites $remote_addr when the connection itself comes
    // from a trusted proxy. An untrusted peer IS the client.
    if !is_trusted(&peer) {
        return peer;
    }
    if recursive {
        // Walk right-to-left; the first untrusted address is the client.
        for ip in forwarded.iter().rev() {
            if !is_trusted(ip) {
                return *ip;
            }
        }
        // Every forwarded entry trusted (or list empty): leftmost, else peer.
        forwarded.first().copied().unwrap_or(peer)
    } else {
        // Non-recursive: the rightmost forwarded entry (the address
        // immediately upstream of the trusted peer). Empty list → peer.
        forwarded.last().copied().unwrap_or(peer)
    }
}

/// Parse the configured forwarded header into a left-to-right IP list.
/// Concatenates every header field instance (a client may send several)
/// in received order, splits each on `,`, trims whitespace, strips an
/// optional `:port` (or `[v6]:port`) suffix, and skips tokens that don't
/// parse as an IP. Using `get_all` (not `get`) so a spoofed extra
/// `x-forwarded-for` field can't hide entries from the trusted-proxy walk.
fn parse_forwarded(headers: &axum::http::HeaderMap, header: &str) -> Vec<IpAddr> {
    headers
        .get_all(header)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|raw| raw.split(','))
        .filter_map(|tok| parse_forwarded_token(tok.trim()))
        .collect()
}

fn parse_forwarded_token(tok: &str) -> Option<IpAddr> {
    if tok.is_empty() {
        return None;
    }
    // Bracketed IPv6, optionally with a port: `[::1]` or `[::1]:443`.
    if let Some(rest) = tok.strip_prefix('[') {
        let inner = rest.split(']').next().unwrap_or("");
        return inner.parse().ok();
    }
    // Bare address first; fall back to stripping a single `:port` (IPv4
    // or hostname:port form). A bare IPv6 contains multiple colons and
    // parses directly, so only strip when there's exactly one colon.
    if let Ok(ip) = tok.parse::<IpAddr>() {
        return Some(ip);
    }
    if tok.matches(':').count() == 1 {
        if let Some((host, _port)) = tok.rsplit_once(':') {
            return host.parse().ok();
        }
    }
    None
}

/// Header carrying request-scoped routing tags for tag/metadata-conditional
/// routing (comma-separated). Read out-of-band from the request headers so the
/// tags never reach the upstream request body.
pub const ROUTING_TAGS_HEADER: &str = "x-aisix-routing-tags";

/// Per-request client attribution. Resolved once via the extractor and
/// threaded into the usage event by each handler's emit fn.
#[derive(Debug, Clone, Default)]
pub struct ClientContext {
    pub source_ip: String,
    pub user_agent: String,
    /// Routing tags from [`ROUTING_TAGS_HEADER`], used to select among a
    /// routing model's tagged targets. Empty when the header is absent.
    pub routing_tags: Vec<String>,
}

#[axum::async_trait]
impl<S> FromRequestParts<S> for ClientContext
where
    S: Send + Sync,
    ProxyState: FromRef<S>,
{
    // Never reject: a missing peer / User-Agent degrades to empty rather
    // than failing the request (matches the wire's omit-when-empty
    // semantics and keeps oneshot tests without ConnectInfo green).
    type Rejection = std::convert::Infallible;

    async fn from_request_parts(parts: &mut Parts, state: &S) -> Result<Self, Self::Rejection> {
        let proxy_state = ProxyState::from_ref(state);
        let cfg = &proxy_state.real_ip;

        let source_ip = parts
            .extensions
            .get::<ConnectInfo<SocketAddr>>()
            .map(|ci| ci.0.ip())
            .map(|peer| {
                let forwarded = parse_forwarded(&parts.headers, &cfg.header);
                resolve_client_ip(peer, &forwarded, &cfg.trusted, cfg.recursive).to_string()
            })
            .unwrap_or_default();

        let user_agent = parts
            .headers
            .get(axum::http::header::USER_AGENT)
            .and_then(|v| v.to_str().ok())
            .map(|s| crate::chat::sanitize_tag(s.to_string()))
            .unwrap_or_default();

        let routing_tags = parts
            .headers
            .get(ROUTING_TAGS_HEADER)
            .and_then(|v| v.to_str().ok())
            .map(parse_routing_tags)
            .unwrap_or_default();

        Ok(ClientContext {
            source_ip,
            user_agent,
            routing_tags,
        })
    }
}

/// Split a comma-separated routing-tags header into trimmed, non-empty tags.
fn parse_routing_tags(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(str::to_owned)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nets(cidrs: &[&str]) -> Vec<IpNet> {
        cidrs.iter().map(|s| s.parse().unwrap()).collect()
    }
    fn ip(s: &str) -> IpAddr {
        s.parse().unwrap()
    }

    #[test]
    fn parse_routing_tags_splits_trims_and_drops_empties() {
        assert_eq!(
            parse_routing_tags("eu, premium ,,us"),
            vec!["eu", "premium", "us"]
        );
        assert!(parse_routing_tags("").is_empty());
        assert!(parse_routing_tags("  ,  ").is_empty());
    }

    #[test]
    fn untrusted_peer_is_the_client_and_xff_is_ignored() {
        let peer = ip("203.0.113.9");
        let fwd = [ip("1.2.3.4")];
        let trusted = nets(&["10.0.0.0/8"]);
        assert_eq!(resolve_client_ip(peer, &fwd, &trusted, true), peer);
    }

    #[test]
    fn trusted_peer_recursive_returns_first_untrusted_from_right() {
        // XFF as received: client, edge, internal-lb. peer = internal-lb.
        let peer = ip("10.0.0.1");
        let fwd = [ip("203.0.113.7"), ip("10.0.0.2"), ip("10.0.0.3")];
        let trusted = nets(&["10.0.0.0/8"]);
        assert_eq!(
            resolve_client_ip(peer, &fwd, &trusted, true),
            ip("203.0.113.7")
        );
    }

    #[test]
    fn trusted_peer_non_recursive_returns_rightmost_entry() {
        let peer = ip("10.0.0.1");
        let fwd = [ip("203.0.113.7"), ip("198.51.100.4")];
        let trusted = nets(&["10.0.0.0/8"]);
        assert_eq!(
            resolve_client_ip(peer, &fwd, &trusted, false),
            ip("198.51.100.4")
        );
    }

    #[test]
    fn trusted_peer_all_forwarded_trusted_recursive_falls_back_to_leftmost() {
        let peer = ip("10.0.0.1");
        let fwd = [ip("10.0.0.9"), ip("10.0.0.8")];
        let trusted = nets(&["10.0.0.0/8"]);
        assert_eq!(
            resolve_client_ip(peer, &fwd, &trusted, true),
            ip("10.0.0.9")
        );
    }

    #[test]
    fn trusted_peer_empty_forwarded_falls_back_to_peer() {
        let peer = ip("10.0.0.1");
        let trusted = nets(&["10.0.0.0/8"]);
        assert_eq!(resolve_client_ip(peer, &[], &trusted, true), peer);
        assert_eq!(resolve_client_ip(peer, &[], &trusted, false), peer);
    }

    #[test]
    fn ipv6_peer_and_forwarded_resolve() {
        let peer = ip("::1");
        let fwd = [ip("2001:db8::1")];
        let trusted = nets(&["::1/128"]);
        assert_eq!(
            resolve_client_ip(peer, &fwd, &trusted, true),
            ip("2001:db8::1")
        );
    }

    #[test]
    fn header_parser_handles_whitespace_ports_and_garbage() {
        let mut h = axum::http::HeaderMap::new();
        h.insert(
            "x-forwarded-for",
            "203.0.113.7:1234, garbage, 198.51.100.4 , [2001:db8::1]:443"
                .parse()
                .unwrap(),
        );
        let parsed = parse_forwarded(&h, "x-forwarded-for");
        assert_eq!(
            parsed,
            vec![ip("203.0.113.7"), ip("198.51.100.4"), ip("2001:db8::1")]
        );
    }

    #[test]
    fn header_parser_absent_header_is_empty() {
        let h = axum::http::HeaderMap::new();
        assert!(parse_forwarded(&h, "x-forwarded-for").is_empty());
    }

    #[test]
    fn header_parser_concatenates_multiple_header_fields_in_order() {
        // A client may send several x-forwarded-for fields; all must be
        // parsed in received order so a spoofed extra field can't hide
        // entries from the trusted-proxy walk.
        let mut h = axum::http::HeaderMap::new();
        h.append("x-forwarded-for", "203.0.113.7, 10.0.0.1".parse().unwrap());
        h.append("x-forwarded-for", "10.0.0.2".parse().unwrap());
        let parsed = parse_forwarded(&h, "x-forwarded-for");
        assert_eq!(
            parsed,
            vec![ip("203.0.113.7"), ip("10.0.0.1"), ip("10.0.0.2")]
        );
    }
}
