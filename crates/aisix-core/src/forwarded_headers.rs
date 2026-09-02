//! Forwarding inbound client headers to an upstream.
//!
//! Four resources let an operator name headers that must reach the
//! upstream — `provider_key`'s `request.forward_client_headers`, and the
//! same field on `passthrough_route` and `mcp_server` — because the
//! gateway fronts upstreams on three different surfaces and the choice is
//! made per upstream, never gateway-wide: the same deployment reaches
//! internal services that need the caller's own credentials and claims AND
//! public model providers that must never see them.
//!
//! The rule lives here rather than at each call site so every surface
//! answers "may this header be forwarded, and does the pattern name it"
//! the same way — the drift this repo's handler-family rule warns about.
//!
//! ## What is blocked, and why only this much
//!
//! Two tiers, and both are deliberately narrow. An operator naming a
//! header on a specific upstream has made a decision about that upstream;
//! a list that second-guesses it just means the capability does not exist
//! for the deployments that need it most (an internal service behind the
//! gateway that authorizes on the end user's own `Authorization`).
//!
//! 1. [`header_forward_blocked`] — headers whose forwarding breaks the
//!    exchange itself ([`NON_FORWARDABLE_HEADERS`]: `host`, and RFC 9110
//!    §7.6.1 hop-by-hop headers, which describe the CALLER's connection and
//!    not the one the gateway opens), or forges a gateway assertion (the
//!    [`GATEWAY_HEADER_PREFIX`] namespace). Absolute, on every surface.
//! 2. [`client_header_forwardable`] — additionally, on the surfaces that
//!    rebuild the outbound message ([`NEVER_FORWARD_FROM_CLIENT`]), the
//!    headers that describe a body this gateway re-serializes or a response
//!    shape it parses. `/passthrough/*` relays the body verbatim and so
//!    does NOT apply this second tier.
//!
//! Credential slots — `authorization`, `x-api-key`, `api-key`,
//! `x-goog-api-key`, `proxy-authorization`, `cookie` — are in neither tier
//! on purpose. Handing an internal upstream the end user's own credential,
//! in the slot that upstream already reads, is the capability's whole
//! point, and the surface that injects a gateway credential into the same
//! slot stands aside for it (see each dispatch site).

use http::{HeaderMap, HeaderName, HeaderValue};

use crate::wildcard::wildcard_matches;

/// Headers no configuration may put on an upstream request, on any surface.
///
/// Every entry describes the MESSAGE's transport rather than its content:
/// `host` selects which server the request reaches at all, and the rest are
/// RFC 9110 §7.6.1 hop-by-hop headers, which describe the connection the
/// CALLER opened to this gateway and not the one the gateway opens to the
/// upstream. Relaying either corrupts the exchange rather than changing who
/// the request claims to be from.
pub const NON_FORWARDABLE_HEADERS: &[&str] = &[
    "connection",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
];

/// Prefix of the gateway's own header namespace.
///
/// `x-aisix-request-id` and its siblings are assertions the gateway makes
/// about a request it handled. Forwarding a caller's copy would let the
/// caller forge them upstream, and would lose the assertion itself.
pub const GATEWAY_HEADER_PREFIX: &str = "x-aisix-";

/// Client headers never forwarded by a surface that REBUILDS the outbound
/// message, on top of [`NON_FORWARDABLE_HEADERS`].
///
/// The content and negotiation entries describe a body this gateway
/// re-serializes and a response shape it parses, so relaying the caller's
/// copies would describe the wrong message. `set-cookie` is a response
/// header with no meaning on a request. `anthropic-version` selects the
/// wire format an Anthropic-shaped upstream answers in, which the gateway
/// then decodes — a caller's value there breaks the decode.
pub const NEVER_FORWARD_FROM_CLIENT: &[&str] = &[
    "accept",
    "accept-encoding",
    "anthropic-version",
    "content-encoding",
    "content-length",
    "content-type",
    "expect",
    "set-cookie",
];

/// Client header prefixes never forwarded by a surface that rebuilds the
/// outbound message.
///
/// `x-stainless-*` is the calling SDK's self-description; relaying one
/// SDK's version headers to a provider that reads them for its own SDK
/// breaks the call. Mainstream gateways exclude it from their forwarding
/// for the same reason.
pub const NEVER_FORWARD_FROM_CLIENT_PREFIXES: &[&str] = &["x-stainless-"];

/// Headers a glob never sweeps in — only a pattern naming them exactly.
///
/// The caller's `traceparent` names a span in the CALLER's trace. Relayed
/// to an upstream it links that upstream's internal telemetry into the
/// caller's tracing backend and discloses the caller's trace ids to a third
/// party. An operator fronting a first-party service may well want exactly
/// that continuity, so naming the header is honoured; a `"*"` or `"x-*"`
/// pattern is a decision about the operator's OWN headers and is not read
/// as consent to graft one tenant's trace onto another party's.
pub const EXACT_MATCH_ONLY_HEADERS: &[&str] = &["traceparent", "tracestate"];

/// Whether `name` (already lowercase) may never reach an upstream, on any
/// surface and through any configuration path.
pub fn header_forward_blocked(name: &str) -> bool {
    NON_FORWARDABLE_HEADERS.contains(&name) || name.starts_with(GATEWAY_HEADER_PREFIX)
}

/// Whether a client's copy of `name` (already lowercase) may be forwarded
/// by a surface that rebuilds the outbound message — the standard `/v1/*`
/// endpoints and the MCP faces.
pub fn client_header_forwardable(name: &str) -> bool {
    !header_forward_blocked(name)
        && !NEVER_FORWARD_FROM_CLIENT.contains(&name)
        && !NEVER_FORWARD_FROM_CLIENT_PREFIXES
            .iter()
            .any(|p| name.starts_with(p))
}

/// Whether an operator allowlist admits a header name.
///
/// Patterns are matched case-insensitively against the lowercase name, and
/// a single `*` glob is supported (`"x-trace-*"`), matching how
/// `allowed_models` / `allowed_agents` patterns behave elsewhere in the
/// snapshot. [`EXACT_MATCH_ONLY_HEADERS`] are the one exception: they need
/// a pattern that spells them out.
pub fn forward_pattern_admits(patterns: &[String], name: &str) -> bool {
    if EXACT_MATCH_ONLY_HEADERS.contains(&name) {
        return patterns.iter().any(|p| p.eq_ignore_ascii_case(name));
    }
    patterns
        .iter()
        .any(|p| wildcard_matches(&p.to_ascii_lowercase(), name))
}

/// The client headers `patterns` forwards out of `client`, for a surface
/// that rebuilds the outbound message.
///
/// `extra_blocked` carries the names one surface owns that the shared
/// lists cannot know about (the MCP protocol slots, which the upstream
/// transport rejects outright when they carry a foreign value).
///
/// A repeated header (`anthropic-beta: a` twice) forwards its first value
/// only: the upstream sees one well-formed header rather than a list this
/// gateway never interpreted.
pub fn resolve_forwarded_client_headers(
    patterns: &[String],
    client: &HeaderMap,
    extra_blocked: &[&str],
) -> Vec<(HeaderName, HeaderValue)> {
    if patterns.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for name in client.keys() {
        // `HeaderName` is already lowercase on the wire-parsed side.
        let lower = name.as_str();
        if !client_header_forwardable(lower)
            || extra_blocked.contains(&lower)
            || !forward_pattern_admits(patterns, lower)
        {
            continue;
        }
        let Some(value) = client.get(name) else {
            continue;
        };
        let mut value = value.clone();
        // A forwarded credential must not be echoed by a `Debug` of the
        // outbound map. Marking every forwarded value is cheaper than
        // deciding per name which ones are secrets.
        value.set_sensitive(true);
        out.push((name.clone(), value));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut m = HeaderMap::new();
        for (k, v) in pairs {
            m.insert(
                HeaderName::try_from(*k).unwrap(),
                HeaderValue::from_str(v).unwrap(),
            );
        }
        m
    }

    fn names(pairs: &[(HeaderName, HeaderValue)]) -> Vec<&str> {
        pairs.iter().map(|(n, _)| n.as_str()).collect()
    }

    #[test]
    fn credential_slots_are_forwardable() {
        // The capability's whole point: an internal upstream reading the
        // end user's own credential keeps reading it.
        for name in [
            "authorization",
            "proxy-authorization",
            "x-api-key",
            "api-key",
            "x-goog-api-key",
            "cookie",
        ] {
            assert!(client_header_forwardable(name), "{name}");
            assert!(!header_forward_blocked(name), "{name}");
        }
    }

    #[test]
    fn transport_slots_are_blocked_on_every_surface() {
        for name in NON_FORWARDABLE_HEADERS {
            assert!(header_forward_blocked(name), "{name}");
            assert!(!client_header_forwardable(name), "{name}");
        }
        assert!(header_forward_blocked("x-aisix-request-id"));
        assert!(header_forward_blocked("x-aisix-anything"));
        assert!(!header_forward_blocked("x-user-jwt"));
    }

    #[test]
    fn framing_is_blocked_only_where_the_message_is_rebuilt() {
        // Tier two: `/passthrough/*` relays the body verbatim and asks
        // `header_forward_blocked` alone, so `content-type` survives there.
        assert!(!client_header_forwardable("content-type"));
        assert!(!header_forward_blocked("content-type"));
        assert!(!client_header_forwardable("x-stainless-lang"));
        assert!(!client_header_forwardable("anthropic-version"));
        assert!(client_header_forwardable("anthropic-beta"));
    }

    #[test]
    fn trace_context_needs_its_own_name_not_a_glob() {
        assert!(!forward_pattern_admits(&["*".into()], "traceparent"));
        assert!(!forward_pattern_admits(&["trace*".into()], "tracestate"));
        assert!(forward_pattern_admits(
            &["traceparent".into()],
            "traceparent"
        ));
        assert!(forward_pattern_admits(
            &["TraceParent".into()],
            "traceparent"
        ));
        // Everything else still answers to a glob.
        assert!(forward_pattern_admits(&["*".into()], "authorization"));
        assert!(forward_pattern_admits(&["x-trace-*".into()], "x-trace-id"));
        assert!(!forward_pattern_admits(&["x-trace-*".into()], "x-other"));
    }

    #[test]
    fn an_empty_allowlist_forwards_nothing() {
        let client = map(&[("authorization", "Bearer caller"), ("x-trace-id", "t")]);
        assert!(resolve_forwarded_client_headers(&[], &client, &[]).is_empty());
    }

    #[test]
    fn resolution_applies_both_tiers_and_the_surface_list() {
        let client = map(&[
            ("authorization", "Bearer caller"),
            ("content-type", "application/json"),
            ("host", "gateway.internal"),
            ("x-aisix-request-id", "forged"),
            ("mcp-session-id", "s1"),
            ("x-trace-id", "t"),
        ]);
        let got = resolve_forwarded_client_headers(&["*".into()], &client, &["mcp-session-id"]);
        let mut got = names(&got);
        got.sort_unstable();
        assert_eq!(got, vec!["authorization", "x-trace-id"]);
    }

    #[test]
    fn forwarded_values_are_marked_sensitive() {
        let client = map(&[("authorization", "Bearer caller")]);
        let got = resolve_forwarded_client_headers(&["authorization".into()], &client, &[]);
        assert!(got[0].1.is_sensitive());
    }
}
