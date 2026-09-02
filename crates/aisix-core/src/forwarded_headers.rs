//! Forwarding inbound client headers to an upstream.
//!
//! Three resources let an operator name headers that must reach the
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

/// Headers whose value is, or stands in for, a caller credential.
///
/// Forwardable — an internal upstream reading the end user's own
/// credential in the slot it already reads is what this capability is for
/// — and the one set a forwarded value may DISPLACE a gateway-injected
/// header in. Everything else the gateway put on the request it put there
/// to make the exchange work, so a forward leaves it alone.
pub const CREDENTIAL_SLOT_HEADERS: &[&str] = &[
    "api-key",             // Azure OpenAI key
    "authorization",       // OpenAI / Anthropic / Vertex Bearer
    "cookie",              // session credential
    "proxy-authorization", // proxy auth
    "x-api-key",           // Anthropic raw, also OpenAI legacy proxies
    "x-goog-api-key",      // Gemini API key
];

/// Trace-context headers, which a glob never sweeps in.
///
/// The caller's `traceparent` names a span in the CALLER's trace. Relayed
/// to an upstream it links that upstream's internal telemetry into the
/// caller's tracing backend and discloses the caller's trace ids to a
/// third party.
pub const TRACE_CONTEXT_HEADERS: &[&str] = &["traceparent", "tracestate"];

/// Whether `name` (already lowercase) needs a pattern that spells it out.
///
/// A `"*"` or `"x-*"` pattern is a statement about the operator's OWN
/// headers. It is not consent to hand a third-party provider the caller's
/// credential, nor to graft one tenant's trace onto that provider's
/// telemetry — both of which a broad glob would otherwise do the moment
/// the caller happened to send the header. Naming the header IS that
/// consent, and is honoured on every face.
///
/// This is also what keeps an already-configured broad pattern meaning
/// what it meant when it was written: a deployment carrying
/// `forward_client_headers: ["x-*"]` does not start relaying the caller's
/// `x-api-key` — which on `/v1/*` is the caller's own gateway key — the
/// day it upgrades.
///
/// The list here is the one every surface shares, and it is complete only
/// where the caller's credential always arrives in a slot it names — on
/// `/v1/*` and MCP the gateway reads `authorization` or `x-api-key` and
/// nothing else. A surface where the OPERATOR chooses the slot has to add
/// its own; see [`exact_match_only_with`].
pub fn exact_match_only(name: &str) -> bool {
    CREDENTIAL_SLOT_HEADERS.contains(&name) || TRACE_CONTEXT_HEADERS.contains(&name)
}

/// [`exact_match_only`] plus the slots one surface names for itself.
///
/// A `passthrough_route` picks its own header names: under `auth_mode:
/// header_key` the gateway credential arrives in the route's
/// `auth_header_name`, and `identity_header` carries an end-user identity
/// the route promises to record and strip. The route schema forbids MOST
/// of the shared credential names for those two fields, and the two it
/// does allow (`api-key`, `x-goog-api-key`) are on the shared list
/// already — so the union below is what makes the rule complete, and
/// without it a `["x-*"]` pattern sweeps in exactly the header the
/// gateway just consumed to authenticate
/// the caller and relays it upstream, where it can be replayed against
/// this gateway.
///
/// The rule is unchanged, only its input: a glob does not reach these
/// names, and a pattern that spells one out still forwards it.
pub fn exact_match_only_with(name: &str, surface_slots: &[&str]) -> bool {
    exact_match_only(name) || surface_slots.iter().any(|s| s.eq_ignore_ascii_case(name))
}

/// Whether a forwarded value may DISPLACE a header the gateway already
/// placed on the outbound request.
///
/// Only a credential slot: the operator named it on this upstream so the
/// caller's own credential is used instead of the gateway's, and two
/// credentials on the wire would let the upstream pick between them.
/// Everything else the gateway set, it set to make the exchange work — a
/// provider's async-mode or API-version selector, say — and a caller's
/// value there breaks the call rather than re-identifying it.
pub fn displaces_a_gateway_header(name: &str) -> bool {
    CREDENTIAL_SLOT_HEADERS.contains(&name)
}

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
/// snapshot. [`exact_match_only`] names the exception: those headers need
/// a pattern that spells them out.
pub fn forward_pattern_admits(patterns: &[String], name: &str) -> bool {
    forward_pattern_admits_with(patterns, name, &[])
}

/// [`forward_pattern_admits`] for a surface that names credential slots of
/// its own — see [`exact_match_only_with`].
pub fn forward_pattern_admits_with(
    patterns: &[String],
    name: &str,
    surface_slots: &[&str],
) -> bool {
    if exact_match_only_with(name, surface_slots) {
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
    fn credential_slots_are_forwardable_but_only_by_name() {
        // The capability's whole point: an internal upstream reading the
        // end user's own credential keeps reading it.
        for name in CREDENTIAL_SLOT_HEADERS {
            assert!(client_header_forwardable(name), "{name}");
            assert!(!header_forward_blocked(name), "{name}");
            assert!(forward_pattern_admits(&[name.to_string()], name), "{name}");
            // But a broad pattern is a statement about the operator's own
            // headers. On `/v1/*` the caller's `authorization` and
            // `x-api-key` carry its AISIX gateway key, and a deployment
            // that wrote `["x-*"]` before this rule existed must not start
            // relaying it to a third party on upgrade.
            assert!(!forward_pattern_admits(&["*".into()], name), "{name}");
            assert!(!forward_pattern_admits(&["x-*".into()], name), "{name}");
        }
    }

    #[test]
    fn only_a_credential_slot_displaces_a_gateway_header() {
        for name in CREDENTIAL_SLOT_HEADERS {
            assert!(displaces_a_gateway_header(name), "{name}");
        }
        // A provider's own wire-shape selectors are gateway decisions, not
        // identity: `x-dashscope-async` picks the submission mode and
        // `x-runway-version` the API revision the gateway then decodes.
        for name in ["x-dashscope-async", "x-runway-version", "anthropic-beta"] {
            assert!(!displaces_a_gateway_header(name), "{name}");
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
        // An ordinary header still answers to a glob — the exact-name
        // rule is the exception, not the policy.
        assert!(forward_pattern_admits(&["*".into()], "anthropic-beta"));
        assert!(forward_pattern_admits(&["x-trace-*".into()], "x-trace-id"));
        assert!(!forward_pattern_admits(&["x-trace-*".into()], "x-other"));
        // Header names are case-insensitive on the wire, so a pattern
        // written in the spelling the operator's own docs use still
        // matches the lowercase name the parser hands us.
        assert!(forward_pattern_admits(&["X-Trace-*".into()], "x-trace-id"));
        assert!(forward_pattern_admits(
            &["Anthropic-Beta".into()],
            "anthropic-beta"
        ));
    }

    /// `forward_pattern_admits` is the ONE predicate every face asks,
    /// `/passthrough/*` included — so the exact-name rule for credential
    /// and trace-context headers reaches the strip-override path too,
    /// where a wildcard would otherwise restore the credential the
    /// gateway just consumed to authenticate the caller.
    #[test]
    fn the_exact_name_rule_is_in_the_predicate_every_face_asks() {
        for name in CREDENTIAL_SLOT_HEADERS.iter().chain(TRACE_CONTEXT_HEADERS) {
            assert!(!forward_pattern_admits(&["*".into()], name), "{name}");
            assert!(
                forward_pattern_admits(&[name.to_uppercase()], name),
                "{name}"
            );
        }
    }

    /// The shared list cannot know the name a `passthrough_route` picked
    /// for its own gateway credential, and the route schema guarantees it
    /// is NOT one of the shared names. So a glob would sweep in exactly
    /// the header the gateway consumed to authenticate this caller.
    #[test]
    fn a_surface_slot_needs_its_own_name_not_a_glob() {
        let slots = ["x-gw-key", "x-end-user"];
        for name in slots {
            assert!(!exact_match_only(name), "{name}");
            assert!(exact_match_only_with(name, &slots), "{name}");
            assert!(!forward_pattern_admits_with(&["*".into()], name, &slots));
            assert!(!forward_pattern_admits_with(&["x-*".into()], name, &slots));
            // Naming it in full is still consent, on this face as on
            // every other.
            assert!(forward_pattern_admits_with(
                &[name.to_string()],
                name,
                &slots
            ));
            assert!(forward_pattern_admits_with(
                &[name.to_uppercase()],
                name,
                &slots
            ));
        }
        // Only the named slots move: an ordinary header on the same
        // surface still answers to the glob.
        assert!(forward_pattern_admits_with(
            &["x-*".into()],
            "x-trace-id",
            &slots
        ));
        // And a surface that names none behaves exactly as before —
        // whether it passes an empty list, or (as the passthrough call
        // site does, to keep a two-element list off the heap) a fixed
        // array with `""` standing for an unset slot. A header name is
        // never empty, on the wire or in the schema, so the sentinel
        // matches nothing and needs no filtering out.
        assert!(forward_pattern_admits_with(
            &["x-*".into()],
            "x-gw-key",
            &[]
        ));
        assert!(!exact_match_only_with("x-gw-key", &["", ""]));
        assert!(forward_pattern_admits_with(
            &["x-*".into()],
            "x-gw-key",
            &["", ""]
        ));
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
        let got = resolve_forwarded_client_headers(
            &["*".into(), "authorization".into()],
            &client,
            &["mcp-session-id"],
        );
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
