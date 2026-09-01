//! The one place outbound headers that are **not** owned by the gateway get
//! added to a standard-protocol upstream request.
//!
//! Two operator-facing features share this pipeline, in this order:
//!
//! 1. `request.default_headers` — static headers the ProviderKey injects,
//!    with `${...}` request-context variables rendered per request
//!    (AISIX-Cloud#1112).
//! 2. `request.forward_client_headers` — inbound client headers matching an
//!    operator-configured allowlist, forwarded verbatim
//!    (AISIX-Cloud#1167).
//!
//! **Precedence is gateway > operator > client, and it is structural.** Every
//! caller inserts its own bridge-owned headers (auth, `content-type`,
//! `x-aisix-request-id`, streaming `accept`) into the `HeaderMap` *first*, and
//! nothing here overwrites a name that is already present. Within this module
//! `default_headers` is resolved before the client allowlist, so an operator
//! header wins over a client header of the same name. On top of that, the
//! [`RESERVED_UPSTREAM_HEADERS`] / [`NEVER_FORWARD_HEADERS`] guards drop the
//! auth and transport families outright, so neither an operator typo nor a
//! `"*"`-happy allowlist can reach them.
//!
//! Before #1167 the standard endpoints rebuilt the outbound header set from
//! scratch and dropped every client header — the default this module
//! preserves. `/passthrough/*` is the opposite policy (forward everything
//! except `pk.strip_headers`) and does not use this pipeline.

use std::collections::HashSet;

use aisix_core::{wildcard::wildcard_matches, HeaderVars, RequestOverrides};
use http::{
    header::{HeaderName, HeaderValue},
    HeaderMap,
};

/// Headers an operator's `default_headers` block may never set, and that are
/// never forwarded from a client.
///
/// Re-exported from `aisix-core`, which owns the list so that
/// `Config::validate` can reject the same names in
/// `proxy.request_id.accept_headers` — reading a request id out of
/// `authorization` would copy the caller's credential into a response header,
/// the logs, and the upstream request, walking straight around this guard.
pub use aisix_core::RESERVED_UPSTREAM_HEADERS;

/// Client headers never forwarded, on top of [`RESERVED_UPSTREAM_HEADERS`].
///
/// Hop-by-hop headers (RFC 9110 §7.6.1) describe *this* connection, not the
/// one the gateway opens to the upstream. The content/accept entries describe
/// a body this gateway re-serializes and a response shape it parses, so
/// relaying the caller's copies would describe the wrong message.
const NEVER_FORWARD_HEADERS: &[&str] = &[
    "accept",
    "accept-encoding",
    "connection",
    "content-encoding",
    "content-length",
    "content-type",
    "expect",
    "keep-alive",
    "proxy-authenticate",
    "set-cookie",
    "te",
    "trailer",
    "transfer-encoding",
    // W3C trace context (AISIX-Cloud#1279): the caller's `traceparent`
    // names a span in the CALLER's trace — relayed to a provider it links
    // the provider's internal telemetry into the caller's tracing backend
    // and leaks the caller's trace ids to a third party. The gateway is a
    // fresh hop; provider-side propagation, if ever wanted, is a separate
    // opt-in that injects the gateway's own context, never the inbound
    // value. Listed here rather than in RESERVED_UPSTREAM_HEADERS so an
    // operator's deliberate `default_headers` entry for a trusted
    // first-party upstream still works — only the client's copy is
    // blocked.
    "traceparent",
    "tracestate",
    "upgrade",
];

/// Client header prefixes never forwarded whatever the allowlist says.
///
/// `x-aisix-*` is the gateway's own namespace (`x-aisix-request-id`,
/// `x-aisix-routing-tags`, …) — forwarding a client's copy would let a caller
/// spoof gateway-asserted context upstream. `x-stainless-*` is the client
/// SDK's self-description; LiteLLM excludes it from its own forwarding for
/// the same reason we do — relaying one SDK's version headers to a provider
/// that reads them for its own SDK breaks the call.
const NEVER_FORWARD_PREFIXES: &[&str] = &["x-aisix-", "x-stainless-"];

/// The authenticated caller's non-secret identity, resolved once per
/// request so `${request.api_key.*}` templates can name it.
///
/// Deliberately holds identifiers and the operator-typed key label only.
/// The plaintext bearer and its hash are not here and must not be added:
/// this struct is the whole reachable surface of the caller from a
/// header template.
#[derive(Debug, Clone, Default)]
pub struct CallerIdentity {
    pub api_key_id: String,
    pub api_key_name: Option<String>,
    pub team_id: Option<String>,
    pub user_id: Option<String>,
    /// Display name of the member `user_id` names, for the `user_name`
    /// metric label (AISIX-Cloud#1455). Resolved here so it always comes
    /// off the same ApiKey row as `user_id`, from the one place a request
    /// reads its caller.
    ///
    /// NOT a header-template variable: `HEADER_TEMPLATE_VARS` lists the
    /// four `request.api_key.*` keys, and `HeaderVars::resolve` matches
    /// them one by one — this is not among them.
    pub user_name: Option<String>,
}

impl CallerIdentity {
    /// Read the identity off the authenticated key's snapshot entry.
    pub fn from_entry(entry: &aisix_core::ResourceEntry<aisix_core::ApiKey>) -> Self {
        Self {
            api_key_id: entry.id.clone(),
            api_key_name: entry.value.display_name.clone(),
            team_id: entry.value.team_id.clone(),
            user_id: entry.value.user_id.clone(),
            user_name: entry.value.user_name.clone(),
        }
    }
}

/// Everything the header pipeline reads for one upstream call.
///
/// A default-constructed context adds nothing, which is what the paths
/// with no operator config and no client request behind them want.
#[derive(Debug, Clone, Copy, Default)]
pub struct UpstreamHeaderContext<'a> {
    /// The ProviderKey's `request` block, source of both `default_headers`
    /// and `forward_client_headers`.
    pub overrides: Option<&'a RequestOverrides>,
    /// Values for `${...}` references in `default_headers`.
    pub vars: HeaderVars<'a>,
    /// The inbound request's headers, source for `forward_client_headers`.
    /// `None` where a call has no client request behind it (a background
    /// poll of an async job, a semantic-routing embedding lookup) — those
    /// requests forward nothing.
    pub client_headers: Option<&'a HeaderMap>,
    /// The caller's verified JWT, delivered to the upstream under the
    /// ProviderKey's `forward_jwt_header` when both are present. `None`
    /// when the caller authenticated with an API key, or on a call with no
    /// client request behind it.
    pub caller_jwt: Option<&'a str>,
}

impl<'a> UpstreamHeaderContext<'a> {
    /// Context for a call with no request-context variables and no client
    /// headers to forward — only static `default_headers` apply.
    pub fn from_overrides(overrides: Option<&'a RequestOverrides>) -> Self {
        Self {
            overrides,
            ..Self::default()
        }
    }

    pub fn with_vars(mut self, vars: HeaderVars<'a>) -> Self {
        self.vars = vars;
        self
    }

    pub fn with_client_headers(mut self, headers: &'a HeaderMap) -> Self {
        self.client_headers = Some(headers);
        self
    }

    /// Carry the caller's verified JWT, so a ProviderKey configured with
    /// `forward_jwt_header` can hand it to the upstream.
    pub fn with_caller_jwt(mut self, token: Option<&'a str>) -> Self {
        self.caller_jwt = token;
        self
    }
}

fn is_reserved(name: &str) -> bool {
    RESERVED_UPSTREAM_HEADERS.contains(&name)
}

fn is_forwardable_name(name: &str) -> bool {
    !is_reserved(name)
        && !NEVER_FORWARD_HEADERS.contains(&name)
        && !NEVER_FORWARD_PREFIXES.iter().any(|p| name.starts_with(p))
}

/// Whether an allowlist pattern admits a header name. Patterns are matched
/// case-insensitively against the lowercase name, and a single `*` glob is
/// supported (`"x-trace-*"`), matching how `allowed_models` / `allowed_agents`
/// patterns behave elsewhere in the snapshot.
fn allowlist_admits(patterns: &[String], name: &str) -> bool {
    patterns
        .iter()
        .any(|p| wildcard_matches(&p.to_ascii_lowercase(), name))
}

/// Resolve the operator-configured headers for one upstream call: rendered
/// `default_headers` first, then the client headers the allowlist admits.
///
/// Names are returned lowercase and de-duplicated (first wins, so a
/// `default_headers` entry shadows a client header of the same name).
/// Entries whose name or value will not parse as HTTP are skipped rather
/// than failing the request — an unparseable entry is a config error one
/// layer up, which cp-api rejects at write time.
///
/// Callers that build a [`HeaderMap`] should use [`apply_request_headers`];
/// this lower-level form exists for the Bedrock path, whose headers have to
/// be handed to the AWS SDK's pre-signing interceptor instead.
pub fn resolve_extra_headers(ctx: &UpstreamHeaderContext<'_>) -> Vec<(HeaderName, HeaderValue)> {
    let Some(r) = ctx.overrides else {
        return Vec::new();
    };
    let mut out: Vec<(HeaderName, HeaderValue)> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    for (name, value) in &r.default_headers {
        let Ok(parsed_name) = name.parse::<HeaderName>() else {
            continue;
        };
        if is_reserved(parsed_name.as_str()) || !seen.insert(parsed_name.as_str().to_string()) {
            continue;
        }
        // An unresolvable template drops just this header — see
        // `aisix_core::header_template`.
        let Some(rendered) = aisix_core::render_header_template(value, &ctx.vars) else {
            continue;
        };
        let Ok(parsed_value) = HeaderValue::from_str(&rendered) else {
            continue;
        };
        out.push((parsed_name, parsed_value));
    }

    if r.forward_client_headers.is_empty() {
        return out;
    }
    let Some(client) = ctx.client_headers else {
        return out;
    };
    for name in client.keys() {
        // `HeaderName` is already lowercase on the wire-parsed side.
        let lower = name.as_str();
        if !is_forwardable_name(lower)
            || !allowlist_admits(&r.forward_client_headers, lower)
            || seen.contains(lower)
        {
            continue;
        }
        // A repeated header (`anthropic-beta: a` twice) forwards its first
        // value only; the upstream sees one well-formed header rather than
        // a list this gateway never interpreted.
        let Some(value) = client.get(name) else {
            continue;
        };
        seen.insert(lower.to_string());
        out.push((name.clone(), value.clone()));
    }
    out
}

/// Merge the operator-configured headers into an outbound request's
/// `HeaderMap`, leaving every name the caller already set untouched.
///
/// Callers MUST insert their bridge-owned headers (auth, `content-type`,
/// `x-aisix-request-id`) before calling this — that ordering is what makes
/// gateway-owned headers un-overridable.
pub fn apply_request_headers(headers: &mut HeaderMap, ctx: &UpstreamHeaderContext<'_>) {
    for (name, value) in resolve_extra_headers(ctx) {
        if headers.contains_key(&name) {
            continue;
        }
        headers.insert(name, value);
    }
    apply_forwarded_jwt(headers, ctx);
}

/// Deliver the caller's verified JWT under the ProviderKey's
/// `forward_jwt_header`, for an internal upstream that authorizes on the
/// end user's claims.
///
/// This is the ONE step that overwrites: everything above it declines a
/// name already present, because a gateway-owned header outranks operator
/// and client config. Here the operator has named a slot explicitly, on
/// this ProviderKey, for this upstream — including `authorization`, which
/// the bridge has already filled with the gateway's own credential. That
/// is the case the field exists for, so the caller's token REPLACES it:
/// `insert` rather than `append`, since `append` would put two
/// credentials on the wire and let the upstream pick (the #411 shape).
fn apply_forwarded_jwt(headers: &mut HeaderMap, ctx: &UpstreamHeaderContext<'_>) {
    let configured = ctx.overrides.and_then(|o| o.forward_jwt_header.as_deref());
    let Some((name, value)) = aisix_core::forwarded_jwt(configured, ctx.caller_jwt) else {
        return;
    };
    let (Ok(name), Ok(mut value)) = (
        HeaderName::try_from(name.as_str()),
        HeaderValue::from_str(&value),
    ) else {
        // A token that cannot be a header value is a broken credential,
        // not a reason to fail the request: the upstream then answers the
        // way it answers any unauthenticated call.
        return;
    };
    value.set_sensitive(true);
    headers.insert(name, value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn overrides(defaults: &[(&str, &str)], forward: &[&str]) -> RequestOverrides {
        RequestOverrides {
            default_headers: defaults
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect::<HashMap<_, _>>(),
            forward_client_headers: forward.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    fn client(headers: &[(&str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (k, v) in headers {
            map.insert(
                k.parse::<HeaderName>().unwrap(),
                HeaderValue::from_str(v).unwrap(),
            );
        }
        map
    }

    #[test]
    fn forwarded_jwt_reaches_the_configured_header() {
        let r = RequestOverrides {
            forward_jwt_header: Some("x-user-jwt".into()),
            ..Default::default()
        };
        let mut headers = HeaderMap::new();
        let ctx = UpstreamHeaderContext::from_overrides(Some(&r)).with_caller_jwt(Some("eyJraw"));
        apply_request_headers(&mut headers, &ctx);
        assert_eq!(headers["x-user-jwt"], "eyJraw");
    }

    #[test]
    fn forwarded_jwt_replaces_the_bridge_owned_credential() {
        // The one slot where operator config outranks a gateway-owned
        // header: the operator named `authorization` on THIS ProviderKey,
        // for an upstream that authorizes on the end user. Two credentials
        // on the wire would let the upstream choose between them.
        let r = RequestOverrides {
            forward_jwt_header: Some("authorization".into()),
            ..Default::default()
        };
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("authorization"),
            HeaderValue::from_static("Bearer gateway-held-key"),
        );
        let ctx = UpstreamHeaderContext::from_overrides(Some(&r)).with_caller_jwt(Some("eyJraw"));
        apply_request_headers(&mut headers, &ctx);
        assert_eq!(headers["authorization"], "Bearer eyJraw");
        assert_eq!(headers.get_all("authorization").iter().count(), 1);
    }

    #[test]
    fn an_api_key_caller_forwards_no_token() {
        let r = RequestOverrides {
            forward_jwt_header: Some("authorization".into()),
            ..Default::default()
        };
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("authorization"),
            HeaderValue::from_static("Bearer gateway-held-key"),
        );
        let ctx = UpstreamHeaderContext::from_overrides(Some(&r)).with_caller_jwt(None);
        apply_request_headers(&mut headers, &ctx);
        // The gateway's own credential is untouched, not replaced by a blank.
        assert_eq!(headers["authorization"], "Bearer gateway-held-key");
    }

    #[test]
    fn an_unconfigured_provider_key_forwards_no_token() {
        // The default for every existing ProviderKey: a caller's JWT never
        // reaches an upstream nobody opted in.
        let r = RequestOverrides::default();
        let mut headers = HeaderMap::new();
        let ctx = UpstreamHeaderContext::from_overrides(Some(&r)).with_caller_jwt(Some("eyJraw"));
        apply_request_headers(&mut headers, &ctx);
        assert!(headers.is_empty(), "got: {headers:?}");
    }

    #[test]
    fn forwarded_jwt_is_marked_sensitive() {
        // `Debug` on the header map is reachable from tracing; a live
        // credential must not be printable through it.
        let r = RequestOverrides {
            forward_jwt_header: Some("x-user-jwt".into()),
            ..Default::default()
        };
        let mut headers = HeaderMap::new();
        let ctx = UpstreamHeaderContext::from_overrides(Some(&r)).with_caller_jwt(Some("eyJraw"));
        apply_request_headers(&mut headers, &ctx);
        // Pin that the token IS on the wire first: without this, a build
        // that forwards nothing would satisfy the redaction assertion
        // below for the wrong reason.
        assert_eq!(headers["x-user-jwt"], "eyJraw");
        assert!(
            !format!("{headers:?}").contains("eyJraw"),
            "got: {headers:?}"
        );
    }

    #[test]
    fn default_headers_are_added_and_templates_rendered() {
        let r = overrides(
            &[
                ("x-corp-trace", "static"),
                ("x-tenant-id", "${request.api_key.team_id}"),
            ],
            &[],
        );
        let vars = HeaderVars {
            api_key_team_id: Some("team-7"),
            ..Default::default()
        };
        let mut headers = HeaderMap::new();
        let ctx = UpstreamHeaderContext::from_overrides(Some(&r)).with_vars(vars);
        apply_request_headers(&mut headers, &ctx);
        assert_eq!(headers["x-corp-trace"], "static");
        assert_eq!(headers["x-tenant-id"], "team-7");
    }

    #[test]
    fn unresolvable_template_drops_only_its_own_header() {
        let r = overrides(
            &[
                ("x-tenant-id", "${request.api_key.team_id}"),
                ("x-key", "${request.api_key.name}"),
            ],
            &[],
        );
        let vars = HeaderVars {
            api_key_name: Some("acme"),
            api_key_team_id: None,
            ..Default::default()
        };
        let mut headers = HeaderMap::new();
        let ctx = UpstreamHeaderContext::from_overrides(Some(&r)).with_vars(vars);
        apply_request_headers(&mut headers, &ctx);
        assert!(
            !headers.contains_key("x-tenant-id"),
            "a key with no team must not send a blank tenant header"
        );
        assert_eq!(headers["x-key"], "acme");
    }

    #[test]
    fn caller_owned_headers_are_never_overwritten() {
        let r = overrides(&[("x-corp-trace", "operator")], &["x-corp-trace"]);
        let mut headers = HeaderMap::new();
        headers.insert("x-corp-trace", HeaderValue::from_static("gateway"));
        let inbound = client(&[("x-corp-trace", "client")]);
        let ctx = UpstreamHeaderContext::from_overrides(Some(&r)).with_client_headers(&inbound);
        apply_request_headers(&mut headers, &ctx);
        assert_eq!(headers["x-corp-trace"], "gateway");
    }

    #[test]
    fn default_headers_win_over_a_forwarded_client_header() {
        let r = overrides(&[("x-tenant-id", "operator")], &["x-tenant-id"]);
        let mut headers = HeaderMap::new();
        let inbound = client(&[("x-tenant-id", "client-claimed")]);
        let ctx = UpstreamHeaderContext::from_overrides(Some(&r)).with_client_headers(&inbound);
        apply_request_headers(&mut headers, &ctx);
        assert_eq!(headers["x-tenant-id"], "operator");
    }

    #[test]
    fn reserved_auth_headers_are_dropped_from_both_sources() {
        let r = overrides(
            &[
                ("authorization", "Bearer attacker"),
                ("api-key", "attacker"),
            ],
            &["authorization", "x-api-key", "cookie", "*"],
        );
        let mut headers = HeaderMap::new();
        let inbound = client(&[
            ("authorization", "Bearer caller"),
            ("x-api-key", "caller"),
            ("cookie", "session=1"),
        ]);
        let ctx = UpstreamHeaderContext::from_overrides(Some(&r)).with_client_headers(&inbound);
        apply_request_headers(&mut headers, &ctx);
        assert!(headers.is_empty(), "leaked: {headers:?}");
    }

    #[test]
    fn transport_and_gateway_namespaces_are_never_forwarded() {
        let r = overrides(&[], &["*"]);
        let mut headers = HeaderMap::new();
        let inbound = client(&[
            ("content-length", "12"),
            ("connection", "keep-alive"),
            ("x-aisix-request-id", "spoofed"),
            ("x-stainless-lang", "js"),
            ("x-keep", "yes"),
        ]);
        let ctx = UpstreamHeaderContext::from_overrides(Some(&r)).with_client_headers(&inbound);
        apply_request_headers(&mut headers, &ctx);
        assert_eq!(headers.len(), 1, "leaked: {headers:?}");
        assert_eq!(headers["x-keep"], "yes");
    }

    // AISIX-Cloud#1279: the caller's W3C trace context must never reach a
    // provider — not under a `*` glob, and not even when an allowlist names
    // the headers outright. The gateway is a fresh tracing hop; a future
    // provider-side propagation opt-in injects the gateway's OWN context.
    #[test]
    fn trace_context_headers_are_never_forwarded() {
        for allowlist in [&["*"][..], &["traceparent", "tracestate"][..]] {
            let patterns: Vec<&str> = allowlist.to_vec();
            let r = overrides(&[], &patterns);
            let mut headers = HeaderMap::new();
            let inbound = client(&[
                (
                    "traceparent",
                    "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
                ),
                ("tracestate", "vendor=x"),
            ]);
            let ctx = UpstreamHeaderContext::from_overrides(Some(&r)).with_client_headers(&inbound);
            apply_request_headers(&mut headers, &ctx);
            assert!(
                headers.is_empty(),
                "trace context leaked under {allowlist:?}: {headers:?}"
            );
        }
    }

    #[test]
    fn allowlist_matches_exact_names_and_a_single_glob() {
        let r = overrides(&[], &["Anthropic-Beta", "x-trace-*"]);
        let mut headers = HeaderMap::new();
        let inbound = client(&[
            ("anthropic-beta", "tools-2024-05-16"),
            ("x-trace-id", "t-1"),
            ("x-trace-parent", "p-1"),
            ("x-tenant-id", "not-allowlisted"),
        ]);
        let ctx = UpstreamHeaderContext::from_overrides(Some(&r)).with_client_headers(&inbound);
        apply_request_headers(&mut headers, &ctx);
        assert_eq!(headers["anthropic-beta"], "tools-2024-05-16");
        assert_eq!(headers["x-trace-id"], "t-1");
        assert_eq!(headers["x-trace-parent"], "p-1");
        assert!(!headers.contains_key("x-tenant-id"));
    }

    #[test]
    fn a_default_header_the_caller_did_not_set_is_added_case_insensitively() {
        // The caller's mixed-case header must still block a lowercase-keyed
        // default — `http::HeaderName` canonicalizes both sides.
        let r = overrides(
            &[("anthropic-version", "2023-06-01"), ("x-foo", "default")],
            &[],
        );
        let mut headers = HeaderMap::new();
        headers.insert("X-Foo", HeaderValue::from_static("caller-value"));
        apply_request_headers(
            &mut headers,
            &UpstreamHeaderContext::from_overrides(Some(&r)),
        );
        assert_eq!(headers["anthropic-version"], "2023-06-01");
        assert_eq!(headers["x-foo"], "caller-value");
    }

    #[test]
    fn an_unparseable_header_name_skips_only_that_entry() {
        let r = overrides(&[("not a valid name", "v"), ("x-foo", "ok")], &[]);
        let mut headers = HeaderMap::new();
        apply_request_headers(
            &mut headers,
            &UpstreamHeaderContext::from_overrides(Some(&r)),
        );
        assert_eq!(headers.len(), 1);
        assert_eq!(headers["x-foo"], "ok");
    }

    #[test]
    fn empty_allowlist_forwards_nothing() {
        let r = overrides(&[], &[]);
        let inbound = client(&[("x-trace-id", "t-1")]);
        let mut headers = HeaderMap::new();
        let ctx = UpstreamHeaderContext::from_overrides(Some(&r)).with_client_headers(&inbound);
        apply_request_headers(&mut headers, &ctx);
        assert!(headers.is_empty());
    }

    #[test]
    fn no_overrides_block_is_a_no_op() {
        let inbound = client(&[("x-trace-id", "t-1")]);
        let mut headers = HeaderMap::new();
        let ctx = UpstreamHeaderContext::default().with_client_headers(&inbound);
        apply_request_headers(&mut headers, &ctx);
        assert!(headers.is_empty());
    }
}
