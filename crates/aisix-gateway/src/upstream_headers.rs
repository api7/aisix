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
}

impl CallerIdentity {
    /// Read the identity off the authenticated key's snapshot entry.
    pub fn from_entry(entry: &aisix_core::ResourceEntry<aisix_core::ApiKey>) -> Self {
        Self {
            api_key_id: entry.id.clone(),
            api_key_name: entry.value.display_name.clone(),
            team_id: entry.value.team_id.clone(),
            user_id: entry.value.user_id.clone(),
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
/// supported (`"x-trace-*"`), matching how `allowed_models` / `allowed_tools`
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
