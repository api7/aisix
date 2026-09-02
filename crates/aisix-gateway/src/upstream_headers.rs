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
//! **`default_headers` defers to the gateway; `forward_client_headers`
//! does not.** Every caller inserts its own bridge-owned headers (auth,
//! `content-type`, `x-aisix-request-id`, streaming `accept`) into the
//! `HeaderMap` *first*, and the `default_headers` merge declines a name
//! that is already present — a static operator value never displaces the
//! gateway's own. A FORWARDED CLIENT header does displace it, because the
//! operator named that header on this specific upstream and the slot they
//! most often name is the credential one: handing an internal service the
//! caller's own `Authorization` in place of the gateway's is the whole
//! reason the field exists. `insert` rather than `append`, so exactly one
//! credential reaches the wire and the upstream never picks between two
//! (the #411 shape). Within this module `default_headers` still resolves
//! first, so an operator's static header beats a client header of the same
//! name.
//!
//! What can never be forwarded is narrow and lives in
//! `aisix_core::forwarded_headers`: the transport slots whose forwarding
//! breaks the exchange, the gateway's own `x-aisix-*` namespace, and — on
//! this pipeline, which REBUILDS the outbound message — the headers that
//! describe a body the gateway re-serializes or a response shape it
//! parses.
//!
//! Before #1167 the standard endpoints rebuilt the outbound header set from
//! scratch and dropped every client header — the default this module
//! preserves. `/passthrough/*` is the opposite policy (forward everything
//! except `pk.strip_headers`) and does not use this pipeline; there
//! `forward_client_headers` overrides the strip instead.

use std::collections::HashSet;

use aisix_core::{HeaderVars, RequestOverrides};
use http::{
    header::{HeaderName, HeaderValue},
    HeaderMap,
};

/// Whether a client's copy of a header may be forwarded by this pipeline.
///
/// Re-exported from `aisix-core`, which owns the criterion so that every
/// surface — `/v1/*`, MCP, and `/passthrough/*` — answers it alike.
pub use aisix_core::{client_header_forwardable, header_forward_blocked};

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
    /// Header names THIS surface owns, which the shared lists cannot know
    /// about. `/v1/realtime` opens its own WebSocket, so the caller's
    /// `sec-websocket-*` slots describe the connection the caller opened
    /// and one of them carries the caller's own gateway key.
    pub surface_blocked: &'a [&'a str],
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

    /// Names this surface refuses on top of the shared lists — see
    /// [`UpstreamHeaderContext::surface_blocked`].
    pub fn with_surface_blocked(mut self, blocked: &'a [&'a str]) -> Self {
        self.surface_blocked = blocked;
        self
    }
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
/// be handed to the AWS SDK's pre-signing interceptor instead. Note that it
/// FLATTENS the two features into one list and so loses the distinction
/// [`apply_request_headers`] draws between them — that is correct for
/// Bedrock, whose interceptor is first-wins and which resolves the
/// credential collision by clearing the slot up front instead.
pub fn resolve_extra_headers(ctx: &UpstreamHeaderContext<'_>) -> Vec<(HeaderName, HeaderValue)> {
    let mut out = resolve_default_headers(ctx);
    let taken: HashSet<HeaderName> = out.iter().map(|(name, _)| name.clone()).collect();
    out.extend(
        ForwardedClientHeaders::resolve(ctx)
            .entries
            .into_iter()
            .filter(|(name, _)| !taken.contains(name)),
    );
    out
}

/// The ProviderKey's rendered `default_headers`, in declaration order.
///
/// Public for the dispatch surfaces that resolve the two features once at
/// target-resolution time and reuse them across several round-trips (the
/// jobs and video surfaces): they hold [`ForwardedClientHeaders`]
/// separately, because the two merge into the outbound map with opposite
/// rules.
pub fn resolve_default_headers(ctx: &UpstreamHeaderContext<'_>) -> Vec<(HeaderName, HeaderValue)> {
    let Some(r) = ctx.overrides else {
        return Vec::new();
    };
    let mut out: Vec<(HeaderName, HeaderValue)> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    for (name, value) in &r.default_headers {
        let Ok(parsed_name) = name.parse::<HeaderName>() else {
            continue;
        };
        if aisix_core::header_forward_blocked(parsed_name.as_str())
            || !seen.insert(parsed_name.as_str().to_string())
        {
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
    out
}

/// The inbound client headers one upstream call forwards, resolved from
/// the operator's allowlist against the request that is actually in
/// flight.
///
/// Two things read it, and they must agree: the merge below, which
/// DELIVERS the headers into the outbound map, and every dispatch site
/// that injects a gateway credential, which asks [`Self::claims`] first
/// and stands aside for the slot the operator named. Resolving once and
/// asking both questions of the same value is what keeps them from
/// disagreeing — a value that will not become a header claims no slot, and
/// the upstream then keeps the credential it would otherwise have had.
#[derive(Debug, Clone, Default)]
pub struct ForwardedClientHeaders {
    entries: Vec<(HeaderName, HeaderValue)>,
}

impl ForwardedClientHeaders {
    /// Resolve the forward for this call. Empty when the ProviderKey
    /// configures none, and on a call with no client request behind it (a
    /// background poll of an async job, a semantic-routing embedding
    /// lookup) — those forward nothing.
    pub fn resolve(ctx: &UpstreamHeaderContext<'_>) -> Self {
        let (Some(r), Some(client)) = (ctx.overrides, ctx.client_headers) else {
            return Self::default();
        };
        Self {
            entries: aisix_core::resolve_forwarded_client_headers(
                &r.forward_client_headers,
                client,
                ctx.surface_blocked,
            ),
        }
    }

    /// Whether the forward claims `name` (lowercase) — the question a site
    /// injecting its own credential into that slot must ask before
    /// injecting, since the forwarded value is going to take it.
    pub fn claims(&self, name: &str) -> bool {
        self.entries.iter().any(|(n, _)| n.as_str() == name)
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Deliver the forwarded headers.
    ///
    /// A CREDENTIAL slot is overwritten — that is the collision the field
    /// exists for, and the operator named it on this upstream. Any other
    /// name already on the request was put there by the gateway (or by
    /// `default_headers`) to make the exchange work — a provider's
    /// async-mode flag, its API-version selector — so it is left alone; a
    /// caller's value there breaks the call rather than re-identifying it.
    ///
    /// `insert` rather than `append` throughout: a slot filled twice would
    /// put two values on the wire and let the upstream pick between them,
    /// which on a credential slot is the #411 shape.
    ///
    /// Every surface delivers through here, including the ones that
    /// resolve once and reuse across several round-trips (jobs, videos) —
    /// the precedence is a property of this type, not of each call site.
    pub fn apply(&self, headers: &mut HeaderMap) {
        for (name, value) in &self.entries {
            if headers.contains_key(name) && !aisix_core::displaces_a_gateway_header(name.as_str())
            {
                continue;
            }
            headers.insert(name.clone(), value.clone());
        }
    }
}

/// Merge the operator-configured headers into an outbound request's
/// `HeaderMap`.
///
/// Callers MUST insert their bridge-owned headers (auth, `content-type`,
/// `x-aisix-request-id`) before calling this. That ordering is what makes
/// a `default_headers` entry unable to displace them — and what lets a
/// forwarded client header displace the credential deliberately.
pub fn apply_request_headers(headers: &mut HeaderMap, ctx: &UpstreamHeaderContext<'_>) {
    for (name, value) in resolve_default_headers(ctx) {
        if headers.contains_key(&name) {
            continue;
        }
        headers.insert(name, value);
    }
    // Runs after, so a `default_headers` entry shadows a forwarded header
    // of the same name — both are operator configuration and the static
    // one is the more specific statement of intent. `apply` is what makes
    // that hold: it declines a name already present unless the slot is a
    // credential one.
    ForwardedClientHeaders::resolve(ctx).apply(headers);
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
    fn a_forwarded_header_reaches_the_upstream() {
        let r = overrides(&[], &["x-user-jwt"]);
        let inbound = client(&[("x-user-jwt", "eyJraw")]);
        let mut headers = HeaderMap::new();
        let ctx = UpstreamHeaderContext::from_overrides(Some(&r)).with_client_headers(&inbound);
        apply_request_headers(&mut headers, &ctx);
        assert_eq!(headers["x-user-jwt"], "eyJraw");
    }

    #[test]
    fn a_forwarded_header_replaces_the_bridge_owned_credential() {
        // The case the field exists for: the operator named
        // `authorization` on THIS ProviderKey, for an upstream that
        // authorizes on the end user. Two credentials on the wire would
        // let the upstream choose between them.
        let r = overrides(&[], &["authorization"]);
        let inbound = client(&[("authorization", "Bearer callers-own")]);
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("authorization"),
            HeaderValue::from_static("Bearer gateway-held-key"),
        );
        let ctx = UpstreamHeaderContext::from_overrides(Some(&r)).with_client_headers(&inbound);
        apply_request_headers(&mut headers, &ctx);
        assert_eq!(headers["authorization"], "Bearer callers-own");
        assert_eq!(headers.get_all("authorization").iter().count(), 1);
    }

    #[test]
    fn a_caller_that_sent_nothing_leaves_the_credential_alone() {
        // The forward is keyed on what the caller actually sent, not on
        // the configuration: an operator who opts the slot in must not
        // blank out the gateway's own credential for every caller who
        // happens not to send one.
        let r = overrides(&[], &["authorization"]);
        let inbound = client(&[("x-other", "v")]);
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("authorization"),
            HeaderValue::from_static("Bearer gateway-held-key"),
        );
        let ctx = UpstreamHeaderContext::from_overrides(Some(&r)).with_client_headers(&inbound);
        apply_request_headers(&mut headers, &ctx);
        assert_eq!(headers["authorization"], "Bearer gateway-held-key");
    }

    #[test]
    fn an_operator_static_header_still_wins_a_forwarded_one() {
        // Both are operator configuration; the static value is the more
        // specific statement of intent, and this is the pre-existing
        // precedence.
        let r = overrides(&[("x-overlap", "from-operator")], &["x-overlap"]);
        let inbound = client(&[("x-overlap", "from-client")]);
        let mut headers = HeaderMap::new();
        let ctx = UpstreamHeaderContext::from_overrides(Some(&r)).with_client_headers(&inbound);
        apply_request_headers(&mut headers, &ctx);
        assert_eq!(headers["x-overlap"], "from-operator");
    }

    #[test]
    fn a_default_header_the_gateway_shut_out_does_not_shut_out_the_forward() {
        // `default_headers` cannot displace a gateway-owned slot, so it
        // never "won" that name — and must not silently suppress the
        // forward, which CAN displace it.
        let r = overrides(&[("authorization", "Bearer static")], &["authorization"]);
        let inbound = client(&[("authorization", "Bearer callers-own")]);
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("authorization"),
            HeaderValue::from_static("Bearer gateway-held-key"),
        );
        let ctx = UpstreamHeaderContext::from_overrides(Some(&r)).with_client_headers(&inbound);
        apply_request_headers(&mut headers, &ctx);
        assert_eq!(headers["authorization"], "Bearer callers-own");
    }

    #[test]
    fn a_call_with_no_client_request_forwards_nothing() {
        // Background job polls and semantic-routing embedding lookups have
        // no inbound headers to draw on.
        let r = overrides(&[], &["*"]);
        let mut headers = HeaderMap::new();
        let ctx = UpstreamHeaderContext::from_overrides(Some(&r));
        apply_request_headers(&mut headers, &ctx);
        assert!(headers.is_empty(), "got: {headers:?}");
        assert!(ForwardedClientHeaders::resolve(&ctx).is_empty());
    }

    #[test]
    fn an_unconfigured_provider_key_forwards_nothing() {
        let r = RequestOverrides::default();
        let inbound = client(&[("authorization", "Bearer callers-own")]);
        let mut headers = HeaderMap::new();
        let ctx = UpstreamHeaderContext::from_overrides(Some(&r)).with_client_headers(&inbound);
        apply_request_headers(&mut headers, &ctx);
        assert!(headers.is_empty(), "got: {headers:?}");
    }

    #[test]
    fn the_claimed_set_is_what_a_credential_site_asks() {
        let r = overrides(&[], &["authorization", "x-trace-*"]);
        let inbound = client(&[("authorization", "Bearer c"), ("x-trace-id", "t")]);
        let ctx = UpstreamHeaderContext::from_overrides(Some(&r)).with_client_headers(&inbound);
        let forwarded = ForwardedClientHeaders::resolve(&ctx);
        assert!(forwarded.claims("authorization"));
        assert!(forwarded.claims("x-trace-id"));
        // Configured but not sent: the site keeps injecting its own.
        assert!(!forwarded.claims("x-api-key"));
    }

    #[test]
    fn a_forwarded_value_is_marked_sensitive() {
        // `Debug` on the header map is reachable from tracing; a forwarded
        // credential must not be printable through it.
        let r = overrides(&[], &["authorization"]);
        let inbound = client(&[("authorization", "Bearer callers-own")]);
        let mut headers = HeaderMap::new();
        let ctx = UpstreamHeaderContext::from_overrides(Some(&r)).with_client_headers(&inbound);
        apply_request_headers(&mut headers, &ctx);
        assert!(headers["authorization"].is_sensitive());
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
    fn a_gateway_owned_header_survives_a_static_operator_one() {
        // `default_headers` is a fallback, never an override: only the
        // forward displaces a slot the gateway already filled.
        let r = overrides(&[("x-corp-trace", "operator")], &[]);
        let mut headers = HeaderMap::new();
        headers.insert("x-corp-trace", HeaderValue::from_static("gateway"));
        let ctx = UpstreamHeaderContext::from_overrides(Some(&r));
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

    /// Credential slots are forwardable — an internal upstream reading the
    /// caller's own credential is the capability's purpose — but only where
    /// the gateway has not already claimed the slot, and only with the
    /// operator's own patterns behind them.
    /// A forwarded header may take a CREDENTIAL slot from the gateway and
    /// nothing else. Every surface delivers through `apply`, including the
    /// ones that resolve once and reuse across round-trips (jobs, videos),
    /// so this is the one place the rule has to hold.
    #[test]
    fn a_forward_displaces_only_a_credential_slot() {
        let r = overrides(&[], &["authorization", "x-dashscope-async", "x-fresh"]);
        let inbound = client(&[
            ("authorization", "Bearer caller"),
            // A provider's own wire-shape selector: the gateway set it to
            // pick the submission mode it then decodes, so a caller's
            // value there breaks the call rather than re-identifying it.
            ("x-dashscope-async", "disable"),
            ("x-fresh", "from-client"),
        ]);
        let mut headers = HeaderMap::new();
        headers.insert(
            HeaderName::from_static("authorization"),
            HeaderValue::from_static("Bearer gateway-held-key"),
        );
        headers.insert(
            HeaderName::from_static("x-dashscope-async"),
            HeaderValue::from_static("enable"),
        );
        let ctx = UpstreamHeaderContext::from_overrides(Some(&r)).with_client_headers(&inbound);
        apply_request_headers(&mut headers, &ctx);
        assert_eq!(headers["authorization"], "Bearer caller");
        assert_eq!(headers["x-dashscope-async"], "enable");
        // A slot nobody had claimed is still filled.
        assert_eq!(headers["x-fresh"], "from-client");
    }

    #[test]
    fn credential_slots_reach_an_upstream_the_operator_opted_in() {
        let r = overrides(&[], &["authorization", "x-api-key", "cookie"]);
        let mut headers = HeaderMap::new();
        let inbound = client(&[
            ("authorization", "Bearer caller"),
            ("x-api-key", "caller"),
            ("cookie", "session=1"),
            ("x-goog-api-key", "not-allowlisted"),
        ]);
        let ctx = UpstreamHeaderContext::from_overrides(Some(&r)).with_client_headers(&inbound);
        apply_request_headers(&mut headers, &ctx);
        assert_eq!(headers["authorization"], "Bearer caller");
        assert_eq!(headers["x-api-key"], "caller");
        assert_eq!(headers["cookie"], "session=1");
        assert!(!headers.contains_key("x-goog-api-key"), "got: {headers:?}");

        // And a credential slot needs its OWN name: a broad glob does not
        // reach one, which is what keeps an already-configured `["x-*"]`
        // meaning what it meant when it was written.
        let r = overrides(&[], &["x-*"]);
        let mut headers = HeaderMap::new();
        let ctx = UpstreamHeaderContext::from_overrides(Some(&r)).with_client_headers(&inbound);
        apply_request_headers(&mut headers, &ctx);
        assert!(!headers.contains_key("x-api-key"), "got: {headers:?}");
        assert!(!headers.contains_key("x-goog-api-key"), "got: {headers:?}");
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

    // AISIX-Cloud#1279: the caller's W3C trace context must not reach a
    // provider under a GLOB. The gateway is a fresh tracing hop, and an
    // operator asking for their own `x-*` headers is not asking to graft
    // one tenant's trace onto a third party's telemetry. Naming the
    // headers outright IS that request, and is honoured.
    #[test]
    fn trace_context_needs_its_own_name_not_a_glob() {
        let inbound = client(&[
            (
                "traceparent",
                "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            ),
            ("tracestate", "vendor=x"),
        ]);

        for glob in [&["*"][..], &["trace*"][..]] {
            let patterns: Vec<&str> = glob.to_vec();
            let r = overrides(&[], &patterns);
            let mut headers = HeaderMap::new();
            let ctx = UpstreamHeaderContext::from_overrides(Some(&r)).with_client_headers(&inbound);
            apply_request_headers(&mut headers, &ctx);
            assert!(
                headers.is_empty(),
                "trace context leaked under {glob:?}: {headers:?}"
            );
        }

        let r = overrides(&[], &["traceparent", "tracestate"]);
        let mut headers = HeaderMap::new();
        let ctx = UpstreamHeaderContext::from_overrides(Some(&r)).with_client_headers(&inbound);
        apply_request_headers(&mut headers, &ctx);
        assert_eq!(
            headers["traceparent"],
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
        );
        assert_eq!(headers["tracestate"], "vendor=x");
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
