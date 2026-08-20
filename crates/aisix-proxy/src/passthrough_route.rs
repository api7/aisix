//! Explicit passthrough routes (`PassthroughRoute` resources).
//!
//! Replaces the removed implicit `/passthrough/:provider/*rest` tunnel: a
//! route binds a gateway entry (path prefix and/or inbound `Host`) to ONE
//! upstream target with its own gateway-auth mode and credential handling.
//! There is no implicit provider→Model credential borrowing
//! (AISIX-Cloud#1127) and no forced `Authorization` replacement
//! (AISIX-Cloud#1312).
//!
//! ## Envelope detection
//!
//! The request body's envelope is detected once per exchange from its
//! top-level keys ([`detect_protocol`]) and drives guardrail text
//! extraction, content capture and usage extraction for both the request
//! and the response (buffered or streamed). Detection never affects the
//! relay itself — bodies are forwarded verbatim regardless — and every
//! extraction degrades to the whole lossy-UTF-8 body when the detected
//! shape yields no text, so a mis-detected envelope loses no audit
//! coverage. SSE upstream responses are always relayed incrementally;
//! anything else is buffered (guardrails and usage need the whole body).
//!
//! A DETECTED envelope must observe what the typed endpoint serving that
//! same envelope observes — every token dimension of
//! [`aisix_obs::UsageEvent`], the caller's model alias, the guardrail text
//! including tool calls, TTFT, and a 499 for a stream the client
//! abandoned. Anything less makes a route a place where enforcement and
//! metering quietly weaken.
//!
//! ## Entry points
//!
//! - [`entry`] — the proxy router's **fallback** handler. Path-prefix
//!   routes match here, after every typed route has had its chance, so a
//!   route can never shadow `/v1/*`, `/mcp`, or `/a2a`. A no-match request
//!   keeps the pre-existing plain 404; `/passthrough/*` no-matches return
//!   the 410 migration tombstone.
//! - [`host_dispatch`] — a **pre-routing** middleware (outermost wrap in
//!   `build_router`). A request whose `Host` matches an enabled route's
//!   `hosts` was never addressed to this gateway's own API, so it must not
//!   fall into a typed route that happens to share the path (forward-proxy
//!   traffic: a TLS-terminating device delivers e.g.
//!   `Host: api.githubcopilot.com` with its original path). On a host
//!   match the middleware dispatches straight to [`entry`].
//!
//! ## Auth
//!
//! Per-route `auth_mode`: `gateway_key` reads the standard
//! `Authorization: Bearer` / `x-api-key` gateway credential; `header_key`
//! reads it from the route's `auth_header_name` (leaving `Authorization`
//! for the upstream credential); `anonymous` binds the request to the
//! route's `anonymous_key_id` principal, gated by `source_cidrs`. Every
//! mode ends in an [`AuthenticatedKey`] whose `allowed_routes` ACL, rate
//! limits, and budget apply unchanged.
//!
//! ## Credentials
//!
//! `inject` strips inbound credential headers (the ProviderKey's
//! `strip_headers`) and injects the configured ProviderKey's secret with
//! the per-provider auth shape (#166). `forward_client` forwards the
//! caller's own credential headers verbatim and strips only the gateway's
//! side-channel headers, so the gateway credential never leaks upstream.

use std::sync::Arc;
use std::time::{Duration, Instant};

use aisix_obs::AccessLog;
use axum::body::{Body, Bytes};
use axum::extract::{Request, State};
use axum::http::{header, HeaderMap, HeaderValue, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use aisix_core::resource::ResourceEntry;
use aisix_core::{PassthroughAuthMode, PassthroughCredentialMode, PassthroughRoute};

use crate::auth::AuthenticatedKey;
use crate::error::ProxyError;
use crate::state::ProxyState;

/// Bounded `model` metric label for passthrough-route requests. Route
/// traffic resolves no Model; per-route attribution lives on the usage
/// event (`passthrough_route_name`), not in Prometheus label space.
const PASSTHROUGH_MODEL_LABEL: &str = "passthrough";

/// `provider` metric label for `forward_client` routes, which have no
/// ProviderKey to take a provider name from.
const BYO_PROVIDER_LABEL: &str = "byo";

/// `provider` label for 410-tombstoned hits on the removed tunnel's
/// namespace — the caller-supplied path segment must never mint a series.
const UNRESOLVED_LABEL: &str = "unresolved";

/// Endpoint label for metrics/usage attribution: one family for all
/// passthrough-route traffic (route names are operator data, not label
/// space).
const ENDPOINT_LABEL: &str = "/passthrough_route";

/// Cap on the recorded `client_identity` value (an operator-injected
/// header, but the value itself arrives from the wire).
const IDENTITY_VALUE_CAP: usize = 256;

/// Headers ALWAYS stripped before forwarding upstream, regardless of route
/// configuration: HTTP protocol metadata the outbound client recomputes,
/// plus RFC 7230 §6.1 hop-by-hop headers.
const ALWAYS_STRIP: &[&str] = &[
    "host",
    "content-length",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "traceparent",
    // W3C trace context (AISIX-Cloud#1279): stripped unconditionally, same
    // as the standard pipeline's never-forward set — the caller's trace
    // ids are not the provider's to see, and passthrough forwarded them
    // verbatim before this entry existed. A future provider-side
    // propagation opt-in would inject the gateway's own context instead.
    "tracestate",
    "transfer-encoding",
    "upgrade",
    // Gateway-owned correlation id: the dispatch sets its own value, and
    // `RequestBuilder::header` appends — an inbound copy would reach the
    // upstream as a duplicate.
    "x-aisix-request-id",
];

/// Fixed 410 message for the removed implicit tunnel.
///
/// COMPAT-SINCE: 0.10.0 #1010 — 0.10.0 removed the implicit
/// `/passthrough/:provider/*rest` tunnel, and a migration pointer beats a
/// bare 404 while un-migrated callers are still out there; the public docs
/// promise the 410 for one release, then plain 404s.
///
/// Retiring it drops this constant, the `entry` branch that serves it, its
/// WARN and the `UNRESOLVED_LABEL` metric series, and leaves the namespace
/// entirely the operator's to claim with explicit routes.
const LEGACY_TUNNEL_GONE: &str = "the implicit /passthrough/:provider tunnel has been removed; \
     configure an explicit passthrough_route resource for this path \
     (see the provider passthrough documentation for the migration)";

// ---------------------------------------------------------------------------
// Routing entry points
// ---------------------------------------------------------------------------

/// `true` when any enabled route's `hosts` matches the request's inbound
/// host. The cheap pre-routing probe [`host_dispatch`] uses to decide
/// whether the request belongs to a foreign-host route at all.
fn has_host_match(snapshot: &aisix_core::AisixSnapshot, host: Option<&str>) -> bool {
    let Some(host) = host else { return false };
    snapshot
        .passthrough_routes
        .entries()
        .iter()
        .any(|e| e.value.enabled && e.value.matches_host(host))
}

/// The request's inbound host: the `Host` header (origin-form requests),
/// falling back to the URI authority (absolute-form requests from a
/// chained proxy). Lowercased, `:port` stripped.
fn inbound_host(req: &Request) -> Option<String> {
    let raw = req
        .headers()
        .get(header::HOST)
        .and_then(|v| v.to_str().ok())
        .or_else(|| req.uri().authority().map(|a| a.as_str()))?;
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

/// Pre-routing middleware: dispatch foreign-host traffic to the entry
/// stack before the typed router can match on the path. See the module
/// doc. The dispatch target is a layered router carrying the same shared
/// per-request layers as the main stack (body limits, in-flight/cancel
/// telemetry, the Server-header override) — calling the bare handler here
/// would silently exempt foreign-host traffic from all of them.
pub async fn host_dispatch(
    State((state, entry_stack)): State<(ProxyState, axum::Router)>,
    req: Request,
    next: Next,
) -> Response {
    let snapshot = state.snapshot.load();
    let matched = has_host_match(&snapshot, inbound_host(&req).as_deref());
    drop(snapshot);
    if matched {
        use tower::ServiceExt;
        return match entry_stack.oneshot(req).await {
            Ok(resp) => resp,
            // `Router`'s service error is `Infallible`.
            Err(never) => match never {},
        };
    }
    next.run(req).await
}

/// One matched route plus how it matched (what the target path remainder
/// is).
struct MatchedRoute {
    entry: Arc<ResourceEntry<PassthroughRoute>>,
    /// The request path with the route's `path_prefix` stripped when the
    /// match used one; the full path for host-only matches. Empty or
    /// `/`-leading.
    remainder: String,
    /// Whether the route's `path_prefix` was STRIPPED from the path (a
    /// `target_url` mount). Enables the `/v1` dedup, which only makes
    /// sense when an operator-written prefix joins an operator-written
    /// target — never for a `preserve_host` mirror of the caller's URL.
    prefix_matched: bool,
    /// The inbound host, when one was present. Needed for
    /// `preserve_host` targets.
    host: Option<String>,
}

/// `true` when `path` sits under `prefix` on a segment boundary:
/// `/copilot` matches `/copilot` and `/copilot/x`, never `/copilotx`.
fn path_under_prefix(path: &str, prefix: &str) -> bool {
    match path.strip_prefix(prefix) {
        Some(rest) => rest.is_empty() || rest.starts_with('/'),
        None => false,
    }
}

/// Select the route serving `(host, path)`, or `None`.
///
/// A route matches when every dimension it configures matches (`hosts`,
/// `path_prefix`, or both). The most specific match wins: host-matched
/// routes beat path-only ones, longer path prefixes beat shorter, and a
/// residual tie picks the smallest resource id so replicas agree.
fn match_route(
    snapshot: &aisix_core::AisixSnapshot,
    host: Option<&str>,
    path: &str,
) -> Option<MatchedRoute> {
    let mut best: Option<(bool, usize, Arc<ResourceEntry<PassthroughRoute>>)> = None;
    for e in snapshot.passthrough_routes.entries() {
        let r = &e.value;
        if !r.enabled {
            continue;
        }
        let host_matched = match &r.hosts {
            Some(_) => match host {
                Some(h) => r.matches_host(h),
                None => false,
            },
            None => false,
        };
        if r.hosts.is_some() && !host_matched {
            continue;
        }
        let prefix_len = match &r.path_prefix {
            Some(p) => {
                if !path_under_prefix(path, p) {
                    continue;
                }
                p.len()
            }
            None => 0,
        };
        if r.hosts.is_none() && r.path_prefix.is_none() {
            // Schema-unreachable, but never let such a row match everything.
            continue;
        }
        let candidate = (host_matched, prefix_len, Arc::clone(&e));
        best = Some(match best.take() {
            None => candidate,
            Some(cur) => {
                let cur_rank = (cur.0, cur.1);
                let cand_rank = (candidate.0, candidate.1);
                match cand_rank.cmp(&cur_rank) {
                    std::cmp::Ordering::Greater => candidate,
                    std::cmp::Ordering::Equal if candidate.2.id < cur.2.id => candidate,
                    _ => cur,
                }
            }
        });
    }
    best.map(|(_, prefix_len, entry)| {
        // A `preserve_host` route mirrors an upstream that owns its own
        // path space (the forward-proxy shape): the prefix is a MATCH
        // condition there, not a mount point, so the path is relayed whole.
        // Stripping is for `target_url` routes, where the prefix is the
        // gateway-side mount and the remainder joins the target's base.
        let prefix_matched = prefix_len > 0 && !entry.value.preserve_host;
        let remainder = if prefix_matched {
            path[prefix_len..].to_string()
        } else {
            path.to_string()
        };
        MatchedRoute {
            entry,
            remainder,
            prefix_matched,
            host: host.map(str::to_string),
        }
    })
}

/// Router fallback + host-dispatch target. Resolves the route, runs the
/// pipeline, and owns the request-level telemetry for both outcomes.
pub async fn entry(
    State(state): State<ProxyState>,
    client: crate::client_ip::ClientContext,
    req: Request,
) -> Response {
    let started = Instant::now();
    let snapshot = state.snapshot.load();
    let host = inbound_host(&req);
    let path = req.uri().path().to_string();

    let method = req.method().clone();
    let request_id = client.request_id.clone();

    let Some(matched) = match_route(&snapshot, host.as_deref(), &path) else {
        if path.starts_with("/passthrough/") {
            // The removed implicit tunnel's namespace: a fixed 410 with
            // the migration pointer beats a bare 404 while callers migrate
            // (release-debt marker on LEGACY_TUNNEL_GONE).
            // Logged AND counted (unresolved-provider labels, like the
            // old tunnel's failure path) so operators can locate and
            // size un-migrated callers.
            tracing::warn!(
                path = %path,
                "removed /passthrough tunnel hit with no matching passthrough_route (410)",
            );
            let err = ProxyError::Gone(LEGACY_TUNNEL_GONE.into());
            let status = err.status().as_u16();
            let elapsed = started.elapsed();
            emit_access_log(
                &method,
                &path,
                UNRESOLVED_LABEL,
                "",
                status,
                elapsed,
                &request_id,
                None,
                Some(&err),
            );
            crate::request_metrics::record(
                &state,
                ENDPOINT_LABEL,
                crate::request_metrics::Caller::unattributed(None),
                crate::request_metrics::Upstream {
                    provider: UNRESOLVED_LABEL,
                    model: PASSTHROUGH_MODEL_LABEL,
                    ..Default::default()
                },
                status,
                elapsed,
            );
            return err.into_response();
        }
        // Preserve the router's pre-existing no-match behavior exactly.
        return StatusCode::NOT_FOUND.into_response();
    };

    let route_name = matched.entry.value.name.clone();

    match dispatch(&state, &snapshot, &matched, req, &client, started).await {
        Ok(resp) => resp,
        Err(RouteError { error, auth }) => {
            let status = error.status().as_u16();
            let elapsed = started.elapsed();
            let api_key_id = auth.as_deref().unwrap_or("");
            emit_access_log(
                &method,
                &path,
                &route_name,
                api_key_id,
                status,
                elapsed,
                &request_id,
                None,
                Some(&error),
            );
            crate::request_metrics::record(
                &state,
                ENDPOINT_LABEL,
                crate::request_metrics::Caller::unattributed(auth.as_deref()),
                crate::request_metrics::Upstream {
                    provider: BYO_PROVIDER_LABEL,
                    model: PASSTHROUGH_MODEL_LABEL,
                    ..Default::default()
                },
                status,
                elapsed,
            );
            let mut event = crate::usage_attr::build_error_usage_event(
                "passthrough",
                &request_id,
                "",
                api_key_id,
                status,
                error.kind(),
                &client,
            );
            // The route matched before the pipeline failed, so a rejected
            // request still attributes to it — an operator triaging 401s
            // per route needs the name on the event, not just in the log.
            event.passthrough_route_name = route_name.clone();
            let usage_model =
                crate::usage_attr::usage_event_model_label(&snapshot, &event.requested_model);
            crate::usage_attr::emit_prepared_usage_event(
                &state,
                &snapshot,
                "passthrough_route",
                event.clone(),
                crate::usage_attr::usage_event_labels(
                    &usage_model,
                    &crate::usage_attr::ResolvedPk::unresolved(),
                ),
                client.trace.as_ref(),
            );
            error.into_response()
        }
    }
}

/// Pipeline error plus whatever caller identity was established before it
/// fired, so the error-path telemetry can still attribute the request.
struct RouteError {
    error: ProxyError,
    auth: Option<String>,
}

impl RouteError {
    fn pre_auth(error: ProxyError) -> Self {
        Self { error, auth: None }
    }
    fn of(error: ProxyError, auth: &AuthenticatedKey) -> Self {
        Self {
            error,
            auth: Some(auth.entry.id.clone()),
        }
    }
}

// ---------------------------------------------------------------------------
// The pipeline
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
async fn dispatch(
    state: &ProxyState,
    snapshot: &aisix_core::AisixSnapshot,
    matched: &MatchedRoute,
    req: Request,
    client: &crate::client_ip::ClientContext,
    started: Instant,
) -> Result<Response, RouteError> {
    let route = &matched.entry.value;
    let route_id: &str = &matched.entry.id;

    // Route-level source allowlist. For `anonymous` it is the only gate in
    // front of the bound principal; for the other modes optional hardening.
    if !source_allowed(route, &client.source_ip) {
        tracing::warn!(
            route = %route.name,
            source_ip = %client.source_ip,
            "request rejected: client IP not in passthrough route source_cidrs"
        );
        return Err(RouteError::pre_auth(ProxyError::RouteIpRestricted(
            route.name.clone(),
        )));
    }

    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let query = req.uri().query().map(str::to_string);
    let incoming_headers = req.headers().clone();

    // Gateway authentication per the route's mode. Every mode ends in a
    // real AuthenticatedKey so ACL / rate limits / budget / attribution
    // downstream need no per-mode branches.
    let auth = authenticate(
        state,
        snapshot,
        route,
        &incoming_headers,
        client,
        &method,
        &path,
    )
    .await
    .map_err(RouteError::pre_auth)?;

    // Route ACL: explicit grant, mirroring allowed_agents.
    if !auth.key().can_access_route(&route.name) {
        return Err(RouteError::of(
            ProxyError::RouteForbidden(route.name.clone()),
            &auth,
        ));
    }

    // Resolve the upstream credential source before spending work on the
    // body: a misconfigured route should fail fast and identically on
    // every request.
    let pk_entry = match route.credential_mode {
        PassthroughCredentialMode::Inject => {
            let id = route.provider_key_id.as_deref().unwrap_or_default();
            let entry = snapshot.provider_keys.get_by_id(id).ok_or_else(|| {
                RouteError::of(
                    ProxyError::InvalidRequest(format!(
                        "passthrough route {:?} references an unknown provider key",
                        route.name
                    )),
                    &auth,
                )
            })?;
            if entry.value.api_key.is_empty() {
                return Err(RouteError::of(
                    ProxyError::InvalidRequest(format!(
                        "passthrough route {:?} provider_key has empty api_key",
                        route.name
                    )),
                    &auth,
                ));
            }
            Some(entry)
        }
        PassthroughCredentialMode::ForwardClient => None,
    };

    let base = if route.preserve_host {
        // `preserve_host` is only schema-legal with a `hosts` allowlist,
        // and only host-matched requests reach a hosts-bearing route — so
        // the derived target is bounded by the operator's own list.
        let host = matched.host.as_deref().ok_or_else(|| {
            RouteError::of(
                ProxyError::InvalidRequest(format!(
                    "passthrough route {:?} preserves the host but the request carries none",
                    route.name
                )),
                &auth,
            )
        })?;
        format!("https://{host}")
    } else {
        route
            .target_url
            .as_deref()
            .unwrap_or_default()
            .trim_end_matches('/')
            .to_string()
    };

    // Build the target URL from the matched remainder. The `/v1` dedup
    // (#164) only applies when an operator-written prefix joins an
    // operator-written target; a host-matched full path is the real
    // client's own URL and is never rewritten.
    let rest_raw = matched.remainder.trim_start_matches('/');
    let rest = if matched.prefix_matched {
        strip_redundant_version_segment(&base, rest_raw)
    } else {
        rest_raw
    };
    let url = if rest.is_empty() {
        base.clone()
    } else {
        format!("{base}/{rest}")
    };
    let url = match &query {
        Some(q) => format!("{url}?{q}"),
        None => url,
    };

    // End-user identity injected by the upstream device, captured before
    // the strip pass and recorded on the usage event.
    let client_identity = route
        .identity_header
        .as_deref()
        .and_then(|h| incoming_headers.get(h))
        .and_then(|v| v.to_str().ok())
        .map(|v| {
            v.chars()
                .filter(|c| !c.is_control())
                .take(IDENTITY_VALUE_CAP)
                .collect::<String>()
        })
        .unwrap_or_default();

    // Buffer the request body under the configured cap (guardrails and
    // the protocol probe need the whole thing; the tunnel forwards it
    // verbatim).
    let body_limit = state.request_body_limit_bytes;
    let body_bytes: Bytes =
        axum::body::to_bytes(req.into_body(), crate::error::body_read_cap(body_limit))
            .await
            .map_err(|err| {
                RouteError::of(
                    if crate::error::is_length_limit_error(&err) {
                        ProxyError::RequestTooLarge {
                            limit_bytes: body_limit,
                        }
                    } else {
                        ProxyError::InvalidRequest("failed to read request body".into())
                    },
                    &auth,
                )
            })?;

    // Guardrail chain for this route (+ the caller's key/team/env scopes).
    let guardrail_ctx = aisix_guardrails::RequestContext {
        passthrough_route_id: route_id,
        model_id: "",
        mcp_server_id: "",
        api_key_id: &auth.entry.id,
        team_id: auth.key().team_id.as_deref(),
    };
    let resolved_chain = state.guardrail_index.resolve(&guardrail_ctx);
    let mut monitor_hits: Vec<aisix_core::GuardrailMonitorHit> = Vec::new();

    // Envelope detection: once per exchange, from the request body's
    // top-level keys; the response and stream frames reuse it.
    let protocol = detect_protocol(&body_bytes);

    // INPUT guardrails on the (envelope-extracted) request text.
    if !resolved_chain.is_empty() {
        let text = request_guardrail_text(protocol, &body_bytes);
        let chat = aisix_gateway::ChatFormat::new(
            route.name.clone(),
            vec![aisix_gateway::ChatMessage::user(text)],
        );
        let (verdict, hits) =
            aisix_guardrails::Guardrail::check_input_observed(&resolved_chain, &chat).await;
        monitor_hits.extend(hits);
        if let aisix_guardrails::GuardrailVerdict::Block {
            reason,
            guardrail_name,
        } = verdict
        {
            // Per #153 the matched-pattern detail stays in ops logs only.
            tracing::warn!(
                guardrail_hook = "input",
                route = %route.name,
                reason = %reason,
                "guardrail blocked passthrough-route request",
            );
            return Err(RouteError::of(
                ProxyError::ContentFiltered(crate::error::guardrail_block_message(
                    "request",
                    guardrail_name.as_deref(),
                )),
                &auth,
            ));
        }
    }

    // Content capture (exporter-gated): the request body text, in the same
    // exporter-only channel the typed endpoints use. Captured after the
    // input guardrail so a blocked request records nothing.
    let content_cap = aisix_obs::content_capture_cap(
        snapshot
            .observability_exporters
            .entries()
            .iter()
            .map(|e| &e.value),
    );
    // The whole request body, as the typed endpoints capture it — they
    // serialize the parsed request, not the text they extracted from it.
    // Structure matters to an audit (roles, tool definitions, parameters),
    // and the capture truncator is JSON-aware, so it reduces the body
    // rather than cutting it mid-token.
    let captured_prompt = content_cap.map(|_| String::from_utf8_lossy(&body_bytes).into_owned());

    // The alias the caller addressed, for the usage event's attribution.
    let requested_model = body_model_name(protocol, &body_bytes);

    // Rate limits AFTER the input guardrail so a content block doesn't burn
    // an RPM slot (matching the typed endpoints). The body's `model` field
    // reserves a configured Model's own layers only for `inject` routes,
    // scoped to the ProviderKey's provider — the #805 contract, minus the
    // credential borrowing. `forward_client` upstreams are not configured
    // Models, so a same-named model of some provider must never match.
    let model_rl = pk_entry
        .as_ref()
        .map(|pk| pk.value.provider.to_ascii_lowercase())
        .filter(|prov| !prov.is_empty())
        .and_then(|prov| body_model_rate_limit(snapshot, &prov, &body_bytes));
    let _reservation = crate::quota::enforce(state, snapshot, &auth, model_rl.as_ref())
        .await
        .map_err(|e| RouteError::of(e, &auth))?;

    // ----- outbound request -----

    let tls = pk_entry.as_ref().and_then(|pk| pk.value.tls.as_ref());
    let http_client = crate::http_client::client_for(tls);

    // Strip set: protocol metadata always; per-mode credential handling.
    let mut strip: std::collections::HashSet<String> =
        ALWAYS_STRIP.iter().map(|s| (*s).to_string()).collect();
    if let Some(h) = route.identity_header.as_deref() {
        strip.insert(h.to_ascii_lowercase());
    }
    match route.credential_mode {
        PassthroughCredentialMode::Inject => {
            // The ProviderKey's configurable strip list (defaults:
            // authorization, cookie, set-cookie, x-api-key — #411).
            if let Some(pk) = pk_entry.as_ref() {
                strip.extend(
                    pk.value
                        .strip_headers
                        .iter()
                        .map(|s| s.to_ascii_lowercase()),
                );
            }
            // The two slots the injection below writes are stripped
            // UNCONDITIONALLY — `RequestBuilder::header` appends, so a
            // `strip_headers` override that keeps `authorization` would
            // put the caller's credential on the wire beside the injected
            // one. Explicit client-credential forwarding is what
            // `forward_client` is for; inject never double-sends.
            strip.insert("authorization".into());
            strip.insert("x-api-key".into());
            // The header-key slot never goes upstream either.
            if let Some(h) = route.auth_header_name.as_deref() {
                strip.insert(h.to_ascii_lowercase());
            }
        }
        PassthroughCredentialMode::ForwardClient => {
            // BYO: forward the caller's credentials, strip exactly the
            // headers the GATEWAY consumed, so its own credential never
            // leaks upstream.
            match route.auth_mode {
                PassthroughAuthMode::GatewayKey => {
                    strip.insert("authorization".into());
                    strip.insert("x-api-key".into());
                }
                PassthroughAuthMode::HeaderKey => {
                    if let Some(h) = route.auth_header_name.as_deref() {
                        strip.insert(h.to_ascii_lowercase());
                    }
                }
                PassthroughAuthMode::Anonymous => {}
            }
        }
    }

    let mut builder = http_client.request(method.clone(), &url);
    for (name, value) in &incoming_headers {
        if strip.contains(&name.as_str().to_ascii_lowercase()) {
            continue;
        }
        builder = builder.header(name, value);
    }

    // Inject the gateway-held upstream credential (inject mode only).
    // Strip ran first, so the wire stays single-valued (#411 ordering).
    if let Some(pk) = pk_entry.as_ref() {
        let api_key = pk.value.api_key.as_str();
        let provider_lower = pk.value.provider.to_ascii_lowercase();
        if provider_lower == "anthropic" {
            // Anthropic's documented auth shape (#166): `x-api-key` +
            // `anthropic-version`, never a redundant Bearer alongside.
            builder = builder.header("x-api-key", api_key);
            builder = builder.header("anthropic-version", "2023-06-01");
        } else {
            builder = builder.header(header::AUTHORIZATION, format!("Bearer {api_key}"));
        }
    }

    builder = builder.header("x-aisix-request-id", &client.request_id);

    if !body_bytes.is_empty() {
        builder = builder.body(body_bytes.clone());
    }

    // Exchange bound. The timeout (route override, else the gateway
    // default) must not bound the relay itself — a healthy long-lived SSE
    // stream is the point — but a blackholed upstream still can't pin the
    // connection: the header phase (and, below, a non-SSE body read) get
    // the bound via an explicit timer.
    let exchange_timeout = route
        .timeout_ms
        .map(Duration::from_millis)
        .or(state.default_timeouts.request);

    let bridge_timeout = |d: Duration| aisix_gateway::BridgeError::Timeout {
        elapsed_ms: d.as_millis().min(u64::MAX as u128) as u64,
        cause: "passthrough route upstream exchange".into(),
    };
    // The attempt begins here. `upstream_latency_ms` / `upstream_ttft_ms`
    // are attempt-scoped by contract, so they must not count the gateway's
    // own pre-dispatch work (auth, guardrail scan, rate-limit reservation)
    // — that belongs to `downstream_latency_ms`, which runs from `started`.
    let attempt_started = Instant::now();
    let send_fut = builder.send();
    let sent = match exchange_timeout {
        Some(d) => match tokio::time::timeout(d, send_fut).await {
            Ok(r) => r,
            Err(_) => return Err(RouteError::of(ProxyError::Bridge(bridge_timeout(d)), &auth)),
        },
        None => send_fut.await,
    };
    let upstream_resp = sent.map_err(|e| {
        RouteError::of(
            ProxyError::Bridge(crate::dispatch::reqwest_error_to_bridge(&e, started)),
            &auth,
        )
    })?;

    let status = upstream_resp.status();
    let resp_headers = upstream_resp.headers().clone();
    let is_sse = resp_headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|v| {
            v.trim_start()
                .to_ascii_lowercase()
                .starts_with("text/event-stream")
        })
        .unwrap_or(false);

    let mut telemetry = RouteTelemetry {
        state: state.clone(),
        route_name: route.name.clone(),
        trace: client.trace.clone(),
        provider_label: pk_entry
            .as_ref()
            .map(|pk| pk.value.provider.to_ascii_lowercase())
            .filter(|p| !p.is_empty())
            .unwrap_or_else(|| BYO_PROVIDER_LABEL.to_string()),
        pk_id: pk_entry
            .as_ref()
            .map(|pk| pk.id.to_string())
            .unwrap_or_default(),
        method: method.clone(),
        path: path.clone(),
        request_id: client.request_id.clone(),
        api_key_id: auth.entry.id.clone(),
        jwt: auth.jwt.clone(),
        anonymous: auth.anonymous,
        client_identity,
        client_source_ip: client.source_ip.clone(),
        client_user_agent: client.user_agent.clone(),
        started,
        attempt_started,
        status: status.as_u16(),
        usage: PassthroughUsage::default(),
        requested_model,
        upstream_ttft_ms: 0,
        downstream_first_ms: None,
        stream_reached_end: false,
        streaming: false,
        error_class: String::new(),
        error_message: String::new(),
        monitor_hits,
        captured_prompt,
        content_cap: content_cap.map(|c| c as usize),
        response_text: String::new(),
        guardrail_blocked: false,
        emitted: false,
    };

    if is_sse {
        telemetry.streaming = true;
        return Ok(stream_response(
            protocol,
            resolved_chain,
            upstream_resp,
            resp_headers,
            status,
            telemetry,
            &client.request_id,
        ));
    }

    // ----- buffered response -----

    // A non-SSE answer: the reqwest request carries no built-in timeout,
    // so the body read gets the exchange bound explicitly (same blackhole
    // guard as the send).
    let body_fut = upstream_resp.bytes();
    let read = match exchange_timeout {
        Some(d) => match tokio::time::timeout(d, body_fut).await {
            Ok(r) => r,
            Err(_) => {
                telemetry.emitted = true;
                return Err(RouteError::of(ProxyError::Bridge(bridge_timeout(d)), &auth));
            }
        },
        None => body_fut.await,
    };
    let resp_body = read.map_err(|e| {
        telemetry.emitted = true;
        RouteError::of(
            ProxyError::Bridge(aisix_gateway::BridgeError::UpstreamDecode(e.to_string())),
            &auth,
        )
    })?;

    // OUTPUT guardrails on the (envelope-extracted) response text.
    if !resolved_chain.is_empty() {
        let text = response_guardrail_text(protocol, &resp_body);
        let synth = aisix_gateway::ChatResponse {
            id: String::new(),
            model: route.name.clone(),
            message: aisix_gateway::ChatMessage::assistant(text),
            finish_reason: aisix_gateway::FinishReason::Stop,
            usage: aisix_gateway::UsageStats::default(),
        };
        let (verdict, hits) =
            aisix_guardrails::Guardrail::check_output_observed(&resolved_chain, &synth).await;
        telemetry.monitor_hits.extend(hits);
        if let aisix_guardrails::GuardrailVerdict::Block {
            reason,
            guardrail_name,
        } = verdict
        {
            tracing::warn!(
                guardrail_hook = "output",
                route = %route.name,
                reason = %reason,
                "guardrail blocked passthrough-route response",
            );
            telemetry.guardrail_blocked = true;
            // The telemetry guard has not emitted yet; drop it silently and
            // let the shared error path report the 422.
            telemetry.emitted = true;
            return Err(RouteError::of(
                ProxyError::ContentFiltered(crate::error::guardrail_block_message(
                    "response",
                    guardrail_name.as_deref(),
                )),
                &auth,
            ));
        }
    }

    if let Some(u) = response_usage(protocol, &resp_body) {
        telemetry.usage.merge(u);
    }
    if telemetry.content_cap.is_some() {
        telemetry.response_text = response_guardrail_text(protocol, &resp_body);
    }

    let mut response = Response::builder()
        .status(status)
        .body(Body::from(resp_body))
        .unwrap();
    copy_safe_headers(&resp_headers, response.headers_mut());
    if let Ok(hv) = HeaderValue::from_str(&client.request_id) {
        response
            .headers_mut()
            .insert(header::HeaderName::from_static("x-aisix-request-id"), hv);
    }

    telemetry.emit();
    Ok(response)
}

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

/// Authenticate the caller per the route's `auth_mode`, ending in a real
/// [`AuthenticatedKey`] in every mode.
async fn authenticate(
    state: &ProxyState,
    snapshot: &aisix_core::AisixSnapshot,
    route: &PassthroughRoute,
    headers: &HeaderMap,
    client: &crate::client_ip::ClientContext,
    method: &Method,
    path: &str,
) -> Result<AuthenticatedKey, ProxyError> {
    let ctx = crate::auth::DenialContext {
        method: method.as_str(),
        path,
        request_id: &client.request_id,
        source_ip: crate::auth::LazySourceIp::Ready(&client.source_ip),
    };
    match route.auth_mode {
        PassthroughAuthMode::GatewayKey => {
            let token = bearer_of(headers.get(header::AUTHORIZATION))
                .or_else(|| raw_of(headers.get("x-api-key")))
                .ok_or(ProxyError::MissingAuth)?;
            crate::auth::authenticate_token(state, &token, ctx).await
        }
        PassthroughAuthMode::HeaderKey => {
            let name = route.auth_header_name.as_deref().unwrap_or_default();
            let token = headers
                .get(name)
                .and_then(|v| v.to_str().ok())
                .map(|v| v.strip_prefix("Bearer ").unwrap_or(v).trim().to_string())
                .filter(|v| !v.is_empty())
                .ok_or_else(|| ProxyError::MissingRouteAuthHeader(name.to_string()))?;
            crate::auth::authenticate_token(state, &token, ctx).await
        }
        PassthroughAuthMode::Anonymous => {
            let id = route.anonymous_key_id.as_deref().unwrap_or_default();
            let entry = snapshot.apikeys.get_by_id(id).ok_or_else(|| {
                // Operator misconfiguration, not a caller mistake — but
                // never an anonymous pass.
                ProxyError::InvalidRequest(format!(
                    "passthrough route {:?} anonymous key is not configured",
                    route.name
                ))
            })?;
            // The bound principal keeps its full lifecycle: a disabled or
            // expired anonymous key closes the route.
            if entry.value.disabled {
                return Err(ProxyError::ApiKeyDisabled);
            }
            if entry.value.expires_at.is_some() && entry.value.is_expired_at(chrono::Utc::now()) {
                return Err(ProxyError::ApiKeyExpired);
            }
            state.metrics.record_auth_decision("anonymous", true, "");
            Ok(AuthenticatedKey {
                entry,
                jwt: None,
                anonymous: true,
            })
        }
    }
}

fn bearer_of(v: Option<&HeaderValue>) -> Option<String> {
    let s = v?.to_str().ok()?;
    let token = s
        .strip_prefix("Bearer ")
        .or_else(|| s.strip_prefix("bearer "))?;
    let token = token.trim();
    (!token.is_empty()).then(|| token.to_string())
}

fn raw_of(v: Option<&HeaderValue>) -> Option<String> {
    let s = v?.to_str().ok()?.trim();
    (!s.is_empty()).then(|| s.to_string())
}

/// Route-level source allowlist: unset means unrestricted (the schema
/// forces a non-empty list for `anonymous` routes).
fn source_allowed(route: &PassthroughRoute, source_ip: &str) -> bool {
    match route.source_cidrs.as_deref() {
        Some(ranges) if !ranges.is_empty() => crate::client_ip::ip_in_cidrs(source_ip, ranges),
        _ => true,
    }
}

// ---------------------------------------------------------------------------
// Protocol-aware body handling
// ---------------------------------------------------------------------------

/// Concatenated text content of an OpenAI-style `content` value: a plain
/// string, or an array of parts with `{"type":"text","text":...}`.
fn content_text(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// The text a guardrail scans from ONE chat-envelope message: its content
/// plus the whole serialized `tool_calls` payload.
///
/// The tool-call half is what the typed endpoints scan (`message_scan_text`
/// in the guardrails crate), and it is not optional: a request whose only
/// sensitive text sits in a tool call's `arguments` would otherwise pass a
/// deny-list that the same body sent to `/v1/chat/completions` trips.
/// Serialising the whole payload means no function name or argument can
/// escape inspection regardless of the provider-specific shape.
fn message_scan_text(msg: &serde_json::Value) -> String {
    let mut parts: Vec<String> = Vec::new();
    let content = msg.get("content").map(content_text).unwrap_or_default();
    if !content.is_empty() {
        parts.push(content);
    }
    if let Some(tool_calls) = msg.get("tool_calls").filter(|t| !t.is_null()) {
        parts.push(tool_calls.to_string());
    }
    parts.join("\n")
}

/// The body envelope detected for one exchange. Not configuration:
/// detected per request from the body's top-level keys
/// ([`detect_protocol`]) and sticky for the exchange — the buffered
/// response and every stream frame are read with the same detection. It
/// drives extraction (guardrail text, capture, usage) only; the relay
/// forwards bytes verbatim regardless.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PassthroughProtocol {
    /// No recognized envelope: bodies are opaque (guardrails scan them as
    /// one lossy-UTF-8 text; buffered responses are not probed for usage).
    /// A streamed opaque response reports usage from an explicit `usage`
    /// object, or — for the flat token shape agent backends use — only
    /// from a frame the server itself labels one (`event: token_usage`),
    /// see [`frame_delta`].
    Raw,
    /// OpenAI-compatible chat envelope (`messages`, streamed
    /// `choices[].delta.content`, final-chunk / response `usage`). Also
    /// carries Anthropic Messages traffic, whose request is the same
    /// `messages` shape: its usage spellings and its `message_start` /
    /// `message_delta` split are read alongside the OpenAI ones.
    OpenaiChat,
    /// OpenAI-compatible legacy completions / FIM envelope (`prompt` [+
    /// `suffix`], streamed `choices[].text`, `usage`).
    OpenaiCompletions,
    /// OpenAI Responses API envelope: `input` on the request, `output`
    /// items on the response, `response.output_text.delta` events while
    /// streaming, and `usage` in the `input_tokens`/`output_tokens`
    /// spelling — carried on the terminal `response.completed` event when
    /// the response streams.
    OpenaiResponses,
}

/// Detect the request envelope from the body's top-level keys. The three
/// LLM envelopes are structurally exclusive — `messages`, `input` and
/// `prompt` are each the required carrier field of exactly one API — so
/// real LLM traffic detects unambiguously, and everything else (JSON-RPC,
/// REST, non-JSON, empty/GET bodies) is `Raw`. An unknown API colliding
/// with a carrier key costs nothing: detection drives extraction only,
/// and extraction degrades to the whole body when the detected shape
/// yields no text.
fn detect_protocol(body: &[u8]) -> PassthroughProtocol {
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(body) else {
        return PassthroughProtocol::Raw;
    };
    if v.get("messages").is_some_and(serde_json::Value::is_array) {
        PassthroughProtocol::OpenaiChat
    } else if v
        .get("input")
        .is_some_and(|i| i.is_string() || i.is_array())
    {
        PassthroughProtocol::OpenaiResponses
    } else if v
        .get("prompt")
        .is_some_and(|p| p.is_string() || p.is_array())
    {
        PassthroughProtocol::OpenaiCompletions
    } else {
        PassthroughProtocol::Raw
    }
}

/// Cap on the recorded `requested_model` value. The body is the caller's,
/// so the alias is bounded before it reaches telemetry.
const REQUESTED_MODEL_CAP: usize = 128;

/// The model alias the caller addressed, from a DETECTED envelope's own
/// `model` field — what the typed endpoint serving that envelope records
/// as `UsageEvent::requested_model`.
///
/// Read only for a recognised envelope: an opaque body's `model`-shaped key
/// belongs to some other API and means nothing the gateway can attribute.
/// The Prometheus side is already collapse-guarded (an unregistered name
/// folds to the `unresolved` sentinel), so an arbitrary alias here cannot
/// mint label cardinality.
fn body_model_name(protocol: PassthroughProtocol, body: &[u8]) -> String {
    if matches!(protocol, PassthroughProtocol::Raw) {
        return String::new();
    }
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(body) else {
        return String::new();
    };
    v.get("model")
        .and_then(|m| m.as_str())
        .map(|m| {
            m.chars()
                .filter(|c| !c.is_control())
                .take(REQUESTED_MODEL_CAP)
                .collect::<String>()
        })
        .unwrap_or_default()
}

/// The request text a guardrail scans, per the detected envelope.
/// Extraction is best-effort: a shape that yields no text degrades to the
/// raw lossy-UTF-8 body, so detection never loses audit coverage.
fn request_guardrail_text(protocol: PassthroughProtocol, body: &[u8]) -> String {
    let raw = || String::from_utf8_lossy(body).into_owned();
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(body) else {
        return raw();
    };
    let extracted = match protocol {
        PassthroughProtocol::Raw => return raw(),
        PassthroughProtocol::OpenaiChat => v
            .get("messages")
            .and_then(|m| m.as_array())
            .map(|msgs| {
                msgs.iter()
                    .map(message_scan_text)
                    .filter(|t| !t.is_empty())
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default(),
        // Responses API: `input` is either a bare string or an array of
        // items whose `content` parts carry the text.
        PassthroughProtocol::OpenaiResponses => match v.get("input") {
            Some(serde_json::Value::String(t)) => t.clone(),
            Some(serde_json::Value::Array(items)) => items
                .iter()
                .filter_map(|i| i.get("content").map(content_text))
                .filter(|t| !t.is_empty())
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        },
        PassthroughProtocol::OpenaiCompletions => {
            let prompt = v.get("prompt").map(|p| match p {
                serde_json::Value::Array(items) => items
                    .iter()
                    .filter_map(|i| i.as_str())
                    .collect::<Vec<_>>()
                    .join("\n"),
                other => content_text(other),
            });
            let suffix = v.get("suffix").and_then(|s| s.as_str());
            let mut out = prompt.unwrap_or_default();
            if let Some(s) = suffix {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(s);
            }
            out
        }
    };
    if extracted.is_empty() {
        raw()
    } else {
        extracted
    }
}

/// The response text a guardrail scans / the capture records, per the
/// route's protocol hint. Best-effort like the request side.
fn response_guardrail_text(protocol: PassthroughProtocol, body: &[u8]) -> String {
    let raw = || String::from_utf8_lossy(body).into_owned();
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(body) else {
        return raw();
    };
    // Responses answers with `output` items, not `choices`.
    if matches!(protocol, PassthroughProtocol::OpenaiResponses) {
        let joined = v
            .get("output")
            .and_then(|o| o.as_array())
            .map(|items| {
                items
                    .iter()
                    .filter_map(|i| i.get("content").map(content_text))
                    .filter(|t| !t.is_empty())
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default();
        return if joined.is_empty() { raw() } else { joined };
    }
    let choices = match protocol {
        PassthroughProtocol::Raw => return raw(),
        _ => v.get("choices").and_then(|c| c.as_array()),
    };
    let Some(choices) = choices else { return raw() };
    let texts: Vec<String> = choices
        .iter()
        .filter_map(|c| match protocol {
            PassthroughProtocol::OpenaiChat => c.get("message").map(message_scan_text),
            PassthroughProtocol::OpenaiCompletions => {
                c.get("text").and_then(|t| t.as_str()).map(str::to_string)
            }
            // Unreachable: handled above by the `output` branch.
            PassthroughProtocol::OpenaiResponses | PassthroughProtocol::Raw => None,
        })
        .filter(|t| !t.is_empty())
        .collect();
    if texts.is_empty() {
        raw()
    } else {
        texts.join("\n")
    }
}

/// Every token dimension a passthrough exchange can report, mirroring the
/// token fields of [`aisix_obs::UsageEvent`] 1:1 so a route reports what
/// the typed endpoint serving the same envelope would.
///
/// Populated from the union of spellings the relayed APIs use — OpenAI's
/// nested `*_tokens_details`, the Responses API's `input`/`output`
/// spelling, Anthropic's separate cache counters, DeepSeek's native
/// `prompt_cache_hit_tokens`, and the flat token object agent backends
/// report on their own SSE event.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct PassthroughUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    cached_prompt_tokens: u32,
    reasoning_tokens: u32,
    cache_creation_tokens: u32,
    cache_read_tokens: u32,
}

impl PassthroughUsage {
    /// Field-wise max, the accumulation the typed streaming paths use.
    ///
    /// One stream reports usage across several frames — Anthropic's
    /// `message_start` carries the input and cache counters while its
    /// terminal `message_delta` carries only the output ones — so a later
    /// partial report must EXTEND the record rather than replace it. Max
    /// also makes a provider that repeats a cumulative usage object
    /// harmless.
    fn merge(&mut self, other: Self) {
        self.prompt_tokens = self.prompt_tokens.max(other.prompt_tokens);
        self.completion_tokens = self.completion_tokens.max(other.completion_tokens);
        self.cached_prompt_tokens = self.cached_prompt_tokens.max(other.cached_prompt_tokens);
        self.reasoning_tokens = self.reasoning_tokens.max(other.reasoning_tokens);
        self.cache_creation_tokens = self.cache_creation_tokens.max(other.cache_creation_tokens);
        self.cache_read_tokens = self.cache_read_tokens.max(other.cache_read_tokens);
    }
}

/// `usage` figures from a buffered protocol-aware response body.
fn response_usage(protocol: PassthroughProtocol, body: &[u8]) -> Option<PassthroughUsage> {
    if matches!(protocol, PassthroughProtocol::Raw) {
        return None;
    }
    let v = serde_json::from_slice::<serde_json::Value>(body).ok()?;
    usage_of(v.get("usage")?)
}

/// Read every token dimension out of one `usage` object (or, for the
/// labelled frame of an opaque stream, a flat token object).
///
/// The spellings are read as a union rather than per protocol because a
/// passthrough route relays whichever API the caller addressed: the same
/// route carries an OpenAI chat envelope, an Anthropic one, and an agent
/// backend's private shape. They do not collide — each name belongs to
/// exactly one API — so reading them all costs nothing and a detected
/// envelope reports what its typed endpoint would.
///
/// `None` when the object carries no recognised counter at all, which is
/// what keeps a `usage`-shaped object that is not a usage report from
/// minting zeros.
fn usage_of(usage: &serde_json::Value) -> Option<PassthroughUsage> {
    let num = |v: Option<&serde_json::Value>| {
        v.and_then(serde_json::Value::as_u64)
            .map(|n| n.min(u32::MAX as u64) as u32)
    };
    // Flat counter under any of `names`, first hit wins.
    let flat = |names: &[&str]| names.iter().find_map(|n| num(usage.get(*n)));
    // `parent.child` counter, e.g. `prompt_tokens_details.cached_tokens`.
    let nested = |parent: &str, child: &str| num(usage.get(parent).and_then(|d| d.get(child)));

    let prompt = flat(&["prompt_tokens", "input_tokens"]);
    let completion = flat(&["completion_tokens", "output_tokens"]);
    // OpenAI nests the cache hit under `prompt_tokens_details`, the
    // Responses API under `input_tokens_details`, DeepSeek reports it flat
    // as `prompt_cache_hit_tokens`. A nested ZERO must not mask a real
    // native count (the typed OpenAI bridge takes the same precedence).
    let cached_prompt = nested("prompt_tokens_details", "cached_tokens")
        .filter(|&n| n > 0)
        .or_else(|| nested("input_tokens_details", "cached_tokens").filter(|&n| n > 0))
        .or_else(|| flat(&["prompt_cache_hit_tokens", "cached_tokens"]));
    let reasoning = nested("completion_tokens_details", "reasoning_tokens")
        .filter(|&n| n > 0)
        .or_else(|| nested("output_tokens_details", "reasoning_tokens").filter(|&n| n > 0))
        .or_else(|| flat(&["reasoning_tokens"]));
    // Anthropic's two cache counters sit beside `input_tokens`, and are
    // ADDITIVE to it rather than a subset.
    let cache_creation = flat(&["cache_creation_input_tokens", "cache_creation_tokens"]);
    let cache_read = flat(&["cache_read_input_tokens", "cache_read_tokens"]);

    let dims = [
        prompt,
        completion,
        cached_prompt,
        reasoning,
        cache_creation,
        cache_read,
    ];
    if dims.iter().all(Option::is_none) {
        return None;
    }
    Some(PassthroughUsage {
        prompt_tokens: prompt.unwrap_or(0),
        completion_tokens: completion.unwrap_or(0),
        cached_prompt_tokens: cached_prompt.unwrap_or(0),
        reasoning_tokens: reasoning.unwrap_or(0),
        cache_creation_tokens: cache_creation.unwrap_or(0),
        cache_read_tokens: cache_read.unwrap_or(0),
    })
}

/// Model-level rate-limit identity from the JSON body's top-level `model`
/// field, scoped to `provider_lower` — the #805 contract carried over from
/// the removed implicit tunnel: `display_name` exact hit first, then the
/// provider-native `model_name` (deterministic on ties, wildcards
/// excluded), with the reservation keyed by `display_name` so route and
/// typed traffic to the same Model draw from one bucket. `None` for
/// non-JSON bodies, absent/unregistered names, or cross-provider names —
/// the request then reserves only the caller-level layers.
fn body_model_rate_limit(
    snapshot: &aisix_core::AisixSnapshot,
    provider_lower: &str,
    body: &[u8],
) -> Option<crate::quota::ModelRateLimit> {
    #[derive(serde::Deserialize)]
    struct BodyModelProbe {
        model: Option<String>,
    }
    let name = serde_json::from_slice::<BodyModelProbe>(body).ok()?.model?;
    let matches_provider = |m: &aisix_core::Model| {
        m.provider
            .as_deref()
            .is_some_and(|p| p.eq_ignore_ascii_case(provider_lower))
    };
    let entry = snapshot
        .models
        .get_by_name(&name)
        .filter(|e| matches_provider(&e.value))
        .or_else(|| {
            snapshot
                .models
                .entries()
                .into_iter()
                .filter(|e| {
                    matches_provider(&e.value)
                        && e.value.model_name.as_deref() == Some(name.as_str())
                        && !e.value.display_name.contains('*')
                })
                .min_by_key(|e| e.id.clone())
        })?;
    Some(crate::quota::ModelRateLimit::from_model(
        &entry.value.display_name,
        &entry.id,
        &entry.value,
    ))
}

/// `true` if `seg` is a strict api-version path component matching `v\d+`.
fn is_api_version_segment(seg: &str) -> bool {
    seg.starts_with('v') && seg.len() > 1 && seg[1..].chars().all(|c| c.is_ascii_digit())
}

/// Strip one leading api-version segment from `rest` when it exactly
/// matches the trailing version segment of `base` (#164): an operator's
/// `target_url` ending in `/v1` joined with a caller path starting `v1/`
/// would otherwise produce `/v1/v1/...`.
fn strip_redundant_version_segment<'a>(base: &str, rest: &'a str) -> &'a str {
    let base_tail = base.rsplit('/').next().unwrap_or("");
    if !is_api_version_segment(base_tail) {
        return rest;
    }
    if let Some(remainder) = rest.strip_prefix(base_tail) {
        if remainder.is_empty() {
            return remainder;
        }
        if let Some(after_slash) = remainder.strip_prefix('/') {
            return after_slash;
        }
    }
    rest
}

// ---------------------------------------------------------------------------
// Streaming relay
// ---------------------------------------------------------------------------

/// Incremental splitter of an SSE byte stream into complete frames
/// (terminated by a blank line). Bytes after the last complete frame stay
/// buffered until more arrive; `take_rest` drains them at end-of-stream.
/// Cap on bytes buffered while waiting for one SSE frame terminator, and on
/// bytes held back by the `Window` policy while its char threshold has not
/// been reached. Both accumulators would otherwise grow without bound on an
/// upstream that never terminates a frame (or streams only delta-free
/// frames) — and a streaming route carries no reqwest-level timeout to end
/// the read. On overflow the oversized run is handed on as if it were a
/// complete frame (splitter) or force-scanned (window), so memory stays
/// bounded while the policy semantics degrade gracefully.
const MAX_HELD_STREAM_BYTES: usize = 1024 * 1024;

struct SseFrameSplitter {
    buf: Vec<u8>,
    /// Resume offset for the boundary scan: everything before it was
    /// already checked in an earlier `push`, so an unterminated frame
    /// costs O(n), not O(n²).
    scanned: usize,
}

impl SseFrameSplitter {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            scanned: 0,
        }
    }

    fn push(&mut self, chunk: &[u8]) -> Vec<Vec<u8>> {
        self.buf.extend_from_slice(chunk);
        let mut frames = Vec::new();
        loop {
            // Rescan the last 3 already-checked bytes: a boundary can
            // straddle the previous chunk edge.
            let from = self.scanned.saturating_sub(3);
            let lf = find_subsequence(&self.buf[from..], b"\n\n").map(|i| (from + i, 2));
            let crlf = find_subsequence(&self.buf[from..], b"\r\n\r\n").map(|i| (from + i, 4));
            let boundary = match (lf, crlf) {
                (Some((li, ll)), Some((ci, cl))) => {
                    if ci < li {
                        (ci, cl)
                    } else {
                        (li, ll)
                    }
                }
                (Some(x), None) | (None, Some(x)) => x,
                (None, None) => {
                    self.scanned = self.buf.len();
                    // Frame-terminator starvation: hand the oversized run on
                    // as-is rather than buffering without bound.
                    if self.buf.len() > MAX_HELD_STREAM_BYTES {
                        frames.push(std::mem::take(&mut self.buf));
                        self.scanned = 0;
                    }
                    break;
                }
            };
            let end = boundary.0 + boundary.1;
            let frame: Vec<u8> = self.buf.drain(..end).collect();
            self.scanned = 0;
            frames.push(frame);
        }
        frames
    }

    fn take_rest(&mut self) -> Vec<u8> {
        self.scanned = 0;
        std::mem::take(&mut self.buf)
    }
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// `true` when the frame carries an `event:` line the server itself names
/// a usage report. The only evidence an opaque stream offers that a
/// token-shaped payload IS usage — see [`frame_delta`].
fn is_usage_labelled_frame(frame_text: &str) -> bool {
    frame_text.lines().any(|line| {
        line.strip_prefix("event:").is_some_and(|name| {
            let name = name.trim();
            name.eq_ignore_ascii_case("token_usage") || name.eq_ignore_ascii_case("usage")
        })
    })
}

/// Text a guardrail scans from one SSE frame, per the protocol hint, plus
/// a usage probe on the same parsed payload.
///
/// Usage is read from, in order of specificity:
///
/// - the payload's own top-level `usage` object (every OpenAI-shape
///   stream, and Anthropic's terminal `message_delta`);
/// - `message.usage` on an Anthropic `message_start`, which is where the
///   input and cache counters arrive — its `message_delta` reports only
///   the output ones, so reading just the top level loses the prompt side
///   of every Anthropic stream;
/// - `response.usage` on a Responses stream's terminal event;
/// - for an OPAQUE (`Raw`) stream only, a FLAT token object on a frame the
///   server labelled a usage report (`event: token_usage`). An opaque
///   stream has no envelope to authenticate a payload against, so the
///   server's own label is the evidence — a payload that merely happens to
///   carry token-shaped fields must never mint billed tokens.
///
/// Frames accumulate field-wise (see [`PassthroughUsage::merge`]) at the
/// call site, so a partial report never truncates an earlier one.
fn frame_delta(protocol: PassthroughProtocol, frame: &[u8]) -> (String, Option<PassthroughUsage>) {
    let frame_text = String::from_utf8_lossy(frame);
    let usage_labelled = matches!(protocol, PassthroughProtocol::Raw)
        && is_usage_labelled_frame(frame_text.as_ref());
    let mut text = String::new();
    let mut usage: Option<PassthroughUsage> = None;
    let mut merge = |found: PassthroughUsage| {
        usage
            .get_or_insert_with(PassthroughUsage::default)
            .merge(found);
    };
    for line in frame_text.lines() {
        let Some(payload) = line.strip_prefix("data:").map(str::trim_start) else {
            continue;
        };
        if payload == "[DONE]" {
            continue;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) else {
            if matches!(protocol, PassthroughProtocol::Raw) {
                text.push_str(payload);
            }
            continue;
        };
        if let Some(u) = v.get("usage").and_then(usage_of) {
            merge(u);
        }
        // Anthropic opens its stream with the prompt + cache counters
        // nested on `message_start`. Gated on the event type so no other
        // envelope's `message` object can be read as usage.
        if v.get("type").and_then(|t| t.as_str()) == Some("message_start") {
            if let Some(u) = v
                .get("message")
                .and_then(|m| m.get("usage"))
                .and_then(usage_of)
            {
                merge(u);
            }
        }
        if matches!(protocol, PassthroughProtocol::OpenaiResponses) {
            // Responses streams carry usage on the terminal
            // `response.completed` event's embedded response object. Read
            // that shape ONLY here: another protocol's frame that happens
            // to nest `response.usage` must not be read as usage.
            if let Some(u) = v
                .get("response")
                .and_then(|r| r.get("usage"))
                .and_then(usage_of)
            {
                merge(u);
            }
        }
        if usage_labelled {
            // The agent-backend shape: a flat token object on the
            // server's own usage event, with no `usage` wrapper.
            if let Some(u) = usage_of(&v) {
                merge(u);
            }
        }
        match protocol {
            PassthroughProtocol::Raw => text.push_str(payload),
            PassthroughProtocol::OpenaiChat => {
                if let Some(choices) = v.get("choices").and_then(|c| c.as_array()) {
                    for c in choices {
                        if let Some(t) = c
                            .get("delta")
                            .and_then(|d| d.get("content"))
                            .map(content_text)
                        {
                            text.push_str(&t);
                        }
                    }
                }
            }
            PassthroughProtocol::OpenaiCompletions => {
                if let Some(choices) = v.get("choices").and_then(|c| c.as_array()) {
                    for c in choices {
                        if let Some(t) = c.get("text").and_then(|t| t.as_str()) {
                            text.push_str(t);
                        }
                    }
                }
            }
            PassthroughProtocol::OpenaiResponses => {
                // Text arrives as `response.output_text.delta` events; the
                // terminal `response.completed` repeats the whole output,
                // which would double the captured text, so take deltas only.
                let is_delta = v
                    .get("type")
                    .and_then(|t| t.as_str())
                    .is_some_and(|t| t.ends_with("output_text.delta"));
                if is_delta {
                    if let Some(t) = v.get("delta").and_then(|d| d.as_str()) {
                        text.push_str(t);
                    }
                }
            }
        }
    }
    (text, usage)
}

/// The SSE error frame appended when an output guardrail blocks mid-relay.
fn guardrail_error_frame(guardrail_name: Option<&str>) -> Bytes {
    let payload = serde_json::json!({
        "error": {
            "type": "content_filter",
            "message": crate::error::guardrail_block_message("response", guardrail_name),
        }
    });
    Bytes::from(format!("event: error\ndata: {payload}\n\n"))
}

/// Build the streamed relay response: upstream SSE frames are forwarded
/// incrementally, tee'd through the chain's [`StreamOutputPolicy`]
/// (window / full-buffer hold-back, end-of-stream check otherwise), while
/// usage and capture accumulate for the end-of-stream telemetry emit. The
/// telemetry guard also fires from `Drop` when the client disconnects
/// mid-relay.
#[allow(clippy::too_many_arguments)]
fn stream_response(
    protocol: PassthroughProtocol,
    chain: aisix_guardrails::GuardrailChain,
    upstream_resp: reqwest::Response,
    resp_headers: HeaderMap,
    status: reqwest::StatusCode,
    mut telemetry: RouteTelemetry,
    request_id: &str,
) -> Response {
    use aisix_guardrails::{Guardrail as _, GuardrailVerdict, StreamOutputPolicy};
    use futures::StreamExt;

    let policy = if chain.is_empty() {
        StreamOutputPolicy::EndOfStreamCheck
    } else {
        chain.stream_output_policy()
    };
    let route_name = telemetry.route_name.clone();
    let capture_cap = telemetry.content_cap;

    let stream = async_stream::stream! {
        let mut upstream = upstream_resp.bytes_stream();
        let mut splitter = SseFrameSplitter::new();
        // Held-back frames (Window / BufferFull) not yet released.
        let mut pending: Vec<Bytes> = Vec::new();
        let mut held_bytes: usize = 0;
        // Unscanned delta text for the CURRENT window / buffer.
        let mut scan_buf = String::new();
        // Overlap carried between Window scans.
        let mut overlap_tail = String::new();
        // Degrades BufferFull to live forwarding after a fail-open cap hit.
        let mut fail_opened = false;
        let mut blocked = false;

        'outer: loop {
            let chunk = match upstream.next().await {
                Some(Ok(c)) => c,
                Some(Err(err)) => {
                    // The response head is already on the wire, so there is
                    // no status left to carry the failure — record it on the
                    // event instead of ending as a silent success.
                    let bridge = crate::dispatch::reqwest_error_to_bridge(&err, telemetry.started);
                    telemetry.error_class =
                        crate::attempt::routing_error_class(&bridge).to_string();
                    telemetry.error_message = crate::attempt::attempt_error_message(&bridge);
                    tracing::warn!(
                        route = %route_name,
                        error = %telemetry.error_message,
                        "passthrough-route upstream stream failed mid-relay",
                    );
                    break;
                }
                None => break,
            };
            // TTFT on the first upstream chunk of any type — the same
            // convention the typed streaming endpoints stamp.
            if telemetry.upstream_ttft_ms == 0 {
                telemetry.upstream_ttft_ms = telemetry
                    .attempt_started
                    .elapsed()
                    .as_millis()
                    .min(u32::MAX as u128) as u32;
            }
            for frame in splitter.push(&chunk) {
                let (delta, usage) = frame_delta(protocol, &frame);
                if let Some(u) = usage {
                    telemetry.usage.merge(u);
                }
                if capture_cap.is_some() {
                    push_capped(&mut telemetry.response_text, &delta, capture_cap);
                }
                let frame = Bytes::from(frame);
                match &policy {
                    _ if fail_opened => {
                        telemetry.mark_first_delivery();
                        yield Ok::<_, std::convert::Infallible>(frame);
                    }
                    StreamOutputPolicy::EndOfStreamCheck => {
                        scan_buf.push_str(&delta);
                        telemetry.mark_first_delivery();
                        yield Ok(frame);
                    }
                    StreamOutputPolicy::Window { size_chars, overlap_chars } => {
                        scan_buf.push_str(&delta);
                        held_bytes += frame.len();
                        pending.push(frame);
                        // The char threshold only advances on extracted delta
                        // text, so a run of delta-free frames (role-only,
                        // keep-alives, usage-only) would hold frames without
                        // bound — force the scan once the held BYTES cross
                        // the cap, mirroring BufferFull's self-bound.
                        if scan_buf.chars().count() >= *size_chars
                            || held_bytes > MAX_HELD_STREAM_BYTES
                        {
                            let text = format!("{overlap_tail}{scan_buf}");
                            match scan_output(&chain, &route_name, &text, &mut telemetry).await {
                                GuardrailVerdict::Block { reason, guardrail_name } => {
                                    tracing::warn!(
                                        guardrail_hook = "output",
                                        route = %route_name,
                                        reason = %reason,
                                        "guardrail blocked passthrough-route stream (window)",
                                    );
                                    blocked = true;
                                    yield Ok(guardrail_error_frame(guardrail_name.as_deref()));
                                    break 'outer;
                                }
                                _ => {
                                    for f in pending.drain(..) {
                                        telemetry.mark_first_delivery();
                                        yield Ok(f);
                                    }
                                    held_bytes = 0;
                                    let combined = format!("{overlap_tail}{scan_buf}");
                                    overlap_tail = tail_chars(&combined, *overlap_chars);
                                    scan_buf.clear();
                                }
                            }
                        }
                    }
                    StreamOutputPolicy::BufferFull { max_buffer_bytes, on_exceeded_fail_open } => {
                        scan_buf.push_str(&delta);
                        held_bytes += frame.len();
                        pending.push(frame);
                        if held_bytes > *max_buffer_bytes {
                            if *on_exceeded_fail_open {
                                for f in pending.drain(..) {
                                    telemetry.mark_first_delivery();
                                    yield Ok(f);
                                }
                                held_bytes = 0;
                                fail_opened = true;
                            } else {
                                tracing::warn!(
                                    route = %route_name,
                                    "passthrough-route stream exceeded the guardrail buffer cap (fail-closed)",
                                );
                                blocked = true;
                                yield Ok(guardrail_error_frame(None));
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }

        if !blocked {
            // Trailing bytes with no frame terminator, plus the final scan
            // of whatever the policy has not cleared yet.
            let rest = splitter.take_rest();
            if !rest.is_empty() {
                let (delta, usage) = frame_delta(protocol, &rest);
                if let Some(u) = usage {
                    telemetry.usage.merge(u);
                }
                if capture_cap.is_some() {
                    push_capped(&mut telemetry.response_text, &delta, capture_cap);
                }
                scan_buf.push_str(&delta);
                let rest = Bytes::from(rest);
                if policy.holds_back() && !fail_opened {
                    pending.push(rest);
                } else {
                    telemetry.mark_first_delivery();
                    yield Ok(rest);
                }
            }
            let text = format!("{overlap_tail}{scan_buf}");
            if !chain.is_empty() && !text.is_empty() {
                if let GuardrailVerdict::Block { reason, guardrail_name } =
                    scan_output(&chain, &route_name, &text, &mut telemetry).await
                {
                    tracing::warn!(
                        guardrail_hook = "output",
                        route = %route_name,
                        reason = %reason,
                        "guardrail blocked passthrough-route stream (end)",
                    );
                    // Held frames are dropped (fail closed); content already
                    // forwarded under EndOfStreamCheck cannot be unsent —
                    // the error frame is the caller-visible signal either way.
                    pending.clear();
                    yield Ok(guardrail_error_frame(guardrail_name.as_deref()));
                    telemetry.guardrail_blocked = true;
                    telemetry.stream_reached_end = true;
                    telemetry.emit();
                    return;
                }
            }
            for f in pending.drain(..) {
                telemetry.mark_first_delivery();
                yield Ok(f);
            }
        } else {
            telemetry.guardrail_blocked = true;
        }
        // The generator ran to its own end (upstream EOF, upstream error, or
        // a guardrail block); only a client that went away first leaves this
        // unset, and the emit turns that into a 499.
        telemetry.stream_reached_end = true;
        telemetry.emit();
    };

    // Re-attach the request span (the body is polled after the request-id
    // middleware returns, so end-of-stream telemetry would otherwise log
    // without a request_id) and heartbeat silence gaps — this branch is
    // SSE-only, where a comment frame is protocol-legal and identical to
    // what the typed endpoints emit; relayed frames are untouched.
    let mut response = Response::builder()
        .status(status)
        .body(Body::from_stream(crate::sse_keepalive::with_heartbeat(
            crate::request_id::in_request_span(stream),
            crate::sse_keepalive::interval(),
        )))
        .unwrap();
    copy_safe_headers(&resp_headers, response.headers_mut());
    // The relay re-chunks the body; a stale upstream length must not ride
    // along (SSE normally has none, but a lying upstream shouldn't wedge
    // the client).
    response.headers_mut().remove(header::CONTENT_LENGTH);
    if let Ok(hv) = HeaderValue::from_str(request_id) {
        response
            .headers_mut()
            .insert(header::HeaderName::from_static("x-aisix-request-id"), hv);
    }
    response
}

/// One output scan over `text`, folding monitor hits into the telemetry.
async fn scan_output(
    chain: &aisix_guardrails::GuardrailChain,
    route_name: &str,
    text: &str,
    telemetry: &mut RouteTelemetry,
) -> aisix_guardrails::GuardrailVerdict {
    use aisix_guardrails::Guardrail as _;
    let synth = aisix_gateway::ChatResponse {
        id: String::new(),
        model: route_name.to_string(),
        message: aisix_gateway::ChatMessage::assistant(text.to_string()),
        finish_reason: aisix_gateway::FinishReason::Stop,
        usage: aisix_gateway::UsageStats::default(),
    };
    let (verdict, hits) = chain.check_output_observed(&synth).await;
    telemetry.monitor_hits.extend(hits);
    verdict
}

/// The last `n` chars of `s` (whole string when shorter).
fn tail_chars(s: &str, n: usize) -> String {
    let count = s.chars().count();
    if count <= n {
        return s.to_string();
    }
    s.chars().skip(count - n).collect()
}

/// Append `delta` to `buf`, bounded by `cap` bytes (capture accumulation
/// must not grow with an unbounded stream). Char boundaries are respected.
fn push_capped(buf: &mut String, delta: &str, cap: Option<usize>) {
    let Some(cap) = cap else { return };
    if buf.len() >= cap {
        return;
    }
    if buf.len() + delta.len() <= cap {
        buf.push_str(delta);
        return;
    }
    for c in delta.chars() {
        if buf.len() + c.len_utf8() > cap {
            break;
        }
        buf.push(c);
    }
}

// ---------------------------------------------------------------------------
// Telemetry
// ---------------------------------------------------------------------------

/// End-of-request telemetry for a passthrough-route exchange: one
/// UsageEvent (CP sink + exporter fan-out, with captured content on the
/// exporter path only), the request metric, and the access log line. The
/// buffered path calls [`RouteTelemetry::emit`] inline; the streaming path
/// calls it at end-of-stream, with `Drop` covering client disconnects.
struct RouteTelemetry {
    state: ProxyState,
    route_name: String,
    provider_label: String,
    /// The request's trace bundle (AISIX-Cloud#1279) — the Drop emit is
    /// the request's terminal emission, so it carries the terminal spans.
    trace: Option<Arc<aisix_obs::RequestTraceBundle>>,
    pk_id: String,
    method: Method,
    path: String,
    request_id: String,
    api_key_id: String,
    jwt: Option<Arc<crate::auth::JwtIdentity>>,
    /// Whether the caller reached this route through `auth_mode:
    /// anonymous` rather than a credential of its own. Stamped onto the
    /// usage event so anonymous traffic stays distinguishable from the
    /// bound key's own (see `usage_attr::apply_auth_type`).
    anonymous: bool,
    client_identity: String,
    client_source_ip: String,
    client_user_agent: String,
    started: Instant,
    /// When the upstream call itself began — the scope the two `upstream_*`
    /// figures are measured in, distinct from `started` (request receipt).
    attempt_started: Instant,
    status: u16,
    /// Every token dimension the exchange reported, accumulated field-wise
    /// across the response (buffered) or its frames (streamed).
    usage: PassthroughUsage,
    /// The model alias the caller addressed, read from a DETECTED
    /// envelope's own `model` field — the same value the typed endpoint
    /// serving that envelope records. Empty for an opaque body, whose
    /// `model`-shaped key means nothing the gateway can trust.
    requested_model: String,
    /// Time from the START OF THE ATTEMPT to the upstream's first streamed
    /// frame. Zero on the buffered path, where there is none.
    upstream_ttft_ms: u32,
    /// What the caller waited for on a streamed relay: the moment the first
    /// relayed frame was handed downstream, measured from `started`. `None`
    /// until one is, so a stream that delivered nothing reports no
    /// caller-wait at all rather than an invented one.
    downstream_first_ms: Option<u32>,
    /// `true` once the relay generator reached its own end — upstream EOF,
    /// upstream error, or a guardrail block. It stays `false` only when the
    /// CLIENT went away first, which is what the emit turns into a 499
    /// (same signal the typed streaming endpoints record).
    stream_reached_end: bool,
    /// Set on a streamed relay so the `Drop` emit can tell an abandoned
    /// stream from the buffered path, which never streams at all.
    streaming: bool,
    /// Bounded error class + message for a failure the relay could not
    /// answer with a status code — an upstream that dies mid-stream, after
    /// the response head is already on the wire.
    error_class: String,
    error_message: String,
    monitor_hits: Vec<aisix_core::GuardrailMonitorHit>,
    captured_prompt: Option<String>,
    content_cap: Option<usize>,
    response_text: String,
    guardrail_blocked: bool,
    emitted: bool,
}

impl RouteTelemetry {
    /// Stamp the caller's wait at the first RELAYED frame handed
    /// downstream.
    ///
    /// Deliberately here and not where the frame was read off the upstream:
    /// a hold-back guardrail policy sits between the two, and
    /// `UsageEvent::downstream_latency_ms` counts that hold-back as part of
    /// what the caller waited for. Called on both the live-forward and the
    /// hold-back release paths, so it catches the first frame either way.
    ///
    /// A synthetic frame (a guardrail block's error event) deliberately
    /// does NOT stamp: nothing the caller asked for was delivered. Same
    /// rule as the typed streaming endpoints, which stamp only in the
    /// chunk renderer.
    fn mark_first_delivery(&mut self) {
        if self.downstream_first_ms.is_none() {
            self.downstream_first_ms =
                Some(self.started.elapsed().as_millis().min(u32::MAX as u128) as u32);
        }
    }

    fn emit(&mut self) {
        if self.emitted {
            return;
        }
        self.emitted = true;
        // A streamed relay the CLIENT abandoned never reached the
        // generator's end. The upstream status is then not what happened
        // to the request, so record the same 499 the typed streaming
        // endpoints do rather than a success the caller never received.
        if self.streaming && !self.stream_reached_end {
            self.status = crate::CLIENT_CLOSED_REQUEST;
        }
        let elapsed = self.started.elapsed();
        let snapshot = self.state.snapshot.load();
        let usage = self.usage;

        emit_access_log(
            &self.method,
            &self.path,
            &self.route_name,
            &self.api_key_id,
            self.status,
            elapsed,
            &self.request_id,
            Some(AccessLogTokens {
                prompt: usage.prompt_tokens,
                completion: usage.completion_tokens,
            }),
            None,
        );

        let pk = crate::usage_attr::ResolvedPk::resolve(&snapshot, &self.pk_id);
        let caller = crate::request_metrics::Caller::from_api_key_id(&snapshot, &self.api_key_id);
        crate::request_metrics::record(
            &self.state,
            ENDPOINT_LABEL,
            caller.as_caller(),
            crate::request_metrics::Upstream {
                provider: &self.provider_label,
                model: PASSTHROUGH_MODEL_LABEL,
                pk: pk.labels(),
                ..Default::default()
            },
            self.status,
            elapsed,
        );

        let mut event = aisix_obs::UsageEvent {
            request_id: self.request_id.clone(),
            occurred_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
            api_key_id: self.api_key_id.clone(),
            status_code: self.status,
            requested_model: self.requested_model.clone(),
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            cached_prompt_tokens: usage.cached_prompt_tokens,
            reasoning_tokens: usage.reasoning_tokens,
            cache_creation_tokens: usage.cache_creation_tokens,
            cache_read_tokens: usage.cache_read_tokens,
            upstream_latency_ms: self
                .attempt_started
                .elapsed()
                .as_millis()
                .min(u32::MAX as u128) as u32,
            upstream_ttft_ms: self.upstream_ttft_ms,
            // Streaming reports the caller's wait to the FIRST relayed
            // frame, per the field's contract — a relay is a delivery
            // mechanism for a response, not the response itself, so it is
            // not the `/a2a` exception. Absent when a stream delivered
            // nothing. The buffered path has no first frame: there the
            // caller waited for the whole response to be written.
            downstream_latency_ms: if self.streaming {
                self.downstream_first_ms.unwrap_or(0)
            } else {
                elapsed.as_millis().min(u32::MAX as u128) as u32
            },
            error_class: std::mem::take(&mut self.error_class),
            error_message: std::mem::take(&mut self.error_message),
            inbound_protocol: "passthrough".to_string(),
            passthrough_route_name: self.route_name.clone(),
            client_identity: self.client_identity.clone(),
            client_source_ip: self.client_source_ip.clone(),
            client_user_agent: self.client_user_agent.clone(),
            guardrail_blocked: self.guardrail_blocked,
            guardrail_monitor_hits: std::mem::take(&mut self.monitor_hits),
            ..Default::default()
        };
        crate::usage_attr::apply_pk_telemetry(&mut event, &pk);
        crate::usage_attr::apply_jwt_identity(&mut event, self.jwt.as_ref());
        if self.anonymous {
            event.auth_type = "anonymous".to_string();
        }
        let usage_model = crate::usage_attr::usage_event_model_label(
            // The snapshot loaded above: a config swap between two loads
            // would make this label disagree with the emit's attribution.
            &snapshot,
            &event.requested_model,
        )
        .into_owned();

        // Captured content rides ONLY on the exporter fan-out, per the
        // content_mode invariant (never the CP telemetry path).
        let content = match (&self.captured_prompt, self.content_cap) {
            (Some(prompt), Some(cap)) => Some(aisix_obs::CapturedContent::new(
                prompt,
                &self.response_text,
                cap,
            )),
            _ => None,
        };
        crate::usage_attr::emit_usage(
            &self.state,
            &snapshot,
            "passthrough_route",
            event,
            crate::usage_attr::usage_event_labels(&usage_model, &pk),
            content.as_ref(),
            self.trace.as_ref(),
            // The Drop emit is the request's end — body EOF or client drop.
            /* terminal */
            true,
            /* dispatched */ true,
        );
    }
}

impl Drop for RouteTelemetry {
    fn drop(&mut self) {
        if std::thread::panicking() {
            return;
        }
        self.emit();
    }
}

/// Copy response headers that are safe to relay to the downstream caller.
/// `append`, not `insert`: `HeaderMap` iteration yields one entry per
/// value, and a header the upstream sent several times (`Set-Cookie`,
/// `WWW-Authenticate`, `Vary`) must keep every value on a relay.
fn copy_safe_headers(src: &HeaderMap, dst: &mut HeaderMap) {
    for (name, value) in src {
        let n = name.as_str().to_lowercase();
        if matches!(
            n.as_str(),
            "transfer-encoding"
                | "connection"
                | "keep-alive"
                | "proxy-authenticate"
                | "proxy-authorization"
                | "te"
                | "trailers"
                | "upgrade"
        ) {
            continue;
        }
        dst.append(name.clone(), value.clone());
    }
}

/// Token counts for one access-log line. `None` on the paths that never
/// reached an upstream, which is what keeps a rejected request out of the
/// token columns instead of logging it as a zero-token success.
struct AccessLogTokens {
    prompt: u32,
    completion: u32,
}

#[allow(clippy::too_many_arguments)]
fn emit_access_log(
    method: &Method,
    path: &str,
    route: &str,
    api_key_id: &str,
    status: u16,
    elapsed: Duration,
    request_id: &str,
    tokens: Option<AccessLogTokens>,
    error: Option<&ProxyError>,
) {
    let (error_kind, error) = match error {
        Some(e) => {
            let (kind, msg) = crate::attempt::access_log_error(e);
            (Some(kind), Some(msg))
        }
        None => (None, None),
    };
    AccessLog {
        method: method.as_str(),
        path,
        status,
        latency: elapsed,
        provider: Some(route),
        model: None,
        api_key_id: Some(api_key_id),
        prompt_tokens: tokens.as_ref().map(|t| u64::from(t.prompt)),
        completion_tokens: tokens.as_ref().map(|t| u64::from(t.completion)),
        total_tokens: tokens
            .as_ref()
            .map(|t| u64::from(t.prompt) + u64::from(t.completion)),
        request_id,
        provider_request_id: None,
        served_by_model: None,
        routing_attempt_count: None,
        routing_fallback_count: None,
        error_kind,
        error: error.as_deref(),
    }
    .emit();
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_core::resource::ResourceEntry;
    use aisix_core::snapshot::SnapshotHandle;
    use aisix_core::{AisixSnapshot, ApiKey, ProviderKey, ProxyConfig};
    use aisix_gateway::Hub;
    use axum::body::to_bytes;
    use axum::http::{Request, StatusCode};
    use std::sync::Arc;
    use tower::ServiceExt;
    use wiremock::matchers::{method as wm_method, path as wm_path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn cfg() -> ProxyConfig {
        ProxyConfig {
            addr: "127.0.0.1:0".into(),
            request_body_limit_bytes: 1_048_576,
            real_ip: Default::default(),
            request_id: Default::default(),
            url_rewrites: Vec::new(),
            tls: None,
            thread_per_core: None,
            workers: None,
        }
    }

    const PK_ID: &str = "11111111-1111-1111-1111-111111111111";

    /// A usage record carrying only the two canonical counters.
    fn usage_dims(prompt: u32, completion: u32) -> PassthroughUsage {
        PassthroughUsage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            ..Default::default()
        }
    }

    fn provider_key_entry(api_base_unused: &str) -> ResourceEntry<ProviderKey> {
        let json = format!(
            r#"{{"display_name":"openai-up","secret":"sk-upstream","api_base":"{api_base_unused}","provider":"openai","adapter":"openai"}}"#
        );
        let pk: ProviderKey = serde_json::from_str(&json).unwrap();
        ResourceEntry::new(PK_ID, pk, 1)
    }

    fn apikey_entry(plaintext: &str, allowed_routes: Option<&[&str]>) -> ResourceEntry<ApiKey> {
        let routes = match allowed_routes {
            Some(r) => format!(
                r#", "allowed_routes": {}"#,
                serde_json::to_string(r).unwrap()
            ),
            None => String::new(),
        };
        let json = format!(
            r#"{{"key_hash":"{}","allowed_models":["*"]{routes}}}"#,
            ApiKey::hash_bearer(plaintext)
        );
        let k: ApiKey = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("k-1", k, 1)
    }

    fn route_entry(id: &str, json: serde_json::Value) -> ResourceEntry<PassthroughRoute> {
        let r: PassthroughRoute = serde_json::from_value(json).unwrap();
        ResourceEntry::new(id, r, 1)
    }

    fn build_app(snap: AisixSnapshot) -> axum::Router {
        let hub = Arc::new(Hub::new());
        let handle = SnapshotHandle::new(snap);
        crate::build_router(crate::ProxyState::new(handle, hub, &cfg()).without_cache())
    }

    #[tokio::test]
    async fn legacy_tunnel_answers_410_with_migration_pointer() {
        let app = build_app(AisixSnapshot::new());
        let req = Request::builder()
            .method("POST")
            .uri("/passthrough/openai/v1/chat/completions")
            .header("authorization", "Bearer whatever")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::GONE);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["code"], "endpoint_removed");
        assert!(v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("passthrough_route"));
    }

    #[tokio::test]
    async fn unmatched_paths_keep_the_plain_404() {
        let app = build_app(AisixSnapshot::new());
        let req = Request::builder()
            .method("GET")
            .uri("/definitely/not/a/route")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    fn inject_route(target: &str) -> ResourceEntry<PassthroughRoute> {
        route_entry(
            "route-1",
            serde_json::json!({
                "name": "openai-tunnel",
                "path_prefix": "/passthrough/openai",
                "target_url": target,
                "provider_key_id": PK_ID
            }),
        )
    }

    #[tokio::test]
    async fn inject_route_replaces_caller_auth_with_provider_key() {
        let upstream = MockServer::start().await;
        Mock::given(wm_method("GET"))
            .and(wm_path("/v1/models"))
            .and(wiremock::matchers::header(
                "authorization",
                "Bearer sk-upstream",
            ))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"object": "list", "data": []})),
            )
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(provider_key_entry("http://unused"));
        snap.apikeys.insert(apikey_entry("sk-caller", Some(&["*"])));
        snap.passthrough_routes
            .insert(inject_route(&upstream.uri()));
        let app = build_app(snap);

        let req = Request::builder()
            .method("GET")
            .uri("/passthrough/openai/v1/models")
            .header("authorization", "Bearer sk-caller")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // The caller's own Authorization must not have reached upstream.
        let received = &upstream.received_requests().await.unwrap()[0];
        let auth_values: Vec<_> = received.headers.get_all("authorization").iter().collect();
        assert_eq!(auth_values.len(), 1);
    }

    #[tokio::test]
    async fn key_without_route_grant_is_403() {
        let upstream = MockServer::start().await;
        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(provider_key_entry("http://unused"));
        snap.apikeys.insert(apikey_entry("sk-caller", None));
        snap.passthrough_routes
            .insert(inject_route(&upstream.uri()));
        let app = build_app(snap);

        let req = Request::builder()
            .method("GET")
            .uri("/passthrough/openai/v1/models")
            .header("authorization", "Bearer sk-caller")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "permission_denied");
    }

    #[tokio::test]
    async fn unauthenticated_route_request_is_401() {
        let upstream = MockServer::start().await;
        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(provider_key_entry("http://unused"));
        snap.passthrough_routes
            .insert(inject_route(&upstream.uri()));
        let app = build_app(snap);

        let req = Request::builder()
            .method("GET")
            .uri("/passthrough/openai/v1/models")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    /// The forward-proxy shadowing case: a host-matched request whose path
    /// collides with a typed gateway route must be served by the
    /// passthrough route, not the typed handler.
    #[tokio::test]
    async fn host_match_wins_over_typed_route_on_colliding_path() {
        let upstream = MockServer::start().await;
        Mock::given(wm_method("POST"))
            .and(wm_path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"routed": "byo"})),
            )
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.apikeys.insert(apikey_entry("sk-caller", Some(&["*"])));
        // forward_client + header_key: Authorization belongs to the caller
        // and must reach upstream verbatim.
        snap.passthrough_routes.insert(route_entry(
            "route-h",
            serde_json::json!({
                "name": "byo-host",
                "hosts": ["ai.example.com"],
                "target_url": upstream.uri(),
                "auth_mode": "header_key",
                "auth_header_name": "x-aisix-api-key",
                "credential_mode": "forward_client"
            }),
        ));
        let app = build_app(snap);

        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("host", "ai.example.com")
            .header("authorization", "Bearer employee-official-token")
            .header("x-aisix-api-key", "sk-caller")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(r#"{"model":"gpt-4o"}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["routed"], "byo", "typed chat handler must not serve this");

        // BYO: the employee credential reached upstream verbatim; the
        // gateway's side-channel header did not.
        let received = &upstream.received_requests().await.unwrap()[0];
        assert_eq!(
            received.headers.get("authorization").unwrap(),
            "Bearer employee-official-token"
        );
        assert!(received.headers.get("x-aisix-api-key").is_none());
    }

    #[tokio::test]
    async fn disabled_route_does_not_match() {
        let upstream = MockServer::start().await;
        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(provider_key_entry("http://unused"));
        snap.apikeys.insert(apikey_entry("sk-caller", Some(&["*"])));
        let mut json = serde_json::json!({
            "name": "openai-tunnel",
            "path_prefix": "/passthrough/openai",
            "target_url": upstream.uri(),
            "provider_key_id": PK_ID,
            "enabled": false
        });
        json["enabled"] = serde_json::Value::Bool(false);
        snap.passthrough_routes.insert(route_entry("route-1", json));
        let app = build_app(snap);

        let req = Request::builder()
            .method("GET")
            .uri("/passthrough/openai/v1/models")
            .header("authorization", "Bearer sk-caller")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        // Disabled → no match → the tunnel namespace answers the 410.
        assert_eq!(resp.status(), StatusCode::GONE);
    }

    #[tokio::test]
    async fn anonymous_route_fails_closed_when_source_ip_is_unresolvable() {
        let upstream = MockServer::start().await;
        let snap = AisixSnapshot::new();
        snap.apikeys.insert(apikey_entry("sk-anon", Some(&["*"])));
        snap.passthrough_routes.insert(route_entry(
            "route-a",
            serde_json::json!({
                "name": "anon",
                "path_prefix": "/anon",
                "target_url": upstream.uri(),
                "auth_mode": "anonymous",
                "anonymous_key_id": "k-1",
                "source_cidrs": ["0.0.0.0/0"],
                "credential_mode": "forward_client"
            }),
        ));
        let app = build_app(snap);

        // In-process requests resolve no client socket; an unparseable
        // source must never satisfy the CIDR gate.
        let req = Request::builder()
            .method("GET")
            .uri("/anon/x")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    // ---- pure helpers ----

    #[test]
    fn path_prefix_matches_on_segment_boundary_only() {
        assert!(path_under_prefix("/copilot", "/copilot"));
        assert!(path_under_prefix("/copilot/chat", "/copilot"));
        assert!(!path_under_prefix("/copilotx", "/copilot"));
    }

    #[test]
    fn inbound_host_strips_port_and_lowercases() {
        let req = Request::builder()
            .uri("/x")
            .header("host", "API.Example.COM:8443")
            .body(axum::body::Body::empty())
            .unwrap();
        assert_eq!(inbound_host(&req).as_deref(), Some("api.example.com"));
    }

    #[test]
    fn longest_prefix_and_host_specificity_win() {
        let snap = AisixSnapshot::new();
        let mk = |id: &str, json: serde_json::Value| {
            snap.passthrough_routes.insert(route_entry(id, json))
        };
        mk(
            "r-short",
            serde_json::json!({"name":"short","path_prefix":"/p","target_url":"http://a","provider_key_id":"pk"}),
        );
        mk(
            "r-long",
            serde_json::json!({"name":"long","path_prefix":"/p/deep","target_url":"http://b","provider_key_id":"pk"}),
        );
        mk(
            "r-host",
            serde_json::json!({"name":"hosty","hosts":["h.example"],"target_url":"http://c","provider_key_id":"pk"}),
        );

        let m = match_route(&snap, None, "/p/deep/x").unwrap();
        assert_eq!(m.entry.value.name, "long");
        assert_eq!(m.remainder, "/x");
        assert!(m.prefix_matched);

        // Host match beats any path-only match.
        let m = match_route(&snap, Some("h.example"), "/p/deep/x").unwrap();
        assert_eq!(m.entry.value.name, "hosty");
        assert_eq!(m.remainder, "/p/deep/x");
        assert!(!m.prefix_matched);

        // A preserve_host route narrowed by a prefix relays the WHOLE path:
        // the prefix is a match condition on an upstream that owns its own
        // path space, not a gateway mount point. GitHub Copilot's CLI needs
        // this — its MCP server answers on /mcp/readonly of the same host it
        // serves chat from, and a stripped "/readonly" 404s.
        mk(
            "r-mirror",
            serde_json::json!({
                "name":"mirror","hosts":["m.example"],"path_prefix":"/mcp",
                "preserve_host":true,"credential_mode":"forward_client"
            }),
        );
        let m = match_route(&snap, Some("m.example"), "/mcp/readonly").unwrap();
        assert_eq!(m.entry.value.name, "mirror");
        assert_eq!(m.remainder, "/mcp/readonly");
        assert!(
            !m.prefix_matched,
            "a mirrored path is never version-deduped"
        );
    }

    #[test]
    fn sse_splitter_emits_complete_frames_and_keeps_partials() {
        let mut s = SseFrameSplitter::new();
        let frames = s.push(b"data: a\n\ndata: b\n\ndata: par");
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0], b"data: a\n\n");
        let frames = s.push(b"tial\n\n");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0], b"data: partial\n\n");
        assert!(s.take_rest().is_empty());
        // CRLF boundaries too.
        let mut s = SseFrameSplitter::new();
        let frames = s.push(b"data: x\r\n\r\nrest");
        assert_eq!(frames.len(), 1);
        assert_eq!(s.take_rest(), b"rest");
    }

    #[test]
    fn frame_delta_extracts_chat_content_and_usage() {
        let frame = br#"data: {"choices":[{"delta":{"content":"hel"}}]}

"#;
        let (text, usage) = frame_delta(PassthroughProtocol::OpenaiChat, frame);
        assert_eq!(text, "hel");
        assert!(usage.is_none());

        let done = br#"data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3}}

"#;
        let (text, usage) = frame_delta(PassthroughProtocol::OpenaiChat, done);
        assert_eq!(text, "");
        assert_eq!(usage, Some(usage_dims(7, 3)));

        let fim = br#"data: {"choices":[{"text":"def "}]}

"#;
        let (text, _) = frame_delta(PassthroughProtocol::OpenaiCompletions, fim);
        assert_eq!(text, "def ");
    }

    #[test]
    fn usage_of_reads_every_dimension_in_every_spelling() {
        // OpenAI chat: the cache hit is nested under `prompt_tokens_details`
        // and the reasoning count under `completion_tokens_details`.
        let openai = serde_json::json!({
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 80},
            "completion_tokens_details": {"reasoning_tokens": 12},
        });
        assert_eq!(
            usage_of(&openai),
            Some(PassthroughUsage {
                prompt_tokens: 100,
                completion_tokens: 20,
                cached_prompt_tokens: 80,
                reasoning_tokens: 12,
                ..Default::default()
            })
        );

        // Responses API: the `input`/`output` spelling, details nested under
        // the matching names.
        let responses = serde_json::json!({
            "input_tokens": 30,
            "output_tokens": 9,
            "input_tokens_details": {"cached_tokens": 25},
            "output_tokens_details": {"reasoning_tokens": 4},
        });
        assert_eq!(
            usage_of(&responses),
            Some(PassthroughUsage {
                prompt_tokens: 30,
                completion_tokens: 9,
                cached_prompt_tokens: 25,
                reasoning_tokens: 4,
                ..Default::default()
            })
        );

        // Anthropic: cache counters are separate, additive fields.
        let anthropic = serde_json::json!({
            "input_tokens": 11,
            "output_tokens": 5,
            "cache_creation_input_tokens": 300,
            "cache_read_input_tokens": 1200,
        });
        assert_eq!(
            usage_of(&anthropic),
            Some(PassthroughUsage {
                prompt_tokens: 11,
                completion_tokens: 5,
                cache_creation_tokens: 300,
                cache_read_tokens: 1200,
                ..Default::default()
            })
        );

        // DeepSeek reports the cache hit flat, and a ZEROED nested detail
        // must not mask it (same precedence the typed OpenAI bridge uses).
        let deepseek = serde_json::json!({
            "prompt_tokens": 40,
            "completion_tokens": 6,
            "prompt_tokens_details": {"cached_tokens": 0},
            "prompt_cache_hit_tokens": 32,
        });
        assert_eq!(usage_of(&deepseek).unwrap().cached_prompt_tokens, 32);

        // The flat agent-backend shape, all five dimensions at the root.
        let flat = serde_json::json!({
            "prompt_tokens": 14603,
            "completion_tokens": 8,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 14272,
            "reasoning_tokens": 8,
        });
        assert_eq!(
            usage_of(&flat),
            Some(PassthroughUsage {
                prompt_tokens: 14603,
                completion_tokens: 8,
                cache_read_tokens: 14272,
                reasoning_tokens: 8,
                ..Default::default()
            })
        );

        // An object with no recognised counter mints nothing.
        assert_eq!(usage_of(&serde_json::json!({"disk": "80%"})), None);
        assert_eq!(usage_of(&serde_json::Value::Null), None);
    }

    #[test]
    fn anthropic_stream_reports_the_prompt_side_from_message_start() {
        // Anthropic splits usage across two frames: `message_start` carries
        // the input + cache counters, the terminal `message_delta` only the
        // output ones. Reading the top level alone loses the prompt side.
        let start = br#"data: {"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":12,"cache_creation_input_tokens":300,"cache_read_input_tokens":1200}}}

"#;
        let (_, start_usage) = frame_delta(PassthroughProtocol::OpenaiChat, start);
        let start_usage = start_usage.expect("message_start must report usage");
        assert_eq!(start_usage.prompt_tokens, 12);
        assert_eq!(start_usage.cache_creation_tokens, 300);
        assert_eq!(start_usage.cache_read_tokens, 1200);

        let delta = br#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}

"#;
        let (_, delta_usage) = frame_delta(PassthroughProtocol::OpenaiChat, delta);
        let mut acc = start_usage;
        acc.merge(delta_usage.expect("message_delta must report usage"));
        // The later, partial report EXTENDS the record instead of
        // truncating the prompt side to zero.
        assert_eq!(
            acc,
            PassthroughUsage {
                prompt_tokens: 12,
                completion_tokens: 7,
                cache_creation_tokens: 300,
                cache_read_tokens: 1200,
                ..Default::default()
            }
        );

        // The nested read is gated on the event type: a `message` object on
        // any other frame is not a usage report.
        let other = br#"data: {"type":"conversation","message":{"usage":{"input_tokens":999}}}

"#;
        assert_eq!(frame_delta(PassthroughProtocol::OpenaiChat, other).1, None);
    }

    #[test]
    fn opaque_stream_reads_flat_usage_only_from_a_labelled_frame() {
        // An agent backend reached through a forward-proxy route has no
        // recognisable envelope, and reports usage on its own event as a
        // flat token object with no `usage` wrapper.
        let labelled = b"event:token_usage\ndata:{\"name\":\"\",\"prompt_tokens\":14603,\"completion_tokens\":8,\"cache_read_input_tokens\":14272,\"reasoning_tokens\":8}\n\n";
        let usage = frame_delta(PassthroughProtocol::Raw, labelled)
            .1
            .expect("a server-labelled usage frame must report usage");
        assert_eq!(usage.prompt_tokens, 14603);
        assert_eq!(usage.completion_tokens, 8);
        assert_eq!(usage.cache_read_tokens, 14272);
        assert_eq!(usage.reasoning_tokens, 8);

        // The same flat shape on a frame the server did NOT name a usage
        // report mints nothing: an opaque stream has no envelope to
        // authenticate token-shaped fields against.
        let unlabelled = b"event:history\ndata:{\"prompt_tokens\":99,\"completion_tokens\":9}\n\n";
        assert_eq!(frame_delta(PassthroughProtocol::Raw, unlabelled).1, None);

        // The flat allowance is Raw-only — a detected envelope keeps
        // reading usage from its own shape.
        assert_eq!(
            frame_delta(PassthroughProtocol::OpenaiChat, labelled).1,
            None
        );

        // An explicit `usage` OBJECT is still read from any opaque frame:
        // it is self-describing, and this is the pre-existing behaviour.
        let wrapped =
            b"event:done\ndata:{\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2}}\n\n";
        assert_eq!(
            frame_delta(PassthroughProtocol::Raw, wrapped).1,
            Some(usage_dims(4, 2))
        );
    }

    #[test]
    fn buffered_opaque_responses_are_never_probed_for_usage() {
        // The no-phantom-tokens guarantee: a REST body that happens to carry
        // a usage-shaped object is not a usage report.
        let rpc = br#"{"jsonrpc":"2.0","id":1,"result":{"usage":{"prompt_tokens":99}}}"#;
        assert_eq!(response_usage(PassthroughProtocol::Raw, rpc), None);
        let top_level = br#"{"usage":{"prompt_tokens":99,"completion_tokens":9}}"#;
        assert_eq!(response_usage(PassthroughProtocol::Raw, top_level), None);
    }

    #[test]
    fn usage_merge_is_field_wise_max() {
        let mut acc = PassthroughUsage {
            prompt_tokens: 10,
            completion_tokens: 4,
            cache_read_tokens: 100,
            ..Default::default()
        };
        // A repeat of a cumulative usage object, and a partial one, are both
        // harmless: no dimension ever regresses.
        acc.merge(PassthroughUsage {
            prompt_tokens: 10,
            completion_tokens: 9,
            reasoning_tokens: 3,
            ..Default::default()
        });
        assert_eq!(
            acc,
            PassthroughUsage {
                prompt_tokens: 10,
                completion_tokens: 9,
                reasoning_tokens: 3,
                cache_read_tokens: 100,
                ..Default::default()
            }
        );
    }

    #[test]
    fn guardrail_text_covers_tool_calls_on_both_hooks() {
        // A deny-listed string hidden in a tool call's arguments must be
        // scanned — a benign `content` beside it would otherwise make the
        // extraction non-empty and skip the raw-body fallback, letting the
        // request pass a check the typed endpoint enforces.
        let req = br#"{"model":"m","messages":[{"role":"assistant","content":"ok","tool_calls":[{"function":{"name":"run","arguments":"{\"cmd\":\"SECRET\"}"}}]}]}"#;
        let text = request_guardrail_text(PassthroughProtocol::OpenaiChat, req);
        assert!(text.contains("ok"), "content still scanned: {text}");
        assert!(
            text.contains("SECRET"),
            "tool-call arguments scanned: {text}"
        );

        let resp = br#"{"choices":[{"message":{"content":"sure","tool_calls":[{"function":{"name":"run","arguments":"{\"cmd\":\"SECRET\"}"}}]}}]}"#;
        let text = response_guardrail_text(PassthroughProtocol::OpenaiChat, resp);
        assert!(text.contains("sure"));
        assert!(text.contains("SECRET"), "tool-call output scanned: {text}");
    }

    #[test]
    fn requested_model_comes_only_from_a_detected_envelope() {
        let chat = br#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#;
        assert_eq!(
            body_model_name(PassthroughProtocol::OpenaiChat, chat),
            "gpt-4o"
        );
        // An opaque body's `model`-shaped key belongs to some other API.
        let opaque = br#"{"model":"whatever","config_name":"x"}"#;
        assert_eq!(body_model_name(PassthroughProtocol::Raw, opaque), "");
        // Caller-supplied, so bounded and control-char free before it
        // reaches telemetry.
        let hostile = format!(
            r#"{{"input":"x","model":"a\u0000b{}"}}"#,
            "z".repeat(REQUESTED_MODEL_CAP * 2)
        );
        let name = body_model_name(PassthroughProtocol::OpenaiResponses, hostile.as_bytes());
        assert_eq!(name.chars().count(), REQUESTED_MODEL_CAP);
        assert!(!name.contains('\0'));
    }

    #[test]
    fn request_text_extraction_per_protocol() {
        let chat = br#"{"model":"m","messages":[{"role":"system","content":"s"},{"role":"user","content":[{"type":"text","text":"part"}]}]}"#;
        assert_eq!(
            request_guardrail_text(PassthroughProtocol::OpenaiChat, chat),
            "s\npart"
        );
        let fim = br#"{"prompt":"def f(","suffix":"return"}"#;
        assert_eq!(
            request_guardrail_text(PassthroughProtocol::OpenaiCompletions, fim),
            "def f(\nreturn"
        );
        // Shape mismatch degrades to the raw body.
        let not_chat = br#"{"input":"x"}"#;
        assert_eq!(
            request_guardrail_text(PassthroughProtocol::OpenaiChat, not_chat),
            r#"{"input":"x"}"#
        );
        // A detected envelope whose items carry no text ALSO degrades to
        // the raw body — detection must never scan less than raw would.
        let empty_chat = br#"{"messages":[{"role":"tool","tool_call_id":"1"}]}"#;
        assert_eq!(
            request_guardrail_text(PassthroughProtocol::OpenaiChat, empty_chat),
            String::from_utf8_lossy(empty_chat)
        );
    }

    #[test]
    fn detect_protocol_from_request_envelope() {
        // The real Copilot CLI surface, one shape per endpoint family.
        let cases: [(&[u8], PassthroughProtocol); 8] = [
            // Chat: `messages` array.
            (
                br#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#,
                PassthroughProtocol::OpenaiChat,
            ),
            // Responses API: `input` as items or a bare string.
            (
                br#"{"model":"m","input":[{"role":"user","content":"hi"}]}"#,
                PassthroughProtocol::OpenaiResponses,
            ),
            (
                br#"{"model":"m","input":"hi"}"#,
                PassthroughProtocol::OpenaiResponses,
            ),
            // FIM / legacy completions: `prompt`.
            (
                br#"{"prompt":"def f(","suffix":"return"}"#,
                PassthroughProtocol::OpenaiCompletions,
            ),
            // MCP JSON-RPC relays as raw.
            (
                br#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#,
                PassthroughProtocol::Raw,
            ),
            // Plain REST / unrecognized JSON relays as raw.
            (br#"{"ref":"main","inputs":{}}"#, PassthroughProtocol::Raw),
            // Carrier keys of the wrong TYPE stay raw: only the API's own
            // shape (array/string) counts as that envelope.
            (br#"{"messages":"not-an-array"}"#, PassthroughProtocol::Raw),
            // Non-JSON / empty (GET) bodies are raw.
            (b"", PassthroughProtocol::Raw),
        ];
        for (body, want) in cases {
            assert_eq!(
                detect_protocol(body),
                want,
                "body {:?}",
                String::from_utf8_lossy(body)
            );
        }
    }

    #[test]
    fn responses_protocol_extracts_prompt_completion_and_usage() {
        // GitHub's Copilot CLI sends every inference turn to POST
        // /responses, so a forward-proxy route left on `openai_chat`
        // recorded that traffic with zero tokens and no captured text.
        let req = br#"{"model":"gpt-5","input":[
            {"role":"user","content":[{"type":"input_text","text":"list the files"}]}
        ]}"#;
        assert_eq!(
            request_guardrail_text(PassthroughProtocol::OpenaiResponses, req),
            "list the files"
        );
        // A bare-string input is equally valid.
        let req_str = br#"{"model":"gpt-5","input":"hello there"}"#;
        assert_eq!(
            request_guardrail_text(PassthroughProtocol::OpenaiResponses, req_str),
            "hello there"
        );

        let resp = br#"{"output":[
            {"type":"message","content":[{"type":"output_text","text":"done"}]}
        ],"usage":{"input_tokens":11,"output_tokens":3}}"#;
        assert_eq!(
            response_guardrail_text(PassthroughProtocol::OpenaiResponses, resp),
            "done"
        );
        assert_eq!(
            response_usage(PassthroughProtocol::OpenaiResponses, resp),
            Some(usage_dims(11, 3))
        );
    }

    #[test]
    fn responses_stream_accumulates_deltas_and_terminal_usage() {
        // Text arrives as output_text.delta events; the terminal
        // response.completed event repeats the full output (which must NOT
        // be appended again) and carries usage nested under `response`.
        let (t1, u1) = frame_delta(
            PassthroughProtocol::OpenaiResponses,
            br#"data: {"type":"response.output_text.delta","delta":"he"}"#,
        );
        assert_eq!(t1, "he");
        assert_eq!(u1, None);
        let (t2, _) = frame_delta(
            PassthroughProtocol::OpenaiResponses,
            br#"data: {"type":"response.output_text.delta","delta":"llo"}"#,
        );
        assert_eq!(t2, "llo");
        let terminal = br#"data: {"type":"response.completed","response":{"output":[{"content":[{"text":"hello"}]}],"usage":{"input_tokens":7,"output_tokens":2}}}"#;
        let (t3, u3) = frame_delta(PassthroughProtocol::OpenaiResponses, terminal);
        assert_eq!(
            t3, "",
            "terminal event must not duplicate the streamed text"
        );
        assert_eq!(u3, Some(usage_dims(7, 2)));

        // The nested shape is read ONLY for Responses: another protocol's
        // frame that happens to carry `response.usage` must not have its
        // reported usage overwritten from there.
        for other in [
            PassthroughProtocol::Raw,
            PassthroughProtocol::OpenaiChat,
            PassthroughProtocol::OpenaiCompletions,
        ] {
            let (_, u) = frame_delta(other, terminal);
            assert_eq!(u, None, "{other:?} must ignore nested response.usage");
        }
    }

    #[test]
    fn response_usage_reads_both_spellings() {
        let openai = br#"{"usage":{"prompt_tokens":5,"completion_tokens":2}}"#;
        assert_eq!(
            response_usage(PassthroughProtocol::OpenaiChat, openai),
            Some(usage_dims(5, 2))
        );
        let anthropicish = br#"{"usage":{"input_tokens":9,"output_tokens":4}}"#;
        assert_eq!(
            response_usage(PassthroughProtocol::OpenaiChat, anthropicish),
            Some(usage_dims(9, 4))
        );
        assert_eq!(response_usage(PassthroughProtocol::Raw, openai), None);
    }

    #[tokio::test]
    async fn inject_strips_caller_credentials_even_with_empty_strip_headers() {
        // A ProviderKey whose strip_headers is explicitly EMPTY: the
        // legacy tunnel documented that as "forward the caller's
        // credential beside the injected one"; routes never double-send —
        // forward_client is the explicit BYO mode.
        let upstream = MockServer::start().await;
        Mock::given(wm_method("GET"))
            .and(wm_path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"ok": true})))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        let pk_json = r#"{"display_name":"openai-up","secret":"sk-upstream","api_base":"http://unused",
                 "provider":"openai","adapter":"openai","strip_headers":[]}"#;
        let pk: ProviderKey = serde_json::from_str(pk_json).unwrap();
        snap.provider_keys.insert(ResourceEntry::new(PK_ID, pk, 1));
        snap.apikeys.insert(apikey_entry("sk-caller", Some(&["*"])));
        snap.passthrough_routes
            .insert(inject_route(&upstream.uri()));
        let app = build_app(snap);

        let req = Request::builder()
            .method("GET")
            .uri("/passthrough/openai/v1/models")
            .header("authorization", "Bearer sk-caller")
            .header("x-api-key", "caller-alt-cred")
            .header("x-aisix-request-id", "caller-forged-id")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let received = &upstream.received_requests().await.unwrap()[0];
        let auths: Vec<_> = received.headers.get_all("authorization").iter().collect();
        assert_eq!(auths.len(), 1, "exactly one Authorization on the wire");
        assert_eq!(auths[0], "Bearer sk-upstream");
        assert!(received.headers.get("x-api-key").is_none());
        // Exactly one correlation id on the wire: the inbound copy is
        // stripped and the dispatch sets the request's resolved id (which
        // `ensure_request_id` may legitimately adopt from the caller) —
        // pre-fix the upstream saw BOTH values as duplicates.
        let rid: Vec<_> = received
            .headers
            .get_all("x-aisix-request-id")
            .iter()
            .collect();
        assert_eq!(rid.len(), 1);
    }

    #[test]
    fn copy_safe_headers_preserves_repeated_values() {
        let mut src = HeaderMap::new();
        src.append("set-cookie", HeaderValue::from_static("a=1"));
        src.append("set-cookie", HeaderValue::from_static("b=2"));
        src.append("vary", HeaderValue::from_static("accept"));
        let mut dst = HeaderMap::new();
        copy_safe_headers(&src, &mut dst);
        let cookies: Vec<_> = dst.get_all("set-cookie").iter().collect();
        assert_eq!(cookies.len(), 2, "both Set-Cookie values must relay");
    }

    #[test]
    fn sse_splitter_bounds_an_unterminated_frame() {
        let mut s = SseFrameSplitter::new();
        // Feed > MAX_HELD_STREAM_BYTES without a frame terminator: the
        // splitter must hand the oversized run on instead of buffering
        // without bound.
        let chunk = vec![b'x'; 256 * 1024];
        let mut emitted = 0usize;
        for _ in 0..8 {
            emitted += s.push(&chunk).iter().map(Vec::len).sum::<usize>();
        }
        assert!(
            emitted >= MAX_HELD_STREAM_BYTES,
            "oversized unterminated run must be flushed ({emitted} emitted)"
        );
        assert!(s.take_rest().len() <= MAX_HELD_STREAM_BYTES);
    }

    #[test]
    fn push_capped_respects_byte_cap_on_char_boundaries() {
        let mut buf = String::new();
        push_capped(&mut buf, "héllo", Some(3));
        assert!(buf.len() <= 3);
        assert!(buf.starts_with('h'));
        push_capped(&mut buf, "more", None);
        assert!(buf.len() <= 3);
    }
}
