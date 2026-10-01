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
//! Two further request shapes are recognised among bodies that carry no
//! LLM envelope ([`detect_raw_usage_shape`]): a Cohere-style rerank
//! (`query` + a `documents` array) and the DashScope native service
//! envelope (`model` + an `input` object). They stay opaque for guardrails,
//! capture and streams; only their `model` and a unary JSON response's
//! token counts are read.
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
//!   keeps the pre-existing plain 404, `/passthrough/*` included — that
//!   namespace is claimed by explicit routes like any other.
//! - [`host_dispatch`] — a **pre-routing** middleware (after URL rewriting in
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
use percent_encoding::percent_decode;
use url::Url;

use aisix_core::resource::ResourceEntry;
use aisix_core::{PassthroughAuthMode, PassthroughCredentialMode, PassthroughRoute};

use crate::auth::AuthenticatedKey;
use crate::error::ProxyError;
use crate::host::inbound_host;
use crate::state::ProxyState;

/// Bounded `model` metric label for passthrough-route requests. Route
/// traffic resolves no Model; per-route attribution lives on the usage
/// event (`passthrough_route_name`), not in Prometheus label space.
const PASSTHROUGH_MODEL_LABEL: &str = "passthrough";

/// `provider` metric label for `forward_client` routes, which have no
/// ProviderKey to take a provider name from.
const BYO_PROVIDER_LABEL: &str = "byo";

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
        // Every unmatched path, `/passthrough/*` included, takes the
        // router's ordinary miss path: the namespace is entirely the
        // operator's to claim with explicit `passthrough_route` resources.
        crate::reject::emit_unrouted_access_log(
            method.as_str(),
            &path,
            &request_id,
            None,
            StatusCode::NOT_FOUND.as_u16(),
            started,
        );
        return StatusCode::NOT_FOUND.into_response();
    };

    let route_name = matched.entry.value.name.clone();
    // The route is this family's attribution — it names no model — so the
    // cancel guard needs it to file a row for a caller that hangs up while
    // the upstream is still thinking (AISIX-Cloud#1571).
    crate::attribution::note_passthrough_route(&route_name);

    // Filled inside `dispatch` at chain resolution, so the failure branch
    // — where an input-guardrail block lands — stamps the enforced hits
    // too (AISIX-Cloud#1330 / #1024).
    let mut audit = crate::usage_attr::GuardrailAudit::default();
    match dispatch(
        &state, &snapshot, &matched, req, &client, started, &mut audit,
    )
    .await
    {
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
                error.is_guardrail_block(),
                &client,
                crate::usage_attr::applied_guardrails(&audit),
                crate::usage_attr::enforced_hits(&audit),
                crate::usage_attr::guardrail_scores(&audit),
                crate::usage_attr::bypass_reason(&audit),
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
                crate::operation::PASSTHROUGH,
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
    audit_out: &mut crate::usage_attr::GuardrailAudit,
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
    let url =
        join_target_url(&base, rest, query.as_deref()).map_err(|e| RouteError::of(e, &auth))?;

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
    *audit_out = resolved_chain.audit_log();
    let mut monitor_hits: Vec<aisix_core::GuardrailMonitorHit> = Vec::new();

    // Envelope detection: once per exchange, from the request body's
    // top-level keys; the response and stream frames reuse it.
    let protocol = detect_protocol(&body_bytes);
    let raw_shape = detect_raw_usage_shape(protocol, &body_bytes);

    // INPUT guardrails on the (envelope-extracted) request text.
    if !resolved_chain.is_empty() {
        let text = match try_request_guardrail_text(protocol, &body_bytes) {
            Ok(text) => Some(text),
            Err(err) if !err.is_depth_exceeded() => {
                Some(request_guardrail_text(protocol, &body_bytes))
            }
            Err(err)
                if !aisix_guardrails::Guardrail::refuses_unevaluable_input(&resolved_chain) =>
            {
                tracing::debug!(
                    guardrail_hook = "input",
                    route = %route.name,
                    error = %err,
                    "cannot scan passthrough-route request to its bounded depth; nothing attached both reads the request and fails closed",
                );
                resolved_chain.record_unevaluable_input_bypass(crate::error::TAG_UNSCANNABLE_BODY);
                None
            }
            Err(err) => {
                tracing::warn!(
                    guardrail_hook = "input",
                    route = %route.name,
                    error = %err,
                    "cannot scan passthrough-route request to its bounded depth; blocking",
                );
                return Err(RouteError::of(
                    crate::error::guardrail_block_error(
                        "request",
                        None,
                        Some(crate::error::TAG_UNSCANNABLE_BODY),
                    ),
                    &auth,
                ));
            }
        };
        if let Some(text) = text {
            let chat = aisix_gateway::ChatFormat::new(
                route.name.clone(),
                vec![aisix_gateway::ChatMessage::user(text)],
            );
            let (verdict, hits) = aisix_guardrails::Guardrail::check_input_unmaskable_observed(
                &resolved_chain,
                &chat,
            )
            .await;
            monitor_hits.extend(hits);
            if let aisix_guardrails::GuardrailVerdict::Block {
                reason,
                guardrail_name,
                unavailable,
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
                    crate::error::guardrail_block_error(
                        "request",
                        guardrail_name.as_deref(),
                        unavailable.as_deref(),
                    ),
                    &auth,
                ));
            }
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
    let requested_model = body_model_name(protocol, raw_shape, &body_bytes);

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
    let reservation = crate::quota::enforce(state, snapshot, &auth, model_rl.as_ref())
        .await
        .map_err(|e| RouteError::of(e, &auth))?;

    // ----- outbound request -----

    let conn = pk_entry
        .as_ref()
        .and_then(|pk| pk.value.upstream_connection());
    let http_client = crate::http_client::client_for(conn.as_ref());

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

    // A route forwards the caller's headers by default, so the operator's
    // `forward_client_headers` is an OVERRIDE of the strip set above: the
    // names it admits ride upstream even though this route would otherwise
    // have removed them. That is what puts the caller's own credential on
    // an internal upstream that authorizes on it — in `gateway_key` mode
    // `authorization` is exactly the header the gateway just consumed to
    // identify this caller, and the strip set would otherwise take it.
    //
    // `header_forward_blocked` still holds: `host`, the hop-by-hop
    // headers, and the gateway's own namespace break the exchange rather
    // than changing who it comes from, so no pattern reaches them. And
    // `content-length` on top of it, which the standard pipeline gets from
    // its second tier: reqwest derives the outbound length from the body
    // it is handed, but hyper honours a caller-set value verbatim instead,
    // so a relayed copy is a request-framing bug waiting for the first
    // body this route rewrites.
    //
    // The exact-name rule is per ROUTE here. `/v1/*` and MCP read the
    // caller's credential out of `authorization` or `x-api-key`, both on
    // the shared list, but a route names its own slots: under `auth_mode:
    // header_key` the gateway credential arrives in `auth_header_name`,
    // and `identity_header` is one the route promises to strip. Neither
    // can be a name the shared list already covers in any way that helps:
    // the route schema rejects most of them outright, and the two it
    // permits are on that list anyway. So without this a `["x-*"]`
    // pattern would relay the very header this gateway authenticated the
    // caller with. Naming either in full still forwards it — the rule is
    // unchanged, only its input.
    //
    // A fixed array rather than a collected `Vec`: there are at most two,
    // on a per-request path. An unset slot stands as `""`, which matches
    // nothing — a header name is never empty, on the wire or in the
    // schema, so the empty entry needs no filtering out.
    let route_slots = [
        route.auth_header_name.as_deref().unwrap_or_default(),
        route.identity_header.as_deref().unwrap_or_default(),
    ];
    let forwards = |name: &str| {
        aisix_core::forward_pattern_admits_with(&route.forward_client_headers, name, &route_slots)
            && !aisix_core::header_forward_blocked(name)
            && name != "content-length"
    };

    let mut builder = http_client.request(method.clone(), &url);
    // Which slots the caller's own headers are taking, so the injection
    // below leaves them alone. Resolved from what the caller ACTUALLY
    // sent, not from the configuration: an operator who opts a slot in
    // must not blank the gateway's credential for every caller who happens
    // to send nothing there.
    let mut forwarded_slots: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (name, value) in &incoming_headers {
        let lower = name.as_str().to_ascii_lowercase();
        // Asked of EVERY inbound header, not only the ones the strip set
        // named: `x-aisix-*` is the gateway's own namespace, and only
        // `x-aisix-request-id` was ever in `ALWAYS_STRIP`, so a caller's
        // `x-aisix-routing-tags` used to ride upstream and forge a
        // gateway assertion there. Nothing an operator writes overrides
        // this, which is what the field's own description promises.
        if aisix_core::header_forward_blocked(&lower) {
            continue;
        }
        if strip.contains(&lower) {
            if !forwards(&lower) {
                continue;
            }
            forwarded_slots.insert(lower);
        }
        builder = builder.header(name, value);
    }

    // Inject the gateway-held upstream credential (inject mode only).
    // Strip ran first, so this never adds a second value to a slot the
    // caller's own header already took (#411 ordering). That is a
    // statement about the INJECTION, not about the wire: a caller who
    // repeated the slot still has every value relayed below, which is
    // what `forward_client_headers` promises on this surface.
    if let Some(pk) = pk_entry.as_ref() {
        let api_key = pk.value.api_key.as_str();
        let provider_lower = pk.value.provider.to_ascii_lowercase();
        if provider_lower == "anthropic" {
            // Anthropic's documented auth shape (#166): `x-api-key` +
            // `anthropic-version`, never a redundant Bearer alongside.
            if !forwarded_slots.contains("x-api-key") {
                builder = builder.header("x-api-key", api_key);
            }
            // Only when the caller sent none. `RequestBuilder::header`
            // appends, and `anthropic-version` is in no strip set — every
            // Anthropic SDK sends its own, so injecting unconditionally
            // put two revisions on the wire and let the upstream pick.
            // A route relays the body verbatim and decodes nothing, so
            // the caller's revision is the right one to keep.
            if !incoming_headers.contains_key("anthropic-version") {
                builder = builder.header("anthropic-version", "2023-06-01");
            }
        } else if !forwarded_slots.contains("authorization") {
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
    // A healthy SSE relay can be arbitrarily long, but no single silence
    // gap may keep its concurrency reservation forever. Route-level timeout
    // is the most specific bound; otherwise mirror the deployment stream →
    // request fallback used by typed streaming routes.
    let stream_read_timeout = route
        .timeout_ms
        .map(Duration::from_millis)
        .or(state.default_timeouts.stream)
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
    // Explicit `text/event-stream` only. Deliberately STRICTER than
    // `dispatch::upstream_body_is_sse`, which the typed relays use: a
    // passthrough route carries arbitrary REST traffic where most responses
    // are not SSE, so an unknown content type buffers — the arm that scans
    // — here, while on a relay that has just asked an LLM to stream the
    // same guess would 502 an upstream that merely mislabels itself. Not
    // drift: see that function's doc comment for why the two populations
    // take opposite defaults.
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
        user_id: auth.entry.value.user_id.clone(),
        user_name: auth.entry.value.user_name.clone(),
        jwt: auth.jwt.clone(),
        anonymous: auth.anonymous,
        client_identity,
        client_source_ip: client.source_ip.clone(),
        client_user_agent: client.user_agent.clone(),
        started,
        attempt_started,
        status: status.as_u16(),
        usage: None,
        requested_model,
        upstream_ttft_ms: 0,
        downstream_first_ms: None,
        stream_reached_end: false,
        streaming: false,
        error_class: String::new(),
        error_message: String::new(),
        failure_status: None,
        monitor_hits,
        audit: audit_out.clone(),
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
            reservation.into_stream_hold(),
            stream_read_timeout,
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
        let text = match try_response_guardrail_text(protocol, &resp_body) {
            Ok(text) => Some(text),
            Err(err) if !err.is_depth_exceeded() => {
                Some(response_guardrail_text(protocol, &resp_body))
            }
            Err(err)
                if !aisix_guardrails::Guardrail::refuses_unevaluable_output(&resolved_chain) =>
            {
                tracing::debug!(
                    guardrail_hook = "output",
                    route = %route.name,
                    error = %err,
                    "cannot scan passthrough-route response to its bounded depth; nothing attached both reads the response and fails closed",
                );
                resolved_chain.record_unevaluable_output_bypass(crate::error::TAG_UNSCANNABLE_BODY);
                None
            }
            Err(err) => {
                tracing::warn!(
                    guardrail_hook = "output",
                    route = %route.name,
                    error = %err,
                    "cannot scan passthrough-route response to its bounded depth; blocking",
                );
                telemetry.guardrail_blocked = true;
                telemetry.emitted = true;
                return Err(RouteError::of(
                    crate::error::guardrail_block_error(
                        "response",
                        None,
                        Some(crate::error::TAG_UNSCANNABLE_BODY),
                    ),
                    &auth,
                ));
            }
        };
        if let Some(text) = text {
            let synth = aisix_gateway::ChatResponse {
                id: String::new(),
                model: route.name.clone(),
                message: aisix_gateway::ChatMessage::assistant(text),
                finish_reason: aisix_gateway::FinishReason::Stop,
                usage: aisix_gateway::UsageStats::default(),
            };
            let (verdict, hits) = aisix_guardrails::Guardrail::check_output_unmaskable_observed(
                &resolved_chain,
                &synth,
            )
            .await;
            telemetry.monitor_hits.extend(hits);
            if let aisix_guardrails::GuardrailVerdict::Block {
                reason,
                guardrail_name,
                unavailable,
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
                    crate::error::guardrail_block_error(
                        "response",
                        guardrail_name.as_deref(),
                        unavailable.as_deref(),
                    ),
                    &auth,
                ));
            }
        }
    }

    if let Some(u) = response_usage(protocol, raw_shape, &resp_body) {
        merge_usage(&mut telemetry.usage, u);
    }
    if telemetry.content_cap.is_some() {
        telemetry.response_text = response_capture_text(protocol, &resp_body);
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
            let authed = AuthenticatedKey {
                entry,
                jwt: None,
                anonymous: true,
            };
            // Verified credentials are noted inside `authenticate_token`;
            // a minted anonymous principal has to note itself, or a caller
            // that hangs up on an anonymous route files no row at all
            // (AISIX-Cloud#1571).
            crate::attribution::note_authenticated(&authed);
            Ok(authed)
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
/// plus the whole serialized `tool_calls` payload, and — on the request
/// side only — an assistant turn's replayed `reasoning_content`.
///
/// The tool-call half is what the typed endpoints scan (`message_scan_text`
/// in the guardrails crate), and it is not optional: a request whose only
/// sensitive text sits in a tool call's `arguments` would otherwise pass a
/// deny-list that the same body sent to `/v1/chat/completions` trips.
/// Serialising the whole payload means no function name or argument can
/// escape inspection regardless of the provider-specific shape. The same
/// argument carries `reasoning_content`, which relays upstream verbatim.
///
/// `reasoning` splits the two callers because this helper reads BOTH the
/// request's `messages[]` and the buffered response's `choices[].message`:
/// caller-replayed reasoning is request text and in scope, while reasoning
/// the model generated is out of the output-guardrail scope.
fn message_scan_text(msg: &serde_json::Value, reasoning: bool) -> String {
    let mut parts: Vec<String> = Vec::new();
    let content = msg
        .get("content")
        .map(|c| {
            if reasoning {
                request_content_text(c)
            } else {
                content_text(c)
            }
        })
        .unwrap_or_default();
    if !content.is_empty() {
        parts.push(content);
    }
    if let Some(tool_calls) = msg.get("tool_calls").filter(|t| !t.is_null()) {
        parts.push(tool_calls.to_string());
    }
    if reasoning {
        if let Some(r) = msg
            .get("reasoning_content")
            .and_then(|v| v.as_str())
            .filter(|r| !r.is_empty())
        {
            parts.push(r.to_string());
        }
    }
    parts.join("\n")
}

/// Request text of one chat-envelope `content` value, which on this
/// envelope may also be an Anthropic Messages block array: `text` blocks,
/// a `tool_result`'s own content, a replayed `tool_use`'s input, and a
/// replayed `thinking` block — the same slots the typed `/v1/messages`
/// route scans once it has parsed the body. Caller-replayed reasoning is
/// request text, as it is for `reasoning_content` above.
fn request_content_text(v: &serde_json::Value) -> String {
    let serde_json::Value::Array(blocks) = v else {
        return content_text(v);
    };
    blocks
        .iter()
        .filter_map(|b| match b.get("type").and_then(|t| t.as_str()) {
            Some("tool_result") => b.get("content").map(request_content_text),
            Some("tool_use") => b.get("input").map(|i| i.to_string()),
            Some("thinking") => b
                .get("thinking")
                .and_then(|t| t.as_str())
                .map(str::to_string),
            _ => b.get("text").and_then(|t| t.as_str()).map(str::to_string),
        })
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// The generated text of a buffered Anthropic Messages response carried on
/// the chat envelope: its `text` blocks and each `tool_use` input.
/// `thinking` is generated reasoning and stays out of the output scan.
fn anthropic_message_output_text(v: &serde_json::Value) -> String {
    v.get("content")
        .and_then(|c| c.as_array())
        .into_iter()
        .flatten()
        .filter_map(|b| match b.get("type").and_then(|t| t.as_str()) {
            Some("text") => b.get("text").and_then(|t| t.as_str()).map(str::to_string),
            Some("tool_use") => b.get("input").map(|i| i.to_string()),
            _ => None,
        })
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// The body envelope detected for one exchange. Not configuration:
/// detected per request from the body's top-level keys
/// ([`detect_protocol`]) and sticky for the exchange — the buffered
/// response and every stream frame are read with the same detection. It
/// drives extraction (guardrail text, capture, usage) only; the relay
/// forwards bytes verbatim regardless.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PassthroughProtocol {
    /// No recognized envelope: guardrails scan every decoded JSON string
    /// value, falling back to one lossy-UTF-8 text when the body is not JSON;
    /// buffered responses are not probed for usage.
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
/// LLM envelopes are structurally exclusive — `messages`, `input` (a string
/// or an array) and `prompt` are each the required carrier field of exactly
/// one API — so real LLM traffic detects unambiguously, and everything else
/// (JSON-RPC, REST, rerank, DashScope native, non-JSON, empty/GET bodies)
/// is `Raw`; [`detect_raw_usage_shape`] then picks out the `Raw` bodies
/// whose unary response is still meterable. An unknown API colliding
/// with a carrier key costs nothing: detection drives extraction only,
/// and extraction degrades to the whole body when the detected shape
/// yields no text.
fn detect_protocol(body: &[u8]) -> PassthroughProtocol {
    // Do not materialize the entire document just to inspect its envelope:
    // a valid request can exceed serde_json::Value's nesting limit in an
    // unrelated forwarded field. `RawValue` keeps the chosen top-level
    // carrier shallow while preserving the last-key behavior of a JSON map.
    if raw_top_level_last_has_shape(body, "messages", false) {
        PassthroughProtocol::OpenaiChat
    } else if raw_top_level_last_has_shape(body, "input", true) {
        PassthroughProtocol::OpenaiResponses
    } else if raw_top_level_last_has_shape(body, "prompt", true) {
        PassthroughProtocol::OpenaiCompletions
    } else {
        PassthroughProtocol::Raw
    }
}

/// A `Raw` request whose `model` and unary JSON response usage the gateway
/// can still read. Consulted for model attribution and the BUFFERED
/// response's usage only — guardrail text, capture and every stream frame
/// keep the `Raw` treatment, so a streamed response reports tokens only
/// from a server-labelled usage frame, exactly as any opaque stream does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RawUsageShape {
    /// Cohere-style rerank: top-level `query` beside a `documents` array
    /// (Cohere, Jina, DashScope's compatible rerank). Usage is read exactly
    /// as `/v1/rerank` reads it ([`crate::rerank::rerank_prompt_tokens`]).
    Rerank,
    /// DashScope native service envelope: `{model, input: {…}, parameters}`
    /// (native rerank, multimodal embedding, …). Usage per
    /// [`dashscope_native_usage`].
    DashscopeNative,
}

/// Classify a body [`detect_protocol`] left `Raw`; `None` for any other
/// protocol, so the three LLM envelopes are never re-read.
fn detect_raw_usage_shape(protocol: PassthroughProtocol, body: &[u8]) -> Option<RawUsageShape> {
    if !matches!(protocol, PassthroughProtocol::Raw) {
        return None;
    }
    let v = serde_json::from_slice::<serde_json::Value>(body).ok()?;
    if v.get("documents").is_some_and(serde_json::Value::is_array) && v.get("query").is_some() {
        Some(RawUsageShape::Rerank)
    } else if v.get("model").is_some_and(serde_json::Value::is_string)
        && v.get("input").is_some_and(serde_json::Value::is_object)
    {
        Some(RawUsageShape::DashscopeNative)
    } else {
        None
    }
}

/// Cap on the recorded `requested_model` value. The body is the caller's,
/// so the alias is bounded before it reaches telemetry.
const REQUESTED_MODEL_CAP: usize = 128;

/// The model alias the caller addressed, from a DETECTED envelope's own
/// `model` field — what the typed endpoint serving that envelope records
/// as `UsageEvent::requested_model`.
///
/// Read only for a recognised envelope or [`RawUsageShape`]: an opaque
/// body's `model`-shaped key belongs to some other API and means nothing
/// the gateway can attribute.
/// The Prometheus side is already collapse-guarded (an unregistered name
/// folds to the `unresolved` sentinel), so an arbitrary alias here cannot
/// mint label cardinality.
fn body_model_name(
    protocol: PassthroughProtocol,
    raw_shape: Option<RawUsageShape>,
    body: &[u8],
) -> String {
    if matches!(protocol, PassthroughProtocol::Raw) && raw_shape.is_none() {
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

fn append_scan_text(out: &mut String, text: &str) {
    if text.is_empty() {
        return;
    }
    if !out.is_empty() {
        out.push('\n');
    }
    out.push_str(text);
}

fn decoded_json_string_values(body: &[u8]) -> Option<String> {
    crate::json_splice::collect_string_values(body)
        .ok()
        .filter(|out| !out.is_empty())
}

fn decoded_json_string_values_where(
    body: &[u8],
    include: impl FnMut(&[crate::json_splice::PathSeg]) -> bool,
) -> Option<String> {
    crate::json_splice::collect_string_values_where(body, include)
        .ok()
        .filter(|out| !out.is_empty())
}

fn decoded_json_string_values_vec_where(
    body: &[u8],
    include: impl FnMut(&[crate::json_splice::PathSeg]) -> bool,
) -> Option<Vec<String>> {
    crate::json_splice::collect_string_values_where_vec(body, include)
        .ok()
        .filter(|out| !out.is_empty())
}

fn is_root_key(path: &[crate::json_splice::PathSeg], key: &str) -> bool {
    path.first().is_some_and(|segment| segment.is_key(key))
}

/// A detected envelope still forwards raw bytes, including duplicate keys and
/// arbitrary nested fields. Scan every decoded string the upstream can read;
/// the root `model` alone is routing metadata rather than caller content.
fn decoded_non_model_json_string_values(body: &[u8]) -> Option<String> {
    decoded_json_string_values_where(body, |path| !is_root_key(path, "model"))
}

fn decoded_json_string_values_including_empty(
    body: &[u8],
    scan_error: &mut Option<crate::json_splice::SpliceError>,
) -> Option<String> {
    match crate::json_splice::collect_string_values(body) {
        Ok(values) => Some(values),
        Err(error) => {
            if scan_error.is_none() {
                *scan_error = Some(error);
            }
            None
        }
    }
}

fn decoded_json_string_values_except_root_keys(
    body: &[u8],
    excluded: &[&str],
    scan_error: &mut Option<crate::json_splice::SpliceError>,
) -> Option<String> {
    let mut out = String::new();
    for value in raw_top_level_values_except(body, excluded)? {
        match crate::json_splice::collect_string_values(value.get().as_bytes()) {
            Ok(values) => append_scan_text(&mut out, &values),
            Err(error) => {
                if scan_error.is_none() {
                    *scan_error = Some(error);
                }
                return None;
            }
        }
    }
    Some(out)
}

/// Source values of all occurrences of one top-level key. `RawValue` keeps
/// repeated keys separate, unlike `serde_json::Value`.
fn raw_top_level_values(
    body: &[u8],
    wanted_key: &str,
) -> Option<Vec<Box<serde_json::value::RawValue>>> {
    struct Values<'a> {
        wanted_key: &'a str,
    }

    impl<'de> serde::de::Visitor<'de> for Values<'_> {
        type Value = Vec<Box<serde_json::value::RawValue>>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("a JSON object")
        }

        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::MapAccess<'de>,
        {
            let mut values = Vec::new();
            while let Some(key) = map.next_key::<String>()? {
                if key == self.wanted_key {
                    values.push(map.next_value::<Box<serde_json::value::RawValue>>()?);
                } else {
                    map.next_value::<serde::de::IgnoredAny>()?;
                }
            }
            Ok(values)
        }
    }

    let mut deserializer = serde_json::Deserializer::from_slice(body);
    let values =
        serde::de::Deserializer::deserialize_map(&mut deserializer, Values { wanted_key }).ok()?;
    deserializer.end().ok()?;
    Some(values)
}

/// Source values of top-level keys other than `excluded`. Values are captured
/// as raw JSON before filtering so a known opaque carrier can be skipped
/// without recursively deserializing its payload.
fn raw_top_level_values_except(
    body: &[u8],
    excluded: &[&str],
) -> Option<Vec<Box<serde_json::value::RawValue>>> {
    struct Values<'a> {
        excluded: &'a [&'a str],
    }

    impl<'de> serde::de::Visitor<'de> for Values<'_> {
        type Value = Vec<Box<serde_json::value::RawValue>>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("a JSON object")
        }

        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::MapAccess<'de>,
        {
            let mut values = Vec::new();
            while let Some(key) = map.next_key::<String>()? {
                let value = map.next_value::<Box<serde_json::value::RawValue>>()?;
                if !self.excluded.iter().any(|excluded| key == *excluded) {
                    values.push(value);
                }
            }
            Ok(values)
        }
    }

    let mut deserializer = serde_json::Deserializer::from_slice(body);
    let values =
        serde::de::Deserializer::deserialize_map(&mut deserializer, Values { excluded }).ok()?;
    deserializer.end().ok()?;
    Some(values)
}

/// Match the last source occurrence, the same duplicate-key convention a
/// materialized JSON map used before protocol detection became shallow.
/// `allow_string` is for the Responses and Completions bare-string forms;
/// Chat requires an array of messages.
fn raw_top_level_last_has_shape(body: &[u8], key: &str, allow_string: bool) -> bool {
    raw_top_level_values(body, key)
        .and_then(|values| values.into_iter().last())
        .is_some_and(|value| match value.get().trim_start().as_bytes().first() {
            Some(b'[') => true,
            Some(b'"') => allow_string,
            _ => false,
        })
}

fn raw_is_object(raw: &serde_json::value::RawValue) -> bool {
    raw.get().trim_start().starts_with('{')
}

fn raw_array_items(
    raw: &serde_json::value::RawValue,
) -> Option<Vec<Box<serde_json::value::RawValue>>> {
    serde_json::from_str(raw.get()).ok()
}

/// `true` only for an unambiguous typed item. Conflicting or non-string
/// duplicate `type` fields stay in the output scan rather than becoming a
/// way to hide content.
fn raw_object_has_only_types(raw: &serde_json::value::RawValue, allowed: &[&str]) -> bool {
    let Some(values) = raw_top_level_values(raw.get().as_bytes(), "type") else {
        return false;
    };
    let mut types = values
        .into_iter()
        .map(|value| serde_json::from_str::<String>(value.get()).ok());
    let Some(Some(first)) = types.next() else {
        return false;
    };
    allowed.iter().any(|allowed| first == *allowed)
        && types.all(|kind| kind.as_deref() == Some(first.as_str()))
}

fn raw_top_level_unique_type(body: &[u8]) -> Option<String> {
    let mut types = raw_top_level_values(body, "type")
        .into_iter()
        .flatten()
        .map(|value| serde_json::from_str::<String>(value.get()).ok());
    let first = types.next()??;
    types
        .all(|kind| kind.as_deref() == Some(first.as_str()))
        .then_some(first)
}

/// `true` only when every source `type` value is one of `allowed`. This is
/// deliberately strict: an audio or image event must not borrow a text
/// event's carrier merely by repeating a conflicting type.
fn raw_top_level_has_only_types(body: &[u8], allowed: &[&str]) -> bool {
    let Some(values) = raw_top_level_values(body, "type") else {
        return false;
    };
    !values.is_empty()
        && values.into_iter().all(|value| {
            serde_json::from_str::<String>(value.get())
                .ok()
                .is_some_and(|kind| allowed.iter().any(|allowed| kind == *allowed))
        })
}

fn raw_top_level_items_have_only_types(body: &[u8], key: &str, allowed: &[&str]) -> Option<bool> {
    let values = raw_top_level_values(body, key)?;
    Some(
        !values.is_empty()
            && values
                .iter()
                .all(|value| raw_object_has_only_types(value, allowed)),
    )
}

fn append_raw_string_value(out: &mut String, raw: &serde_json::value::RawValue) -> Option<()> {
    append_scan_text(out, &serde_json::from_str::<String>(raw.get()).ok()?);
    Some(())
}

fn append_raw_top_level_strings(out: &mut String, body: &[u8], key: &str) -> Option<()> {
    for value in raw_top_level_values(body, key)? {
        // A valid but wrongly typed nominal text field must not make its
        // opaque object/array sibling content eligible for a whole-body raw
        // fallback at the guardrail boundary.
        if let Ok(value) = serde_json::from_str::<String>(value.get()) {
            append_scan_text(out, &value);
        }
    }
    Some(())
}

fn raw_top_level_string_values(body: &[u8], key: &str) -> Option<Vec<String>> {
    raw_top_level_values(body, key)?
        .into_iter()
        .map(|value| serde_json::from_str::<String>(value.get()).ok())
        .collect()
}

fn raw_top_level_unique_string(body: &[u8], key: &str) -> Result<Option<String>, ()> {
    let mut values = raw_top_level_values(body, key).ok_or(())?;
    match values.len() {
        0 => Ok(None),
        1 => serde_json::from_str(values.pop().expect("one value").get())
            .map(Some)
            .map_err(|_| ()),
        _ => Err(()),
    }
}

fn raw_top_level_unique_index(body: &[u8], key: &str) -> Result<Option<usize>, ()> {
    let mut values = raw_top_level_values(body, key).ok_or(())?;
    match values.len() {
        0 => Ok(None),
        1 => serde_json::from_str(values.pop().expect("one value").get())
            .map(Some)
            .map_err(|_| ()),
        _ => Err(()),
    }
}

fn raw_top_level_unique_object(
    body: &[u8],
    key: &str,
) -> Result<Option<Box<serde_json::value::RawValue>>, ()> {
    let mut values = raw_top_level_values(body, key).ok_or(())?;
    match values.len() {
        0 => Ok(None),
        1 => {
            let value = values.pop().expect("one value");
            raw_is_object(&value).then_some(value).map(Some).ok_or(())
        }
        _ => Err(()),
    }
}

fn raw_top_level_unique_array(
    body: &[u8],
    key: &str,
) -> Result<Option<Box<serde_json::value::RawValue>>, ()> {
    let mut values = raw_top_level_values(body, key).ok_or(())?;
    match values.len() {
        0 => Ok(None),
        1 => {
            let value = values.pop().expect("one value");
            value
                .get()
                .trim_start()
                .starts_with('[')
                .then_some(value)
                .map(Some)
                .ok_or(())
        }
        _ => Err(()),
    }
}

/// The typed content extractors inspect a bare string or the direct `text`
/// field of typed parts. Keep that boundary when walking raw source, so image
/// and document payloads never reach external guardrails as text.
fn append_raw_text_value(out: &mut String, raw: &serde_json::value::RawValue) -> Option<()> {
    let value = raw.get().trim_start();
    if value.starts_with('"') {
        return append_raw_string_value(out, raw);
    }
    if !value.starts_with('[') {
        return Some(());
    }
    for part in raw_array_items(raw)? {
        if !raw_is_object(&part) {
            continue;
        }
        append_raw_top_level_strings(out, part.get().as_bytes(), "text")?;
    }
    Some(())
}

/// Source-preserving request text from Anthropic-compatible content blocks.
/// It mirrors [`request_content_text`]: `text`, nested `tool_result`, a
/// `tool_use` input, and plaintext `thinking` are input; image/document and
/// signed `redacted_thinking` payloads are deliberately opaque. An ambiguous
/// duplicate `type` is scanned as source rather than becoming a bypass.
fn append_chat_request_content_strings(
    out: &mut String,
    content: Box<serde_json::value::RawValue>,
    scan_error: &mut Option<crate::json_splice::SpliceError>,
) -> Option<()> {
    enum Work {
        Content {
            value: Box<serde_json::value::RawValue>,
            depth: usize,
        },
        Block {
            value: Box<serde_json::value::RawValue>,
            depth: usize,
        },
    }

    // `tool_result.content` can itself contain another `tool_result`. Keep
    // that caller-controlled nesting off the Rust call stack. This measures
    // the content-carrier depth, while a wide valid array stays valid just as
    // it does for the byte scanner's frame stack.
    let mut work = vec![Work::Content {
        value: content,
        depth: 1,
    }];
    while let Some(work_item) = work.pop() {
        match work_item {
            Work::Content { value, depth } => {
                let value_text = value.get().trim_start();
                if value_text.starts_with('"') {
                    append_raw_string_value(out, &value)?;
                    continue;
                }
                if !value_text.starts_with('[') {
                    continue;
                }
                // Push backwards so the LIFO work stack preserves the
                // previous depth-first, source-order traversal.
                for block in raw_array_items(&value)?.into_iter().rev() {
                    work.push(Work::Block {
                        value: block,
                        depth,
                    });
                }
            }
            Work::Block {
                value: block,
                depth,
            } => {
                if !raw_is_object(&block) {
                    continue;
                }
                let block_body = block.get().as_bytes();
                let types = raw_top_level_values(block_body, "type")?;
                let kind = raw_top_level_unique_type(block_body);
                if !types.is_empty() && kind.is_none() {
                    append_scan_text(
                        out,
                        &decoded_json_string_values_including_empty(block_body, scan_error)?,
                    );
                    continue;
                }
                match kind.as_deref() {
                    Some("redacted_thinking") => {}
                    Some("tool_result") => {
                        let nested = raw_top_level_values(block_body, "content")?;
                        if nested.is_empty() {
                            continue;
                        }
                        let Some(depth) = depth
                            .checked_add(1)
                            .filter(|depth| *depth <= crate::json_splice::MAX_JSON_DEPTH)
                        else {
                            if scan_error.is_none() {
                                *scan_error =
                                    Some(crate::json_splice::SpliceError::depth_exceeded());
                            }
                            return None;
                        };
                        for nested in nested.into_iter().rev() {
                            work.push(Work::Content {
                                value: nested,
                                depth,
                            });
                        }
                    }
                    Some("tool_use") => {
                        for input in raw_top_level_values(block_body, "input")? {
                            append_scan_text(
                                out,
                                &decoded_json_string_values_including_empty(
                                    input.get().as_bytes(),
                                    scan_error,
                                )?,
                            );
                        }
                    }
                    Some("thinking") => append_raw_top_level_strings(out, block_body, "thinking")?,
                    _ => append_raw_top_level_strings(out, block_body, "text")?,
                }
            }
        }
    }
    Some(())
}

fn append_chat_request_message_strings(
    out: &mut String,
    message: &serde_json::value::RawValue,
    scan_error: &mut Option<crate::json_splice::SpliceError>,
) -> Option<()> {
    if !raw_is_object(message) {
        return Some(());
    }
    let message_body = message.get().as_bytes();
    append_scan_text(
        out,
        &decoded_json_string_values_except_root_keys(
            message_body,
            &["content", "tool_calls", "reasoning_content", "reasoning"],
            scan_error,
        )?,
    );
    for content in raw_top_level_values(message_body, "content")? {
        append_chat_request_content_strings(out, content, scan_error)?;
    }
    for tool_calls in raw_top_level_values(message_body, "tool_calls")? {
        append_scan_text(
            out,
            &decoded_json_string_values_including_empty(tool_calls.get().as_bytes(), scan_error)?,
        );
    }
    append_raw_top_level_strings(out, message_body, "reasoning_content")?;
    Some(())
}

fn decoded_chat_request_string_values(
    body: &[u8],
    scan_error: &mut Option<crate::json_splice::SpliceError>,
) -> Option<String> {
    let mut out = decoded_json_string_values_except_root_keys(
        body,
        &["model", "system", "messages"],
        scan_error,
    )?;
    for system in raw_top_level_values(body, "system")? {
        append_chat_request_content_strings(&mut out, system, scan_error)?;
    }
    for array in raw_top_level_values(body, "messages")? {
        // The selected (last) carrier made this a Chat envelope. Preserve
        // other duplicate source values without turning a malformed earlier
        // carrier into a whole-body fallback that exposes opaque media.
        let Some(messages) = raw_array_items(&array) else {
            continue;
        };
        for message in messages {
            append_chat_request_message_strings(&mut out, &message, scan_error)?;
        }
    }
    Some(out)
}

fn append_responses_item_strings(
    out: &mut String,
    item: &serde_json::value::RawValue,
    scan_error: &mut Option<crate::json_splice::SpliceError>,
) -> Option<()> {
    if !raw_is_object(item) {
        return Some(());
    }
    let item_body = item.get().as_bytes();
    let text_keys = [
        "content",
        "output",
        "reason",
        "summary",
        "name",
        "arguments",
        "input",
    ];
    append_scan_text(
        out,
        &decoded_json_string_values_except_root_keys(item_body, &text_keys, scan_error)?,
    );
    for key in text_keys {
        for value in raw_top_level_values(item_body, key)? {
            append_raw_text_value(out, &value)?;
        }
    }
    Some(())
}

fn decoded_responses_request_string_values(
    body: &[u8],
    scan_error: &mut Option<crate::json_splice::SpliceError>,
) -> Option<String> {
    let mut out =
        decoded_json_string_values_except_root_keys(body, &["model", "input"], scan_error)?;
    for input in raw_top_level_values(body, "input")? {
        let value = input.get().trim_start();
        if value.starts_with('"') {
            append_raw_string_value(&mut out, &input)?;
        } else if value.starts_with('[') {
            for item in raw_array_items(&input)? {
                append_responses_item_strings(&mut out, &item, scan_error)?;
            }
        }
    }
    Some(out)
}

fn decoded_completions_request_string_values(
    body: &[u8],
    scan_error: &mut Option<crate::json_splice::SpliceError>,
) -> Option<String> {
    let mut out = decoded_json_string_values_except_root_keys(
        body,
        &["model", "prompt", "suffix"],
        scan_error,
    )?;
    for prompt in raw_top_level_values(body, "prompt")? {
        let value = prompt.get().trim_start();
        if value.starts_with('"') {
            append_raw_string_value(&mut out, &prompt)?;
        } else if value.starts_with('[') {
            for part in raw_array_items(&prompt)? {
                if part.get().trim_start().starts_with('"') {
                    append_raw_string_value(&mut out, &part)?;
                }
            }
        }
    }
    append_raw_top_level_strings(&mut out, body, "suffix")?;
    Some(out)
}

/// The request text a guardrail scans, per the detected envelope.
///
/// The route relays source bytes verbatim, while `serde_json::Value` drops
/// duplicate keys and stops at its default nesting limit. Scan decoded source
/// values while preserving typed opaque boundaries: signed Anthropic
/// `redacted_thinking`, image, and document payloads are not caller text.
fn request_guardrail_text_with_scan_error(
    protocol: PassthroughProtocol,
    body: &[u8],
    scan_error: &mut Option<crate::json_splice::SpliceError>,
) -> String {
    let raw = || String::from_utf8_lossy(body).into_owned();
    match protocol {
        PassthroughProtocol::Raw => decoded_json_string_values(body).unwrap_or_else(raw),
        PassthroughProtocol::OpenaiChat => {
            decoded_chat_request_string_values(body, scan_error).unwrap_or_else(raw)
        }
        PassthroughProtocol::OpenaiCompletions => {
            decoded_completions_request_string_values(body, scan_error).unwrap_or_else(raw)
        }
        PassthroughProtocol::OpenaiResponses => {
            decoded_responses_request_string_values(body, scan_error).unwrap_or_else(raw)
        }
    }
}

fn request_guardrail_text(protocol: PassthroughProtocol, body: &[u8]) -> String {
    let mut ignored_scan_error = None;
    request_guardrail_text_with_scan_error(protocol, body, &mut ignored_scan_error)
}

/// Like [`request_guardrail_text`], but preserves scanner failures from the
/// exact typed source selectors that read a value for input inspection.
fn try_request_guardrail_text(
    protocol: PassthroughProtocol,
    body: &[u8],
) -> Result<String, crate::json_splice::SpliceError> {
    if matches!(protocol, PassthroughProtocol::Raw) {
        return crate::json_splice::collect_string_values(body);
    }
    let mut scan_error = None;
    let text = request_guardrail_text_with_scan_error(protocol, body, &mut scan_error);
    scan_error.map_or(Ok(text), Err)
}

/// Return the string field which an explicitly typed Chat content part
/// exposes to the client. Image, audio, file, and future part types stay
/// opaque at the external output-guardrail boundary.
fn chat_visible_content_part_field(kind: &str) -> Option<&'static str> {
    match kind {
        "text" => Some("text"),
        "refusal" => Some("refusal"),
        _ => None,
    }
}

/// Append a Chat `content` value that is known to be client-visible output.
/// A bare string is the Chat response's ordinary text shape; array entries
/// need one unambiguous known discriminator before their text or refusal
/// field may cross the output-guardrail boundary.
fn append_chat_visible_content_strings(
    out: &mut String,
    content: &serde_json::value::RawValue,
) -> Option<()> {
    let value = content.get().trim_start();
    if value.starts_with('"') {
        return append_raw_string_value(out, content);
    }
    if !value.starts_with('[') {
        return Some(());
    }
    for part in raw_array_items(content)? {
        if !raw_is_object(&part) {
            continue;
        }
        let part_body = part.get().as_bytes();
        if let Some(field) = raw_top_level_unique_type(part_body)
            .as_deref()
            .and_then(chat_visible_content_part_field)
        {
            append_raw_top_level_strings(out, part_body, field)?;
        }
    }
    Some(())
}

/// The stream can omit a tool-call's discriminator after its first delta.
/// Accept that continuation shape, but never let an explicit unknown or
/// conflicting type borrow a function/custom field as visible tool text.
fn chat_tool_continuation_fields(body: &[u8]) -> Option<&'static [(&'static str, &'static str)]> {
    let types = raw_top_level_values(body, "type")?;
    match raw_top_level_unique_type(body).as_deref() {
        Some("function") => Some(&[("function", "arguments")]),
        Some("custom") => Some(&[("custom", "input")]),
        None if types.is_empty() => Some(&[("function", "arguments"), ("custom", "input")]),
        _ => Some(&[]),
    }
}

fn append_chat_tool_call_strings(
    out: &mut String,
    tool_calls: &serde_json::value::RawValue,
) -> Option<()> {
    if !tool_calls.get().trim_start().starts_with('[') {
        return Some(());
    }
    for tool_call in raw_array_items(tool_calls)? {
        if !raw_is_object(&tool_call) {
            continue;
        }
        let tool_body = tool_call.get().as_bytes();
        for (container, field) in chat_tool_continuation_fields(tool_body)? {
            for payload in raw_top_level_values(tool_body, container)? {
                if !raw_is_object(&payload) {
                    continue;
                }
                let payload_body = payload.get().as_bytes();
                append_raw_top_level_strings(out, payload_body, "name")?;
                append_raw_top_level_strings(out, payload_body, field)?;
            }
        }
    }
    Some(())
}

/// Preserve the pre-`tool_calls` Chat tool shape without widening the
/// response walk beyond its explicit `name` and `arguments` fields.
fn append_chat_legacy_function_call_strings(
    out: &mut String,
    function_call: &serde_json::value::RawValue,
) -> Option<()> {
    if !raw_is_object(function_call) {
        return Some(());
    }
    let function_body = function_call.get().as_bytes();
    append_raw_top_level_strings(out, function_body, "name")?;
    append_raw_top_level_strings(out, function_body, "arguments")?;
    Some(())
}

fn append_chat_output_message_strings(
    out: &mut String,
    message: &serde_json::value::RawValue,
) -> Option<()> {
    if !raw_is_object(message) {
        return Some(());
    }
    let message_body = message.get().as_bytes();
    for content in raw_top_level_values(message_body, "content")? {
        append_chat_visible_content_strings(out, &content)?;
    }
    // OpenAI Chat also exposes a refusal as a direct message member rather
    // than a typed content part.
    append_raw_top_level_strings(out, message_body, "refusal")?;
    for tool_calls in raw_top_level_values(message_body, "tool_calls")? {
        append_chat_tool_call_strings(out, &tool_calls)?;
    }
    for function_call in raw_top_level_values(message_body, "function_call")? {
        append_chat_legacy_function_call_strings(out, &function_call)?;
    }
    Some(())
}

/// Anthropic Messages replies can travel through a Chat passthrough route.
/// Their generated text and tool-use payloads are visible output; all other
/// content-block kinds remain opaque.
fn append_anthropic_output_content_strings(
    out: &mut String,
    content: &serde_json::value::RawValue,
    scan_error: &mut Option<crate::json_splice::SpliceError>,
) -> Option<()> {
    if !content.get().trim_start().starts_with('[') {
        return Some(());
    }
    for block in raw_array_items(content)? {
        if !raw_is_object(&block) {
            continue;
        }
        let block_body = block.get().as_bytes();
        match raw_top_level_unique_type(block_body).as_deref() {
            Some("text") => append_raw_top_level_strings(out, block_body, "text")?,
            Some("tool_use") => {
                append_raw_top_level_strings(out, block_body, "name")?;
                for input in raw_top_level_values(block_body, "input")? {
                    append_scan_text(
                        out,
                        &decoded_json_string_values_including_empty(
                            input.get().as_bytes(),
                            scan_error,
                        )?,
                    );
                }
            }
            Some(_) | None => {}
        }
    }
    Some(())
}

fn decoded_chat_response_string_values(
    body: &[u8],
    scan_error: &mut Option<crate::json_splice::SpliceError>,
) -> Option<String> {
    let mut out = String::new();
    let choices = raw_top_level_values(body, "choices")?;
    let has_chat_choices = choices
        .iter()
        .any(|choices| choices.get().trim_start().starts_with('['));
    for array in choices {
        let Some(choices) = raw_array_items(&array) else {
            continue;
        };
        for choice in choices {
            if !raw_is_object(&choice) {
                continue;
            }
            for message in raw_top_level_values(choice.get().as_bytes(), "message")? {
                append_chat_output_message_strings(&mut out, &message)?;
            }
        }
    }
    if !has_chat_choices && raw_top_level_unique_type(body).as_deref() == Some("message") {
        for content in raw_top_level_values(body, "content")? {
            append_anthropic_output_content_strings(&mut out, &content, scan_error)?;
        }
    }
    Some(out)
}

const RESPONSES_VISIBLE_DELTA_EVENTS: &[&str] = &[
    "response.output_text.delta",
    "response.function_call_arguments.delta",
    "response.mcp_call_arguments.delta",
    "response.custom_tool_call_input.delta",
];

/// The only Responses content-part `text` fields the typed output guardrail
/// reads. Other part types can carry image, audio, file, or reasoning data.
const RESPONSES_VISIBLE_TEXT_PART_TYPES: &[&str] = &["output_text", "text", "input_text"];

/// Source-preserving counterpart to the typed Responses output scanner's
/// content-part walk. A missing or conflicting discriminator is opaque: a
/// media item can use any string-shaped field, so only a unique known text
/// part may cross the external guardrail boundary.
fn append_responses_visible_part_strings(
    out: &mut String,
    part: &serde_json::value::RawValue,
) -> Option<()> {
    let part_body = part.get().as_bytes();
    match raw_top_level_unique_type(part_body).as_deref() {
        Some(kind) if RESPONSES_VISIBLE_TEXT_PART_TYPES.contains(&kind) => {
            append_raw_top_level_strings(out, part_body, "text")?
        }
        Some(_) | None => {}
    }
    Some(())
}

fn append_responses_visible_content_strings(
    out: &mut String,
    content: &serde_json::value::RawValue,
) -> Option<()> {
    let value = content.get().trim_start();
    if value.starts_with('"') {
        return append_raw_string_value(out, content);
    }
    if !value.starts_with('[') {
        return Some(());
    }
    let Some(parts) = raw_array_items(content) else {
        return Some(());
    };
    for part in parts {
        append_responses_visible_part_strings(out, &part)?;
    }
    Some(())
}

/// Source-preserving counterpart to `responses::responses_output_text`.
/// Restrict the walk to client-visible message text and the tool payloads the
/// typed output guardrail already reads. A missing or conflicting item type
/// is opaque rather than a generic raw fallback: without a unique item kind,
/// `text`, `arguments`, and `input` could be an image/audio/file payload.
fn append_responses_output_item_strings(
    out: &mut String,
    item: &serde_json::value::RawValue,
) -> Option<()> {
    let item_body = item.get().as_bytes();
    match raw_top_level_unique_type(item_body).as_deref() {
        Some("reasoning") => {}
        Some("message") => {
            for content in raw_top_level_values(item_body, "content")? {
                append_responses_visible_content_strings(out, &content)?;
            }
        }
        Some("function_call" | "mcp_call") => {
            for key in ["name", "arguments"] {
                append_raw_top_level_strings(out, item_body, key)?;
            }
        }
        Some("custom_tool_call") => {
            for key in ["name", "input"] {
                append_raw_top_level_strings(out, item_body, key)?;
            }
        }
        Some(_) | None => {}
    }
    Some(())
}

fn append_responses_output_strings(out: &mut String, body: &[u8]) -> Option<()> {
    for output in raw_top_level_values(body, "output")? {
        let Some(items) = raw_array_items(&output) else {
            continue;
        };
        for item in items {
            append_responses_output_item_strings(out, &item)?;
        }
    }
    Some(())
}

fn decoded_responses_response_string_values(body: &[u8]) -> Option<String> {
    let mut out = String::new();
    append_responses_output_strings(&mut out, body)?;
    Some(out)
}

/// The response text a guardrail scans, per the route's protocol hint.
///
/// This deliberately reads raw source values rather than `Value`, retaining
/// duplicate visible text and tool carriers which the client receives
/// verbatim. Generated reasoning and opaque media remain out of scope.
fn response_guardrail_text_with_scan_error(
    protocol: PassthroughProtocol,
    body: &[u8],
    scan_error: &mut Option<crate::json_splice::SpliceError>,
) -> String {
    let raw = || String::from_utf8_lossy(body).into_owned();
    match protocol {
        PassthroughProtocol::Raw => decoded_json_string_values(body).unwrap_or_else(raw),
        PassthroughProtocol::OpenaiChat => {
            // A detected Chat response can carry opaque multimodal values.
            // Without a successful type-aware selection, relay it but do not
            // send a raw fallback to an external output guardrail.
            decoded_chat_response_string_values(body, scan_error).unwrap_or_default()
        }
        PassthroughProtocol::OpenaiCompletions => {
            decoded_non_model_json_string_values(body).unwrap_or_else(raw)
        }
        PassthroughProtocol::OpenaiResponses => {
            // Without a safely decoded Responses envelope, no discriminator
            // can establish that a string is visible text rather than opaque
            // image/audio/file data. Privacy wins over a raw fallback here.
            decoded_responses_response_string_values(body).unwrap_or_default()
        }
    }
}

fn response_guardrail_text(protocol: PassthroughProtocol, body: &[u8]) -> String {
    let mut ignored_scan_error = None;
    response_guardrail_text_with_scan_error(protocol, body, &mut ignored_scan_error)
}

/// Like [`response_guardrail_text`], but preserves scanner failures from the
/// exact typed source selectors that read a value for output inspection.
fn try_response_guardrail_text(
    protocol: PassthroughProtocol,
    body: &[u8],
) -> Result<String, crate::json_splice::SpliceError> {
    match protocol {
        PassthroughProtocol::Raw => crate::json_splice::collect_string_values(body),
        PassthroughProtocol::OpenaiCompletions => {
            let text = crate::json_splice::collect_string_values_where(body, |path| {
                !is_root_key(path, "model")
            })?;
            Ok(if text.is_empty() {
                response_guardrail_text(protocol, body)
            } else {
                text
            })
        }
        PassthroughProtocol::OpenaiChat => {
            let mut scan_error = None;
            let text = response_guardrail_text_with_scan_error(protocol, body, &mut scan_error);
            scan_error.map_or(Ok(text), Err)
        }
        PassthroughProtocol::OpenaiResponses => Ok(response_guardrail_text(protocol, body)),
    }
}

/// The typed visible-response extraction used for telemetry capture. It is
/// intentionally separate from the broader guardrail source scan above.
fn response_visible_text(protocol: PassthroughProtocol, body: &[u8]) -> String {
    let raw = || String::from_utf8_lossy(body).into_owned();
    if matches!(protocol, PassthroughProtocol::Raw) {
        return raw();
    }
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(body) else {
        return decoded_json_string_values(body).unwrap_or_else(raw);
    };
    if matches!(protocol, PassthroughProtocol::OpenaiResponses) {
        let joined = crate::responses::responses_output_text(&v);
        return if joined.is_empty() { raw() } else { joined };
    }
    if matches!(protocol, PassthroughProtocol::OpenaiChat) && v.get("choices").is_none() {
        let text = anthropic_message_output_text(&v);
        if !text.is_empty() {
            return text;
        }
    }
    let choices = v.get("choices").and_then(|c| c.as_array());
    let Some(choices) = choices else { return raw() };
    let texts: Vec<String> = choices
        .iter()
        .filter_map(|c| match protocol {
            PassthroughProtocol::OpenaiChat => {
                c.get("message").map(|m| message_scan_text(m, false))
            }
            PassthroughProtocol::OpenaiCompletions => {
                c.get("text").and_then(|t| t.as_str()).map(str::to_string)
            }
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

/// Captures preserve the existing typed visible-content contract; guardrail
/// scanning may include additional raw fields that are forwarded downstream.
fn response_capture_text(protocol: PassthroughProtocol, body: &[u8]) -> String {
    response_visible_text(protocol, body)
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
    cache_write_tokens: Option<u32>,
    reasoning_tokens: u32,
    cache_creation_tokens: u32,
    cache_read_tokens: u32,
    /// The upstream's own `total_tokens`, verbatim; `None` when a report
    /// carried none. Never a sum computed here — see
    /// `UsageStats::upstream_total_tokens`.
    upstream_total_tokens: Option<u32>,
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
        self.cache_write_tokens = self.cache_write_tokens.max(other.cache_write_tokens);
        self.reasoning_tokens = self.reasoning_tokens.max(other.reasoning_tokens);
        self.cache_creation_tokens = self.cache_creation_tokens.max(other.cache_creation_tokens);
        self.cache_read_tokens = self.cache_read_tokens.max(other.cache_read_tokens);
        // A total stands only while every merged report carried one.
        self.upstream_total_tokens = self
            .upstream_total_tokens
            .zip(other.upstream_total_tokens)
            .map(|(a, b)| a.max(b));
    }
}

/// Merge one usage report into an exchange's accumulated usage. The first
/// report is adopted whole rather than merged into zeros, so a total it
/// carried survives until a report without one arrives.
fn merge_usage(acc: &mut Option<PassthroughUsage>, report: PassthroughUsage) {
    match acc {
        Some(acc) => acc.merge(report),
        None => *acc = Some(report),
    }
}

/// `usage` figures from a buffered protocol-aware response body.
fn response_usage(
    protocol: PassthroughProtocol,
    raw_shape: Option<RawUsageShape>,
    body: &[u8],
) -> Option<PassthroughUsage> {
    if matches!(protocol, PassthroughProtocol::Raw) && raw_shape.is_none() {
        return None;
    }
    let v = serde_json::from_slice::<serde_json::Value>(body).ok()?;
    match raw_shape {
        Some(RawUsageShape::Rerank) => {
            crate::rerank::rerank_prompt_tokens(&v).map(|prompt_tokens| PassthroughUsage {
                prompt_tokens,
                ..PassthroughUsage::default()
            })
        }
        Some(RawUsageShape::DashscopeNative) => dashscope_native_usage(v.get("usage")?),
        None => usage_of(v.get("usage")?),
    }
}

/// Token counts from a DashScope native response's top-level `usage`.
///
/// Every dimension the generic reader knows (cache hit, reasoning, …) is
/// read by [`usage_of`]; only prompt and completion follow DashScope's own
/// arithmetic. The flat `image_tokens` means different things per service:
/// multimodal generation counts it inside `input_tokens`
/// (`{input_tokens: 79, image_tokens: 66, output_tokens: 14,
/// total_tokens: 93}`), while the multimodal embedding and rerank services
/// count it beside (`{input_tokens: 44, image_tokens: 64,
/// total_tokens: 108}`). So `total_tokens` less the completion is the
/// prompt whenever a total is reported; without one (only some embedding
/// models omit it, and those report images beside the text), prompt is
/// `input_tokens` (or `prompt_tokens`) plus the flat `image_tokens`. A
/// nested `input_tokens_details.image_tokens` is a breakdown already inside
/// `input_tokens` and is never added.
fn dashscope_native_usage(usage: &serde_json::Value) -> Option<PassthroughUsage> {
    let num = |k: &str| {
        usage
            .get(k)
            .and_then(serde_json::Value::as_u64)
            .map(|n| n.min(u32::MAX as u64) as u32)
    };
    let completion = num("output_tokens").or_else(|| num("completion_tokens"));
    let prompt = match num("total_tokens") {
        Some(total) => Some(total.saturating_sub(completion.unwrap_or(0))),
        None => {
            let input = num("input_tokens").or_else(|| num("prompt_tokens"));
            let image = num("image_tokens");
            (input.is_some() || image.is_some())
                .then(|| input.unwrap_or(0).saturating_add(image.unwrap_or(0)))
        }
    };
    let generic = usage_of(usage);
    if prompt.is_none() && completion.is_none() && generic.is_none() {
        return None;
    }
    Some(PassthroughUsage {
        prompt_tokens: prompt.unwrap_or(0),
        completion_tokens: completion.unwrap_or(0),
        upstream_total_tokens: num("total_tokens"),
        ..generic.unwrap_or_default()
    })
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
    let cache_write = nested("prompt_tokens_details", "cache_write_tokens")
        .or_else(|| nested("input_tokens_details", "cache_write_tokens"));
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
        cache_write,
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
        cache_write_tokens: cache_write,
        reasoning_tokens: reasoning.unwrap_or(0),
        cache_creation_tokens: cache_creation.unwrap_or(0),
        cache_read_tokens: cache_read.unwrap_or(0),
        upstream_total_tokens: flat(&["total_tokens"]),
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
    let base_path = Url::parse(base)
        .map(|url| url.path().to_owned())
        .unwrap_or_else(|_| base.to_owned());
    let base_tail = base_path
        .trim_end_matches('/')
        .rsplit('/')
        .next()
        .unwrap_or("");
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

/// Join a caller-controlled route remainder below the operator-configured
/// target URL. Parsing the candidate before the prefix check is deliberate:
/// the URL implementation canonicalises both literal and percent-encoded dot
/// segments, so the check observes the path reqwest will actually send.
fn join_target_url(base: &str, rest: &str, query: Option<&str>) -> Result<String, ProxyError> {
    if has_path_traversal_segment(rest) {
        return Err(ProxyError::InvalidRequest(
            "passthrough path is outside the configured target URL".into(),
        ));
    }

    let mut base_url = Url::parse(base).map_err(|_| {
        ProxyError::InvalidRequest("passthrough route has an invalid target URL".into())
    })?;
    // A configured query is part of the operator-owned target. Preserve it
    // and append a non-conflicting caller query after it, while keeping it
    // out of the URL string used for path joining. A caller must not be able
    // to override an operator-owned key through a last-value query parser.
    let base_query = base_url
        .query()
        .filter(|query| !query.is_empty())
        .map(str::to_owned);
    base_url.set_query(None);
    base_url.set_fragment(None);
    let base_for_join = base_url.as_str().trim_end_matches('/');
    let joined = if rest.is_empty() {
        base_for_join.to_string()
    } else {
        format!("{base_for_join}/{rest}")
    };
    let mut target = Url::parse(&joined).map_err(|_| {
        ProxyError::InvalidRequest("passthrough path is outside the configured target URL".into())
    })?;
    let inbound_query = query.filter(|query| !query.is_empty());
    if let (Some(base), Some(inbound)) = (base_query.as_deref(), inbound_query) {
        if query_keys_overlap(base, inbound) {
            return Err(ProxyError::InvalidRequest(
                "passthrough query conflicts with the configured target URL".into(),
            ));
        }
    }
    let merged_query = match (base_query.as_deref(), inbound_query) {
        (Some(base), Some(inbound)) => Some(format!("{base}&{inbound}")),
        (Some(base), _) => Some(base.to_owned()),
        (_, Some(inbound)) => Some(inbound.to_owned()),
        (None, None) => None,
    };
    target.set_query(merged_query.as_deref());

    if target.origin() != base_url.origin() || !path_is_within_base(target.path(), base_url.path())
    {
        return Err(ProxyError::InvalidRequest(
            "passthrough path is outside the configured target URL".into(),
        ));
    }
    Ok(target.into())
}

/// Query key comparison examines each bounded whole-query decoded form. A
/// backend may decode before splitting on `&` or `;`, so checking only keys
/// from the original form would miss `safe=1%26tenant%3Dcaller` becoming a
/// `tenant` key downstream. Some form parsers also canonicalize a key's
/// bracketed suffix and ASCII dot/space spelling, so compare that normalized
/// form at every decode level. Query sets here are tiny, so a simple vector
/// keeps the parser behavior explicit without adding a dependency.
fn query_keys_overlap(base: &str, inbound: &str) -> bool {
    let Some(base_keys) = query_keys_at_all_decode_levels(base) else {
        return true;
    };
    let Some(inbound_keys) = query_keys_at_all_decode_levels(inbound) else {
        return true;
    };
    inbound_keys
        .iter()
        .any(|key| base_keys.iter().any(|base_key| base_key == key))
}

fn query_keys_at_all_decode_levels(query: &str) -> Option<Vec<Vec<u8>>> {
    let mut decoded = query.as_bytes().to_vec();
    let mut keys = Vec::new();
    for _ in 0..=MAX_PERCENT_DECODE_PASSES {
        for field in decoded.split(|byte| matches!(*byte, b'&' | b';')) {
            for (key, _) in url::form_urlencoded::parse(field) {
                let key = normalize_form_query_key(&key);
                if !keys.iter().any(|existing| existing == &key) {
                    keys.push(key);
                }
            }
        }
        let next = percent_decode(&decoded).collect::<Vec<_>>();
        if next == decoded {
            return Some(keys);
        }
        decoded = next;
    }
    None
}

/// Match PHP-style form-key registration after form decoding: leading ASCII
/// spaces are ignored, a NUL or bracketed suffix terminates the base key, and
/// ASCII dots/spaces in that base key are aliases for underscores.
fn normalize_form_query_key(key: &str) -> Vec<u8> {
    key.as_bytes()
        .iter()
        .skip_while(|byte| **byte == b' ')
        .take_while(|byte| !matches!(**byte, b'\0' | b'['))
        .map(|byte| match *byte {
            b'.' | b' ' => b'_',
            byte => byte,
        })
        .collect()
}

/// A backend may decode percent escapes before routing. Decode a bounded
/// number of times, so nested encoding cannot turn a harmless-looking
/// segment into `..` downstream. Inputs that keep changing beyond the bound
/// are rejected rather than delegated to an upstream with unknown decoding.
const MAX_PERCENT_DECODE_PASSES: usize = 4;

fn has_path_traversal_segment(path: &str) -> bool {
    let mut decoded = path.as_bytes().to_vec();
    for _ in 0..=MAX_PERCENT_DECODE_PASSES {
        if decoded
            .split(|byte| matches!(*byte, b'/' | b'\\'))
            .any(|segment| {
                let path_part = segment
                    .split(|byte| *byte == b';')
                    .next()
                    .unwrap_or_default();
                path_part == b"." || path_part == b".."
            })
        {
            return true;
        }

        let next = percent_decode(&decoded).collect::<Vec<_>>();
        if next == decoded {
            return false;
        }
        decoded = next;
    }

    true
}

/// `candidate` must remain at `base` itself or below it on a path-segment
/// boundary. A root target intentionally permits every absolute path.
fn path_is_within_base(candidate: &str, base: &str) -> bool {
    let base = base.trim_end_matches('/');
    if base.is_empty() {
        return candidate.starts_with('/');
    }
    candidate == base
        || candidate
            .strip_prefix(base)
            .is_some_and(|remainder| remainder.starts_with('/'))
}

// ---------------------------------------------------------------------------
// Streaming relay
// ---------------------------------------------------------------------------

/// Default cap on bytes buffered while waiting for one SSE frame terminator.
/// A hold-back policy instead derives its splitter cap from its own raw-byte
/// limit, so an unterminated frame cannot outgrow the bytes it may hold.
const MAX_HELD_STREAM_BYTES: usize = 1024 * 1024;

/// Bound independently scanned output candidates even when an upstream never
/// emits its terminal item event. Two source branches (first/last) are kept
/// for each identity; supplemental values consume the same budget so one
/// frame cannot fan out into an unbounded number of guardrail calls.
const MAX_STREAM_GUARDRAIL_CHANNELS: usize = 64;

/// Empty epochs still occupy bookkeeping and eventually trigger a custom
/// guardrail scan, so cap them separately from their text candidates.
const MAX_STREAM_GUARDRAIL_EPOCHS: usize = 64;

/// Source identities are bookkeeping only, never output text. Bound them
/// separately so a few large provider item ids cannot dominate the relay's
/// memory while the content-channel cap still sees only 64 branches.
const MAX_STREAM_GUARDRAIL_SOURCE_ID_BYTES: usize = 256;

/// The gateway's shared frame splitter, with this relay's overflow policy:
/// an oversized unterminated run is handed on as a frame. Bytes after the
/// last complete frame stay buffered until more arrive; `take_rest` drains
/// them at end-of-stream.
struct SseFrameSplitter(aisix_gateway::sse::SseFrameSplitter);

struct SseFrame {
    bytes: Vec<u8>,
    /// The upstream did not terminate the frame before the configured
    /// hold-back raw-byte bound, so `bytes` cannot safely be decoded as an
    /// SSE payload.
    overflowed: bool,
}

impl SseFrameSplitter {
    fn with_max_frame_bytes(max_frame_bytes: usize) -> Self {
        Self(aisix_gateway::sse::SseFrameSplitter::new(max_frame_bytes))
    }

    fn push(&mut self, chunk: &[u8]) -> Vec<SseFrame> {
        self.0.push(chunk);
        let mut frames = Vec::new();
        loop {
            match self.0.next_frame() {
                Ok(Some(bytes)) => frames.push(SseFrame {
                    bytes,
                    overflowed: false,
                }),
                Ok(None) => break,
                // Frame-terminator starvation: hand the oversized run on as-is
                // rather than buffering without bound.
                Err(_) => {
                    frames.push(SseFrame {
                        bytes: self.0.take_rest(),
                        overflowed: true,
                    });
                    break;
                }
            }
        }
        frames
    }

    fn take_rest(&mut self) -> Vec<u8> {
        self.0.take_rest()
    }
}

/// `true` when the frame carries an `event:` line the server itself names
/// a usage report. The only evidence an opaque stream offers that a
/// token-shaped payload IS usage — see [`frame_delta`].
fn is_usage_labelled_frame(frame: &[u8]) -> bool {
    aisix_gateway::sse::lines(frame).any(|line| {
        aisix_gateway::sse::parse_field(&frame[line.start..line.end]).is_some_and(
            |(name, value)| {
                let value = value.trim_ascii();
                name == b"event"
                    && (value.eq_ignore_ascii_case(b"token_usage")
                        || value.eq_ignore_ascii_case(b"usage"))
            },
        )
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
/// The upstream failure an SSE frame reports in-band, read with the same
/// mappings the typed endpoints use for the protocol the route carries. An
/// opaque (`Raw`) stream has no error envelope the gateway could recognise.
fn frame_in_band_error(
    protocol: PassthroughProtocol,
    frame: &[u8],
) -> Option<aisix_gateway::BridgeError> {
    if matches!(protocol, PassthroughProtocol::Raw) {
        return None;
    }
    let payload = crate::redact::frame_payload(frame)?;
    let payload = payload.trim();
    let value = serde_json::from_str::<serde_json::Value>(payload).ok()?;
    match protocol {
        PassthroughProtocol::Raw => None,
        PassthroughProtocol::OpenaiResponses => crate::responses::responses_in_band_error(&value),
        // The chat envelope carries Anthropic Messages traffic too, whose
        // in-band failure is a `type: "error"` event.
        PassthroughProtocol::OpenaiChat | PassthroughProtocol::OpenaiCompletions => {
            if value.get("type").and_then(|t| t.as_str()) == Some("error") {
                if let Some(body) = value.get("error").and_then(|e| {
                    serde_json::from_value::<
                            aisix_provider_anthropic::wire::AnthropicStreamErrorBody,
                        >(e.clone())
                        .ok()
                }) {
                    return Some(
                        aisix_provider_anthropic::wire::stream_error_into_bridge_error(&body),
                    );
                }
            }
            aisix_gateway::capture_in_band_error(payload, aisix_gateway::UpstreamWire::OpenAI)
        }
    }
}

#[cfg(test)]
fn frame_delta(protocol: PassthroughProtocol, frame: &[u8]) -> (String, Option<PassthroughUsage>) {
    let (parts, usage) = frame_parts(protocol, frame);
    (parts.scan, usage)
}

/// Whether one unambiguous Anthropic stream event carries only generated
/// reasoning. `None` means the source shape was not safely inspectable, so
/// callers must preserve it rather than treating it as hidden.
fn hidden_chat_stream_reasoning_frame(body: &[u8]) -> Option<bool> {
    match raw_top_level_unique_type(body).as_deref() {
        Some("content_block_delta") => raw_top_level_items_have_only_types(
            body,
            "delta",
            &["thinking_delta", "signature_delta"],
        ),
        Some("content_block_start") => raw_top_level_items_have_only_types(
            body,
            "content_block",
            &["thinking", "redacted_thinking"],
        ),
        _ => Some(false),
    }
}

#[cfg(test)]
fn decoded_chat_frame_string_values(body: &[u8]) -> Option<String> {
    let mut out = String::new();
    for value in decoded_chat_frame_continuations(body)? {
        append_scan_text(&mut out, &value);
    }
    for value in decoded_chat_frame_supplemental_values(body)? {
        append_scan_text(&mut out, &value);
    }
    Some(out)
}

/// The typed stream extractor deliberately takes only text/tool delta events:
/// `.done`, output-item, and terminal response snapshots repeat those
/// carriers. Keep the raw source pass on that same boundary, both to avoid
/// duplicate external moderation and to keep any opaque terminal media out.
#[cfg(test)]
fn decoded_responses_frame_string_values(body: &[u8]) -> Option<String> {
    let mut out = String::new();
    if raw_top_level_has_only_types(body, RESPONSES_VISIBLE_DELTA_EVENTS) {
        append_raw_top_level_strings(&mut out, body, "delta")?;
    }
    Some(out)
}

fn decoded_chat_frame_continuations(body: &[u8]) -> Option<Vec<String>> {
    match raw_top_level_unique_type(body).as_deref() {
        Some("content_block_delta") => {
            let delta = raw_top_level_unique_object(body, "delta").ok()??;
            match raw_top_level_unique_type(delta.get().as_bytes()).as_deref() {
                Some("text_delta") => raw_top_level_string_values(delta.get().as_bytes(), "text"),
                Some("input_json_delta") => {
                    raw_top_level_string_values(delta.get().as_bytes(), "partial_json")
                }
                Some(_) | None => Some(Vec::new()),
            }
        }
        Some("content_block_start") => {
            let block = raw_top_level_unique_object(body, "content_block").ok()??;
            let block_body = block.get().as_bytes();
            match raw_top_level_unique_type(block_body).as_deref() {
                Some("text") => raw_top_level_string_values(block_body, "text"),
                Some("tool_use") => {
                    let mut out = Vec::new();
                    for input in raw_top_level_values(block_body, "input")? {
                        out.extend(
                            crate::json_splice::collect_string_values_where_vec(
                                input.get().as_bytes(),
                                |_| true,
                            )
                            .ok()?,
                        );
                    }
                    Some(out)
                }
                Some(_) | None => Some(Vec::new()),
            }
        }
        _ => {
            let choices = raw_top_level_unique_array(body, "choices").ok()??;
            let choices = raw_array_items(&choices)?;
            let mut out = Vec::new();
            for choice in choices {
                if !raw_is_object(&choice) {
                    return None;
                }
                let delta = match raw_top_level_unique_object(choice.get().as_bytes(), "delta") {
                    Ok(Some(delta)) => delta,
                    Ok(None) => continue,
                    Err(()) => return None,
                };
                let delta_body = delta.get().as_bytes();
                let mut content = raw_top_level_values(delta_body, "content")?;
                if content.len() > 1 {
                    return None;
                }
                if let Some(content) = content.pop() {
                    match content.get().trim_start().as_bytes().first() {
                        Some(b'"') => out.push(serde_json::from_str(content.get()).ok()?),
                        Some(b'[') => {
                            for part in raw_array_items(&content)? {
                                if !raw_is_object(&part) {
                                    return None;
                                }
                                let part_body = part.get().as_bytes();
                                if let Some(field) = raw_top_level_unique_type(part_body)
                                    .as_deref()
                                    .and_then(chat_visible_content_part_field)
                                {
                                    out.extend(raw_top_level_string_values(part_body, field)?);
                                }
                            }
                        }
                        Some(b'n') => {}
                        _ => return None,
                    }
                }
                out.extend(raw_top_level_string_values(delta_body, "refusal")?);
                let tool_calls = raw_top_level_unique_array(delta_body, "tool_calls").ok()?;
                if let Some(tool_calls) = tool_calls {
                    for tool_call in raw_array_items(&tool_calls)? {
                        if !raw_is_object(&tool_call) {
                            return None;
                        }
                        let tool_body = tool_call.get().as_bytes();
                        for (container, field) in chat_tool_continuation_fields(tool_body)? {
                            for payload in raw_top_level_values(tool_body, container)? {
                                if !raw_is_object(&payload) {
                                    return None;
                                }
                                out.extend(raw_top_level_string_values(
                                    payload.get().as_bytes(),
                                    field,
                                )?);
                            }
                        }
                    }
                }
                let function_call =
                    raw_top_level_unique_object(delta_body, "function_call").ok()?;
                if let Some(function_call) = function_call {
                    out.extend(raw_top_level_string_values(
                        function_call.get().as_bytes(),
                        "arguments",
                    )?);
                }
            }
            Some(out)
        }
    }
}

fn decoded_completions_frame_continuations(body: &[u8]) -> Option<Vec<String>> {
    use crate::json_splice::PathSeg;

    decoded_json_string_values_vec_where(body, |path| {
        matches!(
            path,
            [PathSeg::Key(choices), PathSeg::Index(_), PathSeg::Key(text)]
                if choices == "choices" && text == "text"
        )
    })
}

fn is_completions_continuation_path(path: &[crate::json_splice::PathSeg]) -> bool {
    use crate::json_splice::PathSeg;

    matches!(
        path,
        [PathSeg::Key(choices), PathSeg::Index(_), PathSeg::Key(text)]
            if choices == "choices" && text == "text"
    )
}

/// The raw source continuations preserve every occurrence of visible carrier
/// fields. Supplementary scan text therefore excludes those same paths: a
/// normal frame must not send one visible value to a guardrail as typed,
/// source, and supplementary text at once.
fn decoded_chat_frame_supplemental_values(body: &[u8]) -> Option<Vec<String>> {
    if raw_top_level_unique_type(body).as_deref() == Some("content_block_delta") {
        return Some(Vec::new());
    }
    if raw_top_level_unique_type(body).as_deref() == Some("content_block_start") {
        let block = raw_top_level_unique_object(body, "content_block").ok()??;
        return match raw_top_level_unique_type(block.get().as_bytes()).as_deref() {
            Some("tool_use") => raw_top_level_string_values(block.get().as_bytes(), "name"),
            Some(_) | None => Some(Vec::new()),
        };
    }

    let choices = raw_top_level_unique_array(body, "choices").ok()??;
    let choices = raw_array_items(&choices)?;
    let mut out = Vec::new();
    for choice in choices {
        if !raw_is_object(&choice) {
            return None;
        }
        let delta = match raw_top_level_unique_object(choice.get().as_bytes(), "delta") {
            Ok(Some(delta)) => delta,
            Ok(None) => continue,
            Err(()) => return None,
        };
        let tool_calls = raw_top_level_unique_array(delta.get().as_bytes(), "tool_calls").ok()?;
        if let Some(tool_calls) = tool_calls {
            for tool_call in raw_array_items(&tool_calls)? {
                if !raw_is_object(&tool_call) {
                    return None;
                }
                let tool_body = tool_call.get().as_bytes();
                for (container, _) in chat_tool_continuation_fields(tool_body)? {
                    for payload in raw_top_level_values(tool_body, container)? {
                        if !raw_is_object(&payload) {
                            return None;
                        }
                        out.extend(raw_top_level_string_values(
                            payload.get().as_bytes(),
                            "name",
                        )?);
                    }
                }
            }
        }
        let function_call =
            raw_top_level_unique_object(delta.get().as_bytes(), "function_call").ok()?;
        if let Some(function_call) = function_call {
            out.extend(raw_top_level_string_values(
                function_call.get().as_bytes(),
                "name",
            )?);
        }
    }
    Some(out)
}

fn decoded_completions_frame_supplemental_values(body: &[u8]) -> Option<Vec<String>> {
    decoded_json_string_values_vec_where(body, |path| {
        !is_root_key(path, "model") && !is_completions_continuation_path(path)
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct StreamContinuation {
    key: String,
    /// A semantic carrier family. A source form that has no durable carrier
    /// identity is never allowed to merge with a different member of this
    /// family on a later frame.
    family: String,
    identity: String,
    identity_is_ambiguous: bool,
    text: String,
}

enum SourceContinuations {
    Absent,
    Ready(Vec<StreamContinuation>),
    Unevaluable,
}

fn valid_stream_source_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= MAX_STREAM_GUARDRAIL_SOURCE_ID_BYTES
}

fn bounded_stream_source_id(id: String) -> Result<String, ()> {
    if !valid_stream_source_id(&id) {
        Err(())
    } else {
        Ok(id)
    }
}

fn append_source_branches(
    out: &mut Vec<StreamContinuation>,
    seen_keys: &mut std::collections::HashSet<String>,
    source_values: &mut Vec<String>,
    family: String,
    identity: String,
    identity_is_ambiguous: bool,
    values: Vec<String>,
) -> Result<(), ()> {
    let (first, last) = match values.as_slice() {
        [] => return Ok(()),
        [only] => (only, only),
        [first, last] => (first, last),
        _ => return Err(()),
    };
    source_values.extend(values.iter().cloned());
    for (branch, value) in [("first", first), ("last", last)] {
        let key = format!("{family}:{identity}:{branch}");
        if !seen_keys.insert(key.clone()) {
            return Err(());
        }
        out.push(StreamContinuation {
            key,
            family: family.clone(),
            identity: identity.clone(),
            identity_is_ambiguous,
            text: value.clone(),
        });
    }
    Ok(())
}

fn source_values_match_expected(mut source_values: Vec<String>, mut expected: Vec<String>) -> bool {
    // Object member order is not semantic. Keep duplicate counts, but do not
    // reject a valid source frame merely because a provider serialised its
    // tool fields before its text field.
    source_values.sort_unstable();
    expected.sort_unstable();
    source_values == expected
}

fn responses_source_continuations(payload: &[u8]) -> SourceContinuations {
    let kind = match raw_top_level_unique_string(payload, "type") {
        Ok(Some(kind)) if RESPONSES_VISIBLE_DELTA_EVENTS.contains(&kind.as_str()) => kind,
        Ok(Some(_)) | Ok(None) => return SourceContinuations::Absent,
        Err(()) => return SourceContinuations::Unevaluable,
    };
    let item_id = match raw_top_level_unique_string(payload, "item_id") {
        Ok(Some(item_id)) => match bounded_stream_source_id(item_id) {
            Ok(item_id) => item_id,
            Err(()) => return SourceContinuations::Unevaluable,
        },
        Ok(None) | Err(()) => return SourceContinuations::Unevaluable,
    };
    let output_index = match raw_top_level_unique_index(payload, "output_index") {
        Ok(Some(index)) => index.to_string(),
        Ok(None) | Err(()) => return SourceContinuations::Unevaluable,
    };
    let content_index = if kind == "response.output_text.delta" {
        match raw_top_level_unique_index(payload, "content_index") {
            Ok(Some(index)) => index.to_string(),
            Ok(None) | Err(()) => return SourceContinuations::Unevaluable,
        }
    } else {
        // Tool-argument deltas have no content part. Their stable identity is
        // the item plus output index and event kind; an incidental content
        // field must not create a second source channel.
        String::new()
    };
    let values = match raw_top_level_string_values(payload, "delta") {
        Some(values) => values,
        None => return SourceContinuations::Unevaluable,
    };
    if values.is_empty() {
        return SourceContinuations::Absent;
    }
    let mut out = Vec::new();
    let mut keys = std::collections::HashSet::new();
    let mut source_values = Vec::new();
    let family = format!("responses:{item_id:?}");
    let identity = format!("{kind:?}:{output_index}:{content_index}:delta");
    if append_source_branches(
        &mut out,
        &mut keys,
        &mut source_values,
        family,
        identity,
        false,
        values,
    )
    .is_err()
    {
        return SourceContinuations::Unevaluable;
    }
    SourceContinuations::Ready(out)
}

struct SourceBranchIdentity {
    family: String,
    identity: String,
    identity_is_ambiguous: bool,
}

fn append_raw_string_carrier(
    out: &mut Vec<StreamContinuation>,
    keys: &mut std::collections::HashSet<String>,
    source_values: &mut Vec<String>,
    source: SourceBranchIdentity,
    body: &[u8],
    key: &str,
) -> Result<(), ()> {
    append_source_branches(
        out,
        keys,
        source_values,
        source.family,
        source.identity,
        source.identity_is_ambiguous,
        raw_top_level_string_values(body, key).ok_or(())?,
    )
}

fn raw_part_identity(body: &[u8]) -> Result<String, ()> {
    // Content arrays can repeat a visible text or refusal field. Their numeric index
    // is the only canonical identity that remains stable when a provider
    // later adds an optional `id`; id-only arrays use the unevaluable policy
    // rather than silently switching source channels.
    raw_top_level_unique_index(body, "index")?
        .map(|index| format!("index:{index}"))
        .ok_or(())
}

fn anthropic_source_continuations(
    payload: &[u8],
    kind: &str,
    expected: Vec<String>,
) -> SourceContinuations {
    let index = match raw_top_level_unique_index(payload, "index") {
        Ok(Some(index)) => index.to_string(),
        _ => return SourceContinuations::Unevaluable,
    };
    let family = format!("anthropic:{index}");
    let mut out = Vec::new();
    let mut keys = std::collections::HashSet::new();
    let mut source_values = Vec::new();
    let result = match kind {
        "content_block_delta" => {
            let delta = match raw_top_level_unique_object(payload, "delta") {
                Ok(Some(delta)) => delta,
                _ => return SourceContinuations::Unevaluable,
            };
            let delta_body = delta.get().as_bytes();
            match raw_top_level_unique_type(delta_body).as_deref() {
                Some("text_delta") => append_raw_string_carrier(
                    &mut out,
                    &mut keys,
                    &mut source_values,
                    SourceBranchIdentity {
                        family: family.clone(),
                        identity: "text".to_owned(),
                        identity_is_ambiguous: false,
                    },
                    delta_body,
                    "text",
                ),
                Some("input_json_delta") => append_raw_string_carrier(
                    &mut out,
                    &mut keys,
                    &mut source_values,
                    SourceBranchIdentity {
                        family: family.clone(),
                        identity: "partial_json".to_owned(),
                        identity_is_ambiguous: false,
                    },
                    delta_body,
                    "partial_json",
                ),
                Some(_) | None => Ok(()),
            }
        }
        "content_block_start" => {
            let block = match raw_top_level_unique_object(payload, "content_block") {
                Ok(Some(block)) => block,
                _ => return SourceContinuations::Unevaluable,
            };
            let block_body = block.get().as_bytes();
            match raw_top_level_unique_type(block_body).as_deref() {
                Some("text") => append_raw_string_carrier(
                    &mut out,
                    &mut keys,
                    &mut source_values,
                    SourceBranchIdentity {
                        family: family.clone(),
                        identity: "text".to_owned(),
                        identity_is_ambiguous: false,
                    },
                    block_body,
                    "text",
                ),
                Some("tool_use") => {
                    let mut inputs = match raw_top_level_values(block_body, "input") {
                        Some(inputs) => inputs,
                        None => return SourceContinuations::Unevaluable,
                    };
                    let Some(input) = inputs.pop() else {
                        return SourceContinuations::Absent;
                    };
                    if !inputs.is_empty() {
                        return SourceContinuations::Unevaluable;
                    }
                    if input.get().trim_start().starts_with('"') {
                        append_raw_string_carrier(
                            &mut out,
                            &mut keys,
                            &mut source_values,
                            SourceBranchIdentity {
                                family: family.clone(),
                                identity: "input".to_owned(),
                                identity_is_ambiguous: false,
                            },
                            block_body,
                            "input",
                        )
                    } else {
                        // A nested tool input can have many source leaves but no
                        // durable leaf identity on this envelope. An empty object is
                        // harmless; a visible value must use the bounded policy.
                        let text =
                            match crate::json_splice::collect_string_values(input.get().as_bytes())
                            {
                                Ok(text) => text,
                                Err(_) => return SourceContinuations::Unevaluable,
                            };
                        if text.is_empty() {
                            Ok(())
                        } else {
                            Err(())
                        }
                    }
                }
                Some(_) | None => Ok(()),
            }
        }
        _ => return SourceContinuations::Absent,
    };
    if result.is_err() || !source_values_match_expected(source_values, expected) {
        SourceContinuations::Unevaluable
    } else if out.is_empty() {
        SourceContinuations::Absent
    } else {
        SourceContinuations::Ready(out)
    }
}

fn chat_choice_source_continuations(payload: &[u8]) -> SourceContinuations {
    let Some(expected) = decoded_chat_frame_continuations(payload) else {
        return SourceContinuations::Absent;
    };
    let kind = match raw_top_level_unique_string(payload, "type") {
        Ok(Some(kind)) => Some(kind),
        Ok(None) => None,
        Err(()) => return SourceContinuations::Unevaluable,
    };
    if let Some(kind @ ("content_block_delta" | "content_block_start")) = kind.as_deref() {
        return anthropic_source_continuations(payload, kind, expected);
    }
    let choices = match raw_top_level_unique_array(payload, "choices") {
        Ok(Some(choices)) => match raw_array_items(&choices) {
            Some(choices) => choices,
            None => return SourceContinuations::Unevaluable,
        },
        Ok(None) | Err(()) => return SourceContinuations::Unevaluable,
    };
    let mut out = Vec::new();
    let mut keys = std::collections::HashSet::new();
    let mut source_values = Vec::new();
    let mut choice_indexes = std::collections::HashSet::new();
    for choice in choices {
        if !raw_is_object(&choice) {
            return SourceContinuations::Unevaluable;
        }
        let choice_body = choice.get().as_bytes();
        let choice_index = match raw_top_level_unique_index(choice_body, "index") {
            Ok(Some(index)) if choice_indexes.insert(index) => index.to_string(),
            _ => return SourceContinuations::Unevaluable,
        };
        let delta = match raw_top_level_unique_object(choice_body, "delta") {
            Ok(Some(delta)) => delta,
            Ok(None) => continue,
            Err(()) => return SourceContinuations::Unevaluable,
        };
        let delta_body = delta.get().as_bytes();
        let mut content = match raw_top_level_values(delta_body, "content") {
            Some(content) => content,
            None => return SourceContinuations::Unevaluable,
        };
        if content.len() > 1 {
            return SourceContinuations::Unevaluable;
        }
        if let Some(content) = content.pop() {
            match content.get().trim_start().as_bytes().first() {
                Some(b'"') => {
                    let value = match serde_json::from_str::<String>(content.get()) {
                        Ok(value) => value,
                        Err(_) => return SourceContinuations::Unevaluable,
                    };
                    if append_source_branches(
                        &mut out,
                        &mut keys,
                        &mut source_values,
                        format!("chat:{choice_index}:content"),
                        "scalar".to_owned(),
                        true,
                        vec![value],
                    )
                    .is_err()
                    {
                        return SourceContinuations::Unevaluable;
                    }
                }
                Some(b'[') => {
                    let Some(parts) = raw_array_items(&content) else {
                        return SourceContinuations::Unevaluable;
                    };
                    let mut part_ids = std::collections::HashSet::new();
                    for part in parts {
                        if !raw_is_object(&part) {
                            return SourceContinuations::Unevaluable;
                        }
                        let part_body = part.get().as_bytes();
                        let Some(field) = raw_top_level_unique_type(part_body)
                            .as_deref()
                            .and_then(chat_visible_content_part_field)
                        else {
                            continue;
                        };
                        let visible_values = match raw_top_level_string_values(part_body, field) {
                            Some(visible_values) => visible_values,
                            None => return SourceContinuations::Unevaluable,
                        };
                        if visible_values.is_empty() {
                            continue;
                        }
                        let part_id = match raw_part_identity(part_body) {
                            Ok(part_id) if part_ids.insert(part_id.clone()) => part_id,
                            _ => return SourceContinuations::Unevaluable,
                        };
                        if append_source_branches(
                            &mut out,
                            &mut keys,
                            &mut source_values,
                            format!("chat:{choice_index}:content"),
                            format!("part:{part_id}:{field}"),
                            false,
                            visible_values,
                        )
                        .is_err()
                        {
                            return SourceContinuations::Unevaluable;
                        }
                    }
                }
                Some(b'n') => {}
                _ => return SourceContinuations::Unevaluable,
            }
        }
        if append_raw_string_carrier(
            &mut out,
            &mut keys,
            &mut source_values,
            SourceBranchIdentity {
                family: format!("chat:{choice_index}:refusal"),
                identity: "refusal".to_owned(),
                identity_is_ambiguous: false,
            },
            delta_body,
            "refusal",
        )
        .is_err()
        {
            return SourceContinuations::Unevaluable;
        }
        let tool_calls = match raw_top_level_unique_array(delta_body, "tool_calls") {
            Ok(tool_calls) => tool_calls,
            Err(()) => return SourceContinuations::Unevaluable,
        };
        if let Some(tool_calls) = tool_calls {
            let Some(tool_calls) = raw_array_items(&tool_calls) else {
                return SourceContinuations::Unevaluable;
            };
            let mut tool_indexes = std::collections::HashSet::new();
            for tool_call in tool_calls {
                if !raw_is_object(&tool_call) {
                    return SourceContinuations::Unevaluable;
                }
                let tool_body = tool_call.get().as_bytes();
                let tool_index = match raw_top_level_unique_index(tool_body, "index") {
                    Ok(Some(index)) if tool_indexes.insert(index) => index.to_string(),
                    _ => return SourceContinuations::Unevaluable,
                };
                for (container, field) in match chat_tool_continuation_fields(tool_body) {
                    Some(fields) => fields,
                    None => return SourceContinuations::Unevaluable,
                } {
                    let nested = match raw_top_level_unique_object(tool_body, container) {
                        Ok(nested) => nested,
                        Err(()) => return SourceContinuations::Unevaluable,
                    };
                    if let Some(nested) = nested {
                        if append_raw_string_carrier(
                            &mut out,
                            &mut keys,
                            &mut source_values,
                            SourceBranchIdentity {
                                family: format!("chat:{choice_index}:tool:{tool_index}"),
                                identity: format!("{container}:{field}"),
                                identity_is_ambiguous: false,
                            },
                            nested.get().as_bytes(),
                            field,
                        )
                        .is_err()
                        {
                            return SourceContinuations::Unevaluable;
                        }
                    }
                }
            }
        }
        let function_call = match raw_top_level_unique_object(delta_body, "function_call") {
            Ok(function_call) => function_call,
            Err(()) => return SourceContinuations::Unevaluable,
        };
        if let Some(function_call) = function_call {
            if append_raw_string_carrier(
                &mut out,
                &mut keys,
                &mut source_values,
                SourceBranchIdentity {
                    family: format!("chat:{choice_index}:legacy_function"),
                    identity: "arguments".to_owned(),
                    identity_is_ambiguous: false,
                },
                function_call.get().as_bytes(),
                "arguments",
            )
            .is_err()
            {
                return SourceContinuations::Unevaluable;
            }
        }
    }
    if source_values_match_expected(source_values, expected) {
        SourceContinuations::Ready(out)
    } else {
        SourceContinuations::Unevaluable
    }
}

fn completions_source_continuations(payload: &[u8]) -> SourceContinuations {
    let Some(expected) = decoded_completions_frame_continuations(payload) else {
        return SourceContinuations::Absent;
    };
    let choices = match raw_top_level_unique_array(payload, "choices") {
        Ok(Some(choices)) => match raw_array_items(&choices) {
            Some(choices) => choices,
            None => return SourceContinuations::Unevaluable,
        },
        Ok(None) | Err(()) => return SourceContinuations::Unevaluable,
    };
    let mut out = Vec::new();
    let mut keys = std::collections::HashSet::new();
    let mut source_values = Vec::new();
    let mut choice_indexes = std::collections::HashSet::new();
    for choice in choices {
        if !raw_is_object(&choice) {
            return SourceContinuations::Unevaluable;
        }
        let choice_body = choice.get().as_bytes();
        let choice_index = match raw_top_level_unique_index(choice_body, "index") {
            Ok(Some(index)) if choice_indexes.insert(index) => index.to_string(),
            _ => return SourceContinuations::Unevaluable,
        };
        if append_raw_string_carrier(
            &mut out,
            &mut keys,
            &mut source_values,
            SourceBranchIdentity {
                family: format!("completions:{choice_index}"),
                identity: "text".to_owned(),
                identity_is_ambiguous: false,
            },
            choice_body,
            "text",
        )
        .is_err()
        {
            return SourceContinuations::Unevaluable;
        }
    }
    if source_values_match_expected(source_values, expected) {
        SourceContinuations::Ready(out)
    } else {
        SourceContinuations::Unevaluable
    }
}

fn stream_source_continuations(
    protocol: PassthroughProtocol,
    payload: &[u8],
) -> SourceContinuations {
    let payload = payload.trim_ascii();
    if payload.is_empty() || payload == b"[DONE]" {
        return SourceContinuations::Absent;
    }
    match protocol {
        // An opaque protocol offers no carrier identity inside a JSON object
        // or array. A bare JSON string is the one unambiguous source carrier;
        // every broader Raw shape follows the configured unevaluable policy.
        PassthroughProtocol::Raw => match serde_json::from_slice::<String>(payload) {
            Ok(text) if !text.is_empty() => SourceContinuations::Ready(vec![StreamContinuation {
                key: "raw:payload:first".to_owned(),
                family: "raw".to_owned(),
                identity: "payload".to_owned(),
                identity_is_ambiguous: false,
                text,
            }]),
            Ok(_) => SourceContinuations::Absent,
            Err(_) => SourceContinuations::Unevaluable,
        },
        PassthroughProtocol::OpenaiChat => chat_choice_source_continuations(payload),
        PassthroughProtocol::OpenaiCompletions => completions_source_continuations(payload),
        PassthroughProtocol::OpenaiResponses => responses_source_continuations(payload),
    }
}

fn responses_terminal_continuation_prefix(payload: &[u8]) -> Result<Option<String>, ()> {
    let payload = payload.trim_ascii();
    // `[DONE]` terminates an SSE stream but is not a JSON Responses event.
    // Treat it like the source-continuation path does: no carrier closes and
    // no malformed payload is introduced. Other malformed payloads still
    // reach the fail-closed path below.
    if payload.is_empty() || payload == b"[DONE]" {
        return Ok(None);
    }
    let kind = match raw_top_level_unique_string(payload, "type") {
        Ok(Some(kind)) => kind,
        Ok(None) => return Ok(None),
        Err(()) => return Err(()),
    };
    if kind != "response.output_item.done" {
        return Ok(None);
    }
    let top_level_id = raw_top_level_unique_string(payload, "item_id")?;
    let nested_id = match raw_top_level_unique_object(payload, "item")? {
        Some(item) => raw_top_level_unique_string(item.get().as_bytes(), "id")?,
        None => None,
    };
    if top_level_id
        .as_ref()
        .is_some_and(|item_id| !valid_stream_source_id(item_id))
        || nested_id
            .as_ref()
            .is_some_and(|item_id| !valid_stream_source_id(item_id))
    {
        return Err(());
    }
    let item_id = match (top_level_id, nested_id) {
        (Some(top_level_id), Some(nested_id)) if top_level_id == nested_id => Some(top_level_id),
        (Some(top_level_id), None) => Some(top_level_id),
        (None, Some(nested_id)) => Some(nested_id),
        (None, None) => None,
        (Some(_), Some(_)) => return Err(()),
    };
    Ok(item_id.map(|item_id| format!("responses:{item_id:?}:")))
}

fn frame_guardrail_supplemental_values(
    protocol: PassthroughProtocol,
    frame: &[u8],
    has_source_continuations: bool,
    has_typed_continuation: bool,
) -> Vec<String> {
    // A typed continuation without source proof becomes unevaluable; do not
    // add a second, generic supplemental scan for that same frame.
    if has_typed_continuation && !has_source_continuations {
        return Vec::new();
    }
    if !has_source_continuations {
        return frame_guardrail_values(protocol, frame);
    }

    let Some(payload) = crate::redact::frame_payload(frame) else {
        return Vec::new();
    };
    let payload = payload.trim();
    if payload.is_empty() || payload == "[DONE]" {
        return Vec::new();
    }
    match protocol {
        // The raw source continuation is the complete decoded payload.
        PassthroughProtocol::Raw => Vec::new(),
        PassthroughProtocol::OpenaiChat => {
            decoded_chat_frame_supplemental_values(payload.as_bytes()).unwrap_or_default()
        }
        PassthroughProtocol::OpenaiCompletions => {
            decoded_completions_frame_supplemental_values(payload.as_bytes()).unwrap_or_default()
        }
        // Responses source continuations exist only for the explicitly safe
        // text/tool delta events, whose sole output carrier is `delta`.
        PassthroughProtocol::OpenaiResponses => Vec::new(),
    }
}

fn decoded_chat_frame_values(body: &[u8]) -> Option<Vec<String>> {
    decoded_chat_frame_supplemental_values(body)
}

fn decoded_responses_frame_values(body: &[u8]) -> Option<Vec<String>> {
    raw_top_level_has_only_types(body, RESPONSES_VISIBLE_DELTA_EVENTS)
        .then(|| raw_top_level_string_values(body, "delta"))
        .flatten()
        .or_else(|| Some(Vec::new()))
}

fn frame_guardrail_values(protocol: PassthroughProtocol, frame: &[u8]) -> Vec<String> {
    let Some(payload) = crate::redact::frame_payload(frame) else {
        return Vec::new();
    };
    let payload = payload.trim();
    if payload.is_empty() || payload == "[DONE]" {
        return Vec::new();
    }
    let raw = || vec![payload.to_string()];
    match protocol {
        PassthroughProtocol::Raw => {
            decoded_json_string_values_vec_where(payload.as_bytes(), |_| true).unwrap_or_else(raw)
        }
        PassthroughProtocol::OpenaiChat => {
            decoded_chat_frame_values(payload.as_bytes()).unwrap_or_default()
        }
        PassthroughProtocol::OpenaiCompletions => {
            decoded_json_string_values_vec_where(payload.as_bytes(), |path| {
                !is_root_key(path, "model")
            })
            .unwrap_or_else(raw)
        }
        PassthroughProtocol::OpenaiResponses => {
            decoded_responses_frame_values(payload.as_bytes()).unwrap_or_default()
        }
    }
}

/// Guardrail-only text for a streamed frame. Capture and hold-back retain
/// their typed visible-content extraction in [`frame_parts`], while this
/// source-preserving pass retains duplicate selected carriers. Chat and
/// Responses media fields remain opaque even when forwarded verbatim.
#[cfg(test)]
fn frame_guardrail_text(protocol: PassthroughProtocol, frame: &[u8]) -> String {
    let Some(payload) = crate::redact::frame_payload(frame) else {
        return String::new();
    };
    let payload = payload.trim();
    if payload.is_empty() || payload == "[DONE]" {
        return String::new();
    }
    let raw = || payload.to_string();
    match protocol {
        PassthroughProtocol::Raw => {
            decoded_json_string_values(payload.as_bytes()).unwrap_or_else(raw)
        }
        PassthroughProtocol::OpenaiChat => {
            // A detected Chat frame can carry opaque multimodal values.
            // Without a successful type-aware selection, relay it but do not
            // send a raw fallback to an external output guardrail.
            decoded_chat_frame_string_values(payload.as_bytes()).unwrap_or_default()
        }
        PassthroughProtocol::OpenaiCompletions => {
            decoded_non_model_json_string_values(payload.as_bytes()).unwrap_or_else(raw)
        }
        PassthroughProtocol::OpenaiResponses => {
            // As with buffered Responses output, only a successful
            // source-aware selection may cross the external guardrail
            // boundary. A malformed frame stays forwarded but opaque.
            decoded_responses_frame_string_values(payload.as_bytes()).unwrap_or_default()
        }
    }
}

/// The independent source-identified channels scanned for one stream frame.
/// `supplemental` preserves selected non-carrier source values. Keeping them
/// separate prevents frame metadata from interrupting a sensitive literal
/// split across output deltas.
struct StreamGuardrailText {
    continuations: Vec<StreamContinuation>,
    /// Values with no continuation carrier. They are checked separately so
    /// unrelated JSON fields cannot become one regex/remote-model segment.
    supplemental: Vec<String>,
    unevaluable: bool,
    /// Responses item closures wait until their already-buffered text has
    /// passed a guardrail scan. Removing them on the terminal event would
    /// erase a short delta before an end-of-stream or full-buffer scan.
    closed_prefixes: Vec<String>,
}

fn stream_guardrail_text(
    protocol: PassthroughProtocol,
    frame: &[u8],
    continuation: String,
) -> StreamGuardrailText {
    let payload = crate::redact::frame_payload(frame);
    let hidden_reasoning = matches!(protocol, PassthroughProtocol::OpenaiChat)
        && payload.as_ref().is_some_and(|payload| {
            hidden_chat_stream_reasoning_frame(payload.trim().as_bytes()) == Some(true)
        });
    let responses_visible_delta = matches!(protocol, PassthroughProtocol::OpenaiResponses)
        && payload.as_ref().is_some_and(|payload| {
            raw_top_level_has_only_types(payload.trim().as_bytes(), RESPONSES_VISIBLE_DELTA_EVENTS)
        });
    let typed_continuation = if hidden_reasoning
        || (matches!(protocol, PassthroughProtocol::OpenaiResponses) && !responses_visible_delta)
    {
        String::new()
    } else if matches!(protocol, PassthroughProtocol::OpenaiChat) {
        // `frame_parts` retains text-shaped fields for capture and hold-back,
        // including ones on opaque multimodal parts. Its raw scan text must
        // not make such a part an external-guardrail candidate or prevent a
        // sibling, type-allowed text/tool value from being scanned.
        payload
            .as_ref()
            .and_then(|payload| {
                decoded_chat_frame_continuations(payload.trim().as_bytes())
                    .map(|values| values.join("\n"))
            })
            .unwrap_or(continuation)
    } else {
        continuation
    };
    let has_typed_continuation = !typed_continuation.is_empty();
    let source = if hidden_reasoning {
        SourceContinuations::Absent
    } else {
        payload
            .as_ref()
            .map_or(SourceContinuations::Absent, |payload| {
                stream_source_continuations(protocol, payload.trim().as_bytes())
            })
    };
    let (closed_prefixes, terminal_unevaluable) =
        if matches!(protocol, PassthroughProtocol::OpenaiResponses) {
            match payload
                .as_ref()
                .map(|payload| responses_terminal_continuation_prefix(payload.trim().as_bytes()))
            {
                Some(Ok(Some(prefix))) => (vec![prefix], false),
                Some(Ok(None)) | None => (Vec::new(), false),
                Some(Err(())) => (Vec::new(), true),
            }
        } else {
            (Vec::new(), false)
        };
    let (continuations, has_source_continuations, source_unevaluable) = match source {
        SourceContinuations::Ready(source) => {
            let has_source_continuations = !source.is_empty();
            (source, has_source_continuations, false)
        }
        // A typed visible delta without a source carrier proof cannot be
        // continued safely into the next frame. Do not create a generic
        // positional fallback channel for it.
        SourceContinuations::Absent if has_typed_continuation => (Vec::new(), false, true),
        SourceContinuations::Absent => (Vec::new(), false, false),
        // Do not assign an ordinal to an unkeyable source channel. The
        // caller applies the configured fail-open/fail-closed policy.
        SourceContinuations::Unevaluable => (Vec::new(), false, true),
    };
    let unevaluable = source_unevaluable || terminal_unevaluable;
    StreamGuardrailText {
        continuations,
        supplemental: if unevaluable {
            Vec::new()
        } else {
            frame_guardrail_supplemental_values(
                protocol,
                frame,
                has_source_continuations,
                has_typed_continuation,
            )
        },
        unevaluable,
        closed_prefixes,
    }
}

fn append_stream_guardrail_text(
    continuations: &mut Vec<StreamContinuation>,
    _continuation_tails: &mut Vec<StreamContinuation>,
    supplemental: &mut Vec<String>,
    closed_prefixes: &mut Vec<String>,
    text: &StreamGuardrailText,
) {
    for prefix in &text.closed_prefixes {
        // Do not retain a terminal with no outstanding carrier: otherwise a
        // stream of empty item.done events could grow the close set without
        // contributing any text to the channel cap.
        if continuations
            .iter()
            .any(|continuation| continuation.key.starts_with(prefix))
            && !closed_prefixes.contains(prefix)
        {
            closed_prefixes.push(prefix.clone());
        }
    }
    for incoming in &text.continuations {
        if let Some(existing) = continuations
            .iter_mut()
            .find(|continuation| continuation.key == incoming.key)
        {
            existing.text.push_str(&incoming.text);
        } else {
            continuations.push(incoming.clone());
        }
    }
    supplemental.extend(
        text.supplemental
            .iter()
            .filter(|value| !value.is_empty())
            .cloned(),
    );
}

fn stream_continuation_would_exceed_cap(
    continuations: &[StreamContinuation],
    supplemental: &[String],
    _closed_prefixes: &[String],
    text: &StreamGuardrailText,
) -> bool {
    let mut keys: std::collections::HashSet<_> = continuations
        .iter()
        .map(|continuation| continuation.key.as_str())
        .collect();
    for continuation in &text.continuations {
        keys.insert(continuation.key.as_str());
    }
    let mut supplemental_candidates = supplemental
        .iter()
        .filter(|value| !value.is_empty())
        .map(String::as_str)
        .collect::<std::collections::HashSet<_>>();
    supplemental_candidates.extend(
        text.supplemental
            .iter()
            .filter(|value| !value.is_empty())
            .map(String::as_str),
    );
    keys.len() + supplemental_candidates.len() > MAX_STREAM_GUARDRAIL_CHANNELS
}

fn stream_continuation_identity_conflicts(
    continuations: &[StreamContinuation],
    closed_prefixes: &[String],
    text: &StreamGuardrailText,
) -> bool {
    let is_closed = |key: &str| {
        closed_prefixes
            .iter()
            .chain(text.closed_prefixes.iter())
            .any(|prefix| key.starts_with(prefix))
    };
    if text
        .continuations
        .iter()
        .any(|continuation| is_closed(&continuation.key))
    {
        return true;
    }
    let mut all = continuations
        .iter()
        .filter(|continuation| !is_closed(&continuation.key))
        .collect::<Vec<_>>();
    for incoming in &text.continuations {
        if all.iter().any(|existing| {
            existing.family == incoming.family
                && existing.identity != incoming.identity
                && (existing.identity_is_ambiguous || incoming.identity_is_ambiguous)
        }) {
            return true;
        }
        all.push(incoming);
    }
    false
}

fn retire_scanned_stream_continuations(
    continuations: &mut Vec<StreamContinuation>,
    continuation_tails: &mut Vec<StreamContinuation>,
    closed_prefixes: &mut Vec<String>,
) {
    continuations.retain(|continuation| {
        !closed_prefixes
            .iter()
            .any(|prefix| continuation.key.starts_with(prefix))
    });
    continuation_tails.retain(|continuation| {
        !closed_prefixes
            .iter()
            .any(|prefix| continuation.key.starts_with(prefix))
    });
    closed_prefixes.clear();
}

/// A fail-open frame with no provable source identity is a hard boundary:
/// nothing before it may be joined with a later keyed delta. The caller still
/// relays the frame, but starts a fresh guardrail scan epoch afterward.
fn reset_stream_guardrail_epoch(
    continuations: &mut Vec<StreamContinuation>,
    continuation_tails: &mut Vec<StreamContinuation>,
    supplemental: &mut Vec<String>,
    closed_prefixes: &mut Vec<String>,
) {
    continuations.clear();
    continuation_tails.clear();
    supplemental.clear();
    closed_prefixes.clear();
}

/// Preserve the completed epoch for its own end-of-stream scan, then make an
/// unevaluable frame a hard continuity boundary for all following carriers.
/// Keeping epochs separate both retains monitor observations and prevents a
/// literal from joining across the unknown frame.
fn seal_stream_guardrail_epoch(
    sealed_epochs: &mut Vec<Vec<String>>,
    queued_candidates: &mut usize,
    continuations: &mut Vec<StreamContinuation>,
    continuation_tails: &mut Vec<StreamContinuation>,
    supplemental: &mut Vec<String>,
    closed_prefixes: &mut Vec<String>,
) -> bool {
    let candidates = stream_guardrail_scan_text(continuation_tails, continuations, supplemental);
    let queued = candidates.is_empty()
        || try_queue_stream_guardrail_epoch(sealed_epochs, queued_candidates, candidates);
    reset_stream_guardrail_epoch(
        continuations,
        continuation_tails,
        supplemental,
        closed_prefixes,
    );
    queued
}

/// End-of-stream guardrails must keep epochs separate after an unevaluable
/// frame, but untrusted streams cannot queue arbitrarily many independent
/// external scans. An exhausted live fail-open stream records its bypass and
/// discards later candidates instead.
fn try_queue_stream_guardrail_epoch(
    sealed_epochs: &mut Vec<Vec<String>>,
    queued_candidates: &mut usize,
    candidates: Vec<String>,
) -> bool {
    if sealed_epochs.len() >= MAX_STREAM_GUARDRAIL_EPOCHS {
        return false;
    }
    let Some(total) = queued_candidates.checked_add(candidates.len()) else {
        return false;
    };
    if total > MAX_STREAM_GUARDRAIL_CHANNELS {
        return false;
    }
    *queued_candidates = total;
    sealed_epochs.push(candidates);
    true
}

fn stream_guardrail_scan_text(
    continuation_tails: &[StreamContinuation],
    continuations: &[StreamContinuation],
    supplemental: &[String],
) -> Vec<String> {
    let mut text = Vec::new();
    let mut scanned = std::collections::HashSet::new();
    for continuation in continuations {
        let tail = continuation_tails
            .iter()
            .find(|candidate| candidate.key == continuation.key)
            .map(|candidate| candidate.text.as_str())
            .unwrap_or_default();
        let candidate = format!("{tail}{}", continuation.text);
        if !candidate.is_empty() && scanned.insert(candidate.clone()) {
            text.push(candidate);
        }
    }
    for value in supplemental {
        if !value.is_empty() && scanned.insert(value.clone()) {
            text.push(value.clone());
        }
    }
    text
}

/// One frame's generated content, split by [`crate::held_content::Parts`]
/// into what the output guardrails scan and what only counts toward the
/// hold-back cap, plus any usage it reports. The scan and the cap read the
/// same extraction, so they cannot disagree about a frame (#513).
fn frame_parts(
    protocol: PassthroughProtocol,
    frame: &[u8],
) -> (crate::held_content::Parts, Option<PassthroughUsage>) {
    let usage_labelled =
        matches!(protocol, PassthroughProtocol::Raw) && is_usage_labelled_frame(frame);
    let mut parts = crate::held_content::Parts::default();
    let mut usage: Option<PassthroughUsage> = None;
    let mut merge = |found: PassthroughUsage| {
        merge_usage(&mut usage, found);
    };
    // This typed extraction reads and parses one complete payload per frame: a payload spread over several
    // `data:` lines is one document joined with `\n`, so parsing each line
    // independently produced N unparseable fragments — no usage read, and
    // on a `Raw` stream the JSON source text pushed into the guardrail
    // scan instead of the values (#1100). `frame_payload` also strips the
    // per-line `\r` a CRLF-framed upstream leaves behind, and returns
    // `None` for a comment-only frame (`: OPENROUTER PROCESSING`).
    'payload: {
        let Some(payload) = crate::redact::frame_payload(frame) else {
            break 'payload;
        };
        let payload = payload.trim();
        if payload.is_empty() || payload == "[DONE]" {
            break 'payload;
        }
        if matches!(protocol, PassthroughProtocol::Raw) {
            // Raw payloads have no typed content envelope to extract.
            // Scan them with the iterative value walker before touching
            // serde_json::Value: its default recursion limit would otherwise
            // turn a valid deeply nested escaped string into raw source text
            // and let it bypass an output guardrail.
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) {
                // An explicit `usage` object is self-describing even for an
                // opaque stream. A server-labelled event additionally
                // permits the flat agent-backend usage shape below.
                if let Some(u) = v.get("usage").and_then(usage_of) {
                    merge(u);
                }
                if usage_labelled {
                    if let Some(u) = usage_of(&v) {
                        merge(u);
                    }
                }
            }
            parts = crate::held_content::Parts {
                scan: decoded_json_string_values(payload.as_bytes())
                    .unwrap_or_else(|| payload.to_string()),
                reasoning: 0,
            };
            break 'payload;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) else {
            // Unparseable joined payload — a non-conformant upstream that
            // put two independent JSON documents on two `data:` lines, say.
            // The frame is still FORWARDED, so scanning nothing here is a
            // way past an output block rule. Fall back to the raw payload
            // text on every protocol, not just `Raw`: over-scanning can only
            // produce a false positive, while under-scanning a frame the
            // client receives is the bypass. (Per-line parsing used to catch
            // the two-document case incidentally; this covers it and every
            // other shape that does not parse.)
            parts.scan.push_str(
                &decoded_json_string_values(payload.as_bytes())
                    .unwrap_or_else(|| payload.to_string()),
            );
            break 'payload;
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
        parts = match protocol {
            PassthroughProtocol::Raw => unreachable!("handled before typed frame parsing"),
            // The chat envelope also carries Anthropic Messages streams; the
            // two event shapes are disjoint, so reading both is exact.
            PassthroughProtocol::OpenaiChat => {
                let mut p = crate::held_content::chat_chunk_parts(&v);
                let a = crate::held_content::anthropic_event_parts(&v);
                p.scan.push_str(&a.scan);
                p.reasoning += a.reasoning;
                p
            }
            PassthroughProtocol::OpenaiCompletions => {
                crate::held_content::completions_chunk_parts(&v)
            }
            PassthroughProtocol::OpenaiResponses => crate::held_content::responses_event_parts(&v),
        };
    }
    (parts, usage)
}

/// The text persisted for a streamed response. Raw passthrough capture keeps
/// the provider's JSON source representation even though the guardrail scans
/// decoded string values from that same frame.
fn frame_capture_text(protocol: PassthroughProtocol, frame: &[u8], scan: &str) -> String {
    if !matches!(protocol, PassthroughProtocol::Raw) {
        return scan.to_string();
    }
    crate::redact::frame_payload(frame)
        .map(|payload| payload.trim().to_owned())
        .filter(|payload| !payload.is_empty() && payload != "[DONE]")
        .unwrap_or_else(|| scan.to_string())
}

/// The SSE error frame appended when an output guardrail blocks mid-relay,
/// in the protocol of the stream it ends: the frame `/v1/messages` emits for
/// the same refusal on an Anthropic Messages stream, the one
/// `/v1/chat/completions` emits on every other.
fn guardrail_error_frame(
    anthropic: bool,
    guardrail_name: Option<&str>,
    unavailable: Option<&str>,
) -> Bytes {
    if anthropic {
        return Bytes::from(crate::messages::guardrail_block_frame(
            guardrail_name,
            unavailable,
        ));
    }
    Bytes::from(format!(
        "event: error\ndata: {}\n\n",
        crate::error::guardrail_block_frame_payload(guardrail_name, unavailable)
    ))
}

/// Whether a relayed chat-envelope frame is an Anthropic Messages event
/// (`Some(true)`) or an OpenAI chat chunk (`Some(false)`). The two event
/// shapes are disjoint; a frame that is neither (a comment, `[DONE]`, an
/// in-band error) decides nothing.
fn anthropic_stream_frame(frame: &[u8]) -> Option<bool> {
    let payload = crate::redact::frame_payload(frame)?;
    let v: serde_json::Value = serde_json::from_str(payload.trim()).ok()?;
    if v.get("choices").is_some() {
        return Some(false);
    }
    match v.get("type").and_then(serde_json::Value::as_str) {
        Some(
            "message_start"
            | "message_delta"
            | "message_stop"
            | "content_block_start"
            | "content_block_delta"
            | "content_block_stop"
            | "ping",
        ) => Some(true),
        _ => None,
    }
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
    stream_hold: aisix_ratelimit::StreamConcurrencyGuard,
    stream_read_timeout: Option<Duration>,
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
    let splitter_cap = policy
        .hold_cap()
        .map(|(cap, _)| cap.saturating_mul(crate::held_content::RAW_HOLD_FACTOR))
        .unwrap_or(MAX_HELD_STREAM_BYTES);

    let stream = async_stream::stream! {
        // The rate limiter's reservation becomes an owned hold at the handoff
        // from handler to body. It drops only when this body completes or the
        // client cancels it, rather than when the response headers are built.
        let _stream_hold = stream_hold;
        let read_timeout = crate::stream_timeout::ReadTimeoutSignal::default();
        let mut upstream = Box::pin(crate::stream_timeout::with_read_timeout_bytes_signalled(
            upstream_resp.bytes_stream(),
            stream_read_timeout,
            read_timeout.clone(),
        ));
        let mut splitter = SseFrameSplitter::with_max_frame_bytes(splitter_cap);
        // Held-back frames (Window / BufferFull) not yet released.
        let mut pending: Vec<Bytes> = Vec::new();
        let mut pending_held = crate::held_content::HeldBytes::default();
        // What Window and BufferFull hold (#513): content, which `max_buffer_bytes`
        // caps (the SSE framing is not counted), and the raw frame bytes it
        // bounds too.
        let mut held_content = crate::held_content::HeldBuffer::default();
        // Each source-identified semantic delta stays contiguous across
        // frames. Supplementary values remain individual scan candidates so
        // unrelated fields cannot form one guardrail input.
        let mut continuation_bufs: Vec<StreamContinuation> = Vec::new();
        let mut supplemental_buf: Vec<String> = Vec::new();
        // Overlap carried between Window scans, one per source channel.
        let mut continuation_tails: Vec<StreamContinuation> = Vec::new();
        // Item-done frames close a Responses carrier, but its buffered text
        // remains until a successful scan has covered it.
        let mut closed_continuation_prefixes: Vec<String> = Vec::new();
        // Each unevaluable live frame seals the previous source epoch. Those
        // epochs still need their own terminal scan, but must not concatenate
        // with text that follows the unkeyable frame.
        let mut sealed_guardrail_epochs: Vec<Vec<String>> = Vec::new();
        let mut queued_guardrail_candidates = 0;
        let mut scan_budget_exhausted = false;
        // Degrades a hold-back policy to live forwarding after a fail-open cap hit.
        let mut fail_opened = false;
        let mut blocked = false;
        // The chat envelope also carries Anthropic Messages streams; the
        // first frame that says which one decides the refusal frame's shape.
        let mut anthropic: Option<bool> = None;

        'outer: loop {
            let chunk = match upstream.next().await {
                Some(Ok(c)) => c,
                Some(Err(err)) => {
                    // The response head is already on the wire, so there is
                    // no status left to carry the failure — record it on the
                    // event instead of ending as a silent success.
                    let bridge = crate::dispatch::reqwest_error_to_bridge(&err, telemetry.started);
                    telemetry.record_failure(&bridge);
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
                // Both completed and splitter-overflowed frames take the
                // same raw-cap preflight below.
                let _overflowed = frame.overflowed;
                let frame = frame.bytes;
                if !fail_opened {
                    if let Some((max_buffer_bytes, on_exceeded_fail_open)) = policy.hold_cap() {
                        // Apply the raw-byte bound before parsing every frame.
                        // The splitter marks an unterminated frame as overflowed,
                        // but a complete frame can be oversized as well.
                        if held_content.would_exceed_after(0, frame.len(), max_buffer_bytes) {
                            if on_exceeded_fail_open {
                                fail_opened = true;
                                chain.record_bypass(crate::error::TAG_OUTPUT_BUFFER_EXCEEDED);
                                for pending_frame in pending.drain(..) {
                                    telemetry.mark_first_delivery();
                                    yield Ok(pending_frame);
                                }
                                pending_held.clear();
                                held_content = crate::held_content::HeldBuffer::default();
                                telemetry.mark_first_delivery();
                                yield Ok(Bytes::from(frame));
                                continue;
                            }

                            tracing::warn!(
                                route = %route_name,
                                "passthrough-route stream exceeded the guardrail buffer cap (fail-closed)",
                            );
                            blocked = true;
                            chain.record_output_buffer_exceeded();
                            pending.clear();
                            pending_held.clear();
                            yield Ok(guardrail_error_frame(anthropic.unwrap_or(false), None, Some(crate::error::TAG_OUTPUT_BUFFER_EXCEEDED)));
                            break 'outer;
                        }
                    }
                }
                if anthropic.is_none() && matches!(protocol, PassthroughProtocol::OpenaiChat) {
                    anthropic = anthropic_stream_frame(&frame);
                }
                if let Some(err) = frame_in_band_error(protocol, &frame) {
                    telemetry.record_failure(&err);
                }
                let (parts, usage) = frame_parts(protocol, &frame);
                let held = parts.held();
                let delta = parts.scan;
                if let Some(u) = usage {
                    merge_usage(&mut telemetry.usage, u);
                }
                if capture_cap.is_some() {
                    push_capped(
                        &mut telemetry.response_text,
                        &frame_capture_text(protocol, &frame, &delta),
                        capture_cap,
                    );
                }
                let guardrail_text = (!chain.is_empty() && !scan_budget_exhausted && !fail_opened)
                    .then(|| stream_guardrail_text(protocol, &frame, delta.clone()));
                let unevaluable_output = guardrail_text.as_ref().is_some_and(|text| {
                    text.unevaluable
                        || stream_continuation_would_exceed_cap(
                            &continuation_bufs,
                            &supplemental_buf,
                            &closed_continuation_prefixes,
                            text,
                        )
                        || stream_continuation_identity_conflicts(
                            &continuation_bufs,
                            &closed_continuation_prefixes,
                            text,
                        )
                });
                if unevaluable_output {
                    // A holding policy has already promised not to release a
                    // frame until it scans clean. `fail_open` can bypass an
                    // unevaluable live stream, but it cannot release the
                    // held prefix (or this frame) without a scan.
                    let must_refuse = (policy.holds_back() && !fail_opened)
                        || aisix_guardrails::Guardrail::refuses_unevaluable_output(&chain);
                    if must_refuse {
                        tracing::warn!(
                            guardrail_hook = "output",
                            route = %route_name,
                            "cannot preserve passthrough stream source continuity for guardrails; blocking",
                        );
                        pending.clear();
                        pending_held.clear();
                        blocked = true;
                        yield Ok(guardrail_error_frame(
                            anthropic.unwrap_or(false),
                            None,
                            Some(crate::error::TAG_UNSCANNABLE_BODY),
                        ));
                        break 'outer;
                    }
                    chain.record_unevaluable_output_bypass(crate::error::TAG_UNSCANNABLE_BODY);
                    if !seal_stream_guardrail_epoch(
                        &mut sealed_guardrail_epochs,
                        &mut queued_guardrail_candidates,
                        &mut continuation_bufs,
                        &mut continuation_tails,
                        &mut supplemental_buf,
                        &mut closed_continuation_prefixes,
                    ) {
                        scan_budget_exhausted = true;
                    }
                }
                let frame = Bytes::from(frame);
                match &policy {
                    _ if fail_opened => {
                        telemetry.mark_first_delivery();
                        yield Ok::<_, std::convert::Infallible>(frame);
                    }
                    StreamOutputPolicy::EndOfStreamCheck => {
                        if let Some(text) = guardrail_text.as_ref().filter(|_| !unevaluable_output) {
                            append_stream_guardrail_text(
                                &mut continuation_bufs,
                                &mut continuation_tails,
                                &mut supplemental_buf,
                                &mut closed_continuation_prefixes,
                                text,
                            );
                        }
                        telemetry.mark_first_delivery();
                        yield Ok(frame);
                    }
                    StreamOutputPolicy::Window {
                        size_chars,
                        overlap_chars,
                        max_buffer_bytes,
                        on_exceeded_fail_open,
                    } => {
                        if let Some(text) = guardrail_text.as_ref().filter(|_| !unevaluable_output) {
                            append_stream_guardrail_text(
                                &mut continuation_bufs,
                                &mut continuation_tails,
                                &mut supplemental_buf,
                                &mut closed_continuation_prefixes,
                                text,
                            );
                        }
                        held_content.hold(held, frame.len());
                        pending_held.add(frame.len());
                        pending.push(frame);
                        if held_content.exceeds(*max_buffer_bytes) {
                            if *on_exceeded_fail_open {
                                fail_opened = true;
                                chain.record_bypass(crate::error::TAG_OUTPUT_BUFFER_EXCEEDED);
                                for f in pending.drain(..) {
                                    telemetry.mark_first_delivery();
                                    yield Ok(f);
                                }
                                pending_held.clear();
                                held_content = crate::held_content::HeldBuffer::default();
                            } else {
                                tracing::warn!(
                                    route = %route_name,
                                    "passthrough-route stream exceeded the guardrail buffer cap (fail-closed)",
                                );
                                blocked = true;
                                chain.record_output_buffer_exceeded();
                                pending.clear();
                                pending_held.clear();
                                yield Ok(guardrail_error_frame(anthropic.unwrap_or(false), None, Some(crate::error::TAG_OUTPUT_BUFFER_EXCEEDED)));
                                break 'outer;
                            }
                        } else if continuation_bufs
                            .iter()
                            .any(|continuation| continuation.text.chars().count() >= *size_chars)
                            || supplemental_buf
                                .iter()
                                .map(|value| value.chars().count())
                                .sum::<usize>()
                                >= *size_chars
                        {
                            let candidates = stream_guardrail_scan_text(
                                &continuation_tails,
                                &continuation_bufs,
                                &supplemental_buf,
                            );
                            match scan_output_candidates(
                                &chain,
                                &route_name,
                                &candidates,
                                &mut telemetry,
                            )
                            .await
                            {
                                GuardrailVerdict::Block {
                                    reason,
                                    guardrail_name,
                                    unavailable,
                                } => {
                                    tracing::warn!(
                                        guardrail_hook = "output",
                                        route = %route_name,
                                        reason = %reason,
                                        "guardrail blocked passthrough-route stream (window)",
                                    );
                                    blocked = true;
                                    yield Ok(guardrail_error_frame(anthropic.unwrap_or(false), guardrail_name.as_deref(), unavailable.as_deref()));
                                    break 'outer;
                                }
                                _ => {
                                    for f in pending.drain(..) {
                                        telemetry.mark_first_delivery();
                                        yield Ok(f);
                                    }
                                    pending_held.clear();
                                    held_content = crate::held_content::HeldBuffer::default();
                                    for continuation in &mut continuation_bufs {
                                        let tail = continuation_tails
                                            .iter()
                                            .find(|tail| tail.key == continuation.key)
                                            .map(|tail| tail.text.as_str())
                                            .unwrap_or_default();
                                        let combined = format!("{tail}{}", continuation.text);
                                        let next_tail = tail_chars(&combined, *overlap_chars);
                                        if let Some(tail) = continuation_tails
                                            .iter_mut()
                                            .find(|tail| tail.key == continuation.key)
                                        {
                                            tail.text = next_tail;
                                        } else {
                                            continuation_tails.push(StreamContinuation {
                                                key: continuation.key.clone(),
                                                family: continuation.family.clone(),
                                                identity: continuation.identity.clone(),
                                                identity_is_ambiguous: continuation
                                                    .identity_is_ambiguous,
                                                text: next_tail,
                                            });
                                        }
                                        continuation.text.clear();
                                    }
                                    supplemental_buf.clear();
                                    retire_scanned_stream_continuations(
                                        &mut continuation_bufs,
                                        &mut continuation_tails,
                                        &mut closed_continuation_prefixes,
                                    );
                                }
                            }
                        }
                    }
                    StreamOutputPolicy::BufferFull { max_buffer_bytes, on_exceeded_fail_open } => {
                        if let Some(text) = guardrail_text.as_ref().filter(|_| !unevaluable_output) {
                            append_stream_guardrail_text(
                                &mut continuation_bufs,
                                &mut continuation_tails,
                                &mut supplemental_buf,
                                &mut closed_continuation_prefixes,
                                text,
                            );
                        }
                        held_content.hold(held, frame.len());
                        pending_held.add(frame.len());
                        pending.push(frame);
                        if held_content.exceeds(*max_buffer_bytes) {
                            if *on_exceeded_fail_open {
                                fail_opened = true;
                                chain.record_bypass(crate::error::TAG_OUTPUT_BUFFER_EXCEEDED);
                                for f in pending.drain(..) {
                                    telemetry.mark_first_delivery();
                                    yield Ok(f);
                                }
                                pending_held.clear();
                                held_content = crate::held_content::HeldBuffer::default();
                            } else {
                                tracing::warn!(
                                    route = %route_name,
                                    "passthrough-route stream exceeded the guardrail buffer cap (fail-closed)",
                                );
                                blocked = true;
                                chain.record_output_buffer_exceeded();
                                yield Ok(guardrail_error_frame(anthropic.unwrap_or(false), None, Some(crate::error::TAG_OUTPUT_BUFFER_EXCEEDED)));
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }

        if let Some(err) = read_timeout.fired() {
            telemetry.record_failure(&err);
            tracing::warn!(
                route = %route_name,
                error = %telemetry.error_message,
                "passthrough-route upstream stream timed out mid-relay",
            );
        }

        if !blocked {
            // Trailing bytes with no frame terminator, plus the final scan
            // of whatever the policy has not cleared yet.
            let rest = splitter.take_rest();
            if !rest.is_empty() {
                // The raw half of the hold-back cap is knowable without
                // decoding the unterminated tail. Decide it before a parser
                // can turn malformed JSON into an unrelated unscannable-body
                // refusal or capture it into telemetry.
                let tail_raw_cap_hit = match &policy {
                    StreamOutputPolicy::Window {
                        max_buffer_bytes,
                        on_exceeded_fail_open,
                        ..
                    }
                    | StreamOutputPolicy::BufferFull {
                        max_buffer_bytes,
                        on_exceeded_fail_open,
                    } if !fail_opened => held_content
                        .would_exceed_after(0, rest.len(), *max_buffer_bytes)
                        .then_some(*on_exceeded_fail_open),
                    _ => None,
                };
                match tail_raw_cap_hit {
                    Some(false) => {
                        tracing::warn!(
                            route = %route_name,
                            "passthrough-route stream exceeded the guardrail buffer cap (fail-closed)",
                        );
                        chain.record_output_buffer_exceeded();
                        pending.clear();
                        pending_held.clear();
                        yield Ok(guardrail_error_frame(anthropic.unwrap_or(false), None, Some(crate::error::TAG_OUTPUT_BUFFER_EXCEEDED)));
                        telemetry.guardrail_blocked = true;
                        telemetry.stream_reached_end = true;
                        telemetry.emit();
                        return;
                    }
                    Some(true) => {
                        fail_opened = true;
                        chain.record_bypass(crate::error::TAG_OUTPUT_BUFFER_EXCEEDED);
                        for frame in pending.drain(..) {
                            telemetry.mark_first_delivery();
                            yield Ok(frame);
                        }
                        pending_held.clear();
                        telemetry.mark_first_delivery();
                        yield Ok(Bytes::from(rest));
                    }
                    None => {
                if anthropic.is_none() && matches!(protocol, PassthroughProtocol::OpenaiChat) {
                    anthropic = anthropic_stream_frame(&rest);
                }
                let (parts, usage) = frame_parts(protocol, &rest);
                let held = parts.held();
                let delta = parts.scan;
                if let Some(u) = usage {
                    merge_usage(&mut telemetry.usage, u);
                }
                if capture_cap.is_some() {
                    push_capped(
                        &mut telemetry.response_text,
                        &frame_capture_text(protocol, &rest, &delta),
                        capture_cap,
                    );
                }

                // The raw cap was checked above before decoding. The regular
                // held-frame path below additionally applies the decoded
                // content cap to this tail.
                let tail_cap_hit = match &policy {
                    StreamOutputPolicy::Window {
                        max_buffer_bytes,
                        on_exceeded_fail_open,
                        ..
                    }
                    | StreamOutputPolicy::BufferFull {
                        max_buffer_bytes,
                        on_exceeded_fail_open,
                    } if !fail_opened => held_content
                        .would_exceed_after(held, rest.len(), *max_buffer_bytes)
                        .then_some(*on_exceeded_fail_open),
                    _ => None,
                };
                match tail_cap_hit {
                    Some(false) => {
                        tracing::warn!(
                            route = %route_name,
                            "passthrough-route stream exceeded the guardrail buffer cap (fail-closed)",
                        );
                        chain.record_output_buffer_exceeded();
                        pending.clear();
                        pending_held.clear();
                        yield Ok(guardrail_error_frame(anthropic.unwrap_or(false), None, Some(crate::error::TAG_OUTPUT_BUFFER_EXCEEDED)));
                        telemetry.guardrail_blocked = true;
                        telemetry.stream_reached_end = true;
                        telemetry.emit();
                        return;
                    }
                    Some(true) => {
                        fail_opened = true;
                        chain.record_bypass(crate::error::TAG_OUTPUT_BUFFER_EXCEEDED);
                        for frame in pending.drain(..) {
                            telemetry.mark_first_delivery();
                            yield Ok(frame);
                        }
                        pending_held.clear();
                        telemetry.mark_first_delivery();
                        yield Ok(Bytes::from(rest));
                    }
                    None => {
                        let guardrail_text =
                            (!chain.is_empty() && !scan_budget_exhausted && !fail_opened)
                                .then(|| stream_guardrail_text(protocol, &rest, delta.clone()));
                        let unevaluable_output = guardrail_text.as_ref().is_some_and(|text| {
                    text.unevaluable
                        || stream_continuation_would_exceed_cap(
                            &continuation_bufs,
                            &supplemental_buf,
                            &closed_continuation_prefixes,
                            text,
                        )
                        || stream_continuation_identity_conflicts(
                            &continuation_bufs,
                            &closed_continuation_prefixes,
                            text,
                        )
                });
                if unevaluable_output {
                    // See the matching frame path above: a holding policy
                    // must not release its pending prefix unscanned just
                    // because this terminal fragment is unevaluable.
                    let must_refuse = (policy.holds_back() && !fail_opened)
                        || aisix_guardrails::Guardrail::refuses_unevaluable_output(&chain);
                    if must_refuse {
                        tracing::warn!(
                            guardrail_hook = "output",
                            route = %route_name,
                            "cannot preserve passthrough stream source continuity for guardrails; blocking",
                        );
                        pending.clear();
                        pending_held.clear();
                        yield Ok(guardrail_error_frame(
                            anthropic.unwrap_or(false),
                            None,
                            Some(crate::error::TAG_UNSCANNABLE_BODY),
                        ));
                        telemetry.guardrail_blocked = true;
                        telemetry.stream_reached_end = true;
                        telemetry.emit();
                        return;
                    }
                    chain.record_unevaluable_output_bypass(crate::error::TAG_UNSCANNABLE_BODY);
                    if !seal_stream_guardrail_epoch(
                        &mut sealed_guardrail_epochs,
                        &mut queued_guardrail_candidates,
                        &mut continuation_bufs,
                        &mut continuation_tails,
                        &mut supplemental_buf,
                        &mut closed_continuation_prefixes,
                    ) {
                        scan_budget_exhausted = true;
                    }
                }
                if let Some(text) = guardrail_text.as_ref().filter(|_| !unevaluable_output) {
                    append_stream_guardrail_text(
                        &mut continuation_bufs,
                        &mut continuation_tails,
                        &mut supplemental_buf,
                        &mut closed_continuation_prefixes,
                        text,
                    );
                }
                let rest = Bytes::from(rest);
                // The tail is held like any frame, under the same cap.
                let tripped = match &policy {
                    StreamOutputPolicy::Window {
                        max_buffer_bytes,
                        on_exceeded_fail_open,
                        ..
                    }
                    | StreamOutputPolicy::BufferFull {
                        max_buffer_bytes,
                        on_exceeded_fail_open,
                    } if !fail_opened => {
                        held_content.hold(held, rest.len());
                        held_content
                            .exceeds(*max_buffer_bytes)
                            .then_some(*on_exceeded_fail_open)
                    }
                    _ => None,
                };
                match tripped {
                    Some(false) => {
                        tracing::warn!(
                            route = %route_name,
                            "passthrough-route stream exceeded the guardrail buffer cap (fail-closed)",
                        );
                        chain.record_output_buffer_exceeded();
                        pending.clear();
                        pending_held.clear();
                        yield Ok(guardrail_error_frame(anthropic.unwrap_or(false), None, Some(crate::error::TAG_OUTPUT_BUFFER_EXCEEDED)));
                        telemetry.guardrail_blocked = true;
                        telemetry.stream_reached_end = true;
                        telemetry.emit();
                        return;
                    }
                    Some(true) => {
                        fail_opened = true;
                        chain.record_bypass(crate::error::TAG_OUTPUT_BUFFER_EXCEEDED);
                        for f in pending.drain(..) {
                            telemetry.mark_first_delivery();
                            yield Ok(f);
                        }
                        pending_held.clear();
                        telemetry.mark_first_delivery();
                        yield Ok(rest);
                    }
                    None if policy.holds_back() && !fail_opened => {
                        pending_held.add(rest.len());
                        pending.push(rest)
                    }
                    None => {
                        telemetry.mark_first_delivery();
                        yield Ok(rest);
                    }
                }
                    }
                }
                }
                }
            }
            if !fail_opened {
                if !scan_budget_exhausted {
                    let candidates = stream_guardrail_scan_text(
                        &continuation_tails,
                        &continuation_bufs,
                        &supplemental_buf,
                    );
                    if (!candidates.is_empty() || sealed_guardrail_epochs.is_empty())
                        && !try_queue_stream_guardrail_epoch(
                            &mut sealed_guardrail_epochs,
                            &mut queued_guardrail_candidates,
                            candidates,
                        )
                    {
                        chain.record_unevaluable_output_bypass(crate::error::TAG_UNSCANNABLE_BODY);
                    }
                }
                for candidates in sealed_guardrail_epochs {
                    if !chain.is_empty() {
                        if let GuardrailVerdict::Block {
                            reason,
                            guardrail_name,
                            unavailable,
                        } = scan_output_candidates(&chain, &route_name, &candidates, &mut telemetry).await
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
                            pending_held.clear();
                            yield Ok(guardrail_error_frame(anthropic.unwrap_or(false), guardrail_name.as_deref(), unavailable.as_deref()));
                            telemetry.guardrail_blocked = true;
                            telemetry.stream_reached_end = true;
                            telemetry.emit();
                            return;
                        }
                    }
                }
            }
            for f in pending.drain(..) {
                telemetry.mark_first_delivery();
                yield Ok(f);
            }
            pending_held.clear();
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
    let (verdict, hits) = chain.check_output_unmaskable_observed(&synth).await;
    telemetry.monitor_hits.extend(hits);
    verdict
}

/// Scan each independently sourced candidate without letting a fail-open
/// result for one candidate skip a later candidate that another rule blocks.
async fn scan_output_candidates(
    chain: &aisix_guardrails::GuardrailChain,
    route_name: &str,
    candidates: &[String],
    telemetry: &mut RouteTelemetry,
) -> aisix_guardrails::GuardrailVerdict {
    if candidates.is_empty() {
        return scan_output(chain, route_name, "", telemetry).await;
    }
    for candidate in candidates {
        if let verdict @ aisix_guardrails::GuardrailVerdict::Block { .. } =
            scan_output(chain, route_name, candidate, telemetry).await
        {
            return verdict;
        }
    }
    aisix_guardrails::GuardrailVerdict::Allow
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
    /// Org member the authenticating key belongs to (AISIX-Cloud#1389),
    /// and that member's display name for the `user_name` metric label
    /// (AISIX-Cloud#1455). Both `None` for a key bound to no member —
    /// including the anonymous route key, which belongs to the route
    /// rather than to a person.
    user_id: Option<String>,
    user_name: Option<String>,
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
    usage: Option<PassthroughUsage>,
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
    /// The status that same failure gets before the response head
    /// ([`aisix_gateway::BridgeError::http_status`]). The emit records it in
    /// place of the upstream's `200`: the caller's response line cannot
    /// change any more, but the record of what happened can.
    failure_status: Option<u16>,
    monitor_hits: Vec<aisix_core::GuardrailMonitorHit>,
    /// The request's ENFORCE-mode audit handle (AISIX-Cloud#1330). Held
    /// rather than snapshotted at construction: this struct's emit runs
    /// from the relay's `Drop`, long after the output hook has recorded
    /// whatever it masked.
    audit: crate::usage_attr::GuardrailAudit,
    captured_prompt: Option<String>,
    content_cap: Option<usize>,
    response_text: String,
    guardrail_blocked: bool,
    emitted: bool,
}

impl RouteTelemetry {
    /// Record the upstream failure that ended a streamed relay after its
    /// head went out. The first one is the cause; later ones do not replace
    /// it.
    fn record_failure(&mut self, err: &aisix_gateway::BridgeError) {
        if self.failure_status.is_some() {
            return;
        }
        let failure = crate::attempt::StreamFailure::from_bridge(err);
        self.error_class = failure.error_class.to_string();
        self.error_message = failure.error_message;
        self.failure_status = Some(failure.status);
    }

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
        // One an upstream failure ended records that failure's status
        // instead, unless a guardrail refused it.
        if self.streaming {
            match self.failure_status.filter(|_| !self.guardrail_blocked) {
                Some(status) => self.status = status,
                None if !self.stream_reached_end => self.status = crate::CLIENT_CLOSED_REQUEST,
                None => {}
            }
        }
        let elapsed = self.started.elapsed();
        let snapshot = self.state.snapshot.load();
        let usage = self.usage.unwrap_or_default();

        emit_access_log(
            &self.method,
            &self.path,
            &self.route_name,
            &self.api_key_id,
            self.status,
            // Same rule as the typed streaming endpoints, and the same
            // figure this emit puts on the usage event below: a streamed
            // relay reports the wait to its first relayed frame, a buffered
            // one the whole response. A relay that delivered nothing waited
            // the whole request for nothing, which is what `elapsed` says.
            if self.streaming {
                self.downstream_first_ms
                    .map(|ms| Duration::from_millis(u64::from(ms)))
                    .unwrap_or(elapsed)
            } else {
                elapsed
            },
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
            occurred_at: aisix_obs::UsageEvent::occurred_at_now(),
            api_key_id: self.api_key_id.clone(),
            status_code: self.status,
            requested_model: self.requested_model.clone(),
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            cached_prompt_tokens: usage.cached_prompt_tokens,
            cache_write_tokens: usage.cache_write_tokens,
            reasoning_tokens: usage.reasoning_tokens,
            total_tokens: usage.upstream_total_tokens.unwrap_or(0),
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
            applied_guardrails: crate::usage_attr::applied_guardrails(&self.audit),
            guardrail_enforced_hits: crate::usage_attr::enforced_hits(&self.audit),
            guardrail_scores: crate::usage_attr::guardrail_scores(&self.audit),
            guardrail_bypassed_reason: crate::usage_attr::bypass_reason(&self.audit),
            ..Default::default()
        };
        crate::usage_attr::apply_pk_telemetry(&mut event, &pk);
        crate::usage_attr::apply_caller_identity(
            &mut event,
            self.jwt.as_ref(),
            self.user_id.as_deref(),
            self.user_name.as_deref(),
        );
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
            crate::operation::PASSTHROUGH,
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
    // What the caller waited for: the first relayed frame on a streamed
    // relay, the whole response otherwise — the same figure the usage
    // event reports as `downstream_latency_ms`.
    latency: Duration,
    // How long the relay held the gateway, arrival to last byte out. On a
    // streamed relay the two differ by the length of the stream.
    duration: Duration,
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
    let target = crate::attribution::AccessLogTarget::current();
    crate::attribution::emit_access_log(AccessLog {
        method: method.as_str(),
        path,
        status,
        latency,
        duration,
        provider: Some(route),
        model: None,
        upstream_model: target.upstream_model(),
        provider_key_id: target.provider_key_id(),
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
        mcp: None,
        cache: None,
        request_body_bytes: None,
        response_body_bytes: None,
    });
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

    fn scan_candidates_contain(candidates: &[String], expected: &str) -> bool {
        candidates
            .iter()
            .any(|candidate| candidate.contains(expected))
    }

    fn cfg() -> ProxyConfig {
        ProxyConfig {
            addr: "127.0.0.1:0".into(),
            request_body_limit_bytes: 1_048_576,
            real_ip: Default::default(),
            request_id: Default::default(),
            url_rewrites: Vec::new(),
            tls: None,
            listeners: Vec::new(),
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

    /// More than serde_json's default container recursion limit, while still
    /// small enough to fit comfortably under the request body limit.
    fn deeply_nested_escaped_block_json() -> Vec<u8> {
        let mut json = "{\"v\":".repeat(160);
        json.push_str(r#""\u0042LOCKME""#);
        json.push_str(&"}".repeat(160));
        json.into_bytes()
    }

    fn deeply_nested_responses_request_with_opaque_media() -> Vec<u8> {
        let mut json = r#"{"input":[{"type":"message","content":[{"type":"input_image","image_url":"\u0042LOCKME"},{"type":"input_text","text":"clean"}]}],"metadata":"#.to_owned();
        json.push_str(&"{\"next\":".repeat(160));
        json.push_str(r#""deep""#);
        json.push_str(&"}".repeat(160));
        json.push('}');
        json.into_bytes()
    }

    fn nested_anthropic_tool_result_request(depth: usize) -> Vec<u8> {
        let content = format!(
            "{}[{{\"type\":\"text\",\"text\":\"safe\"}}]{}",
            r#"[{"type":"tool_result","tool_use_id":"t","content":"#.repeat(depth),
            "}]".repeat(depth),
        );
        format!(r#"{{"model":"claude","messages":[{{"role":"user","content":{content}}}]}}"#)
            .into_bytes()
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

    /// The `/passthrough/*` namespace carries no special case: with no
    /// route claiming the path it is an ordinary router miss — a bare 404
    /// with an empty body, like any other unmatched path.
    #[tokio::test]
    async fn unclaimed_passthrough_path_takes_the_plain_404() {
        let app = build_app(AisixSnapshot::new());
        let req = Request::builder()
            .method("POST")
            .uri("/passthrough/openai/v1/chat/completions")
            .header("authorization", "Bearer whatever")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        assert!(
            bytes.is_empty(),
            "the miss path carries no error envelope, got {:?}",
            String::from_utf8_lossy(&bytes)
        );
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
        // Disabled → no match → the ordinary router miss.
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
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
    fn joined_target_url_stays_under_its_configured_path() {
        let joined = join_target_url(
            "https://upstream.example/provider/v1",
            "models",
            Some("limit=3"),
        )
        .unwrap();
        assert_eq!(
            joined,
            "https://upstream.example/provider/v1/models?limit=3"
        );

        let joined = join_target_url(
            "https://upstream.example/provider/v1?fixed=1",
            "models",
            Some("limit=3"),
        )
        .unwrap();
        assert_eq!(
            joined,
            "https://upstream.example/provider/v1/models?fixed=1&limit=3"
        );
        assert_eq!(
            strip_redundant_version_segment(
                "https://upstream.example/provider/v1?fixed=1",
                "v1/models",
            ),
            "models"
        );

        assert!(
            join_target_url(
                "https://upstream.example/provider/v1?tenant=operator",
                "models",
                Some("tenant=caller"),
            )
            .is_err(),
            "a caller must not override an operator-owned query key"
        );
        assert!(
            join_target_url(
                "https://upstream.example/provider/v1?tenant=operator",
                "models",
                Some("%74enant=caller"),
            )
            .is_err(),
            "encoded query keys must not bypass the operator-owned key"
        );
        assert!(
            join_target_url(
                "https://upstream.example/provider/v1?tenant=operator",
                "models",
                Some("%2574enant=caller"),
            )
            .is_err(),
            "nested-encoded query keys must not bypass the operator-owned key"
        );
        assert!(
            join_target_url(
                "https://upstream.example/provider/v1?tenant=operator",
                "models",
                Some("safe=1%26tenant%3Dcaller"),
            )
            .is_err(),
            "an encoded query delimiter must not recreate an operator-owned key"
        );
        assert!(
            join_target_url(
                "https://upstream.example/provider/v1?tenant=operator",
                "models",
                Some("safe=1%2526tenant%253Dcaller"),
            )
            .is_err(),
            "a nested-encoded query delimiter must not recreate an operator-owned key"
        );
        for query in [
            "+tenant=caller",
            "%20tenant=caller",
            "%2520tenant=caller",
            "tenant%00suffix=caller",
            "tenant%2500suffix=caller",
        ] {
            assert!(
                join_target_url(
                    "https://upstream.example/provider/v1?tenant=operator",
                    "models",
                    Some(query),
                )
                .is_err(),
                "{query} must not bypass an operator-owned key through PHP-style form-key registration"
            );
        }
        for query in [
            "safe=1;tenant=caller",
            "safe=1%3Btenant%3Dcaller",
            "safe=1%253Btenant%253Dcaller",
        ] {
            assert!(
                join_target_url(
                    "https://upstream.example/provider/v1?tenant=operator",
                    "models",
                    Some(query),
                )
                .is_err(),
                "{query} must not recreate an operator-owned key through a semicolon delimiter"
            );
        }
        for query in [
            "tenant.id=caller",
            "tenant%2Eid=caller",
            "tenant%252Eid=caller",
            "tenant+id=caller",
            "tenant%20id=caller",
        ] {
            assert!(
                join_target_url(
                    "https://upstream.example/provider/v1?tenant_id=operator",
                    "models",
                    Some(query),
                )
                .is_err(),
                "{query} must not bypass an operator-owned key through form-key normalization"
            );
        }
        assert!(
            join_target_url(
                "https://upstream.example/provider/v1?tenant=operator",
                "models",
                Some("tenant%5Brole%5D=caller"),
            )
            .is_err(),
            "a bracketed caller key must not bypass its operator-owned base key"
        );

        // Test raw, percent-encoded, and encoded-separator spellings. The
        // gateway must reject them before a ProviderKey can be sent outside
        // the route's configured target path.
        for remainder in [
            "../models",
            "%2e%2e/models",
            "%2E%2E/models",
            ".%2e/models",
            "%2e./models",
            "%2e%2e%2fmodels",
            "%2e%2e%5cmodels",
            "%252e%252e/models",
            "%252e%252e%252fmodels",
            "%252e%252e%255cmodels",
            "..;ignored/models",
            "%2e%2e%3bignored/models",
            "%252e%252e%253bignored/models",
            "%2e%2e%3bignored/%2e%2e%3bignored/admin",
        ] {
            assert!(
                join_target_url("https://upstream.example/provider/v1", remainder, None).is_err(),
                "{remainder} must not escape the configured target URL path"
            );
        }
    }

    #[tokio::test]
    async fn unsafe_remainder_is_rejected_before_contacting_the_upstream() {
        let upstream = MockServer::start().await;
        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(provider_key_entry("http://unused"));
        snap.apikeys.insert(apikey_entry("sk-caller", Some(&["*"])));
        snap.passthrough_routes
            .insert(inject_route(&format!("{}/provider/v1", upstream.uri())));
        let app = build_app(snap);

        for remainder in [
            "../models",
            "%2e%2e/models",
            "%2e%2e%2fmodels",
            "%252e%252e/models",
            "%252e%252e%252fmodels",
        ] {
            let req = Request::builder()
                .method("GET")
                .uri(format!("/passthrough/openai/{remainder}"))
                .header("authorization", "Bearer sk-caller")
                .body(axum::body::Body::empty())
                .unwrap();
            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "{remainder}");
        }
        assert!(
            upstream.received_requests().await.unwrap().is_empty(),
            "an invalid joined path must not send the ProviderKey upstream"
        );
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
        let mut s = SseFrameSplitter::with_max_frame_bytes(MAX_HELD_STREAM_BYTES);
        let frames = s.push(b"data: a\n\ndata: b\n\ndata: par");
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].bytes, b"data: a\n\n");
        assert!(!frames[0].overflowed);
        let frames = s.push(b"tial\n\n");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].bytes, b"data: partial\n\n");
        assert!(!frames[0].overflowed);
        assert!(s.take_rest().is_empty());
        // CRLF boundaries too.
        let mut s = SseFrameSplitter::with_max_frame_bytes(MAX_HELD_STREAM_BYTES);
        let frames = s.push(b"data: x\r\n\r\nrest");
        assert_eq!(frames.len(), 1);
        assert_eq!(s.take_rest(), b"rest");
    }

    #[test]
    fn sse_splitter_and_usage_label_read_cr_framing() {
        let frame = b"event: token_usage\rdata: {\"input_tokens\":3,\"output_tokens\":4}\r\r";
        let mut s = SseFrameSplitter::with_max_frame_bytes(MAX_HELD_STREAM_BYTES);
        let frames = s.push(&[&frame[..], b"data: next"].concat());
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].bytes, frame);
        assert!(!frames[0].overflowed);
        assert_eq!(s.take_rest(), b"data: next");
        assert!(is_usage_labelled_frame(frame));
        let (_, usage) = frame_delta(PassthroughProtocol::Raw, frame);
        assert!(usage.is_some(), "a CR-framed usage report is read");
    }

    #[test]
    fn frame_in_band_error_reads_the_protocol_s_own_failure_events() {
        let anthropic = b"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"busy\"}}\n\n";
        let err = frame_in_band_error(PassthroughProtocol::OpenaiChat, anthropic).unwrap();
        // Anthropic documents 529 for overloaded; not a 4xx, so it maps to 502.
        assert_eq!(err.http_status(), 502);
        let openai =
            br#"data: {"error":{"message":"slow down","type":"rate_limit_error","code":429}}

"#;
        let err = frame_in_band_error(PassthroughProtocol::OpenaiCompletions, openai).unwrap();
        assert_eq!(err.http_status(), 429);
        let responses = br#"data: {"type":"response.failed","response":{"error":{"code":"server_error","message":"x"}}}

"#;
        assert!(frame_in_band_error(PassthroughProtocol::OpenaiResponses, responses).is_some());
        // An opaque stream is never read for one, and ordinary frames are not one.
        assert!(frame_in_band_error(PassthroughProtocol::Raw, openai).is_none());
        let delta = br#"data: {"choices":[{"delta":{"content":"hel"}}]}

"#;
        assert!(frame_in_band_error(PassthroughProtocol::OpenaiChat, delta).is_none());
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

    /// The recorded total is the upstream's own: adopted from the first
    /// report, kept while every later report carries one, and dropped the
    /// moment one does not — a surviving partial value would be a number
    /// no upstream stated.
    #[test]
    fn merged_usage_keeps_a_total_only_while_every_report_carries_one() {
        let with = usage_of(
            &serde_json::json!({"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}),
        )
        .unwrap();
        let without = usage_of(&serde_json::json!({"output_tokens": 5})).unwrap();
        assert_eq!(with.upstream_total_tokens, Some(10));
        assert_eq!(without.upstream_total_tokens, None);

        let mut acc = None;
        merge_usage(&mut acc, with);
        assert_eq!(acc.unwrap().upstream_total_tokens, Some(10));
        merge_usage(&mut acc, with);
        assert_eq!(acc.unwrap().upstream_total_tokens, Some(10));
        merge_usage(&mut acc, without);
        assert_eq!(acc.unwrap().upstream_total_tokens, None);
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

    /// A frame's payload is ALL of its `data:` lines joined with `\n`
    /// (WHATWG SSE). Parsing each line on its own turns one document into
    /// N unparseable fragments, so the frame's usage went unread and — on
    /// a `Raw` stream — its JSON source text was pushed into the guardrail
    /// scan instead of its values.
    #[test]
    fn a_payload_spread_over_several_data_lines_is_read_as_one_document() {
        let frame = b"event: message_delta\ndata: {\"type\":\"message_delta\",\ndata: \"usage\":{\"output_tokens\":7,\"input_tokens\":12}}\n\n";
        let (_, usage) = frame_delta(PassthroughProtocol::OpenaiChat, frame);
        assert_eq!(
            usage,
            Some(PassthroughUsage {
                prompt_tokens: 12,
                completion_tokens: 7,
                ..Default::default()
            }),
        );

        // The `Raw` scan text is the payload's VALUES for a document that
        // parses — never the raw JSON source, which is what a per-line read
        // fell back to for each fragment.
        let (text, _) = frame_delta(PassthroughProtocol::Raw, frame);
        assert_eq!(text, "message_delta");
    }

    /// Framing varies per ENDPOINT, not per vendor: on one host
    /// `/v1/audio/transcriptions` streams pure CRLF with `\r\n\r\n`
    /// separators and no `event:` lines while `/v1/responses` on the same
    /// host is pure LF. The `\r` belongs to the framing and must reach
    /// neither the parser nor the scan text.
    #[test]
    fn a_crlf_framed_frame_reads_the_same_as_its_lf_twin() {
        let crlf = b"data: {\"usage\":{\"prompt_tokens\":26,\"completion_tokens\":4}}\r\n\r\n";
        let lf = b"data: {\"usage\":{\"prompt_tokens\":26,\"completion_tokens\":4}}\n\n";
        assert_eq!(
            frame_delta(PassthroughProtocol::Raw, crlf),
            frame_delta(PassthroughProtocol::Raw, lf),
        );
        assert_eq!(
            frame_delta(PassthroughProtocol::Raw, crlf).1,
            Some(usage_dims(26, 4)),
        );
        // …and the frame splitter agrees about where such a frame ends.
        let mut splitter = SseFrameSplitter::with_max_frame_bytes(MAX_HELD_STREAM_BYTES);
        let frames = splitter.push(crlf);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].bytes, crlf);
        assert!(!frames[0].overflowed);
    }

    /// A comment-only frame — the keepalive some relays emit while the
    /// upstream thinks — carries no `data:` line at all. It must contribute
    /// no usage and no scan text on every protocol, rather than being read
    /// as an empty or unparseable payload.
    #[test]
    fn a_comment_only_frame_contributes_nothing() {
        for frame in [
            &b": OPENROUTER PROCESSING\n\n"[..],
            &b": OPENROUTER PROCESSING\r\n\r\n"[..],
            &b": keep-alive\nevent: ping\n\n"[..],
        ] {
            for protocol in [
                PassthroughProtocol::Raw,
                PassthroughProtocol::OpenaiChat,
                PassthroughProtocol::OpenaiCompletions,
                PassthroughProtocol::OpenaiResponses,
            ] {
                assert_eq!(
                    frame_delta(protocol, frame),
                    (String::new(), None),
                    "{protocol:?} on {:?}",
                    String::from_utf8_lossy(frame),
                );
            }
        }
    }

    /// A frame whose joined payload does not parse is still FORWARDED to
    /// the client, so producing no scan text for it is a way past an output
    /// block rule. Every protocol falls back to the raw payload text — the
    /// worst case is a false positive, while the alternative is a bypass.
    #[test]
    fn an_unparseable_payload_still_yields_scan_text_on_every_protocol() {
        // Two independent JSON documents on two `data:` lines: joined per
        // the SSE spec this is one unparseable payload, and per-line parsing
        // used to catch it only incidentally.
        let frame = b"data: {\"choices\":[{\"delta\":{\"content\":\"BLOCKME\"}}]}\ndata: {\"choices\":[]}\n\n";
        for protocol in [
            PassthroughProtocol::Raw,
            PassthroughProtocol::OpenaiChat,
            PassthroughProtocol::OpenaiCompletions,
            PassthroughProtocol::OpenaiResponses,
        ] {
            let (text, _) = frame_delta(protocol, frame);
            assert!(
                text.contains("BLOCKME"),
                "{protocol:?} must still offer the forwarded bytes to the scan, got {text:?}",
            );
        }

        let anthropic = b"data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"\\u0042LOCKME\"}}\n\n";
        assert!(
            !frame_guardrail_text(PassthroughProtocol::OpenaiChat, anthropic).contains("BLOCKME")
        );
    }

    /// The `[DONE]` sentinel is not content, on either framing. A stream
    /// that omits it entirely — OpenAI's Responses API sends none — is the
    /// ordinary case, so nothing may depend on having seen one.
    #[test]
    fn the_done_sentinel_contributes_nothing_on_either_framing() {
        for frame in [&b"data: [DONE]\n\n"[..], &b"data: [DONE]\r\n\r\n"[..]] {
            assert_eq!(
                frame_delta(PassthroughProtocol::Raw, frame),
                (String::new(), None),
            );
        }
    }

    /// Reasoning replayed by the caller is REQUEST text and is scanned; the
    /// same field on a buffered RESPONSE is generated reasoning and is out
    /// of the output-guardrail scope. One helper, two answers.
    #[test]
    fn replayed_reasoning_is_request_scan_text_and_not_response_scan_text() {
        let msg = serde_json::json!({
            "role": "assistant",
            "content": "visible",
            "reasoning_content": "hidden reasoning payload",
        });
        assert!(message_scan_text(&msg, true).contains("hidden reasoning payload"));
        assert!(!message_scan_text(&msg, false).contains("hidden reasoning payload"));
        assert!(message_scan_text(&msg, false).contains("visible"));
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
        assert_eq!(response_usage(PassthroughProtocol::Raw, None, rpc), None);
        let top_level = br#"{"usage":{"prompt_tokens":99,"completion_tokens":9}}"#;
        assert_eq!(
            response_usage(PassthroughProtocol::Raw, None, top_level),
            None
        );
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
            body_model_name(PassthroughProtocol::OpenaiChat, None, chat),
            "gpt-4o"
        );
        // An opaque body's `model`-shaped key belongs to some other API.
        let opaque = br#"{"model":"whatever","config_name":"x"}"#;
        assert_eq!(body_model_name(PassthroughProtocol::Raw, None, opaque), "");
        // Caller-supplied, so bounded and control-char free before it
        // reaches telemetry.
        let hostile = format!(
            r#"{{"input":"x","model":"a\u0000b{}"}}"#,
            "z".repeat(REQUESTED_MODEL_CAP * 2)
        );
        let name = body_model_name(
            PassthroughProtocol::OpenaiResponses,
            None,
            hostile.as_bytes(),
        );
        assert_eq!(name.chars().count(), REQUESTED_MODEL_CAP);
        assert!(!name.contains('\0'));
    }

    /// The passthrough route reads the SAME Responses shapes the typed
    /// `/v1/responses` handler does, in the same directions. Request:
    /// a replayed `reasoning` item's `content` AND `summary` are
    /// caller-supplied text and are scanned. Response: a generated
    /// `reasoning` item is out of the output scope and must not be —
    /// the walk reads `content` off every item regardless of type, so
    /// without an explicit skip a block rule matching only inside
    /// reasoning refuses a response the typed route allows.
    #[test]
    fn responses_passthrough_scans_replayed_reasoning_but_not_generated_reasoning() {
        let request = serde_json::json!({
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "VISIBLE"}]},
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "SUMMARYSECRET"}],
                    "content": [{"type": "reasoning_text", "text": "REASONINGSECRET"}]
                },
                {"type": "function_call_output", "call_id": "c1", "output": "TOOLRESULTSECRET"},
                {"type": "mcp_approval_response", "approve": true, "reason": "APPROVALSECRET"}
            ]
        })
        .to_string();
        let scanned =
            request_guardrail_text(PassthroughProtocol::OpenaiResponses, request.as_bytes());
        assert!(scanned.contains("VISIBLE"), "got {scanned:?}");
        assert!(scanned.contains("REASONINGSECRET"), "got {scanned:?}");
        assert!(scanned.contains("SUMMARYSECRET"), "got {scanned:?}");
        // The tool-result and approval slots too. These matter precisely
        // because the items beside them yield text: the raw-body fallback
        // fires only on a WHOLLY empty extraction, so a mixed body would
        // otherwise carry them past the scan while `/v1/responses` blocks
        // the same envelope.
        assert!(scanned.contains("TOOLRESULTSECRET"), "got {scanned:?}");
        assert!(scanned.contains("APPROVALSECRET"), "got {scanned:?}");

        let response = serde_json::json!({
            "output": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "SUMMARYSECRET"}],
                    "content": [{"type": "reasoning_text", "text": "REASONINGSECRET"}]
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "the visible answer"}]
                }
            ]
        })
        .to_string();
        let scanned =
            response_guardrail_text(PassthroughProtocol::OpenaiResponses, response.as_bytes());
        assert!(scanned.contains("the visible answer"), "got {scanned:?}");
        assert!(!scanned.contains("REASONINGSECRET"), "got {scanned:?}");
        // Not a raw-body fallback: the message item yielded text, so a
        // green above means the reasoning item was skipped rather than the
        // whole walk having come back empty.
        assert!(!scanned.contains("\"output\""), "got {scanned:?}");
    }

    #[test]
    fn request_text_extraction_per_protocol() {
        let chat = br#"{"model":"routing-model-only","messages":[{"role":"system","content":"s"},{"role":"user","content":[{"type":"text","text":"part"}]}],"forwarded_extra":"supplement"}"#;
        let scanned = request_guardrail_text(PassthroughProtocol::OpenaiChat, chat);
        for text in ["s", "part", "supplement"] {
            assert!(scanned.contains(text), "{text} missing from {scanned:?}");
        }
        // `model` selects the protocol target rather than supplying caller
        // content. Arbitrary forwarded extras must still be scanned.
        assert!(
            !scanned.contains("routing-model-only"),
            "protocol routing metadata leaked into guardrail text: {scanned:?}"
        );
        let fim = br#"{"prompt":"def f(","suffix":"return"}"#;
        assert_eq!(
            request_guardrail_text(PassthroughProtocol::OpenaiCompletions, fim),
            "def f(\nreturn"
        );
        // Shape mismatch degrades to every decoded JSON string value.
        let not_chat = br#"{"input":"x"}"#;
        assert_eq!(
            request_guardrail_text(PassthroughProtocol::OpenaiChat, not_chat),
            "x"
        );
        // Envelope metadata alone is not input text, but a forwarded
        // supplementary field still is scanned.
        let empty_chat = br#"{"messages":[{"role":"tool","tool_call_id":"1"}],"state":"fallback"}"#;
        let scanned = request_guardrail_text(PassthroughProtocol::OpenaiChat, empty_chat);
        assert!(scanned.contains("fallback"), "{scanned:?}");
    }

    #[test]
    fn request_source_scan_keeps_opaque_media_out_of_guardrail_text() {
        let chat = br#"{"messages":[{"role":"user","content":[{"type":"image","source":{"data":"\u0042LOCKME"}},{"type":"document","source":{"data":"\u0042LOCKME"}},{"type":"text","text":"clean"}]}]}"#;
        let scanned = request_guardrail_text(PassthroughProtocol::OpenaiChat, chat);
        assert!(scanned.contains("clean"), "{scanned:?}");
        assert!(!scanned.contains("BLOCKME"), "{scanned:?}");

        let responses = br#"{"input":[{"type":"message","content":[{"type":"input_image","image_url":"\u0042LOCKME"},{"type":"input_text","text":"clean"}]}]}"#;
        let scanned = request_guardrail_text(PassthroughProtocol::OpenaiResponses, responses);
        assert!(scanned.contains("clean"), "{scanned:?}");
        assert!(!scanned.contains("BLOCKME"), "{scanned:?}");

        let completions = br#"{"prompt":[{"type":"image","data":"\u0042LOCKME"},"clean"]}"#;
        let scanned = request_guardrail_text(PassthroughProtocol::OpenaiCompletions, completions);
        assert!(scanned.contains("clean"), "{scanned:?}");
        assert!(!scanned.contains("BLOCKME"), "{scanned:?}");
    }

    #[test]
    fn deep_responses_request_stays_typed_and_keeps_media_opaque() {
        let request = deeply_nested_responses_request_with_opaque_media();
        assert_eq!(
            detect_protocol(&request),
            PassthroughProtocol::OpenaiResponses
        );
        let scanned = request_guardrail_text(PassthroughProtocol::OpenaiResponses, &request);
        assert!(scanned.contains("clean"), "{scanned:?}");
        assert!(
            !scanned.contains("BLOCKME"),
            "a deep valid envelope must not fall back to raw media: {scanned:?}"
        );
    }

    #[test]
    fn duplicate_chat_carrier_skips_malformed_source_without_leaking_media() {
        let request = br#"{"messages":{"content":[{"type":"image","source":{"data":"\u0042LOCKME"}}]},"messages":[{"role":"user","content":"clean"}]}"#;
        assert_eq!(detect_protocol(request), PassthroughProtocol::OpenaiChat);
        let scanned = request_guardrail_text(PassthroughProtocol::OpenaiChat, request);
        assert!(scanned.contains("clean"), "{scanned:?}");
        assert!(
            !scanned.contains("BLOCKME"),
            "a malformed duplicate must not trigger raw fallback: {scanned:?}"
        );
    }

    #[test]
    fn malformed_typed_array_items_keep_media_opaque() {
        let cases = [
            (
                PassthroughProtocol::OpenaiChat,
                br#"{"messages":["junk",{"role":"user","content":[{"type":"image","source":{"data":"\u0042LOCKME"}},{"type":"text","text":"clean"}]}]}"#.as_slice(),
            ),
            (
                PassthroughProtocol::OpenaiChat,
                br#"{"messages":[{"role":"user","content":["junk",{"type":"image","source":{"data":"\u0042LOCKME"}},{"type":"text","text":"clean"}]}]}"#.as_slice(),
            ),
            (
                PassthroughProtocol::OpenaiChat,
                br#"{"messages":[{"role":"user","content":[{"type":"text","text":{"image_url":"\u0042LOCKME"}},{"type":"text","text":"clean"}]}]}"#.as_slice(),
            ),
            (
                PassthroughProtocol::OpenaiResponses,
                br#"{"input":["junk",{"type":"message","content":[{"type":"input_image","image_url":"\u0042LOCKME"},{"type":"input_text","text":"clean"}]}]}"#.as_slice(),
            ),
            (
                PassthroughProtocol::OpenaiResponses,
                br#"{"input":[{"type":"message","content":[{"type":"input_text","text":{"image_url":"\u0042LOCKME"}},{"type":"input_text","text":"clean"}]}]}"#.as_slice(),
            ),
        ];
        for (protocol, request) in cases {
            let scanned = request_guardrail_text(protocol, request);
            assert!(scanned.contains("clean"), "{protocol:?}: {scanned:?}");
            assert!(
                !scanned.contains("BLOCKME"),
                "a malformed array item must not trigger raw fallback for {protocol:?}: {scanned:?}"
            );
        }
    }

    #[test]
    fn guardrail_text_scans_decoded_forwarded_json_strings() {
        let raw = br#"{"state":"\u0042LOCKME","nested":{"query":"\u4e2d\u6587"}}"#;
        let scanned = request_guardrail_text(PassthroughProtocol::Raw, raw);
        assert!(scanned.contains("BLOCKME"), "got {scanned:?}");
        assert!(scanned.contains("中文"), "got {scanned:?}");
        assert!(!scanned.contains(r#"\u0042"#), "got {scanned:?}");

        let duplicate = br#"{"state":"\u0042LOCKME","state":"clean"}"#;
        let scanned = request_guardrail_text(PassthroughProtocol::Raw, duplicate);
        for text in ["BLOCKME", "clean"] {
            assert!(scanned.contains(text), "{text} missing from {scanned:?}");
        }

        let chat = br#"{"messages":[{"role":"user","content":"clean"}],"state":{"query":"BLOCKME"},"state":"also-clean","documents":["\u4e2d\u6587"]}"#;
        let scanned = request_guardrail_text(PassthroughProtocol::OpenaiChat, chat);
        for text in ["clean", "BLOCKME", "also-clean", "中文"] {
            assert!(scanned.contains(text), "{text} missing from {scanned:?}");
        }

        let response = br#"{"state":"\u0042LOCKME","state":"clean"}"#;
        let scanned = response_guardrail_text(PassthroughProtocol::Raw, response);
        for text in ["BLOCKME", "clean"] {
            assert!(scanned.contains(text), "{text} missing from {scanned:?}");
        }
        assert_eq!(
            response_capture_text(PassthroughProtocol::Raw, response),
            r#"{"state":"\u0042LOCKME","state":"clean"}"#
        );

        let deep = deeply_nested_escaped_block_json();
        assert!(
            request_guardrail_text(PassthroughProtocol::Raw, &deep).contains("BLOCKME"),
            "a valid deep Raw request must not fall back to escaped source"
        );
        assert!(
            response_guardrail_text(PassthroughProtocol::Raw, &deep).contains("BLOCKME"),
            "a valid deep Raw response must not fall back to escaped source"
        );
    }

    #[test]
    fn known_request_envelopes_scan_duplicate_and_nested_source_strings() {
        let cases = [
            (
                PassthroughProtocol::OpenaiChat,
                br#"{"model":"routing-only","messages":[{"role":"user","content":"\u0069nputleakliteral"}],"messages":[{"role":"user","content":"clean","metadata":{"note":"NESTED"}}]}"#
                    .as_slice(),
            ),
            (
                PassthroughProtocol::OpenaiResponses,
                br#"{"model":"routing-only","input":"\u0069nputleakliteral","input":"clean","metadata":{"note":"NESTED"}}"#
                    .as_slice(),
            ),
            (
                PassthroughProtocol::OpenaiCompletions,
                br#"{"model":"routing-only","prompt":"\u0069nputleakliteral","prompt":"clean","metadata":{"note":"NESTED"}}"#
                    .as_slice(),
            ),
        ];
        for (protocol, body) in cases {
            let scanned = request_guardrail_text(protocol, body);
            for expected in ["inputleakliteral", "clean", "NESTED"] {
                assert!(scanned.contains(expected), "{protocol:?}: {scanned:?}");
            }
            assert!(
                !scanned.contains("routing-only"),
                "{protocol:?}: {scanned:?}"
            );
        }
    }

    #[test]
    fn signed_redacted_thinking_is_not_request_guardrail_text() {
        let redacted = br#"{"messages":[{"role":"assistant","content":[{"type":"redacted_thinking","data":"\u0042LOCKME","signature":"signed"},{"type":"text","text":"clean"}]}]}"#;
        let scanned = request_guardrail_text(PassthroughProtocol::OpenaiChat, redacted);
        assert!(scanned.contains("clean"), "{scanned:?}");
        assert!(!scanned.contains("BLOCKME"), "{scanned:?}");

        let ambiguous = br#"{"messages":[{"role":"assistant","content":[{"type":"redacted_thinking","type":"text","data":"\u0042LOCKME"}]}]}"#;
        assert!(
            request_guardrail_text(PassthroughProtocol::OpenaiChat, ambiguous).contains("BLOCKME")
        );
    }

    #[test]
    fn known_response_envelopes_scan_duplicate_selected_source_strings() {
        let chat = br#"{"model":"routing-only","choices":[{"message":{"content":"\u0042LOCKME","metadata":{"note":"NESTED"}}}],"choices":[{"message":{"content":"clean"}}]}"#;
        let scanned = response_guardrail_text(PassthroughProtocol::OpenaiChat, chat);
        for expected in ["BLOCKME", "clean"] {
            assert!(scanned.contains(expected), "{scanned:?}");
        }
        assert!(!scanned.contains("NESTED"), "{scanned:?}");
        assert!(!scanned.contains("routing-only"), "{scanned:?}");

        let responses = br#"{"output":[{"type":"message","content":[{"type":"output_text","text":"\u0042LOCKME","metadata":{"note":"NESTED"}}]}],"output":[{"type":"message","content":[{"type":"output_text","text":"clean"}]}]}"#;
        let scanned = response_guardrail_text(PassthroughProtocol::OpenaiResponses, responses);
        for expected in ["BLOCKME", "clean"] {
            assert!(scanned.contains(expected), "{scanned:?}");
        }
        assert!(!scanned.contains("NESTED"), "{scanned:?}");

        let conflicting_type = br#"{"output":[{"type":"reasoning","type":"message","content":[{"text":"\u0042LOCKME"}]}]}"#;
        assert!(
            !response_guardrail_text(PassthroughProtocol::OpenaiResponses, conflicting_type)
                .contains("BLOCKME")
        );

        let same_hidden_type = br#"{"output":[{"type":"reasoning","type":"reasoning","summary":[{"text":"\u0042LOCKME"}]}]}"#;
        assert!(
            !response_guardrail_text(PassthroughProtocol::OpenaiResponses, same_hidden_type)
                .contains("BLOCKME")
        );

        let duplicate_text = br#"{"output":[{"type":"message","content":[{"type":"output_text","text":"\u0042LOCKME","text":"clean"}]}]}"#;
        assert!(
            response_guardrail_text(PassthroughProtocol::OpenaiResponses, duplicate_text)
                .contains("BLOCKME")
        );
        assert!(
            response_guardrail_text(PassthroughProtocol::OpenaiResponses, duplicate_text)
                .contains("clean")
        );
    }

    #[test]
    fn chat_output_guardrail_keeps_media_and_unknown_parts_opaque() {
        let buffered = br#"{
            "choices": [{
                "message": {
                    "content": [
                        {"type":"image_url","image_url":{"url":"BUFFERED_IMAGE_SENTINEL"}},
                        {"type":"input_audio","input_audio":{"data":"BUFFERED_AUDIO_SENTINEL"}},
                        {"type":"file","file":{"file_data":"BUFFERED_FILE_SENTINEL"}},
                        {"type":"future_media","text":"BUFFERED_OPAQUE_PART_SENTINEL"},
                        {"type":"text","text":"BUFFERED_VISIBLE_SENTINEL"}
                    ],
                    "reasoning_content": "BUFFERED_REASONING_SENTINEL",
                    "metadata": {"note":"BUFFERED_METADATA_SENTINEL"},
                    "tool_calls": [
                        {"type":"function","function":{"name":"lookup","arguments":"BUFFERED_TOOL_ARGUMENT_SENTINEL"}},
                        {"type":"custom","custom":{"name":"custom","input":"BUFFERED_CUSTOM_INPUT_SENTINEL"}}
                    ],
                    "function_call": {"name":"legacy","arguments":"BUFFERED_LEGACY_ARGUMENT_SENTINEL"}
                }
            }]
        }"#;
        let scanned = response_guardrail_text(PassthroughProtocol::OpenaiChat, buffered);
        for expected in [
            "BUFFERED_VISIBLE_SENTINEL",
            "lookup",
            "BUFFERED_TOOL_ARGUMENT_SENTINEL",
            "custom",
            "BUFFERED_CUSTOM_INPUT_SENTINEL",
            "legacy",
            "BUFFERED_LEGACY_ARGUMENT_SENTINEL",
        ] {
            assert!(scanned.contains(expected), "{scanned:?}");
        }
        for opaque in [
            "BUFFERED_IMAGE_SENTINEL",
            "BUFFERED_AUDIO_SENTINEL",
            "BUFFERED_FILE_SENTINEL",
            "BUFFERED_OPAQUE_PART_SENTINEL",
            "BUFFERED_REASONING_SENTINEL",
            "BUFFERED_METADATA_SENTINEL",
        ] {
            assert!(!scanned.contains(opaque), "{scanned:?}");
        }

        let frame = b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":[{\"index\":0,\"type\":\"image_url\",\"image_url\":{\"url\":\"STREAM_IMAGE_SENTINEL\"}},{\"index\":1,\"type\":\"input_audio\",\"input_audio\":{\"data\":\"STREAM_AUDIO_SENTINEL\"}},{\"index\":2,\"type\":\"file\",\"file\":{\"file_data\":\"STREAM_FILE_SENTINEL\"}},{\"index\":3,\"type\":\"future_media\",\"text\":\"STREAM_OPAQUE_PART_SENTINEL\"},{\"index\":4,\"type\":\"text\",\"text\":\"STREAM_VISIBLE_SENTINEL\"}],\"reasoning_content\":\"STREAM_REASONING_SENTINEL\",\"metadata\":{\"note\":\"STREAM_METADATA_SENTINEL\"},\"tool_calls\":[{\"index\":0,\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"arguments\":\"STREAM_TOOL_ARGUMENT_SENTINEL\"}},{\"index\":1,\"type\":\"custom\",\"custom\":{\"name\":\"custom\",\"input\":\"STREAM_CUSTOM_INPUT_SENTINEL\"}}],\"function_call\":{\"name\":\"legacy\",\"arguments\":\"STREAM_LEGACY_ARGUMENT_SENTINEL\"}}}]}\n\n";
        let typed = frame_parts(PassthroughProtocol::OpenaiChat, frame).0.scan;
        assert!(typed.contains("STREAM_OPAQUE_PART_SENTINEL"), "{typed:?}");
        let text = stream_guardrail_text(PassthroughProtocol::OpenaiChat, frame, typed);
        assert!(!text.unevaluable);
        let scanned = stream_guardrail_scan_text(&[], &text.continuations, &text.supplemental);
        for expected in [
            "STREAM_VISIBLE_SENTINEL",
            "lookup",
            "STREAM_TOOL_ARGUMENT_SENTINEL",
            "custom",
            "STREAM_CUSTOM_INPUT_SENTINEL",
            "legacy",
            "STREAM_LEGACY_ARGUMENT_SENTINEL",
        ] {
            assert!(scan_candidates_contain(&scanned, expected), "{scanned:?}");
        }
        for opaque in [
            "STREAM_IMAGE_SENTINEL",
            "STREAM_AUDIO_SENTINEL",
            "STREAM_FILE_SENTINEL",
            "STREAM_OPAQUE_PART_SENTINEL",
            "STREAM_REASONING_SENTINEL",
            "STREAM_METADATA_SENTINEL",
        ] {
            assert!(!scan_candidates_contain(&scanned, opaque), "{scanned:?}");
        }
    }

    #[test]
    fn chat_output_guardrail_scans_client_visible_refusals() {
        let buffered = br#"{
            "choices": [{
                "message": {
                    "content": [
                        {"type":"image_url","image_url":{"url":"BUFFERED_MEDIA_SENTINEL"}},
                        {"type":"refusal","refusal":"CONTENT_REFUSAL_BLOCKME"}
                    ],
                    "refusal":"MESSAGE_REFUSAL_BLOCKME",
                    "reasoning_content":"BUFFERED_REASONING_SENTINEL"
                }
            }]
        }"#;
        let scanned = response_guardrail_text(PassthroughProtocol::OpenaiChat, buffered);
        for refusal in ["CONTENT_REFUSAL_BLOCKME", "MESSAGE_REFUSAL_BLOCKME"] {
            assert!(scanned.contains(refusal), "{scanned:?}");
        }
        for opaque in ["BUFFERED_MEDIA_SENTINEL", "BUFFERED_REASONING_SENTINEL"] {
            assert!(!scanned.contains(opaque), "{scanned:?}");
        }

        let first_frame =
            b"data: {\"choices\":[{\"index\":0,\"delta\":{\"refusal\":\"BLOC\"}}]}\n\n";
        let second_frame =
            b"data: {\"choices\":[{\"index\":0,\"delta\":{\"refusal\":\"KME\"}}]}\n\n";
        let first = stream_guardrail_text(
            PassthroughProtocol::OpenaiChat,
            first_frame,
            frame_parts(PassthroughProtocol::OpenaiChat, first_frame)
                .0
                .scan,
        );
        let second = stream_guardrail_text(
            PassthroughProtocol::OpenaiChat,
            second_frame,
            frame_parts(PassthroughProtocol::OpenaiChat, second_frame)
                .0
                .scan,
        );
        assert!(!first.unevaluable);
        assert!(!second.unevaluable);

        let mut continuations = Vec::new();
        let mut tails = Vec::new();
        let mut supplemental = Vec::new();
        let mut closed_prefixes = Vec::new();
        append_stream_guardrail_text(
            &mut continuations,
            &mut tails,
            &mut supplemental,
            &mut closed_prefixes,
            &first,
        );
        append_stream_guardrail_text(
            &mut continuations,
            &mut tails,
            &mut supplemental,
            &mut closed_prefixes,
            &second,
        );
        let scanned = stream_guardrail_scan_text(&tails, &continuations, &supplemental);
        assert!(scan_candidates_contain(&scanned, "BLOCKME"), "{scanned:?}");
    }

    #[test]
    fn responses_output_guardrail_keeps_generated_media_opaque() {
        let buffered = br#"{"output":[{"type":"image_generation_call","result":"BUFFERED_MEDIA_SENTINEL"},{"type":"message","content":[{"type":"output_text","text":"VISIBLE_TEXT_SENTINEL"}]},{"type":"function_call","name":"lookup","arguments":"{\"query\":\"TOOL_ARGUMENT_SENTINEL\"}"},{"type":"mcp_call","name":"mcp","arguments":"MCP_ARGUMENT_SENTINEL"},{"type":"custom_tool_call","name":"custom","input":"CUSTOM_INPUT_SENTINEL"}]}"#;
        let scanned = response_guardrail_text(PassthroughProtocol::OpenaiResponses, buffered);
        for expected in [
            "VISIBLE_TEXT_SENTINEL",
            "TOOL_ARGUMENT_SENTINEL",
            "MCP_ARGUMENT_SENTINEL",
            "CUSTOM_INPUT_SENTINEL",
        ] {
            assert!(scanned.contains(expected), "{scanned:?}");
        }
        assert!(!scanned.contains("BUFFERED_MEDIA_SENTINEL"), "{scanned:?}");

        // A conflicting item discriminator is opaque as a whole. Even
        // safe-named fields could be an image/audio/file payload attached to
        // the other discriminator.
        let ambiguous = br#"{"output":[{"type":"image_generation_call","type":"message","result":"AMBIGUOUS_MEDIA_SENTINEL","content":[{"type":"output_text","text":"AMBIGUOUS_VISIBLE_SENTINEL"}],"arguments":"AMBIGUOUS_TOOL_SENTINEL"}]}"#;
        let scanned = response_guardrail_text(PassthroughProtocol::OpenaiResponses, ambiguous);
        for opaque in [
            "AMBIGUOUS_MEDIA_SENTINEL",
            "AMBIGUOUS_VISIBLE_SENTINEL",
            "AMBIGUOUS_TOOL_SENTINEL",
        ] {
            assert!(!scanned.contains(opaque), "{scanned:?}");
        }

        let unknown =
            br#"{"output":[{"text":"UNKNOWN_TEXT_SENTINEL","input":"UNKNOWN_MEDIA_SENTINEL"}]}"#;
        let scanned = response_guardrail_text(PassthroughProtocol::OpenaiResponses, unknown);
        assert!(!scanned.contains("UNKNOWN_TEXT_SENTINEL"), "{scanned:?}");
        assert!(!scanned.contains("UNKNOWN_MEDIA_SENTINEL"), "{scanned:?}");

        let media_frame = b"data: {\"type\":\"response.image_generation_call.partial_image\",\"partial_image_b64\":\"STREAM_MEDIA_SENTINEL\"}\n\n";
        assert!(
            !frame_guardrail_text(PassthroughProtocol::OpenaiResponses, media_frame)
                .contains("STREAM_MEDIA_SENTINEL")
        );
        let typed = frame_parts(PassthroughProtocol::OpenaiResponses, media_frame)
            .0
            .scan;
        let media = stream_guardrail_text(PassthroughProtocol::OpenaiResponses, media_frame, typed);
        let scanned = stream_guardrail_scan_text(&[], &media.continuations, &media.supplemental);
        assert!(
            !scan_candidates_contain(&scanned, "STREAM_MEDIA_SENTINEL"),
            "{scanned:?}"
        );

        // The terminal response repeats prior delta content, including media
        // from image-generation output. It is never a second scan carrier.
        let terminal = b"data: {\"type\":\"response.completed\",\"response\":{\"output\":[{\"type\":\"image_generation_call\",\"result\":\"TERMINAL_MEDIA_SENTINEL\"},{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"TERMINAL_VISIBLE_SENTINEL\"}]}]}}\n\n";
        let scanned = frame_guardrail_text(PassthroughProtocol::OpenaiResponses, terminal);
        assert!(!scanned.contains("TERMINAL_MEDIA_SENTINEL"), "{scanned:?}");
        assert!(
            !scanned.contains("TERMINAL_VISIBLE_SENTINEL"),
            "{scanned:?}"
        );

        // `delta` is not a universally textual field: on a conflicting text
        // and audio discriminator it is opaque, rather than a path for audio
        // base64 to reach a text guardrail.
        let conflicting_delta = b"data: {\"type\":\"response.output_audio.delta\",\"type\":\"response.output_text.delta\",\"delta\":\"CONFLICTING_MEDIA_SENTINEL\"}\n\n";
        let typed = frame_parts(PassthroughProtocol::OpenaiResponses, conflicting_delta)
            .0
            .scan;
        assert!(typed.contains("CONFLICTING_MEDIA_SENTINEL"), "{typed:?}");
        let text = stream_guardrail_text(
            PassthroughProtocol::OpenaiResponses,
            conflicting_delta,
            typed,
        );
        let scanned = stream_guardrail_scan_text(&[], &text.continuations, &text.supplemental);
        assert!(
            !scan_candidates_contain(&scanned, "CONFLICTING_MEDIA_SENTINEL"),
            "{scanned:?}"
        );

        let ambiguous_event = b"data: {\"type\":\"response.output_text.done\",\"type\":\"response.output_audio.done\",\"text\":\"AMBIGUOUS_EVENT_TEXT_SENTINEL\",\"input\":\"AMBIGUOUS_EVENT_MEDIA_SENTINEL\"}\n\n";
        let scanned = frame_guardrail_text(PassthroughProtocol::OpenaiResponses, ambiguous_event);
        assert!(
            !scanned.contains("AMBIGUOUS_EVENT_TEXT_SENTINEL"),
            "{scanned:?}"
        );
        assert!(
            !scanned.contains("AMBIGUOUS_EVENT_MEDIA_SENTINEL"),
            "{scanned:?}"
        );

        let text_frame = b"data: {\"type\":\"response.output_text.delta\",\"delta\":\"VISIBLE_STREAM_SENTINEL\"}\n\n";
        assert!(
            frame_guardrail_text(PassthroughProtocol::OpenaiResponses, text_frame)
                .contains("VISIBLE_STREAM_SENTINEL")
        );
    }

    #[test]
    fn generated_reasoning_stays_out_of_the_source_preserving_output_scan() {
        let buffered = br#"{"output":[{"type":"reasoning","summary":[{"type":"summary_text","text":"\u0042LOCKME"}]}]}"#;
        assert!(
            !response_guardrail_text(PassthroughProtocol::OpenaiResponses, buffered)
                .contains("BLOCKME")
        );

        for frame in [
            b"data: {\"type\":\"response.reasoning_text.delta\",\"delta\":\"\\u0042LOCKME\"}\n\n".as_slice(),
            b"data: {\"type\":\"response.reasoning_text.done\",\"text\":\"\\u0042LOCKME\"}\n\n".as_slice(),
            b"data: {\"type\":\"response.reasoning_summary_part.done\",\"part\":{\"type\":\"summary_text\",\"text\":\"\\u0042LOCKME\"}}\n\n".as_slice(),
            b"data: {\"type\":\"response.content_part.done\",\"part\":{\"type\":\"reasoning_text\",\"text\":\"\\u0042LOCKME\"}}\n\n".as_slice(),
            b"data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"reasoning\",\"summary\":[{\"text\":\"\\u0042LOCKME\"}]}}\n\n".as_slice(),
        ] {
            assert!(
                !frame_guardrail_text(PassthroughProtocol::OpenaiResponses, frame)
                    .contains("BLOCKME")
            );
        }
    }

    #[test]
    fn known_sse_envelopes_scan_selected_source_strings() {
        let chat = b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\\u0042LOCKME\",\"metadata\":{\"note\":\"NESTED\"}}},{\"index\":1,\"delta\":{\"content\":\"clean\"}}]}\n\n";
        let scanned = frame_guardrail_text(PassthroughProtocol::OpenaiChat, chat);
        for expected in ["BLOCKME", "clean"] {
            assert!(scanned.contains(expected), "{scanned:?}");
        }
        assert!(!scanned.contains("NESTED"), "{scanned:?}");

        let responses = b"data: {\"type\":\"response.output_text.delta\",\"delta\":\"\\u0042LOCKME\",\"delta\":\"clean\",\"metadata\":{\"note\":\"NESTED\"}}\n\n";
        let scanned = frame_guardrail_text(PassthroughProtocol::OpenaiResponses, responses);
        for expected in ["BLOCKME", "clean"] {
            assert!(scanned.contains(expected), "{scanned:?}");
        }
        assert!(!scanned.contains("NESTED"), "{scanned:?}");

        let conflicting = b"data: {\"type\":\"response.content_part.done\",\"part\":{\"type\":\"reasoning_text\",\"type\":\"output_text\",\"text\":\"\\u0042LOCKME\"}}\n\n";
        assert!(
            !frame_guardrail_text(PassthroughProtocol::OpenaiResponses, conflicting)
                .contains("BLOCKME")
        );

        let conflicting_item = b"data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"reasoning\",\"type\":\"message\",\"content\":[{\"text\":\"\\u0042LOCKME\"}]}}\n\n";
        assert!(
            !frame_guardrail_text(PassthroughProtocol::OpenaiResponses, conflicting_item)
                .contains("BLOCKME")
        );
        let hidden_item = b"data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"reasoning\",\"type\":\"reasoning\",\"summary\":[{\"text\":\"\\u0042LOCKME\"}]}}\n\n";
        assert!(
            !frame_guardrail_text(PassthroughProtocol::OpenaiResponses, hidden_item)
                .contains("BLOCKME")
        );

        let conflicting_anthropic = b"data: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"thinking\",\"type\":\"text\",\"text\":\"\\u0042LOCKME\"}}\n\n";
        assert!(
            !frame_guardrail_text(PassthroughProtocol::OpenaiChat, conflicting_anthropic)
                .contains("BLOCKME")
        );
        let hidden_anthropic = b"data: {\"type\":\"content_block_start\",\"content_block\":{\"type\":\"thinking\",\"type\":\"thinking\",\"thinking\":\"\\u0042LOCKME\"}}\n\n";
        assert!(
            !frame_guardrail_text(PassthroughProtocol::OpenaiChat, hidden_anthropic)
                .contains("BLOCKME")
        );
    }

    #[test]
    fn stream_guardrail_text_preserves_response_duplicate_carrier_branches() {
        let first = b"data: {\"type\":\"response.output_text.delta\",\"item_id\":\"one\",\"output_index\":0,\"content_index\":0,\"delta\":\"FOR\",\"delta\":\"ok\"}\n\n";
        let second = b"data: {\"type\":\"response.output_text.delta\",\"item_id\":\"one\",\"output_index\":0,\"content_index\":0,\"delta\":\"BIDDEN\"}\n\n";
        let first_parts = frame_parts(PassthroughProtocol::OpenaiResponses, first).0;
        let second_parts = frame_parts(PassthroughProtocol::OpenaiResponses, second).0;
        let first = stream_guardrail_text(
            PassthroughProtocol::OpenaiResponses,
            first,
            first_parts.scan,
        );
        let second = stream_guardrail_text(
            PassthroughProtocol::OpenaiResponses,
            second,
            second_parts.scan,
        );
        let mut continuations = Vec::new();
        let mut continuation_tails = Vec::new();
        let mut supplemental = Vec::new();
        let mut closed_prefixes = Vec::new();
        append_stream_guardrail_text(
            &mut continuations,
            &mut continuation_tails,
            &mut supplemental,
            &mut closed_prefixes,
            &first,
        );
        append_stream_guardrail_text(
            &mut continuations,
            &mut continuation_tails,
            &mut supplemental,
            &mut closed_prefixes,
            &second,
        );
        let scanned =
            stream_guardrail_scan_text(&continuation_tails, &continuations, &supplemental);
        assert!(
            scan_candidates_contain(&scanned, "FORBIDDEN"),
            "{scanned:?}"
        );
        assert!(scan_candidates_contain(&scanned, "okBIDDEN"), "{scanned:?}");
    }

    #[test]
    fn stream_guardrail_text_keys_reordered_chat_choices_by_source_index() {
        let first = b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"FOR\"}},{\"index\":1,\"delta\":{\"content\":\"noise\"}}]}\n\n";
        let second = b"data: {\"choices\":[{\"index\":1,\"delta\":{\"content\":\"CLEAN\"}},{\"index\":0,\"delta\":{\"content\":\"BIDDEN\"}}]}\n\n";
        let first_parts = frame_parts(PassthroughProtocol::OpenaiChat, first).0;
        let second_parts = frame_parts(PassthroughProtocol::OpenaiChat, second).0;
        let first = stream_guardrail_text(PassthroughProtocol::OpenaiChat, first, first_parts.scan);
        let second =
            stream_guardrail_text(PassthroughProtocol::OpenaiChat, second, second_parts.scan);
        assert!(!first.unevaluable);
        assert!(!second.unevaluable);
        let mut continuations = Vec::new();
        let mut continuation_tails = Vec::new();
        let mut supplemental = Vec::new();
        let mut closed_prefixes = Vec::new();
        append_stream_guardrail_text(
            &mut continuations,
            &mut continuation_tails,
            &mut supplemental,
            &mut closed_prefixes,
            &first,
        );
        append_stream_guardrail_text(
            &mut continuations,
            &mut continuation_tails,
            &mut supplemental,
            &mut closed_prefixes,
            &second,
        );
        let scanned =
            stream_guardrail_scan_text(&continuation_tails, &continuations, &supplemental);
        assert!(
            scan_candidates_contain(&scanned, "FORBIDDEN"),
            "{scanned:?}"
        );
        assert!(
            !scan_candidates_contain(&scanned, "noiseBIDDEN"),
            "{scanned:?}"
        );
    }

    #[test]
    fn source_continuity_refuses_unkeyable_or_over_cap_response_branches() {
        let over_cap = br#"{"type":"response.output_text.delta","item_id":"one","output_index":0,"content_index":0,"delta":"a","delta":"b","delta":"c"}"#;
        assert!(matches!(
            stream_source_continuations(PassthroughProtocol::OpenaiResponses, over_cap),
            SourceContinuations::Unevaluable
        ));
        let missing_id = br#"{"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"FOR"}"#;
        assert!(matches!(
            stream_source_continuations(PassthroughProtocol::OpenaiResponses, missing_id),
            SourceContinuations::Unevaluable
        ));
        let oversized_id = "x".repeat(MAX_STREAM_GUARDRAIL_SOURCE_ID_BYTES + 1);
        let oversized_delta = format!(
            "{{\"type\":\"response.output_text.delta\",\"item_id\":\"{oversized_id}\",\"output_index\":0,\"content_index\":0,\"delta\":\"safe\"}}"
        );
        assert!(matches!(
            stream_source_continuations(
                PassthroughProtocol::OpenaiResponses,
                oversized_delta.as_bytes()
            ),
            SourceContinuations::Unevaluable
        ));
        let oversized_done = format!(
            "data: {{\"type\":\"response.output_item.done\",\"item\":{{\"id\":\"{oversized_id}\"}}}}\n\n"
        );
        assert!(
            stream_guardrail_text(
                PassthroughProtocol::OpenaiResponses,
                oversized_done.as_bytes(),
                String::new(),
            )
            .unevaluable
        );
        assert!(matches!(
            stream_source_continuations(PassthroughProtocol::Raw, br#"{"state":"FOR"}"#),
            SourceContinuations::Unevaluable
        ));
        assert!(matches!(
            stream_source_continuations(PassthroughProtocol::Raw, br#""FOR""#),
            SourceContinuations::Ready(_)
        ));

        let continuations = (0..MAX_STREAM_GUARDRAIL_CHANNELS)
            .map(|index| StreamContinuation {
                key: format!("responses:\"{index}\":text:first"),
                family: format!("responses:\"{index}\""),
                identity: "text".to_owned(),
                identity_is_ambiguous: false,
                text: "safe".to_owned(),
            })
            .collect::<Vec<_>>();
        let incoming = StreamGuardrailText {
            continuations: vec![StreamContinuation {
                key: "responses:\"next\":text:first".to_owned(),
                family: "responses:\"next\"".to_owned(),
                identity: "text".to_owned(),
                identity_is_ambiguous: false,
                text: "safe".to_owned(),
            }],
            supplemental: Vec::new(),
            unevaluable: false,
            closed_prefixes: Vec::new(),
        };
        assert!(stream_continuation_would_exceed_cap(
            &continuations,
            &[],
            &[],
            &incoming,
        ));

        let incoming_supplemental = StreamGuardrailText {
            continuations: vec![StreamContinuation {
                key: "chat:0:content:first".to_owned(),
                family: "chat:0".to_owned(),
                identity: "content".to_owned(),
                identity_is_ambiguous: false,
                text: "safe".to_owned(),
            }],
            supplemental: (0..MAX_STREAM_GUARDRAIL_CHANNELS)
                .map(|index| format!("metadata-{index}"))
                .collect(),
            unevaluable: false,
            closed_prefixes: Vec::new(),
        };
        assert!(stream_continuation_would_exceed_cap(
            &[],
            &[],
            &[],
            &incoming_supplemental,
        ));
    }

    #[test]
    fn stream_guardrail_epoch_queue_is_bounded() {
        let mut candidate_epochs = Vec::new();
        let mut queued_candidates = 0;
        for index in 0..MAX_STREAM_GUARDRAIL_CHANNELS {
            assert!(try_queue_stream_guardrail_epoch(
                &mut candidate_epochs,
                &mut queued_candidates,
                vec![format!("candidate-{index}")],
            ));
        }
        assert!(!try_queue_stream_guardrail_epoch(
            &mut candidate_epochs,
            &mut queued_candidates,
            vec!["one-too-many".to_owned()],
        ));

        let mut empty_epochs = Vec::new();
        let mut queued_candidates = 0;
        for _ in 0..MAX_STREAM_GUARDRAIL_EPOCHS {
            assert!(try_queue_stream_guardrail_epoch(
                &mut empty_epochs,
                &mut queued_candidates,
                Vec::new(),
            ));
        }
        assert!(!try_queue_stream_guardrail_epoch(
            &mut empty_epochs,
            &mut queued_candidates,
            Vec::new(),
        ));
    }

    #[test]
    fn closed_response_items_still_count_until_a_successful_window_scan() {
        let mut continuations = Vec::new();
        let mut tails = Vec::new();
        let mut supplemental = Vec::new();
        let mut closed_prefixes = Vec::new();
        for index in 0..(MAX_STREAM_GUARDRAIL_CHANNELS / 2) {
            let delta = format!(
                "data: {{\"type\":\"response.output_text.delta\",\"item_id\":\"{index}\",\"output_index\":0,\"content_index\":0,\"delta\":\"safe\"}}\n\n"
            );
            let delta = stream_guardrail_text(
                PassthroughProtocol::OpenaiResponses,
                delta.as_bytes(),
                frame_parts(PassthroughProtocol::OpenaiResponses, delta.as_bytes())
                    .0
                    .scan,
            );
            append_stream_guardrail_text(
                &mut continuations,
                &mut tails,
                &mut supplemental,
                &mut closed_prefixes,
                &delta,
            );
            let done = format!(
                "data: {{\"type\":\"response.output_item.done\",\"item\":{{\"id\":\"{index}\"}}}}\n\n"
            );
            let done = stream_guardrail_text(
                PassthroughProtocol::OpenaiResponses,
                done.as_bytes(),
                String::new(),
            );
            append_stream_guardrail_text(
                &mut continuations,
                &mut tails,
                &mut supplemental,
                &mut closed_prefixes,
                &done,
            );
        }
        assert_eq!(continuations.len(), MAX_STREAM_GUARDRAIL_CHANNELS);
        assert_eq!(closed_prefixes.len(), MAX_STREAM_GUARDRAIL_CHANNELS / 2);
        let next = stream_guardrail_text(
            PassthroughProtocol::OpenaiResponses,
            b"data: {\"type\":\"response.output_text.delta\",\"item_id\":\"next\",\"output_index\":0,\"content_index\":0,\"delta\":\"safe\"}\n\n",
            "safe".to_owned(),
        );
        assert!(stream_continuation_would_exceed_cap(
            &continuations,
            &supplemental,
            &closed_prefixes,
            &next,
        ));
    }

    #[test]
    fn stream_guardrail_text_keeps_keyed_normal_forms_evaluable() {
        let cases = [
            (
                PassthroughProtocol::OpenaiChat,
                b"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ANTHROPIC_TEXT\"}}\n\n".as_slice(),
                "ANTHROPIC_TEXT",
            ),
            (
                PassthroughProtocol::OpenaiChat,
                b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":[{\"index\":0,\"type\":\"text\",\"text\":\"PART_TEXT\"}]}}]}\n\n".as_slice(),
                "PART_TEXT",
            ),
            (
                PassthroughProtocol::OpenaiChat,
                b"data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"TOOL_ARGS\"}}],\"content\":\"CHAT_TEXT\"}}]}\n\n".as_slice(),
                "TOOL_ARGS",
            ),
        ];
        for (protocol, frame, expected) in cases {
            let typed = frame_parts(protocol, frame).0.scan;
            let text = stream_guardrail_text(protocol, frame, typed);
            assert!(!text.unevaluable, "{frame:?}");
            let scanned = stream_guardrail_scan_text(&[], &text.continuations, &text.supplemental);
            assert!(scan_candidates_contain(&scanned, expected), "{scanned:?}");
        }

        let identityless_part = b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":[{\"type\":\"text\",\"text\":\"NO_ID\"}]}}]}\n\n";
        let typed = frame_parts(PassthroughProtocol::OpenaiChat, identityless_part)
            .0
            .scan;
        assert!(
            stream_guardrail_text(PassthroughProtocol::OpenaiChat, identityless_part, typed)
                .unevaluable
        );
    }

    #[test]
    fn chat_content_part_index_survives_optional_id_changes() {
        for (first_frame, second_frame) in [
            (
                b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":[{\"index\":0,\"type\":\"text\",\"text\":\"FOR\"}]}}]}\n\n".as_slice(),
                b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":[{\"index\":0,\"id\":\"later\",\"type\":\"text\",\"text\":\"BIDDEN\"}]}}]}\n\n".as_slice(),
            ),
            (
                b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":[{\"index\":0,\"id\":\"first\",\"type\":\"text\",\"text\":\"FOR\"}]}}]}\n\n".as_slice(),
                b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":[{\"index\":0,\"type\":\"text\",\"text\":\"BIDDEN\"}]}}]}\n\n".as_slice(),
            ),
        ] {
            let first = stream_guardrail_text(
                PassthroughProtocol::OpenaiChat,
                first_frame,
                frame_parts(PassthroughProtocol::OpenaiChat, first_frame).0.scan,
            );
            let second = stream_guardrail_text(
                PassthroughProtocol::OpenaiChat,
                second_frame,
                frame_parts(PassthroughProtocol::OpenaiChat, second_frame).0.scan,
            );
            assert!(!first.unevaluable);
            assert!(!second.unevaluable);
            let mut continuations = Vec::new();
            let mut tails = Vec::new();
            let mut supplemental = Vec::new();
            let mut closed_prefixes = Vec::new();
            append_stream_guardrail_text(
                &mut continuations,
                &mut tails,
                &mut supplemental,
                &mut closed_prefixes,
                &first,
            );
            append_stream_guardrail_text(
                &mut continuations,
                &mut tails,
                &mut supplemental,
                &mut closed_prefixes,
                &second,
            );
            assert!(scan_candidates_contain(
                &stream_guardrail_scan_text(&tails, &continuations, &supplemental),
                "FORBIDDEN",
            ));
        }
    }

    #[test]
    fn stream_continuity_never_joins_distinct_carriers() {
        let first = stream_guardrail_text(
            PassthroughProtocol::OpenaiResponses,
            b"data: {\"type\":\"response.output_text.delta\",\"item_id\":\"one\",\"output_index\":0,\"content_index\":0,\"delta\":\"FOR\"}\n\n",
            "FOR".to_owned(),
        );
        let second = stream_guardrail_text(
            PassthroughProtocol::OpenaiResponses,
            b"data: {\"type\":\"response.output_text.delta\",\"item_id\":\"two\",\"output_index\":0,\"content_index\":0,\"delta\":\"BIDDEN\"}\n\n",
            "BIDDEN".to_owned(),
        );
        let mut continuations = Vec::new();
        let mut tails = Vec::new();
        let mut supplemental = Vec::new();
        let mut closed_prefixes = Vec::new();
        append_stream_guardrail_text(
            &mut continuations,
            &mut tails,
            &mut supplemental,
            &mut closed_prefixes,
            &first,
        );
        append_stream_guardrail_text(
            &mut continuations,
            &mut tails,
            &mut supplemental,
            &mut closed_prefixes,
            &second,
        );
        let scanned = stream_guardrail_scan_text(&tails, &continuations, &supplemental);
        assert_eq!(scanned, vec!["FOR".to_owned(), "BIDDEN".to_owned()]);

        let scalar = stream_guardrail_text(
            PassthroughProtocol::OpenaiChat,
            b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"FOR\"}}]}\n\n",
            "FOR".to_owned(),
        );
        let indexed_part = stream_guardrail_text(
            PassthroughProtocol::OpenaiChat,
            b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":[{\"index\":0,\"type\":\"text\",\"text\":\"BIDDEN\"}]}}]}\n\n",
            "BIDDEN".to_owned(),
        );
        let mut scalar_continuations = Vec::new();
        let mut scalar_tails = Vec::new();
        let mut scalar_supplemental = Vec::new();
        let mut scalar_closed_prefixes = Vec::new();
        append_stream_guardrail_text(
            &mut scalar_continuations,
            &mut scalar_tails,
            &mut scalar_supplemental,
            &mut scalar_closed_prefixes,
            &scalar,
        );
        assert!(stream_continuation_identity_conflicts(
            &scalar_continuations,
            &scalar_closed_prefixes,
            &indexed_part,
        ));
    }

    #[test]
    fn responses_index_identity_cannot_switch_mid_stream() {
        let indexed = b"data: {\"type\":\"response.output_text.delta\",\"item_id\":\"one\",\"output_index\":0,\"content_index\":0,\"delta\":\"FOR\"}\n\n";
        let missing_indexes = b"data: {\"type\":\"response.output_text.delta\",\"item_id\":\"one\",\"delta\":\"BIDDEN\"}\n\n";
        for (first, second) in [
            (&indexed[..], &missing_indexes[..]),
            (&missing_indexes[..], &indexed[..]),
        ] {
            let first = stream_guardrail_text(
                PassthroughProtocol::OpenaiResponses,
                first,
                frame_parts(PassthroughProtocol::OpenaiResponses, first)
                    .0
                    .scan,
            );
            let second = stream_guardrail_text(
                PassthroughProtocol::OpenaiResponses,
                second,
                frame_parts(PassthroughProtocol::OpenaiResponses, second)
                    .0
                    .scan,
            );
            assert!(first.unevaluable || second.unevaluable);
        }
    }

    #[test]
    fn fail_open_unscannable_frame_seals_then_resets_guardrail_continuity() {
        let first_frame = b"data: \"FOR\"\n\n";
        let opaque_frame = b"data: {\"state\":\"safe\"}\n\n";
        let second_frame = b"data: \"BIDDEN\"\n\n";
        let first = stream_guardrail_text(
            PassthroughProtocol::Raw,
            first_frame,
            frame_parts(PassthroughProtocol::Raw, first_frame).0.scan,
        );
        let opaque = stream_guardrail_text(
            PassthroughProtocol::Raw,
            opaque_frame,
            frame_parts(PassthroughProtocol::Raw, opaque_frame).0.scan,
        );
        let second = stream_guardrail_text(
            PassthroughProtocol::Raw,
            second_frame,
            frame_parts(PassthroughProtocol::Raw, second_frame).0.scan,
        );
        assert!(!first.unevaluable);
        assert!(opaque.unevaluable);
        assert!(!second.unevaluable);

        let mut continuations = Vec::new();
        let mut tails = Vec::new();
        let mut supplemental = Vec::new();
        let mut closed_prefixes = Vec::new();
        let mut sealed_epochs = Vec::new();
        let mut queued_candidates = 0;
        append_stream_guardrail_text(
            &mut continuations,
            &mut tails,
            &mut supplemental,
            &mut closed_prefixes,
            &first,
        );
        assert!(seal_stream_guardrail_epoch(
            &mut sealed_epochs,
            &mut queued_candidates,
            &mut continuations,
            &mut tails,
            &mut supplemental,
            &mut closed_prefixes,
        ));
        append_stream_guardrail_text(
            &mut continuations,
            &mut tails,
            &mut supplemental,
            &mut closed_prefixes,
            &second,
        );
        let scanned = stream_guardrail_scan_text(&tails, &continuations, &supplemental);
        assert_eq!(sealed_epochs, vec![vec!["FOR".to_owned()]]);
        assert!(scan_candidates_contain(&scanned, "BIDDEN"), "{scanned:?}");
        assert!(
            !scan_candidates_contain(&scanned, "FORBIDDEN"),
            "{scanned:?}"
        );
    }

    #[test]
    fn responses_item_done_waits_for_a_successful_scan_before_retiring() {
        let delta = b"data: {\"type\":\"response.output_text.delta\",\"item_id\":\"one\",\"output_index\":0,\"content_index\":0,\"delta\":\"FORBIDDEN\"}\n\n";
        let done = b"data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"one\",\"type\":\"message\"}}\n\n";
        let delta = stream_guardrail_text(
            PassthroughProtocol::OpenaiResponses,
            delta,
            frame_parts(PassthroughProtocol::OpenaiResponses, delta)
                .0
                .scan,
        );
        let done = stream_guardrail_text(PassthroughProtocol::OpenaiResponses, done, String::new());
        assert!(!done.unevaluable);
        assert_eq!(done.closed_prefixes, vec!["responses:\"one\":".to_owned()]);

        let mut continuations = Vec::new();
        let mut tails = Vec::new();
        let mut supplemental = Vec::new();
        let mut closed_prefixes = Vec::new();
        append_stream_guardrail_text(
            &mut continuations,
            &mut tails,
            &mut supplemental,
            &mut closed_prefixes,
            &delta,
        );
        append_stream_guardrail_text(
            &mut continuations,
            &mut tails,
            &mut supplemental,
            &mut closed_prefixes,
            &done,
        );
        assert!(scan_candidates_contain(
            &stream_guardrail_scan_text(&tails, &continuations, &supplemental),
            "FORBIDDEN",
        ));
        retire_scanned_stream_continuations(&mut continuations, &mut tails, &mut closed_prefixes);
        assert!(continuations.is_empty());
        assert!(tails.is_empty());
        assert!(closed_prefixes.is_empty());

        let unknown_done =
            b"data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\"}}\n\n";
        let unknown = stream_guardrail_text(
            PassthroughProtocol::OpenaiResponses,
            unknown_done,
            String::new(),
        );
        assert!(!unknown.unevaluable);
        assert!(unknown.closed_prefixes.is_empty());

        let conflicting_done = b"data: {\"type\":\"response.output_item.done\",\"item_id\":\"one\",\"item\":{\"id\":\"two\",\"type\":\"message\"}}\n\n";
        assert!(
            stream_guardrail_text(
                PassthroughProtocol::OpenaiResponses,
                conflicting_done,
                String::new(),
            )
            .unevaluable
        );
    }

    #[test]
    fn done_sentinel_is_not_an_unevaluable_stream_carrier() {
        for protocol in [
            PassthroughProtocol::Raw,
            PassthroughProtocol::OpenaiChat,
            PassthroughProtocol::OpenaiCompletions,
            PassthroughProtocol::OpenaiResponses,
        ] {
            let text = stream_guardrail_text(protocol, b"data: [DONE]\n\n", String::new());
            assert!(!text.unevaluable, "{protocol:?}");
        }
    }

    #[test]
    fn stream_guardrail_text_scans_a_visible_carrier_once() {
        let email = "carol@example.com";
        let frame = b"data: {\"id\":\"chatcmpl-once\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ask carol@example.com\"}}]}\n\n";
        let typed = frame_parts(PassthroughProtocol::OpenaiChat, frame).0.scan;
        let text = stream_guardrail_text(PassthroughProtocol::OpenaiChat, frame, typed);
        let scanned = stream_guardrail_scan_text(&[], &text.continuations, &text.supplemental);
        assert_eq!(
            scanned
                .iter()
                .map(|candidate| candidate.matches(email).count())
                .sum::<usize>(),
            1,
            "typed, raw source, and supplemental channels must not multiply one carrier: {scanned:?}"
        );
    }

    #[test]
    fn stream_guardrail_text_excludes_unambiguous_anthropic_reasoning() {
        let frame = b"data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"thinking_delta\",\"text\":\"BLOCKME\"}}\n\n";
        let typed = frame_parts(PassthroughProtocol::OpenaiChat, frame).0.scan;
        assert!(typed.contains("BLOCKME"), "{typed:?}");
        let text = stream_guardrail_text(PassthroughProtocol::OpenaiChat, frame, typed);
        let scanned = stream_guardrail_scan_text(&[], &text.continuations, &text.supplemental);
        assert!(!scan_candidates_contain(&scanned, "BLOCKME"), "{scanned:?}");

        let signature = b"data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"signature_delta\",\"signature\":\"BLOCKME\"}}\n\n";
        let typed = frame_parts(PassthroughProtocol::OpenaiChat, signature)
            .0
            .scan;
        let text = stream_guardrail_text(PassthroughProtocol::OpenaiChat, signature, typed);
        let scanned = stream_guardrail_scan_text(&[], &text.continuations, &text.supplemental);
        assert!(!scan_candidates_contain(&scanned, "BLOCKME"), "{scanned:?}");
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
        assert!(
            request_guardrail_text(PassthroughProtocol::OpenaiResponses, req)
                .contains("list the files")
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
        assert!(
            response_guardrail_text(PassthroughProtocol::OpenaiResponses, resp).contains("done")
        );
        assert_eq!(
            response_usage(PassthroughProtocol::OpenaiResponses, None, resp),
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
            response_usage(PassthroughProtocol::OpenaiChat, None, openai),
            Some(usage_dims(5, 2))
        );
        let anthropicish = br#"{"usage":{"input_tokens":9,"output_tokens":4}}"#;
        assert_eq!(
            response_usage(PassthroughProtocol::OpenaiChat, None, anthropicish),
            Some(usage_dims(9, 4))
        );
        assert_eq!(response_usage(PassthroughProtocol::Raw, None, openai), None);
    }

    #[test]
    fn raw_usage_shapes_are_claimed_only_from_raw_bodies() {
        let shape = |b: &[u8]| detect_raw_usage_shape(detect_protocol(b), b);
        // The LLM envelopes keep their classification.
        let responses = br#"{"model":"m","input":"hi","documents":[],"query":"q"}"#;
        assert_eq!(
            detect_protocol(responses),
            PassthroughProtocol::OpenaiResponses
        );
        assert_eq!(shape(responses), None);
        let chat = br#"{"model":"m","messages":[],"input":{"x":1}}"#;
        assert_eq!(shape(chat), None);
        // Newly recognised among Raw bodies.
        let rerank = br#"{"model":"m","query":"q","documents":["a"]}"#;
        assert_eq!(shape(rerank), Some(RawUsageShape::Rerank));
        let native = br#"{"model":"m","input":{"contents":[{"text":"hi"}]}}"#;
        assert_eq!(shape(native), Some(RawUsageShape::DashscopeNative));
        // Still opaque.
        assert_eq!(shape(br#"{"model":"m","config_name":"x"}"#), None);
        assert_eq!(
            shape(br#"{"jsonrpc":"2.0","method":"m","params":{"input":{}}}"#),
            None
        );
    }

    #[test]
    fn dashscope_native_usage_counts_image_tokens_once() {
        let usage = |b: &[u8]| {
            response_usage(
                PassthroughProtocol::Raw,
                Some(RawUsageShape::DashscopeNative),
                b,
            )
        };
        // A reported total is recorded verbatim beside the derived prompt.
        let totalled = |prompt, completion, total| PassthroughUsage {
            upstream_total_tokens: Some(total),
            ..usage_dims(prompt, completion)
        };
        // Embedding: flat image tokens beside the text count, total reported.
        let flat = br#"{"usage":{"input_tokens":44,"image_tokens":64,"total_tokens":108}}"#;
        assert_eq!(usage(flat), Some(totalled(108, 0, 108)));
        // Generation: flat image tokens already inside input_tokens.
        let generation = br#"{"usage":{"input_tokens":79,"image_tokens":66,"input_tokens_details":{"image_tokens":66,"text_tokens":13},"output_tokens":14,"total_tokens":93}}"#;
        assert_eq!(usage(generation), Some(totalled(79, 14, 93)));
        let nested = br#"{"usage":{"input_tokens":903,"input_tokens_details":{"image_tokens":896,"text_tokens":7},"output_tokens":3,"total_tokens":906}}"#;
        assert_eq!(usage(nested), Some(totalled(903, 3, 906)));
        // No total: images counted beside the text.
        let no_total =
            br#"{"usage":{"duration":0,"image_count":1,"image_tokens":128,"input_tokens":5}}"#;
        assert_eq!(usage(no_total), Some(usage_dims(133, 0)));
        let total_only = br#"{"output":{"results":[]},"usage":{"total_tokens":29}}"#;
        assert_eq!(usage(total_only), Some(totalled(29, 0, 29)));
        // The generic dimensions ride along.
        let cached = br#"{"usage":{"input_tokens":120,"output_tokens":30,"total_tokens":150,"prompt_tokens_details":{"cached_tokens":100},"output_tokens_details":{"reasoning_tokens":20}}}"#;
        assert_eq!(
            usage(cached),
            Some(PassthroughUsage {
                cached_prompt_tokens: 100,
                reasoning_tokens: 20,
                ..totalled(120, 30, 150)
            })
        );
        assert_eq!(usage(br#"{"code":"InvalidParameter","message":"x"}"#), None);
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

    /// A `header_key` route names the slot its gateway credential
    /// arrives in, and the route schema forbids every name on the shared
    /// credential list — so the shared list can never cover it. A glob
    /// must not sweep it upstream, where the caller's AISIX key would be
    /// replayable against this gateway.
    #[tokio::test]
    async fn a_glob_never_sweeps_the_route_s_own_auth_header() {
        let (upstream, snap) = slot_route_fixture(serde_json::json!({
            "auth_mode": "header_key",
            "auth_header_name": "x-gw-key",
            "forward_client_headers": ["x-*"]
        }))
        .await;

        let resp = build_app(snap)
            .oneshot(slot_request(&[("x-gw-key", "sk-caller")]))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let received = &upstream.received_requests().await.unwrap()[0];
        assert!(
            received.headers.get("x-gw-key").is_none(),
            "`x-*` must not relay the slot this route authenticated the caller with"
        );
        // The SAME `x-*` recovers a stripped header that is not a slot,
        // so the assertion above is the rule firing rather than a pattern
        // that was never asked.
        assert_eq!(
            received.headers.get("x-stripped-control").unwrap(),
            "recovered"
        );
    }

    /// Naming it in full is still consent — the rule narrows how a slot
    /// is reached, never whether it can be.
    #[tokio::test]
    async fn the_route_s_own_auth_header_forwards_when_named_in_full() {
        let (upstream, snap) = slot_route_fixture(serde_json::json!({
            "auth_mode": "header_key",
            "auth_header_name": "x-gw-key",
            "forward_client_headers": ["x-gw-key"]
        }))
        .await;

        let resp = build_app(snap)
            .oneshot(slot_request(&[("x-gw-key", "sk-caller")]))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let received = &upstream.received_requests().await.unwrap()[0];
        assert_eq!(received.headers.get("x-gw-key").unwrap(), "sk-caller");
    }

    /// `identity_header`'s whole contract is that its value is recorded
    /// on the usage event and stripped before forwarding — a glob that
    /// put it back would make the promise false.
    #[tokio::test]
    async fn a_glob_never_sweeps_the_route_s_identity_header() {
        let (upstream, snap) = slot_route_fixture(serde_json::json!({
            "identity_header": "x-end-user",
            "forward_client_headers": ["x-*"]
        }))
        .await;

        let resp = build_app(snap)
            .oneshot(slot_request(&[
                ("authorization", "Bearer sk-caller"),
                ("x-end-user", "alice@example.com"),
            ]))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let received = &upstream.received_requests().await.unwrap()[0];
        assert!(received.headers.get("x-end-user").is_none());
        assert_eq!(
            received.headers.get("x-stripped-control").unwrap(),
            "recovered"
        );
    }

    #[tokio::test]
    async fn the_route_s_identity_header_forwards_when_named_in_full() {
        let (upstream, snap) = slot_route_fixture(serde_json::json!({
            "identity_header": "x-end-user",
            "forward_client_headers": ["x-end-user"]
        }))
        .await;

        let resp = build_app(snap)
            .oneshot(slot_request(&[
                ("authorization", "Bearer sk-caller"),
                ("x-end-user", "alice@example.com"),
            ]))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let received = &upstream.received_requests().await.unwrap()[0];
        assert_eq!(
            received.headers.get("x-end-user").unwrap(),
            "alice@example.com"
        );
    }

    /// `gateway_key` names no slot of its own — the schema forbids
    /// `auth_header_name` outside `header_key` — so nothing joins the
    /// exact-name set and a glob keeps meaning exactly what it did.
    /// (`anonymous` is the same shape and is covered end-to-end, where a
    /// real peer address can satisfy its `source_cidrs` gate.)
    #[tokio::test]
    async fn a_gateway_key_route_keeps_the_shared_rule_and_nothing_more() {
        let (upstream, snap) = slot_route_fixture(serde_json::json!({
            "auth_mode": "gateway_key",
            "forward_client_headers": ["x-*"]
        }))
        .await;

        let resp = build_app(snap)
            .oneshot(slot_request(&[
                ("authorization", "Bearer sk-caller"),
                ("x-gw-key", "not-a-slot-here"),
            ]))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let received = &upstream.received_requests().await.unwrap()[0];
        assert_eq!(
            received.headers.get("x-stripped-control").unwrap(),
            "recovered"
        );
        // `x-gw-key` is in this ProviderKey's strip set, so `forwards()`
        // IS asked about it here — and answers yes, because THIS route
        // declared no slot. That is what makes the narrowing per route
        // rather than a name added to the shared list: widen it to a
        // global and this assertion fails.
        assert_eq!(received.headers.get("x-gw-key").unwrap(), "not-a-slot-here");
        // And the shared rule is untouched: `x-*` never reached
        // `authorization`, so the ProviderKey's credential still rides
        // alone.
        let auths: Vec<_> = received.headers.get_all("authorization").iter().collect();
        assert_eq!(auths.len(), 1);
        assert_eq!(auths[0], "Bearer sk-upstream");
    }

    /// `passthrough_route` is the one surface that relays EVERY value of
    /// a repeated header — the other three collapse to the first — and
    /// its field description now promises that to users. The only thing
    /// keeping the promise is that this path walks the inbound map per
    /// value instead of per name, so collapsing it must go red here.
    #[tokio::test]
    async fn a_repeated_header_forwards_every_value() {
        let (upstream, snap) = slot_route_fixture(serde_json::json!({
            "forward_client_headers": ["x-*"]
        }))
        .await;

        // `x-stripped-control` is in the ProviderKey's strip set and
        // [`slot_request`] always sends one, so the second copy makes
        // this the STRIP-OVERRIDE path rather than the default-forward
        // one — the branch where a per-name decision would be easiest to
        // write and would silently drop a value.
        let resp = build_app(snap)
            .oneshot(slot_request(&[
                ("authorization", "Bearer sk-caller"),
                ("x-stripped-control", "second"),
            ]))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let received = &upstream.received_requests().await.unwrap()[0];
        let got: Vec<_> = received
            .headers
            .get_all("x-stripped-control")
            .iter()
            .map(|v| v.to_str().unwrap())
            .collect();
        assert_eq!(got, vec!["recovered", "second"]);
    }

    /// An `inject` route with the given overrides merged onto it. The
    /// upstream always answers `/v1/models`, and every request through
    /// [`slot_request`] carries an ordinary `x-other`, so each test above
    /// can tell "the rule fired" from "the pattern never matched".
    async fn slot_route_fixture(overrides: serde_json::Value) -> (MockServer, AisixSnapshot) {
        let upstream = MockServer::start().await;
        Mock::given(wm_method("GET"))
            .and(wm_path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"ok": true})))
            .mount(&upstream)
            .await;

        let mut json = serde_json::json!({
            "name": "slot-route",
            "path_prefix": "/passthrough/openai",
            "target_url": upstream.uri(),
            "provider_key_id": PK_ID
        });
        let map = json.as_object_mut().unwrap();
        for (k, v) in overrides.as_object().unwrap() {
            map.insert(k.clone(), v.clone());
        }

        // `strip_headers` names three `x-` headers, so `x-*` is asked
        // about all three and the CONTROL below is a real observation of
        // the glob firing. Without one in the strip set, a passthrough
        // route forwards it by default whatever the patterns say — an
        // assertion that proves nothing about this rule.
        let pk_json = r#"{"display_name":"openai-up","secret":"sk-upstream",
             "api_base":"http://unused","provider":"openai","adapter":"openai",
             "strip_headers":["authorization","x-api-key","x-gw-key","x-end-user",
                              "x-stripped-control"]}"#;
        let pk: ProviderKey = serde_json::from_str(pk_json).unwrap();

        let snap = AisixSnapshot::new();
        snap.provider_keys.insert(ResourceEntry::new(PK_ID, pk, 1));
        snap.apikeys.insert(apikey_entry("sk-caller", Some(&["*"])));
        snap.passthrough_routes
            .insert(route_entry("route-slot", json));
        (upstream, snap)
    }

    /// A caller request carrying `headers` plus the control header — an
    /// `x-` name the ProviderKey strips, so only a live `x-*` pattern
    /// puts it back on the wire.
    fn slot_request(headers: &[(&str, &str)]) -> Request<axum::body::Body> {
        let mut b = Request::builder()
            .method("GET")
            .uri("/passthrough/openai/v1/models")
            .header("x-stripped-control", "recovered");
        for (k, v) in headers {
            b = b.header(*k, *v);
        }
        b.body(axum::body::Body::empty()).unwrap()
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
        let mut s = SseFrameSplitter::with_max_frame_bytes(MAX_HELD_STREAM_BYTES);
        // Feed > MAX_HELD_STREAM_BYTES without a frame terminator: the
        // splitter must hand the oversized run on instead of buffering
        // without bound.
        let chunk = vec![b'x'; 256 * 1024];
        let mut emitted = 0usize;
        for _ in 0..8 {
            emitted += s
                .push(&chunk)
                .iter()
                .map(|frame| frame.bytes.len())
                .sum::<usize>();
        }
        assert!(
            emitted >= MAX_HELD_STREAM_BYTES,
            "oversized unterminated run must be flushed ({emitted} emitted)"
        );
        assert!(s.take_rest().len() <= MAX_HELD_STREAM_BYTES);
    }

    #[test]
    fn sse_splitter_honors_a_route_specific_frame_cap() {
        let mut s = SseFrameSplitter::with_max_frame_bytes(4);
        let frames = s.push(b"12345");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].bytes, b"12345");
        assert!(frames[0].overflowed);
        assert!(s.take_rest().is_empty());
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

    /// AISIX-Cloud#1330 / #1024: an input-guardrail block on a
    /// passthrough route leaves through `RouteError`, and the terminal
    /// event is built by the handler's failure branch — the only place a
    /// refused passthrough request appears in Logs at all.
    #[tokio::test]
    async fn blocked_request_names_the_policy_on_the_usage_event() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        Mock::given(wm_method("POST"))
            .and(wm_path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
            .expect(0)
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(provider_key_entry("http://unused"));
        snap.apikeys.insert(apikey_entry("sk-caller", Some(&["*"])));
        snap.passthrough_routes
            .insert(inject_route(&upstream.uri()));
        let g: aisix_core::Guardrail = serde_json::from_str(
            r#"{"name":"test-block","enabled":true,"hook_point":"input","fail_open":false,"kind":"keyword","patterns":[{"kind":"literal","value":"BLOCKME"}]}"#,
        )
        .unwrap();
        crate::seed_env_scoped_guardrail(&snap, ResourceEntry::new("g-1", g, 1));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let hub = Arc::new(Hub::new());
        let handle = SnapshotHandle::new(snap);
        let app = crate::build_router(
            crate::ProxyState::new(handle, hub, &cfg())
                .without_cache()
                .with_usage_sink(UsageSink::new(tx)),
        );

        let req = Request::builder()
            .method("POST")
            .uri("/passthrough/openai/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                serde_json::json!({"model": "x", "messages": [{"role": "user", "content": "please BLOCKME"}]})
                    .to_string(),
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);

        let ev = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("UsageEvent must be emitted for the refusal")
            .expect("usage_sink sender dropped");
        assert_eq!(ev.guardrail_enforced_hits.len(), 1, "{ev:?}");
        assert_eq!(ev.guardrail_enforced_hits[0].guardrail_name, "test-block");
        assert_eq!(ev.guardrail_enforced_hits[0].hook, "input");
        assert_eq!(ev.guardrail_enforced_hits[0].action, "blocked");
        let wire = serde_json::to_string(&ev).unwrap();
        assert!(!wire.contains("BLOCKME"), "{wire}");
    }

    /// Every protocol's stream frame yields its text and tool-call
    /// arguments for the scan, and its reasoning only for the cap (#513).
    #[test]
    fn frame_parts_split_scan_text_from_reasoning_per_protocol() {
        let parts = |p, f: &str| frame_parts(p, f.as_bytes()).0;
        let a = parts(
            PassthroughProtocol::OpenaiChat,
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
        );
        assert_eq!((a.scan.as_str(), a.reasoning), ("hi", 0));
        let a = parts(
            PassthroughProtocol::OpenaiChat,
            "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"q\\\"\"}}\n\n",
        );
        assert_eq!(a.scan, "{\"q\"");
        let a = parts(
            PassthroughProtocol::OpenaiChat,
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"hmm\"}}\n\n",
        );
        assert_eq!((a.scan.as_str(), a.reasoning), ("", 3));
        let c = parts(
            PassthroughProtocol::OpenaiChat,
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"why\",\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{}\"}}]}}]}\n\n",
        );
        assert_eq!((c.scan.as_str(), c.reasoning), ("{}", 3));
        let r = parts(
            PassthroughProtocol::OpenaiResponses,
            "data: {\"type\":\"response.function_call_arguments.delta\",\"delta\":\"{\\\"a\\\":1}\"}\n\n",
        );
        assert_eq!(r.scan, "{\"a\":1}");
        let r = parts(
            PassthroughProtocol::OpenaiResponses,
            "data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"plan\"}\n\n",
        );
        assert_eq!((r.scan.as_str(), r.reasoning), ("", 4));
        // A Raw payload without strings falls back to its full source text.
        let raw = parts(PassthroughProtocol::Raw, "data: {\"x\":1}\n\n");
        assert_eq!((raw.scan.as_str(), raw.held()), ("{\"x\":1}", 7));
        let frame = b"data: {\"state\":\"\\u0042LOCKME\",\"state\":\"clean\"}\n\n";
        let raw = frame_parts(PassthroughProtocol::Raw, frame).0;
        assert_eq!(raw.scan, "BLOCKME\nclean");
        assert_eq!(
            frame_capture_text(PassthroughProtocol::Raw, frame, &raw.scan),
            r#"{"state":"\u0042LOCKME","state":"clean"}"#
        );
        let deep = deeply_nested_escaped_block_json();
        let frame = format!("data: {}\n\n", String::from_utf8_lossy(&deep));
        assert!(
            frame_parts(PassthroughProtocol::Raw, frame.as_bytes())
                .0
                .scan
                .contains("BLOCKME"),
            "a valid deep Raw SSE payload must not fall back to escaped source"
        );
    }

    /// An Anthropic Messages body on the chat envelope is scanned in every
    /// slot the typed `/v1/messages` route scans.
    #[test]
    fn anthropic_request_blocks_and_system_are_request_scan_text() {
        let body = serde_json::json!({
            "model": "claude",
            "system": [{"type": "text", "text": "SYS"}],
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "THINK", "signature": "s"},
                    {"type": "tool_use", "id": "t", "name": "f", "input": {"q": "ARGS"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t", "content": [{"type": "text", "text": "RESULT"}]}
                ]}
            ]
        })
        .to_string();
        let text = request_guardrail_text(PassthroughProtocol::OpenaiChat, body.as_bytes());
        for slot in ["SYS", "THINK", "ARGS", "RESULT"] {
            assert!(text.contains(slot), "{slot} missing from {text}");
        }
    }

    #[test]
    fn nested_anthropic_tool_result_content_beyond_depth_cap_is_unevaluable() {
        let body = nested_anthropic_tool_result_request(crate::json_splice::MAX_JSON_DEPTH + 1);
        let error = try_request_guardrail_text(PassthroughProtocol::OpenaiChat, &body)
            .expect_err("nested tool results beyond the shared JSON depth cap must not recurse");
        assert!(error.is_depth_exceeded(), "{error}");
    }

    /// Buffered Anthropic and Responses replies are read slot by slot:
    /// text and tool input in, generated reasoning out.
    #[test]
    fn buffered_tool_calls_are_response_scan_text_and_reasoning_is_not() {
        let anthropic = serde_json::json!({
            "type": "message", "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "THINK"},
                {"type": "text", "text": "TEXT"},
                {"type": "tool_use", "id": "t", "name": "f", "input": {"q": "ARGS"}}
            ]
        })
        .to_string();
        let text = response_guardrail_text(PassthroughProtocol::OpenaiChat, anthropic.as_bytes());
        assert!(text.contains("TEXT") && text.contains("ARGS"));
        assert!(!text.contains("THINK"));
        let responses = serde_json::json!({
            "output": [{"type": "function_call", "name": "f", "arguments": "{\"q\":\"ARGS\"}"}]
        })
        .to_string();
        let text =
            response_guardrail_text(PassthroughProtocol::OpenaiResponses, responses.as_bytes());
        assert!(text.contains("ARGS"));
        let request = serde_json::json!({
            "input": [{"type": "function_call", "name": "f", "arguments": "{\"q\":\"ARGS\"}"}]
        })
        .to_string();
        let text = request_guardrail_text(PassthroughProtocol::OpenaiResponses, request.as_bytes());
        assert!(text.contains("ARGS"));
    }

    /// Relay `sse` through a passthrough route governed by `guardrail`, and
    /// return the `data:` payload of the refusal frame that ends it.
    async fn relayed_refusal_frame(sse: &str, guardrail: &str) -> serde_json::Value {
        let upstream = MockServer::start().await;
        Mock::given(wm_method("POST"))
            .and(wm_path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200).set_body_raw(sse.to_owned(), "text/event-stream"),
            )
            .mount(&upstream)
            .await;
        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(provider_key_entry("http://unused"));
        snap.apikeys.insert(apikey_entry("sk-caller", Some(&["*"])));
        snap.passthrough_routes
            .insert(inject_route(&upstream.uri()));
        let g: aisix_core::Guardrail = serde_json::from_str(guardrail).unwrap();
        crate::seed_env_scoped_guardrail(&snap, ResourceEntry::new("g-1", g, 1));
        let req = Request::builder()
            .method("POST")
            .uri("/passthrough/openai/v1/chat/completions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                serde_json::json!({"model": "x", "stream": true, "max_tokens": 64,
                    "messages": [{"role": "user", "content": "hi"}]})
                .to_string(),
            ))
            .unwrap();
        let resp = build_app(snap).oneshot(req).await.unwrap();
        let status = resp.status();
        let wire = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
        assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&wire));
        let wire = String::from_utf8_lossy(&wire).into_owned();
        let frame = wire
            .split("\n\n")
            .find(|f| f.starts_with("event: error\n"))
            .unwrap_or_else(|| panic!("no refusal frame on the wire: {wire}"));
        assert!(!wire.contains("FORBIDDEN"), "held content leaked: {wire}");
        serde_json::from_str(frame.trim_start_matches("event: error\ndata: ")).unwrap()
    }

    const OUTPUT_KEYWORD_BLOCK: &str = r#"{"name":"out-block","enabled":true,"hook_point":"output","kind":"keyword","patterns":[{"kind":"literal","value":"FORBIDDEN"}]}"#;
    const OUTPUT_CAP_FAIL_CLOSED: &str = r#"{"name":"out-cap","enabled":true,"hook_point":"output","kind":"azure_content_safety_text_moderation","endpoint":"http://127.0.0.1:1","api_key":"k","stream_processing_mode":"buffer_full","max_buffer_bytes":4,"on_buffer_exceeded":"fail_closed"}"#;

    const ANTHROPIC_SSE: &str = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude\",\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n\
event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"say FORBIDDEN now\"}}\n\n\
event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":3}}\n\n\
event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";

    const OPENAI_SSE: &str = "data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"say FORBIDDEN now\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
data: [DONE]\n\n";

    /// A refusal ending a relayed Anthropic Messages stream is the frame
    /// `/v1/messages` emits for it: an SDK-legal `error.type` and the
    /// refusal named on `error.code`.
    #[tokio::test]
    async fn a_relayed_anthropic_stream_is_refused_in_the_anthropic_shape() {
        let v = relayed_refusal_frame(ANTHROPIC_SSE, OUTPUT_KEYWORD_BLOCK).await;
        assert_eq!(v["type"], "error", "{v}");
        assert_eq!(v["error"]["type"], "invalid_request_error", "{v}");
        assert_eq!(v["error"]["code"], "content_filter", "{v}");
        assert!(v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("out-block"));

        let v = relayed_refusal_frame(ANTHROPIC_SSE, OUTPUT_CAP_FAIL_CLOSED).await;
        assert_eq!(v["error"]["type"], "invalid_request_error", "{v}");
        assert_eq!(v["error"]["code"], "guardrail_unavailable", "{v}");
    }

    /// On an OpenAI chat stream the refusal frame is the buffered 422 body:
    /// `guardrail_unavailable` on a fail-closed refusal, no code on a policy
    /// block.
    #[tokio::test]
    async fn a_relayed_openai_stream_is_refused_in_the_openai_shape() {
        let v = relayed_refusal_frame(OPENAI_SSE, OUTPUT_KEYWORD_BLOCK).await;
        assert_eq!(v["error"]["type"], "content_filter", "{v}");
        assert!(v["error"].get("code").is_none(), "{v}");
        assert!(v.get("type").is_none(), "{v}");

        let v = relayed_refusal_frame(OPENAI_SSE, OUTPUT_CAP_FAIL_CLOSED).await;
        assert_eq!(v["error"]["type"], "content_filter", "{v}");
        assert_eq!(v["error"]["code"], "guardrail_unavailable", "{v}");
    }
}
