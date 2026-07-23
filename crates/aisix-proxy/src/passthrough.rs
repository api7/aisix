//! `/passthrough/:provider/*rest` — raw provider pass-through.
//!
//! This endpoint proxies any HTTP method to the upstream provider's API
//! without modification, giving callers access to provider-specific endpoints
//! that the gateway does not natively handle (e.g. fine-tuning, batch
//! management, assistants, etc.).
//!
//! ## Routing
//!
//! The `provider` path segment names a configured Model (or matches a Model
//! whose name starts with the provider prefix). The gateway resolves the
//! `api_key` and `api_base` from the first Model found for that provider.
//!
//! ## Request transformation
//!
//! The request body and headers are forwarded verbatim — only the
//! `Authorization` header is replaced with the provider's key. The incoming
//! API key (proxy key) is stripped and never forwarded.
//!
//! ## Auth
//!
//! Standard proxy authentication applies (`Authorization: Bearer <key>` or
//! `x-api-key`). No model-level authorisation is enforced beyond that.

use aisix_obs::{AccessLog, RequestOutcome};
use axum::body::Body;
use axum::extract::{Path, Request, State};
use axum::http::{header, HeaderMap, HeaderValue, Method};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use std::time::{Duration, Instant};

use crate::auth::AuthenticatedKey;
use crate::error::ProxyError;
use crate::state::ProxyState;

/// Bounded `model` metric label for passthrough requests. The wildcard
/// `*rest` suffix is caller-controlled and must never be used directly as
/// a label (unbounded Prometheus cardinality — #451).
const PASSTHROUGH_MODEL_LABEL: &str = "passthrough";

/// Headers that the passthrough endpoint ALWAYS strips before
/// forwarding to upstream, regardless of customer configuration.
///
/// Two categories:
///   1. HTTP protocol metadata (`host`, `content-length`) — the
///      outbound HTTP client recomputes these based on the upstream
///      URL + body bytes.
///   2. RFC 7230 §6.1 hop-by-hop headers — by definition single-
///      hop, never legitimately forwarded.
///
/// Customer-configurable credential strips (`authorization`,
/// `cookie`, `set-cookie`, `x-api-key`) live on the ProviderKey's
/// `strip_headers` field — defaults set in
/// `aisix_core::default_strip_headers`. Per issue #411.
const ALWAYS_STRIP: &[&str] = &[
    "host",
    "content-length",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
];

/// Provider defaults indexed by provider-prefix string.
///
/// NOTE: `"cohere"` and `"jina"` are intentionally absent. #213
/// Phases 1–2 ship those providers on `/v1/rerank` only;
/// passthrough support is out of scope. Operators using
/// `/passthrough/cohere/*` or `/passthrough/jina/*` with the
/// matching provider Model must set `api_base` explicitly on the
/// ProviderKey. The fallback emits a 400
/// `InvalidRequest("no api_base configured for provider '<name>' and no default known")`
/// — graceful, not a crash. Tracked alongside #213's later phases.
fn default_base(provider_prefix: &str) -> Option<&'static str> {
    match provider_prefix {
        "openai" => Some("https://api.openai.com"),
        "anthropic" => Some("https://api.anthropic.com"),
        "google" => Some("https://generativelanguage.googleapis.com"),
        "deepseek" => Some("https://api.deepseek.com"),
        _ => None,
    }
}

/// `true` if `seg` is a strict api-version path component matching
/// `v\d+`: `v1`, `v2`, `v10` are accepted; `v2alpha`, `V1`, `v`,
/// non-ASCII digits are rejected. Used by [`strip_redundant_version_segment`]
/// to decide whether to dedup the leading version of `rest` against
/// the trailing version of `api_base`.
fn is_api_version_segment(seg: &str) -> bool {
    seg.starts_with('v') && seg.len() > 1 && seg[1..].chars().all(|c| c.is_ascii_digit())
}

/// Strip one leading api-version segment from `rest` when it
/// matches the trailing version segment of `base`. Returns `rest`
/// unchanged when no dedup applies.
///
/// See #164: the published examples in `docs/api-admin.md` §4.3
/// (api_base `https://api.openai.com/v1`) and `docs/api-proxy.md`
/// §4.10 (call `/passthrough/openai/v1/batches`) when followed
/// verbatim produce the malformed URL `.../v1/v1/batches`.
fn strip_redundant_version_segment<'a>(base: &str, rest: &'a str) -> &'a str {
    let base_tail = base.rsplit('/').next().unwrap_or("");
    if !is_api_version_segment(base_tail) {
        return rest;
    }
    // Only dedup when `rest`'s leading segment EXACTLY matches
    // `base_tail`. So `/v2` + `v1/foo` does NOT trigger (caller
    // asked for v1 explicitly); `/v1` + `v1/foo` does (the
    // duplicated `/v1` is the docs-example bug).
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

/// Wildcard handler mounted at `/passthrough/:provider/*rest`.
///
/// `method` is not a path parameter — axum merges all HTTP methods for wildcard
/// routes; we read it from the request.
pub async fn passthrough(
    State(state): State<ProxyState>,
    auth: AuthenticatedKey,
    client: crate::client_ip::ClientContext,
    Path((provider, rest)): Path<(String, String)>,
    req: Request,
) -> Response {
    let started = Instant::now();
    let request_id = client.request_id.clone();
    let api_key_id = auth.entry.id.clone();
    let method = req.method().clone();
    let path = format!("/passthrough/{provider}/{rest}");

    let mut monitor_hits: Vec<aisix_core::GuardrailMonitorHit> = Vec::new();
    match dispatch(
        state.clone(),
        &auth,
        &provider,
        &rest,
        req,
        &request_id,
        &client.source_ip,
        &mut monitor_hits,
    )
    .await
    {
        Ok((resp, provider_label, provider_key_id)) => {
            let elapsed = started.elapsed();
            let status = resp.status().as_u16();
            emit_access_log(
                &method,
                &path,
                &provider_label,
                &api_key_id,
                status,
                elapsed,
                &request_id,
            );
            state.metrics.record_request(
                &provider_label,
                // The raw `rest` wildcard is caller-controlled; using it as
                // the `model` label would create unbounded metric
                // cardinality. Passthrough has no resolved model, so record
                // a fixed sentinel (#451).
                PASSTHROUGH_MODEL_LABEL,
                status,
                RequestOutcome::from_status(status),
                elapsed,
            );
            // #699: record the passthrough call in the UsageEvent stream —
            // pre-fix it never appeared in /logs, the budget ledger or the
            // exporters. The raw tunnel parses nothing, so tokens stay zero;
            // the upstream's status is relayed verbatim and recorded as-is.
            emit_usage_event(
                &state,
                &request_id,
                &api_key_id,
                &provider_key_id,
                status,
                elapsed,
                &client,
                monitor_hits,
            );
            resp
        }
        Err(err) => {
            let status = err.status().as_u16();
            let elapsed = started.elapsed();
            emit_access_log(
                &method,
                &path,
                &provider,
                &api_key_id,
                status,
                elapsed,
                &request_id,
            );
            state.metrics.record_request(
                &provider,
                PASSTHROUGH_MODEL_LABEL,
                status,
                RequestOutcome::from_status(status),
                elapsed,
            );
            // #699 / #655 parity: surface the failed request in Logs with a
            // zero-token event (status + error class). No resolved model on
            // the error path -> empty requested_model, like audio's multipart
            // error path.
            crate::usage_attr::emit_error_usage_event(
                &state,
                "passthrough",
                "passthrough",
                &request_id,
                "",
                &api_key_id,
                status,
                err.kind(),
                &client,
            );
            err.into_response()
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn dispatch(
    state: ProxyState,
    auth: &AuthenticatedKey,
    provider: &str,
    rest: &str,
    req: Request,
    request_id: &str,
    source_ip: &str,
    monitor_hits_out: &mut Vec<aisix_core::GuardrailMonitorHit>,
) -> Result<(Response, String, String), ProxyError> {
    let snapshot = state.snapshot.load();

    // Find a model for this provider so we can borrow its provider_key.
    let provider_lower = provider.to_lowercase();
    let all_models = snapshot.models.entries();
    let matches_provider = |e: &aisix_core::resource::ResourceEntry<aisix_core::Model>| {
        e.value
            .provider
            .as_deref()
            .map(|p| p.eq_ignore_ascii_case(&provider_lower))
            .unwrap_or(false)
    };
    let provider_has_model = all_models.iter().any(|e| matches_provider(e));
    // Enforce the authenticated key's model ACL before lending this
    // provider's credentials through generic passthrough: pick the first
    // model of the provider the key is actually allowed to access.
    // Without this any valid key could reach any configured provider's
    // upstream credentials (#449). Mirrors the established gateway behavior,
    // which enforces the key's model access on passthrough when a target is
    // identifiable.
    let model_entry = all_models
        .into_iter()
        .find(|e| matches_provider(e) && auth.key().can_access(&e.value.display_name))
        .ok_or_else(|| {
            if provider_has_model {
                ProxyError::ModelForbidden(format!(
                    "api key is not authorized for any model of provider `{provider}`"
                ))
            } else {
                ProxyError::ModelNotFound(format!("no model found for provider `{provider}`"))
            }
        })?;

    let model = &model_entry.value;

    // Client-IP allowlist gate (#557/#697): the borrowed model's
    // `allowed_cidrs` applies to the raw tunnel too — pre-#697 passthrough
    // was the one surface that skipped it, so an operator's IP restriction
    // was bypassable by lending the same credentials through here. Same
    // borrowed-model basis as the #911 [6] guardrail resolution below.
    crate::dispatch::check_ip_access(model, source_ip)?;

    let pk_entry = crate::dispatch::resolve_provider_key(&snapshot, model)?;
    let api_key = crate::dispatch::require_api_key(&pk_entry.value, model)?.to_string();

    // #911 [6]: resolve the guardrail chain for the model whose credentials
    // this passthrough borrows, so the raw tunnel is subject to the same
    // content/DLP policy as the typed surfaces. Empty chain → no scan, no cost.
    let guardrail_ctx = aisix_guardrails::RequestContext {
        model_id: &model_entry.id,
        api_key_id: &auth.entry.id,
        team_id: auth.key().team_id.as_deref(),
    };
    let resolved_chain = state.guardrail_index.resolve(&guardrail_ctx);

    let base = match pk_entry.value.api_base.as_deref() {
        Some(b) if !b.trim().is_empty() => b.trim_end_matches('/').to_string(),
        _ => default_base(&provider_lower)
            .map(|s| s.to_string())
            .ok_or_else(|| {
                ProxyError::InvalidRequest(format!(
                    "no api_base configured for provider `{provider}` and no default known"
                ))
            })?,
    };

    // Build the target URL: {base}/{rest}.
    //
    // Per #164: when the configured `api_base` ends with an
    // api-version segment (e.g. `/v1`) AND the passthrough rest
    // path starts with the same segment, naive concatenation
    // produces a doubled prefix like `/v1/v1/files`. The published
    // examples in `docs/api-admin.md` §4.3 (api_base with `/v1`)
    // and `docs/api-proxy.md` §4.10 (rest with `/v1/...`)
    // together hit this case.
    //
    // Strip the redundant leading version segment from `rest` ONLY
    // when it exactly matches `api_base`'s trailing version
    // segment. Constraining the dedup to api-version-shaped
    // segments (`v\d+`) prevents false dedups on paths like
    // `/v1/files/v1/foo` where the trailing `v1` is a genuine
    // path component, not a version prefix.
    let rest_after_dedup = strip_redundant_version_segment(&base, rest);
    let url = if rest_after_dedup.is_empty() {
        base.clone()
    } else {
        format!("{base}/{rest_after_dedup}")
    };

    // Preserve the query string.
    let url = if let Some(q) = req.uri().query() {
        format!("{url}?{q}")
    } else {
        url
    };

    let method = req.method().clone();
    let incoming_headers = req.headers().clone();
    // Use the configured cap (not a hard-coded 10 MiB) so an
    // operator who raises `request_body_limit_bytes` for, e.g.,
    // audio passthrough actually gets the larger limit here too.
    // Map an oversize-body read failure to the typed
    // `RequestTooLarge` envelope so callers see a proper 413 rather
    // than a misleading 400 "failed to read body". The
    // `enforce_request_body_limit` middleware short-circuits the
    // Content-Length-known case ahead of this; this map handles the
    // chunked / no-Content-Length / Content-Length-lying case once
    // the actual byte count exceeds the cap.
    let body_limit = state.request_body_limit_bytes;
    let body_bytes: Bytes = axum::body::to_bytes(req.into_body(), body_limit)
        .await
        .map_err(|_| ProxyError::RequestTooLarge {
            limit_bytes: body_limit,
        })?;

    // #911 [6]: run INPUT guardrails on the passthrough request body BEFORE it
    // reaches the upstream. The tunnel forwards arbitrary provider endpoints
    // verbatim, so a content/DLP block enforced on the typed surfaces was
    // bypassable here. Following LiteLLM's passthrough default, scan the whole
    // body as one text blob (UTF-8 lossy so binary bodies degrade to
    // replacement chars rather than being skipped).
    if !resolved_chain.is_empty() {
        let chat = aisix_gateway::ChatFormat::new(
            &model_entry.value.display_name,
            vec![aisix_gateway::ChatMessage::user(
                String::from_utf8_lossy(&body_bytes).into_owned(),
            )],
        );
        let (verdict, hits) =
            aisix_guardrails::Guardrail::check_input_observed(&resolved_chain, &chat).await;
        monitor_hits_out.extend(hits);
        if let aisix_guardrails::GuardrailVerdict::Block {
            reason,
            guardrail_name,
        } = verdict
        {
            // Per #153 the matched-pattern detail stays in ops logs only.
            tracing::warn!(
                guardrail_hook = "input",
                provider = %provider_lower,
                reason = %reason,
                "guardrail blocked passthrough request",
            );
            return Err(ProxyError::ContentFiltered(
                crate::error::guardrail_block_message("request", guardrail_name.as_deref()),
            ));
        }
    }

    // Reserve the rate-limit layers AFTER the input guardrail so a content
    // block doesn't burn an RPM slot, matching the typed endpoints.
    //
    // api7/AISIX-Cloud#1116: provider-native JSON envelopes carry the target
    // model in a top-level `model` field (the shape shared by OpenAI-compatible
    // and DashScope-style bodies). When that name is a configured Model of the
    // addressed provider, reserve its model-level layers (inline `rate_limit`
    // + `model`-scope policies) exactly like the typed endpoints — pre-fix the
    // raw tunnel skipped them entirely, so e.g. a video model reachable only
    // through passthrough had no enforceable model rate limit anywhere.
    // Non-JSON bodies, bodies without a `model` field, and unregistered names
    // keep the previous behavior: request-level layers only. The tunnel never
    // parses usage (tokens stay 0), so only the request-count dimensions
    // (rps/rpm/rph) ever draw from the model buckets here.
    let model_rl = body_model_rate_limit(&snapshot, &provider_lower, &body_bytes);
    let _reservation = crate::quota::enforce(&state, auth, model_rl.as_ref()).await?;

    let client = crate::http_client::client();
    let mut builder = client.request(method.clone(), &url);

    // Inject upstream Authorization; strip the incoming proxy auth.
    //
    // Per #166: Anthropic's documented auth shape is `x-api-key` +
    // `anthropic-version`, NOT `Authorization: Bearer`. Sending both
    // is non-spec wire shape (real Anthropic ignores Bearer today,
    // but operators inspecting upstream traffic captures generate
    // "is this a leak?" tickets every time the redundant header
    // surfaces). Per docs/api-proxy.md §4.10 the gateway is the
    // entity choosing the auth shape per provider; pick exactly one
    // shape per provider.
    //
    // - openai / gemini / deepseek (and any unknown provider that
    //   reuses the OpenAI-compat shape): `Authorization: Bearer …`
    // - anthropic: `x-api-key` + `anthropic-version` only
    //
    // Order matters: strip inbound headers FIRST (per ALWAYS_STRIP +
    // pk.strip_headers), THEN set the gateway's own auth. If we did
    // the reverse, `reqwest::RequestBuilder::header()` would `append`
    // a second value to `Authorization` when the strip list doesn't
    // include `authorization` (the customer-elected override case);
    // the upstream would receive `Authorization: Bearer <gw>, Bearer
    // <client>` on the wire, leaking the client's credential
    // regardless of intent. Strip-then-set keeps the wire single-
    // valued in the default case; in the override case the client's
    // value reaches upstream, which is the documented opt-in cost.
    //
    // Strip rules:
    //   1. ALWAYS_STRIP — HTTP protocol metadata + RFC 7230 §6.1
    //      hop-by-hop headers. Non-configurable; stripping these
    //      is required for protocol correctness.
    //   2. provider_key.strip_headers — customer-configurable
    //      strip list (per ProviderKey, default: authorization,
    //      cookie, set-cookie, x-api-key). See issue #411.
    //
    // Case-insensitive comparison; build a lowercased HashSet once
    // per request to avoid O(N*M) scans on large header lists.
    let strip_set: std::collections::HashSet<String> = pk_entry
        .value
        .strip_headers
        .iter()
        .map(|s| s.to_ascii_lowercase())
        .chain(ALWAYS_STRIP.iter().map(|s| (*s).to_string()))
        .collect();

    for (name, value) in &incoming_headers {
        let n = name.as_str().to_ascii_lowercase();
        if strip_set.contains(&n) {
            continue;
        }
        builder = builder.header(name, value);
    }

    // Gateway's own auth — set AFTER the strip loop. This guarantees
    // the upstream sees exactly one `Authorization` (or `x-api-key`
    // + `anthropic-version`) line in the default case, even when
    // the client sent one of those headers — the client's value
    // was filtered out in the loop above.
    if api_key.is_empty() {
        // Provider key has no secret configured. Nothing to inject —
        // explicit blank-Authorization rather than fall-through-to-
        // Bearer-of-empty-string keeps the wire clean.
    } else if provider_lower == "anthropic" {
        builder = builder.header("x-api-key", &api_key);
        builder = builder.header("anthropic-version", "2023-06-01");
    } else {
        builder = builder.header(header::AUTHORIZATION, format!("Bearer {api_key}"));
    }

    builder = builder.header("x-aisix-request-id", request_id);

    if !body_bytes.is_empty() {
        builder = builder.body(body_bytes);
    }

    // #554/#911: bound the raw tunnel by the selected model's E2E request
    // timeout, matching the first-class non-streaming paths. Without it a
    // slow/blackholed upstream could pin a passthrough connection open
    // indefinitely regardless of the model's configured timeout.
    if let Some(d) = model.request_timeout() {
        builder = builder.timeout(d);
    }

    // #701: transport/decode failures against the shared upstream mark the
    // borrowed model's runtime status (same borrowed-model basis as the
    // #911 [6] guardrail chain), so a dead upstream reached only via the raw
    // tunnel still trips the cooldown. Forwarded HTTP statuses stay the
    // caller's business — the tunnel relays them verbatim.
    let upstream_resp = builder
        .send()
        .await
        .map_err(|e| {
            crate::cooldown::note_failure(
                &state.runtime_status,
                &model_entry.id,
                model.cooldown.as_ref(),
                aisix_gateway::BridgeError::Transport(aisix_gateway::transport_error_message(&e)),
            )
        })
        .map_err(ProxyError::Bridge)?;

    let status = upstream_resp.status();
    let resp_headers = upstream_resp.headers().clone();
    let resp_body = upstream_resp
        .bytes()
        .await
        .map_err(|e| {
            crate::cooldown::note_failure(
                &state.runtime_status,
                &model_entry.id,
                model.cooldown.as_ref(),
                aisix_gateway::BridgeError::UpstreamDecode(e.to_string()),
            )
        })
        .map_err(ProxyError::Bridge)?;

    // #911 [6]: run OUTPUT guardrails on the passthrough response body — the
    // same whole-body text scan as the input hook, so forbidden model output
    // can't be exfiltrated through the raw tunnel.
    if !resolved_chain.is_empty() {
        let synth = aisix_gateway::ChatResponse {
            id: String::new(),
            model: model_entry.value.display_name.clone(),
            message: aisix_gateway::ChatMessage::assistant(
                String::from_utf8_lossy(&resp_body).into_owned(),
            ),
            finish_reason: aisix_gateway::FinishReason::Stop,
            usage: aisix_gateway::UsageStats::default(),
        };
        let (verdict, hits) =
            aisix_guardrails::Guardrail::check_output_observed(&resolved_chain, &synth).await;
        monitor_hits_out.extend(hits);
        if let aisix_guardrails::GuardrailVerdict::Block {
            reason,
            guardrail_name,
        } = verdict
        {
            // Per #153 the matched-pattern detail stays in ops logs only.
            tracing::warn!(
                guardrail_hook = "output",
                provider = %provider_lower,
                reason = %reason,
                "guardrail blocked passthrough response",
            );
            return Err(ProxyError::ContentFiltered(
                crate::error::guardrail_block_message("response", guardrail_name.as_deref()),
            ));
        }
    }

    let mut response = Response::builder()
        .status(status)
        .body(Body::from(resp_body))
        .unwrap();

    // Copy relevant response headers.
    copy_safe_headers(&resp_headers, response.headers_mut());

    if let Ok(hv) = HeaderValue::from_str(request_id) {
        response.headers_mut().insert(
            axum::http::header::HeaderName::from_static("x-aisix-request-id"),
            hv,
        );
    }

    Ok((response, provider_lower, pk_entry.id.to_string()))
}

/// Model-level rate-limit identity for a passthrough request, resolved from
/// the JSON body's top-level `model` field (api7/AISIX-Cloud#1116).
///
/// The name is matched against the addressed provider's configured Models
/// twice over: an exact `display_name` hit first (the gateway alias), then
/// the provider-native `model_name` — the tunnel forwards bodies verbatim,
/// so callers typically name the upstream id, not the gateway alias. Either
/// way the reservation is keyed by the entry's `display_name`, so tunnel
/// and typed traffic to the same Model draw from one bucket.
///
/// Returns `None` when the body is not JSON, carries no string `model`
/// field, or nothing matches within the addressed provider — the request
/// then reserves only the request-level layers, mirroring the pre-fix
/// behavior. A same-named Model of a different provider never matches.
/// Wildcard (`*`) display names are a typed-endpoint resolution feature,
/// not replicated here.
fn body_model_rate_limit(
    snapshot: &aisix_core::AisixSnapshot,
    provider_lower: &str,
    body: &[u8],
) -> Option<crate::quota::ModelRateLimit> {
    // Field probe, not a full `Value` DOM: unknown fields are skipped
    // without allocating, so a large tunnel body costs one `String` here,
    // not a parsed copy of itself. A non-string `model` fails the whole
    // deserialize and lands on the same `None` fallback.
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
            // Provider-native id fallback. `min_by_key` keeps the pick
            // deterministic when several Models pin the same upstream id.
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

/// #699: push one zero-token `UsageEvent` per passthrough request onto the
/// CP sink and the exporter fan-out. The raw tunnel parses neither request
/// nor response, so there are no tokens or model fields — the event records
/// who lent which provider's credentials (per-PK attribution tags), the
/// relayed status, and the latency. `inbound_protocol = "passthrough"`
/// distinguishes these rows from typed-endpoint traffic in /logs.
#[allow(clippy::too_many_arguments)]
fn emit_usage_event(
    state: &ProxyState,
    request_id: &str,
    api_key_id: &str,
    provider_key_id: &str,
    status_code: u16,
    elapsed: Duration,
    client: &crate::client_ip::ClientContext,
    guardrail_monitor_hits: Vec<aisix_core::GuardrailMonitorHit>,
) {
    let snap = state.snapshot.load();
    let mut event = aisix_obs::UsageEvent {
        request_id: request_id.to_string(),
        occurred_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        api_key_id: api_key_id.to_string(),
        status_code,
        latency_ms: elapsed.as_millis().min(u32::MAX as u128) as u32,
        inbound_protocol: "passthrough".to_string(),
        client_source_ip: client.source_ip.clone(),
        client_user_agent: client.user_agent.clone(),
        guardrail_monitor_hits,
        ..Default::default()
    };
    crate::usage_attr::apply_pk_telemetry(&mut event, &snap, provider_key_id);
    state.usage_sink.try_emit("passthrough", event.clone());
    let exporters = snap.observability_exporters.entries();
    state
        .otlp_fan_out
        .fan_out(&event, None, exporters.iter().map(|e| &e.value));
}

/// Copy response headers that are safe to relay to the downstream caller.
fn copy_safe_headers(src: &HeaderMap, dst: &mut HeaderMap) {
    for (name, value) in src {
        let n = name.as_str().to_lowercase();
        // Skip hop-by-hop headers.
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
        dst.insert(name.clone(), value.clone());
    }
}

fn emit_access_log(
    method: &Method,
    path: &str,
    provider: &str,
    api_key_id: &str,
    status: u16,
    elapsed: Duration,
    request_id: &str,
) {
    AccessLog {
        method: method.as_str(),
        path,
        status,
        latency: elapsed,
        provider: Some(provider),
        model: None,
        api_key_id: Some(api_key_id),
        prompt_tokens: None,
        completion_tokens: None,
        total_tokens: None,
        request_id,
        served_by_model: None,
        routing_attempt_count: None,
        routing_fallback_count: None,
    }
    .emit();
}

#[cfg(test)]
mod tests {
    use super::{is_api_version_segment, strip_redundant_version_segment};
    use aisix_core::resource::ResourceEntry;
    use aisix_core::snapshot::SnapshotHandle;
    use aisix_core::{AisixSnapshot, ApiKey, Model, ProxyConfig};
    use aisix_gateway::Hub;
    use axum::body::to_bytes;
    use axum::http::{Request, StatusCode};
    use std::sync::Arc;
    use tower::ServiceExt;
    use wiremock::matchers::{method as wm_method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn is_api_version_segment_recognises_canonical_v_prefix_versions() {
        assert!(is_api_version_segment("v1"));
        assert!(is_api_version_segment("v2"));
        assert!(is_api_version_segment("v10"));
        // Reject look-alikes that would be unsafe to dedup on.
        assert!(!is_api_version_segment("v")); // bare 'v'
        assert!(!is_api_version_segment("v1alpha")); // mixed
        assert!(!is_api_version_segment("V1")); // case-sensitive: gateway sees lowercased URLs
        assert!(!is_api_version_segment("messages")); // ordinary path component
        assert!(!is_api_version_segment("")); // empty
    }

    /// Issue #164: api_base ending in /v1 plus rest starting with v1/
    /// must dedup to a single /v1 prefix. The published examples in
    /// docs/api-admin.md §4.3 (api_base with /v1) and docs/api-proxy.md
    /// §4.10 (call /passthrough/openai/v1/batches) follow this exact
    /// pattern; pre-fix the gateway concatenated to /v1/v1/batches
    /// which the real OpenAI API would 404.
    #[test]
    fn strip_redundant_version_segment_dedups_canonical_docs_example() {
        // The exact docs-example case.
        assert_eq!(
            strip_redundant_version_segment("https://api.openai.com/v1", "v1/batches"),
            "batches"
        );
        // Trailing-slash on `rest` (axum sometimes preserves it).
        assert_eq!(
            strip_redundant_version_segment("https://api.openai.com/v1", "v1/files/"),
            "files/"
        );
        // `rest` exactly equals the trailing version.
        assert_eq!(
            strip_redundant_version_segment("https://api.openai.com/v1", "v1"),
            ""
        );
    }

    #[test]
    fn strip_redundant_version_segment_does_not_touch_non_version_tails() {
        // No version on api_base → no dedup.
        assert_eq!(
            strip_redundant_version_segment("https://api.openai.com", "v1/files"),
            "v1/files"
        );
        // Tail isn't a version segment → no dedup.
        assert_eq!(
            strip_redundant_version_segment("https://api.openai.com/api", "v1/files"),
            "v1/files"
        );
    }

    #[test]
    fn strip_redundant_version_segment_only_strips_leading_match() {
        // Trailing /v1 in `rest` is a genuine path component
        // (not a version prefix), MUST NOT be touched.
        assert_eq!(
            strip_redundant_version_segment("https://api.openai.com/v1", "v1/files/v1/foo"),
            "files/v1/foo"
        );
    }

    #[test]
    fn strip_redundant_version_segment_only_dedups_exact_version_match() {
        // api_base /v2 + rest v1/foo → caller asked for v1
        // explicitly; do NOT dedup (would silently rewrite the call
        // to a different version).
        assert_eq!(
            strip_redundant_version_segment("https://api.openai.com/v2", "v1/foo"),
            "v1/foo"
        );
        // api_base /v1 + rest v2/foo → caller asked for v2, do NOT
        // dedup.
        assert_eq!(
            strip_redundant_version_segment("https://api.openai.com/v1", "v2/foo"),
            "v2/foo"
        );
    }

    #[test]
    fn strip_redundant_version_segment_handles_empty_rest() {
        assert_eq!(
            strip_redundant_version_segment("https://api.openai.com/v1", ""),
            ""
        );
    }

    fn cfg() -> ProxyConfig {
        ProxyConfig {
            addr: "127.0.0.1:0".into(),
            request_body_limit_bytes: 1_048_576,
            real_ip: Default::default(),
            tls: None,
        }
    }

    const PK_ID: &str = "11111111-1111-1111-1111-111111111111";

    fn openai_model(name: &str) -> ResourceEntry<Model> {
        let json = format!(
            r#"{{"display_name":"{name}","provider":"openai","model_name":"gpt-4o","provider_key_id":"{PK_ID}"}}"#
        );
        let m: Model = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("m-1", m, 1)
    }

    fn anthropic_model(name: &str) -> ResourceEntry<Model> {
        let json = format!(
            r#"{{"display_name":"{name}","provider":"anthropic","model_name":"claude-3-5-haiku-20241022","provider_key_id":"{PK_ID}"}}"#
        );
        let m: Model = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("m-1", m, 1)
    }

    fn provider_key_entry(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        let json = format!(
            r#"{{"display_name":"openai-up","secret":"sk-test","api_base":"{api_base}","provider":"openai","adapter":"openai"}}"#
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&json).unwrap();
        ResourceEntry::new(PK_ID, pk, 1)
    }

    fn anthropic_provider_key_entry(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        let json = format!(
            r#"{{"display_name":"anthropic-up","secret":"sk-ant-test","api_base":"{api_base}","provider":"anthropic","adapter":"anthropic"}}"#
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&json).unwrap();
        ResourceEntry::new(PK_ID, pk, 1)
    }

    fn new_snap(api_base: &str) -> AisixSnapshot {
        let snap = AisixSnapshot::new();
        snap.provider_keys.insert(provider_key_entry(api_base));
        snap
    }

    fn apikey_entry(allowed: &[&str]) -> ResourceEntry<ApiKey> {
        let json = format!(
            r#"{{"key_hash":"8b6712790a2089c67aa97a2d80022df18cc65c7814350e33baebe79aab508891","allowed_models":{}}}"#,
            serde_json::to_string(&allowed).unwrap()
        );
        let k: ApiKey = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("k-1", k, 1)
    }

    fn build_app(snap: AisixSnapshot) -> axum::Router {
        let hub = Arc::new(Hub::new());
        let handle = SnapshotHandle::new(snap);
        crate::build_router(crate::ProxyState::new(handle, hub, &cfg()).without_cache())
    }

    #[tokio::test]
    async fn unauthenticated_returns_401() {
        let snap = new_snap("http://unused");
        let app = build_app(snap);

        let req = Request::builder()
            .method("GET")
            .uri("/passthrough/openai/v1/models")
            .body(axum::body::Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn unknown_provider_returns_404() {
        let snap = new_snap("http://unused");
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let req = Request::builder()
            .method("GET")
            .uri("/passthrough/cohere/v1/embed")
            .header("authorization", "Bearer sk-caller")
            .body(axum::body::Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn happy_path_forwards_to_upstream() {
        let upstream = MockServer::start().await;
        Mock::given(wm_method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "object": "list",
                "data": []
            })))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let req = Request::builder()
            .method("GET")
            .uri("/passthrough/openai/v1/models")
            .header("authorization", "Bearer sk-caller")
            .body(axum::body::Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"], "list");
    }

    /// Issue #166 regression: Anthropic passthrough must inject ONLY
    /// `x-api-key` + `anthropic-version`, NEVER `Authorization:
    /// Bearer …` alongside. Pre-fix the gateway emitted both — a
    /// non-spec wire shape that real Anthropic ignores today but
    /// stricter middleware (or future Anthropic gateways) could
    /// reject. The wiremock matchers used here fail the request
    /// match if the headers don't agree; an extra Authorization
    /// header would not violate the matcher (matchers are subset),
    /// so we additionally assert via `received_requests`.
    #[tokio::test]
    async fn anthropic_passthrough_does_not_inject_bearer_auth() {
        use wiremock::matchers::header;

        let upstream = MockServer::start().await;
        Mock::given(wm_method("POST"))
            .and(path("/v1/messages/batches"))
            .and(header("x-api-key", "sk-ant-test"))
            .and(header("anthropic-version", "2023-06-01"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "msgbatch-1",
                "type": "message_batch"
            })))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(anthropic_provider_key_entry(&upstream.uri()));
        snap.models.insert(anthropic_model("claude-3-5-haiku"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let req = Request::builder()
            .method("POST")
            .uri("/passthrough/anthropic/v1/messages/batches")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(r#"{"requests":[]}"#))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Strict check on the upstream-side headers: NO Authorization
        // header at all. wiremock's `matchers::header` only enforces
        // a subset, so we also drain `received_requests` and assert
        // the absence of the redundant header.
        let received = upstream.received_requests().await.unwrap();
        assert_eq!(received.len(), 1, "exactly one upstream call expected");
        let upstream_headers = &received[0].headers;
        assert!(
            !upstream_headers.contains_key("authorization"),
            "Anthropic passthrough must NOT inject Authorization (per #166); got headers: {:?}",
            upstream_headers
                .iter()
                .map(|(k, v)| (k.as_str(), v.to_str().unwrap_or("<binary>")))
                .collect::<Vec<_>>()
        );
        // Sanity: the documented Anthropic auth shape is on the wire.
        assert_eq!(
            upstream_headers
                .get("x-api-key")
                .and_then(|v| v.to_str().ok()),
            Some("sk-ant-test")
        );
        assert_eq!(
            upstream_headers
                .get("anthropic-version")
                .and_then(|v| v.to_str().ok()),
            Some("2023-06-01")
        );
    }

    #[tokio::test]
    async fn upstream_non_200_is_relayed_verbatim() {
        let upstream = MockServer::start().await;
        Mock::given(wm_method("POST"))
            .and(path("/v1/fine_tuning/jobs"))
            .respond_with(ResponseTemplate::new(422).set_body_json(serde_json::json!({
                "error": {"code": "validation_error", "message": "invalid file_id"}
            })))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let req = Request::builder()
            .method("POST")
            .uri("/passthrough/openai/v1/fine_tuning/jobs")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                r#"{"training_file":"file-xyz","model":"gpt-4o"}"#,
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        // 422 from upstream is relayed as-is (not remapped to 502).
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    // ---- Issue #411: header-strip policy --------------------------
    //
    // Inbound headers must be filtered through:
    //   1. ALWAYS_STRIP — RFC 7230 §6.1 hop-by-hop + protocol
    //      metadata. Non-configurable.
    //   2. provider_key.strip_headers — per-PK configurable list.
    //      Defaults to the 4 canonical credentials.
    //
    // These tests pin the wire contract upstream observes.

    /// Builds a PK with a caller-supplied `strip_headers` value
    /// (overriding the serde default of 4 credentials).
    fn provider_key_entry_with_strip(
        api_base: &str,
        strip_headers: &[&str],
    ) -> ResourceEntry<aisix_core::ProviderKey> {
        let strip_json = serde_json::to_string(strip_headers).unwrap();
        let json = format!(
            r#"{{"display_name":"openai-up","secret":"sk-test","api_base":"{api_base}","provider":"openai","adapter":"openai","strip_headers":{strip_json}}}"#
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&json).unwrap();
        ResourceEntry::new(PK_ID, pk, 1)
    }

    /// Helper: returns all values associated with `header_name` in the
    /// upstream's received request. The mock server uses an
    /// http::HeaderMap; `get` only returns the first value when a
    /// header has multiple. To verify "client credential didn't
    /// leak", we must scan all values (including the gateway's own
    /// injection) — anything matching the client's input is a leak.
    fn upstream_header_values<'a>(
        received: &'a wiremock::Request,
        header_name: &str,
    ) -> Vec<&'a str> {
        received
            .headers
            .get_all(header_name)
            .iter()
            .filter_map(|v| v.to_str().ok())
            .collect()
    }

    #[tokio::test]
    async fn default_strip_blocks_client_credentials_leak() {
        let upstream = MockServer::start().await;
        Mock::given(wm_method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
            .mount(&upstream)
            .await;

        // new_snap() uses provider_key_entry() which doesn't set
        // strip_headers → serde default fills in the 4 credentials.
        let snap = new_snap(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let req = Request::builder()
            .method("GET")
            .uri("/passthrough/openai/v1/models")
            // Authorization is BOTH the gateway-auth credential AND
            // a credential that must not leak to upstream. We have
            // to send the valid gateway key here (so AuthenticatedKey
            // succeeds), then assert upstream sees the PK secret
            // (`sk-test`) instead of the client value (`sk-caller`).
            .header("authorization", "Bearer sk-caller")
            // Cookie: client-side leak canary. Unique value lets us
            // verify nothing of this string reached upstream.
            .header("cookie", "session=CLIENT-COOKIE-LEAK-CANARY")
            // X-API-Key: ditto.
            .header("x-api-key", "X-API-KEY-LEAK-CANARY")
            // Set-Cookie is a response header; clients don't usually
            // send it, but the default strip list includes it anyway
            // (defense-in-depth against pathological clients).
            .header("set-cookie", "leak-set-cookie=1")
            // The customer's own trace-correlation header — NOT in
            // the strip list, MUST reach upstream.
            .header("x-trace-id", "trace-abc")
            .body(axum::body::Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let received = upstream.received_requests().await.unwrap();
        assert_eq!(received.len(), 1);
        let r = &received[0];

        // Authorization: gateway sets its own (`Bearer sk-test` from
        // the PK secret); client's `Bearer sk-caller` MUST be
        // stripped before reaching the wire.
        let auths = upstream_header_values(r, "authorization");
        assert_eq!(
            auths,
            vec!["Bearer sk-test"],
            "default strip must replace client Authorization with PK secret only (#411); got: {:?}",
            auths
        );

        // Cookie / x-api-key / set-cookie: the gateway never sets
        // these on the outbound side (for OpenAI), so absence is
        // the right signal.
        assert!(
            upstream_header_values(r, "cookie").is_empty(),
            "cookie must not leak by default (#411)"
        );
        assert!(
            upstream_header_values(r, "x-api-key").is_empty(),
            "x-api-key must not leak by default (#411)"
        );
        assert!(
            upstream_header_values(r, "set-cookie").is_empty(),
            "set-cookie must not leak by default (#411)"
        );

        // Non-stripped header DID reach upstream.
        assert_eq!(
            upstream_header_values(r, "x-trace-id"),
            vec!["trace-abc"],
            "non-stripped header must pass through"
        );
    }

    #[tokio::test]
    async fn always_strip_removes_hop_by_hop_regardless_of_config() {
        // PK with empty strip_headers (customer disabled all default
        // strips). ALWAYS_STRIP still applies — hop-by-hop / protocol
        // headers are non-configurable.
        let upstream = MockServer::start().await;
        Mock::given(wm_method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(provider_key_entry_with_strip(&upstream.uri(), &[]));
        snap.models.insert(openai_model("gpt-4o"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let req = Request::builder()
            .method("GET")
            .uri("/passthrough/openai/v1/models")
            .header("authorization", "Bearer sk-caller")
            .header("connection", "keep-alive, x-custom-fake")
            .header("upgrade", "websocket")
            .body(axum::body::Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let received = upstream.received_requests().await.unwrap();
        let h = &received[0].headers;
        assert!(
            !h.contains_key("connection"),
            "connection must always be stripped (RFC 7230 §6.1)"
        );
        assert!(
            !h.contains_key("upgrade"),
            "upgrade must always be stripped (RFC 7230 §6.1)"
        );
    }

    #[tokio::test]
    async fn empty_strip_list_lets_credentials_through() {
        // The dangerous-but-legal "I unchecked all defaults in the
        // dashboard" override case. Customer takes the risk; the
        // gateway respects the explicit configuration. Documented
        // in the dashboard's confirmation flow.
        //
        // In this case the upstream's Authorization HeaderMap entry
        // has BOTH values (client's + gateway's): `Bearer sk-caller,
        // Bearer sk-test` on the wire. We assert the client's value
        // appears among them — that proves the strip didn't run, the
        // override worked. We DON'T assert the wire order; reqwest
        // append semantics put gateway's value first since it's added
        // after the loop, but the test should be robust to either
        // ordering decision.
        let upstream = MockServer::start().await;
        Mock::given(wm_method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(provider_key_entry_with_strip(&upstream.uri(), &[]));
        snap.models.insert(openai_model("gpt-4o"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let req = Request::builder()
            .method("GET")
            .uri("/passthrough/openai/v1/models")
            .header("authorization", "Bearer sk-caller")
            .header("cookie", "session=letitthrough")
            .body(axum::body::Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let received = upstream.received_requests().await.unwrap();
        let r = &received[0];

        // Authorization: should have BOTH client's and gateway's
        // values present.
        let auths = upstream_header_values(r, "authorization");
        assert!(
            auths.contains(&"Bearer sk-caller"),
            "empty strip_headers must let client Authorization through; got: {:?}",
            auths
        );
        // Cookie: only client's value; gateway doesn't set cookie.
        assert_eq!(
            upstream_header_values(r, "cookie"),
            vec!["session=letitthrough"],
            "empty strip_headers must let cookie through"
        );
    }

    #[tokio::test]
    async fn custom_strip_list_strips_only_named_headers() {
        // Customer overrides defaults — drops `cookie` from the
        // strip list (cookie passes through) but adds custom
        // `x-internal-trace-id` (gets stripped).
        let upstream = MockServer::start().await;
        Mock::given(wm_method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.provider_keys.insert(provider_key_entry_with_strip(
            &upstream.uri(),
            // No "cookie" — cookie passes through. New "x-internal-trace-id"
            // gets stripped. authorization still stripped.
            &["authorization", "x-internal-trace-id"],
        ));
        snap.models.insert(openai_model("gpt-4o"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let req = Request::builder()
            .method("GET")
            .uri("/passthrough/openai/v1/models")
            .header("authorization", "Bearer sk-caller")
            .header("cookie", "tracker=stays")
            .header("x-internal-trace-id", "internal-12345")
            .header("x-public-trace-id", "public-67890")
            .body(axum::body::Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let received = upstream.received_requests().await.unwrap();
        let r = &received[0];

        // Authorization is in the custom strip list → client's
        // value must not appear; gateway's own auth (PK secret)
        // IS present.
        let auths = upstream_header_values(r, "authorization");
        assert!(
            !auths.iter().any(|v| v.contains("sk-caller")),
            "authorization in custom strip list → client value must not leak; got: {:?}",
            auths
        );
        assert!(
            auths.contains(&"Bearer sk-test"),
            "gateway's own auth still present"
        );

        // Cookie is NOT in custom strip list → client's value
        // reaches upstream.
        assert_eq!(
            upstream_header_values(r, "cookie"),
            vec!["tracker=stays"],
            "cookie removed from strip list → passes through"
        );

        // x-internal-trace-id is in custom strip list → gone.
        assert!(
            upstream_header_values(r, "x-internal-trace-id").is_empty(),
            "custom-added strip → removed"
        );

        // x-public-trace-id is NOT stripped → reaches upstream.
        assert_eq!(
            upstream_header_values(r, "x-public-trace-id"),
            vec!["public-67890"],
            "header not in strip list → passes through"
        );
    }

    #[tokio::test]
    async fn strip_header_match_is_case_insensitive() {
        // Inbound header keys can be ANY case. The strip list is
        // lowercased; the comparison must be case-insensitive too.
        let upstream = MockServer::start().await;
        Mock::given(wm_method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(openai_model("gpt-4o"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let req = Request::builder()
            .method("GET")
            .uri("/passthrough/openai/v1/models")
            // axum will lower-case the keys via http::HeaderName but
            // the original-case roundtrip is preserved on some paths;
            // covering it explicitly anchors the contract.
            .header("Authorization", "Bearer sk-caller")
            .header("Cookie", "session=case")
            .body(axum::body::Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let received = upstream.received_requests().await.unwrap();
        let r = &received[0];
        // Even with mixed-case input headers, the lowercased strip
        // set matches → client's leaks must not appear.
        let auths = upstream_header_values(r, "authorization");
        assert!(
            !auths.iter().any(|v| v.contains("sk-caller")),
            "case-insensitive strip: client Authorization must not leak"
        );
        assert!(
            upstream_header_values(r, "cookie").is_empty(),
            "case-insensitive strip: cookie must be removed"
        );
    }

    /// #697: the borrowed model's `allowed_cidrs` (#557) must gate the raw
    /// passthrough tunnel too. A client IP outside the allowlist gets 403
    /// and the upstream is never contacted. Oneshot requests carry no
    /// ConnectInfo → empty source IP, which fails closed against a
    /// configured allowlist (same as the typed handlers).
    #[tokio::test]
    async fn ip_allowlisted_model_rejects_disallowed_client_issue_697() {
        let upstream = MockServer::start().await;
        Mock::given(wm_method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"data": []})))
            .expect(0)
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        let model_json = format!(
            r#"{{"display_name":"gated","provider":"openai","model_name":"gpt-4o","provider_key_id":"{PK_ID}","allowed_cidrs":["10.0.0.0/8"]}}"#
        );
        let m: Model = serde_json::from_str(&model_json).unwrap();
        snap.models.insert(ResourceEntry::new("m-1", m, 1));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let req = Request::builder()
            .method("GET")
            .uri("/passthrough/openai/v1/models")
            .header("authorization", "Bearer sk-caller")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "permission_denied");
    }

    /// #697 companion: a client IP inside the allowlist passes through.
    /// ConnectInfo is injected the way a real listener would provide it.
    #[tokio::test]
    async fn ip_allowlisted_model_allows_matching_client_issue_697() {
        let upstream = MockServer::start().await;
        Mock::given(wm_method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"data": []})))
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        let model_json = format!(
            r#"{{"display_name":"gated","provider":"openai","model_name":"gpt-4o","provider_key_id":"{PK_ID}","allowed_cidrs":["10.0.0.0/8"]}}"#
        );
        let m: Model = serde_json::from_str(&model_json).unwrap();
        snap.models.insert(ResourceEntry::new("m-1", m, 1));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let mut req = Request::builder()
            .method("GET")
            .uri("/passthrough/openai/v1/models")
            .header("authorization", "Bearer sk-caller")
            .body(axum::body::Body::empty())
            .unwrap();
        req.extensions_mut()
            .insert(axum::extract::ConnectInfo(std::net::SocketAddr::from((
                [10, 1, 2, 3],
                50000,
            ))));
        let resp = ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    /// #699: a passthrough request must land in the UsageEvent stream —
    /// zero tokens (nothing is parsed), the relayed upstream status, the
    /// caller's api_key and the lent ProviderKey's attribution tags.
    #[tokio::test]
    async fn emits_usage_event_on_success_issue_699() {
        use aisix_obs::{UsageEvent, UsageSink};

        let upstream = MockServer::start().await;
        Mock::given(wm_method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"data": []})))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        let pk_json = format!(
            r#"{{"display_name":"openai-up","secret":"sk-test","api_base":"{}","provider":"openai","adapter":"openai","telemetry_tags":{{"kind":"catalog","featured":true,"branded_provider":"openai","pk_label":"prod-pt-key"}}}}"#,
            upstream.uri()
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&pk_json).unwrap();
        snap.provider_keys.insert(ResourceEntry::new(PK_ID, pk, 1));
        snap.models.insert(openai_model("gpt"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel::<UsageEvent>(8);
        let hub = Arc::new(Hub::new());
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        let req = Request::builder()
            .method("GET")
            .uri("/passthrough/openai/v1/models")
            .header("authorization", "Bearer sk-caller")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("a passthrough request must emit a UsageEvent")
            .expect("usage_sink sender dropped");
        assert_eq!(event.inbound_protocol, "passthrough");
        assert_eq!(event.status_code, 200);
        assert_eq!(event.api_key_id, "k-1");
        assert_eq!(event.prompt_tokens, 0, "the raw tunnel parses no tokens");
        assert_eq!(event.pk_label, "prod-pt-key", "PK attribution must apply");
        assert!(
            rx.try_recv().is_err(),
            "exactly one event per passthrough request"
        );
    }

    /// #699 / #655 parity: a failed passthrough dispatch (no model for the
    /// provider -> 404) emits one zero-token error event instead of being
    /// dropped.
    #[tokio::test]
    async fn failed_dispatch_emits_zero_token_error_event_issue_699() {
        use aisix_obs::{UsageEvent, UsageSink};

        let snap = new_snap("http://unused");
        snap.models.insert(openai_model("gpt"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel::<UsageEvent>(8);
        let hub = Arc::new(Hub::new());
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        let app = crate::build_router(state);

        let req = Request::builder()
            .method("GET")
            .uri("/passthrough/nonexistent/v1/models")
            .header("authorization", "Bearer sk-caller")
            .body(axum::body::Body::empty())
            .unwrap();
        let resp = ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("a failed passthrough must emit a zero-token UsageEvent")
            .expect("usage_sink sender dropped");
        assert_eq!(event.status_code, 404);
        assert_eq!(event.api_key_id, "k-1");
        assert!(
            !event.error_class.is_empty(),
            "error_class must classify the failure"
        );
        assert_eq!(
            event.inbound_protocol, "passthrough",
            "error event must carry the same inbound_protocol as the success path \
             so Logs protocol filtering sees both"
        );
    }
}
