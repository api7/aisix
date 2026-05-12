//! `POST /v1/chat/completions` handler.
//!
//! Flow:
//! 1. [`AuthenticatedKey`] extractor runs first — rejects unauthenticated
//!    requests with a 401 envelope.
//! 2. Parse [`ChatFormat`] from the JSON body.
//! 3. Resolve `req.model` against the snapshot's Model table → 404 if
//!    absent.
//! 4. Check the ApiKey's `allowed_models` whitelist → 403 if disallowed.
//! 5. Look up the matching `Bridge` on the Hub by `Model::provider()` →
//!    503 if no bridge registered.
//! 6. Rate-limit pre-commit; build [`BridgeContext`] and dispatch:
//!    - `stream == true`  → `chat_stream` + Sse response
//!    - otherwise          → `chat` + JSON response rendered as OpenAI
//! 7. On completion: record metrics + emit one structured access log
//!    line. Errors surface via [`ProxyError`] which carries the right
//!    status, error type, and (for rate-limits) Retry-After.

use aisix_cache::CacheKey;
use aisix_gateway::{BridgeContext, BridgeError, ChatFormat};
use aisix_guardrails::GuardrailVerdict;
use aisix_obs::{AccessLog, Metrics, RequestOutcome, UsageEvent};
use axum::extract::State;
use axum::http::HeaderValue;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::{Stream, StreamExt};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

use crate::auth::AuthenticatedKey;
use crate::error::ProxyError;
use crate::render::{render_chunk, render_response};
use crate::routing::is_retryable;
use crate::state::ProxyState;

/// Header set on every non-streaming response indicating whether the
/// response came from the cache (`hit`) or the upstream (`miss`).
pub const CACHE_HEADER: &str = "x-aisix-cache";

pub async fn chat_completions(
    State(state): State<ProxyState>,
    auth: AuthenticatedKey,
    Json(req): Json<ChatFormat>,
) -> Response {
    let started = Instant::now();
    let method = "POST";
    let path = "/v1/chat/completions";
    // Pure UUID — cp-api's telemetry handler stores this in the
    // `request_id UUID` column of dpmgr_usage_events, so a "req-…"
    // prefix on the wire would be rejected (event 0: request_id must
    // be a uuid). The downstream `x-aisix-call-id` header carries the
    // same value for human correlation.
    let request_id = Uuid::new_v4().to_string();
    let api_key_id = auth.entry.id.clone();
    let model_name = req.model.clone();

    let outcome = dispatch(&state, &auth, &req, &request_id, started).await;

    match outcome {
        Ok(mut success) => {
            let status = 200;
            let elapsed = started.elapsed();
            record_success(
                &state.metrics,
                &success.provider,
                &model_name,
                status,
                &success,
                elapsed,
            );
            emit_access_log(
                method,
                path,
                status,
                elapsed,
                Some(success.provider.as_str()),
                Some(&model_name),
                Some(&api_key_id),
                success.prompt_tokens,
                success.completion_tokens,
                success.total_tokens,
                &request_id,
            );
            // The streaming path wires telemetry into the SSE stream's
            // on_complete callback (it has to wait for the terminal
            // chunk to read the upstream's `usage` block). Calling
            // `emit_usage_event` here for streaming would double-emit
            // — once with all-zero tokens at handler return, then
            // again with the real values at stream end.
            if !success.telemetry_handled_by_stream {
                emit_usage_event(
                    &state,
                    &request_id,
                    &success.model_id,
                    &api_key_id,
                    status,
                    elapsed,
                    success.prompt_tokens.unwrap_or(0) as u32,
                    success.completion_tokens.unwrap_or(0) as u32,
                    UsageExtras {
                        cached_prompt_tokens: success.cached_prompt_tokens,
                        reasoning_tokens: success.reasoning_tokens,
                        cache_creation_tokens: success.cache_creation_tokens,
                        cache_read_tokens: success.cache_read_tokens,
                        provider_request_id: success.provider_request_id.clone(),
                        provider_model_version: success.provider_model_version.clone(),
                        finish_reason: success.finish_reason.clone(),
                        bypass_reason: success.bypass_reason.clone().unwrap_or_default(),
                        cache_status: success.cache_status.as_str().to_string(),
                        cache_hit_saved_input_tokens: success.cache_hit_saved_input_tokens,
                        cache_hit_saved_output_tokens: success.cache_hit_saved_output_tokens,
                    },
                    success.cost_usd,
                    /* guardrail_blocked */ false,
                );
            }
            // Inject x-ratelimit-* headers so OpenAI SDK clients see the
            // current window state. We peek *after* the commit so
            // remaining-requests reflects the post-dispatch tally.
            let rl_limits = auth.key().rate_limit.clone().unwrap_or_default();
            if let Some(rl_status) = state.limiter.peek(&api_key_id, &rl_limits) {
                crate::render::inject_ratelimit_headers(&mut success.response, &rl_status);
            }
            // Correlation / routing headers.
            if let Ok(v) = axum::http::HeaderValue::try_from(request_id.as_str()) {
                success.response.headers_mut().insert("x-aisix-call-id", v);
            }
            success.response
        }
        Err((resolved_model_id, charge, err)) => {
            let status = err.status().as_u16();
            let elapsed = started.elapsed();
            record_error(&state.metrics, &err, &model_name, status, elapsed);
            // Access log: surface the upstream-billed counts when the
            // error fired AFTER the upstream call (output-content-filter
            // block). Pre-upstream errors (input filter, budget,
            // model-not-found) carry no charge — log None there so the
            // line reflects "request never reached the model".
            let (al_prompt, al_completion, al_total) = match charge.as_ref() {
                Some(c) => (
                    Some(u64::from(c.prompt_tokens)),
                    Some(u64::from(c.completion_tokens)),
                    Some(u64::from(c.prompt_tokens) + u64::from(c.completion_tokens)),
                ),
                None => (None, None, None),
            };
            emit_access_log(
                method,
                path,
                status,
                elapsed,
                None,
                Some(&model_name),
                Some(&api_key_id),
                al_prompt,
                al_completion,
                al_total,
                &request_id,
            );
            // Telemetry for the failure path. `resolved_model_id` is
            // populated by `dispatch` once the request's `req.model`
            // has been resolved against the snapshot — so a guardrail
            // / budget / rate-limit / bridge error after that point
            // still records which model the request targeted. Errors
            // before resolution (empty messages, ModelNotFound) keep
            // an empty string. ContentFiltered (guardrail) is recorded
            // with the dedicated `guardrail_blocked` flag so the
            // dashboard can surface it on the Blocked tab.
            let guardrail_blocked = matches!(err, ProxyError::ContentFiltered(_));
            // For output-content-filter blocks the upstream WAS called
            // and the provider already billed for `prompt_tokens` +
            // `completion_tokens`. The captured `charge` carries those
            // counts forward so the customer's `usage_events` row
            // reflects the bill. For all other error paths charge is
            // None and tokens stay at 0 — those errors never reached
            // the upstream.
            let (prompt_tokens, completion_tokens, extras) = match charge {
                Some(c) => (
                    c.prompt_tokens,
                    c.completion_tokens,
                    UsageExtras {
                        cached_prompt_tokens: c.cached_prompt_tokens,
                        reasoning_tokens: c.reasoning_tokens,
                        cache_creation_tokens: c.cache_creation_tokens,
                        cache_read_tokens: c.cache_read_tokens,
                        provider_request_id: c.provider_request_id,
                        provider_model_version: c.provider_model_version,
                        finish_reason: c.finish_reason,
                        bypass_reason: c.bypass_reason,
                        cache_status: c.cache_status.as_str().to_string(),
                        cache_hit_saved_input_tokens: 0,
                        cache_hit_saved_output_tokens: 0,
                    },
                ),
                None => (0, 0, UsageExtras::default()),
            };
            emit_usage_event(
                &state,
                &request_id,
                resolved_model_id.as_deref().unwrap_or(""),
                &api_key_id,
                status,
                elapsed,
                prompt_tokens,
                completion_tokens,
                extras,
                /* cost_usd */ 0.0,
                guardrail_blocked,
            );
            err.into_response()
        }
    }
}

/// Everything needed to populate post-call telemetry for a successful
/// request. For streaming we don't yet have token totals, so those are
/// `None` and `total_tokens` stays at 0.
struct Success {
    response: Response,
    provider: String,
    /// UUID of the v3 Model row this request resolved to (the virtual
    /// model the caller asked for, before any routing fan-out). Empty
    /// in the unlikely case the resolver was bypassed.
    model_id: String,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    total_tokens: Option<u64>,
    /// Provider-specific cache + reasoning token counters. Default 0
    /// for providers that don't expose them; cp-api falls back to the
    /// standard prompt / completion rate when these are 0.
    cached_prompt_tokens: u32,
    reasoning_tokens: u32,
    cache_creation_tokens: u32,
    cache_read_tokens: u32,
    /// Provider response `id` (OpenAI chat.completion.id or Anthropic
    /// message id) — empty when the cached path served the request
    /// (re-using a stored response's id would mislead reconciliation).
    provider_request_id: String,
    /// Resolved model the provider actually billed.
    provider_model_version: String,
    /// finish_reason / stop_reason as the upstream returned it. Empty
    /// for streaming (no terminal event yet) and cache hits.
    finish_reason: String,
    /// Cost computed via the per-key Budget's pricing table, in USD.
    /// 0.0 when no budget is configured for the key.
    cost_usd: f64,
    /// Set when at least one guardrail returned `Bypass` (remote-API
    /// guardrail upstream unreachable + `fail_open=true`). The first
    /// bypass reason wins. Goes onto `usage_events.guardrail_bypassed_reason`
    /// so a compliance audit can see what slipped past during a Bedrock
    /// outage. None for the normal Allow / Block paths.
    bypass_reason: Option<String>,
    /// Cache outcome for this request. `disabled` when no enabled
    /// cache_policy is in snapshot for the env; `miss` when the cache
    /// was consulted but no entry matched; `hit` when a stored entry
    /// served the response. Lands on `usage_events.cache_status` for
    /// the dashboard's /logs column. See `aisix-core::CachePolicy`
    /// (Stage 2) and `aisix-cache::Cache` for the source of truth.
    cache_status: CacheStatus,
    /// True when telemetry emission is wired into the SSE stream's
    /// on_complete callback (streaming path). The top-level handler
    /// must NOT call `emit_usage_event` again — that would emit one
    /// event with all-zero tokens at handler return on top of the real
    /// one from stream completion. Always false for non-streaming
    /// paths (handler emits inline with `success.prompt_tokens` etc.).
    telemetry_handled_by_stream: bool,
    /// On a cache HIT, the prompt + completion tokens of the cached
    /// response — the work the upstream would have repeated had the
    /// cache not served the request. Both 0 on miss / disabled. cp-api
    /// multiplies these by its pricing catalog to derive
    /// `cost_saved_usd` on ingestion (matches the existing `cost_usd`
    /// pattern: DP records tokens, cp-api owns pricing). See #88.
    cache_hit_saved_input_tokens: u32,
    cache_hit_saved_output_tokens: u32,
}

/// Cache decision attached to every successful request. Wire shape
/// (lowercase string) is what cp-api persists in
/// `dpmgr_usage_events.cache_status`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CacheStatus {
    /// No enabled cache policy in snapshot — gate skipped the lookup.
    /// Stage 3 will refine this to "no policy matched applies_to".
    Disabled,
    /// Cache consulted, no entry matched. The successful upstream
    /// response is stored on the way out so future identical requests
    /// hit. Maps to `x-aisix-cache: miss` on the response header.
    Miss,
    /// Cache consulted, entry returned without an upstream call. Maps
    /// to `x-aisix-cache: hit` on the response header.
    Hit,
}

impl CacheStatus {
    /// Lowercase wire string the DP ships to cp-api in `cache_status`.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            CacheStatus::Disabled => "disabled",
            CacheStatus::Miss => "miss",
            CacheStatus::Hit => "hit",
        }
    }
}

/// Returns `Ok(Success)` on the happy path, or `Err((resolved_model_id, err))`
/// where `resolved_model_id` is `Some(id)` once the request's `req.model`
/// has been resolved against the snapshot, and `None` for errors that
/// fire before resolution (empty messages, ModelNotFound). The caller
/// surfaces this id on the failure-path telemetry event so the
/// dashboard's `/logs` page can show the model that was targeted by a
/// guardrail-blocked / budget-exceeded / bridge-failed request.
/// Upstream-billed token counts attached to dispatch errors that fired
/// AFTER the upstream call already responded. Today this is populated
/// only on the output-content-filter block path: the gateway received a
/// full upstream response (which the provider has already charged for),
/// then the output guardrail decided to block delivery to the client.
/// The customer is still on the hook for those tokens, so the charge
/// flows up to the handler so its `usage_events` row reflects the bill.
///
/// `None` on every other error path — input-filter blocks, budget
/// failures, rate-limit rejections, model-not-found, etc. all happen
/// BEFORE any upstream call and the customer pays nothing.
struct UpstreamCharge {
    prompt_tokens: u32,
    completion_tokens: u32,
    cached_prompt_tokens: u32,
    reasoning_tokens: u32,
    cache_creation_tokens: u32,
    cache_read_tokens: u32,
    provider_request_id: String,
    provider_model_version: String,
    finish_reason: String,
    /// First bypass reason observed on this request (input or output
    /// guardrail returned `Bypass` because a remote-API guardrail was
    /// unreachable + `fail_open=true`). Empty string = no bypass.
    /// Carried so the output-block telemetry surfaces "this blocked
    /// request had a degraded input check" — without this, an operator
    /// auditing a guardrail bypass would only see the bypass on
    /// successfully-served requests, never on output-blocked ones.
    bypass_reason: String,
    /// Cache outcome decided BEFORE the output guardrail ran. The
    /// dashboard's cache-status filter counts misses vs disabled vs
    /// hits — without forwarding it on the output-block path the
    /// blocked-but-billed request is mis-bucketed as "no cache decision".
    cache_status: CacheStatus,
}

async fn dispatch(
    state: &ProxyState,
    auth: &AuthenticatedKey,
    req: &ChatFormat,
    request_id: &str,
    started: Instant,
) -> Result<Success, (Option<String>, Option<UpstreamCharge>, ProxyError)> {
    if req.messages.is_empty() {
        return Err((
            None,
            None,
            ProxyError::InvalidRequest("messages array must not be empty".into()),
        ));
    }

    let snapshot = state.snapshot.load();
    let virtual_entry = snapshot
        .models
        .get_by_name(&req.model)
        .ok_or_else(|| (None, None, ProxyError::ModelNotFound(req.model.clone())))?;
    let model_id = virtual_entry.id.clone();

    // Every error from here on attaches the resolved model_id so the
    // failure-path telemetry event in `chat_completions` can surface
    // which model the request targeted.
    // Pre-upstream errors carry `None` for the charge — by definition
    // they fire before any provider billing happens. The output-filter
    // path below is the only site that builds a `Some(UpstreamCharge)`
    // (manually, not via this helper).
    let with_model = |e: ProxyError| (Some(model_id.clone()), None, e);

    if !auth.key().can_access(&req.model) {
        return Err(with_model(ProxyError::ModelForbidden(req.model.clone())));
    }

    // Input guardrails. Run before reservation so a blocked prompt
    // doesn't burn an RPM slot — content-policy refusals shouldn't
    // count against quota. Bypass (remote-API guardrail unavailable
    // + fail_open=true) doesn't short-circuit; the reason is stashed
    // and attached to the telemetry event when the request finishes.
    let mut bypass_reason: Option<String> = None;
    match state.guardrails.check_input(req).await {
        GuardrailVerdict::Allow => {}
        GuardrailVerdict::Block { reason } => {
            // The verdict's `reason` carries matched-pattern detail
            // (e.g. `"input blocked by literal \"forbidden-token\""`).
            // Keep it for operator logs but DO NOT propagate it to the
            // wire envelope — see #153. The redacted public message
            // stays generic so callers can't enumerate the blocklist
            // by inspecting error responses.
            tracing::warn!(
                guardrail_hook = "input",
                model = %req.model,
                reason = %reason,
                "guardrail blocked request"
            );
            return Err(with_model(ProxyError::ContentFiltered(
                "request blocked by content policy".into(),
            )));
        }
        GuardrailVerdict::Bypass { reason } => {
            bypass_reason = Some(reason);
        }
    }

    // Budget pre-check via cp-api. The DP no longer owns budget state;
    // cp-api returns a cached/live decision per api_key.
    let decision = state.budgets.check(&auth.entry.id).await;
    if !decision.allowed {
        return Err(with_model(ProxyError::BudgetExceeded(
            decision.reason.unwrap_or_else(|| auth.entry.id.clone()),
        )));
    }

    // Resolve the attempt-list of underlying Model entries. For a
    // routing model we walk targets per the configured strategy; for a
    // single-provider Model we just dispatch to it directly.
    let attempt_models: Vec<aisix_core::Model> =
        if let Some(routing) = virtual_entry.value.routing.as_ref() {
            let names = state.routing.pick_targets(&req.model, routing);
            if names.is_empty() {
                return Err(with_model(ProxyError::InvalidRequest(
                    "routing model has no targets".into(),
                )));
            }
            let mut resolved = Vec::with_capacity(names.len());
            for name in &names {
                let target_entry = snapshot.models.get_by_name(name).ok_or_else(|| {
                    with_model(ProxyError::InvalidRequest(format!(
                        "routing target {name:?} does not resolve to a Model"
                    )))
                })?;
                resolved.push(target_entry.value.clone());
            }
            resolved
        } else {
            vec![virtual_entry.value.clone()]
        };

    // For non-routing requests, surface a misconfigured bridge as a
    // proper 503 rather than burying it inside a generic Bridge error.
    // Routing requests rely on the loop's `is_retryable` path so a
    // single bad provider doesn't take down the whole request.
    if attempt_models.len() == 1 {
        let only = &attempt_models[0];
        let provider = crate::dispatch::require_provider(only).map_err(with_model)?;
        if state.hub.get(provider).is_none() {
            return Err(with_model(ProxyError::ProviderUnavailable));
        }
    }

    let rl_key = auth.entry.id.clone();
    let rl_limits = auth.key().rate_limit.clone().unwrap_or_default();
    let reservation = state
        .limiter
        .pre_commit(&rl_key, &rl_limits)
        .map_err(|e| with_model(ProxyError::from(e)))?;

    let now = created_ts();

    // Streaming path: only attempt the first target. Streaming fallback
    // is genuinely hard (we'd have to buffer the stream to detect
    // failure mid-flight) and not worth the complexity for V1.
    if req.is_streaming() {
        let model = &attempt_models[0];
        let provider = crate::dispatch::require_provider(model).map_err(with_model)?;
        let pk_entry =
            crate::dispatch::resolve_provider_key(&snapshot, model).map_err(with_model)?;
        let bridge = state
            .hub
            .get(provider)
            .ok_or_else(|| with_model(ProxyError::ProviderUnavailable))?;
        let model_arc = Arc::new(model.clone());
        let pk_arc = Arc::new(pk_entry.value.clone());
        let ctx = BridgeContext::new(request_id, model_arc, pk_arc);
        let upstream = bridge
            .chat_stream(req, &ctx)
            .await
            .map_err(|e| with_model(ProxyError::Bridge(e)))?;
        // Drop the reservation now: concurrency releases (the SSE
        // stream that follows is driven by the client, not by the
        // proxy holding open an upstream-bound future), and RPM was
        // already counted by pre_commit. TPM is updated retroactively
        // on stream-end by `add_tokens_post_stream` — see issue #108.
        // Pre-fix this path called commit_tokens(0) and never came
        // back, leaving TPM caps blind for all streaming traffic.
        drop(reservation);
        // Capture everything the stream-completion callback needs so
        // it can fire `emit_usage_event` once the terminal SSE chunk
        // has yielded its `usage` block. Telemetry emission has to
        // wait until end-of-stream because OpenAI / Anthropic only
        // populate `usage` on the last chunk; emitting at handler
        // return (the non-streaming path's spot) would record zeros.
        let limiter = Arc::clone(&state.limiter);
        let post_stream_key = rl_key.clone();
        let state_for_telem = state.clone();
        let request_id_for_telem = request_id.to_string();
        let model_id_for_telem = model_id.clone();
        let api_key_id_for_telem = auth.entry.id.clone();
        let bypass_reason_for_telem = bypass_reason.clone().unwrap_or_default();
        // Per #204: pass the gateway's guardrail chain so the
        // streaming path can run output guardrails at end-of-stream
        // (buffer-then-check). Mirrors the non-streaming
        // `state.guardrails.check_output(...)` call site.
        //
        // Fast-path: skip the context entirely when no policies are
        // configured. The Guardrail trait's `is_empty()` (audit
        // PR #222 M2) lets `Arc<dyn Guardrail>` answer the question
        // without a downcast. When `None`, `build_sse_stream` skips
        // the per-chunk content accumulation and the post-loop
        // synthesized-ChatResponse construction — both noise on
        // the hot path for the dominant guardrail-free deployment.
        let stream_guardrail = if state.guardrails.is_empty() {
            None
        } else {
            Some(StreamGuardrailContext {
                chain: Arc::clone(&state.guardrails),
                model_name: req.model.clone(),
            })
        };
        let sse_stream = build_sse_stream(
            upstream,
            now,
            stream_guardrail,
            move |comp: StreamCompletion| {
                // Existing: rate-limit accounting (TPM cap) — see #108.
                limiter.add_tokens_post_stream(&post_stream_key, comp.total_tokens);
                // Telemetry: emit with the actual upstream-reported counts.
                // cost_usd stays 0.0; cp-api recomputes server-side from
                // its model_pricing catalog (same pattern as the non-
                // streaming path's cost_usd handling).
                emit_usage_event(
                    &state_for_telem,
                    &request_id_for_telem,
                    &model_id_for_telem,
                    &api_key_id_for_telem,
                    /* status_code */ 200,
                    started.elapsed(),
                    comp.prompt_tokens,
                    comp.completion_tokens,
                    UsageExtras {
                        cached_prompt_tokens: comp.cached_prompt_tokens,
                        reasoning_tokens: comp.reasoning_tokens,
                        cache_creation_tokens: comp.cache_creation_tokens,
                        cache_read_tokens: comp.cache_read_tokens,
                        provider_request_id: comp.provider_request_id,
                        provider_model_version: comp.provider_model_version,
                        finish_reason: comp.finish_reason,
                        // Per #204 audit H1: merge input-side bypass
                        // (`bypass_reason_for_telem` captured before
                        // the upstream stream started) with the
                        // output-side bypass observed at end-of-stream
                        // (`comp.bypass_reason`). First-bypass-wins,
                        // matching the non-streaming convention.
                        bypass_reason: if !bypass_reason_for_telem.is_empty() {
                            bypass_reason_for_telem
                        } else {
                            comp.bypass_reason
                        },
                        // TODO(streaming-cache): when streaming responses
                        // become cacheable, this constant `Disabled` will
                        // silently mis-bucket cached streamed responses.
                        // Propagate the dispatch path's `cache_status`
                        // local at that point. Tracking issue to be filed
                        // alongside the streaming-cache implementation.
                        cache_status: CacheStatus::Disabled.as_str().to_string(),
                        cache_hit_saved_input_tokens: 0,
                        cache_hit_saved_output_tokens: 0,
                    },
                    /* cost_usd */ 0.0,
                    comp.guardrail_blocked,
                );
            },
        );
        let response =
            Sse::new(sse_stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)));
        return Ok(Success {
            response: response.into_response(),
            provider: format!("{provider:?}").to_lowercase(),
            model_id: model_id.clone(),
            // Token totals are populated on the SSE stream's terminal
            // chunk and forwarded into telemetry from on_complete; the
            // top-level handler skips its own `emit_usage_event` for
            // streaming via `telemetry_handled_by_stream` below.
            prompt_tokens: None,
            completion_tokens: None,
            total_tokens: None,
            cost_usd: 0.0,
            cached_prompt_tokens: 0,
            reasoning_tokens: 0,
            cache_creation_tokens: 0,
            cache_read_tokens: 0,
            provider_request_id: String::new(),
            provider_model_version: String::new(),
            finish_reason: String::new(),
            bypass_reason: bypass_reason.clone(),
            // Streaming responses aren't cached at this layer — see
            // crates/aisix-cache/src/lib.rs. Always surface as
            // `disabled` on the streaming path.
            cache_status: CacheStatus::Disabled,
            cache_hit_saved_input_tokens: 0,
            cache_hit_saved_output_tokens: 0,
            telemetry_handled_by_stream: true,
        });
    }

    // Policy gate (Stage 3): the cache is only consulted when at
    // least one enabled `CachePolicy` in the snapshot has an
    // `applies_to` clause that matches THIS request. cp-api owns
    // the policy CRUD surface (`/api/environments/:env/cache_policies`,
    // see Stage 1); kine fans out the rows; the loader populates
    // `snapshot.cache_policies` (see aisix-etcd). Stage 4 will add
    // per-policy `ttl_seconds` propagation into the cache backend.
    //
    // Match order: first enabled policy whose `parsed_applies_to()`
    // accepts (req.model, auth.entry.id) wins. We grab the WHOLE
    // matching entry (not just `any`) so the post-call write below
    // can use that policy's `ttl_seconds` via `put_with_ttl`. When
    // multiple policies match the same request, the entry-table
    // iteration order decides — that's an unspecified-but-stable
    // tiebreak we'll formalise (probably "narrowest scope wins") in a
    // follow-up if operators ever care.
    let matched_policy_ttl = snapshot
        .cache_policies
        .entries()
        .iter()
        .find(|entry| {
            entry.value.enabled
                && entry
                    .value
                    .parsed_applies_to()
                    .matches(&req.model, &auth.entry.id)
        })
        .map(|entry| Duration::from_secs(u64::from(entry.value.ttl_seconds)));
    let cache_active_by_policy = matched_policy_ttl.is_some();

    // Cache lookup keyed on the *virtual* model name so a re-request
    // hits the cache regardless of which target served the original.
    // Even with `cache_active_by_policy = false` we still build the
    // key to keep the cache_status path uniform — `disabled` is the
    // outcome when the gate is closed, but the request itself is
    // shaped the same way.
    let cache_key = state
        .cache
        .as_ref()
        .map(|_| CacheKey::from_request(req).fingerprint());

    let cache_status = if cache_active_by_policy && state.cache.is_some() {
        CacheStatus::Miss
    } else {
        CacheStatus::Disabled
    };

    if let (true, Some(cache), Some(key)) = (
        cache_active_by_policy,
        state.cache.as_ref(),
        cache_key.as_ref(),
    ) {
        match cache.get(key).await {
            Ok(Some(cached)) => {
                reservation.commit_tokens(0);
                let prompt = cached.usage.prompt_tokens as u64;
                let completion = cached.usage.completion_tokens as u64;
                let total = cached.usage.total_tokens as u64;
                // Snapshot the cache / reasoning counters BEFORE the
                // value moves into render_response below. Cache hits
                // replay the original upstream's usage — we already
                // paid the cost first time around — so the dashboard's
                // "of which N were cache hits" stat reflects the
                // original event accurately.
                let cached_prompt_tokens = cached.usage.cached_prompt_tokens;
                let reasoning_tokens = cached.usage.reasoning_tokens;
                let cache_creation_tokens = cached.usage.cache_creation_tokens;
                let cache_read_tokens = cached.usage.cache_read_tokens;
                // The provider label points at the first attempt — for a
                // cache hit we don't know (or care) which target ran the
                // original call; the fingerprint identified the answer.
                let provider_label = attempt_models[0]
                    .provider
                    .map(|p| format!("{p:?}").to_lowercase())
                    .unwrap_or_else(|| "unknown".into());
                let mut response = Json(render_response(now, cached)).into_response();
                response
                    .headers_mut()
                    .insert(CACHE_HEADER, HeaderValue::from_static("hit"));
                return Ok(Success {
                    response,
                    provider: provider_label,
                    model_id: model_id.clone(),
                    prompt_tokens: Some(prompt),
                    completion_tokens: Some(completion),
                    total_tokens: Some(total),
                    cached_prompt_tokens,
                    reasoning_tokens,
                    cache_creation_tokens,
                    cache_read_tokens,
                    // The cache stored the original provider response;
                    // a stable id here would mislead reconciliation
                    // (the request didn't actually hit the upstream),
                    // so we leave these blank deliberately.
                    provider_request_id: String::new(),
                    provider_model_version: String::new(),
                    finish_reason: String::new(),
                    // Cache hits don't burn cost on our side (we already
                    // paid the upstream price the first time around).
                    cost_usd: 0.0,
                    bypass_reason: bypass_reason.clone(),
                    cache_status: CacheStatus::Hit,
                    // The whole point of #88: cache hit replays the
                    // upstream's prompt + completion tokens. Surfacing
                    // them as a dedicated counter (rather than relying
                    // on the existing prompt_tokens column + cache_status
                    // filter) lets cp-api compute `cost_saved_usd`
                    // without joining on the status enum.
                    cache_hit_saved_input_tokens: prompt.try_into().unwrap_or(u32::MAX),
                    cache_hit_saved_output_tokens: completion.try_into().unwrap_or(u32::MAX),
                    telemetry_handled_by_stream: false,
                });
            }
            Ok(None) => {}
            Err(err) => {
                tracing::warn!(error = %err, key = %key, "cache lookup failed");
            }
        }
    }

    // Walk the target list. Retry the current target first, then fail over
    // to later targets only after retries are exhausted. Non-retryable
    // (non-429 4xx) errors stop immediately.
    let mut last_err: Option<BridgeError> = None;
    let mut chosen_provider: Option<String> = None;
    let mut upstream: Option<aisix_gateway::ChatResponse> = None;
    let retries = virtual_entry
        .value
        .routing
        .as_ref()
        .map(|routing| routing.retries_or_default())
        .unwrap_or(0);
    let retry_on_429 = virtual_entry
        .value
        .routing
        .as_ref()
        .map(|routing| routing.retry_on_429_or_default())
        .unwrap_or(false);

    for model in &attempt_models {
        let Some(provider) = model.provider else {
            last_err = Some(BridgeError::Config("model has no provider".into()));
            continue;
        };
        let pk_entry = match crate::dispatch::resolve_provider_key(&snapshot, model) {
            Ok(pk) => pk,
            Err(_) => {
                last_err = Some(BridgeError::Config(
                    "model references unknown provider_key_id".into(),
                ));
                continue;
            }
        };
        let Some(bridge) = state.hub.get(provider) else {
            last_err = Some(BridgeError::Config(
                "no bridge registered for provider".into(),
            ));
            continue;
        };
        let model_arc = Arc::new(model.clone());
        let pk_arc = Arc::new(pk_entry.value.clone());
        let ctx = BridgeContext::new(request_id, model_arc, pk_arc);

        for attempt_idx in 0..=retries {
            match bridge.chat(req, &ctx).await {
                Ok(resp) => {
                    state.health.record_success(&model.display_name);
                    chosen_provider = Some(format!("{provider:?}").to_lowercase());
                    upstream = Some(resp);
                    break;
                }
                Err(err) => {
                    let retryable = is_retryable(&err, retry_on_429);
                    tracing::warn!(
                        target_model = %model.display_name,
                        target_attempt = attempt_idx + 1,
                        error = %err,
                        retryable,
                        "routing target attempt failed",
                    );
                    if retryable {
                        state.health.record_failure(&model.display_name);
                    }
                    last_err = Some(err);
                    if !retryable {
                        break;
                    }
                    if attempt_idx == retries {
                        break;
                    }
                }
            }
        }
        if upstream.is_some() {
            break;
        }
        if let Some(err) = last_err.as_ref() {
            if !is_retryable(err, retry_on_429) {
                break;
            }
        }
    }

    let Some(upstream) = upstream else {
        // Bubble the most recent BridgeError through ProxyError::Bridge.
        let err = last_err.unwrap_or_else(|| {
            BridgeError::Config("routing exhausted with no targets attempted".into())
        });
        return Err(with_model(ProxyError::Bridge(err)));
    };
    let provider_name = chosen_provider.unwrap_or_else(|| "unknown".into());

    // Output guardrail. Tokens still count against quota — the upstream
    // already burned them — so commit before the check, and refuse the
    // refusal-write to the cache so a re-request gets a fresh chance.
    let prompt = upstream.usage.prompt_tokens as u64;
    let completion = upstream.usage.completion_tokens as u64;
    let total = upstream.usage.total_tokens as u64;
    // Snapshot the cache / reasoning counters + provider identity before
    // the upstream gets moved into render_response below — we need them
    // on the Success struct for telemetry.
    let cached_prompt_tokens = upstream.usage.cached_prompt_tokens;
    let reasoning_tokens = upstream.usage.reasoning_tokens;
    let cache_creation_tokens = upstream.usage.cache_creation_tokens;
    let cache_read_tokens = upstream.usage.cache_read_tokens;
    let provider_request_id = upstream.id.clone();
    let provider_model_version = upstream.model.clone();
    let finish_reason = finish_reason_label(&upstream.finish_reason);
    reservation.commit_tokens(total);

    // cp-api recomputes cost server-side from its pricing catalog when
    // ingesting telemetry; the DP just records 0.0 on the wire.
    let cost_usd = 0.0;

    match state.guardrails.check_output(&upstream).await {
        GuardrailVerdict::Allow => {}
        GuardrailVerdict::Block { reason } => {
            // Output filter fires AFTER the upstream call, so the
            // provider has already billed for these tokens. Surface
            // the captured upstream `UsageStats` to the handler so
            // `usage_events` reflects the bill — silently zeroing
            // them would let the customer's dashboard underreport
            // tokens they paid the provider for.
            //
            // `bypass_reason` and `cache_status` are also forwarded
            // so an output-blocked request stays comparable to a
            // succeeded-but-billed one in the dashboard's bypass /
            // cache-status filters. Without these, an operator
            // auditing input-bypass would never see them on
            // output-blocked requests.
            let charge = UpstreamCharge {
                prompt_tokens: upstream.usage.prompt_tokens,
                completion_tokens: upstream.usage.completion_tokens,
                cached_prompt_tokens,
                reasoning_tokens,
                cache_creation_tokens,
                cache_read_tokens,
                provider_request_id: provider_request_id.clone(),
                provider_model_version: provider_model_version.clone(),
                finish_reason: finish_reason.clone(),
                bypass_reason: bypass_reason.clone().unwrap_or_default(),
                cache_status,
            };
            // Per #153, the verdict's `reason` carries the matched-
            // pattern detail (the actual forbidden text from the
            // model's response). Echoing that back to the caller is
            // a real bypass of the output guardrail's purpose:
            // anyone who can trigger the rule can extract the model's
            // forbidden output via the error envelope. Redact on
            // the wire and keep the rich detail in tracing for ops.
            tracing::warn!(
                guardrail_hook = "output",
                model = %req.model,
                reason = %reason,
                "guardrail blocked response"
            );
            return Err((
                Some(model_id.clone()),
                Some(charge),
                ProxyError::ContentFiltered("response blocked by content policy".into()),
            ));
        }
        GuardrailVerdict::Bypass { reason } => {
            // First bypass wins — input bypass already populated
            // bypass_reason if it fired, in which case we keep the
            // earlier signal (it's the policy that failed first).
            if bypass_reason.is_none() {
                bypass_reason = Some(reason);
            }
        }
    }

    // Cache write is gated on the same policy as the lookup at the
    // top of dispatch — without a matching enabled cache_policy in
    // snapshot, we skip both read AND write so the cache backend
    // doesn't fill up with entries that no policy ever asked for.
    //
    // `matched_policy_ttl` carries the policy's `ttl_seconds`; we feed
    // it to `put_with_ttl` so each entry expires per its own policy,
    // not the cache backend's global fallback. Backends without
    // per-entry support (defined via `Cache::put_with_ttl`'s default
    // impl) silently fall back to `put`.
    if let (Some(ttl), Some(cache), Some(key)) =
        (matched_policy_ttl, state.cache.as_ref(), cache_key.as_ref())
    {
        if let Err(err) = cache.put_with_ttl(key, upstream.clone(), ttl).await {
            tracing::warn!(error = %err, key = %key, "cache write failed");
        }
    }

    let mut response = Json(render_response(now, upstream)).into_response();
    if matches!(cache_status, CacheStatus::Miss) {
        // Miss header only when the cache was actually consulted —
        // policy-disabled requests have no cache header at all so a
        // user can tell at a glance whether the gate was open.
        response
            .headers_mut()
            .insert(CACHE_HEADER, HeaderValue::from_static("miss"));
    }

    Ok(Success {
        response,
        provider: provider_name,
        model_id,
        prompt_tokens: Some(prompt),
        completion_tokens: Some(completion),
        total_tokens: Some(total),
        cached_prompt_tokens,
        reasoning_tokens,
        cache_creation_tokens,
        cache_read_tokens,
        provider_request_id,
        provider_model_version,
        finish_reason,
        cost_usd,
        bypass_reason,
        cache_status,
        // Cache-saved counters are zero on the upstream-served path —
        // the request *did* hit the upstream, no work was saved.
        cache_hit_saved_input_tokens: 0,
        cache_hit_saved_output_tokens: 0,
        telemetry_handled_by_stream: false,
    })
}

/// Wire-shape label for `FinishReason`. cp-api stores this verbatim
/// in `dpmgr_usage_events.finish_reason`; the dashboard reads it back
/// to distinguish normal stops from truncation / content_filter.
fn finish_reason_label(reason: &aisix_gateway::FinishReason) -> String {
    use aisix_gateway::FinishReason;
    match reason {
        FinishReason::Stop => "stop".into(),
        FinishReason::Length => "length".into(),
        FinishReason::ContentFilter => "content_filter".into(),
        FinishReason::ToolCalls => "tool_calls".into(),
        FinishReason::Other(s) => s.clone(),
    }
}

fn record_success(
    metrics: &Metrics,
    provider: &str,
    model: &str,
    status: u16,
    s: &Success,
    elapsed: Duration,
) {
    let outcome = RequestOutcome::from_status(status);
    metrics.record_request(provider, model, status, outcome, elapsed);
    if let Some(total) = s.total_tokens {
        metrics.record_tokens(provider, model, total);
    }
}

/// Push one telemetry event onto the CP-side sink **and** fan it out
/// to every per-env OTLP/HTTP exporter in the live snapshot.
/// Non-blocking on both legs: the CP sink drops on full queue, the
/// OTLP fan-out detaches a tokio task per exporter. Centralised here
/// so success / error / streaming / cache-hit paths share one event
/// construction and the two emit legs stay in lockstep.
#[allow(clippy::too_many_arguments)]
fn emit_usage_event(
    state: &ProxyState,
    request_id: &str,
    model_id: &str,
    api_key_id: &str,
    status_code: u16,
    elapsed: Duration,
    prompt_tokens: u32,
    completion_tokens: u32,
    extras: UsageExtras,
    cost_usd: f64,
    guardrail_blocked: bool,
) {
    let event = UsageEvent {
        request_id: request_id.to_string(),
        // RFC 3339 UTC. cp-api parses with time.Parse(time.RFC3339, ...);
        // chrono's `to_rfc3339_opts(Secs, true)` emits the trailing Z.
        occurred_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        model_id: model_id.to_string(),
        api_key_id: api_key_id.to_string(),
        prompt_tokens,
        completion_tokens,
        cached_prompt_tokens: extras.cached_prompt_tokens,
        reasoning_tokens: extras.reasoning_tokens,
        cache_creation_tokens: extras.cache_creation_tokens,
        cache_read_tokens: extras.cache_read_tokens,
        latency_ms: elapsed.as_millis().min(u32::MAX as u128) as u32,
        status_code,
        provider_request_id: extras.provider_request_id,
        provider_model_version: extras.provider_model_version,
        finish_reason: extras.finish_reason,
        cost_usd,
        guardrail_blocked,
        guardrail_bypassed_reason: extras.bypass_reason,
        cache_status: extras.cache_status,
        cache_hit_saved_input_tokens: extras.cache_hit_saved_input_tokens,
        cache_hit_saved_output_tokens: extras.cache_hit_saved_output_tokens,
        // chat.rs is the OpenAI-shape /v1/chat/completions handler.
        // /v1/responses / /v1/embeddings / /v1/audio* / /v1/images* /
        // /v1/rerank don't emit UsageEvents today; when they do they
        // also pass `"openai"` here.
        inbound_protocol: "openai".to_string(),
    };
    state.usage_sink.try_emit(event.clone());
    // Per-env OTLP/HTTP fan-out. The snapshot's exporter table is
    // empty for envs that haven't configured any, so this is a cheap
    // no-op on the common path. Spawned tasks own the POST work and
    // never block the request return.
    let snap = state.snapshot.load();
    let exporters = snap.observability_exporters.entries();
    state
        .otlp_fan_out
        .fan_out(&event, exporters.iter().map(|e| &e.value));
}

/// Provider-detail bundle for `emit_usage_event`. Grouped here so the
/// helper signature stays under the clippy-too-many-arguments bar
/// without losing the structural relationship between the seven new
/// fields. All seven default to "" / 0, which is exactly the
/// "no provider info / no cache or reasoning detail" case (error path,
/// streaming pre-Phase-2).
#[derive(Default)]
struct UsageExtras {
    cached_prompt_tokens: u32,
    reasoning_tokens: u32,
    cache_creation_tokens: u32,
    cache_read_tokens: u32,
    provider_request_id: String,
    provider_model_version: String,
    finish_reason: String,
    /// Set when at least one guardrail returned `Bypass` for this
    /// request (remote-API guardrail upstream unreachable +
    /// `fail_open=true`). Goes onto
    /// `dpmgr_usage_events.guardrail_bypassed_reason`. Default empty
    /// string = no bypass; cp-api stores NULL in that case.
    bypass_reason: String,
    /// Lowercased `CacheStatus` (`"hit"` / `"miss"` / `"disabled"`).
    /// Empty default for the error path where the cache lookup never
    /// fired. Goes onto `dpmgr_usage_events.cache_status`.
    cache_status: String,
    /// On a cache HIT, the cached response's prompt + completion
    /// tokens. Zero otherwise. cp-api derives `cost_saved_usd` on
    /// ingest from these + its pricing catalog (see #88).
    cache_hit_saved_input_tokens: u32,
    cache_hit_saved_output_tokens: u32,
}

fn record_error(metrics: &Metrics, err: &ProxyError, model: &str, status: u16, elapsed: Duration) {
    let outcome = RequestOutcome::from_status(status);
    // Provider is unknown for pre-dispatch errors (auth, 404, etc.).
    metrics.record_request("unknown", model, status, outcome, elapsed);
    if let ProxyError::RateLimit(rl) = err {
        metrics.record_ratelimit_rejection(&rl.scope().to_string());
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_access_log(
    method: &str,
    path: &str,
    status: u16,
    latency: Duration,
    provider: Option<&str>,
    model: Option<&str>,
    api_key_id: Option<&str>,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    total_tokens: Option<u64>,
    request_id: &str,
) {
    AccessLog {
        method,
        path,
        status,
        latency,
        provider,
        model,
        api_key_id,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        request_id,
    }
    .emit();
}

fn created_ts() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Build the SSE stream for a streaming chat response.
///
/// As chunks flow through, we observe each chunk's `usage` field. The
/// upstream typically emits the final usage block on the *last* chunk
/// (OpenAI when `stream_options.include_usage=true`; Anthropic on the
/// `message_delta` carrying `output_tokens`). We carry forward the
/// most recently seen `total_tokens` and, on stream end, hand it to
/// `on_complete` so the caller can post-account it against TPM/TPD.
///
/// Pre-fix the streaming path called `reservation.commit_tokens(0)`
/// up front and never revisited the counter, leaving TPM caps silently
/// bypassed for all streaming traffic and the cost/usage telemetry
/// reporting `$0`. See issue #108.
///
/// `on_complete` runs **once** at end-of-stream. It also runs on the
/// disconnect path (the Drop of the upstream Stream returning `None`
/// — the same loop exit). Errors mid-stream still terminate the loop
/// normally, so partial-response cases commit whatever tokens were
/// observed.
/// What `build_sse_stream` extracts from the upstream stream and hands
/// to `on_complete` once the terminal chunk arrives. Sourced from the
/// `usage` block on whichever chunk carried it (typically the last one
/// before `[DONE]`) plus the most-recently-seen `chunk.id` /
/// `chunk.model` / `chunk.finish_reason`. Each numeric field defaults
/// to 0 (and string field to "") when the upstream never emits a
/// `usage` block — for example, an OpenAI streaming request without
/// `stream_options.include_usage=true`. Callers are responsible for
/// treating 0 as "no signal" the same way the non-streaming path
/// treats `prompt_tokens.unwrap_or(0)`.
#[derive(Default)]
struct StreamCompletion {
    prompt_tokens: u32,
    completion_tokens: u32,
    /// `u64` because the rate-limit accounting consumer (TPM cap)
    /// uses u64; cp-api's wire-shape `prompt_tokens` is u32 but
    /// cumulative-tokens accounting can overflow u32 over a long key.
    total_tokens: u64,
    cached_prompt_tokens: u32,
    reasoning_tokens: u32,
    cache_creation_tokens: u32,
    cache_read_tokens: u32,
    provider_request_id: String,
    provider_model_version: String,
    finish_reason: String,
    /// `true` when an OUTPUT guardrail blocked the response at
    /// end-of-stream (per #204). Read by the on_complete telemetry
    /// closure so `usage_events.guardrail_blocked` reflects the
    /// streaming-blocked path the same way the non-streaming path
    /// already does.
    guardrail_blocked: bool,
    /// First Bypass reason observed at end-of-stream (per #204
    /// audit H1). The non-streaming path captures Bypass into the
    /// telemetry envelope so operators can audit which policy
    /// fail-opened on a streamed response. Empty string = no bypass.
    /// First-bypass-wins matches the non-streaming convention.
    bypass_reason: String,
}

/// Parameters needed to run output-guardrail evaluation at
/// end-of-stream. Per #204 the streaming path used to skip output
/// guardrails entirely — a `kind: "keyword"` deny-list could be
/// trivially bypassed by setting `stream: true`. Buffer-then-check
/// is the right cadence for blocking guardrails (per-chunk evaluation
/// would still leak prefix bytes 1..N-1 by the time chunk N matches).
struct StreamGuardrailContext {
    chain: Arc<dyn aisix_guardrails::Guardrail>,
    /// Surface in tracing only; the wire envelope is intentionally
    /// generic per #153 ("response blocked by content policy").
    model_name: String,
}

/// Fires `on_complete` with whatever `StreamCompletion` has been
/// accumulated when the guard is dropped. The async_stream body holds
/// this guard for the whole stream lifetime so on_complete fires
/// reliably on BOTH normal completion AND mid-stream cancellation.
///
/// Why this is necessary: code AFTER a `yield` in `async_stream::stream!{}`
/// only runs when the consumer pulls. If axum drops the response
/// future (client disconnect, request timeout, etc.), the generator
/// is dropped at its last suspension point and post-yield code never
/// runs. Pre-Drop-guard, that meant a streaming chat with a client
/// disconnect emitted ZERO telemetry events — the customer was billed
/// upstream, the gateway recorded nothing. Drop runs on cancellation,
/// so the captured `StreamCompletion` (potentially zeros if the
/// disconnect beat the upstream's `usage` chunk) is always shipped.
struct CompleteOnDrop<F: FnOnce(StreamCompletion)> {
    /// `Option<(closure, accumulator)>` — Drop calls the closure
    /// exactly once via `.take()`, so manual disarm before the
    /// natural drop is also safe (we don't currently use that, but
    /// it leaves the door open).
    slot: Option<(F, StreamCompletion)>,
}

impl<F: FnOnce(StreamCompletion)> CompleteOnDrop<F> {
    fn comp(&mut self) -> &mut StreamCompletion {
        &mut self
            .slot
            .as_mut()
            .expect("CompleteOnDrop guard accessed after take")
            .1
    }
}

impl<F: FnOnce(StreamCompletion)> Drop for CompleteOnDrop<F> {
    fn drop(&mut self) {
        if let Some((f, c)) = self.slot.take() {
            f(c);
        }
    }
}

fn build_sse_stream<F>(
    upstream: aisix_gateway::ChatChunkStream,
    created: i64,
    output_guardrail: Option<StreamGuardrailContext>,
    on_complete: F,
) -> impl Stream<Item = Result<Event, Infallible>>
where
    F: FnOnce(StreamCompletion) + Send + 'static,
{
    async_stream::stream! {
        // Hold on_complete + the running StreamCompletion accumulator
        // inside a Drop guard so on_complete fires even on client-
        // disconnect cancellation. See `CompleteOnDrop` above for the
        // why; tl;dr without this, async_stream! body code after a
        // yield only runs on consumer pulls.
        let mut guard = CompleteOnDrop {
            slot: Some((on_complete, StreamCompletion::default())),
        };
        futures::pin_mut!(upstream);
        // Per #204: accumulate the assistant's content across chunks
        // so the output guardrail can evaluate the full response at
        // end-of-stream. Allocates only when an output guardrail is
        // configured AND the upstream actually emits content; for
        // requests without an output guardrail this is a no-op
        // borrow of the empty string.
        let mut content_buffer = if output_guardrail.is_some() {
            Some(String::new())
        } else {
            None
        };
        // Accumulate the upstream's `usage` block + per-chunk metadata
        // across the stream. Providers typically populate `usage` on
        // the terminal chunk only; using "max" rather than "last" makes
        // the bookkeeping robust to a provider that double-emits.
        // `chunk.id` / `chunk.model` / `chunk.finish_reason` use
        // last-seen-wins because those are stable per stream.
        //
        // Per docs/api-proxy.md §5: "If the upstream stream
        // terminates abnormally, aisix sends a final error chunk
        // and closes the response without `[DONE]`." Track whether
        // an error has been yielded so we can skip the closing
        // `[DONE]` — without this skip, a downstream SDK that
        // treats `[DONE]` as clean-completion would mis-interpret
        // the truncated response as a successful one.
        let mut errored = false;
        while let Some(item) = upstream.next().await {
            let ev = match item {
                Ok(chunk) => {
                    let comp = guard.comp();
                    if !chunk.id.is_empty() {
                        comp.provider_request_id = chunk.id.clone();
                    }
                    if !chunk.model.is_empty() {
                        comp.provider_model_version = chunk.model.clone();
                    }
                    if let Some(fr) = chunk.finish_reason.as_ref() {
                        comp.finish_reason = finish_reason_label(fr);
                    }
                    // Per #204: accumulate the assistant's content
                    // when an output guardrail is configured. Skip
                    // entirely when none is configured to avoid the
                    // allocation on the hot path.
                    if let (Some(buf), Some(text)) = (
                        content_buffer.as_mut(),
                        chunk.delta.content.as_deref(),
                    ) {
                        buf.push_str(text);
                    }
                    if let Some(u) = chunk.usage.as_ref() {
                        if u.prompt_tokens > comp.prompt_tokens {
                            comp.prompt_tokens = u.prompt_tokens;
                        }
                        if u.completion_tokens > comp.completion_tokens {
                            comp.completion_tokens = u.completion_tokens;
                        }
                        let t = u.total_tokens as u64;
                        if t > comp.total_tokens {
                            comp.total_tokens = t;
                        }
                        if u.cached_prompt_tokens > comp.cached_prompt_tokens {
                            comp.cached_prompt_tokens = u.cached_prompt_tokens;
                        }
                        if u.reasoning_tokens > comp.reasoning_tokens {
                            comp.reasoning_tokens = u.reasoning_tokens;
                        }
                        if u.cache_creation_tokens > comp.cache_creation_tokens {
                            comp.cache_creation_tokens = u.cache_creation_tokens;
                        }
                        if u.cache_read_tokens > comp.cache_read_tokens {
                            comp.cache_read_tokens = u.cache_read_tokens;
                        }
                    }
                    let rendered = render_chunk(created, chunk);
                    match serde_json::to_string(&rendered) {
                        Ok(json) => Event::default().data(json),
                        Err(err) => {
                            errored = true;
                            Event::default()
                                .event("error")
                                .data(error_frame_payload("internal_error", &err.to_string()))
                        }
                    }
                }
                Err(err) => {
                    errored = true;
                    let etype = err.error_type();
                    Event::default()
                        .event("error")
                        .data(error_frame_payload(etype, &err.to_string()))
                }
            };
            yield Ok::<_, Infallible>(ev);
        }
        // Per #204: run the output guardrail on the accumulated
        // assistant content BEFORE emitting `[DONE]`. Buffer-then-
        // check is the right cadence for a blocking guardrail:
        // per-chunk evaluation would still leak prefix bytes
        // 1..N-1 to the caller by the time chunk N matches the
        // forbidden literal — the secret reaches the wire
        // regardless. We accept the latency cost (whole completion
        // buffered) in exchange for the security control.
        //
        // Skip the check entirely when:
        //   - no output guardrail is configured (`content_buffer` is `None`)
        //   - the upstream stream errored (already sending error frame; no `[DONE]`)
        // The `errored` skip means a partially-streamed forbidden
        // literal would have already reached the caller — but per
        // docs §5 abnormal termination, the gateway has already
        // signaled the failure via SSE error event. Running the
        // guardrail on the partial would only add a second error
        // frame, not retroactively redact the leak.
        if !errored {
            if let (Some(content), Some(ctx)) = (content_buffer.as_ref(), output_guardrail.as_ref()) {
                let synthesized = aisix_gateway::ChatResponse {
                    id: guard.comp().provider_request_id.clone(),
                    model: guard.comp().provider_model_version.clone(),
                    message: aisix_gateway::ChatMessage::assistant(content.clone()),
                    finish_reason: aisix_gateway::FinishReason::Stop,
                    usage: aisix_gateway::UsageStats::new(
                        guard.comp().prompt_tokens,
                        guard.comp().completion_tokens,
                    ),
                };
                match ctx.chain.check_output(&synthesized).await {
                    aisix_guardrails::GuardrailVerdict::Block { reason } => {
                        // Mirror the non-streaming path's #153
                        // redaction contract: the wire-level message
                        // is generic ("response blocked by content
                        // policy"), and the rich verdict reason
                        // (which carries the matched-pattern detail)
                        // goes to operator logs only.
                        tracing::warn!(
                            guardrail_hook = "output",
                            model = %ctx.model_name,
                            reason = %reason,
                            "guardrail blocked streaming response",
                        );
                        errored = true;
                        guard.comp().guardrail_blocked = true;
                        yield Ok::<_, Infallible>(
                            Event::default()
                                .event("error")
                                .data(error_frame_payload(
                                    "content_filter",
                                    "response blocked by content policy",
                                )),
                        );
                    }
                    aisix_guardrails::GuardrailVerdict::Bypass { reason } => {
                        // Per #204 audit H1: capture an output
                        // Bypass at end-of-stream so the post-stream
                        // telemetry payload carries it (parallel to
                        // the non-streaming path's first-bypass-wins
                        // capture). An input-side bypass that fired
                        // earlier already populated the closure's
                        // `bypass_reason_for_telem` snapshot; only
                        // record the output bypass when the slot is
                        // still empty.
                        let comp = guard.comp();
                        if comp.bypass_reason.is_empty() {
                            comp.bypass_reason = reason;
                        }
                    }
                    aisix_guardrails::GuardrailVerdict::Allow => {}
                }
            }
        }
        // Stream completed normally. Yield [DONE] BEFORE the guard
        // drops at end-of-scope so the SSE sentinel reaches the
        // client before the on_complete telemetry fan-out. (Any
        // ordering relationship between the two is non-causal — both
        // are independent side effects on this thread — but matching
        // the prior "on_complete then [DONE]" intent feels right.)
        // For providers that don't emit `usage` in the stream the
        // accumulator's numeric fields stay 0; on_complete callers
        // must treat 0 as "no signal" (cp-api does — its pricing
        // catalog falls back to the standard rate when absent).
        //
        // Per docs §5: skip `[DONE]` on abnormal termination so SDK
        // consumers can detect truncation. The `errored` flag is
        // set by the loop above whenever an error event is yielded
        // OR by the output-guardrail check above on a Block verdict.
        if !errored {
            yield Ok::<_, Infallible>(Event::default().data("[DONE]"));
        }
        // `guard` drops here. On client disconnect, the generator
        // drops at the suspension point inside the loop; Drop fires
        // there with whatever StreamCompletion has been captured up
        // to that point.
    }
}

/// Build the `data:` payload for an SSE `event: error` frame.
/// The OpenAI Node SDK calls `JSON.parse(sse.data)` BEFORE checking
/// `sse.event === "error"`, so a plain-string payload yields a
/// `SyntaxError` ("Could not parse message into JSON: ...") on the
/// SDK side rather than the typed `APIError` callers expect. Emit
/// the OpenAI error envelope shape per
/// <https://platform.openai.com/docs/guides/error-codes/api-errors>:
/// `{"error": {"message": "...", "type": "..."}}`.
fn error_frame_payload(error_type: &str, message: &str) -> String {
    serde_json::to_string(&serde_json::json!({
        "error": {
            "message": message,
            "type": error_type,
        }
    }))
    // unreachable in practice — `serde_json::to_string` of a
    // `Value` cannot fail. The fallback emits a minimal valid
    // envelope so SDK consumers' `JSON.parse` still succeeds.
    .unwrap_or_else(|_| {
        r#"{"error":{"message":"error","type":"internal_error"}}"#.into()
    })
}
