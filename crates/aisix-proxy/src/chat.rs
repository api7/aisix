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
use aisix_obs::{
    AccessLog, LlmUsage, Metrics, RequestLabels, RequestOutcome, RoutingAttemptEvent, UsageEvent,
    UsageLabels,
};
use axum::extract::State;
use axum::http::HeaderValue;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::{Stream, StreamExt};
use std::convert::Infallible;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::auth::AuthenticatedKey;
use crate::error::ProxyError;
use crate::render::{render_chunk, render_response};
use crate::request_id::new_request_id;
use crate::routing::is_retryable;
use crate::state::ProxyState;

/// Header set on every non-streaming response indicating whether the
/// response came from the cache (`hit`) or the upstream (`miss`).
pub const CACHE_HEADER: &str = "x-aisix-cache";

/// Default Retry-After (in seconds) returned to the client when every
/// candidate is background-unhealthy and no cooldown timer is available
/// to derive a more precise hint. Operators tune per-model cooldown
/// TTLs via `cooldown.default_seconds`; this is only the all-unhealthy
/// fallback for the `on_all_filtered: fail` path.
const FALLBACK_ALL_UNHEALTHY_RETRY_AFTER: Duration = Duration::from_secs(30);

#[derive(Clone)]
struct AttemptModel {
    id: String,
    model: aisix_core::Model,
}

#[derive(Clone, Default)]
struct RoutingTelemetry {
    served_by_model: String,
    attempt_count: u32,
    fallback_count: u32,
    attempts: Vec<RoutingAttemptEvent>,
}

impl RoutingTelemetry {
    fn attempts_json(&self) -> Option<String> {
        if self.attempts.is_empty() {
            None
        } else {
            serde_json::to_string(&self.attempts).ok()
        }
    }
}

struct DispatchFailure {
    model_id: Option<String>,
    charge: Option<UpstreamCharge>,
    err: ProxyError,
    routing: RoutingTelemetry,
}

impl DispatchFailure {
    fn new(model_id: Option<String>, charge: Option<UpstreamCharge>, err: ProxyError) -> Self {
        Self {
            model_id,
            charge,
            err,
            routing: RoutingTelemetry::default(),
        }
    }

    fn with_routing(mut self, routing: RoutingTelemetry) -> Self {
        self.routing = routing;
        self
    }
}

fn routing_error_class(err: &BridgeError) -> &'static str {
    match err {
        BridgeError::Timeout { .. } => "timeout",
        BridgeError::UpstreamStatus { .. } => "upstream_status",
        BridgeError::UpstreamDecode(_) => "upstream_decode",
        BridgeError::Config(_) => "config",
        BridgeError::Transport(_) => "transport",
        BridgeError::StreamAborted => "stream_aborted",
    }
}

fn routing_error_status(err: &BridgeError) -> Option<u16> {
    match err {
        BridgeError::UpstreamStatus { status, .. } => Some(*status),
        _ => None,
    }
}

// Per-attempt cooldown decision lives in `crate::cooldown` so every
// dispatch path (chat, messages, responses, audio, rerank) shares the
// same logic. See cooldown.rs for the audit context (#264 H-1).
use crate::cooldown::decide_cooldown;

pub async fn chat_completions(
    State(state): State<ProxyState>,
    auth: AuthenticatedKey,
    // Issue #324: catch the JSON-extractor rejection here so we can
    // map it to OpenAI's documented 400 invalid_request_error wire
    // shape. Axum's `Json<T>` extractor returns 422 on JsonDataError
    // (valid-JSON-but-missing-required-field, e.g. no `model`),
    // which diverges from OpenAI — every SDK that branches on 400
    // vs 422 sees different semantics here than it does talking to
    // api.openai.com. Same discriminate-then-map pattern as
    // messages.rs (#336) which already handles this for /v1/messages.
    body: Result<Json<ChatFormat>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let started = Instant::now();
    let method = "POST";
    let path = "/v1/chat/completions";
    let req = match body {
        Ok(Json(r)) => r,
        Err(rej) => {
            use axum::extract::rejection::JsonRejection;
            use axum::http::StatusCode;
            // BytesRejection → distinguish 413 (PAYLOAD_TOO_LARGE,
            // real per-extractor cap exceeded) from 400 (transport-
            // side read failure). `JsonRejection` is `#[non_exhaustive]`
            // so the fallback `_` arm catches today's JsonDataError
            // (the #324 case) / JsonSyntaxError / MissingJsonContentType
            // AND any future variant axum adds, defaulting to 400
            // until each new variant gets an explicit policy decision.
            return match rej {
                JsonRejection::BytesRejection(inner)
                    if inner.status() == StatusCode::PAYLOAD_TOO_LARGE =>
                {
                    ProxyError::RequestTooLarge {
                        limit_bytes: state.request_body_limit_bytes,
                    }
                }
                JsonRejection::BytesRejection(_) => {
                    ProxyError::InvalidRequest("failed to read request body".into())
                }
                _ => ProxyError::InvalidRequest("invalid JSON request body".into()),
            }
            .into_response();
        }
    };
    let request_id = new_request_id();
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
                &api_key_id,
                auth.key().team_id.as_deref(),
                auth.key().owner_id.as_deref(),
                status,
                &success,
                elapsed,
            );
            let routing_attempts_json = success.routing.attempts_json();
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
                &success.routing,
                routing_attempts_json.as_deref(),
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
                        ttft_ms: 0,
                        routing: success.routing.clone(),
                        provider_key_id: success.provider_key_id.clone(),
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
                state.metrics.set_rate_limit_remaining(
                    &api_key_id,
                    &model_name,
                    rl_status.rpm_remaining(),
                    rl_status.tpm_remaining(),
                );
            }
            // Correlation / routing headers.
            if let Ok(v) = axum::http::HeaderValue::try_from(request_id.as_str()) {
                success.response.headers_mut().insert("x-aisix-call-id", v);
            }
            // `x-aisix-served-by` exposes which routing target served
            // the request — see AISIX-Cloud#410. Only emitted when a
            // routing group was the entry point (direct models would
            // just echo `req.model`, which the body already carries).
            //
            // `HeaderValue::try_from` rejects CR/LF and non-visible
            // ASCII (RFC 7230) — correct from a response-splitting
            // standpoint, but if a routing target's `display_name`
            // carries such bytes the header is silently absent and
            // a customer debugging failover would see the same wire
            // shape as a direct-model response. Surface the rejection
            // in DP logs so operators can rename the offending target.
            if let Some(target) = success.served_by_target.as_deref() {
                match axum::http::HeaderValue::try_from(target) {
                    Ok(v) => {
                        success
                            .response
                            .headers_mut()
                            .insert("x-aisix-served-by", v);
                    }
                    Err(err) => {
                        tracing::warn!(
                            target_display_name = %target,
                            error = %err,
                            "target display_name is not a valid HTTP header value; \
                             omitting x-aisix-served-by — rename the target to use \
                             only visible ASCII (no CR/LF, no non-ASCII characters)"
                        );
                    }
                }
            }
            success.response
        }
        Err(failure) => {
            let DispatchFailure {
                model_id: resolved_model_id,
                charge,
                err,
                routing,
            } = failure;
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
            let routing_attempts_json = routing.attempts_json();
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
                &routing,
                routing_attempts_json.as_deref(),
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
                        ttft_ms: 0,
                        routing: c.routing,
                        provider_key_id: c.provider_key_id,
                    },
                ),
                None => {
                    let extras = UsageExtras {
                        routing,
                        ..UsageExtras::default()
                    };
                    (0, 0, extras)
                }
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
    provider_key_id: String,
    upstream_model: String,
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
    /// Display name of the routing target that actually served this
    /// request (AISIX-Cloud#410). Surfaces in the `x-aisix-served-by`
    /// response header so callers can tell which target inside a
    /// routing group won the failover loop.
    ///
    /// `None` in two cases:
    ///   - Direct (non-routing) model — `display_name` of the served
    ///     target equals `req.model`, so the header would be redundant.
    ///   - Cache hit — we don't know which target produced the stored
    ///     response. Re-stamping a stale name would lie.
    ///
    /// Streaming routing responses still set this to the selected target.
    /// The streaming path attempts only that target and does not fail over
    /// mid-stream, but the header remains useful because callers asked for
    /// the routing group's display name.
    served_by_target: Option<String>,
    routing: RoutingTelemetry,
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

/// Compute the value of [`Success::served_by_target`] for a request.
///
/// Centralises the "should we surface routing identity?" policy so
/// the same rule applies on every dispatch branch (non-streaming
/// success, streaming success, cache hit). The rule is:
///
/// - Routing-group request **and** an attempt won → `Some(target)`.
/// - Anything else (direct model, no winner, cache hit) → `None`.
///
/// `None` is the wire signal "this was not a routing request" — the
/// `x-aisix-served-by` response header is emitted only when the
/// returned `Option` is `Some`, so its presence alone tells callers
/// that failover routing fired. Direct-model responses must never
/// carry the header.
fn served_by_target_for_routing(
    is_routing_request: bool,
    served_target: Option<String>,
) -> Option<String> {
    if is_routing_request {
        served_target
    } else {
        None
    }
}

#[cfg(test)]
mod served_by_target_tests {
    use super::served_by_target_for_routing;

    #[test]
    fn direct_model_returns_none_even_when_a_target_was_captured() {
        // A direct model still walks the attempt loop (with one
        // candidate), so `served_target` may be `Some`. The routing
        // flag is what gates header emission — direct models must
        // never carry `x-aisix-served-by`.
        assert_eq!(
            served_by_target_for_routing(false, Some("gpt-4-direct".into())),
            None,
        );
    }

    #[test]
    fn direct_model_with_no_target_is_none() {
        assert_eq!(served_by_target_for_routing(false, None), None);
    }

    #[test]
    fn routing_request_passes_winning_target_through() {
        assert_eq!(
            served_by_target_for_routing(true, Some("healthy-target".into())),
            Some("healthy-target".into()),
        );
    }

    #[test]
    fn routing_request_with_no_winner_is_none() {
        // All targets failed: header omitted. The caller is already
        // surfacing an error response; routing identity is moot.
        assert_eq!(served_by_target_for_routing(true, None), None);
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
    /// UUID of the resolved ProviderKey. Threaded through so
    /// `emit_usage_event` can look up `telemetry_tags` even on the
    /// output-guardrail-block path (where dispatch reached the
    /// upstream and billed the request). Empty if for some reason the
    /// charge was assembled without a known PK.
    provider_key_id: String,
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
    routing: RoutingTelemetry,
}

async fn dispatch(
    state: &ProxyState,
    auth: &AuthenticatedKey,
    req: &ChatFormat,
    request_id: &str,
    started: Instant,
) -> Result<Success, DispatchFailure> {
    if req.messages.is_empty() {
        return Err(DispatchFailure::new(
            None,
            None,
            ProxyError::InvalidRequest("messages array must not be empty".into()),
        ));
    }

    let snapshot = state.snapshot.load();
    let virtual_entry = snapshot.models.get_by_name(&req.model).ok_or_else(|| {
        DispatchFailure::new(None, None, ProxyError::ModelNotFound(req.model.clone()))
    })?;
    let model_id = virtual_entry.id.clone();

    // Every error from here on attaches the resolved model_id so the
    // failure-path telemetry event in `chat_completions` can surface
    // which model the request targeted.
    // Pre-upstream errors carry `None` for the charge — by definition
    // they fire before any provider billing happens. The output-filter
    // path below is the only site that builds a `Some(UpstreamCharge)`
    // (manually, not via this helper).
    let with_model = |e: ProxyError| DispatchFailure::new(Some(model_id.clone()), None, e);

    if !auth.key().can_access(&req.model) {
        return Err(with_model(ProxyError::ModelForbidden(req.model.clone())));
    }

    // Resolve the per-request guardrail chain from the index.
    // Done once here; `resolved_chain` is reused for both the input
    // check below, the output check later, and the streaming output
    // guardrail context.
    let guardrail_ctx = aisix_guardrails::RequestContext {
        model_id: &model_id,
        api_key_id: &auth.entry.id,
        team_id: auth.key().team_id.as_deref(),
    };
    let resolved_chain: std::sync::Arc<dyn aisix_guardrails::Guardrail> =
        std::sync::Arc::new(state.guardrail_index.resolve(&guardrail_ctx));

    // Input guardrails. Run before reservation so a blocked prompt
    // doesn't burn an RPM slot — content-policy refusals shouldn't
    // count against quota. Bypass (remote-API guardrail unavailable
    // + fail_open=true) doesn't short-circuit; the reason is stashed
    // and attached to the telemetry event when the request finishes.
    let mut bypass_reason: Option<String> = None;
    let mut rewritten_req: Option<Box<aisix_gateway::ChatFormat>> = None;
    match resolved_chain.check_input(req).await {
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
        GuardrailVerdict::Rewrite { payload } => {
            // A guardrail rewrote the prompt (e.g. PII scrubbing).
            // Substitute the returned payload for the original before
            // dispatching to the upstream. Downstream guardrails in the
            // chain already saw the rewritten form (propagated by
            // GuardrailChain::check_input via Cow<ChatFormat>).
            rewritten_req = Some(payload);
        }
    }
    // Shadow `req` with the possibly-rewritten payload. All downstream
    // code (budget check, routing, bridge dispatch) uses this reference.
    let req = rewritten_req.as_deref().unwrap_or(req);

    // Budget pre-check via cp-api. The DP no longer owns budget state;
    // cp-api returns a cached/live decision per api_key.
    let decision = state.budgets.check(&auth.entry.id).await;
    if let Some(budget) = decision.budget.as_ref() {
        record_budget_gauges(&state.metrics, auth, Some(budget));
    } else {
        record_budget_gauges(&state.metrics, auth, None);
    }
    if !decision.allowed {
        return Err(with_model(ProxyError::BudgetExceeded(
            decision.reason.unwrap_or_else(|| auth.entry.id.clone()),
        )));
    }

    // Resolve the attempt-list of underlying Model entries. For a
    // routing model we walk targets per the configured strategy; for a
    // single-provider Model we just dispatch to it directly.
    let attempt_models: Vec<AttemptModel> =
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
                resolved.push(AttemptModel {
                    id: target_entry.id.clone(),
                    model: target_entry.value.clone(),
                });
            }
            match filter_attempt_models(
                &state.runtime_status,
                resolved,
                routing.on_all_filtered_or_default(),
            ) {
                FilterOutcome::Selected(list) => list,
                FilterOutcome::AllUnhealthy { retry_after_secs } => {
                    tracing::warn!(
                        virtual_model = %req.model,
                        retry_after_secs,
                        "all routing candidates are unavailable; failing fast",
                    );
                    return Err(with_model(ProxyError::AllCandidatesUnavailable {
                        retry_after_secs,
                    }));
                }
            }
        } else {
            vec![AttemptModel {
                id: virtual_entry.id.clone(),
                model: virtual_entry.value.clone(),
            }]
        };

    // For non-routing requests, surface a misconfigured bridge as a
    // proper 503 rather than burying it inside a generic Bridge error.
    // Routing requests rely on the loop's `is_retryable` path so a
    // single bad provider doesn't take down the whole request.
    if attempt_models.len() == 1 {
        let only = &attempt_models[0].model;
        let _provider = crate::dispatch::require_provider(only).map_err(with_model)?;
        // Pre-flight the PK-based two-tier dispatch so a missing
        // family/specialized bridge surfaces as 503 here, before
        // we commit to a long upstream call.
        let pk_entry =
            crate::dispatch::resolve_provider_key(&snapshot, only).map_err(with_model)?;
        if crate::dispatch::resolve_bridge(&state.hub, &pk_entry.value, only.provider.as_deref())
            .is_none()
        {
            return Err(with_model(ProxyError::ProviderUnavailable));
        }
    }

    // Multi-layer rate-limit reservation (api_key inline + model inline + policies).
    let model_rl = crate::quota::ModelRateLimit::from_model(
        &req.model,
        &virtual_entry.id,
        &virtual_entry.value,
    );
    let reservation =
        crate::quota::enforce_rate_limit(state, auth, Some(&model_rl)).map_err(&with_model)?;

    let now = created_ts();

    // Streaming path: only attempt the first target. Streaming fallback
    // is genuinely hard (we'd have to buffer the stream to detect
    // failure mid-flight) and not worth the complexity for V1.
    if req.is_streaming() {
        let model = &attempt_models[0].model;
        let provider = crate::dispatch::require_provider(model).map_err(with_model)?;
        let pk_entry =
            crate::dispatch::resolve_provider_key(&snapshot, model).map_err(with_model)?;
        let bridge =
            crate::dispatch::resolve_bridge(&state.hub, &pk_entry.value, model.provider.as_deref())
                .ok_or_else(|| with_model(ProxyError::ProviderUnavailable))?;
        let model_arc = Arc::new(model.clone());
        let pk_arc = Arc::new(pk_entry.value.clone());
        let ctx = BridgeContext::new(request_id, model_arc, pk_arc);
        let upstream = match bridge.chat_stream(req, &ctx).await {
            Ok(upstream) => upstream,
            Err(err) => {
                let routing = if virtual_entry.value.routing.is_some() {
                    RoutingTelemetry {
                        served_by_model: String::new(),
                        attempt_count: 1,
                        fallback_count: 0,
                        attempts: vec![RoutingAttemptEvent {
                            model: model.display_name.clone(),
                            attempt: 1,
                            status: routing_error_status(&err),
                            error: routing_error_class(&err).to_string(),
                            success: false,
                        }],
                    }
                } else {
                    RoutingTelemetry::default()
                };
                return Err(with_model(ProxyError::Bridge(err)).with_routing(routing));
            }
        };
        // Drop the reservation now: concurrency releases on all layers.
        // RPM was already counted by pre_commit. TPM is updated
        // retroactively on stream-end by `add_tokens_post_stream`.
        let post_stream_keys = reservation.keys();
        drop(reservation);
        // Capture everything the stream-completion callback needs so
        // it can fire `emit_usage_event` once the terminal SSE chunk
        // has yielded its `usage` block. Telemetry emission has to
        // wait until end-of-stream because OpenAI / Anthropic only
        // populate `usage` on the last chunk; emitting at handler
        // return (the non-streaming path's spot) would record zeros.
        let limiter = Arc::clone(&state.limiter);
        let state_for_telem = state.clone();
        let metrics_for_stream = state.metrics.clone();
        let request_id_for_telem = request_id.to_string();
        let model_id_for_telem = model_id.clone();
        let api_key_id_for_telem = auth.entry.id.clone();
        let team_id_for_metrics = auth.key().team_id.clone();
        let owner_id_for_metrics = auth.key().owner_id.clone();
        let provider_for_metrics = provider.to_ascii_lowercase();
        let model_for_metrics = req.model.clone();
        let provider_key_id_for_metrics = pk_entry.id.clone();
        // Captured for the stream-end telemetry closure so
        // emit_usage_event can look up `telemetry_tags` for per-PK
        // attribution (#302 M17 / AISIX-Cloud#436). The metrics
        // variant above is `&str`-scoped to inner scopes that consume
        // it as a borrow; the telem variant is owned for the move
        // into the on_complete closure.
        let provider_key_id_for_telem = pk_entry.id.clone();
        let upstream_model_for_metrics = model.upstream_model().unwrap_or("unknown").to_string();
        let bypass_reason_for_telem = bypass_reason.clone().unwrap_or_default();
        let stream_routing = if virtual_entry.value.routing.is_some() {
            RoutingTelemetry {
                served_by_model: model.display_name.clone(),
                attempt_count: 1,
                fallback_count: 0,
                attempts: vec![RoutingAttemptEvent {
                    model: model.display_name.clone(),
                    attempt: 1,
                    status: Some(200),
                    error: String::new(),
                    success: true,
                }],
            }
        } else {
            RoutingTelemetry::default()
        };
        let stream_routing_for_telem = stream_routing.clone();
        // Per #204: pass the resolved guardrail chain so the streaming
        // path can run output guardrails at end-of-stream
        // (buffer-then-check). Mirrors the non-streaming
        // `resolved_chain.check_output(...)` call site below.
        //
        // Fast-path: skip the context entirely when the resolved chain
        // for this request is empty (no attachment matched). When `None`,
        // `build_sse_stream` skips per-chunk accumulation — both noise on
        // the hot path for the dominant guardrail-free deployment.
        let stream_guardrail = if resolved_chain.is_empty() {
            None
        } else {
            Some(StreamGuardrailContext {
                chain: Arc::clone(&resolved_chain),
                model_name: req.model.clone(),
            })
        };
        let sse_stream = build_sse_stream(
            upstream,
            now,
            stream_guardrail,
            started,
            req.model.clone(),
            move |comp: StreamCompletion| {
                // Rate-limit accounting (TPM cap) for all layers.
                for key in &post_stream_keys {
                    limiter.add_tokens_post_stream(key, comp.total_tokens);
                }
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
                        ttft_ms: comp.ttft_ms,
                        routing: stream_routing_for_telem.clone(),
                        provider_key_id: provider_key_id_for_telem.clone(),
                    },
                    /* cost_usd */ 0.0,
                    comp.guardrail_blocked,
                );
                metrics_for_stream.record_llm_usage(
                    UsageLabels {
                        endpoint: "/v1/chat/completions",
                        inbound_protocol: "openai",
                        provider: &provider_for_metrics,
                        model: &model_for_metrics,
                        upstream_model: &upstream_model_for_metrics,
                        provider_key_id: &provider_key_id_for_metrics,
                        api_key_id: &api_key_id_for_telem,
                        team_id: team_id_for_metrics.as_deref().unwrap_or("unknown"),
                        owner_id: owner_id_for_metrics.as_deref().unwrap_or("unknown"),
                    },
                    LlmUsage {
                        input_tokens: comp.prompt_tokens,
                        output_tokens: comp.completion_tokens,
                        total_tokens: comp.total_tokens.min(u64::from(u32::MAX)) as u32,
                        spend_usd: 0.0,
                    },
                );
                metrics_for_stream.record_time_to_first_token(
                    UsageLabels {
                        endpoint: "/v1/chat/completions",
                        inbound_protocol: "openai",
                        provider: &provider_for_metrics,
                        model: &model_for_metrics,
                        upstream_model: &upstream_model_for_metrics,
                        provider_key_id: &provider_key_id_for_metrics,
                        api_key_id: &api_key_id_for_telem,
                        team_id: team_id_for_metrics.as_deref().unwrap_or("unknown"),
                        owner_id: owner_id_for_metrics.as_deref().unwrap_or("unknown"),
                    },
                    Duration::from_millis(u64::from(comp.ttft_ms)),
                );
            },
        );
        let response =
            Sse::new(sse_stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)));
        return Ok(Success {
            response: response.into_response(),
            provider: provider.to_ascii_lowercase(),
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
            provider_key_id: pk_entry.id.clone(),
            upstream_model: model.upstream_model().unwrap_or("unknown").to_string(),
            finish_reason: String::new(),
            bypass_reason: bypass_reason.clone(),
            // Streaming responses aren't cached at this layer — see
            // crates/aisix-cache/src/lib.rs. Always surface as
            // `disabled` on the streaming path.
            cache_status: CacheStatus::Disabled,
            cache_hit_saved_input_tokens: 0,
            cache_hit_saved_output_tokens: 0,
            telemetry_handled_by_stream: true,
            // Streaming attempts only `targets[0]` (no mid-stream
            // fallback), so on the routing path the served target is
            // unambiguously the first one. The helper enforces the
            // "direct model → None" rule consistently with the
            // non-streaming and cache paths.
            served_by_target: served_by_target_for_routing(
                virtual_entry.value.routing.is_some(),
                Some(model.display_name.clone()),
            ),
            routing: stream_routing,
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
                    .model
                    .provider
                    .as_deref()
                    .map(|p| p.to_ascii_lowercase())
                    .unwrap_or_else(|| "unknown".into());
                let provider_key_id = attempt_models[0]
                    .model
                    .provider_key_id
                    .clone()
                    .unwrap_or_else(|| "unknown".into());
                let upstream_model = attempt_models[0]
                    .model
                    .upstream_model()
                    .unwrap_or("unknown")
                    .to_string();
                let mut response = Json(render_response(now, cached, &req.model)).into_response();
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
                    provider_key_id,
                    upstream_model,
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
                    // Cache hits don't carry routing-target identity:
                    // the stored response is decoupled from whichever
                    // target produced it on the original miss. See the
                    // `served_by_target` field docs on `Success`.
                    served_by_target: None,
                    routing: RoutingTelemetry::default(),
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
    let mut chosen_provider_key_id: Option<String> = None;
    let mut chosen_upstream_model: Option<String> = None;
    // Display name of the target whose attempt finally succeeded. Used
    // to populate the `x-aisix-served-by` response header for routing
    // requests (AISIX-Cloud#410). Stays `None` until an attempt wins.
    let mut chosen_target_display_name: Option<String> = None;
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
    let is_routing_request = virtual_entry.value.routing.is_some();
    let mut routing = RoutingTelemetry::default();
    let mut last_attempted_target: Option<String> = None;

    for attempt in &attempt_models {
        let model = &attempt.model;
        let Some(provider) = model.provider.as_deref() else {
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
        // Two-tier dispatch via `Hub::dispatch_two_tier`: specialized
        // vendor (ProviderKey.provider) first, then adapter family
        // (ProviderKey.adapter). The legacy `Provider`-keyed registry
        // is gone after #302 Phase A. `resolve_bridge` carries a
        // one-cycle compat shim for pre-Phase-A PK rows that still
        // have empty `provider` and `adapter: None` on disk — those
        // resolve via `Model.provider` (passed below). Once cp-api
        // has backfilled every PK, the shim becomes unreachable.
        let Some(bridge) =
            crate::dispatch::resolve_bridge(&state.hub, &pk_entry.value, model.provider.as_deref())
        else {
            last_err = Some(BridgeError::Config(
                "no bridge registered for provider".into(),
            ));
            continue;
        };
        let model_arc = Arc::new(model.clone());
        let pk_arc = Arc::new(pk_entry.value.clone());
        let ctx = BridgeContext::new(request_id, model_arc, pk_arc);

        for attempt_idx in 0..=retries {
            if is_routing_request {
                if let Some(prev) = last_attempted_target.as_deref() {
                    if prev != model.display_name {
                        routing.fallback_count += 1;
                    }
                }
                last_attempted_target = Some(model.display_name.clone());
                routing.attempt_count += 1;
            }
            match bridge.chat(req, &ctx).await {
                Ok(resp) => {
                    state.health.record_success(&model.display_name);
                    state.runtime_status.mark_healthy(&attempt.id);
                    chosen_provider = Some(provider.to_ascii_lowercase());
                    chosen_provider_key_id = Some(pk_entry.id.clone());
                    chosen_upstream_model =
                        Some(model.upstream_model().unwrap_or("unknown").to_string());
                    chosen_target_display_name = Some(model.display_name.clone());
                    if is_routing_request {
                        routing.served_by_model = model.display_name.clone();
                        routing.attempts.push(RoutingAttemptEvent {
                            model: model.display_name.clone(),
                            attempt: (attempt_idx + 1) as u32,
                            status: Some(200),
                            error: String::new(),
                            success: true,
                        });
                    }
                    upstream = Some(resp);
                    break;
                }
                Err(err) => {
                    if is_routing_request {
                        routing.attempts.push(RoutingAttemptEvent {
                            model: model.display_name.clone(),
                            attempt: (attempt_idx + 1) as u32,
                            status: routing_error_status(&err),
                            error: routing_error_class(&err).to_string(),
                            success: false,
                        });
                    }
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
                    // Cooldown decision is independent of retry — a
                    // non-retryable 401 still cools down because the
                    // same key will keep failing for upcoming
                    // requests; a retryable 502 also cools down so
                    // the next request prefers a different target.
                    if let Some((ttl, reason)) =
                        decide_cooldown(&err, attempt.model.cooldown.as_ref())
                    {
                        state.runtime_status.mark_cooldown(&attempt.id, ttl, reason);
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
        return Err(with_model(ProxyError::Bridge(err)).with_routing(routing));
    };
    let provider_name = chosen_provider.unwrap_or_else(|| "unknown".into());
    let provider_key_id = chosen_provider_key_id.unwrap_or_else(|| "unknown".into());
    let upstream_model = chosen_upstream_model.unwrap_or_else(|| "unknown".into());

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

    match resolved_chain.check_output(&upstream).await {
        GuardrailVerdict::Allow => {}
        GuardrailVerdict::Rewrite { .. } => {
            // Output rewrites are not supported on the non-streaming path
            // in P0c (check_output on GuardrailChain already coerces Rewrite
            // to Allow internally; this arm is for future direct-check paths).
        }
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
                provider_key_id: provider_key_id.clone(),
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
                routing: routing.clone(),
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
            return Err(DispatchFailure::new(
                Some(model_id.clone()),
                Some(charge),
                ProxyError::ContentFiltered("response blocked by content policy".into()),
            )
            .with_routing(routing));
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

    let mut response = Json(render_response(now, upstream, &req.model)).into_response();
    if matches!(cache_status, CacheStatus::Miss) {
        // Miss header only when the cache was actually consulted —
        // policy-disabled requests have no cache header at all so a
        // user can tell at a glance whether the gate was open.
        response
            .headers_mut()
            .insert(CACHE_HEADER, HeaderValue::from_static("miss"));
    }

    // Header presence is the wire signal for "routing happened" —
    // see `served_by_target_for_routing`. The helper covers all
    // three branches (non-streaming, streaming, cache hit) with the
    // same policy so a refactor can't silently flip one of them.
    let served_by_target = served_by_target_for_routing(
        virtual_entry.value.routing.is_some(),
        chosen_target_display_name,
    );

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
        provider_key_id,
        upstream_model,
        finish_reason,
        cost_usd,
        bypass_reason,
        cache_status,
        // Cache-saved counters are zero on the upstream-served path —
        // the request *did* hit the upstream, no work was saved.
        cache_hit_saved_input_tokens: 0,
        cache_hit_saved_output_tokens: 0,
        telemetry_handled_by_stream: false,
        served_by_target,
        routing,
    })
}

/// Outcome of routing-candidate filtering. Lifts the "all candidates
/// excluded" case out into a typed result so the dispatch loop can
/// short-circuit to a 503 + Retry-After instead of sending traffic to
/// a target we just confirmed is bad.
enum FilterOutcome {
    /// At least one candidate survived the filter. The returned vector
    /// is the filtered attempt list, in the original strategy order
    /// minus the excluded entries.
    Selected(Vec<AttemptModel>),
    /// Every candidate is currently background-unhealthy and the
    /// routing model is configured with `on_all_filtered: fail`. The
    /// caller should surface a 503 with the supplied Retry-After hint
    /// (in seconds), if any.
    AllUnhealthy { retry_after_secs: Option<u64> },
}

fn filter_attempt_models(
    runtime_status: &crate::ModelRuntimeStatusTracker,
    attempts: Vec<AttemptModel>,
    policy: aisix_core::OnAllFilteredPolicy,
) -> FilterOutcome {
    use aisix_core::OnAllFilteredPolicy;

    let mut healthy = Vec::new();
    let mut cooldown_only = Vec::new();
    let mut unhealthy_count = 0usize;

    for attempt in attempts.iter().cloned() {
        let stale_after = attempt
            .model
            .background_model_check
            .as_ref()
            .map(|cfg| Duration::from_secs(cfg.stale_after_seconds));
        let snapshot = runtime_status.status_with_stale(&attempt.id, stale_after);
        match snapshot.status {
            crate::RuntimeStatus::Unhealthy => unhealthy_count += 1,
            crate::RuntimeStatus::Cooldown => cooldown_only.push(attempt),
            crate::RuntimeStatus::Healthy | crate::RuntimeStatus::NotApplicable => {
                healthy.push(attempt)
            }
        }
    }

    if !healthy.is_empty() {
        return FilterOutcome::Selected(healthy);
    }
    // No healthy candidates — prefer cooldown over unhealthy when
    // some non-unhealthy candidates exist. Sending to a target whose
    // cooldown timer hasn't expired is still better than sending to
    // a target that an active probe just confirmed is broken.
    if unhealthy_count < attempts.len() && !cooldown_only.is_empty() {
        let filtered: Vec<AttemptModel> = attempts
            .into_iter()
            .filter(|attempt| {
                let stale_after = attempt
                    .model
                    .background_model_check
                    .as_ref()
                    .map(|cfg| Duration::from_secs(cfg.stale_after_seconds));
                runtime_status.should_skip_for_routing(&attempt.id, stale_after)
                    != crate::RuntimeStatus::Unhealthy
            })
            .collect();
        return FilterOutcome::Selected(filtered);
    }
    // All candidates are excluded. Policy decides.
    //
    // Retry-After for the fail path is a coarse fallback (30s by
    // default — see FALLBACK_ALL_UNHEALTHY_RETRY_AFTER). We could
    // try to derive it from per-candidate cooldown timers, but the
    // categorisation above routes cooldown candidates into
    // `cooldown_only` (returned via the Selected branch above), so
    // by construction every candidate that reaches here is in the
    // background-unhealthy state and has no cooldown timer to read.
    match policy {
        OnAllFilteredPolicy::Fail => FilterOutcome::AllUnhealthy {
            retry_after_secs: Some(FALLBACK_ALL_UNHEALTHY_RETRY_AFTER.as_secs()),
        },
        OnAllFilteredPolicy::OriginalOrder => FilterOutcome::Selected(attempts),
    }
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

#[allow(clippy::too_many_arguments)]
fn record_success(
    metrics: &Metrics,
    provider: &str,
    model: &str,
    api_key_id: &str,
    team_id: Option<&str>,
    owner_id: Option<&str>,
    status: u16,
    s: &Success,
    elapsed: Duration,
) {
    let outcome = RequestOutcome::from_status(status);
    metrics.record_request(provider, model, status, outcome, elapsed);
    let request_labels = RequestLabels {
        endpoint: "/v1/chat/completions",
        inbound_protocol: "openai",
        provider,
        model,
        upstream_model: &s.upstream_model,
        provider_key_id: &s.provider_key_id,
        api_key_id,
        team_id: team_id.unwrap_or("unknown"),
        owner_id: owner_id.unwrap_or("unknown"),
        status,
        outcome,
    };
    metrics.record_proxy_request(request_labels, elapsed);
    metrics.record_llm_request(request_labels, elapsed);
    if let Some(total) = s.total_tokens {
        metrics.record_tokens(provider, model, total);
    }
    metrics.record_llm_usage(
        UsageLabels {
            endpoint: "/v1/chat/completions",
            inbound_protocol: "openai",
            provider,
            model,
            upstream_model: &s.upstream_model,
            provider_key_id: &s.provider_key_id,
            api_key_id,
            team_id: team_id.unwrap_or("unknown"),
            owner_id: owner_id.unwrap_or("unknown"),
        },
        LlmUsage {
            input_tokens: s.prompt_tokens.unwrap_or(0).min(u64::from(u32::MAX)) as u32,
            output_tokens: s.completion_tokens.unwrap_or(0).min(u64::from(u32::MAX)) as u32,
            total_tokens: s.total_tokens.unwrap_or(0).min(u64::from(u32::MAX)) as u32,
            spend_usd: s.cost_usd,
        },
    );
}

fn record_budget_gauges(
    metrics: &Metrics,
    auth: &AuthenticatedKey,
    budget: Option<&crate::budget::BudgetDetails>,
) {
    let labels = aisix_obs::BudgetLabels {
        api_key_id: &auth.entry.id,
        team_id: auth.key().team_id.as_deref().unwrap_or("unknown"),
        owner_id: auth.key().owner_id.as_deref().unwrap_or("unknown"),
    };
    if let Some(budget) = budget {
        metrics.set_budget_gauges(
            labels,
            aisix_obs::BudgetGauges {
                limit_usd: budget.limit_usd,
                spent_usd: budget.spent_usd,
                remaining_usd: budget.remaining_usd,
                reset_seconds: budget.reset_seconds,
            },
        );
    } else {
        metrics.clear_budget_gauges(labels);
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
    // Look up per-PK telemetry attribution tags from the live snapshot.
    // Empty `provider_key_id` (pre-dispatch error paths) → default
    // tags (all empty / false) → wire fields skip-serialize → cp-api
    // stores NULL. See AISIX-Cloud#436.
    let snap = state.snapshot.load();
    let tags = if !extras.provider_key_id.is_empty() {
        snap.provider_keys
            .get_by_id(&extras.provider_key_id)
            .map(|e| e.value.telemetry_tags.clone())
            .unwrap_or_default()
    } else {
        Default::default()
    };
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
        ttft_ms: extras.ttft_ms,
        // chat.rs is the OpenAI-shape /v1/chat/completions handler.
        // /v1/responses / /v1/embeddings / /v1/audio* / /v1/images* /
        // /v1/rerank don't emit UsageEvents today; when they do they
        // also pass `"openai"` here.
        inbound_protocol: "openai".to_string(),
        served_by_model: extras.routing.served_by_model,
        routing_attempt_count: extras.routing.attempt_count,
        routing_fallback_count: extras.routing.fallback_count,
        routing_attempts: extras.routing.attempts,
        // Per-PK telemetry attribution (#302 M17 / AISIX-Cloud#436).
        // Source struct is `aisix_core::TelemetryTags`; the wire
        // shape is flat strings + a bool, with skip_serializing_if
        // covering legacy PKs that pre-date attribution.
        // Each operator-defined string is run through `sanitize_tag`
        // as defence-in-depth against log/JSON injection downstream
        // (PR #382 audit MEDIUM-3; admission-side cap tracked
        // separately).
        provider_kind: sanitize_tag(tags.kind.unwrap_or_default()),
        provider_featured: tags.featured,
        branded_provider: sanitize_tag(tags.branded_provider.unwrap_or_default()),
        pk_label: sanitize_tag(tags.pk_label.unwrap_or_default()),
        byo_label: sanitize_tag(tags.byo_label.unwrap_or_default()),
    };
    // Handler label "chat" matches the documented enumeration for
    // `aisix_usage_events_emitted_total` (#408). Keep `&'static str`
    // so prometheus cardinality stays bounded.
    state.usage_sink.try_emit("chat", event.clone());
    // Per-env OTLP/HTTP fan-out. The snapshot's exporter table is
    // empty for envs that haven't configured any, so this is a cheap
    // no-op on the common path. Spawned tasks own the POST work and
    // never block the request return.
    let exporters = snap.observability_exporters.entries();
    state
        .otlp_fan_out
        .fan_out(&event, exporters.iter().map(|e| &e.value));
}

/// Defence-in-depth sanitiser for operator-defined `ProviderKey
/// .telemetry_tags` string fields before they hit the wire.
///
/// Tag values are operator-controlled (set via the dashboard's
/// provider-key form, persisted in etcd). A malicious operator with
/// PK-write privileges could craft a label like
/// `"production\u{0a}injected-internal-key: secret"` that, while
/// safely JSON-escaped on this gateway↔cp-api hop, may forge log
/// lines or muddle downstream consumers (cp-api logs, dashboards,
/// log-aggregation pipelines) if any of them ever uses
/// non-strict line-oriented parsing.
///
/// We mitigate two ways:
///   1. Strip ASCII control characters (`\n`, `\r`, `\0`, etc.)
///   2. Cap length at 256 chars
///
/// The right place to enforce this in depth is at PK admission
/// (cp-api / dashboard validation on `display_name` / tag fields).
/// This sanitiser is a belt-and-suspenders guard on the emit side
/// — it cannot prevent a malicious tag from being *stored*, but it
/// can prevent the stored value from corrupting downstream logs.
///
/// PR #382 audit MEDIUM-3.
pub(crate) fn sanitize_tag(s: String) -> String {
    if s.is_empty() {
        return s;
    }
    s.chars().filter(|c| !c.is_control()).take(256).collect()
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
    ttft_ms: u32,
    routing: RoutingTelemetry,
    /// UUID of the resolved ProviderKey. Used at emit time to look up
    /// `telemetry_tags` from the snapshot and populate UsageEvent's
    /// per-PK attribution fields (`provider_kind` / `provider_featured`
    /// / `branded_provider` / `pk_label` / `byo_label`).
    /// Empty for pre-dispatch error paths (auth fail, guardrail block
    /// before dispatch) where no ProviderKey was resolved — those
    /// emit events land in cp-api with the tag columns NULL.
    /// See AISIX-Cloud#436 / #302 M17.
    provider_key_id: String,
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
    routing: &RoutingTelemetry,
    routing_attempts: Option<&str>,
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
        served_by_model: if routing.served_by_model.is_empty() {
            None
        } else {
            Some(routing.served_by_model.as_str())
        },
        routing_attempt_count: if routing.attempt_count == 0 {
            None
        } else {
            Some(routing.attempt_count)
        },
        routing_fallback_count: if routing.fallback_count == 0 {
            None
        } else {
            Some(routing.fallback_count)
        },
        routing_attempts,
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
    /// Time to first token in milliseconds. Set once when the first
    /// Ok(chunk) arrives in `build_sse_stream`.
    ttft_ms: u32,
    /// Count of SSE events the **consumer actually pulled** from the
    /// stream — incremented on the post-yield resume in
    /// `build_sse_stream`. `async_stream::stream!` semantics: code
    /// AFTER a `yield` runs only when the consumer asks for the
    /// next item, so this counter is a reliable "delivered-to-the-
    /// client" signal even when the consumer disconnects mid-stream.
    ///
    /// `CompleteOnDrop::drop` uses this to gate `completion_tokens`:
    /// if the consumer disconnected before any chunk reached them
    /// (`chunks_delivered == 0`), the upstream's `usage` block is
    /// irrelevant — the customer must not be billed for tokens that
    /// never crossed the wire. Issue #419.
    chunks_delivered: u32,
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
    /// Shared counter of SSE events that crossed the wire to the
    /// consumer. Written by `DeliveryCounter` on each `poll_next ->
    /// Ready(Some(_))`. Read at Drop time to gate completion-side
    /// counters (issue #419).
    ///
    /// Why a shared atomic rather than a field on `StreamCompletion`
    /// incremented post-yield: `async_stream::stream!` resumes
    /// post-yield code only on the **next** consumer poll, so a
    /// consumer that pulls N chunks then disconnects leaves the
    /// counter at N-1, not N. Counting at the wrapper's `poll_next`
    /// returning `Ready(Some(_))` is exact — that's the moment the
    /// item handed to the consumer.
    delivered: Arc<AtomicU32>,
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
        if let Some((f, mut c)) = self.slot.take() {
            // Snapshot the wire-delivered count for telemetry visibility
            // AND for the cost-leak gate below.
            let delivered = self.delivered.load(Ordering::Relaxed);
            c.chunks_delivered = delivered;
            // Issue #419 — cost-leak gate. If the consumer disconnected
            // before any chunk was delivered (e.g. AbortController.abort()
            // mid-`upstream.next().await`, or axum dropped the response
            // future before the first chunk was pulled), the upstream's
            // `usage` block is meaningless for billing — no completion
            // tokens crossed the wire to the client. Zero out the
            // completion-side counters but keep `prompt_tokens` (the
            // prompt was processed by upstream regardless, and the
            // industry contract is "prompts always billed").
            if delivered == 0 {
                c.completion_tokens = 0;
                c.reasoning_tokens = 0;
                c.cache_creation_tokens = 0;
                c.cache_read_tokens = 0;
                c.total_tokens = c.prompt_tokens as u64;
            }
            f(c);
        }
    }
}

fn build_sse_stream<F>(
    upstream: aisix_gateway::ChatChunkStream,
    created: i64,
    output_guardrail: Option<StreamGuardrailContext>,
    started: Instant,
    // Customer-facing model name (alias / routing group), re-stamped
    // onto every SSE chunk's `model` field per AISIX-Cloud#410. Owned
    // so it can move into the `async_stream::stream!` closure.
    client_facing_model: String,
    on_complete: F,
) -> impl Stream<Item = Result<Event, Infallible>>
where
    F: FnOnce(StreamCompletion) + Send + 'static,
{
    // Shared counter for delivered SSE events. Written by the
    // `DeliveryCounter` wrapper below on each `poll_next ->
    // Ready(Some(_))`; read by `CompleteOnDrop::drop` to gate
    // completion-side counters when the consumer disconnected before
    // any chunk reached the wire (#419).
    let delivered = Arc::new(AtomicU32::new(0));
    let delivered_for_drop = Arc::clone(&delivered);
    let inner = async_stream::stream! {
        // Hold on_complete + the running StreamCompletion accumulator
        // inside a Drop guard so on_complete fires even on client-
        // disconnect cancellation. See `CompleteOnDrop` above for the
        // why; tl;dr without this, async_stream! body code after a
        // yield only runs on consumer pulls.
        let mut guard = CompleteOnDrop {
            slot: Some((on_complete, StreamCompletion::default())),
            delivered: delivered_for_drop,
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
        let mut first_chunk_seen = false;
        while let Some(item) = upstream.next().await {
            let ev = match item {
                Ok(chunk) => {
                    // Record TTFT on the first chunk carrying generated
                    // output (content or tool calls). Skip role-only
                    // chunks that OpenAI emits before actual tokens.
                    if !first_chunk_seen
                        && (chunk.delta.content.is_some() || chunk.delta.tool_calls.is_some())
                    {
                        first_chunk_seen = true;
                        guard.comp().ttft_ms =
                            started.elapsed().as_millis().min(u32::MAX as u128) as u32;
                    }
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
                    let rendered = render_chunk(created, chunk, &client_facing_model);
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
            // Delivery is counted by the outer DeliveryCounter wrapper
            // at poll_next time, not here — async_stream's yield
            // suspends BEFORE the consumer has actually pulled, so
            // a post-yield increment under-counts by 1 on every
            // abort path (#419, audit follow-up).
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
                    aisix_guardrails::GuardrailVerdict::Rewrite { .. } => {
                        // Output rewrites on the streaming path are not
                        // supported in P0c — GuardrailChain::check_output
                        // already coerces Rewrite to Allow internally;
                        // this arm handles future direct-check paths.
                    }
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
    };
    DeliveryCounter {
        inner: Box::pin(inner),
        delivered,
    }
}

/// Stream wrapper that increments `delivered` on every `poll_next ->
/// Ready(Some(_))`. This is the canonical "consumer actually received
/// this item" signal — it fires AT the moment axum's SSE driver pulls
/// the event off. Pair with `CompleteOnDrop` so the Drop guard can
/// read the exact count even on mid-stream cancellation. Issue #419.
///
/// The inner stream is `Pin<Box<dyn Stream<...> + Send>>` so this
/// wrapper itself is `Unpin` and avoids the `forbid(unsafe_code)`
/// pin-projection dance. The boxing is per-request and negligible
/// against the rest of the chat-completion hot path.
struct DeliveryCounter<T> {
    inner: std::pin::Pin<Box<dyn Stream<Item = T> + Send>>,
    delivered: Arc<AtomicU32>,
}

impl<T> Stream for DeliveryCounter<T> {
    type Item = T;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        match self.inner.as_mut().poll_next(cx) {
            std::task::Poll::Ready(Some(item)) => {
                self.delivered.fetch_add(1, Ordering::Relaxed);
                std::task::Poll::Ready(Some(item))
            }
            other => other,
        }
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

#[cfg(test)]
mod cooldown_tests {
    use super::*;
    use aisix_core::CooldownConfig;
    use std::time::Duration as StdDuration;

    fn upstream(status: u16) -> BridgeError {
        BridgeError::upstream_status(status, "boom")
    }

    fn upstream_with_retry_after(status: u16, secs: u64) -> BridgeError {
        BridgeError::upstream_status_with_retry_after(
            status,
            "rate limited",
            Some(StdDuration::from_secs(secs)),
        )
    }

    #[test]
    fn default_config_cooldowns_429() {
        let (ttl, reason) = decide_cooldown(&upstream(429), None).unwrap();
        assert_eq!(ttl, StdDuration::from_secs(30));
        assert_eq!(reason, "upstream_rate_limited");
    }

    #[test]
    fn default_config_cooldowns_401_even_though_non_retryable() {
        // H1 contract: 401 cools down. Non-retryable upstream errors
        // (auth failure) should still take the target out of rotation,
        // because the same key will keep failing on subsequent
        // requests. The retry-vs-cooldown split is the whole point.
        let (ttl, reason) = decide_cooldown(&upstream(401), None).unwrap();
        assert_eq!(ttl, StdDuration::from_secs(30));
        assert_eq!(reason, "upstream_auth_failure");
    }

    #[test]
    fn default_config_cooldowns_408() {
        let (_, reason) = decide_cooldown(&upstream(408), None).unwrap();
        assert_eq!(reason, "upstream_request_timeout");
    }

    #[test]
    fn default_config_cooldowns_5xx() {
        for status in [500, 502, 503, 504] {
            let (_, reason) = decide_cooldown(&upstream(status), None).unwrap();
            assert_eq!(reason, "upstream_server_error", "status={status}");
        }
    }

    #[test]
    fn default_config_skips_400_and_other_4xx() {
        // Caller bugs (400, 403, 422) are not cooldown signals — the
        // model didn't fail, the request did.
        assert!(decide_cooldown(&upstream(400), None).is_none());
        assert!(decide_cooldown(&upstream(403), None).is_none());
        assert!(decide_cooldown(&upstream(422), None).is_none());
    }

    #[test]
    fn default_config_cooldowns_timeout_and_transport_errors() {
        assert!(decide_cooldown(&BridgeError::Timeout { elapsed_ms: 30_000 }, None).is_some());
        assert!(decide_cooldown(&BridgeError::Transport("conn refused".into()), None).is_some());
        assert!(decide_cooldown(&BridgeError::StreamAborted, None).is_some());
        assert!(decide_cooldown(&BridgeError::UpstreamDecode("bad json".into()), None).is_some());
    }

    #[test]
    fn config_disabled_skips_cooldown() {
        let cfg = CooldownConfig {
            enabled: Some(false),
            ..Default::default()
        };
        assert!(decide_cooldown(&upstream(429), Some(&cfg)).is_none());
        assert!(decide_cooldown(&upstream(500), Some(&cfg)).is_none());
    }

    #[test]
    fn config_override_trigger_statuses_excludes_429() {
        // Operator wants 429 to NOT cool down (e.g. heavy retry policy
        // already handles burst). 500s still cool down.
        let cfg = CooldownConfig {
            trigger_statuses: Some(vec![500, 502, 503]),
            ..Default::default()
        };
        assert!(decide_cooldown(&upstream(429), Some(&cfg)).is_none());
        assert!(decide_cooldown(&upstream(503), Some(&cfg)).is_some());
    }

    #[test]
    fn honor_retry_after_uses_upstream_hint() {
        let (ttl, _) = decide_cooldown(&upstream_with_retry_after(429, 75), None).unwrap();
        assert_eq!(ttl, StdDuration::from_secs(75));
    }

    #[test]
    fn honor_retry_after_clamps_to_max_seconds() {
        // Upstream is misbehaving — Retry-After: 100000. Clamp to
        // configured max so we don't lose the target for hours.
        let cfg = CooldownConfig {
            max_seconds: Some(60),
            ..Default::default()
        };
        let (ttl, _) =
            decide_cooldown(&upstream_with_retry_after(429, 100_000), Some(&cfg)).unwrap();
        assert_eq!(ttl, StdDuration::from_secs(60));
    }

    #[test]
    fn honor_retry_after_disabled_falls_back_to_default() {
        let cfg = CooldownConfig {
            honor_retry_after: Some(false),
            default_seconds: Some(45),
            ..Default::default()
        };
        let (ttl, _) = decide_cooldown(&upstream_with_retry_after(429, 5), Some(&cfg)).unwrap();
        assert_eq!(ttl, StdDuration::from_secs(45));
    }

    #[test]
    fn trigger_on_timeout_false_disables_timeout_cooldown() {
        let cfg = CooldownConfig {
            trigger_on_timeout: Some(false),
            ..Default::default()
        };
        assert!(decide_cooldown(&BridgeError::Timeout { elapsed_ms: 1 }, Some(&cfg)).is_none());
    }

    #[test]
    fn config_error_never_cools_down() {
        // Misconfig = WE are wrong; cooling down doesn't help.
        assert!(decide_cooldown(&BridgeError::Config("bad key".into()), None).is_none());
    }
}

#[cfg(test)]
mod filter_tests {
    use super::*;
    use aisix_core::{Model, OnAllFilteredPolicy};
    use std::time::Duration as StdDuration;

    fn am(id: &str) -> AttemptModel {
        let model: Model = serde_json::from_str(&format!(
            r#"{{
              "display_name": "{id}",
              "provider": "openai",
              "model_name": "gpt-4o-mini",
              "provider_key_id": "pk-{id}"
            }}"#
        ))
        .unwrap();
        AttemptModel {
            id: id.to_string(),
            model,
        }
    }

    #[test]
    fn healthy_only_returns_all_healthy() {
        let t = crate::ModelRuntimeStatusTracker::new();
        let attempts = vec![am("a"), am("b")];
        match filter_attempt_models(&t, attempts, OnAllFilteredPolicy::Fail) {
            FilterOutcome::Selected(list) => {
                assert_eq!(list.len(), 2);
            }
            other => panic!(
                "expected Selected, got {:?}",
                std::mem::discriminant(&other)
            ),
        }
    }

    #[test]
    fn cooldown_skipped_when_healthy_present() {
        let t = crate::ModelRuntimeStatusTracker::new();
        t.mark_cooldown("a", StdDuration::from_secs(30), "retryable_failure");
        let attempts = vec![am("a"), am("b")];
        match filter_attempt_models(&t, attempts, OnAllFilteredPolicy::Fail) {
            FilterOutcome::Selected(list) => {
                assert_eq!(list.len(), 1);
                assert_eq!(list[0].id, "b");
            }
            _ => panic!("expected Selected"),
        }
    }

    #[test]
    fn all_unhealthy_fail_policy_returns_retry_after_hint() {
        // H3 contract: every candidate background-unhealthy, no
        // cooldown timer → return 503 + fallback Retry-After (30s
        // default). The dispatch loop converts this to a
        // ProxyError::AllCandidatesUnavailable.
        let t = crate::ModelRuntimeStatusTracker::new();
        t.mark_unhealthy("a", Some(503), "background_check_failed");
        t.mark_unhealthy("b", Some(503), "background_check_failed");
        let attempts = vec![am("a"), am("b")];
        match filter_attempt_models(&t, attempts, OnAllFilteredPolicy::Fail) {
            FilterOutcome::AllUnhealthy { retry_after_secs } => {
                assert_eq!(retry_after_secs, Some(30));
            }
            _ => panic!("expected AllUnhealthy"),
        }
    }

    #[test]
    fn one_cooldown_with_all_else_unhealthy_keeps_the_cooldown_candidate() {
        // Mixed scenario: candidates a/b are background-unhealthy, c
        // is in cooldown. The filter should pick c (cooldown beats
        // unhealthy), not fail.
        let t = crate::ModelRuntimeStatusTracker::new();
        t.mark_unhealthy("a", Some(503), "background_check_failed");
        t.mark_unhealthy("b", Some(503), "background_check_failed");
        t.mark_cooldown("c", StdDuration::from_secs(30), "x");
        let attempts = vec![am("a"), am("b"), am("c")];
        match filter_attempt_models(&t, attempts, OnAllFilteredPolicy::Fail) {
            FilterOutcome::Selected(list) => {
                assert_eq!(list.len(), 1);
                assert_eq!(list[0].id, "c");
            }
            _ => panic!("expected Selected with cooldown candidate"),
        }
    }

    #[test]
    fn all_unhealthy_original_order_policy_returns_full_list() {
        // Legacy opt-in: send to all candidates regardless.
        let t = crate::ModelRuntimeStatusTracker::new();
        t.mark_unhealthy("a", Some(503), "background_check_failed");
        t.mark_unhealthy("b", Some(503), "background_check_failed");
        let attempts = vec![am("a"), am("b")];
        match filter_attempt_models(&t, attempts, OnAllFilteredPolicy::OriginalOrder) {
            FilterOutcome::Selected(list) => {
                assert_eq!(list.len(), 2);
            }
            _ => panic!("expected Selected under OriginalOrder policy"),
        }
    }

    #[test]
    fn cooldown_no_unhealthy_returns_cooldown_candidates() {
        // No healthy, no unhealthy — all candidates have a cooldown
        // timer set. Routing should still pick from them (better than
        // erroring out when we don't have evidence anyone is *broken*).
        let t = crate::ModelRuntimeStatusTracker::new();
        t.mark_cooldown("a", StdDuration::from_secs(30), "x");
        t.mark_cooldown("b", StdDuration::from_secs(30), "x");
        let attempts = vec![am("a"), am("b")];
        match filter_attempt_models(&t, attempts, OnAllFilteredPolicy::Fail) {
            FilterOutcome::Selected(list) => {
                assert_eq!(list.len(), 2);
            }
            _ => panic!("expected Selected for cooldown-only"),
        }
    }
}

#[cfg(test)]
mod sanitize_tag_tests {
    use super::*;

    #[test]
    fn sanitize_tag_empty_stays_empty() {
        assert_eq!(sanitize_tag(String::new()), "");
    }

    #[test]
    fn sanitize_tag_strips_newlines_carriage_returns_and_nul() {
        // Injection attempt: a label that, if echoed verbatim into a
        // line-oriented log on the cp-api side, would forge an
        // "injected-internal-key: secret" line. Sanitiser must strip
        // all control chars including \r and \0.
        let evil = "production\ninjected-internal-key: secret\r\0".to_string();
        let safe = sanitize_tag(evil);
        assert!(!safe.contains('\n'));
        assert!(!safe.contains('\r'));
        assert!(!safe.contains('\0'));
        // Visible chars survive untouched.
        assert!(safe.starts_with("production"));
        assert!(safe.contains("injected-internal-key: secret"));
    }

    #[test]
    fn sanitize_tag_caps_length_at_256() {
        let huge = "a".repeat(10_000);
        let safe = sanitize_tag(huge);
        assert_eq!(safe.len(), 256);
    }

    #[test]
    fn sanitize_tag_preserves_normal_ascii_and_unicode() {
        // Real-world labels include hyphens, slashes, spaces, and
        // non-ASCII (operator might label a team in their language).
        let normal = "team-α / prod-east-1".to_string();
        assert_eq!(sanitize_tag(normal.clone()), normal);
    }
}

#[cfg(test)]
mod complete_on_drop_tests {
    //! Issue #419: streaming-abort cost-leak gate.
    //!
    //! The DP's streaming SSE generator (`build_sse_stream`) wraps an
    //! `on_complete` callback in a `CompleteOnDrop` guard so the
    //! callback fires reliably on BOTH normal stream completion AND
    //! mid-stream cancellation. Before the gate, the upstream's
    //! `usage` block populated `completion_tokens` regardless of
    //! whether any chunk actually reached the client — a customer
    //! who aborted mid-await was still billed for tokens the gateway
    //! never delivered.
    //!
    //! Delivery counting is driven by the `DeliveryCounter` stream
    //! wrapper: every `poll_next -> Ready(Some(_))` increments the
    //! shared `Arc<AtomicU32>`. `CompleteOnDrop::drop` reads that
    //! atomic and gates the completion-side counters on zero.
    //!
    //! These tests exercise the Drop logic directly — no real
    //! upstream stream, no axum, no httptest. Construct a
    //! StreamCompletion, set the shared atomic to simulate "N
    //! chunks delivered to the consumer", drop, observe the
    //! callback args.
    use super::{AtomicU32, CompleteOnDrop, StreamCompletion};
    use std::sync::{Arc, Mutex};

    /// Build the guard with `delivered_count` pre-set on the
    /// shared atomic, drop it, return whatever on_complete received.
    fn drop_and_capture(comp: StreamCompletion, delivered_count: u32) -> StreamCompletion {
        let captured: Arc<Mutex<Option<StreamCompletion>>> = Arc::new(Mutex::new(None));
        let cap = captured.clone();
        let delivered = Arc::new(AtomicU32::new(delivered_count));
        {
            let guard = CompleteOnDrop {
                slot: Some((
                    move |c: StreamCompletion| {
                        *cap.lock().unwrap() = Some(c);
                    },
                    comp,
                )),
                delivered,
            };
            drop(guard);
        }
        let out = captured.lock().unwrap().take().expect("on_complete fired");
        out
    }

    #[test]
    fn no_chunks_delivered_zeroes_completion_tokens() {
        // Simulated state: upstream emitted a usage block populating
        // completion_tokens=5, but the consumer aborted before any
        // chunk crossed the wire (DeliveryCounter atomic stayed 0).
        // Drop must zero completion-side counters so the customer
        // isn't billed for tokens that never reached them.
        let comp = StreamCompletion {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            reasoning_tokens: 2,
            cache_creation_tokens: 1,
            cache_read_tokens: 1,
            ..Default::default()
        };

        let out = drop_and_capture(comp, 0);

        assert_eq!(
            out.completion_tokens, 0,
            "completion_tokens must zero when delivered==0 (#419)"
        );
        assert_eq!(out.reasoning_tokens, 0, "reasoning_tokens must zero");
        assert_eq!(out.cache_creation_tokens, 0, "cache_creation must zero");
        assert_eq!(out.cache_read_tokens, 0, "cache_read must zero");
        assert_eq!(
            out.prompt_tokens, 10,
            "prompt_tokens preserved — prompt was processed regardless of delivery"
        );
        assert_eq!(
            out.total_tokens, 10,
            "total_tokens recomputed = prompt_tokens only when completion-side zeroed"
        );
        assert_eq!(out.chunks_delivered, 0, "Drop must persist delivered count");
    }

    #[test]
    fn one_chunk_delivered_preserves_completion_tokens() {
        // At least one chunk reached the consumer (could be a
        // role-only chunk + abort, or full stream + clean exit).
        // Either way, the upstream usage block is the source of
        // truth — pass through unchanged.
        let comp = StreamCompletion {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            reasoning_tokens: 2,
            ..Default::default()
        };

        let out = drop_and_capture(comp, 1);

        assert_eq!(
            out.completion_tokens, 5,
            "completion_tokens preserved when delivered>=1"
        );
        assert_eq!(out.reasoning_tokens, 2);
        assert_eq!(out.total_tokens, 15);
        assert_eq!(out.prompt_tokens, 10);
        assert_eq!(out.chunks_delivered, 1);
    }

    #[test]
    fn no_chunks_no_prompt_zero_total() {
        // Edge: connection dropped so early the upstream didn't even
        // confirm prompt_tokens. Drop produces a clean zeroed
        // completion-side AND prompt_tokens=0.
        let comp = StreamCompletion::default(); // all zeros

        let out = drop_and_capture(comp, 0);

        assert_eq!(out.completion_tokens, 0);
        assert_eq!(out.prompt_tokens, 0);
        assert_eq!(out.total_tokens, 0);
        assert_eq!(out.chunks_delivered, 0);
    }

    #[test]
    fn many_chunks_delivered_preserves_full_state() {
        // Normal stream completion: dozens of chunks pulled, full
        // upstream usage block received, Drop fires on natural exit
        // of the async_stream body. All fields pass through.
        let comp = StreamCompletion {
            prompt_tokens: 100,
            completion_tokens: 250,
            total_tokens: 350,
            cached_prompt_tokens: 30,
            reasoning_tokens: 50,
            ..Default::default()
        };

        let out = drop_and_capture(comp, 42);

        assert_eq!(out.prompt_tokens, 100);
        assert_eq!(out.completion_tokens, 250);
        assert_eq!(out.total_tokens, 350);
        assert_eq!(out.cached_prompt_tokens, 30);
        assert_eq!(out.reasoning_tokens, 50);
        assert_eq!(out.chunks_delivered, 42);
    }
}

#[cfg(test)]
mod delivery_counter_tests {
    //! Integration-style test for the `DeliveryCounter` wrapper +
    //! `CompleteOnDrop` collaboration (audit follow-up to the
    //! original #419 fix). The audit's HIGH-1 finding showed that
    //! the original post-yield approach under-counted by 1 on
    //! every abort path. This test would have caught that.
    //!
    //! Builds a real `Stream` via async_stream::stream!, pulls N
    //! items via `.next().await`, drops the stream, asserts the
    //! `chunks_delivered` value the on_complete callback received
    //! matches N exactly.
    use super::{AtomicU32, CompleteOnDrop, DeliveryCounter, StreamCompletion};
    use futures::Stream;
    use futures::StreamExt;
    use std::sync::{Arc, Mutex};

    /// Build a minimal stream that mirrors `build_sse_stream`'s
    /// shape: an async_stream yielding N units, wrapped in
    /// CompleteOnDrop + DeliveryCounter. The on_complete callback
    /// stores the received StreamCompletion in `captured`.
    fn build_test_stream(
        items: u32,
        captured: Arc<Mutex<Option<StreamCompletion>>>,
    ) -> impl Stream<Item = u32> {
        let delivered = Arc::new(AtomicU32::new(0));
        let delivered_for_drop = Arc::clone(&delivered);
        let inner = async_stream::stream! {
            let mut guard = CompleteOnDrop {
                slot: Some((
                    move |c: StreamCompletion| {
                        *captured.lock().unwrap() = Some(c);
                    },
                    StreamCompletion {
                        prompt_tokens: 10,
                        completion_tokens: 5,
                        total_tokens: 15,
                        ..Default::default()
                    },
                )),
                delivered: delivered_for_drop,
            };
            // Silence unused-field warning on the test side; guard
            // is held by the body for its full lifetime.
            let _ = &mut guard;
            for i in 0..items {
                yield i;
            }
        };
        DeliveryCounter {
            inner: Box::pin(inner),
            delivered,
        }
    }

    // NOTE: there is no "zero polls" test case here — `async_stream`
    // only runs its body on the first poll, so creating a stream
    // without ever polling means the inner guard is never even
    // constructed, and on_complete cannot fire. That's not the
    // production failure mode (#419) anyway: axum's Sse driver always
    // polls at least once. The zero-delivered case is covered
    // synthetically in `complete_on_drop_tests` where the Drop logic
    // is exercised directly.

    #[tokio::test]
    async fn pulling_one_chunk_then_drop_yields_one_delivered() {
        // This is the canary case the audit flagged: under the old
        // post-yield approach, this would have come back as 0
        // (off-by-one). With DeliveryCounter at poll_next, the count
        // is 1 — gate skipped, completion_tokens preserved.
        let captured = Arc::new(Mutex::new(None));
        {
            let stream = build_test_stream(5, captured.clone());
            futures::pin_mut!(stream);
            let v = stream.next().await;
            assert_eq!(v, Some(0));
            // Consumer aborts here. Drop fires.
        }
        let out = captured.lock().unwrap().take().expect("on_complete fired");
        assert_eq!(
            out.chunks_delivered, 1,
            "delivery counter must reflect the 1 pull"
        );
        assert_eq!(
            out.completion_tokens, 5,
            "completion_tokens preserved when delivered>=1"
        );
    }

    #[tokio::test]
    async fn pulling_three_then_drop_yields_three_delivered() {
        let captured = Arc::new(Mutex::new(None));
        {
            let stream = build_test_stream(5, captured.clone());
            futures::pin_mut!(stream);
            assert_eq!(stream.next().await, Some(0));
            assert_eq!(stream.next().await, Some(1));
            assert_eq!(stream.next().await, Some(2));
            // 3 pulls; abort.
        }
        let out = captured.lock().unwrap().take().expect("on_complete fired");
        assert_eq!(out.chunks_delivered, 3);
    }

    #[tokio::test]
    async fn pulling_full_stream_yields_count_equal_to_items() {
        let captured = Arc::new(Mutex::new(None));
        {
            let stream = build_test_stream(4, captured.clone());
            futures::pin_mut!(stream);
            let collected: Vec<u32> = stream.by_ref().collect().await;
            assert_eq!(collected, vec![0, 1, 2, 3]);
            // Stream is exhausted; consumer drops naturally.
        }
        let out = captured.lock().unwrap().take().expect("on_complete fired");
        assert_eq!(
            out.chunks_delivered, 4,
            "exact match — DeliveryCounter fires on every Ready(Some)"
        );
    }
}
