//! `POST /v1/messages/count_tokens` — Anthropic token-counting passthrough.
//!
//! The Anthropic SDK exposes this as `anthropic.messages.countTokens(...)`,
//! the documented endpoint customers use to size a prompt (messages +
//! system + tools + images) before issuing a paid `/v1/messages` call.
//! Claude Code and most Anthropic-SDK apps call it, so a gateway that
//! omits the route forces callers to over-provision or bypass it (#418).
//!
//! This is the sibling sub-route of `/v1/messages`: same model-alias
//! resolution, same `x-api-key` + `anthropic-version` auth shape, same
//! Anthropic-shape error envelope (#336). The only differences from the
//! `/v1/messages` Anthropic passthrough are the upstream suffix
//! (`/messages/count_tokens`), the absence of streaming, and the tiny
//! `{"input_tokens": <int>}` response, which is forwarded verbatim.
//!
//! Guardrails: the **input** hook runs here, exactly as on `/v1/messages`;
//! the **output** hook does not (#555, revising #545).
//!
//! The two halves are asymmetric because the endpoint is. The response is
//! `{"input_tokens": <int>}` — the provider generated nothing, so an output
//! guardrail has no content to moderate and running one would be theatre.
//! The REQUEST is a different matter: this route ships the caller's entire
//! `system` + `messages` + `tools` payload to the provider, which is
//! precisely the transmission a PII / DLP / data-exfiltration guardrail
//! exists to govern. The original exemption argued the same payload gets
//! scanned when the caller issues the real `/v1/messages` call — but nothing
//! obliges a caller to ever issue it. `count_tokens` on its own is a
//! complete egress channel, and an operator's input policy was silently not
//! applied to it.
//!
//! Mask-action rules rewrite the body here too, before it is forwarded. That
//! also keeps the answer honest: `/v1/messages` masks the same spans, so the
//! count now describes the body the gateway would really send.
//!
//! Telemetry: this route is NOT metered and still emits a terminal
//! `UsageEvent` on every outcome, with `prompt_tokens`/`completion_tokens`
//! at zero. Those are two different questions, and the route answered only
//! the first: it generates nothing, so there is nothing to bill — but it
//! does forward the caller's whole payload to a real upstream, so every
//! question Logs exists to answer (did this request happen, which key sent
//! it, how long did it take, did a guardrail refuse it) had no row to read.
//! A refusal was the sharp end: `/v1/messages/count_tokens` can be blocked
//! by an input guardrail, and a refusal that emits no event is a 422 the
//! caller definitely saw and the "Guardrail blocks" view cannot find
//! (AISIX-Cloud#1435, the same failure mode as AISIX-Cloud#1428).
//! `guardrail_coverage`'s census asserts the reporting half over the
//! surfaces it reads out of the router, so this cannot regress quietly.
//!
//! Scope: Anthropic-backed models only. `count_tokens` has no upstream
//! equivalent for OpenAI/Gemini/DeepSeek, so a non-Anthropic Model is
//! rejected with a 400 at the gateway boundary (parallel to `/v1/rerank`
//! §168 / `/v1/responses` §4.6) rather than dispatched to an upstream
//! that would 404 — and rather than the gateway emitting a misleading
//! 404 of its own, which was the bug this route closes.
//!
//! Reference:
//! - Anthropic Count Message Tokens API:
//!   <https://platform.claude.com/docs/en/api/messages-count-tokens>
//!   (`POST /v1/messages/count_tokens` → `{"input_tokens": <int>}`).
//! - Other OpenAI-compatible gateways expose the same route as a
//!   user-facing passthrough and hit the identical "route missing from
//!   the list" bug.

use aisix_obs::AccessLog;
use axum::extract::rejection::JsonRejection;
use axum::extract::State;
use axum::http::{HeaderName, HeaderValue};
use axum::response::Response;
use axum::Json;
use serde_json::Value;
use std::time::{Duration, Instant};

use crate::auth::AuthenticatedKey;
use crate::client_ip::ClientContext;
use crate::error::ProxyError;
use crate::messages::ANTHROPIC_VERSION;
use crate::state::ProxyState;

pub async fn count_tokens(
    State(state): State<ProxyState>,
    auth: Result<AuthenticatedKey, ProxyError>,
    client: ClientContext,
    body: Result<Json<Value>, JsonRejection>,
) -> Response {
    // Auth / body-extractor rejections must render the Anthropic-shape
    // envelope so the Claude SDK's parser recognises them (#336) — same
    // policy as /v1/messages. The shared helper keeps the body-rejection
    // discrimination (malformed JSON vs 413 cap vs transport error) in
    // lockstep with the sibling route.
    let auth = match auth {
        Ok(a) => a,
        Err(e) => return e.into_anthropic_response(),
    };
    let started = Instant::now();
    let Json(mut body) = match body {
        Ok(j) => j,
        // Answer through `reject` — see messages.rs.
        Err(rej) => {
            return crate::reject::reject_before_dispatch(
                &state,
                "POST",
                "/v1/messages/count_tokens",
                &client.request_id,
                Some(&auth.entry.id),
                started,
                crate::reject::Envelope::Anthropic,
                crate::error::proxy_error_from_json_rejection(rej, state.request_body_limit_bytes),
            );
        }
    };

    let request_id = client.request_id.clone();
    let api_key_id = auth.entry.id.clone();

    let model_name = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    // One snapshot for the whole request (#941) — see `embeddings`.
    let snapshot = state.snapshot.load();

    // Filled inside `dispatch`, so the failure branch — where a guardrail
    // block lands — stamps the enforced hits too (AISIX-Cloud#1330 / #1024).
    let mut screening = InputScreening::default();
    match dispatch(
        &state,
        &snapshot,
        &auth,
        &mut body,
        &request_id,
        &client,
        &mut screening,
    )
    .await
    {
        Ok(success) => {
            let elapsed = started.elapsed();
            let status = success.response.status().as_u16();
            emit_access_log(
                &model_name,
                &success.provider,
                &api_key_id,
                status,
                elapsed,
                &request_id,
                None,
            );
            // One ProviderKey lookup per completion (#941).
            let pk = crate::usage_attr::ResolvedPk::resolve(&snapshot, &success.provider_key_id);
            crate::request_metrics::record(
                &state,
                "/v1/messages/count_tokens",
                crate::request_metrics::Caller::new(&auth),
                crate::request_metrics::Upstream {
                    provider: &success.provider,
                    model: &model_name,
                    upstream_model: &success.upstream_model,
                    pk: pk.labels(),
                    ..Default::default()
                },
                status,
                elapsed,
            );
            emit_usage_event(
                &state,
                &snapshot,
                &pk,
                &request_id,
                &success.model_id,
                &model_name,
                &api_key_id,
                status,
                success.upstream_elapsed,
                elapsed,
                &client,
                &screening,
            );
            success.response
        }
        Err(err) => {
            let status = err.status().as_u16();
            let elapsed = started.elapsed();
            emit_access_log(
                &model_name,
                "unknown",
                &api_key_id,
                status,
                elapsed,
                &request_id,
                Some(&err),
            );
            let metric_model = crate::usage_attr::metric_model_label(&snapshot, &model_name);
            // AISIX-Cloud#1325: name the target the request died on. This
            // branch used to emit `Upstream::default()`, so a 502 from a
            // real provider landed on `provider="unknown"` while the same
            // key's successes landed on the real one.
            let attributed = crate::attribution::current().unwrap_or_default();
            let last_target = crate::request_metrics::LastTarget::new(&snapshot, &attributed);
            crate::request_metrics::record(
                &state,
                "/v1/messages/count_tokens",
                crate::request_metrics::Caller::new(&auth),
                last_target.upstream(metric_model.as_ref(), false, false),
                status,
                elapsed,
            );
            // A failed count_tokens is a request the operator has to be
            // able to find, and a guardrail refusal is the one that must
            // carry the flag the "Guardrail blocks" view filters on.
            crate::usage_attr::emit_error_usage_event(
                &state,
                &snapshot,
                crate::operation::COUNT_TOKENS,
                "anthropic",
                &request_id,
                &model_name,
                &api_key_id,
                status,
                err.kind(),
                err.is_guardrail_block(),
                &client,
                crate::usage_attr::enforced_hits(&screening.audit),
                crate::usage_attr::guardrail_scores(&screening.audit),
            );
            // Anthropic-shape envelope (#336) — count_tokens callers are
            // the Anthropic SDK, not OpenAI-compatible clients.
            err.into_anthropic_response()
        }
    }
}

/// What the input hook produced, for the terminal `UsageEvent` to carry.
///
/// An out-param rather than part of [`CountTokensSuccess`] because the
/// failure branch needs it too — a guardrail refusal IS the error, so the
/// error event is the one that must not drop the audit.
#[derive(Default)]
struct InputScreening {
    /// The request's ENFORCE-mode audit handle (AISIX-Cloud#1330).
    audit: crate::usage_attr::GuardrailAudit,
    /// The `{kind, hook}` set of guardrails that governed the request
    /// (#379 parity) — surfaced on the event so Logs can show them.
    applied: Vec<aisix_core::AppliedGuardrail>,
    /// Monitor-mode observations (AISIX-Cloud#562).
    monitor_hits: Vec<aisix_core::GuardrailMonitorHit>,
    /// Per-detector PII mask counts (#932/#696). Empty = no redaction.
    redactions: crate::redact::RedactionCounts,
}

/// Run the resolved input guardrail chain over the Anthropic-shaped body,
/// blocking before dispatch and writing mask-action rewrites back into
/// `body` (which is what `count_tokens_to_target` forwards upstream).
///
/// Deliberately mirrors `messages::dispatch_inner`'s block rather than
/// sharing a helper with it: that one threads the same telemetry through a
/// retrying per-attempt emitter, and this route has a single terminal
/// event. Keeping the shapes parallel is what the `guardrail_coverage`
/// census asserts.
async fn screen_input(
    state: &ProxyState,
    auth: &AuthenticatedKey,
    model_entry_id: &str,
    model_name: &str,
    body: &mut Value,
    screening: &mut InputScreening,
) -> Result<(), ProxyError> {
    let chain = state
        .guardrail_index
        .resolve(&aisix_guardrails::RequestContext {
            passthrough_route_id: "",
            model_id: model_entry_id,
            mcp_server_id: "",
            api_key_id: &auth.entry.id,
            team_id: auth.key().team_id.as_deref(),
        });
    screening.applied = chain.applied().to_vec();
    screening.audit = chain.audit_log();
    if chain.is_empty() {
        return Ok(());
    }
    // Fail closed on a body the scanner cannot read — see the same arm in
    // `messages.rs`.
    let chat = match aisix_provider_anthropic::parse_inbound_request_for_scan(body) {
        Ok(chat) => chat,
        Err(err) => {
            tracing::warn!(
                guardrail_hook = "input",
                model = %model_name,
                error = %err,
                "cannot scan /v1/messages/count_tokens body for guardrails; blocking",
            );
            return Err(crate::error::guardrail_block_error(
                "request",
                None,
                Some(crate::error::TAG_UNSCANNABLE_BODY),
            ));
        }
    };
    let (verdict, monitor_hits) =
        aisix_guardrails::Guardrail::check_input_non_segment_observed(&chain, &chat).await;
    screening.monitor_hits = monitor_hits;
    // Same scan-only submission as `/v1/messages` — the two routes screen
    // the same body with the same chain and must reach the same verdict.
    let signed_reasoning = crate::redact::anthropic_signed_reasoning_texts(body);
    let verdict = crate::redact::moderate_body_scanning(
        &chain,
        crate::redact::Direction::Input,
        verdict,
        &mut screening.redactions,
        // The segment pass's monitor-mode observations belong on the same
        // event as the non-segment ones above. A throwaway `Vec` here made
        // this route report fewer monitor hits than `/v1/messages` for an
        // identical body and chain — and the scan-only channel feeds this
        // pass more text, so the gap would have widened.
        &mut screening.monitor_hits,
        signed_reasoning,
        |g| crate::redact::redact_anthropic_request(g, body),
    )
    .await;
    if let aisix_guardrails::GuardrailVerdict::Block {
        reason,
        guardrail_name,
        unavailable,
    } = verdict
    {
        tracing::warn!(
            guardrail_hook = "input",
            model = %model_name,
            reason = %reason,
            "guardrail blocked /v1/messages/count_tokens request",
        );
        return Err(crate::error::guardrail_block_error(
            "request",
            guardrail_name.as_deref(),
            unavailable.as_deref(),
        ));
    }
    // Mask-action rules rewrite the body that is about to be forwarded.
    // Merged, not discarded: `/v1/messages` merges the same pass into the
    // counts its event reports (#932), and the two routes screen the same
    // body with the same chain — a mask this side under-reported would
    // read as the sibling route masking more of the same payload.
    crate::redact::merge_counts(
        &mut screening.redactions,
        crate::redact::redact_anthropic_request(&chain, body),
    );
    Ok(())
}

/// What the winning attempt resolved, for the request-metric label set —
/// which has to match what chat / messages / responses report
/// (AISIX-Cloud#1234) — and for the terminal `UsageEvent`.
struct CountTokensSuccess {
    response: Response,
    provider: String,
    upstream_model: String,
    provider_key_id: String,
    /// The DISPATCHED target's Model row id: a group resolves to one of
    /// its members, and `UsageEvent::model_id` records that target while
    /// `requested_model` keeps the alias the caller addressed.
    model_id: String,
    /// How long the WINNING attempt took. Not the handler's own elapsed:
    /// this route fails over across a group's Anthropic targets and
    /// retries within one, so on a group the two diverge by every attempt
    /// that lost — and `upstream_latency_ms` is attempt-scoped everywhere
    /// else in Logs (`downstream_latency_ms` is the request-scoped one).
    upstream_elapsed: Duration,
}

#[allow(clippy::too_many_arguments)]
async fn dispatch(
    state: &ProxyState,
    snapshot: &aisix_core::AisixSnapshot,
    auth: &AuthenticatedKey,
    body: &mut Value,
    request_id: &str,
    client: &ClientContext,
    screening: &mut InputScreening,
) -> Result<CountTokensSuccess, ProxyError> {
    let model_name = body
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ProxyError::InvalidRequest("`model` field missing".into()))?
        .to_string();

    let model_entry = crate::model_resolve::resolve_model(snapshot, &model_name)
        .ok_or_else(|| ProxyError::ModelNotFound(model_name.clone()))?;

    if !auth.key().can_access(&model_name) {
        return Err(ProxyError::ModelForbidden(model_name.clone()));
    }

    // Client-IP allowlist gate (#557): reject before quota / upstream.
    crate::dispatch::check_ip_access(&model_entry.value, &client.source_ip)?;

    // Input guardrails (#555). Same chain, same order and same ordering
    // rationale as the `/v1/messages` sibling: before the reservation, so a
    // content-policy refusal doesn't burn an RPM slot. See the module doc
    // for why the input hook applies here and the output hook does not.
    screen_input(state, auth, &model_entry.id, &model_name, body, screening).await?;

    let model_rl =
        crate::quota::ModelRateLimit::from_model(&model_name, &model_entry.id, &model_entry.value);
    let _reservation = crate::quota::enforce(state, snapshot, auth, Some(&model_rl)).await?;

    // Resolve the attempt list (routing-aware). count_tokens is
    // Anthropic-only, so we attempt the group's Anthropic targets in
    // order; a direct model resolves to itself (#471).
    let attempt_models = crate::routing::resolve_attempt_models(
        &state.routing,
        &state.runtime_status,
        snapshot,
        &model_name,
        &model_entry.id,
        &model_entry.value,
        crate::routing::RoutingRequest {
            tags: &client.routing_tags,
            headers: Some(&client.headers),
            api_key_id: auth.entry.id.as_str(),
            source_ip: &client.source_ip,
        },
    )?;
    let retry_on_429 = model_entry
        .value
        .routing
        .as_ref()
        .map(|r| r.retry_on_429_or_default())
        .unwrap_or(false);
    let fallback_statuses: &[u16] = model_entry
        .value
        .routing
        .as_ref()
        .map(|r| r.fallback_on_statuses_or_default())
        .unwrap_or(&[]);

    // NOTE: deliberately narrower than chat's `routing.is_some() ||
    // is_semantic()`. The quota gate defers model-property policies on any
    // routing/ensemble/semantic PARENT (`ModelRateLimit::routing_parent`),
    // expecting the per-target pass to reserve them — which only runs when
    // this flag is true. Safe today because semantic/ensemble parents
    // cannot successfully dispatch on this endpoint (no provider →
    // pre-dispatch 4xx); if this endpoint ever grows semantic support,
    // widen this flag or the deferred policies are silently skipped.
    let is_routing_request = model_entry.value.routing.is_some();
    let mut last_err: Option<ProxyError> = None;
    let mut any_anthropic = false;
    for (target_idx, target) in attempt_models.iter().enumerate() {
        // count_tokens has no upstream equivalent outside the Anthropic
        // protocol; skip foreign targets in a mixed group rather than
        // dispatching to an upstream that would 404.
        if !crate::dispatch::speaks_anthropic(snapshot, &target.model) {
            continue;
        }
        any_anthropic = true;
        // Reserve THIS target's own model rate-limit layers before
        // dispatching to it (AISIX-Cloud#1087); over-limit → skip it and
        // try the remaining targets. Like the handler-level `_reservation` it is
        // never token-committed — count_tokens burns no generation tokens;
        // the drop at scope end releases the concurrency slot.
        let _member_reservation = match crate::quota::reserve_routing_target(
            state,
            snapshot,
            auth,
            is_routing_request.then_some(crate::quota::RoutingParent {
                name: &model_entry.value.display_name,
                entry_id: &model_entry.id,
            }),
            &target.model.display_name,
            &target.id,
            &target.model,
        )
        .await
        {
            Ok(r) => r,
            Err(e) => {
                last_err = Some(e);
                continue;
            }
        };
        // Same-target retries before failing over, like the other
        // group-capable endpoints. This loop had fail-over only: a
        // transient 502 on the sole Anthropic target failed the request
        // outright even with a retry budget configured.
        //
        // "Another target queued" counts only Anthropic targets: the loop
        // `continue`s past everything else, so in a mixed group like
        // [anthropic, openai] the openai entry is not a real fallback —
        // treating it as one would suppress the default budget on the only
        // target that can actually serve the request.
        let has_usable_fallback = attempt_models[target_idx + 1..]
            .iter()
            .any(|t| crate::dispatch::speaks_anthropic(snapshot, &t.model));
        let budget = crate::routing::effective_retries(
            &target.model,
            crate::routing::group_retries_of(&model_entry.value),
            state.default_retries,
            has_usable_fallback,
        );
        for attempt_idx in 0..=budget.attempts {
            if attempt_idx > 0 {
                let hint = last_err.as_ref().and_then(|e| match e {
                    ProxyError::Bridge(be) => crate::routing::retry_after_hint(be),
                    _ => None,
                });
                tokio::time::sleep(crate::routing::retry_backoff(attempt_idx as u32, hint)).await;
            }
            match count_tokens_to_target(
                state,
                snapshot,
                body,
                &target.model,
                &target.id,
                crate::routing::effective_timeouts(
                    &target.model,
                    Some(&model_entry.value),
                    state.default_timeouts,
                ),
                request_id,
                client,
            )
            .await
            {
                Ok(success) => return Ok(success),
                Err(e) => {
                    let retryable = matches!(
                        &e,
                        ProxyError::Bridge(be) if crate::routing::is_retryable(be, retry_on_429, fallback_statuses)
                    );
                    // See `RetryBudget::covers`: a default budget skips
                    // same-target retries for timeouts; fail-over is
                    // unaffected (the outer loop still moves on).
                    let budget_covers = match &e {
                        ProxyError::Bridge(be) => budget.covers(be),
                        _ => true,
                    };
                    last_err = Some(e);
                    if !retryable {
                        return Err(last_err.unwrap_or(ProxyError::ProviderUnavailable));
                    }
                    if !budget_covers {
                        break;
                    }
                }
            }
        }
    }

    // No Anthropic target to serve count_tokens. Reject at the boundary
    // with a 400 (parallel to /v1/rerank's provider gate) rather than
    // dispatching to an upstream that would 404.
    if !any_anthropic {
        return Err(ProxyError::InvalidRequest(format!(
            "model `{model_name}` is not backed by an Anthropic-protocol upstream; \
             /v1/messages/count_tokens requires a provider key that either uses \
             the anthropic adapter or declares `apis.messages`"
        )));
    }
    Err(last_err.unwrap_or(ProxyError::ProviderUnavailable))
}

/// Dispatch one concrete target's count_tokens passthrough. The route is
/// a sub-route of `/v1/messages` and rides the same declaration, so it
/// resolves against that surface's own base when the Provider Key names
/// one and against `api_base` otherwise. The caller has already confirmed
/// the target speaks the Anthropic protocol (`dispatch::speaks_anthropic`).
#[allow(clippy::too_many_arguments)]
async fn count_tokens_to_target(
    state: &ProxyState,
    snapshot: &aisix_core::AisixSnapshot,
    body: &Value,
    model: &aisix_core::Model,
    model_id: &str,
    // Deadlines resolved by the caller across target → group → deployment
    // default (`routing::effective_timeouts`); this fn only applies them.
    timeouts: crate::routing::TimeoutBudget,
    request_id: &str,
    client: &ClientContext,
) -> Result<CountTokensSuccess, ProxyError> {
    let attempt_started = Instant::now();
    let mut body = body.clone();
    let pk_entry = crate::dispatch::resolve_provider_key(snapshot, model)?;
    let api_key = crate::dispatch::require_api_key(&pk_entry.value, model)?;
    let upstream_model = crate::dispatch::require_upstream_model(model)?.to_string();

    // Rewrite the `model` field to the upstream value, exactly as the
    // /v1/messages passthrough does — the caller speaks the gateway's
    // display name; the upstream expects its own id.
    if let Some(m) = body.get_mut("model") {
        *m = Value::String(upstream_model.clone());
    }

    // Apply the PK's `request.*` override block to the outbound body,
    // identically to the /v1/messages passthrough — count_tokens shares
    // the same Anthropic ProviderKey, so operator-configured renames /
    // constraints / defaults must reach this sibling route too. Apply
    // order matches §5: renames → constraints → defaults; each is a
    // no-op when its configured map is empty.
    if let Some(r) = pk_entry.value.request.as_ref() {
        aisix_provider_openai::overrides::apply_param_renames(&mut body, &r.param_renames);
        if let Some(constraints) = &r.param_constraints {
            aisix_provider_openai::overrides::apply_param_constraints(&mut body, constraints);
        }
        aisix_provider_openai::overrides::apply_default_body_fields(
            &mut body,
            &r.default_body_fields,
        );
    }

    // `build_anthropic_url` tolerates an api_base with or without `/v1` (the
    // Anthropic dashboard placeholder and copy-pasted full URLs both
    // resolve to `…/v1/messages/count_tokens`).
    let url = aisix_gateway::url_cache::cached_endpoint_url(
        &pk_entry.id,
        "proxy/messages/count_tokens",
        // Every resolve_base_url_for input (#1017) via the shared constructor.
        &crate::dispatch::pk_surface_url_fingerprint(
            &pk_entry.value,
            aisix_core::ApiSurface::Messages,
        ),
        || {
            let base = crate::dispatch::resolve_base_url_for(
                &pk_entry.value,
                aisix_core::ApiSurface::Messages,
            )?;
            Ok::<_, crate::error::ProxyError>(crate::dispatch::build_anthropic_url(
                &base,
                "/messages/count_tokens",
            ))
        },
    )?;

    // Build the outbound HeaderMap explicitly so the PK's
    // `request.default_headers` / `request.forward_client_headers` can
    // inject operator-supplied and allowlisted client headers (e.g.
    // `anthropic-beta`) via the shared apply pipeline. The bridge-owned
    // headers (x-api-key, anthropic-version, content-type,
    // x-aisix-request-id) are inserted FIRST; `apply_request_headers`
    // skips keys already present + the reserved auth-header blacklist
    // (`x-api-key`), so neither source can clobber auth here
    // (ai-gateway#337). Anthropic auth shape:
    // `x-api-key` + `anthropic-version`, NOT `Authorization: Bearer`.
    let mut headers = axum::http::HeaderMap::new();
    let api_key_hv = HeaderValue::from_str(api_key).map_err(|e| {
        ProxyError::Bridge(aisix_gateway::BridgeError::Config(format!(
            "api key contains invalid header chars: {e}"
        )))
    })?;
    headers.insert(HeaderName::from_static("x-api-key"), api_key_hv);
    headers.insert(
        HeaderName::from_static("anthropic-version"),
        HeaderValue::from_static(ANTHROPIC_VERSION),
    );
    headers.insert(
        axum::http::header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    let rid_hv = HeaderValue::from_str(request_id).map_err(|e| {
        ProxyError::Bridge(aisix_gateway::BridgeError::Config(format!(
            "request_id contains invalid header chars: {e}"
        )))
    })?;
    headers.insert(HeaderName::from_static("x-aisix-request-id"), rid_hv);
    aisix_gateway::apply_request_headers(
        &mut headers,
        &crate::dispatch::upstream_header_ctx(
            &pk_entry.value,
            &pk_entry.id,
            model,
            model_id,
            client,
        ),
    );

    let client = crate::http_client::client_for(pk_entry.value.tls.as_ref());
    let mut req = url.post_on(&client).headers(headers).json(&body);
    // #554: count_tokens is non-streaming; apply the E2E request timeout.
    if let Some(d) = timeouts.request {
        req = req.timeout(d);
    }
    let send_started = Instant::now();
    let upstream_resp = req
        .send()
        .await
        .map_err(|e| {
            crate::cooldown::note_failure(
                &state.runtime_status,
                model_id,
                model.cooldown.as_ref(),
                crate::dispatch::reqwest_error_to_bridge(&e, send_started),
            )
        })
        .map_err(ProxyError::Bridge)?;

    let status = upstream_resp.status();

    if !status.is_success() {
        let status_u16 = status.as_u16();
        let retry_after = aisix_gateway::parse_retry_after(upstream_resp.headers());
        let message = upstream_resp.text().await.unwrap_or_default();
        let truncated = crate::util::truncate_on_char_boundary(&message, 1024);
        let err = aisix_gateway::BridgeError::upstream_status_with_retry_after(
            status_u16,
            truncated,
            retry_after,
        );
        if let Some((ttl, reason)) = crate::cooldown::decide_cooldown(&err, model.cooldown.as_ref())
        {
            state.runtime_status.mark_cooldown(model_id, ttl, reason);
        }
        return Err(ProxyError::Bridge(err));
    }

    state.health.record_success(&model.display_name);
    state.runtime_status.mark_healthy(model_id);

    // Forward the `{"input_tokens": <int>}` response body verbatim — the
    // gateway adds nothing to the token-counting contract.
    let upstream_headers = upstream_resp.headers().clone();
    let body_bytes = upstream_resp
        .bytes()
        .await
        .map_err(|e| {
            crate::cooldown::note_failure(
                &state.runtime_status,
                model_id,
                model.cooldown.as_ref(),
                aisix_gateway::BridgeError::UpstreamDecode(e.to_string()),
            )
        })
        .map_err(ProxyError::Bridge)?;

    let mut resp = axum::response::Response::new(axum::body::Body::from(body_bytes));
    if let Some(ct) = upstream_headers.get("content-type") {
        if let Ok(hv) = HeaderValue::from_bytes(ct.as_bytes()) {
            resp.headers_mut()
                .insert(axum::http::header::CONTENT_TYPE, hv);
        }
    }
    // Only emit the request-id header when it parses — matching the
    // /v1/messages handler. An empty fallback value would hurt log
    // correlation more than an absent header.
    if let Ok(hv) = HeaderValue::from_str(request_id) {
        resp.headers_mut()
            .insert(HeaderName::from_static("x-aisix-request-id"), hv);
    }

    Ok(CountTokensSuccess {
        response: resp,
        // The target model's own vendor id, as every other endpoint
        // labels it. This used to read "anthropic" on the grounds that
        // the loop only dispatches Anthropic targets — true of the wire,
        // never of the vendor: a `byo` + `adapter: anthropic` key already
        // reported `byo` on /v1/chat/completions, and a Provider Key that
        // declares `apis.messages` brings any vendor down this path.
        provider: model
            .provider
            .as_deref()
            .unwrap_or("unknown")
            .to_ascii_lowercase(),
        upstream_model,
        provider_key_id: pk_entry.id.to_string(),
        model_id: model_id.to_string(),
        upstream_elapsed: attempt_started.elapsed(),
    })
}

/// The terminal `UsageEvent` for a served count_tokens.
///
/// Token counters stay at zero, deliberately: the `{"input_tokens": N}` the
/// caller gets back is a MEASUREMENT of a prompt, not tokens any upstream
/// consumed or billed. Copying it into `prompt_tokens` would put spend on
/// a request that cost nothing and double-count the prompt once the caller
/// goes on to issue the real `/v1/messages` call.
///
/// No `request_metrics::record_usage` call for the same reason — the
/// `aisix_llm_*_tokens_total` families are token/spend families, and this
/// route contributes neither. The request families already carry the call
/// (`request_metrics::record`, above), and `aisix_usage_events_emitted_total`
/// counts this event under `handler="count_tokens"`.
#[allow(clippy::too_many_arguments)]
fn emit_usage_event(
    state: &ProxyState,
    snap: &aisix_core::AisixSnapshot,
    pk: &crate::usage_attr::ResolvedPk<'_>,
    request_id: &str,
    model_id: &str,
    requested_model: &str,
    api_key_id: &str,
    status_code: u16,
    // Attempt-scoped, from the winning attempt; see `CountTokensSuccess`.
    upstream_elapsed: Duration,
    // Request-scoped: what the caller actually waited for, guardrails and
    // any lost attempts included.
    elapsed: Duration,
    client: &ClientContext,
    screening: &InputScreening,
) {
    let mut event = aisix_obs::UsageEvent {
        request_id: request_id.to_string(),
        occurred_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        model_id: model_id.to_string(),
        api_key_id: api_key_id.to_string(),
        requested_model: requested_model.to_string(),
        upstream_latency_ms: upstream_elapsed.as_millis().min(u32::MAX as u128) as u32,
        downstream_latency_ms: elapsed.as_millis().min(u32::MAX as u128) as u32,
        status_code,
        inbound_protocol: "anthropic".to_string(),
        applied_guardrails: screening.applied.clone(),
        client_source_ip: client.source_ip.clone(),
        client_user_agent: client.user_agent.clone(),
        redacted_entity_counts: screening.redactions.clone(),
        guardrail_monitor_hits: screening.monitor_hits.clone(),
        guardrail_enforced_hits: crate::usage_attr::enforced_hits(&screening.audit),
        guardrail_scores: crate::usage_attr::guardrail_scores(&screening.audit),
        ..Default::default()
    };
    crate::usage_attr::apply_pk_telemetry(&mut event, pk);
    crate::usage_attr::apply_caller_identity(
        &mut event,
        client.jwt.as_ref(),
        client.caller.user_id.as_deref(),
        client.caller.user_name.as_deref(),
    );
    let usage_model =
        crate::usage_attr::usage_event_model_label(snap, &event.requested_model).into_owned();
    crate::usage_attr::emit_usage(
        state,
        snap,
        crate::operation::COUNT_TOKENS,
        event,
        crate::usage_attr::usage_event_labels(&usage_model, pk),
        // Content capture (#700) is not wired on this route — it is a
        // separate, per-exporter opt-in capability, and #1435 is about the
        // event existing at all.
        None,
        client.trace.as_ref(),
        /* terminal */ true,
        /* dispatched */ true,
    );
}

fn emit_access_log(
    model: &str,
    provider: &str,
    api_key_id: &str,
    status: u16,
    elapsed: Duration,
    request_id: &str,
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
        method: "POST",
        path: "/v1/messages/count_tokens",
        status,
        latency: elapsed,
        provider: Some(provider),
        model: Some(model),
        api_key_id: Some(api_key_id),
        prompt_tokens: None,
        completion_tokens: None,
        total_tokens: None,
        request_id,
        // No provider response id: count_tokens returns only the token
        // estimate, and no upstream response object (AISIX-Cloud#1289).
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
    use aisix_core::resource::ResourceEntry;
    use aisix_core::snapshot::SnapshotHandle;
    use aisix_core::{AisixSnapshot, ApiKey, Model, ProxyConfig};
    use aisix_gateway::Hub;
    use axum::body::to_bytes;
    use axum::http::{Request, StatusCode};
    use std::sync::Arc;
    use tower::ServiceExt;
    use wiremock::matchers::{header, method, path};
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

    fn anthropic_model(name: &str) -> ResourceEntry<Model> {
        let json = format!(
            r#"{{"display_name":"{name}","provider":"anthropic","model_name":"claude-haiku-4-5-20251001","provider_key_id":"{PK_ID}"}}"#
        );
        let m: Model = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("m-1", m, 1)
    }

    fn openai_model(name: &str) -> ResourceEntry<Model> {
        let json = format!(
            r#"{{"display_name":"{name}","provider":"openai","model_name":"gpt-4o","provider_key_id":"{PK_ID}"}}"#
        );
        let m: Model = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("m-1", m, 1)
    }

    fn anthropic_pk(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        let json = format!(
            r#"{{"display_name":"anthropic-up","secret":"sk-ant-test","api_base":"{api_base}","provider":"anthropic","adapter":"anthropic"}}"#
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&json).unwrap();
        ResourceEntry::new(PK_ID, pk, 1)
    }

    fn new_snap(api_base: &str) -> AisixSnapshot {
        let snap = AisixSnapshot::new();
        snap.provider_keys.insert(anthropic_pk(api_base));
        snap
    }

    fn apikey_entry(allowed: &[&str]) -> ResourceEntry<ApiKey> {
        // SHA-256 of "sk-caller".
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

    fn make_req(body: serde_json::Value) -> Request<axum::body::Body> {
        // Anthropic SDK auth shape: x-api-key + anthropic-version.
        Request::builder()
            .method("POST")
            .uri("/v1/messages/count_tokens")
            .header("x-api-key", "sk-caller")
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body.to_string()))
            .unwrap()
    }

    /// AISIX-Cloud#1435: the served request leaves a row, and that row
    /// carries what the guardrail chain did to the body.
    ///
    /// The mask count is the part worth pinning. `/v1/messages` merges the
    /// post-block-check masking pass into the counts its event reports
    /// (#932), and this route screens the same body with the same chain —
    /// so a mask counted on one and not the other reads as the sibling
    /// route masking more of the same payload. It is also invisible from
    /// the audit side: the enforced hit carries its own copy, so a reader
    /// checking only that would see the mask recorded while the field
    /// cp-api persists stayed empty. The upstream answers
    /// `input_tokens: 42`, which must NOT become spend: it measures a
    /// prompt, it does not consume one.
    #[tokio::test]
    async fn a_served_request_emits_a_zero_token_row_carrying_the_mask_it_applied() {
        use aisix_obs::UsageSink;

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages/count_tokens"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"input_tokens": 42})),
            )
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(anthropic_model("ct-mask"));
        snap.apikeys.insert(apikey_entry(&["ct-mask"]));
        let row: aisix_core::models::Guardrail = serde_json::from_str(
            r#"{
                "name": "eda-mask",
                "kind": "pii",
                "hook_point": "input",
                "detectors": [],
                "custom_patterns": [
                    {"name": "eda_version", "regex": "version\\s*:\\s*(\\d+(?:\\.\\d+)+)", "action": "mask", "replacement": "***"}
                ]
            }"#,
        )
        .unwrap();
        crate::seed_env_scoped_guardrail(&snap, ResourceEntry::new("g-mask", row, 1));

        let hub = Arc::new(Hub::new());
        hub.register_specialized(
            "anthropic",
            Arc::new(aisix_provider_anthropic::AnthropicBridge::new()),
        );
        let handle = SnapshotHandle::new(snap);
        let index = aisix_guardrails::LiveGuardrailIndex::new(handle.clone(), None);
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = crate::build_router(
            crate::ProxyState::new(handle, hub, &cfg())
                .without_cache()
                .with_guardrail_index(index)
                .with_usage_sink(UsageSink::new(tx)),
        );

        let res = app
            .oneshot(make_req(serde_json::json!({
                "model": "ct-mask",
                "messages": [{ "role": "user", "content": "version: 9.9.9" }],
            })))
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);

        let event = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
            .await
            .expect("count_tokens must emit a usage event")
            .expect("channel open");
        assert_eq!(event.status_code, 200);
        assert_eq!(event.inbound_protocol, "anthropic");
        assert_eq!(event.requested_model, "ct-mask");
        assert_eq!(event.prompt_tokens, 0);
        assert_eq!(event.completion_tokens, 0);
        assert_eq!(
            event.redacted_entity_counts.get("eda_version").copied(),
            Some(1),
            "the mask this route applied is missing from its own row: {event:?}",
        );
    }

    /// Mixed group [anthropic, openai]: the openai target is `continue`d
    /// past (count_tokens has no upstream there), so it is NOT a usable
    /// fallback — the default retry budget must apply on the anthropic
    /// target as if it were the last one. Counting the skipped target as
    /// a fallback would have suppressed the budget and failed the request
    /// on the first transient 502.
    #[tokio::test]
    async fn mixed_group_spends_the_default_budget_on_the_only_anthropic_target() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages/count_tokens"))
            .respond_with(ResponseTemplate::new(502).set_body_string("upstream down"))
            .up_to_n_times(2)
            .with_priority(1)
            .mount(&upstream)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/messages/count_tokens"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"input_tokens": 7})),
            )
            .with_priority(2)
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        let claude = {
            let json = format!(
                r#"{{"display_name":"ct-claude","provider":"anthropic","model_name":"claude-haiku-4-5-20251001","provider_key_id":"{PK_ID}"}}"#
            );
            let m: Model = serde_json::from_str(&json).unwrap();
            ResourceEntry::new("m-ct-claude", m, 1)
        };
        let gpt = {
            let json = format!(
                r#"{{"display_name":"ct-gpt","provider":"openai","model_name":"gpt-4o","provider_key_id":"{PK_ID}"}}"#
            );
            let m: Model = serde_json::from_str(&json).unwrap();
            ResourceEntry::new("m-ct-gpt", m, 1)
        };
        let group = {
            let json = r#"{"display_name":"ct-mixed","routing":{"strategy":"failover","targets":[{"model":"ct-claude"},{"model":"ct-gpt"}]}}"#;
            let m: Model = serde_json::from_str(json).unwrap();
            ResourceEntry::new("m-ct-mixed", m, 1)
        };
        snap.models.insert(claude);
        snap.models.insert(gpt);
        snap.models.insert(group);
        snap.apikeys.insert(apikey_entry(&["ct-mixed"]));

        let app = build_app(snap);
        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "ct-mixed",
                "messages": [{"role": "user", "content": "hi"}]
            })))
            .await
            .unwrap();

        // Two 502s absorbed by the default budget, third attempt wins.
        assert_eq!(resp.status(), StatusCode::OK);
        let received = upstream.received_requests().await.unwrap();
        assert_eq!(
            received.len(),
            3,
            "initial + 2 retries on the sole anthropic target",
        );
    }

    #[tokio::test]
    async fn unauthenticated_returns_401_anthropic_envelope() {
        let snap = new_snap("http://unused");
        let app = build_app(snap);

        let req = Request::builder()
            .method("POST")
            .uri("/v1/messages/count_tokens")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        // Anthropic-shape envelope: `{type:"error", error:{type,message}}`.
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["type"], "error");
        assert_eq!(v["error"]["type"], "authentication_error");
    }

    #[tokio::test]
    async fn unknown_model_returns_404() {
        let snap = new_snap("http://unused");
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "no-such-model",
                "messages": [{"role": "user", "content": "hi"}]
            })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn forbidden_model_returns_403() {
        let snap = new_snap("http://unused");
        snap.models.insert(anthropic_model("claude-haiku"));
        snap.apikeys.insert(apikey_entry(&["other-model"]));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "claude-haiku",
                "messages": [{"role": "user", "content": "hi"}]
            })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    /// A non-Anthropic Model has no upstream count_tokens surface;
    /// reject at the boundary with 400 (Anthropic-shape) rather than
    /// 404-ing the caller or dispatching to an upstream that would 404.
    #[tokio::test]
    async fn non_anthropic_provider_returns_400() {
        let snap = AisixSnapshot::new();
        let pk_json = r#"{"display_name":"openai-up","secret":"sk-openai","api_base":"https://api.openai.com","provider":"openai","adapter":"openai"}"#;
        let pk: aisix_core::ProviderKey = serde_json::from_str(pk_json).unwrap();
        snap.provider_keys.insert(ResourceEntry::new(PK_ID, pk, 1));
        snap.models.insert(openai_model("gpt-model"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "gpt-model",
                "messages": [{"role": "user", "content": "hi"}]
            })))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["type"], "error");
        assert_eq!(v["error"]["type"], "invalid_request_error");
        let message = v["error"]["message"].as_str().unwrap();
        assert!(message.contains("Anthropic"), "got {message:?}");
    }

    /// Regression: a >1024-byte upstream error body whose 1024th byte
    /// falls mid-codepoint must not panic the handler — a raw
    /// `&message[..1024]` slice would. Reaching the assertions at all
    /// proves no panic; the upstream 5xx collapses to a gateway 5xx with
    /// the Anthropic-shape error envelope.
    #[tokio::test]
    async fn oversize_non_ascii_upstream_error_does_not_panic() {
        // 1023 ASCII bytes + a 3-byte '€' occupying bytes 1023..1026, so
        // byte index 1024 lands in the middle of a multibyte character.
        let big_body = format!("{}€", "a".repeat(1023));
        assert!(!big_body.is_char_boundary(1024), "test setup invariant");

        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages/count_tokens"))
            .respond_with(ResponseTemplate::new(500).set_body_string(big_body))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(anthropic_model("claude-haiku"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "claude-haiku",
                "messages": [{"role": "user", "content": "hi"}]
            })))
            .await
            .unwrap();

        assert!(resp.status().is_server_error());
        let bytes = to_bytes(resp.into_body(), 4096).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["type"], "error");
    }

    /// #418 happy path: the route is registered, dispatches to the
    /// Anthropic upstream at `…/v1/messages/count_tokens`, rewrites the
    /// model field, sends the Anthropic auth headers, and returns the
    /// `{"input_tokens": <n>}` body verbatim.
    #[tokio::test]
    async fn happy_path_forwards_to_anthropic_count_tokens() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages/count_tokens"))
            .and(header("x-api-key", "sk-ant-test"))
            .and(header("anthropic-version", "2023-06-01"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "input_tokens": 17
            })))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(anthropic_model("claude-haiku"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "claude-haiku",
                "messages": [{"role": "user", "content": "hello"}]
            })))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["input_tokens"], 17);

        // The model field must be rewritten to the upstream id, and the
        // request must reach the count_tokens sub-route (not /v1/messages).
        let received = upstream.received_requests().await.unwrap();
        assert_eq!(received.len(), 1);
        assert_eq!(received[0].url.path(), "/v1/messages/count_tokens");
        let sent: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
        assert_eq!(sent["model"], "claude-haiku-4-5-20251001");
        assert_eq!(sent["messages"][0]["content"], "hello");
    }

    /// A `provider: "byo"` model whose ProviderKey carries
    /// `adapter: anthropic` fronts an Anthropic-protocol upstream, so
    /// count_tokens must serve it exactly like the catalog vendor.
    /// Gating on the vendor id rejected it with a 400 while its sibling
    /// `/v1/messages` happily served the same model.
    #[tokio::test]
    async fn byo_model_on_the_anthropic_adapter_is_served() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages/count_tokens"))
            .and(header("x-api-key", "sk-byo"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "input_tokens": 23
            })))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        let pk_json = format!(
            r#"{{"display_name":"byo-anthropic","secret":"sk-byo","api_base":"{}","provider":"byo","adapter":"anthropic"}}"#,
            upstream.uri()
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&pk_json).unwrap();
        snap.provider_keys.insert(ResourceEntry::new(PK_ID, pk, 1));
        let model_json = format!(
            r#"{{"display_name":"byo-claude","provider":"byo","model_name":"claude-sonnet-4-5","provider_key_id":"{PK_ID}"}}"#
        );
        let m: Model = serde_json::from_str(&model_json).unwrap();
        snap.models.insert(ResourceEntry::new("m-1", m, 1));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "byo-claude",
                "messages": [{"role": "user", "content": "hello"}]
            })))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["input_tokens"], 23);
        let received = upstream.received_requests().await.unwrap();
        assert_eq!(received[0].url.path(), "/v1/messages/count_tokens");
    }

    // ─── PK request.* overrides must apply identically to /v1/messages ──
    //
    // count_tokens shares the same Anthropic ProviderKey as /v1/messages,
    // so the operator's `request.*` overrides must reach this sibling too.
    // The mocks strict-match the EXPECTED post-override shape — if an
    // override silently no-ops, the matcher rejects the request and
    // wiremock 404s, surfacing here as a non-200.

    fn anthropic_pk_with_overrides(
        api_base: &str,
        request_overrides: serde_json::Value,
    ) -> ResourceEntry<aisix_core::ProviderKey> {
        let json = serde_json::json!({
            "display_name": "anthropic-up",
            "secret": "sk-ant-test",
            "api_base": api_base,
            "provider": "anthropic",
            "adapter": "anthropic",
            "request": request_overrides,
        });
        let pk: aisix_core::ProviderKey = serde_json::from_value(json).unwrap();
        ResourceEntry::new(PK_ID, pk, 1)
    }

    /// The concrete count_tokens case: an operator `default_headers`
    /// block injecting `anthropic-beta` must reach the upstream request.
    #[tokio::test]
    async fn applies_default_headers_anthropic_beta() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages/count_tokens"))
            .and(header("anthropic-beta", "token-counting-2024-11-01"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "input_tokens": 5
            })))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.provider_keys.insert(anthropic_pk_with_overrides(
            &upstream.uri(),
            serde_json::json!({
                "default_headers": {"anthropic-beta": "token-counting-2024-11-01"}
            }),
        ));
        snap.models.insert(anthropic_model("claude-haiku"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "claude-haiku",
                "messages": [{"role": "user", "content": "hi"}]
            })))
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "default_headers must inject anthropic-beta on count_tokens"
        );
    }

    /// `param_renames` must rewrite the body field on the outbound
    /// count_tokens request, exactly as on /v1/messages.
    #[tokio::test]
    async fn applies_param_renames() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages/count_tokens"))
            .and(wiremock::matchers::body_partial_json(
                serde_json::json!({"renamed_field": "v"}),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "input_tokens": 5
            })))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.provider_keys.insert(anthropic_pk_with_overrides(
            &upstream.uri(),
            serde_json::json!({
                "param_renames": {"orig_field": "renamed_field"}
            }),
        ));
        snap.models.insert(anthropic_model("claude-haiku"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "claude-haiku",
                "messages": [{"role": "user", "content": "hi"}],
                "orig_field": "v"
            })))
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "param_renames must rewrite orig_field → renamed_field on count_tokens"
        );
    }

    /// Operator `default_headers` must NOT be able to overwrite the
    /// gateway-owned `x-api-key` auth header (ai-gateway#337) — the
    /// reserved blacklist in `apply_request_headers` protects it.
    #[tokio::test]
    async fn default_headers_cannot_overwrite_x_api_key() {
        let upstream = MockServer::start().await;
        // Mock only 200s when x-api-key is the PK secret, NOT the value
        // the operator tried to smuggle via default_headers.
        Mock::given(method("POST"))
            .and(path("/v1/messages/count_tokens"))
            .and(header("x-api-key", "sk-ant-test"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "input_tokens": 5
            })))
            .mount(&upstream)
            .await;

        let snap = AisixSnapshot::new();
        snap.provider_keys.insert(anthropic_pk_with_overrides(
            &upstream.uri(),
            serde_json::json!({
                "default_headers": {"x-api-key": "attacker-key"}
            }),
        ));
        snap.models.insert(anthropic_model("claude-haiku"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        let app = build_app(snap);

        let resp = app
            .oneshot(make_req(serde_json::json!({
                "model": "claude-haiku",
                "messages": [{"role": "user", "content": "hi"}]
            })))
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "the PK secret x-api-key must survive an operator default_headers override attempt"
        );
    }
}
