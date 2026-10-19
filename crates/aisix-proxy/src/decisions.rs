//! `POST /v1/systemone` and `POST /v1/decisions` — structured decision
//! endpoints.
//!
//! The two routes take the same kind of request in two wire formats: the
//! System One format (TypeSafe's `{state, questions: {<name>: …}}`) and
//! OpenAI's Decisions format (`{input, questions: [...]}`). Which upstream
//! serves a request is decided by the resolved target's Provider Key:
//!
//! | provider     | upstream                                           | format     |
//! |--------------|----------------------------------------------------|------------|
//! | `typesafe`   | `{api_base or https://api.typesafe.ai}/v1/systemone` | System One |
//! | `azure`      | Microsoft Foundry `{base}/providers/microsoft/v1/systemone` | System One |
//! | `openrouter` | `{api_base}/systemone`                             | System One |
//! | `openai`     | `{api_base or https://api.openai.com/v1}/decisions` | Decisions  |
//!
//! When the caller's format is the upstream's, the body goes out verbatim
//! with only `model` rewritten and comes back verbatim with only `model`
//! restamped (`model_echo`), so fields this gateway does not know pass in
//! both directions. Otherwise the request and response are converted
//! (`decisions_convert`).
//!
//! Attached guardrails do not run here, and nothing is redacted: that is the
//! product decision for these endpoints (`guardrail_coverage` declares it).

use aisix_obs::{content_capture_cap, AccessLog, CapturedContent, UsageEvent};
use axum::extract::State;
use axum::http::HeaderValue;
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use serde_json::Value;
use std::time::{Duration, Instant};

use crate::auth::AuthenticatedKey;
use crate::client_ip::ClientContext;
use crate::decisions_convert::{self as convert, DecisionUsage, Json};
use crate::error::ProxyError;
use crate::operation::Surface;
use crate::state::ProxyState;

/// The two decision wire formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    SystemOne,
    Decisions,
}

/// One of the two client-facing routes.
struct Endpoint {
    path: &'static str,
    format: Format,
    surface: Surface,
}

const SYSTEMONE: Endpoint = Endpoint {
    path: "/v1/systemone",
    format: Format::SystemOne,
    surface: crate::operation::SYSTEMONE,
};

const DECISIONS: Endpoint = Endpoint {
    path: "/v1/decisions",
    format: Format::Decisions,
    surface: crate::operation::DECISIONS,
};

/// The providers that serve a decision request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Upstream {
    TypeSafe,
    /// Microsoft Foundry, reached through an `azure` Provider Key.
    Foundry,
    OpenRouter,
    OpenAi,
}

impl Upstream {
    fn of(provider: &str) -> Option<Self> {
        match provider.trim().to_ascii_lowercase().as_str() {
            aisix_core::models::TYPESAFE_PROVIDER => Some(Upstream::TypeSafe),
            "azure" => Some(Upstream::Foundry),
            "openrouter" => Some(Upstream::OpenRouter),
            "openai" => Some(Upstream::OpenAi),
            _ => None,
        }
    }

    fn format(self) -> Format {
        match self {
            Upstream::OpenAi => Format::Decisions,
            Upstream::TypeSafe | Upstream::Foundry | Upstream::OpenRouter => Format::SystemOne,
        }
    }
}

pub async fn systemone(
    State(state): State<ProxyState>,
    auth: AuthenticatedKey,
    client: ClientContext,
    body: Result<Bytes, axum::extract::rejection::BytesRejection>,
) -> Response {
    handle(&SYSTEMONE, state, auth, client, body).await
}

pub async fn decisions(
    State(state): State<ProxyState>,
    auth: AuthenticatedKey,
    client: ClientContext,
    body: Result<Bytes, axum::extract::rejection::BytesRejection>,
) -> Response {
    handle(&DECISIONS, state, auth, client, body).await
}

/// What a successful dispatch hands the handler for its telemetry.
struct Success {
    response: Response,
    provider: String,
    model_id: String,
    provider_key_id: String,
    upstream_model: String,
    /// The model name the upstream reported answering with.
    reported_model: String,
    provider_request_id: String,
    usage: DecisionUsage,
    captured_content: Option<CapturedContent>,
}

async fn handle(
    endpoint: &Endpoint,
    state: ProxyState,
    auth: AuthenticatedKey,
    client: ClientContext,
    body: Result<Bytes, axum::extract::rejection::BytesRejection>,
) -> Response {
    let started = Instant::now();
    let reject = |err: ProxyError| {
        crate::reject::reject_before_dispatch(
            &state,
            "POST",
            endpoint.path,
            &client.request_id,
            Some(&auth.entry.id),
            started,
            crate::reject::Envelope::OpenAi,
            err,
        )
    };
    let raw = match body {
        Ok(raw) => raw,
        Err(rej) => {
            return reject(crate::error::proxy_error_from_bytes_rejection(
                rej,
                state.request_body_limit_bytes,
            ))
        }
    };
    let doc = match serde_json::from_slice::<Json>(&raw) {
        Ok(doc) => doc,
        Err(_) => {
            return reject(ProxyError::InvalidRequest(
                "invalid JSON request body".into(),
            ))
        }
    };
    let request_id = client.request_id.clone();
    let api_key_id = auth.entry.id.clone();
    let model_name = doc.model().unwrap_or("").to_string();

    let snapshot = state.snapshot.load();
    // No guardrail chain is ever resolved on these endpoints, so the audit
    // the emitters read stays empty.
    let audit = crate::usage_attr::GuardrailAudit::default();
    let mut routing = crate::attempt::RoutingTelemetry::default();
    match dispatch(
        endpoint,
        &state,
        &snapshot,
        &auth,
        &client,
        &raw,
        &doc,
        &mut routing,
    )
    .await
    {
        Ok(success) => {
            let elapsed = started.elapsed();
            let status = success.response.status().as_u16();
            emit_access_log(
                endpoint,
                &model_name,
                &success.provider,
                &api_key_id,
                status,
                elapsed,
                &request_id,
                Some(success.provider_request_id.as_str()),
                &routing,
                None,
            );
            let pk = crate::usage_attr::ResolvedPk::resolve(&snapshot, &success.provider_key_id);
            crate::request_metrics::record(
                &state,
                endpoint.path,
                crate::request_metrics::Caller::new(&auth),
                crate::request_metrics::Upstream {
                    provider: &success.provider,
                    model: &model_name,
                    upstream_model: &success.upstream_model,
                    pk: pk.labels(),
                    is_fallback: routing.fallback_count() > 0,
                    ..Default::default()
                },
                status,
                elapsed,
            );
            crate::usage_attr::emit_failed_attempts(
                &state,
                &snapshot,
                endpoint.surface,
                &request_id,
                &model_name,
                &api_key_id,
                &client,
                &[],
                &routing.attempts,
                /* terminal_last */ false,
                false,
                Vec::new(),
                crate::redact::RedactionCounts::new(),
                &audit,
            );
            emit_usage_event(
                endpoint,
                &state,
                &snapshot,
                &pk,
                &request_id,
                &model_name,
                &api_key_id,
                status,
                elapsed,
                &success,
                &client,
                routing.winner(),
            );
            success.response
        }
        Err(err) => {
            let status = err.status().as_u16();
            let elapsed = started.elapsed();
            emit_access_log(
                endpoint,
                &model_name,
                "unknown",
                &api_key_id,
                status,
                elapsed,
                &request_id,
                None,
                &routing,
                Some(&err),
            );
            let metric_model = crate::usage_attr::metric_model_label(&snapshot, &model_name);
            let attributed = crate::attribution::current().unwrap_or_default();
            let last_target = crate::request_metrics::LastTarget::new(&snapshot, &attributed);
            crate::request_metrics::record(
                &state,
                endpoint.path,
                crate::request_metrics::Caller::new(&auth),
                last_target.upstream(metric_model.as_ref(), false, routing.fallback_count() > 0),
                status,
                elapsed,
            );
            let failed_terminal = crate::usage_attr::failed_attempts_are_terminal(&routing);
            crate::usage_attr::emit_failed_attempts(
                &state,
                &snapshot,
                endpoint.surface,
                &request_id,
                &model_name,
                &api_key_id,
                &client,
                &[],
                &routing.attempts,
                failed_terminal,
                false,
                Vec::new(),
                crate::redact::RedactionCounts::new(),
                &audit,
            );
            if !failed_terminal {
                crate::usage_attr::emit_error_usage_event(
                    &state,
                    &snapshot,
                    endpoint.surface,
                    "openai",
                    &request_id,
                    &model_name,
                    &api_key_id,
                    status,
                    crate::usage_attr::EventError::of(&err),
                    false,
                    &client,
                    crate::usage_attr::applied_guardrails(&audit),
                    crate::usage_attr::enforced_hits(&audit),
                    crate::usage_attr::guardrail_scores(&audit),
                    crate::usage_attr::bypass_reason(&audit),
                );
            }
            err.into_response()
        }
    }
}

/// The kind a decision endpoint refuses to serve, if `model` is one.
fn unserved_kind(model: &aisix_core::Model) -> Option<&'static str> {
    if model.is_ensemble() {
        Some("ensemble")
    } else if model.is_semantic() {
        Some("semantic")
    } else if model.is_embedding() {
        Some("embedding")
    } else {
        None
    }
}

fn unserved_kind_error(endpoint: &Endpoint, label: String, kind: &str) -> ProxyError {
    ProxyError::InvalidRequest(format!(
        "{label} is a `{kind}` model; {} serves direct and routing models only",
        endpoint.path
    ))
}

#[allow(clippy::too_many_arguments)]
async fn dispatch(
    endpoint: &Endpoint,
    state: &ProxyState,
    snapshot: &aisix_core::AisixSnapshot,
    auth: &AuthenticatedKey,
    client_ctx: &ClientContext,
    raw: &Bytes,
    doc: &Json,
    routing_out: &mut crate::attempt::RoutingTelemetry,
) -> Result<Success, ProxyError> {
    if !matches!(doc, Json::Object(_)) {
        return Err(ProxyError::InvalidRequest(
            "request body must be a JSON object".into(),
        ));
    }
    let model_name = doc
        .model()
        .ok_or_else(|| ProxyError::InvalidRequest("`model` field missing".into()))?
        .to_string();
    let model_entry = crate::model_resolve::resolve_model(snapshot, &model_name)
        .ok_or_else(|| ProxyError::ModelNotFound(model_name.clone()))?;
    if !auth.key().can_access(snapshot, &model_name) {
        return Err(ProxyError::ModelForbidden(model_name.clone()));
    }
    crate::dispatch::check_ip_access(&model_entry.value, &client_ctx.source_ip)?;
    if let Some(kind) = unserved_kind(&model_entry.value) {
        return Err(unserved_kind_error(
            endpoint,
            crate::routing::refused_model_label(&model_name, &model_entry.value),
            kind,
        ));
    }

    // The caller's request in the intermediate shape, for a target whose
    // upstream speaks the other format. Read once; a body it cannot convert
    // is only an error for such a target.
    let converted = match endpoint.format {
        Format::SystemOne => convert::parse_systemone_request(doc),
        Format::Decisions => convert::parse_decisions_request(doc),
    };

    let content_cap = content_capture_cap(
        snapshot
            .observability_exporters
            .entries()
            .iter()
            .map(|e| &e.value),
    );

    let model_rl =
        crate::quota::ModelRateLimit::from_model(&model_name, &model_entry.id, &model_entry.value);
    let reservation = crate::quota::enforce(state, snapshot, auth, Some(&model_rl)).await?;

    *routing_out = crate::attempt::RoutingTelemetry::for_request(&model_entry.value.display_name)
        .with_trace(client_ctx.trace.clone());
    let request_id = client_ctx.request_id.as_str();
    let dispatched = crate::routing::dispatch_with_failover(
        state,
        snapshot,
        auth,
        client_ctx,
        &model_name,
        &model_entry,
        routing_out,
        |_| true,
        |_| None,
        |target, timeouts| {
            let model_name = &model_name;
            let converted = &converted;
            async move {
                let model = &target.model;
                let label = || crate::routing::refused_model_label(model_name, model);
                if let Some(kind) = unserved_kind(model) {
                    return Err(unserved_kind_error(endpoint, label(), kind));
                }
                let pk_entry = crate::dispatch::resolve_provider_key(snapshot, model)?;
                let pk = &pk_entry.value;
                let Some(upstream) = Upstream::of(&pk.provider) else {
                    return Err(ProxyError::InvalidRequest(format!(
                        "{} uses provider `{}`; {} requires a typesafe, azure, openai, or \
                         openrouter provider",
                        label(),
                        pk.provider,
                        endpoint.path
                    )));
                };
                let api_key = crate::dispatch::require_api_key(pk, model)?.to_string();
                let upstream_model = crate::dispatch::require_upstream_model(model)?.to_string();

                let outbound = if upstream.format() == endpoint.format {
                    // Verbatim but for `model`, spliced so every other byte
                    // the caller wrote goes out as written.
                    crate::model_echo::restamp_json_bytes(
                        raw,
                        &upstream_model,
                        crate::model_echo::top_level_model,
                    )
                    .map_or_else(
                        // A `model` key the byte scanner does not match
                        // (an escaped spelling, a duplicate) still has to
                        // carry the upstream model.
                        || serde_json::to_vec(&doc.with_model(&upstream_model)).map(Bytes::from),
                        |spliced| Ok(Bytes::from(spliced)),
                    )
                    .map_err(|e| {
                        ProxyError::Bridge(aisix_gateway::BridgeError::Config(format!(
                            "serialize request body: {e}"
                        )))
                    })?
                } else {
                    let request = converted
                        .as_ref()
                        .map_err(|msg| ProxyError::InvalidRequest(msg.clone()))?;
                    let doc = match upstream.format() {
                        Format::SystemOne => {
                            convert::to_systemone_request(&upstream_model, request)
                        }
                        Format::Decisions => {
                            convert::to_decisions_request(&upstream_model, request)
                        }
                    }
                    .map_err(ProxyError::InvalidRequest)?;
                    Bytes::from(serde_json::to_vec(&doc).map_err(|e| {
                        ProxyError::Bridge(aisix_gateway::BridgeError::Config(format!(
                            "serialize request body: {e}"
                        )))
                    })?)
                };
                let outbound = apply_body_overrides(pk, outbound)?;

                let url = aisix_gateway::url_cache::cached_endpoint_url(
                    &pk_entry.id,
                    "proxy/decisions",
                    &[pk.api_base.as_deref().unwrap_or(""), &pk.provider],
                    || upstream_url(upstream, pk),
                )?;

                let mut headers = axum::http::HeaderMap::new();
                let invalid = |what: &str, e: axum::http::header::InvalidHeaderValue| {
                    ProxyError::Bridge(aisix_gateway::BridgeError::Config(format!(
                        "{what} contains invalid header chars: {e}"
                    )))
                };
                let (auth_name, auth_value) = match upstream {
                    Upstream::Foundry => {
                        match aisix_provider_azure_openai::resolve_credential(&api_key)
                            .await
                            .map_err(ProxyError::Bridge)?
                        {
                            aisix_provider_azure_openai::AzureCredential::ApiKey(key) => (
                                axum::http::HeaderName::from_static("api-key"),
                                HeaderValue::from_str(&key).map_err(|e| invalid("api key", e))?,
                            ),
                            aisix_provider_azure_openai::AzureCredential::Bearer(token) => (
                                axum::http::header::AUTHORIZATION,
                                HeaderValue::from_str(&format!("Bearer {token}"))
                                    .map_err(|e| invalid("access token", e))?,
                            ),
                        }
                    }
                    _ => (
                        axum::http::header::AUTHORIZATION,
                        HeaderValue::from_str(&format!("Bearer {api_key}"))
                            .map_err(|e| invalid("api key", e))?,
                    ),
                };
                headers.insert(auth_name, auth_value);
                headers.insert(
                    axum::http::header::CONTENT_TYPE,
                    HeaderValue::from_static("application/json"),
                );
                headers.insert(
                    axum::http::header::HeaderName::from_static("x-aisix-request-id"),
                    HeaderValue::from_str(request_id).map_err(|e| invalid("request_id", e))?,
                );
                aisix_gateway::apply_request_headers(
                    &mut headers,
                    &crate::dispatch::upstream_header_ctx(
                        pk,
                        &pk_entry.id,
                        model,
                        &target.id,
                        client_ctx,
                    ),
                );

                let http = crate::http_client::client_for(pk.upstream_connection().as_ref());
                let tracker = &state.runtime_status;
                let model_id: &str = &target.id;
                let cooldown_cfg = model.cooldown.as_ref();
                let mut req = url.post_on(&http).headers(headers).body(outbound);
                if let Some(d) = timeouts.request {
                    req = req.timeout(d);
                }
                let send_started = Instant::now();
                let upstream_resp = req.send().await.map_err(|e| {
                    ProxyError::Bridge(crate::cooldown::note_failure(
                        tracker,
                        model_id,
                        cooldown_cfg,
                        crate::dispatch::reqwest_error_to_bridge(&e, send_started),
                    ))
                })?;
                aisix_gateway::note_upstream_response(upstream_resp.headers());
                let status = upstream_resp.status();
                if !status.is_success() {
                    let retry_after = aisix_gateway::parse_retry_after(upstream_resp.headers());
                    let error_body = upstream_resp.bytes().await.unwrap_or_default();
                    return Err(ProxyError::Bridge(crate::cooldown::note_failure(
                        tracker,
                        model_id,
                        cooldown_cfg,
                        aisix_gateway::BridgeError::upstream_status_with_retry_after(
                            status.as_u16(),
                            upstream_error_message(&error_body)
                                .chars()
                                .take(1024)
                                .collect::<String>(),
                            retry_after,
                        ),
                    )));
                }
                let body_bytes = upstream_resp.bytes().await.map_err(|e| {
                    ProxyError::Bridge(crate::cooldown::note_failure(
                        tracker,
                        model_id,
                        cooldown_cfg,
                        aisix_gateway::BridgeError::UpstreamDecode(e.to_string()),
                    ))
                })?;
                // A converted answer is read here, inside the attempt, so a 200
                // that is not JSON, or whose answers lack a field their type
                // requires, counts against the target like any other upstream
                // fault instead of after it was marked healthy.
                let answers = match converted {
                    Ok(request) if upstream.format() != endpoint.format => {
                        let read = serde_json::from_slice::<Value>(&body_bytes)
                            .map_err(|_| {
                                "the decision upstream answered with a body that is not JSON"
                                    .to_string()
                            })
                            .and_then(|v| match upstream.format() {
                                Format::SystemOne => convert::answers_from_systemone(&v, request),
                                Format::Decisions => convert::answers_from_decisions(&v, request),
                            });
                        Some(read.map_err(|msg| {
                            ProxyError::Bridge(crate::cooldown::note_failure(
                                tracker,
                                model_id,
                                cooldown_cfg,
                                aisix_gateway::BridgeError::UpstreamDecode(msg),
                            ))
                        })?)
                    }
                    _ => None,
                };
                Ok((
                    body_bytes,
                    answers,
                    upstream,
                    pk.provider.clone(),
                    pk_entry.id.to_string(),
                    upstream_model,
                ))
            }
        },
    )
    .await;
    let captured_prompt = content_cap.map(|_| String::from_utf8_lossy(raw).into_owned());
    let crate::routing::Dispatched {
        value: (body_bytes, answers, upstream, provider, pk_id, upstream_model),
        target,
        member_reservation,
        ..
    } = match dispatched {
        Ok(d) => d,
        Err(e) => {
            reservation.commit_tokens(0).await;
            return Err(e);
        }
    };
    let mut reservation = reservation;
    if let Some(member) = member_reservation {
        reservation.merge(member);
    }
    state.health.record_success(&target.model.display_name);
    state.runtime_status.mark_healthy(&target.id);

    let parsed = serde_json::from_slice::<Value>(&body_bytes);
    let (usage, reported_model, provider_request_id) = match &parsed {
        Ok(v) => (
            match upstream.format() {
                Format::SystemOne => convert::systemone_usage(v),
                Format::Decisions => convert::decisions_usage(v),
            },
            v.get("model")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            crate::usage_attr::provider_response_id(v),
        ),
        Err(e) => {
            tracing::warn!(
                request_id = %request_id,
                model = %model_name,
                error = %e,
                "decision upstream body parse failed; no usage counts to attribute"
            );
            (None, String::new(), String::new())
        }
    };
    let usage = usage.unwrap_or_default();
    reservation
        .commit_tokens(u64::from(usage.input) + u64::from(usage.output))
        .await;

    let client_body = match (answers, &converted) {
        (Some(answers), Ok(request)) => {
            let doc = match endpoint.format {
                Format::Decisions => {
                    convert::to_decisions_response(&model_name, request, &answers, &usage)
                }
                Format::SystemOne => {
                    convert::to_systemone_response(&model_name, request, &answers, &usage)
                }
            };
            Bytes::from(serde_json::to_vec(&doc).unwrap_or_default())
        }
        _ => crate::model_echo::restamp_json_bytes(
            &body_bytes,
            &model_name,
            crate::model_echo::top_level_model,
        )
        .map(Bytes::from)
        .unwrap_or(body_bytes),
    };

    let captured_content = match (&captured_prompt, content_cap) {
        (Some(prompt), Some(cap)) => Some(CapturedContent::new(
            prompt,
            &String::from_utf8_lossy(&client_body),
            cap as usize,
        )),
        _ => None,
    };

    let mut resp = Response::new(axum::body::Body::from(client_body));
    resp.headers_mut().insert(
        axum::http::header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    resp.headers_mut().insert(
        axum::http::header::HeaderName::from_static("x-aisix-request-id"),
        HeaderValue::from_str(request_id).unwrap_or_else(|_| HeaderValue::from_static("")),
    );

    Ok(Success {
        response: resp,
        provider,
        model_id: target.id.clone(),
        provider_key_id: pk_id,
        upstream_model,
        reported_model,
        provider_request_id,
        usage,
        captured_content,
    })
}

/// Apply the Provider Key's `request.*` body overrides, as every other
/// directly-built upstream request does. A no-op without overrides, so the
/// verbatim body is only re-serialised when an operator asked for a change.
fn apply_body_overrides(
    pk: &aisix_core::ProviderKey,
    outbound: Bytes,
) -> Result<Bytes, ProxyError> {
    let Some(r) = pk.request.as_ref() else {
        return Ok(outbound);
    };
    if r.param_renames.is_empty()
        && r.param_constraints.is_none()
        && r.default_body_fields.is_empty()
    {
        return Ok(outbound);
    }
    let mut body: Value = serde_json::from_slice(&outbound).map_err(|e| {
        ProxyError::Bridge(aisix_gateway::BridgeError::Config(format!(
            "re-read request body: {e}"
        )))
    })?;
    aisix_provider_openai::overrides::apply_param_renames(&mut body, &r.param_renames);
    if let Some(constraints) = &r.param_constraints {
        aisix_provider_openai::overrides::apply_param_constraints(&mut body, constraints);
    }
    aisix_provider_openai::overrides::apply_default_body_fields(&mut body, &r.default_body_fields);
    crate::util::outbound_json(&body)
}

/// Endpoint a `typesafe` Provider Key reaches when it sets no `api_base`.
const TYPESAFE_DEFAULT_API_BASE: &str = crate::semantic::TYPESAFE_DEFAULT_API_BASE;

fn upstream_url(upstream: Upstream, pk: &aisix_core::ProviderKey) -> Result<String, ProxyError> {
    let configured = pk
        .api_base
        .as_deref()
        .map(str::trim)
        .filter(|b| !b.is_empty());
    Ok(match upstream {
        Upstream::TypeSafe => {
            let base = configured.unwrap_or(TYPESAFE_DEFAULT_API_BASE);
            let base = base.trim_end_matches('/');
            let base = base.strip_suffix("/systemone").unwrap_or(base);
            format!("{}/v1/systemone", base.strip_suffix("/v1").unwrap_or(base))
        }
        Upstream::Foundry => {
            let base = configured.ok_or_else(|| {
                ProxyError::InvalidRequest(format!(
                    "provider_key {:?} has no api_base configured",
                    pk.display_name
                ))
            })?;
            format!("{}/providers/microsoft/v1/systemone", foundry_root(base))
        }
        Upstream::OpenRouter => {
            crate::dispatch::build_openai_url(&crate::dispatch::resolve_base_url(pk)?, "/systemone")
        }
        Upstream::OpenAi => {
            crate::dispatch::build_openai_url(&crate::dispatch::resolve_base_url(pk)?, "/decisions")
        }
    })
}

/// The Foundry resource root an Azure key's `api_base` names. Operators
/// paste whichever endpoint the portal showed them — the project endpoint,
/// the OpenAI-compatible one, or the models one — and all of them sit under
/// the same root.
fn foundry_root(api_base: &str) -> &str {
    let mut base = api_base.trim_end_matches('/');
    if let Some(i) = base.rfind("/api/projects/") {
        if !base[i + "/api/projects/".len()..].contains('/') {
            base = base[..i].trim_end_matches('/');
        }
    }
    for suffix in ["/openai/v1", "/openai", "/models", "/v1"] {
        if let Some(rest) = base.strip_suffix(suffix) {
            return rest.trim_end_matches('/');
        }
    }
    base
}

/// A readable message out of a decision upstream's error body.
///
/// The shapes in use: TypeSafe answers `{detail: string}` or `{detail:
/// {error_type, message}}`; OpenAI and Foundry `{error: {message, …}}`;
/// OpenRouter the same, except that its validation failures put a
/// JSON-encoded list of issues in `error.message`.
fn upstream_error_message(body: &[u8]) -> String {
    let Ok(v) = serde_json::from_slice::<Value>(body) else {
        return String::from_utf8_lossy(body).into_owned();
    };
    if let Some(message) = v.pointer("/error/message").and_then(Value::as_str) {
        return readable_issue_list(message).unwrap_or_else(|| message.to_string());
    }
    match v.get("detail") {
        Some(Value::String(detail)) => detail.clone(),
        Some(detail) => detail
            .get("message")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| detail.to_string()),
        None => String::from_utf8_lossy(body).into_owned(),
    }
}

/// `path.to.field: message; …` for a message that is itself a JSON list of
/// validation issues.
fn readable_issue_list(message: &str) -> Option<String> {
    let issues: Vec<Value> = serde_json::from_str(message).ok()?;
    let lines: Vec<String> = issues
        .iter()
        .filter_map(|issue| {
            let text = issue.get("message")?.as_str()?;
            let path: Vec<String> = issue
                .get("path")
                .and_then(Value::as_array)
                .map(|p| {
                    p.iter()
                        .map(|seg| seg.as_str().map_or_else(|| seg.to_string(), str::to_string))
                        .collect()
                })
                .unwrap_or_default();
            Some(if path.is_empty() {
                text.to_string()
            } else {
                format!("{}: {text}", path.join("."))
            })
        })
        .collect();
    (!lines.is_empty()).then(|| lines.join("; "))
}

#[allow(clippy::too_many_arguments)]
fn emit_usage_event(
    endpoint: &Endpoint,
    state: &ProxyState,
    snap: &aisix_core::AisixSnapshot,
    pk: &crate::usage_attr::ResolvedPk<'_>,
    request_id: &str,
    requested_model: &str,
    api_key_id: &str,
    status_code: u16,
    elapsed: Duration,
    success: &Success,
    client: &ClientContext,
    winner: Option<&crate::attempt::AttemptRecord>,
) {
    let usage = &success.usage;
    let elapsed_ms = elapsed.as_millis().min(u32::MAX as u128) as u32;
    let mut event = UsageEvent {
        // NO-GUARDRAIL-CHAIN: guardrails do not apply to the decision
        // endpoints, so no chain is ever resolved for this request.
        request_id: request_id.to_string(),
        occurred_at: aisix_obs::UsageEvent::occurred_at_now(),
        model_id: success.model_id.clone(),
        api_key_id: api_key_id.to_string(),
        requested_model: requested_model.to_string(),
        prompt_tokens: usage.input,
        completion_tokens: usage.output,
        cached_prompt_tokens: usage.cached,
        cache_write_tokens: usage.cache_write,
        modality_tokens: usage.modality,
        reasoning_tokens: usage.reasoning,
        total_tokens: usage.upstream_total,
        upstream_latency_ms: elapsed_ms,
        downstream_latency_ms: elapsed_ms,
        status_code,
        provider_request_id: success.provider_request_id.clone(),
        provider_model_version: success.reported_model.clone(),
        inbound_protocol: "openai".to_string(),
        client_source_ip: client.source_ip.clone(),
        client_user_agent: client.user_agent.clone(),
        request_headers: client.request_headers.clone(),
        ..Default::default()
    };
    crate::usage_attr::apply_pk_telemetry(&mut event, pk);
    crate::usage_attr::apply_winning_attempt(&mut event, winner);
    crate::usage_attr::apply_caller_identity(&mut event, client.jwt.as_ref(), &client.caller);
    let usage_model =
        crate::usage_attr::usage_event_model_label(snap, &event.requested_model).into_owned();
    crate::usage_attr::emit_usage(
        state,
        snap,
        endpoint.surface,
        event,
        crate::usage_attr::usage_event_labels(&usage_model, pk),
        success.captured_content.as_ref(),
        client.trace.as_ref(),
        /* terminal */ true,
        /* dispatched */ true,
    );
    let owned_caller = crate::request_metrics::Caller::from_api_key_id(snap, api_key_id);
    crate::request_metrics::record_usage(
        state,
        endpoint.path,
        owned_caller.as_caller(),
        crate::request_metrics::Upstream {
            provider: &success.provider,
            model: requested_model,
            upstream_model: &success.upstream_model,
            pk: pk.labels(),
            ..Default::default()
        },
        crate::request_metrics::Tokens {
            input: usage.input,
            output: usage.output,
            total: usage.input.saturating_add(usage.output),
            cached: usage.cached,
            cache_read: 0,
            cache_creation: 0,
            spend_usd: 0.0,
            client_type: state.client_classifier.classify(&client.user_agent),
        },
    );
}

#[allow(clippy::too_many_arguments)]
fn emit_access_log(
    endpoint: &Endpoint,
    model: &str,
    provider: &str,
    api_key_id: &str,
    status: u16,
    elapsed: Duration,
    request_id: &str,
    provider_request_id: Option<&str>,
    routing: &crate::attempt::RoutingTelemetry,
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
    let summary = routing.access_log_summary();
    crate::attribution::emit_access_log(AccessLog {
        method: "POST",
        path: endpoint.path,
        status,
        latency: elapsed,
        duration: elapsed,
        provider: Some(provider),
        model: Some(model),
        upstream_model: target.upstream_model(),
        provider_key_id: target.provider_key_id(),
        api_key_id: Some(api_key_id),
        prompt_tokens: None,
        completion_tokens: None,
        total_tokens: None,
        request_id,
        provider_request_id: provider_request_id.filter(|s| !s.is_empty()),
        served_by_model: summary.served_by_model,
        routing_attempt_count: summary.attempt_count,
        routing_fallback_count: summary.fallback_count,
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

    #[test]
    fn foundry_root_strips_every_documented_endpoint_form() {
        for base in [
            "https://acme.services.ai.azure.com",
            "https://acme.services.ai.azure.com/",
            "https://acme.services.ai.azure.com/api/projects/proj-1",
            "https://acme.services.ai.azure.com/openai/v1/",
            "https://acme.services.ai.azure.com/openai",
            "https://acme.services.ai.azure.com/models",
            "https://acme.services.ai.azure.com/v1",
        ] {
            assert_eq!(
                foundry_root(base),
                "https://acme.services.ai.azure.com",
                "{base}"
            );
        }
    }

    #[test]
    fn typesafe_url_tolerates_a_pasted_endpoint() {
        for base in [
            None,
            Some("https://api.typesafe.ai/"),
            Some("https://api.typesafe.ai/v1"),
            Some("https://api.typesafe.ai/v1/systemone"),
        ] {
            let pk: aisix_core::ProviderKey = serde_json::from_value(serde_json::json!({
                "display_name": "ts", "secret": "k", "provider": "typesafe", "api_base": base,
            }))
            .unwrap();
            assert_eq!(
                upstream_url(Upstream::TypeSafe, &pk).unwrap(),
                "https://api.typesafe.ai/v1/systemone",
                "{base:?}"
            );
        }
    }

    #[test]
    fn upstream_error_messages_are_read_from_each_shape() {
        assert_eq!(
            upstream_error_message(br#"{"detail":"Noul question must have criteria"}"#),
            "Noul question must have criteria"
        );
        assert_eq!(
            upstream_error_message(
                br#"{"detail":{"error_type":"api_usage_error","message":"Invalid request."}}"#
            ),
            "Invalid request."
        );
        assert_eq!(
            upstream_error_message(
                br#"{"error":{"message":"Missing required parameter","type":"invalid_request_error"}}"#
            ),
            "Missing required parameter"
        );
        let openrouter = serde_json::json!({"error": {"code": 400, "message":
            "[{\"code\":\"invalid_union\",\"path\":[\"questions\",\"is_bug\",\"instructions\"],\"message\":\"Invalid input\"}]"}});
        assert_eq!(
            upstream_error_message(openrouter.to_string().as_bytes()),
            "questions.is_bug.instructions: Invalid input"
        );
        assert_eq!(upstream_error_message(b"bad gateway"), "bad gateway");
    }

    /// A Foundry key whose secret is the Entra ID client-credentials shape
    /// authenticates with a minted bearer token instead of `api-key`, the
    /// same way the Azure bridge does.
    #[tokio::test]
    async fn foundry_entra_id_secret_sends_a_minted_bearer_token() {
        use aisix_core::resource::ResourceEntry;
        use tower::ServiceExt;
        use wiremock::matchers::{body_string_contains, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/tenant-1/oauth2/v2.0/token"))
            .and(body_string_contains("client_id=app-1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                serde_json::json!({"access_token": "entra-token", "expires_in": 3600}),
            ))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/providers/microsoft/v1/systemone"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "model": "jev-1.13.0",
                "answers": {"ok": {"type": "noul", "noul": 0.9}},
                "usage": {"input_tokens": 3, "output_tokens": 1}
            })))
            .expect(1)
            .mount(&server)
            .await;

        let secret = serde_json::json!({
            "tenant_id": "tenant-1",
            "client_id": "app-1",
            "client_secret": "s3cret",
            "authority_host": server.uri(),
        })
        .to_string();
        let snap = aisix_core::AisixSnapshot::new();
        let pk: aisix_core::ProviderKey = serde_json::from_value(serde_json::json!({
            "display_name": "foundry",
            "secret": secret,
            "api_base": format!("{}/api/projects/p1", server.uri()),
            "provider": "azure",
            "adapter": "azure-openai",
        }))
        .unwrap();
        snap.provider_keys.insert(ResourceEntry::new("pk-1", pk, 1));
        let model: aisix_core::Model = serde_json::from_value(serde_json::json!({
            "display_name": "judge",
            "provider": "azure",
            "model_name": "jev-deployment",
            "provider_key_id": "pk-1",
        }))
        .unwrap();
        snap.models.insert(ResourceEntry::new("m-1", model, 1));
        let key: aisix_core::ApiKey = serde_json::from_value(serde_json::json!({
            // SHA-256 of `sk-caller`.
            "key_hash": "8b6712790a2089c67aa97a2d80022df18cc65c7814350e33baebe79aab508891",
            "allowed_models": ["*"],
        }))
        .unwrap();
        snap.apikeys.insert(ResourceEntry::new("k-1", key, 1));
        let app = crate::build_router(
            crate::ProxyState::new(
                aisix_core::snapshot::SnapshotHandle::new(snap),
                std::sync::Arc::new(aisix_gateway::Hub::new()),
                &aisix_core::ProxyConfig {
                    addr: "127.0.0.1:0".into(),
                    request_body_limit_bytes: 1_048_576,
                    real_ip: Default::default(),
                    request_id: Default::default(),
                    url_rewrites: Vec::new(),
                    tls: None,
                    listeners: Vec::new(),
                    thread_per_core: None,
                    workers: None,
                },
            )
            .without_cache(),
        );
        let resp = app
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/systemone")
                    .header("authorization", "Bearer sk-caller")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        r#"{"model":"judge","state":"x","questions":{"ok":{"type":"noul","instructions":"OK?"}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);

        let calls = server.received_requests().await.unwrap();
        let call = calls
            .iter()
            .find(|r| r.url.path() == "/providers/microsoft/v1/systemone")
            .expect("the Foundry endpoint was called");
        assert_eq!(
            call.headers.get("authorization").unwrap(),
            "Bearer entra-token"
        );
        assert!(call.headers.get("api-key").is_none());
        let sent: Value = serde_json::from_slice(&call.body).unwrap();
        assert_eq!(sent["model"], "jev-deployment");
    }
}
