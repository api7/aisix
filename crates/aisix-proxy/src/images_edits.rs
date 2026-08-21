//! `POST /v1/images/edits` — multipart image editing (AISIX-Cloud#1360).
//!
//! The editing models (`gpt-image-1`/`gpt-image-2`) take the source
//! image(s), an optional mask, the prompt and every tuning parameter in
//! one `multipart/form-data` body, so this route follows the audio
//! transcription shape — drain the form, resolve `model`, swap in the
//! upstream model id, rebuild the form and forward it — not the JSON
//! bridge dispatch `/v1/images/generations` uses.
//!
//! Flow:
//! 1. [`AuthenticatedKey`] extractor — 401 if auth fails.
//! 2. Drain every multipart field into memory (the snapshot is loaded
//!    only afterwards — #941 audit M2: a multi-second upload must not
//!    pin config resolved before it began).
//! 3. Resolve `model` → Model row → 404; `allowed_models` → 403;
//!    client-IP allowlist → 403.
//! 4. Input guardrails scan every `prompt` field (#545 parity with
//!    generations; the image bytes are not scannable text). Mask-action
//!    PII rules rewrite the prompt fields in place (#932/#696).
//! 5. Key-level + model-level rate limits reserve capacity (#542: after
//!    the guardrail check so a content block doesn't burn an RPM slot).
//! 6. Only the `openai` provider is dispatched — the documented
//!    `/v1/images/edits` route + form shape is OpenAI's (#168 parallel);
//!    anything else is rejected 400 at the gateway boundary.
//! 7. The form is rebuilt verbatim minus the `model` swap — unknown
//!    fields (`n`, `size`, `quality`, `background`, `input_fidelity`,
//!    repeated `image[]` parts, `mask`) forward untouched, so new
//!    upstream parameters need no gateway release.
//! 8. The JSON response relays back; the `usage` token block
//!    (gpt-image-*) feeds the UsageEvent + TPM/TPD commit exactly like
//!    generations (#911 [21]).
//!
//! `stream=true` (partial-image SSE) is rejected 400 for now: buffering
//! a stream the caller asked to watch grow is silent degradation, and
//! the live-relay plumbing (#998) is a follow-up, not a Phase 1 rider.

use aisix_core::AppliedGuardrail;
use aisix_gateway::ChatMessage;
use aisix_obs::{content_capture_cap, CapturedContent};
use axum::body::Bytes;
use axum::extract::{Multipart, State};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::Json;
use reqwest::multipart;
use serde_json::Value;
use std::time::Instant;

use crate::auth::AuthenticatedKey;
use crate::client_ip::ClientContext;
use crate::error::ProxyError;
use crate::state::ProxyState;

const ENDPOINT: &str = "/v1/images/edits";

/// Per-request payload from a successful dispatch — the same shape the
/// generations handler consumes, minus `upstream_called` (this route has
/// no 501 half-success: every `Ok` came back from the upstream).
struct EditsDispatchSuccess {
    response: Response,
    model_name: String,
    provider: String,
    model_id: String,
    provider_key_id: String,
    upstream_model: String,
    applied_guardrails: Vec<AppliedGuardrail>,
    /// `(prompt_tokens, completion_tokens)` from the upstream `usage`
    /// block (gpt-image-*). `None` still emits a zero-token event so the
    /// request is visible + attributed.
    usage: Option<(u32, u32)>,
    redactions: crate::redact::RedactionCounts,
    monitor_hits: Vec<aisix_core::GuardrailMonitorHit>,
    captured_content: Option<CapturedContent>,
}

pub async fn image_edits(
    State(state): State<ProxyState>,
    auth: AuthenticatedKey,
    client: ClientContext,
    multipart: Result<Multipart, axum::extract::multipart::MultipartRejection>,
) -> Response {
    let started = Instant::now();
    let request_id = client.request_id.clone();
    let api_key_id = auth.entry.id.clone();

    // Same silent class as the body-extractor rejections #863 collected: a
    // non-multipart content-type answered axum's bare 400 with no access
    // log, metrics, or envelope.
    let multipart = match multipart {
        Ok(multipart) => multipart,
        Err(_) => {
            return crate::reject::reject_before_dispatch(
                &state,
                "POST",
                ENDPOINT,
                &request_id,
                Some(&api_key_id),
                started,
                crate::reject::Envelope::OpenAi,
                ProxyError::InvalidRequest("invalid multipart form data".into()),
            );
        }
    };

    // Loaded by `dispatch` AFTER the upload is drained, then reused by
    // the emits below (#941) — see the note on audio's multipart_dispatch.
    let mut snapshot = None;

    match dispatch(
        &state,
        &mut snapshot,
        &auth,
        multipart,
        &request_id,
        &client,
    )
    .await
    {
        Ok(success) => {
            let snapshot = snapshot.unwrap_or_else(|| state.snapshot.load());
            let elapsed = started.elapsed();
            let status = success.response.status().as_u16();
            crate::images::emit_access_log(
                ENDPOINT,
                &success.model_name,
                &success.provider,
                &api_key_id,
                status,
                elapsed,
                &request_id,
                None,
            );
            // One ProviderKey lookup for both terminal emits (#941).
            let pk = crate::usage_attr::ResolvedPk::resolve(&snapshot, &success.provider_key_id);
            crate::request_metrics::record(
                &state,
                ENDPOINT,
                crate::request_metrics::Caller::new(&auth),
                crate::request_metrics::Upstream {
                    provider: &success.provider,
                    model: &success.model_name,
                    upstream_model: &success.upstream_model,
                    pk: pk.labels(),
                    ..Default::default()
                },
                status,
                elapsed,
            );
            let (prompt_tokens, completion_tokens) = success.usage.unwrap_or((0, 0));
            crate::images::emit_usage_event(
                &state,
                &snapshot,
                &pk,
                ENDPOINT,
                &request_id,
                &success.model_id,
                &success.model_name,
                &api_key_id,
                &success.provider,
                &success.upstream_model,
                &success.applied_guardrails,
                status,
                elapsed,
                prompt_tokens,
                completion_tokens,
                &client,
                success.redactions.clone(),
                success.monitor_hits.clone(),
                success.captured_content.as_ref(),
            );
            success.response
        }
        Err(err) => {
            // The dispatch can fail before it ever loaded one (a
            // malformed form), so fall back rather than assume.
            let snapshot = snapshot.unwrap_or_else(|| state.snapshot.load());
            let status = err.status().as_u16();
            let elapsed = started.elapsed();
            crate::images::emit_access_log(
                ENDPOINT,
                "unknown",
                "unknown",
                &api_key_id,
                status,
                elapsed,
                &request_id,
                Some(&err),
            );
            // AISIX-Cloud#1325: the form is parsed inside the dispatch that
            // failed, so this branch never sees the model — the request's
            // attribution cell recorded the target it selected.
            let attributed = crate::attribution::current().unwrap_or_default();
            let metric_model =
                crate::request_metrics::LastTarget::requested_model(&snapshot, &attributed);
            let last_target = crate::request_metrics::LastTarget::new(&snapshot, &attributed);
            crate::request_metrics::record(
                &state,
                ENDPOINT,
                crate::request_metrics::Caller::new(&auth),
                last_target.upstream(metric_model.as_ref(), false, false),
                status,
                elapsed,
            );
            // Per #655 parity: surface the failed request in Logs. The
            // attribution cell carries the requested model for every
            // failure past model resolution (a guardrail 422, a provider
            // 400); earlier failures leave it empty — status + error
            // class still identify those.
            crate::usage_attr::emit_error_usage_event(
                &state,
                &snapshot,
                "images",
                "openai",
                &request_id,
                &attributed.requested_model,
                &api_key_id,
                status,
                err.kind(),
                &client,
            );
            err.into_response()
        }
    }
}

/// Collect all multipart fields, resolve the model, swap in the upstream
/// model id, then rebuild and forward the multipart form. See audio's
/// `multipart_dispatch` for the pattern this follows (minus streaming).
async fn dispatch(
    state: &ProxyState,
    // Out-param: the snapshot is loaded HERE, once the upload has been
    // drained, and handed back so the handler's terminal emits read the
    // same one (#941 audit M2).
    snapshot_out: &mut Option<std::sync::Arc<aisix_core::AisixSnapshot>>,
    auth: &AuthenticatedKey,
    mut multipart: Multipart,
    request_id: &str,
    client_ctx: &ClientContext,
) -> Result<EditsDispatchSuccess, ProxyError> {
    // Collect all fields first so we can find `model` before building the
    // outgoing reqwest multipart.
    let mut fields: Vec<(String, Option<String>, Option<String>, Bytes)> = Vec::new();

    while let Some(field) = multipart.next_field().await.map_err(|e| {
        crate::error::proxy_error_from_multipart(
            e,
            state.request_body_limit_bytes,
            "multipart read error",
        )
    })? {
        let name = field.name().unwrap_or("").to_string();
        let file_name = field.file_name().map(|s| s.to_string());
        let content_type = field.content_type().map(|s| s.to_string());
        let data = field.bytes().await.map_err(|e| {
            crate::error::proxy_error_from_multipart(
                e,
                state.request_body_limit_bytes,
                "multipart field read error",
            )
        })?;
        fields.push((name, file_name, content_type, data));
    }

    let model_name = fields
        .iter()
        .find(|(name, ..)| name == "model")
        .and_then(|(.., data)| std::str::from_utf8(data).ok())
        .map(|s| s.trim().to_string())
        .ok_or_else(|| ProxyError::InvalidRequest("`model` field missing from form".into()))?;

    let snapshot = &**snapshot_out.insert(state.snapshot.load());
    let model_entry = crate::model_resolve::resolve_model(snapshot, &model_name)
        .ok_or_else(|| ProxyError::ModelNotFound(model_name.clone()))?;

    if !auth.key().can_access(&model_name) {
        return Err(ProxyError::ModelForbidden(model_name.clone()));
    }

    // Client-IP allowlist gate (#557): reject before guardrails / upstream.
    crate::dispatch::check_ip_access(&model_entry.value, &client_ctx.source_ip)?;

    // Partial-image SSE (`stream=true`) is not relayed yet — reject
    // rather than buffer a stream the caller asked to watch grow (see
    // the module doc). Rejected AFTER model resolution so an unknown
    // model still answers 404 (matching the JSON endpoints' precedence)
    // and the error carries the attribution `resolve_model` recorded,
    // BEFORE any quota reservation.
    if fields.iter().any(|(name, _, _, data)| {
        name == "stream" && std::str::from_utf8(data).map(str::trim) == Ok("true")
    }) {
        return Err(ProxyError::InvalidRequest(
            "`stream` is not supported on /v1/images/edits".into(),
        ));
    }

    // #545 parity with generations: the `prompt` form field is caller text
    // forwarded verbatim to the provider — scan it (input hook, before the
    // reservation per #542) and mask it (#932). The response is an image,
    // not scannable text, so there is no output hook.
    let guardrail_ctx = aisix_guardrails::RequestContext {
        passthrough_route_id: "",
        model_id: &model_entry.id,
        mcp_server_id: "",
        api_key_id: &auth.entry.id,
        team_id: auth.key().team_id.as_deref(),
    };
    let resolved_chain = state.guardrail_index.resolve(&guardrail_ctx);
    let applied_guardrails = resolved_chain.applied().to_vec();
    let mut redactions = crate::redact::RedactionCounts::new();
    let mut monitor_hits: Vec<aisix_core::GuardrailMonitorHit> = Vec::new();
    if !resolved_chain.is_empty() {
        // EVERY `prompt` field: multipart allows repeated names and the form
        // is rebuilt with all of them, so all are scanned — an empty first
        // field must not skip a later one.
        let prompt_messages: Vec<ChatMessage> = fields
            .iter()
            .filter(|(name, ..)| name == "prompt")
            .filter_map(|(.., data)| std::str::from_utf8(data).ok())
            .filter(|s| !s.is_empty())
            .map(|s| ChatMessage::user(s.to_string()))
            .collect();
        if !prompt_messages.is_empty() {
            let chat = aisix_gateway::ChatFormat::new(&model_name, prompt_messages);
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
                    model = %model_name,
                    reason = %reason,
                    "guardrail blocked /v1/images/edits request (prompt field)",
                );
                return Err(ProxyError::ContentFiltered(
                    crate::error::guardrail_block_message("request", guardrail_name.as_deref()),
                ));
            }
        }
        if aisix_guardrails::Guardrail::redacts_input(&resolved_chain) {
            for (name, _, _, data) in fields.iter_mut() {
                if name != "prompt" {
                    continue;
                }
                if let Ok(text) = std::str::from_utf8(data) {
                    if let Some(r) =
                        aisix_guardrails::Guardrail::redact_input_text(&resolved_chain, text)
                    {
                        *data = Bytes::from(r.text.into_bytes());
                        crate::redact::merge_counts(&mut redactions, r.counts);
                    }
                }
            }
        }
    }

    // Content capture (#700): the image/mask bytes are NOT captured — a
    // binary field is represented by its sha256, text fields verbatim
    // POST-redaction (the audio convention); the response side is the
    // full image JSON (the generations convention).
    let content_cap = content_capture_cap(
        snapshot
            .observability_exporters
            .entries()
            .iter()
            .map(|e| &e.value),
    );
    let captured_prompt = content_cap.map(|_| {
        use sha2::Digest;
        let mut obj = serde_json::Map::new();
        // Appends on a repeated name (multipart allows repeats and all are
        // forwarded) so no field disappears from the export. The filename is
        // deliberately NOT captured — it is user-controlled text that skips
        // the redaction path; the checksum alone represents the file.
        let mut push = |key: String, value: String| match obj.get_mut(&key) {
            Some(Value::String(existing)) => {
                existing.push('\n');
                existing.push_str(&value);
            }
            _ => {
                obj.insert(key, Value::String(value));
            }
        };
        for (name, _, _, data) in &fields {
            // `image` / `mask` are the file slots; represent them by
            // checksum even when the bytes happen to be valid UTF-8 (an
            // SVG source, say) — the capture is an audit trail, not an
            // asset store. Name-based, not content-type-based: browsers
            // and SDKs stamp `text/plain` on ordinary text fields, and
            // hashing a `prompt` for that would erase the very text the
            // capture exists to audit (same rule as audio's `file`).
            let is_binary_slot = name == "image" || name == "mask";
            match std::str::from_utf8(data) {
                Ok(text) if !is_binary_slot => {
                    push(name.clone(), text.to_string());
                }
                _ => {
                    push(
                        format!("{name}_sha256"),
                        format!("{:x}", sha2::Sha256::digest(data)),
                    );
                }
            }
        }
        serde_json::to_string(&Value::Object(obj)).unwrap_or_default()
    });

    let model_rl =
        crate::quota::ModelRateLimit::from_model(&model_name, &model_entry.id, &model_entry.value);
    let reservation = crate::quota::enforce(state, snapshot, auth, Some(&model_rl)).await?;

    let model = &model_entry.value;

    // Per #168's reasoning on generations: only OpenAI's API documents
    // the `/v1/images/edits` route + form shape. Routing another provider
    // here would dispatch to an upstream that 404s — reject explicitly at
    // the gateway boundary instead. Cross-provider editing wires
    // (Gemini / Vertex / BFL) are the AISIX-Cloud#1360 Phase 2 follow-up.
    if model.provider.as_deref() != Some("openai") {
        reservation.commit_tokens(0).await;
        return Err(ProxyError::InvalidRequest(format!(
            "model `{model_name}` is not an OpenAI provider; \
             /v1/images/edits requires OpenAI"
        )));
    }

    let provider = crate::dispatch::require_provider(model)?.to_string();
    let upstream_model = crate::dispatch::require_upstream_model(model)?.to_string();
    let pk_entry = crate::dispatch::resolve_provider_key(snapshot, model)?;
    let api_key = crate::dispatch::require_api_key(&pk_entry.value, model)?;

    let url = aisix_gateway::url_cache::cached_endpoint_url(
        &pk_entry.id,
        "proxy/images/edits",
        // Every resolve_base_url input, via the shared constructor
        // (#1017), plus the endpoint path.
        &{
            let [base, vendor] = crate::dispatch::pk_url_fingerprint(&pk_entry.value);
            [base, vendor, "/images/edits"]
        },
        || {
            let base = crate::dispatch::resolve_base_url(&pk_entry.value)?;
            Ok::<_, crate::error::ProxyError>(crate::dispatch::build_openai_url(
                &base,
                "/images/edits",
            ))
        },
    )?;
    let provider_label = provider.to_ascii_lowercase();

    // Rebuild the multipart form with `model` rewritten. A `multipart::Form`
    // is single-use (sending consumes it), so this is a closure: each retry
    // attempt below builds a fresh one. That is only possible because every
    // part is `Part::bytes` over an in-memory `Bytes`.
    let build_form = || {
        let mut form = multipart::Form::new();
        for (name, file_name, content_type, data) in &fields {
            let field_data = if name == "model" {
                Bytes::copy_from_slice(upstream_model.as_bytes())
            } else {
                data.clone()
            };

            let data_vec = field_data.to_vec();
            let mut part = if let Some(ct) = content_type {
                multipart::Part::bytes(data_vec.clone())
                    .mime_str(ct)
                    .unwrap_or_else(|_| multipart::Part::bytes(data_vec))
            } else {
                multipart::Part::bytes(data_vec)
            };
            if let Some(fname) = file_name {
                part = part.file_name(fname.clone());
            }
            form = form.part(name.clone(), part);
        }
        form
    };

    // Headers built explicitly so the PK's `request.default_headers` and
    // `request.forward_client_headers` apply (AISIX-Cloud#867). The body is
    // a multipart form, so JSON body-field overrides don't apply — only
    // headers do. Content-Type is left to `.multipart()` (it sets the
    // boundary). Reserved auth headers are protected by
    // `apply_request_headers`.
    let mut headers = axum::http::HeaderMap::new();
    let auth_hv = header::HeaderValue::from_str(&format!("Bearer {api_key}")).map_err(|e| {
        ProxyError::Bridge(aisix_gateway::BridgeError::Config(format!(
            "api key contains invalid header chars: {e}"
        )))
    })?;
    headers.insert(header::AUTHORIZATION, auth_hv);
    let rid_hv = header::HeaderValue::from_str(request_id).map_err(|e| {
        ProxyError::Bridge(aisix_gateway::BridgeError::Config(format!(
            "request_id contains invalid header chars: {e}"
        )))
    })?;
    headers.insert(
        header::HeaderName::from_static("x-aisix-request-id"),
        rid_hv,
    );
    aisix_gateway::apply_request_headers(
        &mut headers,
        &crate::dispatch::upstream_header_ctx(
            &pk_entry.value,
            &pk_entry.id,
            model,
            &model_entry.id,
            client_ctx,
        ),
    );

    let client = crate::http_client::client_for(pk_entry.value.tls.as_ref());
    let tracker = &state.runtime_status;
    let model_id: &str = &model_entry.id;
    let cooldown_cfg = model.cooldown.as_ref();
    // #554/#911: the per-model E2E request timeout bounds the whole
    // buffered exchange, like the other direct-upstream paths.
    let request_budget =
        crate::routing::effective_timeouts(model, None, state.default_timeouts).request;
    let body_bytes = match crate::routing::retrying_dispatch(state, model, ENDPOINT, || {
        let mut req = url
            .clone()
            .post_on(&client)
            .headers(headers.clone())
            .multipart(build_form());
        if let Some(d) = request_budget {
            req = req.timeout(d);
        }
        async move {
            // `reqwest_error_to_bridge`: an elapsed `timeout` must surface
            // as `BridgeError::Timeout`, not transport — the distinction
            // decides whether the default retry budget covers it.
            let send_started = Instant::now();
            let resp = req.send().await.map_err(|e| {
                crate::cooldown::note_failure(
                    tracker,
                    model_id,
                    cooldown_cfg,
                    crate::dispatch::reqwest_error_to_bridge(&e, send_started),
                )
            })?;
            let status = resp.status();
            if !status.is_success() {
                let s = status.as_u16();
                let retry_after = aisix_gateway::parse_retry_after(resp.headers());
                let msg = resp.text().await.unwrap_or_default();
                return Err(crate::cooldown::note_failure(
                    tracker,
                    model_id,
                    cooldown_cfg,
                    aisix_gateway::BridgeError::upstream_status_with_retry_after(
                        s,
                        msg.chars().take(1024).collect::<String>(),
                        retry_after,
                    ),
                ));
            }
            resp.bytes().await.map_err(|e| {
                crate::cooldown::note_failure(
                    tracker,
                    model_id,
                    cooldown_cfg,
                    aisix_gateway::BridgeError::UpstreamDecode(e.to_string()),
                )
            })
        }
    })
    .await
    {
        Ok(v) => v,
        Err(err) => {
            reservation.commit_tokens(0).await;
            return Err(ProxyError::Bridge(err));
        }
    };

    // The edits response is a JSON object (`{created, data, usage?}`) on
    // every documented success; a body that doesn't parse is an upstream
    // defect surfaced as 502 rather than relayed as ambiguous bytes.
    // Parsed BEFORE the health marks below, so a 2xx-with-garbage answer
    // doesn't record the model healthy on a request the caller sees fail.
    let resp_json: Value = serde_json::from_slice(&body_bytes).map_err(|e| {
        ProxyError::Bridge(aisix_gateway::BridgeError::UpstreamDecode(format!(
            "image edits response is not JSON: {e}"
        )))
    })?;

    state.health.record_success(&model.display_name);
    state.runtime_status.mark_healthy(&model_entry.id);

    // #911 [21]: commit the actual token cost so TPM/TPD is enforced.
    let usage = crate::images::extract_token_usage(&resp_json);
    let total_tokens = usage
        .map(|(prompt, completion)| u64::from(prompt) + u64::from(completion))
        .unwrap_or(0);
    reservation.commit_tokens(total_tokens).await;

    let captured_content = match (&captured_prompt, content_cap) {
        (Some(prompt), Some(cap)) => Some(CapturedContent::new(
            prompt,
            &serde_json::to_string(&resp_json).unwrap_or_default(),
            cap as usize,
        )),
        _ => None,
    };

    Ok(EditsDispatchSuccess {
        response: Json(resp_json).into_response(),
        model_name,
        provider: provider_label,
        model_id: model_entry.id.to_string(),
        provider_key_id: pk_entry.id.to_string(),
        upstream_model,
        applied_guardrails,
        usage,
        redactions,
        monitor_hits,
        captured_content,
    })
}

#[cfg(test)]
mod tests {
    use aisix_core::resource::ResourceEntry;
    use aisix_core::snapshot::SnapshotHandle;
    use aisix_core::{AisixSnapshot, ApiKey, Model, ProxyConfig};
    use aisix_gateway::Hub;
    use aisix_provider_openai::OpenAiBridge;
    use axum::body::to_bytes;
    use axum::http::{Request, StatusCode};
    use std::sync::Arc;
    use wiremock::matchers::{method, path};
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

    fn model_entry(name: &str) -> ResourceEntry<Model> {
        let json = format!(
            r#"{{
                "display_name": "{name}",
                "provider": "openai",
                "model_name": "gpt-image-2",
                "provider_key_id": "{PK_ID}"
            }}"#
        );
        let m: Model = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("m-1", m, 1)
    }

    fn anthropic_model_entry(name: &str) -> ResourceEntry<Model> {
        let json = format!(
            r#"{{
                "display_name": "{name}",
                "provider": "anthropic",
                "model_name": "claude-3-5-haiku-20241022",
                "provider_key_id": "{PK_ID}"
            }}"#
        );
        let m: Model = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("m-1", m, 1)
    }

    fn provider_key_entry(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        let json = format!(
            r#"{{"display_name":"openai-up","secret":"sk-up","api_base":"{api_base}","provider":"openai","adapter":"openai"}}"#
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
            r#"{{"key_hash": "8b6712790a2089c67aa97a2d80022df18cc65c7814350e33baebe79aab508891", "allowed_models": {}}}"#,
            serde_json::to_string(&allowed).unwrap()
        );
        let k: ApiKey = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("k-1", k, 1)
    }

    fn build_app(snap: AisixSnapshot) -> axum::Router {
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        crate::build_router(crate::ProxyState::new(handle, hub, &cfg()).without_cache())
    }

    fn build_app_with_sink(
        snap: AisixSnapshot,
        tx: tokio::sync::mpsc::Sender<aisix_obs::UsageEvent>,
    ) -> axum::Router {
        use aisix_obs::UsageSink;
        let hub = Arc::new(Hub::new());
        hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
        let handle = SnapshotHandle::new(snap);
        let state = crate::ProxyState::new(handle, hub, &cfg())
            .without_cache()
            .with_usage_sink(UsageSink::new(tx));
        crate::build_router(state)
    }

    fn keyword_input_guardrail(literal: &str) -> ResourceEntry<aisix_core::Guardrail> {
        let json = format!(
            r#"{{"name":"t","enabled":true,"hook_point":"input","fail_open":false,"kind":"keyword","patterns":[{{"kind":"literal","value":"{literal}"}}]}}"#
        );
        let g: aisix_core::Guardrail = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("g-1", g, 1)
    }

    /// A minimal edits form: `model`, `prompt`, and a fake PNG `image`
    /// file part — the shape the OpenAI SDK sends.
    fn edits_multipart(model: &str, prompt: &str) -> (String, axum::body::Body) {
        let body = format!(
            "--b\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}\r\n\
             --b\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\n{prompt}\r\n\
             --b\r\nContent-Disposition: form-data; name=\"size\"\r\n\r\n1024x1024\r\n\
             --b\r\nContent-Disposition: form-data; name=\"image\"; filename=\"a.png\"\r\n\
             Content-Type: image/png\r\n\r\nPNGFAKEBYTES\r\n--b--\r\n"
        );
        (
            "multipart/form-data; boundary=b".to_string(),
            axum::body::Body::from(body),
        )
    }

    fn make_req(model: &str, prompt: &str) -> Request<axum::body::Body> {
        let (ct, body) = edits_multipart(model, prompt);
        Request::builder()
            .method("POST")
            .uri("/v1/images/edits")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", ct)
            .body(body)
            .unwrap()
    }

    fn upstream_response() -> serde_json::Value {
        serde_json::json!({
            "created": 1_700_000_000i64,
            "data": [{"b64_json": "aGVsbG8="}],
            "usage": {
                "input_tokens": 50,
                "output_tokens": 1056,
                "total_tokens": 1106
            }
        })
    }

    /// Happy path: the form forwards with the alias swapped for the
    /// upstream model id, image bytes and unknown fields intact, and the
    /// upstream JSON relays back.
    #[tokio::test]
    async fn happy_path_rebuilds_form_and_relays_json() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/images/edits"))
            .and(wiremock::matchers::body_string_contains("gpt-image-2"))
            .and(wiremock::matchers::body_string_contains("PNGFAKEBYTES"))
            .and(wiremock::matchers::body_string_contains("1024x1024"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_response()))
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("img-edit-prod"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let resp = tower::ServiceExt::oneshot(app, make_req("img-edit-prod", "add a hat"))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(v["data"][0]["b64_json"].is_string());
    }

    /// The headline rebuild contract: repeated `image` parts and the
    /// `mask` part all survive the drain → rebuild round-trip with their
    /// bytes and filenames intact. A rebuild that dedups by field name
    /// (the shape a map-keyed rebuild would take) or drops the mask
    /// fails here.
    #[tokio::test]
    async fn repeated_image_and_mask_parts_forward_intact() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/images/edits"))
            .and(wiremock::matchers::body_string_contains("IMAGE-ONE-BYTES"))
            .and(wiremock::matchers::body_string_contains("IMAGE-TWO-BYTES"))
            .and(wiremock::matchers::body_string_contains("MASK-BYTES"))
            .and(wiremock::matchers::body_string_contains("first.png"))
            .and(wiremock::matchers::body_string_contains("second.png"))
            .and(wiremock::matchers::body_string_contains("mask.png"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_response()))
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("img-edit-prod"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let body = "--b\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nimg-edit-prod\r\n\
             --b\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\ncombine them\r\n\
             --b\r\nContent-Disposition: form-data; name=\"image\"; filename=\"first.png\"\r\n\
             Content-Type: image/png\r\n\r\nIMAGE-ONE-BYTES\r\n\
             --b\r\nContent-Disposition: form-data; name=\"image\"; filename=\"second.png\"\r\n\
             Content-Type: image/png\r\n\r\nIMAGE-TWO-BYTES\r\n\
             --b\r\nContent-Disposition: form-data; name=\"mask\"; filename=\"mask.png\"\r\n\
             Content-Type: image/png\r\n\r\nMASK-BYTES\r\n--b--\r\n";
        let req = Request::builder()
            .method("POST")
            .uri("/v1/images/edits")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "multipart/form-data; boundary=b")
            .body(axum::body::Body::from(body))
            .unwrap();

        let app = build_app(snap);
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    /// The alias must NOT reach the upstream — the form's `model` field
    /// is rewritten to the Model row's upstream name.
    #[tokio::test]
    async fn alias_never_reaches_upstream() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/images/edits"))
            .and(wiremock::matchers::body_string_contains("img-edit-prod"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_response()))
            .expect(0)
            .mount(&upstream)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/images/edits"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_response()))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("img-edit-prod"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let resp = tower::ServiceExt::oneshot(app, make_req("img-edit-prod", "add a hat"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    /// gpt-image usage block → UsageEvent with those tokens (#407 parity).
    #[tokio::test]
    async fn emits_usage_event_with_tokens() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/images/edits"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_response()))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("img-edit-prod"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_app_with_sink(snap, tx);
        let resp = tower::ServiceExt::oneshot(app, make_req("img-edit-prod", "add a hat"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let ev = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("UsageEvent must be emitted")
            .expect("usage_sink sender dropped");
        assert_eq!(ev.prompt_tokens, 50);
        assert_eq!(ev.completion_tokens, 1056);
        assert_eq!(ev.requested_model, "img-edit-prod");
        assert_eq!(ev.inbound_protocol, "openai");
    }

    /// #545 parity: a configured input guardrail fires on the `prompt`
    /// form field — 422 content_filter, upstream never contacted.
    #[tokio::test]
    async fn input_guardrail_blocks_prompt_returns_422() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/images/edits"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upstream_response()))
            .expect(0)
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("img-edit-prod"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_input_guardrail("BLOCKME"));

        let app = build_app(snap);
        let resp = tower::ServiceExt::oneshot(app, make_req("img-edit-prod", "please BLOCKME now"))
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");
        assert!(!v["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("BLOCKME"));
    }

    /// #168 parallel: a non-OpenAI Model is rejected 400 at the gateway
    /// boundary rather than dispatched to an upstream that would 404.
    #[tokio::test]
    async fn non_openai_provider_returns_400_invalid_request() {
        let snap = new_snap("https://api.anthropic.com");
        snap.models.insert(anthropic_model_entry("claude-img"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let resp = tower::ServiceExt::oneshot(app, make_req("claude-img", "add a hat"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert!(v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("requires OpenAI"));
    }

    /// `stream=true` (partial-image SSE) is not relayed yet — explicit
    /// 400, not a silently buffered stream.
    #[tokio::test]
    async fn stream_true_rejected_400() {
        let snap = new_snap("http://unused");
        snap.models.insert(model_entry("img-edit-prod"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let body = "--b\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nimg-edit-prod\r\n\
             --b\r\nContent-Disposition: form-data; name=\"stream\"\r\n\r\ntrue\r\n\
             --b\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nhi\r\n--b--\r\n";
        let req = Request::builder()
            .method("POST")
            .uri("/v1/images/edits")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "multipart/form-data; boundary=b")
            .body(axum::body::Body::from(body))
            .unwrap();

        let app = build_app(snap);
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(v["error"]["message"].as_str().unwrap().contains("stream"));
    }

    #[tokio::test]
    async fn missing_model_field_returns_400() {
        let snap = new_snap("http://unused");
        snap.apikeys.insert(apikey_entry(&["*"]));

        let body = "--b\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nhi\r\n--b--\r\n";
        let req = Request::builder()
            .method("POST")
            .uri("/v1/images/edits")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "multipart/form-data; boundary=b")
            .body(axum::body::Body::from(body))
            .unwrap();

        let app = build_app(snap);
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn non_multipart_content_type_returns_envelope_400() {
        let snap = new_snap("http://unused");
        snap.models.insert(model_entry("img-edit-prod"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let req = Request::builder()
            .method("POST")
            .uri("/v1/images/edits")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(r#"{"model":"img-edit-prod"}"#))
            .unwrap();

        let app = build_app(snap);
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn unauthenticated_returns_401() {
        let snap = new_snap("http://unused");
        snap.models.insert(model_entry("img-edit-prod"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (ct, body) = edits_multipart("img-edit-prod", "hi");
        let req = Request::builder()
            .method("POST")
            .uri("/v1/images/edits")
            .header("content-type", ct)
            .body(body)
            .unwrap();
        let app = build_app(snap);
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn forbidden_model_returns_403() {
        let snap = new_snap("http://unused");
        snap.models.insert(model_entry("img-edit-prod"));
        snap.apikeys.insert(apikey_entry(&["other-model"]));

        let app = build_app(snap);
        let resp = tower::ServiceExt::oneshot(app, make_req("img-edit-prod", "hi"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn unknown_model_returns_404() {
        let snap = new_snap("http://unused");
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let resp = tower::ServiceExt::oneshot(app, make_req("nope", "hi"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn upstream_error_propagates_as_502() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/images/edits"))
            .respond_with(ResponseTemplate::new(500).set_body_string("server error"))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(model_entry("img-edit-prod"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let resp = tower::ServiceExt::oneshot(app, make_req("img-edit-prod", "hi"))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
    }
}
