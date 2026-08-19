//! `POST /v1/audio/{transcriptions,translations,speech}` — audio API
//! pass-through.
//!
//! Three sub-endpoints with different request shapes:
//!
//! * **transcriptions** & **translations** — `multipart/form-data` with an
//!   audio `file`, a `model` field, and optional metadata fields.
//!   The gateway resolves the model name, swaps in the upstream model id,
//!   and re-assembles the multipart form before forwarding.
//!
//! * **speech** — JSON body `{model, input, voice, …}`.
//!   Standard JSON passthrough, identical to `/v1/completions`.
//!
//! In all cases the upstream response is returned verbatim: JSON for
//! transcription/translation results, binary audio bytes for speech.
//!
//! Auth and model authorisation follow the same rules as every other
//! proxy endpoint.

use aisix_core::AppliedGuardrail;
use aisix_gateway::{ChatMessage, ChatResponse, FinishReason, UsageStats};
use aisix_obs::{content_capture_cap, AccessLog, CapturedContent, UsageEvent};
use axum::body::Bytes;
use axum::extract::{Multipart, State};
use axum::http::{header, HeaderMap};
use axum::response::{IntoResponse, Response};
use axum::Json;
use reqwest::multipart;
use serde_json::Value;
use std::time::{Duration, Instant};

use crate::auth::AuthenticatedKey;
use crate::client_ip::ClientContext;
use crate::error::ProxyError;
use crate::state::ProxyState;

/// Per-request payload from a successful multipart dispatch
/// (transcriptions/translations) — adds `model_id` + parsed `usage` to
/// the response/model/provider triplet so the handler can emit a
/// UsageEvent (#406).
struct AudioDispatchSuccess {
    response: Response,
    model_name: String,
    provider: String,
    model_id: String,
    /// Resolved ProviderKey UUID — feeds the per-PK telemetry attribution
    /// tags on the emitted UsageEvent (AISIX-Cloud#867 parity).
    provider_key_id: String,
    /// Provider-side model name, for the `upstream_model` metric label
    /// (AISIX-Cloud#1234 parity with chat / messages / responses).
    upstream_model: String,
    /// `(prompt_tokens, completion_tokens)` from the upstream `usage`
    /// block when the model returns one (gpt-4o-transcribe). `None` for
    /// whisper-1 (no usage block) — those still emit a zero-token event
    /// so the request is visible + attributed.
    usage: Option<(u32, u32)>,
    /// Set on the streamed-transcription relay (#998): the SSE frames are
    /// consumed by the caller after this handler returns, so the terminal
    /// event's counts are only known inside the stream — its Drop guard
    /// owns the UsageEvent and `usage` stays `None` here. Guards the
    /// handler against a second, zero-token emit.
    usage_handled_by_stream: bool,
    /// Audio length in seconds — the cost basis for the duration-billed
    /// models (whisper-1), which report no tokens at all (#457).
    duration_seconds: f64,
    /// The `{kind, hook}` set of guardrails that governed this request
    /// (#379 parity, wired with #696) — surfaced on the emitted UsageEvent.
    applied_guardrails: Vec<AppliedGuardrail>,
    /// Per-detector PII mask counts (#932/#696): the multipart `prompt`
    /// field (input side) + the transcript (output side), merged. Attached
    /// to the emitted UsageEvent. Empty = no redaction.
    redactions: crate::redact::RedactionCounts,
    /// Monitor-mode guardrail observations (AISIX-Cloud#562), input +
    /// output merged.
    monitor_hits: Vec<aisix_core::GuardrailMonitorHit>,
    /// #696: set when an OUTPUT guardrail blocked the transcript AFTER the
    /// upstream billed for it. The response body is the redacted 422, but
    /// `usage` keeps the billed counts so the UsageEvent (marked
    /// `guardrail_blocked`) doesn't under-report spend — same convention as
    /// completions #911 [23] / responses #543.
    guardrail_blocked: bool,
    /// Captured request/response content for content-capturing exporters
    /// (#700, LiteLLM parity: the audio bytes are represented by their
    /// sha256, text form fields verbatim; the response is the post-redaction
    /// transcript). `Some` only when an exporter opted into
    /// `content_mode = full`.
    captured_content: Option<CapturedContent>,
}

// ─────────────────────────────────────────────────────────────────────────────
// /v1/audio/transcriptions
// ─────────────────────────────────────────────────────────────────────────────

pub async fn transcriptions(
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
                "/v1/audio/transcriptions",
                &request_id,
                Some(&api_key_id),
                started,
                crate::reject::Envelope::OpenAi,
                ProxyError::InvalidRequest("invalid multipart form data".into()),
            );
        }
    };

    // Loaded by `multipart_dispatch` AFTER the upload is drained, then
    // reused by the emits below (#941) — see the note on its signature.
    let mut snapshot = None;

    match multipart_dispatch(
        &state,
        &mut snapshot,
        &auth,
        multipart,
        // Version-independent path — multipart_dispatch's URL builder
        // appends it to the configured api_base.
        "/audio/transcriptions",
        &request_id,
        &client,
    )
    .await
    {
        Ok(success) => {
            // `multipart_dispatch` loaded it once the upload was drained.
            let snapshot = snapshot.unwrap_or_else(|| state.snapshot.load());
            let elapsed = started.elapsed();
            // Actual status, not a hardcoded 200 — the #696 billed-then-
            // output-blocked path returns Ok(success) carrying a 422.
            let status = success.response.status().as_u16();
            emit_access_log(
                "POST",
                "/v1/audio/transcriptions",
                &success.model_name,
                &success.provider,
                &api_key_id,
                status,
                elapsed,
                &request_id,
                None,
            );
            // ONE ProviderKey lookup for both terminal emits (#941).
            let pk = crate::usage_attr::ResolvedPk::resolve(&snapshot, &success.provider_key_id);
            record_audio_metrics(
                &state,
                &pk,
                "/v1/audio/transcriptions",
                &auth,
                &success,
                status,
                elapsed,
            );
            // #998: the streamed relay's Drop guard emits the event once
            // the terminal `transcript.text.done` has been parsed off the
            // wire; emitting here too would double-count the request with
            // zero tokens.
            if !success.usage_handled_by_stream {
                emit_audio_usage(
                    &state,
                    &snapshot,
                    &pk,
                    &request_id,
                    "/v1/audio/transcriptions",
                    &success,
                    &api_key_id,
                    status,
                    elapsed,
                    &client,
                );
            }
            success.response
        }
        Err(err) => {
            // The dispatch can fail before it ever loaded one (a malformed
            // form), so fall back rather than assume.
            let snapshot = snapshot.unwrap_or_else(|| state.snapshot.load());
            let status = err.status().as_u16();
            let elapsed = started.elapsed();
            emit_access_log(
                "POST",
                "/v1/audio/transcriptions",
                "unknown",
                "unknown",
                &api_key_id,
                status,
                elapsed,
                &request_id,
                Some(&err),
            );
            // AISIX-Cloud#1325: the multipart form is parsed inside the
            // dispatch that failed, so this branch never sees the model —
            // but the request's attribution cell recorded it along with the
            // target it selected, so an upstream failure here is attributed
            // exactly like on the JSON endpoints instead of landing wholly
            // on `unknown`.
            let attributed = crate::attribution::current().unwrap_or_default();
            let metric_model =
                crate::request_metrics::LastTarget::requested_model(&snapshot, &attributed);
            let last_target = crate::request_metrics::LastTarget::new(&snapshot, &attributed);
            crate::request_metrics::record(
                &state,
                "/v1/audio/transcriptions",
                crate::request_metrics::Caller::new(&auth),
                last_target.upstream(metric_model.as_ref(), false, false),
                status,
                elapsed,
            );
            // Per #655 parity: surface the failed request in Logs. The model
            // isn't extracted from the multipart form on this error path, so
            // requested_model is empty; status + error class still identify it.
            crate::usage_attr::emit_error_usage_event(
                &state,
                &snapshot,
                "audio",
                "openai",
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

// ─────────────────────────────────────────────────────────────────────────────
// /v1/audio/translations
// ─────────────────────────────────────────────────────────────────────────────

pub async fn translations(
    State(state): State<ProxyState>,
    auth: AuthenticatedKey,
    client: ClientContext,
    multipart: Result<Multipart, axum::extract::multipart::MultipartRejection>,
) -> Response {
    let started = Instant::now();
    let request_id = client.request_id.clone();
    let api_key_id = auth.entry.id.clone();

    // See `transcriptions`: the rejection is recorded, not silently bare.
    let multipart = match multipart {
        Ok(multipart) => multipart,
        Err(_) => {
            return crate::reject::reject_before_dispatch(
                &state,
                "POST",
                "/v1/audio/translations",
                &request_id,
                Some(&api_key_id),
                started,
                crate::reject::Envelope::OpenAi,
                ProxyError::InvalidRequest("invalid multipart form data".into()),
            );
        }
    };

    // Loaded by `multipart_dispatch` AFTER the upload is drained, then
    // reused by the emits below (#941) — see the note on its signature.
    let mut snapshot = None;

    match multipart_dispatch(
        &state,
        &mut snapshot,
        &auth,
        multipart,
        // Version-independent path — multipart_dispatch's URL builder
        // appends it to the configured api_base.
        "/audio/translations",
        &request_id,
        &client,
    )
    .await
    {
        Ok(success) => {
            // `multipart_dispatch` loaded it once the upload was drained.
            let snapshot = snapshot.unwrap_or_else(|| state.snapshot.load());
            let elapsed = started.elapsed();
            // Actual status, not a hardcoded 200 — the #696 billed-then-
            // output-blocked path returns Ok(success) carrying a 422.
            let status = success.response.status().as_u16();
            emit_access_log(
                "POST",
                "/v1/audio/translations",
                &success.model_name,
                &success.provider,
                &api_key_id,
                status,
                elapsed,
                &request_id,
                None,
            );
            // ONE ProviderKey lookup for both terminal emits (#941).
            let pk = crate::usage_attr::ResolvedPk::resolve(&snapshot, &success.provider_key_id);
            record_audio_metrics(
                &state,
                &pk,
                "/v1/audio/translations",
                &auth,
                &success,
                status,
                elapsed,
            );
            // #998: the streamed relay's Drop guard emits the event once
            // the terminal `transcript.text.done` has been parsed off the
            // wire; emitting here too would double-count the request with
            // zero tokens.
            if !success.usage_handled_by_stream {
                emit_audio_usage(
                    &state,
                    &snapshot,
                    &pk,
                    &request_id,
                    "/v1/audio/translations",
                    &success,
                    &api_key_id,
                    status,
                    elapsed,
                    &client,
                );
            }
            success.response
        }
        Err(err) => {
            // The dispatch can fail before it ever loaded one (a malformed
            // form), so fall back rather than assume.
            let snapshot = snapshot.unwrap_or_else(|| state.snapshot.load());
            let status = err.status().as_u16();
            let elapsed = started.elapsed();
            emit_access_log(
                "POST",
                "/v1/audio/translations",
                "unknown",
                "unknown",
                &api_key_id,
                status,
                elapsed,
                &request_id,
                Some(&err),
            );
            // AISIX-Cloud#1325: the multipart form is parsed inside the
            // dispatch that failed, so this branch never sees the model —
            // but the request's attribution cell recorded it along with the
            // target it selected, so an upstream failure here is attributed
            // exactly like on the JSON endpoints instead of landing wholly
            // on `unknown`.
            let attributed = crate::attribution::current().unwrap_or_default();
            let metric_model =
                crate::request_metrics::LastTarget::requested_model(&snapshot, &attributed);
            let last_target = crate::request_metrics::LastTarget::new(&snapshot, &attributed);
            crate::request_metrics::record(
                &state,
                "/v1/audio/translations",
                crate::request_metrics::Caller::new(&auth),
                last_target.upstream(metric_model.as_ref(), false, false),
                status,
                elapsed,
            );
            // Per #655 parity: surface the failed request in Logs (model not
            // extracted on the multipart error path → empty requested_model).
            crate::usage_attr::emit_error_usage_event(
                &state,
                &snapshot,
                "audio",
                "openai",
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

// ─────────────────────────────────────────────────────────────────────────────
// /v1/audio/speech
// ─────────────────────────────────────────────────────────────────────────────

pub async fn speech(
    State(state): State<ProxyState>,
    auth: AuthenticatedKey,
    client: ClientContext,
    // Result-wrapped so an extractor-layer 413 maps to the OpenAI
    // envelope — see completions.rs.
    body: Result<Json<Value>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let started = Instant::now();
    let Json(body) = match body {
        Ok(json) => json,
        // Answer through `reject` — see completions.rs.
        Err(rej) => {
            return crate::reject::reject_before_dispatch(
                &state,
                "POST",
                "/v1/audio/speech",
                &client.request_id,
                Some(&auth.entry.id),
                started,
                crate::reject::Envelope::OpenAi,
                crate::error::proxy_error_from_json_rejection(rej, state.request_body_limit_bytes),
            );
        }
    };
    let request_id = client.request_id.clone();
    let api_key_id = auth.entry.id.clone();
    let model_name = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    // One snapshot for the whole request (#941) — see `embeddings`.
    let snapshot = state.snapshot.load();

    match speech_dispatch(&state, &snapshot, &auth, body, &request_id, &client).await {
        Ok(success) => {
            let elapsed = started.elapsed();
            let status = success.response.status().as_u16();
            emit_access_log(
                "POST",
                "/v1/audio/speech",
                &model_name,
                &success.provider,
                &api_key_id,
                status,
                elapsed,
                &request_id,
                None,
            );
            // One ProviderKey lookup for the metric emit + the usage event
            // below (#941).
            let pk = crate::usage_attr::ResolvedPk::resolve(&snapshot, &success.provider_key_id);
            crate::request_metrics::record(
                &state,
                "/v1/audio/speech",
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
            // Issue #406: /v1/audio/speech (TTS) returns binary audio
            // with no usage block — emit a zero-token UsageEvent so the
            // request is visible in /logs and attributed to the api_key.
            // (TTS is billed per input character; that cost basis is the
            // same cross-repo follow-up as audio duration.)
            emit_usage_event(
                &state,
                &snapshot,
                &pk,
                &request_id,
                &success.model_id,
                &model_name,
                &api_key_id,
                "/v1/audio/speech",
                &success.provider,
                &success.upstream_model,
                &success.applied_guardrails,
                status,
                elapsed,
                0,
                0,
                // TTS is billed per input character, not by the length of
                // the audio it produced — no duration cost basis here.
                0.0,
                &client,
                success.redactions,
                success.monitor_hits,
                /* guardrail_blocked */ false,
                success.captured_content.as_ref(),
            );
            success.response
        }
        Err(err) => {
            let status = err.status().as_u16();
            let elapsed = started.elapsed();
            emit_access_log(
                "POST",
                "/v1/audio/speech",
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
                "/v1/audio/speech",
                crate::request_metrics::Caller::new(&auth),
                last_target.upstream(metric_model.as_ref(), false, false),
                status,
                elapsed,
            );
            // Per #655 parity: surface the failed request in Logs with a
            // zero-token event (status + error class).
            crate::usage_attr::emit_error_usage_event(
                &state,
                &snapshot,
                "audio",
                "openai",
                &request_id,
                &model_name,
                &api_key_id,
                status,
                err.kind(),
                &client,
            );
            err.into_response()
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared dispatch functions
// ─────────────────────────────────────────────────────────────────────────────

/// What one upstream audio call handed back.
///
/// A transcription is normally read whole — the usage block, the duration,
/// the output-guardrail scan and the PII mask all need the body in hand.
/// A `stream=true` request answered with `text/event-stream` is the
/// exception (#998): its frames are relayed to the caller as they arrive,
/// so the response is carried live instead of as bytes.
enum AudioUpstreamBody {
    Buffered(Bytes),
    Live(reqwest::Response),
}

/// Whether an upstream response is an SSE stream.
fn is_event_stream(headers: &HeaderMap) -> bool {
    headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.starts_with("text/event-stream"))
}

/// What the relayed transcription stream observed by the time it ended.
#[derive(Default)]
struct StreamedTranscript {
    /// `(prompt, completion)` off the terminal event's `usage` block —
    /// `None` when the stream ended before one arrived (a client that
    /// disconnected, an upstream that reports no tokens).
    usage: Option<(u32, u32)>,
    /// The transcript assembled from the `transcript.text.delta` events,
    /// capped at the exporter's content-capture limit. Empty when no
    /// exporter asked for content and no output chain needs a scan.
    text: String,
    /// False when the caller disconnected before the upstream ended.
    reached_end: bool,
    /// End-of-stream monitor observations (AISIX-Cloud#1010).
    output_hits: Vec<aisix_core::GuardrailMonitorHit>,
}

/// Fires `on_complete` exactly once with what the relay saw — at
/// end-of-stream AND on client-disconnect, where the generator is simply
/// dropped at its suspension point. Same shape as the `/v1/responses`
/// and chat.rs stream guards: without it a caller that closes the
/// connection on the terminal frame takes the whole UsageEvent with it.
struct TranscriptGuard<F: FnOnce(StreamedTranscript)> {
    slot: Option<(F, StreamedTranscript)>,
}

impl<F: FnOnce(StreamedTranscript)> TranscriptGuard<F> {
    fn observed(&mut self) -> &mut StreamedTranscript {
        &mut self
            .slot
            .as_mut()
            .expect("TranscriptGuard accessed after take")
            .1
    }
}

impl<F: FnOnce(StreamedTranscript)> Drop for TranscriptGuard<F> {
    fn drop(&mut self) {
        if let Some((f, observed)) = self.slot.take() {
            f(observed);
        }
    }
}

/// Relay a streamed transcription verbatim while reading its telemetry off
/// the same bytes (#998).
///
/// The caller gets the upstream's exact SSE wire shape — every frame
/// forwarded unchanged, as it arrives — and a side-channel decoder pulls
/// the `transcript.text.delta` text and the terminal
/// `transcript.text.done` usage block out of the copy. `on_complete` fires
/// once the upstream ends or the caller disconnects, whichever comes
/// first; it owns the UsageEvent for this request.
///
/// `capture_cap` bounds the assembled transcript; an output chain that
/// needs a scan raises the floor to its own scan bound, so neither
/// consumer sees past its own limit.
fn transcription_relay<S, F>(
    upstream: S,
    content_cap: Option<u32>,
    eos_scan: Option<crate::guardrail_stream::EosOutputScan>,
    on_complete: F,
) -> impl futures::Stream<Item = reqwest::Result<Bytes>> + Send
where
    S: futures::Stream<Item = reqwest::Result<Bytes>> + Send + 'static,
    F: FnOnce(StreamedTranscript) + Send + 'static,
{
    use futures::StreamExt as _;

    let text_cap = content_cap
        .map(|cap| cap as usize)
        .unwrap_or(0)
        .max(if eos_scan.is_some() {
            aisix_guardrails::DEFAULT_STREAM_OUTPUT_BUFFER_BYTES
        } else {
            0
        });
    // Re-attach the request span: the body is polled after the request-id
    // middleware returns, so anything logged from here would otherwise lose
    // its `request_id` correlation (AISIX-Cloud#1060).
    crate::request_id::in_request_span(async_stream::stream! {
        let mut guard = TranscriptGuard { slot: Some((on_complete, StreamedTranscript::default())) };
        let mut decoder = aisix_gateway::SseDecoder::new();
        futures::pin_mut!(upstream);
        while let Some(item) = upstream.next().await {
            if let Ok(bytes) = &item {
                // Side-channel parse over a copy of the frames; `item` is
                // yielded below untouched.
                let events = decoder.feed(bytes.as_ref());
                observe_transcript_events(guard.observed(), &events, text_cap);
            }
            yield item;
        }
        observe_transcript_events(guard.observed(), &decoder.finish().into_iter().collect::<Vec<_>>(), text_cap);
        // Upstream EOF — the response was delivered in full. Recorded
        // before the scan below, which awaits a remote provider and is a
        // routine drop point for clients that close on the terminal frame.
        guard.observed().reached_end = true;
        if let Some(scan) = eos_scan {
            // The guard stays armed across the await: an SDK that closes
            // on the terminal frame drops this generator here, and the
            // Drop emit must still carry the usage it already parsed.
            let text = guard.observed().text.clone();
            let hits = scan.observe(&text).await;
            guard.observed().output_hits = hits;
        }
        if let Some((f, observed)) = guard.slot.take() {
            f(observed);
        }
    })
}

/// Fold one batch of decoded SSE events into the running observation:
/// `transcript.text.delta` appends to the transcript (bounded by `cap`),
/// and any event carrying a `usage` block updates the counts — the last
/// one wins, so a provider that reports usage on a later event than
/// `transcript.text.done` is read the same way.
fn observe_transcript_events(
    observed: &mut StreamedTranscript,
    events: &[aisix_gateway::SseEvent],
    cap: usize,
) {
    for event in events {
        let aisix_gateway::SseEvent::Data(payload) = event else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<Value>(payload) else {
            continue;
        };
        if let Some(usage) = extract_token_usage(&value) {
            observed.usage = Some(usage);
        }
        if observed.text.len() >= cap {
            continue;
        }
        if let Some(delta) = value.get("delta").and_then(Value::as_str) {
            let room = cap - observed.text.len();
            let mut end = delta.len().min(room);
            while end > 0 && !delta.is_char_boundary(end) {
                end -= 1;
            }
            observed.text.push_str(&delta[..end]);
        }
    }
}

/// Collect all multipart fields, resolve the model, swap in the upstream
/// model id, then rebuild and forward the multipart form.
async fn multipart_dispatch(
    state: &ProxyState,
    // Out-param, not an input: the snapshot is loaded HERE, once the
    // upload has been drained, and handed back so the handler's terminal
    // emits read the same one. Loading it at handler entry instead would
    // resolve the model, the client-IP allowlist, the upstream credential
    // and the rate-limit policies against config captured before a
    // multi-minute upload began (#941 audit M2).
    snapshot_out: &mut Option<std::sync::Arc<aisix_core::AisixSnapshot>>,
    auth: &AuthenticatedKey,
    mut multipart: Multipart,
    upstream_path: &str,
    request_id: &str,
    client_ctx: &ClientContext,
) -> Result<AudioDispatchSuccess, ProxyError> {
    // The request clock for the streamed relay's end-of-stream emit
    // (#998): the handler has long returned by the time it fires, so it
    // can't read `started` there. Taken before the upload is drained, so
    // it measures the same span the handler's own clock does.
    let dispatch_started = Instant::now();

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

    // Extract the `model` field value.
    let model_name = fields
        .iter()
        .find(|(name, ..)| name == "model")
        .and_then(|(.., data)| std::str::from_utf8(data).ok())
        .map(|s| s.trim().to_string())
        .ok_or_else(|| ProxyError::InvalidRequest("`model` field missing from form".into()))?;

    // `stream=true` asks for the transcript incrementally: the transcribe
    // models answer with `text/event-stream` instead of a JSON object.
    // Known before the request is built because it decides the timeout
    // shape — reqwest's request-level timeout bounds the body read too, so
    // it would cut a relayed stream off mid-transcript (#998).
    let stream_requested = fields.iter().any(|(name, _, _, data)| {
        name == "stream" && std::str::from_utf8(data).map(str::trim) == Ok("true")
    });

    let snapshot = &**snapshot_out.insert(state.snapshot.load());
    let model_entry = crate::model_resolve::resolve_model(snapshot, &model_name)
        .ok_or_else(|| ProxyError::ModelNotFound(model_name.clone()))?;

    if !auth.key().can_access(&model_name) {
        return Err(ProxyError::ModelForbidden(model_name.clone()));
    }

    // Client-IP allowlist gate (#557): reject before quota / upstream.
    crate::dispatch::check_ip_access(&model_entry.value, &client_ctx.source_ip)?;

    // #696: transcriptions/translations run the guardrail chain too. The
    // audio bytes aren't scannable text, but the optional `prompt` form
    // field IS caller text forwarded verbatim to the provider — scan it
    // (input hook, before the reservation per #542) and mask it (#932).
    // The transcript RESPONSE is scanned/masked after the upstream call.
    let guardrail_ctx = aisix_guardrails::RequestContext {
        passthrough_route_id: "",
        model_id: &model_entry.id,
        mcp_server_id: "",
        api_key_id: &auth.entry.id,
        team_id: auth.key().team_id.as_deref(),
    };
    let resolved_chain = state.guardrail_index.resolve(&guardrail_ctx);
    let applied_guardrails = resolved_chain.applied().to_vec();

    // #998: relay the upstream SSE live, or hold it back to scan it?
    // An output guardrail that can block or mask has to see the whole
    // transcript before any of it reaches the caller — otherwise
    // `stream=true` is a bypass for the check the non-streaming request
    // gets — so a hold-back chain keeps the buffered relay, the same
    // secure default the chat / responses surfaces use (#719). A
    // monitor-only chain resolves to `EndOfStreamCheck`: it can never
    // block, so it must NOT change delivery (AISIX-Cloud#1010) and takes
    // the live path, scanning once at end-of-stream.
    let live_relay = stream_requested
        && !(aisix_guardrails::Guardrail::runs_on_output(&resolved_chain)
            && aisix_guardrails::Guardrail::stream_output_policy(&resolved_chain).holds_back());
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
                    "guardrail blocked audio request (prompt field)",
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

    // Content capture (#700, LiteLLM parity): the audio bytes are NOT
    // captured — the file is represented by its sha256 (exactly what LiteLLM
    // logs for transcription input); the text form fields (model, prompt,
    // language, …) are captured verbatim, POST-redaction so a masked
    // `prompt` field stays masked in the exported content.
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
        // the redaction path; the checksum alone represents the file,
        // matching LiteLLM.
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
            match std::str::from_utf8(data) {
                Ok(text) if name != "file" => {
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
    let provider = crate::dispatch::require_provider(model)?;
    let upstream_model = crate::dispatch::require_upstream_model(model)?.to_string();
    let pk_entry = crate::dispatch::resolve_provider_key(snapshot, model)?;
    let api_key = crate::dispatch::require_api_key(&pk_entry.value, model)?;

    // Cache key must be `'static`; both callers pass fixed literals.
    let url_cache_key: &'static str = if upstream_path == "/audio/translations" {
        "proxy/audio/translations"
    } else {
        "proxy/audio/transcriptions"
    };
    let url = aisix_gateway::url_cache::cached_endpoint_url(
        &pk_entry.id,
        url_cache_key,
        &[
            pk_entry.value.api_base.as_deref().unwrap_or(""),
            upstream_path,
        ],
        || {
            let base = crate::dispatch::resolve_base_url(&pk_entry.value)?;
            Ok::<_, crate::error::ProxyError>(crate::dispatch::build_openai_url(
                &base,
                upstream_path,
            ))
        },
    )?;
    let provider_label = provider.to_ascii_lowercase();
    // Static label for retry tracing — this dispatch serves both audio
    // sub-routes, and logging translations under the transcription label
    // would mislead an operator reading retry output.
    let retry_endpoint_label: &'static str = if upstream_path == "/audio/translations" {
        "/v1/audio/translations"
    } else {
        "/v1/audio/transcriptions"
    };

    // Rebuild the multipart form with `model` rewritten. A `multipart::Form`
    // is single-use (sending consumes it), so this is a closure rather than a
    // value: each retry attempt below builds a fresh one. That is only
    // possible because every part is `Part::bytes` over an in-memory `Bytes`
    // — a streamed part could not be replayed.
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

    // Build headers explicitly so the PK's `request.default_headers` and
    // `request.forward_client_headers` can inject operator/client headers
    // (AISIX-Cloud#867 follow-up). The body is a multipart form, so the JSON
    // body-field overrides don't apply here — only headers do. Content-Type
    // is left to `.multipart()` (it sets the boundary). Reserved auth
    // headers are protected by `apply_request_headers`.
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
    // Send, check the status, and read the body as one retryable unit. See
    // the same shape in rerank.rs for why `note_failure` stays per attempt.
    let timeouts = crate::routing::effective_timeouts(model, None, state.default_timeouts);
    let request_budget = timeouts.request;
    let stream_budget = timeouts.stream;
    let (upstream_headers, upstream_body) =
        match crate::routing::retrying_dispatch(state, model, retry_endpoint_label, || {
            let mut req = url
                .clone()
                .post_on(&client)
                .headers(headers.clone())
                .multipart(build_form());
            // #554/#911: a buffered audio call takes the per-model E2E
            // request timeout like the other direct-upstream paths
            // (count_tokens/rerank/responses), so a slow/blackholed audio
            // provider fails over and the model's timeout cooldown can
            // engage. A relayed stream must NOT carry it: reqwest's
            // request-level timeout bounds the body read too, so it would
            // cut the transcript off mid-stream (#998). That path bounds the
            // connect phase and each chunk by the stream budget instead —
            // the same split `/v1/responses` uses.
            if !live_relay {
                if let Some(d) = request_budget {
                    req = req.timeout(d);
                }
            }
            let connect_deadline = if live_relay { stream_budget } else { None };
            async move {
                // `reqwest_error_to_bridge`, not a bare `Transport`: an
                // elapsed `timeout` has to surface as `BridgeError::Timeout`
                // or it is indistinguishable from a connection fault. That
                // distinction now decides whether the default retry budget
                // is spent on it (`RetryBudget::covers`) — classifying a
                // timeout as transport made the model's own timeout get
                // retried, turning a 400ms budget into ~2s.
                let send_started = Instant::now();
                let resp =
                    crate::stream_timeout::send_with_deadline(req, connect_deadline, send_started)
                        .await
                        .map_err(|be| {
                            crate::cooldown::note_failure(tracker, model_id, cooldown_cfg, be)
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

                // Relay response headers that matter for the client.
                let upstream_headers = resp.headers().clone();
                // The caller asked to stream and the upstream answered one:
                // hand the live response on so its frames reach the caller
                // as they arrive (#998). Everything else — including a
                // provider that ignored `stream=true` and replied with a
                // JSON transcript — is read whole, so the usage, duration
                // and output-guardrail passes below still see a body.
                if live_relay && is_event_stream(&upstream_headers) {
                    return Ok((upstream_headers, AudioUpstreamBody::Live(resp)));
                }
                let read = async {
                    match stream_budget.filter(|_| live_relay) {
                        // No request-level timeout was set on the live path,
                        // so bound this read rather than leaving it open.
                        Some(d) => tokio::time::timeout(d, resp.bytes())
                            .await
                            .map_err(|_| aisix_gateway::BridgeError::Timeout {
                                elapsed_ms: d.as_millis() as u64,
                                cause: String::new(),
                            })?
                            .map_err(|e| aisix_gateway::BridgeError::UpstreamDecode(e.to_string())),
                        None => resp
                            .bytes()
                            .await
                            .map_err(|e| aisix_gateway::BridgeError::UpstreamDecode(e.to_string())),
                    }
                };
                let body_bytes = read.await.map_err(|be| {
                    crate::cooldown::note_failure(tracker, model_id, cooldown_cfg, be)
                })?;
                Ok((upstream_headers, AudioUpstreamBody::Buffered(body_bytes)))
            }
        })
        .await
        {
            Ok(v) => v,
            Err(err) => return Err(ProxyError::Bridge(err)),
        };

    state.health.record_success(&model_entry.value.display_name);
    state.runtime_status.mark_healthy(&model_entry.id);

    let body_bytes = match upstream_body {
        AudioUpstreamBody::Buffered(bytes) => bytes,
        // #998: forward the upstream's SSE frames as they arrive. The
        // caller sees the same wire bytes it would get straight from the
        // provider, at the same pace; a side-channel decoder reads the
        // terminal `transcript.text.done` off the same stream so the
        // request is still billed and attributed.
        AudioUpstreamBody::Live(resp) => {
            // Duration cost basis (#457): a streamed transcription reports
            // no duration anywhere — the terminal event carries tokens
            // only — so the uploaded file is the only basis. Probed here,
            // while `fields` is still in scope.
            let duration_seconds = fields
                .iter()
                .find(|(name, ..)| name == "file")
                .and_then(|(.., data)| probe_audio_duration_seconds(data))
                .unwrap_or(0.0);

            // #450/#688: the reservation outlives the handler. The
            // concurrency slot stays held until the stream ends, and the
            // terminal token cost is applied to TPM/TPD from the guard —
            // the sync analog of the buffered path's `commit_tokens`.
            let post_stream_keys = reservation.keys();
            let stream_hold = reservation.into_stream_hold();
            let limiter = std::sync::Arc::clone(&state.limiter);

            // Monitor-only output chains still get their end-of-stream
            // observation (AISIX-Cloud#1010); a block-capable chain never
            // reaches here — `live_relay` sent it down the buffered path.
            let eos_scan =
                aisix_guardrails::Guardrail::runs_on_output(&resolved_chain).then(|| {
                    crate::guardrail_stream::EosOutputScan::new(
                        std::sync::Arc::new(resolved_chain),
                        upstream_model.clone(),
                    )
                });

            let state_c = state.clone();
            let request_id_c = request_id.to_string();
            let model_id_c = model_entry.id.to_string();
            let model_name_c = model_name.clone();
            let provider_c = provider_label.clone();
            let upstream_model_c = upstream_model.clone();
            let pk_id_c = pk_entry.id.to_string();
            let api_key_id_c = auth.entry.id.clone();
            let applied_c = applied_guardrails.clone();
            let redactions_c = redactions.clone();
            let input_monitor_hits = monitor_hits.clone();
            let client_c = client_ctx.clone();
            let captured_prompt_c = captured_prompt.clone();

            let relayed = transcription_relay(
                crate::stream_timeout::with_read_timeout_bytes(resp.bytes_stream(), stream_budget),
                content_cap,
                eos_scan,
                move |outcome| {
                    let (prompt_tokens, completion_tokens) = outcome.usage.unwrap_or((0, 0));
                    let total = u64::from(prompt_tokens) + u64::from(completion_tokens);
                    for key in &post_stream_keys {
                        limiter.add_tokens_post_stream(key, total);
                    }
                    drop(stream_hold);
                    // A stream can outlive several config generations, so
                    // the end-of-stream emit reads a FRESH snapshot rather
                    // than the one the request started on (#941).
                    let snap = state_c.snapshot.load();
                    let pk = crate::usage_attr::ResolvedPk::resolve(&snap, &pk_id_c);
                    let mut monitor_hits = input_monitor_hits;
                    monitor_hits.extend(outcome.output_hits);
                    let captured_content = match (&captured_prompt_c, content_cap) {
                        (Some(prompt), Some(cap)) => {
                            Some(CapturedContent::new(prompt, &outcome.text, cap as usize))
                        }
                        _ => None,
                    };
                    emit_usage_event(
                        &state_c,
                        &snap,
                        &pk,
                        &request_id_c,
                        &model_id_c,
                        &model_name_c,
                        &api_key_id_c,
                        retry_endpoint_label,
                        &provider_c,
                        &upstream_model_c,
                        &applied_c,
                        // A caller that walked away mid-transcript is
                        // reported as 499, matching the other streaming
                        // surfaces — the upstream work still happened, so
                        // the event is emitted either way.
                        if outcome.reached_end {
                            200
                        } else {
                            crate::CLIENT_CLOSED_REQUEST
                        },
                        dispatch_started.elapsed(),
                        prompt_tokens,
                        completion_tokens,
                        duration_seconds,
                        &client_c,
                        redactions_c,
                        monitor_hits,
                        /* guardrail_blocked */ false,
                        captured_content.as_ref(),
                    );
                },
            );
            let mut out = axum::response::Response::new(axum::body::Body::from_stream(
                crate::sse_keepalive::with_heartbeat(
                    Box::pin(relayed),
                    crate::sse_keepalive::interval(),
                ),
            ));
            copy_response_header(&upstream_headers, &mut out, header::CONTENT_TYPE);
            return Ok(AudioDispatchSuccess {
                usage_handled_by_stream: true,
                response: out,
                model_name,
                provider: provider_label,
                model_id: model_entry.id.to_string(),
                provider_key_id: pk_entry.id.to_string(),
                upstream_model,
                // The Drop guard owns the emit; the handler must not
                // double-emit with the counts it cannot see yet.
                usage: None,
                duration_seconds,
                applied_guardrails,
                redactions,
                monitor_hits,
                guardrail_blocked: false,
                // The guard's emit carries the captured content too.
                captured_content: None,
            });
        }
    };

    // Parse the response body best-effort for a `usage` token block
    // (gpt-4o-transcribe returns one; whisper-1 returns none, and the
    // `text`/`srt`/`vtt` response_formats aren't JSON at all). Parse
    // failure or absence → None → zero-token emit. Done before the
    // bytes move into the Body.
    //
    // A `stream=true` transcription answers with SSE instead of a JSON
    // object, so the parse above finds nothing — the counts ride the
    // terminal `transcript.text.done` event. Without the second read the
    // whole streaming surface bills zero and never moves TPM/TPD.
    let usage = serde_json::from_slice::<Value>(&body_bytes)
        .ok()
        .as_ref()
        .and_then(extract_token_usage)
        .or_else(|| extract_sse_token_usage(&upstream_headers, &body_bytes));

    // Duration cost basis (#457): what the upstream reported, else what
    // the uploaded file says. The probe runs only when the response
    // carried nothing, so the common `json` path never pays for it.
    let duration_seconds = upstream_duration_seconds(&body_bytes)
        .or_else(|| {
            fields
                .iter()
                .find(|(name, ..)| name == "file")
                .and_then(|(.., data)| probe_audio_duration_seconds(data))
        })
        .unwrap_or(0.0);

    // #911 [21]: commit the actual token cost so TPM/TPD is enforced for the
    // audio transcription/translation endpoints like chat + embeddings.
    // Pre-fix the reservation dropped uncommitted and the counter never moved.
    let total_tokens = usage
        .map(|(prompt, completion)| u64::from(prompt) + u64::from(completion))
        .unwrap_or(0);
    reservation.commit_tokens(total_tokens).await;

    // #696: run the output guardrail chain on the transcript — it is
    // caller-visible model output, scanned like chat's replies. Pre-fix an
    // output block/mask enforced on /v1/chat/completions was bypassable by
    // transcribing audio. The upstream already billed (tokens committed
    // above), so a block returns the redacted 422 while keeping the billed
    // usage marked `guardrail_blocked` — same as completions #911 [23].
    if aisix_guardrails::Guardrail::runs_on_output(&resolved_chain) {
        let transcript = transcription_output_text(&body_bytes);
        if !transcript.is_empty() {
            let synth = ChatResponse {
                id: String::new(),
                model: model_name.clone(),
                message: ChatMessage::assistant(transcript),
                finish_reason: FinishReason::Stop,
                usage: UsageStats::default(),
            };
            let (verdict, hits) =
                aisix_guardrails::Guardrail::check_output_observed(&resolved_chain, &synth).await;
            monitor_hits.extend(hits);
            if let aisix_guardrails::GuardrailVerdict::Block {
                reason,
                guardrail_name,
            } = verdict
            {
                // Per #153 the matched-pattern detail stays in ops logs only.
                tracing::warn!(
                    guardrail_hook = "output",
                    model = %model_name,
                    reason = %reason,
                    "guardrail blocked audio transcript response",
                );
                return Ok(AudioDispatchSuccess {
                    usage_handled_by_stream: false,
                    response: ProxyError::ContentFiltered(crate::error::guardrail_block_message(
                        "response",
                        guardrail_name.as_deref(),
                    ))
                    .into_response(),
                    model_name,
                    provider: provider_label,
                    model_id: model_entry.id.to_string(),
                    provider_key_id: pk_entry.id.to_string(),
                    upstream_model: upstream_model.clone(),
                    usage,
                    duration_seconds,
                    applied_guardrails,
                    redactions,
                    monitor_hits: monitor_hits.clone(),
                    guardrail_blocked: true,
                    // The blocked transcript never reached the client — no
                    // content capture, matching the chat surface.
                    captured_content: None,
                });
            }
        }
    }

    // #932/#696: mask-action PII rules rewrite the transcript AFTER the
    // block check passes, BEFORE it reaches the caller.
    let body_bytes =
        match crate::redact::redact_transcription_response(&resolved_chain, &body_bytes) {
            Some((rewritten, counts)) => {
                crate::redact::merge_counts(&mut redactions, counts);
                Bytes::from(rewritten)
            }
            None => body_bytes,
        };

    // Content capture (#700): the transcript the caller sees — read from
    // the POST-redaction body so masked PII stays masked in the exported
    // content.
    let captured_content = match (&captured_prompt, content_cap) {
        (Some(prompt), Some(cap)) => Some(CapturedContent::new(
            prompt,
            &String::from_utf8_lossy(&body_bytes),
            cap as usize,
        )),
        _ => None,
    };

    let mut out = axum::response::Response::new(axum::body::Body::from(body_bytes));
    copy_response_header(&upstream_headers, &mut out, header::CONTENT_TYPE);
    Ok(AudioDispatchSuccess {
        usage_handled_by_stream: false,
        response: out,
        model_name,
        provider: provider_label,
        model_id: model_entry.id.to_string(),
        provider_key_id: pk_entry.id.to_string(),
        upstream_model,
        usage,
        duration_seconds,
        applied_guardrails,
        redactions,
        monitor_hits,
        guardrail_blocked: false,
        captured_content,
    })
}

/// The caller-visible transcript text for output-guardrail scanning (#696):
/// the JSON `text` field plus `segments[].text` (`json` / `verbose_json`
/// response formats — segments are scanned too so a response carrying text
/// only in segments can't bypass the check), or the raw body for the
/// plain-text formats (`text` / `srt` / `vtt`).
fn transcription_output_text(body: &[u8]) -> String {
    if let Ok(json) = serde_json::from_slice::<Value>(body) {
        let mut parts: Vec<&str> = Vec::new();
        if let Some(t) = json.get("text").and_then(|t| t.as_str()) {
            parts.push(t);
        }
        if let Some(segments) = json.get("segments").and_then(|s| s.as_array()) {
            parts.extend(
                segments
                    .iter()
                    .filter_map(|s| s.get("text").and_then(|t| t.as_str())),
            );
        }
        return parts.join("\n");
    }
    String::from_utf8_lossy(body).into_owned()
}

/// JSON passthrough for `/v1/audio/speech` — returns binary audio bytes.
/// Returns `(response, provider_label, model_id)`; `model_id` lets the
/// handler attribute a (zero-token) UsageEvent (#406).
/// Build a [`ChatFormat`](aisix_gateway::ChatFormat) from the speech `input`
/// text so the input guardrail chain can scan it (#545). Never sent upstream.
fn speech_input_to_chat(model: &str, body: &Value) -> aisix_gateway::ChatFormat {
    let messages = match body.get("input").and_then(|v| v.as_str()) {
        Some(s) if !s.is_empty() => vec![aisix_gateway::ChatMessage::user(s.to_string())],
        _ => Vec::new(),
    };
    aisix_gateway::ChatFormat::new(model, messages)
}

#[allow(clippy::type_complexity)]
/// `/v1/audio/speech`'s dispatch result. TTS reports no usage block, so
/// this carries only what the terminal emit needs — a struct rather than
/// the tuple it used to be, matching `AudioDispatchSuccess` above.
struct SpeechDispatchSuccess {
    response: Response,
    provider: String,
    model_id: String,
    provider_key_id: String,
    /// Provider-side model name, for the `upstream_model` metric label
    /// (AISIX-Cloud#1234).
    upstream_model: String,
    applied_guardrails: Vec<AppliedGuardrail>,
    redactions: crate::redact::RedactionCounts,
    monitor_hits: Vec<aisix_core::GuardrailMonitorHit>,
    captured_content: Option<CapturedContent>,
}

async fn speech_dispatch(
    state: &ProxyState,
    snapshot: &aisix_core::AisixSnapshot,
    auth: &AuthenticatedKey,
    mut body: Value,
    request_id: &str,
    client_ctx: &ClientContext,
) -> Result<SpeechDispatchSuccess, ProxyError> {
    let model_name = body
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ProxyError::InvalidRequest("missing `model` field".into()))?
        .to_string();

    let model_entry = crate::model_resolve::resolve_model(snapshot, &model_name)
        .ok_or_else(|| ProxyError::ModelNotFound(model_name.clone()))?;

    if !auth.key().can_access(&model_name) {
        return Err(ProxyError::ModelForbidden(model_name.clone()));
    }

    // Client-IP allowlist gate (#557): reject before guardrails / upstream.
    crate::dispatch::check_ip_access(&model_entry.value, &client_ctx.source_ip)?;

    // #545: /v1/audio/speech must run input guardrails. Before this it
    // forwarded the user `input` text (synthesized to audio) with no
    // configured content/DLP check, so a block enforced on
    // /v1/chat/completions was bypassable by switching surface. Run before
    // the rate-limit reservation so a content-policy refusal doesn't burn an
    // RPM slot. (Output is binary audio, not scannable text — no output hook.)
    let guardrail_ctx = aisix_guardrails::RequestContext {
        passthrough_route_id: "",
        model_id: &model_entry.id,
        mcp_server_id: "",
        api_key_id: &auth.entry.id,
        team_id: auth.key().team_id.as_deref(),
    };
    let resolved_chain = state.guardrail_index.resolve(&guardrail_ctx);
    // Record which guardrails govern this request (#379 parity) for the emitted
    // UsageEvent. Empty when none attached.
    let applied_guardrails = resolved_chain.applied().to_vec();
    let mut monitor_hits: Vec<aisix_core::GuardrailMonitorHit> = Vec::new();
    if !resolved_chain.is_empty() {
        let chat = speech_input_to_chat(&model_name, &body);
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
                "guardrail blocked /v1/audio/speech request",
            );
            return Err(ProxyError::ContentFiltered(
                crate::error::guardrail_block_message("request", guardrail_name.as_deref()),
            ));
        }
    }

    // #932/#696: mask-action PII rules rewrite the `input` text in place
    // AFTER the block check passes, BEFORE the body is forwarded upstream.
    // Pre-#696 a mask-action detector was a silent no-op here.
    let redactions = crate::redact::redact_speech_request(&resolved_chain, &mut body);

    // Content capture (#700, LiteLLM parity): the post-redaction request
    // body (the `input` text to synthesize) is the prompt; the binary audio
    // response is NOT captured — LiteLLM logs no TTS response either.
    let captured_content = content_capture_cap(
        snapshot
            .observability_exporters
            .entries()
            .iter()
            .map(|e| &e.value),
    )
    .map(|cap| {
        CapturedContent::new(
            &serde_json::to_string(&body).unwrap_or_default(),
            "",
            cap as usize,
        )
    });

    let model_rl =
        crate::quota::ModelRateLimit::from_model(&model_name, &model_entry.id, &model_entry.value);
    let reservation = crate::quota::enforce(state, snapshot, auth, Some(&model_rl)).await?;

    let model = &model_entry.value;
    let provider = crate::dispatch::require_provider(model)?;
    let upstream_model = crate::dispatch::require_upstream_model(model)?.to_string();
    let pk_entry = crate::dispatch::resolve_provider_key(snapshot, model)?;
    let api_key = crate::dispatch::require_api_key(&pk_entry.value, model)?;

    let provider_label = provider.to_ascii_lowercase();

    // Rewrite model field.
    if let Some(m) = body.get_mut("model") {
        *m = Value::String(upstream_model.clone());
    }

    // Apply the PK's `request.*` overrides (body + headers) like the OpenAI
    // bridge's chat() path — /v1/audio/speech is a JSON passthrough that builds
    // the request directly (AISIX-Cloud#867 follow-up). No-op when none set.
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

    let mut headers = axum::http::HeaderMap::new();
    let auth_hv = header::HeaderValue::from_str(&format!("Bearer {api_key}")).map_err(|e| {
        ProxyError::Bridge(aisix_gateway::BridgeError::Config(format!(
            "api key contains invalid header chars: {e}"
        )))
    })?;
    headers.insert(header::AUTHORIZATION, auth_hv);
    headers.insert(
        header::CONTENT_TYPE,
        header::HeaderValue::from_static("application/json"),
    );
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
    let speech_url = aisix_gateway::url_cache::cached_endpoint_url(
        &pk_entry.id,
        "proxy/audio/speech",
        &[pk_entry.value.api_base.as_deref().unwrap_or("")],
        || {
            let base = crate::dispatch::resolve_base_url(&pk_entry.value)?;
            Ok::<_, crate::error::ProxyError>(crate::dispatch::build_openai_url(
                &base,
                "/audio/speech",
            ))
        },
    )?;
    let tracker = &state.runtime_status;
    let model_id: &str = &model_entry.id;
    let cooldown_cfg = model.cooldown.as_ref();
    // Send and check the status as one retryable unit; the audio bytes are
    // relayed, not read here. See the same shape in rerank.rs for why
    // `note_failure` stays per attempt.
    //
    // #998: the synthesized audio is forwarded chunk by chunk (LiteLLM's
    // `/v1/audio/speech` does the same, explicitly for latency) — a player
    // can start on the first bytes instead of waiting for the whole file.
    // That moves the read out of the retryable unit: once the 200 is on the
    // wire a failed read truncates the download rather than failing over,
    // the same trade `/v1/videos`' content proxy makes.
    let stream_budget =
        crate::routing::effective_timeouts(model, None, state.default_timeouts).stream;
    let upstream_resp =
        match crate::routing::retrying_dispatch(state, model, "/v1/audio/speech", || {
            let req = speech_url
                .clone()
                .post_on(&client)
                .headers(headers.clone())
                .json(&body);
            // #554/#911: reqwest's request-level timeout would bound the
            // body read too and cut a long synthesis off mid-file, so the
            // stream budget bounds the connect phase and each chunk
            // instead — the split every relayed path uses.
            async move {
                // `reqwest_error_to_bridge`, not a bare `Transport`: an
                // elapsed `timeout` has to surface as `BridgeError::Timeout`
                // or it is indistinguishable from a connection fault. That
                // distinction now decides whether the default retry budget
                // is spent on it (`RetryBudget::covers`) — classifying a
                // timeout as transport made the model's own timeout get
                // retried, turning a 400ms budget into ~2s.
                let send_started = Instant::now();
                let resp =
                    crate::stream_timeout::send_with_deadline(req, stream_budget, send_started)
                        .await
                        .map_err(|be| {
                            crate::cooldown::note_failure(tracker, model_id, cooldown_cfg, be)
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
                Ok(resp)
            }
        })
        .await
        {
            Ok(v) => v,
            Err(err) => return Err(ProxyError::Bridge(err)),
        };

    state.health.record_success(&model_entry.value.display_name);
    state.runtime_status.mark_healthy(&model_entry.id);

    // #911 [21]: speech synthesis (TTS) reports no token usage — it is billed
    // per input character — so there are no tokens to add to TPM/TPD. The
    // reservation instead becomes a hold that spans the relayed body: the
    // handler returns once the headers are out, so releasing the concurrency
    // slot at handler return would let a key run more simultaneous
    // syntheses than its cap allows (#450).
    let stream_hold = reservation.into_stream_hold();

    let upstream_headers = upstream_resp.headers().clone();
    let relayed = crate::request_id::in_request_span(async_stream::stream! {
        let _hold = stream_hold;
        let inner = crate::stream_timeout::with_read_timeout_bytes(
            upstream_resp.bytes_stream(),
            stream_budget,
        );
        futures::pin_mut!(inner);
        while let Some(item) = futures::StreamExt::next(&mut inner).await {
            yield item;
        }
    });
    let mut out = axum::response::Response::new(axum::body::Body::from_stream(relayed));
    copy_response_header(&upstream_headers, &mut out, header::CONTENT_TYPE);
    // Relayed verbatim when the upstream sent one, like `/v1/videos`'
    // content proxy: reqwest strips it only when it decompresses, which it
    // never does for audio. A mid-stream read timeout then shows up as a
    // short read — the intended signal that the download failed rather than
    // a silently truncated file.
    copy_response_header(&upstream_headers, &mut out, header::CONTENT_LENGTH);
    Ok(SpeechDispatchSuccess {
        response: out,
        provider: provider_label,
        model_id: model_entry.id.to_string(),
        provider_key_id: pk_entry.id.to_string(),
        upstream_model,
        applied_guardrails,
        redactions,
        monitor_hits,
        captured_content,
    })
}

/// Pull `(prompt_tokens, completion_tokens)` from an audio response
/// `usage` block. gpt-4o-transcribe returns
/// `usage: {type:"tokens", input_tokens, output_tokens, ...}`;
/// whisper-1 (and the `text`/`srt`/`vtt` response formats) return no
/// token block → `None`. Spec:
/// <https://platform.openai.com/docs/api-reference/audio/json-object>
/// `(prompt, completion)` from a *streamed* transcription body.
///
/// `stream=true` on the transcribe models answers `text/event-stream`:
/// a run of `transcript.text.delta` events and a terminal
/// `transcript.text.done` that carries the same `usage` block the
/// non-streaming response would have returned. The body is already
/// buffered here, so decode it and take the last event that carries
/// usage — the shape stays the same whether the provider puts it on
/// `transcript.text.done` or on a later event.
///
/// Gated on the upstream content type so a `text`/`srt`/`vtt` transcript
/// that happens to contain a `data:` line is never mistaken for a stream.
fn extract_sse_token_usage(headers: &HeaderMap, body: &[u8]) -> Option<(u32, u32)> {
    if !is_event_stream(headers) {
        return None;
    }
    let mut decoder = aisix_gateway::SseDecoder::new();
    let mut events = decoder.feed(body);
    events.extend(decoder.finish());
    events.iter().rev().find_map(|event| match event {
        aisix_gateway::SseEvent::Data(payload) => serde_json::from_str::<Value>(payload)
            .ok()
            .as_ref()
            .and_then(extract_token_usage),
        aisix_gateway::SseEvent::Done => None,
    })
}

fn extract_token_usage(body: &Value) -> Option<(u32, u32)> {
    let usage = body.get("usage")?;
    let input = usage.get("input_tokens").and_then(Value::as_u64)? as u32;
    let output = usage
        .get("output_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0) as u32;
    Some((input, output))
}

/// Audio length in seconds as the upstream reported it.
///
/// Two shapes, both from OpenAI's transcription object: the default
/// `json` format carries `usage: {type: "duration", seconds: N}`, and
/// `verbose_json` carries a top-level `duration`. Neither is present on
/// the `text` / `srt` / `vtt` formats — those response bodies are not
/// JSON at all — which is what `probe_audio_duration_seconds` covers.
/// <https://platform.openai.com/docs/api-reference/audio/json-object>
fn upstream_duration_seconds(body: &[u8]) -> Option<f64> {
    let json = serde_json::from_slice::<Value>(body).ok()?;
    let from_usage = json
        .get("usage")
        .and_then(|u| u.get("seconds"))
        .and_then(Value::as_f64);
    let seconds = from_usage.or_else(|| json.get("duration").and_then(Value::as_f64))?;
    (seconds.is_finite() && seconds > 0.0).then_some(seconds)
}

/// Audio length in seconds read off the uploaded file.
///
/// The fallback for every response format that carries no duration. Cost
/// basis must not depend on which `response_format` the caller picked —
/// otherwise `response_format=text` is an unmetered channel, the same
/// shape of bypass as an unbilled stream.
///
/// Header/metadata parse only: `lofty` reads container and codec
/// properties without decoding audio, so an arbitrary caller upload
/// costs microseconds and cannot pull in a decode path. Anything it
/// cannot identify yields `None` → a zero cost basis, never an error:
/// the transcript already succeeded and the upstream already billed.
fn probe_audio_duration_seconds(audio: &[u8]) -> Option<f64> {
    use lofty::file::AudioFile;
    use lofty::probe::Probe;

    // WebM/Matroska first: `lofty` does not read EBML, and WebM is what
    // a browser's MediaRecorder uploads by default — leaving it out
    // would keep the most common web-app upload unbilled.
    if audio.starts_with(&crate::ebml::EBML_MAGIC) {
        return crate::ebml::duration_seconds(audio);
    }

    let probed = Probe::new(std::io::Cursor::new(audio))
        .guess_file_type()
        .ok()?
        .read()
        .ok()?;
    let seconds = probed.properties().duration().as_secs_f64();
    (seconds > 0.0).then_some(seconds)
}

/// Terminal request-metric emit for the two transcription-shaped routes,
/// which share `AudioDispatchSuccess` and would otherwise repeat the same
/// label set twice.
fn record_audio_metrics(
    state: &ProxyState,
    pk: &crate::usage_attr::ResolvedPk<'_>,
    endpoint: &'static str,
    auth: &AuthenticatedKey,
    success: &AudioDispatchSuccess,
    status: u16,
    elapsed: Duration,
) {
    crate::request_metrics::record(
        state,
        endpoint,
        crate::request_metrics::Caller::new(auth),
        crate::request_metrics::Upstream {
            provider: &success.provider,
            model: &success.model_name,
            upstream_model: &success.upstream_model,
            pk: pk.labels(),
            // True exactly on the live SSE relay (#998) — the flag that
            // moves the usage emit into the stream is the same condition
            // that makes this a streamed response.
            stream: success.usage_handled_by_stream,
            ..Default::default()
        },
        status,
        elapsed,
    );
}

/// Emit a UsageEvent for a successful transcription/translation. Tokens
/// come from the upstream `usage` block when present (gpt-4o-transcribe);
/// zero otherwise (whisper-1) — the request is still visible/attributed.
#[allow(clippy::too_many_arguments)]
fn emit_audio_usage(
    state: &ProxyState,
    snapshot: &aisix_core::AisixSnapshot,
    pk: &crate::usage_attr::ResolvedPk<'_>,
    request_id: &str,
    endpoint: &'static str,
    success: &AudioDispatchSuccess,
    api_key_id: &str,
    status: u16,
    elapsed: Duration,
    client: &ClientContext,
) {
    let (prompt_tokens, completion_tokens) = success.usage.unwrap_or((0, 0));
    emit_usage_event(
        state,
        snapshot,
        pk,
        request_id,
        &success.model_id,
        &success.model_name,
        api_key_id,
        endpoint,
        &success.provider,
        &success.upstream_model,
        &success.applied_guardrails,
        status,
        elapsed,
        prompt_tokens,
        completion_tokens,
        success.duration_seconds,
        client,
        success.redactions.clone(),
        success.monitor_hits.clone(),
        success.guardrail_blocked,
        success.captured_content.as_ref(),
    );
}

/// Issue #406: push one `UsageEvent` onto cp-api's telemetry sink and
/// fan it out to per-env OTLP exporters. Mirrors
/// `embeddings::emit_usage_event` (#402). `inbound_protocol = "openai"`.
/// Tokens are populated when the upstream returned a `usage` block
/// (gpt-4o-transcribe); zero otherwise — duration-based cost (whisper-1)
/// is a documented cross-repo follow-up (needs duration on the wire +
/// cp-api pricing).
#[allow(clippy::too_many_arguments)]
fn emit_usage_event(
    state: &ProxyState,
    // The request's snapshot + its one ProviderKey observation, resolved
    // by the handler (#941).
    snap: &aisix_core::AisixSnapshot,
    pk: &crate::usage_attr::ResolvedPk<'_>,
    request_id: &str,
    model_id: &str,
    requested_model: &str,
    api_key_id: &str,
    // Metric labels the UsageEvent has no field for (AISIX-Cloud#1234
    // follow-up). `endpoint` too: the three audio routes share this emitter
    // but are three distinct series.
    endpoint: &'static str,
    provider: &str,
    upstream_model: &str,
    applied_guardrails: &[AppliedGuardrail],
    status_code: u16,
    elapsed: Duration,
    prompt_tokens: u32,
    completion_tokens: u32,
    // Cost basis for the duration-billed models (#457); 0 when neither
    // the upstream nor the uploaded file yielded a length.
    audio_duration_seconds: f64,
    client: &ClientContext,
    // Per-detector PII mask counts (#932/#696). Empty = no redaction.
    redacted_entity_counts: crate::redact::RedactionCounts,
    // Monitor-mode guardrail observations (AISIX-Cloud#562).
    guardrail_monitor_hits: Vec<aisix_core::GuardrailMonitorHit>,
    // #696: transcript blocked by an output guardrail after upstream billing.
    guardrail_blocked: bool,
    // Captured request/response content (#700). Forwarded only to `fan_out`,
    // never to the CP sink.
    content: Option<&CapturedContent>,
) {
    let mut event = UsageEvent {
        request_id: request_id.to_string(),
        occurred_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        model_id: model_id.to_string(),
        api_key_id: api_key_id.to_string(),
        requested_model: requested_model.to_string(),
        prompt_tokens,
        completion_tokens,
        audio_duration_seconds,
        // Single-attempt endpoint: the attempt spans the whole request, so
        // the upstream figure and what the caller waited for coincide.
        upstream_latency_ms: elapsed.as_millis().min(u32::MAX as u128) as u32,
        downstream_latency_ms: elapsed.as_millis().min(u32::MAX as u128) as u32,
        status_code,
        inbound_protocol: "openai".to_string(),
        applied_guardrails: applied_guardrails.to_vec(),
        client_source_ip: client.source_ip.clone(),
        client_user_agent: client.user_agent.clone(),
        redacted_entity_counts,
        guardrail_monitor_hits,
        guardrail_blocked,
        ..Default::default()
    };
    // Per-PK telemetry attribution, same lookup as chat / messages /
    // responses (AISIX-Cloud#867 parity).
    crate::usage_attr::apply_pk_telemetry(&mut event, pk);
    // Handler label "audio" — bucketed prometheus counter (#408).
    crate::usage_attr::apply_jwt_identity(&mut event, client.jwt.as_ref());
    let usage_model = crate::usage_attr::usage_event_model_label(snap, &event.requested_model);
    state.usage_sink.try_emit(
        "audio",
        event.clone(),
        crate::usage_attr::usage_event_labels(&usage_model, pk),
    );
    let exporters = crate::usage_attr::live_exporters(state, snap);
    state
        .otlp_fan_out
        .fan_out(&event, content, exporters.iter().map(|e| &e.value));
    // Speech (TTS) reports no tokens at all, so this is a no-op there; the
    // transcription routes report them when the model supplies a usage block.
    let owned_caller = crate::request_metrics::Caller::from_api_key_id(snap, api_key_id);
    crate::request_metrics::record_usage(
        state,
        endpoint,
        owned_caller.as_caller(),
        crate::request_metrics::Upstream {
            provider,
            model: requested_model,
            upstream_model,
            pk: pk.labels(),
            ..Default::default()
        },
        crate::request_metrics::Tokens {
            input: prompt_tokens,
            output: completion_tokens,
            total: prompt_tokens.saturating_add(completion_tokens),
            spend_usd: 0.0,
            client_type: state.client_classifier.classify(&client.user_agent),
        },
    );
}

fn copy_response_header(src: &HeaderMap, dst: &mut Response, name: header::HeaderName) {
    if let Some(val) = src.get(&name) {
        dst.headers_mut().insert(name, val.clone());
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_access_log(
    method: &'static str,
    path: &'static str,
    model: &str,
    provider: &str,
    api_key_id: &str,
    status: u16,
    latency: Duration,
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
        method,
        path,
        status,
        latency,
        provider: Some(provider),
        model: Some(model),
        api_key_id: Some(api_key_id),
        prompt_tokens: None,
        completion_tokens: None,
        total_tokens: None,
        request_id,
        // No provider response id: transcription/translation return
        // `{text, usage}` and speech returns audio bytes — neither carries
        // one (AISIX-Cloud#1289).
        provider_request_id: None,
        served_by_model: None,
        routing_attempt_count: None,
        routing_fallback_count: None,
        error_kind,
        error: error.as_deref(),
    }
    .emit();
}

// The audio handler reuses the same client as messages.rs. It's exported
// from there to avoid creating multiple global Clients.
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
    use wiremock::matchers::{body_partial_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn cfg() -> ProxyConfig {
        ProxyConfig {
            addr: "127.0.0.1:0".into(),
            request_body_limit_bytes: 10_485_760, // 10 MB for audio
            real_ip: Default::default(),
            request_id: Default::default(),
            url_rewrites: Vec::new(),
            tls: None,
            thread_per_core: None,
            workers: None,
        }
    }

    const PK_ID: &str = "11111111-1111-1111-1111-111111111111";

    fn whisper_model(name: &str) -> ResourceEntry<Model> {
        let json = format!(
            r#"{{
                "display_name": "{name}",
                "provider": "openai",
                "model_name": "whisper-1",
                "provider_key_id": "{PK_ID}"
            }}"#
        );
        let m: Model = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("m-1", m, 1)
    }

    fn tts_model(name: &str) -> ResourceEntry<Model> {
        let json = format!(
            r#"{{
                "display_name": "{name}",
                "provider": "openai",
                "model_name": "tts-1",
                "provider_key_id": "{PK_ID}"
            }}"#
        );
        let m: Model = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("m-2", m, 1)
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

    /// A PK carrying per-PK telemetry attribution tags (AISIX-Cloud#867
    /// parity) for asserting they land on the emitted UsageEvent.
    fn provider_key_entry_tagged(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        let json = format!(
            r#"{{"display_name":"openai-up","secret":"sk-up","api_base":"{api_base}","provider":"openai","adapter":"openai","telemetry_tags":{{"kind":"catalog","featured":true,"branded_provider":"openai","pk_label":"prod-audio-key"}}}}"#
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&json).unwrap();
        ResourceEntry::new(PK_ID, pk, 1)
    }

    fn new_snap_tagged(api_base: &str) -> AisixSnapshot {
        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(provider_key_entry_tagged(api_base));
        snap
    }

    /// A PK carrying `request.*` operator overrides (AISIX-Cloud#867):
    /// a default body field + a default header that the audio handlers
    /// must apply to the upstream request.
    fn provider_key_entry_overrides(api_base: &str) -> ResourceEntry<aisix_core::ProviderKey> {
        let json = format!(
            r#"{{"display_name":"openai-up","secret":"sk-up","api_base":"{api_base}","provider":"openai","adapter":"openai","request":{{"default_body_fields":{{"safe_flag":true}},"default_headers":{{"x-custom":"trace-on"}}}}}}"#
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(&json).unwrap();
        ResourceEntry::new(PK_ID, pk, 1)
    }

    fn new_snap_overrides(api_base: &str) -> AisixSnapshot {
        let snap = AisixSnapshot::new();
        snap.provider_keys
            .insert(provider_key_entry_overrides(api_base));
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

    fn keyword_input_guardrail(literal: &str) -> ResourceEntry<aisix_core::Guardrail> {
        let json = format!(
            r#"{{"name":"t","enabled":true,"hook_point":"input","fail_open":false,"kind":"keyword","patterns":[{{"kind":"literal","value":"{literal}"}}]}}"#
        );
        let g: aisix_core::Guardrail = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("g-1", g, 1)
    }

    fn speech_req(body: &str) -> Request<axum::body::Body> {
        Request::builder()
            .method("POST")
            .uri("/v1/audio/speech")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body.to_string()))
            .unwrap()
    }

    /// #379 parity: a successful /v1/audio/speech whose input passes an attached
    /// input guardrail records that guardrail's `{kind, hook}` in the emitted
    /// UsageEvent's `applied_guardrails`.
    #[tokio::test]
    async fn speech_applied_guardrails_recorded_on_usage_event() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/audio/speech"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(b"AUDIO".to_vec()))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(tts_model("my-tts"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_input_guardrail("BLOCKME"));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_app_with_sink(snap, tx);
        // Benign input (no "BLOCKME") → passes the guardrail.
        let req = speech_req(r#"{"model":"my-tts","input":"hello","voice":"alloy"}"#);
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let ev = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("UsageEvent must be emitted")
            .expect("usage_sink sender dropped");
        assert!(
            !ev.applied_guardrails.is_empty(),
            "the attached input guardrail must be recorded"
        );
        assert_eq!(ev.applied_guardrails[0].kind, "keyword");
        assert_eq!(ev.applied_guardrails[0].hook, "input");
    }

    /// #545: a configured input guardrail must fire on /v1/audio/speech — a
    /// blocked `input` returns 422 content_filter, upstream never contacted.
    #[tokio::test]
    async fn input_guardrail_blocks_speech_input_returns_422() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/audio/speech"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "audio/mpeg")
                    .set_body_bytes(b"ID3".to_vec()),
            )
            .expect(0)
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(tts_model("my-tts"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_input_guardrail("BLOCKME"));

        let app = build_app(snap);
        let resp = tower::ServiceExt::oneshot(
            app,
            speech_req(r#"{"model":"my-tts","input":"say BLOCKME aloud","voice":"alloy"}"#),
        )
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

    /// #545 companion: benign input with a guardrail configured still forwards
    /// (`expect(1)`) and returns the audio bytes.
    #[tokio::test]
    async fn input_guardrail_allows_benign_speech_input() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/audio/speech"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "audio/mpeg")
                    .set_body_bytes(b"ID3\x03\x00".to_vec()),
            )
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(tts_model("my-tts"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_input_guardrail("BLOCKME"));

        let app = build_app(snap);
        let resp = tower::ServiceExt::oneshot(
            app,
            speech_req(r#"{"model":"my-tts","input":"Hello there","voice":"alloy"}"#),
        )
        .await
        .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn speech_unauthenticated_returns_401() {
        let snap = new_snap("http://unused");
        snap.models.insert(tts_model("my-tts"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/audio/speech")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                r#"{"model":"my-tts","input":"Hello","voice":"alloy"}"#,
            ))
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn speech_unknown_model_returns_404() {
        let snap = new_snap("http://unused");
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/audio/speech")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                r#"{"model":"nonexistent","input":"Hello","voice":"alloy"}"#,
            ))
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn speech_happy_path_returns_audio_bytes() {
        let upstream = MockServer::start().await;
        // TTS endpoint returns raw MP3 bytes.
        let fake_mp3 = b"ID3\x03\x00\x00\x00";
        Mock::given(method("POST"))
            .and(path("/v1/audio/speech"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "audio/mpeg")
                    .set_body_bytes(fake_mp3.to_vec()),
            )
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(tts_model("my-tts"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/audio/speech")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                r#"{"model":"my-tts","input":"Hello","voice":"alloy"}"#,
            ))
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(
            ct.contains("audio"),
            "expected audio content-type, got {ct}"
        );
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        assert_eq!(&bytes[..3], b"ID3");
    }

    #[tokio::test]
    async fn transcriptions_unauthenticated_returns_401() {
        let snap = new_snap("http://unused");
        snap.models.insert(whisper_model("my-whisper"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        // A minimal multipart body.
        let body = "--boundary\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nmy-whisper\r\n--boundary--\r\n";
        let req = Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("content-type", "multipart/form-data; boundary=boundary")
            .body(axum::body::Body::from(body))
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
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

    /// A minimal multipart body carrying `model` + a tiny fake audio
    /// `file` field — enough for the gateway to extract the model and
    /// forward the form.
    /// `transcription_multipart` plus the `stream=true` field, i.e. what a
    /// caller that wants the transcript incrementally sends.
    fn streaming_transcription_multipart(model: &str) -> (String, axum::body::Body) {
        let body = format!(
            "--b\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}\r\n\
             --b\r\nContent-Disposition: form-data; name=\"stream\"\r\n\r\ntrue\r\n\
             --b\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.mp3\"\r\n\
             Content-Type: audio/mpeg\r\n\r\nID3fakeaudio\r\n--b--\r\n"
        );
        (
            "multipart/form-data; boundary=b".to_string(),
            axum::body::Body::from(body),
        )
    }

    fn transcription_multipart(model: &str) -> (String, axum::body::Body) {
        let body = format!(
            "--b\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}\r\n\
             --b\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.mp3\"\r\n\
             Content-Type: audio/mpeg\r\n\r\nID3fakeaudio\r\n--b--\r\n"
        );
        (
            "multipart/form-data; boundary=b".to_string(),
            axum::body::Body::from(body),
        )
    }

    /// Issue #406: gpt-4o-transcribe returns a `usage` token block —
    /// a successful transcription must emit a UsageEvent with those
    /// tokens, attributed to the api_key + model, inbound_protocol
    /// "openai".
    #[tokio::test]
    async fn transcriptions_emit_usage_event_with_tokens() {
        let upstream = MockServer::start().await;
        let body = serde_json::json!({
            "text": "hello world",
            "usage": {
                "type": "tokens",
                "input_tokens": 14,
                "output_tokens": 4,
                "total_tokens": 18
            }
        });
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(whisper_model("my-transcribe"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_app_with_sink(snap, tx);
        let (ct, body) = transcription_multipart("my-transcribe");
        let req = Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", ct)
            .body(body)
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("UsageEvent must be emitted for /v1/audio/transcriptions 200")
            .expect("usage_sink sender dropped");
        assert_eq!(event.prompt_tokens, 14);
        assert_eq!(event.completion_tokens, 4);
        assert_eq!(event.status_code, 200);
        assert_eq!(event.api_key_id, "k-1");
        assert_eq!(event.model_id, "m-1");
        assert_eq!(event.inbound_protocol, "openai");
    }

    /// A minimal RIFF/WAVE container holding `seconds` of 8 kHz 16-bit
    /// mono PCM — a real audio file for the probe path, small enough to
    /// build inline.
    fn wav_bytes(seconds: u32) -> Vec<u8> {
        let samples = 8000usize * 2 * seconds as usize;
        let mut wav = Vec::with_capacity(44 + samples);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&((36 + samples) as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&8000u32.to_le_bytes());
        wav.extend_from_slice(&16000u32.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(samples as u32).to_le_bytes());
        wav.resize(44 + samples, 0);
        wav
    }

    /// A multipart body whose `file` part is a real WAV, so the handler's
    /// fallback probe has something to read.
    fn transcription_multipart_wav(model: &str, seconds: u32) -> (String, axum::body::Body) {
        let mut body = Vec::new();
        body.extend_from_slice(
            format!(
                "--b\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}\r\n\
                 --b\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\
                 Content-Type: audio/wav\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(&wav_bytes(seconds));
        body.extend_from_slice(b"\r\n--b--\r\n");
        (
            "multipart/form-data; boundary=b".to_string(),
            axum::body::Body::from(body),
        )
    }

    /// AISIX-Cloud#1138: whisper-1 bills by audio length and reports no
    /// tokens, so the emitted event must carry the duration or cp-api has
    /// nothing to price the request with.
    #[tokio::test]
    async fn whisper_response_emits_the_duration_cost_basis() {
        let upstream = MockServer::start().await;
        let body = serde_json::json!({
            "text": "hello world",
            "usage": {"type": "duration", "seconds": 11.0}
        });
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(whisper_model("my-transcribe"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_app_with_sink(snap, tx);
        let (ct, body) = transcription_multipart("my-transcribe");
        let req = Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", ct)
            .body(body)
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("UsageEvent must be emitted")
            .expect("usage_sink sender dropped");
        assert_eq!(event.audio_duration_seconds, 11.0);
        assert_eq!(
            (event.prompt_tokens, event.completion_tokens),
            (0, 0),
            "whisper-1 reports no tokens — duration is the whole cost basis"
        );
    }

    /// AISIX-Cloud#1138: `response_format=text` answers with a body that
    /// carries no usage at all. The cost basis must not depend on which
    /// response format the caller asked for, so the handler falls back to
    /// the uploaded file's own length.
    #[tokio::test]
    async fn text_format_falls_back_to_the_uploaded_file_duration() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .respond_with(ResponseTemplate::new(200).set_body_raw("hello world", "text/plain"))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(whisper_model("my-transcribe"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_app_with_sink(snap, tx);
        let (ct, body) = transcription_multipart_wav("my-transcribe", 3);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", ct)
            .body(body)
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("UsageEvent must be emitted")
            .expect("usage_sink sender dropped");
        assert!(
            (event.audio_duration_seconds - 3.0).abs() < 0.05,
            "3s of uploaded audio must be the cost basis, got {}",
            event.audio_duration_seconds
        );
    }

    /// AISIX-Cloud#1138: `MediaRecorder` uploads `audio/webm`, which
    /// `lofty` cannot read — so a WebM transcription asked for as `text`
    /// would have reported no length and billed nothing, which is the
    /// exact bypass the file probe exists to close. Uses the header of a
    /// file ffmpeg produced (see `ebml::tests`), whose declared length is
    /// 7.008s.
    #[test]
    fn probing_reads_a_webm_upload() {
        let webm = crate::ebml::tests::REAL_FFMPEG_WEBM_HEADER;
        let probed =
            super::probe_audio_duration_seconds(webm).expect("a webm upload must be measurable");
        assert!(
            (probed - 7.008).abs() < 0.01,
            "webm should probe as ~7.008s, got {probed}"
        );
    }

    /// The default `json` transcription reports its length under
    /// `usage.seconds` (the duration variant of OpenAI's usage oneOf),
    /// which is the cost basis for whisper-1 — a model that reports no
    /// tokens at all.
    #[test]
    fn duration_reads_the_usage_seconds_variant() {
        let body = br#"{"text":"hi","usage":{"type":"duration","seconds":11.0}}"#;
        assert_eq!(super::upstream_duration_seconds(body), Some(11.0));
    }

    /// `verbose_json` puts the same figure at the top level instead.
    #[test]
    fn duration_reads_the_verbose_json_field() {
        let body = br#"{"task":"transcribe","duration":2.66,"text":"hi"}"#;
        assert_eq!(super::upstream_duration_seconds(body), Some(2.66));
    }

    /// A token-usage response carries no duration — the caller must fall
    /// back to the file probe rather than read a zero off `usage`.
    #[test]
    fn duration_absent_from_a_token_usage_response() {
        let body =
            br#"{"text":"hi","usage":{"type":"tokens","input_tokens":26,"output_tokens":12}}"#;
        assert_eq!(super::upstream_duration_seconds(body), None);
    }

    /// AISIX-Cloud#1138: `response_format=text` (and `srt`/`vtt`) answers
    /// with a body that is not JSON at all, so the cost basis has to come
    /// off the uploaded audio — otherwise the caller picks whether the
    /// request is metered by picking a response format.
    #[test]
    fn duration_falls_back_to_probing_the_uploaded_file() {
        // 1 second of 8 kHz 16-bit mono PCM in a minimal RIFF/WAVE container.
        let samples = 8000usize * 2;
        let mut wav = Vec::with_capacity(44 + samples);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&((36 + samples) as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&1u16.to_le_bytes()); // mono
        wav.extend_from_slice(&8000u32.to_le_bytes()); // sample rate
        wav.extend_from_slice(&16000u32.to_le_bytes()); // byte rate
        wav.extend_from_slice(&2u16.to_le_bytes()); // block align
        wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(samples as u32).to_le_bytes());
        wav.resize(44 + samples, 0);

        let probed = super::probe_audio_duration_seconds(&wav)
            .expect("a well-formed WAV must yield a duration");
        assert!(
            (probed - 1.0).abs() < 0.05,
            "1s of 8kHz mono PCM should probe as ~1s, got {probed}"
        );
    }

    /// Caller uploads are arbitrary bytes. An unrecognised file must
    /// degrade to "no cost basis", never to an error — the transcript
    /// already succeeded and the upstream already billed for it.
    #[test]
    fn probing_unrecognised_bytes_yields_no_duration() {
        assert_eq!(super::probe_audio_duration_seconds(b"ID3fakeaudio"), None);
        assert_eq!(super::probe_audio_duration_seconds(&[]), None);
    }

    /// AISIX-Cloud#1138: a `stream=true` transcription answers
    /// `text/event-stream`, so the usage block rides the terminal
    /// `transcript.text.done` event instead of a JSON body. Pre-fix the
    /// JSON parse found nothing and the whole streaming surface emitted
    /// zero tokens — unbilled spend that also never moved TPM/TPD, while
    /// the identical non-streaming request billed normally.
    /// <https://platform.openai.com/docs/api-reference/audio/create-transcription>
    #[tokio::test]
    async fn streamed_transcription_bills_the_terminal_event_usage() {
        let upstream = MockServer::start().await;
        let sse = concat!(
            "data: {\"type\":\"transcript.text.delta\",\"delta\":\"hello\"}\n\n",
            "data: {\"type\":\"transcript.text.delta\",\"delta\":\" world\"}\n\n",
            "data: {\"type\":\"transcript.text.done\",\"text\":\"hello world\",",
            "\"usage\":{\"type\":\"tokens\",\"total_tokens\":38,\"input_tokens\":26,",
            "\"input_token_details\":{\"text_tokens\":0,\"audio_tokens\":26},",
            "\"output_tokens\":12}}\n\n",
            "data: [DONE]\n\n",
        );
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream"))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(whisper_model("my-transcribe"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_app_with_sink(snap, tx);
        let (ct, body) = transcription_multipart("my-transcribe");
        let req = Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", ct)
            .body(body)
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("UsageEvent must be emitted for a streamed transcription")
            .expect("usage_sink sender dropped");
        assert_eq!(
            (event.prompt_tokens, event.completion_tokens),
            (26, 12),
            "the terminal transcript.text.done usage must be billed"
        );
    }

    /// #998: with `stream=true` and an SSE answer the relay is live —
    /// the frames reach the caller as they arrive rather than being
    /// buffered — and the UsageEvent comes from the stream's own guard.
    /// The wire bytes must still be the upstream's, event for event, and
    /// the request must be billed exactly once.
    #[tokio::test]
    async fn streamed_transcription_relays_verbatim_and_bills_once() {
        let upstream = MockServer::start().await;
        let sse = concat!(
            "data: {\"type\":\"transcript.text.delta\",\"delta\":\"hello\"}\n\n",
            "data: {\"type\":\"transcript.text.delta\",\"delta\":\" world\"}\n\n",
            "data: {\"type\":\"transcript.text.done\",\"text\":\"hello world\",",
            "\"usage\":{\"type\":\"tokens\",\"total_tokens\":38,\"input_tokens\":26,",
            "\"input_token_details\":{\"text_tokens\":0,\"audio_tokens\":26},",
            "\"output_tokens\":12}}\n\n",
            "data: [DONE]\n\n",
        );
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream"))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(whisper_model("my-transcribe"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_app_with_sink(snap, tx);
        let (ct, body) = streaming_transcription_multipart("my-transcribe");
        let req = Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", ct)
            .body(body)
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get(axum::http::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("text/event-stream"),
            "the caller must keep the upstream's streaming content type"
        );
        let relayed = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("the relayed stream must read cleanly");
        assert_eq!(
            String::from_utf8_lossy(&relayed),
            sse,
            "the relay must forward the upstream SSE verbatim"
        );

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("the stream guard must emit a UsageEvent")
            .expect("usage_sink sender dropped");
        assert_eq!(
            (event.prompt_tokens, event.completion_tokens),
            (26, 12),
            "the terminal transcript.text.done usage must be billed"
        );
        assert_eq!(event.status_code, 200);
        assert!(
            rx.try_recv().is_err(),
            "the handler must not emit a second, zero-token event for the same request"
        );
    }

    /// A provider that ignores `stream=true` and answers with a JSON
    /// transcript falls back to the buffered path, so its `usage` block
    /// is still read — a streamed request must not become an unbilled
    /// channel just because the upstream declined to stream (#998).
    #[tokio::test]
    async fn stream_requested_but_json_answered_is_still_billed() {
        let upstream = MockServer::start().await;
        let body = serde_json::json!({
            "text": "hello world",
            "usage": {"type": "tokens", "input_tokens": 14, "output_tokens": 4},
        });
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&body))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(whisper_model("my-transcribe"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_app_with_sink(snap, tx);
        let (ct, form) = streaming_transcription_multipart("my-transcribe");
        let req = Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", ct)
            .body(form)
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("UsageEvent must be emitted")
            .expect("usage_sink sender dropped");
        assert_eq!((event.prompt_tokens, event.completion_tokens), (14, 4));
    }

    /// The SSE read is content-type gated: a `srt`/`vtt` transcript is
    /// `text/plain` and may legitimately contain a line starting with
    /// `data:`, which must never be decoded as a usage-bearing event.
    #[test]
    fn plain_text_transcript_is_not_read_as_a_stream() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(
            axum::http::header::CONTENT_TYPE,
            axum::http::HeaderValue::from_static("text/plain; charset=utf-8"),
        );
        let body = concat!(
            "1\n00:00:00,000 --> 00:00:02,000\n",
            "data: {\"usage\":{\"input_tokens\":999,\"output_tokens\":999}}\n\n",
        );
        assert_eq!(
            super::extract_sse_token_usage(&headers, body.as_bytes()),
            None
        );
    }

    /// AISIX-Cloud#867 parity: a successful audio request must carry the
    /// resolved ProviderKey's telemetry attribution tags (provider_kind /
    /// provider_featured / branded_provider / pk_label) — same lookup as
    /// chat / messages / responses. Fails before the fix (empty tags).
    #[tokio::test]
    async fn emits_provider_telemetry_tags_issue_867() {
        let upstream = MockServer::start().await;
        let body = serde_json::json!({
            "text": "hello world",
            "usage": {"type": "tokens", "input_tokens": 9, "output_tokens": 2, "total_tokens": 11}
        });
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(&upstream)
            .await;

        let snap = new_snap_tagged(&upstream.uri());
        snap.models.insert(whisper_model("my-transcribe"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_app_with_sink(snap, tx);
        let (ct, body) = transcription_multipart("my-transcribe");
        let req = Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", ct)
            .body(body)
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("UsageEvent must be emitted for /v1/audio/transcriptions 200")
            .expect("usage_sink sender dropped");
        assert_eq!(event.provider_kind, "catalog");
        assert!(event.provider_featured);
        assert_eq!(event.branded_provider, "openai");
        assert_eq!(event.pk_label, "prod-audio-key");
    }

    /// Issue #406: whisper-1 `{"text":"..."}` has no `usage` block —
    /// the request still emits a zero-token UsageEvent so it's visible
    /// in /logs and attributed (duration-based cost is a cross-repo
    /// follow-up).
    #[tokio::test]
    async fn transcriptions_emit_zero_token_event_without_usage() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"text": "hi"})),
            )
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(whisper_model("my-whisper"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_app_with_sink(snap, tx);
        let (ct, body) = transcription_multipart("my-whisper");
        let req = Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", ct)
            .body(body)
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("zero-token UsageEvent must still be emitted (visibility)")
            .expect("usage_sink sender dropped");
        assert_eq!(event.prompt_tokens, 0);
        assert_eq!(event.completion_tokens, 0);
        assert_eq!(event.model_id, "m-1");
        assert_eq!(event.inbound_protocol, "openai");
    }

    /// Issue #406: TTS speech returns binary audio (no usage). It still
    /// emits a zero-token UsageEvent so the request is visible +
    /// attributed.
    #[tokio::test]
    async fn speech_emits_zero_token_usage_event() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/audio/speech"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "audio/mpeg")
                    .set_body_bytes(b"ID3\x03\x00\x00\x00".to_vec()),
            )
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(tts_model("my-tts"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_app_with_sink(snap, tx);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/audio/speech")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                r#"{"model":"my-tts","input":"Hello","voice":"alloy"}"#,
            ))
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("speech must emit a zero-token UsageEvent (visibility)")
            .expect("usage_sink sender dropped");
        assert_eq!(event.prompt_tokens, 0);
        assert_eq!(event.completion_tokens, 0);
        assert_eq!(event.model_id, "m-2");
        assert_eq!(event.inbound_protocol, "openai");
    }

    /// gpt-4o-transcribe with a non-default `response_format` can return
    /// the *duration* usage variant — `{"type":"duration","seconds":N}` —
    /// which carries no `input_tokens`. `extract_token_usage` must degrade
    /// that to `None` (→ a zero-token emit, never a panic or mis-parse),
    /// consistent with the duration-cost being a cross-repo follow-up.
    /// Per OpenAI's `usage` oneOf (TranscriptTextUsageTokens |
    /// TranscriptTextUsageDuration):
    /// <https://platform.openai.com/docs/api-reference/audio/json-object>
    #[test]
    fn extract_token_usage_ignores_duration_variant() {
        let v = serde_json::json!({
            "text": "hello world",
            "usage": {"type": "duration", "seconds": 42.7}
        });
        assert_eq!(super::extract_token_usage(&v), None);
    }

    /// #655 parity: an upstream 5xx on /v1/audio/speech now emits ONE zero-token
    /// UsageEvent so the failed request is visible in Logs (status + error
    /// class) and attributed to the api_key — instead of being dropped. Mirrors
    /// `completions.rs::upstream_5xx_emits_zero_token_error_event`.
    #[tokio::test]
    async fn speech_5xx_emits_zero_token_error_event() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/audio/speech"))
            .respond_with(ResponseTemplate::new(500).set_body_string("Internal"))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(tts_model("my-tts"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_app_with_sink(snap, tx);
        let req = speech_req(r#"{"model":"my-tts","input":"hi","voice":"alloy"}"#);
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);

        let ev = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("a failed /v1/audio/speech must emit a zero-token UsageEvent")
            .expect("usage_sink sender dropped");
        assert_eq!(ev.status_code, 502, "upstream 5xx maps to 502");
        assert_eq!(ev.prompt_tokens, 0);
        assert_eq!(ev.api_key_id, "k-1");
        assert_eq!(ev.requested_model, "my-tts");
        assert!(
            !ev.error_class.is_empty(),
            "error_class must classify the failure"
        );
        assert!(
            rx.try_recv().is_err(),
            "exactly one event per failed request"
        );
    }

    /// AISIX-Cloud#867: `/v1/audio/speech` (JSON body) must apply the PK's
    /// `request.*` overrides to BOTH the request body
    /// (`default_body_fields`) and the request headers (`default_headers`).
    /// The Mock matches only when the upstream request carries the injected
    /// body field AND header, so a 200 proves both were applied.
    #[tokio::test]
    async fn speech_applies_pk_request_overrides_issue_867() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/audio/speech"))
            .and(body_partial_json(serde_json::json!({"safe_flag": true})))
            .and(header("x-custom", "trace-on"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(b"AUDIO".to_vec()))
            .mount(&upstream)
            .await;

        let snap = new_snap_overrides(&upstream.uri());
        snap.models.insert(tts_model("my-tts"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/audio/speech")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                r#"{"model":"my-tts","input":"hi","voice":"alloy"}"#,
            ))
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    /// AISIX-Cloud#867: `/v1/audio/transcriptions` (multipart body) must
    /// apply the PK's `request.default_headers` to the upstream request.
    /// Body `request.*` overrides do NOT apply (the body is a multipart
    /// form, not JSON). The Mock matches only on the injected header, so a
    /// 200 proves the operator header was applied.
    #[tokio::test]
    async fn transcriptions_applies_default_headers_issue_867() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .and(header("x-custom", "trace-on"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({"text": "hi"})),
            )
            .mount(&upstream)
            .await;

        let snap = new_snap_overrides(&upstream.uri());
        snap.models.insert(whisper_model("my-transcribe"));
        snap.apikeys.insert(apikey_entry(&["*"]));

        let app = build_app(snap);
        let (ct, body) = transcription_multipart("my-transcribe");
        let req = Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", ct)
            .body(body)
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    fn pii_guardrail(hook: &str) -> ResourceEntry<aisix_core::Guardrail> {
        let json = format!(
            r#"{{"name":"pii","enabled":true,"hook_point":"{hook}","kind":"pii","detectors":[{{"type":"email","action":"mask"}}]}}"#
        );
        let g: aisix_core::Guardrail = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("g-pii", g, 1)
    }

    fn keyword_output_guardrail(literal: &str) -> ResourceEntry<aisix_core::Guardrail> {
        let json = format!(
            r#"{{"name":"t-out","enabled":true,"hook_point":"output","fail_open":false,"kind":"keyword","patterns":[{{"kind":"literal","value":"{literal}"}}]}}"#
        );
        let g: aisix_core::Guardrail = serde_json::from_str(&json).unwrap();
        ResourceEntry::new("g-out", g, 1)
    }

    /// Multipart body carrying `model` + an optional text `prompt` field +
    /// a tiny fake audio `file` — for the #696 prompt-field guardrail tests.
    fn transcription_multipart_with_prompt(
        model: &str,
        prompt: &str,
    ) -> (String, axum::body::Body) {
        let body = format!(
            "--b\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}\r\n\
             --b\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\n{prompt}\r\n\
             --b\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.mp3\"\r\n\
             Content-Type: audio/mpeg\r\n\r\nID3fakeaudio\r\n--b--\r\n"
        );
        (
            "multipart/form-data; boundary=b".to_string(),
            axum::body::Body::from(body),
        )
    }

    /// #696: a mask-action PII detector must rewrite the TTS `input` text
    /// before the body reaches the upstream. Pre-#696 the mask action was a
    /// silent no-op on /v1/audio/speech.
    #[tokio::test]
    async fn speech_pii_mask_rewrites_input_before_upstream_issue_696() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/audio/speech"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(b"fakeaudio".to_vec()))
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(tts_model("my-tts"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(pii_guardrail("input"));

        let app = build_app(snap);
        let body = serde_json::json!({
            "model": "my-tts",
            "input": "read out a@x.com please",
            "voice": "alloy"
        });
        let resp = tower::ServiceExt::oneshot(app, speech_req(&body.to_string()))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let reqs = upstream.received_requests().await.unwrap();
        let sent = String::from_utf8_lossy(&reqs[0].body).into_owned();
        assert!(sent.contains("[EMAIL_REDACTED]"), "sent: {sent}");
        assert!(
            !sent.contains("a@x.com"),
            "raw PII forwarded upstream: {sent}"
        );
    }

    /// #696: the transcription `prompt` form field is caller text forwarded
    /// verbatim — an input guardrail must scan it. A blocked literal returns
    /// 422 and the upstream is never contacted.
    #[tokio::test]
    async fn transcription_prompt_field_input_guardrail_blocks_issue_696() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"text":"x"})))
            .expect(0)
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(whisper_model("my-whisper"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_input_guardrail("BLOCKME"));

        let app = build_app(snap);
        let (ct, body) = transcription_multipart_with_prompt("my-whisper", "please BLOCKME now");
        let req = Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", ct)
            .body(body)
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    /// #696: a mask-action PII detector must rewrite the transcription
    /// `prompt` form field before the form is forwarded upstream.
    #[tokio::test]
    async fn transcription_prompt_field_pii_masked_before_upstream_issue_696() {
        let upstream = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"text":"x"})))
            .expect(1)
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(whisper_model("my-whisper"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(pii_guardrail("input"));

        let app = build_app(snap);
        let (ct, body) =
            transcription_multipart_with_prompt("my-whisper", "the speaker is a@x.com");
        let req = Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", ct)
            .body(body)
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let reqs = upstream.received_requests().await.unwrap();
        let sent = String::from_utf8_lossy(&reqs[0].body).into_owned();
        assert!(sent.contains("[EMAIL_REDACTED]"), "sent: {sent}");
        assert!(
            !sent.contains("a@x.com"),
            "raw PII forwarded upstream: {sent}"
        );
    }

    /// #696: a mask-action PII detector on the OUTPUT hook must rewrite the
    /// transcript text before it reaches the caller. Pre-#696 the transcript
    /// was returned raw. Counts must land on the emitted UsageEvent.
    #[tokio::test]
    async fn transcription_output_pii_masked_issue_696() {
        let upstream = MockServer::start().await;
        let body = serde_json::json!({
            "text": "my address is a@x.com thanks",
            "usage": {"type": "tokens", "input_tokens": 9, "output_tokens": 6, "total_tokens": 15}
        });
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(whisper_model("my-whisper"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(pii_guardrail("output"));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_app_with_sink(snap, tx);
        let (ct, body) = transcription_multipart("my-whisper");
        let req = Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", ct)
            .body(body)
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let text = v["text"].as_str().unwrap();
        assert!(text.contains("[EMAIL_REDACTED]"), "client got: {text}");
        assert!(
            !text.contains("a@x.com"),
            "raw PII reached the caller: {text}"
        );

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("UsageEvent must be emitted")
            .expect("usage_sink sender dropped");
        assert_eq!(event.redacted_entity_counts.get("email"), Some(&1));
    }

    /// #696: an OUTPUT keyword guardrail must block a transcript carrying a
    /// blocked literal — the caller gets the 422 content_filter envelope,
    /// but the UsageEvent keeps the billed tokens marked guardrail_blocked
    /// (the upstream already charged for the transcription) — same
    /// convention as completions #911 [23].
    #[tokio::test]
    async fn transcription_output_guardrail_blocks_with_billed_usage_issue_696() {
        let upstream = MockServer::start().await;
        let body = serde_json::json!({
            "text": "the secret word is BLOCKME",
            "usage": {"type": "tokens", "input_tokens": 21, "output_tokens": 7, "total_tokens": 28}
        });
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(whisper_model("my-whisper"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_output_guardrail("BLOCKME"));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_app_with_sink(snap, tx);
        let (ct, body) = transcription_multipart("my-whisper");
        let req = Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", ct)
            .body(body)
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["type"], "content_filter");
        assert!(!v["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("BLOCKME"));

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("UsageEvent must be emitted for the billed-then-blocked transcript")
            .expect("usage_sink sender dropped");
        assert_eq!(event.status_code, 422);
        assert!(event.guardrail_blocked, "event must be marked blocked");
        assert_eq!(event.prompt_tokens, 21, "billed tokens must be kept");
        assert_eq!(event.completion_tokens, 7);
    }

    /// #998: `stream=true` must not become a way around the output
    /// guardrail the same transcript gets when it is not streamed. A
    /// block-capable chain keeps the buffered relay — the whole
    /// transcript is scanned before any of it reaches the caller — so the
    /// streamed request is blocked exactly like the non-streamed one.
    #[tokio::test]
    async fn streamed_transcription_is_not_an_output_guardrail_bypass() {
        let upstream = MockServer::start().await;
        let sse = concat!(
            "data: {\"type\":\"transcript.text.delta\",\"delta\":\"the secret word is \"}\n\n",
            "data: {\"type\":\"transcript.text.delta\",\"delta\":\"BLOCKME\"}\n\n",
            "data: {\"type\":\"transcript.text.done\",\"text\":\"the secret word is BLOCKME\",",
            "\"usage\":{\"type\":\"tokens\",\"input_tokens\":21,\"output_tokens\":7}}\n\n",
            "data: [DONE]\n\n",
        );
        Mock::given(method("POST"))
            .and(path("/v1/audio/transcriptions"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream"))
            .mount(&upstream)
            .await;

        let snap = new_snap(&upstream.uri());
        snap.models.insert(whisper_model("my-whisper"));
        snap.apikeys.insert(apikey_entry(&["*"]));
        snap.guardrails.insert(keyword_output_guardrail("BLOCKME"));

        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let app = build_app_with_sink(snap, tx);
        let (ct, body) = streaming_transcription_multipart("my-whisper");
        let req = Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("authorization", "Bearer sk-caller")
            .header("content-type", ct)
            .body(body)
            .unwrap();
        let resp = tower::ServiceExt::oneshot(app, req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::UNPROCESSABLE_ENTITY,
            "a blocked transcript must not be released just because it streamed"
        );
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        assert!(
            !String::from_utf8_lossy(&bytes).contains("BLOCKME"),
            "the blocked transcript must not reach the caller"
        );

        let event = tokio::time::timeout(std::time::Duration::from_millis(500), rx.recv())
            .await
            .expect("UsageEvent must be emitted for the billed-then-blocked transcript")
            .expect("usage_sink sender dropped");
        assert_eq!(event.status_code, 422);
        assert!(event.guardrail_blocked, "event must be marked blocked");
        assert_eq!((event.prompt_tokens, event.completion_tokens), (21, 7));
    }
}
