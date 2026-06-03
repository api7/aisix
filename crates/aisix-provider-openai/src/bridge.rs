//! `OpenAiBridge` — concrete [`Bridge`] implementation for OpenAI.
//!
//! Also reusable by OpenAI-compatible providers (DeepSeek, Gemini's
//! OpenAI-compat endpoint) since their request/response wire shapes
//! match the `/chat/completions` surface almost exactly. Those crates
//! can wrap this bridge or construct it with a different `api_base`.
//!
//! Transport layer:
//! - `reqwest::Client` is shared across requests (connection reuse).
//! - `Authorization: Bearer <api_key>` sourced from the
//!   [`aisix_core::Model`]'s `provider_config`.
//! - Timeout comes from `BridgeContext::deadline` when present;
//!   otherwise the request runs to completion.
//!
//! Error mapping:
//! - reqwest transport error → `BridgeError::Transport`
//! - upstream non-2xx → `BridgeError::UpstreamStatus` (4xx passes through,
//!   5xx collapses to 502 via `http_status()` on the proxy side).
//! - malformed JSON from upstream → `BridgeError::UpstreamDecode`
//! - elapsed deadline → `BridgeError::Timeout { elapsed_ms }`

use aisix_core::{RequestOverrides, ResponseOverrides, StreamDoneMarker};
use aisix_gateway::{
    Bridge, BridgeContext, BridgeError, ChatChunk, ChatChunkStream, ChatFormat, ChatResponse,
    EmbeddingRequest, EmbeddingResponse, SseDecoder, SseEvent,
};
use async_trait::async_trait;
use futures::StreamExt;
use http::{
    header::{HeaderName, HeaderValue},
    HeaderMap,
};
use reqwest::{header, Client, StatusCode};
use serde_json::Value;
use std::time::{Duration, Instant};

use crate::overrides::{
    apply_content_list_to_string, apply_default_body_fields, apply_default_headers,
    apply_param_constraints, apply_param_renames, apply_stream_done_marker_policy,
    extract_reasoning_field, StreamDoneOutcome,
};
use crate::wire::{
    build_request, embed_request_body, embed_response_into, messages_from,
    response_into_chat_response, stream_chunk_into_chat_chunk, OpenAiEmbedResponse, OpenAiResponse,
    OpenAiStreamChunk,
};

/// Default OpenAI upstream host. Used only when `ProviderKey.api_base`
/// is empty AND the dispatching PK identifies the openai vendor — the
/// family-bridge safety guard in `resolve_base` refuses this fallback
/// for any non-openai vendor (would otherwise leak the vendor's API
/// key to api.openai.com).
pub const OPENAI_DEFAULT_BASE: &str = "https://api.openai.com/v1";

/// Path suffixes the bridge appends to `api_base` when building upstream
/// URLs. If an operator accidentally pastes the full upstream URL into
/// `api_base` (e.g. `https://api.openai.com/v1/chat/completions`),
/// strip the suffix so request building still produces a valid URL.
const OPENAI_ENDPOINT_SUFFIXES: &[&str] = &[
    "/chat/completions",
    "/completions",
    "/embeddings",
    "/images/generations",
    "/audio/transcriptions",
    "/audio/translations",
    "/audio/speech",
];

pub struct OpenAiBridge {
    client: Client,
}

impl OpenAiBridge {
    pub fn new() -> Self {
        Self::with_client(default_client())
    }

    pub fn with_client(client: Client) -> Self {
        Self { client }
    }

    /// Resolve `ProviderKey.api_base` into the canonical base URL the
    /// request handlers append paths onto. Tolerates common operator
    /// mistakes:
    ///
    /// - leading and trailing whitespace
    /// - trailing slash
    /// - accidental endpoint suffix (e.g. `/chat/completions` pasted
    ///   along with the host)
    /// - bare canonical OpenAI host without `/v1` (the bridge adds it)
    ///
    /// `/v1` synthesis happens **only for the canonical OpenAI host**.
    /// Corporate proxies, alternative deployments, and any non-default
    /// path the operator chose on purpose pass through unchanged after
    /// suffix stripping — the operator's intent on a non-canonical
    /// host wins.
    ///
    /// Family-bridge safety: when the dispatching `ProviderKey.provider`
    /// identifies a vendor that ISN'T openai AND `api_base` is empty,
    /// the bridge refuses to fall back to `OPENAI_DEFAULT_BASE` — that
    /// would silently route the vendor's API key to `api.openai.com`.
    /// Closes the openrouter / xai / future-long-tail half of
    /// api7/AISIX-Cloud#417. cp-api must populate `api_base` for every
    /// catalog vendor via adapter_map / provider_metadata.api_base_url.
    fn resolve_base(&self, ctx: &BridgeContext) -> Result<String, BridgeError> {
        let raw = match ctx.provider_key.api_base.as_deref() {
            Some(b) if !b.trim().is_empty() => b.trim().to_string(),
            _ => {
                // Vendor identity is the open-string `ProviderKey.provider`.
                // Normalize (trim + ascii_lowercase) before comparing so a
                // crafted PK with `"OpenAI"` / `"openai "` cannot bypass
                // the guard.
                let pk_vendor_raw = ctx.provider_key.provider.as_str();
                let pk_vendor_normalized = pk_vendor_raw.trim().to_ascii_lowercase();
                if !pk_vendor_normalized.is_empty() && pk_vendor_normalized != "openai" {
                    // Operator-facing detail (route, provider topology,
                    // remediation steps) goes to logs only — keep the
                    // customer-visible 500 body short and free of
                    // internal-product taxonomy (cp-api / adapter_map /
                    // provider_metadata field names are not part of any
                    // wire contract a customer should depend on).
                    tracing::error!(
                        target: "aisix_provider_openai::bridge",
                        pk_display_name = %ctx.provider_key.display_name,
                        pk_vendor = %pk_vendor_raw,
                        "provider_key has no api_base; family bridge refusing fallback to \
                         api.openai.com. Operator action: populate `api_base` on the \
                         ProviderKey resource (managed deployments: via adapter_map / \
                         provider_metadata.api_base_url on the control plane; standalone: \
                         directly on the resource)."
                    );
                    return Err(BridgeError::Config(format!(
                        "provider_key for vendor {pk_vendor_raw:?} has no upstream base URL \
                         configured"
                    )));
                }
                return Ok(OPENAI_DEFAULT_BASE.to_string());
            }
        };
        Ok(normalize_api_base(&raw))
    }
}

impl Default for OpenAiBridge {
    fn default() -> Self {
        Self::new()
    }
}

fn default_client() -> Client {
    Client::builder()
        .user_agent("aisix/0.1")
        .build()
        .unwrap_or_else(|_| Client::new())
}

/// Strip a known endpoint suffix from `base`. Idempotent: if no known
/// suffix matches, returns the input with only the trailing slash trimmed.
fn strip_known_endpoint(base: &str) -> &str {
    let trimmed = base.trim_end_matches('/');
    for suffix in OPENAI_ENDPOINT_SUFFIXES {
        if let Some(rest) = trimmed.strip_suffix(suffix) {
            return rest.trim_end_matches('/');
        }
    }
    trimmed
}

/// `api_base` normalization for the OpenAI-compatible family bridge.
///
/// Normalization is intentionally **conservative**: it only
/// synthesizes the `/v1` segment when the operator pasted the bare
/// canonical `https://api.openai.com` host (a common copy-paste
/// habit). Corporate proxies, alternative deployments, and every
/// non-OpenAI vendor's upstream host pass through verbatim after
/// suffix stripping — the operator's path on a non-canonical host
/// is trusted as-is.
///
/// See [`OpenAiBridge::resolve_base`] for accepted forms.
fn normalize_api_base(base: &str) -> String {
    let stripped = strip_known_endpoint(base);
    normalize_canonical_openai(stripped)
}

/// Canonical OpenAI hosts. Both schemes covered for ops convenience —
/// in production only `https://` is meaningful.
const OPENAI_CANONICAL_HOSTS: &[&str] = &["https://api.openai.com", "http://api.openai.com"];

/// Add the canonical `/v1` segment if and only if the operator pasted
/// the bare canonical OpenAI host. Anything past the host root is left
/// as-is so non-default paths the operator chose on purpose win.
fn normalize_canonical_openai(base: &str) -> String {
    for host in OPENAI_CANONICAL_HOSTS {
        if base == *host {
            return format!("{host}/v1");
        }
    }
    base.to_string()
}

fn api_key(ctx: &BridgeContext) -> Result<&str, BridgeError> {
    let k = &ctx.provider_key.secret;
    if k.is_empty() {
        Err(BridgeError::Config("provider_key.secret is empty".into()))
    } else {
        Ok(k.as_str())
    }
}

fn upstream_model(ctx: &BridgeContext) -> Result<&str, BridgeError> {
    ctx.model
        .model_name
        .as_deref()
        .ok_or_else(|| BridgeError::Config("model.model_name missing".into()))
}

async fn map_http_error(status: StatusCode, resp: reqwest::Response) -> BridgeError {
    aisix_gateway::capture_upstream_error_http(
        status,
        resp,
        aisix_gateway::UpstreamWire::OpenAI,
        parse_openai_error_envelope,
    )
    .await
}

/// Parse the canonical OpenAI error envelope:
///
/// ```json
/// {"error": {"message": "...", "type": "...", "code": "...", "param": "..."}}
/// ```
///
/// Returns `None` when the body is not JSON of that shape; the caller
/// falls back to the truncated raw body string for the `message` field
/// and emits a generic `upstream_error` envelope.
///
/// Reference: <https://platform.openai.com/docs/guides/error-codes/api-errors>
fn parse_openai_error_envelope(body: &[u8]) -> Option<aisix_gateway::UpstreamErrorView> {
    #[derive(serde::Deserialize)]
    struct Outer {
        error: Inner,
    }
    #[derive(serde::Deserialize)]
    struct Inner {
        message: Option<String>,
        #[serde(rename = "type")]
        kind: Option<String>,
        code: Option<String>,
        param: Option<String>,
    }
    let outer: Outer = serde_json::from_slice(body).ok()?;
    Some(aisix_gateway::UpstreamErrorView {
        kind: outer.error.kind,
        message: outer.error.message,
        code: outer.error.code,
        param: outer.error.param,
    })
}

/// Wrap a future in the optional deadline. `None` → no timeout.
async fn with_deadline<T, F>(
    deadline: Option<Duration>,
    started: Instant,
    fut: F,
) -> Result<T, BridgeError>
where
    F: std::future::Future<Output = Result<T, BridgeError>>,
{
    match deadline {
        None => fut.await,
        Some(d) => match tokio::time::timeout(d, fut).await {
            Ok(r) => r,
            Err(_) => Err(BridgeError::Timeout {
                elapsed_ms: started.elapsed().as_millis() as u64,
            }),
        },
    }
}

/// Convert the typed [`build_request`] output to a `serde_json::Value`
/// and apply every [`RequestOverrides`] / [`ResponseOverrides`] field
/// whose primitive lives in [`crate::overrides`]. Issue #302 §5 lays
/// out the apply order: `param_renames` → `param_constraints` →
/// `default_body_fields` (request side), `content_list_to_string`
/// (response side flag, but it transforms the *request* body before
/// send when the upstream only accepts string content per the common
/// gateway convention). Anything not configured is a no-op.
fn prepare_outbound_body<T: serde::Serialize>(
    typed: &T,
    request: Option<&RequestOverrides>,
    response: Option<&ResponseOverrides>,
) -> Result<Value, BridgeError> {
    let mut body = serde_json::to_value(typed)
        .map_err(|e| BridgeError::Config(format!("serialize request body: {e}")))?;
    if let Some(r) = request {
        apply_param_renames(&mut body, &r.param_renames);
        if let Some(constraints) = &r.param_constraints {
            apply_param_constraints(&mut body, constraints);
        }
        apply_default_body_fields(&mut body, &r.default_body_fields);
    }
    if response.is_some_and(|r| r.content_list_to_string) {
        apply_content_list_to_string(&mut body);
    }
    Ok(body)
}

/// Build the base outbound `HeaderMap` (Authorization, Content-Type,
/// x-aisix-request-id, and optionally Accept: text/event-stream
/// for streaming calls), then merge any `default_headers` the PK carries.
/// Bridge-owned headers are inserted before the merge so
/// [`apply_default_headers`] cannot overwrite them — the
/// `if headers.contains_key(&parsed_name)` guard inside
/// `apply_default_headers` plus the [`RESERVED_DEFAULT_HEADERS`] list
/// gives two layers of defense against an operator-supplied
/// `default_headers` accidentally clobbering auth.
///
/// The previous `bridge_name` parameter + `X-Aisix-Bridge` outbound
/// header was removed in AISIX-Cloud#468: after the Phase A clean
/// cut (#375, closing AISIX-Cloud#417) most openai-family providers
/// no longer have a distinguishable per-vendor `with_name()`
/// identity, and operator-side diagnostics of which bridge served
/// a request are already covered by the DP's own `tracing::info!`
/// spans. Customer default_headers can no longer set this header
/// (per overrides.rs allowlist, also pruned), upstream APIs don't
/// read it, and the original #368 catch-Hub-typo motivation became
/// untestable for the family-keyed providers.
fn build_request_headers(
    api_key_str: &str,
    request_id: &str,
    sse: bool,
    request: Option<&RequestOverrides>,
) -> Result<HeaderMap, BridgeError> {
    let mut headers = HeaderMap::new();
    let auth = HeaderValue::from_str(&format!("Bearer {api_key_str}"))
        .map_err(|e| BridgeError::Config(format!("api key contains invalid header chars: {e}")))?;
    headers.insert(header::AUTHORIZATION, auth);
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    let rid = HeaderValue::from_str(request_id).map_err(|e| {
        BridgeError::Config(format!("request_id contains invalid header chars: {e}"))
    })?;
    headers.insert(HeaderName::from_static("x-aisix-request-id"), rid);
    if sse {
        headers.insert(
            header::ACCEPT,
            HeaderValue::from_static("text/event-stream"),
        );
    }
    if let Some(r) = request {
        apply_default_headers(&mut headers, &r.default_headers);
    }
    Ok(headers)
}

#[async_trait]
impl Bridge for OpenAiBridge {
    fn name(&self) -> &'static str {
        "openai"
    }

    async fn chat(
        &self,
        req: &ChatFormat,
        ctx: &BridgeContext,
    ) -> Result<ChatResponse, BridgeError> {
        let base = self.resolve_base(ctx)?;
        let key = api_key(ctx)?;
        let upstream = upstream_model(ctx)?;

        let messages = messages_from(req);
        let typed = build_request(req, upstream, &messages, false);
        let body = prepare_outbound_body(
            &typed,
            ctx.provider_key.request.as_ref(),
            ctx.provider_key.response.as_ref(),
        )?;
        let headers = build_request_headers(
            key,
            &ctx.request_id,
            false,
            ctx.provider_key.request.as_ref(),
        )?;
        let url = format!("{base}/chat/completions");
        let client = self.client.clone();
        let started = Instant::now();

        with_deadline(ctx.deadline, started, async move {
            let resp = client
                .post(&url)
                .headers(headers)
                .json(&body)
                .send()
                .await
                .map_err(|e| BridgeError::Transport(e.to_string()))?;

            let status = resp.status();
            if !status.is_success() {
                return Err(map_http_error(status, resp).await);
            }

            let parsed: OpenAiResponse = resp
                .json()
                .await
                .map_err(|e| BridgeError::UpstreamDecode(e.to_string()))?;
            Ok(response_into_chat_response(parsed))
        })
        .await
    }

    async fn embed(
        &self,
        req: &EmbeddingRequest,
        ctx: &BridgeContext,
    ) -> Result<EmbeddingResponse, BridgeError> {
        let base = self.resolve_base(ctx)?;
        let key = api_key(ctx)?;
        let upstream = upstream_model(ctx)?;

        let body = embed_request_body(req, upstream);
        let url = format!("{base}/embeddings");
        let client = self.client.clone();
        let started = Instant::now();
        let request_id = ctx.request_id.clone();
        with_deadline(ctx.deadline, started, async move {
            let resp = client
                .post(&url)
                .header(header::AUTHORIZATION, format!("Bearer {key}"))
                .header(header::CONTENT_TYPE, "application/json")
                .header("x-aisix-request-id", &request_id)
                .json(&body)
                .send()
                .await
                .map_err(|e| BridgeError::Transport(e.to_string()))?;

            let status = resp.status();
            if !status.is_success() {
                return Err(map_http_error(status, resp).await);
            }

            let parsed: OpenAiEmbedResponse = resp
                .json()
                .await
                .map_err(|e| BridgeError::UpstreamDecode(e.to_string()))?;
            Ok(embed_response_into(parsed))
        })
        .await
    }

    async fn complete(
        &self,
        body: &serde_json::Value,
        ctx: &BridgeContext,
    ) -> Result<serde_json::Value, BridgeError> {
        let base = self.resolve_base(ctx)?;
        let key = api_key(ctx)?;
        let upstream = upstream_model(ctx)?;

        // Replace the `model` field with the upstream provider id.
        let mut outbound = body.clone();
        if let Some(obj) = outbound.as_object_mut() {
            obj.insert(
                "model".to_string(),
                serde_json::Value::String(upstream.to_string()),
            );
        }

        let url = format!("{base}/completions");
        let client = self.client.clone();
        let started = Instant::now();
        let request_id = ctx.request_id.clone();
        with_deadline(ctx.deadline, started, async move {
            let resp = client
                .post(&url)
                .header(header::AUTHORIZATION, format!("Bearer {key}"))
                .header(header::CONTENT_TYPE, "application/json")
                .header("x-aisix-request-id", &request_id)
                .json(&outbound)
                .send()
                .await
                .map_err(|e| BridgeError::Transport(e.to_string()))?;

            let status = resp.status();
            if !status.is_success() {
                return Err(map_http_error(status, resp).await);
            }

            resp.json::<serde_json::Value>()
                .await
                .map_err(|e| BridgeError::UpstreamDecode(e.to_string()))
        })
        .await
    }

    async fn generate_image(
        &self,
        body: &serde_json::Value,
        ctx: &BridgeContext,
    ) -> Result<serde_json::Value, BridgeError> {
        let base = self.resolve_base(ctx)?;
        let key = api_key(ctx)?;
        let upstream = upstream_model(ctx)?;

        // Replace the `model` field with the upstream provider id.
        let mut outbound = body.clone();
        if let Some(obj) = outbound.as_object_mut() {
            obj.insert(
                "model".to_string(),
                serde_json::Value::String(upstream.to_string()),
            );
        }

        let url = format!("{base}/images/generations");
        let client = self.client.clone();
        let started = Instant::now();
        let request_id = ctx.request_id.clone();
        with_deadline(ctx.deadline, started, async move {
            let resp = client
                .post(&url)
                .header(header::AUTHORIZATION, format!("Bearer {key}"))
                .header(header::CONTENT_TYPE, "application/json")
                .header("x-aisix-request-id", &request_id)
                .json(&outbound)
                .send()
                .await
                .map_err(|e| BridgeError::Transport(e.to_string()))?;

            let status = resp.status();
            if !status.is_success() {
                return Err(map_http_error(status, resp).await);
            }

            resp.json::<serde_json::Value>()
                .await
                .map_err(|e| BridgeError::UpstreamDecode(e.to_string()))
        })
        .await
    }

    async fn chat_stream(
        &self,
        req: &ChatFormat,
        ctx: &BridgeContext,
    ) -> Result<ChatChunkStream, BridgeError> {
        let base = self.resolve_base(ctx)?;
        let key = api_key(ctx)?;
        let upstream = upstream_model(ctx)?;

        let messages = messages_from(req);
        let typed = build_request(req, upstream, &messages, true);
        let body = prepare_outbound_body(
            &typed,
            ctx.provider_key.request.as_ref(),
            ctx.provider_key.response.as_ref(),
        )?;
        let headers = build_request_headers(
            key,
            &ctx.request_id,
            true,
            ctx.provider_key.request.as_ref(),
        )?;
        let url = format!("{base}/chat/completions");
        let client = self.client.clone();
        let started = Instant::now();

        let resp = with_deadline(ctx.deadline, started, async move {
            client
                .post(&url)
                .headers(headers)
                .json(&body)
                .send()
                .await
                .map_err(|e| BridgeError::Transport(e.to_string()))
        })
        .await?;

        let status = resp.status();
        if !status.is_success() {
            return Err(map_http_error(status, resp).await);
        }

        // Snapshot the response-side override knobs onto the stream
        // closure. `Option<String>` for the reasoning path is cheap and
        // means the stream can run after `ctx` drops.
        let reasoning_path = ctx
            .provider_key
            .response
            .as_ref()
            .and_then(|r| r.reasoning_field.clone());
        let done_marker_policy = ctx
            .provider_key
            .response
            .as_ref()
            .and_then(|r| r.stream_done_marker);
        let bridge_name = "openai";
        let request_id_for_log = ctx.request_id.clone();

        let byte_stream = resp.bytes_stream();
        let stream = build_chunk_stream(
            byte_stream,
            reasoning_path,
            done_marker_policy,
            bridge_name,
            request_id_for_log,
        );
        Ok(Box::pin(stream))
    }
}

fn build_chunk_stream<S>(
    byte_stream: S,
    reasoning_path: Option<String>,
    done_marker_policy: Option<StreamDoneMarker>,
    bridge_name: &'static str,
    request_id: String,
) -> impl futures::Stream<Item = Result<ChatChunk, BridgeError>> + Send
where
    S: futures::Stream<Item = reqwest::Result<bytes::Bytes>> + Send + 'static,
{
    async_stream::try_stream! {
        let mut decoder = SseDecoder::new();
        let mut stream = Box::pin(byte_stream);
        let mut done_marker_seen = false;
        'outer: while let Some(next) = stream.next().await {
            let chunk = next.map_err(|e| BridgeError::Transport(e.to_string()))?;
            for event in decoder.feed(chunk.as_ref()) {
                match event {
                    SseEvent::Done => {
                        done_marker_seen = true;
                        break 'outer;
                    }
                    SseEvent::Data(payload) => {
                        let parsed = parse_stream_chunk(&payload, reasoning_path.as_deref())?;
                        yield stream_chunk_into_chat_chunk(parsed);
                    }
                }
            }
        }
        // `decoder.finish()` flushes any event sitting in the tail
        // buffer — an SSE stream that ends with `data: [DONE]\n\n`
        // gets the Done event from `feed`, but a stream that ends
        // with `data: [DONE]\n` (no trailing blank line) only surfaces
        // it here. Both forms occur in the wild (the OpenAI SDK
        // tolerates both), so we treat `finish()`-returned Done the
        // same as a feed()-returned Done.
        match decoder.finish() {
            Some(SseEvent::Done) => {
                done_marker_seen = true;
            }
            Some(SseEvent::Data(payload)) => {
                let parsed = parse_stream_chunk(&payload, reasoning_path.as_deref())?;
                yield stream_chunk_into_chat_chunk(parsed);
            }
            None => {}
        }
        // Issue #302 §5 `response.stream_done_marker` — evaluate the
        // policy once the stream ends. Violations are logged (operator
        // diagnostic) but never error the request: the customer's
        // chunks have already been delivered, and surfacing a wire-
        // shape violation now would only break a working chat.
        if let Some(policy) = done_marker_policy {
            match apply_stream_done_marker_policy(policy, done_marker_seen) {
                StreamDoneOutcome::Ok => {}
                StreamDoneOutcome::MissingDoneMarker => {
                    tracing::warn!(
                        bridge = bridge_name,
                        request_id = %request_id,
                        "upstream stream ended without [DONE] marker (policy=Required)"
                    );
                }
                StreamDoneOutcome::UnexpectedDoneMarker => {
                    tracing::warn!(
                        bridge = bridge_name,
                        request_id = %request_id,
                        "upstream emitted [DONE] marker (policy=None)"
                    );
                }
            }
        }
    }
}

/// Parse one SSE `data:` payload into [`OpenAiStreamChunk`]. When
/// `reasoning_path` is set, the parse goes typed → `Value` → mutate →
/// typed so the canonical `delta.reasoning_content` slot reflects
/// whatever the upstream put at the configured path (e.g.
/// DeepSeek's `delta.reasoning_content` is already canonical and
/// requires no lift; a hypothetical future `delta.thinking` would).
fn parse_stream_chunk(
    payload: &str,
    reasoning_path: Option<&str>,
) -> Result<OpenAiStreamChunk, BridgeError> {
    match reasoning_path {
        Some(path) => {
            let mut value: Value = serde_json::from_str(payload)
                .map_err(|e| BridgeError::UpstreamDecode(e.to_string()))?;
            extract_reasoning_field(&mut value, path);
            serde_json::from_value(value).map_err(|e| BridgeError::UpstreamDecode(e.to_string()))
        }
        None => {
            serde_json::from_str(payload).map_err(|e| BridgeError::UpstreamDecode(e.to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_core::{Model, ProviderKey};
    use aisix_gateway::{ChatMessage, FinishReason, Role};
    use std::sync::Arc;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn sample_model() -> Arc<Model> {
        Arc::new(
            serde_json::from_str(
                r#"{
                    "display_name": "my-gpt4",
                    "provider": "openai",
                    "model_name": "gpt-4o",
                    "provider_key_id": "11111111-1111-1111-1111-111111111111"
                }"#,
            )
            .unwrap(),
        )
    }

    fn sample_provider_key(base: &str) -> Arc<ProviderKey> {
        let cfg = format!(
            r#"{{"display_name": "openai-prod", "secret": "sk-test", "api_base": "{base}"}}"#
        );
        Arc::new(serde_json::from_str(&cfg).unwrap())
    }

    fn sample_ctx(base: &str) -> BridgeContext {
        BridgeContext::new("req-1", sample_model(), sample_provider_key(base))
    }

    fn req() -> ChatFormat {
        ChatFormat::new("my-gpt4", vec![ChatMessage::user("hi")])
    }

    #[tokio::test]
    async fn non_streaming_happy_path() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer sk-test"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "cmpl-1",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello back"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}
            })))
            .mount(&server)
            .await;

        let bridge = OpenAiBridge::new();
        let ctx = sample_ctx(&server.uri());
        let resp = bridge.chat(&req(), &ctx).await.unwrap();

        assert_eq!(resp.id, "cmpl-1");
        assert_eq!(resp.message.role, Role::Assistant);
        assert_eq!(resp.message.content, "hello back");
        assert_eq!(resp.finish_reason, FinishReason::Stop);
        assert_eq!(resp.usage.total_tokens, 4);
    }

    #[tokio::test]
    async fn non_streaming_429_maps_to_upstream_status() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(429).set_body_string("slow down"))
            .mount(&server)
            .await;

        let bridge = OpenAiBridge::new();
        let ctx = sample_ctx(&server.uri());
        let err = bridge.chat(&req(), &ctx).await.unwrap_err();
        match err {
            BridgeError::UpstreamStatus {
                status, message, ..
            } => {
                assert_eq!(status, 429);
                assert!(message.contains("slow down"));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    /// Audit fix (PR #323): the [`aisix_gateway::MAX_UPSTREAM_ERROR_BODY_BYTES`]
    /// (64 KB) cap must actually fire on an oversized upstream error
    /// body, otherwise a misbehaved upstream could pin a worker's
    /// memory. Pins the cap as a regression test — exercises
    /// `read_body_capped` end-to-end through the OpenAI bridge.
    #[tokio::test]
    async fn non_streaming_oversize_error_body_truncated_to_max_message_bytes() {
        let server = MockServer::start().await;
        // 200 KB body — well above the 64 KB read cap and the 1024-byte
        // message cap.
        let huge_body = "x".repeat(200 * 1024);
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(429).set_body_string(huge_body))
            .mount(&server)
            .await;

        let bridge = OpenAiBridge::new();
        let ctx = sample_ctx(&server.uri());
        let err = bridge.chat(&req(), &ctx).await.unwrap_err();
        match err {
            BridgeError::UpstreamStatus { message, .. } => {
                // Outer message must respect the 1024-byte cap + the
                // ellipsis marker. 8 bytes of slack for the ellipsis
                // (3-byte UTF-8 char) plus any partial-codepoint
                // alignment back-off.
                assert!(
                    message.len() <= aisix_gateway::MAX_UPSTREAM_ERROR_MESSAGE_BYTES + 8,
                    "outer message must be truncated; got {} bytes",
                    message.len()
                );
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    /// Audit fix (PR #323): JSON-shaped envelopes with a huge inner
    /// `error.message` must ALSO be truncated. Without this, an
    /// upstream OpenAI-shape envelope embedding a 60 KB message would
    /// bypass the cap by going through the parsed-view path. Pins
    /// HIGH-2 from the audit.
    #[tokio::test]
    async fn non_streaming_oversize_parsed_message_truncated() {
        let server = MockServer::start().await;
        let huge = "y".repeat(60 * 1024);
        let body = format!(r#"{{"error":{{"message":"{huge}","type":"big","code":"big_code"}}}}"#);
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(400).set_body_raw(body.as_bytes(), "application/json"),
            )
            .mount(&server)
            .await;

        let bridge = OpenAiBridge::new();
        let ctx = sample_ctx(&server.uri());
        let err = bridge.chat(&req(), &ctx).await.unwrap_err();
        match err {
            BridgeError::UpstreamStatus {
                message, parsed, ..
            } => {
                assert!(
                    message.len() <= aisix_gateway::MAX_UPSTREAM_ERROR_MESSAGE_BYTES + 8,
                    "outer message exceeded cap: {} bytes",
                    message.len()
                );
                let parsed = parsed.expect("envelope parsed");
                let pm = parsed.message.as_ref().expect("parsed message present");
                assert!(
                    pm.len() <= aisix_gateway::MAX_UPSTREAM_ERROR_MESSAGE_BYTES + 8,
                    "parsed.message exceeded cap: {} bytes",
                    pm.len()
                );
                assert_eq!(parsed.code.as_deref(), Some("big_code"));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[tokio::test]
    async fn non_streaming_429_surfaces_retry_after_header() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(429)
                    .insert_header("retry-after", "42")
                    .set_body_string("slow down"),
            )
            .mount(&server)
            .await;

        let bridge = OpenAiBridge::new();
        let ctx = sample_ctx(&server.uri());
        let err = bridge.chat(&req(), &ctx).await.unwrap_err();
        match err {
            BridgeError::UpstreamStatus {
                status,
                retry_after,
                ..
            } => {
                assert_eq!(status, 429);
                assert_eq!(retry_after, Some(Duration::from_secs(42)));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[tokio::test]
    async fn non_streaming_503_with_garbled_retry_after_is_none() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(503)
                    .insert_header("retry-after", "Wed, 21 Oct 2026 07:28:00 GMT")
                    .set_body_string("down"),
            )
            .mount(&server)
            .await;

        let bridge = OpenAiBridge::new();
        let ctx = sample_ctx(&server.uri());
        let err = bridge.chat(&req(), &ctx).await.unwrap_err();
        match err {
            BridgeError::UpstreamStatus {
                status,
                retry_after,
                ..
            } => {
                assert_eq!(status, 503);
                // HTTP-date form is intentionally not parsed in V1.
                assert!(retry_after.is_none());
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    // Canonical OpenAI 401 invalid_api_key error body (#543).
    const OPENAI_401_BODY: &str = r#"{"error":{"message":"Incorrect API key provided: sk-inval***c66a. You can find your API key at https://platform.openai.com/account/api-keys.","type":"invalid_request_error","code":"invalid_api_key","param":null}}"#;

    /// Baseline: a 401 whose JSON error body is labelled
    /// `application/json` parses correctly — `code` reaches the
    /// envelope. (Already worked pre-#543.)
    #[tokio::test]
    async fn non_streaming_401_json_content_type_surfaces_code() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(401).set_body_raw(
                OPENAI_401_BODY.as_bytes(),
                "application/json; charset=utf-8",
            ))
            .mount(&server)
            .await;
        let bridge = OpenAiBridge::new();
        let ctx = sample_ctx(&server.uri());
        let err = bridge.chat(&req(), &ctx).await.unwrap_err();
        match err {
            BridgeError::UpstreamStatus { parsed, .. } => {
                assert_eq!(
                    parsed.and_then(|p| p.code),
                    Some("invalid_api_key".to_string())
                );
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    /// Issue #543: the SAME JSON error body labelled with a
    /// non-`application/json` Content-Type (as OpenAI's 401 path / an
    /// edge layer returns it) must STILL surface `code` / `param`.
    /// Pre-fix the Content-Type gate skipped the parse, dumping the raw
    /// body into `message` and emitting an empty `code`.
    #[tokio::test]
    async fn non_streaming_401_non_json_content_type_still_surfaces_code() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(401).set_body_raw(OPENAI_401_BODY.as_bytes(), "text/plain"),
            )
            .mount(&server)
            .await;
        let bridge = OpenAiBridge::new();
        let ctx = sample_ctx(&server.uri());
        let err = bridge.chat(&req(), &ctx).await.unwrap_err();
        match err {
            BridgeError::UpstreamStatus {
                parsed, message, ..
            } => {
                let parsed = parsed.expect("error envelope must parse regardless of Content-Type");
                assert_eq!(parsed.code.as_deref(), Some("invalid_api_key"));
                assert_eq!(parsed.kind.as_deref(), Some("invalid_request_error"));
                // `message` must be the clean upstream message, NOT the
                // raw JSON-stringified body.
                assert!(
                    message.starts_with("Incorrect API key provided"),
                    "message must be the parsed upstream message, got: {message}",
                );
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    /// A genuinely non-JSON error body (HTML / plain text, no
    /// `{"error":...}`) must still fall back cleanly — parse returns
    /// None, no panic. Guards the opportunistic-parse change against
    /// over-parsing.
    #[tokio::test]
    async fn non_streaming_non_json_error_body_falls_back() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(502)
                    .set_body_raw(b"<html>502 Bad Gateway</html>", "text/html"),
            )
            .mount(&server)
            .await;
        let bridge = OpenAiBridge::new();
        let ctx = sample_ctx(&server.uri());
        let err = bridge.chat(&req(), &ctx).await.unwrap_err();
        match err {
            BridgeError::UpstreamStatus { parsed, status, .. } => {
                assert_eq!(status, 502);
                assert!(parsed.is_none(), "non-JSON body must not parse into a view");
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[tokio::test]
    async fn non_streaming_decode_error_on_malformed_body() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not-json"))
            .mount(&server)
            .await;

        let bridge = OpenAiBridge::new();
        let ctx = sample_ctx(&server.uri());
        let err = bridge.chat(&req(), &ctx).await.unwrap_err();
        assert!(matches!(err, BridgeError::UpstreamDecode(_)));
    }

    #[tokio::test]
    async fn deadline_elapses_to_timeout_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_delay(Duration::from_secs(5))
                    .set_body_json(serde_json::json!({"id":"x","model":"x","choices":[]})),
            )
            .mount(&server)
            .await;

        let bridge = OpenAiBridge::new();
        let ctx = sample_ctx(&server.uri()).with_deadline(Duration::from_millis(50));
        let err = bridge.chat(&req(), &ctx).await.unwrap_err();
        assert!(matches!(err, BridgeError::Timeout { .. }));
    }

    #[tokio::test]
    async fn missing_api_key_is_a_config_error() {
        // Bridge's own guard: ProviderKey with an empty `secret` must
        // surface a Config error rather than calling upstream with a
        // bare bearer.
        let mut pk: ProviderKey =
            serde_json::from_str(r#"{"display_name":"empty","secret":"placeholder"}"#).unwrap();
        pk.secret.clear();

        let bridge = OpenAiBridge::new();
        let ctx = BridgeContext::new("req-1", sample_model(), Arc::new(pk));
        let err = bridge.chat(&req(), &ctx).await.unwrap_err();
        assert!(matches!(err, BridgeError::Config(_)));
    }

    #[tokio::test]
    async fn streaming_happy_path_emits_chunks_then_done() {
        let server = MockServer::start().await;
        let sse = "\
data: {\"id\":\"cmpl-s\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"cmpl-s\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hel\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"cmpl-s\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\n\
data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&server)
            .await;

        let bridge = OpenAiBridge::new();
        let ctx = sample_ctx(&server.uri());
        let mut stream = bridge.chat_stream(&req(), &ctx).await.unwrap();

        let mut chunks = Vec::new();
        while let Some(item) = stream.next().await {
            chunks.push(item.unwrap());
        }
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].delta.role, Some(Role::Assistant));
        assert_eq!(chunks[1].delta.content.as_deref(), Some("hel"));
        assert_eq!(chunks[2].delta.content.as_deref(), Some("lo"));
        assert_eq!(chunks[2].finish_reason, Some(FinishReason::Stop));
    }

    #[tokio::test]
    async fn streaming_upstream_error_surfaces_before_stream_start() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(500).set_body_string("oops"))
            .mount(&server)
            .await;

        let bridge = OpenAiBridge::new();
        let ctx = sample_ctx(&server.uri());
        match bridge.chat_stream(&req(), &ctx).await {
            Ok(_) => panic!("expected upstream error, got a live stream"),
            Err(BridgeError::UpstreamStatus { status: 500, .. }) => {}
            Err(other) => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn resolve_base_trims_trailing_slash_and_honours_override() {
        let bridge = OpenAiBridge::new();

        // No api_base set → falls back to OPENAI_DEFAULT_BASE.
        let pk_default: ProviderKey =
            serde_json::from_str(r#"{"display_name":"x","secret":"k"}"#).unwrap();
        let ctx = BridgeContext::new("rid", sample_model(), Arc::new(pk_default));
        assert_eq!(bridge.resolve_base(&ctx).unwrap(), OPENAI_DEFAULT_BASE);

        // api_base override: trailing slash stripped.
        let pk_override: ProviderKey = serde_json::from_str(
            r#"{"display_name":"x","secret":"k","api_base":"https://proxy.example.com/v1/"}"#,
        )
        .unwrap();
        let ctx = BridgeContext::new("rid", sample_model(), Arc::new(pk_override));
        assert_eq!(
            bridge.resolve_base(&ctx).unwrap(),
            "https://proxy.example.com/v1"
        );
    }

    fn pk_with_base(api_base: &str) -> ProviderKey {
        let cfg = format!(r#"{{"display_name":"x","secret":"k","api_base":"{api_base}"}}"#);
        serde_json::from_str(&cfg).unwrap()
    }

    /// `OpenAiBridge::new()` is the `Adapter::Openai` family bridge.
    /// When it serves a non-openai vendor with empty `api_base` it
    /// MUST refuse rather than fall back to `OPENAI_DEFAULT_BASE` —
    /// that fallback would silently route the vendor's API key to
    /// `api.openai.com`. Closes the openrouter / xai half of
    /// api7/AISIX-Cloud#417.
    #[test]
    fn family_bridge_refuses_non_openai_vendor_with_empty_api_base() {
        let bridge = OpenAiBridge::new();
        // Vendor list spans:
        // - Long-tail openai-compat catalog vendors that always need a
        //   populated `api_base` (xai, openrouter, groq, mistral,
        //   perplexity).
        // - The three vendors that #379's clean cut deleted dedicated
        //   `register_specialized` entries for, so they now route only
        //   through the family bridge (google, deepseek, cohere) — the
        //   bridge must refuse if cp-api ever ships them with empty
        //   `api_base`.
        // - Cased + whitespace variants to pin that the guard does
        //   trim + lowercase normalization before matching `"openai"`.
        for vendor in [
            "openrouter",
            "xai",
            "groq",
            "mistral",
            "perplexity",
            "google",
            "deepseek",
            "cohere",
            "OpenRouter",
            " openrouter ",
        ] {
            let pk: ProviderKey = serde_json::from_str(&format!(
                r#"{{"display_name":"x","secret":"k","provider":"{vendor}","adapter":"openai"}}"#
            ))
            .unwrap();
            let ctx = BridgeContext::new("rid", sample_model(), Arc::new(pk));
            let err = bridge.resolve_base(&ctx).unwrap_err();
            match err {
                BridgeError::Config(msg) => {
                    assert!(
                        msg.contains("base URL") && msg.contains(vendor.trim()),
                        "vendor {vendor:?}: error must name vendor + base URL; got: {msg}",
                    );
                    // Sensitive-info-leakage guard: internal product
                    // taxonomy must not leak into the customer-visible
                    // 500 body. Those identifiers go to tracing only.
                    for forbidden in ["cp-api", "adapter_map", "provider_metadata"] {
                        assert!(
                            !msg.contains(forbidden),
                            "vendor {vendor:?}: error body must not leak \
                             internal-product taxonomy {forbidden:?}; got: {msg}",
                        );
                    }
                }
                other => panic!("vendor {vendor:?}: expected BridgeError::Config, got {other:?}"),
            }
        }
    }

    /// Pure-openai PK without `api_base` falls back to
    /// `OPENAI_DEFAULT_BASE` — the historical legacy behavior. The
    /// safety check above only fires for non-openai vendors.
    #[test]
    fn family_bridge_allows_openai_vendor_with_empty_api_base() {
        let bridge = OpenAiBridge::new();
        let pk: ProviderKey = serde_json::from_str(
            r#"{"display_name":"oai","secret":"sk-oai","provider":"openai","adapter":"openai"}"#,
        )
        .unwrap();
        let ctx = BridgeContext::new("rid", sample_model(), Arc::new(pk));
        assert_eq!(bridge.resolve_base(&ctx).unwrap(), OPENAI_DEFAULT_BASE);
    }

    /// Pre-Phase-A PK carries an empty `provider` string. The safety
    /// check must NOT fire here — those rows route via the compat
    /// shim in `crates/aisix-proxy/src/dispatch.rs::resolve_bridge`
    /// to the specialized "openai" bridge. The bridge itself must
    /// tolerate the legacy shape so the compat path doesn't 500.
    #[test]
    fn family_bridge_allows_legacy_empty_provider_with_empty_api_base() {
        let bridge = OpenAiBridge::new();
        let pk: ProviderKey = serde_json::from_str(r#"{"display_name":"x","secret":"k"}"#).unwrap();
        let ctx = BridgeContext::new("rid", sample_model(), Arc::new(pk));
        assert_eq!(bridge.resolve_base(&ctx).unwrap(), OPENAI_DEFAULT_BASE);
    }

    /// xai happy path: cp-api populates `api_base` from the catalog
    /// row (`provider_metadata.api_base_url` or adapter_map
    /// `default_base_url`), so the family bridge sees a populated
    /// base and dispatches normally to the vendor's upstream.
    #[test]
    fn family_bridge_allows_non_openai_vendor_with_populated_api_base() {
        let bridge = OpenAiBridge::new();
        let pk: ProviderKey = serde_json::from_str(
            r#"{"display_name":"xai","secret":"sk-xai","provider":"xai","adapter":"openai","api_base":"https://api.x.ai/v1"}"#,
        )
        .unwrap();
        let ctx = BridgeContext::new("rid", sample_model(), Arc::new(pk));
        assert_eq!(bridge.resolve_base(&ctx).unwrap(), "https://api.x.ai/v1");
    }

    /// All three OpenAI api_base forms a real operator might paste must
    /// resolve to the same canonical `<host>/v1`.
    #[test]
    fn openai_api_base_tolerance_bare_host_v1_and_full_endpoint() {
        let bridge = OpenAiBridge::new();
        let canonical = "https://api.openai.com/v1";

        for form in [
            "https://api.openai.com",
            "https://api.openai.com/",
            "https://api.openai.com/v1",
            "https://api.openai.com/v1/",
            "https://api.openai.com/v1/chat/completions",
            "https://api.openai.com/v1/embeddings",
            "  https://api.openai.com/v1  ",
        ] {
            let ctx = BridgeContext::new("rid", sample_model(), Arc::new(pk_with_base(form)));
            assert_eq!(
                bridge.resolve_base(&ctx).unwrap(),
                canonical,
                "form {form:?} should normalize to {canonical}",
            );
        }
    }

    /// A non-canonical host (corporate proxy, alternative deployment,
    /// any vendor that isn't api.openai.com) passes through verbatim
    /// after endpoint-suffix stripping. The family bridge no longer
    /// synthesizes vendor-specific URL prefixes — operators paste the
    /// documented URL on the dashboard.
    #[test]
    fn non_openai_host_passes_through_verbatim_after_suffix_strip() {
        let bridge = OpenAiBridge::new();

        // Endpoint suffix is stripped on any host.
        let with_suffix =
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions";
        let canonical = "https://generativelanguage.googleapis.com/v1beta/openai";
        let ctx = BridgeContext::new("rid", sample_model(), Arc::new(pk_with_base(with_suffix)));
        assert_eq!(bridge.resolve_base(&ctx).unwrap(), canonical);

        // Corporate proxy: pass-through.
        let custom = "https://corporate-proxy.acme.internal/groq";
        let ctx = BridgeContext::new("rid", sample_model(), Arc::new(pk_with_base(custom)));
        assert_eq!(bridge.resolve_base(&ctx).unwrap(), custom);
    }

    // ----------- issue #302 §5 wire-in: RequestOverrides -----------
    //
    // The matcher-based assertion pattern: a Mock that returns 200
    // only when the matcher passes. If the bridge sends a wrong
    // body / wrong header, the matcher fails, wiremock falls through
    // with a default 404, and `bridge.chat(...).unwrap()` panics. So
    // "the call succeeded" == "the bridge sent what we wanted".

    fn pk_with_overrides(base: &str, overrides_json: &str) -> Arc<ProviderKey> {
        let cfg = format!(
            r#"{{"display_name": "openai-prod", "secret": "sk-test", "api_base": "{base}", {overrides_json}}}"#
        );
        Arc::new(serde_json::from_str(&cfg).unwrap())
    }

    fn ok_chat_response() -> serde_json::Value {
        serde_json::json!({
            "id": "cmpl-1",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        })
    }

    /// `param_renames` must rewrite the outbound body key. Build a
    /// request with `max_tokens=64`; configure rename
    /// `max_tokens → max_completion_tokens`; the bytes leaving the
    /// bridge must carry the renamed key (source-wins convention).
    #[tokio::test]
    async fn chat_applies_param_renames_to_outbound_body() {
        use wiremock::matchers::body_partial_json;
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(body_partial_json(serde_json::json!({
                "max_completion_tokens": 64,
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(ok_chat_response()))
            .mount(&server)
            .await;

        let bridge = OpenAiBridge::new();
        let pk = pk_with_overrides(
            &server.uri(),
            r#""request": {"param_renames": {"max_tokens": "max_completion_tokens"}}"#,
        );
        let ctx = BridgeContext::new("req-1", sample_model(), pk);
        let mut request = ChatFormat::new("my-gpt4", vec![ChatMessage::user("hi")]);
        request.max_tokens = Some(64);
        bridge
            .chat(&request, &ctx)
            .await
            .expect("matcher pinned the renamed key");
    }

    /// `param_constraints.temperature_max` clamps the outbound value.
    /// Customer sends `temperature=2.0`; max=1.0; outbound body must
    /// carry `temperature=1.0`. Negative axis: a `temperature_min`
    /// clamps from below.
    #[tokio::test]
    async fn chat_applies_param_constraints_clamp_to_outbound_body() {
        use wiremock::matchers::body_partial_json;
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(body_partial_json(serde_json::json!({"temperature": 1.0})))
            .respond_with(ResponseTemplate::new(200).set_body_json(ok_chat_response()))
            .mount(&server)
            .await;

        let bridge = OpenAiBridge::new();
        let pk = pk_with_overrides(
            &server.uri(),
            r#""request": {"param_constraints": {"temperature_max": 1.0}}"#,
        );
        let ctx = BridgeContext::new("req-1", sample_model(), pk);
        let mut request = ChatFormat::new("my-gpt4", vec![ChatMessage::user("hi")]);
        request.temperature = Some(2.0);
        bridge
            .chat(&request, &ctx)
            .await
            .expect("matcher pinned the clamped value");
    }

    /// `default_body_fields` fills absent top-level keys without
    /// overwriting caller-set ones. Configure `safe_prompt=true`;
    /// the outbound body must carry it.
    #[tokio::test]
    async fn chat_applies_default_body_fields_to_outbound_body() {
        use wiremock::matchers::body_partial_json;
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(body_partial_json(serde_json::json!({"safe_prompt": true})))
            .respond_with(ResponseTemplate::new(200).set_body_json(ok_chat_response()))
            .mount(&server)
            .await;

        let bridge = OpenAiBridge::new();
        let pk = pk_with_overrides(
            &server.uri(),
            r#""request": {"default_body_fields": {"safe_prompt": true}}"#,
        );
        let ctx = BridgeContext::new("req-1", sample_model(), pk);
        bridge
            .chat(&req(), &ctx)
            .await
            .expect("matcher pinned default_body_fields");
    }

    /// `default_headers` adds operator-supplied headers to the
    /// outbound request. Configure `x-custom: trace-on`; the request
    /// must carry it.
    #[tokio::test]
    async fn chat_applies_default_headers_to_outbound_request() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("x-custom", "trace-on"))
            .respond_with(ResponseTemplate::new(200).set_body_json(ok_chat_response()))
            .mount(&server)
            .await;

        let bridge = OpenAiBridge::new();
        let pk = pk_with_overrides(
            &server.uri(),
            r#""request": {"default_headers": {"x-custom": "trace-on"}}"#,
        );
        let ctx = BridgeContext::new("req-1", sample_model(), pk);
        bridge
            .chat(&req(), &ctx)
            .await
            .expect("matcher pinned the operator header");
    }

    /// The outbound request must carry `Bearer sk-test`, not the
    /// override value.
    #[tokio::test]
    async fn chat_default_headers_cannot_override_authorization() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer sk-test"))
            .respond_with(ResponseTemplate::new(200).set_body_json(ok_chat_response()))
            .mount(&server)
            .await;

        let bridge = OpenAiBridge::new();
        let pk = pk_with_overrides(
            &server.uri(),
            r#""request": {"default_headers": {"authorization": "Bearer evil-override"}}"#,
        );
        let ctx = BridgeContext::new("req-1", sample_model(), pk);
        bridge
            .chat(&req(), &ctx)
            .await
            .expect("auth was kept as-is");
    }

    // ----------- issue #302 §5 wire-in: ResponseOverrides ----------

    /// `content_list_to_string=true` flattens the request body's
    /// `messages[*].content` array of text blocks into a string
    /// before send (common gateway convention — applies to the request).
    /// The outbound body must carry `content: "abc"`, not the array.
    #[tokio::test]
    async fn chat_content_list_to_string_flattens_text_blocks_in_outbound_body() {
        use wiremock::matchers::body_partial_json;
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(body_partial_json(serde_json::json!({
                "messages": [{"role": "user", "content": "abc"}]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(ok_chat_response()))
            .mount(&server)
            .await;

        let bridge = OpenAiBridge::new();
        let pk = pk_with_overrides(
            &server.uri(),
            r#""response": {"content_list_to_string": true}"#,
        );
        let ctx = BridgeContext::new("req-1", sample_model(), pk);
        // Multi-block message: ["a", "b", "c"] → "abc" after flatten.
        let mut msg = ChatMessage::user("abc");
        msg.content_blocks = Some(vec![
            serde_json::json!({"type": "text", "text": "a"}),
            serde_json::json!({"type": "text", "text": "b"}),
            serde_json::json!({"type": "text", "text": "c"}),
        ]);
        let request = ChatFormat::new("my-gpt4", vec![msg]);
        bridge
            .chat(&request, &ctx)
            .await
            .expect("matcher pinned the flattened string");
    }

    /// `reasoning_field` lifts a vendor-specific path on streaming
    /// chunks up to the canonical `delta.reasoning_content` slot.
    /// Upstream emits `delta.thinking="step1"`; with reasoning_field
    /// path `delta.thinking`, the parsed chunk's `delta` must carry
    /// `reasoning_content="step1"` for the downstream emitter.
    #[tokio::test]
    async fn chat_stream_extracts_reasoning_field_onto_canonical_slot() {
        let server = MockServer::start().await;
        // Upstream chunk uses `delta.thinking` (hypothetical vendor
        // shape); after extract_reasoning_field the parsed chunk's
        // delta should have reasoning_content="step1".
        let sse = "\
data: {\"id\":\"cmpl-r\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"thinking\":\"step1\"},\"finish_reason\":null}]}\n\n\
data: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&server)
            .await;

        let bridge = OpenAiBridge::new();
        let pk = pk_with_overrides(
            &server.uri(),
            r#""response": {"reasoning_field": "delta.thinking"}"#,
        );
        let ctx = BridgeContext::new("req-1", sample_model(), pk);
        let mut stream = bridge.chat_stream(&req(), &ctx).await.unwrap();
        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(
            first.delta.reasoning_content.as_deref(),
            Some("step1"),
            "extract_reasoning_field must lift delta.thinking onto delta.reasoning_content"
        );
    }

    /// `stream_done_marker` policy is evaluated but never errors the
    /// stream — the contract is "log-only" so a working chat is not
    /// broken by a wire-shape diagnostic. The test asserts that
    /// (a) the stream completes successfully, and (b) all chunks
    /// are delivered, even when policy=Required and the upstream
    /// omitted the `[DONE]` marker.
    #[tokio::test]
    async fn chat_stream_done_marker_required_but_missing_does_not_error() {
        let server = MockServer::start().await;
        // No `data: [DONE]` line — Required policy will fire a
        // MissingDoneMarker warning, but the stream must still
        // complete cleanly.
        let sse = "\
data: {\"id\":\"cmpl-s\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n";
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&server)
            .await;

        let bridge = OpenAiBridge::new();
        let pk = pk_with_overrides(
            &server.uri(),
            r#""response": {"stream_done_marker": "required"}"#,
        );
        let ctx = BridgeContext::new("req-1", sample_model(), pk);
        let mut stream = bridge.chat_stream(&req(), &ctx).await.unwrap();
        let mut chunks = Vec::new();
        while let Some(item) = stream.next().await {
            chunks.push(item.expect("chunk must not error on missing DONE"));
        }
        assert_eq!(chunks.len(), 1, "all chunks delivered before policy check");
    }

    /// Negative axis on the parse path: a malformed SSE payload still
    /// surfaces as `UpstreamDecode` after the reasoning-field roundtrip
    /// — the typed↔Value detour must not swallow parse errors.
    #[tokio::test]
    async fn chat_stream_malformed_chunk_surfaces_decode_error_via_reasoning_path() {
        let server = MockServer::start().await;
        let sse = "data: not-json\n\ndata: [DONE]\n\n";
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&server)
            .await;

        let bridge = OpenAiBridge::new();
        let pk = pk_with_overrides(
            &server.uri(),
            r#""response": {"reasoning_field": "delta.thinking"}"#,
        );
        let ctx = BridgeContext::new("req-1", sample_model(), pk);
        let mut stream = bridge.chat_stream(&req(), &ctx).await.unwrap();
        let first = stream.next().await.unwrap();
        assert!(matches!(first, Err(BridgeError::UpstreamDecode(_))));
    }

    /// Audit regression: a `data: [DONE]` line without the trailing
    /// blank line is held in `SseDecoder`'s tail buffer and only
    /// surfaces via `decoder.finish()`. The bridge must treat that as
    /// "the marker was seen" — otherwise a policy=Required stream that
    /// happened to omit the trailing blank line would log a false-
    /// positive MissingDoneMarker warning. The customer-visible
    /// behavior we pin here: the chunks are yielded AND the stream
    /// completes cleanly without a spurious warning being the only
    /// observable. (We can't easily assert on tracing output without
    /// adding a dev-dep, so we lock in the next-best contract: chunks
    /// arrive intact.)
    #[tokio::test]
    async fn chat_stream_done_marker_in_decoder_finish_tail_is_counted_as_seen() {
        let server = MockServer::start().await;
        // Note: NO trailing `\n\n` after [DONE] — exercises the
        // decoder.finish() flush path rather than feed().
        let sse = "\
data: {\"id\":\"cmpl-s\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n\
data: [DONE]\n";
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&server)
            .await;

        let bridge = OpenAiBridge::new();
        let pk = pk_with_overrides(
            &server.uri(),
            r#""response": {"stream_done_marker": "required"}"#,
        );
        let ctx = BridgeContext::new("req-1", sample_model(), pk);
        let mut stream = bridge.chat_stream(&req(), &ctx).await.unwrap();
        let mut chunks = Vec::new();
        while let Some(item) = stream.next().await {
            chunks.push(item.expect("chunk must not error"));
        }
        assert_eq!(
            chunks.len(),
            1,
            "chunk before [DONE] must still be delivered"
        );
    }

    /// Backward-compat: a ProviderKey with no `request` / `response`
    /// block (most production rows today) must behave identically to
    /// the pre-wire-in bridge. Sanity-check that the body still goes
    /// through and the response parses.
    #[tokio::test]
    async fn chat_with_no_overrides_behaves_like_pre_d2() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer sk-test"))
            .and(header("x-aisix-request-id", "req-1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(ok_chat_response()))
            .mount(&server)
            .await;

        let bridge = OpenAiBridge::new();
        let ctx = sample_ctx(&server.uri());
        let resp = bridge.chat(&req(), &ctx).await.unwrap();
        assert_eq!(resp.id, "cmpl-1");
    }

    // The previous "issue #368: X-AISIX-Bridge outbound header" block
    // of tests was removed along with the header insertion itself:
    // Phase A clean cut (#375 closing AISIX-Cloud#417) made per-vendor
    // `Hub.register(Provider::X, OpenAiBridge::new().with_name("Y"))`
    // an empty contract for openai-family long-tail providers, and
    // the operator-side bridge identity is already emitted via the
    // `tracing::info!` spans in `build_chunk_stream` / dispatch sites
    // (the `bridge=<name>` field on the span). Tests removed:
    //   - chat_default_headers_cannot_override_x_aisix_bridge
    //   - chat_emits_x_aisix_bridge_openai_for_default_bridge
    //   - chat_emits_x_aisix_bridge_for_with_name_variant
    //   - chat_stream_emits_x_aisix_bridge_for_with_name_variant
}
