//! The [`Bridge`] trait — what every provider crate implements.
//!
//! A Bridge is the provider-specific adapter between the gateway's
//! normalised [`ChatFormat`] and whichever upstream API shape the vendor
//! requires. Bridges are held in [`crate::hub::Hub`] and selected by the
//! Model's [`aisix_core::Provider`] enum.
//!
//! Responsibilities of a Bridge:
//! - Translate `ChatFormat` → upstream request body
//! - Perform the HTTP call (authorisation, timeouts, retries at transport)
//! - For streaming requests, produce a `Stream<Item = ChatChunk>`
//! - For non-streaming, produce a full [`ChatResponse`]
//! - Surface errors as typed [`BridgeError`] variants so the proxy layer
//!   can map them to consistent OpenAI-style error envelopes
//!
//! The trait is deliberately `async_trait` rather than GATs — ergonomic
//! wins outweigh the boxing cost on the provider path.

use aisix_core::{Model, ProviderKey};
use async_trait::async_trait;
use futures::stream::BoxStream;
use std::time::Duration;

use crate::chat::{ChatChunk, ChatFormat, ChatResponse, EmbeddingRequest, EmbeddingResponse};

/// Maximum number of bytes read from an upstream error response body
/// before attempting JSON envelope parse. Bounds memory and parser cost
/// when an upstream returns something pathological (an HTML error page
/// from a fronting WAF, or an unexpectedly large debug dump).
pub const MAX_UPSTREAM_ERROR_BODY_BYTES: usize = 64 * 1024;

/// Maximum length of the human-readable `message` string carried inside
/// [`BridgeError::UpstreamStatus`]. The full body is parsed into
/// [`UpstreamErrorView`] when JSON-shaped; the truncated string is the
/// fallback shown to clients when parsing fails.
pub const MAX_UPSTREAM_ERROR_MESSAGE_BYTES: usize = 1024;

/// Which wire format the upstream that produced this error speaks. The
/// envelope-rendering layer uses this together with [`UpstreamErrorView`]
/// to decide whether the upstream `kind` / `code` can be forwarded
/// verbatim or needs translation to the client's wire shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpstreamWire {
    /// OpenAI-compatible envelope: `{error:{message,type,code,param}}`.
    OpenAI,
    /// Anthropic envelope: `{type:"error",error:{type,message}}`.
    Anthropic,
    /// Azure OpenAI envelope: OpenAI-like with `error.inner_error.code`
    /// quirks for content policy violations.
    AzureOpenAI,
    /// AWS Bedrock structured error from the strongly-typed SDK; `kind`
    /// carries the AWS exception code (e.g. `"ThrottlingException"`).
    Bedrock,
    /// Vertex AI envelope: `{error:{code:int,message,status}}` where
    /// `status` is the canonical gRPC code string.
    Vertex,
    /// Wire format unknown / not applicable (tests, synthesised errors,
    /// the legacy convenience constructors). Renders as the generic
    /// `upstream_error` envelope with no translation attempt.
    Unknown,
}

/// Structured view of an upstream error envelope, populated by each
/// bridge after best-effort parsing of its provider's known shape.
/// `None` everywhere means parsing failed (non-JSON body, malformed
/// JSON, or unfamiliar envelope shape); callers fall back to the
/// truncated raw message on [`BridgeError::UpstreamStatus::message`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UpstreamErrorView {
    /// Provider-native error-type token, unchanged from the upstream
    /// envelope (e.g. Anthropic `"rate_limit_error"`, OpenAI
    /// `"rate_limit_exceeded"`, Bedrock `"ThrottlingException"`).
    pub kind: Option<String>,
    /// Human-readable upstream message, post-parse.
    pub message: Option<String>,
    /// OpenAI envelope only. Other providers populate via the
    /// translation table at render time.
    pub code: Option<String>,
    /// OpenAI envelope only.
    pub param: Option<String>,
}

/// Context carried through the whole request lifecycle.
///
/// The proxy layer fills this in after it has authenticated the request
/// and resolved both the target Model AND its referenced ProviderKey
/// from the [`aisix_core::AisixSnapshot`]. Bridges read from it but
/// do not mutate it.
#[derive(Debug, Clone)]
pub struct BridgeContext {
    /// Correlation id propagated into traces and error envelopes.
    pub request_id: String,
    /// The resolved Model — bridges read `model_name` (the upstream
    /// model id) and metadata (timeout, rate_limit) from here.
    pub model: std::sync::Arc<Model>,
    /// The ProviderKey the Model references — bridges read `secret`
    /// (api key) and `api_base` (optional override) from here.
    pub provider_key: std::sync::Arc<ProviderKey>,
    /// Deadline for the entire upstream call. Bridges are expected to
    /// honour this by cancelling any in-flight HTTP request.
    pub deadline: Option<Duration>,
}

impl BridgeContext {
    pub fn new(
        request_id: impl Into<String>,
        model: std::sync::Arc<Model>,
        provider_key: std::sync::Arc<ProviderKey>,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            model,
            provider_key,
            deadline: None,
        }
    }

    pub fn with_deadline(mut self, deadline: Duration) -> Self {
        self.deadline = Some(deadline);
        self
    }
}

/// `": {cause}"` when a transport-layer cause is known, otherwise empty —
/// keeps the timeout message unchanged for the gateway's own deadlines.
fn timeout_cause_suffix(cause: &str) -> String {
    if cause.is_empty() {
        String::new()
    } else {
        format!(": {cause}")
    }
}

/// Error surfaced by any Bridge. Each variant maps to a stable
/// client-visible HTTP status and OpenAI-style error code so the proxy
/// layer can translate without further inspection.
#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    /// An upstream call exceeded a time budget.
    ///
    /// `cause` names the transport-layer reason when reqwest reported one,
    /// and is empty when one of the gateway's own deadlines elapsed. Three
    /// unrelated conditions all satisfy `reqwest::Error::is_timeout()` — a
    /// `connect_timeout`, hyper's request timeout, and the kernel's own
    /// `ETIMEDOUT` on an unanswered SYN — so without the cause they render
    /// as the same sentence and an operator cannot tell "the upstream is
    /// slow" from "we never reached it" (AISIX-Cloud#1093).
    #[error("upstream request timed out after {elapsed_ms}ms{}", timeout_cause_suffix(.cause))]
    Timeout { elapsed_ms: u64, cause: String },
    /// Upstream returned a non-2xx HTTP status. `retry_after` carries
    /// the upstream's `Retry-After` header parsed to a Duration when
    /// present — used by the cooldown layer to honor provider-supplied
    /// backoff hints. Bridges that cannot parse the header (or where
    /// the header is absent) leave this `None`; the cooldown layer
    /// falls back to its configured default in that case.
    /// `message` is a best-effort human-readable string for logs and
    /// the fallback envelope when [`parsed`] is `None`. When [`parsed`]
    /// is `Some`, the envelope-rendering layer (`error_translate`) uses
    /// the structured fields and [`wire`] to produce a client-shape
    /// envelope; `message` is kept around for logs and as a
    /// last-resort fallback if a parsed field is missing.
    #[error("upstream returned HTTP {status}: {message}")]
    UpstreamStatus {
        status: u16,
        message: String,
        /// Boxed to keep [`BridgeError`] small enough that
        /// `Result<_, ProxyError>` doesn't trip `clippy::result_large_err`
        /// once the four optional envelope fields are added.
        parsed: Option<Box<UpstreamErrorView>>,
        wire: UpstreamWire,
        retry_after: Option<Duration>,
    },
    #[error("upstream returned an unparseable body: {0}")]
    UpstreamDecode(String),
    #[error("bridge is misconfigured: {0}")]
    Config(String),
    /// Customer-fixable upstream config — the admin's ProviderKey/Model
    /// is set up wrong (missing api_base, missing model_name) or the
    /// caller's request is malformed (e.g. split_system shape). Maps to
    /// 400, not 500: it's the caller's mistake, retrying won't help, and
    /// a 5xx wrongly tells SDKs/monitoring it's a server fault (#367).
    /// Contrast [`Config`], reserved for errors *we* cause
    /// (serialization, our generated request_id) which stays 500.
    #[error("invalid upstream configuration: {0}")]
    InvalidUpstreamConfig(String),
    /// Customer-fixable upstream *credential* problem — the admin's
    /// ProviderKey secret/credential is missing, empty, or malformed
    /// (empty secret, api key with invalid HTTP-header bytes, unparseable
    /// service-account / AAD / Bedrock credential JSON). Maps to 401
    /// `authentication_error`, not 400: this is an auth-material problem,
    /// not a request/routing-shape problem. Non-retryable
    /// (#367 follow-up). Distinct from [`InvalidUpstreamConfig`] (400),
    /// which is request/routing shape, not credentials.
    #[error("invalid upstream credentials: {0}")]
    InvalidUpstreamCredentials(String),
    #[error("transport error: {0}")]
    Transport(String),
    #[error("upstream cancelled the response mid-stream")]
    StreamAborted,
}

impl BridgeError {
    /// Convenience constructor for synthesised upstream errors (tests,
    /// cooldown fixtures) where no real upstream envelope is involved.
    /// Sets [`UpstreamWire::Unknown`] and `parsed: None`.
    pub fn upstream_status(status: u16, message: impl Into<String>) -> Self {
        Self::UpstreamStatus {
            status,
            message: message.into(),
            parsed: None,
            wire: UpstreamWire::Unknown,
            retry_after: None,
        }
    }

    /// Convenience constructor for synthesised upstream errors that
    /// carry a parsed `Retry-After` hint. See [`upstream_status`].
    pub fn upstream_status_with_retry_after(
        status: u16,
        message: impl Into<String>,
        retry_after: Option<Duration>,
    ) -> Self {
        Self::UpstreamStatus {
            status,
            message: message.into(),
            parsed: None,
            wire: UpstreamWire::Unknown,
            retry_after,
        }
    }
}

/// Parse the `Retry-After` response header into a Duration.
///
/// Per RFC 9110 §10.2.3, `Retry-After` may be either:
/// - a non-negative integer number of seconds, or
/// - an HTTP-date.
///
/// We accept the seconds form (which is what OpenAI / Anthropic /
/// DeepSeek / Gemini all return on 429). The HTTP-date form is rare
/// for AI providers and parsing it pulls in `httpdate`; skip for V1
/// — callers fall back to the configured default cooldown TTL.
///
/// Returns `None` when the header is absent, unparseable, or the
/// seconds value is unreasonable (the cooldown layer applies a
/// `max_seconds` clamp regardless).
pub fn parse_retry_after(headers: &http::HeaderMap) -> Option<Duration> {
    let raw = headers.get(http::header::RETRY_AFTER)?.to_str().ok()?;
    let seconds: u64 = raw.trim().parse().ok()?;
    Some(Duration::from_secs(seconds))
}

/// Drain an upstream error response (capped at
/// [`MAX_UPSTREAM_ERROR_BODY_BYTES`]) and produce a
/// [`BridgeError::UpstreamStatus`] with a best-effort parsed view of
/// the envelope.
///
/// The `parse` closure runs only when the response declares an
/// `application/json` content-type — this guards against fronting WAFs
/// or load balancers returning HTML error pages that would otherwise be
/// fed to a JSON parser and either fail expensively or surface
/// nonsensical fragments.
///
/// `parse` returning `None` is treated as "envelope shape unknown"; the
/// fallback in that case is the truncated raw body string in
/// [`BridgeError::UpstreamStatus::message`], same as for non-JSON
/// bodies.
pub async fn capture_upstream_error_http(
    status: http::StatusCode,
    resp: reqwest::Response,
    wire: UpstreamWire,
    parse: impl FnOnce(&[u8]) -> Option<UpstreamErrorView>,
) -> BridgeError {
    let retry_after = parse_retry_after(resp.headers());
    let body = read_body_capped(resp, MAX_UPSTREAM_ERROR_BODY_BYTES).await;
    // Parse the error envelope opportunistically, regardless of the
    // upstream's Content-Type (#543). OpenAI's 401 `invalid_api_key`
    // path (and edge / proxy layers fronting some upstreams) return the
    // JSON error body labelled with a non-`application/json`
    // Content-Type; gating the parse on Content-Type silently dropped
    // `code` / `param` and dumped the raw body into `message`. The
    // per-bridge `parse` fn is the real validator — it requires the
    // provider's `{"error": {...}}` shape and returns `None` on any
    // non-matching body (HTML error pages, plain text, 5xx bodies), so
    // attempting it unconditionally is safe and strictly more robust.
    let parsed = parse(&body)
        // Truncate every parsed string at the same cap as the outer
        // `message`. Otherwise a hostile or buggy upstream emitting a
        // 60 KB `error.message` / `error.code` / `error.type` /
        // `error.param` would reach the customer envelope verbatim —
        // the cap exists exactly to prevent that. AWS exception codes
        // / Anthropic types / OpenAI codes are bounded vocabulary in
        // practice but the cap applies defensively.
        .map(|mut v| {
            let cap = MAX_UPSTREAM_ERROR_MESSAGE_BYTES;
            v.message = v.message.map(|m| truncate_lossy(&m, cap));
            v.kind = v.kind.map(|k| truncate_lossy(&k, cap));
            v.code = v.code.map(|c| truncate_lossy(&c, cap));
            v.param = v.param.map(|p| truncate_lossy(&p, cap));
            v
        });
    let message = parsed
        .as_ref()
        .and_then(|v| v.message.clone())
        .unwrap_or_else(|| String::from_utf8_lossy(&body).into_owned());
    BridgeError::UpstreamStatus {
        status: status.as_u16(),
        message: truncate_lossy(&message, MAX_UPSTREAM_ERROR_MESSAGE_BYTES),
        parsed: parsed.map(Box::new),
        wire,
        retry_after,
    }
}

/// Read the response body, stopping after `limit` bytes. Used to bound
/// upstream-error parsing cost regardless of `Content-Length`. Errors
/// during read surface as an empty buffer — the caller falls through
/// to a parse-failure path and emits the generic `upstream_error`
/// envelope, which matches the pre-fix behaviour for that edge.
///
/// Public so non-OpenAI / non-Anthropic bridges (Vertex, Azure) can
/// enforce the same cap when they need a custom parse path (e.g.
/// extracting only `kind` from the upstream envelope while suppressing
/// the `message` for operator-taxonomy redaction).
pub async fn read_body_capped(resp: reqwest::Response, limit: usize) -> bytes::Bytes {
    use futures::StreamExt;
    let mut buf = bytes::BytesMut::with_capacity(limit.min(16 * 1024));
    let mut stream = resp.bytes_stream();
    // Continue draining the stream past `limit` so the underlying
    // hyper connection can be returned to reqwest's keep-alive pool.
    // Stopping the iteration mid-stream taints the connection and
    // forces a new TCP handshake on the next upstream call — a real
    // cost when an upstream is flapping and producing a burst of
    // error responses. The extra reads only discard bytes; memory
    // stays bounded by `limit`.
    while let Some(chunk) = stream.next().await {
        let Ok(chunk) = chunk else { break };
        if buf.len() >= limit {
            continue;
        }
        let remaining = limit - buf.len();
        let take = chunk.len().min(remaining);
        buf.extend_from_slice(&chunk[..take]);
    }
    buf.freeze()
}

/// Content-Type token starts with `application/json` (RFC 7231 §3.1.1.1
/// allows a trailing `; charset=…` parameter, so a prefix match is the
/// right shape here — exact equality misses `application/json; charset=utf-8`).
///
/// Public so non-OpenAI / non-Anthropic bridges (Vertex, Azure) can
/// apply the same JSON-only guard when they need a custom parse path
/// that doesn't route through [`capture_upstream_error_http`].
pub fn content_type_is_json(ct: &str) -> bool {
    let ct = ct.trim_start();
    ct.starts_with("application/json")
}

/// Convenience: read the `Content-Type` header from a [`reqwest::Response`]
/// and decide whether it's `application/json` per [`content_type_is_json`].
/// Returns `false` when the header is missing or non-ASCII.
pub fn response_is_json(resp: &reqwest::Response) -> bool {
    resp.headers()
        .get(http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|ct| content_type_is_json(&ct.to_ascii_lowercase()))
        .unwrap_or(false)
}

/// Truncate a string to at most `max` bytes, splitting only on a UTF-8
/// boundary. Appends an ellipsis when truncation occurred so log
/// readers can tell the message was cut.
fn truncate_lossy(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

impl BridgeError {
    /// Stable HTTP status mapping. The proxy layer uses this to build
    /// its OpenAI-compatible `{error:{message,type,...}}` envelope.
    pub fn http_status(&self) -> u16 {
        match self {
            BridgeError::Timeout { .. } => 504,
            BridgeError::UpstreamStatus { status, .. } => {
                // We only forward 4xx directly; everything else collapses
                // to 502 so clients don't see upstream 5xx bleed through.
                if (400..500).contains(status) {
                    *status
                } else {
                    502
                }
            }
            BridgeError::UpstreamDecode(_) => 502,
            BridgeError::Config(_) => 500,
            BridgeError::InvalidUpstreamConfig(_) => 400,
            BridgeError::InvalidUpstreamCredentials(_) => 401,
            BridgeError::Transport(_) => 502,
            BridgeError::StreamAborted => 502,
        }
    }

    /// Stable error-type token for the error envelope's `type` field.
    pub fn error_type(&self) -> &'static str {
        match self {
            BridgeError::Timeout { .. } => "timeout",
            BridgeError::UpstreamStatus { .. } => "upstream_error",
            BridgeError::UpstreamDecode(_) => "upstream_decode_error",
            BridgeError::Config(_) => "config_error",
            BridgeError::InvalidUpstreamConfig(_) => "invalid_request_error",
            BridgeError::InvalidUpstreamCredentials(_) => "authentication_error",
            BridgeError::Transport(_) => "transport_error",
            BridgeError::StreamAborted => "stream_aborted",
        }
    }
}

/// A live stream of chunks. Boxed so the Bridge trait stays object-safe
/// (the Hub holds `Arc<dyn Bridge>` values).
pub type ChatChunkStream = BoxStream<'static, Result<ChatChunk, BridgeError>>;

/// The provider-agnostic chat operation. Implementors live in the
/// individual `aisix-provider-*` crates.
#[async_trait]
pub trait Bridge: Send + Sync + 'static {
    /// Human-readable name used in logs and metrics labels. Stable across
    /// upgrades so dashboards don't break.
    fn name(&self) -> &'static str;

    /// Non-streaming call: one request, one response.
    async fn chat(
        &self,
        req: &ChatFormat,
        ctx: &BridgeContext,
    ) -> Result<ChatResponse, BridgeError>;

    /// Streaming call: one request, a stream of deltas.
    async fn chat_stream(
        &self,
        req: &ChatFormat,
        ctx: &BridgeContext,
    ) -> Result<ChatChunkStream, BridgeError>;

    /// Embedding call: text(s) → float vectors. Providers that do not
    /// support embeddings return [`BridgeError::Config`] with a clear
    /// message so the proxy can surface a 501 rather than a 502.
    async fn embed(
        &self,
        _req: &EmbeddingRequest,
        _ctx: &BridgeContext,
    ) -> Result<EmbeddingResponse, BridgeError> {
        Err(BridgeError::Config(
            "this provider does not support embeddings".into(),
        ))
    }

    /// Legacy text completions passthrough (`/v1/completions`).
    ///
    /// The request body JSON is forwarded verbatim after replacing the
    /// `model` field with the upstream provider model id. The response
    /// body JSON is returned as-is from the upstream so format differences
    /// between providers are the caller's responsibility.
    ///
    /// Providers that do not expose a `/completions` endpoint should keep
    /// the default, which returns a 501-mapped [`BridgeError::Config`].
    async fn complete(
        &self,
        _body: &serde_json::Value,
        _ctx: &BridgeContext,
    ) -> Result<serde_json::Value, BridgeError> {
        Err(BridgeError::Config(
            "this provider does not support text completions".into(),
        ))
    }

    /// Image generation passthrough (`/v1/images/generations`).
    ///
    /// The request body JSON is forwarded verbatim after replacing the
    /// `model` field with the upstream provider model id. The response
    /// body JSON is returned as-is from the upstream.
    ///
    /// Providers that do not expose an image generation endpoint should keep
    /// the default, which returns a 501-mapped [`BridgeError::Config`].
    async fn generate_image(
        &self,
        _body: &serde_json::Value,
        _ctx: &BridgeContext,
    ) -> Result<serde_json::Value, BridgeError> {
        Err(BridgeError::Config(
            "this provider does not support image generation".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeout_maps_to_504() {
        let e = BridgeError::Timeout {
            cause: String::new(),
            elapsed_ms: 30_000,
        };
        assert_eq!(e.http_status(), 504);
        assert_eq!(e.error_type(), "timeout");
    }

    /// A gateway-owned deadline has no transport cause, and its message
    /// must stay byte-identical to what it was before `cause` existed —
    /// operators and log queries key on this sentence.
    #[test]
    fn timeout_without_cause_renders_unchanged() {
        let e = BridgeError::Timeout {
            elapsed_ms: 30_000,
            cause: String::new(),
        };
        assert_eq!(e.to_string(), "upstream request timed out after 30000ms");
    }

    /// With a transport cause the message names it, so an expired
    /// `connect_timeout` and an expired request budget stop looking
    /// identical (AISIX-Cloud#1093).
    #[test]
    fn timeout_with_cause_appends_it() {
        let e = BridgeError::Timeout {
            elapsed_ms: 5_001,
            cause: "error sending request: client error (Connect): tcp connect error: \
                    Connection timed out (os error 110)"
                .to_string(),
        };
        assert_eq!(
            e.to_string(),
            "upstream request timed out after 5001ms: error sending request: \
             client error (Connect): tcp connect error: Connection timed out (os error 110)"
        );
        // Status and telemetry class are unaffected by the added detail.
        assert_eq!(e.http_status(), 504);
        assert_eq!(e.error_type(), "timeout");
    }

    #[test]
    fn upstream_4xx_passes_through_5xx_collapses_to_502() {
        let e400 = BridgeError::upstream_status(429, "rate limit");
        assert_eq!(e400.http_status(), 429);

        let e500 = BridgeError::upstream_status(503, "busy");
        assert_eq!(e500.http_status(), 502);

        let e3xx = BridgeError::upstream_status(301, "redirect");
        // Non-4xx collapses too — redirects we don't follow are 502-worthy.
        assert_eq!(e3xx.http_status(), 502);
    }

    #[test]
    fn upstream_status_carries_retry_after_when_provided() {
        let e = BridgeError::upstream_status_with_retry_after(
            429,
            "slow down",
            Some(Duration::from_secs(60)),
        );
        match e {
            BridgeError::UpstreamStatus { retry_after, .. } => {
                assert_eq!(retry_after, Some(Duration::from_secs(60)));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn parse_retry_after_handles_seconds_form() {
        let mut h = http::HeaderMap::new();
        h.insert(http::header::RETRY_AFTER, "30".parse().unwrap());
        assert_eq!(parse_retry_after(&h), Some(Duration::from_secs(30)));
    }

    #[test]
    fn parse_retry_after_returns_none_for_http_date_form() {
        let mut h = http::HeaderMap::new();
        h.insert(
            http::header::RETRY_AFTER,
            "Wed, 21 Oct 2026 07:28:00 GMT".parse().unwrap(),
        );
        // V1: HTTP-date form is intentionally not parsed.
        assert_eq!(parse_retry_after(&h), None);
    }

    #[test]
    fn parse_retry_after_returns_none_when_absent() {
        let h = http::HeaderMap::new();
        assert_eq!(parse_retry_after(&h), None);
    }

    #[test]
    fn transport_and_decode_errors_collapse_to_502() {
        assert_eq!(
            BridgeError::Transport("connection refused".into()).http_status(),
            502,
        );
        assert_eq!(
            BridgeError::UpstreamDecode("bad json".into()).http_status(),
            502,
        );
    }

    #[test]
    fn config_error_maps_to_500() {
        assert_eq!(
            BridgeError::Config("missing api_key".into()).http_status(),
            500
        );
        assert_eq!(
            BridgeError::Config("missing api_key".into()).error_type(),
            "config_error"
        );
    }

    #[test]
    fn invalid_upstream_config_maps_to_400_invalid_request() {
        // #367: customer-fixable config (missing api_base, missing
        // model_name, request-shape …) is a 400, not a 500 — retrying
        // won't help and a 5xx wrongly reads as a server fault.
        let e = BridgeError::InvalidUpstreamConfig("provider_key has no api_base".into());
        assert_eq!(e.http_status(), 400);
        assert_eq!(e.error_type(), "invalid_request_error");
    }

    #[test]
    fn invalid_upstream_credentials_maps_to_401_authentication() {
        // #367 follow-up: auth-material problems (empty/invalid secret,
        // unparseable credential JSON) are a 401 authentication_error,
        // not a 400 — they're a distinct class from request/routing shape.
        let e = BridgeError::InvalidUpstreamCredentials("provider_key.api_key is empty".into());
        assert_eq!(e.http_status(), 401);
        assert_eq!(e.error_type(), "authentication_error");
    }

    #[test]
    fn context_defaults_no_deadline_with_helper_setter() {
        let m = std::sync::Arc::new(sample_model());
        let pk = std::sync::Arc::new(sample_provider_key());
        let ctx = BridgeContext::new("req-1", m.clone(), pk);
        assert_eq!(ctx.request_id, "req-1");
        assert!(ctx.deadline.is_none());
        let ctx = ctx.with_deadline(Duration::from_secs(30));
        assert_eq!(ctx.deadline, Some(Duration::from_secs(30)));
    }

    fn sample_model() -> Model {
        serde_json::from_str(
            r#"{
                "display_name": "test",
                "provider": "openai",
                "model_name": "gpt-4o",
                "provider_key_id": "11111111-1111-1111-1111-111111111111"
            }"#,
        )
        .unwrap()
    }

    fn sample_provider_key() -> ProviderKey {
        serde_json::from_str(r#"{"display_name":"openai-prod","secret":"sk-x"}"#).unwrap()
    }

    #[test]
    fn sample_model_resolves_to_openai() {
        let m = sample_model();
        assert_eq!(m.provider.as_deref(), Some("openai"));
    }
}
