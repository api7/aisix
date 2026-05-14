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

/// Error surfaced by any Bridge. Each variant maps to a stable
/// client-visible HTTP status and OpenAI-style error code so the proxy
/// layer can translate without further inspection.
#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    #[error("upstream request timed out after {elapsed_ms}ms")]
    Timeout { elapsed_ms: u64 },
    /// Upstream returned a non-2xx HTTP status. `retry_after` carries
    /// the upstream's `Retry-After` header parsed to a Duration when
    /// present — used by the cooldown layer to honor provider-supplied
    /// backoff hints. Bridges that cannot parse the header (or where
    /// the header is absent) leave this `None`; the cooldown layer
    /// falls back to its configured default in that case.
    #[error("upstream returned HTTP {status}: {message}")]
    UpstreamStatus {
        status: u16,
        message: String,
        retry_after: Option<Duration>,
    },
    #[error("upstream returned an unparseable body: {0}")]
    UpstreamDecode(String),
    #[error("bridge is misconfigured: {0}")]
    Config(String),
    #[error("transport error: {0}")]
    Transport(String),
    #[error("upstream cancelled the response mid-stream")]
    StreamAborted,
}

impl BridgeError {
    /// Convenience constructor for upstream status errors when no
    /// `Retry-After` is available. Keeps existing call sites readable.
    pub fn upstream_status(status: u16, message: impl Into<String>) -> Self {
        Self::UpstreamStatus {
            status,
            message: message.into(),
            retry_after: None,
        }
    }

    /// Convenience constructor for upstream status errors that carry
    /// a parsed `Retry-After` hint.
    pub fn upstream_status_with_retry_after(
        status: u16,
        message: impl Into<String>,
        retry_after: Option<Duration>,
    ) -> Self {
        Self::UpstreamStatus {
            status,
            message: message.into(),
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
            BridgeError::Transport(_) => 502,
            BridgeError::StreamAborted => 502,
        }
    }

    /// Stable error-type token, mirroring OpenAI's error.type field.
    pub fn error_type(&self) -> &'static str {
        match self {
            BridgeError::Timeout { .. } => "timeout",
            BridgeError::UpstreamStatus { .. } => "upstream_error",
            BridgeError::UpstreamDecode(_) => "upstream_decode_error",
            BridgeError::Config(_) => "config_error",
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
    use aisix_core::models::Provider;

    #[test]
    fn timeout_maps_to_504() {
        let e = BridgeError::Timeout { elapsed_ms: 30_000 };
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
        assert_eq!(m.provider, Some(Provider::Openai));
    }
}
