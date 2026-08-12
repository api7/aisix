//! Structured one-line access log. Called by the proxy handler once per
//! request (success or error). Keeping the call explicit rather than inside
//! a tower layer means the handler can attach `provider`, `model`,
//! `api_key_id`, and `tokens` — fields the layer couldn't see.
//!
//! # When the line is written, and what that costs
//!
//! WHEN differs by response type, and it decides which fields can be filled
//! at all. The handler calls this on its way out, so:
//!
//! - **Non-streamed**: at the end of the request. Everything the handler
//!   resolved is available.
//! - **Streamed**: when the SSE body is handed to the server, before a
//!   single frame is polled. The upstream has produced nothing yet, so the
//!   token counts and `provider_request_id` are necessarily absent, and
//!   `status` is the response-OPEN status — a stream that later aborts, or
//!   whose consumer walks away, still logged `200` here.
//!
//! Do not add a field whose value only exists after the upstream responds
//! and expect it on a streamed line; it will be silently empty there. The
//! completion-time figures for a streamed request live on the per-attempt
//! `UsageEvent` (and, for the provider response id, on the
//! `provider call completed` line `UsageSink::try_emit` writes), keyed by
//! the same `request_id`.

use std::time::Duration;

/// Canonical access-log fields. Constructed by the handler on its way out of
/// a request and passed to [`log_access`] — see the module docs for what
/// "on its way out" means for a streamed response, and which fields that
/// leaves empty.
#[derive(Debug, Clone)]
pub struct AccessLog<'a> {
    pub method: &'a str,
    pub path: &'a str,
    pub status: u16,
    pub latency: Duration,
    pub provider: Option<&'a str>,
    pub model: Option<&'a str>,
    pub api_key_id: Option<&'a str>,
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    pub request_id: &'a str,
    /// Provider response object `id` of the attempt that served the request
    /// — OpenAI's `chat.completion.id`, Anthropic's message `id`,
    /// `/v1/responses`' `resp_…`. Distinct from `request_id` (this
    /// gateway's own id) and from the provider's HTTP transport header id;
    /// none of the three may overwrite another (AISIX-Cloud#1289).
    ///
    /// `None` whenever no id exists by the time this line is written:
    /// the request never reached an upstream (guardrail block,
    /// pre-dispatch error), it was served from cache, the endpoint's
    /// provider response carries no id at all (embeddings / audio /
    /// images / count_tokens), or the response is **streamed** — there the
    /// id arrives in the first frame, after this line. Streamed and
    /// mid-stream-failed-over calls are covered instead by the per-attempt
    /// `provider call completed` line (see `UsageSink::try_emit`), which
    /// shares this `request_id`.
    pub provider_request_id: Option<&'a str>,
    /// Routing target that ultimately served the request (the winning
    /// attempt's display name). `None` for direct models / cache hits.
    pub served_by_model: Option<&'a str>,
    /// Total upstream attempts made (initial + retries + fallbacks).
    pub routing_attempt_count: Option<u32>,
    /// How many attempts moved to a different target. Per #655 the
    /// per-attempt detail lives in telemetry (per-attempt UsageEvents),
    /// not in this one-line-per-request access log.
    pub routing_fallback_count: Option<u32>,
    /// Stable failure class (`ProxyError::kind`) — `None` on success.
    /// Machine-readable so an operator can filter or alert on a class
    /// without parsing the free-text message below.
    pub error_kind: Option<&'a str>,
    /// Why the request failed — `None` on success. Without it a 5xx line
    /// carries only `status` + `latency_ms`, which is the same shape for a
    /// kernel-level connect timeout, an upstream 500, and a blocked
    /// guardrail (AISIX-Cloud#1093).
    pub error: Option<&'a str>,
}

impl AccessLog<'_> {
    /// Emit a single `tracing::info!` event carrying every field. The
    /// subscriber's configured format (text or JSON) determines the
    /// wire shape — operators choose via `cfg.observability.log_level`
    /// and (later) a JSON/text knob.
    pub fn emit(&self) {
        tracing::info!(
            method = self.method,
            path = self.path,
            status = self.status,
            latency_ms = self.latency.as_millis() as u64,
            provider = self.provider,
            model = self.model,
            api_key_id = self.api_key_id,
            prompt_tokens = self.prompt_tokens,
            completion_tokens = self.completion_tokens,
            total_tokens = self.total_tokens,
            request_id = self.request_id,
            provider_request_id = self.provider_request_id,
            served_by_model = self.served_by_model,
            routing_attempt_count = self.routing_attempt_count,
            routing_fallback_count = self.routing_fallback_count,
            error_kind = self.error_kind,
            error = self.error,
            "proxy request completed",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing::subscriber::with_default;
    use tracing_subscriber::fmt::MakeWriter;
    use tracing_subscriber::{fmt, EnvFilter};

    /// Collect emitted log bytes into an in-memory buffer.
    #[derive(Clone, Default)]
    struct VecWriter {
        buf: std::sync::Arc<std::sync::Mutex<Vec<u8>>>,
    }
    impl VecWriter {
        fn contents(&self) -> String {
            String::from_utf8_lossy(&self.buf.lock().unwrap()).into_owned()
        }
    }
    impl std::io::Write for VecWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.buf.lock().unwrap().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    impl<'a> MakeWriter<'a> for VecWriter {
        type Writer = VecWriter;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    #[test]
    fn emit_writes_every_field_into_the_subscriber() {
        let writer = VecWriter::default();
        let subscriber = fmt()
            .with_writer(writer.clone())
            .with_ansi(false)
            .with_target(false)
            .with_env_filter(EnvFilter::new("info"))
            .finish();

        with_default(subscriber, || {
            AccessLog {
                method: "POST",
                path: "/v1/chat/completions",
                status: 200,
                latency: Duration::from_millis(42),
                provider: Some("openai"),
                model: Some("my-gpt4"),
                api_key_id: Some("key-id-1"),
                prompt_tokens: Some(2),
                completion_tokens: Some(1),
                total_tokens: Some(3),
                request_id: "req-abc",
                provider_request_id: Some("chatcmpl-abc"),
                served_by_model: Some("fallback-target"),
                routing_attempt_count: Some(2),
                routing_fallback_count: Some(1),
                error_kind: None,
                error: None,
            }
            .emit();
        });

        let out = writer.contents();
        assert!(out.contains("proxy request completed"));
        assert!(out.contains("method=\"POST\"") || out.contains("method=POST"));
        assert!(out.contains("status=200"));
        assert!(out.contains("latency_ms=42"));
        assert!(out.contains("provider=\"openai\"") || out.contains("provider=openai"));
        assert!(out.contains("total_tokens=3"));
        assert!(out.contains("request_id=\"req-abc\"") || out.contains("request_id=req-abc"));
        // AISIX-Cloud#1289: the provider's own response id, next to — never
        // instead of — the gateway's `request_id`.
        assert!(
            out.contains("provider_request_id=\"chatcmpl-abc\"")
                || out.contains("provider_request_id=chatcmpl-abc"),
            "{out}"
        );
        assert!(
            out.contains("served_by_model=\"fallback-target\"")
                || out.contains("served_by_model=fallback-target")
        );
        assert!(out.contains("routing_attempt_count=2"));
        assert!(out.contains("routing_fallback_count=1"));
        // A success line must not carry failure fields at all — an
        // always-present `error=""` would defeat filtering on it.
        assert!(!out.contains("error_kind"), "{out}");
        assert!(!out.contains("error="), "{out}");
    }

    /// The gap this field closes: without it a failed request's only trace
    /// is `status=502 latency_ms=…`, identical for every cause.
    #[test]
    fn emit_carries_the_failure_class_and_reason() {
        let writer = VecWriter::default();
        let subscriber = fmt()
            .with_writer(writer.clone())
            .with_ansi(false)
            .with_target(false)
            .with_env_filter(EnvFilter::new("info"))
            .finish();

        with_default(subscriber, || {
            AccessLog {
                method: "POST",
                path: "/v1/messages",
                status: 504,
                latency: Duration::from_millis(7167),
                provider: None,
                model: Some("claude-sonnet-4"),
                api_key_id: Some("key-id-1"),
                prompt_tokens: None,
                completion_tokens: None,
                total_tokens: None,
                request_id: "req-fail",
                provider_request_id: None,
                served_by_model: None,
                routing_attempt_count: Some(1),
                routing_fallback_count: None,
                error_kind: Some("timeout"),
                error: Some("upstream request timed out after 7167ms"),
            }
            .emit();
        });

        let out = writer.contents();
        assert!(out.contains("status=504"));
        // A call that never got a provider response must not carry an empty
        // `provider_request_id=""` — an always-present field defeats
        // filtering on it, same rule as `error_kind` above.
        assert!(!out.contains("provider_request_id"), "{out}");
        assert!(
            out.contains("error_kind=\"timeout\"") || out.contains("error_kind=timeout"),
            "{out}"
        );
        assert!(
            out.contains("upstream request timed out after 7167ms"),
            "{out}"
        );
    }

    #[test]
    fn emit_handles_missing_optional_fields() {
        let writer = VecWriter::default();
        let subscriber = fmt()
            .with_writer(writer.clone())
            .with_ansi(false)
            .with_target(false)
            .with_env_filter(EnvFilter::new("info"))
            .finish();

        with_default(subscriber, || {
            AccessLog {
                method: "POST",
                path: "/v1/chat/completions",
                status: 401,
                latency: Duration::from_millis(1),
                provider: None,
                model: None,
                api_key_id: None,
                prompt_tokens: None,
                completion_tokens: None,
                total_tokens: None,
                request_id: "req-xyz",
                provider_request_id: None,
                served_by_model: None,
                routing_attempt_count: None,
                routing_fallback_count: None,
                error_kind: None,
                error: None,
            }
            .emit();
        });
        let out = writer.contents();
        assert!(out.contains("status=401"));
        assert!(out.contains("proxy request completed"));
        // The fmt layer elides Option::None values; we should *not* see
        // a concrete provider rendered when the caller supplied None.
        assert!(!out.contains("provider=\"openai\""));
    }
}
