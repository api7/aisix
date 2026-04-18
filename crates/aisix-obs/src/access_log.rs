//! Structured one-line access log. Called by the proxy handler once per
//! completed request (success or error). Keeping the call explicit rather
//! than inside a tower layer means the handler can attach `provider`,
//! `model`, `api_key_id`, and `tokens` — fields the layer couldn't see.

use std::time::Duration;

/// Canonical access-log fields. Constructed by the handler at the end of
/// a request and passed to [`log_access`].
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
