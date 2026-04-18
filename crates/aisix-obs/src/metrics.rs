//! Prometheus metrics registry shared across the proxy middleware and
//! the admin `/metrics` endpoint.
//!
//! Four series cover spec §7:
//! - `aisix_requests_total{provider,model,status,outcome}` — counter
//!   incremented at the end of every proxy request.
//! - `aisix_request_duration_seconds{provider,model,status}` — histogram
//!   of end-to-end proxy latency.
//! - `aisix_ratelimit_rejections_total{scope}` — counter for 429 flows.
//! - `aisix_tokens_consumed_total{provider,model}` — counter of
//!   `usage.total_tokens` summed across completed non-streaming calls.
//!
//! A single [`Metrics`] instance is held `Arc`'d inside `ObsState` and
//! cloned into axum state. The exposition format is emitted via
//! `metrics-exporter-prometheus`'s text renderer; no global recorder is
//! installed, so tests can spin up isolated instances per case.

use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle, PrometheusRecorder};
use std::sync::Arc;
use std::time::Duration;

/// Metric names (public so the admin `/metrics` handler and tests can
/// refer to them without typo risk).
pub const M_REQUESTS_TOTAL: &str = "aisix_requests_total";
pub const M_REQUEST_DURATION: &str = "aisix_request_duration_seconds";
pub const M_RATELIMIT_REJECTIONS: &str = "aisix_ratelimit_rejections_total";
pub const M_TOKENS_CONSUMED: &str = "aisix_tokens_consumed_total";

/// Holds an isolated `PrometheusRecorder` plus its render handle.
/// `metrics::*` macros talk to whatever recorder is in scope; we use
/// `metrics::with_local_recorder` so each write lands on the instance
/// this struct owns — no global state, tests can run in parallel.
#[derive(Clone)]
pub struct Metrics {
    inner: Arc<MetricsInner>,
}

struct MetricsInner {
    recorder: PrometheusRecorder,
    handle: PrometheusHandle,
}

impl std::fmt::Debug for Metrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Metrics").finish_non_exhaustive()
    }
}

impl Metrics {
    /// Build an isolated recorder. `install_global` is kept for future
    /// use but currently has no effect — every Metrics instance runs
    /// with a local recorder so parallel tests don't collide.
    pub fn new(_install_global: bool) -> Self {
        let recorder = PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        Self {
            inner: Arc::new(MetricsInner { recorder, handle }),
        }
    }

    /// Render the current metric values in Prometheus text exposition format.
    pub fn render(&self) -> String {
        self.inner.handle.render()
    }

    /// Record the outcome of one proxy request.
    pub fn record_request(
        &self,
        provider: &str,
        model: &str,
        status: u16,
        outcome: RequestOutcome,
        duration: Duration,
    ) {
        metrics::with_local_recorder(&self.inner.recorder, || {
            metrics::counter!(
                M_REQUESTS_TOTAL,
                "provider" => provider.to_string(),
                "model" => model.to_string(),
                "status" => status.to_string(),
                "outcome" => outcome.as_str().to_string(),
            )
            .increment(1);
            metrics::histogram!(
                M_REQUEST_DURATION,
                "provider" => provider.to_string(),
                "model" => model.to_string(),
                "status" => status.to_string(),
            )
            .record(duration.as_secs_f64());
        });
    }

    pub fn record_ratelimit_rejection(&self, scope: &str) {
        metrics::with_local_recorder(&self.inner.recorder, || {
            metrics::counter!(
                M_RATELIMIT_REJECTIONS,
                "scope" => scope.to_string(),
            )
            .increment(1);
        });
    }

    pub fn record_tokens(&self, provider: &str, model: &str, total_tokens: u64) {
        if total_tokens == 0 {
            return;
        }
        metrics::with_local_recorder(&self.inner.recorder, || {
            metrics::counter!(
                M_TOKENS_CONSUMED,
                "provider" => provider.to_string(),
                "model" => model.to_string(),
            )
            .increment(total_tokens);
        });
    }
}

/// Canonical outcome label for [`Metrics::record_request`]. Keeps the
/// `outcome` dimension bounded so Prometheus cardinality stays sane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestOutcome {
    Success,
    ClientError,
    UpstreamError,
    RateLimited,
}

impl RequestOutcome {
    pub fn from_status(status: u16) -> Self {
        match status {
            429 => Self::RateLimited,
            200..=399 => Self::Success,
            400..=499 => Self::ClientError,
            _ => Self::UpstreamError,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::ClientError => "client_error",
            Self::UpstreamError => "upstream_error",
            Self::RateLimited => "rate_limited",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outcome_from_status_maps_correctly() {
        assert_eq!(RequestOutcome::from_status(200), RequestOutcome::Success);
        assert_eq!(RequestOutcome::from_status(301), RequestOutcome::Success);
        assert_eq!(
            RequestOutcome::from_status(404),
            RequestOutcome::ClientError
        );
        assert_eq!(
            RequestOutcome::from_status(429),
            RequestOutcome::RateLimited
        );
        assert_eq!(
            RequestOutcome::from_status(502),
            RequestOutcome::UpstreamError
        );
    }

    #[test]
    fn recording_a_request_renders_in_exposition_format() {
        let m = Metrics::new(false);
        m.record_request(
            "openai",
            "my-gpt4",
            200,
            RequestOutcome::Success,
            Duration::from_millis(120),
        );
        let rendered = m.render();
        assert!(rendered.contains(M_REQUESTS_TOTAL));
        assert!(rendered.contains("provider=\"openai\""));
        assert!(rendered.contains("outcome=\"success\""));
        assert!(rendered.contains(M_REQUEST_DURATION));
    }

    #[test]
    fn ratelimit_rejection_counter_increments() {
        let m = Metrics::new(false);
        m.record_ratelimit_rejection("requests");
        m.record_ratelimit_rejection("requests");
        let rendered = m.render();
        assert!(rendered.contains(M_RATELIMIT_REJECTIONS));
        assert!(rendered.contains("scope=\"requests\""));
    }

    #[test]
    fn zero_tokens_do_not_emit_a_sample() {
        let m = Metrics::new(false);
        m.record_tokens("openai", "my-gpt4", 0);
        let rendered = m.render();
        // Counter family is never touched so it doesn't appear.
        assert!(!rendered.contains(M_TOKENS_CONSUMED));
    }

    #[test]
    fn token_counts_accumulate_across_calls() {
        let m = Metrics::new(false);
        m.record_tokens("openai", "my-gpt4", 10);
        m.record_tokens("openai", "my-gpt4", 32);
        let rendered = m.render();
        // The rendered counter should be 42. Keep the assertion robust to
        // whitespace variations by searching for the literal value.
        assert!(
            rendered.contains("42"),
            "expected total 42 in exposition, got:\n{rendered}"
        );
    }
}
