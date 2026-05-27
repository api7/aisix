//! Prometheus metrics registry shared across the proxy middleware and
//! the admin `/metrics` endpoint.
//!
//! Existing compatibility series cover spec §7:
//! - `aisix_requests_total{provider,model,status,outcome}` — counter
//!   incremented at the end of every proxy request.
//! - `aisix_request_duration_seconds{provider,model,status}` — histogram
//!   of end-to-end proxy latency.
//! - `aisix_ratelimit_rejections_total{scope}` — counter for 429 flows.
//! - `aisix_tokens_consumed_total{provider,model}` — counter of
//!   `usage.total_tokens` summed across completed non-streaming calls.
//!
//! Newer AISIX-native series use `aisix_proxy_*` and `aisix_llm_*`
//! names with bounded, DP-stable labels. They intentionally do not
//! copy LiteLLM label names that the data plane does not have.
//!
//! A single [`Metrics`] instance is held `Arc`'d inside `ObsState` and
//! cloned into axum state. The exposition format is emitted via
//! `metrics-exporter-prometheus`'s text renderer; no global recorder is
//! installed, so tests can spin up isolated instances per case.

use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle, PrometheusRecorder};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

/// Metric names (public so the admin `/metrics` handler and tests can
/// refer to them without typo risk).
pub const M_REQUESTS_TOTAL: &str = "aisix_requests_total";
pub const M_REQUEST_DURATION: &str = "aisix_request_duration_seconds";
pub const M_RATELIMIT_REJECTIONS: &str = "aisix_ratelimit_rejections_total";
pub const M_TOKENS_CONSUMED: &str = "aisix_tokens_consumed_total";
pub const M_LLM_SPEND_MICRO_USD_TOTAL: &str = "aisix_llm_spend_micro_usd_total";
pub const M_LLM_INPUT_TOKENS_TOTAL: &str = "aisix_llm_input_tokens_total";
pub const M_LLM_OUTPUT_TOKENS_TOTAL: &str = "aisix_llm_output_tokens_total";
pub const M_LLM_TOTAL_TOKENS_TOTAL: &str = "aisix_llm_total_tokens_total";
pub const M_LLM_REQUESTS_TOTAL: &str = "aisix_llm_requests_total";
pub const M_LLM_REQUEST_DURATION: &str = "aisix_llm_request_duration_seconds";
pub const M_LLM_API_LATENCY: &str = "aisix_llm_api_latency_seconds";
pub const M_LLM_TTFT: &str = "aisix_llm_time_to_first_token_seconds";
pub const M_PROXY_IN_FLIGHT: &str = "aisix_proxy_in_flight_requests";
pub const M_PROXY_REQUESTS_TOTAL: &str = "aisix_proxy_requests_total";
pub const M_PROXY_FAILED_REQUESTS_TOTAL: &str = "aisix_proxy_failed_requests_total";
pub const M_PROXY_REQUEST_DURATION: &str = "aisix_proxy_request_duration_seconds";
pub const M_DEPLOYMENT_REQUESTS_TOTAL: &str = "aisix_deployment_requests_total";
pub const M_DEPLOYMENT_SUCCESS_TOTAL: &str = "aisix_deployment_success_responses_total";
pub const M_DEPLOYMENT_FAILURE_TOTAL: &str = "aisix_deployment_failure_responses_total";
pub const M_DEPLOYMENT_STATE: &str = "aisix_deployment_state";
pub const M_DEPLOYMENT_COOLED_DOWN_TOTAL: &str = "aisix_deployment_cooled_down_total";
pub const M_ROUTING_SUCCESSFUL_FALLBACKS_TOTAL: &str = "aisix_routing_successful_fallbacks_total";
pub const M_ROUTING_FAILED_FALLBACKS_TOTAL: &str = "aisix_routing_failed_fallbacks_total";
pub const M_RATELIMIT_REMAINING_REQUESTS: &str = "aisix_ratelimit_remaining_requests";
pub const M_RATELIMIT_REMAINING_TOKENS: &str = "aisix_ratelimit_remaining_tokens";
pub const M_BUDGET_LIMIT_USD: &str = "aisix_budget_limit_usd";
pub const M_BUDGET_SPENT_USD: &str = "aisix_budget_spent_usd";
pub const M_BUDGET_REMAINING_USD: &str = "aisix_budget_remaining_usd";
pub const M_BUDGET_RESET_SECONDS: &str = "aisix_budget_reset_seconds";
pub const M_BUDGET_DETAILS_PRESENT: &str = "aisix_budget_details_present";
pub const M_REDIS_FAILURES_TOTAL: &str = "aisix_redis_failures_total";
pub const M_USAGE_EVENT_DROPS_TOTAL: &str = "aisix_usage_event_drops_total";
/// Issue #408: counter for UsageEvents successfully enqueued onto the
/// `UsageSink` (i.e. handed off to the telemetry worker for delivery
/// to cp-api + per-env OTLP exporters). Operators slice this by:
/// - `handler`: which OpenAI-shape handler emitted (chat /
///   embeddings / responses / completions / rerank / audio /
///   images / messages). Fixed enumeration, low cardinality.
/// - `status_code`: bucketed as `2xx` / `4xx` / `5xx` (avoid the
///   1000-value cardinality blowup of raw u16 codes).
/// - `inbound_protocol`: `openai` / `anthropic`. Matches the
///   wire-level field on UsageEvent.
///
/// Paired with `aisix_usage_event_drops_total{reason}` for the
/// `try_send` failure paths (sink full / closed).
pub const M_USAGE_EVENT_EMITS_TOTAL: &str = "aisix_usage_events_emitted_total";
pub const M_OTLP_FANOUT_DROPS_TOTAL: &str = "aisix_otlp_fanout_drops_total";
pub const M_OTLP_FANOUT_FAILURES_TOTAL: &str = "aisix_otlp_fanout_failures_total";

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
    proxy_in_flight: Mutex<HashMap<(String, String), i64>>,
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
            inner: Arc::new(MetricsInner {
                recorder,
                handle,
                proxy_in_flight: Mutex::new(HashMap::new()),
            }),
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

    pub fn increment_proxy_in_flight(&self, endpoint: &str, inbound_protocol: &str) {
        let value = {
            let mut counters = self.inner.proxy_in_flight.lock().expect("lock in-flight");
            let value = counters
                .entry((endpoint.to_string(), inbound_protocol.to_string()))
                .or_insert(0);
            *value += 1;
            *value
        };
        metrics::with_local_recorder(&self.inner.recorder, || {
            metrics::gauge!(
                M_PROXY_IN_FLIGHT,
                "endpoint" => endpoint.to_string(),
                "inbound_protocol" => inbound_protocol.to_string(),
            )
            .set(value as f64);
        });
    }

    pub fn decrement_proxy_in_flight(&self, endpoint: &str, inbound_protocol: &str) {
        let value = {
            let mut counters = self.inner.proxy_in_flight.lock().expect("lock in-flight");
            let key = (endpoint.to_string(), inbound_protocol.to_string());
            let value = counters.entry(key.clone()).or_insert(0);
            *value = (*value - 1).max(0);
            let current = *value;
            if current == 0 {
                counters.remove(&key);
            }
            current
        };
        metrics::with_local_recorder(&self.inner.recorder, || {
            metrics::gauge!(
                M_PROXY_IN_FLIGHT,
                "endpoint" => endpoint.to_string(),
                "inbound_protocol" => inbound_protocol.to_string(),
            )
            .set(value as f64);
        });
    }

    pub fn record_proxy_request(&self, labels: RequestLabels<'_>, duration: Duration) {
        metrics::with_local_recorder(&self.inner.recorder, || {
            labels.record_request_counter(M_PROXY_REQUESTS_TOTAL);
            metrics::histogram!(
                M_PROXY_REQUEST_DURATION,
                "endpoint" => labels.endpoint.to_string(),
                "inbound_protocol" => labels.inbound_protocol.to_string(),
                "provider" => labels.provider.to_string(),
                "model" => labels.model.to_string(),
                "upstream_model" => labels.upstream_model.to_string(),
                "provider_key_id" => labels.provider_key_id.to_string(),
                "api_key_id" => labels.api_key_id.to_string(),
                "team_id" => labels.team_id.to_string(),
                "owner_id" => labels.owner_id.to_string(),
                "status" => labels.status.to_string(),
                "outcome" => labels.outcome.as_str().to_string(),
            )
            .record(duration.as_secs_f64());
            if labels.outcome != RequestOutcome::Success {
                labels.record_request_counter(M_PROXY_FAILED_REQUESTS_TOTAL);
            }
        });
    }

    pub fn record_llm_request(&self, labels: RequestLabels<'_>, duration: Duration) {
        metrics::with_local_recorder(&self.inner.recorder, || {
            labels.record_request_counter(M_LLM_REQUESTS_TOTAL);
            metrics::histogram!(
                M_LLM_REQUEST_DURATION,
                "endpoint" => labels.endpoint.to_string(),
                "inbound_protocol" => labels.inbound_protocol.to_string(),
                "provider" => labels.provider.to_string(),
                "model" => labels.model.to_string(),
                "upstream_model" => labels.upstream_model.to_string(),
                "provider_key_id" => labels.provider_key_id.to_string(),
                "api_key_id" => labels.api_key_id.to_string(),
                "team_id" => labels.team_id.to_string(),
                "owner_id" => labels.owner_id.to_string(),
                "status" => labels.status.to_string(),
                "outcome" => labels.outcome.as_str().to_string(),
            )
            .record(duration.as_secs_f64());
        });
    }

    pub fn record_llm_usage(&self, labels: UsageLabels<'_>, usage: LlmUsage) {
        if usage.is_empty() {
            return;
        }
        metrics::with_local_recorder(&self.inner.recorder, || {
            if usage.input_tokens > 0 {
                labels.record_counter(M_LLM_INPUT_TOKENS_TOTAL, u64::from(usage.input_tokens));
            }
            if usage.output_tokens > 0 {
                labels.record_counter(M_LLM_OUTPUT_TOKENS_TOTAL, u64::from(usage.output_tokens));
            }
            if usage.total_tokens > 0 {
                labels.record_counter(M_LLM_TOTAL_TOKENS_TOTAL, u64::from(usage.total_tokens));
            }
            if usage.spend_usd > 0.0 {
                labels.record_spend_usd(usage.spend_usd);
            }
        });
    }

    pub fn record_time_to_first_token(&self, labels: UsageLabels<'_>, ttft: Duration) {
        if ttft.is_zero() {
            return;
        }
        metrics::with_local_recorder(&self.inner.recorder, || {
            metrics::histogram!(
                M_LLM_TTFT,
                "endpoint" => labels.endpoint.to_string(),
                "inbound_protocol" => labels.inbound_protocol.to_string(),
                "provider" => labels.provider.to_string(),
                "model" => labels.model.to_string(),
                "upstream_model" => labels.upstream_model.to_string(),
                "provider_key_id" => labels.provider_key_id.to_string(),
                "api_key_id" => labels.api_key_id.to_string(),
                "team_id" => labels.team_id.to_string(),
                "owner_id" => labels.owner_id.to_string(),
            )
            .record(ttft.as_secs_f64());
        });
    }

    pub fn record_deployment_request(&self, labels: DeploymentLabels<'_>, outcome: RequestOutcome) {
        metrics::with_local_recorder(&self.inner.recorder, || {
            labels.record_counter(M_DEPLOYMENT_REQUESTS_TOTAL);
            match outcome {
                RequestOutcome::Success => labels.record_counter(M_DEPLOYMENT_SUCCESS_TOTAL),
                _ => labels.record_counter(M_DEPLOYMENT_FAILURE_TOTAL),
            }
        });
    }

    pub fn set_deployment_state(&self, labels: DeploymentLabels<'_>, state: DeploymentState) {
        metrics::with_local_recorder(&self.inner.recorder, || {
            metrics::gauge!(
                M_DEPLOYMENT_STATE,
                "provider" => labels.provider.to_string(),
                "model" => labels.model.to_string(),
                "upstream_model" => labels.upstream_model.to_string(),
                "provider_key_id" => labels.provider_key_id.to_string(),
            )
            .set(state.as_f64());
        });
    }

    pub fn record_deployment_cooldown(&self, labels: DeploymentLabels<'_>) {
        metrics::with_local_recorder(&self.inner.recorder, || {
            labels.record_counter(M_DEPLOYMENT_COOLED_DOWN_TOTAL);
        });
    }

    pub fn record_routing_fallback(&self, success: bool, model: &str) {
        let metric = if success {
            M_ROUTING_SUCCESSFUL_FALLBACKS_TOTAL
        } else {
            M_ROUTING_FAILED_FALLBACKS_TOTAL
        };
        metrics::with_local_recorder(&self.inner.recorder, || {
            metrics::counter!(metric, "model" => model.to_string()).increment(1);
        });
    }

    pub fn set_rate_limit_remaining(
        &self,
        api_key_id: &str,
        model: &str,
        requests: Option<u64>,
        tokens: Option<u64>,
    ) {
        metrics::with_local_recorder(&self.inner.recorder, || {
            if let Some(value) = requests {
                metrics::gauge!(
                    M_RATELIMIT_REMAINING_REQUESTS,
                    "api_key_id" => api_key_id.to_string(),
                    "model" => model.to_string(),
                )
                .set(value as f64);
            }
            if let Some(value) = tokens {
                metrics::gauge!(
                    M_RATELIMIT_REMAINING_TOKENS,
                    "api_key_id" => api_key_id.to_string(),
                    "model" => model.to_string(),
                )
                .set(value as f64);
            }
        });
    }

    pub fn set_budget_gauges(&self, labels: BudgetLabels<'_>, budget: BudgetGauges) {
        metrics::with_local_recorder(&self.inner.recorder, || {
            labels.record_gauge(M_BUDGET_DETAILS_PRESENT, 1.0);
            if let Some(value) = budget.limit_usd {
                labels.record_gauge(M_BUDGET_LIMIT_USD, value);
            }
            if let Some(value) = budget.spent_usd {
                labels.record_gauge(M_BUDGET_SPENT_USD, value);
            }
            if let Some(value) = budget.remaining_usd {
                labels.record_gauge(M_BUDGET_REMAINING_USD, value);
            }
            if let Some(value) = budget.reset_seconds {
                labels.record_gauge(M_BUDGET_RESET_SECONDS, value as f64);
            }
        });
    }

    pub fn clear_budget_gauges(&self, labels: BudgetLabels<'_>) {
        metrics::with_local_recorder(&self.inner.recorder, || {
            labels.record_gauge(M_BUDGET_DETAILS_PRESENT, 0.0);
        });
    }

    pub fn record_redis_failure(&self, operation: &str) {
        metrics::with_local_recorder(&self.inner.recorder, || {
            metrics::counter!(M_REDIS_FAILURES_TOTAL, "operation" => operation.to_string())
                .increment(1);
        });
    }

    pub fn record_usage_event_drop(&self, reason: &str) {
        metrics::with_local_recorder(&self.inner.recorder, || {
            metrics::counter!(M_USAGE_EVENT_DROPS_TOTAL, "reason" => reason.to_string())
                .increment(1);
        });
    }

    /// Issue #408: bump on every `UsageSink::try_emit` call (the
    /// handler's emission intent — paired with the drops counter so
    /// the invariant `emitted == delivered + dropped` holds strictly).
    ///
    /// All three labels are `&'static str` so prometheus cardinality
    /// is type-system-bounded:
    /// - `handler`: OpenAI-shape endpoint name (`chat`, `embeddings`,
    ///   `messages`, etc.)
    /// - `status_code`: bucketed by `status_bucket()` (one of `2xx` /
    ///   `3xx` / `4xx` / `5xx` / `other`) — never a raw u16
    /// - `inbound_protocol`: normalised by the caller to one of
    ///   `"openai"` / `"anthropic"` / `"other"` (audit MEDIUM-3 —
    ///   `&'static str` here prevents user-controlled cardinality)
    pub fn record_usage_event_emit(
        &self,
        handler: &'static str,
        status_code: u16,
        inbound_protocol: &'static str,
    ) {
        metrics::with_local_recorder(&self.inner.recorder, || {
            metrics::counter!(
                M_USAGE_EVENT_EMITS_TOTAL,
                "handler" => handler,
                "status_code" => status_bucket(status_code),
                "inbound_protocol" => inbound_protocol,
            )
            .increment(1);
        });
    }

    pub fn record_otlp_fanout_drop(&self, exporter: &str, reason: &str) {
        metrics::with_local_recorder(&self.inner.recorder, || {
            metrics::counter!(
                M_OTLP_FANOUT_DROPS_TOTAL,
                "exporter" => exporter.to_string(),
                "reason" => reason.to_string(),
            )
            .increment(1);
        });
    }

    pub fn record_otlp_fanout_failure(&self, exporter: &str) {
        metrics::with_local_recorder(&self.inner.recorder, || {
            metrics::counter!(M_OTLP_FANOUT_FAILURES_TOTAL, "exporter" => exporter.to_string())
                .increment(1);
        });
    }
}

/// Bucket an HTTP status code into one of `2xx` / `3xx` / `4xx` /
/// `5xx` / `other` (the last covers 1xx and out-of-range). Used by
/// the UsageEvent emission counter (#408) to keep prometheus label
/// cardinality bounded — raw `u16` would explode to ~1000 series per
/// handler×protocol combination.
fn status_bucket(status: u16) -> &'static str {
    match status {
        200..=299 => "2xx",
        300..=399 => "3xx",
        400..=499 => "4xx",
        500..=599 => "5xx",
        _ => "other",
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RequestLabels<'a> {
    pub endpoint: &'a str,
    pub inbound_protocol: &'a str,
    pub provider: &'a str,
    pub model: &'a str,
    pub upstream_model: &'a str,
    pub provider_key_id: &'a str,
    pub api_key_id: &'a str,
    pub team_id: &'a str,
    pub owner_id: &'a str,
    pub status: u16,
    pub outcome: RequestOutcome,
}

impl Default for RequestLabels<'_> {
    fn default() -> Self {
        Self {
            endpoint: "unknown",
            inbound_protocol: "openai",
            provider: "unknown",
            model: "unknown",
            upstream_model: "unknown",
            provider_key_id: "unknown",
            api_key_id: "unknown",
            team_id: "unknown",
            owner_id: "unknown",
            status: 0,
            outcome: RequestOutcome::UpstreamError,
        }
    }
}

impl RequestLabels<'_> {
    fn record_request_counter(&self, metric: &'static str) {
        metrics::counter!(
            metric,
            "endpoint" => self.endpoint.to_string(),
            "inbound_protocol" => self.inbound_protocol.to_string(),
            "provider" => self.provider.to_string(),
            "model" => self.model.to_string(),
            "upstream_model" => self.upstream_model.to_string(),
            "provider_key_id" => self.provider_key_id.to_string(),
            "api_key_id" => self.api_key_id.to_string(),
            "team_id" => self.team_id.to_string(),
            "owner_id" => self.owner_id.to_string(),
            "status" => self.status.to_string(),
            "outcome" => self.outcome.as_str().to_string(),
        )
        .increment(1);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct UsageLabels<'a> {
    pub endpoint: &'a str,
    pub inbound_protocol: &'a str,
    pub provider: &'a str,
    pub model: &'a str,
    pub upstream_model: &'a str,
    pub provider_key_id: &'a str,
    pub api_key_id: &'a str,
    pub team_id: &'a str,
    pub owner_id: &'a str,
}

impl Default for UsageLabels<'_> {
    fn default() -> Self {
        Self {
            endpoint: "unknown",
            inbound_protocol: "openai",
            provider: "unknown",
            model: "unknown",
            upstream_model: "unknown",
            provider_key_id: "unknown",
            api_key_id: "unknown",
            team_id: "unknown",
            owner_id: "unknown",
        }
    }
}

impl UsageLabels<'_> {
    fn record_counter(&self, metric: &'static str, value: u64) {
        metrics::counter!(
            metric,
            "endpoint" => self.endpoint.to_string(),
            "inbound_protocol" => self.inbound_protocol.to_string(),
            "provider" => self.provider.to_string(),
            "model" => self.model.to_string(),
            "upstream_model" => self.upstream_model.to_string(),
            "provider_key_id" => self.provider_key_id.to_string(),
            "api_key_id" => self.api_key_id.to_string(),
            "team_id" => self.team_id.to_string(),
            "owner_id" => self.owner_id.to_string(),
        )
        .increment(value);
    }

    fn record_spend_usd(&self, value: f64) {
        if !value.is_finite() || value <= 0.0 {
            return;
        }
        let micro_usd = (value * 1_000_000.0).round();
        if micro_usd <= 0.0 {
            return;
        }
        metrics::counter!(
            M_LLM_SPEND_MICRO_USD_TOTAL,
            "endpoint" => self.endpoint.to_string(),
            "inbound_protocol" => self.inbound_protocol.to_string(),
            "provider" => self.provider.to_string(),
            "model" => self.model.to_string(),
            "upstream_model" => self.upstream_model.to_string(),
            "provider_key_id" => self.provider_key_id.to_string(),
            "api_key_id" => self.api_key_id.to_string(),
            "team_id" => self.team_id.to_string(),
            "owner_id" => self.owner_id.to_string(),
        )
        .increment(micro_usd as u64);
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct LlmUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
    pub spend_usd: f64,
}

impl LlmUsage {
    fn is_empty(self) -> bool {
        self.input_tokens == 0
            && self.output_tokens == 0
            && self.total_tokens == 0
            && self.spend_usd <= 0.0
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DeploymentLabels<'a> {
    pub provider: &'a str,
    pub model: &'a str,
    pub upstream_model: &'a str,
    pub provider_key_id: &'a str,
}

impl Default for DeploymentLabels<'_> {
    fn default() -> Self {
        Self {
            provider: "unknown",
            model: "unknown",
            upstream_model: "unknown",
            provider_key_id: "unknown",
        }
    }
}

impl DeploymentLabels<'_> {
    fn record_counter(&self, metric: &'static str) {
        metrics::counter!(
            metric,
            "provider" => self.provider.to_string(),
            "model" => self.model.to_string(),
            "upstream_model" => self.upstream_model.to_string(),
            "provider_key_id" => self.provider_key_id.to_string(),
        )
        .increment(1);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeploymentState {
    Healthy,
    PartialFailure,
    Down,
}

impl DeploymentState {
    fn as_f64(self) -> f64 {
        match self {
            Self::Healthy => 0.0,
            Self::PartialFailure => 1.0,
            Self::Down => 2.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BudgetLabels<'a> {
    pub api_key_id: &'a str,
    pub team_id: &'a str,
    pub owner_id: &'a str,
}

impl Default for BudgetLabels<'_> {
    fn default() -> Self {
        Self {
            api_key_id: "unknown",
            team_id: "unknown",
            owner_id: "unknown",
        }
    }
}

impl BudgetLabels<'_> {
    fn record_gauge(&self, metric: &'static str, value: f64) {
        metrics::gauge!(
            metric,
            "api_key_id" => self.api_key_id.to_string(),
            "team_id" => self.team_id.to_string(),
            "owner_id" => self.owner_id.to_string(),
        )
        .set(value);
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BudgetGauges {
    pub limit_usd: Option<f64>,
    pub spent_usd: Option<f64>,
    pub remaining_usd: Option<f64>,
    pub reset_seconds: Option<u64>,
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

    #[test]
    fn aisix_native_request_usage_and_latency_metrics_render() {
        let m = Metrics::new(false);
        let labels = RequestLabels {
            endpoint: "/v1/chat/completions",
            inbound_protocol: "openai",
            provider: "openai",
            model: "gpt",
            upstream_model: "gpt-4o",
            provider_key_id: "pk-1",
            api_key_id: "ak-1",
            team_id: "team-1",
            owner_id: "owner-1",
            status: 200,
            outcome: RequestOutcome::Success,
        };
        m.record_proxy_request(labels, Duration::from_millis(25));
        m.record_llm_request(labels, Duration::from_millis(20));
        m.record_llm_usage(
            UsageLabels {
                endpoint: "/v1/chat/completions",
                inbound_protocol: "openai",
                provider: "openai",
                model: "gpt",
                upstream_model: "gpt-4o",
                provider_key_id: "pk-1",
                api_key_id: "ak-1",
                team_id: "team-1",
                owner_id: "owner-1",
            },
            LlmUsage {
                input_tokens: 5,
                output_tokens: 7,
                total_tokens: 12,
                spend_usd: 0.001,
            },
        );
        m.record_time_to_first_token(
            UsageLabels {
                endpoint: "/v1/chat/completions",
                inbound_protocol: "openai",
                provider: "openai",
                model: "gpt",
                upstream_model: "gpt-4o",
                provider_key_id: "pk-1",
                api_key_id: "ak-1",
                team_id: "team-1",
                owner_id: "owner-1",
            },
            Duration::from_millis(42),
        );

        let rendered = m.render();
        assert!(rendered.contains(M_PROXY_REQUESTS_TOTAL));
        assert!(rendered.contains(M_LLM_REQUESTS_TOTAL));
        assert!(rendered.contains(M_LLM_INPUT_TOKENS_TOTAL));
        assert!(rendered.contains(M_LLM_OUTPUT_TOKENS_TOTAL));
        assert!(rendered.contains(M_LLM_TOTAL_TOKENS_TOTAL));
        assert!(rendered.contains(M_LLM_SPEND_MICRO_USD_TOTAL));
        assert!(rendered.contains(M_LLM_REQUEST_DURATION));
        assert!(rendered.contains(M_LLM_TTFT));
        assert!(rendered.contains("endpoint=\"/v1/chat/completions\""));
        assert!(rendered.contains("team_id=\"team-1\""));
    }

    #[test]
    fn zero_llm_usage_does_not_emit_samples() {
        let m = Metrics::new(false);
        m.record_llm_usage(UsageLabels::default(), LlmUsage::default());
        let rendered = m.render();
        assert!(!rendered.contains(M_LLM_INPUT_TOKENS_TOTAL));
        assert!(!rendered.contains(M_LLM_TOTAL_TOKENS_TOTAL));
    }

    #[test]
    fn in_flight_gauge_returns_to_zero() {
        let m = Metrics::new(false);
        m.increment_proxy_in_flight("/v1/chat/completions", "openai");
        m.decrement_proxy_in_flight("/v1/chat/completions", "openai");
        let rendered = m.render();
        assert!(rendered.contains(M_PROXY_IN_FLIGHT));
        assert!(
            rendered.contains(" 0"),
            "expected gauge to return to zero:\n{rendered}"
        );
    }

    /// Issue #408 audit MEDIUM-2: pin every boundary of
    /// `status_bucket` so an off-by-one (e.g. `200..299` excluding
    /// 299) would surface as a CI failure rather than slipping
    /// past as silent re-labelling. Covers all 5 buckets including
    /// the dead-code `3xx` / `other` arms which have no live caller
    /// today.
    #[test]
    fn status_bucket_boundaries_are_inclusive() {
        // 2xx
        assert_eq!(status_bucket(200), "2xx");
        assert_eq!(status_bucket(299), "2xx");
        // 3xx
        assert_eq!(status_bucket(300), "3xx");
        assert_eq!(status_bucket(399), "3xx");
        // 4xx
        assert_eq!(status_bucket(400), "4xx");
        assert_eq!(status_bucket(499), "4xx");
        // 5xx
        assert_eq!(status_bucket(500), "5xx");
        assert_eq!(status_bucket(599), "5xx");
        // out-of-range → other
        assert_eq!(status_bucket(199), "other");
        assert_eq!(status_bucket(600), "other");
        assert_eq!(status_bucket(0), "other");
    }
}
