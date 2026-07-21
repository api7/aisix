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
//! copy label names from other LLM gateways that the data plane does
//! not have.
//!
//! A single [`Metrics`] instance is held `Arc`'d inside `ObsState` and
//! cloned into axum state. The exposition format is emitted via
//! `metrics-exporter-prometheus`'s text renderer; no global recorder is
//! installed, so tests can spin up isolated instances per case.

use metrics_exporter_prometheus::{
    Matcher, PrometheusBuilder, PrometheusHandle, PrometheusRecorder,
};
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
/// Issue #890 req-4: token volume sliced by inbound client type — a
/// DEDICATED low-cardinality series so the client dimension never multiplies
/// the per-key `aisix_llm_*_tokens_total` families. `client_type` is
/// normalised to a bounded allowlist by [`client_type_from_user_agent`]; the
/// raw user-agent + client version stay in logs / `UsageEvent`, never here.
/// AISIX-Cloud#1044 adds a `model` label (the requested logical model, same
/// value as the `aisix_llm_*` families' `model`) so the series answers
/// "which models is each client spending tokens on". The label set stays
/// client_type × model × token_type — per-key/team/user dimensions belong to
/// the `aisix_llm_*_tokens_total` families (or UsageEvent/logs), never here.
pub const M_LLM_TOKENS_BY_CLIENT_TOTAL: &str = "aisix_llm_tokens_by_client_total";
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
/// Guardrail outcomes (#379 observability). `aisix_guardrail_blocks_total`
/// counts requests a guardrail rejected (input or output hook; policy or
/// fail-closed combined). `aisix_guardrail_bypasses_total` counts fail-open
/// events — a remote-API guardrail's upstream was unreachable but `fail_open`
/// let the request through — sliced by the bounded DP-internal `reason`
/// (e.g. `bedrock_5xx` / `bedrock_timeout` / `bedrock_throttled`).
///
/// Scope: recorded for `/v1/chat/completions` only until #519 brings the
/// `/v1/messages` path in — read these as chat-path, not gateway-wide.
pub const M_GUARDRAIL_BLOCKS_TOTAL: &str = "aisix_guardrail_blocks_total";
pub const M_GUARDRAIL_BYPASSES_TOTAL: &str = "aisix_guardrail_bypasses_total";
/// Per-execution guardrail latency histogram (AISIX-Cloud#1076), recorded
/// by the chain fold for every member consulted on any handler — chat,
/// messages, responses, embeddings, streaming end-of-stream/window scans,
/// cache-hit output checks, and the segment (Bedrock-style) pass alike.
/// Labels:
/// - `env_id`: constant per DP process (`unknown` standalone).
/// - `guardrail`: the configured (row) name.
/// - `kind`: the guardrail kind discriminator (`keyword`/`pii` run
///   in-process; every other kind calls a remote service, so this label
///   splits local vs remote latency).
/// - `phase`: `input` / `output`.
/// - `result`: `allowed` / `blocked` / `masked` / `bypassed` (remote
///   failure + fail-open) / `would_block` / `would_mask` (monitor mode).
/// - `error_type`: bounded failure tag (e.g. `lakera_timeout`) when
///   `result="bypassed"`, else `none`. Fail-closed failures surface as
///   `blocked` (the timeout budget shows up in the latency distribution).
///
/// The `_count` series doubles as a per-guardrail execution counter, so
/// there is no separate `aisix_guardrail_requests_total` (LiteLLM's
/// `litellm_guardrail_requests_total` equivalent = `sum by (...)` of it).
pub const M_GUARDRAIL_LATENCY_SECONDS: &str = "aisix_guardrail_latency_seconds";
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
/// AISIX-Cloud#1011: SLO-grade latency distributions as REAL bucketed
/// histograms (`_bucket{le=…}`), aggregatable across DP instances with
/// `histogram_quantile()`. Every other `histogram!` series in this file
/// renders as a summary (no buckets configured) whose quantiles cannot
/// be re-aggregated — these two get explicit buckets in [`Metrics::new`]
/// and a DEDICATED low-cardinality label set ([`LatencyLabels`]) so the
/// per-key/per-user dimensions never multiply the bucket count.
///
/// `aisix_request_e2e_latency_seconds` observes the client-perceived
/// end-to-end latency once per request: at handler return for
/// non-streaming requests and failures, at stream completion for
/// committed streams (full stream duration, matching the usage event's
/// `latency_ms` — NOT the time-to-first-byte the summary series record).
/// A stream the client cancels mid-flight still observes once, with the
/// committed status (2xx) and the duration up to the abort — the same
/// client-perceived semantics as the usage event.
pub const M_REQUEST_E2E_LATENCY_SECONDS: &str = "aisix_request_e2e_latency_seconds";
/// Time-to-first-token for streaming requests, same label set as
/// [`M_REQUEST_E2E_LATENCY_SECONDS`] (with `streaming="true"` always).
pub const M_REQUEST_TTFT_SECONDS: &str = "aisix_request_ttft_seconds";

// ── Config load-observability series (load-observability contract) ─────────
// Reflected from [`aisix_core::ConfigMetricsView`] at scrape time via
// [`Metrics::sync_config_status`]. Standard Prometheus config-reload naming so
// the series read the same as the control plane exposes.
pub const M_CONFIG_LAST_RELOAD_SUCCESSFUL: &str = "aisix_config_last_reload_successful";
pub const M_CONFIG_LAST_RELOAD_SUCCESS_TIMESTAMP: &str =
    "aisix_config_last_reload_success_timestamp_seconds";
pub const M_CONFIG_RELOADS_TOTAL: &str = "aisix_config_reloads_total";
pub const M_CONFIG_RELOAD_FAILURES_TOTAL: &str = "aisix_config_reload_failures_total";
pub const M_CONFIG_REJECTED_RESOURCES: &str = "aisix_config_rejected_resources";
pub const M_CONFIG_OBSERVED_REVISION: &str = "aisix_config_observed_revision";
pub const M_CONFIG_APPLIED_REVISION: &str = "aisix_config_applied_revision";
pub const M_CONFIG_HASH_INFO: &str = "aisix_config_hash_info";
pub const M_CONFIG_SOURCE_CONNECTED: &str = "aisix_config_source_connected";

/// Bucket edges for the two SLO histograms, spanning LLM latency ranges:
/// sub-100ms cache hits / TTFT through multi-minute long generations.
/// 14 edges → 15 `_bucket` series per label combination; keep
/// [`LatencyLabels`] lean before adding edges.
pub const LATENCY_HISTOGRAM_BUCKETS: &[f64] = &[
    0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
];

/// Bucket edges for [`M_GUARDRAIL_LATENCY_SECONDS`] — shifted an order of
/// magnitude below the SLO buckets: local (keyword/pii) checks run in
/// microseconds, the added-latency budget under scrutiny is ~50 ms
/// (AISIX-Cloud#1076), and remote guardrail timeouts default to 5 s. The
/// 30 s top edge outlives any configurable guardrail timeout.
pub const GUARDRAIL_LATENCY_BUCKETS: &[f64] = &[
    0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
];

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
    /// Constant `env_id` label for the SLO latency histograms — one DP
    /// process serves exactly one environment. `"unknown"` when the DP
    /// runs standalone (no control plane).
    env_id: String,
    /// Last labels emitted for the config load-observability gauges, so
    /// [`Metrics::sync_config_status`] can zero out stale label series (the
    /// applied hash changed, or a kind's rejections cleared) instead of
    /// leaving a second, contradictory sample in the exposition.
    config_labels: Mutex<ConfigLabelState>,
}

#[derive(Default)]
struct ConfigLabelState {
    last_hash: Option<String>,
    last_rejected_kinds: std::collections::HashSet<String>,
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
        Self::new_with_env_id("unknown")
    }

    /// Like [`Metrics::new`], stamping `env_id` onto the SLO latency
    /// histograms. Empty ids (standalone DP) collapse to `"unknown"`,
    /// matching the missing-dimension convention used elsewhere.
    pub fn new_with_env_id(env_id: &str) -> Self {
        // Buckets ONLY for the SLO histograms and the guardrail latency
        // histogram: with `metrics-exporter-prometheus`, a distribution
        // without configured buckets renders as a summary — which is what
        // every legacy `histogram!` series here intentionally stays as.
        let recorder = PrometheusBuilder::new()
            .set_buckets_for_metric(
                Matcher::Full(M_REQUEST_E2E_LATENCY_SECONDS.to_string()),
                LATENCY_HISTOGRAM_BUCKETS,
            )
            .expect("static bucket list is non-empty")
            .set_buckets_for_metric(
                Matcher::Full(M_REQUEST_TTFT_SECONDS.to_string()),
                LATENCY_HISTOGRAM_BUCKETS,
            )
            .expect("static bucket list is non-empty")
            .set_buckets_for_metric(
                Matcher::Full(M_GUARDRAIL_LATENCY_SECONDS.to_string()),
                GUARDRAIL_LATENCY_BUCKETS,
            )
            .expect("static bucket list is non-empty")
            .build_recorder();
        let handle = recorder.handle();
        Self {
            inner: Arc::new(MetricsInner {
                recorder,
                handle,
                proxy_in_flight: Mutex::new(HashMap::new()),
                env_id: if env_id.is_empty() {
                    "unknown".to_string()
                } else {
                    env_id.to_string()
                },
                config_labels: Mutex::new(ConfigLabelState::default()),
            }),
        }
    }

    /// Render the current metric values in Prometheus text exposition format.
    pub fn render(&self) -> String {
        self.inner.handle.render()
    }

    /// Reflect the config load-observability state into the recorder. Called
    /// at scrape time by the metrics/status listener so `aisix_config_*`
    /// series always mirror the live [`aisix_core::ConfigStatus`]. Idempotent
    /// and cheap; safe to call on every scrape.
    ///
    /// Etcd-only series (`observed_revision`, `applied_revision`,
    /// `source_connected`) are emitted only in etcd mode. Label churn on the
    /// info/rejected gauges (`hash_info`, `rejected_resources`) zeroes the
    /// prior label set so the exposition never carries two live samples.
    pub fn sync_config_status(&self, view: &aisix_core::ConfigMetricsView) {
        use aisix_core::SourceKind;
        let etcd = matches!(view.source_kind, SourceKind::Etcd);
        metrics::with_local_recorder(&self.inner.recorder, || {
            metrics::gauge!(M_CONFIG_LAST_RELOAD_SUCCESSFUL).set(if view.last_reload_successful {
                1.0
            } else {
                0.0
            });
            if let Some(ts) = view.last_reload_success_ts {
                metrics::gauge!(M_CONFIG_LAST_RELOAD_SUCCESS_TIMESTAMP).set(ts as f64);
            }
            // Counters are tracked authoritatively in ConfigStatus; mirror the
            // absolute value so the counter stays monotonic across scrapes.
            metrics::counter!(M_CONFIG_RELOADS_TOTAL).absolute(view.reloads_total);
            for (reason, count) in &view.reload_failures {
                metrics::counter!(M_CONFIG_RELOAD_FAILURES_TOTAL, "reason" => *reason)
                    .absolute(*count);
            }

            if etcd {
                metrics::gauge!(M_CONFIG_SOURCE_CONNECTED).set(if view.connected == Some(true) {
                    1.0
                } else {
                    0.0
                });
                if let Some(rev) = view.observed_revision {
                    metrics::gauge!(M_CONFIG_OBSERVED_REVISION).set(rev as f64);
                }
                if let Some(rev) = view.applied_revision {
                    metrics::gauge!(M_CONFIG_APPLIED_REVISION).set(rev as f64);
                }
            }

            let mut labels = self.inner.config_labels.lock().expect("config label state");

            // Info-style hash gauge: exactly one live `hash_info{hash=…} 1`;
            // the previously-current hash is zeroed on change so a scraper can
            // filter `== 1`. The `hash` label churns as the applied config
            // changes, and a zeroed series is retained by the recorder — but
            // the churn is bounded by the number of DISTINCT config states
            // (operator/CP edits, a low-frequency event — never per-request),
            // not by traffic. The series name is part of the frozen cross-plane
            // metric contract the control plane also exposes, so it stays.
            if labels.last_hash.as_deref() != view.config_hash.as_deref() {
                if let Some(prev) = labels.last_hash.take() {
                    metrics::gauge!(M_CONFIG_HASH_INFO, "hash" => prev).set(0.0);
                }
            }
            if let Some(hash) = &view.config_hash {
                metrics::gauge!(M_CONFIG_HASH_INFO, "hash" => hash.clone()).set(1.0);
                labels.last_hash = Some(hash.clone());
            }

            // Rejected-resource gauge per kind: set current, zero cleared kinds.
            for kind in &labels.last_rejected_kinds {
                if !view.rejected_by_kind.contains_key(kind) {
                    metrics::gauge!(M_CONFIG_REJECTED_RESOURCES, "kind" => kind.clone()).set(0.0);
                }
            }
            for (kind, count) in &view.rejected_by_kind {
                metrics::gauge!(M_CONFIG_REJECTED_RESOURCES, "kind" => kind.clone())
                    .set(*count as f64);
            }
            labels.last_rejected_kinds = view.rejected_by_kind.keys().cloned().collect();
        });
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

    /// Record one request's guardrail outcome. Called once per request from
    /// the centralised telemetry emit, using the same data as the UsageEvent's
    /// `guardrail_blocked` / `guardrail_bypassed_reason` fields. An empty
    /// `bypass_reason` means no bypass occurred.
    pub fn record_guardrail_outcome(&self, blocked: bool, bypass_reason: &str) {
        metrics::with_local_recorder(&self.inner.recorder, || {
            if blocked {
                metrics::counter!(M_GUARDRAIL_BLOCKS_TOTAL).increment(1);
            }
            if !bypass_reason.is_empty() {
                metrics::counter!(
                    M_GUARDRAIL_BYPASSES_TOTAL,
                    "reason" => bypass_reason.to_string(),
                )
                .increment(1);
            }
        });
    }

    /// Record one guardrail member execution on
    /// [`M_GUARDRAIL_LATENCY_SECONDS`] (AISIX-Cloud#1076). Called by the
    /// chain fold through the `GuardrailMetricsSink` impl below — once per
    /// member per hook pass, on every handler.
    pub fn record_guardrail_execution(&self, exec: &aisix_core::GuardrailExecution<'_>) {
        metrics::with_local_recorder(&self.inner.recorder, || {
            metrics::histogram!(
                M_GUARDRAIL_LATENCY_SECONDS,
                "env_id" => self.inner.env_id.clone(),
                "guardrail" => exec.guardrail_name.to_string(),
                "kind" => exec.kind.to_string(),
                "phase" => exec.phase.to_string(),
                "result" => exec.result.to_string(),
                "error_type" => exec.error_type.unwrap_or("none").to_string(),
            )
            .record(exec.elapsed.as_secs_f64());
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
                "provider_key_name" => labels.provider_key_name.to_string(),
                "api_key_id" => labels.api_key_id.to_string(),
                "team_id" => labels.team_id.to_string(),
                "user_id" => labels.user_id.to_string(),
                "user_name" => labels.user_name.to_string(),
                "stream" => bool_str(labels.stream),
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
                "provider_key_name" => labels.provider_key_name.to_string(),
                "api_key_id" => labels.api_key_id.to_string(),
                "team_id" => labels.team_id.to_string(),
                "user_id" => labels.user_id.to_string(),
                "user_name" => labels.user_name.to_string(),
                "stream" => bool_str(labels.stream),
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
                "provider_key_name" => labels.provider_key_name.to_string(),
                "api_key_id" => labels.api_key_id.to_string(),
                "team_id" => labels.team_id.to_string(),
                "user_id" => labels.user_id.to_string(),
                "user_name" => labels.user_name.to_string(),
            )
            .record(ttft.as_secs_f64());
        });
    }

    /// #890 req-4: record token volume for the inbound `client_type` on the
    /// dedicated [`M_LLM_TOKENS_BY_CLIENT_TOTAL`] series. `client_type` MUST
    /// come from [`ClientTypeClassifier::classify`] (or the built-in
    /// [`client_type_from_user_agent`]) — never raw client input — so the
    /// value set stays bounded by built-ins ∪ boot-validated operator rules
    /// (AISIX-Cloud#1045); zero dims are skipped to keep the series sparse.
    ///
    /// `model` (AISIX-Cloud#1044) is the requested logical model — callers
    /// MUST pass the same value they put in [`UsageLabels::model`] (or its
    /// endpoint's equivalent), never the raw client string of an unresolved
    /// request nor the routed `upstream_model`, so the label stays bounded by
    /// the configured model set and joins cleanly with the `aisix_llm_*`
    /// families.
    ///
    /// `total_tokens` is the caller's canonical cache-inclusive total
    /// (`input + output + Anthropic cache_creation/cache_read`), emitted under
    /// `token_type="total"` (AISIX-Cloud#1002). It is passed in — not derived
    /// from `input + output` — because Anthropic reports cache tokens as
    /// counters SEPARATE from `input_tokens`, so a prompt+completion sum
    /// undercounts cached traffic (same reason as [`total_tokens_with_cache`]
    /// and the `aisix_llm_total_tokens_total` fix in #679).
    pub fn record_llm_tokens_by_client(
        &self,
        client_type: &str,
        model: &str,
        input_tokens: u64,
        output_tokens: u64,
        total_tokens: u64,
    ) {
        if input_tokens == 0 && output_tokens == 0 && total_tokens == 0 {
            return;
        }
        metrics::with_local_recorder(&self.inner.recorder, || {
            if input_tokens > 0 {
                metrics::counter!(
                    M_LLM_TOKENS_BY_CLIENT_TOTAL,
                    "client_type" => client_type.to_string(),
                    "model" => model.to_string(),
                    "token_type" => "input",
                )
                .increment(input_tokens);
            }
            if output_tokens > 0 {
                metrics::counter!(
                    M_LLM_TOKENS_BY_CLIENT_TOTAL,
                    "client_type" => client_type.to_string(),
                    "model" => model.to_string(),
                    "token_type" => "output",
                )
                .increment(output_tokens);
            }
            if total_tokens > 0 {
                metrics::counter!(
                    M_LLM_TOKENS_BY_CLIENT_TOTAL,
                    "client_type" => client_type.to_string(),
                    "model" => model.to_string(),
                    "token_type" => "total",
                )
                .increment(total_tokens);
            }
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

    /// Observe one request's client-perceived end-to-end latency on
    /// [`M_REQUEST_E2E_LATENCY_SECONDS`]. Call exactly once per request:
    /// at handler return for non-streaming requests and failures, at
    /// stream completion for committed streams.
    pub fn record_request_e2e_latency(&self, labels: LatencyLabels<'_>, elapsed: Duration) {
        metrics::with_local_recorder(&self.inner.recorder, || {
            metrics::histogram!(
                M_REQUEST_E2E_LATENCY_SECONDS,
                "env_id" => self.inner.env_id.clone(),
                "endpoint" => labels.endpoint.to_string(),
                "model" => non_empty_or_unknown(labels.model),
                "provider" => non_empty_or_unknown(labels.provider),
                "status_class" => status_bucket(labels.status),
                "streaming" => bool_str(labels.streaming),
            )
            .record(elapsed.as_secs_f64());
        });
    }

    /// Observe a streaming request's time-to-first-token on
    /// [`M_REQUEST_TTFT_SECONDS`]. Zero durations are skipped (TTFT was
    /// never measured — e.g. the stream died before the first token).
    pub fn record_request_ttft(&self, labels: LatencyLabels<'_>, ttft: Duration) {
        if ttft.is_zero() {
            return;
        }
        metrics::with_local_recorder(&self.inner.recorder, || {
            metrics::histogram!(
                M_REQUEST_TTFT_SECONDS,
                "env_id" => self.inner.env_id.clone(),
                "endpoint" => labels.endpoint.to_string(),
                "model" => non_empty_or_unknown(labels.model),
                "provider" => non_empty_or_unknown(labels.provider),
                "status_class" => status_bucket(labels.status),
                "streaming" => bool_str(labels.streaming),
            )
            .record(ttft.as_secs_f64());
        });
    }
}

/// The injection point the guardrail chain records through
/// (AISIX-Cloud#1076): `aisix-guardrails` sees only this core trait, so it
/// stays free of a metrics dependency.
impl aisix_core::GuardrailMetricsSink for Metrics {
    fn record_guardrail_execution(&self, exec: &aisix_core::GuardrailExecution<'_>) {
        Metrics::record_guardrail_execution(self, exec);
    }
}

/// Labels for the SLO latency histograms (AISIX-Cloud#1011). Deliberately
/// low-cardinality: bounded endpoint set, configured model/provider names,
/// bucketed status — never per-key / per-user dimensions, which would
/// multiply every bucket edge.
#[derive(Clone, Copy)]
pub struct LatencyLabels<'a> {
    /// Route template, e.g. `/v1/chat/completions`. Bounded set.
    pub endpoint: &'a str,
    /// Gateway-level model name (the dashboard alias the caller requested).
    pub model: &'a str,
    /// Provider kind (`openai`, `anthropic`, …); `unknown` pre-resolution.
    pub provider: &'a str,
    /// Raw HTTP status; bucketed to `2xx`/`4xx`/… at record time.
    pub status: u16,
    pub streaming: bool,
}

/// Missing dimensions default to `"unknown"`, never an empty label value.
fn non_empty_or_unknown(v: &str) -> String {
    if v.is_empty() {
        "unknown".to_string()
    } else {
        v.to_string()
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

/// Render a boolean metric dimension as a stable `"true"`/`"false"` label
/// value (#890 reqs 1 & 2: `stream`, `is_fallback`).
fn bool_str(value: bool) -> &'static str {
    if value {
        "true"
    } else {
        "false"
    }
}

/// Normalise a raw inbound `User-Agent` into a BOUNDED `client_type` label
/// for [`M_LLM_TOKENS_BY_CLIENT_TOTAL`] (#890 req-4).
///
/// The result is always one of a fixed allowlist (plus `"other"` /
/// `"unknown"`), returned as `&'static str`, so a client-controlled header
/// can never grow prometheus cardinality. Matching is case-insensitive and
/// substring-based, most-specific first (SDK/tool names win over the generic
/// HTTP-library buckets, whose UA a higher-level SDK often embeds). The full
/// user-agent and its version are preserved on the `UsageEvent`/logs — only
/// this coarse, bounded type ever becomes a metric label.
pub fn client_type_from_user_agent(user_agent: &str) -> &'static str {
    let ua = user_agent.trim().to_ascii_lowercase();
    if ua.is_empty() {
        return "unknown";
    }
    // (substring, label) — ordered: products/SDKs before generic libs.
    const TABLE: &[(&str, &str)] = &[
        ("claude-cli", "claude-code"),
        ("claude-code", "claude-code"),
        ("codex", "codex"),
        ("cline", "cline"),
        // Cline-family forks (AISIX-Cloud#1045). Each sends `<Product>/<ver>`
        // on its OpenAI-compatible provider path (Roo since PR #5492,
        // Kilo ≤5.16.2, Zoo ≥3.54.0); the second spelling covers the
        // `roo-code/<ver> (<os>; <arch>)`-style native-path variants.
        ("roo-code", "roo-code"),
        ("roocode", "roo-code"),
        ("kilo-code", "kilocode"),
        ("kilocode", "kilocode"),
        ("zoo-code", "zoo-code"),
        ("zoocode", "zoo-code"),
        // VS Code Copilot Chat BYOK sends `GitHubCopilotChat/<ver>` from
        // the user's machine; the broader needle also catches other
        // `GitHubCopilot*` variants should they surface in a UA.
        ("githubcopilot", "github-copilot"),
        // Cursor routes BYO-endpoint traffic through its own backend,
        // which presents `Cursor/1.0` (version segment is fixed).
        ("cursor", "cursor"),
        // Terminal agents / editors (AISIX-Cloud#1045). opencode PREFIXES
        // the Vercel AI SDK UA (`opencode/<ver> ai-sdk/…`), so it must
        // stay ahead of the `ai-sdk/provider-utils` bucket below. Qwen
        // Code sends `QwenCode/<ver> (<os>; <arch>)` on OpenAI paths but
        // masquerades as `claude-cli/…` toward non-Anthropic hosts on its
        // Anthropic path — that traffic lands in `claude-code`, which a
        // substring table cannot untangle. Gemini CLI embeds the surface
        // (`GeminiCLI-tui/<ver>/<model> (…)`). `zed/` keeps the slash so
        // the needle requires the `Zed/<ver>` token form.
        ("opencode", "opencode"),
        ("qwencode", "qwen-code"),
        ("geminicli", "gemini-cli"),
        ("charm-crush", "crush"),
        ("zed/", "zed"),
        ("aider", "aider"),
        ("openai-python", "openai-python"),
        ("openai/python", "openai-python"),
        ("openai-node", "openai-node"),
        ("openai/js", "openai-node"),
        ("anthropic-sdk-python", "anthropic-python"),
        ("anthropic/python", "anthropic-python"),
        ("anthropic-sdk-typescript", "anthropic-typescript"),
        ("anthropic/js", "anthropic-typescript"),
        ("langchain", "langchain"),
        ("llama-index", "llamaindex"),
        ("llama_index", "llamaindex"),
        ("llamaindex", "llamaindex"),
        ("litellm", "litellm"),
        // Vercel AI SDK default UA (`ai/<v> ai-sdk/provider-utils/<v>
        // runtime/<rt>`) — the whole-SDK bucket for tools that don't
        // override it (Cline 4.x, Kilo Code 7.x, …).
        ("ai-sdk/provider-utils", "vercel-ai-sdk"),
        ("curl", "curl"),
        ("python-requests", "python-requests"),
        ("python-httpx", "httpx"),
        ("httpx", "httpx"),
        ("aiohttp", "aiohttp"),
        ("okhttp", "okhttp"),
        ("go-http-client", "go-http-client"),
        ("node-fetch", "node"),
        ("undici", "node"),
        ("axios", "node"),
        ("postmanruntime", "postman"),
        ("mozilla", "browser"),
    ];
    for (needle, label) in TABLE {
        if ua.contains(needle) {
            return label;
        }
    }
    "other"
}

/// Boot-compiled `client_type` classifier: operator rules from
/// `observability.metrics.client_type_rules` (AISIX-Cloud#1045) tried in
/// config order first, then the built-in
/// [`client_type_from_user_agent`] allowlist. Custom rules deliberately
/// outrank built-ins so a deployment can re-bucket anything — e.g. an
/// in-house tool whose UA embeds `axios` and would otherwise land in
/// `node`. Cardinality stays bounded: a match emits the rule's fixed
/// `client` value (validated at compile), never request-derived text.
#[derive(Debug, Default)]
pub struct ClientTypeClassifier {
    rules: Vec<(regex::Regex, String)>,
}

impl ClientTypeClassifier {
    pub const MAX_RULES: usize = 64;
    pub const MAX_PATTERN_LEN: usize = 512;
    pub const MAX_CLIENT_LEN: usize = 64;

    /// Built-ins only — the behaviour of every deployment without
    /// `client_type_rules` configured.
    pub fn builtin() -> Self {
        Self::default()
    }

    /// Compile + validate operator rules. Errors are boot-fatal by design
    /// (a silently dropped rule would misattribute traffic until someone
    /// notices the label is missing).
    pub fn compile(rules: &[aisix_core::ClientTypeRule]) -> Result<Self, String> {
        if rules.len() > Self::MAX_RULES {
            return Err(format!(
                "observability.metrics.client_type_rules: {} rules exceed the limit of {}",
                rules.len(),
                Self::MAX_RULES
            ));
        }
        let mut compiled = Vec::with_capacity(rules.len());
        for (i, rule) in rules.iter().enumerate() {
            let ctx = format!("observability.metrics.client_type_rules[{i}]");
            if rule.pattern.is_empty() || rule.pattern.len() > Self::MAX_PATTERN_LEN {
                return Err(format!(
                    "{ctx}: pattern must be 1..={} bytes",
                    Self::MAX_PATTERN_LEN
                ));
            }
            if !valid_client_label(&rule.client) {
                return Err(format!(
                    "{ctx}: client {:?} must match [a-z0-9][a-z0-9._-]* and be at most {} chars",
                    rule.client,
                    Self::MAX_CLIENT_LEN
                ));
            }
            let re = regex::RegexBuilder::new(&rule.pattern)
                .case_insensitive(true)
                .build()
                .map_err(|e| format!("{ctx}: invalid pattern: {e}"))?;
            compiled.push((re, rule.client.clone()));
        }
        Ok(Self { rules: compiled })
    }

    /// Classify a raw inbound `User-Agent`. Empty/whitespace UA is always
    /// `unknown` (custom rules never see it — `unknown` keeps meaning "the
    /// client sent nothing"); then custom rules in config order (first
    /// match wins); then the built-in table; then `other`.
    pub fn classify<'a>(&'a self, user_agent: &str) -> &'a str {
        let ua = user_agent.trim();
        if ua.is_empty() {
            return "unknown";
        }
        for (re, client) in &self.rules {
            if re.is_match(ua) {
                return client;
            }
        }
        client_type_from_user_agent(ua)
    }
}

/// Prometheus-safe label value: lowercase alnum start, then alnum/`.`/`_`/`-`.
fn valid_client_label(label: &str) -> bool {
    if label.is_empty() || label.len() > ClientTypeClassifier::MAX_CLIENT_LEN {
        return false;
    }
    let mut chars = label.chars();
    let first = chars.next().expect("non-empty checked above");
    (first.is_ascii_lowercase() || first.is_ascii_digit())
        && chars
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '.' | '_' | '-'))
}

#[derive(Debug, Clone, Copy)]
pub struct RequestLabels<'a> {
    pub endpoint: &'a str,
    pub inbound_protocol: &'a str,
    pub provider: &'a str,
    pub model: &'a str,
    pub upstream_model: &'a str,
    pub provider_key_id: &'a str,
    /// Readable provider-key name (#890 req-3). 1:1 with `provider_key_id`
    /// so it adds no new series; `"unknown"` when unresolved.
    pub provider_key_name: &'a str,
    pub api_key_id: &'a str,
    pub team_id: &'a str,
    pub user_id: &'a str,
    /// Readable user display name (#890 req-3). 1:1 with `user_id`;
    /// `"unknown"` until cp-api syncs it onto the api-key config.
    pub user_name: &'a str,
    /// Whether the client requested a streaming (SSE) response (#890 req-1).
    /// Emitted on the request counter AND duration histogram so a
    /// TTFT-vs-E2E comparison can restrict the E2E latency to the same
    /// streaming-only sample TTFT is measured on.
    pub stream: bool,
    /// Whether serving this request involved a fallback to a different
    /// routing target (#890 req-2). Emitted on the request COUNTERS only
    /// (a success-rate dimension — kept off the bucketed histograms to
    /// avoid ×2 per latency bucket) so a success rate can exclude fallback
    /// requests from the denominator.
    pub is_fallback: bool,
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
            provider_key_name: "unknown",
            api_key_id: "unknown",
            team_id: "unknown",
            user_id: "unknown",
            user_name: "unknown",
            stream: false,
            is_fallback: false,
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
            "provider_key_name" => self.provider_key_name.to_string(),
            "api_key_id" => self.api_key_id.to_string(),
            "team_id" => self.team_id.to_string(),
            "user_id" => self.user_id.to_string(),
            "user_name" => self.user_name.to_string(),
            "stream" => bool_str(self.stream),
            "is_fallback" => bool_str(self.is_fallback),
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
    /// Readable provider-key name (#890 req-3). 1:1 with `provider_key_id`.
    pub provider_key_name: &'a str,
    pub api_key_id: &'a str,
    pub team_id: &'a str,
    pub user_id: &'a str,
    /// Readable user display name (#890 req-3). 1:1 with `user_id`.
    pub user_name: &'a str,
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
            provider_key_name: "unknown",
            api_key_id: "unknown",
            team_id: "unknown",
            user_id: "unknown",
            user_name: "unknown",
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
            "provider_key_name" => self.provider_key_name.to_string(),
            "api_key_id" => self.api_key_id.to_string(),
            "team_id" => self.team_id.to_string(),
            "user_id" => self.user_id.to_string(),
            "user_name" => self.user_name.to_string(),
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
            "provider_key_name" => self.provider_key_name.to_string(),
            "api_key_id" => self.api_key_id.to_string(),
            "team_id" => self.team_id.to_string(),
            "user_id" => self.user_id.to_string(),
            "user_name" => self.user_name.to_string(),
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
    pub user_id: &'a str,
}

impl Default for BudgetLabels<'_> {
    fn default() -> Self {
        Self {
            api_key_id: "unknown",
            team_id: "unknown",
            user_id: "unknown",
        }
    }
}

impl BudgetLabels<'_> {
    fn record_gauge(&self, metric: &'static str, value: f64) {
        metrics::gauge!(
            metric,
            "api_key_id" => self.api_key_id.to_string(),
            "team_id" => self.team_id.to_string(),
            "user_id" => self.user_id.to_string(),
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

    /// AISIX-Cloud#1076: the per-execution guardrail histogram renders with
    /// real `_bucket{le=…}` series (quantile-aggregatable, not a summary)
    /// and the full bounded label set; `error_type` defaults to `none`.
    #[test]
    fn guardrail_execution_renders_bucketed_histogram_with_labels() {
        let m = Metrics::new_with_env_id("env-7");
        m.record_guardrail_execution(&aisix_core::GuardrailExecution {
            guardrail_name: "block-secrets",
            kind: "keyword",
            phase: "input",
            result: "blocked",
            error_type: None,
            elapsed: Duration::from_micros(300),
        });
        m.record_guardrail_execution(&aisix_core::GuardrailExecution {
            guardrail_name: "lakera-prod",
            kind: "lakera",
            phase: "output",
            result: "bypassed",
            error_type: Some("lakera_timeout"),
            elapsed: Duration::from_secs(5),
        });
        let out = m.render();
        assert!(
            out.contains("aisix_guardrail_latency_seconds_bucket"),
            "{out}"
        );
        // 0.3 ms lands in the first (1 ms) bucket — the sub-SLO edges exist.
        assert!(out.contains("le=\"0.001\""), "{out}");
        assert!(out.contains("env_id=\"env-7\""));
        assert!(out.contains("guardrail=\"block-secrets\""));
        assert!(out.contains("kind=\"keyword\""));
        assert!(out.contains("phase=\"input\""));
        assert!(out.contains("result=\"blocked\""));
        assert!(out.contains("error_type=\"none\""));
        assert!(out.contains("guardrail=\"lakera-prod\""));
        assert!(out.contains("result=\"bypassed\""));
        assert!(out.contains("error_type=\"lakera_timeout\""));
        // No per-key/per-user dimension may ride a bucketed histogram.
        assert!(!out.contains("api_key_id="));
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
    fn guardrail_outcome_counters_increment() {
        let m = Metrics::new(false);
        m.record_guardrail_outcome(true, ""); // blocked, no bypass
        m.record_guardrail_outcome(false, "bedrock_5xx"); // fail-open bypass
        m.record_guardrail_outcome(false, ""); // clean request → records nothing
        let rendered = m.render();
        // Exactly one block (the clean call must not increment it).
        assert!(
            rendered.contains(&format!("{M_GUARDRAIL_BLOCKS_TOTAL} 1")),
            "want one block, got:\n{rendered}"
        );
        // Exactly one bypass, sliced by the bounded reason — pinning the count
        // proves the blocked + clean calls didn't touch the bypass counter.
        assert!(
            rendered.contains(&format!(
                "{M_GUARDRAIL_BYPASSES_TOTAL}{{reason=\"bedrock_5xx\"}} 1"
            )),
            "want exactly one bedrock_5xx bypass, got:\n{rendered}"
        );
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
            provider_key_name: "my-openai-key",
            api_key_id: "ak-1",
            team_id: "team-1",
            user_id: "user-1",
            user_name: "alice",
            stream: true,
            is_fallback: true,
            status: 200,
            outcome: RequestOutcome::Success,
        };
        let usage_labels = UsageLabels {
            endpoint: "/v1/chat/completions",
            inbound_protocol: "openai",
            provider: "openai",
            model: "gpt",
            upstream_model: "gpt-4o",
            provider_key_id: "pk-1",
            provider_key_name: "my-openai-key",
            api_key_id: "ak-1",
            team_id: "team-1",
            user_id: "user-1",
            user_name: "alice",
        };
        m.record_proxy_request(labels, Duration::from_millis(25));
        m.record_llm_request(labels, Duration::from_millis(20));
        m.record_llm_usage(
            usage_labels,
            LlmUsage {
                input_tokens: 5,
                output_tokens: 7,
                total_tokens: 12,
                spend_usd: 0.001,
            },
        );
        m.record_time_to_first_token(usage_labels, Duration::from_millis(42));

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
        assert!(rendered.contains("user_id=\"user-1\""));
        // #890 req-3: readable names ride alongside the ids (1:1).
        assert!(rendered.contains("provider_key_name=\"my-openai-key\""));
        assert!(rendered.contains("user_name=\"alice\""));
        // #890 req-1/req-2: stream on counter + duration; is_fallback on
        // the counter only (verified absent from the duration below).
        assert!(rendered.contains("stream=\"true\""));
        assert!(rendered.contains("is_fallback=\"true\""));
        // is_fallback must NOT appear on the duration histogram series.
        for line in rendered.lines() {
            if line.starts_with(M_LLM_REQUEST_DURATION)
                || line.starts_with(M_PROXY_REQUEST_DURATION)
            {
                assert!(
                    !line.contains("is_fallback="),
                    "is_fallback must stay off the duration histogram: {line}"
                );
            }
        }
    }

    #[test]
    fn tokens_by_client_records_bounded_client_type() {
        let m = Metrics::new(false);
        // The caller's canonical total is cache-inclusive, so it can exceed
        // input+output: 155 = 100 + 40 + 15 cache tokens (#1002).
        m.record_llm_tokens_by_client("openai-python", "gpt-4o", 100, 40, 155);
        m.record_llm_tokens_by_client("openai-python", "gpt-4o", 10, 0, 10);
        // All-zero is a no-op (keeps the series sparse).
        m.record_llm_tokens_by_client("curl", "gpt-4o", 0, 0, 0);
        let rendered = m.render();
        assert!(rendered.contains(M_LLM_TOKENS_BY_CLIENT_TOTAL));
        assert!(rendered.contains("client_type=\"openai-python\""));
        assert!(rendered.contains("token_type=\"input\""));
        assert!(rendered.contains("token_type=\"output\""));
        assert!(rendered.contains("token_type=\"total\""));
        // input=110, output=40, total=165 — the total series counts the 15
        // cache tokens the input series omits (165 > 110 + 40).
        assert!(rendered
            .lines()
            .any(|l| l.starts_with("aisix_llm_tokens_by_client_total{")
                && l.contains("token_type=\"total\"")
                && l.contains("model=\"gpt-4o\"")
                && l.trim_end().ends_with(" 165")));
        // The all-zero curl call recorded nothing.
        assert!(!rendered.contains("client_type=\"curl\""));
    }

    #[test]
    fn tokens_by_client_splits_series_per_model() {
        // AISIX-Cloud#1044: one client type spending on two models must
        // produce two independent series per token_type, and every series
        // must carry the model label.
        let m = Metrics::new(false);
        m.record_llm_tokens_by_client("claude-code", "claude-sonnet", 100, 60, 160);
        m.record_llm_tokens_by_client("claude-code", "claude-haiku", 30, 10, 40);
        let rendered = m.render();
        let series: Vec<&str> = rendered
            .lines()
            .filter(|l| l.starts_with("aisix_llm_tokens_by_client_total{"))
            .collect();
        // 2 models × 3 token types, all under the same client_type.
        assert_eq!(series.len(), 6);
        assert!(series
            .iter()
            .all(|l| l.contains("client_type=\"claude-code\"") && l.contains("model=")));
        let value_of = |model: &str, token_type: &str| {
            series
                .iter()
                .find(|l| {
                    l.contains(&format!("model=\"{model}\""))
                        && l.contains(&format!("token_type=\"{token_type}\""))
                })
                .and_then(|l| l.trim_end().rsplit(' ').next())
                .map(|v| v.parse::<u64>().unwrap())
        };
        assert_eq!(value_of("claude-sonnet", "input"), Some(100));
        assert_eq!(value_of("claude-sonnet", "output"), Some(60));
        assert_eq!(value_of("claude-sonnet", "total"), Some(160));
        assert_eq!(value_of("claude-haiku", "input"), Some(30));
        assert_eq!(value_of("claude-haiku", "output"), Some(10));
        assert_eq!(value_of("claude-haiku", "total"), Some(40));
    }

    #[test]
    fn client_type_from_user_agent_normalises_to_allowlist() {
        // Known SDKs/tools normalise to a stable bounded label.
        assert_eq!(
            client_type_from_user_agent("OpenAI/Python 1.30.1"),
            "openai-python"
        );
        assert_eq!(
            client_type_from_user_agent("openai-node/4.20.0"),
            "openai-node"
        );
        assert_eq!(
            client_type_from_user_agent("claude-cli/1.2.3"),
            "claude-code"
        );
        assert_eq!(client_type_from_user_agent("curl/8.4.0"), "curl");
        // Version differences collapse to the SAME bounded type — no
        // per-version cardinality blowup.
        assert_eq!(
            client_type_from_user_agent("OpenAI/Python 1.0.0"),
            client_type_from_user_agent("OpenAI/Python 2.99.9"),
        );
        // Empty → unknown; unrecognised → other (the only unbounded inputs
        // both collapse into bounded buckets).
        assert_eq!(client_type_from_user_agent(""), "unknown");
        assert_eq!(client_type_from_user_agent("   "), "unknown");
        assert_eq!(
            client_type_from_user_agent("SomeRandomBespokeClient/9.9"),
            "other"
        );
    }

    /// AISIX-Cloud#1045: coding clients added from source-verified UA
    /// samples (real formats quoted from each product's provider code —
    /// see the issue's evidence table).
    #[test]
    fn client_type_recognises_coding_clients_1045() {
        // Cline v3.56+ sends `Cline/<ver>` on both BYO paths (PR #8872).
        assert_eq!(client_type_from_user_agent("Cline/3.89.2"), "cline");
        // Roo Code OpenAI-compatible path (DEFAULT_HEADERS since PR #5492)
        // and its `roo-code/<ver> (<os>; <arch>)` native-path variant.
        assert_eq!(client_type_from_user_agent("RooCode/3.54.0"), "roo-code");
        assert_eq!(
            client_type_from_user_agent("roo-code/3.54.0 (darwin 23.5.0; arm64) node/20.19.0"),
            "roo-code"
        );
        // Kilo Code ≤5.16.2 (legacy Roo fork lineage).
        assert_eq!(client_type_from_user_agent("Kilo-Code/5.16.2"), "kilocode");
        // Zoo Code — the community continuation of archived Roo Code;
        // marketplace builds carry large patch numbers.
        assert_eq!(
            client_type_from_user_agent("ZooCode/3.71.100268"),
            "zoo-code"
        );
        // Vercel AI SDK default UA — the bucket for AI-SDK-based tools
        // that don't override it (Cline 4.x on node, Kilo 7.x on bun).
        assert_eq!(
            client_type_from_user_agent(
                "ai/6.0.144 ai-sdk/provider-utils/4.0.22 runtime/node.js/26"
            ),
            "vercel-ai-sdk"
        );
        assert_eq!(
            client_type_from_user_agent(
                "ai/6.0.168 ai-sdk/provider-utils/4.0.29 runtime/bun/1.3.6"
            ),
            "vercel-ai-sdk"
        );
        // VS Code Copilot Chat BYOK (nodeFetcher.ts default UA).
        assert_eq!(
            client_type_from_user_agent("GitHubCopilotChat/0.44.0"),
            "github-copilot"
        );
        // Cursor's backend presents a fixed version segment.
        assert_eq!(client_type_from_user_agent("Cursor/1.0"), "cursor");
        // Copilot CLI BYOK exposes only the SDK UA — classified as the
        // SDK, not as Copilot (identification limit recorded in #1045).
        assert_eq!(
            client_type_from_user_agent("OpenAI/JS 5.20.1"),
            "openai-node"
        );
        // opencode prefixes the AI-SDK UA — the product token must win
        // over the `ai-sdk/provider-utils` bucket also present in the UA.
        assert_eq!(
            client_type_from_user_agent(
                "opencode/1.18.3 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14"
            ),
            "opencode"
        );
        // Qwen Code, OpenAI-compatible path (live-captured format).
        assert_eq!(
            client_type_from_user_agent("QwenCode/0.20.0 (linux; x64)"),
            "qwen-code"
        );
        // Qwen Code's Anthropic path masquerades as Claude Code toward
        // gateways — a KNOWN collision: it lands in `claude-code`.
        assert_eq!(
            client_type_from_user_agent("claude-cli/0.20.0 (external, cli)"),
            "claude-code"
        );
        // Gemini CLI (Gemini-protocol only today; UA still recognised).
        assert_eq!(
            client_type_from_user_agent(
                "GeminiCLI-tui/0.51.0/gemini-3.1-pro-preview (linux; x64; terminal)"
            ),
            "gemini-cli"
        );
        // Crush and Zed set product UAs on their shared HTTP clients.
        assert_eq!(
            client_type_from_user_agent("Charm-Crush/1.0.0 (https://charm.land/crush)"),
            "crush"
        );
        assert_eq!(
            client_type_from_user_agent("Zed/0.198.0 (linux; x86_64)"),
            "zed"
        );
    }

    /// AISIX-Cloud#1045: operator rules outrank built-ins, first match
    /// wins, non-matches fall back to the built-in table, and empty UA
    /// stays `unknown` even under a match-anything rule.
    #[test]
    fn classifier_custom_rules_first_match_then_builtin_fallback() {
        let rule = |pattern: &str, client: &str| aisix_core::ClientTypeRule {
            pattern: pattern.into(),
            client: client.into(),
        };
        let c = ClientTypeClassifier::compile(&[
            rule("^internal-agent/", "internal-agent"),
            // Overlaps the rule above — order decides (first match wins).
            rule("internal", "internal-other"),
            // Re-buckets a UA the built-in table would call "node".
            rule("billing-batcher", "billing-batcher"),
            rule(".*", "catch-all"),
        ])
        .expect("valid rules");

        assert_eq!(c.classify("internal-agent/2.1"), "internal-agent");
        assert_eq!(c.classify("acme-internal-tool/1.0"), "internal-other");
        // Case-insensitive by default.
        assert_eq!(c.classify("Internal-Agent/9.9"), "internal-agent");
        // axios UA would be built-in "node"; the custom rule outranks it.
        assert_eq!(
            c.classify("billing-batcher/3.0 axios/1.6.0"),
            "billing-batcher"
        );
        // Empty/whitespace UA never reaches custom rules — even ".*".
        assert_eq!(c.classify(""), "unknown");
        assert_eq!(c.classify("   "), "unknown");
        // The ".*" rule shadows the built-in fallback for everything else.
        assert_eq!(c.classify("curl/8.4.0"), "catch-all");

        // Without a catch-all, non-matching UAs use the built-in table.
        let c = ClientTypeClassifier::compile(&[rule("^internal-agent/", "internal-agent")])
            .expect("valid rules");
        assert_eq!(c.classify("curl/8.4.0"), "curl");
        assert_eq!(c.classify("SomeRandomBespokeClient/9.9"), "other");

        // Built-ins only (no config) — same behaviour as the free function.
        let c = ClientTypeClassifier::builtin();
        assert_eq!(c.classify("claude-cli/1.2.3"), "claude-code");
        assert_eq!(c.classify(""), "unknown");
    }

    /// AISIX-Cloud#1045: invalid rule sets are rejected at compile (boot)
    /// time — count cap, pattern syntax/length, label charset/length.
    #[test]
    fn classifier_rejects_invalid_rule_sets() {
        let rule = |pattern: &str, client: &str| aisix_core::ClientTypeRule {
            pattern: pattern.into(),
            client: client.into(),
        };
        // Broken regex syntax.
        assert!(ClientTypeClassifier::compile(&[rule("([unclosed", "x")])
            .unwrap_err()
            .contains("invalid pattern"));
        // Empty and oversized patterns.
        assert!(ClientTypeClassifier::compile(&[rule("", "x")]).is_err());
        let oversized = "a".repeat(ClientTypeClassifier::MAX_PATTERN_LEN + 1);
        assert!(ClientTypeClassifier::compile(&[rule(&oversized, "x")]).is_err());
        // Label charset: uppercase, leading dash, spaces, empty, too long.
        for bad in ["Upper", "-lead", "has space", "", "田"] {
            assert!(
                ClientTypeClassifier::compile(&[rule("ok", bad)]).is_err(),
                "label {bad:?} should be rejected"
            );
        }
        let long_label = "a".repeat(ClientTypeClassifier::MAX_CLIENT_LEN + 1);
        assert!(ClientTypeClassifier::compile(&[rule("ok", &long_label)]).is_err());
        // Valid edge labels pass.
        assert!(ClientTypeClassifier::compile(&[rule("ok", "0-tool_v2.beta")]).is_ok());
        // Rule-count cap.
        let too_many: Vec<_> = (0..=ClientTypeClassifier::MAX_RULES)
            .map(|i| rule(&format!("tool-{i}"), "tool"))
            .collect();
        assert!(ClientTypeClassifier::compile(&too_many)
            .unwrap_err()
            .contains("exceed"));
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

    /// AISIX-Cloud#1011: the two SLO series must render as REAL bucketed
    /// histograms (`_bucket{le=…}` + `_sum`/`_count`) — the property that
    /// makes `histogram_quantile()` and cross-instance aggregation work.
    /// Every other `histogram!` series stays a summary (no buckets), so a
    /// bucket-config regression is invisible without this pin.
    #[test]
    fn slo_latency_series_render_as_bucketed_histograms() {
        let m = Metrics::new_with_env_id("env-42");
        let labels = LatencyLabels {
            endpoint: "/v1/chat/completions",
            model: "gpt-4o",
            provider: "openai",
            status: 200,
            streaming: false,
        };
        m.record_request_e2e_latency(labels, Duration::from_millis(1500));
        m.record_request_ttft(
            LatencyLabels {
                streaming: true,
                ..labels
            },
            Duration::from_millis(80),
        );
        let out = m.render();

        // Real histogram exposition: le-bucketed series + sum/count.
        assert!(
            out.contains("aisix_request_e2e_latency_seconds_bucket"),
            "e2e series must expose _bucket lines:\n{out}"
        );
        assert!(out.contains("aisix_request_e2e_latency_seconds_sum"));
        assert!(out.contains("aisix_request_e2e_latency_seconds_count"));
        assert!(out.contains("aisix_request_ttft_seconds_bucket"));
        assert!(
            out.contains("le=\"2.5\""),
            "configured bucket edges present"
        );

        // The label contract: constant env_id, bucketed status, bounded
        // dims — and none of the per-key/per-user dimensions.
        assert!(out.contains("env_id=\"env-42\""));
        assert!(out.contains("status_class=\"2xx\""));
        assert!(out.contains("streaming=\"false\""));
        assert!(out.contains("streaming=\"true\""));
        for high_card in ["api_key_id", "user_id", "team_id", "provider_key_id"] {
            for line in out.lines().filter(|l| l.contains("aisix_request_")) {
                assert!(
                    !line.contains(high_card),
                    "SLO histogram must not carry {high_card}: {line}"
                );
            }
        }
    }

    /// A 1.5s observation lands in the 2.5 bucket but not the 1.0 bucket —
    /// pins that the configured edges actually apply (a default-bucket
    /// fallback would place them differently or render a summary).
    #[test]
    fn slo_latency_observation_lands_in_the_right_bucket() {
        let m = Metrics::new_with_env_id("");
        m.record_request_e2e_latency(
            LatencyLabels {
                endpoint: "/v1/messages",
                model: "m",
                provider: "anthropic",
                status: 502,
                streaming: false,
            },
            Duration::from_millis(1500),
        );
        let out = m.render();
        let bucket_val = |le: &str| -> u64 {
            out.lines()
                .find(|l| {
                    l.starts_with("aisix_request_e2e_latency_seconds_bucket")
                        && l.contains(&format!("le=\"{le}\""))
                })
                .and_then(|l| l.rsplit(' ').next())
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| panic!("no bucket le={le} in:\n{out}"))
        };
        assert_eq!(bucket_val("1"), 0, "1.5s must not land in le=1");
        assert_eq!(bucket_val("2.5"), 1, "1.5s must land in le=2.5");
        // Empty env_id collapses to the missing-dimension convention.
        assert!(out.contains("env_id=\"unknown\""));
        assert!(out.contains("status_class=\"5xx\""));
    }

    /// Zero TTFT (never measured) is skipped, and the legacy duration
    /// series keep their summary exposition — no `_bucket` lines appear
    /// for them even after the SLO buckets are configured.
    #[test]
    fn slo_ttft_skips_zero_and_legacy_series_stay_summaries() {
        let m = Metrics::new_with_env_id("e");
        let labels = LatencyLabels {
            endpoint: "/v1/chat/completions",
            model: "m",
            provider: "openai",
            status: 200,
            streaming: true,
        };
        m.record_request_ttft(labels, Duration::ZERO);
        assert!(
            !m.render().contains("aisix_request_ttft_seconds"),
            "zero TTFT must not be observed"
        );

        m.record_request(
            "openai",
            "m",
            200,
            RequestOutcome::Success,
            Duration::from_millis(100),
        );
        let out = m.render();
        assert!(
            !out.contains("aisix_request_duration_seconds_bucket"),
            "legacy duration series must stay a summary (quantiles), got:\n{out}"
        );
        assert!(out.contains("aisix_request_duration_seconds"));
    }

    fn config_metrics_view(source_kind: aisix_core::SourceKind) -> aisix_core::ConfigMetricsView {
        aisix_core::ConfigMetricsView {
            source_kind,
            last_reload_successful: true,
            last_reload_success_ts: Some(1_760_000_000),
            reloads_total: 3,
            reload_failures: std::collections::BTreeMap::new(),
            rejected_by_kind: std::collections::BTreeMap::new(),
            observed_revision: Some(42),
            applied_revision: Some(42),
            config_hash: Some("abc123".into()),
            connected: Some(true),
        }
    }

    #[test]
    fn config_status_sync_renders_all_series_in_etcd_mode() {
        let m = Metrics::new(false);
        let mut view = config_metrics_view(aisix_core::SourceKind::Etcd);
        view.last_reload_successful = false;
        view.reload_failures.insert("validate", 2);
        view.rejected_by_kind.insert("models".to_string(), 1);
        m.sync_config_status(&view);
        let out = m.render();

        assert!(out.contains(&format!("{M_CONFIG_LAST_RELOAD_SUCCESSFUL} 0")));
        assert!(out.contains(M_CONFIG_LAST_RELOAD_SUCCESS_TIMESTAMP));
        assert!(out.contains(&format!("{M_CONFIG_RELOADS_TOTAL} 3")));
        assert!(out.contains(&format!(
            "{M_CONFIG_RELOAD_FAILURES_TOTAL}{{reason=\"validate\"}} 2"
        )));
        assert!(out.contains(&format!(
            "{M_CONFIG_REJECTED_RESOURCES}{{kind=\"models\"}} 1"
        )));
        assert!(out.contains(&format!("{M_CONFIG_OBSERVED_REVISION} 42")));
        assert!(out.contains(&format!("{M_CONFIG_APPLIED_REVISION} 42")));
        assert!(out.contains(&format!("{M_CONFIG_HASH_INFO}{{hash=\"abc123\"}} 1")));
        assert!(out.contains(&format!("{M_CONFIG_SOURCE_CONNECTED} 1")));
    }

    #[test]
    fn config_status_sync_omits_etcd_only_series_in_file_mode() {
        let m = Metrics::new(false);
        let view = config_metrics_view(aisix_core::SourceKind::File);
        m.sync_config_status(&view);
        let out = m.render();
        // Source-agnostic series still present.
        assert!(out.contains(M_CONFIG_LAST_RELOAD_SUCCESSFUL));
        assert!(out.contains(M_CONFIG_RELOADS_TOTAL));
        // Etcd-only series absent in file mode.
        assert!(!out.contains(M_CONFIG_OBSERVED_REVISION));
        assert!(!out.contains(M_CONFIG_APPLIED_REVISION));
        assert!(!out.contains(M_CONFIG_SOURCE_CONNECTED));
    }

    #[test]
    fn config_status_sync_zeroes_stale_hash_and_rejected_labels() {
        let m = Metrics::new(false);
        let mut first = config_metrics_view(aisix_core::SourceKind::Etcd);
        first.config_hash = Some("hash-A".into());
        first.rejected_by_kind.insert("models".to_string(), 2);
        m.sync_config_status(&first);

        // The applied config changes and the models rejection clears.
        let mut second = config_metrics_view(aisix_core::SourceKind::Etcd);
        second.config_hash = Some("hash-B".into());
        // rejected_by_kind empty now.
        m.sync_config_status(&second);

        let out = m.render();
        // Exactly one live hash sample: old zeroed, new is 1.
        assert!(out.contains(&format!("{M_CONFIG_HASH_INFO}{{hash=\"hash-A\"}} 0")));
        assert!(out.contains(&format!("{M_CONFIG_HASH_INFO}{{hash=\"hash-B\"}} 1")));
        // The cleared kind is zeroed, not left at its stale count.
        assert!(out.contains(&format!(
            "{M_CONFIG_REJECTED_RESOURCES}{{kind=\"models\"}} 0"
        )));
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
