//! Metric label selections are applied when a series is registered, before
//! counters or distributions accumulate. Filtering scrape text would lose
//! observations when several original series project onto the same labels.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};

use metrics::{
    Counter, CounterFn, Gauge, Histogram, Key, KeyName, Label, Metadata, Recorder, SharedString,
    Unit,
};
use metrics_exporter_prometheus::PrometheusRecorder;
use serde::Serialize;

use crate::metrics::*;

#[derive(Debug, Serialize)]
pub struct MetricVariable {
    pub name: &'static str,
    pub description: &'static str,
}

macro_rules! variable {
    ($name:literal, $description:literal) => {
        MetricVariable {
            name: $name,
            description: $description,
        }
    };
}

pub static METRIC_VARIABLES: &[MetricVariable] = &[
    variable!("env_id", "The gateway's AISIX Cloud environment ID; unknown when not connected to AISIX Cloud. Available on every metric."),
    variable!("endpoint", "The matched route template, without caller-supplied path parameters."),
    variable!("inbound_protocol", "The protocol used by the caller, derived from the matched endpoint."),
    variable!("upstream_protocol", "The wire protocol of the selected provider credential; unknown before upstream selection or for non-LLM traffic."),
    variable!("provider", "The provider kind attributed to this observation; ensemble when no single provider owns the result."),
    variable!("model", "The configured gateway model identity attributed to the observation. Wildcard requests use the configured model pattern. See the metric reference for its request or attempt scope."),
    variable!("upstream_model", "The upstream model identity, bounded to the configured model pattern for wildcard models."),
    variable!("provider_key_id", "The identifier of the selected provider credential. This is an identifier, never its secret."),
    variable!("provider_key_name", "The display name of that same provider credential. Renaming a credential starts a new series when this variable is selected."),
    variable!("api_key_id", "The identifier of the authenticating gateway API key, never its plaintext value."),
    variable!("team_id", "The team associated with the authenticating API key."),
    variable!("user_id", "The member associated with the authenticating API key."),
    variable!("user_name", "The member display name carried by the API key's configuration snapshot."),
    variable!("stream", "Whether the request asked for streaming, encoded as true or false. Used by the detailed request metrics."),
    variable!("streaming", "Whether the observed request is streaming, encoded as true or false. Used by the request latency histograms."),
    variable!("is_fallback", "Whether request attribution identifies a fallback attempt, encoded as true or false."),
    variable!("status", "The HTTP status code on request and usage-event metrics; the HTTP status class on A2A request metrics."),
    variable!("status_class", "The HTTP status class: 2xx, 3xx, 4xx, 5xx, or other."),
    variable!("status_code", "The HTTP status class on usage-event metrics: 2xx, 3xx, 4xx, 5xx, or other. The status variable retains the exact code."),
    variable!("outcome", "The outcome defined by the metric, such as the request result, cache decision, or request-body-limit result."),
    variable!("client_type", "The classified client name from built-in or configured client_type_rules; never the raw User-Agent."),
    variable!("token_type", "The token count category: input, output, or total. Total already includes the input and output categories."),
    variable!("fallback_model", "The configured fallback model attributed to a routing fallback."),
    variable!("scope", "The scope of the rate-limit decision."),
    variable!("layer", "The layer of the rate-limit decision."),
    variable!("policy_id", "The rate-limit policy identifier, or the metric's missing-policy value."),
    variable!("reason", "The bounded reason code defined by the emitting authentication, guardrail, configuration, or exporter metric."),
    variable!("method", "The authentication method used for the credential decision."),
    variable!("result", "The authentication or guardrail execution result, as defined by the metric."),
    variable!("guardrail", "The configured guardrail name."),
    variable!("kind", "The guardrail kind on guardrail metrics, or the resource kind on configuration metrics."),
    variable!("phase", "The guardrail execution phase."),
    variable!("error_type", "The bounded guardrail failure category, or none when no error occurred."),
    variable!("handler", "The usage event's handler family, such as chat, messages, embeddings, or mcp."),
    variable!("policy", "The configured cache policy name."),
    variable!("cause", "The bounded semantic-cache embedding failure cause."),
    variable!("op", "The semantic-cache storage operation."),
    variable!("operation", "The Redis operation or A2A operation, as defined by the metric."),
    variable!("exporter", "The configured observability exporter name."),
    variable!("agent", "The registered A2A agent name."),
    variable!("state", "The A2A task state reported by the upstream agent."),
    variable!("hash", "The hash of the applied gateway resource configuration."),
];

#[derive(Debug, Serialize)]
pub struct MetricDefinition {
    pub name: &'static str,
    pub default_labels: &'static [&'static str],
    pub extra_labels: &'static [&'static str],
    pub required_labels: &'static [&'static str],
}

impl MetricDefinition {
    pub fn supports(&self, label: &str) -> bool {
        label == "env_id"
            || self.default_labels.contains(&label)
            || self.extra_labels.contains(&label)
    }
}

const REQUEST: &[&str] = &[
    "endpoint",
    "inbound_protocol",
    "upstream_protocol",
    "provider",
    "model",
    "upstream_model",
    "provider_key_id",
    "provider_key_name",
    "api_key_id",
    "team_id",
    "user_id",
    "user_name",
    "stream",
    "is_fallback",
    "status",
    "outcome",
];
const REQUEST_DURATION: &[&str] = &[
    "endpoint",
    "inbound_protocol",
    "upstream_protocol",
    "provider",
    "model",
    "upstream_model",
    "provider_key_id",
    "provider_key_name",
    "api_key_id",
    "team_id",
    "user_id",
    "user_name",
    "stream",
    "status",
    "outcome",
];
const USAGE: &[&str] = &[
    "endpoint",
    "inbound_protocol",
    "upstream_protocol",
    "provider",
    "model",
    "upstream_model",
    "provider_key_id",
    "provider_key_name",
    "api_key_id",
    "team_id",
    "user_id",
    "user_name",
];
const LATENCY: &[&str] = &[
    "env_id",
    "endpoint",
    "model",
    "provider",
    "status_class",
    "streaming",
];
const DEPLOYMENT: &[&str] = &["provider", "model", "upstream_model", "provider_key_id"];
const BUDGET: &[&str] = &["api_key_id", "team_id", "user_id", "user_name"];

macro_rules! metric {
    ($name:ident, $defaults:expr) => {
        MetricDefinition {
            name: $name,
            default_labels: $defaults,
            extra_labels: &[],
            required_labels: &[],
        }
    };
    ($name:ident, $defaults:expr, extra = $extra:expr) => {
        MetricDefinition {
            name: $name,
            default_labels: $defaults,
            extra_labels: $extra,
            required_labels: &[],
        }
    };
    ($name:ident, $defaults:expr, required = $required:expr) => {
        MetricDefinition {
            name: $name,
            default_labels: $defaults,
            extra_labels: &[],
            required_labels: $required,
        }
    };
}

pub static METRIC_DEFINITIONS: &[MetricDefinition] = &[
    metric!(
        M_REQUESTS_TOTAL,
        &["provider", "model", "status", "outcome"]
    ),
    metric!(M_REQUEST_DURATION, &["provider", "model", "status"]),
    metric!(M_RATELIMIT_REJECTIONS, &["scope", "layer", "policy_id"]),
    metric!(M_TOKENS_CONSUMED, &["provider", "model"]),
    metric!(M_LLM_SPEND_MICRO_USD_TOTAL, USAGE),
    metric!(M_LLM_INPUT_TOKENS_TOTAL, USAGE),
    metric!(M_LLM_OUTPUT_TOKENS_TOTAL, USAGE),
    metric!(M_LLM_TOTAL_TOKENS_TOTAL, USAGE),
    metric!(M_LLM_CACHED_INPUT_TOKENS_TOTAL, USAGE),
    metric!(M_LLM_CACHE_READ_INPUT_TOKENS_TOTAL, USAGE),
    metric!(M_LLM_CACHE_CREATION_INPUT_TOKENS_TOTAL, USAGE),
    metric!(M_LLM_REQUESTS_TOTAL, REQUEST),
    metric!(
        M_LLM_REQUEST_DURATION,
        REQUEST_DURATION,
        extra = &["is_fallback"]
    ),
    metric!(M_LLM_TTFT, USAGE),
    metric!(
        M_LLM_TOKENS_BY_CLIENT_TOTAL,
        &["client_type", "model", "token_type"]
    ),
    metric!(
        M_PROXY_IN_FLIGHT,
        &["endpoint", "inbound_protocol"],
        required = &["endpoint"]
    ),
    metric!(M_PROXY_REQUESTS_TOTAL, REQUEST),
    metric!(M_PROXY_FAILED_REQUESTS_TOTAL, REQUEST),
    metric!(
        M_PROXY_REQUEST_DURATION,
        REQUEST_DURATION,
        extra = &["is_fallback"]
    ),
    metric!(
        M_PROXY_CLIENT_CANCELLED_TOTAL,
        &["endpoint", "model", "provider_key_id", "provider_key_name"]
    ),
    metric!(
        M_PROXY_BODY_LIMIT_REJECTIONS_TOTAL,
        &["endpoint", "inbound_protocol", "outcome"]
    ),
    metric!(M_DEPLOYMENT_REQUESTS_TOTAL, DEPLOYMENT),
    metric!(M_DEPLOYMENT_SUCCESS_TOTAL, DEPLOYMENT),
    metric!(M_DEPLOYMENT_FAILURE_TOTAL, DEPLOYMENT),
    metric!(M_DEPLOYMENT_STATE, DEPLOYMENT, required = DEPLOYMENT),
    metric!(M_DEPLOYMENT_COOLED_DOWN_TOTAL, DEPLOYMENT),
    metric!(
        M_ROUTING_SUCCESSFUL_FALLBACKS_TOTAL,
        &["model", "fallback_model"]
    ),
    metric!(
        M_ROUTING_FAILED_FALLBACKS_TOTAL,
        &["model", "fallback_model"]
    ),
    metric!(
        M_RATELIMIT_REMAINING_REQUESTS,
        &["api_key_id", "model"],
        required = &["api_key_id", "model"]
    ),
    metric!(
        M_RATELIMIT_REMAINING_TOKENS,
        &["api_key_id", "model"],
        required = &["api_key_id", "model"]
    ),
    metric!(M_BUDGET_LIMIT_USD, BUDGET, required = BUDGET),
    metric!(M_BUDGET_SPENT_USD, BUDGET, required = BUDGET),
    metric!(M_BUDGET_REMAINING_USD, BUDGET, required = BUDGET),
    metric!(M_BUDGET_RESET_SECONDS, BUDGET, required = BUDGET),
    metric!(M_BUDGET_DETAILS_PRESENT, BUDGET, required = BUDGET),
    metric!(M_REDIS_FAILURES_TOTAL, &["operation"]),
    metric!(
        M_USAGE_EVENT_DROPS_TOTAL,
        &[
            "reason",
            "model",
            "provider_key_id",
            "provider_key_name",
            "user_id",
            "user_name",
            "upstream_protocol"
        ]
    ),
    metric!(M_GUARDRAIL_BLOCKS_TOTAL, &[]),
    metric!(M_GUARDRAIL_BYPASSES_TOTAL, &["reason"]),
    metric!(M_AUTH_DECISIONS_TOTAL, &["method", "result", "reason"]),
    metric!(
        M_GUARDRAIL_LATENCY_SECONDS,
        &[
            "env_id",
            "guardrail",
            "kind",
            "phase",
            "result",
            "error_type"
        ]
    ),
    metric!(
        M_USAGE_EVENT_EMITS_TOTAL,
        &[
            "handler",
            "status_code",
            "status",
            "inbound_protocol",
            "upstream_protocol",
            "model",
            "provider_key_id",
            "provider_key_name",
            "user_id",
            "user_name"
        ]
    ),
    metric!(M_CACHE_REQUESTS_TOTAL, &["policy", "outcome"]),
    metric!(M_CACHE_SEMANTIC_EMBED_SECONDS, &["policy"]),
    metric!(M_CACHE_SEMANTIC_EMBED_FAILURES_TOTAL, &["policy", "cause"]),
    metric!(M_CACHE_SEMANTIC_STORE_FAILURES_TOTAL, &["policy", "op"]),
    metric!(M_OTLP_FANOUT_DROPS_TOTAL, &["exporter", "reason"]),
    metric!(M_OTLP_FANOUT_FAILURES_TOTAL, &["exporter"]),
    metric!(M_REQUEST_E2E_LATENCY_SECONDS, LATENCY, extra = USAGE),
    metric!(M_REQUEST_TTFT_SECONDS, LATENCY, extra = USAGE),
    metric!(M_A2A_REQUESTS_TOTAL, &["agent", "operation", "status"]),
    metric!(M_A2A_TTFB_SECONDS, &["agent", "operation"]),
    metric!(M_A2A_STREAM_EVENTS_TOTAL, &["agent", "operation"]),
    metric!(M_A2A_TASK_STATE_TOTAL, &["agent", "state"]),
    metric!(M_CONFIG_LAST_RELOAD_SUCCESSFUL, &[]),
    metric!(M_CONFIG_LAST_RELOAD_SUCCESS_TIMESTAMP, &[]),
    metric!(M_CONFIG_RELOADS_TOTAL, &[]),
    metric!(M_CONFIG_RELOAD_FAILURES_TOTAL, &["reason"]),
    metric!(M_CONFIG_REJECTED_RESOURCES, &["kind"], required = &["kind"]),
    metric!(
        M_CONFIG_PARTIALLY_COMPATIBLE_RESOURCES,
        &["kind"],
        required = &["kind"]
    ),
    metric!(
        M_CONFIG_STALE_SERVED_RESOURCES,
        &["kind"],
        required = &["kind"]
    ),
    metric!(M_CONFIG_OBSERVED_REVISION, &[]),
    metric!(M_CONFIG_APPLIED_REVISION, &[]),
    metric!(M_CONFIG_HASH_INFO, &["hash"], required = &["hash"]),
    metric!(M_CONFIG_SOURCE_CONNECTED, &[]),
];

#[derive(Debug)]
pub struct LabelSelection {
    labels: HashMap<String, Vec<String>>,
}

impl LabelSelection {
    pub fn compile(config: &BTreeMap<String, Vec<String>>) -> Result<Self, String> {
        for (name, labels) in config {
            let definition = METRIC_DEFINITIONS
                .iter()
                .find(|metric| metric.name == name)
                .ok_or_else(|| {
                    format!("observability.metrics.labels: unknown metric family {name:?}")
                })?;
            let mut seen = HashSet::new();
            for label in labels {
                if !definition.supports(label) {
                    return Err(format!(
                        "observability.metrics.labels.{name}: unsupported variable {label:?}"
                    ));
                }
                if !seen.insert(label) {
                    return Err(format!(
                        "observability.metrics.labels.{name}: duplicate variable {label:?}"
                    ));
                }
            }
            for required in definition.required_labels {
                if !labels.iter().any(|label| label == required) {
                    return Err(format!("observability.metrics.labels.{name}: must retain identity label {required:?}"));
                }
            }
        }
        let mut labels: HashMap<String, Vec<String>> = METRIC_DEFINITIONS
            .iter()
            .map(|metric| {
                (
                    metric.name.to_owned(),
                    metric
                        .default_labels
                        .iter()
                        .map(|label| (*label).to_owned())
                        .collect(),
                )
            })
            .collect();
        labels.extend(config.clone());
        Ok(Self { labels })
    }

    fn project(&self, key: &Key, env_id: &str) -> Key {
        let Some(selected) = self.labels.get(key.name()) else {
            return key.clone();
        };
        if key.labels().len() == selected.len()
            && key
                .labels()
                .all(|label| selected.iter().any(|name| name == label.key()))
        {
            return key.clone();
        }
        let labels = selected
            .iter()
            .map(|name| {
                let value = if name == "env_id" {
                    env_id
                } else {
                    key.labels()
                        .find(|label| label.key() == name)
                        .map_or("unknown", Label::value)
                };
                Label::new(name.to_owned(), value.to_owned())
            })
            .collect::<Vec<_>>();
        Key::from_parts(key.name().to_owned(), labels)
    }
}

pub(crate) struct LabelRecorder {
    inner: PrometheusRecorder,
    selection: LabelSelection,
    env_id: String,
    counters: Mutex<HashMap<Key, Counter>>,
}

impl LabelRecorder {
    pub(crate) fn selected_labels(&self, metric: &str) -> &[String] {
        self.selection.labels.get(metric).map_or(&[], Vec::as_slice)
    }

    pub fn new(inner: PrometheusRecorder, selection: LabelSelection, env_id: &str) -> Self {
        Self {
            inner,
            selection,
            env_id: if env_id.is_empty() { "unknown" } else { env_id }.to_owned(),
            counters: Mutex::new(HashMap::new()),
        }
    }
}

struct ProjectedCounter {
    target: Counter,
    last: AtomicU64,
}

impl CounterFn for ProjectedCounter {
    fn increment(&self, value: u64) {
        self.last.fetch_add(value, Ordering::Relaxed);
        self.target.increment(value);
    }

    fn absolute(&self, value: u64) {
        // Each original counter owns its own absolute value. Taking the
        // maximum on the merged handle would lose the other sources.
        let previous = self.last.fetch_max(value, Ordering::Relaxed);
        if value > previous {
            self.target.increment(value - previous);
        }
    }
}

impl Recorder for LabelRecorder {
    fn describe_counter(&self, key: KeyName, unit: Option<Unit>, description: SharedString) {
        self.inner.describe_counter(key, unit, description);
    }

    fn describe_gauge(&self, key: KeyName, unit: Option<Unit>, description: SharedString) {
        self.inner.describe_gauge(key, unit, description);
    }

    fn describe_histogram(&self, key: KeyName, unit: Option<Unit>, description: SharedString) {
        self.inner.describe_histogram(key, unit, description);
    }

    fn register_counter(&self, key: &Key, metadata: &Metadata<'_>) -> Counter {
        let projected = self.selection.project(key, &self.env_id);
        // Only ConfigStatus counters supply absolute source values. Keep
        // those sources separate; increment-only traffic counters can share
        // the projected handle without retaining the removed label values.
        if !matches!(
            key.name(),
            M_CONFIG_RELOADS_TOTAL | M_CONFIG_RELOAD_FAILURES_TOTAL
        ) || key
            .labels()
            .all(|label| projected.labels().any(|selected| selected == label))
        {
            return self.inner.register_counter(&projected, metadata);
        }
        self.counters
            .lock()
            .expect("projected counters")
            .entry(key.clone())
            .or_insert_with(|| {
                Counter::from_arc(Arc::new(ProjectedCounter {
                    target: self.inner.register_counter(&projected, metadata),
                    last: AtomicU64::new(0),
                }))
            })
            .clone()
    }

    fn register_gauge(&self, key: &Key, metadata: &Metadata<'_>) -> Gauge {
        self.inner
            .register_gauge(&self.selection.project(key, &self.env_id), metadata)
    }

    fn register_histogram(&self, key: &Key, metadata: &Metadata<'_>) -> Histogram {
        self.inner
            .register_histogram(&self.selection.project(key, &self.env_id), metadata)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn selection(name: &str, labels: &[&str]) -> BTreeMap<String, Vec<String>> {
        BTreeMap::from([(
            name.to_owned(),
            labels.iter().map(|label| (*label).to_owned()).collect(),
        )])
    }

    fn sample<'a>(out: &'a str, name: &str) -> Vec<&'a str> {
        out.lines()
            .filter(|line| {
                line.starts_with(name)
                    && matches!(line.as_bytes().get(name.len()), Some(b'{' | b' '))
            })
            .collect()
    }

    #[test]
    fn selected_counter_labels_merge_observations_and_keep_unconfigured_defaults() {
        let metrics = Metrics::new_with_labels(
            "test-env",
            &HistogramBuckets::default(),
            &selection(M_PROXY_REQUESTS_TOTAL, &["provider", "env_id"]),
        )
        .unwrap();
        for name in ["key-a", "key-b", "key-a"] {
            metrics.record_proxy_request(
                RequestLabels {
                    provider: "openai",
                    provider_key_name: name,
                    ..Default::default()
                },
                Duration::from_millis(100),
            );
        }
        let out = metrics.render();
        let selected = sample(&out, M_PROXY_REQUESTS_TOTAL);
        assert_eq!(selected.len(), 1, "{out}");
        assert!(selected[0].contains("provider=\"openai\""));
        assert!(selected[0].contains("env_id=\"test-env\""));
        assert!(!selected[0].contains("provider_key_name"));
        assert!(selected[0].ends_with(" 3"));
        let defaults = sample(&out, "aisix_proxy_request_duration_seconds_count");
        assert_eq!(
            defaults.len(),
            2,
            "unconfigured family preserves the key split: {out}"
        );
    }

    #[test]
    fn selected_histogram_labels_partition_buckets_sum_and_count() {
        let metrics = Metrics::new_with_labels(
            "env",
            &HistogramBuckets::default(),
            &selection(M_REQUEST_TTFT_SECONDS, &["provider_key_name"]),
        )
        .unwrap();
        for (name, ms) in [("primary", 100), ("backup", 200), ("primary", 300)] {
            metrics.record_request_ttft(
                LatencyLabels {
                    streaming: true,
                    details: UsageLabels {
                        provider_key_name: name,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                Duration::from_millis(ms),
            );
        }
        let out = metrics.render();
        for suffix in ["_bucket", "_sum", "_count"] {
            let name = format!("{M_REQUEST_TTFT_SECONDS}{suffix}");
            let rows = sample(&out, &name);
            assert!(!rows.is_empty());
            assert!(rows.iter().all(|line| line.contains("provider_key_name=")));
            assert!(rows.iter().all(|line| !line.contains("endpoint=")));
        }
        let counts = sample(&out, "aisix_request_ttft_seconds_count");
        assert_eq!(counts.len(), 2);
        assert!(counts
            .iter()
            .any(|line| line.contains("\"primary\"") && line.ends_with(" 2")));
        assert!(counts
            .iter()
            .any(|line| line.contains("\"backup\"") && line.ends_with(" 1")));
        let sums = sample(&out, "aisix_request_ttft_seconds_sum");
        assert!(
            sums.iter()
                .any(|line| line.contains("\"primary\"") && line.ends_with(" 0.4")),
            "{out}"
        );
        assert!(
            out.lines()
                .any(|line| line.contains("provider_key_name=\"primary\"")
                    && line.contains("le=\"0.1\"")
                    && line.ends_with(" 1")),
            "{out}"
        );
    }

    #[test]
    fn removing_all_labels_merges_absolute_counters_without_losing_sources() {
        let inner = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = inner.handle();
        let recorder = LabelRecorder::new(
            inner,
            LabelSelection::compile(&selection(M_CONFIG_RELOAD_FAILURES_TOTAL, &[])).unwrap(),
            "env",
        );
        metrics::with_local_recorder(&recorder, || {
            metrics::counter!(M_CONFIG_RELOAD_FAILURES_TOTAL, "reason" => "parse").absolute(3);
            metrics::counter!(M_CONFIG_RELOAD_FAILURES_TOTAL, "reason" => "io").absolute(5);
            metrics::counter!(M_CONFIG_RELOAD_FAILURES_TOTAL, "reason" => "parse").absolute(4);
            metrics::counter!(M_CONFIG_RELOAD_FAILURES_TOTAL, "reason" => "io").absolute(5);
        });
        let out = handle.render();
        assert_eq!(
            sample(&out, M_CONFIG_RELOAD_FAILURES_TOTAL),
            vec!["aisix_config_reload_failures_total 9"]
        );
    }

    #[test]
    fn rejects_unknown_families_variables_and_duplicate_selections() {
        for (name, labels) in [
            ("typo", vec![]),
            (
                "aisix_request_ttft_seconds_bucket",
                vec!["provider_key_name"],
            ),
            (M_REQUEST_TTFT_SECONDS, vec!["request_id"]),
            (M_REQUEST_TTFT_SECONDS, vec!["le"]),
            (
                M_REQUEST_TTFT_SECONDS,
                vec!["provider_key_name", "provider_key_name"],
            ),
            (M_PROXY_IN_FLIGHT, vec!["provider_key_name"]),
            (M_PROXY_IN_FLIGHT, vec![]),
            (
                M_DEPLOYMENT_STATE,
                vec!["provider", "model", "upstream_model"],
            ),
        ] {
            assert!(
                LabelSelection::compile(&selection(name, &labels)).is_err(),
                "{name}: {labels:?}"
            );
        }
    }

    #[test]
    fn catalog_covers_each_metric_constant_and_documents_every_variable() {
        let source = include_str!("metrics.rs");
        let names = regex::Regex::new(r#"pub const M_\w+: &str =\s*\"([^\"]+)\""#).unwrap();
        let actual: HashSet<_> = names
            .captures_iter(source)
            .map(|cap| cap[1].to_owned())
            .collect();
        let registered: HashSet<_> = METRIC_DEFINITIONS
            .iter()
            .map(|metric| metric.name.to_owned())
            .collect();
        assert_eq!(actual, registered);
        assert_eq!(registered.len(), METRIC_DEFINITIONS.len());
        let variables: HashSet<_> = METRIC_VARIABLES
            .iter()
            .map(|variable| variable.name)
            .collect();
        assert_eq!(variables.len(), METRIC_VARIABLES.len());
        for metric in METRIC_DEFINITIONS {
            for label in metric.default_labels.iter().chain(metric.extra_labels) {
                assert!(
                    variables.contains(label),
                    "{} uses undocumented {label}",
                    metric.name
                );
            }
        }
    }
}
