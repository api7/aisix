use metrics::{Unit, describe_counter, describe_histogram};

#[derive(Clone, Copy)]
pub struct MetricDef {
    pub key: &'static str,
    pub description: &'static str,
    pub unit: Unit,
}

pub const REQUEST_COUNT_KEY: &str = "aisix_request_count";
pub const TOKEN_COUNT_KEY: &str = "aisix_token_count";
pub const REQUEST_LATENCY_KEY: &str = "aisix_request_latency";
pub const LLM_LATENCY_KEY: &str = "aisix_llm_latency";
pub const LLM_FIRST_TOKEN_LATENCY_KEY: &str = "aisix_llm_first_token_latency";

pub const COUNTER_METRICS: [MetricDef; 2] = [
    MetricDef {
        key: REQUEST_COUNT_KEY,
        description: "Total number of requests processed",
        unit: Unit::Count,
    },
    MetricDef {
        key: TOKEN_COUNT_KEY,
        description: "Total number of tokens processed",
        unit: Unit::Count,
    },
];

pub const HISTOGRAM_METRICS: [MetricDef; 3] = [
    MetricDef {
        key: REQUEST_LATENCY_KEY,
        description: "Request latency",
        unit: Unit::Milliseconds,
    },
    MetricDef {
        key: LLM_LATENCY_KEY,
        description: "LLM provider latency",
        unit: Unit::Milliseconds,
    },
    MetricDef {
        key: LLM_FIRST_TOKEN_LATENCY_KEY,
        description: "LLM provider first token latency (only for streaming requests)",
        unit: Unit::Milliseconds,
    },
];

pub fn describe_metrics() {
    for metric in COUNTER_METRICS {
        describe_counter!(metric.key, metric.unit, metric.description);
    }

    for metric in HISTOGRAM_METRICS {
        describe_histogram!(metric.key, metric.unit, metric.description);
    }
}
