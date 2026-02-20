use std::sync::LazyLock;

use opentelemetry::metrics::{Counter, Histogram, Meter};

static METER: LazyLock<Meter> = LazyLock::new(|| opentelemetry::global::meter("ai-gateway"));

pub static METRIC_REQUEST_COUNT: LazyLock<Counter<u64>> = LazyLock::new(|| {
    METER
        .u64_counter("aisix_request_count")
        .with_description("Total number of requests processed")
        .build()
});

pub static METRIC_TOKEN_COUNT: LazyLock<Counter<u64>> = LazyLock::new(|| {
    METER
        .u64_counter("aisix_token_count")
        .with_description("Total number of tokens processed")
        .build()
});

pub static METRIC_REQUEST_LATENCY: LazyLock<Histogram<u64>> = LazyLock::new(|| {
    METER
        .u64_histogram("aisix_request_latency")
        .with_description("Request latency")
        .with_unit("ms")
        .build()
});

pub static METRIC_LLM_LATENCY: LazyLock<Histogram<u64>> = LazyLock::new(|| {
    METER
        .u64_histogram("aisix_llm_latency")
        .with_description("LLM provider latency")
        .with_unit("ms")
        .build()
});

pub static METRIC_LLM_FIRST_TOKEN_LATENCY: LazyLock<Histogram<u64>> = LazyLock::new(|| {
    METER
        .u64_histogram("aisix_llm_first_token_latency")
        .with_description("LLM provider first token latency (only for streaming requests)")
        .with_unit("ms")
        .build()
});
