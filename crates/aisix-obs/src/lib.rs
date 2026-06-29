//! aisix-obs — observability primitives shared by the proxy and admin
//! crates.
//!
//! Scope for PR #14:
//! - [`init_tracing`] installs the process-wide tracing subscriber.
//! - [`access_log::AccessLog`] — one-line structured request log, called
//!   by the proxy handler at end-of-request.
//! - [`metrics::Metrics`] — Prometheus counters + histogram for
//!   requests/duration/rate-limits/tokens.
//! - [`otlp::install_otlp_tracer`] — optional OTLP export handshake
//!   (concrete pipeline wired in a follow-up PR).
//! - [`sink`] — pluggable observability-sink framework: the
//!   capability-typed [`sink::ObservabilitySink`] adapter contract
//!   (AISIX-Cloud#692).

#![deny(rust_2018_idioms)]

pub mod access_log;
pub mod metrics;
pub mod otlp;
pub mod otlp_http_sink;
pub mod sink;
pub mod usage;

use aisix_core::ObservabilityConfig;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

pub use access_log::AccessLog;
pub use metrics::{
    client_type_from_user_agent, BudgetGauges, BudgetLabels, DeploymentLabels, DeploymentState,
    LlmUsage, Metrics, RequestLabels, RequestOutcome, UsageLabels,
};
pub use otlp::{install_otlp_tracer, shutdown_otlp, OtlpError, OtlpHandle};
pub use otlp_http_sink::{content_capture_cap, OtlpHttpFanOut, OtlpSink};
pub use sink::{
    AliyunSlsSink, BatchUnit, CapturedContent, ChannelKey, DatadogSink, EventBatch,
    ExporterPipelines, IdempotencyMarker, IdempotencyScheme, ObservabilitySink, OrderingScope,
    PipelineConfig, SinkAck, SinkCapabilities, SinkContent, SinkError, SinkHandle, SinkHealth,
    SinkPipeline, SinkRecord, SinkResult, SinkStatsSnapshot, SCHEMA_VERSION,
};
pub use usage::{UsageEvent, UsageSink};

#[derive(Debug, thiserror::Error)]
pub enum ObsError {
    #[error("invalid log filter directive {directive:?}: {source}")]
    Filter {
        directive: String,
        #[source]
        source: tracing_subscriber::filter::ParseError,
    },
    #[error("tracing subscriber already initialised")]
    AlreadyInitialised,
}

/// Install a process-wide tracing subscriber.
pub fn init_tracing(cfg: &ObservabilityConfig) -> Result<(), ObsError> {
    let filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(&cfg.log_level))
        .map_err(|source| ObsError::Filter {
            directive: cfg.log_level.clone(),
            source,
        })?;

    let fmt_layer = fmt::layer().with_target(true).with_writer(std::io::stderr);

    tracing_subscriber::registry()
        .with(filter)
        .with(fmt_layer)
        .try_init()
        .map_err(|_| ObsError::AlreadyInitialised)?;

    tracing::info!(
        service = %cfg.service_name,
        level = %cfg.log_level,
        "tracing initialised",
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn already_initialised_variant_is_displayable() {
        let err = ObsError::AlreadyInitialised;
        assert_eq!(err.to_string(), "tracing subscriber already initialised");
    }

    #[test]
    fn filter_error_carries_the_bad_directive() {
        let bad = "BAD=@notalevel";
        let err = tracing_subscriber::EnvFilter::try_new(bad).unwrap_err();
        let wrapped = ObsError::Filter {
            directive: bad.into(),
            source: err,
        };
        assert!(wrapped.to_string().contains(bad));
    }
}
