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

use std::io::IsTerminal as _;

use aisix_core::ObservabilityConfig;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

pub use access_log::AccessLog;
pub use metrics::{
    client_type_from_user_agent, A2aCallOutcome, A2aLabels, BudgetGauges, BudgetLabels,
    CancelledLabels, ClientTypeClassifier, DeploymentLabels, DeploymentState, HistogramBuckets,
    LatencyLabels, LlmUsage, Metrics, RequestLabels, RequestOutcome, UsageEventLabels, UsageLabels,
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

/// Third-party log targets pinned below WARN, applied last so they hold
/// whatever the operator asked for otherwise.
///
/// `lofty` is the audio-metadata reader behind the transcription duration
/// probe (`aisix-proxy::audio`). Every real mp3 upload logs one or more
/// parse observations at WARN — `Chunk exceeds reader size, stopping`,
/// `MPEG: Using bitrate to estimate duration` — that describe the
/// uploaded container, not a gateway fault, and that no operator can act
/// on: the probe already treats an unreadable file as a zero cost basis
/// rather than an error. Left at WARN they turn every transcription into
/// a warning line and fail the e2e log scan (#998).
const QUIET_TARGETS: &[&str] = &["lofty=error"];

/// The subscriber's level filter: `RUST_LOG` when set, else the
/// configured level, with [`QUIET_TARGETS`] applied on top.
fn build_filter(cfg: &ObservabilityConfig) -> Result<EnvFilter, ObsError> {
    let mut filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(&cfg.log_level))
        .map_err(|source| ObsError::Filter {
            directive: cfg.log_level.clone(),
            source,
        })?;
    for directive in QUIET_TARGETS {
        filter = filter.add_directive(
            directive
                .parse()
                .expect("QUIET_TARGETS holds valid filter directives"),
        );
    }
    Ok(filter)
}

/// Install a process-wide tracing subscriber.
pub fn init_tracing(cfg: &ObservabilityConfig) -> Result<(), ObsError> {
    let filter = build_filter(cfg)?;

    // Colorize only for a human at a terminal. When stderr is a pipe or a
    // file — every real deployment, where logs go to a container runtime and
    // on to a log store — the escapes land BETWEEN a field's name and its
    // value, so `grep 'aliyun_request_id=<id>'` matches nothing and the
    // structured fields are only searchable by bare value
    // (AISIX-Cloud#1060). tracing-subscriber's `ansi` default feature is on
    // and it does not probe the writer itself.
    let fmt_layer = fmt::layer()
        .with_target(true)
        .with_ansi(std::io::stderr().is_terminal())
        .with_writer(std::io::stderr);

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
    use tracing_subscriber::fmt::MakeWriter;

    /// Collect emitted log bytes into an in-memory buffer.
    #[derive(Clone, Default)]
    struct VecWriter {
        buf: std::sync::Arc<std::sync::Mutex<Vec<u8>>>,
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

    /// #998: the audio-metadata reader's parse chatter must not reach the
    /// log as a WARN, while the gateway's own WARNs still do.
    #[test]
    fn quiet_targets_drop_lofty_warnings_only() {
        let writer = VecWriter::default();
        let cfg = ObservabilityConfig {
            log_level: "info".into(),
            ..Default::default()
        };
        let subscriber = tracing_subscriber::registry()
            .with(build_filter(&cfg).expect("filter builds"))
            .with(fmt::layer().with_ansi(false).with_writer(writer.clone()));
        tracing::subscriber::with_default(subscriber, || {
            tracing::warn!(target: "lofty::mpeg::properties", "MPEG: Using bitrate to estimate duration");
            tracing::error!(target: "lofty::iff::chunk", "unreadable");
            tracing::warn!(target: "aisix_proxy::audio", "transcription failed");
        });
        let logged = String::from_utf8_lossy(&writer.buf.lock().unwrap()).into_owned();
        assert!(
            !logged.contains("Using bitrate"),
            "lofty WARN chatter must be filtered out, got: {logged}"
        );
        assert!(
            logged.contains("unreadable"),
            "a lofty ERROR must still surface, got: {logged}"
        );
        assert!(
            logged.contains("transcription failed"),
            "the gateway's own WARNs must be unaffected, got: {logged}"
        );
    }

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
