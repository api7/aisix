mod trace;

use std::borrow::Cow;

use anyhow::{Context, Result};
use fastrace::collector::Config as FastraceConfig;
use fastrace_opentelemetry::OpenTelemetryReporter;
use log::error;
use logforth::{
    append::{FastraceEvent, Stdout},
    filter::env_filter::EnvFilterBuilder,
    layout::TextLayout,
};
use metrics_exporter_otel::OpenTelemetryRecorder;
use opentelemetry::{InstrumentationScope, metrics::MeterProvider};
use opentelemetry_otlp::SpanExporter;
use opentelemetry_sdk::{
    Resource,
    metrics::{PeriodicReader, SdkMeterProvider},
};
use tokio::{sync::oneshot, task::JoinHandle};
pub use trace::{BoxedSpanExporter, DynSpanExporter};

use crate::utils;

pub const INSTRUMENTATION_NAME: &str = "aisix";

pub fn shutdown_handler<F, Fut>(f: F) -> (oneshot::Sender<()>, tokio::task::JoinHandle<()>)
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: Future<Output = ()> + Send + 'static,
{
    let (tx, rx) = oneshot::channel::<()>();
    let shutdown_handle = tokio::spawn(async move {
        let _ = rx.await;
        f().await;
    });
    (tx, shutdown_handle)
}

/// Initialize observability logging.
pub fn init_observability_log() -> Result<(oneshot::Sender<()>, JoinHandle<()>)> {
    logforth::starter_log::builder()
        .dispatch(|d| {
            d.filter(EnvFilterBuilder::from_default_env_or("info,opentelemetry_sdk=off").build())
                .append(Stdout::default().with_layout(TextLayout::default()))
        })
        .dispatch(|d| {
            d.filter(EnvFilterBuilder::from_default_env_or("info").build())
                .append(FastraceEvent::default())
        })
        .apply();

    Ok(shutdown_handler(|| async move {
        logforth::core::default_logger().flush();
        logforth::core::default_logger().exit();
    }))
}

/// Initialize observability tracing.
pub fn init_observability_trace(
    span_exporter: Option<BoxedSpanExporter>,
    config: Option<FastraceConfig>,
) -> Result<(oneshot::Sender<()>, JoinHandle<()>)> {
    let reporter = OpenTelemetryReporter::new(
        match span_exporter {
            Some(exporter) => exporter,
            None => BoxedSpanExporter::new(
                SpanExporter::builder()
                    .build()
                    .context("failed to initialize otlp exporter")?,
            ),
        },
        Cow::Owned(Resource::builder().build()),
        InstrumentationScope::builder(INSTRUMENTATION_NAME)
            .with_version(env!("CARGO_PKG_VERSION"))
            .build(),
    );
    fastrace::set_reporter(reporter, config.unwrap_or_default());

    Ok(shutdown_handler(|| async move { fastrace::flush() }))
}

/// Initialize observability metrics.
pub fn init_observability_metric() -> Result<(oneshot::Sender<()>, JoinHandle<()>)> {
    let exporter = opentelemetry_otlp::MetricExporter::builder().build()?;

    let reader = PeriodicReader::builder(exporter).build();

    let meter_provider = SdkMeterProvider::builder().with_reader(reader).build();
    let meter = meter_provider.meter(INSTRUMENTATION_NAME);

    metrics::set_global_recorder(OpenTelemetryRecorder::new(meter))
        .context("failed to initialize metrics recorder")?;
    utils::metrics::describe_metrics();

    // shutting down signal handler
    Ok(shutdown_handler(|| async move {
        if let Err(e) = meter_provider.shutdown() {
            error!("Error shutting down meter provider: {}", e);
        }
    }))
}

/// Initialize observability (logging, tracing, metrics).
///
/// Returns `(shutdown_sender, shutdown_task_handle)`.
/// Call `shutdown_sender.send(())` to flush and shut down observability.
pub fn init_observability() -> Result<(oneshot::Sender<()>, tokio::task::JoinHandle<()>)> {
    let (log_tx, log_shutdown_handle) = init_observability_log()?;
    let (trace_tx, trace_shutdown_handle) = init_observability_trace(None, None)?;
    let (metric_tx, metric_shutdown_handle) = init_observability_metric()?;

    Ok(shutdown_handler(|| async move {
        let _ = trace_tx.send(());
        let _ = trace_shutdown_handle.await;
        let _ = metric_tx.send(());
        let _ = metric_shutdown_handle.await;
        let _ = log_tx.send(());
        let _ = log_shutdown_handle.await;
    }))
}
