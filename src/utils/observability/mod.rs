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
use opentelemetry_otlp::{MetricExporter, SpanExporter};
use opentelemetry_sdk::{
    Resource,
    metrics::{SdkMeterProvider, periodic_reader_with_async_runtime::PeriodicReader},
};
use tokio::{runtime::Handle, sync::oneshot, task::JoinHandle};

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
    span_exporter: Option<SpanExporter>,
    config: Option<FastraceConfig>,
) -> Result<(oneshot::Sender<()>, JoinHandle<()>)> {
    let handle = Handle::current();
    let reporter = OpenTelemetryReporter::new(
        match span_exporter {
            Some(exporter) => exporter,
            None => SpanExporter::builder()
                .build()
                .context("failed to initialize otlp exporter")?,
        },
        Cow::Owned(
            Resource::builder()
                .with_service_name(INSTRUMENTATION_NAME)
                .build(),
        ),
        InstrumentationScope::builder(INSTRUMENTATION_NAME)
            .with_version(env!("CARGO_PKG_VERSION"))
            .build(),
    )
    .with_block_on(move |fut| handle.block_on(fut));
    fastrace::set_reporter(reporter, config.unwrap_or_default());

    Ok(shutdown_handler(|| async move { fastrace::flush() }))
}

/// Initialize observability metrics.
pub fn init_observability_metric(
    metric_exporter: Option<MetricExporter>,
) -> Result<(oneshot::Sender<()>, JoinHandle<()>)> {
    let exporter = match metric_exporter {
        Some(exporter) => exporter,
        None => MetricExporter::builder()
            .build()
            .context("failed to initialize metric exporter")?,
    };

    let meter_provider = SdkMeterProvider::builder()
        .with_reader(
            PeriodicReader::builder(exporter, opentelemetry_sdk::runtime::Tokio)
                .with_interval(std::time::Duration::from_secs(15))
                .build(),
        )
        .build();
    let meter = meter_provider.meter(INSTRUMENTATION_NAME);

    metrics::set_global_recorder(OpenTelemetryRecorder::new(meter))
        .context("failed to initialize metric recorder")?;
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
    let (metric_tx, metric_shutdown_handle) = init_observability_metric(None)?;

    Ok(shutdown_handler(|| async move {
        let _ = trace_tx.send(());
        let _ = trace_shutdown_handle.await;
        let _ = metric_tx.send(());
        let _ = metric_shutdown_handle.await;
        let _ = log_tx.send(());
        let _ = log_shutdown_handle.await;
    }))
}
