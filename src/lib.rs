mod admin;
mod config;
mod gateway;
mod proxy;
mod utils;

use std::{process::exit, sync::Arc};

use anyhow::{Context, Result};
use axum::Router;
use clap::Parser;
use log::{error, info};
use tokio::{select, sync::oneshot};

pub const GIT_HASH: &str = env!("VERGEN_GIT_SHA");

fn long_version() -> &'static str {
    concat!(env!("CARGO_PKG_VERSION"), " (", env!("VERGEN_GIT_SHA"), ")",)
}

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version = env!("CARGO_PKG_VERSION"), long_version = long_version())]
pub struct Args {
    /// Path to the configuration file
    #[arg(short, long)]
    config: Option<String>,
}

pub async fn run() -> Result<()> {
    let args = Args::parse();

    let (ob_shutdown_signal, ob_shutdown_task) =
        init_observability().context("failed to initialize observability")?;

    let config = Arc::new(config::load(args.config).context("failed to load configuration")?);

    let config_provider = config::create_provider(&config)
        .await
        .context("failed to create config provider")?;
    let resources =
        Arc::new(config::entities::ResourceRegistry::new(config_provider.clone()).await);

    let gateway = Arc::new(gateway::Gateway::new(
        gateway::providers::default_provider_registry()
            .context("failed to build default gateway provider registry")?,
    ));

    let proxy_router = proxy::create_router(proxy::AppState::new(
        config.clone(),
        resources.clone(),
        gateway,
    ));

    let mut exception = false;
    select! {
        res = tokio::signal::ctrl_c() => {
            if let Err(e) = res {
                error!("Failed to listen for shutdown signal: {}", e);
                exception = true;
            }
        }
        res = serve_proxy(config.clone(), proxy_router.clone()) => {
            if let Err(e) = res {
                error!("Proxy server error: {}", e);
                exception = true;
            }
        }
        res = serve_admin(config.clone(), admin::AppState::new(config, config_provider.clone(), resources, Some(proxy_router))) => {
            if let Err(e) = res {
                error!("Admin server error: {}", e);
                exception = true;
            }
        }
    }

    if let Err(e) = config_provider.shutdown().await {
        error!("Config provider shutdown error: {}", e);
        exception = true;
    }

    info!("Stopping, see you next time!");
    let _ = ob_shutdown_signal.send(());
    ob_shutdown_task
        .await
        .context("failed to shutdown observability")?;

    exit(if exception { 1 } else { 0 });
}

fn init_observability() -> Result<(oneshot::Sender<()>, tokio::task::JoinHandle<()>)> {
    use std::{borrow::Cow, time::Duration};

    use fastrace::collector::Config;
    use fastrace_opentelemetry::OpenTelemetryReporter;
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

    const INSTRUMENTATION_NAME: &str = "aisix";

    let (tx, rx) = oneshot::channel::<()>();

    // log
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

    // trace
    let reporter = OpenTelemetryReporter::new(
        SpanExporter::builder()
            .build()
            .context("failed to initialize otlp exporter")?,
        Cow::Owned(Resource::builder().build()),
        InstrumentationScope::builder(INSTRUMENTATION_NAME)
            .with_version(env!("CARGO_PKG_VERSION"))
            .build(),
    );
    fastrace::set_reporter(
        reporter,
        Config::default().report_interval(Duration::from_secs(1)),
    );

    // metric
    let exporter = opentelemetry_otlp::MetricExporter::builder().build()?;

    let reader = PeriodicReader::builder(exporter).build();

    let meter_provider = SdkMeterProvider::builder().with_reader(reader).build();
    let meter = meter_provider.meter(INSTRUMENTATION_NAME);

    metrics::set_global_recorder(OpenTelemetryRecorder::new(meter))
        .context("failed to initialize metrics recorder")?;
    utils::metrics::describe_metrics();

    // shutting down signal handler
    let shutdown_handle = tokio::spawn(async move {
        let _ = rx.await;

        fastrace::flush();

        if let Err(e) = meter_provider.shutdown() {
            error!("Error shutting down meter provider: {}", e);
        }

        logforth::core::default_logger().flush();
        logforth::core::default_logger().exit();
    });

    Ok((tx, shutdown_handle))
}

async fn serve_proxy(config: Arc<config::Config>, router: Router) -> Result<(), std::io::Error> {
    serve(
        "Proxy",
        config.server.proxy.listen,
        &config.server.proxy.tls,
        router,
    )
    .await
}

async fn serve_admin(
    config: Arc<config::Config>,
    state: admin::AppState,
) -> Result<(), std::io::Error> {
    serve(
        "Admin",
        config.server.admin.listen,
        &config.server.admin.tls,
        admin::create_router(state),
    )
    .await
}

async fn serve(
    name: &str,
    addr: std::net::SocketAddr,
    tls: &config::ServerCommonTls,
    router: Router,
) -> Result<(), std::io::Error> {
    if tls.enabled {
        let Some(cert) = tls.cert_file.as_deref() else {
            error!("{} TLS cert_file is required when TLS is enabled", name);
            exit(1);
        };

        if !std::path::Path::new(cert).exists() {
            error!("{} TLS cert_file '{}' does not exist", name, cert);
            exit(1);
        }

        let Some(key) = tls.key_file.as_deref() else {
            error!("{} TLS key_file is required when TLS is enabled", name);
            exit(1);
        };

        if !std::path::Path::new(key).exists() {
            error!("{} TLS key_file '{}' does not exist", name, key);
            exit(1);
        }

        info!("{} API listening on https://{}", name, addr);
        let tls_config = axum_server::tls_openssl::OpenSSLConfig::from_pem_file(cert, key)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        axum_server::bind_openssl(addr, tls_config)
            .serve(router.into_make_service_with_connect_info::<std::net::SocketAddr>())
            .await
    } else {
        let listener = tokio::net::TcpListener::bind(addr).await?;
        info!("{} API listening on http://{}", name, addr);
        axum::serve(
            listener,
            router.into_make_service_with_connect_info::<std::net::SocketAddr>(),
        )
        .await
    }
}
