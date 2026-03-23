use std::{process::exit, sync::Arc};

use ai_gateway::{config::Config, *};
use axum::Router;
use clap::Parser;
use log::{error, info};
use tokio::{select, sync::oneshot};

#[derive(Parser, Debug)]
#[command(version)]
struct Args {
    /// Path to the configuration file
    #[arg(short, long)]
    config: Option<String>,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let (ob_shutdown_signal, ob_shutdown_task) = init_observability();

    let config = Arc::new(config::load(args.config).expect("Failed to load configuration"));

    let config_provider = config::create_provider(&config)
        .await
        .expect("Failed to create config provider");
    let resources =
        Arc::new(config::entities::ResourceRegistry::new(config_provider.clone()).await);

    providers::init_client();

    let proxy_router =
        proxy::create_router(proxy::AppState::new(config.clone(), resources.clone()));

    let mut exception = false;
    select! {
        _ = tokio::signal::ctrl_c() => {
            info!("Stopping, see you next time!");
        }
        res = serve_proxy(config.clone(), proxy_router.clone()) => {
            if let Err(e) = res {
                error!("Proxy server error: {}", e);
                exception = true;
            }
        }
        res = serve_admin(config.clone(), admin::AppState::new(config, config_provider, resources, Some(proxy_router))) => {
            if let Err(e) = res {
                error!("Admin server error: {}", e);
                exception = true;
            }
        }
    }

    let _ = ob_shutdown_signal.send(());
    ob_shutdown_task
        .await
        .expect("Failed to shutdown observability");
    exit(if exception { 1 } else { 0 });
}

fn init_observability() -> (oneshot::Sender<()>, tokio::task::JoinHandle<()>) {
    use std::{borrow::Cow, time::Duration};

    use fastrace::collector::Config;
    use fastrace_opentelemetry::OpenTelemetryReporter;
    use logforth::{
        append::{FastraceEvent, Stdout},
        filter::env_filter::EnvFilterBuilder,
        layout::TextLayout,
    };
    use metrics_exporter_otel::OpenTelemetryRecorder;
    use opentelemetry::{InstrumentationScope, KeyValue, metrics::MeterProvider};
    use opentelemetry_otlp::{SpanExporter, WithExportConfig};
    use opentelemetry_sdk::{
        Resource,
        metrics::{PeriodicReader, SdkMeterProvider, Temporality},
    };

    let (tx, rx) = oneshot::channel::<()>();

    // log
    logforth::starter_log::builder()
        .dispatch(|d| {
            d.filter(EnvFilterBuilder::from_default_env_or("info").build())
                .append(Stdout::default().with_layout(TextLayout::default()))
        })
        .dispatch(|d| {
            d.filter(EnvFilterBuilder::from_default_env_or("info").build())
                .append(FastraceEvent::default())
        })
        .apply();

    // trace
    if cfg!(feature = "trace") {
        let reporter = OpenTelemetryReporter::new(
            SpanExporter::builder()
                .with_tonic()
                //TODO make endpoint configurable
                .with_endpoint("http://127.0.0.1:4317".to_string())
                .with_protocol(opentelemetry_otlp::Protocol::Grpc)
                .with_timeout(opentelemetry_otlp::OTEL_EXPORTER_OTLP_TIMEOUT_DEFAULT)
                .build()
                .expect("initialize otlp exporter"),
            Cow::Owned(
                Resource::builder()
                    .with_attributes([KeyValue::new("service.name", "ai-gateway")])
                    .build(),
            ),
            InstrumentationScope::builder("ai-gateway")
                .with_version(env!("CARGO_PKG_VERSION"))
                .build(),
        );
        fastrace::set_reporter(
            reporter,
            Config::default().report_interval(Duration::from_secs(1)),
        );
    }

    // metric
    let exporter = opentelemetry_otlp::MetricExporter::builder()
        .with_http()
        .with_temporality(Temporality::default())
        .with_endpoint("http://127.0.0.1:9090/api/v1/otlp/v1/metrics")
        .build()
        .unwrap();

    let reader = PeriodicReader::builder(exporter)
        .with_interval(std::time::Duration::from_secs(10))
        .build();

    let meter_provider = SdkMeterProvider::builder().with_reader(reader).build();
    let meter = meter_provider.meter("ai-gateway");

    metrics::set_global_recorder(OpenTelemetryRecorder::new(meter))
        .expect("initialize metrics recorder");
    ai_gateway::utils::metrics::describe_metrics();

    // shuting down signal handler
    let shutdown_handle = tokio::spawn(async move {
        let _ = rx.await;

        if cfg!(feature = "trace") {
            fastrace::flush();
        }

        if let Err(e) = meter_provider.shutdown() {
            error!("Error shutting down meter provider: {}", e);
        }

        logforth::core::default_logger().flush();
        logforth::core::default_logger().exit();
    });

    (tx, shutdown_handle)
}

async fn serve_proxy(config: Arc<Config>, router: Router) -> Result<(), std::io::Error> {
    serve(
        "Proxy",
        config.server.proxy.listen,
        &config.server.proxy.tls,
        router,
    )
    .await
}

async fn serve_admin(config: Arc<Config>, state: admin::AppState) -> Result<(), std::io::Error> {
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
