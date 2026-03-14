use std::sync::Arc;

use ai_gateway::*;
use axum::Router;
use log::info;
use tokio::select;

#[tokio::main]
async fn main() {
    init_observability();

    let config = config::load().expect("Failed to load configuration");
    info!("Loaded config: {:?}", config);

    let config_provider = config::create_provider(config.clone())
        .await
        .expect("Failed to create config provider");
    let resources =
        Arc::new(config::entities::ResourceRegistry::new(config_provider.clone()).await);

    providers::init_client();

    let proxy_state = proxy::AppState::new(config.clone(), resources.clone());
    let proxy_router = proxy::create_router(proxy_state);

    select! {
        _ = tokio::signal::ctrl_c() => {
            info!("Received Ctrl+C, shutting down");
        }
        res = serve_proxy(proxy_router.clone()) => {
            if let Err(e) = res {
                eprintln!("Proxy server error: {}", e);
            }
        }
        res = serve_admin(admin::AppState::new(config.clone(), config_provider, resources, Some(proxy_router))) => {
            if let Err(e) = res {
                eprintln!("Admin server error: {}", e);
            }
        }
    }

    if cfg!(feature = "trace") {
        fastrace::flush();
    }
}

fn init_observability() {
    use std::{borrow::Cow, time::Duration};

    use fastrace::collector::Config;
    use fastrace_opentelemetry::OpenTelemetryReporter;
    use logforth::{
        append::{FastraceEvent, Stdout},
        filter::env_filter::EnvFilterBuilder,
        layout::TextLayout,
    };
    use opentelemetry::{InstrumentationScope, KeyValue};
    use opentelemetry_otlp::{SpanExporter, WithExportConfig};
    use opentelemetry_sdk::{
        Resource,
        metrics::{PeriodicReader, SdkMeterProvider, Temporality},
    };

    // log
    logforth::starter_log::builder()
        .dispatch(|d| {
            d.filter(EnvFilterBuilder::from_default_env().build())
                .append(Stdout::default().with_layout(TextLayout::default()))
        })
        .dispatch(|d| {
            d.filter(EnvFilterBuilder::from_default_env().build())
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

    let provider = SdkMeterProvider::builder().with_reader(reader).build();

    // TODO: provider.shutdown()
    opentelemetry::global::set_meter_provider(provider.clone());

    let _ = opentelemetry::global::meter("ai-gateway");
}

async fn serve_proxy(router: Router) -> Result<(), std::io::Error> {
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;

    info!("Server listening on http://0.0.0.0:3000");

    axum::serve(
        listener,
        router.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .await
}

async fn serve_admin(state: admin::AppState) -> Result<(), std::io::Error> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:3001").await?;

    info!("Admin API listening on http://127.0.0.1:3001");

    axum::serve(
        listener,
        admin::create_router(state).into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .await
}
