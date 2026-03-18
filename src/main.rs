use std::{process::exit, sync::Arc};

use ai_gateway::{config::Config, *};
use axum::Router;
use log::{error, info};
use tokio::select;
use validator::Validate;

#[tokio::main]
async fn main() {
    init_observability();

    let config = Arc::new(config::load().expect("Failed to load configuration"));
    if let Err(e) = config.validate() {
        error!("Configuration validation error: {}", e);
        exit(1);
    }

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

    if cfg!(feature = "trace") {
        fastrace::flush();
    }
    exit(if exception { 1 } else { 0 });
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

    let provider = SdkMeterProvider::builder().with_reader(reader).build();

    // TODO: provider.shutdown()
    opentelemetry::global::set_meter_provider(provider.clone());

    let _ = opentelemetry::global::meter("ai-gateway");
}

async fn serve_proxy(config: Arc<Config>, router: Router) -> Result<(), std::io::Error> {
    let addr = config.listen;
    let listener = tokio::net::TcpListener::bind(addr).await?;

    info!("Proxy API listening on http://{}", addr);

    axum::serve(
        listener,
        router.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .await
}

async fn serve_admin(config: Arc<Config>, state: admin::AppState) -> Result<(), std::io::Error> {
    let addr = config
        .deployment
        .admin
        .as_ref()
        .map(|admin| admin.listen)
        .unwrap_or_else(crate::config::defaults::admin_listen);
    let listener = tokio::net::TcpListener::bind(addr).await?;

    info!("Admin API listening on http://{}", addr);

    axum::serve(
        listener,
        admin::create_router(state).into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .await
}
