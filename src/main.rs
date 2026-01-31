use log::info;

use crate::config::entities::ResourceRegistry;

mod config;
mod providers;
mod proxy;

#[tokio::main]
async fn main() {
    init_observability();

    let config = config::load().expect("Failed to load configuration");
    info!("Loaded config: {:?}", config);

    let config_provider = config::create_provider(config.clone())
        .await
        .expect("Failed to create config provider");
    let resources = ResourceRegistry::init(config_provider).await;

    // Initialize global rate limiter
    proxy::policies::rate_limit::init_rate_limiter();

    serve(proxy::AppState::new(config.clone(), resources.clone())).await;

    if cfg!(feature = "trace") {
        fastrace::flush();
    }
}

fn init_observability() {
    use fastrace::collector::Config;
    use fastrace_opentelemetry::OpenTelemetryReporter;
    use logforth::{
        append::{FastraceEvent, Stdout},
        filter::env_filter::EnvFilterBuilder,
        layout::TextLayout,
    };
    use opentelemetry::{InstrumentationScope, KeyValue};
    use opentelemetry_otlp::{SpanExporter, WithExportConfig};
    use opentelemetry_sdk::Resource;
    use std::{borrow::Cow, time::Duration};

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
}

async fn serve(state: proxy::AppState) {
    use std::net::SocketAddr;

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();

    info!("Server listening on http://0.0.0.0:3000");

    let _ = tokio::join!(axum::serve(
        listener,
        proxy::create_router(state).into_make_service_with_connect_info::<SocketAddr>(),
    ),);
}
