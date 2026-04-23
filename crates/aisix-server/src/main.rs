//! aisix — single-binary AI gateway entrypoint.
//!
//! Startup sequence (spec §1):
//!  1. Parse CLI args (`--config <path>`)
//!  2. Load + validate config (YAML/TOML/JSON, `AISIX__*` env overrides)
//!  3. Initialise tracing
//!  4. Connect to etcd with 5s × 5 retry
//!  5. Bootstrap initial snapshot
//!  6. Spawn watch supervisor
//!  7. Build proxy router
//!  8. Build admin router
//!  9. Bind + serve both ports (tokio::select! with shutdown signal)
//! 10. On SIGINT/SIGTERM: cancel supervisor, stop accepting, join

use std::path::PathBuf;
use std::sync::Arc;

use aisix_admin::{AdminState, ConfigStore, EtcdConfigStore};
use aisix_cache::{Cache, MemoryCache, RedisCache};
use aisix_core::models::Provider;
use aisix_core::{CacheBackend, Config, EtcdConfig};
use aisix_etcd::{EtcdConfigProvider, Supervisor};
use aisix_gateway::Hub;
use aisix_obs::{init_tracing, install_otlp_tracer, langfuse, Metrics};
use aisix_provider_anthropic::AnthropicBridge;
use aisix_provider_deepseek::deepseek_bridge;
use aisix_provider_gemini::gemini_bridge;
use aisix_provider_openai::OpenAiBridge;
use aisix_proxy::ProxyState;
use aisix_ratelimit::Limiter;
use clap::Parser;
use etcd_client::{Certificate, ConnectOptions, Identity, TlsOptions};
use tokio::sync::watch;

#[derive(Debug, Parser)]
#[command(name = "aisix", version, about = "aisix AI Gateway")]
struct Cli {
    /// Path to the bootstrap config file (YAML / TOML / JSON).
    #[arg(short, long, env = "AISIX_CONFIG")]
    config: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Steps 1-2: config.
    let cfg = Config::load_from_path(Some(&cli.config))
        .map_err(|e| anyhow::anyhow!("config load failed: {e}"))?;

    // Step 3: tracing + optional OTLP export.
    init_tracing(&cfg.observability).map_err(|e| anyhow::anyhow!("tracing init failed: {e}"))?;
    let _otlp = install_otlp_tracer(&cfg.observability)
        .map_err(|e| anyhow::anyhow!("otlp init failed: {e}"))?;

    run(cfg).await
}

/// Factored out of `main` so the integration tests can drive the full
/// startup with a real config struct and still use `#[tokio::test]`.
async fn run(cfg: Config) -> anyhow::Result<()> {
    // Steps 4-6: etcd + supervisor.
    let connect_options = build_etcd_connect_options(&cfg.etcd)?;
    let provider = Arc::new(
        EtcdConfigProvider::connect(
            &cfg.etcd.endpoints,
            cfg.etcd.prefix.clone(),
            connect_options.clone(),
        )
        .await
        .map_err(|e| anyhow::anyhow!("etcd connect failed: {e}"))?,
    );
    // Separate client for the admin write path — only needed when the
    // admin surface is bound. We could share a single underlying
    // connection via `Client::clone()` but keeping two is cleaner:
    // writes and the watch stream don't contend on the same mutex.
    // In managed mode this client is simply skipped.
    let admin_client = if cfg.managed.is_managed() {
        None
    } else {
        Some(
            etcd_client::Client::connect(&cfg.etcd.endpoints, connect_options.clone())
                .await
                .map_err(|e| anyhow::anyhow!("etcd admin client connect failed: {e}"))?,
        )
    };
    let supervisor = Arc::new(Supervisor::new(provider, cfg.etcd.prefix.clone()));
    let snapshot_handle = supervisor.handle();

    let (cancel_tx, cancel_rx) = watch::channel(false);
    let watch_task = tokio::spawn(supervisor.clone().run(cancel_rx.clone()));

    // Steps 7-8: build Hub, shared components, then routers.
    let hub = Arc::new(build_hub());
    let limiter = Arc::new(Limiter::new());
    let metrics = Arc::new(Metrics::new(true));
    // Cache backend selection. Memory by default; Redis when configured.
    // Qdrant / semantic backends drop in here in a follow-up PR.
    let cache: Option<Arc<dyn Cache>> = match cfg.cache.backend {
        CacheBackend::Memory => Some(Arc::new(MemoryCache::with_defaults())),
        CacheBackend::Redis => {
            let url = cfg
                .cache
                .redis
                .as_ref()
                .map(|r| r.url.clone())
                .ok_or_else(|| anyhow::anyhow!("cache.backend = redis but cache.redis missing"))?;
            tracing::info!(target: "aisix::cache", backend = "redis", "connecting cache backend");
            let redis = RedisCache::connect(&url)
                .await
                .map_err(|e| anyhow::anyhow!("redis cache connect failed (url={url}): {e}"))?;
            Some(Arc::new(redis) as Arc<dyn Cache>)
        }
        CacheBackend::Qdrant => {
            tracing::warn!(target: "aisix::cache", "qdrant cache not yet implemented — falling back to in-memory");
            Some(Arc::new(MemoryCache::with_defaults()))
        }
    };

    // Optional Langfuse exporter — disabled in config by default.
    // When enabled, the proxy gets an Arc<LangfuseSender> through
    // ProxyState and emits one event per chat completion at
    // end-of-request. We keep the handle alive for the lifetime of
    // the process so the background flush task continues running.
    let langfuse_handle = match langfuse::spawn(&cfg.observability) {
        Ok(h) => h,
        Err(e) => {
            tracing::warn!(error = %e, "langfuse exporter disabled");
            None
        }
    };
    let langfuse_sender = langfuse_handle.as_ref().map(|h| h.sender());

    let mut proxy_state = ProxyState::with_components(
        snapshot_handle.clone(),
        hub.clone(),
        limiter.clone(),
        metrics.clone(),
        cache.clone(),
        &cfg.proxy,
    );
    if let Some(sender) = langfuse_sender {
        proxy_state = proxy_state.with_langfuse(sender);
    }
    // Clone shared trackers before consuming proxy_state in build_router.
    let budget_tracker = proxy_state.budgets.clone();
    let health_tracker = proxy_state.health.clone();
    let proxy_router = aisix_proxy::build_router(proxy_state);

    // Admin router + listener are only built in standalone mode.
    // In managed mode (`cfg.managed.enabled = true`) the DP reads
    // configuration exclusively from etcd; exposing admin writes or
    // the Playground would bypass the aisix.cloud control plane.
    let admin_serve_handle = if let Some(admin_client) = admin_client {
        let admin_store: Arc<dyn ConfigStore> =
            Arc::new(EtcdConfigStore::new(admin_client, cfg.etcd.prefix.clone()));
        let admin_state = AdminState::new(snapshot_handle.clone(), admin_store, &cfg.admin)
            .with_metrics(metrics.clone())
            // Share the in-process budget tracker so /admin/v1/spend reports
            // live current-month spend without a database round-trip.
            .with_budget_tracker(budget_tracker)
            // Share the health tracker so /admin/v1/health reflects live
            // per-model upstream failure counts.
            .with_health_tracker(health_tracker)
            // Share the proxy router so the playground endpoint can forward
            // requests in-process without an extra network hop.
            .with_proxy_router(proxy_router.clone());
        let admin_router = aisix_admin::build_router(admin_state);

        let admin_addr: std::net::SocketAddr = cfg.admin.addr.parse()?;
        let admin_listener = tokio::net::TcpListener::bind(admin_addr).await?;
        tracing::info!(admin = %admin_addr, "aisix admin listening");
        let admin_serve = axum::serve(admin_listener, admin_router)
            .with_graceful_shutdown(shutdown_signal(cancel_rx.clone(), "admin"));
        Some(tokio::spawn(async move { admin_serve.await }))
    } else {
        // Drop unused shared components so the compiler can see they
        // don't escape managed mode. The budget/health trackers exist
        // on proxy_state and keep working regardless.
        let _ = (&budget_tracker, &health_tracker);
        tracing::info!("managed mode enabled — admin surface not bound");
        None
    };

    // Step 9: bind + serve the proxy (always). Admin is handled above.
    let proxy_addr: std::net::SocketAddr = cfg.proxy.addr.parse()?;
    let proxy_listener = tokio::net::TcpListener::bind(proxy_addr).await?;
    tracing::info!(proxy = %proxy_addr, "aisix proxy listening");

    let proxy_serve = axum::serve(proxy_listener, proxy_router)
        .with_graceful_shutdown(shutdown_signal(cancel_rx.clone(), "proxy"));

    // Step 10: shutdown coordinator. Whichever of (signal, proxy, admin)
    // completes first triggers the rest.
    let signal_task = tokio::spawn(wait_for_signal(cancel_tx.clone()));

    let proxy_res = proxy_serve.await;
    proxy_res.map_err(|e| anyhow::anyhow!("proxy serve error: {e}"))?;
    if let Some(handle) = admin_serve_handle {
        match handle.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(anyhow::anyhow!("admin serve error: {e}")),
            Err(e) => return Err(anyhow::anyhow!("admin task join error: {e}")),
        }
    }

    // Ask the supervisor to stop (no-op if the signal task already did).
    let _ = cancel_tx.send(true);
    let _ = signal_task.await;
    let _ = watch_task.await;
    tracing::info!("aisix shut down cleanly");
    Ok(())
}

/// Build the etcd-client `ConnectOptions` from `cfg.etcd`, wiring in
/// the mTLS bundle when `cfg.etcd.tls` is present.
///
/// Returns `Ok(None)` for plain HTTP etcd (no TLS, no user/password) so
/// callers can pass the value straight into `Client::connect`.
///
/// Design notes:
///
/// - We deliberately read the cert / key files inside this helper
///   rather than in a `load_from_path` prologue. It keeps the config
///   struct a pure POD — serialisable round-trippable — and the I/O
///   failure bubbles up as a nicely-contextualised BootstrapError at
///   the same point as other etcd connection errors.
/// - `domain_name` defaults to the hostname portion of the first
///   endpoint. Callers only need to override when the CA issues certs
///   under a different name than the DNS they're dialing (rare but
///   possible when the endpoint is an IP or internal alias).
fn build_etcd_connect_options(etcd: &EtcdConfig) -> anyhow::Result<Option<ConnectOptions>> {
    let mut needs_options = false;
    let mut options = ConnectOptions::new();

    if let (Some(user), Some(env_key)) = (etcd.user.as_ref(), etcd.password_env.as_ref()) {
        let pw = std::env::var(env_key).map_err(|_| {
            anyhow::anyhow!("etcd.password_env = {env_key:?} is set but the env var is missing")
        })?;
        options = options.with_user(user.clone(), pw);
        needs_options = true;
    }

    if let Some(tls) = etcd.tls.as_ref() {
        let ca_pem = std::fs::read(&tls.ca_cert_file)
            .map_err(|e| anyhow::anyhow!("etcd.tls.ca_cert_file = {:?}: {e}", tls.ca_cert_file))?;
        let cert_pem = std::fs::read(&tls.client_cert_file).map_err(|e| {
            anyhow::anyhow!(
                "etcd.tls.client_cert_file = {:?}: {e}",
                tls.client_cert_file
            )
        })?;
        let key_pem = std::fs::read(&tls.client_key_file).map_err(|e| {
            anyhow::anyhow!("etcd.tls.client_key_file = {:?}: {e}", tls.client_key_file)
        })?;

        let domain = match tls.domain_name.clone() {
            Some(d) => d,
            None => default_domain_from_endpoint(&etcd.endpoints[0])?,
        };

        let tls_opts = TlsOptions::new()
            .domain_name(domain)
            .ca_certificate(Certificate::from_pem(ca_pem))
            .identity(Identity::from_pem(cert_pem, key_pem));
        options = options.with_tls(tls_opts);
        needs_options = true;
    }

    Ok(needs_options.then_some(options))
}

/// Extract the host portion of a URL-like endpoint (`http://host:2379`,
/// `https://host:2379`, or bare `host:2379`) for use as the TLS SNI.
fn default_domain_from_endpoint(endpoint: &str) -> anyhow::Result<String> {
    let without_scheme = endpoint
        .split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or(endpoint);
    let host = without_scheme
        .rsplit_once(':')
        .map(|(h, _)| h)
        .unwrap_or(without_scheme)
        .trim_matches(|c| c == '[' || c == ']'); // strip IPv6 brackets
    if host.is_empty() {
        anyhow::bail!("cannot derive TLS domain_name from endpoint {endpoint:?}");
    }
    Ok(host.to_string())
}

/// Register all four provider bridges on a fresh Hub. The Hub is
/// created once at startup; future dynamic reload lands behind the
/// same `register()` call.
fn build_hub() -> Hub {
    let hub = Hub::new();
    hub.register(Provider::Openai, Arc::new(OpenAiBridge::new()));
    hub.register(Provider::Anthropic, Arc::new(AnthropicBridge::new()));
    hub.register(Provider::Gemini, Arc::new(gemini_bridge()));
    hub.register(Provider::Deepseek, Arc::new(deepseek_bridge()));
    hub
}

/// Completes when the process receives SIGINT or SIGTERM (best-effort on
/// Windows — Ctrl+C only) OR when another part of the system has already
/// flipped the cancel channel.
async fn shutdown_signal(mut cancel: watch::Receiver<bool>, label: &'static str) {
    loop {
        if *cancel.borrow() {
            tracing::info!(label, "shutdown signal observed — stopping listener");
            return;
        }
        if cancel.changed().await.is_err() {
            return;
        }
    }
}

async fn wait_for_signal(cancel_tx: watch::Sender<bool>) {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let term = async {
        use tokio::signal::unix::{signal, SignalKind};
        if let Ok(mut s) = signal(SignalKind::terminate()) {
            s.recv().await;
        } else {
            std::future::pending::<()>().await;
        }
    };

    #[cfg(not(unix))]
    let term = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => tracing::info!("received SIGINT"),
        _ = term => tracing::info!("received SIGTERM"),
    }

    let _ = cancel_tx.send(true);
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn cli_requires_config_path() {
        // Missing --config must error (either from env var or arg).
        let result = Cli::try_parse_from(["aisix"]);
        assert!(result.is_err());
    }

    #[test]
    fn cli_accepts_short_and_long_flags() {
        let a = Cli::try_parse_from(["aisix", "-c", "/tmp/x.yaml"]).unwrap();
        let b = Cli::try_parse_from(["aisix", "--config", "/tmp/x.yaml"]).unwrap();
        assert_eq!(a.config, b.config);
        assert_eq!(a.config, PathBuf::from("/tmp/x.yaml"));
    }

    #[test]
    fn default_domain_strips_scheme_port_and_brackets() {
        // Plain hostnames.
        assert_eq!(
            default_domain_from_endpoint("http://etcd.aisix.cloud:2379").unwrap(),
            "etcd.aisix.cloud"
        );
        assert_eq!(
            default_domain_from_endpoint("https://etcd.aisix.cloud:2379").unwrap(),
            "etcd.aisix.cloud"
        );
        assert_eq!(
            default_domain_from_endpoint("etcd.aisix.cloud:2379").unwrap(),
            "etcd.aisix.cloud"
        );
        assert_eq!(
            default_domain_from_endpoint("etcd.aisix.cloud").unwrap(),
            "etcd.aisix.cloud"
        );
        // IPv6 addresses show up with brackets; the SNI value should be
        // the bare numeric literal (TLS libraries reject brackets).
        assert_eq!(
            default_domain_from_endpoint("https://[::1]:2379").unwrap(),
            "::1"
        );
    }

    #[test]
    fn build_connect_options_none_when_plain_http() {
        let etcd = aisix_core::EtcdConfig {
            endpoints: vec!["http://127.0.0.1:2379".into()],
            prefix: "/aisix".into(),
            user: None,
            password_env: None,
            dial_timeout_ms: 5000,
            request_timeout_ms: 5000,
            tls: None,
        };
        let opts = build_etcd_connect_options(&etcd).unwrap();
        assert!(
            opts.is_none(),
            "plain HTTP etcd must not synthesise options"
        );
    }

    #[test]
    fn build_connect_options_surfaces_missing_cert_files() {
        let etcd = aisix_core::EtcdConfig {
            endpoints: vec!["https://etcd.aisix.cloud:2379".into()],
            prefix: "/aisix".into(),
            user: None,
            password_env: None,
            dial_timeout_ms: 5000,
            request_timeout_ms: 5000,
            tls: Some(aisix_core::EtcdTlsConfig {
                ca_cert_file: "/definitely/does/not/exist/ca.crt".into(),
                client_cert_file: "/tmp/c.crt".into(),
                client_key_file: "/tmp/c.key".into(),
                domain_name: None,
            }),
        };
        let err = build_etcd_connect_options(&etcd).unwrap_err();
        // The error must mention which file was missing — operators
        // should not have to diff config against filesystem state.
        assert!(
            err.to_string().contains("ca_cert_file"),
            "unexpected error: {err}"
        );
    }
}
