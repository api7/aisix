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

use std::error::Error as StdError;
use std::path::PathBuf;
use std::sync::Arc;

mod heartbeat;
mod register;

use aisix_admin::{AdminState, ConfigStore, EtcdConfigStore};
use aisix_cache::{Cache, MemoryCache, RedisCache};
use aisix_core::models::Provider;
use aisix_core::{CacheBackend, Config, EtcdConfig, EtcdTlsConfig};
use aisix_etcd::{EtcdConfigProvider, SnapshotCache, Supervisor};
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
    // Install the process-level rustls CryptoProvider before anything
    // else touches TLS. rustls 0.23 dropped implicit provider selection
    // and panics at first use when both `aws-lc-rs` and `ring` features
    // are reachable (or neither is) — which is the case here through
    // transitive deps on reqwest + etcd-client + tokio-rustls.
    //
    // We pick aws-lc-rs because it's the upstream default as of
    // rustls 0.23, FIPS-capable, and what every compiled-in crate
    // already depends on transitively. Falls back to ring only if
    // the process somehow has a provider installed already (idempotent).
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();

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
async fn run(mut cfg: Config) -> anyhow::Result<()> {
    // Managed-mode bootstrap. If we have to register (first boot),
    // we also capture the heartbeat config the CP sent back so the
    // worker can be spawned a few lines below. If the bundle is
    // already on disk (subsequent boot), we synthesise the same
    // values from config + dp_id_file.
    let heartbeat_cfg: Option<heartbeat::HeartbeatConfig> = if cfg.managed.is_managed() {
        if !register::bundle_exists(&cfg.managed.mtls_dir) && cfg.managed.registration_enabled() {
            tracing::info!("managed mode: registering with aisix.cloud CP");
            let r = register::register_and_persist(&cfg.managed)
                .await
                .map_err(|e| anyhow::anyhow!("DP registration failed: {e:#}"))?;
            tracing::info!(
                dp_id = %r.dp_id,
                gateway_id = %r.gateway_id,
                etcd = %r.etcd_endpoint,
                "registered with control plane",
            );
            // Override the static config with what the CP handed back.
            // Endpoints get the https:// scheme re-attached (the CP
            // sends a bare host:port, but tonic-based etcd-client
            // expects a full URL for TLS endpoints).
            cfg.etcd.endpoints = vec![format!("https://{}", r.etcd_endpoint)];
            cfg.etcd.tls = Some(EtcdTlsConfig {
                ca_cert_file: r.ca_cert_path.to_string_lossy().into_owned(),
                client_cert_file: r.client_cert_path.to_string_lossy().into_owned(),
                client_key_file: r.client_key_path.to_string_lossy().into_owned(),
                domain_name: None, // derive from endpoint host
            });
            Some(heartbeat::HeartbeatConfig::sanitised(
                r.heartbeat_url,
                r.dp_id,
                r.heartbeat_interval,
            ))
        } else if register::bundle_exists(&cfg.managed.mtls_dir) {
            // Bundle persisted from a previous boot; load the dp_id
            // and synthesise heartbeat config from the configured
            // cp_base_url. Registration doesn't re-run.
            match load_heartbeat_config_from_disk(&cfg.managed) {
                Ok(h) => Some(h),
                Err(e) => {
                    tracing::warn!(error = %e,
                        "managed mode: heartbeat worker disabled (dp_id unreadable)");
                    None
                }
            }
        } else {
            None
        }
    } else {
        None
    };

    // Steps 4-6: etcd + supervisor.
    //
    // Before handing endpoints to tonic, probe each one via the
    // stdlib resolver. tonic's HTTP connector collapses any DNS
    // failure into an opaque "dns error" Status (see
    // hyper-util/src/client/legacy/connect/http.rs) — even after the
    // cause-chain logging in aisix-etcd, the deepest cause we see is
    // still whatever getaddrinfo returned. The probe either logs the
    // resolved addresses (DNS works; the failure is higher in the
    // tonic / TLS stack) or logs the raw io::Error (DNS actually
    // fails). Both outcomes narrow triage substantially.
    probe_etcd_dns(&cfg.etcd.endpoints).await;

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
    // Snapshot cache: in managed mode persist to disk (default
    // /var/lib/aisix/config_cache.json) so the DP can serve traffic
    // from the last-known config across CP outages and restarts.
    // Disabled outside managed mode and when the operator clears the
    // path explicitly.
    let snapshot_cache = if cfg.managed.is_managed() && !cfg.managed.snapshot_cache_path.is_empty()
    {
        SnapshotCache::new(&cfg.managed.snapshot_cache_path)
    } else {
        SnapshotCache::disabled()
    };
    let supervisor = Arc::new(Supervisor::with_cache(
        provider,
        cfg.etcd.prefix.clone(),
        snapshot_cache,
    ));
    // Seed the snapshot from disk before the etcd cycle starts so the
    // proxy is ready to serve from cached config the moment the watch
    // task takes its first iteration.
    supervisor.restore_from_cache();
    let snapshot_handle = supervisor.handle();

    let (cancel_tx, cancel_rx) = watch::channel(false);
    let watch_task = tokio::spawn(supervisor.clone().run(cancel_rx.clone()));
    // Spawn heartbeat worker if we have a config for it. The
    // JoinHandle is awaited after graceful shutdown below so the
    // final in-flight beat drains cleanly.
    let heartbeat_task = heartbeat_cfg.map(|h| heartbeat::spawn(h, cancel_rx.clone()));

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
    if let Some(task) = heartbeat_task {
        let _ = task.await;
    }
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
/// Per-endpoint DNS probe logged at info / warn. Not part of the
/// connect path — purely diagnostic. See the call site in [`run`]
/// for why this exists.
async fn probe_etcd_dns(endpoints: &[String]) {
    for raw in endpoints {
        let (host, port) = match parse_host_port(raw) {
            Ok(hp) => hp,
            Err(err) => {
                tracing::warn!(
                    endpoint = %raw,
                    error = %err,
                    "etcd endpoint parse failed; skipping DNS probe",
                );
                continue;
            }
        };
        match tokio::net::lookup_host((host.clone(), port)).await {
            Ok(iter) => {
                let addrs: Vec<String> = iter.map(|a| a.to_string()).collect();
                tracing::info!(
                    endpoint = %raw,
                    host = %host,
                    port,
                    addrs = ?addrs,
                    "etcd endpoint DNS probe resolved",
                );
            }
            Err(err) => {
                // Walk the io::Error chain so the OS-level detail
                // ("Name or service not known", "Temporary failure
                // in name resolution", …) makes it into the log.
                let mut chain = err.to_string();
                let mut cur: Option<&(dyn StdError + 'static)> = StdError::source(&err);
                while let Some(src) = cur {
                    chain.push_str(": ");
                    chain.push_str(&src.to_string());
                    cur = src.source();
                }
                tracing::warn!(
                    endpoint = %raw,
                    host = %host,
                    port,
                    error = %chain,
                    kind = ?err.kind(),
                    "etcd endpoint DNS probe failed",
                );
            }
        }
    }
}

/// Shared endpoint → (host, port) splitter. Mirrors the logic in
/// [`default_domain_from_endpoint`] plus a port parse.
fn parse_host_port(endpoint: &str) -> anyhow::Result<(String, u16)> {
    let without_scheme = endpoint
        .split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or(endpoint);
    let (host, port) = match without_scheme.rsplit_once(':') {
        Some((h, p)) => (
            h.trim_matches(|c| c == '[' || c == ']'),
            p.parse::<u16>()
                .map_err(|e| anyhow::anyhow!("invalid port {p:?} in {endpoint:?}: {e}"))?,
        ),
        // No explicit port — default to the etcd v3 port.
        None => (without_scheme.trim_matches(|c| c == '[' || c == ']'), 2379),
    };
    if host.is_empty() {
        anyhow::bail!("endpoint {endpoint:?} has no host");
    }
    Ok((host.to_string(), port))
}

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

/// Synthesise a HeartbeatConfig when the mTLS bundle is already on
/// disk from a previous boot. Reads `managed.dp_id_file` and
/// combines with `managed.cp_base_url` — the register response is
/// not available on this code path.
///
/// Returns an error (not None) when the user has configured managed
/// mode AND the bundle exists BUT the dp_id is unreadable — that's
/// an inconsistent on-disk state an operator should know about.
fn load_heartbeat_config_from_disk(
    managed: &aisix_core::ManagedConfig,
) -> anyhow::Result<heartbeat::HeartbeatConfig> {
    let base = managed
        .cp_base_url
        .as_deref()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            anyhow::anyhow!("managed.cp_base_url must be set for heartbeat on subsequent boots")
        })?;
    let dp_id = std::fs::read_to_string(&managed.dp_id_file)
        .map_err(|e| anyhow::anyhow!("read dp_id from {}: {e}", managed.dp_id_file))?
        .trim()
        .to_string();
    if dp_id.is_empty() {
        anyhow::bail!("dp_id file {} is empty", managed.dp_id_file);
    }
    let url = format!("{}/dp/heartbeat", base.trim_end_matches('/'));
    Ok(heartbeat::HeartbeatConfig::sanitised(
        url,
        dp_id,
        std::time::Duration::from_secs(15),
    ))
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

    #[test]
    fn parse_host_port_strips_scheme_and_keeps_port() {
        let (h, p) = parse_host_port("https://dp-manager:7943").unwrap();
        assert_eq!(h, "dp-manager");
        assert_eq!(p, 7943);
    }

    #[test]
    fn parse_host_port_defaults_to_2379_when_port_is_omitted() {
        let (h, p) = parse_host_port("http://etcd.aisix.cloud").unwrap();
        assert_eq!(h, "etcd.aisix.cloud");
        assert_eq!(p, 2379);
    }

    #[test]
    fn parse_host_port_accepts_bare_host_port() {
        let (h, p) = parse_host_port("etcd.aisix.cloud:2379").unwrap();
        assert_eq!(h, "etcd.aisix.cloud");
        assert_eq!(p, 2379);
    }

    #[test]
    fn parse_host_port_rejects_empty_host() {
        // Host portion before the port colon is empty — real-world
        // shape: a stripped prefix that left just ":<port>".
        let err = parse_host_port(":7943").unwrap_err();
        assert!(err.to_string().contains("no host"), "unexpected: {err}");
    }

    #[test]
    fn parse_host_port_rejects_non_numeric_port() {
        let err = parse_host_port("host:abc").unwrap_err();
        assert!(
            err.to_string().contains("invalid port"),
            "unexpected: {err}"
        );
    }
}
