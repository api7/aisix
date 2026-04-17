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
use aisix_core::models::Provider;
use aisix_core::Config;
use aisix_etcd::{EtcdConfigProvider, Supervisor};
use aisix_gateway::Hub;
use aisix_obs::init_tracing;
use aisix_provider_anthropic::AnthropicBridge;
use aisix_provider_deepseek::deepseek_bridge;
use aisix_provider_gemini::gemini_bridge;
use aisix_provider_openai::OpenAiBridge;
use aisix_proxy::ProxyState;
use clap::Parser;
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

    // Step 3: tracing.
    init_tracing(&cfg.observability).map_err(|e| anyhow::anyhow!("tracing init failed: {e}"))?;

    run(cfg).await
}

/// Factored out of `main` so the integration tests can drive the full
/// startup with a real config struct and still use `#[tokio::test]`.
async fn run(cfg: Config) -> anyhow::Result<()> {
    // Steps 4-6: etcd + supervisor.
    let provider = Arc::new(
        EtcdConfigProvider::connect(&cfg.etcd.endpoints, cfg.etcd.prefix.clone(), None)
            .await
            .map_err(|e| anyhow::anyhow!("etcd connect failed: {e}"))?,
    );
    // Separate client for the admin write path. We could share a single
    // underlying connection via `Client::clone()` but keeping two is
    // cleaner — writes and the watch stream don't contend on the same
    // mutex.
    let admin_client = etcd_client::Client::connect(&cfg.etcd.endpoints, None)
        .await
        .map_err(|e| anyhow::anyhow!("etcd admin client connect failed: {e}"))?;
    let supervisor = Arc::new(Supervisor::new(provider, cfg.etcd.prefix.clone()));
    let snapshot_handle = supervisor.handle();

    let (cancel_tx, cancel_rx) = watch::channel(false);
    let watch_task = tokio::spawn(supervisor.clone().run(cancel_rx.clone()));

    // Steps 7-8: build Hub, then routers.
    let hub = Arc::new(build_hub());
    let proxy_router = aisix_proxy::build_router(ProxyState::new(
        snapshot_handle.clone(),
        hub.clone(),
        &cfg.proxy,
    ));
    // Admin CRUD writes through etcd. The watch supervisor's read path
    // is on a separate client (see above) so a long range scan during a
    // list doesn't stall the watch stream.
    let admin_store: Arc<dyn ConfigStore> =
        Arc::new(EtcdConfigStore::new(admin_client, cfg.etcd.prefix.clone()));
    let admin_router = aisix_admin::build_router(AdminState::new(
        snapshot_handle.clone(),
        admin_store,
        &cfg.admin,
    ));

    // Step 9: bind + serve.
    let proxy_addr: std::net::SocketAddr = cfg.proxy.addr.parse()?;
    let admin_addr: std::net::SocketAddr = cfg.admin.addr.parse()?;
    let proxy_listener = tokio::net::TcpListener::bind(proxy_addr).await?;
    let admin_listener = tokio::net::TcpListener::bind(admin_addr).await?;
    tracing::info!(proxy = %proxy_addr, admin = %admin_addr, "aisix listening");

    let proxy_serve = axum::serve(proxy_listener, proxy_router)
        .with_graceful_shutdown(shutdown_signal(cancel_rx.clone(), "proxy"));
    let admin_serve = axum::serve(admin_listener, admin_router)
        .with_graceful_shutdown(shutdown_signal(cancel_rx.clone(), "admin"));

    // Step 10: shutdown coordinator. Whichever of (signal, proxy, admin)
    // completes first triggers the rest.
    let signal_task = tokio::spawn(wait_for_signal(cancel_tx.clone()));

    let (proxy_res, admin_res) = tokio::join!(proxy_serve, admin_serve);
    proxy_res.map_err(|e| anyhow::anyhow!("proxy serve error: {e}"))?;
    admin_res.map_err(|e| anyhow::anyhow!("admin serve error: {e}"))?;

    // Ask the supervisor to stop (no-op if the signal task already did).
    let _ = cancel_tx.send(true);
    let _ = signal_task.await;
    let _ = watch_task.await;
    tracing::info!("aisix shut down cleanly");
    Ok(())
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
}
