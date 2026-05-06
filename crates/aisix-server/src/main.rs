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

mod cert_bundle;
mod heartbeat;
mod register;
mod telemetry;

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
use aisix_proxy::budget::BudgetClient;
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
    // Operator-supplied extra trust root, threaded into every
    // outbound mTLS client (register, etcd, heartbeat, telemetry,
    // BudgetClient). Needed for e2e / on-prem deployments where the
    // CP serves a cert distinct from the cert-manager-issued client-
    // cert CA. Production with public-CA certs leaves this `None`.
    let extra_ca_pem = register::read_optional_ca_pem(cfg.managed.cp_ca_cert_file.as_deref())?;

    // Managed-mode bootstrap. If we have to register (first boot),
    // we also capture the heartbeat config the CP sent back so the
    // worker can be spawned a few lines below. If the bundle is
    // already on disk (subsequent boot), we synthesise the same
    // values from config + dp_id_file.
    let heartbeat_cfg: Option<heartbeat::HeartbeatConfig> = if cfg.managed.is_managed() {
        let bundle_on_disk = register::bundle_exists(&cfg.managed.mtls_dir);
        let can_register = cfg.managed.registration_enabled();
        let bundle_provided = cfg.managed.cert_bundle_provided();
        // Log the branch inputs so operators don't have to guess why
        // their DP didn't register (or why it tried to).
        tracing::info!(
            bundle_exists = bundle_on_disk,
            registration_enabled = can_register,
            cert_bundle_provided = bundle_provided,
            mtls_dir = %cfg.managed.mtls_dir,
            "managed-mode bootstrap branch inputs",
        );
        if !bundle_on_disk && !can_register && !bundle_provided {
            // In managed mode we MUST have at least one of:
            //   - a persisted bundle in mtls_dir (subsequent boot)
            //   - registration_token + cp_base_url (legacy /dp/register)
            //   - cert + key + CA PEMs (api7ee parity, dashboard mint)
            // Silently proceeding with the placeholder etcd endpoint
            // from config.managed.yaml turns into an opaque gRPC "dns
            // error" minutes later — instead, fail the boot loudly
            // with exactly what's missing.
            anyhow::bail!(
                "managed mode is enabled but no boot path is available: \
                 registration_token={}, cp_base_url={}, cert_bundle_provided={}; \
                 set AISIX_MANAGED__CP_CERT_PEM + _KEY_PEM + _CA_PEM (recommended) \
                 OR AISIX_MANAGED__REGISTRATION_TOKEN + _CP_BASE_URL (legacy), \
                 or persist an mTLS bundle at {:?}",
                cfg.managed
                    .registration_token
                    .as_deref()
                    .unwrap_or("<unset>"),
                cfg.managed.cp_base_url.as_deref().unwrap_or("<unset>"),
                bundle_provided,
                cfg.managed.mtls_dir,
            );
        }
        if !bundle_on_disk && bundle_provided {
            // api7ee-parity bootstrap: operator minted a cert via the
            // dashboard, inlined the three PEMs into env vars (or
            // referenced files on disk). Materialise the bundle to
            // `mtls_dir`, parse env_id + dp_id from the leaf SAN, and
            // populate cfg.etcd.* exactly like the register branch
            // does. No /dp/register round-trip.
            tracing::info!("managed mode: provisioning from supplied cert bundle (api7ee parity)");
            let p = cert_bundle::provision(&cfg.managed)
                .await
                .map_err(|e| anyhow::anyhow!("DP cert-bundle provisioning failed: {e:#}"))?;
            // The cert-bundle path requires cp_base_url for the
            // heartbeat worker but skips registration_token. cp-api
            // also needs cp_etcd_endpoint to be set (cmux serves
            // gRPC + REST on the same port, but the etcd-client
            // crate wants a host:port string). We accept the same
            // endpoint as the cp_base_url's host:port if
            // cp_etcd_endpoint is unset.
            let cp_base = cfg
                .managed
                .cp_base_url
                .clone()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "managed.cp_base_url required when cert bundle is provided",
                    )
                })?;
            let cp_etcd = cfg
                .managed
                .cp_etcd_endpoint
                .clone()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| {
                    cp_base
                        .trim_start_matches("https://")
                        .trim_start_matches("http://")
                        .to_string()
                });
            tracing::info!(
                dp_id = %p.dp_id,
                env_id = %p.env_id,
                etcd = %cp_etcd,
                "provisioned with dashboard-issued cert bundle",
            );
            cfg.etcd.endpoints = vec![format!("https://{cp_etcd}")];
            cfg.etcd.env_id = p.env_id.clone();
            cfg.etcd.tls = Some(EtcdTlsConfig {
                ca_cert_file: p.ca_cert_path.to_string_lossy().into_owned(),
                client_cert_file: p.client_cert_path.to_string_lossy().into_owned(),
                client_key_file: p.client_key_path.to_string_lossy().into_owned(),
                domain_name: None,
            });
            // Persist dp_id + env_id to the same on-disk paths the
            // register branch uses, so subsequent boots take the
            // bundle-on-disk path without re-running provisioning
            // (which would be a no-op anyway, but the dp_id_file
            // path is what the heartbeat-restore helper reads).
            register::persist_dp_id_for_provisioning(&cfg.managed, &p.dp_id, &p.env_id)
                .await
                .map_err(|e| anyhow::anyhow!("persist dp_id/env_id sidecars: {e:#}"))?;
            // Heartbeat — same shape as register branch. The
            // heartbeat path under cp_base_url is fixed
            // (`/dp/heartbeat`); we don't need a server response to
            // know it.
            Some(heartbeat::HeartbeatConfig::sanitised(
                format!("{}/dp/heartbeat", cp_base.trim_end_matches('/')),
                p.dp_id,
                std::time::Duration::from_secs(15),
                heartbeat::MtlsBundle {
                    ca_cert_path: p.ca_cert_path,
                    client_cert_path: p.client_cert_path,
                    client_key_path: p.client_key_path,
                    extra_ca_pem: extra_ca_pem.clone(),
                },
            ))
        } else if !bundle_on_disk && can_register {
            tracing::info!("managed mode: registering with aisix.cloud CP");
            let cp_etcd = cfg
                .managed
                .cp_etcd_endpoint
                .as_deref()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "managed mode: cp_etcd_endpoint must be set (v3 register no longer \
                         returns the etcd endpoint; the DP container must know it at boot — \
                         set AISIX_MANAGED__CP_ETCD_ENDPOINT to host:port form)"
                    )
                })?
                .to_string();
            let r = register::register_and_persist(&cfg.managed)
                .await
                .map_err(|e| anyhow::anyhow!("DP registration failed: {e:#}"))?;
            tracing::info!(
                dp_id = %r.dp_id,
                env_id = %r.env_id,
                etcd = %cp_etcd,
                "registered with control plane",
            );
            // Plumb the v3 register output into the static config:
            //   - etcd endpoint comes from cp_etcd_endpoint (v3 no
            //     longer returns it in the response).
            //   - env_id comes from the register response and scopes
            //     every etcd read/watch to /aisix/<env_id>/.
            //   - mTLS bundle paths are the freshly persisted files.
            cfg.etcd.endpoints = vec![format!("https://{cp_etcd}")];
            cfg.etcd.env_id = r.env_id.clone();
            cfg.etcd.tls = Some(EtcdTlsConfig {
                ca_cert_file: r.ca_cert_path.to_string_lossy().into_owned(),
                client_cert_file: r.client_cert_path.to_string_lossy().into_owned(),
                client_key_file: r.client_key_path.to_string_lossy().into_owned(),
                domain_name: None, // derive from endpoint host
            });
            // v3 heartbeat is mTLS-only — cp-api derives dp_id from
            // the peer cert SAN URI, so the request carries no
            // Authorization header (§9A.7.2). Hand the heartbeat
            // worker the freshly-persisted bundle paths.
            let cp_base = cfg.managed.cp_base_url.clone().unwrap_or_default();
            Some(heartbeat::HeartbeatConfig::sanitised(
                format!("{}{}", cp_base.trim_end_matches('/'), r.heartbeat_path),
                r.dp_id,
                std::time::Duration::from_secs(15),
                heartbeat::MtlsBundle {
                    ca_cert_path: r.ca_cert_path.clone(),
                    client_cert_path: r.client_cert_path.clone(),
                    client_key_path: r.client_key_path.clone(),
                    extra_ca_pem: extra_ca_pem.clone(),
                },
            ))
        } else if bundle_on_disk {
            // Bundle persisted from a previous boot; load the dp_id
            // and env_id from disk and synthesise heartbeat config
            // from the configured cp_base_url. Registration doesn't
            // re-run — but we still have to carry over the etcd
            // bundle paths and env_id, otherwise the etcd client
            // falls back to the unencrypted placeholder from
            // config.managed.yaml and reads/writes against the wrong
            // (empty) tenant prefix.
            tracing::info!("managed mode: reusing persisted mTLS bundle");
            cfg.etcd.tls = Some(EtcdTlsConfig {
                ca_cert_file: register::ca_cert_path(&cfg.managed.mtls_dir)
                    .to_string_lossy()
                    .into_owned(),
                client_cert_file: register::client_cert_path(&cfg.managed.mtls_dir)
                    .to_string_lossy()
                    .into_owned(),
                client_key_file: register::client_key_path(&cfg.managed.mtls_dir)
                    .to_string_lossy()
                    .into_owned(),
                domain_name: None,
            });
            // Restore env_id from the sibling file written at register
            // time so `etcd.effective_prefix()` keeps scoping reads to
            // `/aisix/<env_id>/` across DP restarts. Missing file is a
            // hard error — proceeding without env_id would silently
            // pull the wrong (empty-prefix) tenant.
            cfg.etcd.env_id = register::read_env_id(&cfg.managed.mtls_dir).map_err(|e| {
                anyhow::anyhow!(
                    "managed mode: bundle on disk but env_id file unreadable at {:?}: {e}",
                    register::env_id_path(&cfg.managed.mtls_dir),
                )
            })?;
            match load_heartbeat_config_from_disk(&cfg.managed, extra_ca_pem.clone()) {
                Ok(h) => Some(h),
                Err(e) => {
                    tracing::warn!(error = %e,
                        "managed mode: heartbeat worker disabled (dp_id unreadable)");
                    None
                }
            }
        } else {
            // can_register branch above caught the "neither bundle nor
            // registration possible" case and bailed. This arm is
            // unreachable in managed mode; kept for exhaustiveness.
            unreachable!("managed-mode branch check is exhaustive")
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

    // Same extra trust root reused by the etcd connect options.
    let connect_options =
        build_etcd_connect_options_with_extra_ca(&cfg.etcd, extra_ca_pem.as_deref())?;
    // effective_prefix() is `<prefix>/<env_id>` in v3 managed mode
    // (env_id populated from the register response above), bare
    // `<prefix>` in self-hosted dev where env_id is empty.
    let etcd_prefix = cfg.etcd.effective_prefix();
    let provider = Arc::new(
        EtcdConfigProvider::connect(
            &cfg.etcd.endpoints,
            etcd_prefix.clone(),
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
        etcd_prefix.clone(),
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
    //
    // Telemetry shares the heartbeat config: same on-disk mTLS bundle
    // (issued by /dp/register) + same cp_base URL host. We derive the
    // /dp/telemetry URL from the /dp/heartbeat URL by swapping the
    // path suffix so the two stay in lock-step on cp_base changes.
    let telemetry_cfg = heartbeat_cfg.as_ref().map(|h| {
        telemetry::TelemetryConfig::new(
            h.url.replace("/dp/heartbeat", "/dp/telemetry"),
            h.mtls.clone(),
        )
    });
    // Budget gate. Same on-disk mTLS bundle as heartbeat; URL is the
    // dpmgr origin (heartbeat URL minus the /dp/heartbeat suffix), the
    // BudgetClient appends /dp/budget_check itself. See prd-09b rev 2
    // §5.5 and AISIX-Cloud PR #95. When the bundle build fails the DP
    // logs and falls back to the default disabled() (allow-all) — a
    // mid-boot config glitch shouldn't take the proxy down.
    let budget_client = heartbeat_cfg.as_ref().and_then(|h| {
        let dpmgr_base = h
            .url
            .strip_suffix("/dp/heartbeat")
            .unwrap_or(h.url.as_str())
            .to_string();
        match heartbeat::build_mtls_client(&h.mtls) {
            Ok(http) => Some(Arc::new(BudgetClient::new(dpmgr_base, http))),
            Err(e) => {
                tracing::warn!(error = %e, "budget_check disabled: mTLS client build failed");
                None
            }
        }
    });
    let heartbeat_task = heartbeat_cfg.map(|h| heartbeat::spawn(h, cancel_rx.clone()));
    let (usage_sink, telemetry_task) = match telemetry_cfg {
        Some(cfg) => {
            let (sink, handle) = telemetry::spawn(cfg, cancel_rx.clone());
            (sink, Some(handle))
        }
        None => (aisix_obs::UsageSink::disabled(), None),
    };

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
    proxy_state = proxy_state.with_usage_sink(usage_sink);
    if let Some(client) = budget_client {
        proxy_state = proxy_state.with_budget_client(client);
    }
    // Live guardrail chain: rebuilds itself whenever the etcd watch
    // supervisor stores a fresh snapshot, so dashboard mutations
    // (`/guardrails` create / enable / delete) take effect within
    // one watch tick. Empty `guardrails` table → chain is a noop;
    // adding the wrapper costs one mutex + ptr-compare per chat,
    // never a regex compile on the hot path. See
    // `aisix_guardrails::LiveGuardrailChain`.
    proxy_state = proxy_state.with_guardrails(aisix_guardrails::LiveGuardrailChain::new(
        snapshot_handle.clone(),
    ));
    // Clone shared trackers before consuming proxy_state in build_router.
    let health_tracker = proxy_state.health.clone();
    let proxy_router = aisix_proxy::build_router(proxy_state);

    // Admin router + listener are only built in standalone mode.
    // In managed mode (`cfg.managed.enabled = true`) the DP reads
    // configuration exclusively from etcd; exposing admin writes or
    // the Playground would bypass the aisix.cloud control plane.
    let admin_serve_handle = if let Some(admin_client) = admin_client {
        let admin_store: Arc<dyn ConfigStore> =
            Arc::new(EtcdConfigStore::new(admin_client, etcd_prefix.clone()));
        let admin_state = AdminState::new(snapshot_handle.clone(), admin_store, &cfg.admin)
            .with_metrics(metrics.clone())
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
        // don't escape managed mode. The health tracker exists on
        // proxy_state and keeps working regardless.
        let _ = &health_tracker;
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
    if let Some(task) = telemetry_task {
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
#[cfg(test)]
fn build_etcd_connect_options(etcd: &EtcdConfig) -> anyhow::Result<Option<ConnectOptions>> {
    build_etcd_connect_options_with_extra_ca(etcd, None)
}

fn build_etcd_connect_options_with_extra_ca(
    etcd: &EtcdConfig,
    extra_ca_pem: Option<&[u8]>,
) -> anyhow::Result<Option<ConnectOptions>> {
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
        let mut ca_pem = std::fs::read(&tls.ca_cert_file)
            .map_err(|e| anyhow::anyhow!("etcd.tls.ca_cert_file = {:?}: {e}", tls.ca_cert_file))?;
        // Append the operator-supplied extra trust root (typically a
        // self-signed dev CA in e2e). rustls's PEM parser handles
        // multi-cert blobs natively, so concatenation is enough — no
        // need to construct a chain explicitly.
        if let Some(extra) = extra_ca_pem {
            if !ca_pem.ends_with(b"\n") {
                ca_pem.push(b'\n');
            }
            ca_pem.extend_from_slice(extra);
        }
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
    extra_ca_pem: Option<Vec<u8>>,
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
        heartbeat::MtlsBundle {
            ca_cert_path: register::ca_cert_path(&managed.mtls_dir),
            client_cert_path: register::client_cert_path(&managed.mtls_dir),
            client_key_path: register::client_key_path(&managed.mtls_dir),
            extra_ca_pem,
        },
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
            env_id: String::new(),
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
            env_id: String::new(),
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
