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
//!  8. Build admin router + dedicated metrics listener
//!  9. Bind + serve the ports (tokio::select! with shutdown signal)
//! 10. On SIGINT/SIGTERM: cancel supervisor, stop accepting, join

use std::error::Error as StdError;
use std::path::{Path, PathBuf};
use std::sync::Arc;

// jemalloc as the global allocator on the shipped/benched targets
// (Linux glibc): under the thread-per-core saturation load, allocator
// time drops from ~15% of request CPU (glibc malloc) to ~6%. Other
// targets keep the system allocator. One deploy caveat: jemalloc bakes
// the build host's page size into the binary, so an aarch64 binary
// built on a 4K-page host aborts at startup on a 64K-page kernel —
// cross-building for such kernels needs JEMALLOC_SYS_WITH_LG_PAGE=16,
// which runs on both page sizes.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

// jemalloc parks freed pages as "dirty" and only advances their decay clock
// on later allocator activity in the same arena, so after a burst of
// large-payload traffic an idle gateway keeps its peak RSS indefinitely
// (#968: a 60s burst of ~120KiB bodies left +38MB resident, flat, on an
// otherwise idle process). The background purge thread decouples purging
// from traffic. Enabled via runtime mallctl on purpose: the equivalent
// `opt.background_thread` startup path carries an upstream warning that it
// "may cause crash or deadlock during initialization". Failure is never
// fatal — foreground decay still bounds RSS under load; only idle-time
// reclamation is lost, which the warning makes visible.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn enable_jemalloc_background_thread() -> Result<bool, tikv_jemalloc_ctl::Error> {
    use tikv_jemalloc_ctl::background_thread;
    // Write-then-read-back: the write is a request, the read is the fact.
    // The outcome is returned so the unit test exercises this function
    // itself — a broken body must fail the test, not stay silently green.
    let outcome = background_thread::write(true).and_then(|()| background_thread::read());
    match &outcome {
        Ok(true) => tracing::info!("jemalloc background purge thread enabled"),
        Ok(false) => tracing::warn!(
            "jemalloc background purge thread did not enable on this target; \
             freed memory will not return to the OS while the process is idle"
        ),
        Err(e) => tracing::warn!(
            error = %e,
            "failed to enable jemalloc background purge thread; freed memory \
             will not return to the OS while the process is idle"
        ),
    }
    outcome
}

mod cert_bundle;
mod export;
mod heartbeat;
mod managed_bundle;
mod telemetry;

use aisix_admin::{AdminState, ConfigStore, EtcdConfigStore, FileManagedStore};
use aisix_cache::{Cache, MemoryCache, RedisCache};
use aisix_core::models::Adapter;
use aisix_core::snapshot::SnapshotHandle;
use aisix_core::{
    AisixSnapshot, CacheBackend, Config, ConfigStatus, EtcdConfig, EtcdTlsConfig, RateLimitBackend,
    SourceKind,
};
use aisix_etcd::{EtcdConfigProvider, SnapshotCache, Supervisor};
use aisix_gateway::{Hub, UpstreamHttpConfig};
use aisix_obs::{init_tracing, Metrics};
use aisix_provider_anthropic::AnthropicBridge;
use aisix_provider_azure_openai::AzureOpenAiBridge;
use aisix_provider_bedrock::BedrockBridge;
use aisix_provider_openai::OpenAiBridge;
use aisix_provider_vertex::VertexBridge;
use aisix_proxy::background::run_background_model_check_once;
use aisix_proxy::budget::BudgetClient;
use aisix_proxy::{CacheBackends, ProxyState};
use aisix_ratelimit::{Limiter, RedisStore};
use clap::Parser;
use etcd_client::{Certificate, ConnectOptions, Identity, TlsOptions};
use std::time::Duration;
use tokio::sync::watch;

#[derive(Debug, Parser)]
#[command(
    name = "aisix",
    version = aisix_core::BUILD_VERSION,
    about = "aisix AI Gateway",
    subcommand_negates_reqs = true
)]
struct Cli {
    /// Path to the bootstrap config file (YAML / TOML / JSON).
    #[arg(short, long, env = "AISIX_CONFIG", required = true)]
    config: Option<PathBuf>,

    #[command(subcommand)]
    command: Option<CliCommand>,
}

#[derive(Debug, clap::Subcommand)]
enum CliCommand {
    /// Validate a resources file (the `resources_file` source) without
    /// booting any listener: the identical read → interpolate → desugar
    /// → validate pipeline runs and the process exits 0 on success or
    /// non-zero with the full aggregated error report. `${VAR}`
    /// references resolve against this process's environment.
    Validate {
        /// Path to the resources file to validate.
        #[arg(long)]
        resources: PathBuf,
    },
    /// Export the resources currently stored in etcd as a `resources.yaml`
    /// the file source (`resources_file`) can load back — the migration /
    /// backup path for a standalone deployment moving from the Admin API
    /// plus etcd to the declarative file. References are resugared to
    /// names, ids are dropped (the file derives them), and live
    /// credentials are replaced with `${VAR}` placeholders unless
    /// `--reveal-secrets` is given.
    Export {
        /// etcd endpoints to read from (comma-separated or repeated).
        #[arg(long, value_delimiter = ',', required = true)]
        etcd: Vec<String>,
        /// Key prefix the resources are stored under. Defaults to the same
        /// canonical prefix the gateway reads from (`etcd.prefix`).
        #[arg(long, default_value_t = EtcdConfig::default().prefix)]
        prefix: String,
        /// Emit real stored credential values inline instead of `${VAR}`
        /// placeholders. UNSAFE — the output then contains live secrets;
        /// intended only for air-gapped, same-host migration.
        #[arg(long)]
        reveal_secrets: bool,
        /// Write the resources file here instead of stdout.
        #[arg(short = 'o', long)]
        output: Option<PathBuf>,
    },
}

/// Threads left to the control surfaces when the proxy serves from
/// thread-per-core workers.
///
/// The proxy's own runtimes belong to its workers in that mode, so this
/// runtime is only running the etcd watch, the admin and metrics
/// listeners, signal handling, and the background exporters. Two threads
/// keep a config reload from stalling behind a telemetry flush without
/// taking a core back from the workers.
const CONTROL_RUNTIME_THREADS: usize = 2;

fn main() -> anyhow::Result<()> {
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

    // Subcommands run without loading the bootstrap config or booting
    // any listener.
    match cli.command {
        Some(CliCommand::Validate { resources }) => return run_validate(&resources),
        Some(CliCommand::Export {
            etcd,
            prefix,
            reveal_secrets,
            output,
        }) => {
            return tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?
                .block_on(export::run(export::ExportArgs {
                    endpoints: etcd,
                    prefix,
                    reveal_secrets,
                    output,
                }));
        }
        None => {}
    }

    let config_path = cli
        .config
        .expect("clap enforces --config unless a subcommand is given");

    // Steps 1-2: config. Read before the runtime exists because
    // `proxy.thread_per_core` and `proxy.workers` decide how many threads
    // this runtime gets — and, in thread-per-core mode, that it is not
    // the runtime serving proxy traffic at all.
    let cfg = Config::load_from_path(Some(&config_path))
        .map_err(|e| anyhow::anyhow!("config load failed: {e}"))?;

    let mut builder = tokio::runtime::Builder::new_multi_thread();
    builder.enable_all();
    builder.worker_threads(if cfg.proxy.thread_per_core_enabled() {
        CONTROL_RUNTIME_THREADS
    } else {
        cfg.proxy.worker_threads()
    });
    builder.build()?.block_on(async_main(cfg))
}

async fn async_main(cfg: Config) -> anyhow::Result<()> {
    // Step 3: process logging. Trace export is not wired here — it is an
    // `observability_exporters` entry (kind = otlp_http) resolved from the
    // live resource snapshot, not a boot-time pipeline.
    init_tracing(&cfg.observability).map_err(|e| anyhow::anyhow!("tracing init failed: {e}"))?;

    // Settings that still parse but do nothing. Warned about only now,
    // because the subscriber above is what makes a warning visible at all.
    for retired in cfg.observability.retired_settings() {
        tracing::warn!("{retired}");
    }

    // After tracing so the enable outcome is observable in the logs; the
    // returned outcome is already logged inside.
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    let _ = enable_jemalloc_background_thread();

    // Before any bridge builds its `reqwest::Client` — the connection
    // pools are constructed once and can't be reconfigured afterwards.
    aisix_gateway::upstream_http::init(upstream_http_config(&cfg.upstream)?)
        .map_err(|e| anyhow::anyhow!("upstream TLS init failed: {e}"))?;

    run(cfg).await
}

/// `aisix validate --resources <file>`: run the identical file-source
/// pipeline (read → interpolate → desugar → canonical schema validation
/// → typed decode → cross-reference checks) that boot and SIGHUP reload
/// use, without booting any listener, then check that every enabled
/// guardrail row actually builds. Exit 0 on success; on failure the
/// aggregated error report (every problem, with kind / entry / field
/// context) goes to stderr and the process exits non-zero.
fn run_validate(resources: &Path) -> anyhow::Result<()> {
    let snapshot = match aisix_core::filesource::load_resources_file(resources, 1) {
        Ok(snapshot) => snapshot,
        Err(report) => {
            eprintln!("{report}");
            std::process::exit(1);
        }
    };
    // Loading only proves the file parses. A guardrail row whose config
    // does not build — an invalid regex, an unknown detector, a
    // `kind: custom` script with a syntax error — is dropped from the
    // chain with a warn line and nothing else: the gateway serves, the
    // config status stays `synced` with an empty `rejected` list, and the
    // screening that row describes never runs. Reporting OK for that
    // would validate the file while missing the thing the file is for.
    let unbuildable = aisix_guardrails::unbuildable_guardrail_rows(&snapshot.guardrails, None);
    if !unbuildable.is_empty() {
        eprintln!(
            "resources file {}: {} guardrail(s) load but cannot run:",
            resources.display(),
            unbuildable.len(),
        );
        for row in &unbuildable {
            eprintln!("  - guardrails ({:?}): {}", row.name, row.reason);
        }
        std::process::exit(1);
    }
    // An enabled guardrail nothing attaches is inert — a scope target can be
    // deleted, so this is legitimate rather than malformed, and the exit code
    // stays 0. But `validate` is the one place an operator looks BEFORE
    // deploying, and the runtime WARN they would otherwise have to notice in
    // a log is the only other signal: the row loads, the resource count
    // includes it, and no request ever mentions it.
    let unattached = aisix_guardrails::unattached_guardrail_names(
        &snapshot.guardrails,
        &snapshot.guardrail_attachments,
    );
    if !unattached.is_empty() {
        eprintln!(
            "resources file {}: {} enabled guardrail(s) have no attachment and inspect no traffic:",
            resources.display(),
            unattached.len(),
        );
        for name in &unattached {
            eprintln!(
                "  - guardrails ({name:?}): add a guardrail_attachments entry to put it in force"
            );
        }
    }
    println!(
        "OK: {} loaded {} resource(s)",
        resources.display(),
        snapshot.total_entries(),
    );
    Ok(())
}

/// Is this gauge label set still describing something the configuration
/// contains?
///
/// The retirement sweep asks this per series; `false` marks the series as
/// no longer current (see `Metrics::retire_stale_gauges`). Two rules keep
/// it from retiring live data:
///
/// Absence has to be POSITIVE: only a label this can resolve, and whose
/// resource is genuinely gone, counts as dead. Anything undecidable stays
/// live, because a wrongly retired gauge hides real state — a worse outcome
/// than the stale sample retirement exists to clean up.
///
/// Both families are therefore judged on `api_key_id` alone, which is the
/// one label here that is an id the snapshot can be asked about:
///
/// - The budget family additionally compares the whole member triple. A
///   bare existence check would call BOTH label sets live after a rebind or
///   a rename and leave the pre-change sample frozen under the same
///   `api_key_id`, which is the case it exists to catch. The triple is read
///   off the same api-key row the emitter reads, so the comparison is exact.
/// - The rate-limit family's `model` is checked because the emit site
///   collapses it to the configured set first (`usage_attr::
///   metric_model_label`): an exact model name, a wildcard ROW name like
///   `openai/*`, or the `unresolved` placeholder. All three are decidable
///   — the first two resolve by name, the third is a placeholder. Were the
///   raw caller string still reaching the label, this check would retire
///   live series every sweep, because a wildcard alias serves concrete
///   names that are in no `models` row.
fn gauge_series_is_live(snap: &AisixSnapshot, series: aisix_obs::LiveGaugeSeries<'_>) -> bool {
    const UNKNOWN: &str = "unknown";
    // What `metric_model_label` emits when nothing resolved. It names no
    // row, so it is a placeholder like `unknown` and must never be retired.
    // Taken from the emitter's own constant rather than spelled again here:
    // the two drifting apart would silently start retiring live series.
    const UNRESOLVED: &str = aisix_proxy::UNRESOLVED_MODEL_LABEL;
    match series {
        aisix_obs::LiveGaugeSeries::Budget {
            api_key_id,
            team_id,
            user_id,
            user_name,
        } => {
            let Some(entry) = snap.apikeys.get_by_id(api_key_id) else {
                return false;
            };
            let key = &entry.value;
            key.team_id.as_deref().unwrap_or(UNKNOWN) == team_id
                && key.user_id.as_deref().unwrap_or(UNKNOWN) == user_id
                && key.user_name.as_deref().unwrap_or(UNKNOWN) == user_name
        }
        aisix_obs::LiveGaugeSeries::RatelimitRemaining { api_key_id, model } => {
            snap.apikeys.get_by_id(api_key_id).is_some()
                && (model == UNKNOWN
                    || model == UNRESOLVED
                    || snap.models.get_by_name(model).is_some())
        }
    }
}

/// SIGHUP → re-run the whole file-source pipeline against the same
/// resources file. Success swaps the snapshot atomically (the same
/// [`SnapshotHandle::store`] the etcd watch supervisor uses), stamping
/// entries with the next generation as their revision; failure keeps
/// serving the previous snapshot and logs the aggregated error report
/// at WARN. There is deliberately no file watcher — reloads are
/// explicit.
async fn file_reload_loop(
    path: PathBuf,
    handle: SnapshotHandle<AisixSnapshot>,
    config_status: ConfigStatus,
    mut cancel: watch::Receiver<bool>,
) {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut hup = match signal(SignalKind::hangup()) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!(error = %e, "cannot install SIGHUP handler; resources file reload disabled");
                return;
            }
        };
        // The boot load stamped revision 1; each successful reload
        // bumps the generation so snapshot consumers observe a change.
        let mut generation: i64 = 1;
        loop {
            tokio::select! {
                _ = hup.recv() => {
                    // File IO + the full parse/validate pipeline are
                    // synchronous — run them off the async workers so a
                    // large file can't stall in-flight requests.
                    let load_path = path.clone();
                    let next_generation = generation + 1;
                    let reload_status = config_status.clone();
                    let loaded = tokio::task::spawn_blocking(move || {
                        aisix_core::filesource::load_resources_file_tracked(
                            &load_path,
                            next_generation,
                            true,
                            &reload_status,
                        )
                    })
                    .await;
                    let loaded = match loaded {
                        Ok(result) => result,
                        Err(join_err) => {
                            tracing::warn!(
                                file = %path.display(),
                                error = %join_err,
                                "resources file reload task failed — keeping the previous snapshot",
                            );
                            continue;
                        }
                    };
                    match loaded {
                        Ok(snapshot) => {
                            generation += 1;
                            let resources = snapshot.total_entries();
                            handle.store(snapshot);
                            tracing::info!(
                                file = %path.display(),
                                resources,
                                generation,
                                "resources file reloaded",
                            );
                        }
                        Err(report) => {
                            tracing::warn!(
                                file = %path.display(),
                                "resources file reload failed — keeping the previous snapshot:\n{report}",
                            );
                        }
                    }
                }
                changed = cancel.changed() => {
                    if changed.is_err() || *cancel.borrow() {
                        break;
                    }
                }
            }
        }
    }
    #[cfg(not(unix))]
    {
        // No SIGHUP on this platform: the boot-time load stays in
        // effect until restart.
        let _ = (path, handle);
        loop {
            if cancel.changed().await.is_err() || *cancel.borrow() {
                break;
            }
        }
    }
}

/// Which managed-mode mTLS bootstrap path to take, given whether a
/// bundle is persisted on disk and whether the env/file vars supply a
/// fresh one. Pure so the precedence rule is unit-tested independently
/// of the side-effecting boot.
#[derive(Debug, PartialEq, Eq)]
enum ManagedBootPath {
    /// Neither a persisted bundle nor supplied certs — cannot boot.
    MissingBundle,
    /// Supplied certs take precedence: (re)provision from them,
    /// overwriting any persisted bundle. This is what makes a CA
    /// rotation land — the on-disk bundle may be stale (#265).
    ProvisionFromEnv,
    /// No supplied certs; reuse the bundle persisted by a prior boot.
    ReusePersisted,
}

/// Supplied certs win over the persisted bundle. Before #265 a persisted
/// bundle was preferred even when env vars carried freshly-rotated
/// certs, so a rotated CP CA left the DP pinned to a stale CA and every
/// etcd/heartbeat connection failed with `UnknownIssuer`.
fn select_managed_boot_path(bundle_on_disk: bool, bundle_provided: bool) -> ManagedBootPath {
    if bundle_provided {
        ManagedBootPath::ProvisionFromEnv
    } else if bundle_on_disk {
        ManagedBootPath::ReusePersisted
    } else {
        ManagedBootPath::MissingBundle
    }
}

/// Run `job` every `period` until cancelled. Two callers: the metrics
/// upkeep sweep and the inert-guardrail sweep.
async fn run_periodic<F>(mut cancel: watch::Receiver<bool>, period: Duration, job: F)
where
    F: Fn(),
{
    let mut interval = tokio::time::interval(period);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        if *cancel.borrow() {
            break;
        }
        tokio::select! {
            _ = interval.tick() => job(),
            changed = cancel.changed() => {
                if changed.is_err() || *cancel.borrow() {
                    break;
                }
            }
        }
    }
}

/// Factored out of `main` so the integration tests can drive the full
/// startup with a real config struct and still use `#[tokio::test]`.
async fn run(mut cfg: Config) -> anyhow::Result<()> {
    // Applied to every listener. `0` (the default) keeps idle client
    // connections open until the peer closes them.
    let downstream_idle_timeout = (cfg.downstream.idle_timeout_secs > 0)
        .then(|| Duration::from_secs(cfg.downstream.idle_timeout_secs));
    // Here rather than in `main` so anything that drives `run` directly —
    // the integration tests — gets the configured interval instead of
    // silently falling back to the default.
    aisix_proxy::sse_keepalive::init(
        (cfg.downstream.sse_keepalive_interval_secs > 0)
            .then(|| Duration::from_secs(cfg.downstream.sse_keepalive_interval_secs)),
    );

    // Operator-supplied extra trust root, threaded into every
    // outbound mTLS client (etcd, heartbeat, telemetry, BudgetClient).
    // Needed for e2e / on-prem deployments where the
    // CP serves a cert distinct from the cert-manager-issued client-
    // cert CA. Production with public-CA certs leaves this `None`.
    let extra_ca_pem =
        managed_bundle::read_optional_ca_pem(cfg.managed.cp_ca_cert_file.as_deref())?;

    // Managed-mode bootstrap. First boot materialises the dashboard-
    // issued cert bundle. Subsequent boots re-use the persisted files
    // and synthesise heartbeat config from config + dp_id_file.
    let heartbeat_cfg: Option<heartbeat::HeartbeatConfig> = if cfg.managed.is_managed() {
        let bundle_on_disk = managed_bundle::bundle_exists(&cfg.managed.mtls_dir);
        let bundle_provided = cfg.managed.cert_bundle_provided();
        // Log the branch inputs so operators don't have to guess why
        // their DP could not bootstrap.
        tracing::info!(
            bundle_exists = bundle_on_disk,
            cert_bundle_provided = bundle_provided,
            mtls_dir = %cfg.managed.mtls_dir,
            "managed-mode bootstrap branch inputs",
        );
        let boot_path = select_managed_boot_path(bundle_on_disk, bundle_provided);
        if boot_path == ManagedBootPath::MissingBundle {
            // In managed mode we MUST have at least one of:
            //   - a persisted bundle in mtls_dir (subsequent boot)
            //   - cert + key + CA PEMs (api7ee parity, dashboard mint)
            // Silently proceeding with the placeholder etcd endpoint
            // from config.managed.yaml turns into an opaque gRPC "dns
            // error" minutes later — instead, fail the boot loudly
            // with exactly what's missing.
            anyhow::bail!(
                "managed mode is enabled but no boot path is available: \
                 cert_bundle_provided={}; \
                 set AISIX_MANAGED__CP_CERT_PEM + _KEY_PEM + _CA_PEM \
                 (or AISIX_MANAGED__CP_CERT_FILE + _KEY_FILE + _CA_FILE), \
                 or persist an mTLS bundle at {:?}",
                bundle_provided,
                cfg.managed.mtls_dir,
            );
        }
        if boot_path == ManagedBootPath::ProvisionFromEnv {
            // Supplied certs win over any persisted bundle: materialise
            // them to `mtls_dir` (overwriting a stale bundle — the #265
            // CA-rotation fix), parse env_id + dp_id from the leaf SAN,
            // and populate cfg.etcd.*. No /dp/register round-trip.
            tracing::info!("managed mode: provisioning from supplied cert bundle (api7ee parity)");
            let p = cert_bundle::provision(&cfg.managed)
                .await
                .map_err(|e| anyhow::anyhow!("DP cert-bundle provisioning failed: {e:#}"))?;
            let etcd_url = derive_cp_etcd_url(&cfg.managed)?;
            tracing::info!(
                dp_id = %p.dp_id,
                env_id = %p.env_id,
                etcd = %etcd_url,
                "provisioned with dashboard-issued cert bundle",
            );
            cfg.etcd.endpoints = vec![etcd_url];
            cfg.etcd.env_id = p.env_id.clone();
            cfg.etcd.tls = Some(EtcdTlsConfig {
                ca_cert_file: p.ca_cert_path.to_string_lossy().into_owned(),
                client_cert_file: p.client_cert_path.to_string_lossy().into_owned(),
                client_key_file: p.client_key_path.to_string_lossy().into_owned(),
                domain_name: None,
            });
            // Persist dp_id + env_id so subsequent boots take the
            // bundle-on-disk path without re-running provisioning.
            managed_bundle::persist_dp_id_for_provisioning(&cfg.managed, &p.dp_id, &p.env_id)
                .await
                .map_err(|e| anyhow::anyhow!("persist dp_id/env_id sidecars: {e:#}"))?;
            // Heartbeat — same shape as register branch. The
            // heartbeat path under cp_base_url is fixed
            // (`/dp/heartbeat`); we don't need a server response to
            // know it.
            let cp_base = cfg
                .managed
                .cp_base_url
                .as_deref()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "managed.cp_base_url required for heartbeat when cert bundle is provided"
                    )
                })?;
            Some(heartbeat::HeartbeatConfig::sanitised(
                format!("{}/dp/heartbeat", cp_base.trim_end_matches('/')),
                p.dp_id,
                std::time::Duration::from_secs(cfg.managed.heartbeat_interval_secs),
                heartbeat::MtlsBundle {
                    ca_cert_path: p.ca_cert_path,
                    client_cert_path: p.client_cert_path,
                    client_key_path: p.client_key_path,
                    extra_ca_pem: extra_ca_pem.clone(),
                },
            ))
        } else if boot_path == ManagedBootPath::ReusePersisted {
            // Bundle persisted from a previous boot; load the dp_id
            // and env_id from disk and synthesise heartbeat config
            // from the configured cp_base_url. Registration doesn't
            // re-run — but we still have to carry over the etcd
            // endpoint, bundle paths and env_id, otherwise the etcd
            // client uses the placeholder from config.managed.yaml
            // and reads/writes against the wrong (empty) tenant
            // prefix.
            tracing::info!("managed mode: reusing persisted mTLS bundle");
            // Derive the real etcd endpoint from cp_base_url /
            // cp_etcd_endpoint — same logic as the cert-bundle
            // provision path. Without this the placeholder
            // "https://placeholder-overridden-at-register:2379"
            // from config.managed.yaml survives into the etcd dial,
            // causing the stale-endpoint bug (AISIX-Cloud#289).
            let etcd_url = derive_cp_etcd_url(&cfg.managed)?;
            tracing::info!(etcd = %etcd_url, "managed mode: etcd endpoint for subsequent boot");
            cfg.etcd.endpoints = vec![etcd_url];
            cfg.etcd.tls = Some(EtcdTlsConfig {
                ca_cert_file: managed_bundle::ca_cert_path(&cfg.managed.mtls_dir)
                    .to_string_lossy()
                    .into_owned(),
                client_cert_file: managed_bundle::client_cert_path(&cfg.managed.mtls_dir)
                    .to_string_lossy()
                    .into_owned(),
                client_key_file: managed_bundle::client_key_path(&cfg.managed.mtls_dir)
                    .to_string_lossy()
                    .into_owned(),
                domain_name: None,
            });
            // Restore env_id from the sibling file written at provision
            // time so `etcd.effective_prefix()` keeps scoping reads to
            // `/aisix/<env_id>/` across DP restarts. Missing file is a
            // hard error — proceeding without env_id would silently
            // pull the wrong (empty-prefix) tenant.
            cfg.etcd.env_id = managed_bundle::read_env_id(&cfg.managed.mtls_dir).map_err(|e| {
                anyhow::anyhow!(
                    "managed mode: bundle on disk but env_id file unreadable at {:?}: {e}",
                    managed_bundle::env_id_path(&cfg.managed.mtls_dir),
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
            // The branch above caught the "neither supplied bundle nor
            // persisted bundle" case and bailed. This arm is
            // unreachable in managed mode; kept for exhaustiveness.
            unreachable!("managed-mode branch check is exhaustive")
        }
    } else {
        None
    };

    let (cancel_tx, cancel_rx) = watch::channel(false);
    // Flipped when the drain STARTS, where `cancel` is flipped when it
    // ends. Only a listener reads it, and only to hand an HTTP/2
    // downstream its GOAWAY without closing the listener under traffic
    // a balancer is still routing here (AISIX-Cloud#1395).
    let (retire_tx, retire_rx) = watch::channel(false);
    let shutdown = ShutdownWatch {
        retire: retire_rx,
        cancel: cancel_rx.clone(),
    };

    // Steps 4-6: the resource source — either the standalone resources
    // file (`resources_file` in config) or etcd + watch supervisor.
    // Config validation already guaranteed exactly one is selected.
    let file_source_path = cfg.resources_file.clone().map(PathBuf::from);
    let (snapshot_handle, supervisor, watch_task, admin_client, config_status) =
        if let Some(path) = &file_source_path {
            // FILE MODE: load once at boot, fail fast with the aggregated
            // error report on any problem. SIGHUP re-runs the identical
            // pipeline; a failed reload keeps the last-good snapshot.
            let config_status = ConfigStatus::new(SourceKind::File);
            let snapshot =
                aisix_core::filesource::load_resources_file_tracked(path, 1, true, &config_status)
                    .map_err(|report| anyhow::anyhow!("{report}"))?;
            tracing::info!(
                file = %path.display(),
                resources = snapshot.total_entries(),
                "resources loaded from file",
            );
            let handle = SnapshotHandle::new(snapshot);
            let reload_task = tokio::spawn(file_reload_loop(
                path.clone(),
                handle.clone(),
                config_status.clone(),
                cancel_rx.clone(),
            ));
            (handle, None, reload_task, None, config_status)
        } else {
            // ETCD MODE (unchanged behavior).
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
            // Separate client for the admin read surface — only needed when
            // the admin listener is bound. We could share a single underlying
            // connection via `Client::clone()` but keeping two is cleaner:
            // admin reads and the watch stream don't contend on the same mutex.
            // Skipped whenever the admin listener is not bound — managed mode,
            // or `admin.enabled = false` — so admin-off doesn't pay for (or
            // fail boot on) a connection it immediately drops; `/status/models`
            // then reads through the snapshot, as it does in managed mode.
            let admin_client = if cfg.managed.is_managed() || !cfg.admin.enabled {
                None
            } else {
                Some((
                    etcd_client::Client::connect(&cfg.etcd.endpoints, connect_options.clone())
                        .await
                        .map_err(|e| anyhow::anyhow!("etcd admin client connect failed: {e}"))?,
                    etcd_prefix.clone(),
                ))
            };
            // Snapshot cache: persist to disk so the DP can serve traffic
            // from the last-known config across CP outages and restarts.
            // Managed mode defaults to /var/lib/aisix/config_cache.json;
            // self-hosted etcd mode enables it only when the operator sets
            // a path explicitly; "" disables it in either mode.
            let snapshot_cache = match cfg.managed.effective_snapshot_cache_path() {
                Some(path) => SnapshotCache::new(path),
                None => SnapshotCache::disabled(),
            };
            let supervisor = Arc::new(Supervisor::with_cache(
                provider,
                etcd_prefix,
                snapshot_cache,
            ));
            // Seed the snapshot from disk before the etcd cycle starts so the
            // proxy is ready to serve from cached config the moment the watch
            // task takes its first iteration.
            supervisor.restore_from_cache();
            let config_status = supervisor.config_status();
            let handle = supervisor.handle();
            let watch_task = tokio::spawn(supervisor.clone().run(cancel_rx.clone()));
            (
                handle,
                Some(supervisor),
                watch_task,
                admin_client,
                config_status,
            )
        };
    // Spawn heartbeat worker if we have a config for it. The
    // JoinHandle is awaited after graceful shutdown below so the
    // final in-flight beat drains cleanly.
    //
    // Telemetry shares the heartbeat config: same on-disk mTLS bundle
    // + same cp_base URL host. We derive the
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
    let (usage_sink, telemetry_task) = match telemetry_cfg {
        Some(cfg) => {
            let (sink, handle) = telemetry::spawn(cfg, cancel_rx.clone());
            (sink, Some(handle))
        }
        None => (aisix_obs::UsageSink::disabled(), None),
    };

    // Steps 7-8: build Hub, shared components, then routers.
    let hub = Arc::new(build_hub());
    // Rate-limit backend (#798). Default `memory` keeps per-process
    // counters; `redis` shares them across every replica so a cluster
    // enforces one global window instead of one-per-replica. The
    // `ratelimit.backend` field is the selector — a stray `redis` block
    // under `backend: memory` is ignored (Config::validate already
    // guarantees a `redis` block when `backend = redis`).
    let limiter = Arc::new(match cfg.ratelimit.backend {
        RateLimitBackend::Redis => {
            let redis_cfg = cfg
                .ratelimit
                .redis
                .as_ref()
                .expect("validated: ratelimit.redis present when backend = redis");
            tracing::info!(
                target: "aisix::ratelimit",
                backend = "redis",
                "connecting shared rate-limit backend"
            );
            // No URL in the message: redis URLs carry credentials.
            let store = RedisStore::connect(redis_cfg)
                .await
                .map_err(|e| {
                    anyhow::anyhow!("redis rate-limit connect failed (ratelimit.redis): {e}")
                })?
                .with_conc_ttl(cfg.ratelimit.concurrency_ttl_secs)
                .with_env_namespace(&cfg.etcd.env_id);
            Limiter::with_store(Arc::new(store))
        }
        RateLimitBackend::Memory => Limiter::new(),
    });
    // env_id is resolved by now (managed provisioning / sidecar restore
    // above); it becomes the constant `env_id` label on the SLO latency
    // histograms. Standalone DPs leave it empty → "unknown".
    // AISIX-Cloud#1226: operator bucket overrides. Validation errors are
    // boot-fatal — a silently ignored override would leave the deployment
    // reading quantiles off bucket edges it did not choose.
    let histogram_buckets =
        aisix_obs::HistogramBuckets::from_config(&cfg.observability.metrics.buckets)
            .map_err(|e| anyhow::anyhow!(e))?;
    let metrics = Arc::new(Metrics::new_with_buckets(
        &cfg.etcd.env_id,
        &histogram_buckets,
    ));
    let metrics_upkeep_task = {
        let metrics = metrics.clone();
        let snapshot = snapshot_handle.clone();
        tokio::spawn(run_periodic(
            cancel_rx.clone(),
            Duration::from_secs(5),
            move || {
                metrics.run_upkeep();
                // Retire the gauge series whose key is gone. These families
                // are written only from the request (or health-check) path,
                // and the recorder registers no idle timeout, so a deleted
                // or rebound api key leaves a sample frozen at its last
                // value while still claiming to describe the present.
                let snap = snapshot.load();
                metrics.retire_stale_gauges(|series| gauge_series_is_live(&snap, series));
            },
        ))
    };
    // Cache backends (#519 B.8). The memory cache is always built
    // (in-process, cheap); the redis cache is built iff `cache.redis`
    // is configured. Which instance serves a request is selected by
    // the matched CachePolicy's `backend` field at the proxy's cache
    // gate — `cache.backend` no longer picks a single global cache.
    // It still fails fast on the contradictory `backend = redis`
    // without a `cache.redis` block, so old configs that relied on it
    // surface the misconfiguration at boot instead of per request.
    if cfg.cache.backend == CacheBackend::Redis && cfg.cache.redis.is_none() {
        anyhow::bail!("cache.backend = redis but cache.redis missing");
    }
    let redis_cache: Option<Arc<dyn Cache>> = match cfg.cache.redis.as_ref() {
        Some(redis_cfg) => {
            tracing::info!(target: "aisix::cache", backend = "redis", "connecting cache backend");
            let redis = RedisCache::connect(redis_cfg)
                .await
                .map_err(|e| {
                    // Deliberately no URL in the message: redis URLs carry
                    // credentials (redis://user:pass@host) and this error
                    // lands in logs that may ship to centralized sinks.
                    anyhow::anyhow!("redis cache connect failed (cache.redis): {e}")
                })?
                .with_env_namespace(&cfg.etcd.env_id);
            Some(Arc::new(redis) as Arc<dyn Cache>)
        }
        None => None,
    };
    // Shared semantic (L2) store for `backend: redis` policies. Wired
    // only when the server passes the vector-search probe — a plain
    // Redis 6/7 (or cluster mode, unsupported yet) degrades those
    // policies to exact-only, loudly, HERE at boot rather than
    // silently per request.
    let semantic_redis: Option<Arc<dyn aisix_cache::SemanticCacheStore>> =
        match cfg.cache.redis.as_ref() {
            Some(redis_cfg) if redis_cfg.mode == aisix_core::RedisMode::Cluster => {
                tracing::warn!(
                    target: "aisix::cache",
                    "cache.redis is in cluster mode; semantic matching on backend=redis \
                     policies is not supported yet and stays exact-only"
                );
                None
            }
            Some(redis_cfg) => {
                // Degrade (never abort) on any failure here: the exact
                // redis cache above is the load-bearing connection; the
                // semantic store is an optimization layer.
                match aisix_cache::RedisSemanticCache::connect(redis_cfg).await {
                    Err(e) => {
                        tracing::warn!(
                            target: "aisix::cache",
                            error = %e,
                            "redis semantic cache connect failed; semantic matching \
                             on backend=redis policies stays exact-only"
                        );
                        None
                    }
                    Ok(store) => {
                        let store = store.with_env_namespace(&cfg.etcd.env_id);
                        match store.probe().await {
                            Ok(()) => {
                                match store.sweep_empty_indexes().await {
                                    Ok(dropped) if dropped > 0 => {
                                        tracing::info!(
                                            target: "aisix::cache",
                                            dropped,
                                            "reclaimed empty semantic-cache indexes"
                                        );
                                    }
                                    Ok(_) => {}
                                    Err(e) => {
                                        tracing::warn!(
                                            target: "aisix::cache",
                                            error = %e,
                                            "semantic-cache index sweep failed; continuing"
                                        );
                                    }
                                }
                                tracing::info!(
                                    target: "aisix::cache",
                                    "cache.redis supports vector search; semantic matching \
                                     enabled for backend=redis policies"
                                );
                                Some(Arc::new(store) as Arc<dyn aisix_cache::SemanticCacheStore>)
                            }
                            Err(e) => {
                                tracing::warn!(
                                    target: "aisix::cache",
                                    error = %e,
                                    "cache.redis has no vector-search support; semantic matching \
                                     on backend=redis policies stays exact-only"
                                );
                                None
                            }
                        }
                    }
                }
            }
            None => None,
        };
    let mut cache_backends =
        CacheBackends::new(Arc::new(MemoryCache::with_defaults()), redis_cache);
    if let Some(store) = semantic_redis {
        cache_backends = cache_backends.with_semantic_redis(store);
    }
    let cache = Some(cache_backends);

    let mut proxy_state = ProxyState::with_components(
        snapshot_handle.clone(),
        hub.clone(),
        limiter.clone(),
        metrics.clone(),
        cache.clone(),
        &cfg.proxy,
    );
    // Wire the prometheus emit/drop counters into the sink (#408)
    // so a real DP scrape surfaces UsageEvent throughput without
    // needing cp-api or an OTLP receiver in the loop.
    proxy_state = proxy_state.with_usage_sink(usage_sink.with_metrics((*metrics).clone()));
    // AISIX-Cloud#1045: operator UA→client_type rules. Compile errors are
    // boot-fatal — a dropped rule would silently misattribute traffic.
    let client_classifier =
        aisix_obs::ClientTypeClassifier::compile(&cfg.observability.metrics.client_type_rules)
            .map_err(|e| anyhow::anyhow!(e))?;
    proxy_state = proxy_state.with_client_classifier(Arc::new(client_classifier));
    proxy_state = proxy_state.with_default_retries(cfg.upstream.retries);
    proxy_state =
        proxy_state.with_default_timeouts(cfg.upstream.timeout_ms, cfg.upstream.stream_timeout_ms);
    if let Some(client) = budget_client {
        proxy_state = proxy_state.with_budget_client(client);
    }
    // Live guardrail index: resolves per-request chains from
    // attachment scope + priority, rebuilding lazily whenever the
    // etcd watch supervisor stores a fresh snapshot. Dashboard
    // mutations (`/guardrails` and `/guardrail_attachments` CRUD)
    // take effect within one watch tick. Empty attachment table →
    // every resolved chain is empty (no-op). See
    // `aisix_guardrails::LiveGuardrailIndex`.
    //
    // `bedrock_endpoint_url` is the deployment-wide override for
    // kind=bedrock guardrails; empty string is normalized to
    // `None` so a `docker run -e AISIX_BEDROCK_ENDPOINT_URL=`
    // doesn't accidentally redirect Bedrock calls into thin air.
    let bedrock_endpoint_url = cfg.bedrock_endpoint_url.clone().filter(|s| !s.is_empty());
    let guardrail_metrics_sink = proxy_state.metrics.clone();
    let guardrail_embedder = proxy_state.guardrail_embedder();
    proxy_state =
        proxy_state.with_guardrail_index(aisix_guardrails::LiveGuardrailIndex::new_with_sink(
            snapshot_handle.clone(),
            bedrock_endpoint_url,
            Some(guardrail_metrics_sink),
            guardrail_embedder,
        ));
    // Heartbeat worker — spawned after proxy_state exists so it can read
    // the exporter fan-out's delivery counters. Each tick reports:
    //   - rejected_resources: the supervisor's loader rejections (#115)
    //   - applied_revision: the highest etcd revision the supervisor has
    //     applied, so cp-api can show "propagating…" until the DP catches
    //     up with a kine write (#519 B.3)
    //   - config_hash: the hash of the applied (served) config set, so
    //     cp-api can diff the hash a node reports against the hash it
    //     expects that node to be serving (#774)
    //   - supported_guardrail_kinds + exporter_health (#519 B.6 / D.2)
    let heartbeat_task = heartbeat_cfg.map(|mut h| {
        // Heartbeat only exists in managed mode, which config
        // validation pins to the etcd source — the supervisor is
        // always present here.
        let supervisor = supervisor
            .as_ref()
            .expect("managed mode implies the etcd resource source (validated at boot)");
        let supervisor_for_heartbeat = Arc::clone(supervisor);
        h = h.with_rejection_fetcher(Arc::new(move || {
            supervisor_for_heartbeat.recent_rejections()
        }));
        let watch_status = supervisor.watch_status();
        h = h.with_applied_revision_fetcher(Arc::new(move || watch_status.snapshot().revision));
        let config_status_for_heartbeat = supervisor.config_status();
        h = h.with_config_hash_fetcher(Arc::new(move || {
            config_status_for_heartbeat.applied_config_hash()
        }));
        let supervisor_for_partial = Arc::clone(supervisor);
        h = h.with_partial_compat_fetcher(Arc::new(move || {
            supervisor_for_partial.recent_partial_compat()
        }));
        let fan_out = proxy_state.otlp_fan_out.clone();
        h = h.with_exporter_health_fetcher(Arc::new(move || fan_out.exporter_stats()));
        heartbeat::spawn(h, cancel_rx.clone())
    });

    // Name the guardrails that are attached to nothing, on a timer.
    //
    // On a timer rather than from the index build, because the thing being
    // reported does not coincide with a configuration change. A build
    // happens only when the snapshot version moves, so a notice emitted
    // from one is delivered only if somebody writes something — and the two
    // cases that matter most are the ones where nobody does: a scope target
    // deleted out from under a rule that was screening traffic yesterday,
    // and a gateway restarted onto standing configuration.
    //
    // Nor is it gated on a readiness flag. `ConfigStatus::is_ready` means
    // "a load has been applied", which `Supervisor::restore_from_cache`
    // satisfies before the watch task has reached etcd at all — a one-shot
    // report behind that gate reads the CACHED generation and never looks
    // again, so it goes quiet exactly when the cache is the stale thing
    // (someone changed an attachment while this gateway was down). A sweep
    // that keeps running converges either way. It may still name a row
    // that live etcd has attached, during a long outage: that is not a
    // false alarm, because the gateway is serving the cached generation at
    // that moment and in it the rule really does inspect nothing.
    {
        let sweep_snapshot = snapshot_handle.clone();
        tokio::spawn(run_periodic(
            cancel_rx.clone(),
            aisix_guardrails::UNATTACHED_SWEEP_INTERVAL,
            move || {
                let snap = sweep_snapshot.load();
                aisix_guardrails::sweep_unattached_guardrails(
                    &snap.guardrails,
                    &snap.guardrail_attachments,
                );
            },
        ));
    }

    // Clone shared trackers before consuming proxy_state in build_router.
    let health_tracker = proxy_state.health.clone();
    let livez_state = proxy_state.livez.clone();
    let runtime_status_tracker = proxy_state.runtime_status.clone();
    let background_snapshot = snapshot_handle.clone();
    let background_hub = hub.clone();
    let background_runtime_status_tracker = runtime_status_tracker.clone();
    let background_cancel_rx = cancel_rx.clone();
    // Wire the config-freshness probe so the proxy's /readyz reflects etcd
    // watch staleness (and pre-first-apply startup), not just shutdown (#591).
    proxy_state = match supervisor.as_ref() {
        Some(supervisor) => {
            let readyz_watch_status = supervisor.watch_status();
            proxy_state.with_config_apply_age(Arc::new(move || {
                readyz_watch_status.snapshot().last_apply_age
            }))
        }
        // File mode: the snapshot is applied synchronously at boot /
        // SIGHUP and there is no watch stream to go stale — report the
        // config as always freshly applied.
        None => proxy_state.with_config_apply_age(Arc::new(|| Some(std::time::Duration::ZERO))),
    };
    let proxy_router = aisix_proxy::build_router(proxy_state);

    let background_check_task = tokio::spawn(async move {
        let mut cancel = background_cancel_rx;
        loop {
            if *cancel.borrow() {
                break;
            }
            let snapshot = background_snapshot.load();
            run_background_model_check_once(
                snapshot.clone(),
                background_hub.clone(),
                background_runtime_status_tracker.clone(),
                "background-model-check",
            )
            .await;
            let sleep_for = background_check_interval(snapshot.as_ref());
            tokio::select! {
                changed = cancel.changed() => {
                    if changed.is_err() || *cancel.borrow() {
                        break;
                    }
                }
                _ = tokio::time::sleep(sleep_for) => {}
            }
        }
    });

    // Admin router + listener are only built in standalone mode.
    // In managed mode (`cfg.managed.enabled = true`) the DP reads
    // configuration exclusively from etcd; exposing the admin surface
    // or the Playground would bypass the AISIX Cloud control plane.
    //
    // Which store backs the admin surface depends on the resource
    // source: etcd standalone reads the etcd store; file mode reads a
    // view of the file-loaded snapshot. The admin resource surface is
    // read-only — writes were removed with the Admin API write path.
    let admin_store: Option<Arc<dyn ConfigStore>> = match (admin_client, &file_source_path) {
        (Some((client, prefix)), _) => Some(Arc::new(EtcdConfigStore::new(client, prefix))),
        (None, Some(_)) if !cfg.managed.is_managed() => {
            Some(Arc::new(FileManagedStore::new(snapshot_handle.clone())))
        }
        _ => None,
    };
    // Per-model runtime health as an operational read on the metrics/status
    // listener (`GET /status/models`). Standalone mode shares the admin
    // surface's store handle, so the status-listener view and
    // `GET /admin/v1/models/status` read the very same source; managed mode
    // has no admin store and serves the applied snapshot through the same
    // read-only view file mode uses.
    let status_models_state = aisix_admin::ModelsStatusState {
        store: match &admin_store {
            Some(store) => Arc::clone(store),
            None => Arc::new(FileManagedStore::new(snapshot_handle.clone())),
        },
        runtime_status_tracker: Some(runtime_status_tracker.clone()),
    };
    let admin_serve_handle = if let Some(admin_store) = admin_store.filter(|_| cfg.admin.enabled) {
        let mut admin_state = AdminState::new(snapshot_handle.clone(), admin_store, &cfg.admin)
            // Share the health tracker so /admin/v1/health reflects live
            // per-model upstream failure counts.
            .with_health_tracker(health_tracker)
            .with_livez_state(livez_state.clone())
            // Share runtime status so /admin/v1/models/status exposes
            // direct-model cooldown/background-health state.
            .with_runtime_status_tracker(runtime_status_tracker)
            // Share the proxy router so the playground endpoint can forward
            // requests in-process without an extra network hop.
            .with_proxy_router(proxy_router.clone());
        if let Some(supervisor) = supervisor.as_ref() {
            // Share the supervisor's freshness state so /admin/v1/health
            // exposes etcd watch staleness — without this, a wedged
            // watch lets the gateway serve stale config indefinitely
            // while reporting healthy. See issue #114.
            admin_state = admin_state.with_watch_status(supervisor.watch_status());
        }
        let admin_router = aisix_admin::build_router(admin_state);

        let admin_addr: std::net::SocketAddr = cfg.admin.addr.parse()?;
        let admin_tls = cfg.admin.tls.clone();
        Some(tokio::spawn(serve_http(
            admin_addr,
            admin_router,
            admin_tls,
            downstream_idle_timeout,
            shutdown.clone(),
            "admin",
            None,
            None,
        )))
    } else {
        // Drop unused shared components so the compiler can see they
        // don't escape the admin-less paths. The health tracker exists on
        // proxy_state and keeps working regardless.
        let _ = (&health_tracker, &livez_state, &runtime_status_tracker);
        if cfg.managed.is_managed() {
            tracing::info!("managed mode enabled — admin surface not bound");
        } else {
            tracing::info!("admin.enabled = false — admin surface not bound");
        }
        None
    };

    // Dedicated metrics listener — the only Prometheus scrape surface,
    // bound whenever prometheus is enabled, identical in standalone and
    // managed mode (default `0.0.0.0:9090`). Shares the same `metrics`
    // handle as the proxy, so one scrape reflects all request paths.
    let metrics_serve_handle = {
        let prom = &cfg.observability.metrics.prometheus;
        if prom.enabled {
            let metrics_addr: std::net::SocketAddr = prom.addr.parse()?;
            // Fail boot loudly if the metrics port is unavailable, rather
            // than silently serving no metrics until shutdown — the
            // listener is spawned and only joined post-shutdown, so a
            // swallowed bind error would leave the gateway looking healthy
            // while every scrape gets connection-refused (the exact
            // observability gap this listener exists to close). `serve_http`
            // re-binds; the brief gap before re-bind is benign for a boot
            // probe.
            std::net::TcpListener::bind(metrics_addr)
                .map_err(|e| anyhow::anyhow!("metrics listener bind {metrics_addr} failed: {e}"))?;
            let metrics_router = aisix_admin::metrics_router(
                metrics.clone(),
                config_status.clone(),
                prom,
                status_models_state,
            );
            Some(tokio::spawn(serve_http(
                metrics_addr,
                metrics_router,
                None,
                downstream_idle_timeout,
                shutdown.clone(),
                "metrics",
                None,
                None,
            )))
        } else {
            None
        }
    };

    // Step 9: bind + serve the proxy (always). Admin is handled above.
    let proxy_addr: std::net::SocketAddr = cfg.proxy.addr.parse()?;
    let proxy_tls = cfg.proxy.tls.clone();
    let proxy_workers = cfg
        .proxy
        .thread_per_core_enabled()
        .then(|| cfg.proxy.worker_threads());
    let proxy_serve = serve_http(
        proxy_addr,
        proxy_router,
        proxy_tls,
        downstream_idle_timeout,
        shutdown,
        "proxy",
        proxy_workers,
        // Only the proxy listener's connections are counted: the drain is
        // about client traffic, and folding the platform's probes into the
        // number would report a connection nobody is waiting on.
        Some(livez_state.clone()),
    );

    // Step 10: shutdown coordinator. Whichever of (signal, proxy, admin)
    // completes first triggers the rest.
    let signal_task = tokio::spawn(wait_for_signal(
        cancel_tx.clone(),
        retire_tx,
        livez_state,
        Duration::from_secs(cfg.shutdown.min_drain_secs),
    ));

    proxy_serve
        .await
        .map_err(|e| anyhow::anyhow!("proxy serve error: {e}"))?;
    if let Some(handle) = admin_serve_handle {
        match handle.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(anyhow::anyhow!("admin serve error: {e}")),
            Err(e) => return Err(anyhow::anyhow!("admin task join error: {e}")),
        }
    }
    if let Some(handle) = metrics_serve_handle {
        match handle.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(anyhow::anyhow!("metrics serve error: {e}")),
            Err(e) => return Err(anyhow::anyhow!("metrics task join error: {e}")),
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
    let _ = metrics_upkeep_task.await;
    let _ = background_check_task.await;
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

/// Derive the etcd endpoint from `managed.cp_base_url` or
/// `managed.cp_etcd_endpoint`. Returns a fully-qualified
/// `https://<host:port>` URL for the etcd gRPC dial.
///
/// Logic: if `cp_etcd_endpoint` is set, use it as `host:port`;
/// otherwise strip the scheme from `cp_base_url` (cmux means the
/// same port serves both REST and etcd gRPC).
fn derive_cp_etcd_url(managed: &aisix_core::ManagedConfig) -> anyhow::Result<String> {
    if let Some(ep) = managed
        .cp_etcd_endpoint
        .as_deref()
        .filter(|s| !s.is_empty())
    {
        return Ok(format!("https://{ep}"));
    }
    let cp_base = managed
        .cp_base_url
        .as_deref()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "managed mode: cp_base_url must be set \
                 (set AISIX_MANAGED__CP_BASE_URL)"
            )
        })?;
    let host_port = cp_base
        .strip_prefix("https://")
        .or_else(|| cp_base.strip_prefix("http://"))
        .unwrap_or(cp_base)
        .trim_end_matches('/');
    Ok(format!("https://{host_port}"))
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
        std::time::Duration::from_secs(managed.heartbeat_interval_secs),
        heartbeat::MtlsBundle {
            ca_cert_path: managed_bundle::ca_cert_path(&managed.mtls_dir),
            client_cert_path: managed_bundle::client_cert_path(&managed.mtls_dir),
            client_key_path: managed_bundle::client_key_path(&managed.mtls_dir),
            extra_ca_pem,
        },
    ))
}

/// Register all bridge-backed provider implementations on a fresh
/// Hub. The Hub is created once at startup; future dynamic reload
/// lands behind the same `register()` call.
///
/// Jina is intentionally NOT registered: per #213 Phase 2 Jina is
/// exposed only via `/v1/rerank`, which is a verbatim HTTP forward
/// (`aisix-proxy::rerank`) and bypasses the Bridge trait entirely.
///
/// Cohere chat is served by the `Adapter::Openai` family bridge —
/// cp-api stores Cohere's PK with `adapter: "openai"` and `api_base`
/// pointing at `https://api.cohere.com/compatibility/v1` (per
/// <https://docs.cohere.com/reference/chat>). Cohere's `/v1/rerank`
/// native surface is keyed off `Model.provider == "cohere"` in
/// `crates/aisix-proxy/src/rerank.rs` and bypasses the Bridge.
/// Translate the `upstream:` config block into the gateway's client
/// settings. Every duration treats `0` as "leave this knob off".
///
/// Fails when `upstream.tls` names a file that cannot be read or does
/// not hold the PEM it claims to — the boot is where an operator can
/// still act on that, and it is a far better signal than the generic
/// `UnknownIssuer` transport error the misconfiguration otherwise
/// produces on every upstream call.
fn upstream_http_config(
    cfg: &aisix_core::config::UpstreamConfig,
) -> anyhow::Result<UpstreamHttpConfig> {
    fn ms(v: u64) -> Option<Duration> {
        (v > 0).then(|| Duration::from_millis(v))
    }
    fn secs(v: u64) -> Option<Duration> {
        (v > 0).then(|| Duration::from_secs(v))
    }
    let tls = aisix_gateway::TlsSettings::load("upstream.tls", &cfg.tls)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    if !tls.verify {
        tracing::warn!(
            "upstream.tls.verify is false: upstream certificates are NOT checked, so any \
             peer able to intercept the connection can read and rewrite prompts, responses, \
             and upstream API keys"
        );
    }
    Ok(UpstreamHttpConfig {
        connect_timeout: ms(cfg.connect_timeout_ms),
        tcp_keepalive: secs(cfg.tcp_keepalive_secs),
        tcp_keepalive_interval: secs(cfg.tcp_keepalive_interval_secs),
        tcp_keepalive_retries: (cfg.tcp_keepalive_retries > 0).then_some(cfg.tcp_keepalive_retries),
        pool_idle_timeout: secs(cfg.pool_idle_timeout_secs),
        pool_max_idle_per_host: cfg.pool_max_idle_per_host,
        tls,
    })
}

fn build_hub() -> Hub {
    let hub = Hub::new();

    // ─── Family bridges (closed 5-value Adapter enum) ────────────────
    //
    // Catches every catalog vendor whose `ProviderKey.adapter` matches
    // one of these. Any new long-tail OpenAI-compat vendor cp-api
    // admits (xai, openrouter, cerebras, moonshotai, …) routes here
    // through `Hub::dispatch_two_tier` without a DP code change.
    //
    // CUTOVER CAUTION (non-openai families): cp-api admits
    // `google-vertex`, `azure`, `amazon-bedrock` Provider Keys via
    // its adapter_map (#302 Phase B). The Vertex / Azure / Bedrock
    // bridges below are functional implementations (Phases E/F/G).
    hub.register_family(Adapter::Openai, Arc::new(OpenAiBridge::new()));
    hub.register_family(Adapter::Anthropic, Arc::new(AnthropicBridge::new()));
    hub.register_family(Adapter::Vertex, Arc::new(VertexBridge::new()));
    hub.register_family(Adapter::AzureOpenai, Arc::new(AzureOpenAiBridge::new()));
    hub.register_family(Adapter::Bedrock, Arc::new(BedrockBridge::new()));

    // ─── Specialized vendor bridges ─────────────────────────────────
    //
    // `openai` and `anthropic` are the two canonical vendors with a
    // dedicated specialized bridge, so a ProviderKey whose `provider`
    // is exactly `"openai"`/`"anthropic"` resolves through the
    // specialized tier of `dispatch_two_tier`. Long-tail OpenAI-compat
    // vendors (xai, openrouter, groq, deepseek, …) carry `adapter:
    // openai` and resolve through the family tier above instead.
    hub.register_specialized("openai", Arc::new(OpenAiBridge::new()));
    hub.register_specialized("anthropic", Arc::new(AnthropicBridge::new()));

    hub
}

fn background_check_interval(snapshot: &aisix_core::AisixSnapshot) -> std::time::Duration {
    let min_interval = snapshot
        .models
        .entries()
        .into_iter()
        .filter_map(|entry| entry.value.background_model_check.clone())
        .filter(|cfg| cfg.enabled)
        .map(|cfg| cfg.interval_seconds)
        .min()
        .unwrap_or(1);
    std::time::Duration::from_secs(min_interval.max(1))
}

/// The two moments a listener has to tell apart during a shutdown.
///
/// `retire` flips when the drain STARTS (SIGTERM): open connections are
/// told to stop taking new work while the listener KEEPS accepting,
/// because a balancer only learns about the `/readyz` 503 on its next
/// probe and refusing everything it routes in between is the failure a
/// graceful shutdown exists to avoid. `cancel` flips when the drain is
/// OVER: stop accepting, and close what is left.
///
/// The two must stay separate signals. Collapsing them into one is the
/// same as having no HTTP/2 retirement signal at all, because the only
/// way to retire a connection would be to close the listener with it
/// (AISIX-Cloud#1395).
#[derive(Clone)]
struct ShutdownWatch {
    retire: watch::Receiver<bool>,
    cancel: watch::Receiver<bool>,
}

impl ShutdownWatch {
    /// Resolves once `rx` is true — immediately when it was already true
    /// before this connection was accepted, which is the case for every
    /// connection accepted during a drain. `Receiver::changed` would
    /// wait for the NEXT flip and so never return for those.
    ///
    /// A dropped sender resolves too: the process is going away, and
    /// "retire now" is the safe reading of that.
    async fn signalled(rx: &mut watch::Receiver<bool>) {
        let _ = rx.wait_for(|flag| *flag).await;
    }

    /// Whichever comes first. `cancel` belongs here because it can flip
    /// without `retire` ever flipping: a failure elsewhere in the
    /// process cancels it without a drain.
    async fn retire_or_cancel(&mut self) {
        tokio::select! {
            _ = Self::signalled(&mut self.retire) => {}
            _ = Self::signalled(&mut self.cancel) => {}
        }
    }

    async fn cancelled(&mut self) {
        Self::signalled(&mut self.cancel).await;
    }
}

/// Which HTTP version an accepted connection settled on.
///
/// The gateway decides this itself rather than leaving it to
/// `hyper_util`'s auto builder, because the version decides which
/// retirement signal the connection can be handed and the auto
/// connection does not report which version it negotiated. Its
/// `graceful_shutdown` is version-blind, and on an HTTP/1.1 connection
/// it means "close as soon as idle" — the server-initiated close of a
/// pooled connection that AISIX-Cloud#1394 ruled out.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Downstream {
    Http1,
    Http2,
}

/// Holds a connection's slot in [`LivezState::open_connections`] for as
/// long as the connection's task runs.
///
/// The count is what separates work the gateway had already taken on
/// from traffic still being routed at it during a drain: a pooled
/// connection sitting idle carries no in-flight request, so a count that
/// stays up while `in_flight` falls is how an operator sees that the
/// balancer has not stopped routing here yet (AISIX-Cloud#1394).
struct ConnectionGuard(std::sync::Arc<aisix_proxy::LivezState>);

impl Drop for ConnectionGuard {
    fn drop(&mut self) {
        self.0.connection_closed();
    }
}

/// Replays the bytes the version sniff had to consume, so hyper still
/// sees the connection from byte zero.
struct Rewind<I> {
    prefix: Vec<u8>,
    replayed: usize,
    inner: I,
}

impl<I> Rewind<I> {
    fn new(prefix: Vec<u8>, inner: I) -> Self {
        Self {
            prefix,
            replayed: 0,
            inner,
        }
    }
}

impl<I: tokio::io::AsyncRead + Unpin> tokio::io::AsyncRead for Rewind<I> {
    fn poll_read(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        let me = &mut *self;
        if me.replayed < me.prefix.len() && buf.remaining() > 0 {
            let n = (me.prefix.len() - me.replayed).min(buf.remaining());
            buf.put_slice(&me.prefix[me.replayed..me.replayed + n]);
            me.replayed += n;
            return std::task::Poll::Ready(Ok(()));
        }
        std::pin::Pin::new(&mut me.inner).poll_read(cx, buf)
    }
}

impl<I: tokio::io::AsyncWrite + Unpin> tokio::io::AsyncWrite for Rewind<I> {
    fn poll_write(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        std::pin::Pin::new(&mut self.inner).poll_write(cx, buf)
    }

    fn poll_flush(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::pin::Pin::new(&mut self.inner).poll_flush(cx)
    }

    fn poll_shutdown(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::pin::Pin::new(&mut self.inner).poll_shutdown(cx)
    }

    fn poll_write_vectored(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        bufs: &[std::io::IoSlice<'_>],
    ) -> std::task::Poll<std::io::Result<usize>> {
        std::pin::Pin::new(&mut self.inner).poll_write_vectored(cx, bufs)
    }

    fn is_write_vectored(&self) -> bool {
        self.inner.is_write_vectored()
    }
}

/// Read enough of a plaintext connection to tell HTTP/2 from HTTP/1.1,
/// and hand back a stream that still starts at byte zero.
///
/// Four bytes settle it: RFC 9113 §3.4 reserves the `PRI` method for the
/// client connection preface precisely so that no HTTP/1 request can
/// begin with it. `hyper_util` reads the same bytes one layer down and
/// keeps the answer to itself.
///
/// A TLS listener never reaches here — ALPN has already answered.
async fn sniff_downstream_version(
    mut stream: tokio::net::TcpStream,
    first_bytes_timeout: Option<Duration>,
) -> std::io::Result<(Downstream, Rewind<tokio::net::TcpStream>)> {
    use tokio::io::AsyncReadExt;

    const PREFACE_HEAD: &[u8] = b"PRI ";

    let mut head = [0u8; PREFACE_HEAD.len()];
    let mut filled = 0;
    let read = async {
        while filled < head.len() {
            match stream.read(&mut head[filled..]).await? {
                0 => break,
                n => filled += n,
            }
        }
        Ok::<(), std::io::Error>(())
    };

    match first_bytes_timeout {
        // The same window `downstream.idle_timeout_secs` gives a request
        // head, applied here because this read happens BEFORE hyper has
        // a connection to arm its own timer on. hyper's version read has
        // no timer of its own, so without this a peer that connects and
        // then says nothing holds its slot until the platform reclaims
        // the socket (AISIX-Cloud#1126).
        Some(d) => tokio::time::timeout(d, read).await.map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "downstream opened a connection and sent nothing",
            )
        })??,
        None => read.await?,
    }

    let version = if head[..filled] == *PREFACE_HEAD {
        Downstream::Http2
    } else {
        Downstream::Http1
    };
    Ok((version, Rewind::new(head[..filled].to_vec(), stream)))
}

/// Serve `router` on `addr`, choosing HTTPS when `tls` is configured and
/// plain HTTP otherwise. Wired for #473: `proxy.tls` / `admin.tls` were
/// parsed but never reached the listener, so the documented config
/// silently served plain HTTP.
///
/// The gateway runs its own accept loop rather than `axum::serve` or
/// `axum_server`. Two things it has to do are not reachable through
/// either: applying `downstream.idle_timeout_secs` needs the hyper
/// connection builder (AISIX-Cloud#1126), which `axum::serve` does not
/// expose; and retiring a connection WITHOUT closing the listener needs
/// the connection future (AISIX-Cloud#1395), which `axum_server` owns
/// internally and only retires together with the listener.
#[allow(clippy::too_many_arguments)]
async fn serve_http(
    addr: std::net::SocketAddr,
    router: axum::Router,
    tls: Option<aisix_core::TlsConfig>,
    idle_timeout: Option<Duration>,
    shutdown: ShutdownWatch,
    label: &'static str,
    workers: Option<usize>,
    drain: Option<std::sync::Arc<aisix_proxy::LivezState>>,
) -> anyhow::Result<()> {
    // Resolved before binding so a bad cert path still fails with the
    // same error it always did, before a port is taken.
    let tls = match tls {
        Some(tls) => Some(downstream_tls_acceptor(&tls, label).await?),
        None => None,
    };

    // Thread-per-core serving is the proxy's; the admin and metrics
    // surfaces are control traffic and keep their single listener.
    if let Some(workers) = workers {
        return serve_http_tpc(
            addr,
            router,
            tls,
            idle_timeout,
            shutdown,
            label,
            workers,
            drain,
        )
        .await;
    }

    let listener = std::net::TcpListener::bind(addr)
        .map_err(|e| anyhow::anyhow!("{label} listener bind {addr} failed: {e}"))?;
    match tls {
        None => tracing::info!(%addr, label, "aisix listening (http)"),
        Some(_) => tracing::info!(%addr, label, "aisix listening (https)"),
    }
    accept_loop(listener, router, tls, idle_timeout, shutdown, label, drain).await
}

/// Build the downstream TLS acceptor from the configured PEM files.
///
/// ALPN offers `h2` ahead of `http/1.1`, which is what the gateway has
/// always advertised — a downstream that prefers HTTP/2 has to keep
/// getting it. (#535/#536 guard the dp-manager's REST *clients* against
/// negotiating h2; this is the server side, and it does offer it.)
async fn downstream_tls_acceptor(
    tls: &aisix_core::TlsConfig,
    label: &'static str,
) -> anyhow::Result<tokio_rustls::TlsAcceptor> {
    let failed = |e: String| {
        anyhow::anyhow!(
            "{label}.tls: failed to load cert_file={:?} / key_file={:?}: {e}",
            tls.cert_file,
            tls.key_file,
        )
    };

    let cert_pem = tokio::fs::read(&tls.cert_file)
        .await
        .map_err(|e| failed(e.to_string()))?;
    let key_pem = tokio::fs::read(&tls.key_file)
        .await
        .map_err(|e| failed(e.to_string()))?;

    let certs = rustls_pemfile::certs(&mut cert_pem.as_slice())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| failed(e.to_string()))?;
    let key = rustls_pemfile::private_key(&mut key_pem.as_slice())
        .map_err(|e| failed(e.to_string()))?
        .ok_or_else(|| failed("no private key in key_file".to_string()))?;

    let mut config = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .map_err(|e| failed(e.to_string()))?;
    config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
    Ok(tokio_rustls::TlsAcceptor::from(std::sync::Arc::new(config)))
}

/// Accept and serve on `listener` until the drain ends, then wait for
/// the connections still open to finish.
///
/// The single accept path for every listener the gateway binds — both
/// serving modes, TLS and plaintext, proxy and control surfaces. The
/// per-listener differences are all parameters, so a rule proved on one
/// listener holds on all of them.
#[allow(clippy::too_many_arguments)]
async fn accept_loop(
    listener: std::net::TcpListener,
    router: axum::Router,
    tls: Option<tokio_rustls::TlsAcceptor>,
    idle_timeout: Option<Duration>,
    mut shutdown: ShutdownWatch,
    label: &'static str,
    drain: Option<std::sync::Arc<aisix_proxy::LivezState>>,
) -> anyhow::Result<()> {
    listener.set_nonblocking(true)?;
    let listener = tokio::net::TcpListener::from_std(listener)?;

    // ConnectInfo<SocketAddr> exposes the TCP peer to the proxy's real-ip
    // resolver (#492). Harmless for the admin listener, which ignores it.
    let mut make_service = router.into_make_service_with_connect_info::<std::net::SocketAddr>();

    // Close an accepted HTTP/1.1 connection that sits idle for
    // `idle_timeout`. hyper arms this timer only when it is waiting for a
    // request head, and it only waits for one once the previous response
    // has been fully written (`Conn::can_read_head` requires the read
    // half to be back at `Init`, which `try_keep_alive` reaches only when
    // reading *and* writing are done). So a slow model or a long SSE
    // stream is never interrupted — the timer covers exactly the
    // between-requests window.
    //
    // hyper defaults this to 30s but drops the default unless a timer is
    // installed, and axum does not install one — which is why the gateway
    // held idle connections forever before AISIX-Cloud#1126.
    //
    // HTTP/2 has no equivalent knob in hyper; h2 connections are
    // unaffected. `sniff_downstream_version` covers the one window that
    // precedes either builder.
    let mut http1 = hyper::server::conn::http1::Builder::new();
    if let Some(d) = idle_timeout {
        http1
            .timer(hyper_util::rt::TokioTimer::new())
            .header_read_timeout(d);
    }
    let http2 = hyper::server::conn::http2::Builder::new(hyper_util::rt::TokioExecutor::new());

    // Every connection task holds a clone of the sender; the receiver
    // below completes once the last one is dropped. Cheaper than keeping
    // join handles, and it cannot leak one for a connection that already
    // ended.
    let (alive, mut all_connections_ended) = tokio::sync::mpsc::channel::<()>(1);

    loop {
        let accepted = tokio::select! {
            biased;
            _ = shutdown.cancelled() => {
                tracing::info!(label, "shutdown signal observed — stopping listener");
                break;
            }
            accepted = listener.accept() => accepted,
        };

        let (stream, peer) = match accepted {
            Ok(accepted) => accepted,
            // Transient: fd exhaustion, or a peer that vanished between
            // the SYN and the accept. Backing off keeps a listener that
            // cannot accept from spinning a core.
            Err(e) => {
                tracing::debug!(label, error = %e, "accept failed");
                tokio::time::sleep(Duration::from_millis(50)).await;
                continue;
            }
        };

        // Nagle's algorithm holds a small segment back until the previous
        // one has been acknowledged. A short buffered answer leaves in a
        // single segment and never notices — which is why this is
        // invisible in the throughput benchmark — but a streamed one
        // writes a frame at a time, and a frame that lands while an
        // earlier one is still unacknowledged waits out the client's
        // delayed ACK (up to 40ms on Linux) before it leaves the box.
        // HTTP/2 control frames and the WebSocket surfaces run on the
        // same accepted socket and inherit the option. reqwest already
        // sets it on the upstream side.
        //
        // A failure fails that connection: on Linux the option carries no
        // socket-state precondition and cannot fail, and elsewhere it
        // reports a peer that has already gone away — a connection that
        // was not answerable regardless.
        if let Err(e) = stream.set_nodelay(true) {
            tracing::debug!(label, error = %e, "TCP_NODELAY failed; dropping connection");
            continue;
        }

        // `SocketAddr` is `Connected` both for a bare peer address and
        // for axum's own `IncomingStream`, so the target type has to be
        // named or the make-service call is ambiguous.
        std::future::poll_fn(|cx| {
            tower::Service::<std::net::SocketAddr>::poll_ready(&mut make_service, cx)
        })
        .await
        .map_err(|e| anyhow::anyhow!("{label} make_service not ready: {e}"))?;
        let service =
            match tower::Service::<std::net::SocketAddr>::call(&mut make_service, peer).await {
                Ok(service) => service,
                Err(never) => match never {},
            };

        // Accepting is the only place either fact is observable, and
        // during a drain they are what separates work the gateway had
        // already taken on from traffic still being routed at it
        // (AISIX-Cloud#1394).
        //
        // Neither can tell a client apart from the platform, though:
        // `/livez` and `/readyz` are served on the proxy listener too,
        // and the probes keep arriving throughout the drain, each on its
        // own connection. So the count includes them and the accept line
        // does not claim to know who connected — the arrival line in
        // `record_request_telemetry` knows the path and is where that
        // judgement is made. This line covers what the arrival line
        // cannot: a connection opened and never used.
        //
        // `drain` is `None` for the admin and metrics listeners, which
        // are control surfaces nothing routes client traffic to.
        let counted = drain.as_ref().map(|drain| {
            drain.connection_opened();
            if drain.is_shutting_down() {
                tracing::info!(
                    peer = %peer,
                    open_connections = drain.open_connections(),
                    "accepted a new downstream connection while draining"
                );
            }
            ConnectionGuard(drain.clone())
        });

        let alive = alive.clone();
        let shutdown = shutdown.clone();
        let http1 = http1.clone();
        let http2 = http2.clone();
        let tls = tls.clone();
        tokio::spawn(async move {
            let _alive = alive;
            let _counted = counted;
            // The TLS handshake and the version read both wait on the
            // peer, so they belong to the connection's own task: a client
            // that stalls in either must not hold up the accept loop.
            match tls {
                Some(tls) => match tls.accept(stream).await {
                    Ok(stream) => {
                        let version = match stream.get_ref().1.alpn_protocol() {
                            Some(proto) if proto == b"h2" => Downstream::Http2,
                            _ => Downstream::Http1,
                        };
                        serve_connection(stream, version, service, &http1, &http2, shutdown, label)
                            .await;
                    }
                    Err(e) => tracing::debug!(label, error = %e, "TLS handshake failed"),
                },
                None => match sniff_downstream_version(stream, idle_timeout).await {
                    Ok((version, stream)) => {
                        serve_connection(stream, version, service, &http1, &http2, shutdown, label)
                            .await;
                    }
                    Err(e) => {
                        tracing::debug!(label, error = %e, "reading the request preface failed")
                    }
                },
            }
        });
    }

    // Refuse connections immediately rather than letting the ones the
    // kernel already queued hang: tokio accepts into that queue whether
    // or not this loop is still reading from it.
    drop(listener);

    // The drain has already run every in-flight request to completion, so
    // what is left here is idle pooled connections; each one's task ends
    // as soon as `serve_connection` hands it the cancel signal.
    drop(alive);
    let _ = all_connections_ended.recv().await;
    Ok(())
}

/// Serve one accepted connection to completion, giving it the retirement
/// signal its protocol can actually carry.
///
/// The two versions are dispatched here rather than left to
/// `hyper_util`'s auto builder because they retire at different moments,
/// and the difference is a deliberate one:
///
/// * **HTTP/2 is retired when the drain STARTS**, with GOAWAY. hyper
///   sends RFC 9113 §6.8's two-phase form — an advisory
///   `GOAWAY(2^31-1)`, a PING, then a second GOAWAY naming the last
///   stream it actually processed. The peer finishes the streams it had
///   already dispatched and opens no new ones, and nothing closes until
///   they are done, so there is no race with a request in flight. It is
///   the only in-band retirement signal h2 has: RFC 9113 §8.2.2 forbids
///   `Connection: close`, and before this the gateway had no way to send
///   one during a drain (AISIX-Cloud#1395).
/// * **HTTP/1.1 is retired in band**, by `Connection: close` on the
///   responses served during the drain
///   (`aisix_proxy::retire_connection`), and hears nothing here until
///   the drain ENDS. hyper retires an h1 connection by disabling
///   keep-alive, which closes an IDLE one at once — a server-initiated
///   close of a pooled connection, where a client dispatching onto it in
///   that same instant loses the request. AISIX-Cloud#1394 ruled that
///   out: a long connection is the client's to close, and the server's
///   job is only to say so.
async fn serve_connection<I, S>(
    io: I,
    version: Downstream,
    service: S,
    http1: &hyper::server::conn::http1::Builder,
    http2: &hyper::server::conn::http2::Builder<hyper_util::rt::TokioExecutor>,
    mut shutdown: ShutdownWatch,
    label: &'static str,
) where
    I: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + Send + 'static,
    S: tower::Service<
            axum::http::Request<hyper::body::Incoming>,
            Response = axum::http::Response<axum::body::Body>,
            Error = std::convert::Infallible,
        > + Clone
        + Send
        + 'static,
    S::Future: Send,
{
    let io = hyper_util::rt::TokioIo::new(io);
    let service = hyper_util::service::TowerToHyperService::new(service);

    let served = match version {
        Downstream::Http1 => {
            // `with_upgrades` keeps the WebSocket surfaces (`/v1/realtime`)
            // working: hyper cannot upgrade a connection it does not own.
            let conn = http1.serve_connection(io, service).with_upgrades();
            tokio::pin!(conn);
            let mut retired = false;
            loop {
                tokio::select! {
                    served = conn.as_mut() => break served.map_err(|e| e.to_string()),
                    _ = shutdown.cancelled(), if !retired => {
                        conn.as_mut().graceful_shutdown();
                        retired = true;
                    }
                }
            }
        }
        Downstream::Http2 => {
            // Needs h2 >= 0.4.14. A peer that answers our GOAWAY with one
            // of its own — Node's HTTP/2 client does, on any graceful
            // GOAWAY — sends `last_stream_id: 0`, because a client's
            // last-stream-id speaks only for server-PUSHED streams.
            // Before hyperium/h2#886 the server applied it to every
            // stream in the store, so the reciprocal frame reset each
            // request still running on the connection: the drain would
            // kill exactly the in-flight work it exists to protect.
            // Pinned by the `graceful-drain-h2` e2e spec, which fails
            // against 0.4.13.
            let conn = http2.serve_connection(io, service);
            tokio::pin!(conn);
            let mut retired = false;
            loop {
                tokio::select! {
                    served = conn.as_mut() => break served.map_err(|e| e.to_string()),
                    _ = shutdown.retire_or_cancel(), if !retired => {
                        conn.as_mut().graceful_shutdown();
                        retired = true;
                    }
                }
            }
        }
    };

    if let Err(e) = served {
        // A downstream that hangs up mid-request lands here, which is
        // ordinary traffic rather than a fault.
        tracing::debug!(label, error = %e, "downstream connection ended");
    }
}

/// Serve from `workers` independent threads, each with its own runtime
/// and its own `SO_REUSEPORT` listener on the same address.
///
/// A connection is accepted, read, dispatched, and answered on one
/// thread, and the upstream call it makes runs on that thread's own
/// connection pool (see `upstream_tls::mark_worker_thread`). That is the
/// point of the mode: the shared runtime hands a request between threads
/// roughly once per request — first when a worker steals the task, again
/// when the upstream response lands on whichever thread happens to own
/// that connection — and each handoff costs a wakeup and a context
/// switch. Here there are none.
///
/// The kernel decides which listener gets each connection, hashing the
/// 4-tuple. That spreads evenly across many client connections and
/// unevenly across few, which is why the mode is documented as a
/// throughput setting rather than a latency one.
#[allow(clippy::too_many_arguments)]
async fn serve_http_tpc(
    addr: std::net::SocketAddr,
    router: axum::Router,
    tls: Option<tokio_rustls::TlsAcceptor>,
    idle_timeout: Option<Duration>,
    shutdown: ShutdownWatch,
    label: &'static str,
    workers: usize,
    drain: Option<std::sync::Arc<aisix_proxy::LivezState>>,
) -> anyhow::Result<()> {
    // An address someone else already holds has to stay a loud boot
    // failure. Every socket on a `SO_REUSEPORT` address has to set the
    // option, so a socket that does not set it fails to bind exactly
    // when something else is there — which is the check the worker
    // sockets below deliberately give up, since they must co-bind with
    // each other. Without this a second gateway would start silently
    // and split traffic with the first.
    drop(
        std::net::TcpListener::bind(addr)
            .map_err(|e| anyhow::anyhow!("{label} listener bind {addr} failed: {e}"))?,
    );

    // Every listener is bound before any worker spawns, so a bind
    // failure — fd exhaustion on the last socket included — aborts
    // startup before a single connection is accepted, exactly as the
    // single listener does.
    let mut listeners = Vec::with_capacity(workers);
    for _ in 0..workers {
        let listener = bind_reuseport_listener(addr)
            .map_err(|e| anyhow::anyhow!("{label} listener bind {addr} failed: {e}"))?;
        listeners.push(listener);
    }

    // One slot per worker so no worker blocks reporting its exit.
    let (exit_tx, exit_rx) = std::sync::mpsc::sync_channel::<anyhow::Result<()>>(workers);
    for (worker, listener) in listeners.into_iter().enumerate() {
        let router = router.clone();
        let shutdown = shutdown.clone();
        let tls = tls.clone();
        let exit_tx = exit_tx.clone();
        // Every worker raises and lowers the same process-wide count, so
        // the drain heartbeat reports the listener as a whole rather than
        // whichever worker happened to accept.
        let drain = drain.clone();
        std::thread::Builder::new()
            // Names the mode in `ps -T` / `top -H`: `tpc-N` here,
            // tokio's own `tokio-rt-worker` on the shared runtime.
            .name(format!("tpc-{worker}"))
            .spawn(move || {
                let mut exit = WorkerExit {
                    tx: exit_tx,
                    worker,
                    outcome: None,
                };
                exit.outcome = Some(run_tpc_worker(
                    addr,
                    listener,
                    router,
                    tls,
                    idle_timeout,
                    shutdown,
                    label,
                    worker,
                    drain,
                ));
            })?;
    }
    // Only the workers hold senders from here, so a `RecvError` means
    // every worker is gone.
    drop(exit_tx);

    // A graceful shutdown ends the workers together, but each drains its
    // own connections on its own clock — an idle worker returns at once
    // while a sibling may hold an in-flight stream for minutes. Wait for
    // every worker, so the fastest drain cannot end the process under
    // the slowest. Everything else stays immediately fatal: an accept
    // loop failing, a panic unwinding, or a worker stopping without a
    // shutdown signal has to bring the process down rather than leave it
    // serving on fewer listeners than it reported binding.
    let shutdown_seen = shutdown.cancel;
    tokio::task::spawn_blocking(move || {
        for _ in 0..workers {
            match exit_rx.recv() {
                Ok(Ok(())) if *shutdown_seen.borrow() => {}
                Ok(Ok(())) => {
                    return Err(anyhow::anyhow!(
                        "a proxy worker stopped serving without a shutdown signal"
                    ));
                }
                Ok(Err(e)) => return Err(e),
                Err(_) => {
                    return Err(anyhow::anyhow!("a proxy worker exited without reporting"));
                }
            }
        }
        Ok(())
    })
    .await?
}

/// Reports a worker's exit exactly once, including when a panic unwinds
/// out of it and there is no return value to send.
struct WorkerExit {
    tx: std::sync::mpsc::SyncSender<anyhow::Result<()>>,
    worker: usize,
    outcome: Option<anyhow::Result<()>>,
}

impl Drop for WorkerExit {
    fn drop(&mut self) {
        let outcome = self.outcome.take().unwrap_or_else(|| {
            Err(anyhow::anyhow!(
                "proxy worker {} panicked; see the panic above",
                self.worker
            ))
        });
        let _ = self.tx.send(outcome);
    }
}

/// One thread-per-core worker: its own current-thread runtime, its own
/// listener, its own upstream connection pool.
#[allow(clippy::too_many_arguments)]
fn run_tpc_worker(
    addr: std::net::SocketAddr,
    listener: std::net::TcpListener,
    router: axum::Router,
    tls: Option<tokio_rustls::TlsAcceptor>,
    idle_timeout: Option<Duration>,
    shutdown: ShutdownWatch,
    label: &'static str,
    worker: usize,
    drain: Option<std::sync::Arc<aisix_proxy::LivezState>>,
) -> anyhow::Result<()> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    // Every dispatch from this thread now uses this thread's pool, so an
    // upstream response is read by the same runtime that is waiting for
    // it. Marked inside the worker because the marker is per thread.
    aisix_gateway::upstream_tls::mark_worker_thread();
    rt.block_on(async move {
        match tls {
            None => {
                tracing::info!(%addr, label, worker, "aisix listening (http, thread-per-core)")
            }
            Some(_) => {
                tracing::info!(%addr, label, worker, "aisix listening (https, thread-per-core)")
            }
        }
        accept_loop(listener, router, tls, idle_timeout, shutdown, label, drain).await
    })
}

/// A listener that shares `addr` with the other workers' listeners.
///
/// `SO_REUSEPORT` has to be set before the bind, and every socket on the
/// address has to set it, which is why this cannot go through
/// `TcpListener::bind`.
fn bind_reuseport_listener(addr: std::net::SocketAddr) -> std::io::Result<std::net::TcpListener> {
    #[cfg(not(unix))]
    return Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "thread-per-core serving needs SO_REUSEPORT, which this platform \
         does not have; set proxy.thread_per_core: false",
    ));

    #[cfg(unix)]
    {
        let domain = if addr.is_ipv4() {
            socket2::Domain::IPV4
        } else {
            socket2::Domain::IPV6
        };
        let socket =
            socket2::Socket::new(domain, socket2::Type::STREAM, Some(socket2::Protocol::TCP))?;
        socket.set_reuse_address(true)?;
        socket.set_reuse_port(true)?;
        socket.bind(&addr.into())?;
        // Matches the backlog `std::net::TcpListener::bind` requests, so
        // the two modes queue the same number of pending connections per
        // socket.
        socket.listen(128)?;
        Ok(socket.into())
    }
}

/// How often the drain loop re-reads the in-flight count once the
/// minimum window has elapsed.
const DRAIN_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// How often the drain loop reports that it is still waiting, so a drain
/// that never finishes is visible in the logs rather than looking like a
/// hung process.
const DRAIN_LOG_INTERVAL: Duration = Duration::from_secs(5);

async fn wait_for_signal(
    cancel_tx: watch::Sender<bool>,
    retire_tx: watch::Sender<bool>,
    livez_state: std::sync::Arc<aisix_proxy::LivezState>,
    min_drain: Duration,
) {
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

    // `/readyz` answers 503 from here on. Everything below decides when
    // it is safe to stop accepting, which is deliberately NOT the same
    // moment: a balancer only learns about the 503 on its next health
    // check, and closing the listener before then refuses every
    // connection it routes in between.
    livez_state.mark_shutting_down();
    // Retirement rides the same moment. An HTTP/1.1 connection is
    // retired in band, on the next response it carries; an HTTP/2 one
    // has no such header (RFC 9113 §8.2.2) and gets its GOAWAY here,
    // which asks the peer to finish the streams it has and open no new
    // ones without closing anything (AISIX-Cloud#1395).
    let _ = retire_tx.send(true);
    tracing::info!(
        min_drain_secs = min_drain.as_secs(),
        in_flight = livez_state.in_flight(),
        open_connections = livez_state.open_connections(),
        "draining — /readyz now reports 503, still accepting new connections"
    );

    if !min_drain.is_zero() {
        tokio::time::sleep(min_drain).await;
    }

    // The window is a minimum, not a deadline. A balancer slower than
    // configured is still routing traffic here, and that traffic is
    // exactly what the in-flight count shows — so keep serving until it
    // reaches zero, at which point closing the listener cannot interrupt
    // anything. Unbounded on purpose: an inference call or an SSE stream
    // may run for minutes, and the platform (Kubernetes
    // `terminationGracePeriodSeconds`, systemd `TimeoutStopSec`) is the
    // one hard bound.
    let mut last_log = std::time::Instant::now();
    loop {
        let in_flight = livez_state.in_flight();
        if in_flight == 0 {
            break;
        }
        if last_log.elapsed() >= DRAIN_LOG_INTERVAL {
            // The open-connection count next to it separates the two
            // reasons a drain does not end: requests this gateway already
            // took on, versus a load balancer still routing here — only
            // the second of which is a balancer problem
            // (AISIX-Cloud#1394).
            tracing::info!(
                in_flight,
                open_connections = livez_state.open_connections(),
                "still draining in-flight requests"
            );
            last_log = std::time::Instant::now();
        }
        tokio::time::sleep(DRAIN_POLL_INTERVAL).await;
    }

    tracing::info!("drain complete — closing listeners");
    let _ = cancel_tx.send(true);
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    // The shipped-target contract: the runtime mallctl enable must actually
    // take effect here — an Ok(false) read-back would mean the #968 fix
    // silently does nothing. Drives the delivered function, not an inline
    // re-implementation of the mallctl pair; the test binary links the same
    // #[global_allocator] as the shipped one.
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    #[test]
    fn jemalloc_background_thread_enables_at_runtime() {
        assert!(
            matches!(enable_jemalloc_background_thread(), Ok(true)),
            "background_thread did not enable on a linux-gnu target"
        );
    }

    fn snapshot_with_key(
        id: &str,
        team: Option<&str>,
        user: Option<&str>,
        name: Option<&str>,
    ) -> AisixSnapshot {
        let key: aisix_core::ApiKey = serde_json::from_value(serde_json::json!({
            "key_hash": "h",
            "allowed_models": [],
            "team_id": team,
            "user_id": user,
            "user_name": name,
        }))
        .unwrap();
        let snap = AisixSnapshot::new();
        snap.apikeys
            .insert(aisix_core::resource::ResourceEntry::new(id, key, 1));
        snap
    }

    /// The case the sweep exists for: the api key is gone, so its budget
    /// label set describes nothing and must be reported dead.
    #[test]
    fn a_deleted_key_is_not_live() {
        let snap = AisixSnapshot::new();
        assert!(!gauge_series_is_live(
            &snap,
            aisix_obs::LiveGaugeSeries::Budget {
                api_key_id: "ak-gone",
                team_id: "t",
                user_id: "u",
                user_name: "alice",
            }
        ));
    }

    /// A bare "does the key exist" check would call BOTH label sets live
    /// after a rebind and leave the pre-rebind sample frozen forever. The
    /// whole member triple has to match.
    #[test]
    fn a_rebound_key_leaves_only_its_current_label_set_live() {
        let snap = snapshot_with_key("ak-1", Some("team-new"), Some("u"), Some("alice"));
        let at = |team, user, name| aisix_obs::LiveGaugeSeries::Budget {
            api_key_id: "ak-1",
            team_id: team,
            user_id: user,
            user_name: name,
        };
        assert!(gauge_series_is_live(&snap, at("team-new", "u", "alice")));
        assert!(!gauge_series_is_live(&snap, at("team-old", "u", "alice")));
        assert!(!gauge_series_is_live(
            &snap,
            at("team-new", "u-old", "alice")
        ));
        // A rename strands the old name under the same id, same as a rebind.
        assert!(!gauge_series_is_live(
            &snap,
            at("team-new", "u", "Alice Before")
        ));
    }

    /// An unbound key projects `unknown` for the member triple; that IS its
    /// current label set, so it must stay live.
    #[test]
    fn an_unbound_key_is_live_under_its_placeholders() {
        let snap = snapshot_with_key("ak-1", None, None, None);
        assert!(gauge_series_is_live(
            &snap,
            aisix_obs::LiveGaugeSeries::Budget {
                api_key_id: "ak-1",
                team_id: "unknown",
                user_id: "unknown",
                user_name: "unknown",
            }
        ));
    }

    /// The rate-limit family is judged on BOTH halves of its key, which is
    /// only safe because the emit site collapses the caller's model string
    /// to the configured set first. A wildcard alias serves concrete names
    /// that are in no `models` row, so if the raw string still reached the
    /// label this check would retire live series on every sweep.
    #[test]
    fn rate_limit_remaining_is_judged_on_both_halves_of_its_key() {
        let snap = snapshot_with_key("ak-1", None, None, None);
        let model: aisix_core::Model = serde_json::from_value(serde_json::json!({
            "display_name": "openai/*",
            "provider": "openai",
            "model_name": "gpt-4o-mini",
        }))
        .unwrap();
        snap.models
            .insert(aisix_core::resource::ResourceEntry::new("m-1", model, 1));
        let at = |key, model| aisix_obs::LiveGaugeSeries::RatelimitRemaining {
            api_key_id: key,
            model,
        };
        // The wildcard ROW name is what the emit site stamps for every
        // concrete name that row serves, and it resolves.
        assert!(gauge_series_is_live(&snap, at("ak-1", "openai/*")));
        // Placeholders name no row, so they are never retired.
        assert!(gauge_series_is_live(&snap, at("ak-1", "unresolved")));
        assert!(gauge_series_is_live(&snap, at("ak-1", "unknown")));
        // Either half going away retires the series.
        assert!(!gauge_series_is_live(&snap, at("ak-gone", "openai/*")));
        assert!(!gauge_series_is_live(&snap, at("ak-1", "deleted-model")));
    }

    #[tokio::test(start_paused = true)]
    async fn a_periodic_job_runs_on_its_period_and_stops_on_cancel() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let (cancel_tx, cancel_rx) = watch::channel(false);
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = calls.clone();
        let task = tokio::spawn(run_periodic(cancel_rx, Duration::from_secs(5), move || {
            observed.fetch_add(1, Ordering::SeqCst);
        }));

        tokio::task::yield_now().await;
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        tokio::time::advance(Duration::from_secs(5)).await;
        tokio::task::yield_now().await;
        assert_eq!(calls.load(Ordering::SeqCst), 2);

        cancel_tx.send(true).unwrap();
        task.await.unwrap();
        tokio::time::advance(Duration::from_secs(10)).await;
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn supplied_certs_take_precedence_over_persisted_bundle() {
        // The #265 fix: when env/file vars supply a fresh bundle it must
        // win even if a (possibly stale) bundle is already on disk —
        // otherwise a rotated CP CA leaves the DP pinned to the old one.
        assert_eq!(
            select_managed_boot_path(true, true),
            ManagedBootPath::ProvisionFromEnv,
        );
        // Supplied-only (first boot): provision.
        assert_eq!(
            select_managed_boot_path(false, true),
            ManagedBootPath::ProvisionFromEnv,
        );
        // Persisted-only (no env): reuse the disk bundle.
        assert_eq!(
            select_managed_boot_path(true, false),
            ManagedBootPath::ReusePersisted,
        );
        // Neither: cannot boot.
        assert_eq!(
            select_managed_boot_path(false, false),
            ManagedBootPath::MissingBundle,
        );
    }

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
        assert_eq!(a.config, Some(PathBuf::from("/tmp/x.yaml")));
        assert!(a.command.is_none());
    }

    #[test]
    fn cli_validate_subcommand_does_not_require_config() {
        // `aisix validate --resources f` runs without --config …
        let cli = Cli::try_parse_from(["aisix", "validate", "--resources", "/tmp/r.yaml"]).unwrap();
        assert!(cli.config.is_none());
        match cli.command {
            Some(CliCommand::Validate { resources }) => {
                assert_eq!(resources, PathBuf::from("/tmp/r.yaml"));
            }
            other => panic!("expected Validate subcommand, got {other:?}"),
        }
        // … and --resources itself is mandatory for the subcommand.
        assert!(Cli::try_parse_from(["aisix", "validate"]).is_err());
    }

    #[test]
    fn run_validate_accepts_a_valid_resources_file() {
        use std::io::Write as _;
        let mut f = tempfile::Builder::new().suffix(".yaml").tempfile().unwrap();
        f.write_all(
            br#"
_format_version: "1"
provider_keys:
  - display_name: pk
    api_key: sk-test
models:
  - display_name: m1
    provider: openai
    model_name: gpt-4o
    provider_key: pk
"#,
        )
        .unwrap();
        // Success path returns Ok; the failure path exits the process,
        // which is covered end-to-end by the e2e fail-fast case.
        run_validate(f.path()).unwrap();
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

    fn managed_with_urls(
        base_url: Option<&str>,
        etcd_endpoint: Option<&str>,
    ) -> aisix_core::ManagedConfig {
        aisix_core::ManagedConfig {
            enabled: true,
            cp_base_url: base_url.map(String::from),
            cp_etcd_endpoint: etcd_endpoint.map(String::from),
            ..Default::default()
        }
    }

    #[test]
    fn derive_etcd_url_from_base_url_strips_scheme() {
        let m = managed_with_urls(Some("https://dpm.example.com:7944"), None);
        assert_eq!(
            derive_cp_etcd_url(&m).unwrap(),
            "https://dpm.example.com:7944"
        );
    }

    #[test]
    fn derive_etcd_url_prefers_explicit_endpoint() {
        let m = managed_with_urls(
            Some("https://dpm.example.com:7944"),
            Some("etcd.internal:2379"),
        );
        assert_eq!(
            derive_cp_etcd_url(&m).unwrap(),
            "https://etcd.internal:2379"
        );
    }

    #[test]
    fn derive_etcd_url_explicit_endpoint_without_base_url() {
        let m = managed_with_urls(None, Some("etcd.internal:2379"));
        assert_eq!(
            derive_cp_etcd_url(&m).unwrap(),
            "https://etcd.internal:2379"
        );
    }

    #[test]
    fn derive_etcd_url_strips_http_scheme() {
        let m = managed_with_urls(Some("http://localhost:7944"), None);
        assert_eq!(derive_cp_etcd_url(&m).unwrap(), "https://localhost:7944");
    }

    #[test]
    fn derive_etcd_url_strips_trailing_slash() {
        let m = managed_with_urls(Some("https://dpm.example.com:7944/"), None);
        assert_eq!(
            derive_cp_etcd_url(&m).unwrap(),
            "https://dpm.example.com:7944"
        );
    }

    #[test]
    fn derive_etcd_url_errors_without_base_url() {
        let m = managed_with_urls(None, None);
        let err = derive_cp_etcd_url(&m).unwrap_err();
        assert!(err.to_string().contains("cp_base_url"), "unexpected: {err}");
    }

    #[test]
    fn derive_etcd_url_errors_on_empty_base_url() {
        let m = managed_with_urls(Some(""), None);
        let err = derive_cp_etcd_url(&m).unwrap_err();
        assert!(err.to_string().contains("cp_base_url"), "unexpected: {err}");
    }

    /// The `upstream_protocol` metric label (AISIX-Cloud#1403) MUST
    /// agree with the bridge `build_hub()` actually dispatches to.
    ///
    /// `aisix_gateway::upstream_protocol` is a pure function of the
    /// ProviderKey row — it has to be, because the metric emit path
    /// resolves the row once per request and has no Hub in hand. That
    /// makes it a SECOND copy of `Hub::dispatch_two_tier`'s rule, and
    /// this test is what keeps the copy honest: it sweeps the REGISTERED
    /// specialized vendors (so a third one cannot be added without being
    /// covered) crossed with every `Adapter` variant plus the absent
    /// adapter, and asserts the label equals the dispatched bridge's own
    /// `wire_protocol()`.
    ///
    /// A mismatch is not a cosmetic label bug: it puts one ProviderKey's
    /// traffic under a protocol it never spoke, which is exactly the
    /// misattribution the label was added to remove.
    #[test]
    fn upstream_protocol_label_matches_dispatched_bridge() {
        let hub = build_hub();
        let mut vendors = hub.specialized_vendors();
        // A vendor with no specialized registration, to exercise the
        // family tier the long-tail catalog resolves through.
        vendors.push("some-long-tail-vendor".to_string());

        let adapters: Vec<Option<aisix_core::Adapter>> = std::iter::once(None)
            .chain(aisix_core::Adapter::ALL.into_iter().map(Some))
            .collect();

        for vendor in &vendors {
            for adapter in &adapters {
                let adapter_field = match adapter {
                    Some(a) => format!(r#","adapter":"{}""#, a.wire_protocol()),
                    None => String::new(),
                };
                let pk: aisix_core::ProviderKey = serde_json::from_str(&format!(
                    r#"{{"display_name":"pk","secret":"k","provider":"{vendor}"{adapter_field}}}"#
                ))
                .unwrap();
                let dispatched = hub
                    .dispatch_two_tier(&pk)
                    .map(|b| b.wire_protocol())
                    .unwrap_or(aisix_gateway::UPSTREAM_PROTOCOL_UNKNOWN);
                assert_eq!(
                    aisix_gateway::upstream_protocol(&pk),
                    dispatched,
                    "provider={vendor:?} adapter={adapter:?}: the upstream_protocol \
                     label disagrees with the bridge dispatch selects",
                );
            }
        }
    }

    /// A vendor with no specialized bridge and no `adapter` reaches no
    /// bridge at all, so its protocol is genuinely unknown rather than a
    /// guess at the most common wire shape. Pinned separately because
    /// the sweep above would still pass if BOTH sides defaulted to
    /// `"openai"`.
    #[test]
    fn upstream_protocol_is_unknown_without_a_resolvable_bridge() {
        let pk: aisix_core::ProviderKey = serde_json::from_str(
            r#"{"display_name":"pk","secret":"k","provider":"some-long-tail-vendor"}"#,
        )
        .unwrap();
        assert_eq!(
            aisix_gateway::upstream_protocol(&pk),
            "unknown",
            "an unresolvable ProviderKey must not be labelled with a guessed protocol",
        );
        assert!(build_hub().dispatch_two_tier(&pk).is_none());
    }

    /// The canonical `provider: "openai"` / `provider: "anthropic"` keys
    /// cp-api writes reach a SPECIALIZED bridge, so their protocol comes
    /// from the vendor tier — reading `pk.adapter` alone would report
    /// `unknown` for a key that omits it.
    #[test]
    fn upstream_protocol_reads_the_specialized_tier_before_the_adapter() {
        for (vendor, want) in [("openai", "openai"), ("anthropic", "anthropic")] {
            let pk: aisix_core::ProviderKey = serde_json::from_str(&format!(
                r#"{{"display_name":"pk","secret":"k","provider":"{vendor}"}}"#
            ))
            .unwrap();
            assert!(pk.adapter.is_none());
            assert_eq!(aisix_gateway::upstream_protocol(&pk), want);
        }
    }

    /// `build_hub()` must NOT register `cohere` as a specialized chat
    /// bridge. Post-#302 Phase A, Cohere's chat surface is served by
    /// the `Adapter::Openai` family bridge: cp-api stores Cohere's PK
    /// with `adapter: "openai"` and `api_base: "https://api.cohere.com/compatibility/v1"`
    /// (per <https://docs.cohere.com/reference/chat>). A specialized
    /// chat bridge here would re-introduce the vendor-enumeration
    /// pattern the clean cut deleted.
    #[test]
    fn build_hub_does_not_register_cohere_as_specialized_chat_bridge() {
        let hub = build_hub();
        assert!(
            hub.get_specialized("cohere").is_none(),
            "cohere chat must fall through to `Adapter::Openai` family — \
             a specialized chat registration re-introduces the deleted vendor-enumeration pattern",
        );
    }

    /// `build_hub()` must NOT register `jina` as a specialized chat
    /// bridge. Jina is rerank-only (#213 Phase 2) — its
    /// `/v1/chat/completions` traffic falls through to the family
    /// bridge `Adapter::Openai`, which is fine because the chat
    /// envelope is OpenAI-shaped if cp-api populates `adapter`.
    /// Registering a specialized Jina chat bridge here would
    /// silently change the metric label / behavior on a future
    /// `provider: "jina"` chat request.
    #[test]
    fn build_hub_does_not_register_jina_for_chat() {
        let hub = build_hub();
        assert!(
            hub.get_specialized("jina").is_none(),
            "jina is rerank-only (#213 Phase 2); a specialized chat bridge here would \
             change the metric label silently on the first jina chat request",
        );
    }

    /// `build_hub()` MUST register `Adapter::Openai` as a family
    /// bridge so any catalog vendor admitted by cp-api with
    /// `adapter: "openai"` (xai, openrouter, groq, mistral, etc. —
    /// every models.dev long-tail) resolves through the family
    /// fallthrough. Without it, dispatch returns None and the
    /// customer sees a 503. Closes the dispatch half of
    /// api7/AISIX-Cloud#417.
    #[test]
    fn build_hub_registers_openai_family_bridge_for_long_tail_catalog_vendors() {
        let hub = build_hub();
        // Synthesize a PK for a vendor that's NOT in the specialized
        // registrations above (e.g. xai). It must resolve via the
        // family bridge.
        let pk: aisix_core::ProviderKey = serde_json::from_str(
            r#"{"display_name":"xai-pk","secret":"sk-test","provider":"xai","adapter":"openai","api_base":"https://api.x.ai/v1"}"#,
        )
        .unwrap();
        let bridge = hub.dispatch_two_tier(&pk).unwrap_or_else(|| {
            panic!(
                "Adapter::Openai family bridge must be registered so any catalog \
                 vendor admitted by cp-api with `adapter: \"openai\"` resolves \
                 through the family fallthrough — a missing family bridge \
                 re-introduces api7/AISIX-Cloud#417"
            )
        });
        assert_eq!(
            bridge.name(),
            "openai",
            "OpenAI family bridge MUST be the bare `OpenAiBridge::new()` so it \
             dispatches through `ProviderKey.api_base` for any vendor",
        );
    }

    /// `build_hub()` MUST register `Adapter::Anthropic` as a family
    /// bridge for symmetry with `Adapter::Openai`. The Anthropic
    /// family bridge serves any Anthropic-compat vendor cp-api admits.
    #[test]
    fn build_hub_registers_anthropic_family_bridge() {
        let hub = build_hub();
        // Tighten: assert the dispatch comes from the family tier,
        // not from an accidentally-registered specialized bridge.
        // The bare vendor string `"some-anthropic-compat"` is not in
        // the specialized list, so `dispatch_two_tier` must fall
        // through to the `Adapter::Anthropic` family registration.
        assert!(
            hub.get_specialized("some-anthropic-compat").is_none(),
            "`some-anthropic-compat` must not be specialized; the test must exercise the family tier"
        );
        let pk: aisix_core::ProviderKey = serde_json::from_str(
            r#"{"display_name":"anth-compat-pk","secret":"sk-test","provider":"some-anthropic-compat","adapter":"anthropic","api_base":"https://example.com"}"#,
        )
        .unwrap();
        let bridge = hub
            .dispatch_two_tier(&pk)
            .unwrap_or_else(|| panic!("Adapter::Anthropic family bridge must be registered"));
        assert_eq!(
            bridge.name(),
            "anthropic",
            "family Anthropic bridge MUST be the bare `AnthropicBridge::new()`",
        );
    }

    /// `build_hub()` MUST register the specialized `openai` vendor so a
    /// ProviderKey with `provider: "openai"` dispatches to the dedicated
    /// `OpenAiBridge`. This pins the registration end-to-end against the
    /// real `build_hub()` registry (not a stub Hub), so it fails the
    /// moment the registration disappears.
    #[test]
    fn build_hub_registers_specialized_openai_vendor() {
        let hub = build_hub();
        let bridge = hub
            .get_specialized("openai")
            .expect("openai vendor must be registered as specialized");
        assert_eq!(
            bridge.name(),
            "openai",
            "specialized 'openai' MUST be `OpenAiBridge::new()` (bridge name 'openai')",
        );
    }

    /// Parallel of the openai specialized-registration test, for the
    /// Anthropic side.
    #[test]
    fn build_hub_registers_specialized_anthropic_vendor() {
        let hub = build_hub();
        let bridge = hub
            .get_specialized("anthropic")
            .expect("anthropic vendor must be registered as specialized");
        assert_eq!(
            bridge.name(),
            "anthropic",
            "specialized 'anthropic' MUST be `AnthropicBridge::new()` (bridge name 'anthropic')",
        );
    }

    /// One accept path, and it has to keep setting `TCP_NODELAY`.
    ///
    /// This stays a source probe because the option is invisible from
    /// outside: a listener wired without it serves correctly, benchmarks
    /// identically, and only pays the delayed-ACK tax on streamed
    /// responses. One accept path is what makes that cheap to guard, so
    /// the invariant is that there is still only one — a second would be
    /// a second place to forget the option, the connection count and the
    /// version sniff.
    #[test]
    fn the_gateway_has_one_accept_path_and_it_sets_tcp_nodelay() {
        let src = include_str!("main.rs");
        let production = src
            .split("\n#[cfg(test)]\nmod ")
            .next()
            .expect("production half");

        let sites: Vec<usize> = production
            .lines()
            .enumerate()
            .filter(|(_, line)| line.contains("listener.accept()"))
            .map(|(n, _)| n)
            .collect();
        assert_eq!(
            sites.len(),
            1,
            "expected exactly one accept site, found {}. A second one is a \
             second place TCP_NODELAY, the connection count and the version \
             sniff can be forgotten; fold it into `accept_loop` instead",
            sites.len(),
        );

        let armed: Vec<usize> = production
            .lines()
            .enumerate()
            .filter(|(_, line)| line.contains("set_nodelay(true)"))
            .map(|(n, _)| n)
            .collect();
        assert_eq!(
            armed.len(),
            1,
            "expected exactly one TCP_NODELAY site, found {}",
            armed.len(),
        );
        assert!(
            sites[0] < armed[0],
            "TCP_NODELAY must be set on the socket this accept produced",
        );

        // And on the accept path itself, not inside the connection task:
        // a socket handed off before the option is set would serve a
        // request or two with Nagle still on.
        let lines: Vec<&str> = production.lines().collect();
        let between = lines[sites[0]..armed[0]].join(" ");
        assert!(
            !between.contains("tokio::spawn("),
            "TCP_NODELAY must be set before the connection is handed to a task",
        );
    }

    /// The preface decides, and the bytes it consumed have to come back.
    /// A sniff that swallowed them would leave hyper reading an HTTP/2
    /// connection that starts four bytes in.
    #[tokio::test]
    async fn sniffing_recognises_the_http2_preface_and_replays_it() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        const PREFACE: &[u8] = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("local addr");
        let mut client = tokio::net::TcpStream::connect(addr).await.expect("connect");
        client.write_all(PREFACE).await.expect("write preface");
        let (accepted, _peer) = listener.accept().await.expect("accept");

        let (version, mut stream) = sniff_downstream_version(accepted, None)
            .await
            .expect("sniff");
        assert_eq!(version, Downstream::Http2);

        let mut seen = vec![0u8; PREFACE.len()];
        stream
            .read_exact(&mut seen)
            .await
            .expect("replayed preface");
        assert_eq!(
            seen, PREFACE,
            "hyper must still see the connection from byte zero"
        );
    }

    /// The HTTP/1.1 side of the same contract.
    #[tokio::test]
    async fn sniffing_falls_back_to_http1_and_replays_what_it_read() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        const HEAD: &[u8] = b"GET /livez HTTP/1.1\r\n\r\n";
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("local addr");
        let mut client = tokio::net::TcpStream::connect(addr).await.expect("connect");
        client.write_all(HEAD).await.expect("write head");
        let (accepted, _peer) = listener.accept().await.expect("accept");

        let (version, mut stream) = sniff_downstream_version(accepted, None)
            .await
            .expect("sniff");
        assert_eq!(version, Downstream::Http1);

        let mut seen = vec![0u8; HEAD.len()];
        stream.read_exact(&mut seen).await.expect("replayed head");
        assert_eq!(seen, HEAD);
    }

    /// A peer that opens a connection and closes it without speaking
    /// must not be read as HTTP/2, and must not hang the task.
    #[tokio::test]
    async fn sniffing_treats_an_empty_connection_as_http1() {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("local addr");
        let client = tokio::net::TcpStream::connect(addr).await.expect("connect");
        let (accepted, _peer) = listener.accept().await.expect("accept");
        drop(client);

        let (version, _stream) = sniff_downstream_version(accepted, None)
            .await
            .expect("sniff");
        assert_eq!(version, Downstream::Http1);
    }

    /// A peer that connects and then says nothing holds a connection
    /// slot. hyper's own version read has no timer, so this is the one
    /// place the configured idle timeout can bound it
    /// (AISIX-Cloud#1126).
    #[tokio::test]
    async fn sniffing_gives_up_on_a_peer_that_sends_nothing() {
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("local addr");
        let _client = tokio::net::TcpStream::connect(addr).await.expect("connect");
        let (accepted, _peer) = listener.accept().await.expect("accept");

        let err = match sniff_downstream_version(accepted, Some(Duration::from_millis(50))).await {
            Ok(_) => panic!("a silent peer must not be waited on forever"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::TimedOut);
    }

    /// The replay has to survive a reader that takes fewer bytes than
    /// are buffered — hyper reads into whatever buffer it has.
    #[tokio::test]
    async fn rewind_replays_across_short_reads() {
        use tokio::io::AsyncReadExt;

        let (mut client, server) = tokio::io::duplex(64);
        tokio::io::AsyncWriteExt::write_all(&mut client, b"XYZ")
            .await
            .expect("write tail");
        drop(client);

        let mut stream = Rewind::new(b"PRI ".to_vec(), server);
        let mut all = Vec::new();
        let mut one = [0u8; 1];
        while stream.read(&mut one).await.expect("read") == 1 {
            all.push(one[0]);
        }
        assert_eq!(all, b"PRI XYZ");
    }

    /// A listener running the real `accept_loop`, plus the two levers a
    /// shutdown pulls: `retire` at the start of the drain, `cancel` at
    /// the end.
    struct DrainHarness {
        addr: std::net::SocketAddr,
        retire: watch::Sender<bool>,
        cancel: watch::Sender<bool>,
        serving: tokio::task::JoinHandle<anyhow::Result<()>>,
    }

    async fn drain_harness() -> DrainHarness {
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).expect("bind");
        let addr = listener.local_addr().expect("local addr");
        let (retire, retire_rx) = watch::channel(false);
        let (cancel, cancel_rx) = watch::channel(false);
        let router = axum::Router::new()
            .route("/", axum::routing::get(|| async { "ok" }))
            .route(
                "/slow",
                axum::routing::get(|| async {
                    tokio::time::sleep(Duration::from_millis(1_500)).await;
                    "ok"
                }),
            );
        let serving = tokio::spawn(accept_loop(
            listener,
            router,
            None,
            None,
            ShutdownWatch {
                retire: retire_rx,
                cancel: cancel_rx,
            },
            "test",
            None,
        ));
        DrainHarness {
            addr,
            retire,
            cancel,
            serving,
        }
    }

    async fn get_ok<B>(sender: &mut hyper::client::conn::http1::SendRequest<B>) -> hyper::Result<()>
    where
        B: hyper::body::Body + Default + Send + 'static + Unpin,
        B::Data: Send,
        B::Error: Into<Box<dyn StdError + Send + Sync>>,
    {
        use http_body_util::BodyExt;
        let response = sender
            .send_request(
                axum::http::Request::builder()
                    .uri("/")
                    .body(B::default())
                    .expect("request"),
            )
            .await?;
        assert_eq!(response.status(), 200);
        // An HTTP/1.1 connection cannot carry the next request until the
        // previous body is read off it.
        response.into_body().collect().await?;
        Ok(())
    }

    /// The whole of AISIX-Cloud#1395: an HTTP/2 downstream is told to
    /// retire when the drain STARTS, not when the listener finally
    /// closes. Asserted against a real hyper client, because GOAWAY is
    /// only observable as a peer: the connection future resolves once
    /// the frame lands and the streams it permits are done.
    #[tokio::test]
    async fn an_http2_downstream_is_retired_when_the_drain_starts() {
        use http_body_util::BodyExt;

        let harness = drain_harness().await;
        let tcp = tokio::net::TcpStream::connect(harness.addr)
            .await
            .expect("connect");
        let (mut sender, conn) = hyper::client::conn::http2::handshake::<_, _, axum::body::Body>(
            hyper_util::rt::TokioExecutor::new(),
            hyper_util::rt::TokioIo::new(tcp),
        )
        .await
        .expect("h2 handshake");
        let driving = tokio::spawn(conn);

        let response = sender
            .send_request(
                axum::http::Request::builder()
                    .uri("/")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("first request");
        assert_eq!(response.status(), 200);
        response.into_body().collect().await.expect("body");

        // A stream still running when the drain starts. GOAWAY asks the
        // peer to open no NEW streams; RFC 9113 §6.8's second frame names
        // this one as processed, so it has to be answered, not reset.
        let slow = sender
            .send_request(
                axum::http::Request::builder()
                    .uri("/slow")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await;

        // The drain starts. The listener is NOT closed; the only thing
        // that can reach this peer is a frame.
        harness.retire.send(true).expect("retire");

        let slow = slow.expect("in-flight response head");
        assert_eq!(slow.status(), 200);
        assert_eq!(
            slow.into_body()
                .collect()
                .await
                .expect("in-flight body")
                .to_bytes(),
            "ok".as_bytes(),
            "GOAWAY must not cut off a stream it named as processed",
        );

        let ended = tokio::time::timeout(Duration::from_secs(5), driving)
            .await
            .expect("the h2 connection must end after GOAWAY, not hang until the listener closes")
            .expect("join");
        assert!(
            ended.is_ok(),
            "GOAWAY should end the connection cleanly, not error it: {ended:?}",
        );

        harness.cancel.send(true).expect("cancel");
        let _ = tokio::time::timeout(Duration::from_secs(5), harness.serving).await;
    }

    /// The other half of the same decision. An HTTP/1.1 connection is
    /// retired in band, by `Connection: close` on the responses it
    /// carries — never by the server closing it. Disabling keep-alive
    /// here would close an idle pooled connection, and a client
    /// dispatching onto it in that instant loses the request
    /// (AISIX-Cloud#1394).
    #[tokio::test]
    async fn an_http1_downstream_is_not_closed_when_the_drain_starts() {
        let harness = drain_harness().await;
        let tcp = tokio::net::TcpStream::connect(harness.addr)
            .await
            .expect("connect");
        let (mut sender, conn) = hyper::client::conn::http1::handshake::<_, axum::body::Body>(
            hyper_util::rt::TokioIo::new(tcp),
        )
        .await
        .expect("h1 handshake");
        let driving = tokio::spawn(conn);

        get_ok(&mut sender).await.expect("first request");

        harness.retire.send(true).expect("retire");
        tokio::time::sleep(Duration::from_millis(100)).await;

        get_ok(&mut sender)
            .await
            .expect("the connection must still serve after the drain starts");
        assert!(
            !driving.is_finished(),
            "h1 connection was closed by the server"
        );

        harness.cancel.send(true).expect("cancel");
        let _ = tokio::time::timeout(Duration::from_secs(5), harness.serving).await;
    }

    /// Retiring connections must not take the listener with it. A
    /// balancer only learns about the `/readyz` 503 on its next probe,
    /// and everything it routes here in between still has to be served.
    #[tokio::test]
    async fn the_listener_keeps_accepting_after_the_drain_starts() {
        let harness = drain_harness().await;
        harness.retire.send(true).expect("retire");
        tokio::time::sleep(Duration::from_millis(100)).await;

        let tcp = tokio::net::TcpStream::connect(harness.addr)
            .await
            .expect("a connection opened during the drain must be accepted");
        let (mut sender, conn) = hyper::client::conn::http1::handshake::<_, axum::body::Body>(
            hyper_util::rt::TokioIo::new(tcp),
        )
        .await
        .expect("h1 handshake");
        let _driving = tokio::spawn(conn);
        get_ok(&mut sender)
            .await
            .expect("a request arriving during the drain must be served");

        harness.cancel.send(true).expect("cancel");
        let _ = tokio::time::timeout(Duration::from_secs(5), harness.serving).await;
    }

    /// The listener staying open is only worth something if what it
    /// accepts is actually served. A connection opened during the drain
    /// is retired the moment it is accepted, because `retire` is already
    /// true — so this asks the question that matters: does its FIRST
    /// request still get an answer, or did retiring it refuse exactly
    /// the traffic the open listener exists to take?
    ///
    /// RFC 9113 §6.8's two-phase form is what makes the answer yes. The
    /// advisory `GOAWAY(2^31-1)` goes out first and explicitly allows
    /// in-flight stream creation; the settled GOAWAY that bounds it only
    /// follows a PING round trip. The request lands inside that window
    /// and is one of the streams the second frame names as processed.
    #[tokio::test]
    async fn an_http2_connection_opened_during_the_drain_still_serves_its_request() {
        use http_body_util::BodyExt;

        let harness = drain_harness().await;
        harness.retire.send(true).expect("retire");
        tokio::time::sleep(Duration::from_millis(100)).await;

        let tcp = tokio::net::TcpStream::connect(harness.addr)
            .await
            .expect("a connection opened during the drain must be accepted");
        let (mut sender, conn) = hyper::client::conn::http2::handshake::<_, _, axum::body::Body>(
            hyper_util::rt::TokioExecutor::new(),
            hyper_util::rt::TokioIo::new(tcp),
        )
        .await
        .expect("h2 handshake");
        let _driving = tokio::spawn(conn);

        let response = sender
            .send_request(
                axum::http::Request::builder()
                    .uri("/")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("a request arriving during the drain must be served");
        assert_eq!(response.status(), 200);
        assert_eq!(
            response
                .into_body()
                .collect()
                .await
                .expect("body")
                .to_bytes(),
            "ok".as_bytes(),
        );

        harness.cancel.send(true).expect("cancel");
        let _ = tokio::time::timeout(Duration::from_secs(5), harness.serving).await;
    }

    /// And the end of the drain does close it, and does return — an
    /// accept loop that outlived its cancel would hold the process up
    /// until the platform killed it.
    #[tokio::test]
    async fn the_listener_stops_accepting_when_the_drain_ends() {
        let harness = drain_harness().await;
        let tcp = tokio::net::TcpStream::connect(harness.addr)
            .await
            .expect("connect");
        let (mut sender, conn) = hyper::client::conn::http1::handshake::<_, axum::body::Body>(
            hyper_util::rt::TokioIo::new(tcp),
        )
        .await
        .expect("h1 handshake");
        let _driving = tokio::spawn(conn);
        get_ok(&mut sender).await.expect("first request");

        harness.cancel.send(true).expect("cancel");
        let served = tokio::time::timeout(Duration::from_secs(5), harness.serving)
            .await
            .expect("accept_loop must return once the drain ends")
            .expect("join");
        assert!(served.is_ok(), "{served:?}");

        assert!(
            tokio::net::TcpStream::connect(harness.addr)
                .await
                .and_then(|s| s.peer_addr())
                .is_err(),
            "the listener must be closed once the drain has ended",
        );
    }

    /// The count is what tells an operator during a drain whether the
    /// balancer has stopped routing here. Nothing else checks that the
    /// slot comes back: every drain assertion is "at least one", so a
    /// guard that outlived its connection would keep them all green
    /// while the number climbed for the life of the process.
    #[tokio::test]
    async fn an_accepted_connection_holds_a_slot_until_its_task_ends() {
        let livez = std::sync::Arc::new(aisix_proxy::LivezState::new());
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).expect("bind");
        let addr = listener.local_addr().expect("local addr");
        let (_retire, retire_rx) = watch::channel(false);
        let (cancel, cancel_rx) = watch::channel(false);
        let router = axum::Router::new().route("/", axum::routing::get(|| async { "ok" }));
        let serving = tokio::spawn(accept_loop(
            listener,
            router,
            None,
            None,
            ShutdownWatch {
                retire: retire_rx,
                cancel: cancel_rx,
            },
            "test",
            Some(livez.clone()),
        ));

        let tcp = tokio::net::TcpStream::connect(addr).await.expect("connect");
        let (mut sender, conn) = hyper::client::conn::http1::handshake::<_, axum::body::Body>(
            hyper_util::rt::TokioIo::new(tcp),
        )
        .await
        .expect("h1 handshake");
        let driving = tokio::spawn(conn);
        get_ok(&mut sender).await.expect("first request");
        assert_eq!(livez.open_connections(), 1);

        drop(sender);
        let _ = driving.await;
        for _ in 0..50 {
            if livez.open_connections() == 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert_eq!(
            livez.open_connections(),
            0,
            "the slot must go back when the connection's task ends",
        );

        cancel.send(true).expect("cancel");
        let _ = tokio::time::timeout(Duration::from_secs(5), serving).await;
    }
}
