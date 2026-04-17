# Control Plane Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `src/agent/` module to AISIX that manages periodic heartbeats with the API7 Enterprise control plane, including mTLS client auth, endpoint failover, and graceful startup modes.

**Architecture:** A new `src/agent/` module runs alongside the existing etcd config subsystem. It reads its endpoint and TLS config from `deployment.etcd` (or environment variable overrides), builds a separate mTLS `reqwest::Client`, and spawns a background heartbeat task. The `AgentHandle` is injected into `proxy::AppState` as `Option<AgentHandle>` (None = standalone mode, no changes needed to existing tests).

**Tech Stack:** Rust, tokio, reqwest (native-tls for mTLS), backon (retry), serde_json, uuid, anyhow, thiserror

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/config/etcd.rs` | Modify | Add `tls: Option<EtcdTlsConfig>` to `Config` |
| `src/config/types.rs` | Modify | Add `control_plane: Option<ControlPlaneConfig>` to `Deployment`; add `ControlPlaneConfig`, `StartupMode` types |
| `src/config/mod.rs` | Modify | Add `Environment` source to `load()` |
| `src/agent/mod.rs` | Create | `AgentHandle`, `AgentState`, `start()` public API |
| `src/agent/types.rs` | Create | `HeartbeatRequest`, `HeartbeatResponse`, `HeartbeatConfig` |
| `src/agent/client.rs` | Create | `CpClient` — mTLS reqwest client with endpoint failover |
| `src/agent/heartbeat.rs` | Create | `run_heartbeat()` background task |
| `src/lib.rs` | Modify | `pub mod agent;` |
| `src/proxy/mod.rs` | Modify | `AppState` gains `agent: Option<AgentHandle>` |
| `src/main.rs` | Modify | Wire agent startup into main, inject into AppState |

---

## Task 1: Add `EtcdTlsConfig` to etcd config

**Files:**
- Modify: `src/config/etcd.rs`

- [ ] **Step 1: Add `EtcdTlsConfig` struct and `tls` field to `Config`**

In `src/config/etcd.rs`, add after the existing imports:

```rust
#[derive(Clone, Debug, Default, Deserialize)]
pub struct EtcdTlsConfig {
    pub cert_file: Option<String>,
    pub key_file: Option<String>,
    pub ca_file: Option<String>,
}
```

And modify the `Config` struct (lines 23-30) to add `tls`:

```rust
#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub host: Vec<String>,
    pub prefix: String,
    pub timeout: u32,
    pub user: Option<String>,
    pub password: Option<String>,
    pub tls: Option<EtcdTlsConfig>,
}
```

And update `Default for Config` (lines 32-41) to include `tls: None`:

```rust
impl Default for Config {
    fn default() -> Self {
        Self {
            host: vec!["http://127.0.0.1:2379".to_string()],
            prefix: "/aisix".to_string(),
            timeout: 5,
            user: None,
            password: None,
            tls: None,
        }
    }
}
```

- [ ] **Step 2: Verify the build compiles**

```bash
cargo build 2>&1 | head -30
```

Expected: no errors (tls field is unused for now, which is fine)

- [ ] **Step 3: Commit**

```bash
git add src/config/etcd.rs
git commit -m "feat(config): add optional TLS fields to etcd Config"
```

---

## Task 2: Add `ControlPlaneConfig` and `StartupMode` to config types

**Files:**
- Modify: `src/config/types.rs`

- [ ] **Step 1: Add new types**

After the existing `DeploymentAdmin` struct (around line 35), add:

```rust
/// Startup behavior when the first heartbeat to the control plane fails.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum StartupMode {
    /// Only log a warning; process starts normally and retries in the background (default).
    #[default]
    Soft,
    /// Retry with backoff; if all attempts fail, exit(1).
    Strict,
}

fn default_heartbeat_interval() -> u64 {
    10
}

/// Configuration for the API7 Enterprise control plane integration.
/// When absent, AISIX runs in standalone mode (agent not started).
#[derive(Clone, Debug, Deserialize)]
pub struct ControlPlaneConfig {
    /// Interval between heartbeats in seconds (default 10).
    #[serde(default = "default_heartbeat_interval")]
    pub heartbeat_interval: u64,
    /// Behavior when the initial heartbeat fails.
    #[serde(default)]
    pub startup_mode: StartupMode,
}
```

Modify the `Deployment` struct (lines 37-43) to add `control_plane`:

```rust
#[derive(Clone, Debug, Default, Deserialize)]
pub struct Deployment {
    #[serde(default)]
    pub etcd: etcd::Config,
    #[serde(default)]
    pub admin: DeploymentAdmin,
    /// When None, AISIX runs in standalone mode without a control plane agent.
    pub control_plane: Option<ControlPlaneConfig>,
}
```

- [ ] **Step 2: Verify the build compiles**

```bash
cargo build 2>&1 | head -30
```

Expected: no errors

- [ ] **Step 3: Commit**

```bash
git add src/config/types.rs
git commit -m "feat(config): add ControlPlaneConfig and StartupMode types"
```

---

## Task 3: Add environment variable override support to config loader

**Files:**
- Modify: `src/config/mod.rs`

- [ ] **Step 1: Add `Environment` source to `load()`**

Replace the existing `load()` function body (the entire function from line 11 to line 26):

```rust
/// Load configuration from a file and environment variable overrides.
///
/// Environment variables override file values using the prefix `AISIX` and
/// double-underscore (`__`) as the path separator, e.g.:
///   `AISIX__DEPLOYMENT__CONTROL_PLANE__STARTUP_MODE=strict`
pub fn load(config_file: Option<String>) -> Result<Config, config::ConfigError> {
    let mut builder = config::Config::builder();

    if let Some(ref file) = config_file {
        // If a config file is specified, it must exist
        builder = builder.add_source(config::File::with_name(file).required(true));
    } else {
        // If no config file is specified, use the default "config" file, which is optional
        builder = builder.add_source(config::File::with_name("config").required(false));
    }

    // Environment variables (prefix AISIX, separator __) override file values.
    builder = builder.add_source(
        config::Environment::with_prefix("AISIX")
            .separator("__")
            .try_parsing(true),
    );

    builder
        .build()?
        // If the file cannot be found, the `Config::default()` will be used.
        .try_deserialize::<Config>()
}
```

- [ ] **Step 2: Verify the build compiles and existing tests pass**

```bash
cargo build 2>&1 | head -20
cargo test 2>&1 | tail -20
```

Expected: no errors, all tests pass (environment source is additive)

- [ ] **Step 3: Commit**

```bash
git add src/config/mod.rs
git commit -m "feat(config): add environment variable override support (AISIX__ prefix)"
```

---

## Task 4: Create `src/agent/types.rs`

**Files:**
- Create: `src/agent/types.rs`

- [ ] **Step 1: Write the payload types**

Create `src/agent/types.rs`:

```rust
use serde::{Deserialize, Serialize};

/// AISIX → control plane: heartbeat request body.
#[derive(Debug, Serialize)]
pub struct HeartbeatRequest {
    /// Unique instance ID; read from config or auto-generated and persisted.
    pub instance_id: String,
    /// Runtime ID: a new UUID v4 generated each time the process starts.
    /// Used by the control plane to detect restarts.
    pub run_id: String,
    pub hostname: String,
    pub ip: String,
    pub version: String,
    pub ports: Vec<u16>,
}

/// Control plane → AISIX: heartbeat response body.
#[derive(Debug, Deserialize)]
pub struct HeartbeatResponse {
    pub instance_id: String,
    /// Reserved for future config push from the control plane.
    pub config: Option<HeartbeatConfig>,
}

/// Reserved config structure returned by the control plane (currently empty).
#[derive(Debug, Deserialize)]
pub struct HeartbeatConfig {
    /// Config version; empty string means no config has been pushed yet.
    pub config_version: String,
    /// Config payload (currently an empty JSON object).
    pub config_payload: serde_json::Value,
}
```

- [ ] **Step 2: Commit**

```bash
git add src/agent/types.rs
git commit -m "feat(agent): add heartbeat payload types"
```

---

## Task 5: Create `src/agent/client.rs`

**Files:**
- Create: `src/agent/client.rs`

- [ ] **Step 1: Write the `CpClient`**

Create `src/agent/client.rs`:

```rust
use std::{
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};

use anyhow::{Context, Result};
use serde::{Serialize, de::DeserializeOwned};
use thiserror::Error;

use crate::config::etcd;

#[derive(Debug, Error)]
pub enum CpClientError {
    #[error("all endpoints failed: {0}")]
    AllEndpointsFailed(String),
    #[error("HTTP {status}: {body}")]
    HttpError {
        status: reqwest::StatusCode,
        body: String,
    },
    #[error("response deserialization failed: {0}")]
    DeserializationError(String),
}

pub struct CpClient {
    /// Endpoint list from deployment.etcd.host (or API7_CONTROL_PLANE_ENDPOINTS override).
    endpoints: Vec<String>,
    /// Index of the currently active endpoint (round-robin on failure).
    current: AtomicUsize,
    /// reqwest client configured with mTLS (when certificates are available).
    http: reqwest::Client,
}

impl CpClient {
    /// Build a `CpClient` from the etcd configuration.
    ///
    /// Endpoint resolution (highest priority first):
    ///   1. `API7_CONTROL_PLANE_ENDPOINTS` env var (JSON array of strings)
    ///   2. `deployment.etcd.host`
    ///
    /// mTLS certificate resolution (highest priority first):
    ///   1. Env vars `API7_CONTROL_PLANE_CERT` / `API7_CONTROL_PLANE_KEY` / `API7_CONTROL_PLANE_CA`
    ///      (PEM content — injected by Docker deployment scripts)
    ///   2. File paths in `deployment.etcd.tls.cert_file` / `key_file` / `ca_file`
    ///      (Helm / local development)
    pub fn new(etcd_config: &etcd::Config) -> Result<Self> {
        let endpoints: Vec<String> = std::env::var("API7_CONTROL_PLANE_ENDPOINTS")
            .ok()
            .and_then(|s| serde_json::from_str::<Vec<String>>(&s).ok())
            .unwrap_or_else(|| etcd_config.host.clone());

        if endpoints.is_empty() {
            anyhow::bail!("no control plane endpoints configured");
        }

        let mut builder = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(30));

        // Load mTLS identity (client cert + key) if available.
        if let Some(identity) = Self::load_identity(etcd_config.tls.as_ref())
            .context("failed to load mTLS client identity")?
        {
            builder = builder.identity(identity);
        }

        // Load CA certificate if available.
        if let Some(ca_cert) = Self::load_ca_cert(etcd_config.tls.as_ref())
            .context("failed to load CA certificate")?
        {
            builder = builder.add_root_certificate(ca_cert);
        }

        let http = builder.build().context("failed to build HTTP client")?;

        Ok(Self {
            endpoints,
            current: AtomicUsize::new(0),
            http,
        })
    }

    /// POST `path` with a JSON body, trying each endpoint in turn.
    /// Returns the deserialized response body on success.
    pub async fn post<Req, Resp>(&self, path: &str, body: &Req) -> Result<Resp, CpClientError>
    where
        Req: Serialize,
        Resp: DeserializeOwned,
    {
        let total = self.endpoints.len();
        let start = self.current.load(Ordering::Relaxed);

        let mut last_err = String::new();

        for i in 0..total {
            let idx = (start + i) % total;
            let base = &self.endpoints[idx];
            let url = format!("{}{}", base.trim_end_matches('/'), path);

            match self.http.post(&url).json(body).send().await {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        // Advance the current endpoint pointer to this working one.
                        self.current.store(idx, Ordering::Relaxed);
                        let bytes = resp
                            .bytes()
                            .await
                            .map_err(|e| CpClientError::AllEndpointsFailed(e.to_string()))?;
                        return serde_json::from_slice::<Resp>(&bytes).map_err(|e| {
                            CpClientError::DeserializationError(e.to_string())
                        });
                    } else {
                        let body_text = resp.text().await.unwrap_or_default();
                        if status.is_server_error() {
                            // Server errors: try next endpoint.
                            last_err = format!("endpoint {}: HTTP {}: {}", base, status, body_text);
                            continue;
                        }
                        // Client errors (4xx): don't retry other endpoints.
                        return Err(CpClientError::HttpError {
                            status,
                            body: body_text,
                        });
                    }
                }
                Err(e) => {
                    last_err = format!("endpoint {}: {}", base, e);
                    continue;
                }
            }
        }

        Err(CpClientError::AllEndpointsFailed(last_err))
    }

    /// Load the mTLS client identity (cert + key) from env vars or file paths.
    fn load_identity(
        tls: Option<&etcd::EtcdTlsConfig>,
    ) -> Result<Option<reqwest::Identity>> {
        // Try environment variables first (Docker deployment).
        let cert_pem = std::env::var("API7_CONTROL_PLANE_CERT").ok();
        let key_pem = std::env::var("API7_CONTROL_PLANE_KEY").ok();

        if let (Some(cert), Some(key)) = (cert_pem, key_pem) {
            let pem = format!("{}\n{}", cert, key);
            let identity = reqwest::Identity::from_pem(pem.as_bytes())
                .context("failed to parse mTLS identity from environment variables")?;
            return Ok(Some(identity));
        }

        // Fall back to file paths from config.
        if let Some(tls) = tls {
            if let (Some(cert_file), Some(key_file)) =
                (tls.cert_file.as_deref(), tls.key_file.as_deref())
            {
                let cert =
                    std::fs::read_to_string(cert_file).with_context(|| {
                        format!("failed to read cert_file '{}'", cert_file)
                    })?;
                let key =
                    std::fs::read_to_string(key_file).with_context(|| {
                        format!("failed to read key_file '{}'", key_file)
                    })?;
                let pem = format!("{}\n{}", cert, key);
                let identity = reqwest::Identity::from_pem(pem.as_bytes())
                    .context("failed to parse mTLS identity from files")?;
                return Ok(Some(identity));
            }
        }

        Ok(None)
    }

    /// Load the CA certificate from env var or file path.
    fn load_ca_cert(
        tls: Option<&etcd::EtcdTlsConfig>,
    ) -> Result<Option<reqwest::Certificate>> {
        // Try environment variable first (Docker deployment).
        if let Ok(ca_pem) = std::env::var("API7_CONTROL_PLANE_CA") {
            let cert = reqwest::Certificate::from_pem(ca_pem.as_bytes())
                .context("failed to parse CA certificate from environment variable")?;
            return Ok(Some(cert));
        }

        // Fall back to file path from config.
        if let Some(tls) = tls {
            if let Some(ca_file) = tls.ca_file.as_deref() {
                let ca_pem = std::fs::read_to_string(ca_file)
                    .with_context(|| format!("failed to read ca_file '{}'", ca_file))?;
                let cert = reqwest::Certificate::from_pem(ca_pem.as_bytes())
                    .context("failed to parse CA certificate from file")?;
                return Ok(Some(cert));
            }
        }

        Ok(None)
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add src/agent/client.rs
git commit -m "feat(agent): add CpClient with mTLS and endpoint failover"
```

---

## Task 6: Create `src/agent/heartbeat.rs`

**Files:**
- Create: `src/agent/heartbeat.rs`

- [ ] **Step 1: Write the heartbeat task**

Create `src/agent/heartbeat.rs`:

```rust
use std::{sync::Arc, time::Duration};

use log::{info, warn};
use tokio::{sync::Notify, time};

use crate::agent::{AgentState, client::CpClient, types::HeartbeatRequest};

/// Configuration for the heartbeat task.
pub struct HeartbeatTaskConfig {
    pub interval: Duration,
}

/// Run a periodic heartbeat loop.
///
/// On each tick:
///   1. Build a `HeartbeatRequest` from current `AgentState`.
///   2. POST it to `/api/ai_dataplane/heartbeat`.
///   3. On success: log at debug level and update `cp_config_version`.
///   4. On failure: log a warning; the loop continues (network blips are recoverable).
///
/// Returns when `shutdown` is notified.
pub async fn run_heartbeat(
    client: Arc<CpClient>,
    state: Arc<AgentState>,
    config: HeartbeatTaskConfig,
    shutdown: Arc<Notify>,
) {
    let mut interval = time::interval(config.interval);
    // The first tick fires immediately; subsequent ticks respect the interval.
    interval.set_missed_tick_behavior(time::MissedTickBehavior::Delay);

    loop {
        tokio::select! {
            biased;
            _ = shutdown.notified() => {
                info!("agent heartbeat: shutdown requested");
                break;
            }
            _ = interval.tick() => {
                let req = build_request(&state);
                match client
                    .post::<HeartbeatRequest, crate::agent::types::HeartbeatResponse>(
                        "/api/ai_dataplane/heartbeat",
                        &req,
                    )
                    .await
                {
                    Ok(resp) => {
                        log::debug!("agent heartbeat: ok (instance_id={})", resp.instance_id);
                        if let Some(cfg) = resp.config {
                            state.cp_config_version.store(
                                Arc::new(cfg.config_version),
                            );
                        }
                    }
                    Err(err) => {
                        warn!("agent heartbeat: failed: {}", err);
                    }
                }
            }
        }
    }
}

fn build_request(state: &AgentState) -> HeartbeatRequest {
    HeartbeatRequest {
        instance_id: state.instance_id.clone(),
        run_id: state.run_id.clone(),
        hostname: state.hostname.clone(),
        ip: state.ip.clone(),
        version: state.version.clone(),
        ports: state.ports.clone(),
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add src/agent/heartbeat.rs
git commit -m "feat(agent): add periodic heartbeat task"
```

---

## Task 7: Create `src/agent/mod.rs` (public API)

**Files:**
- Create: `src/agent/mod.rs`

- [ ] **Step 1: Write the module**

Create `src/agent/mod.rs`:

```rust
mod client;
mod heartbeat;
pub mod types;

use std::{sync::Arc, time::Duration};

use anyhow::{Context, Result};
use arc_swap::ArcSwap;
use backon::{ExponentialBuilder, Retryable};
use log::{error, info, warn};
use tokio::sync::Notify;
use uuid::Uuid;

use crate::config::{Config, types::StartupMode};

use self::{
    client::CpClient,
    heartbeat::{HeartbeatTaskConfig, run_heartbeat},
    types::HeartbeatRequest,
};

/// Internal state shared between `AgentHandle` and the heartbeat task.
pub struct AgentState {
    /// Unique stable instance ID (from config or auto-generated).
    pub instance_id: String,
    /// Per-process runtime ID (new UUID on each start).
    pub run_id: String,
    pub hostname: String,
    pub ip: String,
    pub version: String,
    pub ports: Vec<u16>,
    /// Config version last pushed by the control plane (initially empty).
    pub cp_config_version: ArcSwap<String>,
    /// AI Gateway Group ID injected by the deployment script.
    pub ai_gateway_group_id: String,
}

/// A cloneable, cheaply shareable handle to the agent's state.
/// Safe to place in `axum::extract::State`.
#[derive(Clone)]
pub struct AgentHandle {
    inner: Arc<AgentState>,
}

impl AgentHandle {
    /// Returns the AI Gateway Group ID this instance belongs to.
    pub fn group_id(&self) -> &str {
        &self.inner.ai_gateway_group_id
    }

    /// Returns the config version last received from the control plane.
    pub fn cp_config_version(&self) -> String {
        self.inner.cp_config_version.load().as_ref().clone()
    }
}

/// Start the control plane agent.
///
/// Reads `API7_GATEWAY_GROUP_ID` from the environment.
/// Performs an initial heartbeat; behavior on failure is governed by
/// `config.deployment.control_plane.startup_mode`.
///
/// Returns `None` if `control_plane` is not configured (standalone mode).
pub async fn start(config: &Config, shutdown: Arc<Notify>) -> Result<Option<AgentHandle>> {
    let Some(cp_config) = config.deployment.control_plane.as_ref() else {
        info!("agent: control_plane not configured, running in standalone mode");
        return Ok(None);
    };

    let group_id = std::env::var("API7_GATEWAY_GROUP_ID").unwrap_or_default();
    let instance_id = Uuid::new_v4().to_string(); // TODO: persist across restarts if needed
    let run_id = Uuid::new_v4().to_string();
    let hostname = hostname::get()
        .map(|h| h.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "unknown".to_string());
    let ip = local_ip().unwrap_or_else(|| "0.0.0.0".to_string());
    let version = env!("CARGO_PKG_VERSION").to_string();
    let ports: Vec<u16> = vec![
        config.server.proxy.listen.port(),
        config.server.admin.listen.port(),
    ];

    let cp_client = Arc::new(
        CpClient::new(&config.deployment.etcd).context("failed to build control plane client")?,
    );

    let state = Arc::new(AgentState {
        instance_id,
        run_id,
        hostname,
        ip,
        version,
        ports,
        cp_config_version: ArcSwap::new(Arc::new(String::new())),
        ai_gateway_group_id: group_id,
    });

    // Perform the initial heartbeat.
    let initial_req = HeartbeatRequest {
        instance_id: state.instance_id.clone(),
        run_id: state.run_id.clone(),
        hostname: state.hostname.clone(),
        ip: state.ip.clone(),
        version: state.version.clone(),
        ports: state.ports.clone(),
    };

    let initial_result = send_initial_heartbeat(&cp_client, &initial_req, &cp_config.startup_mode).await;

    match initial_result {
        Ok(()) => {
            info!("agent: initial heartbeat succeeded");
        }
        Err(e) => {
            match cp_config.startup_mode {
                StartupMode::Soft => {
                    warn!("agent: initial heartbeat failed (soft mode, continuing): {}", e);
                }
                StartupMode::Strict => {
                    return Err(e.context("agent: initial heartbeat failed in strict mode"));
                }
            }
        }
    }

    // Spawn the background heartbeat task.
    let task_client = cp_client.clone();
    let task_state = state.clone();
    let interval = Duration::from_secs(cp_config.heartbeat_interval);
    let task_shutdown = shutdown.clone();
    tokio::spawn(async move {
        run_heartbeat(
            task_client,
            task_state,
            HeartbeatTaskConfig { interval },
            task_shutdown,
        )
        .await;
    });

    Ok(Some(AgentHandle { inner: state }))
}

/// Send a single heartbeat, with retry logic for `strict` startup mode.
async fn send_initial_heartbeat(
    client: &Arc<CpClient>,
    req: &HeartbeatRequest,
    mode: &StartupMode,
) -> Result<()> {
    match mode {
        StartupMode::Soft => {
            // One attempt only; failures are handled by the caller.
            client
                .post::<HeartbeatRequest, types::HeartbeatResponse>(
                    "/api/ai_dataplane/heartbeat",
                    req,
                )
                .await
                .map(|_| ())
                .map_err(|e| anyhow::anyhow!(e))
        }
        StartupMode::Strict => {
            // Retry with exponential backoff (max 5 attempts, starting at 5s).
            let attempt = || async {
                client
                    .post::<HeartbeatRequest, types::HeartbeatResponse>(
                        "/api/ai_dataplane/heartbeat",
                        req,
                    )
                    .await
                    .map(|_| ())
                    .map_err(|e| anyhow::anyhow!(e))
            };
            attempt
                .retry(
                    ExponentialBuilder::default()
                        .with_min_delay(Duration::from_secs(5))
                        .with_max_delay(Duration::from_secs(60))
                        .with_max_times(5),
                )
                .notify(|err, dur| {
                    warn!("agent: heartbeat retry in {:?}: {}", dur, err);
                })
                .await
        }
    }
}

/// Attempt to find a non-loopback local IPv4 address.
fn local_ip() -> Option<String> {
    use std::net::{IpAddr, UdpSocket};
    // Trick: connect to an external address (doesn't actually send packets).
    let socket = UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("8.8.8.8:80").ok()?;
    match socket.local_addr().ok()?.ip() {
        IpAddr::V4(ip) => Some(ip.to_string()),
        IpAddr::V6(ip) => Some(ip.to_string()),
    }
}
```

- [ ] **Step 2: Add `hostname` crate to Cargo.toml**

In `Cargo.toml`, add after the `uuid` line:

```toml
hostname = "0.4"
```

- [ ] **Step 3: Register the agent module in lib.rs**

In `src/lib.rs`, add:

```rust
pub mod agent;
```

So the file becomes:

```rust
pub mod admin;
pub mod agent;
pub mod config;
pub mod gateway;
pub mod providers;
pub mod proxy;
pub mod utils;
```

- [ ] **Step 4: Verify the build compiles**

```bash
cargo build 2>&1 | head -40
```

Expected: no errors

- [ ] **Step 5: Commit**

```bash
git add src/agent/mod.rs src/agent/types.rs src/agent/client.rs src/agent/heartbeat.rs src/lib.rs Cargo.toml Cargo.lock
git commit -m "feat(agent): add control plane agent module with heartbeat support"
```

---

## Task 8: Inject `AgentHandle` into `proxy::AppState`

**Files:**
- Modify: `src/proxy/mod.rs`

- [ ] **Step 1: Add `agent` field to `AppState`**

In `src/proxy/mod.rs`, update the imports and `AppState`:

```rust
mod handlers;
mod hooks;
mod middlewares;

use std::sync::Arc;

use axum::{
    Router,
    extract::DefaultBodyLimit,
    middleware::{from_fn, from_fn_with_state},
    routing::{get, post},
};

use crate::{
    agent::AgentHandle,
    config::{Config, entities::ResourceRegistry},
};

// types
pub mod types {
    pub use super::handlers::{
        chat_completions::{
            ChatCompletionChoice, ChatCompletionChunk, ChatCompletionChunkChoice,
            ChatCompletionChunkDelta, ChatCompletionRequest, ChatCompletionResponse,
            ChatCompletionUsage, ChatMessage,
        },
        embeddings::{EmbeddingRequest, EmbeddingResponse},
    };
}

const DEFAULT_REQUEST_BODY_LIMIT_BYTES: usize = 10 * 1024 * 1024;

#[derive(Clone)]
pub struct AppState {
    #[allow(dead_code)]
    config: Arc<Config>,
    resources: Arc<ResourceRegistry>,
    /// `None` when running in standalone mode (no control_plane configured).
    #[allow(dead_code)]
    agent: Option<AgentHandle>,
}

impl AppState {
    pub fn new(
        config: Arc<Config>,
        resources: Arc<ResourceRegistry>,
        agent: Option<AgentHandle>,
    ) -> Self {
        Self {
            config,
            resources,
            agent,
        }
    }

    pub fn resources(&self) -> Arc<ResourceRegistry> {
        self.resources.clone()
    }
}

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .merge(Router::new().route("/v1/models", get(handlers::models::list_models)))
        .route(
            "/v1/chat/completions",
            post(handlers::chat_completions::chat_completions),
        )
        .route("/v1/embeddings", post(handlers::embeddings::embeddings))
        .layer(DefaultBodyLimit::max(DEFAULT_REQUEST_BODY_LIMIT_BYTES))
        .layer(from_fn_with_state(state.clone(), middlewares::auth))
        .layer(from_fn(middlewares::trace))
        .with_state(state)
}
```

- [ ] **Step 2: Verify the build compiles**

```bash
cargo build 2>&1 | head -40
```

Expected: one compile error in `main.rs` (AppState::new now requires 3 args) — that's expected; fix in next task.

- [ ] **Step 3: Commit (will compile after next task)**

```bash
git add src/proxy/mod.rs
git commit -m "feat(proxy): add Option<AgentHandle> to AppState"
```

---

## Task 9: Wire agent startup in `main.rs`

**Files:**
- Modify: `src/main.rs`

- [ ] **Step 1: Update main() to start the agent and pass it to AppState**

Replace the relevant section of `main.rs`. The full updated function:

```rust
use std::{process::exit, sync::Arc};

use aisix::{config::Config, *};
use anyhow::{Context, Result};
use axum::Router;
use clap::Parser;
use log::{error, info};
use tokio::{select, sync::{Notify, oneshot}};

#[derive(Parser, Debug)]
#[command(version)]
struct Args {
    /// Path to the configuration file
    #[arg(short, long)]
    config: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let (ob_shutdown_signal, ob_shutdown_task) =
        init_observability().context("failed to initialize observability")?;

    let config = Arc::new(config::load(args.config).context("failed to load configuration")?);

    let config_provider = config::create_provider(&config)
        .await
        .context("failed to create config provider")?;
    let resources =
        Arc::new(config::entities::ResourceRegistry::new(config_provider.clone()).await);

    providers::init_client();

    // Start the control plane agent (None in standalone mode).
    let agent_shutdown = Arc::new(Notify::new());
    let agent = agent::start(&config, agent_shutdown.clone())
        .await
        .context("failed to start control plane agent")?;

    let proxy_router = proxy::create_router(proxy::AppState::new(
        config.clone(),
        resources.clone(),
        agent,
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

    // Shut down agent background task.
    agent_shutdown.notify_one();

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
```

(The `init_observability`, `serve_proxy`, `serve_admin`, and `serve` functions remain unchanged.)

- [ ] **Step 2: Build and run tests**

```bash
cargo build 2>&1 | head -40
cargo test 2>&1 | tail -30
```

Expected: build succeeds, all existing tests pass (no test uses `control_plane` config, so agent.start returns None)

- [ ] **Step 3: Run clippy**

```bash
cargo clippy --all-targets --all-features --locked -- -D warnings 2>&1 | head -50
```

Expected: no warnings

- [ ] **Step 4: Commit**

```bash
git add src/main.rs
git commit -m "feat(main): wire control plane agent startup into server lifecycle"
```

---

## Task 10: Final verification

- [ ] **Step 1: Run full test suite**

```bash
cargo test 2>&1 | tail -20
```

Expected: all tests pass

- [ ] **Step 2: Run clippy**

```bash
cargo clippy --all-targets --all-features --locked -- -D warnings 2>&1
```

Expected: no warnings or errors

- [ ] **Step 3: Run cargo fmt check**

```bash
cargo fmt -- --check 2>&1
```

Expected: no formatting differences; if there are, run `cargo fmt` and re-check

- [ ] **Step 4: Format if needed and commit**

```bash
cargo fmt
git add -u
git diff --staged --quiet || git commit -m "chore: apply rustfmt"
```

- [ ] **Step 5: Create PR**

```bash
gh pr create \
  --title "feat(agent): add control plane agent module (RFC-004)" \
  --body "$(cat <<'EOF'
## Summary

- Add `src/agent/` module as the unified abstraction layer for all AISIX ↔ control plane communication
- Implement periodic heartbeat (`POST /api/ai_dataplane/heartbeat`) with mTLS client auth and endpoint failover
- Support two certificate injection methods: env var PEM content (Docker) and file paths (Helm)
- Add `deployment.control_plane` config block with `startup_mode: soft|strict` and `heartbeat_interval`
- Add environment variable override support for all config values (`AISIX__` prefix)
- Inject `Option<AgentHandle>` into `proxy::AppState`; existing tests are unaffected (standalone mode when `control_plane` is absent)

## Config Example

```yaml
deployment:
  etcd:
    host: ["https://dp-manager.api7.internal:9443"]
    prefix: /aisix
    timeout: 30
  control_plane:
    heartbeat_interval: 10
    startup_mode: soft
```

## Environment Variables

| Variable | Purpose |
|---|---|
| `API7_GATEWAY_GROUP_ID` | AI Gateway Group ID |
| `API7_CONTROL_PLANE_ENDPOINTS` | Override etcd endpoints (JSON array) |
| `API7_CONTROL_PLANE_CERT` | mTLS client cert (PEM content) |
| `API7_CONTROL_PLANE_KEY` | mTLS client key (PEM content) |
| `API7_CONTROL_PLANE_CA` | CA cert (PEM content) |

## Backward Compatibility

- Existing config files without `control_plane` field → standalone mode, zero behavior change
- All existing tests pass without modification
EOF
)"
```

---

## Self-Review Notes

- **Task 7** references `AgentState` fields in `heartbeat.rs` (Task 6) — field names are consistent (`instance_id`, `run_id`, `hostname`, `ip`, `version`, `ports`, `cp_config_version`).
- **Task 7** uses `ExponentialBuilder` from `backon` — this crate is already in `Cargo.toml` with `tokio-sleep` feature.
- **Task 7** uses `ArcSwap` — `arc-swap` crate is already in `Cargo.toml`.
- **Task 5** uses `reqwest::Identity::from_pem` — available with `native-tls` feature (already enabled).
- **Task 9** uses `agent_shutdown.notify_one()` — the heartbeat loop uses `shutdown.notified()` in a `biased` select, so it will cleanly exit.
- The `hostname` crate is new and must be added to `Cargo.toml` in Task 7.
- All file paths in tasks are exact and consistent across tasks.
