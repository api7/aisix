//! One-shot DP registration against aisix.cloud's control plane.
//!
//! Protocol: `POST /dp/register` per prd-09 §9.3.5. The DP sends
//! `{ hostname, version, os, ip }` with a Deployment Token Bearer
//! and receives an mTLS bundle + endpoint map in return.
//!
//! The registration is **single-use** — the token gets consumed on
//! the CP side and the private key appears in the response exactly
//! once. This module writes the bundle to disk atomically so a
//! partial write never leaves the DP with an unusable cert.
//!
//! Boot-time flow (called from main.rs):
//!
//! 1. If `managed.enabled = false` → skip entirely.
//! 2. If mTLS files already exist under `managed.mtls_dir` → skip;
//!    the DP has registered before and should reuse the bundle.
//! 3. Otherwise, if `registration_token` is set → call this module's
//!    `register_and_persist`.
//! 4. If registration succeeds, the rest of startup uses the
//!    refreshed config (endpoints + tls paths) as if the user had
//!    put them there themselves.

use std::path::{Path, PathBuf};
use std::time::Duration;

use aisix_core::ManagedConfig;
use anyhow::{anyhow, bail, Context};
use serde::{Deserialize, Serialize};

/// Outcome of a successful registration. Mirrors the §9.3.5 response
/// body with file paths attached for `main` to wire into the etcd
/// config.
///
/// Note: `heartbeat_*` / `telemetry_*` fields are captured now so the
/// heartbeat + telemetry PR can wire them into the supervisor task
/// without another config round-trip. `main` currently only consumes
/// the cert paths and etcd endpoint.
#[derive(Debug, Clone)]
#[allow(
    dead_code,
    reason = "heartbeat + telemetry fields consumed in follow-up PR"
)]
pub struct Registered {
    pub dp_id: String,
    pub gateway_id: String,
    pub etcd_endpoint: String,
    pub heartbeat_url: String,
    pub telemetry_url: String,
    pub heartbeat_interval: Duration,
    pub telemetry_interval: Duration,
    pub ca_cert_path: PathBuf,
    pub client_cert_path: PathBuf,
    pub client_key_path: PathBuf,
}

/// HTTP client call + on-disk persistence in one shot. Split into
/// smaller helpers below so tests can drive each step in isolation.
pub async fn register_and_persist(cfg: &ManagedConfig) -> anyhow::Result<Registered> {
    let token = cfg
        .registration_token
        .as_deref()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow!("managed.registration_token must be set"))?;
    let base = cfg
        .cp_base_url
        .as_deref()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow!("managed.cp_base_url must be set"))?;

    let resp = call_register(base, token, &gather_host_info()?).await?;
    let (ca_path, cert_path, key_path) = persist_mtls(&cfg.mtls_dir, &resp.mtls).await?;
    persist_dp_id(&cfg.dp_id_file, &resp.dp_id).await?;

    Ok(Registered {
        dp_id: resp.dp_id,
        gateway_id: resp.gateway_id,
        etcd_endpoint: resp.cp_endpoints.etcd,
        heartbeat_url: resp.cp_endpoints.heartbeat,
        telemetry_url: resp.cp_endpoints.telemetry,
        heartbeat_interval: Duration::from_secs(resp.heartbeat_interval_seconds.max(1) as u64),
        telemetry_interval: Duration::from_secs(resp.telemetry_interval_seconds.max(1) as u64),
        ca_cert_path: ca_path,
        client_cert_path: cert_path,
        client_key_path: key_path,
    })
}

/// True when the mTLS bundle is already on disk. `main` uses this to
/// skip registration on subsequent boots.
pub fn bundle_exists(mtls_dir: impl AsRef<Path>) -> bool {
    let dir = mtls_dir.as_ref();
    ["ca.crt", "client.crt", "client.key"]
        .iter()
        .all(|name| dir.join(name).is_file())
}

// ---------------------------------------------------------------------
// HTTP client
// ---------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct RegisterRequest<'a> {
    hostname: &'a str,
    version: &'a str,
    os: &'a str,
    ip: &'a str,
}

#[derive(Debug, Deserialize)]
struct RegisterResponse {
    dp_id: String,
    gateway_id: String,
    #[serde(default)]
    heartbeat_interval_seconds: u32,
    #[serde(default)]
    telemetry_interval_seconds: u32,
    cp_endpoints: CpEndpoints,
    mtls: MTLSBundle,
}

#[derive(Debug, Deserialize)]
struct CpEndpoints {
    etcd: String,
    heartbeat: String,
    telemetry: String,
}

#[derive(Debug, Deserialize)]
struct MTLSBundle {
    ca_certificate: String,
    certificate: String,
    private_key: String,
}

async fn call_register(
    base_url: &str,
    token: &str,
    host: &HostInfo,
) -> anyhow::Result<RegisterResponse> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .user_agent(format!("aisix-dp/{}", env!("CARGO_PKG_VERSION")))
        .build()
        .context("build reqwest client")?;

    let url = format!("{}/dp/register", base_url.trim_end_matches('/'));
    let resp = client
        .post(&url)
        .bearer_auth(token)
        .json(&RegisterRequest {
            hostname: &host.hostname,
            version: env!("CARGO_PKG_VERSION"),
            os: &host.os,
            ip: &host.ip,
        })
        .send()
        .await
        .with_context(|| format!("POST {url}"))?;

    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        bail!(
            "/dp/register returned {} — {}",
            status,
            body.trim().chars().take(300).collect::<String>()
        );
    }

    resp.json::<RegisterResponse>()
        .await
        .context("decode /dp/register response")
}

// ---------------------------------------------------------------------
// Host info gathering
// ---------------------------------------------------------------------

#[derive(Debug)]
struct HostInfo {
    hostname: String,
    os: String,
    ip: String,
}

fn gather_host_info() -> anyhow::Result<HostInfo> {
    let hostname = hostname::get()
        .context("read hostname")?
        .into_string()
        .map_err(|_| anyhow!("hostname is not valid UTF-8"))?;
    let os = format!("{}/{}", std::env::consts::OS, std::env::consts::ARCH);
    // Best-effort outbound IP: we don't want to do a UDP dance here;
    // the CP uses `ip` purely for display. An empty string is accepted.
    let ip = String::new();
    Ok(HostInfo { hostname, os, ip })
}

// Small hostname crate would be nice — but keep dependency surface low.
// Implement inline via libc.
mod hostname {
    use std::ffi::{CStr, OsString};
    use std::os::unix::ffi::OsStringExt;

    pub fn get() -> std::io::Result<OsString> {
        const MAX: usize = 256;
        let mut buf = vec![0u8; MAX];
        // SAFETY: gethostname writes into buf up to MAX-1 bytes and
        // null-terminates; we read back the C string below.
        let rc = unsafe { libc_gethostname(buf.as_mut_ptr() as *mut _, buf.len()) };
        if rc != 0 {
            return Err(std::io::Error::last_os_error());
        }
        let cstr = CStr::from_bytes_until_nul(&buf)
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "no nul"))?;
        Ok(OsString::from_vec(cstr.to_bytes().to_vec()))
    }

    extern "C" {
        #[link_name = "gethostname"]
        fn libc_gethostname(name: *mut core::ffi::c_char, len: usize) -> core::ffi::c_int;
    }
}

// ---------------------------------------------------------------------
// On-disk persistence
// ---------------------------------------------------------------------

async fn persist_mtls(
    dir: &str,
    bundle: &MTLSBundle,
) -> anyhow::Result<(PathBuf, PathBuf, PathBuf)> {
    let dir = PathBuf::from(dir);
    tokio::fs::create_dir_all(&dir)
        .await
        .with_context(|| format!("create {}", dir.display()))?;

    let ca = dir.join("ca.crt");
    let cert = dir.join("client.crt");
    let key = dir.join("client.key");

    write_atomic(&ca, bundle.ca_certificate.as_bytes(), 0o600).await?;
    write_atomic(&cert, bundle.certificate.as_bytes(), 0o600).await?;
    write_atomic(&key, bundle.private_key.as_bytes(), 0o600).await?;

    Ok((ca, cert, key))
}

async fn persist_dp_id(path: &str, id: &str) -> anyhow::Result<()> {
    let path = PathBuf::from(path);
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("create {}", parent.display()))?;
    }
    write_atomic(&path, id.as_bytes(), 0o600).await
}

/// Write `data` to `path` atomically with the given mode. Strategy:
/// write to `path.tmp`, fsync, rename → path. A crash between write
/// and rename leaves either the old file or no file, never a
/// truncated file.
#[cfg(unix)]
async fn write_atomic(path: &Path, data: &[u8], mode: u32) -> anyhow::Result<()> {
    // tokio::fs::OpenOptions exposes its own `.mode()` method on Unix;
    // `std::os::unix::fs::OpenOptionsExt` is not needed here.
    use tokio::io::AsyncWriteExt;

    let tmp = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .map(|e| e.to_string_lossy().into_owned())
            .unwrap_or_default()
    ));
    {
        let mut f = tokio::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(mode)
            .open(&tmp)
            .await
            .with_context(|| format!("open {} for write", tmp.display()))?;
        f.write_all(data)
            .await
            .with_context(|| format!("write {}", tmp.display()))?;
        f.sync_all()
            .await
            .with_context(|| format!("fsync {}", tmp.display()))?;
    }
    tokio::fs::rename(&tmp, path)
        .await
        .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

// Managed mode is Linux-only (prd-09 assumes the DP runs inside the
// user's infra on a Unix box). Provide a stub for cfg(not(unix)) so
// cross-builds still link; the runtime path is gated by managed.enabled.
#[cfg(not(unix))]
async fn write_atomic(_path: &Path, _data: &[u8], _mode: u32) -> anyhow::Result<()> {
    anyhow::bail!("managed mode is only supported on Unix")
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn fake_response_body() -> serde_json::Value {
        serde_json::json!({
            "dp_id": "dp_7kq9mxplwrtnvb",
            "gateway_id": "aigg_abc123",
            "org_id": "org_foo",
            "heartbeat_interval_seconds": 15,
            "telemetry_interval_seconds": 30,
            "cp_endpoints": {
                "etcd": "etcd.aisix.cloud:2379",
                "heartbeat": "https://api.aisix.cloud/dp/heartbeat",
                "telemetry": "https://api.aisix.cloud/dp/telemetry"
            },
            "mtls": {
                "ca_certificate": "-----BEGIN CERTIFICATE-----\nCA\n-----END CERTIFICATE-----\n",
                "certificate":    "-----BEGIN CERTIFICATE-----\nCLIENT\n-----END CERTIFICATE-----\n",
                "private_key":    "-----BEGIN PRIVATE KEY-----\nKEY\n-----END PRIVATE KEY-----\n"
            }
        })
    }

    #[tokio::test]
    async fn register_and_persist_happy_path() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/dp/register"))
            .and(header("Authorization", "Bearer tok-xyz"))
            .respond_with(ResponseTemplate::new(200).set_body_json(fake_response_body()))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let mtls_dir = dir.path().join("mtls");
        let dp_id_file = dir.path().join("dp_id");

        let cfg = ManagedConfig {
            enabled: true,
            registration_token: Some("tok-xyz".into()),
            cp_base_url: Some(server.uri()),
            mtls_dir: mtls_dir.to_string_lossy().into_owned(),
            dp_id_file: dp_id_file.to_string_lossy().into_owned(),
            snapshot_cache_path: String::new(),
        };

        let out = register_and_persist(&cfg).await.expect("register");

        assert_eq!(out.dp_id, "dp_7kq9mxplwrtnvb");
        assert_eq!(out.gateway_id, "aigg_abc123");
        assert_eq!(out.etcd_endpoint, "etcd.aisix.cloud:2379");
        assert_eq!(out.heartbeat_interval, Duration::from_secs(15));

        // Bundle on disk with the right contents.
        let ca = std::fs::read_to_string(&out.ca_cert_path).unwrap();
        let cert = std::fs::read_to_string(&out.client_cert_path).unwrap();
        let key = std::fs::read_to_string(&out.client_key_path).unwrap();
        assert!(ca.contains("CA"));
        assert!(cert.contains("CLIENT"));
        assert!(key.contains("KEY"));

        // Files are 0600.
        for p in [
            &out.ca_cert_path,
            &out.client_cert_path,
            &out.client_key_path,
        ] {
            let m = std::fs::metadata(p).unwrap();
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                m.permissions().mode() & 0o777,
                0o600,
                "file {p:?} perms wrong"
            );
        }

        // dp_id persisted.
        let id = std::fs::read_to_string(&dp_id_file).unwrap();
        assert_eq!(id, "dp_7kq9mxplwrtnvb");
    }

    #[tokio::test]
    async fn register_propagates_4xx_body() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/dp/register"))
            .respond_with(ResponseTemplate::new(401).set_body_json(serde_json::json!({
                "error": {"code": "INVALID_TOKEN", "message": "token already consumed"}
            })))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let cfg = ManagedConfig {
            enabled: true,
            registration_token: Some("spent".into()),
            cp_base_url: Some(server.uri()),
            mtls_dir: dir.path().join("mtls").to_string_lossy().into_owned(),
            dp_id_file: dir.path().join("dp_id").to_string_lossy().into_owned(),
            snapshot_cache_path: String::new(),
        };

        let err = register_and_persist(&cfg).await.unwrap_err();
        let s = format!("{err:#}");
        // The exact CP error body must surface so operators can
        // tell "consumed token" from "revoked token" from the logs.
        assert!(s.contains("401"), "expected status in error: {s}");
        assert!(
            s.contains("INVALID_TOKEN") || s.contains("consumed"),
            "expected upstream body in error: {s}"
        );
    }

    #[tokio::test]
    async fn register_requires_token_and_url() {
        let err = register_and_persist(&ManagedConfig {
            enabled: true,
            registration_token: None,
            cp_base_url: Some("https://x".into()),
            ..Default::default()
        })
        .await
        .unwrap_err();
        assert!(err.to_string().contains("registration_token"));

        let err = register_and_persist(&ManagedConfig {
            enabled: true,
            registration_token: Some("t".into()),
            cp_base_url: None,
            ..Default::default()
        })
        .await
        .unwrap_err();
        assert!(err.to_string().contains("cp_base_url"));
    }

    #[test]
    fn bundle_exists_detects_complete_set() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!bundle_exists(dir.path()));

        for name in ["ca.crt", "client.crt"] {
            std::fs::write(dir.path().join(name), "x").unwrap();
        }
        // Incomplete — client.key missing.
        assert!(!bundle_exists(dir.path()));

        std::fs::write(dir.path().join("client.key"), "x").unwrap();
        assert!(bundle_exists(dir.path()));
    }
}
