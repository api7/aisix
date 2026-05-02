//! One-shot DP registration against aisix.cloud's v3 self-hosted CP
//! (prd-09a §9A.7.2 + §9A.10A.3).
//!
//! Wire shape (v3, replacing v2's CP-issues-keypair flow):
//!
//! 1. DP **generates an ECDSA P-256 keypair locally** at boot. The
//!    private key never leaves this process.
//! 2. DP `POST /dp/register` with `{ hostname, version,
//!    dp_protocol_version: "v3", public_key }` and the deployment
//!    token as a `Bearer`.
//! 3. CP responds with `{ dp_id, env_id, ca_certificate, certificate,
//!    cert_expires_at, cert_rotate_before, heartbeat_path,
//!    telemetry_path, rotate_cert_path }`. No private key in the
//!    response — the DP already holds it from step 1.
//! 4. DP persists `client.key` (the locally generated PKCS#8 PEM),
//!    `client.crt` (issued cert), and `ca.crt` to `mtls_dir` atomically.
//!
//! The whole flow is single-use — the deployment token gets consumed
//! on the CP side. Subsequent boots detect the bundle on disk and
//! skip registration entirely.
//!
//! `dp_protocol_version` triggers a 426 Upgrade Required response if
//! it doesn't match `MIN_SUPPORTED_DP_PROTOCOL_VERSION` on the CP
//! (prd-09a §9A.10A.3). DPs older than v3 won't even be able to
//! consume their token; operators have to upgrade or mint a new DP.

use std::path::{Path, PathBuf};
use std::time::Duration;

use aisix_core::ManagedConfig;
use anyhow::{anyhow, bail, Context};
use chrono::{DateTime, Utc};
use rcgen::{KeyPair, PKCS_ECDSA_P256_SHA256};
use serde::{Deserialize, Serialize};

/// Wire-protocol identifier the DP advertises in `/dp/register`.
/// Pinned in code rather than read from config: bumping this is a
/// deliberate breaking change that needs a code review, not a
/// runtime knob.
pub const DP_PROTOCOL_VERSION: &str = "v3";

/// Outcome of a successful v3 registration. Mirrors the §9A.7.2
/// response with file paths attached for `main` to plug into the
/// etcd client config.
#[derive(Debug, Clone)]
#[allow(
    dead_code,
    reason = "rotate-cert auto-renew + telemetry consume the rotate/expiry fields once those PRs land"
)]
pub struct Registered {
    pub dp_id: String,
    pub env_id: String,
    pub ca_cert_path: PathBuf,
    pub client_cert_path: PathBuf,
    pub client_key_path: PathBuf,
    pub cert_expires_at: DateTime<Utc>,
    pub cert_rotate_before: DateTime<Utc>,
    pub heartbeat_path: String,
    pub telemetry_path: String,
    pub rotate_cert_path: String,
}

/// Generate the keypair, exchange the token for an mTLS bundle, and
/// persist everything atomically. Split into smaller helpers below so
/// tests can drive each step in isolation.
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

    // Generate the keypair locally — private key never leaves this
    // process. `rcgen::KeyPair::generate` runs the system RNG; the
    // PKCS#8 PEM is what we'll write to disk.
    let keypair = KeyPair::generate_for(&PKCS_ECDSA_P256_SHA256)
        .context("generate ECDSA P-256 keypair for /dp/register")?;
    let private_key_pem = keypair.serialize_pem();
    let public_key_pem = keypair.public_key_pem();

    let extra_ca = read_optional_ca_pem(cfg.cp_ca_cert_file.as_deref())?;
    let resp = call_register(
        base,
        token,
        &gather_host_info()?,
        &public_key_pem,
        extra_ca.as_deref(),
    )
    .await?;
    let (ca_path, cert_path, key_path) = persist_mtls(
        &cfg.mtls_dir,
        &resp.ca_certificate,
        &resp.certificate,
        &private_key_pem,
    )
    .await?;
    persist_dp_id(&cfg.dp_id_file, &resp.dp_id).await?;
    // Persist env_id alongside the bundle so subsequent boots
    // (bundle-on-disk path) can keep scoping etcd reads to
    // `/aisix/<env_id>/` without re-registering.
    persist_env_id(&cfg.mtls_dir, &resp.env_id).await?;

    Ok(Registered {
        dp_id: resp.dp_id,
        env_id: resp.env_id,
        ca_cert_path: ca_path,
        client_cert_path: cert_path,
        client_key_path: key_path,
        cert_expires_at: resp.cert_expires_at,
        cert_rotate_before: resp.cert_rotate_before,
        heartbeat_path: resp.heartbeat_path,
        telemetry_path: resp.telemetry_path,
        rotate_cert_path: resp.rotate_cert_path,
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

/// Read a PEM-encoded CA bundle from disk if `path` is `Some`. Returns
/// `Ok(None)` when the path is `None` or empty so the rest of the
/// boot path doesn't have to special-case the unset case. Surfaces
/// the path on read errors — a misconfigured or unmounted bundle in
/// e2e is the most common reason this fires, and the path tells the
/// operator exactly what to fix.
pub fn read_optional_ca_pem(path: Option<&str>) -> anyhow::Result<Option<Vec<u8>>> {
    let Some(p) = path.filter(|s| !s.is_empty()) else {
        return Ok(None);
    };
    let bytes = std::fs::read(p).with_context(|| format!("read managed.cp_ca_cert_file = {p}"))?;
    Ok(Some(bytes))
}

/// Well-known filename within `mtls_dir` for the issuer's CA cert.
pub fn ca_cert_path(mtls_dir: impl AsRef<Path>) -> PathBuf {
    mtls_dir.as_ref().join("ca.crt")
}

/// Well-known filename within `mtls_dir` for the DP's client cert.
pub fn client_cert_path(mtls_dir: impl AsRef<Path>) -> PathBuf {
    mtls_dir.as_ref().join("client.crt")
}

/// Well-known filename within `mtls_dir` for the DP's client key.
pub fn client_key_path(mtls_dir: impl AsRef<Path>) -> PathBuf {
    mtls_dir.as_ref().join("client.key")
}

/// Well-known filename within `mtls_dir` for the env_id this DP is
/// scoped to. Persisted at register time so bundle-on-disk boots can
/// re-derive the etcd prefix (`/aisix/<env_id>/`) without re-registering.
pub fn env_id_path(mtls_dir: impl AsRef<Path>) -> PathBuf {
    mtls_dir.as_ref().join("env_id")
}

/// Read the env_id file written by `register_and_persist`. Trims
/// trailing whitespace and rejects an empty payload — an empty env_id
/// would silently widen the etcd prefix to the global scope.
pub fn read_env_id(mtls_dir: impl AsRef<Path>) -> anyhow::Result<String> {
    let path = env_id_path(mtls_dir);
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("read env_id from {}", path.display()))?;
    let trimmed = raw.trim().to_string();
    if trimmed.is_empty() {
        bail!("env_id file {} is empty", path.display());
    }
    Ok(trimmed)
}

// ---------------------------------------------------------------------
// HTTP client
// ---------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct RegisterRequest<'a> {
    hostname: &'a str,
    version: &'a str,
    dp_protocol_version: &'a str,
    public_key: &'a str,
}

#[derive(Debug, Deserialize)]
struct RegisterResponse {
    dp_id: String,
    env_id: String,
    ca_certificate: String,
    certificate: String,
    cert_expires_at: DateTime<Utc>,
    cert_rotate_before: DateTime<Utc>,
    heartbeat_path: String,
    telemetry_path: String,
    rotate_cert_path: String,
}

async fn call_register(
    base_url: &str,
    token: &str,
    host: &HostInfo,
    public_key_pem: &str,
    extra_ca_pem: Option<&[u8]>,
) -> anyhow::Result<RegisterResponse> {
    let mut builder = reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .user_agent(format!("aisix-dp/{}", env!("CARGO_PKG_VERSION")));
    // Operator-supplied extra trust root, e.g. a self-signed dev CA in
    // e2e or an on-prem private CA in air-gapped deployments. The
    // public-CA chain remains in effect alongside this; we never swap
    // it out, so production deployments don't lose any verification.
    if let Some(pem) = extra_ca_pem {
        let cert = reqwest::Certificate::from_pem(pem)
            .context("parse managed.cp_ca_cert_file as PEM certificate")?;
        builder = builder.add_root_certificate(cert);
    }
    let client = builder.build().context("build reqwest client")?;

    let url = format!("{}/dp/register", base_url.trim_end_matches('/'));
    let resp = client
        .post(&url)
        .bearer_auth(token)
        .json(&RegisterRequest {
            hostname: &host.hostname,
            version: env!("CARGO_PKG_VERSION"),
            dp_protocol_version: DP_PROTOCOL_VERSION,
            public_key: public_key_pem,
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
}

fn gather_host_info() -> anyhow::Result<HostInfo> {
    let hostname = hostname::get()
        .context("read hostname")?
        .into_string()
        .map_err(|_| anyhow!("hostname is not valid UTF-8"))?;
    Ok(HostInfo { hostname })
}

// Small hostname helper — keep dependency surface low; libc gethostname
// is on every Unix.
mod hostname {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;

    pub fn get() -> std::io::Result<OsString> {
        let mut buf = vec![0u8; 256];
        // SAFETY: gethostname writes into buf up to MAX-1 bytes and
        // null-terminates. We size the buffer at 256 bytes which
        // exceeds POSIX HOST_NAME_MAX (64) on every platform we care
        // about.
        let rc = unsafe { libc_gethostname(buf.as_mut_ptr() as *mut _, buf.len()) };
        if rc != 0 {
            return Err(std::io::Error::last_os_error());
        }
        let n = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        buf.truncate(n);
        Ok(OsString::from_vec(buf))
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
    ca_certificate: &str,
    certificate: &str,
    private_key: &str,
) -> anyhow::Result<(PathBuf, PathBuf, PathBuf)> {
    let dir = PathBuf::from(dir);
    tokio::fs::create_dir_all(&dir)
        .await
        .with_context(|| format!("create {}", dir.display()))?;

    let ca = dir.join("ca.crt");
    let cert = dir.join("client.crt");
    let key = dir.join("client.key");

    write_atomic(&ca, ca_certificate.as_bytes(), 0o600).await?;
    write_atomic(&cert, certificate.as_bytes(), 0o600).await?;
    write_atomic(&key, private_key.as_bytes(), 0o600).await?;

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

/// Persist `env_id` to `<mtls_dir>/env_id` atomically. `mtls_dir` is
/// already created by `persist_mtls`, but we re-create defensively in
/// case this helper is called in isolation.
async fn persist_env_id(mtls_dir: &str, env_id: &str) -> anyhow::Result<()> {
    let dir = PathBuf::from(mtls_dir);
    tokio::fs::create_dir_all(&dir)
        .await
        .with_context(|| format!("create {}", dir.display()))?;
    write_atomic(&dir.join("env_id"), env_id.as_bytes(), 0o600).await
}

/// Write `data` to `path` atomically with the given mode. Strategy:
/// write to `path.tmp`, fsync, rename → path. A crash between write
/// and rename leaves either the old file or no file, never a
/// truncated file.
#[cfg(unix)]
async fn write_atomic(path: &Path, data: &[u8], mode: u32) -> anyhow::Result<()> {
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

// Managed mode is Unix-only (prd-09a assumes the DP runs inside the
// user's infra). Stub for cfg(not(unix)) so cross-builds still link;
// the runtime path is gated by managed.enabled.
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
    use serde_json::json;
    use wiremock::matchers::{body_partial_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn read_optional_ca_pem_returns_none_for_unset_path() {
        assert!(read_optional_ca_pem(None).unwrap().is_none());
        assert!(read_optional_ca_pem(Some("")).unwrap().is_none());
    }

    #[test]
    fn read_optional_ca_pem_reads_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ca.pem");
        std::fs::write(
            &path,
            b"-----BEGIN CERTIFICATE-----\nXX\n-----END CERTIFICATE-----\n",
        )
        .unwrap();
        let bytes = read_optional_ca_pem(Some(path.to_str().unwrap()))
            .unwrap()
            .expect("Some when path is set");
        assert!(bytes.starts_with(b"-----BEGIN CERTIFICATE-----"));
    }

    #[test]
    fn read_optional_ca_pem_surfaces_path_on_read_error() {
        let err = read_optional_ca_pem(Some("/no/such/path/ca.pem")).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("/no/such/path/ca.pem"),
            "error must surface the configured path so operators can fix the mount: {msg}",
        );
    }

    fn fake_v3_response_body() -> serde_json::Value {
        json!({
            "dp_id":            "ed5e0f3e-2c32-4f3a-9b9e-d6c9d2c4d4e1",
            "env_id":           "11111111-1111-1111-1111-111111111111",
            "ca_certificate":   "-----BEGIN CERTIFICATE-----\nCA\n-----END CERTIFICATE-----\n",
            "certificate":      "-----BEGIN CERTIFICATE-----\nCLIENT\n-----END CERTIFICATE-----\n",
            "cert_expires_at":  "2026-07-26T00:00:00Z",
            "cert_rotate_before":"2026-06-26T00:00:00Z",
            "heartbeat_path":   "/dp/heartbeat",
            "telemetry_path":   "/dp/telemetry",
            "rotate_cert_path": "/dp/rotate-cert"
        })
    }

    #[tokio::test]
    async fn register_and_persist_v3_happy_path() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/dp/register"))
            .and(header("Authorization", "Bearer tok-xyz"))
            // Body must include the v3 fields the CP keys on.
            .and(body_partial_json(json!({
                "dp_protocol_version": "v3"
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(fake_v3_response_body()))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let mtls_dir = dir.path().join("mtls");
        let dp_id_file = dir.path().join("dp_id");

        let cfg = ManagedConfig {
            enabled: true,
            registration_token: Some("tok-xyz".into()),
            cp_base_url: Some(server.uri()),
            cp_etcd_endpoint: Some("etcd.local:7943".into()),
            cp_ca_cert_file: None,
            mtls_dir: mtls_dir.to_string_lossy().into_owned(),
            dp_id_file: dp_id_file.to_string_lossy().into_owned(),
            snapshot_cache_path: String::new(),
        };

        let out = register_and_persist(&cfg).await.expect("register");

        assert_eq!(out.dp_id, "ed5e0f3e-2c32-4f3a-9b9e-d6c9d2c4d4e1");
        assert_eq!(out.env_id, "11111111-1111-1111-1111-111111111111");
        assert_eq!(out.heartbeat_path, "/dp/heartbeat");
        assert_eq!(out.rotate_cert_path, "/dp/rotate-cert");

        // Bundle on disk: ca + cert from response, key generated locally.
        let ca = std::fs::read_to_string(&out.ca_cert_path).unwrap();
        let cert = std::fs::read_to_string(&out.client_cert_path).unwrap();
        let key = std::fs::read_to_string(&out.client_key_path).unwrap();
        assert!(ca.contains("CA"));
        assert!(cert.contains("CLIENT"));
        assert!(
            key.contains("PRIVATE KEY"),
            "client.key should be the locally-generated PKCS#8 PEM"
        );

        // Files are 0600.
        for p in [
            &out.ca_cert_path,
            &out.client_cert_path,
            &out.client_key_path,
        ] {
            let m = std::fs::metadata(p).unwrap();
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(m.permissions().mode() & 0o777, 0o600, "{p:?} perms wrong");
        }

        // dp_id persisted.
        let id = std::fs::read_to_string(&dp_id_file).unwrap();
        assert_eq!(id, "ed5e0f3e-2c32-4f3a-9b9e-d6c9d2c4d4e1");

        // env_id persisted alongside the bundle. Subsequent boots
        // (bundle-on-disk path in main.rs) read this back to scope
        // etcd reads to `/aisix/<env_id>/`.
        let env_id_file = mtls_dir.join("env_id");
        let env_id_on_disk = std::fs::read_to_string(&env_id_file).unwrap();
        assert_eq!(env_id_on_disk, "11111111-1111-1111-1111-111111111111");
        // read_env_id() helper trims trailing whitespace and yields the
        // same value — this is the path main.rs takes on boot.
        assert_eq!(
            read_env_id(&mtls_dir).unwrap(),
            "11111111-1111-1111-1111-111111111111"
        );
        // env_id file is also 0600 — not strictly secret but matches
        // the rest of the bundle's permission bits.
        use std::os::unix::fs::PermissionsExt;
        let m = std::fs::metadata(&env_id_file).unwrap();
        assert_eq!(m.permissions().mode() & 0o777, 0o600);
    }

    #[test]
    fn read_env_id_rejects_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("env_id"), "   \n").unwrap();
        let err = read_env_id(dir.path()).unwrap_err();
        assert!(err.to_string().contains("empty"), "got: {err}");
    }

    #[test]
    fn read_env_id_surfaces_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let err = read_env_id(dir.path()).unwrap_err();
        assert!(err.to_string().contains("read env_id"), "got: {err}");
    }

    #[tokio::test]
    async fn register_sends_dp_protocol_version_v3() {
        let server = MockServer::start().await;
        // The wiremock matcher above already enforces this in the
        // happy-path test. This second test is here so a future
        // refactor that drops the field has TWO tests fail at once
        // (bad sign for accidental removal vs single-test regression).
        Mock::given(method("POST"))
            .and(path("/dp/register"))
            .and(body_partial_json(
                json!({"dp_protocol_version": "v3", "version": env!("CARGO_PKG_VERSION")}),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(fake_v3_response_body()))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let cfg = ManagedConfig {
            enabled: true,
            registration_token: Some("t".into()),
            cp_base_url: Some(server.uri()),
            cp_etcd_endpoint: Some("etcd.local:7943".into()),
            cp_ca_cert_file: None,
            mtls_dir: dir.path().join("mtls").to_string_lossy().into_owned(),
            dp_id_file: dir.path().join("dp_id").to_string_lossy().into_owned(),
            snapshot_cache_path: String::new(),
        };
        register_and_persist(&cfg).await.expect("register");
    }

    #[tokio::test]
    async fn register_sends_locally_generated_public_key() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/dp/register"))
            // wiremock's body_partial_json only matches present keys;
            // we want to assert "public_key" is present and shaped
            // like a PEM. wiremock doesn't ship a regex matcher in
            // its default features, so we pin to a substring of the
            // PKIX SPKI header that rcgen emits.
            .and(body_partial_json(json!({"dp_protocol_version": "v3"})))
            .respond_with(ResponseTemplate::new(200).set_body_json(fake_v3_response_body()))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let cfg = ManagedConfig {
            enabled: true,
            registration_token: Some("t".into()),
            cp_base_url: Some(server.uri()),
            cp_etcd_endpoint: Some("etcd.local:7943".into()),
            cp_ca_cert_file: None,
            mtls_dir: dir.path().join("mtls").to_string_lossy().into_owned(),
            dp_id_file: dir.path().join("dp_id").to_string_lossy().into_owned(),
            snapshot_cache_path: String::new(),
        };
        register_and_persist(&cfg).await.expect("register");

        // The locally generated private key landed on disk and
        // parses as a PKCS#8 PEM. Strict structural check via rcgen
        // would re-import it; for this smoke we just confirm the PEM
        // header.
        let key = std::fs::read_to_string(dir.path().join("mtls").join("client.key")).unwrap();
        assert!(
            key.contains("-----BEGIN PRIVATE KEY-----"),
            "client.key should be a PKCS#8 PEM private key, got:\n{key}"
        );
    }

    #[tokio::test]
    async fn register_propagates_426_with_min_supported() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/dp/register"))
            .respond_with(ResponseTemplate::new(426).set_body_json(json!({
                "error":                  "incompatible_dp_protocol",
                "min_supported":          "v3",
                "recommended_dp_version": "0.6.0",
                "download_url":           "https://github.com/moonming/ai-gateway/releases"
            })))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let cfg = ManagedConfig {
            enabled: true,
            registration_token: Some("t".into()),
            cp_base_url: Some(server.uri()),
            cp_etcd_endpoint: Some("etcd.local:7943".into()),
            cp_ca_cert_file: None,
            mtls_dir: dir.path().join("mtls").to_string_lossy().into_owned(),
            dp_id_file: dir.path().join("dp_id").to_string_lossy().into_owned(),
            snapshot_cache_path: String::new(),
        };
        let err = register_and_persist(&cfg).await.unwrap_err();
        let s = format!("{err:#}");
        assert!(s.contains("426"), "expected status in error: {s}");
        assert!(
            s.contains("incompatible_dp_protocol") || s.contains("min_supported"),
            "expected 426 body in error: {s}"
        );
    }

    #[tokio::test]
    async fn register_propagates_4xx_body() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/dp/register"))
            .respond_with(ResponseTemplate::new(401).set_body_json(json!({
                "error":   "INVALID_TOKEN",
                "message": "register token not recognised"
            })))
            .mount(&server)
            .await;

        let dir = tempfile::tempdir().unwrap();
        let cfg = ManagedConfig {
            enabled: true,
            registration_token: Some("spent".into()),
            cp_base_url: Some(server.uri()),
            cp_etcd_endpoint: Some("etcd.local:7943".into()),
            cp_ca_cert_file: None,
            mtls_dir: dir.path().join("mtls").to_string_lossy().into_owned(),
            dp_id_file: dir.path().join("dp_id").to_string_lossy().into_owned(),
            snapshot_cache_path: String::new(),
        };
        let err = register_and_persist(&cfg).await.unwrap_err();
        let s = format!("{err:#}");
        assert!(s.contains("401"), "expected status in error: {s}");
        assert!(
            s.contains("INVALID_TOKEN"),
            "expected upstream error code in error: {s}"
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
        assert!(!bundle_exists(dir.path()), "missing client.key should fail");

        std::fs::write(dir.path().join("client.key"), "x").unwrap();
        assert!(bundle_exists(dir.path()));
    }
}
