//! Pre-provisioned mTLS bundle bootstrap path (api7ee parity).
//!
//! Counterpart to `register.rs`. When the operator mints a cert via
//! the dashboard's `CertIssueCard` (see AISIX-Cloud cp-api's
//! `POST /api/environments/:env_id/gateway_certificates`), the
//! resulting cert + key + CA PEM trio is inlined into the DP's
//! environment vars (or written to disk for systemd / k8s Secret
//! mounts). The DP boots, materialises the bundle to `mtls_dir`,
//! parses `env_id` out of the leaf cert's URI SAN, and proceeds
//! straight to the etcd connect — no `/dp/register` round-trip.
//!
//! Wire shape, three required PEMs:
//!
//!   AISIX_MANAGED__CP_CERT_PEM   — leaf certificate
//!   AISIX_MANAGED__CP_KEY_PEM    — SEC1 EC private key paired with
//!                                  the leaf
//!   AISIX_MANAGED__CP_CA_PEM     — CA cert the DP installs as the
//!                                  trust anchor for dp-manager mTLS
//!
//! `_FILE` variants take the same content but load from disk; useful
//! for systemd hosts where pasting multi-line PEMs into env vars is
//! awkward. Inline and file variants are mutually exclusive per
//! triplet; mixing gets a hard error at boot rather than a silent
//! pick-one.
//!
//! The cert's URI SAN encodes `x-aisix://env/<env_id>` per cp-api's
//! `internal/dpmgr/certmgr/issue.go`. We parse it out and plant it
//! into `cfg.etcd.env_id` so `effective_prefix()` returns
//! `/aisix/<env_id>/` — the same scoping the legacy register-derived
//! `r.env_id` used to populate.

use std::path::{Path, PathBuf};

use aisix_core::ManagedConfig;
use anyhow::{anyhow, bail, Context};
use x509_parser::extensions::GeneralName;
use x509_parser::pem::parse_x509_pem;
use x509_parser::prelude::{FromDer, X509Certificate};

/// Outcome of materialising a pre-provisioned bundle. Mirrors
/// `register::Registered` minus the metadata cp-api would have
/// returned (we don't have a register response to crib from).
#[derive(Debug, Clone)]
pub struct Provisioned {
    /// env_id parsed from the leaf cert's URI SAN
    /// (`x-aisix://env/<env_id>`). Plumbed into `cfg.etcd.env_id` so
    /// every Range/Watch lands on `/aisix/<env_id>/`.
    pub env_id: String,
    /// dp_id parsed from the leaf cert's URI SAN
    /// (`x-aisix://dp/<dp_id>`). Used for heartbeat payload + log
    /// correlation.
    pub dp_id: String,
    /// On-disk path to the materialised CA cert.
    pub ca_cert_path: PathBuf,
    /// On-disk path to the materialised leaf cert.
    pub client_cert_path: PathBuf,
    /// On-disk path to the materialised leaf private key.
    pub client_key_path: PathBuf,
}

/// Load + materialise a pre-provisioned bundle. Reads PEM contents
/// from either the inline env-var fields (`cp_cert_pem` etc.) or
/// the file-path fields (`cp_cert_file` etc.), writes them
/// atomically to `mtls_dir`, parses the leaf cert's SAN to extract
/// (env_id, dp_id), and returns paths the caller plumbs into
/// `cfg.etcd.tls`.
///
/// Idempotent: re-running with the same inputs over an existing
/// `mtls_dir` is safe; the atomic-write helper truncates+rewrites
/// the targets.
pub async fn provision(cfg: &ManagedConfig) -> anyhow::Result<Provisioned> {
    let cert_pem = load_pem("cert", cfg.cp_cert_pem.as_deref(), cfg.cp_cert_file.as_deref()).await?;
    let key_pem = load_pem("key", cfg.cp_key_pem.as_deref(), cfg.cp_key_file.as_deref()).await?;
    let ca_pem = load_pem("ca", cfg.cp_ca_pem.as_deref(), cfg.cp_ca_file.as_deref()).await?;

    let (env_id, dp_id) = parse_san_uris(&cert_pem)
        .with_context(|| "parse env_id + dp_id from leaf cert SAN URIs")?;
    if env_id.is_empty() {
        bail!("leaf cert is missing the `x-aisix://env/<uuid>` SAN URI");
    }
    if dp_id.is_empty() {
        bail!("leaf cert is missing the `x-aisix://dp/<uuid>` SAN URI");
    }

    let mtls_dir = Path::new(&cfg.mtls_dir);
    if !mtls_dir.exists() {
        tokio::fs::create_dir_all(mtls_dir)
            .await
            .with_context(|| format!("create mtls_dir at {mtls_dir:?}"))?;
    }
    let ca_path = ca_cert_path(mtls_dir);
    let cert_path = client_cert_path(mtls_dir);
    let key_path = client_key_path(mtls_dir);
    write_atomic(&ca_path, ca_pem.as_bytes(), 0o644)
        .await
        .with_context(|| format!("write CA cert to {ca_path:?}"))?;
    write_atomic(&cert_path, cert_pem.as_bytes(), 0o644)
        .await
        .with_context(|| format!("write client cert to {cert_path:?}"))?;
    write_atomic(&key_path, key_pem.as_bytes(), 0o600)
        .await
        .with_context(|| format!("write client key to {key_path:?}"))?;

    Ok(Provisioned {
        env_id,
        dp_id,
        ca_cert_path: ca_path,
        client_cert_path: cert_path,
        client_key_path: key_path,
    })
}

/// Resolve PEM content from either the inline `pem_arg` (env-var
/// form) or the `file_arg` (path-on-disk form). Setting both is a
/// boot-time error — the operator made a mistake we should not
/// silently paper over.
async fn load_pem(
    role: &str,
    pem_arg: Option<&str>,
    file_arg: Option<&str>,
) -> anyhow::Result<String> {
    let inline = pem_arg.filter(|s| !s.is_empty());
    let file = file_arg.filter(|s| !s.is_empty());
    match (inline, file) {
        (Some(p), None) => Ok(p.to_string()),
        (None, Some(path)) => {
            let bytes = tokio::fs::read(path)
                .await
                .with_context(|| format!("read {role} PEM from {path:?}"))?;
            String::from_utf8(bytes).with_context(|| format!("{role} PEM at {path:?} is not UTF-8"))
        }
        (Some(_), Some(_)) => bail!(
            "managed config sets BOTH cp_{role}_pem and cp_{role}_file — pick one (inline or path)",
        ),
        (None, None) => bail!("managed config is missing cp_{role}_pem / cp_{role}_file"),
    }
}

/// Parse a leaf cert's URI SANs and return `(env_id, dp_id)`. The
/// cert manager (`internal/dpmgr/certmgr/issue.go`) encodes them
/// as:
///
///   x-aisix://env/<env_id>
///   x-aisix://dp/<dp_id>
fn parse_san_uris(pem: &str) -> anyhow::Result<(String, String)> {
    let (_, p) = parse_x509_pem(pem.as_bytes())
        .map_err(|e| anyhow!("decode PEM: {e:?}"))?;
    let (_, cert) = X509Certificate::from_der(&p.contents)
        .map_err(|e| anyhow!("parse X.509: {e:?}"))?;
    let san = cert
        .subject_alternative_name()
        .map_err(|e| anyhow!("read SAN ext: {e:?}"))?
        .ok_or_else(|| anyhow!("leaf cert has no SubjectAlternativeName extension"))?;
    let mut env_id = String::new();
    let mut dp_id = String::new();
    for name in &san.value.general_names {
        if let GeneralName::URI(uri) = name {
            if let Some(rest) = uri.strip_prefix("x-aisix://env/") {
                env_id = rest.to_string();
            } else if let Some(rest) = uri.strip_prefix("x-aisix://dp/") {
                dp_id = rest.to_string();
            }
        }
    }
    Ok((env_id, dp_id))
}

pub fn ca_cert_path(mtls_dir: impl AsRef<Path>) -> PathBuf {
    mtls_dir.as_ref().join("ca.crt")
}
pub fn client_cert_path(mtls_dir: impl AsRef<Path>) -> PathBuf {
    mtls_dir.as_ref().join("client.crt")
}
pub fn client_key_path(mtls_dir: impl AsRef<Path>) -> PathBuf {
    mtls_dir.as_ref().join("client.key")
}

#[cfg(unix)]
async fn write_atomic(path: &Path, data: &[u8], mode: u32) -> anyhow::Result<()> {
    use tokio::io::AsyncWriteExt;

    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("path {path:?} has no parent dir"))?;
    let tmp = parent.join(format!(
        ".aisix-tmp-{}.{}",
        path.file_name().and_then(|n| n.to_str()).unwrap_or("blob"),
        std::process::id()
    ));
    {
        let mut f = tokio::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(mode)
            .open(&tmp)
            .await
            .with_context(|| format!("open tmp file {tmp:?}"))?;
        f.write_all(data).await?;
        f.flush().await?;
        f.sync_all().await?;
    }
    tokio::fs::rename(&tmp, path)
        .await
        .with_context(|| format!("atomic rename {tmp:?} → {path:?}"))?;
    Ok(())
}

#[cfg(not(unix))]
async fn write_atomic(_path: &Path, _data: &[u8], _mode: u32) -> anyhow::Result<()> {
    bail!("non-Unix file write is not implemented; managed-mode bundle bootstrap requires Unix")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rcgen::{
        CertificateParams, DistinguishedName, IsCa, KeyIdMethod, KeyPair, KeyUsagePurpose, SanType,
    };
    use tempfile::TempDir;

    /// Generate a self-signed cert with the same SAN URI shape cp-
    /// api's certmgr signs, so we can test the SAN parsing path
    /// without touching cp-api at all.
    fn synth_leaf(env_id: &str, dp_id: &str) -> (String, String) {
        let mut params = CertificateParams::default();
        params.distinguished_name = DistinguishedName::new();
        params.is_ca = IsCa::NoCa;
        params.key_usages = vec![KeyUsagePurpose::DigitalSignature];
        params.key_identifier_method = KeyIdMethod::Sha256;
        params.subject_alt_names = vec![
            SanType::URI(format!("x-aisix://env/{env_id}").try_into().unwrap()),
            SanType::URI(format!("x-aisix://dp/{dp_id}").try_into().unwrap()),
        ];
        let key = KeyPair::generate().unwrap();
        let cert = params.self_signed(&key).unwrap();
        (cert.pem(), key.serialize_pem())
    }

    #[test]
    fn parse_san_uris_extracts_env_and_dp() {
        let (cert_pem, _) = synth_leaf("env-uuid-123", "dp-uuid-456");
        let (env, dp) = parse_san_uris(&cert_pem).unwrap();
        assert_eq!(env, "env-uuid-123");
        assert_eq!(dp, "dp-uuid-456");
    }

    #[test]
    fn parse_san_uris_rejects_cert_without_uri_san() {
        // Only DNS SAN, no URI — should fail to find env/dp.
        let mut params = CertificateParams::default();
        params.distinguished_name = DistinguishedName::new();
        params.is_ca = IsCa::NoCa;
        params.subject_alt_names = vec![SanType::DnsName("dp.example.com".try_into().unwrap())];
        let key = KeyPair::generate().unwrap();
        let cert = params.self_signed(&key).unwrap();
        let (env, dp) = parse_san_uris(&cert.pem()).unwrap();
        assert!(env.is_empty());
        assert!(dp.is_empty());
    }

    #[tokio::test]
    async fn provision_writes_three_files_with_san_extracted() {
        let tmp = TempDir::new().unwrap();
        let (cert_pem, key_pem) = synth_leaf("env-1", "dp-1");
        // Reuse the leaf as the "CA" PEM — the path doesn't validate
        // the chain, just shovels bytes to disk.
        let ca_pem = cert_pem.clone();

        let cfg = ManagedConfig {
            enabled: true,
            mtls_dir: tmp.path().to_string_lossy().into_owned(),
            cp_cert_pem: Some(cert_pem.clone()),
            cp_key_pem: Some(key_pem),
            cp_ca_pem: Some(ca_pem),
            ..ManagedConfig::default()
        };

        let p = provision(&cfg).await.unwrap();
        assert_eq!(p.env_id, "env-1");
        assert_eq!(p.dp_id, "dp-1");
        assert!(p.ca_cert_path.exists());
        assert!(p.client_cert_path.exists());
        assert!(p.client_key_path.exists());

        // Verify perms on the key file: 0600 on Unix.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let m = std::fs::metadata(&p.client_key_path).unwrap();
            assert_eq!(m.permissions().mode() & 0o777, 0o600);
        }
    }

    #[tokio::test]
    async fn provision_rejects_mixed_inline_and_file_for_same_role() {
        let tmp = TempDir::new().unwrap();
        let (cert_pem, key_pem) = synth_leaf("env-1", "dp-1");

        // cert role: BOTH inline and file. Should error.
        let cert_file = tmp.path().join("cert.pem");
        tokio::fs::write(&cert_file, &cert_pem).await.unwrap();
        let cfg = ManagedConfig {
            mtls_dir: tmp.path().to_string_lossy().into_owned(),
            cp_cert_pem: Some(cert_pem),
            cp_cert_file: Some(cert_file.to_string_lossy().into_owned()),
            cp_key_pem: Some(key_pem.clone()),
            cp_ca_pem: Some(key_pem),
            ..ManagedConfig::default()
        };

        let err = provision(&cfg).await.unwrap_err();
        assert!(err.to_string().contains("BOTH"), "err: {err}");
    }

    #[tokio::test]
    async fn provision_loads_from_files_when_inline_unset() {
        let tmp = TempDir::new().unwrap();
        let (cert_pem, key_pem) = synth_leaf("env-2", "dp-2");
        let cert_file = tmp.path().join("cert.pem");
        let key_file = tmp.path().join("key.pem");
        let ca_file = tmp.path().join("ca.pem");
        tokio::fs::write(&cert_file, &cert_pem).await.unwrap();
        tokio::fs::write(&key_file, &key_pem).await.unwrap();
        tokio::fs::write(&ca_file, &cert_pem).await.unwrap();

        let cfg = ManagedConfig {
            mtls_dir: tmp.path().join("bundle").to_string_lossy().into_owned(),
            cp_cert_file: Some(cert_file.to_string_lossy().into_owned()),
            cp_key_file: Some(key_file.to_string_lossy().into_owned()),
            cp_ca_file: Some(ca_file.to_string_lossy().into_owned()),
            ..ManagedConfig::default()
        };

        let p = provision(&cfg).await.unwrap();
        assert_eq!(p.env_id, "env-2");
        assert_eq!(p.dp_id, "dp-2");
    }

    #[test]
    fn cert_bundle_provided_inline_only() {
        let cfg = ManagedConfig {
            cp_cert_pem: Some("cert".into()),
            cp_key_pem: Some("key".into()),
            cp_ca_pem: Some("ca".into()),
            ..ManagedConfig::default()
        };
        assert!(cfg.cert_bundle_provided());
    }

    #[test]
    fn cert_bundle_provided_file_only() {
        let cfg = ManagedConfig {
            cp_cert_file: Some("/x/cert".into()),
            cp_key_file: Some("/x/key".into()),
            cp_ca_file: Some("/x/ca".into()),
            ..ManagedConfig::default()
        };
        assert!(cfg.cert_bundle_provided());
    }

    #[test]
    fn cert_bundle_provided_partial_inline_is_false() {
        let cfg = ManagedConfig {
            cp_cert_pem: Some("cert".into()),
            cp_key_pem: Some("key".into()),
            // missing ca
            ..ManagedConfig::default()
        };
        assert!(!cfg.cert_bundle_provided());
    }
}
