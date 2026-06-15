use std::path::{Path, PathBuf};

use aisix_core::ManagedConfig;
use anyhow::{bail, Context};

/// True when the mTLS bundle is already on disk.
pub fn bundle_exists(mtls_dir: impl AsRef<Path>) -> bool {
    let dir = mtls_dir.as_ref();
    ["ca.crt", "client.crt", "client.key"]
        .iter()
        .all(|name| dir.join(name).is_file())
}

/// Read a PEM-encoded CA bundle from disk if `path` is `Some`.
pub fn read_optional_ca_pem(path: Option<&str>) -> anyhow::Result<Option<Vec<u8>>> {
    let Some(p) = path.filter(|s| !s.is_empty()) else {
        return Ok(None);
    };
    let bytes = std::fs::read(p).with_context(|| format!("read managed.cp_ca_cert_file = {p}"))?;
    Ok(Some(bytes))
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

pub fn env_id_path(mtls_dir: impl AsRef<Path>) -> PathBuf {
    mtls_dir.as_ref().join("env_id")
}

/// Read the env_id file written during bundle provisioning.
pub fn read_env_id(mtls_dir: impl AsRef<Path>) -> anyhow::Result<String> {
    let path = env_id_path(mtls_dir);
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("read env_id from {}", path.display()))?;
    let trimmed = raw.trim().to_string();
    if trimmed.is_empty() {
        bail!("env_id file {} is empty", path.display());
    }
    if uuid::Uuid::parse_str(&trimmed).is_err() {
        bail!(
            "env_id file {} contains invalid env_id {:?}",
            path.display(),
            trimmed,
        );
    }
    Ok(trimmed)
}

pub async fn persist_dp_id_for_provisioning(
    cfg: &ManagedConfig,
    dp_id: &str,
    env_id: &str,
) -> anyhow::Result<()> {
    persist_dp_id(&cfg.dp_id_file, dp_id).await?;
    persist_env_id(&cfg.mtls_dir, env_id).await?;
    Ok(())
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

async fn persist_env_id(mtls_dir: &str, env_id: &str) -> anyhow::Result<()> {
    let dir = PathBuf::from(mtls_dir);
    tokio::fs::create_dir_all(&dir)
        .await
        .with_context(|| format!("create {}", dir.display()))?;
    write_atomic(&dir.join("env_id"), env_id.as_bytes(), 0o600).await
}

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

#[cfg(not(unix))]
async fn write_atomic(_path: &Path, _data: &[u8], _mode: u32) -> anyhow::Result<()> {
    anyhow::bail!("managed mode is only supported on Unix")
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn read_env_id_rejects_invalid_uuid() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("env_id"), "not/a/uuid").unwrap();
        let err = read_env_id(dir.path()).unwrap_err();
        assert!(err.to_string().contains("invalid env_id"), "got: {err}");
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
