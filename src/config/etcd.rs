use std::{sync::Arc, time::Duration};

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use backon::{ConstantBuilder, Retryable};
use dashmap::{DashMap, Entry};
use etcd_client::{GetOptions, PutOptions, WatchOptions};
use log::{debug, error, warn};
use serde::Deserialize;
use thiserror::Error;
use tokio::{
    sync::{Mutex, Notify, mpsc},
    task::JoinHandle,
    time::sleep,
};

use crate::config::{ConfigEvent, ConfigEventReceiver, ConfigProvider, GetEntry, PutEntry};

// ── TLS certificate material ──────────────────────────────────────────────────

/// Certificate material for connecting to etcd over HTTPS.
///
/// Supports two exclusive sources:
/// - [`EtcdTlsCertConfig::Strings`] – inline PEM-encoded content (e.g. from
///   environment variables or secrets managers).
/// - [`EtcdTlsCertConfig::Files`] – paths to PEM files on disk (default).
///
/// All fields within each variant are optional; omit a field to disable the
/// corresponding TLS feature.  Mixing fields from both variants (e.g.
/// `ca_file` and `cert`) is rejected at parse time.
#[derive(Clone, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
pub enum EtcdTlsCertConfig {
    /// File paths to PEM-encoded certificate material (tried first during deserialization).
    Files {
        /// Path to a PEM-encoded CA certificate file.
        ca_file: Option<String>,
        /// Path to a PEM-encoded client certificate file for mTLS.
        /// Must be paired with `key_file`.
        cert_file: Option<String>,
        /// Path to a PEM-encoded private key file for mTLS.
        /// Must be paired with `cert_file`.
        key_file: Option<String>,
    },
    /// Inline PEM-encoded certificate material.
    Strings {
        /// PEM-encoded CA certificate used to validate the etcd server.
        ca: Option<String>,
        /// PEM-encoded client certificate for mTLS.  Must be paired with `key`.
        cert: Option<String>,
        /// PEM-encoded private key for mTLS.  Must be paired with `cert`.
        key: Option<String>,
    },
}

impl Default for EtcdTlsCertConfig {
    fn default() -> Self {
        Self::Files { ca_file: None, cert_file: None, key_file: None }
    }
}

impl EtcdTlsCertConfig {
    fn read_file(label: &str, path: &Option<String>) -> Result<Option<Vec<u8>>> {
        match path {
            None => Ok(None),
            Some(p) => {
                let bytes = std::fs::read(p)
                    .with_context(|| format!("etcd TLS: failed to read {label}_file '{p}'"))?;
                Ok(Some(bytes))
            }
        }
    }

    /// Return the CA certificate as PEM bytes, or `None` if not configured.
    pub fn ca_pem(&self) -> Result<Option<Vec<u8>>> {
        match self {
            Self::Strings { ca, .. } => Ok(ca.as_deref().map(|s| s.as_bytes().to_vec())),
            Self::Files { ca_file, .. } => Self::read_file("ca", ca_file),
        }
    }

    /// Return the client certificate as PEM bytes, or `None` if not configured.
    pub fn cert_pem(&self) -> Result<Option<Vec<u8>>> {
        match self {
            Self::Strings { cert, .. } => Ok(cert.as_deref().map(|s| s.as_bytes().to_vec())),
            Self::Files { cert_file, .. } => Self::read_file("cert", cert_file),
        }
    }

    /// Return the private key as PEM bytes, or `None` if not configured.
    pub fn key_pem(&self) -> Result<Option<Vec<u8>>> {
        match self {
            Self::Strings { key, .. } => Ok(key.as_deref().map(|s| s.as_bytes().to_vec())),
            Self::Files { key_file, .. } => Self::read_file("key", key_file),
        }
    }

    fn has_cert(&self) -> bool {
        match self {
            Self::Strings { cert, .. } => cert.is_some(),
            Self::Files { cert_file, .. } => cert_file.is_some(),
        }
    }

    fn has_key(&self) -> bool {
        match self {
            Self::Strings { key, .. } => key.is_some(),
            Self::Files { key_file, .. } => key_file.is_some(),
        }
    }
}

impl std::fmt::Debug for EtcdTlsCertConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Strings { ca, cert, key } => f
                .debug_struct("EtcdTlsCertConfig::Strings")
                .field("ca", &ca.as_deref().map(|_| "***redacted***"))
                .field("cert", &cert.as_deref().map(|_| "***redacted***"))
                .field("key", &key.as_deref().map(|_| "***redacted***"))
                .finish(),
            Self::Files { ca_file, cert_file, key_file } => f
                .debug_struct("EtcdTlsCertConfig::Files")
                .field("ca_file", ca_file)
                .field("cert_file", cert_file)
                .field("key_file", key_file)
                .finish(),
        }
    }
}

/// TLS settings for connecting to etcd over HTTPS.
///
/// Certificate material (`ca`, `cert`, `key` inline strings, or `ca_file`,
/// `cert_file`, `key_file` file paths) is deserialized from a flat structure.
/// Mixing the two forms in the same config block is rejected at parse time.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct EtcdTlsConfig {
    /// Certificate material; defaults to the `Files` variant with all paths unset.
    #[serde(flatten)]
    pub cert: EtcdTlsCertConfig,
    /// Skip TLS certificate verification entirely.
    ///
    /// **WARNING**: This disables all certificate validation including hostname
    /// and CA checks. Use only in development or testing environments.
    #[serde(default)]
    pub insecure_skip_verify: bool,
}

impl std::ops::Deref for EtcdTlsConfig {
    type Target = EtcdTlsCertConfig;
    fn deref(&self) -> &Self::Target {
        &self.cert
    }
}

// ── Config validation errors ─────────────────────────────────────────────────

/// Errors produced during etcd connection-configuration validation.
#[derive(Debug, Error)]
pub enum EtcdConfigError {
    /// The host list contains a mix of `http://` and `https://` endpoints,
    /// which is unsupported.
    #[error("etcd hosts must use a single scheme (all http:// or all https://)")]
    MixedSchemes,

    /// One of the host strings is missing the `http://` or `https://` scheme
    /// prefix.
    #[error("etcd host '{0}' is missing a scheme; use the prefix http:// or https://")]
    MissingScheme(String),

    /// Only one of `cert`/`key` was provided; both are required for mTLS.
    #[error(
        "both tls cert and key must be set together \
         (cert/key for inline PEM, or cert_file/key_file for file paths)"
    )]
    PartialMtlsKeypair,
}

/// Validate the connection configuration before attempting any I/O.
///
/// Returns `Ok(())` if the configuration is valid, or an [`EtcdConfigError`]
/// describing the first validation failure.
fn validate_connect_config(config: &Config) -> std::result::Result<(), EtcdConfigError> {
    let has_https = config.host.iter().any(|h| h.starts_with("https://"));
    let has_http = config.host.iter().any(|h| h.starts_with("http://"));

    if has_http && has_https {
        return Err(EtcdConfigError::MixedSchemes);
    }
    if let Some(invalid) = config
        .host
        .iter()
        .find(|h| !h.starts_with("http://") && !h.starts_with("https://"))
    {
        return Err(EtcdConfigError::MissingScheme(invalid.clone()));
    }

    if has_https {
        if let Some(t) = &config.tls {
            if t.cert.has_cert() != t.cert.has_key() {
                return Err(EtcdConfigError::PartialMtlsKeypair);
            }
        }
    }

    Ok(())
}


#[derive(Clone, Deserialize)]
pub struct Config {
    pub host: Vec<String>,
    pub prefix: String,
    pub timeout: u32,
    pub user: Option<String>,
    pub password: Option<String>,
    /// Optional TLS settings used when etcd endpoints use `https://`.
    pub tls: Option<EtcdTlsConfig>,
}

impl std::fmt::Debug for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Config")
            .field("host", &self.host)
            .field("prefix", &self.prefix)
            .field("timeout", &self.timeout)
            .field("user", &self.user)
            .field(
                "password",
                &self.password.as_deref().map(|_| "***redacted***"),
            )
            .field("tls", &self.tls)
            .finish()
    }
}

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

pub struct EtcdConfigProvider {
    client: etcd_client::Client,
    prefix: String,
    txs: Arc<DashMap<String, mpsc::Sender<ConfigEvent>>>,
    /// Signals the supervisor loop to stop.
    shutdown: Arc<Notify>,
    /// Handle to the watch supervisor task; taken on shutdown.
    supervisor_handle: Mutex<Option<JoinHandle<()>>>,
}

/// Maximum backoff delay between reconnect attempts.
const MAX_BACKOFF: Duration = Duration::from_secs(60);
/// Initial backoff delay.
const INITIAL_BACKOFF: Duration = Duration::from_secs(1);

impl EtcdConfigProvider {
    pub async fn new(config: Config) -> Result<Self> {
        validate_connect_config(&config).map_err(|e| anyhow::anyhow!(e))?;

        let client = (|| Self::connect_client(&config))
            .retry(
                ConstantBuilder::default()
                    .with_delay(Duration::from_secs(5))
                    .with_max_times(5),
            )
            .notify(|err, dur| error!("Failed to connect to etcd: {err}, retrying after {:?}", dur))
            .await
            .context("failed to connect to etcd and retry exhausted")?;
        let txs = Arc::new(DashMap::<String, mpsc::Sender<ConfigEvent>>::new());
        let shutdown = Arc::new(Notify::new());

        let handle = Self::spawn_supervisor(
            client.clone(),
            config.prefix.clone(),
            txs.clone(),
            shutdown.clone(),
        );

        Ok(Self {
            client,
            prefix: config.prefix.clone(),
            txs,
            shutdown,
            supervisor_handle: Mutex::new(Some(handle)),
        })
    }

    async fn connect_client(config: &Config) -> Result<etcd_client::Client> {
        let mut opts = etcd_client::ConnectOptions::default()
            .with_connect_timeout(Duration::from_secs(config.timeout as u64));

        if let (Some(user), Some(password)) = (config.user.clone(), config.password.clone()) {
            opts = opts.with_user(user, password);
        }

        let has_https = config.host.iter().any(|h| h.starts_with("https://"));

        if has_https {
            let mut tls_cfg = etcd_client::OpenSslClientConfig::default();
            if let Some(t) = &config.tls {
                if t.insecure_skip_verify {
                    tls_cfg = tls_cfg.manually(|b| {
                        b.set_verify(openssl::ssl::SslVerifyMode::NONE);
                        Ok(())
                    });
                }

                if let Some(ca) = t.ca_pem()? {
                    tls_cfg = tls_cfg.ca_cert_pem(ca.as_slice());
                }
                // mTLS: cert and key pair completeness already validated.
                if let (Some(cert), Some(key)) = (t.cert_pem()?, t.key_pem()?) {
                    tls_cfg = tls_cfg.client_cert_pem_and_key(cert.as_slice(), key.as_slice());
                }
            }
            opts = opts.with_openssl_tls(tls_cfg);
        }

        let mut client = etcd_client::Client::connect(
            config
                .host
                .iter()
                .map(|h: &String| h.as_str())
                .collect::<Vec<_>>(),
            Some(opts),
        )
        .await?;

        client.status().await?;
        Ok(client)
    }

    /// Spawn the long-running supervisor task that manages the watch stream
    /// lifecycle: reconnects on failure, resumes from the last seen revision,
    /// and triggers a full resync when etcd compaction makes resumption
    /// impossible.
    fn spawn_supervisor(
        mut client: etcd_client::Client,
        prefix: String,
        txs: Arc<DashMap<String, mpsc::Sender<ConfigEvent>>>,
        shutdown: Arc<Notify>,
    ) -> JoinHandle<()> {
        tokio::spawn(async move {
            // The revision from which the next watch attempt should start.
            // 0 means "from newest" (first connect or after resync).
            let mut start_revision: i64 = 0;
            let mut backoff = INITIAL_BACKOFF;
            let mut attempt: u32 = 0;

            'supervisor: loop {
                // Build watch options: resume from last seen revision when possible.
                let watch_opts = if start_revision > 0 {
                    WatchOptions::new()
                        .with_prefix()
                        .with_start_revision(start_revision)
                } else {
                    WatchOptions::new().with_prefix()
                };

                debug!(
                    "etcd watch: connecting (attempt={attempt}, start_revision={start_revision})"
                );

                // Establish the watch stream, with shutdown interruptibility.
                let stream_result = tokio::select! {
                    biased;
                    _ = shutdown.notified() => {
                        debug!("etcd watch supervisor: shutdown requested before stream open");
                        break 'supervisor;
                    }
                    r = client.watch(prefix.as_str(), Some(watch_opts)) => r,
                };

                let mut stream = match stream_result {
                    Ok(s) => {
                        attempt = 0;
                        backoff = INITIAL_BACKOFF;
                        debug!("etcd watch: stream established (start_revision={start_revision})");
                        s
                    }
                    Err(err) => {
                        warn!("etcd watch: failed to establish stream (attempt={attempt}): {err}");
                        attempt += 1;
                        if Self::backoff_or_shutdown(&shutdown, backoff).await {
                            break 'supervisor;
                        }
                        backoff = (backoff * 2).min(MAX_BACKOFF);
                        continue 'supervisor;
                    }
                };

                // Consume the stream until it ends or shutdown is requested.
                loop {
                    let msg = tokio::select! {
                        biased;
                        _ = shutdown.notified() => {
                            debug!("etcd watch supervisor: shutdown requested");
                            break 'supervisor;
                        }
                        m = stream.message() => m,
                    };

                    match msg {
                        Err(err) => {
                            warn!("etcd watch: stream error, will reconnect: {err}");
                            break; // inner loop → reconnect in outer loop
                        }
                        Ok(None) => {
                            warn!("etcd watch: stream ended, will reconnect");
                            break;
                        }
                        Ok(Some(resp)) => {
                            if resp.canceled() {
                                let compact_rev = resp.compact_revision();
                                if compact_rev > 0 {
                                    debug!(
                                        "etcd watch: canceled due to compaction \
                                         (compact_revision={compact_rev}), triggering resync",
                                    );
                                    Self::broadcast(&txs, ConfigEvent::Resync).await;
                                    // After a full resync the consumer will reach the
                                    // current head; reset so the next watch starts
                                    // from newest rather than a compacted revision.
                                    start_revision = 0;
                                } else {
                                    warn!("etcd watch: canceled, will reconnect");
                                }
                                break;
                            }

                            for event in resp.events() {
                                if let Some(kv) = event.kv() {
                                    let key = match kv.key_str() {
                                        Ok(k) => k.to_string(),
                                        Err(err) => {
                                            warn!("etcd watch: failed to parse key: {err}");
                                            continue;
                                        }
                                    };

                                    let targets: Vec<mpsc::Sender<ConfigEvent>> = txs
                                        .iter()
                                        .filter(|e| key.starts_with(e.key().as_str()))
                                        .map(|e| e.value().clone())
                                        .collect();

                                    if targets.is_empty() {
                                        continue;
                                    }

                                    let payload = match event.event_type() {
                                        etcd_client::EventType::Put => ConfigEvent::Put((
                                            key,
                                            kv.value().to_vec(),
                                            kv.mod_revision(),
                                        )),
                                        etcd_client::EventType::Delete => {
                                            ConfigEvent::Delete((key, kv.mod_revision()))
                                        }
                                    };

                                    // Advance resume point past the last processed event.
                                    if let ConfigEvent::Put((_, _, rev))
                                    | ConfigEvent::Delete((_, rev)) = &payload
                                    {
                                        start_revision = rev + 1;
                                    }

                                    for tx in targets {
                                        if let Err(err) = tx.send(payload.clone()).await {
                                            warn!("etcd watch: failed to dispatch event: {err}");
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // Back-off before reconnecting (unless shutdown was requested).
                attempt += 1;
                if Self::backoff_or_shutdown(&shutdown, backoff).await {
                    break 'supervisor;
                }
                backoff = (backoff * 2).min(MAX_BACKOFF);
            }

            debug!("etcd watch supervisor: exited");
        })
    }

    /// Broadcast an event to all registered subscribers.
    async fn broadcast(txs: &DashMap<String, mpsc::Sender<ConfigEvent>>, event: ConfigEvent) {
        for entry in txs.iter() {
            if let Err(err) = entry.value().send(event.clone()).await {
                warn!("etcd watch: failed to broadcast event: {}", err);
            }
        }
    }

    /// Sleep for `delay`, but return early (returning `true`) if shutdown is
    /// requested. Returns `false` when the sleep completes normally.
    async fn backoff_or_shutdown(shutdown: &Notify, delay: Duration) -> bool {
        tokio::select! {
            biased;
            _ = shutdown.notified() => true,
            _ = sleep(delay) => false,
        }
    }
}

#[async_trait]
impl ConfigProvider for EtcdConfigProvider {
    async fn get_all_raw(&self, prefix: Option<&str>) -> Result<Vec<GetEntry<Vec<u8>>>, String> {
        let full_prefix = format!("{}{}", self.prefix, prefix.unwrap_or(""));

        let mut client = self.client.clone();
        match client
            .get(full_prefix.as_str(), Some(GetOptions::new().with_prefix()))
            .await
        {
            Ok(resp) => {
                let mut results = Vec::new();
                for kv in resp.kvs() {
                    if let Ok(key) = kv.key_str() {
                        results.push(GetEntry {
                            key: key.strip_prefix(&self.prefix).unwrap_or(key).to_string(),
                            value: kv.value().to_vec(),
                            create_revision: kv.create_revision(),
                            mod_revision: kv.mod_revision(),
                        });
                    }
                }
                Ok(results)
            }
            Err(err) => Err(format!("etcd get all failed: {}", err)),
        }
    }

    async fn get_raw(&self, key: &str) -> Result<Option<GetEntry<Vec<u8>>>, String> {
        let full_key = format!("{}{}", self.prefix, key);

        let mut client = self.client.clone();
        match client.get(full_key.as_str(), None).await {
            Ok(resp) => {
                if let Some(kv) = resp.kvs().first() {
                    Ok(Some(GetEntry {
                        key: key.to_string(),
                        value: kv.value().to_vec(),
                        create_revision: kv.create_revision(),
                        mod_revision: kv.mod_revision(),
                    }))
                } else {
                    Ok(None)
                }
            }
            Err(err) => Err(format!("etcd get failed: {}", err)),
        }
    }

    async fn put_raw(&self, key: &str, value: Vec<u8>) -> Result<PutEntry<Vec<u8>>, String> {
        let full_key = format!("{}{}", self.prefix, key);

        let mut client = self.client.clone();
        match client
            .put(full_key, value, Some(PutOptions::new().with_prev_key()))
            .await
        {
            Ok(resp) => match resp.prev_key() {
                Some(kv) => Ok(PutEntry::Updated(GetEntry {
                    key: key.to_string(),
                    value: kv.value().to_vec(),
                    create_revision: kv.create_revision(),
                    mod_revision: kv.mod_revision(),
                })),
                None => Ok(PutEntry::Created),
            },
            Err(err) => Err(format!("etcd put failed: {}", err)),
        }
    }

    async fn delete(&self, key: &str) -> Result<i64, String> {
        let full_key = format!("{}{}", self.prefix, key);

        let mut client = self.client.clone();
        match client.delete(full_key, None).await {
            Ok(resp) => Ok(resp.deleted()),
            Err(err) => Err(format!("etcd delete failed: {}", err)),
        }
    }

    async fn watch(&self, prefix: Option<&str>) -> Result<ConfigEventReceiver> {
        let full_prefix = format!("{}{}", self.prefix, prefix.unwrap_or(""));

        match self.txs.entry(full_prefix) {
            Entry::Occupied(_) => Err(anyhow!("Prefix has been watched")),
            Entry::Vacant(v) => {
                let (tx, rx) = mpsc::channel(32);
                v.insert(tx);
                Ok(rx)
            }
        }
    }

    async fn shutdown(&self) -> Result<()> {
        // Signal the supervisor to stop.
        self.shutdown.notify_one();

        // Close all dispatch channels so consumers see channel-closed.
        self.txs.clear();

        let handle = self.supervisor_handle.lock().await.take();
        if let Some(mut h) = handle {
            match tokio::time::timeout(Duration::from_secs(10), &mut h).await {
                Ok(joined) => joined.context("failed to shutdown watch supervisor")?,
                Err(_) => {
                    return Err(anyhow!(
                        "timed out waiting for watch supervisor to shutdown"
                    ));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;
    use tempfile::NamedTempFile;

    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn tls_files(
        ca_file: Option<&str>,
        cert_file: Option<&str>,
        key_file: Option<&str>,
    ) -> EtcdTlsConfig {
        EtcdTlsConfig {
            cert: EtcdTlsCertConfig::Files {
                ca_file: ca_file.map(str::to_owned),
                cert_file: cert_file.map(str::to_owned),
                key_file: key_file.map(str::to_owned),
            },
            ..Default::default()
        }
    }

    fn tls_strings(
        ca: Option<&str>,
        cert: Option<&str>,
        key: Option<&str>,
    ) -> EtcdTlsConfig {
        EtcdTlsConfig {
            cert: EtcdTlsCertConfig::Strings {
                ca: ca.map(str::to_owned),
                cert: cert.map(str::to_owned),
                key: key.map(str::to_owned),
            },
            ..Default::default()
        }
    }

    // ── EtcdTlsConfig defaults ────────────────────────────────────────────────

    #[test]
    fn test_etcd_tls_config_default() {
        let tls = EtcdTlsConfig::default();
        assert_matches!(
            &tls.cert,
            EtcdTlsCertConfig::Files { ca_file, cert_file, key_file }
                if ca_file.is_none() && cert_file.is_none() && key_file.is_none()
        );
        assert!(!tls.insecure_skip_verify);
    }

    #[test]
    fn test_config_default_no_tls() {
        let cfg = Config::default();
        assert!(cfg.tls.is_none());
        assert_eq!(cfg.host, vec!["http://127.0.0.1:2379"]);
    }

    // ── EtcdTlsCertConfig pem accessors ──────────────────────────────────────

    #[test]
    fn test_cert_config_strings_ca_pem() {
        let cfg = EtcdTlsCertConfig::Strings {
            ca: Some("ca-content".to_owned()),
            cert: None,
            key: None,
        };
        assert_eq!(cfg.ca_pem().unwrap(), Some(b"ca-content".to_vec()));
        assert_eq!(cfg.cert_pem().unwrap(), None);
        assert_eq!(cfg.key_pem().unwrap(), None);
    }

    #[test]
    fn test_cert_config_files_reads_from_disk() {
        let mut tmp = NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut tmp, b"file-pem-content").unwrap();
        let path = tmp.path().to_str().unwrap().to_owned();

        let cfg = EtcdTlsCertConfig::Files {
            ca_file: Some(path),
            cert_file: None,
            key_file: None,
        };
        assert_eq!(cfg.ca_pem().unwrap(), Some(b"file-pem-content".to_vec()));
        assert_eq!(cfg.cert_pem().unwrap(), None);
    }

    #[test]
    fn test_cert_config_files_missing_returns_error() {
        let cfg = EtcdTlsCertConfig::Files {
            ca_file: Some("/nonexistent/ca.pem".to_owned()),
            cert_file: None,
            key_file: None,
        };
        assert_matches!(
            cfg.ca_pem(),
            Err(e) if e.to_string().contains("failed to read ca_file")
        );
    }

    #[test]
    fn test_tls_config_delegates_pem_methods() {
        let tls = tls_strings(Some("ca-content"), None, None);
        assert_eq!(tls.ca_pem().unwrap(), Some(b"ca-content".to_vec()));
        assert_eq!(tls.cert_pem().unwrap(), None);
        assert_eq!(tls.key_pem().unwrap(), None);
    }

    #[test]
    fn test_tls_config_no_cert_returns_none() {
        // Default EtcdTlsConfig has Files variant with all None; pem accessors return None.
        let tls = EtcdTlsConfig::default();
        assert_eq!(tls.ca_pem().unwrap(), None);
        assert_eq!(tls.cert_pem().unwrap(), None);
        assert_eq!(tls.key_pem().unwrap(), None);
    }

    // ── TLS scheme detection ──────────────────────────────────────────────────

    #[test]
    fn test_tls_detected_from_https_host() {
        let cfg = Config {
            host: vec!["https://etcd.example.com:2379".to_string()],
            ..Config::default()
        };
        let has_https = cfg.host.iter().any(|h| h.starts_with("https://"));
        let has_http = cfg.host.iter().any(|h| h.starts_with("http://"));
        assert!(has_https);
        assert!(!has_http);
    }

    #[test]
    fn test_tls_not_detected_for_http_host() {
        let cfg = Config::default();
        let has_https = cfg.host.iter().any(|h| h.starts_with("https://"));
        assert!(!has_https);
    }

    #[test]
    fn test_mixed_http_https_hosts_detected() {
        let cfg = Config {
            host: vec![
                "http://etcd1.example.com:2379".to_string(),
                "https://etcd2.example.com:2379".to_string(),
            ],
            ..Config::default()
        };
        let has_https = cfg.host.iter().any(|h| h.starts_with("https://"));
        let has_http = cfg.host.iter().any(|h| h.starts_with("http://"));
        assert!(has_https && has_http);
    }

    // ── connect_client validation (via connect_client) ────────────────────────

    // ── validate_connect_config unit tests ───────────────────────────────────

    #[test]
    fn test_validate_rejects_mixed_schemes() {
        let cfg = Config {
            host: vec![
                "http://etcd1:2379".to_string(),
                "https://etcd2:2379".to_string(),
            ],
            ..Config::default()
        };
        assert_matches!(
            validate_connect_config(&cfg),
            Err(EtcdConfigError::MixedSchemes)
        );
    }

    #[test]
    fn test_validate_rejects_missing_scheme() {
        let cfg = Config {
            host: vec!["127.0.0.1:2379".to_string()],
            ..Config::default()
        };
        assert_matches!(
            validate_connect_config(&cfg),
            Err(EtcdConfigError::MissingScheme(h)) if h == "127.0.0.1:2379"
        );
    }

    #[test]
    fn test_validate_rejects_partial_mtls() {
        // Both file-based and string-based partial mTLS should fail.
        for cfg in [
            Config {
                host: vec!["https://etcd:2379".to_string()],
                tls: Some(tls_files(None, Some("cert.pem"), None)),
                ..Config::default()
            },
            Config {
                host: vec!["https://etcd:2379".to_string()],
                tls: Some(tls_files(None, None, Some("key.pem"))),
                ..Config::default()
            },
            Config {
                host: vec!["https://etcd:2379".to_string()],
                tls: Some(tls_strings(None, Some("cert"), None)),
                ..Config::default()
            },
            Config {
                host: vec!["https://etcd:2379".to_string()],
                tls: Some(tls_strings(None, None, Some("key"))),
                ..Config::default()
            },
        ] {
            assert_matches!(
                validate_connect_config(&cfg),
                Err(EtcdConfigError::PartialMtlsKeypair)
            );
        }
    }

    #[test]
    fn test_validate_http_ok() {
        let cfg = Config::default();
        assert!(validate_connect_config(&cfg).is_ok());
    }

    #[test]
    fn test_validate_https_ok() {
        let cfg = Config {
            host: vec!["https://etcd:2379".to_string()],
            ..Config::default()
        };
        assert!(validate_connect_config(&cfg).is_ok());
    }

    // ── deserialization ───────────────────────────────────────────────────────

    #[test]
    fn test_tls_deserialize_files() {
        let json = r#"{"ca_file":"ca.pem","cert_file":"cert.pem","key_file":"key.pem"}"#;
        let tls: EtcdTlsConfig = serde_json::from_str(json).unwrap();
        assert_matches!(
            &tls.cert,
            EtcdTlsCertConfig::Files { ca_file, cert_file, key_file }
                if ca_file.as_deref() == Some("ca.pem")
                && cert_file.as_deref() == Some("cert.pem")
                && key_file.as_deref() == Some("key.pem")
        );
        assert!(!tls.insecure_skip_verify);
    }

    #[test]
    fn test_tls_deserialize_strings() {
        let json = r#"{"ca":"ca-content","cert":"cert-content","key":"key-content"}"#;
        let tls: EtcdTlsConfig = serde_json::from_str(json).unwrap();
        assert_matches!(
            &tls.cert,
            EtcdTlsCertConfig::Strings { ca, cert, key }
                if ca.as_deref() == Some("ca-content")
                && cert.as_deref() == Some("cert-content")
                && key.as_deref() == Some("key-content")
        );
    }

    #[test]
    fn test_tls_deserialize_mixed_rejects() {
        let json = r#"{"ca_file":"ca.pem","cert":"cert-content"}"#;
        let result = serde_json::from_str::<EtcdTlsConfig>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_tls_deserialize_insecure_only() {
        let json = r#"{"insecure_skip_verify":true}"#;
        let tls: EtcdTlsConfig = serde_json::from_str(json).unwrap();
        assert_matches!(
            &tls.cert,
            EtcdTlsCertConfig::Files { ca_file, cert_file, key_file }
                if ca_file.is_none() && cert_file.is_none() && key_file.is_none()
        );
        assert!(tls.insecure_skip_verify);
    }

    #[test]
    fn test_tls_deserialize_empty() {
        let json = r#"{}"#;
        let tls: EtcdTlsConfig = serde_json::from_str(json).unwrap();
        assert_matches!(
            &tls.cert,
            EtcdTlsCertConfig::Files { ca_file, cert_file, key_file }
                if ca_file.is_none() && cert_file.is_none() && key_file.is_none()
        );
        assert!(!tls.insecure_skip_verify);
    }

    #[test]
    fn test_config_deserialize_with_tls_files() {
        let json = r#"{
            "host": ["https://etcd.example.com:2379"],
            "prefix": "/aisix",
            "timeout": 30,
            "tls": {"ca_file": "ca.pem", "insecure_skip_verify": false}
        }"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.host, vec!["https://etcd.example.com:2379"]);
        let tls = cfg.tls.unwrap();
        assert_matches!(
            &tls.cert,
            EtcdTlsCertConfig::Files { ca_file, .. }
                if ca_file.as_deref() == Some("ca.pem")
        );
        assert!(!tls.insecure_skip_verify);
    }

    #[test]
    fn test_config_deserialize_with_tls_strings() {
        let json = r#"{
            "host": ["https://etcd.example.com:2379"],
            "prefix": "/aisix",
            "timeout": 30,
            "tls": {"ca": "ca-content", "cert": "cert-content", "key": "key-content"}
        }"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        let tls = cfg.tls.unwrap();
        assert_matches!(
            &tls.cert,
            EtcdTlsCertConfig::Strings { ca, .. }
                if ca.as_deref() == Some("ca-content")
        );
    }

    #[test]
    fn test_config_deserialize_without_tls() {
        let json = r#"{
            "host": ["http://127.0.0.1:2379"],
            "prefix": "/aisix",
            "timeout": 5
        }"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert!(cfg.tls.is_none());
        assert!(cfg.user.is_none());
        assert!(cfg.password.is_none());
    }
}
