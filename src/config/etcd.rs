use std::{sync::Arc, time::Duration};

use anyhow::{Context, Result, anyhow};
use thiserror::Error;
use async_trait::async_trait;
use backon::{ConstantBuilder, Retryable};
use dashmap::{DashMap, Entry};
use etcd_client::{GetOptions, PutOptions, WatchOptions};
use log::{debug, error, warn};
use serde::Deserialize;
use tokio::{
    sync::{Mutex, Notify, mpsc},
    task::JoinHandle,
    time::sleep,
};

use crate::config::{ConfigEvent, ConfigEventReceiver, ConfigProvider, GetEntry, PutEntry};

/// Read PEM bytes from a file path. Returns `Ok(None)` when `path` is `None`.
fn read_pem(label: &str, path: &Option<String>) -> Result<Option<Vec<u8>>> {
    match path {
        None => Ok(None),
        Some(p) => {
            let bytes = std::fs::read(p)
                .with_context(|| format!("etcd TLS: failed to read {label}_file '{p}'"))?;
            Ok(Some(bytes))
        }
    }
}

/// Errors produced during etcd connection-configuration validation.
#[derive(Debug, Error)]
pub enum EtcdConfigError {
    /// The host list contains a mix of `http://` and `https://` endpoints,
    /// which is unsupported.
    #[error("etcd hosts must use a single scheme (all http:// or all https://)")]
    MixedSchemes,

    /// One of the host strings is missing the `http://` or `https://` scheme
    /// prefix.
    #[error("etcd host '{0}' is missing a scheme; use http:// or https://")]
    MissingScheme(String),

    /// Only one of `cert`/`key` was provided; both are required for mTLS.
    #[error(
        "both tls cert and key must be set together \
         (via cert_file/cert_pem and key_file/key_pem)"
    )]
    PartialMtlsKeypair,
}

/// Validate the connection configuration before attempting any I/O.
///
/// Returns `Ok(has_https)` where `has_https` indicates whether the host list
/// uses HTTPS, or an [`EtcdConfigError`] describing the first validation
/// failure.
fn validate_connect_config(config: &Config) -> std::result::Result<bool, EtcdConfigError> {
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
            let has_cert = t.cert_file.is_some() || t.cert_pem.is_some();
            let has_key = t.key_file.is_some() || t.key_pem.is_some();
            if has_cert != has_key {
                return Err(EtcdConfigError::PartialMtlsKeypair);
            }
        }
    }

    Ok(has_https)
}


const MAX_BACKOFF: Duration = Duration::from_secs(60);
/// Initial backoff delay.
const INITIAL_BACKOFF: Duration = Duration::from_secs(1);

/// TLS settings for connecting to etcd over HTTPS.
///
/// Certificate material can be provided as file paths **or** as inline PEM
/// strings.  When both a file path and an inline PEM string are supplied for
/// the same field, the file path takes precedence.
///
/// All fields are optional; omit a field to disable the corresponding TLS
/// feature.
#[derive(Clone, Default, Deserialize)]
pub struct EtcdTlsConfig {
    /// Path to a PEM-encoded CA certificate file used to validate the etcd
    /// server certificate.  Takes precedence over `ca_pem` when both are set.
    pub ca_file: Option<String>,
    /// Path to a PEM-encoded client certificate file for mTLS authentication.
    /// Must be provided together with `key_file` (or `key_pem`).
    /// Takes precedence over `cert_pem` when both are set.
    pub cert_file: Option<String>,
    /// Path to a PEM-encoded private key file matching `cert_file` (or
    /// `cert_pem`).  Takes precedence over `key_pem` when both are set.
    pub key_file: Option<String>,

    /// Inline PEM-encoded CA certificate used to validate the etcd server
    /// certificate.  Ignored when `ca_file` is also set.
    pub ca_pem: Option<String>,
    /// Inline PEM-encoded client certificate for mTLS authentication.
    /// Must be provided together with `key_pem` (or `key_file`).
    /// Ignored when `cert_file` is also set.
    pub cert_pem: Option<String>,
    /// Inline PEM-encoded private key matching `cert_pem` (or `cert_file`).
    /// Ignored when `key_file` is also set.
    pub key_pem: Option<String>,

    /// Skip TLS certificate verification entirely.
    ///
    /// **WARNING**: This disables all certificate validation including hostname
    /// and CA checks. Use only in development or testing environments.
    #[serde(default)]
    pub insecure_skip_verify: bool,
}

impl std::fmt::Debug for EtcdTlsConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EtcdTlsConfig")
            .field("ca_file", &self.ca_file)
            .field("cert_file", &self.cert_file)
            .field("key_file", &self.key_file)
            .field("ca_pem", &self.ca_pem.as_deref().map(|_| "***redacted***"))
            .field("cert_pem", &self.cert_pem.as_deref().map(|_| "***redacted***"))
            .field("key_pem", &self.key_pem.as_deref().map(|_| "***redacted***"))
            .field("insecure_skip_verify", &self.insecure_skip_verify)
            .finish()
    }
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

impl EtcdConfigProvider {
    pub async fn new(config: Config) -> Result<Self> {
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

        let has_https = validate_connect_config(config)?;

        if has_https {
            let mut tls_cfg = etcd_client::OpenSslClientConfig::default();
            if let Some(t) = &config.tls {
                if t.insecure_skip_verify {
                    tls_cfg = tls_cfg.manually(|b| {
                        b.set_verify(openssl::ssl::SslVerifyMode::NONE);
                        Ok(())
                    });
                }

                // CA certificate: file takes precedence over inline PEM.
                let ca_bytes = if t.ca_file.is_some() {
                    read_pem("ca", &t.ca_file)?
                } else {
                    t.ca_pem.as_deref().map(|s| s.as_bytes().to_vec())
                };
                if let Some(ca) = ca_bytes {
                    tls_cfg = tls_cfg.ca_cert_pem(ca.as_slice());
                }

                // Resolve cert bytes: cert_file takes precedence over cert_pem.
                let cert_bytes: Option<Vec<u8>> = if t.cert_file.is_some() {
                    read_pem("cert", &t.cert_file)?
                } else {
                    t.cert_pem.as_deref().map(|s| s.as_bytes().to_vec())
                };
                // Resolve key bytes: key_file takes precedence over key_pem.
                let key_bytes: Option<Vec<u8>> = if t.key_file.is_some() {
                    read_pem("key", &t.key_file)?
                } else {
                    t.key_pem.as_deref().map(|s| s.as_bytes().to_vec())
                };

                if let (Some(cert), Some(key)) = (&cert_bytes, &key_bytes) {
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

    // ── helpers ──────────────────────────────────────────────────────────────

    fn tls_with(
        ca_file: Option<&str>,
        cert_file: Option<&str>,
        key_file: Option<&str>,
    ) -> EtcdTlsConfig {
        EtcdTlsConfig {
            ca_file: ca_file.map(str::to_owned),
            cert_file: cert_file.map(str::to_owned),
            key_file: key_file.map(str::to_owned),
            ..Default::default()
        }
    }

    fn tls_with_pem(ca_pem: Option<&str>, cert_pem: Option<&str>, key_pem: Option<&str>) -> EtcdTlsConfig {
        EtcdTlsConfig {
            ca_pem: ca_pem.map(str::to_owned),
            cert_pem: cert_pem.map(str::to_owned),
            key_pem: key_pem.map(str::to_owned),
            ..Default::default()
        }
    }

    // ── EtcdTlsConfig defaults ────────────────────────────────────────────────

    #[test]
    fn test_etcd_tls_config_default() {
        let tls = EtcdTlsConfig::default();
        assert!(tls.ca_file.is_none());
        assert!(tls.cert_file.is_none());
        assert!(tls.key_file.is_none());
        assert!(tls.ca_pem.is_none());
        assert!(tls.cert_pem.is_none());
        assert!(tls.key_pem.is_none());
        assert!(!tls.insecure_skip_verify);
    }

    #[test]
    fn test_config_default_no_tls() {
        let cfg = Config::default();
        assert!(cfg.tls.is_none());
        assert_eq!(cfg.host, vec!["http://127.0.0.1:2379"]);
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

    // ── connect_client validation ─────────────────────────────────────────────

    #[tokio::test]
    async fn test_connect_client_rejects_mixed_schemes() {
        let cfg = Config {
            host: vec![
                "http://127.0.0.1:2379".to_string(),
                "https://127.0.0.1:2379".to_string(),
            ],
            ..Config::default()
        };
        let result = EtcdConfigProvider::connect_client(&cfg).await.map(|_| ());
        assert_matches!(result, Err(e) if e.to_string().contains("single scheme"));
    }

    #[tokio::test]
    async fn test_connect_client_rejects_missing_scheme() {
        let cfg = Config {
            host: vec!["127.0.0.1:2379".to_string()],
            ..Config::default()
        };
        let result = EtcdConfigProvider::connect_client(&cfg).await.map(|_| ());
        assert_matches!(result, Err(e) if e.to_string().contains("missing a scheme"));
    }

    #[tokio::test]
    async fn test_connect_client_rejects_partial_mtls_cert_only() {
        let cfg = Config {
            host: vec!["https://127.0.0.1:2379".to_string()],
            tls: Some(tls_with(None, Some("cert.pem"), None)),
            ..Config::default()
        };
        let result = EtcdConfigProvider::connect_client(&cfg).await.map(|_| ());
        assert_matches!(result, Err(e) if e.to_string().contains("cert and key must be set together"));
    }

    #[tokio::test]
    async fn test_connect_client_rejects_partial_mtls_key_only() {
        let cfg = Config {
            host: vec!["https://127.0.0.1:2379".to_string()],
            tls: Some(tls_with(None, None, Some("key.pem"))),
            ..Config::default()
        };
        let result = EtcdConfigProvider::connect_client(&cfg).await.map(|_| ());
        assert_matches!(result, Err(e) if e.to_string().contains("cert and key must be set together"));
    }

    #[tokio::test]
    async fn test_connect_client_rejects_partial_inline_pem_cert_only() {
        let cfg = Config {
            host: vec!["https://127.0.0.1:2379".to_string()],
            tls: Some(tls_with_pem(None, Some("cert-content"), None)),
            ..Config::default()
        };
        let result = EtcdConfigProvider::connect_client(&cfg).await.map(|_| ());
        assert_matches!(result, Err(e) if e.to_string().contains("cert and key must be set together"));
    }

    #[tokio::test]
    async fn test_connect_client_rejects_partial_inline_pem_key_only() {
        let cfg = Config {
            host: vec!["https://127.0.0.1:2379".to_string()],
            tls: Some(tls_with_pem(None, None, Some("key-content"))),
            ..Config::default()
        };
        let result = EtcdConfigProvider::connect_client(&cfg).await.map(|_| ());
        assert_matches!(result, Err(e) if e.to_string().contains("cert and key must be set together"));
    }

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
        for cfg in [
            Config {
                host: vec!["https://etcd:2379".to_string()],
                tls: Some(tls_with(None, Some("cert.pem"), None)),
                ..Config::default()
            },
            Config {
                host: vec!["https://etcd:2379".to_string()],
                tls: Some(tls_with(None, None, Some("key.pem"))),
                ..Config::default()
            },
            Config {
                host: vec!["https://etcd:2379".to_string()],
                tls: Some(tls_with_pem(None, Some("cert"), None)),
                ..Config::default()
            },
            Config {
                host: vec!["https://etcd:2379".to_string()],
                tls: Some(tls_with_pem(None, None, Some("key"))),
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
        assert_matches!(validate_connect_config(&cfg), Ok(false));
    }

    #[test]
    fn test_validate_https_ok() {
        let cfg = Config {
            host: vec!["https://etcd:2379".to_string()],
            ..Config::default()
        };
        assert_matches!(validate_connect_config(&cfg), Ok(true));
    }

    // ── read_pem unit tests ───────────────────────────────────────────────────

    #[test]
    fn test_read_pem_none_returns_none() {
        let result = read_pem("ca", &None).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_read_pem_file_reads_and_returns_bytes() {
        let mut tmp = NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut tmp, b"file-pem-content").unwrap();
        let path = tmp.path().to_str().unwrap().to_owned();

        let result = read_pem("ca", &Some(path)).unwrap();
        assert_eq!(result, Some(b"file-pem-content".to_vec()));
    }

    #[test]
    fn test_read_pem_file_not_found_returns_error() {
        let result = read_pem("ca", &Some("/nonexistent/ca.pem".to_owned()));
        assert_matches!(result, Err(e) if e.to_string().contains("failed to read ca_file"));
    }

    // ── deserialization ───────────────────────────────────────────────────────

    #[test]
    fn test_etcd_tls_config_deserialization_files() {
        let json = r#"{"ca_file":"ca.pem","cert_file":"cert.pem","key_file":"key.pem"}"#;
        let tls: EtcdTlsConfig = serde_json::from_str(json).unwrap();
        assert_eq!(tls.ca_file.as_deref(), Some("ca.pem"));
        assert_eq!(tls.cert_file.as_deref(), Some("cert.pem"));
        assert_eq!(tls.key_file.as_deref(), Some("key.pem"));
        assert!(tls.ca_pem.is_none());
        assert!(tls.cert_pem.is_none());
        assert!(tls.key_pem.is_none());
        assert!(!tls.insecure_skip_verify);
    }

    #[test]
    fn test_etcd_tls_config_deserialization_inline_pem() {
        let json = r#"{"ca_pem":"ca-content","cert_pem":"cert-content","key_pem":"key-content"}"#;
        let tls: EtcdTlsConfig = serde_json::from_str(json).unwrap();
        assert_eq!(tls.ca_pem.as_deref(), Some("ca-content"));
        assert_eq!(tls.cert_pem.as_deref(), Some("cert-content"));
        assert_eq!(tls.key_pem.as_deref(), Some("key-content"));
        assert!(tls.ca_file.is_none());
        assert!(tls.cert_file.is_none());
        assert!(tls.key_file.is_none());
        assert!(!tls.insecure_skip_verify);
    }

    #[test]
    fn test_etcd_tls_config_deserialization_insecure() {
        let json = r#"{"insecure_skip_verify":true}"#;
        let tls: EtcdTlsConfig = serde_json::from_str(json).unwrap();
        assert!(tls.insecure_skip_verify);
        assert!(tls.ca_file.is_none());
        assert!(tls.ca_pem.is_none());
    }

    #[test]
    fn test_etcd_tls_config_insecure_defaults_false() {
        let json = r#"{}"#;
        let tls: EtcdTlsConfig = serde_json::from_str(json).unwrap();
        assert!(!tls.insecure_skip_verify);
    }

    #[test]
    fn test_config_deserialization_with_tls() {
        let json = r#"{
            "host": ["https://etcd.example.com:2379"],
            "prefix": "/aisix",
            "timeout": 30,
            "tls": {"ca_file": "ca.pem", "insecure_skip_verify": false}
        }"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.host, vec!["https://etcd.example.com:2379"]);
        let tls = cfg.tls.unwrap();
        assert_eq!(tls.ca_file.as_deref(), Some("ca.pem"));
        assert!(!tls.insecure_skip_verify);
    }

    #[test]
    fn test_config_deserialization_with_inline_pem_tls() {
        let json = r#"{
            "host": ["https://etcd.example.com:2379"],
            "prefix": "/aisix",
            "timeout": 30,
            "tls": {"ca_pem": "ca-content", "cert_pem": "cert-content", "key_pem": "key-content"}
        }"#;
        let cfg: Config = serde_json::from_str(json).unwrap();
        let tls = cfg.tls.unwrap();
        assert_eq!(tls.ca_pem.as_deref(), Some("ca-content"));
        assert_eq!(tls.cert_pem.as_deref(), Some("cert-content"));
        assert_eq!(tls.key_pem.as_deref(), Some("key-content"));
        assert!(tls.ca_file.is_none());
    }

    #[test]
    fn test_config_deserialization_without_tls() {
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
