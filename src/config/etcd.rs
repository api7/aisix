use std::{fmt::Debug, sync::Arc, time::Duration};

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use backon::{ConstantBuilder, Retryable};
use dashmap::{DashMap, Entry};
use etcd_client::{GetOptions, PutOptions, WatchOptions};
use log::{debug, error, warn};
use openssl::{
    pkey::{self, PKey},
    x509::X509,
};
use serde::{Deserialize, Deserializer, de};
use thiserror::Error;
use tokio::{
    sync::{Mutex, Notify, mpsc},
    task::JoinHandle,
    time::sleep,
};

use crate::config::{ConfigEvent, ConfigEventReceiver, ConfigProvider, GetEntry, PutEntry};

// ── TLS certificate material ──────────────────────────────────────────────────

/// Client mTLS material for connecting to etcd over HTTPS.
///
/// Stores the eagerly parsed client certificate chain and private key.
#[derive(Clone)]
pub struct EtcdMtlsConfig {
    certs: Vec<X509>,
    key: PKey<pkey::Private>,
}

impl EtcdMtlsConfig {
    /// Return the parsed client certificate chain.
    pub fn certs(&self) -> &[X509] {
        &self.certs
    }

    /// Return the parsed client private key.
    pub fn key(&self) -> &PKey<pkey::Private> {
        &self.key
    }
}

impl Debug for EtcdMtlsConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EtcdTlsCertConfig")
            .field("certs", &self.certs)
            .field("key", &"***redacted***")
            .finish()
    }
}

/// TLS settings for connecting to etcd over HTTPS.
///
/// Carries an optional parsed CA certificate chain, optional client-certificate
/// material, and the `insecure_skip_verify` switch. PEM material is loaded and
/// parsed during deserialization.
#[derive(Clone, Debug, Default)]
pub struct EtcdTlsConfig {
    /// Optional parsed CA certificate chain.
    pub ca_cert: Option<Vec<X509>>,

    /// Optional client certificate material.
    pub client_cert: Option<EtcdMtlsConfig>,

    /// Skip TLS certificate verification entirely.
    ///
    /// **WARNING**: This disables all certificate validation including hostname
    /// and CA checks. Use only in development or testing environments.
    pub insecure_skip_verify: bool,
}

impl<'de> Deserialize<'de> for EtcdTlsConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawEtcdTlsConfig {
            insecure_skip_verify: Option<bool>,
            ca: Option<String>,
            ca_file: Option<String>,
            cert: Option<String>,
            cert_file: Option<String>,
            key: Option<String>,
            key_file: Option<String>,
        }

        let value = RawEtcdTlsConfig::deserialize(deserializer)?;

        let ca_cert = match Self::map_err(Self::extract_pem("ca", &value.ca, &value.ca_file))? {
            Some(pem) => Some(Self::map_err(Self::parse_x509_chain("ca", &pem))?),
            _ => None,
        };

        let client_cert = match (
            Self::map_err(Self::extract_pem("cert", &value.cert, &value.cert_file))?,
            Self::map_err(Self::extract_pem("key", &value.key, &value.key_file))?,
        ) {
            (Some(cert_pem), Some(key_pem)) => Some(EtcdMtlsConfig {
                certs: Self::map_err(Self::parse_x509_chain("cert", &cert_pem))?,
                key: Self::map_err(PKey::private_key_from_pem(key_pem.as_bytes()))?,
            }),
            (None, None) => None,
            _ => {
                return Err(de::Error::custom("cert and key must both be provided"));
            }
        };

        Ok(Self {
            ca_cert,
            client_cert,
            insecure_skip_verify: value.insecure_skip_verify.unwrap_or(false),
        })
    }
}

impl EtcdTlsConfig {
    fn map_err<T, E: std::fmt::Display, D: de::Error>(
        result: std::result::Result<T, E>,
    ) -> std::result::Result<T, D> {
        result.map_err(D::custom)
    }

    fn extract_pem(
        label: &str,
        pem: &Option<String>,
        pem_file: &Option<String>,
    ) -> Result<Option<String>> {
        match (pem, pem_file) {
            (Some(_), Some(_)) => Err(anyhow!(
                "both {label} and {label}_file are set; only one must be provided"
            )),
            (Some(pem), None) => Ok(Some(pem.clone())),
            (None, Some(path)) => std::fs::read_to_string(path)
                .with_context(|| format!("failed to read {label} file from \"{path}\"",))
                .map(Some),
            _ => Ok(None),
        }
    }

    fn parse_x509_chain(label: &str, pem: &str) -> Result<Vec<X509>> {
        let certs = X509::stack_from_pem(pem.as_bytes())
            .with_context(|| format!("etcd TLS: failed to parse {label} PEM"))?;
        if certs.is_empty() {
            return Err(anyhow!("etcd TLS: failed to parse {label} PEM"));
        }
        Ok(certs)
    }
}

// ── Config validation errors ─────────────────────────────────────────────────

/// Errors produced during etcd connection-configuration validation.
#[derive(Debug, Error)]
pub enum EtcdConfigError {
    #[error("etcd host list is empty")]
    MissingHost,

    /// The host list contains a mix of `http://` and `https://` endpoints,
    /// which is unsupported.
    #[error("etcd hosts must use a single scheme (all http:// or all https://)")]
    MixedSchemes,

    /// One of the host strings is missing the `http://` or `https://` scheme
    /// prefix.
    #[error("etcd host '{0}' is missing a scheme; use the prefix http:// or https://")]
    MissingScheme(String),
}

/// Validate the connection configuration before attempting any I/O.
///
/// Returns `Ok(())` if the configuration is valid, or an [`EtcdConfigError`]
/// describing the first validation failure.
fn validate_connect_config(config: &Config) -> Result<(), EtcdConfigError> {
    if config.host.is_empty() {
        return Err(EtcdConfigError::MissingHost);
    }

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

impl Debug for Config {
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
        validate_connect_config(&config)?;

        let client = (|| Self::connect_client(&config))
            .retry(
                ConstantBuilder::default()
                    .with_delay(Duration::from_secs(5))
                    .with_max_times(5),
            )
            .notify(|err, dur| error!("failed to connect to etcd: {err}, retrying after {:?}", dur))
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

        if config.host.iter().any(|h| h.starts_with("https://")) {
            let mut tls_cfg = etcd_client::OpenSslClientConfig::default();
            if let Some(t) = &config.tls {
                if t.insecure_skip_verify {
                    tls_cfg = tls_cfg.manually(|b| {
                        b.set_verify(openssl::ssl::SslVerifyMode::NONE);
                        Ok(())
                    });
                }

                if let Some(ca) = &t.ca_cert {
                    tls_cfg = tls_cfg.manually(move |cb| {
                        for cert in ca.iter() {
                            cb.cert_store_mut().add_cert(cert.to_owned())?;
                        }
                        Ok(())
                    });
                }

                if let Some(client_cert) = &t.client_cert {
                    let cert = client_cert.certs();
                    let key = client_cert.key();
                    tls_cfg = tls_cfg.manually(move |cb| {
                        for (i, cert) in cert.iter().enumerate() {
                            if i == 0 {
                                cb.set_certificate(cert)?;
                            } else {
                                cb.add_extra_chain_cert(cert.to_owned())?;
                            }
                        }
                        cb.set_private_key(key)?;

                        Ok(())
                    });
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
    use serde_json::json;

    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    const CA_CERT_PEM: &str = include_str!("../../tests/fixtures/tls/ca.cer");
    const CLIENT_CERT_PEM: &str = include_str!("../../tests/fixtures/tls/client.cer");
    const CLIENT_KEY_PEM: &str = include_str!("../../tests/fixtures/tls/client.key");
    const CA_CERT_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/tls/ca.cer");
    const CLIENT_CERT_PATH: &str =
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/tls/client.cer");
    const CLIENT_KEY_PATH: &str =
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/tls/client.key");
    const INVALID_PEM: &str = "not a PEM";

    fn expected_cert_chain(pem: &str) -> Vec<Vec<u8>> {
        X509::stack_from_pem(pem.as_bytes())
            .unwrap()
            .into_iter()
            .map(|cert| cert.to_der().unwrap())
            .collect()
    }

    fn assert_cert_chain_matches(actual: &[X509], expected_pem: &str) {
        let actual = actual
            .iter()
            .map(|cert| cert.to_der().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected_cert_chain(expected_pem));
    }

    fn assert_key_matches(actual: &PKey<pkey::Private>, expected_pem: &str) {
        let expected = PKey::private_key_from_pem(expected_pem.as_bytes()).unwrap();
        assert_eq!(
            actual.private_key_to_der().unwrap(),
            expected.private_key_to_der().unwrap()
        );
    }

    fn tls_strings(ca: Option<&str>, cert: Option<&str>, key: Option<&str>) -> EtcdTlsConfig {
        assert_eq!(cert.is_some(), key.is_some());
        EtcdTlsConfig {
            ca_cert: ca.map(|ca| EtcdTlsConfig::parse_x509_chain("ca", ca).unwrap()),
            client_cert: cert.zip(key).map(|(cert, key)| EtcdMtlsConfig {
                certs: EtcdTlsConfig::parse_x509_chain("cert", cert).unwrap(),
                key: PKey::private_key_from_pem(key.as_bytes()).unwrap(),
            }),
            ..Default::default()
        }
    }

    // ── EtcdTlsConfig defaults ────────────────────────────────────────────────

    #[test]
    fn test_etcd_tls_config_default() {
        let tls = EtcdTlsConfig::default();
        assert!(tls.ca_cert.is_none());
        assert!(tls.client_cert.is_none());
        assert!(!tls.insecure_skip_verify);
    }

    #[test]
    fn test_config_default_no_tls() {
        let cfg = Config::default();
        assert!(cfg.tls.is_none());
        assert_eq!(cfg.host, vec!["http://127.0.0.1:2379"]);
    }

    // ── TLS material parsing helpers ─────────────────────────────────────────

    #[test]
    fn test_cert_config_strings_ca_pem() {
        let certs = EtcdTlsConfig::parse_x509_chain("ca", CA_CERT_PEM).unwrap();
        assert_cert_chain_matches(&certs, CA_CERT_PEM);
    }

    #[test]
    fn test_cert_config_files_reads_from_disk() {
        let pem = EtcdTlsConfig::extract_pem("ca", &None, &Some(CA_CERT_PATH.to_owned()))
            .expect("ca_file should be present")
            .unwrap();
        assert_eq!(pem, CA_CERT_PEM);
        assert_cert_chain_matches(
            &EtcdTlsConfig::parse_x509_chain("ca", &pem).unwrap(),
            CA_CERT_PEM,
        );
    }

    #[test]
    fn test_cert_config_files_missing_returns_error() {
        let cfg = serde_json::from_value::<EtcdTlsConfig>(json!({
            "ca_file": "/nonexistent/ca.pem"
        }));
        assert_matches!(
            cfg,
            Err(e) if e.to_string().contains("failed to read ca file")
        );
    }

    #[test]
    fn test_tls_config_delegates_pem_methods() {
        let tls = tls_strings(
            Some(CA_CERT_PEM),
            Some(CLIENT_CERT_PEM),
            Some(CLIENT_KEY_PEM),
        );
        assert_cert_chain_matches(tls.ca_cert.as_deref().unwrap(), CA_CERT_PEM);
        assert_cert_chain_matches(tls.client_cert.as_ref().unwrap().certs(), CLIENT_CERT_PEM);
        let key = tls.client_cert.as_ref().unwrap().key();
        assert_key_matches(key, CLIENT_KEY_PEM);
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
        let tls: EtcdTlsConfig = serde_json::from_value(json!({
            "ca_file": CA_CERT_PATH,
            "cert_file": CLIENT_CERT_PATH,
            "key_file": CLIENT_KEY_PATH,
        }))
        .unwrap();
        assert_cert_chain_matches(tls.ca_cert.as_deref().unwrap(), CA_CERT_PEM);
        let client_cert = tls.client_cert.as_ref().unwrap();
        assert_cert_chain_matches(client_cert.certs(), CLIENT_CERT_PEM);
        let key = client_cert.key();
        assert_key_matches(key, CLIENT_KEY_PEM);
        assert!(!tls.insecure_skip_verify);
    }

    #[test]
    fn test_tls_deserialize_strings() {
        let tls: EtcdTlsConfig = serde_json::from_value(json!({
            "ca": CA_CERT_PEM,
            "cert": CLIENT_CERT_PEM,
            "key": CLIENT_KEY_PEM,
        }))
        .unwrap();
        assert_cert_chain_matches(tls.ca_cert.as_deref().unwrap(), CA_CERT_PEM);
        let client_cert = tls.client_cert.as_ref().unwrap();
        assert_cert_chain_matches(client_cert.certs(), CLIENT_CERT_PEM);
        let key = client_cert.key();
        assert_key_matches(key, CLIENT_KEY_PEM);
    }

    #[test]
    fn test_tls_deserialize_mixed_sources() {
        let tls: EtcdTlsConfig = serde_json::from_value(json!({
            "ca_file": CA_CERT_PATH,
            "cert": CLIENT_CERT_PEM,
            "key": CLIENT_KEY_PEM,
        }))
        .unwrap();
        assert_cert_chain_matches(tls.ca_cert.as_deref().unwrap(), CA_CERT_PEM);
        assert_cert_chain_matches(tls.client_cert.as_ref().unwrap().certs(), CLIENT_CERT_PEM);
        assert!(!tls.insecure_skip_verify);
    }

    #[test]
    fn test_tls_deserialize_rejects_partial_inline_mtls() {
        let tls = serde_json::from_value::<EtcdTlsConfig>(json!({
            "cert": CLIENT_CERT_PEM,
        }));
        assert_matches!(
            tls,
            Err(e) if e.to_string().contains("cert and key must both be provided")
        );
    }

    #[test]
    fn test_tls_deserialize_rejects_partial_file_mtls() {
        let tls = serde_json::from_value::<EtcdTlsConfig>(json!({
            "cert_file": CLIENT_CERT_PATH,
        }));
        assert_matches!(
            tls,
            Err(e) if e.to_string().contains("cert and key must both be provided")
        );
    }

    #[test]
    fn test_tls_deserialize_rejects_invalid_inline_pem() {
        let tls = serde_json::from_value::<EtcdTlsConfig>(json!({
            "ca": INVALID_PEM,
        }));
        assert_matches!(
            tls,
            Err(e) if e.to_string().contains("failed to parse ca PEM")
        );
    }

    #[test]
    fn test_tls_deserialize_insecure_only() {
        let tls: EtcdTlsConfig = serde_json::from_value(json!({
            "insecure_skip_verify": true,
        }))
        .unwrap();
        assert!(tls.ca_cert.is_none());
        assert!(tls.client_cert.is_none());
        assert!(tls.insecure_skip_verify);
    }

    #[test]
    fn test_tls_deserialize_empty() {
        let tls: EtcdTlsConfig = serde_json::from_value(json!({})).unwrap();
        assert!(tls.ca_cert.is_none());
        assert!(tls.client_cert.is_none());
        assert!(!tls.insecure_skip_verify);
    }

    #[test]
    fn test_config_deserialize_with_tls_files() {
        let cfg: Config = serde_json::from_value(json!({
            "host": ["https://etcd.example.com:2379"],
            "prefix": "/aisix",
            "timeout": 30,
            "tls": {"ca_file": CA_CERT_PATH, "insecure_skip_verify": false}
        }))
        .unwrap();
        assert_eq!(cfg.host, vec!["https://etcd.example.com:2379"]);
        let tls = cfg.tls.unwrap();
        assert_cert_chain_matches(tls.ca_cert.as_deref().unwrap(), CA_CERT_PEM);
        assert!(tls.client_cert.is_none());
        assert!(!tls.insecure_skip_verify);
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

    #[test]
    fn test_config_deserialize_with_tls_strings() {
        let cfg: Config = serde_json::from_value(json!({
            "host": ["https://etcd.example.com:2379"],
            "prefix": "/aisix",
            "timeout": 30,
            "tls": {
                "ca": CA_CERT_PEM,
                "cert": CLIENT_CERT_PEM,
                "key": CLIENT_KEY_PEM,
            }
        }))
        .unwrap();
        let tls = cfg.tls.unwrap();
        assert_cert_chain_matches(tls.ca_cert.as_deref().unwrap(), CA_CERT_PEM);
        let client_cert = tls.client_cert.as_ref().unwrap();
        assert_cert_chain_matches(client_cert.certs(), CLIENT_CERT_PEM);
        let key = client_cert.key();
        assert_key_matches(key, CLIENT_KEY_PEM);
    }
}
