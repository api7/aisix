use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use dashmap::{DashMap, Entry};
use etcd_client::{GetOptions, PutOptions, WatchOptions};
use log::{debug, info, warn};
use serde::Deserialize;
use tokio::{
    sync::{Notify, mpsc},
    task::JoinHandle,
    time::sleep,
};

use crate::config::{ConfigEvent, ConfigEventReceiver, ConfigProvider, GetEntry, PutEntry};

/// Maximum backoff delay between reconnect attempts.
const MAX_BACKOFF: Duration = Duration::from_secs(60);
/// Initial backoff delay.
const INITIAL_BACKOFF: Duration = Duration::from_secs(1);
/// Timeout for waiting the watch supervisor task to stop on shutdown.
const SHUTDOWN_WAIT: Duration = Duration::from_secs(10);

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub host: Vec<String>,
    pub prefix: String,
    pub timeout: u32,
    pub user: Option<String>,
    pub password: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            host: vec!["http://127.0.0.1:2379".to_string()],
            prefix: "/aisix".to_string(),
            timeout: 5,
            user: None,
            password: None,
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
        let client = Self::connect_client(&config).await?;
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

        etcd_client::Client::connect(
            config
                .host
                .iter()
                .map(|h: &String| h.as_str())
                .collect::<Vec<&str>>(),
            Some(opts),
        )
        .await
        .map_err(|e| anyhow::anyhow!("Failed to connect to etcd: {e}"))
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
                        info!("etcd watch supervisor: shutdown requested before stream open");
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
                            info!("etcd watch supervisor: shutdown requested");
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

    async fn shutdown(&self) -> anyhow::Result<()> {
        // Signal the supervisor to stop.
        self.shutdown.notify_one();

        // Close all dispatch channels so consumers see channel-closed.
        self.txs.clear();

        // Wait for the supervisor task to exit (with timeout).
        let handle = self.supervisor_handle.lock().unwrap().take();
        if let Some(h) = handle {
            match tokio::time::timeout(SHUTDOWN_WAIT, h).await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => warn!("etcd supervisor task panicked: {}", e),
                Err(_) => warn!(
                    "etcd supervisor task did not stop within {:?}",
                    SHUTDOWN_WAIT
                ),
            }
        }
        Ok(())
    }
}
