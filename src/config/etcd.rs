use std::{collections::HashMap, sync::Arc, time::Duration};

use super::{
    ConfigEvent, ConfigEventReceiver, ConfigItemKey, ConfigItemRevision, ConfigItemValue,
    ConfigProvider,
};
use async_trait::async_trait;
use log::{error, info, warn};
use serde::Deserialize;
use tokio::sync::{Mutex, mpsc};

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub host: Vec<String>,
    pub prefix: String,
    pub timeout: u32,
    pub user: Option<String>,
    pub password: Option<String>,
}

#[derive(Clone)]
pub struct EtcdConfigProvider {
    client: etcd_client::Client,
    prefix: String,
    txs: Arc<Mutex<HashMap<String, mpsc::Sender<ConfigEvent>>>>,
}

impl EtcdConfigProvider {
    pub async fn new(config: Config) -> Self {
        let client = Self::connect_client(&config).await;

        let txs = Arc::new(Mutex::new(
            HashMap::<String, mpsc::Sender<ConfigEvent>>::new(),
        ));

        let prefix = config.prefix.clone();
        Self::spawn_watch_loop(client.clone(), prefix.clone(), txs.clone());

        Self {
            client,
            prefix: config.prefix.clone(),
            txs,
        }
    }

    fn spawn_watch_loop(
        mut client: etcd_client::Client,
        prefix: String,
        txs: Arc<Mutex<HashMap<String, mpsc::Sender<ConfigEvent>>>>,
    ) {
        tokio::spawn(async move {
            let mut backoff_secs = 1u64;
            loop {
                let watch_opts = etcd_client::WatchOptions::new().with_prefix();
                match client.watch(prefix.as_str(), Some(watch_opts)).await {
                    Ok((_watcher, mut stream)) => {
                        info!("Started etcd prefix watch: {}", prefix);
                        backoff_secs = 1;

                        while let Ok(msg) = stream.message().await {
                            let resp = match msg {
                                Some(resp) => resp,
                                None => {
                                    warn!("Watch stream ended, preparing to retry");
                                    break;
                                }
                            };

                            if resp.canceled() {
                                warn!("Watch was canceled, preparing to retry");
                                break;
                            }

                            for event in resp.events() {
                                if let Some(kv) = event.kv() {
                                    let key = match kv.key_str() {
                                        Ok(k) => k.to_string(),
                                        Err(err) => {
                                            warn!("Failed to parse watch key: {}", err);
                                            continue;
                                        }
                                    };

                                    let targets: Vec<mpsc::Sender<ConfigEvent>> = {
                                        let guard = txs.lock().await;
                                        guard
                                            .iter()
                                            .filter(|(k, _)| key.starts_with(k.as_str()))
                                            .map(|(_, tx)| tx.clone())
                                            .collect()
                                    };

                                    if targets.is_empty() {
                                        continue;
                                    }

                                    /* let payload = ConfigEvent {
                                        key: key.clone(),
                                        value: if kv.value().is_empty() {
                                            None
                                        } else {
                                            Some(kv.value().to_vec())
                                        },
                                        event_type: event.event_type().clone(),
                                        mod_revision: kv.mod_revision(),
                                    }; */
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

                                    for tx in targets {
                                        if let Err(err) = tx.send(payload.clone()).await {
                                            warn!("Failed to dispatch watch event: {}", err);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(err) => {
                        error!("Failed to start etcd watch: {}", err);
                    }
                }

                warn!("Watch error, retrying in {}s", backoff_secs);
                tokio::time::sleep(Duration::from_secs(backoff_secs)).await;
                backoff_secs = (backoff_secs * 2).min(30);
            }
        });
    }

    async fn connect_client(config: &Config) -> etcd_client::Client {
        let mut opts = etcd_client::ConnectOptions::default()
            .with_timeout(Duration::from_secs(config.timeout as u64));

        if let (Some(user), Some(password)) = (config.user.clone(), config.password.clone()) {
            opts = opts.with_user(user, password);
        }

        etcd_client::Client::connect(
            config
                .host
                .clone()
                .iter()
                .map(|h: &String| h.as_str())
                .collect::<Vec<&str>>(),
            Some(opts),
        )
        .await
        .expect("Failed to connect to etcd")
    }
}

#[async_trait]
impl ConfigProvider for EtcdConfigProvider {
    async fn get_all(
        &self,
        prefix: Option<&str>,
    ) -> Result<Vec<(ConfigItemKey, ConfigItemValue, ConfigItemRevision)>, String> {
        let full_prefix = format!("{}{}", self.prefix, prefix.unwrap_or(""));
        let get_opts = etcd_client::GetOptions::new().with_prefix();

        let mut client = self.client.clone();
        match client.get(full_prefix.as_str(), Some(get_opts)).await {
            Ok(resp) => {
                let mut results = Vec::new();
                for kv in resp.kvs() {
                    if let Ok(key) = kv.key_str() {
                        results.push((key.to_string(), kv.value().to_vec(), kv.mod_revision()));
                    }
                }
                Ok(results)
            }
            Err(err) => Err(format!("etcd get all failed: {}", err)),
        }
    }

    async fn get(
        &self,
        key: &str,
    ) -> Result<Option<(ConfigItemValue, ConfigItemRevision)>, String> {
        let key = format!("{}{}", self.prefix, key);

        let mut client = self.client.clone();
        match client.get(key.as_str(), None).await {
            Ok(resp) => {
                if let Some(kv) = resp.kvs().first() {
                    Ok(Some((kv.value().to_vec(), kv.mod_revision())))
                } else {
                    Ok(None)
                }
            }
            Err(err) => Err(format!("etcd get all failed: {}", err)),
        }
    }

    async fn put(&self) {
        todo!()
    }

    async fn watch(&self, prefix: Option<&str>) -> Option<ConfigEventReceiver> {
        let full_prefix = format!("{}{}", self.prefix, prefix.unwrap_or(""));

        {
            let guard = self.txs.lock().await;
            if guard.contains_key(&full_prefix) {
                return None;
            }
        }

        let (tx, rx) = mpsc::channel(32);
        self.txs.lock().await.insert(full_prefix, tx);

        Some(rx)
    }
}
