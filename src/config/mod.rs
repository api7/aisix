use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use std::sync::Arc;
use tokio::sync::mpsc;

pub mod entities;
mod etcd;

#[derive(Clone, Debug, Deserialize)]
pub struct Deployment {
    pub etcd: etcd::Config,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct Config {
    pub deployment: Deployment,
}

/// Load configuration file
pub fn load() -> Result<Config, config::ConfigError> {
    config::Config::builder()
        .add_source(config::File::with_name("config"))
        .build()?
        .try_deserialize::<Config>()
}

pub async fn create_provider(config: Config) -> Arc<dyn ConfigProvider + Send + Sync> {
    Arc::new(etcd::EtcdConfigProvider::new(config.deployment.etcd.clone()).await)
}

type ConfigItemKey = String;
type ConfigItemValue = Vec<u8>;
type ConfigItemRevision = i64;

#[derive(Clone, Debug)]
pub enum ConfigEvent {
    Put((ConfigItemKey, ConfigItemValue, ConfigItemRevision)),
    Delete((ConfigItemKey, ConfigItemRevision)),
}
pub type ConfigEventReceiver = mpsc::Receiver<ConfigEvent>;

#[async_trait]
pub trait ConfigProvider {
    async fn get_all(
        &self,
        prefix: Option<&str>,
    ) -> Result<Vec<(ConfigItemKey, ConfigItemValue, ConfigItemRevision)>, String>;

    async fn get(&self, key: &str)
    -> Result<Option<(ConfigItemValue, ConfigItemRevision)>, String>;

    async fn put(&self);

    async fn watch(&self, prefix: Option<&str>) -> Option<ConfigEventReceiver>;
}
