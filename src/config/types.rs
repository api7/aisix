use std::net::SocketAddr;

use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use tokio::sync::mpsc;

use crate::config::etcd;

pub mod defaults {
    use super::*;

    pub fn listen() -> SocketAddr {
        "0.0.0.0:3000".parse().unwrap()
    }

    pub fn admin_listen() -> SocketAddr {
        "127.0.0.1:3001".parse().unwrap()
    }

    pub fn server() -> Server {
        Server::default()
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct AdminKey {
    pub key: String,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct DeploymentAdmin {
    #[serde(default = "Vec::new")]
    pub admin_key: Vec<AdminKey>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Deployment {
    #[serde(default)]
    pub etcd: etcd::Config,
    #[serde(default)]
    pub admin: DeploymentAdmin,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct ServerCommonTls {
    #[serde(default)]
    pub enabled: bool,
    pub cert_file: Option<String>,
    pub key_file: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ServerProxy {
    #[serde(default = "defaults::listen")]
    pub listen: SocketAddr,
    #[serde(default)]
    pub tls: ServerCommonTls,
}

impl Default for ServerProxy {
    fn default() -> Self {
        Self {
            listen: defaults::listen(),
            tls: ServerCommonTls::default(),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct ServerAdmin {
    #[serde(default = "defaults::admin_listen")]
    pub listen: SocketAddr,
    #[serde(default)]
    pub tls: ServerCommonTls,
}

impl Default for ServerAdmin {
    fn default() -> Self {
        Self {
            listen: defaults::admin_listen(),
            tls: ServerCommonTls::default(),
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Server {
    #[serde(default)]
    pub proxy: ServerProxy,
    #[serde(default)]
    pub admin: ServerAdmin,
}

#[derive(Clone, Debug, Deserialize, Default)]
pub struct Config {
    #[serde(default)]
    pub deployment: Deployment,
    #[serde(default)]
    pub server: Server,
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

pub struct GetEntry<T> {
    pub key: String,
    pub value: T,
    pub create_revision: ConfigItemRevision,
    pub mod_revision: ConfigItemRevision,
}

pub enum PutEntry<T> {
    Created,
    Updated(GetEntry<T>),
}

#[async_trait]
pub trait ConfigProvider: Send + Sync {
    async fn get_all_raw(&self, prefix: Option<&str>) -> Result<Vec<GetEntry<Vec<u8>>>, String>;

    async fn get_raw(&self, key: &str) -> Result<Option<GetEntry<Vec<u8>>>, String>;

    async fn put_raw(&self, key: &str, value: Vec<u8>) -> Result<PutEntry<Vec<u8>>, String>;

    async fn delete(&self, key: &str) -> Result<i64, String>;

    async fn watch(&self, prefix: Option<&str>) -> Result<ConfigEventReceiver>;
}

impl dyn ConfigProvider {
    pub async fn get_all<T: serde::de::DeserializeOwned + Send>(
        &self,
        prefix: &str,
    ) -> Result<Vec<GetEntry<T>>, String> {
        let items = self.get_all_raw(Some(prefix)).await?;
        Ok(items
            .iter()
            .filter_map(|item| match serde_json::from_slice::<T>(&item.value) {
                Ok(parsed) => Some(GetEntry {
                    key: item.key.clone(),
                    value: parsed,
                    create_revision: item.create_revision,
                    mod_revision: item.mod_revision,
                }),
                Err(err) => {
                    log::warn!("Failed to parse config item {}: {}", item.key, err);
                    None
                }
            })
            .collect::<Vec<GetEntry<T>>>())
    }

    pub async fn get<T: serde::de::DeserializeOwned + Send>(
        &self,
        key: &str,
    ) -> Result<Option<GetEntry<T>>, String> {
        match self.get_raw(key).await? {
            Some(GetEntry {
                key,
                value,
                create_revision,
                mod_revision,
            }) => match serde_json::from_slice::<T>(&value) {
                Ok(parsed) => Ok(Some(GetEntry {
                    key,
                    value: parsed,
                    create_revision,
                    mod_revision,
                })),
                Err(err) => Err(format!("Failed to parse config item {}: {}", key, err)),
            },
            None => Ok(None),
        }
    }

    pub async fn put<T: serde::de::DeserializeOwned + serde::ser::Serialize + Send>(
        &self,
        key: &str,
        value: &T,
    ) -> Result<PutEntry<T>, String> {
        let value_bytes = serde_json::to_vec(value)
            .map_err(|err| format!("Failed to serialize config item {}: {}", key, err))?;

        match self.put_raw(key, value_bytes).await? {
            PutEntry::Created => Ok(PutEntry::Created),
            PutEntry::Updated(GetEntry {
                key,
                value,
                create_revision,
                mod_revision,
            }) => Ok(PutEntry::Updated(GetEntry {
                key: key.clone(),
                value: serde_json::from_slice::<T>(&value)
                    .map_err(|err| format!("Failed to parse config item {}: {}", key, err))?,
                create_revision,
                mod_revision,
            })),
        }
    }
}
