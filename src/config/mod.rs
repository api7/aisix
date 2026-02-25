pub mod entities;
mod etcd;
mod types;

use std::sync::Arc;

use anyhow::Result;

pub use types::*;

/// Load configuration file
pub fn load() -> Result<Config, config::ConfigError> {
    config::Config::builder()
        .add_source(config::File::with_name("config"))
        .build()?
        .try_deserialize::<Config>()
}

pub async fn create_provider(config: Config) -> Result<Arc<dyn ConfigProvider>> {
    Ok(Arc::new(
        etcd::EtcdConfigProvider::new(config.deployment.etcd.clone()).await?,
    ))
}
