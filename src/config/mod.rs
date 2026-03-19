pub mod entities;
pub mod etcd;
mod types;

use std::sync::Arc;

use anyhow::Result;
pub use types::*;

/// Load configuration file
pub fn load(config_file: Option<String>) -> Result<Config, config::ConfigError> {
    config::Config::builder()
        .add_source(config::File::with_name(
            config_file.as_deref().unwrap_or("config"),
        ))
        .build()?
        .try_deserialize::<Config>()
}

pub async fn create_provider(config: &Config) -> Result<Arc<dyn ConfigProvider>> {
    Ok(Arc::new(
        etcd::EtcdConfigProvider::new(config.deployment.etcd.clone()).await?,
    ))
}
