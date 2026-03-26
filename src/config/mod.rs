pub mod entities;
pub mod etcd;
mod types;

use std::sync::Arc;

use anyhow::Result;
pub use types::*;

/// Load configuration file
pub fn load(config_file: Option<String>) -> Result<Config, config::ConfigError> {
    let mut builder = config::Config::builder();

    if let Some(ref file) = config_file {
        // If a config file is specified, it must exist
        builder = builder.add_source(config::File::with_name(file).required(true));
    } else {
        // If no config file is specified, use the default "config" file, which is optional
        builder = builder.add_source(config::File::with_name("config").required(false));
    }

    builder
        .build()?
        // If the file cannot be found, the `Config::default()` will be used.
        .try_deserialize::<Config>()
}

pub async fn create_provider(config: &Config) -> Result<Arc<dyn ConfigProvider>> {
    Ok(Arc::new(
        etcd::EtcdConfigProvider::new(config.deployment.etcd.clone()).await?,
    ))
}
