//! Bootstrap configuration loaded from a YAML/TOML/JSON file at startup.
//!
//! Everything in here is the *static* config (addresses, TLS, etcd endpoints,
//! observability sinks). Dynamic resources — Models, API keys, budgets — live
//! in etcd and are loaded via the `aisix-etcd` crate.
//!
//! Loading order (spec §2):
//! 1. Defaults
//! 2. File contents (path from CLI `--config` or discovery list)
//! 3. Environment-variable overrides (prefix `AISIX_`, separator `__`)
//!
//! Example (see `config.example.yaml`):
//!
//! ```yaml
//! etcd:
//!   endpoints: ["http://127.0.0.1:2379"]
//!   prefix: "/aisix"
//! proxy:
//!   addr: "0.0.0.0:3000"
//! admin:
//!   addr: "127.0.0.1:3001"
//!   admin_keys: ["admin-local-only-change-me"]
//! ```

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;

use crate::error::BootstrapError;

/// Root config struct. Construct via [`Config::load_from_path`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub etcd: EtcdConfig,
    pub proxy: ProxyConfig,
    pub admin: AdminConfig,
    #[serde(default)]
    pub observability: ObservabilityConfig,
    #[serde(default)]
    pub cache: CacheConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EtcdConfig {
    pub endpoints: Vec<String>,
    #[serde(default = "EtcdConfig::default_prefix")]
    pub prefix: String,
    #[serde(default)]
    pub user: Option<String>,
    /// Name of the env var that contains the password. The actual secret is
    /// read at connect time — never stored in the config struct.
    #[serde(default)]
    pub password_env: Option<String>,
    #[serde(default = "EtcdConfig::default_dial_timeout")]
    pub dial_timeout_ms: u64,
    #[serde(default = "EtcdConfig::default_request_timeout")]
    pub request_timeout_ms: u64,
}

impl EtcdConfig {
    fn default_prefix() -> String {
        "/aisix".into()
    }
    const fn default_dial_timeout() -> u64 {
        5_000
    }
    const fn default_request_timeout() -> u64 {
        5_000
    }

    pub const fn dial_timeout(&self) -> Duration {
        Duration::from_millis(self.dial_timeout_ms)
    }

    pub const fn request_timeout(&self) -> Duration {
        Duration::from_millis(self.request_timeout_ms)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProxyConfig {
    pub addr: String,
    #[serde(default = "ProxyConfig::default_body_limit")]
    pub request_body_limit_bytes: usize,
    #[serde(default)]
    pub tls: Option<TlsConfig>,
}

impl ProxyConfig {
    const fn default_body_limit() -> usize {
        10 * 1024 * 1024
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdminConfig {
    pub addr: String,
    /// Statically-provisioned admin keys. A request is authorised if it
    /// presents any of these via `Authorization: Bearer <k>` or `x-api-key`.
    pub admin_keys: Vec<String>,
    #[serde(default)]
    pub tls: Option<TlsConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TlsConfig {
    pub cert_file: String,
    pub key_file: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ObservabilityConfig {
    #[serde(default = "ObservabilityConfig::default_service_name")]
    pub service_name: String,
    #[serde(default = "ObservabilityConfig::default_log_level")]
    pub log_level: String,
    #[serde(default = "ObservabilityConfig::default_access_log")]
    pub access_log: bool,
    pub metrics: MetricsConfig,
    pub tracing: TracingConfig,
    pub langfuse: LangfuseConfig,
}

impl ObservabilityConfig {
    fn default_service_name() -> String {
        "aisix".into()
    }
    fn default_log_level() -> String {
        "info".into()
    }
    const fn default_access_log() -> bool {
        true
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct MetricsConfig {
    pub prometheus: PrometheusConfig,
    pub otlp: OtlpConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct PrometheusConfig {
    pub enabled: bool,
    pub path: String,
}

impl Default for PrometheusConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            path: "/metrics".into(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct OtlpConfig {
    pub enabled: bool,
    pub endpoint: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct TracingConfig {
    pub otlp: OtlpTracingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct OtlpTracingConfig {
    pub enabled: bool,
    pub endpoint: Option<String>,
    pub sample_ratio: f64,
}

impl Default for OtlpTracingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: None,
            sample_ratio: 1.0,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct LangfuseConfig {
    pub enabled: bool,
    pub host: Option<String>,
    pub public_key_env: Option<String>,
    pub secret_key_env: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct CacheConfig {
    pub backend: CacheBackend,
    pub redis: Option<RedisCacheConfig>,
    pub qdrant: Option<QdrantCacheConfig>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            backend: CacheBackend::Memory,
            redis: None,
            qdrant: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CacheBackend {
    Memory,
    Redis,
    Qdrant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RedisCacheConfig {
    pub url: String,
    #[serde(default = "RedisCacheConfig::default_mode")]
    pub mode: String,
}

impl RedisCacheConfig {
    fn default_mode() -> String {
        "single".into()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QdrantCacheConfig {
    pub url: String,
    pub collection: String,
}

impl Config {
    /// Load + merge + validate.
    ///
    /// - If `path` is Some, the file is loaded (format inferred from extension).
    /// - Env vars prefixed `AISIX_` override anything in the file.
    /// - Basic invariants are checked (non-empty etcd endpoints, at least one
    ///   admin key, bind addresses parse).
    pub fn load_from_path(path: Option<&Path>) -> Result<Self, BootstrapError> {
        use ::config::{Config as CConfig, Environment, File};

        let mut builder = CConfig::builder();

        if let Some(p) = path {
            let source = File::from(p).required(true);
            builder = builder.add_source(source);
        }

        builder = builder.add_source(
            Environment::with_prefix("AISIX")
                .separator("__")
                .list_separator(",")
                .try_parsing(true),
        );

        let raw = builder
            .build()
            .map_err(|e| BootstrapError::Config(format!("build: {e}")))?;

        let cfg: Self = raw
            .try_deserialize()
            .map_err(|e| BootstrapError::Config(format!("deserialize: {e}")))?;

        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<(), BootstrapError> {
        if self.etcd.endpoints.is_empty() {
            return Err(BootstrapError::Config(
                "etcd.endpoints must contain at least one endpoint".into(),
            ));
        }
        if self.admin.admin_keys.is_empty() {
            return Err(BootstrapError::Config(
                "admin.admin_keys must contain at least one key".into(),
            ));
        }
        if self.proxy.addr.parse::<std::net::SocketAddr>().is_err() {
            return Err(BootstrapError::Config(format!(
                "proxy.addr invalid socket address: {}",
                self.proxy.addr
            )));
        }
        if self.admin.addr.parse::<std::net::SocketAddr>().is_err() {
            return Err(BootstrapError::Config(format!(
                "admin.addr invalid socket address: {}",
                self.admin.addr
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_yaml(body: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::Builder::new().suffix(".yaml").tempfile().unwrap();
        f.write_all(body.as_bytes()).unwrap();
        f
    }

    #[test]
    fn loads_minimal_config() {
        let f = write_yaml(
            r#"
etcd:
  endpoints: ["http://127.0.0.1:2379"]
  prefix: "/aisix"
proxy:
  addr: "0.0.0.0:3000"
admin:
  addr: "127.0.0.1:3001"
  admin_keys: ["k1"]
"#,
        );
        let cfg = Config::load_from_path(Some(f.path())).unwrap();
        assert_eq!(cfg.etcd.endpoints, vec!["http://127.0.0.1:2379"]);
        assert_eq!(cfg.proxy.request_body_limit_bytes, 10 * 1024 * 1024);
        assert!(cfg.observability.metrics.prometheus.enabled);
        assert_eq!(cfg.cache.backend, CacheBackend::Memory);
    }

    #[test]
    fn rejects_empty_etcd_endpoints() {
        let f = write_yaml(
            r#"
etcd:
  endpoints: []
proxy:
  addr: "0.0.0.0:3000"
admin:
  addr: "127.0.0.1:3001"
  admin_keys: ["k1"]
"#,
        );
        let err = Config::load_from_path(Some(f.path())).unwrap_err();
        assert!(err.to_string().contains("etcd.endpoints"));
    }

    #[test]
    fn rejects_empty_admin_keys() {
        let f = write_yaml(
            r#"
etcd:
  endpoints: ["http://localhost:2379"]
proxy:
  addr: "0.0.0.0:3000"
admin:
  addr: "127.0.0.1:3001"
  admin_keys: []
"#,
        );
        let err = Config::load_from_path(Some(f.path())).unwrap_err();
        assert!(err.to_string().contains("admin.admin_keys"));
    }

    #[test]
    fn rejects_invalid_bind_addr() {
        let f = write_yaml(
            r#"
etcd:
  endpoints: ["http://localhost:2379"]
proxy:
  addr: "not-a-socket-addr"
admin:
  addr: "127.0.0.1:3001"
  admin_keys: ["k1"]
"#,
        );
        let err = Config::load_from_path(Some(f.path())).unwrap_err();
        assert!(err.to_string().contains("proxy.addr"));
    }

    #[test]
    fn rejects_unknown_fields() {
        let f = write_yaml(
            r#"
etcd:
  endpoints: ["http://localhost:2379"]
proxy:
  addr: "0.0.0.0:3000"
admin:
  addr: "127.0.0.1:3001"
  admin_keys: ["k1"]
bogus_field: 1
"#,
        );
        let err = Config::load_from_path(Some(f.path())).unwrap_err();
        assert!(err.to_string().contains("bogus_field"));
    }
}
