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
    /// Admin surface. Defaulted so managed-mode configs can omit this
    /// block entirely; the default values are NOT bound at runtime —
    /// [`ManagedConfig::is_managed`] gates the listener.
    #[serde(default)]
    pub admin: AdminConfig,
    #[serde(default)]
    pub observability: ObservabilityConfig,
    #[serde(default)]
    pub cache: CacheConfig,
    /// Optional managed-mode configuration. When `managed.enabled = true`
    /// the admin API, admin UI, and Playground endpoints are **not**
    /// bound — the DP is a pure etcd reader driven by the aisix.cloud
    /// control plane. Missing or `enabled = false` runs standalone.
    #[serde(default)]
    pub managed: ManagedConfig,
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
    /// Optional TLS / mTLS bundle used to authenticate to the etcd
    /// endpoint. Required when talking to an aisix.cloud DP Manager
    /// (see prd-09 §9.3.3 — the CP issues a 10-year client cert via
    /// `IssueAIDataplaneCertificate`). Leave unset for plain-HTTP
    /// etcd (local dev, integration tests).
    #[serde(default)]
    pub tls: Option<EtcdTlsConfig>,
}

/// Paths to the mTLS bundle used for etcd client auth. Files are read
/// lazily at connect time — absent files surface as a BootstrapError.
///
/// When `domain_name` is unset, callers typically derive it from the
/// first endpoint's hostname so the tonic TLS layer knows what SNI /
/// cert-subject-alt-name to match against.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EtcdTlsConfig {
    /// PEM-encoded CA bundle used to verify the etcd server cert.
    pub ca_cert_file: String,
    /// PEM-encoded client certificate (from `IssueAIDataplaneCertificate`).
    pub client_cert_file: String,
    /// PEM-encoded client private key. Paired with `client_cert_file`.
    pub client_key_file: String,
    /// Expected server name for TLS verification. Usually the hostname
    /// portion of `etcd.endpoints[0]`. Only required when the CA issues
    /// certs under a different SNI than the endpoint DNS name.
    #[serde(default)]
    pub domain_name: Option<String>,
}

/// Optional managed-mode configuration (prd-09 §9.2.2).
///
/// When `enabled = true`, aisix runs as a tenant of aisix.cloud:
///
/// - The admin API listener is **not** bound.
/// - The admin UI is **not** served.
/// - The Playground endpoint is **not** exposed.
///
/// All configuration is read from etcd via the TLS channel (see
/// [`EtcdTlsConfig`]).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ManagedConfig {
    pub enabled: bool,

    /// When set, aisix performs a one-shot `POST /dp/register` against
    /// `cp_base_url` at boot to exchange this token for an mTLS bundle
    /// and a dp_id. Subsequent boots detect the existing bundle at
    /// `mtls_dir` and skip re-registration (the token is single-use).
    ///
    /// Leave empty if the mTLS bundle is already on disk — typical for
    /// configs installed via an out-of-band "download bundle" flow.
    #[serde(default)]
    pub registration_token: Option<String>,

    /// aisix.cloud CP base URL, e.g. "https://api.us.aisix.cloud".
    /// Required whenever `registration_token` is set.
    #[serde(default)]
    pub cp_base_url: Option<String>,

    /// Directory where the DP persists `ca.crt`, `client.crt`,
    /// `client.key` received from the register response. Files are
    /// written `0600`. Parent directory must already exist and be
    /// writable by the aisix process user.
    #[serde(default = "ManagedConfig::default_mtls_dir")]
    pub mtls_dir: String,

    /// File where the DP persists its `dp_id`. Read back on restart
    /// for heartbeat / telemetry payloads. Same permission rules as
    /// the mTLS files.
    #[serde(default = "ManagedConfig::default_dp_id_file")]
    pub dp_id_file: String,

    /// Optional path to the on-disk snapshot cache the DP keeps as a
    /// fallback when etcd is unreachable (prd-09 §9.7.2). When set, the
    /// supervisor flushes every applied resync / put / delete to this
    /// file and re-loads it at boot before opening the etcd connection,
    /// so the proxy can serve traffic from cached config across CP
    /// outages and full container restarts.
    ///
    /// Empty string disables persistence — useful for ephemeral test
    /// runs where you don't want a stale cache to mask a real failure.
    #[serde(default = "ManagedConfig::default_snapshot_cache_path")]
    pub snapshot_cache_path: String,
}

impl ManagedConfig {
    /// True if the DP should behave as an aisix.cloud tenant.
    pub const fn is_managed(&self) -> bool {
        self.enabled
    }

    /// True when both the token and CP URL are set — i.e. the DP
    /// should attempt `/dp/register` at boot.
    pub fn registration_enabled(&self) -> bool {
        self.registration_token
            .as_deref()
            .is_some_and(|s| !s.is_empty())
            && self.cp_base_url.as_deref().is_some_and(|s| !s.is_empty())
    }

    fn default_mtls_dir() -> String {
        "/var/lib/aisix/mtls".into()
    }
    fn default_dp_id_file() -> String {
        "/var/lib/aisix/dp_id".into()
    }
    fn default_snapshot_cache_path() -> String {
        "/var/lib/aisix/config_cache.json".into()
    }
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
    #[serde(default = "AdminConfig::default_addr")]
    pub addr: String,
    /// Statically-provisioned admin keys. A request is authorised if it
    /// presents any of these via `Authorization: Bearer <k>` or `x-api-key`.
    #[serde(default)]
    pub admin_keys: Vec<String>,
    #[serde(default)]
    pub tls: Option<TlsConfig>,
}

impl AdminConfig {
    fn default_addr() -> String {
        // Intentionally non-routable. Managed-mode configs never bind
        // this; standalone configs are rejected by `Config::validate`
        // if they leave it at the default without overriding.
        "127.0.0.1:0".into()
    }
}

impl Default for AdminConfig {
    fn default() -> Self {
        Self {
            addr: Self::default_addr(),
            admin_keys: Vec::new(),
            tls: None,
        }
    }
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

        // config-rs default: when `separator` is set, the prefix
        // separator inherits from it — so `separator("__")` alone
        // would demand `AISIX__FOO__BAR` env vars. That's at odds
        // with every other aisix.cloud service (and the existing
        // docs / Dockerfile / e2e harness), which all use
        // `AISIX_FOO__BAR` (single underscore between prefix and
        // first key segment, double underscore for nested keys).
        // Pin prefix_separator explicitly so the two shapes are
        // distinct: `AISIX_` strips the prefix, `__` splits keys.
        builder = builder.add_source(
            Environment::with_prefix("AISIX")
                .prefix_separator("_")
                .separator("__")
                // Per-key list parsing. Setting `list_separator`
                // without explicit `with_list_parse_key` would force
                // EVERY string env override through comma-splitting —
                // which blows up secrets that happen to contain a
                // comma (AISIX_MANAGED__REGISTRATION_TOKEN has been
                // the visible victim) with a serde "invalid type:
                // sequence, expected a string" error. Opt in only for
                // fields that are actually Vec<String>.
                .list_separator(",")
                .with_list_parse_key("etcd.endpoints")
                .with_list_parse_key("admin.admin_keys")
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
        // In managed mode the admin listener is not bound, so requiring
        // admin_keys or a valid admin.addr would be punishing the user
        // for fields that aren't going to be used. In standalone mode
        // we keep the original invariants.
        if !self.managed.is_managed() {
            if self.admin.admin_keys.is_empty() {
                return Err(BootstrapError::Config(
                    "admin.admin_keys must contain at least one key \
                     (required when managed.enabled is false)"
                        .into(),
                ));
            }
            if self.admin.addr.parse::<std::net::SocketAddr>().is_err() {
                return Err(BootstrapError::Config(format!(
                    "admin.addr invalid socket address: {}",
                    self.admin.addr
                )));
            }
        }
        if self.proxy.addr.parse::<std::net::SocketAddr>().is_err() {
            return Err(BootstrapError::Config(format!(
                "proxy.addr invalid socket address: {}",
                self.proxy.addr
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

    #[test]
    fn managed_mode_lets_admin_fields_be_omitted() {
        // A managed-mode config is the minimum aisix.cloud tenant
        // shape: etcd + tls + proxy + managed.enabled = true. Admin
        // keys / addr are fine to leave out entirely because the
        // admin surface is never bound.
        let f = write_yaml(
            r#"
etcd:
  endpoints: ["https://etcd.aisix.cloud:2379"]
  prefix: "/aisix"
  tls:
    ca_cert_file: "/etc/aisix/mtls/ca.crt"
    client_cert_file: "/etc/aisix/mtls/client.crt"
    client_key_file: "/etc/aisix/mtls/client.key"
proxy:
  addr: "0.0.0.0:3000"
managed:
  enabled: true
"#,
        );
        let cfg = Config::load_from_path(Some(f.path())).unwrap();
        assert!(cfg.managed.is_managed());
        assert_eq!(
            cfg.etcd.tls.as_ref().unwrap().client_cert_file,
            "/etc/aisix/mtls/client.crt"
        );
        assert!(cfg.admin.admin_keys.is_empty());
    }

    #[test]
    fn standalone_still_requires_admin_keys_even_with_managed_false() {
        // managed.enabled = false (or missing) must keep the original
        // "admin_keys must be non-empty" invariant. Otherwise a user
        // accidentally dropping admin_keys would silently lose auth
        // on their admin listener.
        let f = write_yaml(
            r#"
etcd:
  endpoints: ["http://127.0.0.1:2379"]
proxy:
  addr: "0.0.0.0:3000"
admin:
  addr: "127.0.0.1:3001"
  admin_keys: []
managed:
  enabled: false
"#,
        );
        let err = Config::load_from_path(Some(f.path())).unwrap_err();
        assert!(err.to_string().contains("admin.admin_keys"));
    }

    #[test]
    fn parses_managed_block_with_register_fields() {
        // Mirrors the shape of the baked-in config.managed.yaml so the
        // image's bootstrap template stays a valid Config; if anyone
        // adds a required ManagedConfig field they have to update both
        // the YAML and this test.
        let f = write_yaml(
            r#"
etcd:
  endpoints: ["https://placeholder:2379"]
proxy:
  addr: "0.0.0.0:3000"
admin:
  addr: "127.0.0.1:0"
  admin_keys: ["disabled"]
managed:
  enabled: true
  mtls_dir: "/var/lib/aisix/mtls"
  dp_id_file: "/var/lib/aisix/dp_id"
"#,
        );
        let cfg = Config::load_from_path(Some(f.path())).unwrap();
        assert!(cfg.managed.is_managed());
        assert_eq!(cfg.managed.mtls_dir, "/var/lib/aisix/mtls");
        assert_eq!(cfg.managed.dp_id_file, "/var/lib/aisix/dp_id");
        // Default snapshot cache path keeps offline-resilience on by
        // default; operators opt out by setting the field to "".
        assert_eq!(
            cfg.managed.snapshot_cache_path,
            "/var/lib/aisix/config_cache.json",
        );
        // Token / CP URL come from env at runtime — empty here is fine.
        assert!(cfg.managed.registration_token.is_none());
        assert!(cfg.managed.cp_base_url.is_none());
        assert!(!cfg.managed.registration_enabled());
    }

    // NOTE: the env-prefix regression (config-rs default
    // prefix_separator inheriting from separator, silently dropping
    // `AISIX_FOO__BAR` shaped vars) is covered end-to-end by
    // aisix.cloud's Go e2e suite, which runs the DP docker image
    // with `AISIX_MANAGED__REGISTRATION_TOKEN` + `AISIX_MANAGED__CP_BASE_URL`
    // and asserts on registration_enabled=true via the structured
    // boot log added in the same release. A unit test here would
    // need std::env::set_var (forbidden by the crate-level
    // #![forbid(unsafe_code)]) or an extra dev-dep for a guarded
    // env helper; the downstream integration coverage is enough.

    #[test]
    fn parses_etcd_tls_block() {
        let f = write_yaml(
            r#"
etcd:
  endpoints: ["https://etcd.aisix.cloud:2379"]
  tls:
    ca_cert_file: "/a.crt"
    client_cert_file: "/c.crt"
    client_key_file: "/c.key"
    domain_name: "etcd.aisix.cloud"
proxy:
  addr: "0.0.0.0:3000"
admin:
  addr: "127.0.0.1:3001"
  admin_keys: ["k1"]
"#,
        );
        let cfg = Config::load_from_path(Some(f.path())).unwrap();
        let tls = cfg.etcd.tls.expect("tls parsed");
        assert_eq!(tls.ca_cert_file, "/a.crt");
        assert_eq!(tls.client_cert_file, "/c.crt");
        assert_eq!(tls.client_key_file, "/c.key");
        assert_eq!(tls.domain_name.as_deref(), Some("etcd.aisix.cloud"));
    }
}
