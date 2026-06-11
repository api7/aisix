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
    /// the admin API and Playground endpoints are **not** bound — the DP
    /// is a pure etcd reader driven by the aisix.cloud control plane.
    /// Missing or `enabled = false` runs standalone.
    #[serde(default)]
    pub managed: ManagedConfig,
    /// Deployment-wide override for the AWS Bedrock endpoint URL,
    /// applied to every kind=bedrock guardrail dispatcher built from
    /// the snapshot. Unset (the default) → SDK default (real AWS).
    ///
    /// Set this when pointing the DP at a local Bedrock-compatible
    /// service (LocalStack, a fakecloud / WireMock sidecar in e2e),
    /// or when an outbound HTTP proxy needs to terminate the call.
    /// Empty string is treated as unset so a `docker run -e
    /// AISIX_BEDROCK_ENDPOINT_URL=` doesn't accidentally redirect.
    ///
    /// Top-level on purpose — overriding the Bedrock endpoint is a
    /// deployment concern, not a per-guardrail-row configuration that
    /// a tenant should be able to set. The matching env var
    /// `AISIX_BEDROCK_ENDPOINT_URL` is what gets picked up by
    /// config-rs via the `AISIX_` prefix.
    #[serde(default)]
    pub bedrock_endpoint_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EtcdConfig {
    pub endpoints: Vec<String>,
    /// Base namespace shared by every aisix DP. v2 used the bare
    /// `prefix` as the etcd key root (`/aisix/{kind}/{id}`); v3
    /// inserts an env scope so each DP only sees its own env's
    /// resources (`/aisix/<env_id>/{kind}/{id}`, prd-09a §9A.6).
    /// The DP populates `env_id` from the v3 register response at
    /// boot; in self-managed mode the operator sets it directly.
    #[serde(default = "EtcdConfig::default_prefix")]
    pub prefix: String,
    /// Tenant scope inserted between `prefix` and the resource kind
    /// segment. Empty string = legacy/unscoped behavior (v2). The
    /// register flow overwrites this from the CP's response.
    #[serde(default)]
    pub env_id: String,
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
/// - The Playground endpoint is **not** exposed.
///
/// All configuration is read from etcd via the TLS channel (see
/// [`EtcdTlsConfig`]).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ManagedConfig {
    pub enabled: bool,

    /// aisix.cloud CP base URL, e.g. "https://api.us.aisix.cloud".
    /// Required for heartbeat when managed mode is enabled.
    #[serde(default)]
    pub cp_base_url: Option<String>,

    /// aisix.cloud CP etcd endpoint, e.g. "etcd.us.aisix.cloud:7943".
    /// In v2 the CP returned this in the register response; v3
    /// (prd-09a §9A.7.2) no longer ships it back, so the DP must
    /// know its etcd endpoint at boot. Bare `host:port` without
    /// scheme — the DP attaches `https://` for the gRPC dial.
    #[serde(default)]
    pub cp_etcd_endpoint: Option<String>,

    /// Optional path to a PEM-encoded CA bundle the DP adds as an
    /// additional trust root for outbound calls to the CP and the etcd
    /// v3 gRPC connection.
    ///
    /// In production the CP terminates TLS with a public-CA-issued
    /// certificate that the system trust store already covers, so
    /// this is `None`. In e2e / dev / on-prem deployments the CP
    /// often serves a self-signed or private-CA-signed cert; pointing
    /// this at the issuing CA's PEM bundle lets the DP trust it
    /// without disabling verification entirely.
    ///
    /// The file is read at boot — rotation requires a DP restart.
    /// When set but unreadable the boot fails fast with the path so
    /// the operator can fix the mount; we never silently fall through
    /// to `InsecureSkipVerify`.
    #[serde(default)]
    pub cp_ca_cert_file: Option<String>,

    /// Inline PEM-encoded leaf certificate for the api7ee-parity
    /// cert-via-env-var bootstrap path (cp-api's
    /// /api/environments/:id/gateway_certificates endpoint, dashboard
    /// CertIssueCard). When all three of `cp_cert_pem` / `cp_key_pem`
    /// / `cp_ca_pem` are set, the DP materialises the operator-minted
    /// dashboard bundle at boot. env_id is parsed from the cert's URI SAN
    /// (`x-aisix://env/<env_id>`).
    ///
    /// File-based variants below let operators store PEMs on disk
    /// (e.g. systemd unit on a host VM) instead of inlining into env
    /// vars. Inline-PEM and file-path variants are mutually exclusive
    /// per pair (cert/key/ca); mixing them is a config error caught
    /// at boot.
    #[serde(default)]
    pub cp_cert_pem: Option<String>,

    /// Inline PEM-encoded private key paired with `cp_cert_pem`.
    /// Mutually exclusive with `cp_key_file`.
    #[serde(default)]
    pub cp_key_pem: Option<String>,

    /// Inline PEM-encoded CA certificate paired with `cp_cert_pem`.
    /// The DP installs this as the trust anchor for outbound mTLS
    /// to dp-manager. Mutually exclusive with `cp_ca_file`.
    #[serde(default)]
    pub cp_ca_pem: Option<String>,

    /// File-path variant of `cp_cert_pem`.
    #[serde(default)]
    pub cp_cert_file: Option<String>,

    /// File-path variant of `cp_key_pem`.
    #[serde(default)]
    pub cp_key_file: Option<String>,

    /// File-path variant of `cp_ca_pem`.
    #[serde(default)]
    pub cp_ca_file: Option<String>,

    /// Directory where the DP persists `ca.crt`, `client.crt`,
    /// `client.key`. Files are written `0600`. Parent directory must
    /// already exist and be writable by the aisix process user.
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

    /// True when the operator pre-provisioned a cert/key/CA bundle
    /// via the api7ee-parity dashboard flow — either inlined as
    /// PEM env vars (`cp_cert_pem` / `cp_key_pem` / `cp_ca_pem`) or
    /// referenced by file path (`cp_cert_file` / `cp_key_file` /
    /// `cp_ca_file`). All three slots in the same triplet must be
    /// present together; mixing inline-and-file forms within a
    /// single role is rejected at boot for clarity.
    pub fn cert_bundle_provided(&self) -> bool {
        let has_pem = self.cp_cert_pem.as_deref().is_some_and(|s| !s.is_empty())
            && self.cp_key_pem.as_deref().is_some_and(|s| !s.is_empty())
            && self.cp_ca_pem.as_deref().is_some_and(|s| !s.is_empty());
        let has_file = self.cp_cert_file.as_deref().is_some_and(|s| !s.is_empty())
            && self.cp_key_file.as_deref().is_some_and(|s| !s.is_empty())
            && self.cp_ca_file.as_deref().is_some_and(|s| !s.is_empty());
        has_pem || has_file
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

    /// The full env-scoped key prefix the DP watches and parses.
    /// v3: `<prefix>/<env_id>/` (e.g. `/aisix/<uuid>/`); v2 fallback
    /// (env_id empty): bare `<prefix>` for backwards compat with
    /// self-managed deployments that haven't migrated yet.
    ///
    /// The trailing slash matters for the kine etcd-auth interceptor
    /// (internal/dpmgr/etcdauth on the dp-manager side): it requires
    /// the DP's Range key to start with `<prefix>/<env_id>/`, NOT
    /// `<prefix>/<env_id>`. Without the slash a bare `<prefix>/<env_id>`
    /// Range request gets `PermissionDenied: outside env <env_id> prefix`
    /// because the auth check sees the bare-prefix Range as escaping
    /// into a sibling env's space (the env-id substring could be any
    /// prefix-of-prefix until the slash terminates it).
    pub fn effective_prefix(&self) -> String {
        if self.env_id.is_empty() {
            self.prefix.clone()
        } else {
            let trimmed = self.prefix.trim_end_matches('/');
            format!("{trimmed}/{}/", self.env_id)
        }
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
    /// Real-client-IP resolution from forwarded headers (#492). Default
    /// trusts nothing, so the logged source IP is always the immediate
    /// TCP peer. Configure `trusted_proxies` when the gateway sits behind
    /// an L7 LB / ingress that sets `x-forwarded-for`.
    #[serde(default)]
    pub real_ip: RealIpConfig,
}

impl ProxyConfig {
    const fn default_body_limit() -> usize {
        10 * 1024 * 1024
    }
}

/// nginx `set_real_ip_from` + `real_ip_recursive` equivalent. Resolves
/// the downstream client IP for usage logs (#492) from a forwarded
/// header, trusting only addresses inside `trusted_proxies`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct RealIpConfig {
    /// Trusted upstream proxy CIDRs (e.g. `["10.0.0.0/8", "127.0.0.1/32"]`).
    /// When the immediate TCP peer matches one of these, the configured
    /// forwarded header is trusted and walked to find the real client.
    /// Empty (the default) = trust nothing → always log the TCP peer.
    pub trusted_proxies: Vec<String>,
    /// nginx `real_ip_recursive`. When true, walk the forwarded header
    /// right-to-left skipping every trusted address; the first untrusted
    /// one is the client. When false, take the rightmost header entry
    /// once the peer is trusted.
    pub recursive: bool,
    /// Forwarded header to consult. Defaults to `x-forwarded-for`.
    pub header: String,
}

impl Default for RealIpConfig {
    fn default() -> Self {
        Self {
            trusted_proxies: Vec::new(),
            recursive: false,
            header: Self::default_header(),
        }
    }
}

impl RealIpConfig {
    fn default_header() -> String {
        "x-forwarded-for".into()
    }

    /// Parse `trusted_proxies` strings into CIDRs, rejecting malformed
    /// entries. A bare IP (no `/prefix`) is accepted as a host route.
    pub fn parse_trusted(&self) -> Result<Vec<ipnet::IpNet>, String> {
        self.trusted_proxies
            .iter()
            .map(|s| {
                s.parse::<ipnet::IpNet>()
                    .or_else(|_| s.parse::<std::net::IpAddr>().map(ipnet::IpNet::from))
                    .map_err(|_| s.clone())
            })
            .collect()
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
    /// Optional bind address for a **dedicated** metrics listener
    /// (e.g. `0.0.0.0:9090`). When set, the gateway serves `path` on
    /// its own listener — separate from the admin listener that
    /// normally hosts `/metrics`.
    ///
    /// This is what lets a managed-mode DP be scraped at all: managed
    /// mode never binds the admin listener (`ManagedConfig::is_managed`),
    /// so without a dedicated address `/metrics` is configured but
    /// stranded on a listener that never comes up. The baked-in
    /// `config.managed.yaml` sets this so managed DPs expose metrics by
    /// default with no control-plane change.
    ///
    /// Unset (the default) keeps `/metrics` on the admin listener only,
    /// preserving the prior behavior for self-hosted deployments.
    #[serde(default)]
    pub addr: Option<String>,
}

impl Default for PrometheusConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            path: "/metrics".into(),
            addr: None,
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

/// Boot-level cache backend availability (#519 B.8).
///
/// The in-process memory cache is always built; the redis cache is
/// built iff `redis` is set. Which instance serves a given request is
/// selected by the matched `CachePolicy.backend` (etcd-managed, per
/// policy) — NOT by this struct.
///
/// `backend` is a legacy knob kept parsing for config compatibility:
/// it no longer selects "the one global cache". Its only remaining
/// effect is fail-fast validation — `backend = "redis"` without a
/// `redis` block is rejected at boot.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct CacheConfig {
    pub backend: CacheBackend,
    pub redis: Option<RedisCacheConfig>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            backend: CacheBackend::Memory,
            redis: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CacheBackend {
    Memory,
    Redis,
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
                // EVERY string env override through comma-splitting,
                // which blows up secrets that happen to contain a
                // comma with a serde "invalid type: sequence, expected
                // a string" error. Opt in only for fields that are
                // actually Vec<String>.
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
        if let Err(bad) = self.proxy.real_ip.parse_trusted() {
            return Err(BootstrapError::Config(format!(
                "proxy.real_ip.trusted_proxies invalid CIDR/IP: {bad}"
            )));
        }
        // Dedicated metrics listener address, when configured, must be a
        // bindable socket address. Unset keeps `/metrics` on the admin
        // listener (no dedicated listener), so only validate when present.
        if let Some(addr) = self.observability.metrics.prometheus.addr.as_deref() {
            if addr.parse::<std::net::SocketAddr>().is_err() {
                return Err(BootstrapError::Config(format!(
                    "observability.metrics.prometheus.addr invalid socket address: {addr}"
                )));
            }
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
        // No dedicated metrics listener unless an addr is set — `/metrics`
        // stays on the admin listener for self-hosted deployments.
        assert!(cfg.observability.metrics.prometheus.addr.is_none());
        assert_eq!(cfg.cache.backend, CacheBackend::Memory);
        // real_ip defaults: trust nothing, non-recursive, x-forwarded-for.
        assert!(cfg.proxy.real_ip.trusted_proxies.is_empty());
        assert!(!cfg.proxy.real_ip.recursive);
        assert_eq!(cfg.proxy.real_ip.header, "x-forwarded-for");
        assert!(cfg.proxy.real_ip.parse_trusted().unwrap().is_empty());
    }

    #[test]
    fn loads_real_ip_block() {
        let f = write_yaml(
            r#"
etcd:
  endpoints: ["http://127.0.0.1:2379"]
  prefix: "/aisix"
proxy:
  addr: "0.0.0.0:3000"
  real_ip:
    trusted_proxies: ["10.0.0.0/8", "127.0.0.1"]
    recursive: true
admin:
  addr: "127.0.0.1:3001"
  admin_keys: ["k1"]
"#,
        );
        let cfg = Config::load_from_path(Some(f.path())).unwrap();
        assert!(cfg.proxy.real_ip.recursive);
        // bare IP normalises to a /32 host route.
        let nets = cfg.proxy.real_ip.parse_trusted().unwrap();
        assert_eq!(nets.len(), 2);
        assert!(nets.iter().any(|n| n.to_string() == "10.0.0.0/8"));
    }

    #[test]
    fn rejects_malformed_trusted_proxy_cidr() {
        let f = write_yaml(
            r#"
etcd:
  endpoints: ["http://127.0.0.1:2379"]
  prefix: "/aisix"
proxy:
  addr: "0.0.0.0:3000"
  real_ip:
    trusted_proxies: ["not-a-cidr"]
admin:
  addr: "127.0.0.1:3001"
  admin_keys: ["k1"]
"#,
        );
        let err = Config::load_from_path(Some(f.path())).unwrap_err();
        assert!(
            format!("{err}").contains("trusted_proxies"),
            "error should name the bad field: {err}"
        );
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
    fn parses_prometheus_addr_for_dedicated_listener() {
        // A dedicated metrics listener address parses and round-trips.
        // This is the managed-mode shape: `/metrics` is served on its
        // own bound listener because the admin listener is never bound.
        let f = write_yaml(
            r#"
etcd:
  endpoints: ["http://127.0.0.1:2379"]
proxy:
  addr: "0.0.0.0:3000"
admin:
  addr: "127.0.0.1:3001"
  admin_keys: ["k1"]
observability:
  metrics:
    prometheus:
      enabled: true
      path: "/metrics"
      addr: "0.0.0.0:9090"
"#,
        );
        let cfg = Config::load_from_path(Some(f.path())).unwrap();
        assert_eq!(
            cfg.observability.metrics.prometheus.addr.as_deref(),
            Some("0.0.0.0:9090"),
        );
    }

    #[test]
    fn rejects_invalid_prometheus_addr() {
        // A malformed dedicated-listener address must fail validation at
        // boot, not at bind time — operators get a clear config error.
        let f = write_yaml(
            r#"
etcd:
  endpoints: ["http://127.0.0.1:2379"]
proxy:
  addr: "0.0.0.0:3000"
admin:
  addr: "127.0.0.1:3001"
  admin_keys: ["k1"]
observability:
  metrics:
    prometheus:
      addr: "not-a-socket-addr"
"#,
        );
        let err = Config::load_from_path(Some(f.path())).unwrap_err();
        assert!(
            err.to_string().contains("prometheus.addr"),
            "error should name the bad field: {err}"
        );
    }

    #[test]
    fn shipped_managed_config_binds_a_dedicated_metrics_listener() {
        // The baked managed-image config (`config.managed.yaml`) is the
        // entire "no control-plane change" lever: a managed DP exposes
        // `/metrics` only because this file binds a dedicated listener
        // while the admin listener stays the unbindable sentinel. A typo
        // here would silently un-scrape every managed DP — that file is
        // only COPYd into the image, so nothing else catches it. Pin it.
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config.managed.yaml");
        let cfg =
            Config::load_from_path(Some(Path::new(path))).expect("config.managed.yaml must load");
        assert!(cfg.managed.is_managed());
        assert_eq!(
            cfg.observability.metrics.prometheus.addr.as_deref(),
            Some("0.0.0.0:9090"),
            "managed DPs must bind a dedicated metrics listener so they can be scraped",
        );
        // Admin is the unbindable sentinel, so the dedicated listener is
        // the only metrics surface in managed mode.
        assert_eq!(cfg.admin.addr, "127.0.0.1:0");
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
    fn parses_managed_block_without_register_fields() {
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
        // CP URL comes from env at runtime — empty here is fine.
        assert!(cfg.managed.cp_base_url.is_none());
    }

    #[test]
    fn rejects_legacy_registration_token_field() {
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
  registration_token: "unused"
"#,
        );
        let err = Config::load_from_path(Some(f.path())).unwrap_err();
        assert!(
            err.to_string().contains("registration_token"),
            "expected unknown legacy field error, got {err}",
        );
    }

    #[test]
    fn bedrock_endpoint_url_defaults_to_none_when_unset() {
        // Minimal config without bedrock_endpoint_url → field should
        // be `None`, matching "real AWS Bedrock" semantics.
        let f = write_yaml(
            r#"
etcd:
  endpoints: ["http://127.0.0.1:2379"]
proxy:
  addr: "0.0.0.0:3000"
admin:
  addr: "127.0.0.1:3001"
  admin_keys: ["k1"]
"#,
        );
        let cfg = Config::load_from_path(Some(f.path())).unwrap();
        assert!(cfg.bedrock_endpoint_url.is_none());
    }

    #[test]
    fn bedrock_endpoint_url_round_trips_through_yaml() {
        // Operators set this when pointing the DP at LocalStack /
        // fakecloud / a Bedrock-compatible mock; pin that the field
        // makes it through `deny_unknown_fields` and back out.
        let f = write_yaml(
            r#"
etcd:
  endpoints: ["http://127.0.0.1:2379"]
proxy:
  addr: "0.0.0.0:3000"
admin:
  addr: "127.0.0.1:3001"
  admin_keys: ["k1"]
bedrock_endpoint_url: "http://fakecloud:8000"
"#,
        );
        let cfg = Config::load_from_path(Some(f.path())).unwrap();
        assert_eq!(
            cfg.bedrock_endpoint_url.as_deref(),
            Some("http://fakecloud:8000"),
        );
    }

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
