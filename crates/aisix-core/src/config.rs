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
    /// Dynamic-resource source A: etcd. Required unless `resources_file`
    /// selects the file source below; the two are mutually exclusive.
    #[serde(default)]
    pub etcd: EtcdConfig,
    /// Dynamic-resource source B: a standalone resources file
    /// (`resources.yaml`). When set, the gateway loads every resource
    /// (provider keys, models, API keys, …) from this file at boot and
    /// re-reads it on SIGHUP; the `etcd` section must be absent or left
    /// unconfigured, and the admin listener serves the resource surface
    /// read-only. Mutually exclusive with configured `etcd.endpoints`
    /// and with `managed.enabled`.
    #[serde(default)]
    pub resources_file: Option<String>,
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
    /// Rate-limit counter backend. Defaults to per-process memory
    /// (historical behaviour). Set `backend: redis` with a `redis` block
    /// to share counters across every DP replica so a cluster enforces
    /// one global window instead of one-per-replica (api7/AISIX-Cloud#798).
    #[serde(default)]
    pub ratelimit: RateLimitConfig,
    /// Connection-layer tuning for outbound calls to LLM providers.
    /// Defaults bound the connect phase, keep TCP keepalive on, and expire
    /// pooled connections well before a typical LB/NAT/proxy hop would —
    /// see [`UpstreamConfig`].
    #[serde(default)]
    pub upstream: UpstreamConfig,
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

    /// Heartbeat interval, in seconds. The DP POSTs a heartbeat to
    /// dp-manager every `heartbeat_interval_secs`; CP surfaces a DP as
    /// "connected" on its first heartbeat. Clamped to [5, 300] by
    /// [`crate`]-external `HeartbeatConfig::sanitised`. Default 15s in
    /// production; e2e/dev can lower it (min 5s) so connect-detection
    /// tests aren't bound by the interval.
    #[serde(default = "ManagedConfig::default_heartbeat_interval_secs")]
    pub heartbeat_interval_secs: u64,
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
    const fn default_heartbeat_interval_secs() -> u64 {
        15
    }
}

/// Default is the "unconfigured" shape (no endpoints) so a
/// `resources_file` deployment can omit the `etcd` section entirely.
/// [`Config::validate`] still rejects empty endpoints whenever the file
/// source is not selected, so etcd-mode behavior is unchanged.
impl Default for EtcdConfig {
    fn default() -> Self {
        Self {
            endpoints: Vec::new(),
            prefix: Self::default_prefix(),
            env_id: String::new(),
            user: None,
            password_env: None,
            dial_timeout_ms: Self::default_dial_timeout(),
            request_timeout_ms: Self::default_request_timeout(),
            tls: None,
        }
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
    /// When `false`, the admin listener is not bound even in standalone
    /// (etcd or file) mode. The proxy and the metrics/status listener are
    /// unaffected, so resources are managed declaratively — a resources
    /// file, or direct writes to the configuration store — with
    /// `GET /status/config` and the proxy `GET /livez` as the operational
    /// feedback. Managed mode never binds the admin listener regardless.
    /// Defaults to `true`.
    #[serde(default = "AdminConfig::default_enabled")]
    pub enabled: bool,
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

    fn default_enabled() -> bool {
        true
    }
}

impl Default for AdminConfig {
    fn default() -> Self {
        Self {
            enabled: Self::default_enabled(),
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
    /// Operator-defined User-Agent → `client_type` mapping rules
    /// (AISIX-Cloud#1045), consulted BEFORE the built-in allowlist so a
    /// deployment can classify in-house tools (or re-bucket a built-in
    /// match). Deployment-scoped on purpose: the labels these rules mint
    /// go to this DP's own Prometheus scrape surface, so the operator who
    /// owns the scrape owns the label set. Order matters (first match
    /// wins); compiled + validated at boot (fail-fast), never hot-reloaded.
    pub client_type_rules: Vec<ClientTypeRule>,
}

/// One `client_type_rules` entry: a regex tried against the raw inbound
/// `User-Agent` (case-insensitive, unanchored — anchor with `^` yourself),
/// and the bounded label value emitted on match. The label — not the UA —
/// becomes the Prometheus `client_type` value, so cardinality stays capped
/// by the rule count.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClientTypeRule {
    pub pattern: String,
    pub client: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct PrometheusConfig {
    pub enabled: bool,
    pub path: String,
    /// Bind address of the **dedicated** metrics listener (default
    /// `0.0.0.0:9090`). The scrape endpoint always lives on its own
    /// listener — identical in standalone and managed mode — so the
    /// scrape surface never depends on which other listeners a
    /// deployment binds. The admin listener does not serve `/metrics`.
    pub addr: String,
}

impl PrometheusConfig {
    pub const DEFAULT_ADDR: &'static str = "0.0.0.0:9090";
}

impl Default for PrometheusConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            path: "/metrics".into(),
            addr: Self::DEFAULT_ADDR.into(),
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
    pub redis: Option<RedisConnConfig>,
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

/// Connection topology for a shared Redis backend (cache + rate-limit).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum RedisMode {
    /// One Redis endpoint (`url`). The historical default.
    #[default]
    Single,
    /// Redis Cluster — seeded from `nodes`, topology discovered at connect.
    Cluster,
    /// Redis Sentinel — the master is discovered (and re-discovered after
    /// failover) via `sentinels` for the group named `master_name`.
    Sentinel,
}

/// Shared connection shape for the Redis-backed response cache and the
/// shared rate-limit counter store. `mode` selects the topology; the
/// fields each mode needs are validated at boot ([`Self::validate`]):
///
/// - `single`   → `url` (e.g. `redis://host:6379`)
/// - `cluster`  → `nodes` (one or more seed node URLs)
/// - `sentinel` → `sentinels` (sentinel node URLs) + `master_name`
///
/// In `single` mode all credentials and TLS (`rediss://`) travel inside
/// `url`. In `cluster`/`sentinel` mode they can travel in the node /
/// sentinel URLs the same way, but the **data node** (cluster nodes, or
/// the Sentinel-discovered master) can also be authenticated explicitly
/// with `username` + `password` (Redis ACL) and, for sentinel, a
/// `database` — useful because the Sentinel-discovered master has no URL
/// of its own. Sentinel-node auth still travels in the `sentinels` URLs,
/// so Sentinel and master credentials may differ.
///
/// To keep secrets out of the config file, supply `password` via the
/// matching env var instead, e.g. `AISIX_RATELIMIT__REDIS__PASSWORD`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields, default)]
pub struct RedisConnConfig {
    pub mode: RedisMode,
    /// Single-node URL. Required when `mode = single`.
    pub url: Option<String>,
    /// Cluster seed node URLs. Required (≥1) when `mode = cluster`.
    pub nodes: Vec<String>,
    /// Sentinel node URLs. Required (≥1) when `mode = sentinel`.
    pub sentinels: Vec<String>,
    /// Monitored master group name. Required when `mode = sentinel`.
    pub master_name: Option<String>,
    /// ACL username for the data node (cluster nodes / sentinel master).
    pub username: Option<String>,
    /// Password for the data node (cluster nodes / sentinel master).
    pub password: Option<String>,
    /// Database index for the Sentinel-discovered master (default 0).
    /// Not applicable to `cluster` (Redis Cluster only has DB 0).
    pub database: Option<i64>,
}

impl RedisConnConfig {
    /// Fail-fast check that the fields the selected `mode` needs are
    /// present. `ctx` labels the offending block (e.g. `cache.redis`).
    pub fn validate(&self, ctx: &str) -> Result<(), String> {
        let non_empty = |v: &[String]| v.iter().any(|s| !s.trim().is_empty());
        match self.mode {
            RedisMode::Single => {
                if self.url.as_deref().unwrap_or("").trim().is_empty() {
                    return Err(format!("{ctx}.url is required when mode = single"));
                }
            }
            RedisMode::Cluster => {
                if !non_empty(&self.nodes) {
                    return Err(format!(
                        "{ctx}.nodes must list at least one node when mode = cluster"
                    ));
                }
            }
            RedisMode::Sentinel => {
                if !non_empty(&self.sentinels) {
                    return Err(format!(
                        "{ctx}.sentinels must list at least one sentinel when mode = sentinel"
                    ));
                }
                if self.master_name.as_deref().unwrap_or("").trim().is_empty() {
                    return Err(format!(
                        "{ctx}.master_name is required when mode = sentinel"
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Rate-limit counter backend (api7/AISIX-Cloud#798).
///
/// `Memory` is the default: per-process fixed-window counters, so an
/// N-replica cluster enforces N× the configured limit. `Redis` shares
/// the counters across replicas via a single Redis so the whole cluster
/// enforces one global window. The `redis` block is required iff
/// `backend = redis` (validated at boot). Reuses [`RedisConnConfig`]
/// for the connection shape, so it supports `single`/`cluster`/`sentinel`
/// modes too; may point at the same Redis as `cache` (keys are namespaced
/// `aisix:rl:`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct RateLimitConfig {
    pub backend: RateLimitBackend,
    pub redis: Option<RedisConnConfig>,
    /// Seconds after which an unreleased concurrency slot is reclaimed
    /// (crashed replica / hung upstream). Generous enough for a long
    /// streaming response. Redis backend only.
    pub concurrency_ttl_secs: u64,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            backend: RateLimitBackend::Memory,
            redis: None,
            concurrency_ttl_secs: 300,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RateLimitBackend {
    Memory,
    Redis,
}

/// Connection-layer settings for the HTTP clients that call LLM providers.
///
/// These are deployment properties of the network path to the upstream, not
/// per-tenant configuration, so they live in the DP config file rather than
/// on a Model or ProviderKey resource.
///
/// The defaults exist because reqwest's own are wrong for a gateway sitting
/// behind an LB/NAT/proxy hop: no connect timeout, TCP keepalive off, and a
/// 90s pooled-connection lifetime that outlives the idle timeout of a
/// typical hop — so a connection reaped upstream can still be handed out
/// here, and the request fails with an opaque transport error.
///
/// Every duration accepts `0` to disable that individual knob.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct UpstreamConfig {
    /// Max time for DNS + TCP + TLS before an attempt fails. Without it a
    /// black-holed upstream is bounded only by the model's overall timeout.
    pub connect_timeout_ms: u64,
    /// Idle seconds before the kernel sends its first TCP keepalive probe.
    /// Keeps a long wait for a slow first token from being reaped by a NAT
    /// or LB idle timer.
    pub tcp_keepalive_secs: u64,
    /// Seconds between subsequent keepalive probes.
    pub tcp_keepalive_interval_secs: u64,
    /// Unacknowledged probes before the kernel drops the connection.
    pub tcp_keepalive_retries: u32,
    /// How long an idle connection may sit in the pool before it is
    /// discarded. **Keep this below the shortest idle timeout on the path
    /// to the provider** (LB, NAT gateway, corporate proxy, service mesh),
    /// or the pool will hand out connections the far end already closed.
    pub pool_idle_timeout_secs: u64,
    /// Cap on idle connections kept per upstream host. `null` (the
    /// default) leaves reqwest's unbounded behaviour.
    pub pool_max_idle_per_host: Option<usize>,
}

impl Default for UpstreamConfig {
    fn default() -> Self {
        Self {
            connect_timeout_ms: 5_000,
            tcp_keepalive_secs: 60,
            tcp_keepalive_interval_secs: 30,
            tcp_keepalive_retries: 5,
            pool_idle_timeout_secs: 30,
            pool_max_idle_per_host: None,
        }
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
        if let Some(path) = self.resources_file.as_deref() {
            // File source selected: exactly one resource source may be
            // active. A configured etcd endpoint list alongside the file
            // is ambiguous — fail loudly instead of silently ignoring one.
            if path.trim().is_empty() {
                return Err(BootstrapError::Config(
                    "resources_file must not be empty when set".into(),
                ));
            }
            if !self.etcd.endpoints.is_empty() {
                return Err(BootstrapError::Config(
                    "config sets both etcd.endpoints and resources_file — the etcd \
                     source and the file source are mutually exclusive; remove one"
                        .into(),
                ));
            }
            if self.managed.is_managed() {
                return Err(BootstrapError::Config(
                    "resources_file cannot be combined with managed.enabled = true \
                     (managed mode reads resources from the control plane)"
                        .into(),
                ));
            }
        } else if self.etcd.endpoints.is_empty() {
            return Err(BootstrapError::Config(
                "etcd.endpoints must contain at least one endpoint \
                 (or set resources_file to load resources from a file)"
                    .into(),
            ));
        }
        // The admin listener is not bound in managed mode, nor when
        // `admin.enabled = false`, so requiring admin_keys or a valid
        // admin.addr in those cases would be punishing the user for
        // fields that aren't going to be used. When it will bind, keep
        // the original invariants.
        if !self.managed.is_managed() && self.admin.enabled {
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
        // The dedicated metrics listener address must be a bindable
        // socket address — it is always bound when prometheus is enabled.
        let metrics_addr = &self.observability.metrics.prometheus.addr;
        if metrics_addr.parse::<std::net::SocketAddr>().is_err() {
            return Err(BootstrapError::Config(format!(
                "observability.metrics.prometheus.addr invalid socket address: {metrics_addr}"
            )));
        }
        if self.ratelimit.backend == RateLimitBackend::Redis {
            match &self.ratelimit.redis {
                None => {
                    return Err(BootstrapError::Config(
                        "ratelimit.backend = redis requires a ratelimit.redis block".into(),
                    ));
                }
                Some(redis) => redis
                    .validate("ratelimit.redis")
                    .map_err(BootstrapError::Config)?,
            }
            // A zero concurrency TTL would prune a slot in the same second
            // it was taken, silently disabling concurrency limiting.
            if self.ratelimit.concurrency_ttl_secs == 0 {
                return Err(BootstrapError::Config(
                    "ratelimit.concurrency_ttl_secs must be > 0 for the redis backend".into(),
                ));
            }
        }
        // A `cache.redis` block, when present, is built regardless of the
        // legacy `cache.backend` knob, so validate its mode fields too.
        if let Some(redis) = &self.cache.redis {
            redis
                .validate("cache.redis")
                .map_err(BootstrapError::Config)?;
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
        // The dedicated metrics listener defaults to 0.0.0.0:9090 in
        // every mode — no admin-listener fallback to fall out of sync with.
        assert_eq!(cfg.observability.metrics.prometheus.addr, "0.0.0.0:9090");
        assert_eq!(cfg.cache.backend, CacheBackend::Memory);
        // real_ip defaults: trust nothing, non-recursive, x-forwarded-for.
        assert!(cfg.proxy.real_ip.trusted_proxies.is_empty());
        assert!(!cfg.proxy.real_ip.recursive);
        assert_eq!(cfg.proxy.real_ip.header, "x-forwarded-for");
        assert!(cfg.proxy.real_ip.parse_trusted().unwrap().is_empty());
    }

    #[test]
    fn managed_heartbeat_interval_defaults_to_15_and_can_be_lowered() {
        let base = r#"
etcd:
  endpoints: ["http://127.0.0.1:2379"]
  prefix: "/aisix"
proxy:
  addr: "0.0.0.0:3000"
admin:
  addr: "127.0.0.1:3001"
  admin_keys: ["k1"]
managed:
  enabled: true
  cp_base_url: "https://cp.example"
"#;
        // Omitted → production default 15s.
        let f = write_yaml(base);
        let cfg = Config::load_from_path(Some(f.path())).unwrap();
        assert_eq!(cfg.managed.heartbeat_interval_secs, 15);

        // Explicit → e2e/dev can lower it (the 5s floor is enforced later
        // by HeartbeatConfig::sanitised, not here).
        let f = write_yaml(&format!("{base}  heartbeat_interval_secs: 5\n"));
        let cfg = Config::load_from_path(Some(f.path())).unwrap();
        assert_eq!(cfg.managed.heartbeat_interval_secs, 5);
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
    fn resources_file_makes_etcd_section_optional() {
        let f = write_yaml(
            r#"
resources_file: "/etc/aisix/resources.yaml"
proxy:
  addr: "0.0.0.0:3000"
admin:
  addr: "127.0.0.1:3001"
  admin_keys: ["k1"]
"#,
        );
        let cfg = Config::load_from_path(Some(f.path())).unwrap();
        assert_eq!(
            cfg.resources_file.as_deref(),
            Some("/etc/aisix/resources.yaml")
        );
        assert!(cfg.etcd.endpoints.is_empty());
        // Untouched etcd defaults still materialize for downstream code.
        assert_eq!(cfg.etcd.prefix, "/aisix");
    }

    #[test]
    fn resources_file_conflicts_with_configured_etcd_endpoints() {
        let f = write_yaml(
            r#"
etcd:
  endpoints: ["http://127.0.0.1:2379"]
resources_file: "/etc/aisix/resources.yaml"
proxy:
  addr: "0.0.0.0:3000"
admin:
  addr: "127.0.0.1:3001"
  admin_keys: ["k1"]
"#,
        );
        let err = Config::load_from_path(Some(f.path())).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("mutually exclusive"), "unexpected: {msg}");
        assert!(msg.contains("resources_file"), "unexpected: {msg}");
    }

    #[test]
    fn resources_file_conflicts_with_managed_mode() {
        let f = write_yaml(
            r#"
resources_file: "/etc/aisix/resources.yaml"
proxy:
  addr: "0.0.0.0:3000"
admin:
  addr: "127.0.0.1:3001"
  admin_keys: ["k1"]
managed:
  enabled: true
"#,
        );
        let err = Config::load_from_path(Some(f.path())).unwrap_err();
        assert!(err.to_string().contains("managed"), "unexpected: {err}");
    }

    #[test]
    fn resources_file_rejects_empty_path() {
        let f = write_yaml(
            r#"
resources_file: ""
proxy:
  addr: "0.0.0.0:3000"
admin:
  addr: "127.0.0.1:3001"
  admin_keys: ["k1"]
"#,
        );
        let err = Config::load_from_path(Some(f.path())).unwrap_err();
        assert!(
            err.to_string().contains("resources_file"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn resources_file_mode_still_requires_admin_keys() {
        // The admin listener stays bound (read-only resource surface) in
        // file mode, so the standalone auth invariant holds.
        let f = write_yaml(
            r#"
resources_file: "/etc/aisix/resources.yaml"
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
    fn admin_enabled_defaults_to_true() {
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
        assert!(cfg.admin.enabled);
    }

    #[test]
    fn admin_disabled_relaxes_admin_key_requirement() {
        // With the admin listener switched off, there is no bound surface
        // to authenticate, so an empty admin_keys is no longer an error —
        // resources are managed declaratively (etcd here).
        let f = write_yaml(
            r#"
etcd:
  endpoints: ["http://127.0.0.1:2379"]
  prefix: "/aisix"
proxy:
  addr: "0.0.0.0:3000"
admin:
  enabled: false
"#,
        );
        let cfg = Config::load_from_path(Some(f.path())).unwrap();
        assert!(!cfg.admin.enabled);
        assert!(cfg.admin.admin_keys.is_empty());
    }

    #[test]
    fn admin_disabled_relaxes_admin_key_requirement_in_file_mode() {
        // File mode routes through a distinct admin store variant, and it
        // too binds a read-only admin surface by default. With the admin
        // listener switched off, the same relaxation applies — no
        // admin_keys required.
        let f = write_yaml(
            r#"
resources_file: "/etc/aisix/resources.yaml"
proxy:
  addr: "0.0.0.0:3000"
admin:
  enabled: false
"#,
        );
        let cfg = Config::load_from_path(Some(f.path())).unwrap();
        assert!(!cfg.admin.enabled);
        assert!(cfg.admin.admin_keys.is_empty());
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
    fn ratelimit_defaults_to_memory_backend() {
        let f = write_yaml(
            r#"
etcd:
  endpoints: ["http://localhost:2379"]
proxy:
  addr: "0.0.0.0:3000"
admin:
  addr: "127.0.0.1:3001"
  admin_keys: ["k1"]
"#,
        );
        let cfg = Config::load_from_path(Some(f.path())).unwrap();
        assert_eq!(cfg.ratelimit.backend, RateLimitBackend::Memory);
        assert!(cfg.ratelimit.redis.is_none());
        assert_eq!(cfg.ratelimit.concurrency_ttl_secs, 300);
    }

    /// An `upstream:` block is optional; the defaults must still bound the
    /// connect phase, keep TCP keepalive on, and expire pooled connections
    /// sooner than reqwest's own 90s (AISIX-Cloud#1122).
    #[test]
    fn upstream_defaults_apply_when_the_block_is_absent() {
        let f = write_yaml(
            r#"
etcd:
  endpoints: ["http://localhost:2379"]
proxy:
  addr: "0.0.0.0:3000"
admin:
  addr: "127.0.0.1:3001"
  admin_keys: ["k1"]
"#,
        );
        let cfg = Config::load_from_path(Some(f.path())).unwrap();
        assert_eq!(cfg.upstream.connect_timeout_ms, 5_000);
        assert_eq!(cfg.upstream.tcp_keepalive_secs, 60);
        assert_eq!(cfg.upstream.tcp_keepalive_interval_secs, 30);
        assert_eq!(cfg.upstream.tcp_keepalive_retries, 5);
        assert!(cfg.upstream.pool_idle_timeout_secs < 90);
        assert!(cfg.upstream.pool_max_idle_per_host.is_none());
    }

    /// Operators behind a proxy with a short idle timeout need to lower
    /// `pool_idle_timeout_secs`; every knob must be individually settable
    /// and `0` must round-trip (it means "leave this one off").
    #[test]
    fn upstream_block_overrides_individual_knobs() {
        let f = write_yaml(
            r#"
etcd:
  endpoints: ["http://localhost:2379"]
proxy:
  addr: "0.0.0.0:3000"
admin:
  addr: "127.0.0.1:3001"
  admin_keys: ["k1"]
upstream:
  connect_timeout_ms: 2000
  pool_idle_timeout_secs: 10
  tcp_keepalive_secs: 0
  pool_max_idle_per_host: 16
"#,
        );
        let cfg = Config::load_from_path(Some(f.path())).unwrap();
        assert_eq!(cfg.upstream.connect_timeout_ms, 2_000);
        assert_eq!(cfg.upstream.pool_idle_timeout_secs, 10);
        assert_eq!(cfg.upstream.tcp_keepalive_secs, 0);
        assert_eq!(cfg.upstream.pool_max_idle_per_host, Some(16));
        // Unspecified knobs keep their defaults.
        assert_eq!(cfg.upstream.tcp_keepalive_interval_secs, 30);
    }

    #[test]
    fn ratelimit_redis_backend_requires_redis_block() {
        let f = write_yaml(
            r#"
etcd:
  endpoints: ["http://localhost:2379"]
proxy:
  addr: "0.0.0.0:3000"
admin:
  addr: "127.0.0.1:3001"
  admin_keys: ["k1"]
ratelimit:
  backend: "redis"
"#,
        );
        let err = Config::load_from_path(Some(f.path())).unwrap_err();
        assert!(err.to_string().contains("ratelimit.redis"));
    }

    #[test]
    fn rejects_zero_concurrency_ttl_for_redis_backend() {
        let f = write_yaml(
            r#"
etcd:
  endpoints: ["http://localhost:2379"]
proxy:
  addr: "0.0.0.0:3000"
admin:
  addr: "127.0.0.1:3001"
  admin_keys: ["k1"]
ratelimit:
  backend: "redis"
  redis:
    url: "redis://127.0.0.1:6379"
  concurrency_ttl_secs: 0
"#,
        );
        let err = Config::load_from_path(Some(f.path())).unwrap_err();
        assert!(err.to_string().contains("concurrency_ttl_secs"));
    }

    fn redis_backend_yaml(redis_block: &str) -> tempfile::NamedTempFile {
        write_yaml(&format!(
            r#"
etcd:
  endpoints: ["http://localhost:2379"]
proxy:
  addr: "0.0.0.0:3000"
admin:
  addr: "127.0.0.1:3001"
  admin_keys: ["k1"]
ratelimit:
  backend: "redis"
  redis:
{redis_block}
"#
        ))
    }

    #[test]
    fn redis_mode_defaults_to_single() {
        let f = redis_backend_yaml("    url: \"redis://127.0.0.1:6379\"");
        let cfg = Config::load_from_path(Some(f.path())).unwrap();
        let redis = cfg.ratelimit.redis.unwrap();
        assert_eq!(redis.mode, RedisMode::Single);
        assert_eq!(redis.url.as_deref(), Some("redis://127.0.0.1:6379"));
    }

    #[test]
    fn redis_single_mode_requires_url() {
        let f = redis_backend_yaml("    mode: \"single\"");
        let err = Config::load_from_path(Some(f.path())).unwrap_err();
        assert!(err.to_string().contains("ratelimit.redis.url"));
    }

    #[test]
    fn redis_cluster_mode_parses_and_requires_nodes() {
        let ok = redis_backend_yaml(
            "    mode: \"cluster\"\n    nodes: [\"redis://n1:6379\", \"redis://n2:6379\"]",
        );
        let cfg = Config::load_from_path(Some(ok.path())).unwrap();
        let redis = cfg.ratelimit.redis.unwrap();
        assert_eq!(redis.mode, RedisMode::Cluster);
        assert_eq!(redis.nodes.len(), 2);

        let bad = redis_backend_yaml("    mode: \"cluster\"");
        let err = Config::load_from_path(Some(bad.path())).unwrap_err();
        assert!(err.to_string().contains("ratelimit.redis.nodes"));
    }

    #[test]
    fn redis_sentinel_mode_parses_and_requires_master_name() {
        let ok = redis_backend_yaml(
            "    mode: \"sentinel\"\n    sentinels: [\"redis://s1:26379\"]\n    master_name: \"mymaster\"",
        );
        let cfg = Config::load_from_path(Some(ok.path())).unwrap();
        let redis = cfg.ratelimit.redis.unwrap();
        assert_eq!(redis.mode, RedisMode::Sentinel);
        assert_eq!(redis.master_name.as_deref(), Some("mymaster"));

        // ACL username/password + database for the discovered master parse.
        let acl = redis_backend_yaml(
            "    mode: \"sentinel\"\n    sentinels: [\"redis://s1:26379\"]\n    master_name: \"m\"\n    username: \"default\"\n    password: \"s3cret\"\n    database: 2",
        );
        let redis = Config::load_from_path(Some(acl.path()))
            .unwrap()
            .ratelimit
            .redis
            .unwrap();
        assert_eq!(redis.username.as_deref(), Some("default"));
        assert_eq!(redis.password.as_deref(), Some("s3cret"));
        assert_eq!(redis.database, Some(2));

        let no_master =
            redis_backend_yaml("    mode: \"sentinel\"\n    sentinels: [\"redis://s1:26379\"]");
        let err = Config::load_from_path(Some(no_master.path())).unwrap_err();
        assert!(err.to_string().contains("ratelimit.redis.master_name"));

        let no_sentinels = redis_backend_yaml("    mode: \"sentinel\"\n    master_name: \"m\"");
        let err = Config::load_from_path(Some(no_sentinels.path())).unwrap_err();
        assert!(err.to_string().contains("ratelimit.redis.sentinels"));
    }

    #[test]
    fn loads_ratelimit_redis_config() {
        let f = write_yaml(
            r#"
etcd:
  endpoints: ["http://localhost:2379"]
proxy:
  addr: "0.0.0.0:3000"
admin:
  addr: "127.0.0.1:3001"
  admin_keys: ["k1"]
ratelimit:
  backend: "redis"
  redis:
    url: "redis://127.0.0.1:6379"
  concurrency_ttl_secs: 120
"#,
        );
        let cfg = Config::load_from_path(Some(f.path())).unwrap();
        assert_eq!(cfg.ratelimit.backend, RateLimitBackend::Redis);
        assert_eq!(
            cfg.ratelimit.redis.as_ref().unwrap().url.as_deref(),
            Some("redis://127.0.0.1:6379")
        );
        assert_eq!(cfg.ratelimit.concurrency_ttl_secs, 120);
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
        // An explicit metrics listener address parses and round-trips.
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
      addr: "127.0.0.1:19090"
"#,
        );
        let cfg = Config::load_from_path(Some(f.path())).unwrap();
        assert_eq!(cfg.observability.metrics.prometheus.addr, "127.0.0.1:19090");
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
    fn shipped_managed_config_binds_the_metrics_listener() {
        // The baked managed-image config (`config.managed.yaml`) is only
        // COPYd into the image, so nothing else catches a typo that would
        // silently un-scrape every managed DP. Pin the scrape address.
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config.managed.yaml");
        let cfg =
            Config::load_from_path(Some(Path::new(path))).expect("config.managed.yaml must load");
        assert!(cfg.managed.is_managed());
        assert!(cfg.observability.metrics.prometheus.enabled);
        assert_eq!(
            cfg.observability.metrics.prometheus.addr, "0.0.0.0:9090",
            "managed DPs are scraped on the dedicated metrics listener",
        );
        assert_eq!(cfg.admin.addr, "127.0.0.1:0");
    }

    #[test]
    fn shipped_example_config_binds_the_metrics_listener() {
        // `config.example.yaml` is the self-hosted reference shape; pin
        // the explicit unified scrape address so standalone and managed
        // deployments document the same metrics surface.
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config.example.yaml");
        let cfg =
            Config::load_from_path(Some(Path::new(path))).expect("config.example.yaml must load");
        assert!(cfg.observability.metrics.prometheus.enabled);
        assert_eq!(cfg.observability.metrics.prometheus.addr, "0.0.0.0:9090");
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
