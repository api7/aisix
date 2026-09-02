//! aisix-core — primitives shared across every other aisix crate.
//!
//! Four responsibilities:
//! 1. **Config** ([`config::Config`]) — bootstrap YAML/TOML/JSON loader.
//! 2. **Resources** ([`resource::Resource`], [`resource::ResourceEntry`]) — trait
//!    and wrapper for every entity stored in etcd.
//! 3. **Snapshot** ([`snapshot::ResourceTable`], [`snapshot::SnapshotHandle`]) —
//!    lock-free read path via `ArcSwap`, O(1) lookup by id or name.
//! 4. **Errors** ([`error::ProxyError`], [`error::AdminError`],
//!    [`error::BootstrapError`]) — the three error envelopes that show up at
//!    the two HTTP surfaces plus startup.
//!
//! This crate is intentionally framework-agnostic — no axum, no reqwest.
//! `IntoResponse` impls live in `aisix-proxy` / `aisix-admin`.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

pub mod config;
pub mod config_status;
pub mod error;
pub mod filesource;
pub mod forwarded_headers;
pub mod header_template;
pub mod models;
pub mod resource;
pub mod similarity;
pub mod snapshot;
pub mod version;
pub mod wildcard;

pub use config::{
    AdminConfig, CacheBackend, CacheConfig, ClientTypeRule, Config, EtcdConfig, EtcdTlsConfig,
    HistogramBucketsConfig, ManagedConfig, ObservabilityConfig, ProxyConfig, RateLimitBackend,
    RateLimitConfig, RealIpConfig, RedisConnConfig, RedisMode, RequestIdConfig, TlsConfig,
    UrlRewriteRule, CREDENTIAL_HEADERS,
};
pub use config_status::{
    hash_bytes, hash_entries, AppliedSnapshot, ConfigMetricsView, ConfigRejectionSnapshot,
    ConfigState, ConfigStatus, ConfigStatusView, IncomingRejection, LoadObservation,
    RejectedResource, SourceKind,
};
pub use error::{
    AdminError, AdminErrorEnvelope, BootstrapError, ProxyError, ProxyErrorEnvelope, RateLimitScope,
};
pub use forwarded_headers::{
    client_header_forwardable, forward_pattern_admits, header_forward_blocked,
    resolve_forwarded_client_headers, EXACT_MATCH_ONLY_HEADERS, GATEWAY_HEADER_PREFIX,
    NEVER_FORWARD_FROM_CLIENT, NEVER_FORWARD_FROM_CLIENT_PREFIXES, NON_FORWARDABLE_HEADERS,
};
pub use header_template::{render_header_template, HeaderVars, HEADER_TEMPLATE_VARS};
pub use models::{
    validate_a2a_agent, validate_apikey, validate_cache_policy, validate_guardrail,
    validate_mcp_server, validate_model, validate_observability_exporter, validate_provider_key,
    validate_rate_limit_policy, A2aAgent, A2aAuthType, A2aProtocolVersion, Adapter, AisixSnapshot,
    ApiEndpoint, ApiKey, ApiSurface, AppliedGuardrail, CachePolicy, CooldownConfig, ExporterKind,
    Guardrail, GuardrailEnforcedHit, GuardrailExecution, GuardrailHookPoint, GuardrailKind,
    GuardrailMetricsSink, GuardrailMonitorHit, GuardrailScore, HashOnSource, HashOnType,
    KeywordConfig, KeywordPattern, McpAuthType, McpProtocolVersion, McpRateLimit, McpServer,
    McpServerType, McpTransport, Model, ObservabilityExporter, ParamConstraints,
    PassthroughAuthMode, PassthroughCredentialMode, PassthroughRoute, PolicyScope, PolicyWindow,
    ProviderApis, ProviderKey, RateLimit, RateLimitPolicy, RequestOverrides, ResponseOverrides,
    Routing, RoutingStrategy, RoutingTarget, SchemaError, StreamDoneMarker, TelemetryKind,
    TelemetryTags, WhenAllUnavailablePolicy, DEFAULT_COOLDOWN_TRIGGER_STATUSES,
};
pub use resource::{Resource, ResourceEntry};
pub use similarity::{best_similarity, best_similarity_by, cosine_similarity};
pub use snapshot::{ResourceTable, SnapshotHandle};
pub use version::BUILD_VERSION;
