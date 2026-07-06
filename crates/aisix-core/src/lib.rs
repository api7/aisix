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
pub mod error;
pub mod models;
pub mod resource;
pub mod snapshot;
pub mod wildcard;

pub use config::{
    AdminConfig, CacheBackend, CacheConfig, Config, EtcdConfig, EtcdTlsConfig, ManagedConfig,
    ObservabilityConfig, ProxyConfig, RateLimitBackend, RateLimitConfig, RealIpConfig,
    RedisConnConfig, RedisMode, TlsConfig,
};
pub use error::{
    AdminError, AdminErrorEnvelope, BootstrapError, ProxyError, ProxyErrorEnvelope, RateLimitScope,
};
pub use models::{
    validate_a2a_agent, validate_apikey, validate_cache_policy, validate_guardrail,
    validate_mcp_server, validate_model, validate_observability_exporter, validate_provider_key,
    validate_rate_limit_policy, A2aAgent, A2aAuthType, A2aProtocolVersion, Adapter, AisixSnapshot,
    ApiKey, AppliedGuardrail, CachePolicy, CooldownConfig, ExporterKind, Guardrail,
    GuardrailHookPoint, GuardrailKind, GuardrailMonitorHit, KeywordConfig, KeywordPattern,
    McpAuthType, McpServer, McpTransport, Model, ObservabilityExporter, ParamConstraints,
    PolicyScope, PolicyWindow, ProviderKey, RateLimit, RateLimitPolicy, RequestOverrides,
    ResponseOverrides, Routing, RoutingStrategy, RoutingTarget, SchemaError, StreamDoneMarker,
    TelemetryKind, TelemetryTags, WhenAllUnavailablePolicy, DEFAULT_COOLDOWN_TRIGGER_STATUSES,
};
pub use resource::{Resource, ResourceEntry};
pub use snapshot::{ResourceTable, SnapshotHandle};
