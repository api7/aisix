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

pub use config::{
    AdminConfig, CacheBackend, CacheConfig, Config, EtcdConfig, EtcdTlsConfig, LangfuseConfig,
    ManagedConfig, ObservabilityConfig, ProxyConfig, TlsConfig,
};
pub use error::{
    AdminError, AdminErrorEnvelope, BootstrapError, ProxyError, ProxyErrorEnvelope, RateLimitScope,
};
pub use models::{
    validate_apikey, validate_budget, validate_credential, validate_model, validate_team,
    AisixSnapshot, ApiKey, Budget, Credential, Model, Provider, ProviderConfig, RateLimit, Routing,
    RoutingStrategy, RoutingTarget, SchemaError, Team,
};
pub use resource::{Resource, ResourceEntry};
pub use snapshot::{ResourceTable, SnapshotHandle};
