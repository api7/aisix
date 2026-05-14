//! Typed entities persisted in etcd and loaded into the gateway snapshot.
//!
//! Each entity is paired with a JSON Schema (spec §3) compiled once at
//! startup and reused on both the admin write path and the watch read path.
//!
//! Entities landing across the live PR series:
//! - [`Model`] — routing target (§3)
//! - [`ApiKey`] — caller credential (§3)
//! - [`RateLimit`] — shared rate-limit config (§3.4 / §8)
//! - [`Routing`] — virtual-router strategy + targets (§3.5, PR #17)
//! - [`ProviderKey`] — managed upstream secret (§3.6)
//!
//! Team is intentionally absent: it's a SaaS-tier concept owned by
//! the AISIX-Cloud control plane, not by the standalone gateway.
//! Standalone deployments do per-key rate-limiting via
//! `ApiKey::rate_limit`.

pub mod apikey;
pub mod cache_policy;
pub mod guardrail;
pub mod model;
pub mod observability_exporter;
pub mod provider_key;
pub mod rate_limit;
pub mod rate_limit_policy;
pub mod routing;
pub mod schema;
pub mod snapshot;

pub use apikey::ApiKey;
pub use cache_policy::{AppliesTo, CacheBackend, CachePolicy};
pub use guardrail::{
    BedrockAWSCredentials, BedrockConfig, BedrockLatencyMode, Guardrail, GuardrailHookPoint,
    GuardrailKind, KeywordConfig, KeywordPattern,
};
pub use model::{
    BackgroundModelCheck, CooldownConfig, Model, Provider, DEFAULT_COOLDOWN_TRIGGER_STATUSES,
};
pub use observability_exporter::{ExporterKind, ObservabilityExporter, OtlpHttpConfig};
pub use provider_key::ProviderKey;
pub use rate_limit::RateLimit;
pub use rate_limit_policy::RateLimitPolicy;
pub use routing::{OnAllFilteredPolicy, Routing, RoutingStrategy, RoutingTarget};
pub use schema::{
    validate_apikey, validate_cache_policy, validate_guardrail, validate_model,
    validate_observability_exporter, validate_provider_key, validate_rate_limit_policy,
    SchemaError,
};
pub use snapshot::AisixSnapshot;
