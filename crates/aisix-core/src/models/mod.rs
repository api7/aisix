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

pub mod a2a_agent;
pub mod apikey;
pub mod cache_policy;
pub mod embedding;
pub mod ensemble;
pub mod guardrail;
pub mod mcp_server;
pub mod model;
pub mod observability_exporter;
pub mod provider_key;
pub mod rate_limit;
pub mod rate_limit_policy;
pub mod routing;
pub mod schema;
pub mod semantic;
pub mod snapshot;

pub use a2a_agent::{A2aAgent, A2aAuthType, A2aProtocolVersion};
pub use apikey::ApiKey;
pub use cache_policy::{AppliesTo, CacheBackend, CachePolicy};
pub use embedding::EmbeddingConfig;
pub use ensemble::{EnsembleConfig, Judge, PanelMember};
pub use guardrail::{
    AliyunTextModerationConfig, AppliedGuardrail, AzureContentSafetyConfig,
    AzureContentSafetyTextModerationConfig, BedrockAWSCredentials, BedrockConfig,
    BedrockLatencyMode, Guardrail, GuardrailAttachment, GuardrailHookPoint, GuardrailKind,
    GuardrailMonitorHit, GuardrailScopeType, KeywordConfig, KeywordPattern, LakeraConfig,
    OpenaiModerationConfig, PiiConfig, PiiCustomPattern, PiiDetectorConfig, PresidioConfig,
    PresidioEntityConfig,
};
pub use mcp_server::{McpAuthType, McpServer, McpTransport};
pub use model::{
    Adapter, BackgroundModelCheck, CooldownConfig, Model, DEFAULT_COOLDOWN_TRIGGER_STATUSES,
};
pub use observability_exporter::{
    AliyunSlsConfig, DatadogConfig, ExporterKind, ObjectStoreCompression, ObjectStoreConfig,
    ObjectStoreProvider, ObservabilityExporter, OtlpHttpConfig, SlsContentMode,
};
pub use provider_key::{
    ParamConstraints, ProviderKey, RequestOverrides, ResponseOverrides, StreamDoneMarker,
    TelemetryKind, TelemetryTags,
};
pub use rate_limit::RateLimit;
pub use rate_limit_policy::{PolicyScope, PolicyWindow, RateLimitPolicy};
pub use routing::{Routing, RoutingStrategy, RoutingTarget, WhenAllUnavailablePolicy};
pub use schema::{
    validate_a2a_agent, validate_apikey, validate_cache_policy, validate_guardrail,
    validate_guardrail_attachment, validate_mcp_server, validate_model,
    validate_observability_exporter, validate_provider_key, validate_rate_limit_policy,
    SchemaError,
};
pub use semantic::{
    Aggregation, DistanceMetric, EmbeddingFailureMode, OnEmbeddingFailure, Semantic, SemanticMatch,
    SemanticRoute,
};
pub use snapshot::AisixSnapshot;
