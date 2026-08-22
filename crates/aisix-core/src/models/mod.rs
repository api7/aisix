//! Typed entities persisted in etcd and loaded into the gateway snapshot.
//!
//! Each entity is paired with a JSON Schema (spec §3) compiled once at
//! startup and reused on both the declarative write paths and the
//! watch read path.
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
pub mod claim_mapping;
pub mod embedding;
pub mod ensemble;
pub mod guardrail;
pub mod mcp_auth_settings;
pub mod mcp_policy;
pub mod mcp_server;
pub mod model;
pub mod observability_exporter;
pub mod oidc_provider;
pub mod passthrough_route;
pub mod policy_conditions;
pub mod provider_key;
pub mod rate_limit;
pub mod rate_limit_policy;
pub mod routing;
pub mod schema;
pub mod semantic;
pub mod snapshot;

pub use a2a_agent::{A2aAgent, A2aAuthType, A2aProtocolVersion};
pub use apikey::ApiKey;
pub use cache_policy::{AppliesTo, CacheBackend, CachePolicy, CacheScope, SemanticCacheConfig};
pub use claim_mapping::{ClaimMapping, ClaimMatch, ClaimMatchOp, ClaimResolve};
pub use embedding::EmbeddingConfig;
pub use ensemble::{EnsembleConfig, Judge, PanelMember};
pub use guardrail::{
    AliyunAiGuardrailConfig, AliyunTextModerationConfig, AppliedGuardrail,
    AzureContentSafetyConfig, AzureContentSafetyTextModerationConfig, BedrockAWSCredentials,
    BedrockConfig, BedrockLatencyMode, Guardrail, GuardrailAttachment, GuardrailEnforcedHit,
    GuardrailExecution, GuardrailHookPoint, GuardrailKind, GuardrailMetricsSink,
    GuardrailMonitorHit, GuardrailScopeType, KeywordConfig, KeywordPattern, LakeraConfig,
    OpenaiModerationConfig, PiiConfig, PiiCustomPattern, PiiDetectorConfig, PresidioConfig,
    PresidioEntityConfig, SemanticCategory, SemanticConfig, SemanticHotwordGroup,
};
pub use mcp_auth_settings::McpAuthSettings;
pub use mcp_policy::{McpAccess, McpPolicy, McpPolicyScope};
pub use mcp_server::{McpAuthType, McpProtocolVersion, McpServer, McpServerType, McpTransport};
pub use model::{
    Adapter, BackgroundModelCheck, CooldownConfig, Model, DEFAULT_COOLDOWN_TRIGGER_STATUSES,
};
pub use observability_exporter::{
    AliyunSlsConfig, DatadogConfig, ExporterKind, ObjectStoreCompression, ObjectStoreConfig,
    ObjectStoreProvider, ObservabilityExporter, OtlpHttpConfig, SlsContentMode,
};
pub use oidc_provider::{BoundClaimExpect, OidcProvider};
pub use passthrough_route::{PassthroughAuthMode, PassthroughCredentialMode, PassthroughRoute};
pub use policy_conditions::{
    eval_condition_nodes, validate_condition_nodes, ConditionGroup, ConditionInput, ConditionLogic,
    ConditionNode, ConditionOperator, ConditionValue, GroupByDimension, PolicyAction,
    PolicyCondition, PolicyDimension,
};
pub use provider_key::{
    ParamConstraints, ProviderKey, RequestOverrides, ResponseOverrides, StreamDoneMarker,
    TelemetryKind, TelemetryTags,
};
pub use rate_limit::{McpRateLimit, RateLimit};
pub use rate_limit_policy::{PolicyScope, PolicyWindow, RateLimitPolicy};
pub use routing::{
    default_hash_on, HashOnSource, HashOnType, Routing, RoutingStrategy, RoutingTarget,
    WhenAllUnavailablePolicy,
};
pub use schema::{
    unknown_field_paths, validate_a2a_agent, validate_a2a_agent_lenient, validate_apikey,
    validate_apikey_lenient, validate_cache_policy, validate_cache_policy_lenient,
    validate_claim_mapping, validate_claim_mapping_lenient, validate_guardrail,
    validate_guardrail_attachment, validate_guardrail_attachment_lenient,
    validate_guardrail_lenient, validate_mcp_auth_settings, validate_mcp_auth_settings_lenient,
    validate_mcp_policy, validate_mcp_policy_lenient, validate_mcp_server,
    validate_mcp_server_lenient, validate_model, validate_model_lenient,
    validate_observability_exporter, validate_observability_exporter_lenient,
    validate_oidc_provider, validate_oidc_provider_lenient, validate_passthrough_route,
    validate_passthrough_route_lenient, validate_provider_key, validate_provider_key_lenient,
    validate_rate_limit_policy, validate_rate_limit_policy_lenient, SchemaError,
};
pub use semantic::{
    Aggregation, DistanceMetric, EmbeddingFailureMode, OnEmbeddingFailure, Semantic, SemanticMatch,
    SemanticRoute,
};
pub use snapshot::AisixSnapshot;
