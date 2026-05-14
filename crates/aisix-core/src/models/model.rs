//! `Model` entity — the routing target users reference from API requests.
//!
//! A Model has a user-chosen unique `display_name`, an explicit
//! `provider` enum, an upstream `model_name` (e.g. "gpt-4o"), and a
//! `provider_key_id` referencing a [`ProviderKey`] entry that supplies
//! the secret + optional `api_base` override.
//!
//! Routing models — virtual routers that pick a target Model per request
//! — set `routing` instead of `provider`/`model_name`/`provider_key_id`.
//! See [`Model::is_routing`].
//!
//! etcd path: `{prefix}/models/{uuid}`. Secondary index on `display_name`.

use serde::{Deserialize, Serialize};

use super::rate_limit::RateLimit;
use super::routing::Routing;
use crate::resource::Resource;

/// Supported upstream providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    Openai,
    Anthropic,
    Gemini,
    Deepseek,
    /// Cohere — currently exposed for `/v1/rerank` only (#213 Phase 1).
    /// Cohere's chat / generate APIs are not OpenAI-compatible; a
    /// future bridge implementation can extend coverage.
    Cohere,
    /// Jina AI — currently exposed for `/v1/rerank` only (#213 Phase 2).
    /// Jina's rerank wire shape is identity-mapped to the OpenAI-compat
    /// shape (`{model, query, documents, top_n}` with Bearer auth at
    /// `https://api.jina.ai/v1/rerank`), so the gateway forwards
    /// verbatim with no transform. Jina's chat / embeddings APIs are
    /// out of scope for this phase.
    Jina,
}

impl Provider {
    pub const fn default_base_url(self) -> &'static str {
        match self {
            Self::Openai => "https://api.openai.com",
            Self::Anthropic => "https://api.anthropic.com",
            Self::Gemini => "https://generativelanguage.googleapis.com/v1beta/openai",
            Self::Deepseek => "https://api.deepseek.com",
            Self::Cohere => "https://api.cohere.com",
            Self::Jina => "https://api.jina.ai",
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Openai => "openai",
            Self::Anthropic => "anthropic",
            Self::Gemini => "gemini",
            Self::Deepseek => "deepseek",
            Self::Cohere => "cohere",
            Self::Jina => "jina",
        }
    }
}

/// Per-token cost for budget tracking. Both values are in USD per 1,000 tokens.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelCost {
    /// Input (prompt) token cost in USD per 1,000 tokens.
    pub input_per_1k: f64,
    /// Output (completion) token cost in USD per 1,000 tokens.
    pub output_per_1k: f64,
}

impl ModelCost {
    /// Calculate USD cost for the given token counts.
    pub fn calculate(&self, input_tokens: u64, output_tokens: u64) -> f64 {
        let input_cost = self.input_per_1k * (input_tokens as f64) / 1000.0;
        let output_cost = self.output_per_1k * (output_tokens as f64) / 1000.0;
        input_cost + output_cost
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BackgroundModelCheck {
    pub enabled: bool,
    pub interval_seconds: u64,
    pub timeout_seconds: u64,
    pub prompt: String,
    pub max_tokens: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ignore_statuses: Vec<u16>,
    pub stale_after_seconds: u64,
}

/// Request-path cooldown configuration for a direct model. Controls
/// which upstream failures temporarily exclude this model from routing
/// candidate selection, and for how long.
///
/// Cooldown is **independent** of request retry semantics — i.e.
/// `Routing.retry_on_429` governs whether a 429 is retried within the
/// current request, but `CooldownConfig.trigger_statuses` governs
/// whether 429 takes the model out of rotation for subsequent
/// requests. The two layers serve different purposes:
/// - retry: short-window in-request recovery
/// - cooldown: medium-window cross-request backpressure
///
/// All fields are optional; defaults preserve a safe behavior for any
/// direct model that doesn't ship a `cooldown` block.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(deny_unknown_fields)]
pub struct CooldownConfig {
    /// Whether cooldown is active for this model. Default: true.
    /// Set to `false` to disable cooldown entirely (the model stays in
    /// rotation regardless of upstream failures).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    /// Cooldown TTL in seconds when the upstream did not supply a
    /// `Retry-After` header (or `honor_retry_after=false`). Default: 30.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_seconds: Option<u64>,
    /// Upper bound on cooldown TTL. Caps a misbehaving upstream that
    /// returns an unreasonable `Retry-After` value. Default: 600 (10 min).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_seconds: Option<u64>,
    /// Whether to use the upstream's `Retry-After` header (seconds form)
    /// as the cooldown TTL when present. Default: true.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub honor_retry_after: Option<bool>,
    /// Status codes that trigger cooldown. Default:
    /// `[401, 408, 429, 500, 502, 503, 504]` — auth failures and rate
    /// limits + transient server errors. `400/403/422` etc. are caller
    /// mistakes and intentionally excluded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trigger_statuses: Option<Vec<u16>>,
    /// Whether request-path timeouts trigger cooldown. Default: true.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trigger_on_timeout: Option<bool>,
    /// Whether transport / decode / stream-abort errors trigger
    /// cooldown. Default: true.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trigger_on_transport: Option<bool>,
}

/// Default cooldown trigger statuses applied when the operator does
/// not override `trigger_statuses` on a direct model.
pub const DEFAULT_COOLDOWN_TRIGGER_STATUSES: &[u16] = &[401, 408, 429, 500, 502, 503, 504];

const DEFAULT_COOLDOWN_SECONDS: u64 = 30;
const DEFAULT_COOLDOWN_MAX_SECONDS: u64 = 600;

impl CooldownConfig {
    pub fn enabled_or_default(&self) -> bool {
        self.enabled.unwrap_or(true)
    }

    pub fn default_seconds_or_default(&self) -> u64 {
        self.default_seconds.unwrap_or(DEFAULT_COOLDOWN_SECONDS)
    }

    pub fn max_seconds_or_default(&self) -> u64 {
        self.max_seconds.unwrap_or(DEFAULT_COOLDOWN_MAX_SECONDS)
    }

    pub fn honor_retry_after_or_default(&self) -> bool {
        self.honor_retry_after.unwrap_or(true)
    }

    /// Effective trigger-status list — operator override OR built-in
    /// default. Returned as `Cow` so callers can avoid copies on the
    /// default path.
    pub fn effective_trigger_statuses(&self) -> std::borrow::Cow<'_, [u16]> {
        match &self.trigger_statuses {
            Some(list) => std::borrow::Cow::Borrowed(list.as_slice()),
            None => std::borrow::Cow::Borrowed(DEFAULT_COOLDOWN_TRIGGER_STATUSES),
        }
    }

    pub fn trigger_on_timeout_or_default(&self) -> bool {
        self.trigger_on_timeout.unwrap_or(true)
    }

    pub fn trigger_on_transport_or_default(&self) -> bool {
        self.trigger_on_transport.unwrap_or(true)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    /// Operator-facing unique label. Surfaces on `/v1/models`,
    /// `req.model` on chat completions, ApiKey.allowed_models, and
    /// the dashboard model list. `Resource::name()` returns this.
    pub display_name: String,

    /// Upstream provider. None for routing models (the router picks
    /// a target whose own `provider` is used at dispatch time).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<Provider>,

    /// Upstream model id sent to the provider (e.g. "gpt-4o",
    /// "claude-sonnet-4-5"). None for routing models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,

    /// References a `ProviderKey` row by id. The bridge resolves this
    /// against `AisixSnapshot::provider_keys` at dispatch time to
    /// fetch the upstream secret + optional `api_base`. None for
    /// routing models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_key_id: Option<String>,

    /// Request timeout in ms. 0 or absent = no timeout.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout: Option<u64>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate_limit: Option<RateLimit>,

    /// Virtual-router config. When set, the proxy walks `routing.targets`
    /// to pick a downstream Model and dispatches against THAT model's
    /// `provider` / `model_name` / `provider_key_id`. The fields on
    /// this entity are intentionally absent in that case.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing: Option<Routing>,

    /// Per-token cost for budget tracking. Absent = no cost tracked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<ModelCost>,

    /// Optional direct-model-only background health-check configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub background_model_check: Option<BackgroundModelCheck>,

    /// Optional direct-model-only request-path cooldown configuration.
    /// When absent, default cooldown semantics apply (see
    /// [`CooldownConfig`] field docs for defaults).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cooldown: Option<CooldownConfig>,

    /// Non-schema runtime id. Not part of the JSON payload — filled in by
    /// the snapshot loader from the etcd key path. Kept here so `Resource`
    /// can return a `&str` id.
    #[serde(skip)]
    pub(crate) runtime_id: String,
}

impl Model {
    /// Whether this Model is a virtual router (proxy walks `routing.targets`
    /// instead of dispatching its own upstream config).
    pub fn is_routing(&self) -> bool {
        self.routing.is_some()
    }

    /// Convenience: borrow the upstream model id if this Model is a
    /// direct (non-routing) entry.
    pub fn upstream_model(&self) -> Option<&str> {
        self.model_name.as_deref()
    }
}

impl Resource for Model {
    fn id(&self) -> &str {
        &self.runtime_id
    }

    fn name(&self) -> &str {
        &self.display_name
    }

    fn kind() -> &'static str {
        "models"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_json() -> &'static str {
        r#"{
          "display_name": "my-gpt4",
          "provider": "openai",
          "model_name": "gpt-4o",
          "provider_key_id": "11111111-1111-1111-1111-111111111111",
          "timeout": 30000,
          "rate_limit": {"rpm": 100, "tpm": 100000}
        }"#
    }

    #[test]
    fn deserialises_spec_sample() {
        let m: Model = serde_json::from_str(sample_json()).unwrap();
        assert_eq!(m.display_name, "my-gpt4");
        assert_eq!(m.provider, Some(Provider::Openai));
        assert_eq!(m.model_name.as_deref(), Some("gpt-4o"));
        assert_eq!(
            m.provider_key_id.as_deref(),
            Some("11111111-1111-1111-1111-111111111111")
        );
        assert_eq!(m.timeout, Some(30_000));
        assert_eq!(m.rate_limit.as_ref().unwrap().rpm, Some(100));
    }

    #[test]
    fn rejects_unknown_top_level_fields() {
        let r: Result<Model, _> = serde_json::from_str(
            r#"{
              "display_name":"x","provider":"openai","model_name":"g",
              "provider_key_id":"pk-1",
              "foo": 1
            }"#,
        );
        assert!(r.is_err());
    }

    #[test]
    fn routing_form_has_no_provider_or_provider_key_id() {
        let m: Model = serde_json::from_str(
            r#"{
              "display_name": "router-1",
              "routing": {
                "strategy": "round_robin",
                "targets": [{"model": "my-gpt4"}, {"model": "my-claude"}]
              }
            }"#,
        )
        .unwrap();
        assert!(m.is_routing());
        assert!(m.provider.is_none());
        assert!(m.model_name.is_none());
        assert!(m.provider_key_id.is_none());
    }

    #[test]
    fn resource_trait_routes_through_display_name() {
        let mut m: Model = serde_json::from_str(sample_json()).unwrap();
        m.runtime_id = "uuid-1".into();
        assert_eq!(<Model as Resource>::kind(), "models");
        assert_eq!(m.id(), "uuid-1");
        assert_eq!(m.name(), "my-gpt4");
    }

    #[test]
    fn cooldown_config_defaults_via_helpers() {
        let cfg = CooldownConfig::default();
        assert!(cfg.enabled_or_default());
        assert_eq!(cfg.default_seconds_or_default(), 30);
        assert_eq!(cfg.max_seconds_or_default(), 600);
        assert!(cfg.honor_retry_after_or_default());
        assert_eq!(
            cfg.effective_trigger_statuses().as_ref(),
            DEFAULT_COOLDOWN_TRIGGER_STATUSES,
        );
        assert!(cfg.trigger_on_timeout_or_default());
        assert!(cfg.trigger_on_transport_or_default());
    }

    #[test]
    fn cooldown_default_trigger_statuses_match_advertised_set() {
        // Lock the documented default so a future change has to update
        // both the constant and the test, surfaced as one diff.
        assert_eq!(
            DEFAULT_COOLDOWN_TRIGGER_STATUSES,
            &[401, 408, 429, 500, 502, 503, 504]
        );
    }

    #[test]
    fn cooldown_config_partial_override_keeps_other_defaults() {
        let cfg: CooldownConfig = serde_json::from_str(r#"{"default_seconds": 90}"#).unwrap();
        assert_eq!(cfg.default_seconds_or_default(), 90);
        // Other fields fall back to defaults.
        assert!(cfg.enabled_or_default());
        assert_eq!(cfg.max_seconds_or_default(), 600);
        assert!(cfg.honor_retry_after_or_default());
    }

    #[test]
    fn cooldown_config_disable_via_enabled_false() {
        let cfg: CooldownConfig = serde_json::from_str(r#"{"enabled": false}"#).unwrap();
        assert!(!cfg.enabled_or_default());
    }

    #[test]
    fn cooldown_config_override_trigger_statuses() {
        let cfg: CooldownConfig = serde_json::from_str(r#"{"trigger_statuses": [503]}"#).unwrap();
        assert_eq!(cfg.effective_trigger_statuses().as_ref(), &[503]);
    }

    #[test]
    fn direct_model_can_deserialize_cooldown_config() {
        let m: Model = serde_json::from_str(
            r#"{
              "display_name": "my-gpt4",
              "provider": "openai",
              "model_name": "gpt-4o",
              "provider_key_id": "11111111-1111-1111-1111-111111111111",
              "cooldown": {
                "enabled": true,
                "default_seconds": 45,
                "trigger_statuses": [429, 503]
              }
            }"#,
        )
        .unwrap();
        let cooldown = m.cooldown.unwrap();
        assert!(cooldown.enabled_or_default());
        assert_eq!(cooldown.default_seconds_or_default(), 45);
        assert_eq!(cooldown.effective_trigger_statuses().as_ref(), &[429, 503]);
    }

    #[test]
    fn direct_model_can_deserialize_background_check() {
        let m: Model = serde_json::from_str(
            r#"{
              "display_name": "my-gpt4",
              "provider": "openai",
              "model_name": "gpt-4o",
              "provider_key_id": "11111111-1111-1111-1111-111111111111",
              "background_model_check": {
                "enabled": true,
                "interval_seconds": 30,
                "timeout_seconds": 10,
                "prompt": "Respond with OK",
                "max_tokens": 8,
                "ignore_statuses": [408, 429],
                "stale_after_seconds": 90
              }
            }"#,
        )
        .unwrap();
        let bg = m.background_model_check.unwrap();
        assert!(bg.enabled);
        assert_eq!(bg.ignore_statuses, vec![408, 429]);
    }

    #[test]
    fn provider_default_urls_are_stable() {
        assert_eq!(
            Provider::Openai.default_base_url(),
            "https://api.openai.com"
        );
        assert_eq!(
            Provider::Anthropic.default_base_url(),
            "https://api.anthropic.com"
        );
        assert_eq!(
            Provider::Gemini.default_base_url(),
            "https://generativelanguage.googleapis.com/v1beta/openai"
        );
        assert_eq!(
            Provider::Deepseek.default_base_url(),
            "https://api.deepseek.com"
        );
        // #213 Phase 1 / Phase 2: rerank-only providers. Per audit
        // LOW-3 on PR #229 — pin these too so a regression that
        // swaps the host (e.g. Jina's `https://api.jina.ai` →
        // `https://api.jina.com`) fails at the unit level.
        assert_eq!(
            Provider::Cohere.default_base_url(),
            "https://api.cohere.com"
        );
        assert_eq!(Provider::Jina.default_base_url(), "https://api.jina.ai");
    }
}
