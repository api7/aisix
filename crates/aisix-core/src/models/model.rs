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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    Openai,
    Anthropic,
    Google,
    Deepseek,
    /// Cohere — `/v1/rerank` (native, via `aisix-proxy::rerank`) and
    /// chat/completions / embeddings (via `OpenAiBridge::with_name("cohere")`
    /// against `https://api.cohere.com/compatibility/v1`, per
    /// <https://docs.cohere.com/reference/chat>). Chat-compat is the
    /// OpenAI-shape namespace Cohere ships alongside its native
    /// `/v1/chat`. Chat-compat coverage today: the `command-r` /
    /// `command-a` family per
    /// <https://docs.cohere.com/v2/docs/compatibility-api>; legacy
    /// `command` / `command-light` / `command-nightly` go to native
    /// (not yet bridged) and will return upstream 400 if configured
    /// against the chat-compat path.
    Cohere,
    /// Jina AI — currently exposed for `/v1/rerank` only (#213 Phase 2).
    /// Jina's rerank wire shape is identity-mapped to the OpenAI-compat
    /// shape (`{model, query, documents, top_n}` with Bearer auth at
    /// `https://api.jina.ai/v1/rerank`), so the gateway forwards
    /// verbatim with no transform. Jina's chat / embeddings APIs are
    /// out of scope for this phase.
    Jina,
    // ─── OpenAI-adapter long-tail providers (#60 P2-A) ────────────────
    //
    // The 11 variants below all expose an OpenAI-compatible chat
    // completions surface and route through `OpenAiBridge::with_name`
    // — the same dispatch path Deepseek / Google already use. Default
    // base URLs are sourced from each vendor's official OpenAI-compat
    // documentation; operators can override via `ProviderKey.api_base`.
    //
    // serde casing is `lowercase` (inherited from the parent enum) so
    // each variant's wire id is its single-token lowercase form. The
    // `fireworks-ai` variant carries an explicit `#[serde(rename)]`
    // because its catalog id (per models.dev) is hyphenated and would
    // not survive the default lowercase rule.
    /// Groq — OpenAI-compat at `https://api.groq.com/openai/v1`.
    Groq,
    /// Mistral AI — OpenAI-compat at `https://api.mistral.ai/v1`.
    Mistral,
    /// Together.ai — OpenAI-compat at `https://api.together.ai/v1`.
    Togetherai,
    /// Fireworks AI — OpenAI-compat at `https://api.fireworks.ai/inference/v1`.
    /// Wire id is the hyphenated `fireworks-ai` (matches the
    /// models.dev catalog id used by cp-api's adapter_map).
    #[serde(rename = "fireworks-ai")]
    FireworksAi,
    /// Perplexity — OpenAI-compat at `https://api.perplexity.ai`
    /// (chat lives at the host root).
    Perplexity,
    // xAI Grok was scoped out of this PR after audit: `xai` is not in
    // AISIX-Cloud's `internal/cpapi/adapter_map/adapter_map.yaml`
    // Featured table (PR #335 swapped xai → google for rank 8), so
    // adding `Provider::Xai` here without a catalog entry would leave
    // a DP variant cp-api can't write to. Tracked as a separate
    // follow-up: add xai to adapter_map.yaml first, then a one-line
    // Provider::Xai DP-side commit will follow.
    /// Moonshot AI (Kimi) — OpenAI-compat at `https://api.moonshot.cn/v1`.
    Moonshotai,
    /// Alibaba Cloud DashScope — OpenAI-compat at
    /// `https://dashscope.aliyuncs.com/compatible-mode/v1`.
    Alibaba,
    /// Zhipu AI (智谱) — OpenAI-compat at
    /// `https://open.bigmodel.cn/api/paas/v4`.
    Zhipuai,
    /// Baseten — OpenAI-compat at `https://inference.baseten.co/v1`.
    /// Per <https://docs.baseten.co/development/model-apis/openai-clients>,
    /// model APIs are exposed at this host root.
    Baseten,
    /// Hugging Face — OpenAI-compat router at
    /// `https://router.huggingface.co/v1` per
    /// <https://huggingface.co/docs/inference-providers/en/index#openai-compatible-api>.
    Huggingface,
    /// Cerebras Inference — OpenAI-compat at `https://api.cerebras.ai/v1`.
    Cerebras,
}

impl Provider {
    pub const fn default_base_url(self) -> &'static str {
        match self {
            Self::Openai => "https://api.openai.com",
            Self::Anthropic => "https://api.anthropic.com",
            Self::Google => "https://generativelanguage.googleapis.com/v1beta/openai",
            Self::Deepseek => "https://api.deepseek.com",
            Self::Cohere => "https://api.cohere.com",
            Self::Jina => "https://api.jina.ai",
            // Long-tail OpenAI-adapter providers (#60 P2-A) — each
            // url is the vendor's documented OpenAI-compat endpoint.
            Self::Groq => "https://api.groq.com/openai/v1",
            Self::Mistral => "https://api.mistral.ai/v1",
            Self::Togetherai => "https://api.together.ai/v1",
            Self::FireworksAi => "https://api.fireworks.ai/inference/v1",
            Self::Perplexity => "https://api.perplexity.ai",
            Self::Moonshotai => "https://api.moonshot.cn/v1",
            Self::Alibaba => "https://dashscope.aliyuncs.com/compatible-mode/v1",
            Self::Zhipuai => "https://open.bigmodel.cn/api/paas/v4",
            Self::Baseten => "https://inference.baseten.co/v1",
            Self::Huggingface => "https://router.huggingface.co/v1",
            Self::Cerebras => "https://api.cerebras.ai/v1",
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Openai => "openai",
            Self::Anthropic => "anthropic",
            Self::Google => "google",
            Self::Deepseek => "deepseek",
            Self::Cohere => "cohere",
            Self::Jina => "jina",
            // Long-tail OpenAI-adapter providers (#60 P2-A). The
            // wire id matches the catalog id used by cp-api's
            // adapter_map / models.dev.
            Self::Groq => "groq",
            Self::Mistral => "mistral",
            Self::Togetherai => "togetherai",
            Self::FireworksAi => "fireworks-ai",
            Self::Perplexity => "perplexity",
            Self::Moonshotai => "moonshotai",
            Self::Alibaba => "alibaba",
            Self::Zhipuai => "zhipuai",
            Self::Baseten => "baseten",
            Self::Huggingface => "huggingface",
            Self::Cerebras => "cerebras",
        }
    }
}

/// Wire-shape adapter used to talk to an upstream. This is the closed
/// set of upstream protocols the gateway knows how to encode against —
/// distinct from a vendor identity (which is captured separately on
/// `ProviderKey`).
///
/// Introduced as a skeleton for issue #302 Phase A. This type is purely
/// additive in this PR: nothing in the gateway dispatches off `Adapter`
/// yet, no entity field is changed, and `Provider` continues to drive
/// all runtime behavior. Follow-up PRs in Phase A migrate entities and
/// the Hub to consume `Adapter` directly.
///
/// Note on serde casing: `Adapter` uses `kebab-case` so the `AzureOpenai`
/// variant serializes as `"azure-openai"`. This intentionally differs
/// from `Provider`'s `lowercase` casing, which produced no hyphens
/// because all current `Provider` names are single tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum Adapter {
    Openai,
    Anthropic,
    Bedrock,
    Vertex,
    AzureOpenai,
}

impl From<Provider> for Adapter {
    /// Mapping from the current `Provider` enum onto the `Adapter`
    /// wire-shape set, for use during the issue #302 Phase A skeleton
    /// stage. This mapping is **not** wired into dispatch in this PR;
    /// it exists as a forward-looking reference that follow-up Phase A
    /// PRs will consume when migrating entities, the Hub, and Bridges
    /// onto `Adapter`.
    ///
    /// Rationale per variant:
    ///
    /// - `Openai` / `Anthropic`: direct equivalents.
    /// - `Google` → `Vertex`: forward-looking choice for the Phase A
    ///   migration target. **Note:** this intentionally diverges from
    ///   the current runtime behavior — `Provider::Google` today calls
    ///   the Gemini Generative Language API via its OpenAI-compatible
    ///   endpoint (`/v1beta/openai`), which is an `openai`-shaped
    ///   wire. Any downstream PR that flips dispatch onto `Adapter`
    ///   MUST either also migrate the Google bridge to native Vertex
    ///   AI request encoding, or revisit this arm before merging — a
    ///   silent flip would change the wire shape sent to Google
    ///   upstreams. See issue #302 Phase A for the migration plan.
    /// - `Deepseek` → `Openai`: DeepSeek exposes an OpenAI-compatible
    ///   chat completions endpoint, so the adapter shape is `openai`.
    /// - `Cohere` → `Openai`: the gateway currently talks to Cohere's
    ///   OpenAI-compatible endpoints (#213 Phase 1 — rerank-only),
    ///   so the wire adapter is `openai`. A native Cohere adapter is
    ///   not part of this skeleton.
    /// - `Jina` → `Openai`: Jina's rerank is identity-mapped to the
    ///   OpenAI-compat rerank shape (#213 Phase 2), so the wire
    ///   adapter is `openai`.
    fn from(provider: Provider) -> Self {
        match provider {
            Provider::Openai => Adapter::Openai,
            Provider::Anthropic => Adapter::Anthropic,
            Provider::Google => Adapter::Vertex,
            Provider::Deepseek => Adapter::Openai,
            Provider::Cohere => Adapter::Openai,
            Provider::Jina => Adapter::Openai,
            // All long-tail variants (#60 P2-A) expose OpenAI-compat
            // chat surfaces and route through OpenAiBridge.
            Provider::Groq
            | Provider::Mistral
            | Provider::Togetherai
            | Provider::FireworksAi
            | Provider::Perplexity
            | Provider::Moonshotai
            | Provider::Alibaba
            | Provider::Zhipuai
            | Provider::Baseten
            | Provider::Huggingface
            | Provider::Cerebras => Adapter::Openai,
        }
    }
}

/// Per-token cost for budget tracking. Both values are in USD per 1,000 tokens.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq)]
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

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
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
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq, Default)]
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

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
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
    fn adapter_from_provider_covers_every_variant() {
        // Every Provider variant must map to a defined Adapter. This
        // test exists to fail the build if a new Provider is added
        // without an explicit From arm — the match in `From<Provider>`
        // is exhaustive so the real safety net is the compiler, but
        // pinning the chosen Adapter per variant guards against a
        // future edit silently changing the mapping.
        assert_eq!(Adapter::from(Provider::Openai), Adapter::Openai);
        assert_eq!(Adapter::from(Provider::Anthropic), Adapter::Anthropic);
        assert_eq!(Adapter::from(Provider::Google), Adapter::Vertex);
        assert_eq!(Adapter::from(Provider::Deepseek), Adapter::Openai);
        assert_eq!(Adapter::from(Provider::Cohere), Adapter::Openai);
        assert_eq!(Adapter::from(Provider::Jina), Adapter::Openai);
    }

    #[test]
    fn adapter_serializes_to_kebab_case_wire_strings() {
        // Pin each Adapter's wire form. AzureOpenai → "azure-openai"
        // is the load-bearing case for the kebab-case choice; the
        // others are pinned to lock the contract so a future
        // rename_all change is surfaced as a test failure.
        assert_eq!(
            serde_json::to_string(&Adapter::Openai).unwrap(),
            "\"openai\""
        );
        assert_eq!(
            serde_json::to_string(&Adapter::Anthropic).unwrap(),
            "\"anthropic\""
        );
        assert_eq!(
            serde_json::to_string(&Adapter::Bedrock).unwrap(),
            "\"bedrock\""
        );
        assert_eq!(
            serde_json::to_string(&Adapter::Vertex).unwrap(),
            "\"vertex\""
        );
        assert_eq!(
            serde_json::to_string(&Adapter::AzureOpenai).unwrap(),
            "\"azure-openai\""
        );
    }

    #[test]
    fn adapter_deserializes_from_kebab_case_wire_strings() {
        assert_eq!(
            serde_json::from_str::<Adapter>("\"openai\"").unwrap(),
            Adapter::Openai
        );
        assert_eq!(
            serde_json::from_str::<Adapter>("\"anthropic\"").unwrap(),
            Adapter::Anthropic
        );
        assert_eq!(
            serde_json::from_str::<Adapter>("\"bedrock\"").unwrap(),
            Adapter::Bedrock
        );
        assert_eq!(
            serde_json::from_str::<Adapter>("\"vertex\"").unwrap(),
            Adapter::Vertex
        );
        assert_eq!(
            serde_json::from_str::<Adapter>("\"azure-openai\"").unwrap(),
            Adapter::AzureOpenai
        );
    }

    #[test]
    fn adapter_rejects_unknown_variant_strings() {
        // Closed enum — any string outside the kebab-case wire set
        // must fail to deserialize so callers can't silently smuggle
        // in a typo or a legacy provider name.
        assert!(serde_json::from_str::<Adapter>("\"gemini\"").is_err());
        assert!(serde_json::from_str::<Adapter>("\"azureopenai\"").is_err());
        assert!(serde_json::from_str::<Adapter>("\"azure_openai\"").is_err());
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
            Provider::Google.default_base_url(),
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

    /// Every `Provider` variant must have a non-empty `default_base_url`,
    /// `as_str`, and an `Adapter::from` arm. A regression that added a
    /// new variant but forgot to update one of the three would compile
    /// fine but silently break dispatch downstream — this iterator-based
    /// test makes the regression class explicit (audit MEDIUM-2 on
    /// PR #345).
    #[test]
    fn every_provider_variant_has_default_base_url_as_str_and_adapter() {
        let variants = [
            Provider::Openai,
            Provider::Anthropic,
            Provider::Google,
            Provider::Deepseek,
            Provider::Cohere,
            Provider::Jina,
            // Long-tail OpenAI-adapter providers (#60 P2-A).
            Provider::Groq,
            Provider::Mistral,
            Provider::Togetherai,
            Provider::FireworksAi,
            Provider::Perplexity,
            Provider::Moonshotai,
            Provider::Alibaba,
            Provider::Zhipuai,
            Provider::Baseten,
            Provider::Huggingface,
            Provider::Cerebras,
        ];
        for v in variants {
            let url = v.default_base_url();
            assert!(
                url.starts_with("https://"),
                "{v:?}: default_base_url must be an https URL, got {url:?}",
            );
            let id = v.as_str();
            assert!(!id.is_empty(), "{v:?}: as_str must be non-empty");
            // Adapter::from must compile and return some variant.
            let _: Adapter = Adapter::from(v);
        }
    }
}
