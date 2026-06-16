//! `Model` entity — the routing target users reference from API requests.
//!
//! A Model has a user-chosen unique `display_name`, an open vendor
//! string `provider` (e.g. `"openai"`, `"xai"`), an upstream
//! `model_name` (e.g. `"gpt-4o"`), and a `provider_key_id` referencing
//! a [`ProviderKey`] entry that supplies the secret + optional
//! `api_base` override.
//!
//! Routing models — virtual routers that pick a target Model per request
//! — set `routing` instead of `provider`/`model_name`/`provider_key_id`.
//! See [`Model::is_routing`].
//!
//! etcd path: `{prefix}/models/{uuid}`. Secondary index on `display_name`.

use serde::{Deserialize, Serialize};

use super::ensemble::EnsembleConfig;
use super::rate_limit::RateLimit;
use super::routing::Routing;
use crate::resource::Resource;

// `Provider` enum removed as part of #302 Phase A clean cut. Vendor
// identity is an open string on `ProviderKey.provider` /
// `Model.provider` — DP no longer enumerates vendors at compile time.
// Code paths that need vendor-aware dispatch (rerank, messages
// cross-provider routing) compare the string directly.

/// Upstream API protocol family used for provider dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "kebab-case")]
pub enum Adapter {
    Openai,
    Anthropic,
    Bedrock,
    Vertex,
    AzureOpenai,
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

/// Request-path cooldown settings for a direct model after retryable upstream failures.
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
    /// `req.model` on chat completions, `ApiKey.allowed_models`, and
    /// the dashboard model list. `Resource::name()` returns this.
    pub display_name: String,

    /// Upstream vendor identity used for dispatch, compatibility checks, telemetry, and access logs. `None` for routing and ensemble models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,

    /// Upstream model identifier sent in provider requests. `None` for routing and ensemble models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,

    /// References a `ProviderKey` row by id. The bridge resolves this
    /// against `AisixSnapshot::provider_keys` at dispatch time to
    /// fetch the upstream secret + optional `api_base`. None for
    /// routing models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_key_id: Option<String>,

    /// End-to-end timeout in milliseconds for non-streaming upstream calls. `0` or absent disables the non-streaming timeout.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout: Option<u64>,

    /// Maximum gap in milliseconds between upstream streaming chunks. `0` or absent falls back to `timeout`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_timeout: Option<u64>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate_limit: Option<RateLimit>,

    /// Client IP allowlist in CIDR notation. Empty or absent allows all clients.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_cidrs: Option<Vec<String>>,

    /// Virtual-router config. When set, the proxy walks `routing.targets`
    /// to pick a downstream Model and dispatches against THAT model's
    /// `provider` / `model_name` / `provider_key_id`. The fields on
    /// this entity are intentionally absent in that case.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing: Option<Routing>,

    /// Ensemble configuration for fan-out panel calls and judge synthesis.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ensemble: Option<EnsembleConfig>,

    /// Per-token cost for budget tracking. Omit it when cost tracking is not needed.
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

    /// Whether this Model is an ensemble (fans out to a panel + judge
    /// instead of dispatching a single upstream).
    pub fn is_ensemble(&self) -> bool {
        self.ensemble.is_some()
    }

    /// Convenience: borrow the upstream model id if this Model is a
    /// direct (non-routing) entry.
    pub fn upstream_model(&self) -> Option<&str> {
        self.model_name.as_deref()
    }

    /// Non-streaming request deadline derived from `timeout`. Folds the
    /// `0`/absent "no timeout" sentinel into `None` so callers can apply
    /// it unconditionally with `if let Some(d) = ...`.
    pub fn request_timeout(&self) -> Option<std::time::Duration> {
        self.timeout
            .filter(|&ms| ms > 0)
            .map(std::time::Duration::from_millis)
    }

    /// Streaming per-chunk read deadline derived from `stream_timeout`.
    /// Same `0`/absent → `None` folding as [`Model::request_timeout`].
    pub fn stream_read_timeout(&self) -> Option<std::time::Duration> {
        self.stream_timeout
            .filter(|&ms| ms > 0)
            .map(std::time::Duration::from_millis)
    }

    /// Effective deadline for a streaming request: a positive
    /// `stream_timeout`, otherwise the non-streaming `timeout`. Applied to the
    /// connect phase, the per-chunk read timeout, and the first-chunk
    /// failover gate. Because `stream_read_timeout()` folds `0` to `None`,
    /// `stream_timeout: 0` is treated the same as absent — it falls back to
    /// `timeout` rather than disabling the streaming timeout. `None` (both
    /// unset or `0`) = no streaming timeout. Note: a model that sets only a
    /// small `timeout` therefore also gets that value as its streaming
    /// budget.
    pub fn stream_timeout_effective(&self) -> Option<std::time::Duration> {
        self.stream_read_timeout()
            .or_else(|| self.request_timeout())
    }

    /// Whether a client at `source_ip` may access this model (#557).
    ///
    /// Returns `true` when no `allowed_cidrs` restriction is configured (the
    /// common case). When a restriction is set, returns `true` only if
    /// `source_ip` parses as an IP address contained in at least one range.
    /// Malformed CIDR entries are skipped; an empty or unparseable `source_ip`
    /// against a configured restriction returns `false` (fail closed) so an
    /// unattributable request can never slip past an allowlist.
    pub fn ip_allowed(&self, source_ip: &str) -> bool {
        let ranges = match self.allowed_cidrs.as_deref() {
            Some(r) if !r.is_empty() => r,
            _ => return true,
        };
        let ip: std::net::IpAddr = match source_ip.parse() {
            Ok(ip) => ip,
            Err(_) => return false,
        };
        ranges
            .iter()
            .filter_map(|cidr| cidr.parse::<ipnet::IpNet>().ok())
            .any(|net| net.contains(&ip))
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
        assert_eq!(m.provider.as_deref(), Some("openai"));
        assert_eq!(m.model_name.as_deref(), Some("gpt-4o"));
        assert_eq!(
            m.provider_key_id.as_deref(),
            Some("11111111-1111-1111-1111-111111111111")
        );
        assert_eq!(m.timeout, Some(30_000));
        assert_eq!(m.rate_limit.as_ref().unwrap().rpm, Some(100));
    }

    #[test]
    fn deserialises_stream_timeout_and_helpers_fold_zero() {
        let m: Model = serde_json::from_str(
            r#"{
              "display_name": "my-gpt4",
              "provider": "openai",
              "model_name": "gpt-4o",
              "provider_key_id": "pk-1",
              "timeout": 30000,
              "stream_timeout": 2500
            }"#,
        )
        .unwrap();
        assert_eq!(m.stream_timeout, Some(2_500));
        assert_eq!(
            m.request_timeout(),
            Some(std::time::Duration::from_millis(30_000))
        );
        assert_eq!(
            m.stream_read_timeout(),
            Some(std::time::Duration::from_millis(2_500))
        );

        // Absent → None.
        let none: Model = serde_json::from_str(
            r#"{"display_name":"x","provider":"openai","model_name":"g","provider_key_id":"pk-1"}"#,
        )
        .unwrap();
        assert_eq!(none.request_timeout(), None);
        assert_eq!(none.stream_read_timeout(), None);

        // Explicit 0 is the "no timeout" sentinel → None.
        let zero: Model = serde_json::from_str(
            r#"{"display_name":"x","provider":"openai","model_name":"g","provider_key_id":"pk-1","timeout":0,"stream_timeout":0}"#,
        )
        .unwrap();
        assert_eq!(zero.request_timeout(), None);
        assert_eq!(zero.stream_read_timeout(), None);

        // stream_timeout_effective cascade: prefer stream_timeout when set.
        assert_eq!(
            m.stream_timeout_effective(),
            Some(std::time::Duration::from_millis(2_500))
        );
        // Falls back to `timeout` when stream_timeout is absent.
        let timeout_only: Model = serde_json::from_str(
            r#"{"display_name":"x","provider":"openai","model_name":"g","provider_key_id":"pk-1","timeout":5000}"#,
        )
        .unwrap();
        assert_eq!(
            timeout_only.stream_timeout_effective(),
            Some(std::time::Duration::from_millis(5_000))
        );
        // None when neither is set, and when both are the 0 sentinel.
        assert_eq!(none.stream_timeout_effective(), None);
        assert_eq!(zero.stream_timeout_effective(), None);

        // Explicit `stream_timeout: 0` folds to absent → falls back to
        // `timeout`, not "disable streaming".
        let stream_zero_timeout_set: Model = serde_json::from_str(
            r#"{"display_name":"x","provider":"openai","model_name":"g","provider_key_id":"pk-1","timeout":5000,"stream_timeout":0}"#,
        )
        .unwrap();
        assert_eq!(stream_zero_timeout_set.stream_read_timeout(), None);
        assert_eq!(
            stream_zero_timeout_set.stream_timeout_effective(),
            Some(std::time::Duration::from_millis(5_000))
        );
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
    fn ip_allowed_matrix() {
        fn model_with(cidrs: Option<Vec<&str>>) -> Model {
            let mut m: Model = serde_json::from_str(sample_json()).unwrap();
            m.allowed_cidrs = cidrs.map(|c| c.into_iter().map(String::from).collect());
            m
        }

        // No restriction → everything allowed.
        let open = model_with(None);
        assert!(open.ip_allowed("203.0.113.7"));
        assert!(open.ip_allowed("")); // even an unresolved IP

        // Empty list behaves like no restriction.
        assert!(model_with(Some(vec![])).ip_allowed("203.0.113.7"));

        // IPv4 allowlist: in-range allowed, out-of-range denied.
        let v4 = model_with(Some(vec!["10.0.0.0/8", "192.168.1.0/24"]));
        assert!(v4.ip_allowed("10.1.2.3"));
        assert!(v4.ip_allowed("192.168.1.42"));
        assert!(!v4.ip_allowed("114.114.114.114"));
        assert!(!v4.ip_allowed("192.168.2.1"));

        // Fail closed: a restriction set but the client IP is empty/garbage.
        assert!(!v4.ip_allowed(""));
        assert!(!v4.ip_allowed("not-an-ip"));

        // IPv6 allowlist.
        let v6 = model_with(Some(vec!["2001:db8::/32"]));
        assert!(v6.ip_allowed("2001:db8::1"));
        assert!(!v6.ip_allowed("2001:db9::1"));

        // Malformed CIDR entries are skipped, valid ones still apply.
        let mixed = model_with(Some(vec!["garbage", "10.0.0.0/8"]));
        assert!(mixed.ip_allowed("10.0.0.1"));
        assert!(!mixed.ip_allowed("203.0.113.1"));
    }

    #[test]
    fn deserialises_allowed_cidrs() {
        let m: Model = serde_json::from_str(
            r#"{"display_name":"x","provider":"openai","model_name":"g","provider_key_id":"pk-1","allowed_cidrs":["10.0.0.0/8"]}"#,
        )
        .unwrap();
        assert_eq!(
            m.allowed_cidrs.as_deref(),
            Some(&["10.0.0.0/8".to_string()][..])
        );

        // Absent field → None → no restriction.
        let none: Model = serde_json::from_str(
            r#"{"display_name":"x","provider":"openai","model_name":"g","provider_key_id":"pk-1"}"#,
        )
        .unwrap();
        assert!(none.allowed_cidrs.is_none());
        assert!(none.ip_allowed("203.0.113.7"));
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
    fn ensemble_form_has_no_provider_and_reports_is_ensemble() {
        let m: Model = serde_json::from_str(
            r#"{
              "display_name": "council",
              "ensemble": {
                "panel": [{"model": "my-gpt4"}, {"model": "my-claude"}],
                "judge": {"model": "my-opus"}
              }
            }"#,
        )
        .unwrap();
        assert!(m.is_ensemble());
        assert!(!m.is_routing());
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

    // `adapter_from_provider_covers_every_variant` removed alongside
    // the `From<Provider> for Adapter` impl — both are dead post-#302
    // Phase A. ProviderKey.adapter carries the Adapter directly.

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

    // `every_provider_variant_has_as_str_and_adapter` removed —
    // the `Provider` enum it pinned no longer exists post-#302
    // Phase A. Vendor identity is now a free-form string on
    // `ProviderKey.provider` / `Model.provider`.
}
