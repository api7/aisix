//! `Model` entity — the routing target users reference from API requests
//! (spec §3).
//!
//! A Model has a user-chosen unique `name`, a canonical `model` id of the
//! form `<provider>/<model_name>`, a `provider_config` block with the
//! upstream API key, an optional `timeout`, and an optional `rate_limit`.
//!
//! etcd path: `{prefix}/models/{uuid}`. Secondary index on `name`.

use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};

use super::rate_limit::RateLimit;
use super::routing::Routing;
use crate::resource::Resource;

static MODEL_ID_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^(anthropic|deepseek|gemini|openai|router)/.+$").unwrap());

/// Supported provider prefixes. The `model` field must start with one of these
/// followed by `/<upstream-model-id>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    Openai,
    Anthropic,
    Gemini,
    Deepseek,
}

impl Provider {
    pub const fn default_base_url(self) -> &'static str {
        match self {
            Self::Openai => "https://api.openai.com",
            Self::Anthropic => "https://api.anthropic.com",
            Self::Gemini => "https://generativelanguage.googleapis.com/v1beta/openai",
            Self::Deepseek => "https://api.deepseek.com",
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Openai => "openai",
            Self::Anthropic => "anthropic",
            Self::Gemini => "gemini",
            Self::Deepseek => "deepseek",
        }
    }
}

/// Upstream provider credentials and endpoint override.
///
/// Per spec §3, ProviderConfig is *untagged* — the provider type is derived
/// from the `model` field's prefix, not from a discriminator in this struct.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ProviderConfig {
    pub api_key: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_base: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    pub name: String,
    pub model: String,
    pub provider_config: ProviderConfig,

    /// Request timeout in ms. 0 or absent = no timeout.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout: Option<u64>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate_limit: Option<RateLimit>,

    /// Optional routing config. When present, the proxy treats this Model
    /// as a *virtual* router: it picks one of the listed targets per the
    /// strategy and dispatches through that target's bridge instead of
    /// using `model` / `provider_config` on this entity. The base `model`
    /// field is still required (use the `router/<name>` prefix to make
    /// the intent obvious to operators).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing: Option<Routing>,

    /// Non-schema runtime id. Not part of the JSON payload — filled in by
    /// the snapshot loader from the etcd key path. Kept here so `Resource`
    /// can return a `&str` id.
    #[serde(skip)]
    pub(crate) runtime_id: String,
}

impl Model {
    /// Parse the `<provider>/<model_name>` prefix. Returns `None` for
    /// the `router/...` virtual-routing form (which intentionally has
    /// no upstream provider) and for malformed values.
    pub fn provider(&self) -> Option<Provider> {
        let (prefix, _) = self.model.split_once('/')?;
        match prefix {
            "openai" => Some(Provider::Openai),
            "anthropic" => Some(Provider::Anthropic),
            "gemini" => Some(Provider::Gemini),
            "deepseek" => Some(Provider::Deepseek),
            _ => None,
        }
    }

    /// Whether this Model is a virtual router (proxy walks `routing.targets`
    /// instead of dispatching its own provider config).
    pub fn is_routing(&self) -> bool {
        self.routing.is_some()
    }

    /// Returns the upstream-facing model id (everything after the first `/`).
    pub fn upstream_model(&self) -> Option<&str> {
        self.model.split_once('/').map(|(_, m)| m)
    }

    /// The base URL reqwest should use — provider default or the explicit
    /// `api_base` override.
    pub fn base_url(&self) -> Option<String> {
        let base = self
            .provider_config
            .api_base
            .clone()
            .or_else(|| self.provider().map(|p| p.default_base_url().to_string()))?;
        Some(base)
    }

    /// True if the model id matches the spec regex.
    pub fn has_valid_model_id(&self) -> bool {
        MODEL_ID_RE.is_match(&self.model)
    }
}

impl Resource for Model {
    fn id(&self) -> &str {
        &self.runtime_id
    }

    fn name(&self) -> &str {
        &self.name
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
          "name": "my-gpt4",
          "model": "openai/gpt-4o",
          "provider_config": {
            "api_key": "sk-redacted",
            "api_base": "https://api.openai.com/v1"
          },
          "timeout": 30000,
          "rate_limit": {"rpm": 100, "tpm": 100000}
        }"#
    }

    #[test]
    fn deserialises_spec_sample() {
        let m: Model = serde_json::from_str(sample_json()).unwrap();
        assert_eq!(m.name, "my-gpt4");
        assert_eq!(m.provider(), Some(Provider::Openai));
        assert_eq!(m.upstream_model(), Some("gpt-4o"));
        assert_eq!(m.base_url().as_deref(), Some("https://api.openai.com/v1"));
        assert_eq!(m.timeout, Some(30_000));
        assert_eq!(m.rate_limit.as_ref().unwrap().rpm, Some(100));
    }

    #[test]
    fn base_url_falls_back_to_provider_default() {
        let m: Model = serde_json::from_str(
            r#"{
              "name": "x",
              "model": "anthropic/claude-3",
              "provider_config": {"api_key": "k"}
            }"#,
        )
        .unwrap();
        assert_eq!(m.base_url().as_deref(), Some("https://api.anthropic.com"));
    }

    #[test]
    fn rejects_unknown_top_level_fields() {
        let r: Result<Model, _> = serde_json::from_str(
            r#"{
              "name":"x","model":"openai/gpt-4","provider_config":{"api_key":"k"},
              "foo": 1
            }"#,
        );
        assert!(r.is_err());
    }

    #[test]
    fn has_valid_model_id_matches_spec_regex() {
        let mk = |s: &str| Model {
            name: "x".into(),
            model: s.into(),
            provider_config: ProviderConfig {
                api_key: "k".into(),
                api_base: None,
            },
            timeout: None,
            rate_limit: None,
            routing: None,
            runtime_id: String::new(),
        };

        assert!(mk("openai/gpt-4").has_valid_model_id());
        assert!(mk("anthropic/claude-3.5").has_valid_model_id());
        assert!(mk("deepseek/dsv3").has_valid_model_id());
        assert!(mk("gemini/gemini-1.5-pro").has_valid_model_id());
        assert!(!mk("mistral/large").has_valid_model_id());
        assert!(!mk("openai").has_valid_model_id());
        assert!(!mk("openai/").has_valid_model_id());
    }

    #[test]
    fn resource_trait_routes_through_name() {
        let mut m: Model = serde_json::from_str(sample_json()).unwrap();
        m.runtime_id = "uuid-1".into();
        assert_eq!(<Model as Resource>::kind(), "models");
        assert_eq!(m.id(), "uuid-1");
        assert_eq!(m.name(), "my-gpt4");
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
    }
}
