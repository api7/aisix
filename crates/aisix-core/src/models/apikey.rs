//! `ApiKey` entity — the caller-facing credential presented in
//! `Authorization: Bearer <key>` (spec §3, §7).

use serde::{Deserialize, Serialize};

use super::rate_limit::RateLimit;
use crate::resource::Resource;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApiKey {
    /// The actual secret bearer value. Secondary-indexed for O(1) auth.
    pub key: String,

    /// Whitelisted Model names. An **empty array** denies every model —
    /// it is not a shortcut for "all" (spec §3 authz rule).
    pub allowed_models: Vec<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rate_limit: Option<RateLimit>,

    /// Maximum USD spend per calendar month. When the accumulated spend
    /// for this key reaches or exceeds this cap the proxy returns 429.
    /// Absent = no budget enforcement.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_budget_usd: Option<f64>,

    /// etcd-key uuid; filled by the loader, never in the JSON payload.
    #[serde(skip)]
    pub(crate) runtime_id: String,
}

impl ApiKey {
    /// True if this key is allowed to call the given Model.
    ///
    /// A wildcard entry `"*"` grants access to every model, matching
    /// LiteLLM's convention.  An empty `allowed_models` list denies
    /// everything (spec §3 authz rule).
    pub fn can_access(&self, model_name: &str) -> bool {
        self.allowed_models
            .iter()
            .any(|n| n == "*" || n == model_name)
    }

    /// Iterate over the names of models this key may access, filtering
    /// them against a known universe of model names. The `*` wildcard
    /// expands to the full universe so callers don't have to special-case
    /// it themselves.
    pub fn accessible_models<'a>(
        &'a self,
        all_models: impl Iterator<Item = &'a str> + 'a,
    ) -> Vec<&'a str> {
        let has_wildcard = self.allowed_models.iter().any(|n| n == "*");
        if has_wildcard {
            all_models.collect()
        } else {
            all_models
                .filter(|name| self.allowed_models.iter().any(|n| n.as_str() == *name))
                .collect()
        }
    }
}

impl Resource for ApiKey {
    fn id(&self) -> &str {
        &self.runtime_id
    }

    /// For ApiKey the "secondary-indexed" field is `key`, not a human name.
    /// The name-index in the snapshot therefore points from key → id.
    fn name(&self) -> &str {
        &self.key
    }

    fn kind() -> &'static str {
        "apikeys"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> ApiKey {
        serde_json::from_str(
            r#"{
              "key": "sk-my-api-key-123",
              "allowed_models": ["my-gpt4", "my-claude"],
              "rate_limit": {"rpm": 60, "concurrency": 5}
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn deserialises_spec_sample() {
        let k = sample();
        assert_eq!(k.key, "sk-my-api-key-123");
        assert_eq!(k.allowed_models.len(), 2);
        assert_eq!(k.rate_limit.as_ref().unwrap().concurrency, Some(5));
    }

    #[test]
    fn empty_allowed_models_denies_everything() {
        let k = ApiKey {
            key: "sk-x".into(),
            allowed_models: vec![],
            rate_limit: None,
            max_budget_usd: None,
            runtime_id: String::new(),
        };
        assert!(!k.can_access("my-gpt4"));
        assert!(!k.can_access("anything"));
    }

    #[test]
    fn can_access_checks_whitelist() {
        let k = sample();
        assert!(k.can_access("my-gpt4"));
        assert!(k.can_access("my-claude"));
        assert!(!k.can_access("other"));
    }

    #[test]
    fn wildcard_grants_access_to_any_model() {
        let k: ApiKey =
            serde_json::from_str(r#"{"key":"sk-x","allowed_models":["*"]}"#).unwrap();
        assert!(k.can_access("my-gpt4"));
        assert!(k.can_access("literally-anything"));
    }

    #[test]
    fn accessible_models_expands_wildcard_to_full_universe() {
        let k: ApiKey =
            serde_json::from_str(r#"{"key":"sk-x","allowed_models":["*"]}"#).unwrap();
        let universe = ["a", "b", "c"];
        let accessible = k.accessible_models(universe.iter().copied());
        assert_eq!(accessible, vec!["a", "b", "c"]);
    }

    #[test]
    fn accessible_models_filters_explicit_list() {
        let k = sample(); // allowed: ["my-gpt4", "my-claude"]
        let universe = ["my-gpt4", "my-claude", "other"];
        let mut accessible = k.accessible_models(universe.iter().copied());
        accessible.sort_unstable();
        assert_eq!(accessible, vec!["my-claude", "my-gpt4"]);
    }

    #[test]
    fn accessible_models_empty_list_returns_nothing() {
        let k: ApiKey =
            serde_json::from_str(r#"{"key":"sk-x","allowed_models":[]}"#).unwrap();
        let universe = ["a", "b"];
        assert!(k.accessible_models(universe.iter().copied()).is_empty());
    }

    #[test]
    fn rejects_unknown_fields() {
        let r: Result<ApiKey, _> =
            serde_json::from_str(r#"{"key":"x","allowed_models":[],"extra":1}"#);
        assert!(r.is_err());
    }

    #[test]
    fn resource_trait_points_at_key_and_kind() {
        let mut k = sample();
        k.runtime_id = "uuid-ak".into();
        assert_eq!(<ApiKey as Resource>::kind(), "apikeys");
        assert_eq!(k.id(), "uuid-ak");
        assert_eq!(k.name(), "sk-my-api-key-123");
    }
}
