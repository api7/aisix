//! Helpers shared by every endpoint that needs to dispatch to a Bridge.
//!
//! Every endpoint follows the same shape after Model resolution:
//!
//! 1. Take the resolved `Model` (already looked up by display_name).
//! 2. Resolve the `ProviderKey` it references (via `provider_key_id`).
//! 3. Compute the upstream base URL by combining the `Provider`'s
//!    default with the `ProviderKey`'s optional `api_base` override.
//!
//! These helpers existed inline in each endpoint as
//! `Model::base_url()`, `Model::upstream_model()`, and
//! `Model::provider_config.api_key` accessors. Phase B moved the
//! Model from "self-contained inline secret" to "ProviderKey
//! reference"; this module is the join point that recovers the old
//! ergonomics on the proxy side.
//!
//! Returns typed [`ProxyError`] variants so the caller's `?`
//! plumbing flows naturally.

use std::sync::Arc;

use aisix_core::models::Provider;
use aisix_core::resource::ResourceEntry;
use aisix_core::{AisixSnapshot, Model, ProviderKey};

use crate::error::ProxyError;

/// Look up the `ProviderKey` a given `Model` references. Returns a
/// 400 if the Model is a virtual router (those don't dispatch
/// directly — caller should walk `routing.targets` first), or if the
/// referenced ProviderKey row is missing from the snapshot.
pub(crate) fn resolve_provider_key(
    snapshot: &AisixSnapshot,
    model: &Model,
) -> Result<Arc<ResourceEntry<ProviderKey>>, ProxyError> {
    let pk_id = model.provider_key_id.as_deref().ok_or_else(|| {
        ProxyError::InvalidRequest(format!(
            "model {:?} has no provider_key_id (routing models can't be dispatched directly)",
            model.display_name
        ))
    })?;
    snapshot.provider_keys.get_by_id(pk_id).ok_or_else(|| {
        ProxyError::InvalidRequest(format!(
            "model {:?} references unknown provider_key_id {pk_id:?}",
            model.display_name
        ))
    })
}

/// Required `provider` for a non-routing Model. 400 if absent.
pub(crate) fn require_provider(model: &Model) -> Result<Provider, ProxyError> {
    model.provider.ok_or_else(|| {
        ProxyError::InvalidRequest(format!(
            "model {:?} has no provider (routing models can't be dispatched directly)",
            model.display_name
        ))
    })
}

/// Required upstream model id (`model_name`) for a non-routing Model.
pub(crate) fn require_upstream_model(model: &Model) -> Result<&str, ProxyError> {
    model.model_name.as_deref().ok_or_else(|| {
        ProxyError::InvalidRequest(format!(
            "model {:?} has no model_name (routing models can't be dispatched directly)",
            model.display_name
        ))
    })
}

/// The upstream base URL: `provider_key.api_base` override if set,
/// otherwise the `Provider`'s built-in default.
pub(crate) fn resolve_base_url(provider: Provider, provider_key: &ProviderKey) -> String {
    match provider_key.api_base.as_deref() {
        Some(b) if !b.trim().is_empty() => b.trim_end_matches('/').to_string(),
        _ => provider.default_base_url().to_string(),
    }
}

/// The upstream API key — `provider_key.secret`. Empty string is
/// treated as a config error (ProviderKey rows shouldn't be empty,
/// but a hand-edited kine row could surface one).
pub(crate) fn require_secret<'a>(
    provider_key: &'a ProviderKey,
    model: &Model,
) -> Result<&'a str, ProxyError> {
    if provider_key.secret.is_empty() {
        return Err(ProxyError::InvalidRequest(format!(
            "model {:?} provider_key has empty secret",
            model.display_name
        )));
    }
    Ok(provider_key.secret.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_core::resource::ResourceEntry;

    fn snapshot_with(provider_key_id: &str) -> AisixSnapshot {
        let snap = AisixSnapshot::new();
        let pk: ProviderKey = serde_json::from_str(
            r#"{"display_name":"openai-prod","secret":"sk-x","api_base":"https://proxy.example.com/v1"}"#,
        )
        .unwrap();
        snap.provider_keys
            .insert(ResourceEntry::new(provider_key_id, pk, 1));
        snap
    }

    fn direct_model(provider_key_id: &str) -> Model {
        let cfg = format!(
            r#"{{
                "display_name": "my-gpt4",
                "provider": "openai",
                "model_name": "gpt-4o",
                "provider_key_id": "{provider_key_id}"
            }}"#
        );
        serde_json::from_str(&cfg).unwrap()
    }

    fn routing_model() -> Model {
        serde_json::from_str(
            r#"{
                "display_name": "router-1",
                "routing": {
                    "strategy": "round_robin",
                    "targets": [{"model": "my-gpt4"}]
                }
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn resolve_provider_key_happy_path() {
        let snap = snapshot_with("pk-1");
        let m = direct_model("pk-1");
        let entry = resolve_provider_key(&snap, &m).unwrap();
        assert_eq!(entry.value.display_name, "openai-prod");
    }

    #[test]
    fn resolve_provider_key_unknown_id_is_400_with_helpful_message() {
        let snap = snapshot_with("pk-1");
        let m = direct_model("pk-MISSING");
        let err = resolve_provider_key(&snap, &m).unwrap_err();
        match err {
            ProxyError::InvalidRequest(msg) => {
                assert!(msg.contains("provider_key_id"), "{msg}");
                assert!(msg.contains("my-gpt4"), "{msg}");
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn resolve_provider_key_routing_model_is_400() {
        let snap = snapshot_with("pk-1");
        let m = routing_model();
        let err = resolve_provider_key(&snap, &m).unwrap_err();
        assert!(matches!(err, ProxyError::InvalidRequest(_)));
    }

    #[test]
    fn require_provider_returns_provider_for_direct_model() {
        let m = direct_model("pk-1");
        assert_eq!(require_provider(&m).unwrap(), Provider::Openai);
    }

    #[test]
    fn require_provider_rejects_routing_model() {
        let m = routing_model();
        assert!(require_provider(&m).is_err());
    }

    #[test]
    fn resolve_base_url_uses_override_when_set() {
        let snap = snapshot_with("pk-1");
        let m = direct_model("pk-1");
        let pk_entry = resolve_provider_key(&snap, &m).unwrap();
        let base = resolve_base_url(Provider::Openai, &pk_entry.value);
        assert_eq!(base, "https://proxy.example.com/v1");
    }

    #[test]
    fn resolve_base_url_falls_back_to_provider_default_when_override_blank() {
        let pk: ProviderKey = serde_json::from_str(r#"{"display_name":"x","secret":"k"}"#).unwrap();
        let base = resolve_base_url(Provider::Anthropic, &pk);
        assert_eq!(base, Provider::Anthropic.default_base_url());
    }
}
