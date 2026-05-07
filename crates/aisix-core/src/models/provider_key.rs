//! `ProviderKey` entity — managed upstream provider credential.
//!
//! A ProviderKey lets operators store an upstream provider's API key
//! (OpenAI, Anthropic, Gemini, DeepSeek, …) once and have many Models
//! reference it by id (`provider_key_id`). Rotating the secret then
//! becomes a single PUT against the ProviderKey rather than rewriting
//! every Model that uses it.
//!
//! Naming intentionally aligns with the AISIX-Cloud control plane's
//! `ProviderKey` table — same concept, same name. The standalone
//! Admin API and the SaaS-tier dashboard exposition stay in lockstep.
//!
//! etcd path: `{prefix}/provider_keys/{uuid}`. Secondary index on
//! `display_name`.

use serde::{Deserialize, Serialize};

use crate::resource::Resource;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ProviderKey {
    /// Operator-facing label, unique within the gateway. Surfaces in
    /// the Admin API list view and in dashboard UIs that wrap this
    /// resource.
    pub display_name: String,

    /// Upstream provider's API key, stored in plaintext on the
    /// standalone path (the etcd channel is mTLS-only — same trust
    /// boundary as Guardrail credentials and ObservabilityExporter
    /// headers). On the AISIX-Cloud path cp-api decrypts the
    /// envelope-encrypted secret at projection time and writes the
    /// plaintext here.
    pub secret: String,

    /// Override for the upstream base URL. Empty/None means the
    /// provider default applies (see `Provider::default_base_url`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_base: Option<String>,

    /// Filled in by the snapshot loader from the etcd key path.
    #[serde(skip)]
    pub(crate) runtime_id: String,
}

impl Resource for ProviderKey {
    fn id(&self) -> &str {
        &self.runtime_id
    }

    fn name(&self) -> &str {
        &self.display_name
    }

    fn kind() -> &'static str {
        "provider_keys"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialises_minimal_provider_key() {
        let p: ProviderKey =
            serde_json::from_str(r#"{"display_name":"openai-prod","secret":"sk-prod-xxxx"}"#)
                .unwrap();
        assert_eq!(p.display_name, "openai-prod");
        assert_eq!(p.secret, "sk-prod-xxxx");
        assert!(p.api_base.is_none());
    }

    #[test]
    fn deserialises_with_api_base() {
        let p: ProviderKey = serde_json::from_str(
            r#"{"display_name":"openai-proxy","secret":"sk-x","api_base":"https://proxy.example.com/v1"}"#,
        )
        .unwrap();
        assert_eq!(p.api_base.as_deref(), Some("https://proxy.example.com/v1"));
    }

    #[test]
    fn rejects_unknown_fields() {
        let r: Result<ProviderKey, _> =
            serde_json::from_str(r#"{"display_name":"x","secret":"k","extra":1}"#);
        assert!(r.is_err());
    }

    #[test]
    fn resource_trait_routes_through_display_name() {
        let mut p: ProviderKey =
            serde_json::from_str(r#"{"display_name":"openai-prod","secret":"sk-x"}"#).unwrap();
        p.runtime_id = "uuid-pk-1".into();
        assert_eq!(<ProviderKey as Resource>::kind(), "provider_keys");
        assert_eq!(p.id(), "uuid-pk-1");
        assert_eq!(p.name(), "openai-prod");
    }
}
