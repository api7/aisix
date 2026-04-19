//! `Credential` entity — managed upstream secret.
//!
//! A Credential lets operators store an upstream `api_key` once and
//! have many Models reference it by name (`credential_ref`). Rotating
//! the secret then becomes a single PUT against the Credential rather
//! than rewriting every Model that uses it.
//!
//! etcd path: `{prefix}/credentials/{uuid}`. Secondary index on `name`.

use serde::{Deserialize, Serialize};

use crate::resource::Resource;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Credential {
    pub name: String,
    pub api_key: String,
    /// Override for the upstream base URL. Same semantics as
    /// `Model::provider_config.api_base` — empty/None means the
    /// provider default applies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_base: Option<String>,

    /// Filled in by the snapshot loader from the etcd key path.
    #[serde(skip)]
    pub(crate) runtime_id: String,
}

impl Resource for Credential {
    fn id(&self) -> &str {
        &self.runtime_id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn kind() -> &'static str {
        "credentials"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialises_minimal_credential() {
        let c: Credential = serde_json::from_str(
            r#"{"name":"openai-prod","api_key":"sk-prod-xxxx"}"#,
        )
        .unwrap();
        assert_eq!(c.name, "openai-prod");
        assert_eq!(c.api_key, "sk-prod-xxxx");
        assert!(c.api_base.is_none());
    }

    #[test]
    fn deserialises_credential_with_api_base() {
        let c: Credential = serde_json::from_str(
            r#"{"name":"x","api_key":"k","api_base":"https://proxy.local/v1"}"#,
        )
        .unwrap();
        assert_eq!(c.api_base.as_deref(), Some("https://proxy.local/v1"));
    }

    #[test]
    fn rejects_unknown_top_level_fields() {
        let r: Result<Credential, _> =
            serde_json::from_str(r#"{"name":"x","api_key":"k","extra":1}"#);
        assert!(r.is_err());
    }

    #[test]
    fn resource_trait_uses_name_and_credentials_kind() {
        let mut c: Credential =
            serde_json::from_str(r#"{"name":"openai-prod","api_key":"k"}"#).unwrap();
        c.runtime_id = "uuid-cred".into();
        assert_eq!(<Credential as Resource>::kind(), "credentials");
        assert_eq!(c.id(), "uuid-cred");
        assert_eq!(c.name(), "openai-prod");
    }
}
