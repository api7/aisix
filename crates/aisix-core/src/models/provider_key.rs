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

use crate::models::Adapter;
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

    /// Vendor identity (e.g. `"deepseek"`, `"openai"`). Introduced as a
    /// skeleton for issue #302 Phase A. Empty in this PR — no dispatch
    /// path consumes it yet; the field exists so future Phase A sub-PRs
    /// can populate it without an on-disk schema break. Old payloads
    /// that omit `provider` continue to deserialize via
    /// `#[serde(default)]`.
    #[serde(default)]
    pub provider: String,

    /// Wire-shape adapter (`openai` / `anthropic` / `bedrock` /
    /// `vertex` / `azure-openai`). Introduced as a skeleton for issue
    /// #302 Phase A. `None` in this PR — no dispatch path consumes it
    /// yet; the field exists so future Phase A sub-PRs can populate it
    /// without an on-disk schema break. Old payloads that omit
    /// `adapter` continue to deserialize via `#[serde(default)]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter: Option<Adapter>,

    /// Telemetry tags carried alongside the key for metric/log
    /// emission. Introduced as a skeleton for issue #302 Phase A.
    /// No metric path consumes these tags yet; the field exists so
    /// future Phase A sub-PRs can attribute traffic without an
    /// on-disk schema break. Old payloads that omit `telemetry_tags`
    /// fall back to the `Default` impl via `#[serde(default)]`.
    #[serde(default)]
    pub telemetry_tags: TelemetryTags,

    /// Filled in by the snapshot loader from the etcd key path.
    #[serde(skip)]
    pub(crate) runtime_id: String,
}

/// Telemetry attribution tags emitted alongside requests routed
/// through this `ProviderKey`. Introduced as a skeleton for issue
/// #302 Phase A — no metric/log path consumes these fields yet.
///
/// The `#[serde(default)]` on each field plus `#[derive(Default)]`
/// means an omitted block or omitted individual key both yield the
/// zero-value `TelemetryTags`, preserving backward compatibility
/// with existing `ProviderKey` payloads.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TelemetryTags {
    /// `"catalog"` for first-party curated providers, `"byo"` for
    /// bring-your-own. `None` until Phase A wires attribution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,

    /// Whether this provider is surfaced in the featured list.
    /// Defaults to `false`.
    #[serde(default)]
    pub featured: bool,

    /// Branded provider slug for catalog entries (e.g. `"openai"`,
    /// `"anthropic"`). `None` for byo or until Phase A wires
    /// attribution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub branded_provider: Option<String>,

    /// Operator-defined label for this provider key (e.g.
    /// `"production"`, `"shared-test"`). `None` until Phase A wires
    /// attribution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pk_label: Option<String>,

    /// Operator-defined label for bring-your-own entries (e.g. an
    /// internal team name). `None` for catalog entries or until
    /// Phase A wires attribution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub byo_label: Option<String>,
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

    // ---- issue #302 Phase A skeleton ----

    #[test]
    fn legacy_payload_without_phase_a_fields_deserialises_with_defaults() {
        // Wire-shape proof for the on-disk compatibility contract: an
        // existing payload from before Phase A (no `provider`, no
        // `adapter`, no `telemetry_tags`) must still deserialize, and
        // the new fields must land at their zero values.
        let p: ProviderKey = serde_json::from_str(
            r#"{"display_name":"openai-prod","secret":"sk-x","api_base":"https://api.openai.com/v1"}"#,
        )
        .unwrap();
        assert_eq!(p.provider, "");
        assert_eq!(p.adapter, None);
        assert_eq!(p.telemetry_tags, TelemetryTags::default());
    }

    #[test]
    fn payload_with_all_phase_a_fields_deserialises() {
        let p: ProviderKey = serde_json::from_str(
            r#"{
                "display_name": "deepseek-prod",
                "secret": "sk-x",
                "api_base": "https://api.deepseek.com/v1",
                "provider": "deepseek",
                "adapter": "openai",
                "telemetry_tags": {
                    "kind": "catalog",
                    "featured": true,
                    "branded_provider": "deepseek",
                    "pk_label": "production"
                }
            }"#,
        )
        .unwrap();
        assert_eq!(p.provider, "deepseek");
        assert_eq!(p.adapter, Some(Adapter::Openai));
        assert_eq!(p.telemetry_tags.kind.as_deref(), Some("catalog"));
        assert!(p.telemetry_tags.featured);
        assert_eq!(
            p.telemetry_tags.branded_provider.as_deref(),
            Some("deepseek")
        );
        assert_eq!(p.telemetry_tags.pk_label.as_deref(), Some("production"));
        assert_eq!(p.telemetry_tags.byo_label, None);
    }

    #[test]
    fn byo_telemetry_shape_deserialises() {
        // BYO entries have null branded_provider and a non-null
        // byo_label — the dual-label shape Phase A introduces.
        let p: ProviderKey = serde_json::from_str(
            r#"{
                "display_name": "internal-llm",
                "secret": "sk-x",
                "telemetry_tags": {
                    "kind": "byo",
                    "branded_provider": null,
                    "byo_label": "platform-team"
                }
            }"#,
        )
        .unwrap();
        assert_eq!(p.telemetry_tags.kind.as_deref(), Some("byo"));
        assert!(!p.telemetry_tags.featured);
        assert_eq!(p.telemetry_tags.branded_provider, None);
        assert_eq!(p.telemetry_tags.byo_label.as_deref(), Some("platform-team"));
    }

    #[test]
    fn telemetry_tags_rejects_unknown_field() {
        // TelemetryTags is `deny_unknown_fields` — stops cp-api from
        // silently shipping a new tag the DP can't see.
        let r: Result<ProviderKey, _> = serde_json::from_str(
            r#"{
                "display_name": "x",
                "secret": "k",
                "telemetry_tags": { "unknown_tag": "v" }
            }"#,
        );
        assert!(r.is_err());
    }

    #[test]
    fn adapter_rejects_unknown_string() {
        // `adapter` is the closed `Adapter` enum — unknown shape
        // strings must fail loudly rather than silently fall through.
        let r: Result<ProviderKey, _> = serde_json::from_str(
            r#"{"display_name":"x","secret":"k","adapter":"not-a-real-adapter"}"#,
        );
        assert!(r.is_err());
    }

    #[test]
    fn round_trip_omits_default_phase_a_fields() {
        // A ProviderKey built without setting the Phase A fields
        // serializes with `provider:""` and `telemetry_tags` defaulted,
        // and `adapter` absent (skipped because None). Re-deserializing
        // must reproduce the original struct.
        let original = ProviderKey {
            display_name: "openai-prod".into(),
            secret: "sk-x".into(),
            api_base: None,
            provider: String::new(),
            adapter: None,
            telemetry_tags: TelemetryTags::default(),
            runtime_id: String::new(),
        };
        let s = serde_json::to_string(&original).unwrap();
        let back: ProviderKey = serde_json::from_str(&s).unwrap();
        assert_eq!(original, back);
    }
}
