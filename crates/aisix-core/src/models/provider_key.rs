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

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::models::Adapter;
use crate::resource::Resource;

// `PartialEq` (not `Eq`) on `ProviderKey` because `RequestOverrides`
// carries `f64` (in `ParamConstraints`) and `serde_json::Value` (in
// `default_body_fields`), neither of which can implement `Eq` due to
// NaN / Number-equality semantics. Tests compare via `assert_eq!`
// which only needs `PartialEq`.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq)]
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

    /// Override for the upstream base URL. Empty/None is rejected by
    /// the family bridges for any vendor whose identity isn't
    /// `"openai"` — see `OpenAiBridge::resolve_base` for the safety
    /// guard that prevents a missing api_base from silently routing
    /// non-OpenAI traffic to `api.openai.com`. cp-api populates this
    /// from `adapter_map.yaml`'s `default_base_url` / `provider_metadata.api_base_url`.
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

    /// Per-key request-shape overrides — see issue #302 §5
    /// `RuntimeConfig.request`. `None` until cp-api ships the block.
    /// No dispatch path reads it in this PR; #301 already provides
    /// the primitive apply functions in `aisix-provider-openai` that
    /// Phase D will call once the wire stage cuts over.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request: Option<RequestOverrides>,

    /// Per-key response-shape overrides — see issue #302 §5
    /// `RuntimeConfig.response`. `None` until cp-api ships the block.
    /// Same Phase D wiring story as [`Self::request`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response: Option<ResponseOverrides>,

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
#[derive(Debug, Clone, Default, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
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

/// Per-`ProviderKey` request-shape overrides — see issue #302 §5
/// `RuntimeConfig.request`. Each field maps 1:1 onto a primitive
/// apply function in [`aisix-provider-openai`'s `overrides`
/// module](https://github.com/api7/ai-gateway/blob/main/crates/aisix-provider-openai/src/overrides.rs):
///
/// - `param_renames` → `apply_param_renames`
/// - `param_constraints` → `apply_param_constraints`
/// - `default_headers` → `apply_default_headers`
/// - `default_body_fields` → `apply_default_body_fields`
///
/// `f64` in [`ParamConstraints`] is the reason the parent
/// [`ProviderKey`] derives `PartialEq` rather than `Eq`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, schemars::JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RequestOverrides {
    /// `apply_param_renames` input. Top-level body keys named on the
    /// left are renamed to the right. Empty map is the default.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub param_renames: HashMap<String, String>,

    /// `apply_param_constraints` input. `None` means no clamping.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub param_constraints: Option<ParamConstraints>,

    /// `apply_default_headers` input. Top-level headers added to the
    /// outbound request when the caller did not set them. Reserved
    /// auth headers are dropped by `apply_default_headers` as
    /// defense-in-depth.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub default_headers: HashMap<String, String>,

    /// `apply_default_body_fields` input. Top-level body fields added
    /// when the caller did not set them. `serde_json::Map` preserves
    /// insertion order on serialize, matching the etcd round-trip.
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub default_body_fields: Map<String, Value>,
}

/// Numeric range clamps applied to chat-completion request bodies —
/// the on-disk shape of issue #302 §5 `param_constraints`. Phase A
/// scope is `temperature` only; `top_p` / `frequency_penalty` are
/// deferred until a real upstream quirk demands them (YAGNI per
/// `CLAUDE.md` §2).
///
/// `f64` not `Eq`: NaN comparisons make a derived `Eq` unsound.
/// [`PartialEq`] is enough for the round-trip test.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, schemars::JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ParamConstraints {
    /// Upper bound for `temperature`. Values above this are clamped
    /// to this value. `None` means "no upper clamp".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature_max: Option<f64>,

    /// Lower bound for `temperature`. Values below this are clamped
    /// to this value. `None` means "no lower clamp".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature_min: Option<f64>,
}

/// Per-`ProviderKey` response-shape overrides — see issue #302 §5
/// `RuntimeConfig.response`. Each field maps onto behavior the
/// [`aisix-provider-openai`'s `overrides`
/// module](https://github.com/api7/ai-gateway/blob/main/crates/aisix-provider-openai/src/overrides.rs)
/// already implements:
///
/// - `stream_done_marker` → `apply_stream_done_marker_policy`
/// - `content_list_to_string` → `apply_content_list_to_string`
///   (applied to the *request* body before send when the upstream
///   only accepts string content)
/// - `reasoning_field` → `extract_reasoning_field`
///
/// `error_envelope` is on-disk only — issue #302 §5 keeps it as a
/// `"openai" | "passthrough"` string so cp-api can iterate without
/// a Rust-side enum migration. Phase D pins the closed set.
#[derive(Debug, Clone, Default, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ResponseOverrides {
    /// Stream `[DONE]` terminator expectation. `None` means "no
    /// opinion" — same effect as [`StreamDoneMarker::Optional`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_done_marker: Option<StreamDoneMarker>,

    /// When `true`, the request-body `messages[*].content` array of
    /// text blocks gets flattened to a single string before dispatch.
    /// Defaults to `false` (no flattening).
    #[serde(default)]
    pub content_list_to_string: bool,

    /// On-disk discriminator for the error-translation strategy.
    /// `"openai"` projects upstream errors into the OpenAI envelope;
    /// `"passthrough"` returns the upstream body as-is. Open string
    /// in this PR (issue #302 §5 wire shape); Phase D pins the
    /// closed set in a follow-up.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_envelope: Option<String>,

    /// `extract_reasoning_field` path. Empty / `None` means no lift.
    /// Example: `"delta.reasoning_content"` (DeepSeek's canonical
    /// shape, already aligned with the gateway's emit slot).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_field: Option<String>,
}

/// Stream `[DONE]` terminator policy for an SSE response — the
/// on-disk shape of issue #302 §5 `stream_done_marker`. The wire
/// form is the lowercased variant name (`"required"` / `"optional"`
/// / `"none"`) so cp-api JSON keeps the same set the original spec
/// drafted.
///
/// The runtime apply function lives in `aisix-provider-openai`
/// (`apply_stream_done_marker_policy`) and consumes this enum
/// directly via re-export from `aisix-core`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum StreamDoneMarker {
    /// Upstream must emit `data: [DONE]`. Absence is a wire-shape
    /// violation. OpenAI proper, DeepSeek, Groq.
    Required,
    /// Either presence or absence is acceptable. Used when the
    /// upstream is OpenAI-compat but does not promise the terminator.
    Optional,
    /// Upstream is expected to *omit* the marker. Some Azure / Vertex
    /// flavors terminate cleanly on connection close.
    None,
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
        // and `adapter` / `request` / `response` absent (skipped
        // because None). Re-deserializing must reproduce the original
        // struct.
        let original = ProviderKey {
            display_name: "openai-prod".into(),
            secret: "sk-x".into(),
            api_base: None,
            provider: String::new(),
            adapter: None,
            telemetry_tags: TelemetryTags::default(),
            request: None,
            response: None,
            runtime_id: String::new(),
        };
        let s = serde_json::to_string(&original).unwrap();
        let back: ProviderKey = serde_json::from_str(&s).unwrap();
        assert_eq!(original, back);
    }

    // ---- issue #302 Phase A2.5: ProviderKey.request / .response ----

    #[test]
    fn legacy_payload_without_request_response_blocks_deserialises_to_none() {
        // Backward-compat: an existing on-disk payload that pre-dates
        // the Phase A2.5 PR must still deserialize, and `request` /
        // `response` must land at `None`.
        let p: ProviderKey =
            serde_json::from_str(r#"{"display_name":"openai-prod","secret":"sk-x"}"#).unwrap();
        assert!(p.request.is_none());
        assert!(p.response.is_none());
    }

    #[test]
    fn request_overrides_empty_object_deserialises_to_defaults() {
        // `{"request": {}}` must succeed and yield an all-default
        // RequestOverrides — empty maps, no constraints.
        let p: ProviderKey =
            serde_json::from_str(r#"{"display_name":"x","secret":"k","request":{}}"#).unwrap();
        let req = p.request.expect("request was Some");
        assert!(req.param_renames.is_empty());
        assert!(req.param_constraints.is_none());
        assert!(req.default_headers.is_empty());
        assert!(req.default_body_fields.is_empty());
    }

    #[test]
    fn request_overrides_full_payload_deserialises() {
        // Mirror the on-disk example in issue #302 §5 exactly.
        let p: ProviderKey = serde_json::from_str(
            r#"{
                "display_name": "deepseek-prod",
                "secret": "sk-x",
                "request": {
                    "param_renames":      { "max_completion_tokens": "max_tokens" },
                    "param_constraints":  { "temperature_max": 1.0 },
                    "default_headers":    { "X-Foo": "bar" },
                    "default_body_fields": { "safe_prompt": true }
                }
            }"#,
        )
        .unwrap();
        let req = p.request.expect("request was Some");
        assert_eq!(
            req.param_renames.get("max_completion_tokens"),
            Some(&"max_tokens".to_string())
        );
        let constraints = req.param_constraints.expect("param_constraints was Some");
        assert_eq!(constraints.temperature_max, Some(1.0));
        assert_eq!(constraints.temperature_min, None);
        assert_eq!(req.default_headers.get("X-Foo"), Some(&"bar".to_string()));
        assert_eq!(
            req.default_body_fields.get("safe_prompt"),
            Some(&serde_json::Value::Bool(true))
        );
    }

    #[test]
    fn request_overrides_rejects_unknown_field() {
        // deny_unknown_fields on RequestOverrides stops a typo in
        // cp-api JSON from silently no-oping the apply call.
        let r: Result<ProviderKey, _> = serde_json::from_str(
            r#"{
                "display_name": "x",
                "secret": "k",
                "request": { "param_rename": {} }
            }"#,
        );
        assert!(r.is_err());
    }

    #[test]
    fn response_overrides_empty_object_deserialises_to_defaults() {
        let p: ProviderKey =
            serde_json::from_str(r#"{"display_name":"x","secret":"k","response":{}}"#).unwrap();
        let resp = p.response.expect("response was Some");
        assert!(resp.stream_done_marker.is_none());
        assert!(!resp.content_list_to_string);
        assert!(resp.error_envelope.is_none());
        assert!(resp.reasoning_field.is_none());
    }

    #[test]
    fn response_overrides_full_payload_deserialises() {
        // Mirror the on-disk example in issue #302 §5 exactly.
        let p: ProviderKey = serde_json::from_str(
            r#"{
                "display_name": "deepseek-prod",
                "secret": "sk-x",
                "response": {
                    "stream_done_marker":     "required",
                    "content_list_to_string": false,
                    "error_envelope":         "openai",
                    "reasoning_field":        "delta.reasoning_content"
                }
            }"#,
        )
        .unwrap();
        let resp = p.response.expect("response was Some");
        assert_eq!(resp.stream_done_marker, Some(StreamDoneMarker::Required));
        assert!(!resp.content_list_to_string);
        assert_eq!(resp.error_envelope.as_deref(), Some("openai"));
        assert_eq!(
            resp.reasoning_field.as_deref(),
            Some("delta.reasoning_content")
        );
    }

    #[test]
    fn response_overrides_rejects_unknown_field() {
        let r: Result<ProviderKey, _> = serde_json::from_str(
            r#"{
                "display_name": "x",
                "secret": "k",
                "response": { "reasoning_fields": "delta.foo" }
            }"#,
        );
        assert!(r.is_err());
    }

    #[test]
    fn stream_done_marker_deserialises_all_three_variants() {
        // The on-disk wire form is the lowercased variant — verify
        // every literal the cp-api spec promises.
        for (raw, expected) in [
            ("required", StreamDoneMarker::Required),
            ("optional", StreamDoneMarker::Optional),
            ("none", StreamDoneMarker::None),
        ] {
            let resp: ResponseOverrides =
                serde_json::from_str(&format!(r#"{{"stream_done_marker":"{raw}"}}"#)).unwrap();
            assert_eq!(resp.stream_done_marker, Some(expected));
        }
    }

    #[test]
    fn stream_done_marker_rejects_unknown_variant() {
        // Closed enum — uppercase or unknown variants must fail loudly.
        let r: Result<ResponseOverrides, _> =
            serde_json::from_str(r#"{"stream_done_marker":"Required"}"#);
        assert!(r.is_err());

        let r: Result<ResponseOverrides, _> =
            serde_json::from_str(r#"{"stream_done_marker":"maybe"}"#);
        assert!(r.is_err());
    }

    #[test]
    fn param_constraints_round_trips() {
        // Both clamps set → both come back identical after a
        // JSON round-trip. f64 equality holds for finite values.
        let original = ParamConstraints {
            temperature_max: Some(1.0),
            temperature_min: Some(0.0),
        };
        let s = serde_json::to_string(&original).unwrap();
        let back: ParamConstraints = serde_json::from_str(&s).unwrap();
        assert_eq!(back.temperature_max, Some(1.0));
        assert_eq!(back.temperature_min, Some(0.0));
    }

    #[test]
    fn param_constraints_rejects_unknown_field() {
        let r: Result<ParamConstraints, _> = serde_json::from_str(r#"{"top_p_max": 0.9}"#);
        assert!(r.is_err());
    }
}
