//! `ProviderKey` entity — managed upstream provider credential.
//!
//! A ProviderKey lets operators store an upstream provider's API key
//! (OpenAI, Anthropic, Gemini, DeepSeek, …) once and have many Models
//! reference it by id (`provider_key_id`). Rotating the key then
//! becomes a single PUT against the ProviderKey rather than rewriting
//! every Model that uses it.
//!
//! Naming intentionally aligns with the AISIX-Cloud control plane's
//! `ProviderKey` table — same concept, same name. The self-hosted
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
pub struct ProviderKey {
    /// Operator-facing label, unique within the gateway. Surfaces in
    /// the Admin API list view and in dashboard UIs that wrap this
    /// resource.
    #[schemars(length(min = 1))]
    pub display_name: String,

    /// Upstream provider's API key. The gateway receives plaintext so it
    /// can authenticate to the upstream provider. Protect the configuration
    /// store and transport accordingly.
    // `secret` is the field's former name; stored documents and callers
    // that still use it keep deserializing (schema-side acceptance lives
    // in `schema::provider_key_root_schema`). Re-serialization always
    // emits `api_key`.
    #[serde(alias = "secret")]
    #[schemars(length(min = 1))]
    pub api_key: String,

    /// Override base URL for the upstream provider. Required for custom or OpenAI-compatible providers that should not use a built-in vendor endpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_base: Option<String>,

    /// Upstream provider identifier, such as `"deepseek"`, `"openai"`, or a model catalog ID. The gateway uses this value for provider-specific dispatch and base URL validation.
    #[serde(default)]
    pub provider: String,

    /// Upstream API protocol family used when provider-specific dispatch is unavailable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter: Option<Adapter>,

    /// API surfaces this upstream serves natively, beyond the one its
    /// `adapter` already implies, and the base URL each one lives at.
    ///
    /// One upstream account often exposes more than one protocol, on
    /// different paths of the same host — an OpenAI-compatible
    /// `/v1/chat/completions` under `…/v1` and an Anthropic-compatible
    /// `/v1/messages` under `…/anthropic`, both authenticated by the same
    /// credential. `api_base` can only name one of them, so without this
    /// field every request the declared path cannot serve gets translated
    /// instead — losing whatever the target protocol carries that the
    /// canonical chat shape does not (prompt-cache breakpoints, thinking
    /// blocks). Declaring the second entry here lets each inbound protocol
    /// reach its own native path under the one credential.
    ///
    /// Each surface resolves on its own terms; see [`ProviderApis`].
    /// Surfaces this map has no key for — embeddings, audio, images,
    /// videos, files/batches/fine-tuning, rerank — always use `api_base`,
    /// exactly as before.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub apis: Option<ProviderApis>,

    /// Telemetry tags carried alongside the key for metric and log emission.
    #[serde(default)]
    pub telemetry_tags: TelemetryTags,

    /// Per-key request-shape overrides applied by supported provider paths before dispatch to the upstream provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request: Option<RequestOverrides>,

    /// Per-key response-shape overrides applied by provider bridges that support response transformation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response: Option<ResponseOverrides>,

    /// Inbound headers removed before passthrough forwarding.
    #[serde(
        default = "default_strip_headers",
        deserialize_with = "deserialize_normalized_strip_headers"
    )]
    pub strip_headers: Vec<String>,

    /// TLS settings for connections to this key's `api_base`. Omit to use the
    /// gateway's deployment-wide trust settings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tls: Option<ProviderKeyTls>,

    /// Filled in by the snapshot loader from the etcd key path.
    #[serde(skip)]
    pub(crate) runtime_id: String,
}

/// The API surfaces a Provider Key can declare in [`ProviderKey::apis`].
///
/// Only surfaces whose native path the gateway can choose *instead of*
/// translating belong here — declaring one is a statement about which of
/// the two it takes. `/v1/chat/completions` is deliberately absent: every
/// path reaches it through a provider bridge at `api_base`, so there is
/// no choice to declare.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ApiSurface {
    /// OpenAI-wire `/v1/responses`.
    Responses,
    /// Anthropic-wire `/v1/messages`, and its `/count_tokens` sub-route.
    Messages,
}

/// Per-surface native entry points on one upstream.
///
/// The two surfaces resolve differently, because the evidence for them
/// differs:
///
/// - `messages` is **additive**. An `adapter: anthropic` key (or the
///   `anthropic` vendor) speaks that wire by declaration and keeps serving
///   `/v1/messages` natively whatever this map says; listing it here adds
///   the route to a key whose adapter is something else. That is the
///   DeepSeek/Zhipu shape — an OpenAI-compatible key whose vendor also
///   fronts an Anthropic-compatible path. An entry's own `base` always
///   decides WHERE the Anthropic wire lives, for the verbatim
///   passthrough and for the bridge that translates into it alike;
///   without one it is `api_base`.
/// - `responses` is **authoritative**. Once this map exists, `/v1/responses`
///   is served natively only if it is listed. The Responses API is a strict
///   superset of chat completions rather than a rename, so an
///   OpenAI-compatible endpoint serving one does not necessarily serve the
///   other, and `adapter: openai` is not evidence either way. Leaving it
///   out is how an operator says "this endpoint has no `/v1/responses`" and
///   gets the request translated to chat completions instead of 404'd
///   upstream.
///
/// With no map at all, both fall back to the vendor-id inference the
/// gateway used before this field existed.
#[derive(Debug, Clone, Default, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
pub struct ProviderApis {
    /// OpenAI-wire `/v1/responses`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub responses: Option<ApiEndpoint>,

    /// Anthropic-wire `/v1/messages` (and `/v1/messages/count_tokens`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub messages: Option<ApiEndpoint>,
}

impl ProviderApis {
    /// The declared entry for `surface`, if any.
    pub fn get(&self, surface: ApiSurface) -> Option<&ApiEndpoint> {
        match surface {
            ApiSurface::Responses => self.responses.as_ref(),
            ApiSurface::Messages => self.messages.as_ref(),
        }
    }
}

/// One declared entry point.
#[derive(Debug, Clone, Default, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
pub struct ApiEndpoint {
    /// Base URL this surface is served at. Omit when it is the same one
    /// `api_base` names — an entry with no `base` still carries the
    /// declaration that the surface exists.
    ///
    /// Deliberately NOT length-constrained, matching `api_base`. The
    /// lenient read schema keeps every constraint but the open-object
    /// one, so a `minLength` here would make an empty string skip the
    /// whole Provider Key row — and with it every model that references
    /// the key — where the same empty string on `api_base` is the
    /// control plane's documented way to clear an override. An empty
    /// value is treated as "no override" at resolution time instead.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base: Option<String>,
}

/// TLS settings for connections to one Provider Key's `api_base`.
///
/// Use this when a single upstream endpoint needs trust settings that
/// differ from the gateway's deployment-wide ones — typically a
/// self-hosted model endpoint whose certificate is signed by a private
/// certificate authority.
///
/// The certificate is supplied inline rather than as a file path, because
/// the endpoint is declared here rather than in the gateway's own
/// configuration file. For a certificate authority that applies to every
/// upstream, prefer the gateway's `upstream.tls.ca_file` setting.
// `Default` is written out rather than derived: a derived one would make
// `verify` false, so a `tls: {}` block — or any future code path that
// reaches for the default — would silently stop checking certificates.
// `Hash` so the data plane can key its per-key client cache on the
// settings themselves, sharing one connection pool across every Provider
// Key configured the same way.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq, Hash)]
pub struct ProviderKeyTls {
    /// PEM-encoded certificate authority certificates trusted as issuers for
    /// this endpoint, in addition to the gateway's default trust store. A
    /// bundle containing several certificates is accepted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ca_cert: Option<String>,

    /// Whether the endpoint's certificate is verified. Setting this to `false`
    /// accepts any certificate, including one presented by an intercepting
    /// party, and is intended only for test environments.
    #[serde(default = "default_verify")]
    pub verify: bool,
}

fn default_verify() -> bool {
    true
}

impl Default for ProviderKeyTls {
    fn default() -> Self {
        Self {
            ca_cert: None,
            verify: true,
        }
    }
}

impl ProviderKeyTls {
    /// Whether this leaves the connection exactly as the deployment-wide
    /// settings would build it, so the shared client can be reused.
    pub fn is_noop(&self) -> bool {
        self.ca_cert.as_ref().is_none_or(|p| p.trim().is_empty()) && self.verify
    }
}

/// Default header-strip list for a freshly-created ProviderKey
/// on the passthrough endpoint, per issue #411. These four headers
/// are credentials that the upstream LLM provider has no legitimate
/// use for. Stripping by default protects against accidental
/// session-token disclosure. Customers can remove entries via the
/// dashboard (with a warning) if they have a specific audit /
/// forwarding need.
pub fn default_strip_headers() -> Vec<String> {
    vec![
        "authorization".to_string(),
        "cookie".to_string(),
        "set-cookie".to_string(),
        "x-api-key".to_string(),
    ]
}

/// Normalize a single strip-list entry: trim whitespace, lowercase
/// ASCII. Returns `None` for entries that, post-trim, are empty or
/// reference-invalid HTTP header names. Non-ASCII chars survive
/// `to_ascii_lowercase` (no-op for them) but are unusual in practice.
/// the passthrough handler's `to_ascii_lowercase` comparison will
/// still match correctly.
fn normalize_strip_entry(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed.to_ascii_lowercase())
}

/// Deserialize + normalize: drop empties, lowercase, dedup. Preserves
/// first-occurrence order so a hand-curated list reads sanely in the
/// dashboard. Per issue #411 audit MEDIUM-1.
fn deserialize_normalized_strip_headers<'de, D>(de: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize as _;
    let raw: Vec<String> = Vec::deserialize(de)?;
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(raw.len());
    for entry in raw {
        if let Some(normalized) = normalize_strip_entry(&entry) {
            if seen.insert(normalized.clone()) {
                out.push(normalized);
            }
        }
    }
    Ok(out)
}

/// Provider-key category: `catalog` for curated providers, `byo` for
/// bring-your-own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum TelemetryKind {
    Catalog,
    Byo,
}

impl TelemetryKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Catalog => "catalog",
            Self::Byo => "byo",
        }
    }
}

/// Telemetry attribution tags emitted with requests routed through this provider key.
#[derive(Debug, Clone, Default, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
pub struct TelemetryTags {
    /// Provider-key category, such as `"catalog"` for curated providers or
    /// `"byo"` for bring-your-own providers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<TelemetryKind>,

    /// Whether this provider is surfaced in the featured list.
    #[serde(default)]
    pub featured: bool,

    /// Branded provider slug for catalog entries, such as `"openai"` or
    /// `"anthropic"`. Bring-your-own providers leave this field unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub branded_provider: Option<String>,

    /// Operator-defined label for this provider key, such as `"production"` or
    /// `"shared-test"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pk_label: Option<String>,

    /// Operator-defined label for bring-your-own entries, such as an internal
    /// team name. Catalog entries leave this field unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub byo_label: Option<String>,
}

/// Per-`ProviderKey` request-shape overrides. Use these fields to rename
/// request body parameters, clamp supported numeric parameters, add fallback
/// outbound headers, or add fallback outbound body fields.
#[derive(Debug, Clone, Default, Serialize, Deserialize, schemars::JsonSchema, PartialEq)]
pub struct RequestOverrides {
    /// `apply_param_renames` input. Top-level body keys named on the left are renamed to the right. Leave empty to preserve request parameter names.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub param_renames: HashMap<String, String>,

    /// Parameter constraints applied to the outbound request. If omitted,
    /// no clamping is applied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub param_constraints: Option<ParamConstraints>,

    /// Top-level headers added to the outbound request when the caller did
    /// not set them. Values may reference the request context with `${...}`
    /// variables, such as `"${request.api_key.team_id}"`; a header whose
    /// variables do not all resolve is dropped rather than sent blank. See
    /// [`crate::header_template`] for the closed variable vocabulary.
    /// Reserved auth headers are dropped as defense-in-depth.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub default_headers: HashMap<String, String>,

    /// Inbound client headers forwarded to the upstream provider, as
    /// single-`*` glob patterns matched case-insensitively against the
    /// header name (`"anthropic-beta"`, `"x-trace-*"`). Empty — the
    /// default — forwards nothing, which is the behavior of every
    /// standard-protocol endpoint before AISIX-Cloud#1167. Auth,
    /// transport, and gateway-owned headers are never forwarded whatever
    /// the patterns say.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub forward_client_headers: Vec<String>,

    /// `apply_default_body_fields` input. Top-level body fields added
    /// when the caller did not set them. `serde_json::Map` preserves
    /// insertion order on serialize, matching the etcd round-trip.
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub default_body_fields: Map<String, Value>,
}

/// Numeric range clamps applied to chat-completion request bodies.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, schemars::JsonSchema, PartialEq)]
pub struct ParamConstraints {
    /// Upper bound for `temperature`. Values above this are clamped
    /// to this value. If omitted, no upper bound is applied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature_max: Option<f64>,

    /// Lower bound for `temperature`. Values below this are clamped
    /// to this value. If omitted, no lower bound is applied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature_min: Option<f64>,
}

/// Per-`ProviderKey` response-shape overrides. Use these fields to describe
/// stream termination behavior, flatten list-style content when needed, select
/// an error envelope strategy, or lift provider-specific reasoning content.
#[derive(Debug, Clone, Default, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
pub struct ResponseOverrides {
    /// Stream `[DONE]` terminator expectation. If omitted, either presence
    /// or absence of the terminator is accepted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_done_marker: Option<StreamDoneMarker>,

    /// When `true`, the request-body `messages[*].content` array of text blocks gets flattened to a single string before dispatch.
    #[serde(default)]
    pub content_list_to_string: bool,

    /// Stored error-envelope preference for compatibility with control-plane
    /// configuration. The proxy does not currently apply this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_envelope: Option<String>,

    /// Path used to extract reasoning content from the provider response.
    /// If omitted or empty, no reasoning field is lifted. Example:
    /// `"delta.reasoning_content"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_field: Option<String>,
}

/// Stream `[DONE]` terminator policy for an SSE response. Values are `"required"`, `"optional"`, or `"none"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum StreamDoneMarker {
    /// Upstream is expected to emit `data: [DONE]`. Absence is logged as a diagnostic warning.
    Required,
    /// Either presence or absence is acceptable. Used when the
    /// upstream is OpenAI-compatible but does not require the terminator.
    Optional,
    /// Upstream is expected to omit the marker and terminate on connection close.
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
        assert_eq!(p.api_key, "sk-prod-xxxx");
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
    fn tolerates_unknown_fields_for_forward_compat() {
        // cp-api may ship new fields ahead of the DP rolling out; serde
        // must accept them. The write path still rejects them via the
        // strict schema validator (validate_provider_key in
        // models/schema.rs).
        let p: ProviderKey =
            serde_json::from_str(r#"{"display_name":"x","secret":"k","extra":1}"#).unwrap();
        assert_eq!(p.display_name, "x");
    }

    // ---- `secret` → `api_key` rename ----

    #[test]
    fn accepts_canonical_api_key_spelling() {
        let p: ProviderKey =
            serde_json::from_str(r#"{"display_name":"openai-prod","api_key":"sk-prod-xxxx"}"#)
                .unwrap();
        assert_eq!(p.api_key, "sk-prod-xxxx");
    }

    #[test]
    fn legacy_secret_spelling_still_deserialises() {
        // Stored documents written before the rename spell the field
        // `secret`; the serde alias must keep loading them. (Most other
        // fixtures in this module double as coverage for this, but pin
        // it explicitly so the intent survives fixture migrations.)
        let p: ProviderKey =
            serde_json::from_str(r#"{"display_name":"openai-prod","secret":"sk-legacy"}"#).unwrap();
        assert_eq!(p.api_key, "sk-legacy");
    }

    #[test]
    fn serialises_credential_under_api_key_only() {
        // Emission contract: re-serialization (admin GET responses,
        // admin-written documents) uses the canonical name, never the
        // former spelling.
        let p: ProviderKey =
            serde_json::from_str(r#"{"display_name":"x","secret":"sk-x"}"#).unwrap();
        let s = serde_json::to_string(&p).unwrap();
        assert!(s.contains(r#""api_key":"sk-x""#), "got: {s}");
        assert!(!s.contains(r#""secret""#), "got: {s}");
    }

    #[test]
    fn rejects_document_carrying_both_spellings() {
        // serde maps the alias onto the same field, so a document that
        // carries both spellings is a duplicate-field error — the
        // ambiguity is rejected instead of one value silently winning.
        let r: Result<ProviderKey, _> =
            serde_json::from_str(r#"{"display_name":"x","api_key":"sk-new","secret":"sk-old"}"#);
        let err = r.expect_err("both spellings in one document must be rejected");
        assert!(
            err.to_string().contains("duplicate field"),
            "expected a duplicate-field error, got: {err}"
        );
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
        assert_eq!(p.telemetry_tags.kind, Some(TelemetryKind::Catalog));
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
        assert_eq!(p.telemetry_tags.kind, Some(TelemetryKind::Byo));
        assert!(!p.telemetry_tags.featured);
        assert_eq!(p.telemetry_tags.branded_provider, None);
        assert_eq!(p.telemetry_tags.byo_label.as_deref(), Some("platform-team"));
    }

    #[test]
    fn telemetry_tags_tolerates_unknown_field_for_forward_compat() {
        // cp-api may ship a new tag ahead of the DP rolling out; serde
        // must accept it. The write path still rejects it via the
        // strict schema validator (validate_provider_key in
        // models/schema.rs).
        let p: ProviderKey = serde_json::from_str(
            r#"{
                "display_name": "x",
                "secret": "k",
                "telemetry_tags": { "unknown_tag": "v", "featured": true }
            }"#,
        )
        .unwrap();
        assert!(p.telemetry_tags.featured);
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
            api_key: "sk-x".into(),
            api_base: None,
            provider: String::new(),
            adapter: None,
            apis: None,
            telemetry_tags: TelemetryTags::default(),
            request: None,
            response: None,
            strip_headers: default_strip_headers(),
            tls: None,
            runtime_id: String::new(),
        };
        let s = serde_json::to_string(&original).unwrap();
        let back: ProviderKey = serde_json::from_str(&s).unwrap();
        assert_eq!(original, back);
    }

    /// A stored document written before `tls` existed must keep loading,
    /// and must land on "verify, no extra roots" rather than on a derived
    /// `Default` that would leave `verify` false.
    #[test]
    fn a_document_without_tls_loads_with_no_override() {
        let pk: ProviderKey = serde_json::from_str(
            r#"{"display_name":"legacy","api_key":"sk-x","strip_headers":[]}"#,
        )
        .unwrap();
        assert!(pk.tls.is_none());
    }

    /// `tls: {}` and `tls: {"ca_cert": ...}` both have to verify unless
    /// the operator says otherwise, since `verify` is the one field whose
    /// absent value is dangerous.
    #[test]
    fn tls_verify_defaults_to_on_when_the_block_omits_it() {
        let pk: ProviderKey = serde_json::from_str(
            r#"{"display_name":"n","api_key":"k","strip_headers":[],"tls":{}}"#,
        )
        .unwrap();
        let tls = pk.tls.expect("tls block present");
        assert!(tls.verify);
        assert!(
            tls.is_noop(),
            "an empty block must not split the client pool"
        );

        let pk: ProviderKey = serde_json::from_str(
            r#"{"display_name":"n","api_key":"k","strip_headers":[],
                "tls":{"ca_cert":"-----BEGIN CERTIFICATE-----\nx\n-----END CERTIFICATE-----\n"}}"#,
        )
        .unwrap();
        let tls = pk.tls.expect("tls block present");
        assert!(tls.verify);
        assert!(!tls.is_noop());
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
    fn request_overrides_tolerates_unknown_field_for_forward_compat() {
        // cp-api may ship new override fields ahead of the DP rolling
        // out; serde must accept them. Typos on the write path are
        // still rejected by the strict schema validator
        // (validate_provider_key in models/schema.rs).
        let p: ProviderKey = serde_json::from_str(
            r#"{
                "display_name": "x",
                "secret": "k",
                "request": { "param_rename": {}, "default_headers": { "X-Foo": "bar" } }
            }"#,
        )
        .unwrap();
        let req = p.request.expect("request was Some");
        assert_eq!(req.default_headers.get("X-Foo"), Some(&"bar".to_string()));
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
    fn response_overrides_tolerates_unknown_field_for_forward_compat() {
        // cp-api may ship new override fields ahead of the DP rolling
        // out; serde must accept them (the strict write-path schema
        // still rejects them — validate_provider_key in models/schema.rs).
        let p: ProviderKey = serde_json::from_str(
            r#"{
                "display_name": "x",
                "secret": "k",
                "response": { "reasoning_fields": "delta.foo", "error_envelope": "openai" }
            }"#,
        )
        .unwrap();
        let resp = p.response.expect("response was Some");
        assert_eq!(resp.error_envelope.as_deref(), Some("openai"));
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
    fn param_constraints_tolerates_unknown_field_for_forward_compat() {
        // cp-api may ship a new clamp ahead of the DP rolling out;
        // serde must accept it (the strict write-path schema still
        // rejects it — validate_provider_key in models/schema.rs).
        let c: ParamConstraints =
            serde_json::from_str(r#"{"top_p_max": 0.9, "temperature_max": 1.0}"#).unwrap();
        assert_eq!(c.temperature_max, Some(1.0));
    }

    // ---- Issue #411 strip_headers deserialize/normalize ----

    fn pk_with_strip(strip_json: &str) -> ProviderKey {
        let json = format!(r#"{{"display_name":"x","secret":"sk","strip_headers":{strip_json}}}"#);
        serde_json::from_str(&json).unwrap()
    }

    #[test]
    fn strip_headers_default_applies_when_field_absent() {
        let pk: ProviderKey =
            serde_json::from_str(r#"{"display_name":"x","secret":"sk"}"#).unwrap();
        assert_eq!(pk.strip_headers, default_strip_headers());
    }

    #[test]
    fn strip_headers_explicit_empty_array_is_preserved() {
        // The "customer cleared all defaults" override case must
        // produce an empty Vec, NOT fall through to the default.
        let pk = pk_with_strip("[]");
        assert!(pk.strip_headers.is_empty());
    }

    #[test]
    fn strip_headers_trims_whitespace() {
        // Without the normalize hook, "  cookie  " would never match
        // an inbound `cookie` header → silent credential leak.
        let pk = pk_with_strip(r#"["  cookie  ", "\tauthorization\n"]"#);
        assert_eq!(pk.strip_headers, vec!["cookie", "authorization"]);
    }

    #[test]
    fn strip_headers_lowercases_input() {
        let pk = pk_with_strip(r#"["Authorization", "COOKIE", "X-Custom-Header"]"#);
        assert_eq!(
            pk.strip_headers,
            vec!["authorization", "cookie", "x-custom-header"]
        );
    }

    #[test]
    fn strip_headers_drops_empty_entries() {
        // Operators pasting from a comma-split tool may end up with
        // stray empty strings. Silently ignored, not fatal.
        let pk = pk_with_strip(r#"["", "  ", "cookie", ""]"#);
        assert_eq!(pk.strip_headers, vec!["cookie"]);
    }

    #[test]
    fn strip_headers_dedupes_case_insensitively() {
        // Customer accidentally added "Cookie" and "cookie" both.
        // Dedup post-lowercase. First-occurrence order is preserved
        // so the dashboard reads sanely.
        let pk = pk_with_strip(r#"["Cookie", "x-trace", "cookie", "X-Trace"]"#);
        assert_eq!(pk.strip_headers, vec!["cookie", "x-trace"]);
    }
}
