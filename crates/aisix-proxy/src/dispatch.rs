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
use aisix_gateway::{Bridge, Hub};

use crate::error::ProxyError;

/// Resolve the Bridge to dispatch this request through.
///
/// Tries the two-tier dispatch first (specialized vendor → adapter
/// family), and falls back to the legacy `Provider`-keyed registry if
/// neither tier matches. This is the Phase D cutover join point: new
/// `ProviderKey` payloads carrying `adapter` and `provider` hit the
/// two-tier path; pre-cutover payloads (`adapter: None`, empty
/// `provider`) fall through to the legacy registry unchanged.
///
/// Returns `None` only when both layers miss — i.e. the operator has no
/// bridge wired for this request at all.
pub(crate) fn resolve_bridge(
    hub: &Hub,
    provider_key: &ProviderKey,
    provider: Provider,
) -> Option<Arc<dyn Bridge>> {
    hub.dispatch_two_tier(provider_key)
        .or_else(|| hub.get(provider))
}

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

/// Path suffixes the proxy-side handlers (audio, responses, messages)
/// build via [`build_v1_url`]. If an operator accidentally pasted the
/// full upstream URL into `api_base`, strip the suffix here so the
/// later [`build_v1_url`] call does not double-append. The list mirrors
/// every concrete endpoint the proxy currently routes to upstream,
/// covered in both the bare (`/responses`) and `/v1`-prefixed
/// (`/v1/responses`) form an operator might paste.
///
/// Longer suffixes appear first so `/v1/audio/transcriptions` matches
/// before `/audio/transcriptions`, etc.
const API_BASE_ENDPOINT_SUFFIXES: &[&str] = &[
    "/v1/audio/transcriptions",
    "/v1/audio/translations",
    "/v1/audio/speech",
    "/v1/chat/completions",
    "/v1/images/generations",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/responses",
    "/v1/messages",
    "/v1/rerank",
    "/audio/transcriptions",
    "/audio/translations",
    "/audio/speech",
    "/chat/completions",
    "/images/generations",
    "/completions",
    "/embeddings",
    "/responses",
    "/messages",
    "/rerank",
];

/// Strip a known endpoint suffix from `base` and its trailing slash.
/// Idempotent. Mirrors the suffix-stripping the bridge crates do on
/// their own `resolve_base`, so handlers that bypass the bridge (audio,
/// responses, messages) get the same tolerance.
fn strip_endpoint_suffix(base: &str) -> &str {
    let trimmed = base.trim_end_matches('/');
    for suffix in API_BASE_ENDPOINT_SUFFIXES {
        if let Some(rest) = trimmed.strip_suffix(suffix) {
            return rest.trim_end_matches('/');
        }
    }
    trimmed
}

/// The upstream base URL: `provider_key.api_base` override if set,
/// otherwise the `Provider`'s built-in default. Tolerates an operator
/// pasting the full upstream URL into `api_base` by stripping any
/// trailing endpoint suffix — see [`API_BASE_ENDPOINT_SUFFIXES`] for
/// the full list and [`build_v1_url`] for the matching `/v1` synthesis.
pub(crate) fn resolve_base_url(provider: Provider, provider_key: &ProviderKey) -> String {
    match provider_key.api_base.as_deref() {
        Some(b) if !b.trim().is_empty() => strip_endpoint_suffix(b.trim()).to_string(),
        _ => provider.default_base_url().to_string(),
    }
}

/// Build a `/v1`-prefixed upstream URL while tolerating either
/// convention for the configured `api_base`:
///
/// * `https://api.openai.com` builds `…/v1/<path>` — the provider
///   default convention used by `Provider::Openai.default_base_url()`
///   and `aisix-proxy`'s pre-existing handlers.
/// * `https://api.openai.com/v1` also builds `…/v1/<path>` — the
///   OpenAI SDK convention every published example uses, and the
///   exact placeholder the dashboard's provider-keys form pre-fills.
///
/// Without this normalization, a customer who follows OpenAI SDK
/// docs (api_base = `…/v1`) hits `…/v1/v1/responses` upstream — the
/// upstream 404s, the DP wraps it as 502 upstream_error, and the
/// failure surfaces as "intermittent SDK-incompatible behaviour"
/// (chat works because aisix-provider-openai/src/bridge.rs uses
/// the OpenAI-SDK convention; the proxy crate handlers — responses,
/// rerank, audio — follow the provider-default convention, so the
/// customer's api_base satisfies one but not the other).
///
/// `path` MUST start with `/` and SHOULD start with the version-
/// independent route (e.g. `/responses`, not `/v1/responses`); this
/// helper owns the `/v1` prefix.
pub(crate) fn build_v1_url(base: &str, path: &str) -> String {
    // assert!, not debug_assert! — the cost of a single bounds check
    // per upstream dispatch is negligible compared to the network
    // round-trip, and a release-mode caller passing a malformed path
    // would silently produce a wrong URL (e.g. `…/v1responses`).
    assert!(
        path.starts_with('/'),
        "build_v1_url path must start with /, got {path:?}",
    );
    let trimmed = base.trim_end_matches('/');
    if trimmed.ends_with("/v1") {
        format!("{trimmed}{path}")
    } else {
        format!("{trimmed}/v1{path}")
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

    fn pk_with_base(api_base: &str) -> ProviderKey {
        let cfg = format!(r#"{{"display_name":"x","secret":"k","api_base":"{api_base}"}}"#);
        serde_json::from_str(&cfg).unwrap()
    }

    /// Every OpenAI-shape paste an operator might make must, when fed
    /// to `build_v1_url(base, "/<endpoint>")`, produce the canonical
    /// upstream URL. The intermediate `resolve_base_url` result may be
    /// either bare-host or `<host>/v1` — `build_v1_url` accepts both —
    /// so the assertion is on the final URL the handler dispatches to,
    /// not on the intermediate base.
    ///
    /// Without suffix stripping, pasting `…/v1/audio/transcriptions`
    /// into `api_base` produces `…/v1/audio/transcriptions/v1/audio/transcriptions`.
    #[test]
    fn resolve_base_url_strips_openai_endpoint_suffixes() {
        let cases: &[(&str, &str)] = &[
            ("https://api.openai.com/v1", "/responses"),
            ("https://api.openai.com/v1/", "/responses"),
            ("https://api.openai.com/v1/responses", "/responses"),
            (
                "https://api.openai.com/v1/audio/transcriptions",
                "/audio/transcriptions",
            ),
            (
                "https://api.openai.com/v1/audio/translations",
                "/audio/translations",
            ),
            ("https://api.openai.com/v1/audio/speech", "/audio/speech"),
            (
                "https://api.openai.com/v1/chat/completions",
                "/chat/completions",
            ),
            ("https://api.openai.com/v1/completions", "/completions"),
            ("https://api.openai.com/v1/embeddings", "/embeddings"),
            (
                "https://api.openai.com/v1/images/generations",
                "/images/generations",
            ),
            ("https://api.openai.com/v1/rerank", "/rerank"),
        ];
        for (paste, endpoint) in cases {
            let pk = pk_with_base(paste);
            let base = resolve_base_url(Provider::Openai, &pk);
            let url = build_v1_url(&base, endpoint);
            let expected = format!("https://api.openai.com/v1{endpoint}");
            assert_eq!(
                url, expected,
                "paste {paste:?} + endpoint {endpoint:?} must build to {expected:?}",
            );
        }
    }

    /// DeepSeek serves OpenAI-compatible endpoints at the host root.
    /// Same contract: every paste must build to the canonical URL.
    #[test]
    fn resolve_base_url_strips_deepseek_endpoint_suffixes() {
        for paste in [
            "https://api.deepseek.com",
            "https://api.deepseek.com/",
            "https://api.deepseek.com/chat/completions",
            "https://api.deepseek.com/embeddings",
        ] {
            let pk = pk_with_base(paste);
            let base = resolve_base_url(Provider::Deepseek, &pk);
            let url = build_v1_url(&base, "/chat/completions");
            assert_eq!(
                url, "https://api.deepseek.com/v1/chat/completions",
                "paste {paste:?} must build to the canonical chat-completions URL",
            );
        }
    }

    /// Anthropic: the messages handler builds `…/v1/messages`. A paste
    /// of the full upstream URL must strip so `build_v1_url("/messages")`
    /// does not produce `…/v1/messages/v1/messages`.
    #[test]
    fn resolve_base_url_strips_anthropic_messages_suffix() {
        for paste in [
            "https://api.anthropic.com",
            "https://api.anthropic.com/",
            "https://api.anthropic.com/v1",
            "https://api.anthropic.com/v1/messages",
            "https://api.anthropic.com/v1/messages/",
        ] {
            let pk = pk_with_base(paste);
            let base = resolve_base_url(Provider::Anthropic, &pk);
            assert_eq!(
                build_v1_url(&base, "/messages"),
                "https://api.anthropic.com/v1/messages",
                "paste {paste:?} must build to the canonical messages URL",
            );
        }
    }

    /// Non-canonical hosts (corporate proxies, test mocks) pass through
    /// after suffix-stripping. The operator's path on a non-default
    /// host is trusted as-is.
    #[test]
    fn resolve_base_url_passes_non_canonical_hosts_through() {
        let pk = pk_with_base("https://proxy.example.com/openai-shim");
        assert_eq!(
            resolve_base_url(Provider::Openai, &pk),
            "https://proxy.example.com/openai-shim",
        );

        // Suffix stripping still applies on non-canonical hosts —
        // operator pasting the full upstream URL is still recovered.
        let pk = pk_with_base("https://proxy.example.com/openai-shim/v1/responses");
        let base = resolve_base_url(Provider::Openai, &pk);
        assert_eq!(
            build_v1_url(&base, "/responses"),
            "https://proxy.example.com/openai-shim/v1/responses",
        );
    }

    /// Whitespace trim must compose with suffix stripping.
    #[test]
    fn resolve_base_url_trims_whitespace_and_endpoint_suffix() {
        let pk = pk_with_base("  https://api.openai.com/v1/chat/completions/  ");
        let base = resolve_base_url(Provider::Openai, &pk);
        assert_eq!(
            build_v1_url(&base, "/chat/completions"),
            "https://api.openai.com/v1/chat/completions",
        );
    }

    // ---------------------------------------------------------------
    // build_v1_url — the path-doubling regression fixture.
    // ---------------------------------------------------------------

    #[test]
    fn build_v1_url_appends_v1_when_base_lacks_it() {
        // Provider-default convention (Provider::Openai.default_base_url()
        // returns `https://api.openai.com`, no /v1).
        assert_eq!(
            build_v1_url("https://api.openai.com", "/responses"),
            "https://api.openai.com/v1/responses",
        );
    }

    #[test]
    fn build_v1_url_skips_v1_when_base_already_has_it() {
        // Customer follows the OpenAI SDK convention + the dashboard's
        // provider-keys form pre-fill (`https://api.openai.com/v1`).
        // A naive `format!("{base}/v1/responses")` would produce
        // `https://api.openai.com/v1/v1/responses` and 404 upstream.
        assert_eq!(
            build_v1_url("https://api.openai.com/v1", "/responses"),
            "https://api.openai.com/v1/responses",
        );
    }

    #[test]
    fn build_v1_url_strips_trailing_slash() {
        assert_eq!(
            build_v1_url("https://api.openai.com/", "/rerank"),
            "https://api.openai.com/v1/rerank",
        );
        assert_eq!(
            build_v1_url("https://api.openai.com/v1/", "/rerank"),
            "https://api.openai.com/v1/rerank",
        );
    }

    #[test]
    fn build_v1_url_handles_nested_paths() {
        // /audio/speech, /audio/transcriptions, /audio/translations all
        // pass nested paths — make sure the helper doesn't try to be
        // clever about them.
        assert_eq!(
            build_v1_url("https://api.openai.com", "/audio/speech"),
            "https://api.openai.com/v1/audio/speech",
        );
        assert_eq!(
            build_v1_url("https://api.openai.com/v1", "/audio/transcriptions"),
            "https://api.openai.com/v1/audio/transcriptions",
        );
    }

    #[test]
    #[should_panic(expected = "build_v1_url path must start with /")]
    fn build_v1_url_rejects_path_without_leading_slash() {
        // Misuse — handlers should always pass a `/`-prefixed path.
        let _ = build_v1_url("https://api.openai.com", "responses");
    }
}
