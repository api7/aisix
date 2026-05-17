//! `VertexBridge` — family Bridge for [`Adapter::Vertex`].
//!
//! Skeleton: structure + publisher resolution + Hub-registrable
//! shell. The actual HTTP dispatch lands in follow-up PRs (see
//! crate-level docs).

use aisix_gateway::{
    Bridge, BridgeContext, BridgeError, ChatChunkStream, ChatFormat, ChatResponse,
};
use async_trait::async_trait;

use crate::wire;

/// Family Bridge for Google Vertex AI. Registered as the
/// `Adapter::Vertex` family entry in `Hub::register_family` — a
/// provider_key with `adapter: "vertex"` dispatches here regardless of
/// which Vertex publisher it targets (Gemini, Anthropic-on-Vertex,
/// Llama, Mistral, AI21, GPT-OSS). The publisher is resolved from the
/// upstream model id at dispatch time per the LiteLLM `vertex_ai/`
/// convention.
///
/// **Skeleton:** the bridge compiles, registers, and surfaces a clear
/// `BridgeError::Config` on every call. Real dispatch is wired in
/// follow-up PRs — see [`crate`] docs.
pub struct VertexBridge {
    /// Static `name()` returned to the Hub. Kept for metrics-label
    /// stability even though we don't have a transport yet.
    name: &'static str,
}

impl VertexBridge {
    /// Construct a Vertex bridge with the canonical name `"vertex"`.
    /// The Hub looks this up via [`Bridge::name`] when emitting
    /// per-request metrics (provider label).
    pub fn new() -> Self {
        Self { name: "vertex" }
    }
}

impl Default for VertexBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// The set of Vertex publishers we will eventually dispatch to.
/// Public so cp-api / dashboard can surface "which Vertex publishers
/// are supported" without re-deriving the list from the upstream id
/// parser.
///
/// New publishers added here MUST also be handled in
/// [`VertexPublisher::from_upstream_id`] and the dispatch match in
/// `chat` / `chat_stream`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexPublisher {
    /// `publishers/google/models/gemini-*` — Google's own Gemini line.
    Google,
    /// `publishers/anthropic/models/claude-*` — Anthropic models hosted
    /// on Vertex. Wire shape is `streamRawPredict`, not canonical
    /// Anthropic Messages.
    Anthropic,
    /// `publishers/meta/models/llama-*` — Meta's Llama family.
    Meta,
    /// `publishers/mistralai/models/mistral-*` — Mistral on Vertex.
    Mistral,
    /// `publishers/ai21/models/jamba-*` — AI21 Jamba family.
    Ai21,
}

impl VertexPublisher {
    /// Resolve the publisher from the upstream model id, following the
    /// LiteLLM `vertex_ai/` convention. Returns `None` for ids that
    /// don't match any known publisher prefix — caller surfaces a
    /// clear `BridgeError::Config` so the operator can correct the
    /// model registration.
    ///
    /// Recognized prefixes (case-insensitive on the model name):
    ///
    /// - `gemini-*` → [`Self::Google`]
    /// - `claude-*` → [`Self::Anthropic`]
    /// - `meta/llama-*` / `llama*` (e.g. `llama3-405b-...`) →
    ///   [`Self::Meta`] (per LiteLLM `vertex_ai_partner_models/main.py`
    ///   META_PREFIX = "meta/" plus the bare `llama` family)
    /// - `mistral-*` / `codestral-*` → [`Self::Mistral`]
    /// - `jamba-*` → [`Self::Ai21`]
    ///
    /// Publishers known to LiteLLM but not yet handled here (filed for
    /// follow-up): `deepseek-ai/*`, `qwen*`, `openai/gpt-oss-*`,
    /// `minimaxai/*`, `moonshotai/*`, `zai-org/*`. The current
    /// implementation surfaces "publisher unknown" for these — an
    /// operator hitting them gets a clear error before any traffic
    /// reaches Vertex.
    pub fn from_upstream_id(upstream_id: &str) -> Option<Self> {
        let lower = upstream_id.to_ascii_lowercase();
        if lower.starts_with("gemini-") {
            Some(Self::Google)
        } else if lower.starts_with("claude-") {
            Some(Self::Anthropic)
        } else if lower.starts_with("meta/") || lower.starts_with("llama") {
            // Both `meta/llama-3.3-70b-instruct-maas` and the bare
            // `llama3-405b-instruct-maas` form occur on real Vertex
            // deployments (per LiteLLM `vertex_ai_partner_models/
            // main.py:33-34`). The bare-llama branch deliberately
            // does NOT require a trailing hyphen — `llama3-...`
            // would miss otherwise.
            Some(Self::Meta)
        } else if lower.starts_with("mistral-") || lower.starts_with("codestral-") {
            Some(Self::Mistral)
        } else if lower.starts_with("jamba-") {
            Some(Self::Ai21)
        } else {
            None
        }
    }

    /// The `publishers/<tag>` URL segment Vertex expects on the
    /// `:streamRawPredict` and `:rawPredict` request paths. Used by
    /// the follow-up dispatch PR when building per-publisher endpoint
    /// URLs.
    ///
    /// **Returns `None` for [`Self::Meta`]** — Llama on Vertex does
    /// NOT use a `publishers/meta/...` URL. LiteLLM routes Llama
    /// through an OpenAPI shim at `endpoints/openapi/chat/completions`
    /// instead (see `litellm/llms/vertex_ai/vertex_llm_base.py:277`).
    /// A future D5.4 (Meta dispatch) builds that URL separately rather
    /// than via this helper.
    pub fn url_segment(self) -> Option<&'static str> {
        Some(match self {
            Self::Google => "publishers/google",
            Self::Anthropic => "publishers/anthropic",
            Self::Mistral => "publishers/mistralai",
            Self::Ai21 => "publishers/ai21",
            Self::Meta => return None,
        })
    }
}

#[async_trait]
impl Bridge for VertexBridge {
    fn name(&self) -> &'static str {
        self.name
    }

    async fn chat(
        &self,
        _req: &ChatFormat,
        ctx: &BridgeContext,
    ) -> Result<ChatResponse, BridgeError> {
        // Skeleton: validate the publisher resolution path so a
        // misconfigured upstream_id surfaces a clear error today,
        // even though the actual HTTP call is TODO.
        //
        // IMPORTANT: the Vertex upstream model id (e.g.
        // `gemini-1.5-pro`, `claude-3-5-sonnet@20241022`) lives on
        // Model.model_name, NOT on req.model (which is the
        // gateway-internal display name the customer typed in
        // `/v1/chat/completions`). See OpenAiBridge /
        // `upstream_model(ctx)` for the established pattern. This
        // is the D6 audit HIGH-1 fix backported.
        let upstream_id = upstream_model(ctx)?;
        let _publisher = VertexPublisher::from_upstream_id(upstream_id).ok_or_else(|| {
            BridgeError::Config(format!(
                "vertex publisher unknown for upstream model id {upstream_id:?}; \
                 expected one of gemini-* / claude-* / meta/llama-* or llama* / \
                 mistral-* / jamba-*"
            ))
        })?;
        // Reserved-config helper exercised by tests: keeps the wire
        // module reachable from the public surface so a future
        // dispatch PR can drop its body straight in.
        let _ = wire::reserved_query_params();
        Err(BridgeError::Config(
            "vertex bridge is not yet implemented — \
             tracked under api7/AISIX-Cloud#302 Phase E (D5)"
                .into(),
        ))
    }

    async fn chat_stream(
        &self,
        _req: &ChatFormat,
        ctx: &BridgeContext,
    ) -> Result<ChatChunkStream, BridgeError> {
        let upstream_id = upstream_model(ctx)?;
        let _publisher = VertexPublisher::from_upstream_id(upstream_id).ok_or_else(|| {
            BridgeError::Config(format!(
                "vertex publisher unknown for upstream model id {upstream_id:?}; \
                 expected one of gemini-* / claude-* / meta/llama-* or llama* / \
                 mistral-* / jamba-*"
            ))
        })?;
        Err(BridgeError::Config(
            "vertex bridge is not yet implemented — \
             tracked under api7/AISIX-Cloud#302 Phase E (D5)"
                .into(),
        ))
    }
}

/// Pull the upstream model id off the BridgeContext. Vertex model ids
/// (e.g. `gemini-1.5-pro`, `claude-3-5-sonnet@20241022`,
/// `meta/llama-3.3-70b-instruct-maas`) live on Model.model_name. The
/// customer-facing display name in req.model is NOT the source of
/// truth — that was D6 audit HIGH-1, backported here proactively.
fn upstream_model(ctx: &BridgeContext) -> Result<&str, BridgeError> {
    ctx.model
        .model_name
        .as_deref()
        .ok_or_else(|| BridgeError::Config("model.model_name missing".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publisher_resolves_gemini_prefix() {
        assert_eq!(
            VertexPublisher::from_upstream_id("gemini-1.5-pro"),
            Some(VertexPublisher::Google),
        );
        assert_eq!(
            VertexPublisher::from_upstream_id("gemini-2.0-flash-exp"),
            Some(VertexPublisher::Google),
        );
    }

    #[test]
    fn publisher_resolves_anthropic_prefix() {
        // Vertex hosts Claude under the `claude-*` model id with an
        // `@version` suffix; the prefix match must tolerate that.
        assert_eq!(
            VertexPublisher::from_upstream_id("claude-3-5-sonnet@20241022"),
            Some(VertexPublisher::Anthropic),
        );
        assert_eq!(
            VertexPublisher::from_upstream_id("claude-3-haiku@20240307"),
            Some(VertexPublisher::Anthropic),
        );
    }

    #[test]
    fn publisher_resolves_meta_mistral_ai21_prefixes() {
        // Both wire forms occur on real Vertex deployments:
        //   - `meta/llama-3.3-70b-instruct-maas` (META_PREFIX in
        //     LiteLLM vertex_ai_partner_models/main.py)
        //   - `llama3-405b-instruct-maas` (bare-llama form, no
        //     trailing hyphen between "llama" and the version)
        assert_eq!(
            VertexPublisher::from_upstream_id("meta/llama-3.3-70b-instruct-maas"),
            Some(VertexPublisher::Meta),
        );
        assert_eq!(
            VertexPublisher::from_upstream_id("llama3-405b-instruct-maas"),
            Some(VertexPublisher::Meta),
        );
        assert_eq!(
            VertexPublisher::from_upstream_id("mistral-large-2411"),
            Some(VertexPublisher::Mistral),
        );
        assert_eq!(
            VertexPublisher::from_upstream_id("codestral-2501"),
            Some(VertexPublisher::Mistral),
        );
        assert_eq!(
            VertexPublisher::from_upstream_id("jamba-1.5-large"),
            Some(VertexPublisher::Ai21),
        );
    }

    #[test]
    fn publisher_case_insensitive_on_model_name() {
        assert_eq!(
            VertexPublisher::from_upstream_id("Gemini-1.5-Pro"),
            Some(VertexPublisher::Google),
        );
    }

    #[test]
    fn publisher_unknown_prefix_returns_none() {
        assert_eq!(VertexPublisher::from_upstream_id("gpt-4o"), None);
        assert_eq!(VertexPublisher::from_upstream_id(""), None);
        assert_eq!(
            VertexPublisher::from_upstream_id("not-a-vendor-model"),
            None
        );
    }

    #[test]
    fn publisher_url_segment_matches_vertex_api_path() {
        // Tight pin on the URL fragment Vertex expects — a typo here
        // would surface as a 404 from every Vertex dispatch.
        assert_eq!(
            VertexPublisher::Google.url_segment(),
            Some("publishers/google"),
        );
        assert_eq!(
            VertexPublisher::Anthropic.url_segment(),
            Some("publishers/anthropic"),
        );
        // Mistral's Vertex publisher tag is `mistralai`, not `mistral`
        // — Google's catalog convention.
        assert_eq!(
            VertexPublisher::Mistral.url_segment(),
            Some("publishers/mistralai"),
        );
        assert_eq!(VertexPublisher::Ai21.url_segment(), Some("publishers/ai21"));
    }

    #[test]
    fn publisher_url_segment_meta_is_none() {
        // Llama on Vertex does NOT use `publishers/meta/...` —
        // LiteLLM routes through `endpoints/openapi/chat/completions`
        // (see vertex_llm_base.py:277). Pinning `None` here so a
        // future dispatch PR can rely on the helper's signal and
        // build the OpenAPI-shim URL separately for Meta instead of
        // synthesizing a 404-producing path.
        assert_eq!(VertexPublisher::Meta.url_segment(), None);
    }

    #[test]
    fn bridge_name_is_stable() {
        // Metrics label is part of the public contract — a rename
        // would silently break customer dashboards.
        assert_eq!(VertexBridge::new().name(), "vertex");
    }

    use aisix_core::{Model, ProviderKey};
    use aisix_gateway::ChatMessage;
    use std::sync::Arc;

    /// Build a Model fixture where `model_name` (the upstream Vertex
    /// id) and `display_name` (the customer-facing name) deliberately
    /// differ. The bridge must dispatch off `model_name`, not the
    /// typed display name in `req.model`.
    fn sample_model_with(model_name: &str) -> Arc<Model> {
        let cfg = format!(
            r#"{{
                "display_name": "customer-facing-name",
                "provider": "google",
                "model_name": {model_name:?},
                "provider_key_id": "11111111-1111-1111-1111-111111111111"
            }}"#
        );
        Arc::new(serde_json::from_str(&cfg).unwrap())
    }

    fn sample_pk() -> Arc<ProviderKey> {
        Arc::new(
            serde_json::from_str(r#"{"display_name": "vertex-prod", "secret": "ya29.test"}"#)
                .unwrap(),
        )
    }

    #[tokio::test]
    async fn chat_surfaces_clear_not_implemented_error() {
        // Skeleton contract: dispatch returns Config error with the
        // tracking-issue link, so an operator who lands here knows
        // the path is intentional-WIP not silently-broken.
        let bridge = VertexBridge::new();
        let ctx = BridgeContext::new("req-1", sample_model_with("gemini-1.5-pro"), sample_pk());
        // req.model is the customer-facing display name; the bridge
        // must ignore it and resolve off Model.model_name.
        let req = ChatFormat::new("customer-facing-name", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(
                    msg.contains("vertex bridge is not yet implemented"),
                    "error message must call out the WIP status; got {msg}"
                );
                assert!(
                    msg.contains("#302"),
                    "error message must link to the tracking issue; got {msg}"
                );
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    /// D6 audit HIGH-1 regression (backported to D5): dispatch must
    /// read the upstream model id from ctx.model.model_name, NOT
    /// from req.model. req.model is the customer-typed display
    /// name; resolving off it would produce `/<region>/projects/
    /// .../publishers/google/models/<customer-display-name>` and
    /// 404 from every Vertex request.
    #[tokio::test]
    async fn chat_ignores_req_model_and_uses_ctx_model_name() {
        let bridge = VertexBridge::new();
        // model_name = valid Vertex id; req.model = something the
        // publisher resolver would reject if it were the source of
        // truth. Expectation: the not-implemented stub fires,
        // proving model_name was the actual input.
        let ctx = BridgeContext::new("req-1", sample_model_with("gemini-1.5-pro"), sample_pk());
        let req = ChatFormat::new("gpt-4o", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(
                    msg.contains("not yet implemented"),
                    "must hit the not-implemented stub (proving model_name was used); got {msg}"
                );
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_with_unknown_publisher_prefix_errors_before_dispatch() {
        // Publisher-resolution guard fires when model_name is
        // unrecognized — proves the bridge will route per
        // upstream_id once the dispatch lands.
        let bridge = VertexBridge::new();
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("totally-bogus-vertex-id"),
            sample_pk(),
        );
        let req = ChatFormat::new("customer-facing-name", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(
                    msg.contains("vertex publisher unknown"),
                    "must mention publisher resolution failure; got {msg}"
                );
                assert!(
                    msg.contains("totally-bogus-vertex-id"),
                    "must include the offending model id; got {msg}"
                );
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    /// Defense: if Model.model_name is absent (shouldn't happen in
    /// practice — cp-api requires it — but the field is
    /// Option<String>), the bridge surfaces a clear error rather
    /// than panicking or treating "" as a publisher.
    #[tokio::test]
    async fn chat_with_missing_model_name_errors_before_dispatch() {
        let bridge = VertexBridge::new();
        let model_no_name: Arc<Model> = Arc::new(
            serde_json::from_str(
                r#"{
                    "display_name": "no-upstream-id",
                    "provider": "google",
                    "provider_key_id": "11111111-1111-1111-1111-111111111111"
                }"#,
            )
            .unwrap(),
        );
        let ctx = BridgeContext::new("req-1", model_no_name, sample_pk());
        let req = ChatFormat::new("customer-facing", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(msg.contains("model_name missing"), "got {msg}");
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }
}
