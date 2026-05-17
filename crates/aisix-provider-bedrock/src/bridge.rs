//! `BedrockBridge` — family Bridge for [`Adapter::Bedrock`].
//!
//! Skeleton: structure + publisher resolution + Hub-registrable
//! shell. Actual SigV4 + per-publisher dispatch lands in follow-up
//! PRs (see crate-level docs).

use aisix_gateway::{
    Bridge, BridgeContext, BridgeError, ChatChunkStream, ChatFormat, ChatResponse,
};
use async_trait::async_trait;

use crate::wire;

/// Family Bridge for AWS Bedrock Runtime.
///
/// **Skeleton:** compiles, registers, surfaces a clear
/// `BridgeError::Config` on every call. Real SigV4-signed dispatch
/// and per-publisher request building are wired in follow-up PRs —
/// see [`crate`] docs.
pub struct BedrockBridge {
    /// Static `name()` returned to the Hub. Stable across upgrades so
    /// metrics dashboards keep their existing `provider="bedrock"`
    /// filters working.
    name: &'static str,
}

impl BedrockBridge {
    /// Construct a Bedrock bridge with the canonical name
    /// `"bedrock"`. Matches the Adapter enum's wire form.
    pub fn new() -> Self {
        Self { name: "bedrock" }
    }
}

impl Default for BedrockBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// The set of Bedrock publishers the bridge will dispatch to.
/// Public so cp-api / dashboard can surface "which Bedrock
/// publishers are supported" without re-deriving the list from the
/// model id parser.
///
/// New publishers MUST be handled in [`BedrockPublisher::from_model_id`]
/// and the per-publisher request builder match in `chat` /
/// `chat_stream` (once dispatch lands).
///
/// Source: AWS Bedrock model catalog
/// <https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html>
/// cross-referenced with LiteLLM `bedrock/`.
///
/// **MVP coverage** (the variants with per-publisher dispatch already
/// planned in D7.2 / D7.3 / D7.4):
///
/// - [`Self::Anthropic`] — `anthropic.claude-*`
/// - [`Self::Meta`] — `meta.llama*`
/// - [`Self::Mistral`] — `mistral.*`
/// - [`Self::AmazonTitan`] — `amazon.titan-*`
/// - [`Self::AmazonNova`] — `amazon.nova-*`
/// - [`Self::Cohere`] — `cohere.command*`
/// - [`Self::Ai21`] — `ai21.jamba-*`
///
/// **Catch-all** ([`Self::Other`]) — every other Bedrock publisher
/// AWS hosts but we haven't pinned wire-shape dispatch for yet:
/// DeepSeek, Writer (Palmyra), Stability AI, Google (Gemma on
/// Bedrock), NVIDIA, Qwen, Moonshot AI, MiniMax, Z.AI, TwelveLabs,
/// OpenAI (gpt-oss on Bedrock). The resolver returns `Other` for
/// these so a customer registering e.g. `deepseek.r1-v1:0` doesn't
/// get a confusing "publisher unknown" at registration time — the
/// bridge knows it's a Bedrock id, dispatch just isn't wired yet.
/// Once D7.x lands per-publisher dispatch, `Other` shrinks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BedrockPublisher {
    /// `anthropic.claude-*` — Claude on Bedrock. Wire shape is
    /// Anthropic Messages with `anthropic_version: "bedrock-2023-05-31"`
    /// in the body (not header).
    Anthropic,
    /// `meta.llama*` — Llama 3 / 3.1 / 3.2 / 3.3 on Bedrock. Flat
    /// `prompt / max_gen_len / temperature` body shape.
    Meta,
    /// `mistral.mistral-*` / `mistral.mixtral-*` — Mistral on Bedrock.
    Mistral,
    /// `amazon.titan-*` — Titan Text / Embed. Uses
    /// `inputText + textGenerationConfig` body shape.
    AmazonTitan,
    /// `amazon.nova-*` — Nova Pro / Nova Lite / Nova Micro (2024 Q4).
    /// Uses Converse API natively.
    AmazonNova,
    /// `cohere.command-*` — Cohere Command R / R+ on Bedrock.
    Cohere,
    /// `ai21.jamba-*` — AI21 Jamba on Bedrock.
    Ai21,
    /// Recognized Bedrock publisher we haven't wired per-publisher
    /// dispatch for yet. Includes DeepSeek, Writer, Stability AI,
    /// Google Gemma, NVIDIA, Qwen, Moonshot AI, MiniMax, Z.AI,
    /// TwelveLabs, OpenAI gpt-oss. `chat()` returns
    /// `BridgeError::Config("not yet implemented")` referencing
    /// #302 Phase G follow-ups; future D7.x PRs add variants
    /// per concrete publisher as wire-shape dispatch lands.
    Other,
}

/// Publisher tags recognized by [`BedrockPublisher::from_model_id`]
/// as second-segment (or first-after-region) Bedrock-catalog
/// identifiers. Used by both the publisher resolver AND
/// [`strip_region_prefix`] (which only strips a region if the
/// segment after is a known publisher — otherwise an actual
/// publisher like `amazon.titan-...` would lose its `amazon.` segment).
///
/// Source: AWS Bedrock catalog enumeration as of 2026-05.
const KNOWN_PUBLISHER_TAGS: &[&str] = &[
    // MVP publishers (per-publisher dispatch planned in D7.2/3/4)
    "anthropic",
    "meta",
    "mistral",
    "amazon",
    "cohere",
    "ai21",
    // Other catalog publishers (resolve to BedrockPublisher::Other
    // until per-publisher dispatch lands)
    "deepseek",
    "writer",
    "stability",
    "google",
    "nvidia",
    "qwen",
    "moonshotai",
    "moonshot",
    "minimaxai",
    "minimax",
    "zai-org",
    "zai",
    "twelvelabs",
    "openai",
];

impl BedrockPublisher {
    /// Resolve the publisher from the Bedrock model id, tolerating
    /// cross-region inference profile prefixes.
    ///
    /// Recognized publisher tags (case-insensitive): see
    /// [`KNOWN_PUBLISHER_TAGS`]. Unknown tags return `None` —
    /// callers surface a clear `BridgeError::Config` so the operator
    /// can correct the model registration.
    ///
    /// **Cross-region prefix tolerance:** Bedrock supports inference
    /// profiles per
    /// <https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html>.
    /// Tolerated prefixes (per current AWS catalog as of 2026-05):
    /// `us.`, `eu.`, `apac.`, `global.`, `us-gov.`. The resolver
    /// strips a leading region tag before matching. New AWS geos
    /// will need [`strip_region_prefix`] updated.
    pub fn from_model_id(model_id: &str) -> Option<Self> {
        let stripped = strip_region_prefix(model_id);
        let (publisher_tag, _rest) = stripped.split_once('.')?;
        let tag_lower = publisher_tag.to_ascii_lowercase();
        let body_lower = stripped.to_ascii_lowercase();

        Some(match tag_lower.as_str() {
            "anthropic" => Self::Anthropic,
            "meta" => Self::Meta,
            "mistral" => Self::Mistral,
            "amazon" if body_lower.starts_with("amazon.nova-") => Self::AmazonNova,
            "amazon" if body_lower.starts_with("amazon.titan-") => Self::AmazonTitan,
            "amazon" => Self::Other, // future amazon.* families
            "cohere" => Self::Cohere,
            "ai21" => Self::Ai21,
            // Catalog publishers we know exist but haven't wired
            // dispatch for yet. Resolved to Other so registration
            // doesn't fail — operator sees a "publisher X not yet
            // implemented" at dispatch time once D7.x lands.
            "deepseek" | "writer" | "stability" | "google" | "nvidia" | "qwen" | "moonshotai"
            | "moonshot" | "minimaxai" | "minimax" | "zai-org" | "zai" | "twelvelabs"
            | "openai" => Self::Other,
            _ => return None,
        })
    }
}

/// Strip a leading cross-region inference profile prefix.
///
/// Returns the input unchanged when no recognized prefix matches.
///
/// Recognized prefixes (per AWS catalog as of 2026-05):
/// `us.`, `eu.`, `apac.`, `global.`, `us-gov.`.
///
/// The criterion: the leading segment must be 2–7 ASCII
/// lowercase letters / digits / hyphens, AND the segment after the
/// `.` separator must itself start with a known publisher tag.
/// Otherwise an actual publisher like `amazon.titan-...` would
/// lose its `amazon.` segment.
fn strip_region_prefix(model_id: &str) -> &str {
    let Some((maybe_region, rest)) = model_id.split_once('.') else {
        return model_id;
    };
    let len = maybe_region.len();
    // 2-7 covers `us` through `us-gov` (6) and gives a 1-char
    // safety margin without admitting random long prefixes.
    if !(2..=7).contains(&len) {
        return model_id;
    }
    // Hyphens are valid in `us-gov`. We deliberately do NOT accept
    // dots or other URL-control chars — that would let an attacker
    // smuggle path segments via the model id.
    if !maybe_region
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
    {
        return model_id;
    }
    // Only strip if what follows looks like a known publisher tag.
    let next_tag = rest.split('.').next().unwrap_or("").to_ascii_lowercase();
    if KNOWN_PUBLISHER_TAGS.contains(&next_tag.as_str()) {
        rest
    } else {
        model_id
    }
}

#[async_trait]
impl Bridge for BedrockBridge {
    fn name(&self) -> &'static str {
        self.name
    }

    async fn chat(
        &self,
        _req: &ChatFormat,
        ctx: &BridgeContext,
    ) -> Result<ChatResponse, BridgeError> {
        // Skeleton: validate the publisher resolution path so a
        // misconfigured model id surfaces a clear error today, even
        // though the actual SigV4-signed call is TODO.
        //
        // IMPORTANT: the Bedrock model id is on Model.model_name (the
        // upstream id the operator pinned when registering the model),
        // NOT on req.model (which is the gateway-internal display
        // name the customer typed in `/v1/chat/completions`). See
        // OpenAiBridge / `upstream_model(ctx)` for the established
        // pattern.
        let upstream_id = upstream_model(ctx)?;
        let _publisher = BedrockPublisher::from_model_id(upstream_id).ok_or_else(|| {
            BridgeError::Config(format!(
                "bedrock publisher unknown for model id {upstream_id:?}; \
                 expected one of anthropic.claude-* / meta.llama* / mistral.* / \
                 amazon.titan-* / amazon.nova-* / cohere.command* / ai21.jamba-* \
                 (optionally prefixed with a cross-region inference profile like us. / eu. / apac.)"
            ))
        })?;
        // Reserved-config helpers exercised by tests: keep wire module
        // reachable from the public surface so a future dispatch PR
        // can drop its body straight in.
        let _ = wire::reserved_sigv4_headers();
        Err(BridgeError::Config(
            "bedrock bridge is not yet implemented — \
             tracked under api7/AISIX-Cloud#302 Phase G (D7)"
                .into(),
        ))
    }

    async fn chat_stream(
        &self,
        _req: &ChatFormat,
        ctx: &BridgeContext,
    ) -> Result<ChatChunkStream, BridgeError> {
        let upstream_id = upstream_model(ctx)?;
        let _publisher = BedrockPublisher::from_model_id(upstream_id).ok_or_else(|| {
            BridgeError::Config(format!(
                "bedrock publisher unknown for model id {upstream_id:?}; \
                 expected one of anthropic.claude-* / meta.llama* / mistral.* / \
                 amazon.titan-* / amazon.nova-* / cohere.command* / ai21.jamba-* \
                 (optionally prefixed with a cross-region inference profile like us. / eu. / apac.)"
            ))
        })?;
        Err(BridgeError::Config(
            "bedrock bridge is not yet implemented — \
             tracked under api7/AISIX-Cloud#302 Phase G (D7)"
                .into(),
        ))
    }
}

/// Pull the upstream model id off the BridgeContext. Mirrors
/// OpenAiBridge's same-named helper — Bedrock model ids
/// (`anthropic.claude-...`, `meta.llama...`) live on Model.model_name,
/// not on the customer-facing display name in req.model.
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
    fn publisher_resolves_anthropic_claude_on_bedrock() {
        assert_eq!(
            BedrockPublisher::from_model_id("anthropic.claude-3-5-sonnet-20241022-v2:0"),
            Some(BedrockPublisher::Anthropic),
        );
        assert_eq!(
            BedrockPublisher::from_model_id("anthropic.claude-3-haiku-20240307-v1:0"),
            Some(BedrockPublisher::Anthropic),
        );
        // Matcher is on the `anthropic` tag, not `anthropic.claude`,
        // so any future non-Claude family AWS hosts under the
        // `anthropic.` namespace also resolves correctly. The
        // bridge resolves the publisher, the dispatch arm decides
        // which wire shape to use.
        assert_eq!(
            BedrockPublisher::from_model_id("anthropic.opus-4-1-20250805-v1:0"),
            Some(BedrockPublisher::Anthropic),
        );
    }

    #[test]
    fn publisher_resolves_meta_llama_variants() {
        // Bedrock's Llama wire form is `meta.llama3-X-...` —
        // single hyphen between `llama` and the version digit.
        assert_eq!(
            BedrockPublisher::from_model_id("meta.llama3-3-70b-instruct-v1:0"),
            Some(BedrockPublisher::Meta),
        );
        assert_eq!(
            BedrockPublisher::from_model_id("meta.llama3-405b-instruct-v1:0"),
            Some(BedrockPublisher::Meta),
        );
    }

    #[test]
    fn publisher_resolves_mistral_and_mixtral() {
        assert_eq!(
            BedrockPublisher::from_model_id("mistral.mistral-large-2402-v1:0"),
            Some(BedrockPublisher::Mistral),
        );
        assert_eq!(
            BedrockPublisher::from_model_id("mistral.mixtral-8x7b-instruct-v0:1"),
            Some(BedrockPublisher::Mistral),
        );
    }

    #[test]
    fn publisher_resolves_amazon_titan_and_nova_distinctly() {
        // Tight ordering pin: nova must resolve to AmazonNova,
        // titan to AmazonTitan. A future refactor that collapses
        // both to a single Amazon variant would lose the wire-
        // shape distinction (Nova uses Converse, Titan uses the
        // legacy inputText shape).
        assert_eq!(
            BedrockPublisher::from_model_id("amazon.nova-pro-v1:0"),
            Some(BedrockPublisher::AmazonNova),
        );
        assert_eq!(
            BedrockPublisher::from_model_id("amazon.nova-lite-v1:0"),
            Some(BedrockPublisher::AmazonNova),
        );
        assert_eq!(
            BedrockPublisher::from_model_id("amazon.titan-text-premier-v1:0"),
            Some(BedrockPublisher::AmazonTitan),
        );
        assert_eq!(
            BedrockPublisher::from_model_id("amazon.titan-text-express-v1"),
            Some(BedrockPublisher::AmazonTitan),
        );
    }

    #[test]
    fn publisher_resolves_cohere_command_r() {
        assert_eq!(
            BedrockPublisher::from_model_id("cohere.command-r-plus-v1:0"),
            Some(BedrockPublisher::Cohere),
        );
        assert_eq!(
            BedrockPublisher::from_model_id("cohere.command-r-v1:0"),
            Some(BedrockPublisher::Cohere),
        );
    }

    #[test]
    fn publisher_resolves_ai21_jamba_on_bedrock() {
        assert_eq!(
            BedrockPublisher::from_model_id("ai21.jamba-1-5-large-v1:0"),
            Some(BedrockPublisher::Ai21),
        );
    }

    #[test]
    fn publisher_strips_cross_region_us_prefix() {
        // `us.anthropic.claude-...` must resolve the same as the
        // non-prefixed form. The cross-region inference profile is
        // a routing detail — the publisher's wire shape is
        // identical regardless.
        assert_eq!(
            BedrockPublisher::from_model_id("us.anthropic.claude-3-5-sonnet-20241022-v2:0"),
            Some(BedrockPublisher::Anthropic),
        );
        assert_eq!(
            BedrockPublisher::from_model_id("eu.anthropic.claude-3-5-sonnet-20241022-v2:0"),
            Some(BedrockPublisher::Anthropic),
        );
        assert_eq!(
            BedrockPublisher::from_model_id("apac.anthropic.claude-3-5-sonnet-20241022-v2:0"),
            Some(BedrockPublisher::Anthropic),
        );
    }

    /// D7 audit HIGH-2 regression: `global.` and `us-gov.` are live
    /// Bedrock cross-region inference profile prefixes. The earlier
    /// matcher's "2-6 lowercase letters/digits only" rule silently
    /// failed on `us-gov.` (hyphen) and was lucky on `global.`
    /// (exactly 6 chars). A real GovCloud customer would have hit
    /// publisher-unknown with no actionable error.
    #[test]
    fn publisher_strips_global_and_us_gov_prefixes() {
        assert_eq!(
            BedrockPublisher::from_model_id("global.anthropic.claude-opus-4-5-20251101-v1:0"),
            Some(BedrockPublisher::Anthropic),
        );
        assert_eq!(
            BedrockPublisher::from_model_id("us-gov.anthropic.claude-3-5-sonnet-20241022-v2:0"),
            Some(BedrockPublisher::Anthropic),
        );
    }

    /// D7 audit HIGH-1 regression: catalog publishers AWS hosts on
    /// Bedrock today that we don't have wire-shape dispatch for yet.
    /// Resolving these to `Other` (rather than `None`) means a
    /// customer registering `deepseek.r1-v1:0` or
    /// `writer.palmyra-x5-v1:0` doesn't get a confusing "publisher
    /// unknown" at registration time — the bridge knows it's a
    /// Bedrock id, dispatch just isn't wired. Once D7.x lands per-
    /// publisher dispatch, the enum gets per-publisher variants.
    #[test]
    fn publisher_resolves_catalog_others_to_other_variant() {
        assert_eq!(
            BedrockPublisher::from_model_id("deepseek.r1-v1:0"),
            Some(BedrockPublisher::Other),
        );
        assert_eq!(
            BedrockPublisher::from_model_id("writer.palmyra-x5-v1:0"),
            Some(BedrockPublisher::Other),
        );
        assert_eq!(
            BedrockPublisher::from_model_id("stability.sd3-large-v1:0"),
            Some(BedrockPublisher::Other),
        );
        // Cross-region prefix also works for Other-variant publishers.
        assert_eq!(
            BedrockPublisher::from_model_id("us.deepseek.r1-v1:0"),
            Some(BedrockPublisher::Other),
        );
    }

    #[test]
    fn publisher_does_not_strip_publisher_segment_as_region() {
        // Guard the strip_region_prefix logic: `amazon.titan-...`
        // must NOT have its `amazon.` segment treated as a region
        // prefix. If it did, the rest would be `titan-...` which
        // doesn't start with `amazon.titan-`, so we'd lose the
        // publisher entirely.
        assert_eq!(
            BedrockPublisher::from_model_id("amazon.titan-text-premier-v1:0"),
            Some(BedrockPublisher::AmazonTitan),
        );
        // Same guard for `cohere.command-*`:
        assert_eq!(
            BedrockPublisher::from_model_id("cohere.command-r-v1:0"),
            Some(BedrockPublisher::Cohere),
        );
    }

    #[test]
    fn publisher_unknown_id_returns_none() {
        // `gpt-4o` has no `.` separator → no publisher tag.
        assert_eq!(BedrockPublisher::from_model_id("gpt-4o"), None);
        // Empty string → no tag.
        assert_eq!(BedrockPublisher::from_model_id(""), None);
        // `truly-unknown.foo-v1:0` has a tag but it's not in
        // KNOWN_PUBLISHER_TAGS — explicit "this is not a Bedrock
        // catalog id at all" signal so the caller can surface a
        // clear configuration error.
        assert_eq!(
            BedrockPublisher::from_model_id("truly-unknown.foo-v1:0"),
            None,
        );
    }

    #[test]
    fn bridge_name_is_stable() {
        // Metrics label is part of the public contract — a rename
        // would silently break customer dashboards.
        assert_eq!(BedrockBridge::new().name(), "bedrock");
    }

    use aisix_core::{Model, ProviderKey};
    use aisix_gateway::ChatMessage;
    use std::sync::Arc;

    /// Build a Model fixture where `model_name` (the upstream id) and
    /// `display_name` (the customer-facing name) deliberately differ.
    /// The bridge must dispatch off `model_name`, not the typed
    /// display name in `req.model` — pinning that contract here.
    fn sample_model_with(model_name: &str) -> Arc<Model> {
        // Note: Model.provider uses the legacy 6-value Provider enum.
        // amazon-bedrock isn't a Provider variant; the Adapter::Bedrock
        // routing happens off ProviderKey.adapter, not Model.provider.
        let cfg = format!(
            r#"{{
                "display_name": "customer-facing-name",
                "provider": "openai",
                "model_name": {model_name:?},
                "provider_key_id": "11111111-1111-1111-1111-111111111111"
            }}"#
        );
        Arc::new(serde_json::from_str(&cfg).unwrap())
    }

    fn sample_pk() -> Arc<ProviderKey> {
        Arc::new(
            serde_json::from_str(r#"{"display_name": "bedrock-prod", "secret": "AKIA-test"}"#)
                .unwrap(),
        )
    }

    #[tokio::test]
    async fn chat_surfaces_clear_not_implemented_error() {
        let bridge = BedrockBridge::new();
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("anthropic.claude-3-5-sonnet-20241022-v2:0"),
            sample_pk(),
        );
        // The customer-typed model name (req.model) is the display
        // name, NOT the Bedrock upstream id. The bridge must ignore
        // it and resolve off ctx.model.model_name instead.
        let req = ChatFormat::new("customer-facing-name", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(
                    msg.contains("bedrock bridge is not yet implemented"),
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

    #[tokio::test]
    async fn chat_ignores_req_model_and_uses_ctx_model_name() {
        // Regression test for D6 audit HIGH-1 (also caught here):
        // the bridge dispatches off Model.model_name (operator-
        // pinned upstream id), not req.model (customer-typed
        // display name). If a future refactor accidentally swaps
        // them, this test fails — `req.model = "gpt-4o"` would
        // surface "publisher unknown for gpt-4o", but with
        // model_name pointing at a real Bedrock id we expect the
        // skeleton's not-implemented error instead.
        let bridge = BedrockBridge::new();
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("anthropic.claude-3-haiku-20240307-v1:0"),
            sample_pk(),
        );
        // req.model deliberately set to something the publisher
        // resolver would reject if it were the source of truth.
        let req = ChatFormat::new("gpt-4o", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(
                    msg.contains("not yet implemented"),
                    "must hit the not-implemented stub (proving model_name was used), not the publisher-resolution guard; got {msg}"
                );
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_with_unknown_model_id_errors_before_dispatch() {
        // Publisher-resolution guard fires when model_name is
        // unrecognized — proves the bridge rejects malformed
        // registrations early.
        let bridge = BedrockBridge::new();
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("totally-bogus-model-id"),
            sample_pk(),
        );
        let req = ChatFormat::new("customer-facing-name", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(
                    msg.contains("bedrock publisher unknown"),
                    "must mention publisher resolution failure; got {msg}"
                );
                assert!(
                    msg.contains("totally-bogus-model-id"),
                    "must include the offending model id; got {msg}"
                );
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_with_missing_model_name_errors_before_dispatch() {
        // Defense: if Model.model_name is absent (shouldn't happen
        // in practice — cp-api requires it — but the field is
        // Option<String>), the bridge surfaces a clear error
        // rather than panicking or treating "" as a publisher.
        let bridge = BedrockBridge::new();
        let pk = sample_pk();
        let model_no_name: Arc<Model> = Arc::new(
            serde_json::from_str(
                r#"{
                    "display_name": "no-upstream-id",
                    "provider": "openai",
                    "provider_key_id": "11111111-1111-1111-1111-111111111111"
                }"#,
            )
            .unwrap(),
        );
        let ctx = BridgeContext::new("req-1", model_no_name, pk);
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
