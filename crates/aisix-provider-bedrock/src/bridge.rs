//! `BedrockBridge` — family Bridge for [`Adapter::Bedrock`].
//!
//! Multi-publisher dispatch backed by `aws-sdk-bedrockruntime`. The
//! SDK handles SigV4 signing, retries, and (for the streaming
//! follow-up D7.2.b) the binary event-stream framing. Per-publisher
//! request bodies + response decoding live in this crate.
//!
//! **Currently wired:** `anthropic.*` (Claude on Bedrock) chat. Other
//! publishers + streaming surface clear `not yet implemented` errors
//! referencing D7.x follow-ups — see crate-level docs.
//!
//! Credentials: `ProviderKey.secret` is a JSON-encoded
//! `{access_key_id, secret_access_key, session_token?, region}`
//! struct. The bridge parses it per request (cheap — strings only)
//! and constructs a per-call SDK client. `ProviderKey.api_base` (if
//! set) is forwarded as the SDK's `endpoint_url` so operators can
//! point at a private deployment / VPC endpoint.

use aisix_gateway::{
    Bridge, BridgeContext, BridgeError, ChatChunkStream, ChatFormat, ChatResponse,
};
use async_trait::async_trait;
use aws_credential_types::provider::SharedCredentialsProvider;
use aws_credential_types::Credentials;
use aws_sdk_bedrockruntime::config::{BehaviorVersion, Region};
use aws_sdk_bedrockruntime::error::SdkError;
use aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError;
use aws_sdk_bedrockruntime::primitives::Blob;
use aws_sdk_bedrockruntime::Client as BedrockClient;
use aws_smithy_runtime_api::client::result::ServiceError;
use serde::Deserialize;

use aisix_provider_anthropic::wire::{
    build_request, response_into_chat_response, split_system, AnthropicResponse,
};

use crate::wire;

/// Anthropic-on-Bedrock body-shape version pin per
/// <https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html>.
/// Goes in the request body as the `anthropic_version` field; the
/// `model` field is stripped because Bedrock keys dispatch off the
/// URL path, not the body.
const BEDROCK_ANTHROPIC_VERSION: &str = "bedrock-2023-05-31";

/// Family Bridge for AWS Bedrock Runtime.
pub struct BedrockBridge {
    /// Static `name()` returned to the Hub. Stable across upgrades so
    /// metrics dashboards keep their existing `provider="bedrock"`
    /// filters working.
    name: &'static str,
    /// Test-only endpoint URL override. When set, the SDK config's
    /// `endpoint_url` is pinned to this value so wiremock can stand
    /// in for `bedrock-runtime.<region>.amazonaws.com`. Credentials,
    /// region, and SigV4 signing still run normally.
    #[cfg(test)]
    endpoint_url_override: Option<String>,
}

impl BedrockBridge {
    /// Construct a Bedrock bridge with the canonical name
    /// `"bedrock"`. Matches the Adapter enum's wire form.
    pub fn new() -> Self {
        Self {
            name: "bedrock",
            #[cfg(test)]
            endpoint_url_override: None,
        }
    }

    /// Test-only seam: rewrite the SDK's endpoint URL so wiremock can
    /// stand in for AWS. Credentials / region / SigV4 paths all run
    /// normally; only the host is different.
    #[cfg(test)]
    pub(crate) fn with_endpoint_override(mut self, url: impl Into<String>) -> Self {
        self.endpoint_url_override = Some(url.into());
        self
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
/// `chat_stream`.
///
/// Source: AWS Bedrock model catalog
/// <https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html>.
///
/// **MVP coverage** (the variants with per-publisher dispatch already
/// planned in D7.2 / D7.3 / D7.4):
///
/// - [`Self::Anthropic`] — `anthropic.claude-*` (wired in this PR)
/// - [`Self::Meta`] — `meta.llama*` (D7.3)
/// - [`Self::Mistral`] — `mistral.*` (D7.4)
/// - [`Self::AmazonTitan`] — `amazon.titan-*` (D7.4)
/// - [`Self::AmazonNova`] — `amazon.nova-*` (D7.4)
/// - [`Self::Cohere`] — `cohere.command*` (D7.4)
/// - [`Self::Ai21`] — `ai21.jamba-*` (D7.4)
///
/// **Catch-all** ([`Self::Other`]) — every other Bedrock publisher
/// AWS hosts but we haven't pinned wire-shape dispatch for yet:
/// DeepSeek, Writer (Palmyra), Stability AI, Google (Gemma on
/// Bedrock), NVIDIA, Qwen, Moonshot AI, MiniMax, Z.AI, TwelveLabs,
/// OpenAI (gpt-oss on Bedrock). Resolver returns `Other` for these
/// so a customer registering e.g. `deepseek.r1-v1:0` doesn't get a
/// confusing "publisher unknown" at registration time — the bridge
/// knows it's a Bedrock id, dispatch just isn't wired yet.
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
    /// `amazon.nova-*` — Nova Pro / Nova Lite / Nova Micro. Uses
    /// Converse API natively.
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
    /// #302 Phase G follow-ups.
    Other,
}

/// Publisher tags recognized as second-segment (or first-after-region)
/// Bedrock-catalog identifiers.
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
    /// cross-region inference profile prefixes (`us.`, `eu.`,
    /// `apac.`, `global.`, `us-gov.`).
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
            "amazon" => Self::Other,
            "cohere" => Self::Cohere,
            "ai21" => Self::Ai21,
            "deepseek" | "writer" | "stability" | "google" | "nvidia" | "qwen" | "moonshotai"
            | "moonshot" | "minimaxai" | "minimax" | "zai-org" | "zai" | "twelvelabs"
            | "openai" => Self::Other,
            _ => return None,
        })
    }

    /// Human-readable name used in publisher-not-implemented errors.
    fn name(&self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::Meta => "meta",
            Self::Mistral => "mistral",
            Self::AmazonTitan => "amazon.titan",
            Self::AmazonNova => "amazon.nova",
            Self::Cohere => "cohere",
            Self::Ai21 => "ai21",
            Self::Other => "<unspecified>",
        }
    }
}

/// Strip a leading cross-region inference profile prefix.
///
/// Recognized prefixes (per AWS catalog as of 2026-05):
/// `us.`, `eu.`, `apac.`, `global.`, `us-gov.`.
fn strip_region_prefix(model_id: &str) -> &str {
    let Some((maybe_region, rest)) = model_id.split_once('.') else {
        return model_id;
    };
    let len = maybe_region.len();
    if !(2..=7).contains(&len) {
        return model_id;
    }
    if !maybe_region
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
    {
        return model_id;
    }
    let next_tag = rest.split('.').next().unwrap_or("").to_ascii_lowercase();
    if KNOWN_PUBLISHER_TAGS.contains(&next_tag.as_str()) {
        rest
    } else {
        model_id
    }
}

/// Schema for `ProviderKey.secret` on a Bedrock provider key.
///
/// Convention: AWS credentials are JSON-encoded into the `secret`
/// field. The cp-api side delivers them already-decrypted (mTLS-only
/// etcd channel; see ProviderKey doc).
///
/// `endpoint_url` is intentionally NOT in here — that goes in
/// `ProviderKey.api_base` so the cp-api validator can apply normal
/// URL-shape rules. Region is in here because Bedrock keys dispatch
/// off region (`bedrock-runtime.<region>.amazonaws.com`).
#[derive(Debug, Deserialize)]
struct BedrockSecret {
    access_key_id: String,
    secret_access_key: String,
    /// AWS STS session token. Optional — long-lived static keys
    /// don't have one; assume-role credentials do.
    #[serde(default)]
    session_token: Option<String>,
    /// AWS region the Bedrock dispatch targets (e.g. `us-west-2`).
    /// Required — Bedrock's URL is region-keyed and the SDK won't
    /// dispatch without it.
    region: String,
}

impl BedrockSecret {
    /// Parse the JSON-encoded credential blob. Audit M1: error
    /// messages here MUST NOT echo the raw secret content — only
    /// generic shape errors.
    fn parse(secret: &str) -> Result<Self, BridgeError> {
        if secret.trim().is_empty() {
            return Err(BridgeError::Config(
                "bedrock provider_key.secret is empty — \
                 expected JSON {access_key_id, secret_access_key, region, session_token?}"
                    .into(),
            ));
        }
        serde_json::from_str::<BedrockSecret>(secret).map_err(|_e| {
            // Intentionally do NOT include the underlying serde error
            // message — it can leak partial secret contents (e.g.
            // "invalid character 'X' at position N" reveals what's
            // in the JSON). Generic shape hint is enough for the
            // operator who controls the registration.
            BridgeError::Config(
                "bedrock provider_key.secret must be valid JSON: \
                 {access_key_id, secret_access_key, region, session_token?}"
                    .into(),
            )
        })
    }
}

/// Build a Bedrock SDK Client from the parsed credentials plus the
/// optional endpoint override.
fn build_client(
    creds: &BedrockSecret,
    endpoint_url: Option<&str>,
) -> Result<BedrockClient, BridgeError> {
    if creds.region.trim().is_empty() {
        return Err(BridgeError::Config(
            "bedrock provider_key.secret.region is empty — \
             AWS Bedrock dispatch is region-keyed and requires e.g. \"us-west-2\""
                .into(),
        ));
    }
    let aws_creds = Credentials::new(
        creds.access_key_id.clone(),
        creds.secret_access_key.clone(),
        creds.session_token.clone(),
        None,
        "aisix-provider-bedrock",
    );
    let mut builder = aws_config::SdkConfig::builder()
        .behavior_version(BehaviorVersion::latest())
        .region(Region::new(creds.region.clone()))
        .credentials_provider(SharedCredentialsProvider::new(aws_creds))
        .sleep_impl(aws_smithy_async::rt::sleep::SharedAsyncSleep::new(
            aws_smithy_async::rt::sleep::TokioSleep::new(),
        ));
    if let Some(url) = endpoint_url {
        builder = builder.endpoint_url(url);
    }
    let sdk_cfg = builder.build();
    Ok(BedrockClient::new(&sdk_cfg))
}

/// Pull the upstream model id off the BridgeContext.
fn upstream_model(ctx: &BridgeContext) -> Result<&str, BridgeError> {
    ctx.model
        .model_name
        .as_deref()
        .ok_or_else(|| BridgeError::Config("model.model_name missing".into()))
}

/// Translate an SDK error into the canonical `BridgeError`.
///
/// **Audit M1 — sensitive-info redaction:** Bedrock error envelopes
/// frequently include the operator's model id, region, account
/// numbers (in ARNs), and IAM role names. Surfacing these to a
/// downstream customer leaks operator-internal taxonomy. We map to
/// canned status-keyed phrases.
fn map_sdk_error(err: SdkError<InvokeModelError>) -> BridgeError {
    match err {
        SdkError::TimeoutError(_) => BridgeError::Timeout { elapsed_ms: 0 },
        SdkError::DispatchFailure(_) => BridgeError::Transport("upstream dispatch failed".into()),
        SdkError::ConstructionFailure(_) => {
            BridgeError::Config("upstream request construction failed".into())
        }
        SdkError::ResponseError(_) => {
            BridgeError::UpstreamDecode("upstream response could not be parsed".into())
        }
        SdkError::ServiceError(svc) => map_service_error(svc),
        _ => BridgeError::Transport("upstream dispatch failed".into()),
    }
}

fn map_service_error(
    svc: ServiceError<InvokeModelError, aws_smithy_runtime_api::http::Response>,
) -> BridgeError {
    let raw = svc.into_raw();
    let status = raw.status().as_u16();
    let message = match status {
        401 | 403 => "upstream authentication failed".to_string(),
        404 => "upstream model not found".to_string(),
        408 => "upstream request timeout".to_string(),
        429 => "upstream rate limited".to_string(),
        500..=599 => format!("upstream returned {status}"),
        _ => format!("upstream returned {status}"),
    };
    BridgeError::UpstreamStatus {
        status,
        message,
        retry_after: None,
    }
}

#[async_trait]
impl Bridge for BedrockBridge {
    fn name(&self) -> &'static str {
        self.name
    }

    async fn chat(
        &self,
        req: &ChatFormat,
        ctx: &BridgeContext,
    ) -> Result<ChatResponse, BridgeError> {
        let upstream_id = upstream_model(ctx)?;
        let publisher = BedrockPublisher::from_model_id(upstream_id).ok_or_else(|| {
            BridgeError::Config(format!(
                "bedrock publisher unknown for model id {upstream_id:?}; \
                 expected one of anthropic.claude-* / meta.llama* / mistral.* / \
                 amazon.titan-* / amazon.nova-* / cohere.command* / ai21.jamba-* \
                 (optionally prefixed with a cross-region inference profile like us. / eu. / apac.)"
            ))
        })?;
        // Keep wire module reachable from the public surface so the
        // streaming follow-up can wire SigV4-reserved-header checks
        // for any operator default_headers override.
        let _ = wire::reserved_sigv4_headers();

        match publisher {
            BedrockPublisher::Anthropic => self.chat_anthropic(req, ctx, upstream_id).await,
            other => Err(BridgeError::Config(format!(
                "bedrock publisher {publisher:?} not yet implemented — \
                 tracked under api7/AISIX-Cloud#302 Phase G (D7.3+, publisher={})",
                other.name()
            ))),
        }
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
            "bedrock streaming is not yet implemented — \
             tracked under api7/AISIX-Cloud#302 Phase G (D7.2.b)"
                .into(),
        ))
    }
}

impl BedrockBridge {
    /// Dispatch Anthropic-on-Bedrock chat. Body shape per
    /// <https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html>:
    /// the Anthropic Messages JSON minus the `model` field (Bedrock
    /// keys dispatch off the URL) plus `anthropic_version:
    /// "bedrock-2023-05-31"`.
    async fn chat_anthropic(
        &self,
        req: &ChatFormat,
        ctx: &BridgeContext,
        upstream_id: &str,
    ) -> Result<ChatResponse, BridgeError> {
        // Parse credentials. Per-request to keep the bridge stateless
        // — credential rotation lands as soon as the PK snapshot
        // refreshes, no client cache invalidation needed.
        let creds = BedrockSecret::parse(&ctx.provider_key.secret)?;
        let endpoint_url = {
            #[cfg(test)]
            {
                self.endpoint_url_override
                    .as_deref()
                    .or(ctx.provider_key.api_base.as_deref())
            }
            #[cfg(not(test))]
            {
                ctx.provider_key.api_base.as_deref()
            }
        };
        let client = build_client(&creds, endpoint_url)?;

        // Build the Anthropic Messages body via the shared
        // serializers, then shape it for Bedrock:
        //   1. Strip `model` (Bedrock takes it via URL path)
        //   2. Strip `stream` (Bedrock decides via Invoke vs InvokeWithResponseStream)
        //   3. Add `anthropic_version` (Bedrock-specific pin)
        let (system, messages) =
            split_system(req).map_err(|e| BridgeError::Config(format!("{e}")))?;
        let anthropic_req = build_request(req, upstream_id, system, messages, false);
        let mut body_value = serde_json::to_value(&anthropic_req)
            .map_err(|e| BridgeError::Config(format!("serialize Anthropic request body: {e}")))?;
        if let Some(obj) = body_value.as_object_mut() {
            obj.remove("model");
            obj.remove("stream");
            obj.insert(
                "anthropic_version".to_string(),
                serde_json::Value::String(BEDROCK_ANTHROPIC_VERSION.to_string()),
            );
        }
        let body_bytes = serde_json::to_vec(&body_value).map_err(|e| {
            BridgeError::Config(format!("serialize Anthropic request body bytes: {e}"))
        })?;

        // Dispatch via the SDK. SigV4 + retries + content-type
        // headers are handled by the SDK; we pass model id +
        // accept/content-type + body bytes.
        let resp = client
            .invoke_model()
            .model_id(upstream_id)
            .content_type("application/json")
            .accept("application/json")
            .body(Blob::new(body_bytes))
            .send()
            .await
            .map_err(map_sdk_error)?;

        let parsed: AnthropicResponse = serde_json::from_slice(resp.body().as_ref())
            .map_err(|e| BridgeError::UpstreamDecode(e.to_string()))?;
        Ok(response_into_chat_response(parsed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Publisher resolution (preserved from skeleton) ───────────────

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
        assert_eq!(
            BedrockPublisher::from_model_id("anthropic.opus-4-1-20250805-v1:0"),
            Some(BedrockPublisher::Anthropic),
        );
    }

    #[test]
    fn publisher_resolves_meta_llama_variants() {
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
        assert_eq!(
            BedrockPublisher::from_model_id("amazon.nova-pro-v1:0"),
            Some(BedrockPublisher::AmazonNova),
        );
        assert_eq!(
            BedrockPublisher::from_model_id("amazon.titan-text-premier-v1:0"),
            Some(BedrockPublisher::AmazonTitan),
        );
    }

    #[test]
    fn publisher_resolves_cohere_command_r() {
        assert_eq!(
            BedrockPublisher::from_model_id("cohere.command-r-plus-v1:0"),
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
            BedrockPublisher::from_model_id("us.deepseek.r1-v1:0"),
            Some(BedrockPublisher::Other),
        );
    }

    #[test]
    fn publisher_does_not_strip_publisher_segment_as_region() {
        assert_eq!(
            BedrockPublisher::from_model_id("amazon.titan-text-premier-v1:0"),
            Some(BedrockPublisher::AmazonTitan),
        );
        assert_eq!(
            BedrockPublisher::from_model_id("cohere.command-r-v1:0"),
            Some(BedrockPublisher::Cohere),
        );
    }

    #[test]
    fn publisher_unknown_id_returns_none() {
        assert_eq!(BedrockPublisher::from_model_id("gpt-4o"), None);
        assert_eq!(BedrockPublisher::from_model_id(""), None);
        assert_eq!(
            BedrockPublisher::from_model_id("truly-unknown.foo-v1:0"),
            None,
        );
    }

    #[test]
    fn bridge_name_is_stable() {
        assert_eq!(BedrockBridge::new().name(), "bedrock");
    }

    // ─── BedrockSecret parsing ────────────────────────────────────────

    #[test]
    fn bedrock_secret_parses_full_form() {
        let json =
            r#"{"access_key_id":"AKIA-test","secret_access_key":"sk-test","region":"us-west-2"}"#;
        let s = BedrockSecret::parse(json).unwrap();
        assert_eq!(s.access_key_id, "AKIA-test");
        assert_eq!(s.secret_access_key, "sk-test");
        assert_eq!(s.region, "us-west-2");
        assert!(s.session_token.is_none());
    }

    #[test]
    fn bedrock_secret_parses_with_session_token() {
        let json = r#"{"access_key_id":"AKIA","secret_access_key":"sk","region":"us-west-2","session_token":"AQo..."}"#;
        let s = BedrockSecret::parse(json).unwrap();
        assert_eq!(s.session_token.as_deref(), Some("AQo..."));
    }

    #[test]
    fn bedrock_secret_rejects_empty() {
        let err = BedrockSecret::parse("").unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(
                    msg.contains("secret is empty"),
                    "must mention empty secret; got {msg}"
                );
                assert!(
                    msg.contains("access_key_id"),
                    "must hint at required JSON shape; got {msg}"
                );
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn bedrock_secret_rejects_non_json() {
        let err = BedrockSecret::parse("AKIA-test").unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(
                    msg.contains("must be valid JSON"),
                    "must mention JSON requirement; got {msg}"
                );
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    /// Audit M1: the error path must not echo the raw secret content
    /// — serde error messages include "invalid character X at
    /// position N" which reveals partial secret bytes.
    #[test]
    fn bedrock_secret_error_does_not_leak_secret_content() {
        let secret_with_distinctive_bytes = "X-DISTINCTIVE-LEAK-MARKER-Y";
        let err = BedrockSecret::parse(secret_with_distinctive_bytes).unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(
                    !msg.contains("X-DISTINCTIVE-LEAK-MARKER-Y"),
                    "error must NOT echo raw secret bytes; got {msg}"
                );
                assert!(
                    !msg.contains("DISTINCTIVE"),
                    "error must NOT leak partial secret bytes; got {msg}"
                );
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[test]
    fn bedrock_secret_rejects_missing_region() {
        // serde rejects missing required field — bridge surfaces
        // the generic shape-error, not the field name (defense in
        // depth against accidental field-name leakage to customer
        // error path; the operator-side schema docs say what's
        // required).
        let json = r#"{"access_key_id":"AKIA","secret_access_key":"sk"}"#;
        let err = BedrockSecret::parse(json).unwrap_err();
        assert!(matches!(err, BridgeError::Config(_)));
    }

    // ─── Pre-dispatch validation tests ─────────────────────────────────

    use aisix_core::{Model, ProviderKey};
    use aisix_gateway::ChatMessage;
    use std::sync::Arc;

    fn sample_model_with(model_name: &str) -> Arc<Model> {
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

    /// Build a PK with a valid Bedrock-shape secret. `endpoint_url`
    /// arg is the test-only override path — set this to a wiremock
    /// URI to drive `bridge.chat()` end-to-end.
    fn sample_pk_with_secret(secret_json: &str) -> Arc<ProviderKey> {
        let cfg = format!(
            r#"{{"display_name": "bedrock-prod", "secret": {}}}"#,
            serde_json::to_string(secret_json).unwrap()
        );
        Arc::new(serde_json::from_str(&cfg).unwrap())
    }

    fn valid_secret_json() -> &'static str {
        r#"{"access_key_id":"AKIA-test","secret_access_key":"sk-test","region":"us-west-2"}"#
    }

    #[tokio::test]
    async fn chat_rejects_unknown_publisher() {
        let bridge = BedrockBridge::new();
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("totally-bogus-model-id"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("customer-facing-name", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(msg.contains("bedrock publisher unknown"));
                assert!(msg.contains("totally-bogus-model-id"));
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_rejects_non_anthropic_publishers_with_publisher_named() {
        // Other Bedrock publishers are recognized but not yet wired
        // for dispatch — the error must call out which publisher
        // got rejected so the operator can pin the follow-up task.
        let bridge = BedrockBridge::new();
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("meta.llama3-3-70b-instruct-v1:0"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("customer-facing-name", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(msg.contains("not yet implemented"));
                assert!(
                    msg.contains("meta") || msg.contains("Meta"),
                    "publisher name must appear in error; got {msg}"
                );
                assert!(msg.contains("D7.3+") || msg.contains("Phase G"));
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_with_invalid_secret_errors_before_dispatch() {
        let bridge = BedrockBridge::new();
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("anthropic.claude-3-5-sonnet-20241022-v2:0"),
            sample_pk_with_secret("not-valid-json"),
        );
        let req = ChatFormat::new("customer-facing", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(msg.contains("must be valid JSON"));
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_with_empty_secret_errors_before_dispatch() {
        let bridge = BedrockBridge::new();
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("anthropic.claude-3-5-sonnet-20241022-v2:0"),
            sample_pk_with_secret(""),
        );
        let req = ChatFormat::new("customer-facing", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(msg.contains("secret is empty"));
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_with_missing_model_name_errors_before_dispatch() {
        let bridge = BedrockBridge::new();
        let pk = sample_pk_with_secret(valid_secret_json());
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
                assert!(msg.contains("model_name missing"));
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_ignores_req_model_and_uses_ctx_model_name() {
        // D6 audit HIGH-1 regression: dispatch must read upstream
        // id from ctx.model.model_name, NOT from req.model. We use
        // a non-anthropic publisher on the upstream id so the chat
        // call hits the publisher-not-implemented branch (proving
        // dispatch read model_name), not the publisher-unknown branch.
        let bridge = BedrockBridge::new();
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("meta.llama3-3-70b-instruct-v1:0"),
            sample_pk_with_secret(valid_secret_json()),
        );
        // req.model deliberately set to something the publisher
        // resolver would also reject if used as source of truth.
        let req = ChatFormat::new("totally-bogus-model-id", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(
                    msg.contains("not yet implemented"),
                    "must hit publisher-not-implemented (proving model_name was used); got {msg}"
                );
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_stream_returns_clear_not_implemented_error() {
        let bridge = BedrockBridge::new();
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("anthropic.claude-3-5-sonnet-20241022-v2:0"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("customer-facing", vec![ChatMessage::user("hi")]);
        let err = bridge.chat_stream(&req, &ctx).await.err().unwrap();
        match err {
            BridgeError::Config(msg) => {
                assert!(msg.contains("streaming is not yet implemented"));
                assert!(msg.contains("D7.2.b"));
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    // ─── Dispatch end-to-end against wiremock via endpoint_url override ──

    use wiremock::matchers::{method, path_regex};
    use wiremock::{Mock, MockServer, Request as MockRequest, Respond, ResponseTemplate};

    // Audit lesson from D6 PR #319: drive the **real**
    // `bridge.chat()` entry point via the `endpoint_url_override`
    // seam — credentials, region, SigV4 signing, body shaping all
    // run normally; only the destination host is rewritten to
    // wiremock.

    /// Recording responder: captures request body + headers so tests
    /// can assert what reached the wire. Always returns the canned
    /// default response — tests that need a custom response use the
    /// standard `ResponseTemplate` arg to `Mock::given(...).respond_with(...)`
    /// without capture (no need for both modes in one helper).
    #[derive(Clone, Default)]
    struct CapturingResponder {
        captured_body: std::sync::Arc<std::sync::Mutex<Option<serde_json::Value>>>,
        captured_headers: std::sync::Arc<std::sync::Mutex<Option<http::HeaderMap>>>,
    }

    impl Respond for CapturingResponder {
        fn respond(&self, req: &MockRequest) -> ResponseTemplate {
            let body: serde_json::Value = serde_json::from_slice(&req.body).unwrap_or_default();
            *self.captured_body.lock().unwrap() = Some(body);
            *self.captured_headers.lock().unwrap() = Some(req.headers.clone());
            default_anthropic_response_template()
        }
    }

    fn default_anthropic_response_template() -> ResponseTemplate {
        ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "msg_01",
            "model": "claude-3-5-sonnet-20241022-v2",
            "content": [{"type": "text", "text": "hello from bedrock"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 4}
        }))
    }

    #[tokio::test]
    async fn chat_anthropic_dispatches_via_invoke_model_url() {
        let server = MockServer::start().await;
        let responder = CapturingResponder::default();
        // Bedrock's InvokeModel URL: `/model/<urlencoded_id>/invoke`.
        // The `:` in `anthropic.claude-3-5-sonnet-20241022-v2:0` gets
        // percent-encoded to `%3A`; we use a regex to stay tolerant
        // across SDK version upgrades.
        Mock::given(method("POST"))
            .and(path_regex(
                r"^/model/anthropic\.claude-3-5-sonnet-20241022-v2(:0|%3A0)/invoke$",
            ))
            .respond_with(responder.clone())
            .expect(1)
            .mount(&server)
            .await;

        let bridge = BedrockBridge::new().with_endpoint_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("anthropic.claude-3-5-sonnet-20241022-v2:0"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("my-claude", vec![ChatMessage::user("hi")]);
        let chat = bridge.chat(&req, &ctx).await.unwrap();
        assert_eq!(chat.message.content, "hello from bedrock");
        assert_eq!(chat.usage.total_tokens, 9);
    }

    #[tokio::test]
    async fn chat_anthropic_body_contains_bedrock_anthropic_version_and_no_model_field() {
        let server = MockServer::start().await;
        let responder = CapturingResponder::default();
        Mock::given(method("POST"))
            .and(path_regex(r"^/model/.+/invoke$"))
            .respond_with(responder.clone())
            .expect(1)
            .mount(&server)
            .await;

        let bridge = BedrockBridge::new().with_endpoint_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("anthropic.claude-3-5-sonnet-20241022-v2:0"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("my-claude", vec![ChatMessage::user("hi")]);
        bridge.chat(&req, &ctx).await.unwrap();

        let body = responder.captured_body.lock().unwrap().clone().unwrap();
        // Bedrock-Anthropic body shape pins:
        //   1. `anthropic_version` MUST be present + the canonical
        //      `bedrock-2023-05-31` string (per AWS docs URL above).
        //   2. `model` MUST be absent — Bedrock dispatches off URL path.
        //   3. `stream` MUST be absent — InvokeModel is non-streaming;
        //      Bedrock would error on a stream:true with the wrong op.
        //   4. `messages` must be the translated user turn.
        assert_eq!(
            body.get("anthropic_version").and_then(|v| v.as_str()),
            Some("bedrock-2023-05-31"),
            "body must carry anthropic_version=bedrock-2023-05-31; body={body}"
        );
        assert!(
            body.get("model").is_none(),
            "body must NOT carry `model` (Bedrock dispatches via URL); body={body}"
        );
        assert!(
            body.get("stream").is_none(),
            "body must NOT carry `stream` (InvokeModel is non-streaming); body={body}"
        );
        let messages = body.get("messages").and_then(|v| v.as_array()).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(
            messages[0].get("role").and_then(|v| v.as_str()),
            Some("user")
        );
    }

    #[tokio::test]
    async fn chat_anthropic_uses_sigv4_authorization_header() {
        // The SDK signs with SigV4: `Authorization: AWS4-HMAC-SHA256 ...`.
        // This is a wire-level pin that the SDK actually signed (vs.
        // sending unauthenticated). If a future bug accidentally
        // bypassed the SDK, the canned auth header would change.
        let server = MockServer::start().await;
        let responder = CapturingResponder::default();
        Mock::given(method("POST"))
            .and(path_regex(r"^/model/.+/invoke$"))
            .respond_with(responder.clone())
            .expect(1)
            .mount(&server)
            .await;

        let bridge = BedrockBridge::new().with_endpoint_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("anthropic.claude-3-5-sonnet-20241022-v2:0"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("my-claude", vec![ChatMessage::user("hi")]);
        bridge.chat(&req, &ctx).await.unwrap();

        let headers = responder.captured_headers.lock().unwrap().clone().unwrap();
        let auth = headers
            .get("authorization")
            .and_then(|v: &http::HeaderValue| v.to_str().ok())
            .unwrap_or("");
        assert!(
            auth.starts_with("AWS4-HMAC-SHA256"),
            "expected AWS SigV4 Authorization header; got {auth:?}"
        );
        // The SDK must include x-amz-date for SigV4.
        assert!(
            headers.contains_key("x-amz-date"),
            "SigV4 requires x-amz-date; headers={headers:?}"
        );
        // Body hash header should be set by the SDK.
        assert!(
            headers.contains_key("x-amz-content-sha256") || headers.contains_key("content-length"),
            "expected x-amz-content-sha256 or content-length on a SigV4 request; got {headers:?}"
        );
    }

    #[tokio::test]
    async fn chat_anthropic_handles_tool_use_response_blocks() {
        // Anthropic on Bedrock returns `tool_use` content blocks for
        // tool-call responses. The bridge's reused
        // `response_into_chat_response` must translate them to
        // OpenAI's `tool_calls` shape so downstream agents work.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(r"^/model/.+/invoke$"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "msg_02",
                "model": "claude-3-5-sonnet-20241022-v2",
                "content": [
                    {"type": "text", "text": "calling tool"},
                    {
                        "type": "tool_use",
                        "id": "toolu_01abc",
                        "name": "get_weather",
                        "input": {"city": "SF"}
                    }
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 8, "output_tokens": 12}
            })))
            .expect(1)
            .mount(&server)
            .await;

        let bridge = BedrockBridge::new().with_endpoint_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("anthropic.claude-3-5-sonnet-20241022-v2:0"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("my-claude", vec![ChatMessage::user("hi")]);
        let chat = bridge.chat(&req, &ctx).await.unwrap();
        assert_eq!(chat.message.content, "calling tool");
        // Tool calls translated into OpenAI shape via the reused
        // anthropic crate's converter.
        let tool_calls = chat
            .message
            .extra
            .get("tool_calls")
            .unwrap()
            .as_array()
            .unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(
            tool_calls[0].get("type").and_then(|v| v.as_str()),
            Some("function")
        );
        assert_eq!(
            tool_calls[0]
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|v| v.as_str()),
            Some("get_weather")
        );
    }

    #[tokio::test]
    async fn chat_maps_upstream_4xx_to_canned_message_not_body_echo() {
        // Audit M1: Bedrock error envelopes can contain account
        // numbers (in ARNs), model IDs, IAM role names — must not
        // leak into customer-visible error.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(r"^/model/.+/invoke$"))
            .respond_with(
                ResponseTemplate::new(400).set_body_json(serde_json::json!({
                    "message": "Operation cannot be performed by IAM role arn:aws:iam::123456789012:role/internal-leaky-role"
                })),
            )
            .expect(1)
            .mount(&server)
            .await;

        let bridge = BedrockBridge::new().with_endpoint_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("anthropic.claude-3-5-sonnet-20241022-v2:0"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("my-claude", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::UpstreamStatus { message, .. } => {
                assert!(
                    !message.contains("123456789012") && !message.contains("internal-leaky-role"),
                    "upstream body must not leak account / role info into customer error; got {message:?}"
                );
            }
            other => panic!("expected UpstreamStatus, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_maps_upstream_429_to_canned_rate_limited() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(r"^/model/.+/invoke$"))
            .respond_with(ResponseTemplate::new(429).set_body_json(serde_json::json!({
                "message": "Too many requests for account 123456789012"
            })))
            .mount(&server)
            .await;

        let bridge = BedrockBridge::new().with_endpoint_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("anthropic.claude-3-5-sonnet-20241022-v2:0"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("my-claude", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::UpstreamStatus {
                status, message, ..
            } => {
                assert_eq!(status, 429);
                assert!(message.contains("rate limited"));
                assert!(
                    !message.contains("123456789012"),
                    "must not leak account id; got {message:?}"
                );
            }
            other => panic!("expected UpstreamStatus, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_with_cross_region_inference_profile_dispatches_correctly() {
        // The `us.` cross-region inference profile is a real Bedrock
        // routing detail — the publisher's wire shape is identical
        // regardless. Critical: the URL path must include the FULL
        // model id with the region prefix; only the publisher resolver
        // strips it.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(
                r"^/model/us\.anthropic\.claude-3-5-sonnet-20241022-v2(:0|%3A0)/invoke$",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "msg_xr", "model": "claude-3-5-sonnet",
                "content": [{"type": "text", "text": "cross-region ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1}
            })))
            .expect(1)
            .mount(&server)
            .await;

        let bridge = BedrockBridge::new().with_endpoint_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("us.anthropic.claude-3-5-sonnet-20241022-v2:0"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("my-claude", vec![ChatMessage::user("hi")]);
        let chat = bridge.chat(&req, &ctx).await.unwrap();
        assert_eq!(chat.message.content, "cross-region ok");
    }

    #[tokio::test]
    async fn chat_anthropic_translates_system_messages_to_system_field() {
        // Anthropic's Messages API takes `system` as a top-level
        // field, NOT a role in `messages[]`. The reused
        // `split_system` helper from aisix-provider-anthropic must
        // pull system turns out of the messages array into the
        // top-level `system` field.
        let server = MockServer::start().await;
        let responder = CapturingResponder::default();
        Mock::given(method("POST"))
            .and(path_regex(r"^/model/.+/invoke$"))
            .respond_with(responder.clone())
            .expect(1)
            .mount(&server)
            .await;

        let bridge = BedrockBridge::new().with_endpoint_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("anthropic.claude-3-5-sonnet-20241022-v2:0"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new(
            "my-claude",
            vec![
                ChatMessage::system("you are a helpful assistant"),
                ChatMessage::user("hi"),
            ],
        );
        bridge.chat(&req, &ctx).await.unwrap();

        let body = responder.captured_body.lock().unwrap().clone().unwrap();
        assert_eq!(
            body.get("system").and_then(|v| v.as_str()),
            Some("you are a helpful assistant"),
            "system role must become top-level `system` field; body={body}"
        );
        let messages = body.get("messages").and_then(|v| v.as_array()).unwrap();
        assert_eq!(
            messages.len(),
            1,
            "system role must NOT appear in messages[]; body={body}"
        );
        assert_eq!(
            messages[0].get("role").and_then(|v| v.as_str()),
            Some("user")
        );
    }
}
