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
//! Credentials: `ProviderKey.api_key` is a JSON-encoded
//! `{access_key_id, secret_access_key, session_token?, region}`
//! struct. The bridge parses it per request (cheap — strings only)
//! and constructs a per-call SDK client. `ProviderKey.api_base` (if
//! set) is forwarded as the SDK's `endpoint_url` so operators can
//! point at a private deployment / VPC endpoint.

use aisix_gateway::{
    Bridge, BridgeContext, BridgeError, ChatChunk, ChatChunkStream, ChatDelta, ChatFormat,
    ChatMessage, ChatResponse, EmbeddingObject, EmbeddingRequest, EmbeddingResponse,
    EmbeddingUsage, EmbeddingVector, FinishReason, Role, UsageStats,
};
use async_trait::async_trait;
use aws_credential_types::provider::SharedCredentialsProvider;
use aws_credential_types::Credentials;
use aws_sdk_bedrockruntime::config::{BehaviorVersion, Region};
use aws_sdk_bedrockruntime::error::SdkError;
use aws_sdk_bedrockruntime::operation::converse::ConverseError;
use aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamError;
use aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError;
use aws_sdk_bedrockruntime::primitives::Blob;
use aws_sdk_bedrockruntime::types::{
    AnyToolChoice, ContentBlock, ContentBlockDelta, ContentBlockStart, ConversationRole,
    ConverseStreamOutput, InferenceConfiguration, Message as BedrockMessage, SpecificToolChoice,
    StopReason as SdkStopReason, SystemContentBlock, Tool, ToolChoice, ToolConfiguration,
    ToolInputSchema, ToolResultBlock, ToolResultContentBlock, ToolSpecification, ToolUseBlock,
};
use aws_sdk_bedrockruntime::Client as BedrockClient;
use aws_smithy_runtime_api::client::result::ServiceError;
use aws_smithy_types::error::metadata::ProvideErrorMetadata;
use serde::Deserialize;
use std::time::{Duration, Instant};

use aisix_provider_anthropic::wire::{
    build_request, response_into_chat_response, split_system, AnthropicResponse,
};

// Per-`ProviderKey` request override pipeline (#302 §5 / #340). The JSON-body
// transforms reuse the shared primitives the OpenAI / Vertex bridges call;
// `default_headers` ride a pre-signing interceptor (see
// [`DefaultHeadersInterceptor`]) so they land inside the SigV4-signed
// canonical request rather than being appended after the signature is computed.
use aisix_core::{ParamConstraints, RequestOverrides};
use aisix_provider_openai::overrides::{
    apply_content_list_to_string, apply_default_body_fields, apply_param_constraints,
    apply_param_renames,
};
use aws_sdk_bedrockruntime::config::interceptors::BeforeTransmitInterceptorContextMut;
use aws_sdk_bedrockruntime::config::{ConfigBag, Intercept, RuntimeComponents};
use aws_smithy_runtime_api::box_error::BoxError;

use crate::convert::{document_to_json, json_to_document};
use crate::wire;

/// Anthropic-on-Bedrock body-shape version pin per
/// <https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html>.
/// Used by the legacy /invoke Anthropic non-stream path
/// ([`BedrockBridge::chat_anthropic`]); the Converse path doesn't
/// need it (the SDK shapes the body).
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
    /// cross-region inference profile prefixes (`us.`, `eu.`, `apac.`,
    /// `global.`, `us-gov.`, `au.`, `ca.`, `jp.`, …) — see
    /// [`strip_region_prefix`] for the exact rule.
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

    /// Human-readable name. Was used by the old "publisher-not-
    /// implemented" error path; kept after Phase G Step 3's Converse
    /// dispatch wiring made the legacy path obsolete because the same
    /// label is still useful for telemetry tagging follow-ups.
    #[allow(dead_code)]
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
/// Rather than hardcode AWS's region list (which grows over time),
/// this strips any leading 2–7 char `[a-z0-9-]` token that is
/// immediately followed by a known publisher tag. That covers every
/// documented cross-region prefix without a code change when AWS adds
/// a new region: `us.`, `eu.`, `apac.`, `global.`, `us-gov.`, `au.`
/// (Australia), `ca.` (Canada), `jp.` (Japan), … The publisher-tag
/// gate means a token that isn't a real prefix (no known publisher
/// after it) is left untouched, so a plain `anthropic.claude-…` id is
/// never mangled.
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

/// Schema for `ProviderKey.api_key` on a Bedrock provider key.
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
            return Err(BridgeError::InvalidUpstreamCredentials(
                "bedrock provider_key.api_key is empty — \
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
            BridgeError::InvalidUpstreamCredentials(
                "bedrock provider_key.api_key must be valid JSON: \
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
    request: Option<&RequestOverrides>,
) -> Result<BedrockClient, BridgeError> {
    if creds.region.trim().is_empty() {
        return Err(BridgeError::InvalidUpstreamConfig(
            "bedrock provider_key.api_key.region is empty — \
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

    // Register the default-headers interceptor only when the PK actually
    // carries (non-reserved) default_headers, so the common no-override path
    // builds the client byte-for-byte as before. The interceptor injects at
    // `modify_before_signing`, so the headers are covered by the SigV4
    // signature (#340).
    let mut conf = aws_sdk_bedrockruntime::config::Builder::from(&sdk_cfg);
    if let Some(r) = request {
        let headers = filtered_default_headers(&r.default_headers);
        if !headers.is_empty() {
            conf = conf.interceptor(DefaultHeadersInterceptor { headers });
        }
    }
    Ok(BedrockClient::from_conf(conf.build()))
}

/// Drop SigV4-owned headers ([`wire::reserved_sigv4_headers`]) from an
/// operator-supplied `default_headers` block before they reach the signing
/// interceptor. cp-api SHOULD reject these at write time (#302 §5), but the DP
/// enforces it again here as defense-in-depth — an override naming e.g.
/// `x-amz-date` or `authorization` must never perturb the signature. Matching
/// is case-insensitive (HTTP header names are).
fn filtered_default_headers(
    defaults: &std::collections::HashMap<String, String>,
) -> Vec<(String, String)> {
    let reserved = wire::reserved_sigv4_headers();
    defaults
        .iter()
        .filter(|(name, _)| {
            let lower = name.to_ascii_lowercase();
            !reserved.contains(&lower.as_str())
        })
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect()
}

/// Apply the per-`ProviderKey` request/response **body** override pipeline to
/// an already-serialized JSON body, mirroring `OpenAiBridge`'s order:
/// `param_renames` -> `param_constraints` -> `default_body_fields` ->
/// (`content_list_to_string`, when the response override sets it). Header
/// overrides are NOT applied here — Bedrock signs via the AWS SDK, so
/// `default_headers` ride [`DefaultHeadersInterceptor`] instead.
///
/// Only the `invoke_model` (Anthropic) path carries a JSON body; the Converse
/// path builds its request through the SDK's typed `InferenceConfiguration`
/// builder, which has no JSON body for these top-level-key transforms to act
/// on. (`param_constraints`'s temperature clamp IS still re-applied on the
/// Converse path via `build_inference_config` — see #463 — but `param_renames`
/// / `default_body_fields` / `content_list_to_string` have no Converse target.)
fn apply_body_overrides(body: &mut serde_json::Value, ctx: &BridgeContext) {
    if let Some(r) = ctx.provider_key.request.as_ref() {
        apply_param_renames(body, &r.param_renames);
        if let Some(constraints) = &r.param_constraints {
            apply_param_constraints(body, constraints);
        }
        apply_default_body_fields(body, &r.default_body_fields);
    }
    if ctx
        .provider_key
        .response
        .as_ref()
        .is_some_and(|r| r.content_list_to_string)
    {
        apply_content_list_to_string(body);
    }
}

/// Injects `request.default_headers` into the outbound Bedrock request at
/// `modify_before_signing`, so the headers are part of the SigV4-signed
/// canonical request (#302 §5 / #340). The header list is pre-filtered by
/// [`filtered_default_headers`] (SigV4-owned names removed) at construction.
/// A header already present on the request is left untouched (the SDK / caller
/// wins); names or values that fail header parsing are skipped silently — the
/// block came from cp-api validation and an unparseable entry is a config
/// error one layer up, not a runtime failure the dispatch should hard-fail on.
#[derive(Debug)]
struct DefaultHeadersInterceptor {
    headers: Vec<(String, String)>,
}

impl Intercept for DefaultHeadersInterceptor {
    fn name(&self) -> &'static str {
        "AisixBedrockDefaultHeaders"
    }

    fn modify_before_signing(
        &self,
        context: &mut BeforeTransmitInterceptorContextMut<'_>,
        _runtime_components: &RuntimeComponents,
        _cfg: &mut ConfigBag,
    ) -> Result<(), BoxError> {
        let request = context.request_mut();
        for (name, value) in &self.headers {
            if request.headers().contains_key(name.as_str()) {
                continue;
            }
            // `try_insert` returns Err on a non-ASCII name / non-UTF-8 value;
            // we already guarded `contains_key`, so a parse failure just skips
            // this one entry rather than failing the whole request.
            let _ = request
                .headers_mut()
                .try_insert(name.clone(), value.clone());
        }
        Ok(())
    }
}

/// Pull the upstream model id off the BridgeContext.
fn upstream_model(ctx: &BridgeContext) -> Result<&str, BridgeError> {
    ctx.model
        .model_name
        .as_deref()
        .ok_or_else(|| BridgeError::InvalidUpstreamConfig("model.model_name missing".into()))
}

/// Translate an SDK error into the canonical `BridgeError`.
///
/// **Audit M1 — sensitive-info redaction:** Bedrock error envelopes
/// frequently include the operator's model id, region, account
/// numbers (in ARNs), and IAM role names. Surfacing these to a
/// downstream customer leaks operator-internal taxonomy. We map to
/// canned status-keyed phrases.
///
/// **Audit H3** — `deadline` is threaded through so a SDK-side timeout
/// reports the actual elapsed budget instead of `0ms` (which formats
/// as "timed out after 0ms" in customer logs).
fn map_sdk_error(
    err: SdkError<InvokeModelError>,
    started: Instant,
    deadline: Option<Duration>,
) -> BridgeError {
    match err {
        SdkError::TimeoutError(_) => {
            // Prefer the actual elapsed budget; fall back to the
            // deadline if elapsed somehow rounds to 0 (clock skew).
            let elapsed_ms = started.elapsed().as_millis() as u64;
            let reported = if elapsed_ms > 0 {
                elapsed_ms
            } else {
                deadline.map(|d| d.as_millis() as u64).unwrap_or(0)
            };
            BridgeError::Timeout {
                elapsed_ms: reported,
                cause: String::new(),
            }
        }
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

/// Audit H1 — propagate `Retry-After` from the upstream's HTTP
/// response so the gateway's cooldown layer gets the actual upstream
/// hint instead of falling back to its configured default. Bedrock
/// returns `Retry-After` on 429 throttle responses; collapsing it to
/// `None` silently degrades multi-region / burst behavior.
fn map_service_error(
    svc: ServiceError<InvokeModelError, aws_smithy_runtime_api::http::Response>,
) -> BridgeError {
    // SECURITY: AWS error messages embed operator-internal taxonomy
    // (ARNs, region, account id, IAM role names). The canned status-
    // keyed phrase reaches the customer; the parsed view surfaces only
    // the AWS error CODE (e.g. "ThrottlingException") for the
    // error_translate layer to translate to OpenAI / Anthropic shape.
    // `parsed.message` is intentionally left `None`.
    let kind = svc.err().meta().code().map(str::to_string);
    let raw = svc.raw();
    let status = raw.status().as_u16();
    // Convert smithy HeaderMap → http::HeaderMap so we can reuse the
    // gateway-level `parse_retry_after` helper. Headers with invalid
    // bytes are dropped (defensive — SDK should not produce them).
    let mut hdrs = http::HeaderMap::new();
    for (k, v) in raw.headers() {
        if let (Ok(name), Ok(val)) = (
            http::HeaderName::from_bytes(k.as_bytes()),
            http::HeaderValue::from_str(v),
        ) {
            hdrs.insert(name, val);
        }
    }
    let retry_after = aisix_gateway::parse_retry_after(&hdrs);
    let message = match status {
        401 | 403 => "upstream authentication failed".to_string(),
        404 => "upstream model not found".to_string(),
        408 => "upstream request timeout".to_string(),
        429 => "upstream rate limited".to_string(),
        _ => format!("upstream returned {status}"),
    };
    let parsed = kind.as_ref().map(|k| {
        Box::new(aisix_gateway::UpstreamErrorView {
            kind: Some(k.clone()),
            message: None,
            code: None,
            param: None,
        })
    });
    BridgeError::UpstreamStatus {
        status,
        message,
        parsed,
        wire: aisix_gateway::UpstreamWire::Bedrock,
        retry_after,
    }
}

/// **Audit M2** — defense-in-depth check on the upstream model id
/// before it's URL-encoded into the Bedrock `/model/<id>/invoke`
/// path. The SDK encodes reserved characters, but pinning the
/// allowed set at the gateway layer prevents log-injection /
/// dashboard-label corruption (the model id propagates into metrics
/// labels) and forces typos to fail loudly at registration time.
///
/// Bedrock model ids are documented as
/// `<publisher>.<family>-<version>:<revision>` with all-ASCII tokens.
fn validate_model_id_chars(model_id: &str) -> Result<(), BridgeError> {
    if !model_id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | ':' | '_' | '/'))
    {
        return Err(BridgeError::Config(format!(
            "bedrock model id {model_id:?} contains unexpected characters — \
             only [A-Za-z0-9._:/-] are allowed"
        )));
    }
    Ok(())
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
        validate_model_id_chars(upstream_id)?;
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

        // Phase G productionization (#302 Step 3): unified Converse
        // wire for non-Anthropic publishers (Meta / Mistral / Cohere /
        // Amazon Titan / Amazon Nova / AI21). Anthropic stays on the
        // legacy /invoke path (chat_anthropic, wired via #320) for
        // backward compat with existing operator deployments — the
        // Anthropic Messages JSON envelope is what cp-api / dashboard
        // tests already pin. Moving Anthropic to Converse is a clean
        // follow-up since the SDK-decoded Converse response shape is
        // equivalent for the customer-visible ChatResponse.
        //
        // chat_stream (below) dispatches ALL publishers through
        // Converse because the legacy /invoke path never had a
        // streaming variant (D7.2.b was the gap).
        match publisher {
            BedrockPublisher::Anthropic => self.chat_anthropic(req, ctx, upstream_id).await,
            _ => self.chat_converse(req, ctx, upstream_id).await,
        }
    }

    async fn chat_stream(
        &self,
        req: &ChatFormat,
        ctx: &BridgeContext,
    ) -> Result<ChatChunkStream, BridgeError> {
        let upstream_id = upstream_model(ctx)?;
        validate_model_id_chars(upstream_id)?;
        let publisher = BedrockPublisher::from_model_id(upstream_id).ok_or_else(|| {
            BridgeError::Config(format!(
                "bedrock publisher unknown for model id {upstream_id:?}; \
                 expected one of anthropic.claude-* / meta.llama* / mistral.* / \
                 amazon.titan-* / amazon.nova-* / cohere.command* / ai21.jamba-* \
                 (optionally prefixed with a cross-region inference profile like us. / eu. / apac.)"
            ))
        })?;
        // Phase G productionization (#302 Step 3): unified Converse
        // stream path for all publishers — same SDK call (.converse_stream)
        // owns the AWS event-stream binary frame decoding internally,
        // so the bridge layer doesn't ship its own decoder.
        let _ = publisher;
        self.chat_converse_stream(req, ctx, upstream_id).await
    }

    /// Native Bedrock embeddings (#723), LiteLLM `litellm/llms/bedrock/embed/`
    /// parity: `amazon.titan-embed-*` (Titan G1/V2 — the model embeds ONE
    /// text per InvokeModel, so batch inputs loop; LiteLLM iterates the
    /// same way) and `cohere.embed-*` (whole batch in one call). Other
    /// publishers expose no embedding models on Bedrock.
    async fn embed(
        &self,
        req: &EmbeddingRequest,
        ctx: &BridgeContext,
    ) -> Result<EmbeddingResponse, BridgeError> {
        let upstream_id = upstream_model(ctx)?;
        validate_model_id_chars(upstream_id)?;
        let client = self.build_client_from_ctx(ctx)?;
        let started = Instant::now();
        let deadline = ctx.deadline;

        let mut data = Vec::with_capacity(req.input.len());
        let mut prompt_tokens: u64 = 0;

        let base_id = strip_region_prefix(upstream_id);
        if base_id.starts_with("amazon.titan-embed") {
            for (i, text) in req.input.iter().enumerate() {
                let mut body = serde_json::json!({"inputText": text});
                // `dimensions` is a Titan V2 knob; G1 rejects unknown keys.
                if let Some(dims) = req.dimensions {
                    if base_id.contains("v2") {
                        body["dimensions"] = dims.into();
                    }
                }
                let body_bytes = serde_json::to_vec(&body)
                    .map_err(|e| BridgeError::Config(format!("serialize Titan embed body: {e}")))?;
                let resp = client
                    .invoke_model()
                    .model_id(upstream_id)
                    .content_type("application/json")
                    .accept("application/json")
                    .body(Blob::new(body_bytes))
                    .send()
                    .await
                    .map_err(|e| map_sdk_error(e, started, deadline))?;
                let parsed: TitanEmbedResponse = serde_json::from_slice(resp.body().as_ref())
                    .map_err(|e| BridgeError::UpstreamDecode(e.to_string()))?;
                prompt_tokens += parsed.input_text_token_count;
                data.push(EmbeddingObject {
                    index: i as u32,
                    object: "embedding".to_string(),
                    embedding: EmbeddingVector::Float(parsed.embedding),
                });
            }
        } else if base_id.starts_with("cohere.embed") {
            let body = serde_json::json!({
                "texts": req.input,
                // LiteLLM's default when the caller gives no input_type.
                "input_type": "search_document",
            });
            let body_bytes = serde_json::to_vec(&body)
                .map_err(|e| BridgeError::Config(format!("serialize Cohere embed body: {e}")))?;
            let resp = client
                .invoke_model()
                .model_id(upstream_id)
                .content_type("application/json")
                .accept("application/json")
                .body(Blob::new(body_bytes))
                .send()
                .await
                .map_err(|e| map_sdk_error(e, started, deadline))?;
            let parsed: CohereEmbedResponse = serde_json::from_slice(resp.body().as_ref())
                .map_err(|e| BridgeError::UpstreamDecode(e.to_string()))?;
            for (i, values) in parsed.embeddings.into_iter().enumerate() {
                data.push(EmbeddingObject {
                    index: i as u32,
                    object: "embedding".to_string(),
                    embedding: EmbeddingVector::Float(values),
                });
            }
            // Cohere-on-Bedrock returns no token statistics; usage stays 0
            // (LiteLLM surfaces the same absence).
        } else {
            return Err(BridgeError::Config(format!(
                "bedrock embeddings support amazon.titan-embed-* and cohere.embed-* \
                 model ids; got {upstream_id:?}"
            )));
        }

        let tokens = prompt_tokens.min(u32::MAX as u64) as u32;
        Ok(EmbeddingResponse {
            object: "list".to_string(),
            model: req.model.clone(),
            data,
            usage: EmbeddingUsage {
                prompt_tokens: tokens,
                total_tokens: tokens,
            },
        })
    }
}

/// Titan embed response per the Amazon Titan Embeddings docs
/// (`embedding` + `inputTextTokenCount`).
#[derive(Debug, Deserialize)]
struct TitanEmbedResponse {
    embedding: Vec<f32>,
    #[serde(rename = "inputTextTokenCount", default)]
    input_text_token_count: u64,
}

/// Cohere-on-Bedrock embed response (`embeddings` = one vector per text).
#[derive(Debug, Deserialize)]
struct CohereEmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

impl BedrockBridge {
    /// Dispatch Anthropic-on-Bedrock chat via the legacy /invoke
    /// path. Body shape per
    /// <https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html>:
    /// the Anthropic Messages JSON minus the `model` field (Bedrock
    /// keys dispatch off the URL) plus `anthropic_version:
    /// "bedrock-2023-05-31"`. Kept alongside `chat_converse` because
    /// the existing customer deployments + e2e tests pin the
    /// Anthropic-shape outbound body; future PR can fold this into
    /// the Converse path once cp-api / dashboard tests have been
    /// updated to assert the Converse envelope instead.
    async fn chat_anthropic(
        &self,
        req: &ChatFormat,
        ctx: &BridgeContext,
        upstream_id: &str,
    ) -> Result<ChatResponse, BridgeError> {
        let client = self.build_client_from_ctx(ctx)?;

        let (system, messages) =
            split_system(req).map_err(|e| BridgeError::InvalidUpstreamConfig(format!("{e}")))?;
        let anthropic_req = build_request(req, upstream_id, system, messages, false);
        let mut body_value = serde_json::to_value(&anthropic_req)
            .map_err(|e| BridgeError::Config(format!("serialize Anthropic request body: {e}")))?;
        // Apply the per-ProviderKey body override pipeline (#340) BEFORE the
        // Vertex-style Bedrock shaping below, so the `model`/`stream` strip
        // keeps the final say and an override can never reintroduce a
        // URL-borne `model` into the /invoke body.
        apply_body_overrides(&mut body_value, ctx);
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

        let started = Instant::now();
        let deadline = ctx.deadline;
        let resp = client
            .invoke_model()
            .model_id(upstream_id)
            .content_type("application/json")
            .accept("application/json")
            .body(Blob::new(body_bytes))
            .send()
            .await
            .map_err(|e| map_sdk_error(e, started, deadline))?;

        let parsed: AnthropicResponse = serde_json::from_slice(resp.body().as_ref())
            .map_err(|e| BridgeError::UpstreamDecode(e.to_string()))?;
        Ok(response_into_chat_response(parsed))
    }

    /// Resolve credentials + build an SDK client. Pulled out of
    /// `chat_converse` / `chat_converse_stream` because both methods
    /// need it and the body is identical.
    fn build_client_from_ctx(&self, ctx: &BridgeContext) -> Result<BedrockClient, BridgeError> {
        // Parse credentials per-request to keep the bridge stateless —
        // credential rotation lands as soon as the PK snapshot
        // refreshes, no client cache invalidation needed.
        let creds = BedrockSecret::parse(&ctx.provider_key.api_key)?;
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
        // Converse paths carry no JSON body, but default_headers still apply
        // (header-level, publisher-agnostic) via the signing interceptor.
        build_client(&creds, endpoint_url, ctx.provider_key.request.as_ref())
    }

    /// Dispatch Bedrock chat via the unified Converse API.
    ///
    /// Request: `POST /model/<modelId>/converse`. The SDK builds the
    /// Converse JSON envelope (`messages[]`, `system[]`,
    /// `inferenceConfig`) from typed builders, signs SigV4, and POSTs
    /// to the resolved endpoint. Response: typed `ConverseOutput` with
    /// `output.message.content[]` text blocks, `stop_reason`, `usage`
    /// (input/output/total tokens), and `metrics.latency_ms`.
    ///
    /// Unlike the per-publisher `/invoke` legacy path, Converse uses
    /// ONE wire shape for every Bedrock publisher (Anthropic, Meta,
    /// Mistral, Cohere, Amazon Titan/Nova, AI21). Adding a new
    /// publisher to AWS's supported list requires zero gateway-side
    /// code change — the SDK upgrade picks it up.
    ///
    /// Reference: <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html>
    async fn chat_converse(
        &self,
        req: &ChatFormat,
        ctx: &BridgeContext,
        upstream_id: &str,
    ) -> Result<ChatResponse, BridgeError> {
        let client = self.build_client_from_ctx(ctx)?;
        let (system_blocks, message_blocks) = build_converse_inputs(req)?;
        if message_blocks.is_empty() {
            return Err(BridgeError::Config(
                "bedrock converse: messages must include at least one user / \
                 assistant turn (system-only requests are not supported by Converse)"
                    .into(),
            ));
        }

        let started = Instant::now();
        let deadline = ctx.deadline;
        let mut call = client.converse().model_id(upstream_id);
        for sb in system_blocks {
            call = call.system(sb);
        }
        for mb in message_blocks {
            call = call.messages(mb);
        }
        // Audit MEDIUM-2 (PR #389): wire ChatFormat's
        // temperature/max_tokens/top_p through to Converse's
        // InferenceConfiguration. The legacy chat_anthropic path
        // forwards these via build_request → AnthropicRequest;
        // dropping them on the Converse path is a silent
        // behavioural regression for every non-Anthropic customer.
        // #463: re-apply the PK's param_constraints temperature clamp here —
        // only the /invoke body path runs apply_param_constraints otherwise,
        // so without this a Bedrock-wide temperature ceiling silently no-ops
        // for every Converse (non-Anthropic) publisher.
        if let Some(cfg) = build_inference_config(req, pk_param_constraints(ctx)) {
            call = call.inference_config(cfg);
        }
        // #560: forward OpenAI `tools` / `tool_choice` into Converse's
        // `toolConfig`. Without this every Converse publisher silently
        // drops tool calling and improvises the call as prose
        // (finish_reason=stop, tool_calls=[]).
        if let Some(tc) = build_tool_config(req) {
            call = call.tool_config(tc);
        }

        let resp = call
            .send()
            .await
            .map_err(|e| map_converse_sdk_error(e, started, deadline))?;
        Ok(converse_output_into_chat_response(resp, upstream_id))
    }

    /// Dispatch Bedrock chat via the unified Converse stream API.
    ///
    /// Request: `POST /model/<modelId>/converse-stream`. The SDK
    /// reads the binary `vnd.amazon.eventstream` body internally and
    /// yields typed [`ConverseStreamOutput`] events through the
    /// returned stream — no per-byte frame decoder lives in the
    /// bridge layer.
    ///
    /// Event sequence per AWS docs:
    ///
    /// ```text
    /// MessageStart      → {role: "assistant"}
    /// ContentBlockStart → {start: {text: ""}, contentBlockIndex: 0}
    /// ContentBlockDelta → {delta: {text: "..."}, contentBlockIndex: 0}
    /// ContentBlockStop  → {contentBlockIndex: 0}
    /// MessageStop       → {stopReason: "end_turn"}
    /// Metadata          → {usage: {...}, metrics: {latencyMs: ...}}
    /// ```
    ///
    /// Each event maps to zero, one, or two [`ChatChunk`]s — see
    /// [`emit_converse_chunk`]. Stream closes cleanly when AWS sends
    /// the final Metadata event and the underlying eventstream
    /// connection terminates.
    ///
    /// Reference: <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html>
    async fn chat_converse_stream(
        &self,
        req: &ChatFormat,
        ctx: &BridgeContext,
        upstream_id: &str,
    ) -> Result<ChatChunkStream, BridgeError> {
        let client = self.build_client_from_ctx(ctx)?;
        let (system_blocks, message_blocks) = build_converse_inputs(req)?;
        if message_blocks.is_empty() {
            return Err(BridgeError::Config(
                "bedrock converse: messages must include at least one user / \
                 assistant turn (system-only requests are not supported by Converse)"
                    .into(),
            ));
        }

        let started = Instant::now();
        let deadline = ctx.deadline;
        let mut call = client.converse_stream().model_id(upstream_id);
        for sb in system_blocks {
            call = call.system(sb);
        }
        for mb in message_blocks {
            call = call.messages(mb);
        }
        // Mirror chat_converse on the stream path — same audit fix +
        // the #463 param_constraints temperature clamp.
        if let Some(cfg) = build_inference_config(req, pk_param_constraints(ctx)) {
            call = call.inference_config(cfg);
        }
        // #560: forward tools on the stream path too (all publishers,
        // incl. Anthropic, stream through Converse).
        if let Some(tc) = build_tool_config(req) {
            call = call.tool_config(tc);
        }

        let mut resp = call
            .send()
            .await
            .map_err(|e| map_converse_stream_sdk_error(e, started, deadline))?;

        let upstream_id_owned = upstream_id.to_string();
        let stream = async_stream::try_stream! {
            let mut emitted_role = false;
            // #560: maps each tool-use block's Converse `contentBlockIndex`
            // to a dense 0-based OpenAI `tool_calls[].index` (a preceding
            // text block at index 0 would otherwise shift every tool call).
            let mut tool_blocks: Vec<i32> = Vec::new();
            loop {
                match resp.stream.recv().await {
                    Ok(Some(event)) => {
                        for chunk in emit_converse_chunk(
                            event,
                            &upstream_id_owned,
                            &mut emitted_role,
                            &mut tool_blocks,
                        ) {
                            yield chunk;
                        }
                    }
                    Ok(None) => break,
                    Err(e) => {
                        // SDK already maps eventstream-decode and
                        // transport failures into its error type.
                        // Surface as Transport to flow through the
                        // existing customer-visible error envelope.
                        Err(BridgeError::Transport(format!(
                            "bedrock converse-stream: {e}"
                        )))?;
                    }
                }
            }
        };
        Ok(Box::pin(stream))
    }
}

/// Translate the gateway [`ChatFormat`] into the SDK's typed
/// `(Vec<SystemContentBlock>, Vec<Message>)` Converse inputs.
///
/// System messages → top-level `system[]` blocks (Converse splits
/// them out of `messages[]` per AWS spec).
/// User messages → `messages[]` with a text content block.
/// Assistant messages → a text block plus a `toolUse` block per OpenAI
/// `tool_calls` entry (#560, multi-turn history).
/// `role:"tool"` results → `toolResult` blocks carried in a user
/// message; consecutive results coalesce into one (Converse requires all
/// of a turn's results in a single user message). Bedrock Converse
/// rejects empty `messages[]` so the caller must check the returned
/// `Vec` length.
fn build_converse_inputs(
    req: &ChatFormat,
) -> Result<(Vec<SystemContentBlock>, Vec<BedrockMessage>), BridgeError> {
    let mut systems: Vec<SystemContentBlock> = Vec::new();
    let mut messages: Vec<BedrockMessage> = Vec::new();
    // #560: buffer for coalescing consecutive `role:"tool"` results
    // (parallel tool calls) into ONE Converse user message — Converse
    // rejects two consecutive user messages.
    let mut pending_tool_results: Vec<ContentBlock> = Vec::new();

    for msg in &req.messages {
        // Flush buffered tool results as one user message before any
        // non-tool message opens.
        if !matches!(msg.role, Role::Tool) && !pending_tool_results.is_empty() {
            messages.push(build_message(
                ConversationRole::User,
                std::mem::take(&mut pending_tool_results),
            )?);
        }
        match msg.role {
            Role::System => {
                let content = msg.content_str();
                if !content.is_empty() {
                    systems.push(SystemContentBlock::Text(content.to_string()));
                }
            }
            Role::User => {
                messages.push(build_message(
                    ConversationRole::User,
                    vec![ContentBlock::Text(msg.content_str().to_string())],
                )?);
            }
            Role::Assistant => {
                messages.push(build_message(
                    ConversationRole::Assistant,
                    assistant_content_blocks(msg)?,
                )?);
            }
            Role::Tool => {
                // #560: `role:"tool"` → Converse `toolResult` block. The
                // correlation id rides `tool_call_id`, the result text
                // rides `content`. Without a `tool_call_id` the result
                // can't be correlated to a `toolUse`, so skip it.
                if let Some(id) = msg.tool_call_id.as_deref() {
                    pending_tool_results.push(ContentBlock::ToolResult(
                        ToolResultBlock::builder()
                            .tool_use_id(id)
                            .content(ToolResultContentBlock::Text(msg.content_str().to_string()))
                            .build()
                            .map_err(|e| {
                                BridgeError::Config(format!(
                                    "bedrock converse: build tool_result: {e}"
                                ))
                            })?,
                    ));
                }
            }
        }
    }
    // Flush any trailing tool results (a conversation ending on a result).
    if !pending_tool_results.is_empty() {
        messages.push(build_message(ConversationRole::User, pending_tool_results)?);
    }
    Ok((systems, messages))
}

/// Build a Converse [`BedrockMessage`] from a role and its content blocks.
fn build_message(
    role: ConversationRole,
    blocks: Vec<ContentBlock>,
) -> Result<BedrockMessage, BridgeError> {
    let mut builder = BedrockMessage::builder().role(role);
    for block in blocks {
        builder = builder.content(block);
    }
    builder
        .build()
        .map_err(|e| BridgeError::Config(format!("bedrock converse: build message: {e}")))
}

/// Translate an assistant [`ChatMessage`] into Converse content blocks:
/// the assistant text (when non-empty) followed by a `toolUse` block per
/// OpenAI `tool_calls` entry (#560, multi-turn history).
///
/// When the assistant carried no tool calls this returns a single text
/// block — byte-identical to the pre-#560 behaviour.
fn assistant_content_blocks(msg: &ChatMessage) -> Result<Vec<ContentBlock>, BridgeError> {
    let tool_calls = msg
        .extra
        .get("tool_calls")
        .and_then(|v| v.as_array())
        .filter(|calls| !calls.is_empty());
    let Some(tool_calls) = tool_calls else {
        // No tool calls: preserve the original single-text-block shape.
        return Ok(vec![ContentBlock::Text(msg.content_str().to_string())]);
    };

    let mut blocks: Vec<ContentBlock> = Vec::new();
    let text = msg.content_str();
    if !text.is_empty() {
        blocks.push(ContentBlock::Text(text.to_string()));
    }
    for call in tool_calls {
        let function = call.get("function");
        let id = call.get("id").and_then(|v| v.as_str()).unwrap_or_default();
        let name = function
            .and_then(|f| f.get("name"))
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        if id.is_empty() || name.is_empty() {
            continue; // malformed history entry — can't build a toolUse
        }
        // OpenAI `arguments` is a JSON-ENCODED STRING; parse it back to a
        // value for the Document. Tolerate a blank/invalid string as `{}`.
        let arguments = function
            .and_then(|f| f.get("arguments"))
            .and_then(|v| v.as_str())
            .unwrap_or("{}");
        let parsed: serde_json::Value =
            serde_json::from_str(arguments.trim()).unwrap_or_else(|_| serde_json::json!({}));
        blocks.push(ContentBlock::ToolUse(
            ToolUseBlock::builder()
                .tool_use_id(id)
                .name(name)
                .input(json_to_document(&parsed))
                .build()
                .map_err(|e| {
                    BridgeError::Config(format!("bedrock converse: build tool_use: {e}"))
                })?,
        ));
    }
    // If every tool_calls entry was malformed, fall back to a text block
    // so Converse still receives a valid (non-empty) assistant message.
    if blocks.is_empty() {
        blocks.push(ContentBlock::Text(msg.content_str().to_string()));
    }
    Ok(blocks)
}

/// Translate a Converse response into the gateway [`ChatResponse`].
/// Text content blocks are concatenated; `toolUse` blocks become OpenAI
/// `tool_calls` (#560); image / document blocks (rare in chat scenarios)
/// are skipped.
fn converse_output_into_chat_response(
    resp: aws_sdk_bedrockruntime::operation::converse::ConverseOutput,
    upstream_id: &str,
) -> ChatResponse {
    let (message, finish) = match resp.output() {
        Some(aws_sdk_bedrockruntime::types::ConverseOutput::Message(msg)) => {
            let mut text = String::new();
            let mut tool_calls: Vec<serde_json::Value> = Vec::new();
            for block in msg.content() {
                match block {
                    ContentBlock::Text(t) => text.push_str(t),
                    // #560: translate Converse `toolUse` → OpenAI `tool_calls`.
                    ContentBlock::ToolUse(tu) => tool_calls.push(tool_use_to_openai(tu)),
                    _ => {}
                }
            }
            (
                build_assistant_message(text, tool_calls),
                map_stop_reason(resp.stop_reason()),
            )
        }
        _ => (ChatMessage::assistant(String::new()), FinishReason::Stop),
    };
    let usage = resp
        .usage()
        .map(|u| UsageStats {
            prompt_tokens: u.input_tokens().max(0) as u32,
            completion_tokens: u.output_tokens().max(0) as u32,
            total_tokens: u.total_tokens().max(0) as u32,
            ..Default::default()
        })
        .unwrap_or_default();
    ChatResponse {
        id: String::new(),
        model: upstream_id.to_string(),
        message,
        finish_reason: finish,
        usage,
    }
}

/// Translate a Converse `toolUse` content block into an OpenAI
/// `tool_calls[]` entry (#560). `arguments` MUST be a JSON-**encoded
/// string** per the OpenAI Chat Completions spec — SDK consumers do
/// `JSON.parse(toolCall.function.arguments)`, so passing the parsed
/// object would break every OpenAI-SDK caller.
/// <https://platform.openai.com/docs/api-reference/chat/object>
fn tool_use_to_openai(tu: &ToolUseBlock) -> serde_json::Value {
    let arguments =
        serde_json::to_string(&document_to_json(tu.input())).unwrap_or_else(|_| "{}".to_string());
    serde_json::json!({
        "id": tu.tool_use_id(),
        "type": "function",
        "function": { "name": tu.name(), "arguments": arguments },
    })
}

/// Build the assistant [`ChatMessage`] for a Converse response, folding
/// any translated `tool_calls` into the `extra["tool_calls"]` slot the
/// rest of the gateway already reads (output guardrails, response
/// rendering). When tool calls are present with no prose, `content` is
/// surfaced as `null` (`None`) to mirror OpenAI's
/// assistant-with-tool_calls shape (#395 / #514).
fn build_assistant_message(text: String, tool_calls: Vec<serde_json::Value>) -> ChatMessage {
    let mut message = ChatMessage::assistant(text);
    if tool_calls.is_empty() {
        return message;
    }
    if message.content.as_deref() == Some("") {
        message.content = None;
    }
    message.extra.insert(
        "tool_calls".to_string(),
        serde_json::Value::Array(tool_calls),
    );
    message
}

/// Map AWS Converse `StopReason` to the gateway's [`FinishReason`]
/// taxonomy. Reference:
/// <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html#API_runtime_Converse_ResponseSyntax>
fn map_stop_reason(stop: &SdkStopReason) -> FinishReason {
    match stop {
        SdkStopReason::EndTurn => FinishReason::Stop,
        SdkStopReason::MaxTokens => FinishReason::Length,
        SdkStopReason::ContentFiltered | SdkStopReason::GuardrailIntervened => {
            FinishReason::ContentFilter
        }
        SdkStopReason::ToolUse => FinishReason::ToolCalls,
        SdkStopReason::StopSequence => FinishReason::Stop,
        _ => FinishReason::Stop,
    }
}

/// Translate one Converse stream event into zero, one, or two
/// [`ChatChunk`]s.
///
/// - `MessageStart` → role chunk (role=assistant)
/// - `ContentBlockStart` (toolUse) → tool_call delta opening the call
///   with `id` + `function.name` (#560)
/// - `ContentBlockDelta` (text) → content chunk
/// - `ContentBlockDelta` (toolUse) → tool_call delta carrying an
///   `arguments` fragment (#560)
/// - `MessageStop` → finish-reason chunk
/// - `Metadata` → usage chunk
/// - Other variants (`ContentBlockStop`, image deltas) → no chunk.
///
/// `tool_blocks` records each tool-use block's Converse
/// `contentBlockIndex` in arrival order; a block's position there is the
/// dense 0-based OpenAI `tool_calls[].index`, so a leading text block at
/// index 0 doesn't shift the first tool call.
fn emit_converse_chunk(
    event: ConverseStreamOutput,
    upstream_id: &str,
    emitted_role: &mut bool,
    tool_blocks: &mut Vec<i32>,
) -> Vec<ChatChunk> {
    let mut out = Vec::new();
    match event {
        ConverseStreamOutput::MessageStart(_m) => {
            if !*emitted_role {
                *emitted_role = true;
                out.push(ChatChunk {
                    id: String::new(),
                    model: upstream_id.to_string(),
                    delta: ChatDelta {
                        role: Some(Role::Assistant),
                        ..Default::default()
                    },
                    finish_reason: None,
                    usage: None,
                });
            }
        }
        // #560: a tool-use block opens — emit the OpenAI tool_call delta
        // carrying `id` + `function.name`. Arguments stream in via the
        // subsequent ContentBlockDelta(toolUse) events.
        ConverseStreamOutput::ContentBlockStart(s) => {
            let block_index = s.content_block_index;
            if let Some(ContentBlockStart::ToolUse(tu)) = s.start {
                let oai_index = tool_blocks.len() as i32;
                tool_blocks.push(block_index);
                out.push(tool_call_chunk(
                    upstream_id,
                    oai_index,
                    Some(tu.tool_use_id()),
                    Some(tu.name()),
                    "",
                ));
            }
        }
        ConverseStreamOutput::ContentBlockDelta(d) => {
            let block_index = d.content_block_index;
            match d.delta {
                Some(ContentBlockDelta::Text(text)) => {
                    out.push(ChatChunk {
                        id: String::new(),
                        model: upstream_id.to_string(),
                        delta: ChatDelta {
                            content: Some(text),
                            ..Default::default()
                        },
                        finish_reason: None,
                        usage: None,
                    });
                }
                // #560: a tool-use `arguments` fragment. Correlate to the
                // call opened by ContentBlockStart via the block index.
                Some(ContentBlockDelta::ToolUse(tu)) => {
                    if let Some(pos) = tool_blocks.iter().position(|&i| i == block_index) {
                        out.push(tool_call_chunk(
                            upstream_id,
                            pos as i32,
                            None,
                            None,
                            tu.input(),
                        ));
                    }
                }
                _ => {}
            }
        }
        ConverseStreamOutput::MessageStop(m) => {
            out.push(ChatChunk {
                id: String::new(),
                model: upstream_id.to_string(),
                delta: ChatDelta::default(),
                finish_reason: Some(map_stop_reason(&m.stop_reason)),
                usage: None,
            });
        }
        ConverseStreamOutput::Metadata(m) => {
            if let Some(u) = m.usage {
                out.push(ChatChunk {
                    id: String::new(),
                    model: upstream_id.to_string(),
                    delta: ChatDelta::default(),
                    finish_reason: None,
                    usage: Some(UsageStats {
                        prompt_tokens: u.input_tokens.max(0) as u32,
                        completion_tokens: u.output_tokens.max(0) as u32,
                        total_tokens: u.total_tokens.max(0) as u32,
                        ..Default::default()
                    }),
                });
            }
        }
        // ContentBlockStop carries no customer-visible payload. The
        // catch-all is non-exhaustive because the SDK enum is
        // non_exhaustive — future AWS event types fall here safely.
        _ => {}
    }
    out
}

/// Build a streaming OpenAI `tool_calls` delta chunk (#560). The opening
/// chunk for a call carries `id` + `function.name` (+ an empty
/// `arguments`); subsequent chunks carry only an `arguments` fragment.
/// `index` is the call's dense 0-based position in the `tool_calls`
/// array. Reference:
/// <https://platform.openai.com/docs/api-reference/chat/streaming>
fn tool_call_chunk(
    upstream_id: &str,
    index: i32,
    id: Option<&str>,
    name: Option<&str>,
    arguments: &str,
) -> ChatChunk {
    let mut function = serde_json::Map::new();
    if let Some(name) = name {
        function.insert(
            "name".to_string(),
            serde_json::Value::String(name.to_string()),
        );
    }
    function.insert(
        "arguments".to_string(),
        serde_json::Value::String(arguments.to_string()),
    );
    let mut call = serde_json::Map::new();
    call.insert("index".to_string(), serde_json::Value::from(index));
    if let Some(id) = id {
        call.insert("id".to_string(), serde_json::Value::String(id.to_string()));
        call.insert(
            "type".to_string(),
            serde_json::Value::String("function".to_string()),
        );
    }
    call.insert("function".to_string(), serde_json::Value::Object(function));
    ChatChunk {
        id: String::new(),
        model: upstream_id.to_string(),
        delta: ChatDelta {
            tool_calls: Some(vec![serde_json::Value::Object(call)]),
            ..Default::default()
        },
        finish_reason: None,
        usage: None,
    }
}

/// Map the SDK's `ConverseError` SdkError variant to BridgeError.
/// Delegates to the generic helper so both Converse and ConverseStream
/// share the wire-shape + Retry-After preservation logic.
fn map_converse_sdk_error(
    e: SdkError<ConverseError, aws_smithy_runtime_api::http::Response>,
    started: Instant,
    deadline: Option<Duration>,
) -> BridgeError {
    map_aws_sdk_error_generic(e, started, deadline)
}

fn map_converse_stream_sdk_error(
    e: SdkError<ConverseStreamError, aws_smithy_runtime_api::http::Response>,
    started: Instant,
    deadline: Option<Duration>,
) -> BridgeError {
    map_aws_sdk_error_generic(e, started, deadline)
}

/// Common AWS SDK error classifier. Routes through the same
/// UpstreamStatus / Transport / Timeout taxonomy the legacy
/// `map_service_error` uses for `/invoke`, preserving:
///
/// - `wire: UpstreamWire::Bedrock` — so `error_translate` renders
///   Bedrock-shape errors back to OpenAI/Anthropic-shape clients
/// - `retry_after: Some(...)` — parsed from the upstream's
///   `Retry-After` header so the cooldown layer honours the AWS
///   throttle hint instead of falling back to its default
/// - `parsed.kind: Some(<AWS error code>)` — so a 429 from
///   `ThrottlingException` is distinguishable from a 429 from a
///   different throttle source
///
/// The `E: ProvideErrorMetadata` bound lets the helper extract
/// `.meta().code()` regardless of which operation the error came
/// from (Converse vs. ConverseStream variants of the same shape).
///
/// Audit MEDIUM / HIGH on PR #389 — the prior implementation
/// collapsed to `BridgeError::upstream_status(status, msg)` which
/// sets `wire: Unknown` and `retry_after: None`, silently regressing
/// the legacy /invoke path's hardening (PR #323 MEDIUM-2).
fn map_aws_sdk_error_generic<E>(
    err: SdkError<E, aws_smithy_runtime_api::http::Response>,
    started: Instant,
    deadline: Option<Duration>,
) -> BridgeError
where
    E: std::fmt::Debug + ProvideErrorMetadata,
{
    // Deadline-elapsed wins over upstream classification — surfacing
    // an UpstreamStatus on a timed-out request would hide the
    // operator's configured deadline.
    if let Some(d) = deadline {
        if started.elapsed() >= d {
            return BridgeError::Timeout {
                elapsed_ms: started.elapsed().as_millis() as u64,
                cause: String::new(),
            };
        }
    }
    match err {
        SdkError::ServiceError(svc) => bedrock_service_error_to_upstream_status(svc),
        SdkError::TimeoutError(_) => BridgeError::Timeout {
            elapsed_ms: started.elapsed().as_millis() as u64,
            cause: String::new(),
        },
        other => BridgeError::Transport(format!("{other}")),
    }
}

/// Translate an AWS SDK `ServiceError` into a fully-shaped
/// `BridgeError::UpstreamStatus` — same fields as the legacy
/// `map_service_error` would emit for `/invoke`. Generic over the
/// operation error type so the same logic covers Converse +
/// ConverseStream + any future Bedrock op.
fn bedrock_service_error_to_upstream_status<E>(
    svc: ServiceError<E, aws_smithy_runtime_api::http::Response>,
) -> BridgeError
where
    E: ProvideErrorMetadata,
{
    let kind = svc.err().meta().code().map(str::to_string);
    let raw = svc.raw();
    let status = raw.status().as_u16();
    // Convert smithy HeaderMap → http::HeaderMap so we can reuse the
    // gateway-level `parse_retry_after` helper. Headers with invalid
    // bytes are dropped (defensive — SDK should not produce them).
    let mut hdrs = http::HeaderMap::new();
    for (k, v) in raw.headers() {
        if let (Ok(name), Ok(val)) = (
            http::HeaderName::from_bytes(k.as_bytes()),
            http::HeaderValue::from_str(v),
        ) {
            hdrs.insert(name, val);
        }
    }
    let retry_after = aisix_gateway::parse_retry_after(&hdrs);
    let message = match status {
        401 | 403 => "upstream authentication failed".to_string(),
        404 => "upstream model not found".to_string(),
        408 => "upstream request timeout".to_string(),
        429 => "upstream rate limited".to_string(),
        _ => format!("upstream returned {status}"),
    };
    // SECURITY: the AWS error message embeds operator-internal
    // taxonomy (ARNs, region, account id, IAM role names). We surface
    // only the AWS error CODE (e.g. "ThrottlingException") for the
    // error_translate layer; the customer-visible `message` is the
    // canned status-keyed phrase above.
    let parsed = kind.as_ref().map(|k| {
        Box::new(aisix_gateway::UpstreamErrorView {
            kind: Some(k.clone()),
            message: None,
            code: None,
            param: None,
        })
    });
    BridgeError::UpstreamStatus {
        status,
        message,
        parsed,
        wire: aisix_gateway::UpstreamWire::Bedrock,
        retry_after,
    }
}

/// Pull the PK's `request.param_constraints` off the context — the single
/// source both Converse dispatch paths read, so they cannot drift in how the
/// clamp input is resolved (#463).
fn pk_param_constraints(ctx: &BridgeContext) -> Option<&ParamConstraints> {
    ctx.provider_key
        .request
        .as_ref()
        .and_then(|r| r.param_constraints.as_ref())
}

/// Build an `InferenceConfiguration` from the gateway's
/// [`ChatFormat`] knobs, or `None` if no knobs are set. Caller
/// only attaches when `Some` so we don't ship an empty
/// `inferenceConfig: {}` to Bedrock (some publishers tolerate it,
/// others 400).
///
/// `constraints` is the PK's `request.param_constraints` (#463). The
/// Converse path carries no JSON body, so the `/invoke` body clamp
/// (`apply_param_constraints`) never runs for it; we re-apply the
/// `temperature_{min,max}` guardrail here against Converse's typed
/// `inferenceConfig.temperature` so the clamp is uniform across every
/// publisher routed through the PK. Applied as two independent
/// comparisons (NOT `f32::clamp`, which panics when `min > max`) to
/// mirror `apply_param_constraints` and stay panic-safe on a
/// misconfigured `min > max`. `ParamConstraints` is temperature-only
/// today, so `max_tokens` / `top_p` are forwarded unclamped (no
/// constraint field exists for them).
fn build_inference_config(
    req: &ChatFormat,
    constraints: Option<&ParamConstraints>,
) -> Option<InferenceConfiguration> {
    if req.temperature.is_none() && req.max_tokens.is_none() && req.top_p.is_none() {
        return None;
    }
    let mut b = InferenceConfiguration::builder();
    if let Some(t) = req.temperature {
        let mut next = t;
        if let Some(c) = constraints {
            // Compare in f64-space (exact) and cast only on assignment, so
            // an f64 bound not representable in f32 still clamps at the
            // right boundary. Max-then-min order matches the /invoke path.
            if let Some(max) = c.temperature_max {
                if f64::from(next) > max {
                    next = max as f32;
                }
            }
            if let Some(min) = c.temperature_min {
                if f64::from(next) < min {
                    next = min as f32;
                }
            }
        }
        b = b.temperature(next);
    }
    if let Some(m) = req.max_tokens {
        // Bedrock's API uses i32; ChatFormat's u32 is always non-negative.
        let clamped = i32::try_from(m).unwrap_or(i32::MAX);
        b = b.max_tokens(clamped);
    }
    if let Some(p) = req.top_p {
        b = b.top_p(p);
    }
    Some(b.build())
}

/// Build a Converse [`ToolConfiguration`] from the OpenAI `tools` /
/// `tool_choice` the caller sent — they arrive in [`ChatFormat::extra`]
/// (the gateway captures unknown top-level request fields there). Returns
/// `None` when there are no usable tools (#560).
///
/// OpenAI → Converse `tool_choice` mapping. Converse's `toolChoice` is
/// `auto | any | tool`, with no `none`; `auto` is Converse's default, so
/// the common cases send no explicit `toolChoice` (which also avoids the
/// field on publishers that only accept the default):
///   * `tool_choice:"none"` → no toolConfig at all (model answers in prose)
///   * `tool_choice:"auto"` / omitted → tools sent, `toolChoice` omitted
///   * `tool_choice:"required"` → `toolChoice:{any:{}}`
///   * `tool_choice:{function:{name}}` → `toolChoice:{tool:{name}}`
///
/// Forcing (`any` / `tool`) is only honoured by models that support it
/// (per AWS, specific-tool choice is Anthropic Claude 3 + Amazon Nova
/// only); on other publishers Bedrock returns a validation error, which
/// is the correct surface for an explicit unsupported force.
/// References:
/// <https://platform.openai.com/docs/api-reference/chat/create#chat-create-tool_choice>,
/// <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html>
fn build_tool_config(req: &ChatFormat) -> Option<ToolConfiguration> {
    // OpenAI `tool_choice:"none"` means "don't call any tool this turn".
    // Converse has no `none`, so send no toolConfig — the model answers
    // in prose, honouring the user-visible contract (no tool call). Tool
    // visibility is lost, but that matches the intent of "none".
    if req.extra.get("tool_choice").and_then(|v| v.as_str()) == Some("none") {
        return None;
    }
    let tools_json = req.extra.get("tools").and_then(|v| v.as_array())?;
    let mut tools: Vec<Tool> = Vec::new();
    for entry in tools_json {
        // OpenAI only defines `type:"function"` tools today; skip any
        // other shape rather than guessing how to translate it.
        if entry
            .get("type")
            .and_then(|v| v.as_str())
            .is_some_and(|ty| ty != "function")
        {
            continue;
        }
        let Some(function) = entry.get("function") else {
            continue;
        };
        let Some(name) = function.get("name").and_then(|v| v.as_str()) else {
            continue;
        };
        let mut spec = ToolSpecification::builder().name(name);
        if let Some(description) = function.get("description").and_then(|v| v.as_str()) {
            spec = spec.description(description);
        }
        // OpenAI `parameters` is a JSON Schema object. Default to an empty
        // object schema when omitted; skip the tool if `parameters` is
        // present but not an object — a non-object schema is malformed and
        // Bedrock would reject the whole Converse request with a 400.
        let parameters = match function.get("parameters") {
            Some(p) if p.is_object() => p.clone(),
            Some(_) => continue,
            None => serde_json::json!({"type": "object", "properties": {}}),
        };
        spec = spec.input_schema(ToolInputSchema::Json(json_to_document(&parameters)));
        if let Ok(spec) = spec.build() {
            tools.push(Tool::ToolSpec(spec));
        }
    }
    if tools.is_empty() {
        return None;
    }
    let mut config = ToolConfiguration::builder().set_tools(Some(tools));
    if let Some(choice) = req.extra.get("tool_choice").and_then(map_tool_choice) {
        config = config.tool_choice(choice);
    }
    config.build().ok()
}

/// Map an OpenAI `tool_choice` to a Converse [`ToolChoice`], or `None` to
/// leave it unset (Converse then applies its `auto` default). `"none"` is
/// handled by the caller (no toolConfig at all).
fn map_tool_choice(choice: &serde_json::Value) -> Option<ToolChoice> {
    match choice {
        serde_json::Value::String(s) => match s.as_str() {
            "required" => Some(ToolChoice::Any(AnyToolChoice::builder().build())),
            // "auto" is Converse's default — omit rather than send an
            // explicit field some publishers reject.
            _ => None,
        },
        serde_json::Value::Object(map) => {
            let name = map
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|v| v.as_str())?;
            SpecificToolChoice::builder()
                .name(name)
                .build()
                .ok()
                .map(ToolChoice::Tool)
        }
        _ => None,
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
    fn publisher_strips_au_ca_jp_cross_region_prefixes() {
        // #412: Australia/Canada/Japan cross-region inference profiles
        // must resolve to their publisher, same as us./eu./apac.
        assert_eq!(
            BedrockPublisher::from_model_id("au.meta.llama3-1-70b-instruct-v1:0"),
            Some(BedrockPublisher::Meta),
        );
        assert_eq!(
            BedrockPublisher::from_model_id("ca.anthropic.claude-3-5-sonnet-20241022-v2:0"),
            Some(BedrockPublisher::Anthropic),
        );
        assert_eq!(
            BedrockPublisher::from_model_id("jp.amazon.nova-pro-v1:0"),
            Some(BedrockPublisher::AmazonNova),
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
        assert_eq!(err.http_status(), 401);
        match err {
            BridgeError::InvalidUpstreamCredentials(msg) => {
                assert!(
                    msg.contains("api_key is empty"),
                    "must mention the empty api_key; got {msg}"
                );
                assert!(
                    msg.contains("access_key_id"),
                    "must hint at required JSON shape; got {msg}"
                );
            }
            other => panic!("expected InvalidUpstreamCredentials error, got {other:?}"),
        }
    }

    #[test]
    fn bedrock_secret_rejects_non_json() {
        let err = BedrockSecret::parse("AKIA-test").unwrap_err();
        assert_eq!(err.http_status(), 401);
        match err {
            BridgeError::InvalidUpstreamCredentials(msg) => {
                assert!(
                    msg.contains("must be valid JSON"),
                    "must mention JSON requirement; got {msg}"
                );
            }
            other => panic!("expected InvalidUpstreamCredentials error, got {other:?}"),
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
            BridgeError::InvalidUpstreamCredentials(msg) => {
                assert!(
                    !msg.contains("X-DISTINCTIVE-LEAK-MARKER-Y"),
                    "error must NOT echo raw secret bytes; got {msg}"
                );
                assert!(
                    !msg.contains("DISTINCTIVE"),
                    "error must NOT leak partial secret bytes; got {msg}"
                );
            }
            other => panic!("expected InvalidUpstreamCredentials error, got {other:?}"),
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
        assert!(matches!(err, BridgeError::InvalidUpstreamCredentials(_)));
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

    // Removed `chat_rejects_non_anthropic_publishers_with_publisher_named`:
    // Phase G Step 3 wires non-Anthropic publishers via Converse, so the
    // "not yet implemented" error path it pinned no longer exists.
    // Coverage replaced by the new Converse-path tests in this module.

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
            BridgeError::InvalidUpstreamCredentials(msg) => {
                assert!(msg.contains("must be valid JSON"));
            }
            other => panic!("expected InvalidUpstreamCredentials error, got {other:?}"),
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
        assert_eq!(err.http_status(), 401);
        match err {
            BridgeError::InvalidUpstreamCredentials(msg) => {
                assert!(msg.contains("api_key is empty"));
            }
            other => panic!("expected InvalidUpstreamCredentials error, got {other:?}"),
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
            BridgeError::InvalidUpstreamConfig(msg) => {
                assert!(msg.contains("model_name missing"));
            }
            other => panic!("expected InvalidUpstreamConfig error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_ignores_req_model_and_uses_ctx_model_name() {
        // D6 audit HIGH-1 regression: dispatch must read upstream id
        // from ctx.model.model_name, NOT from req.model. Phase G
        // Step 3 wires Converse for all known publishers, so the old
        // "publisher-not-implemented" proof path is gone — we now use
        // an UNKNOWN publisher prefix on ctx.model.model_name so the
        // chat call hits the publisher-resolution error AND the error
        // message echoes the unknown id (proving model_name was the
        // source). req.model is set to a known Anthropic id; if
        // dispatch had used req.model the call would have routed
        // successfully or hit a different error class.
        let bridge = BedrockBridge::new();
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("unknown-publisher.foo-bar-v1:0"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new(
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            vec![ChatMessage::user("hi")],
        );
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(
                    msg.contains("publisher unknown"),
                    "must hit publisher-resolution error (proving model_name was used); got {msg}"
                );
                assert!(
                    msg.contains("unknown-publisher.foo-bar-v1:0"),
                    "must echo the ctx.model.model_name value (NOT req.model); got {msg}"
                );
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    // Removed `chat_stream_returns_clear_not_implemented_error`:
    // Phase G Step 3 wires Converse streaming for all publishers; the
    // "streaming not yet implemented" error path it pinned no longer
    // exists. The new chat_converse_stream-side tests in this module
    // exercise the actual streaming path.

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
        assert_eq!(chat.message.content_str(), "hello from bedrock");
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
        assert_eq!(chat.message.content_str(), "calling tool");
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
        // Audit H4: `arguments` MUST be a JSON-encoded STRING per the
        // OpenAI Chat Completions spec, not a parsed object. SDK
        // consumers do `JSON.parse(toolCall.function.arguments)` — a
        // future refactor that passes the parsed object would silently
        // break every OpenAI-SDK caller against an Anthropic upstream.
        let args = tool_calls[0]
            .get("function")
            .and_then(|f| f.get("arguments"))
            .and_then(|v| v.as_str())
            .expect("arguments must be a JSON-encoded STRING per OpenAI spec");
        let parsed: serde_json::Value =
            serde_json::from_str(args).expect("arguments string must itself be valid JSON");
        assert_eq!(parsed.get("city").and_then(|v| v.as_str()), Some("SF"));
    }

    #[tokio::test]
    async fn chat_maps_upstream_4xx_to_canned_message_not_body_echo() {
        // Audit M1: Bedrock error envelopes can contain account
        // numbers (in ARNs), model IDs, IAM role names — must not
        // leak into customer-visible error.
        //
        // Audit M5 follow-up: assert the canned message EXACTLY,
        // not just absence-of-leak. A future refactor that re-renders
        // SDK metadata into the message would pass an absence check
        // but fail the exact-match assertion.
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
            BridgeError::UpstreamStatus {
                status, message, ..
            } => {
                assert_eq!(status, 400);
                assert!(
                    !message.contains("123456789012") && !message.contains("internal-leaky-role"),
                    "upstream body must not leak account / role info into customer error; got {message:?}"
                );
                // Positive pin (audit M5): exact-match the canned
                // status-keyed phrase. Bedrock returns 400 → bucket
                // is "upstream returned 400" per `map_service_error`.
                assert_eq!(
                    message, "upstream returned 400",
                    "must emit canned 4xx phrasing only; got {message:?}"
                );
            }
            other => panic!("expected UpstreamStatus, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_maps_upstream_429_with_retry_after_and_canned_rate_limited() {
        // Audit H1: Bedrock's `Retry-After` header on 429 must reach
        // the cooldown layer. Collapsing it to `None` silently
        // degrades multi-region / burst behavior.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(r"^/model/.+/invoke$"))
            .respond_with(
                ResponseTemplate::new(429)
                    .insert_header("retry-after", "42")
                    .set_body_json(serde_json::json!({
                        "message": "Too many requests for account 123456789012"
                    })),
            )
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
                status,
                message,
                retry_after,
                wire,
                parsed,
                ..
            } => {
                assert_eq!(status, 429);
                assert_eq!(message, "upstream rate limited");
                assert!(
                    !message.contains("123456789012"),
                    "must not leak account id; got {message:?}"
                );
                // Audit H1 pin: the SDK / smithy headers must round-trip
                // Retry-After into the BridgeError so the cooldown
                // layer sees the upstream's hint instead of falling
                // back to a configured default.
                assert_eq!(
                    retry_after,
                    Some(std::time::Duration::from_secs(42)),
                    "Retry-After must reach BridgeError::UpstreamStatus"
                );
                // Audit fix (PR #323 MEDIUM-2): pin `wire` so a
                // refactor that breaks cross-wire translation fails
                // here. `parsed.kind` should carry the AWS exception
                // name (the SDK derives this from `__type` /
                // X-Amzn-ErrorType). `parsed.message` stays None for
                // operator-taxonomy redaction (ARNs, account ids).
                assert_eq!(wire, aisix_gateway::UpstreamWire::Bedrock);
                if let Some(view) = parsed {
                    assert!(
                        view.message.is_none(),
                        "bedrock must NOT surface upstream message; got {:?}",
                        view.message
                    );
                }
            }
            other => panic!("expected UpstreamStatus with retry_after, got {other:?}"),
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
        assert_eq!(chat.message.content_str(), "cross-region ok");
    }

    /// Audit M6: cross-region dispatch coverage was only `us.`; the
    /// historically-broken case (`us-gov.` with hyphen) and `global.`
    /// (exactly 6 chars — accidentally working under the old matcher)
    /// need real dispatch-path tests so a future regression in
    /// `strip_region_prefix` is caught at the wire layer, not just at
    /// the unit-test layer.
    #[tokio::test]
    async fn chat_with_us_gov_cross_region_prefix_dispatches_with_full_model_id() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(
                r"^/model/us-gov\.anthropic\.claude-3-5-sonnet-20241022-v2(:0|%3A0)/invoke$",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "msg_xr", "model": "claude-3-5-sonnet",
                "content": [{"type": "text", "text": "us-gov ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1}
            })))
            .expect(1)
            .mount(&server)
            .await;

        let bridge = BedrockBridge::new().with_endpoint_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("us-gov.anthropic.claude-3-5-sonnet-20241022-v2:0"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("my-claude", vec![ChatMessage::user("hi")]);
        let chat = bridge.chat(&req, &ctx).await.unwrap();
        assert_eq!(chat.message.content_str(), "us-gov ok");
    }

    #[tokio::test]
    async fn chat_with_global_cross_region_prefix_dispatches_with_full_model_id() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(
                r"^/model/global\.anthropic\.claude-3-5-sonnet-20241022-v2(:0|%3A0)/invoke$",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "msg_xr", "model": "claude-3-5-sonnet",
                "content": [{"type": "text", "text": "global ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1}
            })))
            .expect(1)
            .mount(&server)
            .await;

        let bridge = BedrockBridge::new().with_endpoint_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("global.anthropic.claude-3-5-sonnet-20241022-v2:0"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("my-claude", vec![ChatMessage::user("hi")]);
        let chat = bridge.chat(&req, &ctx).await.unwrap();
        assert_eq!(chat.message.content_str(), "global ok");
    }

    // Removed `chat_publisher_not_implemented_error_includes_model_id_and_publisher_name`
    // (and its preceding doc comment about the H2 regression):
    // Phase G Step 3 wires Converse for all non-Anthropic publishers,
    // so the "not yet implemented" error path the test pinned no
    // longer exists. New Converse-path tests below cover the
    // non-Anthropic dispatch contract.

    /// Audit M2 regression: defense-in-depth model-id char check.
    /// Even though the AWS SDK URL-encodes reserved chars, the gateway
    /// layer must reject upfront so the model id can't carry
    /// log-injection / dashboard-corruption payloads (it propagates
    /// into metrics labels).
    #[tokio::test]
    async fn chat_rejects_model_id_with_path_injection_chars() {
        let bridge = BedrockBridge::new();
        // Whitespace + tab — would corrupt metrics labels even if the
        // SDK URL-encoded the path correctly.
        let evil_model = "anthropic.claude\t evil model";
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with(evil_model),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("customer-facing", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(
                    msg.contains("unexpected characters"),
                    "must reject invalid model id chars; got {msg}"
                );
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }

    // Removed `chat_stream_anthropic_returns_d7_2_b_specific_error`
    // and `chat_stream_non_anthropic_publisher_returns_d7_3_specific_error`:
    // Phase G Step 3 wires Converse streaming for all publishers, so
    // the "not yet implemented" error paths these tests pinned no
    // longer exist. New chat_converse_stream-path tests below cover
    // the actual streaming dispatch.

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

    // ─── Phase G Step 3 — Converse path tests ──────────────────────

    /// Canned Converse non-stream response body. Shape per AWS docs:
    /// <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html#API_runtime_Converse_ResponseSyntax>.
    fn default_converse_response_template() -> ResponseTemplate {
        ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "hello from converse"}]
                }
            },
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 7,
                "outputTokens": 3,
                "totalTokens": 10
            },
            "metrics": {"latencyMs": 1}
        }))
    }

    #[tokio::test]
    async fn chat_meta_publisher_dispatches_via_converse_url() {
        // Non-Anthropic publisher: Phase G Step 3 wires Meta (and the
        // other 4 publishers) through the unified Converse API.
        // Asserts URL path is `/model/<id>/converse` (NOT /invoke) and
        // the SDK's typed response decodes into the gateway's
        // ChatResponse correctly.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(
                r"^/model/meta\.llama3-3-70b-instruct-v1(:0|%3A0)/converse$",
            ))
            .respond_with(default_converse_response_template())
            .expect(1)
            .mount(&server)
            .await;

        let bridge = BedrockBridge::new().with_endpoint_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("meta.llama3-3-70b-instruct-v1:0"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("my-llama", vec![ChatMessage::user("hi")]);
        let chat = bridge.chat(&req, &ctx).await.unwrap();
        assert_eq!(chat.message.content_str(), "hello from converse");
        assert_eq!(chat.usage.prompt_tokens, 7);
        assert_eq!(chat.usage.completion_tokens, 3);
        assert_eq!(chat.usage.total_tokens, 10);
        assert_eq!(chat.finish_reason, FinishReason::Stop);
    }

    #[tokio::test]
    async fn chat_amazon_nova_publisher_dispatches_via_converse_url() {
        // Second non-Anthropic publisher — Amazon Nova. Same Converse
        // dispatch as Meta, different model id prefix; pins that the
        // dispatch is genuinely uniform across publishers (regression
        // guard against a future change that hard-codes per-publisher
        // routing inside chat_converse itself).
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(
                r"^/model/amazon\.nova-pro-v1(:0|%3A0)/converse$",
            ))
            .respond_with(default_converse_response_template())
            .expect(1)
            .mount(&server)
            .await;

        let bridge = BedrockBridge::new().with_endpoint_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("amazon.nova-pro-v1:0"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("my-nova", vec![ChatMessage::user("hi")]);
        let chat = bridge.chat(&req, &ctx).await.unwrap();
        assert_eq!(chat.message.content_str(), "hello from converse");
    }

    #[tokio::test]
    async fn chat_stream_for_anthropic_dispatches_via_converse_stream_url() {
        // Stream dispatch goes through Converse for ALL publishers
        // including Anthropic (the legacy /invoke path never had a
        // stream variant). Wiremock here only needs to acknowledge
        // the request — full event-stream binary frame decode is
        // exercised at the SDK level, beyond the scope of unit
        // tests; this guard just verifies the URL routing.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(
                r"^/model/anthropic\.claude-3-5-sonnet-20241022-v2(:0|%3A0)/converse-stream$",
            ))
            // Returning a 200 with an empty body causes the SDK to
            // emit a transport / decode error which we catch below.
            // The test's job is to pin the URL pattern, not to
            // exercise the eventstream decoder.
            .respond_with(ResponseTemplate::new(200))
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
        // Either the chat_stream() call returns Ok (and the stream
        // errors on first .next() because the empty body isn't a
        // valid eventstream) OR returns Err directly from the SDK.
        // Both prove the dispatch reached wiremock — wiremock's
        // `.expect(1)` enforces this on drop.
        let _ = bridge.chat_stream(&req, &ctx).await;
    }

    #[tokio::test]
    async fn chat_stream_for_meta_dispatches_via_converse_stream_url() {
        // Same as the Anthropic-stream test but for a non-Anthropic
        // publisher — pins that chat_stream wiring is uniform across
        // publishers.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(
                r"^/model/meta\.llama3-3-70b-instruct-v1(:0|%3A0)/converse-stream$",
            ))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;

        let bridge = BedrockBridge::new().with_endpoint_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("meta.llama3-3-70b-instruct-v1:0"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("my-llama", vec![ChatMessage::user("hi")]);
        let _ = bridge.chat_stream(&req, &ctx).await;
    }

    /// Audit HIGH-1+HIGH-2 (PR #389): Converse 4xx must preserve
    /// `wire: Bedrock`, parse `Retry-After` into `retry_after`, and
    /// extract the AWS error code into `parsed.kind` — matching the
    /// legacy /invoke path's `map_service_error`. Without this fix,
    /// non-Anthropic publishers and all streaming silently regress
    /// `wire` to `Unknown` (breaks error_translate) and drop the
    /// Retry-After hint (breaks cooldown auto-tuning).
    #[tokio::test]
    async fn chat_converse_maps_upstream_429_with_retry_after_and_bedrock_wire() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(r"^/model/meta\..+/converse$"))
            .respond_with(
                ResponseTemplate::new(429)
                    .insert_header("Retry-After", "42")
                    .insert_header("x-amzn-ErrorType", "ThrottlingException")
                    .set_body_json(serde_json::json!({
                        "__type": "ThrottlingException",
                        "message": "Rate exceeded"
                    })),
            )
            .mount(&server)
            .await;

        let bridge = BedrockBridge::new().with_endpoint_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("meta.llama3-3-70b-instruct-v1:0"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("my-llama", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::UpstreamStatus {
                status,
                message,
                wire,
                retry_after,
                parsed,
            } => {
                assert_eq!(status, 429);
                assert_eq!(message, "upstream rate limited");
                assert_eq!(
                    wire,
                    aisix_gateway::UpstreamWire::Bedrock,
                    "must preserve Bedrock wire for error_translate"
                );
                assert_eq!(
                    retry_after,
                    Some(std::time::Duration::from_secs(42)),
                    "must propagate upstream Retry-After to cooldown layer"
                );
                let parsed = parsed.expect("must expose AWS error kind to error_translate");
                assert_eq!(
                    parsed.kind.as_deref(),
                    Some("ThrottlingException"),
                    "must extract AWS error code into parsed.kind"
                );
            }
            other => panic!("expected UpstreamStatus, got {other:?}"),
        }
    }

    /// Audit HIGH-1 (PR #389): Converse 4xx that isn't 429 must
    /// still get the canned status-keyed message + Bedrock wire,
    /// matching the legacy /invoke path.
    #[tokio::test]
    async fn chat_converse_maps_upstream_4xx_to_canned_message_with_bedrock_wire() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(r"^/model/meta\..+/converse$"))
            .respond_with(
                ResponseTemplate::new(403)
                    .insert_header("x-amzn-ErrorType", "AccessDeniedException")
                    .set_body_json(serde_json::json!({
                        "__type": "AccessDeniedException",
                        "message": "User: arn:aws:iam::123:user/x is not authorized"
                    })),
            )
            .mount(&server)
            .await;

        let bridge = BedrockBridge::new().with_endpoint_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("meta.llama3-3-70b-instruct-v1:0"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("my-llama", vec![ChatMessage::user("hi")]);
        let err = bridge.chat(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::UpstreamStatus {
                status,
                message,
                wire,
                parsed,
                ..
            } => {
                assert_eq!(status, 403);
                assert_eq!(
                    message, "upstream authentication failed",
                    "must use canned phrase (NOT echo upstream ARN)"
                );
                assert_eq!(wire, aisix_gateway::UpstreamWire::Bedrock);
                assert!(
                    !message.contains("arn:aws:iam"),
                    "canned phrase must not leak the operator-internal ARN; got {message}"
                );
                let parsed = parsed.expect("must expose AWS error kind");
                assert_eq!(parsed.kind.as_deref(), Some("AccessDeniedException"));
            }
            other => panic!("expected UpstreamStatus, got {other:?}"),
        }
    }

    /// Audit MEDIUM-2 (PR #389): ChatFormat.temperature /
    /// .max_tokens / .top_p must flow through to Converse's
    /// InferenceConfiguration. Dropping them silently is a
    /// behavioural regression for every non-Anthropic Bedrock
    /// customer who was setting these knobs via the gateway.
    #[tokio::test]
    async fn chat_converse_wires_temperature_max_tokens_top_p_into_inference_config() {
        let server = MockServer::start().await;
        let responder = CapturingResponder::default();
        Mock::given(method("POST"))
            .and(path_regex(r"^/model/.+/converse$"))
            .respond_with(responder.clone())
            .expect(1)
            .mount(&server)
            .await;

        let bridge = BedrockBridge::new().with_endpoint_override(server.uri());
        let mut req = ChatFormat::new("my-llama", vec![ChatMessage::user("hi")]);
        // Pick values that survive the f32 → JSON → f64 round-trip
        // cleanly (powers of 1/2 / 1/4 / 1/8 ...). Avoids brittle
        // `0.7 ≠ 0.6999999881` precision-comparison failures.
        req.temperature = Some(0.5);
        req.max_tokens = Some(128);
        req.top_p = Some(0.75);
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("meta.llama3-3-70b-instruct-v1:0"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let _ = bridge.chat(&req, &ctx).await;

        let body = responder.captured_body.lock().unwrap().clone().unwrap();
        let cfg = body
            .get("inferenceConfig")
            .and_then(|v| v.as_object())
            .expect("inferenceConfig must be set when ChatFormat carries knobs");
        // Bedrock uses camelCase per AWS spec.
        assert_eq!(
            cfg.get("temperature").and_then(|v| v.as_f64()),
            Some(0.5),
            "temperature must propagate; body={body}"
        );
        assert_eq!(
            cfg.get("maxTokens").and_then(|v| v.as_i64()),
            Some(128),
            "maxTokens (camelCase) must propagate; body={body}"
        );
        assert_eq!(
            cfg.get("topP").and_then(|v| v.as_f64()),
            Some(0.75),
            "topP (camelCase) must propagate; body={body}"
        );
    }

    /// Companion to the wires-temperature test: when no knobs are
    /// set, inferenceConfig must be OMITTED from the body — sending
    /// an empty `{}` causes some Bedrock publishers to 400.
    #[tokio::test]
    async fn chat_converse_omits_inference_config_when_no_knobs_set() {
        let server = MockServer::start().await;
        let responder = CapturingResponder::default();
        Mock::given(method("POST"))
            .and(path_regex(r"^/model/.+/converse$"))
            .respond_with(responder.clone())
            .expect(1)
            .mount(&server)
            .await;

        let bridge = BedrockBridge::new().with_endpoint_override(server.uri());
        // ChatFormat::new leaves temperature/max_tokens/top_p None.
        let req = ChatFormat::new("my-llama", vec![ChatMessage::user("hi")]);
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("meta.llama3-3-70b-instruct-v1:0"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let _ = bridge.chat(&req, &ctx).await;

        let body = responder.captured_body.lock().unwrap().clone().unwrap();
        assert!(
            body.get("inferenceConfig").is_none(),
            "inferenceConfig must be absent when ChatFormat has no knobs; body={body}"
        );
    }

    #[tokio::test]
    async fn chat_converse_request_body_uses_text_content_block_shape() {
        // Pin the outbound request body shape: Converse expects
        // `messages: [{role, content: [{text: "..."}]}]`. A
        // regression that emitted Anthropic-shape `messages: [{role,
        // content: "..."}]` (string instead of typed block array)
        // would 400 upstream. We use Meta here so the test routes
        // through chat_converse (the Anthropic publisher would route
        // through legacy /invoke and skip the Converse body shape).
        let server = MockServer::start().await;
        let responder = CapturingResponder::default();
        Mock::given(method("POST"))
            .and(path_regex(r"^/model/.+/converse$"))
            .respond_with(responder.clone())
            .expect(1)
            .mount(&server)
            .await;

        let bridge = BedrockBridge::new().with_endpoint_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("meta.llama3-3-70b-instruct-v1:0"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new(
            "my-llama",
            vec![
                ChatMessage::system("you are concise"),
                ChatMessage::user("hi"),
            ],
        );
        let _ = bridge.chat(&req, &ctx).await;

        let body = responder.captured_body.lock().unwrap().clone().unwrap();
        // System message must be lifted to top-level `system` array
        // (Converse splits systems out of `messages` per AWS spec).
        let system = body.get("system").and_then(|v| v.as_array()).unwrap();
        assert_eq!(system.len(), 1, "system block must be present; body={body}");
        assert_eq!(
            system[0].get("text").and_then(|v| v.as_str()),
            Some("you are concise")
        );
        // messages[] must use typed content-block shape, NOT Anthropic-
        // style flat string.
        let messages = body.get("messages").and_then(|v| v.as_array()).unwrap();
        assert_eq!(messages.len(), 1);
        let content = messages[0]
            .get("content")
            .and_then(|v| v.as_array())
            .expect("content must be an array of typed blocks");
        assert_eq!(content.len(), 1);
        assert_eq!(
            content[0].get("text").and_then(|v| v.as_str()),
            Some("hi"),
            "content block must carry `text` field; body={body}"
        );
        // Per Bedrock Converse spec, the `model` field is NOT part of
        // the body — model id flows via URL path. The SDK should not
        // emit it.
        assert!(
            body.get("model").is_none(),
            "Converse body must NOT carry top-level `model` field; body={body}"
        );
    }

    // ─── #560 Bedrock Converse tool calling ────────────────────────────
    //
    // OpenAI `tools` / `tool_choice` (carried in ChatFormat.extra) must
    // reach Converse's `toolConfig`, and Converse `toolUse` blocks must
    // come back as OpenAI `tool_calls`. Before #560 the Converse builder
    // never set `toolConfig`, so every non-Anthropic publisher silently
    // dropped tool calling and improvised the call as prose.

    /// Build a tools-bearing request whose `tools` / `tool_choice` land in
    /// `ChatFormat.extra` exactly as an inbound OpenAI request would.
    fn tools_request(tool_choice: serde_json::Value) -> ChatFormat {
        serde_json::from_value(serde_json::json!({
            "model": "my-llama",
            "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"]
                    }
                }
            }],
            "tool_choice": tool_choice,
        }))
        .expect("tools request deserializes")
    }

    /// Capture the outbound Converse body for a tools-bearing request.
    async fn capture_converse_body(req: &ChatFormat) -> serde_json::Value {
        let server = MockServer::start().await;
        let responder = CapturingResponder::default();
        Mock::given(method("POST"))
            .and(path_regex(r"^/model/.+/converse$"))
            .respond_with(responder.clone())
            .expect(1)
            .mount(&server)
            .await;
        let bridge = BedrockBridge::new().with_endpoint_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("meta.llama3-3-70b-instruct-v1:0"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let _ = bridge.chat(req, &ctx).await;
        let body = responder.captured_body.lock().unwrap().clone().unwrap();
        body
    }

    #[tokio::test]
    async fn chat_converse_forwards_tools_and_forced_tool_choice() {
        // Root cause of #560: the Converse builder never set `toolConfig`.
        // Pin that OpenAI `tools` translate to `toolConfig.tools[].toolSpec`
        // (schema verbatim) and a forced `tool_choice` to `toolChoice.tool`.
        let req = tools_request(serde_json::json!({
            "type": "function",
            "function": {"name": "get_weather"}
        }));
        let body = capture_converse_body(&req).await;

        let tool_config = body
            .get("toolConfig")
            .unwrap_or_else(|| panic!("toolConfig must be present; body={body}"));
        let tools = tool_config
            .get("tools")
            .and_then(|v| v.as_array())
            .unwrap_or_else(|| panic!("toolConfig.tools must be an array; body={body}"));
        assert_eq!(tools.len(), 1);
        let spec = tools[0]
            .get("toolSpec")
            .unwrap_or_else(|| panic!("tool must be a toolSpec; body={body}"));
        assert_eq!(
            spec.get("name").and_then(|v| v.as_str()),
            Some("get_weather")
        );
        assert_eq!(
            spec.get("description").and_then(|v| v.as_str()),
            Some("Get the current weather for a city.")
        );
        let schema = spec
            .get("inputSchema")
            .and_then(|s| s.get("json"))
            .unwrap_or_else(|| panic!("inputSchema.json must carry the params; body={body}"));
        assert_eq!(schema.get("type").and_then(|v| v.as_str()), Some("object"));
        assert!(
            schema
                .get("properties")
                .and_then(|p| p.get("location"))
                .is_some(),
            "OpenAI parameters JSON Schema must round-trip into inputSchema.json; schema={schema}"
        );
        assert_eq!(
            tool_config
                .get("toolChoice")
                .and_then(|c| c.get("tool"))
                .and_then(|t| t.get("name"))
                .and_then(|v| v.as_str()),
            Some("get_weather"),
            "forced tool_choice must map to toolChoice.tool; body={body}"
        );
    }

    #[tokio::test]
    async fn chat_converse_maps_required_tool_choice_to_any() {
        let body = capture_converse_body(&tools_request(serde_json::json!("required"))).await;
        let tool_config = body.get("toolConfig").unwrap();
        assert!(
            tool_config
                .get("toolChoice")
                .and_then(|c| c.get("any"))
                .is_some(),
            "tool_choice:required must map to toolChoice.any; body={body}"
        );
    }

    #[tokio::test]
    async fn chat_converse_auto_tool_choice_omits_explicit_choice() {
        // "auto" is Converse's default; we send the tools but omit an
        // explicit toolChoice so publishers that only accept the default
        // aren't rejected.
        let body = capture_converse_body(&tools_request(serde_json::json!("auto"))).await;
        let tool_config = body.get("toolConfig").unwrap();
        assert!(
            tool_config.get("tools").is_some(),
            "tools must still be sent"
        );
        assert!(
            tool_config.get("toolChoice").is_none(),
            "auto must omit an explicit toolChoice (Converse default); body={body}"
        );
    }

    #[tokio::test]
    async fn chat_converse_tool_choice_none_sends_no_tool_config() {
        // Converse has no `none`; the only faithful mapping is to send no
        // toolConfig at all so the model answers in prose.
        let body = capture_converse_body(&tools_request(serde_json::json!("none"))).await;
        assert!(
            body.get("toolConfig").is_none(),
            "tool_choice:none must send no toolConfig; body={body}"
        );
    }

    #[tokio::test]
    async fn chat_converse_translates_tool_use_response_to_tool_calls() {
        // The other half of #560: a Converse `toolUse` content block must
        // surface as an OpenAI `tool_calls` entry with `arguments` as a
        // JSON-encoded STRING, and `content: null` when there's no prose.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(r"^/model/.+/converse$"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "output": {"message": {"role": "assistant", "content": [
                    {"toolUse": {
                        "toolUseId": "tooluse_abc",
                        "name": "get_weather",
                        "input": {"location": "Paris"}
                    }}
                ]}},
                "stopReason": "tool_use",
                "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
                "metrics": {"latencyMs": 1}
            })))
            .expect(1)
            .mount(&server)
            .await;

        let bridge = BedrockBridge::new().with_endpoint_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("meta.llama3-3-70b-instruct-v1:0"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("my-llama", vec![ChatMessage::user("weather in Paris?")]);
        let chat = bridge.chat(&req, &ctx).await.unwrap();

        assert_eq!(chat.finish_reason, FinishReason::ToolCalls);
        let tool_calls = chat
            .message
            .extra
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .expect("tool_calls must be present");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(
            tool_calls[0].get("id").and_then(|v| v.as_str()),
            Some("tooluse_abc")
        );
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
        let arguments = tool_calls[0]
            .get("function")
            .and_then(|f| f.get("arguments"))
            .and_then(|v| v.as_str())
            .expect("arguments MUST be a JSON-encoded STRING per the OpenAI spec");
        let parsed: serde_json::Value =
            serde_json::from_str(arguments).expect("arguments string must itself be valid JSON");
        assert_eq!(
            parsed.get("location").and_then(|v| v.as_str()),
            Some("Paris")
        );
        assert!(
            chat.message.content.is_none(),
            "content must be null alongside tool_calls when there is no prose"
        );
    }

    #[test]
    fn build_converse_inputs_translates_tool_history_and_coalesces_results() {
        // Multi-turn replay: an assistant `tool_calls` history entry must
        // become `toolUse` blocks, and consecutive `role:"tool"` results
        // (parallel calls) must coalesce into ONE Converse user message.
        let req: ChatFormat = serde_json::from_value(serde_json::json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "weather in Paris and Tokyo?"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}},
                    {"id": "call_2", "type": "function",
                     "function": {"name": "get_weather", "arguments": "{\"city\":\"Tokyo\"}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": "15C"},
                {"role": "tool", "tool_call_id": "call_2", "content": "20C"}
            ]
        }))
        .unwrap();

        let (systems, messages) = build_converse_inputs(&req).unwrap();
        assert!(systems.is_empty());
        // user → assistant(2× toolUse) → user(2× toolResult, coalesced)
        assert_eq!(
            messages.len(),
            3,
            "two consecutive tool results must coalesce into one user message"
        );

        assert_eq!(*messages[1].role(), ConversationRole::Assistant);
        let tool_uses: Vec<_> = messages[1]
            .content()
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolUse(tu) => Some(tu),
                _ => None,
            })
            .collect();
        assert_eq!(
            tool_uses.len(),
            2,
            "both tool_calls must become toolUse blocks"
        );
        assert_eq!(tool_uses[0].tool_use_id(), "call_1");
        assert_eq!(tool_uses[0].name(), "get_weather");

        assert_eq!(*messages[2].role(), ConversationRole::User);
        let results: Vec<_> = messages[2]
            .content()
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolResult(tr) => Some(tr),
                _ => None,
            })
            .collect();
        assert_eq!(
            results.len(),
            2,
            "both tool results must share ONE user message"
        );
        assert_eq!(results[0].tool_use_id(), "call_1");
        assert_eq!(results[1].tool_use_id(), "call_2");
    }

    #[test]
    fn emit_converse_chunk_streams_tool_call_deltas() {
        use aws_sdk_bedrockruntime::types::{
            ContentBlockDeltaEvent, ContentBlockStartEvent, ToolUseBlockDelta, ToolUseBlockStart,
        };
        let mut emitted_role = false;
        let mut tool_blocks: Vec<i32> = Vec::new();

        // A tool-use block opens at contentBlockIndex 0.
        let start = ConverseStreamOutput::ContentBlockStart(
            ContentBlockStartEvent::builder()
                .content_block_index(0)
                .start(ContentBlockStart::ToolUse(
                    ToolUseBlockStart::builder()
                        .tool_use_id("tooluse_1")
                        .name("get_weather")
                        .build()
                        .unwrap(),
                ))
                .build()
                .unwrap(),
        );
        let chunks = emit_converse_chunk(start, "m", &mut emitted_role, &mut tool_blocks);
        assert_eq!(chunks.len(), 1);
        let call = &chunks[0].delta.tool_calls.as_ref().unwrap()[0];
        assert_eq!(call.get("index").and_then(|v| v.as_i64()), Some(0));
        assert_eq!(call.get("id").and_then(|v| v.as_str()), Some("tooluse_1"));
        assert_eq!(call.get("type").and_then(|v| v.as_str()), Some("function"));
        assert_eq!(
            call.get("function")
                .and_then(|f| f.get("name"))
                .and_then(|v| v.as_str()),
            Some("get_weather")
        );

        // Arguments stream in via a ToolUse delta on the same block index.
        let delta = ConverseStreamOutput::ContentBlockDelta(
            ContentBlockDeltaEvent::builder()
                .content_block_index(0)
                .delta(ContentBlockDelta::ToolUse(
                    ToolUseBlockDelta::builder()
                        .input("{\"location\":\"Paris\"}")
                        .build()
                        .unwrap(),
                ))
                .build()
                .unwrap(),
        );
        let chunks = emit_converse_chunk(delta, "m", &mut emitted_role, &mut tool_blocks);
        assert_eq!(chunks.len(), 1);
        let call = &chunks[0].delta.tool_calls.as_ref().unwrap()[0];
        assert_eq!(call.get("index").and_then(|v| v.as_i64()), Some(0));
        assert!(
            call.get("id").is_none(),
            "an arguments-continuation chunk carries no id/type"
        );
        assert_eq!(
            call.get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(|v| v.as_str()),
            Some("{\"location\":\"Paris\"}")
        );
    }

    #[test]
    fn emit_converse_chunk_tool_call_index_is_dense_despite_leading_text() {
        // The headline streaming claim: a leading text block (contentBlockIndex
        // 0) must NOT shift the first tool call's OpenAI index, and parallel
        // tool calls must get dense 0-based indices (0, 1) — not their raw
        // Converse block indices (1, 2).
        use aws_sdk_bedrockruntime::types::{
            ContentBlockDeltaEvent, ContentBlockStartEvent, ToolUseBlockDelta, ToolUseBlockStart,
        };
        let mut emitted_role = false;
        let mut tool_blocks: Vec<i32> = Vec::new();

        // A text block streams first at contentBlockIndex 0.
        let text = ConverseStreamOutput::ContentBlockDelta(
            ContentBlockDeltaEvent::builder()
                .content_block_index(0)
                .delta(ContentBlockDelta::Text("Let me check. ".to_string()))
                .build()
                .unwrap(),
        );
        let chunks = emit_converse_chunk(text, "m", &mut emitted_role, &mut tool_blocks);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].delta.content.as_deref(), Some("Let me check. "));

        // Two tool-use blocks open at contentBlockIndex 1 and 2.
        let mut open_indices = Vec::new();
        for (block_index, id) in [(1, "tu_a"), (2, "tu_b")] {
            let start = ConverseStreamOutput::ContentBlockStart(
                ContentBlockStartEvent::builder()
                    .content_block_index(block_index)
                    .start(ContentBlockStart::ToolUse(
                        ToolUseBlockStart::builder()
                            .tool_use_id(id)
                            .name("get_weather")
                            .build()
                            .unwrap(),
                    ))
                    .build()
                    .unwrap(),
            );
            let chunks = emit_converse_chunk(start, "m", &mut emitted_role, &mut tool_blocks);
            open_indices.push(
                chunks[0].delta.tool_calls.as_ref().unwrap()[0]["index"]
                    .as_i64()
                    .unwrap(),
            );
        }
        assert_eq!(
            open_indices,
            vec![0, 1],
            "tool calls must get dense 0-based indices despite the leading text block at index 0"
        );

        // An arguments fragment on block 2 must correlate to dense index 1.
        let frag = ConverseStreamOutput::ContentBlockDelta(
            ContentBlockDeltaEvent::builder()
                .content_block_index(2)
                .delta(ContentBlockDelta::ToolUse(
                    ToolUseBlockDelta::builder().input("{}").build().unwrap(),
                ))
                .build()
                .unwrap(),
        );
        let chunks = emit_converse_chunk(frag, "m", &mut emitted_role, &mut tool_blocks);
        assert_eq!(
            chunks[0].delta.tool_calls.as_ref().unwrap()[0]["index"].as_i64(),
            Some(1),
            "the second tool call's argument fragments must carry dense index 1"
        );
    }

    #[tokio::test]
    async fn chat_converse_preserves_prose_alongside_tool_calls() {
        // When a Converse response carries BOTH a text block and a toolUse
        // block, the assistant `content` must be preserved (non-null)
        // alongside tool_calls — the OpenAI prose-plus-tool_calls shape.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(r"^/model/.+/converse$"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "output": {"message": {"role": "assistant", "content": [
                    {"text": "On it."},
                    {"toolUse": {
                        "toolUseId": "tu_1",
                        "name": "get_weather",
                        "input": {"location": "Paris"}
                    }}
                ]}},
                "stopReason": "tool_use",
                "usage": {"inputTokens": 9, "outputTokens": 4, "totalTokens": 13},
                "metrics": {"latencyMs": 1}
            })))
            .expect(1)
            .mount(&server)
            .await;

        let bridge = BedrockBridge::new().with_endpoint_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("meta.llama3-3-70b-instruct-v1:0"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = ChatFormat::new("my-llama", vec![ChatMessage::user("weather in Paris?")]);
        let chat = bridge.chat(&req, &ctx).await.unwrap();

        assert_eq!(
            chat.message.content.as_deref(),
            Some("On it."),
            "prose must be preserved as content when tool_calls are also present"
        );
        let tool_calls = chat
            .message
            .extra
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .expect("tool_calls must be present");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(chat.finish_reason, FinishReason::ToolCalls);
    }

    #[tokio::test]
    async fn chat_converse_skips_tool_with_non_object_parameters() {
        // A malformed tool (non-object `parameters`) must be skipped rather
        // than forwarded as an invalid inputSchema that Bedrock would 400 on;
        // a valid sibling tool still goes through.
        let req: ChatFormat = serde_json::from_value(serde_json::json!({
            "model": "my-llama",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {"type": "function", "function": {"name": "bad", "parameters": 42}},
                {"type": "function", "function": {
                    "name": "good",
                    "parameters": {"type": "object", "properties": {}}
                }}
            ],
        }))
        .expect("request deserializes");
        let body = capture_converse_body(&req).await;
        let tools = body
            .get("toolConfig")
            .and_then(|c| c.get("tools"))
            .and_then(|v| v.as_array())
            .unwrap_or_else(|| panic!("toolConfig.tools must be present; body={body}"));
        assert_eq!(
            tools.len(),
            1,
            "the malformed tool must be skipped; body={body}"
        );
        assert_eq!(
            tools[0]
                .get("toolSpec")
                .and_then(|s| s.get("name"))
                .and_then(|v| v.as_str()),
            Some("good"),
        );
    }

    // ─── RequestOverrides applied on the wire (#340) ───────────────────
    //
    // Pin that the per-`ProviderKey` override pipeline actually reshapes the
    // OUTBOUND Bedrock request. Body transforms ride the `/invoke` (Anthropic)
    // JSON body; `default_headers` ride the pre-signing interceptor and land
    // inside the SigV4-signed request on BOTH the /invoke and Converse paths.

    /// Build a Bedrock `ProviderKey` carrying a `request` override block.
    fn sample_pk_with_request_overrides(request: serde_json::Value) -> Arc<ProviderKey> {
        Arc::new(
            serde_json::from_value(serde_json::json!({
                "display_name": "bedrock-prod",
                "secret": valid_secret_json(),
                "request": request,
            }))
            .expect("provider_key with request overrides deserializes"),
        )
    }

    /// Build a Bedrock `ProviderKey` carrying a `response` override block.
    fn sample_pk_with_response_overrides(response: serde_json::Value) -> Arc<ProviderKey> {
        Arc::new(
            serde_json::from_value(serde_json::json!({
                "display_name": "bedrock-prod",
                "secret": valid_secret_json(),
                "response": response,
            }))
            .expect("provider_key with response overrides deserializes"),
        )
    }

    /// Capturing responder for the Converse path: records request headers and
    /// returns the canned Converse envelope so the SDK call succeeds.
    #[derive(Clone, Default)]
    struct CapturingConverseResponder {
        captured_headers: std::sync::Arc<std::sync::Mutex<Option<http::HeaderMap>>>,
    }

    impl Respond for CapturingConverseResponder {
        fn respond(&self, req: &MockRequest) -> ResponseTemplate {
            *self.captured_headers.lock().unwrap() = Some(req.headers.clone());
            default_converse_response_template()
        }
    }

    #[tokio::test]
    async fn bedrock_applies_param_renames() {
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
            sample_pk_with_request_overrides(serde_json::json!({
                "param_renames": { "temperature": "temperature_legacy" }
            })),
        );
        let mut req = ChatFormat::new("my-claude", vec![ChatMessage::user("hi")]);
        req.temperature = Some(0.5);
        bridge.chat(&req, &ctx).await.unwrap();

        let body = responder.captured_body.lock().unwrap().clone().unwrap();
        assert_eq!(
            body.get("temperature_legacy").and_then(|v| v.as_f64()),
            Some(0.5),
            "param_renames must move `temperature` to the renamed key on the /invoke body; got {body}",
        );
        assert!(
            body.get("temperature").is_none(),
            "the source key must be gone after the rename; got {body}",
        );
    }

    #[tokio::test]
    async fn bedrock_clamps_temperature_via_param_constraints() {
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
            sample_pk_with_request_overrides(serde_json::json!({
                "param_constraints": { "temperature_max": 1.0 }
            })),
        );
        let mut req = ChatFormat::new("my-claude", vec![ChatMessage::user("hi")]);
        req.temperature = Some(1.9);
        bridge.chat(&req, &ctx).await.unwrap();

        let body = responder.captured_body.lock().unwrap().clone().unwrap();
        assert_eq!(
            body.get("temperature").and_then(|v| v.as_f64()),
            Some(1.0),
            "param_constraints must clamp the over-max temperature; got {body}",
        );
    }

    #[tokio::test]
    async fn bedrock_fills_default_body_fields_when_caller_omits() {
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
            sample_pk_with_request_overrides(serde_json::json!({
                "default_body_fields": { "top_k": 5 }
            })),
        );
        let req = ChatFormat::new("my-claude", vec![ChatMessage::user("hi")]);
        bridge.chat(&req, &ctx).await.unwrap();

        let body = responder.captured_body.lock().unwrap().clone().unwrap();
        assert_eq!(
            body.get("top_k").and_then(|v| v.as_u64()),
            Some(5),
            "default_body_fields must inject the absent key into the /invoke body; got {body}",
        );
    }

    #[tokio::test]
    async fn bedrock_injects_default_headers() {
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
            sample_pk_with_request_overrides(serde_json::json!({
                "default_headers": { "x-corp-trace": "abc123" }
            })),
        );
        let req = ChatFormat::new("my-claude", vec![ChatMessage::user("hi")]);
        bridge.chat(&req, &ctx).await.unwrap();

        let headers = responder.captured_headers.lock().unwrap().clone().unwrap();
        assert_eq!(
            headers.get("x-corp-trace").and_then(|v| v.to_str().ok()),
            Some("abc123"),
            "default_headers must reach the outbound Bedrock request",
        );
        // The custom header must NOT have broken signing: a valid SigV4
        // Authorization header is still present on the captured request.
        let auth = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .expect("a SigV4 Authorization header");
        assert!(
            auth.starts_with("AWS4-HMAC-SHA256"),
            "the request must still carry a valid SigV4 Authorization; got {auth}",
        );
        // Prove the injected header was SIGNED — it appears in the SigV4
        // `SignedHeaders=` list — rather than merely appended after signing.
        // A post-signing injection would leave a valid-looking Authorization
        // but omit `x-corp-trace` from SignedHeaders, so this assertion is
        // what actually pins `modify_before_signing` ordering. (#462 audit LOW-1)
        assert!(
            auth.contains("SignedHeaders=") && auth.contains("x-corp-trace"),
            "the injected default header must be covered by the SigV4 signature \
             (present in SignedHeaders); auth={auth}",
        );
    }

    #[tokio::test]
    async fn bedrock_default_headers_cannot_overwrite_x_amz_date() {
        // SigV4 timestamp guard: an operator default_headers entry naming the
        // reserved `x-amz-date` must be dropped, leaving the SDK's own signed
        // timestamp intact. A non-reserved companion header still applies.
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
            sample_pk_with_request_overrides(serde_json::json!({
                "default_headers": {
                    "x-amz-date": "20200101T000000Z",
                    "x-ok": "1"
                }
            })),
        );
        let req = ChatFormat::new("my-claude", vec![ChatMessage::user("hi")]);
        bridge.chat(&req, &ctx).await.unwrap();

        let headers = responder.captured_headers.lock().unwrap().clone().unwrap();
        let amz_date = headers.get("x-amz-date").and_then(|v| v.to_str().ok());
        assert!(
            amz_date.is_some() && amz_date != Some("20200101T000000Z"),
            "the SDK's signed x-amz-date must survive a reserved-header override; got {amz_date:?}",
        );
        assert_eq!(
            headers.get("x-ok").and_then(|v| v.to_str().ok()),
            Some("1"),
            "the non-reserved companion default header still applies",
        );
    }

    #[tokio::test]
    async fn bedrock_default_headers_apply_on_converse_path() {
        // The interceptor is registered on the client config, so default
        // headers reach the Converse path (non-Anthropic publishers) too.
        let server = MockServer::start().await;
        let responder = CapturingConverseResponder::default();
        Mock::given(method("POST"))
            .and(path_regex(r"^/model/.+/converse$"))
            .respond_with(responder.clone())
            .expect(1)
            .mount(&server)
            .await;

        let bridge = BedrockBridge::new().with_endpoint_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("meta.llama3-3-70b-instruct-v1:0"),
            sample_pk_with_request_overrides(serde_json::json!({
                "default_headers": { "x-corp-trace": "xyz789" }
            })),
        );
        let req = ChatFormat::new("my-llama", vec![ChatMessage::user("hi")]);
        bridge.chat(&req, &ctx).await.unwrap();

        let headers = responder.captured_headers.lock().unwrap().clone().unwrap();
        assert_eq!(
            headers.get("x-corp-trace").and_then(|v| v.to_str().ok()),
            Some("xyz789"),
            "default_headers must reach the Converse path too",
        );
        assert!(
            headers
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .is_some_and(|a| a.starts_with("AWS4-HMAC-SHA256")),
            "Converse request must still carry a valid SigV4 Authorization; headers={headers:?}",
        );
    }

    #[tokio::test]
    async fn bedrock_overrides_run_before_model_strip() {
        // The body pipeline runs BEFORE the Bedrock model/stream strip, so a
        // default_body_fields block can never reintroduce a URL-borne model
        // into the /invoke body — while a genuine extra field still lands.
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
            sample_pk_with_request_overrides(serde_json::json!({
                "default_body_fields": { "model": "should-be-stripped", "top_k": 5 }
            })),
        );
        let req = ChatFormat::new("my-claude", vec![ChatMessage::user("hi")]);
        bridge.chat(&req, &ctx).await.unwrap();

        let body = responder.captured_body.lock().unwrap().clone().unwrap();
        let obj = body.as_object().unwrap();
        assert!(
            !obj.contains_key("model"),
            "model must stay stripped even when default_body_fields tries to set it; got {body}",
        );
        assert_eq!(
            obj.get("top_k").and_then(|v| v.as_u64()),
            Some(5),
            "a genuine extra default body field still lands; got {body}",
        );
        assert_eq!(
            obj.get("anthropic_version").and_then(|v| v.as_str()),
            Some(BEDROCK_ANTHROPIC_VERSION),
            "the Bedrock anthropic_version shaping still runs after overrides; got {body}",
        );
    }

    #[tokio::test]
    async fn bedrock_response_content_list_to_string_flattens_outbound_body() {
        // The `content_list_to_string` branch is gated on the response
        // override block and flattens array content to a string on the body.
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
            sample_pk_with_response_overrides(serde_json::json!({
                "content_list_to_string": true
            })),
        );
        let mut msg = ChatMessage::user("abc");
        msg.content_blocks = Some(vec![
            serde_json::json!({"type": "text", "text": "a"}),
            serde_json::json!({"type": "text", "text": "b"}),
            serde_json::json!({"type": "text", "text": "c"}),
        ]);
        let req = ChatFormat::new("my-claude", vec![msg]);
        bridge.chat(&req, &ctx).await.unwrap();

        let body = responder.captured_body.lock().unwrap().clone().unwrap();
        assert_eq!(
            body["messages"][0]["content"].as_str(),
            Some("abc"),
            "response.content_list_to_string must flatten array content to a string; got {body}",
        );
    }

    // ─── #463: param_constraints temperature clamp on the Converse path ───
    //
    // The /invoke body path runs apply_param_constraints; the Converse path
    // builds a typed inferenceConfig and must re-apply the same clamp so a
    // Bedrock-wide temperature ceiling holds for non-Anthropic publishers too.
    // These mirror the existing Converse wires-temperature test: drive a real
    // SDK Converse round-trip, ignore the response, assert the CAPTURED request
    // body's inferenceConfig.temperature.

    #[tokio::test]
    async fn chat_converse_clamps_temperature_via_param_constraints() {
        let server = MockServer::start().await;
        let responder = CapturingResponder::default();
        Mock::given(method("POST"))
            .and(path_regex(r"^/model/.+/converse$"))
            .respond_with(responder.clone())
            .mount(&server)
            .await;

        let bridge = BedrockBridge::new().with_endpoint_override(server.uri());
        let mut req = ChatFormat::new("my-llama", vec![ChatMessage::user("hi")]);
        req.temperature = Some(1.9); // over the ceiling
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("meta.llama3-3-70b-instruct-v1:0"), // Converse publisher
            sample_pk_with_request_overrides(serde_json::json!({
                "param_constraints": { "temperature_max": 1.0 }
            })),
        );
        let _ = bridge.chat(&req, &ctx).await;

        let body = responder.captured_body.lock().unwrap().clone().unwrap();
        let cfg = body
            .get("inferenceConfig")
            .and_then(|v| v.as_object())
            .expect("inferenceConfig must be set when temperature is carried");
        assert_eq!(
            cfg.get("temperature").and_then(|v| v.as_f64()),
            Some(1.0),
            "Converse temperature must be clamped to temperature_max; body={body}",
        );
    }

    #[tokio::test]
    async fn chat_converse_clamps_temperature_min_via_param_constraints() {
        let server = MockServer::start().await;
        let responder = CapturingResponder::default();
        Mock::given(method("POST"))
            .and(path_regex(r"^/model/.+/converse$"))
            .respond_with(responder.clone())
            .mount(&server)
            .await;

        let bridge = BedrockBridge::new().with_endpoint_override(server.uri());
        let mut req = ChatFormat::new("my-llama", vec![ChatMessage::user("hi")]);
        req.temperature = Some(0.1); // under the floor
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("meta.llama3-3-70b-instruct-v1:0"),
            sample_pk_with_request_overrides(serde_json::json!({
                "param_constraints": { "temperature_min": 0.5 }
            })),
        );
        let _ = bridge.chat(&req, &ctx).await;

        let body = responder.captured_body.lock().unwrap().clone().unwrap();
        let cfg = body
            .get("inferenceConfig")
            .and_then(|v| v.as_object())
            .expect("inferenceConfig must be set when temperature is carried");
        assert_eq!(
            cfg.get("temperature").and_then(|v| v.as_f64()),
            Some(0.5),
            "Converse temperature must be clamped up to temperature_min; body={body}",
        );
    }

    #[tokio::test]
    async fn chat_converse_temperature_clamp_does_not_panic_on_min_gt_max() {
        // Panic-safety regression: a misconfigured `temperature_min >
        // temperature_max` must NOT panic the request (`f32::clamp` would).
        // The two-`if` clamp applies max then min, so min wins — matching the
        // /invoke path's `apply_param_constraints`. The test merely completing
        // (no panic) is the core assertion; the value pins the max-then-min order.
        let server = MockServer::start().await;
        let responder = CapturingResponder::default();
        Mock::given(method("POST"))
            .and(path_regex(r"^/model/.+/converse$"))
            .respond_with(responder.clone())
            .mount(&server)
            .await;

        let bridge = BedrockBridge::new().with_endpoint_override(server.uri());
        let mut req = ChatFormat::new("my-llama", vec![ChatMessage::user("hi")]);
        req.temperature = Some(0.5);
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("meta.llama3-3-70b-instruct-v1:0"),
            sample_pk_with_request_overrides(serde_json::json!({
                "param_constraints": { "temperature_min": 1.0, "temperature_max": 0.0 }
            })),
        );
        let _ = bridge.chat(&req, &ctx).await;

        let body = responder.captured_body.lock().unwrap().clone().unwrap();
        let cfg = body
            .get("inferenceConfig")
            .and_then(|v| v.as_object())
            .expect("inferenceConfig must be set when temperature is carried");
        assert_eq!(
            cfg.get("temperature").and_then(|v| v.as_f64()),
            Some(1.0),
            "max-then-min order: min wins on a misconfigured min>max; body={body}",
        );
    }

    #[tokio::test]
    async fn chat_converse_stream_clamps_temperature_via_param_constraints() {
        // #468 audit MEDIUM-1: the stream path (`chat_converse_stream`) wires
        // the same clamp as `chat_converse` but lacked a regression guard.
        // Capture the outbound `/converse-stream` body and assert the clamp
        // reached the wire. (The SDK sends + wiremock captures before the
        // eventstream decode fails on the non-eventstream response, so the
        // ignored `chat_stream` result is fine — same pattern as the
        // non-stream clamp tests.)
        let server = MockServer::start().await;
        let responder = CapturingResponder::default();
        Mock::given(method("POST"))
            .and(path_regex(r"^/model/.+/converse-stream$"))
            .respond_with(responder.clone())
            .mount(&server)
            .await;

        let bridge = BedrockBridge::new().with_endpoint_override(server.uri());
        let mut req = ChatFormat::new("my-llama", vec![ChatMessage::user("hi")]);
        req.temperature = Some(1.9); // over the ceiling
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("meta.llama3-3-70b-instruct-v1:0"),
            sample_pk_with_request_overrides(serde_json::json!({
                "param_constraints": { "temperature_max": 1.0 }
            })),
        );
        let _ = bridge.chat_stream(&req, &ctx).await;

        let body = responder.captured_body.lock().unwrap().clone().unwrap();
        let cfg = body
            .get("inferenceConfig")
            .and_then(|v| v.as_object())
            .expect("inferenceConfig must be set when temperature is carried");
        assert_eq!(
            cfg.get("temperature").and_then(|v| v.as_f64()),
            Some(1.0),
            "stream-path Converse temperature must be clamped to temperature_max; body={body}",
        );
    }

    // ─── #723: native embeddings via InvokeModel ──────────────────────

    #[tokio::test]
    async fn titan_embed_loops_one_invoke_per_input() {
        use wiremock::matchers::{method as wm_method, path as wm_path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(wm_method("POST"))
            .and(wm_path("/model/amazon.titan-embed-text-v1/invoke"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "embedding": [0.25, 0.5],
                "inputTextTokenCount": 4
            })))
            .mount(&server)
            .await;

        let bridge = BedrockBridge::new().with_endpoint_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("amazon.titan-embed-text-v1"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = EmbeddingRequest {
            model: "customer-facing-name".into(),
            input: vec!["a".into(), "b".into()],
            input_was_single: false,
            encoding_format: None,
            dimensions: None,
        };
        let resp = bridge.embed(&req, &ctx).await.expect("titan embed");
        assert_eq!(resp.data.len(), 2);
        assert_eq!(resp.data[1].index, 1);
        assert_eq!(resp.usage.prompt_tokens, 8, "4 tokens per input, summed");

        // Titan embeds ONE text per call -> two upstream invokes, in order.
        let received = server.received_requests().await.unwrap();
        assert_eq!(received.len(), 2);
        let first: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
        assert_eq!(first, serde_json::json!({"inputText": "a"}));
        // SigV4 signing ran (SDK path) — the Authorization header carries
        // the AWS algorithm marker.
        assert!(received[0]
            .headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .contains("AWS4-HMAC-SHA256"));
    }

    #[tokio::test]
    async fn titan_v2_embed_forwards_dimensions_but_g1_does_not() {
        use wiremock::matchers::method as wm_method;
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(wm_method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "embedding": [0.1],
                "inputTextTokenCount": 1
            })))
            .mount(&server)
            .await;

        let bridge = BedrockBridge::new().with_endpoint_override(server.uri());
        let req = EmbeddingRequest {
            model: "customer-facing-name".into(),
            input: vec!["a".into()],
            input_was_single: true,
            encoding_format: None,
            dimensions: Some(512),
        };

        // V2: dimensions forwards.
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("amazon.titan-embed-text-v2:0"),
            sample_pk_with_secret(valid_secret_json()),
        );
        bridge.embed(&req, &ctx).await.expect("v2 embed");
        // G1: dimensions must be omitted (Titan G1 rejects unknown keys).
        let ctx = BridgeContext::new(
            "req-2",
            sample_model_with("amazon.titan-embed-text-v1"),
            sample_pk_with_secret(valid_secret_json()),
        );
        bridge.embed(&req, &ctx).await.expect("g1 embed");

        let received = server.received_requests().await.unwrap();
        assert_eq!(received.len(), 2);
        let v2_body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
        assert_eq!(v2_body["dimensions"], 512);
        let g1_body: serde_json::Value = serde_json::from_slice(&received[1].body).unwrap();
        assert!(g1_body.get("dimensions").is_none());
    }

    #[tokio::test]
    async fn cohere_embed_sends_whole_batch_in_one_call() {
        use wiremock::matchers::{method as wm_method, path as wm_path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(wm_method("POST"))
            .and(wm_path("/model/cohere.embed-english-v3/invoke"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "embeddings": [[0.1, 0.2], [0.3, 0.4]]
            })))
            .mount(&server)
            .await;

        let bridge = BedrockBridge::new().with_endpoint_override(server.uri());
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("cohere.embed-english-v3"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = EmbeddingRequest {
            model: "customer-facing-name".into(),
            input: vec!["a".into(), "b".into()],
            input_was_single: false,
            encoding_format: None,
            dimensions: None,
        };
        let resp = bridge.embed(&req, &ctx).await.expect("cohere embed");
        assert_eq!(resp.data.len(), 2);

        let received = server.received_requests().await.unwrap();
        assert_eq!(received.len(), 1, "cohere batches in a single call");
        let body: serde_json::Value = serde_json::from_slice(&received[0].body).unwrap();
        assert_eq!(body["texts"], serde_json::json!(["a", "b"]));
        assert_eq!(body["input_type"], "search_document");
    }

    #[tokio::test]
    async fn embed_rejects_non_embedding_publishers() {
        let bridge = BedrockBridge::new();
        let ctx = BridgeContext::new(
            "req-1",
            sample_model_with("anthropic.claude-3-5-sonnet-20241022-v2:0"),
            sample_pk_with_secret(valid_secret_json()),
        );
        let req = EmbeddingRequest {
            model: "customer-facing-name".into(),
            input: vec!["a".into()],
            input_was_single: true,
            encoding_format: None,
            dimensions: None,
        };
        let err = bridge.embed(&req, &ctx).await.unwrap_err();
        match err {
            BridgeError::Config(msg) => {
                assert!(msg.contains("amazon.titan-embed-"));
                assert!(msg.contains("cohere.embed-"));
            }
            other => panic!("expected Config error, got {other:?}"),
        }
    }
}
