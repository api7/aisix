//! `Guardrail` entity — content-policy hooks the DP runs on every
//! chat request. The control plane (cp-api) writes these to etcd at
//! `/aisix/<env>/guardrails/<uuid>`; the DP loads them on watch and
//! the `aisix-proxy::ProxyState::guardrail_index` resolves the
//! applicable chain per request.
//!
//! P0b added `enforcement_mode`, `mandatory`, and `direction` columns
//! to the CP `guardrails` table. P0c wires them to the kine payload
//! and adds the `GuardrailAttachment` row type (`/aisix/<env>/guardrail_attachments/<uuid>`).
//! The outer `Guardrail` struct accepts but defaults the three new fields
//! so old kine rows (written before P0c CP lands) still parse.
//!
//! Two run sites per request (matches `aisix-guardrails::Guardrail`):
//!   * `input`  — runs before bridge dispatch; a block here means the
//!     prompt never reaches the upstream.
//!   * `output` — runs after the upstream response lands; a block
//!     here means the response never reaches the caller.
//!
//! Production keeps both sides on by default. The `hook_point` field
//! lets operators narrow a rule to just one side (e.g. a PII regex
//! that's expensive to run on long outputs).
//!
//! Rule kinds:
//!
//!   * `keyword` — literal/regex blocklist; runs entirely in DP
//!     process. Configured via `keyword.patterns` (list of
//!     `{ kind: "literal" | "regex", value: "..." }`).
//!   * `bedrock` — calls AWS Bedrock's `ApplyGuardrail` on input
//!     and/or output. The DP signs the call with SigV4 and maps
//!     `GUARDRAIL_INTERVENED` to a block (PRD-09c §6.7).
//!   * `azure_content_safety` — calls Azure AI Content Safety Prompt
//!     Shield (`/contentsafety/text:shieldPrompt`). Detects jailbreak
//!     and indirect injection attacks. P1 (PRD-09c §6 P1).
//!   * `aliyun_text_moderation` — calls Aliyun's content-safety
//!     guardrail (`TextModerationPlus` on `green-cip.<region>.aliyuncs.com`).
//!     Risk-level moderation on input (`llm_query_moderation`) and output
//!     (`llm_response_moderation`).
//!   * `pii` — in-process sensitive-data detection + redaction
//!     (built-in detectors + custom regex, `mask`/`block`).
//!   * `lakera` — calls Lakera Guard `/v2/guard`; injection/jailbreak
//!     blocks, PII-only detections mask via returned offsets.
//!   * `openai_moderation` — calls the OpenAI Moderation API;
//!     detection-only block.
//!   * `presidio` — self-hosted Presidio analyze→anonymize; per-entity
//!     `mask`/`block` + selectable anonymize operator.
//!
//! See `aisix-guardrails/src/keyword.rs` for the runtime semantics
//! the snapshot is parsed into.

use serde::{Deserialize, Serialize};

use crate::resource::Resource;

/// What part of the request lifecycle a guardrail inspects.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum GuardrailHookPoint {
    /// Run on the request payload before the upstream call.
    Input,
    /// Run on the upstream response before the cache write + render.
    Output,
    /// Run on both input and output.
    #[default]
    Both,
}

/// Literal or regular-expression pattern used by a keyword guardrail.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(tag = "kind", content = "value", rename_all = "lowercase")]
pub enum KeywordPattern {
    /// Literal string to match.
    Literal(#[schemars(length(min = 1))] String),
    /// Regular expression pattern to match.
    Regex(#[schemars(length(min = 1))] String),
}

/// Config block for `kind: "keyword"`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct KeywordConfig {
    /// Blocklist patterns. An empty list is valid and allows every
    /// request, equivalent to `enabled: false`.
    pub patterns: Vec<KeywordPattern>,
}

/// AWS credentials for a Bedrock guardrail.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum BedrockAWSCredentials {
    Static {
        /// AWS access key ID for static Bedrock guardrail credentials.
        #[schemars(length(min = 1))]
        access_key_id: String,
        /// Decrypted before projection. Plaintext is held in memory only
        /// and is not logged. The data plane passes it to the
        /// AWS SDK's static credentials provider.
        #[schemars(length(min = 1))]
        secret_access_key: String,
    },
}

/// Per-guardrail latency policy for `kind: "bedrock"`. Serial mode waits
/// for the guardrail response; timed mode aborts at `timeout_ms` and applies
/// `fail_open`.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum BedrockLatencyMode {
    Serial,
    Timed {
        /// Maximum time in milliseconds to wait for the Bedrock guardrail response.
        #[schemars(range(min = 100, max = 5000))]
        timeout_ms: u32,
    },
}

/// Config block for `kind: "azure_content_safety"`. Calls Azure AI
/// Content Safety Prompt Shield API to detect jailbreak and indirect
/// injection attacks.
///
/// The CP (cp-api) decrypts the envelope-encrypted `api_key` at kine-
/// projection time so the DP always holds plaintext in memory. The
/// key is never logged.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AzureContentSafetyConfig {
    /// Azure Cognitive Services resource endpoint, e.g.
    /// `https://my-resource.cognitiveservices.azure.com`.
    /// The data plane appends `/contentsafety/text:shieldPrompt?api-version=2024-09-01`.
    #[schemars(length(min = 1))]
    pub endpoint: String,
    /// Azure subscription key sent with the `Ocp-Apim-Subscription-Key` header. Decrypted before
    /// projection. Plaintext is held in memory only and is not logged.
    #[schemars(length(min = 1))]
    pub api_key: String,
    /// HTTP call timeout in milliseconds. A value of `0` triggers the timeout immediately.
    #[serde(default = "default_acs_timeout_ms")]
    #[schemars(range(max = 4_294_967_295u32))]
    pub timeout_ms: u32,
    /// Fail-open policy for the output hook. When disabled (the default), an
    /// Azure outage blocks model output instead of releasing unscanned content.
    /// The input hook continues to use the top-level `fail_open` policy.
    #[serde(default)]
    pub output_fail_open: bool,
}

fn default_acs_timeout_ms() -> u32 {
    5_000
}

/// Config block for `kind: "azure_content_safety_text_moderation"`. Calls
/// Azure AI Content Safety `text:analyze` for category-severity and blocklist
/// moderation on input and/or output, including streaming output.
///
/// Reuses the P1 connection block (endpoint + api_key + timeout_ms). cp-api
/// projects only operator-set fields (omitempty), so every optional field
/// carries a serde default matching the cp-api validator's documented
/// default. Only `api_key` is a secret (decrypted by cp-api before kine
/// projection). Every other field travels in the clear.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AzureContentSafetyTextModerationConfig {
    /// Azure Cognitive Services resource endpoint. The data plane appends
    /// `/contentsafety/text:analyze?api-version=2024-09-01`.
    #[schemars(length(min = 1))]
    pub endpoint: String,
    /// Azure subscription key sent with the `Ocp-Apim-Subscription-Key` header. Plaintext is held in
    /// memory only and is not logged.
    #[schemars(length(min = 1))]
    pub api_key: String,
    /// HTTP call timeout in milliseconds. `fail_open` and `output_fail_open`
    /// govern the verdict when it elapses. A value of `0` triggers the timeout
    /// immediately.
    #[serde(default = "default_acs_timeout_ms")]
    #[schemars(range(max = 4_294_967_295u32))]
    pub timeout_ms: u32,

    // --- moderation parameters ---
    /// Severity scale used for Azure category scores. The four-level scale
    /// returns even severities; the eight-level scale returns every integer
    /// severity.
    #[serde(default = "default_acs_output_type")]
    pub output_type: String,
    /// Categories to analyze.
    #[serde(default = "default_acs_categories")]
    pub categories: Vec<String>,
    /// General severity threshold. A category at or above it blocks.
    #[serde(default = "default_acs_severity_threshold")]
    #[schemars(range(max = 7))]
    pub severity_threshold: u8,
    /// Per-category threshold overrides. These take precedence over the general threshold.
    #[serde(default)]
    pub severity_threshold_by_category: std::collections::BTreeMap<String, u8>,
    /// Azure CS blocklist names to match against.
    #[serde(default)]
    pub blocklist_names: Vec<String>,
    /// Forwarded to Azure's `haltOnBlocklistHit`.
    #[serde(default)]
    pub halt_on_blocklist_hit: bool,
    /// Input-hook text selection. The default scans user messages only; the
    /// alternate mode includes all message content. Ignored on the output hook.
    #[serde(default = "default_acs_text_source")]
    pub text_source: String,

    // --- streaming-output controls (consumed by aisix-proxy build_sse_stream) ---
    /// Streaming output moderation mode: sliding-window incremental release
    /// or whole-response hold-back.
    #[serde(default = "default_acs_stream_processing_mode")]
    pub stream_processing_mode: String,
    /// Sliding-window size in characters for window mode.
    #[serde(default = "default_acs_window_size")]
    #[schemars(range(min = 1, max = 10_000))]
    pub window_size: u32,
    /// Chars carried between windows so a span split across a boundary is still caught.
    #[serde(default = "default_acs_window_overlap_size")]
    pub window_overlap_size: u32,
    /// Max bytes buffered in `buffer_full` mode before `on_buffer_exceeded` applies.
    #[serde(default = "default_acs_max_buffer_bytes")]
    #[schemars(range(min = 1))]
    pub max_buffer_bytes: u64,
    /// Buffer-overflow policy for streamed output when the buffer cap is hit.
    #[serde(default = "default_acs_on_buffer_exceeded")]
    pub on_buffer_exceeded: String,
    /// Fail-open policy for the output hook. When disabled, an Azure outage does not release unscanned model output.
    #[serde(default)]
    pub output_fail_open: bool,
}

fn default_acs_output_type() -> String {
    "FourSeverityLevels".to_owned()
}

fn default_acs_categories() -> Vec<String> {
    vec![
        "Hate".to_owned(),
        "Sexual".to_owned(),
        "SelfHarm".to_owned(),
        "Violence".to_owned(),
    ]
}

fn default_acs_severity_threshold() -> u8 {
    2
}

fn default_acs_text_source() -> String {
    "concatenate_user_content".to_owned()
}

fn default_acs_stream_processing_mode() -> String {
    "window".to_owned()
}

fn default_acs_window_size() -> u32 {
    10_000
}

fn default_acs_window_overlap_size() -> u32 {
    256
}

fn default_acs_max_buffer_bytes() -> u64 {
    262_144
}

fn default_acs_on_buffer_exceeded() -> String {
    "fail_closed".to_owned()
}

/// Config block for `kind: "aliyun_text_moderation"`. Calls Aliyun's
/// content-safety guardrail (`TextModerationPlus`, action version
/// `2022-03-02`) on the `green-cip.<region>.aliyuncs.com` endpoint for
/// category-risk moderation on input and/or output (including streaming
/// output.
///
/// The input hook uses the `llm_query_moderation` service code, the
/// output hook `llm_response_moderation`. Aliyun grades each call with a
/// `RiskLevel` (`none`/`low`/`medium`/`high`). The DP blocks when the
/// returned level reaches `risk_level_threshold`.
///
/// Only `access_key_secret` is a secret (decrypted by cp-api before kine
/// projection. It is plaintext in DP memory only and never logged). Every other
/// field travels in the clear.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AliyunTextModerationConfig {
    /// Aliyun region the guardrail lives in, e.g. `cn-shanghai`. The data plane
    /// builds the endpoint `https://green-cip.<region>.aliyuncs.com`.
    #[schemars(length(min = 1))]
    pub region: String,
    /// Explicit endpoint override as a full URL with no trailing slash. When set, it takes precedence over `region`.
    #[serde(default)]
    #[schemars(length(min = 1))]
    pub endpoint: Option<String>,
    /// Aliyun AccessKey ID.
    #[schemars(length(min = 1))]
    pub access_key_id: String,
    /// Aliyun AccessKey secret. Decrypted before projection. Plaintext is held
    /// in memory only and is not logged. Used to sign the request.
    #[schemars(length(min = 1))]
    pub access_key_secret: String,
    /// Minimum risk level that triggers a block. A returned level at or above
    /// this threshold blocks.
    #[serde(default = "default_aliyun_risk_level_threshold")]
    pub risk_level_threshold: String,
    /// HTTP call timeout in milliseconds. `fail_open` and `output_fail_open`
    /// govern the verdict when it elapses. A value of `0` triggers the timeout
    /// immediately.
    #[serde(default = "default_acs_timeout_ms")]
    #[schemars(range(max = 4_294_967_295u32))]
    pub timeout_ms: u32,
    /// Fail-open policy for the output hook. When disabled, an Aliyun outage does not release unscanned model output.
    #[serde(default)]
    pub output_fail_open: bool,

    // --- streaming-output controls (consumed by aisix-proxy build_sse_stream) ---
    /// Streaming output moderation mode: sliding-window incremental release
    /// or whole-response hold-back.
    #[serde(default = "default_acs_stream_processing_mode")]
    pub stream_processing_mode: String,
    /// Sliding-window size in characters when window mode is used. Aliyun limits each `llm_response_moderation` call to 2,000 characters.
    #[serde(default = "default_aliyun_window_size")]
    #[schemars(range(min = 1, max = 2_000))]
    pub window_size: u32,
    /// Chars carried between windows so a span split across a boundary is still caught.
    #[serde(default = "default_aliyun_window_overlap_size")]
    pub window_overlap_size: u32,
    /// Max bytes buffered in `buffer_full` mode before `on_buffer_exceeded` applies.
    #[serde(default = "default_acs_max_buffer_bytes")]
    #[schemars(range(min = 1))]
    pub max_buffer_bytes: u64,
    /// Buffer-overflow policy for streamed output when the buffer cap is hit.
    #[serde(default = "default_acs_on_buffer_exceeded")]
    pub on_buffer_exceeded: String,
}

fn default_aliyun_risk_level_threshold() -> String {
    "high".to_owned()
}

/// Config block for `kind: "aliyun_ai_guardrail"`. Calls Aliyun's AI
/// Guardrails product (AI 安全护栏, action `MultiModalGuard`, version
/// `2022-03-02`) on the `green-cip.<region>.aliyuncs.com` endpoint for
/// policy-driven moderation on input and/or output (including streaming
/// output).
///
/// A DIFFERENT Aliyun product from `aliyun_text_moderation`
/// (TextModerationPlus / Content Moderation): AI Guardrails is activated,
/// billed, and policy-configured separately (commodity
/// `lvwang_guardrail_public_cn`), and its calls appear in the AI
/// Guardrails console. The input hook uses the `query_security_check_pro`
/// service code (`query_security_check` at `service_level: "basic"`), the
/// output hook `response_security_check_pro` / `response_security_check`.
///
/// Verdicts follow the returned `Data.Suggestion`, which Aliyun computes
/// from the check/block policies configured in its console — there is no
/// local risk threshold. `block` blocks; anything else passes (detection
/// detail lands in logs/telemetry).
///
/// Only `access_key_secret` is a secret (decrypted by cp-api before kine
/// projection. It is plaintext in DP memory only and never logged). Every
/// other field travels in the clear.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AliyunAiGuardrailConfig {
    /// Aliyun region the guardrail lives in, e.g. `cn-shanghai`. The data plane
    /// builds the endpoint `https://green-cip.<region>.aliyuncs.com`.
    #[schemars(length(min = 1))]
    pub region: String,
    /// Explicit endpoint override as a full URL with no trailing slash. When set, it takes precedence over `region`.
    #[serde(default)]
    #[schemars(length(min = 1))]
    pub endpoint: Option<String>,
    /// Aliyun AccessKey ID.
    #[schemars(length(min = 1))]
    pub access_key_id: String,
    /// Aliyun AccessKey secret. Decrypted before projection. Plaintext is held
    /// in memory only and is not logged. Used to sign the request.
    #[schemars(length(min = 1))]
    pub access_key_secret: String,
    /// Which AI Guardrails service tier to call: `pro` uses
    /// `query_security_check_pro` / `response_security_check_pro`, `basic`
    /// uses `query_security_check` / `response_security_check`. Must match
    /// the tier activated on the Aliyun account.
    #[serde(default = "default_aliyun_aig_service_level")]
    pub service_level: String,
    /// HTTP call timeout in milliseconds. `fail_open` and `output_fail_open`
    /// govern the verdict when it elapses. A value of `0` triggers the timeout
    /// immediately.
    #[serde(default = "default_acs_timeout_ms")]
    #[schemars(range(max = 4_294_967_295u32))]
    pub timeout_ms: u32,
    /// Fail-open policy for the output hook. When disabled, an Aliyun outage does not release unscanned model output.
    #[serde(default)]
    pub output_fail_open: bool,

    // --- streaming-output controls (consumed by aisix-proxy build_sse_stream) ---
    /// Streaming output moderation mode: sliding-window incremental release
    /// or whole-response hold-back.
    #[serde(default = "default_acs_stream_processing_mode")]
    pub stream_processing_mode: String,
    /// Sliding-window size in characters when window mode is used. Aliyun limits each MultiModalGuard call to 2,000 characters of text.
    #[serde(default = "default_aliyun_window_size")]
    #[schemars(range(min = 1, max = 2_000))]
    pub window_size: u32,
    /// Chars carried between windows so a span split across a boundary is still caught.
    #[serde(default = "default_aliyun_window_overlap_size")]
    pub window_overlap_size: u32,
    /// Max bytes buffered in `buffer_full` mode before `on_buffer_exceeded` applies.
    #[serde(default = "default_acs_max_buffer_bytes")]
    #[schemars(range(min = 1))]
    pub max_buffer_bytes: u64,
    /// Buffer-overflow policy for streamed output when the buffer cap is hit.
    #[serde(default = "default_acs_on_buffer_exceeded")]
    pub on_buffer_exceeded: String,
}

fn default_aliyun_aig_service_level() -> String {
    "pro".to_owned()
}

/// One built-in detector selection for `kind: "pii"`. The `type` names the
/// detector to enable; `action` optionally overrides the guardrail-level
/// `default_action` for this detector only.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PiiDetectorConfig {
    /// Built-in detector to enable for this PII guardrail entry.
    #[serde(rename = "type")]
    #[schemars(length(min = 1))]
    pub detector_type: String,
    /// Per-detector action override. Falls back to the guardrail's
    /// `default_action` when omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<String>,
}

/// One operator-supplied regex detector for `kind: "pii"`.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PiiCustomPattern {
    /// Detector name surfaced in the mask token (`[<NAME>_REDACTED]`),
    /// telemetry counts, and block reasons. Never the matched value.
    #[schemars(length(min = 1, max = 64))]
    pub name: String,
    /// Regular expression AISIX compiles when building the guardrail chain.
    /// An invalid pattern makes AISIX log and skip the guardrail.
    #[schemars(length(min = 1))]
    pub regex: String,
    /// Per-pattern action override. Falls back to the guardrail's
    /// `default_action` when omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<String>,
}

/// Config block for `kind: "pii"`. Built-in sensitive-data detection and
/// redaction. AISIX evaluates built-in detectors and custom regex patterns
/// inside the gateway. Each detector can mask the matched span and let traffic
/// continue, or block the request or response according to the row's
/// `hook_point`.
///
/// `mask` rewrites each matched span to `[<DETECTOR>_REDACTED]` and lets the
/// request/response continue; `block` rejects with the standard 422
/// content-filter envelope. Matched values never appear in logs, telemetry,
/// or error envelopes — only detector names and match counts do.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PiiConfig {
    /// Built-in detectors to enable. The resource schema rejects unknown
    /// detector ids, so a typo cannot silently disable the policy.
    #[serde(default)]
    pub detectors: Vec<PiiDetectorConfig>,
    /// Operator-supplied regex detectors, evaluated after the built-ins.
    #[serde(default)]
    pub custom_patterns: Vec<PiiCustomPattern>,
    /// Action for detectors that do not set their own override.
    #[serde(default = "default_pii_action")]
    pub default_action: String,

    // --- streaming-output controls (consumed by aisix-proxy) ---
    // Masking a streamed response requires the whole response to be held
    // back (the mask spans chunk boundaries), so kind=pii always uses the
    // buffer_full policy on the output hook. These knobs mirror the ACS/
    // Aliyun buffer_full parameters.
    /// Max bytes buffered for a streamed response before `on_buffer_exceeded` applies.
    #[serde(default = "default_acs_max_buffer_bytes")]
    #[schemars(range(min = 1))]
    pub max_buffer_bytes: u64,
    /// Buffer-overflow policy for streamed output when the buffer cap is hit.
    #[serde(default = "default_acs_on_buffer_exceeded")]
    pub on_buffer_exceeded: String,
}

fn default_pii_action() -> String {
    "mask".to_owned()
}

fn default_aliyun_window_size() -> u32 {
    2_000
}

fn default_aliyun_window_overlap_size() -> u32 {
    128
}

/// Config block for `kind: "bedrock"`. The DP builds an
/// `aisix-guardrails::BedrockGuardrail` from this and dispatches
/// `ApplyGuardrail` on every governed request.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BedrockConfig {
    /// Guardrail identifier issued by the AWS console.
    #[schemars(length(min = 1, max = 64))]
    pub guardrail_id: String,
    /// Version label: `DRAFT`, `1`, `2`, ...
    #[schemars(length(min = 1, max = 16))]
    pub guardrail_version: String,
    /// AWS region for the Bedrock endpoint, such as `us-east-1`.
    #[schemars(length(min = 1))]
    pub region: String,
    /// IAM credentials for Bedrock requests.
    pub aws_credentials: BedrockAWSCredentials,
    /// Bedrock guardrail latency policy. Timed mode caps wait time with `timeout_ms`.
    pub latency_mode: BedrockLatencyMode,
    /// Fail-open policy for the output hook. When disabled (the default), a
    /// Bedrock outage blocks model output instead of releasing unscanned content.
    /// The input hook continues to use the top-level `fail_open` policy.
    #[serde(default)]
    pub output_fail_open: bool,
}

/// Config block for `kind: "lakera"`. Calls Lakera Guard
/// (`POST {endpoint}/v2/guard`) with conversation text and translates the
/// screening result into a verdict. Prompt-injection, jailbreak, and content
/// detections block traffic. Detections that involve only PII mask the detected
/// spans using the offsets Lakera returns and let traffic continue.
///
/// The `api_key` is stored encrypted and decrypted only when the
/// configuration is applied; the plaintext is held in memory only and is
/// never logged.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LakeraConfig {
    /// Lakera API key sent as a `Authorization: Bearer` header. Decrypted
    /// before projection. Plaintext is held in memory only and is not logged.
    #[schemars(length(min = 1))]
    pub api_key: String,
    /// Endpoint override, e.g. a regional or self-hosted Lakera deployment.
    /// The data plane appends `/v2/guard`. Defaults to `https://api.lakera.ai`.
    #[serde(default)]
    #[schemars(length(min = 1))]
    pub endpoint: Option<String>,
    /// Lakera project whose policy applies (`project-...`). Omitted → the
    /// account's default policy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(length(min = 1))]
    pub project_id: Option<String>,
    /// HTTP call timeout in milliseconds. `fail_open` and `output_fail_open`
    /// govern the verdict when it elapses. A value of `0` triggers the timeout
    /// immediately.
    #[serde(default = "default_acs_timeout_ms")]
    #[schemars(range(max = 4_294_967_295u32))]
    pub timeout_ms: u32,
    /// Fail-open policy for the output hook. When disabled (the default), a
    /// Lakera outage blocks model output instead of releasing unscanned content.
    /// The input hook continues to use the top-level `fail_open` policy.
    #[serde(default)]
    pub output_fail_open: bool,

    // --- streaming-output controls (consumed by aisix-proxy) ---
    // Masking a streamed response requires the whole response held back
    // (a masked span can cross any chunk boundary), so kind=lakera always
    // uses the buffer_full policy on the output hook, like kind=pii.
    /// Max bytes buffered for a streamed response before `on_buffer_exceeded` applies.
    #[serde(default = "default_acs_max_buffer_bytes")]
    #[schemars(range(min = 1))]
    pub max_buffer_bytes: u64,
    /// Buffer-overflow policy for streamed output when the buffer cap is hit.
    #[serde(default = "default_acs_on_buffer_exceeded")]
    pub on_buffer_exceeded: String,
}

/// Config block for `kind: "openai_moderation"`. Calls the OpenAI Moderation
/// API (`POST {endpoint}/moderations`) and blocks when the provider flags
/// content or when configured category thresholds are reached. This guardrail
/// is detection-only and never rewrites content. Monitor-before-enforce comes
/// from the row's `enforcement_mode`.
///
/// The `api_key` is stored encrypted and decrypted only when the
/// configuration is applied; the plaintext is held in memory only and is
/// never logged.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenaiModerationConfig {
    /// OpenAI API key sent as a `Authorization: Bearer` header. Decrypted
    /// before projection. Plaintext is held in memory only and is not logged.
    #[schemars(length(min = 1))]
    pub api_key: String,
    /// Endpoint override, such as an Azure OpenAI deployment. AISIX appends
    /// `/moderations`. Defaults to `https://api.openai.com/v1`.
    #[serde(default)]
    #[schemars(length(min = 1))]
    pub endpoint: Option<String>,
    /// Moderation model sent to the provider. The default is
    /// `omni-moderation-latest`; provider model names are not restricted by
    /// AISIX.
    #[serde(default = "default_openai_moderation_model")]
    #[schemars(length(min = 1))]
    pub model: String,
    /// Per-category score thresholds, e.g. `{"violence": 0.5}`. When set,
    /// only the listed categories are enforced and a category blocks when
    /// its score reaches the threshold. When empty (the default), the
    /// provider's `flagged` decision determines whether to block.
    #[serde(default)]
    pub category_thresholds: std::collections::BTreeMap<String, f64>,
    /// HTTP call timeout in milliseconds. `fail_open` and `output_fail_open`
    /// govern the verdict when it elapses. A value of `0` triggers the timeout
    /// immediately.
    #[serde(default = "default_acs_timeout_ms")]
    #[schemars(range(max = 4_294_967_295u32))]
    pub timeout_ms: u32,
    /// Fail-open policy for the output hook. When disabled (the default), an
    /// OpenAI outage blocks model output instead of releasing unscanned content.
    /// The input hook continues to use the top-level `fail_open` policy.
    #[serde(default)]
    pub output_fail_open: bool,
}

fn default_openai_moderation_model() -> String {
    "omni-moderation-latest".to_owned()
}

/// One entity selection for `kind: "presidio"`. The `type` names a Presidio
/// entity (`EMAIL_ADDRESS`, `PHONE_NUMBER`, `PERSON`, `CREDIT_CARD`, …);
/// `action` optionally overrides the guardrail-level `default_action` for
/// this entity only — the same per-detector shape as `kind: "pii"`.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PresidioEntityConfig {
    /// Presidio entity type, e.g. `EMAIL_ADDRESS`, `PERSON`, `US_SSN`.
    #[serde(rename = "type")]
    #[schemars(length(min = 1))]
    pub entity_type: String,
    /// Per-entity action override. Falls back to the guardrail's
    /// `default_action` when omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<String>,
}

/// Config block for `kind: "presidio"`. Calls a self-hosted Microsoft
/// Presidio analyzer to find PII entities and, when the effective action is
/// `mask`, calls the anonymizer to rewrite text before traffic continues.
/// `block` rejects with the standard 422 content-filter envelope.
///
/// vs. the built-in `kind: "pii"`: Presidio adds NER/ML entities a regex
/// cannot express (`PERSON`, `LOCATION`, `NRP`, …), a self-hosted
/// compliance posture, and selectable anonymize operators. No vendor secret:
/// both URLs point at customer-run containers.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PresidioConfig {
    /// Presidio analyzer base URL, e.g. `http://presidio-analyzer:3000`.
    /// The data plane appends `/analyze`.
    #[schemars(length(min = 1))]
    pub analyzer_url: String,
    /// Presidio anonymizer base URL, e.g. `http://presidio-anonymizer:3000`.
    /// The data plane appends `/anonymize`. Only called when a detected
    /// entity's effective action is `mask`.
    #[schemars(length(min = 1))]
    pub anonymizer_url: String,
    /// Entities to detect. Empty (the default) analyzes with Presidio's
    /// full recognizer set and applies `default_action` to every hit.
    #[serde(default)]
    pub entities: Vec<PresidioEntityConfig>,
    /// Action for entities that do not set their own override.
    #[serde(default = "default_pii_action")]
    pub default_action: String,
    /// Anonymize operator applied to masked entities.
    #[serde(default = "default_presidio_operator")]
    pub operator: String,
    /// Analyzer language code.
    #[serde(default = "default_presidio_language")]
    #[schemars(length(min = 1))]
    pub language: String,
    /// Minimum analyzer confidence for a hit to count. Omitted → every
    /// result the analyzer returns counts (Presidio's own per-recognizer
    /// defaults apply).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 0.0, max = 1.0))]
    pub score_threshold: Option<f64>,
    /// HTTP call timeout in milliseconds, applied per analyzer/anonymizer
    /// call. `fail_open` and `output_fail_open` govern the verdict when it
    /// elapses. A value of `0` triggers the timeout immediately.
    #[serde(default = "default_acs_timeout_ms")]
    #[schemars(range(max = 4_294_967_295u32))]
    pub timeout_ms: u32,
    /// Fail-open policy for the output hook. When disabled (the default), a
    /// Presidio outage blocks model output instead of releasing unscanned content.
    /// The input hook continues to use the top-level `fail_open` policy.
    #[serde(default)]
    pub output_fail_open: bool,

    // --- streaming-output controls (consumed by aisix-proxy) ---
    // Masking a streamed response requires the whole response held back,
    // so kind=presidio always uses the buffer_full policy on the output
    // hook, like kind=pii.
    /// Max bytes buffered for a streamed response before `on_buffer_exceeded` applies.
    #[serde(default = "default_acs_max_buffer_bytes")]
    #[schemars(range(min = 1))]
    pub max_buffer_bytes: u64,
    /// Buffer-overflow policy for streamed output when the buffer cap is hit.
    #[serde(default = "default_acs_on_buffer_exceeded")]
    pub on_buffer_exceeded: String,
}

fn default_presidio_operator() -> String {
    "replace".to_owned()
}

fn default_presidio_language() -> String {
    "en".to_owned()
}

/// Provider discriminator. The kind drives which `*_config` block is
/// expected. Serde's `tag = "kind"` keeps us honest at parse time.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GuardrailKind {
    /// In-process literal/regex blocklist. Always available.
    Keyword(KeywordConfig),
    /// AWS Bedrock managed guardrail using `ApplyGuardrail` on input, output, or both.
    Bedrock(BedrockConfig),
    /// Azure AI Content Safety Prompt Shield. Detects jailbreak and
    /// indirect injection attacks via the `/contentsafety/text:shieldPrompt`
    /// API.
    AzureContentSafety(AzureContentSafetyConfig),
    /// Azure AI Content Safety Text Moderation. Category-severity and
    /// blocklist moderation via the `/contentsafety/text:analyze` API, on input
    /// and/or output, including streaming output.
    AzureContentSafetyTextModeration(AzureContentSafetyTextModerationConfig),
    /// Aliyun content-safety guardrail. Risk-level moderation via the
    /// `TextModerationPlus` action on `green-cip.<region>.aliyuncs.com`, on
    /// input and/or output, including streaming output.
    AliyunTextModeration(AliyunTextModerationConfig),
    /// Aliyun AI Guardrails (AI 安全护栏). Policy-driven moderation via the
    /// `MultiModalGuard` action on `green-cip.<region>.aliyuncs.com`; the
    /// verdict follows the console-configured policy's `Suggestion`. On
    /// input and/or output, including streaming output.
    AliyunAiGuardrail(AliyunAiGuardrailConfig),
    /// Built-in sensitive-data detection and redaction inside AISIX. Built-in
    /// detectors and custom regex patterns can mask matched spans or block
    /// traffic on input, output, or both, including buffered streaming output.
    /// Always available and does not call an external service.
    Pii(PiiConfig),
    /// Lakera Guard screening via `POST /v2/guard`. Prompt-injection,
    /// jailbreak, and content-policy detections block traffic. Detections
    /// that involve only PII mask spans using the returned offsets. Applies
    /// on input, output, or both, including buffered streaming output.
    Lakera(LakeraConfig),
    /// OpenAI Moderation API category screening via `POST /moderations`.
    /// Flagged content or configured category thresholds block traffic. This
    /// guardrail is detection-only and never rewrites content. Applies on
    /// input, output, or both, including buffered streaming output.
    OpenaiModeration(OpenaiModerationConfig),
    /// Self-hosted Microsoft Presidio PII detection and anonymization.
    /// Analyzer entities can mask or block per entity, and masked entities
    /// use the selected anonymize operator. Applies on input, output, or both,
    /// including buffered streaming output.
    Presidio(PresidioConfig),
}

impl GuardrailKind {
    /// The wire `kind` discriminator string — matches the cp-api kind enum
    /// and the dashboard. Used for applied-guardrail telemetry.
    pub fn kind_str(&self) -> &'static str {
        match self {
            GuardrailKind::Keyword(_) => "keyword",
            GuardrailKind::Bedrock(_) => "bedrock",
            GuardrailKind::AzureContentSafety(_) => "azure_content_safety",
            GuardrailKind::AzureContentSafetyTextModeration(_) => {
                "azure_content_safety_text_moderation"
            }
            GuardrailKind::AliyunTextModeration(_) => "aliyun_text_moderation",
            GuardrailKind::AliyunAiGuardrail(_) => "aliyun_ai_guardrail",
            GuardrailKind::Pii(_) => "pii",
            GuardrailKind::Lakera(_) => "lakera",
            GuardrailKind::OpenaiModeration(_) => "openai_moderation",
            GuardrailKind::Presidio(_) => "presidio",
        }
    }
}

impl GuardrailHookPoint {
    /// Lowercase wire string: "input" / "output" / "both".
    pub fn as_str(&self) -> &'static str {
        match self {
            GuardrailHookPoint::Input => "input",
            GuardrailHookPoint::Output => "output",
            GuardrailHookPoint::Both => "both",
        }
    }
}

/// One guardrail that applied to a request, captured at chain-build time:
/// the guardrail `kind` and the `hook` it's configured for. Carried on the
/// telemetry UsageEvent so the dashboard can show which guardrails governed a
/// request. Records the attached set (`kind` + `hook`), not per-guardrail
/// verdicts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppliedGuardrail {
    pub kind: String,
    pub hook: String,
}

/// One monitor-mode observation: what an `enforcement_mode: monitor`
/// guardrail WOULD have done to this request had it been enforcing
/// (AISIX-Cloud#562). Carried on the telemetry UsageEvent so operators can
/// stage a policy, watch its hit rate in the dashboard, and only then flip
/// it to `block`.
///
/// `reason` and `counts` carry detector/entity/category NAMES only — never
/// matched content (#153 no-leak criterion).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuardrailMonitorHit {
    /// The configured (row) name of the monitor-mode guardrail that fired.
    pub guardrail_name: String,
    /// Which side observed the hit: `input` or `output`.
    pub hook: String,
    /// `would_block` (a Block verdict was downgraded) or `would_mask`
    /// (maskable spans were observed but not rewritten).
    pub action: String,
    /// The suppressed Block's operator-facing reason (`would_block` only;
    /// empty for `would_mask`).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub reason: String,
    /// detector/entity name → span count the guardrail would have masked
    /// (`would_mask` only; empty for `would_block`).
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub counts: std::collections::BTreeMap<String, u32>,
}

/// One guardrail member execution as observed by the chain fold
/// (AISIX-Cloud#1076): identity, phase, enforced outcome, and wall-clock
/// duration. All fields are bounded values safe for metric labels — never
/// matched content (#153 no-leak criterion).
#[derive(Debug, Clone, Copy)]
pub struct GuardrailExecution<'a> {
    /// Configured (row) name of the guardrail.
    pub guardrail_name: &'a str,
    /// The `kind` discriminator (`GuardrailKind::kind_str`). `keyword` and
    /// `pii` run in-process; every other kind calls a remote service.
    pub kind: &'a str,
    /// Which side ran: `input` or `output`.
    pub phase: &'static str,
    /// Enforced outcome: `allowed` / `blocked` / `masked` / `bypassed`
    /// (remote failure + fail-open) / `would_block` / `would_mask`
    /// (monitor mode).
    pub result: &'static str,
    /// Bounded failure tag (e.g. `lakera_timeout`) when the guardrail could
    /// not evaluate and failed open (`result = bypassed`); `None` otherwise.
    pub error_type: Option<&'a str>,
    /// Wall-clock time the member call took.
    pub elapsed: std::time::Duration,
}

/// Receiver for per-execution guardrail telemetry. Implemented by the
/// metrics layer (aisix-obs) and injected into the resolved chain at
/// build time, so aisix-guardrails stays free of a metrics dependency.
pub trait GuardrailMetricsSink: Send + Sync + 'static {
    fn record_guardrail_execution(&self, exec: &GuardrailExecution<'_>);
}

/// Content policy evaluated before or after upstream calls.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq)]
pub struct Guardrail {
    /// Operator-facing name that surfaces in metric labels and error reasons.
    #[schemars(length(min = 1))]
    pub name: String,

    /// When false, the chain skips this rule entirely. Allows operators
    /// to stage a rule before enabling it.
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// Where in the lifecycle this rule runs.
    #[serde(default)]
    pub hook_point: GuardrailHookPoint,

    /// Behavior when a remote API guardrail cannot reach its upstream.
    /// `true` allows the request and records the bypass reason in
    /// `usage_events.guardrail_bypassed_reason`. `false` blocks with
    /// 422. Keyword guardrails do not use this setting.
    #[serde(default = "default_fail_open")]
    pub fail_open: bool,

    /// The provider discriminator + its config. Use serde's flattening
    /// so the wire shape is `{ kind: "keyword", patterns: [...] }`
    /// rather than `{ kind: "keyword", keyword: { patterns: [...] }}`.
    #[serde(flatten)]
    pub config: GuardrailKind,

    /// How AISIX handles matching content. Enforcing mode applies the
    /// guardrail verdict; monitor mode records what would have happened
    /// without blocking or redacting the caller-visible response.
    #[serde(default = "default_enforcement_mode")]
    pub enforcement_mode: String,

    /// Whether guardrail evaluation errors are fatal. When `true`, a remote
    /// guardrail that cannot reach its upstream blocks the request instead of
    /// failing open, overriding `fail_open` on the failure path. The default
    /// `false` keeps the `fail_open` behavior.
    #[serde(default)]
    pub mandatory: bool,

    /// Attachment direction hint. Guardrail execution still follows
    /// `hook_point`.
    #[serde(default = "default_direction")]
    pub direction: String,

    /// RFC3339 creation timestamp. When present, guardrails are evaluated from oldest to newest. Resources without this timestamp sort after resources that have it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,

    #[serde(skip)]
    pub(crate) runtime_id: String,
}

fn default_enabled() -> bool {
    true
}

fn default_fail_open() -> bool {
    true
}

fn default_enforcement_mode() -> String {
    "block".to_owned()
}

fn default_direction() -> String {
    "both".to_owned()
}

impl Resource for Guardrail {
    fn id(&self) -> &str {
        &self.runtime_id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn kind() -> &'static str {
        "guardrails"
    }
}

// ---------------------------------------------------------------------------
// GuardrailAttachment — P0c
// ---------------------------------------------------------------------------

/// Which dimension of the request a guardrail attachment is scoped to.
///
/// `Env` applies to every request in the environment. The narrower scopes let
/// operators attach a guardrail to only the models, API keys, or teams that
/// need it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum GuardrailScopeType {
    Env,
    Model,
    ApiKey,
    Team,
}

/// Guardrail attachment that scopes one guardrail to an environment, model,
/// caller API key, or team. AISIX loads attachments with the guardrail
/// definitions and uses `scope_type` plus `scope_id` to decide which
/// guardrails apply to each request.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
pub struct GuardrailAttachment {
    /// UUID of the guardrail definition this attachment points to.
    #[schemars(length(min = 1))]
    pub guardrail_id: String,

    /// What dimension of the request this attachment is scoped to.
    pub scope_type: GuardrailScopeType,

    /// The UUID of the specific resource (model / api_key / team).
    /// `None` when `scope_type` is `Env` (applies to all requests).
    pub scope_id: Option<String>,

    /// Higher number = higher precedence. When the same guardrail appears
    /// via multiple matching scopes, the highest-priority attachment wins
    /// and duplicates are dropped.
    pub priority: i32,

    /// When `false`, AISIX ignores this attachment.
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    #[serde(skip)]
    pub(crate) runtime_id: String,
}

impl Resource for GuardrailAttachment {
    fn id(&self) -> &str {
        &self.runtime_id
    }

    /// Keyed by `guardrail_id` in the `ResourceTable` name-index so
    /// callers can look up attachments by guardrail.
    ///
    /// WARNING: the name-index is a flat map and silently overwrites
    /// earlier entries when a guardrail has multiple attachments (e.g.
    /// one Env-scope and one Model-scope attachment share the same key).
    /// Use `ResourceTable::entries()` (not `get_by_name`) to enumerate
    /// all attachments. `build_index_from_snapshot` already does this.
    fn name(&self) -> &str {
        &self.guardrail_id
    }

    fn kind() -> &'static str {
        "guardrail_attachments"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn deserialises_keyword_with_mixed_patterns() {
        let v = json!({
            "name": "block-secrets",
            "enabled": true,
            "hook_point": "input",
            "kind": "keyword",
            "patterns": [
                { "kind": "literal", "value": "AKIA" },
                { "kind": "regex",   "value": "\\bssn:\\s*\\d{3}-\\d{2}-\\d{4}" }
            ]
        });
        let g: Guardrail = serde_json::from_value(v).unwrap();
        assert_eq!(g.name, "block-secrets");
        assert!(g.enabled);
        assert_eq!(g.hook_point, GuardrailHookPoint::Input);
        match g.config {
            GuardrailKind::Keyword(KeywordConfig { patterns }) => {
                assert_eq!(patterns.len(), 2);
                assert_eq!(patterns[0], KeywordPattern::Literal("AKIA".into()));
                assert_eq!(
                    patterns[1],
                    KeywordPattern::Regex(r"\bssn:\s*\d{3}-\d{2}-\d{4}".into())
                );
            }
            _ => panic!("expected Keyword variant"),
        }
    }

    #[test]
    fn enabled_defaults_to_true_when_omitted() {
        let v = json!({
            "name": "g",
            "kind": "keyword",
            "patterns": []
        });
        let g: Guardrail = serde_json::from_value(v).unwrap();
        assert!(g.enabled);
        assert_eq!(g.hook_point, GuardrailHookPoint::Both);
        assert!(g.fail_open);
    }

    #[test]
    fn fail_open_round_trips() {
        let v = json!({
            "name": "strict-bedrock",
            "kind": "keyword",
            "patterns": [],
            "fail_open": false
        });
        let g: Guardrail = serde_json::from_value(v).unwrap();
        assert!(!g.fail_open);
    }

    #[test]
    fn unknown_field_rejected_by_inner_kind_struct() {
        // The outer Guardrail can't use deny_unknown_fields (see its
        // doc comment), but the inner KeywordConfig does — and serde
        // surfaces unknown fields from the flattened inner type at
        // the top level. Net effect: typos are still caught.
        let v = json!({
            "name": "g",
            "kind": "keyword",
            "patterns": [],
            "extra": "nope"
        });
        let r: Result<Guardrail, _> = serde_json::from_value(v);
        assert!(r.is_err());
    }

    #[test]
    fn p0c_fields_dont_trip_keyword_config_deny_unknown_fields() {
        // `KeywordConfig` has `deny_unknown_fields`. The P0c fields
        // (`enforcement_mode`, `mandatory`, `direction`) are declared on
        // the outer `Guardrail` struct with `#[serde(default)]`, so serde
        // absorbs them at the outer level before the flattened inner sees
        // the remaining fields. This test pins that routing: if any of
        // these fields accidentally reached `KeywordConfig`, the parse
        // would return an unknown-field error.
        let v = json!({
            "name": "g",
            "kind": "keyword",
            "patterns": [],
            "enforcement_mode": "monitor",
            "mandatory": true,
            "direction": "input"
        });
        let g: Guardrail = serde_json::from_value(v)
            .expect("P0c fields must not trip KeywordConfig deny_unknown_fields");
        assert_eq!(g.enforcement_mode, "monitor");
        assert!(g.mandatory);
        assert_eq!(g.direction, "input");
    }

    #[test]
    fn bedrock_kind_parses_with_serial_latency() {
        let v = json!({
            "name": "block-pii",
            "kind": "bedrock",
            "guardrail_id": "abcdefgh1234",
            "guardrail_version": "DRAFT",
            "region": "us-east-1",
            "aws_credentials": {
                "kind": "static",
                "access_key_id": "AKIAEXAMPLE",
                "secret_access_key": "PLAINTEXT_FOR_TEST"
            },
            "latency_mode": { "kind": "serial" }
        });
        let g: Guardrail = serde_json::from_value(v).unwrap();
        match g.config {
            GuardrailKind::Bedrock(b) => {
                assert_eq!(b.guardrail_id, "abcdefgh1234");
                assert_eq!(b.region, "us-east-1");
                assert!(matches!(b.latency_mode, BedrockLatencyMode::Serial));
                match b.aws_credentials {
                    BedrockAWSCredentials::Static {
                        access_key_id,
                        secret_access_key,
                    } => {
                        assert_eq!(access_key_id, "AKIAEXAMPLE");
                        assert_eq!(secret_access_key, "PLAINTEXT_FOR_TEST");
                    }
                }
            }
            _ => panic!("expected Bedrock variant"),
        }
    }

    #[test]
    fn bedrock_kind_parses_with_timed_latency() {
        let v = json!({
            "name": "block-pii",
            "kind": "bedrock",
            "guardrail_id": "id",
            "guardrail_version": "1",
            "region": "us-east-1",
            "aws_credentials": {
                "kind": "static",
                "access_key_id": "AKIA",
                "secret_access_key": "secret"
            },
            "latency_mode": { "kind": "timed", "timeout_ms": 500 }
        });
        let g: Guardrail = serde_json::from_value(v).unwrap();
        match g.config {
            GuardrailKind::Bedrock(b) => match b.latency_mode {
                BedrockLatencyMode::Timed { timeout_ms } => assert_eq!(timeout_ms, 500),
                _ => panic!("expected Timed"),
            },
            _ => panic!("expected Bedrock variant"),
        }
    }

    #[test]
    fn azure_content_safety_kind_parses() {
        let v = json!({
            "name": "shield",
            "kind": "azure_content_safety",
            "endpoint": "https://my-resource.cognitiveservices.azure.com",
            "api_key": "plaintext-key",
            "timeout_ms": 3000
        });
        let g: Guardrail = serde_json::from_value(v).unwrap();
        assert_eq!(g.name, "shield");
        match g.config {
            GuardrailKind::AzureContentSafety(ref c) => {
                assert_eq!(
                    c.endpoint,
                    "https://my-resource.cognitiveservices.azure.com"
                );
                assert_eq!(c.api_key, "plaintext-key");
                assert_eq!(c.timeout_ms, 3000);
            }
            _ => panic!("expected AzureContentSafety variant"),
        }
    }

    #[test]
    fn azure_content_safety_timeout_defaults_to_5000() {
        let v = json!({
            "name": "shield",
            "kind": "azure_content_safety",
            "endpoint": "https://my-resource.cognitiveservices.azure.com",
            "api_key": "k"
        });
        let g: Guardrail = serde_json::from_value(v).unwrap();
        match g.config {
            GuardrailKind::AzureContentSafety(ref c) => assert_eq!(c.timeout_ms, 5_000),
            _ => panic!("expected AzureContentSafety variant"),
        }
    }

    #[test]
    fn azure_text_moderation_kind_parses_with_defaults() {
        // cp-api omits unset fields (omitempty); the DP must apply the
        // documented defaults so a minimal row still moderates correctly.
        let v = json!({
            "name": "moderate",
            "kind": "azure_content_safety_text_moderation",
            "endpoint": "https://my-resource.cognitiveservices.azure.com",
            "api_key": "plaintext-key"
        });
        let g: Guardrail = serde_json::from_value(v).unwrap();
        match g.config {
            GuardrailKind::AzureContentSafetyTextModeration(ref c) => {
                assert_eq!(c.timeout_ms, 5_000);
                assert_eq!(c.output_type, "FourSeverityLevels");
                assert_eq!(c.severity_threshold, 2);
                assert_eq!(c.categories.len(), 4);
                assert_eq!(c.text_source, "concatenate_user_content");
                assert_eq!(c.stream_processing_mode, "window");
                assert_eq!(c.window_size, 10_000);
                assert_eq!(c.window_overlap_size, 256);
                assert_eq!(c.max_buffer_bytes, 262_144);
                assert_eq!(c.on_buffer_exceeded, "fail_closed");
                assert!(!c.output_fail_open);
            }
            _ => panic!("expected AzureContentSafetyTextModeration variant"),
        }
    }

    #[test]
    fn azure_text_moderation_kind_round_trips_set_fields() {
        let v = json!({
            "name": "moderate",
            "kind": "azure_content_safety_text_moderation",
            "endpoint": "https://e.cognitiveservices.azure.com",
            "api_key": "k",
            "output_type": "EightSeverityLevels",
            "categories": ["Hate", "Violence"],
            "severity_threshold": 0,
            "severity_threshold_by_category": { "Violence": 6 },
            "stream_processing_mode": "buffer_full",
            "window_overlap_size": 0,
            "output_fail_open": true
        });
        let g: Guardrail = serde_json::from_value(v).unwrap();
        match g.config {
            GuardrailKind::AzureContentSafetyTextModeration(ref c) => {
                assert_eq!(c.output_type, "EightSeverityLevels");
                assert_eq!(
                    c.severity_threshold, 0,
                    "explicit 0 must survive (not defaulted to 2)"
                );
                assert_eq!(c.severity_threshold_by_category.get("Violence"), Some(&6));
                assert_eq!(c.stream_processing_mode, "buffer_full");
                assert_eq!(c.window_overlap_size, 0, "explicit 0 overlap must survive");
                assert!(c.output_fail_open);
            }
            _ => panic!("expected AzureContentSafetyTextModeration variant"),
        }
    }

    #[test]
    fn aliyun_text_moderation_kind_parses_with_defaults() {
        // cp-api omits unset fields (omitempty); the DP must apply the
        // documented defaults so a minimal row still moderates correctly.
        let v = json!({
            "name": "aliyun-guard",
            "kind": "aliyun_text_moderation",
            "region": "cn-shanghai",
            "access_key_id": "LTAI_EXAMPLE",
            "access_key_secret": "PLAINTEXT_FOR_TEST"
        });
        let g: Guardrail = serde_json::from_value(v).unwrap();
        match g.config {
            GuardrailKind::AliyunTextModeration(ref c) => {
                assert_eq!(c.region, "cn-shanghai");
                assert_eq!(c.access_key_id, "LTAI_EXAMPLE");
                assert_eq!(c.access_key_secret, "PLAINTEXT_FOR_TEST");
                assert_eq!(c.endpoint, None);
                assert_eq!(c.risk_level_threshold, "high");
                assert_eq!(c.timeout_ms, 5_000);
                assert!(!c.output_fail_open);
                assert_eq!(c.stream_processing_mode, "window");
                assert_eq!(c.window_size, 2_000);
                assert_eq!(c.window_overlap_size, 128);
                assert_eq!(c.max_buffer_bytes, 262_144);
                assert_eq!(c.on_buffer_exceeded, "fail_closed");
            }
            _ => panic!("expected AliyunTextModeration variant"),
        }
    }

    #[test]
    fn aliyun_text_moderation_kind_round_trips_set_fields() {
        let v = json!({
            "name": "aliyun-guard",
            "kind": "aliyun_text_moderation",
            "region": "cn-beijing",
            "endpoint": "http://127.0.0.1:8080",
            "access_key_id": "id",
            "access_key_secret": "secret",
            "risk_level_threshold": "medium",
            "timeout_ms": 3000,
            "output_fail_open": true,
            "stream_processing_mode": "buffer_full",
            "window_size": 1000,
            "window_overlap_size": 0
        });
        let g: Guardrail = serde_json::from_value(v).unwrap();
        match g.config {
            GuardrailKind::AliyunTextModeration(ref c) => {
                assert_eq!(c.endpoint.as_deref(), Some("http://127.0.0.1:8080"));
                assert_eq!(c.risk_level_threshold, "medium");
                assert_eq!(c.timeout_ms, 3000);
                assert!(c.output_fail_open);
                assert_eq!(c.stream_processing_mode, "buffer_full");
                assert_eq!(c.window_size, 1000);
                assert_eq!(c.window_overlap_size, 0, "explicit 0 overlap must survive");
            }
            _ => panic!("expected AliyunTextModeration variant"),
        }
    }

    #[test]
    fn aliyun_ai_guardrail_kind_parses_with_defaults() {
        // cp-api omits unset fields (omitempty); the DP must apply the
        // documented defaults so a minimal row still moderates correctly.
        let v = json!({
            "name": "aig-guard",
            "kind": "aliyun_ai_guardrail",
            "region": "cn-shanghai",
            "access_key_id": "LTAI_EXAMPLE",
            "access_key_secret": "PLAINTEXT_FOR_TEST"
        });
        let g: Guardrail = serde_json::from_value(v).unwrap();
        match g.config {
            GuardrailKind::AliyunAiGuardrail(ref c) => {
                assert_eq!(c.region, "cn-shanghai");
                assert_eq!(c.access_key_id, "LTAI_EXAMPLE");
                assert_eq!(c.access_key_secret, "PLAINTEXT_FOR_TEST");
                assert_eq!(c.endpoint, None);
                assert_eq!(c.service_level, "pro");
                assert_eq!(c.timeout_ms, 5_000);
                assert!(!c.output_fail_open);
                assert_eq!(c.stream_processing_mode, "window");
                assert_eq!(c.window_size, 2_000);
                assert_eq!(c.window_overlap_size, 128);
                assert_eq!(c.max_buffer_bytes, 262_144);
                assert_eq!(c.on_buffer_exceeded, "fail_closed");
            }
            _ => panic!("expected AliyunAiGuardrail variant"),
        }
    }

    #[test]
    fn aliyun_ai_guardrail_kind_round_trips_set_fields() {
        let v = json!({
            "name": "aig-guard",
            "kind": "aliyun_ai_guardrail",
            "region": "cn-beijing",
            "endpoint": "http://127.0.0.1:8080",
            "access_key_id": "id",
            "access_key_secret": "secret",
            "service_level": "basic",
            "timeout_ms": 3000,
            "output_fail_open": true,
            "stream_processing_mode": "buffer_full",
            "window_size": 1000,
            "window_overlap_size": 0
        });
        let g: Guardrail = serde_json::from_value(v).unwrap();
        match g.config {
            GuardrailKind::AliyunAiGuardrail(ref c) => {
                assert_eq!(c.endpoint.as_deref(), Some("http://127.0.0.1:8080"));
                assert_eq!(c.service_level, "basic");
                assert_eq!(c.timeout_ms, 3000);
                assert!(c.output_fail_open);
                assert_eq!(c.stream_processing_mode, "buffer_full");
                assert_eq!(c.window_size, 1000);
                assert_eq!(c.window_overlap_size, 0, "explicit 0 overlap must survive");
            }
            _ => panic!("expected AliyunAiGuardrail variant"),
        }
    }

    #[test]
    fn pii_kind_parses_with_defaults() {
        // cp-api omits unset fields (omitempty); the DP must apply the
        // documented defaults so a minimal row still works.
        let v = json!({
            "name": "mask-pii",
            "kind": "pii",
            "detectors": [
                { "type": "email" },
                { "type": "china_id_card", "action": "block" }
            ]
        });
        let g: Guardrail = serde_json::from_value(v).unwrap();
        match g.config {
            GuardrailKind::Pii(ref c) => {
                assert_eq!(c.detectors.len(), 2);
                assert_eq!(c.detectors[0].detector_type, "email");
                assert_eq!(c.detectors[0].action, None);
                assert_eq!(c.detectors[1].action.as_deref(), Some("block"));
                assert!(c.custom_patterns.is_empty());
                assert_eq!(c.default_action, "mask");
                assert_eq!(c.max_buffer_bytes, 262_144);
                assert_eq!(c.on_buffer_exceeded, "fail_closed");
            }
            _ => panic!("expected Pii variant"),
        }
    }

    #[test]
    fn pii_kind_round_trips_custom_patterns_and_overrides() {
        let v = json!({
            "name": "mask-pii",
            "kind": "pii",
            "default_action": "block",
            "custom_patterns": [
                { "name": "employee_id", "regex": "\\bEMP-\\d{6}\\b", "action": "mask" }
            ],
            "max_buffer_bytes": 1024,
            "on_buffer_exceeded": "fail_open"
        });
        let g: Guardrail = serde_json::from_value(v).unwrap();
        match g.config {
            GuardrailKind::Pii(ref c) => {
                assert!(c.detectors.is_empty());
                assert_eq!(c.custom_patterns.len(), 1);
                assert_eq!(c.custom_patterns[0].name, "employee_id");
                assert_eq!(c.custom_patterns[0].action.as_deref(), Some("mask"));
                assert_eq!(c.default_action, "block");
                assert_eq!(c.max_buffer_bytes, 1024);
                assert_eq!(c.on_buffer_exceeded, "fail_open");
            }
            _ => panic!("expected Pii variant"),
        }
    }

    #[test]
    fn resource_trait_uses_name_and_guardrails_kind() {
        let mut g: Guardrail = serde_json::from_value(json!({
            "name": "g1",
            "kind": "keyword",
            "patterns": []
        }))
        .unwrap();
        g.runtime_id = "uuid-1".into();
        assert_eq!(<Guardrail as Resource>::kind(), "guardrails");
        assert_eq!(g.id(), "uuid-1");
        assert_eq!(g.name(), "g1");
    }
}
