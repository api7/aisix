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
//!     (`llm_response_moderation`). #603.
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
    /// Run on the request payload before bridge dispatch.
    Input,
    /// Run on the upstream response before the cache write + render.
    Output,
    /// Run on both. Default for keyword blocklists.
    #[default]
    Both,
}

/// One pattern in a `keyword`-kind guardrail's blocklist. The DP
/// translates `Literal` to a case-insensitive substring match and
/// `Regex` to a compiled `regex::Regex`. Invalid regex at parse
/// time is loader-rejected (the DP refuses to apply a guardrail it
/// can't compile, so a typo doesn't silently disarm the policy).
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(tag = "kind", content = "value", rename_all = "lowercase")]
pub enum KeywordPattern {
    Literal(String),
    Regex(String),
}

/// Config block for `kind: "keyword"`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct KeywordConfig {
    /// Blocklist patterns. Empty list is legal but pointless — the
    /// guardrail will allow every request, same as `enabled: false`.
    pub patterns: Vec<KeywordPattern>,
}

/// AWS credentials for `kind: "bedrock"`. Phase 2 supports
/// `static` (access-key pair); Phase 4 adds `role_arn`
/// (sts:AssumeRole) under the same tag.
///
/// Wire shape on the kine path is plaintext: cp-api decrypts the
/// envelope-encrypted secret at projection time (same trust
/// boundary as `provider_keys` — see PRD-09c §6.3). The DP only
/// ever holds plaintext in memory; it does not need a master key.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum BedrockAWSCredentials {
    Static {
        access_key_id: String,
        /// Decrypted by cp-api before kine projection; plaintext
        /// in memory only, never logged. The DP feeds it to the
        /// AWS SDK's static credentials provider.
        secret_access_key: String,
    },
}

/// Per-guardrail latency policy for `kind: "bedrock"`. `serial`
/// waits unconditionally; `timed` aborts at `timeout_ms` and
/// applies the row-level `fail_open` flag. Range matches cp-api's
/// validator (100..5000ms) — see PRD-09c §6.6.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum BedrockLatencyMode {
    Serial,
    Timed { timeout_ms: u32 },
}

/// Config block for `kind: "azure_content_safety"`. Calls Azure AI
/// Content Safety Prompt Shield API to detect jailbreak and indirect
/// injection attacks. PRD-09c §6 P1.
///
/// The CP (cp-api) decrypts the envelope-encrypted `api_key` at kine-
/// projection time so the DP always holds plaintext in memory; the
/// key is never logged.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AzureContentSafetyConfig {
    /// Azure Cognitive Services resource endpoint, e.g.
    /// `https://my-resource.cognitiveservices.azure.com`.
    /// The DP appends `/contentsafety/text:shieldPrompt?api-version=2024-09-01`.
    pub endpoint: String,
    /// Subscription key (`Ocp-Apim-Subscription-Key`). Decrypted by
    /// cp-api before kine projection; plaintext in memory only, never
    /// logged.
    pub api_key: String,
    /// HTTP call timeout in milliseconds. When elapsed the `fail_open`
    /// flag governs the verdict. Defaults to 5 000 ms.
    ///
    /// **`0` does not mean "no timeout".** `Duration::ZERO` causes
    /// `tokio::time::timeout` to fire on the first poll, so every call
    /// immediately maps to a timeout failure. Use `u32::MAX` (≈ 49 days)
    /// for an effectively unlimited timeout.
    #[serde(default = "default_acs_timeout_ms")]
    pub timeout_ms: u32,
}

fn default_acs_timeout_ms() -> u32 {
    5_000
}

/// Config block for `kind: "azure_content_safety_text_moderation"`. Calls
/// Azure AI Content Safety `text:analyze` for category-severity + blocklist
/// moderation on input and/or output (including streaming output). P2
/// (PRD-09c §6 P2, #379).
///
/// Reuses the P1 connection block (endpoint + api_key + timeout_ms). cp-api
/// projects only operator-set fields (omitempty), so every optional field
/// carries a serde default matching the cp-api validator's documented
/// default. Only `api_key` is a secret (decrypted by cp-api before kine
/// projection); every other field travels in the clear.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AzureContentSafetyTextModerationConfig {
    /// Azure Cognitive Services resource endpoint. The DP appends
    /// `/contentsafety/text:analyze?api-version=2024-09-01`.
    pub endpoint: String,
    /// Subscription key (`Ocp-Apim-Subscription-Key`). Plaintext in memory
    /// only, never logged.
    pub api_key: String,
    /// HTTP call timeout (ms); `fail_open` / `output_fail_open` govern the
    /// verdict when it elapses. See the `AzureContentSafetyConfig` note on
    /// why `0` means "fire immediately", not "no timeout".
    #[serde(default = "default_acs_timeout_ms")]
    pub timeout_ms: u32,

    // --- moderation parameters ---
    /// `FourSeverityLevels` (0,2,4,6; default) or `EightSeverityLevels` (0..7).
    #[serde(default = "default_acs_output_type")]
    pub output_type: String,
    /// Categories to analyze. Defaults to all four.
    #[serde(default = "default_acs_categories")]
    pub categories: Vec<String>,
    /// General severity threshold; a category at or above it blocks.
    #[serde(default = "default_acs_severity_threshold")]
    pub severity_threshold: u8,
    /// Per-category threshold overrides (take precedence over the general one).
    #[serde(default)]
    pub severity_threshold_by_category: std::collections::BTreeMap<String, u8>,
    /// Azure CS blocklist names to match against.
    #[serde(default)]
    pub blocklist_names: Vec<String>,
    /// Forwarded to Azure's `haltOnBlocklistHit`.
    #[serde(default)]
    pub halt_on_blocklist_hit: bool,
    /// Input-hook text selection: `concatenate_user_content` (default) or
    /// `concatenate_all_content`. Ignored on the output hook.
    #[serde(default = "default_acs_text_source")]
    pub text_source: String,

    // --- streaming-output controls (consumed by aisix-proxy build_sse_stream) ---
    /// `window` (sliding-window incremental release; default) or
    /// `buffer_full` (whole-response hold-back).
    #[serde(default = "default_acs_stream_processing_mode")]
    pub stream_processing_mode: String,
    /// Sliding-window size in chars (window mode); cp-api caps it at the
    /// 10 000-char Azure limit. Default 10 000.
    #[serde(default = "default_acs_window_size")]
    pub window_size: u32,
    /// Chars carried between windows so a span split across a boundary is
    /// still caught. Default 256.
    #[serde(default = "default_acs_window_overlap_size")]
    pub window_overlap_size: u32,
    /// Max bytes buffered in `buffer_full` mode before `on_buffer_exceeded`
    /// applies. Default 262 144.
    #[serde(default = "default_acs_max_buffer_bytes")]
    pub max_buffer_bytes: u64,
    /// `fail_closed` (default) or `fail_open` when the buffer cap is hit.
    #[serde(default = "default_acs_on_buffer_exceeded")]
    pub on_buffer_exceeded: String,
    /// Fail-open policy for the OUTPUT hook. Defaults `false` (fail-closed)
    /// so an Azure outage can't release unscanned model output.
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
/// output). Issue #603.
///
/// The input hook uses the `llm_query_moderation` service code, the
/// output hook `llm_response_moderation`. Aliyun grades each call with a
/// `RiskLevel` (`none`/`low`/`medium`/`high`); the DP blocks when the
/// returned level reaches `risk_level_threshold`.
///
/// Only `access_key_secret` is a secret (decrypted by cp-api before kine
/// projection — plaintext in DP memory only, never logged); every other
/// field travels in the clear.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AliyunTextModerationConfig {
    /// Aliyun region the guardrail lives in, e.g. `cn-shanghai`. The DP
    /// builds the endpoint `https://green-cip.<region>.aliyuncs.com`.
    pub region: String,
    /// Explicit endpoint override (full URL, no trailing slash). When set
    /// it wins over `region`; used by tests/dev to point at a mock server.
    #[serde(default)]
    pub endpoint: Option<String>,
    /// Aliyun AccessKey ID.
    pub access_key_id: String,
    /// Aliyun AccessKey secret. Decrypted by cp-api before kine projection;
    /// plaintext in memory only, never logged. Used to sign the request.
    pub access_key_secret: String,
    /// Minimum risk level that triggers a block: `low`, `medium`, or
    /// `high` (default). A returned level at or above this blocks.
    #[serde(default = "default_aliyun_risk_level_threshold")]
    pub risk_level_threshold: String,
    /// HTTP call timeout (ms); `fail_open` / `output_fail_open` govern the
    /// verdict when it elapses. See the `AzureContentSafetyConfig` note on
    /// why `0` means "fire immediately", not "no timeout".
    #[serde(default = "default_acs_timeout_ms")]
    pub timeout_ms: u32,
    /// Fail-open policy for the OUTPUT hook. Defaults `false` (fail-closed)
    /// so an Aliyun outage can't release unscanned model output.
    #[serde(default)]
    pub output_fail_open: bool,

    // --- streaming-output controls (consumed by aisix-proxy build_sse_stream) ---
    /// `window` (sliding-window incremental release; default) or
    /// `buffer_full` (whole-response hold-back).
    #[serde(default = "default_acs_stream_processing_mode")]
    pub stream_processing_mode: String,
    /// Sliding-window size in chars (window mode). Default 2 000 — Aliyun's
    /// `llm_response_moderation` per-call content cap.
    #[serde(default = "default_aliyun_window_size")]
    pub window_size: u32,
    /// Chars carried between windows so a span split across a boundary is
    /// still caught. Default 128.
    #[serde(default = "default_aliyun_window_overlap_size")]
    pub window_overlap_size: u32,
    /// Max bytes buffered in `buffer_full` mode before `on_buffer_exceeded`
    /// applies. Default 262 144.
    #[serde(default = "default_acs_max_buffer_bytes")]
    pub max_buffer_bytes: u64,
    /// `fail_closed` (default) or `fail_open` when the buffer cap is hit.
    #[serde(default = "default_acs_on_buffer_exceeded")]
    pub on_buffer_exceeded: String,
}

fn default_aliyun_risk_level_threshold() -> String {
    "high".to_owned()
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
    /// AWS-console-issued guardrail identifier (12 chars today).
    pub guardrail_id: String,
    /// Version label: `DRAFT`, `1`, `2`, ...
    pub guardrail_version: String,
    /// AWS region the Bedrock endpoint lives in (e.g. `us-east-1`).
    pub region: String,
    /// IAM credentials. v1 = static access keys (encrypted).
    pub aws_credentials: BedrockAWSCredentials,
    /// `serial` (default) or `timed { timeout_ms }`.
    pub latency_mode: BedrockLatencyMode,
}

/// Provider discriminator. The kind drives which `*_config` block is
/// expected; serde's `tag = "kind"` keeps us honest at parse time.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GuardrailKind {
    /// In-process literal/regex blocklist. Always available.
    Keyword(KeywordConfig),
    /// AWS Bedrock managed guardrail. The chain builder constructs a
    /// `BedrockGuardrail` that calls AWS `ApplyGuardrail` (real SigV4
    /// dispatch) on input and/or output per the row's hook point.
    Bedrock(BedrockConfig),
    /// Azure AI Content Safety Prompt Shield. Detects jailbreak and
    /// indirect injection attacks via the `/contentsafety/text:shieldPrompt`
    /// API. P1 (PRD-09c §6 P1).
    AzureContentSafety(AzureContentSafetyConfig),
    /// Azure AI Content Safety Text Moderation. Category-severity +
    /// blocklist moderation via the `/contentsafety/text:analyze` API,
    /// on input and/or output (including streaming output). P2
    /// (PRD-09c §6 P2, #379).
    AzureContentSafetyTextModeration(AzureContentSafetyTextModerationConfig),
    /// Aliyun content-safety guardrail. Risk-level moderation via the
    /// `TextModerationPlus` action on `green-cip.<region>.aliyuncs.com`,
    /// on input and/or output (including streaming output). #603.
    AliyunTextModeration(AliyunTextModerationConfig),
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
/// request (#379 observability). v1 records the attached set (kind + hook),
/// not per-guardrail verdicts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppliedGuardrail {
    pub kind: String,
    pub hook: String,
}

/// Top-level `Guardrail` resource shape. Mirrors what cp-api writes
/// to kine at `/aisix/<env>/guardrails/<uuid>`.
///
/// `deny_unknown_fields` is intentionally NOT set here: serde's
/// `flatten` + `tag = "kind"` interaction can't pass the
/// "I consumed this field" signal up to the outer struct, so a
/// `deny_unknown_fields` outer would reject the very `kind` the
/// inner enum needs. Strict typo-rejection happens earlier in the
/// JSON Schema (`schema::validate_guardrail`) which the loader
/// runs before deserialise on every watch event.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
pub struct Guardrail {
    /// Operator-facing name; surfaces in metric labels + error reasons.
    pub name: String,

    /// When false the chain skips this rule entirely. Lets operators
    /// stage a rule (write it, sanity-check it via dry runs, then flip
    /// it on) without deleting + recreating.
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// Where in the lifecycle this rule runs. Defaults to `both`.
    #[serde(default)]
    pub hook_point: GuardrailHookPoint,

    /// Behavior when a remote-API guardrail (today `kind=bedrock`)
    /// can't reach its upstream. `true` lets the request through
    /// (recorded in usage_events.guardrail_bypassed_reason);
    /// `false` blocks with 422. No-op for `kind=keyword`. Defaults
    /// `true` (matches the PG schema default + PRD-09c §6.4).
    #[serde(default = "default_fail_open")]
    pub fail_open: bool,

    /// The provider discriminator + its config. Use serde's flattening
    /// so the wire shape is `{ kind: "keyword", patterns: [...] }`
    /// rather than `{ kind: "keyword", keyword: { patterns: [...] }}`.
    #[serde(flatten)]
    pub config: GuardrailKind,

    // --- P0c additive fields (outer-struct level; no deny_unknown_fields) ---
    //
    // cp-api's marshalGuardrailKV will start emitting these once the P0c
    // CP PR lands. Until then, old kine rows omit them and the defaults apply.
    /// How the DP behaves when this guardrail fires.
    /// `"block"` (default) — reject the request.
    /// `"monitor"` — let the request through and record the event
    ///   (**not yet implemented**; the DP currently always blocks regardless
    ///   of this field — do not set `"monitor"` expecting pass-through
    ///   behavior until a future release wires it into the chain).
    #[serde(default = "default_enforcement_mode")]
    pub enforcement_mode: String,

    /// When `true`, a runtime error in this guardrail's evaluation
    /// (e.g. a Bedrock timeout) is treated as fatal: the request is
    /// blocked regardless of `fail_open`.
    /// When `false` (default), `fail_open` governs the error path.
    ///
    /// **Not yet implemented** — the field is stored and forwarded to
    /// the CP dashboard but the DP does not yet consult it; `fail_open`
    /// alone governs error behavior in the current release.
    #[serde(default)]
    pub mandatory: bool,

    /// Which traffic directions this guardrail applies to when resolved
    /// through an attachment. Values: `"input"`, `"output"`, `"both"` (default).
    ///
    /// Stored and forwarded to the CP dashboard. Direction-based filtering
    /// in `GuardrailIndex::resolve` is not yet implemented; the `hook_point`
    /// field on the guardrail definition provides equivalent per-hook-point
    /// control for keyword rules.
    #[serde(default = "default_direction")]
    pub direction: String,

    /// RFC3339 creation timestamp of the row, projected by cp-api so the
    /// DP can evaluate guardrail chains oldest-first — matching the order
    /// the dashboard lists them in (#519 B.4a). Optional: cp-api's
    /// `marshalGuardrailKV` doesn't emit it yet, and admin-API rows may
    /// omit it; rows without it sort after rows that have it, tied broken
    /// by id, so chain order stays total and deterministic either way.
    /// RFC3339 timestamps in a fixed (UTC) offset compare correctly as
    /// strings, so no parsing is needed.
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
/// `Env` applies to every request in the environment (the pre-P0c behaviour).
/// The narrower scopes let operators attach a guardrail to just the models,
/// API keys, or teams that need it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum GuardrailScopeType {
    Env,
    Model,
    ApiKey,
    Team,
}

/// One attachment row — written by cp-api to `/aisix/<env>/guardrail_attachments/<uuid>`.
///
/// The DP loads these alongside the guardrail definitions and builds a
/// `GuardrailIndex` that resolves the applicable chain per request via
/// `scope_type` + `scope_id` matching.
///
/// `deny_unknown_fields` is intentionally NOT set: cp-api includes `env_id`
/// in the payload (for its own idempotency checks) which the DP doesn't need.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
pub struct GuardrailAttachment {
    /// UUID of the guardrail definition this attachment points to.
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

    /// When `false`, `GuardrailIndex::resolve` skips this attachment
    /// entirely (same as the row not existing).
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
