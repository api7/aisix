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
//!   * `bedrock` — calls AWS Bedrock's `ApplyGuardrail`. Phase 1
//!     parses + accepts the kind but the chain builder logs
//!     "bedrock not yet implemented" and skips the row; Phase 2
//!     wires the actual dispatch (PRD-09c §6.7).
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

/// Config block for `kind: "bedrock"`. Phase 1 stores the shape +
/// passes it through `aisix-guardrails::build` which logs
/// `bedrock not yet implemented` and skips the row.
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
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum GuardrailKind {
    /// In-process literal/regex blocklist. Always available.
    Keyword(KeywordConfig),
    /// AWS Bedrock managed guardrail. Phase 1 parses + persists;
    /// the chain builder skips it with a warn log. Phase 2 wires
    /// real `ApplyGuardrail` dispatch.
    Bedrock(BedrockConfig),
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
            GuardrailKind::Bedrock(_) => panic!("expected Keyword variant"),
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
