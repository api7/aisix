//! `Guardrail` entity — content-policy hooks the DP runs on every
//! chat request. The control plane (cp-api) writes these to etcd at
//! `/aisix/<env>/guardrails/<uuid>`; the DP loads them on watch and
//! the `aisix-proxy::ProxyState::guardrails` chain composes the
//! enabled ones.
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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", content = "value", rename_all = "lowercase")]
pub enum KeywordPattern {
    Literal(String),
    Regex(String),
}

/// Config block for `kind: "keyword"`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum BedrockLatencyMode {
    Serial,
    Timed { timeout_ms: u32 },
}

/// Config block for `kind: "bedrock"`. Phase 1 stores the shape +
/// passes it through `aisix-guardrails::build` which logs
/// `bedrock not yet implemented` and skips the row.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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

    #[serde(skip)]
    pub(crate) runtime_id: String,
}

fn default_enabled() -> bool {
    true
}

fn default_fail_open() -> bool {
    true
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
