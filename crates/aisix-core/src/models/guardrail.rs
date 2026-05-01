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
//! Rule kinds. v1 ships a single in-process kind plus a placeholder
//! for the AWS Bedrock provider:
//!
//!   * `keyword` — literal/regex blocklist; runs entirely in DP
//!     process. Configured via `keyword.patterns` (list of
//!     `{ kind: "literal" | "regex", value: "..." }`).
//!   * `bedrock` — calls AWS Bedrock's ApplyGuardrail (TODO; the v1
//!     loader rejects this kind so a half-wired DP doesn't silently
//!     skip the policy).
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

/// Provider discriminator. The kind drives which `*_config` block is
/// expected; serde's `tag = "kind"` keeps us honest at parse time.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum GuardrailKind {
    /// In-process literal/regex blocklist. Always available.
    Keyword(KeywordConfig),
    // Bedrock placeholder lives in the spec/PRD only for v1 — the DP
    // refuses to load it. Adding the variant here lands with the
    // ApplyGuardrail wiring; left out now to keep the snapshot
    // schema strict.
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
    fn bedrock_kind_does_not_deserialise_yet() {
        // Adding the variant lands when ApplyGuardrail is implemented.
        // Until then the loader errors on `kind: "bedrock"` rather
        // than silently letting a half-wired policy through.
        let v = json!({
            "name": "g",
            "kind": "bedrock",
            "guardrail_id": "abc"
        });
        let r: Result<Guardrail, _> = serde_json::from_value(v);
        assert!(r.is_err());
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
