//! In-process sensitive-data detection + redaction guardrail (#932).
//!
//! Two detector sources mix in one rule list:
//! - built-in detectors (`BUILTIN_DETECTORS`): curated regexes for common
//!   PII/secret shapes, several backed by a checksum validator (Luhn for
//!   bank cards, ISO 7064 MOD 11-2 for Chinese national IDs) so a random
//!   digit run doesn't false-positive;
//! - custom patterns: operator-supplied regexes.
//!
//! Each rule carries an action:
//! - `Block`: the request/response is rejected (422 content-filter) —
//!   same enforcement path as the keyword blocklist.
//! - `Mask`: each matched span is rewritten to `[<DETECTOR>_REDACTED]`
//!   and processing continues. Callers apply the rewrite through
//!   [`crate::Guardrail::redact_input_text`] /
//!   [`crate::Guardrail::redact_output_text`].
//!
//! The detector NAME is the only thing that ever leaves this module —
//! block reasons, telemetry counts, and mask tokens all carry the name,
//! never the matched value (#153 anti-leak rule, and the #932 acceptance
//! criterion that redacted values must not appear in gateway logs).

use std::borrow::Cow;
use std::collections::BTreeMap;

use aisix_gateway::{ChatFormat, ChatResponse};
use async_trait::async_trait;
use regex::Regex;

use crate::{Guardrail, GuardrailVerdict, Redaction, StreamOutputPolicy};

/// What to do when a detector matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PiiAction {
    /// Rewrite the matched span to `[<DETECTOR>_REDACTED]` and continue.
    Mask,
    /// Reject the request/response (422 content-filter).
    Block,
}

impl PiiAction {
    /// Parse the wire string. `None` for anything unrecognised — the
    /// build layer decides whether to fall back or reject.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "mask" => Some(PiiAction::Mask),
            "block" => Some(PiiAction::Block),
            _ => None,
        }
    }
}

/// Secondary validator applied to each regex match before it counts as a
/// detection. Cuts false positives on digit-run detectors where the regex
/// alone is too permissive.
type MatchValidator = fn(&str) -> bool;

/// One compiled detector rule.
pub struct PiiRule {
    /// Detector id (built-in) or operator-assigned custom-pattern name.
    /// Surfaced in mask tokens / reasons / counts.
    name: String,
    regex: Regex,
    action: PiiAction,
    validate: Option<MatchValidator>,
    /// Pre-computed `[<NAME>_REDACTED]` token.
    mask_token: String,
}

impl PiiRule {
    pub fn new(
        name: impl Into<String>,
        pattern: &str,
        action: PiiAction,
        validate: Option<MatchValidator>,
    ) -> Result<Self, regex::Error> {
        let name = name.into();
        let mask_token = mask_token(&name);
        Ok(Self {
            regex: Regex::new(pattern)?,
            name,
            action,
            validate,
            mask_token,
        })
    }

    /// `true` when `candidate` is a real detection (regex already matched;
    /// this applies the optional checksum validator).
    fn accepts(&self, candidate: &str) -> bool {
        self.validate.is_none_or(|v| v(candidate))
    }

    /// First validated match in `text`, or `None`.
    fn detects(&self, text: &str) -> bool {
        self.regex.find_iter(text).any(|m| self.accepts(m.as_str()))
    }

    /// Rewrite every validated match in `text` to the rule's mask token.
    /// Returns the match count (0 = `text` returned unchanged).
    fn mask_all<'t>(&self, text: &'t str) -> (Cow<'t, str>, u32) {
        let mut count = 0u32;
        let out = self.regex.replace_all(text, |caps: &regex::Captures<'_>| {
            let m = &caps[0];
            if self.accepts(m) {
                count += 1;
                self.mask_token.clone()
            } else {
                m.to_string()
            }
        });
        (out, count)
    }
}

/// `[<NAME>_REDACTED]`, uppercased, non-alphanumerics folded to `_` so an
/// operator-assigned custom name can't inject brackets/spaces into the token.
fn mask_token(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect();
    format!("[{cleaned}_REDACTED]")
}

/// One built-in detector: `(id, pattern, validator)`.
///
/// Patterns lean conservative — `\b` boundaries and checksum validators —
/// because a mask false-positive corrupts user content (worse than a
/// keyword-blocklist false positive, which merely rejects). IDs are wire
/// contract: cp-api's validator and the dashboard's detector list carry
/// the same set.
pub const BUILTIN_DETECTORS: &[(&str, &str, Option<MatchValidator>)] = &[
    (
        "email",
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        None,
    ),
    // Chinese mainland mobile: 1[3-9] + 9 digits, standalone digit run.
    ("china_mobile", r"\b1[3-9]\d{9}\b", None),
    // 18-char Chinese national ID (17 digits + check char), verified with
    // the ISO 7064 MOD 11-2 checksum.
    (
        "china_id_card",
        r"\b\d{17}[0-9Xx]\b",
        Some(china_id_checksum),
    ),
    // 13–19 digit PAN (bank/credit card), Luhn-verified.
    ("bank_card", r"\b\d{13,19}\b", Some(luhn_checksum)),
    ("us_ssn", r"\b\d{3}-\d{2}-\d{4}\b", None),
    (
        "ip_address",
        r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b",
        None,
    ),
    // Well-known credential signatures: OpenAI, AWS access key id, GitHub
    // (classic + fine-grained), Slack, Google API key.
    (
        "api_key",
        r"\b(?:sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{22,}|xox[baprs]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{35})\b",
        None,
    ),
    (
        "jwt",
        r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b",
        None,
    ),
    (
        "private_key",
        r"(?s)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        None,
    ),
];

/// Look up a built-in detector by id and compile it with `action`.
pub fn builtin_rule(id: &str, action: PiiAction) -> Option<PiiRule> {
    BUILTIN_DETECTORS
        .iter()
        .find(|(name, _, _)| *name == id)
        .map(|(name, pattern, validate)| {
            PiiRule::new(*name, pattern, action, *validate)
                .expect("built-in PII detector pattern must compile")
        })
}

/// ISO 7064 MOD 11-2 check for the 18-char Chinese national ID.
fn china_id_checksum(s: &str) -> bool {
    let bytes = s.as_bytes();
    if bytes.len() != 18 {
        return false;
    }
    const WEIGHTS: [u32; 17] = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2];
    const CHECK: [u8; 11] = *b"10X98765432";
    let mut sum = 0u32;
    for (i, b) in bytes[..17].iter().enumerate() {
        if !b.is_ascii_digit() {
            return false;
        }
        sum += u32::from(b - b'0') * WEIGHTS[i];
    }
    let expected = CHECK[(sum % 11) as usize];
    bytes[17].to_ascii_uppercase() == expected
}

/// Luhn check for payment-card numbers.
fn luhn_checksum(s: &str) -> bool {
    let mut sum = 0u32;
    let mut double = false;
    for b in s.bytes().rev() {
        if !b.is_ascii_digit() {
            return false;
        }
        let mut d = u32::from(b - b'0');
        if double {
            d *= 2;
            if d > 9 {
                d -= 9;
            }
        }
        sum += d;
        double = !double;
    }
    sum.is_multiple_of(10)
}

/// The runtime guardrail for `kind: "pii"`.
pub struct PiiGuardrail {
    rules: Vec<PiiRule>,
    check_input_enabled: bool,
    check_output_enabled: bool,
    max_buffer_bytes: usize,
    on_buffer_exceeded_fail_open: bool,
}

impl PiiGuardrail {
    pub fn new(
        rules: Vec<PiiRule>,
        hook_point: aisix_core::models::GuardrailHookPoint,
        max_buffer_bytes: usize,
        on_buffer_exceeded_fail_open: bool,
    ) -> Self {
        use aisix_core::models::GuardrailHookPoint as HP;
        Self {
            rules,
            check_input_enabled: matches!(hook_point, HP::Input | HP::Both),
            check_output_enabled: matches!(hook_point, HP::Output | HP::Both),
            max_buffer_bytes,
            on_buffer_exceeded_fail_open,
        }
    }

    /// First block-action rule that detects in `text`, for the verdict
    /// reason. Mask-action rules never block.
    fn first_block_match(&self, text: &str) -> Option<&PiiRule> {
        self.rules
            .iter()
            .find(|r| r.action == PiiAction::Block && r.detects(text))
    }

    /// Apply every mask-action rule to `text`, in rule order. `None` when
    /// nothing matched (caller keeps the original untouched).
    fn mask_text(&self, text: &str) -> Option<Redaction> {
        let mut counts: BTreeMap<String, u32> = BTreeMap::new();
        let mut current = Cow::Borrowed(text);
        for rule in self.rules.iter().filter(|r| r.action == PiiAction::Mask) {
            let (next, count) = rule.mask_all(&current);
            if count > 0 {
                *counts.entry(rule.name.clone()).or_insert(0) += count;
                current = Cow::Owned(next.into_owned());
            }
        }
        if counts.is_empty() {
            None
        } else {
            Some(Redaction {
                text: current.into_owned(),
                counts,
            })
        }
    }
}

#[async_trait]
impl Guardrail for PiiGuardrail {
    fn name(&self) -> &'static str {
        "pii"
    }

    fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    fn runs_on_output(&self) -> bool {
        self.check_output_enabled
    }

    /// Masking a streamed response requires the whole response held back —
    /// a span can cross any chunk boundary, and a mask can't be applied
    /// retroactively to bytes already on the wire. Cap + overflow policy
    /// come from the row config.
    fn stream_output_policy(&self) -> StreamOutputPolicy {
        StreamOutputPolicy::BufferFull {
            max_buffer_bytes: self.max_buffer_bytes,
            on_exceeded_fail_open: self.on_buffer_exceeded_fail_open,
        }
    }

    async fn check_input(&self, req: &ChatFormat) -> GuardrailVerdict {
        if !self.check_input_enabled {
            return GuardrailVerdict::Allow;
        }
        let combined: String = req
            .messages
            .iter()
            .map(crate::message_scan_text)
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        match self.first_block_match(&combined) {
            // Reason carries the detector NAME only — never the matched
            // value (#153 + #932 no-leak criterion).
            Some(rule) => {
                GuardrailVerdict::block(format!("input blocked by pii detector '{}'", rule.name))
            }
            None => GuardrailVerdict::Allow,
        }
    }

    async fn check_output(&self, resp: &ChatResponse) -> GuardrailVerdict {
        if !self.check_output_enabled {
            return GuardrailVerdict::Allow;
        }
        let text = resp.guardrail_output_text();
        match self.first_block_match(&text) {
            Some(rule) => {
                GuardrailVerdict::block(format!("output blocked by pii detector '{}'", rule.name))
            }
            None => GuardrailVerdict::Allow,
        }
    }

    fn redacts_input(&self) -> bool {
        self.check_input_enabled && self.rules.iter().any(|r| r.action == PiiAction::Mask)
    }

    fn redacts_output(&self) -> bool {
        self.check_output_enabled && self.rules.iter().any(|r| r.action == PiiAction::Mask)
    }

    fn redact_input_text(&self, text: &str) -> Option<Redaction> {
        if !self.redacts_input() {
            return None;
        }
        self.mask_text(text)
    }

    fn redact_output_text(&self, text: &str) -> Option<Redaction> {
        if !self.redacts_output() {
            return None;
        }
        self.mask_text(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_core::models::GuardrailHookPoint;
    use aisix_gateway::{ChatMessage, FinishReason, UsageStats};

    fn guardrail(rules: Vec<PiiRule>) -> PiiGuardrail {
        PiiGuardrail::new(rules, GuardrailHookPoint::Both, 262_144, false)
    }

    fn builtin(id: &str, action: PiiAction) -> PiiRule {
        builtin_rule(id, action).expect("known builtin")
    }

    fn req(msg: &str) -> ChatFormat {
        ChatFormat::new("m", vec![ChatMessage::user(msg)])
    }

    fn resp(content: &str) -> ChatResponse {
        ChatResponse {
            id: "r".into(),
            model: "m".into(),
            message: ChatMessage::assistant(content),
            finish_reason: FinishReason::Stop,
            usage: UsageStats::new(0, 0),
        }
    }

    #[test]
    fn email_masks_span_only() {
        let g = guardrail(vec![builtin("email", PiiAction::Mask)]);
        let r = g
            .redact_input_text("contact me at alice@example.com please")
            .unwrap();
        assert_eq!(r.text, "contact me at [EMAIL_REDACTED] please");
        assert_eq!(r.counts.get("email"), Some(&1));
    }

    #[test]
    fn china_mobile_masks_standalone_number_only() {
        let g = guardrail(vec![builtin("china_mobile", PiiAction::Mask)]);
        let r = g
            .redact_input_text("我的手机号是 13800138000，请回电")
            .unwrap();
        assert_eq!(r.text, "我的手机号是 [CHINA_MOBILE_REDACTED]，请回电");
        // Embedded in a longer digit run → not a phone → untouched.
        assert!(g.redact_input_text("订单号 913800138000123").is_none());
    }

    #[test]
    fn china_id_card_requires_checksum() {
        let g = guardrail(vec![builtin("china_id_card", PiiAction::Mask)]);
        // Valid checksum (11010519491231002X is the canonical example).
        let r = g
            .redact_input_text("身份证 11010519491231002X 已登记")
            .unwrap();
        assert_eq!(r.text, "身份证 [CHINA_ID_CARD_REDACTED] 已登记");
        // Same shape, broken check digit → untouched.
        assert!(g
            .redact_input_text("身份证 110105194912310021 已登记")
            .is_none());
    }

    #[test]
    fn bank_card_requires_luhn() {
        let g = guardrail(vec![builtin("bank_card", PiiAction::Mask)]);
        // 4111111111111111 passes Luhn.
        let r = g.redact_input_text("card 4111111111111111 ok").unwrap();
        assert_eq!(r.text, "card [BANK_CARD_REDACTED] ok");
        assert!(g.redact_input_text("card 4111111111111112 ok").is_none());
    }

    #[test]
    fn api_key_and_jwt_signatures_mask() {
        let g = guardrail(vec![
            builtin("api_key", PiiAction::Mask),
            builtin("jwt", PiiAction::Mask),
        ]);
        let r = g
            .redact_input_text(
                "key sk-abcdefghijklmnopqrstuv and token eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.dGVzdHNpZ25hdHVyZQ",
            )
            .unwrap();
        assert_eq!(r.text, "key [API_KEY_REDACTED] and token [JWT_REDACTED]");
        assert_eq!(r.counts.len(), 2);
    }

    #[test]
    fn private_key_block_masks_across_lines() {
        let g = guardrail(vec![builtin("private_key", PiiAction::Mask)]);
        let text = "-----BEGIN RSA PRIVATE KEY-----\nMIIE...\n-----END RSA PRIVATE KEY-----";
        let r = g.redact_input_text(text).unwrap();
        assert_eq!(r.text, "[PRIVATE_KEY_REDACTED]");
    }

    #[test]
    fn custom_pattern_masks_with_sanitised_token() {
        let rule = PiiRule::new("employee id", r"\bEMP-\d{6}\b", PiiAction::Mask, None).unwrap();
        let g = guardrail(vec![rule]);
        let r = g.redact_input_text("badge EMP-123456").unwrap();
        assert_eq!(r.text, "badge [EMPLOYEE_ID_REDACTED]");
        assert_eq!(r.counts.get("employee id"), Some(&1));
    }

    #[tokio::test]
    async fn block_action_blocks_and_names_detector_only() {
        let g = guardrail(vec![builtin("china_id_card", PiiAction::Block)]);
        let v = g.check_input(&req("id 11010519491231002X")).await;
        match v {
            GuardrailVerdict::Block { reason, .. } => {
                assert!(reason.contains("china_id_card"), "reason: {reason}");
                assert!(
                    !reason.contains("11010519491231002X"),
                    "matched value must never appear in the reason: {reason}",
                );
            }
            other => panic!("expected Block, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn mask_action_does_not_block() {
        let g = guardrail(vec![builtin("email", PiiAction::Mask)]);
        assert_eq!(
            g.check_input(&req("mail alice@example.com")).await,
            GuardrailVerdict::Allow,
        );
        assert_eq!(
            g.check_output(&resp("mail alice@example.com")).await,
            GuardrailVerdict::Allow,
        );
    }

    #[tokio::test]
    async fn block_action_checks_output_tool_calls() {
        let g = guardrail(vec![builtin("email", PiiAction::Block)]);
        let mut msg = ChatMessage::assistant("");
        msg.extra.insert(
            "tool_calls".into(),
            serde_json::json!([{
                "function": { "name": "send", "arguments": "{\"to\":\"bob@example.com\"}" }
            }]),
        );
        let r = ChatResponse {
            id: "r".into(),
            model: "m".into(),
            message: msg,
            finish_reason: FinishReason::Stop,
            usage: UsageStats::new(0, 0),
        };
        assert!(g.check_output(&r).await.is_block());
    }

    #[test]
    fn hook_point_gates_redaction_sides() {
        use aisix_core::models::GuardrailHookPoint as HP;
        let input_only = PiiGuardrail::new(
            vec![builtin("email", PiiAction::Mask)],
            HP::Input,
            262_144,
            false,
        );
        assert!(input_only.redacts_input());
        assert!(!input_only.redacts_output());
        assert!(input_only.redact_output_text("a@b.co").is_none());
        assert!(!input_only.runs_on_output());

        let output_only = PiiGuardrail::new(
            vec![builtin("email", PiiAction::Mask)],
            HP::Output,
            262_144,
            false,
        );
        assert!(!output_only.redacts_input());
        assert!(output_only.redacts_output());
        assert!(output_only.redact_input_text("a@b.co").is_none());
    }

    #[test]
    fn stream_policy_uses_row_buffer_config() {
        let g = PiiGuardrail::new(
            vec![builtin("email", PiiAction::Mask)],
            GuardrailHookPoint::Both,
            1_024,
            true,
        );
        assert_eq!(
            g.stream_output_policy(),
            StreamOutputPolicy::BufferFull {
                max_buffer_bytes: 1_024,
                on_exceeded_fail_open: true,
            },
        );
    }

    #[test]
    fn multiple_matches_count_per_detector() {
        let g = guardrail(vec![builtin("email", PiiAction::Mask)]);
        let r = g.redact_input_text("a@x.com then b@y.org").unwrap();
        assert_eq!(r.text, "[EMAIL_REDACTED] then [EMAIL_REDACTED]");
        assert_eq!(r.counts.get("email"), Some(&2));
    }

    #[test]
    fn no_match_returns_none() {
        let g = guardrail(vec![builtin("email", PiiAction::Mask)]);
        assert!(g.redact_input_text("nothing sensitive here").is_none());
    }

    #[test]
    fn ssn_and_ip_mask() {
        let g = guardrail(vec![
            builtin("us_ssn", PiiAction::Mask),
            builtin("ip_address", PiiAction::Mask),
        ]);
        let r = g
            .redact_input_text("ssn 123-45-6789 from 192.168.1.100")
            .unwrap();
        assert_eq!(r.text, "ssn [US_SSN_REDACTED] from [IP_ADDRESS_REDACTED]");
    }

    #[test]
    fn every_builtin_detector_compiles() {
        for (id, _, _) in BUILTIN_DETECTORS {
            assert!(
                builtin_rule(id, PiiAction::Mask).is_some(),
                "builtin {id} must compile",
            );
        }
        assert!(builtin_rule("no_such_detector", PiiAction::Mask).is_none());
    }
}
