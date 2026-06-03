//! Keyword + regex blocklist guardrail.
//!
//! Two pattern flavours mix freely in the same blocklist:
//! - `Literal(s)`: case-insensitive substring match.
//! - `Regex(re)`: compiled once at construction; case-sensitivity is
//!   the caller's responsibility (use `(?i)` in the regex if needed).
//!
//! Applies to both input messages (concatenated content of every
//! message) and output content. Whichever side a request triggers, the
//! verdict carries the matching pattern text in `reason` so operators
//! can debug from the access log.

use aisix_gateway::{ChatFormat, ChatResponse};
use async_trait::async_trait;
use regex::Regex;

use crate::{Guardrail, GuardrailVerdict};

#[derive(Debug, Clone)]
pub enum KeywordRule {
    Literal(String),
    Regex(Regex),
}

impl KeywordRule {
    pub fn literal(s: impl Into<String>) -> Self {
        Self::Literal(s.into())
    }

    pub fn regex(pattern: &str) -> Result<Self, regex::Error> {
        Regex::new(pattern).map(Self::Regex)
    }

    fn matches(&self, haystack: &str) -> bool {
        match self {
            KeywordRule::Literal(needle) => {
                let h = haystack.to_lowercase();
                let n = needle.to_lowercase();
                !needle.is_empty() && h.contains(&n)
            }
            KeywordRule::Regex(re) => re.is_match(haystack),
        }
    }

    fn description(&self) -> String {
        match self {
            KeywordRule::Literal(s) => format!("literal {s:?}"),
            KeywordRule::Regex(r) => format!("regex /{}/", r.as_str()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct KeywordBlocklist {
    rules: Vec<KeywordRule>,
    /// Apply on input messages.
    pub check_input_enabled: bool,
    /// Apply on output content.
    pub check_output_enabled: bool,
}

impl KeywordBlocklist {
    pub fn new(rules: Vec<KeywordRule>) -> Self {
        Self {
            rules,
            check_input_enabled: true,
            check_output_enabled: true,
        }
    }

    pub fn input_only(rules: Vec<KeywordRule>) -> Self {
        Self {
            rules,
            check_input_enabled: true,
            check_output_enabled: false,
        }
    }

    pub fn output_only(rules: Vec<KeywordRule>) -> Self {
        Self {
            rules,
            check_input_enabled: false,
            check_output_enabled: true,
        }
    }

    fn first_match<'a>(&'a self, text: &str) -> Option<&'a KeywordRule> {
        self.rules.iter().find(|r| r.matches(text))
    }
}

#[async_trait]
impl Guardrail for KeywordBlocklist {
    fn name(&self) -> &'static str {
        "keyword_blocklist"
    }

    async fn check_input(&self, req: &ChatFormat) -> GuardrailVerdict {
        if !self.check_input_enabled {
            return GuardrailVerdict::Allow;
        }
        // Concatenate the message contents — checking each message
        // separately would cost the same and never catch any extra
        // hits since rules don't span messages.
        let combined: String = req
            .messages
            .iter()
            .map(crate::message_scan_text)
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        match self.first_match(&combined) {
            Some(rule) => GuardrailVerdict::Block {
                reason: format!("input blocked by {}", rule.description()),
            },
            None => GuardrailVerdict::Allow,
        }
    }

    async fn check_output(&self, resp: &ChatResponse) -> GuardrailVerdict {
        if !self.check_output_enabled {
            return GuardrailVerdict::Allow;
        }
        // Inspect content + tool-call output (#448), not just content.
        let text = resp.guardrail_output_text();
        match self.first_match(&text) {
            Some(rule) => GuardrailVerdict::Block {
                reason: format!("output blocked by {}", rule.description()),
            },
            None => GuardrailVerdict::Allow,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_gateway::{ChatMessage, FinishReason, UsageStats};

    fn req(messages: &[(&str, &str)]) -> ChatFormat {
        let msgs = messages
            .iter()
            .map(|(role, content)| match *role {
                "system" => ChatMessage::system(*content),
                "user" => ChatMessage::user(*content),
                _ => ChatMessage::assistant(*content),
            })
            .collect();
        ChatFormat::new("m", msgs)
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

    #[tokio::test]
    async fn literal_match_is_case_insensitive() {
        let g = KeywordBlocklist::new(vec![KeywordRule::literal("Forbidden")]);
        let v = g
            .check_input(&req(&[("user", "say the FORBIDDEN word")]))
            .await;
        assert!(v.is_block());
    }

    #[tokio::test]
    async fn matches_text_in_content_blocks_when_flat_content_empty() {
        // #465: a message whose text lives only in content_blocks (empty
        // top-level content — the round-trip shape) must still be
        // scanned. Before the shared message_scan_text helper this
        // bypassed moderation entirely.
        let msg: ChatMessage = serde_json::from_value(serde_json::json!({
            "role": "user",
            "content": "",
            "content_blocks": [{"type": "text", "text": "the FORBIDDEN word"}]
        }))
        .unwrap();
        let g = KeywordBlocklist::new(vec![KeywordRule::literal("Forbidden")]);
        let v = g.check_input(&ChatFormat::new("m", vec![msg])).await;
        assert!(v.is_block(), "content-block text must be scanned");
    }

    #[tokio::test]
    async fn empty_literal_pattern_never_matches() {
        let g = KeywordBlocklist::new(vec![KeywordRule::literal("")]);
        let v = g.check_input(&req(&[("user", "anything")])).await;
        assert_eq!(v, GuardrailVerdict::Allow);
    }

    #[tokio::test]
    async fn regex_pattern_matches_when_provided() {
        let g = KeywordBlocklist::new(vec![
            KeywordRule::regex(r"\bssn:\s*\d{3}-\d{2}-\d{4}").unwrap()
        ]);
        let v = g
            .check_input(&req(&[("user", "the user's ssn: 123-45-6789 is")]))
            .await;
        assert!(v.is_block());
    }

    #[tokio::test]
    async fn no_match_returns_allow() {
        let g = KeywordBlocklist::new(vec![KeywordRule::literal("nothing here")]);
        let v = g.check_input(&req(&[("user", "hello world")])).await;
        assert_eq!(v, GuardrailVerdict::Allow);
    }

    #[tokio::test]
    async fn output_check_runs_against_response_content() {
        let g = KeywordBlocklist::new(vec![KeywordRule::literal("dangerous")]);
        let v = g.check_output(&resp("here is a dangerous answer")).await;
        assert!(v.is_block());
    }

    #[tokio::test]
    async fn output_check_inspects_tool_call_arguments() {
        // A forbidden word that appears ONLY inside tool_call arguments
        // (message.content is empty) must still be blocked — tool-call
        // output is client-visible and must not bypass guardrails (#448).
        let g = KeywordBlocklist::new(vec![KeywordRule::literal("dangerous")]);
        let mut msg = ChatMessage::assistant("");
        msg.extra.insert(
            "tool_calls".into(),
            serde_json::json!([{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "run",
                    "arguments": "{\"cmd\":\"do something dangerous\"}"
                }
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

    #[tokio::test]
    async fn input_only_skips_output_checks() {
        let g = KeywordBlocklist::input_only(vec![KeywordRule::literal("zeta")]);
        let v = g.check_output(&resp("zeta zeta zeta")).await;
        assert_eq!(v, GuardrailVerdict::Allow);
    }

    #[tokio::test]
    async fn output_only_skips_input_checks() {
        let g = KeywordBlocklist::output_only(vec![KeywordRule::literal("zeta")]);
        let v = g.check_input(&req(&[("user", "zeta zeta")])).await;
        assert_eq!(v, GuardrailVerdict::Allow);
    }

    #[tokio::test]
    async fn first_matching_rule_wins_so_reason_is_deterministic() {
        let g = KeywordBlocklist::new(vec![
            KeywordRule::literal("alpha"),
            KeywordRule::literal("beta"),
        ]);
        let v = g.check_input(&req(&[("user", "alpha and beta")])).await;
        if let GuardrailVerdict::Block { reason } = v {
            assert!(reason.contains("alpha"));
        } else {
            panic!("expected Block");
        }
    }

    #[test]
    fn invalid_regex_is_a_clean_error_not_a_panic() {
        assert!(KeywordRule::regex("[unclosed").is_err());
    }
}
