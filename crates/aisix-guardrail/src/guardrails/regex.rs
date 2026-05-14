use std::convert::Infallible;

use async_trait::async_trait;
use regex::Regex;
use serde::{Deserialize, Deserializer, Serialize, de};
use utoipa::ToSchema;

use crate::traits::{
    GuardrailCheckPayload, GuardrailContentPart, GuardrailMessage, GuardrailMessageContent,
    GuardrailMeta, GuardrailOutcome, GuardrailRuntime,
};

pub const IDENTIFIER: &str = "regex";

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct RegexGuardrailConfig {
    pub pattern: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_reason: Option<String>,

    #[serde(skip)]
    #[schema(ignore)]
    compiled_pattern: Regex,
}

impl RegexGuardrailConfig {
    pub fn new(
        pattern: impl Into<String>,
        block_reason: Option<String>,
    ) -> Result<Self, regex::Error> {
        let pattern = pattern.into();
        let compiled_pattern = Regex::new(&pattern)?;

        Ok(Self {
            pattern,
            block_reason,
            compiled_pattern,
        })
    }

    pub fn compiled_pattern(&self) -> &Regex {
        &self.compiled_pattern
    }
}

impl<'de> Deserialize<'de> for RegexGuardrailConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RawRegexGuardrailConfig {
            pattern: String,
            #[serde(default)]
            block_reason: Option<String>,
        }

        let raw = RawRegexGuardrailConfig::deserialize(deserializer)?;

        Self::new(raw.pattern, raw.block_reason)
            .map_err(|error| de::Error::custom(format!("invalid regex guardrail pattern: {error}")))
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RegexGuardrailMeta;

impl GuardrailMeta for RegexGuardrailMeta {
    fn name(&self) -> &'static str {
        IDENTIFIER
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RegexGuardrailRuntime;

impl RegexGuardrailRuntime {
    pub fn new() -> Self {
        Self
    }
}

impl GuardrailMeta for RegexGuardrailRuntime {
    fn name(&self) -> &'static str {
        IDENTIFIER
    }
}

#[async_trait]
impl GuardrailRuntime<RegexGuardrailConfig> for RegexGuardrailRuntime {
    type Error = Infallible;

    async fn check(
        &self,
        payload: &GuardrailCheckPayload,
        config: &RegexGuardrailConfig,
    ) -> Result<GuardrailOutcome, Self::Error> {
        if payload_matches(config.compiled_pattern(), payload) {
            return Ok(GuardrailOutcome::Block {
                reason: config
                    .block_reason
                    .clone()
                    .unwrap_or_else(|| "regex guardrail blocked".into()),
            });
        }

        Ok(GuardrailOutcome::Allow)
    }
}

fn payload_matches(pattern: &Regex, payload: &GuardrailCheckPayload) -> bool {
    let messages = match payload {
        GuardrailCheckPayload::Input(payload) => &payload.messages,
        GuardrailCheckPayload::Output(payload) => &payload.messages,
    };

    messages
        .iter()
        .any(|message| message_matches(pattern, message))
}

fn message_matches(pattern: &Regex, message: &GuardrailMessage) -> bool {
    match &message.content {
        Some(GuardrailMessageContent::Text(text)) => pattern.is_match(text),
        Some(GuardrailMessageContent::Parts(parts)) => parts.iter().any(|part| match part {
            GuardrailContentPart::Text { text } => pattern.is_match(text),
            GuardrailContentPart::ImageUrl { .. } => false,
        }),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{RegexGuardrailConfig, RegexGuardrailRuntime};
    use crate::traits::{
        GuardrailCheckPayload, GuardrailContentPart, GuardrailImageUrl, GuardrailMessage,
        GuardrailMessageContent, GuardrailOutcome, GuardrailRole, GuardrailRuntime,
        InputGuardrailPayload,
    };

    fn config(pattern: &str) -> RegexGuardrailConfig {
        RegexGuardrailConfig::new(pattern, Some("matched blocked content".into())).unwrap()
    }

    fn runtime() -> RegexGuardrailRuntime {
        RegexGuardrailRuntime::new()
    }

    fn input_payload(content: GuardrailMessageContent) -> GuardrailCheckPayload {
        GuardrailCheckPayload::Input(InputGuardrailPayload {
            messages: vec![GuardrailMessage {
                role: GuardrailRole::User,
                content: Some(content),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }],
        })
    }

    #[tokio::test]
    async fn blocks_when_plain_text_matches_pattern() {
        let outcome = runtime()
            .check(
                &input_payload(GuardrailMessageContent::Text(
                    "my secret token is 12345".into(),
                )),
                &config(r"secret token"),
            )
            .await
            .unwrap();

        assert_eq!(
            outcome,
            GuardrailOutcome::Block {
                reason: "matched blocked content".into(),
            }
        );
    }

    #[tokio::test]
    async fn allows_when_no_message_text_matches_pattern() {
        let outcome = runtime()
            .check(
                &input_payload(GuardrailMessageContent::Text("hello world".into())),
                &config(r"secret token"),
            )
            .await
            .unwrap();

        assert_eq!(outcome, GuardrailOutcome::Allow);
    }

    #[tokio::test]
    async fn matches_text_parts_and_ignores_non_text_parts() {
        let outcome = runtime()
            .check(
                &input_payload(GuardrailMessageContent::Parts(vec![
                    GuardrailContentPart::ImageUrl {
                        image_url: GuardrailImageUrl {
                            url: "https://example.com/cat.png".into(),
                            detail: Some("high".into()),
                        },
                    },
                    GuardrailContentPart::Text {
                        text: "contains credit card 4111111111111111".into(),
                    },
                ])),
                &config(r"\b\d{16}\b"),
            )
            .await
            .unwrap();

        assert_eq!(
            outcome,
            GuardrailOutcome::Block {
                reason: "matched blocked content".into(),
            }
        );
    }

    #[test]
    fn deserialize_rejects_invalid_patterns() {
        let error = serde_json::from_value::<RegexGuardrailConfig>(json!({
            "pattern": "[",
            "block_reason": "matched blocked content"
        }))
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("invalid regex guardrail pattern")
        );
    }
}
