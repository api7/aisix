use async_trait::async_trait;

/// Declares metadata about a guardrail implementation.
pub trait GuardrailMeta: Send + Sync + 'static {
    fn name(&self) -> &'static str;

    fn supported_stages(&self) -> &'static [GuardrailStage] {
        &[GuardrailStage::Input, GuardrailStage::Output]
    }

    fn supports_stage(&self, stage: GuardrailStage) -> bool {
        self.supported_stages().contains(&stage)
    }
}

/// Request lifecycle stage where a guardrail can run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardrailStage {
    Input,
    Output,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuardrailRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuardrailMessageContent {
    Text(String),
    Parts(Vec<GuardrailContentPart>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuardrailContentPart {
    Text { text: String },
    ImageUrl { image_url: GuardrailImageUrl },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuardrailImageUrl {
    pub url: String,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuardrailToolCall {
    pub id: String,
    pub r#type: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuardrailMessage {
    pub role: GuardrailRole,
    pub content: Option<GuardrailMessageContent>,
    pub name: Option<String>,
    pub tool_calls: Option<Vec<GuardrailToolCall>>,
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InputGuardrailPayload {
    pub messages: Vec<GuardrailMessage>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OutputGuardrailPayload {
    pub messages: Vec<GuardrailMessage>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuardrailCheckPayload {
    Input(InputGuardrailPayload),
    Output(OutputGuardrailPayload),
}

impl GuardrailCheckPayload {
    pub fn stage(&self) -> GuardrailStage {
        match self {
            Self::Input(_) => GuardrailStage::Input,
            Self::Output(_) => GuardrailStage::Output,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuardrailOutcome {
    Allow,
    Rewrite(GuardrailCheckPayload),
    Block { reason: String },
}

/// Runtime contract for message-level guardrail checks.
#[async_trait]
pub trait GuardrailRuntime<C>: GuardrailMeta {
    type Error: std::error::Error + Send + Sync + 'static;

    async fn check(
        &self,
        payload: &GuardrailCheckPayload,
        config: &C,
    ) -> Result<GuardrailOutcome, Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::{
        GuardrailCheckPayload, GuardrailContentPart, GuardrailImageUrl, GuardrailMessage,
        GuardrailMessageContent, GuardrailMeta, GuardrailOutcome, GuardrailRole, GuardrailStage,
        GuardrailToolCall, InputGuardrailPayload, OutputGuardrailPayload,
    };

    struct DummyGuardrailMeta;

    impl GuardrailMeta for DummyGuardrailMeta {
        fn name(&self) -> &'static str {
            "dummy"
        }
    }

    #[test]
    fn guardrail_meta_should_support_input_and_output_by_default() {
        let meta = DummyGuardrailMeta;

        assert!(meta.supports_stage(GuardrailStage::Input));
        assert!(meta.supports_stage(GuardrailStage::Output));
    }

    #[test]
    fn guardrail_check_payload_should_report_its_stage() {
        let input_payload = GuardrailCheckPayload::Input(InputGuardrailPayload::default());
        let output_payload = GuardrailCheckPayload::Output(OutputGuardrailPayload::default());

        assert_eq!(input_payload.stage(), GuardrailStage::Input);
        assert_eq!(output_payload.stage(), GuardrailStage::Output);
    }

    #[test]
    fn guardrail_check_payload_should_embed_tool_calls_within_assistant_messages() {
        let payload = GuardrailCheckPayload::Output(OutputGuardrailPayload {
            messages: vec![GuardrailMessage {
                role: GuardrailRole::Assistant,
                content: None,
                name: None,
                tool_calls: Some(vec![GuardrailToolCall {
                    id: "call_weather_1".into(),
                    r#type: "function".into(),
                    name: "get_weather".into(),
                    arguments: r#"{"city":"Hangzhou"}"#.into(),
                }]),
                tool_call_id: None,
            }],
        });

        let GuardrailCheckPayload::Output(payload) = payload else {
            panic!("expected output payload");
        };

        assert_eq!(payload.messages.len(), 1);
        assert_eq!(payload.messages[0].role, GuardrailRole::Assistant);
        assert_eq!(payload.messages[0].tool_calls.as_ref().unwrap().len(), 1);
        assert_eq!(
            payload.messages[0].tool_calls.as_ref().unwrap()[0].name,
            "get_weather"
        );
    }

    #[test]
    fn guardrail_check_payload_should_represent_tool_results_as_tool_messages() {
        let payload = GuardrailCheckPayload::Input(InputGuardrailPayload {
            messages: vec![GuardrailMessage {
                role: GuardrailRole::Tool,
                content: Some(GuardrailMessageContent::Text(
                    r#"{"temperature":23}"#.into(),
                )),
                name: None,
                tool_calls: None,
                tool_call_id: Some("call_weather_1".into()),
            }],
        });

        let GuardrailCheckPayload::Input(payload) = payload else {
            panic!("expected input payload");
        };

        assert_eq!(payload.messages.len(), 1);
        assert_eq!(payload.messages[0].role, GuardrailRole::Tool);
        assert_eq!(
            payload.messages[0].tool_call_id.as_deref(),
            Some("call_weather_1")
        );
        assert_eq!(
            payload.messages[0].content,
            Some(GuardrailMessageContent::Text(
                r#"{"temperature":23}"#.into()
            ))
        );
    }

    #[test]
    fn guardrail_message_should_preserve_multimodal_content_parts() {
        let payload = GuardrailCheckPayload::Input(InputGuardrailPayload {
            messages: vec![GuardrailMessage {
                role: GuardrailRole::User,
                content: Some(GuardrailMessageContent::Parts(vec![
                    GuardrailContentPart::Text {
                        text: "describe this image".into(),
                    },
                    GuardrailContentPart::ImageUrl {
                        image_url: GuardrailImageUrl {
                            url: "https://example.com/cat.png".into(),
                            detail: Some("high".into()),
                        },
                    },
                ])),
                name: Some("alice".into()),
                tool_calls: None,
                tool_call_id: None,
            }],
        });

        let GuardrailCheckPayload::Input(payload) = payload else {
            panic!("expected input payload");
        };

        assert_eq!(payload.messages[0].name.as_deref(), Some("alice"));
        assert_eq!(
            payload.messages[0].content,
            Some(GuardrailMessageContent::Parts(vec![
                GuardrailContentPart::Text {
                    text: "describe this image".into(),
                },
                GuardrailContentPart::ImageUrl {
                    image_url: GuardrailImageUrl {
                        url: "https://example.com/cat.png".into(),
                        detail: Some("high".into()),
                    },
                },
            ]))
        );
    }

    #[test]
    fn guardrail_outcome_should_allow_rewriting_full_payloads() {
        let outcome =
            GuardrailOutcome::Rewrite(GuardrailCheckPayload::Input(InputGuardrailPayload {
                messages: vec![GuardrailMessage {
                    role: GuardrailRole::User,
                    content: Some(GuardrailMessageContent::Text("hello".into())),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                }],
            }));

        let GuardrailOutcome::Rewrite(GuardrailCheckPayload::Input(payload)) = outcome else {
            panic!("expected rewrite input outcome");
        };

        assert_eq!(
            payload.messages[0].content,
            Some(GuardrailMessageContent::Text("hello".into()))
        );
    }
}
