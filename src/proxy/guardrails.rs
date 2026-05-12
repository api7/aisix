use aisix_guardrail::{
    guardrails::BedrockGuardrailRuntime,
    traits::{
        GuardrailCheckPayload, GuardrailContentPart, GuardrailImageUrl, GuardrailMessage,
        GuardrailMessageContent, GuardrailOutcome, GuardrailRole, GuardrailRuntime, GuardrailStage,
        GuardrailToolCall, InputGuardrailPayload, OutputGuardrailPayload,
    },
};
use async_trait::async_trait;
use thiserror::Error;

use crate::{
    config::entities::{Model, ResourceEntry, ResourceRegistry, guardrails::GuardrailConfig},
    gateway::{
        error::GatewayError,
        types::openai::{
            ChatMessage, ContentPart, FunctionCall, ImageUrl, MessageContent, ToolCall,
        },
    },
};

#[derive(Debug, Error, PartialEq, Eq)]
pub(crate) enum GuardrailBridgeError {
    #[error("unsupported chat message role: {0}")]
    UnsupportedRole(String),
    #[error("guardrail payload did not match the expected stage: {0}")]
    UnexpectedStage(&'static str),
}

#[cfg(test)]
#[derive(Debug, Error)]
pub(crate) enum GuardrailExecutionError<E>
where
    E: std::error::Error + 'static,
{
    #[error(transparent)]
    Bridge(#[from] GuardrailBridgeError),
    #[error(transparent)]
    Runtime(E),
}

#[async_trait]
pub(crate) trait ConfiguredGuardrailRuntime: Send + Sync {
    fn name(&self) -> &'static str;

    fn supports_stage(&self, stage: GuardrailStage) -> bool;

    async fn check(
        &self,
        payload: &GuardrailCheckPayload,
    ) -> Result<Option<GuardrailOutcome>, GatewayError>;
}

struct GuardrailRuntimeHandle<R, C> {
    runtime: R,
    config: C,
}

impl<R, C> GuardrailRuntimeHandle<R, C> {
    fn new(runtime: R, config: C) -> Self {
        Self { runtime, config }
    }
}

#[async_trait]
impl<R, C> ConfiguredGuardrailRuntime for GuardrailRuntimeHandle<R, C>
where
    R: GuardrailRuntime<C> + Send + Sync,
    C: Send + Sync,
    R::Error: std::error::Error + Send + Sync + 'static,
{
    fn name(&self) -> &'static str {
        self.runtime.name()
    }

    fn supports_stage(&self, stage: GuardrailStage) -> bool {
        self.runtime.supports_stage(stage)
    }

    async fn check(
        &self,
        payload: &GuardrailCheckPayload,
    ) -> Result<Option<GuardrailOutcome>, GatewayError> {
        if !self.supports_stage(payload.stage()) {
            return Ok(None);
        }

        self.runtime
            .check(payload, &self.config)
            .await
            .map(Some)
            .map_err(|error| {
                GatewayError::Internal(format!(
                    "guardrail {} failed: {}",
                    self.runtime.name(),
                    error
                ))
            })
    }
}

pub(crate) fn chat_message_to_guardrail_message(
    message: &ChatMessage,
) -> Result<GuardrailMessage, GuardrailBridgeError> {
    Ok(GuardrailMessage {
        role: guardrail_role_from_chat_role(&message.role)?,
        content: message
            .content
            .as_ref()
            .map(guardrail_content_from_chat_content),
        name: message.name.clone(),
        tool_calls: message.tool_calls.as_ref().map(|tool_calls| {
            tool_calls
                .iter()
                .map(|tool_call| GuardrailToolCall {
                    id: tool_call.id.clone(),
                    r#type: tool_call.r#type.clone(),
                    name: tool_call.function.name.clone(),
                    arguments: tool_call.function.arguments.clone(),
                })
                .collect()
        }),
        tool_call_id: message.tool_call_id.clone(),
    })
}

pub(crate) fn guardrail_message_to_chat_message(
    message: &GuardrailMessage,
) -> Result<ChatMessage, GuardrailBridgeError> {
    Ok(ChatMessage {
        role: chat_role_from_guardrail_role(message.role.clone()).to_owned(),
        content: message
            .content
            .as_ref()
            .map(chat_content_from_guardrail_content),
        name: message.name.clone(),
        tool_calls: message.tool_calls.as_ref().map(|tool_calls| {
            tool_calls
                .iter()
                .map(|tool_call| ToolCall {
                    id: tool_call.id.clone(),
                    r#type: tool_call.r#type.clone(),
                    function: FunctionCall {
                        name: tool_call.name.clone(),
                        arguments: tool_call.arguments.clone(),
                    },
                })
                .collect()
        }),
        tool_call_id: message.tool_call_id.clone(),
    })
}

pub(crate) fn input_guardrail_payload_from_chat_messages(
    messages: &[ChatMessage],
) -> Result<InputGuardrailPayload, GuardrailBridgeError> {
    Ok(InputGuardrailPayload {
        messages: messages
            .iter()
            .map(chat_message_to_guardrail_message)
            .collect::<Result<Vec<_>, _>>()?,
    })
}

pub(crate) fn output_guardrail_payload_from_chat_messages(
    messages: &[ChatMessage],
) -> Result<OutputGuardrailPayload, GuardrailBridgeError> {
    Ok(OutputGuardrailPayload {
        messages: messages
            .iter()
            .map(chat_message_to_guardrail_message)
            .collect::<Result<Vec<_>, _>>()?,
    })
}

pub(crate) fn input_payload_to_chat_messages(
    payload: &InputGuardrailPayload,
) -> Result<Vec<ChatMessage>, GuardrailBridgeError> {
    payload
        .messages
        .iter()
        .map(guardrail_message_to_chat_message)
        .collect()
}

pub(crate) fn output_payload_to_chat_messages(
    payload: &OutputGuardrailPayload,
) -> Result<Vec<ChatMessage>, GuardrailBridgeError> {
    payload
        .messages
        .iter()
        .map(guardrail_message_to_chat_message)
        .collect()
}

pub(crate) fn input_payload_from_check_payload(
    payload: GuardrailCheckPayload,
) -> Result<InputGuardrailPayload, GuardrailBridgeError> {
    match payload {
        GuardrailCheckPayload::Input(payload) => Ok(payload),
        GuardrailCheckPayload::Output(_) => Err(GuardrailBridgeError::UnexpectedStage("input")),
    }
}

pub(crate) fn output_payload_from_check_payload(
    payload: GuardrailCheckPayload,
) -> Result<OutputGuardrailPayload, GuardrailBridgeError> {
    match payload {
        GuardrailCheckPayload::Output(payload) => Ok(payload),
        GuardrailCheckPayload::Input(_) => Err(GuardrailBridgeError::UnexpectedStage("output")),
    }
}

pub(crate) fn resolve_model_guardrails(
    model: &ResourceEntry<Model>,
    resources: &ResourceRegistry,
) -> Result<Vec<Box<dyn ConfiguredGuardrailRuntime>>, GatewayError> {
    // This direct Model -> guardrail lookup is intentionally temporary. The long-term attachment
    // point should come from policy evaluation so request-time guardrail selection is not encoded
    // in the model resource itself.
    model
        .guardrail_ids
        .iter()
        .map(|guardrail_id| {
            let guardrail = resources
                .guardrails
                .get_by_id(guardrail_id)
                .ok_or_else(|| {
                    GatewayError::Internal(format!("guardrail {} not found", guardrail_id))
                })?;

            configured_guardrail_runtime_from_configs(&guardrail.guardrail)
        })
        .collect()
}

#[cfg(test)]
pub(crate) async fn run_guardrail_check<R, C>(
    runtime: &R,
    config: &C,
    payload: GuardrailCheckPayload,
) -> Result<Option<GuardrailOutcome>, GuardrailExecutionError<R::Error>>
where
    R: GuardrailRuntime<C>,
{
    if !runtime.supports_stage(payload.stage()) {
        return Ok(None);
    }

    Ok(Some(
        runtime
            .check(&payload, config)
            .await
            .map_err(GuardrailExecutionError::Runtime)?,
    ))
}

#[cfg(test)]
pub(crate) async fn run_input_guardrail_check<R, C>(
    runtime: &R,
    config: &C,
    messages: &[ChatMessage],
) -> Result<Option<GuardrailOutcome>, GuardrailExecutionError<R::Error>>
where
    R: GuardrailRuntime<C>,
{
    let payload =
        GuardrailCheckPayload::Input(input_guardrail_payload_from_chat_messages(messages)?);

    run_guardrail_check(runtime, config, payload).await
}

fn configured_guardrail_runtime_from_configs(
    guardrail: &GuardrailConfig,
) -> Result<Box<dyn ConfiguredGuardrailRuntime>, GatewayError> {
    match guardrail {
        GuardrailConfig::Bedrock(config) => Ok(Box::new(GuardrailRuntimeHandle::new(
            BedrockGuardrailRuntime::new(),
            config.clone(),
        ))),
    }
}

fn guardrail_role_from_chat_role(role: &str) -> Result<GuardrailRole, GuardrailBridgeError> {
    match role {
        "system" => Ok(GuardrailRole::System),
        "user" => Ok(GuardrailRole::User),
        "assistant" => Ok(GuardrailRole::Assistant),
        "tool" => Ok(GuardrailRole::Tool),
        other => Err(GuardrailBridgeError::UnsupportedRole(other.to_string())),
    }
}

fn chat_role_from_guardrail_role(role: GuardrailRole) -> &'static str {
    match role {
        GuardrailRole::System => "system",
        GuardrailRole::User => "user",
        GuardrailRole::Assistant => "assistant",
        GuardrailRole::Tool => "tool",
    }
}

fn guardrail_content_from_chat_content(content: &MessageContent) -> GuardrailMessageContent {
    match content {
        MessageContent::Text(text) => GuardrailMessageContent::Text(text.clone()),
        MessageContent::Parts(parts) => GuardrailMessageContent::Parts(
            parts
                .iter()
                .map(guardrail_content_part_from_chat_content_part)
                .collect(),
        ),
    }
}

fn chat_content_from_guardrail_content(content: &GuardrailMessageContent) -> MessageContent {
    match content {
        GuardrailMessageContent::Text(text) => MessageContent::Text(text.clone()),
        GuardrailMessageContent::Parts(parts) => MessageContent::Parts(
            parts
                .iter()
                .map(chat_content_part_from_guardrail_content_part)
                .collect(),
        ),
    }
}

fn guardrail_content_part_from_chat_content_part(part: &ContentPart) -> GuardrailContentPart {
    match part {
        ContentPart::Text { text } => GuardrailContentPart::Text { text: text.clone() },
        ContentPart::ImageUrl { image_url } => GuardrailContentPart::ImageUrl {
            image_url: GuardrailImageUrl {
                url: image_url.url.clone(),
                detail: image_url.detail.clone(),
            },
        },
    }
}

fn chat_content_part_from_guardrail_content_part(part: &GuardrailContentPart) -> ContentPart {
    match part {
        GuardrailContentPart::Text { text } => ContentPart::Text { text: text.clone() },
        GuardrailContentPart::ImageUrl { image_url } => ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: image_url.url.clone(),
                detail: image_url.detail.clone(),
            },
        },
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use aisix_guardrail::{
        guardrails::configs::BedrockGuardrailConfig,
        traits::{
            GuardrailCheckPayload, GuardrailContentPart, GuardrailMessage, GuardrailMessageContent,
            GuardrailMeta, GuardrailOutcome, GuardrailRole, GuardrailRuntime, GuardrailStage,
            GuardrailToolCall, InputGuardrailPayload, OutputGuardrailPayload,
        },
    };
    use async_trait::async_trait;
    use thiserror::Error;

    use super::{
        GuardrailBridgeError, chat_message_to_guardrail_message,
        configured_guardrail_runtime_from_configs, guardrail_message_to_chat_message,
        input_guardrail_payload_from_chat_messages, input_payload_from_check_payload,
        input_payload_to_chat_messages, output_guardrail_payload_from_chat_messages,
        output_payload_from_check_payload, output_payload_to_chat_messages,
        run_input_guardrail_check,
    };
    use crate::{
        config::entities::guardrails::GuardrailConfig,
        gateway::types::openai::{
            ChatMessage, ContentPart, FunctionCall, MessageContent, ToolCall,
        },
    };

    const INPUT_ONLY_STAGES: &[GuardrailStage] = &[GuardrailStage::Input];
    const OUTPUT_ONLY_STAGES: &[GuardrailStage] = &[GuardrailStage::Output];

    #[derive(Debug, Error)]
    #[error("mock guardrail runtime error")]
    struct MockGuardrailError;

    struct RecordingGuardrailRuntime {
        supported_stages: &'static [GuardrailStage],
        outcome: GuardrailOutcome,
        seen_payloads: Mutex<Vec<GuardrailCheckPayload>>,
    }

    impl RecordingGuardrailRuntime {
        fn new(supported_stages: &'static [GuardrailStage], outcome: GuardrailOutcome) -> Self {
            Self {
                supported_stages,
                outcome,
                seen_payloads: Mutex::new(Vec::new()),
            }
        }
    }

    impl GuardrailMeta for RecordingGuardrailRuntime {
        fn name(&self) -> &'static str {
            "recording"
        }

        fn supported_stages(&self) -> &'static [GuardrailStage] {
            self.supported_stages
        }
    }

    #[async_trait]
    impl GuardrailRuntime<()> for RecordingGuardrailRuntime {
        type Error = MockGuardrailError;

        async fn check(
            &self,
            payload: &GuardrailCheckPayload,
            _config: &(),
        ) -> Result<GuardrailOutcome, Self::Error> {
            self.seen_payloads.lock().unwrap().push(payload.clone());
            Ok(self.outcome.clone())
        }
    }

    #[test]
    fn chat_message_to_guardrail_message_preserves_core_fields() {
        let message = ChatMessage {
            role: "assistant".into(),
            content: Some(MessageContent::Text("hello".into())),
            name: Some("planner".into()),
            tool_calls: Some(vec![ToolCall {
                id: "call_weather_1".into(),
                r#type: "function".into(),
                function: FunctionCall {
                    name: "get_weather".into(),
                    arguments: r#"{"city":"Hangzhou"}"#.into(),
                },
            }]),
            tool_call_id: None,
        };

        let guardrail_message = chat_message_to_guardrail_message(&message).unwrap();

        assert_eq!(guardrail_message.role, GuardrailRole::Assistant);
        assert_eq!(
            guardrail_message.content,
            Some(GuardrailMessageContent::Text("hello".into()))
        );
        assert_eq!(guardrail_message.name.as_deref(), Some("planner"));
        assert_eq!(
            guardrail_message.tool_calls,
            Some(vec![GuardrailToolCall {
                id: "call_weather_1".into(),
                r#type: "function".into(),
                name: "get_weather".into(),
                arguments: r#"{"city":"Hangzhou"}"#.into(),
            }])
        );
    }

    #[test]
    fn guardrail_message_to_chat_message_round_trips_core_fields() {
        let message = GuardrailMessage {
            role: GuardrailRole::Assistant,
            content: Some(GuardrailMessageContent::Parts(vec![
                GuardrailContentPart::Text {
                    text: "describe this image".into(),
                },
                GuardrailContentPart::ImageUrl {
                    image_url: aisix_guardrail::traits::GuardrailImageUrl {
                        url: "https://example.com/cat.png".into(),
                        detail: Some("high".into()),
                    },
                },
            ])),
            name: Some("planner".into()),
            tool_calls: Some(vec![GuardrailToolCall {
                id: "call_weather_1".into(),
                r#type: "function".into(),
                name: "get_weather".into(),
                arguments: r#"{"city":"Hangzhou"}"#.into(),
            }]),
            tool_call_id: None,
        };

        let chat_message = guardrail_message_to_chat_message(&message).unwrap();

        assert_eq!(chat_message.role, "assistant");
        assert_eq!(chat_message.name.as_deref(), Some("planner"));
        match chat_message.content {
            Some(MessageContent::Parts(parts)) => {
                assert_eq!(parts.len(), 2);
                assert!(matches!(
                    &parts[0],
                    ContentPart::Text { text } if text == "describe this image"
                ));
                assert!(matches!(
                    &parts[1],
                    ContentPart::ImageUrl { image_url }
                        if image_url.url == "https://example.com/cat.png"
                        && image_url.detail.as_deref() == Some("high")
                ));
            }
            other => panic!("expected multipart content, got {other:?}"),
        }
    }

    #[test]
    fn input_guardrail_payload_from_chat_messages_builds_message_list() {
        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: Some(MessageContent::Text("be concise".into())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            ChatMessage {
                role: "tool".into(),
                content: Some(MessageContent::Text(r#"{"ok":true}"#.into())),
                name: None,
                tool_calls: None,
                tool_call_id: Some("call_weather_1".into()),
            },
        ];

        let payload = input_guardrail_payload_from_chat_messages(&messages).unwrap();

        assert_eq!(payload.messages.len(), 2);
        assert_eq!(payload.messages[0].role, GuardrailRole::System);
        assert_eq!(payload.messages[1].role, GuardrailRole::Tool);
        assert_eq!(
            payload.messages[1].tool_call_id.as_deref(),
            Some("call_weather_1")
        );
    }

    #[test]
    fn output_payload_round_trips_chat_messages() {
        let messages = vec![ChatMessage {
            role: "assistant".into(),
            content: Some(MessageContent::Text("hello".into())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];

        let payload = output_guardrail_payload_from_chat_messages(&messages).unwrap();
        let round_trip = output_payload_to_chat_messages(&payload).unwrap();

        assert_eq!(round_trip.len(), 1);
        assert_eq!(round_trip[0].role, "assistant");
        assert!(matches!(
            &round_trip[0].content,
            Some(MessageContent::Text(text)) if text == "hello"
        ));
    }

    #[test]
    fn check_payload_stage_extractors_reject_mismatches() {
        assert!(matches!(
            input_payload_from_check_payload(GuardrailCheckPayload::Output(
                OutputGuardrailPayload::default(),
            )),
            Err(GuardrailBridgeError::UnexpectedStage("input"))
        ));
        assert!(matches!(
            output_payload_from_check_payload(GuardrailCheckPayload::Input(
                InputGuardrailPayload::default(),
            )),
            Err(GuardrailBridgeError::UnexpectedStage("output"))
        ));
    }

    #[tokio::test]
    async fn run_input_guardrail_check_should_bridge_and_call_runtime() {
        let runtime = RecordingGuardrailRuntime::new(INPUT_ONLY_STAGES, GuardrailOutcome::Allow);
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: Some(MessageContent::Text("hello".into())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];

        let outcome = run_input_guardrail_check(&runtime, &(), &messages)
            .await
            .unwrap();

        assert_eq!(outcome, Some(GuardrailOutcome::Allow));
        assert_eq!(
            runtime.seen_payloads.lock().unwrap().as_slice(),
            &[GuardrailCheckPayload::Input(InputGuardrailPayload {
                messages: vec![GuardrailMessage {
                    role: GuardrailRole::User,
                    content: Some(GuardrailMessageContent::Text("hello".into())),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                }],
            })]
        );
    }

    #[tokio::test]
    async fn run_input_guardrail_check_should_skip_unsupported_stage() {
        let runtime = RecordingGuardrailRuntime::new(OUTPUT_ONLY_STAGES, GuardrailOutcome::Allow);
        let messages = vec![ChatMessage {
            role: "user".into(),
            content: Some(MessageContent::Text("hello".into())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];

        let outcome = run_input_guardrail_check(&runtime, &(), &messages)
            .await
            .unwrap();

        assert_eq!(outcome, None);
        assert!(runtime.seen_payloads.lock().unwrap().is_empty());
    }

    #[test]
    fn configured_guardrail_runtime_from_configs_builds_bedrock_runtime() {
        let runtime = configured_guardrail_runtime_from_configs(&GuardrailConfig::Bedrock(
            BedrockGuardrailConfig {
                identifier: "guardrail-123".into(),
                version: "1".into(),
                region: "us-east-1".into(),
                access_key_id: "AKIA123".into(),
                secret_access_key: "secret".into(),
                session_token: None,
                endpoint: None,
            },
        ))
        .unwrap();

        assert_eq!(runtime.name(), "bedrock");
        assert!(runtime.supports_stage(GuardrailStage::Input));
    }

    #[test]
    fn input_payload_to_chat_messages_round_trips() {
        let payload = InputGuardrailPayload {
            messages: vec![GuardrailMessage {
                role: GuardrailRole::Tool,
                content: Some(GuardrailMessageContent::Text(r#"{"ok":true}"#.into())),
                name: None,
                tool_calls: None,
                tool_call_id: Some("call_weather_1".into()),
            }],
        };

        let messages = input_payload_to_chat_messages(&payload).unwrap();

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "tool");
        assert!(matches!(
            &messages[0].content,
            Some(MessageContent::Text(text)) if text == r#"{"ok":true}"#
        ));
        assert_eq!(messages[0].tool_call_id.as_deref(), Some("call_weather_1"));
    }

    #[test]
    fn chat_message_to_guardrail_message_rejects_unsupported_roles() {
        let message = ChatMessage {
            role: "developer".into(),
            content: Some(MessageContent::Text("hello".into())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        };

        let error = chat_message_to_guardrail_message(&message).unwrap_err();

        assert_eq!(
            error,
            GuardrailBridgeError::UnsupportedRole("developer".into())
        );
    }
}
