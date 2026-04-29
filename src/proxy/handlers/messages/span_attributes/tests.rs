use pretty_assertions::assert_eq;
use serde_json::{Value, json};

use super::{
    StreamOutputCollector, chunk_span_properties, request_span_properties, response_span_properties,
};
use crate::gateway::{
    providers::AnthropicDef,
    types::{
        anthropic::{
            AnthropicContent, AnthropicContentBlock, AnthropicMessage, AnthropicMessagesRequest,
            AnthropicMessagesResponse, AnthropicMetadata, AnthropicStreamEvent, AnthropicTool,
            AnthropicToolChoice, AnthropicUsage, ContentDelta, DeltaUsage, MessageDelta,
            MessageStartPayload, MessageStartUsage, SystemPrompt,
        },
        common::Usage,
    },
};

fn property_value<'a>(properties: &'a [(String, String)], key: &str) -> Option<&'a str> {
    properties
        .iter()
        .find(|(property_key, _)| property_key == key)
        .map(|(_, value)| value.as_str())
}

#[test]
fn request_span_properties_include_system_tool_and_user_attributes() {
    let request = AnthropicMessagesRequest {
        model: "claude-3-5-sonnet-20241022".into(),
        messages: vec![AnthropicMessage {
            role: "user".into(),
            content: AnthropicContent::Blocks(vec![
                AnthropicContentBlock::Text {
                    text: "What's the weather?".into(),
                    cache_control: None,
                },
                AnthropicContentBlock::ToolResult {
                    tool_use_id: "tool_1".into(),
                    content: Some(AnthropicContent::Text("72F and sunny".into())),
                    is_error: None,
                    cache_control: None,
                },
            ]),
        }],
        max_tokens: 1024,
        cache_control: None,
        system: Some(SystemPrompt::Text("You are helpful.".into())),
        temperature: Some(0.2),
        top_p: Some(0.9),
        top_k: Some(5),
        stop_sequences: Some(vec!["DONE".into()]),
        stream: Some(true),
        metadata: Some(AnthropicMetadata {
            user_id: Some("user-123".into()),
        }),
        tools: Some(vec![AnthropicTool {
            name: "get_weather".into(),
            description: Some("Get current weather".into()),
            input_schema: json!({
                "type": "object",
                "properties": {"city": {"type": "string"}}
            }),
        }]),
        tool_choice: Some(AnthropicToolChoice::Auto),
    };
    let provider = AnthropicDef;

    let properties = request_span_properties(&request, &provider, None);

    assert_eq!(property_value(&properties, "user.id"), Some("user-123"));
    assert_eq!(
        property_value(&properties, "gen_ai.request.top_k"),
        Some("5")
    );
    assert_eq!(
        property_value(&properties, "llm.input_messages.0.message.role"),
        Some("system")
    );
    assert_eq!(
        property_value(&properties, "llm.input_messages.1.message.role"),
        Some("user")
    );
    assert_eq!(
        property_value(&properties, "llm.input_messages.2.message.role"),
        Some("tool")
    );
    assert_eq!(
        property_value(&properties, "llm.input_messages.2.message.tool_call_id"),
        Some("tool_1")
    );
    assert_eq!(
        property_value(&properties, "llm.tools.0.tool.name"),
        Some("get_weather")
    );

    let input_messages: Value =
        serde_json::from_str(property_value(&properties, "gen_ai.input.messages").unwrap())
            .unwrap();
    assert_eq!(input_messages[0]["role"], "system");
    assert_eq!(input_messages[2]["role"], "tool");
    assert_eq!(input_messages[2]["parts"][0]["type"], "tool_call_response");

    let tool_definitions: Value =
        serde_json::from_str(property_value(&properties, "gen_ai.tool.definitions").unwrap())
            .unwrap();
    assert_eq!(tool_definitions[0]["name"], "get_weather");
}

#[test]
fn response_span_properties_include_output_messages_and_usage() {
    let response = AnthropicMessagesResponse {
        id: "msg_123".into(),
        r#type: "message".into(),
        role: "assistant".into(),
        content: vec![
            AnthropicContentBlock::Text {
                text: "Let me check.".into(),
                cache_control: None,
            },
            AnthropicContentBlock::ToolUse {
                id: "tool_1".into(),
                name: "get_weather".into(),
                input: json!({"city": "SF"}),
                cache_control: None,
            },
        ],
        model: "claude-3-5-sonnet-20241022".into(),
        stop_reason: Some("tool_use".into()),
        stop_sequence: None,
        usage: AnthropicUsage {
            input_tokens: 10,
            output_tokens: 20,
            cache_creation_input_tokens: 3,
            cache_read_input_tokens: 2,
            cache_creation: None,
        },
    };

    let properties = response_span_properties(&response, &Usage::default());

    assert_eq!(
        property_value(&properties, "llm.output_messages.0.message.role"),
        Some("assistant")
    );
    assert_eq!(
        property_value(
            &properties,
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.name",
        ),
        Some("get_weather")
    );
    assert_eq!(
        property_value(&properties, "gen_ai.usage.input_tokens"),
        Some("15")
    );
    assert_eq!(
        property_value(&properties, "llm.token_count.total"),
        Some("35")
    );

    let output_messages: Value =
        serde_json::from_str(property_value(&properties, "gen_ai.output.messages").unwrap())
            .unwrap();
    assert_eq!(output_messages[0]["finish_reason"], "tool_use");
    assert_eq!(output_messages[0]["parts"][1]["type"], "tool_call");
}

#[test]
fn chunk_span_properties_include_message_start_and_stop_reason() {
    let message_start = AnthropicStreamEvent::MessageStart {
        message: MessageStartPayload {
            id: "msg_123".into(),
            r#type: "message".into(),
            role: "assistant".into(),
            model: "claude-3-5-sonnet-20241022".into(),
            usage: MessageStartUsage {
                input_tokens: Some(7),
                output_tokens: Some(1),
                cache_creation_input_tokens: Some(3),
                cache_read_input_tokens: Some(2),
                cache_creation: None,
            },
        },
    };
    let message_delta = AnthropicStreamEvent::MessageDelta {
        delta: MessageDelta {
            stop_reason: Some("end_turn".into()),
            stop_sequence: None,
        },
        usage: DeltaUsage {
            output_tokens: Some(11),
            input_tokens: Some(7),
            cache_creation_input_tokens: None,
            cache_read_input_tokens: None,
        },
    };

    let start_properties = chunk_span_properties(&message_start);
    let delta_properties = chunk_span_properties(&message_delta);

    assert_eq!(
        property_value(&start_properties, "gen_ai.response.id"),
        Some("msg_123")
    );
    assert_eq!(
        property_value(&start_properties, "gen_ai.usage.input_tokens"),
        Some("12")
    );
    assert_eq!(
        property_value(&delta_properties, "llm.finish_reason"),
        Some("end_turn")
    );
    assert_eq!(
        property_value(&delta_properties, "gen_ai.usage.output_tokens"),
        Some("11")
    );
}

#[test]
fn stream_output_collector_accumulates_events_into_output_messages() {
    let mut collector = StreamOutputCollector::default();

    collector.record_event(&AnthropicStreamEvent::MessageStart {
        message: MessageStartPayload {
            id: "msg_123".into(),
            r#type: "message".into(),
            role: "assistant".into(),
            model: "claude-3-5-sonnet-20241022".into(),
            usage: MessageStartUsage::default(),
        },
    });
    collector.record_event(&AnthropicStreamEvent::ContentBlockStart {
        index: 0,
        content_block: AnthropicContentBlock::Text {
            text: String::new(),
            cache_control: None,
        },
    });
    collector.record_event(&AnthropicStreamEvent::ContentBlockDelta {
        index: 0,
        delta: ContentDelta::TextDelta {
            text: "Hello".into(),
        },
    });
    collector.record_event(&AnthropicStreamEvent::ContentBlockStart {
        index: 1,
        content_block: AnthropicContentBlock::ToolUse {
            id: "tool_1".into(),
            name: "get_weather".into(),
            input: json!({}),
            cache_control: None,
        },
    });
    collector.record_event(&AnthropicStreamEvent::ContentBlockDelta {
        index: 1,
        delta: ContentDelta::InputJsonDelta {
            partial_json: r#"{"city":"SF"}"#.into(),
        },
    });
    collector.record_event(&AnthropicStreamEvent::MessageDelta {
        delta: MessageDelta {
            stop_reason: Some("tool_use".into()),
            stop_sequence: None,
        },
        usage: DeltaUsage::default(),
    });

    let properties = collector.output_message_span_properties();

    assert_eq!(
        property_value(&properties, "llm.output_messages.0.message.content"),
        Some("Hello")
    );
    assert_eq!(
        property_value(
            &properties,
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments",
        ),
        Some(r#"{"city":"SF"}"#),
    );

    let output_messages: Value =
        serde_json::from_str(property_value(&properties, "gen_ai.output.messages").unwrap())
            .unwrap();
    assert_eq!(output_messages[0]["parts"][1]["arguments"]["city"], "SF");
}
