use pretty_assertions::assert_eq;
use serde_json::{Value, json};

use super::{
    StreamOutputCollector, chunk_span_properties, request_span_properties, response_span_properties,
};
use crate::gateway::{
    providers::openai::OpenAIDef,
    types::{
        common::Usage,
        openai::responses::{
            ResponsesApiRequest, ResponsesApiResponse, ResponsesApiStreamEvent, ResponsesInput,
            ResponsesInputItem, ResponsesOutputContent, ResponsesOutputItem, ResponsesTool,
            ResponsesUsage,
        },
    },
};

fn property_value<'a>(properties: &'a [(String, String)], key: &str) -> Option<&'a str> {
    properties
        .iter()
        .find(|(property_key, _)| property_key == key)
        .map(|(_, value)| value.as_str())
}

#[test]
fn request_span_properties_include_messages_tools_and_session_fields() {
    let request = ResponsesApiRequest {
        model: "gpt-4.1".into(),
        input: ResponsesInput::Items(vec![
            ResponsesInputItem::Message {
                role: "user".into(),
                content: crate::gateway::types::openai::responses::ResponsesContent::Parts(vec![
                    crate::gateway::types::openai::responses::ResponsesContentPart::InputText {
                        text: "Describe this image".into(),
                    },
                    crate::gateway::types::openai::responses::ResponsesContentPart::InputImage {
                        image_url: Some("https://example.com/cat.png".into()),
                        file_id: None,
                        detail: None,
                    },
                ]),
            },
            ResponsesInputItem::FunctionCallOutput {
                call_id: "call_1".into(),
                output: "72F and sunny".into(),
            },
        ]),
        instructions: Some("Be concise".into()),
        max_output_tokens: Some(256),
        temperature: Some(0.2),
        top_p: Some(0.9),
        tools: Some(vec![
            ResponsesTool::Function {
                name: "get_weather".into(),
                description: Some("Get current weather".into()),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}}
                })),
                strict: Some(true),
            },
            ResponsesTool::FileSearch {
                vector_store_ids: vec!["vs_1".into()],
                max_num_results: Some(5),
            },
        ]),
        metadata: Some(json!({"user_id": "user-123"})),
        text: Some(
            crate::gateway::types::openai::responses::ResponseTextConfig {
                format: Some(json!({"type": "json_schema"})),
                verbosity: Some("low".into()),
            },
        ),
        previous_response_id: Some("resp_prev".into()),
        conversation: Some(
            crate::gateway::types::openai::responses::ConversationReference::Descriptor {
                id: "conv_123".into(),
            },
        ),
        stream: Some(true),
        store: Some(true),
        ..serde_json::from_value(json!({"model": "ignored", "input": "ignored"})).unwrap()
    };
    let provider = OpenAIDef;

    let properties = request_span_properties(&request, &provider, None);

    assert_eq!(property_value(&properties, "user.id"), Some("user-123"));
    assert_eq!(
        property_value(&properties, "gen_ai.request.max_tokens"),
        Some("256")
    );
    assert_eq!(
        property_value(&properties, "gen_ai.output.type"),
        Some("json")
    );
    assert_eq!(
        property_value(&properties, "aisix.responses.previous_response_id"),
        Some("resp_prev")
    );
    assert_eq!(
        property_value(&properties, "aisix.responses.conversation_id"),
        Some("conv_123")
    );
    assert_eq!(
        property_value(&properties, "llm.input_messages.0.message.role"),
        Some("system")
    );
    assert_eq!(
        property_value(
            &properties,
            "llm.input_messages.1.message.contents.1.message_content.image.image.url",
        ),
        Some("https://example.com/cat.png"),
    );
    assert_eq!(
        property_value(&properties, "llm.input_messages.2.message.role"),
        Some("tool")
    );
    assert_eq!(
        property_value(&properties, "llm.tools.0.tool.name"),
        Some("get_weather")
    );
    assert_eq!(
        property_value(&properties, "llm.tools.1.tool.name"),
        Some("file_search")
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
    assert_eq!(tool_definitions[1]["type"], "file_search");
}

#[test]
fn response_span_properties_include_output_messages_and_usage() {
    let response = ResponsesApiResponse {
        id: "resp_123".into(),
        object: "response".into(),
        created_at: 0,
        model: "gpt-4.1".into(),
        output: vec![
            ResponsesOutputItem::Message {
                id: "msg_1".into(),
                role: "assistant".into(),
                content: vec![ResponsesOutputContent::OutputText {
                    text: "Hello".into(),
                }],
                status: "completed".into(),
            },
            ResponsesOutputItem::FunctionCall {
                id: "fc_1".into(),
                call_id: "call_1".into(),
                name: "get_weather".into(),
                arguments: r#"{"city":"SF"}"#.into(),
                status: "completed".into(),
            },
        ],
        status: "completed".into(),
        usage: ResponsesUsage {
            input_tokens: 10,
            output_tokens: 8,
            total_tokens: 18,
        },
        metadata: None,
        previous_response_id: Some("resp_prev".into()),
    };

    let properties = response_span_properties(&response, &Usage::default());

    assert_eq!(
        property_value(&properties, "llm.output_messages.0.message.role"),
        Some("assistant")
    );
    assert_eq!(
        property_value(&properties, "llm.output_messages.0.message.content"),
        Some("Hello")
    );
    assert_eq!(
        property_value(
            &properties,
            "llm.output_messages.1.message.tool_calls.0.tool_call.function.name",
        ),
        Some("get_weather")
    );
    assert_eq!(
        property_value(&properties, "gen_ai.usage.input_tokens"),
        Some("10")
    );
    assert_eq!(
        property_value(&properties, "llm.token_count.total"),
        Some("18")
    );

    let output_messages: Value =
        serde_json::from_str(property_value(&properties, "gen_ai.output.messages").unwrap())
            .unwrap();
    assert_eq!(output_messages[0]["finish_reason"], "stop");
    assert_eq!(output_messages[1]["finish_reason"], "tool_calls");
}

#[test]
fn chunk_span_properties_include_completed_response_usage_and_finish_reason() {
    let event = ResponsesApiStreamEvent::ResponseCompleted {
        response: ResponsesApiResponse {
            id: "resp_123".into(),
            object: "response".into(),
            created_at: 0,
            model: "gpt-4.1".into(),
            output: vec![ResponsesOutputItem::FunctionCall {
                id: "fc_1".into(),
                call_id: "call_1".into(),
                name: "get_weather".into(),
                arguments: "{}".into(),
                status: "completed".into(),
            }],
            status: "completed".into(),
            usage: ResponsesUsage {
                input_tokens: 7,
                output_tokens: 9,
                total_tokens: 16,
            },
            metadata: None,
            previous_response_id: None,
        },
    };

    let properties = chunk_span_properties(&event);

    assert_eq!(
        property_value(&properties, "gen_ai.response.id"),
        Some("resp_123")
    );
    assert_eq!(
        property_value(&properties, "llm.finish_reason"),
        Some("tool_calls")
    );
    assert_eq!(
        property_value(&properties, "gen_ai.usage.output_tokens"),
        Some("9")
    );
}

#[test]
fn stream_output_collector_accumulates_events_into_output_messages() {
    let mut collector = StreamOutputCollector::default();

    collector.record_event(&ResponsesApiStreamEvent::OutputItemAdded {
        output_index: 0,
        item: ResponsesOutputItem::Message {
            id: "msg_1".into(),
            role: "assistant".into(),
            content: vec![],
            status: "in_progress".into(),
        },
    });
    collector.record_event(&ResponsesApiStreamEvent::OutputTextDelta {
        output_index: 0,
        content_index: 0,
        delta: "Hello".into(),
    });
    collector.record_event(&ResponsesApiStreamEvent::OutputItemAdded {
        output_index: 1,
        item: ResponsesOutputItem::FunctionCall {
            id: "fc_1".into(),
            call_id: "call_1".into(),
            name: "get_weather".into(),
            arguments: String::new(),
            status: "in_progress".into(),
        },
    });
    collector.record_event(&ResponsesApiStreamEvent::FunctionCallArgumentsDelta {
        output_index: 1,
        delta: r#"{"city":"SF"}"#.into(),
    });
    collector.record_event(&ResponsesApiStreamEvent::OutputItemDone {
        output_index: 0,
        item: ResponsesOutputItem::Message {
            id: "msg_1".into(),
            role: "assistant".into(),
            content: vec![ResponsesOutputContent::OutputText {
                text: "Hello".into(),
            }],
            status: "completed".into(),
        },
    });
    collector.record_event(&ResponsesApiStreamEvent::FunctionCallArgumentsDone {
        output_index: 1,
        arguments: r#"{"city":"SF"}"#.into(),
    });

    let properties = collector.output_message_span_properties();

    assert_eq!(
        property_value(&properties, "llm.output_messages.0.message.content"),
        Some("Hello")
    );
    assert_eq!(
        property_value(
            &properties,
            "llm.output_messages.1.message.tool_calls.0.tool_call.function.arguments",
        ),
        Some(r#"{"city":"SF"}"#),
    );

    let output_messages: Value =
        serde_json::from_str(property_value(&properties, "gen_ai.output.messages").unwrap())
            .unwrap();
    assert_eq!(output_messages[0]["finish_reason"], "stop");
    assert_eq!(output_messages[1]["parts"][0]["arguments"]["city"], "SF");
}
