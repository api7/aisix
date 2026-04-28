use serde_json::{Value, json};

use super::{StreamOutputCollector, request_span_properties, response_span_properties};
use crate::gateway::{
    providers::openai::OpenAIDef,
    types::{
        common::Usage,
        openai::{
            ChatCompletionChoice, ChatCompletionChunk, ChatCompletionChunkChoice,
            ChatCompletionChunkDelta, ChatCompletionRequest, ChatCompletionResponse,
            ChatCompletionUsage, ChatMessage, ChunkFunctionCall, ChunkToolCall,
            CompletionTokensDetails, ContentPart, FunctionCall, FunctionDefinition, ImageUrl,
            MessageContent, Tool,
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
fn request_span_properties_include_message_tool_and_user_attributes() {
    let mut request = ChatCompletionRequest {
        model: "gpt-4o".into(),
        messages: vec![
            ChatMessage {
                role: "system".into(),
                content: Some(MessageContent::Text("Be concise".into())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
            ChatMessage {
                role: "user".into(),
                content: Some(MessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "Describe this image".into(),
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: "https://example.com/cat.png".into(),
                            detail: None,
                        },
                    },
                ])),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            },
        ],
        tools: Some(vec![Tool {
            r#type: "function".into(),
            function: FunctionDefinition {
                name: "get_weather".into(),
                description: Some("Get current weather".into()),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}}
                })),
                strict: Some(true),
            },
        }]),
        user: Some("user-123".into()),
        ..Default::default()
    };
    request.extra.insert("top_k".into(), json!(5));
    let provider = OpenAIDef;

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
        property_value(
            &properties,
            "llm.input_messages.1.message.contents.1.message_content.image.image.url",
        ),
        Some("https://example.com/cat.png"),
    );
    assert_eq!(
        property_value(&properties, "llm.tools.0.tool.name"),
        Some("get_weather"),
    );

    let input_messages: Value =
        serde_json::from_str(property_value(&properties, "gen_ai.input.messages").unwrap())
            .unwrap();
    assert_eq!(input_messages[0]["role"], "system");
    assert_eq!(input_messages[1]["parts"][1]["type"], "uri");

    let tool_definitions: Value =
        serde_json::from_str(property_value(&properties, "gen_ai.tool.definitions").unwrap())
            .unwrap();
    assert_eq!(tool_definitions[0]["name"], "get_weather");
}

#[test]
fn response_span_properties_include_output_messages_and_reasoning_tokens() {
    let response = ChatCompletionResponse {
        id: "chatcmpl-1".into(),
        object: "chat.completion".into(),
        created: 0,
        model: "gpt-4o".into(),
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".into(),
                content: Some(MessageContent::Text("I'll check.".into())),
                name: None,
                tool_calls: Some(vec![crate::gateway::types::openai::ToolCall {
                    id: "call_1".into(),
                    r#type: "function".into(),
                    function: FunctionCall {
                        name: "get_weather".into(),
                        arguments: r#"{"city":"SF"}"#.into(),
                    },
                }]),
                tool_call_id: None,
            },
            finish_reason: Some("tool_calls".into()),
        }],
        usage: Some(ChatCompletionUsage {
            prompt_tokens: 9,
            completion_tokens: 7,
            total_tokens: 16,
            prompt_tokens_details: None,
            completion_tokens_details: Some(CompletionTokensDetails {
                reasoning_tokens: Some(3),
                audio_tokens: None,
            }),
        }),
        system_fingerprint: None,
    };
    let usage = Usage {
        input_tokens: Some(9),
        output_tokens: Some(7),
        total_tokens: Some(16),
        ..Default::default()
    };

    let properties = response_span_properties(&response, &usage);

    assert_eq!(
        property_value(&properties, "llm.output_messages.0.message.role"),
        Some("assistant")
    );
    assert_eq!(
        property_value(
            &properties,
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.name",
        ),
        Some("get_weather"),
    );
    assert_eq!(
        property_value(&properties, "llm.token_count.completion_details.reasoning"),
        Some("3")
    );

    let output_messages: Value =
        serde_json::from_str(property_value(&properties, "gen_ai.output.messages").unwrap())
            .unwrap();
    assert_eq!(output_messages[0]["finish_reason"], "tool_calls");
    assert_eq!(output_messages[0]["parts"][1]["type"], "tool_call");
}

#[test]
fn stream_output_collector_accumulates_chunks_into_output_messages() {
    let mut collector = StreamOutputCollector::default();

    collector.record_chunk(&ChatCompletionChunk {
        id: "chatcmpl-1".into(),
        object: "chat.completion.chunk".into(),
        created: 0,
        model: "gpt-4o".into(),
        choices: vec![ChatCompletionChunkChoice {
            index: 0,
            delta: ChatCompletionChunkDelta {
                role: Some("assistant".into()),
                content: Some("Hello ".into()),
                tool_calls: Some(vec![ChunkToolCall {
                    index: 0,
                    id: Some("call_1".into()),
                    r#type: Some("function".into()),
                    function: Some(ChunkFunctionCall {
                        name: Some("get_weather".into()),
                        arguments: Some(r#"{"city":""#.into()),
                    }),
                }]),
            },
            finish_reason: None,
        }],
        usage: None,
        system_fingerprint: None,
    });

    collector.record_chunk(&ChatCompletionChunk {
        id: "chatcmpl-1".into(),
        object: "chat.completion.chunk".into(),
        created: 0,
        model: "gpt-4o".into(),
        choices: vec![ChatCompletionChunkChoice {
            index: 0,
            delta: ChatCompletionChunkDelta {
                role: None,
                content: Some("world".into()),
                tool_calls: Some(vec![ChunkToolCall {
                    index: 0,
                    id: None,
                    r#type: None,
                    function: Some(ChunkFunctionCall {
                        name: None,
                        arguments: Some(r#"SF"}"#.into()),
                    }),
                }]),
            },
            finish_reason: Some("tool_calls".into()),
        }],
        usage: None,
        system_fingerprint: None,
    });

    let properties = collector.output_message_span_properties();

    assert_eq!(
        property_value(&properties, "llm.output_messages.0.message.content"),
        Some("Hello world")
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
