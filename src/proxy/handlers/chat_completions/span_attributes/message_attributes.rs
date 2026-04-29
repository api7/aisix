use serde_json::{Map, Value};

pub(super) use crate::proxy::utils::trace::span_message_attributes::{
    ContentPartView, MessageContentView, MessageView, OutputMessageView, ToolCallView,
    append_openinference_message_properties, append_openinference_output_message_properties,
    gen_ai_input_messages_json, gen_ai_output_messages_json,
};
use crate::{
    gateway::types::openai::{
        ChatCompletionResponse, ChatMessage, ContentPart, MessageContent, Tool,
    },
    proxy::utils::trace::span_message_attributes::serialize_to_json_string,
};

pub(super) fn message_view_from_chat_message(message: &ChatMessage) -> MessageView {
    MessageView {
        role: message.role.clone(),
        content: message
            .content
            .as_ref()
            .map(message_content_view_from_message_content),
        name: message.name.clone(),
        tool_calls: message
            .tool_calls
            .as_deref()
            .unwrap_or_default()
            .iter()
            .map(|tool_call| ToolCallView {
                id: Some(tool_call.id.clone()),
                name: tool_call.function.name.clone(),
                arguments: tool_call.function.arguments.clone(),
            })
            .collect(),
        tool_call_id: message.tool_call_id.clone(),
    }
}

pub(super) fn response_output_message_views(
    response: &ChatCompletionResponse,
) -> Vec<OutputMessageView> {
    response
        .choices
        .iter()
        .map(|choice| OutputMessageView {
            message: message_view_from_chat_message(&choice.message),
            finish_reason: choice.finish_reason.clone(),
        })
        .collect()
}

pub(super) fn gen_ai_tool_definitions_json(tools: &[Tool]) -> Option<String> {
    if tools.is_empty() {
        return None;
    }

    let values: Vec<Value> = tools
        .iter()
        .map(|tool| {
            let mut value = Map::new();
            value.insert("type".into(), Value::String(tool.r#type.clone()));
            value.insert("name".into(), Value::String(tool.function.name.clone()));

            if let Some(description) = &tool.function.description {
                value.insert("description".into(), Value::String(description.clone()));
            }

            if let Some(parameters) = &tool.function.parameters {
                value.insert("parameters".into(), parameters.clone());
            }

            Value::Object(value)
        })
        .collect();

    serialize_to_json_string(&values)
}

pub(super) fn append_openinference_tool_properties(
    properties: &mut Vec<(String, String)>,
    tools: &[Tool],
) {
    for (tool_index, tool) in tools.iter().enumerate() {
        let prefix = format!("llm.tools.{tool_index}.tool");
        properties.push((format!("{prefix}.name"), tool.function.name.clone()));

        if let Some(description) = &tool.function.description {
            properties.push((format!("{prefix}.description"), description.clone()));
        }

        if let Some(parameters) = &tool.function.parameters {
            if let Some(value) = serialize_to_json_string(parameters) {
                properties.push((format!("{prefix}.parameters"), value));
            }
        }

        if let Some(value) = serialize_to_json_string(tool) {
            properties.push((format!("{prefix}.json_schema"), value));
        }
    }
}

fn content_part_view_from_content_part(part: &ContentPart) -> ContentPartView {
    match part {
        ContentPart::Text { text } => ContentPartView::Text(text.clone()),
        ContentPart::ImageUrl { image_url } => ContentPartView::ImageUrl {
            url: image_url.url.clone(),
        },
    }
}

fn message_content_view_from_message_content(content: &MessageContent) -> MessageContentView {
    match content {
        MessageContent::Text(text) => MessageContentView::Text(text.clone()),
        MessageContent::Parts(parts) => MessageContentView::Parts(
            parts
                .iter()
                .map(content_part_view_from_content_part)
                .collect(),
        ),
    }
}
