use serde::Serialize;
use serde_json::{Map, Value};

use crate::gateway::types::openai::{
    ChatCompletionResponse, ChatMessage, ContentPart, MessageContent, Tool,
};

#[derive(Clone)]
pub(super) enum MessageContentView {
    Text(String),
    Parts(Vec<ContentPartView>),
}

#[derive(Clone)]
pub(super) enum ContentPartView {
    Text(String),
    ImageUrl { url: String },
}

#[derive(Clone)]
pub(super) struct ToolCallView {
    pub(super) id: Option<String>,
    pub(super) name: String,
    pub(super) arguments: String,
}

#[derive(Clone)]
pub(super) struct MessageView {
    pub(super) role: String,
    pub(super) content: Option<MessageContentView>,
    pub(super) name: Option<String>,
    pub(super) tool_calls: Vec<ToolCallView>,
    pub(super) tool_call_id: Option<String>,
}

#[derive(Clone)]
pub(super) struct OutputMessageView {
    pub(super) message: MessageView,
    pub(super) finish_reason: Option<String>,
}

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

pub(super) fn gen_ai_input_messages_json(messages: &[MessageView]) -> Option<String> {
    if messages.is_empty() {
        return None;
    }

    let values: Vec<Value> = messages
        .iter()
        .map(|message| {
            let mut value = Map::new();
            value.insert("role".into(), Value::String(message.role.clone()));
            value.insert("parts".into(), Value::Array(gen_ai_message_parts(message)));

            if let Some(name) = &message.name {
                value.insert("name".into(), Value::String(name.clone()));
            }

            Value::Object(value)
        })
        .collect();

    serialize_to_json_string(&values)
}

pub(super) fn gen_ai_output_messages_json(messages: &[OutputMessageView]) -> Option<String> {
    if messages.is_empty() {
        return None;
    }

    let values: Vec<Value> = messages
        .iter()
        .map(|message| {
            let mut value = Map::new();
            value.insert("role".into(), Value::String(message.message.role.clone()));
            value.insert(
                "parts".into(),
                Value::Array(gen_ai_message_parts(&message.message)),
            );
            value.insert(
                "finish_reason".into(),
                Value::String(
                    message
                        .finish_reason
                        .clone()
                        .unwrap_or_else(|| "unknown".into()),
                ),
            );

            if let Some(name) = &message.message.name {
                value.insert("name".into(), Value::String(name.clone()));
            }

            Value::Object(value)
        })
        .collect();

    serialize_to_json_string(&values)
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

pub(super) fn append_openinference_message_properties(
    properties: &mut Vec<(String, String)>,
    prefix: &str,
    messages: &[MessageView],
) {
    for (message_index, message) in messages.iter().enumerate() {
        let prefix = format!("{prefix}.{message_index}.message");
        properties.push((format!("{prefix}.role"), message.role.clone()));

        if let Some(name) = &message.name {
            properties.push((format!("{prefix}.name"), name.clone()));
        }

        if let Some(tool_call_id) = &message.tool_call_id {
            properties.push((format!("{prefix}.tool_call_id"), tool_call_id.clone()));
        }

        match &message.content {
            Some(MessageContentView::Text(text)) if !text.is_empty() => {
                properties.push((format!("{prefix}.content"), text.clone()));
            }
            Some(MessageContentView::Parts(parts)) => {
                for (content_index, part) in parts.iter().enumerate() {
                    let part_prefix = format!("{prefix}.contents.{content_index}.message_content");
                    match part {
                        ContentPartView::Text(text) => {
                            properties.push((format!("{part_prefix}.type"), "text".into()));
                            properties.push((format!("{part_prefix}.text"), text.clone()));
                        }
                        ContentPartView::ImageUrl { url } => {
                            properties.push((format!("{part_prefix}.type"), "image".into()));
                            properties
                                .push((format!("{part_prefix}.image.image.url"), url.clone()));
                        }
                    }
                }
            }
            _ => {}
        }

        for (tool_call_index, tool_call) in message.tool_calls.iter().enumerate() {
            let tool_call_prefix = format!("{prefix}.tool_calls.{tool_call_index}.tool_call");

            if let Some(id) = &tool_call.id {
                properties.push((format!("{tool_call_prefix}.id"), id.clone()));
            }

            properties.push((
                format!("{tool_call_prefix}.function.name"),
                tool_call.name.clone(),
            ));

            if !tool_call.arguments.is_empty() {
                properties.push((
                    format!("{tool_call_prefix}.function.arguments"),
                    tool_call.arguments.clone(),
                ));
            }
        }
    }
}

pub(super) fn append_openinference_output_message_properties(
    properties: &mut Vec<(String, String)>,
    prefix: &str,
    messages: &[OutputMessageView],
) {
    let message_views: Vec<_> = messages
        .iter()
        .map(|message| message.message.clone())
        .collect();
    append_openinference_message_properties(properties, prefix, &message_views);
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

fn serialize_to_json_string(value: &impl Serialize) -> Option<String> {
    serde_json::to_string(value).ok()
}

fn parse_json_or_string(raw: &str) -> Value {
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
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

fn gen_ai_message_parts(message: &MessageView) -> Vec<Value> {
    if message.role == "tool" {
        let mut part = Map::new();
        part.insert("type".into(), Value::String("tool_call_response".into()));

        if let Some(tool_call_id) = &message.tool_call_id {
            part.insert("id".into(), Value::String(tool_call_id.clone()));
        }

        let response = match &message.content {
            Some(MessageContentView::Text(text)) => parse_json_or_string(text),
            Some(MessageContentView::Parts(parts)) => {
                Value::Array(parts.iter().map(gen_ai_content_part_value).collect())
            }
            None => Value::Null,
        };
        part.insert("response".into(), response);

        return vec![Value::Object(part)];
    }

    let mut parts = Vec::new();

    match &message.content {
        Some(MessageContentView::Text(text)) if !text.is_empty() => {
            parts.push(gen_ai_text_part(text));
        }
        Some(MessageContentView::Parts(content_parts)) => {
            parts.extend(content_parts.iter().map(gen_ai_content_part_value));
        }
        _ => {}
    }

    for tool_call in &message.tool_calls {
        let mut part = Map::new();
        part.insert("type".into(), Value::String("tool_call".into()));

        if let Some(id) = &tool_call.id {
            part.insert("id".into(), Value::String(id.clone()));
        }

        part.insert("name".into(), Value::String(tool_call.name.clone()));

        if !tool_call.arguments.is_empty() {
            part.insert(
                "arguments".into(),
                parse_json_or_string(&tool_call.arguments),
            );
        }

        parts.push(Value::Object(part));
    }

    parts
}

fn gen_ai_text_part(text: &str) -> Value {
    let mut part = Map::new();
    part.insert("type".into(), Value::String("text".into()));
    part.insert("content".into(), Value::String(text.to_string()));
    Value::Object(part)
}

fn gen_ai_content_part_value(part: &ContentPartView) -> Value {
    match part {
        ContentPartView::Text(text) => gen_ai_text_part(text),
        ContentPartView::ImageUrl { url } => {
            let mut part = Map::new();
            part.insert("type".into(), Value::String("uri".into()));
            part.insert("modality".into(), Value::String("image".into()));
            part.insert("uri".into(), Value::String(url.clone()));
            Value::Object(part)
        }
    }
}
