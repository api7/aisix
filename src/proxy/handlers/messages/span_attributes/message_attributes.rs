use serde_json::{Map, Value};

pub(super) use crate::proxy::utils::trace::span_message_attributes::{
    ContentPartView, MessageContentView, MessageView, OutputMessageView, ToolCallView,
    append_openinference_message_properties, append_openinference_output_message_properties,
    gen_ai_input_messages_json, gen_ai_output_messages_json,
    message_content_view_from_content_parts,
};
use crate::{
    gateway::types::anthropic::{
        AnthropicContent, AnthropicContentBlock, AnthropicMessage, AnthropicMessagesRequest,
        AnthropicMessagesResponse, AnthropicTool, ImageSource, SystemPrompt,
    },
    proxy::utils::trace::span_message_attributes::serialize_to_json_string,
};

pub(super) fn request_input_message_views(request: &AnthropicMessagesRequest) -> Vec<MessageView> {
    let mut messages = system_prompt_message_views(request.system.as_ref());

    for message in &request.messages {
        messages.extend(message_views_from_anthropic_message(message));
    }

    messages
}

pub(super) fn response_output_message_views(
    response: &AnthropicMessagesResponse,
) -> Vec<OutputMessageView> {
    vec![OutputMessageView {
        message: message_view_from_blocks(&response.role, &response.content),
        finish_reason: response.stop_reason.clone(),
    }]
}

pub(super) fn gen_ai_tool_definitions_json(tools: &[AnthropicTool]) -> Option<String> {
    if tools.is_empty() {
        return None;
    }

    let values: Vec<Value> = tools.iter().map(anthropic_tool_to_json).collect();

    serialize_to_json_string(&values)
}

pub(super) fn append_openinference_tool_properties(
    properties: &mut Vec<(String, String)>,
    tools: &[AnthropicTool],
) {
    for (tool_index, tool) in tools.iter().enumerate() {
        let prefix = format!("llm.tools.{tool_index}.tool");
        properties.push((format!("{prefix}.name"), tool.name.clone()));

        if let Some(description) = &tool.description {
            properties.push((format!("{prefix}.description"), description.clone()));
        }

        if let Some(value) = serialize_to_json_string(&tool.input_schema) {
            properties.push((format!("{prefix}.parameters"), value));
        }

        if let Some(value) = serialize_to_json_string(&anthropic_tool_to_json(tool)) {
            properties.push((format!("{prefix}.json_schema"), value));
        }
    }
}

pub(super) fn message_view_from_blocks(
    role: &str,
    blocks: &[AnthropicContentBlock],
) -> MessageView {
    MessageView {
        role: role.to_string(),
        content: message_content_view_from_content_parts(content_parts_from_blocks(blocks)),
        name: None,
        tool_calls: tool_calls_from_blocks(blocks),
        tool_call_id: None,
    }
}

pub(super) fn image_source_to_data_url(source: &ImageSource) -> String {
    if source.r#type == "base64" {
        format!("data:{};base64,{}", source.media_type, source.data)
    } else {
        source.data.clone()
    }
}

fn anthropic_tool_to_json(tool: &AnthropicTool) -> Value {
    let mut value = Map::new();
    value.insert("type".into(), Value::String("function".into()));
    value.insert("name".into(), Value::String(tool.name.clone()));

    if let Some(description) = &tool.description {
        value.insert("description".into(), Value::String(description.clone()));
    }

    value.insert("parameters".into(), tool.input_schema.clone());
    Value::Object(value)
}

fn system_prompt_message_views(system: Option<&SystemPrompt>) -> Vec<MessageView> {
    match system {
        None => vec![],
        Some(SystemPrompt::Text(text)) => vec![MessageView {
            role: "system".into(),
            content: Some(MessageContentView::Text(text.clone())),
            name: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        }],
        Some(SystemPrompt::Blocks(blocks)) => blocks
            .iter()
            .map(|block| MessageView {
                role: "system".into(),
                content: Some(MessageContentView::Text(block.text.clone())),
                name: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
            })
            .collect(),
    }
}

fn message_views_from_anthropic_message(message: &AnthropicMessage) -> Vec<MessageView> {
    match &message.content {
        AnthropicContent::Text(text) => vec![MessageView {
            role: message.role.clone(),
            content: Some(MessageContentView::Text(text.clone())),
            name: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        }],
        AnthropicContent::Blocks(blocks) => {
            if message.role == "user" {
                return user_message_views_from_blocks(blocks);
            }

            vec![message_view_from_blocks(&message.role, blocks)]
        }
    }
}

fn user_message_views_from_blocks(blocks: &[AnthropicContentBlock]) -> Vec<MessageView> {
    let mut messages = Vec::new();
    let mut pending_blocks = Vec::new();

    for block in blocks {
        match block {
            AnthropicContentBlock::Text { .. }
            | AnthropicContentBlock::Image { .. }
            | AnthropicContentBlock::ToolUse { .. } => pending_blocks.push(block.clone()),
            AnthropicContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } => {
                if let Some(message) = message_view_from_pending_user_blocks(&pending_blocks) {
                    messages.push(message);
                    pending_blocks.clear();
                }

                messages.push(MessageView {
                    role: "tool".into(),
                    content: anthropic_optional_content_to_message_content_view(content.as_ref()),
                    name: None,
                    tool_calls: Vec::new(),
                    tool_call_id: Some(tool_use_id.clone()),
                });
            }
        }
    }

    if let Some(message) = message_view_from_pending_user_blocks(&pending_blocks) {
        messages.push(message);
    }

    if messages.is_empty() {
        messages.push(MessageView {
            role: "user".into(),
            content: Some(MessageContentView::Text(String::new())),
            name: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        });
    }

    messages
}

fn message_view_from_pending_user_blocks(blocks: &[AnthropicContentBlock]) -> Option<MessageView> {
    if blocks.is_empty() {
        return None;
    }

    Some(message_view_from_blocks("user", blocks))
}

fn anthropic_optional_content_to_message_content_view(
    content: Option<&AnthropicContent>,
) -> Option<MessageContentView> {
    content.and_then(anthropic_content_to_message_content_view)
}

fn anthropic_content_to_message_content_view(
    content: &AnthropicContent,
) -> Option<MessageContentView> {
    match content {
        AnthropicContent::Text(text) => Some(MessageContentView::Text(text.clone())),
        AnthropicContent::Blocks(blocks) => {
            message_content_view_from_content_parts(content_parts_from_blocks(blocks))
        }
    }
}

fn content_parts_from_blocks(blocks: &[AnthropicContentBlock]) -> Vec<ContentPartView> {
    blocks
        .iter()
        .filter_map(|block| match block {
            AnthropicContentBlock::Text { text, .. } => Some(ContentPartView::Text(text.clone())),
            AnthropicContentBlock::Image { source, .. } => Some(ContentPartView::ImageUrl {
                url: image_source_to_data_url(source),
            }),
            AnthropicContentBlock::ToolUse { .. } | AnthropicContentBlock::ToolResult { .. } => {
                None
            }
        })
        .collect()
}

fn tool_calls_from_blocks(blocks: &[AnthropicContentBlock]) -> Vec<ToolCallView> {
    blocks
        .iter()
        .filter_map(|block| match block {
            AnthropicContentBlock::ToolUse {
                id, name, input, ..
            } => Some(ToolCallView {
                id: Some(id.clone()),
                name: name.clone(),
                arguments: input.to_string(),
            }),
            _ => None,
        })
        .collect()
}
