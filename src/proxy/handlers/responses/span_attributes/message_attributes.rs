use serde_json::{Map, Value};

pub(super) use crate::proxy::utils::trace::span_message_attributes::{
    ContentPartView, MessageContentView, MessageView, OutputMessageView, ToolCallView,
    append_openinference_message_properties, append_openinference_output_message_properties,
    gen_ai_input_messages_json, gen_ai_output_messages_json,
    message_content_view_from_content_parts,
};
use crate::{
    gateway::types::openai::responses::{
        ResponsesApiRequest, ResponsesApiResponse, ResponsesContent, ResponsesContentPart,
        ResponsesInput, ResponsesInputItem, ResponsesOutputContent, ResponsesOutputItem,
        ResponsesTool,
    },
    proxy::utils::trace::span_message_attributes::serialize_to_json_string,
};

pub(super) fn request_input_message_views(request: &ResponsesApiRequest) -> Vec<MessageView> {
    let mut messages = Vec::new();

    if let Some(instructions) = request
        .instructions
        .as_ref()
        .filter(|instructions| !instructions.is_empty())
    {
        messages.push(MessageView {
            role: "system".into(),
            content: Some(MessageContentView::Text(instructions.clone())),
            name: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        });
    }

    match &request.input {
        ResponsesInput::Text(text) if !text.is_empty() => messages.push(MessageView {
            role: "user".into(),
            content: Some(MessageContentView::Text(text.clone())),
            name: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        }),
        ResponsesInput::Items(items) => {
            messages.extend(items.iter().filter_map(input_item_message_view))
        }
        ResponsesInput::Text(_) => {}
    }

    messages
}

pub(super) fn response_output_message_views(
    response: &ResponsesApiResponse,
) -> Vec<OutputMessageView> {
    output_message_views_from_output_items(&response.output)
}

pub(super) fn output_message_views_from_output_items(
    items: &[ResponsesOutputItem],
) -> Vec<OutputMessageView> {
    items
        .iter()
        .filter_map(output_message_view_from_output_item)
        .collect()
}

pub(super) fn gen_ai_tool_definitions_json(tools: &[ResponsesTool]) -> Option<String> {
    if tools.is_empty() {
        return None;
    }

    let values: Vec<_> = tools.iter().map(tool_definition_value).collect();
    serialize_to_json_string(&values)
}

pub(super) fn append_openinference_tool_properties(
    properties: &mut Vec<(String, String)>,
    tools: &[ResponsesTool],
) {
    for (tool_index, tool) in tools.iter().enumerate() {
        let prefix = format!("llm.tools.{tool_index}.tool");
        properties.push((format!("{prefix}.name"), tool_name(tool).to_string()));

        if let ResponsesTool::Function {
            description,
            parameters,
            ..
        } = tool
        {
            if let Some(description) = description {
                properties.push((format!("{prefix}.description"), description.clone()));
            }

            if let Some(parameters) = parameters
                && let Some(value) = serialize_to_json_string(parameters)
            {
                properties.push((format!("{prefix}.parameters"), value));
            }
        }

        if let Some(value) = serialize_to_json_string(tool) {
            properties.push((format!("{prefix}.json_schema"), value));
        }
    }
}

pub(super) fn output_item_finish_reason(item: &ResponsesOutputItem) -> Option<String> {
    match item {
        ResponsesOutputItem::Message { status, .. } if status == "completed" => Some("stop".into()),
        ResponsesOutputItem::FunctionCall { status, .. } if status == "completed" => {
            Some("tool_calls".into())
        }
        _ => None,
    }
}

fn input_item_message_view(item: &ResponsesInputItem) -> Option<MessageView> {
    match item {
        ResponsesInputItem::Message { role, content } => Some(MessageView {
            role: role.clone(),
            content: message_content_view_from_responses_content(content),
            name: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
        }),
        ResponsesInputItem::FunctionCallOutput { call_id, output } => Some(MessageView {
            role: "tool".into(),
            content: (!output.is_empty()).then(|| MessageContentView::Text(output.clone())),
            name: None,
            tool_calls: Vec::new(),
            tool_call_id: Some(call_id.clone()),
        }),
    }
}

fn message_content_view_from_responses_content(
    content: &ResponsesContent,
) -> Option<MessageContentView> {
    match content {
        ResponsesContent::Text(text) => {
            (!text.is_empty()).then(|| MessageContentView::Text(text.clone()))
        }
        ResponsesContent::Parts(parts) => {
            let parts = parts
                .iter()
                .filter_map(content_part_view_from_responses_part)
                .collect();
            message_content_view_from_content_parts(parts)
        }
    }
}

fn content_part_view_from_responses_part(part: &ResponsesContentPart) -> Option<ContentPartView> {
    match part {
        ResponsesContentPart::InputText { text } => {
            (!text.is_empty()).then(|| ContentPartView::Text(text.clone()))
        }
        ResponsesContentPart::InputImage {
            image_url, file_id, ..
        } => image_url
            .clone()
            .or_else(|| {
                file_id
                    .as_ref()
                    .map(|file_id| format!("openai://file/{file_id}"))
            })
            .map(|url| ContentPartView::ImageUrl { url }),
    }
}

fn output_message_view_from_output_item(item: &ResponsesOutputItem) -> Option<OutputMessageView> {
    match item {
        ResponsesOutputItem::Message { role, content, .. } => Some(OutputMessageView {
            message: MessageView {
                role: role.clone(),
                content: message_content_view_from_responses_output_content(content),
                name: None,
                tool_calls: Vec::new(),
                tool_call_id: None,
            },
            finish_reason: output_item_finish_reason(item),
        }),
        ResponsesOutputItem::FunctionCall {
            id,
            call_id,
            name,
            arguments,
            ..
        } => Some(OutputMessageView {
            message: MessageView {
                role: "assistant".into(),
                content: None,
                name: None,
                tool_calls: vec![ToolCallView {
                    id: Some(if call_id.is_empty() {
                        id.clone()
                    } else {
                        call_id.clone()
                    }),
                    name: name.clone(),
                    arguments: arguments.clone(),
                }],
                tool_call_id: None,
            },
            finish_reason: output_item_finish_reason(item),
        }),
    }
}

fn message_content_view_from_responses_output_content(
    content: &[ResponsesOutputContent],
) -> Option<MessageContentView> {
    let parts: Vec<_> = content
        .iter()
        .filter_map(|part| match part {
            ResponsesOutputContent::OutputText { text } => {
                (!text.is_empty()).then(|| ContentPartView::Text(text.clone()))
            }
        })
        .collect();

    message_content_view_from_content_parts(parts)
}

fn tool_name(tool: &ResponsesTool) -> &str {
    match tool {
        ResponsesTool::Function { name, .. } => name,
        ResponsesTool::WebSearch { .. } => "web_search_preview",
        ResponsesTool::FileSearch { .. } => "file_search",
    }
}

fn tool_definition_value(tool: &ResponsesTool) -> Value {
    let mut value = Map::new();

    match tool {
        ResponsesTool::Function {
            name,
            description,
            parameters,
            strict,
        } => {
            value.insert("type".into(), Value::String("function".into()));
            value.insert("name".into(), Value::String(name.clone()));

            if let Some(description) = description {
                value.insert("description".into(), Value::String(description.clone()));
            }

            if let Some(parameters) = parameters {
                value.insert("parameters".into(), parameters.clone());
            }

            if let Some(strict) = strict {
                value.insert("strict".into(), Value::Bool(*strict));
            }
        }
        ResponsesTool::WebSearch {
            user_location,
            search_context_size,
        } => {
            value.insert("type".into(), Value::String("web_search_preview".into()));

            if let Some(user_location) = user_location {
                value.insert("user_location".into(), user_location.clone());
            }

            if let Some(search_context_size) = search_context_size {
                value.insert(
                    "search_context_size".into(),
                    Value::String(search_context_size.clone()),
                );
            }
        }
        ResponsesTool::FileSearch {
            vector_store_ids,
            max_num_results,
        } => {
            value.insert("type".into(), Value::String("file_search".into()));
            value.insert(
                "vector_store_ids".into(),
                Value::Array(
                    vector_store_ids
                        .iter()
                        .cloned()
                        .map(Value::String)
                        .collect(),
                ),
            );

            if let Some(max_num_results) = max_num_results {
                value.insert("max_num_results".into(), Value::from(*max_num_results));
            }
        }
    }

    Value::Object(value)
}
