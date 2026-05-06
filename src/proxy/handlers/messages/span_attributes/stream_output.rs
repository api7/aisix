use std::collections::BTreeMap;

use super::message_attributes::{
    ContentPartView, MessageView, OutputMessageView, ToolCallView, image_source_to_data_url,
    message_content_view_from_content_parts,
};
use crate::{
    gateway::types::anthropic::{AnthropicContentBlock, AnthropicStreamEvent, ContentDelta},
    proxy::utils::trace::span_message_attributes::output_message_span_properties,
};

enum StreamOutputBlock {
    Text(String),
    ImageUrl {
        url: String,
    },
    ToolUse {
        id: Option<String>,
        name: String,
        arguments: String,
    },
}

#[derive(Default)]
pub(crate) struct StreamOutputCollector {
    role: Option<String>,
    blocks: BTreeMap<usize, StreamOutputBlock>,
    finish_reason: Option<String>,
}

impl StreamOutputCollector {
    pub(crate) fn record_event(&mut self, event: &AnthropicStreamEvent) {
        match event {
            AnthropicStreamEvent::MessageStart { message } => {
                self.role = Some(message.role.clone());
            }
            AnthropicStreamEvent::ContentBlockStart {
                index,
                content_block,
            } => match content_block {
                AnthropicContentBlock::Text { text, .. } => {
                    self.blocks
                        .insert(*index, StreamOutputBlock::Text(text.clone()));
                }
                AnthropicContentBlock::Image { source, .. } => {
                    self.blocks.insert(
                        *index,
                        StreamOutputBlock::ImageUrl {
                            url: image_source_to_data_url(source),
                        },
                    );
                }
                AnthropicContentBlock::ToolUse {
                    id, name, input, ..
                } => {
                    self.blocks.insert(
                        *index,
                        StreamOutputBlock::ToolUse {
                            id: Some(id.clone()),
                            name: name.clone(),
                            arguments: match input {
                                serde_json::Value::Object(map) if map.is_empty() => String::new(),
                                _ => input.to_string(),
                            },
                        },
                    );
                }
                AnthropicContentBlock::ToolResult { .. } => {}
            },
            AnthropicStreamEvent::ContentBlockDelta { index, delta } => match delta {
                ContentDelta::TextDelta { text } => match self.blocks.get_mut(index) {
                    Some(StreamOutputBlock::Text(existing_text)) => existing_text.push_str(text),
                    _ => {
                        self.blocks
                            .insert(*index, StreamOutputBlock::Text(text.clone()));
                    }
                },
                ContentDelta::InputJsonDelta { partial_json } => match self.blocks.get_mut(index) {
                    Some(StreamOutputBlock::ToolUse { arguments, .. }) => {
                        arguments.push_str(partial_json);
                    }
                    _ => {
                        self.blocks.insert(
                            *index,
                            StreamOutputBlock::ToolUse {
                                id: None,
                                name: String::new(),
                                arguments: partial_json.clone(),
                            },
                        );
                    }
                },
            },
            AnthropicStreamEvent::MessageDelta { delta, .. } => {
                self.finish_reason = delta.stop_reason.clone();
            }
            AnthropicStreamEvent::ContentBlockStop { .. }
            | AnthropicStreamEvent::MessageStop
            | AnthropicStreamEvent::Ping
            | AnthropicStreamEvent::Error { .. } => {}
        }
    }

    pub(crate) fn output_message_span_properties(&self) -> Vec<(String, String)> {
        output_message_span_properties(&self.output_message_views())
    }

    fn output_message_views(&self) -> Vec<OutputMessageView> {
        if self.role.is_none() && self.blocks.is_empty() && self.finish_reason.is_none() {
            return Vec::new();
        }

        let mut content_parts = Vec::new();
        let mut tool_calls = Vec::new();

        for block in self.blocks.values() {
            match block {
                StreamOutputBlock::Text(text) if !text.is_empty() => {
                    content_parts.push(ContentPartView::Text(text.clone()));
                }
                StreamOutputBlock::ImageUrl { url } => {
                    content_parts.push(ContentPartView::ImageUrl { url: url.clone() });
                }
                StreamOutputBlock::ToolUse {
                    id,
                    name,
                    arguments,
                } => {
                    tool_calls.push(ToolCallView {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: arguments.clone(),
                    });
                }
                StreamOutputBlock::Text(_) => {}
            }
        }

        vec![OutputMessageView {
            message: MessageView {
                role: self.role.clone().unwrap_or_else(|| "assistant".into()),
                content: message_content_view_from_content_parts(content_parts),
                name: None,
                tool_calls,
                tool_call_id: None,
            },
            finish_reason: self.finish_reason.clone(),
        }]
    }
}
