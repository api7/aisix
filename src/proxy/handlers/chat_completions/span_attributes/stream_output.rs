use std::collections::BTreeMap;

use super::message_attributes::{
    MessageContentView, MessageView, OutputMessageView, ToolCallView,
    append_openinference_output_message_properties, gen_ai_output_messages_json,
};
use crate::gateway::types::openai::ChatCompletionChunk;

#[derive(Default)]
struct StreamOutputToolCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

#[derive(Default)]
struct StreamOutputChoice {
    role: Option<String>,
    content: String,
    tool_calls: BTreeMap<usize, StreamOutputToolCall>,
    finish_reason: Option<String>,
}

#[derive(Default)]
pub(in crate::proxy::handlers::chat_completions) struct StreamOutputCollector {
    choices: BTreeMap<u32, StreamOutputChoice>,
}

impl StreamOutputCollector {
    pub(in crate::proxy::handlers::chat_completions) fn record_chunk(
        &mut self,
        chunk: &ChatCompletionChunk,
    ) {
        for choice in &chunk.choices {
            let output_choice = self.choices.entry(choice.index).or_default();

            if let Some(role) = &choice.delta.role {
                output_choice.role = Some(role.clone());
            }

            if let Some(content) = &choice.delta.content {
                output_choice.content.push_str(content);
            }

            if let Some(tool_calls) = &choice.delta.tool_calls {
                for tool_call in tool_calls {
                    let output_tool_call =
                        output_choice.tool_calls.entry(tool_call.index).or_default();

                    if let Some(id) = &tool_call.id {
                        output_tool_call.id = Some(id.clone());
                    }

                    if let Some(function) = &tool_call.function {
                        if let Some(name) = &function.name {
                            output_tool_call.name = Some(name.clone());
                        }

                        if let Some(arguments) = &function.arguments {
                            output_tool_call.arguments.push_str(arguments);
                        }
                    }
                }
            }

            if let Some(finish_reason) = &choice.finish_reason {
                output_choice.finish_reason = Some(finish_reason.clone());
            }
        }
    }

    pub(in crate::proxy::handlers::chat_completions) fn output_message_span_properties(
        &self,
    ) -> Vec<(String, String)> {
        let output_messages = self.output_message_views();
        let mut properties = Vec::new();

        append_openinference_output_message_properties(
            &mut properties,
            "llm.output_messages",
            &output_messages,
        );

        if let Some(value) = gen_ai_output_messages_json(&output_messages) {
            properties.push(("gen_ai.output.messages".into(), value));
        }

        properties
    }

    fn output_message_views(&self) -> Vec<OutputMessageView> {
        self.choices
            .values()
            .map(|choice| OutputMessageView {
                message: MessageView {
                    role: choice.role.clone().unwrap_or_else(|| "assistant".into()),
                    content: (!choice.content.is_empty())
                        .then(|| MessageContentView::Text(choice.content.clone())),
                    name: None,
                    tool_calls: choice
                        .tool_calls
                        .values()
                        .filter_map(|tool_call| {
                            tool_call.name.clone().map(|name| ToolCallView {
                                id: tool_call.id.clone(),
                                name,
                                arguments: tool_call.arguments.clone(),
                            })
                        })
                        .collect(),
                    tool_call_id: None,
                },
                finish_reason: choice.finish_reason.clone(),
            })
            .collect()
    }
}
