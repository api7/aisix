use std::collections::BTreeMap;

use super::message_attributes::output_message_views_from_output_items;
use crate::{
    gateway::types::openai::responses::{
        ResponsesApiResponse, ResponsesApiStreamEvent, ResponsesOutputContent, ResponsesOutputItem,
    },
    proxy::utils::trace::span_message_attributes::output_message_span_properties,
};

#[derive(Default)]
pub(crate) struct StreamOutputCollector {
    items: BTreeMap<usize, ResponsesOutputItem>,
    completed_response: Option<ResponsesApiResponse>,
}

impl StreamOutputCollector {
    pub(crate) fn record_event(&mut self, event: &ResponsesApiStreamEvent) {
        match event {
            ResponsesApiStreamEvent::ResponseCreated { response }
            | ResponsesApiStreamEvent::ResponseInProgress { response } => {
                self.sync_response_output(response);
            }
            ResponsesApiStreamEvent::ResponseCompleted { response } => {
                self.completed_response = Some(response.clone());
                self.sync_response_output(response);
            }
            ResponsesApiStreamEvent::OutputItemAdded { output_index, item }
            | ResponsesApiStreamEvent::OutputItemDone { output_index, item } => {
                self.items.insert(*output_index, item.clone());
            }
            ResponsesApiStreamEvent::OutputTextDelta {
                output_index,
                delta,
                ..
            } => {
                if !delta.is_empty() {
                    append_message_text(
                        self.items.entry(*output_index).or_insert_with(|| {
                            ResponsesOutputItem::Message {
                                id: String::new(),
                                role: "assistant".into(),
                                content: vec![],
                                status: "in_progress".into(),
                            }
                        }),
                        delta,
                    );
                }
            }
            ResponsesApiStreamEvent::OutputTextDone {
                output_index, text, ..
            } => {
                set_message_text(
                    self.items.entry(*output_index).or_insert_with(|| {
                        ResponsesOutputItem::Message {
                            id: String::new(),
                            role: "assistant".into(),
                            content: vec![],
                            status: "completed".into(),
                        }
                    }),
                    text,
                );
            }
            ResponsesApiStreamEvent::FunctionCallArgumentsDelta {
                output_index,
                delta,
            } => {
                if !delta.is_empty() {
                    append_function_arguments(
                        self.items.entry(*output_index).or_insert_with(|| {
                            ResponsesOutputItem::FunctionCall {
                                id: String::new(),
                                call_id: String::new(),
                                name: String::new(),
                                arguments: String::new(),
                                status: "in_progress".into(),
                            }
                        }),
                        delta,
                    );
                }
            }
            ResponsesApiStreamEvent::FunctionCallArgumentsDone {
                output_index,
                arguments,
            } => {
                set_function_arguments(
                    self.items.entry(*output_index).or_insert_with(|| {
                        ResponsesOutputItem::FunctionCall {
                            id: String::new(),
                            call_id: String::new(),
                            name: String::new(),
                            arguments: String::new(),
                            status: "completed".into(),
                        }
                    }),
                    arguments,
                );
            }
            ResponsesApiStreamEvent::ContentPartAdded { .. }
            | ResponsesApiStreamEvent::ContentPartDone { .. }
            | ResponsesApiStreamEvent::Error { .. } => {}
        }
    }

    pub(crate) fn output_message_span_properties(&self) -> Vec<(String, String)> {
        if let Some(response) = &self.completed_response {
            return output_message_span_properties(&output_message_views_from_output_items(
                &response.output,
            ));
        }

        let output: Vec<_> = self.items.values().cloned().collect();
        output_message_span_properties(&output_message_views_from_output_items(&output))
    }

    fn sync_response_output(&mut self, response: &ResponsesApiResponse) {
        for (output_index, item) in response.output.iter().cloned().enumerate() {
            self.items.insert(output_index, item);
        }
    }
}

fn append_message_text(item: &mut ResponsesOutputItem, delta: &str) {
    let ResponsesOutputItem::Message { content, .. } = item else {
        return;
    };

    if let Some(ResponsesOutputContent::OutputText { text }) = content.first_mut() {
        text.push_str(delta);
    } else {
        content.push(ResponsesOutputContent::OutputText {
            text: delta.to_string(),
        });
    }
}

fn set_message_text(item: &mut ResponsesOutputItem, text: &str) {
    let ResponsesOutputItem::Message {
        content, status, ..
    } = item
    else {
        return;
    };

    *status = "completed".into();
    content.clear();
    content.push(ResponsesOutputContent::OutputText {
        text: text.to_string(),
    });
}

fn append_function_arguments(item: &mut ResponsesOutputItem, delta: &str) {
    let ResponsesOutputItem::FunctionCall { arguments, .. } = item else {
        return;
    };

    arguments.push_str(delta);
}

fn set_function_arguments(item: &mut ResponsesOutputItem, value: &str) {
    let ResponsesOutputItem::FunctionCall {
        arguments, status, ..
    } = item
    else {
        return;
    };

    *status = "completed".into();
    *arguments = value.to_string();
}
