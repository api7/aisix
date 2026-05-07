use std::collections::{BTreeMap, HashMap};

use serde_json::{Map, Value};
use uuid::Uuid;

use crate::{
    gateway::{
        error::{GatewayError, Result},
        types::{
            common::Usage,
            openai::{
                ChatMessage, ContentPart, FunctionCall, MessageContent, ToolCall,
                responses::{
                    ResponsesApiRequest, ResponsesApiResponse, ResponsesApiStreamEvent,
                    ResponsesContent, ResponsesContentPart, ResponsesInput, ResponsesInputItem,
                    ResponsesOutputContent, ResponsesOutputItem, ResponsesUsage,
                },
            },
        },
    },
    proxy::message_history::{
        MessageHistoryStorage, StoredMessageHistory, StoredMessageHistoryStatus,
    },
};

#[derive(Debug, Clone, Default)]
pub(crate) struct ResponsesLifecycleState {
    pub response_id: String,
    pub previous_response_id: Option<String>,
    pub merged_input_messages: Vec<ChatMessage>,
    pub model: String,
    pub metadata: HashMap<String, Value>,
    pub store: bool,
    pub accumulator: ResponsesStreamAccumulator,
}

impl ResponsesLifecycleState {
    fn request_metadata_value(&self) -> Option<Value> {
        if self.metadata.is_empty() {
            None
        } else {
            Some(Value::Object(
                self.metadata
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect::<Map<String, Value>>(),
            ))
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ResponsesStreamAccumulator {
    response_id: Option<String>,
    model: Option<String>,
    created_at: Option<u64>,
    metadata: Option<Value>,
    usage: Option<ResponsesUsage>,
    output_items: BTreeMap<usize, ResponsesOutputItem>,
    failed: bool,
}

impl ResponsesStreamAccumulator {
    pub(crate) fn record_event(&mut self, event: &ResponsesApiStreamEvent) {
        match event {
            ResponsesApiStreamEvent::ResponseCreated { response }
            | ResponsesApiStreamEvent::ResponseInProgress { response }
            | ResponsesApiStreamEvent::ResponseCompleted { response } => {
                self.record_response(response);
            }
            ResponsesApiStreamEvent::OutputItemAdded { output_index, item }
            | ResponsesApiStreamEvent::OutputItemDone { output_index, item } => {
                self.output_items.insert(*output_index, item.clone());
            }
            ResponsesApiStreamEvent::ContentPartAdded {
                output_index,
                content_index,
                part,
            }
            | ResponsesApiStreamEvent::ContentPartDone {
                output_index,
                content_index,
                part,
            } => {
                if let Some(content_part) =
                    self.ensure_output_text_part(*output_index, *content_index)
                {
                    *content_part = part.clone();
                }
            }
            ResponsesApiStreamEvent::OutputTextDelta {
                output_index,
                content_index,
                delta,
            } => {
                if let Some(ResponsesOutputContent::OutputText { text }) =
                    self.ensure_output_text_part(*output_index, *content_index)
                {
                    text.push_str(delta);
                }
            }
            ResponsesApiStreamEvent::OutputTextDone {
                output_index,
                content_index,
                text,
            } => {
                if let Some(ResponsesOutputContent::OutputText { text: current }) =
                    self.ensure_output_text_part(*output_index, *content_index)
                {
                    *current = text.clone();
                }
            }
            ResponsesApiStreamEvent::FunctionCallArgumentsDelta {
                output_index,
                delta,
            } => {
                if let ResponsesOutputItem::FunctionCall { arguments, .. } =
                    self.ensure_function_call_item(*output_index)
                {
                    arguments.push_str(delta);
                }
            }
            ResponsesApiStreamEvent::FunctionCallArgumentsDone {
                output_index,
                arguments,
            } => {
                if let ResponsesOutputItem::FunctionCall {
                    arguments: current, ..
                } = self.ensure_function_call_item(*output_index)
                {
                    *current = arguments.clone();
                }
            }
            ResponsesApiStreamEvent::Error { .. } => {
                self.failed = true;
            }
        }
    }

    pub(crate) fn response_snapshot(
        &self,
        previous_response_id: Option<String>,
        fallback_model: &str,
        fallback_metadata: Option<Value>,
        fallback_usage: Option<Usage>,
    ) -> Result<ResponsesApiResponse> {
        if self.failed {
            return Err(GatewayError::Validation(
                "responses stream ended in error; no completed snapshot available".into(),
            ));
        }

        let response_id = self.response_id.clone().ok_or_else(|| {
            GatewayError::Validation(
                "responses stream completed without an upstream response id".into(),
            )
        })?;

        let usage = fallback_usage
            .as_ref()
            .map(responses_usage_from_common)
            .or_else(|| self.usage.clone())
            .unwrap_or_default();

        Ok(ResponsesApiResponse {
            id: response_id,
            object: "response".into(),
            created_at: self.created_at.unwrap_or_default(),
            model: self
                .model
                .clone()
                .unwrap_or_else(|| fallback_model.to_owned()),
            output: self.output_items.values().cloned().collect(),
            status: "completed".into(),
            usage,
            metadata: self.metadata.clone().or(fallback_metadata),
            previous_response_id,
        })
    }

    fn record_response(&mut self, response: &ResponsesApiResponse) {
        if self.response_id.is_none() && !response.id.is_empty() {
            self.response_id = Some(response.id.clone());
        }
        if self.model.is_none() && !response.model.is_empty() {
            self.model = Some(response.model.clone());
        }
        if self.created_at.is_none() {
            self.created_at = Some(response.created_at);
        }
        if self.metadata.is_none() && response.metadata.is_some() {
            self.metadata = response.metadata.clone();
        }
        self.usage = Some(response.usage.clone());
        for (output_index, item) in response.output.iter().cloned().enumerate() {
            self.output_items.insert(output_index, item);
        }
    }

    fn ensure_output_text_part(
        &mut self,
        output_index: usize,
        content_index: usize,
    ) -> Option<&mut ResponsesOutputContent> {
        let item = self.ensure_message_item(output_index);
        let ResponsesOutputItem::Message { content, .. } = item else {
            return None;
        };

        if content.len() <= content_index {
            content.resize_with(content_index + 1, || ResponsesOutputContent::OutputText {
                text: String::new(),
            });
        }
        content.get_mut(content_index)
    }

    fn ensure_message_item(&mut self, output_index: usize) -> &mut ResponsesOutputItem {
        let response_id = self.response_id.as_deref().unwrap_or("response").to_owned();
        self.output_items
            .entry(output_index)
            .or_insert_with(|| ResponsesOutputItem::Message {
                id: response_message_output_id(&response_id, output_index),
                role: "assistant".into(),
                content: vec![],
                status: "in_progress".into(),
            })
    }

    fn ensure_function_call_item(&mut self, output_index: usize) -> &mut ResponsesOutputItem {
        let response_id = self.response_id.as_deref().unwrap_or("response").to_owned();
        self.output_items.entry(output_index).or_insert_with(|| {
            let id = response_function_call_output_id(&response_id, output_index);
            ResponsesOutputItem::FunctionCall {
                id: id.clone(),
                call_id: id,
                name: format!("tool_{}", output_index),
                arguments: String::new(),
                status: "in_progress".into(),
            }
        })
    }
}

pub(crate) fn init_lifecycle(request: &ResponsesApiRequest) -> ResponsesLifecycleState {
    ResponsesLifecycleState {
        response_id: generate_response_id(),
        previous_response_id: request.previous_response_id.clone(),
        merged_input_messages: vec![],
        model: request.model.clone(),
        metadata: request_metadata(request),
        store: request.store != Some(false),
        accumulator: ResponsesStreamAccumulator::default(),
    }
}

pub(crate) async fn load_previous_messages<S>(
    storage: &S,
    previous_response_id: Option<&str>,
) -> Result<Vec<ChatMessage>>
where
    S: MessageHistoryStorage + ?Sized,
{
    let Some(previous_response_id) = previous_response_id else {
        return Ok(vec![]);
    };

    let history = storage
        .get_by_response_id(previous_response_id)
        .await?
        .ok_or_else(|| {
            GatewayError::Validation(format!(
                "previous_response_not_found: {}",
                previous_response_id
            ))
        })?;
    Ok(history.cumulative_messages)
}

pub(crate) fn build_merged_input_messages(
    request: &ResponsesApiRequest,
    previous_messages: &[ChatMessage],
) -> Result<Vec<ChatMessage>> {
    let mut merged_input_messages = previous_messages.to_vec();
    merged_input_messages.extend(request_input_messages(request)?);
    Ok(merged_input_messages)
}

pub(crate) fn accumulate_stream_event(
    state: &mut ResponsesLifecycleState,
    event: &ResponsesApiStreamEvent,
) {
    state.accumulator.record_event(event);
}

pub(crate) fn accumulate_complete(
    state: &ResponsesLifecycleState,
    response: &ResponsesApiResponse,
    usage: Usage,
) -> Result<StoredMessageHistory> {
    completed_history(state, response, usage)
}

pub(crate) fn accumulate_stream_success(
    state: &ResponsesLifecycleState,
    usage: Option<&Usage>,
) -> Result<StoredMessageHistory> {
    let response = state.accumulator.response_snapshot(
        state.previous_response_id.clone(),
        &state.model,
        state.request_metadata_value(),
        usage.cloned(),
    )?;
    let usage = usage
        .cloned()
        .unwrap_or_else(|| responses_usage_to_common(&response.usage))
        .with_derived_total();
    completed_history(state, &response, usage)
}

pub(crate) async fn persist_if_enabled<S>(
    storage: &S,
    state: &ResponsesLifecycleState,
    history: &StoredMessageHistory,
) -> Result<()>
where
    S: MessageHistoryStorage + ?Sized,
{
    if state.store {
        storage.put(history).await?;
    }
    Ok(())
}

fn completed_history(
    state: &ResponsesLifecycleState,
    response: &ResponsesApiResponse,
    usage: Usage,
) -> Result<StoredMessageHistory> {
    let mut cumulative_messages = state.merged_input_messages.clone();
    cumulative_messages.extend(response_output_to_chat_messages(&response.output));

    let mut metadata = state.metadata.clone();
    merge_metadata_value(&mut metadata, response.metadata.as_ref());

    Ok(StoredMessageHistory {
        response_id: state.response_id.clone(),
        previous_response_id: state.previous_response_id.clone(),
        upstream_response_id: Some(response.id.clone()),
        cumulative_messages,
        model: response.model.clone(),
        created_at: response.created_at,
        finished_at: Some(response.created_at),
        usage: Some(usage),
        status: StoredMessageHistoryStatus::Completed,
        metadata,
    })
}

fn generate_response_id() -> String {
    format!("aresp_{}", Uuid::new_v4().simple())
}

fn request_metadata(request: &ResponsesApiRequest) -> HashMap<String, Value> {
    request
        .metadata
        .as_ref()
        .and_then(Value::as_object)
        .map(|metadata| metadata.clone().into_iter().collect())
        .unwrap_or_default()
}

fn request_input_messages(request: &ResponsesApiRequest) -> Result<Vec<ChatMessage>> {
    match &request.input {
        ResponsesInput::Text(text) => Ok(vec![ChatMessage {
            role: "user".into(),
            content: Some(MessageContent::Text(text.clone())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }]),
        ResponsesInput::Items(items) => items.iter().try_fold(vec![], |mut messages, item| {
            if let Some(message) = request_input_item_to_chat_message(item)? {
                messages.push(message);
            }
            Ok(messages)
        }),
    }
}

fn request_input_item_to_chat_message(item: &ResponsesInputItem) -> Result<Option<ChatMessage>> {
    match item {
        ResponsesInputItem::Message { role, content } => {
            let Some(content) = request_content_to_message_content(content)? else {
                return Ok(None);
            };

            Ok(Some(ChatMessage {
                role: role.clone(),
                content: Some(content),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }))
        }
        ResponsesInputItem::FunctionCallOutput { call_id, output } => Ok(Some(ChatMessage {
            role: "tool".into(),
            content: Some(MessageContent::Text(output.clone())),
            name: None,
            tool_calls: None,
            tool_call_id: Some(call_id.clone()),
        })),
    }
}

fn request_content_to_message_content(
    content: &ResponsesContent,
) -> Result<Option<MessageContent>> {
    match content {
        ResponsesContent::Text(text) => Ok(Some(MessageContent::Text(text.clone()))),
        ResponsesContent::Parts(parts) => {
            let parts = parts
                .iter()
                .map(request_content_part_to_content_part)
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();

            if parts.is_empty() {
                Ok(None)
            } else {
                Ok(Some(MessageContent::Parts(parts)))
            }
        }
    }
}

fn request_content_part_to_content_part(
    part: &ResponsesContentPart,
) -> Result<Option<ContentPart>> {
    match part {
        ResponsesContentPart::InputText { text } => {
            Ok(Some(ContentPart::Text { text: text.clone() }))
        }
        ResponsesContentPart::InputImage {
            image_url, detail, ..
        } => Ok(image_url.as_ref().map(|url| ContentPart::ImageUrl {
            image_url: crate::gateway::types::openai::ImageUrl {
                url: url.clone(),
                detail: detail.clone(),
            },
        })),
    }
}

fn response_output_to_chat_messages(output: &[ResponsesOutputItem]) -> Vec<ChatMessage> {
    let mut messages = Vec::new();
    let mut current_assistant_index = None;

    for item in output {
        match item {
            ResponsesOutputItem::Message { role, content, .. } => {
                let message = ChatMessage {
                    role: role.clone(),
                    content: response_output_content_to_message_content(content),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                };
                messages.push(message);
                current_assistant_index = (role == "assistant").then_some(messages.len() - 1);
            }
            ResponsesOutputItem::FunctionCall {
                id,
                name,
                arguments,
                ..
            } => {
                let assistant_index = current_assistant_index.unwrap_or_else(|| {
                    messages.push(ChatMessage {
                        role: "assistant".into(),
                        content: None,
                        name: None,
                        tool_calls: Some(vec![]),
                        tool_call_id: None,
                    });
                    let index = messages.len() - 1;
                    current_assistant_index = Some(index);
                    index
                });

                messages[assistant_index]
                    .tool_calls
                    .get_or_insert_with(Vec::new)
                    .push(ToolCall {
                        id: id.clone(),
                        r#type: "function".into(),
                        function: FunctionCall {
                            name: name.clone(),
                            arguments: arguments.clone(),
                        },
                    });
            }
        }
    }

    messages
        .into_iter()
        .map(|mut message| {
            if message
                .tool_calls
                .as_ref()
                .is_some_and(|tool_calls| tool_calls.is_empty())
            {
                message.tool_calls = None;
            }
            message
        })
        .collect()
}

fn response_output_content_to_message_content(
    content: &[ResponsesOutputContent],
) -> Option<MessageContent> {
    match content {
        [] => None,
        [ResponsesOutputContent::OutputText { text }] => Some(MessageContent::Text(text.clone())),
        multiple => Some(MessageContent::Parts(
            multiple
                .iter()
                .map(|part| match part {
                    ResponsesOutputContent::OutputText { text } => {
                        ContentPart::Text { text: text.clone() }
                    }
                })
                .collect(),
        )),
    }
}

fn merge_metadata_value(metadata: &mut HashMap<String, Value>, extra: Option<&Value>) {
    let Some(extra) = extra.and_then(Value::as_object) else {
        return;
    };

    metadata.extend(extra.clone());
}

fn responses_usage_from_common(usage: &Usage) -> ResponsesUsage {
    ResponsesUsage {
        input_tokens: usage.input_tokens.unwrap_or_default(),
        output_tokens: usage.output_tokens.unwrap_or_default(),
        total_tokens: usage.resolved_total_tokens().unwrap_or_default(),
    }
}

fn responses_usage_to_common(usage: &ResponsesUsage) -> Usage {
    Usage {
        input_tokens: Some(usage.input_tokens),
        output_tokens: Some(usage.output_tokens),
        total_tokens: Some(usage.total_tokens),
        ..Default::default()
    }
}

fn response_message_output_id(response_id: &str, output_index: usize) -> String {
    format!("{}_message_{}", response_id, output_index)
}

fn response_function_call_output_id(response_id: &str, output_index: usize) -> String {
    format!("{}_call_{}", response_id, output_index)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use pretty_assertions::assert_eq;
    use serde_json::json;

    use crate::{
        gateway::types::{
            common::Usage,
            openai::{
                ChatMessage, MessageContent,
                responses::{
                    ResponsesApiRequest, ResponsesApiResponse, ResponsesApiStreamEvent,
                    ResponsesContent, ResponsesContentPart, ResponsesInput, ResponsesInputItem,
                    ResponsesOutputContent, ResponsesOutputItem, ResponsesUsage,
                },
            },
        },
        proxy::message_history::{
            InMemoryMessageHistoryStorage, MessageHistoryStorage, StoredMessageHistory,
            StoredMessageHistoryStatus,
        },
    };

    fn user_message(text: &str) -> ChatMessage {
        ChatMessage {
            role: "user".into(),
            content: Some(MessageContent::Text(text.into())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    fn assistant_message(text: &str) -> ChatMessage {
        ChatMessage {
            role: "assistant".into(),
            content: Some(MessageContent::Text(text.into())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    fn message_text(message: &ChatMessage) -> Option<&str> {
        match message.content.as_ref() {
            Some(MessageContent::Text(text)) => Some(text.as_str()),
            _ => None,
        }
    }

    fn text_request(text: &str) -> ResponsesApiRequest {
        ResponsesApiRequest {
            background: None,
            context_management: None,
            conversation: None,
            include: None,
            model: "gpt-4.1".into(),
            input: ResponsesInput::Text(text.into()),
            instructions: None,
            max_output_tokens: None,
            max_tool_calls: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            prompt: None,
            prompt_cache_key: None,
            prompt_cache_retention: None,
            reasoning: None,
            safety_identifier: None,
            service_tier: None,
            stream: None,
            stream_options: None,
            metadata: None,
            text: None,
            top_logprobs: None,
            previous_response_id: None,
            store: None,
            truncation: None,
            replay_messages: vec![],
        }
    }

    fn completed_response(id: &str, text: &str) -> ResponsesApiResponse {
        ResponsesApiResponse {
            id: id.into(),
            object: "response".into(),
            created_at: 123,
            model: "gpt-4.1".into(),
            output: vec![ResponsesOutputItem::Message {
                id: format!("{}_message_0", id),
                role: "assistant".into(),
                content: vec![ResponsesOutputContent::OutputText { text: text.into() }],
                status: "completed".into(),
            }],
            status: "completed".into(),
            usage: ResponsesUsage {
                input_tokens: 1,
                output_tokens: 2,
                total_tokens: 3,
            },
            metadata: Some(json!({"response_meta": true})),
            previous_response_id: None,
        }
    }

    #[tokio::test]
    async fn load_previous_messages_restores_previous_snapshot_and_lifecycle_generates_response_id()
    {
        let store = Arc::new(InMemoryMessageHistoryStorage::default());
        store
            .put(&StoredMessageHistory {
                response_id: "resp_prev".into(),
                cumulative_messages: vec![user_message("old"), assistant_message("done")],
                model: "gpt-4.1".into(),
                created_at: 100,
                status: StoredMessageHistoryStatus::Completed,
                ..Default::default()
            })
            .await
            .unwrap();

        let mut request = text_request("next");
        request.previous_response_id = Some("resp_prev".into());
        request.metadata = Some(json!({"trace": "abc"}));

        let state = super::init_lifecycle(&request);
        let previous_messages =
            super::load_previous_messages(store.as_ref(), request.previous_response_id.as_deref())
                .await
                .unwrap();
        let merged_input_messages =
            super::build_merged_input_messages(&request, &previous_messages).unwrap();

        assert_eq!(state.previous_response_id.as_deref(), Some("resp_prev"));
        assert_eq!(previous_messages.len(), 2);
        assert_eq!(merged_input_messages.len(), 3);
        assert!(state.response_id.starts_with("aresp_"));
        assert_ne!(state.response_id, "resp_prev");
        assert_eq!(state.metadata.get("trace"), Some(&json!("abc")));
    }

    #[tokio::test]
    async fn finalize_complete_persists_combined_snapshot() {
        let store = Arc::new(InMemoryMessageHistoryStorage::default());
        let mut request = text_request("hello");
        request.metadata = Some(json!({"trace": "abc"}));

        let mut state = super::init_lifecycle(&request);
        state.merged_input_messages = super::build_merged_input_messages(&request, &[]).unwrap();
        let response_id = state.response_id.clone();

        let stored = super::accumulate_complete(
            &state,
            &completed_response("up_resp_1", "world"),
            Usage {
                input_tokens: Some(1),
                output_tokens: Some(2),
                total_tokens: Some(3),
                ..Default::default()
            },
        )
        .unwrap();
        super::persist_if_enabled(store.as_ref(), &state, &stored)
            .await
            .unwrap();

        assert_eq!(stored.response_id, response_id);
        assert_eq!(stored.upstream_response_id.as_deref(), Some("up_resp_1"));
        assert_eq!(stored.cumulative_messages.len(), 2);
        assert_eq!(message_text(&stored.cumulative_messages[0]), Some("hello"));
        assert_eq!(message_text(&stored.cumulative_messages[1]), Some("world"));
        assert_eq!(stored.metadata.get("trace"), Some(&json!("abc")));
        assert_eq!(stored.metadata.get("response_meta"), Some(&json!(true)));

        let loaded = store
            .get_by_response_id(&response_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(loaded.response_id, response_id);
        assert_eq!(
            loaded.usage.as_ref().and_then(|usage| usage.total_tokens),
            Some(3)
        );
    }

    #[tokio::test]
    async fn finalize_stream_success_builds_snapshot_from_events() {
        let store = Arc::new(InMemoryMessageHistoryStorage::default());
        let request = text_request("hello");
        let mut state = super::init_lifecycle(&request);
        state.merged_input_messages = super::build_merged_input_messages(&request, &[]).unwrap();
        let response_id = state.response_id.clone();

        super::accumulate_stream_event(
            &mut state,
            &ResponsesApiStreamEvent::ResponseCreated {
                response: ResponsesApiResponse {
                    id: "up_resp_1".into(),
                    object: "response".into(),
                    created_at: 123,
                    model: "gpt-4.1".into(),
                    output: vec![],
                    status: "in_progress".into(),
                    usage: ResponsesUsage::default(),
                    metadata: Some(json!({"stream": true})),
                    previous_response_id: None,
                },
            },
        );
        super::accumulate_stream_event(
            &mut state,
            &ResponsesApiStreamEvent::OutputItemAdded {
                output_index: 0,
                item: ResponsesOutputItem::Message {
                    id: "up_resp_1_message_0".into(),
                    role: "assistant".into(),
                    content: vec![],
                    status: "in_progress".into(),
                },
            },
        );
        super::accumulate_stream_event(
            &mut state,
            &ResponsesApiStreamEvent::ContentPartAdded {
                output_index: 0,
                content_index: 0,
                part: ResponsesOutputContent::OutputText {
                    text: String::new(),
                },
            },
        );
        super::accumulate_stream_event(
            &mut state,
            &ResponsesApiStreamEvent::OutputTextDelta {
                output_index: 0,
                content_index: 0,
                delta: "world".into(),
            },
        );
        super::accumulate_stream_event(
            &mut state,
            &ResponsesApiStreamEvent::OutputItemDone {
                output_index: 0,
                item: ResponsesOutputItem::Message {
                    id: "up_resp_1_message_0".into(),
                    role: "assistant".into(),
                    content: vec![ResponsesOutputContent::OutputText {
                        text: "world".into(),
                    }],
                    status: "completed".into(),
                },
            },
        );

        let stored = super::accumulate_stream_success(
            &state,
            Some(&Usage {
                input_tokens: Some(1),
                output_tokens: Some(2),
                total_tokens: Some(3),
                ..Default::default()
            }),
        )
        .unwrap();
        super::persist_if_enabled(store.as_ref(), &state, &stored)
            .await
            .unwrap();

        assert_eq!(stored.response_id, response_id);
        assert_eq!(stored.upstream_response_id.as_deref(), Some("up_resp_1"));
        assert_eq!(stored.cumulative_messages.len(), 2);
        assert_eq!(message_text(&stored.cumulative_messages[1]), Some("world"));
        assert_eq!(stored.metadata.get("stream"), Some(&json!(true)));

        let loaded = store
            .get_by_response_id(&response_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(loaded.response_id, response_id);
        assert_eq!(loaded.status, StoredMessageHistoryStatus::Completed);
    }

    #[tokio::test]
    async fn load_previous_messages_returns_validation_when_previous_response_is_missing() {
        let store = Arc::new(InMemoryMessageHistoryStorage::default());
        let mut request = text_request("hello");
        request.previous_response_id = Some("resp_missing".into());

        let error =
            super::load_previous_messages(store.as_ref(), request.previous_response_id.as_deref())
                .await
                .unwrap_err();
        assert_eq!(
            error.to_string(),
            "validation: previous_response_not_found: resp_missing"
        );
    }

    #[test]
    fn response_output_to_chat_messages_groups_function_calls_under_assistant() {
        let messages = super::response_output_to_chat_messages(&[
            ResponsesOutputItem::Message {
                id: "resp_message_0".into(),
                role: "assistant".into(),
                content: vec![ResponsesOutputContent::OutputText {
                    text: "hello".into(),
                }],
                status: "completed".into(),
            },
            ResponsesOutputItem::FunctionCall {
                id: "call_1".into(),
                call_id: "call_1".into(),
                name: "lookup".into(),
                arguments: "{\"city\":\"Paris\"}".into(),
                status: "completed".into(),
            },
        ]);

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "assistant");
        assert_eq!(messages[0].tool_calls.as_ref().map(Vec::len), Some(1));
    }

    #[test]
    fn request_input_messages_keep_tool_outputs() {
        let request = ResponsesApiRequest {
            input: ResponsesInput::Items(vec![
                ResponsesInputItem::Message {
                    role: "user".into(),
                    content: ResponsesContent::Text("hello".into()),
                },
                ResponsesInputItem::FunctionCallOutput {
                    call_id: "call_1".into(),
                    output: "42".into(),
                },
            ]),
            ..text_request("ignored")
        };

        let messages = super::request_input_messages(&request).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[1].role, "tool");
        assert_eq!(messages[1].tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn request_input_messages_skip_non_persistable_input_image_file_references() {
        let request = ResponsesApiRequest {
            input: ResponsesInput::Items(vec![ResponsesInputItem::Message {
                role: "user".into(),
                content: ResponsesContent::Parts(vec![ResponsesContentPart::InputImage {
                    image_url: None,
                    file_id: Some("file_123".into()),
                    detail: Some("high".into()),
                }]),
            }]),
            ..text_request("ignored")
        };

        let messages = super::request_input_messages(&request).unwrap();
        assert!(messages.is_empty());
    }
}
