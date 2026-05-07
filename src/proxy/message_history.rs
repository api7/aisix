use std::collections::HashMap;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::RwLock;

use crate::gateway::{
    error::Result,
    types::{common::Usage, openai::ChatMessage},
};

#[async_trait]
pub(crate) trait MessageHistoryStorage: Send + Sync + 'static {
    async fn get_by_response_id(&self, response_id: &str) -> Result<Option<StoredMessageHistory>>;

    async fn put(&self, history: &StoredMessageHistory) -> Result<()>;
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum StoredMessageHistoryStatus {
    #[default]
    Completed,
    Failed,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub(crate) struct StoredMessageHistory {
    pub response_id: String,
    pub previous_response_id: Option<String>,
    pub upstream_response_id: Option<String>,
    pub cumulative_messages: Vec<ChatMessage>,
    pub model: String,
    pub created_at: u64,
    pub finished_at: Option<u64>,
    pub usage: Option<Usage>,
    pub status: StoredMessageHistoryStatus,
    pub metadata: HashMap<String, Value>,
}

/// In-memory message history storage intended for tests and short-lived local runs.
///
/// This implementation keeps every stored history entry for the lifetime of the
/// process. It has no eviction policy, no size limits, and no persistence, so
/// sustained use can grow without bound and eventually OOM the process.
#[derive(Debug, Default)]
pub(crate) struct InMemoryMessageHistoryStorage {
    histories: RwLock<HashMap<String, StoredMessageHistory>>,
}

#[async_trait]
impl MessageHistoryStorage for InMemoryMessageHistoryStorage {
    async fn get_by_response_id(&self, response_id: &str) -> Result<Option<StoredMessageHistory>> {
        Ok(self.histories.read().await.get(response_id).cloned())
    }

    async fn put(&self, history: &StoredMessageHistory) -> Result<()> {
        self.histories
            .write()
            .await
            .insert(history.response_id.clone(), history.clone());
        Ok(())
    }
}

#[cfg(test)]
impl InMemoryMessageHistoryStorage {
    async fn delete(&self, response_id: &str) -> Result<()> {
        self.histories.write().await.remove(response_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::{
        InMemoryMessageHistoryStorage, MessageHistoryStorage, StoredMessageHistory,
        StoredMessageHistoryStatus,
    };
    use crate::gateway::types::{
        common::Usage,
        openai::{ChatMessage, MessageContent},
    };

    fn sample_message(text: &str) -> ChatMessage {
        ChatMessage {
            role: "user".into(),
            content: Some(MessageContent::Text(text.into())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    #[tokio::test]
    async fn in_memory_storage_round_trips_message_history() {
        let storage = InMemoryMessageHistoryStorage::default();
        let history = StoredMessageHistory {
            response_id: "resp_1".into(),
            previous_response_id: Some("resp_0".into()),
            upstream_response_id: Some("chatcmpl-upstream-1".into()),
            cumulative_messages: vec![sample_message("hello")],
            model: "gpt-test".into(),
            created_at: 10,
            finished_at: Some(11),
            usage: Some(Usage {
                input_tokens: Some(1),
                output_tokens: Some(2),
                total_tokens: Some(3),
                ..Default::default()
            }),
            status: StoredMessageHistoryStatus::Completed,
            metadata: HashMap::from([("trace".into(), json!("abc"))]),
        };

        storage.put(&history).await.unwrap();

        let loaded = storage.get_by_response_id("resp_1").await.unwrap().unwrap();
        assert_eq!(loaded.response_id, "resp_1");
        assert_eq!(loaded.previous_response_id.as_deref(), Some("resp_0"));
        assert_eq!(
            loaded.upstream_response_id.as_deref(),
            Some("chatcmpl-upstream-1")
        );
        assert_eq!(loaded.cumulative_messages.len(), 1);
        assert_eq!(loaded.model, "gpt-test");
        assert_eq!(loaded.metadata.get("trace"), Some(&json!("abc")));
    }

    #[tokio::test]
    async fn delete_removes_entry() {
        let storage = InMemoryMessageHistoryStorage::default();
        let history = StoredMessageHistory {
            response_id: "resp_1".into(),
            cumulative_messages: vec![sample_message("hello")],
            model: "gpt-test".into(),
            created_at: 10,
            status: StoredMessageHistoryStatus::Completed,
            ..Default::default()
        };

        storage.put(&history).await.unwrap();
        assert!(
            storage
                .get_by_response_id("resp_1")
                .await
                .unwrap()
                .is_some()
        );

        storage.delete("resp_1").await.unwrap();
        assert!(
            storage
                .get_by_response_id("resp_1")
                .await
                .unwrap()
                .is_none()
        );
    }
}
