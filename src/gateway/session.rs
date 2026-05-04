use std::{
    collections::HashMap,
    sync::atomic::{AtomicU64, Ordering},
};

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::RwLock;

use crate::gateway::{error::Result, types::openai::ChatMessage};

#[allow(dead_code)]
#[async_trait]
pub trait SessionStore: Send + Sync + 'static {
    async fn get_by_response_id(&self, response_id: &str) -> Result<Option<StoredSession>>;
    async fn get_by_conversation_id(&self, conv_id: &str) -> Result<Vec<StoredSession>>;
    async fn put_session(&self, session: &StoredSession) -> Result<()>;
    async fn delete_session(&self, response_id: &str) -> Result<()>;
}

#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct StoredSession {
    pub response_id: String,
    pub conversation_id: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub model: String,
    pub created_at: u64,
    pub insertion_index: u64,
    pub metadata: HashMap<String, Value>,
}

#[allow(dead_code)]
#[derive(Debug, Default)]
pub struct InMemorySessionStore {
    sessions: RwLock<HashMap<String, StoredSession>>,
    next_insertion_index: AtomicU64,
}

#[async_trait]
impl SessionStore for InMemorySessionStore {
    async fn get_by_response_id(&self, response_id: &str) -> Result<Option<StoredSession>> {
        Ok(self.sessions.read().await.get(response_id).cloned())
    }

    async fn get_by_conversation_id(&self, conv_id: &str) -> Result<Vec<StoredSession>> {
        let mut sessions: Vec<_> = self
            .sessions
            .read()
            .await
            .values()
            .filter(|session| session.conversation_id.as_deref() == Some(conv_id))
            .cloned()
            .collect();
        sessions.sort_by_key(|session| (session.created_at, session.insertion_index));
        Ok(sessions)
    }

    async fn put_session(&self, session: &StoredSession) -> Result<()> {
        let mut stored = session.clone();
        let mut sessions = self.sessions.write().await;
        stored.insertion_index = sessions
            .get(&stored.response_id)
            .map(|existing| existing.insertion_index)
            .unwrap_or_else(|| self.next_insertion_index.fetch_add(1, Ordering::Relaxed));
        sessions.insert(stored.response_id.clone(), stored);
        Ok(())
    }

    async fn delete_session(&self, response_id: &str) -> Result<()> {
        self.sessions.write().await.remove(response_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use pretty_assertions::assert_eq;
    use serde_json::json;

    use super::{InMemorySessionStore, SessionStore, StoredSession};
    use crate::gateway::types::openai::{ChatMessage, MessageContent};

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
    async fn in_memory_session_store_crud_round_trip() {
        let store = InMemorySessionStore::default();
        let session = StoredSession {
            response_id: "resp_1".into(),
            conversation_id: Some("conv_1".into()),
            messages: vec![sample_message("hello")],
            model: "gpt-test".into(),
            created_at: 10,
            insertion_index: 0,
            metadata: HashMap::from([("trace".into(), json!("abc"))]),
        };

        store.put_session(&session).await.unwrap();

        let loaded = store.get_by_response_id("resp_1").await.unwrap().unwrap();
        assert_eq!(loaded.response_id, "resp_1");
        assert_eq!(loaded.conversation_id.as_deref(), Some("conv_1"));
        assert_eq!(loaded.messages.len(), 1);
        assert_eq!(loaded.model, "gpt-test");
        assert_eq!(loaded.metadata.get("trace"), Some(&json!("abc")));

        let by_conversation = store.get_by_conversation_id("conv_1").await.unwrap();
        assert_eq!(by_conversation.len(), 1);
        assert_eq!(by_conversation[0].response_id, "resp_1");

        store.delete_session("resp_1").await.unwrap();
        assert!(store.get_by_response_id("resp_1").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn in_memory_session_store_returns_conversation_sessions_in_created_order() {
        let store = InMemorySessionStore::default();
        let newer = StoredSession {
            response_id: "resp_2".into(),
            conversation_id: Some("conv_1".into()),
            messages: vec![sample_message("newer")],
            model: "gpt-test".into(),
            created_at: 20,
            insertion_index: 0,
            metadata: HashMap::new(),
        };
        let older = StoredSession {
            response_id: "resp_1".into(),
            conversation_id: Some("conv_1".into()),
            messages: vec![sample_message("older")],
            model: "gpt-test".into(),
            created_at: 10,
            insertion_index: 0,
            metadata: HashMap::new(),
        };

        store.put_session(&newer).await.unwrap();
        store.put_session(&older).await.unwrap();

        let sessions = store.get_by_conversation_id("conv_1").await.unwrap();
        assert_eq!(sessions.len(), 2);
        assert_eq!(sessions[0].response_id, "resp_1");
        assert_eq!(sessions[1].response_id, "resp_2");
    }

    #[tokio::test]
    async fn in_memory_session_store_breaks_created_at_ties_by_insertion_order() {
        let store = InMemorySessionStore::default();
        let first = StoredSession {
            response_id: "resp_1".into(),
            conversation_id: Some("conv_1".into()),
            messages: vec![sample_message("first")],
            model: "gpt-test".into(),
            created_at: 10,
            insertion_index: 0,
            metadata: HashMap::new(),
        };
        let second = StoredSession {
            response_id: "resp_2".into(),
            conversation_id: Some("conv_1".into()),
            messages: vec![sample_message("second")],
            model: "gpt-test".into(),
            created_at: 10,
            insertion_index: 0,
            metadata: HashMap::new(),
        };

        store.put_session(&first).await.unwrap();
        store.put_session(&second).await.unwrap();

        let sessions = store.get_by_conversation_id("conv_1").await.unwrap();
        assert_eq!(sessions.len(), 2);
        assert_eq!(sessions[0].response_id, "resp_1");
        assert_eq!(sessions[1].response_id, "resp_2");
        assert!(sessions[0].insertion_index < sessions[1].insertion_index);
    }
}
