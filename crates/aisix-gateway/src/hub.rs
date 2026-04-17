//! The [`Hub`] dispatches `ChatFormat` requests to the matching
//! [`Bridge`] based on the target Model's `Provider` enum.
//!
//! Hubs are constructed once at startup (spec §1 step 7 — before the
//! proxy router is built) and hold an `Arc<dyn Bridge>` per provider.
//! Lookups are `O(1)` — a 4-entry hashmap keyed on the Provider enum.
//!
//! There is no fallback logic here — that is the proxy layer's job and
//! lands in its own PR. The Hub exists purely to resolve Provider →
//! Bridge cheaply and consistently.

use aisix_core::models::Provider;
use dashmap::DashMap;
use std::sync::Arc;

use crate::bridge::Bridge;

/// Registry of providers → bridges.
///
/// `DashMap` lets us register bridges after construction (useful for tests
/// and for future dynamic-reload scenarios) without taking out a lock on
/// the lookup path.
#[derive(Default)]
pub struct Hub {
    bridges: DashMap<Provider, Arc<dyn Bridge>>,
}

impl Hub {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a bridge for a provider. Overwrites any previous entry,
    /// which is what we want during live reconfigure — the etcd watcher
    /// can swap a broken bridge without tearing down the Hub.
    pub fn register(&self, provider: Provider, bridge: Arc<dyn Bridge>) {
        self.bridges.insert(provider, bridge);
    }

    pub fn get(&self, provider: Provider) -> Option<Arc<dyn Bridge>> {
        self.bridges.get(&provider).map(|r| r.clone())
    }

    pub fn providers(&self) -> Vec<Provider> {
        self.bridges.iter().map(|r| *r.key()).collect()
    }

    pub fn len(&self) -> usize {
        self.bridges.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bridges.is_empty()
    }
}

impl std::fmt::Debug for Hub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Hub")
            .field("providers", &self.providers())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bridge::{BridgeContext, BridgeError, ChatChunkStream};
    use crate::chat::{ChatFormat, ChatMessage, ChatResponse, FinishReason, UsageStats};
    use async_trait::async_trait;
    use futures::stream;

    /// Minimal Bridge that short-circuits to a canned response. Used to
    /// verify Hub wiring without dragging in reqwest or a real provider.
    struct StubBridge {
        name: &'static str,
    }

    #[async_trait]
    impl Bridge for StubBridge {
        fn name(&self) -> &'static str {
            self.name
        }

        async fn chat(
            &self,
            req: &ChatFormat,
            _ctx: &BridgeContext,
        ) -> Result<ChatResponse, BridgeError> {
            Ok(ChatResponse {
                id: "stub-1".into(),
                model: req.model.clone(),
                message: ChatMessage::assistant("stubbed"),
                finish_reason: FinishReason::Stop,
                usage: UsageStats::new(0, 0),
            })
        }

        async fn chat_stream(
            &self,
            _req: &ChatFormat,
            _ctx: &BridgeContext,
        ) -> Result<ChatChunkStream, BridgeError> {
            Ok(Box::pin(stream::iter(Vec::new())))
        }
    }

    #[test]
    fn empty_hub_returns_none_for_any_provider() {
        let hub = Hub::new();
        assert!(hub.is_empty());
        assert!(hub.get(Provider::Openai).is_none());
    }

    #[test]
    fn register_and_get_round_trip() {
        let hub = Hub::new();
        hub.register(
            Provider::Openai,
            Arc::new(StubBridge {
                name: "stub-openai",
            }),
        );
        let b = hub.get(Provider::Openai).unwrap();
        assert_eq!(b.name(), "stub-openai");
    }

    #[test]
    fn register_overwrites_previous_bridge_for_same_provider() {
        let hub = Hub::new();
        hub.register(Provider::Openai, Arc::new(StubBridge { name: "v1" }));
        hub.register(Provider::Openai, Arc::new(StubBridge { name: "v2" }));
        assert_eq!(hub.len(), 1);
        assert_eq!(hub.get(Provider::Openai).unwrap().name(), "v2");
    }

    #[test]
    fn providers_returns_all_registered_keys() {
        let hub = Hub::new();
        hub.register(Provider::Openai, Arc::new(StubBridge { name: "a" }));
        hub.register(Provider::Anthropic, Arc::new(StubBridge { name: "b" }));
        let mut ps = hub.providers();
        ps.sort_by_key(|p| format!("{p:?}"));
        assert_eq!(ps.len(), 2);
    }

    #[tokio::test]
    async fn registered_bridge_is_callable() {
        let hub = Hub::new();
        hub.register(Provider::Openai, Arc::new(StubBridge { name: "stub" }));
        let bridge = hub.get(Provider::Openai).unwrap();

        let m = std::sync::Arc::new(
            serde_json::from_str::<aisix_core::Model>(
                r#"{"name":"t","model":"openai/gpt-4o","provider_config":{"api_key":"k"}}"#,
            )
            .unwrap(),
        );
        let ctx = BridgeContext::new("req-1", m);
        let req = ChatFormat::new("t", vec![ChatMessage::user("hi")]);

        let resp = bridge.chat(&req, &ctx).await.unwrap();
        assert_eq!(resp.message.content, "stubbed");
        assert_eq!(resp.finish_reason, FinishReason::Stop);
    }
}
