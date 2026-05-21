//! The [`Hub`] dispatches `ChatFormat` requests to the matching
//! [`Bridge`] based on the target [`ProviderKey`]'s vendor identity and
//! wire-shape adapter. There is no per-`Provider`-enum lookup — vendor
//! identity is an open string (`ProviderKey.provider`) and routing
//! falls through a closed 5-value `Adapter` family enum.
//!
//! Hubs are constructed once at startup (spec §1 step 7 — before the
//! proxy router is built) and hold:
//!
//! - `specialized_bridges` — keyed on `ProviderKey.provider` (vendor
//!   string, e.g. `"deepseek"`, `"cohere"`). Used when a specific
//!   vendor needs handling that diverges from its wire-shape default
//!   (e.g. DeepSeek's `reasoning_content` lift, Cohere's chat-compat
//!   namespace).
//! - `family_bridges` — keyed on [`Adapter`] (wire shape: `openai`,
//!   `anthropic`, `bedrock`, `vertex`, `azure-openai`). The default
//!   bridge for any vendor whose `ProviderKey.adapter` matches that
//!   wire shape and has no specialized override.
//!
//! [`Hub::dispatch_two_tier`] looks up specialized first, then falls
//! back to the family bridge. A new catalog vendor admitted by cp-api
//! works without a DP code change: the family bridge for its adapter
//! handles the request using `ProviderKey.api_base` for the upstream.
//!
//! Closes the dispatch half of api7/AISIX-Cloud#302 Phase A and the
//! routing half of api7/AISIX-Cloud#417.

use aisix_core::models::{Adapter, ProviderKey};
use dashmap::DashMap;
use std::sync::Arc;

use crate::bridge::Bridge;

/// Registry of vendor strings + adapter families → bridges.
///
/// `DashMap` lets us register bridges after construction (useful for tests
/// and for future dynamic-reload scenarios) without taking out a lock on
/// the lookup path.
#[derive(Default)]
pub struct Hub {
    /// Wire-shape default bridges. Keyed on [`Adapter`] — the closed
    /// 5-value protocol family enum. Every catalog vendor whose
    /// `ProviderKey.adapter` matches one of these resolves here if
    /// no specialized override is registered.
    family_bridges: DashMap<Adapter, Arc<dyn Bridge>>,
    /// Vendor-specific override bridges. Keyed on
    /// `ProviderKey.provider` (vendor string). Used when a specific
    /// vendor needs handling that diverges from its wire-shape default.
    specialized_bridges: DashMap<String, Arc<dyn Bridge>>,
}

impl Hub {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register the family-tier (wire-shape default) bridge for an
    /// [`Adapter`]. Overwrites any previous entry for the same adapter,
    /// matching live-reconfigure semantics.
    pub fn register_family(&self, adapter: Adapter, bridge: Arc<dyn Bridge>) {
        self.family_bridges.insert(adapter, bridge);
    }

    /// Register a specialized (vendor-specific) bridge keyed on the
    /// `ProviderKey.provider` vendor string. Overwrites any previous
    /// entry for the same vendor key.
    pub fn register_specialized(&self, provider: impl Into<String>, bridge: Arc<dyn Bridge>) {
        self.specialized_bridges.insert(provider.into(), bridge);
    }

    /// Look up a specialized vendor bridge by its `ProviderKey.provider`
    /// vendor string. Used by handlers (chat preflight, background
    /// model checks) that need to confirm a vendor has a registered
    /// bridge without committing to a full dispatch. Returns `None`
    /// when no specialized override is registered — callers should
    /// then fall through to `family_bridge_for` for the catalog
    /// default path.
    pub fn get_specialized(&self, provider: &str) -> Option<Arc<dyn Bridge>> {
        self.specialized_bridges.get(provider).map(|r| r.clone())
    }

    /// Look up a family bridge by [`Adapter`]. Mirrors
    /// [`Hub::get_specialized`] for the family tier.
    pub fn family_bridge_for(&self, adapter: Adapter) -> Option<Arc<dyn Bridge>> {
        self.family_bridges.get(&adapter).map(|r| r.clone())
    }

    /// Two-tier dispatch: specialized vendor bridge first, then the
    /// adapter-family default. Returns `None` if neither is registered
    /// — the caller decides how to report a missing bridge so this
    /// layer stays panic-free.
    pub fn dispatch_two_tier(&self, pk: &ProviderKey) -> Option<Arc<dyn Bridge>> {
        if let Some(b) = self.specialized_bridges.get(&pk.provider) {
            return Some(b.clone());
        }
        let adapter = pk.adapter?;
        self.family_bridges.get(&adapter).map(|r| r.clone())
    }
}

impl std::fmt::Debug for Hub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let families: Vec<Adapter> = self.family_bridges.iter().map(|r| *r.key()).collect();
        let specialized: Vec<String> = self
            .specialized_bridges
            .iter()
            .map(|r| r.key().clone())
            .collect();
        f.debug_struct("Hub")
            .field("families", &families)
            .field("specialized", &specialized)
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

    #[tokio::test]
    async fn registered_bridge_is_callable() {
        let hub = Hub::new();
        hub.register_specialized("openai", Arc::new(StubBridge { name: "stub" }));
        let bridge = hub.get_specialized("openai").unwrap();

        let m = std::sync::Arc::new(
            serde_json::from_str::<aisix_core::Model>(
                r#"{"display_name":"t","provider":"openai","model_name":"gpt-4o","provider_key_id":"pk-1"}"#,
            )
            .unwrap(),
        );
        let pk = std::sync::Arc::new(
            serde_json::from_str::<aisix_core::ProviderKey>(
                r#"{"display_name":"pk","secret":"k"}"#,
            )
            .unwrap(),
        );
        let ctx = BridgeContext::new("req-1", m, pk);
        let req = ChatFormat::new("t", vec![ChatMessage::user("hi")]);

        let resp = bridge.chat(&req, &ctx).await.unwrap();
        assert_eq!(resp.message.content, "stubbed");
        assert_eq!(resp.finish_reason, FinishReason::Stop);
    }

    // ---- two-tier dispatch (issue #302 Phase A) ----

    /// Build a `ProviderKey` carrying just the vendor + adapter fields
    /// the two-tier dispatcher reads. JSON deserialization matches the
    /// existing test style and exercises the on-disk schema we expect
    /// future PRs to populate.
    fn pk(provider: &str, adapter: Option<&str>) -> aisix_core::ProviderKey {
        let adapter_field = match adapter {
            Some(a) => format!(r#","adapter":"{a}""#),
            None => String::new(),
        };
        let json = format!(
            r#"{{"display_name":"pk","secret":"k","provider":"{provider}"{adapter_field}}}"#
        );
        serde_json::from_str::<aisix_core::ProviderKey>(&json).unwrap()
    }

    #[test]
    fn register_family_and_dispatch_via_adapter() {
        let hub = Hub::new();
        hub.register_family(
            Adapter::Openai,
            Arc::new(StubBridge {
                name: "family-openai",
            }),
        );
        let b = hub
            .dispatch_two_tier(&pk("deepseek", Some("openai")))
            .unwrap();
        assert_eq!(b.name(), "family-openai");
    }

    #[test]
    fn register_specialized_and_dispatch_overrides_family() {
        let hub = Hub::new();
        hub.register_family(
            Adapter::Openai,
            Arc::new(StubBridge {
                name: "family-openai",
            }),
        );
        hub.register_specialized(
            "deepseek",
            Arc::new(StubBridge {
                name: "specialized-deepseek",
            }),
        );
        let b = hub
            .dispatch_two_tier(&pk("deepseek", Some("openai")))
            .unwrap();
        assert_eq!(b.name(), "specialized-deepseek");
    }

    #[test]
    fn dispatch_two_tier_returns_none_when_neither_registered() {
        let hub = Hub::new();
        assert!(hub
            .dispatch_two_tier(&pk("unknown", Some("openai")))
            .is_none());
    }

    #[test]
    fn dispatch_two_tier_returns_none_when_adapter_missing_and_no_specialized() {
        let hub = Hub::new();
        hub.register_family(
            Adapter::Openai,
            Arc::new(StubBridge {
                name: "family-openai",
            }),
        );
        // Old payloads / un-migrated keys land here: provider doesn't
        // match a specialized entry, and adapter is None so the family
        // tier has nothing to key on.
        assert!(hub.dispatch_two_tier(&pk("legacy", None)).is_none());
    }

    #[test]
    fn dispatch_two_tier_specialized_hits_even_when_adapter_missing() {
        let hub = Hub::new();
        hub.register_specialized(
            "jina",
            Arc::new(StubBridge {
                name: "specialized-jina",
            }),
        );
        // No adapter on the key, but the vendor string matches a
        // specialized registration — first tier still wins.
        let b = hub.dispatch_two_tier(&pk("jina", None)).unwrap();
        assert_eq!(b.name(), "specialized-jina");
    }

    #[test]
    fn register_family_overwrites_previous_entry() {
        let hub = Hub::new();
        hub.register_family(Adapter::Openai, Arc::new(StubBridge { name: "v1" }));
        hub.register_family(Adapter::Openai, Arc::new(StubBridge { name: "v2" }));
        let b = hub
            .dispatch_two_tier(&pk("anyvendor", Some("openai")))
            .unwrap();
        assert_eq!(b.name(), "v2");
    }

    #[test]
    fn register_specialized_overwrites_previous_entry() {
        let hub = Hub::new();
        hub.register_specialized("deepseek", Arc::new(StubBridge { name: "v1" }));
        hub.register_specialized("deepseek", Arc::new(StubBridge { name: "v2" }));
        let b = hub
            .dispatch_two_tier(&pk("deepseek", Some("openai")))
            .unwrap();
        assert_eq!(b.name(), "v2");
    }

    #[test]
    fn specialized_and_family_tiers_are_independent() {
        // A registration on one tier must not surface in the other.
        let hub = Hub::new();
        hub.register_family(Adapter::Openai, Arc::new(StubBridge { name: "family" }));
        hub.register_specialized(
            "deepseek",
            Arc::new(StubBridge {
                name: "specialized",
            }),
        );
        // Direct lookups return their own tier only.
        assert_eq!(
            hub.family_bridge_for(Adapter::Openai).unwrap().name(),
            "family"
        );
        assert_eq!(
            hub.get_specialized("deepseek").unwrap().name(),
            "specialized"
        );
        assert!(hub.family_bridge_for(Adapter::Anthropic).is_none());
        assert!(hub.get_specialized("openai").is_none());
    }
}
