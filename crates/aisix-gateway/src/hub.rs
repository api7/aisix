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
//!
//! # Two-tier dispatch skeleton (issue #302 Phase A)
//!
//! In addition to the existing `Provider`-keyed registry, the Hub now
//! also carries two forward-looking maps keyed by [`ProviderKey`]
//! fields introduced in Phase A:
//!
//! - `family_bridges` — keyed on `Adapter` (wire shape: `openai`,
//!   `anthropic`, `bedrock`, `vertex`, `azure-openai`). The default
//!   bridge for any provider that matches that wire shape.
//! - `specialized_bridges` — keyed on `ProviderKey.provider` (vendor
//!   string, e.g. `"deepseek"`, `"jina"`). Used when a specific vendor
//!   needs handling that diverges from its wire-shape default.
//!
//! [`Hub::dispatch_two_tier`] looks up specialized first, then falls
//! back to the family bridge. It is **not consumed by any caller** in
//! this PR — `build_hub()` does not register family/specialized
//! bridges and the proxy path continues to dispatch through the
//! existing `Provider`-keyed `get()`. The new methods exist so future
//! Phase A / Phase D sub-PRs can wire dispatch through them without
//! re-touching this file.

use aisix_core::models::{Adapter, Provider, ProviderKey};
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
    /// Wire-shape default bridges. Keyed on [`Adapter`]. Phase A
    /// skeleton — empty until follow-up PRs register the per-family
    /// default bridge. See module docs.
    family_bridges: DashMap<Adapter, Arc<dyn Bridge>>,
    /// Vendor-specific override bridges. Keyed on
    /// `ProviderKey.provider` (vendor string). Phase A skeleton —
    /// empty until follow-up PRs register specialized handlers for
    /// vendors that diverge from their wire-shape default. See module
    /// docs.
    specialized_bridges: DashMap<String, Arc<dyn Bridge>>,
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

    /// Register the family-tier (wire-shape default) bridge for an
    /// [`Adapter`]. Phase A skeleton — see module docs.
    ///
    /// Overwrites any previous entry for the same adapter, matching
    /// the live-reconfigure semantics of [`Hub::register`].
    pub fn register_family(&self, adapter: Adapter, bridge: Arc<dyn Bridge>) {
        self.family_bridges.insert(adapter, bridge);
    }

    /// Register a specialized (vendor-specific) bridge keyed on the
    /// `ProviderKey.provider` vendor string. Phase A skeleton — see
    /// module docs.
    ///
    /// Overwrites any previous entry for the same vendor key.
    pub fn register_specialized(&self, provider: impl Into<String>, bridge: Arc<dyn Bridge>) {
        self.specialized_bridges.insert(provider.into(), bridge);
    }

    /// Two-tier dispatch: specialized vendor bridge first, then the
    /// adapter-family default. Returns `None` if neither is registered
    /// — the caller decides how to report a missing bridge so this
    /// layer stays panic-free. Phase A skeleton — see module docs.
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
            .field("providers", &self.providers())
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

    // ---- two-tier dispatch (issue #302 Phase A skeleton) ----

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
    fn legacy_provider_registry_is_unaffected_by_two_tier_maps() {
        // Registering only on the new tiers must not satisfy a lookup
        // through the legacy `Provider`-keyed API, and vice versa.
        // Co-existence is the explicit contract of this skeleton PR.
        let hub = Hub::new();
        hub.register_family(Adapter::Openai, Arc::new(StubBridge { name: "family" }));
        hub.register_specialized(
            "deepseek",
            Arc::new(StubBridge {
                name: "specialized",
            }),
        );
        assert!(hub.get(Provider::Openai).is_none());
        assert!(hub.is_empty());

        hub.register(Provider::Openai, Arc::new(StubBridge { name: "legacy" }));
        // Legacy bump touches only the Provider-keyed map.
        assert_eq!(hub.get(Provider::Openai).unwrap().name(), "legacy");
        assert_eq!(hub.len(), 1);
        // Two-tier maps still resolve their own registrations
        // independently.
        assert_eq!(
            hub.dispatch_two_tier(&pk("deepseek", Some("openai")))
                .unwrap()
                .name(),
            "specialized"
        );
    }
}
