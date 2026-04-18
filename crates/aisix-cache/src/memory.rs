//! In-memory backend backed by `moka`.
//!
//! TTL is per-cache (set at construction); per-entry overrides are not
//! supported by moka's TTL eviction strategy, which fits our model —
//! every entry expires the same way.

use aisix_gateway::ChatResponse;
use async_trait::async_trait;
use moka::future::Cache as MokaCache;
use std::time::Duration;

use crate::cache::{Cache, CacheError};

pub const DEFAULT_TTL: Duration = Duration::from_secs(300);
pub const DEFAULT_CAPACITY: u64 = 10_000;

#[derive(Debug)]
pub struct MemoryCache {
    inner: MokaCache<String, ChatResponse>,
    ttl: Duration,
}

impl MemoryCache {
    pub fn new(ttl: Duration, capacity: u64) -> Self {
        let inner = MokaCache::builder()
            .max_capacity(capacity)
            .time_to_live(ttl)
            .build();
        Self { inner, ttl }
    }

    pub fn with_defaults() -> Self {
        Self::new(DEFAULT_TTL, DEFAULT_CAPACITY)
    }

    pub fn ttl(&self) -> Duration {
        self.ttl
    }
}

impl Default for MemoryCache {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[async_trait]
impl Cache for MemoryCache {
    async fn get(&self, key: &str) -> Result<Option<ChatResponse>, CacheError> {
        Ok(self.inner.get(key).await)
    }

    async fn put(&self, key: &str, value: ChatResponse) -> Result<(), CacheError> {
        self.inner.insert(key.to_string(), value).await;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_gateway::{ChatMessage, FinishReason, UsageStats};

    fn sample_response() -> ChatResponse {
        ChatResponse {
            id: "cmpl-1".into(),
            model: "m".into(),
            message: ChatMessage::assistant("hi back"),
            finish_reason: FinishReason::Stop,
            usage: UsageStats::new(2, 3),
        }
    }

    #[tokio::test]
    async fn put_then_get_round_trips() {
        let cache = MemoryCache::with_defaults();
        cache.put("k1", sample_response()).await.unwrap();
        let got = cache.get("k1").await.unwrap().unwrap();
        assert_eq!(got.message.content, "hi back");
        assert_eq!(got.usage.total_tokens, 5);
    }

    #[tokio::test]
    async fn get_for_missing_key_returns_none() {
        let cache = MemoryCache::with_defaults();
        assert!(cache.get("absent").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn ttl_eviction_drops_stale_entries() {
        let cache = MemoryCache::new(Duration::from_millis(50), 100);
        cache.put("k1", sample_response()).await.unwrap();
        assert!(cache.get("k1").await.unwrap().is_some());
        // Wait past TTL. Moka uses lazy eviction on read; one extra
        // milli of slack to clear the boundary.
        tokio::time::sleep(Duration::from_millis(120)).await;
        // Force housekeeping so the test isn't dependent on the random
        // background eviction tick.
        cache.inner.run_pending_tasks().await;
        assert!(cache.get("k1").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn put_overwrites_previous_value_for_same_key() {
        let cache = MemoryCache::with_defaults();
        cache.put("k", sample_response()).await.unwrap();
        let mut updated = sample_response();
        updated.message.content = "second".into();
        cache.put("k", updated).await.unwrap();
        let got = cache.get("k").await.unwrap().unwrap();
        assert_eq!(got.message.content, "second");
    }
}
