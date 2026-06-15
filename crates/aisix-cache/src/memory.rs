//! In-memory backend backed by `moka`.
//!
//! TTL strategy:
//! - The constructor's `ttl` argument is the **fallback** TTL — used
//!   when the proxy calls `put` (no per-policy override available).
//! - When the proxy calls `put_with_ttl` it ships the matching
//!   `CachePolicy::ttl_seconds`. Each entry then expires according to
//!   its own policy. moka's `Expiry` trait reads the per-entry TTL
//!   we stash next to the response.

use aisix_gateway::ChatResponse;
use async_trait::async_trait;
use moka::future::Cache as MokaCache;
use moka::Expiry;
use std::time::{Duration, Instant};

use crate::cache::{Cache, CacheError};

pub const DEFAULT_TTL: Duration = Duration::from_secs(300);
pub const DEFAULT_CAPACITY: u64 = 10_000;

/// What we actually store inside moka — the response plus the TTL the
/// caller asked for. The Expiry impl below reads the second field on
/// `expire_after_create` to set the per-entry deadline.
#[derive(Debug, Clone)]
struct Entry {
    response: ChatResponse,
    ttl: Duration,
}

#[derive(Debug)]
pub struct MemoryCache {
    inner: MokaCache<String, Entry>,
    /// Fallback TTL used by the no-override `put` path. Kept for the
    /// `ttl()` accessor + tests; not consulted by the Expiry impl.
    ttl: Duration,
}

/// Per-entry expiry that defers to the value's stashed `ttl`.
/// `expire_after_read` / `expire_after_update` return `None` so reads
/// don't extend an entry's life (semantic is "expires N seconds from
/// insert", not "expires N seconds from last access").
struct PerEntryExpiry;

impl Expiry<String, Entry> for PerEntryExpiry {
    fn expire_after_create(
        &self,
        _key: &String,
        value: &Entry,
        _current_time: Instant,
    ) -> Option<Duration> {
        Some(value.ttl)
    }
}

impl MemoryCache {
    pub fn new(ttl: Duration, capacity: u64) -> Self {
        let inner = MokaCache::builder()
            .max_capacity(capacity)
            .expire_after(PerEntryExpiry)
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
        Ok(self.inner.get(key).await.map(|e| e.response))
    }

    async fn put(&self, key: &str, value: ChatResponse) -> Result<(), CacheError> {
        self.inner
            .insert(
                key.to_string(),
                Entry {
                    response: value,
                    ttl: self.ttl,
                },
            )
            .await;
        Ok(())
    }

    async fn put_with_ttl(
        &self,
        key: &str,
        value: ChatResponse,
        ttl: Duration,
    ) -> Result<(), CacheError> {
        self.inner
            .insert(
                key.to_string(),
                Entry {
                    response: value,
                    ttl,
                },
            )
            .await;
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
        assert_eq!(got.message.content_str(), "hi back");
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
        updated.message.content = Some("second".into());
        cache.put("k", updated).await.unwrap();
        let got = cache.get("k").await.unwrap().unwrap();
        assert_eq!(got.message.content_str(), "second");
    }

    /// Per-entry TTL: two keys inserted at the same time with
    /// different TTLs must expire independently. Without the
    /// `Expiry` impl moka would use one global TTL and either both
    /// entries survive or both die — this test catches that
    /// regression.
    #[tokio::test]
    async fn put_with_ttl_uses_per_entry_expiry() {
        // Long-fallback cache so a regression that ignores the
        // per-entry TTL doesn't accidentally pass by global eviction.
        let cache = MemoryCache::new(Duration::from_secs(300), 100);
        cache
            .put_with_ttl("short", sample_response(), Duration::from_millis(50))
            .await
            .unwrap();
        cache
            .put_with_ttl("long", sample_response(), Duration::from_secs(60))
            .await
            .unwrap();

        // Both alive immediately after insert.
        assert!(cache.get("short").await.unwrap().is_some());
        assert!(cache.get("long").await.unwrap().is_some());

        // Wait past the short TTL only.
        tokio::time::sleep(Duration::from_millis(120)).await;
        cache.inner.run_pending_tasks().await;

        assert!(
            cache.get("short").await.unwrap().is_none(),
            "short-TTL entry should have expired",
        );
        assert!(
            cache.get("long").await.unwrap().is_some(),
            "long-TTL entry must survive past the short TTL",
        );
    }
}
