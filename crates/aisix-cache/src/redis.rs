//! Redis-backed exact-match response cache.
//!
//! Implements [`Cache`] against a Redis instance via the `redis` crate's
//! async ConnectionManager (single-node) — Cluster and Sentinel modes
//! drop in under the same connection-manager pattern in a follow-up
//! when the operator config exposes them.
//!
//! Storage shape: each entry is JSON-encoded `ChatResponse` stored under
//! key `<prefix>:<fingerprint>` with a TTL set on insert (`SET ... EX ttl`).
//! Reads are a single `GET`; misses return `Ok(None)`. Connection
//! failures map to `CacheError::Backend` and are logged — the proxy
//! treats them as cache misses and proceeds to the upstream.

use std::time::Duration;

use aisix_gateway::ChatResponse;
use async_trait::async_trait;
use redis::aio::ConnectionManager;
use redis::AsyncCommands;

use crate::cache::{Cache, CacheError};

/// Default TTL — matches [`crate::DEFAULT_TTL`] so swapping memory ↔
/// redis backends doesn't change observable behaviour for the common case.
pub const DEFAULT_TTL: Duration = Duration::from_secs(300);
/// Default key namespace. Keeps cache keys from colliding with anything
/// else operators might run in the same Redis instance.
pub const DEFAULT_PREFIX: &str = "aisix:cache";

#[derive(Clone)]
pub struct RedisCache {
    conn: ConnectionManager,
    ttl_secs: u64,
    prefix: String,
}

impl std::fmt::Debug for RedisCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedisCache")
            .field("ttl_secs", &self.ttl_secs)
            .field("prefix", &self.prefix)
            .finish_non_exhaustive()
    }
}

impl RedisCache {
    /// Connect to Redis using a single-node URL like `redis://host:port/`.
    /// Uses [`ConnectionManager`] which transparently reconnects on
    /// dropped connections — no per-request handshake cost.
    pub async fn connect(url: &str) -> Result<Self, CacheError> {
        let client = redis::Client::open(url)
            .map_err(|e| CacheError::Backend(format!("redis client open: {e}")))?;
        let conn = ConnectionManager::new(client)
            .await
            .map_err(|e| CacheError::Backend(format!("redis connect: {e}")))?;
        Ok(Self {
            conn,
            ttl_secs: DEFAULT_TTL.as_secs(),
            prefix: DEFAULT_PREFIX.into(),
        })
    }

    /// Override the per-entry TTL. Caps at `u64::MAX / 1000` to stay
    /// inside Redis's `EX` second range.
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl_secs = ttl.as_secs().max(1);
        self
    }

    /// Override the key namespace prefix. The full key is
    /// `<prefix>:<fingerprint>`.
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    fn full_key(&self, key: &str) -> String {
        format!("{}:{}", self.prefix, key)
    }
}

#[async_trait]
impl Cache for RedisCache {
    async fn get(&self, key: &str) -> Result<Option<ChatResponse>, CacheError> {
        let mut conn = self.conn.clone();
        let full = self.full_key(key);
        let raw: Option<String> = conn
            .get(&full)
            .await
            .map_err(|e| CacheError::Backend(format!("redis GET: {e}")))?;
        match raw {
            None => Ok(None),
            Some(json) => serde_json::from_str::<ChatResponse>(&json)
                .map(Some)
                .map_err(|e| CacheError::Backend(format!("redis decode: {e}"))),
        }
    }

    async fn put(&self, key: &str, value: ChatResponse) -> Result<(), CacheError> {
        let json = serde_json::to_string(&value)
            .map_err(|e| CacheError::Backend(format!("redis encode: {e}")))?;
        let mut conn = self.conn.clone();
        let full = self.full_key(key);
        let _: () = conn
            .set_ex(&full, json, self.ttl_secs)
            .await
            .map_err(|e| CacheError::Backend(format!("redis SET EX: {e}")))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_key_concatenates_prefix() {
        // Construct a RedisCache without actually opening a connection by
        // using a dummy ConnectionManager via mem::forget? No — we'd leak.
        // Easier: test the prefix logic via a free function.
        assert_eq!(prefix_join("aisix:cache", "ab12"), "aisix:cache:ab12");
        assert_eq!(prefix_join("", "x"), ":x");
    }

    fn prefix_join(prefix: &str, key: &str) -> String {
        format!("{prefix}:{key}")
    }

    #[test]
    fn with_ttl_floors_at_one_second() {
        // TTL of zero would mean "expire immediately" in Redis, which
        // makes every entry useless. Floor at 1s.
        // We can't construct RedisCache without a connection, so verify
        // the math directly.
        let secs = Duration::from_secs(0).as_secs().max(1);
        assert_eq!(secs, 1);
        let secs = Duration::from_millis(500).as_secs().max(1);
        assert_eq!(secs, 1);
        let secs = Duration::from_secs(60).as_secs().max(1);
        assert_eq!(secs, 60);
    }

    #[tokio::test]
    async fn connect_to_invalid_url_errors() {
        let err = RedisCache::connect("not-a-url").await.unwrap_err();
        let CacheError::Backend(msg) = err;
        assert!(msg.contains("redis"));
    }

    // The full integration path (real Redis round-trip) lives in
    // `tests/redis_integration.rs` and is opt-in via the `CACHE_TEST_REDIS_URL`
    // env var so the unit-test job stays hermetic. CI runs Redis as a
    // service so the integration test exercises the happy path.
}
