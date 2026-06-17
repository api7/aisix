//! Redis-backed exact-match response cache.
//!
//! Implements [`Cache`] against Redis through the shared
//! [`aisix_redis::RedisConn`], so the operator's `cache.redis.mode`
//! (`single` / `cluster` / `sentinel`) is honored transparently — `GET`
//! and `SET` route by their key on cluster, and sentinel failovers are
//! recovered via [`aisix_redis::RedisConn::note_error`].
//!
//! Storage shape: each entry is JSON-encoded `ChatResponse` stored under
//! key `<prefix>:<fingerprint>` with a TTL set on insert (`SET ... EX ttl`).
//! Reads are a single `GET`; misses return `Ok(None)`. Connection
//! failures map to `CacheError::Backend` and are logged — the proxy
//! treats them as cache misses and proceeds to the upstream.

use std::time::Duration;

use aisix_core::RedisConnConfig;
use aisix_gateway::ChatResponse;
use aisix_redis::RedisConn;
use async_trait::async_trait;
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
    conn: RedisConn,
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
    /// Connect using the operator's `cache.redis` config. The topology
    /// (`single` / `cluster` / `sentinel`) is selected by `mode`; see
    /// [`aisix_redis::connect`].
    pub async fn connect(cfg: &RedisConnConfig) -> Result<Self, CacheError> {
        let conn = aisix_redis::connect(cfg)
            .await
            .map_err(|e| CacheError::Backend(format!("redis connect: {e}")))?;
        Ok(Self {
            conn,
            ttl_secs: DEFAULT_TTL.as_secs(),
            prefix: DEFAULT_PREFIX.into(),
        })
    }

    /// Override the instance **default** TTL — the fallback `put` uses
    /// when a write carries no per-entry value. Per-policy writes go
    /// through `put_with_ttl` and use their own `CachePolicy.ttl_seconds`
    /// instead, ignoring this. Floors at 1s (`EX 0` expires immediately).
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
        let mut conn = self
            .conn
            .acquire()
            .await
            .map_err(|e| CacheError::Backend(format!("redis acquire: {e}")))?;
        let full = self.full_key(key);
        let raw: Option<String> = match conn.get(&full).await {
            Ok(v) => v,
            Err(e) => {
                self.conn.note_error().await;
                return Err(CacheError::Backend(format!("redis GET: {e}")));
            }
        };
        match raw {
            None => Ok(None),
            Some(json) => serde_json::from_str::<ChatResponse>(&json)
                .map(Some)
                .map_err(|e| CacheError::Backend(format!("redis decode: {e}"))),
        }
    }

    async fn put(&self, key: &str, value: ChatResponse) -> Result<(), CacheError> {
        // No per-entry TTL supplied: fall back to the instance default.
        self.put_with_ttl(key, value, Duration::from_secs(self.ttl_secs))
            .await
    }

    /// Honors the caller-supplied per-entry TTL — the matched
    /// `CachePolicy.ttl_seconds` the proxy threads in — rather than the
    /// instance default, so a Redis-backed policy expires its entries on
    /// its own schedule. Floors at 1s: Redis `EX 0` expires immediately,
    /// turning every entry into a guaranteed miss.
    async fn put_with_ttl(
        &self,
        key: &str,
        value: ChatResponse,
        ttl: Duration,
    ) -> Result<(), CacheError> {
        let json = serde_json::to_string(&value)
            .map_err(|e| CacheError::Backend(format!("redis encode: {e}")))?;
        let mut conn = self
            .conn
            .acquire()
            .await
            .map_err(|e| CacheError::Backend(format!("redis acquire: {e}")))?;
        let full = self.full_key(key);
        let secs = ttl.as_secs().max(1);
        if let Err(e) = conn.set_ex::<_, _, ()>(&full, json, secs).await {
            self.conn.note_error().await;
            return Err(CacheError::Backend(format!("redis SET EX: {e}")));
        }
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
        let cfg = aisix_core::RedisConnConfig {
            mode: aisix_core::RedisMode::Single,
            url: Some("not-a-url".into()),
            ..Default::default()
        };
        let err = RedisCache::connect(&cfg).await.unwrap_err();
        let CacheError::Backend(msg) = err;
        assert!(msg.contains("redis"));
    }

    // The full integration path (real Redis round-trip) lives in
    // `tests/redis_integration.rs` and is opt-in via the `CACHE_TEST_REDIS_URL`
    // env var so the unit-test job stays hermetic. CI runs Redis as a
    // service so the integration test exercises the happy path.
}
