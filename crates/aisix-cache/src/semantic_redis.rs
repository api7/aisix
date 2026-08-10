//! Redis-backed semantic (embedding-similarity) store — the shared
//! counterpart of [`crate::MemorySemanticCache`], for `backend: redis`
//! policies. Requires a Redis server with the vector-search query
//! engine (Redis 8+; earlier versions need the search module). The
//! server bootstrap probes support once ([`RedisSemanticCache::probe`])
//! and only wires this store when the probe passes, so the per-request
//! path never re-checks capability.
//!
//! Layout per `(policy, generation, dims)`:
//! - one vector index `<prefix>:idx:<policy>:<gen>:<dims>` over HASH
//!   documents with prefix `<prefix>:doc:<policy>:<gen>:<dims>:`;
//! - one HASH document per exact wording, keyed by the exact-layer
//!   fingerprint — a refresh of the same wording overwrites in place
//!   (the upsert contract), and every document carries its own TTL, so
//!   growth is TTL-bounded and `max_entries` is intentionally ignored
//!   (see the trait docs).
//!
//! Purge / embedding-model changes rotate the index identity: a new
//! `(generation, dims)` gets a fresh index, and the previous one is
//! dropped (documents included) the next time this instance touches the
//! policy. Replicas that never touch it again leave only expired
//! documents plus an empty index definition behind.
//!
//! The candidate partition (`scope_fp`) is stored as a TAG field. TAG
//! values tokenize on punctuation, so the raw fingerprint string (which
//! contains `:` separators) is hashed to a plain hex token first.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

use aisix_core::RedisConnConfig;
use aisix_gateway::ChatResponse;
use aisix_redis::RedisConn;
use async_trait::async_trait;
use dashmap::DashMap;

use crate::cache::CacheError;
use crate::semantic::{SemanticCacheStore, SemanticHit};

/// Default key/index namespace. Distinct from the exact cache's
/// `aisix:cache` so `FT` index prefixes never overlap plain keys.
pub const DEFAULT_PREFIX: &str = "aisix:semcache";

/// One ensured index identity for a policy. A mismatch on either field
/// rotates the index (drop old, create new).
struct EnsuredIndex {
    generation: u32,
    dims: usize,
    index: String,
}

pub struct RedisSemanticCache {
    conn: RedisConn,
    prefix: String,
    /// Per-policy memo of the index this instance has ensured, so the
    /// hot path pays one `DashMap` read instead of an `FT.CREATE`
    /// round-trip per request.
    ensured: DashMap<String, EnsuredIndex>,
}

impl std::fmt::Debug for RedisSemanticCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedisSemanticCache")
            .field("prefix", &self.prefix)
            .finish_non_exhaustive()
    }
}

impl RedisSemanticCache {
    /// Connect using the operator's `cache.redis` config — the same
    /// config the exact redis cache uses; this opens its own
    /// connection so the two caches don't serialize on one pipeline.
    pub async fn connect(cfg: &RedisConnConfig) -> Result<Self, CacheError> {
        let conn = aisix_redis::connect(cfg)
            .await
            .map_err(|e| CacheError::Backend(format!("redis connect: {e}")))?;
        Ok(Self {
            conn,
            prefix: DEFAULT_PREFIX.into(),
            ensured: DashMap::new(),
        })
    }

    /// Namespace by environment id, mirroring the exact cache's
    /// isolation on a shared user-provided Redis.
    pub fn with_env_namespace(mut self, env_id: &str) -> Self {
        if !env_id.is_empty() {
            self.prefix = format!("{}:{}", self.prefix, env_id);
        }
        self
    }

    /// One-shot capability probe: succeeds iff the server speaks the
    /// vector-search command family. The bootstrap calls this once and
    /// skips wiring the store when it fails, so a plain Redis 6/7
    /// degrades semantic matching visibly at boot, never silently at
    /// request time.
    pub async fn probe(&self) -> Result<(), CacheError> {
        let mut conn = self.acquire().await?;
        redis::cmd("FT._LIST")
            .query_async::<()>(&mut conn)
            .await
            .map_err(|e| {
                CacheError::Backend(format!(
                    "vector search unsupported (requires Redis 8+ or the search module): {e}"
                ))
            })
    }

    async fn acquire(&self) -> Result<aisix_redis::RedisConnHandle, CacheError> {
        self.conn
            .acquire()
            .await
            .map_err(|e| CacheError::Backend(format!("redis acquire: {e}")))
    }

    fn index_name(&self, policy_id: &str, generation: u32, dims: usize) -> String {
        format!("{}:idx:{policy_id}:{generation}:{dims}", self.prefix)
    }

    fn doc_prefix(&self, policy_id: &str, generation: u32, dims: usize) -> String {
        format!("{}:doc:{policy_id}:{generation}:{dims}:", self.prefix)
    }

    /// TAG-safe token for the partition fingerprint: TAG fields
    /// tokenize on punctuation and the raw value contains separators,
    /// so it is hashed to bare hex (same 64-bit space as the cache
    /// fingerprints themselves).
    fn scope_tag(scope_fp: &str) -> String {
        let mut h = DefaultHasher::new();
        scope_fp.hash(&mut h);
        format!("{:016x}", h.finish())
    }

    /// Make sure the index for `(policy, generation, dims)` exists,
    /// rotating (drop + recreate) when the identity moved FORWARD since
    /// this instance last saw the policy. Returns `Ok(None)` for a
    /// STALE caller — one whose generation is lower than the ensured
    /// one (an in-flight request holding a pre-purge policy snapshot):
    /// rotating backwards would recreate the old index empty and then
    /// drop the live one, documents included. Stale lookups miss, stale
    /// stores are dropped, mirroring the in-process store's contract.
    async fn ensure_index(
        &self,
        policy_id: &str,
        generation: u32,
        dims: usize,
    ) -> Result<Option<String>, CacheError> {
        if let Some(e) = self.ensured.get(policy_id) {
            if e.generation == generation && e.dims == dims {
                return Ok(Some(e.index.clone()));
            }
            if generation < e.generation {
                return Ok(None);
            }
        }
        let index = self.index_name(policy_id, generation, dims);
        let doc_prefix = self.doc_prefix(policy_id, generation, dims);
        let mut conn = self.acquire().await?;
        let created = redis::cmd("FT.CREATE")
            .arg(&index)
            .arg("ON")
            .arg("HASH")
            .arg("PREFIX")
            .arg(1)
            .arg(&doc_prefix)
            .arg("SCHEMA")
            .arg("scope_fp")
            .arg("TAG")
            .arg("embedding")
            .arg("VECTOR")
            .arg("HNSW")
            .arg(6)
            .arg("TYPE")
            .arg("FLOAT32")
            .arg("DIM")
            .arg(dims)
            .arg("DISTANCE_METRIC")
            .arg("COSINE")
            .query_async::<()>(&mut conn)
            .await;
        match created {
            Ok(()) => {}
            // Another replica (or an earlier run) already created it.
            Err(e)
                if e.to_string()
                    .to_ascii_lowercase()
                    .contains("already exists") => {}
            Err(e) => {
                self.conn.note_error().await;
                return Err(CacheError::Backend(format!("redis FT.CREATE: {e}")));
            }
        }
        // Rotate away the previously ensured identity, documents
        // included — its entries belong to a purged generation or a
        // different vector space and must never serve again. Guarded by
        // the monotonic check above, so this only ever drops an OLDER
        // identity.
        if let Some((_, old)) = self.ensured.remove(policy_id) {
            if old.index != index {
                let _ = redis::cmd("FT.DROPINDEX")
                    .arg(&old.index)
                    .arg("DD")
                    .query_async::<()>(&mut conn)
                    .await;
            }
        }
        self.ensured.insert(
            policy_id.to_string(),
            EnsuredIndex {
                generation,
                dims,
                index: index.clone(),
            },
        );
        Ok(Some(index))
    }

    /// Forget a memoized index after the server reported it missing
    /// (`no such index`): the server lost state (restart / external
    /// cleanup / a concurrent boot sweep), so the next call must
    /// re-create instead of trusting the memo forever.
    fn forget_missing_index(&self, policy_id: &str, err: &CacheError) {
        let msg = err.to_string().to_ascii_lowercase();
        if msg.contains("no such index") || msg.contains("unknown index") {
            self.ensured.remove(policy_id);
        }
    }

    /// Boot-time GC: drop indexes under this instance's prefix whose
    /// document count is zero. Index DEFINITIONS never expire on their
    /// own, so deleted policies and rotated-away generations would
    /// otherwise accumulate empty definitions forever. Dropping a
    /// still-live empty index is safe: the next request that touches
    /// its policy re-creates it (the missing-index guard above clears
    /// any stale memo). Best-effort — errors are returned for the
    /// caller to log, never fatal.
    pub async fn sweep_empty_indexes(&self) -> Result<usize, CacheError> {
        let mut conn = self.acquire().await?;
        let names: Vec<String> = redis::cmd("FT._LIST")
            .query_async(&mut conn)
            .await
            .map_err(|e| CacheError::Backend(format!("redis FT._LIST: {e}")))?;
        let ours = format!("{}:idx:", self.prefix);
        let mut dropped = 0usize;
        for name in names.iter().filter(|n| n.starts_with(&ours)) {
            let info: redis::Value =
                match redis::cmd("FT.INFO").arg(name).query_async(&mut conn).await {
                    Ok(v) => v,
                    Err(_) => continue, // racing another sweep; skip
                };
            let redis::Value::Array(items) = info else {
                continue;
            };
            let mut it = items.iter();
            let mut num_docs: Option<i64> = None;
            while let (Some(k), Some(v)) = (it.next(), it.next()) {
                if value_name_eq(k, "num_docs") {
                    num_docs = match v {
                        redis::Value::Int(n) => Some(*n),
                        other => value_str(other).and_then(|s| s.parse().ok()),
                    };
                    break;
                }
            }
            if num_docs == Some(0)
                && redis::cmd("FT.DROPINDEX")
                    .arg(name)
                    .query_async::<()>(&mut conn)
                    .await
                    .is_ok()
            {
                dropped += 1;
            }
        }
        Ok(dropped)
    }
}

fn vector_blob(embedding: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(embedding.len() * 4);
    for f in embedding {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

/// Pull a named field out of an `FT.SEARCH` document field list
/// (`[name, value, name, value, …]`).
fn field<'a>(fields: &'a [redis::Value], name: &str) -> Option<&'a redis::Value> {
    let mut it = fields.iter();
    while let (Some(k), Some(v)) = (it.next(), it.next()) {
        if value_name_eq(k, name) {
            return Some(v);
        }
    }
    None
}

/// Field names in `FT.*` replies arrive as either simple or bulk
/// strings depending on the server version — match both.
fn value_name_eq(v: &redis::Value, name: &str) -> bool {
    match v {
        redis::Value::BulkString(b) => b == name.as_bytes(),
        redis::Value::SimpleString(s) => s == name,
        _ => false,
    }
}

fn value_str(v: &redis::Value) -> Option<String> {
    match v {
        redis::Value::BulkString(b) => String::from_utf8(b.clone()).ok(),
        redis::Value::SimpleString(s) => Some(s.clone()),
        _ => None,
    }
}

#[async_trait]
impl SemanticCacheStore for RedisSemanticCache {
    async fn lookup(
        &self,
        policy_id: &str,
        generation: u32,
        scope_fp: &str,
        embedding: &[f32],
        threshold: f32,
    ) -> Result<Option<SemanticHit>, CacheError> {
        let Some(index) = self
            .ensure_index(policy_id, generation, embedding.len())
            .await?
        else {
            // Stale generation: miss, never rotate backwards.
            return Ok(None);
        };
        let mut conn = self.acquire().await?;
        let query = format!(
            "(@scope_fp:{{{}}})=>[KNN 1 @embedding $vec AS dist]",
            Self::scope_tag(scope_fp)
        );
        let reply = redis::cmd("FT.SEARCH")
            .arg(&index)
            .arg(&query)
            .arg("PARAMS")
            .arg(2)
            .arg("vec")
            .arg(vector_blob(embedding))
            .arg("SORTBY")
            .arg("dist")
            .arg("RETURN")
            .arg(2)
            .arg("response")
            .arg("dist")
            .arg("LIMIT")
            .arg(0)
            .arg(1)
            .arg("DIALECT")
            .arg(2)
            .query_async::<redis::Value>(&mut conn)
            .await;
        let reply = match reply {
            Ok(v) => v,
            Err(e) => {
                self.conn.note_error().await;
                let err = CacheError::Backend(format!("redis FT.SEARCH: {e}"));
                self.forget_missing_index(policy_id, &err);
                return Err(err);
            }
        };
        // Reply shape: [total, key, [field, value, …], …].
        let redis::Value::Array(items) = reply else {
            return Err(CacheError::Backend(
                "redis FT.SEARCH: non-array reply".into(),
            ));
        };
        let (Some(doc_key), Some(redis::Value::Array(fields))) = (items.get(1), items.get(2))
        else {
            return Ok(None); // total == 0
        };
        let dist: f32 = field(fields, "dist")
            .and_then(value_str)
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| CacheError::Backend("redis FT.SEARCH: missing dist".into()))?;
        let similarity = 1.0 - dist;
        if !(similarity >= threshold && similarity > 0.0) {
            return Ok(None);
        }
        let response: ChatResponse = field(fields, "response")
            .and_then(value_str)
            .and_then(|s| serde_json::from_str(&s).ok())
            .ok_or_else(|| CacheError::Backend("redis FT.SEARCH: bad response field".into()))?;
        // Remaining lifetime for the exact-layer backfill cap.
        let doc_key = value_str(doc_key)
            .ok_or_else(|| CacheError::Backend("redis FT.SEARCH: bad document key".into()))?;
        let pttl_ms: i64 = redis::cmd("PTTL")
            .arg(&doc_key)
            .query_async(&mut conn)
            .await
            .map_err(|e| CacheError::Backend(format!("redis PTTL: {e}")))?;
        let expires_at = Instant::now() + Duration::from_millis(pttl_ms.max(0) as u64);
        Ok(Some(SemanticHit {
            response,
            similarity,
            expires_at,
        }))
    }

    async fn store(
        &self,
        policy_id: &str,
        generation: u32,
        scope_fp: &str,
        exact_key: &str,
        embedding: Vec<f32>,
        response: ChatResponse,
        ttl: Duration,
        _max_entries: u32,
    ) -> Result<(), CacheError> {
        let dims = embedding.len();
        if self
            .ensure_index(policy_id, generation, dims)
            .await?
            .is_none()
        {
            // Stale generation: drop the write, mirroring the
            // in-process store.
            return Ok(());
        }
        let doc_key = format!(
            "{}{exact_key}",
            self.doc_prefix(policy_id, generation, dims)
        );
        let json = serde_json::to_string(&response)
            .map_err(|e| CacheError::Backend(format!("redis encode: {e}")))?;
        let mut conn = self.acquire().await?;
        let mut pipe = redis::pipe();
        pipe.cmd("HSET")
            .arg(&doc_key)
            .arg("scope_fp")
            .arg(Self::scope_tag(scope_fp))
            .arg("response")
            .arg(json)
            .arg("embedding")
            .arg(vector_blob(&embedding))
            .ignore()
            .cmd("PEXPIRE")
            .arg(&doc_key)
            .arg(ttl.as_millis().max(1) as u64)
            .ignore();
        if let Err(e) = pipe.query_async::<()>(&mut conn).await {
            self.conn.note_error().await;
            let err = CacheError::Backend(format!("redis HSET: {e}"));
            self.forget_missing_index(policy_id, &err);
            return Err(err);
        }
        Ok(())
    }
}
