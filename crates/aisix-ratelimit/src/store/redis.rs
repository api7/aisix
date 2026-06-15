//! Redis-backed shared counter store — the fix for api7/AISIX-Cloud#798.
//!
//! Every DP replica points at the same Redis, so one global window is
//! enforced across the cluster instead of one-per-replica. The counter
//! math mirrors [`super::local::LocalStore`] / [`crate::window`] exactly
//! (wall-clock-aligned fixed windows: `window_start = now - now % window`),
//! so swapping `memory ↔ redis` doesn't change observable limits — only
//! whether the count is shared.
//!
//! Key layout (hash-tagged so every key for a bucket shares one Redis
//! Cluster slot, keeping the per-bucket Lua atomic):
//! - `aisix:rl:{<bucket>}:<rps|rpm|rph|rpd|tpm|tpd>:<window_start>` — plain
//!   `INCR`/`GET` counters, `EXPIRE = window + grace`.
//! - `aisix:rl:{<bucket>}:conc` — a ZSET (`member → score=now`) acting as a
//!   crash-safe distributed semaphore: acquire prunes entries older than
//!   `conc_ttl` then counts, so a slot leaked by a crashed/hung replica is
//!   reclaimed within `conc_ttl`. (LiteLLM's latest tracks parallel
//!   requests as a window-TTL counter; we use a ZSET with a request-
//!   lifetime ttl because our streaming requests can outlive a 60s window
//!   — the same reason `StreamConcurrencyGuard`/#450 exists.)
//!
//! `now` is read from `redis.call('TIME')` inside every script so window
//! boundaries are identical across replicas regardless of host clock skew.
//!
//! On any Redis error the store fails **open** to a per-process
//! [`LocalStore`] (logged once): traffic keeps flowing with per-replica
//! enforcement during an outage instead of being blocked. Counts may
//! diverge from Redis until it recovers — availability over strict global
//! enforcement, matching the cache's "Redis error → proceed" stance.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use aisix_core::RateLimit;
use async_trait::async_trait;
use redis::aio::ConnectionManager;
use redis::Script;

use super::{local::LocalStore, token_dims, Dim, RateStore};
use crate::error::RateLimitError;
use crate::limiter::RateLimitStatus;

/// Default key namespace, kept distinct from `aisix:cache` so the same
/// Redis can back both the response cache and the rate limiter.
pub const DEFAULT_PREFIX: &str = "aisix:rl";
/// Default concurrency-slot lifetime ceiling. A slot not released within
/// this many seconds (crashed replica, hung upstream) is pruned from the
/// count. Generous enough to cover a long streaming response.
pub const DEFAULT_CONC_TTL_SECS: u64 = 300;
/// Extra seconds added to each window counter's TTL so a counter doesn't
/// expire a hair before its window mathematically closes.
pub const DEFAULT_GRACE_SECS: u64 = 5;

// Result codes returned by the acquire script's first element.
const CODE_OK: i64 = 0;
const CODE_CONCURRENCY: i64 = 1;
const CODE_TOKENS: i64 = 2;
const CODE_REQUESTS: i64 = 3;

/// Atomic per-bucket acquire: concurrency gate + token check-only +
/// request check-and-increment, all-or-nothing. See module docs for the
/// key layout. Returns `{code, retry_after}`.
const ACQUIRE_LUA: &str = r#"
local prefix = ARGV[1]
local member = ARGV[2]
local conc_max = tonumber(ARGV[3])
local conc_ttl = tonumber(ARGV[4])
local grace = tonumber(ARGV[5])
local t = redis.call('TIME')
local now = tonumber(t[1])

local conc_key = prefix .. ':conc'
if conc_max >= 0 then
  redis.call('ZREMRANGEBYSCORE', conc_key, 0, now - conc_ttl)
  if redis.call('ZCARD', conc_key) >= conc_max then
    return {1, 0}
  end
end

local idx = 6
local nreq = tonumber(ARGV[idx]); idx = idx + 1
local req = {}
for i = 1, nreq do
  req[i] = {ARGV[idx], tonumber(ARGV[idx+1]), tonumber(ARGV[idx+2])}
  idx = idx + 3
end
local ntok = tonumber(ARGV[idx]); idx = idx + 1
local tok = {}
for i = 1, ntok do
  tok[i] = {ARGV[idx], tonumber(ARGV[idx+1]), tonumber(ARGV[idx+2])}
  idx = idx + 3
end

for i = 1, ntok do
  local name, window, limit = tok[i][1], tok[i][2], tok[i][3]
  local ws = now - (now % window)
  local cur = tonumber(redis.call('GET', prefix .. ':' .. name .. ':' .. ws) or '0')
  if cur > limit then
    local retry = window - (now - ws); if retry < 1 then retry = 1 end
    return {2, retry}
  end
end

for i = 1, nreq do
  local name, window, limit = req[i][1], req[i][2], req[i][3]
  local ws = now - (now % window)
  local cur = tonumber(redis.call('GET', prefix .. ':' .. name .. ':' .. ws) or '0')
  if cur + 1 > limit then
    local retry = window - (now - ws); if retry < 1 then retry = 1 end
    return {3, retry}
  end
end

for i = 1, nreq do
  local name, window = req[i][1], req[i][2]
  local ws = now - (now % window)
  local k = prefix .. ':' .. name .. ':' .. ws
  if redis.call('INCR', k) == 1 then
    redis.call('EXPIRE', k, window + grace)
  end
end
if conc_max >= 0 then
  redis.call('ZADD', conc_key, now, member)
  redis.call('EXPIRE', conc_key, conc_ttl)
end
return {0, 0}
"#;

/// Post-deduct: add `tokens` to the tpm/tpd windows AND release the
/// concurrency slot held by `member`. Both token windows are always
/// touched (matching the local backend); an unread tpd counter just
/// expires. ARGV: prefix, member, tokens, grace.
const COMMIT_LUA: &str = r#"
local prefix = ARGV[1]
local member = ARGV[2]
local tokens = tonumber(ARGV[3])
local grace = tonumber(ARGV[4])
local t = redis.call('TIME')
local now = tonumber(t[1])
if tokens > 0 then
  for _, d in ipairs({{'tpm', 60}, {'tpd', 86400}}) do
    local ws = now - (now % d[2])
    local k = prefix .. ':' .. d[1] .. ':' .. ws
    if redis.call('INCRBY', k, tokens) == tokens then
      redis.call('EXPIRE', k, d[2] + grace)
    end
  end
end
redis.call('ZREM', prefix .. ':conc', member)
return 1
"#;

/// Post-stream token add only (no concurrency change). ARGV: prefix,
/// tokens, grace.
const ADD_TOKENS_LUA: &str = r#"
local prefix = ARGV[1]
local tokens = tonumber(ARGV[2])
local grace = tonumber(ARGV[3])
local t = redis.call('TIME')
local now = tonumber(t[1])
if tokens > 0 then
  for _, d in ipairs({{'tpm', 60}, {'tpd', 86400}}) do
    local ws = now - (now % d[2])
    local k = prefix .. ':' .. d[1] .. ':' .. ws
    if redis.call('INCRBY', k, tokens) == tokens then
      redis.call('EXPIRE', k, d[2] + grace)
    end
  end
end
return 1
"#;

/// Read-only snapshot for headers: current-minute rpm/tpm counts +
/// pruned concurrency count + seconds-to-minute-reset. ARGV: prefix,
/// conc_ttl. Returns {rpm_used, tpm_used, in_flight, minute_reset}.
const PEEK_LUA: &str = r#"
local prefix = ARGV[1]
local conc_ttl = tonumber(ARGV[2])
local t = redis.call('TIME')
local now = tonumber(t[1])
local ws = now - (now % 60)
local rpm = tonumber(redis.call('GET', prefix .. ':rpm:' .. ws) or '0')
local tpm = tonumber(redis.call('GET', prefix .. ':tpm:' .. ws) or '0')
redis.call('ZREMRANGEBYSCORE', prefix .. ':conc', 0, now - conc_ttl)
local inflight = redis.call('ZCARD', prefix .. ':conc')
return {rpm, tpm, inflight, 60 - (now % 60)}
"#;

pub struct RedisStore {
    conn: ConnectionManager,
    prefix: String,
    conc_ttl: u64,
    grace: u64,
    /// Per-process fallback used when Redis is unreachable (fail-open).
    local: Arc<LocalStore>,
    /// One-shot guard so the degradation warning is logged once, not per
    /// request, while Redis stays down.
    degraded_logged: AtomicBool,
}

impl std::fmt::Debug for RedisStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedisStore")
            .field("prefix", &self.prefix)
            .field("conc_ttl", &self.conc_ttl)
            .finish_non_exhaustive()
    }
}

impl RedisStore {
    /// Connect to Redis via [`ConnectionManager`] (transparent reconnect,
    /// no per-request handshake). Single-node URL like `redis://host:port/`.
    pub async fn connect(url: &str) -> Result<Self, redis::RedisError> {
        let client = redis::Client::open(url)?;
        let conn = ConnectionManager::new(client).await?;
        Ok(Self {
            conn,
            prefix: DEFAULT_PREFIX.into(),
            conc_ttl: DEFAULT_CONC_TTL_SECS,
            grace: DEFAULT_GRACE_SECS,
            local: Arc::new(LocalStore::new()),
            degraded_logged: AtomicBool::new(false),
        })
    }

    pub fn with_conc_ttl(mut self, secs: u64) -> Self {
        self.conc_ttl = secs.max(1);
        self
    }

    /// `aisix:rl:{<bucket>}` — the hash tag co-locates every key for the
    /// bucket on one Redis Cluster slot.
    fn bucket_prefix(&self, key: &str) -> String {
        format!("{}:{{{}}}", self.prefix, key)
    }

    /// Log the fail-open transition once per outage.
    fn warn_degraded(&self, op: &str, err: &redis::RedisError) {
        if !self.degraded_logged.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                target: "aisix::ratelimit",
                op,
                error = %err,
                "shared rate-limit Redis unavailable; failing open to per-replica \
                 in-memory counting until it recovers (cluster limits not enforced \
                 during the outage)"
            );
        }
    }

    /// Mark Redis healthy again after a successful op (re-arms the warn).
    fn mark_ok(&self) {
        self.degraded_logged.store(false, Ordering::Relaxed);
    }
}

fn push_dims(args: &mut Vec<String>, dims: &[Dim]) {
    args.push(dims.len().to_string());
    for d in dims {
        args.push(d.name.to_string());
        args.push(d.window_secs.to_string());
        args.push(d.limit.to_string());
    }
}

#[async_trait]
impl RateStore for RedisStore {
    async fn acquire(
        &self,
        key: &str,
        limits: &RateLimit,
        member: &str,
    ) -> Result<(), RateLimitError> {
        let prefix = self.bucket_prefix(key);
        let mut args = vec![
            prefix,
            member.to_string(),
            limits.concurrency.map(i64::from).unwrap_or(-1).to_string(),
            self.conc_ttl.to_string(),
            self.grace.to_string(),
        ];
        push_dims(&mut args, &super::request_dims(limits));
        push_dims(&mut args, &token_dims(limits));

        let script = Script::new(ACQUIRE_LUA);
        let mut invocation = script.prepare_invoke();
        for a in &args {
            invocation.arg(a);
        }
        let mut conn = self.conn.clone();
        match invocation.invoke_async::<Vec<i64>>(&mut conn).await {
            Ok(reply) => {
                self.mark_ok();
                let code = reply.first().copied().unwrap_or(CODE_OK);
                let retry = reply.get(1).copied().unwrap_or(0).max(0) as u64;
                match code {
                    CODE_OK => Ok(()),
                    CODE_CONCURRENCY => Err(RateLimitError::Concurrency),
                    CODE_TOKENS => Err(RateLimitError::Tokens {
                        scope: aisix_core::RateLimitScope::Tokens,
                        retry_after_secs: retry,
                    }),
                    CODE_REQUESTS => Err(RateLimitError::Requests {
                        scope: aisix_core::RateLimitScope::Requests,
                        retry_after_secs: retry,
                    }),
                    _ => Ok(()),
                }
            }
            Err(e) => {
                self.warn_degraded("acquire", &e);
                self.local.acquire(key, limits, member).await
            }
        }
    }

    async fn commit(&self, key: &str, tokens: u64, member: &str) {
        let prefix = self.bucket_prefix(key);
        let mut conn = self.conn.clone();
        let res: Result<i64, redis::RedisError> = Script::new(COMMIT_LUA)
            .arg(&prefix)
            .arg(member)
            .arg(tokens)
            .arg(self.grace)
            .invoke_async(&mut conn)
            .await;
        match res {
            Ok(_) => self.mark_ok(),
            Err(e) => {
                self.warn_degraded("commit", &e);
                self.local.commit(key, tokens, member).await;
            }
        }
    }

    fn release(&self, key: &str, member: &str) {
        // Drop the local slot first (a cheap no-op when the bucket was
        // never acquired locally); covers the degraded-acquire case.
        self.local.release(key, member);
        let conc_key = format!("{}:conc", self.bucket_prefix(key));
        let mut conn = self.conn.clone();
        let member = member.to_string();
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                let _: Result<(), redis::RedisError> = redis::cmd("ZREM")
                    .arg(&conc_key)
                    .arg(&member)
                    .query_async(&mut conn)
                    .await;
            });
        }
    }

    fn add_tokens(&self, key: &str, tokens: u64) {
        if tokens == 0 {
            return;
        }
        // Best-effort fire-and-forget to Redis; also record locally so the
        // count is right if a later request falls back during an outage.
        self.local.add_tokens(key, tokens);
        let prefix = self.bucket_prefix(key);
        let grace = self.grace;
        let mut conn = self.conn.clone();
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                let _: Result<i64, redis::RedisError> = Script::new(ADD_TOKENS_LUA)
                    .arg(&prefix)
                    .arg(tokens)
                    .arg(grace)
                    .invoke_async(&mut conn)
                    .await;
            });
        }
    }

    async fn peek(&self, key: &str, limits: &RateLimit) -> Option<RateLimitStatus> {
        if limits.is_unrestricted() {
            return None;
        }
        let prefix = self.bucket_prefix(key);
        let mut conn = self.conn.clone();
        let reply: Result<Vec<i64>, redis::RedisError> = Script::new(PEEK_LUA)
            .arg(&prefix)
            .arg(self.conc_ttl)
            .invoke_async(&mut conn)
            .await;
        match reply {
            Ok(v) => {
                self.mark_ok();
                Some(RateLimitStatus {
                    rpm_limit: limits.rpm,
                    rpm_used: v.first().copied().unwrap_or(0).max(0) as u64,
                    rpm_reset_secs: v.get(3).copied().unwrap_or(0).max(0) as u64,
                    tpm_limit: limits.tpm,
                    tpm_used: v.get(1).copied().unwrap_or(0).max(0) as u64,
                    tpm_reset_secs: v.get(3).copied().unwrap_or(0).max(0) as u64,
                    concurrency_limit: limits.concurrency,
                    in_flight: v.get(2).copied().unwrap_or(0).max(0) as u32,
                })
            }
            Err(e) => {
                self.warn_degraded("peek", &e);
                self.local.peek(key, limits).await
            }
        }
    }
}
