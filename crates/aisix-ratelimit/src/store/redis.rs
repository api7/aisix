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
//!
//! The connection is the shared [`aisix_redis::RedisConn`], so `single`,
//! `cluster`, and `sentinel` topologies all work. Because every script
//! touches several keys, each `EVAL` declares the bucket key (carrying the
//! `{<bucket>}` hash tag) so Redis Cluster routes it to the slot that owns
//! every key the script reads/writes; on single/sentinel the declared key
//! is simply ignored by the script body, which still reads the prefix from
//! `ARGV[1]`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use aisix_core::{RateLimit, RedisConnConfig};
use aisix_obs::metrics::Metrics;
use aisix_redis::{FailurePolicy, RedisConn, RedisConnHandle};
use arc_swap::ArcSwapOption;
use async_trait::async_trait;
use redis::Script;

use super::{local::LocalStore, token_dims, Dim, RateStore};
use crate::error::{LimitDetail, RateLimitError};
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
/// How often a gateway that started without its shared backend re-states
/// that it is still counting per replica. Long enough that a Redis which
/// is merely late says nothing beyond the boot WARN, short enough that a
/// permanent misconfiguration is greppable in any retained log window.
const DEGRADED_REMINDER: std::time::Duration = std::time::Duration::from_secs(300);

// Result codes returned by the acquire script's first element.
/// Dimension name reported when the script names an index this build
/// no longer knows — only reachable if the script and the caller ever
/// disagree on the dimension list, which they cannot today (both come
/// from the same `request_dims` / `token_dims` call).
const UNKNOWN_DIMENSION: &str = "unknown";

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
  local in_flight = redis.call('ZCARD', conc_key)
  if in_flight >= conc_max then
    return {1, 0, 0, conc_max, in_flight}
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
  -- `>=`, matching the local backend: a token window whose committed
  -- usage has reached the cap is spent, not still admissible (#950).
  if cur >= limit then
    local retry = window - (now - ws); if retry < 1 then retry = 1 end
    return {2, retry, i, limit, cur}
  end
end

for i = 1, nreq do
  local name, window, limit = req[i][1], req[i][2], req[i][3]
  local ws = now - (now % window)
  local cur = tonumber(redis.call('GET', prefix .. ':' .. name .. ':' .. ws) or '0')
  if cur + 1 > limit then
    local retry = window - (now - ws); if retry < 1 then retry = 1 end
    return {3, retry, i, limit, cur}
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
return {0, 0, 0, 0, 0}
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

/// The connection the store runs its commands on, held in a slot rather
/// than inline because it may not exist yet.
///
/// A Redis that is unreachable when the gateway starts must not keep it
/// from binding its listeners, so boot carries on with the slot empty and
/// a background task fills it in when Redis answers.
/// An empty slot is the state a mid-flight outage already puts this store
/// in — every operation gets a connectivity error and runs on the
/// per-replica fallback — which is why it needs no branch of its own
/// anywhere below.
#[derive(Clone)]
struct ConnSlot(Arc<ArcSwapOption<RedisConn>>);

impl ConnSlot {
    fn filled(conn: RedisConn) -> Self {
        Self(Arc::new(ArcSwapOption::from_pointee(conn)))
    }

    fn empty() -> Self {
        Self(Arc::new(ArcSwapOption::empty()))
    }

    fn attach(&self, conn: RedisConn) {
        self.0.store(Some(Arc::new(conn)));
    }

    /// `ArcSwapOption` rather than a lock around the value: cloning a
    /// `RedisConn` out on every operation is a deep clone of the
    /// driver's connection info (host, username, password), and
    /// `RedisConn::acquire` already makes one of those internally.
    fn get(&self) -> Option<Arc<RedisConn>> {
        self.0.load_full()
    }

    /// A live handle, or the error that makes the caller fail open.
    async fn acquire(&self) -> Result<RedisConnHandle, redis::RedisError> {
        match self.get() {
            Some(conn) => conn.acquire().await,
            None => Err(aisix_redis::not_connected_error()),
        }
    }

    async fn note_error(&self) {
        if let Some(conn) = self.get() {
            conn.note_error().await;
        }
    }
}

pub struct RedisStore {
    conn: ConnSlot,
    prefix: String,
    conc_ttl: u64,
    grace: u64,
    /// Per-process fallback used when Redis is unreachable (fail-open).
    local: Arc<LocalStore>,
    /// One-shot guard so the degradation warning is logged once, not per
    /// request, while Redis stays down. Shared with the boot attach task,
    /// which starts it already set (the boot WARN said it) and clears it
    /// on attach so the NEXT outage still warns even if no store
    /// operation succeeded in between.
    degraded_logged: Arc<AtomicBool>,
    /// Prometheus handle for `aisix_redis_failures_total`. The warning
    /// above fires once per outage by design, so it cannot answer how
    /// long or how hard Redis is failing; the counter can.
    metrics: Option<Metrics>,
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
    /// Connect using the operator's `ratelimit.redis` config. The topology
    /// (`single` / `cluster` / `sentinel`) is selected by `mode`; see
    /// [`aisix_redis::connect`].
    pub async fn connect(cfg: &RedisConnConfig) -> Result<Self, redis::RedisError> {
        let conn = aisix_redis::connect(cfg).await?;
        Ok(Self::with_slot(
            ConnSlot::filled(conn),
            Arc::new(AtomicBool::new(false)),
        ))
    }

    /// Connect for the boot path: bounded by `ratelimit.redis.timeout_secs`
    /// and never fatal.
    ///
    /// The gateway must bind its listeners whether or not Redis is up —
    /// the node-reboot case has Redis coming up *after* the gateway, and
    /// a gateway that refuses to serve until its
    /// rate-limit counters are shareable trades an availability outage for
    /// an accuracy one. So a connection that does not land inside the
    /// budget leaves the store in its ordinary degraded state (per-replica
    /// in-memory counting) and a background task attaches the shared
    /// backend the moment Redis answers. The returned error is the boot
    /// diagnostic, not a failure: it is `Some` exactly when the store
    /// started degraded.
    ///
    /// Note what this does NOT do: fall back to the `memory` backend. That
    /// would be permanent, and the operator asked for cluster-wide
    /// counting.
    pub async fn connect_or_attach_later(
        cfg: &RedisConnConfig,
    ) -> (Self, Option<redis::RedisError>) {
        let policy = FailurePolicy::new(cfg);
        match aisix_redis::connect_bounded(cfg, &policy).await {
            Ok(conn) => (
                Self::with_slot(ConnSlot::filled(conn), Arc::new(AtomicBool::new(false))),
                None,
            ),
            Err(e) => {
                let slot = ConnSlot::empty();
                // Already "logged": the caller WARNs this failure itself,
                // naming the endpoint, so the first degraded request must
                // not repeat it.
                let degraded_logged = Arc::new(AtomicBool::new(true));
                spawn_attach(
                    slot.clone(),
                    Arc::clone(&degraded_logged),
                    cfg.clone(),
                    policy,
                );
                (Self::with_slot(slot, degraded_logged), Some(e))
            }
        }
    }

    fn with_slot(conn: ConnSlot, degraded_logged: Arc<AtomicBool>) -> Self {
        Self {
            conn,
            prefix: DEFAULT_PREFIX.into(),
            conc_ttl: DEFAULT_CONC_TTL_SECS,
            grace: DEFAULT_GRACE_SECS,
            local: Arc::new(LocalStore::new()),
            degraded_logged,
            metrics: None,
        }
    }

    /// Count Redis operation failures on `metrics`. Without it the
    /// fail-open degradation below is invisible to a scrape — the whole
    /// point of the family (#1060).
    pub fn with_metrics(mut self, metrics: Metrics) -> Self {
        self.metrics = Some(metrics);
        self
    }

    pub fn with_conc_ttl(mut self, secs: u64) -> Self {
        self.conc_ttl = secs.max(1);
        self
    }

    /// Namespace the prefix by the DP's environment id. Most bucket keys are
    /// globally-unique CP UUIDs (api_key / policy ids), but the model inline
    /// limit buckets on the env-local model alias (`model:<name>`), so two
    /// environments sharing one (user-provided) Redis would otherwise merge
    /// their `model:<alias>` counters. Scoping the prefix to `<prefix>:<env_id>`
    /// isolates every bucket. Empty `env_id` (standalone / v2) leaves it
    /// unchanged. The env segment sits before the `{bucket}` hash tag, so
    /// Redis Cluster slot placement (by bucket) is unaffected.
    pub fn with_env_namespace(mut self, env_id: &str) -> Self {
        if !env_id.is_empty() {
            self.prefix = format!("{}:{}", self.prefix, env_id);
        }
        self
    }

    /// `aisix:rl:{<bucket>}` — the hash tag co-locates every key for the
    /// bucket on one Redis Cluster slot.
    fn bucket_prefix(&self, key: &str) -> String {
        format!("{}:{{{}}}", self.prefix, key)
    }

    /// Count the failure and log the fail-open transition once per
    /// outage. Every failed operation counts; only the first logs.
    fn note_failure(&self, op: &str, err: &redis::RedisError) {
        record_failure(&self.metrics, op);
        self.warn_degraded(op, err);
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

/// Keep trying the shared backend until it answers, then attach it.
///
/// The cadence is the configured budget, not the breaker's cool-off
/// window: the reason the request-path prober waits a whole window is
/// that a probe against a still-dead Redis costs a caller its budget,
/// and nothing here is billed to a caller. What it buys instead is the
/// case this exists for — Redis coming up seconds after the gateway on a
/// node reboot — recovering in seconds rather than half a minute.
///
/// One attempt per iteration, and `connect_bounded` bounds each one, so
/// an unreachable Redis costs one connect attempt per sleep-plus-budget
/// for as long as the outage lasts.
///
/// A permanent misconfiguration — a wrong host, a dead Redis, an
/// unreachable subnet — used to announce itself by killing the boot. Now
/// it is a gateway with a healthy `/livez` that quietly never enforces a
/// cluster-wide limit, and the only other trace is
/// `aisix_redis_failures_total`, which moves only while rate-limited
/// traffic actually arrives. So the retry re-states it at WARN on
/// [`DEGRADED_REMINDER`], leaving the individual attempts at debug.
fn spawn_attach(
    slot: ConnSlot,
    degraded_logged: Arc<AtomicBool>,
    cfg: RedisConnConfig,
    policy: FailurePolicy,
) {
    tokio::spawn(async move {
        let retry = std::time::Duration::from_secs(cfg.timeout_secs.max(1));
        let endpoint = aisix_redis::endpoint_label(&cfg);
        let mut last_reminder = std::time::Instant::now();
        loop {
            tokio::time::sleep(retry).await;
            match aisix_redis::connect_bounded(&cfg, &policy).await {
                Ok(conn) => {
                    slot.attach(conn);
                    // Re-arm the store's one-shot degradation warning:
                    // it was set at boot so the first degraded request
                    // would not repeat the boot WARN, and a LATER outage
                    // has to be able to warn again even if no store
                    // operation succeeded in between (which is what
                    // `mark_ok` would otherwise be relied on for).
                    degraded_logged.store(false, Ordering::Relaxed);
                    tracing::info!(
                        target: "aisix::ratelimit",
                        %endpoint,
                        "shared rate-limit backend attached; cluster-wide counting resumes"
                    );
                    return;
                }
                Err(e) => {
                    if last_reminder.elapsed() >= DEGRADED_REMINDER {
                        last_reminder = std::time::Instant::now();
                        tracing::warn!(
                            target: "aisix::ratelimit",
                            %endpoint,
                            error = %e,
                            "shared rate-limit backend still unreachable; cluster-wide \
                             rate limits are NOT enforced and counting stays per-replica"
                        );
                    } else {
                        tracing::debug!(
                            target: "aisix::ratelimit",
                            %endpoint,
                            error = %e,
                            retry_secs = retry.as_secs(),
                            "shared rate-limit backend still unreachable; per-replica \
                             counting continues"
                        );
                    }
                }
            }
        }
    });
}

/// Bump `aisix_redis_failures_total` for one failed rate-limit store
/// operation. The label is subsystem-qualified because a deployment
/// points the limiter and the caches at the same Redis, and a bare
/// `acquire` could have come from either.
fn record_failure(metrics: &Option<Metrics>, op: &str) {
    if let Some(m) = metrics {
        m.record_redis_failure(&format!("ratelimit_{op}"));
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
        // KEYS[1] carries the {bucket} hash tag for Redis Cluster routing;
        // args[0] is the same prefix the script reads from ARGV[1].
        invocation.key(&args[0]);
        for a in &args {
            invocation.arg(a);
        }
        let mut conn = match self.conn.acquire().await {
            Ok(c) => c,
            Err(e) => {
                self.note_failure("acquire", &e);
                return self.local.acquire(key, limits, member).await;
            }
        };
        match invocation.invoke_async::<Vec<i64>>(&mut conn).await {
            Ok(reply) => {
                self.mark_ok();
                let code = reply.first().copied().unwrap_or(CODE_OK);
                let retry = reply.get(1).copied().unwrap_or(0).max(0) as u64;
                // The script reports which dimension refused as its
                // 1-based index into the same `request_dims` /
                // `token_dims` vectors this call pushed into ARGV, so
                // the name never has to cross the wire.
                let refused = |dims: &[super::Dim]| {
                    let idx = reply.get(2).copied().unwrap_or(0);
                    let name = usize::try_from(idx)
                        .ok()
                        .and_then(|i| i.checked_sub(1))
                        .and_then(|i| dims.get(i))
                        .map(|d| d.name)
                        .unwrap_or(UNKNOWN_DIMENSION);
                    let limit = reply.get(3).copied().unwrap_or(0).max(0) as u64;
                    let used = reply.get(4).copied().unwrap_or(0).max(0) as u64;
                    LimitDetail::window(name, limit, used, retry)
                };
                match code {
                    CODE_OK => Ok(()),
                    CODE_CONCURRENCY => {
                        let limit = reply.get(3).copied().unwrap_or(0).max(0) as u64;
                        let in_flight = reply.get(4).copied().unwrap_or(0).max(0) as u64;
                        Err(RateLimitError::Concurrency {
                            detail: LimitDetail::concurrency(limit, in_flight),
                        })
                    }
                    CODE_TOKENS => Err(RateLimitError::Tokens {
                        scope: aisix_core::RateLimitScope::Tokens,
                        detail: refused(&token_dims(limits)),
                    }),
                    CODE_REQUESTS => Err(RateLimitError::Requests {
                        scope: aisix_core::RateLimitScope::Requests,
                        detail: refused(&super::request_dims(limits)),
                    }),
                    _ => Ok(()),
                }
            }
            Err(e) => {
                self.note_failure("acquire", &e);
                self.conn.note_error().await;
                self.local.acquire(key, limits, member).await
            }
        }
    }

    async fn commit(&self, key: &str, tokens: u64, member: &str) {
        let prefix = self.bucket_prefix(key);
        let mut conn = match self.conn.acquire().await {
            Ok(c) => c,
            Err(e) => {
                self.note_failure("commit", &e);
                return self.local.commit(key, tokens, member).await;
            }
        };
        let res: Result<i64, redis::RedisError> = Script::new(COMMIT_LUA)
            .key(&prefix)
            .arg(&prefix)
            .arg(member)
            .arg(tokens)
            .arg(self.grace)
            .invoke_async(&mut conn)
            .await;
        match res {
            Ok(_) => {
                self.mark_ok();
                // Symmetric with `release`, and for the same reason: a
                // reservation whose `acquire` fell back to the local
                // store still holds a local concurrency slot, and this
                // commit went to the shared backend, whose ZSET never
                // held the member. Without this the slot is never given
                // back — `LocalStore` has no ttl — so the local in-flight
                // baseline for that bucket is permanently raised by the
                // number of requests in flight when the backend
                // attached, and a LATER outage then fails CLOSED on a
                // path whose whole contract is to fail open.
                // Non-inserting, hence a no-op for a bucket that never
                // acquired locally.
                self.local.release(key, member);
            }
            Err(e) => {
                self.note_failure("commit", &e);
                self.conn.note_error().await;
                self.local.commit(key, tokens, member).await;
            }
        }
    }

    fn release(&self, key: &str, member: &str) {
        // Drop the local slot first (a cheap no-op when the bucket was
        // never acquired locally); covers the degraded-acquire case.
        self.local.release(key, member);
        let conc_key = format!("{}:conc", self.bucket_prefix(key));
        let conn = self.conn.clone();
        let member = member.to_string();
        let metrics = self.metrics.clone();
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                // ZREM's first arg is the key, so Redis Cluster routes it.
                let mut c = match conn.acquire().await {
                    Ok(c) => c,
                    Err(_) => {
                        record_failure(&metrics, "release");
                        return;
                    }
                };
                let res: Result<(), redis::RedisError> = redis::cmd("ZREM")
                    .arg(&conc_key)
                    .arg(&member)
                    .query_async(&mut c)
                    .await;
                if res.is_err() {
                    // The detached tasks are the only Redis failures with
                    // no log at all — nothing awaits them, so the counter
                    // is their only trace.
                    record_failure(&metrics, "release");
                    conn.note_error().await;
                }
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
        let conn = self.conn.clone();
        let metrics = self.metrics.clone();
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                let mut c = match conn.acquire().await {
                    Ok(c) => c,
                    Err(_) => {
                        record_failure(&metrics, "add_tokens");
                        return;
                    }
                };
                let res: Result<i64, redis::RedisError> = Script::new(ADD_TOKENS_LUA)
                    .key(&prefix)
                    .arg(&prefix)
                    .arg(tokens)
                    .arg(grace)
                    .invoke_async(&mut c)
                    .await;
                if res.is_err() {
                    record_failure(&metrics, "add_tokens");
                    conn.note_error().await;
                }
            });
        }
    }

    async fn peek(&self, key: &str, limits: &RateLimit) -> Option<RateLimitStatus> {
        if limits.is_unrestricted() {
            return None;
        }
        let prefix = self.bucket_prefix(key);
        let mut conn = match self.conn.acquire().await {
            Ok(c) => c,
            Err(e) => {
                self.note_failure("peek", &e);
                return self.local.peek(key, limits).await;
            }
        };
        let reply: Result<Vec<i64>, redis::RedisError> = Script::new(PEEK_LUA)
            .key(&prefix)
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
                self.note_failure("peek", &e);
                self.conn.note_error().await;
                self.local.peek(key, limits).await
            }
        }
    }
}
