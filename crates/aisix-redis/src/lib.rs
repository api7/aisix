//! Shared Redis connection layer for the response cache
//! ([`aisix-cache`]) and the shared rate-limit counter store
//! ([`aisix-ratelimit`]).
//!
//! Both subsystems used to hold a single-node [`ConnectionManager`]
//! directly. This crate factors the connection out behind one
//! [`RedisConn`] so the operator can pick the topology with
//! `redis.mode` — `single`, `cluster`, or `sentinel` — and both
//! subsystems support all three without each re-implementing the
//! dispatch (and drifting apart).
//!
//! A live connection is obtained per operation via
//! [`RedisConn::acquire`], which yields a [`RedisConnHandle`] that
//! implements [`redis::aio::ConnectionLike`], so existing call sites
//! keep running `Script::invoke_async`/`cmd().query_async` against it
//! unchanged.
//!
//! - **single** — a [`ConnectionManager`] (transparent reconnect). The
//!   handle is a cheap clone; `acquire` never fails.
//! - **cluster** — a [`cluster_async::ClusterConnection`] that discovers
//!   the slot topology and reconnects internally. The handle is a cheap
//!   clone; `acquire` never fails. NOTE: scripts that touch multiple keys
//!   must declare a key carrying the bucket hash tag so the `EVAL` routes
//!   to the slot owning every key it touches — callers are responsible
//!   for that (see `aisix-ratelimit`).
//! - **sentinel** — a [`SentinelClient`] that resolves the current master
//!   for `master_name`. `redis` 0.27 has no auto-reconnecting sentinel
//!   connection, so on a master failover the cached connection breaks;
//!   [`RedisConn::note_error`] drops it and the next `acquire` re-resolves
//!   the new master through the sentinels.
//!
//! # Bounded failure
//!
//! Every consumer of this layer fails **open** on a Redis error — the
//! rate limiter degrades to per-replica counters, the caches to a miss —
//! but that only helps if the error *arrives*. A peer that stops
//! answering without closing the socket (host down, network partition, a
//! stopped container) leaves a command blocked on TCP retransmission for
//! minutes, so the fallback is never reached and the request hangs.
//!
//! Two mechanisms, both applied here so no consumer can forget them:
//!
//! - **A timeout on every command and every connection attempt**, from
//!   `redis.timeout_secs` (default 5s). It is set natively on the driver
//!   for all three topologies *and* enforced around each command, because
//!   the native response timeout does not cover time spent waiting on the
//!   connection manager's own in-progress reconnect — which is where the
//!   remaining unbounded wait lived.
//! - **A cool-off breaker, shared per subsystem.** Paying the timeout on *every* request during
//!   an outage is still a several-second latency floor for as long as the
//!   outage lasts. After a connectivity failure the subsystem is held
//!   open for [`BREAKER_WINDOW`] — 30s, long enough to span an upstream
//!   call so a request's cache write is still covered by the cool-off its
//!   cache read opened; commands issued inside that window
//!   return an error immediately, with no round trip, so each consumer's
//!   existing fail-open branch runs at once.
//! - **A background prober closes it.** Opening the breaker starts a task
//!   that PINGs Redis when the window expires, closes the breaker on
//!   success and re-arms the window on failure. Business commands keep
//!   short-circuiting until it closes, so none of them is ever the
//!   half-open probe: a probe that lands on a still-unreachable Redis
//!   pays the full command budget, and during a minutes-long outage that
//!   is one request per window paying it for nothing. The recovery
//!   latency is unchanged — the breaker still closes within one window of
//!   Redis coming back — it is simply no longer billed to a caller.
//!   What a PING proves is narrower than what the old probe proved,
//!   because the old probe was a real command: a server that answers
//!   PING while the operations this subsystem actually runs keep timing
//!   out (a loaded vector search, a partly-down cluster) closes the
//!   breaker, and the commands behind it each pay one budget until the
//!   first failure re-opens it. That is one window's worth of concurrent
//!   commands rather than one command — bounded, and the alternative is
//!   billing a caller for detection on every window of every outage.
//!
//! The same reasoning applies to the connection a subsystem opens at
//! **boot**, which nothing else bounds: [`connect_bounded`] spends at
//! most one budget there, because the driver's own retry schedule for
//! the initial connect is measured in minutes and boot holds the
//! listeners closed while it runs.
//!
//! The breaker belongs to a **subsystem**, not to a connection — see
//! [`FailurePolicy`]. A subsystem may hold several connections (the cache
//! holds two: exact-KV and vector search) and one request touches all of
//! them, so a per-connection breaker let a single request pay the budget
//! once per connection.
//!
//! Breaker short-circuits are ordinary `Err`s, so they are counted by the
//! consumers' existing `aisix_redis_failures_total{operation=...}` calls
//! with no new metric.

use std::sync::Arc;
use std::time::{Duration, Instant};

use aisix_core::{RedisConnConfig, RedisMode};
use redis::aio::{
    ConnectionLike, ConnectionManager, ConnectionManagerConfig, MultiplexedConnection,
};
use redis::cluster::ClusterClient;
use redis::cluster_async::ClusterConnection;
use redis::sentinel::{SentinelClient, SentinelNodeConnectionInfo, SentinelServerType};
use redis::{AsyncConnectionConfig, IntoConnectionInfo, RedisResult};
use tokio::sync::Mutex;

/// How long commands short-circuit after a connectivity failure. Fixed,
/// not configurable: it trades at most this much staleness (a Redis that
/// recovered mid-window is not noticed until the window ends) for a hard
/// ceiling on how often a request pays the full timeout during an outage.
///
/// Sized to outlast an upstream call, not to be a small multiple of the
/// command budget. A request's cache read and its cache write straddle
/// the upstream leg, so a window shorter than that leg leaves the write
/// to find the cool-off expired and pay the budget a second time — which
/// is what a 5s window did, since the non-streaming completions this
/// cache stores routinely take longer than that.
pub const BREAKER_WINDOW: Duration = Duration::from_secs(30);

/// A [`RedisConn`] that may not exist yet, shared by every operation of
/// one subsystem.
///
/// A Redis that is unreachable when the gateway starts must not keep it
/// from binding its listeners, so a subsystem carries on with the slot
/// empty and a background task fills it in when Redis answers. An empty
/// slot is the state a mid-flight outage already puts the subsystem in —
/// every operation gets a connectivity error and runs its existing
/// fail-open branch — which is why no consumer needs a branch of its own
/// for it.
///
/// `ArcSwapOption` rather than a lock around the value: cloning a
/// `RedisConn` out per operation is a deep clone of the driver's
/// connection info (host, username, password), and [`RedisConn::acquire`]
/// already makes one of those internally.
#[derive(Clone)]
pub struct ConnSlot(Arc<arc_swap::ArcSwapOption<RedisConn>>);

impl ConnSlot {
    pub fn filled(conn: RedisConn) -> Self {
        Self(Arc::new(arc_swap::ArcSwapOption::from_pointee(conn)))
    }

    pub fn empty() -> Self {
        Self(Arc::new(arc_swap::ArcSwapOption::empty()))
    }

    pub fn attach(&self, conn: RedisConn) {
        self.0.store(Some(Arc::new(conn)));
    }

    /// Whether a connection has landed, so a retry loop does not re-dial
    /// a slot it already filled.
    pub fn is_attached(&self) -> bool {
        self.get().is_some()
    }

    fn get(&self) -> Option<Arc<RedisConn>> {
        self.0.load_full()
    }

    /// A live handle, or the error that makes the caller fail open.
    pub async fn acquire(&self) -> RedisResult<RedisConnHandle> {
        match self.get() {
            Some(conn) => conn.acquire().await,
            None => Err(not_connected_error()),
        }
    }

    /// [`RedisConn::note_error`] on the connection if there is one.
    pub async fn note_error(&self) {
        if let Some(conn) = self.get() {
            conn.note_error().await;
        }
    }
}

/// A long-lived Redis client handle. Cheap to [`Clone`] (every variant is
/// `Arc`-backed). Build one with [`connect`].
#[derive(Clone)]
pub struct RedisConn {
    inner: ConnKind,
    guard: Arc<Guard>,
}

// `Single` (the hot, common path) is the largest variant; boxing it to
// equalize variant size would add an allocation to the common case to
// shrink the rarer ones — not worth it for a handful of instances.
#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
enum ConnKind {
    Single(ConnectionManager),
    Cluster(ClusterConnection),
    Sentinel(SentinelPool),
}

/// Sentinel client plus the most recently resolved master connection.
/// The cache is cleared on error ([`RedisConn::note_error`]) so the next
/// [`RedisConn::acquire`] re-discovers the master after a failover.
#[derive(Clone)]
pub struct SentinelPool {
    client: Arc<Mutex<SentinelClient>>,
    cached: Arc<Mutex<Option<MultiplexedConnection>>>,
    /// Budget for one master discovery, which is NOT one round trip: the
    /// client library walks the sentinels **serially** and applies no
    /// timeout of its own to the sentinel hops, so a single budget for
    /// the whole walk would let one unreachable sentinel consume it and
    /// starve the healthy ones — they would never be tried, and the
    /// master would never resolve even with quorum intact. One budget per
    /// sentinel, plus one for the master connection that follows.
    discovery_timeout: Duration,
}

/// A live connection usable for one or more operations. Implements
/// [`ConnectionLike`] by delegating to the underlying connection, with
/// the timeout and breaker of the [`RedisConn`] it came from applied to
/// every command.
pub struct RedisConnHandle {
    inner: HandleKind,
    guard: Arc<Guard>,
}

#[allow(clippy::large_enum_variant)]
enum HandleKind {
    Single(ConnectionManager),
    Cluster(ClusterConnection),
    Sentinel(MultiplexedConnection),
}

/// One command budget and one breaker. Reached through [`FailurePolicy`],
/// which is what decides how widely it is shared.
struct Guard {
    timeout: Duration,
    breaker: Breaker,
    /// A connection the background prober tests Redis on while the breaker
    /// is open, registered by the first [`connect_with`] against this
    /// policy. Every connection a subsystem opens addresses the same
    /// server, so one of them answers for all of them — and the prober
    /// must hold its own handle because the `RedisConn`s it belongs to
    /// are not reachable from here.
    prober: std::sync::Mutex<Option<ConnKind>>,
}

/// The failure policy of one Redis **subsystem**: a command budget and a
/// cool-off window shared by every connection that subsystem opens.
///
/// A subsystem is the unit that fails open together, and it is not the
/// same thing as a connection. The cache is one subsystem holding two
/// connections — exact-KV and vector search — and a single chat request
/// touches both twice: exact lookup, semantic lookup, exact write,
/// semantic write. With a breaker per connection each of those four
/// operations found a breaker that no earlier operation had opened, so
/// one request against a black-holed Redis paid the budget four times
/// (measured: 20s at the default 5s budget). Sharing the policy makes the
/// first failure short-circuit the operations that follow it inside the
/// cool-off window.
///
/// The two lookups run back to back, but the two writes run after the
/// upstream call, so this holds for a whole request only while
/// [`BREAKER_WINDOW`] outlasts that upstream leg — which is why the
/// window is 30s rather than a small multiple of the budget. A request
/// whose upstream call runs longer than the window still finds it
/// expired and pays a second budget on the write that follows.
///
/// Sharing the policy does NOT share the connection: the two cache
/// connections stay separate so they do not serialize on one pipeline.
///
/// One policy per `redis:` config block. The rate limiter reads a
/// different block and is a different subsystem, so it holds its own —
/// a Redis outage that is really one outage still costs a request one
/// budget for the cache and one for the limiter, which is the price of
/// letting the two degrade independently.
#[derive(Clone)]
pub struct FailurePolicy(Arc<Guard>);

impl FailurePolicy {
    /// Build the policy for one `redis:` config block. Pass the same
    /// value to every [`connect_with`] call for that block.
    pub fn new(cfg: &RedisConnConfig) -> Self {
        Self(Arc::new(Guard {
            // `validate` rejects 0, but this type is constructible from a
            // hand-built config in tests; clamping keeps "no budget at
            // all" unreachable rather than merely unconfigurable.
            timeout: Duration::from_secs(cfg.timeout_secs.max(1)),
            breaker: Breaker::new(BREAKER_WINDOW),
            prober: std::sync::Mutex::new(None),
        }))
    }
}

impl Guard {
    /// Run one Redis operation under the breaker and the command budget.
    ///
    /// A success closes the breaker; a *connectivity* failure or a
    /// timeout opens it. A failure the server itself reported (a script
    /// error, `WRONGTYPE`, an ACL refusal) is returned untouched — Redis
    /// answered, so short-circuiting the subsystem's next half minute of
    /// traffic would be wrong.
    async fn run<T>(
        guard: &Arc<Self>,
        fut: impl std::future::Future<Output = RedisResult<T>>,
    ) -> RedisResult<T> {
        Self::run_with(guard, guard.timeout, fut).await
    }

    /// [`Guard::run`] with a budget other than the per-command one. Only
    /// sentinel master discovery uses it — see [`SentinelPool`].
    async fn run_with<T>(
        guard: &Arc<Self>,
        budget: Duration,
        fut: impl std::future::Future<Output = RedisResult<T>>,
    ) -> RedisResult<T> {
        // The generation `admit` returns is read before the await, because
        // a command already in flight when the outage began can land its
        // success after a *concurrent* command opened the breaker, and
        // closing on that stale evidence would send the requests behind it
        // back into the full budget.
        let Some(seen) = guard.breaker.admit() else {
            return Err(breaker_open_error());
        };
        match tokio::time::timeout(budget, fut).await {
            Ok(Ok(v)) => {
                guard.breaker.close_unless_reopened(seen);
                Ok(v)
            }
            Ok(Err(e)) => {
                // `is_io_error` is the whole test: `is_timeout` and
                // `is_connection_dropped` are strict subsets of it, and
                // the connectivity errors that are NOT io errors —
                // `ClusterConnectionNotFound`, `MasterNameNotFoundBySentinel`
                // — return instantly, so the caller already failed open
                // without paying anything and has nothing to cool off from.
                if e.is_io_error() {
                    Self::trip(guard);
                }
                Err(e)
            }
            Err(_) => {
                Self::trip(guard);
                Err(timed_out_error(budget))
            }
        }
    }

    /// Open the breaker and put a background prober in charge of closing
    /// it, so the command that re-tests Redis is never a business one.
    ///
    /// Without a registered connection there is nothing to probe with —
    /// the only way in is a failure during [`connect_with`] itself, before
    /// any connection exists — and the breaker falls back to admitting one
    /// caller once the window expires, or it would never close at all.
    fn trip(guard: &Arc<Self>) {
        guard.breaker.open();
        let Some(conn) = lock(&guard.prober).clone() else {
            return;
        };
        let Some(claim) = guard.breaker.claim_prober() else {
            return;
        };
        let guard = Arc::clone(guard);
        tokio::spawn(async move { probe_until_closed(guard, conn, claim).await });
    }
}

/// Re-test Redis once per window until the breaker closes. Runs detached,
/// off the request path; exactly one of these exists per open breaker
/// ([`Breaker::claim_prober`]).
async fn probe_until_closed(guard: Arc<Guard>, conn: ConnKind, claim: u64) {
    // What one probe can cost at worst. Not the command budget for a
    // sentinel, where a probe may re-walk the sentinels to re-resolve the
    // master first — and `timeout_secs` has no upper bound, so a deadline
    // sized on a constant would expire under a live prober on some
    // configurations and let callers pay the budget after all.
    let probe_budget = match &conn {
        ConnKind::Sentinel(pool) => pool.discovery_timeout + guard.timeout,
        _ => guard.timeout,
    };
    loop {
        if !guard
            .breaker
            .hold_off(claim, guard.breaker.window + probe_budget)
        {
            return;
        }
        tokio::time::sleep(guard.breaker.window).await;
        let seen = guard.breaker.generation();
        match probe_ping(&conn, guard.timeout).await {
            Ok(()) => {
                if guard.breaker.close_from_prober(claim, seen) {
                    tracing::info!(
                        target: "aisix::redis",
                        "redis answered the cool-off probe; commands resume"
                    );
                    return;
                }
                // A command that was already in flight when the outage
                // began landed its failure while this probe ran. Its
                // evidence is newer than this probe's — keep probing.
            }
            Err(e) => {
                tracing::debug!(
                    target: "aisix::redis",
                    error = %e,
                    window_secs = guard.breaker.window.as_secs(),
                    "redis cool-off probe failed; commands keep short-circuiting"
                );
            }
        }
    }
}

/// One PING for the prober. Deliberately outside [`Guard::run`]: the
/// breaker is open, so the guard would short-circuit the very command
/// that is meant to reach the network.
async fn probe_ping(kind: &ConnKind, budget: Duration) -> RedisResult<()> {
    async fn ping<C: ConnectionLike + Send>(conn: &mut C, budget: Duration) -> RedisResult<()> {
        match tokio::time::timeout(budget, redis::cmd("PING").query_async(conn)).await {
            Ok(r) => r,
            Err(_) => Err(timed_out_error(budget)),
        }
    }

    match kind {
        ConnKind::Single(c) => ping(&mut c.clone(), budget).await,
        ConnKind::Cluster(c) => ping(&mut c.clone(), budget).await,
        ConnKind::Sentinel(pool) => {
            let cached = pool.cached.lock().await.clone();
            let mut conn = match cached {
                Some(conn) => conn,
                None => {
                    let mut client = pool.client.lock().await;
                    match tokio::time::timeout(
                        pool.discovery_timeout,
                        client.get_async_connection_with_config(&conn_config(budget)),
                    )
                    .await
                    {
                        Ok(conn) => conn?,
                        Err(_) => return Err(timed_out_error(pool.discovery_timeout)),
                    }
                }
            };
            let answered = ping(&mut conn, budget).await;
            // A failed probe drops the cached master so the next one
            // re-resolves it: an outage that was really a failover leaves
            // this connection pointing at a demoted node.
            *pool.cached.lock().await = answered.is_ok().then_some(conn);
            answered
        }
    }
}

/// Nothing under these locks can panic, so they cannot be poisoned;
/// taking the value through a poisoned guard would be equally correct if
/// one ever were.
fn lock<T>(m: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// A fixed-window cool-off. Open until `open_until` has passed, then the
/// next command probes Redis for real.
///
/// `generation` counts openings, so a success can tell "the breaker I saw
/// closed when I started" from "a breaker something else opened while I
/// was in flight". A plain timestamp comparison cannot: the two events
/// are microseconds apart.
struct Breaker {
    window: Duration,
    state: std::sync::Mutex<BreakerState>,
}

#[derive(Default)]
struct BreakerState {
    open_until: Option<Instant>,
    generation: u64,
    /// The prober in charge of closing this breaker, if any. An id rather
    /// than a flag, because the claim can change hands: a prober whose
    /// deadline lapsed has it taken back by the caller that got through,
    /// and the failure of THAT caller starts a replacement. The original
    /// must not then renew the replacement's window or close the breaker
    /// on evidence the replacement never asked for — with a flag, both
    /// would read as "a prober is in charge" and both probers would run
    /// on one connection.
    prober: Option<u64>,
    /// Ids handed out so far, so a returning prober cannot match a
    /// claim it no longer owns.
    claims: u64,
}

impl Breaker {
    fn new(window: Duration) -> Self {
        Self {
            window,
            state: std::sync::Mutex::new(BreakerState::default()),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, BreakerState> {
        lock(&self.state)
    }

    /// True once something has closed the breaker on evidence, rather
    /// than its window merely having expired with nothing to show for it.
    #[cfg(test)]
    fn is_closed(&self) -> bool {
        self.lock().open_until.is_none()
    }

    /// Whether a caller arriving now is held off. ONE predicate, read by
    /// both the pre-flight check on [`RedisConn::acquire`] and by
    /// [`Breaker::admit`] — a caller the pre-flight check turns away
    /// never reaches `admit`, so anything `admit` alone knew (a grace
    /// period, a fallback) would never be reached on the request path at
    /// all, and the breaker could latch open for the life of the process.
    ///
    /// A live prober is expressed as a deadline in the future rather than
    /// as a flag, precisely so it needs no second predicate: it claims
    /// the sleep AND the probe that follows before each round
    /// ([`Breaker::hold_off`]), and one that stopped without closing the
    /// breaker simply lets its deadline pass.
    fn is_open(&self) -> bool {
        let st = self.lock();
        st.open_until.is_some_and(|until| Instant::now() < until)
    }

    /// Decide whether this command reaches Redis, and hand back the
    /// generation it is allowed to close.
    ///
    /// `None` short-circuits. While a prober is in charge that is every
    /// caller, until the prober itself closes the breaker — the point of
    /// the prober being that a re-test against an unreachable Redis costs
    /// the full command budget, and no business request should pay it.
    ///
    /// The last arm is the no-prober fallback ([`Guard::trip`]): one
    /// caller is admitted once the window expires and the window is
    /// re-armed behind it, so commands arriving while that probe is in
    /// flight still short-circuit instead of all paying the budget.
    fn admit(&self) -> Option<u64> {
        let mut st = self.lock();
        match st.open_until {
            None => Some(st.generation),
            Some(until) if Instant::now() < until => None,
            // The deadline has passed: either no prober was ever
            // registered (a failure during `connect_with` itself), or the
            // one that claimed it stopped without closing the breaker —
            // its task was dropped, or the runtime is gone. Either way,
            // admit one caller and re-arm behind it, which is what the
            // window alone used to do; releasing the claim also lets the
            // next failure start a fresh prober.
            Some(_) => {
                st.prober = None;
                st.open_until = Some(Instant::now() + self.window);
                Some(st.generation)
            }
        }
    }

    fn open(&self) {
        let mut st = self.lock();
        st.open_until = Some(Instant::now() + self.window);
        st.generation = st.generation.wrapping_add(1);
    }

    /// Take charge of probing, if nothing else already has. True means the
    /// caller must start the prober task; it stays true until that task
    /// closes the breaker.
    fn claim_prober(&self) -> Option<u64> {
        let mut st = self.lock();
        if st.prober.is_some() {
            return None;
        }
        st.claims += 1;
        st.prober = Some(st.claims);
        st.prober
    }

    fn generation(&self) -> u64 {
        self.lock().generation
    }

    /// The prober claiming the next `for_` of short-circuiting: its wait
    /// plus the probe that follows, so nothing can read it as stopped
    /// while it is working. False means the claim is no longer this
    /// prober's — a caller took it back after the deadline passed, or the
    /// breaker closed — and the task must end rather than probe on;
    /// otherwise a second prober started meanwhile would double up, and
    /// the two would serialize on the same connection.
    fn hold_off(&self, claim: u64, for_: Duration) -> bool {
        let mut st = self.lock();
        if st.prober != Some(claim) {
            return false;
        }
        st.open_until = Some(Instant::now() + for_);
        true
    }

    /// Close the breaker on the prober's evidence and release the
    /// prober's claim. False when a failure newer than `seen` re-opened
    /// it, which leaves the claim in place — the prober keeps going.
    fn close_from_prober(&self, claim: u64, seen: u64) -> bool {
        let mut st = self.lock();
        if st.prober != Some(claim) || st.generation != seen {
            return false;
        }
        st.open_until = None;
        st.prober = None;
        true
    }

    /// Close the breaker unless it was opened after `seen` was read.
    fn close_unless_reopened(&self, seen: u64) {
        let mut st = self.lock();
        if st.generation == seen {
            st.open_until = None;
        }
    }
}

/// The error a short-circuited command returns. `IoError` so consumers
/// classify it exactly as they classify the real connectivity failure it
/// stands in for.
fn breaker_open_error() -> redis::RedisError {
    redis::RedisError::from((
        redis::ErrorKind::IoError,
        "redis is in a failure cool-off",
        format!(
            "a Redis command failed within the last {}s; commands short-circuit until the \
             cool-off expires so the caller's fallback runs without paying the timeout again",
            BREAKER_WINDOW.as_secs()
        ),
    ))
}

/// The error an operation gets while its subsystem holds no connection
/// yet — the state [`connect_bounded`] leaves behind when Redis is
/// unreachable at startup and the subsystem chose to carry on without
/// it. `IoError` for the same reason as [`breaker_open_error`]: every
/// consumer already classifies a connectivity failure as the cue to run
/// its fail-open branch, and this is one.
pub fn not_connected_error() -> redis::RedisError {
    redis::RedisError::from((
        redis::ErrorKind::IoError,
        "redis is not connected yet",
        "the backend was unreachable at startup; a background task keeps trying and \
         commands run on the local fallback until it attaches"
            .to_string(),
    ))
}

/// True when a connect error can never come good on a retry: the
/// operator wrote something the driver cannot use at all — a malformed
/// `url`, TLS material that will not read or parse — or a credential the
/// server answered and refused.
///
/// It matters because everything else here is now retried forever in the
/// background. Retrying a typo turns a boot that said exactly what was
/// wrong into a gateway that comes up healthy and is quietly never going
/// to enforce a shared limit or cache anything, which is strictly worse
/// than the failure it replaced. A refused credential is the same
/// judgement the etcd side already makes: no amount of waiting turns a
/// wrong password into a right one, and a server that ANSWERED is not
/// the unreachable case the retry exists for.
///
/// It only classifies what the driver reports as such. A refusal the
/// budget cut short arrives as a timeout instead, and is retried — which
/// is the safe direction for a misclassification, since the alternative
/// is refusing to start over a slow handshake.
pub fn is_permanent_config_error(e: &redis::RedisError) -> bool {
    if matches!(
        e.kind(),
        redis::ErrorKind::InvalidClientConfig | redis::ErrorKind::AuthenticationFailed
    ) {
        return true;
    }
    // A refusal the server raises against the startup PING rather than
    // against the handshake. redis-rs has no `ErrorKind` for these — they
    // arrive as extension errors carrying the server's own code — so
    // match the code. `NOAUTH` is what a `requirepass` server answers a
    // connection that never authenticated at all, and `WRONGPASS` is the
    // ACL refusal; neither comes good on a retry.
    //
    // Deliberately NOT `NOPERM`: that answer comes from a connection the
    // server has already authenticated and only says this user may not
    // run this command. An ACL scoped to what a subsystem actually uses
    // routinely excludes `PING`, and such a deployment works.
    matches!(e.code(), Some("NOAUTH" | "WRONGPASS"))
}

/// The error a connect that outran its whole budget returns.
///
/// Separate from [`timed_out_error`] because the budget it spent is a
/// MULTIPLE of the field the operator set, and a message that named
/// `redis.timeout_secs (20s)` would send someone grepping their config
/// for a 20 that is not in it. Name the product and the field it came
/// from, both.
fn connect_timed_out_error(budget: Duration, per_attempt: Duration) -> redis::RedisError {
    redis::RedisError::from((
        redis::ErrorKind::IoError,
        "redis connect timed out",
        format!(
            "no connection within {}s (redis.timeout_secs = {}s, once per configured \
             endpoint the connect walks plus one)",
            budget.as_secs(),
            per_attempt.as_secs()
        ),
    ))
}

fn timed_out_error(budget: Duration) -> redis::RedisError {
    redis::RedisError::from((
        redis::ErrorKind::IoError,
        "redis command timed out",
        format!("no reply within redis.timeout_secs ({}s)", budget.as_secs()),
    ))
}

impl RedisConn {
    /// Obtain a live connection. For `single`/`cluster` this is an
    /// infallible cheap clone of the multiplexed connection. For
    /// `sentinel` it returns the cached master connection, resolving one
    /// through the sentinels on the first call or after a failover.
    ///
    /// While the breaker is open this returns the short-circuit error
    /// without touching the network, so the caller's fail-open branch
    /// runs immediately — including for `sentinel`, whose master
    /// re-resolution is itself a round trip to a host that may be gone.
    pub async fn acquire(&self) -> RedisResult<RedisConnHandle> {
        if self.guard.breaker.is_open() {
            return Err(breaker_open_error());
        }
        let inner = match &self.inner {
            ConnKind::Single(c) => HandleKind::Single(c.clone()),
            ConnKind::Cluster(c) => HandleKind::Cluster(c.clone()),
            ConnKind::Sentinel(pool) => {
                let cached = pool.cached.lock().await.clone();
                match cached {
                    Some(conn) => HandleKind::Sentinel(conn),
                    None => {
                        let mut client = pool.client.lock().await;
                        let cfg = conn_config(self.guard.timeout);
                        let conn = Guard::run_with(
                            &self.guard,
                            pool.discovery_timeout,
                            client.get_async_connection_with_config(&cfg),
                        )
                        .await?;
                        *pool.cached.lock().await = Some(conn.clone());
                        HandleKind::Sentinel(conn)
                    }
                }
            }
        };
        Ok(RedisConnHandle {
            inner,
            guard: Arc::clone(&self.guard),
        })
    }

    /// Invalidate any cached connection after an operation error. Only
    /// meaningful for `sentinel`, where it forces the next [`acquire`] to
    /// re-resolve the master (the prior one may have failed over).
    ///
    /// Re-resolution is not immediate any more: the failure that prompted
    /// this call also opened the breaker, so the next `acquire` inside
    /// [`BREAKER_WINDOW`] short-circuits and the master is re-discovered
    /// by the first probe after it. A failover therefore costs up to one
    /// window of per-replica counting / cache misses — fail-open, and the
    /// alternative is every request racing to re-walk the sentinels.
    ///
    /// [`acquire`]: RedisConn::acquire
    pub async fn note_error(&self) {
        if let ConnKind::Sentinel(pool) = &self.inner {
            *pool.cached.lock().await = None;
        }
    }
}

/// The driver-native timeouts, shared by every topology that accepts them.
fn conn_config(timeout: Duration) -> AsyncConnectionConfig {
    AsyncConnectionConfig::new()
        .set_connection_timeout(timeout)
        .set_response_timeout(timeout)
}

/// Apply the explicitly configured connection settings on top of what
/// the URL carried.
///
/// The explicit fields WIN. That is what they exist for: `password` is
/// documented as the way to keep the secret out of the config file
/// (`AISIX_CACHE__REDIS__PASSWORD`), and a value supplied that way is
/// useless if a stale credential left in the URL quietly outranks it.
///
/// The credential is taken as a PAIR. Overriding the two halves
/// independently can compose a login that exists in neither source — a
/// URL's `alice:s3cret` plus a leftover `username: bob` would send
/// `AUTH bob s3cret`, a credential nobody configured, and its refusal
/// would be reported against config that looks correct in both places.
/// So setting either half replaces both, and a half-specified override
/// fails as itself.
///
/// `database` is left alone in `cluster` mode, where Redis has only DB 0
/// and `SELECT` is rejected outright.
fn apply_explicit_settings(info: &mut redis::RedisConnectionInfo, cfg: &RedisConnConfig) {
    if cfg.username.is_some() || cfg.password.is_some() {
        info.username = cfg.username.clone();
        info.password = cfg.password.clone();
    }
    if cfg.mode != RedisMode::Cluster {
        if let Some(db) = cfg.database {
            info.db = db;
        }
    }
}

/// Say so when a credential is configured in both places, since only one
/// of them is in force.
///
/// `password` had no effect at all in `single` mode until this release,
/// so a stale one could sit in the config for a long time without ever
/// being noticed — and it now outranks the URL's. Never the values, and
/// never the URL: both carry the secret.
fn warn_on_credential_shadowing(cfg: &RedisConnConfig) {
    if cfg.username.is_none() && cfg.password.is_none() {
        return;
    }
    let in_url = |url: &String| {
        url.trim()
            .into_connection_info()
            .is_ok_and(|i| i.redis.username.is_some() || i.redis.password.is_some())
    };
    let shadowed = match cfg.mode {
        RedisMode::Single => cfg.url.as_ref().is_some_and(in_url),
        RedisMode::Cluster => cfg.nodes.iter().any(in_url),
        // The sentinels' own credentials are a different credential, not
        // a second copy of the master's, so they shadow nothing.
        RedisMode::Sentinel => false,
    };
    if !shadowed {
        return;
    }
    tracing::warn!(
        target: "aisix::redis",
        endpoint = %endpoint_label(cfg),
        "redis credentials are configured in two places: the URL carries one and \
         redis.username/redis.password carry another. The explicit fields are what is \
         sent; remove the credential from the URL, or clear the fields, so the \
         configuration says which one is in force."
    );
}

/// One real command on a freshly opened connection, so "connected" can
/// never be logged for a connection the server has not authenticated.
///
/// Opening the connection is not proof of that. A `requirepass` server
/// accepts the socket and refuses the *commands*, so a credential that
/// never reached the handshake looks exactly like a healthy boot — the
/// shape this gateway shipped in when `password` was parsed and then
/// dropped on the floor in `single` mode: `connected`, no warning, and
/// every counter and cache operation failing per request from then on.
///
/// `NOPERM` counts as success, because it is an answer from an
/// authenticated connection: the server knows who we are and has
/// declined this one command. An ACL user scoped to what a subsystem
/// actually runs (`+get +set +eval …`) commonly has no `+ping`, and
/// reading its refusal as a bad credential would refuse to start a
/// deployment that works.
async fn ping(conn: &mut impl ConnectionLike, timeout: Duration) -> RedisResult<()> {
    let answer = tokio::time::timeout(timeout, redis::cmd("PING").query_async::<()>(conn))
        .await
        .map_err(|_| timed_out_error(timeout))?;
    match answer {
        Err(e) if e.code() == Some("NOPERM") => {
            tracing::debug!(
                target: "aisix::redis",
                "the configured Redis user is authenticated but not permitted to PING; \
                 taking the refusal itself as proof the credential was accepted"
            );
            Ok(())
        }
        other => other,
    }
}

/// What one direct connection to a data node established.
enum Probe {
    /// It answered, on an authenticated connection.
    Answered,
    /// It answered, and it refused the settings we connected with.
    Refused(redis::RedisError),
    /// Nothing usable came back. Not evidence of anything.
    Silent,
}

/// The shape a refusal is reported in, whatever the server called it.
///
/// `InvalidClientConfig` because that is what it is — the operator wrote
/// something this server will not accept — and because it is already the
/// kind [`is_permanent_config_error`] treats as never coming good on a
/// retry. The server's own words are kept as the detail, since they are
/// the only part that says WHICH setting.
fn settings_refused(e: redis::RedisError) -> redis::RedisError {
    redis::RedisError::from((
        redis::ErrorKind::InvalidClientConfig,
        "redis refused the configured connection settings",
        e.to_string(),
    ))
}

/// Ask the data node directly whether it refuses the settings we would
/// connect with, and return the refusal if it does.
///
/// This exists because none of the three drivers will say so. Each hides
/// it differently and all three land in the same place — the gateway
/// concludes the backend is merely unreachable, binds its listeners and
/// serves fail-open for the life of the process against configuration
/// that is never going to start working, while the log blames the
/// network:
///
/// - `single` — the connection manager's retry ladder swallows every
///   attempt's error but the last, and runs past the boot budget, so the
///   caller only ever sees the budget expire. This catches a refused
///   password and equally a `database` the server does not have, which
///   the same ladder hides the same way.
/// - `cluster` — a seed that refuses is dropped from the initial
///   connection map; an empty map is an `IoError`.
/// - `sentinel` — a master that refuses fails discovery's `ROLE` check
///   and is skipped, so the group reads as not found.
///
/// A plain connection has none of that machinery, so the server's own
/// answer is what returns. It runs a command rather than only
/// connecting, because connecting proves nothing: a `requirepass` server
/// accepts the socket and refuses the *commands*, which is how a
/// `password` that never reached the handshake could look like a healthy
/// boot and then fail every operation.
///
/// It reports ONLY a refusal. Every other outcome is discarded and the
/// mode's own error stands, so this can never turn a slow or unreachable
/// backend into a boot failure, and it can never be the thing that
/// decides how long a boot spends. `Ok(true)` means the data node
/// answered us, which is what lets the caller skip proving the same
/// thing a second time.
async fn settings_refusal(
    cfg: &RedisConnConfig,
    tls: &Option<redis::TlsCertificates>,
    timeout: Duration,
) -> Result<bool, redis::RedisError> {
    // Nothing configured that the server could reject, so there is
    // nothing to ask. Not an optimisation: it is what keeps a plain
    // deployment from paying for the question — one connection on a
    // healthy boot, and against an unreachable backend a share of the
    // budget, which would make an outage take longer to report than it
    // does today.
    if !settings_the_server_can_reject(cfg) {
        return Ok(false);
    }
    let probed = match cfg.mode {
        RedisMode::Single => match cfg.url.as_deref() {
            Some(url) => probe(url, cfg, tls, timeout).await,
            None => Probe::Silent,
        },
        RedisMode::Cluster => {
            // Serially, stopping at the first node that answers either
            // way: one answer settles the question for every node, since
            // the settings are applied to all of them identically. A
            // node that stays silent is not an answer, so the walk goes
            // on — and if they all do, the walk is free, because they
            // are the same sockets the connect just failed on.
            let mut seen = Probe::Silent;
            for node in &cfg.nodes {
                seen = probe(node, cfg, tls, timeout).await;
                if !matches!(seen, Probe::Silent) {
                    break;
                }
            }
            seen
        }
        RedisMode::Sentinel => match sentinel_master_address(cfg, tls, timeout).await {
            Some(master) => probe(&master, cfg, tls, timeout).await,
            None => Probe::Silent,
        },
    };
    match probed {
        Probe::Refused(e) => Err(e),
        Probe::Answered => Ok(true),
        Probe::Silent => Ok(false),
    }
}

/// Whether anything is configured that the data node itself can reject —
/// a credential, from the explicit fields or from inside its own URL, or
/// a `database` it may not have.
///
/// The Sentinel-discovered master has no URL of its own, so in that mode
/// the fields are the only source; the `sentinels` URLs carry the
/// sentinels' credentials, which are a different thing. `database` is
/// not counted for `cluster`, which never sends it.
fn settings_the_server_can_reject(cfg: &RedisConnConfig) -> bool {
    if cfg.username.is_some() || cfg.password.is_some() {
        return true;
    }
    if cfg.mode != RedisMode::Cluster && cfg.database.is_some_and(|db| db != 0) {
        return true;
    }
    let in_url = |url: &String| {
        url.trim().into_connection_info().is_ok_and(|i| {
            i.redis.username.is_some() || i.redis.password.is_some() || i.redis.db != 0
        })
    };
    match cfg.mode {
        RedisMode::Single => cfg.url.as_ref().is_some_and(in_url),
        RedisMode::Cluster => cfg.nodes.iter().any(in_url),
        RedisMode::Sentinel => false,
    }
}

/// One plain, single-attempt connection to `url` with the configured
/// settings applied, plus one command on it.
///
/// The handshake and the command are classified apart on purpose. The
/// handshake runs only `AUTH` and `SELECT`, so a server error raised
/// there is a refusal of the configuration by construction, whatever
/// kind the driver gave it. A server error raised by the command after
/// it is a refusal only when it says so.
async fn probe(
    url: &str,
    cfg: &RedisConnConfig,
    tls: &Option<redis::TlsCertificates>,
    timeout: Duration,
) -> Probe {
    let url = insecure_url(url.trim(), cfg);
    let Ok(mut info) = url.as_str().into_connection_info() else {
        return Probe::Silent;
    };
    apply_explicit_settings(&mut info.redis, cfg);
    let client = match tls {
        Some(certs) => redis::Client::build_with_tls(info, certs.clone()),
        None => redis::Client::open(info),
    };
    let Ok(client) = client else {
        return Probe::Silent;
    };
    let mut conn = match client
        .get_multiplexed_async_connection_with_config(&conn_config(timeout))
        .await
    {
        Ok(conn) => conn,
        // `IoError` and the timeouts are the unreachable case, which is
        // not ours to report. Anything else came from the server.
        Err(e) if e.kind() == redis::ErrorKind::IoError => return Probe::Silent,
        Err(e) => return Probe::Refused(settings_refused(e)),
    };
    match ping(&mut conn, timeout).await {
        Ok(()) => Probe::Answered,
        Err(e) if is_permanent_config_error(&e) => Probe::Refused(settings_refused(e)),
        Err(_) => Probe::Silent,
    }
}

/// `host:port` of the monitored master, asked of the sentinels directly.
///
/// redis-rs offers no way to get the address without also connecting to
/// it and validating it — which is the step that swallows the refusal
/// this is trying to find. So the raw command, on the first sentinel
/// that answers it. The sentinels carry their own credentials in their
/// URLs, independent of the master's, and are reached exactly the way
/// the real discovery reaches them — same `insecure_url` treatment, same
/// trust material — or this would go silent on every TLS deployment and
/// take the diagnostic with it.
async fn sentinel_master_address(
    cfg: &RedisConnConfig,
    tls: &Option<redis::TlsCertificates>,
    timeout: Duration,
) -> Option<String> {
    let master_name = cfg.master_name.as_deref()?;
    for sentinel in &cfg.sentinels {
        let sentinel = insecure_url(sentinel.trim(), cfg);
        let client = match tls {
            Some(certs) => redis::Client::build_with_tls(sentinel.as_str(), certs.clone()),
            None => redis::Client::open(sentinel.as_str()),
        };
        let Ok(client) = client else { continue };
        let addr = async {
            let mut conn = client
                .get_multiplexed_async_connection_with_config(&conn_config(timeout))
                .await?;
            tokio::time::timeout(
                timeout,
                redis::cmd("SENTINEL")
                    .arg("get-master-addr-by-name")
                    .arg(master_name)
                    .query_async::<(String, String)>(&mut conn),
            )
            .await
            .map_err(|_| timed_out_error(timeout))?
        }
        .await;
        if let Ok((host, port)) = addr {
            // The master's scheme follows the sentinels', the same
            // uniform-TLS assumption the sentinel connect itself makes.
            let scheme = if sentinel.starts_with("rediss://") {
                "rediss"
            } else {
                "redis"
            };
            return Some(format!("{scheme}://{host}:{port}"));
        }
    }
    None
}

// Every command from every consumer — `Script::invoke_async`,
// `cmd().query_async`, pipelines — funnels through this impl, which is
// why the budget and the breaker live here rather than in each store.
impl ConnectionLike for RedisConnHandle {
    fn req_packed_command<'a>(
        &'a mut self,
        cmd: &'a redis::Cmd,
    ) -> redis::RedisFuture<'a, redis::Value> {
        let guard = Arc::clone(&self.guard);
        let fut = match &mut self.inner {
            HandleKind::Single(c) => c.req_packed_command(cmd),
            HandleKind::Cluster(c) => c.req_packed_command(cmd),
            HandleKind::Sentinel(c) => c.req_packed_command(cmd),
        };
        Box::pin(async move { Guard::run(&guard, fut).await })
    }

    fn req_packed_commands<'a>(
        &'a mut self,
        cmd: &'a redis::Pipeline,
        offset: usize,
        count: usize,
    ) -> redis::RedisFuture<'a, Vec<redis::Value>> {
        let guard = Arc::clone(&self.guard);
        let fut = match &mut self.inner {
            HandleKind::Single(c) => c.req_packed_commands(cmd, offset, count),
            HandleKind::Cluster(c) => c.req_packed_commands(cmd, offset, count),
            HandleKind::Sentinel(c) => c.req_packed_commands(cmd, offset, count),
        };
        Box::pin(async move { Guard::run(&guard, fut).await })
    }

    fn get_db(&self) -> i64 {
        match &self.inner {
            HandleKind::Single(c) => c.get_db(),
            HandleKind::Cluster(c) => c.get_db(),
            HandleKind::Sentinel(c) => c.get_db(),
        }
    }
}

/// Build a [`RedisConn`] from operator config. Validates connectivity
/// eagerly: a single/cluster handshake or an initial sentinel master
/// resolution must succeed, so a misconfigured backend fails at boot
/// rather than per request. Assumes [`RedisConnConfig::validate`] already
/// passed (the boot path validates before calling this).
pub async fn connect(cfg: &RedisConnConfig) -> RedisResult<RedisConn> {
    connect_with(cfg, &FailurePolicy::new(cfg)).await
}

/// [`connect_with`], bounded in aggregate by the subsystem's command
/// budget (`redis.timeout_secs`) rather than only per attempt.
///
/// `connect_with` sets the budget natively on the driver, which bounds
/// each connection ATTEMPT — but not the call, because the single-node
/// connection manager retries the initial connect on the driver's own
/// schedule before it reports a failure. That schedule is not derived
/// from anything the operator configured and it runs to minutes: a boot
/// against a black-holed Redis measured **eight minutes** on `redis`
/// 0.27 at `timeout_secs: 2`, with no log line and no error until the
/// last attempt gave up. Boot awaits this connection before it binds a
/// listener, so the gateway answered nothing at all for that whole
/// stretch. The budget the operator set is what a boot may spend on it.
///
/// Reconnects after a successful start are NOT bounded here — they run
/// inside the driver on that same schedule, behind the breaker and the
/// per-command budget, so they cost a caller nothing.
pub async fn connect_bounded(
    cfg: &RedisConnConfig,
    policy: &FailurePolicy,
) -> RedisResult<RedisConn> {
    let per_attempt = policy.0.timeout;
    let budget = discovery_budget(configured_endpoints(cfg), per_attempt);
    match tokio::time::timeout(budget, connect_with(cfg, policy)).await {
        Ok(r) => r,
        Err(_) => Err(connect_timed_out_error(budget, per_attempt)),
    }
}

/// How many endpoints a connect may have to pay for before it lands.
fn configured_endpoints(cfg: &RedisConnConfig) -> usize {
    // Trimmed-empty entries are what `validate` tolerates and what
    // `connect_with` then filters out, so they are never dialled and must
    // not each buy a budget the walk will not spend.
    let live = |urls: &[String]| urls.iter().filter(|u| !u.trim().is_empty()).count();
    match cfg.mode {
        RedisMode::Single => 0,
        RedisMode::Cluster => live(&cfg.nodes),
        RedisMode::Sentinel => live(&cfg.sentinels),
    }
}

/// What a whole connect may spend, which is NOT one command budget in
/// every topology: one per endpoint it may have to pay for, plus one for
/// the connection it ends in.
///
/// `sentinel` discovery walks its sentinel list serially. `cluster`
/// dials its seeds concurrently, but then walks the resulting connection
/// map serially for `CLUSTER SLOTS` and again to dial the slot map's
/// nodes — so either way a list whose front entries are unreachable is
/// what spends the budgets, and that is the normal case those topologies
/// exist for. One budget for the whole call would fail a connect that is
/// working exactly as designed, in the one deployment shape built to
/// survive a dead node.
///
/// For `cluster` this is an approximation rather than a strict ceiling:
/// the slot map can name more nodes than the seed list does. Reaching it
/// needs several of them black-holed at once, by which point the cluster
/// itself is already broken.
///
/// One expression, used by both the outer bound in [`connect_bounded`]
/// and sentinel's own discovery bound inside [`connect_with`], so the
/// two cannot drift. They are equal, and the outer one starts marginally
/// earlier (before the TLS material is read), so the outer is the one
/// that reports — with the same budget in its message.
fn discovery_budget(endpoints: usize, per_attempt: Duration) -> Duration {
    let budgets = u32::try_from(endpoints)
        .unwrap_or(u32::MAX)
        .saturating_add(1);
    per_attempt.saturating_mul(budgets)
}

/// The backend's address with the scheme, any userinfo and any path
/// stripped, so a diagnostic can name WHICH Redis is unreachable. The
/// configured URL itself must never be logged — it carries the password
/// in `redis://user:pass@host` form.
pub fn endpoint_label(cfg: &RedisConnConfig) -> String {
    /// What a label says when the text it was given is not a URL whose
    /// host can be identified. Never echo the input: the thing that makes
    /// it unparseable is usually an unescaped character in the password.
    const UNPARSEABLE: &str = "<unparseable redis endpoint>";

    fn host(url: &str) -> &str {
        let rest = url.split_once("://").map_or(url, |(_, r)| r);
        // Authority first: an `@` can appear after the host too (in a
        // path or a query), and only the one inside the authority is
        // userinfo.
        let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
        // Then the LAST `@` of the authority, so a password containing
        // one cannot leave a fragment of itself in front of the host.
        let candidate = authority.rsplit_once('@').map_or(authority, |(_, r)| r);
        // And finally a shape check, because the two steps above trust
        // the input to be well formed and a password is exactly what is
        // most likely to make it not be. `redis://user:pw/x@host:6379`
        // has authority `user:pw` by RFC 3986 — the `@` is in the path —
        // so slicing alone would print the password. Emitting only text
        // that looks like `host[:port]` makes that impossible whatever
        // the input.
        if is_host_port(candidate) {
            candidate
        } else {
            UNPARSEABLE
        }
    }

    /// `host`, `host:port`, or `[v6]:port` — nothing else.
    fn is_host_port(s: &str) -> bool {
        let (host, port) = match s.rsplit_once(':') {
            Some((h, p)) if !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()) => (h, Some(p)),
            _ => (s, None),
        };
        let host = match host.strip_prefix('[').and_then(|h| h.strip_suffix(']')) {
            // An IPv6 literal: hex groups and separators only.
            Some(v6) => {
                return !v6.is_empty()
                    && v6
                        .bytes()
                        .all(|b| b.is_ascii_hexdigit() || b == b':' || b == b'.')
                    && port.is_none_or(|p| p.len() <= 5);
            }
            None => host,
        };
        !host.is_empty()
            && host
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'-' || b == b'_')
    }
    fn hosts(urls: &[String]) -> String {
        urls.iter().map(|u| host(u)).collect::<Vec<_>>().join(",")
    }
    match cfg.mode {
        RedisMode::Single => host(cfg.url.as_deref().unwrap_or_default()).to_string(),
        RedisMode::Cluster => hosts(&cfg.nodes),
        RedisMode::Sentinel => format!(
            "master {} via {}",
            cfg.master_name.as_deref().unwrap_or_default(),
            hosts(&cfg.sentinels)
        ),
    }
}

/// [`connect`] against an existing [`FailurePolicy`].
///
/// Every connection a subsystem opens from the same `redis:` block must
/// be built this way from ONE policy, or the subsystem's operations each
/// pay the command budget separately — see [`FailurePolicy`].
pub async fn connect_with(cfg: &RedisConnConfig, policy: &FailurePolicy) -> RedisResult<RedisConn> {
    let tls = load_tls(cfg)?;
    warn_on_credential_shadowing(cfg);
    let guard = Arc::clone(&policy.0);
    let timeout = guard.timeout;
    let inner = match cfg.mode {
        RedisMode::Single => {
            let url = insecure_url(cfg.url.as_deref().unwrap_or_default(), cfg);
            let mut info = url.as_str().into_connection_info()?;
            apply_explicit_settings(&mut info.redis, cfg);
            let client = match &tls {
                Some(certs) => redis::Client::build_with_tls(info, certs.clone())?,
                None => redis::Client::open(info)?,
            };
            // Ask the server BEFORE the manager does, because after it
            // there is nothing left to ask with: `ConnectionManager`'s
            // initial connect runs a retry ladder built from the
            // driver's defaults, reports only the LAST attempt's error,
            // and outlasts the whole boot budget. A Redis that answers
            // and refuses the credential therefore had exactly one
            // outcome — the budget expired, the operator was told the
            // backend had timed out, and the gateway served fail-open
            // for the life of the process against a password that was
            // never going to become right.
            let proven = settings_refusal(cfg, &tls, timeout).await?;
            // Only the two timeouts are set; the retry policy stays the
            // library default, which is what the boot retry loop is tuned
            // around.
            let manager_cfg = ConnectionManagerConfig::new()
                .set_connection_timeout(timeout)
                .set_response_timeout(timeout);
            let mut conn = ConnectionManager::new_with_config(client, manager_cfg).await?;
            // Only when the probe above did not already run a command on
            // this server. Proving the same thing twice would spend a
            // second round trip out of the one budget the whole connect
            // gets, and a healthy-but-slow endpoint would start degraded
            // for it.
            if !proven {
                ping(&mut conn, timeout).await?;
            }
            tracing::info!(
                target: "aisix::redis",
                mode = "single",
                timeout_secs = timeout.as_secs(),
                "connected"
            );
            ConnKind::Single(conn)
        }
        RedisMode::Cluster => {
            let nodes: Vec<String> = cfg
                .nodes
                .iter()
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .map(|s| insecure_url(s, cfg))
                .collect();
            // ACL creds for the nodes can travel in the node URLs, or be
            // set explicitly here (applied to every node). Cluster has no
            // DB index, so `database` is ignored in this mode.
            let mut builder = ClusterClient::builder(nodes)
                .connection_timeout(timeout)
                .response_timeout(timeout);
            if let Some(u) = &cfg.username {
                builder = builder.username(u.clone());
            }
            if let Some(p) = &cfg.password {
                builder = builder.password(p.clone());
            }
            if let Some(certs) = &tls {
                builder = builder.certs(certs.clone());
            }
            let client = builder.build()?;
            let conn = match client.get_async_connection().await {
                Ok(conn) => conn,
                // A seed that refuses the credential is not added to the
                // initial connection map, and an empty map is reported as
                // `Failed to create initial connections` — the same
                // `IoError` an unplugged cluster gets. Unlike `single`
                // this arrives fast, with the budget barely touched, so
                // the question can be asked after the fact.
                Err(e) => {
                    return Err(settings_refusal(cfg, &tls, timeout)
                        .await
                        .err()
                        .unwrap_or(e))
                }
            };
            // No PING here. redis-rs routes it to ALL masters and fails
            // the command unless every one of them answers, so a cluster
            // running with a master down — the failure the topology
            // exists to survive — would stop connecting. The probe above
            // is what proves the credential, against one node.
            tracing::info!(
                target: "aisix::redis",
                mode = "cluster",
                nodes = cfg.nodes.len(),
                timeout_secs = timeout.as_secs(),
                "connected"
            );
            ConnKind::Cluster(conn)
        }
        RedisMode::Sentinel => {
            let sentinels: Vec<String> = cfg
                .sentinels
                .iter()
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .map(|s| insecure_url(s, cfg))
                .collect();
            let sentinels_len = sentinels.len().max(1);
            let master_name = cfg.master_name.clone().unwrap_or_default();
            // The master/data node may need its own auth and TLS; the
            // sentinels themselves carry theirs in `sentinels` URLs. Derive
            // the master's TLS from whether the sentinels are reached over
            // `rediss://`, the common uniform-TLS deployment.
            let tls_mode = sentinels
                .first()
                .filter(|u| u.starts_with("rediss://"))
                .map(|_| {
                    if cfg.tls.verify {
                        redis::TlsMode::Secure
                    } else {
                        redis::TlsMode::Insecure
                    }
                });
            // `SentinelClient::build` takes no `TlsCertificates`, so the
            // master connection redis-rs opens after discovery can only
            // use the built-in trust store. Say so rather than let a
            // configured CA look applied — `SSL_CERT_FILE` is the working
            // alternative in this one mode.
            if tls_mode.is_some() && cfg.tls.ca_file.is_some() {
                tracing::warn!(
                    target: "aisix::redis",
                    "redis.tls.ca_file is not applied in sentinel mode: the client library \
                     accepts no custom trust roots for the sentinel-discovered master. Put \
                     the CA in the system trust store, or point SSL_CERT_FILE at it."
                );
            }
            // Auth/DB for the discovered master. It has no URL of its own,
            // so ACL username/password and the DB index are configured
            // here; this is independent of the sentinels' own auth.
            let redis_connection_info =
                if cfg.username.is_some() || cfg.password.is_some() || cfg.database.is_some() {
                    Some(redis::RedisConnectionInfo {
                        db: cfg.database.unwrap_or(0),
                        username: cfg.username.clone(),
                        password: cfg.password.clone(),
                        ..Default::default()
                    })
                } else {
                    None
                };
            let node_info = SentinelNodeConnectionInfo {
                tls_mode,
                redis_connection_info,
            };
            let mut client = SentinelClient::build(
                sentinels,
                master_name,
                Some(node_info),
                SentinelServerType::Master,
            )?;
            // Eagerly resolve the master once so a broken sentinel/master
            // setup fails at boot, and seed the cache.
            //
            // Discovery talks to the sentinels before it reaches the
            // master, and `connection_timeout` bounds only the individual
            // connect attempts inside it — so the whole exchange gets an
            // outer budget too, or an unreachable sentinel stalls the
            // boot attempt indefinitely.
            let discovery_timeout = discovery_budget(sentinels_len, timeout);
            let discovered = tokio::time::timeout(
                discovery_timeout,
                client.get_async_connection_with_config(&conn_config(timeout)),
            )
            .await
            .map_err(|_| connect_timed_out_error(discovery_timeout, timeout))
            .and_then(|r| r);
            let mut conn = match discovered {
                Ok(conn) => conn,
                // Discovery validates a candidate master by running
                // `ROLE` on it, so a master that refuses the credential
                // fails that check and is skipped — and a correctly
                // configured group comes back as
                // `MasterNameNotFoundBySentinel`, which is also what a
                // typo'd `master_name` returns. Same fast failure as
                // cluster, same question asked afterwards.
                Err(e) => {
                    return Err(settings_refusal(cfg, &tls, timeout)
                        .await
                        .err()
                        .unwrap_or(e))
                }
            };
            // One connection to one master, so this is a single round
            // trip — unlike the cluster case above.
            ping(&mut conn, timeout).await?;
            tracing::info!(
                target: "aisix::redis",
                mode = "sentinel",
                master = %cfg.master_name.as_deref().unwrap_or_default(),
                timeout_secs = timeout.as_secs(),
                "connected"
            );
            ConnKind::Sentinel(SentinelPool {
                client: Arc::new(Mutex::new(client)),
                cached: Arc::new(Mutex::new(Some(conn))),
                discovery_timeout,
            })
        }
    };
    // The first connection built against this policy is the one the
    // background prober tests Redis on while the breaker is open.
    lock(&guard.prober).get_or_insert_with(|| inner.clone());
    Ok(RedisConn { inner, guard })
}

/// Read the `redis.tls` PEM files into the shape redis-rs wants, or
/// `None` when the operator configured no custom trust material (the
/// built-in root set, plus `SSL_CERT_FILE`, then applies).
fn load_tls(cfg: &RedisConnConfig) -> RedisResult<Option<redis::TlsCertificates>> {
    let read = |path: &str, field: &str| -> RedisResult<Vec<u8>> {
        std::fs::read(path).map_err(|e| {
            redis::RedisError::from((
                redis::ErrorKind::InvalidClientConfig,
                "redis TLS material could not be read",
                format!("redis.tls.{field}: read {path}: {e}"),
            ))
        })
    };

    let root_cert = match &cfg.tls.ca_file {
        Some(path) => Some(read(path, "ca_file")?),
        None => None,
    };
    let client_tls = match (&cfg.tls.client_cert_file, &cfg.tls.client_key_file) {
        // The mismatched pairs are rejected by `RedisConnConfig::validate`.
        (Some(cert), Some(key)) => Some(redis::ClientTlsConfig {
            client_cert: read(cert, "client_cert_file")?,
            client_key: read(key, "client_key_file")?,
        }),
        _ => None,
    };

    if root_cert.is_none() && client_tls.is_none() {
        return Ok(None);
    }
    Ok(Some(redis::TlsCertificates {
        client_tls,
        root_cert,
    }))
}

/// redis-rs carries "do not verify the server certificate" in the URL
/// rather than in a builder option, as the `#insecure` fragment on a
/// `rediss://` URL. Translate `redis.tls.verify: false` into that.
///
/// Left alone for a plaintext `redis://` URL, where the fragment is
/// rejected outright, and for a URL that already carries a fragment,
/// which the operator set deliberately.
fn insecure_url(url: &str, cfg: &RedisConnConfig) -> String {
    if cfg.tls.verify || !url.starts_with("rediss://") || url.contains('#') {
        return url.to_string();
    }
    // The fragment must follow a path segment: redis-rs parses the URL
    // with the `url` crate, and `rediss://host:6379#insecure` leaves the
    // fragment attached to an empty path, which it then reads as a
    // database index.
    if url.rsplit('/').next().is_some_and(|s| s.contains(':')) {
        format!("{url}/#insecure")
    } else {
        format!("{url}#insecure")
    }
}

/// Re-export so dependents don't need a direct `redis` dependency just to
/// name the connect error.
pub use redis::RedisError as ConnectError;

#[cfg(test)]
mod tls_tests {
    use super::*;
    use aisix_core::config::OutboundTlsConfig;

    fn cfg_with(url: &str, verify: bool) -> RedisConnConfig {
        RedisConnConfig {
            mode: RedisMode::Single,
            url: Some(url.to_string()),
            tls: OutboundTlsConfig {
                verify,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn verify_true_leaves_the_url_alone() {
        let cfg = cfg_with("rediss://redis.internal:6379", true);
        assert_eq!(
            insecure_url("rediss://redis.internal:6379", &cfg),
            "rediss://redis.internal:6379"
        );
    }

    /// redis-rs reads the fragment only after a path separator; without
    /// the trailing slash it parses `6379#insecure` as the database
    /// index and the connection fails to open at all.
    #[test]
    fn verify_false_appends_the_insecure_fragment_after_a_path_separator() {
        let cfg = cfg_with("rediss://redis.internal:6379", false);
        assert_eq!(
            insecure_url("rediss://redis.internal:6379", &cfg),
            "rediss://redis.internal:6379/#insecure"
        );
    }

    #[test]
    fn verify_false_keeps_an_existing_path() {
        let cfg = cfg_with("rediss://redis.internal:6379/2", false);
        assert_eq!(
            insecure_url("rediss://redis.internal:6379/2", &cfg),
            "rediss://redis.internal:6379/2#insecure"
        );
    }

    /// A plaintext connection never negotiates TLS, and redis-rs rejects
    /// any fragment on a `redis://` URL — adding one would turn "verify
    /// is off" into "the backend does not connect".
    #[test]
    fn verify_false_does_not_touch_a_plaintext_url() {
        let cfg = cfg_with("redis://redis.internal:6379", false);
        assert_eq!(
            insecure_url("redis://redis.internal:6379", &cfg),
            "redis://redis.internal:6379"
        );
    }

    #[test]
    fn no_tls_material_configured_leaves_the_default_trust_store() {
        let cfg = cfg_with("rediss://redis.internal:6379", true);
        assert!(load_tls(&cfg).unwrap().is_none());
    }

    #[test]
    fn an_unreadable_ca_file_names_the_field_and_the_path() {
        let mut cfg = cfg_with("rediss://redis.internal:6379", true);
        cfg.tls.ca_file = Some("/nonexistent/redis-ca.pem".into());
        // `TlsCertificates` is not `Debug`, so `unwrap_err` is out.
        let Err(err) = load_tls(&cfg) else {
            panic!("an unreadable ca_file must fail the connect")
        };
        let err = err.to_string();
        assert!(err.contains("redis.tls.ca_file"), "{err}");
        assert!(err.contains("/nonexistent/redis-ca.pem"), "{err}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn single_mode_bad_url_errors() {
        let cfg = RedisConnConfig {
            mode: RedisMode::Single,
            url: Some("not-a-url".into()),
            ..Default::default()
        };
        // Any error is fine — the point is it returns Err, not panics.
        assert!(connect(&cfg).await.is_err());
    }
}

/// The failure policy in isolation: what a command does when Redis stops
/// answering, and what the command after it does.
///
/// Exercised here rather than only through a live Redis because the
/// property under test is a *timing* one — that the caller regains
/// control — and a store integration test can only observe it by waiting
/// out the very budget it is meant to prove exists.
#[cfg(test)]
mod guard_tests {
    use super::*;
    use std::future::pending;

    /// A policy with no registered prober, which is the fallback shape
    /// ([`Guard::trip`]): these cases are about the breaker's own
    /// arithmetic, and the prober is covered against a real socket in
    /// `probe_tests`.
    fn guard(timeout_ms: u64, window_ms: u64) -> Arc<Guard> {
        Arc::new(Guard {
            timeout: Duration::from_millis(timeout_ms),
            breaker: Breaker::new(Duration::from_millis(window_ms)),
            prober: std::sync::Mutex::new(None),
        })
    }

    fn dropped_connection() -> redis::RedisError {
        redis::RedisError::from(std::io::Error::from(std::io::ErrorKind::ConnectionReset))
    }

    fn server_side_error() -> redis::RedisError {
        // What a live Redis returns for, say, a bad script: it answered,
        // so it is not a connectivity failure.
        redis::RedisError::from((redis::ErrorKind::ExtensionError, "ERR bad script"))
    }

    /// The bug: with no budget the command never returns, so the caller's
    /// fail-open branch is never reached and the request hangs.
    #[tokio::test]
    async fn a_peer_that_never_answers_gives_the_caller_control_back() {
        let g = guard(80, 5_000);
        let started = Instant::now();
        let err = Guard::run(&g, pending::<RedisResult<()>>())
            .await
            .expect_err("a silent peer must surface as an error, not a hang");
        assert!(err.is_timeout() || err.is_io_error(), "{err:?}");
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "{:?}",
            started.elapsed()
        );
    }

    /// Paying the budget on every request during an outage is still a
    /// multi-second latency floor for as long as the outage lasts.
    #[tokio::test]
    async fn the_command_after_a_failure_short_circuits_without_waiting() {
        let g = guard(80, 5_000);
        let _ = Guard::run(&g, pending::<RedisResult<()>>()).await;

        let started = Instant::now();
        let err = Guard::run(&g, pending::<RedisResult<()>>())
            .await
            .expect_err("the breaker is open");
        assert!(
            started.elapsed() < Duration::from_millis(40),
            "short-circuit must not pay the budget, took {:?}",
            started.elapsed()
        );
        assert!(err.is_io_error(), "{err:?}");
        assert!(err.to_string().contains("cool-off"), "{err}");
    }

    /// The window is a cool-off, not a latch: Redis coming back must be
    /// noticed without anything resetting the breaker by hand.
    #[tokio::test]
    async fn the_window_expires_and_the_next_command_probes_for_real() {
        let g = guard(80, 60);
        let _ = Guard::run(&g, pending::<RedisResult<()>>()).await;
        assert!(g.breaker.is_open());

        tokio::time::sleep(Duration::from_millis(120)).await;
        assert!(!g.breaker.is_open(), "the window must expire on its own");

        Guard::run(&g, async { Ok::<_, redis::RedisError>(7) })
            .await
            .expect("the probe reaches Redis again");
        assert!(!g.breaker.is_open(), "a success closes the breaker");
    }

    /// The window has to date from when the failing command RETURNED.
    /// Dating it from when the command started would leave it already
    /// expired by the time that command gave up whenever the window is
    /// not longer than the budget — and the next operation of the same
    /// request would then probe and pay the budget all over again, which
    /// is how one request came to pay four.
    #[tokio::test]
    async fn the_window_dates_from_when_the_command_returned() {
        // Window shorter than the budget, so "from start" and "from
        // return" give opposite answers. Both are generous: the margin
        // between them is the whole tolerance for scheduling delay on a
        // loaded CI runner.
        let g = guard(400, 300);
        Guard::run(&g, pending::<RedisResult<()>>())
            .await
            .expect_err("the command spends its budget and gives up");

        let started = Instant::now();
        let err = Guard::run(&g, pending::<RedisResult<()>>())
            .await
            .expect_err("the operation behind it short-circuits");
        assert!(
            started.elapsed() < Duration::from_millis(40),
            "the cool-off must still be open when the failure returns, took {:?}",
            started.elapsed()
        );
        assert!(err.to_string().contains("cool-off"), "{err}");
    }

    /// Two connections of one subsystem share the policy, so the budget a
    /// request spends on the first is not spent again on the second.
    #[tokio::test]
    async fn connections_sharing_a_policy_share_the_cool_off() {
        let policy = FailurePolicy(guard(2_000, 5_000));
        let a = Arc::clone(&policy.0);
        let b = Arc::clone(&policy.0);

        Guard::run(&a, async { Err::<(), _>(dropped_connection()) })
            .await
            .expect_err("the first connection fails");

        let started = Instant::now();
        Guard::run(&b, pending::<RedisResult<()>>())
            .await
            .expect_err("the second connection short-circuits on the shared cool-off");
        assert!(
            started.elapsed() < Duration::from_millis(40),
            "a second connection of the same subsystem must not pay the budget \
             again, took {:?}",
            started.elapsed()
        );
    }

    /// A cool-off that only gates the window itself still lets every
    /// command that arrives while the probe is in flight pay the full
    /// budget — during a sustained outage that is most of them, and no
    /// serial test can see it.
    #[tokio::test]
    async fn only_one_command_probes_when_the_window_expires() {
        let g = guard(2_000, 60);
        Guard::run(&g, async { Err::<(), _>(dropped_connection()) })
            .await
            .expect_err("the failure opens the breaker");
        tokio::time::sleep(Duration::from_millis(120)).await;
        assert!(!g.breaker.is_open(), "the window has expired");

        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let (admitted_tx, admitted_rx) = tokio::sync::oneshot::channel::<()>();
        let probe = tokio::spawn({
            let g = Arc::clone(&g);
            async move {
                Guard::run(&g, async move {
                    // Sent from inside the guarded future, so receiving it
                    // proves `admit` already ran. Yielding would only
                    // *probably* get the task that far.
                    let _ = admitted_tx.send(());
                    let _ = rx.await;
                    Ok::<_, redis::RedisError>(1)
                })
                .await
            }
        });
        admitted_rx.await.expect("the probe was admitted");

        let started = Instant::now();
        let err = Guard::run(&g, pending::<RedisResult<()>>())
            .await
            .expect_err("only the probe reaches Redis");
        assert!(
            started.elapsed() < Duration::from_millis(40),
            "a command behind the probe must not pay the budget, took {:?}",
            started.elapsed()
        );
        assert!(err.to_string().contains("cool-off"), "{err}");

        tx.send(()).expect("the probe is still waiting");
        probe.await.expect("join").expect("the probe succeeds");
        assert!(
            !g.breaker.is_open(),
            "a successful probe closes the breaker"
        );
    }

    /// A command that was already in flight when the outage began can
    /// land its success *after* another command has opened the breaker.
    /// Closing on that stale evidence puts every request behind it back
    /// on the full budget.
    #[tokio::test]
    async fn a_success_that_started_first_does_not_wipe_a_newer_cool_off() {
        let g = guard(2_000, 5_000);
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();

        let (admitted_tx, admitted_rx) = tokio::sync::oneshot::channel::<()>();
        let inflight = tokio::spawn({
            let g = Arc::clone(&g);
            async move {
                Guard::run(&g, async move {
                    let _ = admitted_tx.send(());
                    let _ = rx.await;
                    Ok::<_, redis::RedisError>(1)
                })
                .await
            }
        });
        // Sent from inside the guarded future, so receiving it proves the
        // generation was read before anything below fails. Yielding would
        // only *probably* get the task that far.
        admitted_rx.await.expect("the in-flight command started");

        Guard::run(&g, async { Err::<(), _>(dropped_connection()) })
            .await
            .expect_err("the concurrent command fails");
        assert!(g.breaker.is_open());

        tx.send(()).expect("the in-flight command is still waiting");
        inflight
            .await
            .expect("join")
            .expect("the in-flight command succeeds");
        assert!(
            g.breaker.is_open(),
            "a success from before the failure must not clear the cool-off"
        );
    }

    /// A prober whose deadline lapsed has its claim taken back by the
    /// caller that got through, and the failure of THAT caller starts a
    /// replacement. The original must find out and stop: renewing the
    /// replacement's window would put two probers on one connection —
    /// each keeping the other alive, and on a sentinel serializing on the
    /// same discovery lock, so probes get slower the longer the outage
    /// runs. A flag cannot tell the two apart; the claim is an id.
    #[tokio::test]
    async fn a_replacement_prober_takes_the_claim_from_the_first() {
        let g = guard(80, 40);
        Guard::run(&g, async { Err::<(), _>(dropped_connection()) })
            .await
            .expect_err("the failure opens the breaker");
        let first = g.breaker.claim_prober().expect("stands in for the task");

        // Nothing renews the deadline, so the caller after it is admitted
        // as the fallback probe and takes the claim back.
        tokio::time::sleep(Duration::from_millis(60)).await;
        Guard::run(&g, async { Err::<(), _>(dropped_connection()) })
            .await
            .expect_err("the fallback probe reaches Redis and fails");
        let second = g
            .breaker
            .claim_prober()
            .expect("the claim is free for a replacement");
        assert_ne!(first, second);

        assert!(
            !g.breaker.hold_off(first, Duration::from_secs(1)),
            "the first prober must not renew the replacement's window"
        );
        assert!(
            !g.breaker.close_from_prober(first, g.breaker.generation()),
            "nor close the breaker on evidence the replacement never asked for"
        );
        assert!(
            g.breaker.hold_off(second, Duration::from_secs(1)),
            "the prober that owns the claim still can"
        );
    }

    /// A reply from a live Redis — a script error, `WRONGTYPE`, an ACL
    /// refusal — is not an outage. Tripping on it would short-circuit
    /// the whole cool-off of healthy traffic every time one command is
    /// wrong.
    #[tokio::test]
    async fn an_error_redis_itself_reported_does_not_open_the_breaker() {
        let g = guard(80, 5_000);
        let err = Guard::run(&g, async { Err::<(), _>(server_side_error()) })
            .await
            .expect_err("the server error is passed through");
        assert!(!err.is_io_error(), "{err:?}");
        assert!(
            !g.breaker.is_open(),
            "a server reply is not a connectivity failure"
        );
    }
}

/// The background prober, against a real socket that can be black-holed
/// and healed. The breaker's own arithmetic is covered in `guard_tests`;
/// what these cases pin is WHO pays for re-testing Redis, which only
/// shows up when a command actually has to cross the network.
#[cfg(test)]
mod probe_tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    /// Command budget. Long enough that paying it is unmistakable next to
    /// a short-circuit, short enough to keep the case quick.
    const BUDGET: Duration = Duration::from_millis(900);
    /// Cool-off window. Stands in for [`BREAKER_WINDOW`], which is 30s and
    /// would make every case here a minute long.
    const WINDOW: Duration = Duration::from_millis(250);

    /// A server that speaks just enough RESP for redis-rs to connect and
    /// PING, and that can stop answering without closing the socket —
    /// what a dropped-packet policy or a paused container looks like from
    /// the client end. A refusal would be the cheap failure (the client
    /// learns at once and spends no budget) and would not exercise this.
    struct FakeRedis {
        url: String,
        answering: Arc<AtomicBool>,
        reset: tokio::sync::broadcast::Sender<()>,
    }

    impl FakeRedis {
        async fn start() -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
            let url = format!("redis://{}", listener.local_addr().expect("addr"));
            let answering = Arc::new(AtomicBool::new(true));
            let (reset, _) = tokio::sync::broadcast::channel(8);
            tokio::spawn({
                let answering = Arc::clone(&answering);
                let reset = reset.clone();
                async move {
                    while let Ok((sock, _)) = listener.accept().await {
                        tokio::spawn(serve(sock, Arc::clone(&answering), reset.subscribe()));
                    }
                }
            });
            Self {
                url,
                answering,
                reset,
            }
        }

        fn blackhole(&self) {
            self.answering.store(false, Ordering::SeqCst);
        }

        /// Answer again, and drop the sockets that were black-holed: a
        /// peer that comes back has not been holding the client's
        /// un-answered commands, and the driver reconnects on its own.
        fn heal(&self) {
            self.answering.store(true, Ordering::SeqCst);
            let _ = self.reset.send(());
        }
    }

    async fn serve(
        mut sock: tokio::net::TcpStream,
        answering: Arc<AtomicBool>,
        mut reset: tokio::sync::broadcast::Receiver<()>,
    ) {
        let mut buf = [0u8; 4096];
        let mut pending: Vec<u8> = Vec::new();
        loop {
            let read = tokio::select! {
                _ = reset.recv() => return,
                read = sock.read(&mut buf) => read,
            };
            match read {
                Ok(0) | Err(_) => return,
                Ok(n) => {
                    if !answering.load(Ordering::SeqCst) {
                        continue;
                    }
                    pending.extend_from_slice(&buf[..n]);
                    for _ in 0..take_commands(&mut pending) {
                        // Every command this fake sees is a handshake
                        // `CLIENT SETINFO` (whose reply redis-rs ignores)
                        // or the prober's PING.
                        if sock.write_all(b"+PONG\r\n").await.is_err() {
                            return;
                        }
                    }
                }
            }
        }
    }

    /// Consume whole RESP commands from `buf` and return how many: a
    /// `*N\r\n` header followed by N `$len\r\n<bytes>\r\n` bulk strings.
    fn take_commands(buf: &mut Vec<u8>) -> usize {
        let mut at = 0usize;
        let mut commands = 0usize;
        let line_end = |buf: &[u8], from: usize| {
            buf[from..]
                .windows(2)
                .position(|w| w == b"\r\n")
                .map(|p| from + p)
        };
        while let Some(end) = line_end(buf, at) {
            if buf[at] != b'*' {
                break;
            }
            let Ok(argc) = std::str::from_utf8(&buf[at + 1..end])
                .unwrap_or_default()
                .parse::<usize>()
            else {
                break;
            };
            let mut cursor = end + 2;
            let mut complete = true;
            for _ in 0..argc {
                let Some(head) = line_end(buf, cursor) else {
                    complete = false;
                    break;
                };
                let Ok(len) = std::str::from_utf8(&buf[cursor + 1..head])
                    .unwrap_or_default()
                    .parse::<usize>()
                else {
                    complete = false;
                    break;
                };
                cursor = head + 2 + len + 2;
                if cursor > buf.len() {
                    complete = false;
                    break;
                }
            }
            if !complete {
                break;
            }
            at = cursor;
            commands += 1;
        }
        buf.drain(..at);
        commands
    }

    fn policy(timeout: Duration, window: Duration) -> FailurePolicy {
        FailurePolicy(Arc::new(Guard {
            timeout,
            breaker: Breaker::new(window),
            prober: std::sync::Mutex::new(None),
        }))
    }

    /// One business command, timed. Fails open like every consumer does.
    async fn command(conn: &RedisConn) -> (RedisResult<()>, Duration) {
        let started = Instant::now();
        let outcome = match conn.acquire().await {
            Ok(mut handle) => redis::cmd("PING").query_async(&mut handle).await,
            Err(e) => Err(e),
        };
        (outcome, started.elapsed())
    }

    async fn connected(fake: &FakeRedis, policy: &FailurePolicy) -> RedisConn {
        let cfg = RedisConnConfig {
            url: Some(fake.url.clone()),
            ..Default::default()
        };
        connect_with(&cfg, policy).await.expect("connect")
    }

    /// The window expiring must not hand the next caller the bill for
    /// re-testing a Redis that is still down. It used to: the first
    /// command after the window was admitted as the probe and spent the
    /// whole budget discovering what the last one already knew — once per
    /// window, for as long as the outage lasted.
    #[tokio::test]
    async fn a_command_after_the_window_is_not_delayed_while_redis_is_still_down() {
        let fake = FakeRedis::start().await;
        let policy = policy(BUDGET, WINDOW);
        let conn = connected(&fake, &policy).await;

        fake.blackhole();
        let (outcome, elapsed) = command(&conn).await;
        outcome.expect_err("a black-holed Redis must surface as an error");
        assert!(
            elapsed >= BUDGET,
            "the first failure pays the budget: {elapsed:?}"
        );

        // Long enough for the prober to wake, spend the whole budget on a
        // PING that is never answered, and re-arm the window behind it.
        // Waiting only for the window to expire would prove less: the
        // command would still short-circuit, but on the prober being in
        // flight rather than on the window it re-armed.
        tokio::time::sleep(WINDOW + BUDGET + WINDOW).await;

        let (outcome, elapsed) = command(&conn).await;
        let err = outcome.expect_err("Redis is still down");
        assert!(
            elapsed < WINDOW,
            "a command arriving after the window must not pay the budget to \
             re-test a Redis that is still down, took {elapsed:?}"
        );
        assert!(err.to_string().contains("cool-off"), "{err}");
    }

    /// The prober is the only thing that closes the breaker now, so one
    /// that ends without closing it — task dropped, runtime gone — must
    /// not short-circuit the subsystem for the rest of the process. The
    /// window alone used to guarantee recovery; taking the probe off the
    /// request path must not take that guarantee with it.
    ///
    /// Driven through `acquire()`, which is how every consumer reaches
    /// Redis and which short-circuits on its own pre-flight check: a
    /// fallback that only `admit` knew about would never run on this
    /// path, and the subsystem would stay dead with Redis healthy.
    #[tokio::test]
    async fn a_prober_that_stops_without_closing_does_not_wedge_the_subsystem() {
        let fake = FakeRedis::start().await;
        let policy = policy(BUDGET, WINDOW);
        let conn = connected(&fake, &policy).await;

        // Claim the probe before the outage, so the failure below finds
        // the claim taken and starts no task: what is left behind when a
        // prober goes away mid-flight.
        assert!(policy.0.breaker.claim_prober().is_some());

        fake.blackhole();
        command(&conn)
            .await
            .0
            .expect_err("the outage opens the breaker");
        assert!(policy.0.breaker.is_open());

        fake.heal();
        // Past the deadline the failure armed, which nothing is renewing.
        tokio::time::sleep(WINDOW + BUDGET).await;

        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let (outcome, _) = command(&conn).await;
            if outcome.is_ok() {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "with Redis healthy and no prober alive, commands must reach it again"
            );
        }
    }

    /// And the flip side: nothing may have to arrive for the breaker to
    /// close. With the probe on the request path, a gateway seeing no
    /// traffic on that subsystem stayed in cool-off however long ago
    /// Redis came back, and the next request after that paid for the
    /// discovery.
    #[tokio::test]
    async fn redis_coming_back_closes_the_breaker_with_no_command_at_all() {
        let fake = FakeRedis::start().await;
        let policy = policy(BUDGET, WINDOW);
        let conn = connected(&fake, &policy).await;

        fake.blackhole();
        command(&conn)
            .await
            .0
            .expect_err("the outage opens the breaker");
        assert!(policy.0.breaker.is_open());

        fake.heal();

        // No command is issued in this loop: closing the breaker is the
        // prober's job, and the assertion is that it does it alone. The
        // window expiring is not the same thing — that leaves the breaker
        // waiting for evidence it has not got, which is the whole state
        // this change removes.
        let deadline = Instant::now() + Duration::from_secs(10);
        while !policy.0.breaker.is_closed() {
            assert!(
                Instant::now() < deadline,
                "the breaker must close on the prober's evidence, with no traffic"
            );
            tokio::time::sleep(WINDOW / 2).await;
        }

        let (outcome, elapsed) = command(&conn).await;
        outcome.expect("Redis is back, so the command goes through");
        assert!(elapsed < BUDGET, "{elapsed:?}");
    }
}

#[cfg(test)]
mod boot_connect_tests {
    use super::*;

    fn single(url: &str) -> RedisConnConfig {
        RedisConnConfig {
            mode: RedisMode::Single,
            url: Some(url.to_string()),
            ..Default::default()
        }
    }

    // The label goes into a boot WARN, so what it must never carry is the
    // password — which is exactly what the shipped way of supplying one
    // (`AISIX_RATELIMIT__REDIS__URL=redis://user:pass@host`) puts in the
    // URL this is derived from.
    #[test]
    fn strips_scheme_userinfo_and_path() {
        assert_eq!(
            endpoint_label(&single("redis://10.0.0.1:6379")),
            "10.0.0.1:6379"
        );
        assert_eq!(
            endpoint_label(&single("rediss://user:p%40ss@10.0.0.1:6379/2")),
            "10.0.0.1:6379"
        );
        // A password containing '@' must not leave its tail in front of
        // the host, which is what splitting on the FIRST '@' would do.
        assert_eq!(
            endpoint_label(&single("redis://user:p@ss@10.0.0.1:6379")),
            "10.0.0.1:6379"
        );
        assert_eq!(
            endpoint_label(&single("redis://10.0.0.1:6379/#insecure")),
            "10.0.0.1:6379"
        );
    }

    // A one-budget bound would cut sentinel and cluster discovery short
    // in exactly the deployment those modes exist for — the one with a
    // dead node at the front of the list.
    #[test]
    fn the_connect_budget_pays_for_every_endpoint_the_walk_may_touch() {
        let per_attempt = Duration::from_secs(5);
        let budget =
            |cfg: &RedisConnConfig| discovery_budget(configured_endpoints(cfg), per_attempt);

        assert_eq!(budget(&single("redis://10.0.0.1:6379")), per_attempt);

        let cluster = RedisConnConfig {
            mode: RedisMode::Cluster,
            nodes: vec!["redis://a:6379".into(), "redis://b:6379".into()],
            ..Default::default()
        };
        assert_eq!(budget(&cluster), per_attempt * 3);

        let sentinel = RedisConnConfig {
            mode: RedisMode::Sentinel,
            master_name: Some("mymaster".into()),
            sentinels: vec![
                "redis://s1:26379".into(),
                "redis://s2:26379".into(),
                "redis://s3:26379".into(),
            ],
            ..Default::default()
        };
        assert_eq!(budget(&sentinel), per_attempt * 4);
    }

    // The message is the only place an operator meets the aggregate, and
    // the product is not a number they can find in their config — so it
    // has to name the field it was derived from as well. Pinned because
    // an e2e asserts on this text to prove the CONFIGURED budget, not
    // the default, is what a boot spent.
    #[test]
    fn a_connect_timeout_names_both_the_aggregate_and_the_field() {
        let err = connect_timed_out_error(Duration::from_secs(20), Duration::from_secs(5));
        let msg = format!("{err}");
        assert!(msg.contains("no connection within 20s"), "{msg}");
        assert!(msg.contains("redis.timeout_secs = 5s"), "{msg}");
    }

    // The label goes into a log line, and the thing most likely to make
    // a redis URL unparseable is an unescaped character in the password.
    // So the rule is not "slice carefully" — it is "emit nothing that is
    // not shaped like a host", whatever the input.
    #[test]
    fn a_url_it_cannot_read_yields_no_text_from_the_url() {
        // RFC 3986 puts the authority at `user:secret` here — the `@` is
        // inside the path — so slicing alone prints the password.
        assert_eq!(
            endpoint_label(&single("redis://user:secret/extra@redis.internal:6379")),
            "<unparseable redis endpoint>"
        );
        assert_eq!(
            endpoint_label(&single("redis://user:secret?x@redis.internal:6379")),
            "<unparseable redis endpoint>"
        );
        assert_eq!(
            endpoint_label(&single("redis://user:secret#x@redis.internal:6379")),
            "<unparseable redis endpoint>"
        );
        assert_eq!(endpoint_label(&single("")), "<unparseable redis endpoint>");
        // A port that is not a number is not a port, so the whole thing
        // fails the shape check rather than being printed as a host.
        assert_eq!(
            endpoint_label(&single("redis://host:not-a-port")),
            "<unparseable redis endpoint>"
        );
    }

    #[test]
    fn an_ipv6_literal_survives_the_shape_check() {
        assert_eq!(
            endpoint_label(&single("redis://[2001:db8::1]:6379")),
            "[2001:db8::1]:6379"
        );
        assert_eq!(
            endpoint_label(&single("redis://user:pw@[::1]:6379")),
            "[::1]:6379"
        );
    }

    // An entry `validate` tolerates and `connect_with` then filters out
    // must not buy a budget the walk will never spend.
    #[test]
    fn a_blank_endpoint_buys_no_budget() {
        let per_attempt = Duration::from_secs(5);
        let cluster = RedisConnConfig {
            mode: RedisMode::Cluster,
            nodes: vec!["redis://a:6379".into(), "  ".into(), String::new()],
            ..Default::default()
        };
        assert_eq!(
            discovery_budget(configured_endpoints(&cluster), per_attempt),
            per_attempt * 2
        );
    }

    #[test]
    fn names_every_topology() {
        let cluster = RedisConnConfig {
            mode: RedisMode::Cluster,
            nodes: vec!["redis://a:6379".into(), "redis://admin:pw@b:6380".into()],
            ..Default::default()
        };
        assert_eq!(endpoint_label(&cluster), "a:6379,b:6380");

        let sentinel = RedisConnConfig {
            mode: RedisMode::Sentinel,
            master_name: Some("mymaster".into()),
            sentinels: vec!["redis://s1:26379".into()],
            ..Default::default()
        };
        assert_eq!(endpoint_label(&sentinel), "master mymaster via s1:26379");
    }
}
