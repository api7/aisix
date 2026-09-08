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
//! - **A cool-off breaker.** Paying the timeout on *every* request during
//!   an outage is still a several-second latency floor for as long as the
//!   outage lasts. After a connectivity failure the connection is held
//!   open for [`BREAKER_WINDOW`]; commands issued inside that window
//!   return an error immediately, with no round trip, so each consumer's
//!   existing fail-open branch runs at once. The first command after the
//!   window probes Redis normally and closes the breaker on success.
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
use redis::{AsyncConnectionConfig, RedisResult};
use tokio::sync::Mutex;

/// How long commands short-circuit after a connectivity failure. Fixed,
/// not configurable: it trades at most this much staleness (a Redis that
/// recovered mid-window is not noticed until the window ends) for a hard
/// ceiling on how often a request pays the full timeout during an outage.
pub const BREAKER_WINDOW: Duration = Duration::from_secs(5);

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

/// The per-connection failure policy: one command budget and one breaker,
/// shared by every handle [`RedisConn::acquire`] hands out.
struct Guard {
    timeout: Duration,
    breaker: Breaker,
}

impl Guard {
    /// Run one Redis operation under the breaker and the command budget.
    ///
    /// A success closes the breaker; a *connectivity* failure or a
    /// timeout opens it. A failure the server itself reported (a script
    /// error, `WRONGTYPE`, an ACL refusal) is returned untouched — Redis
    /// answered, so short-circuiting the next five seconds of traffic
    /// would be wrong.
    async fn run<T>(
        &self,
        fut: impl std::future::Future<Output = RedisResult<T>>,
    ) -> RedisResult<T> {
        self.run_with(self.timeout, fut).await
    }

    /// [`Guard::run`] with a budget other than the per-command one. Only
    /// sentinel master discovery uses it — see [`SentinelPool`].
    async fn run_with<T>(
        &self,
        budget: Duration,
        fut: impl std::future::Future<Output = RedisResult<T>>,
    ) -> RedisResult<T> {
        // `admit` both decides and, when it lets a probe through, re-arms
        // the window behind it. The generation it returns is read before
        // the await, because a command already in flight when the outage
        // began can land its success after a *concurrent* command opened
        // the breaker, and closing on that stale evidence would send the
        // requests behind it back into the full budget.
        let Some(seen) = self.breaker.admit() else {
            return Err(breaker_open_error());
        };
        match tokio::time::timeout(budget, fut).await {
            Ok(Ok(v)) => {
                self.breaker.close_unless_reopened(seen);
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
                    self.breaker.open();
                }
                Err(e)
            }
            Err(_) => {
                self.breaker.open();
                Err(timed_out_error(budget))
            }
        }
    }
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
}

impl Breaker {
    fn new(window: Duration) -> Self {
        Self {
            window,
            state: std::sync::Mutex::new(BreakerState::default()),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, BreakerState> {
        // Nothing under this lock can panic, so it cannot be poisoned;
        // taking the value through a poisoned guard would be equally
        // correct if it ever were.
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn is_open(&self) -> bool {
        self.lock()
            .open_until
            .is_some_and(|until| Instant::now() < until)
    }

    /// Decide whether this command reaches Redis, and hand back the
    /// generation it is allowed to close.
    ///
    /// `None` short-circuits. Once the window expires, exactly ONE caller
    /// is admitted as the probe and the window is re-armed behind it:
    /// without that, every command arriving while the probe is in flight
    /// is admitted too and pays the full budget, which during a sustained
    /// outage is most of them. A probe whose caller is dropped rather
    /// than finishing leaves nothing stuck — the re-armed window simply
    /// expires and the next caller probes.
    fn admit(&self) -> Option<u64> {
        let mut st = self.lock();
        match st.open_until {
            None => Some(st.generation),
            Some(until) if Instant::now() < until => None,
            Some(_) => {
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
                        let conn = self
                            .guard
                            .run_with(
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
        Box::pin(async move { guard.run(fut).await })
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
        Box::pin(async move { guard.run(fut).await })
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
    let tls = load_tls(cfg)?;
    let timeout = Duration::from_secs(cfg.timeout_secs.max(1));
    let guard = Arc::new(Guard {
        timeout,
        breaker: Breaker::new(BREAKER_WINDOW),
    });
    let inner = match cfg.mode {
        RedisMode::Single => {
            let url = insecure_url(cfg.url.as_deref().unwrap_or_default(), cfg);
            let client = match &tls {
                Some(certs) => redis::Client::build_with_tls(url.as_str(), certs.clone())?,
                None => redis::Client::open(url.as_str())?,
            };
            // Only the two timeouts are set; the retry policy stays the
            // library default, which is what the boot retry loop is tuned
            // around.
            let manager_cfg = ConnectionManagerConfig::new()
                .set_connection_timeout(timeout)
                .set_response_timeout(timeout);
            let conn = ConnectionManager::new_with_config(client, manager_cfg).await?;
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
            let conn = client.get_async_connection().await?;
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
            let sentinels_len = sentinels.len().max(1) as u32;
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
            let discovery_timeout = timeout * (sentinels_len + 1);
            let conn = tokio::time::timeout(
                discovery_timeout,
                client.get_async_connection_with_config(&conn_config(timeout)),
            )
            .await
            .map_err(|_| timed_out_error(discovery_timeout))??;
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

    fn guard(timeout_ms: u64, window_ms: u64) -> Guard {
        Guard {
            timeout: Duration::from_millis(timeout_ms),
            breaker: Breaker::new(Duration::from_millis(window_ms)),
        }
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
        let err = g
            .run(pending::<RedisResult<()>>())
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
        let _ = g.run(pending::<RedisResult<()>>()).await;

        let started = Instant::now();
        let err = g
            .run(pending::<RedisResult<()>>())
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
        let _ = g.run(pending::<RedisResult<()>>()).await;
        assert!(g.breaker.is_open());

        tokio::time::sleep(Duration::from_millis(120)).await;
        assert!(!g.breaker.is_open(), "the window must expire on its own");

        g.run(async { Ok::<_, redis::RedisError>(7) })
            .await
            .expect("the probe reaches Redis again");
        assert!(!g.breaker.is_open(), "a success closes the breaker");
    }

    /// A cool-off that only gates the window itself still lets every
    /// command that arrives while the probe is in flight pay the full
    /// budget — during a sustained outage that is most of them, and no
    /// serial test can see it.
    #[tokio::test]
    async fn only_one_command_probes_when_the_window_expires() {
        let g = Arc::new(guard(2_000, 60));
        g.run(async { Err::<(), _>(dropped_connection()) })
            .await
            .expect_err("the failure opens the breaker");
        tokio::time::sleep(Duration::from_millis(120)).await;
        assert!(!g.breaker.is_open(), "the window has expired");

        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let probe = tokio::spawn({
            let g = Arc::clone(&g);
            async move {
                g.run(async move {
                    let _ = rx.await;
                    Ok::<_, redis::RedisError>(1)
                })
                .await
            }
        });
        tokio::task::yield_now().await;

        let started = Instant::now();
        let err = g
            .run(pending::<RedisResult<()>>())
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
        let g = Arc::new(guard(2_000, 5_000));
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();

        let inflight = tokio::spawn({
            let g = Arc::clone(&g);
            async move {
                g.run(async move {
                    let _ = rx.await;
                    Ok::<_, redis::RedisError>(1)
                })
                .await
            }
        });
        // Let it enter `run` and read the generation before anything fails.
        tokio::task::yield_now().await;

        g.run(async { Err::<(), _>(dropped_connection()) })
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

    /// A reply from a live Redis — a script error, `WRONGTYPE`, an ACL
    /// refusal — is not an outage. Tripping on it would short-circuit
    /// five seconds of healthy traffic every time one command is wrong.
    #[tokio::test]
    async fn an_error_redis_itself_reported_does_not_open_the_breaker() {
        let g = guard(80, 5_000);
        let err = g
            .run(async { Err::<(), _>(server_side_error()) })
            .await
            .expect_err("the server error is passed through");
        assert!(!err.is_io_error(), "{err:?}");
        assert!(
            !g.breaker.is_open(),
            "a server reply is not a connectivity failure"
        );
    }
}
