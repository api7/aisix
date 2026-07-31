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

use std::sync::Arc;

use aisix_core::{RedisConnConfig, RedisMode};
use redis::aio::{ConnectionLike, ConnectionManager, MultiplexedConnection};
use redis::cluster::ClusterClient;
use redis::cluster_async::ClusterConnection;
use redis::sentinel::{SentinelClient, SentinelNodeConnectionInfo, SentinelServerType};
use redis::RedisResult;
use tokio::sync::Mutex;

/// A long-lived Redis client handle. Cheap to [`Clone`] (every variant is
/// `Arc`-backed). Build one with [`connect`].
// `Single` (the hot, common path) is the largest variant; boxing it to
// equalize variant size would add an allocation to the common case to
// shrink the rarer ones — not worth it for a handful of instances.
#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub enum RedisConn {
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
}

/// A live connection usable for one or more operations. Implements
/// [`ConnectionLike`] by delegating to the underlying connection.
#[allow(clippy::large_enum_variant)]
pub enum RedisConnHandle {
    Single(ConnectionManager),
    Cluster(ClusterConnection),
    Sentinel(MultiplexedConnection),
}

impl RedisConn {
    /// Obtain a live connection. For `single`/`cluster` this is an
    /// infallible cheap clone of the multiplexed connection. For
    /// `sentinel` it returns the cached master connection, resolving one
    /// through the sentinels on the first call or after a failover.
    pub async fn acquire(&self) -> RedisResult<RedisConnHandle> {
        match self {
            RedisConn::Single(c) => Ok(RedisConnHandle::Single(c.clone())),
            RedisConn::Cluster(c) => Ok(RedisConnHandle::Cluster(c.clone())),
            RedisConn::Sentinel(pool) => {
                if let Some(conn) = pool.cached.lock().await.clone() {
                    return Ok(RedisConnHandle::Sentinel(conn));
                }
                let mut client = pool.client.lock().await;
                let conn = client.get_async_connection().await?;
                *pool.cached.lock().await = Some(conn.clone());
                Ok(RedisConnHandle::Sentinel(conn))
            }
        }
    }

    /// Invalidate any cached connection after an operation error. Only
    /// meaningful for `sentinel`, where it forces the next [`acquire`] to
    /// re-resolve the master (the prior one may have failed over).
    ///
    /// [`acquire`]: RedisConn::acquire
    pub async fn note_error(&self) {
        if let RedisConn::Sentinel(pool) = self {
            *pool.cached.lock().await = None;
        }
    }
}

impl ConnectionLike for RedisConnHandle {
    fn req_packed_command<'a>(
        &'a mut self,
        cmd: &'a redis::Cmd,
    ) -> redis::RedisFuture<'a, redis::Value> {
        match self {
            RedisConnHandle::Single(c) => c.req_packed_command(cmd),
            RedisConnHandle::Cluster(c) => c.req_packed_command(cmd),
            RedisConnHandle::Sentinel(c) => c.req_packed_command(cmd),
        }
    }

    fn req_packed_commands<'a>(
        &'a mut self,
        cmd: &'a redis::Pipeline,
        offset: usize,
        count: usize,
    ) -> redis::RedisFuture<'a, Vec<redis::Value>> {
        match self {
            RedisConnHandle::Single(c) => c.req_packed_commands(cmd, offset, count),
            RedisConnHandle::Cluster(c) => c.req_packed_commands(cmd, offset, count),
            RedisConnHandle::Sentinel(c) => c.req_packed_commands(cmd, offset, count),
        }
    }

    fn get_db(&self) -> i64 {
        match self {
            RedisConnHandle::Single(c) => c.get_db(),
            RedisConnHandle::Cluster(c) => c.get_db(),
            RedisConnHandle::Sentinel(c) => c.get_db(),
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
    match cfg.mode {
        RedisMode::Single => {
            let url = insecure_url(cfg.url.as_deref().unwrap_or_default(), cfg);
            let client = match &tls {
                Some(certs) => redis::Client::build_with_tls(url.as_str(), certs.clone())?,
                None => redis::Client::open(url.as_str())?,
            };
            let conn = ConnectionManager::new(client).await?;
            tracing::info!(target: "aisix::redis", mode = "single", "connected");
            Ok(RedisConn::Single(conn))
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
            let mut builder = ClusterClient::builder(nodes);
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
                "connected"
            );
            Ok(RedisConn::Cluster(conn))
        }
        RedisMode::Sentinel => {
            let sentinels: Vec<String> = cfg
                .sentinels
                .iter()
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .map(|s| insecure_url(s, cfg))
                .collect();
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
            let conn = client.get_async_connection().await?;
            tracing::info!(
                target: "aisix::redis",
                mode = "sentinel",
                master = %cfg.master_name.as_deref().unwrap_or_default(),
                "connected"
            );
            Ok(RedisConn::Sentinel(SentinelPool {
                client: Arc::new(Mutex::new(client)),
                cached: Arc::new(Mutex::new(Some(conn))),
            }))
        }
    }
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
