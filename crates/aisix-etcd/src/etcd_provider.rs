//! Real [`ConfigProvider`] backed by `etcd-client`.
//!
//! Connection sequence (spec §2):
//! - Fixed-interval retry on initial connect: 5s × up to 5 attempts
//! - On success, `get` with prefix to bootstrap
//! - `watch` with `start_revision = range_revision + 1` to avoid a gap
//! - Compaction errors map to [`ProviderError::Compacted`] so the
//!   supervisor can trigger a full resync

use async_trait::async_trait;
use etcd_client::{
    Client, ConnectOptions, Error as EtcdError, EventType, GetOptions, WatchOptions,
};
use futures::{Stream, StreamExt};
use std::collections::VecDeque;
use std::error::Error as StdError;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;
use tokio::sync::Mutex;

/// Flatten an error and its source chain into a single readable line.
/// Without this, tonic surfaces opaque strings like "dns error" while
/// the real cause (`getaddrinfo: Name or service not known`, TLS
/// handshake reason, …) hides in `.source()`. The supervisor logs
/// the returned string, so CI triage gets the full picture.
fn format_error_chain(err: &(dyn StdError + 'static)) -> String {
    let mut out = err.to_string();
    let mut cur = err.source();
    while let Some(next) = cur {
        let s = next.to_string();
        if !s.is_empty() && !out.ends_with(&s) {
            out.push_str(": ");
            out.push_str(&s);
        }
        cur = next.source();
    }
    out
}

use crate::provider::{ConfigProvider, ProviderError, RawEntry, WatchEvent};

/// Fixed-interval retry: 5s × 5 attempts (spec §2).
pub const CONNECT_RETRY_INTERVAL: Duration = Duration::from_secs(5);
pub const CONNECT_MAX_ATTEMPTS: u32 = 5;

/// Retry policy used on the initial connect. Exposed for tests so they
/// can shrink the interval; production uses [`ConnectPolicy::default`].
#[derive(Debug, Clone, Copy)]
pub struct ConnectPolicy {
    pub interval: Duration,
    pub attempts: u32,
}

impl Default for ConnectPolicy {
    fn default() -> Self {
        Self {
            interval: CONNECT_RETRY_INTERVAL,
            attempts: CONNECT_MAX_ATTEMPTS,
        }
    }
}

pub struct EtcdConfigProvider {
    /// The etcd client itself is `Clone`-cheap (internally Arc'd), but we
    /// still serialise access for watches through a Mutex because the
    /// underlying channel is not Sync at construction time.
    client: Mutex<Client>,
    prefix: String,
}

impl std::fmt::Debug for EtcdConfigProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EtcdConfigProvider")
            .field("prefix", &self.prefix)
            .finish_non_exhaustive()
    }
}

/// etcd's default `--auth-token-ttl` is 300s; 240s leaves a 60s margin.
/// Override via `EtcdConfig.auth_token_refresh_secs` for a shorter TTL.
const DEFAULT_TOKEN_REFRESH_SECS: u64 = 240;

/// `config_secs` is `EtcdConfig.auth_token_refresh_secs`; `None` or `0`
/// (an operator-supplied 0, or the pre-migration config default) falls
/// back to the hardcoded default.
fn resolve_refresh_secs(config_secs: Option<u64>) -> u64 {
    config_secs
        .filter(|&secs| secs > 0)
        .unwrap_or(DEFAULT_TOKEN_REFRESH_SECS)
}

/// Spawn a background task that refreshes the etcd auth token at the
/// given interval. Safe to call unconditionally: `Client::refresh_token`
/// (v0.18+) no-ops when no credentials were configured.
///
/// `config_secs` (typically `EtcdConfig.auth_token_refresh_secs`) sets the
/// interval; `None` or `0` falls back to the hardcoded default (240s).
/// `Client` is `Clone`-cheap (internally `Arc`'d) and shares its
/// auth-token cell across clones, so this works on any clone of the
/// caller's client.
///
/// No-ops outside a Tokio runtime (e.g. a sync unit test building a
/// `Client`/store via `Runtime::block_on` and returning it) — `tokio::spawn`
/// requires an active runtime, and such contexts never carry credentials.
pub fn start_token_refresh_task(client: Client, config_secs: Option<u64>) {
    if tokio::runtime::Handle::try_current().is_err() {
        return;
    }

    if config_secs == Some(0) {
        tracing::warn!(
            interval_secs = DEFAULT_TOKEN_REFRESH_SECS,
            "etcd.auth_token_refresh_secs = 0 is not a valid interval — falling back to the default",
        );
    }

    let refresh_interval = Duration::from_secs(resolve_refresh_secs(config_secs));

    tracing::info!(
        interval_secs = refresh_interval.as_secs(),
        "etcd auth token refresh loop started",
    );
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(refresh_interval);
        // First tick fires immediately — skip it; `Client::connect`
        // already authenticated once.
        interval.tick().await;
        loop {
            interval.tick().await;
            match client.refresh_token().await {
                Ok(()) => {
                    tracing::debug!("etcd auth token refreshed");
                }
                Err(e) => {
                    tracing::warn!(
                        error = %format_error_chain(&e),
                        "etcd auth token refresh failed — watch may see \
                         UNAUTHENTICATED on next expiry",
                    );
                }
            }
        }
    });
}

impl EtcdConfigProvider {
    /// Connect with the spec §2 default retry policy. `refresh_secs`
    /// overrides the auth-token refresh interval (typically
    /// `EtcdConfig.auth_token_refresh_secs`); `None` uses the default —
    /// see `start_token_refresh_task`.
    pub async fn connect(
        endpoints: &[String],
        prefix: impl Into<String>,
        options: Option<ConnectOptions>,
        refresh_secs: Option<u64>,
    ) -> Result<Self, ProviderError> {
        Self::connect_with_policy(
            endpoints,
            prefix,
            options,
            ConnectPolicy::default(),
            refresh_secs,
        )
        .await
    }

    /// Connect with a caller-chosen retry policy and auth-token refresh
    /// interval override (`refresh_secs`, typically
    /// `EtcdConfig.auth_token_refresh_secs` — `None` uses the default).
    /// Returns the last-seen error on failure to surface useful context
    /// in the bootstrap logs.
    pub async fn connect_with_policy(
        endpoints: &[String],
        prefix: impl Into<String>,
        options: Option<ConnectOptions>,
        policy: ConnectPolicy,
        refresh_secs: Option<u64>,
    ) -> Result<Self, ProviderError> {
        let prefix = prefix.into();
        let mut last_err: Option<EtcdError> = None;
        for attempt in 1..=policy.attempts {
            match Client::connect(endpoints, options.clone()).await {
                Ok(client) => {
                    tracing::info!(attempt, prefix = %prefix, "etcd connected");
                    start_token_refresh_task(client.clone(), refresh_secs);
                    return Ok(Self {
                        client: Mutex::new(client),
                        prefix,
                    });
                }
                Err(err) => {
                    tracing::warn!(
                        attempt,
                        max = policy.attempts,
                        error = %format_error_chain(&err),
                        "etcd connect failed — retrying",
                    );
                    last_err = Some(err);
                    if attempt < policy.attempts {
                        tokio::time::sleep(policy.interval).await;
                    }
                }
            }
        }
        Err(ProviderError::Connect(
            last_err
                .as_ref()
                .map(|e| format_error_chain(e))
                .unwrap_or_else(|| "exhausted retries".to_string()),
        ))
    }

    pub fn prefix(&self) -> &str {
        &self.prefix
    }
}

#[async_trait]
impl ConfigProvider for EtcdConfigProvider {
    async fn load_all(&self) -> Result<(Vec<RawEntry>, i64), ProviderError> {
        let mut client = self.client.lock().await;
        let resp = client
            .get(
                self.prefix.as_bytes(),
                Some(GetOptions::new().with_prefix()),
            )
            .await
            .map_err(|e| ProviderError::Range(format_error_chain(&e)))?;

        let revision = resp.header().map(|h| h.revision()).unwrap_or(0);

        let entries = resp
            .kvs()
            .iter()
            .map(|kv| RawEntry {
                key: String::from_utf8_lossy(kv.key()).into_owned(),
                value: kv.value().to_vec(),
                revision: kv.mod_revision(),
            })
            .collect();

        Ok((entries, revision))
    }

    async fn watch(
        &self,
        start_revision: i64,
    ) -> Result<
        Box<dyn Stream<Item = Result<WatchEvent, ProviderError>> + Send + Unpin>,
        ProviderError,
    > {
        let mut client = self.client.lock().await;
        let opts = WatchOptions::new()
            .with_prefix()
            .with_start_revision(start_revision);
        let stream = client
            .watch(self.prefix.as_bytes(), Some(opts))
            .await
            .map_err(|e| ProviderError::Watch(format_error_chain(&e)))?;

        Ok(Box::new(EtcdWatchStream {
            inner: stream,
            buf: VecDeque::new(),
        }))
    }
}

/// Adapter from `etcd-client`'s WatchStream to our typed [`WatchEvent`].
/// A `VecDeque` buffer drains multi-event responses across successive
/// `poll_next` calls so no events are silently dropped.
///
/// Pre-0.18 `etcd-client` required a separate `Watcher` handle alongside
/// the stream (dropping it killed the watch — issue #237). Since 0.18
/// `WatchStream` owns both halves, so `inner` alone keeps it alive.
pub struct EtcdWatchStream {
    inner: etcd_client::WatchStream,
    buf: VecDeque<WatchEvent>,
}

fn convert_event(ev: &etcd_client::Event) -> Option<WatchEvent> {
    match ev.event_type() {
        EventType::Put => ev.kv().map(|kv| {
            WatchEvent::Put(RawEntry {
                key: String::from_utf8_lossy(kv.key()).into_owned(),
                value: kv.value().to_vec(),
                revision: kv.mod_revision(),
            })
        }),
        EventType::Delete => ev.kv().map(|kv| WatchEvent::Delete {
            key: String::from_utf8_lossy(kv.key()).into_owned(),
            revision: kv.mod_revision(),
        }),
    }
}

impl Stream for EtcdWatchStream {
    type Item = Result<WatchEvent, ProviderError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Drain buffered events from a previous multi-event response first.
        if let Some(item) = self.buf.pop_front() {
            return Poll::Ready(Some(Ok(item)));
        }

        match self.inner.poll_next_unpin(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Ready(Some(Err(err))) => {
                let shallow = err.to_string();
                if shallow.contains("required revision has been compacted")
                    || shallow.contains("mvcc: required revision")
                {
                    Poll::Ready(Some(Err(ProviderError::Compacted)))
                } else {
                    Poll::Ready(Some(Err(ProviderError::Watch(format_error_chain(&err)))))
                }
            }
            Poll::Ready(Some(Ok(resp))) => {
                if resp.compact_revision() > 0 {
                    return Poll::Ready(Some(Err(ProviderError::Compacted)));
                }

                for ev in resp.events() {
                    if let Some(item) = convert_event(ev) {
                        self.buf.push_back(item);
                    }
                }

                if let Some(item) = self.buf.pop_front() {
                    return Poll::Ready(Some(Ok(item)));
                }

                // Empty response (e.g. header-only): tell the runtime
                // to poll us again rather than stalling.
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connect_retry_constants_match_spec() {
        assert_eq!(CONNECT_RETRY_INTERVAL, Duration::from_secs(5));
        assert_eq!(CONNECT_MAX_ATTEMPTS, 5);
    }

    #[test]
    fn default_policy_matches_spec() {
        let p = ConnectPolicy::default();
        assert_eq!(p.interval, CONNECT_RETRY_INTERVAL);
        assert_eq!(p.attempts, CONNECT_MAX_ATTEMPTS);
    }

    #[tokio::test]
    async fn connect_with_malformed_endpoint_returns_connect_error() {
        // Empty endpoint list is treated as a parse failure by etcd-client,
        // which lets us exercise the retry loop's error branch without
        // waiting on a real TCP timeout. A compressed policy keeps the
        // test sub-millisecond.
        let policy = ConnectPolicy {
            interval: Duration::from_millis(1),
            attempts: 1,
        };
        let endpoints: Vec<String> = vec![];
        let err = EtcdConfigProvider::connect_with_policy(&endpoints, "/aisix", None, policy, None)
            .await
            .unwrap_err();
        assert!(matches!(err, ProviderError::Connect(_)));
    }

    #[test]
    fn format_error_chain_joins_sources_without_duplicating() {
        #[derive(Debug)]
        struct Inner;
        impl std::fmt::Display for Inner {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("Name or service not known")
            }
        }
        impl StdError for Inner {}

        #[derive(Debug)]
        struct Outer {
            inner: Inner,
        }
        impl std::fmt::Display for Outer {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("dns error")
            }
        }
        impl StdError for Outer {
            fn source(&self) -> Option<&(dyn StdError + 'static)> {
                Some(&self.inner)
            }
        }

        let joined = format_error_chain(&Outer { inner: Inner });
        assert_eq!(joined, "dns error: Name or service not known");
    }

    #[test]
    fn format_error_chain_handles_empty_source() {
        let err = std::io::Error::other("bare");
        assert_eq!(format_error_chain(&err), "bare");
    }

    #[test]
    fn resolve_refresh_secs_defaults_when_unset() {
        assert_eq!(resolve_refresh_secs(None), DEFAULT_TOKEN_REFRESH_SECS);
    }

    #[test]
    fn resolve_refresh_secs_honors_config_value() {
        assert_eq!(resolve_refresh_secs(Some(90)), 90);
    }

    #[test]
    fn resolve_refresh_secs_falls_back_on_zero() {
        assert_eq!(resolve_refresh_secs(Some(0)), DEFAULT_TOKEN_REFRESH_SECS);
    }
}
