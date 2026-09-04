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
    Client, ConnectOptions, Error as EtcdError, EventType, GetOptions, KvClient, WatchClient,
    WatchOptions,
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

use crate::client::{kv_client, watch_client};
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
    /// The sub-clients are `Clone`-cheap (internally Arc'd), but we still
    /// serialise access through a Mutex because their RPC methods take
    /// `&mut self`. They are held rather than derived from a `Client` per
    /// call so the raised decode limit cannot be lost by a later call site
    /// reaching for `Client::get` / `Client::watch`, which keep tonic's
    /// 4 MiB default.
    kv: Mutex<KvClient>,
    watch: Mutex<WatchClient>,
    prefix: String,
    /// Bound applied to each unary call this provider makes — currently
    /// the range read in [`Self::load_all`]. `None` leaves it unbounded.
    ///
    /// Applied per call with [`tokio::time::timeout`] rather than through
    /// `ConnectOptions::with_timeout`, for two reasons. That option is
    /// channel-wide, so it would also bound opening the watch. And it
    /// bounds only the response *future*: tonic's `GrpcTimeout` races the
    /// deadline against the arrival of response headers and then lets the
    /// body stream run untimed, so an etcd that answers a range request
    /// and stalls part-way through a large body would still hang here
    /// forever — the very case a bound on the configuration read is for.
    /// Wrapping the call bounds it whole, body included.
    request_timeout: Option<Duration>,
}

impl std::fmt::Debug for EtcdConfigProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EtcdConfigProvider")
            .field("prefix", &self.prefix)
            .finish_non_exhaustive()
    }
}

impl EtcdConfigProvider {
    /// Connect with the spec §2 default retry policy.
    pub async fn connect(
        endpoints: &[String],
        prefix: impl Into<String>,
        options: Option<ConnectOptions>,
        request_timeout: Option<Duration>,
    ) -> Result<Self, ProviderError> {
        Self::connect_with_policy(
            endpoints,
            prefix,
            options,
            request_timeout,
            ConnectPolicy::default(),
        )
        .await
    }

    /// Connect with a caller-chosen retry policy. Returns the last-seen
    /// error on failure to surface useful context in the bootstrap logs.
    pub async fn connect_with_policy(
        endpoints: &[String],
        prefix: impl Into<String>,
        options: Option<ConnectOptions>,
        request_timeout: Option<Duration>,
        policy: ConnectPolicy,
    ) -> Result<Self, ProviderError> {
        let prefix = prefix.into();
        let mut last_err: Option<EtcdError> = None;
        for attempt in 1..=policy.attempts {
            match Client::connect(endpoints, options.clone()).await {
                Ok(client) => {
                    tracing::info!(attempt, prefix = %prefix, "etcd connected");
                    return Ok(Self {
                        kv: Mutex::new(kv_client(&client)),
                        watch: Mutex::new(watch_client(&client)),
                        prefix,
                        request_timeout,
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

/// Apply `timeout`, when set, to one in-flight etcd call.
///
/// Expiry is reported as [`ProviderError::Range`] so it lands on the
/// supervisor's existing reconnect-with-backoff path: the aborted read is
/// retried rather than ending the watch task.
async fn bound<T>(
    timeout: Option<Duration>,
    what: &str,
    fut: impl std::future::Future<Output = T>,
) -> Result<T, ProviderError> {
    match timeout {
        None => Ok(fut.await),
        Some(d) => tokio::time::timeout(d, fut).await.map_err(|_| {
            ProviderError::Range(format!(
                "{what} exceeded etcd.request_timeout_ms ({} ms)",
                d.as_millis()
            ))
        }),
    }
}

#[async_trait]
impl ConfigProvider for EtcdConfigProvider {
    async fn load_all(&self) -> Result<(Vec<RawEntry>, i64), ProviderError> {
        let mut kv = self.kv.lock().await;
        let read = kv.get(
            self.prefix.as_bytes(),
            Some(GetOptions::new().with_prefix()),
        );
        let resp = bound(self.request_timeout, "range read", read)
            .await?
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
        let mut watch = self.watch.lock().await;
        let opts = WatchOptions::new()
            .with_prefix()
            .with_start_revision(start_revision);
        let (watcher, stream) = watch
            .watch(self.prefix.as_bytes(), Some(opts))
            .await
            .map_err(|e| ProviderError::Watch(format_error_chain(&e)))?;

        Ok(Box::new(EtcdWatchStream {
            inner: stream,
            _watcher: watcher,
            buf: VecDeque::new(),
        }))
    }
}

/// Adapter from `etcd-client`'s WatchStream to our typed [`WatchEvent`].
///
/// Each `WatchResponse` carries a batch of events; we flatten them
/// into individual `WatchEvent` items. A `VecDeque` buffer drains
/// multi-event responses across successive `poll_next` calls so that
/// no events are silently dropped.
pub struct EtcdWatchStream {
    inner: etcd_client::WatchStream,
    // Must outlive `inner` — dropping the Watcher closes the client→server
    // half of the gRPC stream, causing the server to tear down the watch.
    _watcher: etcd_client::Watcher,
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
        let err = EtcdConfigProvider::connect_with_policy(&endpoints, "/aisix", None, None, policy)
            .await
            .unwrap_err();
        assert!(matches!(err, ProviderError::Connect(_)));
    }

    #[tokio::test]
    async fn bound_passes_through_when_no_timeout_is_set() {
        let out = bound(None, "range read", async { 7u8 }).await.unwrap();
        assert_eq!(out, 7);
    }

    #[tokio::test]
    async fn expiry_maps_to_range_so_the_supervisor_backs_off_and_retries() {
        // The variant matters as much as the expiry: `Range` is what
        // `Supervisor::run` treats as a retryable source outage. Any other
        // variant either ends the watch task or (for `Compacted`) skips
        // the backoff entirely.
        let err = bound(
            Some(Duration::from_millis(1)),
            "range read",
            std::future::pending::<()>(),
        )
        .await
        .unwrap_err();
        let ProviderError::Range(msg) = err else {
            panic!("expiry must surface as ProviderError::Range, got {err:?}");
        };
        assert!(
            msg.contains("range read") && msg.contains("etcd.request_timeout_ms"),
            "the message must name the call and the key that bounded it: {msg}"
        );
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
}
