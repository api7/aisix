//! Real [`ConfigProvider`] backed by `etcd-client`.
//!
//! Connection sequence (spec §2):
//! - The initial connect splits its failures by cause: an etcd that never
//!   answered leaves the connection pending for the supervisor's retry
//!   loop, while credentials etcd refused are returned to the caller. The
//!   fixed-interval 5s × 5 ladder is now only the `export` CLI's
//!   ([`EtcdConfigProvider::connect_with_policy`])
//! - On success, `get` with prefix to bootstrap
//! - `watch` with `start_revision = range_revision + 1` to avoid a gap
//! - Compaction errors map to [`ProviderError::Compacted`] so the
//!   supervisor can trigger a full resync

use async_trait::async_trait;
use etcd_client::{ConnectOptions, EventType, GetOptions, WatchOptions};
use futures::{Stream, StreamExt};
use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use crate::client::{
    classify, format_etcd_error, kv_client, watch_client, CallError, ConnectError, LazyEtcdClient,
};
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
    /// The connection, which may not have been established yet: an etcd
    /// that could not be reached at boot is dialled again by whichever
    /// call the supervisor's retry loop makes next. Sub-clients are taken
    /// from it per call — they are `Clone`-cheap (internally Arc'd) and
    /// carry the raised decode limit, which `Client::get` /
    /// `Client::watch` would not.
    client: Arc<LazyEtcdClient>,
    prefix: String,
    /// Bound applied to each request/response call this provider makes —
    /// the range read in [`Self::load_all`] and the watch-create
    /// handshake in [`Self::watch`]. `None` leaves them unbounded, and
    /// the established watch stream is never bounded by it.
    ///
    /// Applied per call with [`tokio::time::timeout`] rather than through
    /// `ConnectOptions::with_timeout`, for two reasons. That option is
    /// channel-wide, so it would also bound consuming the watch. And it
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
    /// Connect for a gateway boot, splitting the failure by cause.
    ///
    /// A dial that etcd never answered is not an error here: the provider
    /// comes back with its connection still pending and the watch
    /// supervisor's retry loop dials again, which is what an
    /// unauthenticated deployment has always done (`Client::connect` does
    /// no I/O without credentials, so every failure lands on the first
    /// read). The gateway boots, holds the proxy listener closed and binds
    /// as soon as a configuration arrives.
    ///
    /// Credentials etcd has refused are fatal and returned as
    /// [`ProviderError::Rejected`] on the first answer — no retry ladder,
    /// because no amount of waiting turns a wrong password into a right
    /// one, and a boot that hangs on one is worse than a boot that stops
    /// and says so.
    pub async fn connect(
        endpoints: &[String],
        prefix: impl Into<String>,
        options: Option<ConnectOptions>,
        request_timeout: Option<Duration>,
        dial_timeout: Option<Duration>,
    ) -> Result<Self, ProviderError> {
        let prefix = prefix.into();
        let client = Arc::new(LazyEtcdClient::new(
            endpoints.to_vec(),
            options,
            dial_timeout,
        ));
        match client.connect_now().await {
            Ok(()) => tracing::info!(prefix = %prefix, "etcd connected"),
            // A token etcd will not accept is grouped with an etcd that
            // did not answer, not with a refused credential: nothing
            // about the configured user is in question, and the next
            // call dials again and authenticates again.
            Err(err @ (ConnectError::Unreachable(_) | ConnectError::Unauthenticated(_))) => {
                tracing::warn!(
                    error = %err.detail(),
                    prefix = %prefix,
                    "etcd is not reachable yet — starting anyway and retrying in the background; \
                     the proxy listener stays closed until a configuration is applied",
                )
            }
            Err(err @ ConnectError::Rejected(_)) => {
                return Err(ProviderError::Rejected(err.detail().to_string()))
            }
        }
        Ok(Self {
            client,
            prefix,
            request_timeout,
        })
    }

    /// Connect eagerly, retrying an unreachable etcd on `policy` and
    /// failing once it is exhausted. For callers that cannot wait for a
    /// source to come back — the `export` CLI — rather than the gateway,
    /// which uses [`Self::connect`] and defers to the supervisor.
    ///
    /// Refused credentials end it on the first answer, as they do there.
    pub async fn connect_with_policy(
        endpoints: &[String],
        prefix: impl Into<String>,
        options: Option<ConnectOptions>,
        request_timeout: Option<Duration>,
        dial_timeout: Option<Duration>,
        policy: ConnectPolicy,
    ) -> Result<Self, ProviderError> {
        let prefix = prefix.into();
        let client = Arc::new(LazyEtcdClient::new(
            endpoints.to_vec(),
            options,
            dial_timeout,
        ));
        let mut last_err: Option<String> = None;
        for attempt in 1..=policy.attempts {
            match client.connect_now().await {
                Ok(()) => {
                    tracing::info!(attempt, prefix = %prefix, "etcd connected");
                    return Ok(Self {
                        client,
                        prefix,
                        request_timeout,
                    });
                }
                Err(err @ ConnectError::Rejected(_)) => {
                    return Err(ProviderError::Rejected(err.detail().to_string()))
                }
                Err(err @ (ConnectError::Unreachable(_) | ConnectError::Unauthenticated(_))) => {
                    let detail = err.detail().to_string();
                    tracing::warn!(
                        attempt,
                        max = policy.attempts,
                        error = %detail,
                        "etcd connect failed — retrying",
                    );
                    last_err = Some(detail);
                    if attempt < policy.attempts {
                        tokio::time::sleep(policy.interval).await;
                    }
                }
            }
        }
        Err(ProviderError::Connect(
            last_err.unwrap_or_else(|| "exhausted retries".to_string()),
        ))
    }

    pub fn prefix(&self) -> &str {
        &self.prefix
    }
}

impl From<ConnectError> for ProviderError {
    /// Every class reaches the supervisor, which backs off and retries
    /// whichever it gets — the split is what the boot path acts on, and
    /// what the operator reads in the warn line.
    fn from(err: ConnectError) -> Self {
        match &err {
            ConnectError::Unreachable(detail) => ProviderError::Connect(detail.clone()),
            ConnectError::Rejected(detail) => ProviderError::Rejected(detail.clone()),
            // Only the DIAL reaches here with this — a call's has been
            // re-authenticated and retried first (see `provider_error`).
            // A dial refused with `Unauthenticated` has nothing to
            // re-authenticate: it just tried. Retryable rather than
            // fatal, because a cluster whose authentication was being
            // re-enabled as the gateway dialled answers exactly this and
            // then starts working.
            ConnectError::Unauthenticated(detail) => ProviderError::Connect(detail.clone()),
        }
    }
}

/// Report one failed etcd call.
///
/// `into_err` names the call — the range read, the watch create — so the
/// supervisor's warn line attributes the failure to the right step. It is
/// attribution, not control flow: `Supervisor::run` routes every
/// `ProviderError` out of a cycle to the same reconnect-with-backoff
/// path, so the aborted call is retried whichever variant carries it.
///
/// A refusal is reported as such wherever etcd hands it over, not only on
/// the dial: a user who authenticates but lacks the permission is refused
/// by the range read. That never heals by retrying, and reporting it as
/// ordinary transport trouble is what buries it in a warn line that
/// repeats forever.
fn provider_error(
    err: CallError,
    what: &str,
    into_err: fn(String) -> ProviderError,
) -> ProviderError {
    match err {
        CallError::Connect(err) => err.into(),
        CallError::ConnectTimeout(d) => ProviderError::Connect(format!(
            "etcd connect exceeded etcd.request_timeout_ms ({} ms)",
            d.as_millis()
        )),
        CallError::CallTimeout(d) => into_err(format!(
            "{what} exceeded etcd.request_timeout_ms ({} ms)",
            d.as_millis()
        )),
        CallError::Call(err) => match classify(&err) {
            // An `Unauthenticated` reaching here has already been
            // retried on a freshly authenticated connection and was
            // refused again, so it belongs with the refusals: waiting
            // does not fix a token etcd will not accept however new it
            // is, and reporting it as ordinary transport trouble is what
            // buries it in a warn line that repeats forever.
            ConnectError::Rejected(detail) | ConnectError::Unauthenticated(detail) => {
                ProviderError::Rejected(detail)
            }
            ConnectError::Unreachable(_) => into_err(format_etcd_error(&err)),
        },
    }
}

#[async_trait]
impl ConfigProvider for EtcdConfigProvider {
    async fn load_all(&self) -> Result<(Vec<RawEntry>, i64), ProviderError> {
        // Through `LazyEtcdClient::call`, so the read is bounded, and so
        // a token etcd has forgotten — it restarted, or the token's TTL
        // elapsed — is re-authenticated and retried here instead of
        // failing every cycle until the gateway is restarted.
        //
        // The dial it may have to make first is bounded like the call it
        // is part of. Without credentials there is nothing to dial here —
        // the channel connects inside the `get`, under the same bound —
        // but with them the `Authenticate` round trip happens on the way
        // in, and an endpoint that accepts TCP and answers nothing would
        // otherwise hold this read open forever: no backoff, no failure
        // recorded, and `/status/config` still reporting connected.
        let prefix = self.prefix.as_bytes().to_vec();
        let resp = self
            .client
            .call(self.request_timeout, move |client| {
                let prefix = prefix.clone();
                async move {
                    kv_client(&client)
                        .get(prefix, Some(GetOptions::new().with_prefix()))
                        .await
                }
            })
            .await
            .map_err(|e| provider_error(e, "range read", ProviderError::Range))?;

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
        let prefix = self.prefix.as_bytes().to_vec();
        // Creating the watch is request/response shaped — etcd-client
        // sends the create request and awaits the server's create
        // confirmation before handing back the stream — so it is bounded
        // like any other unary call. Consuming the stream that comes back
        // is not: see the `request_timeout` field docs.
        //
        // The failure this closes is silent. An etcd that answers range
        // reads but never confirms the watch leaves the gateway serving
        // its first snapshot forever, blind to every later change, while
        // `/status/config` still reports it connected.
        let (watcher, stream) = self
            .client
            .call(self.request_timeout, move |client| {
                let prefix = prefix.clone();
                async move {
                    let opts = WatchOptions::new()
                        .with_prefix()
                        .with_start_revision(start_revision);
                    watch_client(&client).watch(prefix, Some(opts)).await
                }
            })
            .await
            .map_err(|e| provider_error(e, "watch create", ProviderError::Watch))?;

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
                    // No re-authentication here: an established stream
                    // cannot be moved onto a new connection, and a poll
                    // is not a place to dial one. Ending the stream is
                    // the recovery — the supervisor re-enters its cycle,
                    // and `load_all` re-authenticates on the way through.
                    Poll::Ready(Some(Err(provider_error(
                        CallError::Call(err),
                        "watch stream",
                        ProviderError::Watch,
                    ))))
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

    /// Credentials for a dial. Any user is enough to make
    /// `Client::connect` issue the `Authenticate` RPC, which is the only
    /// thing that makes it perform I/O — and the reason an authenticated
    /// deployment used to exit on an etcd an unauthenticated one waits for.
    fn credentials() -> Option<ConnectOptions> {
        Some(ConnectOptions::new().with_user("root", "rootpw"))
    }

    /// An address nothing is listening on: every dial is refused.
    async fn closed_endpoint() -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        format!("http://{addr}")
    }

    /// An etcd that answers every call with one gRPC status and nothing
    /// else — a trailers-only response, which is how a real server
    /// reports a refusal. `code` is the wire number: 3 is
    /// `InvalidArgument`, what etcd answers a wrong password with; 14 is
    /// `Unavailable`, what it answers while it has no leader.
    async fn spawn_grpc_status_server(code: u16, message: &'static str) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((socket, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let Ok(mut conn) = h2::server::handshake(socket).await else {
                        return;
                    };
                    while let Some(Ok((_req, mut respond))) = conn.accept().await {
                        let response = http::Response::builder()
                            .status(200)
                            .header("content-type", "application/grpc")
                            .header("grpc-status", code.to_string())
                            .header("grpc-message", message)
                            .body(())
                            .unwrap();
                        let _ = respond.send_response(response, true);
                    }
                });
            }
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn an_unreachable_authenticated_etcd_does_not_end_the_boot() {
        // The bug: with `etcd.user` set, `Client::connect` issues the
        // `Authenticate` RPC, so an etcd that is down failed the connect
        // — and the boot path exited on it. An unauthenticated
        // deployment, whose connect performs no I/O, has always started
        // and waited. Both must wait.
        let endpoint = closed_endpoint().await;
        let started = std::time::Instant::now();
        let provider = tokio::time::timeout(
            Duration::from_secs(5),
            EtcdConfigProvider::connect(&[endpoint], "/aisix", credentials(), None, None),
        )
        .await
        .expect("an unreachable etcd must not hold the boot path")
        .expect("an unreachable etcd must not fail the boot");
        assert!(
            started.elapsed() < CONNECT_RETRY_INTERVAL,
            "the boot must not spend a retry ladder before starting: {:?}",
            started.elapsed(),
        );

        // …and the failure lands where the supervisor already retries it.
        let err = provider.load_all().await.expect_err("etcd is not there");
        assert!(
            matches!(err, ProviderError::Connect(_)),
            "an unreachable etcd must stay retryable, got {err:?}",
        );
    }

    #[tokio::test]
    async fn an_unreachable_etcd_connects_on_the_first_call_that_reaches_it() {
        // The other half of deferring: the connection is pending, not
        // abandoned. Nothing about the provider has to be rebuilt for the
        // gateway to pick etcd up once it answers — which is what lets
        // the supervisor's retry loop bind a configuration on its own.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        let provider = EtcdConfigProvider::connect(
            &[format!("http://{addr}")],
            "/aisix",
            credentials(),
            Some(Duration::from_secs(2)),
            None,
        )
        .await
        .expect("an unreachable etcd must not fail the boot");
        assert!(matches!(
            provider.load_all().await,
            Err(ProviderError::Connect(_))
        ));

        // The endpoint comes back — as a server that refuses the
        // credentials, which is an answer, and so proves the dial was
        // re-attempted rather than short-circuited by the first failure.
        let server = tokio::net::TcpListener::bind(addr).await.unwrap();
        tokio::spawn(async move {
            while let Ok((socket, _)) = server.accept().await {
                tokio::spawn(async move {
                    let Ok(mut conn) = h2::server::handshake(socket).await else {
                        return;
                    };
                    while let Some(Ok((_req, mut respond))) = conn.accept().await {
                        let response = http::Response::builder()
                            .status(200)
                            .header("content-type", "application/grpc")
                            .header("grpc-status", "3")
                            .header("grpc-message", "etcdserver: authentication failed")
                            .body(())
                            .unwrap();
                        let _ = respond.send_response(response, true);
                    }
                });
            }
        });
        let err = provider
            .load_all()
            .await
            .expect_err("the credentials are refused");
        assert!(
            matches!(err, ProviderError::Rejected(_)),
            "a re-dial that reached the server must report its answer, got {err:?}",
        );
    }

    #[tokio::test]
    async fn credentials_the_server_refuses_end_the_boot_on_the_first_answer() {
        // The other class: etcd answered. Waiting cannot turn a wrong
        // password into a right one, so the boot stops and names the
        // cause instead of retrying — and it stops on the first answer,
        // not after the connect ladder.
        let endpoint = spawn_grpc_status_server(
            3,
            "etcdserver: authentication failed, invalid user ID or password",
        )
        .await;
        let started = std::time::Instant::now();
        let err = EtcdConfigProvider::connect(&[endpoint], "/aisix", credentials(), None, None)
            .await
            .expect_err("refused credentials cannot produce a usable provider");
        let ProviderError::Rejected(msg) = err else {
            panic!("a refusal must be its own class, got {err:?}");
        };
        assert!(
            msg.contains("authentication failed"),
            "the error must carry what etcd said: {msg}",
        );
        assert!(
            started.elapsed() < CONNECT_RETRY_INTERVAL,
            "a refusal must not walk the retry ladder: {:?}",
            started.elapsed(),
        );
    }

    #[tokio::test]
    async fn the_export_path_stops_on_a_refusal_instead_of_retrying_it() {
        // Same split on the CLI's eager path, which does retry an
        // unreachable etcd: three attempts a second apart would be
        // visible in the elapsed time if a refusal walked the ladder.
        let endpoint = spawn_grpc_status_server(7, "etcdserver: permission denied").await;
        let policy = ConnectPolicy {
            interval: Duration::from_secs(1),
            attempts: 3,
        };
        let started = std::time::Instant::now();
        let err = EtcdConfigProvider::connect_with_policy(
            &[endpoint],
            "/aisix",
            credentials(),
            None,
            None,
            policy,
        )
        .await
        .expect_err("refused credentials cannot produce a usable provider");
        assert!(matches!(err, ProviderError::Rejected(_)), "got {err:?}");
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "a refusal must not walk the retry ladder: {:?}",
            started.elapsed(),
        );
    }

    #[tokio::test]
    async fn a_server_that_cannot_serve_yet_is_treated_as_unreachable() {
        // etcd answers `Unavailable` while it has no leader or is
        // stopping, and tonic manufactures the same code for a refused
        // TCP connect. Both heal on their own, so neither may end a boot
        // — and classifying on the status code rather than the message
        // is what keeps them together.
        let endpoint = spawn_grpc_status_server(14, "etcdserver: no leader").await;
        let provider =
            EtcdConfigProvider::connect(&[endpoint], "/aisix", credentials(), None, None)
                .await
                .expect("an etcd that cannot serve yet must not fail the boot");
        assert!(matches!(
            provider.load_all().await,
            Err(ProviderError::Connect(_))
        ));
    }

    #[tokio::test]
    async fn the_dial_timeout_bounds_an_authenticated_connect_that_is_never_answered() {
        // `ConnectOptions::with_connect_timeout` bounds the TCP connect
        // only. An endpoint that completes TCP and then goes silent
        // leaves the TLS handshake and the `Authenticate` round trip
        // unbounded, so the dial hangs — and with it the boot, before any
        // listener is up. `etcd.dial_timeout_ms` has to cover the whole
        // dial, and its expiry is a reachability failure: retried, not
        // fatal.
        let endpoint = spawn_silent_h2_server().await;
        let provider = tokio::time::timeout(
            Duration::from_secs(10),
            EtcdConfigProvider::connect(
                &[endpoint],
                "/aisix",
                credentials(),
                None,
                Some(Duration::from_millis(300)),
            ),
        )
        .await
        .expect("the dial must be bounded; without the bound this hangs")
        .expect("an endpoint that never answers must not fail the boot");

        let err = provider
            .load_all()
            .await
            .expect_err("a dial that never completes cannot read");
        let ProviderError::Connect(msg) = err else {
            panic!("an expired dial must stay retryable, got {err:?}");
        };
        assert!(
            msg.contains("etcd.dial_timeout_ms"),
            "the message must name the key that bounded it: {msg}",
        );
    }

    #[tokio::test]
    async fn a_refusal_the_read_meets_is_reported_as_a_refusal() {
        // Where etcd refuses is not the caller's choice. A wrong password
        // is refused by `Authenticate`, but a user who authenticates and
        // then lacks the permission is refused by the range read, and a
        // token an etcd restart invalidated is refused by every call
        // after it. Classifying only the dial would report those as
        // ordinary transport trouble and bury them in a warn line that
        // repeats forever.
        //
        // No credentials here on purpose: that is the shape where the
        // dial cannot classify anything at all, because it does no I/O.
        let endpoint = spawn_grpc_status_server(7, "etcdserver: permission denied").await;
        let provider = EtcdConfigProvider::connect(&[endpoint], "/aisix", None, None, None)
            .await
            .expect("connect is lazy without credentials");

        let err = provider
            .load_all()
            .await
            .expect_err("a refused read cannot succeed");
        let ProviderError::Rejected(msg) = err else {
            panic!("a refusal on the read must be its own class, got {err:?}");
        };
        assert!(
            msg.contains("permission denied"),
            "the error must carry what etcd said: {msg}",
        );
    }

    #[tokio::test]
    async fn the_request_timeout_also_bounds_the_dial_a_read_has_to_make() {
        // Without credentials there is nothing to dial on the way into a
        // read: the channel connects inside the `get`, under
        // `etcd.request_timeout_ms` like the rest of the call. With them
        // the `Authenticate` round trip happens first, so a dial left
        // outside that bound would hold the read open forever against an
        // endpoint that accepts TCP and answers nothing — no backoff, no
        // recorded failure, and a `/status/config` still reporting
        // connected. No `dial_timeout_ms` here on purpose: the operator
        // set one bound, and it has to cover the whole read.
        //
        // The endpoint is refused at connect time and only becomes the
        // silent one afterwards, so the dial under test is the read's.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        let provider = EtcdConfigProvider::connect(
            &[format!("http://{addr}")],
            "/aisix",
            credentials(),
            Some(Duration::from_millis(300)),
            None,
        )
        .await
        .expect("an unreachable etcd must not fail the boot");

        serve_silently(tokio::net::TcpListener::bind(addr).await.unwrap());

        let err = tokio::time::timeout(Duration::from_secs(10), provider.load_all())
            .await
            .expect("the dial the read makes must be bounded; without it this hangs")
            .expect_err("a dial that never completes cannot read");
        let ProviderError::Connect(msg) = err else {
            panic!("an expired dial must stay retryable, got {err:?}");
        };
        assert!(
            msg.contains("etcd connect") && msg.contains("etcd.request_timeout_ms"),
            "the message must name the call and the key that bounded it: {msg}",
        );
    }

    #[tokio::test]
    async fn an_unset_dial_timeout_leaves_the_dial_unbounded() {
        // `0` and unset both mean unbounded (#1134), and that has to keep
        // meaning it here: a bound reintroduced as a default would abort
        // dials an operator asked to leave alone.
        let endpoint = spawn_silent_h2_server().await;
        let deferred = tokio::time::timeout(
            Duration::from_millis(750),
            EtcdConfigProvider::connect(&[endpoint], "/aisix", credentials(), None, None),
        )
        .await;
        assert!(
            deferred.is_err(),
            "with no dial timeout the connect must still be in flight",
        );
    }

    #[tokio::test]
    async fn each_status_code_lands_on_one_side_deliberately() {
        // The whole fix rests on telling "no answer" from "answered and
        // refused", so the split is pinned per wire code against a server
        // that really sends them. The retryable set is an allowlist:
        // anything outside it ends the boot, which is what keeps a wrong
        // password from being waited out forever.
        //
        // 2 Unknown, 4 DeadlineExceeded, 13 Internal and 14 Unavailable
        // are what a call that never got a usable answer arrives as —
        // tonic manufactures Unavailable for a refused connect, a DNS
        // failure and a failed TLS handshake, and etcd answers it while
        // it has no leader. 1 Cancelled and 10 Aborted are a peer or an
        // intermediary resetting the stream; 8 ResourceExhausted is
        // throttling. All of them heal without anyone editing a config.
        for code in [1u16, 2, 4, 8, 10, 13, 14] {
            let endpoint = spawn_grpc_status_server(code, "no answer").await;
            let provider =
                EtcdConfigProvider::connect(&[endpoint], "/aisix", credentials(), None, None)
                    .await
                    .unwrap_or_else(|e| panic!("code {code} must be waited out, got {e:?}"));
            assert!(matches!(
                provider.load_all().await,
                Err(ProviderError::Connect(_))
            ));
        }
        // 16 Unauthenticated is the third class: etcd answered, and what
        // it refused is the token rather than the credentials — it
        // restarted, or `--auth-token-ttl` elapsed. Dialling again is
        // what cures that, so the boot waits like the codes above rather
        // than ending. On a call rather than a dial,
        // `LazyEtcdClient::call` re-authenticates and retries it; here
        // it is the dial itself that is refused, and there is nothing
        // left to re-authenticate.
        {
            let endpoint = spawn_grpc_status_server(16, "etcdserver: invalid auth token").await;
            let provider =
                EtcdConfigProvider::connect(&[endpoint], "/aisix", credentials(), None, None)
                    .await
                    .expect("an invalid token must be waited out, not fatal");
            assert!(matches!(
                provider.load_all().await,
                Err(ProviderError::Connect(_))
            ));
        }
        // 3 InvalidArgument (etcd's answer to a wrong user or password),
        // 7 PermissionDenied, 9 FailedPrecondition (credentials sent to a
        // cluster with authentication disabled).
        for code in [3u16, 7, 9] {
            let endpoint = spawn_grpc_status_server(code, "refused").await;
            let err = EtcdConfigProvider::connect(&[endpoint], "/aisix", credentials(), None, None)
                .await
                .expect_err("a refusal must end the boot");
            assert!(
                matches!(err, ProviderError::Rejected(_)),
                "code {code} must be fatal, got {err:?}",
            );
        }
    }

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
    async fn endpoints_that_cannot_be_used_at_all_end_the_boot() {
        // An empty endpoint list is a parse failure inside etcd-client,
        // not a reachability one: no retry ladder can turn it into a
        // usable endpoint, so it must reach the caller as a refusal
        // rather than deferring to a supervisor that would dial it
        // forever.
        let endpoints: Vec<String> = vec![];
        let err = EtcdConfigProvider::connect(&endpoints, "/aisix", None, None, None)
            .await
            .unwrap_err();
        assert!(
            matches!(err, ProviderError::Rejected(_)),
            "unusable endpoints must be fatal, got {err:?}",
        );
    }

    /// An etcd that is reachable and answers nothing.
    ///
    /// A real HTTP/2 server: the connection preface and SETTINGS exchange
    /// complete, every request is accepted, and no response is ever sent.
    /// That is the failure the bound below exists for — the call reached
    /// a reachable etcd, which simply never answered it.
    ///
    /// Returns the endpoint. The listener task lives for the test.
    async fn spawn_silent_h2_server() -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        serve_silently(listener);
        format!("http://{addr}")
    }

    /// The silent server on a listener the caller owns, so a test can
    /// leave a port dead first and make it silent afterwards.
    fn serve_silently(listener: tokio::net::TcpListener) {
        tokio::spawn(async move {
            while let Ok((socket, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let Ok(mut conn) = h2::server::handshake(socket).await else {
                        return;
                    };
                    // Hold each responder without ever using it: dropping
                    // one would RST_STREAM the request, which the client
                    // would see as an answer.
                    let mut accepted = Vec::new();
                    while let Some(Ok((_req, respond))) = conn.accept().await {
                        accepted.push(respond);
                    }
                });
            }
        });
    }

    #[tokio::test]
    async fn the_request_timeout_bounds_a_range_read_that_is_never_answered() {
        // The symmetric half of the watch-create spec below, and the only
        // thing that pins `load_all`'s own variant now that `bound` takes
        // the constructor as a parameter: the helper-level spec passes
        // `ProviderError::Range` in itself, so it would stay green if this
        // call site started reporting expiry as `Watch`. The supervisor
        // routes both to the same backoff, so the variant is what the warn
        // line says — which is the whole point of naming the call.
        let endpoint = spawn_silent_h2_server().await;
        let provider = EtcdConfigProvider::connect(
            &[endpoint],
            "/aisix",
            None,
            Some(Duration::from_millis(300)),
            None,
        )
        .await
        .expect("connect is lazy without credentials");

        let err = tokio::time::timeout(Duration::from_secs(10), provider.load_all())
            .await
            .expect("the range read must be bounded; without the bound this hangs")
            .expect_err("an unanswered range read cannot succeed");

        let ProviderError::Range(msg) = err else {
            panic!("expiry must surface as ProviderError::Range, got {err:?}");
        };
        assert!(
            msg.contains("range read") && msg.contains("etcd.request_timeout_ms"),
            "the message must name the call and the key that bounded it: {msg}",
        );
    }

    #[tokio::test]
    async fn the_request_timeout_bounds_a_watch_creation_that_is_never_confirmed() {
        // Creating a watch is request/response shaped: etcd-client sends
        // the create request and awaits the server's create confirmation
        // before handing back the stream. Unbounded, an etcd that answers
        // range reads and never confirms the watch leaves the gateway
        // serving its first snapshot forever, blind to every later
        // change, while `/status/config` still reports it connected.
        let endpoint = spawn_silent_h2_server().await;
        // No user, so `Client::connect` is lazy and returns without a
        // round trip — the stall this asserts on is the watch create.
        let provider = EtcdConfigProvider::connect(
            &[endpoint],
            "/aisix",
            None,
            Some(Duration::from_millis(300)),
            None,
        )
        .await
        .expect("connect is lazy without credentials");

        let err = tokio::time::timeout(Duration::from_secs(10), provider.watch(1))
            .await
            .expect("the watch create must be bounded; without the bound this hangs")
            .err()
            .expect("an unconfirmed watch create cannot succeed");

        let ProviderError::Watch(msg) = err else {
            panic!("expiry must surface as ProviderError::Watch, got {err:?}");
        };
        assert!(
            msg.contains("watch create") && msg.contains("etcd.request_timeout_ms"),
            "the message must name the call and the key that bounded it: {msg}",
        );
    }

    #[test]
    fn expiry_reports_the_call_and_the_key_that_bounded_it() {
        // An expiry has to reach the operator as a diagnosable failure,
        // not as a bare cancellation: the message names both the call it
        // aborted and the config key that set the bound, so a boot loop
        // is traceable to `etcd.request_timeout_ms` without a code read.
        // `Range` is chosen to match the variant this call's transport
        // failures already use — the supervisor sends every
        // `ProviderError` from `load_all` down the same backoff path, so
        // the variant is what the warn line says, not what it does.
        let err = provider_error(
            CallError::CallTimeout(Duration::from_millis(1)),
            "range read",
            ProviderError::Range,
        );
        let ProviderError::Range(msg) = err else {
            panic!("expiry must surface as ProviderError::Range, got {err:?}");
        };
        assert!(
            msg.contains("range read") && msg.contains("etcd.request_timeout_ms"),
            "the message must name the call and the key that bounded it: {msg}"
        );
    }

    #[tokio::test]
    async fn a_freshly_authenticated_connection_etcd_still_refuses_is_a_refusal() {
        // By the time a call's `Unauthenticated` reaches `provider_error`
        // it has already been retried on a connection that authenticated
        // moments earlier. Whatever is wrong with it is not something
        // another 60s of backoff fixes, so it gets the refusal line an
        // operator can act on rather than the routine one they filter
        // out. No credentials here, so the dial is lazy and both attempts
        // land on the read — which is where the retry lives.
        let endpoint = spawn_grpc_status_server(16, "etcdserver: invalid auth token").await;
        let provider = EtcdConfigProvider::connect(&[endpoint], "/aisix", None, None, None)
            .await
            .expect("connect is lazy without credentials");
        let err = provider
            .load_all()
            .await
            .expect_err("a server that only answers 16 cannot serve a read");
        assert!(
            matches!(err, ProviderError::Rejected(_)),
            "a re-authenticated call etcd refused again must be loud, got {err:?}",
        );
    }

    #[test]
    fn a_dial_that_ran_out_of_time_is_attributed_to_the_connect() {
        // The connect is bounded separately from the call it carries, so
        // an operator reading the warn line can tell "etcd never finished
        // answering the dial" from "etcd never finished the read" —
        // different causes, and after a re-authentication both are
        // reachable from the same `load_all`.
        let err = provider_error(
            CallError::ConnectTimeout(Duration::from_millis(1)),
            "range read",
            ProviderError::Range,
        );
        let ProviderError::Connect(msg) = err else {
            panic!("a dial expiry must surface as ProviderError::Connect, got {err:?}");
        };
        assert!(
            msg.contains("etcd connect") && msg.contains("etcd.request_timeout_ms"),
            "the message must name the step and the key that bounded it: {msg}"
        );
    }
}
