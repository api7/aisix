//! etcd sub-clients configured with this gateway's gRPC decode limit.
//!
//! `Client::get` / `Client::watch` delegate to sub-clients the `Client`
//! keeps private, and `Client::kv_client()` / `watch_client()` hand back
//! clones — so the decode limit can only be raised on the sub-client that
//! actually issues the call. Every etcd read the gateway itself makes is
//! built here so no call site can drift back onto tonic's default; test
//! fixtures that talk to etcd directly are their own business.

use etcd_client::{Client, ConnectOptions, Error as EtcdError, KvClient, WatchClient};
use std::error::Error as StdError;
use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::sync::Mutex;

/// Maximum size, in bytes, of a gRPC message decoded from etcd.
///
/// The gateway reads its whole configuration set in a single range
/// response and keeps it resident in memory, so tonic's 4 MiB default is
/// really a cap on how much configuration a deployment may hold — and
/// crossing it is unrecoverable on its own: the supervisor backs off and
/// re-issues the identical oversized range forever. Any finite ceiling
/// would only move that failure to a larger config set, so the limit goes
/// to `i32::MAX`, the conventional ceiling for a gRPC message size.
///
/// The trade-off that buys: an oversized length prefix is no longer
/// rejected before the buffer for it is reserved. The peer is the etcd
/// the operator configured, whose entire contents this process already
/// trusts and holds resident, so the ceiling was never what stood between
/// it and this gateway's memory.
pub const MAX_DECODING_MESSAGE_SIZE: usize = i32::MAX as usize;

/// KV client for reads under [`MAX_DECODING_MESSAGE_SIZE`].
pub fn kv_client(client: &Client) -> KvClient {
    client
        .kv_client()
        .max_decoding_message_size(MAX_DECODING_MESSAGE_SIZE)
}

/// Watch client for streams under [`MAX_DECODING_MESSAGE_SIZE`].
pub fn watch_client(client: &Client) -> WatchClient {
    client
        .watch_client()
        .max_decoding_message_size(MAX_DECODING_MESSAGE_SIZE)
}

/// Flatten an error and its source chain into a single readable line.
/// Without this, tonic surfaces opaque strings like "dns error" while
/// the real cause (`getaddrinfo: Name or service not known`, TLS
/// handshake reason, …) hides in `.source()`. The supervisor logs
/// the returned string, so CI triage gets the full picture.
/// The readable cause of one `etcd-client` failure.
///
/// `etcd_client::Error` implements `Error` without `source()`, so
/// [`format_error_chain`] alone stops at its own `Display` and the cause
/// that matters — the transport error under a tonic `Status`, and the
/// `getaddrinfo` / TLS reason under that — never appears. Reach into the
/// payload to get there.
pub(crate) fn format_etcd_error(err: &EtcdError) -> String {
    let mut out = format_error_chain(err);
    if let EtcdError::GRpcStatus(status) = err {
        if let Some(source) = StdError::source(status) {
            let chain = format_error_chain(source);
            if !chain.is_empty() && !out.ends_with(&chain) {
                out.push_str(": ");
                out.push_str(&chain);
            }
        }
    }
    out
}

pub(crate) fn format_error_chain(err: &(dyn StdError + 'static)) -> String {
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

/// Why a dial failed, split by the only distinction that changes what the
/// gateway should do about it.
///
/// The two are not interchangeable. An etcd that cannot be reached comes
/// back on its own and the gateway must wait for it; credentials etcd has
/// refused never start working, and waiting on them turns a configuration
/// mistake into a gateway that is up and permanently empty.
#[derive(Debug, thiserror::Error)]
pub enum ConnectError {
    /// No server answered: TCP refused, DNS failure, TLS handshake
    /// failure, an expired `etcd.dial_timeout_ms`, or an etcd that is up
    /// but cannot serve (no leader, shutting down). Retryable.
    #[error("etcd did not answer: {0}")]
    Unreachable(String),
    /// etcd answered and refused — a wrong user or password, a user
    /// without the required permission, credentials sent to a cluster
    /// with authentication disabled — or the endpoints are not usable at
    /// all (unparseable URI, empty list). Retrying cannot change any of
    /// these, so they are fatal at boot.
    #[error("etcd refused the connection: {0}")]
    Rejected(String),
    /// etcd answered and refused *the token this connection is using*,
    /// not the configured credentials: the gateway authenticated once
    /// and etcd has since forgotten that token. Either the token's
    /// `--auth-token-ttl` elapsed — it is refreshed by use, so only an
    /// idle connection reaches it — or etcd's token store was cleared:
    /// authentication re-enabled on the cluster, a JWT signing key
    /// regenerated at startup, a member brought up on an empty data
    /// directory.
    ///
    /// Kept apart from [`Self::Rejected`] because the two need opposite
    /// handling. A refused credential never becomes a good one, so
    /// waiting on it is what turns a typo into a permanently empty
    /// gateway. A forgotten token is cured by the one thing the gateway
    /// has not done since it started: authenticate again. See
    /// [`LazyEtcdClient::call`].
    #[error("etcd rejected the connection's auth token: {0}")]
    Unauthenticated(String),
}

/// Canonical gRPC status codes, as sent on the wire
/// (<https://grpc.io/docs/guides/status-codes/>). Compared numerically so
/// this crate does not have to depend on — and keep in step with — the
/// exact `tonic` release `etcd-client` builds against.
const GRPC_CANCELLED: i32 = 1;
const GRPC_UNKNOWN: i32 = 2;
const GRPC_DEADLINE_EXCEEDED: i32 = 4;
const GRPC_RESOURCE_EXHAUSTED: i32 = 8;
const GRPC_ABORTED: i32 = 10;
const GRPC_INTERNAL: i32 = 13;
const GRPC_UNAVAILABLE: i32 = 14;
const GRPC_INVALID_ARGUMENT: i32 = 3;
const GRPC_UNAUTHENTICATED: i32 = 16;

/// etcd's own error strings for the two answers that mean "the token you
/// sent is stale" but arrive as `InvalidArgument`, sharing a status code
/// with "your password is wrong".
///
/// **Do not replace these with a status-code check, and do not delete
/// them as a fragile string match.** The text *is* the signal here.
/// etcd's own reference client recovers these errors the same way: its
/// `rpctypes.Error()` maps a gRPC error back to a typed one through a
/// table keyed on the error string, and `shouldRefreshToken` in the v3
/// client's retry interceptor then treats exactly these two, plus
/// `Unauthenticated`, as "re-authenticate and retry" — while
/// deliberately leaving `authentication failed, invalid user ID or
/// password` out. This match is that list, matched the way the
/// reference implementation matches it.
///
/// - `revision of auth store is old` — the deployment-facing one, and
///   the reason this matters. Under `--auth-token jwt` (what a
///   multi-member cluster runs) **any** change to the auth store bumps
///   its revision, and every token issued before it is refused from that
///   moment: one `etcdctl user add` on the cluster and a gateway can no
///   longer read its configuration until it authenticates again.
///   Verified against etcd 3.5.18.
/// - `user name is empty` — carried for parity with the reference
///   client, which refreshes on it too. No reachable scenario was
///   constructed for it here; it is included because it fails in the
///   safe direction (one extra re-authentication) and because a list
///   that silently drops one of upstream's three arms is the kind of
///   difference nobody finds later.
const STALE_TOKEN_MESSAGES: [&str; 2] = [
    "etcdserver: revision of auth store is old",
    "etcdserver: user name is empty",
];

fn is_stale_token(message: &str) -> bool {
    STALE_TOKEN_MESSAGES.iter().any(|m| message.contains(m))
}

impl ConnectError {
    /// The flattened cause, without this type's own framing — so a
    /// caller that re-reports it under its own wording does not stack
    /// two prefixes in front of what etcd actually said.
    pub fn detail(&self) -> &str {
        match self {
            Self::Unreachable(detail) | Self::Rejected(detail) | Self::Unauthenticated(detail) => {
                detail
            }
        }
    }
}

/// Sort one `etcd-client` failure into [`ConnectError`].
///
/// The split is structural, on the gRPC status code — never on the text
/// of a message. Verified against etcd 3.5 and etcd-client 0.14: a
/// refused TCP connect, a DNS failure and a failed TLS handshake all
/// arrive as `GRpcStatus` with code `Unavailable` (tonic manufactures
/// the status locally; the message is "tcp connect error" / "dns error"
/// / "tls handshake eof"), while a wrong password arrives as
/// `InvalidArgument` carrying etcd's own "authentication failed, invalid
/// user ID or password".
///
/// Used on the reads too, not only on the dial. Which failures etcd
/// answers *where* is not something a caller can assume: a user whose
/// password is wrong is refused by `Authenticate`, but a user who
/// authenticates and lacks the permission is refused by the range read
/// (`PermissionDenied`), and a token etcd has stopped holding is refused
/// by every later call (`Unauthenticated`). Classifying only the dial
/// would report those as ordinary transport trouble.
///
/// This is what the safety condition behind re-authenticating rests on.
/// A wrong user or password is answered on `Authenticate` with
/// `InvalidArgument` carrying `etcdserver: authentication failed, invalid
/// user ID or password`, and a user without the rights with
/// `PermissionDenied`; neither can reach the retry path however many
/// times it is presented.
///
/// `InvalidArgument` is not only that, though, which is why two of its
/// members are named by message below. See [`STALE_TOKEN_MESSAGES`].
pub(crate) fn classify(err: &EtcdError) -> ConnectError {
    let detail = format_etcd_error(err);
    match err {
        // The call never reached a server.
        EtcdError::TransportError(_) | EtcdError::IoError(_) => ConnectError::Unreachable(detail),
        EtcdError::GRpcStatus(status) => match status.code() as i32 {
            // `Unavailable` is both what tonic synthesises for every
            // failure to reach a server and what etcd itself answers
            // while it has no leader or is stopping. `DeadlineExceeded`
            // is a call that ran out of time. `Unknown` and `Internal`
            // are what a connection broken mid-call surfaces as.
            // `Cancelled` and `Aborted` are what a peer or an
            // intermediary resetting the stream produces — an ingress in
            // front of etcd restarting looks like this — and
            // `ResourceExhausted` is an etcd throttling a fleet of
            // gateways that all restarted at once. None of them is an
            // answer about the credentials, and all of them heal on
            // their own.
            GRPC_UNAVAILABLE
            | GRPC_DEADLINE_EXCEEDED
            | GRPC_UNKNOWN
            | GRPC_INTERNAL
            | GRPC_CANCELLED
            | GRPC_ABORTED
            | GRPC_RESOURCE_EXHAUSTED => ConnectError::Unreachable(detail),
            // etcd knows the call arrived with a token it no longer
            // holds — the TTL elapsed, or its token store was cleared.
            // The configured credentials are not what it is complaining
            // about, so this one heals by authenticating again.
            GRPC_UNAUTHENTICATED => ConnectError::Unauthenticated(detail),
            // The two `InvalidArgument` answers that mean the same
            // thing. See [`STALE_TOKEN_MESSAGES`] for why they are
            // matched by message and why that is not a workaround.
            GRPC_INVALID_ARGUMENT if is_stale_token(status.message()) => {
                ConnectError::Unauthenticated(detail)
            }
            // Everything else is etcd having evaluated the request and
            // said no: the rest of `InvalidArgument` — a wrong user or
            // password — `PermissionDenied` for a user without the
            // rights, `FailedPrecondition` for credentials sent to a
            // cluster that has authentication disabled.
            _ => ConnectError::Rejected(detail),
        },
        // Unparseable endpoints, an empty endpoint list, a bad header
        // value: configuration, not reachability.
        _ => ConnectError::Rejected(detail),
    }
}

/// Why one [`LazyEtcdClient::call`] attempt failed.
///
/// Split by step so a caller can attribute it in its own error type: the
/// connect and the call it carries are separate lines in an operator's
/// log, and each carries a different one of the two `etcd.*_timeout_ms`
/// keys.
#[derive(Debug, thiserror::Error)]
pub enum CallError {
    /// The connection could not be established.
    #[error(transparent)]
    Connect(ConnectError),
    /// Establishing the connection outran the caller's bound.
    #[error("etcd connect exceeded its bound ({} ms)", .0.as_millis())]
    ConnectTimeout(Duration),
    /// The connection was there and etcd failed the call.
    #[error("{0}")]
    Call(EtcdError),
    /// The call outran the caller's bound.
    #[error("the etcd call exceeded its bound ({} ms)", .0.as_millis())]
    CallTimeout(Duration),
}

/// How often a dial that has not finished says so.
///
/// Mirrors `FIRST_CONFIG_WAIT_LOG_INTERVAL` in `aisix-server`, which
/// reports the other half of the same wait: that gate explains why the
/// proxy listener is still closed, this one explains why no configuration
/// has arrived to open it.
const DIAL_WAIT_LOG_INTERVAL: Duration = Duration::from_secs(10);

/// Run `fut`, saying every `interval` that it has not finished yet.
///
/// What this closes is a silence, not a hang. `Client::connect` waits for
/// the `Authenticate` round trip, and against an endpoint that accepts the
/// TCP connection and then answers nothing it waits for as long as
/// `etcd.dial_timeout_ms` allows — unset, the shipped default, means
/// forever. The gateway is then stuck before any listener binds and, until
/// this, wrote not one line about it: an operator saw a process with no
/// port and no explanation. The bound itself is deliberately unchanged;
/// this only makes the wait visible.
///
/// First tick one interval in, not immediately, so a healthy dial — which
/// finishes in milliseconds — logs nothing at all.
async fn announce_while_pending<T>(
    interval: Duration,
    endpoints: &[String],
    fut: impl Future<Output = T>,
) -> T {
    tokio::pin!(fut);
    let mut ticker = tokio::time::interval_at(tokio::time::Instant::now() + interval, interval);
    // A runtime stalled past several ticks — a suspended host, a starved
    // scheduler — would otherwise wake to a burst of identical lines
    // under the default `Burst` behaviour. One line per interval that
    // actually elapsed is the whole point; the count of ticks missed
    // while nobody was running is not information.
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let started = tokio::time::Instant::now();
    loop {
        tokio::select! {
            // `biased` so a dial that completes in the same wakeup as a
            // tick reports success rather than one last complaint.
            biased;
            outcome = &mut fut => return outcome,
            _ = ticker.tick() => {
                tracing::warn!(
                    endpoints = ?endpoints,
                    waited_secs = started.elapsed().as_secs(),
                    "still connecting to etcd — nothing waiting on this connection can \
                     proceed until it answers",
                );
            }
        }
    }
}

/// An etcd connection established on demand, and kept once it is.
///
/// `Client::connect` only performs I/O when credentials are configured —
/// it issues the `Authenticate` RPC — so without them a client is handed
/// back before anything has been dialled and every failure surfaces later,
/// on the read, where the watch supervisor retries it. Connecting through
/// this type gives the authenticated deployment the same shape: a dial
/// that could not reach etcd leaves the connection pending instead of
/// failing the caller, so the gateway boots, waits for its first
/// configuration and picks etcd up when it comes back.
pub struct LazyEtcdClient {
    endpoints: Vec<String>,
    options: Option<ConnectOptions>,
    /// `etcd.dial_timeout_ms`. Wraps the whole of `Client::connect`, not
    /// just the TCP connect that `ConnectOptions::with_connect_timeout`
    /// bounds: the TLS handshake and the `Authenticate` round trip sit
    /// outside that option, so an endpoint that completes TCP and then
    /// goes silent would otherwise hang the dial with no bound at all.
    /// `None` — the default, and what `0` means — leaves it unbounded.
    dial_timeout: Option<Duration>,
    connected: Mutex<Option<Connected>>,
    /// Handed to the next connection [`LazyEtcdClient::dial`] establishes.
    /// See [`Connected::generation`].
    next_generation: AtomicU64,
}

/// One established connection, tagged so a caller whose call etcd refused
/// can ask for *that* connection to be replaced.
struct Connected {
    client: Client,
    /// Which dial produced this connection. A burst of calls that all
    /// failed on the same invalidated token quote the same generation, so
    /// the first to arrive replaces it and the rest see a connection that
    /// is already newer than the one they failed on and reuse it — one
    /// re-authentication for the burst, not one per call.
    generation: u64,
}

impl std::fmt::Debug for LazyEtcdClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyEtcdClient")
            .field("endpoints", &self.endpoints)
            .finish_non_exhaustive()
    }
}

impl LazyEtcdClient {
    pub fn new(
        endpoints: Vec<String>,
        options: Option<ConnectOptions>,
        dial_timeout: Option<Duration>,
    ) -> Self {
        Self {
            endpoints,
            options,
            dial_timeout,
            connected: Mutex::new(None),
            next_generation: AtomicU64::new(0),
        }
    }

    /// Dial now, so a caller that wants to classify the failure itself —
    /// the boot path, which exits on [`ConnectError::Rejected`] — can do
    /// it before anything else is started. A successful dial is cached
    /// like any other.
    pub async fn connect_now(&self) -> Result<(), ConnectError> {
        self.connection().await.map(|_| ())
    }

    /// The connection, dialling if this is the first call to get there,
    /// and the generation it belongs to — what [`Self::invalidate`] needs
    /// to replace exactly this connection.
    ///
    /// The lock is held across the dial on purpose: concurrent first
    /// calls wait for the one dial rather than opening a connection each.
    async fn connection(&self) -> Result<(Client, u64), ConnectError> {
        let mut slot = self.connected.lock().await;
        if let Some(connected) = slot.as_ref() {
            return Ok((connected.client.clone(), connected.generation));
        }
        let client = self.dial().await?;
        let generation = self.next_generation.fetch_add(1, Ordering::Relaxed);
        *slot = Some(Connected {
            client: client.clone(),
            generation,
        });
        Ok((client, generation))
    }

    /// Discard the connection `stale` names so the next call dials a new
    /// one, which authenticates on the way in.
    ///
    /// A no-op once someone else has already replaced that generation —
    /// see [`Connected::generation`].
    async fn invalidate(&self, stale: u64) {
        let mut slot = self.connected.lock().await;
        if slot.as_ref().is_some_and(|c| c.generation == stale) {
            *slot = None;
        }
    }

    /// Run one etcd call on the shared connection, authenticating again
    /// and retrying it **once** if etcd rejects the connection's auth
    /// token.
    ///
    /// This is how a gateway survives an elapsed `--auth-token-ttl`, or
    /// an etcd that has stopped holding its token at all:
    /// `etcd-client` authenticates inside
    /// `Client::connect` and never again, so the token this connection
    /// carries is the only one it will ever have, and once etcd forgets
    /// it every later call is refused until the process is restarted.
    /// Rebuilding the client is what re-runs `Authenticate`.
    ///
    /// The retry is bounded at one attempt, and reached only from
    /// [`ConnectError::Unauthenticated`] — never from
    /// [`ConnectError::Rejected`]. Credentials etcd has refused
    /// therefore fail on the first answer, exactly as they did before:
    /// a wrong password cannot spin here, and neither can a token etcd
    /// keeps refusing, which surfaces after the second attempt.
    ///
    /// `timeout` bounds each attempt — the dial and the call separately,
    /// as [`CallError`]'s two timeout variants say — rather than the pair
    /// of them together, so a re-authenticated retry gets the same window
    /// the original call had instead of whatever was left of it.
    pub async fn call<T, F, Fut>(
        &self,
        timeout: Option<Duration>,
        mut op: F,
    ) -> Result<T, CallError>
    where
        F: FnMut(Client) -> Fut,
        Fut: Future<Output = Result<T, EtcdError>>,
    {
        let mut reauthenticated = false;
        loop {
            let (client, generation) = match timeout {
                None => self.connection().await,
                Some(d) => tokio::time::timeout(d, self.connection())
                    .await
                    .map_err(|_| CallError::ConnectTimeout(d))?,
            }
            .map_err(CallError::Connect)?;

            let call = op(client);
            let outcome = match timeout {
                None => call.await,
                Some(d) => tokio::time::timeout(d, call)
                    .await
                    .map_err(|_| CallError::CallTimeout(d))?,
            };

            match outcome {
                Ok(value) => return Ok(value),
                Err(err) if !reauthenticated => {
                    let ConnectError::Unauthenticated(detail) = classify(&err) else {
                        return Err(CallError::Call(err));
                    };
                    tracing::info!(
                        error = %detail,
                        "etcd no longer accepts this connection's auth token — \
                         authenticating again and retrying",
                    );
                    self.invalidate(generation).await;
                    reauthenticated = true;
                }
                Err(err) => return Err(CallError::Call(err)),
            }
        }
    }

    async fn dial(&self) -> Result<Client, ConnectError> {
        let connect = announce_while_pending(
            DIAL_WAIT_LOG_INTERVAL,
            &self.endpoints,
            Client::connect(&self.endpoints, self.options.clone()),
        );
        let outcome = match self.dial_timeout {
            None => connect.await,
            Some(d) => match tokio::time::timeout(d, connect).await {
                Ok(outcome) => outcome,
                // An endpoint that accepts the TCP connection and then
                // answers nothing is unreachable in every sense that
                // matters here, so this joins the retry path rather than
                // ending the boot.
                Err(_) => {
                    return Err(ConnectError::Unreachable(format!(
                        "connect exceeded etcd.dial_timeout_ms ({} ms)",
                        d.as_millis()
                    )))
                }
            },
        };
        outcome.map_err(|err| classify(&err))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn unusable_endpoints_are_rejected_not_retried() {
        // No server was ever involved, and no retry ladder can make an
        // empty endpoint list dialable — so it is fatal, not a wait.
        let err = EtcdError::InvalidArgs("empty endpoints".to_string());
        assert!(matches!(classify(&err), ConnectError::Rejected(_)));
    }

    #[tokio::test]
    async fn the_cause_under_a_status_reaches_the_message() {
        // `etcd_client::Error` has no `source()`, so walking the chain
        // from it alone stops at its own `Display` and the reason the
        // dial failed — which lives under the tonic `Status` — never
        // reaches the log line the supervisor prints.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        let err = match Client::connect(
            [format!("http://{addr}")],
            Some(ConnectOptions::new().with_user("u", "p")),
        )
        .await
        {
            Err(err) => err,
            Ok(_) => panic!("nothing is listening on {addr}"),
        };

        let shallow = format_error_chain(&err);
        let deep = format_etcd_error(&err);
        assert!(
            deep.len() > shallow.len() && deep.starts_with(&shallow),
            "the payload's own cause chain must be appended: {deep:?} vs {shallow:?}",
        );
    }

    #[test]
    fn a_local_io_failure_is_reachability_not_refusal() {
        let err = EtcdError::IoError(std::io::Error::other("connection reset by peer"));
        assert!(matches!(classify(&err), ConnectError::Unreachable(_)));
    }
}

#[cfg(test)]
mod reauth_tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    use std::sync::{Arc, Mutex as StdMutex};

    /// A server that answers each call with the next gRPC status in its
    /// script — trailers-only, which is how a real server reports a
    /// refusal — and counts both the calls it was asked and the TCP
    /// connections it was asked them on.
    ///
    /// The connection count is what separates "the call was retried" from
    /// "the client was rebuilt": re-authenticating means a new
    /// `Client::connect`, and a new `Client::connect` means a second
    /// connection here. A retry on the same channel would carry the same
    /// token and change nothing.
    struct ScriptedEtcd {
        endpoint: String,
        connections: Arc<AtomicUsize>,
        calls: Arc<AtomicUsize>,
    }

    async fn spawn_scripted_etcd(script: &[(u16, &'static str)]) -> ScriptedEtcd {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let connections = Arc::new(AtomicUsize::new(0));
        let calls = Arc::new(AtomicUsize::new(0));
        let remaining = Arc::new(StdMutex::new(script.to_vec()));
        let (conns, reqs) = (connections.clone(), calls.clone());
        tokio::spawn(async move {
            while let Ok((socket, _)) = listener.accept().await {
                conns.fetch_add(1, Ordering::Relaxed);
                let (reqs, remaining) = (reqs.clone(), remaining.clone());
                tokio::spawn(async move {
                    let Ok(mut conn) = h2::server::handshake(socket).await else {
                        return;
                    };
                    while let Some(Ok((_req, mut respond))) = conn.accept().await {
                        reqs.fetch_add(1, Ordering::Relaxed);
                        let mut script = remaining.lock().unwrap();
                        // Past the end of the script the server keeps
                        // answering, so an unexpected extra call shows up
                        // as a count rather than as a hang.
                        let (code, message) = if script.is_empty() {
                            (14, "off the end of the script")
                        } else {
                            script.remove(0)
                        };
                        drop(script);
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
        ScriptedEtcd {
            endpoint: format!("http://{addr}"),
            connections,
            calls,
        }
    }

    /// One range read through [`LazyEtcdClient::call`], unbounded.
    ///
    /// No credentials, so `Client::connect` performs no I/O and every
    /// answer in the script lands on the read — the shape a token that
    /// went stale after the dial has.
    async fn read_once(endpoint: &str) -> Result<(), CallError> {
        let client = LazyEtcdClient::new(vec![endpoint.to_string()], None, None);
        client
            .call(None, |c| async move {
                kv_client(&c).get("/aisix", None).await.map(|_| ())
            })
            .await
    }

    #[tokio::test]
    async fn a_stale_token_is_retried_on_a_freshly_authenticated_connection() {
        // etcd forgot the token this connection authenticated with — it
        // restarted, or `--auth-token-ttl` elapsed. Nothing about the
        // configured credentials changed, and `etcd-client` never
        // authenticates again on its own, so without rebuilding the
        // client every later call is refused until the gateway is
        // restarted. The second answer is `Unavailable` only so the
        // retry is distinguishable from the first attempt.
        let etcd = spawn_scripted_etcd(&[
            (16, "etcdserver: invalid auth token"),
            (14, "etcdserver: no leader"),
        ])
        .await;

        let err = read_once(&etcd.endpoint)
            .await
            .expect_err("both scripted answers are failures");
        let CallError::Call(err) = err else {
            panic!("the server answered, so this is a call failure: {err:?}");
        };
        assert!(
            matches!(classify(&err), ConnectError::Unreachable(_)),
            "the retry's own answer is what surfaces, not the stale token: {err:?}",
        );
        assert_eq!(
            etcd.calls.load(Ordering::Relaxed),
            2,
            "a stale token must be retried exactly once",
        );
        assert_eq!(
            etcd.connections.load(Ordering::Relaxed),
            2,
            "the retry must run on a new connection — that is what re-authenticates",
        );
    }

    #[tokio::test]
    async fn a_token_etcd_keeps_refusing_stops_after_one_retry() {
        // The safety condition, at the mechanism: however etcd came to
        // refuse the token, one re-authentication is all it gets. A
        // second refusal ends the call instead of dialling again.
        let etcd = spawn_scripted_etcd(&[
            (16, "etcdserver: invalid auth token"),
            (16, "etcdserver: invalid auth token"),
        ])
        .await;

        let err = read_once(&etcd.endpoint).await.expect_err("both refuse");
        let CallError::Call(err) = err else {
            panic!("the server answered, so this is a call failure: {err:?}");
        };
        assert!(matches!(classify(&err), ConnectError::Unauthenticated(_)));
        assert_eq!(
            etcd.calls.load(Ordering::Relaxed),
            2,
            "re-authenticating must not become a loop",
        );
    }

    #[tokio::test]
    async fn a_stale_auth_store_revision_is_retried_like_an_invalid_token() {
        // `revision of auth store is old` is `InvalidArgument`, the same
        // code as a wrong password, and it is the one a deployment
        // actually meets: under `--auth-token jwt` any change to the
        // auth store — one `etcdctl user add` — bumps the revision and
        // every token issued before it is refused from that moment. Only
        // etcd's message separates the two, which is why the message is
        // matched. Second answer is `Unavailable` so the retry is
        // distinguishable from the first attempt.
        let etcd = spawn_scripted_etcd(&[
            (3, "etcdserver: revision of auth store is old"),
            (14, "etcdserver: no leader"),
        ])
        .await;

        let err = read_once(&etcd.endpoint).await.expect_err("both fail");
        let CallError::Call(err) = err else {
            panic!("the server answered, so this is a call failure: {err:?}");
        };
        assert!(
            matches!(classify(&err), ConnectError::Unreachable(_)),
            "the retry's own answer is what surfaces: {err:?}",
        );
        assert_eq!(
            etcd.calls.load(Ordering::Relaxed),
            2,
            "an auth-store revision change must be re-authenticated, not reported as a \
             wrong password",
        );
        assert_eq!(
            etcd.connections.load(Ordering::Relaxed),
            2,
            "the retry must run on a new connection — that is what re-authenticates",
        );
    }

    #[tokio::test]
    async fn an_empty_user_name_is_retried_like_an_invalid_token() {
        // Carried for parity with etcd's own reference client, which
        // refreshes its token on this too. No reachable scenario was
        // constructed for it — this pins the wiring, not a reproduction.
        let etcd = spawn_scripted_etcd(&[
            (3, "etcdserver: user name is empty"),
            (14, "etcdserver: no leader"),
        ])
        .await;

        read_once(&etcd.endpoint).await.expect_err("both fail");
        assert_eq!(etcd.calls.load(Ordering::Relaxed), 2);
        assert_eq!(etcd.connections.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn a_refused_credential_is_answered_once_and_never_retried() {
        // The safety condition, and the one line in this file that must
        // never move: a wrong user or password shares `InvalidArgument`
        // with the two stale-token answers above, and is told apart from
        // them by etcd's own message. It must fail on the first answer,
        // as it did before any of this — no re-authentication, no second
        // connection, nothing for an operator's typo to spin on.
        let etcd = spawn_scripted_etcd(&[(
            3,
            "etcdserver: authentication failed, invalid user ID or password",
        )])
        .await;

        let err = read_once(&etcd.endpoint).await.expect_err("refused");
        let CallError::Call(err) = err else {
            panic!("the server answered, so this is a call failure: {err:?}");
        };
        assert!(
            matches!(classify(&err), ConnectError::Rejected(_)),
            "a refused credential must stay refused: {err:?}",
        );
        assert_eq!(
            etcd.calls.load(Ordering::Relaxed),
            1,
            "a refused credential must not be presented a second time",
        );
        assert_eq!(
            etcd.connections.load(Ordering::Relaxed),
            1,
            "a refused credential must not cause a re-dial",
        );
    }

    /// A writer every clone of which appends to the same buffer, so a
    /// test can read back what was logged.
    #[derive(Clone)]
    struct SharedBuf(Arc<StdMutex<Vec<u8>>>);

    impl std::io::Write for SharedBuf {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl tracing_subscriber::fmt::MakeWriter<'_> for SharedBuf {
        type Writer = Self;
        fn make_writer(&self) -> Self {
            self.clone()
        }
    }

    #[tokio::test]
    async fn a_dial_that_never_finishes_says_so_while_it_waits() {
        // `etcd.dial_timeout_ms` unset — the shipped default — against an
        // endpoint that accepts the TCP connection and then answers
        // nothing. `Client::connect` waits for the `Authenticate` round
        // trip forever, so the gateway is stuck before any listener
        // binds; until this line it wrote nothing at all, and an operator
        // saw a process with no port and no explanation.
        //
        // The bound is deliberately untouched: what is asserted here is
        // that the wait is audible, not that it ends.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let mut held = Vec::new();
            while let Ok((socket, _)) = listener.accept().await {
                // Held, never spoken to: closing the socket would be an
                // answer, and the dial would end.
                held.push(socket);
            }
        });

        let logs = Arc::new(StdMutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_writer(SharedBuf(logs.clone()))
            .finish();
        let _guard = tracing::subscriber::set_default(subscriber);

        let client = LazyEtcdClient::new(
            vec![format!("http://{addr}")],
            Some(ConnectOptions::new().with_user("root", "rootpw")),
            None,
        );
        let dial = client.connect_now();
        tokio::pin!(dial);

        tokio::time::pause();
        assert!(
            futures::poll!(&mut dial).is_pending(),
            "a silent endpoint cannot complete a dial",
        );
        assert!(
            logs.lock().unwrap().is_empty(),
            "a dial that has only just started says nothing — a line on every \
             healthy boot is one operators learn to filter out",
        );

        tokio::time::advance(DIAL_WAIT_LOG_INTERVAL + Duration::from_secs(1)).await;
        assert!(
            futures::poll!(&mut dial).is_pending(),
            "the endpoint is still silent",
        );

        let logged = String::from_utf8(logs.lock().unwrap().clone()).unwrap();
        assert!(
            logged.contains("still connecting to etcd"),
            "an outstanding dial must report itself: {logged:?}",
        );
        assert!(
            logged.contains(&addr.to_string()),
            "the line must name the endpoint it is waiting on: {logged:?}",
        );
    }
}
