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
}

/// Canonical gRPC status codes, as sent on the wire
/// (<https://grpc.io/docs/guides/status-codes/>). Compared numerically so
/// this crate does not have to depend on — and keep in step with — the
/// exact `tonic` release `etcd-client` builds against.
const GRPC_UNKNOWN: i32 = 2;
const GRPC_DEADLINE_EXCEEDED: i32 = 4;
const GRPC_INTERNAL: i32 = 13;
const GRPC_UNAVAILABLE: i32 = 14;

impl ConnectError {
    /// The flattened cause, without this type's own framing — so a
    /// caller that re-reports it under its own wording does not stack
    /// two prefixes in front of what etcd actually said.
    pub fn detail(&self) -> &str {
        match self {
            Self::Unreachable(detail) | Self::Rejected(detail) => detail,
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
fn classify(err: &EtcdError) -> ConnectError {
    let detail = format_error_chain(err);
    match err {
        // The call never reached a server.
        EtcdError::TransportError(_) | EtcdError::IoError(_) => ConnectError::Unreachable(detail),
        EtcdError::GRpcStatus(status) => match status.code() as i32 {
            // `Unavailable` is both what tonic synthesises for every
            // failure to reach a server and what etcd itself answers
            // while it has no leader or is stopping. `DeadlineExceeded`
            // is a call that ran out of time. `Unknown` and `Internal`
            // are what a connection broken mid-call surfaces as. None of
            // them is an answer about the credentials, and all of them
            // can heal on their own.
            GRPC_UNAVAILABLE | GRPC_DEADLINE_EXCEEDED | GRPC_UNKNOWN | GRPC_INTERNAL => {
                ConnectError::Unreachable(detail)
            }
            // Everything else is etcd having evaluated the request and
            // said no: `InvalidArgument` for a wrong user or password,
            // `PermissionDenied` for a user without the rights,
            // `FailedPrecondition` for credentials sent to a cluster
            // that has authentication disabled.
            _ => ConnectError::Rejected(detail),
        },
        // Unparseable endpoints, an empty endpoint list, a bad header
        // value: configuration, not reachability.
        _ => ConnectError::Rejected(detail),
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
    connected: Mutex<Option<Client>>,
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
        }
    }

    /// Dial now, so a caller that wants to classify the failure itself —
    /// the boot path, which exits on [`ConnectError::Rejected`] — can do
    /// it before anything else is started. A successful dial is cached
    /// like any other.
    pub async fn connect_now(&self) -> Result<(), ConnectError> {
        self.client().await.map(|_| ())
    }

    /// The connection, dialling if this is the first call to get there.
    ///
    /// The lock is held across the dial on purpose: concurrent first
    /// calls wait for the one dial rather than opening a connection each.
    pub async fn client(&self) -> Result<Client, ConnectError> {
        let mut slot = self.connected.lock().await;
        if let Some(client) = slot.as_ref() {
            return Ok(client.clone());
        }
        let client = self.dial().await?;
        *slot = Some(client.clone());
        Ok(client)
    }

    /// KV sub-client for reads, under [`MAX_DECODING_MESSAGE_SIZE`].
    pub async fn kv(&self) -> Result<KvClient, ConnectError> {
        Ok(kv_client(&self.client().await?))
    }

    /// Watch sub-client, under [`MAX_DECODING_MESSAGE_SIZE`].
    pub async fn watch(&self) -> Result<WatchClient, ConnectError> {
        Ok(watch_client(&self.client().await?))
    }

    async fn dial(&self) -> Result<Client, ConnectError> {
        let connect = Client::connect(&self.endpoints, self.options.clone());
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

    #[test]
    fn a_local_io_failure_is_reachability_not_refusal() {
        let err = EtcdError::IoError(std::io::Error::other("connection reset by peer"));
        assert!(matches!(classify(&err), ConnectError::Unreachable(_)));
    }
}
