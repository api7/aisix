//! Per-chunk read-timeout combinators for streaming upstreams (#554).
//!
//! Per-chunk streaming read-timeout: the deadline bounds the
//! wait for EACH chunk — the first one and every inter-chunk gap — and
//! resets after each successful read. A *first-chunk* timeout lets the
//! caller fail over before any bytes reach the client (issue AC2); a
//! *mid-stream* timeout terminates the stream like any other upstream
//! error, because once the `200` is committed a clean fallback is no
//! longer possible.
//!
//! Two flavours:
//! - [`with_read_timeout`] for the typed [`ChatChunkStream`] path
//!   (`/v1/chat/completions`, cross-provider `/v1/messages`): a read
//!   timeout surfaces as [`BridgeError::Timeout`], which the SSE pump
//!   already renders as an error frame.
//! - [`with_read_timeout_bytes`] for the raw byte passthroughs
//!   (`/v1/responses`, native-Anthropic `/v1/messages`): a read timeout
//!   simply ends the forwarded byte stream (the client sees a truncated
//!   response); there is no in-band error frame to inject into an opaque
//!   passthrough.
//!
//! [`send_with_deadline`] bounds the connect phase of a raw passthrough so
//! a slow upstream that never returns response headers also fails over.

use std::time::{Duration, Instant};

use aisix_gateway::{BridgeError, ChatChunkStream};
use bytes::Bytes;
use futures::{Stream, StreamExt};

/// Wrap a [`ChatChunkStream`] so each `next()` is bounded by `per_chunk`.
/// On elapse, yield a single [`BridgeError::Timeout`] and end the stream.
/// `None` returns the stream unchanged (zero overhead on the hot path).
pub(crate) fn with_read_timeout(
    upstream: ChatChunkStream,
    per_chunk: Option<Duration>,
) -> ChatChunkStream {
    let Some(d) = per_chunk else {
        return upstream;
    };
    Box::pin(async_stream::stream! {
        // `ChatChunkStream` is a `Pin<Box<..>>`, hence `Unpin`; a plain
        // `mut` binding is enough to poll it via `StreamExt::next`.
        let mut upstream = upstream;
        loop {
            match tokio::time::timeout(d, upstream.next()).await {
                Ok(Some(item)) => yield item,
                Ok(None) => break,
                Err(_) => {
                    yield Err(BridgeError::Timeout {
                        elapsed_ms: d.as_millis() as u64,
                        cause: String::new(),
                    });
                    break;
                }
            }
        }
    })
}

/// Wrap a raw byte stream (`reqwest::Response::bytes_stream()`) so each
/// `next()` is bounded by `per_chunk`. On elapse, end the stream (the
/// forwarded client response is truncated). `None` returns a pass-through.
pub(crate) fn with_read_timeout_bytes<S>(
    upstream: S,
    per_chunk: Option<Duration>,
) -> impl Stream<Item = reqwest::Result<Bytes>> + Send
where
    S: Stream<Item = reqwest::Result<Bytes>> + Send + 'static,
{
    async_stream::stream! {
        let mut upstream = std::pin::pin!(upstream);
        loop {
            match per_chunk {
                Some(d) => match tokio::time::timeout(d, upstream.next()).await {
                    Ok(Some(item)) => yield item,
                    Ok(None) => break,
                    // Read timeout mid-passthrough: truncate the forwarded
                    // stream. We can't inject a typed error into opaque bytes.
                    Err(_) => break,
                },
                None => match upstream.next().await {
                    Some(item) => yield item,
                    None => break,
                },
            }
        }
    }
}

/// Send a raw-passthrough request, optionally bounding the connect phase
/// (everything up to and including response headers) by `deadline`. Maps
/// both reqwest's own timeout and the outer deadline to
/// [`BridgeError::Timeout`] so a slow connect fails over like the
/// Bridge-trait path. `started` anchors the reported elapsed time.
pub(crate) async fn send_with_deadline(
    req: reqwest::RequestBuilder,
    deadline: Option<Duration>,
    started: Instant,
) -> Result<reqwest::Response, BridgeError> {
    match deadline {
        Some(d) => match tokio::time::timeout(d, req.send()).await {
            Ok(res) => res.map_err(|e| crate::dispatch::reqwest_error_to_bridge(&e, started)),
            Err(_) => Err(BridgeError::Timeout {
                elapsed_ms: started.elapsed().as_millis() as u64,
                cause: String::new(),
            }),
        },
        None => req
            .send()
            .await
            .map_err(|e| crate::dispatch::reqwest_error_to_bridge(&e, started)),
    }
}
