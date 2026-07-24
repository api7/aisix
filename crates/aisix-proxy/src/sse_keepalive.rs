//! Downstream SSE heartbeat for streaming responses (AISIX-Cloud#1126).
//!
//! A model that takes a long time to produce its first token leaves the
//! response connection silent, and a proxy between the client and the
//! gateway can decide the connection is idle and drop it. Emitting an SSE
//! comment on that silence keeps bytes moving without changing what the
//! client parses.
//!
//! The interval is a deployment property — it depends on what sits in
//! front of the gateway, not on which model was asked — so it is process-
//! wide, installed once at boot by [`init`], the same shape
//! `aisix_gateway::upstream_http` uses for the outbound connection layer.

use std::sync::OnceLock;
use std::time::Duration;

use bytes::Bytes;
use futures::{Stream, StreamExt};

/// An SSE comment line. Valid mid-stream, ignored by every conforming
/// client, and the same bytes axum's `Sse` keep-alive writes by default,
/// so a bridged and a passthrough stream heartbeat identically.
const HEARTBEAT: &[u8] = b":\n\n";

static INTERVAL: OnceLock<Option<Duration>> = OnceLock::new();

/// Install the process-wide heartbeat interval. `None` disables it.
/// Called once during boot, before any streaming response is built; later
/// calls are ignored.
pub fn init(interval: Option<Duration>) {
    let _ = INTERVAL.set(interval);
}

/// The active interval, defaulting to `DownstreamConfig`'s 15s when
/// [`init`] was never called (unit tests, embedded uses).
pub(crate) fn interval() -> Option<Duration> {
    *INTERVAL.get_or_init(|| Some(Duration::from_secs(15)))
}

/// Wrap an SSE byte stream so a silence longer than `interval` emits a
/// heartbeat comment instead of nothing. `None` returns a pass-through.
///
/// Only for streams the client parses as `text/event-stream`; injecting a
/// comment into an opaque binary passthrough (audio, images) would corrupt
/// it.
///
/// Unlike [`crate::stream_timeout::with_read_timeout_bytes`], which ends
/// the stream when the upstream goes quiet for too long, this keeps
/// waiting — the two compose, with the read timeout deciding when a silent
/// upstream has failed and this one keeping the connection warm until then.
pub(crate) fn with_heartbeat<S, E>(
    upstream: S,
    interval: Option<Duration>,
) -> impl Stream<Item = Result<Bytes, E>> + Send
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    E: Send + 'static,
{
    async_stream::stream! {
        let mut upstream = std::pin::pin!(upstream);
        loop {
            match interval {
                None => match upstream.next().await {
                    Some(item) => yield item,
                    None => break,
                },
                Some(d) => match tokio::time::timeout(d, upstream.next()).await {
                    Ok(Some(item)) => yield item,
                    Ok(None) => break,
                    Err(_) => yield Ok(Bytes::from_static(HEARTBEAT)),
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The point of the module: an upstream that says nothing for longer
    /// than the interval still puts bytes on the wire.
    #[tokio::test(start_paused = true)]
    async fn silence_emits_heartbeats_until_the_chunk_arrives() {
        let upstream = async_stream::stream! {
            tokio::time::sleep(Duration::from_millis(250)).await;
            yield Ok::<_, std::io::Error>(Bytes::from_static(b"data: hi\n\n"));
        };
        let out: Vec<_> = with_heartbeat(upstream, Some(Duration::from_millis(100)))
            .collect()
            .await;
        let frames: Vec<Bytes> = out.into_iter().map(|r| r.unwrap()).collect();
        assert_eq!(
            frames,
            vec![
                Bytes::from_static(HEARTBEAT),
                Bytes::from_static(HEARTBEAT),
                Bytes::from_static(b"data: hi\n\n"),
            ],
        );
    }

    /// A stream that keeps producing must not have anything injected —
    /// the heartbeat exists for silence only.
    #[tokio::test(start_paused = true)]
    async fn a_busy_stream_is_forwarded_verbatim() {
        let upstream = async_stream::stream! {
            for i in 0..3u8 {
                tokio::time::sleep(Duration::from_millis(10)).await;
                yield Ok::<_, std::io::Error>(Bytes::from(vec![i]));
            }
        };
        let out: Vec<_> = with_heartbeat(upstream, Some(Duration::from_millis(100)))
            .collect()
            .await;
        let frames: Vec<Vec<u8>> = out.into_iter().map(|r| r.unwrap().to_vec()).collect();
        assert_eq!(frames, vec![vec![0], vec![1], vec![2]]);
    }

    #[tokio::test(start_paused = true)]
    async fn disabled_interval_is_a_pass_through() {
        let upstream = async_stream::stream! {
            tokio::time::sleep(Duration::from_secs(60)).await;
            yield Ok::<_, std::io::Error>(Bytes::from_static(b"data: hi\n\n"));
        };
        let out: Vec<_> = with_heartbeat(upstream, None).collect().await;
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].as_ref().unwrap(),
            &Bytes::from_static(b"data: hi\n\n")
        );
    }

    /// The heartbeat must not swallow the stream's terminal error — that
    /// is what tells the client the response ended badly.
    #[tokio::test(start_paused = true)]
    async fn a_terminal_error_still_reaches_the_client() {
        let upstream = async_stream::stream! {
            tokio::time::sleep(Duration::from_millis(150)).await;
            yield Err(std::io::Error::other("upstream died"));
        };
        let out: Vec<_> = with_heartbeat(upstream, Some(Duration::from_millis(100)))
            .collect()
            .await;
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].as_ref().unwrap(), &Bytes::from_static(HEARTBEAT));
        assert!(out[1].is_err());
    }
}
