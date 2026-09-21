use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use tokio::sync::{mpsc, Semaphore};

/// Rendered pieces in flight between the renderer and the response.
///
/// One. The renderer is what should wait when the response is slow, not
/// a queue: a scrape's peak memory is this many pieces, whatever the
/// exposition's size.
const PIECES_IN_FLIGHT: usize = 1;

/// How long one piece may wait for the response to take it.
///
/// A reader can stop reading without closing — a zero window, a wedged
/// sidecar, a half-open socket — and nothing below this notices: the
/// listener's only timeout covers the gap BETWEEN requests, not a
/// response being written. Without a bound the render parks forever
/// holding the gate below, and every later scrape is answered with a
/// `200` whose body never arrives. A reader this slow has lost its own
/// scrape either way; what must not be lost is everyone else's.
#[cfg(not(test))]
const PIECE_TIMEOUT: Duration = Duration::from_secs(30);
/// Shortened so a case can drive a reader that stops without waiting one
/// out; what the case pins is the abandonment, not the number.
#[cfg(test)]
const PIECE_TIMEOUT: Duration = Duration::from_millis(200);

/// How long to leave a full channel before looking again. Small against
/// [`PIECE_TIMEOUT`], and only ever reached when the reader is behind.
const PIECE_RETRY: Duration = Duration::from_millis(10);

pub(crate) struct Scrape {
    /// One render at a time. Walking and formatting every series is the
    /// most expensive thing this process does off the request path, and
    /// two scrapers whose requests overlap must not both pay for it at
    /// once. Each still gets its own current exposition rather than a
    /// copy of the other's, which a streamed body has no way to share.
    gate: Arc<Semaphore>,
}

impl Default for Scrape {
    fn default() -> Self {
        Self {
            gate: Arc::new(Semaphore::new(1)),
        }
    }
}

impl Scrape {
    /// Start a render and hand back the pieces of the response body.
    ///
    /// `render` is called with a sink it must feed in order; the sink
    /// returns `false` once nothing is reading any more. It runs on a
    /// blocking thread at background priority — the exposition is
    /// hundreds of megabytes of text at this cardinality on a core a
    /// request worker also wants, and it is a scrape: late is fine.
    ///
    /// The render is paced by the response, so a reader that stops
    /// costs a stalled render rather than a growing buffer, and a
    /// dropped response ends it.
    pub(crate) fn stream(
        &self,
        render: impl FnOnce(&mut dyn FnMut(String) -> bool) + Send + 'static,
    ) -> mpsc::Receiver<Result<Bytes, std::io::Error>> {
        let (sender, receiver) = mpsc::channel(PIECES_IN_FLIGHT);
        let gate = Arc::clone(&self.gate);
        // The producer outlives the handler that started it, so a
        // cancelled client releases the gate through the sink below
        // rather than by abandoning a permit.
        tokio::spawn(async move {
            // Nothing closes the semaphore; the error arm is unreachable.
            let Ok(_permit) = gate.acquire_owned().await else {
                return;
            };
            let failed = {
                let sender = sender.clone();
                tokio::task::spawn_blocking(move || {
                    aisix_core::run_demoted("metrics-render", || {
                        render(&mut |piece| hand_over(&sender, Bytes::from(piece)))
                    })
                })
                .await
            };
            if let Err(error) = failed {
                tracing::error!(%error, "metrics render task failed");
                // The headers left long ago, so the only way left to say
                // the exposition is incomplete is to end the body
                // abnormally. Ending it cleanly would hand the scraper a
                // truncated exposition as a successful scrape, and every
                // series past the failure would read as gone rather than
                // unknown.
                let _ = sender
                    .send(Err(std::io::Error::other("metrics render failed")))
                    .await;
            }
        });
        receiver
    }
}

/// Give one piece to the response, waiting for it to take the previous
/// one. `false` means stop rendering: the response is gone, or it has
/// stopped taking pieces for longer than any live scrape would.
///
/// A plain blocking send would be simpler and is what a render wants —
/// but it waits on the reader without limit, and this render holds a
/// process-wide gate while it does.
fn hand_over(sender: &mpsc::Sender<Result<Bytes, std::io::Error>>, piece: Bytes) -> bool {
    let deadline = std::time::Instant::now() + PIECE_TIMEOUT;
    let mut piece = Ok(piece);
    loop {
        match sender.try_send(piece) {
            Ok(()) => return true,
            Err(mpsc::error::TrySendError::Closed(_)) => return false,
            Err(mpsc::error::TrySendError::Full(unsent)) => {
                if std::time::Instant::now() >= deadline {
                    tracing::warn!(
                        "a metrics scrape stopped reading; abandoning its render so later \
                         scrapes are not blocked behind it",
                    );
                    return false;
                }
                piece = unsent;
                std::thread::sleep(PIECE_RETRY);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    async fn collect(mut receiver: mpsc::Receiver<Result<Bytes, std::io::Error>>) -> String {
        let mut out = String::new();
        while let Some(piece) = receiver.recv().await {
            out.push_str(std::str::from_utf8(&piece.unwrap()).unwrap());
        }
        out
    }

    #[tokio::test]
    async fn the_body_is_the_pieces_in_order_and_the_render_runs_off_the_runtime() {
        let scrape = Scrape::default();
        let runtime_thread = std::thread::current().id();
        let body = scrape.stream(move |emit| {
            assert_ne!(std::thread::current().id(), runtime_thread);
            for piece in ["first\n", "second\n", "third\n"] {
                assert!(emit(piece.to_owned()));
            }
        });
        assert_eq!(collect(body).await, "first\nsecond\nthird\n");
    }

    /// A reader that goes away must stop the render rather than let it
    /// keep formatting series into a channel nobody drains.
    #[tokio::test]
    async fn a_dropped_response_stops_the_render_and_frees_the_gate() {
        let scrape = Scrape::default();
        let (rendered, count) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let pieces = Arc::clone(&rendered);
        let body = scrape.stream(move |emit| {
            // More pieces than the channel can hold, so the second one
            // waits for a reader that is gone.
            for _ in 0..64 {
                pieces.fetch_add(1, Ordering::SeqCst);
                if !emit("series 1\n".to_owned()) {
                    return;
                }
            }
        });
        drop(body);
        // The next scrape needs the gate the abandoned one held.
        let second = Arc::clone(&count);
        let body = scrape.stream(move |emit| {
            second.fetch_add(1, Ordering::SeqCst);
            assert!(emit("series 2\n".to_owned()));
        });
        assert_eq!(collect(body).await, "series 2\n");
        assert_eq!(count.load(Ordering::SeqCst), 1);
        assert!(
            rendered.load(Ordering::SeqCst) < 64,
            "the abandoned render must stop, not run to completion",
        );
    }

    /// A reader that stops reading WITHOUT closing — a zero window, a
    /// wedged sidecar — used to be indistinguishable from a slow one,
    /// and the render waits on it while holding the only gate. It must
    /// give up so the next scrape can run.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_reader_that_stops_without_closing_does_not_hold_the_gate() {
        let scrape = Scrape::default();
        let stalled = scrape.stream(|emit| {
            for _ in 0..64 {
                if !emit("series 1\n".to_owned()) {
                    return;
                }
            }
        });

        // Held, never read from: the channel fills and stays full.
        let second = scrape.stream(|emit| {
            assert!(emit("series 2\n".to_owned()));
        });
        let body = tokio::time::timeout(Duration::from_secs(10), collect(second))
            .await
            .expect("the second scrape must not wait on the first reader");
        assert_eq!(body, "series 2\n");
        drop(stalled);
    }

    /// A render that dies has already had its headers sent, so the only
    /// way left to say the exposition is incomplete is to end the body
    /// abnormally. Ending it cleanly would pass a truncated exposition
    /// off as a whole one.
    #[tokio::test]
    async fn a_failed_render_ends_the_body_with_an_error() {
        let scrape = Scrape::default();
        let mut body = scrape.stream(|emit| {
            assert!(emit("partial\n".to_owned()));
            panic!("render died");
        });
        assert_eq!(body.recv().await.unwrap().unwrap(), "partial\n");
        assert!(
            body.recv().await.expect("a piece, not the end").is_err(),
            "the body must end abnormally, not cleanly",
        );
        assert!(body.recv().await.is_none());

        // And the gate is free for the next one.
        assert_eq!(
            collect(scrape.stream(|emit| {
                assert!(emit("recovered\n".to_owned()));
            }))
            .await,
            "recovered\n",
        );
    }

    /// Two overlapping scrapes each get their own exposition, and the
    /// second's render does not start while the first is still going.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn overlapping_scrapes_render_one_at_a_time() {
        let scrape = Scrape::default();
        let live = Arc::new(AtomicUsize::new(0));
        let overlapped = Arc::new(AtomicUsize::new(0));
        let bodies: Vec<_> = (0..4)
            .map(|i| {
                let (live, overlapped) = (Arc::clone(&live), Arc::clone(&overlapped));
                scrape.stream(move |emit| {
                    if live.fetch_add(1, Ordering::SeqCst) != 0 {
                        overlapped.fetch_add(1, Ordering::SeqCst);
                    }
                    std::thread::sleep(std::time::Duration::from_millis(20));
                    live.fetch_sub(1, Ordering::SeqCst);
                    assert!(emit(format!("scrape {i}\n")));
                })
            })
            .collect();
        for (i, body) in bodies.into_iter().enumerate() {
            assert_eq!(collect(body).await, format!("scrape {i}\n"));
        }
        assert_eq!(overlapped.load(Ordering::SeqCst), 0);
    }
}
