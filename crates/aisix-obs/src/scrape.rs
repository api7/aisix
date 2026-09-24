use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use tokio::sync::{mpsc, Semaphore};

/// Rendered pieces in flight between the renderer and the response.
///
/// Two, so the renderer can be formatting the next piece while the
/// socket drains the previous one. One made them strictly alternate,
/// which costs the whole render the socket's latency once per piece. A
/// scrape's peak memory is still this many pieces plus the one being
/// filled, whatever the exposition's size.
const PIECES_IN_FLIGHT: usize = 2;

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
        // One slot beyond the pieces in flight, held in reserve below so
        // a render that dies can always say so.
        let (sender, receiver) = mpsc::channel(PIECES_IN_FLIGHT + 1);
        let gate = Arc::clone(&self.gate);
        // Taken here, where there is a runtime to take it from: the
        // render runs on a thread of its own and needs it to wait for
        // the response without polling for capacity.
        let runtime = tokio::runtime::Handle::current();
        // The producer outlives the handler that started it, so a
        // cancelled client releases the gate through the sink below
        // rather than by abandoning a permit.
        tokio::spawn(async move {
            // Nothing closes the semaphore; the error arm is unreachable.
            let Ok(_permit) = gate.acquire_owned().await else {
                return;
            };
            // Taken before the render can fill the channel. Saying the
            // exposition is incomplete must not itself have to wait for
            // a reader — that is the wait this task holds the gate
            // through, and a reader can stop without closing.
            let Ok(terminal) = sender.clone().reserve_owned().await else {
                return;
            };
            let failed = {
                let sender = sender.clone();
                tokio::task::spawn_blocking(move || {
                    aisix_core::run_demoted("metrics-render", || {
                        render(&mut |piece| hand_over(&runtime, &sender, Bytes::from(piece)))
                    })
                })
                .await
            };
            match failed {
                Ok(()) => drop(terminal),
                Err(error) => {
                    tracing::error!(%error, "metrics render task failed");
                    // The headers left long ago, so the only way left to
                    // say the exposition is incomplete is to end the body
                    // abnormally. Ending it cleanly would hand the
                    // scraper a truncated exposition as a successful
                    // scrape, and every series past the failure would
                    // read as gone rather than unknown.
                    terminal.send(Err(std::io::Error::other("metrics render failed")));
                }
            }
        });
        receiver
    }
}

/// Give one piece to the response, waiting for it to take an earlier
/// one. `false` means stop rendering: the response is gone, or it has
/// stopped taking pieces for longer than any live scrape would.
///
/// The wait is a real wait, not a poll. Looking again on a timer costs
/// every full channel the whole interval even when the response drained
/// it immediately, and at this exposition's size that is thousands of
/// intervals — measured at 13.6x the time per byte against a reader
/// that was never actually behind.
fn hand_over(
    runtime: &tokio::runtime::Handle,
    sender: &mpsc::Sender<Result<Bytes, std::io::Error>>,
    piece: Bytes,
) -> bool {
    // The timeout is built INSIDE the block, where the runtime context
    // exists: a timer registers when it is created, not when it is
    // awaited.
    match runtime
        .block_on(async { tokio::time::timeout(PIECE_TIMEOUT, sender.send(Ok(piece))).await })
    {
        Ok(Ok(())) => true,
        // The response is gone; there is nothing left to render for.
        Ok(Err(_)) => false,
        Err(_) => {
            tracing::warn!(
                "a metrics scrape stopped reading; abandoning its render so later \
                 scrapes are not blocked behind it",
            );
            false
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

    /// The handover must WAIT for the response, not look again on a
    /// timer.
    ///
    /// A reader that is keeping up never fills the channel, so it cannot
    /// tell the two apart — what does is a reader that is merely
    /// *slower* than the renderer, which is every real one: the socket
    /// drains 256 KiB more slowly than the recorder formats it. A poll
    /// then costs each piece its whole interval instead of the reader's
    /// actual latency, and an exposition this size is thousands of
    /// pieces. The first version of this shipped a 10 ms retry and took
    /// 13.6x as long per byte on a warm registry.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_reader_only_costs_the_render_its_own_latency() {
        const PIECES: usize = 1200;
        /// Well under the retry interval this replaced, so the interval
        /// rather than this is what a poll would spend.
        const READER_LATENCY: Duration = Duration::from_millis(1);

        let scrape = Scrape::default();
        let mut body = scrape.stream(|emit| {
            for _ in 0..PIECES {
                assert!(emit("a series line\n".to_owned()));
            }
        });
        let started = std::time::Instant::now();
        let mut taken = 0;
        while let Some(piece) = body.recv().await {
            piece.expect("a piece, not an error");
            taken += 1;
            tokio::time::sleep(READER_LATENCY).await;
        }
        let elapsed = started.elapsed();
        assert_eq!(taken, PIECES);
        // Against the reader's OWN cost rather than a wall-clock number,
        // so a slow machine moves both sides together. Measured here:
        // ~2.1x the floor waiting on the reader, ~5x polling for it.
        let floor = READER_LATENCY * PIECES as u32;
        assert!(
            elapsed < floor * 3,
            "handing over {PIECES} pieces to a reader that takes each in \
             {READER_LATENCY:?} took {elapsed:?}, against a floor of {floor:?} — \
             the renderer is waiting on something other than the reader",
        );
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

    /// Saying the exposition is incomplete must not have to wait for a
    /// reader either. A render that dies with the channel already full,
    /// against a receiver that is open but not being read, would
    /// otherwise wait there holding the gate — the same way the pieces
    /// themselves used to.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_render_that_dies_on_a_full_channel_still_frees_the_gate() {
        let scrape = Scrape::default();
        // Fills every slot, then dies. The receiver below is held and
        // never read from.
        let stalled = scrape.stream(|emit| {
            for _ in 0..PIECES_IN_FLIGHT {
                assert!(emit("queued\n".to_owned()));
            }
            panic!("render died with the channel full");
        });

        let second = scrape.stream(|emit| {
            assert!(emit("the next scrape\n".to_owned()));
        });
        let body = tokio::time::timeout(Duration::from_secs(10), collect(second))
            .await
            .expect("the gate must not be held by the dead render");
        assert_eq!(body, "the next scrape\n");
        drop(stalled);
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
