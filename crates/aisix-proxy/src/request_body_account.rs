//! How much request-body data the proxy is holding for requests still in
//! flight: a large non-streaming request (inline images, documents) waiting
//! minutes on its upstream keeps its whole body in memory, and at
//! concurrency that is where the memory goes.
//!
//! Counted once, in the shared proxy layer every route passes through, as
//! the body is read: a request's bytes stay counted until its handler has
//! returned and the body stream is gone.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use http_body_util::BodyExt;

static BODIES: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

/// In-flight requests that have read body bytes, and those bytes.
pub fn in_flight_request_bodies() -> (usize, usize) {
    (
        BODIES.load(Ordering::Relaxed),
        BYTES.load(Ordering::Relaxed),
    )
}

/// One request's share of [`in_flight_request_bodies`], returned when the
/// last holder — the middleware or the body stream — lets go.
#[derive(Default)]
pub(crate) struct BodyAccount {
    bytes: AtomicUsize,
}

impl BodyAccount {
    fn add(&self, n: usize) {
        if n == 0 {
            return;
        }
        if self.bytes.fetch_add(n, Ordering::Relaxed) == 0 {
            BODIES.fetch_add(1, Ordering::Relaxed);
        }
        BYTES.fetch_add(n, Ordering::Relaxed);
    }
}

impl Drop for BodyAccount {
    fn drop(&mut self) {
        let bytes = *self.bytes.get_mut();
        if bytes > 0 {
            BODIES.fetch_sub(1, Ordering::Relaxed);
            BYTES.fetch_sub(bytes, Ordering::Relaxed);
        }
    }
}

/// Count `body` as it is read, into an account the caller keeps until the
/// request is done.
pub(crate) fn counted(body: axum::body::Body) -> (axum::body::Body, Arc<BodyAccount>) {
    let account = Arc::new(BodyAccount::default());
    let reader = Arc::clone(&account);
    let body = body.map_frame(move |frame| {
        if let Some(data) = frame.data_ref() {
            reader.add(data.len());
        }
        frame
    });
    (axum::body::Body::new(body), account)
}
