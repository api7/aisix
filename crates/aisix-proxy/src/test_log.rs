//! Access-log capture for the tests that pin what a request's ONE line says.
//!
//! A streamed request's line is not written where its handler returns: it is
//! parked on the request's attribution cell and written by whichever
//! terminal emitter ends the request (AISIX-Cloud#1571). That makes "how
//! many lines did this request write, and what did the one line say" a
//! question only a log capture can answer — the value is not returned
//! anywhere a test could read it.
//!
//! [`three_stream_endings`] drives one streaming request through all three
//! endings a stream has, so a family cannot pass the delivered case and
//! silently lose the other two.

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::Request;
use axum::Router;

/// The access log's own `tracing` message. Counting its occurrences is how
/// a test asks "how many lines did this request write" without matching the
/// other events the same subscriber sees.
const ACCESS_LOG_MESSAGE: &str = "proxy request completed";

struct LogBuf(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for LogBuf {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl tracing_subscriber::fmt::MakeWriter<'_> for LogBuf {
    type Writer = LogBuf;
    fn make_writer(&self) -> Self::Writer {
        LogBuf(self.0.clone())
    }
}

/// A capturing subscriber installed on THIS thread, plus its buffer.
///
/// Thread-local rather than global: a `#[tokio::test]` runs its future on
/// the calling thread, which is where the handler, the body polls and the
/// guard drops all write from.
pub(crate) struct Capture {
    buf: Arc<Mutex<Vec<u8>>>,
    _guard: tracing::subscriber::DefaultGuard,
}

impl Capture {
    pub(crate) fn install() -> Self {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            let _ = tracing::subscriber::set_global_default(tracing_subscriber::registry());
        });
        let buf = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(LogBuf(buf.clone()))
            .finish();
        let _guard = tracing::subscriber::set_default(subscriber);
        Self { buf, _guard }
    }

    fn text(&self) -> String {
        String::from_utf8_lossy(&self.buf.lock().unwrap()).into_owned()
    }

    /// The request's one and only line. Panics with the captured text when
    /// there is not exactly one — which is the failure this whole module
    /// exists to catch.
    pub(crate) fn only(&self, what: &str) -> AccessLine {
        let text = self.text();
        let lines: Vec<&str> = text
            .lines()
            .filter(|l| l.contains(ACCESS_LOG_MESSAGE))
            .collect();
        assert_eq!(
            lines.len(),
            1,
            "{what}: expected exactly one access-log line, got {}:\n{text}",
            lines.len(),
        );
        AccessLine(lines[0].to_string())
    }
}

/// One rendered access-log line, read field by field.
pub(crate) struct AccessLine(String);

impl AccessLine {
    /// The value of `name=`, unquoted. `None` when the field is absent —
    /// which is meaningful here: this log omits `None` fields rather than
    /// rendering them empty, so an operator can filter on their presence.
    pub(crate) fn field(&self, name: &str) -> Option<String> {
        let needle = format!("{name}=");
        let mut from = 0usize;
        loop {
            let idx = self.0[from..].find(&needle)? + from;
            let starts_token = idx == 0 || self.0.as_bytes()[idx - 1] == b' ';
            let after = &self.0[idx + needle.len()..];
            if !starts_token {
                from = idx + needle.len();
                continue;
            }
            return Some(match after.strip_prefix('"') {
                Some(quoted) => {
                    let mut out = String::new();
                    let mut chars = quoted.chars();
                    while let Some(c) = chars.next() {
                        match c {
                            '\\' => out.extend(chars.next()),
                            '"' => break,
                            _ => out.push(c),
                        }
                    }
                    out
                }
                None => after.split(' ').next().unwrap_or_default().to_string(),
            });
        }
    }

    pub(crate) fn num(&self, name: &str) -> Option<u64> {
        self.field(name)?.parse().ok()
    }

    pub(crate) fn status(&self) -> u64 {
        self.num("status").expect("every line carries a status")
    }
}

/// The three ways a stream ends, each with the one line it wrote.
pub(crate) struct StreamEndings {
    /// Read to the end — slowly, with a pause after the first frame, so the
    /// caller's wait and the stream's length cannot be the same number by
    /// accident.
    pub delivered: AccessLine,
    /// One frame read, then the caller walks away.
    pub abandoned: AccessLine,
    /// The body dropped before its first poll.
    pub unread: AccessLine,
}

/// How long the delivered ending holds the stream open after its first
/// frame. Large enough that a line reporting the whole stream in
/// `latency_ms` cannot be mistaken for one reporting the first frame.
pub(crate) const SLOW_DRAIN: std::time::Duration = std::time::Duration::from_millis(200);

/// Drive `request` against `app` three times, once per stream ending, and
/// return the single access-log line each produced.
pub(crate) async fn three_stream_endings(
    app: Router,
    request: impl Fn() -> Request<Body>,
) -> StreamEndings {
    use futures::StreamExt as _;
    use tower::ServiceExt as _;

    let delivered = {
        let capture = Capture::install();
        let response = app.clone().oneshot(request()).await.unwrap();
        assert!(
            response.status().is_success(),
            "premise: the stream has to open, got {}",
            response.status(),
        );
        let mut body = response.into_body().into_data_stream();
        let first = body.next().await;
        assert!(first.is_some(), "premise: the stream delivered no frame");
        tokio::time::sleep(SLOW_DRAIN).await;
        while body.next().await.is_some() {}
        drop(body);
        capture.only("a delivered stream")
    };

    let abandoned = {
        let capture = Capture::install();
        let response = app.clone().oneshot(request()).await.unwrap();
        let mut body = response.into_body().into_data_stream();
        let first = body.next().await;
        assert!(first.is_some(), "premise: the stream delivered no frame");
        drop(body);
        capture.only("a stream abandoned mid-flight")
    };

    let unread = {
        let capture = Capture::install();
        let response = app.clone().oneshot(request()).await.unwrap();
        let (_parts, body) = response.into_parts();
        drop(body);
        capture.only("a stream dropped before its first poll")
    };

    StreamEndings {
        delivered,
        abandoned,
        unread,
    }
}

/// What every family's three endings must say, whatever it streams.
///
/// The two abandoned endings report the SAME outcome the request's terminal
/// usage event reports — `499` / `client_disconnected` — because the line
/// now rides that event out of one chokepoint instead of being written when
/// the response head was handed over, which is what used to log an
/// abandoned stream as a `200` (AISIX-Cloud#1571).
pub(crate) fn assert_one_line_per_ending(endings: &StreamEndings, path: &str, api_key_id: &str) {
    for (what, line, status) in [
        ("delivered", &endings.delivered, 200),
        (
            "abandoned",
            &endings.abandoned,
            u64::from(crate::CLIENT_CLOSED_REQUEST),
        ),
        (
            "unread",
            &endings.unread,
            u64::from(crate::CLIENT_CLOSED_REQUEST),
        ),
    ] {
        assert_eq!(line.status(), status, "{what}: wrong status on the line");
        assert_eq!(
            line.field("path").as_deref(),
            Some(path),
            "{what}: wrong path",
        );
        assert_eq!(
            line.field("api_key_id").as_deref(),
            Some(api_key_id),
            "{what}: the line must still name the caller",
        );
        let latency = line
            .num("latency_ms")
            .unwrap_or_else(|| panic!("{what}: no latency_ms"));
        let duration = line
            .num("duration_ms")
            .unwrap_or_else(|| panic!("{what}: no duration_ms"));
        assert!(
            duration >= latency,
            "{what}: duration_ms ({duration}) is what the request took and \
             latency_ms ({latency}) is the wait inside it — transposed",
        );
        let expected_class = (status != 200).then_some(crate::CLIENT_DISCONNECTED_KIND);
        assert_eq!(
            line.field("error_kind").as_deref(),
            expected_class,
            "{what}: the line and the usage event must name the same failure class",
        );
    }
}

/// The families whose streamed line reports time-to-first-token: the
/// delivered ending held the stream open for [`SLOW_DRAIN`] after its first
/// frame, so a line reporting the whole stream cannot pass.
pub(crate) fn assert_latency_is_time_to_first_token(endings: &StreamEndings) {
    let latency = endings.delivered.num("latency_ms").unwrap();
    let duration = endings.delivered.num("duration_ms").unwrap();
    assert!(
        duration >= latency + SLOW_DRAIN.as_millis() as u64 / 2,
        "latency_ms ({latency}) is supposed to be the wait to the FIRST token, \
         but it is within a rounding error of the whole stream ({duration})",
    );
}

/// A local SSE upstream that emits `frames` one at a time, pausing between
/// them.
///
/// wiremock answers with the whole body at once, which the byte-relaying
/// families (`/v1/responses`, the audio transcription relay) forward as a
/// SINGLE frame — so "read one frame, then walk away" would read the entire
/// stream and never model an abandoned one. Pausing between chunks is what
/// makes the three endings actually different.
pub(crate) async fn spawn_sse_upstream(frames: Vec<String>) -> String {
    use axum::response::IntoResponse;
    use futures::StreamExt as _;

    let frames = Arc::new(frames);
    let app = Router::new().fallback(axum::routing::any(move || {
        let frames = frames.clone();
        async move {
            let stream = futures::stream::iter(frames.as_ref().clone()).then(|frame| async move {
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                Ok::<_, std::convert::Infallible>(frame)
            });
            (
                [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
                Body::from_stream(stream),
            )
                .into_response()
        }
    }));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app.into_make_service())
            .await
            .unwrap();
    });
    format!("http://{addr}")
}
