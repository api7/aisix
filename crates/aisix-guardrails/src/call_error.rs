//! What a failed remote guardrail call knows beyond its failure class.
//!
//! Every remote kind buckets a failure into a small enum (`IoError`,
//! `Timeout`, `MalformedResponse`, …) because those buckets are what the
//! bypass tags and metric labels are made of. The bucket alone does not tell
//! an operator *why*: `IoError` is a DNS failure, a refused connection, a TLS
//! error and a connection the peer reset, all at once. [`CallFailure`] carries
//! the rest, and each kind's "<provider> call failed" warn logs it as the same
//! fields — `http_status`, `error`, `error_kind`, `elapsed_ms` — so the family
//! cannot drift.
//!
//! [`CallClock`] also owns the call's deadline: `timeout_ms` bounds each HTTP
//! call end to end, so every await on its response — the send, an error
//! body, the JSON decode — goes through [`CallClock::within`]. A check that
//! makes several calls (chunks, Presidio's analyze then anonymize) gives each
//! its own `timeout_ms`, as before.

// Only the feature-gated HTTP kinds use the clock; `error_chain` and
// `error_kind` are used unconditionally.
#![cfg_attr(
    not(any(
        feature = "azure-content-safety",
        feature = "aliyun-text-moderation",
        feature = "lakera",
        feature = "openai-moderation",
        feature = "presidio",
    )),
    allow(dead_code)
)]

use std::time::Duration;

use tokio::time::Instant;

/// A failure bucket plus the cause the warn logs next to it.
#[derive(Debug)]
pub(crate) struct CallFailure<F> {
    pub(crate) failure: F,
    /// The response status, when a response arrived before the failure.
    pub(crate) http_status: Option<u16>,
    /// The underlying error with its whole `source()` chain, or only where it
    /// failed for a response that did not decode; `None` when the failure is
    /// a bucketed status or a timer, which carry no error value.
    pub(crate) error: Option<String>,
    /// For a reqwest error, which phase it failed in (see [`error_kind`]).
    pub(crate) error_kind: Option<&'static str>,
    /// Time from just before the request was sent to the failure.
    pub(crate) elapsed_ms: u64,
}

impl<F> CallFailure<F> {
    /// A failure with no cause attached, for tests that drive a kind's
    /// failure handling directly.
    #[cfg(test)]
    pub(crate) fn bare(failure: F) -> Self {
        Self {
            failure,
            http_status: None,
            error: None,
            error_kind: None,
            elapsed_ms: 0,
        }
    }
}

/// Started right before a guardrail call is sent; every failure of that call
/// is built from it so the elapsed time is measured the same way everywhere,
/// and every await of that call is bounded by its one deadline.
pub(crate) struct CallClock {
    started: Instant,
    deadline: Instant,
    http_status: Option<u16>,
}

impl CallClock {
    /// Start the clock for a call bounded by `timeout` end to end.
    pub(crate) fn start(timeout: Duration) -> Self {
        let started = Instant::now();
        Self {
            started,
            deadline: started + timeout,
            http_status: None,
        }
    }

    /// Run `fut` against the call's deadline. `Err` means the deadline
    /// passed, which every kind reports as its `Timeout` failure — whether
    /// the provider never answered or answered and then stalled the body.
    pub(crate) async fn within<T>(
        &self,
        fut: impl std::future::Future<Output = T>,
    ) -> Result<T, tokio::time::error::Elapsed> {
        tokio::time::timeout_at(self.deadline, fut).await
    }

    /// Record the status of the response that arrived, so every later
    /// failure of this call logs it.
    pub(crate) fn responded(&mut self, status: reqwest::StatusCode) {
        self.http_status = Some(status.as_u16());
    }

    fn elapsed_ms(&self) -> u64 {
        u64::try_from(self.started.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    /// A failure decided from what the provider answered (a status, a code)
    /// or from our own timer — there is no error value to report.
    pub(crate) fn fail<F>(&self, failure: F) -> CallFailure<F> {
        CallFailure {
            failure,
            http_status: self.http_status,
            error: None,
            error_kind: None,
            elapsed_ms: self.elapsed_ms(),
        }
    }

    /// A failure caused by a reqwest error: sending the request, or reading
    /// or decoding the response it returned.
    pub(crate) fn fail_with<F>(&self, failure: F, err: &reqwest::Error) -> CallFailure<F> {
        CallFailure {
            failure,
            http_status: self.http_status,
            error: Some(decode_failure(err).unwrap_or_else(|| error_chain(err))),
            error_kind: Some(error_kind(err)),
            elapsed_ms: self.elapsed_ms(),
        }
    }
}

/// `err` and every `source()` below it, joined with `": "`.
///
/// reqwest's own Display stops at "error sending request for url (…)"; the
/// actual cause — "dns error", "tcp connect error", a certificate failure,
/// "connection closed before message completed" — is only in the chain. A
/// level whose text is already part of what was written (some errors repeat
/// their source in their own message) is skipped.
///
/// The request URL is part of reqwest's text. No guardrail kind puts a
/// credential in it: every kind sends its key in a header or a signed form
/// body, so the URL is the configured endpoint plus a fixed path.
pub fn error_chain(err: &(dyn std::error::Error + 'static)) -> String {
    let mut out = err.to_string();
    let mut source = err.source();
    while let Some(s) = source {
        let msg = s.to_string();
        if !msg.is_empty() && !out.contains(&msg) {
            out.push_str(": ");
            out.push_str(&msg);
        }
        source = s.source();
    }
    out
}

/// A response that did not decode, described by serde's category and
/// position alone; `None` when `err` has no serde error beneath it.
///
/// serde's message quotes the offending value, and a moderation response
/// can carry the screened text (Presidio's anonymized output, a provider
/// echoing its input), which a guardrail log must never hold (#153).
/// Reading a body the peer cut short is also reported as `decode` but has
/// no serde source, so it keeps its full chain like any transport error.
fn decode_failure(err: &(dyn std::error::Error + 'static)) -> Option<String> {
    let mut source = Some(err);
    while let Some(e) = source {
        if let Some(serde) = e.downcast_ref::<serde_json::Error>() {
            let category = match serde.classify() {
                serde_json::error::Category::Io => "io",
                serde_json::error::Category::Syntax => "syntax",
                serde_json::error::Category::Data => "data",
                serde_json::error::Category::Eof => "eof",
            };
            return Some(format!(
                "response could not be decoded ({category}, line {} column {})",
                serde.line(),
                serde.column()
            ));
        }
        source = e.source();
    }
    None
}

/// The phase a reqwest error failed in. Checked in this order because the
/// predicates overlap: a connect timeout is both `is_connect` and
/// `is_timeout`, and is reported as `connect` (`elapsed_ms` tells the two
/// apart). On a response body, reqwest reports both a read that the peer
/// cut short and a parse failure as `decode`; `error` tells those apart.
pub(crate) fn error_kind(err: &reqwest::Error) -> &'static str {
    if err.is_connect() {
        "connect"
    } else if err.is_timeout() {
        "timeout"
    } else if err.is_request() {
        "request"
    } else if err.is_body() {
        "body"
    } else if err.is_decode() {
        "decode"
    } else {
        "other"
    }
}

/// Shared by every remote kind's test of its "call failed" warn.
#[cfg(test)]
pub(crate) mod testing {
    use std::sync::{Arc, Mutex};

    /// Text a test sends for moderation; it must never reach a failure log.
    pub(crate) const PROMPT_MARKER: &str = "promptmarkerneverlogged";

    /// An address nothing listens on: bound, then released, so a connect
    /// gets an immediate refusal.
    pub(crate) fn refused_endpoint() -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        format!("http://127.0.0.1:{port}")
    }

    #[derive(Clone)]
    struct BufWriter(Arc<Mutex<Vec<u8>>>);
    impl std::io::Write for BufWriter {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    impl tracing_subscriber::fmt::MakeWriter<'_> for BufWriter {
        type Writer = BufWriter;
        fn make_writer(&self) -> Self::Writer {
            self.clone()
        }
    }

    /// Run `f` with a capturing subscriber and return what it logged.
    pub(crate) async fn capture_logs<F, Fut>(f: F) -> String
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = ()>,
    {
        crate::keep_callsites_enabled();
        let _capture_guard = crate::TRACING_CAPTURE_LOCK.lock().await;
        let buf = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_max_level(tracing::Level::DEBUG)
            .with_writer(BufWriter(buf.clone()))
            .finish();
        {
            let _guard = tracing::subscriber::set_default(subscriber);
            f().await;
        }
        let bytes = buf.lock().unwrap().clone();
        String::from_utf8(bytes).unwrap()
    }

    /// The `message` line in `logged`, which must exist.
    pub(crate) fn failure_line<'a>(logged: &'a str, message: &str) -> &'a str {
        logged
            .lines()
            .find(|l| l.contains(message))
            .unwrap_or_else(|| panic!("no `{message}` line; got: {logged}"))
    }

    /// The warn for a refused connection carries the cause, its phase and a
    /// duration on the line that names the failure bucket — and none of the
    /// row's credentials or the moderated text.
    pub(crate) fn assert_refusal_logged(logged: &str, message: &str, secrets: &[&str]) {
        let line = failure_line(logged, message);
        assert!(line.contains("failure=IoError"), "bucket kept: {line}");
        assert!(line.contains("error_kind=connect"), "phase logged: {line}");
        assert!(
            line.to_lowercase().contains("connection refused"),
            "underlying cause logged: {line}"
        );
        assert!(line.contains("elapsed_ms="), "duration logged: {line}");
        assert!(
            !line.contains("http_status="),
            "no response, no status: {line}"
        );
        for secret in secrets.iter().chain(&[PROMPT_MARKER]) {
            assert!(!logged.contains(secret), "`{secret}` leaked: {logged}");
        }
    }

    /// A provider that answers every call with `200` and a body no kind
    /// can decode: a bare JSON string holding [`PROMPT_MARKER`], the way a
    /// provider that echoes its input would. Tagged with `request_id` in
    /// the `x-acs-request-id` header.
    pub(crate) async fn undecodable_server(request_id: &str) -> wiremock::MockServer {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::any())
            .respond_with(
                wiremock::ResponseTemplate::new(200)
                    .set_body_string(format!("\"{PROMPT_MARKER}\""))
                    .insert_header("x-acs-request-id", request_id),
            )
            .mount(&server)
            .await;
        server
    }

    /// The warn for a response that could not be decoded says where it
    /// failed to decode, next to the failure bucket — and nothing of what
    /// the response said, which quoted the moderated text.
    pub(crate) fn assert_decode_logged(logged: &str, message: &str, failure: &str) {
        let line = failure_line(logged, message);
        assert!(
            line.contains(&format!("failure={failure}")),
            "bucket kept: {line}"
        );
        assert!(line.contains("error_kind=decode"), "phase logged: {line}");
        assert!(
            line.contains("response could not be decoded (data, line 1 column"),
            "decode failure and its position logged: {line}"
        );
        assert!(line.contains("elapsed_ms="), "duration logged: {line}");
        assert!(
            !logged.contains(PROMPT_MARKER),
            "the response's echo of the moderated text leaked: {logged}"
        );
    }

    /// A provider that answers every call with `status` and an empty JSON
    /// object.
    pub(crate) async fn status_server(status: u16) -> wiremock::MockServer {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::any())
            .respond_with(wiremock::ResponseTemplate::new(status).set_body_string("{}"))
            .mount(&server)
            .await;
        server
    }

    /// The warn for a call that got a response carries its status.
    pub(crate) fn assert_status_logged(logged: &str, message: &str, status: u16) {
        let line = failure_line(logged, message);
        assert!(
            line.contains(&format!("http_status={status}")),
            "status logged: {line}"
        );
        assert!(line.contains("elapsed_ms="), "duration logged: {line}");
    }

    /// A provider that sends `status` and its headers, promises a body,
    /// sends one byte of it and then never another.
    pub(crate) async fn stalled_body_server(status: u16) -> String {
        use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let mut held = Vec::new();
            while let Ok((mut sock, _)) = listener.accept().await {
                let mut buf = [0u8; 16 * 1024];
                let _ = sock.read(&mut buf).await;
                let head = format!(
                    "HTTP/1.1 {status} X\r\ncontent-type: application/json\r\n\
                     content-length: 1000\r\n\r\n{{"
                );
                let _ = sock.write_all(head.as_bytes()).await;
                held.push(sock);
            }
        });
        format!("http://{addr}")
    }

    /// A body that stalls past `timeout_ms` ends the call as the kind's
    /// `Timeout`, with the status of the response that did arrive.
    pub(crate) fn assert_stall_logged(logged: &str, message: &str, status: u16) {
        let line = failure_line(logged, message);
        assert!(line.contains("failure=Timeout"), "bucket: {line}");
        assert!(
            line.contains(&format!("http_status={status}")),
            "status logged: {line}"
        );
    }
}
