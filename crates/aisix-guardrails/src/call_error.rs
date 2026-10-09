//! What a failed remote guardrail call knows beyond its failure class.
//!
//! Every remote kind buckets a failure into a small enum (`IoError`,
//! `Timeout`, `MalformedResponse`, …) because those buckets are what the
//! bypass tags and metric labels are made of. The bucket alone does not tell
//! an operator *why*: `IoError` is a DNS failure, a refused connection, a TLS
//! error and a connection the peer reset, all at once. [`CallFailure`] carries
//! the rest, and each kind's "<provider> call failed" warn logs it as the same
//! three fields — `error`, `error_kind`, `elapsed_ms` — so the family cannot
//! drift.

use std::time::Instant;

/// A failure bucket plus the cause the warn logs next to it.
#[cfg_attr(
    not(any(
        feature = "azure-content-safety",
        feature = "aliyun-text-moderation",
        feature = "lakera",
        feature = "openai-moderation",
        feature = "presidio",
    )),
    allow(dead_code)
)]
#[derive(Debug)]
pub(crate) struct CallFailure<F> {
    pub(crate) failure: F,
    /// The underlying error with its whole `source()` chain; `None` when the
    /// failure is a bucketed status or a timer, which carry no error value.
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
            error: None,
            error_kind: None,
            elapsed_ms: 0,
        }
    }
}

/// Started right before a guardrail call is sent; every failure of that call
/// is built from it so the elapsed time is measured the same way everywhere.
#[cfg_attr(
    not(any(
        feature = "azure-content-safety",
        feature = "aliyun-text-moderation",
        feature = "lakera",
        feature = "openai-moderation",
        feature = "presidio",
    )),
    allow(dead_code)
)]
pub(crate) struct CallClock(Instant);

#[cfg_attr(
    not(any(
        feature = "azure-content-safety",
        feature = "aliyun-text-moderation",
        feature = "lakera",
        feature = "openai-moderation",
        feature = "presidio",
    )),
    allow(dead_code)
)]
impl CallClock {
    pub(crate) fn start() -> Self {
        Self(Instant::now())
    }

    fn elapsed_ms(&self) -> u64 {
        u64::try_from(self.0.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    /// A failure decided from what the provider answered (a status, a code)
    /// or from our own timer — there is no error value to report.
    pub(crate) fn fail<F>(&self, failure: F) -> CallFailure<F> {
        CallFailure {
            failure,
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
            error: Some(error_chain(err)),
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
pub(crate) fn error_chain(err: &(dyn std::error::Error + 'static)) -> String {
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
        for secret in secrets.iter().chain(&[PROMPT_MARKER]) {
            assert!(!logged.contains(secret), "`{secret}` leaked: {logged}");
        }
    }

    /// A provider that answers every call with `200` and a body that is not
    /// JSON, tagged with `request_id` in the `x-acs-request-id` header.
    pub(crate) async fn not_json_server(request_id: &str) -> wiremock::MockServer {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::any())
            .respond_with(
                wiremock::ResponseTemplate::new(200)
                    .set_body_string("<html>not json</html>")
                    .insert_header("x-acs-request-id", request_id),
            )
            .mount(&server)
            .await;
        server
    }

    /// The warn for a response that could not be decoded carries the decode
    /// error next to the failure bucket.
    pub(crate) fn assert_decode_logged(logged: &str, message: &str, failure: &str) {
        let line = failure_line(logged, message);
        assert!(
            line.contains(&format!("failure={failure}")),
            "bucket kept: {line}"
        );
        assert!(line.contains("error_kind=decode"), "phase logged: {line}");
        assert!(
            line.contains("error decoding response body"),
            "underlying cause logged: {line}"
        );
        assert!(line.contains("elapsed_ms="), "duration logged: {line}");
    }
}
