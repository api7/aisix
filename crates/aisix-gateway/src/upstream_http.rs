//! Connection-layer settings and error rendering for upstream HTTP calls.
//!
//! Every provider bridge talks to its upstream through a `reqwest::Client`.
//! Two things live here because they must be identical across all of them:
//!
//! - [`client_builder`] — a `ClientBuilder` pre-loaded with the process-wide
//!   connection settings ([`UpstreamHttpConfig`]). reqwest's own defaults
//!   leave TCP keepalive off, impose no connect timeout, and keep idle
//!   pooled connections for 90s — longer than the idle timeout of a typical
//!   LB/NAT/proxy hop in front of a provider, so a pooled connection can be
//!   reaped upstream and still be handed out here, failing the next request.
//! - [`transport_error_message`] — renders a `reqwest::Error` with its full
//!   `source()` chain. The top-level `Display` is only ever
//!   "error sending request for url (…)", which is the same string for a DNS
//!   failure, a TCP reset, a TLS handshake error, and a stale pooled
//!   connection. The chain is what tells them apart.

use std::sync::OnceLock;
use std::time::Duration;

/// Suffixes marking a query parameter whose value is a credential and must
/// be redacted out of logged URLs. Vertex/Gemini accept `?key=` and
/// `?access_token=`, and an operator can put either directly in a
/// ProviderKey `api_base`, so a URL echoed into a log line can carry live
/// credentials.
///
/// Matched as a **suffix** of the parameter name after lowercasing and
/// stripping `-`/`_`, which covers the vendor-prefixed and punctuation
/// variants without enumerating them: `api-key` / `api_key` / `apiKey` all
/// end in `key`; `client_secret` in `secret`; SigV4's `X-Amz-Signature`,
/// `X-Amz-Security-Token`, and `X-Amz-Credential` in `signature`, `token`,
/// and `credential`. Over-redacting an unrelated parameter costs a little
/// diagnostic detail; under-redacting leaks a live key into a log store.
const SENSITIVE_PARAM_SUFFIXES: &[&str] = &[
    "key",
    "token",
    "secret",
    "password",
    "credential",
    "sig",
    "signature",
];

/// Whether a query parameter's value is credential material.
fn is_sensitive_param(name: &str) -> bool {
    let normalized: String = name
        .chars()
        .filter(|c| *c != '-' && *c != '_')
        .flat_map(|c| c.to_lowercase())
        .collect();
    SENSITIVE_PARAM_SUFFIXES
        .iter()
        .any(|s| normalized.ends_with(s))
}

/// Cap on how many `source()` links are walked. Real reqwest/hyper chains
/// are 3-5 deep; the bound just keeps a pathological cycle from running away.
const MAX_SOURCE_DEPTH: usize = 8;

/// Connection-layer settings shared by every upstream provider client.
///
/// Defaults follow the same reasoning LiteLLM applies to its own upstream
/// pool: bound the connect phase, keep the kernel probing so a NAT/LB hop
/// can't silently reap a connection while a slow model is still thinking,
/// and expire pooled connections well before a typical upstream idle
/// timeout would.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpstreamHttpConfig {
    /// Max time for DNS + TCP + TLS before the attempt fails. Without it a
    /// black-holed upstream is only bounded by the model's overall timeout.
    pub connect_timeout: Option<Duration>,
    /// Idle time before the kernel sends the first TCP keepalive probe.
    /// Keeps a long wait for a slow first token from being reaped by a NAT
    /// or LB idle timer.
    pub tcp_keepalive: Option<Duration>,
    /// Interval between subsequent keepalive probes.
    pub tcp_keepalive_interval: Option<Duration>,
    /// Unacknowledged probes before the kernel drops the connection.
    pub tcp_keepalive_retries: Option<u32>,
    /// How long an idle connection may sit in the pool before it is
    /// discarded. Must stay below the shortest idle timeout on the path to
    /// the provider, or the pool will hand out connections the far end has
    /// already closed.
    pub pool_idle_timeout: Option<Duration>,
    /// Cap on idle connections kept per upstream host. `None` leaves
    /// reqwest's default (unbounded).
    pub pool_max_idle_per_host: Option<usize>,
}

impl Default for UpstreamHttpConfig {
    fn default() -> Self {
        Self {
            connect_timeout: Some(Duration::from_secs(5)),
            tcp_keepalive: Some(Duration::from_secs(60)),
            tcp_keepalive_interval: Some(Duration::from_secs(30)),
            tcp_keepalive_retries: Some(5),
            pool_idle_timeout: Some(Duration::from_secs(30)),
            pool_max_idle_per_host: None,
        }
    }
}

static CONFIG: OnceLock<UpstreamHttpConfig> = OnceLock::new();

/// Install the process-wide upstream connection settings. Called once
/// during boot, before any bridge builds its client. Later calls are
/// ignored — the pools are already built, so a second set would silently
/// not apply.
pub fn init(cfg: UpstreamHttpConfig) {
    let _ = CONFIG.set(cfg);
}

/// The active settings, defaulting when [`init`] was never called (tests,
/// embedded uses).
pub fn config() -> &'static UpstreamHttpConfig {
    CONFIG.get_or_init(UpstreamHttpConfig::default)
}

/// A `reqwest::ClientBuilder` with the connection settings applied. Callers
/// add their own `user_agent` / TLS options and `build()`.
pub fn client_builder() -> reqwest::ClientBuilder {
    let cfg = config();
    let mut b = reqwest::Client::builder()
        .pool_idle_timeout(cfg.pool_idle_timeout)
        .tcp_keepalive(cfg.tcp_keepalive);
    if let Some(d) = cfg.connect_timeout {
        b = b.connect_timeout(d);
    }
    if let Some(d) = cfg.tcp_keepalive_interval {
        b = b.tcp_keepalive_interval(d);
    }
    if let Some(n) = cfg.tcp_keepalive_retries {
        b = b.tcp_keepalive_retries(n);
    }
    if let Some(n) = cfg.pool_max_idle_per_host {
        b = b.pool_max_idle_per_host(n);
    }
    b
}

/// Render a `reqwest::Error` as a single diagnostic line: the top-level
/// message followed by every distinct `source()` cause, with credentials
/// stripped from any embedded URL.
///
/// reqwest's own `Display` stops at "error sending request for url (…)",
/// which is identical for a DNS failure, a refused connection, a TLS
/// handshake error, and a pooled connection the far end already closed.
/// The causes below it are what name the actual fault, e.g.
/// `… : client error (Connect): tcp connect error: Connection refused (os error 111)`.
pub fn transport_error_message(err: &reqwest::Error) -> String {
    let mut msg = err.to_string();
    if let Some(url) = err.url() {
        let raw = url.as_str();
        if msg.contains(raw) {
            msg = msg.replace(raw, &redact_url(url));
        }
    }
    append_source_chain(&mut msg, err);
    msg
}

/// Same as [`transport_error_message`] for error types that aren't
/// `reqwest::Error` (websocket handshakes, SDK dispatch errors) — no URL
/// is available to redact, so only the cause chain is appended.
pub fn error_with_causes(err: &(dyn std::error::Error + 'static)) -> String {
    let mut msg = err.to_string();
    append_source_chain(&mut msg, err);
    msg
}

fn append_source_chain(msg: &mut String, err: &(dyn std::error::Error + 'static)) {
    let mut source = err.source();
    let mut depth = 0;
    while let Some(cause) = source {
        if depth >= MAX_SOURCE_DEPTH {
            break;
        }
        let text = cause.to_string();
        // hyper repeats the innermost message at several levels; only add
        // a cause that isn't already the tail of what we have.
        if !text.is_empty() && !msg.ends_with(&text) {
            msg.push_str(": ");
            msg.push_str(&text);
        }
        source = cause.source();
        depth += 1;
    }
}

/// Replace the values of credential-bearing query parameters with
/// `REDACTED`, leaving everything else (host, path, `api-version`, …)
/// intact for diagnosis.
fn redact_url(url: &reqwest::Url) -> String {
    if url.query().is_none() {
        return url.as_str().to_string();
    }
    let mut out = url.clone();
    let pairs: Vec<(String, String)> = url
        .query_pairs()
        .map(|(k, v)| {
            if is_sensitive_param(&k) {
                (k.into_owned(), "REDACTED".to_string())
            } else {
                (k.into_owned(), v.into_owned())
            }
        })
        .collect();
    {
        let mut q = out.query_pairs_mut();
        q.clear();
        for (k, v) in &pairs {
            q.append_pair(k, v);
        }
    }
    out.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_bound_connect_and_expire_idle_before_reqwest_would() {
        let cfg = UpstreamHttpConfig::default();
        assert!(cfg.connect_timeout.is_some(), "connect must be bounded");
        assert!(cfg.tcp_keepalive.is_some(), "keepalive must be on");
        // reqwest's own default is 90s; anything at or above that reopens
        // the stale-pooled-connection window this config exists to close.
        assert!(cfg.pool_idle_timeout.unwrap() < Duration::from_secs(90));
    }

    #[test]
    fn client_builder_applies_settings() {
        // Smoke: the builder must accept every configured knob.
        let client = client_builder().user_agent("aisix-test").build();
        assert!(client.is_ok(), "{:?}", client.err());
    }

    #[test]
    fn redacts_credential_query_params_only() {
        let url = reqwest::Url::parse(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini:generateContent\
             ?key=AIzaSy-super-secret&alt=sse",
        )
        .unwrap();
        let out = redact_url(&url);
        assert!(!out.contains("AIzaSy-super-secret"), "{out}");
        assert!(out.contains("key=REDACTED"), "{out}");
        // Non-credential params stay readable — they're the diagnostic bit.
        assert!(out.contains("alt=sse"), "{out}");
    }

    #[test]
    fn redacts_access_token_case_insensitively() {
        let url =
            reqwest::Url::parse("https://example.com/v1/chat?Access_Token=ya29.live&x=1").unwrap();
        let out = redact_url(&url);
        assert!(!out.contains("ya29.live"), "{out}");
        assert!(out.contains("x=1"), "{out}");
    }

    /// Suffix matching exists so vendor-prefixed and punctuation variants
    /// don't have to be enumerated one by one — each of these would have
    /// slipped through an exact-name denylist.
    #[test]
    fn redacts_credential_parameter_aliases() {
        for name in [
            "api-key",
            "api_key",
            "apiKey",
            "client_secret",
            "client-secret",
            "X-Amz-Signature",
            "X-Amz-Security-Token",
            "X-Amz-Credential",
            "SIG",
            "refresh_token",
            "subscription-key",
        ] {
            let url = reqwest::Url::parse(&format!("https://h/p?{name}=live-secret-value&keep=1"))
                .unwrap();
            let out = redact_url(&url);
            assert!(
                !out.contains("live-secret-value"),
                "{name} was not redacted: {out}"
            );
            assert!(out.contains("keep=1"), "{name} over-redacted: {out}");
        }
    }

    /// The flip side: parameters that merely look credential-ish must stay
    /// readable, since they are the diagnostic content of the URL.
    #[test]
    fn keeps_non_credential_parameters_readable() {
        let url = reqwest::Url::parse(
            "https://h/p?api-version=2024-10-21&alt=sse&keyword=hello&signature_version=4",
        )
        .unwrap();
        let out = redact_url(&url);
        assert!(out.contains("api-version=2024-10-21"), "{out}");
        assert!(out.contains("alt=sse"), "{out}");
        assert!(out.contains("keyword=hello"), "{out}");
        assert!(out.contains("signature_version=4"), "{out}");
    }

    #[test]
    fn url_without_query_is_untouched() {
        let url = reqwest::Url::parse("https://api.openai.com/v1/chat/completions").unwrap();
        assert_eq!(
            redact_url(&url),
            "https://api.openai.com/v1/chat/completions"
        );
    }

    #[derive(Debug)]
    struct Layer(&'static str, Option<Box<Layer>>);

    impl std::fmt::Display for Layer {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(self.0)
        }
    }

    impl std::error::Error for Layer {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            self.1
                .as_ref()
                .map(|b| b.as_ref() as &(dyn std::error::Error + 'static))
        }
    }

    #[test]
    fn cause_chain_is_flattened_into_one_line() {
        let err = Layer(
            "error sending request",
            Some(Box::new(Layer(
                "client error (Connect)",
                Some(Box::new(Layer("tcp connect error: refused", None))),
            ))),
        );
        assert_eq!(
            error_with_causes(&err),
            "error sending request: client error (Connect): tcp connect error: refused"
        );
    }

    #[test]
    fn repeated_tail_cause_is_not_duplicated() {
        // hyper commonly restates the innermost message one level up.
        let err = Layer("outer: refused", Some(Box::new(Layer("refused", None))));
        assert_eq!(error_with_causes(&err), "outer: refused");
    }

    /// The whole point of `transport_error_message`, against a real
    /// `reqwest::Error` rather than a hand-built chain: reqwest's own
    /// `Display` is "error sending request for url (…)" for a refused
    /// connection, a DNS failure, a TLS error, and a stale pooled
    /// connection alike. Operators can't tell those apart, which is what
    /// made AISIX-Cloud#1122 undiagnosable from the logs.
    #[tokio::test]
    async fn real_transport_error_names_the_root_cause() {
        // Bind an ephemeral loopback port and immediately release it, so the
        // connect is refused straight away without assuming any fixed port
        // is free (no timeout, no external network needed).
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");
        drop(listener);

        let client = client_builder().build().expect("client builds");
        let err = client
            .get(format!("http://{addr}/v1/chat/completions"))
            .send()
            .await
            .expect_err("connect to a closed port must fail");

        let top_level = err.to_string();
        let with_causes = transport_error_message(&err);

        assert!(
            !top_level.to_lowercase().contains("refused"),
            "reqwest's Display is expected to hide the cause; got: {top_level}"
        );
        assert!(
            with_causes.to_lowercase().contains("refused"),
            "the cause chain must name the actual fault; got: {with_causes}"
        );
        assert!(
            with_causes.len() > top_level.len(),
            "causes must add information"
        );
    }
}
