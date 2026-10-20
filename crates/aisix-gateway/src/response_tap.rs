//! Upstream response headers an observability export asked for.
//!
//! An exporter may list `upstream_response_headers`; the values come off the
//! upstream's HTTP response, which only the code that sent the request ever
//! sees — a provider bridge, or a proxy route that talks to its upstream
//! itself. Each of those calls [`note_upstream_response`] right after the
//! response head arrives. The proxy installs one [`ResponseHeaderTap`] per
//! request (carrying the names the request's exporters want) and drains it
//! per attempt; outside such a scope the call is a no-op, which is what a
//! background poll or a gateway-initiated sub-call relies on.
//!
//! The provider's own request id is read off the same head whether or not
//! an exporter asked for any header: it is what a provider's support asks
//! for about a failed call, and an error response carries no body id.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

/// The longest value recorded per header, in characters.
const MAX_VALUE_CHARS: usize = 256;

/// The values of `names` on `headers`, keyed by name: repeated occurrences
/// joined with `", "` in arrival order, bytes decoded lossily as UTF-8,
/// control characters stripped and the result capped at 256 characters. A
/// header that is absent, or whose value strips to nothing, has no key.
/// Shared by the request-header and response-header captures so both follow
/// one rule.
pub fn capture_header_values(
    headers: &http::HeaderMap,
    names: &[String],
) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for name in names {
        let joined = headers
            .get_all(name.as_str())
            .iter()
            .map(|v| String::from_utf8_lossy(v.as_bytes()))
            .collect::<Vec<_>>()
            .join(", ");
        let value: String = joined
            .chars()
            .filter(|c| !c.is_control())
            .take(MAX_VALUE_CHARS)
            .collect();
        if !value.is_empty() {
            out.insert(name.clone(), value);
        }
    }
    out
}

/// The headers a provider names its own request id in, in the order they
/// are preferred when a response carries more than one (Azure sends both
/// `x-request-id` and `apim-request-id`).
pub const PROVIDER_REQUEST_ID_HEADERS: &[&str] = &[
    "x-amzn-requestid",
    "x-request-id",
    "request-id",
    "x-ms-request-id",
    "apim-request-id",
    "x-goog-request-id",
    "cf-ray",
];

/// The provider's request id on `headers`, under the value rules of
/// [`capture_header_values`]; empty when none of
/// [`PROVIDER_REQUEST_ID_HEADERS`] carries one.
pub fn provider_request_id(headers: &http::HeaderMap) -> String {
    PROVIDER_REQUEST_ID_HEADERS
        .iter()
        .find_map(|name| {
            let value: String = headers
                .get(*name)?
                .to_str()
                .ok()?
                .chars()
                .filter(|c| !c.is_control())
                .take(MAX_VALUE_CHARS)
                .collect();
            let value = value.trim();
            (!value.is_empty()).then(|| value.to_string())
        })
        .unwrap_or_default()
}

/// What one upstream response head left for the request's telemetry.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct UpstreamResponseCapture {
    /// The headers the request's exporters asked for.
    pub headers: BTreeMap<String, String>,
    /// See [`provider_request_id`].
    pub request_id: String,
}

impl UpstreamResponseCapture {
    pub fn is_empty(&self) -> bool {
        self.headers.is_empty() && self.request_id.is_empty()
    }
}

/// One request's response-header capture: the names its exporters asked
/// for, and what the latest upstream response carried.
#[derive(Debug, Default)]
pub struct ResponseHeaderTap {
    names: Vec<String>,
    latest: Mutex<UpstreamResponseCapture>,
}

impl ResponseHeaderTap {
    pub fn new(names: Vec<String>) -> Self {
        Self {
            names,
            latest: Mutex::default(),
        }
    }

    fn record(&self, headers: &http::HeaderMap) {
        let captured = UpstreamResponseCapture {
            headers: capture_header_values(headers, &self.names),
            request_id: provider_request_id(headers),
        };
        *self.lock() = captured;
    }

    /// Move the latest capture out, leaving nothing behind.
    pub fn take(&self) -> UpstreamResponseCapture {
        std::mem::take(&mut *self.lock())
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, UpstreamResponseCapture> {
        self.latest
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

tokio::task_local! {
    static TAP: Arc<ResponseHeaderTap>;
}

/// Install `tap` for everything `fut` does.
pub fn scope<F: std::future::Future>(
    tap: Arc<ResponseHeaderTap>,
    fut: F,
) -> impl std::future::Future<Output = F::Output> {
    TAP.scope(tap, fut)
}

/// Install `tap` for one synchronous call (a polled response body).
pub fn sync_scope<T>(tap: &Arc<ResponseHeaderTap>, f: impl FnOnce() -> T) -> T {
    TAP.sync_scope(tap.clone(), f)
}

/// Record an upstream response head for the current request's exporters.
/// Call it once per upstream HTTP response that answers a dispatched
/// attempt, as soon as the head is in hand — success or failure.
pub fn note_upstream_response(headers: &http::HeaderMap) {
    let _ = TAP.try_with(|tap| tap.record(headers));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The value rules the e2e cannot drive through a real HTTP client:
    /// non-UTF-8 bytes, a value that sanitises to nothing, and the cap.
    #[test]
    fn captured_header_values_follow_the_value_rules() {
        let names: Vec<String> = ["x-a", "x-b", "x-c", "x-d", "x-absent"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut h = http::HeaderMap::new();
        h.append("x-a", "one".parse().unwrap());
        h.append("x-a", "two".parse().unwrap());
        h.append(
            "x-b",
            http::HeaderValue::from_bytes(b"caf\xe9\tbar").unwrap(),
        );
        h.append("x-c", http::HeaderValue::from_bytes(b"\t").unwrap());
        h.append("x-d", "y".repeat(300).parse().unwrap());
        h.append("x-other", "not listed".parse().unwrap());
        let got = capture_header_values(&h, &names);
        assert_eq!(got.get("x-a").map(String::as_str), Some("one, two"));
        assert_eq!(got.get("x-b").map(String::as_str), Some("caf\u{fffd}bar"));
        assert!(
            !got.contains_key("x-c"),
            "a value sanitised to nothing has no key"
        );
        assert_eq!(got["x-d"].chars().count(), 256);
        assert!(!got.contains_key("x-absent"));
        assert!(!got.contains_key("x-other"));
        assert_eq!(got.len(), 3);
    }

    #[tokio::test]
    async fn a_response_is_recorded_only_inside_a_scope() {
        let mut h = http::HeaderMap::new();
        h.insert("x-upstream-id", "u-1".parse().unwrap());
        h.insert("request-id", "req_1".parse().unwrap());
        note_upstream_response(&h);

        let tap = Arc::new(ResponseHeaderTap::new(vec!["x-upstream-id".into()]));
        scope(tap.clone(), async { note_upstream_response(&h) }).await;
        let got = tap.take();
        assert_eq!(
            got.headers.get("x-upstream-id").map(String::as_str),
            Some("u-1")
        );
        assert_eq!(got.request_id, "req_1");
        assert!(tap.take().is_empty(), "take leaves nothing behind");

        // No exporter asked for a header: the request id is still read.
        let idle = Arc::new(ResponseHeaderTap::new(Vec::new()));
        scope(idle.clone(), async { note_upstream_response(&h) }).await;
        let got = idle.take();
        assert!(got.headers.is_empty());
        assert_eq!(got.request_id, "req_1");
    }

    #[test]
    fn the_request_id_follows_the_header_preference_order() {
        let mut h = http::HeaderMap::new();
        h.insert("apim-request-id", "apim-1".parse().unwrap());
        assert_eq!(provider_request_id(&h), "apim-1");
        h.insert("x-request-id", "xr-1".parse().unwrap());
        assert_eq!(provider_request_id(&h), "xr-1");
        h.insert("x-request-id", http::HeaderValue::from_static(" "));
        assert_eq!(
            provider_request_id(&h),
            "apim-1",
            "a blank value is skipped"
        );
        assert_eq!(provider_request_id(&http::HeaderMap::new()), "");
    }
}
