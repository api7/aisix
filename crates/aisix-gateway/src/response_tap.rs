//! Upstream response headers and bodies for observability exports.
//!
//! An exporter may list `upstream_response_headers`; the values come off the
//! upstream's HTTP response, which only the code that sent the request ever
//! sees — a provider bridge, or a proxy route that talks to its upstream
//! itself. [`tap_upstream_response`] records the head and taps bytes as the
//! route reads them. The proxy drains the tap per attempt, moving the body
//! capture with the headers even if the stream has only just started.
//!
//! The provider's own request id is read off the same head whether or not
//! an exporter asked for any header: it is what a provider's support asks
//! for about a failed call, and an error response carries no body id.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use bytes::{Buf, Bytes, BytesMut};
use futures::{Stream, StreamExt};
use http::header;

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

/// What one upstream response left for the request's telemetry.
#[derive(Debug, Default)]
pub struct UpstreamResponseCapture {
    /// The headers the request's exporters asked for.
    pub headers: BTreeMap<String, String>,
    /// See [`provider_request_id`].
    pub request_id: String,
    body: Option<Arc<UpstreamBodyTap>>,
}

impl UpstreamResponseCapture {
    pub fn is_empty(&self) -> bool {
        self.headers.is_empty() && self.request_id.is_empty() && self.body.is_none()
    }

    pub fn take_body(&mut self) -> Option<CapturedUpstreamBody> {
        let body = self.body.take()?;
        let mut prefix = std::mem::take(&mut *body.prefix.lock().unwrap());
        prefix.finish(body.cap);
        (!prefix.bytes.is_empty() || prefix.truncated).then(|| CapturedUpstreamBody {
            bytes: prefix.bytes.freeze(),
            truncated: prefix.truncated,
            format: body.format,
        })
    }

    pub fn discard_success_body(&mut self) {
        if self.body.as_ref().is_some_and(|body| body.success) {
            self.body = None;
        }
    }
}

pub struct CapturedUpstreamBody {
    pub bytes: Bytes,
    pub truncated: bool,
    pub format: UpstreamBodyFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpstreamBodyFormat {
    Text,
    Sse,
    JsonLines,
}

#[derive(Debug, Default)]
struct BodyPrefix {
    bytes: BytesMut,
    truncated: bool,
    sse: Option<SsePrefix>,
}

impl BodyPrefix {
    fn push(&mut self, cap: usize, bytes: &[u8]) {
        if let Some(sse) = &mut self.sse {
            sse.push(cap, bytes, &mut self.bytes, &mut self.truncated);
            return;
        }
        let kept = bytes.len().min(cap - self.bytes.len());
        self.bytes.extend_from_slice(&bytes[..kept]);
        self.truncated |= kept < bytes.len();
    }

    fn finish(&mut self, cap: usize) {
        if let Some(sse) = &mut self.sse {
            if sse.boundary.ends_with_blank_cr() {
                sse.finish_frame(cap, &mut self.bytes, &mut self.truncated);
            }
            self.truncated |= sse.oversized || !sse.frame.is_empty();
        }
    }
}

#[derive(Debug, Default)]
struct SsePrefix {
    frame: BytesMut,
    oversized: bool,
    boundary: crate::sse::FrameBoundary,
}

impl SsePrefix {
    fn push(&mut self, cap: usize, mut bytes: &[u8], kept: &mut BytesMut, truncated: &mut bool) {
        while !bytes.is_empty() {
            let end = self.boundary.next_end(bytes);
            let (part, rest) = bytes.split_at(end.unwrap_or(bytes.len()));
            if !self.oversized {
                if part.len() <= cap - self.frame.len() {
                    self.frame.extend_from_slice(part);
                } else {
                    self.frame.clear();
                    self.oversized = true;
                }
            }
            if end.is_some() {
                self.finish_frame(cap, kept, truncated);
            }
            bytes = rest;
        }
    }

    fn finish_frame(&mut self, cap: usize, kept: &mut BytesMut, truncated: &mut bool) {
        if self.oversized
            || std::str::from_utf8(&self.frame).is_err()
            || self
                .frame
                .iter()
                .any(|b| b.is_ascii_control() && !b"\t\r\n".contains(b))
        {
            *truncated = true;
        } else {
            // Keep the most recent complete frames when the total reaches
            // the cap, so a long stream can still retain its terminal error.
            while self.frame.len() > cap - kept.len() {
                let end = crate::sse::find_frame_end(kept).expect("captured complete SSE frame");
                kept.advance(end);
                *truncated = true;
            }
            kept.extend_from_slice(&self.frame);
        }
        self.frame.clear();
        self.oversized = false;
    }
}

/// A body poll owns this handle even after routing settles its attempt and the
/// request tap starts tracking another one.
#[derive(Debug)]
pub struct UpstreamBodyTap {
    cap: usize,
    success: bool,
    format: UpstreamBodyFormat,
    prefix: Mutex<BodyPrefix>,
}

impl UpstreamBodyTap {
    pub fn push(&self, bytes: &[u8]) {
        self.prefix.lock().unwrap().push(self.cap, bytes);
    }

    pub fn push_json_frame(&self, bytes: &[u8]) {
        let mut prefix = self.prefix.lock().unwrap();
        if !prefix.bytes.is_empty() {
            prefix.push(self.cap, b"\n");
        }
        prefix.push(self.cap, bytes);
    }
}

/// One request's response-header capture: the names its exporters asked
/// for, and what the latest upstream response carried.
#[derive(Debug, Default)]
pub struct ResponseHeaderTap {
    names: Vec<String>,
    /// Whether an exporter of this request runs at `content_mode: full`.
    body_cap: Option<u32>,
    /// Generated output protected by a guardrail uses its post-policy capture.
    protects_output: AtomicBool,
    request_id: String,
    latest: Mutex<UpstreamResponseCapture>,
}

impl ResponseHeaderTap {
    pub fn new(names: Vec<String>, body_cap: Option<u32>, request_id: String) -> Self {
        Self {
            names,
            body_cap,
            request_id,
            ..Self::default()
        }
    }

    pub fn protects_output(&self) -> bool {
        self.protects_output.load(Ordering::Relaxed)
    }

    /// Concurrent sub-calls own separate body captures, with the same
    /// output policy and content limit as the parent request.
    pub fn for_subcall(&self) -> Option<Arc<Self>> {
        Some(Arc::new(Self {
            body_cap: Some(self.body_cap?),
            protects_output: AtomicBool::new(self.protects_output()),
            request_id: self.request_id.clone(),
            ..Self::default()
        }))
    }

    fn new_body(&self, success: bool, format: UpstreamBodyFormat) -> Option<Arc<UpstreamBodyTap>> {
        let cap = self.body_cap? as usize;
        if success && self.protects_output() {
            return None;
        }
        // Keep the complete ordinary error body up to the parser's existing
        // bound so an exporter with a lower cap can still shrink valid JSON.
        let cap = if !success && format == UpstreamBodyFormat::Text {
            cap.max(crate::MAX_UPSTREAM_ERROR_BODY_BYTES)
        } else {
            cap
        };
        Some(Arc::new(UpstreamBodyTap {
            cap,
            success,
            format,
            prefix: Mutex::new(BodyPrefix {
                sse: (format == UpstreamBodyFormat::Sse).then(SsePrefix::default),
                ..BodyPrefix::default()
            }),
        }))
    }

    fn record(&self, headers: &http::HeaderMap, body: Option<Arc<UpstreamBodyTap>>) {
        let captured = UpstreamResponseCapture {
            headers: capture_header_values(headers, &self.names),
            request_id: provider_request_id(headers),
            body,
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
    let _ = TAP.try_with(|tap| tap.record(headers, None));
}

/// Clear the body before an internal retry that shares the same usage row.
pub fn clear_upstream_body() {
    let _ = TAP.try_with(|tap| {
        if tap.body_cap.is_some() {
            tap.lock().body = None;
        }
    });
}

/// Use the post-policy output of this request's resolved guardrail chain.
pub fn protect_upstream_output() {
    let _ = TAP.try_with(|tap| {
        tap.protects_output.store(true, Ordering::Relaxed);
    });
}

pub fn upstream_output_is_protected() -> bool {
    TAP.try_with(|tap| tap.protects_output()).unwrap_or(false)
}

/// SDKs that consume a response themselves can attach its text payload to the
/// last noted response. Hold the returned handle when reading it later.
pub fn tap_upstream_body(
    success: bool,
    format: UpstreamBodyFormat,
) -> Option<Arc<UpstreamBodyTap>> {
    TAP.try_with(|tap| {
        let body = tap.new_body(success, format)?;
        tap.lock().body = Some(body.clone());
        Some(body)
    })
    .ok()
    .flatten()
}

pub fn note_upstream_error_body(bytes: &[u8]) {
    if !bytes.is_empty() {
        if let Some(body) = tap_upstream_body(false, UpstreamBodyFormat::Text) {
            body.push(bytes);
        }
    }
}

/// Register an HTTP error body a route already buffered and approved.
pub fn note_checked_http_error_body(headers: &http::HeaderMap, bytes: &[u8]) {
    if let Some(body) =
        text_content_type(headers).and_then(|format| tap_upstream_body(false, format))
    {
        body.push(bytes);
    }
}

/// Record the response at its head. For a full-content request, observe its
/// text bytes exactly where reqwest delivers them. In particular, a body read
/// that gets dropped on timeout leaves the prefix it already consumed.
pub fn tap_upstream_response(response: reqwest::Response) -> reqwest::Response {
    tap_http_response(response, text_content_type)
}

fn tap_http_response(
    mut response: reqwest::Response,
    format: impl FnOnce(&http::HeaderMap) -> Option<UpstreamBodyFormat>,
) -> reqwest::Response {
    let capture = TAP
        .try_with(|tap| {
            let capture = tap
                .body_cap
                .and_then(|_| format(response.headers()))
                .and_then(|format| tap.new_body(response.status().is_success(), format));
            tap.record(response.headers(), capture.clone());
            capture
        })
        .ok()
        .flatten();
    if let Some(capture) = capture {
        response.extensions_mut().insert(capture);
    }
    response
}

fn has_content_encoding(headers: &http::HeaderMap) -> bool {
    headers
        .get_all(header::CONTENT_ENCODING)
        .iter()
        .any(|value| {
            value.to_str().map_or(true, |value| {
                value
                    .split(',')
                    .any(|coding| !coding.trim().eq_ignore_ascii_case("identity"))
            })
        })
}

fn text_content_type(headers: &http::HeaderMap) -> Option<UpstreamBodyFormat> {
    if has_content_encoding(headers) {
        return None;
    }
    let Some(content_type) = headers.get(header::CONTENT_TYPE) else {
        return Some(UpstreamBodyFormat::Text);
    };
    let mime = content_type
        .to_str()
        .ok()?
        .split(';')
        .next()?
        .trim()
        .to_ascii_lowercase();
    match mime.as_str() {
        "text/event-stream" => Some(UpstreamBodyFormat::Sse),
        "application/jsonl"
        | "application/x-jsonl"
        | "application/ndjson"
        | "application/x-ndjson"
        | "text/jsonl"
        | "text/x-jsonl"
        | "text/ndjson"
        | "text/x-ndjson" => Some(UpstreamBodyFormat::JsonLines),
        "application/json" | "application/json-rpc" | "application/xml" => {
            Some(UpstreamBodyFormat::Text)
        }
        mime if mime.starts_with("text/")
            || mime.starts_with("application/x-amz-json-")
            || mime.ends_with("+json")
            || mime.ends_with("+xml") =>
        {
            Some(UpstreamBodyFormat::Text)
        }
        _ => None,
    }
}

/// Read a response that the caller has selected for its SSE parser. Some
/// providers deliver valid frames without a matching text content type.
/// Construct this while the response still belongs to the current attempt.
pub fn upstream_sse_stream(
    mut response: reqwest::Response,
) -> impl Stream<Item = reqwest::Result<Bytes>> {
    if wants_upstream_error_body()
        && !has_content_encoding(response.headers())
        && !crate::response_is_json(&response)
        && !response
            .extensions()
            .get::<Arc<UpstreamBodyTap>>()
            .is_some_and(|body| body.format != UpstreamBodyFormat::Text)
    {
        response = tap_http_response(response, |_| Some(UpstreamBodyFormat::Sse));
    }
    upstream_body_stream(response)
}

/// Read the response stream and keep the prefix actually consumed by this
/// attempt, or the latest complete SSE frames. Responses that were never
/// registered have no capture to update.
pub fn upstream_body_stream(
    response: reqwest::Response,
) -> impl Stream<Item = reqwest::Result<Bytes>> {
    let capture = response.extensions().get::<Arc<UpstreamBodyTap>>().cloned();
    response.bytes_stream().inspect(move |item| {
        if let (Some(capture), Ok(bytes)) = (&capture, item) {
            capture.push(bytes);
        }
    })
}

pub async fn upstream_body(response: reqwest::Response) -> reqwest::Result<Bytes> {
    if response
        .extensions()
        .get::<Arc<UpstreamBodyTap>>()
        .is_none()
    {
        return response.bytes().await;
    }
    let mut stream = std::pin::pin!(upstream_body_stream(response));
    let Some(first) = stream.next().await.transpose()? else {
        return Ok(Bytes::new());
    };
    let Some(second) = stream.next().await.transpose()? else {
        return Ok(first);
    };
    let mut body = BytesMut::with_capacity(first.len() + second.len());
    body.extend_from_slice(&first);
    body.extend_from_slice(&second);
    while let Some(chunk) = stream.next().await {
        body.extend_from_slice(&chunk?);
    }
    Ok(body.freeze())
}

/// Decode just as reqwest's `Response::json` does, with the same error shown
/// to the caller. A serde error may quote response content, so log only the
/// causes of real read errors.
pub async fn upstream_json<T: serde::de::DeserializeOwned>(
    response: reqwest::Response,
) -> Result<T, crate::BridgeError> {
    let body = upstream_body(response)
        .await
        .map_err(|e| crate::upstream_body_read_error(&e))?;
    serde_json::from_slice(&body)
        .map_err(|_| crate::BridgeError::UpstreamDecode("error decoding response body".to_string()))
}

/// Log an error response whose body could not be read. Names the request
/// itself rather than relying on the request span being attached here.
pub fn warn_error_body_unread(status: u16, reason: &str) {
    let request_id = TAP
        .try_with(|tap| tap.request_id.clone())
        .unwrap_or_default();
    tracing::warn!(
        status,
        %request_id,
        %reason,
        "failed to read upstream error response body"
    );
}

/// Whether a route that does not otherwise need a non-2xx body should read it.
pub fn wants_upstream_error_body() -> bool {
    TAP.try_with(|tap| tap.body_cap.is_some()).unwrap_or(false)
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

        let tap = Arc::new(ResponseHeaderTap::new(
            vec!["x-upstream-id".into()],
            None,
            String::new(),
        ));
        scope(tap.clone(), async { note_upstream_response(&h) }).await;
        let got = tap.take();
        assert_eq!(
            got.headers.get("x-upstream-id").map(String::as_str),
            Some("u-1")
        );
        assert_eq!(got.request_id, "req_1");
        assert!(tap.take().is_empty(), "take leaves nothing behind");

        // No exporter asked for a header: the request id is still read.
        let idle = Arc::new(ResponseHeaderTap::new(Vec::new(), None, String::new()));
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
