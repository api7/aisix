//! Downloading a chat request's media URL for an upstream that cannot read
//! the URL itself, so the bytes can travel inline instead.
//!
//! The URL is the caller's, and the download runs from inside the
//! operator's network, so each hop goes through `outbound.ssrf_protection`
//! ([`crate::ssrf`]); redirects are followed by hand so every hop passes
//! the same check. Otherwise it is an ordinary upstream request: the
//! environment proxies and `upstream.tls` certificate verification apply,
//! but never the upstream client identity.

use std::borrow::Cow;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::Duration;

use base64::Engine as _;
use futures::StreamExt;
use reqwest::{header, Url};
use tokio::time::Instant;

use crate::bridge::{BridgeContext, BridgeError};
use crate::chat::{ChatFormat, Role};
use crate::content_part::{
    display_uri, mime_type_from_path, normalize_mime_type, parse_part, ContentPart, Media,
    MediaSource,
};
use crate::ssrf::{Refusal, Route, SsrfProtection};

/// Attempts per URL on a timeout or transport failure. An HTTP error
/// status is an answer and is not retried.
const MAX_ATTEMPTS: u32 = 3;
const MAX_REDIRECTS: usize = 10;
const RETRY_BACKOFF: Duration = Duration::from_millis(100);
/// Downloads of one request in flight at once.
const MAX_CONCURRENT: usize = 8;

/// `upstream.media_fetch`, parsed once at boot.
#[derive(Debug, Clone)]
pub struct MediaFetchSettings {
    enabled: bool,
    max_bytes: u64,
    /// `outbound.ssrf_protection`, which every hop goes through.
    ssrf: SsrfProtection,
    /// `proxy.request_body_limit_bytes`: the inbound body and everything
    /// downloaded for the request share it. `0` is no limit.
    request_body_limit_bytes: usize,
}

impl MediaFetchSettings {
    pub fn new(cfg: &aisix_core::config::MediaFetchConfig, ssrf: SsrfProtection) -> Self {
        Self {
            enabled: cfg.enabled,
            max_bytes: cfg.max_bytes,
            ssrf,
            request_body_limit_bytes: 0,
        }
    }
}

static SETTINGS: OnceLock<MediaFetchSettings> = OnceLock::new();

/// Install `upstream.media_fetch`, with `proxy.request_body_limit_bytes`
/// as the per-request total. Called once during boot, after
/// [`crate::ssrf::init`].
pub fn init(cfg: &aisix_core::config::MediaFetchConfig, request_body_limit_bytes: usize) {
    let mut settings = MediaFetchSettings::new(cfg, crate::ssrf::settings().clone());
    settings.request_body_limit_bytes = request_body_limit_bytes;
    let _ = SETTINGS.set(settings);
}

/// What one request may still hold: its inbound body plus every byte
/// downloaded for it, against `proxy.request_body_limit_bytes`.
struct RequestBudget {
    inbound: usize,
    used: AtomicUsize,
    limit: usize,
}

impl RequestBudget {
    fn new(inbound: usize, limit: usize) -> Self {
        Self {
            inbound,
            used: AtomicUsize::new(inbound),
            limit,
        }
    }

    #[cfg(test)]
    fn unlimited() -> Self {
        Self::new(0, 0)
    }

    /// Refuses a body of `more` bytes the request cannot hold.
    fn check(&self, more: u64) -> Result<(), Failure> {
        let total = self.used.load(Ordering::Relaxed) as u64 + more;
        if self.limit > 0 && total > self.limit as u64 {
            return Err(self.over(total));
        }
        Ok(())
    }

    fn claim(&self, n: usize) -> Result<(), Failure> {
        let used = self.used.fetch_add(n, Ordering::Relaxed) + n;
        if self.limit > 0 && used > self.limit {
            return Err(self.over(used as u64));
        }
        Ok(())
    }

    fn over(&self, total: u64) -> Failure {
        Failure::OverRequestLimit {
            downloaded: total.saturating_sub(self.inbound as u64),
        }
    }

    fn release(&self, n: usize) {
        self.used.fetch_sub(n, Ordering::Relaxed);
    }
}

fn settings() -> &'static MediaFetchSettings {
    SETTINGS.get_or_init(|| {
        MediaFetchSettings::new(&Default::default(), crate::ssrf::settings().clone())
    })
}

/// What the bridge's wire carries the bytes as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    Base64,
    Bytes,
}

#[derive(Debug, Clone)]
pub enum FetchedData {
    Base64(String),
    Bytes(Vec<u8>),
}

impl FetchedData {
    pub fn into_base64(self) -> String {
        match self {
            Self::Base64(s) => s,
            Self::Bytes(b) => base64::engine::general_purpose::STANDARD.encode(b),
        }
    }

    pub fn into_bytes(self) -> Vec<u8> {
        match self {
            Self::Bytes(b) => b,
            // Our own encoding, so it always decodes.
            Self::Base64(s) => base64::engine::general_purpose::STANDARD
                .decode(s)
                .unwrap_or_default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Fetched {
    pub mime_type: String,
    pub data: FetchedData,
}

/// The downloads of one request, keyed by URL. A URL the request repeats
/// is downloaded once; [`take`](Self::take) hands its last use the buffer
/// itself, so a URL used once is never copied.
#[derive(Debug, Default)]
pub struct FetchedMedia {
    slots: Mutex<HashMap<String, (Fetched, usize)>>,
    /// Time the downloads took, `None` when nothing was downloaded.
    spent: Option<Duration>,
}

impl FetchedMedia {
    /// `ctx` with the time the downloads took charged to its end-to-end
    /// budget, so the downloads and the upstream call share it. A
    /// streaming context's chunk-gap budget is not an end-to-end one and
    /// stays as it is.
    pub fn charge<'a>(&self, ctx: &'a BridgeContext) -> Cow<'a, BridgeContext> {
        let Some(spent) = self.spent else {
            return Cow::Borrowed(ctx);
        };
        let mut charged = ctx.clone();
        let left = |budget: Duration| budget.saturating_sub(spent).max(Duration::from_millis(1));
        if ctx.non_streaming_deadline.is_some() {
            charged.non_streaming_deadline = ctx.non_streaming_deadline.map(left);
        } else {
            charged.deadline = ctx.deadline.map(left);
        }
        Cow::Owned(charged)
    }

    pub fn take(&self, uri: &str) -> Option<Fetched> {
        let mut slots = self.slots.lock().unwrap_or_else(PoisonError::into_inner);
        let (media, uses) = slots.get_mut(uri)?;
        if *uses > 1 {
            *uses -= 1;
            return Some(media.clone());
        }
        slots.remove(uri).map(|(media, _)| media)
    }
}

/// One URL to download, with the first part that names it.
struct Wanted {
    uri: String,
    explicit_mime_type: Option<String>,
    location: String,
    uses: usize,
}

/// A media URL a request names, for a wire the chat form does not carry
/// (an Anthropic Messages body forwarded as it is).
pub struct MediaUrl {
    pub uri: String,
    /// The media type the request states for it, if any.
    pub mime_type: Option<String>,
    /// Where the request names it (`messages[2].content[0].content[1]`),
    /// prefixed to the error a failed download returns.
    pub location: String,
}

/// Download every `http(s)` media URL in the request's user turns that
/// `needs_fetch` says the upstream cannot read, as `encoding`. The
/// translation then takes each from the result instead of the URL.
///
/// Off (`upstream.media_fetch.enabled: false`), nothing is downloaded and
/// the translation refuses those parts. A failed download is a 400 naming
/// the part. All downloads share the request's own end-to-end upstream
/// budget; [`FetchedMedia::charge`] takes what they spent off the upstream
/// call's.
pub async fn fetch_media_urls(
    req: &ChatFormat,
    ctx: &BridgeContext,
    encoding: Encoding,
    needs_fetch: impl Fn(&Media<'_>) -> bool,
) -> Result<FetchedMedia, BridgeError> {
    fetch_media_urls_in(req, ctx, encoding, needs_fetch, false).await
}

/// [`fetch_media_urls`] for an upstream that also takes media inside a
/// tool result (`role:"tool"` content parts).
pub async fn fetch_media_urls_including_tool_results(
    req: &ChatFormat,
    ctx: &BridgeContext,
    encoding: Encoding,
    needs_fetch: impl Fn(&Media<'_>) -> bool,
) -> Result<FetchedMedia, BridgeError> {
    fetch_media_urls_in(req, ctx, encoding, needs_fetch, true).await
}

async fn fetch_media_urls_in(
    req: &ChatFormat,
    ctx: &BridgeContext,
    encoding: Encoding,
    needs_fetch: impl Fn(&Media<'_>) -> bool,
    tool_results: bool,
) -> Result<FetchedMedia, BridgeError> {
    if !settings().enabled {
        return Ok(FetchedMedia::default());
    }
    let mut urls: Vec<MediaUrl> = Vec::new();
    // Every part naming a URL takes the download from the result, whether
    // or not that part alone would have needed it, so all are counted.
    let mut occurrences: HashMap<&str, usize> = HashMap::new();
    // System turns are included: the Anthropic translation carries a
    // system turn that follows the conversation as a user turn.
    for (message, m) in req.messages.iter().enumerate() {
        let scanned = matches!(m.role, Role::User | Role::System | Role::Developer)
            || (tool_results && m.role == Role::Tool);
        if !scanned {
            continue;
        }
        for (part, block) in m.content_blocks.iter().flatten().enumerate() {
            let Ok(ContentPart::Media(media)) = parse_part(block) else {
                continue;
            };
            let MediaSource::Uri {
                uri,
                mime_type,
                explicit,
            } = &media.source
            else {
                continue;
            };
            *occurrences.entry(uri).or_default() += 1;
            if !matches!(media.source.scheme().as_deref(), Some("http" | "https"))
                || !needs_fetch(&media)
            {
                continue;
            }
            urls.push(MediaUrl {
                uri: uri.to_string(),
                mime_type: mime_type.clone().filter(|_| *explicit),
                location: format!("messages[{message}].content[{part}]"),
            });
        }
    }
    let uses = |uri: &str| occurrences.get(uri).copied().unwrap_or(1);
    fetch_wanted(dedupe(urls, uses), ctx, encoding).await
}

/// Download `urls` as `encoding` — the [`fetch_media_urls`] rules for a
/// request whose media the caller has already located. Each URL is
/// downloaded once however often it is named, and every naming takes it
/// from the result.
pub async fn fetch_urls(
    urls: Vec<MediaUrl>,
    ctx: &BridgeContext,
    encoding: Encoding,
) -> Result<FetchedMedia, BridgeError> {
    if !settings().enabled {
        return Ok(FetchedMedia::default());
    }
    let mut counts: HashMap<String, usize> = HashMap::new();
    for u in &urls {
        *counts.entry(u.uri.clone()).or_default() += 1;
    }
    let urls: Vec<MediaUrl> = urls
        .into_iter()
        .filter(|u| Url::parse(&u.uri).is_ok_and(|p| matches!(p.scheme(), "http" | "https")))
        .collect();
    let uses = |uri: &str| counts.get(uri).copied().unwrap_or(1);
    fetch_wanted(dedupe(urls, uses), ctx, encoding).await
}

/// Whether `upstream.media_fetch` is on, i.e. whether a URL the upstream
/// cannot read is downloaded rather than refused.
pub fn enabled() -> bool {
    settings().enabled
}

fn dedupe(urls: Vec<MediaUrl>, uses: impl Fn(&str) -> usize) -> Vec<Wanted> {
    let mut wanted: Vec<Wanted> = Vec::new();
    let mut seen: HashMap<String, ()> = HashMap::new();
    for u in urls {
        if let Entry::Vacant(at) = seen.entry(u.uri.clone()) {
            at.insert(());
            wanted.push(Wanted {
                uses: uses(&u.uri),
                uri: u.uri,
                explicit_mime_type: u.mime_type,
                location: u.location,
            });
        }
    }
    wanted
}

async fn fetch_wanted(
    wanted: Vec<Wanted>,
    ctx: &BridgeContext,
    encoding: Encoding,
) -> Result<FetchedMedia, BridgeError> {
    let settings = settings();
    if wanted.is_empty() {
        return Ok(FetchedMedia::default());
    }
    let budget = ctx.non_streaming_deadline.or(ctx.deadline);
    let started = Instant::now();
    let deadline = budget.map(|b| started + b);
    let held = RequestBudget::new(ctx.inbound_body_bytes, settings.request_body_limit_bytes);
    let held = &held;
    let mut downloads = futures::stream::iter(wanted.into_iter().map(|w| async move {
        let download = fetch_one(&w, settings, encoding, held);
        let result = match deadline {
            Some(at) => tokio::time::timeout_at(at, download)
                .await
                .unwrap_or(Err(Failure::Timeout.into())),
            None => download.await,
        };
        (w, result)
    }))
    .buffer_unordered(MAX_CONCURRENT);
    let mut slots = HashMap::new();
    while let Some((w, result)) = downloads.next().await {
        match result {
            Ok(media) => {
                slots.insert(w.uri, (media, w.uses));
            }
            Err(Failed {
                failure: Failure::OverRequestLimit { downloaded },
                ..
            }) => {
                return Err(BridgeError::RequestTooLarge {
                    limit_bytes: settings.request_body_limit_bytes,
                    inbound_bytes: ctx.inbound_body_bytes,
                    downloaded_bytes: downloaded,
                });
            }
            Err(failure) => {
                tracing::warn!(
                    request_id = %ctx.request_id,
                    url = %display_uri(&w.uri),
                    redirected_to = failure.redirected_to.as_ref().map(|u| display_uri(u.as_str())),
                    failure = ?failure.failure,
                    "media URL not downloaded"
                );
                return Err(BridgeError::InvalidRequest(format!(
                    "{}: {}",
                    w.location,
                    failure.describe(&w.uri, settings),
                )));
            }
        }
    }
    Ok(FetchedMedia {
        slots: Mutex::new(slots),
        spent: Some(started.elapsed()),
    })
}

async fn fetch_one(
    w: &Wanted,
    settings: &MediaFetchSettings,
    encoding: Encoding,
    held: &RequestBudget,
) -> Result<Fetched, Failed> {
    let url = Url::parse(&w.uri).map_err(|_| Failure::BadUrl)?;
    let mut attempt = 1;
    let (bytes, content_type, final_url) = loop {
        match download(&url, settings, held).await {
            Err(f) if f.failure.retryable() && attempt < MAX_ATTEMPTS => {
                tokio::time::sleep(RETRY_BACKOFF * attempt).await;
                attempt += 1;
            }
            result => break result?,
        }
    };
    let mime_type = w
        .explicit_mime_type
        .clone()
        .or(content_type)
        .or_else(|| mime_type_from_path(&w.uri).map(str::to_string))
        .or_else(|| mime_type_from_path(final_url.path()).map(str::to_string))
        .or_else(|| sniff(&bytes).map(str::to_string))
        .ok_or(Failure::UnknownType)?;
    let data = match encoding {
        Encoding::Base64 => {
            FetchedData::Base64(base64::engine::general_purpose::STANDARD.encode(&bytes))
        }
        Encoding::Bytes => FetchedData::Bytes(bytes),
    };
    Ok(Fetched { mime_type, data })
}

/// One attempt: the URL and its redirects, each hop checked on its own.
/// Returns the body, the response's non-generic media type, and the URL
/// that answered.
async fn download(
    url: &Url,
    settings: &MediaFetchSettings,
    held: &RequestBudget,
) -> Result<(Vec<u8>, Option<String>, Url), Failed> {
    let mut current = url.clone();
    for hop in 0..=MAX_REDIRECTS {
        let at = |failure: Failure| Failed {
            failure,
            redirected_to: (hop > 0).then(|| current.clone()),
        };
        let resp = match fetch_hop(&current, settings).await {
            Ok(Hop::Answered(resp)) => resp,
            Ok(Hop::Redirect(_)) if hop == MAX_REDIRECTS => {
                return Err(Failure::TooManyRedirects.into())
            }
            Ok(Hop::Redirect(next)) => {
                current = next;
                continue;
            }
            Err(failure) => return Err(at(failure)),
        };
        let content_type = resp
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|v| normalize_mime_type(v.split(';').next().unwrap_or("")))
            .filter(|m| {
                !m.is_empty() && m != "application/octet-stream" && !m.starts_with("binary/")
            });
        let body = read_capped(resp, settings.max_bytes, held)
            .await
            .map_err(at)?;
        return Ok((body, content_type, current));
    }
    Err(Failure::TooManyRedirects.into())
}

enum Hop {
    Answered(reqwest::Response),
    Redirect(Url),
}

/// One hop: checked, sent, and either answered with a success status or
/// redirected.
async fn fetch_hop(url: &Url, settings: &MediaFetchSettings) -> Result<Hop, Failure> {
    let route = settings.ssrf.route(url).await?;
    let resp = client(route)?
        .get(url.clone())
        .send()
        .await
        .map_err(classify)?;
    let status = resp.status();
    if status.is_redirection() {
        if let Some(location) = resp.headers().get(header::LOCATION) {
            let next = location
                .to_str()
                .ok()
                .and_then(|l| url.join(l).ok())
                .ok_or(Failure::BadRedirect)?;
            if !matches!(next.scheme(), "http" | "https") {
                return Err(Failure::BadRedirect);
            }
            return Ok(Hop::Redirect(next));
        }
    }
    if !status.is_success() {
        return Err(Failure::Status(status.as_u16()));
    }
    Ok(Hop::Answered(resp))
}

/// Read the body into one buffer, refusing it once it passes `max` or
/// takes the request past its total. A body that is not kept gives its
/// bytes back to the request's total.
async fn read_capped(
    mut resp: reqwest::Response,
    max: u64,
    held: &RequestBudget,
) -> Result<Vec<u8>, Failure> {
    let declared = resp.content_length();
    if declared.is_some_and(|len| len > max) {
        return Err(Failure::TooLarge);
    }
    if let Some(len) = declared {
        held.check(len)?;
    }
    let mut body = Vec::with_capacity(declared.unwrap_or(0) as usize);
    let mut claimed = 0;
    let read = async {
        while let Some(chunk) = resp.chunk().await.map_err(classify)? {
            if body.len() as u64 + chunk.len() as u64 > max {
                return Err(Failure::TooLarge);
            }
            claimed += chunk.len();
            held.claim(chunk.len())?;
            body.extend_from_slice(&chunk);
        }
        Ok(())
    }
    .await;
    match read {
        Ok(()) => Ok(body),
        Err(failure) => {
            held.release(claimed);
            Err(failure)
        }
    }
}

fn client(route: Route) -> Result<&'static reqwest::Client, Failure> {
    static PINNED: OnceLock<Result<reqwest::Client, String>> = OnceLock::new();
    static DIRECT: OnceLock<Result<reqwest::Client, String>> = OnceLock::new();
    static PROXIED: OnceLock<Result<reqwest::Client, String>> = OnceLock::new();
    let cell = match route {
        Route::Pinned => &PINNED,
        Route::Direct => &DIRECT,
        Route::Proxied => &PROXIED,
    };
    cell.get_or_init(|| {
        let b = crate::upstream_http::caller_chosen_peer_client_builder()
            .redirect(reqwest::redirect::Policy::none());
        // The route already decided whether a proxy takes the URL. The
        // proxied client keeps the shared resolver: it only resolves the
        // proxy's own host.
        let b = match route {
            Route::Pinned => b
                .dns_resolver(crate::ssrf::direct_resolver(true))
                .no_proxy(),
            Route::Direct => b
                .dns_resolver(crate::ssrf::direct_resolver(false))
                .no_proxy(),
            Route::Proxied => b,
        };
        b.build().map_err(|e| e.to_string())
    })
    .as_ref()
    .map_err(|_| Failure::Transport)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Failure {
    BadUrl,
    Blocked(IpAddr),
    Unresolvable,
    Transport,
    /// The peer's TLS certificate failed verification; carries the
    /// verifier's error for the log, never for the caller.
    UntrustedCertificate(String),
    Timeout,
    Status(u16),
    TooLarge,
    /// The request's inbound body plus its downloads passed
    /// `proxy.request_body_limit_bytes`; `downloaded` is what the downloads
    /// had reached when they did.
    OverRequestLimit {
        downloaded: u64,
    },
    TooManyRedirects,
    BadRedirect,
    UnknownType,
}

/// A failed download, and the redirect target it failed at when that is
/// not the URL the request names.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Failed {
    failure: Failure,
    redirected_to: Option<Url>,
}

impl From<Failure> for Failed {
    fn from(failure: Failure) -> Self {
        Self {
            failure,
            redirected_to: None,
        }
    }
}

impl Failed {
    /// The 400 sentence for `uri`, naming the hop that failed.
    fn describe(&self, uri: &str, settings: &MediaFetchSettings) -> String {
        let shown = display_uri(uri);
        let hop = self.redirected_to.as_ref();
        let why = match &self.failure {
            Failure::Blocked(_) => {
                let direct = Url::parse(uri).ok();
                let hint = Refusal::blocked_hint(hop.or(direct.as_ref()));
                return match hop {
                    Some(to) => format!(
                        "the gateway does not download the media URL {shown}: it redirected to \
                         {}, whose host is not a publicly routable address. {hint}",
                        display_uri(to.as_str())
                    ),
                    None => format!(
                        "the gateway does not download the media URL {shown}: its host is not \
                         a publicly routable address. {hint}"
                    ),
                };
            }
            Failure::UnknownType => {
                return format!(
                    "the media type of {shown} cannot be told from the URL or the download; \
                     set it in image_url.format (or file.format), for example \"image/png\""
                );
            }
            other => other.reason(settings),
        };
        match hop {
            Some(to) => format!(
                "could not download the media URL {shown} from {}, where it redirected: {why}",
                display_uri(to.as_str())
            ),
            None => format!("could not download the media URL {shown}: {why}"),
        }
    }
}

impl Failure {
    fn retryable(&self) -> bool {
        matches!(self, Self::Transport | Self::Timeout)
    }

    fn reason(&self, settings: &MediaFetchSettings) -> String {
        match self {
            Self::Blocked(_) | Self::UnknownType => unreachable!("described by Failed::describe"),
            Self::BadUrl => "it is not a valid URL".to_string(),
            Self::Unresolvable => "its host does not resolve".to_string(),
            Self::Transport => "the connection failed".to_string(),
            Self::UntrustedCertificate(_) => {
                "its TLS certificate could not be verified (see upstream.tls.ca_file)".to_string()
            }
            Self::Timeout => "the download timed out".to_string(),
            Self::Status(code) => format!("the server answered HTTP {code}"),
            Self::TooLarge => format!(
                "it is larger than the {}-byte limit (upstream.media_fetch.max_bytes)",
                settings.max_bytes
            ),
            // Answered as a 413 before it is described.
            Self::OverRequestLimit { .. } => format!(
                "the request would exceed its {}-byte limit",
                settings.request_body_limit_bytes
            ),
            Self::TooManyRedirects => format!("it redirects more than {MAX_REDIRECTS} times"),
            Self::BadRedirect => {
                "it redirects to a location that is not an http:// or https:// URL".to_string()
            }
        }
    }
}

impl From<Refusal> for Failure {
    fn from(r: Refusal) -> Self {
        match r {
            Refusal::BadUrl => Self::BadUrl,
            Refusal::Blocked(ip) => Self::Blocked(ip),
            Refusal::Unresolvable => Self::Unresolvable,
        }
    }
}

fn classify(err: reqwest::Error) -> Failure {
    if let Some(refusal) = Refusal::from_error(&err) {
        return refusal.into();
    }
    if let Some(cert) = certificate_error(&err) {
        return Failure::UntrustedCertificate(cert.to_string());
    }
    if err.is_timeout() {
        Failure::Timeout
    } else {
        Failure::Transport
    }
}

/// The certificate-verification error in `err`'s source chain, if any.
/// An `io::Error` hides the error it wraps from `source()`, so each one is
/// opened by hand.
fn certificate_error<'a>(err: &'a (dyn std::error::Error + 'static)) -> Option<&'a rustls::Error> {
    let mut source = Some(err);
    while let Some(e) = source {
        if let Some(wrapped) = e
            .downcast_ref::<std::io::Error>()
            .and_then(|io| io.get_ref())
        {
            source = Some(wrapped as &(dyn std::error::Error + 'static));
            continue;
        }
        if let Some(
            tls @ (rustls::Error::InvalidCertificate(_) | rustls::Error::NoCertificatesPresented),
        ) = e.downcast_ref::<rustls::Error>()
        {
            return Some(tls);
        }
        source = e.source();
    }
    None
}

/// The media type a body's first bytes announce, for the formats the
/// multimodal upstreams accept.
fn sniff(b: &[u8]) -> Option<&'static str> {
    let at = |offset: usize, magic: &[u8]| b.get(offset..offset + magic.len()) == Some(magic);
    Some(if at(0, b"\x89PNG\r\n\x1a\n") {
        "image/png"
    } else if at(0, &[0xff, 0xd8, 0xff]) {
        "image/jpeg"
    } else if at(0, b"GIF87a") || at(0, b"GIF89a") {
        "image/gif"
    } else if at(0, b"RIFF") && at(8, b"WEBP") {
        "image/webp"
    } else if at(0, b"RIFF") && at(8, b"WAVE") {
        "audio/wav"
    } else if at(0, b"%PDF-") {
        "application/pdf"
    } else if at(0, b"ID3") || at(0, &[0xff, 0xfb]) || at(0, &[0xff, 0xf3]) || at(0, &[0xff, 0xf2])
    {
        "audio/mp3"
    } else if at(0, &[0xff, 0xf1]) || at(0, &[0xff, 0xf9]) {
        "audio/aac"
    } else if at(0, b"OggS") {
        "audio/ogg"
    } else if at(0, b"fLaC") {
        "audio/flac"
    } else if at(0, &[0x1a, 0x45, 0xdf, 0xa3]) {
        "video/webm"
    } else if at(4, b"ftyp") {
        match b.get(8..12) {
            Some(b"heic" | b"heix" | b"mif1") => "image/heic",
            Some(b"M4A ") => "audio/m4a",
            Some(b"qt  ") => "video/mov",
            _ => "video/mp4",
        }
    } else {
        return None;
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    async fn dl(
        url: &Url,
        settings: &MediaFetchSettings,
    ) -> Result<(Vec<u8>, Option<String>, Url), Failure> {
        download(url, settings, &RequestBudget::unlimited())
            .await
            .map_err(|f| f.failure)
    }

    #[tokio::test]
    async fn downloads_count_against_the_request_total_and_give_back_what_they_drop() {
        let port = serve(b"HTTP/1.1 200 OK\r\nconnection: close\r\n\r\n0123456789").await;
        let url = Url::parse(&format!("http://127.0.0.1:{port}/a")).unwrap();
        let settings = with_allowed(&["127.0.0.1"]);
        // 5 inbound + 10 downloaded > 12.
        let held = RequestBudget::new(5, 12);
        let got = download(&url, &settings, &held).await;
        assert_eq!(
            got.map(|_| ()).map_err(|f| f.failure),
            Err(Failure::OverRequestLimit { downloaded: 10 })
        );
        assert_eq!(
            held.used.load(Ordering::Relaxed),
            5,
            "the dropped body is given back"
        );
        let roomy = RequestBudget::new(5, 15);
        assert!(download(&url, &settings, &roomy).await.is_ok());
        assert_eq!(roomy.used.load(Ordering::Relaxed), 15);
    }

    /// A one-shot HTTP server on loopback answering every request with
    /// `response`.
    async fn serve(response: &'static [u8]) -> u16 {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            while let Ok((mut sock, _)) = listener.accept().await {
                let mut buf = [0u8; 4096];
                let _ = sock.read(&mut buf).await;
                let _ = sock.write_all(response).await;
            }
        });
        port
    }

    fn with_allowed(hosts: &[&str]) -> MediaFetchSettings {
        MediaFetchSettings::new(
            &aisix_core::config::MediaFetchConfig {
                max_bytes: 16,
                ..Default::default()
            },
            ssrf(true, hosts),
        )
    }

    fn ssrf(enabled: bool, hosts: &[&str]) -> SsrfProtection {
        SsrfProtection::from_config(&aisix_core::config::SsrfProtectionConfig {
            enabled,
            allowed_hosts: hosts.iter().map(|h| h.to_string()).collect(),
        })
        .unwrap()
        .without_proxies()
    }

    #[tokio::test]
    async fn loopback_is_refused_by_name_and_by_literal_unless_allowed() {
        let port =
            serve(b"HTTP/1.1 200 OK\r\ncontent-length: 4\r\nconnection: close\r\n\r\nPNG!").await;
        let closed = with_allowed(&[]);
        for url in [
            format!("http://localhost:{port}/a.png"),
            format!("http://127.0.0.1:{port}/a.png"),
            format!("http://[::ffff:127.0.0.1]:{port}/a.png"),
        ] {
            let got = dl(&Url::parse(&url).unwrap(), &closed).await;
            assert!(matches!(got, Err(Failure::Blocked(_))), "{url}: {got:?}");
        }
        let open = with_allowed(&[&format!("localhost:{port}")]);
        let (body, ..) = dl(
            &Url::parse(&format!("http://localhost:{port}/a.png")).unwrap(),
            &open,
        )
        .await
        .unwrap();
        assert_eq!(body, b"PNG!");
        let other_port = with_allowed(&["localhost:1"]);
        let got = dl(
            &Url::parse(&format!("http://localhost:{port}/a.png")).unwrap(),
            &other_port,
        )
        .await;
        assert!(matches!(got, Err(Failure::Blocked(_))), "{got:?}");
    }

    #[tokio::test]
    async fn with_protection_off_a_private_address_is_fetched_and_redirects_stay_capped() {
        let port =
            serve(b"HTTP/1.1 200 OK\r\ncontent-length: 4\r\nconnection: close\r\n\r\nPNG!").await;
        let off = MediaFetchSettings::new(&Default::default(), ssrf(false, &[]));
        for url in [
            format!("http://localhost:{port}/a.png"),
            format!("http://127.0.0.1:{port}/a.png"),
        ] {
            let (body, ..) = dl(&Url::parse(&url).unwrap(), &off).await.unwrap();
            assert_eq!(body, b"PNG!", "{url}");
        }
        let redirector = serve_self_redirect().await;
        let got = dl(
            &Url::parse(&format!("http://127.0.0.1:{redirector}/a")).unwrap(),
            &off,
        )
        .await;
        assert_eq!(got.map(|_| ()), Err(Failure::TooManyRedirects));
    }

    /// A loopback server that redirects every request back to itself.
    async fn serve_self_redirect() -> u16 {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let response = format!(
            "HTTP/1.1 302 Found\r\nlocation: http://127.0.0.1:{port}/again\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
        );
        tokio::spawn(async move {
            while let Ok((mut sock, _)) = listener.accept().await {
                let mut buf = [0u8; 4096];
                let _ = sock.read(&mut buf).await;
                let _ = sock.write_all(response.as_bytes()).await;
            }
        });
        port
    }

    #[tokio::test]
    async fn a_body_past_the_cap_is_refused_whether_declared_or_not() {
        let declared = serve(
            b"HTTP/1.1 200 OK\r\ncontent-length: 17\r\nconnection: close\r\n\r\n01234567890123456",
        )
        .await;
        let chunked = serve(b"HTTP/1.1 200 OK\r\nconnection: close\r\n\r\n01234567890123456").await;
        for port in [declared, chunked] {
            let got = dl(
                &Url::parse(&format!("http://127.0.0.1:{port}/a")).unwrap(),
                &with_allowed(&["127.0.0.1"]),
            )
            .await;
            assert_eq!(got.map(|_| ()), Err(Failure::TooLarge));
        }
    }

    #[tokio::test]
    async fn a_redirect_is_checked_hop_by_hop() {
        let target =
            serve(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\nconnection: close\r\n\r\nok").await;
        let location = Box::leak(
            format!(
                "HTTP/1.1 302 Found\r\nlocation: http://127.0.0.1:{target}/b.png\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
            )
            .into_boxed_str(),
        );
        let redirector = serve(location.as_bytes()).await;
        let url = Url::parse(&format!("http://localhost:{redirector}/a.png")).unwrap();
        // The first hop is allowed; the second is a loopback literal that is not.
        let got = dl(&url, &with_allowed(&[&format!("localhost:{redirector}")])).await;
        assert!(matches!(got, Err(Failure::Blocked(_))), "{got:?}");
        let (body, _, answered) = dl(
            &url,
            &with_allowed(&[&format!("localhost:{redirector}"), "127.0.0.1"]),
        )
        .await
        .unwrap();
        assert_eq!(body, b"ok");
        assert_eq!(answered.path(), "/b.png");
    }

    /// A refusal on a later hop names that hop, not the URL the request
    /// named, and never the address the hop's name resolved to.
    #[tokio::test]
    async fn a_failure_after_a_redirect_names_the_hop_that_failed() {
        let target =
            serve(b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n")
                .await;
        let location = Box::leak(
            format!(
                "HTTP/1.1 302 Found\r\nlocation: http://localhost:{target}/b.png\r\ncontent-length: 0\r\nconnection: close\r\n\r\n"
            )
            .into_boxed_str(),
        );
        let redirector = serve(location.as_bytes()).await;
        let uri = format!("http://127.0.0.1:{redirector}/a.png?sig=secret");
        let url = Url::parse(&uri).unwrap();
        let hop = format!("\"http://localhost:{target}/b.png\"");

        let settings = with_allowed(&["127.0.0.1"]);
        let got = download(&url, &settings, &RequestBudget::unlimited())
            .await
            .unwrap_err();
        let message = got.describe(&uri, &settings);
        assert!(matches!(got.failure, Failure::Blocked(_)), "{got:?}");
        assert!(
            message.contains(&format!("redirected to {hop}")),
            "{message}"
        );
        assert!(message.contains("adds localhost to"), "{message}");
        assert!(
            !message.contains("127.0.0.1\"") && !message.contains("::1"),
            "{message}"
        );
        assert!(!message.contains("secret"), "{message}");

        let settings = with_allowed(&["127.0.0.1", "localhost"]);
        let got = download(&url, &settings, &RequestBudget::unlimited())
            .await
            .unwrap_err();
        let message = got.describe(&uri, &settings);
        assert_eq!(got.failure, Failure::Status(404));
        assert!(message.contains(&format!("from {hop}")), "{message}");

        let direct = Failed::from(Failure::Blocked("127.0.0.1".parse().unwrap()));
        let message = direct.describe(&uri, &settings);
        assert!(message.contains("adds 127.0.0.1 to"), "{message}");
        assert!(!message.contains("redirected"), "{message}");
    }

    /// A loopback https server whose self-signed certificate nobody
    /// trusts; counts the connections it accepts.
    async fn serve_untrusted_tls() -> (u16, Arc<AtomicUsize>) {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).unwrap();
        let config = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(
                vec![cert.cert.der().clone()],
                rustls::pki_types::PrivateKeyDer::Pkcs8(cert.key_pair.serialize_der().into()),
            )
            .unwrap();
        let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(config));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let accepted = Arc::new(AtomicUsize::new(0));
        let counter = accepted.clone();
        tokio::spawn(async move {
            while let Ok((sock, _)) = listener.accept().await {
                counter.fetch_add(1, Ordering::Relaxed);
                let acceptor = acceptor.clone();
                tokio::spawn(async move {
                    let _ = acceptor.accept(sock).await;
                });
            }
        });
        (port, accepted)
    }

    #[tokio::test]
    async fn an_untrusted_certificate_is_named_and_not_retried() {
        let (port, accepted) = serve_untrusted_tls().await;
        let settings = with_allowed(&["localhost"]);
        let w = Wanted {
            uri: format!("https://localhost:{port}/a.png"),
            explicit_mime_type: None,
            location: String::new(),
            uses: 1,
        };
        let got = fetch_one(&w, &settings, Encoding::Bytes, &RequestBudget::unlimited())
            .await
            .unwrap_err();
        assert!(
            matches!(got.failure, Failure::UntrustedCertificate(_)),
            "{got:?}"
        );
        assert_eq!(accepted.load(Ordering::Relaxed), 1, "not retried");
        let message = got.describe(&w.uri, &settings);
        assert!(
            message
                .ends_with("its TLS certificate could not be verified (see upstream.tls.ca_file)"),
            "{message}"
        );
    }

    #[tokio::test]
    async fn an_error_status_is_reported_not_retried() {
        let port =
            serve(b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n")
                .await;
        let got = dl(
            &Url::parse(&format!("http://127.0.0.1:{port}/a")).unwrap(),
            &with_allowed(&["127.0.0.1"]),
        )
        .await;
        assert_eq!(got.map(|_| ()), Err(Failure::Status(404)));
        assert!(!Failure::Status(404).retryable());
    }

    fn ctx() -> BridgeContext {
        use aisix_core::models::{Model, ProviderKey};
        BridgeContext::new(
            "req-1",
            Arc::new(
                serde_json::from_str::<Model>(
                    r#"{"display_name":"m","provider":"bedrock","model_name":"x","provider_key_id":"pk"}"#,
                )
                .unwrap(),
            ),
            Arc::new(serde_json::from_str::<ProviderKey>(r#"{"display_name":"pk","secret":"s"}"#).unwrap()),
        )
    }

    /// Every part naming a downloaded URL can take it, including one that
    /// would not have needed the download by itself; and the time spent is
    /// charged to the end-to-end budget only.
    #[tokio::test]
    async fn every_part_naming_a_downloaded_url_gets_it_and_the_time_is_charged() {
        let _ = SETTINGS.set(MediaFetchSettings::new(
            &Default::default(),
            ssrf(true, &["127.0.0.1"]),
        ));
        let port = serve(b"HTTP/1.1 200 OK\r\ncontent-type: image/png\r\ncontent-length: 4\r\nconnection: close\r\n\r\nPNG!").await;
        let url = format!("http://127.0.0.1:{port}/img");
        let req: ChatFormat = serde_json::from_value(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url, "format": "image/png"}},
                {"type": "image_url", "image_url": {"url": url}},
            ]}],
        }))
        .unwrap();
        let fetched = fetch_media_urls(&req, &ctx(), Encoding::Bytes, |m| {
            matches!(
                m.source,
                MediaSource::Uri {
                    explicit: false,
                    ..
                }
            )
        })
        .await
        .unwrap();
        for _ in 0..2 {
            let got = fetched.take(&url).expect("each part gets the download");
            assert_eq!(got.data.into_bytes(), b"PNG!");
        }
        assert!(fetched.take(&url).is_none());

        let streaming = ctx()
            .with_deadline(Duration::from_secs(10))
            .with_non_streaming_deadline(Some(Duration::from_secs(60)));
        let charged = fetched.charge(&streaming);
        assert_eq!(charged.deadline, Some(Duration::from_secs(10)));
        assert!(charged.non_streaming_deadline.unwrap() < Duration::from_secs(60));
        // A streaming dispatch's non-streaming leg is charged on its own budget.
        let leg = streaming.non_streaming_ctx();
        assert!(fetched.charge(&leg).deadline.unwrap() < Duration::from_secs(60));
        let plain = ctx().with_deadline(Duration::from_secs(60));
        assert!(fetched.charge(&plain).deadline.unwrap() < Duration::from_secs(60));
        assert!(matches!(
            FetchedMedia::default().charge(&plain),
            Cow::Borrowed(_)
        ));
    }

    #[test]
    fn magic_bytes_name_the_common_formats() {
        assert_eq!(sniff(b"\x89PNG\r\n\x1a\nrest"), Some("image/png"));
        assert_eq!(sniff(b"%PDF-1.7"), Some("application/pdf"));
        assert_eq!(sniff(b"RIFF\0\0\0\0WEBPVP8"), Some("image/webp"));
        assert_eq!(sniff(b"plain text"), None);
    }
}
