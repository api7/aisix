//! Downloading a chat request's media URL for an upstream that cannot read
//! the URL itself, so the bytes can travel inline instead.
//!
//! The URL is the caller's, and the download runs from inside the
//! operator's network, so every address it connects to must be publicly
//! routable. The name is resolved here, every address the answer holds is
//! checked, and the connection goes to exactly those addresses — never to
//! a second lookup that a rebinding name could answer differently. That
//! holds for `http` and `https` alike, because the check lives in the
//! client's resolver; the host name still rides `Host` and SNI. Redirects
//! are followed by hand so each hop passes the same check.
//! `upstream.media_fetch.allowed_hosts` is the only way past it.

use std::borrow::Cow;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::Duration;

use base64::Engine as _;
use futures::StreamExt;
use reqwest::dns::{Addrs, Name, Resolve, Resolving};
use reqwest::{header, Url};
use tokio::time::Instant;

use crate::bridge::{BridgeContext, BridgeError};
use crate::chat::{ChatFormat, Role};
use crate::content_part::{
    display_uri, mime_type_from_path, normalize_mime_type, parse_part, part_error, ContentPart,
    Media, MediaSource,
};

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
    allowed_hosts: Vec<AllowedHost>,
    max_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AllowedHost {
    host: String,
    port: Option<u16>,
}

impl MediaFetchSettings {
    pub fn from_config(cfg: &aisix_core::config::MediaFetchConfig) -> Result<Self, String> {
        let allowed_hosts = cfg
            .allowed_hosts
            .iter()
            .map(|entry| {
                parse_allowed_host(entry).ok_or_else(|| {
                    format!(
                        "upstream.media_fetch.allowed_hosts: {entry:?} is not a host or \
                         host:port (write an IPv6 address with a port as [addr]:port)"
                    )
                })
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            enabled: cfg.enabled,
            allowed_hosts,
            max_bytes: cfg.max_bytes,
        })
    }

    fn allows(&self, host: &str, port: u16) -> bool {
        self.allowed_hosts
            .iter()
            .any(|a| a.host == host && a.port.is_none_or(|p| p == port))
    }
}

fn parse_allowed_host(entry: &str) -> Option<AllowedHost> {
    let entry = entry.trim();
    if entry.is_empty() || entry.contains(['/', '@', '?', '#']) {
        return None;
    }
    let port = |p: &str| p.parse::<u16>().ok();
    if let Some(rest) = entry.strip_prefix('[') {
        let (addr, tail) = rest.split_once(']')?;
        let host = addr.parse::<Ipv6Addr>().ok()?.to_string();
        let port = match tail {
            "" => None,
            tail => Some(port(tail.strip_prefix(':')?)?),
        };
        return Some(AllowedHost { host, port });
    }
    if let Ok(v6) = entry.parse::<Ipv6Addr>() {
        return Some(AllowedHost {
            host: v6.to_string(),
            port: None,
        });
    }
    let (host, port) = match entry.rsplit_once(':') {
        Some((host, p)) => (host, Some(port(p)?)),
        None => (entry, None),
    };
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    (!host.is_empty()).then_some(AllowedHost { host, port })
}

static SETTINGS: OnceLock<MediaFetchSettings> = OnceLock::new();

/// Install `upstream.media_fetch`. Called once during boot; fails on an
/// `allowed_hosts` entry that does not parse.
pub fn init(cfg: &aisix_core::config::MediaFetchConfig) -> Result<(), String> {
    let _ = SETTINGS.set(MediaFetchSettings::from_config(cfg)?);
    Ok(())
}

fn settings() -> &'static MediaFetchSettings {
    SETTINGS.get_or_init(|| {
        MediaFetchSettings::from_config(&Default::default()).expect("the default config parses")
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
    message: usize,
    part: usize,
    uses: usize,
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
    let settings = settings();
    if !settings.enabled {
        return Ok(FetchedMedia::default());
    }
    let budget = ctx.non_streaming_deadline.or(ctx.deadline);
    let mut wanted: Vec<Wanted> = Vec::new();
    let mut seen: HashMap<&str, usize> = HashMap::new();
    // Every part naming a URL takes the download from the result, whether
    // or not that part alone would have needed it, so all are counted.
    let mut occurrences: HashMap<&str, usize> = HashMap::new();
    // System turns are included: the Anthropic translation carries a
    // system turn that follows the conversation as a user turn.
    for (message, m) in req.messages.iter().enumerate() {
        if !matches!(m.role, Role::User | Role::System | Role::Developer) {
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
            match seen.entry(uri) {
                Entry::Occupied(_) => {}
                Entry::Vacant(at) => {
                    at.insert(wanted.len());
                    wanted.push(Wanted {
                        uri: uri.to_string(),
                        explicit_mime_type: mime_type.clone().filter(|_| *explicit),
                        message,
                        part,
                        uses: 0,
                    });
                }
            }
        }
    }
    if wanted.is_empty() {
        return Ok(FetchedMedia::default());
    }
    for w in &mut wanted {
        w.uses = occurrences.get(w.uri.as_str()).copied().unwrap_or(1);
    }

    let started = Instant::now();
    let deadline = budget.map(|b| started + b);
    let mut downloads = futures::stream::iter(wanted.into_iter().map(|w| async move {
        let download = fetch_one(&w, settings, encoding);
        let result = match deadline {
            Some(at) => tokio::time::timeout_at(at, download)
                .await
                .unwrap_or(Err(Failure::Timeout)),
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
            Err(failure) => {
                tracing::warn!(
                    request_id = %ctx.request_id,
                    url = %display_uri(&w.uri),
                    failure = ?failure,
                    "media URL not downloaded"
                );
                return Err(BridgeError::InvalidRequest(part_error(
                    w.message,
                    w.part,
                    &failure.describe(&w.uri, settings),
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
) -> Result<Fetched, Failure> {
    let url = Url::parse(&w.uri).map_err(|_| Failure::BadUrl)?;
    let mut attempt = 1;
    let (bytes, content_type, final_url) = loop {
        match download(&url, settings).await {
            Err(f) if f.retryable() && attempt < MAX_ATTEMPTS => {
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
) -> Result<(Vec<u8>, Option<String>, Url), Failure> {
    let mut current = url.clone();
    for hop in 0..=MAX_REDIRECTS {
        let guarded = check_target(&current, settings)?;
        let resp = client(guarded)?
            .get(current.clone())
            .send()
            .await
            .map_err(classify)?;
        let status = resp.status();
        if status.is_redirection() {
            if let Some(location) = resp.headers().get(header::LOCATION) {
                if hop == MAX_REDIRECTS {
                    return Err(Failure::TooManyRedirects);
                }
                let next = location
                    .to_str()
                    .ok()
                    .and_then(|l| current.join(l).ok())
                    .ok_or(Failure::BadRedirect)?;
                if !matches!(next.scheme(), "http" | "https") {
                    return Err(Failure::BadRedirect);
                }
                current = next;
                continue;
            }
        }
        if !status.is_success() {
            return Err(Failure::Status(status.as_u16()));
        }
        let content_type = resp
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|v| normalize_mime_type(v.split(';').next().unwrap_or("")))
            .filter(|m| {
                !m.is_empty() && m != "application/octet-stream" && !m.starts_with("binary/")
            });
        let body = read_capped(resp, settings.max_bytes).await?;
        return Ok((body, content_type, current));
    }
    Err(Failure::TooManyRedirects)
}

/// Read the body into one buffer, refusing it once it passes `max`.
async fn read_capped(mut resp: reqwest::Response, max: u64) -> Result<Vec<u8>, Failure> {
    let declared = resp.content_length();
    if declared.is_some_and(|len| len > max) {
        return Err(Failure::TooLarge);
    }
    let mut body = Vec::with_capacity(declared.unwrap_or(0) as usize);
    while let Some(chunk) = resp.chunk().await.map_err(classify)? {
        if body.len() as u64 + chunk.len() as u64 > max {
            return Err(Failure::TooLarge);
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

/// Whether `url` may be requested, and through which client: `true` for
/// the one whose resolver refuses non-public addresses, `false` for an
/// allow-listed host.
fn check_target(url: &Url, settings: &MediaFetchSettings) -> Result<bool, Failure> {
    let (host, ip) = match url.host() {
        Some(url::Host::Domain(d)) => (d.trim_end_matches('.').to_ascii_lowercase(), None),
        Some(url::Host::Ipv4(a)) => (a.to_string(), Some(IpAddr::V4(a))),
        Some(url::Host::Ipv6(a)) => (a.to_string(), Some(IpAddr::V6(a))),
        None => return Err(Failure::BadUrl),
    };
    let port = url.port_or_known_default().ok_or(Failure::BadUrl)?;
    if settings.allows(&host, port) {
        return Ok(false);
    }
    // The resolver never sees an address literal, so it is checked here.
    if let Some(ip) = ip.filter(|ip| !is_public(*ip)) {
        return Err(Failure::Blocked(ip));
    }
    Ok(true)
}

fn client(guarded: bool) -> Result<&'static reqwest::Client, Failure> {
    static GUARDED: OnceLock<Result<reqwest::Client, String>> = OnceLock::new();
    static ALLOWED: OnceLock<Result<reqwest::Client, String>> = OnceLock::new();
    let cell = if guarded { &GUARDED } else { &ALLOWED };
    cell.get_or_init(|| {
        crate::upstream_http::caller_chosen_peer_client_builder()
            .dns_resolver(Arc::new(MediaResolver { guarded }))
            .redirect(reqwest::redirect::Policy::none())
            // A proxy would resolve the name itself, past the check.
            .no_proxy()
            .build()
            .map_err(|e| e.to_string())
    })
    .as_ref()
    .map_err(|_| Failure::Transport)
}

/// Resolves a name and, when `guarded`, refuses the whole answer if any
/// address in it is not publicly routable. The client connects only to
/// the addresses returned here.
struct MediaResolver {
    guarded: bool,
}

impl Resolve for MediaResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let guarded = self.guarded;
        Box::pin(async move {
            let addrs: Vec<SocketAddr> = tokio::net::lookup_host((name.as_str(), 0))
                .await
                .map_err(|_| Box::new(Unresolvable) as Box<dyn std::error::Error + Send + Sync>)?
                .collect();
            if addrs.is_empty() {
                return Err(Box::new(Unresolvable) as _);
            }
            if guarded {
                if let Some(ip) = addrs.iter().map(SocketAddr::ip).find(|ip| !is_public(*ip)) {
                    return Err(Box::new(BlockedAddress(ip)) as _);
                }
            }
            Ok(Box::new(addrs.into_iter()) as Addrs)
        })
    }
}

#[derive(Debug)]
struct BlockedAddress(IpAddr);

impl std::fmt::Display for BlockedAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} is not a publicly routable address", self.0)
    }
}

impl std::error::Error for BlockedAddress {}

#[derive(Debug)]
struct Unresolvable;

impl std::fmt::Display for Unresolvable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the host does not resolve")
    }
}

impl std::error::Error for Unresolvable {}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Failure {
    BadUrl,
    Blocked(IpAddr),
    Unresolvable,
    Transport,
    Timeout,
    Status(u16),
    TooLarge,
    TooManyRedirects,
    BadRedirect,
    UnknownType,
}

impl Failure {
    fn retryable(&self) -> bool {
        matches!(self, Self::Transport | Self::Timeout)
    }

    fn describe(&self, uri: &str, settings: &MediaFetchSettings) -> String {
        let shown = display_uri(uri);
        let why = match self {
            Self::Blocked(_) => {
                return format!(
                    "the gateway does not download the media URL {shown}: its host is not \
                     a publicly routable address. To allow it, an operator adds the host to \
                     upstream.media_fetch.allowed_hosts"
                );
            }
            Self::UnknownType => {
                return format!(
                    "the media type of {shown} cannot be told from the URL or the download; \
                     set it in image_url.format (or file.format), for example \"image/png\""
                );
            }
            Self::BadUrl => "it is not a valid URL".to_string(),
            Self::Unresolvable => "its host does not resolve".to_string(),
            Self::Transport => "the connection failed".to_string(),
            Self::Timeout => "the download timed out".to_string(),
            Self::Status(code) => format!("the server answered HTTP {code}"),
            Self::TooLarge => format!(
                "it is larger than the {}-byte limit (upstream.media_fetch.max_bytes)",
                settings.max_bytes
            ),
            Self::TooManyRedirects => format!("it redirects more than {MAX_REDIRECTS} times"),
            Self::BadRedirect => {
                "it redirects to a location that is not an http:// or https:// URL".to_string()
            }
        };
        format!("could not download the media URL {shown}: {why}")
    }
}

fn classify(err: reqwest::Error) -> Failure {
    let mut source: Option<&(dyn std::error::Error + 'static)> = Some(&err);
    while let Some(e) = source {
        if let Some(BlockedAddress(ip)) = e.downcast_ref::<BlockedAddress>() {
            return Failure::Blocked(*ip);
        }
        if e.downcast_ref::<Unresolvable>().is_some() {
            return Failure::Unresolvable;
        }
        source = e.source();
    }
    if err.is_timeout() {
        Failure::Timeout
    } else {
        Failure::Transport
    }
}

/// Whether an address is one the download may reach without an
/// allow-list entry: globally routable unicast, and not a cloud
/// platform's own service address.
fn is_public(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_public_v4(v4),
        IpAddr::V6(v6) => match v6.to_ipv4_mapped() {
            Some(v4) => is_public_v4(v4),
            None => is_public_v6(v6),
        },
    }
}

fn is_public_v4(a: Ipv4Addr) -> bool {
    let [o0, o1, o2, _] = a.octets();
    let blocked = o0 == 0 // "this network", incl. 0.0.0.0
        || a.is_loopback()
        || a.is_private()
        || (o0 == 100 && (o1 & 0xc0) == 64) // CGNAT 100.64.0.0/10
        || a.is_link_local() // 169.254.0.0/16, incl. instance metadata
        || (o0 == 192 && o1 == 0 && o2 == 0) // IETF protocol assignments
        || a.is_documentation()
        || (o0 == 198 && (o1 & 0xfe) == 18) // benchmarking 198.18.0.0/15
        || (o0 == 192 && o1 == 88 && o2 == 99) // 6to4 relay anycast
        || a.is_multicast()
        || o0 >= 240 // reserved, incl. broadcast
        || a == Ipv4Addr::new(168, 63, 129, 16); // Azure platform endpoint
    !blocked
}

fn is_public_v6(a: Ipv6Addr) -> bool {
    let s = a.segments();
    let embedded = |hi: u16, lo: u16| Ipv4Addr::from(((hi as u32) << 16) | lo as u32);
    if a.is_unspecified() || a.is_loopback() || a.is_multicast() {
        return false;
    }
    match s {
        // IPv4-compatible (deprecated) ::/96
        [0, 0, 0, 0, 0, 0, _, _] => false,
        // NAT64 64:ff9b::/96 reaches the IPv4 address it embeds.
        [0x64, 0xff9b, 0, 0, 0, 0, hi, lo] => is_public_v4(embedded(hi, lo)),
        // Local-use NAT64 64:ff9b:1::/48
        [0x64, 0xff9b, 1, ..] => false,
        // Discard-only 100::/64
        [0x100, 0, 0, 0, ..] => false,
        // IETF protocol assignments 2001::/23 (Teredo included), documentation 2001:db8::/32
        [0x2001, b, ..] if b < 0x200 || b == 0xdb8 => false,
        // 6to4 2002::/16 reaches the IPv4 address it embeds.
        [0x2002, hi, lo, ..] => is_public_v4(embedded(hi, lo)),
        // Documentation 3fff::/20
        [0x3fff, b, ..] if b < 0x1000 => false,
        // Unique-local fc00::/7, link-local fe80::/10, site-local fec0::/10
        [first, ..]
            if (first & 0xfe00) == 0xfc00
                || (first & 0xffc0) == 0xfe80
                || (first & 0xffc0) == 0xfec0 =>
        {
            false
        }
        _ => true,
    }
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
    use super::*;

    #[test]
    fn only_globally_routable_addresses_are_public() {
        for blocked in [
            "0.0.0.0",
            "127.0.0.1",
            "10.1.2.3",
            "172.16.0.1",
            "192.168.1.1",
            "100.64.0.1",
            "100.127.255.255",
            "169.254.169.254",
            "192.0.0.8",
            "192.0.2.1",
            "198.18.0.1",
            "198.51.100.1",
            "203.0.113.1",
            "224.0.0.1",
            "255.255.255.255",
            "168.63.129.16",
            "::",
            "::1",
            "::ffff:127.0.0.1",
            "::ffff:169.254.169.254",
            "64:ff9b::a00:1",
            "2002:a00:1::",
            "2001:db8::1",
            "fd00::1",
            "fe80::1",
            "ff02::1",
        ] {
            let ip: IpAddr = blocked.parse().unwrap();
            assert!(!is_public(ip), "{blocked} must be refused");
        }
        for public in [
            "1.1.1.1",
            "8.8.8.8",
            "100.128.0.1",
            "::ffff:8.8.8.8",
            "2606:4700:4700::1111",
            "64:ff9b::808:808",
        ] {
            let ip: IpAddr = public.parse().unwrap();
            assert!(is_public(ip), "{public} must be allowed");
        }
    }

    #[test]
    fn allowed_hosts_match_host_and_optional_port() {
        let settings = MediaFetchSettings::from_config(&aisix_core::config::MediaFetchConfig {
            allowed_hosts: vec![
                "Media.Internal".into(),
                "127.0.0.1:9000".into(),
                "[fd00::1]:8080".into(),
            ],
            ..Default::default()
        })
        .unwrap();
        assert!(settings.allows("media.internal", 443));
        assert!(settings.allows("127.0.0.1", 9000));
        assert!(!settings.allows("127.0.0.1", 9001));
        assert!(settings.allows("fd00::1", 8080));
        for bad in ["", "http://x", "a:b", "[::1", "user@host"] {
            assert!(parse_allowed_host(bad).is_none(), "{bad:?}");
        }
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
        MediaFetchSettings::from_config(&aisix_core::config::MediaFetchConfig {
            allowed_hosts: hosts.iter().map(|h| h.to_string()).collect(),
            max_bytes: 16,
            ..Default::default()
        })
        .unwrap()
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
            let got = download(&Url::parse(&url).unwrap(), &closed).await;
            assert!(matches!(got, Err(Failure::Blocked(_))), "{url}: {got:?}");
        }
        let open = with_allowed(&[&format!("localhost:{port}")]);
        let (body, ..) = download(
            &Url::parse(&format!("http://localhost:{port}/a.png")).unwrap(),
            &open,
        )
        .await
        .unwrap();
        assert_eq!(body, b"PNG!");
        let other_port = with_allowed(&["localhost:1"]);
        let got = download(
            &Url::parse(&format!("http://localhost:{port}/a.png")).unwrap(),
            &other_port,
        )
        .await;
        assert!(matches!(got, Err(Failure::Blocked(_))), "{got:?}");
    }

    #[tokio::test]
    async fn a_body_past_the_cap_is_refused_whether_declared_or_not() {
        let declared = serve(
            b"HTTP/1.1 200 OK\r\ncontent-length: 17\r\nconnection: close\r\n\r\n01234567890123456",
        )
        .await;
        let chunked = serve(b"HTTP/1.1 200 OK\r\nconnection: close\r\n\r\n01234567890123456").await;
        for port in [declared, chunked] {
            let got = download(
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
        let got = download(&url, &with_allowed(&[&format!("localhost:{redirector}")])).await;
        assert!(matches!(got, Err(Failure::Blocked(_))), "{got:?}");
        let (body, _, answered) = download(
            &url,
            &with_allowed(&[&format!("localhost:{redirector}"), "127.0.0.1"]),
        )
        .await
        .unwrap();
        assert_eq!(body, b"ok");
        assert_eq!(answered.path(), "/b.png");
    }

    #[tokio::test]
    async fn an_error_status_is_reported_not_retried() {
        let port =
            serve(b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n")
                .await;
        let got = download(
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
        init(&aisix_core::config::MediaFetchConfig {
            allowed_hosts: vec!["127.0.0.1".into()],
            ..Default::default()
        })
        .unwrap();
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
