//! Per-resource cache of parsed upstream endpoint URLs.
//!
//! Every bridge dispatch used to rebuild its endpoint URL per request —
//! base resolution (trim + suffix scan + allocation), a `format!`, and a
//! full `Url` parse inside reqwest's `IntoUrl` when the request builder
//! receives a string. The URL is a pure function of the owning resource's
//! configuration and the endpoint, so it is parsed once here and handed
//! back as a cloned `Url` — which reqwest accepts by value without
//! re-parsing.
//!
//! Correctness model: a cached row stores the *fingerprint* of the inputs
//! the URL was built from (the raw `api_base` plus whatever else the
//! bridge derives the URL from). A hit whose fingerprint differs from the
//! caller's current inputs rebuilds, so an edited resource takes effect on
//! the next request with no invalidation hook. A resource id names one
//! etcd resource, so a deleted-and-recreated resource either reuses the id
//! (fingerprint decides) or leaves a dead row behind (bounded by
//! [`MAX_RESOURCES`]).
//!
//! Two row shapes, because two kinds of URL exist:
//!
//! - **Per endpoint** ([`cached_endpoint_url`]) — the URL depends only on
//!   the resource's configuration. One row per endpoint.
//! - **Per endpoint and upstream model** ([`cached_model_endpoint_url`]) —
//!   the URL embeds the upstream model id, as the Vertex model path and the
//!   Azure deployment segment do. These *must* use the model-keyed form:
//!   folding the model into the fingerprint instead would give one row per
//!   endpoint that every model in turn invalidates, so a key serving more
//!   than one model would miss and rebuild on every request — worse than
//!   not caching at all.

use dashmap::DashMap;
use reqwest::Url;
use std::sync::{Arc, OnceLock};

/// Outer map: resource id → that resource's endpoint URLs. Two levels so
/// the hit path looks up by `&str` (no per-request key allocation).
type Cache = DashMap<String, Arc<EndpointRows>>;

/// One resource's rows, keyed by the endpoint literal or, for
/// model-dependent URLs, by `endpoint \x1f upstream_model`.
type EndpointRows = DashMap<Box<str>, CachedUrl>;

static URL_CACHE: OnceLock<Cache> = OnceLock::new();

struct CachedUrl {
    /// The inputs this URL was built from; compared on every hit.
    fingerprint: Box<[Box<str>]>,
    url: Url,
}

/// Joins the endpoint and the upstream model into one row key. A control
/// byte no upstream model id contains, so two (endpoint, model) pairs
/// cannot collide on one row; a model that does contain it skips the
/// cache rather than risk aliasing.
const KEY_SEP: char = '\x1f';

/// Upper bound on distinct resource ids the cache will hold. Rows for
/// deleted resources are never individually evicted (matching the TLS
/// client cache's model); this cap turns unbounded admin-churn growth
/// into a full reset, after which live resources re-parse once each.
const MAX_RESOURCES: usize = 8192;

/// Upper bound on rows for one resource. Endpoints are a fixed literal
/// set, but the model-keyed rows scale with the models configured against
/// one Provider Key, so this bounds the worst case the same way.
const MAX_ROWS_PER_RESOURCE: usize = 1024;

/// A ready-to-post endpoint URL.
#[derive(Clone, Debug)]
pub enum EndpointUrl {
    /// Parsed once (possibly cached); reqwest takes `Url` by value
    /// without re-parsing.
    Parsed(Url),
    /// The built string failed to parse. Handed back verbatim so the
    /// request builder produces exactly the error it always produced
    /// for a malformed `api_base`; never cached.
    Unparsed(String),
}

impl EndpointUrl {
    /// Start a POST to this URL on `client`.
    pub fn post_on(self, client: &reqwest::Client) -> reqwest::RequestBuilder {
        match self {
            Self::Parsed(url) => client.post(url),
            Self::Unparsed(raw) => client.post(raw),
        }
    }
}

/// Resolve the parsed URL for (`resource_id`, `endpoint`), building it
/// with `build` only on the first request or after the resource's
/// configuration changed.
///
/// - `endpoint` must be a bridge-namespaced literal (e.g.
///   `"openai/chat"`) so two bridges can never collide on one resource.
/// - `fingerprint` is every raw input the URL is derived from — the
///   `api_base` and, where applicable, the vendor, the project, the
///   region. An input left out is an edit the cache will not notice.
/// - An empty `resource_id` (contexts built without snapshot ids, e.g.
///   tests) bypasses the cache entirely.
/// - `build` errors pass through untouched; nothing is cached on error.
pub fn cached_endpoint_url<E>(
    resource_id: &str,
    endpoint: &'static str,
    fingerprint: &[&str],
    build: impl FnOnce() -> Result<String, E>,
) -> Result<EndpointUrl, E> {
    let Some(rows) = rows_for(resource_id) else {
        return Ok(parse_or_raw(build()?));
    };
    resolve(&rows, endpoint, fingerprint, build)
}

/// [`cached_endpoint_url`] for a URL that also embeds the request's
/// upstream model id — a Vertex `publishers/…/models/<model>:method`
/// path, an Azure `deployments/<deployment>/…` segment.
///
/// `upstream_model` is the operator-configured upstream id
/// (`model.model_name`), not a caller-supplied string, so the row set is
/// bounded by the configuration rather than by traffic.
pub fn cached_model_endpoint_url<E>(
    resource_id: &str,
    endpoint: &'static str,
    upstream_model: &str,
    fingerprint: &[&str],
    build: impl FnOnce() -> Result<String, E>,
) -> Result<EndpointUrl, E> {
    let Some(rows) = rows_for(resource_id) else {
        return Ok(parse_or_raw(build()?));
    };
    if upstream_model.contains(KEY_SEP) {
        // Cannot be keyed without risking a collision with another
        // (endpoint, model) pair. Same URL, uncached.
        return Ok(parse_or_raw(build()?));
    }
    ROW_KEY.with(|buf| {
        let mut buf = buf.borrow_mut();
        buf.clear();
        buf.push_str(endpoint);
        buf.push(KEY_SEP);
        buf.push_str(upstream_model);
        resolve(&rows, &buf, fingerprint, build)
    })
}

thread_local! {
    /// Reused row-key buffer, so the model-keyed hit path allocates
    /// nothing; only a miss allocates the stored `Box<str>`.
    static ROW_KEY: std::cell::RefCell<String> =
        const { std::cell::RefCell::new(String::new()) };
}

/// Parse `raw` once, for a caller whose URL *is* a configured string with
/// nothing derived from the request — an A2A agent's JSON-RPC endpoint.
///
/// No resource id and no fingerprint: the string is its own identity, so
/// an edited agent simply lands on a different row and a renamed or
/// deleted one leaves a dead row behind, bounded by [`MAX_PARSED_URLS`].
/// Callers whose URL is *built* from configuration want
/// [`cached_endpoint_url`] instead, which revalidates against the inputs.
pub fn cached_url(raw: &str) -> EndpointUrl {
    static PARSED: OnceLock<DashMap<Box<str>, Url>> = OnceLock::new();
    let cache = PARSED.get_or_init(DashMap::new);
    if let Some(hit) = cache.get(raw) {
        return EndpointUrl::Parsed(hit.clone());
    }
    let Ok(url) = Url::parse(raw) else {
        return EndpointUrl::Unparsed(raw.to_string());
    };
    if cache.len() >= MAX_PARSED_URLS {
        cache.clear();
    }
    cache.insert(Box::from(raw), url.clone());
    EndpointUrl::Parsed(url)
}

/// Upper bound on [`cached_url`] rows, on the same reasoning as
/// [`MAX_RESOURCES`].
const MAX_PARSED_URLS: usize = 8192;

/// This resource's rows, creating them on first sight. `None` means the
/// caller passed no id and wants the cache bypassed.
fn rows_for(resource_id: &str) -> Option<Arc<EndpointRows>> {
    if resource_id.is_empty() {
        return None;
    }
    let cache = URL_CACHE.get_or_init(DashMap::new);
    // Clone the inner Arc out of the outer guard so no outer shard lock
    // is held while reading, building, or parsing.
    if let Some(rows) = cache.get(resource_id) {
        return Some(Arc::clone(&rows));
    }
    // First sighting of this resource: bound the outer map before
    // inserting. `len()` walks shards, but only resource creation
    // (admin-rate) pays it.
    if cache.len() >= MAX_RESOURCES {
        cache.clear();
    }
    Some(
        cache
            .entry(resource_id.to_string())
            .or_insert_with(|| Arc::new(DashMap::new()))
            .clone(),
    )
}

/// Parse without caching (bypass paths).
fn parse_or_raw(raw: String) -> EndpointUrl {
    match Url::parse(&raw) {
        Ok(url) => EndpointUrl::Parsed(url),
        Err(_) => EndpointUrl::Unparsed(raw),
    }
}

fn fingerprint_matches(stored: &[Box<str>], current: &[&str]) -> bool {
    stored.len() == current.len() && stored.iter().zip(current).all(|(a, b)| a.as_ref() == *b)
}

fn own(fingerprint: &[&str]) -> Box<[Box<str>]> {
    fingerprint.iter().map(|s| Box::from(*s)).collect()
}

/// Hit-check under a read guard, and on miss build/parse/cache under the
/// row's write guard — single-flight per row: that guard is held across
/// the build, so concurrent cold requests (or requests racing a
/// fingerprint change) serialize and the build closure runs once per
/// transition. The closure is pure CPU (string assembly), so holding the
/// shard lock across it is bounded and touches no other lock.
fn resolve<E>(
    rows: &EndpointRows,
    row: &str,
    fingerprint: &[&str],
    build: impl FnOnce() -> Result<String, E>,
) -> Result<EndpointUrl, E> {
    if let Some(hit) = rows.get(row) {
        if fingerprint_matches(&hit.fingerprint, fingerprint) {
            return Ok(EndpointUrl::Parsed(hit.url.clone()));
        }
    }
    // Miss or stale row. Bound this resource's rows here, where no guard
    // is held and where only a row transition (not a hit) pays `len()`.
    if rows.len() > MAX_ROWS_PER_RESOURCE {
        rows.clear();
    }
    match rows.entry(Box::from(row)) {
        dashmap::mapref::entry::Entry::Occupied(mut occupied) => {
            // Re-checked under the write guard: another thread may have
            // rebuilt this row since the read above.
            if fingerprint_matches(&occupied.get().fingerprint, fingerprint) {
                return Ok(EndpointUrl::Parsed(occupied.get().url.clone()));
            }
            let raw = build()?;
            match Url::parse(&raw) {
                Ok(url) => {
                    occupied.insert(CachedUrl {
                        fingerprint: own(fingerprint),
                        url: url.clone(),
                    });
                    Ok(EndpointUrl::Parsed(url))
                }
                Err(_) => Ok(EndpointUrl::Unparsed(raw)),
            }
        }
        dashmap::mapref::entry::Entry::Vacant(vacant) => {
            let raw = build()?;
            match Url::parse(&raw) {
                Ok(url) => {
                    vacant.insert(CachedUrl {
                        fingerprint: own(fingerprint),
                        url: url.clone(),
                    });
                    Ok(EndpointUrl::Parsed(url))
                }
                Err(_) => Ok(EndpointUrl::Unparsed(raw)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn build_counted<'a>(
        counter: &'a AtomicUsize,
        url: &str,
    ) -> impl FnOnce() -> Result<String, ()> + 'a {
        let url = url.to_string();
        move || {
            counter.fetch_add(1, Ordering::Relaxed);
            Ok(url)
        }
    }

    #[test]
    fn caches_per_key_and_endpoint_until_fingerprint_changes() {
        let n = AtomicUsize::new(0);
        let fp = ["https://api.example.com", "vendor-a"];

        // First call builds…
        let u1 = cached_endpoint_url(
            "pk-url-1",
            "test/chat",
            &fp,
            build_counted(&n, "https://api.example.com/v1/chat"),
        )
        .unwrap();
        assert!(matches!(u1, EndpointUrl::Parsed(_)));
        assert_eq!(n.load(Ordering::Relaxed), 1);

        // …second call with the same fingerprint hits.
        let u2 = cached_endpoint_url(
            "pk-url-1",
            "test/chat",
            &fp,
            build_counted(&n, "https://api.example.com/v1/chat"),
        )
        .unwrap();
        match (u1, u2) {
            (EndpointUrl::Parsed(a), EndpointUrl::Parsed(b)) => assert_eq!(a, b),
            _ => panic!("expected parsed URLs"),
        }
        assert_eq!(n.load(Ordering::Relaxed), 1, "second call must not rebuild");

        // A different endpoint on the same key builds its own row.
        cached_endpoint_url(
            "pk-url-1",
            "test/embeddings",
            &fp,
            build_counted(&n, "https://api.example.com/v1/embeddings"),
        )
        .unwrap();
        assert_eq!(n.load(Ordering::Relaxed), 2);

        // An edited api_base (fingerprint change) rebuilds on next use.
        let u4 = cached_endpoint_url(
            "pk-url-1",
            "test/chat",
            &["https://eu.example.com", "vendor-a"],
            build_counted(&n, "https://eu.example.com/v1/chat"),
        )
        .unwrap();
        assert_eq!(n.load(Ordering::Relaxed), 3);
        match u4 {
            EndpointUrl::Parsed(u) => assert_eq!(u.as_str(), "https://eu.example.com/v1/chat"),
            _ => panic!("expected parsed URL"),
        }

        // A fingerprint that gained a component is a change too — a
        // prefix comparison would call this a hit.
        cached_endpoint_url(
            "pk-url-1",
            "test/chat",
            &["https://eu.example.com", "vendor-a", "project-x"],
            build_counted(&n, "https://eu.example.com/v1/chat"),
        )
        .unwrap();
        assert_eq!(n.load(Ordering::Relaxed), 4);
    }

    /// The failure the Azure deployment audit found on the first
    /// version of this cache: when the URL embeds the upstream model,
    /// one row per endpoint means every model invalidates the previous
    /// one, so a key serving several models rebuilds on every request.
    #[test]
    fn model_keyed_rows_do_not_evict_each_other() {
        let n = AtomicUsize::new(0);
        let fp = ["https://vertex.example.com", "proj", "us-central1"];

        for _ in 0..3 {
            for model in ["gemini-2.0-flash", "gemini-1.5-pro"] {
                cached_model_endpoint_url(
                    "pk-vertex-1",
                    "test/generate",
                    model,
                    &fp,
                    build_counted(&n, &format!("https://vertex.example.com/{model}:generate")),
                )
                .unwrap();
            }
        }
        assert_eq!(
            n.load(Ordering::Relaxed),
            2,
            "one build per model, not one per request",
        );

        // Each model still gets its own URL back.
        let u = cached_model_endpoint_url(
            "pk-vertex-1",
            "test/generate",
            "gemini-1.5-pro",
            &fp,
            build_counted(&n, "https://wrong.example.com/should-not-build"),
        )
        .unwrap();
        match u {
            EndpointUrl::Parsed(u) => {
                assert_eq!(
                    u.as_str(),
                    "https://vertex.example.com/gemini-1.5-pro:generate"
                )
            }
            _ => panic!("expected parsed URL"),
        }

        // A model-keyed row and a plain row on the same endpoint name
        // are distinct rows, not one row two callers fight over.
        cached_endpoint_url(
            "pk-vertex-1",
            "test/generate",
            &fp,
            build_counted(&n, "https://vertex.example.com/shim"),
        )
        .unwrap();
        assert_eq!(n.load(Ordering::Relaxed), 3);
    }

    /// An edited `api_base` has to reach the model-keyed rows too, or a
    /// re-pointed Provider Key keeps dispatching to the old host for
    /// every model that was already warm.
    #[test]
    fn model_keyed_rows_follow_a_fingerprint_change() {
        let n = AtomicUsize::new(0);
        let build = |host: &str| format!("https://{host}/gemini-2.0-flash:generate");

        cached_model_endpoint_url(
            "pk-vertex-2",
            "test/generate",
            "gemini-2.0-flash",
            &["https://a.example.com", "proj", "us-central1"],
            build_counted(&n, &build("a.example.com")),
        )
        .unwrap();
        let after_edit = cached_model_endpoint_url(
            "pk-vertex-2",
            "test/generate",
            "gemini-2.0-flash",
            &["https://b.example.com", "proj", "us-central1"],
            build_counted(&n, &build("b.example.com")),
        )
        .unwrap();
        assert_eq!(n.load(Ordering::Relaxed), 2);
        match after_edit {
            EndpointUrl::Parsed(u) => assert_eq!(u.host_str(), Some("b.example.com")),
            _ => panic!("expected parsed URL"),
        }
    }

    /// A model id carrying the row separator must not be able to answer
    /// for a different (endpoint, model) pair.
    #[test]
    fn a_model_id_containing_the_separator_bypasses_the_cache() {
        let n = AtomicUsize::new(0);
        for _ in 0..2 {
            cached_model_endpoint_url(
                "pk-sep",
                "test/generate",
                "evil\u{1f}other",
                &["https://x.example.com"],
                build_counted(&n, "https://x.example.com/evil"),
            )
            .unwrap();
        }
        assert_eq!(n.load(Ordering::Relaxed), 2, "must not be cached");
    }

    #[test]
    fn empty_key_bypasses_cache_and_malformed_url_stays_raw() {
        let n = AtomicUsize::new(0);
        for _ in 0..2 {
            cached_endpoint_url("", "test/chat", &["b"], build_counted(&n, "https://x/y")).unwrap();
            cached_model_endpoint_url(
                "",
                "test/chat",
                "m",
                &["b"],
                build_counted(&n, "https://x/y"),
            )
            .unwrap();
        }
        assert_eq!(n.load(Ordering::Relaxed), 4, "empty id must never cache");

        // Malformed URL: handed back raw (reqwest will surface its usual
        // builder error), and never cached.
        let bad = cached_endpoint_url("pk-url-2", "test/chat", &["b"], || {
            Ok::<_, ()>("not a url".to_string())
        })
        .unwrap();
        match bad {
            EndpointUrl::Unparsed(raw) => assert_eq!(raw, "not a url"),
            _ => panic!("malformed URL must stay raw"),
        }
        let again = cached_endpoint_url("pk-url-2", "test/chat", &["b"], || {
            Ok::<_, ()>("https://fixed.example.com/v1".to_string())
        })
        .unwrap();
        assert!(
            matches!(again, EndpointUrl::Parsed(_)),
            "bad URL was cached"
        );
    }

    #[test]
    fn concurrent_cold_misses_build_once() {
        // Single-flight: the row's shard guard is held across the
        // build, so N racing cold requests produce exactly one build.
        let n = AtomicUsize::new(0);
        let barrier = std::sync::Barrier::new(8);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    barrier.wait();
                    let u = cached_endpoint_url("pk-url-sf", "test/chat", &["b"], || {
                        n.fetch_add(1, Ordering::Relaxed);
                        Ok::<_, ()>("https://sf.example.com/v1".to_string())
                    })
                    .unwrap();
                    assert!(matches!(u, EndpointUrl::Parsed(_)));
                });
            }
        });
        assert_eq!(n.load(Ordering::Relaxed), 1, "one build across 8 racers");
    }

    #[test]
    fn build_errors_pass_through_and_cache_nothing() {
        let r: Result<EndpointUrl, &'static str> =
            cached_endpoint_url("pk-url-3", "test/chat", &["b"], || Err("boom"));
        assert_eq!(r.err(), Some("boom"));
        let ok = cached_endpoint_url("pk-url-3", "test/chat", &["b"], || {
            Ok::<_, &'static str>("https://ok.example.com/v1".to_string())
        })
        .unwrap();
        assert!(matches!(ok, EndpointUrl::Parsed(_)));
    }

    /// One resource's rows cannot grow without bound: a key configured
    /// with more models than the cap resets rather than accumulating.
    #[test]
    fn rows_for_one_resource_are_bounded() {
        let n = AtomicUsize::new(0);
        for i in 0..(MAX_ROWS_PER_RESOURCE + 2) {
            cached_model_endpoint_url(
                "pk-bound",
                "test/generate",
                &format!("model-{i}"),
                &["https://x.example.com"],
                build_counted(&n, "https://x.example.com/m"),
            )
            .unwrap();
        }
        let rows = rows_for("pk-bound").expect("rows");
        assert!(
            rows.len() <= MAX_ROWS_PER_RESOURCE,
            "rows grew to {} past the {MAX_ROWS_PER_RESOURCE} cap",
            rows.len(),
        );
    }
}
