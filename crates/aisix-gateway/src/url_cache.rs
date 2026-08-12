//! Per-ProviderKey cache of parsed upstream endpoint URLs.
//!
//! Every bridge dispatch used to rebuild its endpoint URL per request —
//! base resolution (trim + suffix scan + allocation), a `format!`, and a
//! full `Url` parse inside reqwest's `IntoUrl` when the request builder
//! receives a string. The URL is a pure function of the ProviderKey's
//! configuration and the endpoint, so it is parsed once here and handed
//! back as a cloned `Url` — which reqwest accepts by value without
//! re-parsing.
//!
//! Correctness model: a cached row stores the *fingerprint* of the
//! inputs the URL was built from (the raw `api_base` plus a per-bridge
//! second component such as the vendor or the Azure deployment). A hit
//! whose fingerprint differs from the caller's current inputs rebuilds,
//! so an edited ProviderKey takes effect on the next request with no
//! invalidation hook. A ProviderKey id names one etcd resource, so a
//! deleted-and-recreated key either reuses the id (fingerprint decides)
//! or leaves a dead row behind (bounded by [`MAX_PROVIDER_KEYS`]).

use dashmap::DashMap;
use reqwest::Url;
use std::sync::{Arc, OnceLock};

/// Outer map: ProviderKey id → that key's endpoint URLs. Two levels so
/// the hit path looks up by `&str` (no per-request key allocation).
type Cache = DashMap<String, Arc<DashMap<&'static str, CachedUrl>>>;

static URL_CACHE: OnceLock<Cache> = OnceLock::new();

struct CachedUrl {
    /// The inputs this URL was built from; compared on every hit.
    fingerprint: (String, String),
    url: Url,
}

/// Upper bound on distinct ProviderKey ids the cache will hold. Rows for
/// deleted keys are never individually evicted (matching the TLS client
/// cache's model); this cap turns unbounded admin-churn growth into a
/// full reset, after which live keys re-parse once each.
const MAX_PROVIDER_KEYS: usize = 8192;

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

/// Resolve the parsed URL for (`provider_key_id`, `endpoint`), building
/// it with `build` only on the first request or after the key's
/// configuration changed.
///
/// - `endpoint` must be a bridge-namespaced literal (e.g.
///   `"openai/chat"`) so two bridges can never collide on one key.
/// - `fingerprint` is the pair of raw inputs the URL is derived from;
///   pass every input that can change the result (`api_base` and, where
///   applicable, vendor or deployment).
/// - An empty `provider_key_id` (contexts built without snapshot ids,
///   e.g. tests) bypasses the cache entirely.
/// - `build` errors pass through untouched; nothing is cached on error.
pub fn cached_endpoint_url<E>(
    provider_key_id: &str,
    endpoint: &'static str,
    fingerprint: (&str, &str),
    build: impl FnOnce() -> Result<String, E>,
) -> Result<EndpointUrl, E> {
    if provider_key_id.is_empty() {
        return Ok(parse_or_raw(build()?));
    }
    let cache = URL_CACHE.get_or_init(DashMap::new);

    // Clone the inner Arc out of the outer guard so no outer shard lock
    // is held while reading, building, or parsing.
    let known = cache.get(provider_key_id).map(|g| Arc::clone(&g));
    if let Some(per_key) = known {
        if let Some(hit) = per_key.get(endpoint) {
            if hit.fingerprint.0 == fingerprint.0 && hit.fingerprint.1 == fingerprint.1 {
                return Ok(EndpointUrl::Parsed(hit.url.clone()));
            }
        }
        return build_into(&per_key, endpoint, fingerprint, build);
    }

    // First sighting of this key: bound the outer map before inserting.
    // `len()` walks shards, but only key-creation (admin-rate) pays it.
    if cache.len() >= MAX_PROVIDER_KEYS {
        cache.clear();
    }
    let per_key = cache
        .entry(provider_key_id.to_string())
        .or_insert_with(|| Arc::new(DashMap::new()))
        .clone();
    build_into(&per_key, endpoint, fingerprint, build)
}

/// Parse without caching (empty-id bypass path).
fn parse_or_raw(raw: String) -> EndpointUrl {
    match Url::parse(&raw) {
        Ok(url) => EndpointUrl::Parsed(url),
        Err(_) => EndpointUrl::Unparsed(raw),
    }
}

/// Build, parse, and (on parse success) cache the URL for `endpoint`.
fn build_into<E>(
    per_key: &DashMap<&'static str, CachedUrl>,
    endpoint: &'static str,
    fingerprint: (&str, &str),
    build: impl FnOnce() -> Result<String, E>,
) -> Result<EndpointUrl, E> {
    let raw = build()?;
    match Url::parse(&raw) {
        Ok(url) => {
            per_key.insert(
                endpoint,
                CachedUrl {
                    fingerprint: (fingerprint.0.to_string(), fingerprint.1.to_string()),
                    url: url.clone(),
                },
            );
            Ok(EndpointUrl::Parsed(url))
        }
        Err(_) => Ok(EndpointUrl::Unparsed(raw)),
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
        let fp = ("https://api.example.com", "vendor-a");

        // First call builds…
        let u1 = cached_endpoint_url(
            "pk-url-1",
            "test/chat",
            fp,
            build_counted(&n, "https://api.example.com/v1/chat"),
        )
        .unwrap();
        assert!(matches!(u1, EndpointUrl::Parsed(_)));
        assert_eq!(n.load(Ordering::Relaxed), 1);

        // …second call with the same fingerprint hits.
        let u2 = cached_endpoint_url(
            "pk-url-1",
            "test/chat",
            fp,
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
            fp,
            build_counted(&n, "https://api.example.com/v1/embeddings"),
        )
        .unwrap();
        assert_eq!(n.load(Ordering::Relaxed), 2);

        // An edited api_base (fingerprint change) rebuilds on next use.
        let u4 = cached_endpoint_url(
            "pk-url-1",
            "test/chat",
            ("https://eu.example.com", "vendor-a"),
            build_counted(&n, "https://eu.example.com/v1/chat"),
        )
        .unwrap();
        assert_eq!(n.load(Ordering::Relaxed), 3);
        match u4 {
            EndpointUrl::Parsed(u) => assert_eq!(u.as_str(), "https://eu.example.com/v1/chat"),
            _ => panic!("expected parsed URL"),
        }
    }

    #[test]
    fn empty_key_bypasses_cache_and_malformed_url_stays_raw() {
        let n = AtomicUsize::new(0);
        for _ in 0..2 {
            cached_endpoint_url("", "test/chat", ("b", ""), build_counted(&n, "https://x/y"))
                .unwrap();
        }
        assert_eq!(n.load(Ordering::Relaxed), 2, "empty id must never cache");

        // Malformed URL: handed back raw (reqwest will surface its usual
        // builder error), and never cached.
        let bad = cached_endpoint_url("pk-url-2", "test/chat", ("b", ""), || {
            Ok::<_, ()>("not a url".to_string())
        })
        .unwrap();
        match bad {
            EndpointUrl::Unparsed(raw) => assert_eq!(raw, "not a url"),
            _ => panic!("malformed URL must stay raw"),
        }
        let again = cached_endpoint_url("pk-url-2", "test/chat", ("b", ""), || {
            Ok::<_, ()>("https://fixed.example.com/v1".to_string())
        })
        .unwrap();
        assert!(
            matches!(again, EndpointUrl::Parsed(_)),
            "bad URL was cached"
        );
    }

    #[test]
    fn build_errors_pass_through_and_cache_nothing() {
        let r: Result<EndpointUrl, &'static str> =
            cached_endpoint_url("pk-url-3", "test/chat", ("b", ""), || Err("boom"));
        assert_eq!(r.err(), Some("boom"));
        let ok = cached_endpoint_url("pk-url-3", "test/chat", ("b", ""), || {
            Ok::<_, &'static str>("https://ok.example.com/v1".to_string())
        })
        .unwrap();
        assert!(matches!(ok, EndpointUrl::Parsed(_)));
    }
}
