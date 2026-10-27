//! In-process cache of upstream access tokens the gateway obtained from a
//! token endpoint, shared by the `oauth2` client-credentials grant and the
//! `oauth2_token_exchange` grant. Each grant owns one process-global
//! [`TokenCache`] and its own key derivation; this type only holds tokens
//! and decides when one is still fit to reuse.

use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{Duration, Instant};

/// Stop reusing a cached token this long before its reported expiry, so an
/// upstream operation never starts with a token about to lapse mid-flight.
/// A token that lives no longer than this is handed out once and never
/// reused.
pub(crate) const EXPIRY_SKEW: Duration = Duration::from_secs(60);

/// Assumed lifetime when the token endpoint omits `expires_in` (RFC 6749 §5.1
/// only recommends it).
const DEFAULT_TOKEN_LIFETIME: Duration = Duration::from_secs(3600);

/// Cap on a reported `expires_in`. A hostile or broken identity provider can
/// return any u64; an absurd value would overflow `Instant + Duration` and
/// panic the connect task. Thirty days is beyond any sane token lifetime and
/// far below the overflow horizon.
const MAX_TOKEN_LIFETIME: Duration = Duration::from_secs(30 * 24 * 3600);

/// How long a token whose endpoint reported `expires_in` seconds (or
/// nothing) is held.
pub(crate) fn token_lifetime(expires_in: Option<u64>) -> Duration {
    expires_in
        .map(Duration::from_secs)
        .unwrap_or(DEFAULT_TOKEN_LIFETIME)
        .min(MAX_TOKEN_LIFETIME)
}

/// One cached upstream access token. Never printed: the struct deliberately
/// has no `Debug` impl.
struct CachedToken {
    access_token: String,
    expires_at: Instant,
}

/// Tokens by an opaque key the owning grant derives. Guards are held only
/// for map lookups and writes, never across an await.
pub(crate) struct TokenCache {
    entries: RwLock<HashMap<String, CachedToken>>,
    /// `None` leaves the map unbounded. Past `Some(n)` entries, an insert
    /// first purges the expired ones and then the ones closest to expiry.
    capacity: Option<usize>,
}

impl TokenCache {
    pub(crate) fn new(capacity: Option<usize>) -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            capacity,
        }
    }

    /// The token under `key`, unless it is within [`EXPIRY_SKEW`] of expiry.
    pub(crate) fn get(&self, key: &str) -> Option<String> {
        let entries = self
            .entries
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let entry = entries.get(key)?;
        if entry.expires_at <= Instant::now() + EXPIRY_SKEW {
            return None;
        }
        Some(entry.access_token.clone())
    }

    pub(crate) fn insert(&self, key: String, access_token: &str, lifetime: Duration) {
        let mut entries = self
            .entries
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(capacity) = self.capacity {
            if entries.len() >= capacity {
                shrink(&mut entries, capacity);
            }
        }
        entries.insert(
            key,
            CachedToken {
                access_token: access_token.to_owned(),
                expires_at: Instant::now() + lifetime,
            },
        );
    }

    /// Drop the token under `key` — called when the upstream rejects it.
    pub(crate) fn remove(&self, key: &str) {
        self.entries
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(key);
    }
}

fn shrink(entries: &mut HashMap<String, CachedToken>, capacity: usize) {
    let now = Instant::now();
    entries.retain(|_, entry| entry.expires_at > now + EXPIRY_SKEW);
    if entries.len() < capacity {
        return;
    }
    let mut by_expiry: Vec<(Instant, String)> = entries
        .iter()
        .map(|(key, entry)| (entry.expires_at, key.clone()))
        .collect();
    by_expiry.sort_unstable();
    for (_, key) in by_expiry.into_iter().take(capacity / 10 + 1) {
        entries.remove(&key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LONG: Duration = Duration::from_secs(3600);

    #[test]
    fn a_token_within_the_skew_of_expiry_is_not_reused() {
        let cache = TokenCache::new(None);
        cache.insert("fresh".into(), "a", LONG);
        cache.insert("short".into(), "b", EXPIRY_SKEW);
        assert_eq!(cache.get("fresh").as_deref(), Some("a"));
        assert_eq!(cache.get("short"), None);
        cache.remove("fresh");
        assert_eq!(cache.get("fresh"), None);
    }

    #[test]
    fn lifetime_defaults_and_is_capped() {
        assert_eq!(token_lifetime(None), DEFAULT_TOKEN_LIFETIME);
        assert_eq!(token_lifetime(Some(120)), Duration::from_secs(120));
        assert_eq!(token_lifetime(Some(u64::MAX)), MAX_TOKEN_LIFETIME);
    }

    #[test]
    fn a_full_bounded_cache_drops_expired_then_soonest_to_expire() {
        let cache = TokenCache::new(Some(10));
        cache.insert("expired".into(), "x", Duration::ZERO);
        for i in 0..9 {
            cache.insert(format!("k{i}"), "t", LONG + Duration::from_secs(i));
        }
        // Full: the expired entry makes room.
        cache.insert("k9".into(), "t", LONG + Duration::from_secs(9));
        assert_eq!(cache.entries.read().unwrap().len(), 10);
        assert!(!cache.entries.read().unwrap().contains_key("expired"));

        // Full of live tokens: the two closest to expiry go.
        cache.insert("new".into(), "t", LONG + Duration::from_secs(100));
        let entries = cache.entries.read().unwrap();
        assert_eq!(entries.len(), 9);
        assert!(!entries.contains_key("k0") && !entries.contains_key("k1"));
        assert!(entries.contains_key("k2") && entries.contains_key("new"));
    }

    #[test]
    fn an_unbounded_cache_never_evicts() {
        let cache = TokenCache::new(None);
        for i in 0..50 {
            cache.insert(format!("k{i}"), "t", Duration::ZERO);
        }
        assert_eq!(cache.entries.read().unwrap().len(), 50);
    }
}
