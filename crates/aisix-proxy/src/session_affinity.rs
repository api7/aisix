//! Session pins: which route a session of one semantic router stays on.
//!
//! Pins live in the environment's [`SharedStore`] so every gateway honours a
//! pin any of them created, behind a bounded in-process cache so a session in
//! steady state touches no store. The store is a convenience, never a
//! dependency: any failure to reach it degrades the key to a pin this process
//! alone keeps, and no request waits on it for more than [`STORE_CALL_TIMEOUT`].
//!
//! Store record (namespace [`NAMESPACE`]): key is the lowercase hex SHA-256 of
//! `<router model id> NUL <api key id> NUL <session id>`; value is the JSON
//! `{"v":1,"target_id":"…","route":"…"}`. A value with any other `v` is not
//! this build's and is neither read nor overwritten.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aisix_mcp::{PutCondition, SharedStore, StoreError, StoredValue};
use sha2::{Digest, Sha256};
use tokio::time::Instant;

pub(crate) const NAMESPACE: &str = "session_affinity";
const VALUE_VERSION: u64 = 1;
const STORE_CALL_TIMEOUT: Duration = Duration::from_secs(1);
/// After a store failure, how long the store is not consulted at all.
const STORE_BACKOFF: Duration = Duration::from_secs(30);
/// How long a key the store holds nothing for is not read again.
const MISS_TTL: Duration = Duration::from_secs(60);
const LOCAL_CAPACITY: u64 = 100_000;
static NO_LABELS: BTreeMap<&str, &str> = BTreeMap::new();

/// What a session is pinned to: the route target's model id, and the route
/// it was pinned through.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Pin {
    pub target_id: String,
    pub route: String,
}

/// A pin [`SessionPins::lookup`] found, with what [`SessionPins::touch`] and
/// [`SessionPins::discard`] need to keep the store in step.
#[derive(Debug, Clone)]
pub(crate) struct Hit {
    pub pin: Pin,
    store: Option<StoreRecord>,
}

#[derive(Debug, Clone, Copy)]
struct StoreRecord {
    version: i64,
    /// Unix seconds.
    expires_at: i64,
}

#[derive(Debug, Clone)]
enum Local {
    Pinned {
        pin: Pin,
        /// `None`: kept by this process alone.
        store: Option<StoreRecord>,
        expires: Instant,
    },
    Miss {
        expires: Instant,
    },
}

impl Local {
    fn expires(&self) -> Instant {
        match self {
            Local::Pinned { expires, .. } | Local::Miss { expires } => *expires,
        }
    }
}

struct LocalExpiry;

impl moka::Expiry<String, Local> for LocalExpiry {
    fn expire_after_create(
        &self,
        _: &String,
        value: &Local,
        _: std::time::Instant,
    ) -> Option<Duration> {
        Some(value.expires().saturating_duration_since(Instant::now()))
    }

    fn expire_after_update(
        &self,
        _: &String,
        value: &Local,
        _: std::time::Instant,
        _: Option<Duration>,
    ) -> Option<Duration> {
        Some(value.expires().saturating_duration_since(Instant::now()))
    }
}

/// The store key of one session of one router under one API key. No part of
/// the session id reaches the store in the clear.
pub(crate) fn pin_key(router_id: &str, api_key_id: &str, session_id: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(router_id.as_bytes());
    hasher.update([0]);
    hasher.update(api_key_id.as_bytes());
    hasher.update([0]);
    hasher.update(session_id.as_bytes());
    hex::encode(hasher.finalize())
}

pub struct SessionPins {
    store: Arc<dyn SharedStore>,
    local: moka::future::Cache<String, Local>,
    skip_store_until: Mutex<Option<Instant>>,
}

impl SessionPins {
    pub fn new(store: Arc<dyn SharedStore>) -> Self {
        Self {
            store,
            local: moka::future::Cache::builder()
                .max_capacity(LOCAL_CAPACITY)
                .expire_after(LocalExpiry)
                .build(),
            skip_store_until: Mutex::new(None),
        }
    }

    /// The pin `key` holds, if any.
    pub(crate) async fn lookup(&self, key: &str, ttl: Duration) -> Option<Hit> {
        if let Some(local) = self.local.get(key).await {
            if local.expires() > Instant::now() {
                return match local {
                    Local::Pinned { pin, store, .. } => Some(Hit { pin, store }),
                    Local::Miss { .. } => None,
                };
            }
        }
        let read = self.store_call(self.store.get(NAMESPACE, key)).await?;
        match read.as_ref().and_then(decode) {
            Some((pin, record)) => {
                self.remember(key, pin.clone(), Some(record), ttl).await;
                Some(Hit {
                    pin,
                    store: Some(record),
                })
            }
            None => {
                self.local
                    .insert(
                        key.to_owned(),
                        Local::Miss {
                            expires: Instant::now() + MISS_TTL,
                        },
                    )
                    .await;
                None
            }
        }
    }

    /// A request used `hit`: its pin lasts `ttl` from now. The store's copy
    /// is extended only once less than half of `ttl` remains there, off the
    /// request path.
    pub(crate) async fn touch(self: &Arc<Self>, key: &str, hit: Hit, ttl: Duration) {
        self.remember(key, hit.pin.clone(), hit.store, ttl).await;
        let Some(record) = hit.store else {
            return;
        };
        if record.expires_at - unix_now() >= (ttl.as_secs() / 2) as i64 || self.store_skipped() {
            return;
        }
        let this = Arc::clone(self);
        let key = key.to_owned();
        tokio::spawn(async move { this.refresh(key, hit.pin, record, ttl).await });
    }

    async fn refresh(&self, key: String, pin: Pin, record: StoreRecord, ttl: Duration) {
        let put = self.store.put(
            NAMESPACE,
            &key,
            encode(&pin),
            ttl,
            &NO_LABELS,
            PutCondition::IfVersion(record.version),
        );
        match tokio::time::timeout(STORE_CALL_TIMEOUT, put).await {
            Ok(Ok(version)) => {
                let record = StoreRecord {
                    version,
                    expires_at: unix_now() + ttl.as_secs() as i64,
                };
                self.remember(&key, pin, Some(record), ttl).await;
            }
            Ok(Err(StoreError::PreconditionFailed)) => {
                // Someone else wrote it since: take theirs.
                if let Some(Some((pin, record))) = self
                    .store_call(self.store.get(NAMESPACE, &key))
                    .await
                    .map(|read| read.as_ref().and_then(decode))
                {
                    self.remember(&key, pin, Some(record), ttl).await;
                }
            }
            Ok(Err(_)) | Err(_) => self.note_store_failure(),
        }
    }

    /// `hit` names a target its router no longer routes to: forget it here
    /// and in the store, so the session can be pinned afresh.
    pub(crate) async fn discard(&self, key: &str, hit: &Hit) {
        self.local.invalidate(key).await;
        if let Some(record) = hit.store {
            self.store_call(self.store.delete(NAMESPACE, key, Some(record.version)))
                .await;
        }
    }

    /// Pin `key` to `pin` unless another request got there first, and return
    /// the pin this request must use: the first writer's when it is one
    /// `usable` accepts, else `pin`.
    pub(crate) async fn create(
        &self,
        key: &str,
        pin: Pin,
        ttl: Duration,
        usable: impl Fn(&Pin) -> bool,
    ) -> Pin {
        let put = self.store.put(
            NAMESPACE,
            key,
            encode(&pin),
            ttl,
            &NO_LABELS,
            PutCondition::CreateOnly,
        );
        match self.store_call(put).await {
            Some(version) => {
                let record = StoreRecord {
                    version,
                    expires_at: unix_now() + ttl.as_secs() as i64,
                };
                self.remember(key, pin.clone(), Some(record), ttl).await;
                pin
            }
            None if self.store_skipped() => {
                self.remember(key, pin.clone(), None, ttl).await;
                pin
            }
            None => {
                // The create lost: someone holds the key.
                let winner = self
                    .store_call(self.store.get(NAMESPACE, key))
                    .await
                    .flatten()
                    .as_ref()
                    .and_then(decode)
                    .filter(|(winner, _)| usable(winner));
                match winner {
                    Some((winner, record)) => {
                        self.remember(key, winner.clone(), Some(record), ttl).await;
                        winner
                    }
                    None => {
                        self.remember(key, pin.clone(), None, ttl).await;
                        pin
                    }
                }
            }
        }
    }

    async fn remember(&self, key: &str, pin: Pin, store: Option<StoreRecord>, ttl: Duration) {
        self.local
            .insert(
                key.to_owned(),
                Local::Pinned {
                    pin,
                    store,
                    expires: Instant::now() + ttl,
                },
            )
            .await;
    }

    /// Run one store call within [`STORE_CALL_TIMEOUT`]. `None` when the
    /// store is being skipped, failed, or — for a write — its precondition
    /// did not hold; only the last leaves the store in use.
    async fn store_call<T>(
        &self,
        call: impl std::future::Future<Output = Result<T, StoreError>>,
    ) -> Option<T> {
        if self.store_skipped() {
            return None;
        }
        match tokio::time::timeout(STORE_CALL_TIMEOUT, call).await {
            Ok(Ok(value)) => Some(value),
            Ok(Err(StoreError::PreconditionFailed)) => None,
            Ok(Err(error)) => {
                tracing::debug!(%error, "session affinity store call failed; pinning locally");
                self.note_store_failure();
                None
            }
            Err(_) => {
                tracing::debug!("session affinity store call timed out; pinning locally");
                self.note_store_failure();
                None
            }
        }
    }

    fn store_skipped(&self) -> bool {
        let guard = self
            .skip_store_until
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        guard.is_some_and(|until| Instant::now() < until)
    }

    fn note_store_failure(&self) {
        *self
            .skip_store_until
            .lock()
            .unwrap_or_else(|p| p.into_inner()) = Some(Instant::now() + STORE_BACKOFF);
    }
}

fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64)
}

fn encode(pin: &Pin) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "v": VALUE_VERSION,
        "target_id": pin.target_id,
        "route": pin.route,
    }))
    .expect("a JSON object of strings encodes")
}

fn decode(stored: &StoredValue) -> Option<(Pin, StoreRecord)> {
    let value: serde_json::Value = serde_json::from_slice(&stored.value).ok()?;
    if value.get("v")?.as_u64()? != VALUE_VERSION {
        return None;
    }
    let pin = Pin {
        target_id: value.get("target_id")?.as_str()?.to_owned(),
        route: value.get("route")?.as_str()?.to_owned(),
    };
    // A backend that does not say when the record expires gets it extended
    // on the next use.
    let expires_at = stored.expires_at.unwrap_or_else(unix_now);
    Some((
        pin,
        StoreRecord {
            version: stored.version,
            expires_at,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_mcp::MemoryStore;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    /// A [`MemoryStore`] that counts calls and can be taken down.
    #[derive(Default)]
    struct Probe {
        inner: MemoryStore,
        gets: AtomicUsize,
        puts: AtomicUsize,
        down: AtomicBool,
    }

    #[async_trait]
    impl SharedStore for Probe {
        async fn get(&self, ns: &str, key: &str) -> Result<Option<StoredValue>, StoreError> {
            self.gets.fetch_add(1, Ordering::SeqCst);
            if self.down.load(Ordering::SeqCst) {
                return Err(StoreError::Unavailable);
            }
            self.inner.get(ns, key).await
        }
        async fn put(
            &self,
            ns: &str,
            key: &str,
            value: Vec<u8>,
            ttl: Duration,
            labels: &BTreeMap<&str, &str>,
            condition: PutCondition,
        ) -> Result<i64, StoreError> {
            self.puts.fetch_add(1, Ordering::SeqCst);
            if self.down.load(Ordering::SeqCst) {
                return Err(StoreError::Unavailable);
            }
            self.inner.put(ns, key, value, ttl, labels, condition).await
        }
        async fn delete(&self, ns: &str, key: &str, v: Option<i64>) -> Result<(), StoreError> {
            self.inner.delete(ns, key, v).await
        }
    }

    const TTL: Duration = Duration::from_secs(600);

    fn pin(target: &str) -> Pin {
        Pin {
            target_id: target.into(),
            route: format!("route-{target}"),
        }
    }

    fn setup() -> (Arc<Probe>, Arc<SessionPins>) {
        let probe = Arc::new(Probe::default());
        let pins = Arc::new(SessionPins::new(probe.clone()));
        (probe, pins)
    }

    #[test]
    fn key_hashes_all_three_parts() {
        let k = pin_key("router", "key", "session");
        assert_eq!(k.len(), 64);
        assert!(!k.contains("session"));
        assert_ne!(k, pin_key("router", "key2", "session"));
        assert_ne!(k, pin_key("router2", "key", "session"));
        assert_ne!(pin_key("a", "bc", "s"), pin_key("ab", "c", "s"));
        assert_eq!(
            k,
            hex::encode(Sha256::digest(b"router\0key\0session")),
            "the key spelling is a contract with the store's owner"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_miss_is_read_once_per_minute() {
        let (probe, pins) = setup();
        assert!(pins.lookup("k", TTL).await.is_none());
        assert!(pins.lookup("k", TTL).await.is_none());
        assert_eq!(probe.gets.load(Ordering::SeqCst), 1);
        tokio::time::advance(MISS_TTL + Duration::from_secs(1)).await;
        assert!(pins.lookup("k", TTL).await.is_none());
        assert_eq!(probe.gets.load(Ordering::SeqCst), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn another_gateway_reads_the_pin_and_a_local_hit_reads_nothing() {
        let (probe, a) = setup();
        let b = Arc::new(SessionPins::new(probe.clone()));
        assert_eq!(a.create("k", pin("x"), TTL, |_| true).await, pin("x"));
        let hit = b.lookup("k", TTL).await.expect("b reads a's pin");
        assert_eq!(hit.pin, pin("x"));
        let gets = probe.gets.load(Ordering::SeqCst);
        assert!(b.lookup("k", TTL).await.is_some());
        assert_eq!(probe.gets.load(Ordering::SeqCst), gets);
    }

    #[tokio::test(start_paused = true)]
    async fn a_lost_create_takes_the_winner() {
        let (probe, a) = setup();
        let b = Arc::new(SessionPins::new(probe.clone()));
        a.create("k", pin("x"), TTL, |_| true).await;
        assert_eq!(b.create("k", pin("y"), TTL, |_| true).await, pin("x"));
        assert_eq!(b.lookup("k", TTL).await.unwrap().pin, pin("x"));
        // A winner the caller cannot use leaves its own pin, locally.
        let c = Arc::new(SessionPins::new(probe.clone()));
        assert_eq!(
            c.create("k", pin("z"), TTL, |p| p.target_id != "x").await,
            pin("z")
        );
    }

    #[tokio::test(start_paused = true)]
    async fn an_unknown_value_version_is_absent_and_kept() {
        let (probe, pins) = setup();
        let foreign = br#"{"v":2,"target_id":"x","route":"r"}"#.to_vec();
        probe
            .inner
            .put(
                NAMESPACE,
                "k",
                foreign.clone(),
                TTL,
                &NO_LABELS,
                PutCondition::CreateOnly,
            )
            .await
            .unwrap();
        assert!(pins.lookup("k", TTL).await.is_none());
        assert_eq!(pins.create("k", pin("y"), TTL, |_| true).await, pin("y"));
        let stored = probe.inner.get(NAMESPACE, "k").await.unwrap().unwrap();
        assert_eq!(stored.value, foreign);
    }

    #[tokio::test(start_paused = true)]
    async fn the_local_pin_slides_and_expires() {
        let (probe, pins) = setup();
        probe.down.store(true, Ordering::SeqCst);
        pins.create("k", pin("x"), TTL, |_| true).await;
        for _ in 0..3 {
            tokio::time::advance(TTL - Duration::from_secs(1)).await;
            let hit = pins.lookup("k", TTL).await.expect("used within its ttl");
            pins.touch("k", hit, TTL).await;
        }
        tokio::time::advance(TTL + Duration::from_secs(1)).await;
        assert!(pins.lookup("k", TTL).await.is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn a_store_failure_pins_locally_and_skips_the_store_for_30s() {
        let (probe, pins) = setup();
        probe.down.store(true, Ordering::SeqCst);
        assert_eq!(pins.create("k", pin("x"), TTL, |_| true).await, pin("x"));
        assert_eq!(pins.lookup("k", TTL).await.unwrap().pin, pin("x"));
        assert!(pins.lookup("other", TTL).await.is_none());
        pins.create("other2", pin("y"), TTL, |_| true).await;
        assert_eq!(probe.puts.load(Ordering::SeqCst), 1);
        assert_eq!(probe.gets.load(Ordering::SeqCst), 0);
        probe.down.store(false, Ordering::SeqCst);
        tokio::time::advance(STORE_BACKOFF + Duration::from_secs(1)).await;
        assert!(pins.lookup("other3", TTL).await.is_none());
        assert_eq!(probe.gets.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn a_hung_store_costs_at_most_one_second() {
        struct Hung;
        #[async_trait]
        impl SharedStore for Hung {
            async fn get(&self, _: &str, _: &str) -> Result<Option<StoredValue>, StoreError> {
                std::future::pending().await
            }
            async fn put(
                &self,
                _: &str,
                _: &str,
                _: Vec<u8>,
                _: Duration,
                _: &BTreeMap<&str, &str>,
                _: PutCondition,
            ) -> Result<i64, StoreError> {
                std::future::pending().await
            }
            async fn delete(&self, _: &str, _: &str, _: Option<i64>) -> Result<(), StoreError> {
                std::future::pending().await
            }
        }
        let pins = SessionPins::new(Arc::new(Hung));
        let start = Instant::now();
        assert!(pins.lookup("k", TTL).await.is_none());
        assert_eq!(start.elapsed(), STORE_CALL_TIMEOUT);
        // Skipped from here on: no further wait.
        assert_eq!(pins.create("k", pin("x"), TTL, |_| true).await, pin("x"));
        assert_eq!(start.elapsed(), STORE_CALL_TIMEOUT);
        assert_eq!(pins.lookup("k", TTL).await.unwrap().pin, pin("x"));
    }

    #[tokio::test(start_paused = true)]
    async fn the_store_copy_is_extended_only_past_half_its_ttl() {
        let (probe, pins) = setup();
        pins.create("k", pin("x"), TTL, |_| true).await;
        assert_eq!(probe.puts.load(Ordering::SeqCst), 1);
        let hit = pins.lookup("k", TTL).await.unwrap();
        pins.touch("k", hit, TTL).await;
        tokio::task::yield_now().await;
        assert_eq!(probe.puts.load(Ordering::SeqCst), 1, "fresh: no refresh");

        // A record read with less than half its TTL left is extended.
        let short = Duration::from_secs(200);
        probe
            .inner
            .put(
                NAMESPACE,
                "j",
                encode(&pin("y")),
                short,
                &NO_LABELS,
                PutCondition::CreateOnly,
            )
            .await
            .unwrap();
        let hit = pins.lookup("j", TTL).await.unwrap();
        pins.touch("j", hit, TTL).await;
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
        assert_eq!(probe.puts.load(Ordering::SeqCst), 2, "one refresh");
        let stored = probe.inner.get(NAMESPACE, "j").await.unwrap().unwrap();
        assert!(stored.expires_at.unwrap() - unix_now() > 500);
    }

    #[tokio::test(start_paused = true)]
    async fn discard_lets_the_session_be_pinned_afresh() {
        let (probe, a) = setup();
        a.create("k", pin("x"), TTL, |_| true).await;
        let hit = a.lookup("k", TTL).await.unwrap();
        a.discard("k", &hit).await;
        assert_eq!(a.create("k", pin("y"), TTL, |_| true).await, pin("y"));
        let b = SessionPins::new(probe.clone());
        assert_eq!(b.lookup("k", TTL).await.unwrap().pin, pin("y"));
    }
}
