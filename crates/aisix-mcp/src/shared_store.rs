//! Storage shared by every gateway of one environment: small values with a
//! TTL and a version, written with a create-only or compare-and-set
//! precondition.
//!
//! Two backends implement [`SharedStore`]: the dp-manager's store API in
//! managed mode (`aisix-server`), and [`MemoryStore`] wherever there is no
//! dp-manager. The environment is the backend's concern, never a parameter.
//!
//! Values include upstream tokens and the key sessions are sealed with, in the
//! clear: encrypting them at rest and restricting who can read them is the
//! backend's job.
//!
//! Every write states a precondition, so a writer can always tell whether it
//! won. [`consume`] builds an atomic single-use read out of that.

use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use async_trait::async_trait;

/// A stored value and the version a later compare-and-set names.
#[derive(Clone, PartialEq, Eq)]
pub struct StoredValue {
    pub value: Vec<u8>,
    pub version: i64,
    /// When the record's TTL runs out, in Unix seconds, if the backend said.
    pub expires_at: Option<i64>,
}

// By hand: a value here may be a token.
impl std::fmt::Debug for StoredValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StoredValue")
            .field("value", &"***redacted***")
            .field("version", &self.version)
            .field("expires_at", &self.expires_at)
            .finish()
    }
}

/// The precondition of a [`SharedStore::put`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PutCondition {
    /// Fail unless the key holds nothing.
    CreateOnly,
    /// Fail unless the key holds exactly this version.
    IfVersion(i64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum StoreError {
    /// The precondition did not hold. The only outcome of a write that says
    /// someone else got there first.
    #[error("precondition failed")]
    PreconditionFailed,
    #[error("value too large")]
    TooLarge,
    #[error("over quota")]
    OverQuota,
    /// The backend could not answer, or answered something this gateway does
    /// not understand. Says nothing about what the key holds.
    #[error("storage unavailable")]
    Unavailable,
}

#[async_trait]
pub trait SharedStore: Send + Sync {
    async fn get(&self, namespace: &str, key: &str) -> Result<Option<StoredValue>, StoreError>;

    /// Write `value`, returning its new version. `labels` let the owner of the
    /// backend find a record's owners without reading its value.
    async fn put(
        &self,
        namespace: &str,
        key: &str,
        value: Vec<u8>,
        ttl: Duration,
        labels: &BTreeMap<&str, &str>,
        condition: PutCondition,
    ) -> Result<i64, StoreError>;

    /// Remove `key`; removing nothing succeeds. With `if_version`, a key at
    /// another version may be left in place.
    async fn delete(
        &self,
        namespace: &str,
        key: &str,
        if_version: Option<i64>,
    ) -> Result<(), StoreError>;
}

/// What a consumed key holds until the delete that follows lands.
const CONSUMED: &[u8] = b"consumed";
const CONSUMED_TTL: Duration = Duration::from_secs(60);

/// Read `key` so that, of any number of concurrent callers on any number of
/// gateways, exactly one gets the value. `Ok(None)` is everyone else, as well
/// as a key that holds nothing.
///
/// The value is replaced by compare-and-set rather than deleted: a delete is
/// idempotent, so two readers of one version would both see it succeed.
pub async fn consume(
    store: &dyn SharedStore,
    namespace: &str,
    key: &str,
) -> Result<Option<Vec<u8>>, StoreError> {
    let Some(stored) = store.get(namespace, key).await? else {
        return Ok(None);
    };
    if stored.value == CONSUMED {
        return Ok(None);
    }
    match store
        .put(
            namespace,
            key,
            CONSUMED.to_vec(),
            CONSUMED_TTL,
            &BTreeMap::new(),
            PutCondition::IfVersion(stored.version),
        )
        .await
    {
        Ok(_) => {}
        Err(StoreError::PreconditionFailed) => return Ok(None),
        Err(error) => return Err(error),
    }
    // Best effort: the tombstone expires on its own.
    let _ = store.delete(namespace, key, None).await;
    Ok(Some(stored.value))
}

/// Past this many live entries in one namespace a write answers
/// [`StoreError::OverQuota`]. Per namespace so that a flood of one kind of record
/// cannot take the room that signed-in users' tokens and leases need.
///
/// Sized for spent refresh tokens, the namespace that grows fastest: one record
/// per hour per connected client, each kept for up to 14 days.
const MEMORY_STORE_MAX_ENTRIES_PER_NAMESPACE: usize = 200_000;

/// The backend for a gateway with no dp-manager. Its contents live in this
/// process alone: a restart loses every sign-in, and replicas do not share
/// them.
#[derive(Default)]
pub struct MemoryStore {
    state: Mutex<MemoryState>,
}

#[derive(Default)]
struct MemoryState {
    namespaces: HashMap<String, HashMap<String, MemoryEntry>>,
    next_version: i64,
}

struct MemoryEntry {
    value: Vec<u8>,
    version: i64,
    expires_at: Instant,
}

impl MemoryStore {
    fn state(&self) -> std::sync::MutexGuard<'_, MemoryState> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }
}

#[async_trait]
impl SharedStore for MemoryStore {
    async fn get(&self, namespace: &str, key: &str) -> Result<Option<StoredValue>, StoreError> {
        let state = self.state();
        Ok(state
            .namespaces
            .get(namespace)
            .and_then(|entries| entries.get(key))
            .filter(|entry| entry.expires_at > Instant::now())
            .map(|entry| StoredValue {
                value: entry.value.clone(),
                version: entry.version,
                expires_at: Some(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map_or(0, |now| now.as_secs() as i64)
                        + entry
                            .expires_at
                            .saturating_duration_since(Instant::now())
                            .as_secs() as i64,
                ),
            }))
    }

    async fn put(
        &self,
        namespace: &str,
        key: &str,
        value: Vec<u8>,
        ttl: Duration,
        _labels: &BTreeMap<&str, &str>,
        condition: PutCondition,
    ) -> Result<i64, StoreError> {
        let mut state = self.state();
        let MemoryState {
            namespaces,
            next_version,
        } = &mut *state;
        let now = Instant::now();
        let entries = namespaces.entry(namespace.to_owned()).or_default();
        let live = entries
            .get(key)
            .filter(|entry| entry.expires_at > now)
            .map(|entry| entry.version);
        let holds = match condition {
            PutCondition::CreateOnly => live.is_none(),
            PutCondition::IfVersion(version) => live == Some(version),
        };
        if !holds {
            return Err(StoreError::PreconditionFailed);
        }
        if live.is_none() && entries.len() >= MEMORY_STORE_MAX_ENTRIES_PER_NAMESPACE {
            entries.retain(|_, entry| entry.expires_at > now);
            if entries.len() >= MEMORY_STORE_MAX_ENTRIES_PER_NAMESPACE {
                return Err(StoreError::OverQuota);
            }
        }
        *next_version += 1;
        entries.insert(
            key.to_owned(),
            MemoryEntry {
                value,
                version: *next_version,
                expires_at: now + ttl,
            },
        );
        Ok(*next_version)
    }

    async fn delete(
        &self,
        namespace: &str,
        key: &str,
        if_version: Option<i64>,
    ) -> Result<(), StoreError> {
        let mut state = self.state();
        if let Some(entries) = state.namespaces.get_mut(namespace) {
            let matches = if_version.is_none_or(|v| entries.get(key).map(|e| e.version) == Some(v));
            if matches {
                entries.remove(key);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TTL: Duration = Duration::from_secs(60);

    async fn create(store: &MemoryStore, key: &str, value: &str) -> Result<i64, StoreError> {
        store
            .put(
                "ns",
                key,
                value.as_bytes().to_vec(),
                TTL,
                &BTreeMap::new(),
                PutCondition::CreateOnly,
            )
            .await
    }

    #[tokio::test]
    async fn preconditions_decide_who_writes() {
        let store = MemoryStore::default();
        let version = create(&store, "k", "a").await.unwrap();
        assert_eq!(
            create(&store, "k", "b").await,
            Err(StoreError::PreconditionFailed)
        );
        let stale = store
            .put(
                "ns",
                "k",
                b"b".to_vec(),
                TTL,
                &BTreeMap::new(),
                PutCondition::IfVersion(version + 1),
            )
            .await;
        assert_eq!(stale, Err(StoreError::PreconditionFailed));
        store
            .put(
                "ns",
                "k",
                b"b".to_vec(),
                TTL,
                &BTreeMap::new(),
                PutCondition::IfVersion(version),
            )
            .await
            .unwrap();
        let read = store.get("ns", "k").await.unwrap().unwrap();
        assert_eq!(read.value, b"b");
        assert_ne!(read.version, version);
    }

    #[tokio::test]
    async fn an_expired_entry_holds_nothing() {
        let store = MemoryStore::default();
        store
            .put(
                "ns",
                "k",
                b"a".to_vec(),
                Duration::ZERO,
                &BTreeMap::new(),
                PutCondition::CreateOnly,
            )
            .await
            .unwrap();
        assert!(store.get("ns", "k").await.unwrap().is_none());
        create(&store, "k", "b").await.unwrap();
    }

    #[tokio::test]
    async fn a_full_namespace_leaves_the_others_writable() {
        let store = MemoryStore::default();
        let put = |ns: &'static str, key: String| {
            let store = &store;
            async move {
                store
                    .put(
                        ns,
                        &key,
                        Vec::new(),
                        TTL,
                        &BTreeMap::new(),
                        PutCondition::CreateOnly,
                    )
                    .await
            }
        };
        for i in 0..MEMORY_STORE_MAX_ENTRIES_PER_NAMESPACE {
            put("flows", i.to_string()).await.unwrap();
        }
        assert_eq!(
            put("flows", "one-more".to_owned()).await,
            Err(StoreError::OverQuota)
        );
        put("tokens", "k".to_owned()).await.unwrap();
    }

    #[tokio::test]
    async fn consume_hands_the_value_to_one_caller() {
        let store = MemoryStore::default();
        create(&store, "k", "a").await.unwrap();
        assert_eq!(
            consume(&store, "ns", "k").await.unwrap().as_deref(),
            Some(&b"a"[..])
        );
        assert_eq!(consume(&store, "ns", "k").await.unwrap(), None);
    }

    /// The tombstone is what makes the race safe: a second reader that got
    /// its GET in first still loses the compare-and-set.
    #[tokio::test]
    async fn a_reader_of_a_consumed_version_loses() {
        let store = MemoryStore::default();
        let version = create(&store, "k", "a").await.unwrap();
        store
            .put(
                "ns",
                "k",
                CONSUMED.to_vec(),
                TTL,
                &BTreeMap::new(),
                PutCondition::IfVersion(version),
            )
            .await
            .unwrap();
        assert_eq!(consume(&store, "ns", "k").await.unwrap(), None);
    }
}
