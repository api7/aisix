use std::{collections::HashMap, sync::Arc};

use aisix_core::{resource::ResourceEntry, snapshot::ResourceTable, ApiKey};
use arc_swap::ArcSwapOption;

type Entry = Arc<ResourceEntry<ApiKey>>;

struct Bindings {
    generation: u64,
    // A present None marks an ambiguous binding, including disabled keys.
    // Keep ids so this cache cannot retain retired API-key objects.
    by_provider: HashMap<String, HashMap<String, Option<String>>>,
}

impl Bindings {
    fn build(table: &ResourceTable<ApiKey>) -> Self {
        let mut by_provider: HashMap<String, HashMap<String, Option<String>>> = HashMap::new();
        for entry in table
            .matching_entries(|e| e.value.jwt_provider.is_some() && e.value.jwt_subject.is_some())
        {
            let provider = entry.value.jwt_provider.as_ref().unwrap().clone();
            let subject = entry.value.jwt_subject.as_ref().unwrap().clone();
            by_provider
                .entry(provider)
                .or_default()
                .entry(subject)
                .and_modify(|binding| *binding = None)
                .or_insert(Some(entry.id.clone()));
        }
        Self {
            generation: table.generation(),
            by_provider,
        }
    }
}

#[derive(Default)]
pub(crate) struct LiveJwtBindings {
    cached: ArcSwapOption<Bindings>,
}

impl LiveJwtBindings {
    fn for_table(&self, table: &ResourceTable<ApiKey>) -> Arc<Bindings> {
        if let Some(index) = self.cached.load_full() {
            if index.generation == table.generation() {
                return index;
            }
        }
        let index = Arc::new(Bindings::build(table));
        self.cached.store(Some(Arc::clone(&index)));
        index
    }

    pub(crate) fn resolve(
        &self,
        table: &ResourceTable<ApiKey>,
        provider: &str,
        subject: &str,
    ) -> (Option<Entry>, bool) {
        let index = self.for_table(table);
        match index
            .by_provider
            .get(provider)
            .and_then(|subjects| subjects.get(subject))
        {
            None => (None, false),
            Some(None) => (None, true),
            Some(Some(id)) => (table.get_by_id(id), false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_core::AisixSnapshot;

    fn key(id: &str, provider: Option<&str>, subject: Option<&str>) -> ResourceEntry<ApiKey> {
        ResourceEntry::new(
            id,
            serde_json::from_value(serde_json::json!({
                "key_hash": id, "allowed_models": ["*"],
                "jwt_provider": provider, "jwt_subject": subject,
            }))
            .unwrap(),
            1,
        )
    }

    fn identity(answer: (Option<Entry>, bool)) -> (Option<String>, bool) {
        (answer.0.map(|entry| entry.id.clone()), answer.1)
    }

    #[test]
    fn bindings_match_scan_for_namespaces_missing_and_ambiguous_identities() {
        let table = ResourceTable::new();
        table.insert(key("one", Some("corp"), Some("subject")));
        table.insert(key("two", Some("partner"), Some("subject")));
        table.insert(key("missing-provider", None, Some("subject")));
        table.insert(key("missing-subject", Some("corp"), None));
        let cache = LiveJwtBindings::default();
        for duplicate in [false, true] {
            if duplicate {
                let mut key = key("disabled-duplicate", Some("corp"), Some("subject"));
                key.value.disabled = true;
                table.insert(key);
            }
            for (provider, subject) in [
                ("corp", "subject"),
                ("partner", "subject"),
                ("unknown", "subject"),
                ("corp", "unknown"),
            ] {
                let expected = table.find_unique_by(|e| {
                    e.value.jwt_provider.as_deref() == Some(provider)
                        && e.value.jwt_subject.as_deref() == Some(subject)
                });
                assert_eq!(
                    identity(cache.resolve(&table, provider, subject)),
                    identity(expected)
                );
            }
        }
        assert_eq!(
            identity(cache.resolve(&table, "corp", "subject")),
            (None, true)
        );
    }

    #[test]
    fn bindings_follow_lifecycle_edits_deletion_rebuilds_and_held_snapshots() {
        let original = ResourceTable::new();
        original.insert(key("key", Some("corp"), Some("subject")));
        let cache = Arc::new(LiveJwtBindings::default());
        let before = cache.resolve(&original, "corp", "subject").0.unwrap();
        let next = original.clone();
        let mut changed = key("key", Some("corp"), Some("subject"));
        changed.value.disabled = true;
        changed.value.expires_at = Some("2000-01-01T00:00:00Z".parse().unwrap());
        next.insert(changed);
        let after = cache.resolve(&next, "corp", "subject").0.unwrap();
        assert!(after.value.disabled && after.value.expires_at.is_some());
        assert!(!before.value.disabled && before.value.expires_at.is_none());
        assert!(Arc::ptr_eq(&after, &next.get_by_id("key").unwrap()));
        std::thread::scope(|scope| {
            for table in [&original, &next] {
                let cache = &cache;
                scope.spawn(move || {
                    for _ in 0..100 {
                        let got = cache.resolve(table, "corp", "subject").0.unwrap();
                        assert!(Arc::ptr_eq(&got, &table.get_by_id("key").unwrap()));
                    }
                });
            }
        });
        next.remove("key");
        assert_eq!(
            identity(cache.resolve(&next, "corp", "subject")),
            (None, false)
        );
        next.insert(key("key", Some("partner"), Some("renamed")));
        assert_eq!(
            identity(cache.resolve(&next, "corp", "subject")),
            (None, false)
        );
        assert_eq!(
            identity(cache.resolve(&next, "partner", "renamed")),
            (Some("key".into()), false)
        );
        let rebuilt = ResourceTable::new();
        rebuilt.insert(key("replacement", Some("corp"), Some("subject")));
        assert_eq!(
            identity(cache.resolve(&rebuilt, "corp", "subject")),
            (Some("replacement".into()), false)
        );
        assert_eq!(
            identity(cache.resolve(&original, "corp", "subject")),
            (Some("key".into()), false)
        );
    }

    #[test]
    fn index_reuses_unchanged_table_and_stores_only_configured_bindings() {
        let snapshot = AisixSnapshot::new();
        for i in 0..10_000 {
            snapshot
                .apikeys
                .insert(key(&format!("ordinary-{i}"), None, None));
        }
        snapshot
            .apikeys
            .insert(key("bound", Some("corp"), Some("subject")));
        let cache = LiveJwtBindings::default();
        let index = cache.for_table(&snapshot.apikeys);
        let bound = snapshot.apikeys.get_by_id("bound").unwrap();
        assert_eq!(Arc::strong_count(&bound), 2);
        assert_eq!(index.by_provider.len(), 1);
        assert_eq!(index.by_provider["corp"].len(), 1);
        let copy = snapshot.clone();
        copy.models.insert(ResourceEntry::new(
            "model",
            serde_json::from_value(serde_json::json!({
                "display_name":"model", "provider":"openai", "model_name":"model"
            }))
            .unwrap(),
            1,
        ));
        for i in 0..1000 {
            assert_eq!(
                identity(cache.resolve(&copy.apikeys, "corp", &format!("unknown-{i}"))),
                (None, false)
            );
        }
        assert!(Arc::ptr_eq(&index, &cache.for_table(&copy.apikeys)));
        assert_eq!(index.by_provider["corp"].len(), 1);
    }
}
