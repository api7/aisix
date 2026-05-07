//! [`Resource`] trait and [`ResourceEntry`] wrapper.
//!
//! Every entity stored in etcd (Model, ApiKey, ProviderKey, …) is wrapped in a
//! [`ResourceEntry<T>`] carrying its UUID and the etcd revision it came from.
//!
//! Downstream code (proxy handlers, admin handlers, routing) holds
//! `Arc<ResourceEntry<T>>` and usually wants to reach the `T` directly —
//! hence the `Deref` impl.
//!
//! Spec references: §3 (ResourceEntry<T> Deref to T), §2 (secondary indices
//! keyed by name / api-key value).

use serde::{Deserialize, Serialize};
use std::ops::Deref;

/// Trait every gateway entity implements so the [`crate::snapshot::Snapshot`]
/// can build a secondary name-index without knowing the concrete type.
pub trait Resource: Send + Sync + 'static {
    /// Stable UUID v4 identifying this resource (etcd key suffix).
    fn id(&self) -> &str;

    /// Human-readable unique name within the resource kind. Used for
    /// `name → id` lookups and for duplicate-detection on create/update.
    fn name(&self) -> &str;

    /// Prefix segment used for the etcd key (e.g. `"models"`, `"apikeys"`).
    /// Constant per type.
    fn kind() -> &'static str
    where
        Self: Sized;
}

/// Generic wrapper over a typed resource with its etcd coordinates.
///
/// Cheap to clone (fields are small / already Arc'd for nested payloads).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceEntry<T> {
    pub id: String,
    pub value: T,
    pub revision: i64,
}

impl<T> ResourceEntry<T> {
    pub fn new(id: impl Into<String>, value: T, revision: i64) -> Self {
        Self {
            id: id.into(),
            value,
            revision,
        }
    }
}

/// Deref-through so callers can write `entry.name()` instead of `entry.value.name()`.
///
/// Example:
/// ```ignore
/// let entry: ResourceEntry<Model> = …;
/// let name: &str = entry.name();   // routes through Deref → Model::name()
/// ```
impl<T> Deref for ResourceEntry<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct Widget {
        id: String,
        name: String,
    }

    impl Resource for Widget {
        fn id(&self) -> &str {
            &self.id
        }

        fn name(&self) -> &str {
            &self.name
        }

        fn kind() -> &'static str {
            "widgets"
        }
    }

    #[test]
    fn deref_forwards_to_inner_resource_methods() {
        let w = Widget {
            id: "w-1".into(),
            name: "alpha".into(),
        };
        let entry = ResourceEntry::new("w-1", w, 42);
        // The whole point: .name() resolves through Deref without `entry.value.`.
        assert_eq!(entry.name(), "alpha");
        assert_eq!(entry.id(), "w-1");
        assert_eq!(entry.revision, 42);
    }

    #[test]
    fn serialises_as_flat_id_value_revision() {
        #[derive(Serialize, Deserialize, Debug, PartialEq)]
        struct Tiny {
            x: u32,
        }

        let e = ResourceEntry::new("t-1", Tiny { x: 7 }, 3);
        let json = serde_json::to_value(&e).unwrap();
        assert_eq!(json["id"], "t-1");
        assert_eq!(json["revision"], 3);
        assert_eq!(json["value"]["x"], 7);
    }
}
