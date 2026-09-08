//! Resolving a document's reference to another Model.
//!
//! Several projected documents point at a Model: a routing target, an
//! ensemble panel member and its judge, a semantic router's embedding
//! model / default / route targets, a cache policy's scope and its
//! semantic embedder, a `kind: semantic` guardrail's embedder. Each was
//! written as a display NAME, which means renaming the referenced model
//! silently breaks every document that named it — the reference resolves
//! to nothing and the site degrades.
//!
//! Every one of those sites now also accepts the referenced model's
//! resource id, resolved through [`resolve_model_ref`]. The id is stable
//! across renames, so the referencing document never has to be rewritten.

use super::AisixSnapshot;
use std::borrow::Cow;

/// Resolve a model reference written as a display name plus an optional
/// resource id, yielding the name the models table is keyed by.
///
/// The id decides whenever it is present: it is looked up in the models
/// table and the entry's CURRENT display name is returned, so renaming the
/// referenced model takes effect on the next request with no change to the
/// referencing document. `name` is not consulted at all in that case.
/// Resolution happens per request against the live table rather than being
/// cached, and the table is the only input, so nothing derived from an
/// unrelated resource can hold a stale answer.
///
/// An id matching no model yields the id itself. Every caller goes on to
/// look the result up by name and finds nothing — which is exactly what a
/// DANGLING NAME does at the same site. So an unresolvable id degrades the
/// way the site already degrades (a routing target that does not exist, an
/// embedder that cannot be resolved, a cache policy that matches nothing)
/// rather than introducing a failure mode of its own, and in particular
/// never fails the row.
pub fn resolve_model_ref<'a>(
    snapshot: &AisixSnapshot,
    name: &'a str,
    id: Option<&'a str>,
) -> Cow<'a, str> {
    match id {
        Some(id) => match snapshot.models.get_by_id(id) {
            Some(entry) => Cow::Owned(entry.value.display_name.clone()),
            None => Cow::Borrowed(id),
        },
        None => Cow::Borrowed(name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Model;
    use crate::resource::ResourceEntry;

    fn snapshot_with(models: &[(&str, &str)]) -> AisixSnapshot {
        let snap = AisixSnapshot::default();
        for (id, display_name) in models {
            let model: Model = serde_json::from_str(&format!(
                r#"{{
                  "display_name": "{display_name}",
                  "provider": "openai",
                  "model_name": "gpt-4o",
                  "provider_key_id": "11111111-1111-1111-1111-111111111111"
                }}"#
            ))
            .unwrap();
            snap.models.insert(ResourceEntry::new(*id, model, 1));
        }
        snap
    }

    #[test]
    fn absent_id_keeps_the_name() {
        let snap = snapshot_with(&[("m-1", "gpt")]);
        assert_eq!(resolve_model_ref(&snap, "whatever", None), "whatever");
    }

    #[test]
    fn present_id_wins_over_the_name() {
        let snap = snapshot_with(&[("m-1", "gpt"), ("m-2", "claude")]);
        assert_eq!(resolve_model_ref(&snap, "claude", Some("m-1")), "gpt");
    }

    #[test]
    fn present_id_follows_a_rename() {
        let before = snapshot_with(&[("m-1", "gpt")]);
        assert_eq!(resolve_model_ref(&before, "", Some("m-1")), "gpt");
        let after = snapshot_with(&[("m-1", "gpt-v2")]);
        assert_eq!(resolve_model_ref(&after, "", Some("m-1")), "gpt-v2");
    }

    #[test]
    fn unresolvable_id_yields_a_name_that_resolves_to_nothing() {
        let snap = snapshot_with(&[("m-1", "gpt")]);
        let resolved = resolve_model_ref(&snap, "gpt", Some("m-gone"));
        assert_eq!(resolved, "m-gone");
        assert!(snap.models.get_by_name(&resolved).is_none());
    }
}
