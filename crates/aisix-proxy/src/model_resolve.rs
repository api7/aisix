//! Wildcard-aware model resolution (`provider/*` aliases).
//!
//! Client-facing handlers resolve `req.model` through [`resolve_model`] instead
//! of calling `snapshot.models.get_by_name` directly, so a request for
//! `openai/gpt-4o` can be served by an operator-defined `openai/*` wildcard
//! Model. Exact names always win; among wildcards the most specific (longest
//! literal) pattern wins.
//!
//! A wildcard match returns a synthetic clone of the Model with its
//! `model_name` resolved to the concrete upstream id (the captured segment
//! substituted into the template's `*`, or the template kept as-is when it has
//! no `*`). Everything else — id, provider, provider_key_id, rate_limit, cost,
//! guardrail scope — is inherited from the wildcard Model, so downstream
//! dispatch/telemetry treat it like a normal Model and attribution stays on the
//! wildcard Model's id. The client still sees the name it requested, because
//! every handler echoes `req.model` rather than the resolved `display_name`.

use std::sync::Arc;

use aisix_core::wildcard::wildcard_capture;
use aisix_core::{AisixSnapshot, Model, ResourceEntry};

/// Resolve `requested` against the snapshot's Model table, honoring `provider/*`
/// wildcard Models when no exact match exists. Returns `None` if nothing matches.
pub(crate) fn resolve_model(
    snapshot: &AisixSnapshot,
    requested: &str,
) -> Option<Arc<ResourceEntry<Model>>> {
    if let Some(exact) = snapshot.models.get_by_name(requested) {
        return Some(exact);
    }
    // Wildcard fallback: the most specific direct Model whose `*`-glob display
    // name matches the request. Only runs when the exact lookup missed.
    let mut best: Option<(usize, Arc<ResourceEntry<Model>>, String)> = None;
    for entry in snapshot.models.entries() {
        let model = &entry.value;
        if !model.display_name.contains('*') {
            continue;
        }
        // Only direct Models can serve a wildcard alias — routers / ensembles /
        // semantic routers have no upstream `model_name` to dispatch.
        if model.is_routing() || model.is_ensemble() || model.is_semantic() {
            continue;
        }
        let Some(capture) = wildcard_capture(&model.display_name, requested) else {
            continue;
        };
        // Specificity = literal (non-`*`) length; longest wins, first on a tie.
        let specificity = model.display_name.len() - 1;
        if best.as_ref().is_none_or(|(s, _, _)| specificity > *s) {
            let upstream = resolve_upstream_model_name(model, &capture);
            best = Some((specificity, entry.clone(), upstream));
        }
    }
    let (_, entry, upstream) = best?;
    let mut model = entry.value.clone();
    model.model_name = Some(upstream);
    Some(Arc::new(ResourceEntry::new(
        entry.id.clone(),
        model,
        entry.revision,
    )))
}

/// Concrete upstream model id for a wildcard match: substitute the captured
/// segment into the `model_name` template's `*`, keep the template as-is when it
/// has no `*` (a fixed upstream for every match), or send the captured segment
/// verbatim when there is no template.
fn resolve_upstream_model_name(model: &Model, capture: &str) -> String {
    match model.model_name.as_deref() {
        Some(t) if t.contains('*') => t.replacen('*', capture, 1),
        Some(t) => t.to_string(),
        None => capture.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_core::snapshot::ResourceTable;

    fn direct_model(display_name: &str, model_name: Option<&str>) -> Model {
        serde_json::from_value(serde_json::json!({
            "display_name": display_name,
            "provider": "openai",
            "model_name": model_name,
            "provider_key_id": "pk-1",
        }))
        .unwrap()
    }

    fn snapshot_with(models: Vec<(&str, Model)>) -> AisixSnapshot {
        let table = ResourceTable::default();
        for (id, model) in models {
            table.insert(ResourceEntry::new(id, model, 1));
        }
        AisixSnapshot {
            models: table,
            ..Default::default()
        }
    }

    #[test]
    fn exact_match_wins_over_wildcard() {
        let snap = snapshot_with(vec![
            ("m-star", direct_model("openai/*", Some("*"))),
            (
                "m-exact",
                direct_model("openai/gpt-4o", Some("gpt-4o-2024")),
            ),
        ]);
        let resolved = resolve_model(&snap, "openai/gpt-4o").unwrap();
        assert_eq!(resolved.id, "m-exact");
        assert_eq!(resolved.value.model_name.as_deref(), Some("gpt-4o-2024"));
    }

    #[test]
    fn wildcard_substitutes_capture_into_template() {
        let snap = snapshot_with(vec![("m-star", direct_model("openai/*", Some("*")))]);
        let resolved = resolve_model(&snap, "openai/gpt-4o").unwrap();
        // Attribution stays on the wildcard Model; upstream id is the capture.
        assert_eq!(resolved.id, "m-star");
        assert_eq!(resolved.value.model_name.as_deref(), Some("gpt-4o"));
    }

    #[test]
    fn wildcard_with_fixed_template_pins_all_matches() {
        let snap = snapshot_with(vec![("m-star", direct_model("gpt-*", Some("gpt-4o")))]);
        let resolved = resolve_model(&snap, "gpt-anything").unwrap();
        assert_eq!(resolved.value.model_name.as_deref(), Some("gpt-4o"));
    }

    #[test]
    fn most_specific_wildcard_wins() {
        let snap = snapshot_with(vec![
            ("m-broad", direct_model("openai/*", Some("*"))),
            ("m-narrow", direct_model("openai/gpt-*", Some("*"))),
        ]);
        let resolved = resolve_model(&snap, "openai/gpt-4o").unwrap();
        assert_eq!(resolved.id, "m-narrow");
        // `openai/gpt-*` captures only `4o`.
        assert_eq!(resolved.value.model_name.as_deref(), Some("4o"));
    }

    #[test]
    fn no_match_returns_none() {
        let snap = snapshot_with(vec![("m-star", direct_model("openai/*", Some("*")))]);
        assert!(resolve_model(&snap, "anthropic/claude").is_none());
    }
}
