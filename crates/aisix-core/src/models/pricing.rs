//! `Pricing` entity — a per-1,000-token price a model refers to by
//! `pricing_key` instead of carrying inline.
//!
//! Pricing documents have their own lifecycle: a price changes far more
//! often than the models priced by it, and one price is usually shared by
//! many models. Two prefixes carry them. The environment prefix
//! (`<prefix>/<env>/pricing/<uuid>`) holds an organization's own
//! overrides; the global prefix (`<prefix>/global/pricing/<uuid>`) holds
//! the catalog every environment reads. An environment document wins over
//! a global one with the same `key`.
//!
//! The gateway resolves the reference on the read path, so a price edit
//! takes effect on the next request without any model document being
//! rewritten.

use arc_swap::ArcSwapOption;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::models::model::{Model, ModelCost};
use crate::models::snapshot::AisixSnapshot;
use crate::resource::Resource;
use crate::snapshot::ResourceTable;

/// A per-1,000-token price shared by every model that names it.
///
/// A model refers to one of these with `pricing_key`. The gateway looks
/// the price up in the environment's own pricing documents first and in
/// the global catalog second; a model with no `pricing_key`, or one whose
/// key matches no document, falls back to its inline `cost`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Pricing {
    /// Value a model's `pricing_key` is matched against, compared as an
    /// exact string. Conventionally `<provider>/<model_name>`, but the
    /// gateway attaches no meaning to its parts.
    #[schemars(length(min = 1, max = 255))]
    pub key: String,

    /// Prompt token price in USD per 1,000 tokens.
    #[schemars(range(min = 0.0))]
    pub input_per_1k: f64,

    /// Completion token price in USD per 1,000 tokens.
    #[schemars(range(min = 0.0))]
    pub output_per_1k: f64,

    /// Set by the loader from the kine path's UUID segment. Not part of
    /// the wire shape.
    #[serde(skip)]
    pub(crate) runtime_id: String,
}

impl Resource for Pricing {
    fn id(&self) -> &str {
        &self.runtime_id
    }

    /// The lookup `key`, not a display name: the name index is what
    /// `pricing_key` resolution reads.
    fn name(&self) -> &str {
        &self.key
    }

    fn kind() -> &'static str {
        "pricing"
    }
}

/// Prices by lookup key, flattened from the two pricing tables.
///
/// Derived state, so it is rebuilt only when the rows behind it change:
/// [`LivePricingIndex`] keys the cached copy on the two tables'
/// generations rather than on the snapshot version, which moves on every
/// published write of any kind (AISIX-Cloud#1542).
#[derive(Debug, Default)]
pub struct PricingIndex {
    by_key: HashMap<String, ModelCost>,
}

impl PricingIndex {
    /// Flatten both tables, environment documents last so one of them
    /// replaces the global document carrying the same `key`.
    pub fn build(snap: &AisixSnapshot) -> Self {
        let global = flatten(&snap.global_pricing);
        let mut by_key: HashMap<String, ModelCost> =
            global.into_iter().map(|(k, v)| (k, v.2)).collect();
        by_key.extend(flatten(&snap.pricing).into_iter().map(|(k, v)| (k, v.2)));
        Self { by_key }
    }

    /// The price a `pricing_key` names, environment documents winning
    /// over global ones. `None` when no document carries the key.
    pub fn get(&self, key: &str) -> Option<&ModelCost> {
        self.by_key.get(key)
    }

    /// The price to charge `model` by: the document its `pricing_key`
    /// names, then the model's own inline `cost`, then nothing.
    ///
    /// Every reader of a model's price goes through here, so ranking and
    /// the usage events cannot disagree about what a model costs.
    pub fn resolve<'a>(&'a self, model: &'a Model) -> Option<&'a ModelCost> {
        model
            .pricing_key
            .as_deref()
            .and_then(|key| self.get(key))
            .or(model.cost.as_ref())
    }

    pub fn len(&self) -> usize {
        self.by_key.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_key.is_empty()
    }
}

/// One table's prices by `key`, carrying the revision and id that won.
///
/// Nothing constrains `key` to be unique within a table, and the control
/// plane writes a replacement before deleting the row it replaces — so
/// two documents can carry one key for a window. Picking by highest
/// revision, then by id, makes the effective price the later write
/// rather than whichever row the table happened to iterate first: an
/// order that is not stable across rebuilds would otherwise let the
/// price flip back and forth while the window is open, changing both
/// `least_cost` ordering and the cost on emitted usage events.
fn flatten(table: &ResourceTable<Pricing>) -> HashMap<String, (i64, String, ModelCost)> {
    let mut out: HashMap<String, (i64, String, ModelCost)> = HashMap::new();
    for entry in table.entries() {
        let candidate = (entry.revision, entry.id.clone(), cost_of(&entry.value));
        match out.get(&entry.value.key) {
            Some(current) if (current.0, &current.1) >= (candidate.0, &candidate.1) => {}
            _ => {
                out.insert(entry.value.key.clone(), candidate);
            }
        }
    }
    out
}

fn cost_of(p: &Pricing) -> ModelCost {
    ModelCost {
        input_per_1k: p.input_per_1k,
        output_per_1k: p.output_per_1k,
    }
}

/// The generations of exactly the tables [`PricingIndex::build`] reads.
fn index_key(snap: &AisixSnapshot) -> (u64, u64) {
    (snap.pricing.generation(), snap.global_pricing.generation())
}

#[derive(Debug)]
struct Cached {
    key: (u64, u64),
    index: Arc<PricingIndex>,
}

/// Lazily-rebuilt [`PricingIndex`] shared by every reader of a model's
/// price. Cheap on the request path: a hit is one atomic load.
///
/// A racing rebuild produces two equal indexes and one of them is
/// discarded — the same benign duplication the guardrail index accepts,
/// and cheaper than holding a lock across the build.
#[derive(Debug, Default)]
pub struct LivePricingIndex {
    cached: ArcSwapOption<Cached>,
}

impl LivePricingIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// The index for `snap`, rebuilt only if a pricing table changed
    /// since the cached copy was built. Takes the caller's snapshot
    /// rather than loading its own, so a request ranks and bills against
    /// the same published configuration it resolved its models from.
    pub fn for_snapshot(&self, snap: &AisixSnapshot) -> Arc<PricingIndex> {
        let key = index_key(snap);
        if let Some(cached) = self.cached.load_full() {
            if cached.key == key {
                return Arc::clone(&cached.index);
            }
        }
        let index = Arc::new(PricingIndex::build(snap));
        self.cached.store(Some(Arc::new(Cached {
            key,
            index: Arc::clone(&index),
        })));
        index
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_name_is_the_lookup_key() {
        let p: Pricing = serde_json::from_str(
            r#"{"key":"openai/gpt-4o","input_per_1k":0.005,"output_per_1k":0.015}"#,
        )
        .unwrap();
        assert_eq!(p.name(), "openai/gpt-4o");
        assert_eq!(p.input_per_1k, 0.005);
        assert_eq!(p.output_per_1k, 0.015);
    }

    #[test]
    fn resource_kind_matches_kine_path_segment() {
        assert_eq!(<Pricing as Resource>::kind(), "pricing");
    }

    use crate::resource::ResourceEntry;

    fn price(key: &str, input: f64, output: f64) -> Pricing {
        Pricing {
            key: key.into(),
            input_per_1k: input,
            output_per_1k: output,
            runtime_id: String::new(),
        }
    }

    fn model(json: &str) -> Model {
        serde_json::from_str(json).unwrap()
    }

    fn direct(extra: &str) -> Model {
        model(&format!(
            r#"{{"display_name":"m","provider":"openai","model_name":"gpt-4o",
                "provider_key_id":"11111111-1111-1111-1111-111111111111"{extra}}}"#
        ))
    }

    #[test]
    fn an_environment_document_wins_over_the_global_one() {
        let snap = AisixSnapshot::new();
        snap.global_pricing
            .insert(ResourceEntry::new("g", price("k", 1.0, 1.0), 1));
        snap.pricing
            .insert(ResourceEntry::new("e", price("k", 9.0, 9.0), 2));

        let index = PricingIndex::build(&snap);
        assert_eq!(index.len(), 1);
        assert_eq!(index.get("k").unwrap().input_per_1k, 9.0);
    }

    #[test]
    fn two_documents_with_one_key_resolve_to_the_later_write() {
        // The control plane writes a replacement before deleting the row
        // it replaces, so one key can name two documents for a window.
        // Whichever the table iterates first is not stable across
        // rebuilds; the price must not flip while the window is open.
        let snap = AisixSnapshot::new();
        snap.global_pricing
            .insert(ResourceEntry::new("old", price("k", 1.0, 1.0), 4));
        snap.global_pricing
            .insert(ResourceEntry::new("new", price("k", 9.0, 9.0), 7));

        for _ in 0..20 {
            assert_eq!(
                PricingIndex::build(&snap).get("k").unwrap().input_per_1k,
                9.0
            );
        }
    }

    #[test]
    fn a_global_document_applies_where_the_environment_has_none() {
        let snap = AisixSnapshot::new();
        snap.global_pricing
            .insert(ResourceEntry::new("g", price("k", 1.0, 2.0), 1));
        snap.pricing
            .insert(ResourceEntry::new("e", price("other", 9.0, 9.0), 2));

        let index = PricingIndex::build(&snap);
        assert_eq!(index.get("k").unwrap().output_per_1k, 2.0);
    }

    #[test]
    fn resolution_falls_through_pricing_key_then_inline_cost() {
        let snap = AisixSnapshot::new();
        snap.global_pricing
            .insert(ResourceEntry::new("g", price("k", 1.0, 1.0), 1));
        let index = PricingIndex::build(&snap);

        // A resolving key wins over the inline cost sitting next to it,
        // which is what makes the reference authoritative rather than
        // advisory.
        let keyed =
            direct(r#","pricing_key":"k","cost":{"input_per_1k":50.0,"output_per_1k":50.0}"#);
        assert_eq!(index.resolve(&keyed).unwrap().input_per_1k, 1.0);

        // A key naming no document falls through to the inline cost.
        let missing =
            direct(r#","pricing_key":"absent","cost":{"input_per_1k":7.0,"output_per_1k":7.0}"#);
        assert_eq!(index.resolve(&missing).unwrap().input_per_1k, 7.0);

        // No key at all: unchanged behaviour for every model written
        // before this feature existed.
        let inline = direct(r#","cost":{"input_per_1k":3.0,"output_per_1k":3.0}"#);
        assert_eq!(index.resolve(&inline).unwrap().input_per_1k, 3.0);

        // Neither: no price, which ranks last rather than free.
        let none = direct(r#","pricing_key":"absent""#);
        assert!(index.resolve(&none).is_none());
        assert!(index.resolve(&direct("")).is_none());
    }

    #[test]
    fn the_live_index_rebuilds_when_a_pricing_table_changes() {
        let snap = AisixSnapshot::new();
        snap.global_pricing
            .insert(ResourceEntry::new("g", price("k", 1.0, 1.0), 1));
        let live = LivePricingIndex::new();

        let first = live.for_snapshot(&snap);
        assert!(Arc::ptr_eq(&first, &live.for_snapshot(&snap)));

        // A write to an unrelated table must NOT invalidate: that is the
        // difference between keying on the table generation and keying on
        // the snapshot version (AISIX-Cloud#1542).
        snap.apikeys.insert(ResourceEntry::new(
            "k-1",
            serde_json::from_str::<crate::models::ApiKey>(
                r#"{"key_hash":"91ed2dbc407561556f3e7be98ba0bd2a57986d6a868c482d867d19c6d40d201c"}"#,
            )
            .unwrap(),
            1,
        ));
        assert!(Arc::ptr_eq(&first, &live.for_snapshot(&snap)));

        // A price edit does invalidate, and the new price is what the
        // next reader sees — with no model document involved.
        snap.global_pricing
            .insert(ResourceEntry::new("g", price("k", 5.0, 5.0), 2));
        let second = live.for_snapshot(&snap);
        assert!(!Arc::ptr_eq(&first, &second));
        assert_eq!(second.get("k").unwrap().input_per_1k, 5.0);

        // The environment table is the other half of the invalidation
        // key; a write there must invalidate too.
        snap.pricing
            .insert(ResourceEntry::new("e", price("k", 8.0, 8.0), 3));
        let third = live.for_snapshot(&snap);
        assert!(!Arc::ptr_eq(&second, &third));
        assert_eq!(third.get("k").unwrap().input_per_1k, 8.0);
    }

    #[test]
    fn pricing_key_is_direct_only_like_cost() {
        // `pricing_key` and `cost` are two spellings of the same knob, so
        // a kind that strips one must strip the other — otherwise a
        // routing group could carry a price the runtime never reads.
        let mut group: Model = serde_json::from_str(
            r#"{"display_name":"g","routing":{"targets":[{"model":"a"}]},
                "cost":{"input_per_1k":1.0,"output_per_1k":1.0},"pricing_key":"k"}"#,
        )
        .unwrap();
        let stripped = group.strip_kind_inapplicable();
        assert!(stripped.contains(&"pricing_key"), "{stripped:?}");
        assert!(stripped.contains(&"cost"), "{stripped:?}");
        assert!(group.pricing_key.is_none() && group.cost.is_none());
    }
}
