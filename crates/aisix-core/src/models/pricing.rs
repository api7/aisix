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
        let mut by_key = HashMap::new();
        for entry in snap.global_pricing.entries() {
            by_key.insert(entry.value.key.clone(), cost_of(&entry.value));
        }
        for entry in snap.pricing.entries() {
            by_key.insert(entry.value.key.clone(), cost_of(&entry.value));
        }
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
}
