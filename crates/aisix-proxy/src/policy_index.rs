use std::{collections::HashMap, sync::Arc};

use aisix_core::{
    models::{
        ConditionInput, ConditionLogic, ConditionNode, ConditionOperator, ConditionValue,
        PolicyDimension, PolicyScope, RateLimitPolicy,
    },
    resource::ResourceEntry,
    snapshot::ResourceTable,
};
use arc_swap::ArcSwapOption;

type Entry = Arc<ResourceEntry<RateLimitPolicy>>;

pub(crate) struct PolicyIndex {
    generation: u64,
    entries: Vec<Entry>,
    by_value: HashMap<PolicyDimension, HashMap<String, Vec<usize>>>,
    fallback: Vec<usize>,
}

impl PolicyIndex {
    fn build(table: &ResourceTable<RateLimitPolicy>) -> Self {
        let mut index = Self {
            generation: table.generation(),
            entries: table.entries(),
            by_value: HashMap::new(),
            fallback: Vec::new(),
        };
        for (position, entry) in index.entries.iter().enumerate() {
            if let Some((dimension, values)) = selector(&entry.value) {
                let by_value = index.by_value.entry(dimension).or_default();
                for value in values {
                    by_value.entry(value).or_default().push(position);
                }
            } else {
                index.fallback.push(position);
            }
        }
        index
    }

    /// A superset of matching policies. Full conditions, schedules and the
    /// request/target reservation phase still decide whether each one applies.
    /// Model selectors include both dispatched target and caller-addressed parent.
    pub(crate) fn candidates(&self, input: &ConditionInput<'_>) -> impl Iterator<Item = &Entry> {
        let mut positions = self.fallback.clone();
        for (dimension, by_value) in &self.by_value {
            let parent = match dimension {
                PolicyDimension::Model => input.routing_parent_model,
                PolicyDimension::ModelName => input.routing_parent_model_name,
                _ => None,
            };
            for value in [input.get(*dimension), parent].into_iter().flatten() {
                if let Some(matches) = by_value.get(value) {
                    positions.extend_from_slice(matches);
                }
            }
        }
        // Preserve the snapshot's policy order and reserve a row only once
        // when both model identities (or repeated set values) select it.
        positions.sort_unstable();
        positions.dedup();
        positions
            .into_iter()
            .map(|position| &self.entries[position])
    }
}

fn selector(policy: &RateLimitPolicy) -> Option<(PolicyDimension, Vec<String>)> {
    if let Some(conditions) = &policy.conditions {
        required_match(conditions)
    } else {
        let dimension = match policy.scope? {
            PolicyScope::ApiKey => PolicyDimension::ApiKey,
            PolicyScope::Model => PolicyDimension::Model,
            PolicyScope::Team | PolicyScope::TeamMember => PolicyDimension::Team,
            PolicyScope::Member => PolicyDimension::Member,
        };
        Some((dimension, vec![policy.scope_ref.clone()?]))
    }
}

fn required_match(nodes: &[ConditionNode]) -> Option<(PolicyDimension, Vec<String>)> {
    nodes.iter().find_map(|node| match node {
        ConditionNode::Leaf(leaf) if !leaf.negate => match (&leaf.operator, &leaf.value) {
            (ConditionOperator::Eq, ConditionValue::One(value)) => {
                Some((leaf.dimension, vec![value.clone()]))
            }
            (ConditionOperator::In, ConditionValue::Many(values)) => {
                Some((leaf.dimension, values.clone()))
            }
            _ => None,
        },
        ConditionNode::Group(group) if group.logic == ConditionLogic::And && !group.negate => {
            required_match(&group.children)
        }
        // OR, negation and non-equality predicates cannot supply a necessary
        // value match. Retain those policies on the ordinary evaluation path.
        _ => None,
    })
}

#[derive(Default)]
pub(crate) struct LivePolicyIndex {
    cached: ArcSwapOption<PolicyIndex>,
}

impl LivePolicyIndex {
    pub(crate) fn for_table(&self, table: &ResourceTable<RateLimitPolicy>) -> Arc<PolicyIndex> {
        if let Some(index) = self.cached.load_full() {
            if index.generation == table.generation() {
                return index;
            }
        }
        let index = Arc::new(PolicyIndex::build(table));
        self.cached.store(Some(Arc::clone(&index)));
        index
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn policy(member: &str) -> RateLimitPolicy {
        serde_json::from_value(json!({
            "name": member,
            "conditions": [{"dimension":"member","operator":"==","value":member}],
            "limits": {"rpm":100}
        }))
        .unwrap()
    }

    #[test]
    fn unrelated_policies_do_not_enter_the_request_candidate_set() {
        let table = ResourceTable::new();
        for i in 0..5_000 {
            table.insert(ResourceEntry::new(
                format!("p{i}"),
                policy(&format!("member{i}")),
                1,
            ));
        }
        let index = PolicyIndex::build(&table);
        let input = ConditionInput {
            member: Some("member123"),
            ..Default::default()
        };
        let selected: Vec<_> = index.candidates(&input).map(|e| e.id.as_str()).collect();
        assert_eq!(selected, ["p123"]);
        assert_eq!(index.candidates(&ConditionInput::default()).count(), 0);
    }

    #[test]
    fn policy_generation_invalidates_create_update_delete_and_full_replacement() {
        let cache = LivePolicyIndex::default();
        let original = ResourceTable::new();
        let empty = cache.for_table(&original);
        let next = original.clone();
        next.insert(ResourceEntry::new("p", policy("before"), 1));
        let before = cache.for_table(&next);
        assert!(!Arc::ptr_eq(&empty, &before));
        assert!(Arc::ptr_eq(&before, &cache.for_table(&next.clone())));
        let changed = next.clone();
        changed.insert(ResourceEntry::new("p", policy("after"), 2));
        let after = cache.for_table(&changed);
        let input = ConditionInput {
            member: Some("before"),
            ..Default::default()
        };
        assert_eq!(before.candidates(&input).count(), 1);
        assert_eq!(after.candidates(&input).count(), 0);
        assert_eq!(
            after
                .candidates(&ConditionInput {
                    member: Some("after"),
                    ..input
                })
                .count(),
            1
        );
        let removed = changed.clone();
        removed.remove("p");
        assert_eq!(cache.for_table(&removed).entries.len(), 0);
        let replacement = ResourceTable::new();
        replacement.insert(ResourceEntry::new("p", policy("replacement"), 3));
        assert_eq!(
            cache
                .for_table(&replacement)
                .candidates(&ConditionInput {
                    member: Some("replacement"),
                    ..input
                })
                .count(),
            1
        );
        // An in-flight request can still use its older immutable snapshot.
        assert_eq!(cache.for_table(&next).candidates(&input).count(), 1);
    }
}
