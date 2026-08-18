//! Conditional-form vocabulary for [`RateLimitPolicy`]: the condition
//! node tree a policy matches requests with, and the dimensions its
//! counters bucket on (AISIX-Cloud#892).
//!
//! The tree is the Rust equivalent of
//! [lua-resty-expr](https://github.com/api7/lua-resty-expr): a node is
//! either a leaf `{dimension, operator, negate?, value}` (`negate` = the
//! `!` operator prefix) or a group `{logic: and|or, negate?, children}`
//! (`negate` = `!AND`/`!OR`). A node list combines as an implicit AND —
//! the same convention as an APISIX route `vars` array.
//!
//! Operator tokens mirror lua-resty-expr verbatim. Which operators a
//! dimension admits is a **validation** concern
//! ([`validate_condition_nodes`]): identity dimensions carry opaque
//! UUIDs, so only equality/set operators make sense; string dimensions
//! additionally admit the regex operators. `has`, the numeric
//! comparisons and `ipmatch` are part of the wire vocabulary so future
//! array/numeric/IP dimensions need no protocol change, but no v1
//! dimension admits them yet.
//!
//! Evaluation semantics ([`eval_condition_nodes`]):
//! - a leaf whose dimension the request does not carry is `false`, even
//!   under `negate` — a request missing the dimension belongs to
//!   neither the set nor its complement; OR siblings can still match;
//!   note this guarantee is leaf-level only: a **negated group** over
//!   model-property leaves evaluates `true` on model-less requests
//!   (children all false → `!OR`/`!AND` flips it), exactly like
//!   lua-resty-expr — "everything except gpt-4" written as `!(...)`
//!   deliberately includes MCP/A2A traffic;
//! - groups short-circuit (AND on the first false child, OR on the
//!   first true child);
//! - the model dimensions (`model` / `model_name`) evaluate against a
//!   PAIR on a routing/ensemble/semantic dispatch: the dispatched
//!   target and the caller-addressed parent entry (AISIX-Cloud#1267).
//!   Positive operators are raw-true when EITHER identity satisfies
//!   them (∃); `~=` is raw-true only when BOTH differ (∀ — the two
//!   identities are distinct strings, so an ∃ reading would be
//!   vacuously true on every routed request), preserving `a ~= b` ≡
//!   `!(a == b)`. `negate` flips the combined result, so `!(model in
//!   [group])` excludes every request addressed to the group instead
//!   of matching all of them;
//! - regexes are compiled once per distinct pattern into a process-wide
//!   cache. Load-time validation guarantees compilability, so a cache
//!   miss at evaluation time never fails in practice; a pattern that
//!   somehow does not compile evaluates to `false`.
//!
//! [`RateLimitPolicy`]: super::rate_limit_policy::RateLimitPolicy

use std::sync::Arc;

use dashmap::DashMap;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};

/// Maximum group-nesting depth of a condition tree (top-level nodes are
/// depth 1). Mirrored by cp-api validation and the dashboard builder.
pub const MAX_CONDITION_DEPTH: usize = 3;
/// Maximum total leaf count of a condition tree.
pub const MAX_CONDITION_LEAVES: usize = 16;
/// Maximum values in one `in` list.
pub const MAX_CONDITION_VALUES: usize = 64;
/// Maximum length of one regex pattern (`~~`/`~*`).
pub const MAX_CONDITION_REGEX_LEN: usize = 256;

/// Request dimension a condition leaf matches on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PolicyDimension {
    /// `ApiKey.team_id` (UUID).
    Team,
    /// `ApiKey.user_id` (UUID).
    Member,
    /// Authenticated api_key entry id (UUID).
    ApiKey,
    /// Model entry id (UUID). Matches the dispatched model; on a
    /// Model-Group / semantic-router / ensemble dispatch it also
    /// matches the requested parent entry, so a group's own id selects
    /// every request addressed to that group.
    Model,
    /// Model display name — the string dimension for regex/prefix
    /// matching ("every gpt-4-family alias"). Matches the dispatched
    /// model's name, and on a virtual-parent dispatch also the
    /// requested parent's name.
    ModelName,
    /// Dispatched model's `provider` (models.dev catalog id).
    Provider,
}

impl PolicyDimension {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Team => "team",
            Self::Member => "member",
            Self::ApiKey => "api_key",
            Self::Model => "model",
            Self::ModelName => "model_name",
            Self::Provider => "provider",
        }
    }

    /// Identity dimensions carry opaque ids: only equality/set
    /// operators are meaningful. String dimensions additionally admit
    /// the regex operators.
    fn is_identity(self) -> bool {
        matches!(self, Self::Team | Self::Member | Self::ApiKey | Self::Model)
    }

    /// Whether the dimension names a property of the dispatched model
    /// (vs. the caller identity). Decides the reservation point: the
    /// quota gate evaluates model-property policies where the concrete
    /// model is known (per routing target), see `aisix-proxy::quota`.
    pub fn is_model_property(self) -> bool {
        matches!(self, Self::Model | Self::ModelName | Self::Provider)
    }
}

impl std::fmt::Display for PolicyDimension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Dimension a policy's counters split on (`group_by`). The subset of
/// [`PolicyDimension`] with a stable per-request value to key a bucket
/// segment on — `model_name` is excluded (it duplicates `model` as a
/// bucket identity, less precisely).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum GroupByDimension {
    Team,
    Member,
    ApiKey,
    Model,
    Provider,
}

impl GroupByDimension {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Team => "team",
            Self::Member => "member",
            Self::ApiKey => "api_key",
            Self::Model => "model",
            Self::Provider => "provider",
        }
    }

    /// Canonical bucket-segment order. Bucket keys append `group_by`
    /// segments in this order regardless of the row's declared order,
    /// so `[team, model]` and `[model, team]` address the same bucket.
    pub const CANONICAL_ORDER: [GroupByDimension; 5] = [
        Self::Team,
        Self::Member,
        Self::ApiKey,
        Self::Model,
        Self::Provider,
    ];
}

impl std::fmt::Display for GroupByDimension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Condition leaf operator — lua-resty-expr tokens, verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub enum ConditionOperator {
    #[serde(rename = "==")]
    Eq,
    #[serde(rename = "~=")]
    Ne,
    #[serde(rename = "~~")]
    Regex,
    #[serde(rename = "~*")]
    RegexCi,
    #[serde(rename = "in")]
    In,
    #[serde(rename = "has")]
    Has,
    #[serde(rename = ">")]
    Gt,
    #[serde(rename = ">=")]
    Ge,
    #[serde(rename = "<")]
    Lt,
    #[serde(rename = "<=")]
    Le,
    #[serde(rename = "ipmatch")]
    IpMatch,
}

impl ConditionOperator {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Eq => "==",
            Self::Ne => "~=",
            Self::Regex => "~~",
            Self::RegexCi => "~*",
            Self::In => "in",
            Self::Has => "has",
            Self::Gt => ">",
            Self::Ge => ">=",
            Self::Lt => "<",
            Self::Le => "<=",
            Self::IpMatch => "ipmatch",
        }
    }

    /// Whether `value` must be a list (vs. a scalar) under this operator.
    fn takes_list(self) -> bool {
        matches!(self, Self::In | Self::Has | Self::IpMatch)
    }

    /// Whether a v1 dimension admits this operator. Identity dimensions
    /// (UUID values) take equality/set operators only; string dimensions
    /// additionally take the regex pair. The remaining tokens are wire
    /// vocabulary reserved for future array/numeric/IP dimensions.
    fn admitted_by(self, dimension: PolicyDimension) -> bool {
        match self {
            Self::Eq | Self::Ne | Self::In => true,
            Self::Regex | Self::RegexCi => !dimension.is_identity(),
            Self::Has | Self::Gt | Self::Ge | Self::Lt | Self::Le | Self::IpMatch => false,
        }
    }
}

impl std::fmt::Display for ConditionOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Group combinator — lua-resty-expr `AND`/`OR` (with `negate` for
/// `!AND`/`!OR`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ConditionLogic {
    And,
    Or,
}

/// What the policy does past its limits. v1 has the single `reject`
/// (429); the enum reserves the field for `fallback`/`queue`/`alert`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PolicyAction {
    Reject,
}

/// A leaf's comparison value: `in` (and the reserved list operators)
/// carry a string list, every scalar operator a single string.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum ConditionValue {
    One(String),
    Many(Vec<String>),
}

/// One condition leaf: `dimension operator value`, with `negate` as the
/// lua-resty-expr `!` prefix.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct PolicyCondition {
    pub dimension: PolicyDimension,
    pub operator: ConditionOperator,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub negate: bool,
    pub value: ConditionValue,
}

/// A group node combining child nodes under an explicit AND/OR.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ConditionGroup {
    pub logic: ConditionLogic,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub negate: bool,
    pub children: Vec<ConditionNode>,
}

/// A slot in a condition list: leaf or nested group. Untagged — the
/// shapes are disjoint (a leaf requires `dimension`/`operator`/`value`,
/// a group `logic`/`children`), and the schema closes both variants
/// against unknown fields in **both** validator sets because serde
/// silently swallows unknown fields inside untagged content (same
/// reasoning as `OnEmbeddingFailure` in the model schema).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum ConditionNode {
    Leaf(PolicyCondition),
    Group(ConditionGroup),
}

/// Validate a condition tree's structural caps and per-leaf shape:
/// depth ≤ [`MAX_CONDITION_DEPTH`], total leaves ≤
/// [`MAX_CONDITION_LEAVES`], groups non-empty, operator admitted by its
/// dimension, value shape matching the operator, `in` lists within
/// [`MAX_CONDITION_VALUES`] with non-empty items, regex patterns within
/// [`MAX_CONDITION_REGEX_LEN`] and compilable.
///
/// None of this is expressible in the JSON Schema (draft-07 cannot
/// count across recursion or compile regexes), so the loader and the
/// file source call it after parse and reject the whole row on error —
/// a policy is enforced exactly as written or not at all.
pub fn validate_condition_nodes(nodes: &[ConditionNode]) -> Result<(), String> {
    let mut leaves = 0usize;
    for node in nodes {
        validate_node(node, 1, &mut leaves)?;
    }
    if leaves > MAX_CONDITION_LEAVES {
        return Err(format!(
            "conditions carry {leaves} leaves; the maximum is {MAX_CONDITION_LEAVES}"
        ));
    }
    Ok(())
}

fn validate_node(node: &ConditionNode, depth: usize, leaves: &mut usize) -> Result<(), String> {
    if depth > MAX_CONDITION_DEPTH {
        return Err(format!(
            "conditions nest deeper than {MAX_CONDITION_DEPTH} levels"
        ));
    }
    match node {
        ConditionNode::Leaf(leaf) => {
            *leaves += 1;
            validate_leaf(leaf)
        }
        ConditionNode::Group(group) => {
            if group.children.is_empty() {
                return Err("condition group has no children".into());
            }
            for child in &group.children {
                validate_node(child, depth + 1, leaves)?;
            }
            Ok(())
        }
    }
}

fn validate_leaf(leaf: &PolicyCondition) -> Result<(), String> {
    let ctx = |msg: String| format!("condition on `{}`: {msg}", leaf.dimension);
    if !leaf.operator.admitted_by(leaf.dimension) {
        return Err(ctx(format!(
            "operator `{}` is not supported on this dimension",
            leaf.operator
        )));
    }
    match (&leaf.value, leaf.operator.takes_list()) {
        (ConditionValue::One(_), true) => {
            return Err(ctx(format!(
                "operator `{}` takes a list value",
                leaf.operator
            )));
        }
        (ConditionValue::Many(_), false) => {
            return Err(ctx(format!(
                "operator `{}` takes a single string value",
                leaf.operator
            )));
        }
        (ConditionValue::Many(items), true) => {
            if items.is_empty() || items.len() > MAX_CONDITION_VALUES {
                return Err(ctx(format!(
                    "`in` list must carry 1..={MAX_CONDITION_VALUES} values"
                )));
            }
            if items.iter().any(String::is_empty) {
                return Err(ctx("`in` list values must be non-empty".into()));
            }
        }
        (ConditionValue::One(v), false) => {
            if v.is_empty() {
                return Err(ctx("value must be non-empty".into()));
            }
            if matches!(
                leaf.operator,
                ConditionOperator::Regex | ConditionOperator::RegexCi
            ) {
                if v.len() > MAX_CONDITION_REGEX_LEN {
                    return Err(ctx(format!(
                        "regex pattern exceeds {MAX_CONDITION_REGEX_LEN} bytes"
                    )));
                }
                if compiled_regex(v, leaf.operator == ConditionOperator::RegexCi).is_none() {
                    return Err(ctx(format!("regex pattern {v:?} does not compile")));
                }
            }
        }
    }
    Ok(())
}

/// The request's value for each dimension at the quota gate; `None` =
/// the request does not carry the dimension.
#[derive(Debug, Clone, Copy, Default)]
pub struct ConditionInput<'a> {
    pub team: Option<&'a str>,
    pub member: Option<&'a str>,
    pub api_key: Option<&'a str>,
    pub model: Option<&'a str>,
    pub model_name: Option<&'a str>,
    pub provider: Option<&'a str>,
    /// Entry id of the caller-addressed virtual parent (routing group /
    /// ensemble / semantic router) when `model` below is a dispatch
    /// target that parent selected. A `model` leaf matches when EITHER
    /// id satisfies it (AISIX-Cloud#1267) — the group a caller
    /// addressed is as much "the model" as the member it dispatched to.
    /// `None` on direct dispatch and at the request gate (where `model`
    /// already IS the requested entry).
    pub routing_parent_model: Option<&'a str>,
    /// Display name of that parent, the `model_name` twin of
    /// [`Self::routing_parent_model`].
    pub routing_parent_model_name: Option<&'a str>,
}

impl<'a> ConditionInput<'a> {
    pub fn get(&self, dimension: PolicyDimension) -> Option<&'a str> {
        match dimension {
            PolicyDimension::Team => self.team,
            PolicyDimension::Member => self.member,
            PolicyDimension::ApiKey => self.api_key,
            PolicyDimension::Model => self.model,
            PolicyDimension::ModelName => self.model_name,
            PolicyDimension::Provider => self.provider,
        }
    }

    /// The caller-addressed parent's value for a model-property
    /// dimension, when the primary value describes a dispatch target
    /// that parent selected. Identity/provider dimensions have no
    /// parent variant (the parent shares the caller identity and
    /// carries no provider).
    fn routing_parent(&self, dimension: PolicyDimension) -> Option<&'a str> {
        match dimension {
            PolicyDimension::Model => self.routing_parent_model,
            PolicyDimension::ModelName => self.routing_parent_model_name,
            _ => None,
        }
    }

    pub fn get_group_by(&self, dimension: GroupByDimension) -> Option<&'a str> {
        match dimension {
            GroupByDimension::Team => self.team,
            GroupByDimension::Member => self.member,
            GroupByDimension::ApiKey => self.api_key,
            GroupByDimension::Model => self.model,
            GroupByDimension::Provider => self.provider,
        }
    }
}

/// Evaluate a condition list against the request (implicit AND across
/// the slice; an empty list matches everything).
pub fn eval_condition_nodes(nodes: &[ConditionNode], input: &ConditionInput<'_>) -> bool {
    nodes.iter().all(|n| eval_node(n, input))
}

fn eval_node(node: &ConditionNode, input: &ConditionInput<'_>) -> bool {
    match node {
        ConditionNode::Leaf(leaf) => eval_leaf(leaf, input),
        ConditionNode::Group(group) => {
            let raw = match group.logic {
                ConditionLogic::And => group.children.iter().all(|c| eval_node(c, input)),
                ConditionLogic::Or => group.children.iter().any(|c| eval_node(c, input)),
            };
            raw != group.negate
        }
    }
}

fn eval_leaf(leaf: &PolicyCondition, input: &ConditionInput<'_>) -> bool {
    // A request without the dimension matches neither the condition nor
    // its negation: it is outside the dimension's universe, not in the
    // complement set. (`team ∉ {T}` must not capture team-less keys.)
    let Some(var) = input.get(leaf.dimension) else {
        return false;
    };
    // Model dimensions carry a {dispatched target, requested parent}
    // pair on a routing dispatch — the request's model-identity SET.
    // Positive operators (==/in/regex) ask "does ANY identity satisfy"
    // (∃); the negative operator `~=` asks "do ALL identities differ"
    // (∀) — the two identities are distinct strings, so an ∃ reading
    // of `~=` would be vacuously true on every routed request. This
    // keeps `a ~= b` ≡ `!(a == b)` over the pair, the equivalence the
    // operator vocabulary (and the dashboard's normalization) is
    // built on; `negate` then flips the combined result, so a negated
    // leaf excluding the parent excludes every request addressed to it.
    let parent = input.routing_parent(leaf.dimension);
    let raw = if leaf.operator == ConditionOperator::Ne {
        eval_operator(leaf, var) && parent.is_none_or(|p| eval_operator(leaf, p))
    } else {
        eval_operator(leaf, var) || parent.is_some_and(|p| eval_operator(leaf, p))
    };
    raw != leaf.negate
}

fn eval_operator(leaf: &PolicyCondition, var: &str) -> bool {
    match (leaf.operator, &leaf.value) {
        (ConditionOperator::Eq, ConditionValue::One(v)) => var == v,
        (ConditionOperator::Ne, ConditionValue::One(v)) => var != v,
        (ConditionOperator::In, ConditionValue::Many(items)) => {
            items.iter().any(|item| item == var)
        }
        (ConditionOperator::Regex, ConditionValue::One(pattern)) => {
            compiled_regex(pattern, false).is_some_and(|re| re.is_match(var))
        }
        (ConditionOperator::RegexCi, ConditionValue::One(pattern)) => {
            compiled_regex(pattern, true).is_some_and(|re| re.is_match(var))
        }
        // Numeric comparisons coerce both sides like lua-resty-expr; a
        // non-numeric side never matches. Unreachable until a numeric
        // dimension exists (validation admits none), kept total so the
        // evaluator needs no protocol change when one lands.
        (
            ConditionOperator::Gt
            | ConditionOperator::Ge
            | ConditionOperator::Lt
            | ConditionOperator::Le,
            ConditionValue::One(v),
        ) => match (var.parse::<f64>(), v.parse::<f64>()) {
            (Ok(l), Ok(r)) => match leaf.operator {
                ConditionOperator::Gt => l > r,
                ConditionOperator::Ge => l >= r,
                ConditionOperator::Lt => l < r,
                _ => l <= r,
            },
            _ => false,
        },
        // `has` needs an array-valued dimension and `ipmatch` an IP
        // dimension with CIDR parsing; neither exists in v1 (validation
        // rejects them), so they conservatively never match.
        (ConditionOperator::Has | ConditionOperator::IpMatch, _) => false,
        // Value shape mismatching the operator (validation rejects it).
        _ => false,
    }
}

/// Process-wide compiled-regex caches, one per case-sensitivity
/// variant so the hot-path lookup borrows the pattern (`&str`) without
/// allocating a key. Bounded in practice by the distinct patterns
/// across configured policies; entries for retired patterns are
/// harmless. `None` is cached for uncompilable patterns so a bad
/// pattern costs one compile attempt, not one per request.
type CachedRegex = Option<Arc<regex::Regex>>;
static REGEX_CACHE_CS: Lazy<DashMap<String, CachedRegex>> = Lazy::new(DashMap::new);
static REGEX_CACHE_CI: Lazy<DashMap<String, CachedRegex>> = Lazy::new(DashMap::new);

/// Hard cap per cache. Only distinct configured patterns can insert
/// (request payloads never reach here), so this is a slow-leak backstop
/// for long-lived processes whose policies churn patterns — same spirit
/// as OpenResty's `lua_regex_cache_max_entries` (1024). Blowing the cap
/// clears the map; live patterns recompile once on next use.
const REGEX_CACHE_MAX_ENTRIES: usize = 1024;

fn compiled_regex(pattern: &str, case_insensitive: bool) -> Option<Arc<regex::Regex>> {
    let cache = if case_insensitive {
        &REGEX_CACHE_CI
    } else {
        &REGEX_CACHE_CS
    };
    if let Some(hit) = cache.get(pattern) {
        return hit.clone();
    }
    let compiled = regex::RegexBuilder::new(pattern)
        .case_insensitive(case_insensitive)
        .build()
        .ok()
        .map(Arc::new);
    if cache.len() >= REGEX_CACHE_MAX_ENTRIES {
        cache.clear();
    }
    cache.insert(pattern.to_string(), compiled.clone());
    compiled
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leaf(
        dimension: PolicyDimension,
        operator: ConditionOperator,
        value: ConditionValue,
    ) -> ConditionNode {
        ConditionNode::Leaf(PolicyCondition {
            dimension,
            operator,
            negate: false,
            value,
        })
    }

    fn neg_leaf(
        dimension: PolicyDimension,
        operator: ConditionOperator,
        value: ConditionValue,
    ) -> ConditionNode {
        ConditionNode::Leaf(PolicyCondition {
            dimension,
            operator,
            negate: true,
            value,
        })
    }

    fn one(v: &str) -> ConditionValue {
        ConditionValue::One(v.into())
    }

    fn many(vs: &[&str]) -> ConditionValue {
        ConditionValue::Many(vs.iter().map(|s| s.to_string()).collect())
    }

    fn input<'a>() -> ConditionInput<'a> {
        ConditionInput {
            team: Some("team-1"),
            member: Some("user-1"),
            api_key: Some("key-1"),
            model: Some("model-1"),
            model_name: Some("gpt-4.1-prod"),
            provider: Some("openai"),
            routing_parent_model: None,
            routing_parent_model_name: None,
        }
    }

    /// `input()` as the per-target gate of a routing dispatch sees it:
    /// the target as the primary values, the caller-addressed group as
    /// the parent pair.
    fn routed_input<'a>() -> ConditionInput<'a> {
        ConditionInput {
            routing_parent_model: Some("group-1"),
            routing_parent_model_name: Some("chat-group"),
            ..input()
        }
    }

    #[test]
    fn wire_shape_matches_the_rfc() {
        // The exact JSON the RFC and cp-api produce: a leaf row plus an
        // OR group, snake_case dimensions, lua-resty-expr operator
        // tokens, `negate` omitted when false.
        let nodes: Vec<ConditionNode> = serde_json::from_value(serde_json::json!([
            { "dimension": "team", "operator": "in", "value": ["team-1"] },
            { "logic": "or", "children": [
                { "dimension": "model_name", "operator": "~~", "value": "^gpt-4\\.1" },
                { "dimension": "provider", "operator": "==", "value": "anthropic" }
            ]}
        ]))
        .unwrap();
        assert!(matches!(nodes[0], ConditionNode::Leaf(_)));
        assert!(matches!(nodes[1], ConditionNode::Group(_)));
        validate_condition_nodes(&nodes).unwrap();
        assert!(eval_condition_nodes(&nodes, &input()));
        // Round-trips without inventing fields.
        let back = serde_json::to_value(&nodes).unwrap();
        assert_eq!(back[0]["dimension"], "team");
        assert!(back[0].get("negate").is_none());
        assert_eq!(back[1]["logic"], "or");
    }

    #[test]
    fn implicit_and_across_top_level() {
        let nodes = vec![
            leaf(
                PolicyDimension::Team,
                ConditionOperator::In,
                many(&["team-1"]),
            ),
            leaf(
                PolicyDimension::Provider,
                ConditionOperator::Eq,
                one("anthropic"),
            ),
        ];
        // team matches, provider does not → AND fails.
        assert!(!eval_condition_nodes(&nodes, &input()));
    }

    #[test]
    fn or_group_matches_on_any_branch() {
        let nodes = vec![ConditionNode::Group(ConditionGroup {
            logic: ConditionLogic::Or,
            negate: false,
            children: vec![
                leaf(
                    PolicyDimension::Provider,
                    ConditionOperator::Eq,
                    one("anthropic"),
                ),
                leaf(
                    PolicyDimension::ModelName,
                    ConditionOperator::Regex,
                    one("^gpt-4"),
                ),
            ],
        })];
        assert!(eval_condition_nodes(&nodes, &input()));
    }

    #[test]
    fn group_negate_is_not_and_not_or() {
        let and_group = |negate| {
            vec![ConditionNode::Group(ConditionGroup {
                logic: ConditionLogic::And,
                negate,
                children: vec![leaf(
                    PolicyDimension::Team,
                    ConditionOperator::Eq,
                    one("team-1"),
                )],
            })]
        };
        assert!(eval_condition_nodes(&and_group(false), &input()));
        assert!(!eval_condition_nodes(&and_group(true), &input()));
    }

    #[test]
    fn leaf_negate_inverts_membership() {
        let nodes = vec![neg_leaf(
            PolicyDimension::Team,
            ConditionOperator::In,
            many(&["other-team"]),
        )];
        // team-1 ∉ {other-team} → negated `in` matches.
        assert!(eval_condition_nodes(&nodes, &input()));
    }

    #[test]
    fn model_leaf_matches_routing_parent_id() {
        // AISIX-Cloud#1267: `model in [group uuid]` must select requests
        // dispatched THROUGH the group even though the per-target gate's
        // primary value is the member id.
        let nodes = vec![leaf(
            PolicyDimension::Model,
            ConditionOperator::In,
            many(&["group-1"]),
        )];
        assert!(eval_condition_nodes(&nodes, &routed_input()));
        // Direct dispatch to the member (no parent): the group condition
        // must NOT capture it.
        assert!(!eval_condition_nodes(&nodes, &input()));
    }

    #[test]
    fn model_name_leaf_matches_routing_parent_name() {
        let nodes = vec![leaf(
            PolicyDimension::ModelName,
            ConditionOperator::Regex,
            one("^chat-"),
        )];
        assert!(eval_condition_nodes(&nodes, &routed_input()));
        assert!(!eval_condition_nodes(&nodes, &input()));
    }

    #[test]
    fn negated_model_leaf_excludes_parent_dispatch() {
        // `!(model in [group-1])` means "everything except the group":
        // negate flips the COMBINED pair result, so a request routed via
        // the group is excluded — not (absurdly) matched because the
        // member id alone misses the set.
        let nodes = vec![neg_leaf(
            PolicyDimension::Model,
            ConditionOperator::In,
            many(&["group-1"]),
        )];
        assert!(!eval_condition_nodes(&nodes, &routed_input()));
        // The same member reached directly stays matched.
        assert!(eval_condition_nodes(&nodes, &input()));
    }

    #[test]
    fn ne_leaf_requires_both_identities_to_differ() {
        // `model ~= group-1` on a request ADDRESSED to group-1: the
        // dispatched member differs from the value, but ∃-combining
        // would make `~=` vacuously true on every routed request (the
        // two identities are distinct strings). ∀-combining keeps the
        // exclusion meaningful…
        let ne = |v: &str| vec![leaf(PolicyDimension::Model, ConditionOperator::Ne, one(v))];
        assert!(!eval_condition_nodes(&ne("group-1"), &routed_input()));
        assert!(!eval_condition_nodes(&ne("model-1"), &routed_input()));
        assert!(eval_condition_nodes(&ne("other"), &routed_input()));
        // …and preserves `a ~= b` ≡ `!(a == b)` over the pair.
        let neg_eq = |v: &str| {
            vec![neg_leaf(
                PolicyDimension::Model,
                ConditionOperator::Eq,
                one(v),
            )]
        };
        for v in ["group-1", "model-1", "other"] {
            assert_eq!(
                eval_condition_nodes(&ne(v), &routed_input()),
                eval_condition_nodes(&neg_eq(v), &routed_input()),
                "~= and !(==) diverged for {v}"
            );
        }
        // Direct dispatch (no parent): plain not-equal, unchanged.
        assert!(eval_condition_nodes(&ne("group-1"), &input()));
        assert!(!eval_condition_nodes(&ne("model-1"), &input()));
    }

    #[test]
    fn member_leaf_still_matches_through_parent() {
        // The 1087 principle survives the pair: a member-id condition
        // keeps matching when the member is reached via a group.
        let nodes = vec![leaf(
            PolicyDimension::Model,
            ConditionOperator::In,
            many(&["model-1"]),
        )];
        assert!(eval_condition_nodes(&nodes, &routed_input()));
    }

    #[test]
    fn parent_pair_never_leaks_into_identity_dimensions() {
        // A team leaf must not consult the parent pair even when set.
        let nodes = vec![leaf(
            PolicyDimension::Team,
            ConditionOperator::In,
            many(&["group-1"]),
        )];
        assert!(!eval_condition_nodes(&nodes, &routed_input()));
    }

    #[test]
    fn missing_dimension_is_false_even_negated() {
        let no_team = ConditionInput {
            team: None,
            ..input()
        };
        let plain = vec![leaf(
            PolicyDimension::Team,
            ConditionOperator::In,
            many(&["team-1"]),
        )];
        let negated = vec![neg_leaf(
            PolicyDimension::Team,
            ConditionOperator::In,
            many(&["team-1"]),
        )];
        assert!(!eval_condition_nodes(&plain, &no_team));
        // A team-less request is not in the complement either.
        assert!(!eval_condition_nodes(&negated, &no_team));
    }

    #[test]
    fn missing_dimension_still_matches_via_or_sibling() {
        let nodes = vec![ConditionNode::Group(ConditionGroup {
            logic: ConditionLogic::Or,
            negate: false,
            children: vec![
                leaf(
                    PolicyDimension::ModelName,
                    ConditionOperator::Regex,
                    one("^gpt"),
                ),
                leaf(PolicyDimension::Team, ConditionOperator::Eq, one("team-1")),
            ],
        })];
        let no_model = ConditionInput {
            model: None,
            model_name: None,
            provider: None,
            ..input()
        };
        assert!(eval_condition_nodes(&nodes, &no_model));
    }

    #[test]
    fn case_insensitive_regex_variant() {
        let nodes = vec![leaf(
            PolicyDimension::ModelName,
            ConditionOperator::RegexCi,
            one("^GPT-4"),
        )];
        assert!(eval_condition_nodes(&nodes, &input()));
    }

    #[test]
    fn empty_conditions_match_everything() {
        assert!(eval_condition_nodes(&[], &input()));
        assert!(eval_condition_nodes(&[], &ConditionInput::default()));
    }

    #[test]
    fn depth_cap_rejects_level_four() {
        let mut node = leaf(PolicyDimension::Team, ConditionOperator::Eq, one("t"));
        for _ in 0..3 {
            node = ConditionNode::Group(ConditionGroup {
                logic: ConditionLogic::And,
                negate: false,
                children: vec![node],
            });
        }
        // Groups at depth 1..=3 put the leaf at depth 4.
        let err = validate_condition_nodes(&[node]).unwrap_err();
        assert!(err.contains("deeper than 3"), "{err}");
    }

    #[test]
    fn leaf_cap_rejects_seventeen() {
        let nodes: Vec<ConditionNode> = (0..17)
            .map(|_| leaf(PolicyDimension::Team, ConditionOperator::Eq, one("t")))
            .collect();
        let err = validate_condition_nodes(&nodes).unwrap_err();
        assert!(err.contains("17 leaves"), "{err}");
    }

    #[test]
    fn empty_group_rejected() {
        let nodes = vec![ConditionNode::Group(ConditionGroup {
            logic: ConditionLogic::Or,
            negate: false,
            children: vec![],
        })];
        assert!(validate_condition_nodes(&nodes).is_err());
    }

    #[test]
    fn identity_dimension_rejects_regex() {
        let nodes = vec![leaf(
            PolicyDimension::Team,
            ConditionOperator::Regex,
            one("^t"),
        )];
        let err = validate_condition_nodes(&nodes).unwrap_err();
        assert!(err.contains("not supported"), "{err}");
    }

    #[test]
    fn reserved_operators_rejected_on_every_v1_dimension() {
        for op in [
            ConditionOperator::Has,
            ConditionOperator::Gt,
            ConditionOperator::Ge,
            ConditionOperator::Lt,
            ConditionOperator::Le,
            ConditionOperator::IpMatch,
        ] {
            let value = if op.takes_list() {
                many(&["v"])
            } else {
                one("1")
            };
            let nodes = vec![leaf(PolicyDimension::ModelName, op, value)];
            assert!(
                validate_condition_nodes(&nodes).is_err(),
                "operator {op} must be rejected in v1"
            );
        }
    }

    #[test]
    fn value_shape_must_match_operator() {
        // `in` with a scalar.
        let nodes = vec![leaf(PolicyDimension::Team, ConditionOperator::In, one("t"))];
        assert!(validate_condition_nodes(&nodes).is_err());
        // `==` with a list.
        let nodes = vec![leaf(
            PolicyDimension::Team,
            ConditionOperator::Eq,
            many(&["t"]),
        )];
        assert!(validate_condition_nodes(&nodes).is_err());
    }

    #[test]
    fn bad_regex_rejected_at_validation_and_false_at_eval() {
        let nodes = vec![leaf(
            PolicyDimension::ModelName,
            ConditionOperator::Regex,
            one("(unclosed"),
        )];
        assert!(validate_condition_nodes(&nodes).is_err());
        // Defense in depth: were such a row ever evaluated, it matches
        // nothing rather than everything.
        assert!(!eval_condition_nodes(&nodes, &input()));
    }

    #[test]
    fn oversized_in_list_rejected() {
        let items: Vec<String> = (0..65).map(|i| format!("v{i}")).collect();
        let nodes = vec![leaf(
            PolicyDimension::Team,
            ConditionOperator::In,
            ConditionValue::Many(items),
        )];
        assert!(validate_condition_nodes(&nodes).is_err());
    }

    #[test]
    fn unknown_operator_token_fails_deserialize() {
        let r: Result<Vec<ConditionNode>, _> = serde_json::from_value(serde_json::json!([
            { "dimension": "team", "operator": "regex", "value": "^t" }
        ]));
        assert!(r.is_err());
    }
}
