//! Budget enforcement from the `budgets` documents the control plane pushes.
//!
//! The control plane owns the spend counters and marks a budget document
//! `exhausted` once its limit is reached; every decision here is local, so a
//! request never waits on the control plane and an outage cannot turn into
//! blanket rejections. An exhausted mark names the period it was set in and
//! stops applying at the next period boundary on its own, so a block lifts on
//! time even while the control plane is unreachable.
//!
//! Two kinds of target:
//!
//! - **Key-level** scopes (everything except `provider_key`) are judged once
//!   per request against the authenticated key, by [`BudgetGate::key_verdict`].
//! - **`provider_key`** budgets cap one upstream credential, so they are judged
//!   per candidate upstream by [`BudgetGate::provider_key_verdict`]: an
//!   exhausted candidate is skipped like one outside its client-IP allowlist,
//!   and a request that has no candidate left is rejected.
//!
//! Under `budget_policy.fail_mode: closed`, once usage delivery to the control
//! plane has failed for `closed_grace_seconds` the gateway can no longer trust
//! that a budget it sees as not exhausted really is not, so everything a
//! budget applies to is answered `503 budget_unverified` (or skipped, for a
//! provider key) instead.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use aisix_core::models::{Budget, BudgetFailMode, BudgetPeriod, BudgetScope};
use aisix_core::resource::ResourceEntry;
use aisix_core::snapshot::ResourceTable;
use aisix_core::AisixSnapshot;
use arc_swap::ArcSwapOption;
use chrono::{DateTime, Datelike, NaiveTime, Utc};

use crate::auth::AuthenticatedKey;
use crate::error::ProxyError;

type Entry = Arc<ResourceEntry<Budget>>;

/// Customer-facing detail for a budget denial, lifted into the 429 `error`
/// block so a programmatic client can see which budget tripped (`scope` /
/// `scope_ref`) and by how much (`limit_usd` / `spent_usd`), not just the
/// human `message`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BudgetReason {
    pub message: String,
    pub scope: Option<String>,
    pub scope_ref: Option<String>,
    pub limit_usd: Option<String>,
    pub spent_usd: Option<String>,
    /// Feeds `aisix_budget_remaining_usd`; not part of the response body.
    pub remaining_usd: Option<String>,
    pub period: Option<String>,
    pub period_resets_at: Option<String>,
    pub retry_after_seconds: Option<u64>,
}

impl BudgetReason {
    #[cfg(test)]
    pub(crate) fn message_only(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            ..Default::default()
        }
    }

    fn gauges(&self) -> aisix_obs::BudgetGauges {
        let amount = |v: &Option<String>| v.as_deref().and_then(|s| s.parse::<f64>().ok());
        aisix_obs::BudgetGauges {
            limit_usd: amount(&self.limit_usd),
            spent_usd: amount(&self.spent_usd),
            remaining_usd: amount(&self.remaining_usd),
            reset_seconds: self.retry_after_seconds,
        }
    }
}

/// The budget documents, indexed by what each one is matched against.
/// Rebuilt only when the `budgets` table changes.
pub(crate) struct BudgetIndex {
    generation: u64,
    /// `org` and `environment`: every key.
    everyone: Vec<Entry>,
    api_key: HashMap<String, Vec<Entry>>,
    member: HashMap<String, Vec<Entry>>,
    team: HashMap<String, Vec<Entry>>,
    team_member: HashMap<String, Vec<Entry>>,
    /// Keyed by tag key; the value is compared per request.
    tag: HashMap<String, Vec<Entry>>,
    provider_key: HashMap<String, Vec<Entry>>,
}

impl BudgetIndex {
    fn build(table: &ResourceTable<Budget>) -> Self {
        let mut index = Self {
            generation: table.generation(),
            everyone: Vec::new(),
            api_key: HashMap::new(),
            member: HashMap::new(),
            team: HashMap::new(),
            team_member: HashMap::new(),
            tag: HashMap::new(),
            provider_key: HashMap::new(),
        };
        let mut entries = table.entries();
        // Deterministic order, so ties between equally specific budgets
        // resolve the same way on every request.
        entries.sort_by(|a, b| a.id.cmp(&b.id));
        for entry in entries {
            let budget = &entry.value;
            let by_ref = match budget.scope {
                BudgetScope::Org | BudgetScope::Environment => {
                    index.everyone.push(entry);
                    continue;
                }
                BudgetScope::Tag => {
                    if let Some(tag) = &budget.tag {
                        index.tag.entry(tag.key.clone()).or_default().push(entry);
                    }
                    continue;
                }
                BudgetScope::ApiKey => &mut index.api_key,
                BudgetScope::Member => &mut index.member,
                BudgetScope::Team => &mut index.team,
                BudgetScope::TeamMember => &mut index.team_member,
                BudgetScope::ProviderKey => &mut index.provider_key,
            };
            if let Some(scope_ref) = &budget.scope_ref {
                by_ref.entry(scope_ref.clone()).or_default().push(entry);
            }
        }
        index
    }

    /// The key-level budgets that apply to `auth`'s key.
    fn applicable<'a>(&'a self, auth: &'a AuthenticatedKey) -> impl Iterator<Item = &'a Entry> {
        let key = &auth.entry.value;
        let by = |map: &'a HashMap<String, Vec<Entry>>, id: Option<&str>| {
            id.and_then(|id| map.get(id))
                .map(Vec::as_slice)
                .unwrap_or_default()
                .iter()
        };
        let team_member = match (key.team_id.as_deref(), key.user_id.as_deref()) {
            (Some(team), Some(_)) => Some(team),
            _ => None,
        };
        let tags = auth.tags.iter().flat_map(move |(k, v)| {
            self.tag
                .get(k)
                .map(Vec::as_slice)
                .unwrap_or_default()
                .iter()
                .filter(move |e| e.value.tag.as_ref().is_some_and(|t| t.value == *v))
        });
        self.everyone
            .iter()
            .chain(by(&self.api_key, Some(auth.entry.id.as_str())))
            .chain(by(&self.member, key.user_id.as_deref()))
            .chain(by(&self.team, key.team_id.as_deref()))
            .chain(by(&self.team_member, team_member))
            .chain(tags)
    }
}

#[derive(Default)]
pub(crate) struct LiveBudgetIndex {
    cached: ArcSwapOption<BudgetIndex>,
}

impl LiveBudgetIndex {
    fn for_table(&self, table: &ResourceTable<Budget>) -> Arc<BudgetIndex> {
        if let Some(index) = self.cached.load_full() {
            if index.generation == table.generation() {
                return index;
            }
        }
        let index = Arc::new(BudgetIndex::build(table));
        self.cached.store(Some(Arc::clone(&index)));
        index
    }
}

/// The answer for one key or one provider key.
#[derive(Debug)]
pub(crate) enum Verdict {
    Allow,
    Exceeded(Box<BudgetReason>),
    /// Strict mode with usage delivery failing past the grace period, and a
    /// budget applies.
    Unverified,
}

impl Verdict {
    fn into_result(self) -> Result<(), ProxyError> {
        match self {
            Verdict::Allow => Ok(()),
            Verdict::Exceeded(reason) => Err(ProxyError::BudgetExceeded(reason)),
            Verdict::Unverified => Err(ProxyError::BudgetUnverified),
        }
    }
}

/// Everything one request's budget decisions read, taken once.
pub(crate) struct BudgetGate {
    index: Arc<BudgetIndex>,
    now: DateTime<Utc>,
    /// Strict mode is in force: delivery has been failing for at least the
    /// policy's grace period.
    unverified: bool,
}

/// The gate for a request served from `snapshot`.
pub(crate) fn gate(state: &crate::ProxyState, snapshot: &AisixSnapshot) -> BudgetGate {
    BudgetGate {
        index: state.budgets.for_table(&snapshot.budgets),
        now: Utc::now(),
        unverified: strict_grace(snapshot)
            .is_some_and(|grace| state.metrics.usage_delivery().unhealthy_for() >= grace),
    }
}

/// The grace period when the environment's policy is `closed`. Two rows can
/// only be a stale key; the stricter reading wins rather than whichever
/// sorts first.
fn strict_grace(snapshot: &AisixSnapshot) -> Option<Duration> {
    snapshot
        .budget_policy
        .entries()
        .iter()
        .filter(|e| e.value.fail_mode == BudgetFailMode::Closed)
        .map(|e| Duration::from_secs(e.value.closed_grace_seconds))
        .min()
}

impl BudgetGate {
    /// The key-level budgets for `auth`. When several are exhausted, the
    /// most specific is reported.
    pub(crate) fn key_verdict(&self, auth: &AuthenticatedKey) -> Verdict {
        let user_id = auth.entry.value.user_id.as_deref();
        let mut applies = false;
        let mut denied: Option<(u8, BudgetScope, &Entry, &aisix_core::models::BudgetReason)> = None;
        for entry in self.index.applicable(auth) {
            applies = true;
            let budget = &entry.value;
            let Some(reason) = self.in_force_reason(budget, user_id) else {
                continue;
            };
            let rank = scope_rank(budget.scope);
            // Higher rank wins; on a tie the scope that sorts first by name,
            // as the control plane breaks it.
            let better = denied.as_ref().is_none_or(|(r, scope, ..)| {
                rank > *r || (rank == *r && budget.scope.as_str() < scope.as_str())
            });
            if better {
                denied = Some((rank, budget.scope, entry, reason));
            }
        }
        match denied {
            Some((_, _, entry, reason)) => {
                Verdict::Exceeded(Box::new(self.reason(&entry.value, reason)))
            }
            None if applies && self.unverified => Verdict::Unverified,
            None => Verdict::Allow,
        }
    }

    /// The `provider_key` budgets of the upstream credential `provider_key_id`.
    pub(crate) fn provider_key_verdict(&self, provider_key_id: &str) -> Verdict {
        let Some(budgets) = self.index.provider_key.get(provider_key_id) else {
            return Verdict::Allow;
        };
        for entry in budgets {
            if let Some(reason) = self.in_force_reason(&entry.value, None) {
                return Verdict::Exceeded(Box::new(self.reason(&entry.value, reason)));
            }
        }
        if self.unverified {
            Verdict::Unverified
        } else {
            Verdict::Allow
        }
    }

    /// [`Self::provider_key_verdict`] for a model, as an error. A model with
    /// no provider key has nothing to cap.
    pub(crate) fn check_model(&self, model: &aisix_core::Model) -> Result<(), ProxyError> {
        match model.provider_key_id.as_deref() {
            Some(id) => self.provider_key_verdict(id).into_result(),
            None => Ok(()),
        }
    }

    /// [`Self::provider_key_verdict`] as an error.
    pub(crate) fn check_provider_key(&self, provider_key_id: &str) -> Result<(), ProxyError> {
        self.provider_key_verdict(provider_key_id).into_result()
    }

    /// The exhausted mark that blocks `budget` now, if any. `user_id` selects
    /// the member's own mark on a `team_member` budget.
    fn in_force_reason<'a>(
        &self,
        budget: &'a Budget,
        user_id: Option<&str>,
    ) -> Option<&'a aisix_core::models::BudgetReason> {
        let current = period_start(self.now, budget.period);
        let in_force = |start: Option<&str>| {
            start.is_none_or(|s| {
                DateTime::parse_from_rfc3339(s).is_ok_and(|t| t.with_timezone(&Utc) == current)
            })
        };
        if let Some(exhausted) = &budget.exhausted {
            if in_force(exhausted.period_start.as_deref()) {
                return Some(&exhausted.reason);
            }
        }
        if budget.scope == BudgetScope::TeamMember {
            let member = budget.exhausted_members.get(user_id?)?;
            if in_force(Some(&member.period_start)) {
                return Some(&member.reason);
            }
        }
        None
    }

    fn reason(&self, budget: &Budget, doc: &aisix_core::models::BudgetReason) -> BudgetReason {
        let resets_at = next_period_start(period_start(self.now, budget.period), budget.period);
        // Amounts are restated in the dollar-string form the response has
        // always carried.
        let amount = |v: &Option<String>| {
            v.as_deref()
                .and_then(|s| s.trim().parse::<f64>().ok())
                .map(|n| format!("{n:.2}"))
        };
        let scope = doc
            .scope
            .clone()
            .unwrap_or_else(|| budget.scope.as_str().to_string());
        BudgetReason {
            message: doc
                .message
                .clone()
                .filter(|m| !m.is_empty())
                .unwrap_or_else(|| format!("{} budget exceeded", scope.replace('_', " "))),
            scope_ref: doc.scope_ref.clone().or_else(|| budget.scope_ref.clone()),
            scope: Some(scope),
            limit_usd: amount(&doc.limit_usd),
            spent_usd: amount(&doc.spent_usd),
            remaining_usd: amount(&doc.remaining_usd),
            period: Some(budget.period.as_str().to_string()),
            period_resets_at: Some(resets_at.format("%Y-%m-%dT%H:%M:%SZ").to_string()),
            retry_after_seconds: Some((resets_at - self.now).num_seconds().max(0) as u64),
        }
    }
}

/// The control plane's precedence among scopes when several deny at once:
/// most specific first.
fn scope_rank(scope: BudgetScope) -> u8 {
    match scope {
        BudgetScope::ApiKey => 7,
        BudgetScope::Member | BudgetScope::TeamMember => 6,
        BudgetScope::Team => 5,
        BudgetScope::Tag => 4,
        BudgetScope::ProviderKey => 3,
        BudgetScope::Environment => 2,
        BudgetScope::Org => 1,
    }
}

/// Start of the UTC calendar period containing `now`.
pub(crate) fn period_start(now: DateTime<Utc>, period: BudgetPeriod) -> DateTime<Utc> {
    let day = now.date_naive();
    let first = match period {
        BudgetPeriod::Day => day,
        BudgetPeriod::Week => {
            day - chrono::Days::new(u64::from(day.weekday().num_days_from_monday()))
        }
        BudgetPeriod::Month => day.with_day(1).expect("the 1st exists in every month"),
    };
    first.and_time(NaiveTime::MIN).and_utc()
}

fn next_period_start(start: DateTime<Utc>, period: BudgetPeriod) -> DateTime<Utc> {
    match period {
        BudgetPeriod::Day => start + chrono::Days::new(1),
        BudgetPeriod::Week => start + chrono::Days::new(7),
        BudgetPeriod::Month => start + chrono::Months::new(1),
    }
}

/// The key-level budget decision for one request: refreshes the
/// `aisix_budget_*` gauges for the key and rejects the request when a budget
/// that applies to the key is exhausted (or cannot be verified).
pub(crate) fn enforce_key(
    state: &crate::ProxyState,
    snapshot: &AisixSnapshot,
    auth: &AuthenticatedKey,
) -> Result<(), ProxyError> {
    let verdict = gate(state, snapshot).key_verdict(auth);
    let labels = crate::quota::budget_labels(auth);
    match &verdict {
        Verdict::Exceeded(reason) => state.metrics.set_budget_gauges(labels, reason.gauges()),
        Verdict::Allow | Verdict::Unverified => state.metrics.clear_budget_gauges(labels),
    }
    verdict.into_result()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn at(y: i32, m: u32, d: u32, h: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(y, m, d, h, 30, 0).unwrap()
    }

    #[test]
    fn periods_follow_the_utc_calendar() {
        // Thursday 2026-10-29.
        let now = at(2026, 10, 29, 13);
        assert_eq!(
            period_start(now, BudgetPeriod::Day),
            at(2026, 10, 29, 0) - chrono::Duration::minutes(30)
        );
        assert_eq!(
            period_start(now, BudgetPeriod::Week).to_rfc3339(),
            "2026-10-26T00:00:00+00:00"
        );
        assert_eq!(
            period_start(now, BudgetPeriod::Month).to_rfc3339(),
            "2026-10-01T00:00:00+00:00"
        );
        // A Monday is its own week's start.
        assert_eq!(
            period_start(at(2026, 10, 26, 0), BudgetPeriod::Week).to_rfc3339(),
            "2026-10-26T00:00:00+00:00"
        );
        assert_eq!(
            next_period_start(
                period_start(at(2026, 12, 31, 23), BudgetPeriod::Month),
                BudgetPeriod::Month
            )
            .to_rfc3339(),
            "2027-01-01T00:00:00+00:00"
        );
    }

    fn snapshot_with(budgets: &[(&str, serde_json::Value)]) -> AisixSnapshot {
        let snap = AisixSnapshot::new();
        for (id, doc) in budgets {
            let budget: Budget = serde_json::from_value(doc.clone()).unwrap();
            snap.budgets.insert(ResourceEntry::new(*id, budget, 1));
        }
        snap.teams.insert(ResourceEntry::new(
            "team-1",
            serde_json::from_value(
                serde_json::json!({"name": "T", "tags": {"dept": "team-dept", "cc": "team-cc"}}),
            )
            .unwrap(),
            1,
        ));
        snap.users.insert(ResourceEntry::new(
            "user-1",
            serde_json::from_value(
                serde_json::json!({"name": "U", "tags": {"dept": "user-dept", "region": "eu"}}),
            )
            .unwrap(),
            1,
        ));
        snap
    }

    fn key(snap: &AisixSnapshot, id: &str, doc: serde_json::Value) -> AuthenticatedKey {
        let mut doc = doc;
        doc["key_hash"] = serde_json::json!("h");
        let entry = Arc::new(ResourceEntry::new(
            id,
            serde_json::from_value(doc).unwrap(),
            1,
        ));
        AuthenticatedKey::new(snap, entry, None, false)
    }

    fn gate_at(snap: &AisixSnapshot, now: DateTime<Utc>, unverified: bool) -> BudgetGate {
        BudgetGate {
            index: Arc::new(BudgetIndex::build(&snap.budgets)),
            now,
            unverified,
        }
    }

    fn exhausted(
        scope: &str,
        target: serde_json::Value,
        period_start: Option<&str>,
    ) -> serde_json::Value {
        let mut exhausted =
            serde_json::json!({"reason": {"message": format!("{scope} out"), "scope": scope}});
        if let Some(start) = period_start {
            exhausted["period_start"] = serde_json::json!(start);
        }
        let mut doc =
            serde_json::json!({"scope": scope, "period": "month", "exhausted": exhausted});
        doc.as_object_mut()
            .unwrap()
            .extend(target.as_object().unwrap().clone());
        doc
    }

    fn denied_scope(v: Verdict) -> Option<String> {
        match v {
            Verdict::Exceeded(r) => r.scope,
            _ => None,
        }
    }

    #[test]
    fn the_most_specific_exhausted_budget_is_reported() {
        let snap = snapshot_with(&[
            (
                "b-org",
                exhausted("org", serde_json::json!({"scope_ref": "o"}), None),
            ),
            (
                "b-team",
                exhausted("team", serde_json::json!({"scope_ref": "team-1"}), None),
            ),
            (
                "b-member",
                exhausted("member", serde_json::json!({"scope_ref": "user-1"}), None),
            ),
            (
                "b-tm",
                exhausted(
                    "team_member",
                    serde_json::json!({"scope_ref": "team-1"}),
                    None,
                ),
            ),
        ]);
        let now = at(2026, 10, 29, 13);
        let k = key(
            &snap,
            "k-1",
            serde_json::json!({"team_id": "team-1", "user_id": "user-1"}),
        );
        // member and team_member share a rank; the control plane keeps the
        // one whose scope sorts first.
        assert_eq!(
            denied_scope(gate_at(&snap, now, false).key_verdict(&k)).as_deref(),
            Some("member")
        );
        // A key of no team and no member meets only the org budget.
        let lone = key(&snap, "k-2", serde_json::json!({}));
        assert_eq!(
            denied_scope(gate_at(&snap, now, false).key_verdict(&lone)).as_deref(),
            Some("org")
        );
    }

    #[test]
    fn an_exhausted_mark_lifts_at_the_period_boundary() {
        let snap = snapshot_with(&[(
            "b-1",
            exhausted(
                "api_key",
                serde_json::json!({"scope_ref": "k-1"}),
                Some("2026-10-01T00:00:00Z"),
            ),
        )]);
        let k = key(&snap, "k-1", serde_json::json!({}));
        assert!(matches!(
            gate_at(&snap, at(2026, 10, 31, 23), false).key_verdict(&k),
            Verdict::Exceeded(_)
        ));
        assert!(matches!(
            gate_at(&snap, at(2026, 11, 1, 0), false).key_verdict(&k),
            Verdict::Allow
        ));
        // An unparsable start never matches.
        let snap = snapshot_with(&[(
            "b-1",
            exhausted(
                "api_key",
                serde_json::json!({"scope_ref": "k-1"}),
                Some("October"),
            ),
        )]);
        assert!(matches!(
            gate_at(&snap, at(2026, 10, 2, 0), false).key_verdict(&k),
            Verdict::Allow
        ));
    }

    #[test]
    fn team_member_blocks_per_member() {
        let doc = serde_json::json!({
            "scope": "team_member", "scope_ref": "team-1", "period": "day",
            "exhausted_members": {"user-1": {"period_start": "2026-10-29T00:00:00Z", "reason": {"message": "you"}}}
        });
        let snap = snapshot_with(&[("b-1", doc)]);
        let now = at(2026, 10, 29, 13);
        let blocked = key(
            &snap,
            "k-1",
            serde_json::json!({"team_id": "team-1", "user_id": "user-1"}),
        );
        let other = key(
            &snap,
            "k-2",
            serde_json::json!({"team_id": "team-1", "user_id": "user-2"}),
        );
        let no_member = key(&snap, "k-3", serde_json::json!({"team_id": "team-1"}));
        let gate = gate_at(&snap, now, false);
        assert!(matches!(gate.key_verdict(&blocked), Verdict::Exceeded(_)));
        assert!(matches!(gate.key_verdict(&other), Verdict::Allow));
        assert!(matches!(gate.key_verdict(&no_member), Verdict::Allow));
        // Strict mode: the budget applies to a member of the team only.
        let strict = gate_at(&snap, now, true);
        assert!(matches!(strict.key_verdict(&other), Verdict::Unverified));
        assert!(matches!(strict.key_verdict(&no_member), Verdict::Allow));
    }

    #[test]
    fn a_tag_budget_matches_the_effective_tag_value() {
        let snap = snapshot_with(&[
            (
                "b-team",
                exhausted(
                    "tag",
                    serde_json::json!({"tag": {"key": "dept", "value": "team-dept"}}),
                    None,
                ),
            ),
            (
                "b-user",
                exhausted(
                    "tag",
                    serde_json::json!({"tag": {"key": "dept", "value": "user-dept"}}),
                    None,
                ),
            ),
        ]);
        let now = at(2026, 10, 29, 13);
        let gate = gate_at(&snap, now, false);
        // The team's value beats the member's; the member's budget does not apply.
        let k = key(
            &snap,
            "k-1",
            serde_json::json!({"team_id": "team-1", "user_id": "user-1"}),
        );
        let Verdict::Exceeded(r) = gate.key_verdict(&k) else {
            panic!("team value applies")
        };
        assert_eq!(r.message, "tag out");
        // The key's own value beats both.
        let own = key(
            &snap,
            "k-2",
            serde_json::json!({"team_id": "team-1", "user_id": "user-1", "tags": {"dept": "mine"}}),
        );
        assert!(matches!(gate.key_verdict(&own), Verdict::Allow));
        // With no team, the member's value applies.
        let member_only = key(&snap, "k-3", serde_json::json!({"user_id": "user-1"}));
        assert!(matches!(
            gate.key_verdict(&member_only),
            Verdict::Exceeded(_)
        ));
    }

    #[test]
    fn the_reason_is_restated_with_the_gateways_own_reset() {
        let mut doc = exhausted(
            "api_key",
            serde_json::json!({"scope_ref": "k-1"}),
            Some("2026-10-01T00:00:00Z"),
        );
        doc["exhausted"]["reason"] = serde_json::json!({
            "message": "api key budget 'x' exceeded", "scope": "api_key", "scope_ref": "k-1",
            "limit_usd": "10", "spent_usd": "10.5", "remaining_usd": "-0.5", "period": "month",
            "period_resets_at": "1999-01-01T00:00:00Z", "retry_after_seconds": 5
        });
        let snap = snapshot_with(&[("b-1", doc)]);
        let now = Utc.with_ymd_and_hms(2026, 10, 31, 23, 0, 0).unwrap();
        let k = key(&snap, "k-1", serde_json::json!({}));
        let Verdict::Exceeded(r) = gate_at(&snap, now, false).key_verdict(&k) else {
            panic!()
        };
        assert_eq!(r.limit_usd.as_deref(), Some("10.00"));
        assert_eq!(r.spent_usd.as_deref(), Some("10.50"));
        assert_eq!(r.period_resets_at.as_deref(), Some("2026-11-01T00:00:00Z"));
        assert_eq!(r.retry_after_seconds, Some(3600));
        assert_eq!(r.gauges().remaining_usd, Some(-0.5));
    }

    #[test]
    fn provider_key_budgets_are_judged_per_credential() {
        let snap = snapshot_with(&[
            (
                "b-1",
                exhausted(
                    "provider_key",
                    serde_json::json!({"scope_ref": "pk-out"}),
                    None,
                ),
            ),
            (
                "b-2",
                serde_json::json!({"scope": "provider_key", "scope_ref": "pk-ok", "period": "day"}),
            ),
        ]);
        let now = at(2026, 10, 29, 13);
        let gate = gate_at(&snap, now, false);
        assert!(matches!(
            gate.provider_key_verdict("pk-out"),
            Verdict::Exceeded(_)
        ));
        assert!(matches!(gate.provider_key_verdict("pk-ok"), Verdict::Allow));
        assert!(matches!(
            gate.provider_key_verdict("pk-none"),
            Verdict::Allow
        ));
        // A provider_key budget never applies at key level.
        let k = key(&snap, "k-1", serde_json::json!({}));
        assert!(matches!(
            gate_at(&snap, now, true).key_verdict(&k),
            Verdict::Allow
        ));
        let strict = gate_at(&snap, now, true);
        assert!(matches!(
            strict.provider_key_verdict("pk-ok"),
            Verdict::Unverified
        ));
        assert!(matches!(
            strict.provider_key_verdict("pk-none"),
            Verdict::Allow
        ));
    }
}
