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
}
