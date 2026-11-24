//! `Budget` and `BudgetPolicy` — the spend caps the control plane enforces
//! through the gateway.
//!
//! The control plane owns the spend counters. It writes one `budgets`
//! document per hard-stop budget that applies in the environment, and
//! marks the document `exhausted` once spend reaches the limit; the gateway
//! decides every request locally from those documents and never asks the
//! control plane per request. `budget_policy` is the environment's
//! singleton that says what the gateway does while it cannot deliver usage
//! to the control plane — the only input those counters have — and which
//! calendar timezone defines budget period boundaries.

use std::collections::BTreeMap;

use chrono::{DateTime, Datelike, LocalResult, NaiveDate, NaiveTime, TimeZone, Utc};
use chrono_tz::Tz;
use serde::{Deserialize, Serialize};

use crate::resource::Resource;

/// A hard-stop spend cap that applies in this environment, and whether it
/// is currently exhausted.
///
/// The document's id is the budget's id. Only hard-stop budgets are
/// present: a budget that only warns never blocks a request, so it has no
/// document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Budget {
    /// What the budget caps.
    ///
    /// - `org`, `environment`: every API key in this environment.
    /// - `api_key`: the API key whose id is `scope_ref`.
    /// - `member`: every API key whose `user_id` is `scope_ref`.
    /// - `team`: every API key whose `team_id` is `scope_ref`.
    /// - `team_member`: every API key whose `team_id` is `scope_ref` and that
    ///   also names a member; each member is capped separately.
    /// - `tag`: every request whose effective value for `tag.key` is
    ///   `tag.value`. That value is what the identity provider's `tag_claims`
    ///   reads from the token, else the API key's, else its team's, else its
    ///   member's, so requests on one API key can differ.
    /// - `provider_key`: every upstream call made with the provider key whose
    ///   id is `scope_ref`.
    pub scope: BudgetScope,

    /// Id of the capped resource. Present for every scope except `tag`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope_ref: Option<String>,

    /// The capped tag. Present only for scope `tag`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tag: Option<BudgetTag>,

    /// Accounting period. Calendar boundaries follow `budget_policy.timezone`
    /// when configured, or UTC when omitted: a day starts at 00:00, a week on
    /// Monday at 00:00, and a month on the 1st at 00:00.
    pub period: BudgetPeriod,

    /// Set while the budget is exhausted for everything it applies to.
    /// For scope `team_member` it applies to every member of the team.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exhausted: Option<BudgetExhausted>,

    /// Members of the team who have exhausted their own share of a
    /// `team_member` budget, keyed by member id. Read only for scope
    /// `team_member`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub exhausted_members: BTreeMap<String, BudgetMemberExhausted>,

    /// Set by the loader from the key's id segment. Not part of the wire
    /// shape.
    #[serde(skip)]
    pub(crate) runtime_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum BudgetScope {
    Org,
    Environment,
    ApiKey,
    Member,
    Team,
    TeamMember,
    Tag,
    ProviderKey,
}

impl BudgetScope {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Org => "org",
            Self::Environment => "environment",
            Self::ApiKey => "api_key",
            Self::Member => "member",
            Self::Team => "team",
            Self::TeamMember => "team_member",
            Self::Tag => "tag",
            Self::ProviderKey => "provider_key",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct BudgetTag {
    /// Tag key.
    pub key: String,
    /// Tag value.
    pub value: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum BudgetPeriod {
    Day,
    Week,
    Month,
}

impl BudgetPeriod {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Day => "day",
            Self::Week => "week",
            Self::Month => "month",
        }
    }
}

/// The budget is exhausted for everything it applies to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct BudgetExhausted {
    /// Start of the period the budget was exhausted in, as an RFC 3339
    /// timestamp. The block holds only while this is the start of the
    /// current period, so it lifts at the next period boundary on its own.
    /// Omitted, the block holds in every period.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub period_start: Option<String>,
    /// What a rejected request is told.
    pub reason: BudgetReason,
}

/// One member's share of a `team_member` budget is exhausted.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct BudgetMemberExhausted {
    /// Start of the period the member exhausted their share in, as an
    /// RFC 3339 timestamp. The block holds only while this is the start of
    /// the current period.
    pub period_start: String,
    /// What a rejected request is told.
    pub reason: BudgetReason,
}

/// The detail a request rejected by an exhausted budget is answered with.
/// Amounts are decimal strings in the organization's currency.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct BudgetReason {
    /// Error type. The response always reports `billing_error`.
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    /// Error code. The response always reports `budget_exceeded`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    /// Human-readable message returned to the caller.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// Scope of the exhausted budget.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    /// Id of the capped resource, or the tag for scope `tag`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope_ref: Option<String>,
    /// The budget's limit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<String>,
    /// Spend in the period when the budget was exhausted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spent: Option<String>,
    /// Limit minus spend; zero or negative.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remaining: Option<String>,
    /// Accounting period.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub period: Option<String>,
    /// Ignored: the gateway computes when the period resets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub period_resets_at: Option<serde_json::Value>,
    /// Ignored: the gateway computes how long until the period resets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_after_seconds: Option<serde_json::Value>,
}

impl Resource for Budget {
    fn id(&self) -> &str {
        &self.runtime_id
    }

    /// Budgets are referenced by id only.
    fn name(&self) -> &str {
        "budget"
    }

    fn kind() -> &'static str {
        "budgets"
    }
}

/// Environment-wide budget policy: what the gateway does with budgeted traffic
/// while it cannot deliver usage to the control plane (whose spend counters
/// mark a budget exhausted), and which calendar timezone defines budget period
/// boundaries.
///
/// At most one document exists per environment, keyed by the environment
/// id. Without it the gateway behaves as `sticky` and uses UTC period
/// boundaries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct BudgetPolicy {
    /// `sticky` keeps enforcing the last exhausted state the gateway
    /// received and lets everything else through. `closed` additionally
    /// rejects budgeted traffic once usage delivery has failed for
    /// `closed_grace_seconds`: a request from an API key that at least one
    /// budget applies to is answered `503 budget_unverified`, and an
    /// upstream whose provider key has a budget is not used.
    pub fail_mode: BudgetFailMode,

    /// How long usage delivery may fail before `closed` starts rejecting
    /// budgeted traffic, in seconds.
    #[schemars(range(min = 1))]
    pub closed_grace_seconds: u64,

    /// IANA timezone used for calendar budget period boundaries (`day`,
    /// `week`, `month`). Absent means UTC.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,

    /// Set by the loader from the key's id segment. Not part of the wire
    /// shape.
    #[serde(skip)]
    pub(crate) runtime_id: String,
}

/// Parse an IANA timezone name for budget period calculations.
/// Rejects empty strings, `"Local"`, whitespace-padded values, and unknown names.
pub fn parse_budget_timezone(tz: &str) -> Result<Tz, String> {
    if tz.is_empty() || tz == "Local" {
        return Err(format!(
            "budget_policy timezone {tz:?} must be a valid IANA timezone name"
        ));
    }
    tz.parse::<Tz>()
        .map_err(|_| format!("budget_policy timezone {tz:?} must be a valid IANA timezone name"))
}

impl BudgetPolicy {
    /// Rejects a `timezone` that is not a valid IANA timezone name (including
    /// empty strings and `"Local"`). Absent `timezone` means UTC.
    pub fn validate_semantics(&self) -> Result<(), String> {
        if let Some(ref tz) = self.timezone {
            parse_budget_timezone(tz)?;
        }
        Ok(())
    }
}

/// First valid local instant on `date` in `tz`.
///
/// Handles midnight DST transitions consistently with the control plane:
/// - Normal midnight (`Single`) returns `00:00:00`.
/// - Fall-back repeated midnight (`Ambiguous`) returns the earliest occurrence.
/// - Spring-forward skipped midnight (`None`, e.g. `America/Havana` 2024-03-10,
///   `Asia/Beirut` 2024-03-31) probes forward by 1 minute up to 24 hours to
///   find the first representable instant on that date.
fn local_day_start(date: NaiveDate, tz: Tz) -> DateTime<Tz> {
    let midnight = date.and_time(NaiveTime::MIN);
    match tz.from_local_datetime(&midnight) {
        LocalResult::Single(dt) | LocalResult::Ambiguous(dt, _) => dt,
        LocalResult::None => {
            for mins in 1..=(24 * 60) {
                let candidate = midnight + chrono::Duration::minutes(mins);
                match tz.from_local_datetime(&candidate) {
                    LocalResult::Single(dt) | LocalResult::Ambiguous(dt, _) => return dt,
                    LocalResult::None => continue,
                }
            }
            midnight.and_utc().with_timezone(&tz)
        }
    }
}

fn budget_period_dates(now: DateTime<Utc>, period: BudgetPeriod, tz: Tz) -> (NaiveDate, NaiveDate) {
    let day = now.with_timezone(&tz).date_naive();
    match period {
        BudgetPeriod::Day => (day, day + chrono::Days::new(1)),
        BudgetPeriod::Week => {
            let start = day - chrono::Days::new(u64::from(day.weekday().num_days_from_monday()));
            (start, start + chrono::Days::new(7))
        }
        BudgetPeriod::Month => {
            let start = day.with_day(1).expect("the 1st exists in every month");
            (start, start + chrono::Months::new(1))
        }
    }
}

/// Start of the calendar budget period containing `now` in `tz`.
pub fn budget_period_start(now: DateTime<Utc>, period: BudgetPeriod, tz: Tz) -> DateTime<Tz> {
    let (start_date, _) = budget_period_dates(now, period, tz);
    local_day_start(start_date, tz)
}

/// Start of the calendar budget period immediately following the one containing
/// `now` in `tz`.
pub fn budget_next_period_start(now: DateTime<Utc>, period: BudgetPeriod, tz: Tz) -> DateTime<Tz> {
    let (_, next_date) = budget_period_dates(now, period, tz);
    local_day_start(next_date, tz)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum BudgetFailMode {
    Sticky,
    Closed,
}

impl Resource for BudgetPolicy {
    fn id(&self) -> &str {
        &self.runtime_id
    }

    /// Fixed identity: the row is a per-environment singleton.
    fn name(&self) -> &str {
        "budget_policy"
    }

    fn kind() -> &'static str {
        "budget_policy"
    }
}

#[cfg(test)]
mod tests {
    use chrono::{DateTime, SecondsFormat, TimeZone, Utc};
    use chrono_tz::Tz;
    use serde_json::json;

    use super::{budget_next_period_start, budget_period_start, BudgetPeriod, BudgetPolicy};
    use crate::models::{
        validate_budget, validate_budget_lenient, validate_budget_policy,
        validate_budget_policy_lenient,
    };

    fn utc(y: i32, m: u32, d: u32, h: u32, min: u32, s: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(y, m, d, h, min, s).unwrap()
    }

    fn fmt_rfc3339(dt: DateTime<Tz>) -> String {
        dt.to_rfc3339_opts(SecondsFormat::Secs, true)
    }

    fn fmt_utc(dt: DateTime<Tz>) -> String {
        dt.with_timezone(&Utc)
            .to_rfc3339_opts(SecondsFormat::Secs, true)
    }

    /// The per-scope shape is a write-path rule only: the stored row still
    /// loads, and a row without its target matches nothing.
    #[test]
    fn the_per_scope_shape_binds_the_write_path_only() {
        for doc in [
            json!({"scope": "tag", "period": "day"}),
            json!({"scope": "tag", "scope_ref": "x", "tag": {"key": "k", "value": "v"}, "period": "day"}),
            json!({"scope": "team", "period": "day"}),
            json!({"scope": "team", "scope_ref": "t", "tag": {"key": "k", "value": "v"}, "period": "day"}),
            json!({"scope": "team", "scope_ref": "t", "period": "day", "exhausted_members": {}}),
        ] {
            assert!(validate_budget(&doc).is_err(), "strict accepted {doc}");
            assert!(
                validate_budget_lenient(&doc).is_ok(),
                "lenient rejected {doc}"
            );
        }
        for doc in [
            json!({"scope": "tag", "tag": {"key": "k", "value": "v"}, "period": "week"}),
            json!({"scope": "team_member", "scope_ref": "t", "period": "month",
                   "exhausted_members": {"u": {"period_start": "2026-10-01T00:00:00Z", "reason": {}}}}),
            json!({"scope": "provider_key", "scope_ref": "pk", "period": "day",
                   "exhausted": {"reason": {"message": "m", "limit": "0.00"}}}),
        ] {
            assert!(validate_budget(&doc).is_ok(), "strict rejected {doc}");
        }
    }

    /// Every release's control plane from the support floor up may have left
    /// the old amount keys in a reason, beside the current ones or alone:
    /// either row loads, and the old keys are never read.
    #[test]
    fn a_reason_with_the_old_amount_keys_loads() {
        let old = json!({"limit_usd": "9", "spent_usd": "9.5", "remaining_usd": "-0.5"});
        let both = json!({"limit_usd": "9", "spent_usd": "9.5", "remaining_usd": "-0.5",
                          "limit": "10", "spent": "10.5", "remaining": "-0.5"});
        for (reason, limit) in [(both, Some("10")), (old, None)] {
            let doc = json!({"scope": "api_key", "scope_ref": "k", "period": "month",
                             "exhausted": {"reason": reason}});
            assert!(
                validate_budget_lenient(&doc).is_ok(),
                "lenient rejected {doc}"
            );
            let budget: super::Budget = serde_json::from_value(doc).unwrap();
            let reason = budget.exhausted.unwrap().reason;
            assert_eq!(reason.limit.as_deref(), limit);
        }
    }

    #[test]
    fn the_policy_grace_is_a_positive_integer() {
        assert!(
            validate_budget_policy(&json!({"fail_mode": "closed", "closed_grace_seconds": 1}))
                .is_ok()
        );
        assert!(
            validate_budget_policy(&json!({"fail_mode": "closed", "closed_grace_seconds": 0}))
                .is_err()
        );
        assert!(
            validate_budget_policy(&json!({"fail_mode": "open", "closed_grace_seconds": 60}))
                .is_err()
        );
    }

    #[test]
    fn budget_policy_timezone_is_optional_in_strict_and_lenient_schemas() {
        let without = json!({"fail_mode": "sticky", "closed_grace_seconds": 300});
        let with_tz = json!({
            "fail_mode": "sticky",
            "closed_grace_seconds": 300,
            "timezone": "Asia/Shanghai"
        });
        let wrong_type = json!({
            "fail_mode": "sticky",
            "closed_grace_seconds": 300,
            "timezone": 8
        });

        assert!(validate_budget_policy(&without).is_ok());
        assert!(validate_budget_policy_lenient(&without).is_ok());
        assert!(validate_budget_policy(&with_tz).is_ok());
        assert!(validate_budget_policy_lenient(&with_tz).is_ok());
        assert!(validate_budget_policy(&wrong_type).is_err());
        assert!(validate_budget_policy_lenient(&wrong_type).is_err());

        // Serialization omits `timezone` when None and preserves it when Some.
        let p_none: BudgetPolicy = serde_json::from_value(without).unwrap();
        assert_eq!(p_none.timezone, None);
        assert!(serde_json::to_value(&p_none)
            .unwrap()
            .get("timezone")
            .is_none());

        let p_some: BudgetPolicy = serde_json::from_value(with_tz).unwrap();
        assert_eq!(p_some.timezone.as_deref(), Some("Asia/Shanghai"));
        assert_eq!(
            serde_json::to_value(&p_some).unwrap()["timezone"],
            json!("Asia/Shanghai")
        );
    }

    #[test]
    fn budget_policy_validate_semantics_checks_iana_timezone() {
        let policy_with = |tz: Option<&str>| -> BudgetPolicy {
            let mut doc = json!({"fail_mode": "sticky", "closed_grace_seconds": 300});
            if let Some(tz) = tz {
                doc["timezone"] = json!(tz);
            }
            serde_json::from_value(doc).unwrap()
        };

        for valid in [
            None,
            Some("UTC"),
            Some("Etc/UTC"),
            Some("Asia/Shanghai"),
            Some("America/New_York"),
            Some("Asia/Kathmandu"),
        ] {
            assert!(
                policy_with(valid).validate_semantics().is_ok(),
                "expected valid timezone: {valid:?}"
            );
        }

        for invalid in [
            "",
            "Local",
            "Mars/Olympus",
            " Asia/Shanghai",
            "Asia/Shanghai ",
            "UTC+8",
        ] {
            let err = policy_with(Some(invalid))
                .validate_semantics()
                .expect_err(&format!("expected invalid timezone: {invalid:?}"));
            assert!(
                err.contains("IANA timezone"),
                "unexpected error for {invalid:?}: {err}"
            );
        }
    }

    #[test]
    fn shanghai_periods_cross_utc_16_00_boundary() {
        let tz = Tz::Asia__Shanghai;

        // Before Beijing midnight: 2026-11-24T15:59:59Z == 2026-11-24T23:59:59+08:00 (Tuesday)
        let before = utc(2026, 11, 24, 15, 59, 59);
        assert_eq!(
            fmt_rfc3339(budget_period_start(before, BudgetPeriod::Day, tz)),
            "2026-11-24T00:00:00+08:00"
        );
        assert_eq!(
            fmt_utc(budget_period_start(before, BudgetPeriod::Day, tz)),
            "2026-11-23T16:00:00Z"
        );
        assert_eq!(
            fmt_rfc3339(budget_next_period_start(before, BudgetPeriod::Day, tz)),
            "2026-11-25T00:00:00+08:00"
        );

        // At Beijing midnight: 2026-11-24T16:00:00Z == 2026-11-25T00:00:00+08:00 (Wednesday)
        let after = utc(2026, 11, 24, 16, 0, 0);
        assert_eq!(
            fmt_rfc3339(budget_period_start(after, BudgetPeriod::Day, tz)),
            "2026-11-25T00:00:00+08:00"
        );
        assert_eq!(
            fmt_utc(budget_period_start(after, BudgetPeriod::Day, tz)),
            "2026-11-24T16:00:00Z"
        );
        assert_eq!(
            fmt_rfc3339(budget_next_period_start(after, BudgetPeriod::Day, tz)),
            "2026-11-26T00:00:00+08:00"
        );

        // Week rollover: Sunday 2026-11-22T15:59:59Z is still Sunday 23:59:59+08 (week of Mon Nov 16);
        // Sunday 2026-11-22T16:00:00Z is Monday 2026-11-23T00:00:00+08 (new week).
        let sun_before = utc(2026, 11, 22, 15, 59, 59);
        assert_eq!(
            fmt_rfc3339(budget_period_start(sun_before, BudgetPeriod::Week, tz)),
            "2026-11-16T00:00:00+08:00"
        );
        assert_eq!(
            fmt_rfc3339(budget_next_period_start(sun_before, BudgetPeriod::Week, tz)),
            "2026-11-23T00:00:00+08:00"
        );
        let mon_start = utc(2026, 11, 22, 16, 0, 0);
        assert_eq!(
            fmt_rfc3339(budget_period_start(mon_start, BudgetPeriod::Week, tz)),
            "2026-11-23T00:00:00+08:00"
        );
        assert_eq!(
            fmt_rfc3339(budget_next_period_start(mon_start, BudgetPeriod::Week, tz)),
            "2026-11-30T00:00:00+08:00"
        );

        // Month rollover: 2026-11-30T15:59:59Z is Nov 30 23:59:59+08;
        // 2026-11-30T16:00:00Z is Dec 1 00:00:00+08.
        let nov_end = utc(2026, 11, 30, 15, 59, 59);
        assert_eq!(
            fmt_rfc3339(budget_period_start(nov_end, BudgetPeriod::Month, tz)),
            "2026-11-01T00:00:00+08:00"
        );
        assert_eq!(
            fmt_rfc3339(budget_next_period_start(nov_end, BudgetPeriod::Month, tz)),
            "2026-12-01T00:00:00+08:00"
        );
        let dec_start = utc(2026, 11, 30, 16, 0, 0);
        assert_eq!(
            fmt_rfc3339(budget_period_start(dec_start, BudgetPeriod::Month, tz)),
            "2026-12-01T00:00:00+08:00"
        );
        assert_eq!(
            fmt_rfc3339(budget_next_period_start(dec_start, BudgetPeriod::Month, tz)),
            "2027-01-01T00:00:00+08:00"
        );

        // Year rollover: 2026-12-31T16:00:00Z is 2027-01-01T00:00:00+08:00.
        let new_year = utc(2026, 12, 31, 16, 0, 0);
        assert_eq!(
            fmt_rfc3339(budget_period_start(new_year, BudgetPeriod::Month, tz)),
            "2027-01-01T00:00:00+08:00"
        );
        assert_eq!(
            fmt_rfc3339(budget_next_period_start(new_year, BudgetPeriod::Month, tz)),
            "2027-02-01T00:00:00+08:00"
        );
    }

    #[test]
    fn non_whole_hour_offset_kathmandu_periods() {
        // UTC+05:45: 18:14:59Z == 23:59:59+05:45 (still 2026-11-24).
        let tz = Tz::Asia__Kathmandu;
        let before = utc(2026, 11, 24, 18, 14, 59);
        assert_eq!(
            fmt_rfc3339(budget_period_start(before, BudgetPeriod::Day, tz)),
            "2026-11-24T00:00:00+05:45"
        );
        assert_eq!(
            fmt_utc(budget_period_start(before, BudgetPeriod::Day, tz)),
            "2026-11-23T18:15:00Z"
        );
        assert_eq!(
            fmt_rfc3339(budget_next_period_start(before, BudgetPeriod::Day, tz)),
            "2026-11-25T00:00:00+05:45"
        );
        // 18:15:00Z == 00:00:00+05:45 (2026-11-25)
        let after = utc(2026, 11, 24, 18, 15, 0);
        assert_eq!(
            fmt_rfc3339(budget_period_start(after, BudgetPeriod::Day, tz)),
            "2026-11-25T00:00:00+05:45"
        );
        assert_eq!(
            fmt_utc(budget_period_start(after, BudgetPeriod::Day, tz)),
            "2026-11-24T18:15:00Z"
        );
    }

    #[test]
    fn dst_midnight_spring_forward_and_fall_back_golden_vectors() {
        // America/Havana 2024-03-10: 00:00-05:00 does not exist; day starts at 01:00-04:00 (05:00Z).
        let havana = Tz::America__Havana;
        let before_havana_jump = utc(2024, 3, 10, 4, 59, 59); // 2024-03-09T23:59:59-05:00
        let at_havana_jump = utc(2024, 3, 10, 5, 0, 0); // 2024-03-10T01:00:00-04:00
        let havana_next = budget_next_period_start(before_havana_jump, BudgetPeriod::Day, havana);
        let havana_start = budget_period_start(at_havana_jump, BudgetPeriod::Day, havana);
        assert_eq!(havana_next, havana_start);
        assert_eq!(fmt_utc(havana_start), "2024-03-10T05:00:00Z");
        assert_eq!(fmt_rfc3339(havana_start), "2024-03-10T01:00:00-04:00");

        // Asia/Beirut 2024-03-31: 00:00+02:00 does not exist; day starts at 01:00+03:00 (2024-03-30T22:00Z).
        let beirut = Tz::Asia__Beirut;
        let before_beirut_jump = utc(2024, 3, 30, 21, 59, 59); // 2024-03-30T23:59:59+02:00
        let at_beirut_jump = utc(2024, 3, 30, 22, 0, 0); // 2024-03-31T01:00:00+03:00
        let beirut_next = budget_next_period_start(before_beirut_jump, BudgetPeriod::Day, beirut);
        let beirut_start = budget_period_start(at_beirut_jump, BudgetPeriod::Day, beirut);
        assert_eq!(beirut_next, beirut_start);
        assert_eq!(fmt_utc(beirut_start), "2024-03-30T22:00:00Z");
        assert_eq!(fmt_rfc3339(beirut_start), "2024-03-31T01:00:00+03:00");

        // America/Havana 2024-11-03: midnight occurs twice (-04:00 then -05:00); earliest wins (04:00Z).
        let before_fallback = utc(2024, 11, 3, 3, 59, 59); // 2024-11-02T23:59:59-04:00
        let first_midnight = utc(2024, 11, 3, 4, 30, 0); // 2024-11-03T00:30:00-04:00
        let second_midnight = utc(2024, 11, 3, 5, 30, 0); // 2024-11-03T00:30:00-05:00
        let fallback_next = budget_next_period_start(before_fallback, BudgetPeriod::Day, havana);
        let fallback_start_1 = budget_period_start(first_midnight, BudgetPeriod::Day, havana);
        let fallback_start_2 = budget_period_start(second_midnight, BudgetPeriod::Day, havana);
        assert_eq!(fallback_next, fallback_start_1);
        assert_eq!(fallback_start_1, fallback_start_2);
        assert_eq!(fmt_utc(fallback_start_1), "2024-11-03T04:00:00Z");
        assert_eq!(fmt_rfc3339(fallback_start_1), "2024-11-03T00:00:00-04:00");
    }

    #[test]
    fn non_midnight_dst_transition_days_reset_at_local_midnight() {
        let ny = Tz::America__New_York;

        // 2024-03-10 is a 23-hour day (02:00 EST -> 03:00 EDT).
        let spring = utc(2024, 3, 10, 15, 0, 0);
        let spring_start = budget_period_start(spring, BudgetPeriod::Day, ny);
        let spring_next = budget_next_period_start(spring, BudgetPeriod::Day, ny);
        assert_eq!(fmt_rfc3339(spring_start), "2024-03-10T00:00:00-05:00");
        assert_eq!(fmt_rfc3339(spring_next), "2024-03-11T00:00:00-04:00");
        assert_eq!(
            (spring_next.with_timezone(&Utc) - spring_start.with_timezone(&Utc)).num_hours(),
            23
        );

        // 2024-11-03 is a 25-hour day (02:00 EDT -> 01:00 EST).
        let fall = utc(2024, 11, 3, 15, 0, 0);
        let fall_start = budget_period_start(fall, BudgetPeriod::Day, ny);
        let fall_next = budget_next_period_start(fall, BudgetPeriod::Day, ny);
        assert_eq!(fmt_rfc3339(fall_start), "2024-11-03T00:00:00-04:00");
        assert_eq!(fmt_rfc3339(fall_next), "2024-11-04T00:00:00-05:00");
        assert_eq!(
            (fall_next.with_timezone(&Utc) - fall_start.with_timezone(&Utc)).num_hours(),
            25
        );
    }
}
