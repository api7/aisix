//! `RateLimitPolicy` entity — standalone rate-limit rules stored in etcd
//! under `rate_limit_policies/<uuid>`.
//!
//! Each policy targets a single subject via `(scope, scope_ref)`:
//! - `api_key`     — matches by API key entry ID
//! - `model`       — matches by model entry ID
//! - `team`        — matches by team ID on the API key (one shared bucket)
//! - `member`      — matches by user ID on the API key
//! - `team_member` — matches by team ID, but buckets per member: every
//!   key in the team inherits this default with its own independent
//!   counter keyed on the API key's user ID
//!
//! The proxy iterates all policies on each request, converts the
//! `window`+`max_requests`/`max_tokens` into a `RateLimit`, and
//! reserves under `policy:<scope>:<scope_ref>:<policy_id>` — with the
//! member's `user_id` appended for the `team_member` scope.

use chrono::{DateTime, Datelike, NaiveDate, Timelike, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::resource::Resource;

/// Subject a [`RateLimitPolicy`] targets, paired with `scope_ref`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PolicyScope {
    ApiKey,
    Model,
    Team,
    Member,
    TeamMember,
}

impl PolicyScope {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ApiKey => "api_key",
            Self::Model => "model",
            Self::Team => "team",
            Self::Member => "member",
            Self::TeamMember => "team_member",
        }
    }
}

impl std::fmt::Display for PolicyScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Fixed-window length a [`RateLimitPolicy`] applies its limits over.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PolicyWindow {
    Second,
    Minute,
    Hour,
}

impl PolicyWindow {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Second => "second",
            Self::Minute => "minute",
            Self::Hour => "hour",
        }
    }
}

impl std::fmt::Display for PolicyWindow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Day-of-week selector for a [`PolicySchedule`], in the schedule's
/// timezone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum ScheduleWeekday {
    Mon,
    Tue,
    Wed,
    Thu,
    Fri,
    Sat,
    Sun,
}

impl ScheduleWeekday {
    fn as_chrono(self) -> chrono::Weekday {
        match self {
            Self::Mon => chrono::Weekday::Mon,
            Self::Tue => chrono::Weekday::Tue,
            Self::Wed => chrono::Weekday::Wed,
            Self::Thu => chrono::Weekday::Thu,
            Self::Fri => chrono::Weekday::Fri,
            Self::Sat => chrono::Weekday::Sat,
            Self::Sun => chrono::Weekday::Sun,
        }
    }
}

/// One recurring wall-clock window during which the owning policy is
/// suspended (not enforced). Days are selected by `days_of_week` OR by
/// an explicit `dates` list (exactly one selector; the JSON Schema's
/// injected `oneOf` enforces this — see
/// [`crate::models::schema::rate_limit_policy_root_schema`]), evaluated
/// in `timezone`. Time bounds compare as wall-clock minutes:
/// `start_time < end_time` is a same-day window; `start_time >
/// end_time` crosses midnight and belongs to its **start** day
/// (`days_of_week: [fri], 22:00 → 09:00` covers Friday 22:00 through
/// Saturday 09:00); equal times are an empty window that never
/// matches.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct PolicySchedule {
    /// IANA timezone the window's wall-clock fields are interpreted in
    /// (e.g. `Asia/Shanghai`).
    #[schemars(length(min = 1))]
    pub timezone: String,
    /// Recurring weekly selector: the window opens on each listed day.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[schemars(length(min = 1))]
    pub days_of_week: Vec<ScheduleWeekday>,
    /// Explicit calendar dates (`YYYY-MM-DD`, in `timezone`) the window
    /// opens on — for holidays and other irregular days.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[schemars(length(min = 1), inner(regex(pattern = r"^\d{4}-\d{2}-\d{2}$")))]
    pub dates: Vec<String>,
    /// Window start, `HH:MM` wall clock (inclusive).
    #[schemars(regex(pattern = r"^([01]\d|2[0-3]):[0-5]\d$"))]
    pub start_time: String,
    /// Window end, `HH:MM` wall clock (exclusive); `24:00` = end of
    /// day. An end before the start crosses into the following day; an
    /// end equal to the start is an empty window that never matches.
    #[schemars(regex(pattern = r"^([01]\d|2[0-3]):[0-5]\d$|^24:00$"))]
    pub end_time: String,
}

/// Parse `HH:MM` to minutes since midnight. `24:00` (end-of-day, only
/// meaningful as an exclusive end bound) is admitted when `allow_2400`.
fn parse_hhmm(s: &str, allow_2400: bool) -> Option<u32> {
    if allow_2400 && s == "24:00" {
        return Some(24 * 60);
    }
    let (h, m) = s.split_once(':')?;
    if h.len() != 2 || m.len() != 2 {
        return None;
    }
    let h: u32 = h.parse().ok()?;
    let m: u32 = m.parse().ok()?;
    if h > 23 || m > 59 {
        return None;
    }
    Some(h * 60 + m)
}

impl PolicySchedule {
    /// Whether the instant `now` falls inside this window. Evaluation
    /// converts UTC → the schedule's local wall clock, which is total
    /// (no DST ambiguity — that only exists in the local→UTC
    /// direction). Any unparseable field makes the window non-matching,
    /// so a malformed schedule fails toward *enforcing* the policy.
    fn matches(&self, now: DateTime<Utc>) -> bool {
        let Ok(tz) = self.timezone.parse::<chrono_tz::Tz>() else {
            return false;
        };
        let Some(start) = parse_hhmm(&self.start_time, false) else {
            return false;
        };
        let Some(end) = parse_hhmm(&self.end_time, true) else {
            return false;
        };
        let local = now.with_timezone(&tz);
        let minute = local.time().hour() * 60 + local.time().minute();
        let today = local.date_naive();
        if start < end {
            // Same-day window [start, end).
            self.selects_day(today) && minute >= start && minute < end
        } else if start > end {
            // Cross-midnight window owned by the start day:
            // [start, 24:00) on a selected day D, plus [00:00, end) on D+1.
            (self.selects_day(today) && minute >= start)
                || (today
                    .pred_opt()
                    .is_some_and(|yesterday| self.selects_day(yesterday))
                    && minute < end)
        } else {
            // start == end: an empty window (cp-api rejects the shape;
            // the DP's own write surface admits it and it matches
            // nothing).
            false
        }
    }

    /// Whether `date` (local to the schedule's timezone) is selected.
    /// `days_of_week` takes precedence if a row ever carries both
    /// selectors (the schema forbids that shape).
    fn selects_day(&self, date: NaiveDate) -> bool {
        if !self.days_of_week.is_empty() {
            return self
                .days_of_week
                .iter()
                .any(|d| d.as_chrono() == date.weekday());
        }
        self.dates
            .iter()
            .any(|s| NaiveDate::parse_from_str(s, "%Y-%m-%d") == Ok(date))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RateLimitPolicy {
    #[schemars(length(min = 1))]
    pub name: String,
    pub scope: PolicyScope,
    #[schemars(length(min = 1))]
    pub scope_ref: String,
    pub window: PolicyWindow,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub max_requests: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1))]
    pub max_tokens: Option<u64>,
    /// Recurring windows during which this policy is suspended — the
    /// quota gate skips it while `now` falls in any listed window and
    /// enforcement resumes automatically afterwards (AISIX-Cloud#1104).
    /// The counters' bucket key never changes, so suspension does not
    /// reset the surrounding window's counts. Empty/absent = always
    /// enforced. cp-api omits the field when empty, keeping
    /// schedule-less rows parseable by pre-`schedules` strict data
    /// planes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub schedules: Vec<PolicySchedule>,

    #[serde(skip)]
    pub(crate) runtime_id: String,
}

impl RateLimitPolicy {
    /// Whether the policy is suspended at `now` — true when any
    /// schedule window contains the instant (windows are a union).
    pub fn suspended_at(&self, now: DateTime<Utc>) -> bool {
        self.schedules.iter().any(|s| s.matches(now))
    }
}

/// The one cross-field invariant `schemars` can't derive: a policy must cap at
/// least one of `max_requests` / `max_tokens`. Injected as a top-level `anyOf`
/// by [`crate::models::schema::rate_limit_policy_root_schema`].
pub fn rate_limit_policy_any_of() -> Value {
    json!([
        { "required": ["max_requests"] },
        { "required": ["max_tokens"] }
    ])
}

/// The [`PolicySchedule`] day-selector XOR `schemars` can't derive: exactly
/// one of `days_of_week` / `dates` must be present. Injected into the
/// `PolicySchedule` definition by
/// [`crate::models::schema::rate_limit_policy_root_schema`].
pub fn policy_schedule_one_of() -> Value {
    json!([
        { "required": ["days_of_week"] },
        { "required": ["dates"] }
    ])
}

impl Resource for RateLimitPolicy {
    fn id(&self) -> &str {
        &self.runtime_id
    }

    #[allow(clippy::misnamed_getters)]
    fn name(&self) -> &str {
        &self.scope_ref
    }

    fn kind() -> &'static str {
        "rate_limit_policies"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialises_with_all_fields() {
        let p: RateLimitPolicy = serde_json::from_str(
            r#"{
              "name": "team-quota",
              "scope": "team",
              "scope_ref": "team-uuid-1",
              "window": "minute",
              "max_requests": 100,
              "max_tokens": 50000
            }"#,
        )
        .unwrap();
        assert_eq!(p.name, "team-quota");
        assert_eq!(p.scope, PolicyScope::Team);
        assert_eq!(p.scope_ref, "team-uuid-1");
        assert_eq!(p.window, PolicyWindow::Minute);
        assert_eq!(p.max_requests, Some(100));
        assert_eq!(p.max_tokens, Some(50000));
    }

    #[test]
    fn deserialises_with_only_max_requests() {
        let p: RateLimitPolicy = serde_json::from_str(
            r#"{
              "name": "key-rpm",
              "scope": "api_key",
              "scope_ref": "key-uuid-1",
              "window": "minute",
              "max_requests": 60
            }"#,
        )
        .unwrap();
        assert_eq!(p.max_requests, Some(60));
        assert!(p.max_tokens.is_none());
    }

    #[test]
    fn tolerates_unknown_fields_for_forward_compat() {
        // cp-api may ship new fields ahead of the DP rolling out; serde must
        // accept them. The write path still rejects them via the strict
        // schema validator (validate_rate_limit_policy in models/schema.rs).
        let p: RateLimitPolicy = serde_json::from_str(
            r#"{
              "name": "x",
              "scope": "team",
              "scope_ref": "t1",
              "window": "minute",
              "extra": true
            }"#,
        )
        .unwrap();
        assert_eq!(p.name, "x");
    }

    #[test]
    fn resource_trait_returns_correct_kind() {
        assert_eq!(RateLimitPolicy::kind(), "rate_limit_policies");
    }

    // --- schedules (AISIX-Cloud#1104) --------------------------------

    /// Parse an RFC3339 instant for schedule sweeps.
    fn at(s: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(s).unwrap().with_timezone(&Utc)
    }

    fn policy_with_schedules(schedules: Value) -> RateLimitPolicy {
        serde_json::from_value(json!({
            "name": "sched",
            "scope": "team",
            "scope_ref": "t1",
            "window": "minute",
            "max_requests": 1,
            "schedules": schedules
        }))
        .unwrap()
    }

    #[test]
    fn no_schedules_never_suspends_and_old_rows_deserialize() {
        // Pre-`schedules` etcd rows carry no field — they must keep
        // enforcing unchanged after upgrade.
        let p: RateLimitPolicy = serde_json::from_str(
            r#"{"name":"x","scope":"team","scope_ref":"t1","window":"minute","max_requests":1}"#,
        )
        .unwrap();
        assert!(p.schedules.is_empty());
        assert!(!p.suspended_at(at("2026-08-08T02:00:00Z")));
    }

    #[test]
    fn weekend_full_day_window_matches_in_its_timezone() {
        // 2026-08-08 is a Saturday. Beijing Sat 10:00 = 02:00Z.
        let p = policy_with_schedules(json!([{
            "timezone": "Asia/Shanghai",
            "days_of_week": ["sat", "sun"],
            "start_time": "00:00",
            "end_time": "24:00"
        }]));
        assert!(p.suspended_at(at("2026-08-08T02:00:00Z")), "Sat 10:00 CST");
        // 23:59 local still inside a 24:00-bounded day.
        assert!(p.suspended_at(at("2026-08-08T15:59:00Z")), "Sat 23:59 CST");
        assert!(!p.suspended_at(at("2026-08-10T02:00:00Z")), "Mon 10:00 CST");
    }

    #[test]
    fn same_instant_differs_by_timezone() {
        // 2026-08-08T02:00Z is Saturday 10:00 in Beijing but still
        // Friday 22:00 in New York — the selector must follow the
        // schedule's timezone, not UTC.
        let entry = |tz: &str| {
            policy_with_schedules(json!([{
                "timezone": tz,
                "days_of_week": ["sat"],
                "start_time": "00:00",
                "end_time": "24:00"
            }]))
        };
        assert!(entry("Asia/Shanghai").suspended_at(at("2026-08-08T02:00:00Z")));
        assert!(!entry("America/New_York").suspended_at(at("2026-08-08T02:00:00Z")));
    }

    #[test]
    fn cross_midnight_window_belongs_to_its_start_day() {
        // Workday off-peak: Mon–Fri 22:00 → next morning 09:00 (CST).
        let p = policy_with_schedules(json!([{
            "timezone": "Asia/Shanghai",
            "days_of_week": ["mon", "tue", "wed", "thu", "fri"],
            "start_time": "22:00",
            "end_time": "09:00"
        }]));
        // 2026-08-07 is a Friday (CST = UTC+8).
        assert!(!p.suspended_at(at("2026-08-07T13:59:00Z")), "Fri 21:59");
        assert!(
            p.suspended_at(at("2026-08-07T14:00:00Z")),
            "Fri 22:00 opens"
        );
        assert!(p.suspended_at(at("2026-08-07T15:00:00Z")), "Fri 23:00");
        // Saturday morning is the spill-over of Friday's window...
        assert!(p.suspended_at(at("2026-08-08T00:59:00Z")), "Sat 08:59");
        assert!(
            !p.suspended_at(at("2026-08-08T01:00:00Z")),
            "Sat 09:00 closes"
        );
        // ...but Saturday evening opens nothing (sat not selected).
        assert!(!p.suspended_at(at("2026-08-08T15:00:00Z")), "Sat 23:00");
        // And Monday 08:00 stays enforced: its night belongs to Sunday,
        // which is not selected. (Cover Sunday nights by adding "sun".)
        assert!(!p.suspended_at(at("2026-08-10T00:00:00Z")), "Mon 08:00");
        assert!(p.suspended_at(at("2026-08-10T14:30:00Z")), "Mon 22:30");
        assert!(p.suspended_at(at("2026-08-11T00:30:00Z")), "Tue 08:30");
        assert!(
            !p.suspended_at(at("2026-08-11T01:00:00Z")),
            "Tue 09:00 closes"
        );
    }

    #[test]
    fn dates_selector_matches_explicit_holidays() {
        let p = policy_with_schedules(json!([{
            "timezone": "Asia/Shanghai",
            "dates": ["2026-10-01", "2026-10-02"],
            "start_time": "00:00",
            "end_time": "24:00"
        }]));
        assert!(p.suspended_at(at("2026-10-01T04:00:00Z")), "Oct 1 noon CST");
        assert!(p.suspended_at(at("2026-10-02T04:00:00Z")), "Oct 2 noon CST");
        assert!(
            !p.suspended_at(at("2026-10-03T04:00:00Z")),
            "Oct 3 noon CST"
        );
        // CST midnight boundary: Sep 30 16:00Z is Oct 1 00:00 CST.
        assert!(
            p.suspended_at(at("2026-09-30T16:00:00Z")),
            "Oct 1 00:00 CST"
        );
        assert!(
            !p.suspended_at(at("2026-09-30T15:59:00Z")),
            "Sep 30 23:59 CST"
        );
    }

    #[test]
    fn dated_cross_midnight_spills_into_the_next_day() {
        let p = policy_with_schedules(json!([{
            "timezone": "Asia/Shanghai",
            "dates": ["2026-10-01"],
            "start_time": "22:00",
            "end_time": "02:00"
        }]));
        assert!(
            p.suspended_at(at("2026-10-01T15:30:00Z")),
            "Oct 1 23:30 CST"
        );
        assert!(
            p.suspended_at(at("2026-10-01T17:00:00Z")),
            "Oct 2 01:00 CST"
        );
        assert!(
            !p.suspended_at(at("2026-10-01T18:00:00Z")),
            "Oct 2 02:00 CST"
        );
        assert!(
            !p.suspended_at(at("2026-10-02T15:30:00Z")),
            "Oct 2 23:30 CST"
        );
    }

    #[test]
    fn schedule_union_any_match_suspends() {
        let p = policy_with_schedules(json!([
            {
                "timezone": "Asia/Shanghai",
                "days_of_week": ["sat", "sun"],
                "start_time": "00:00",
                "end_time": "24:00"
            },
            {
                "timezone": "Asia/Shanghai",
                "days_of_week": ["mon", "tue", "wed", "thu", "fri", "sun"],
                "start_time": "22:00",
                "end_time": "09:00"
            }
        ]));
        // Monday 08:00 CST — covered by the second entry via Sunday.
        assert!(p.suspended_at(at("2026-08-10T00:00:00Z")));
        // Saturday noon — covered by the first entry only.
        assert!(p.suspended_at(at("2026-08-08T04:00:00Z")));
        // Monday noon — covered by neither.
        assert!(!p.suspended_at(at("2026-08-10T04:00:00Z")));
    }

    #[test]
    fn malformed_schedule_fields_fail_toward_enforcing() {
        // Unknown timezone / bad times / start==end: the window simply
        // never matches, so the policy keeps enforcing (conservative).
        for sched in [
            json!([{"timezone": "Not/AZone", "days_of_week": ["sat"],
                    "start_time": "00:00", "end_time": "24:00"}]),
            json!([{"timezone": "Asia/Shanghai", "days_of_week": ["sat"],
                    "start_time": "9:00", "end_time": "10:00"}]),
            json!([{"timezone": "Asia/Shanghai", "days_of_week": ["sat"],
                    "start_time": "09:00", "end_time": "09:00"}]),
        ] {
            let p = policy_with_schedules(sched);
            assert!(!p.suspended_at(at("2026-08-08T02:00:00Z")));
        }
    }

    #[test]
    fn empty_schedules_are_omitted_from_serialization() {
        // cp-api relies on this shape: schedule-less rows stay
        // byte-compatible with pre-`schedules` data planes.
        let p: RateLimitPolicy = serde_json::from_str(
            r#"{"name":"x","scope":"team","scope_ref":"t1","window":"minute","max_requests":1}"#,
        )
        .unwrap();
        let out = serde_json::to_value(&p).unwrap();
        assert!(out.get("schedules").is_none());

        let with = policy_with_schedules(json!([{
            "timezone": "Asia/Shanghai",
            "days_of_week": ["sat"],
            "start_time": "00:00",
            "end_time": "24:00"
        }]));
        let out = serde_json::to_value(&with).unwrap();
        assert_eq!(out["schedules"][0]["timezone"], "Asia/Shanghai");
    }

    #[test]
    fn resource_name_returns_scope_ref() {
        let mut p: RateLimitPolicy = serde_json::from_str(
            r#"{
              "name": "test",
              "scope": "member",
              "scope_ref": "member-uuid-1",
              "window": "hour",
              "max_tokens": 1000000
            }"#,
        )
        .unwrap();
        p.runtime_id = "policy-1".into();
        assert_eq!(p.id(), "policy-1");
        assert_eq!(p.name(), "member-uuid-1");
    }
}
