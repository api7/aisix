//! Budget client — asks cp-api per request whether an api_key may proceed.
//!
//! Wire: `GET {dpmgr_base}/dp/budget_check?api_key_id=<uuid>`. Auth is
//! mTLS — the caller supplies a `reqwest::Client` already loaded with
//! the same client cert + CA bundle the heartbeat worker uses. cp-api
//! authenticates the DP by peer cert SAN (env_id, dp_id) and rejects
//! requests for api_keys outside that env (403). See prd-09b rev 2 §5.5
//! and AISIX-Cloud PR #95 for the CP-side route.
//!
//! Decisions are cached in an LRU (capacity 10000, TTL 5s) keyed by
//! api_key_id. When cp-api is unreachable we honor the last cached
//! decision (sticky) up to AISIX_DP_BUDGET_STALE_MAX_SECONDS (default
//! 600s); past that we apply the fail_mode that came back on the last
//! good response.

use dashmap::DashMap;
use serde::Deserialize;
use serde_json::Value;
use std::time::{Duration, Instant};

const CACHE_TTL: Duration = Duration::from_secs(5);
const CACHE_CAPACITY: usize = 10_000;
const DEFAULT_STALE_MAX_SECONDS: u64 = 600;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailMode {
    Sticky,
    Open,
    Closed,
}

impl FailMode {
    fn parse(s: &str) -> Self {
        match s {
            "open" => FailMode::Open,
            "closed" => FailMode::Closed,
            _ => FailMode::Sticky,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Decision {
    pub allowed: bool,
    pub fail_mode: FailMode,
    pub reason: Option<String>,
    pub budget: Option<BudgetDetails>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct BudgetDetails {
    pub limit_usd: Option<f64>,
    pub spent_usd: Option<f64>,
    pub remaining_usd: Option<f64>,
    pub reset_seconds: Option<u64>,
}

impl Decision {
    fn allow_all() -> Self {
        Self {
            allowed: true,
            fail_mode: FailMode::Open,
            reason: None,
            budget: None,
        }
    }
}

#[derive(Debug, Clone)]
struct CacheEntry {
    decision: Decision,
    fetched_at: Instant,
}

/// Mode for the client: live (talks to cp-api) or disabled (allow-all).
enum Mode {
    Live {
        http: reqwest::Client,
        base_url: String,
        stale_max: Duration,
    },
    Disabled,
}

pub struct BudgetClient {
    mode: Mode,
    cache: DashMap<String, CacheEntry>,
}

impl std::fmt::Debug for BudgetClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mode = match &self.mode {
            Mode::Live { base_url, .. } => format!("live({base_url})"),
            Mode::Disabled => "disabled".into(),
        };
        f.debug_struct("BudgetClient")
            .field("mode", &mode)
            .field("cached", &self.cache.len())
            .finish()
    }
}

impl BudgetClient {
    /// Live client that asks cp-api per request via mTLS. `base_url` is
    /// the same dpmgr origin the heartbeat worker hits (e.g.
    /// `https://cp.aisix.cloud:9101`); `http` must be a reqwest client
    /// already loaded with the DP's client cert + CA bundle. Build it
    /// with `aisix_server::heartbeat::build_mtls_client` (or its
    /// equivalent) using the same persisted `MtlsBundle`.
    pub fn new(base_url: impl Into<String>, http: reqwest::Client) -> Self {
        let base_url = base_url.into().trim_end_matches('/').to_string();
        let stale_max = std::env::var("AISIX_DP_BUDGET_STALE_MAX_SECONDS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(DEFAULT_STALE_MAX_SECONDS);
        Self {
            mode: Mode::Live {
                http,
                base_url,
                stale_max: Duration::from_secs(stale_max),
            },
            cache: DashMap::new(),
        }
    }

    /// Allow-all client. Used in dev / tests / standalone mode where no
    /// cp-api is reachable.
    pub fn disabled() -> Self {
        Self {
            mode: Mode::Disabled,
            cache: DashMap::new(),
        }
    }

    /// Check whether `api_key_id` may proceed. Returns a `Decision`; the
    /// caller maps `!allowed` to `ProxyError::BudgetExceeded`.
    pub async fn check(&self, api_key_id: &str) -> Decision {
        match &self.mode {
            Mode::Disabled => Decision::allow_all(),
            Mode::Live {
                http,
                base_url,
                stale_max,
            } => {
                // Fast path: cache hit within TTL.
                if let Some(cached) = self.cache.get(api_key_id) {
                    if cached.fetched_at.elapsed() < CACHE_TTL {
                        return cached.decision.clone();
                    }
                }

                match fetch_decision(http, base_url, api_key_id).await {
                    Ok(decision) => {
                        self.insert(api_key_id, decision.clone());
                        decision
                    }
                    Err(err) => {
                        tracing::warn!(
                            api_key_id = %api_key_id,
                            error = %err,
                            "budget_check failed; falling back to cache or fail_mode",
                        );
                        self.fallback(api_key_id, *stale_max)
                    }
                }
            }
        }
    }

    fn fallback(&self, api_key_id: &str, stale_max: Duration) -> Decision {
        if let Some(cached) = self.cache.get(api_key_id) {
            if cached.fetched_at.elapsed() < stale_max {
                return cached.decision.clone();
            }
            // Outer staleness ceiling exceeded: apply the fail_mode from
            // the last good response.
            return apply_fail_mode(&cached.decision);
        }
        // No cache at all — sticky-default to deny.
        Decision {
            allowed: false,
            fail_mode: FailMode::Sticky,
            reason: Some("cp-api unreachable and no cached decision".to_string()),
            budget: None,
        }
    }

    fn insert(&self, api_key_id: &str, decision: Decision) {
        if self.cache.len() >= CACHE_CAPACITY {
            self.evict_oldest();
        }
        self.cache.insert(
            api_key_id.to_string(),
            CacheEntry {
                decision,
                fetched_at: Instant::now(),
            },
        );
    }

    fn evict_oldest(&self) {
        let oldest_key = self
            .cache
            .iter()
            .min_by_key(|e| e.value().fetched_at)
            .map(|e| e.key().clone());
        if let Some(k) = oldest_key {
            self.cache.remove(&k);
        }
    }
}

fn apply_fail_mode(prev: &Decision) -> Decision {
    match prev.fail_mode {
        FailMode::Open => Decision {
            allowed: true,
            fail_mode: FailMode::Open,
            reason: None,
            budget: prev.budget.clone(),
        },
        FailMode::Closed => Decision {
            allowed: false,
            fail_mode: FailMode::Closed,
            reason: Some("cp-api unreachable; fail_mode=closed".to_string()),
            budget: prev.budget.clone(),
        },
        FailMode::Sticky => Decision {
            allowed: false,
            fail_mode: FailMode::Sticky,
            reason: Some("cp-api unreachable; cached decision stale".to_string()),
            budget: prev.budget.clone(),
        },
    }
}

// Wire shape mirrors cp-api's `budgetCheckResponse` in
// internal/cpapi/resources/budget_check.go (prd-09b rev 2 §5.5/§5.8):
//
//   {
//     "allow": bool,
//     "fail_mode": "sticky"|"open"|"closed",
//     "reason": {                         // present iff allow == false
//       "type": "billing_error",
//       "code": "budget_exceeded",
//       "message": "...",
//       "scope": "...", "scope_ref": "...",
//       "limit_usd": "...", "spent_usd": "...",
//       "period": "...", "period_resets_at": "...",
//       "retry_after_seconds": <int>
//     }
//   }
//
// We surface only `message` to ProxyError::BudgetExceeded; the other
// fields exist for the dashboard banner once we plumb them through.
#[derive(Debug, Deserialize)]
struct WireDecision {
    allow: bool,
    #[serde(default)]
    fail_mode: String,
    #[serde(default)]
    reason: Option<WireReason>,
    #[serde(default)]
    budget: Option<WireBudget>,
}

#[derive(Debug, Deserialize)]
struct WireReason {
    #[serde(default)]
    message: String,
    #[serde(default)]
    limit_usd: Option<Value>,
    #[serde(default)]
    spent_usd: Option<Value>,
    #[serde(default)]
    remaining_usd: Option<Value>,
    #[serde(default, alias = "period_resets_at")]
    reset_seconds: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct WireBudget {
    #[serde(default, alias = "max_usd")]
    limit_usd: Option<Value>,
    #[serde(default)]
    spent_usd: Option<Value>,
    #[serde(default)]
    remaining_usd: Option<Value>,
    #[serde(default, alias = "period_resets_at")]
    reset_seconds: Option<Value>,
}

async fn fetch_decision(
    http: &reqwest::Client,
    base_url: &str,
    api_key_id: &str,
) -> Result<Decision, reqwest::Error> {
    let url = format!("{base_url}/dp/budget_check");
    let resp = http
        .get(url)
        .query(&[("api_key_id", api_key_id)])
        .send()
        .await?
        .error_for_status()?;
    let wire: WireDecision = resp.json().await?;
    let reason_budget = wire.reason.as_ref().map(|r| BudgetDetails {
        limit_usd: value_as_f64(r.limit_usd.as_ref()),
        spent_usd: value_as_f64(r.spent_usd.as_ref()),
        remaining_usd: value_as_f64(r.remaining_usd.as_ref()),
        reset_seconds: value_as_u64(r.reset_seconds.as_ref()),
    });
    let top_budget = wire.budget.as_ref().map(|b| BudgetDetails {
        limit_usd: value_as_f64(b.limit_usd.as_ref()),
        spent_usd: value_as_f64(b.spent_usd.as_ref()),
        remaining_usd: value_as_f64(b.remaining_usd.as_ref()),
        reset_seconds: value_as_u64(b.reset_seconds.as_ref()),
    });
    let reason = wire.reason.and_then(|r| {
        if r.message.is_empty() {
            None
        } else {
            Some(r.message)
        }
    });
    Ok(Decision {
        allowed: wire.allow,
        fail_mode: FailMode::parse(&wire.fail_mode),
        reason,
        budget: top_budget.or(reason_budget),
    })
}

fn value_as_f64(value: Option<&Value>) -> Option<f64> {
    match value? {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

fn value_as_u64(value: Option<&Value>) -> Option<u64> {
    match value? {
        Value::Number(n) => n.as_u64(),
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn disabled_client_always_allows() {
        let c = BudgetClient::disabled();
        let d = c.check("any-key").await;
        assert!(d.allowed);
    }

    #[tokio::test]
    async fn live_client_returns_cp_api_decision() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/dp/budget_check"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "allow": true, "fail_mode": "sticky"
            })))
            .mount(&server)
            .await;

        let c = BudgetClient::new(server.uri(), reqwest::Client::new());
        let d = c.check("k-1").await;
        assert!(d.allowed);
        assert_eq!(d.fail_mode, FailMode::Sticky);
    }

    #[tokio::test]
    async fn live_client_parses_optional_budget_details() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/dp/budget_check"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "allow": true,
                "fail_mode": "sticky",
                "budget": {
                    "limit_usd": "10.5",
                    "spent_usd": 4.25,
                    "remaining_usd": "6.25",
                    "reset_seconds": 3600
                }
            })))
            .mount(&server)
            .await;

        let c = BudgetClient::new(server.uri(), reqwest::Client::new());
        let d = c.check("k-1").await;
        assert_eq!(
            d.budget,
            Some(BudgetDetails {
                limit_usd: Some(10.5),
                spent_usd: Some(4.25),
                remaining_usd: Some(6.25),
                reset_seconds: Some(3600),
            })
        );
    }

    #[tokio::test]
    async fn live_client_returns_deny_when_cp_says_no() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/dp/budget_check"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "allow": false,
                "fail_mode": "closed",
                "reason": {
                    "type": "billing_error",
                    "code": "budget_exceeded",
                    "message": "org budget 'monthly' exceeded ($10.00/month). Resets 2026-05-01 00:00 UTC.",
                    "scope": "org",
                    "scope_ref": "org-uuid-1",
                    "limit_usd": "10.00",
                    "spent_usd": "10.50",
                    "period": "month",
                    "period_resets_at": "2026-05-01T00:00:00Z",
                    "retry_after_seconds": 86400
                }
            })))
            .mount(&server)
            .await;

        let c = BudgetClient::new(server.uri(), reqwest::Client::new());
        let d = c.check("k-1").await;
        assert!(!d.allowed);
        assert_eq!(d.fail_mode, FailMode::Closed);
        assert!(d
            .reason
            .as_deref()
            .unwrap()
            .contains("org budget 'monthly' exceeded"));
    }

    #[tokio::test]
    async fn cache_hit_skips_network_within_ttl() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/dp/budget_check"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "allow": true, "fail_mode": "sticky"
            })))
            .expect(1)
            .mount(&server)
            .await;

        let c = BudgetClient::new(server.uri(), reqwest::Client::new());
        let _ = c.check("k-1").await;
        let _ = c.check("k-1").await;
        let _ = c.check("k-1").await;
        // expect(1) on Drop validates only one network call landed.
    }

    #[tokio::test]
    async fn fallback_serves_cache_when_cp_fails() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/dp/budget_check"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "allow": true, "fail_mode": "open"
            })))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        // Subsequent calls 500.
        Mock::given(method("GET"))
            .and(path("/dp/budget_check"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;

        let c = BudgetClient::new(server.uri(), reqwest::Client::new());
        let first = c.check("k-1").await;
        assert!(first.allowed);

        // Force expiry by manually invalidating the cache entry's age.
        // Easier: insert a stale fetched_at via direct cache mutation.
        if let Some(mut e) = c.cache.get_mut("k-1") {
            e.fetched_at = Instant::now() - Duration::from_secs(10);
        }
        let second = c.check("k-1").await;
        // cp-api now 500s but stale_max default is 600s, so the cached
        // decision is still served.
        assert!(second.allowed);
    }

    #[tokio::test]
    async fn fallback_with_no_cache_denies() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/dp/budget_check"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;

        let c = BudgetClient::new(server.uri(), reqwest::Client::new());
        let d = c.check("k-1").await;
        assert!(!d.allowed);
    }
}
