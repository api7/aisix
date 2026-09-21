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
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
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
    pub reason: Option<BudgetReason>,
    pub budget: Option<BudgetDetails>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct BudgetDetails {
    pub limit_usd: Option<f64>,
    pub spent_usd: Option<f64>,
    pub remaining_usd: Option<f64>,
    pub reset_seconds: Option<u64>,
}

/// Customer-facing detail for a budget denial, forwarded from cp-api's
/// `BudgetCheckReason` (prd-09b §5.8). The DP lifts these into the 429
/// `error` block so a programmatic client can see *which* budget tripped
/// (`scope` / `scope_ref`) and by how much (`limit_usd` / `spent_usd`),
/// not just the human `message`. Every field beyond `message` is
/// optional: the cp-api-unreachable fallback decisions carry only a
/// message, with no structured detail.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BudgetReason {
    pub message: String,
    pub scope: Option<String>,
    pub scope_ref: Option<String>,
    pub limit_usd: Option<String>,
    pub spent_usd: Option<String>,
    pub period: Option<String>,
    pub period_resets_at: Option<String>,
    pub retry_after_seconds: Option<u64>,
}

impl BudgetReason {
    /// A reason carrying only a human message — used by the
    /// cp-api-unreachable fallback paths (and the api_key fallback in
    /// chat dispatch), which have no structured scope detail.
    pub(crate) fn message_only(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            ..Default::default()
        }
    }
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
    /// Which insertion produced this entry, so the expiry queue can tell
    /// its own record of a key from a later refresh of the same key.
    seq: u64,
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
    /// The cached keys in insertion order, which is also expiry order:
    /// every entry carries the same TTL, so the entry inserted first is
    /// the one to drop when the cache is full.
    ///
    /// The map alone cannot answer that without reading all of it, and
    /// at capacity every miss is an insert — a deployment with more
    /// active api_keys than [`CACHE_CAPACITY`] would scan ten thousand
    /// entries on a request worker for each one.
    ///
    /// A key refreshed while it is already cached leaves its earlier
    /// record behind; such a record names a `seq` the map no longer
    /// holds, and is discarded when it reaches the front rather than
    /// evicting a key that is not in fact the oldest.
    expiry: Mutex<VecDeque<(u64, String)>>,
    inserts: AtomicU64,
    /// Cached entries inspected while choosing one to evict. The point
    /// of the queue is that this stays flat as the cache fills.
    #[cfg(test)]
    inspected: AtomicU64,
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
            expiry: Mutex::new(VecDeque::new()),
            inserts: AtomicU64::new(0),
            #[cfg(test)]
            inspected: AtomicU64::new(0),
        }
    }

    /// Allow-all client. Used in dev / tests / standalone mode where no
    /// cp-api is reachable.
    pub fn disabled() -> Self {
        Self {
            mode: Mode::Disabled,
            cache: DashMap::new(),
            expiry: Mutex::new(VecDeque::new()),
            inserts: AtomicU64::new(0),
            #[cfg(test)]
            inspected: AtomicU64::new(0),
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
            reason: Some(BudgetReason::message_only(
                "cp-api unreachable and no cached decision",
            )),
            budget: None,
        }
    }

    fn insert(&self, api_key_id: &str, decision: Decision) {
        let seq = self.inserts.fetch_add(1, Ordering::Relaxed);
        let mut expiry = self.expiry.lock().expect("budget cache expiry order");
        // Drop records the map has moved past, and — only when the cache
        // is full — the oldest key still in it. Both walk the front of
        // the same queue, and each record is looked at once in its life,
        // so an insert costs a constant number of entries however large
        // the cache is.
        let mut evicting = self.cache.len() >= CACHE_CAPACITY;
        while let Some((recorded, key)) = expiry.front() {
            #[cfg(test)]
            self.inspected.fetch_add(1, Ordering::Relaxed);
            let current = self
                .cache
                .get(key)
                .is_some_and(|entry| entry.seq == *recorded);
            if !current {
                expiry.pop_front();
                continue;
            }
            if !evicting {
                break;
            }
            let (_, oldest) = expiry.pop_front().expect("a front that was just read");
            self.cache.remove(&oldest);
            evicting = false;
        }
        expiry.push_back((seq, api_key_id.to_string()));
        self.cache.insert(
            api_key_id.to_string(),
            CacheEntry {
                decision,
                fetched_at: Instant::now(),
                seq,
            },
        );
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
            reason: Some(BudgetReason::message_only(
                "cp-api unreachable; fail_mode=closed",
            )),
            budget: prev.budget.clone(),
        },
        FailMode::Sticky => Decision {
            allowed: false,
            fail_mode: FailMode::Sticky,
            reason: Some(BudgetReason::message_only(
                "cp-api unreachable; cached decision stale",
            )),
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
/// A data plane can run against a control plane many releases newer than
/// itself, so this struct — and every other one that decodes a control-plane
/// response — never carries `#[serde(deny_unknown_fields)]`, and every field
/// except `allow`, the one the decision hinges on, is `#[serde(default)]`.
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
    scope: Option<String>,
    #[serde(default)]
    scope_ref: Option<String>,
    #[serde(default)]
    limit_usd: Option<Value>,
    #[serde(default)]
    spent_usd: Option<Value>,
    #[serde(default)]
    remaining_usd: Option<Value>,
    #[serde(default)]
    period: Option<String>,
    #[serde(default)]
    period_resets_at: Option<String>,
    // cp-api's reason carries `retry_after_seconds` (int). Aliased to
    // `reset_seconds` so the existing BudgetDetails (gauge) path keeps
    // reading the same field, and the BudgetReason path reuses it for
    // retry_after_seconds.
    #[serde(default, alias = "retry_after_seconds")]
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

/// Path the budget gate asks cp-api on, under the dpmgr origin the
/// heartbeat worker uses.
pub const BUDGET_CHECK_PATH: &str = "/dp/budget_check";

async fn fetch_decision(
    http: &reqwest::Client,
    base_url: &str,
    api_key_id: &str,
) -> Result<Decision, reqwest::Error> {
    let url = format!("{base_url}{BUDGET_CHECK_PATH}");
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
    // Lift cp-api's structured reason into the customer-facing detail
    // (prd-09b §5.8). limit_usd / spent_usd are formatted to the 2dp
    // dollar-string cp-api itself uses. A reason with neither a message
    // nor any structured field is treated as absent.
    let reason = wire.reason.map(|r| BudgetReason {
        message: r.message,
        scope: r.scope,
        scope_ref: r.scope_ref,
        limit_usd: value_as_f64(r.limit_usd.as_ref()).map(|v| format!("{v:.2}")),
        spent_usd: value_as_f64(r.spent_usd.as_ref()).map(|v| format!("{v:.2}")),
        period: r.period,
        period_resets_at: r.period_resets_at,
        retry_after_seconds: value_as_u64(r.reset_seconds.as_ref()),
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

    /// The budget gate is handed the dpmgr origin `main` derives from
    /// `managed.cp_base_url`. A scheme-less value used to make that
    /// `127.0.0.1:7944/dp/budget_check`, which reqwest rejects at
    /// request BUILD — and since a build failure is indistinguishable
    /// from an unreachable CP, `fallback()` with no cache sticky-denies,
    /// so every proxied request answered `429 budget_exceeded` while
    /// etcd stayed connected and the console showed the gateway healthy
    /// (AISIX-Cloud#1643).
    ///
    /// A peer that accepts and hangs up separates the two: a builder
    /// error means no request was formed, any transport error means one
    /// was and only the exchange failed.
    #[tokio::test]
    async fn budget_check_request_is_built_from_a_scheme_less_cp_base_url() {
        // The listener is HELD for the whole test and answers by
        // closing the connection immediately. Binding a port and
        // dropping it would leave a window in which another process on
        // a busy CI box takes it, and the probe below needs the peer's
        // behaviour to be deterministic, not merely likely.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                drop(stream);
            }
        });
        let file = tempfile::Builder::new().suffix(".yaml").tempfile().unwrap();
        std::fs::write(
            file.path(),
            format!(
                r#"
etcd:
  endpoints: ["http://127.0.0.1:2379"]
  prefix: "/aisix"
proxy:
  addr: "0.0.0.0:3000"
admin:
  addr: "127.0.0.1:3001"
  admin_keys: ["k1"]
managed:
  enabled: true
  cp_base_url: "127.0.0.1:{port}"
"#
            ),
        )
        .unwrap();
        let loaded = aisix_core::Config::load_from_path(Some(file.path())).unwrap();
        // `main` hands BudgetClient the heartbeat URL minus its path,
        // which for a base carrying no trailing slash is the base itself.
        let base = loaded.managed.cp_base_url.clone().unwrap();
        assert_eq!(base, format!("https://127.0.0.1:{port}"));

        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap();
        let err = fetch_decision(&http, &base, "ak_test")
            .await
            .expect_err("the peer hangs up without answering");
        assert!(
            !err.is_builder(),
            "the budget_check request was never built: {err}"
        );
    }

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
        let r = d.reason.expect("reason present");
        assert!(r.message.contains("org budget 'monthly' exceeded"));
        // Structured fields must be lifted from cp-api's reason (#433),
        // not dropped at deserialization.
        assert_eq!(r.scope.as_deref(), Some("org"));
        assert_eq!(r.scope_ref.as_deref(), Some("org-uuid-1"));
        assert_eq!(r.limit_usd.as_deref(), Some("10.00"));
        assert_eq!(r.spent_usd.as_deref(), Some("10.50"));
        assert_eq!(r.period.as_deref(), Some("month"));
        assert_eq!(r.period_resets_at.as_deref(), Some("2026-05-01T00:00:00Z"));
        assert_eq!(r.retry_after_seconds, Some(86_400));
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

    fn decision(allowed: bool) -> Decision {
        Decision {
            allowed,
            fail_mode: FailMode::Sticky,
            reason: None,
            budget: None,
        }
    }

    /// A deployment with more active api_keys than the cache holds
    /// inserts on every miss, and every one of those inserts happens on
    /// a request worker. What must hold is that the work of choosing
    /// what to drop does not grow with the cache: a full cache costs the
    /// same per insert as an empty one.
    #[test]
    fn evicting_from_a_full_cache_does_not_read_the_cache() {
        let client = BudgetClient::disabled();
        for i in 0..CACHE_CAPACITY {
            client.insert(&format!("key-{i:06}"), decision(true));
        }
        assert_eq!(client.cache.len(), CACHE_CAPACITY);
        let filling = client.inspected.load(Ordering::Relaxed);

        const MISSES: u64 = 2_000;
        for i in 0..MISSES {
            client.insert(&format!("fresh-{i:06}"), decision(true));
        }
        let evicting = client.inspected.load(Ordering::Relaxed) - filling;
        assert!(
            evicting <= MISSES * 2,
            "choosing what to evict must cost a constant per insert, not the \
             cache's size: {evicting} entries read over {MISSES} inserts into a \
             cache of {CACHE_CAPACITY}",
        );
        assert_eq!(client.cache.len(), CACHE_CAPACITY, "the cap still holds");
        assert!(
            client.cache.get("key-000000").is_none(),
            "the oldest entry is the one evicted",
        );
        assert!(
            client.cache.get(&format!("key-{:06}", CACHE_CAPACITY - 1)).is_some(),
            "and the newest survivors are not",
        );
    }

    /// A key seen again before it expires is the NEWEST entry, not the
    /// oldest, whatever its first insertion recorded.
    #[test]
    fn a_refreshed_key_is_not_evicted_as_the_oldest() {
        let client = BudgetClient::disabled();
        for i in 0..CACHE_CAPACITY {
            client.insert(&format!("key-{i:06}"), decision(true));
        }
        // The two oldest keys by first insertion; refresh one of them.
        client.insert("key-000000", decision(false));

        client.insert("brand-new", decision(true));
        assert!(
            client.cache.get("key-000000").is_some(),
            "a refreshed key must not be dropped on its stale record",
        );
        assert!(
            client.cache.get("key-000001").is_none(),
            "the genuinely oldest key is the one dropped",
        );
        assert_eq!(
            client.cache.get("key-000000").map(|e| e.decision.allowed),
            Some(false),
            "and the refreshed key keeps its refreshed decision",
        );
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
