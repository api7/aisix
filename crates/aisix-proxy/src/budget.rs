//! Budget client — asks cp-api per request whether an api_key may proceed.
//!
//! Decisions are cached in an LRU (capacity 10000, TTL 5s) keyed by
//! api_key_id. When cp-api is unreachable we honor the last cached
//! decision (sticky) up to AISIX_DP_BUDGET_STALE_MAX_SECONDS (default
//! 600s); past that we apply the fail_mode that came back on the last
//! good response.

use dashmap::DashMap;
use serde::Deserialize;
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
}

impl Decision {
    fn allow_all() -> Self {
        Self {
            allowed: true,
            fail_mode: FailMode::Open,
            reason: None,
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
        token: Option<String>,
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
    /// Live client that asks cp-api per request. The base URL should be the
    /// cp-api root (e.g. `https://cp.aisix.cloud`). Auth token comes from the
    /// `AISIX_DP_CP_TOKEN` env var; if unset, requests go without an
    /// Authorization header (cp-api will reject them — operators should run
    /// `disabled()` instead in that case).
    pub fn new(base_url: impl Into<String>, http: reqwest::Client) -> Self {
        let base_url = base_url.into().trim_end_matches('/').to_string();
        let token = std::env::var("AISIX_DP_CP_TOKEN").ok();
        let stale_max = std::env::var("AISIX_DP_BUDGET_STALE_MAX_SECONDS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(DEFAULT_STALE_MAX_SECONDS);
        Self {
            mode: Mode::Live {
                http,
                base_url,
                token,
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
                token,
                stale_max,
            } => {
                // Fast path: cache hit within TTL.
                if let Some(cached) = self.cache.get(api_key_id) {
                    if cached.fetched_at.elapsed() < CACHE_TTL {
                        return cached.decision.clone();
                    }
                }

                match fetch_decision(http, base_url, token.as_deref(), api_key_id).await {
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
        },
        FailMode::Closed => Decision {
            allowed: false,
            fail_mode: FailMode::Closed,
            reason: Some("cp-api unreachable; fail_mode=closed".to_string()),
        },
        FailMode::Sticky => Decision {
            allowed: false,
            fail_mode: FailMode::Sticky,
            reason: Some("cp-api unreachable; cached decision stale".to_string()),
        },
    }
}

#[derive(Debug, Deserialize)]
struct WireEnvelope {
    data: WireDecision,
}

#[derive(Debug, Deserialize)]
struct WireDecision {
    allowed: bool,
    #[serde(default)]
    fail_mode: String,
    #[serde(default)]
    reason: Option<String>,
}

async fn fetch_decision(
    http: &reqwest::Client,
    base_url: &str,
    token: Option<&str>,
    api_key_id: &str,
) -> Result<Decision, reqwest::Error> {
    let url = format!("{base_url}/api/internal/budget_check");
    let mut req = http
        .post(url)
        .json(&serde_json::json!({ "api_key_id": api_key_id }));
    if let Some(tok) = token {
        req = req.bearer_auth(tok);
    }
    let resp = req.send().await?.error_for_status()?;
    let env: WireEnvelope = resp.json().await?;
    Ok(Decision {
        allowed: env.data.allowed,
        fail_mode: FailMode::parse(&env.data.fail_mode),
        reason: env.data.reason,
    })
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
        Mock::given(method("POST"))
            .and(path("/api/internal/budget_check"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": {"allowed": true, "fail_mode": "sticky"}
            })))
            .mount(&server)
            .await;

        let c = BudgetClient::new(server.uri(), reqwest::Client::new());
        let d = c.check("k-1").await;
        assert!(d.allowed);
        assert_eq!(d.fail_mode, FailMode::Sticky);
    }

    #[tokio::test]
    async fn live_client_returns_deny_when_cp_says_no() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/internal/budget_check"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": {"allowed": false, "fail_mode": "closed", "reason": "monthly cap"}
            })))
            .mount(&server)
            .await;

        let c = BudgetClient::new(server.uri(), reqwest::Client::new());
        let d = c.check("k-1").await;
        assert!(!d.allowed);
        assert_eq!(d.fail_mode, FailMode::Closed);
        assert_eq!(d.reason.as_deref(), Some("monthly cap"));
    }

    #[tokio::test]
    async fn cache_hit_skips_network_within_ttl() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/internal/budget_check"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": {"allowed": true, "fail_mode": "sticky"}
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
        Mock::given(method("POST"))
            .and(path("/api/internal/budget_check"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": {"allowed": true, "fail_mode": "open"}
            })))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        // Subsequent calls 500.
        Mock::given(method("POST"))
            .and(path("/api/internal/budget_check"))
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
        Mock::given(method("POST"))
            .and(path("/api/internal/budget_check"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;

        let c = BudgetClient::new(server.uri(), reqwest::Client::new());
        let d = c.check("k-1").await;
        assert!(!d.allowed);
    }
}
