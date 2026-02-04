use std::{
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};

use async_trait::async_trait;
use skp_ratelimit::Quota;
use thiserror::Error;

use crate::config::entities::types::{RateLimit as RateLimitConfig, RateLimitMetric};

mod local;
mod utils;

pub use utils::RateLimitResponse;
pub use utils::RateLimitState;
pub use utils::post_check;
pub use utils::pre_check;

// Re-export types needed by middleware
pub use crate::config::entities::types::RateLimitMetric as Metric;

/// Rate limiter error types
#[derive(Debug, Clone, Error)]
pub enum RateLimitError {
    #[error("Rate limit exceeded")]
    Exceeded(RateLimitInfo),
    #[error("Internal error: {0}")]
    Internal(String),
}

#[derive(Clone)]
pub struct RateLimitRule {
    limit: u64,
    window_secs: u64,
}

impl RateLimitRule {
    pub fn new(limit: u64, window_secs: u64) -> Self {
        Self { limit, window_secs }
    }
}

impl RateLimitRule {
    pub async fn incoming(
        self,
        limiter: Arc<dyn RateLimiter>,
        key: &str,
        cost: u64,
        commit: bool,
    ) -> RateLimitResult {
        limiter.incoming(key, self, cost, commit).await
    }
}

impl Into<Quota> for RateLimitRule {
    fn into(self) -> Quota {
        Quota::new(self.limit, Duration::from_secs(self.window_secs))
    }
}

impl Into<Vec<(RateLimitMetric, RateLimitRule)>> for RateLimitConfig {
    fn into(self) -> Vec<(RateLimitMetric, RateLimitRule)> {
        let mut rules = Vec::<(RateLimitMetric, RateLimitRule)>::new();

        if let Some(n) = self.token_per_minute {
            rules.push((RateLimitMetric::TPM, RateLimitRule::new(n, 60)));
        }

        if let Some(n) = self.token_per_day {
            rules.push((RateLimitMetric::TPD, RateLimitRule::new(n, 86400)));
        }

        if let Some(n) = self.request_per_minute {
            rules.push((RateLimitMetric::RPM, RateLimitRule::new(n, 60)));
        }

        if let Some(n) = self.request_per_day {
            rules.push((RateLimitMetric::RPD, RateLimitRule::new(n, 86400)));
        }

        rules
    }
}

#[derive(Debug, Clone)]
pub struct RateLimitInfo {
    /// The limit of the evaluation
    pub limit: u64,
    /// Remaining credit in the current window
    pub remaining: u64,
    /// End of the current window
    pub reset_at: Instant,
    /// Start of the current window
    pub window_start: Instant,
    /// When will the next window open (available only when rate limited)
    pub retry_after: Option<Duration>,
}

pub type RateLimitResult = Result<RateLimitInfo, RateLimitError>;

#[async_trait]
pub trait RateLimiter: Send + Sync {
    /// Check and optionally increment a rate limit counter
    ///
    /// # Arguments
    /// * `key` - The rate limit key (e.g., "apikey:user1:rpm")
    /// * `rule` - The rate limit rule to apply
    /// * `cost` - The cost/amount to check/increment
    /// * `commit` - If false, only check without incrementing; if true, actually increment
    ///
    /// # Returns
    /// * `Ok(remaining)` - Check passed, returns remaining quota
    /// * `Err(RateLimitError::Exceeded)` - Limit exceeded
    async fn incoming(
        &self,
        key: &str,
        rule: RateLimitRule,
        cost: u64,
        commit: bool,
    ) -> RateLimitResult;
}

static RATE_LIMITER: OnceLock<Arc<dyn RateLimiter>> = OnceLock::new();

/// Create a rate limiter instance (local implementation by default)
pub fn get_rate_limiter() -> Arc<dyn RateLimiter> {
    RATE_LIMITER
        .get_or_init(|| Arc::new(local::LocalRateLimiter::default()))
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::entities::types::RateLimit;

    /// Test RateLimitRule::new() creates a rule with correct values
    /// Verifies that limit and window_secs are properly set
    #[test]
    fn test_rate_limit_rule_new() {
        let rule = RateLimitRule::new(100, 60);
        assert_eq!(rule.limit, 100);
        assert_eq!(rule.window_secs, 60);
    }

    /// Test RateLimitRule conversion to Quota
    /// Verifies that RateLimitRule can be converted to skp-ratelimit Quota
    #[test]
    fn test_rate_limit_rule_to_quota() {
        let rule = RateLimitRule::new(100, 60);
        let _quota: Quota = rule.into();
        // We can't directly inspect Quota fields, but we can verify it was created
        // The conversion is tested implicitly through the rate limiter tests
    }

    /// Test RateLimit to rules conversion with all metrics
    /// Verifies that all four metrics (TPM, TPD, RPM, RPD) are correctly converted to rules
    #[test]
    fn test_rate_limit_to_rules_all_metrics() {
        let rate_limit = RateLimit {
            token_per_minute: Some(1000),
            token_per_day: Some(100000),
            request_per_minute: Some(60),
            request_per_day: Some(5000),
            request_concurrency: None,
        };

        let rules: Vec<(RateLimitMetric, RateLimitRule)> = rate_limit.into();

        assert_eq!(rules.len(), 4);

        // Check TPM
        let tpm = rules
            .iter()
            .find(|(m, _)| matches!(m, RateLimitMetric::TPM));
        assert!(tpm.is_some());
        let (_, tpm_rule) = tpm.unwrap();
        assert_eq!(tpm_rule.limit, 1000);
        assert_eq!(tpm_rule.window_secs, 60);

        // Check TPD
        let tpd = rules
            .iter()
            .find(|(m, _)| matches!(m, RateLimitMetric::TPD));
        assert!(tpd.is_some());
        let (_, tpd_rule) = tpd.unwrap();
        assert_eq!(tpd_rule.limit, 100000);
        assert_eq!(tpd_rule.window_secs, 86400);

        // Check RPM
        let rpm = rules
            .iter()
            .find(|(m, _)| matches!(m, RateLimitMetric::RPM));
        assert!(rpm.is_some());
        let (_, rpm_rule) = rpm.unwrap();
        assert_eq!(rpm_rule.limit, 60);
        assert_eq!(rpm_rule.window_secs, 60);

        // Check RPD
        let rpd = rules
            .iter()
            .find(|(m, _)| matches!(m, RateLimitMetric::RPD));
        assert!(rpd.is_some());
        let (_, rpd_rule) = rpd.unwrap();
        assert_eq!(rpd_rule.limit, 5000);
        assert_eq!(rpd_rule.window_secs, 86400);
    }

    /// Test RateLimit to rules conversion with partial metrics
    /// Verifies that only specified metrics are converted to rules
    #[test]
    fn test_rate_limit_to_rules_partial_metrics() {
        let rate_limit = RateLimit {
            token_per_minute: Some(1000),
            token_per_day: None,
            request_per_minute: Some(60),
            request_per_day: None,
            request_concurrency: None,
        };

        let rules: Vec<(RateLimitMetric, RateLimitRule)> = rate_limit.into();

        assert_eq!(rules.len(), 2);

        // Should only have TPM and RPM
        assert!(rules.iter().any(|(m, _)| matches!(m, RateLimitMetric::TPM)));
        assert!(rules.iter().any(|(m, _)| matches!(m, RateLimitMetric::RPM)));
        assert!(!rules.iter().any(|(m, _)| matches!(m, RateLimitMetric::TPD)));
        assert!(!rules.iter().any(|(m, _)| matches!(m, RateLimitMetric::RPD)));
    }

    /// Test RateLimit to rules conversion with empty config
    /// Verifies that no rules are generated when all metrics are None
    #[test]
    fn test_rate_limit_to_rules_empty() {
        let rate_limit = RateLimit {
            token_per_minute: None,
            token_per_day: None,
            request_per_minute: None,
            request_per_day: None,
            request_concurrency: None,
        };

        let rules: Vec<(RateLimitMetric, RateLimitRule)> = rate_limit.into();

        assert_eq!(rules.len(), 0);
    }

    /// Test RateLimitError Display implementation
    /// Verifies that error messages are formatted correctly
    #[test]
    fn test_rate_limit_error_display() {
        let now = Instant::now();
        let info = RateLimitInfo {
            limit: 100,
            remaining: 0,
            reset_at: now + Duration::from_secs(60),
            window_start: now,
            retry_after: Some(Duration::from_secs(60)),
        };

        let error = RateLimitError::Exceeded(info);
        assert_eq!(error.to_string(), "Rate limit exceeded");

        let error = RateLimitError::Internal("test error".to_string());
        assert_eq!(error.to_string(), "Internal error: test error");
    }

    // Integration tests - Testing the complete pre_check and post_check flow

    // Mock entity for testing
    #[derive(Clone)]
    struct MockEntity {
        id: String,
        rate_limit: Option<RateLimit>,
    }

    impl crate::config::entities::types::HasRateLimit for MockEntity {
        fn rate_limit(&self) -> Option<RateLimit> {
            self.rate_limit.clone()
        }

        fn rate_limit_key(&self, metric: RateLimitMetric) -> String {
            format!("{}:{}:{}", "test", self.id, metric)
        }
    }

    /// Integration test: pre_check commits request metrics
    /// Verifies that RPM/RPD metrics are committed (decremented) during pre_check
    #[tokio::test]
    async fn test_pre_check_commits_request_metrics() {
        let entity = MockEntity {
            id: "pre_check_req_1".to_string(),
            rate_limit: Some(RateLimit {
                token_per_minute: None,
                token_per_day: None,
                request_per_minute: Some(10),
                request_per_day: None,
                request_concurrency: None,
            }),
        };

        // First request should succeed
        let result1 = utils::pre_check(&entity).await;
        assert!(result1.is_ok());
        let info1 = result1.unwrap();
        assert_eq!(info1.len(), 1);
        assert_eq!(info1[0].1.remaining, 9);

        // Second request should also succeed and decrement
        let result2 = utils::pre_check(&entity).await;
        assert!(result2.is_ok());
        let info2 = result2.unwrap();
        assert_eq!(info2[0].1.remaining, 8);
    }

    /// Integration test: pre_check does NOT commit token metrics
    /// Verifies that TPM/TPD metrics are only checked (not decremented) during pre_check
    #[tokio::test]
    async fn test_pre_check_does_not_commit_token_metrics() {
        let entity = MockEntity {
            id: "pre_check_token_1".to_string(),
            rate_limit: Some(RateLimit {
                token_per_minute: Some(1000),
                token_per_day: None,
                request_per_minute: None,
                request_per_day: None,
                request_concurrency: None,
            }),
        };

        // Multiple pre_checks should not decrement token count
        let result1 = utils::pre_check(&entity).await;
        assert!(result1.is_ok());
        let info1 = result1.unwrap();
        assert_eq!(info1[0].1.remaining, 1000);

        let result2 = utils::pre_check(&entity).await;
        assert!(result2.is_ok());
        let info2 = result2.unwrap();
        assert_eq!(info2[0].1.remaining, 1000); // Should still be 1000
    }

    /// Integration test: post_check commits token metrics
    /// Verifies that TPM/TPD metrics are committed (decremented) during post_check
    #[tokio::test]
    async fn test_post_check_commits_token_metrics() {
        let entity = MockEntity {
            id: "post_check_token_1".to_string(),
            rate_limit: Some(RateLimit {
                token_per_minute: Some(1000),
                token_per_day: None,
                request_per_minute: None,
                request_per_day: None,
                request_concurrency: None,
            }),
        };

        // post_check should commit tokens
        let result1 = utils::post_check(&entity, 100).await;
        assert!(result1.is_ok());
        let info1 = result1.unwrap();
        assert_eq!(info1[0].1.remaining, 900);

        // Another post_check should further decrement
        let result2 = utils::post_check(&entity, 50).await;
        assert!(result2.is_ok());
        let info2 = result2.unwrap();
        assert_eq!(info2[0].1.remaining, 850);
    }

    /// Integration test: post_check skips request metrics
    /// Verifies that RPM/RPD metrics are ignored during post_check (already committed in pre_check)
    #[tokio::test]
    async fn test_post_check_skips_request_metrics() {
        let entity = MockEntity {
            id: "post_check_req_1".to_string(),
            rate_limit: Some(RateLimit {
                token_per_minute: None,
                token_per_day: None,
                request_per_minute: Some(10),
                request_per_day: None,
                request_concurrency: None,
            }),
        };

        // post_check should return empty results for request-only limits
        let result = utils::post_check(&entity, 1).await;
        assert!(result.is_ok());
        let info = result.unwrap();
        assert_eq!(info.len(), 0); // No results for request metrics
    }

    /// Integration test: Full flow with RateLimitState
    /// Verifies the complete request lifecycle: pre_check -> store -> post_check -> update state
    #[tokio::test]
    async fn test_full_flow_with_rate_limit_state() {
        let entity = MockEntity {
            id: "full_flow_1".to_string(),
            rate_limit: Some(RateLimit {
                token_per_minute: Some(1000),
                token_per_day: None,
                request_per_minute: Some(10),
                request_per_day: None,
                request_concurrency: None,
            }),
        };

        let mut state = utils::RateLimitState::new();

        // Pre-check: commits requests, checks tokens
        let pre_result = utils::pre_check(&entity).await;
        assert!(pre_result.is_ok());
        state.store_pre_check(pre_result.unwrap());

        assert!(state.request_info.is_some());
        assert!(state.token_info.is_some());
        assert_eq!(state.request_info.as_ref().unwrap().remaining, 9);
        assert_eq!(state.token_info.as_ref().unwrap().remaining, 1000);

        // Post-check: commits tokens with actual cost
        let post_result = utils::post_check(&entity, 150).await;
        assert!(post_result.is_ok());
        state.store_post_check(post_result.unwrap());

        // Request info should remain unchanged
        assert_eq!(state.request_info.as_ref().unwrap().remaining, 9);
        // Token info should be updated
        assert_eq!(state.token_info.as_ref().unwrap().remaining, 850);
    }

    /// Integration test: Rate limit exceeded in pre_check
    /// Verifies that pre_check returns an error when request limit is exceeded
    #[tokio::test]
    async fn test_rate_limit_exceeded_in_pre_check() {
        let entity = MockEntity {
            id: "exceeded_1".to_string(),
            rate_limit: Some(RateLimit {
                token_per_minute: None,
                token_per_day: None,
                request_per_minute: Some(3),
                request_per_day: None,
                request_concurrency: None,
            }),
        };

        // Use up the limit
        for _ in 0..3 {
            let result = utils::pre_check(&entity).await;
            assert!(result.is_ok());
        }

        // Next request should fail
        let result = utils::pre_check(&entity).await;
        assert!(result.is_err());

        let (metric, error) = result.unwrap_err();
        assert!(matches!(metric, RateLimitMetric::RPM));
        assert!(matches!(error, RateLimitError::Exceeded(_)));
    }

    /// Integration test: Rate limit exceeded in post_check
    /// Verifies that post_check returns an error when token limit is exceeded
    #[tokio::test]
    async fn test_rate_limit_exceeded_in_post_check() {
        let entity = MockEntity {
            id: "exceeded_2".to_string(),
            rate_limit: Some(RateLimit {
                token_per_minute: Some(100),
                token_per_day: None,
                request_per_minute: None,
                request_per_day: None,
                request_concurrency: None,
            }),
        };

        // Use up most of the limit
        let result1 = utils::post_check(&entity, 90).await;
        assert!(result1.is_ok());

        // This should exceed the limit
        let result2 = utils::post_check(&entity, 20).await;
        assert!(result2.is_err());

        let (metric, error) = result2.unwrap_err();
        assert!(matches!(metric, RateLimitMetric::TPM));
        assert!(matches!(error, RateLimitError::Exceeded(_)));
    }

    /// Integration test: No rate limit configured
    /// Verifies that pre_check and post_check return empty results when no rate limit is set
    #[tokio::test]
    async fn test_no_rate_limit_configured() {
        let entity = MockEntity {
            id: "no_limit_1".to_string(),
            rate_limit: None,
        };

        let pre_result = utils::pre_check(&entity).await;
        assert!(pre_result.is_ok());
        assert_eq!(pre_result.unwrap().len(), 0);

        let post_result = utils::post_check(&entity, 100).await;
        assert!(post_result.is_ok());
        assert_eq!(post_result.unwrap().len(), 0);
    }

    /// Integration test: Multiple entities are isolated
    /// Verifies that different entities have independent rate limit counters
    #[tokio::test]
    async fn test_multiple_entities_isolated() {
        let entity1 = MockEntity {
            id: "entity_1".to_string(),
            rate_limit: Some(RateLimit {
                token_per_minute: None,
                token_per_day: None,
                request_per_minute: Some(5),
                request_per_day: None,
                request_concurrency: None,
            }),
        };

        let entity2 = MockEntity {
            id: "entity_2".to_string(),
            rate_limit: Some(RateLimit {
                token_per_minute: None,
                token_per_day: None,
                request_per_minute: Some(5),
                request_per_day: None,
                request_concurrency: None,
            }),
        };

        // Use up entity1's limit
        for _ in 0..5 {
            let result = utils::pre_check(&entity1).await;
            assert!(result.is_ok());
        }

        // entity1 should be rate limited
        let result = utils::pre_check(&entity1).await;
        assert!(result.is_err());

        // entity2 should still work
        let result = utils::pre_check(&entity2).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap()[0].1.remaining, 4);
    }

    /// Integration test: Rate limit state generates correct HTTP headers
    /// Verifies that RateLimitState produces all required x-ratelimit-* headers
    #[tokio::test]
    async fn test_rate_limit_state_headers() {
        let entity = MockEntity {
            id: "headers_1".to_string(),
            rate_limit: Some(RateLimit {
                token_per_minute: Some(1000),
                token_per_day: None,
                request_per_minute: Some(60),
                request_per_day: None,
                request_concurrency: None,
            }),
        };

        let mut state = utils::RateLimitState::new();

        // Pre-check
        let pre_result = utils::pre_check(&entity).await;
        assert!(pre_result.is_ok());
        state.store_pre_check(pre_result.unwrap());

        // Post-check
        let post_result = utils::post_check(&entity, 100).await;
        assert!(post_result.is_ok());
        state.store_post_check(post_result.unwrap());

        // Generate headers
        let mut headers = http::HeaderMap::new();
        state.add_headers(&mut headers);

        // Verify headers exist
        assert!(headers.get("x-ratelimit-limit-requests").is_some());
        assert!(headers.get("x-ratelimit-remaining-requests").is_some());
        assert!(headers.get("x-ratelimit-reset-requests").is_some());
        assert!(headers.get("x-ratelimit-limit-tokens").is_some());
        assert!(headers.get("x-ratelimit-remaining-tokens").is_some());
        assert!(headers.get("x-ratelimit-reset-tokens").is_some());

        // Verify header values
        assert_eq!(headers.get("x-ratelimit-limit-requests").unwrap(), "60");
        assert_eq!(headers.get("x-ratelimit-remaining-requests").unwrap(), "59");
        assert_eq!(headers.get("x-ratelimit-limit-tokens").unwrap(), "1000");
        assert_eq!(headers.get("x-ratelimit-remaining-tokens").unwrap(), "900");
    }

    /// Integration test: Mixed per-minute and per-day metrics
    /// Verifies that both per-minute and per-day limits work together correctly
    #[tokio::test]
    async fn test_mixed_metrics_per_minute_and_per_day() {
        let entity = MockEntity {
            id: "mixed_1".to_string(),
            rate_limit: Some(RateLimit {
                token_per_minute: Some(100),
                token_per_day: Some(10000),
                request_per_minute: Some(10),
                request_per_day: Some(1000),
                request_concurrency: None,
            }),
        };

        // Pre-check should return 4 results (RPM, RPD, TPM, TPD)
        let pre_result = utils::pre_check(&entity).await;
        assert!(pre_result.is_ok());
        let pre_info = pre_result.unwrap();
        assert_eq!(pre_info.len(), 4);

        // Verify all metrics are present
        let metrics: Vec<_> = pre_info.iter().map(|(m, _)| m).collect();
        assert!(metrics.iter().any(|m| matches!(m, RateLimitMetric::RPM)));
        assert!(metrics.iter().any(|m| matches!(m, RateLimitMetric::RPD)));
        assert!(metrics.iter().any(|m| matches!(m, RateLimitMetric::TPM)));
        assert!(metrics.iter().any(|m| matches!(m, RateLimitMetric::TPD)));

        // Post-check should return 2 results (TPM, TPD only)
        let post_result = utils::post_check(&entity, 50).await;
        assert!(post_result.is_ok());
        let post_info = post_result.unwrap();
        assert_eq!(post_info.len(), 2);

        let post_metrics: Vec<_> = post_info.iter().map(|(m, _)| m).collect();
        assert!(
            post_metrics
                .iter()
                .any(|m| matches!(m, RateLimitMetric::TPM))
        );
        assert!(
            post_metrics
                .iter()
                .any(|m| matches!(m, RateLimitMetric::TPD))
        );
    }
}
