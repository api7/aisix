//! Local in-memory rate limiter implementation

use std::sync::LazyLock;

use async_trait::async_trait;
use skp_ratelimit::Algorithm;

use super::{RateLimitError, RateLimitInfo, RateLimitRule, RateLimiter};

static FIXED_WINDOW: LazyLock<skp_ratelimit::FixedWindow> =
    LazyLock::new(skp_ratelimit::FixedWindow::new);
static MEMORY_STORAGE: LazyLock<skp_ratelimit::MemoryStorage> =
    LazyLock::new(skp_ratelimit::MemoryStorage::new);

#[derive(Default)]
pub struct LocalRateLimiter;

#[async_trait]
impl RateLimiter for LocalRateLimiter {
    async fn incoming(
        &self,
        key: &str,
        rule: RateLimitRule,
        cost: u64,
        commit: bool,
    ) -> Result<RateLimitInfo, RateLimitError> {
        let limit = rule.limit;
        let res = if commit {
            FIXED_WINDOW
                .check_and_record(&*MEMORY_STORAGE, key, &rule.into(), cost)
                .await
        } else {
            FIXED_WINDOW
                .check(&*MEMORY_STORAGE, key, &rule.into())
                .await
        };

        match res {
            Ok(decision) => {
                let info = decision.info();
                if decision.is_allowed() {
                    Ok(RateLimitInfo {
                        limit,
                        remaining: info.remaining,
                        reset_at: info.reset_at,
                        window_start: info.window_start,
                        retry_after: None,
                    })
                } else {
                    Err(RateLimitError::Exceeded(RateLimitInfo {
                        limit,
                        remaining: 0,
                        reset_at: info.reset_at,
                        window_start: info.window_start,
                        retry_after: info.retry_after,
                    }))
                }
            }
            Err(err) => Err(RateLimitError::Internal(format!("{err}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use http::HeaderMap;

    use super::{
        super::{CheckPhase, RateLimitState, run_check},
        *,
    };
    use crate::config::entities::types::{HasRateLimit, RateLimit, RateLimitMetric};

    // --- MockEntity helper shared by integration tests ---

    #[derive(Clone)]
    struct MockEntity {
        id: String,
        rate_limit: Option<RateLimit>,
    }

    impl HasRateLimit for MockEntity {
        fn rate_limit(&self) -> Option<RateLimit> {
            self.rate_limit.clone()
        }
        fn rate_limit_key(&self, metric: RateLimitMetric) -> String {
            format!("test:{}:{}", self.id, metric)
        }
    }

    fn make_entity(id: &str, rpm: Option<u64>, tpm: Option<u64>) -> MockEntity {
        MockEntity {
            id: id.to_string(),
            rate_limit: Some(RateLimit {
                token_per_minute: tpm,
                token_per_day: None,
                request_per_minute: rpm,
                request_per_day: None,
                request_concurrency: None,
            }),
        }
    }

    /// Test that LocalRateLimiter allows requests within the limit
    /// Verifies that a request within quota succeeds and decrements the remaining count
    #[tokio::test]
    async fn test_local_rate_limiter_allows_within_limit() {
        let limiter = LocalRateLimiter;
        let rule = RateLimitRule::new(10, 60);

        let result = limiter.incoming("test_key_1", rule, 1, true).await;

        assert!(result.is_ok());
        let info = result.unwrap();
        assert_eq!(info.limit, 10);
        assert_eq!(info.remaining, 9);
    }

    /// Test that LocalRateLimiter rejects requests exceeding the limit
    /// Verifies that requests beyond quota are rejected with RateLimitError::Exceeded
    #[tokio::test]
    async fn test_local_rate_limiter_rejects_exceeding_limit() {
        let limiter = LocalRateLimiter;
        let rule = RateLimitRule::new(3, 60);
        let key = "test_key_2";

        // Use up the limit
        for _ in 0..3 {
            let result = limiter.incoming(key, rule.clone(), 1, true).await;
            assert!(result.is_ok());
        }

        // Next request should be rejected
        let result = limiter.incoming(key, rule, 1, true).await;
        assert!(result.is_err());

        match result {
            Err(RateLimitError::Exceeded(info)) => {
                assert_eq!(info.limit, 3);
                assert_eq!(info.remaining, 0);
                assert!(info.retry_after.is_some());
            }
            _ => panic!("Expected RateLimitError::Exceeded"),
        }
    }

    /// Test LocalRateLimiter check-only mode (commit=false)
    /// Verifies that check-only mode doesn't decrement the counter
    #[tokio::test]
    async fn test_local_rate_limiter_check_only_mode() {
        let limiter = LocalRateLimiter;
        let rule = RateLimitRule::new(10, 60);
        let key = "test_key_3";

        // Check without committing (commit=false)
        let result1 = limiter.incoming(key, rule.clone(), 1, false).await;
        assert!(result1.is_ok());
        assert_eq!(result1.unwrap().remaining, 10); // Should not decrement

        // Check again without committing
        let result2 = limiter.incoming(key, rule.clone(), 1, false).await;
        assert!(result2.is_ok());
        assert_eq!(result2.unwrap().remaining, 10); // Still not decremented

        // Now commit
        let result3 = limiter.incoming(key, rule, 1, true).await;
        assert!(result3.is_ok());
        assert_eq!(result3.unwrap().remaining, 9); // Now decremented
    }

    /// Test LocalRateLimiter with custom cost values
    /// Verifies that custom cost values are properly deducted from the quota
    #[tokio::test]
    async fn test_local_rate_limiter_custom_cost() {
        let limiter = LocalRateLimiter;
        let rule = RateLimitRule::new(100, 60);
        let key = "test_key_4";

        // Use 50 tokens
        let result1 = limiter.incoming(key, rule.clone(), 50, true).await;
        assert!(result1.is_ok());
        assert_eq!(result1.unwrap().remaining, 50);

        // Use another 30 tokens
        let result2 = limiter.incoming(key, rule, 30, true).await;
        assert!(result2.is_ok());
        assert_eq!(result2.unwrap().remaining, 20);
    }

    /// Test LocalRateLimiter key isolation
    /// Verifies that different keys have independent rate limit counters
    #[tokio::test]
    async fn test_local_rate_limiter_key_isolation() {
        let limiter = LocalRateLimiter;
        let rule = RateLimitRule::new(5, 60);

        // Use up limit for key1
        for _ in 0..5 {
            let result = limiter.incoming("key1", rule.clone(), 1, true).await;
            assert!(result.is_ok());
        }

        // key1 should be rate limited
        let result = limiter.incoming("key1", rule.clone(), 1, true).await;
        assert!(result.is_err());

        // key2 should still work
        let result = limiter.incoming("key2", rule, 1, true).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().remaining, 4);
    }

    /// Test LocalRateLimiter per-minute window duration
    /// Verifies that the window duration is correctly set to 60 seconds for per-minute limits
    #[tokio::test]
    async fn test_local_rate_limiter_per_minute_window() {
        let limiter = LocalRateLimiter;
        let rule = RateLimitRule::new(10, 60); // 60 seconds window

        let result = limiter.incoming("test_key_5", rule, 1, true).await;
        assert!(result.is_ok());

        let info = result.unwrap();
        let window_duration = info.reset_at.duration_since(info.window_start);
        // Window should be approximately 60 seconds
        assert!(window_duration.as_secs() >= 59 && window_duration.as_secs() <= 61);
    }

    /// Test LocalRateLimiter per-day window duration
    /// Verifies that the window duration is correctly set to 86400 seconds for per-day limits
    #[tokio::test]
    async fn test_local_rate_limiter_per_day_window() {
        let limiter = LocalRateLimiter;
        let rule = RateLimitRule::new(1000, 86400); // 86400 seconds (1 day) window

        let result = limiter.incoming("test_key_6", rule, 1, true).await;
        assert!(result.is_ok());

        let info = result.unwrap();
        let window_duration = info.reset_at.duration_since(info.window_start);
        // Window should be approximately 86400 seconds
        assert!(window_duration.as_secs() >= 86399 && window_duration.as_secs() <= 86401);
    }

    // --- Integration tests: run_check with in-memory limiter ---

    #[tokio::test]
    async fn test_pre_check_commits_request_metrics() {
        let e = make_entity("pre_req_1", Some(10), None);
        assert_eq!(
            run_check(&e, CheckPhase::Pre).await.unwrap()[0].1.remaining,
            9
        );
        assert_eq!(
            run_check(&e, CheckPhase::Pre).await.unwrap()[0].1.remaining,
            8
        );
    }

    #[tokio::test]
    async fn test_pre_check_does_not_commit_token_metrics() {
        let e = make_entity("pre_tok_1", None, Some(1000));
        assert_eq!(
            run_check(&e, CheckPhase::Pre).await.unwrap()[0].1.remaining,
            1000
        );
        assert_eq!(
            run_check(&e, CheckPhase::Pre).await.unwrap()[0].1.remaining,
            1000
        );
    }

    #[tokio::test]
    async fn test_post_check_commits_token_metrics() {
        let e = make_entity("post_tok_1", None, Some(1000));
        assert_eq!(
            run_check(&e, CheckPhase::Post(100)).await.unwrap()[0]
                .1
                .remaining,
            900
        );
        assert_eq!(
            run_check(&e, CheckPhase::Post(50)).await.unwrap()[0]
                .1
                .remaining,
            850
        );
    }

    #[tokio::test]
    async fn test_post_check_skips_request_metrics() {
        let e = make_entity("post_req_1", Some(10), None);
        assert_eq!(run_check(&e, CheckPhase::Post(1)).await.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_full_flow_with_state() {
        let e = make_entity("flow_1", Some(10), Some(1000));
        let mut state = RateLimitState::new();
        state.store_pre_check(run_check(&e, CheckPhase::Pre).await.unwrap());
        assert_eq!(state.request_info.as_ref().unwrap().remaining, 9);
        assert_eq!(state.token_info.as_ref().unwrap().remaining, 1000);
        state.store_post_check(run_check(&e, CheckPhase::Post(150)).await.unwrap());
        assert_eq!(state.request_info.as_ref().unwrap().remaining, 9);
        assert_eq!(state.token_info.as_ref().unwrap().remaining, 850);
    }

    #[tokio::test]
    async fn test_pre_check_exceeded() {
        let e = make_entity("exc_1", Some(3), None);
        for _ in 0..3 {
            assert!(run_check(&e, CheckPhase::Pre).await.is_ok());
        }
        let (m, err) = run_check(&e, CheckPhase::Pre).await.unwrap_err();
        assert!(matches!(m, RateLimitMetric::RPM));
        assert!(matches!(err, RateLimitError::Exceeded(_)));
    }

    #[tokio::test]
    async fn test_post_check_exceeded() {
        let e = make_entity("exc_2", None, Some(100));
        assert!(run_check(&e, CheckPhase::Post(90)).await.is_ok());
        let (m, err) = run_check(&e, CheckPhase::Post(20)).await.unwrap_err();
        assert!(matches!(m, RateLimitMetric::TPM));
        assert!(matches!(err, RateLimitError::Exceeded(_)));
    }

    #[tokio::test]
    async fn test_no_rate_limit() {
        let e = MockEntity {
            id: "none".into(),
            rate_limit: None,
        };
        assert_eq!(run_check(&e, CheckPhase::Pre).await.unwrap().len(), 0);
        assert_eq!(run_check(&e, CheckPhase::Post(100)).await.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_entities_isolated() {
        let e1 = make_entity("iso_1", Some(5), None);
        let e2 = make_entity("iso_2", Some(5), None);
        for _ in 0..5 {
            assert!(run_check(&e1, CheckPhase::Pre).await.is_ok());
        }
        assert!(run_check(&e1, CheckPhase::Pre).await.is_err());
        assert_eq!(
            run_check(&e2, CheckPhase::Pre).await.unwrap()[0]
                .1
                .remaining,
            4
        );
    }

    #[tokio::test]
    async fn test_headers_generated() {
        let e = make_entity("hdr_1", Some(60), Some(1000));
        let mut state = RateLimitState::new();
        state.store_pre_check(run_check(&e, CheckPhase::Pre).await.unwrap());
        state.store_post_check(run_check(&e, CheckPhase::Post(100)).await.unwrap());
        let mut headers = HeaderMap::new();
        state.add_headers(&mut headers);
        assert_eq!(headers.get("x-ratelimit-limit-requests").unwrap(), "60");
        assert_eq!(headers.get("x-ratelimit-remaining-requests").unwrap(), "59");
        assert_eq!(headers.get("x-ratelimit-limit-tokens").unwrap(), "1000");
        assert_eq!(headers.get("x-ratelimit-remaining-tokens").unwrap(), "900");
    }
}
