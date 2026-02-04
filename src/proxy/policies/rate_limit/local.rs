//! Local in-memory rate limiter implementation

use std::sync::OnceLock;

use async_trait::async_trait;
use skp_ratelimit::Algorithm;

use super::{RateLimitError, RateLimitInfo, RateLimitRule, RateLimiter};

static FIXED_WINDOW: OnceLock<skp_ratelimit::FixedWindow> = OnceLock::new();
static MEMORY_STORAGE: OnceLock<skp_ratelimit::MemoryStorage> = OnceLock::new();

fn fixed_window() -> &'static skp_ratelimit::FixedWindow {
    FIXED_WINDOW.get_or_init(skp_ratelimit::FixedWindow::new)
}
fn memory_storage() -> &'static skp_ratelimit::MemoryStorage {
    MEMORY_STORAGE.get_or_init(skp_ratelimit::MemoryStorage::new)
}

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
            fixed_window()
                .check_and_record(memory_storage(), key, &rule.into(), cost)
                .await
        } else {
            fixed_window()
                .check(memory_storage(), key, &rule.into())
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
    use super::*;
    use crate::proxy::policies::rate_limit::RateLimitRule;

    /// Test that LocalRateLimiter allows requests within the limit
    /// Verifies that a request within quota succeeds and decrements the remaining count
    #[tokio::test]
    async fn test_local_rate_limiter_allows_within_limit() {
        let limiter = LocalRateLimiter::default();
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
        let limiter = LocalRateLimiter::default();
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
        let limiter = LocalRateLimiter::default();
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
        let limiter = LocalRateLimiter::default();
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
        let limiter = LocalRateLimiter::default();
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
        let limiter = LocalRateLimiter::default();
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
        let limiter = LocalRateLimiter::default();
        let rule = RateLimitRule::new(1000, 86400); // 86400 seconds (1 day) window

        let result = limiter.incoming("test_key_6", rule, 1, true).await;
        assert!(result.is_ok());

        let info = result.unwrap();
        let window_duration = info.reset_at.duration_since(info.window_start);
        // Window should be approximately 86400 seconds
        assert!(window_duration.as_secs() >= 86399 && window_duration.as_secs() <= 86401);
    }
}
