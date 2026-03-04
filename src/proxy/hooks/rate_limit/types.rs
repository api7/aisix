use std::{
    sync::{Arc, LazyLock},
    time::{Duration, Instant},
};

use anyhow::Result;
use async_trait::async_trait;
use skp_ratelimit::Quota;
use thiserror::Error;

use super::local;
use crate::config::entities::types::{RateLimit as RateLimitConfig, RateLimitMetric};

/// Rate limit error types
#[derive(Debug, Clone, Error)]
pub enum RateLimitError {
    #[error("Rate limit exceeded")]
    Exceeded(RateLimitInfo),
    #[error("Internal error: {0}")]
    Internal(String),
}

/// A single rate limit rule: (limit, window)
#[derive(Clone)]
pub struct RateLimitRule {
    pub(super) limit: u64,
    pub(super) window_secs: u64,
}

impl RateLimitRule {
    pub fn new(limit: u64, window_secs: u64) -> Self {
        Self { limit, window_secs }
    }
}

impl From<RateLimitRule> for Quota {
    fn from(rule: RateLimitRule) -> Self {
        Quota::new(rule.limit, Duration::from_secs(rule.window_secs))
    }
}

/// Convert a `RateLimitConfig` into a list of (metric, rule) pairs
impl From<RateLimitConfig> for Vec<(RateLimitMetric, RateLimitRule)> {
    fn from(config: RateLimitConfig) -> Self {
        let mut rules = Vec::new();
        if let Some(n) = config.token_per_minute {
            rules.push((RateLimitMetric::TPM, RateLimitRule::new(n, 60)));
        }
        if let Some(n) = config.token_per_day {
            rules.push((RateLimitMetric::TPD, RateLimitRule::new(n, 86400)));
        }
        if let Some(n) = config.request_per_minute {
            rules.push((RateLimitMetric::RPM, RateLimitRule::new(n, 60)));
        }
        if let Some(n) = config.request_per_day {
            rules.push((RateLimitMetric::RPD, RateLimitRule::new(n, 86400)));
        }
        rules
    }
}

/// Snapshot of rate limit state for a single metric dimension
#[allow(unused)]
#[derive(Debug, Clone)]
pub struct RateLimitInfo {
    pub limit: u64,
    pub remaining: u64,
    pub reset_at: Instant,
    pub window_start: Instant,
    pub retry_after: Option<Duration>,
}

pub type RateLimitResult = Result<RateLimitInfo, RateLimitError>;

/// Trait for rate limiter backends
#[async_trait]
pub trait RateLimiter: Send + Sync {
    /// Check and optionally increment a rate limit counter.
    ///
    /// * `commit = false` — read-only check, counter unchanged
    /// * `commit = true`  — consume `cost` units from the quota
    async fn incoming(
        &self,
        key: &str,
        rule: RateLimitRule,
        cost: u64,
        commit: bool,
    ) -> RateLimitResult;
}

static RATE_LIMITER: LazyLock<Arc<dyn RateLimiter>> =
    LazyLock::new(|| Arc::new(local::LocalRateLimiter));

pub fn get_rate_limiter() -> Arc<dyn RateLimiter> {
    RATE_LIMITER.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::entities::types::RateLimit;

    #[test]
    fn test_rate_limit_rule_new() {
        let rule = RateLimitRule::new(100, 60);
        assert_eq!(rule.limit, 100);
        assert_eq!(rule.window_secs, 60);
    }

    #[test]
    fn test_rate_limit_rule_to_quota() {
        let rule = RateLimitRule::new(100, 60);
        let _quota: Quota = rule.into();
    }

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
        assert_eq!(
            rules
                .iter()
                .find(|(m, _)| matches!(m, RateLimitMetric::TPM))
                .unwrap()
                .1
                .limit,
            1000
        );
        assert_eq!(
            rules
                .iter()
                .find(|(m, _)| matches!(m, RateLimitMetric::TPD))
                .unwrap()
                .1
                .limit,
            100000
        );
        assert_eq!(
            rules
                .iter()
                .find(|(m, _)| matches!(m, RateLimitMetric::RPM))
                .unwrap()
                .1
                .limit,
            60
        );
        assert_eq!(
            rules
                .iter()
                .find(|(m, _)| matches!(m, RateLimitMetric::RPD))
                .unwrap()
                .1
                .limit,
            5000
        );
    }

    #[test]
    fn test_rate_limit_to_rules_empty() {
        let rate_limit = RateLimit {
            token_per_minute: None,
            token_per_day: None,
            request_per_minute: None,
            request_per_day: None,
            request_concurrency: None,
        };
        assert_eq!(
            <RateLimit as Into<Vec<(RateLimitMetric, RateLimitRule)>>>::into(rate_limit).len(),
            0
        );
    }

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
        assert_eq!(
            RateLimitError::Exceeded(info).to_string(),
            "Rate limit exceeded"
        );
        assert_eq!(
            RateLimitError::Internal("e".into()).to_string(),
            "Internal error: e"
        );
    }
}
