use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::OwnedSemaphorePermit;

use crate::config::entities::types::RateLimit;

mod local;
mod window;

/// Rate limiter error types
#[derive(Debug, Clone)]
pub enum RateLimitError {
    RequestPerMinuteExceeded {
        scope: String,
        id: String,
        current: u64,
        limit: u64,
    },
    RequestPerDayExceeded {
        scope: String,
        id: String,
        current: u64,
        limit: u64,
    },
    TokenPerMinuteExceeded {
        scope: String,
        id: String,
        current: u64,
        limit: u64,
    },
    TokenPerDayExceeded {
        scope: String,
        id: String,
        current: u64,
        limit: u64,
    },
    ConcurrencyExceeded {
        scope: String,
        id: String,
        limit: u64,
    },
}

impl std::fmt::Display for RateLimitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RateLimitError::RequestPerMinuteExceeded {
                scope,
                id,
                current,
                limit,
            } => {
                write!(
                    f,
                    "Request per minute limit exceeded for {}:{} (current: {}, limit: {})",
                    scope, id, current, limit
                )
            }
            RateLimitError::RequestPerDayExceeded {
                scope,
                id,
                current,
                limit,
            } => {
                write!(
                    f,
                    "Request per day limit exceeded for {}:{} (current: {}, limit: {})",
                    scope, id, current, limit
                )
            }
            RateLimitError::TokenPerMinuteExceeded {
                scope,
                id,
                current,
                limit,
            } => {
                write!(
                    f,
                    "Token per minute limit exceeded for {}:{} (current: {}, limit: {})",
                    scope, id, current, limit
                )
            }
            RateLimitError::TokenPerDayExceeded {
                scope,
                id,
                current,
                limit,
            } => {
                write!(
                    f,
                    "Token per day limit exceeded for {}:{} (current: {}, limit: {})",
                    scope, id, current, limit
                )
            }
            RateLimitError::ConcurrencyExceeded { scope, id, limit } => {
                write!(
                    f,
                    "Concurrency limit exceeded for {}:{} (limit: {})",
                    scope, id, limit
                )
            }
        }
    }
}

impl std::error::Error for RateLimitError {}

/// Guard that releases concurrency permit on drop
#[derive(Debug)]
pub struct ConcurrencyGuard {
    #[allow(dead_code)]
    permit: Option<OwnedSemaphorePermit>,
}

impl Drop for ConcurrencyGuard {
    fn drop(&mut self) {
        // Permit is automatically released when dropped
    }
}

/// Rate limiter trait for checking and recording usage
#[async_trait]
pub trait RateLimiter: Send + Sync {
    /// Check if request can proceed and reserve resources
    /// Returns a concurrency guard that must be held until request completes
    async fn check_and_reserve(
        &self,
        scope: &str,
        id: &str,
        rate_limit: &RateLimit,
    ) -> Result<Option<ConcurrencyGuard>, RateLimitError>;

    /// Record token usage after request completes
    async fn record_usage(
        &self,
        scope: &str,
        id: &str,
        rate_limit: &RateLimit,
        tokens: u64,
    ) -> Result<(), RateLimitError>;
}

/// Global rate limiter instance
static RATE_LIMITER: std::sync::OnceLock<Arc<dyn RateLimiter + Send + Sync>> =
    std::sync::OnceLock::new();

/// Initialize the global rate limiter
pub fn init_rate_limiter() {
    RATE_LIMITER
        .set(Arc::new(local::LocalRateLimiter::new()))
        .ok()
        .expect("Rate limiter already initialized");
}

/// Get the global rate limiter instance
pub fn get_rate_limiter() -> &'static Arc<dyn RateLimiter + Send + Sync> {
    RATE_LIMITER.get().expect("Rate limiter not initialized")
}
