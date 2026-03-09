mod local;
pub mod utils;

use std::sync::{Arc, LazyLock};

use async_trait::async_trait;
use thiserror::Error;

/// Error types for concurrency limiting
#[derive(Debug, Clone, Error)]
pub enum ConcurrencyError {
    #[error("Concurrency limit exceeded: limit={limit}, current={current}")]
    Exceeded { limit: u64, current: u64 },
    #[error("Internal error: {0}")]
    #[allow(dead_code)]
    Internal(String),
}

/// Snapshot of concurrency state for a single dimension
#[derive(Debug, Clone)]
pub struct ConcurrencyInfo {
    pub limit: u64,
    pub current: u64,
}

impl ConcurrencyInfo {
    pub fn remaining(&self) -> u64 {
        self.limit.saturating_sub(self.current)
    }
}

/// Trait for RAII concurrency guard.
/// Dropping the guard releases the concurrency slot.
pub trait ConcurrencyGuard: Send + Sync {}

/// A concurrency permit that holds a slot via RAII.
/// When this value is dropped, the underlying guard releases the slot.
#[derive(Clone)]
pub struct ConcurrencyPermit {
    pub info: ConcurrencyInfo,
    _guard: Arc<dyn ConcurrencyGuard>,
}

impl ConcurrencyPermit {
    pub fn new(info: ConcurrencyInfo, guard: impl ConcurrencyGuard + 'static) -> Self {
        Self {
            info,
            _guard: Arc::new(guard),
        }
    }
}

/// Container for concurrency permits stored in HookContext.
/// Permits are released (slots freed) when this value is dropped.
#[derive(Clone)]
pub struct ConcurrencyPermits(#[allow(dead_code)] pub Vec<ConcurrencyPermit>);

/// Trait for concurrency limiter backends.
/// Implementations must be thread-safe and support RAII-based slot release.
#[async_trait]
pub trait ConcurrencyLimiter: Send + Sync {
    /// Try to acquire a concurrency slot.
    ///
    /// * `key` — unique identifier for the concurrency scope (e.g. `model:xxx:concurrency`)
    /// * `max_concurrent` — maximum allowed concurrent requests (read from current config)
    ///
    /// Returns a `ConcurrencyPermit` on success. The slot is held until
    /// the permit (and its inner guard) is dropped.
    async fn try_acquire(
        &self,
        key: &str,
        max_concurrent: u64,
    ) -> Result<ConcurrencyPermit, ConcurrencyError>;
}

static CONCURRENCY_LIMITER: LazyLock<Arc<dyn ConcurrencyLimiter>> =
    LazyLock::new(|| Arc::new(local::LocalConcurrencyLimiter::new()));

pub fn get_concurrency_limiter() -> Arc<dyn ConcurrencyLimiter> {
    CONCURRENCY_LIMITER.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concurrency_info_remaining() {
        let info = ConcurrencyInfo {
            limit: 10,
            current: 3,
        };
        assert_eq!(info.remaining(), 7);
    }

    #[test]
    fn test_concurrency_info_remaining_at_limit() {
        let info = ConcurrencyInfo {
            limit: 5,
            current: 5,
        };
        assert_eq!(info.remaining(), 0);
    }

    #[test]
    fn test_concurrency_info_remaining_saturates() {
        let info = ConcurrencyInfo {
            limit: 3,
            current: 10,
        };
        assert_eq!(info.remaining(), 0);
    }

    #[test]
    fn test_concurrency_error_display() {
        let err = ConcurrencyError::Exceeded {
            limit: 5,
            current: 5,
        };
        assert_eq!(
            err.to_string(),
            "Concurrency limit exceeded: limit=5, current=5"
        );

        let err = ConcurrencyError::Internal("something broke".into());
        assert_eq!(err.to_string(), "Internal error: something broke");
    }
}
