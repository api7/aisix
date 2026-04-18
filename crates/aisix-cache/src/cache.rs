//! [`Cache`] trait — the storage seam every backend implements against.
//!
//! Returning `Result<Option<…>, _>` rather than `Result<…, NotFound>`
//! makes the call site read naturally: cache miss is an expected control
//! flow, not an error.
//!
//! Held behind `Arc<dyn Cache>` in `ProxyState`. Trait objects need
//! `async_trait` until native async-fn-in-traits become dyn-compatible.

use aisix_gateway::ChatResponse;
use async_trait::async_trait;

#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    #[error("cache backend error: {0}")]
    Backend(String),
}

/// Outcome of a cache lookup. Public so the proxy can attach the
/// `x-aisix-cache: hit|miss` header without owning string literals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheOutcome {
    Hit,
    Miss,
}

impl CacheOutcome {
    pub fn as_header_value(self) -> &'static str {
        match self {
            CacheOutcome::Hit => "hit",
            CacheOutcome::Miss => "miss",
        }
    }
}

#[async_trait]
pub trait Cache: Send + Sync + 'static {
    async fn get(&self, key: &str) -> Result<Option<ChatResponse>, CacheError>;
    async fn put(&self, key: &str, value: ChatResponse) -> Result<(), CacheError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_outcome_emits_canonical_header_string() {
        assert_eq!(CacheOutcome::Hit.as_header_value(), "hit");
        assert_eq!(CacheOutcome::Miss.as_header_value(), "miss");
    }
}
