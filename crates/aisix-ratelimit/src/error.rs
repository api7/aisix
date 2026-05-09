//! Limiter error taxonomy.
//!
//! Uses [`aisix_core::RateLimitScope`] so the proxy layer can plug the
//! error straight into its OpenAI-style envelope without translation.

use aisix_core::RateLimitScope;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RateLimitError {
    #[error("request limit exceeded ({scope})")]
    Requests {
        scope: RateLimitScope,
        retry_after_secs: u64,
    },
    #[error("token limit exceeded ({scope})")]
    Tokens {
        scope: RateLimitScope,
        retry_after_secs: u64,
    },
    #[error("concurrency limit exceeded")]
    Concurrency,
}

impl RateLimitError {
    pub fn scope(&self) -> RateLimitScope {
        match self {
            RateLimitError::Requests { scope, .. } => *scope,
            RateLimitError::Tokens { scope, .. } => *scope,
            RateLimitError::Concurrency => RateLimitScope::Requests,
        }
    }

    pub fn retry_after_secs(&self) -> Option<u64> {
        match self {
            RateLimitError::Requests {
                retry_after_secs, ..
            }
            | RateLimitError::Tokens {
                retry_after_secs, ..
            } => Some(*retry_after_secs),
            RateLimitError::Concurrency => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_scope_preserved_on_access() {
        let e = RateLimitError::Requests {
            scope: RateLimitScope::Requests,
            retry_after_secs: 42,
        };
        assert_eq!(e.scope(), RateLimitScope::Requests);
        assert_eq!(e.retry_after_secs(), Some(42));
    }

    #[test]
    fn concurrency_has_no_retry_after_hint() {
        let e = RateLimitError::Concurrency;
        assert_eq!(e.scope(), RateLimitScope::Requests);
        assert!(e.retry_after_secs().is_none());
    }
}
