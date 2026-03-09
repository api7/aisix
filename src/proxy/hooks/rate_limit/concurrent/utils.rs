use axum::{Json, response::IntoResponse};
use http::{HeaderMap, HeaderValue, StatusCode};
use log::error;

use super::{ConcurrencyError, ConcurrencyInfo, ConcurrencyPermit, get_concurrency_limiter};
use crate::config::entities::types::{HasRateLimit, RateLimitMetric};

fn insert_header(headers: &mut HeaderMap, name: &'static str, value: String) {
    headers.insert(name, HeaderValue::from_str(&value).unwrap());
}

/// Storage for concurrency limit information, used for response headers.
/// When both apikey and model have concurrency limits, keeps the stricter one.
#[derive(Debug, Clone)]
pub struct ConcurrencyState {
    pub info: Option<ConcurrencyInfo>,
}

impl ConcurrencyState {
    pub fn new() -> Self {
        Self { info: None }
    }

    /// Store concurrency info, keeping the stricter (lower remaining) one
    pub fn store_check(&mut self, info: ConcurrencyInfo) {
        self.info = Some(match self.info.take() {
            None => info,
            Some(existing) => {
                if info.remaining() < existing.remaining() {
                    info
                } else {
                    existing
                }
            }
        });
    }

    /// Add concurrency limit headers to response
    pub fn add_headers(&self, headers: &mut HeaderMap) {
        if let Some(ref info) = self.info {
            insert_header(
                headers,
                "x-ratelimit-limit-concurrent",
                info.limit.to_string(),
            );
            insert_header(
                headers,
                "x-ratelimit-remaining-concurrent",
                info.remaining().to_string(),
            );
        }
    }
}

/// Run concurrency check for an entity.
/// Returns `None` if the entity has no concurrency limit configured.
/// Returns `Some(Ok(permit))` if acquired, or `Some(Err(error))` if exceeded.
pub async fn run_concurrency_check<T: HasRateLimit>(
    entity: &T,
) -> Option<Result<ConcurrencyPermit, ConcurrencyError>> {
    let rate_limit = entity.rate_limit()?;
    let max_concurrent = rate_limit.request_concurrency?;
    let key = entity.rate_limit_key(RateLimitMetric::Concurrency);
    let limiter = get_concurrency_limiter();
    Some(limiter.try_acquire(&key, max_concurrent).await)
}

/// HTTP error response for concurrency limit exceeded
pub struct ConcurrencyLimitResponse {
    id: String,
    error: ConcurrencyError,
}

impl ConcurrencyLimitResponse {
    pub fn new(id: String, error: ConcurrencyError) -> Self {
        Self { id, error }
    }
}

impl IntoResponse for ConcurrencyLimitResponse {
    fn into_response(self) -> axum::response::Response {
        match self.error {
            ConcurrencyError::Exceeded { limit, current } => (
                StatusCode::TOO_MANY_REQUESTS,
                Json(serde_json::json!({
                    "error": {
                        "message": format!(
                            "Concurrency limit exceeded for resource ID: {}. Limit: {}, current: {}",
                            self.id, limit, current
                        ),
                        "type": "rate_limit_error",
                        "code": "concurrency_limit_exceeded"
                    }
                })),
            )
                .into_response(),
            ConcurrencyError::Internal(msg) => {
                error!("Concurrency limit internal error: {}", msg);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": {
                            "message": "Internal server error",
                            "type": "internal_error",
                            "code": "internal_error"
                        }
                    })),
                )
                    .into_response()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concurrency_state_new() {
        let state = ConcurrencyState::new();
        assert!(state.info.is_none());
    }

    #[test]
    fn test_concurrency_state_store_single() {
        let mut state = ConcurrencyState::new();
        state.store_check(ConcurrencyInfo {
            limit: 10,
            current: 3,
        });
        assert_eq!(state.info.as_ref().unwrap().limit, 10);
        assert_eq!(state.info.as_ref().unwrap().current, 3);
    }

    #[test]
    fn test_concurrency_state_keeps_stricter() {
        let mut state = ConcurrencyState::new();

        // First: 10 limit, 3 current → remaining=7
        state.store_check(ConcurrencyInfo {
            limit: 10,
            current: 3,
        });

        // Second: 5 limit, 4 current → remaining=1 (stricter)
        state.store_check(ConcurrencyInfo {
            limit: 5,
            current: 4,
        });

        assert_eq!(state.info.as_ref().unwrap().limit, 5);
        assert_eq!(state.info.as_ref().unwrap().current, 4);
    }

    #[test]
    fn test_concurrency_state_keeps_existing_when_less_strict() {
        let mut state = ConcurrencyState::new();

        // First: 5 limit, 4 current → remaining=1 (stricter)
        state.store_check(ConcurrencyInfo {
            limit: 5,
            current: 4,
        });

        // Second: 100 limit, 2 current → remaining=98 (less strict)
        state.store_check(ConcurrencyInfo {
            limit: 100,
            current: 2,
        });

        assert_eq!(state.info.as_ref().unwrap().limit, 5);
        assert_eq!(state.info.as_ref().unwrap().current, 4);
    }

    #[test]
    fn test_concurrency_state_add_headers() {
        let mut state = ConcurrencyState::new();
        state.store_check(ConcurrencyInfo {
            limit: 10,
            current: 3,
        });

        let mut headers = HeaderMap::new();
        state.add_headers(&mut headers);

        assert_eq!(headers.get("x-ratelimit-limit-concurrent").unwrap(), "10");
        assert_eq!(
            headers.get("x-ratelimit-remaining-concurrent").unwrap(),
            "7"
        );
    }

    #[test]
    fn test_concurrency_state_add_headers_empty() {
        let state = ConcurrencyState::new();
        let mut headers = HeaderMap::new();
        state.add_headers(&mut headers);

        assert!(headers.get("x-ratelimit-limit-concurrent").is_none());
        assert!(headers.get("x-ratelimit-remaining-concurrent").is_none());
    }
}
