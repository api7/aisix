use std::time::Instant;

use axum::{Json, response::IntoResponse};
use http::{HeaderMap, HeaderValue, StatusCode};
use log::error;

use super::{RateLimitError, RateLimitInfo, RateLimitRule, get_rate_limiter};
use crate::config::entities::types::{HasRateLimit, RateLimitMetric};

/// Helper to convert duration to human-readable format (e.g., "1s", "6m0s")
fn format_duration(duration: std::time::Duration) -> String {
    let secs = duration.as_secs();
    if secs < 60 {
        format!("{}s", secs)
    } else {
        let mins = secs / 60;
        let remaining_secs = secs % 60;
        if remaining_secs == 0 {
            format!("{}m", mins)
        } else {
            format!("{}m{}s", mins, remaining_secs)
        }
    }
}

fn insert_header(headers: &mut HeaderMap, name: &'static str, value: String) {
    headers.insert(name, HeaderValue::from_str(&value).unwrap());
}

/// Storage for rate limit information from pre_check and post_check
#[derive(Debug, Clone)]
pub struct RateLimitState {
    pub request_info: Option<RateLimitInfo>,
    pub token_info: Option<RateLimitInfo>,
}

impl RateLimitState {
    pub fn new() -> Self {
        Self {
            request_info: None,
            token_info: None,
        }
    }

    /// Choose the stricter rate limit info between existing and new
    /// Prioritizes lower remaining count, then earlier reset time
    fn choose_stricter(existing: Option<RateLimitInfo>, new: RateLimitInfo) -> RateLimitInfo {
        match existing {
            None => new,
            Some(existing) => {
                // Compare remaining count - lower is stricter
                if new.remaining < existing.remaining {
                    new
                } else if new.remaining > existing.remaining {
                    existing
                } else {
                    // Same remaining count - choose earlier reset time (stricter)
                    if new.reset_at < existing.reset_at {
                        new
                    } else {
                        existing
                    }
                }
            }
        }
    }

    /// Store rate limit info from pre_check results
    /// When multiple metrics exist (RPM+RPD or TPM+TPD), keeps the stricter one
    pub fn store_pre_check(&mut self, results: Vec<(RateLimitMetric, RateLimitInfo)>) {
        for (metric, info) in results {
            match metric {
                RateLimitMetric::RPM | RateLimitMetric::RPD => {
                    self.request_info =
                        Some(Self::choose_stricter(self.request_info.clone(), info));
                }
                RateLimitMetric::TPM | RateLimitMetric::TPD => {
                    self.token_info = Some(Self::choose_stricter(self.token_info.clone(), info));
                }
                RateLimitMetric::Concurrency => {}
            }
        }
    }

    /// Store rate limit info from post_check results (updates token info)
    /// When multiple metrics exist (TPM+TPD), keeps the stricter one
    pub fn store_post_check(&mut self, results: Vec<(RateLimitMetric, RateLimitInfo)>) {
        for (metric, info) in results {
            match metric {
                RateLimitMetric::TPM | RateLimitMetric::TPD => {
                    self.token_info = Some(Self::choose_stricter(self.token_info.clone(), info));
                }
                _ => {}
            }
        }
    }

    /// Add rate limit headers to response
    pub fn add_headers(&self, headers: &mut HeaderMap) {
        let now = Instant::now();

        if let Some(ref req_info) = self.request_info {
            insert_header(
                headers,
                "x-ratelimit-limit-requests",
                req_info.limit.to_string(),
            );
            insert_header(
                headers,
                "x-ratelimit-remaining-requests",
                req_info.remaining.to_string(),
            );
            insert_header(
                headers,
                "x-ratelimit-reset-requests",
                format_duration(req_info.reset_at.saturating_duration_since(now)),
            );
        }

        if let Some(ref token_info) = self.token_info {
            insert_header(
                headers,
                "x-ratelimit-limit-tokens",
                token_info.limit.to_string(),
            );
            insert_header(
                headers,
                "x-ratelimit-remaining-tokens",
                token_info.remaining.to_string(),
            );
            insert_header(
                headers,
                "x-ratelimit-reset-tokens",
                format_duration(token_info.reset_at.saturating_duration_since(now)),
            );
        }
    }
}

#[derive(Copy, Clone)]
pub enum CheckPhase {
    Pre,
    Post(u64),
}

impl CheckPhase {
    fn should_skip(self, metric: &RateLimitMetric) -> bool {
        matches!(metric, RateLimitMetric::Concurrency)
            || (matches!(self, Self::Post(_))
                && matches!(metric, RateLimitMetric::RPM | RateLimitMetric::RPD))
    }
    fn commit(self, metric: &RateLimitMetric) -> bool {
        match self {
            Self::Pre => matches!(metric, RateLimitMetric::RPM | RateLimitMetric::RPD),
            Self::Post(_) => matches!(metric, RateLimitMetric::TPM | RateLimitMetric::TPD),
        }
    }
    fn cost(self) -> u64 {
        match self {
            Self::Pre => 1,
            Self::Post(cost) => cost,
        }
    }
}

pub async fn run_check<T: HasRateLimit>(
    entity: &T,
    phase: CheckPhase,
) -> Result<Vec<(RateLimitMetric, RateLimitInfo)>, (RateLimitMetric, RateLimitError)> {
    let Some(rate_limit) = entity.rate_limit() else {
        return Ok(Vec::new());
    };
    let limiter = get_rate_limiter();
    let rules: Vec<(RateLimitMetric, RateLimitRule)> = rate_limit.into();
    let mut results = Vec::new();
    for (metric, rule) in rules {
        if phase.should_skip(&metric) {
            continue;
        }
        let key = entity.rate_limit_key(metric.clone());

        match limiter
            .incoming(&key, rule, phase.cost(), phase.commit(&metric))
            .await
        {
            Ok(info) => results.push((metric, info)),
            Err(e) => return Err((metric, e)),
        }
    }
    Ok(results)
}

pub struct RateLimitResponse(String, RateLimitMetric, RateLimitError);

impl RateLimitResponse {
    /// Create a new rate limit response
    ///
    /// # Arguments
    /// * `api_key_id` - The API key resource ID (not the key itself, for security)
    /// * `metric` - The rate limit metric that was exceeded
    /// * `error` - The rate limit error details
    pub fn new(api_key_id: String, metric: RateLimitMetric, error: RateLimitError) -> Self {
        Self(api_key_id, metric, error)
    }
}

impl IntoResponse for RateLimitResponse {
    fn into_response(self) -> axum::response::Response {
        match self.2 {
            RateLimitError::Exceeded(info) => {
                let mut response = (
                    StatusCode::TOO_MANY_REQUESTS,
                    Json(serde_json::json!({
                        "error": {
                            "message": format!(
                                "Rate limit exceeded for API key ID: {}. Limited on {}, current limit: {}, remaining: {}",
                                self.0,
                                 self.1,
                                  info.limit,
                                   info.remaining
                                 ),
                            "type": "rate_limit_error",
                            "code": "rate_limit_exceeded"
                        }
                    })),
                )
                    .into_response();

                // Only add Retry-After header for 429 responses
                let now = Instant::now();
                let retry_after_secs = info.reset_at.saturating_duration_since(now).as_secs();
                response.headers_mut().insert(
                    http::header::RETRY_AFTER,
                    HeaderValue::from_str(&retry_after_secs.to_string()).unwrap(),
                );
                response
            }
            RateLimitError::Internal(msg) => {
                error!("Rate limit internal error: {}", msg);
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
    use std::time::Duration;

    use super::*;

    /// Test format_duration function with seconds only
    /// Verifies that durations less than 60 seconds are formatted as "Xs"
    #[test]
    fn test_format_duration_seconds_only() {
        assert_eq!(format_duration(Duration::from_secs(0)), "0s");
        assert_eq!(format_duration(Duration::from_secs(1)), "1s");
        assert_eq!(format_duration(Duration::from_secs(30)), "30s");
        assert_eq!(format_duration(Duration::from_secs(59)), "59s");
    }

    /// Test format_duration function with minutes only
    /// Verifies that whole minutes are formatted as "Xm" without seconds
    #[test]
    fn test_format_duration_minutes_only() {
        assert_eq!(format_duration(Duration::from_secs(60)), "1m");
        assert_eq!(format_duration(Duration::from_secs(120)), "2m");
        assert_eq!(format_duration(Duration::from_secs(3600)), "60m");
    }

    /// Test format_duration function with mixed minutes and seconds
    /// Verifies that non-whole minutes are formatted as "XmYs"
    #[test]
    fn test_format_duration_minutes_and_seconds() {
        assert_eq!(format_duration(Duration::from_secs(61)), "1m1s");
        assert_eq!(format_duration(Duration::from_secs(90)), "1m30s");
        assert_eq!(format_duration(Duration::from_secs(125)), "2m5s");
        assert_eq!(format_duration(Duration::from_secs(360)), "6m");
    }

    /// Test RateLimitState::new() creates empty state
    /// Verifies that both request_info and token_info are None initially
    #[test]
    fn test_rate_limit_state_new() {
        let state = RateLimitState::new();
        assert!(state.request_info.is_none());
        assert!(state.token_info.is_none());
    }

    /// Test RateLimitState::store_pre_check() stores request metrics
    /// Verifies that RPM/RPD metrics are stored in request_info
    #[test]
    fn test_rate_limit_state_store_pre_check_request_metrics() {
        let mut state = RateLimitState::new();
        let now = Instant::now();

        let info = RateLimitInfo {
            limit: 100,
            remaining: 99,
            reset_at: now + Duration::from_secs(60),
            window_start: now,
            retry_after: None,
        };

        state.store_pre_check(vec![(RateLimitMetric::RPM, info.clone())]);

        assert!(state.request_info.is_some());
        assert_eq!(state.request_info.as_ref().unwrap().limit, 100);
        assert_eq!(state.request_info.as_ref().unwrap().remaining, 99);
        assert!(state.token_info.is_none());
    }

    /// Test RateLimitState::store_pre_check() stores token metrics
    /// Verifies that TPM/TPD metrics are stored in token_info
    #[test]
    fn test_rate_limit_state_store_pre_check_token_metrics() {
        let mut state = RateLimitState::new();
        let now = Instant::now();

        let info = RateLimitInfo {
            limit: 1000,
            remaining: 999,
            reset_at: now + Duration::from_secs(60),
            window_start: now,
            retry_after: None,
        };

        state.store_pre_check(vec![(RateLimitMetric::TPM, info.clone())]);

        assert!(state.token_info.is_some());
        assert_eq!(state.token_info.as_ref().unwrap().limit, 1000);
        assert_eq!(state.token_info.as_ref().unwrap().remaining, 999);
        assert!(state.request_info.is_none());
    }

    /// Test RateLimitState::store_pre_check() stores both request and token metrics
    /// Verifies that RPM and TPM metrics can be stored simultaneously
    #[test]
    fn test_rate_limit_state_store_pre_check_both_metrics() {
        let mut state = RateLimitState::new();
        let now = Instant::now();

        let req_info = RateLimitInfo {
            limit: 100,
            remaining: 99,
            reset_at: now + Duration::from_secs(60),
            window_start: now,
            retry_after: None,
        };

        let token_info = RateLimitInfo {
            limit: 1000,
            remaining: 999,
            reset_at: now + Duration::from_secs(60),
            window_start: now,
            retry_after: None,
        };

        state.store_pre_check(vec![
            (RateLimitMetric::RPM, req_info.clone()),
            (RateLimitMetric::TPM, token_info.clone()),
        ]);

        assert!(state.request_info.is_some());
        assert!(state.token_info.is_some());
        assert_eq!(state.request_info.as_ref().unwrap().limit, 100);
        assert_eq!(state.token_info.as_ref().unwrap().limit, 1000);
    }

    /// Test RateLimitState::store_post_check() only updates token metrics
    /// Verifies that post_check updates token_info but leaves request_info unchanged
    #[test]
    fn test_rate_limit_state_store_post_check_updates_token_only() {
        let mut state = RateLimitState::new();
        let now = Instant::now();

        // Set initial state
        let initial_req_info = RateLimitInfo {
            limit: 100,
            remaining: 99,
            reset_at: now + Duration::from_secs(60),
            window_start: now,
            retry_after: None,
        };

        let initial_token_info = RateLimitInfo {
            limit: 1000,
            remaining: 999,
            reset_at: now + Duration::from_secs(60),
            window_start: now,
            retry_after: None,
        };

        state.store_pre_check(vec![
            (RateLimitMetric::RPM, initial_req_info.clone()),
            (RateLimitMetric::TPM, initial_token_info.clone()),
        ]);

        // Update with post_check (should only update token info)
        let updated_token_info = RateLimitInfo {
            limit: 1000,
            remaining: 950, // Changed
            reset_at: now + Duration::from_secs(60),
            window_start: now,
            retry_after: None,
        };

        state.store_post_check(vec![(RateLimitMetric::TPM, updated_token_info.clone())]);

        // Request info should remain unchanged
        assert_eq!(state.request_info.as_ref().unwrap().remaining, 99);
        // Token info should be updated
        assert_eq!(state.token_info.as_ref().unwrap().remaining, 950);
    }

    /// Test RateLimitState::store_post_check() ignores request metrics
    /// Verifies that request metrics passed to post_check are ignored
    #[test]
    fn test_rate_limit_state_store_post_check_ignores_request_metrics() {
        let mut state = RateLimitState::new();
        let now = Instant::now();

        let initial_req_info = RateLimitInfo {
            limit: 100,
            remaining: 99,
            reset_at: now + Duration::from_secs(60),
            window_start: now,
            retry_after: None,
        };

        state.store_pre_check(vec![(RateLimitMetric::RPM, initial_req_info.clone())]);

        // Try to update with request metric in post_check (should be ignored)
        let new_req_info = RateLimitInfo {
            limit: 100,
            remaining: 50, // Different value
            reset_at: now + Duration::from_secs(60),
            window_start: now,
            retry_after: None,
        };

        state.store_post_check(vec![(RateLimitMetric::RPM, new_req_info)]);

        // Request info should remain unchanged
        assert_eq!(state.request_info.as_ref().unwrap().remaining, 99);
    }

    /// Test RateLimitState::add_headers() with empty state
    /// Verifies that no headers are added when state is empty
    #[test]
    fn test_rate_limit_state_add_headers_empty_state() {
        let state = RateLimitState::new();
        let mut headers = HeaderMap::new();

        state.add_headers(&mut headers);

        // No headers should be added for empty state
        assert!(headers.get("x-ratelimit-limit-requests").is_none());
        assert!(headers.get("x-ratelimit-limit-tokens").is_none());
    }

    /// Test RateLimitState::add_headers() with request metrics only
    /// Verifies that only request-related headers are added when only request_info is present
    #[test]
    fn test_rate_limit_state_add_headers_request_only() {
        let mut state = RateLimitState::new();
        let now = Instant::now();

        let req_info = RateLimitInfo {
            limit: 100,
            remaining: 99,
            reset_at: now + Duration::from_secs(65),
            window_start: now,
            retry_after: None,
        };

        state.store_pre_check(vec![(RateLimitMetric::RPM, req_info)]);

        let mut headers = HeaderMap::new();
        state.add_headers(&mut headers);

        assert_eq!(headers.get("x-ratelimit-limit-requests").unwrap(), "100");
        assert_eq!(headers.get("x-ratelimit-remaining-requests").unwrap(), "99");
        // Reset time should be formatted (approximately 1m5s, but may vary slightly)
        let reset = headers
            .get("x-ratelimit-reset-requests")
            .unwrap()
            .to_str()
            .unwrap();
        assert!(reset.starts_with("1m") || reset == "65s");

        // Token headers should not be present
        assert!(headers.get("x-ratelimit-limit-tokens").is_none());
    }

    /// Test RateLimitState::add_headers() with both request and token metrics
    /// Verifies that all rate limit headers are added when both metrics are present
    #[test]
    fn test_rate_limit_state_add_headers_both_metrics() {
        let mut state = RateLimitState::new();
        let now = Instant::now();

        let req_info = RateLimitInfo {
            limit: 100,
            remaining: 99,
            reset_at: now + Duration::from_secs(30),
            window_start: now,
            retry_after: None,
        };

        let token_info = RateLimitInfo {
            limit: 1000,
            remaining: 950,
            reset_at: now + Duration::from_secs(45),
            window_start: now,
            retry_after: None,
        };

        state.store_pre_check(vec![
            (RateLimitMetric::RPM, req_info),
            (RateLimitMetric::TPM, token_info),
        ]);

        let mut headers = HeaderMap::new();
        state.add_headers(&mut headers);

        // Request headers
        assert_eq!(headers.get("x-ratelimit-limit-requests").unwrap(), "100");
        assert_eq!(headers.get("x-ratelimit-remaining-requests").unwrap(), "99");

        // Token headers
        assert_eq!(headers.get("x-ratelimit-limit-tokens").unwrap(), "1000");
        assert_eq!(headers.get("x-ratelimit-remaining-tokens").unwrap(), "950");
    }

    #[test]
    fn test_rate_limit_state_overwrite_previous_values() {
        let mut state = RateLimitState::new();
        let now = Instant::now();

        let first_info = RateLimitInfo {
            limit: 100,
            remaining: 99,
            reset_at: now + Duration::from_secs(60),
            window_start: now,
            retry_after: None,
        };

        state.store_pre_check(vec![(RateLimitMetric::RPM, first_info)]);
        assert_eq!(state.request_info.as_ref().unwrap().remaining, 99);

        // Overwrite with new value
        let second_info = RateLimitInfo {
            limit: 100,
            remaining: 50,
            reset_at: now + Duration::from_secs(60),
            window_start: now,
            retry_after: None,
        };

        state.store_pre_check(vec![(RateLimitMetric::RPM, second_info)]);
        assert_eq!(state.request_info.as_ref().unwrap().remaining, 50);
    }

    /// Test that store_pre_check keeps the stricter limit when RPM+RPD are both present
    /// Verifies that the limit with lower remaining count is preserved
    #[test]
    fn test_rate_limit_state_keeps_stricter_rpm_rpd() {
        let mut state = RateLimitState::new();
        let now = Instant::now();

        // RPM: 60 limit, 5 remaining (stricter)
        let rpm_info = RateLimitInfo {
            limit: 60,
            remaining: 5,
            reset_at: now + Duration::from_secs(60),
            window_start: now,
            retry_after: None,
        };

        // RPD: 1000 limit, 500 remaining (less strict)
        let rpd_info = RateLimitInfo {
            limit: 1000,
            remaining: 500,
            reset_at: now + Duration::from_secs(86400),
            window_start: now,
            retry_after: None,
        };

        // Store both - should keep RPM (stricter)
        state.store_pre_check(vec![
            (RateLimitMetric::RPM, rpm_info.clone()),
            (RateLimitMetric::RPD, rpd_info),
        ]);

        // Should keep the stricter one (RPM with remaining=5)
        assert_eq!(state.request_info.as_ref().unwrap().remaining, 5);
        assert_eq!(state.request_info.as_ref().unwrap().limit, 60);
    }

    /// Test that store_pre_check keeps the stricter limit when TPM+TPD are both present
    #[test]
    fn test_rate_limit_state_keeps_stricter_tpm_tpd() {
        let mut state = RateLimitState::new();
        let now = Instant::now();

        // TPM: 1000 limit, 100 remaining (less strict)
        let tpm_info = RateLimitInfo {
            limit: 1000,
            remaining: 100,
            reset_at: now + Duration::from_secs(60),
            window_start: now,
            retry_after: None,
        };

        // TPD: 10000 limit, 50 remaining (stricter)
        let tpd_info = RateLimitInfo {
            limit: 10000,
            remaining: 50,
            reset_at: now + Duration::from_secs(86400),
            window_start: now,
            retry_after: None,
        };

        // Store both - should keep TPD (stricter)
        state.store_pre_check(vec![
            (RateLimitMetric::TPM, tpm_info),
            (RateLimitMetric::TPD, tpd_info.clone()),
        ]);

        // Should keep the stricter one (TPD with remaining=50)
        assert_eq!(state.token_info.as_ref().unwrap().remaining, 50);
        assert_eq!(state.token_info.as_ref().unwrap().limit, 10000);
    }

    /// Test that when remaining counts are equal, earlier reset time is chosen
    #[test]
    fn test_rate_limit_state_chooses_earlier_reset_when_equal_remaining() {
        let mut state = RateLimitState::new();
        let now = Instant::now();

        // RPM: 60 limit, 10 remaining, resets in 30 seconds (stricter - earlier reset)
        let rpm_info = RateLimitInfo {
            limit: 60,
            remaining: 10,
            reset_at: now + Duration::from_secs(30),
            window_start: now,
            retry_after: None,
        };

        // RPD: 1000 limit, 10 remaining, resets in 1 day
        let rpd_info = RateLimitInfo {
            limit: 1000,
            remaining: 10,
            reset_at: now + Duration::from_secs(86400),
            window_start: now,
            retry_after: None,
        };

        // Store both - should keep RPM (earlier reset)
        state.store_pre_check(vec![
            (RateLimitMetric::RPM, rpm_info.clone()),
            (RateLimitMetric::RPD, rpd_info),
        ]);

        // Should keep the one with earlier reset (RPM)
        assert_eq!(state.request_info.as_ref().unwrap().remaining, 10);
        assert_eq!(state.request_info.as_ref().unwrap().limit, 60);
        assert_eq!(
            state.request_info.as_ref().unwrap().reset_at,
            now + Duration::from_secs(30)
        );
    }
}
