//! OpenAI-compatible error envelope used by every proxy endpoint.
//!
//! OpenAI's clients expect this exact shape (spec §3):
//!
//! ```json
//! {
//!   "error": {
//!     "message": "…",
//!     "type": "invalid_request_error",
//!     "param": null,
//!     "code": null
//!   }
//! }
//! ```
//!
//! `ProxyError` is the internal error taxonomy; it implements
//! `IntoResponse` so handlers can `?`-propagate without touching
//! JSON shape boilerplate.

use aisix_gateway::BridgeError;
use aisix_ratelimit::RateLimitError;
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;

#[derive(Debug, Serialize, Clone)]
pub struct ErrorEnvelope {
    pub error: ErrorBody,
}

#[derive(Debug, Serialize, Clone)]
pub struct ErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

impl ErrorEnvelope {
    pub fn new(message: impl Into<String>, kind: &'static str) -> Self {
        Self {
            error: ErrorBody {
                message: message.into(),
                kind,
                param: None,
                code: None,
            },
        }
    }

    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.error.code = Some(code.into());
        self
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ProxyError {
    #[error("missing or malformed Authorization header")]
    MissingAuth,
    #[error("invalid API key")]
    InvalidApiKey,
    #[error("model {0:?} not found")]
    ModelNotFound(String),
    #[error("API key is not allowed to use model {0:?}")]
    ModelForbidden(String),
    #[error("request payload is invalid: {0}")]
    InvalidRequest(String),
    #[error("no bridge registered for provider")]
    ProviderUnavailable,
    #[error("content blocked by policy: {0}")]
    ContentFiltered(String),
    #[error("budget exceeded for ApiKey {0:?}")]
    BudgetExceeded(String),
    #[error(transparent)]
    RateLimit(#[from] RateLimitError),
    #[error(transparent)]
    Bridge(#[from] BridgeError),
}

impl ProxyError {
    pub fn status(&self) -> StatusCode {
        match self {
            ProxyError::MissingAuth | ProxyError::InvalidApiKey => StatusCode::UNAUTHORIZED,
            ProxyError::ModelForbidden(_) => StatusCode::FORBIDDEN,
            ProxyError::ModelNotFound(_) => StatusCode::NOT_FOUND,
            ProxyError::InvalidRequest(_) => StatusCode::BAD_REQUEST,
            ProxyError::ProviderUnavailable => StatusCode::SERVICE_UNAVAILABLE,
            ProxyError::ContentFiltered(_) => StatusCode::UNPROCESSABLE_ENTITY,
            ProxyError::BudgetExceeded(_) => StatusCode::TOO_MANY_REQUESTS,
            ProxyError::RateLimit(_) => StatusCode::TOO_MANY_REQUESTS,
            ProxyError::Bridge(b) => {
                StatusCode::from_u16(b.http_status()).unwrap_or(StatusCode::BAD_GATEWAY)
            }
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            ProxyError::MissingAuth | ProxyError::InvalidApiKey => "invalid_api_key",
            ProxyError::ModelForbidden(_) => "permission_denied",
            ProxyError::ModelNotFound(_) => "model_not_found",
            ProxyError::InvalidRequest(_) => "invalid_request_error",
            ProxyError::ProviderUnavailable => "provider_unavailable",
            ProxyError::ContentFiltered(_) => "content_filter",
            ProxyError::BudgetExceeded(_) => "billing_error",
            ProxyError::RateLimit(_) => "rate_limit_exceeded",
            ProxyError::Bridge(b) => b.error_type(),
        }
    }

    /// Seconds the client should wait before retrying. Only present for
    /// rate-limit-style rejections so the proxy can emit a `Retry-After`
    /// header.
    pub fn retry_after_secs(&self) -> Option<u64> {
        match self {
            ProxyError::RateLimit(e) => e.retry_after_secs(),
            _ => None,
        }
    }

    pub fn envelope(&self) -> ErrorEnvelope {
        let env = ErrorEnvelope::new(self.to_string(), self.kind());
        match self {
            ProxyError::BudgetExceeded(_) => env.with_code("budget_exceeded"),
            _ => env,
        }
    }
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        let status = self.status();
        let retry_after = self.retry_after_secs();
        let body = self.envelope();
        let mut response = (status, Json(body)).into_response();
        if let Some(secs) = retry_after {
            if let Ok(value) = HeaderValue::from_str(&secs.to_string()) {
                response.headers_mut().insert("retry-after", value);
            }
        }
        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_auth_maps_to_401_invalid_api_key() {
        let e = ProxyError::MissingAuth;
        assert_eq!(e.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(e.kind(), "invalid_api_key");
    }

    #[test]
    fn model_forbidden_is_403_permission_denied() {
        let e = ProxyError::ModelForbidden("gpt-4o".into());
        assert_eq!(e.status(), StatusCode::FORBIDDEN);
        assert_eq!(e.kind(), "permission_denied");
    }

    #[test]
    fn bridge_error_inherits_status_and_type() {
        let bridge_err = BridgeError::UpstreamStatus {
            status: 429,
            message: "rate limited".into(),
        };
        let wrapped = ProxyError::Bridge(bridge_err);
        assert_eq!(wrapped.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(wrapped.kind(), "upstream_error");
    }

    #[test]
    fn bridge_5xx_collapses_via_bridge_error_mapping() {
        let bridge_err = BridgeError::UpstreamStatus {
            status: 503,
            message: "busy".into(),
        };
        let wrapped = ProxyError::Bridge(bridge_err);
        assert_eq!(wrapped.status(), StatusCode::BAD_GATEWAY);
    }

    #[test]
    fn envelope_omits_null_param_and_code_on_wire() {
        let env = ProxyError::ModelNotFound("x".into()).envelope();
        let json = serde_json::to_value(&env).unwrap();
        assert_eq!(json["error"]["type"], "model_not_found");
        assert!(json["error"].get("param").is_none());
        assert!(json["error"].get("code").is_none());
    }
}
