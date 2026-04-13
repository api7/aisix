//! Gateway error types.
//!
//! `GatewayError` is the unified error type for the gateway SDK layer
//! (Layer 1-3). It covers validation errors, format bridging errors,
//! provider HTTP errors, and stream errors. Each variant carries enough
//! context for the proxy layer to produce an appropriate HTTP response.

use http::StatusCode;
use serde_json::Value;

/// Unified error type for the gateway SDK.
#[derive(Debug, thiserror::Error)]
pub enum GatewayError {
    // ── Client errors (not retryable) ──
    /// Request validation failed (e.g., missing required field).
    #[error("validation: {0}")]
    Validation(String),

    /// Format bridging failed (e.g., cannot map an Anthropic field to hub format).
    #[error("format bridge: {0}")]
    Bridge(String),

    /// Data transformation failed (e.g., JSON deserialization of provider response).
    #[error("data transform: {0}")]
    Transform(String),

    /// The requested format is not natively supported by the provider.
    #[error("format not natively supported by provider {provider}")]
    NativeNotSupported { provider: String },

    /// Internal gateway or server-side configuration error.
    #[error("internal: {0}")]
    Internal(String),

    // ── Provider errors (may be retryable) ──
    /// The upstream provider returned an error response.
    #[error("provider {provider} returned {status}: {body}")]
    Provider {
        status: StatusCode,
        body: Value,
        provider: String,
        retryable: bool,
    },

    // ── Infrastructure errors (usually retryable) ──
    /// HTTP transport error (connection, timeout, etc.).
    #[error("HTTP: {0}")]
    Http(#[source] reqwest::Error),

    /// Error during stream processing.
    #[error("stream: {0}")]
    Stream(String),
}

impl GatewayError {
    /// Whether this error is safe to retry.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Provider { retryable, .. } => *retryable,
            Self::Http(e) => e.is_timeout() || e.is_connect(),
            Self::Stream(_) => true,
            _ => false,
        }
    }

    /// Map to an HTTP status code for proxy-layer responses.
    pub fn status_code(&self) -> StatusCode {
        match self {
            Self::Validation(_) | Self::Bridge(_) => StatusCode::BAD_REQUEST,
            Self::Transform(_) => StatusCode::UNPROCESSABLE_ENTITY,
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::Provider { status, .. } => *status,
            Self::Http(_) | Self::Stream(_) => StatusCode::BAD_GATEWAY,
            Self::NativeNotSupported { .. } => StatusCode::NOT_IMPLEMENTED,
        }
    }
}

/// Convenience alias for gateway results.
pub type Result<T> = std::result::Result<T, GatewayError>;

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn validation_not_retryable() {
        let e = GatewayError::Validation("missing field".into());
        assert!(!e.is_retryable());
        assert_eq!(e.status_code(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn bridge_not_retryable() {
        let e = GatewayError::Bridge("cannot map field X".into());
        assert!(!e.is_retryable());
        assert_eq!(e.status_code(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn transform_not_retryable() {
        let e = GatewayError::Transform("bad json".into());
        assert!(!e.is_retryable());
        assert_eq!(e.status_code(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[test]
    fn native_not_supported() {
        let e = GatewayError::NativeNotSupported {
            provider: "gemini".into(),
        };
        assert!(!e.is_retryable());
        assert_eq!(e.status_code(), StatusCode::NOT_IMPLEMENTED);
        assert!(e.to_string().contains("gemini"));
    }

    #[test]
    fn provider_retryable_when_flagged() {
        let e = GatewayError::Provider {
            status: StatusCode::TOO_MANY_REQUESTS,
            body: json!({"error": "rate limited"}),
            provider: "openai".into(),
            retryable: true,
        };
        assert!(e.is_retryable());
        assert_eq!(e.status_code(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[test]
    fn provider_not_retryable_when_not_flagged() {
        let e = GatewayError::Provider {
            status: StatusCode::BAD_REQUEST,
            body: json!({"error": "bad request"}),
            provider: "anthropic".into(),
            retryable: false,
        };
        assert!(!e.is_retryable());
        assert_eq!(e.status_code(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn stream_error_retryable() {
        let e = GatewayError::Stream("connection reset".into());
        assert!(e.is_retryable());
        assert_eq!(e.status_code(), StatusCode::BAD_GATEWAY);
    }

    #[test]
    fn display_messages() {
        assert_eq!(
            GatewayError::Validation("x".into()).to_string(),
            "validation: x"
        );
        assert_eq!(
            GatewayError::Bridge("y".into()).to_string(),
            "format bridge: y"
        );
        let provider_err = GatewayError::Provider {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            body: json!("err"),
            provider: "openai".into(),
            retryable: false,
        };
        assert!(provider_err.to_string().contains("openai"));
        assert!(provider_err.to_string().contains("500"));
        assert_eq!(
            GatewayError::Internal("boom".into()).to_string(),
            "internal: boom"
        );
    }

    #[test]
    fn internal_error_is_not_retryable_and_maps_to_500() {
        let e = GatewayError::Internal("misconfigured provider registry".into());
        assert!(!e.is_retryable());
        assert_eq!(e.status_code(), StatusCode::INTERNAL_SERVER_ERROR);
    }
}
