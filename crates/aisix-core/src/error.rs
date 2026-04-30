//! Error taxonomies used across the gateway.
//!
//! Two public envelopes:
//! - [`ProxyError`] — OpenAI-compatible `{error: {message,type,param,code}}` envelope
//!   returned from the proxy surface (spec §4.1.1).
//! - [`AdminError`] — simple `{error_msg}` envelope returned from the admin surface
//!   (spec §10).
//!
//! Framework-specific `IntoResponse` impls live in the `aisix-proxy` / `aisix-admin`
//! crates so this crate stays dependency-light.

use serde::Serialize;
use thiserror::Error;

/// Errors surfaced on the `:3000` proxy API.
///
/// Each variant carries the information needed to build the OpenAI-compatible
/// response envelope, plus an HTTP status code.
#[derive(Debug, Error)]
pub enum ProxyError {
    #[error("missing or invalid API key")]
    Unauthorized,

    #[error("model `{0}` not found")]
    ModelNotFound(String),

    #[error("API key does not have access to model `{0}`")]
    ModelAccessForbidden(String),

    #[error("request body exceeds limit")]
    PayloadTooLarge,

    #[error("validation failed: {0}")]
    InvalidRequest(String),

    #[error("rate limit exceeded ({scope})")]
    RateLimitExceeded {
        scope: RateLimitScope,
        retry_after_secs: Option<u64>,
    },

    #[error("concurrency limit exceeded")]
    ConcurrencyLimitExceeded,

    #[error("budget exceeded")]
    BudgetExceeded,

    #[error("request timed out after {0} ms")]
    RequestTimeout(u64),

    #[error("upstream provider error: {0}")]
    ProviderError(String),

    #[error("content policy violation")]
    ContentPolicyViolation,

    #[error("internal error: {0}")]
    Internal(String),
}

/// Which quota was exhausted. Used for the OpenAI error envelope `code` field
/// and to decide which `x-ratelimit-*` headers to emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateLimitScope {
    Requests,
    Tokens,
    Concurrency,
}

impl std::fmt::Display for RateLimitScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Requests => "requests",
            Self::Tokens => "tokens",
            Self::Concurrency => "concurrency",
        })
    }
}

impl ProxyError {
    /// HTTP status code per spec §4.
    pub const fn status(&self) -> u16 {
        match self {
            Self::Unauthorized => 401,
            Self::ModelNotFound(_) | Self::InvalidRequest(_) => 400,
            Self::ModelAccessForbidden(_) | Self::ContentPolicyViolation => 403,
            Self::PayloadTooLarge => 413,
            Self::RateLimitExceeded { .. }
            | Self::ConcurrencyLimitExceeded
            | Self::BudgetExceeded => 429,
            Self::RequestTimeout(_) => 504,
            Self::ProviderError(_) => 502,
            Self::Internal(_) => 500,
        }
    }

    /// OpenAI `error.type` field.
    pub const fn error_type(&self) -> &'static str {
        match self {
            Self::Unauthorized => "invalid_request_error",
            Self::ModelNotFound(_) => "invalid_request_error",
            Self::ModelAccessForbidden(_) => "permission_error",
            Self::PayloadTooLarge => "invalid_request_error",
            Self::InvalidRequest(_) => "invalid_request_error",
            Self::RateLimitExceeded { .. } => "rate_limit_error",
            Self::ConcurrencyLimitExceeded => "rate_limit_error",
            Self::BudgetExceeded => "billing_error",
            Self::RequestTimeout(_) => "api_error",
            Self::ProviderError(_) => "api_error",
            Self::ContentPolicyViolation => "invalid_request_error",
            Self::Internal(_) => "api_error",
        }
    }

    /// OpenAI `error.code` field.
    pub const fn error_code(&self) -> &'static str {
        match self {
            Self::Unauthorized => "invalid_api_key",
            Self::ModelNotFound(_) => "model_not_found",
            Self::ModelAccessForbidden(_) => "model_access_forbidden",
            Self::PayloadTooLarge => "request_too_large",
            Self::InvalidRequest(_) => "invalid_request",
            Self::RateLimitExceeded { scope, .. } => match scope {
                RateLimitScope::Requests => "rate_limit_exceeded",
                RateLimitScope::Tokens => "token_limit_exceeded",
                RateLimitScope::Concurrency => "concurrency_limit_exceeded",
            },
            Self::ConcurrencyLimitExceeded => "concurrency_limit_exceeded",
            Self::BudgetExceeded => "budget_exceeded",
            Self::RequestTimeout(_) => "request_timeout",
            Self::ProviderError(_) => "provider_error",
            Self::ContentPolicyViolation => "content_policy_violation",
            Self::Internal(_) => "internal_error",
        }
    }

    /// Build the OpenAI-compatible response body.
    pub fn to_envelope(&self) -> ProxyErrorEnvelope {
        ProxyErrorEnvelope {
            error: ProxyErrorBody {
                message: self.to_string(),
                r#type: self.error_type(),
                param: None,
                code: self.error_code(),
            },
        }
    }

    /// If the error corresponds to a quota reset, returns seconds to wait.
    pub const fn retry_after_secs(&self) -> Option<u64> {
        match self {
            Self::RateLimitExceeded {
                retry_after_secs, ..
            } => *retry_after_secs,
            _ => None,
        }
    }
}

/// Serialisable OpenAI error envelope.
#[derive(Debug, Serialize)]
pub struct ProxyErrorEnvelope {
    pub error: ProxyErrorBody,
}

#[derive(Debug, Serialize)]
pub struct ProxyErrorBody {
    pub message: String,
    pub r#type: &'static str,
    pub param: Option<String>,
    pub code: &'static str,
}

/// Errors surfaced on the `:3001` admin API.
///
/// Admin errors use a dead-simple `{ "error_msg": ... }` envelope (spec §10).
#[derive(Debug, Error)]
pub enum AdminError {
    #[error("missing or invalid admin key")]
    Unauthorized,

    #[error("resource not found: {0}")]
    NotFound(String),

    #[error("duplicate name: {0}")]
    DuplicateName(String),

    #[error("validation failed: {0}")]
    Validation(String),

    #[error("storage error: {0}")]
    Storage(String),

    #[error("internal error: {0}")]
    Internal(String),
}

impl AdminError {
    pub const fn status(&self) -> u16 {
        match self {
            Self::Unauthorized => 401,
            Self::NotFound(_) => 404,
            Self::DuplicateName(_) | Self::Validation(_) => 400,
            Self::Storage(_) | Self::Internal(_) => 500,
        }
    }

    pub fn to_envelope(&self) -> AdminErrorEnvelope {
        AdminErrorEnvelope {
            error_msg: self.to_string(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct AdminErrorEnvelope {
    pub error_msg: String,
}

/// Bootstrap-time failures (config load, TLS, bind). Returned by `aisix-server`.
#[derive(Debug, Error)]
pub enum BootstrapError {
    #[error("failed to load configuration: {0}")]
    Config(String),

    #[error("etcd connection failed after {attempts} attempts: {source}")]
    Etcd {
        attempts: u32,
        #[source]
        source: anyhow::Error,
    },

    #[error("failed to bind {addr}: {source}")]
    Bind {
        addr: String,
        #[source]
        source: std::io::Error,
    },

    #[error("invalid TLS material (field={field}): {reason}")]
    Tls { field: &'static str, reason: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proxy_error_statuses_match_spec() {
        assert_eq!(ProxyError::Unauthorized.status(), 401);
        assert_eq!(ProxyError::ModelNotFound("x".into()).status(), 400);
        assert_eq!(ProxyError::ModelAccessForbidden("x".into()).status(), 403);
        assert_eq!(ProxyError::PayloadTooLarge.status(), 413);
        assert_eq!(
            ProxyError::RateLimitExceeded {
                scope: RateLimitScope::Requests,
                retry_after_secs: Some(5),
            }
            .status(),
            429
        );
        assert_eq!(ProxyError::RequestTimeout(1000).status(), 504);
        assert_eq!(ProxyError::ProviderError("x".into()).status(), 502);
    }

    #[test]
    fn proxy_error_envelope_is_openai_shape() {
        let err = ProxyError::ModelNotFound("gpt-9".into());
        let env = err.to_envelope();
        let body = serde_json::to_value(env).unwrap();
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], "model_not_found");
        assert!(body["error"]["message"].as_str().unwrap().contains("gpt-9"));
    }

    #[test]
    fn admin_error_envelope_is_simple_shape() {
        let env = AdminError::DuplicateName("openai/gpt-4".into()).to_envelope();
        let body = serde_json::to_value(env).unwrap();
        assert!(body.get("error_msg").is_some());
        assert!(body.get("error").is_none());
    }

    #[test]
    fn retry_after_only_set_for_rpm_tpm_not_concurrency() {
        let rpm = ProxyError::RateLimitExceeded {
            scope: RateLimitScope::Requests,
            retry_after_secs: Some(12),
        };
        assert_eq!(rpm.retry_after_secs(), Some(12));

        let conc = ProxyError::ConcurrencyLimitExceeded;
        assert_eq!(conc.retry_after_secs(), None);
    }
}
