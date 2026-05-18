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
    /// `error.type` token. Was `&'static str` before #322 — widened to
    /// owned `String` because the type can now reflect an upstream-
    /// derived OpenAI taxonomy token (`rate_limit_exceeded`,
    /// `insufficient_quota`, …) when the error_translate layer maps a
    /// non-OpenAI upstream to OpenAI shape.
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

impl ErrorEnvelope {
    pub fn new(message: impl Into<String>, kind: impl Into<String>) -> Self {
        Self {
            error: ErrorBody {
                message: message.into(),
                kind: kind.into(),
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
    /// Every routing candidate was excluded by the runtime status layer
    /// (all in cooldown or background-unhealthy) and the routing model
    /// is configured with `on_all_filtered: fail`. Caller-visible as
    /// 503 with a Retry-After hint derived from the nearest cooldown
    /// expiry. See [`aisix_core::OnAllFilteredPolicy`].
    #[error("all routing candidates are unavailable")]
    AllCandidatesUnavailable { retry_after_secs: Option<u64> },
    /// Caller-visible message MUST NOT carry the matched-pattern detail.
    /// Per #153, leaking the matched literal back to the caller defeats
    /// the point of an output guardrail (the whole purpose is to keep the
    /// forbidden content from reaching the caller; echoing it in the
    /// error envelope is a partial bypass and lets anyone who can
    /// trigger the guardrail enumerate the policy's blocklist).
    /// Constructors at `chat.rs::route_chat_completions` and
    /// `chat.rs::dispatch_and_render` build a redacted public message
    /// (`"request blocked by content policy"` /
    /// `"response blocked by content policy"`) and emit the rich detail
    /// to `tracing` for operators.
    #[error("{0}")]
    ContentFiltered(String),
    #[error("budget exceeded for ApiKey {0:?}")]
    BudgetExceeded(String),
    /// Per RFC 9110 §15.5.14, a request body that exceeds a server-
    /// imposed limit gets a `413 Content Too Large`. The caller-visible
    /// `message` is intentionally bare of the actual incoming size
    /// (the limit is the only stable detail the caller needs). Set by
    /// the body-limit middleware in `lib.rs::enforce_request_body_limit`
    /// when the inbound `Content-Length` exceeds the configured cap.
    #[error("request body exceeds {limit_bytes}-byte limit")]
    RequestTooLarge { limit_bytes: usize },
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
            ProxyError::AllCandidatesUnavailable { .. } => StatusCode::SERVICE_UNAVAILABLE,
            ProxyError::ContentFiltered(_) => StatusCode::UNPROCESSABLE_ENTITY,
            ProxyError::BudgetExceeded(_) => StatusCode::TOO_MANY_REQUESTS,
            ProxyError::RequestTooLarge { .. } => StatusCode::PAYLOAD_TOO_LARGE,
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
            ProxyError::RequestTooLarge { .. } => "invalid_request_error",
            ProxyError::ProviderUnavailable => "provider_unavailable",
            ProxyError::AllCandidatesUnavailable { .. } => "all_candidates_unavailable",
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
            ProxyError::AllCandidatesUnavailable { retry_after_secs } => *retry_after_secs,
            _ => None,
        }
    }

    pub fn envelope(&self) -> ErrorEnvelope {
        // Bridge-surface upstream errors get special handling: the
        // bridge has best-effort-parsed the upstream envelope into a
        // structured [`UpstreamErrorView`], and for same-wire 4xx
        // (OpenAI upstream + OpenAI client) we forward the parsed
        // fields directly instead of wrapping them inside the
        // gateway's generic `upstream_error` envelope.
        //
        // 5xx and non-JSON bodies fall back to the generic envelope —
        // upstream internal-server-error detail (engine names, queue
        // depth, etc.) is operator-internal and must not bleed through.
        // Cross-wire translation (Anthropic / Bedrock / Vertex / Azure
        // → OpenAI shape) ships in a follow-up via `error_translate`.
        if let ProxyError::Bridge(aisix_gateway::BridgeError::UpstreamStatus {
            status,
            message,
            parsed,
            wire,
            ..
        }) = self
        {
            return render_bridge_upstream_envelope(*status, message, parsed.as_deref(), *wire);
        }
        let env = ErrorEnvelope::new(self.to_string(), self.kind());
        match self {
            ProxyError::BudgetExceeded(_) => env.with_code("budget_exceeded"),
            _ => env,
        }
    }
}

/// Build the customer-visible envelope for an upstream HTTP error.
///
/// **4xx**: delegate to [`crate::error_translate::render_openai_envelope`],
/// which (a) passes OpenAI-wire fields verbatim, (b) translates
/// Anthropic / Bedrock / Vertex / AzureOpenAI taxonomy via per-wire
/// tables so the OpenAI-shape `error.type` and `error.code` carry the
/// retry semantics SDKs depend on.
///
/// **5xx**: emit a canned `upstream returned {status}` message under
/// `type: upstream_error`. Upstream 5xx bodies routinely embed
/// operator-internal detail (engine names, shard ids, queue depth,
/// ARNs in raw AWS messages) — surfacing them to the customer leaks
/// internal taxonomy. The full upstream body remains in operator
/// logs via tracing.
///
/// **`UpstreamWire::Unknown`** (cooldown fixtures / synthesised
/// errors): legacy generic envelope.
fn render_bridge_upstream_envelope(
    status: u16,
    message: &str,
    parsed: Option<&aisix_gateway::UpstreamErrorView>,
    wire: aisix_gateway::UpstreamWire,
) -> ErrorEnvelope {
    let is_4xx = (400..500).contains(&status);
    if is_4xx && !matches!(wire, aisix_gateway::UpstreamWire::Unknown) {
        return ErrorEnvelope {
            error: crate::error_translate::render_openai_envelope(parsed, wire, message),
        };
    }
    let safe_message = if (500..600).contains(&status) {
        // Suppress upstream `error.message` on 5xx — engine names /
        // shard ids / ARNs commonly appear here and are not customer
        // information.
        format!("upstream returned {status}")
    } else {
        message.to_string()
    };
    ErrorEnvelope::new(safe_message, "upstream_error")
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
        let bridge_err = BridgeError::upstream_status(429, "rate limited");
        let wrapped = ProxyError::Bridge(bridge_err);
        assert_eq!(wrapped.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(wrapped.kind(), "upstream_error");
    }

    #[test]
    fn bridge_5xx_collapses_via_bridge_error_mapping() {
        let bridge_err = BridgeError::upstream_status(503, "busy");
        let wrapped = ProxyError::Bridge(bridge_err);
        assert_eq!(wrapped.status(), StatusCode::BAD_GATEWAY);
    }

    #[test]
    fn all_candidates_unavailable_is_503_with_optional_retry_after() {
        let with_hint = ProxyError::AllCandidatesUnavailable {
            retry_after_secs: Some(42),
        };
        assert_eq!(with_hint.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(with_hint.kind(), "all_candidates_unavailable");
        assert_eq!(with_hint.retry_after_secs(), Some(42));

        let no_hint = ProxyError::AllCandidatesUnavailable {
            retry_after_secs: None,
        };
        assert_eq!(no_hint.retry_after_secs(), None);
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
