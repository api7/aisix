//! Error envelopes for the proxy endpoints.
//!
//! Two on-the-wire envelope shapes — one per inbound protocol:
//!
//! - **OpenAI** (default, used by `/v1/chat/completions` and every other
//!   non-Anthropic endpoint) — spec §3 shape:
//!
//!   ```json
//!   {
//!     "error": {
//!       "message": "…",
//!       "type": "invalid_request_error",
//!       "param": null,
//!       "code": null
//!     }
//!   }
//!   ```
//!
//! - **Anthropic** (used by `/v1/messages` — closes #336). Per
//!   <https://docs.anthropic.com/en/api/errors>:
//!
//!   ```json
//!   {
//!     "type": "error",
//!     "error": {
//!       "type": "…",
//!       "message": "…"
//!     }
//!   }
//!   ```
//!
//!   The nested `error.type` maps from HTTP status onto the
//!   Anthropic SDK's strict `ErrorType` literal
//!   (`invalid_request_error` / `authentication_error` /
//!   `permission_error` / `not_found_error` / `request_too_large` /
//!   `rate_limit_error` / `timeout_error` / `overloaded_error` /
//!   `api_error`). Diverges from the OpenAI envelope's DP-stable
//!   taxonomy because the Anthropic SDK's `ErrorType` is a strict
//!   literal — emitting `"upstream_error"` would silently break
//!   customers branching on `e.body['error']['type']`. See
//!   [`anthropic_kind_from_status`] for the ecosystem-aligned mapping
//!   table.
//!
//! `ProxyError` is the internal error taxonomy; it implements
//! `IntoResponse` for the OpenAI shape so non-Anthropic handlers
//! `?`-propagate without ceremony. `/v1/messages` calls
//! [`ProxyError::into_anthropic_response`] explicitly so the
//! Anthropic shape lands on its responses.

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
    /// Budget-denial detail (prd-09b §5.8), flattened into the `error`
    /// block on a `budget_exceeded` 429 only. `None` (and thus absent
    /// from the wire) for every other error — upstream-translated
    /// errors, rate limits, validation, etc. — so the bare OpenAI
    /// {message,type,param,code} shape is preserved everywhere else.
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub budget: Option<BudgetErrorFields>,
}

/// The structured budget fields that `budget_exceeded` 429s lift from
/// cp-api's reason. Flattened into `ErrorBody`. Each field is omitted
/// when absent so a fallback-mode denial (cp-api unreachable, no
/// structured detail) still serializes cleanly with just a message.
#[derive(Debug, Serialize, Clone)]
pub struct BudgetErrorFields {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope_ref: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit_usd: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spent_usd: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub period: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub period_resets_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_after_seconds: Option<u64>,
}

impl ErrorEnvelope {
    pub fn new(message: impl Into<String>, kind: impl Into<String>) -> Self {
        Self {
            error: ErrorBody {
                message: message.into(),
                kind: kind.into(),
                param: None,
                code: None,
                budget: None,
            },
        }
    }

    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.error.code = Some(code.into());
        self
    }

    /// Attach the structured budget detail to the error block. Only
    /// the budget_exceeded path calls this.
    pub fn with_budget(mut self, r: &crate::budget::BudgetReason) -> Self {
        self.error.budget = Some(BudgetErrorFields {
            scope: r.scope.clone(),
            scope_ref: r.scope_ref.clone(),
            limit_usd: r.limit_usd.clone(),
            spent_usd: r.spent_usd.clone(),
            period: r.period.clone(),
            period_resets_at: r.period_resets_at.clone(),
            retry_after_seconds: r.retry_after_seconds,
        });
        self
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ProxyError {
    #[error("missing or malformed Authorization header")]
    MissingAuth,
    #[error("invalid API key")]
    InvalidApiKey,
    /// The presented key exists but its `expires_at` deadline has
    /// passed (#933). Deliberately caller-visible as "expired" (not a
    /// generic invalid-key 401): the caller already holds the secret,
    /// so naming the reason leaks nothing and tells them to request a
    /// fresh key instead of debugging a typo.
    #[error("API key has expired")]
    ApiKeyExpired,
    /// The presented key exists but was administratively disabled
    /// (#933). Same disclosure reasoning as [`Self::ApiKeyExpired`].
    #[error("API key has been disabled")]
    ApiKeyDisabled,
    #[error("model {0:?} not found")]
    ModelNotFound(String),
    /// A `/v1/videos/{video_id}` id that this gateway could not have
    /// minted — undecodable, or referencing a Model entry that no longer
    /// exists in the snapshot. 404, mirroring how the upstream videos
    /// API treats unknown job ids. The id echoes back verbatim: the
    /// caller supplied it, so it leaks nothing.
    #[error("video {0:?} not found")]
    VideoNotFound(String),
    #[error("API key is not allowed to use model {0:?}")]
    ModelForbidden(String),
    /// The resolved client IP is outside the model's `allowed_cidrs`
    /// allowlist (#557). Caller-visible message is intentionally generic and
    /// MUST NOT echo the configured CIDR ranges — handing back the allowlist
    /// lets a probe enumerate it (mirrors the #153 guardrail-redaction rule).
    /// The model name rides along for operator logs only.
    #[error("Access denied: your client IP is not allowed to access this model")]
    ModelIpRestricted(String),
    #[error("request payload is invalid: {0}")]
    InvalidRequest(String),
    #[error("no bridge registered for provider")]
    ProviderUnavailable,
    /// Every routing candidate was excluded by the runtime status layer
    /// (all in cooldown or background-unhealthy) and the routing model
    /// is configured with `when_all_unavailable: fail`. Caller-visible as
    /// 503 with a Retry-After hint derived from the nearest cooldown
    /// expiry. See [`aisix_core::WhenAllUnavailablePolicy`].
    #[error("all routing candidates are unavailable")]
    AllCandidatesUnavailable { retry_after_secs: Option<u64> },
    /// Caller-visible message MUST NOT carry the matched-pattern detail.
    /// Per #153, leaking the matched literal back to the caller defeats
    /// the point of an output guardrail (the whole purpose is to keep the
    /// forbidden content from reaching the caller; echoing it in the
    /// error envelope is a partial bypass and lets anyone who can
    /// trigger the guardrail enumerate the policy's blocklist).
    /// Every guardrail-block site builds the redacted public message via
    /// [`guardrail_block_message`] — generic policy wording plus the NAME
    /// of the guardrail that fired (#519 B.4b) — and emits the rich
    /// detail to `tracing` for operators.
    #[error("{0}")]
    ContentFiltered(String),
    // Carries cp-api's structured reason. Display forwards the cp-api
    // message verbatim (it's already a complete customer sentence —
    // "<scope> budget '<name>' exceeded ($X/period). Resets …"); the
    // structured fields ride along in the 429 error block via
    // `with_budget` (prd-09b §5.8).
    // Boxed: BudgetReason is ~184 bytes; inlining it would make this the
    // largest ProxyError variant and trip clippy::result_large_err across
    // every `Result<_, ProxyError>` in the hot path. The box keeps the
    // enum small (budget denial is rare, so the extra alloc is fine).
    #[error("{}", .0.message)]
    BudgetExceeded(Box<crate::budget::BudgetReason>),
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

/// The caller-visible message for a guardrail `Block` verdict.
///
/// Carries WHICH guardrail fired — `guardrail_name` is operator-assigned
/// metadata, safe to surface (#519 B.4b) — but never the matched-pattern
/// detail (per #153 that detail stays in `tracing` only; echoing it lets
/// callers enumerate the blocklist or extract the blocked output).
/// `side` is `"request"` (input hook) or `"response"` (output hook).
/// Every proxy endpoint family builds its 422 / SSE-error message through
/// this helper so the envelope shape can't drift between siblings.
pub(crate) fn guardrail_block_message(side: &str, guardrail_name: Option<&str>) -> String {
    match guardrail_name {
        Some(name) => format!("{side} blocked by content policy (guardrail '{name}')"),
        None => format!("{side} blocked by content policy"),
    }
}

impl ProxyError {
    pub fn status(&self) -> StatusCode {
        match self {
            ProxyError::MissingAuth
            | ProxyError::InvalidApiKey
            | ProxyError::ApiKeyExpired
            | ProxyError::ApiKeyDisabled => StatusCode::UNAUTHORIZED,
            ProxyError::ModelForbidden(_) => StatusCode::FORBIDDEN,
            ProxyError::ModelIpRestricted(_) => StatusCode::FORBIDDEN,
            ProxyError::ModelNotFound(_) => StatusCode::NOT_FOUND,
            ProxyError::VideoNotFound(_) => StatusCode::NOT_FOUND,
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
            ProxyError::MissingAuth
            | ProxyError::InvalidApiKey
            | ProxyError::ApiKeyExpired
            | ProxyError::ApiKeyDisabled => "invalid_api_key",
            ProxyError::ModelForbidden(_) => "permission_denied",
            ProxyError::ModelIpRestricted(_) => "permission_denied",
            ProxyError::ModelNotFound(_) => "model_not_found",
            ProxyError::VideoNotFound(_) => "video_not_found",
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
            // Source the Retry-After header from the same value the 429
            // body carries (prd-09b §5.8 retry_after_seconds), so the
            // header and body agree — SDKs back off on the header.
            ProxyError::BudgetExceeded(r) => r.retry_after_seconds,
            // Forward an upstream 429's Retry-After hint so SDK clients
            // back off on the provider's actual cool-down instead of a
            // default. The hint is parsed into `UpstreamStatus.retry_after`
            // by the bridge; only 429 carries a meaningful value.
            ProxyError::Bridge(BridgeError::UpstreamStatus {
                status: 429,
                retry_after: Some(d),
                ..
            }) => Some(d.as_secs()),
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
            ProxyError::BudgetExceeded(r) => env.with_code("budget_exceeded").with_budget(r),
            // Stable machine-readable code for SDKs to branch on, distinct
            // from the generic `permission_denied` type shared with
            // ModelForbidden (#557 AC-1).
            ProxyError::ModelIpRestricted(_) => env.with_code("ip_restricted"),
            // Stable machine-readable codes so SDKs can branch on the
            // lifecycle reason without parsing the message, while the
            // `error.type` stays the family-wide `invalid_api_key`.
            ProxyError::ApiKeyExpired => env.with_code("api_key_expired"),
            ProxyError::ApiKeyDisabled => env.with_code("api_key_disabled"),
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

/// Anthropic-shape error envelope serialized on the wire.
///
/// Matches the shape Anthropic SDKs parse — `body.type === "error"`
/// is the discriminator the official SDK branches on
/// (anthropic-sdk-python `_response.py::_to_api_error`). The nested
/// `error.type` carries the DP's stable taxonomy (the same string
/// the OpenAI envelope's `error.type` carries), so SDKs that branch
/// on the inner type still see the gateway-normalized value
/// (e.g. `"upstream_error"` per ai-gateway#327).
#[derive(Debug, Serialize, Clone)]
struct AnthropicErrorEnvelope {
    /// Top-level discriminator. Always `"error"` for error envelopes.
    #[serde(rename = "type")]
    discriminator: &'static str,
    error: AnthropicErrorBody,
}

#[derive(Debug, Serialize, Clone)]
struct AnthropicErrorBody {
    #[serde(rename = "type")]
    kind: String,
    message: String,
}

/// Map an HTTP status code to the Anthropic-canonical `error.type`
/// string (the SDK's `ErrorType` literal at
/// `anthropic-sdk-python/src/anthropic/types/shared/error_type.py`).
///
/// This deliberately diverges from the DP-stable OpenAI-shape inner
/// taxonomy (`upstream_error`, `model_not_found`, …) because the
/// Anthropic SDK's `ErrorType` is a strict `Literal[...]` — non-
/// canonical strings on `error.type` are static-type violations for
/// any customer doing `isinstance(e, anthropic.RateLimitError)` plus
/// `e.body['error']['type'] == 'rate_limit_error'`. Per CLAUDE.md §7
/// reference-implementation rule, this mapping mirrors the established
/// ecosystem's Anthropic status-to-type table verbatim — divergence from
/// the established ecosystem here would silently break Claude SDK users.
///
/// (The OpenAI envelope's inner `error.type` keeps the DP-stable
/// strings per ai-gateway#327; that contract is unchanged on
/// `/v1/chat/completions`.)
fn anthropic_kind_from_status(status: StatusCode) -> &'static str {
    match status.as_u16() {
        400 | 422 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        413 => "request_too_large",
        429 => "rate_limit_error",
        503 => "overloaded_error",
        // 408 timeout maps to `timeout_error` in the SDK literal; the
        // gateway doesn't emit 408 today (timeouts surface as 502 via
        // the Bridge), but the case is kept for completeness.
        408 => "timeout_error",
        // 500 / 502 / 504-599 plus anything outside the recognised
        // codes fall back to `api_error` — the SDK literal's catch-all
        // for generic upstream / server faults.
        _ => "api_error",
    }
}

impl ProxyError {
    /// Render this error as an Anthropic-shape `{type:"error", error:
    /// {type, message}}` HTTP response. Used by `/v1/messages` so the
    /// Anthropic SDK's envelope parser sees a shape the official
    /// SDK and the broader ecosystem both treat as canonical.
    ///
    /// **Inner `error.type` policy:** maps from the HTTP status code
    /// to the Anthropic SDK's `ErrorType` literal via
    /// [`anthropic_kind_from_status`] — NOT the DP-stable OpenAI-shape
    /// inner taxonomy. The Anthropic SDK's `ErrorType` is a strict
    /// `Literal[...]`, so emitting DP-internal strings like
    /// `"upstream_error"` would break customers branching on
    /// `error.type`. The DP-stable taxonomy is preserved on the
    /// OpenAI envelope only (ai-gateway#327); the Anthropic envelope
    /// follows ecosystem convention.
    ///
    /// Reuses [`Self::envelope`] for the 4xx/5xx message-classification
    /// and upstream-message redaction logic so the two envelope
    /// renderers can't drift on those rules.
    pub fn into_anthropic_response(self) -> Response {
        let status = self.status();
        let retry_after = self.retry_after_secs();
        let kind = anthropic_kind_from_status(status).to_string();
        // Reuse OpenAI envelope only for the SAFE-MESSAGE logic
        // (5xx body redaction, 4xx upstream-message pass-through).
        // The inner type is overwritten to the Anthropic-canonical
        // string above.
        let openai_env = self.envelope();
        let anth_body = AnthropicErrorEnvelope {
            discriminator: "error",
            error: AnthropicErrorBody {
                kind,
                message: openai_env.error.message,
            },
        };
        let mut response = (status, Json(anth_body)).into_response();
        if let Some(secs) = retry_after {
            if let Ok(value) = HeaderValue::from_str(&secs.to_string()) {
                response.headers_mut().insert("retry-after", value);
            }
        }
        response
    }
}

/// Map an axum `JsonRejection` (the body-extractor failure on a POST
/// handler) onto the internal [`ProxyError`] taxonomy. The caller
/// decides the wire envelope (`into_response` for OpenAI shape /
/// `into_anthropic_response` for the Anthropic shape) — this helper
/// only classifies the failure.
///
/// Shared by `/v1/messages` and `/v1/messages/count_tokens` so the two
/// Anthropic-protocol handlers can't drift on the discrimination rules
/// below:
///
/// - `BytesRejection` is a composite rejection whose inner
///   `FailedToBufferBody` has two variants: `LengthLimitError`
///   (`413 PAYLOAD_TOO_LARGE` — the configured body cap was exceeded
///   during read; the chunked / no-Content-Length case the
///   `enforce_request_body_limit` middleware can't catch up front) and
///   `UnknownBodyError` (`400 BAD_REQUEST` — a transport-side body-read
///   failure, e.g. peer reset mid-body). They MUST map to
///   `RequestTooLarge` vs `InvalidRequest` respectively, because the
///   Anthropic SDK's non-retriable-cap branch assumes a true cap hit —
///   mislabelling a transport failure as `request_too_large` breaks it.
///   Discriminate via the rejection's own `.status()`.
/// - `JsonRejection` is `#[non_exhaustive]`, so the fallback arm catches
///   today's `JsonDataError` / `JsonSyntaxError` / `MissingJsonContentType`
///   AND any future variant axum adds, defaulting to a 400
///   `invalid_request_error` until each gets an explicit policy.
pub(crate) fn proxy_error_from_json_rejection(
    rej: axum::extract::rejection::JsonRejection,
    limit_bytes: usize,
) -> ProxyError {
    use axum::extract::rejection::JsonRejection;
    match rej {
        JsonRejection::BytesRejection(inner) if inner.status() == StatusCode::PAYLOAD_TOO_LARGE => {
            ProxyError::RequestTooLarge { limit_bytes }
        }
        JsonRejection::BytesRejection(_) => {
            ProxyError::InvalidRequest("failed to read request body".into())
        }
        _ => ProxyError::InvalidRequest("invalid JSON request body".into()),
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
    fn model_ip_restricted_is_403_with_ip_restricted_code() {
        let e = ProxyError::ModelIpRestricted("gpt-4o".into());
        assert_eq!(e.status(), StatusCode::FORBIDDEN);
        assert_eq!(e.kind(), "permission_denied");
        let json = serde_json::to_value(e.envelope()).unwrap();
        assert_eq!(json["error"]["type"], "permission_denied");
        assert_eq!(json["error"]["code"], "ip_restricted");
        // Message must stay generic — never echo the configured CIDR list.
        let msg = json["error"]["message"].as_str().unwrap();
        assert!(msg.contains("not allowed to access this model"));
        assert!(!msg.contains("gpt-4o"));
    }

    #[test]
    fn bridge_error_inherits_status_and_type() {
        let bridge_err = BridgeError::upstream_status(429, "rate limited");
        let wrapped = ProxyError::Bridge(bridge_err);
        assert_eq!(wrapped.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(wrapped.kind(), "upstream_error");
    }

    #[test]
    fn upstream_429_forwards_retry_after_hint() {
        // An upstream 429 carrying a parsed Retry-After must surface
        // through retry_after_secs so the response writer emits the
        // header SDKs back off on (#144).
        let wrapped = ProxyError::Bridge(BridgeError::upstream_status_with_retry_after(
            429,
            "rate limited",
            Some(std::time::Duration::from_secs(30)),
        ));
        assert_eq!(wrapped.retry_after_secs(), Some(30));
    }

    #[test]
    fn upstream_429_without_hint_has_no_retry_after() {
        let wrapped = ProxyError::Bridge(BridgeError::upstream_status(429, "rate limited"));
        assert_eq!(wrapped.retry_after_secs(), None);
    }

    #[test]
    fn non_429_upstream_does_not_forward_retry_after() {
        // Only 429 carries a meaningful hint; a stray Retry-After on a
        // 5xx must not leak to the client.
        let wrapped = ProxyError::Bridge(BridgeError::upstream_status_with_retry_after(
            503,
            "unavailable",
            Some(std::time::Duration::from_secs(30)),
        ));
        assert_eq!(wrapped.retry_after_secs(), None);
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

    // ─── Anthropic envelope (#336) ────────────────────────────────────
    //
    // /v1/messages must emit `{type:"error", error:{type, message}}`
    // — the Anthropic-SDK strict envelope discriminator
    // (anthropic-sdk-python `_response.py::_to_api_error`). These tests
    // assert the wire shape AND that the DP-stable inner `error.type`
    // taxonomy (`upstream_error`, `invalid_api_key`, …) is preserved
    // unchanged from the OpenAI envelope per ai-gateway#327.

    use axum::body::to_bytes;

    async fn body_to_json(resp: Response) -> serde_json::Value {
        let bytes = to_bytes(resp.into_body(), 65536).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    /// Shared envelope-shape assertion used across every Anthropic
    /// envelope test below — keeps the contract surface tight against
    /// a future regression that flipped any single error variant back
    /// to the OpenAI envelope.
    async fn assert_anthropic_envelope(
        resp: Response,
        expected_status: StatusCode,
        expected_kind: &str,
    ) -> serde_json::Value {
        assert_eq!(resp.status(), expected_status);
        let json = body_to_json(resp).await;
        assert_eq!(
            json["type"], "error",
            "top-level discriminator must be the literal string \"error\""
        );
        assert_eq!(
            json["error"]["type"], expected_kind,
            "inner error.type must follow Anthropic SDK ErrorType literal"
        );
        assert!(
            json["error"]["message"].is_string(),
            "error.message must be present and a string"
        );
        assert!(
            json["error"].get("code").is_none(),
            "OpenAI-only field `code` must be absent from the Anthropic envelope"
        );
        assert!(
            json["error"].get("param").is_none(),
            "OpenAI-only field `param` must be absent from the Anthropic envelope"
        );
        json
    }

    #[tokio::test]
    async fn anthropic_envelope_404_maps_to_not_found_error() {
        let err = ProxyError::ModelNotFound("claude-x".into());
        let resp = err.into_anthropic_response();
        let json = assert_anthropic_envelope(resp, StatusCode::NOT_FOUND, "not_found_error").await;
        assert!(
            json["error"]["message"]
                .as_str()
                .unwrap_or("")
                .contains("claude-x"),
            "error message must surface the missing model id",
        );
    }

    #[tokio::test]
    async fn anthropic_envelope_401_maps_to_authentication_error() {
        let err = ProxyError::MissingAuth;
        let resp = err.into_anthropic_response();
        assert_anthropic_envelope(resp, StatusCode::UNAUTHORIZED, "authentication_error").await;
    }

    #[tokio::test]
    async fn anthropic_envelope_403_maps_to_permission_error() {
        let err = ProxyError::ModelForbidden("gpt-4o".into());
        let resp = err.into_anthropic_response();
        assert_anthropic_envelope(resp, StatusCode::FORBIDDEN, "permission_error").await;
    }

    #[tokio::test]
    async fn anthropic_envelope_403_ip_restricted_maps_to_permission_error() {
        let err = ProxyError::ModelIpRestricted("claude-sonnet-4-5".into());
        let resp = err.into_anthropic_response();
        assert_anthropic_envelope(resp, StatusCode::FORBIDDEN, "permission_error").await;
    }

    #[tokio::test]
    async fn anthropic_envelope_400_maps_to_invalid_request_error() {
        let err = ProxyError::InvalidRequest("`max_tokens` is required".into());
        let resp = err.into_anthropic_response();
        assert_anthropic_envelope(resp, StatusCode::BAD_REQUEST, "invalid_request_error").await;
    }

    #[tokio::test]
    async fn anthropic_envelope_413_maps_to_request_too_large() {
        let err = ProxyError::RequestTooLarge {
            limit_bytes: 1_048_576,
        };
        let resp = err.into_anthropic_response();
        assert_anthropic_envelope(resp, StatusCode::PAYLOAD_TOO_LARGE, "request_too_large").await;
    }

    #[tokio::test]
    async fn anthropic_envelope_422_content_filter_maps_to_invalid_request_error() {
        // Content-filter rejections share 422 with the OpenAI side;
        // Anthropic-canonical 422 maps to `invalid_request_error`
        // (no dedicated content-filter type in the SDK literal).
        let err = ProxyError::ContentFiltered("request blocked by content policy".into());
        let resp = err.into_anthropic_response();
        assert_anthropic_envelope(
            resp,
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_request_error",
        )
        .await;
    }

    #[tokio::test]
    async fn anthropic_envelope_429_budget_exceeded_maps_to_rate_limit_error() {
        let err =
            ProxyError::BudgetExceeded(Box::new(crate::budget::BudgetReason::message_only("ak-1")));
        let resp = err.into_anthropic_response();
        assert_anthropic_envelope(resp, StatusCode::TOO_MANY_REQUESTS, "rate_limit_error").await;
    }

    #[test]
    fn openai_envelope_budget_exceeded_carries_structured_fields() {
        // prd-09b §5.8: the budget_exceeded 429 lifts cp-api's structured
        // reason into the error block. Pin scope / scope_ref / limit_usd /
        // spent_usd / period so a regression that drops them (the old
        // String-only variant) fails here.
        let err = ProxyError::BudgetExceeded(Box::new(crate::budget::BudgetReason {
            message: "team budget 'frontend' exceeded ($1.00/month). Resets soon.".into(),
            scope: Some("team".into()),
            scope_ref: Some("team-uuid-1".into()),
            limit_usd: Some("1.00".into()),
            spent_usd: Some("2.00".into()),
            period: Some("month".into()),
            period_resets_at: Some("2026-06-01T00:00:00Z".into()),
            retry_after_seconds: Some(259_200),
        }));
        // The Retry-After *header* must source the same value the body
        // carries — otherwise SDKs (which back off on the header) and
        // the body disagree.
        assert_eq!(err.retry_after_secs(), Some(259_200));
        let v = serde_json::to_value(err.envelope()).unwrap();
        let e = &v["error"];
        assert_eq!(e["type"], "billing_error");
        assert_eq!(e["code"], "budget_exceeded");
        assert_eq!(e["scope"], "team");
        assert_eq!(e["scope_ref"], "team-uuid-1");
        assert_eq!(e["limit_usd"], "1.00");
        assert_eq!(e["spent_usd"], "2.00");
        assert_eq!(e["period"], "month");
        assert_eq!(e["period_resets_at"], "2026-06-01T00:00:00Z");
        assert_eq!(e["retry_after_seconds"], 259_200);
        assert!(e["message"]
            .as_str()
            .unwrap()
            .contains("team budget 'frontend'"));

        // A non-budget error must NOT carry these fields — the flatten
        // omits them so every other error keeps the bare OpenAI shape.
        let other = serde_json::to_value(ProxyError::ModelNotFound("m".into()).envelope()).unwrap();
        assert!(other["error"].get("scope").is_none());
        assert!(other["error"].get("limit_usd").is_none());
    }

    #[tokio::test]
    async fn anthropic_envelope_budget_exceeded_omits_structured_fields() {
        // The structured budget fields are an OpenAI-envelope extension
        // only. The Anthropic /v1/messages error block is the strict
        // {type, message} shape — a fully-populated reason must NOT leak
        // scope / limit_usd etc. into it.
        let err = ProxyError::BudgetExceeded(Box::new(crate::budget::BudgetReason {
            message: "team budget 'frontend' exceeded ($1.00/month). Resets soon.".into(),
            scope: Some("team".into()),
            scope_ref: Some("team-uuid-1".into()),
            limit_usd: Some("1.00".into()),
            spent_usd: Some("2.00".into()),
            period: Some("month".into()),
            period_resets_at: Some("2026-06-01T00:00:00Z".into()),
            retry_after_seconds: Some(259_200),
        }));
        let resp = err.into_anthropic_response();
        let json =
            assert_anthropic_envelope(resp, StatusCode::TOO_MANY_REQUESTS, "rate_limit_error")
                .await;
        assert!(json["error"].get("scope").is_none());
        assert!(json["error"].get("scope_ref").is_none());
        assert!(json["error"].get("limit_usd").is_none());
        assert!(json["error"].get("spent_usd").is_none());
    }

    #[tokio::test]
    async fn anthropic_envelope_503_all_candidates_unavailable_maps_to_overloaded_error() {
        let err = ProxyError::AllCandidatesUnavailable {
            retry_after_secs: Some(7),
        };
        let resp = err.into_anthropic_response();
        assert_anthropic_envelope(resp, StatusCode::SERVICE_UNAVAILABLE, "overloaded_error").await;
    }

    #[tokio::test]
    async fn anthropic_envelope_503_carries_retry_after_header() {
        // Anthropic SDK honors the `Retry-After` header on 503 + 429
        // (anthropic-sdk-python `_base_client.py::_should_retry`).
        // The Anthropic envelope renderer must propagate it the same
        // way the OpenAI envelope renderer does.
        let err = ProxyError::AllCandidatesUnavailable {
            retry_after_secs: Some(42),
        };
        let resp = err.into_anthropic_response();
        let retry_after = resp.headers().get("retry-after").expect("retry-after set");
        assert_eq!(retry_after.to_str().unwrap(), "42");
    }

    #[tokio::test]
    async fn anthropic_envelope_bridge_5xx_maps_to_api_error_with_message_redacted() {
        // 5xx collapse contract from ai-gateway#322/#327 — upstream
        // body redacted, customer sees a generic 502 wrapped in the
        // Anthropic-shape envelope with `error.type = "api_error"`
        // (Anthropic's catch-all for upstream/server failure).
        let bridge_err = BridgeError::upstream_status(503, "engine internal panic");
        let err = ProxyError::Bridge(bridge_err);
        let resp = err.into_anthropic_response();
        let json = assert_anthropic_envelope(resp, StatusCode::BAD_GATEWAY, "api_error").await;
        let msg = json["error"]["message"].as_str().unwrap_or("");
        assert!(
            !msg.contains("engine internal panic"),
            "upstream 5xx body must be redacted from the Anthropic envelope, got: {msg}",
        );
        assert!(
            msg.contains("503"),
            "redacted message must still surface the upstream status, got: {msg}",
        );
    }

    #[tokio::test]
    async fn anthropic_envelope_bridge_429_maps_to_rate_limit_error() {
        // Upstream 429 passes through verbatim status; Anthropic-side
        // `error.type` maps to `rate_limit_error`. The upstream
        // message is preserved on 4xx (vs 5xx redaction).
        let bridge_err = BridgeError::upstream_status(429, "rate limited by anthropic");
        let err = ProxyError::Bridge(bridge_err);
        let resp = err.into_anthropic_response();
        let json =
            assert_anthropic_envelope(resp, StatusCode::TOO_MANY_REQUESTS, "rate_limit_error")
                .await;
        // 4xx message pass-through.
        let msg = json["error"]["message"].as_str().unwrap_or("");
        assert!(
            msg.contains("rate limited"),
            "4xx upstream message must pass through to Anthropic envelope, got: {msg}",
        );
    }
}
