//! Cross-wire error-envelope translation.
//!
//! Each upstream provider speaks a different error-envelope taxonomy:
//!
//! | Wire        | `error.type` examples                  | Has `code`/`param`? |
//! |-------------|----------------------------------------|---------------------|
//! | OpenAI      | `rate_limit_exceeded`, `invalid_api_key` | yes               |
//! | Anthropic   | `rate_limit_error`, `overloaded_error`   | no                |
//! | Bedrock     | `ThrottlingException`, `ValidationException` | no            |
//! | Vertex      | `RESOURCE_EXHAUSTED`, `PERMISSION_DENIED` (gRPC) | no        |
//! | AzureOpenAI | mostly OpenAI-shape; quirks for content-policy | partial     |
//!
//! OpenAI SDKs that drive the customer's retry strategy switch on
//! `error.code` (e.g. `rate_limit_exceeded` vs `insufficient_quota`)
//! and `error.type`. If we forward an Anthropic `rate_limit_error`
//! verbatim to a downstream OpenAI SDK, that retry logic doesn't fire.
//! This module maps each non-OpenAI upstream taxonomy to the OpenAI
//! taxonomy so the client-side SDK keeps working regardless of which
//! upstream the gateway routed to.
//!
//! Authoritative sources for the taxonomies:
//! - OpenAI: <https://platform.openai.com/docs/guides/error-codes/api-errors>
//! - Anthropic: <https://docs.anthropic.com/en/api/errors>
//! - Bedrock: <https://docs.aws.amazon.com/bedrock/latest/APIReference/CommonErrors.html>
//!   and per-operation error variants on `InvokeModelError`.
//! - Vertex / Google: <https://cloud.google.com/apis/design/errors>
//!   (canonical gRPC `Status.code` enum).

use aisix_gateway::{UpstreamErrorView, UpstreamWire};

use crate::error::ErrorBody;

/// Customer-visible OpenAI-shape envelope body for an upstream error.
///
/// Caller is responsible for gating on a 4xx status — 5xx and non-4xx
/// upstream errors stay in the generic `upstream_error` envelope.
pub(crate) fn render_openai_envelope(
    view: Option<&UpstreamErrorView>,
    wire: UpstreamWire,
    fallback_message: &str,
) -> ErrorBody {
    let Some(view) = view else {
        return generic(fallback_message);
    };
    let message = view
        .message
        .clone()
        .unwrap_or_else(|| fallback_message.to_string());
    let upstream_kind = view.kind.as_deref();
    let (kind, derived_code) = match wire {
        UpstreamWire::OpenAI => (
            upstream_kind
                .map(str::to_string)
                .unwrap_or_else(|| "upstream_error".to_string()),
            view.code.clone(),
        ),
        UpstreamWire::AzureOpenAI => translate_azure(upstream_kind),
        UpstreamWire::Anthropic => translate_anthropic(upstream_kind),
        UpstreamWire::Bedrock => translate_bedrock(upstream_kind),
        UpstreamWire::Vertex => translate_vertex(upstream_kind),
        UpstreamWire::Unknown => (
            upstream_kind
                .map(str::to_string)
                .unwrap_or_else(|| "upstream_error".to_string()),
            view.code.clone(),
        ),
    };
    ErrorBody {
        message,
        kind,
        param: view.param.clone(),
        // - OpenAI same-wire: pass through the upstream's `code` verbatim.
        // - AzureOpenAI: prefer the derived code when the translation
        //   table has an explicit mapping (e.g. `DeploymentNotFound`
        //   → `model_not_found`), otherwise pass through the upstream
        //   `code` (Azure shares OpenAI's taxonomy for the bulk of
        //   codes, so `rate_limit_exceeded` etc. should flow through).
        // - Anthropic / Bedrock / Vertex: the upstream `code` field is
        //   either absent or operator-leaky (Vertex numeric codes
        //   embed internal taxonomy) — only the derived code reaches
        //   the customer.
        code: match wire {
            UpstreamWire::OpenAI => view.code.clone(),
            UpstreamWire::AzureOpenAI => derived_code.or_else(|| view.code.clone()),
            _ => derived_code,
        },
    }
}

fn generic(message: &str) -> ErrorBody {
    ErrorBody {
        message: message.to_string(),
        kind: "upstream_error".to_string(),
        param: None,
        code: None,
    }
}

/// Anthropic `error.type` → OpenAI `(type, code)`. Reference:
/// <https://docs.anthropic.com/en/api/errors>. `permission_error` and
/// `request_too_large` are deliberately exhaustive — the upstream
/// reference impl this gateway is benchmarked against falls through to
/// a generic error on those two cases.
fn translate_anthropic(kind: Option<&str>) -> (String, Option<String>) {
    match kind {
        Some("invalid_request_error") => ("invalid_request_error".into(), None),
        Some("authentication_error") => (
            "invalid_request_error".into(),
            Some("invalid_api_key".into()),
        ),
        Some("permission_error") => (
            "invalid_request_error".into(),
            Some("permission_denied".into()),
        ),
        Some("not_found_error") => (
            "invalid_request_error".into(),
            Some("model_not_found".into()),
        ),
        Some("request_too_large") => (
            "invalid_request_error".into(),
            Some("request_too_large".into()),
        ),
        Some("rate_limit_error") => (
            "rate_limit_exceeded".into(),
            Some("rate_limit_exceeded".into()),
        ),
        Some("overloaded_error") => ("api_error".into(), Some("overloaded".into())),
        Some("api_error") => ("api_error".into(), None),
        _ => ("upstream_error".into(), None),
    }
}

/// AWS Bedrock `InvokeModelError` variant name → OpenAI `(type, code)`.
/// Reference: AWS SDK for Rust, `aws-sdk-bedrockruntime`'s generated
/// `InvokeModelError` enum, and
/// <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModel.html#API_runtime_InvokeModel_Errors>.
fn translate_bedrock(kind: Option<&str>) -> (String, Option<String>) {
    match kind {
        Some("ThrottlingException") => (
            "rate_limit_exceeded".into(),
            Some("rate_limit_exceeded".into()),
        ),
        Some("ServiceQuotaExceededException") => (
            "rate_limit_exceeded".into(),
            Some("insufficient_quota".into()),
        ),
        Some("ValidationException") => ("invalid_request_error".into(), None),
        Some("AccessDeniedException") => (
            "invalid_request_error".into(),
            Some("permission_denied".into()),
        ),
        Some("ResourceNotFoundException") => (
            "invalid_request_error".into(),
            Some("model_not_found".into()),
        ),
        Some("ModelNotReadyException") => ("api_error".into(), Some("model_not_ready".into())),
        Some("ModelTimeoutException") => ("api_error".into(), Some("timeout".into())),
        Some("ModelStreamErrorException") => ("api_error".into(), Some("stream_error".into())),
        Some("ModelErrorException") => ("api_error".into(), Some("model_error".into())),
        Some("InternalServerException") => ("api_error".into(), None),
        Some("ServiceUnavailableException") => ("api_error".into(), Some("overloaded".into())),
        _ => ("api_error".into(), None),
    }
}

/// Google canonical gRPC status code → OpenAI `(type, code)`. The
/// upstream `error.status` field carries the gRPC code as a string
/// (e.g. `"RESOURCE_EXHAUSTED"`). Reference:
/// <https://cloud.google.com/apis/design/errors> and the protobuf
/// `google.rpc.Code` enum.
fn translate_vertex(kind: Option<&str>) -> (String, Option<String>) {
    match kind {
        Some("RESOURCE_EXHAUSTED") => (
            "rate_limit_exceeded".into(),
            Some("rate_limit_exceeded".into()),
        ),
        Some("PERMISSION_DENIED") => (
            "invalid_request_error".into(),
            Some("permission_denied".into()),
        ),
        Some("UNAUTHENTICATED") => (
            "invalid_request_error".into(),
            Some("invalid_api_key".into()),
        ),
        Some("INVALID_ARGUMENT") => ("invalid_request_error".into(), None),
        Some("NOT_FOUND") => (
            "invalid_request_error".into(),
            Some("model_not_found".into()),
        ),
        Some("FAILED_PRECONDITION") | Some("OUT_OF_RANGE") | Some("ALREADY_EXISTS") => {
            ("invalid_request_error".into(), None)
        }
        Some("UNAVAILABLE") => ("api_error".into(), Some("overloaded".into())),
        Some("DEADLINE_EXCEEDED") => ("api_error".into(), Some("timeout".into())),
        Some("INTERNAL") | Some("ABORTED") | Some("CANCELLED") | Some("UNKNOWN") => {
            ("api_error".into(), None)
        }
        _ => ("api_error".into(), None),
    }
}

/// Azure OpenAI `error.code` → OpenAI `(type, code)`. Azure error
/// codes are mostly identical to OpenAI's, with a handful of
/// Azure-specific tokens. Reference: Azure OpenAI REST docs, error
/// codes section.
fn translate_azure(kind: Option<&str>) -> (String, Option<String>) {
    match kind {
        // Azure-specific deployment / content-policy codes.
        Some("DeploymentNotFound") => (
            "invalid_request_error".into(),
            Some("model_not_found".into()),
        ),
        Some("ResponsibleAIPolicyViolation") => (
            "invalid_request_error".into(),
            Some("content_policy_violation".into()),
        ),
        Some("content_filter") => (
            "invalid_request_error".into(),
            Some("content_policy_violation".into()),
        ),
        Some("invalid_encrypted_content") => (
            "invalid_request_error".into(),
            Some("invalid_encrypted_content".into()),
        ),
        // Everything else: Azure aligns with OpenAI taxonomy — pass
        // through the upstream kind as the OpenAI type, no derived
        // code (the caller will prefer any upstream-supplied `code`).
        Some(k) => (k.to_string(), None),
        None => ("upstream_error".into(), None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view(kind: &str) -> UpstreamErrorView {
        UpstreamErrorView {
            kind: Some(kind.into()),
            message: Some("upstream said hi".into()),
            code: None,
            param: None,
        }
    }

    #[test]
    fn anthropic_rate_limit_translates_to_openai_rate_limit_exceeded() {
        let v = view("rate_limit_error");
        let body = render_openai_envelope(Some(&v), UpstreamWire::Anthropic, "fallback");
        assert_eq!(body.kind, "rate_limit_exceeded");
        assert_eq!(body.code.as_deref(), Some("rate_limit_exceeded"));
        assert_eq!(body.message, "upstream said hi");
    }

    #[test]
    fn anthropic_overloaded_maps_to_api_error_with_overloaded_code() {
        let v = view("overloaded_error");
        let body = render_openai_envelope(Some(&v), UpstreamWire::Anthropic, "fb");
        assert_eq!(body.kind, "api_error");
        assert_eq!(body.code.as_deref(), Some("overloaded"));
    }

    #[test]
    fn anthropic_authentication_carries_invalid_api_key_code() {
        let v = view("authentication_error");
        let body = render_openai_envelope(Some(&v), UpstreamWire::Anthropic, "fb");
        assert_eq!(body.kind, "invalid_request_error");
        assert_eq!(body.code.as_deref(), Some("invalid_api_key"));
    }

    #[test]
    fn anthropic_permission_error_maps_to_permission_denied_code() {
        let v = view("permission_error");
        let body = render_openai_envelope(Some(&v), UpstreamWire::Anthropic, "fb");
        assert_eq!(body.kind, "invalid_request_error");
        assert_eq!(body.code.as_deref(), Some("permission_denied"));
    }

    #[test]
    fn anthropic_not_found_maps_to_model_not_found() {
        let v = view("not_found_error");
        let body = render_openai_envelope(Some(&v), UpstreamWire::Anthropic, "fb");
        assert_eq!(body.kind, "invalid_request_error");
        assert_eq!(body.code.as_deref(), Some("model_not_found"));
    }

    #[test]
    fn anthropic_unknown_falls_back_to_upstream_error() {
        let v = view("brand_new_anthropic_error_type");
        let body = render_openai_envelope(Some(&v), UpstreamWire::Anthropic, "fb");
        assert_eq!(body.kind, "upstream_error");
        assert!(body.code.is_none());
    }

    #[test]
    fn bedrock_throttling_translates_to_openai_rate_limit_exceeded() {
        let v = view("ThrottlingException");
        let body = render_openai_envelope(Some(&v), UpstreamWire::Bedrock, "fb");
        assert_eq!(body.kind, "rate_limit_exceeded");
        assert_eq!(body.code.as_deref(), Some("rate_limit_exceeded"));
    }

    #[test]
    fn bedrock_service_quota_exceeded_distinguishes_insufficient_quota() {
        // SDK retry logic should pick `insufficient_quota` over generic
        // `rate_limit_exceeded` because the recovery path differs
        // (quota lift vs backoff).
        let v = view("ServiceQuotaExceededException");
        let body = render_openai_envelope(Some(&v), UpstreamWire::Bedrock, "fb");
        assert_eq!(body.kind, "rate_limit_exceeded");
        assert_eq!(body.code.as_deref(), Some("insufficient_quota"));
    }

    #[test]
    fn bedrock_validation_maps_to_invalid_request_with_no_code() {
        let v = view("ValidationException");
        let body = render_openai_envelope(Some(&v), UpstreamWire::Bedrock, "fb");
        assert_eq!(body.kind, "invalid_request_error");
        assert!(body.code.is_none());
    }

    #[test]
    fn bedrock_access_denied_carries_permission_denied_code() {
        let v = view("AccessDeniedException");
        let body = render_openai_envelope(Some(&v), UpstreamWire::Bedrock, "fb");
        assert_eq!(body.kind, "invalid_request_error");
        assert_eq!(body.code.as_deref(), Some("permission_denied"));
    }

    #[test]
    fn bedrock_unhandled_falls_back_to_api_error() {
        let v = view("BrandNewBedrockException");
        let body = render_openai_envelope(Some(&v), UpstreamWire::Bedrock, "fb");
        assert_eq!(body.kind, "api_error");
        assert!(body.code.is_none());
    }

    #[test]
    fn vertex_resource_exhausted_translates_to_openai_rate_limit_exceeded() {
        let v = view("RESOURCE_EXHAUSTED");
        let body = render_openai_envelope(Some(&v), UpstreamWire::Vertex, "fb");
        assert_eq!(body.kind, "rate_limit_exceeded");
        assert_eq!(body.code.as_deref(), Some("rate_limit_exceeded"));
    }

    #[test]
    fn vertex_permission_denied_maps_to_permission_denied_code() {
        let v = view("PERMISSION_DENIED");
        let body = render_openai_envelope(Some(&v), UpstreamWire::Vertex, "fb");
        assert_eq!(body.kind, "invalid_request_error");
        assert_eq!(body.code.as_deref(), Some("permission_denied"));
    }

    #[test]
    fn vertex_unauthenticated_maps_to_invalid_api_key_code() {
        let v = view("UNAUTHENTICATED");
        let body = render_openai_envelope(Some(&v), UpstreamWire::Vertex, "fb");
        assert_eq!(body.kind, "invalid_request_error");
        assert_eq!(body.code.as_deref(), Some("invalid_api_key"));
    }

    #[test]
    fn vertex_unavailable_maps_to_api_error_with_overloaded_code() {
        let v = view("UNAVAILABLE");
        let body = render_openai_envelope(Some(&v), UpstreamWire::Vertex, "fb");
        assert_eq!(body.kind, "api_error");
        assert_eq!(body.code.as_deref(), Some("overloaded"));
    }

    #[test]
    fn vertex_deadline_exceeded_maps_to_timeout_code() {
        let v = view("DEADLINE_EXCEEDED");
        let body = render_openai_envelope(Some(&v), UpstreamWire::Vertex, "fb");
        assert_eq!(body.kind, "api_error");
        assert_eq!(body.code.as_deref(), Some("timeout"));
    }

    #[test]
    fn azure_deployment_not_found_maps_to_model_not_found() {
        let v = view("DeploymentNotFound");
        let body = render_openai_envelope(Some(&v), UpstreamWire::AzureOpenAI, "fb");
        assert_eq!(body.kind, "invalid_request_error");
        assert_eq!(body.code.as_deref(), Some("model_not_found"));
    }

    #[test]
    fn azure_content_policy_violation_translates_through_inner_error() {
        // Azure surfaces ResponsibleAIPolicyViolation under
        // inner_error.code; the bridge parser lifts it to the top-level
        // kind, and this translation rewrites it to the OpenAI string
        // code that SDKs recognise.
        let v = view("ResponsibleAIPolicyViolation");
        let body = render_openai_envelope(Some(&v), UpstreamWire::AzureOpenAI, "fb");
        assert_eq!(body.kind, "invalid_request_error");
        assert_eq!(body.code.as_deref(), Some("content_policy_violation"));
    }

    #[test]
    fn azure_unknown_kind_passes_through_when_openai_compatible() {
        // Azure shares OpenAI's taxonomy for the vast majority of error
        // codes; new ones should pass through rather than collapse to
        // a generic `upstream_error`.
        let v = view("some_future_openai_compat_code");
        let body = render_openai_envelope(Some(&v), UpstreamWire::AzureOpenAI, "fb");
        assert_eq!(body.kind, "some_future_openai_compat_code");
    }

    #[test]
    fn openai_same_wire_preserves_upstream_code_and_param() {
        // The same-wire path treats the upstream as authoritative —
        // `code` and `param` flow through unchanged, including codes
        // that aren't in any table (forward-compat for OpenAI taxonomy
        // additions).
        let v = UpstreamErrorView {
            kind: Some("rate_limit_exceeded".into()),
            message: Some("hi".into()),
            code: Some("custom_code_added_yesterday".into()),
            param: Some("model".into()),
        };
        let body = render_openai_envelope(Some(&v), UpstreamWire::OpenAI, "fb");
        assert_eq!(body.kind, "rate_limit_exceeded");
        assert_eq!(body.code.as_deref(), Some("custom_code_added_yesterday"));
        assert_eq!(body.param.as_deref(), Some("model"));
    }

    #[test]
    fn missing_view_uses_fallback_message_and_generic_kind() {
        let body = render_openai_envelope(None, UpstreamWire::Anthropic, "raw upstream text");
        assert_eq!(body.kind, "upstream_error");
        assert_eq!(body.message, "raw upstream text");
        assert!(body.code.is_none());
    }

    #[test]
    fn missing_parsed_message_falls_back_to_raw_message() {
        let v = UpstreamErrorView {
            kind: Some("rate_limit_error".into()),
            message: None,
            code: None,
            param: None,
        };
        let body = render_openai_envelope(Some(&v), UpstreamWire::Anthropic, "raw fallback");
        assert_eq!(body.kind, "rate_limit_exceeded");
        assert_eq!(body.message, "raw fallback");
    }
}
