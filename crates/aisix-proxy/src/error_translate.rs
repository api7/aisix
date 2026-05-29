//! Cross-wire upstream-error `code` derivation.
//!
//! Each upstream provider emits a different error taxonomy:
//!
//! | Wire        | Has structured `code`? | Native `type` examples                    |
//! |-------------|------------------------|-------------------------------------------|
//! | OpenAI      | yes                    | `rate_limit_exceeded`, `invalid_api_key`  |
//! | Anthropic   | no                     | `rate_limit_error`, `overloaded_error`    |
//! | Bedrock     | no                     | `ThrottlingException`, `ValidationException` |
//! | Vertex      | no                     | `RESOURCE_EXHAUSTED`, `PERMISSION_DENIED` (gRPC) |
//! | AzureOpenAI | partial                | mostly OpenAI-shape; quirks for content-policy |
//!
//! Customer SDKs that drive retry strategy switch on `error.code`
//! (e.g. `rate_limit_exceeded` vs `insufficient_quota`). When the
//! upstream doesn't expose a stable string `code` (Anthropic, Bedrock,
//! Vertex), this module derives one from the upstream `type` so the
//! client-side SDK keeps working regardless of which upstream the
//! gateway routed to.
//!
//! Per issue #327, `error.type` itself is **not** derived per-upstream
//! — the DP renders the stable token `"upstream_error"` for any
//! upstream-originated error. The DP acts as a normalising gateway:
//! `error.type` is its closed taxonomy; the upstream's private
//! taxonomy (`upstream_test_fixture`, `ValidationException`, etc.)
//! never reaches the customer. SDKs branch on `error.type ==
//! "upstream_error"` for upstream-class detection and on
//! `error.code` for granular retry routing.
//!
//! Authoritative sources for the input taxonomies:
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
    let derived_code = match wire {
        UpstreamWire::OpenAI | UpstreamWire::Unknown => view.code.clone(),
        UpstreamWire::AzureOpenAI => derive_azure_code(upstream_kind),
        UpstreamWire::Anthropic => derive_anthropic_code(upstream_kind),
        UpstreamWire::Bedrock => derive_bedrock_code(upstream_kind),
        UpstreamWire::Vertex => derive_vertex_code(upstream_kind),
    };
    ErrorBody {
        message,
        // Issue #327: `error.type` is the DP's stable taxonomy, NOT
        // the upstream's. Customers branch on
        // `error.type == "upstream_error"` for upstream-class
        // detection; SDK retry granularity comes from `error.code`.
        kind: UPSTREAM_ERROR_TYPE.to_string(),
        param: view.param.clone(),
        // - OpenAI same-wire: pass through the upstream's `code` verbatim.
        // - AzureOpenAI: prefer the derived code when the table has an
        //   explicit Azure-specific mapping (e.g. `DeploymentNotFound`
        //   → `model_not_found`), otherwise pass through the upstream
        //   `code` (Azure shares OpenAI's taxonomy for most codes, so
        //   `rate_limit_exceeded` etc. flow through).
        // - Anthropic / Bedrock / Vertex: the upstream `code` field is
        //   either absent or operator-leaky (Vertex numeric codes
        //   embed internal taxonomy) — only the derived code reaches
        //   the customer.
        code: match wire {
            UpstreamWire::OpenAI => view.code.clone(),
            UpstreamWire::AzureOpenAI => derived_code.or_else(|| view.code.clone()),
            _ => derived_code,
        },
        budget: None,
    }
}

/// DP-stable `error.type` token surfaced for any upstream-originated
/// error. See module docstring + issue #327.
const UPSTREAM_ERROR_TYPE: &str = "upstream_error";

fn generic(message: &str) -> ErrorBody {
    ErrorBody {
        message: message.to_string(),
        kind: UPSTREAM_ERROR_TYPE.to_string(),
        param: None,
        code: None,
        budget: None,
    }
}

/// Anthropic `error.type` → OpenAI string `code`. Reference:
/// <https://docs.anthropic.com/en/api/errors>. The upstream `type`
/// itself is not propagated (see issue #327) — only a derived
/// OpenAI-shape `code` reaches the customer, so SDK retry logic that
/// switches on `error.code` works regardless of which upstream the
/// gateway routed to.
fn derive_anthropic_code(kind: Option<&str>) -> Option<String> {
    match kind? {
        "authentication_error" => Some("invalid_api_key".into()),
        "permission_error" => Some("permission_denied".into()),
        "not_found_error" => Some("model_not_found".into()),
        "request_too_large" => Some("request_too_large".into()),
        "rate_limit_error" => Some("rate_limit_exceeded".into()),
        "overloaded_error" => Some("overloaded".into()),
        // `invalid_request_error`, `api_error`, and unknown values
        // have no clean OpenAI string-code counterpart.
        _ => None,
    }
}

/// AWS Bedrock `InvokeModelError` variant name → OpenAI string `code`.
/// Reference: AWS SDK for Rust, `aws-sdk-bedrockruntime`'s generated
/// `InvokeModelError` enum, and
/// <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModel.html#API_runtime_InvokeModel_Errors>.
fn derive_bedrock_code(kind: Option<&str>) -> Option<String> {
    match kind? {
        "ThrottlingException" => Some("rate_limit_exceeded".into()),
        "ServiceQuotaExceededException" => Some("insufficient_quota".into()),
        "AccessDeniedException" => Some("permission_denied".into()),
        "ResourceNotFoundException" => Some("model_not_found".into()),
        "ModelNotReadyException" => Some("model_not_ready".into()),
        "ModelTimeoutException" => Some("timeout".into()),
        "ModelStreamErrorException" => Some("stream_error".into()),
        "ModelErrorException" => Some("model_error".into()),
        "ServiceUnavailableException" => Some("overloaded".into()),
        // `ValidationException`, `InternalServerException`, and
        // unhandled variants have no clean OpenAI string-code
        // counterpart.
        _ => None,
    }
}

/// Google canonical gRPC status code → OpenAI string `code`. The
/// upstream `error.status` field carries the gRPC code as a string
/// (e.g. `"RESOURCE_EXHAUSTED"`). Reference:
/// <https://cloud.google.com/apis/design/errors> and the protobuf
/// `google.rpc.Code` enum.
fn derive_vertex_code(kind: Option<&str>) -> Option<String> {
    match kind? {
        "RESOURCE_EXHAUSTED" => Some("rate_limit_exceeded".into()),
        "PERMISSION_DENIED" => Some("permission_denied".into()),
        "UNAUTHENTICATED" => Some("invalid_api_key".into()),
        "NOT_FOUND" => Some("model_not_found".into()),
        "UNAVAILABLE" => Some("overloaded".into()),
        "DEADLINE_EXCEEDED" => Some("timeout".into()),
        // `INVALID_ARGUMENT`, `FAILED_PRECONDITION`, `INTERNAL`,
        // `ABORTED`, `CANCELLED`, `UNKNOWN` have no clean OpenAI
        // string-code counterpart.
        _ => None,
    }
}

/// Azure OpenAI `error.code` → OpenAI string `code`. Azure error
/// codes are mostly identical to OpenAI's, with a handful of
/// Azure-specific tokens that get rewritten to the OpenAI equivalent.
/// Reference: Azure OpenAI REST docs, error codes section.
fn derive_azure_code(kind: Option<&str>) -> Option<String> {
    match kind? {
        "DeploymentNotFound" => Some("model_not_found".into()),
        "ResponsibleAIPolicyViolation" | "content_filter" => {
            Some("content_policy_violation".into())
        }
        "invalid_encrypted_content" => Some("invalid_encrypted_content".into()),
        // For OpenAI-compat codes (e.g. `rate_limit_exceeded`), the
        // renderer falls back to `view.code`, which already carries
        // the upstream code; emitting `None` here lets that fallback
        // win.
        _ => None,
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
    fn anthropic_rate_limit_derives_rate_limit_exceeded_code() {
        let v = view("rate_limit_error");
        let body = render_openai_envelope(Some(&v), UpstreamWire::Anthropic, "fallback");
        // Issue #327: `kind` is the DP-stable taxonomy
        // (`upstream_error`); SDK retry routing uses `code`.
        assert_eq!(body.kind, "upstream_error");
        assert_eq!(body.code.as_deref(), Some("rate_limit_exceeded"));
        assert_eq!(body.message, "upstream said hi");
    }

    #[test]
    fn anthropic_overloaded_derives_overloaded_code() {
        let v = view("overloaded_error");
        let body = render_openai_envelope(Some(&v), UpstreamWire::Anthropic, "fb");
        // Issue #327: `kind` is the DP-stable taxonomy regardless of
        // which upstream produced the error.
        assert_eq!(body.kind, "upstream_error");
        assert_eq!(body.code.as_deref(), Some("overloaded"));
    }

    #[test]
    fn anthropic_authentication_derives_invalid_api_key_code() {
        let v = view("authentication_error");
        let body = render_openai_envelope(Some(&v), UpstreamWire::Anthropic, "fb");
        assert_eq!(body.kind, "upstream_error");
        assert_eq!(body.code.as_deref(), Some("invalid_api_key"));
    }

    #[test]
    fn anthropic_permission_error_derives_permission_denied_code() {
        let v = view("permission_error");
        let body = render_openai_envelope(Some(&v), UpstreamWire::Anthropic, "fb");
        assert_eq!(body.kind, "upstream_error");
        assert_eq!(body.code.as_deref(), Some("permission_denied"));
    }

    #[test]
    fn anthropic_not_found_derives_model_not_found_code() {
        let v = view("not_found_error");
        let body = render_openai_envelope(Some(&v), UpstreamWire::Anthropic, "fb");
        assert_eq!(body.kind, "upstream_error");
        assert_eq!(body.code.as_deref(), Some("model_not_found"));
    }

    #[test]
    fn anthropic_unknown_kind_yields_null_code_under_upstream_error_type() {
        let v = view("brand_new_anthropic_error_type");
        let body = render_openai_envelope(Some(&v), UpstreamWire::Anthropic, "fb");
        assert_eq!(body.kind, "upstream_error");
        assert!(body.code.is_none());
    }

    #[test]
    fn bedrock_throttling_derives_rate_limit_exceeded_code() {
        let v = view("ThrottlingException");
        let body = render_openai_envelope(Some(&v), UpstreamWire::Bedrock, "fb");
        assert_eq!(body.kind, "upstream_error");
        assert_eq!(body.code.as_deref(), Some("rate_limit_exceeded"));
    }

    #[test]
    fn bedrock_service_quota_exceeded_distinguishes_insufficient_quota() {
        // SDK retry logic should pick `insufficient_quota` over generic
        // `rate_limit_exceeded` because the recovery path differs
        // (quota lift vs backoff).
        let v = view("ServiceQuotaExceededException");
        let body = render_openai_envelope(Some(&v), UpstreamWire::Bedrock, "fb");
        assert_eq!(body.kind, "upstream_error");
        assert_eq!(body.code.as_deref(), Some("insufficient_quota"));
    }

    #[test]
    fn bedrock_validation_yields_null_code() {
        let v = view("ValidationException");
        let body = render_openai_envelope(Some(&v), UpstreamWire::Bedrock, "fb");
        assert_eq!(body.kind, "upstream_error");
        assert!(body.code.is_none());
    }

    #[test]
    fn bedrock_access_denied_derives_permission_denied_code() {
        let v = view("AccessDeniedException");
        let body = render_openai_envelope(Some(&v), UpstreamWire::Bedrock, "fb");
        assert_eq!(body.kind, "upstream_error");
        assert_eq!(body.code.as_deref(), Some("permission_denied"));
    }

    #[test]
    fn bedrock_unhandled_yields_null_code() {
        let v = view("BrandNewBedrockException");
        let body = render_openai_envelope(Some(&v), UpstreamWire::Bedrock, "fb");
        assert_eq!(body.kind, "upstream_error");
        assert!(body.code.is_none());
    }

    #[test]
    fn vertex_resource_exhausted_derives_rate_limit_exceeded_code() {
        let v = view("RESOURCE_EXHAUSTED");
        let body = render_openai_envelope(Some(&v), UpstreamWire::Vertex, "fb");
        assert_eq!(body.kind, "upstream_error");
        assert_eq!(body.code.as_deref(), Some("rate_limit_exceeded"));
    }

    #[test]
    fn vertex_permission_denied_derives_permission_denied_code() {
        let v = view("PERMISSION_DENIED");
        let body = render_openai_envelope(Some(&v), UpstreamWire::Vertex, "fb");
        assert_eq!(body.kind, "upstream_error");
        assert_eq!(body.code.as_deref(), Some("permission_denied"));
    }

    #[test]
    fn vertex_unauthenticated_derives_invalid_api_key_code() {
        let v = view("UNAUTHENTICATED");
        let body = render_openai_envelope(Some(&v), UpstreamWire::Vertex, "fb");
        assert_eq!(body.kind, "upstream_error");
        assert_eq!(body.code.as_deref(), Some("invalid_api_key"));
    }

    #[test]
    fn vertex_unavailable_derives_overloaded_code() {
        let v = view("UNAVAILABLE");
        let body = render_openai_envelope(Some(&v), UpstreamWire::Vertex, "fb");
        assert_eq!(body.kind, "upstream_error");
        assert_eq!(body.code.as_deref(), Some("overloaded"));
    }

    #[test]
    fn vertex_deadline_exceeded_derives_timeout_code() {
        let v = view("DEADLINE_EXCEEDED");
        let body = render_openai_envelope(Some(&v), UpstreamWire::Vertex, "fb");
        assert_eq!(body.kind, "upstream_error");
        assert_eq!(body.code.as_deref(), Some("timeout"));
    }

    #[test]
    fn azure_deployment_not_found_derives_model_not_found_code() {
        let v = view("DeploymentNotFound");
        let body = render_openai_envelope(Some(&v), UpstreamWire::AzureOpenAI, "fb");
        assert_eq!(body.kind, "upstream_error");
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
        assert_eq!(body.kind, "upstream_error");
        assert_eq!(body.code.as_deref(), Some("content_policy_violation"));
    }

    #[test]
    fn azure_openai_compatible_code_falls_through_to_upstream_value() {
        // Azure shares OpenAI's taxonomy for the vast majority of error
        // codes; for unknown codes the derived value is None, so the
        // renderer falls back to view.code — which the Azure bridge
        // populates from the upstream `error.code` field.
        let v = UpstreamErrorView {
            kind: Some("rate_limit_exceeded".into()),
            message: Some("hi".into()),
            code: Some("rate_limit_exceeded".into()),
            param: None,
        };
        let body = render_openai_envelope(Some(&v), UpstreamWire::AzureOpenAI, "fb");
        assert_eq!(body.kind, "upstream_error");
        assert_eq!(body.code.as_deref(), Some("rate_limit_exceeded"));
    }

    #[test]
    fn openai_same_wire_preserves_upstream_code_and_param() {
        // The same-wire path treats the upstream as authoritative —
        // `code` and `param` flow through unchanged, including codes
        // that aren't in any table (forward-compat for OpenAI taxonomy
        // additions). `kind` is the DP-stable token, not the upstream's
        // `error.type`.
        let v = UpstreamErrorView {
            kind: Some("rate_limit_exceeded".into()),
            message: Some("hi".into()),
            code: Some("custom_code_added_yesterday".into()),
            param: Some("model".into()),
        };
        let body = render_openai_envelope(Some(&v), UpstreamWire::OpenAI, "fb");
        assert_eq!(body.kind, "upstream_error");
        assert_eq!(body.code.as_deref(), Some("custom_code_added_yesterday"));
        assert_eq!(body.param.as_deref(), Some("model"));
    }

    #[test]
    fn missing_view_uses_fallback_message_and_upstream_error_kind() {
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
        assert_eq!(body.kind, "upstream_error");
        assert_eq!(body.code.as_deref(), Some("rate_limit_exceeded"));
        assert_eq!(body.message, "raw fallback");
    }
}
