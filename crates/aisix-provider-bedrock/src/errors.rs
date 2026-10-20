//! Bedrock error classification, shared by every Bedrock operation the
//! bridge calls (InvokeModel, Converse, ConverseStream, the runtime's
//! OpenAI-compatible Chat Completions endpoint, embeddings).
//!
//! The caller receives the upstream's own error message. The exception name
//! decides how the gateway treats the failure: a model timeout is a timeout,
//! a model-side processing failure is an upstream fault worth another
//! target, and a context-window overflow carries the `context_length_exceeded`
//! code clients act on.
//!
//! Reference: <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModel.html#API_runtime_InvokeModel_Errors>

use std::time::{Duration, Instant};

use aisix_gateway::{
    truncate_lossy, BridgeError, UpstreamErrorView, UpstreamWire, MAX_UPSTREAM_ERROR_MESSAGE_BYTES,
};
use aws_sdk_bedrockruntime::error::SdkError;
use aws_smithy_runtime_api::client::result::ServiceError;
use aws_smithy_types::error::metadata::ProvideErrorMetadata;

/// Substrings Bedrock's publishers use for a prompt that does not fit the
/// model's context window.
const CONTEXT_WINDOW_MARKERS: &[&str] = &[
    "too many tokens",
    "expected maxLength:",
    "Input is too long",
    "prompt is too long",
    "prompt: length: 1..",
    "Too many input tokens",
];

pub(crate) const CONTEXT_LENGTH_EXCEEDED: &str = "context_length_exceeded";

fn context_window_code(message: &str) -> Option<String> {
    CONTEXT_WINDOW_MARKERS
        .iter()
        .any(|m| message.contains(m))
        .then(|| CONTEXT_LENGTH_EXCEEDED.to_string())
}

/// The AWS exception name, without the namespace suffix the
/// `x-amzn-errortype` header carries (`ValidationException:http://…`).
pub(crate) fn exception_name(raw: &str) -> &str {
    raw.split(':').next().unwrap_or(raw).trim()
}

/// An upstream HTTP error, classified by its AWS exception name.
pub(crate) fn upstream_error(
    status: u16,
    kind: Option<String>,
    message: String,
    retry_after: Option<Duration>,
    started: Instant,
) -> BridgeError {
    let message = truncate_lossy(&message, MAX_UPSTREAM_ERROR_MESSAGE_BYTES);
    let kind = kind.map(|k| exception_name(&k).to_string());
    if kind.as_deref() == Some("ModelTimeoutException") {
        return BridgeError::Timeout {
            elapsed_ms: started.elapsed().as_millis() as u64,
            cause: message,
        };
    }
    // A model that failed while processing the request is an upstream
    // fault, not the caller's: another target can answer it.
    let status = if kind.as_deref() == Some("ModelErrorException") {
        500
    } else {
        status
    };
    let code = context_window_code(&message);
    BridgeError::UpstreamStatus {
        status,
        parsed: Some(Box::new(UpstreamErrorView {
            kind,
            message: (!message.is_empty()).then(|| message.clone()),
            code,
            param: None,
        })),
        message,
        wire: UpstreamWire::Bedrock,
        retry_after,
    }
}

/// A modeled exception delivered inside a committed event stream.
pub(crate) fn in_band_error(
    status: Option<u16>,
    kind: Option<&str>,
    message: String,
) -> BridgeError {
    let message = truncate_lossy(&message, MAX_UPSTREAM_ERROR_MESSAGE_BYTES);
    let code = context_window_code(&message);
    BridgeError::UpstreamInBand {
        status,
        parsed: Some(Box::new(UpstreamErrorView {
            kind: kind.map(str::to_string),
            message: (!message.is_empty()).then(|| message.clone()),
            code,
            param: None,
        })),
        message,
        wire: UpstreamWire::Bedrock,
    }
}

/// An AWS SDK operation failure.
pub(crate) fn sdk_error<E>(
    err: SdkError<E, aws_smithy_runtime_api::http::Response>,
    started: Instant,
    deadline: Option<Duration>,
) -> BridgeError
where
    E: std::fmt::Debug + ProvideErrorMetadata,
{
    // Deadline-elapsed wins over upstream classification — surfacing an
    // upstream status on a timed-out request would hide the configured
    // deadline.
    if let Some(d) = deadline {
        if started.elapsed() >= d {
            return BridgeError::Timeout {
                elapsed_ms: started.elapsed().as_millis() as u64,
                cause: String::new(),
            };
        }
    }
    match err {
        SdkError::ServiceError(svc) => service_error(svc, started),
        SdkError::TimeoutError(_) => {
            let elapsed_ms = started.elapsed().as_millis() as u64;
            let reported = if elapsed_ms > 0 {
                elapsed_ms
            } else {
                deadline.map(|d| d.as_millis() as u64).unwrap_or(0)
            };
            BridgeError::Timeout {
                elapsed_ms: reported,
                cause: String::new(),
            }
        }
        other => BridgeError::Transport(format!("{other}")),
    }
}

fn service_error<E>(
    svc: ServiceError<E, aws_smithy_runtime_api::http::Response>,
    started: Instant,
) -> BridgeError
where
    E: ProvideErrorMetadata,
{
    let kind = svc.err().meta().code().map(str::to_string);
    let raw = svc.raw();
    // The SDK reads the message out of an error it can model; for one it
    // cannot, the body still carries it.
    let message = match svc.err().meta().message() {
        Some(m) if !m.is_empty() => m.to_string(),
        _ => raw
            .body()
            .bytes()
            .map(aisix_gateway::upstream_error_message)
            .unwrap_or_default(),
    };
    let status = raw.status().as_u16();
    let mut hdrs = http::HeaderMap::new();
    for (k, v) in raw.headers() {
        if let (Ok(name), Ok(val)) = (
            http::HeaderName::from_bytes(k.as_bytes()),
            http::HeaderValue::from_str(v),
        ) {
            hdrs.insert(name, val);
        }
    }
    let retry_after = aisix_gateway::parse_retry_after(&hdrs);
    upstream_error(status, kind, message, retry_after, started)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_error_is_an_upstream_fault() {
        let e = upstream_error(
            424,
            Some("ModelErrorException".into()),
            "the model failed".into(),
            None,
            Instant::now(),
        );
        assert!(
            matches!(e, BridgeError::UpstreamStatus { status: 500, .. }),
            "{e:?}"
        );
    }

    #[test]
    fn model_timeout_is_a_timeout() {
        let e = upstream_error(
            408,
            Some("ModelTimeoutException:http://internal".into()),
            "took too long".into(),
            None,
            Instant::now(),
        );
        assert!(matches!(e, BridgeError::Timeout { .. }), "{e:?}");
    }

    #[test]
    fn context_overflow_carries_the_code_and_the_upstream_text() {
        let e = upstream_error(
            400,
            Some("ValidationException".into()),
            "Input is too long for requested model.".into(),
            None,
            Instant::now(),
        );
        let BridgeError::UpstreamStatus {
            message, parsed, ..
        } = e
        else {
            panic!("expected UpstreamStatus");
        };
        assert_eq!(message, "Input is too long for requested model.");
        assert_eq!(
            parsed.unwrap().code.as_deref(),
            Some(CONTEXT_LENGTH_EXCEEDED)
        );
    }
}
