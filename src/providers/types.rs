use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("Not implemented")]
    NotYetImplemented,

    #[error("API error {0}: {1}")]
    ServiceError(http::StatusCode, String),

    #[error("Request error: {0}")]
    RequestError(#[from] reqwest::Error),

    #[error("Failed to parse response: {0}")]
    CodecError(#[from] serde_json::Error),
}
