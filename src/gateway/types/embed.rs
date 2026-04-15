use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Transport body abstractions for embedding gateway calls.
#[derive(Debug, Clone)]
pub enum EmbedRequestBody {
    Json(Value),
    Binary(Bytes),
}

/// Parsed response body abstractions for embedding gateway calls.
#[derive(Debug, Clone)]
pub enum EmbedResponseBody {
    Json(Value),
    Binary(Bytes),
}

/// Embedding input that may be either one string or multiple strings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum OneOrMany<T> {
    One(T),
    Many(Vec<T>),
}

/// OpenAI-compatible embeddings request payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingRequest {
    pub input: OneOrMany<String>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

/// A single embedding vector entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: Option<i32>,
}

/// Token usage for an embedding response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

/// OpenAI-compatible embeddings response payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Option<EmbeddingUsage>,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{EmbeddingRequest, EmbeddingResponse, OneOrMany};

    #[test]
    fn embedding_request_round_trips() {
        let request: EmbeddingRequest = serde_json::from_value(json!({
            "model": "text-embedding-3-large",
            "input": ["hello", "world"],
            "dimensions": 256,
            "encoding_format": "float",
            "user": "user-123"
        }))
        .unwrap();

        assert_eq!(request.model, "text-embedding-3-large");
        assert!(matches!(request.input, OneOrMany::Many(_)));

        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(value["dimensions"], 256);
        assert_eq!(value["encoding_format"], "float");
    }

    #[test]
    fn embedding_request_single_input_round_trips() {
        let request: EmbeddingRequest = serde_json::from_value(json!({
            "model": "text-embedding-3-large",
            "input": "hello"
        }))
        .unwrap();

        assert!(matches!(request.input, OneOrMany::One(_)));

        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(value["input"], "hello");
    }

    #[test]
    fn embedding_response_round_trips() {
        let response: EmbeddingResponse = serde_json::from_value(json!({
            "object": "list",
            "data": [{
                "object": "embedding",
                "embedding": [0.1, 0.2],
                "index": 0
            }],
            "model": "text-embedding-3-large",
            "usage": {
                "prompt_tokens": 8,
                "total_tokens": 8
            }
        }))
        .unwrap();

        assert_eq!(response.data.len(), 1);
        assert_eq!(response.usage.unwrap().prompt_tokens, 8);
    }
}
