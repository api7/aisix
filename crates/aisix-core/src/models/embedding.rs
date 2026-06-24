//! Embedding-modality config attached to a [`Model`](super::Model).
//!
//! A direct Model that carries an `embedding` block is an embedding model:
//! it keeps the direct-upstream triple (`provider` / `model_name` /
//! `provider_key_id`) pointing at an OpenAI-compatible `/v1/embeddings`
//! endpoint, and the block records the modality metadata the gateway needs
//! to use the vectors — the output `dimensions` and whether the endpoint
//! already returns L2-normalized vectors.
//!
//! Such a model can be called directly via `/v1/embeddings` and referenced
//! by a [`Semantic`](super::Semantic) router as its `embedding_model`. The
//! presence of this block is what marks the modality; it does not change
//! the dispatch kind (the model stays `Direct`).

use serde::{Deserialize, Serialize};

fn default_normalize() -> bool {
    true
}

/// Embedding-modality metadata for a direct Model.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingConfig {
    /// Output vector dimensionality. Used to validate vectors, key the
    /// example-vector cache, and (for endpoints that support it) request a
    /// reduced output size.
    #[schemars(range(min = 1))]
    pub dimensions: u32,
    /// Whether the endpoint already returns L2-normalized vectors. When
    /// `false`, the gateway normalizes before computing cosine similarity.
    /// Defaults to `true`.
    #[serde(default = "default_normalize")]
    pub normalize: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialises_full_embedding_block() {
        let e: EmbeddingConfig =
            serde_json::from_str(r#"{"dimensions": 1024, "normalize": false}"#).unwrap();
        assert_eq!(e.dimensions, 1024);
        assert!(!e.normalize);
    }

    #[test]
    fn normalize_defaults_to_true() {
        let e: EmbeddingConfig = serde_json::from_str(r#"{"dimensions": 1536}"#).unwrap();
        assert_eq!(e.dimensions, 1536);
        assert!(e.normalize);
    }

    #[test]
    fn rejects_unknown_field() {
        let r: Result<EmbeddingConfig, _> =
            serde_json::from_str(r#"{"dimensions": 1024, "bogus": true}"#);
        assert!(r.is_err());
    }
}
