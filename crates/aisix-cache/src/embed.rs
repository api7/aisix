//! Embedding helper for the pgvector semantic cache (Stage 4b).
//!
//! When the proxy matches a `cache_policy` with `backend = pgvector`,
//! we need a vector to look up against. The embedding is computed on
//! the data plane (so embedding-provider credentials stay on DP per
//! the Stage 4b design note); this module is the glue between
//! `aisix_proxy::chat` and the existing `Bridge::embed` surface.
//!
//! Provider key resolution: we pick the first OpenAI Model in the
//! current snapshot and reuse its provider_config (api_key + api_base).
//! Per the Stage 4b decision (option C in the design doc), there is
//! no separate `embedding_provider_key_id` field on `CachePolicy`
//! today — that's a follow-up if operators need to bill embedding
//! against a different key.
//!
//! Failure mode is fail-open: any error from this module surfaces to
//! the proxy as `EmbedError`, which the chat handler maps to
//! `CacheStatus::Disabled` + skip-the-lookup. The caller's request
//! still reaches the upstream, just without semantic-cache benefit.

use aisix_core::models::{Model, Provider};
use aisix_core::resource::ResourceEntry;
use aisix_core::AisixSnapshot;
use aisix_gateway::{BridgeContext, EmbeddingRequest, Hub};
use std::sync::Arc;

/// Errors the embedder can surface. The proxy logs these at
/// `tracing::warn!` and then falls open — the caller's request still
/// reaches the upstream, just without a cache lookup.
#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    /// No Model in the env snapshot has provider == OpenAI. Operator
    /// needs to add an OpenAI provider_key + Model to the env before
    /// the pgvector backend can produce embeddings.
    #[error("no OpenAI model in snapshot to source embedding credentials from")]
    NoOpenAiModel,
    /// `Bridge::embed` failed (transport error, upstream error,
    /// decode failure). Carries the upstream message for the warn
    /// log so an operator can debug why semantic caching went dark.
    #[error("embedding bridge call failed: {0}")]
    Bridge(String),
    /// The provider returned a successful response but with no
    /// embedding data (e.g. empty `data: []`). Shouldn't happen
    /// against OpenAI but the wire allows it.
    #[error("embedding response had no data")]
    EmptyResponse,
}

/// Compute an embedding for the prompt text using the env's first
/// OpenAI Model as the credentials source.
///
/// `embedding_model` is the model name from the policy (e.g.
/// `"text-embedding-3-small"`). The OpenAI Model's api_key + api_base
/// are reused — the policy's embedding model swaps in for the chat
/// model's name on the embeddings endpoint.
///
/// `request_id` is the chat request's id, threaded through so the
/// embedding call shows up under the same id in upstream logs.
pub async fn embed_prompt(
    snapshot: &AisixSnapshot,
    hub: &Hub,
    embedding_model: &str,
    prompt_text: &str,
    request_id: &str,
) -> Result<Vec<f32>, EmbedError> {
    let openai_model = first_openai_model(snapshot).ok_or(EmbedError::NoOpenAiModel)?;
    let bridge = hub
        .get(Provider::Openai)
        .ok_or(EmbedError::NoOpenAiModel)?;

    // The chat model carries provider_config (api_key + api_base);
    // we override the model name for the embeddings call so the
    // bridge sends the policy's embedding_model on the wire instead
    // of the chat model name.
    let mut model_for_embed = openai_model.value.clone();
    model_for_embed.model = format!("openai/{embedding_model}");
    model_for_embed.name = format!("__embedder__{embedding_model}");
    let model_arc = Arc::new(model_for_embed);
    let ctx = BridgeContext::new(request_id, model_arc);

    let req = EmbeddingRequest {
        model: embedding_model.to_string(),
        input: vec![prompt_text.to_string()],
        encoding_format: None,
        dimensions: None,
    };
    let resp = bridge
        .embed(&req, &ctx)
        .await
        .map_err(|e| EmbedError::Bridge(e.to_string()))?;
    let first = resp.data.into_iter().next().ok_or(EmbedError::EmptyResponse)?;
    Ok(first.embedding)
}

/// Pick the first Model in the snapshot whose provider is OpenAI.
/// "First" is by `entries()` order (which is by id) — stable across
/// snapshot rebuilds for the same set of models, so the same key gets
/// charged for embeddings consistently.
fn first_openai_model(snapshot: &AisixSnapshot) -> Option<Arc<ResourceEntry<Model>>> {
    snapshot
        .models
        .entries()
        .into_iter()
        .find(|entry| matches!(entry.value.provider(), Some(Provider::Openai)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap_with(model_json: &str) -> AisixSnapshot {
        let s = AisixSnapshot::new();
        let m: Model = serde_json::from_str(model_json).unwrap();
        s.models.insert(ResourceEntry::new("m-1", m, 1));
        s
    }

    #[test]
    fn picks_openai_model_when_present() {
        let s = snap_with(
            r#"{"name":"gpt","model":"openai/gpt-4o","provider_config":{"api_key":"sk-x"}}"#,
        );
        let picked = first_openai_model(&s);
        assert!(picked.is_some());
        assert_eq!(picked.unwrap().value.name, "gpt");
    }

    #[test]
    fn returns_none_when_only_non_openai_models() {
        let s = snap_with(
            r#"{"name":"c","model":"anthropic/claude","provider_config":{"api_key":"k"}}"#,
        );
        assert!(first_openai_model(&s).is_none());
    }

    #[test]
    fn returns_none_on_empty_snapshot() {
        let s = AisixSnapshot::new();
        assert!(first_openai_model(&s).is_none());
    }
}
