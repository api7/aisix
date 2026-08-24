//! The [`GuardrailEmbedder`] the proxy hands to the guardrail chain
//! builders, backing `kind: "semantic"` (AISIX-Cloud#1375).
//!
//! The guardrail crate sits BELOW this one and cannot dispatch an
//! upstream call itself. It could not simply take a `&ProxyState`
//! either: `ProxyState` owns the guardrail index, which owns the chain,
//! which owns the guardrails — a guardrail holding the state back would
//! close an `Arc` cycle and leak the whole graph. So this holds only the
//! two things the dispatch actually needs, the provider hub and a
//! snapshot handle, plus the vector cache.
//!
//! Vector caching is asymmetric on purpose. EXAMPLE prototypes are
//! config: a fixed set per row, worth memoising process-wide so a chain
//! rebuild (any snapshot change, however unrelated) does not re-embed
//! them. SCREENED text is request data: unbounded cardinality, so it is
//! never cached — a cache keyed on user content would grow without
//! limit and would double as a store of exactly the text a content
//! guardrail exists to keep. The prototype half reuses the semantic
//! ROUTER's cache, whose key is already
//! `(embedding_model_id, dimensions, text)`, so the two features share
//! a vector for the same text under the same model instead of each
//! paying for it.

use std::sync::Arc;
use std::time::Duration;

use aisix_core::snapshot::SnapshotHandle;
use aisix_core::{AisixSnapshot, Model};
use aisix_gateway::{BridgeError, Hub};
use aisix_guardrails::{EmbedFailure, GuardrailEmbedder};
use async_trait::async_trait;

use crate::error::ProxyError;
use crate::semantic::SemanticVectorCache;

pub struct ProxyGuardrailEmbedder {
    hub: Arc<Hub>,
    snapshot: SnapshotHandle<AisixSnapshot>,
    cache: Arc<SemanticVectorCache>,
}

impl ProxyGuardrailEmbedder {
    pub fn new(
        hub: Arc<Hub>,
        snapshot: SnapshotHandle<AisixSnapshot>,
        cache: Arc<SemanticVectorCache>,
    ) -> Self {
        Self {
            hub,
            snapshot,
            cache,
        }
    }
}

impl std::fmt::Debug for ProxyGuardrailEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProxyGuardrailEmbedder").finish()
    }
}

#[async_trait]
impl GuardrailEmbedder for ProxyGuardrailEmbedder {
    async fn embed(
        &self,
        model_alias: &str,
        texts: &[String],
        cacheable: bool,
        timeout: Duration,
    ) -> Result<Vec<Vec<f32>>, EmbedFailure> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let snapshot = self.snapshot.load();
        let Some(entry) = snapshot.models.get_by_name(model_alias) else {
            return Err(EmbedFailure::Unresolved);
        };
        // The alias must name an EMBEDDING model. A chat model would
        // answer the dispatch with a completion, not a vector, and the
        // resulting error would read as a provider outage rather than
        // the configuration mistake it is.
        let Some(dimensions) = embedding_dimensions(&entry.value) else {
            return Err(EmbedFailure::Unresolved);
        };

        // Cache lookup first, so a fully warm example set costs no call.
        let mut cached: Vec<Option<Vec<f32>>> = Vec::with_capacity(texts.len());
        let mut misses: Vec<String> = Vec::new();
        for text in texts {
            let hit = if cacheable {
                self.cache
                    .get(&entry.id, dimensions, text)
                    .map(|v| v.as_ref().clone())
            } else {
                None
            };
            if hit.is_none() {
                misses.push(text.clone());
            }
            cached.push(hit);
        }
        if misses.is_empty() {
            return Ok(cached.into_iter().map(Option::unwrap).collect());
        }

        let fetched = crate::semantic::embed_texts(
            &self.hub,
            &snapshot,
            &entry,
            Some(timeout),
            "guardrail-semantic",
            &misses,
        )
        .await
        .map_err(classify)?;
        if fetched.len() != misses.len() {
            return Err(EmbedFailure::Upstream);
        }

        if cacheable {
            for (text, vector) in misses.iter().zip(fetched.iter()) {
                self.cache
                    .insert(&entry.id, dimensions, text, Arc::new(vector.clone()));
            }
        }

        // Re-interleave: `fetched` is in `misses` order, which is the
        // order the `None` holes appear in.
        let mut fetched = fetched.into_iter();
        Ok(cached
            .into_iter()
            .map(|slot| match slot {
                Some(v) => v,
                None => fetched.next().expect("miss count checked above"),
            })
            .collect())
    }
}

/// The declared output dimension of an `embedding`-kind Model, or `None`
/// when the alias is not one.
fn embedding_dimensions(model: &Model) -> Option<u32> {
    model.embedding.as_ref().map(|e| e.dimensions)
}

/// Map a dispatch failure onto the guardrail's bounded failure
/// vocabulary.
///
/// The distinction that matters operationally is deadline-vs-everything
/// else: a timeout says raise `timeout_ms` or move the embedding model
/// closer, while the rest say the model or its credential is wrong.
/// Both land on the same fail-open/fail-closed decision, so mis-binning
/// one can only mislabel a log line, never change a verdict.
fn classify(err: ProxyError) -> EmbedFailure {
    match err {
        ProxyError::Bridge(BridgeError::Timeout { .. }) => EmbedFailure::Timeout,
        // No bridge for the provider key, or the model has no provider /
        // upstream model name — configuration, not an outage.
        ProxyError::ProviderUnavailable | ProxyError::InvalidRequest(_) => EmbedFailure::Unresolved,
        _ => EmbedFailure::Upstream,
    }
}
