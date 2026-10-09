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
use aisix_guardrails::{EmbedError, EmbedFailure, Embedded, GuardrailEmbedder};
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
        model_id: Option<&str>,
        texts: &[String],
        cacheable: bool,
        timeout: Duration,
    ) -> Result<Embedded, EmbedError> {
        if texts.is_empty() {
            // No call, so nothing resolved and nothing to name. Safe
            // because a caller with no texts also produces no score: the
            // semantic guardrail's report is driven by the candidates it
            // judged, and an empty batch judges none. A score carrying an
            // empty `embedding_model` would be dropped whole downstream
            // rather than shown without one.
            return Ok(Embedded {
                model: String::new(),
                vectors: Vec::new(),
            });
        }
        let snapshot = self.snapshot.load();
        // Resolved per call against the live table, so a rename of the
        // embedding model takes effect on the next screened request
        // without the guardrail row being rewritten or its chain rebuilt.
        let alias = aisix_core::models::resolve_model_ref(&snapshot, model_alias, model_id);
        let Some(entry) = snapshot.models.get_by_name(&alias) else {
            return Err(EmbedFailure::Unresolved.into());
        };
        // The alias must name an EMBEDDING model. A chat model would
        // answer the dispatch with a completion, not a vector, and the
        // resulting error would read as a provider outage rather than
        // the configuration mistake it is.
        let Some(dimensions) = embedding_dimensions(&entry.value) else {
            return Err(EmbedFailure::Unresolved.into());
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
            return Ok(Embedded {
                model: entry.value.display_name.clone(),
                vectors: cached.into_iter().map(Option::unwrap).collect(),
            });
        }

        let fetched = crate::semantic::embed_texts(
            &self.hub,
            &snapshot,
            &entry,
            Some(timeout),
            "guardrail-semantic",
            aisix_obs::GatewayEmbeddingPurpose::Guardrail,
            &misses,
        )
        .await
        .map_err(classify)?;
        if fetched.len() != misses.len() {
            return Err(EmbedFailure::Upstream.into());
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
        Ok(Embedded {
            model: entry.value.display_name.clone(),
            vectors: cached
                .into_iter()
                .map(|slot| match slot {
                    Some(v) => v,
                    None => fetched.next().expect("miss count checked above"),
                })
                .collect(),
        })
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
///
/// The error itself rides along for the guardrail's failure log, which is
/// the only place it is reported: nothing on the dispatch path logs it.
/// Any text derived from an upstream response is left out, because a
/// guardrail log must never carry the screened input (#153) and a
/// response can quote it back: an error envelope keeps only its status, a
/// decode failure only its position (serde's message quotes the offending
/// value), and a misconfiguration only its variant (it can embed a
/// token endpoint's response body). Transport errors keep their full chain.
fn classify(err: ProxyError) -> EmbedError {
    let failure = match &err {
        ProxyError::Bridge(BridgeError::Timeout { .. }) => EmbedFailure::Timeout,
        // No bridge for the provider key, or the model has no provider /
        // upstream model name — configuration, not an outage.
        ProxyError::ProviderUnavailable | ProxyError::InvalidRequest(_) => EmbedFailure::Unresolved,
        _ => EmbedFailure::Upstream,
    };
    let error = match &err {
        ProxyError::Bridge(BridgeError::UpstreamStatus { status, .. }) => {
            format!("upstream returned HTTP {status}")
        }
        ProxyError::Bridge(BridgeError::UpstreamInBand { status, .. }) => match status {
            Some(status) => format!("upstream reported an in-band error (status {status})"),
            None => "upstream reported an in-band error".to_owned(),
        },
        ProxyError::Bridge(BridgeError::UpstreamDecode(message)) => {
            match decode_position(message) {
                Some((line, column)) => {
                    format!("upstream response could not be decoded (line {line} column {column})")
                }
                None => "upstream response could not be decoded".to_owned(),
            }
        }
        ProxyError::Bridge(BridgeError::Config(_)) => "bridge is misconfigured".to_owned(),
        _ => aisix_guardrails::error_chain(&err),
    };
    EmbedError {
        failure,
        error: Some(error),
    }
}

/// The `line N column M` suffix serde_json appends to its messages. Only
/// the two numbers are taken, so nothing of the message itself survives.
fn decode_position(message: &str) -> Option<(u64, u64)> {
    let (_, position) = message.rsplit_once(" at line ")?;
    let (line, column) = position.split_once(" column ")?;
    Some((line.parse().ok()?, column.parse().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logged(err: BridgeError) -> String {
        classify(ProxyError::Bridge(err)).error.expect("error text")
    }

    #[test]
    fn a_decode_failure_keeps_only_its_position() {
        let quoted =
            "invalid type: string \"screened text\", expected a sequence at line 1 column 42";
        assert_eq!(
            logged(BridgeError::UpstreamDecode(format!(
                "upstream body: {quoted}"
            ))),
            "upstream response could not be decoded (line 1 column 42)"
        );
        assert_eq!(
            logged(BridgeError::UpstreamDecode(
                "error decoding response body".into()
            )),
            "upstream response could not be decoded"
        );
    }

    #[test]
    fn a_misconfiguration_drops_its_text() {
        let err =
            BridgeError::Config("token mint upstream returned HTTP 400: screened text".into());
        assert_eq!(logged(err), "bridge is misconfigured");
    }

    #[test]
    fn a_transport_error_keeps_its_cause() {
        let err = BridgeError::Transport("connection refused".into());
        assert!(logged(err).contains("connection refused"));
    }
}
