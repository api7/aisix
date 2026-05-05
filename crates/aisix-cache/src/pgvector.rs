//! pgvector-backed semantic cache (Stage 4b of cache-policies).
//!
//! Wire path: the DP sends precomputed embeddings to dp-manager's
//! `/dp/cache/{lookup,put}` mTLS endpoints; dp-manager owns the PG
//! connection and runs the cosine-ANN search over
//! `cache_entries_semantic`. Multi-tenant isolation lives at the
//! env_id check inside the cp-api handler — the DP never gets a PG
//! client, which keeps the trust boundary thin.
//!
//! The DP is responsible for:
//!   1. computing the embedding (so embedding-provider credentials
//!      stay on the data plane — see `crate::embed`)
//!   2. POSTing it to `/dp/cache/lookup`
//!   3. on miss, calling the upstream model and POSTing the result
//!      to `/dp/cache/put`
//!
//! This module is just the HTTP client layer. The chat handler in
//! `aisix-proxy::chat` orchestrates 1 + 2 + 3.

use aisix_gateway::ChatResponse;
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;

/// Errors the pgvector cache can surface to the proxy. The proxy's
/// chat handler maps `LookupFailed` / `PutFailed` to a tracing::warn!
/// and cache-miss fallthrough (fail-open per the Stage 4b design
/// note); upstream callers never see these.
#[derive(Debug, thiserror::Error)]
pub enum SemanticCacheError {
    #[error("dp-manager /dp/cache/lookup failed: {0}")]
    LookupFailed(String),
    #[error("dp-manager /dp/cache/put failed: {0}")]
    PutFailed(String),
    /// The handler returned a non-2xx with the canonical
    /// `{error: {code, message}}` envelope. Surfaced as the message
    /// for log readability; the proxy still falls open.
    #[error("dp-manager error {status}: {code} — {message}")]
    HandlerError {
        status: u16,
        code: String,
        message: String,
    },
}

/// One semantic cache hit, materialised from the lookup envelope.
/// Mirrors the fields the chat handler needs at the cache-hit return
/// site (response body, the original usage so the dashboard's
/// "tokens saved" stats reflect the original event).
#[derive(Debug, Clone)]
pub struct SemanticHit {
    pub response: ChatResponse,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    /// Cosine similarity of this match (1 - cosine_distance) against
    /// the lookup query. Logged on hit but not surfaced to clients.
    pub similarity: f32,
}

/// HTTP client wrapper for the dp-manager-side semantic cache. Cheap
/// to clone (`reqwest::Client` is internally `Arc<…>`, the wrapper
/// just adds a base URL).
#[derive(Debug, Clone)]
pub struct PgvectorCache {
    client: Client,
    /// Absolute base URL of the dp-manager listener, e.g.
    /// `https://dp-manager.aisix.svc:7944`. Trailing slash NOT
    /// included — `format_url` joins with `/dp/cache/...` directly.
    base_url: Arc<str>,
}

impl PgvectorCache {
    /// Build with an externally-configured mTLS-presenting client.
    /// The client must already have the DP's client cert + the
    /// dp-manager's CA loaded — same bundle the telemetry sender
    /// uses (see `aisix-server::heartbeat::build_mtls_client`).
    pub fn new(client: Client, base_url: impl Into<String>) -> Self {
        let mut url = base_url.into();
        while url.ends_with('/') {
            url.pop();
        }
        Self {
            client,
            base_url: Arc::from(url.as_str()),
        }
    }

    /// Look up a semantically-similar entry. `threshold` is the
    /// cosine-similarity floor (0..=1). Returns:
    ///   - `Ok(Some(hit))` when the best entry matches above threshold
    ///   - `Ok(None)` when no entry matches (or all matches are
    ///     below the threshold)
    ///   - `Err(...)` on transport / handler failure (proxy maps
    ///     these to a warn + cache miss fallthrough)
    pub async fn lookup(
        &self,
        policy_id: &str,
        embedding: &[f32],
        threshold: Option<f32>,
    ) -> Result<Option<SemanticHit>, SemanticCacheError> {
        let body = LookupRequest {
            policy_id,
            embedding,
            threshold,
        };
        let url = self.format_url("/dp/cache/lookup");
        let resp = self
            .client
            .post(&url)
            .json(&body)
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .map_err(|e| SemanticCacheError::LookupFailed(e.to_string()))?;

        let status = resp.status();
        if !status.is_success() {
            return Err(handler_error(status, resp).await);
        }
        let parsed: LookupResponse = resp
            .json()
            .await
            .map_err(|e| SemanticCacheError::LookupFailed(format!("decode: {e}")))?;
        if !parsed.hit {
            return Ok(None);
        }
        Ok(Some(SemanticHit {
            response: parsed.response.ok_or_else(|| {
                SemanticCacheError::LookupFailed("hit=true but response missing".into())
            })?,
            prompt_tokens: parsed.prompt_tokens,
            completion_tokens: parsed.completion_tokens,
            similarity: parsed.similarity,
        }))
    }

    /// Persist a fresh entry. Called from the proxy's post-success
    /// path. `prompt_text` is the canonical text we embedded
    /// (typically the last user message); cp-api stores it verbatim
    /// for "show me what this entry caches" debugging.
    ///
    /// `ttl_seconds = None` makes the cp-api handler use the
    /// policy's stored TTL — that's the normal case. The override is
    /// kept on the wire for future per-request TTL hints.
    #[allow(clippy::too_many_arguments)]
    pub async fn put(
        &self,
        policy_id: &str,
        prompt_text: &str,
        embedding: &[f32],
        response: &ChatResponse,
        prompt_tokens: u32,
        completion_tokens: u32,
        ttl_seconds: Option<u32>,
    ) -> Result<(), SemanticCacheError> {
        let body = PutRequest {
            policy_id,
            prompt_text,
            embedding,
            response,
            prompt_tokens,
            completion_tokens,
            ttl_seconds,
        };
        let url = self.format_url("/dp/cache/put");
        let resp = self
            .client
            .post(&url)
            .json(&body)
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .map_err(|e| SemanticCacheError::PutFailed(e.to_string()))?;
        let status = resp.status();
        if !status.is_success() {
            return Err(handler_error(status, resp).await);
        }
        // Body is `{entry_id, expires_at}` — we don't need either on
        // the proxy side, so don't decode.
        Ok(())
    }

    fn format_url(&self, path: &str) -> String {
        format!("{}{path}", self.base_url)
    }
}

// --- Wire types -----------------------------------------------------

#[derive(Serialize)]
struct LookupRequest<'a> {
    policy_id: &'a str,
    embedding: &'a [f32],
    #[serde(skip_serializing_if = "Option::is_none")]
    threshold: Option<f32>,
}

#[derive(Deserialize)]
struct LookupResponse {
    hit: bool,
    #[serde(default)]
    response: Option<ChatResponse>,
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
    #[serde(default)]
    similarity: f32,
}

#[derive(Serialize)]
struct PutRequest<'a> {
    policy_id: &'a str,
    prompt_text: &'a str,
    embedding: &'a [f32],
    response: &'a ChatResponse,
    #[serde(skip_serializing_if = "is_zero_u32")]
    prompt_tokens: u32,
    #[serde(skip_serializing_if = "is_zero_u32")]
    completion_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    ttl_seconds: Option<u32>,
}

#[inline]
fn is_zero_u32(n: &u32) -> bool {
    *n == 0
}

/// Decode a non-2xx response into the canonical
/// `{error: {code, message}}` envelope. Falls back to a generic
/// HandlerError if the body isn't shaped like that — never panics.
async fn handler_error(status: StatusCode, resp: reqwest::Response) -> SemanticCacheError {
    #[derive(Deserialize)]
    struct Envelope {
        error: Option<EnvelopeBody>,
    }
    #[derive(Deserialize)]
    struct EnvelopeBody {
        code: String,
        message: String,
    }
    match resp.json::<Envelope>().await {
        Ok(env) => match env.error {
            Some(b) => SemanticCacheError::HandlerError {
                status: status.as_u16(),
                code: b.code,
                message: b.message,
            },
            None => SemanticCacheError::HandlerError {
                status: status.as_u16(),
                code: "UNKNOWN".into(),
                message: "no error envelope".into(),
            },
        },
        Err(_) => SemanticCacheError::HandlerError {
            status: status.as_u16(),
            code: "UNKNOWN".into(),
            message: "non-JSON response".into(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aisix_gateway::{ChatMessage, ChatResponse, FinishReason, UsageStats};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn sample_response() -> ChatResponse {
        ChatResponse {
            id: "cmpl-test".into(),
            model: "gpt-4o".into(),
            message: ChatMessage::assistant("cached"),
            finish_reason: FinishReason::Stop,
            usage: UsageStats {
                prompt_tokens: 1,
                completion_tokens: 2,
                ..Default::default()
            },
        }
    }

    #[tokio::test]
    async fn lookup_hit_returns_response_and_similarity() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/dp/cache/lookup"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "hit": true,
                "response": {
                    "id": "cmpl-cached",
                    "model": "gpt-4o",
                    "message": {"role": "assistant", "content": "from cache"},
                    "finish_reason": "stop",
                    "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11}
                },
                "prompt_tokens": 5,
                "completion_tokens": 6,
                "similarity": 0.97
            })))
            .mount(&server)
            .await;

        let cache = PgvectorCache::new(reqwest::Client::new(), server.uri());
        let hit = cache
            .lookup("policy-1", &[0.0_f32; 3], Some(0.92))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(hit.response.message.content, "from cache");
        assert_eq!(hit.prompt_tokens, 5);
        assert_eq!(hit.completion_tokens, 6);
        assert!((hit.similarity - 0.97).abs() < 1e-6);
    }

    #[tokio::test]
    async fn lookup_miss_returns_none() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/dp/cache/lookup"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"hit": false, "similarity": 0.4})),
            )
            .mount(&server)
            .await;

        let cache = PgvectorCache::new(reqwest::Client::new(), server.uri());
        let res = cache
            .lookup("policy-1", &[0.0_f32; 3], None)
            .await
            .unwrap();
        assert!(res.is_none());
    }

    #[tokio::test]
    async fn lookup_handler_error_surfaces_envelope() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/dp/cache/lookup"))
            .respond_with(
                ResponseTemplate::new(404).set_body_json(serde_json::json!({
                    "error": {"code": "NOT_FOUND", "message": "cache_policy not found"}
                })),
            )
            .mount(&server)
            .await;

        let cache = PgvectorCache::new(reqwest::Client::new(), server.uri());
        let err = cache
            .lookup("policy-1", &[0.0_f32; 3], None)
            .await
            .unwrap_err();
        match err {
            SemanticCacheError::HandlerError { status, code, .. } => {
                assert_eq!(status, 404);
                assert_eq!(code, "NOT_FOUND");
            }
            other => panic!("expected HandlerError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn put_returns_ok_on_success() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/dp/cache/put"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "entry_id": "11111111-1111-1111-1111-111111111111",
                "expires_at": "2026-01-01T00:00:00Z"
            })))
            .expect(1)
            .mount(&server)
            .await;

        let cache = PgvectorCache::new(reqwest::Client::new(), server.uri());
        cache
            .put(
                "policy-1",
                "the prompt",
                &[0.0_f32; 3],
                &sample_response(),
                1,
                2,
                None,
            )
            .await
            .unwrap();
    }

    #[test]
    fn base_url_strips_trailing_slashes() {
        let cache = PgvectorCache::new(reqwest::Client::new(), "https://dpmgr///");
        assert_eq!(cache.format_url("/dp/cache/lookup"), "https://dpmgr/dp/cache/lookup");
    }
}
