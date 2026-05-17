//! aisix-provider-vertex — Google Vertex AI provider bridge.
//!
//! **Skeleton crate** for issue #302 Phase E. Registers as the family
//! bridge for [`Adapter::Vertex`] in the gateway Hub. The actual GCP
//! OAuth integration and per-publisher request building are TODOs
//! filled in by follow-up PRs:
//!
//! - [ ] D5.1 — google-cloud-auth / yup-oauth2 bearer token acquisition
//!   (service account JSON key, ADC, metadata server)
//! - [ ] D5.2 — Gemini publisher dispatch
//!   (`publishers/google/models/<model>:streamGenerateContent`)
//! - [ ] D5.3 — Anthropic-on-Vertex dispatch
//!   (`publishers/anthropic/models/<model>:streamRawPredict` —
//!   different wire shape from canonical Anthropic Messages)
//! - [ ] D5.4 — Llama / Mistral / AI21 publisher dispatch
//! - [ ] D5.5 — `BridgeContext.deadline` + Retry-After plumbing
//!
//! For now the bridge's `chat()` / `chat_stream()` return a clear
//! `BridgeError::Config(...)` so a misconfigured `provider: "google-vertex"`
//! row in the kine catalog surfaces a 501 / 502 with an actionable
//! message rather than silently dropping the dispatch.
//!
//! # Multi-publisher single-entry model
//!
//! Google Vertex AI hosts several **publishers** (Google's own Gemini
//! plus partner offerings from Anthropic, Meta, Mistral, AI21,
//! together's GPT-OSS) under a single API surface. The publisher is
//! encoded in the upstream model id:
//!
//! - `gemini-1.5-pro` → publisher `google`
//! - `claude-3-5-sonnet@20241022` → publisher `anthropic`
//!   (the `@20241022` is the model version tag Anthropic uses)
//! - `llama-3-70b-instruct-maas` → publisher `meta`
//!
//! This mirrors LiteLLM's `vertex_ai/` single-prefix design: every
//! Vertex-hosted model goes through one provider name in cp-api's
//! catalog (`google-vertex`), and the publisher is resolved inside
//! the bridge from the upstream model id. See
//! <https://github.com/BerriAI/litellm/tree/main/litellm/llms/vertex_ai>.
//!
//! Diverging from this would force every customer to register a
//! separate provider_key per publisher even though the GCP credential
//! is the same — exactly the operator pain `google-vertex` solves.
//!
//! # References
//!
//! - LiteLLM `vertex_ai/` design — <https://github.com/BerriAI/litellm/tree/main/litellm/llms/vertex_ai>
//! - Vertex AI REST API — <https://cloud.google.com/vertex-ai/docs/reference/rest>
//! - Vertex publishers index — <https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models>

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

mod bridge;
mod wire;

pub use bridge::{VertexBridge, VertexPublisher};
