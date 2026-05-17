//! aisix-provider-vertex — Google Vertex AI provider bridge.
//!
//! Family bridge for [`Adapter::Vertex`] in the gateway Hub.
//!
//! ## Status (issue #302 Phase E)
//!
//! - [x] D5.2.a — Gemini publisher chat dispatch
//!   (`publishers/google/models/<model>:generateContent`)
//! - [x] D5.5 — `BridgeContext.deadline` plumbing on `chat()`
//! - [ ] D5.1 — In-process GCP OAuth2 token mint (`yup-oauth2` /
//!   `gcp_auth` service-account JSON → access token with
//!   auto-refresh). **Today the bridge expects a pre-minted
//!   access token** in `ProviderKey.secret.access_token`;
//!   operators are responsible for refresh (GCP tokens TTL
//!   ~1 hour). Follow-up will lift that burden into the bridge.
//! - [ ] D5.2.b — Gemini streaming via
//!   `:streamGenerateContent?alt=sse`
//! - [ ] D5.3 — Anthropic-on-Vertex dispatch
//!   (`publishers/anthropic/models/<model>:rawPredict`,
//!   `anthropic_version: "vertex-2023-10-16"`)
//! - [ ] D5.4 — Llama / Mistral / AI21 publisher dispatch
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
//! Single-prefix routing: every Vertex-hosted model goes through one
//! provider name in cp-api's catalog (`google-vertex`), and the
//! publisher is resolved inside the bridge from the upstream model
//! id. Diverging from this would force every customer to register a
//! separate provider_key per publisher even though the GCP credential
//! is the same — exactly the operator pain `google-vertex` solves.
//!
//! # References
//!
//! - Vertex AI REST API — <https://cloud.google.com/vertex-ai/docs/reference/rest>
//! - Gemini generateContent body shape — <https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini>
//! - Vertex publishers index — <https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models>
//! - Google Gemini Python SDK — <https://github.com/google-gemini/generative-ai-python>

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

mod bridge;
mod wire;

pub use bridge::{VertexBridge, VertexPublisher};
