//! aisix-provider-azure-openai — Azure OpenAI Service provider bridge.
//!
//! Family bridge for [`Adapter::AzureOpenai`] in the gateway Hub.
//!
//! ## Status (issue #302 Phase F)
//!
//! - [x] D6.1 — `api-key` header auth (resource-key scheme;
//!   `provider_key.secret` is a verbatim string)
//! - [x] D6.2 — Azure URL pattern:
//!   `https://<resource>.openai.azure.com/openai/deployments/<deployment>/chat/completions?api-version=<version>`
//! - [x] D6.3 — Deployment-keyed dispatch from `Model.model_name`
//!   (operator-pinned deployment name, NOT the customer-facing
//!   display name in `req.model`)
//! - [x] D6.5 — Content-filter response tolerance: Azure adds
//!   `prompt_filter_results` / `content_filter_results` to the OpenAI
//!   chat-completions response. The reused `OpenAiResponse` /
//!   `OpenAiStreamChunk` parsers ignore unknown fields by default
//!   (no `deny_unknown_fields`), so the extension passes through
//!   without breaking decoding.
//! - [x] D6.6 — AAD (Entra ID) Bearer auth as a second auth scheme.
//!   `provider_key.secret` autodetects between the resource-key path
//!   (verbatim string) and the AAD client_credentials path (JSON
//!   `{tenant_id, client_id, client_secret}`) by checking the leading
//!   character. The AAD path mints + caches tokens in-process via the
//!   client_credentials grant (no JWT signing) and sends
//!   `Authorization: Bearer <token>` instead of the `api-key:` header.
//!   Backward-compatible — existing api-key deployments keep working
//!   unchanged.
//! - [ ] D6.4 — Per-PK `api_version` override. Today the bridge pins
//!   `AzureUpstreamRef::DEFAULT_API_VERSION` (GA). Follow-up will
//!   accept an explicit version from `provider_key.api_base` query
//!   string or a dedicated PK field.
//!
//! # Why Azure-OpenAI is a separate `Adapter::AzureOpenai` family bridge
//!
//! 1. **Auth header differs** — Azure uses `api-key: <key>`, not
//!    `Authorization: Bearer <key>`. The OpenAiBridge's header-builder
//!    hard-codes Bearer; using it for Azure would either reject or
//!    silently 401.
//! 2. **URL pattern differs** — Azure embeds the deployment name in
//!    the path AND requires `?api-version=YYYY-MM-DD` as a query
//!    parameter. OpenAiBridge's `{base}/chat/completions` won't shape
//!    correctly even with a custom `api_base`.
//! 3. **Model field semantics differ** — the customer's
//!    `upstream_id` is a deployment name, not an OpenAI model id.
//!    Two customers with the same Azure region can have a deployment
//!    "gpt4-prod" pointing at different OpenAI model versions.
//! 4. **Content filter injection** — Azure injects filter-result
//!    objects that downstream OpenAI SDK clients don't know about.
//!    The bridge needs to either pass them through or strip them.
//!
//! These are exactly the cases #302 §3 carves a separate
//! [`Adapter::AzureOpenai`] for.
//!
//! # References
//!
//! - Azure OpenAI Service REST API —
//!   <https://learn.microsoft.com/en-us/azure/ai-services/openai/reference>
//! - api-version compatibility table —
//!   <https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation>
//! - Content filtering response fields —
//!   <https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/content-filter>
//! - Azure OpenAI Python SDK (canonical wire-shape reference for
//!   request building, streaming chunk parsing, and content-filter
//!   field handling) —
//!   <https://github.com/openai/openai-python/blob/main/src/openai/lib/azure.py>

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

mod aad_token_mint;
mod bridge;
mod wire;

pub use bridge::{AzureOpenAiBridge, AzureUpstreamRef};
