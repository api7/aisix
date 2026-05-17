//! aisix-provider-azure-openai — Azure OpenAI Service provider bridge.
//!
//! **Skeleton crate** for issue #302 Phase F. Registers as the family
//! bridge for [`Adapter::AzureOpenai`] in the gateway Hub. The actual
//! deployment-keyed dispatch is TODO and filled by follow-up PRs:
//!
//! - [ ] D6.1 — `api-key` header auth (NOT `Authorization: Bearer`)
//! - [ ] D6.2 — Azure URL pattern:
//!   `https://<resource>.openai.azure.com/openai/deployments/<deployment>/chat/completions?api-version=<version>`
//! - [ ] D6.3 — `upstream_id` parsing as `<deployment-name>` rather
//!   than OpenAI model id (e.g. customer's deployment "prod-gpt4o" maps
//!   to whichever underlying OpenAI model their Azure tenancy
//!   provisioned)
//! - [ ] D6.4 — `api_version` parameter handling (Azure pins it via
//!   query string; the cp-api side ships it in `provider_key.api_base`
//!   or a dedicated field)
//! - [ ] D6.5 — Content filter response: Azure injects
//!   `prompt_filter_results` / `content_filter_results` into responses;
//!   the bridge must surface these without confusing the OpenAI-shape
//!   translation
//!
//! For now the bridge's `chat()` / `chat_stream()` return a clear
//! `BridgeError::Config(...)` so a misconfigured `provider: "azure"`
//! row in the kine catalog surfaces a 501 / 502 with an actionable
//! message rather than silently dropping the dispatch.
//!
//! # Why Azure-OpenAI is a separate bridge (not OpenAiBridge::with_name)
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
//! [`Adapter::AzureOpenai`] for. See LiteLLM `azure/`:
//! <https://github.com/BerriAI/litellm/tree/main/litellm/llms/azure>.
//!
//! # References
//!
//! - Azure OpenAI Service REST API —
//!   <https://learn.microsoft.com/en-us/azure/ai-services/openai/reference>
//! - api-version compatibility table —
//!   <https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation>
//! - Content filtering response fields —
//!   <https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/content-filter>
//! - LiteLLM `azure/` reference impl —
//!   <https://github.com/BerriAI/litellm/tree/main/litellm/llms/azure>

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

mod bridge;
mod wire;

pub use bridge::{AzureOpenAiBridge, AzureUpstreamRef};
