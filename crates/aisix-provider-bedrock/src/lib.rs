//! aisix-provider-bedrock — AWS Bedrock runtime provider bridge.
//!
//! Family bridge for [`Adapter::Bedrock`] in the gateway Hub.
//!
//! ## Status (issue #302 Phase G)
//!
//! - [x] D7.1 — AWS SigV4 v4 signature (handled by `aws-sdk-bedrockruntime`)
//! - [x] D7.2.a — Anthropic-on-Bedrock non-streaming dispatch
//!   (`/model/anthropic.claude-*/invoke`, `anthropic_version:
//!   "bedrock-2023-05-31"` in body not header)
//! - [x] D7.6 — Cross-region inference profiles (`us.`/`eu.`/`apac.`/
//!   `global.`/`us-gov.` prefixes stripped by [`bridge::BedrockPublisher::from_model_id`])
//! - [ ] D7.2.b — Anthropic-on-Bedrock streaming via
//!   `invoke_model_with_response_stream` (AWS event-stream framed,
//!   NOT canonical SSE; reuses the Anthropic typed-event stream state
//!   machine from `aisix-provider-anthropic`)
//! - [ ] D7.3 — Meta-on-Bedrock dispatch (Llama 3 / 3.1 / 3.2 / 3.3)
//! - [ ] D7.4 — Mistral / Amazon Titan / Amazon Nova / Cohere / AI21
//!   per-publisher request bodies
//!
//! Until D7.2.b lands, `chat_stream()` returns a clear
//! `BridgeError::Config(...)` referencing the streaming follow-up.
//! Publishers other than Anthropic return a publisher-specific
//! "not yet implemented" error from `chat()` / `chat_stream()`.
//!
//! # Multi-publisher single-entry model
//!
//! AWS Bedrock hosts seven publishers (Anthropic, Meta, Mistral,
//! Amazon Titan, Amazon Nova, Cohere, AI21) under a single Bedrock
//! Runtime API surface. The publisher is encoded in the model id with
//! a `.` separator:
//!
//! - `anthropic.claude-3-5-sonnet-20241022-v2:0`
//! - `meta.llama3-3-70b-instruct-v1:0`
//! - `mistral.mixtral-8x7b-instruct-v0:1`
//! - `amazon.titan-text-premier-v1:0`
//! - `amazon.nova-pro-v1:0`
//! - `cohere.command-r-plus-v1:0`
//! - `ai21.jamba-1-5-large-v1:0`
//!
//! Cross-region inference profiles prefix the publisher with a region
//! code (`us.`, `eu.`, `apac.`):
//!
//! - `us.anthropic.claude-3-5-sonnet-20241022-v2:0`
//!
//! Single-entry routing: every Bedrock-hosted model goes through one
//! provider name (`amazon-bedrock`) in cp-api's catalog, and the
//! publisher + region are resolved inside the bridge from the model
//! id. Diverging from this would force every customer to register a
//! separate provider_key per publisher even though the IAM role + AWS
//! region are the same — exactly the operator pain `amazon-bedrock`
//! solves.
//!
//! # Why a separate bridge (not OpenAiBridge / AnthropicBridge)
//!
//! 1. **Auth is SigV4** — every request needs canonical signing of
//!    method + path + headers + body + region. OpenAiBridge's
//!    `Authorization: Bearer` and AnthropicBridge's `x-api-key` are
//!    both inapplicable.
//! 2. **URL pattern is per-model** — `/model/<model-id>/invoke` for
//!    non-stream, `/invoke-with-response-stream` for streaming.
//!    OpenAiBridge's single `{base}/chat/completions` won't shape
//!    correctly.
//! 3. **Streaming framing differs** — AWS event-stream binary frames,
//!    NOT Server-Sent Events. The DP's `SseDecoder` doesn't apply.
//! 4. **Per-publisher request bodies differ** — Claude on Bedrock
//!    expects an Anthropic Messages-style body with `anthropic_version`
//!    in the body (not header); Llama on Bedrock expects a flat
//!    `prompt + max_gen_len + temperature` shape; Titan expects
//!    `inputText + textGenerationConfig`. The OpenAI-shape body from
//!    the gateway needs per-publisher translation.
//!
//! # References
//!
//! - Bedrock Runtime API — <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModel.html>
//! - Bedrock model IDs — <https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html>
//! - Cross-region inference profiles — <https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html>
//! - Anthropic on Bedrock body shape — <https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html>
//! - AWS Rust SDK `aws-sdk-bedrockruntime` — <https://docs.rs/aws-sdk-bedrockruntime>

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

mod bridge;
mod wire;

pub use bridge::{BedrockBridge, BedrockPublisher};
