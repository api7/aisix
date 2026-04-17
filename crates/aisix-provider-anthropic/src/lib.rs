//! aisix-provider-anthropic — Anthropic Messages API [`AnthropicBridge`].
//!
//! Translates the gateway's OpenAI-shaped [`ChatFormat`] into Claude's
//! `/v1/messages` contract and back. Streaming support maps Anthropic's
//! typed SSE events (`message_start`, `content_block_delta`,
//! `message_delta`, `message_stop`) to the gateway's flat `ChatChunk`
//! stream.
//!
//! See the `Bridge` trait in `aisix-gateway` for the contract this crate
//! implements against.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

mod bridge;
mod wire;

pub use bridge::{AnthropicBridge, ANTHROPIC_DEFAULT_BASE, ANTHROPIC_VERSION};
