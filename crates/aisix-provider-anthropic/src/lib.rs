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
pub mod wire;

pub use bridge::{AnthropicBridge, ANTHROPIC_DEFAULT_BASE, ANTHROPIC_VERSION};

/// Inbound Anthropic-protocol translation surface — used by the
/// proxy's `/v1/messages` handler when the targeted Model points at
/// a non-Anthropic upstream. The flow is symmetric to the existing
/// outbound path:
///
/// - [`parse_inbound_request`] turns the request body into
///   `ChatFormat` so any Bridge can dispatch it.
/// - [`chat_response_into_anthropic_json`] renders the bridge's
///   `ChatResponse` back as Anthropic JSON.
/// - [`AnthropicSseEncoder`] re-encodes the bridge's `ChatChunk`
///   stream as Anthropic typed SSE events.
pub use wire::{
    chat_response_into_anthropic_json, parse_inbound_request,
    translate_anthropic_tool_choice_to_openai, translate_anthropic_tools_to_openai,
    translate_extras_to_openai_shape, AnthropicInboundError, AnthropicSseEncoder,
    AnthropicSseEvent,
};
