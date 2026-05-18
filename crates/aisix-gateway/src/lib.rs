//! aisix-gateway — the Hub-and-Bridge core.
//!
//! This crate holds the provider-agnostic primitives shared by every
//! `aisix-provider-*` crate and by the proxy router:
//!
//! - [`chat`] — normalised `ChatFormat`, `ChatMessage`, `ChatResponse`,
//!   streaming `ChatChunk`, and the usage/finish-reason taxonomy.
//! - [`bridge`] — the `Bridge` trait every provider implements, plus
//!   `BridgeContext` and typed `BridgeError` with stable HTTP status
//!   mapping.
//! - [`hub`] — a small registry keyed on [`aisix_core::models::Provider`]
//!   that dispatches `ChatFormat` to the right `Bridge`.
//! - [`sse`] — a provider-agnostic SSE line decoder. Bridges that stream
//!   over SSE feed it raw bytes and pull typed events back out.
//!
//! The concrete HTTP transport lives in the provider crates — keeping
//! this crate free of `reqwest` at the public-API level makes it testable
//! without wiremock.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

pub mod bridge;
pub mod chat;
pub mod hub;
pub mod sse;

pub use bridge::{
    capture_upstream_error_http, content_type_is_json, parse_retry_after, read_body_capped,
    response_is_json, Bridge, BridgeContext, BridgeError, ChatChunkStream, UpstreamErrorView,
    UpstreamWire, MAX_UPSTREAM_ERROR_BODY_BYTES, MAX_UPSTREAM_ERROR_MESSAGE_BYTES,
};
pub use chat::{
    ChatChunk, ChatDelta, ChatFormat, ChatMessage, ChatResponse, EmbeddingObject, EmbeddingRequest,
    EmbeddingResponse, EmbeddingUsage, FinishReason, Role, UsageStats,
};
pub use hub::Hub;
pub use sse::{SseDecoder, SseEvent};
