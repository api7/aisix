//! aisix-provider-openai — OpenAI provider [`OpenAiBridge`] impl.
//!
//! This crate is also the transport used by other OpenAI-compatible
//! upstreams (DeepSeek today, Gemini's OpenAI-compat endpoint later).
//! Those provider crates can wrap [`OpenAiBridge`] with their own
//! `api_base` and metrics label rather than duplicating the transport.
//!
//! See the `Bridge` trait in `aisix-gateway` for the contract this crate
//! implements against.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

mod bridge;
pub mod overrides;
pub mod wire;

pub use bridge::{OpenAiBridge, OPENAI_DEFAULT_BASE};
