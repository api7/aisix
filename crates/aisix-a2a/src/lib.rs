//! aisix-a2a — the A2A (Agent-to-Agent) gateway data-plane crate.
//!
//! First step: a governed client tunnel to a single upstream A2A agent over
//! HTTP + JSON-RPC 2.0, exposed through the [`A2aBridge`] trait. The bridge
//! fetches the agent's card (RFC 8615 well-known URI) and forwards JSON-RPC
//! requests to its service endpoint, holding the upstream credential so the
//! calling client never sees it. Aggregating the downstream-facing
//! `/a2a/<agent>` endpoint, agent-card URL rewriting, and wiring into the
//! shared guardrail/quota pipeline come in later steps — this layer only proves
//! a governed tunnel to one real upstream.
//!
//! Unlike the MCP gateway, there is no official A2A Rust SDK, so the JSON-RPC
//! plumbing is hand-rolled here on the workspace HTTP client. The bridge
//! forwards requests verbatim and does NOT translate between the A2A 0.3 and
//! 1.0 wire formats — version normalization is a later step; a single agent is
//! reached in whichever version it speaks (pinned on the `A2aAgent` resource).

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

pub mod bridge;
pub mod error;

pub use bridge::{
    upstream_from_a2a_agent, A2aAuth, A2aBridge, A2aUpstream, AgentCard, HttpBridge,
    DEFAULT_UPSTREAM_TIMEOUT,
};
pub use error::A2aError;
