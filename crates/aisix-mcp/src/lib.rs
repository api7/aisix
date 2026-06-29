//! aisix-mcp — the MCP gateway data-plane crate.
//!
//! First step (DP-2): a governed client tunnel to a single upstream MCP server
//! over Streamable HTTP, exposed through the [`McpBridge`] trait. Later steps
//! aggregate many upstreams behind the downstream-facing `/mcp` endpoint and
//! route MCP tool traffic through the same auth / ACL / guardrail / quota
//! pipeline as LLM traffic.
//!
//! The official `rmcp` SDK does the JSON-RPC + Streamable HTTP plumbing; this
//! crate keeps every rmcp type behind [`McpBridge`] so the SDK's still-moving
//! API stays contained here.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

pub mod bridge;
pub mod error;

pub use bridge::{McpAuth, McpBridge, McpTool, McpToolResult, McpUpstream, RmcpBridge};
pub use error::McpError;
