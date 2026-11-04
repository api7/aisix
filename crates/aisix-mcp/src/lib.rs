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
pub mod gateway;
mod oauth;
pub mod openapi;
pub mod shared_store;
mod token_cache;
pub mod token_exchange;
pub mod user_oauth;

pub use bridge::{
    forwarded_client_headers, upstream_from_mcp_server, CallerCredentials, EphemeralBridge,
    McpAuth, McpBridge, McpProtocol, McpTool, McpToolResult, McpUpstream, OAuthClientConfig,
    RmcpBridge, UserOAuthCaller,
};
pub use error::{
    McpError, TokenExchangeError, TokenExchangeFailure, UserOAuthError, UserOAuthFailure,
};
pub use gateway::{
    streamable_http_service, strip_server_prefix, McpGateway, ToolAcl, ToolsListCounts,
    SUPPORTED_PROTOCOL_VERSION_NAMES, TOOL_NAMESPACE_SEPARATOR,
};
pub use openapi::OpenApiBridge;
pub use shared_store::{MemoryStore, PutCondition, SharedStore, StoreError, StoredValue};
pub use user_oauth::{UserOAuthManager, UserOAuthServer, UserSubject};
