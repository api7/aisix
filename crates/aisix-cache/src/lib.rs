//! aisix-cache — response cache for chat completions: an exact-match
//! layer, plus an embedding-similarity (semantic) layer for policies
//! that configure one.
//!
//! The proxy looks up the cache before dispatching to the upstream
//! Bridge. On hit it returns the cached `ChatResponse` directly with an
//! `x-aisix-cache: hit` header; on miss it falls through to the bridge
//! and stores the response with `x-aisix-cache: miss`. When the matched
//! policy carries a `semantic` block, an exact miss additionally probes
//! [`SemanticCacheStore`] with the request's embedding.
//!
//! Backends:
//! - [`MemoryCache`] / [`MemorySemanticCache`] (in-process) — always
//!   available.
//! - `RedisCache` (behind the `redis` feature) — built when the boot
//!   config carries `cache.redis`.
//!
//! The proxy picks the backend per request from the matched
//! `CachePolicy.backend` (see `aisix-proxy::state::CacheBackends`);
//! the boot config only determines which instances exist.
//!
//! Streaming responses aren't cached at this layer — the upstream stream
//! has no terminal value to store.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

mod cache;
mod key;
mod memory;
#[cfg(feature = "redis")]
mod redis;
mod semantic;

pub use cache::{Cache, CacheError, CacheOutcome};
pub use key::{semantic_prompt_text, CacheKey};
pub use memory::{MemoryCache, DEFAULT_CAPACITY, DEFAULT_TTL};
#[cfg(feature = "redis")]
pub use redis::{
    RedisCache, DEFAULT_PREFIX as REDIS_DEFAULT_PREFIX, DEFAULT_TTL as REDIS_DEFAULT_TTL,
};
pub use semantic::{MemorySemanticCache, SemanticCacheStore, SemanticHit};
