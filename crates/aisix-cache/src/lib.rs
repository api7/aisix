//! aisix-cache — exact-match response cache for chat completions.
//!
//! The proxy looks up the cache before dispatching to the upstream
//! Bridge. On hit it returns the cached `ChatResponse` directly with an
//! `x-aisix-cache: hit` header; on miss it falls through to the bridge
//! and stores the response with `x-aisix-cache: miss`.
//!
//! Backends:
//! - [`MemoryCache`] (moka, in-process) — default, configured by
//!   `cfg.cache.backend = "memory"`.
//! - Redis backend lands in a follow-up PR behind the `redis` feature.
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

pub use cache::{Cache, CacheError, CacheOutcome};
pub use key::CacheKey;
pub use memory::{MemoryCache, DEFAULT_CAPACITY, DEFAULT_TTL};
#[cfg(feature = "redis")]
pub use redis::{
    RedisCache, DEFAULT_PREFIX as REDIS_DEFAULT_PREFIX, DEFAULT_TTL as REDIS_DEFAULT_TTL,
};
