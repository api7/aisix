//! aisix-cache — exact-match response cache for chat completions.
//!
//! The proxy looks up the cache before dispatching to the upstream
//! Bridge. On hit it returns the cached `ChatResponse` directly with an
//! `x-aisix-cache: hit` header; on miss it falls through to the bridge
//! and stores the response with `x-aisix-cache: miss`.
//!
//! Backends:
//! - [`MemoryCache`] (moka, in-process) — always available.
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

pub use cache::{Cache, CacheError, CacheOutcome};
pub use key::CacheKey;
pub use memory::{MemoryCache, DEFAULT_CAPACITY, DEFAULT_TTL};
#[cfg(feature = "redis")]
pub use redis::{
    RedisCache, DEFAULT_PREFIX as REDIS_DEFAULT_PREFIX, DEFAULT_TTL as REDIS_DEFAULT_TTL,
};
