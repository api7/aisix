//! aisix-ratelimit — two-phase RPM/TPM/concurrency limiter.
//!
//! The proxy middleware calls [`Limiter::pre_commit`] before dispatching
//! a chat request; the returned [`Reservation`] is finalised with
//! [`Reservation::commit_tokens`] after the upstream response completes.
//!
//! Limits come from [`aisix_core::RateLimit`] — currently attached to
//! `ApiKey` entries. RPM/RPD are checked-and-incremented up front so
//! burst traffic fails fast; TPM/TPD are checked-only up front and
//! incremented on commit because the token cost is only known after
//! the upstream response lands.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

pub mod clock;
mod error;
mod limiter;
pub mod store;
mod window;

pub use clock::{Clock, SystemClock, TestClock};
pub use error::RateLimitError;
pub use limiter::{
    Limiter, MultiReservation, RateLimitStatus, Reservation, StreamConcurrencyGuard,
};
pub use store::local::LocalStore;
pub use store::redis::RedisStore;
pub use store::RateStore;
pub use window::{FixedWindowCounter, WindowCheck};
