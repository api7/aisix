#[path = "gateway/error.rs"]
pub mod error;
#[path = "gateway/formats/mod.rs"]
pub mod formats;
#[path = "gateway/provider_instance.rs"]
pub mod provider_instance;
#[path = "gateway/providers/mod.rs"]
pub mod providers;
#[path = "gateway/session.rs"]
pub mod session;
#[path = "gateway/streams/mod.rs"]
pub mod streams;
#[path = "gateway/traits/mod.rs"]
pub mod traits;
#[path = "gateway/types/mod.rs"]
pub mod types;

#[path = "gateway/gateway.rs"]
mod gateway_impl;

pub use gateway_impl::Gateway;
