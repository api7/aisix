//! aisix-core — shared primitives used across the gateway.
//!
//! This crate only contains types and errors. It must not depend on any
//! I/O framework (no tokio, no axum, no reqwest) so that it can be reused
//! by every other crate in the workspace.

#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]
