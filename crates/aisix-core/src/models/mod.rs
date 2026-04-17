//! Typed entities persisted in etcd and loaded into the gateway snapshot.
//!
//! Each entity is paired with a JSON Schema (spec §3) compiled once at
//! startup and reused on both the admin write path and the watch read path.
//!
//! The entities landing in this PR:
//! - [`Model`] — routing target (§3)
//! - [`ApiKey`] — caller credential (§3)
//! - [`RateLimit`] — shared rate-limit config (§3.4 / §8)
//!
//! Further entities (`Team`, `Budget`, `Credential`, `Guardrail`,
//! `FallbackPolicy`, `RoutingPolicy`) land alongside the feature PRs that
//! consume them so the schema lives next to its runtime usage.

pub mod apikey;
pub mod model;
pub mod rate_limit;
pub mod schema;
pub mod snapshot;

pub use apikey::ApiKey;
pub use model::{Model, Provider, ProviderConfig};
pub use rate_limit::RateLimit;
pub use schema::{validate_apikey, validate_model, SchemaError};
pub use snapshot::AisixSnapshot;
