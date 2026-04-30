//! Typed entities persisted in etcd and loaded into the gateway snapshot.
//!
//! Each entity is paired with a JSON Schema (spec §3) compiled once at
//! startup and reused on both the admin write path and the watch read path.
//!
//! Entities landing across the live PR series:
//! - [`Model`] — routing target (§3)
//! - [`ApiKey`] — caller credential (§3)
//! - [`RateLimit`] — shared rate-limit config (§3.4 / §8)
//! - [`Routing`] — virtual-router strategy + targets (§3.5, PR #17)
//! - [`Credential`] — managed upstream secret (§3.6, PR #19)
//!
//! Further entities (`Team`, `Guardrail`, `FallbackPolicy`) land
//! alongside the feature PRs that consume them so the schema lives
//! next to its runtime usage.

pub mod apikey;
pub mod credential;
pub mod model;
pub mod rate_limit;
pub mod routing;
pub mod schema;
pub mod snapshot;
pub mod team;

pub use apikey::ApiKey;
pub use credential::Credential;
pub use model::{Model, Provider, ProviderConfig};
pub use rate_limit::RateLimit;
pub use routing::{Routing, RoutingStrategy, RoutingTarget};
pub use schema::{
    validate_apikey, validate_credential, validate_model, validate_team, SchemaError,
};
pub use snapshot::AisixSnapshot;
pub use team::Team;
