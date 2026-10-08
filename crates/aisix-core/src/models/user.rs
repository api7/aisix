//! `User` entity — the readable name of an organization member that API
//! keys reference by `user_id`.
//!
//! An API key carries only the member's id. The gateway resolves the name
//! from this document when it labels telemetry, so renaming a member takes
//! effect without rewriting any key.

use serde::{Deserialize, Serialize};

use crate::resource::Resource;

/// The display name of an organization member who owns API keys.
///
/// The document's id is the value API keys carry in `user_id`. The gateway
/// reads the name only to label telemetry, as the `user_name` metric label,
/// where it takes precedence over the key's own `user_name`; it never uses
/// it for authentication, routing, or limits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct User {
    /// Display name of the member. An empty name is reported as `unknown`.
    pub name: String,

    /// Set by the loader from the key's id segment. Not part of the wire
    /// shape.
    #[serde(skip)]
    pub(crate) runtime_id: String,
}

impl Resource for User {
    fn id(&self) -> &str {
        &self.runtime_id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn kind() -> &'static str {
        "users"
    }
}
