//! `Team` entity — the readable name of a team that API keys reference by
//! `team_id`.
//!
//! An API key carries only the team's id. The gateway resolves the name
//! from this document when it labels telemetry, so renaming a team takes
//! effect without rewriting any key.

use serde::{Deserialize, Serialize};

use crate::resource::Resource;

/// The display name of a team that API keys belong to.
///
/// The document's id is the value API keys carry in `team_id`. The gateway
/// reads the name only to label telemetry, as the `team_name` metric label;
/// it never uses it for authentication, routing, or limits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Team {
    /// Display name of the team. An empty name is reported as `unknown`.
    pub name: String,

    /// Set by the loader from the key's id segment. Not part of the wire
    /// shape.
    #[serde(skip)]
    pub(crate) runtime_id: String,
}

impl Resource for Team {
    fn id(&self) -> &str {
        &self.runtime_id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn kind() -> &'static str {
        "teams"
    }
}
