//! `Team` entity — the readable name of a team that API keys reference by
//! `team_id`.
//!
//! An API key carries only the team's id. The gateway resolves the name
//! from this document when it labels telemetry, so renaming a team takes
//! effect without rewriting any key.

use serde::{Deserialize, Serialize};

use crate::resource::Resource;

/// The display name and cost-attribution tags of a team that API keys
/// belong to.
///
/// The document's id is the value API keys carry in `team_id`. The gateway
/// reads the name only to label telemetry, as the `team_name` metric label,
/// and the tags only to attribute requests and to match `tag:` rate-limit
/// policy dimensions; it never uses either for authentication or routing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Team {
    /// Display name of the team. An empty name is reported as `unknown`.
    pub name: String,

    /// Cost-attribution tags of this team, as tag key to tag value. Each
    /// request made with an API key whose `team_id` names this team is
    /// attributed to them, per tag key, unless the key itself sets the same
    /// key; they take precedence over the owning member's tags. Keys match
    /// `^[a-z][a-z0-9_]{0,62}$`, values are 1 to 128 characters, and at most
    /// 32 entries are allowed. Omitted or empty means the team adds no tags.
    #[serde(
        default,
        deserialize_with = "crate::models::tags::null_as_empty",
        skip_serializing_if = "crate::models::Tags::is_empty"
    )]
    #[schemars(schema_with = "crate::models::tags::tags_schema")]
    pub tags: crate::models::Tags,

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
