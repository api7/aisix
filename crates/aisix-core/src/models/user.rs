//! `User` entity — the readable name of an organization member that API
//! keys reference by `user_id`.
//!
//! An API key carries only the member's id. The gateway resolves the name
//! from this document when it labels telemetry, so renaming a member takes
//! effect without rewriting any key.

use serde::{Deserialize, Serialize};

use crate::resource::Resource;

/// The display name and cost-attribution tags of an organization member
/// who owns API keys.
///
/// The document's id is the value API keys carry in `user_id`. The gateway
/// reads the name only to label telemetry, as the `user_name` metric label,
/// where it takes precedence over the key's own `user_name`, and the tags
/// only to attribute requests and to match `tag:` rate-limit policy
/// dimensions; it never uses either for authentication or routing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct User {
    /// Display name of the member. An empty name is reported as `unknown`.
    pub name: String,

    /// Cost-attribution tags of this member, as tag key to tag value. Each
    /// request made with an API key whose `user_id` names this member is
    /// attributed to them, per tag key, unless the key itself or the key's
    /// team sets the same key. Keys match `^[a-z][a-z0-9_]{0,62}$`, values
    /// are 1 to 128 characters, and at most 32 entries are allowed. Omitted
    /// or empty means the member adds no tags.
    #[serde(default, skip_serializing_if = "crate::models::Tags::is_empty")]
    #[schemars(schema_with = "crate::models::tags::tags_schema")]
    pub tags: crate::models::Tags,

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
