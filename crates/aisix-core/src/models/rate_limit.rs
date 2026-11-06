//! Rate-limit configuration attached to Models and ApiKeys.
//!
//! All fields are optional; absence means "no limit on that dimension".
//! Windows per spec §3:
//! - `rps` — 1s fixed window (request count only)
//! - `tpm`/`rpm` — 60s fixed window
//! - `rph` — 3600s fixed window (request count only)
//! - `tpd`/`rpd` — 86400s fixed window
//! - `concurrency` — semaphore capacity (not windowed)
//!
//! Token-rate counters are minute/day only; there is no `tps` or `tph`
//! field.
//!
//! `window_type: sliding` swaps the minute/hour/day windows for a rolling
//! sum of epoch-aligned sub-buckets (see [`RateLimitWindowType`]); `rps`
//! and `concurrency` are unaffected.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RateLimit {
    /// Tokens per 60-second window.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tpm: Option<u64>,

    /// Tokens per 86,400-second window.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tpd: Option<u64>,

    /// Requests per 1-second window. There is no per-second token limit field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rps: Option<u64>,

    /// Requests per 60-second window.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rpm: Option<u64>,

    /// Requests per 3,600-second window. There is no per-hour token limit field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rph: Option<u64>,

    /// Requests per 86,400-second window.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rpd: Option<u64>,

    /// Max concurrent in-flight requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub concurrency: Option<u32>,

    /// How the minute, hour, and day limits count. `rps` and `concurrency`
    /// are unaffected.
    #[serde(default, skip_serializing_if = "RateLimitWindowType::is_fixed")]
    #[schemars(schema_with = "window_type_schema")]
    pub window_type: RateLimitWindowType,
}

/// Counting algorithm for the minute, hour, and day limits of a rate limit.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum RateLimitWindowType {
    /// Each window starts at a fixed boundary (the top of the minute, hour,
    /// or day, UTC) and its count resets to zero when the next one starts.
    /// A burst at the end of one window and another at the start of the next
    /// can therefore admit up to twice the limit within one window's length.
    #[default]
    Fixed,
    /// A rolling window. Usage is recorded in sub-buckets aligned to a fixed
    /// step: 10 seconds for `rpm` and `tpm`, 5 minutes for `rph`, and 1 hour
    /// for `rpd` and `tpd`. A request is checked against the sum of the
    /// current sub-bucket and the preceding sub-buckets that cover one full
    /// window, so no window-length interval ever admits more than the limit.
    /// Usage leaves the count between one window and one window plus one step
    /// after it was recorded.
    Sliding,
}

/// The derived schema carries no `default` for a field skipped when it holds
/// its default, so state it explicitly.
fn window_type_schema(gen: &mut schemars::gen::SchemaGenerator) -> schemars::schema::Schema {
    use schemars::schema::{Schema, SchemaObject, SubschemaValidation};
    let mut schema = SchemaObject {
        subschemas: Some(Box::new(SubschemaValidation {
            all_of: Some(vec![gen.subschema_for::<RateLimitWindowType>()]),
            ..Default::default()
        })),
        ..Default::default()
    };
    schema.metadata().default = Some(serde_json::json!("fixed"));
    Schema::Object(schema)
}

impl RateLimitWindowType {
    pub const fn is_fixed(&self) -> bool {
        matches!(self, Self::Fixed)
    }

    pub const fn is_sliding(&self) -> bool {
        matches!(self, Self::Sliding)
    }
}

impl RateLimit {
    pub const fn is_unrestricted(&self) -> bool {
        self.tpm.is_none()
            && self.tpd.is_none()
            && self.rps.is_none()
            && self.rpm.is_none()
            && self.rph.is_none()
            && self.rpd.is_none()
            && self.concurrency.is_none()
    }
}

/// Limits for one API key's calls to one MCP server
/// (`ApiKey::mcp_rate_limits`).
///
/// Carries only the request-count dimensions of [`RateLimit`]: an MCP
/// `tools/call` commits no tokens, so a token-rate cap here would never be
/// consumed — the shape leaves `tpm`/`tpd` out rather than accepting a knob
/// that is silently inert.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct McpRateLimit {
    /// Tool calls per 1-second window.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rps: Option<u64>,

    /// Tool calls per 60-second window.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rpm: Option<u64>,

    /// Tool calls per 3,600-second window.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rph: Option<u64>,

    /// Tool calls per 86,400-second window.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rpd: Option<u64>,

    /// Max concurrent in-flight tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub concurrency: Option<u32>,

    /// How the minute, hour, and day limits count. `rps` and `concurrency`
    /// are unaffected.
    #[serde(default, skip_serializing_if = "RateLimitWindowType::is_fixed")]
    #[schemars(schema_with = "window_type_schema")]
    pub window_type: RateLimitWindowType,
}

impl McpRateLimit {
    pub const fn is_unrestricted(&self) -> bool {
        self.rps.is_none()
            && self.rpm.is_none()
            && self.rph.is_none()
            && self.rpd.is_none()
            && self.concurrency.is_none()
    }
}

impl From<&McpRateLimit> for RateLimit {
    fn from(mcp: &McpRateLimit) -> Self {
        Self {
            tpm: None,
            tpd: None,
            rps: mcp.rps,
            rpm: mcp.rpm,
            rph: mcp.rph,
            rpd: mcp.rpd,
            concurrency: mcp.concurrency,
            window_type: mcp.window_type,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_unrestricted() {
        assert!(RateLimit::default().is_unrestricted());
    }

    #[test]
    fn omits_none_fields_on_serialise() {
        let rl = RateLimit {
            rpm: Some(60),
            ..Default::default()
        };
        let json = serde_json::to_value(&rl).unwrap();
        assert_eq!(json["rpm"], 60);
        assert!(json.get("tpm").is_none());
        assert!(json.get("concurrency").is_none());
    }

    #[test]
    fn tolerates_unknown_fields_for_forward_compat() {
        // cp-api may ship new fields ahead of the DP rolling out; serde must
        // accept them (the write path still rejects them via the strict
        // schema validators in models/schema.rs).
        let rl: RateLimit = serde_json::from_str(r#"{"rpm": 10, "extra": 1}"#).unwrap();
        assert_eq!(rl.rpm, Some(10));
    }
}
