//! `ObservabilityExporter` — env-scoped fan-out target the DP ships
//! per-request telemetry to.
//!
//! cp-api writes one row per configured exporter to
//! `/aisix/<env>/observability_exporters/<uuid>`; the DP loads them on
//! the watch and composes a fan-out sink that runs alongside the
//! existing `usage::UsageSink` (which still feeds the cp-api telemetry
//! pipeline). The DP is the authoritative consumer of these configs —
//! cp-api never opens an HTTP connection to the user's exporter
//! endpoint, which is the whole reason for DP-direct egress (sensitive
//! prompt / response content stays on the data plane).
//!
//! MVP scope: `kind = "otlp_http"` only — covers Tempo / Loki / Jaeger
//! / Honeycomb / Grafana Cloud / Langfuse-via-OTLP because all of them
//! accept the OTLP/HTTP wire format.
//!
//! Wire shape on kine — flat object with the kind-tagged config fields
//! at top level, matching the `Guardrail` pattern in this crate:
//!
//! ```json
//! {
//!   "name": "honeycomb-prod",
//!   "enabled": true,
//!   "kind": "otlp_http",
//!   "endpoint": "https://api.honeycomb.io/v1/traces",
//!   "headers": { "x-honeycomb-team": "abc..." }
//! }
//! ```

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::resource::Resource;

/// Discriminated union of exporter back-ends. MVP ships only
/// `otlp_http`; Helicone / Langfuse-native / Datadog / S3 land in
/// follow-ups, each as a new variant whose serde tag matches the
/// wire-side `kind` discriminator.
///
/// `tag = "kind"` puts the variant tag inline with the inner struct's
/// fields — same shape as `GuardrailKind` so the kine wire stays
/// consistent across resource types.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExporterKind {
    OtlpHttp(OtlpHttpConfig),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OtlpHttpConfig {
    /// Full URL of the OTLP/HTTP traces endpoint. Must already include
    /// the `/v1/traces` path the receiver expects — we don't append it
    /// because some vendors (Honeycomb, Grafana) use a different path.
    pub endpoint: String,

    /// Static headers to attach to every export request. Typical use:
    /// `Authorization: Bearer <api-token>` or vendor-specific keys
    /// like `x-honeycomb-team`. Values are plaintext at this MVP — the
    /// kine path is mTLS-only, so the trust boundary matches
    /// `provider_keys`. Field-level encryption arrives in Phase 2.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
}

/// Top-level `ObservabilityExporter` resource. `deny_unknown_fields`
/// deliberately NOT set — serde's `flatten` + `tag = "kind"`
/// interaction makes outer-strict-mode reject the inner discriminator
/// field. Strict typo rejection happens at the JSON Schema layer
/// (`schema::validate_observability_exporter`) which the etcd loader
/// runs before the serde deserialize.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ObservabilityExporter {
    /// Operator-facing label, surfaced in /logs and the dashboard list.
    /// Not used for routing — the etcd-key uuid is the identity.
    pub name: String,

    /// Soft kill switch. Disabled exporters stay in the snapshot but
    /// the fan-out sink skips them. Lets operators pause an exporter
    /// without losing the row's headers / endpoint.
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Discriminated kind block — flattened so `kind` and the inner
    /// struct's fields land at the same level on the wire.
    #[serde(flatten)]
    pub kind: ExporterKind,

    /// etcd-key uuid; filled by the loader, never in the JSON payload.
    #[serde(skip)]
    pub(crate) runtime_id: String,
}

const fn default_true() -> bool {
    true
}

impl Resource for ObservabilityExporter {
    fn id(&self) -> &str {
        &self.runtime_id
    }

    /// Name doubles as the secondary index for human lookup. Identity
    /// is the runtime_id (uuid); the name is only used for log /
    /// dashboard display.
    fn name(&self) -> &str {
        &self.name
    }

    /// Path segment under `/aisix/<env>/`. Matches the cp-api kine
    /// kind written by the Go-side `mustMarshalObservabilityExporterKV`.
    fn kind() -> &'static str {
        "observability_exporters"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_OTLP: &str = r#"{
        "name": "honeycomb-prod",
        "enabled": true,
        "kind": "otlp_http",
        "endpoint": "https://api.honeycomb.io/v1/traces",
        "headers": { "x-honeycomb-team": "abc123" }
    }"#;

    #[test]
    fn deserialises_otlp_http() {
        let e: ObservabilityExporter = serde_json::from_str(VALID_OTLP).unwrap();
        assert_eq!(e.name, "honeycomb-prod");
        assert!(e.enabled);
        match &e.kind {
            ExporterKind::OtlpHttp(c) => {
                assert_eq!(c.endpoint, "https://api.honeycomb.io/v1/traces");
                assert_eq!(
                    c.headers.get("x-honeycomb-team").map(String::as_str),
                    Some("abc123"),
                );
            }
        }
    }

    #[test]
    fn enabled_defaults_to_true() {
        let e: ObservabilityExporter =
            serde_json::from_str(r#"{"name":"x","kind":"otlp_http","endpoint":"https://x"}"#)
                .unwrap();
        assert!(e.enabled);
    }

    #[test]
    fn empty_headers_round_trip() {
        let e: ObservabilityExporter =
            serde_json::from_str(r#"{"name":"x","kind":"otlp_http","endpoint":"https://x"}"#)
                .unwrap();
        match &e.kind {
            ExporterKind::OtlpHttp(c) => assert!(c.headers.is_empty()),
        }
    }

    #[test]
    fn rejects_unknown_kind() {
        // serde rejects unknown discriminator values for the tagged enum.
        let r: Result<ObservabilityExporter, _> =
            serde_json::from_str(r#"{"name":"x","kind":"slack_alert","webhook":"https://x"}"#);
        assert!(r.is_err(), "unknown kind should fail to parse");
    }

    #[test]
    fn rejects_unknown_inner_field() {
        // OtlpHttpConfig has deny_unknown_fields so a typo at the
        // flattened level is rejected.
        let r: Result<ObservabilityExporter, _> = serde_json::from_str(
            r#"{"name":"x","kind":"otlp_http","endpoint":"https://x","timeout":5}"#,
        );
        assert!(r.is_err());
    }

    #[test]
    fn resource_trait_kind_matches_cp_api() {
        // The kine kind segment is the contract between cp-api's
        // `mustMarshalObservabilityExporterKV` and this loader. Pin it.
        assert_eq!(
            <ObservabilityExporter as Resource>::kind(),
            "observability_exporters",
        );
    }

    #[test]
    fn round_trips_through_serde() {
        let e: ObservabilityExporter = serde_json::from_str(VALID_OTLP).unwrap();
        let v = serde_json::to_value(&e).unwrap();
        // The wire shape must be flat — `endpoint` and `headers` at
        // the top level, NOT nested under `otlp_http`.
        assert_eq!(v["kind"], "otlp_http");
        assert_eq!(v["endpoint"], "https://api.honeycomb.io/v1/traces");
        assert!(v.get("otlp_http").is_none(), "kind block must not nest");
    }
}
