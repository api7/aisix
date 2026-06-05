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
//! Kinds: `otlp_http` (Tempo / Loki / Jaeger / Honeycomb / Grafana Cloud /
//! Langfuse-via-OTLP — anything speaking OTLP/HTTP) and `aliyun_sls`
//! (Aliyun SLS PutLogs). Further targets (Datadog / S3 / …) land as
//! additional kinds.
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

/// Discriminated union of exporter back-ends. Ships `otlp_http` and
/// `aliyun_sls`; Datadog / S3 / … land in follow-ups, each as a new
/// variant whose serde tag matches the wire-side `kind` discriminator.
///
/// `tag = "kind"` puts the variant tag inline with the inner struct's
/// fields — same shape as `GuardrailKind` so the kine wire stays
/// consistent across resource types.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExporterKind {
    OtlpHttp(OtlpHttpConfig),
    AliyunSls(AliyunSlsConfig),
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
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

/// Aliyun SLS (Simple Log Service) PutLogs target. Unlike `otlp_http`,
/// the AccessKey is **never** part of this config: it would otherwise
/// sit in plaintext on the kine path, and SLS keys grant broad account
/// access. Instead the config carries a [`credential_ref`] pointer that
/// the customer-side DP resolves to the real key locally (env / mounted
/// secret); API7's control plane stores only the reference. This is what
/// lets the DP run in the customer's environment without API7 ever
/// holding the plaintext key.
///
/// [`credential_ref`]: AliyunSlsConfig::credential_ref
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AliyunSlsConfig {
    /// SLS region endpoint host with no scheme, e.g.
    /// `ap-southeast-3.log.aliyuncs.com`. The request host the DP signs
    /// and posts to is `<project>.<endpoint>`.
    pub endpoint: String,

    /// SLS project name (the `<project>` in the request host).
    pub project: String,

    /// SLS logstore that receives the request-event logs.
    pub logstore: String,

    /// Opaque pointer to the AccessKey credential, resolved locally by
    /// the DP at delivery time. The plaintext AccessKey MUST NOT live in
    /// etcd/kine — the control plane stores only this reference, never the
    /// key itself. BYOK variants (customer KMS / uploaded key) resolve the
    /// same reference through a different unwrap path in a later phase.
    pub credential_ref: String,
}

/// Top-level `ObservabilityExporter` resource. `deny_unknown_fields`
/// deliberately NOT set — serde's `flatten` + `tag = "kind"`
/// interaction makes outer-strict-mode reject the inner discriminator
/// field. Strict typo rejection happens at the JSON Schema layer
/// (`schema::validate_observability_exporter`) which the etcd loader
/// runs before the serde deserialize.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
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
            other => panic!("expected otlp_http, got {other:?}"),
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
            other => panic!("expected otlp_http, got {other:?}"),
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

    const VALID_SLS: &str = r#"{
        "name": "sls-prod",
        "enabled": true,
        "kind": "aliyun_sls",
        "endpoint": "ap-southeast-3.log.aliyuncs.com",
        "project": "aisix-obs",
        "logstore": "request-events",
        "credential_ref": "sls-prod"
    }"#;

    #[test]
    fn deserialises_aliyun_sls() {
        let e: ObservabilityExporter = serde_json::from_str(VALID_SLS).unwrap();
        assert_eq!(e.name, "sls-prod");
        assert!(e.enabled);
        match &e.kind {
            ExporterKind::AliyunSls(c) => {
                assert_eq!(c.endpoint, "ap-southeast-3.log.aliyuncs.com");
                assert_eq!(c.project, "aisix-obs");
                assert_eq!(c.logstore, "request-events");
                assert_eq!(c.credential_ref, "sls-prod");
            }
            other => panic!("expected aliyun_sls, got {other:?}"),
        }
    }

    #[test]
    fn aliyun_sls_round_trips_flat() {
        let e: ObservabilityExporter = serde_json::from_str(VALID_SLS).unwrap();
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(v["kind"], "aliyun_sls");
        assert_eq!(v["project"], "aisix-obs");
        assert_eq!(v["credential_ref"], "sls-prod");
        assert!(v.get("aliyun_sls").is_none(), "kind block must not nest");
    }

    #[test]
    fn rejects_plaintext_credentials_in_config() {
        // The AccessKey must NEVER be a config field — only a
        // `credential_ref`. `deny_unknown_fields` on the inner config
        // rejects any attempt to smuggle a plaintext key onto the kine path.
        for key in ["access_key_secret", "access_key_id", "ak", "sk"] {
            let json = format!(
                r#"{{"name":"x","kind":"aliyun_sls","endpoint":"ap-southeast-3.log.aliyuncs.com","project":"p","logstore":"l","credential_ref":"r","{key}":"AKIASECRET"}}"#
            );
            let r: Result<ObservabilityExporter, _> = serde_json::from_str(&json);
            assert!(
                r.is_err(),
                "plaintext credential field `{key}` must be rejected"
            );
        }
    }
}
