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

/// Exporter backend selected by the `kind` discriminator.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExporterKind {
    OtlpHttp(OtlpHttpConfig),
    AliyunSls(AliyunSlsConfig),
    ObjectStore(ObjectStoreConfig),
    Datadog(DatadogConfig),
}

/// OTLP/HTTP trace exporter configuration.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OtlpHttpConfig {
    /// Full URL of the OTLP/HTTP traces endpoint. Include the receiver's
    /// expected path, such as `/v1/traces`.
    pub endpoint: String,

    /// Static headers attached to every export request, such as authorization or vendor-specific API-key headers.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,

    /// Fraction of requests exported as traces, from `0.0` to `1.0`. Defaults to `1.0`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 0.0, max = 1.0))]
    pub sample_rate: Option<f64>,

    /// Controls whether spans include prompt and response content. `metadata_only` omits content. `full` includes content truncated by `content_max_bytes`.
    #[serde(default)]
    pub content_mode: SlsContentMode,

    /// Maximum bytes per captured prompt or response field when `content_mode` is `full`. Defaults to 128 KiB.
    #[serde(default = "default_content_max_bytes")]
    #[schemars(range(min = 1))]
    pub content_max_bytes: u32,
}

/// Aliyun SLS PutLogs exporter configuration.
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

    /// Credential reference resolved by the data plane at delivery time. The plaintext AccessKey is not stored in this resource.
    pub credential_ref: String,

    /// Controls whether logs include prompt and response content. `metadata_only` omits content. `full` includes content truncated by `content_max_bytes`.
    #[serde(default)]
    pub content_mode: SlsContentMode,

    /// Maximum bytes per captured prompt or response field when `content_mode` is `full`. Defaults to 128 KiB.
    #[serde(default = "default_content_max_bytes")]
    #[schemars(range(min = 1))]
    pub content_max_bytes: u32,
}

/// Content-capture mode for an observability exporter.
#[derive(
    Debug, Clone, Copy, Default, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq, Hash,
)]
#[serde(rename_all = "snake_case")]
pub enum SlsContentMode {
    /// Operational metadata only. Prompt and response content are omitted.
    #[default]
    MetadataOnly,
    /// Metadata plus the captured request prompt and assembled response.
    Full,
}

/// Default per-field content cap: 128 KiB.
const fn default_content_max_bytes() -> u32 {
    128 * 1024
}

/// Datadog native Logs HTTP intake exporter configuration.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DatadogConfig {
    /// Datadog site, such as `datadoghq.com`, `us3.datadoghq.com`, or `datadoghq.eu`.
    pub site: String,

    /// Credential reference resolved by the data plane at delivery time. The plaintext Datadog API key is not stored in this resource.
    pub credential_ref: String,

    /// Datadog `service` reserved attribute. Every log from this exporter is
    /// tagged with this service name in Datadog Log Explorer.
    pub service: String,

    /// Datadog `ddsource` reserved attribute. Identifies the integration or source.
    /// Defaults to `aisix-ai-gateway`.
    #[serde(default = "default_ddsource")]
    pub ddsource: String,

    /// Operator-defined tags rendered into Datadog's comma-joined `ddtags`
    /// reserved attribute (e.g. `["team:platform", "tier:prod"]` →
    /// `team:platform,tier:prod`). Empty by default.
    #[serde(default)]
    pub tags: Vec<String>,

    /// Controls whether logs include prompt and response content. `metadata_only` omits content. `full` includes content truncated by `content_max_bytes`.
    #[serde(default)]
    pub content_mode: SlsContentMode,

    /// Maximum bytes per captured prompt or response field when `content_mode` is `full`. Keep this under Datadog intake limits to avoid delivery errors.
    #[serde(default = "default_dd_content_max_bytes")]
    #[schemars(range(min = 1, max = 1_048_576))]
    pub content_max_bytes: u32,
}

/// Default Datadog `ddsource` reserved attribute.
fn default_ddsource() -> String {
    "aisix-ai-gateway".to_string()
}

/// Default per-field content cap for a Datadog exporter: 128 KiB.
const fn default_dd_content_max_bytes() -> u32 {
    128 * 1024
}

/// Object-storage exporter configuration for S3, GCS, Azure Blob, and compatible S3 backends.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ObjectStoreConfig {
    /// Which object-storage backend the bucket lives in.
    pub provider: ObjectStoreProvider,

    /// Bucket for S3 or GCS, or container for Azure Blob, that receives exported files.
    pub bucket: String,

    /// Key prefix the partition path is appended to, e.g. `ai-gateway`.
    /// The full key is `<prefix>/org=…/env=…/table=…/dt=…/hh=…/<file>`.
    pub prefix: String,

    /// AWS region for S3 SigV4 signature scope. Recommended because `object_store`
    /// defaults to `us-east-1` when unset, so a non-default-region bucket
    /// should set it. Ignored for GCS / Azure Blob.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,

    /// Backend host override for S3-compatible stores such as MinIO, Aliyun OSS, or Cloudflare R2. When omitted, the provider's native endpoint is used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,

    /// Compression applied to each NDJSON file before upload. Default gzip.
    #[serde(default)]
    pub compression: ObjectStoreCompression,

    /// How the DP authenticates to the bucket. Default `credential_ref`.
    #[serde(default)]
    pub auth_mode: ObjectStoreAuthMode,

    /// Credential reference resolved by the data plane at delivery time. Required when `auth_mode` is `credential_ref`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub credential_ref: String,
}

/// Object-storage backend selector. The sink builds one backend client per
/// variant. Batching, key layout, and retry behavior are shared.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ObjectStoreProvider {
    S3,
    Gcs,
    AzureBlob,
}

/// File compression for object-storage uploads.
#[derive(
    Debug, Clone, Copy, Default, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq, Hash,
)]
#[serde(rename_all = "snake_case")]
pub enum ObjectStoreCompression {
    /// gzip compression (RFC 1952). This is the default and is accepted by Snowpipe and Auto Loader.
    #[default]
    Gzip,
    /// No compression. Emits raw NDJSON.
    None,
}

/// How the DP obtains credentials for the object-storage bucket.
#[derive(
    Debug, Clone, Copy, Default, Serialize, Deserialize, schemars::JsonSchema, PartialEq, Eq, Hash,
)]
#[serde(rename_all = "snake_case")]
pub enum ObjectStoreAuthMode {
    /// Resolve `credential_ref` to static keys from the DP's local env
    /// (`OBJSTORE_CRED_<SLUG>_<FIELD>`). The default.
    #[default]
    CredentialRef,
    /// Use the data plane host's attached cloud identity. Supported for S3 and GCS only.
    CloudIdentity,
}

/// Telemetry exporter configuration.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema, PartialEq)]
pub struct ObservabilityExporter {
    /// Operator-facing label, surfaced in logs and dashboard lists. The etcd
    /// key UUID is the resource identity.
    pub name: String,

    /// Whether this exporter is active. Disabled exporters remain configured but do not receive telemetry.
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Discriminated kind block — flattened so `kind` and the inner
    /// struct's fields land at the same level on the wire.
    #[serde(flatten)]
    pub kind: ExporterKind,

    /// etcd-key uuid. Filled by the loader and never included in the JSON payload.
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
    /// is the runtime_id UUID. The name is only used for log /
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
    fn otlp_http_defaults_pin_pre_knob_behaviour() {
        // A CP payload carrying only endpoint+headers (the current kine
        // projection) must parse unchanged: no sampling (rate absent =
        // 1.0) and metadata-only content — the exact pre-knob behaviour.
        let e: ObservabilityExporter = serde_json::from_str(VALID_OTLP).unwrap();
        match &e.kind {
            ExporterKind::OtlpHttp(c) => {
                assert_eq!(c.sample_rate, None);
                assert_eq!(c.content_mode, SlsContentMode::MetadataOnly);
                assert_eq!(c.content_max_bytes, 128 * 1024);
            }
            other => panic!("expected otlp_http, got {other:?}"),
        }
        // Absent knobs stay off the wire on re-serialize, so a round-trip
        // through the DP cannot change what the CP wrote.
        let v = serde_json::to_value(&e).unwrap();
        assert!(v.get("sample_rate").is_none());
    }

    #[test]
    fn otlp_http_opts_into_sampling_and_full_content_capture() {
        let json = r#"{
            "name": "otlp-knobs",
            "kind": "otlp_http",
            "endpoint": "https://api.honeycomb.io/v1/traces",
            "sample_rate": 0.25,
            "content_mode": "full",
            "content_max_bytes": 4096
        }"#;
        let e: ObservabilityExporter = serde_json::from_str(json).unwrap();
        match &e.kind {
            ExporterKind::OtlpHttp(c) => {
                assert_eq!(c.sample_rate, Some(0.25));
                assert_eq!(c.content_mode, SlsContentMode::Full);
                assert_eq!(c.content_max_bytes, 4096);
            }
            other => panic!("expected otlp_http, got {other:?}"),
        }
        // Round-trips flat with the knobs present.
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(v["kind"], "otlp_http");
        assert_eq!(v["sample_rate"], 0.25);
        assert_eq!(v["content_mode"], "full");
        assert_eq!(v["content_max_bytes"], 4096);
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
                // Content capture is off by default (privacy-preserving).
                assert_eq!(c.content_mode, SlsContentMode::MetadataOnly);
                assert_eq!(c.content_max_bytes, 128 * 1024);
            }
            other => panic!("expected aliyun_sls, got {other:?}"),
        }
    }

    #[test]
    fn aliyun_sls_opts_into_full_content_capture() {
        let json = r#"{
            "name": "sls-content",
            "kind": "aliyun_sls",
            "endpoint": "ap-southeast-3.log.aliyuncs.com",
            "project": "p",
            "logstore": "l",
            "credential_ref": "r",
            "content_mode": "full",
            "content_max_bytes": 4096
        }"#;
        let e: ObservabilityExporter = serde_json::from_str(json).unwrap();
        match &e.kind {
            ExporterKind::AliyunSls(c) => {
                assert_eq!(c.content_mode, SlsContentMode::Full);
                assert_eq!(c.content_max_bytes, 4096);
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

    const VALID_OBJECT_STORE: &str = r#"{
        "name": "acme-s3-events",
        "enabled": true,
        "kind": "object_store",
        "provider": "s3",
        "bucket": "acme-aisix-events",
        "prefix": "ai-gateway",
        "region": "us-east-1",
        "compression": "gzip",
        "credential_ref": "acme-s3"
    }"#;

    #[test]
    fn deserialises_object_store() {
        let e: ObservabilityExporter = serde_json::from_str(VALID_OBJECT_STORE).unwrap();
        assert_eq!(e.name, "acme-s3-events");
        assert!(e.enabled);
        match &e.kind {
            ExporterKind::ObjectStore(c) => {
                assert_eq!(c.provider, ObjectStoreProvider::S3);
                assert_eq!(c.bucket, "acme-aisix-events");
                assert_eq!(c.prefix, "ai-gateway");
                assert_eq!(c.region.as_deref(), Some("us-east-1"));
                assert_eq!(c.compression, ObjectStoreCompression::Gzip);
                assert_eq!(c.credential_ref, "acme-s3");
            }
            other => panic!("expected object_store, got {other:?}"),
        }
    }

    #[test]
    fn object_store_defaults_compression_to_gzip_and_omits_optionals() {
        // gcs with no region/endpoint/compression — defaults + Options apply.
        let e: ObservabilityExporter = serde_json::from_str(
            r#"{"name":"x","kind":"object_store","provider":"gcs","bucket":"b","prefix":"p","credential_ref":"r"}"#,
        )
        .unwrap();
        match &e.kind {
            ExporterKind::ObjectStore(c) => {
                assert_eq!(c.provider, ObjectStoreProvider::Gcs);
                assert_eq!(c.compression, ObjectStoreCompression::Gzip);
                assert!(c.region.is_none());
                assert!(c.endpoint.is_none());
            }
            other => panic!("expected object_store, got {other:?}"),
        }
    }

    #[test]
    fn object_store_round_trips_flat() {
        let e: ObservabilityExporter = serde_json::from_str(VALID_OBJECT_STORE).unwrap();
        let v = serde_json::to_value(&e).unwrap();
        // Flat wire — kind tag + fields at the top level, never nested.
        assert_eq!(v["kind"], "object_store");
        assert_eq!(v["provider"], "s3");
        assert_eq!(v["bucket"], "acme-aisix-events");
        assert_eq!(v["credential_ref"], "acme-s3");
        assert!(v.get("object_store").is_none(), "kind block must not nest");
    }

    #[test]
    fn object_store_auth_mode_defaults_and_cloud_identity() {
        // Absent auth_mode → credential_ref (back-compat default).
        let e: ObservabilityExporter = serde_json::from_str(VALID_OBJECT_STORE).unwrap();
        match &e.kind {
            ExporterKind::ObjectStore(c) => {
                assert_eq!(c.auth_mode, ObjectStoreAuthMode::CredentialRef);
            }
            other => panic!("expected object_store, got {other:?}"),
        }
        // cloud_identity deserializes and credential_ref may be omitted.
        let e: ObservabilityExporter = serde_json::from_str(
            r#"{"name":"x","kind":"object_store","provider":"s3","bucket":"b","prefix":"p","auth_mode":"cloud_identity"}"#,
        )
        .unwrap();
        match &e.kind {
            ExporterKind::ObjectStore(c) => {
                assert_eq!(c.auth_mode, ObjectStoreAuthMode::CloudIdentity);
                assert_eq!(c.credential_ref, "");
            }
            other => panic!("expected object_store, got {other:?}"),
        }
        // An empty credential_ref is omitted on the wire (skip_serializing_if),
        // so a keyless config never carries an empty key reference.
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(v["auth_mode"], "cloud_identity");
        assert!(
            v.get("credential_ref").is_none(),
            "empty credential_ref must be omitted under cloud_identity, got {v}"
        );
    }

    #[test]
    fn rejects_plaintext_credentials_in_object_store_config() {
        // Cloud keys must NEVER be config fields — only `credential_ref`.
        // `deny_unknown_fields` rejects any smuggled plaintext secret.
        for key in [
            "access_key_id",
            "secret_access_key",
            "sas_token",
            "service_account_json",
        ] {
            let json = format!(
                r#"{{"name":"x","kind":"object_store","provider":"s3","bucket":"b","prefix":"p","credential_ref":"r","{key}":"PLAINTEXT"}}"#
            );
            let r: Result<ObservabilityExporter, _> = serde_json::from_str(&json);
            assert!(
                r.is_err(),
                "plaintext credential field `{key}` must be rejected"
            );
        }
    }

    const VALID_DATADOG: &str = r#"{
        "name": "datadog-prod",
        "enabled": true,
        "kind": "datadog",
        "site": "datadoghq.com",
        "credential_ref": "datadog-prod",
        "service": "ai-gateway"
    }"#;

    #[test]
    fn deserialises_datadog() {
        let e: ObservabilityExporter = serde_json::from_str(VALID_DATADOG).unwrap();
        assert_eq!(e.name, "datadog-prod");
        assert!(e.enabled);
        match &e.kind {
            ExporterKind::Datadog(c) => {
                assert_eq!(c.site, "datadoghq.com");
                assert_eq!(c.credential_ref, "datadog-prod");
                assert_eq!(c.service, "ai-gateway");
                // ddsource defaults; tags empty.
                assert_eq!(c.ddsource, "aisix-ai-gateway");
                assert!(c.tags.is_empty());
                // Content capture is off by default (privacy-preserving).
                assert_eq!(c.content_mode, SlsContentMode::MetadataOnly);
                assert_eq!(c.content_max_bytes, 128 * 1024);
            }
            other => panic!("expected datadog, got {other:?}"),
        }
    }

    #[test]
    fn datadog_opts_into_full_content_capture_and_tags() {
        let json = r#"{
            "name": "datadog-content",
            "kind": "datadog",
            "site": "datadoghq.eu",
            "credential_ref": "r",
            "service": "ai-gateway",
            "ddsource": "custom-source",
            "tags": ["team:platform", "tier:prod"],
            "content_mode": "full",
            "content_max_bytes": 4096
        }"#;
        let e: ObservabilityExporter = serde_json::from_str(json).unwrap();
        match &e.kind {
            ExporterKind::Datadog(c) => {
                assert_eq!(c.site, "datadoghq.eu");
                assert_eq!(c.ddsource, "custom-source");
                assert_eq!(c.tags, vec!["team:platform", "tier:prod"]);
                assert_eq!(c.content_mode, SlsContentMode::Full);
                assert_eq!(c.content_max_bytes, 4096);
            }
            other => panic!("expected datadog, got {other:?}"),
        }
    }

    #[test]
    fn datadog_round_trips_flat() {
        let e: ObservabilityExporter = serde_json::from_str(VALID_DATADOG).unwrap();
        let v = serde_json::to_value(&e).unwrap();
        // Flat wire — kind tag + fields at the top level, never nested.
        assert_eq!(v["kind"], "datadog");
        assert_eq!(v["site"], "datadoghq.com");
        assert_eq!(v["credential_ref"], "datadog-prod");
        assert_eq!(v["service"], "ai-gateway");
        assert!(v.get("datadog").is_none(), "kind block must not nest");
    }

    #[test]
    fn rejects_plaintext_api_key_in_datadog_config() {
        // The Datadog API key must NEVER be a config field — only a
        // `credential_ref`. `deny_unknown_fields` on the inner config rejects
        // any attempt to smuggle a plaintext key onto the kine path.
        for key in ["api_key", "apikey", "dd_api_key", "key"] {
            let json = format!(
                r#"{{"name":"x","kind":"datadog","site":"datadoghq.com","credential_ref":"r","service":"s","{key}":"DDSECRET"}}"#
            );
            let r: Result<ObservabilityExporter, _> = serde_json::from_str(&json);
            assert!(
                r.is_err(),
                "plaintext credential field `{key}` must be rejected"
            );
        }
    }
}
