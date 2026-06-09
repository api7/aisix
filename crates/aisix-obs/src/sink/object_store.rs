//! Object-storage sink — ONE implementation covering S3 / GCS / Azure Blob
//! (and S3-compatible MinIO / Cloudflare R2) behind the `object_store`
//! crate's single `ObjectStore` trait. The sink is written **once**: only
//! [`build_object_store`] is provider-specific (it picks the right builder,
//! which owns SigV4 / GCS OAuth / Azure shared-key signing). Everything
//! else — NDJSON encoding, gzip, the partitioned object key, error
//! classification — is shared across all three backends.
//!
//! A batch becomes ONE object: the records are serialized to NDJSON (one
//! [`SinkRecord`] per line, including opt-in captured content), optionally
//! gzipped, and `put` under a **content-addressed** key
//! `…/dt=YYYY-MM-DD/hh=HH/<sha256(body)[..32]>.ndjson[.gz]`. The content
//! address makes an *in-process* retry reuse the **same** key (the pipeline
//! retries the same batch on a transient error), so a downstream Snowpipe /
//! Auto Loader dedups that retry by filename and does not double-load — the
//! load-bearing property in AISIX-Cloud#689 §8. This is at-least-once across a
//! DP restart (a re-batch yields a new key); restart-stable `FileSequence`
//! markers and org/env partitioning are tracked as follow-ups.

use std::sync::Arc;

use async_trait::async_trait;
use object_store::path::Path as ObjectPath;
// `put` / `get` are on the `ObjectStoreExt` extension trait in object_store
// 0.13; `ObjectStore` itself must also be in scope for `dyn ObjectStore`.
use object_store::{ObjectStore, ObjectStoreExt, PutPayload};
use sha2::{Digest, Sha256};

use aisix_core::models::observability_exporter::{
    ObjectStoreAuthMode, ObjectStoreCompression, ObjectStoreConfig, ObjectStoreProvider,
};

use super::{
    BatchUnit, EventBatch, IdempotencyMarker, IdempotencyScheme, ObservabilitySink, OrderingScope,
    SinkAck, SinkCapabilities, SinkError, SinkHealth, SinkResult,
};

/// Cap on a masked error-detail string surfaced to logs / health.
const DETAIL_MAX_CHARS: usize = 200;

/// A delivery target for one object-storage bucket (any provider).
pub struct ObjectStoreSink {
    name: String,
    /// Provider-agnostic backend handle, built by [`build_object_store`].
    store: Arc<dyn ObjectStore>,
    /// Key prefix the partition path is appended to.
    prefix: String,
    compression: ObjectStoreCompression,
}

impl ObjectStoreSink {
    /// Build a sink over an already-constructed backend. Production goes
    /// through [`build_object_store`] to make the `store`; tests pass an
    /// `InMemory` / `LocalFileSystem` store directly.
    pub fn new(
        name: impl Into<String>,
        store: Arc<dyn ObjectStore>,
        prefix: impl Into<String>,
        compression: ObjectStoreCompression,
    ) -> Self {
        Self {
            name: name.into(),
            store,
            prefix: prefix.into(),
            compression,
        }
    }

    /// Serialize a batch to (optionally gzipped) NDJSON bytes — the object
    /// body. One line per record; opt-in `content` rides along when present,
    /// and is absent on the default metadata-only path.
    fn encode(&self, batch: &EventBatch) -> Result<Vec<u8>, SinkError> {
        let mut ndjson = Vec::with_capacity(batch.len() * 512);
        for record in &batch.records {
            serde_json::to_writer(&mut ndjson, record.as_ref())
                .map_err(|e| SinkError::Permanent(format!("object_store: ndjson encode: {e}")))?;
            ndjson.push(b'\n');
        }
        match self.compression {
            ObjectStoreCompression::None => Ok(ndjson),
            ObjectStoreCompression::Gzip => gzip(&ndjson),
        }
    }

    /// Content-addressed object key. Deterministic in `(body, occurred_at)`,
    /// so a retried batch maps to the same key (idempotent put + downstream
    /// filename dedup). The `dt`/`hh` Hive partitions come from the first
    /// record's event time, so they too are stable across a retry.
    fn object_key(&self, body: &[u8], occurred_at: &str) -> String {
        let (date, hour) = partition(occurred_at);
        let digest = sha256_hex16(body);
        let ext = match self.compression {
            ObjectStoreCompression::Gzip => "ndjson.gz",
            ObjectStoreCompression::None => "ndjson",
        };
        let prefix = self.prefix.trim_matches('/');
        if prefix.is_empty() {
            format!("dt={date}/hh={hour}/{digest}.{ext}")
        } else {
            format!("{prefix}/dt={date}/hh={hour}/{digest}.{ext}")
        }
    }
}

#[async_trait]
impl ObservabilitySink for ObjectStoreSink {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> SinkCapabilities {
        SinkCapabilities {
            // At-least-once. Content-addressed keys let a downstream loader
            // dedup an *in-process* pipeline retry by filename (same batch →
            // same body → same key), but this is NOT a restart-stable monotonic
            // sequence: a re-batch after a DP restart yields a new key. So it
            // declares `None` (like the SLS sink) until real `FileSequence`
            // markers land (follow-up) — don't overstate the guarantee here.
            idempotency: IdempotencyScheme::None,
            // One object per flush; no cross-object ordering requirement.
            ordering: OrderingScope::None,
            // Count-bounded by the pipeline today (bodies stay small on the
            // metadata-only path). Byte-aware chunking is required before full
            // prompt/response content rides these records — else a large-content
            // batch buffers raw + gzip in RAM unbounded per flush. Declaring a
            // ceiling the sink can't yet self-enforce would be a false promise,
            // so it stays `None` until rollover/chunking lands (follow-up).
            batch_unit: BatchUnit::Bytes,
            max_batch_bytes: None,
            // The object uploads whole or is retried whole.
            supports_partial_batch: false,
            supports_streaming_ingest: false,
        }
    }

    async fn append_batch(&self, batch: &EventBatch, _marker: &IdempotencyMarker) -> SinkResult {
        if batch.is_empty() {
            return Ok(SinkAck::default());
        }
        let body = self.encode(batch)?;
        let occurred_at = batch
            .records
            .first()
            .map(|r| r.usage.occurred_at.as_str())
            .unwrap_or("");
        let key = self.object_key(&body, occurred_at);
        let path = ObjectPath::parse(&key)
            .map_err(|e| SinkError::Permanent(format!("object_store: bad key {key}: {e}")))?;

        match self.store.put(&path, PutPayload::from(body)).await {
            Ok(_) => Ok(SinkAck {
                accepted: batch.len(),
                ..SinkAck::default()
            }),
            Err(e) => Err(map_object_store_err(e)),
        }
    }

    async fn healthcheck(&self) -> SinkHealth {
        // A real connectivity probe (and the CP "test connection" affordance)
        // lands with the health surface; until then delivery errors surface
        // via `SinkStats::last_error` (mirrors the other sinks).
        SinkHealth::healthy()
    }
}

/// Provider-resolved cloud credentials. The plaintext key never lives in the
/// exporter config / kine path — it is resolved DP-side from a
/// `credential_ref` (see [`resolve_object_store_credential`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObjectStoreCredentials {
    S3 {
        access_key_id: String,
        secret_access_key: String,
        session_token: Option<String>,
    },
    Gcs {
        /// Service-account JSON.
        service_account_key: String,
    },
    Azure {
        account: String,
        access_key: String,
    },
}

/// Build the provider-agnostic [`ObjectStore`] handle for a configured
/// exporter. **This is the only provider-specific code** — each arm picks the
/// matching `object_store` builder, which owns that cloud's signing. An
/// `endpoint` (S3-compatible MinIO / R2, or an emulator) is honored and HTTP
/// is allowed only for explicit `http://` endpoints (loopback emulators).
pub fn build_object_store(
    provider: ObjectStoreProvider,
    bucket: &str,
    region: Option<&str>,
    endpoint: Option<&str>,
    creds: ObjectStoreCredentials,
) -> Result<Arc<dyn ObjectStore>, SinkError> {
    match (provider, creds) {
        (
            ObjectStoreProvider::S3,
            ObjectStoreCredentials::S3 {
                access_key_id,
                secret_access_key,
                session_token,
            },
        ) => {
            let mut b = object_store::aws::AmazonS3Builder::new()
                .with_bucket_name(bucket)
                .with_access_key_id(access_key_id)
                .with_secret_access_key(secret_access_key);
            if let Some(r) = region {
                b = b.with_region(r);
            }
            if let Some(t) = session_token {
                b = b.with_token(t);
            }
            if let Some(ep) = endpoint {
                b = b.with_endpoint(ep);
                // S3-compatible servers (MinIO / emulators) are path-style and
                // commonly plaintext on loopback.
                b = b.with_virtual_hosted_style_request(false);
                if ep.starts_with("http://") {
                    b = b.with_allow_http(true);
                }
            }
            let store = b
                .build()
                .map_err(|e| SinkError::Permanent(format!("object_store: build s3: {e}")))?;
            Ok(Arc::new(store))
        }
        (
            ObjectStoreProvider::Gcs,
            ObjectStoreCredentials::Gcs {
                service_account_key,
            },
        ) => {
            // GCS has no HTTP-endpoint override on the builder the way S3 /
            // Azure do (`with_url` expects a `gs://` location, not a host). An
            // emulator base URL (fake-gcs-server) or a private endpoint rides
            // the service-account JSON's `gcs_base_url` field instead, so the
            // `endpoint` config is intentionally not applied for GCS.
            let b = object_store::gcp::GoogleCloudStorageBuilder::new()
                .with_bucket_name(bucket)
                .with_service_account_key(service_account_key);
            let store = b
                .build()
                .map_err(|e| SinkError::Permanent(format!("object_store: build gcs: {e}")))?;
            Ok(Arc::new(store))
        }
        (
            ObjectStoreProvider::AzureBlob,
            ObjectStoreCredentials::Azure {
                account,
                access_key,
            },
        ) => {
            let mut b = object_store::azure::MicrosoftAzureBuilder::new()
                .with_container_name(bucket)
                .with_account(account)
                .with_access_key(access_key);
            if let Some(ep) = endpoint {
                if ep.starts_with("http://") {
                    b = b.with_allow_http(true);
                }
                b = b.with_endpoint(ep.to_string());
            }
            let store = b
                .build()
                .map_err(|e| SinkError::Permanent(format!("object_store: build azure: {e}")))?;
            Ok(Arc::new(store))
        }
        // Provider / credential kind mismatch — a wiring bug at the call site.
        (provider, _) => Err(SinkError::Permanent(format!(
            "object_store: credentials do not match provider {provider:?}"
        ))),
    }
}

/// Build the backend using the DP host's OWN attached cloud identity, with no
/// static keys ("cloud identity" auth). **S3** uses the AWS default credential
/// chain via [`object_store::aws::AmazonS3Builder::from_env`] (env / EKS IRSA
/// web-identity / ECS task role / EC2 instance profile); a custom `endpoint`
/// is rejected for S3, since ambient credentials require the provider's
/// metadata service. **GCS** uses
/// Application Default Credentials (GKE Workload Identity / GCE metadata) by
/// constructing the builder with no service-account key. **Azure** is not
/// supported here — its managed identity still needs a non-secret account name
/// the keyless config does not carry; cp-api rejects that combination at create
/// time, and this arm returns a clear permanent error as a backstop.
pub fn build_object_store_ambient(
    provider: ObjectStoreProvider,
    bucket: &str,
    region: Option<&str>,
    endpoint: Option<&str>,
) -> Result<Arc<dyn ObjectStore>, SinkError> {
    match provider {
        ObjectStoreProvider::S3 => {
            // Ambient AWS credentials come from the instance metadata service
            // (IMDS / IRSA / ECS task role), which only the provider's native
            // endpoint exposes. A custom endpoint means an S3-compatible host
            // (MinIO / R2 / OSS) with no cloud IAM identity, so fail fast with a
            // clear permanent error instead of building a sink that fails soft
            // on every delivery. (cp-api also rejects this at create time.)
            if endpoint.is_some() {
                return Err(SinkError::Permanent(
                    "object_store: cloud_identity for s3 does not support a custom \
                     endpoint (ambient AWS credentials require the provider's \
                     metadata service); use credential_ref for S3-compatible hosts"
                        .to_string(),
                ));
            }
            let mut b = object_store::aws::AmazonS3Builder::from_env().with_bucket_name(bucket);
            if let Some(r) = region {
                b = b.with_region(r);
            }
            let store = b.build().map_err(|e| {
                SinkError::Permanent(format!("object_store: build s3 (cloud identity): {e}"))
            })?;
            Ok(Arc::new(store))
        }
        ObjectStoreProvider::Gcs => {
            // No service-account key set → `object_store` sources Application
            // Default Credentials (GKE Workload Identity / GCE metadata).
            let store = object_store::gcp::GoogleCloudStorageBuilder::new()
                .with_bucket_name(bucket)
                .build()
                .map_err(|e| {
                    SinkError::Permanent(format!("object_store: build gcs (cloud identity): {e}"))
                })?;
            Ok(Arc::new(store))
        }
        ObjectStoreProvider::AzureBlob => Err(SinkError::Permanent(
            "object_store: cloud_identity auth is not supported for azure_blob \
             (managed identity needs a non-secret account name the keyless config \
             does not carry); use credential_ref"
                .to_string(),
        )),
    }
}

/// Build the ready-to-run sink for a configured `object_store` exporter:
/// resolve credentials from the DP's local env, construct the backend, and
/// wrap it in an [`ObjectStoreSink`]. If credentials are missing or the
/// backend cannot be built, return a sink that reports the reason on every
/// delivery (and as unhealthy), so the failure surfaces on the exporter-health
/// panel rather than silently dropping events — mirroring the SLS path's
/// "missing creds → auth error, not a silent drop" rule.
pub fn build_object_store_sink(
    name: String,
    cfg: &ObjectStoreConfig,
) -> Arc<dyn ObservabilitySink> {
    let store = match cfg.auth_mode {
        ObjectStoreAuthMode::CredentialRef => {
            let Some(creds) = resolve_object_store_credential(cfg.provider, &cfg.credential_ref)
            else {
                return Arc::new(BrokenSink::new(
                    name,
                    format!(
                        "object_store: no credentials resolved for credential_ref {:?}",
                        cfg.credential_ref
                    ),
                ));
            };
            build_object_store(
                cfg.provider,
                &cfg.bucket,
                cfg.region.as_deref(),
                cfg.endpoint.as_deref(),
                creds,
            )
        }
        ObjectStoreAuthMode::CloudIdentity => build_object_store_ambient(
            cfg.provider,
            &cfg.bucket,
            cfg.region.as_deref(),
            cfg.endpoint.as_deref(),
        ),
    };
    match store {
        Ok(store) => Arc::new(ObjectStoreSink::new(
            name,
            store,
            cfg.prefix.clone(),
            cfg.compression,
        )),
        Err(e) => Arc::new(BrokenSink::new(name, e.to_string())),
    }
}

/// A placeholder for an exporter that could not be constructed (missing
/// credentials / un-buildable backend). It never drops silently: every
/// delivery fails permanently with the reason — recorded by the pipeline as
/// `last_error` and shown on the health panel.
struct BrokenSink {
    name: String,
    reason: String,
}

impl BrokenSink {
    fn new(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            reason: reason.into(),
        }
    }
}

#[async_trait]
impl ObservabilitySink for BrokenSink {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> SinkCapabilities {
        SinkCapabilities {
            idempotency: IdempotencyScheme::None,
            ordering: OrderingScope::None,
            batch_unit: BatchUnit::Records,
            max_batch_bytes: None,
            supports_partial_batch: false,
            supports_streaming_ingest: false,
        }
    }

    async fn append_batch(&self, _batch: &EventBatch, _marker: &IdempotencyMarker) -> SinkResult {
        Err(SinkError::Permanent(self.reason.clone()))
    }

    async fn healthcheck(&self) -> SinkHealth {
        SinkHealth::unhealthy(self.reason.clone())
    }
}

/// Resolve an exporter's `credential_ref` to provider credentials from the
/// DP's local environment. The plaintext key never travels on the kine path
/// (the control plane stores only the reference), so the DP looks it up where
/// it runs — mirroring `resolve_sls_credential`. Env keys are
/// `OBJSTORE_CRED_<SLUG>_<FIELD>`, where `<SLUG>` upper-cases the ref with
/// non-alphanumerics folded to `_` (the `OBJSTORE_` prefix is deliberately
/// not `AISIX_`, which the config loader owns). Returns `None` when a required
/// field is unset/blank — the caller surfaces a delivery-health auth error
/// rather than building a half-credentialed client.
pub fn resolve_object_store_credential(
    provider: ObjectStoreProvider,
    credential_ref: &str,
) -> Option<ObjectStoreCredentials> {
    resolve_object_store_credential_with(provider, credential_ref, |k| std::env::var(k).ok())
}

/// Resolution core, parameterized over the variable source so it is testable
/// without mutating the process environment.
fn resolve_object_store_credential_with(
    provider: ObjectStoreProvider,
    credential_ref: &str,
    lookup: impl Fn(&str) -> Option<String>,
) -> Option<ObjectStoreCredentials> {
    let slug: String = credential_ref
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect();
    let get = |field: &str| -> Option<String> {
        lookup(&format!("OBJSTORE_CRED_{slug}_{field}")).filter(|v| !v.is_empty())
    };
    match provider {
        ObjectStoreProvider::S3 => Some(ObjectStoreCredentials::S3 {
            access_key_id: get("AWS_ACCESS_KEY_ID")?,
            secret_access_key: get("AWS_SECRET_ACCESS_KEY")?,
            session_token: get("AWS_SESSION_TOKEN"),
        }),
        ObjectStoreProvider::Gcs => Some(ObjectStoreCredentials::Gcs {
            service_account_key: get("GCS_SERVICE_ACCOUNT_KEY")?,
        }),
        ObjectStoreProvider::AzureBlob => Some(ObjectStoreCredentials::Azure {
            account: get("AZURE_ACCOUNT")?,
            access_key: get("AZURE_ACCESS_KEY")?,
        }),
    }
}

/// gzip a byte slice (RFC 1952). Permanent on the rare encode failure.
fn gzip(data: &[u8]) -> Result<Vec<u8>, SinkError> {
    use std::io::Write as _;
    let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    enc.write_all(data)
        .and_then(|_| enc.finish())
        .map_err(|e| SinkError::Permanent(format!("object_store: gzip: {e}")))
}

/// First 16 bytes of SHA-256, hex-encoded (32 chars) — the content address.
fn sha256_hex16(body: &[u8]) -> String {
    let digest = Sha256::digest(body);
    hex::encode(&digest[..16])
}

/// `(date, hour)` Hive partition from an RFC-3339 timestamp (the record's
/// `occurred_at`), normalised to UTC. Falls back to `unknown` when the
/// timestamp is absent/unparseable, so the key stays deterministic.
fn partition(occurred_at: &str) -> (String, String) {
    match chrono::DateTime::parse_from_rfc3339(occurred_at) {
        Ok(dt) => {
            let utc = dt.with_timezone(&chrono::Utc);
            (
                utc.format("%Y-%m-%d").to_string(),
                utc.format("%H").to_string(),
            )
        }
        Err(_) => ("unknown".to_string(), "00".to_string()),
    }
}

/// Map an `object_store` error to the pipeline's retry/permanent decision.
/// Auth, missing-bucket and unsupported-operation errors are permanent (a
/// retry fails the same way); transport / throttle / 5xx surface as
/// `Error::Generic` and are transient. The detail is length-capped; the crate
/// does not echo secrets in error text.
fn map_object_store_err(e: object_store::Error) -> SinkError {
    use object_store::Error as E;
    let detail = truncate(&e.to_string());
    match e {
        E::PermissionDenied { .. }
        | E::Unauthenticated { .. }
        | E::NotFound { .. }
        | E::NotSupported { .. }
        | E::NotImplemented { .. }
        | E::InvalidPath { .. }
        | E::UnknownConfigurationKey { .. } => SinkError::Permanent(detail),
        _ => SinkError::Transient(detail),
    }
}

/// Truncate a masked detail string to a bounded length for logs / health.
fn truncate(s: &str) -> String {
    s.chars().take(DETAIL_MAX_CHARS).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::{EventBatch, IdempotencyMarker, ObservabilitySink, SinkContent, SinkRecord};
    use crate::usage::UsageEvent;
    use futures::StreamExt;
    use object_store::memory::InMemory;
    use std::sync::Arc;

    fn event(request_id: &str) -> UsageEvent {
        UsageEvent {
            request_id: request_id.into(),
            occurred_at: "2026-05-01T12:34:56Z".into(),
            model_id: "gpt-4o".into(),
            status_code: 200,
            prompt_tokens: 5,
            completion_tokens: 7,
            ..UsageEvent::default()
        }
    }

    fn batch_of(records: Vec<SinkRecord>) -> EventBatch {
        EventBatch::new(records.into_iter().map(Arc::new).collect())
    }

    async fn list_keys(store: &Arc<dyn ObjectStore>) -> Vec<String> {
        store
            .list(None)
            .map(|m| m.unwrap().location.to_string())
            .collect::<Vec<_>>()
            .await
    }

    fn sink(store: Arc<dyn ObjectStore>, compression: ObjectStoreCompression) -> ObjectStoreSink {
        ObjectStoreSink::new("obj-test", store, "ai-gateway", compression)
    }

    #[tokio::test]
    async fn writes_gzipped_ndjson_under_partitioned_content_addressed_key() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let s = sink(store.clone(), ObjectStoreCompression::Gzip);
        let ack = s
            .append_batch(
                &batch_of(vec![
                    SinkRecord::metadata_only(event("req-1")),
                    SinkRecord::metadata_only(event("req-2")),
                ]),
                &IdempotencyMarker::None,
            )
            .await
            .expect("delivery succeeds");
        assert_eq!(ack.accepted, 2);

        let keys = list_keys(&store).await;
        assert_eq!(keys.len(), 1, "one object per flush");
        let key = &keys[0];
        // Hive-partitioned, content-addressed, gzip extension.
        assert!(
            key.starts_with("ai-gateway/dt=2026-05-01/hh=12/"),
            "partitioned key: {key}"
        );
        assert!(key.ends_with(".ndjson.gz"), "gzip extension: {key}");

        // Body round-trips: gunzip → two NDJSON lines, both real events.
        let raw = store
            .get(&ObjectPath::parse(key).unwrap())
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let mut gz = flate2::read::GzDecoder::new(&raw[..]);
        let mut text = String::new();
        std::io::Read::read_to_string(&mut gz, &mut text).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2);
        let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(first["request_id"], "req-1");
        assert_eq!(first["schema_version"], "1.0");
        // Metadata-only path carries no prompt.
        assert!(first.get("content").is_none());
    }

    #[tokio::test]
    async fn retry_of_same_batch_reuses_the_same_key() {
        // The pipeline retries the SAME batch on a transient error. The
        // content-addressed key must be identical so a downstream loader
        // dedups by filename (AISIX-Cloud#689 §8) — assert exactly one object.
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let s = sink(store.clone(), ObjectStoreCompression::Gzip);
        let b = batch_of(vec![SinkRecord::metadata_only(event("req-dup"))]);
        s.append_batch(&b, &IdempotencyMarker::None).await.unwrap();
        s.append_batch(&b, &IdempotencyMarker::None).await.unwrap();
        assert_eq!(
            list_keys(&store).await.len(),
            1,
            "a re-delivered batch overwrites the same content-addressed key"
        );
    }

    #[tokio::test]
    async fn different_content_lands_under_different_keys() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let s = sink(store.clone(), ObjectStoreCompression::Gzip);
        s.append_batch(
            &batch_of(vec![SinkRecord::metadata_only(event("a"))]),
            &IdempotencyMarker::None,
        )
        .await
        .unwrap();
        s.append_batch(
            &batch_of(vec![SinkRecord::metadata_only(event("b"))]),
            &IdempotencyMarker::None,
        )
        .await
        .unwrap();
        assert_eq!(list_keys(&store).await.len(), 2);
    }

    #[tokio::test]
    async fn compression_none_writes_plain_ndjson() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let s = sink(store.clone(), ObjectStoreCompression::None);
        s.append_batch(
            &batch_of(vec![SinkRecord::metadata_only(event("req-1"))]),
            &IdempotencyMarker::None,
        )
        .await
        .unwrap();
        let keys = list_keys(&store).await;
        assert!(
            keys[0].ends_with(".ndjson"),
            "no gzip extension: {}",
            keys[0]
        );
        let raw = store
            .get(&ObjectPath::parse(&keys[0]).unwrap())
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let text = String::from_utf8(raw.to_vec()).unwrap();
        let v: serde_json::Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
        assert_eq!(v["request_id"], "req-1");
    }

    #[tokio::test]
    async fn opt_in_content_is_written_when_present() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let s = sink(store.clone(), ObjectStoreCompression::None);
        let rec = SinkRecord::metadata_only(event("req-c")).with_content(SinkContent {
            prompt: "what is 2+2?".into(),
            response: "4".into(),
            truncated: false,
        });
        s.append_batch(&batch_of(vec![rec]), &IdempotencyMarker::None)
            .await
            .unwrap();
        let keys = list_keys(&store).await;
        let raw = store
            .get(&ObjectPath::parse(&keys[0]).unwrap())
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let text = String::from_utf8(raw.to_vec()).unwrap();
        let v: serde_json::Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
        assert_eq!(v["content"]["prompt"], "what is 2+2?");
        assert_eq!(v["content"]["response"], "4");
    }

    #[tokio::test]
    async fn empty_batch_is_a_noop() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let s = sink(store.clone(), ObjectStoreCompression::Gzip);
        let ack = s
            .append_batch(&EventBatch::default(), &IdempotencyMarker::None)
            .await
            .unwrap();
        assert_eq!(ack.accepted, 0);
        assert_eq!(list_keys(&store).await.len(), 0);
    }

    #[test]
    fn build_object_store_dispatches_s3() {
        // S3 builds offline (no connection until put); proves the S3 arm wires.
        let store = build_object_store(
            ObjectStoreProvider::S3,
            "bucket",
            Some("us-east-1"),
            Some("http://minio:9000"),
            ObjectStoreCredentials::S3 {
                access_key_id: "id".into(),
                secret_access_key: "secret".into(),
                session_token: None,
            },
        );
        assert!(store.is_ok(), "s3 build: {store:?}");
    }

    #[test]
    fn build_object_store_rejects_provider_credential_mismatch() {
        let r = build_object_store(
            ObjectStoreProvider::S3,
            "b",
            None,
            None,
            ObjectStoreCredentials::Azure {
                account: "a".into(),
                access_key: "k".into(),
            },
        );
        assert!(matches!(r, Err(SinkError::Permanent(_))));
    }

    #[test]
    fn build_object_store_ambient_dispatches_s3() {
        // Cloud-identity S3 builds offline via the AWS default credential chain
        // (from_env → IMDS / IRSA / ECS); creds are fetched lazily at request
        // time, so build() succeeds with no static keys.
        let store =
            build_object_store_ambient(ObjectStoreProvider::S3, "bucket", Some("us-east-1"), None);
        assert!(store.is_ok(), "s3 ambient build: {store:?}");
    }

    #[test]
    fn build_object_store_ambient_s3_rejects_endpoint() {
        // A custom endpoint with cloud_identity S3 is a misconfiguration:
        // ambient AWS credentials need the provider's metadata service, which an
        // S3-compatible host (MinIO / R2 / OSS) does not expose. Fail fast with
        // a clear permanent error pointing back to credential_ref.
        let r = build_object_store_ambient(
            ObjectStoreProvider::S3,
            "bucket",
            Some("us-east-1"),
            Some("https://minio.internal:9000"),
        );
        match r {
            Err(SinkError::Permanent(msg)) => {
                assert!(msg.contains("endpoint"), "msg: {msg}");
                assert!(msg.contains("credential_ref"), "msg: {msg}");
            }
            other => {
                panic!("expected Permanent error for s3 cloud_identity + endpoint, got {other:?}")
            }
        }
    }

    #[test]
    fn build_object_store_ambient_dispatches_gcs() {
        // Cloud-identity GCS builds with no service-account key → Application
        // Default Credentials at request time; build() succeeds offline.
        let store = build_object_store_ambient(ObjectStoreProvider::Gcs, "bucket", None, None);
        assert!(store.is_ok(), "gcs ambient build: {store:?}");
    }

    #[test]
    fn build_object_store_ambient_rejects_azure() {
        // Azure cloud_identity is not supported (managed identity needs a
        // non-secret account name the keyless config does not carry) — a clear
        // permanent error pointing the operator back to credential_ref.
        let r = build_object_store_ambient(ObjectStoreProvider::AzureBlob, "container", None, None);
        match r {
            Err(SinkError::Permanent(msg)) => {
                assert!(msg.contains("azure_blob"), "msg: {msg}");
                assert!(msg.contains("credential_ref"), "msg: {msg}");
            }
            other => panic!("expected Permanent error, got {other:?}"),
        }
    }

    #[test]
    fn resolve_credential_maps_ref_to_env_per_provider() {
        let store = |key: &str| match key {
            "OBJSTORE_CRED_ACME_S3_AWS_ACCESS_KEY_ID" => Some("akid".to_string()),
            "OBJSTORE_CRED_ACME_S3_AWS_SECRET_ACCESS_KEY" => Some("secret".to_string()),
            _ => None,
        };
        let c = resolve_object_store_credential_with(ObjectStoreProvider::S3, "acme-s3", store);
        assert_eq!(
            c,
            Some(ObjectStoreCredentials::S3 {
                access_key_id: "akid".into(),
                secret_access_key: "secret".into(),
                session_token: None,
            })
        );
    }

    #[test]
    fn resolve_credential_is_none_when_required_field_missing() {
        // S3 secret missing → None (never build a half-credentialed client).
        let id_only =
            |key: &str| (key == "OBJSTORE_CRED_X_AWS_ACCESS_KEY_ID").then(|| "id".to_string());
        assert_eq!(
            resolve_object_store_credential_with(ObjectStoreProvider::S3, "x", id_only),
            None
        );
        // Blank is treated as unset.
        assert_eq!(
            resolve_object_store_credential_with(ObjectStoreProvider::Gcs, "x", |_| Some(
                String::new()
            )),
            None
        );
    }

    #[test]
    fn maps_object_store_errors_to_retry_decision() {
        let perm = map_object_store_err(object_store::Error::PermissionDenied {
            path: "p".into(),
            source: "denied".into(),
        });
        assert!(!perm.is_transient(), "auth error is permanent");

        let transient = map_object_store_err(object_store::Error::Generic {
            store: "s3",
            source: "connection reset".into(),
        });
        assert!(transient.is_transient(), "transport error is transient");
    }

    #[test]
    fn partition_parses_event_time_and_falls_back() {
        assert_eq!(
            partition("2026-05-01T12:34:56Z"),
            ("2026-05-01".to_string(), "12".to_string())
        );
        assert_eq!(partition(""), ("unknown".to_string(), "00".to_string()));
    }
}

/// Real-emulator smoke tests — the one-off validation that the sink's real
/// SigV4 / GCS / Azure signing + HTTP `put` actually round-trips against a live
/// S3 / GCS / Azure-compatible server (the wire a mock / `InMemory` cannot
/// check). `#[ignore]` + env-gated, so the normal suite never touches the
/// network; bring up the emulators with
/// `tests/object-store-emulators.compose.yml` and run with:
/// `cargo test -p aisix-obs -- --ignored objstore_smoke`.
#[cfg(test)]
mod smoke {
    use super::{build_object_store, ObjectStoreCredentials, ObjectStoreSink};
    use crate::sink::{EventBatch, IdempotencyMarker, ObservabilitySink, SinkRecord};
    use crate::usage::UsageEvent;
    use aisix_core::models::observability_exporter::{ObjectStoreCompression, ObjectStoreProvider};
    use futures::StreamExt;
    use object_store::path::Path as ObjectPath;
    use object_store::{ObjectStore, ObjectStoreExt};
    use std::sync::Arc;

    fn env(key: &str) -> Option<String> {
        std::env::var(key).ok().filter(|v| !v.is_empty())
    }

    fn smoke_event() -> UsageEvent {
        UsageEvent {
            request_id: format!("objstore-smoke-{}", uuid::Uuid::new_v4()),
            occurred_at: "2026-05-01T12:00:00Z".into(),
            model_id: "smoke-model".into(),
            status_code: 200,
            prompt_tokens: 1,
            completion_tokens: 2,
            ..UsageEvent::default()
        }
    }

    async fn list_under(store: &Arc<dyn ObjectStore>, prefix: &str) -> Vec<ObjectPath> {
        let p = ObjectPath::parse(prefix).expect("prefix parses");
        store
            .list(Some(&p))
            .map(|m| m.expect("list entry").location)
            .collect::<Vec<_>>()
            .await
    }

    /// Drive a real put through the sink, read the object back via a fresh
    /// client, confirm the gzipped NDJSON round-trips, prove a re-delivery
    /// reuses the same content-addressed key (one object, not two), then clean
    /// up. Works against any provider — the only per-provider part is `store`.
    async fn smoke_roundtrip(store: Arc<dyn ObjectStore>) {
        let prefix = format!("aisix-smoke/{}", uuid::Uuid::new_v4());
        let sink = ObjectStoreSink::new(
            "objstore-smoke",
            store.clone(),
            prefix.clone(),
            ObjectStoreCompression::Gzip,
        );
        let event = smoke_event();
        let want_request_id = event.request_id.clone();
        let batch = EventBatch::new(vec![Arc::new(SinkRecord::metadata_only(event))]);

        sink.append_batch(&batch, &IdempotencyMarker::None)
            .await
            .expect("real put accepted by the emulator");

        let keys = list_under(&store, &prefix).await;
        assert_eq!(keys.len(), 1, "exactly one object written");
        assert!(
            keys[0].as_ref().ends_with(".ndjson.gz"),
            "gzip extension: {}",
            keys[0]
        );

        // Read it back through a real GET and gunzip → the event round-trips.
        let raw = store
            .get(&keys[0])
            .await
            .expect("get object")
            .bytes()
            .await
            .expect("object bytes");
        let mut gz = flate2::read::GzDecoder::new(&raw[..]);
        let mut text = String::new();
        std::io::Read::read_to_string(&mut gz, &mut text).expect("gunzip");
        let line = text.lines().next().expect("one ndjson line");
        let v: serde_json::Value = serde_json::from_str(line).expect("valid ndjson");
        assert_eq!(v["request_id"], want_request_id);

        // Re-deliver the SAME batch (the pipeline's retry shape): the content
        // address means it overwrites the same key — still exactly one object.
        sink.append_batch(&batch, &IdempotencyMarker::None)
            .await
            .expect("re-delivery accepted");
        let keys2 = list_under(&store, &prefix).await;
        assert_eq!(keys2.len(), 1, "re-delivery reuses the same key (no dupe)");

        for k in keys2 {
            let _ = store.delete(&k).await;
        }
    }

    #[tokio::test]
    #[ignore = "hits a real S3-compatible server (MinIO/LocalStack). \
                Run: docker compose -f crates/aisix-obs/tests/object-store-emulators.compose.yml up -d \
                then AISIX_E2E_OBJSTORE_S3_* set; cargo test -p aisix-obs -- --ignored objstore_smoke_s3"]
    async fn objstore_smoke_s3_roundtrip() {
        let (Some(endpoint), Some(bucket), Some(key_id), Some(secret)) = (
            env("AISIX_E2E_OBJSTORE_S3_ENDPOINT"),
            env("AISIX_E2E_OBJSTORE_S3_BUCKET"),
            env("AISIX_E2E_OBJSTORE_S3_ACCESS_KEY_ID"),
            env("AISIX_E2E_OBJSTORE_S3_SECRET_ACCESS_KEY"),
        ) else {
            eprintln!("objstore_smoke_s3: AISIX_E2E_OBJSTORE_S3_* not set — skipping");
            return;
        };
        let store = build_object_store(
            ObjectStoreProvider::S3,
            &bucket,
            Some("us-east-1"),
            Some(&endpoint),
            ObjectStoreCredentials::S3 {
                access_key_id: key_id,
                secret_access_key: secret,
                session_token: None,
            },
        )
        .expect("build S3 store");
        smoke_roundtrip(store).await;
    }

    #[tokio::test]
    #[ignore = "hits a real Azure Blob emulator (Azurite). \
                Run with AISIX_E2E_OBJSTORE_AZURE_* set; \
                cargo test -p aisix-obs -- --ignored objstore_smoke_azure"]
    async fn objstore_smoke_azure_roundtrip() {
        let (Some(endpoint), Some(container), Some(account), Some(access_key)) = (
            env("AISIX_E2E_OBJSTORE_AZURE_ENDPOINT"),
            env("AISIX_E2E_OBJSTORE_AZURE_CONTAINER"),
            env("AISIX_E2E_OBJSTORE_AZURE_ACCOUNT"),
            env("AISIX_E2E_OBJSTORE_AZURE_ACCESS_KEY"),
        ) else {
            eprintln!("objstore_smoke_azure: AISIX_E2E_OBJSTORE_AZURE_* not set — skipping");
            return;
        };
        let store = build_object_store(
            ObjectStoreProvider::AzureBlob,
            &container,
            None,
            Some(&endpoint),
            ObjectStoreCredentials::Azure {
                account,
                access_key,
            },
        )
        .expect("build Azure store");
        smoke_roundtrip(store).await;
    }

    #[tokio::test]
    #[ignore = "GCS round-trip — set AISIX_E2E_OBJSTORE_GCS_* against REAL GCS or a \
                conformant emulator. NOTE: fake-gcs-server's XML API does not round-trip \
                object_store's percent-encoded object names (the `/` in partition keys → \
                %2F), so build + auth verify there but the PUT only greens on real GCS. \
                cargo test -p aisix-obs -- --ignored objstore_smoke_gcs"]
    async fn objstore_smoke_gcs_roundtrip() {
        let (Some(endpoint), Some(bucket), Some(service_account_key)) = (
            env("AISIX_E2E_OBJSTORE_GCS_ENDPOINT"),
            env("AISIX_E2E_OBJSTORE_GCS_BUCKET"),
            env("AISIX_E2E_OBJSTORE_GCS_SERVICE_ACCOUNT"),
        ) else {
            eprintln!("objstore_smoke_gcs: AISIX_E2E_OBJSTORE_GCS_* not set — skipping");
            return;
        };
        let store = build_object_store(
            ObjectStoreProvider::Gcs,
            &bucket,
            None,
            Some(&endpoint),
            ObjectStoreCredentials::Gcs {
                service_account_key,
            },
        )
        .expect("build GCS store");
        smoke_roundtrip(store).await;
    }
}
