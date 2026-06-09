---
title: Observability Exporters
description: Configure observability exporters — OTLP/HTTP traces and cloud object-storage (NDJSON) — for AISIX AI Gateway data-plane telemetry fan-out.
sidebar_position: 40
---

Observability exporters let the data plane send request telemetry directly to a destination you control — an OTLP/HTTP endpoint, or a cloud object-storage bucket.

This page documents all current exporter kinds:

- `otlp_http` — request traces to an OTLP/HTTP endpoint.
- `object_store` — batched NDJSON files to Amazon S3, Google Cloud Storage, or Azure Blob (or any S3-compatible target).
- `aliyun_sls` — request-event logs to an Alibaba Cloud Simple Log Service (SLS) logstore.
- `datadog` — request events to the Datadog Logs HTTP intake.

Use this page when you want request-level telemetry fan-out without restarting the process for every destination change.

## OTLP/HTTP Exporter (`kind: otlp_http`)

Fields:

- `name`
- `enabled`
- `kind`
- `endpoint`
- optional `headers`

The basic operator questions for this resource are:

- where should telemetry be sent
- what auth headers are required for that destination
- should the exporter currently participate in fan-out

Example:

```bash title="Create an OTLP exporter"
curl -sS -X POST http://127.0.0.1:3001/admin/v1/observability_exporters \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "honeycomb-prod",
    "kind": "otlp_http",
    "endpoint": "https://api.honeycomb.io/v1/traces",
    "headers": {
      "x-honeycomb-team": "YOUR_TEAM_KEY"
    }
  }'
```

## Object Storage Exporter (`kind: object_store`)

`kind: "object_store"` writes batched, newline-delimited JSON (NDJSON) telemetry to a cloud object-storage bucket. One sink covers Amazon S3, Google Cloud Storage, and Azure Blob — selected by `provider` — plus any S3-compatible target (MinIO, Cloudflare R2) via an `endpoint` override. Files are written under the configured `prefix` with date/hour partitioning, gzip-compressed by default.

Fields:

- `provider` — `s3`, `gcs`, or `azure_blob`.
- `bucket` — bucket name (the container name for `azure_blob`).
- `prefix` — key prefix the partitioned object path is appended to, e.g. `ai-gateway`.
- `region` — optional; AWS region for S3 SigV4 scope. Defaults to `us-east-1` when unset, so set it for any non-default-region bucket.
- `endpoint` — optional; S3-compatible host override (MinIO / Cloudflare R2 / OSS). Leave unset for a provider's native endpoint.
- `compression` — `gzip` (default) or `none`.
- `auth_mode` — `credential_ref` (default) or `cloud_identity`. See **Object storage authentication** below.
- `credential_ref` — required when `auth_mode = credential_ref`; omit it for `cloud_identity`.

As with every exporter, cloud credentials are **never** stored in the control plane or sent on the wire. The config carries only a `credential_ref` pointer (or nothing, for `cloud_identity`); the data plane resolves the actual credential locally.

### Object storage authentication

#### `credential_ref` (default) — static keys on the data plane

The data plane resolves `credential_ref` to environment variables it reads locally, named `OBJSTORE_CRED_<SLUG>_<FIELD>`. `<SLUG>` is `credential_ref` upper-cased with every character that is not an ASCII letter or digit folded to `_`. To keep that mapping unambiguous — so two different refs cannot silently fold onto the same variables — use only lowercase letters, digits, and underscores in `credential_ref`; the control plane and dashboard enforce `^[a-z0-9_]+$`.

Per-provider variables to set on the data plane (shown for `credential_ref = acme_s3_prod`, where `<SLUG>` = `ACME_S3_PROD`):

| Provider | Required | Optional |
|----------|----------|----------|
| `s3` | `OBJSTORE_CRED_<SLUG>_AWS_ACCESS_KEY_ID`, `OBJSTORE_CRED_<SLUG>_AWS_SECRET_ACCESS_KEY` | `OBJSTORE_CRED_<SLUG>_AWS_SESSION_TOKEN` |
| `gcs` | `OBJSTORE_CRED_<SLUG>_GCS_SERVICE_ACCOUNT_KEY` (full service-account JSON) | — |
| `azure_blob` | `OBJSTORE_CRED_<SLUG>_AZURE_ACCOUNT`, `OBJSTORE_CRED_<SLUG>_AZURE_ACCESS_KEY` | — |

A required variable that is missing or empty makes the sink fail every delivery and report unhealthy even though the exporter config itself is valid — set these before, or right after, creating the exporter.

```bash title="Create an S3 object_store exporter (static keys)"
curl -sS -X POST http://127.0.0.1:3001/admin/v1/observability_exporters \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "acme-events-s3",
    "kind": "object_store",
    "provider": "s3",
    "bucket": "acme-aisix-events",
    "prefix": "ai-gateway",
    "region": "us-east-1",
    "credential_ref": "acme_s3_prod"
  }'
```

```bash title="...and the matching variables on the data plane"
OBJSTORE_CRED_ACME_S3_PROD_AWS_ACCESS_KEY_ID=<your key id>
OBJSTORE_CRED_ACME_S3_PROD_AWS_SECRET_ACCESS_KEY=<your secret>
# optional, for temporary credentials:
OBJSTORE_CRED_ACME_S3_PROD_AWS_SESSION_TOKEN=<token>
```

#### `cloud_identity` — keyless, no static keys anywhere

When the data plane runs inside the same cloud as the bucket, set `auth_mode: "cloud_identity"` and provide no `credential_ref`. The data plane authenticates with the host's own attached cloud identity through the cloud SDK's default credential chain:

- **S3** — EC2 instance role, EKS IRSA, or ECS task role.
- **GCS** — GKE Workload Identity or the GCE metadata service (Application Default Credentials).

No static keys exist anywhere: none in the control plane, none in the data-plane environment. Grant the data plane's identity write access to the bucket:

- **S3** — `s3:PutObject` on `<bucket>/<prefix>/*`.
- **GCS** — `storage.objects.create` (role `roles/storage.objectCreator`) on the bucket.

`cloud_identity` is supported for `s3` and `gcs` only — `azure_blob` with `cloud_identity` is rejected. Do not set a custom `endpoint` with `cloud_identity`: ambient credentials authenticate against the provider's native service, and S3-compatible targets (MinIO, Cloudflare R2) have no cloud IAM identity, so they must use `credential_ref`. The control plane rejects the `cloud_identity` + `endpoint` combination.

```bash title="Create a keyless S3 object_store exporter (cloud identity)"
curl -sS -X POST http://127.0.0.1:3001/admin/v1/observability_exporters \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "acme-events-s3-keyless",
    "kind": "object_store",
    "provider": "s3",
    "bucket": "acme-aisix-events",
    "prefix": "ai-gateway",
    "region": "us-east-1",
    "auth_mode": "cloud_identity"
  }'
```

## Aliyun SLS Exporter (`kind: aliyun_sls`)

`kind: "aliyun_sls"` sends request-event logs to an Alibaba Cloud Simple Log Service (SLS) logstore. The data plane signs each batch with an AccessKey it resolves locally and posts to `https://<project>.<endpoint>`.

Fields:

- `endpoint` — SLS region host with no scheme, e.g. `ap-southeast-3.log.aliyuncs.com`. The data plane posts to `https://<project>.<endpoint>`.
- `project` — SLS project name.
- `logstore` — SLS logstore that receives the logs.
- `credential_ref` — pointer to the AccessKey, resolved on the data plane (see below). The AccessKey is never stored in the control plane or sent on the wire.
- `content_mode` — `metadata_only` (default) or `full`. See **Content capture** below.
- `content_max_bytes` — per-field byte cap under `content_mode: full`; defaults to `131072` (128 KiB), minimum `1`, maximum `1048576` (1 MiB). Ignored under `metadata_only`.

`endpoint`, `project`, `logstore`, and `credential_ref` are required.

The data plane resolves `credential_ref` to two environment variables it reads locally — `SLS_CRED_<SLUG>_AK_ID` and `SLS_CRED_<SLUG>_AK_SECRET` — where `<SLUG>` is the reference upper-cased with every character that is not an ASCII letter or digit folded to `_` (so `acme_sls` → `ACME_SLS`).

```bash title="Create an aliyun_sls exporter"
curl -sS -X POST http://127.0.0.1:3001/admin/v1/observability_exporters \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "acme-sls",
    "kind": "aliyun_sls",
    "endpoint": "ap-southeast-3.log.aliyuncs.com",
    "project": "acme-observability",
    "logstore": "ai-gateway",
    "credential_ref": "acme_sls"
  }'
```

```bash title="...and the matching variables on the data plane"
SLS_CRED_ACME_SLS_AK_ID=<your accesskey id>
SLS_CRED_ACME_SLS_AK_SECRET=<your accesskey secret>
```

## Datadog Exporter (`kind: datadog`)

`kind: "datadog"` sends each request event as one log to the Datadog Logs HTTP intake. The data plane gzip-compresses each batch and posts it to `https://http-intake.logs.<site>/api/v2/logs`.

Fields:

- `site` — Datadog site: one of `datadoghq.com`, `us3.datadoghq.com`, `us5.datadoghq.com`, `datadoghq.eu`, `ap1.datadoghq.com`, `ap2.datadoghq.com`, `ddog-gov.com`. The intake host is `http-intake.logs.<site>`.
- `credential_ref` — pointer to the Datadog API key, resolved on the data plane (see below). The API key is never stored in the control plane or sent on the wire.
- `service` — the Datadog `service` reserved attribute every log from this exporter carries.
- `ddsource` — the Datadog `ddsource` reserved attribute. Defaults to `aisix-ai-gateway`.
- `tags` — a list of tags rendered into Datadog's comma-joined `ddtags` attribute, e.g. `["team:platform", "tier:prod"]` becomes `team:platform,tier:prod`. Empty by default.
- `content_mode` — `metadata_only` (default) or `full`. See **Content capture** below.
- `content_max_bytes` — per-field byte cap under `content_mode: full`; defaults to `131072` (128 KiB), minimum `1`, maximum `1048576` (1 MiB). Ignored under `metadata_only`.

`site`, `credential_ref`, and `service` are required.

The data plane resolves `credential_ref` to `DD_CRED_<SLUG>_API_KEY`, read from its local environment, where `<SLUG>` is the reference upper-cased with every character that is not an ASCII letter or digit folded to `_` (so `acme_dd` → `ACME_DD`).

```bash title="Create a datadog exporter"
curl -sS -X POST http://127.0.0.1:3001/admin/v1/observability_exporters \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "acme-datadog",
    "kind": "datadog",
    "site": "datadoghq.com",
    "service": "ai-gateway",
    "tags": ["team:platform", "tier:prod"],
    "credential_ref": "acme_dd"
  }'
```

```bash title="...and the matching variable on the data plane"
DD_CRED_ACME_DD_API_KEY=<your datadog api key>
```

:::caution Large `full`-mode batches can be rejected by Datadog
Under `content_mode: full`, each log carries both the prompt and the response — each capped at `content_max_bytes` — so a single encoded log can approach twice the cap. Datadog rejects any single log over 1 MB and any request over 5 MB or 1000 logs, and the data plane does not yet split batches to those limits, so a high cap on a busy exporter can cause Datadog to reject an oversized batch (surfaced as a delivery error in the exporter's health). The 128 KiB default keeps a log well under the per-log limit.
:::

## Content capture

`aliyun_sls` and `datadog` exporters control whether end-user content is delivered through `content_mode`:

- `metadata_only` (the default) ships only operational metadata — never the request prompt or the response.
- `full` additionally captures the request prompt and the assembled response, each truncated to `content_max_bytes` on a UTF-8 boundary. A log whose prompt or response was cut carries a `content_truncated` marker.

:::caution `full` writes end-user content to a third party
Enabling `content_mode: full` sends end-user prompt and response text to the destination (your SLS logstore or Datadog org). Confirm this is compatible with your data-handling and privacy obligations before turning it on.
:::

## Endpoint Restriction

The admin validation layer currently rejects plain `http://` endpoints unless they point to an allowed loopback-style target.

Allowed non-TLS development cases include:

- `http://127.0.0.1/...`
- `http://localhost/...`
- `http://mock-otlp/...`
- `http://otel-collector/...`

For non-loopback deployments, use `https://...`.

This protects against accidentally configuring plain HTTP exporters for non-local destinations.

## Runtime Model

Current exporter behavior:

- exporters are environment-scoped dynamic resources
- the data plane, not the control plane, sends the HTTP export traffic
- disabled exporters remain in the snapshot but are skipped

This means the request content and telemetry egress path stay with the data plane.

This keeps sensitive prompt and response content on the data plane egress path.

## Operator Guidance

- start with one exporter and verify delivery before adding several
- keep credentials in `headers` aligned with the destination's OTLP/HTTP auth model
- for `object_store`, prefer `auth_mode: cloud_identity` when the data plane runs in the bucket's cloud — there are no keys to provision or rotate; use `credential_ref` for S3-compatible targets (MinIO / R2) or cross-cloud setups
- disable exporters rather than deleting them immediately when you are diagnosing delivery issues

## Troubleshooting

### The exporter saves but no telemetry appears downstream

Check endpoint correctness, destination auth headers, and whether the exporter is enabled.

### An `object_store` exporter saves but no objects appear

With `credential_ref`, confirm the `OBJSTORE_CRED_<SLUG>_*` variables are set and non-empty on the data plane — a missing or empty key makes every delivery fail while the exporter config still validates. With `cloud_identity`, confirm the data plane's attached identity has bucket write access (`s3:PutObject` / `storage.objects.create`).

### The admin API rejects an `http://` endpoint

That is expected unless the destination is one of the allowed local-development forms.

## Related Pages

- [Admin API](admin-api.md)
- [Metrics And Logs](../operations/metrics-and-logs.md)
- [Reference: Resource Schemas](../reference/resource-schemas.md)
