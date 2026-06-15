---
title: Observability exporters
description: Configure AISIX AI Gateway observability exporters — OTLP/HTTP, cloud object storage, Aliyun SLS, and Datadog — to fan out data-plane request telemetry.
sidebar_position: 40
---

Observability exporters fan out the AISIX AI Gateway data plane's request telemetry to a destination you control, without restarting the gateway for every change. Each exporter is an environment-scoped dynamic resource discriminated by `kind`.

Supported kinds:

- `otlp_http` — request traces to an OTLP/HTTP endpoint.
- `object_store` — batched NDJSON files to Amazon S3, Google Cloud Storage, or Azure Blob (or any S3-compatible target).
- `aliyun_sls` — request-event logs to an Alibaba Cloud Simple Log Service (SLS) logstore.
- `datadog` — request events to the Datadog Logs HTTP intake.

## How exporters work

- The data plane (DP), not the control plane (CP), performs the export. Request and response content stays on the DP egress path and is never sent to the control plane.
- Exporters are environment-scoped. The proxy applies admin writes dynamically — no restart.
- `enabled: false` keeps the row but skips it during fan-out, so you can pause an exporter without losing its configuration.
- Cloud credentials are never stored in the control plane or placed on the wire. An exporter carries an opaque `credential_ref` that the DP resolves locally — or, for `object_store` in keyless mode, no reference at all.

Every kind shares three top-level fields:

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | string | Yes | — | Operator-facing label, shown in `/logs` and the dashboard. Not used for routing. |
| `enabled` | boolean | No | `true` | Soft kill switch. A disabled exporter stays in the snapshot but is skipped during fan-out. |
| `kind` | string | Yes | — | One of `otlp_http`, `object_store`, `aliyun_sls`, `datadog`. |

The remaining fields depend on `kind`, as documented in each section below.

## OTLP/HTTP exporter (`kind: otlp_http`)

Sends request traces to an OTLP/HTTP endpoint.

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `endpoint` | string | Yes | — | OTLP/HTTP traces URL. Must be `https://`; plain `http://` is rejected except for loopback development hosts (see [Endpoint validation](#endpoint-validation)). |
| `headers` | object | No | — | Static headers attached to every export request, such as the destination's auth header. |

```bash title="Create an OTLP/HTTP exporter"
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

## Object storage exporter (`kind: object_store`)

Writes batched, newline-delimited JSON (NDJSON) telemetry to a cloud object-storage bucket. One sink covers Amazon S3, Google Cloud Storage, and Azure Blob — selected by `provider` — plus any S3-compatible target (MinIO, Cloudflare R2) via an `endpoint` override. Files are written under the configured `prefix` with date/hour partitioning, gzip-compressed by default.

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `provider` | string | Yes | — | `s3`, `gcs`, or `azure_blob`. |
| `bucket` | string | Yes | — | Bucket name (the container name for `azure_blob`). |
| `prefix` | string | Yes | — | Key prefix the date/hour-partitioned object path is appended to, e.g. `ai-gateway`. |
| `region` | string | No | `us-east-1` | AWS region for S3 SigV4 scope. Defaults to `us-east-1` when unset, so set it for any non-default-region bucket. |
| `endpoint` | string | No | — | S3-compatible host override (MinIO / Cloudflare R2 / OSS). Leave unset for a provider's native endpoint. `credential_ref` mode only. |
| `compression` | string | No | `gzip` | `gzip` or `none`. |
| `auth_mode` | string | No | `credential_ref` | `credential_ref` or `cloud_identity`. See [Authentication](#authentication). |
| `credential_ref` | string | Conditional | — | Required when `auth_mode` is `credential_ref`; omit it for `cloud_identity`. |

### Authentication

#### Static key reference (`credential_ref`)

The data plane resolves `credential_ref` to environment variables it reads locally, named `OBJSTORE_CRED_<SLUG>_<FIELD>`. `<SLUG>` is `credential_ref` upper-cased with every character that is not an ASCII letter or digit folded to `_`. To keep that mapping unambiguous — so two different references cannot silently fold onto the same variables — use only lowercase letters, digits, and underscores in `credential_ref`; the control plane and dashboard enforce `^[a-z0-9_]+$`.

Set these variables on the data plane (shown for `credential_ref = acme_s3_prod`, where `<SLUG>` is `ACME_S3_PROD`):

| Provider | Required variables | Optional variables |
|---|---|---|
| `s3` | `OBJSTORE_CRED_<SLUG>_AWS_ACCESS_KEY_ID`, `OBJSTORE_CRED_<SLUG>_AWS_SECRET_ACCESS_KEY` | `OBJSTORE_CRED_<SLUG>_AWS_SESSION_TOKEN` |
| `gcs` | `OBJSTORE_CRED_<SLUG>_GCS_SERVICE_ACCOUNT_KEY` (full service-account JSON) | — |
| `azure_blob` | `OBJSTORE_CRED_<SLUG>_AZURE_ACCOUNT`, `OBJSTORE_CRED_<SLUG>_AZURE_ACCESS_KEY` | — |

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

```bash title="Set the matching variables on the data plane"
OBJSTORE_CRED_ACME_S3_PROD_AWS_ACCESS_KEY_ID=<your key id>
OBJSTORE_CRED_ACME_S3_PROD_AWS_SECRET_ACCESS_KEY=<your secret>
# optional, for temporary credentials:
OBJSTORE_CRED_ACME_S3_PROD_AWS_SESSION_TOKEN=<token>
```

:::caution A missing key fails delivery silently
If a required `OBJSTORE_CRED_<SLUG>_*` variable is unset or empty, the exporter config still validates and the row shows as enabled, but every delivery fails and the sink reports unhealthy. Set the variables before, or right after, creating the exporter.
:::

#### Data-plane cloud identity (`cloud_identity`)

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

## Aliyun SLS exporter (`kind: aliyun_sls`)

Sends request-event logs to an Alibaba Cloud Simple Log Service (SLS) logstore. The data plane signs each batch with an AccessKey it resolves locally and posts to `https://<project>.<endpoint>`.

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `endpoint` | string | Yes | — | SLS region host with no scheme, e.g. `ap-southeast-3.log.aliyuncs.com`. The data plane posts to `https://<project>.<endpoint>`. |
| `project` | string | Yes | — | SLS project name. |
| `logstore` | string | Yes | — | SLS logstore that receives the logs. |
| `credential_ref` | string | Yes | — | Opaque pointer to the AccessKey, resolved on the data plane (see below). |
| `content_mode` | string | No | `metadata_only` | `metadata_only` or `full`. See [Content capture](#content-capture). |
| `content_max_bytes` | integer | No | `131072` | Per-field byte cap under `full`; minimum `1`, maximum `1048576` (1 MiB). Ignored under `metadata_only`. |

The data plane resolves `credential_ref` to two environment variables it reads locally — `SLS_CRED_<SLUG>_AK_ID` and `SLS_CRED_<SLUG>_AK_SECRET` — where `<SLUG>` is the reference upper-cased with every character that is not an ASCII letter or digit folded to `_` (so `acme_sls` becomes `ACME_SLS`).

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

```bash title="Set the matching variables on the data plane"
SLS_CRED_ACME_SLS_AK_ID=<your accesskey id>
SLS_CRED_ACME_SLS_AK_SECRET=<your accesskey secret>
```

## Datadog exporter (`kind: datadog`)

Sends each request event as one log to the Datadog Logs HTTP intake. The data plane gzip-compresses each batch and posts it to `https://http-intake.logs.<site>/api/v2/logs`.

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `site` | string | Yes | — | Datadog site: one of `datadoghq.com`, `us3.datadoghq.com`, `us5.datadoghq.com`, `datadoghq.eu`, `ap1.datadoghq.com`, `ap2.datadoghq.com`, `ddog-gov.com`. The intake host is `http-intake.logs.<site>`. |
| `credential_ref` | string | Yes | — | Opaque pointer to the Datadog API key, resolved on the data plane (see below). |
| `service` | string | Yes | — | The Datadog `service` reserved attribute every log from this exporter carries. |
| `ddsource` | string | No | `aisix-ai-gateway` | The Datadog `ddsource` reserved attribute. |
| `tags` | string[] | No | `[]` | Tags rendered into Datadog's comma-joined `ddtags` attribute, e.g. `["team:platform", "tier:prod"]` becomes `team:platform,tier:prod`. |
| `content_mode` | string | No | `metadata_only` | `metadata_only` or `full`. See [Content capture](#content-capture). |
| `content_max_bytes` | integer | No | `131072` | Per-field byte cap under `full`; minimum `1`, maximum `1048576` (1 MiB). Ignored under `metadata_only`. |

The data plane resolves `credential_ref` to `DD_CRED_<SLUG>_API_KEY`, read from its local environment, where `<SLUG>` is the reference upper-cased with every character that is not an ASCII letter or digit folded to `_` (so `acme_dd` becomes `ACME_DD`). To keep that mapping unambiguous — so two different references cannot silently fold onto the same variable — use only lowercase letters, digits, and underscores in `credential_ref`; the control plane and dashboard enforce `^[a-z0-9_]+$`.

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

```bash title="Set the matching variable on the data plane"
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

## Endpoint validation

The admin API validates each kind's destination and rejects plain `http://` targets unless they point to an allowed loopback-style host. For `otlp_http`, the allowed non-TLS development hosts are:

- `http://127.0.0.1/...`
- `http://localhost/...`
- `http://mock-otlp/...`
- `http://otel-collector/...`

For non-loopback deployments, use `https://...`. The `object_store`, `aliyun_sls`, and `datadog` kinds apply analogous per-kind host validation (an S3-compatible host, a `*.aliyuncs.com` region host, and a Datadog site from the allow-list, respectively), each with the same loopback bypass for local emulators. This protects against accidentally configuring plain HTTP exporters for non-local destinations.

## Verification

After creating an exporter, list the exporters in the environment and confirm yours is present and active:

```bash title="List exporters"
curl -sS http://127.0.0.1:3001/admin/v1/observability_exporters \
  -H "Authorization: Bearer YOUR_ADMIN_KEY"
```

Expected: the response includes the exporter you created, with `"enabled": true`, the `kind` you set, and the destination fields (for example `provider` / `bucket` / `prefix` for `object_store`). No credential secret appears in the response — only the `credential_ref` (or none, for `cloud_identity`).

To confirm telemetry actually arrives, generate gateway traffic and check the destination directly: traces in your OTLP backend, NDJSON objects under the bucket prefix, log entries in the SLS logstore, or logs in Datadog's Log Explorer filtered by `service`.

## Operational notes

- Start with one exporter and verify delivery before adding several.
- For `otlp_http`, keep the destination's auth credentials in `headers`, aligned with its OTLP/HTTP auth model.
- For `object_store`, prefer `auth_mode: cloud_identity` when the data plane runs in the bucket's cloud — there are no keys to provision or rotate; use `credential_ref` for S3-compatible targets (MinIO / R2) or cross-cloud setups.
- Disable an exporter (`enabled: false`) rather than deleting it while you diagnose delivery issues — this preserves its configuration.

## Troubleshooting

### The exporter saves but no telemetry appears downstream

Check endpoint correctness, the destination's auth (`headers` for `otlp_http`, `credential_ref` resolution for the others), and whether the exporter is enabled.

### An `object_store`, `aliyun_sls`, or `datadog` exporter saves but nothing arrives

The configuration validates without the credential being present, so a missing reference resolves to an unhealthy sink. With a static `credential_ref`, confirm the per-kind variables are set and non-empty on the data plane — `OBJSTORE_CRED_<SLUG>_*`, `SLS_CRED_<SLUG>_AK_ID` / `_AK_SECRET`, or `DD_CRED_<SLUG>_API_KEY`. With `object_store` `cloud_identity`, confirm the data plane's attached identity has bucket write access (`s3:PutObject` / `storage.objects.create`).

### The admin API rejects an `http://` endpoint

That is expected unless the destination is one of the allowed loopback development hosts. See [Endpoint validation](#endpoint-validation).

## Related pages

- [Admin API](admin-api.md) — the full admin resource surface and auth model.
- [Metrics and logs](../operations/metrics-and-logs.md) — the data plane's own metrics and request logs.
- [Resource schemas](../reference/resource-schemas.md) — the schema map for every dynamic resource, including `ObservabilityExporter`.
