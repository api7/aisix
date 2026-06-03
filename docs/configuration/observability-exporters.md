---
title: Observability Exporters
description: Configure OTLP/HTTP observability exporters for AISIX AI Gateway data-plane telemetry fan-out.
sidebar_position: 40
---

Observability exporters send request telemetry from the data plane to an
OTLP/HTTP traces endpoint.

Use an exporter when you want telemetry delivery to be configurable through
dynamic resources instead of only through process bootstrap settings.

Current scope is `kind: "otlp_http"`.

## Create an exporter

```shell
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

The endpoint must be the full OTLP/HTTP traces URL. The gateway does not append
`/v1/traces`, because vendors can use different paths.

Set `enabled: false` to keep the exporter in the snapshot but skip fan-out.

## Endpoint restrictions

Use `https://` for non-local exporter endpoints.

The admin validation layer allows plain `http://` only for local-development
targets:

- `http://127.0.0.1/...`
- `http://localhost/...`
- `http://mock-otlp/...`
- `http://otel-collector/...`

This prevents accidentally sending telemetry over plaintext HTTP to a remote
destination.

## Headers and credentials

Use `headers` for static destination credentials, such as:

```json
{
  "headers": {
    "Authorization": "Bearer YOUR_OTLP_TOKEN"
  }
}
```

or vendor-specific headers:

```json
{
  "headers": {
    "x-honeycomb-team": "YOUR_TEAM_KEY"
  }
}
```

Header values are plaintext in the runtime resource. Treat them with the same
care as provider-key secrets: restrict access to the config store and keep the
data-plane trust boundary explicit.

## Runtime behavior

Exporter traffic is sent by the data plane. The control plane does not open an
HTTP connection to your exporter endpoint.

Current fan-out is metadata-oriented. It includes request status, token counts,
model and provider identifiers, request ids, finish reason, and timing. Prompt
and response bodies are not included in the OTLP/HTTP span payload.

Disabled exporters remain in the snapshot and are skipped.

## Operator guidance

Start with one exporter and verify delivery before adding several destinations.

Keep destination credentials scoped to telemetry export only.

Disable an exporter before deleting it when you are diagnosing delivery issues.
That keeps the endpoint and headers available for rollback.

## Troubleshooting

### The exporter saves but no telemetry appears downstream

Check that the endpoint is the full OTLP/HTTP traces URL, the destination
headers are valid, and the exporter is enabled.

### The admin API rejects an `http://` endpoint

That is expected for non-local destinations. Use `https://`, or use one of the
allowed local-development hostnames.

### The downstream service expects a different path

Set `endpoint` to the exact receiver path. The gateway does not rewrite or
append the OTLP path.

## Next steps

- [Admin API](admin-api.md) explains standalone admin writes.
- [Metrics and logs](../operations/metrics-and-logs.md) covers runtime telemetry.
- [Resource schemas](../reference/resource-schemas.md) explains generated schema source of truth.
