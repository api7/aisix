---
title: Observability Exporters
description: Configure OTLP/HTTP observability exporters for AISIX AI Gateway data-plane telemetry fan-out.
sidebar_position: 40
---

Observability exporters let the data plane send request telemetry directly to your OTLP/HTTP endpoint.

Current scope is `kind: "otlp_http"` only.

Use this page when you want request-level telemetry fan-out without restarting the process for every endpoint change.

## Current Fields

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
- disable exporters rather than deleting them immediately when you are diagnosing delivery issues

## Troubleshooting

### The exporter saves but no telemetry appears downstream

Check endpoint correctness, destination auth headers, and whether the exporter is enabled.

### The admin API rejects an `http://` endpoint

That is expected unless the destination is one of the allowed local-development forms.

## Related Pages

- [Admin API](admin-api.md)
- [Metrics And Logs](../operations/metrics-and-logs.md)
- [Reference: Resource Schemas](../reference/resource-schemas.md)
