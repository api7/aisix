---
title: Observability Exporters
description: Configure OTLP/HTTP observability exporters for AISIX AI Gateway data-plane telemetry fan-out.
sidebar_position: 40
---

Observability exporters let the data plane send request telemetry directly to your OTLP/HTTP endpoint.

Current scope is `kind: "otlp_http"` only.

## Current Fields

- `name`
- `enabled`
- `kind`
- `endpoint`
- optional `headers`

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

## Runtime Model

Current exporter behavior:

- exporters are environment-scoped dynamic resources
- the data plane, not the control plane, sends the HTTP export traffic
- disabled exporters remain in the snapshot but are skipped

This keeps sensitive prompt and response content on the data plane egress path.

## Related Pages

- [Admin API](admin-api.md)
- [Metrics And Logs](../operations/metrics-and-logs.md)
- [Reference: Resource Schemas](../reference/resource-schemas.md)
