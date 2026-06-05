---
title: Observability Exporters
description: Configure OTLP/HTTP observability exporters for AISIX AI Gateway data-plane telemetry fan-out.
toc_max_heading_level: 2
sidebar_position: 40
---

Observability exporters send request telemetry from the data plane to an OTLP/HTTP traces endpoint.

Use an exporter when telemetry delivery should be configurable through dynamic resources instead of only through process bootstrap settings. The supported exporter kind is `otlp_http`.

## Configure an Exporter

Create an exporter resource that sends traces to an OTLP/HTTP endpoint:

```shell
curl -sS -X POST "http://127.0.0.1:3001/admin/v1/observability_exporters" \
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

The endpoint must be the full OTLP/HTTP traces URL. The gateway does not append `/v1/traces`, because vendors can use different paths.

Set `enabled: false` to keep the exporter configured but skip fan-out. Use `https://` for non-local exporter endpoints.

The admin validation layer allows plain `http://` only for local-development targets:

| Allowed HTTP target | Typical use |
| --- | --- |
| `http://127.0.0.1/...` | Local collector on the same host. |
| `http://localhost/...` | Local collector addressed by hostname. |
| `http://mock-otlp/...` | Test or development OTLP receiver. |
| `http://otel-collector/...` | Local container or service-network collector. |

This prevents accidentally sending telemetry over plaintext HTTP to a remote destination.

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

Header values are plaintext in the runtime resource. Treat them with the same care as provider-key secrets: restrict access to the config store and keep the data-plane trust model explicit.

## Behavior Details

Exporter traffic is sent by the data plane. The control plane does not open an HTTP connection to your exporter endpoint.

Exporter fan-out is metadata-oriented. It includes request status, token counts, model and provider identifiers, request IDs, finish reason, and timing. Prompt and response bodies are not included in the OTLP/HTTP span payload.

Disabled exporters remain configured and are skipped. Start with one exporter and verify delivery before adding several destinations. Keep destination credentials scoped to telemetry export only.

When diagnosing delivery issues, disable an exporter before deleting it. That keeps the endpoint and headers available for rollback.

## Troubleshooting

Use these checks when the exporter resource saves but telemetry does not arrive downstream.

### The Exporter Saves but No Telemetry Appears Downstream

Check that the endpoint is the full OTLP/HTTP traces URL, the destination headers are valid, and the exporter is enabled.

### The Admin API Rejects a Plain HTTP Endpoint

Non-local destinations require `https://`. For local development, use one of the allowed local-development hostnames.

### The Downstream Service Expects a Different Path

Set `endpoint` to the exact receiver path. The gateway does not rewrite or append the OTLP path.

## Related Reading

[Admin API](admin-api.md) covers standalone admin writes, and [Metrics and logs](../operations/metrics-and-logs.md) explains runtime telemetry. For exact accepted fields, see [Resource schemas](../reference/resource-schemas.md).
