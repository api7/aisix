---
title: Metrics And Logs
description: Observe AISIX AI Gateway through admin metrics, access logs, usage events, and exporter fan-out.
sidebar_position: 54
---

The gateway currently exposes observability through multiple paths.

Use them together. No single signal tells the whole story.

## Metrics

`GET /metrics` on the admin listener is the Prometheus scrape endpoint.

This endpoint is unauthenticated by design on the private admin listener.

Treat `/metrics` as infrastructure-facing, not as a public diagnostics surface.

## Access Logs And Usage Signals

Current proxy behavior emits:

- structured access logs
- metrics updates
- usage-event emission on request paths that support it

These signals answer different questions:

- access logs: what happened to one request
- metrics: what is happening over time
- usage events: what usage/accounting-oriented event was emitted on supported paths

### Streaming TTFT

For streaming chat completions, the per-request usage event carries `ttft_ms` — the elapsed milliseconds from request entry to the first upstream chunk that contains generated content (text or tool-call delta). Role-only opening chunks are skipped so the value reflects the time to actual output, not the time to the first SSE frame.

`ttft_ms` is meaningful only on the streaming path. Non-streaming, cache-hit, and error paths do not surface a TTFT value.

## Response Headers With Operational Value

Current response headers include:

- endpoint-specific correlation headers such as `x-aisix-call-id` or `x-aisix-request-id`
- `x-aisix-cache` on chat cache hit or miss paths
- `Retry-After` on rate-limit-style rejections when applicable

Those headers are often the fastest per-request debugging hints available to a client team.

## Exporters

Observability exporters are dynamic resources configured through `/admin/v1/observability_exporters`.

Current exporter support is `otlp_http` only.

## Operator Workflow

1. use `/metrics` for scrape-based monitoring
2. use access logs for request-level diagnosis
3. use response headers for caller-visible operational hints
4. use dynamic exporters when you need telemetry fan-out to external observability systems

## Troubleshooting

### Metrics look healthy but callers report failures

Inspect access logs and caller-visible headers for request-level evidence.

### Exporters are configured but downstream traces are missing

Validate exporter enablement, endpoint correctness, and outbound connectivity from the data plane.

## Related Pages

- [Observability Exporters](../configuration/observability-exporters.md)
- [Health Checks](health-checks.md)
- [Reference: Headers And Error Codes](../reference/headers-and-error-codes.md)
