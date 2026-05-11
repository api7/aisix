---
title: Metrics And Logs
description: Observe AISIX AI Gateway through admin metrics, access logs, usage events, and exporter fan-out.
sidebar_position: 54
---

The gateway currently exposes observability through multiple paths.

## Metrics

`GET /metrics` on the admin listener is the Prometheus scrape endpoint.

This endpoint is unauthenticated by design on the private admin listener.

## Access Logs And Usage Signals

Current proxy behavior emits:

- structured access logs
- metrics updates
- usage-event emission on request paths that support it

## Response Headers With Operational Value

Current response headers include:

- endpoint-specific correlation headers such as `x-aisix-call-id` or `x-aisix-request-id`
- `x-aisix-cache` on chat cache hit or miss paths
- `Retry-After` on rate-limit-style rejections when applicable

## Exporters

Observability exporters are dynamic resources configured through `/admin/v1/observability_exporters`.

Current exporter support is `otlp_http` only.

## Related Pages

- [Observability Exporters](../configuration/observability-exporters.md)
- [Health Checks](health-checks.md)
- [Reference: Headers And Error Codes](../reference/headers-and-error-codes.md)
