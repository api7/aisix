---
title: Metrics And Logs
description: Observe AISIX AI Gateway through admin metrics, access logs, usage events, and exporter fan-out.
sidebar_position: 54
---

The gateway currently exposes observability through multiple paths.

Use them together. No single signal tells the whole story.

## Metrics

`GET /metrics` is the Prometheus scrape endpoint, served on a **dedicated metrics listener** bound at `observability.metrics.prometheus.addr` (default `0.0.0.0:9090`). The mechanism is identical in every deployment mode — standalone and Cloud managed data planes are scraped the same way, and the admin listener does not serve `/metrics`. You can change the path with `observability.metrics.prometheus.path` or disable the listener entirely with `observability.metrics.prometheus.enabled: false`.

This endpoint is unauthenticated by design. Keep it on a private network and restrict access at the network layer (firewall, security group, or Kubernetes NetworkPolicy).

Treat `/metrics` as infrastructure-facing, not as a public diagnostics surface.

:::note `/metrics` is empty before the first request
Metric families are registered lazily on first observation — the gateway does not pre-register `# HELP` / `# TYPE` lines at startup. Immediately after boot, `GET /metrics` returns an empty body. This is expected, not a misconfigured endpoint. To smoke-test, send one model request and then re-check `/metrics`; you should now see series such as `aisix_requests_total` and `aisix_tokens_consumed_total`.
:::

### Scraping The Data Plane

To scrape any data plane (standalone or managed):

1. **Expose the metrics port** in your data plane deployment. The proxy and admin ports are unaffected — only the metrics port needs to be reachable by your Prometheus.

   Docker:

   ```bash
   docker run ... -p 9090:9090 <aisix-dp-image>
   ```

   Kubernetes — add the port to the container and a `Service`:

   ```yaml
   ports:
     - { name: metrics, containerPort: 9090 }
   ```

2. **Point Prometheus at it.** A static scrape config:

   ```yaml
   scrape_configs:
     - job_name: aisix-data-plane
       metrics_path: /metrics
       static_configs:
         - targets: ["<data-plane-host>:9090"]
   ```

   Or, with the Prometheus Operator, a `ServiceMonitor`:

   ```yaml
   apiVersion: monitoring.coreos.com/v1
   kind: ServiceMonitor
   metadata:
     name: aisix-data-plane
   spec:
     selector:
       matchLabels: { app: aisix-dp }
     endpoints:
       - port: metrics
         path: /metrics
   ```

Restrict access to the metrics port at the network layer — it is unauthenticated, like every Prometheus exporter.

AISIX exposes native metric names with the `aisix_` prefix. Existing compatibility series remain:

- `aisix_requests_total`
- `aisix_request_duration_seconds`
- `aisix_ratelimit_rejections_total`
- `aisix_tokens_consumed_total`

The Prometheus integration also emits common LLM-gateway-category equivalents under AISIX-native names:

- usage and cost: `aisix_llm_input_tokens_total`, `aisix_llm_output_tokens_total`, `aisix_llm_total_tokens_total`, `aisix_llm_spend_micro_usd_total`
- request volume and latency: `aisix_llm_requests_total`, `aisix_llm_request_duration_seconds`, `aisix_llm_time_to_first_token_seconds`
- proxy health: `aisix_proxy_requests_total`, `aisix_proxy_failed_requests_total`, `aisix_proxy_request_duration_seconds`, `aisix_proxy_in_flight_requests`
- quotas and budgets: `aisix_ratelimit_remaining_requests`, `aisix_ratelimit_remaining_tokens`, and budget gauges when the control plane returns budget detail fields; `aisix_budget_details_present` tells scrapers whether the current budget response carried those optional fields
- deployment and routing: `aisix_deployment_*` and `aisix_routing_*` metric families when the request path has those events
- exporter/cache health: Redis, usage-event drop, and OTLP fan-out drop/failure counters

Labels are limited to values the data plane has reliably: `endpoint`, `inbound_protocol`, `provider`, `model`, `upstream_model`, `provider_key_id`, `api_key_id`, `team_id`, `user_id`, `status`, and `outcome`. User email, team alias, and end-user labels are not fabricated by the data plane.

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

Four kinds are supported — `otlp_http` (request traces), `object_store` (batched NDJSON to S3 / GCS / Azure Blob), `aliyun_sls` (Alibaba Cloud SLS logstore), and `datadog` (Datadog Logs intake). See [Observability Exporters](../configuration/observability-exporters.md) for each kind's fields, credential resolution, and destination validation.

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
