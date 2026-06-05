---
title: Metrics and Logs
description: Observe AISIX AI Gateway through admin metrics, access logs, usage events, and exporter fan-out.
toc_max_heading_level: 2
sidebar_position: 54
---

AISIX AI Gateway exposes observability through metrics, logs, response headers, usage events, and optional exporter fan-out. Use these signals together; no single signal explains every failure mode.

## Observability Signals

Start with Prometheus metrics when you need trends: request volume, latency, rate-limit outcomes, token counters, cost counters, and exporter health.

Use access logs when you need to diagnose one request. Logs can tie a caller-visible failure to the selected model, upstream provider, and gateway decision path.

Use response headers for caller-visible hints, such as request correlation, cache outcome, or retry timing.

Use usage events for accounting-oriented workflows, including usage, cost, budgets, and Cloud billing flows.

Use OTLP exporters when telemetry needs to leave the data plane and flow into an external collector.

## Prometheus Metrics

`GET /metrics` on the admin listener is the default Prometheus scrape endpoint. Change the path with `observability.metrics.prometheus.path`, or disable the endpoint with `observability.metrics.prometheus.enabled: false`.

This endpoint is unauthenticated by design on the private admin listener. Keep the admin listener private.

:::note Metrics are empty before the first request
Metric families are registered lazily on first observation. Immediately after boot, `GET /metrics` can return an empty body. Send one model request, then check again for series such as `aisix_requests_total` and `aisix_tokens_consumed_total`.
:::

AISIX emits native metric names with the `aisix_` prefix. Start with the category that matches the question you are answering.

| Question | Metrics to inspect |
| --- | --- |
| Request volume and latency | `aisix_requests_total`, `aisix_request_duration_seconds`, `aisix_llm_requests_total`, `aisix_llm_request_duration_seconds`, `aisix_llm_time_to_first_token_seconds` |
| Usage and cost | `aisix_tokens_consumed_total`, `aisix_llm_input_tokens_total`, `aisix_llm_output_tokens_total`, `aisix_llm_total_tokens_total`, `aisix_llm_spend_micro_usd_total` |
| Rate limits and budgets | `aisix_ratelimit_rejections_total`, `aisix_ratelimit_remaining_requests`, `aisix_ratelimit_remaining_tokens`, budget gauges, `aisix_budget_details_present` |
| Proxy health | `aisix_proxy_requests_total`, `aisix_proxy_failed_requests_total`, `aisix_proxy_request_duration_seconds`, `aisix_proxy_in_flight_requests` |
| Routing, cache, and exporters | `aisix_deployment_*`, `aisix_routing_*`, Redis counters, usage-event drop counters, OTLP fan-out drop or failure counters |

Labels are limited to values the data plane can reliably know, such as `endpoint`, `inbound_protocol`, `provider`, `model`, `upstream_model`, `provider_key_id`, `api_key_id`, `team_id`, `user_id`, `status`, and `outcome`.

The local `/metrics` endpoint lives on the admin listener. A Cloud managed data plane does not bind the standalone admin listener and does not expose local `/metrics` scraping.

To export metrics or telemetry from a managed data plane, configure an OTLP exporter through AISIX Cloud. The data plane sends telemetry to the configured collector instead of waiting to be scraped locally.

## Logs and Usage Events

Access logs answer what happened to a single request. Metrics answer what is happening over time. Usage events answer what accounting-oriented event was emitted for supported request paths.

For streaming chat completions, usage events can include `ttft_ms`, the elapsed time from request entry to the first upstream chunk that contains generated output. Role-only opening chunks are skipped so the value tracks time to actual output.

`ttft_ms` is meaningful only on streaming paths. Non-streaming, cache-hit, and error paths do not emit a TTFT value.

Response headers can provide fast per-request hints:

| Header | Use |
| --- | --- |
| `x-aisix-call-id` or `x-aisix-request-id` | Correlate a caller-visible request with logs and telemetry. |
| `x-aisix-cache` | Identify chat cache hit or miss paths. |
| `Retry-After` | Show retry timing on supported rate-limit rejections. |

Use headers with logs and metrics. A header can identify the request; it does not replace backend observability.

## Export Telemetry

Observability exporters are dynamic resources configured through the admin API or, in managed operation, through AISIX Cloud. The supported exporter type is `otlp_http`.

Use exporters when you need telemetry fan-out to an external observability system.

## Troubleshooting

Use these checks when telemetry appears healthy but does not explain caller-visible failures.

### Metrics Look Healthy but Callers Report Failures

Inspect access logs and caller-visible headers for request-level evidence. Metrics can hide individual failures inside aggregates.

### Exporters Are Configured but Downstream Traces Are Missing

Check exporter enablement, endpoint correctness, outbound connectivity from the data plane, and exporter drop/failure counters.

## Related Reading

Configure exporter resources with [Observability exporters](/ai-gateway/configuration/observability-exporters). For health endpoints and caller-visible headers, see [Health checks](/ai-gateway/operations/health-checks) and [Headers and error codes](/ai-gateway/reference/headers-and-error-codes).
