---
title: Metrics and Logs
description: Observe AISIX AI Gateway through admin metrics, access logs, usage events, and exporter fan-out.
sidebar_position: 54
---

AISIX AI Gateway exposes observability through metrics, logs, response
headers, usage events, and optional exporter fan-out. Use these signals
together; no single signal explains every failure mode.

## Start with the right observability signal

Start with Prometheus metrics when you need trends: request volume,
latency, rate-limit outcomes, token counters, cost counters, and exporter
health.

Use access logs when you need to diagnose one request. Logs can tie a
caller-visible failure to the selected model, upstream provider, and
gateway decision path.

Use response headers for caller-visible hints, such as request
correlation, cache outcome, or retry timing.

Use usage events for accounting-oriented workflows, including usage,
cost, budgets, and Cloud billing flows.

Use OTLP exporters when telemetry needs to leave the data plane and flow
into an external collector.

## Prometheus metrics

`GET /metrics` on the admin listener is the default Prometheus scrape
endpoint. Operators can change the path with
`observability.metrics.prometheus.path`, or disable the endpoint with
`observability.metrics.prometheus.enabled: false`.

This endpoint is unauthenticated by design on the private admin listener.
Keep the admin listener private.

:::note Metrics are empty before the first request
Metric families are registered lazily on first observation. Immediately
after boot, `GET /metrics` can return an empty body. Send one model
request, then check again for series such as `aisix_requests_total` and
`aisix_tokens_consumed_total`.
:::

AISIX emits native metric names with the `aisix_` prefix. Common metric
families include:

- request volume and latency, such as `aisix_requests_total` and
  `aisix_request_duration_seconds`
- token and cost counters, such as `aisix_llm_input_tokens_total`,
  `aisix_llm_output_tokens_total`, and `aisix_llm_spend_micro_usd_total`
- rate-limit and budget signals, such as
  `aisix_ratelimit_rejections_total` and budget gauges
- proxy health and in-flight request signals
- routing, cache, Redis, usage-event, and exporter health signals when
  those paths are used

Labels are limited to values the data plane can reliably know, such as
`endpoint`, `inbound_protocol`, `provider`, `model`, `upstream_model`,
`provider_key_id`, `api_key_id`, `team_id`, `user_id`, `status`, and
`outcome`.

## Managed data-plane metrics

The local `/metrics` endpoint lives on the admin listener. A Cloud
managed data plane does not bind the standalone admin listener and does
not expose local `/metrics` scraping.

To export metrics or telemetry from a managed data plane, configure an
OTLP exporter through AISIX Cloud. The data plane sends telemetry to the
configured collector instead of waiting to be scraped locally.

## Logs and usage events

Access logs answer what happened to a single request. Metrics answer what
is happening over time. Usage events answer what accounting-oriented
event was emitted for supported request paths.

For streaming chat completions, usage events can include `ttft_ms`, the
elapsed time from request entry to the first upstream chunk that contains
generated output. Role-only opening chunks are skipped so the value
tracks time to actual output.

`ttft_ms` is meaningful only on streaming paths. Non-streaming,
cache-hit, and error paths do not emit a TTFT value.

## Response headers

Response headers can provide fast per-request hints:

- `x-aisix-call-id` or `x-aisix-request-id` for correlation
- `x-aisix-cache` on chat cache hit or miss paths
- `Retry-After` on supported rate-limit rejections

Use headers with logs and metrics. A header can identify the request; it
does not replace backend observability.

## Exporters

Observability exporters are dynamic resources configured through the
admin API or, in managed operation, through AISIX Cloud. Current exporter
support is `otlp_http`.

Use exporters when you need telemetry fan-out to an external
observability system.

## Troubleshooting

### Metrics look healthy but callers report failures

Inspect access logs and caller-visible headers for request-level
evidence. Metrics can hide individual failures inside aggregates.

### Exporters are configured but downstream traces are missing

Check exporter enablement, endpoint correctness, outbound connectivity
from the data plane, and exporter drop/failure counters.

## Next steps

- [Observability exporters](/ai-gateway/configuration/observability-exporters)
  explains exporter resources.
- [Health checks](/ai-gateway/operations/health-checks) explains health
  surfaces.
- [Headers and error codes](/ai-gateway/reference/headers-and-error-codes)
  documents caller-visible headers.
