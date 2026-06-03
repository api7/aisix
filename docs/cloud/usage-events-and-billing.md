---
title: Usage Events and Billing
description: Understand AISIX Cloud usage-event ingestion and billing-oriented control-plane workflows.
sidebar_position: 75
---

AISIX Cloud collects usage information from the managed data plane and
uses it for customer-facing usage, billing, and budget workflows.

In managed mode, the data plane does more than serve AI traffic. It also
emits usage events to Cloud so operators can understand consumption and
so Cloud can make budget decisions.

## What usage events support

Usage telemetry supports:

- usage visibility in Cloud
- billing-oriented workflows
- managed budget evaluation
- budget-driven `429` responses on live data-plane traffic

This is one of the main differences between standalone operation and
Cloud-managed operation. A standalone gateway can serve the request
locally; Cloud-managed operation also reports usage back to the control
plane.

## How the data plane reports telemetry

The managed data plane sends usage-oriented data to the control plane
through `/dp/telemetry`.

Usage events include request outcome and consumption signals such as
token usage, status, cost, and latency. Two latency fields are especially
useful:

- `latency_ms` is the total elapsed time for the request.
- `ttft_ms` is time to first token for streaming chat completions. It is
  populated when the first generated output arrives, including text
  content or a tool-call delta.

`ttft_ms` is omitted when it would otherwise be zero. Non-streaming,
cache-hit, and error paths do not contribute a TTFT value.

## How usage connects to budgets

Managed budgets can affect live data-plane traffic. When a budget policy
is exceeded, the data plane can return `429` for affected requests.

If a caller receives a budget-related `429`, inspect both the configured
budget policy and the data-plane telemetry path. A request can fail from
a budget decision even when provider credentials and model routing are
otherwise valid.

## Troubleshooting

### Usage appears incomplete in Cloud

Check:

1. The managed data plane is healthy.
2. The data plane can reach the `/dp/telemetry` endpoint.
3. The request path is live data-plane traffic, not a preview-only path.
4. Budget or telemetry errors are not being hidden behind general proxy
   failures.

## Next steps

- [Budgets](/ai-gateway/configuration/budgets) explains budget policy
  configuration.
- [Metrics and logs](/ai-gateway/operations/metrics-and-logs) explains
  live data-plane observability.
- [Offline resilience](/ai-gateway/cloud/offline-resilience) explains
  what happens when Cloud connectivity is temporarily unavailable.
