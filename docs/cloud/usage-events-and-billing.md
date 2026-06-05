---
title: Usage Events and Billing
description: Understand AISIX Cloud usage-event ingestion and billing-oriented control-plane workflows.
sidebar_position: 75
toc_max_heading_level: 2
---

AISIX Cloud collects usage information from the managed data plane and uses it for customer-facing usage, billing, and budget workflows.

In managed mode, the data plane serves AI traffic and emits usage events to Cloud. These events help teams understand consumption and let Cloud make budget decisions.

## Usage Flow

Usage telemetry supports Cloud visibility, billing-oriented workflows, managed budget evaluation, and budget-driven `429` responses on live data-plane traffic.

This is one of the main differences between standalone operation and Cloud-managed operation. A standalone gateway can serve the request locally; Cloud-managed operation also reports usage back to the control plane.

The managed data plane sends usage-oriented data to the control plane through `/dp/telemetry`.

Usage events include request outcome and consumption signals such as token usage, status, cost, and latency. `latency_ms` records the total elapsed time for the request. `ttft_ms` records time to first token for streaming chat completions, when the first generated output arrives, including text content or a tool-call delta.

`ttft_ms` is omitted when it would otherwise be zero. Non-streaming, cache-hit, and error paths do not contribute a TTFT value.

## Budget Behavior

Managed budgets can affect live data-plane traffic. When a budget policy is exceeded, the data plane can return `429` for affected requests.

If a caller receives a budget-related `429`, inspect both the configured budget policy and the data-plane telemetry path. A request can fail from a budget decision even when provider credentials and model routing are otherwise valid.

## Troubleshooting

Use these checks when reported usage does not match the traffic you expected.

### Usage Appears Incomplete in Cloud

Confirm the managed data plane is healthy, the data plane can reach `/dp/telemetry`, and the request is live data-plane traffic rather than only a Cloud UI check. Also check whether budget or telemetry errors are hidden behind general proxy failures.

## Related Reading

For budget policy behavior, see [Budgets](/ai-gateway/configuration/budgets). For live data-plane observability, see [Metrics and logs](/ai-gateway/operations/metrics-and-logs). For temporary Cloud connectivity loss, see [Offline resilience](/ai-gateway/cloud/offline-resilience).
