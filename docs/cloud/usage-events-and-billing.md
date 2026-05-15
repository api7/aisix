---
title: Usage Events And Billing
description: Understand the current AISIX Cloud usage-event ingestion and billing-oriented control-plane workflows.
sidebar_position: 75
---

AISIX Cloud collects usage information from the managed data plane and exposes customer-facing usage and billing workflows above that telemetry.

Current documented behavior includes:

- `/dp/telemetry` ingestion on the control-plane side
- usage-event views surfaced from the control plane
- managed budget enforcement and budget-driven `429` outcomes on real DP traffic

This is where Cloud behavior goes beyond a standalone gateway instance with only local request handling.

## Operational Meaning

In managed mode, the data plane is not only serving traffic. It is also emitting usage-oriented signals back to the control plane.

Those signals support:

- usage visibility
- budget workflows
- billing-oriented control-plane features

## Latency Signals On The Usage Event

In addition to token counts, status, and cost, each usage event the data plane emits carries:

- `latency_ms` — total elapsed time for the request, populated on every request path
- `ttft_ms` — time to first token, populated on streaming chat completions when the first chunk carrying generated output arrives (text content **or** a tool-call delta)

`ttft_ms` is omitted from the wire when it would otherwise be zero — non-streaming, cache-hit, and error paths do not contribute a TTFT value. Use it to break down end-to-end latency into the upstream-warm-up portion versus the streaming portion.

## Budget Relationship

The most important customer-visible effect is that managed deployments can apply real budget decisions on data-plane traffic, which can result in `429` denials when a budget policy is exceeded.

## Troubleshooting

### Usage appears incomplete in Cloud

Look first at telemetry delivery and managed data-plane health, not just caller request success.

## Related Pages

- [AISIX Cloud Overview](overview.md)
- [Offline Resilience](offline-resilience.md)
- [Budgets](../configuration/budgets.md)
