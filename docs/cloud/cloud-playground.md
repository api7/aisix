---
title: Cloud Playground
description: Understand AISIX Cloud playground behavior and its limitations relative to the managed data plane.
sidebar_position: 74
---

The AISIX Cloud playground is a preview surface for trying a model or
checking early configuration. It is not a production-path simulator.

Use the playground for quick feedback while setting up resources. Use
live requests through the managed data plane when you need to validate
routing, cache, guardrails, rate limits, budgets, or observability.

## How the playground differs

The current playground path sends the request from the control plane to
the upstream provider. It does not exercise the managed data plane's:

- model routing path
- response cache
- guardrail execution
- rate-limit enforcement
- budget enforcement on live data-plane traffic
- data-plane access logs and metrics path

That boundary matters when a playground request succeeds but the live
managed request behaves differently.

## When to use the playground

Use it for:

- quick model-selection checks
- early provider credential validation
- exploratory prompts from the Cloud UI
- confirming that a basic provider call can succeed

## When to use live data-plane traffic

Use the managed data plane for:

- validating caller API keys
- validating model aliases and routing rules
- validating cache behavior
- validating guardrails
- validating budgets and rate limits
- checking the actual request logs and metrics path

## Troubleshooting

### The playground succeeds but real managed traffic behaves differently

Treat the playground result as a provider/configuration preview, then
check the live data-plane path:

1. Confirm the resource belongs to the right environment.
2. Confirm projection reached the data plane.
3. Send the request through the managed data-plane endpoint.
4. Check data-plane logs, metrics, and gateway response headers.

## Next steps

- [Resource projection](/ai-gateway/cloud/resource-projection) explains
  how saved Cloud state reaches live traffic.
- [Metrics and logs](/ai-gateway/operations/metrics-and-logs) explains
  live data-plane observability.
- [Feature Status](/ai-gateway/overview/feature-matrix) shows the current
  support boundary for Cloud features.
