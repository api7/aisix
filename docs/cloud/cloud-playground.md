---
title: Cloud Playground
description: Understand AISIX Cloud playground behavior and its limitations relative to the managed data plane.
sidebar_position: 74
toc_max_heading_level: 2
---

The AISIX Cloud playground lets you try a model from the control plane while you are setting up Cloud resources. It is useful for early feedback, but it does not simulate the full production request path.

Use live requests through the managed data plane when you need to validate routing, cache, guardrails, rate limits, budgets, or observability.

## Playground and Live Traffic Paths

Playground requests run from the control plane to the upstream provider. They do not exercise managed data-plane behavior such as model routing, response caching, guardrail execution, rate-limit enforcement, budget enforcement, or data-plane observability.

That difference matters when a playground request succeeds but a live managed request behaves differently.

The playground is appropriate for model-selection checks, early provider credential validation, exploratory prompts from the Cloud UI, and basic provider-call confirmation. The managed data plane is the right place to validate caller API keys, model aliases, routing rules, cache behavior, guardrails, budgets, rate limits, logs, and metrics.

## Troubleshooting

Use these checks when playground results do not match the managed traffic path.

### The Playground Succeeds but Managed Traffic Behaves Differently

Treat the playground result as an early provider and configuration check, then verify the live data-plane path. Confirm the resource belongs to the environment served by the data plane, the saved resource reached the managed data plane, the request uses the managed data-plane endpoint, and data-plane logs, metrics, and gateway response headers match the live request.

## Related Reading

For how saved Cloud state reaches live traffic, see [Resource projection](/ai-gateway/cloud/resource-projection). For live data-plane observability and production readiness, see [Metrics and logs](/ai-gateway/operations/metrics-and-logs) and [Feature availability](/ai-gateway/overview/feature-matrix).
