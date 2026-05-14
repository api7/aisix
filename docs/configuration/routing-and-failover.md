---
title: Routing And Failover
description: Configure virtual models, target selection strategies, and retry behavior in AISIX AI Gateway.
sidebar_position: 35
---

Routing lets one caller-visible model alias dispatch across multiple direct models.

This is the gateway's current virtual-model mechanism.

Use it when you want to separate the caller contract from the individual upstream target that serves a given request.

## Current Strategies

- `failover`
- `round_robin`
- `weighted`

Each strategy answers a different operator question:

- `failover`: what should happen when the primary target is down or retryable-failing
- `round_robin`: how should traffic spread across peers over time
- `weighted`: how should the first target be biased when targets have different desired shares

## Example: Failover Routing

```json title="Routing block"
{
  "routing": {
    "strategy": "failover",
    "targets": [
      { "model": "gpt-4o-primary" },
      { "model": "gpt-4o-secondary" }
    ],
    "retries": 1,
    "max_fallbacks": 1,
    "retry_on_429": true
  }
}
```

## Strategy Semantics

### `failover`

- starts at the first target every time
- only moves to the next target when the prior attempt fails with a retryable error

Choose this when one target is clearly primary and the others are backups.

### `round_robin`

- advances the starting target for each new request to that virtual model
- fallback still walks forward from that starting point

Choose this when several targets are near-peers and you want simple distribution.

### `weighted`

- uses `weight` only for the first target choice
- fallback then walks forward in declaration order
- missing weights default to `1`

Choose this when you need unequal primary traffic share across targets.

## Retry Behavior

`retries` controls how many extra attempts the proxy makes on the current target before failing over.

`max_fallbacks` controls how many later targets the proxy may attempt after the initial target.

Current rules:

- omitted `retries` means no same-target retry
- omitted `max_fallbacks` means all later targets may be attempted
- `max_fallbacks: 0` disables cross-target failover
- values above later-target count are clamped to the available later targets
- `retry_on_429: true` lets upstream `429` participate in both same-target retry and cross-target failover

The proxy retries only on retryable upstream or transport failures. Upstream `4xx` responses are treated as caller-side problems and do not trigger retry or failover, except optional `429` handling when `retry_on_429` is enabled.

This is an important operational boundary. Routing is not a way to mask bad caller requests or invalid model usage.

## Runtime Filtering

Before dispatch, routing consults direct-model runtime state.

Current filtering order is:

- filter targets currently marked `unhealthy`
- then filter targets currently marked `cooldown`
- if cooldown filtering would empty the candidate set, the proxy falls back to retrying while still excluding `unhealthy` targets

The runtime state itself is exposed on `GET /admin/v1/models/status`.

Source of each state:

- `cooldown` comes from request-path retryable failures on a direct target
- `unhealthy` comes from direct-model `background_model_check`
- routing models themselves are never runtime-filtered and report `not_applicable`

## Design Constraints

- routing targets refer to other model aliases through `targets[].model`
- routing models omit `provider`, `model_name`, and `provider_key_id`
- direct models omit `routing`

## Operator Guidance

- start with direct models first
- add routing only when you have a clear resilience or traffic-shaping goal
- keep target aliases explicit and easy to reason about
- set `retries` and `max_fallbacks` intentionally so resilience does not create surprise cost or latency

## Troubleshooting

### Traffic never reaches the secondary target

That may be expected if the primary target is healthy and your strategy is `failover`.

### A request fails on one target and does not fall back

Check whether the failure is retryable. Upstream `4xx` responses do not trigger cross-target retry.

## Related Pages

- [Models](models.md)
- [Rate Limits](rate-limits.md)
- [Configuration Propagation](configuration-propagation.md)
