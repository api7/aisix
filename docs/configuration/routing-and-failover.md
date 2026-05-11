---
title: Routing And Failover
description: Configure virtual models, target selection strategies, and retry behavior in AISIX AI Gateway.
sidebar_position: 35
---

Routing lets one caller-visible model alias dispatch across multiple direct models.

This is the gateway's current virtual-model mechanism.

## Current Strategies

- `failover`
- `round_robin`
- `weighted`

## Example: Failover Routing

```json title="Routing block"
{
  "routing": {
    "strategy": "failover",
    "targets": [
      { "model": "gpt-4o-primary" },
      { "model": "gpt-4o-secondary" }
    ],
    "retry_budget": 2
  }
}
```

## Strategy Semantics

### `failover`

- starts at the first target every time
- only moves to the next target when the prior attempt fails with a retryable error

### `round_robin`

- advances the starting target for each new request to that virtual model
- fallback still walks forward from that starting point

### `weighted`

- uses `weight` only for the first target choice
- fallback then walks forward in declaration order
- missing weights default to `1`

## Retry Behavior

`retry_budget` limits how many distinct targets the proxy will attempt for one request.

Current rules:

- omitted means use `targets.len()`
- `0` is treated as full target count
- `1` means no fallback beyond the first target
- values above target count are clamped to target count

The proxy retries only on retryable upstream or transport failures. Upstream `4xx` responses are treated as caller-side problems and do not trigger cross-target retry.

## Design Constraints

- routing targets refer to other model aliases through `targets[].model`
- routing models omit `provider`, `model_name`, and `provider_key_id`
- direct models omit `routing`

## Related Pages

- [Models](models.md)
- [Rate Limits](rate-limits.md)
- [Configuration Propagation](configuration-propagation.md)
