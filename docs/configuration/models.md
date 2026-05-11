---
title: Models
description: Configure direct models and virtual routing models in AISIX AI Gateway.
sidebar_position: 32
---

Models define what callers can ask the gateway to run.

A model can be one of two shapes:

- a **direct model** that maps one caller-visible alias to one upstream provider model
- a **routing model** that maps one caller-visible alias to a routing strategy over multiple direct models

## Direct Models

Use a direct model when you want one stable gateway alias for one upstream model.

Current required fields are:

- `display_name`
- `provider`
- `model_name`
- `provider_key_id`

Optional fields include:

- `timeout`
- `rate_limit`
- `cost`

Example:

```bash title="Create a direct model"
curl -sS -X POST http://127.0.0.1:3001/admin/v1/models \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "gpt-4o-prod",
    "provider": "openai",
    "model_name": "gpt-4o",
    "provider_key_id": "YOUR_PROVIDER_KEY_ID",
    "timeout": 30000,
    "cost": {
      "input_per_1k": 0.005,
      "output_per_1k": 0.015
    }
  }'
```

## Routing Models

Use a routing model when you want one caller-visible alias to choose among multiple target models.

Current routing strategies are:

- `failover`
- `round_robin`
- `weighted`

For a routing model, `routing` is required and the direct upstream fields must be omitted.

Example:

```bash title="Create a routing model"
curl -sS -X POST http://127.0.0.1:3001/admin/v1/models \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "chat-prod",
    "routing": {
      "strategy": "failover",
      "targets": [
        {"model": "gpt-4o-primary"},
        {"model": "gpt-4o-secondary"}
      ],
      "retry_budget": 2
    }
  }'
```

## Field Notes

- `display_name` is the alias clients send in proxy requests.
- `provider` currently supports `openai`, `anthropic`, `gemini`, and `deepseek`.
- `provider_key_id` must reference an existing `ProviderKey` resource.
- `timeout` is in milliseconds. `0` or omission means no timeout.
- `cost` stores pricing metadata used by budget and usage accounting paths.

## Routing Behavior

Current routing behavior is:

- `failover` always starts with the first target, then walks forward only on retryable failures
- `round_robin` advances the starting target per request for that virtual model
- `weighted` uses target weights only for the first pick, then falls forward in declaration order on retry

`retry_budget` limits how many distinct targets are attempted per request.

- omitted means all configured targets may be attempted
- `1` disables fallback
- `0` is normalized to the full target count

## What `/v1/models` Exposes

Only non-routing models are currently listed on `GET /v1/models`.

Routing aliases are intentionally hidden from that list today, even though callers can still target them directly if they know the alias.

## Operational Notes

- Admin writes become visible to the proxy asynchronously through the watch-driven snapshot path.
- In practice, allow a short propagation delay or poll the target endpoint until the new model resolves.
- Duplicate `display_name` values are rejected with `409`.

## Related Pages

- [Provider Keys](provider-keys.md)
- [API Keys](api-keys.md)
- [Routing And Failover](routing-and-failover.md)
- [Configuration Propagation](configuration-propagation.md)
