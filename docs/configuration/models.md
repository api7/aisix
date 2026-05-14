---
title: Models
description: Configure direct models and virtual routing models in AISIX AI Gateway.
sidebar_position: 32
---

Models define what callers can ask the gateway to run.

This is the most important dynamic resource in the system because it defines the caller-visible contract.

A model can be one of two shapes:

- a **direct model** that maps one caller-visible alias to one upstream provider model
- a **routing model** that maps one caller-visible alias to a routing strategy over multiple direct models

## Direct Models

Use a direct model when you want one stable gateway alias for one upstream model.

This is the right default for most first deployments.

Current required fields are:

- `display_name`
- `provider`
- `model_name`
- `provider_key_id`

Optional fields include:

- `timeout`
- `rate_limit`
- `cost`
- `background_model_check`

Read those optional fields as metadata and policy hints layered onto the basic alias mapping.

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

Optional direct-model background probing:

```json title="Direct model background_model_check"
{
  "background_model_check": {
    "enabled": true,
    "interval_seconds": 30,
    "timeout_seconds": 10,
    "prompt": "Respond with OK",
    "max_tokens": 8,
    "ignore_statuses": [408, 429],
    "stale_after_seconds": 90
  }
}
```

Current semantics:

- only direct models may carry `background_model_check`
- routing models reject `background_model_check`
- `ignore_statuses` records the last probe result without marking the model unhealthy
- `stale_after_seconds` is a safety valve for old unhealthy probe state when the checker stops refreshing

## Routing Models

Use a routing model when you want one caller-visible alias to choose among multiple target models.

Routing models are virtual aliases. They do not directly hold upstream credential wiring the way direct models do.

Current routing strategies are:

- `failover`
- `round_robin`
- `weighted`

For a routing model, `routing` is required and the direct upstream fields must be omitted.

That is the easiest operator check for whether a model row is direct or virtual.

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
      "retries": 1,
      "max_fallbacks": 1,
      "retry_on_429": true
    }
  }'
```

## Field Notes

- `display_name` is the alias clients send in proxy requests.
- `provider` currently supports `openai`, `anthropic`, `gemini`, and `deepseek`.
- `provider_key_id` must reference an existing `ProviderKey` resource.
- `timeout` is in milliseconds. `0` or omission means no timeout.
- `cost` stores pricing metadata used by budget and usage accounting paths.
- `background_model_check` drives direct-model runtime unhealthy state and the `/admin/v1/models/status` view.

Practical guidance:

- choose `display_name` as the public contract you want client teams to depend on
- avoid leaking raw upstream naming into aliases unless that is intentional
- use direct models first, then layer routing models on top once you know the target set you want to orchestrate

## Routing Behavior

Current routing behavior is:

- `failover` always starts with the first target, then walks forward only on retryable failures
- `round_robin` advances the starting target per request for that virtual model
- `weighted` uses target weights only for the first pick, then falls forward in declaration order on retry

`retries` limits how many extra attempts stay on the current target before failover.

`max_fallbacks` limits how many later targets are attempted per request.

- omitted `retries` means no same-target retry
- omitted `max_fallbacks` means all later targets may be attempted
- `max_fallbacks: 0` disables fallback
- `retry_on_429: true` lets upstream `429` participate in retry and failover

These fields are the main operator knobs for balancing resilience versus extra upstream cost and latency.

## What `/v1/models` Exposes

Only non-routing models are currently listed on `GET /v1/models`.

Routing aliases are intentionally hidden from that list today, even though callers can still target them directly if they know the alias.

That means `/v1/models` is not currently a full discovery surface for every valid caller target.

## Operational Notes

- Admin writes become visible to the proxy asynchronously through the watch-driven snapshot path.
- In practice, allow a short propagation delay or poll the target endpoint until the new model resolves.
- Duplicate `display_name` values are rejected with `409`.
- Runtime routing exclusion is exposed on `GET /admin/v1/models/status`, not on `GET /admin/v1/health`.

## Troubleshooting

### A model was created but callers get `404`

Most often, the new model has not propagated into the current proxy snapshot yet.

### A direct model exists but dispatch still fails

Check `provider_key_id`, upstream `api_base`, and provider/model-name alignment.

### A routing alias works even though it is not listed in `/v1/models`

That is expected with the current discovery boundary.

## Related Pages

- [Provider Keys](provider-keys.md)
- [API Keys](api-keys.md)
- [Routing And Failover](routing-and-failover.md)
- [Configuration Propagation](configuration-propagation.md)
