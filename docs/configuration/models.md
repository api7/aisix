---
title: Models
description: Configure direct models and virtual routing models in AISIX AI Gateway.
sidebar_position: 33
---

Models define the names that callers send to the gateway.

A model is either a direct upstream target or a virtual routing alias. Direct
models hold provider wiring. Routing models point to other models and let the
gateway choose a target per request.

Use direct models first. Add routing models when you need failover, round-robin,
or weighted target selection behind one stable caller-facing name.

## Prerequisites

This page assumes you have:

- a running self-hosted gateway with the admin listener available
- an admin key for `Authorization: Bearer YOUR_ADMIN_KEY`
- a provider key id to use as `provider_key_id`

If you do not have a provider key yet, start with
[Provider keys](provider-keys.md), then return to this page.

## Direct model

A direct model maps one gateway alias to one upstream model.

```shell
curl -sS -X POST http://127.0.0.1:3001/admin/v1/models \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    # highlight-start
    "display_name": "gpt-4o-prod",
    "provider": "openai",
    "model_name": "gpt-4o",
    "provider_key_id": "YOUR_PROVIDER_KEY_ID",
    # highlight-end
    "timeout": 30000
  }'
```

For a direct model, the gateway expects `display_name`, `provider`,
`model_name`, and `provider_key_id`.

| Field | Description |
| --- | --- |
| `display_name` | Caller-facing alias. Callers send this value as `model`, and the gateway echoes it as `response.model`. |
| `provider` | Vendor label used for metrics, access logs, and endpoint-specific vendor gates. |
| `model_name` | Upstream model id sent to the provider, such as `gpt-4o`, an Azure deployment name, or a Bedrock model id. |
| `provider_key_id` | Provider key id that supplies the upstream credential, optional `api_base`, provider identity, and adapter family. |

`provider` is an open label, not a closed enum. It must be lowercase, start
with a letter or number, and use only letters, numbers, `.`, `_`, or `-`.
The generated schema caps it at 64 characters.

## Routing model

A routing model is a virtual alias. It has a `routing` block instead of direct
upstream fields.

```shell
curl -sS -X POST http://127.0.0.1:3001/admin/v1/models \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "chat-prod",
    # highlight-start
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
    # highlight-end
  }'
```

Each `routing.targets[*].model` references another model's `display_name`. The
targets should be direct models.

| Strategy | Behavior |
| --- | --- |
| `failover` | Start with the first target, then walk forward on retryable failures. |
| `round_robin` | Rotate the starting target per request for this routing alias. |
| `weighted` | Choose the first target by weight, then fall forward in declaration order on retry. |

`retries` controls how many extra attempts stay on the current target before
failover. `max_fallbacks` controls how many later targets may be attempted.
When omitted, `retries` defaults to `0` and `max_fallbacks` allows all later
targets. Set `max_fallbacks: 0` to disable fallback.

By default, upstream `429` responses are not retried. Set `retry_on_429: true`
when rate-limit responses should participate in retry and failover.

## Direct and routing shapes

Do not mix direct-model fields and routing fields in the same model.

Use this shape for direct upstream targets:

```json
{
  "display_name": "gpt-4o-prod",
  "provider": "openai",
  "model_name": "gpt-4o",
  "provider_key_id": "provider-key-id"
}
```

Use this shape for virtual routing aliases:

```json
{
  "display_name": "chat-prod",
  "routing": {
    "targets": [
      {"model": "gpt-4o-primary"},
      {"model": "gpt-4o-secondary"}
    ]
  }
}
```

The generated JSON Schema and the admin OpenAPI document are the source of
truth for the accepted request and response shape.

## Timeout

`timeout` is measured in milliseconds. Omit it or set it to `0` for no
per-request timeout at the model layer.

Timeouts are direct-model behavior. A routing model dispatches through the
selected target model, so configure timeouts on the direct targets.

## Background model checks

`background_model_check` probes a direct model outside the request path and
marks the target `unhealthy` when probes fail.

```json
{
  # highlight-start
  "background_model_check": {
    "enabled": true,
    "interval_seconds": 30,
    "timeout_seconds": 10,
    "prompt": "Respond with OK",
    "max_tokens": 8,
    "ignore_statuses": [408, 429],
    "stale_after_seconds": 90
  }
  # highlight-end
}
```

Only direct models may use `background_model_check`. Routing models reject it.

| Field | Description |
| --- | --- |
| `interval_seconds` | How often AISIX probes the model. Minimum: `5`. |
| `timeout_seconds` | Probe timeout. Minimum: `1`. |
| `prompt` | Prompt sent by the probe. |
| `max_tokens` | Maximum tokens for the probe response. Minimum: `1`. |
| `ignore_statuses` | Upstream statuses that do not mark the model unhealthy. |
| `stale_after_seconds` | How long a probe result remains fresh. Minimum: `1`. |

If `ignore_statuses` is omitted, no statuses are ignored. `[408, 429]` is a
common starting point when transient timeouts and rate limits should remain
visible without immediately marking the model unhealthy.

Runtime model status is exposed by `GET /admin/v1/models/status`. The generated
admin API reference describes the route shape.

## Cooldown

`cooldown` is the request-path complement to background checks. It temporarily
excludes a direct model after failures observed on real traffic.

```json
{
  # highlight-start
  "cooldown": {
    "enabled": true,
    "default_seconds": 30,
    "max_seconds": 600,
    "honor_retry_after": true,
    "trigger_statuses": [401, 408, 429, 500, 502, 503, 504],
    "trigger_on_timeout": true,
    "trigger_on_transport": true
  }
  # highlight-end
}
```

All fields are optional. Omitting the `cooldown` block uses the effective
defaults shown above.

| Field | Default |
| --- | --- |
| `enabled` | `true` |
| `default_seconds` | `30` |
| `max_seconds` | `600` |
| `honor_retry_after` | `true` |
| `trigger_statuses` | `[401, 408, 429, 500, 502, 503, 504]` |
| `trigger_on_timeout` | `true` |
| `trigger_on_transport` | `true` |

Cooldown is independent of retry. For example, an upstream `429` can put a
model into cooldown even when the current request is not retried.

When a target enters cooldown, routing models prefer other available targets.
If every candidate is filtered, behavior is controlled by
[`routing.on_all_filtered`](routing-and-failover.md#all-targets-filtered-policy).

## Cost metadata

`cost` stores pricing metadata for usage and budget workflows.

The standalone proxy does not price requests at dispatch time and emits
`cost_usd=0.0`. Pricing-aware budget enforcement requires the AISIX Cloud
control plane.

## What `/v1/models` exposes

`GET /v1/models` currently lists non-routing models.

Routing aliases are intentionally hidden from this discovery response today,
even though callers can target them directly on `/v1/chat/completions` if they
know the alias.

## Verify the model

After creating a direct model, check that the admin API returns it:

```shell
curl -sS http://127.0.0.1:3001/admin/v1/models \
  -H "Authorization: Bearer YOUR_ADMIN_KEY"
```

Then check that the proxy snapshot has seen the model by listing models with a
caller key that is allowed to use it:

```shell
curl -sS http://127.0.0.1:3000/v1/models \
  -H "Authorization: Bearer YOUR_CALLER_KEY"
```

If the admin API returns the model but the proxy does not, wait briefly and
retry. Admin writes become visible through the watch-driven snapshot path.

## Operational notes

Admin writes become visible to the proxy asynchronously through the watch-driven
snapshot path. After creating or updating a model, poll `/v1/models` with the
caller key, or poll the target proxy endpoint, until the model resolves.

Duplicate `display_name` values are rejected with `409`.

Runtime routing exclusion is exposed by `GET /admin/v1/models/status`, not by
`GET /admin/v1/health`.

## Troubleshooting

### Callers get `404` after a model is created

Most often, the new model has not propagated into the current proxy snapshot
yet. Wait briefly and retry, or check [Configuration propagation](configuration-propagation.md).

### A direct model exists but dispatch fails

Check the referenced `provider_key_id`, the provider key's `api_base`, and the
relationship between `display_name`, `model_name`, `provider`, and `adapter`.

### A routing alias works but does not appear in `/v1/models`

That is expected with the current discovery boundary. `/v1/models` is not a
complete list of every valid caller target.

## Next steps

- [Provider keys](provider-keys.md) explains upstream credentials and base URLs.
- [API keys](api-keys.md) lets callers use model aliases.
- [Routing and failover](routing-and-failover.md) covers virtual model behavior.
- [Configuration propagation](configuration-propagation.md) explains when admin writes become visible to the proxy.
- [Adapter protocol families](../reference/adapters.md) explains how provider keys select upstream bridges.
