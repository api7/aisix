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
- `stream_timeout`
- `rate_limit`
- `cost`
- `cooldown`
- `background_model_check`

Read those optional fields as metadata and policy hints layered onto the basic alias mapping.

`cooldown` and `background_model_check` are the two runtime-status sources that feed [`GET /admin/v1/models/status`](../reference/admin-api-reference.md#runtime-model-status) and the [routing filter](routing-and-failover.md#runtime-filtering). Both are direct-model-only and rejected on routing models.

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
- `ignore_statuses` records the last probe result without marking the model unhealthy. If omitted, **no** statuses are ignored — a 408 or 429 probe response would mark the model unhealthy. For most deployments, `[408, 429]` is a reasonable starting point.
- `stale_after_seconds` is a safety valve for old unhealthy probe state when the checker stops refreshing
- `interval_seconds` has a minimum of `5`; `timeout_seconds`, `max_tokens`, and `stale_after_seconds` have a minimum of `1`

A failed probe transitions the model to `unhealthy` in the runtime status tracker. A subsequent successful probe clears that state. The routing filter excludes `unhealthy` candidates ahead of `cooldown` candidates.

### Timeouts

Two optional per-model knobs bound how long the gateway waits on the upstream. Both are in **milliseconds**. `timeout` of `0` or absent means no non-streaming timeout. `stream_timeout` of a positive value bounds streaming; `0` or absent makes streaming fall back to `timeout` instead (so to disable streaming timeouts entirely, set `timeout` to `0` as well).

```json title="Direct model timeouts"
{
  "timeout": 15000,
  "stream_timeout": 2500
}
```

- `timeout` — end-to-end budget for a **non-streaming** (`stream:false`) upstream call. If the call doesn't finish in time, the gateway abandons it.
- `stream_timeout` — read timeout for a **streaming** (`stream:true`) call, applied to the **first** chunk and to **every inter-chunk gap** (it resets after each chunk). When `stream_timeout` is `0` or absent, the streaming budget falls back to `timeout`.

An elapsed timeout surfaces as a retryable upstream failure (HTTP `504`), so on a [routing model](routing-and-failover.md) it triggers failover to the next target — see [Routing § Timeout-Triggered Failover](routing-and-failover.md#timeout-triggered-failover). On a direct model with no fallback target it simply returns `504` to the caller.

A timeout also feeds [Cooldown](#cooldown) (`trigger_on_timeout`), so a repeatedly-slow target is taken out of rotation for subsequent requests.

> A small `timeout` set for non-streaming traffic also becomes the streaming read budget when `stream_timeout` is unset. Set `stream_timeout` explicitly if streaming needs a different budget.

### Cooldown

`cooldown` is the request-path complement to `background_model_check`. Where the background probe sets `unhealthy` from out-of-band probes, `cooldown` sets a short-lived skip window from the failures observed on real traffic.

```json title="Direct model cooldown"
{
  "cooldown": {
    "enabled": true,
    "default_seconds": 30,
    "max_seconds": 600,
    "honor_retry_after": true,
    "trigger_statuses": [401, 408, 429, 500, 502, 503, 504],
    "trigger_on_timeout": true,
    "trigger_on_transport": true
  }
}
```

All fields are optional. The example shows the *effective* defaults the proxy applies; at the schema level every field is `null` until set, but every accessor falls back to the value shown above. Omitting the `cooldown` block entirely is equivalent to writing the example above verbatim.

Field semantics:

- `enabled` (default `true`) — set to `false` to keep the model in rotation no matter what request-path failures look like.
- `default_seconds` (default `30`) — cooldown TTL when the upstream did not return a `Retry-After` header, or when `honor_retry_after` is `false`. Setting this to `0` disables cooldown for the model (alternative to `enabled: false`).
- `max_seconds` (default `600`) — upper bound on the cooldown TTL. Caps a misbehaving upstream that returns an unreasonable `Retry-After` value.
- `honor_retry_after` (default `true`) — when the upstream OpenAI / Anthropic bridge parses a `Retry-After: <seconds>` header, the cooldown layer uses that value (clamped by `max_seconds`).
- `trigger_statuses` (default `[401, 408, 429, 500, 502, 503, 504]`) — upstream HTTP status codes that put the target into cooldown. The default set covers auth failures, request timeouts, rate limits, and transient server errors. Caller-mistake classes (`400`, `403`, `422`) are intentionally excluded so a single bad request does not cool down a healthy upstream.
- `trigger_on_timeout` (default `true`) — request-path timeouts trigger cooldown.
- `trigger_on_transport` (default `true`) — transport, decode, and stream-abort errors trigger cooldown.

Cooldown triggers independently of whether the failure is retryable. A `429`, for example, cools the model down even when the request itself is not retried.

Once a target enters cooldown, the routing filter prefers other targets within the same routing model. If every candidate is filtered, behavior is governed by [`routing.on_all_filtered`](routing-and-failover.md#all-targets-filtered-policy).

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

- `display_name` is the alias clients send in proxy requests, and the value `response.model` echoes back. It is **not** the upstream model id.
- `model_name` is the upstream model id — the literal string the upstream provider expects in its own `model` field (for example `gpt-4o`, `claude-sonnet-4-5`, an Azure deployment name, or a Bedrock model id). Despite the name, this field holds the upstream id, not a caller alias; the caller alias is `display_name`.
- `provider` is a free-form vendor label, not a closed enum. The value must match the pattern `^[a-z0-9][a-z0-9._-]*$` (lowercase alphanumerics plus `.`, `-`, `_`, and no leading separator) and be at most 64 characters. In AISIX Cloud it is the catalog provider id (for example `openai`, `anthropic`, `deepseek`, `amazon-bedrock`); in the self-hosted gateway it can be any label you choose for a vendor or endpoint (for example `vllm`, `openrouter`, `xai`). Dispatch reads the referenced provider key's `adapter` and `provider`; this field also serves as a metrics and access-log label and gates a few vendor-specific endpoints. See [Adapter protocol families](../reference/adapters.md#how-a-model-resolves-to-a-bridge).
- `provider_key_id` must reference an existing `ProviderKey` resource.
- `timeout` is in milliseconds. `0` or omission means no timeout.
- `cost` stores pricing metadata that AISIX Cloud's cp-api consumes when emitting usage events. The standalone OSS proxy does not consult this field at request time and always emits `cost_usd=0.0`; pricing-aware budget enforcement requires the AISIX Cloud control plane.
- `background_model_check` drives direct-model runtime unhealthy state and the `/admin/v1/models/status` view.
- `cooldown` drives direct-model request-path cooldown and is also surfaced through `/admin/v1/models/status`.

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
- [Adapter protocol families](../reference/adapters.md) — how `provider` and the provider key's `adapter` select an upstream bridge.
- [API Keys](api-keys.md)
- [Routing And Failover](routing-and-failover.md)
- [Configuration Propagation](configuration-propagation.md)
