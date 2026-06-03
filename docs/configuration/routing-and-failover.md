---
title: Routing and Failover
description: Configure virtual models, target selection strategies, and retry behavior in AISIX AI Gateway.
sidebar_position: 35
---

Routing lets one caller-visible model alias dispatch across multiple direct
models. Use it when you want callers to keep one stable model name while the
gateway handles failover, simple load distribution, or weighted target
selection behind that name.

Routing is the gateway's current virtual-model mechanism. Configure direct
models first, then add a routing model that points at those direct model
aliases.

## When to use routing

Use a routing model when you need one stable caller contract in front of more
than one direct model.

| Goal | Recommended strategy | Operator note |
| --- | --- | --- |
| Keep a primary target with one or more backups | `failover` | Put the preferred target first and keep fallback count explicit. |
| Spread traffic across similar targets | `round_robin` | Each request starts from the next target, but fallback still follows target order. |
| Send unequal traffic shares | `weighted` | Weights affect the first target choice only; fallback still walks forward. |

Do not use routing to hide invalid caller requests. Upstream `4xx` responses are
treated as caller-side problems and do not trigger retry or failover, except
optional `429` handling when `retry_on_429` is enabled.

## Prerequisites

Create the direct models that can serve traffic. A routing model only references
other model aliases through `routing.targets[].model`; it does not carry
`provider`, `model_name`, or `provider_key_id` itself.

Keep the target aliases explicit and easy to reason about. Routing is most
useful when you have a clear resilience or traffic-shaping goal.

## Create a failover routing model

```json
{
  "display_name": "gpt-4o-prod",
  "routing": {
    "strategy": "failover",
    "targets": [
      { "model": "gpt-4o-primary" },
      { "model": "gpt-4o-secondary" }
    ],
    "retries": 1,
    "max_fallbacks": 1,
    "retry_on_429": true,
    "on_all_filtered": "fail"
  }
}
```

This example makes `gpt-4o-prod` the caller-facing alias. The gateway starts
with `gpt-4o-primary`; if that target has a retryable failure, it can retry once
on the same target and then fail over once to `gpt-4o-secondary`.

## Choose a strategy

Each strategy decides the first target for a request. Fallback then walks
forward through the target list, bounded by `max_fallbacks`.

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

## Endpoint support

Routing models apply to model-resolving proxy endpoints. The endpoint still
decides which provider families are eligible after the routing alias expands:

| Endpoint | Routing support | Provider boundary |
| --- | --- | --- |
| `/v1/chat/completions` | Yes | Uses the selected eligible target. |
| `/v1/messages` | Yes | Uses Anthropic-style request handling for eligible targets. |
| `/v1/messages/count_tokens` | Yes | Attempts only Anthropic-backed targets. |
| `/v1/responses` | Yes | Attempts only OpenAI-backed targets. |

Non-streaming requests can fail over across eligible targets. Streaming requests
on `/v1/chat/completions`, `/v1/messages`, and `/v1/responses` attempt only the
first selected eligible target and do not perform mid-stream fallback.

## Set retry and fallback boundaries

`retries` controls how many extra attempts the proxy makes on the current target
before failing over. `max_fallbacks` controls how many later targets the proxy
may attempt after the initial target.

| Field | Default behavior | Use when |
| --- | --- | --- |
| `retries` | No same-target retry when omitted. | A transient failure on the current target should get another attempt before fallback. |
| `max_fallbacks` | All later targets may be attempted when omitted. | You want to cap how many backup targets a single request can try. |
| `max_fallbacks: 0` | Disables cross-target failover. | You want target selection without fallback. |
| `retry_on_429` | Upstream `429` is not retried when omitted or `false`. | Rate-limit responses should participate in retry and failover. |

Values above the later-target count are clamped to the available later targets.

## Runtime target filtering

Before dispatch, routing consults direct-model runtime state and produces the actual attempt list in this order:

1. partition targets into `healthy`, `cooldown`, and `unhealthy` based on the runtime status tracker
2. if any healthy targets exist, dispatch to those
3. if no healthy targets exist but at least one target is in `cooldown`, dispatch to every target whose runtime status is not `unhealthy` (cooldown candidates are preferred over background-confirmed-unhealthy ones)
4. if every target is filtered out, apply the routing model's [`on_all_filtered`](#all-targets-filtered-policy) policy

The runtime state itself is exposed on `GET /admin/v1/models/status`.

Source of each state:

- `cooldown` comes from request-path failures on a direct target — see [Models § Cooldown](models.md#cooldown) for the trigger configuration
- `unhealthy` comes from direct-model `background_model_check`
- routing models themselves are never runtime-filtered and report `not_applicable`

### All-targets-filtered policy

`routing.on_all_filtered` decides what happens when step 4 of the filter loop is reached — every candidate is excluded by runtime status:

- `fail` (default) — return `503 all_candidates_unavailable` to the caller with `Retry-After: 30`. Use this when serving a known-broken target is worse than failing fast.
- `original_order` — dispatch to the original target list, in declaration order, ignoring runtime state for this request. Use this when availability matters more than honoring the probe verdict.

The `Retry-After` value on the `fail` path is a coarse fixed hint. By the time the filter reaches this branch, every candidate is in background-unhealthy state with no live cooldown timer to read.

## Verify the response shape

Routing keeps the caller's view of the response stable across failover.

### `response.model`

On `POST /v1/chat/completions`, `response.model` echoes the **model name the caller put on the request** — for a routing model that is the routing alias itself, not the underlying target's display name and not the upstream provider's raw id.

```http
POST /v1/chat/completions
{ "model": "failover-group-XYZ", ... }
```

```json
{
  "id": "chatcmpl-...",
  "model": "failover-group-XYZ",
  ...
}
```

This holds whether the response came from `targets[0]` on the happy path or from a later target after failover. A cross-provider routing group (e.g. mixing an OpenAI target with an Anthropic target) never leaks the underlying provider's vocabulary into `response.model`.

Direct (non-routing) models follow the same contract — `response.model` echoes the caller's requested name.

### `x-aisix-served-by`

For successful chat-completions routing responses, the proxy emits an `x-aisix-served-by` response header. The value is the display name of the target that actually served the request.

```http
x-aisix-served-by: gpt-4o-secondary
```

After failover, the value reflects the target whose attempt succeeded — not the target that was tried first and failed. The header is the wire-level signal for "did failover fire, and which target won."

The header applies to successful `/v1/chat/completions` responses. It is **absent** in these cases:

- **Direct (non-routing) models.** The body's `response.model` already names the served model, so the header would be redundant — its presence is itself the routing signal.
- **Cache hits.** A stored response is decoupled from whichever target produced it on the original miss; surfacing a stale name would lie. Operators inspecting routing must look at `x-aisix-cache` first.
- **Error responses** (e.g. failover exhausted, every target unhealthy). No target served the request, so there is no name to report.
- **Other endpoints.** `/v1/messages`, `/v1/messages/count_tokens`, and `/v1/responses` can resolve routing aliases, but they do not emit this chat-completions routing header today.

If a routing target's `display_name` contains bytes that are not valid HTTP header values (CR/LF or non-visible-ASCII), the header is omitted and the data plane logs a warning carrying the offending name. Rename the target with operator-side tools to restore the header.

## Troubleshooting

### Traffic never reaches the secondary target

That may be expected if the primary target is healthy and your strategy is `failover`.

### A request fails on one target and does not fall back

Check whether the failure is retryable. Upstream `4xx` responses do not trigger cross-target retry.

### `response.model` shows the routing alias, not the target that served

That is the documented contract — see [Verify the response shape](#verify-the-response-shape). Read `x-aisix-served-by` to learn which target actually served the request.

### `x-aisix-served-by` is missing on a routing-model response

Check the response headers first:

- `x-aisix-cache: hit` — header is intentionally absent on cache hits.
- Data-plane logs for a warning mentioning `target_display_name` — your target's display name contains characters that are not valid in an HTTP header value. Rename the target.

### A routing alias fails on `/v1/messages/count_tokens` or `/v1/responses`

Check the target providers in the routing group. `/v1/messages/count_tokens`
requires at least one Anthropic-backed target, and `/v1/responses` requires at
least one OpenAI-backed target. Mixed groups are allowed, but targets from the
wrong provider family are skipped for those provider-specific endpoints.

## Next steps

- [Models](models.md)
- [Rate limits](rate-limits.md)
- [Configuration propagation](configuration-propagation.md)
