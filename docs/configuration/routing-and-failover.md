---
title: Routing and Failover
description: Configure virtual models, target selection strategies, and retry behavior in AISIX AI Gateway.
toc_max_heading_level: 2
sidebar_position: 35
---

Routing lets one caller-visible model alias route across multiple direct models. Use it when you want callers to keep one stable model name while the gateway handles failover, simple load distribution, or weighted target selection behind that name.

Routing is the gateway's virtual-model mechanism. Configure direct models first, then add a routing model that points at those direct model aliases.

## Routing Model Fit

Use a routing model when you need one stable caller-facing model name in front of more than one direct model.

| Goal | Recommended strategy | Configuration note |
| --- | --- | --- |
| Keep a primary target with one or more backups | `failover` | Put the preferred target first and keep fallback count explicit. |
| Spread traffic across similar targets | `round_robin` | Each request starts from the next target, but fallback still follows target order. |
| Send unequal traffic shares | `weighted` | Weights affect the first target choice only; fallback still walks forward. |

Do not use routing to hide invalid caller requests. Upstream `4xx` responses are treated as caller-side problems and do not trigger retry or failover, except optional `429` handling when `retry_on_429` is enabled.

## Prerequisites

Create the direct models that can serve traffic. A routing model only references other model aliases through `routing.targets[].model`; it does not carry `provider`, `model_name`, or `provider_key_id` itself.

Keep the target aliases explicit and predictable. Routing is most useful when you have a clear resilience or traffic-shaping goal.

## Configure a Routing Model

Create the routing model, choose the target-selection strategy, and confirm which proxy endpoints can use the routing alias.

### Create a Failover Routing Model

Use a routing model when one caller-facing model alias should fail over across multiple targets:

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

This example makes `gpt-4o-prod` the caller-facing alias. The gateway starts with `gpt-4o-primary`; if that target has a retryable failure, it can retry once on the same target and then fail over once to `gpt-4o-secondary`.

### Choose a Strategy

Each strategy decides the first target for a request. Fallback then walks forward through the target list, bounded by `max_fallbacks`.

#### Failover Strategy

The gateway starts at the first target every time and moves to the next target only when the prior attempt fails with a retryable error. Choose this when one target is clearly primary and the others are backups.

#### Round-Robin Strategy

The gateway advances the starting target for each new request to that virtual model. Fallback still walks forward from that starting point. Choose this when several targets are near-peers and you want simple distribution.

#### Weighted Strategy

The gateway uses `weight` only for the first target choice. Fallback then walks forward in declaration order, and missing weights default to `1`. Choose this when you need unequal primary traffic share across targets.

### Endpoint Support

Routing models apply to model-resolving proxy endpoints. The endpoint still decides which provider families are eligible after the routing alias expands:

| Endpoint | Routing support | Provider support |
| --- | --- | --- |
| `/v1/chat/completions` | Yes | Uses the selected eligible target. |
| `/v1/messages` | Yes | Uses Anthropic-style request handling for eligible targets. |
| `/v1/messages/count_tokens` | Yes | Attempts only Anthropic-backed targets. |
| `/v1/responses` | Yes | Attempts only OpenAI-backed targets. |

Non-streaming requests can fail over across eligible targets. Streaming requests on `/v1/chat/completions`, `/v1/messages`, and `/v1/responses` attempt only the first selected eligible target and do not perform mid-stream fallback.

## Runtime Behavior

Retry limits, target health, and provider-family filtering determine the actual attempt list for each request.

### Set Retry and Fallback Limits

`retries` controls how many extra attempts the proxy makes on the selected target before failing over. `max_fallbacks` controls how many later targets the proxy may attempt after the initial target.

| Field | Default behavior | Use when |
| --- | --- | --- |
| `retries` | No same-target retry when omitted. | A transient failure on the selected target should get another attempt before fallback. |
| `max_fallbacks` | All later targets may be attempted when omitted. | You want to cap how many backup targets a single request can try. |
| `max_fallbacks: 0` | Disables cross-target failover. | You want target selection without fallback. |
| `retry_on_429` | Upstream `429` is not retried when omitted or `false`. | Rate-limit responses should participate in retry and failover. |

Values above the later-target count are clamped to the available later targets.

### Runtime Target Filtering

Before a provider request, routing consults direct-model runtime state and produces the actual attempt list. If at least one target is `healthy`, AISIX uses healthy targets. If no target is `healthy` but at least one target is in `cooldown`, AISIX uses every target whose runtime status is not `unhealthy`; cooldown candidates are preferred over background-confirmed-unhealthy targets. If every target is filtered out, AISIX applies the routing model's [`on_all_filtered`](#all-targets-filtered-policy) policy.

The runtime state itself is exposed on `GET /admin/v1/models/status`. `cooldown` comes from request-path failures on a direct target. See [Models cooldown](models.md#cooldown) for the trigger configuration. `unhealthy` comes from direct-model `background_model_check`. Routing models themselves are marked `not_applicable` because they are never runtime-filtered.

### All-Targets-Filtered Policy

`routing.on_all_filtered` decides what happens when every candidate is excluded by runtime status:

| Policy | Behavior |
| --- | --- |
| `fail` (default) | Return `503 all_candidates_unavailable` to the caller with `Retry-After: 30`. Use this when serving a known-broken target is worse than failing fast. |
| `original_order` | Use the original target list, in declaration order, ignoring runtime state for this request. Use this when availability matters more than honoring the probe verdict. |

The `Retry-After` value on the `fail` path is a coarse fixed hint. By the time the filter reaches this branch, every candidate is in background-unhealthy state with no live cooldown timer to read.

## Verify Routing Behavior

Routing keeps the caller's view of the response stable across failover.

### Response Model Alias

On `POST /v1/chat/completions`, `response.model` echoes the **model name the caller put on the request**. For a routing model, that is the routing alias itself, not the underlying target's display name and not the upstream provider's raw id.

```http
POST /v1/chat/completions
{ "model": "failover-group-XYZ", ... }
```

The response keeps the same caller-facing model alias:

```json
{
  "id": "chatcmpl-...",
  "model": "failover-group-XYZ",
  ...
}
```

This holds whether the response came from `targets[0]` on the happy path or from a later target after failover. A cross-provider routing group never leaks the underlying provider's vocabulary into `response.model`.

Direct models follow the same behavior: `response.model` echoes the caller's requested name.

### Served-By Header

For successful chat-completions routing responses, the proxy emits an `x-aisix-served-by` response header. The value is the display name of the target that served the request.

```http
x-aisix-served-by: gpt-4o-secondary
```

After failover, the value reflects the target whose attempt succeeded, not the target that was tried first and failed. Use the header to confirm whether failover ran and which target served the response.

The header applies to successful `/v1/chat/completions` responses. It is absent for direct non-routing models because the response body already names the served model. It is also absent for cache hits, because a stored response is decoupled from whichever target produced it on the original miss; inspect `x-aisix-cache` first when diagnosing routing behavior.

Error responses do not include the header because no target served the request. Other endpoints, including `/v1/messages`, `/v1/messages/count_tokens`, and `/v1/responses`, can resolve routing aliases, but they do not emit this chat-completions routing header.

If a routing target's `display_name` contains bytes that are not valid HTTP header values (CR/LF or non-visible-ASCII), the header is omitted and the data plane logs a warning carrying the offending name. Rename the target to restore the header.

## Troubleshooting

**Traffic Never Reaches the Secondary Target**

With `strategy: "failover"`, traffic stays on the primary target while that target is healthy. Use a distribution strategy when healthy secondary targets should receive normal traffic.

**A Request Fails on One Target and Does Not Fall Back**

Check whether the failure is retryable. Upstream `4xx` responses do not trigger cross-target retry.

**`response.model` Shows the Routing Alias, Not the Target That Served**

`response.model` preserves the caller-facing routing alias. Read `x-aisix-served-by` to learn which target actually served the request.

**`x-aisix-served-by` Is Missing on a Routing-Model Response**

Check the response headers first:

If `x-aisix-cache: hit` is present, the header is intentionally absent on cache hits. Otherwise, check the data-plane logs for a warning mentioning `target_display_name`; that warning means the target display name contains characters that are not valid in an HTTP header value. Rename the target.

**A Routing Alias Fails on `/v1/messages/count_tokens` or `/v1/responses`**

Check the target providers in the routing group. `/v1/messages/count_tokens` requires at least one Anthropic-backed target, and `/v1/responses` requires at least one OpenAI-backed target. Mixed groups are allowed, but targets from the wrong provider family are skipped for those provider-specific endpoints.

## Related Reading

[Models](models.md) covers direct and routing model aliases. For request, token, and concurrency controls around routed traffic, see [Rate limits](rate-limits.md). For propagation timing, see [Configuration propagation](configuration-propagation.md).
