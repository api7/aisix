---
title: Routing And Failover
description: Configure virtual models, target selection strategies, and retry behavior in AISIX AI Gateway.
sidebar_position: 35
---

Routing lets one caller-visible model alias dispatch across multiple direct models.

This is the gateway's current virtual-model mechanism.

Use it when you want to separate the caller contract from the individual upstream target that serves a given request.

A routing alias works across the proxy's passthrough endpoints: `/v1/chat/completions` (OpenAI shape), `/v1/messages` (Anthropic shape), `/v1/responses`, and `/v1/messages/count_tokens`. Targets in one group may mix providers — e.g. an OpenAI target and an Anthropic target — and the gateway dispatches each target through the right path regardless of which endpoint the caller used. Streaming requests attempt only the first target (no mid-stream fallback); non-streaming requests fail over across targets.

`/v1/responses` and `/v1/messages/count_tokens` are provider-restricted (OpenAI-only and Anthropic-only respectively). When a group is used on one of these endpoints, only the targets whose provider matches the endpoint are attempted, in order; if the group has no matching target the request is rejected with a 400.

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
    "retry_on_429": true,
    "on_all_filtered": "fail"
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

### All-Targets-Filtered Policy

`routing.on_all_filtered` decides what happens when step 4 of the filter loop is reached — every candidate is excluded by runtime status:

- `fail` (default) — return `503 all_candidates_unavailable` to the caller with `Retry-After: 30`. Use this when serving a known-broken target is worse than failing fast.
- `original_order` — dispatch to the original target list, in declaration order, ignoring runtime state for this request. Use this when availability matters more than honoring the probe verdict.

The `Retry-After` value on the `fail` path is a coarse fixed hint. By the time the filter reaches this branch, every candidate is in background-unhealthy state with no live cooldown timer to read.

## Design Constraints

- routing targets refer to other model aliases through `targets[].model`
- routing models omit `provider`, `model_name`, and `provider_key_id`
- direct models omit `routing`

## Operator Guidance

- start with direct models first
- add routing only when you have a clear resilience or traffic-shaping goal
- keep target aliases explicit and easy to reason about
- set `retries` and `max_fallbacks` intentionally so resilience does not create surprise cost or latency

## Response Shape

Routing keeps the caller's view of the response stable across failover.

### `response.model`

`response.model` always echoes the **model name the caller put on the request** — for a routing model that is the routing alias itself, not the underlying target's display name and not the upstream provider's raw id.

```http
POST /v1/chat/completions
{ "model": "failover-group-XYZ", ... }
```

```json title="Response body"
{
  "id": "chatcmpl-...",
  "model": "failover-group-XYZ",
  ...
}
```

This holds whether the response came from `targets[0]` on the happy path or from a later target after failover. A cross-provider routing group (e.g. mixing an OpenAI target with an Anthropic target) never leaks the underlying provider's vocabulary into `response.model`.

Direct (non-routing) models follow the same contract — `response.model` echoes the caller's requested name.

### `x-aisix-served-by`

The proxy emits an `x-aisix-served-by` response header on every routing-model response. The value is the display name of the target that actually served the request.

```http title="Response headers"
x-aisix-served-by: gpt-4o-secondary
```

After failover, the value reflects the target whose attempt succeeded — not the target that was tried first and failed. The header is the wire-level signal for "did failover fire, and which target won."

The header applies to successful `/v1/chat/completions` responses. It is **absent** in these cases:

- **Direct (non-routing) models.** The body's `response.model` already names the served model, so the header would be redundant — its presence is itself the routing signal.
- **Cache hits.** A stored response is decoupled from whichever target produced it on the original miss; surfacing a stale name would lie. Operators inspecting routing must look at `x-aisix-cache` first.
- **Error responses** (e.g. failover exhausted, every target unhealthy). No target served the request, so there is no name to report.
- **Other endpoints.** The Anthropic-shape `/v1/messages` path resolves routing groups (including failover) but is on a separate code path and does not currently emit this header.

If a routing target's `display_name` contains bytes that are not valid HTTP header values (CR/LF or non-visible-ASCII), the header is omitted and the DP logs a `tracing::warn!` carrying the offending name. Rename the target with operator-side tools to restore the header.

## Troubleshooting

### Traffic never reaches the secondary target

That may be expected if the primary target is healthy and your strategy is `failover`.

### A request fails on one target and does not fall back

Check whether the failure is retryable. Upstream `4xx` responses do not trigger cross-target retry.

### `response.model` shows the routing alias, not the target that served

That is the documented contract — see [Response Shape](#response-shape). Read `x-aisix-served-by` to learn which target actually served the request.

### `x-aisix-served-by` is missing on a routing-model response

Check the response headers first:

- `x-aisix-cache: hit` — header is intentionally absent on cache hits.
- DP logs for a `tracing::warn!` mentioning `target_display_name` — your target's display name contains characters that are not valid in an HTTP header value. Rename the target.

## Related Pages

- [Models](models.md)
- [Rate Limits](rate-limits.md)
- [Configuration Propagation](configuration-propagation.md)
