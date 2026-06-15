---
title: Rate Limits
description: Configure multi-layer per-key, per-model, and policy-based rate limits in AISIX AI Gateway.
sidebar_position: 36
---

AISIX AI Gateway evaluates every LLM request against multiple rate-limit layers. Each layer is independent — the request must pass **all** of them, otherwise the proxy returns `429`.

Use this page to decide where each limit belongs and what caller-visible behavior to expect when a layer trips.

## Current Rate-Limit Sources

The proxy applies these layers in order, on every LLM endpoint that goes through the shared quota gate:

1. **API-key inline limit** — `ApiKey.rate_limit` on the authenticated key.
2. **Model inline limit** — `Model.rate_limit` on the resolved model.
3. **Rate-limit policy entities** — standalone `RateLimitPolicy` rows that match the current request by scope.

Layers are AND-combined: every layer with a configured limit must have headroom, or the request is rejected before dispatch.

## Inline Rate-Limit Fields

`ApiKey.rate_limit` and `Model.rate_limit` share the same shape:

- `tpm`: tokens per minute
- `tpd`: tokens per day
- `rpm`: requests per minute
- `rpd`: requests per day
- `concurrency`: maximum in-flight requests

All fields are optional. A missing field means no limit on that dimension. An empty `rate_limit` object behaves as no limit.

In practice, most deployments start with:

- `rpm` for request burst control
- `concurrency` for in-flight protection
- `tpm` or `tpd` where usage-based control matters

Example on an API key:

```json title="ApiKey rate limits"
{
  "key_hash": "YOUR_CALLER_KEY_HASH",
  "allowed_models": ["gpt-4o-prod"],
  "rate_limit": {
    "rpm": 60,
    "tpm": 100000,
    "concurrency": 5
  }
}
```

## Rate-Limit Policy Entities

`RateLimitPolicy` is a standalone, scope-targeted rate-limit rule stored in etcd under `rate_limit_policies/<id>`. Use it when the limit you want is not naturally attached to a single API key or model — for example, a per-team request quota or a per-member token quota.

### Policy Fields

- `name`: human label (string, required).
- `scope`: which subject the policy targets — one of `api_key`, `model`, `team`, `member`, `team_member` (required).
- `scope_ref`: the resource ID the policy applies to. Interpretation depends on `scope`:
  - `api_key` → matches when the authenticated `ApiKey` entry id equals `scope_ref`.
  - `model` → matches when the resolved `Model` entry id equals `scope_ref`.
  - `team` → matches when the authenticated `ApiKey.team_id` equals `scope_ref`. **One shared bucket** is pooled across every key in the team.
  - `member` → matches when the authenticated `ApiKey.user_id` equals `scope_ref`.
  - `team_member` → matches when the authenticated `ApiKey.team_id` equals `scope_ref` (like `team`), but the counter is bucketed **per member** (`ApiKey.user_id`). One policy thus gives *every* member of the team their own independent, identical quota — a per-member default. New members inherit it automatically; no per-member policy needed.
- `window`: `second`, `minute`, or `hour` (required).
- `max_requests`: maximum requests allowed in the window (optional).
- `max_tokens`: maximum tokens allowed in the window (optional).

At least one of `max_requests` or `max_tokens` must be set, or the policy is rejected by validation.

### Window Mapping

Policies are normalised to the same internal limit fields used by inline limits:

| `window` | `max_requests` becomes | `max_tokens` becomes |
| --- | --- | --- |
| `second` | `rpm` (× 60) | `tpm` (× 60) |
| `minute` | `rpm` | `tpm` |
| `hour` | `rpd` (× 24) | `tpd` (× 24) |

Out-of-enum window values are rejected by the JSON Schema at etcd load — the row never enters the snapshot and is surfaced through the rejection signal.

### Example Policies

A team-wide token cap of 1M tokens per minute:

```json title="RateLimitPolicy: per-team tokens-per-minute"
{
  "name": "team-acme-tpm",
  "scope": "team",
  "scope_ref": "team-uuid-acme",
  "window": "minute",
  "max_tokens": 1000000
}
```

A per-member burst limit:

```json title="RateLimitPolicy: per-member requests-per-minute"
{
  "name": "member-burst",
  "scope": "member",
  "scope_ref": "member-uuid-1234",
  "window": "minute",
  "max_requests": 60
}
```

A per-member default — every member of a team independently capped at 1M tokens per minute:

```json title="RateLimitPolicy: per-member default for a team"
{
  "name": "team-acme-per-member-tpm",
  "scope": "team_member",
  "scope_ref": "team-uuid-acme",
  "window": "minute",
  "max_tokens": 1000000
}
```

Unlike `scope = team` (one shared bucket for the whole team), `team_member` gives each member their own bucket: member A exhausting the cap never throttles member B, and a member's multiple keys share one bucket (the counter keys on `user_id`).

For `scope = team`, `scope = member`, or `scope = team_member` to match, the authenticated `ApiKey` must carry the corresponding `team_id` / `user_id` field. `team_member` requires **both** `team_id` (to match) and `user_id` (to bucket). Set those on the API key resource at create time.

### Provisioning

`RateLimitPolicy` rows are loaded directly from etcd into the gateway snapshot. The standalone admin API does not currently expose CRUD routes for them — write rows under `<prefix>/rate_limit_policies/<id>` through your control-plane projection or directly via `etcdctl` in self-hosted setups.

The data plane validates each row against the JSON Schema on load: a malformed row is skipped and surfaced through the rejection signal, but does not stop other rows from loading.

## Response Behavior

When any layer rejects the request, the proxy returns `429`. For rate-limit-style rejections that have a retry window, the proxy also emits `Retry-After`.

Successful non-streaming chat responses include `x-ratelimit-*` headers based on the post-dispatch limiter state. Those headers are useful for debugging and for client-side adaptive throttling.

## Counter Storage: Single Node vs Cluster

Every limit above is enforced against a counter. Where that counter lives is set by the `ratelimit` block in the gateway bootstrap config:

```yaml title="ratelimit backend"
ratelimit:
  backend: "memory"   # memory | redis
  # redis:
  #   url: "redis://127.0.0.1:6379"
  #   mode: "single"
  # concurrency_ttl_secs: 300
```

- `memory` (default) — counters live in each gateway process. With a single replica this is exact. With **N replicas behind a load balancer, every limit is effectively multiplied by N**: a key capped at `rpm: 60` gets up to `60 × N` per minute, because each replica counts only the traffic it personally served.
- `redis` — counters are shared across every replica through one Redis, so the whole cluster enforces **one global window** regardless of replica count. Enable this on any multi-replica deployment. The Redis may be the same instance used for the response cache; rate-limit keys are namespaced `aisix:rl:` and hash-tagged per bucket. All dimensions are shared — requests, tokens, and `concurrency` (tracked as a crash-safe distributed semaphore; a slot held by a crashed replica is reclaimed after `concurrency_ttl_secs`, default 300s).

Enable it via config, or via env on a managed/containerized deployment:

```bash
AISIX_RATELIMIT__BACKEND=redis
AISIX_RATELIMIT__REDIS__URL=redis://my-redis:6379
```

If Redis becomes unreachable, the limiter **fails open** to per-replica in-memory counting (logged once) so requests keep flowing; cluster-wide limits are not enforced for the duration of the outage and resume automatically when Redis recovers.

## Operator Guidance

- put caller-facing safety limits on `ApiKey.rate_limit`
- on multi-replica deployments, set `ratelimit.backend: redis` so configured limits are enforced cluster-wide instead of per replica
- use `Model.rate_limit` to protect a specific upstream model alias
- use `RateLimitPolicy` rows when the limit applies to a population that is wider than one key or one model — for example, a whole team
- keep token-based caps proportionate to the burst-control caps; a tight `rpm` with an unlimited `tpm` lets a single long completion still saturate upstream

## Troubleshooting

### A caller sees `429` unexpectedly

Walk the layers in order:

1. inspect the `ApiKey.rate_limit` on the authenticated key
2. inspect the resolved `Model.rate_limit`
3. list the `rate_limit_policies` rows that match the key's `team_id` / `user_id` and the resolved model entry id

Any one of those can be the gating layer.

### A team-scope or member-scope policy is not taking effect

Check the API key. `team` and `member` policies match against `ApiKey.team_id` and `ApiKey.user_id` respectively. If those fields are missing on the key, the policy will never match.

### Limits work for chat but appear silent on other endpoints

The shared quota gate runs across the current LLM endpoint set. If you only see limits triggering on chat, the most likely explanation is that the other endpoint isn't seeing enough traffic to hit the cap, not that the gate is chat-only.

## Related Pages

- [API Keys](api-keys.md)
- [Models](models.md)
- [OpenAI-Compatible API](../integration/openai-compatible-api.md)
- [Headers And Error Codes](../reference/headers-and-error-codes.md)
