---
title: Caching
description: Configure cache policies, TTL scope matching, and current cache-backend behavior in AISIX AI Gateway.
sidebar_position: 39
---

Caching is controlled by dynamic `CachePolicy` resources plus the bootstrap cache backend selection.

Use this page to answer two separate questions:

- is a cache backend available in the process
- which requests are allowed to use it

## Current Fields

- `name`
- `enabled`
- `backend`
- `ttl_seconds`
- `applies_to`

Example:

```bash title="Create a cache policy"
curl -sS -X POST http://127.0.0.1:3001/admin/v1/cache_policies \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "default-chat-cache",
    "backend": "memory",
    "ttl_seconds": 3600,
    "applies_to": "all"
  }'
```

This example only defines the policy. The process also needs a compatible bootstrap cache backend.

## Scope Matching

`applies_to` currently supports:

- `all`
- `model:<display_name>`
- `api_key:<api_key_id>`

Current matching is done against:

- the caller-visible model alias in the request
- the authenticated API key resource `id`

Unknown `applies_to` prefixes currently fall back to `all` on the data-plane side, so operators should rely on the documented forms only.

That means undocumented matcher prefixes are unsafe from an operator predictability standpoint.

## Runtime Behavior

Current cache gating behavior is:

- the proxy selects the first enabled policy whose `applies_to` matcher accepts the request
- the selected policy's `ttl_seconds` is used for the cache write
- if no policy matches, the cache gate stays closed for that request

On chat responses, the proxy can emit `x-aisix-cache` with:

- `hit`
- `miss`

Those headers are the easiest caller-visible sign that the request participated in the cache path.

If no enabled policy matches the request, the response should not be treated as a cache hit or miss path.

## Backend Boundary

Current schema supports:

- `memory`
- `redis`

The proxy selects the cache instance per request from the matched policy's `backend` field:

- `memory` uses the in-process cache — always available
- `redis` uses the shared Redis cache — available only when the bootstrap config carries a `cache.redis` block

A `redis` policy on a process without `cache.redis` gets no caching for its requests: responses carry no `x-aisix-cache` header, telemetry reports `cache_status = disabled`, and the gateway logs a warning once per policy. There is no silent fallback to the in-process memory cache — a memory stand-in would serve per-node answers while the policy claims shared-cache semantics.

### `cache.redis` connection modes

The `cache.redis` block accepts the same `mode` field as the rate-limiter — `single` (default), `cluster`, or `sentinel` — so the shared cache can target a standalone Redis, a Redis Cluster, or a Sentinel-managed master. The field shapes are identical to the rate-limit backend; see [Redis connection modes](rate-limits.md#redis-connection-modes).

```yaml title="cache.redis examples"
cache:
  backend: "redis"
  redis:
    mode: "single"                       # one endpoint
    url: "redis://127.0.0.1:6379"
    # mode: "cluster"                     # seed nodes:
    # nodes: ["redis://10.0.0.1:6379", "redis://10.0.0.2:6379"]
    # mode: "sentinel"                    # sentinels + master group:
    # sentinels: ["redis://10.0.0.1:26379"]
    # master_name: "mymaster"
    # username: "default"                 # ACL auth for the data node
    # password: "s3cret"                  # (or AISIX_CACHE__REDIS__PASSWORD)
    # database: 0
```

Cache reads/writes are single-key (`GET`/`SET`), so Redis Cluster routes them automatically; in sentinel mode the master is resolved through the sentinels and re-resolved after a failover. ACL `username`/`password` and the sentinel-vs-master credential split work exactly as for the rate-limiter — see [Sentinel vs master credentials](rate-limits.md#redis-connection-modes).

## Operator Guidance

- start with `memory` plus a narrowly scoped policy
- use `all` only when you truly want broad cache participation
- prefer `model:<alias>` or `api_key:<id>` when you need targeted rollout

## Troubleshooting

### Responses never show `x-aisix-cache`

Check all three:

- an enabled cache policy must match the request
- the matched policy's `backend` must be available in the process — `redis` requires `cache.redis` in the bootstrap config
- look for the `cache policy requests backend=redis but this DP has no redis cache configured` warning in the gateway log

### A policy matches too broadly

Revisit `applies_to` and avoid undocumented matcher forms.

## Related Pages

- [Bootstrap Configuration](bootstrap-config.md)
- [Admin API](admin-api.md)
- [Roadmap](../roadmap.md)
