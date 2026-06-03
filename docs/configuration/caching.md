---
title: Caching
description: Configure cache policies, TTL scope matching, and current cache-backend behavior in AISIX AI Gateway.
sidebar_position: 39
---

Caching has two layers:

The process cache backend decides whether the data plane has a cache available.
The `CachePolicy` resources decide which requests are allowed to use that
cache.

Current runtime caching is exact-match response caching for non-streaming
chat-completions requests. Streaming responses are not cached.

## Caching at a glance

| Layer | Configured by | What it controls |
| --- | --- | --- |
| Process cache backend | Bootstrap configuration | Whether the data plane uses in-memory cache or Redis. |
| Cache policy | Dynamic `CachePolicy` resource | Which non-streaming chat-completions requests may use the configured backend. |

Both layers must line up before a response can be cached. A Redis value on a
policy does not move that policy to Redis; the process backend is selected at
startup.

## Configure the process backend

The server selects one cache backend at startup.

Memory cache is the default in-process backend. It is useful for a single data
plane instance or local testing.

Redis can be configured through bootstrap config when multiple data-plane
instances should share cached responses. The current Redis path uses a
single-node connection. Cluster and sentinel modes are not exposed through
bootstrap config today.

```yaml title="config.yaml"
cache:
  backend: redis
  redis:
    url: redis://127.0.0.1:6379/
```

See [Bootstrap configuration](bootstrap-config.md) for process configuration.

## Create a cache policy

A cache policy opens the cache gate for matching requests.

```shell
curl -sS -X POST http://127.0.0.1:3001/admin/v1/cache_policies \
  -H "Authorization: Bearer YOUR_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "default-chat-cache",
    "enabled": true,
    "ttl_seconds": 3600,
    "applies_to": "model:gpt-4o-prod"
  }'
```

This policy does not choose the process backend. It only says that matching
requests may use whichever cache backend the process was started with.

## Scope matching

`applies_to` controls which requests match the policy.

Use `all` to match every non-streaming chat-completions request:

```json
{
  "applies_to": "all"
}
```

Use `model:<display_name>` to match the caller-visible model alias:

```json
{
  "applies_to": "model:gpt-4o-prod"
}
```

Use `api_key:<api_key_id>` to match the authenticated API-key resource id:

```json
{
  "applies_to": "api_key:550e8400-e29b-41d4-a716-446655440000"
}
```

The runtime matcher compares against the request's model alias and the
authenticated API-key id. For routing models, the cache key uses the virtual
model alias the caller requested, not the direct target that served the miss.

Avoid undocumented matcher prefixes. The data plane currently treats unknown
forms as `all`, so a typo can make a policy broader than intended.

## Runtime behavior

For each non-streaming chat-completions request, the proxy finds the first
enabled cache policy whose `applies_to` matcher accepts the request.

If a policy matches, the proxy checks the cache. A miss is written back with the
policy's `ttl_seconds`. If no enabled policy matches, the cache path stays
closed for that request.

When the request participates in caching, the proxy can emit:

```text
x-aisix-cache: miss
x-aisix-cache: hit
```

If no policy matches, the response should not be treated as a cache hit or
miss.

## Backend field on a policy

The `CachePolicy` schema includes `backend` with `memory` and `redis` values.

Treat this as a persisted hint, not as a per-policy backend selector. Runtime
traffic uses the backend selected by bootstrap config for the whole process.
Changing `backend` on an individual policy does not move that policy to a
different cache backend.

## Operator guidance

Start with a narrow policy, such as `model:<alias>` or `api_key:<id>`.

Use `all` only when every non-streaming chat-completions request in the
environment should participate in caching.

Use Redis at bootstrap time when several data-plane instances should share
cached responses.

Disable a policy with `enabled: false` when you want to stage or temporarily
turn off caching without deleting the policy.

## Troubleshooting

### Responses never show `x-aisix-cache`

Check all three gates:

- the process must have a cache backend
- an enabled cache policy must match the request
- the request must be a non-streaming chat-completions request

### A policy matches too broadly

Check `applies_to`. Unknown matcher prefixes currently fall back to `all`, so
stick to `all`, `model:<display_name>`, or `api_key:<api_key_id>`.

### Redis is configured on the policy but traffic still uses memory

Set Redis in bootstrap config. `CachePolicy.backend` is not a runtime selector
for individual policies.

## Next steps

- [Bootstrap configuration](bootstrap-config.md) configures process-level cache backend settings.
- [Configuration overview](overview.md) explains the split between bootstrap
  settings and dynamic resources.
- [Admin API](admin-api.md) explains standalone admin writes.
- [Rate limits](rate-limits.md) covers another request-control policy layer.
